"""SE-floor geometry: the corpus's per-attempt information as a function of rank.

Single source of truth for two steps that must agree by construction:

  pipeline.fit_population_constants  CHOOSES the rank grid (rank_prior)
  pipeline.package_for_pwa           EMITS the floor curve on that same grid

They agree because the grid is chosen as the span this module reports measurable,
under the same kernel and the same threshold the emitter then checks. Splitting
these was the defect: the grid was a hardcoded envelope, so a refit that moved the
corpus's sparse tail could put a knot where no measurement exists, and the emitter
could only abort.

The client mirrors these transforms — pwa/shared/rank-window.js slopeInRankSpace
and pwa/js/irt.js anchorInfo — so the shipped curve is in the units the reader
accumulates. Keep them in step.
"""

import numpy as np

from utils.rating.ratings_io import interp_extrap_vec as interp_extrap, load_size_value_maps

# Gaussian kernel half-width in stones, over problem rank. Measured: against a
# 500-problem local mean it is the point where the estimate stops moving with the
# window (a fixed 50-nearest window moves ~5x more per quarter-stone) without yet
# flattening the one sharp feature in the curve, the fall from the kyu plateau into
# the dan range. Wider trades that feature away; see CLAUDE.md for the sweep.
SE_FLOOR_BANDWIDTH = 0.25

# Below this effective sample a knot is a handful of problems and it is not
# measurable. It is a tripwire, NOT a bandwidth trigger: widening the kernel where
# the corpus thins is the obvious repair and it is wrong here, because the nearest
# dense region is the slope-clamp ridge just below 20k, so a widened tail knot
# imports that ridge and lands 5x off its own neighbours. A sparse knot's honest
# answer is its own few problems: noisy, but LOCAL and static, and static noise
# moves nobody's rank around. The grid is bounded to where this holds instead.
SE_FLOOR_MIN_EFFECTIVE = 5

# How many CONSECUTIVE interior knots may be bridged by neighbour interpolation.
# At the 0.5-stone grid this spans one stone, four kernel half-widths — short
# enough that the two ends still describe the same stretch of the curve. Longer
# than this is a real gap in the corpus rather than a pinhole in it, and the run
# aborts instead of inventing a shape across it.
SE_FLOOR_MAX_INTERP_RUN = 2

_CALIBRATION_P = 0.8
_FLOOR_MAX = 0.78
_SLOPE_LO, _SLOPE_HI, _MSLOPE_FLOOR = 0.05, 3.0, 1e-3


def rank_space_slope(rating, a_p, cr, ck):
    """Raw 2PL discrimination → logit-per-internal-rank, via M's local steepness."""
    h = 1e-3
    m_slope = (interp_extrap(rating + h, cr, ck)
               - interp_extrap(rating - h, cr, ck)) / (2 * h)
    return np.clip(a_p / np.maximum(np.abs(m_slope), _MSLOPE_FLOOR),
                   _SLOPE_LO, _SLOPE_HI)


def anchor_info(floor_raw, lapse):
    """Fisher information one AT-rank attempt carries, per unit squared slope."""
    c = np.clip(np.asarray(floor_raw, dtype=float), 0.0, _FLOOR_MAX)
    span = np.maximum(0.0, 1.0 - c - lapse)
    q = np.clip((_CALIBRATION_P - c) / (1.0 - c - lapse), 1e-6, 1 - 1e-6)
    g = span * q * (1 - q)
    return g * g / (_CALIBRATION_P * (1 - _CALIBRATION_P))


def kernel_stats(rank, ranks, infos, *, absorb_below=False, absorb_above=False):
    """(mean info, effective sample count) at `rank` under the fixed-width kernel.

    An OUTERMOST knot absorbs everything past it at full weight instead of letting
    the Gaussian decay away. That is the semantics the reader already has —
    seFloorAt clamps to the range, so a user beyond the edge is served the edge
    knot — and it is what the prior does with its own outside mass, folding it in
    by normalisation. It also makes an edge knot measurable regardless of how thin
    the corpus gets out there, which is what lets the grid END at a sparse rank
    rather than needing dense support past where anybody plays.

    Absorbing LOWERS the edge's information (it averages in the far tail, whose
    problems carry less), so the floor there rises. That is the safe direction: a
    falsely tight se is what every display and serve offset over-reads.
    """
    w = np.exp(-0.5 * ((ranks - rank) / SE_FLOOR_BANDWIDTH) ** 2)
    if absorb_below:
        w = np.where(ranks <= rank, 1.0, w)
    if absorb_above:
        w = np.where(ranks >= rank, 1.0, w)
    sw = w.sum()
    if not (sw > 0):
        return float("nan"), 0.0
    return float((w * infos).sum() / sw), float(sw * sw / (w * w).sum())


def corpus_rank_information(ratings_dir, curves, lapse, sizes):
    """(ranks, infos) over the WHOLE rated corpus, sorted by rank.

    Reads the backend ratings/<size>/ CSVs directly — the same source
    package_for_pwa ships — so neither caller depends on pwa/tsumegos/. A size
    with no curve or no ratings raises: leaving it out would price the floor on
    part of the corpus while every size is served.
    """
    ranks, infos = [], []
    for size in sizes:
        curve = curves.get(str(size))
        if curve is None:
            raise KeyError(f"no rating→rank curve for size {size}")
        cr = np.asarray(curve["ratings"], dtype=float)
        ck = np.asarray(curve["ranks"], dtype=float)
        rmap, smap, fmap = load_size_value_maps(ratings_dir, size)
        tids = list(rmap)
        if not tids:
            raise ValueError(f"ratings/{size}: no rated problems")
        rating = np.array([rmap[t] for t in tids], dtype=float)
        slope = np.array([smap[t] for t in tids], dtype=float)
        floor = np.array([fmap[t] for t in tids], dtype=float)
        ranks.append(interp_extrap(rating, cr, ck))
        infos.append(rank_space_slope(rating, slope, cr, ck) ** 2
                     * anchor_info(floor, lapse))
    r = np.concatenate(ranks)
    inf = np.concatenate(infos)
    order = np.argsort(r)
    return r[order], inf[order]


def grid_from_bounds(lo_target, hi_target, step):
    """The step-aligned knots inside [lo_target, hi_target] (snapped INWARD, so the
    grid never claims support the targets did not ask for)."""
    lo = np.ceil(lo_target / step) * step
    hi = np.floor(hi_target / step) * step
    return np.arange(lo, hi + 1e-9, step)


def mean_info_curve(grid, ranks, infos):
    """(mean_info, n_eff, interpolated) over `grid`.

    Two departures from a plain kernel read, each covering a failure the other
    cannot:

      EDGES absorb everything past them (see kernel_stats). An outermost knot has
      no outward neighbour to borrow from, so absorption is the only way to make it
      measurable where the corpus thins — and it is the reader's own semantics.

      INTERIOR thin knots are linearly interpolated from their nearest measurable
      neighbours. The corpus's sparse regions are pocked rather than cut off, so a
      lone knot can fall under the bar between two that hold; truncating the grid
      there would throw away everything beyond a pinhole. This is NOT the forbidden
      repair of widening the kernel: that reaches out to the slope-clamp ridge near
      rank -20 and lands ~5x off its neighbours, whereas two-neighbour interpolation
      is local and imports nothing from outside the gap it spans.

    A thin RUN longer than SE_FLOOR_MAX_INTERP_RUN is refused rather than bridged:
    past that the curve would be invented across a real gap rather than filled
    across a pinhole. The caller is expected to report every interpolated knot —
    a silently patched curve reads as a measured one.
    """
    n = grid.size
    mean_info = np.empty(n)
    n_eff = np.empty(n)
    for i, x in enumerate(grid):
        mean_info[i], n_eff[i] = kernel_stats(
            x, ranks, infos,
            absorb_below=(i == 0), absorb_above=(i == n - 1))
    thin = n_eff < SE_FLOOR_MIN_EFFECTIVE
    if thin[0] or thin[-1]:
        raise RuntimeError(
            "se_floor: an EDGE knot is unmeasurable even absorbing everything "
            "beyond it — the corpus has essentially nothing out there. "
            f"lo n_eff {n_eff[0]:.2f}, hi n_eff {n_eff[-1]:.2f}")
    interpolated = []
    i = 1
    while i < n - 1:
        if not thin[i]:
            i += 1
            continue
        j = i
        while j + 1 < n - 1 and thin[j + 1]:
            j += 1
        run = j - i + 1
        if run > SE_FLOOR_MAX_INTERP_RUN:
            raise RuntimeError(
                f"se_floor: {run} consecutive knots from rank {grid[i]:+.1f} to "
                f"{grid[j]:+.1f} are below {SE_FLOOR_MIN_EFFECTIVE} effective "
                f"problems — too wide to interpolate (max "
                f"{SE_FLOOR_MAX_INTERP_RUN}). The corpus has a real gap where the "
                "estimator can place a user, not a pinhole.")
        lo_i, hi_i = i - 1, j + 1
        for k in range(i, j + 1):
            t = (grid[k] - grid[lo_i]) / (grid[hi_i] - grid[lo_i])
            mean_info[k] = mean_info[lo_i] * (1 - t) + mean_info[hi_i] * t
            interpolated.append((float(grid[k]), float(n_eff[k])))
        i = j + 1
    if not np.isfinite(mean_info).all():
        raise RuntimeError("se_floor: non-finite mean information on the grid")
    return mean_info, n_eff, interpolated
