#!/usr/bin/env python3
"""Geometry of the two shipped curves: rating→rank, and the SE floor.

Both are published artifacts a reader consumes directly, and both are quoted for
their SHAPE rather than for any single value — how much a rank costs in rating
units at each end of the ladder, and how much irreducible standard error a full
window of at-rank problems leaves at each end. Nothing here fits anything: it
reads `pwa/tsumegos/{rating_to_rank,se_floor,population_constants}.json` and
measures them.

Two conversions matter and neither is re-derived here.

  A SEGMENT COST is the rating distance between two neighbouring rank knots,
  divided by the rank distance between them — rating units per rank, local to
  that stretch of the ladder. Its reciprocal is the curve's gradient in rank per
  rating unit, which is what turns a problem's discrimination into a rank-space
  slope, and therefore what makes the information (and the floor below) jump
  across a knot where the curve's steepness changes.

  A FLOOR is the standard error a full window at that rank would still leave.
  The shipped curve carries mean per-attempt INFORMATION, because information is
  the quantity that averages; the reader applies
  `1 / sqrt(prior_info + steady_window_weight * mean_info)`
  (`seFloorForMeanInfo`, pwa/shared/rank-window.js). The two constants come from
  the packaging step's own mirror of that formula rather than being restated,
  and the prior information is read from the shipped population constants.

Floors are evaluated AT THE KNOTS and the extremes are exact: the reader
interpolates information linearly between knots and the floor is monotone
decreasing in information, so the curve's largest and smallest floors are both
attained at a knot.

The read checks what every consumer assumes: one curve shared by both board
sizes, knots strictly increasing in both coordinates, strictly positive
information everywhere, and an SE-floor grid identical to the rank prior's grid
(which is what keeps the reader's lookup off its extrapolation path).

    ./.venv/bin/python3 -m ablations.curve_shape_figures
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
# The client's floor formula, single-sourced from the step that emits the curve
# (its own mirror of pwa/shared/rank-window.js seFloorForMeanInfo).
from pipeline.package_for_pwa import _WIN_STEADY_WEIGHT, _win_prior_info  # noqa: E402
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.ratings_io import load_M  # noqa: E402

TIER = "about"

TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "pwa", "tsumegos")
SE_FLOOR_PATH = os.path.join(TSUMEGOS_DIR, "se_floor.json")
POPULATION_PATH = os.path.join(TSUMEGOS_DIR, "population_constants.json")

# Which stretch of the ladder each regional figure covers, in internal rank (0 is
# the 1k/1d boundary, one unit per rank, integers are band edges). Stated here so
# a figure's note and the sentence quoting it cannot drift apart.
DAN_LO, DAN_HI = 0.0, 9.0            # the 1d…9d bands
MIDKYU_LO, MIDKYU_HI = -15.0, -5.0   # the 15k…6k bands

# The weak-end trough is read within this many ranks of the rating→rank curve's
# WEAKEST knot. It has to be a neighbourhood of that knot rather than "the lowest
# floor on the kyu side": the curve has a second information ridge at the 1k/1d
# boundary, whose floor is lower still, and the two are different features.
TROUGH_HALF_WIDTH = 2.0


def _load_curve():
    """The published rating→rank knots, checked to be one shared curve."""
    curve_path = os.path.join(TSUMEGOS_DIR, "rating_to_rank.json")
    require(os.path.exists(curve_path),
            f"no rating_to_rank.json in {TSUMEGOS_DIR} — run pipeline.package_for_pwa")
    # Every board size's key must be PRESENT and identical: the loader falls back
    # to any curve it can find, so an absent key would otherwise read as agreement.
    with open(curve_path) as f:
        shipped = json.load(f)
    ratings, ranks = load_M(TSUMEGOS_DIR)
    for size in ALL_BOARD_SIZES:
        require(str(size) in shipped, f"rating_to_rank.json has no curve for size {size}")
        rs, ks = load_M(TSUMEGOS_DIR, size=str(size))
        require(np.array_equal(ratings, rs) and np.array_equal(ranks, ks),
                "the board sizes ship different rating→rank curves")
    require(ratings.size >= 3, "the published rating→rank curve has too few knots")
    require(np.all(np.isfinite(ratings)) and np.all(np.isfinite(ranks)),
            "rating_to_rank.json: non-finite knot")
    require(np.all(np.diff(ratings) > 0), "rating_to_rank.json: ratings not strictly increasing")
    require(np.all(np.diff(ranks) > 0), "rating_to_rank.json: ranks not strictly increasing")
    return ratings, ranks


def _load_floor():
    """(rank grid, floor per knot) from the shipped information curve."""
    require(os.path.exists(SE_FLOOR_PATH), f"no {SE_FLOOR_PATH} — run pipeline.package_for_pwa")
    require(os.path.exists(POPULATION_PATH), f"no {POPULATION_PATH} — run pipeline.package_for_pwa")
    with open(SE_FLOOR_PATH) as f:
        sf = json.load(f)
    with open(POPULATION_PATH) as f:
        prior = json.load(f)["rank_prior"]
    for key in ("rank_lo", "rank_step", "mean_info"):
        require(key in sf, f"se_floor.json has no {key!r}")
    info = np.asarray(sf["mean_info"], dtype=float)
    require(info.size >= 2, "se_floor.json has too few knots")
    require(np.all(np.isfinite(info)), "se_floor.json: non-finite mean information")
    require(np.all(info > 0), "se_floor.json: non-positive mean information")
    lo, step = float(sf["rank_lo"]), float(sf["rank_step"])
    require(step > 0, "se_floor.json: non-positive rank step")
    # The floor grid IS the rank prior's grid — that is what keeps the reader's
    # lookup inside the knots, so a disagreement means the two artifacts came out
    # of different runs.
    require(float(prior["rank_lo"]) == lo and float(prior["rank_step"]) == step
            and len(prior["log_density"]) == info.size,
            "se_floor.json and the rank prior are on different grids")
    grid = lo + step * np.arange(info.size)
    floors = 1.0 / np.sqrt(_win_prior_info(prior) + _WIN_STEADY_WEIGHT * info)
    return grid, floors


def measure():
    ratings, ranks = _load_curve()
    # Rating units per rank, on each segment between neighbouring knots.
    cost = np.diff(ratings) / np.diff(ranks)
    seg_lo = ranks[:-1]
    dan_seg = (seg_lo >= DAN_LO) & (seg_lo < DAN_HI)
    require(dan_seg.any(), "the rating→rank curve has no knots in the dan range")
    cheapest_i = int(np.argmin(cost))

    grid, floors = _load_floor()
    dan = (grid >= DAN_LO) & (grid < DAN_HI)
    midkyu = (grid >= MIDKYU_LO) & (grid < MIDKYU_HI)
    require(dan.any(), "the SE-floor grid does not reach the dan range")
    require(midkyu.any(), "the SE-floor grid does not reach the middle kyus")
    weak_knot = float(ranks[0])
    trough = np.abs(grid - weak_knot) <= TROUGH_HALF_WIDTH
    require(trough.any(),
            "the SE-floor grid has no knot near the rating→rank curve's weakest knot")
    trough_i = int(np.argmin(np.where(trough, floors, np.inf)))

    return {
        "n_knots": int(ranks.size),
        "knot_rank_weak": weak_knot,
        "knot_rank_strong": float(ranks[-1]),
        "rating_span": float(ratings[-1] - ratings[0]),
        "cost_cheapest": float(cost[cheapest_i]),
        "cost_cheapest_at": float(seg_lo[cheapest_i]),
        "cost_dan_min": float(cost[dan_seg].min()),
        "cost_dan_max": float(cost[dan_seg].max()),
        "cost_dan_mean": float(cost[dan_seg].mean()),
        "cost_weak_edge": float(cost[0]),
        "gradient_ratio": float(cost[0] / cost[1]),   # rank-per-rating, second ÷ first
        "floor_dan_min": float(floors[dan].min()),
        "floor_dan_mean": float(floors[dan].mean()),
        "floor_midkyu_max": float(floors[midkyu].max()),
        "floor_midkyu_mean": float(floors[midkyu].mean()),
        "floor_trough": float(floors[trough_i]),
        "floor_trough_at": float(grid[trough_i]),
        "floor_grid_lo": float(grid[0]),
        "floor_grid_hi": float(grid[-1]),
    }


def figures():
    m = measure()
    dan_span = f"[{DAN_LO:+.0f}, {DAN_HI:+.0f}) in internal rank, the 1d…9d bands"
    kyu_span = f"[{MIDKYU_LO:+.0f}, {MIDKYU_HI:+.0f}) in internal rank, the 15k…6k bands"
    return [
        Figure("curve.knot_rank_weak", m["knot_rank_weak"], ".0f",
               "weakest knot of the published rating→rank curve, in internal rank "
               "(integers are band edges, so this is the weak edge of that band)"),
        Figure("curve.knot_rank_strong", m["knot_rank_strong"], ".0f",
               "strongest knot of the published rating→rank curve, in internal rank "
               "(the strong edge of the strongest band it covers)"),
        Figure("curve.knot_rating_span", m["rating_span"], ".1f",
               "rating distance between the curve's outermost knots"),
        Figure("curve.rating_per_rank_cheapest", m["cost_cheapest"], ".3f",
               "rating units one rank costs on the curve's CHEAPEST segment"),
        Figure("curve.rating_per_rank_cheapest_at", m["cost_cheapest_at"], "rank",
               "band the cheapest segment spans (its weak-edge knot)"),
        Figure("curve.rating_per_rank_dan_min", m["cost_dan_min"], ".2f",
               f"cheapest segment through the dan range — segments starting in {dan_span}"),
        Figure("curve.rating_per_rank_dan_max", m["cost_dan_max"], ".2f",
               f"widest segment through the dan range — segments starting in {dan_span}"),
        Figure("curve.dan_vs_cheapest_min", m["cost_dan_min"] / m["cost_cheapest"], ".1f",
               "cheapest dan-range segment ÷ the curve's cheapest segment"),
        Figure("curve.dan_vs_cheapest_max", m["cost_dan_max"] / m["cost_cheapest"], ".1f",
               "widest dan-range segment ÷ the curve's cheapest segment"),
        Figure("curve.rating_per_rank_weak_edge", m["cost_weak_edge"], ".2f",
               "rating units one rank costs on the outermost weak-side segment"),
        Figure("curve.first_second_gradient_ratio", m["gradient_ratio"], ".1f",
               "the curve's SECOND segment ÷ its FIRST in rank per rating unit — how "
               "far a rank-space slope jumps across that one knot"),
        Figure("sefloor.dan_min", m["floor_dan_min"], ".2f",
               f"tightest SE floor over the dan range, in ranks — knots in {dan_span}"),
        Figure("sefloor.midkyu_max", m["floor_midkyu_max"], ".2f",
               f"loosest SE floor over the middle kyus, in ranks — knots in {kyu_span}"),
        Figure("sefloor.spread_ratio", m["floor_midkyu_max"] / m["floor_dan_min"], ".1f",
               "loosest middle-kyu floor ÷ tightest dan-range floor — the spread "
               "between the extremes a constant floor would have to cover"),
        Figure("sefloor.kyu_vs_dan_ratio", m["floor_midkyu_mean"] / m["floor_dan_mean"], ".1f",
               f"mean floor over the middle kyus ÷ mean floor over the dan range, each "
               f"an unweighted mean over that region's knots ({kyu_span}; {dan_span}) — "
               f"the TYPICAL kyu-vs-dan comparison, not the extremes"),
        Figure("sefloor.weak_trough", m["floor_trough"], ".2f",
               f"tightest SE floor within {TROUGH_HALF_WIDTH:.0f} ranks of the rating→rank "
               f"curve's weakest knot, in ranks — the weak-end trough, which is a "
               f"different feature from the ridge at the 1k/1d boundary"),
    ]


def main():
    from utils.rating.rank_surveys import value_to_rank
    m = measure()
    print(f"rating→rank curve: {m['n_knots']} knots, internal rank "
          f"{m['knot_rank_weak']:+.0f} ({value_to_rank(m['knot_rank_weak'])}) .. "
          f"{m['knot_rank_strong']:+.0f} (the strong edge of "
          f"{value_to_rank(m['knot_rank_strong'] - 1)}), spanning "
          f"{m['rating_span']:.3f} rating units\n")
    print("  segment cost = rating units one rank costs between two neighbouring knots")
    print(f"{'cheapest anywhere':>26}{m['cost_cheapest']:>10.4f}   "
          f"in the {value_to_rank(m['cost_cheapest_at'])} band")
    print(f"{'dan range':>26}{m['cost_dan_min']:>10.4f} .. {m['cost_dan_max']:.4f}"
          f"   mean {m['cost_dan_mean']:.4f}   "
          f"({m['cost_dan_min'] / m['cost_cheapest']:.1f}x .. "
          f"{m['cost_dan_max'] / m['cost_cheapest']:.1f}x the cheapest)")
    print(f"{'outermost weak segment':>26}{m['cost_weak_edge']:>10.4f}   "
          f"({m['cost_weak_edge'] / m['cost_cheapest']:.1f}x the cheapest; "
          f"{'wider' if m['cost_weak_edge'] > m['cost_dan_max'] else 'no wider'} "
          f"than every dan segment)")
    print(f"{'segment 2 / segment 1':>26}{m['gradient_ratio']:>10.4f}   "
          f"in rank per rating unit — the jump in rank-space slope across that knot")

    print(f"\nSE floor: the standard error in RANKS a full at-rank window still leaves, "
          f"over the shipped\n  information curve's knots (internal rank "
          f"{m['floor_grid_lo']:+.1f} .. {m['floor_grid_hi']:+.1f})")
    print(f"{'dan range ' + f'[{DAN_LO:+.0f}, {DAN_HI:+.0f})':>26}"
          f"{m['floor_dan_min']:>10.4f}   tightest knot; mean over the region "
          f"{m['floor_dan_mean']:.4f}")
    print(f"{'middle kyus ' + f'[{MIDKYU_LO:+.0f}, {MIDKYU_HI:+.0f})':>26}"
          f"{m['floor_midkyu_max']:>10.4f}   loosest knot;  mean over the region "
          f"{m['floor_midkyu_mean']:.4f}")
    print(f"{'extremes ratio':>26}"
          f"{m['floor_midkyu_max'] / m['floor_dan_min']:>10.4f}   "
          f"loosest middle-kyu ÷ tightest dan")
    print(f"{'typical ratio':>26}"
          f"{m['floor_midkyu_mean'] / m['floor_dan_mean']:>10.4f}   "
          f"mean middle-kyu ÷ mean dan")
    print(f"{'weak-end trough':>26}{m['floor_trough']:>10.4f}   "
          f"at internal rank {m['floor_trough_at']:+.1f} "
          f"({value_to_rank(m['floor_trough_at'])}), within {TROUGH_HALF_WIDTH:.0f} ranks "
          f"of the curve's weakest knot")


if __name__ == "__main__":
    main()
