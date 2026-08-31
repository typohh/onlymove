"""Map the combined cross-board rating to a CONSENSUS rank, and fit the per-server
rank couplings — self-contained, from rank_surveys alone.

The rating model (pipeline.train_mirt) puts problem difficulty and user ability on
ONE shared logit scale across both boards, so the rating→rank curve is a SINGLE shared
curve. This step produces, from the survey declarations:

  rating_to_rank.json   per-board (identical) {ratings, ranks} + a top-level
                        `time_adjust` block (the jointly-fit solve-time dock).
                        `ranks` is on the ONE project rank scale — 0 is the 1k/1d
                        boundary, every rank one unit wide (pwa/shared/rank-math.js).
  rank_map.json         per-server consensus↔server rank couplings — one per server,
                        EGF INCLUDED (EGF is not special; see below).

written to ratings/ (the backend). package_for_pwa is the single writer of the served
pwa/tsumegos/ — it copies these two curve files across.

THE MODEL — a latent CONSENSUS rank, every server (EGF included) a coupling off it:

    time-adjusted rating ──curve M──▶ CONSENSUS rank ──per server──▶ server rank

  * The user's rating is their 80%-CLEAR rating (CALIBRATION_P), pooled across boards
    (`fit_user_rating`, full 4PL with each problem's own slope/floor + the global
    ceiling λ) — the SAME quantity the client computes, so the loop stays closed.
  * ANCHORS ARE CAREER-SEGMENT RATINGS, NOT ONE POOLED RATING PER USER. Each user's
    first-time attempts are cut into overlapping 2-month blocks (BLOCK_DAYS stepped
    BLOCK_STEP); every block with ≥ MIN_ATTEMPTS joined attempts gets its own
    80%-clear rating and becomes an anchor row, attached to the user's
    nearest-in-time declaration per server ("no rank drift unless a survey says
    otherwise" — surveys only began 2026-05, so old attempt eras carry the same
    declared rank). Rows are weighted by ATTEMPT MASS — each attempt's unit mass is
    split over the blocks covering it, so the overlap can't double-count an era and
    the career's edge eras aren't underweighted — and normalized so each declaration
    contributes total weight 1 regardless of career length. A user with no fittable
    block (never MIN_ATTEMPTS inside one window) falls back to the pooled career
    rating as a single anchor. Validated by paired 50-folding CV
    (ablations.eval_rank_map_cv): held-out NLL −0.04±0.01 and deep-kyu MAE 5.2→4.2
    ranks vs the pooled-career anchor, dan band unchanged. The same harness tested
    integrating each rating's Fisher CI into the likelihood (errors-in-variables)
    and REJECTED it — declaration noise dominates, so it over-steepens the curve and
    is consistently worse held-out.
  * RATINGS ARE TIME-ADJUSTED THROUGHOUT THE MAPPING. Each declaration's effective
    rating is `rating + α·(log10(median solve min) − center)`, with the slope α fit
    JOINTLY in the same objective (it is identified — the time feature is per-user
    while the curve is a function of rating and each coupling is a per-server constant,
    so neither can absorb a per-user time shift). The client reproduces α·(…) for every
    server translation; the ONLY place it drops the dock is the headline onlymove rank
    (a display choice — see pwa/js/rank-translation.js). α + center ride in
    rating_to_rank.json's `time_adjust` and fit_population_constants reads them through
    into population_constants.json (no separate fit). A SIMPLE log-time slope is all the
    survey data supports — no rating-interaction, no power-law (see
    docs/methodology.md), and no rank-residualization line (the joint
    couplings absorb the rank-correlated part of solve time on their own).
  * M: a MONOTONE per-segment fit with the EGD even_game_winrates.txt cubic SHAPE as a PRIOR.
    The gap between consecutive rating-thresholds is `β·ΔS(g)·exp(δ_g)` (>0 ⇒ monotone);
    δ_g=0 reproduces the rigid affine `α_c + β·S`, and `LAMBDA_SHAPE·Σδ_g²` penalises
    deviation. The shape is the ΔAIC-best fit to the even-game win-rate table
    (utils.rating.rank_curve_shape) — the right convex backbone for the sparse tails. That
    table is EGD tournament outcomes between declared grades, which is a DIFFERENT thing
    from the EGF coupling below: the shape is a prior on the curve, EGF is one server
    coupling off it, and neither privileges the other. The
    prior is GENTLE (LAMBDA_SHAPE small): once the curve is anchored by ALL servers it
    is no longer kyu-starved, so a strong prior over-constrains.
  * EGF IS JUST ANOTHER SERVER. Every server, EGF included, gets free ordinal thresholds
    (a cumulative-link coupling, uniform-spacing prior SERVER_LAMBDA). The consensus
    rank is the hub; each server couples off it.
  * ROBUST TO JUNK DECLARATIONS. Each declaration's ordinal likelihood is a
    contamination mixture: with probability EPS_CONTAMINATION the declaration is
    junk (a lie/misclick — e.g. a lone Fox 6d declared on an account measuring
    ~19k), uniform over that server's categories. This caps a junk row's NLL and
    sends its gradient to 0 instead of letting it steer the coupling; honest
    extreme declarations (merely improbable, often cross-server-consistent) keep
    most of their weight. EPS IS DELIBERATELY SMALL: the paired CV
    (ablations.eval_rank_map_cv, the +rob variants, scored on honest declarations
    only via --drop-eval) shows held-out NLL degrades MONOTONICALLY in eps — the
    plain logistic's linear tails already bound junk influence, so a large junk
    floor lets the fit over-sharpen its couplings (worse calibration, point
    metrics ~neutral): +0.014±0.010 nats/declaration at 0.5%, +0.021±0.009 at 1%,
    +0.041±0.011 at 2.5%. 0.5% is the accepted compromise — near-neutral NLL
    while still capping any single junk row's pull at −log(eps/N) ≈ 8.7 nats
    (vs ~19 unbounded). Don't raise it without re-running that CV.
  * THE GAUGE — TWO PRIORS, NOT ONE. With no server privileged the LIKELIHOOD leaves the
    consensus scale free up to a MONOTONE reparameterisation (remap the latent, absorb it
    into every coupling's free thresholds and the curve's free per-segment gaps). That is
    infinite-dimensional, so GAUGE_WEIGHT alone cannot pin it and never did:
      - SERVER_LAMBDA (the uniform-spacing prior pulling each coupling's spacings toward
        the baseline of 1 consensus unit per server rank) is what removes all but the
        AFFINE part, scale included — a non-affine remap unbalances the spacings of every
        coupling at once, a dilation scales them all. The curve pins nothing: its
        steepness (log β) is free and its shape prior is deliberately gentle.
      - GAUGE_WEIGHT then pins the one direction spacings are blind to, a rigid
        TRANSLATION: a VERY WEAK penalty on the reliability-weighted (by declaration
        count) mean coupling offset toward identity. Weak because the likelihood is flat
        along it, so the location has no motivation to drift — a strong pin is unneeded
        and would shrink the real per-server offsets.
    Both are soft: they select a preferred point on a flat manifold rather than removing
    the freedom. The data-rich servers set the scale, which lands ≈ EGF, so the headline
    barely moves from the previous EGF-anchored rank.

Run pipeline.fetch_database first to refresh db/d1.{sql,sqlite} (the survey +
attempts source); this step then needs neither GPU nor the heavy pipeline. Then publish
with pipeline.package_for_pwa.
"""

import argparse
import json
import csv
import os
import sqlite3
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, load_rank_surveys, rank_band, value_to_rank,
)
from utils.rating.rank_curve_shape import fit_shape, rank_to_axis  # noqa: E402
from utils.rating.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec, junk_posterior_vec,
    ordinal_nll_vec, separated_head, _unpack_thresholds,
)
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.ratings_io import (  # noqa: E402
    FLOOR_CSV, RATINGS_CSV, SLOPE_CSV, load_lapse, load_value_map,
)

SIZES = list(ALL_BOARD_SIZES)

# --- Model config (baked; this module is production-only, no experimentation) ---
RATING_SHAPE = "cubic"      # even_game_winrates.txt shape used as the curve's prior MEAN
LAMBDA_SHAPE = 0.3          # GENTLE shape prior (the all-server latent isn't kyu-starved)
SERVER_LAMBDA = 5.0         # uniform-spacing prior on each server coupling
GAUGE_WEIGHT = 1.0          # VERY WEAK pin on the n-weighted mean coupling offset → 0
EPS_CONTAMINATION = 0.005   # P(a declaration is junk) — robust mixture; small ON PURPOSE
                            # (held-out NLL degrades monotonically in eps, see docstring)
LAMBDA_TIME = 0.1           # light ridge keeping the learned time slope bounded
# Rank BANDS (rank_surveys.rank_band): band b spans [b, b+1), so band 0 is 1d and
# band -1 is 1k, and a band's lower EDGE is the integer b itself. Every threshold
# below is such an edge.
EGF_LO, EGF_HI = -21, 9     # consensus-rank band range the curve spans
RANK_PADDING = 2            # extend a server's thresholds past its declared min/max
EXTEND_TO = 8               # force each server range to cover at least up to 9d
MIN_SERVER_SAMPLES = 3      # skip a server with less pooled declaration-weight
JUNK_FLOOR = 0.5            # exclude above this junk posterior — see the screening note
MIN_ATTEMPTS = 20           # drop survey users (and career blocks) with fewer attempts
BLOCK_DAYS = 60             # career-segment anchor blocks: window length ...
BLOCK_STEP = 30             # ... and step (overlapping; see the docstring)

# Solve-time feature (server side): non-repeat, timed, AFK-capped median over the
# most-recent WORKING_SET attempts, in minutes. This approximates but no longer
# exactly matches the client — the client's timeAdjustDeltaRating now uses a
# positional-weighted working-set median with no AFK cap (a deliberate divergence),
# so treat this as the mapping's own solve-time definition, not a strict mirror.
AFK_CAP_MS = 180000
MIN_TIMED = 20
WORKING_SET = 3000


def load_ratings(ratings_dir, size):
    """{tsumego_id: rating} from ratings/<size>/tsumego_ratings.csv — the 80%-clear
    black-box scalar package_for_pwa ships."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), RATINGS_CSV), "rating", missing_ok=True)


def load_slopes(ratings_dir, size):
    """{tsumego_id: a_p} from tsumego_slopes.csv — the raw 2PL discrimination
    (`quality`). fit_user_rating uses these so the survey 80%-clear ratings anchor M with
    the SAME per-problem slope the client estimator uses."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), SLOPE_CSV), "slope", missing_ok=True)


def load_floors(ratings_dir, size):
    """{tsumego_id: c_p} from tsumego_floors.csv — the per-problem 4PL guessing floor,
    fed (with the global ceiling λ) through fit_user_rating to match the client's irt.js."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), FLOOR_CSV), "floor", missing_ok=True)


def _shape_at(ys):
    """even_game_winrates.txt RATING_SHAPE value at each consensus-boundary rank in `ys`, the
    curve's prior mean. even_game_winrates.txt is fitted on its OWN axis (its origin is part
    of the polynomial family), so convert at the boundary via rank_to_axis."""
    params, theta = fit_shape(RATING_SHAPE)
    return np.asarray(theta(rank_to_axis(ys), params), dtype=float)


def load_user_median_solve_minutes(sqlite_path):
    """{user_prefix8: median novel solve minutes} over each user's working-set window —
    reproducing the client's medianNovelSolveMinutes() on getMergedAnswerHistory(): the
    most-recent WORKING_SET attempts unified across boards (ordered by created_at so
    backfilled rows land at their true position), then filtered to non-repeat + timed
    (0, AFK cap]. None below MIN_TIMED."""
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.execute(
            "SELECT user_id, is_repeat, time_used FROM tsumego_reports ORDER BY created_at, id")
        windows = defaultdict(lambda: deque(maxlen=WORKING_SET))
        for uid, is_repeat, t in cur:
            windows[uid[:8]].append((is_repeat, t))
    finally:
        con.close()
    out = {}
    for pref, rows in windows.items():
        ts = [float(t) for (rep, t) in rows
              if not rep and t is not None and 0 < float(t) <= AFK_CAP_MS]
        if len(ts) >= MIN_TIMED:
            out[pref] = float(np.median(ts)) / 60000.0
    return out


def load_timed_attempts(sqlite_path, user_ids):
    """{user_id: (created_days ascending np.array, [(tsumego_id, success), ...]
    aligned)} — first-time attempts with event times (the block anchors need them;
    the .sql regex loader doesn't carry timestamps)."""
    ids = sorted(user_ids)
    con = sqlite3.connect(sqlite_path)
    try:
        q = ("SELECT user_id, problem, success, created_at FROM tsumego_reports "
             "WHERE (is_repeat IS NULL OR is_repeat != 1) AND user_id IN ({})"
             .format(",".join("?" * len(ids))))
        df = pd.read_sql_query(q, con, params=ids)
    finally:
        con.close()
    df["t"] = (pd.to_datetime(df["created_at"], utc=True, format="mixed")
               .astype("int64") / 86400e9)
    df = df.sort_values("t")
    return {uid: (grp["t"].to_numpy(),
                  list(zip(grp["problem"].tolist(), grp["success"].astype(int).tolist())))
            for uid, grp in df.groupby("user_id")}


def block_ratings(times, rows, ratings, slopes, floors, lapse):
    """[(t_center_days, rating, attempt_mass)] — one 80%-clear rating per
    overlapping BLOCK_DAYS window stepped BLOCK_STEP through the user's career
    (windows below MIN_ATTEMPTS joined attempts are skipped). mass = each attempt's
    unit mass split equally over the blocks covering it, so every attempt counts
    exactly once however the grid overlaps. Empty list = no fittable block; the
    caller falls back to the pooled career rating."""
    if len(times) == 0:
        return []
    spans, seen = [], set()
    for s in np.arange(times[0], times[-1] + 1e-9, BLOCK_STEP):
        lo = int(np.searchsorted(times, s, side="left"))
        hi = int(np.searchsorted(times, s + BLOCK_DAYS, side="right"))
        if hi - lo < MIN_ATTEMPTS or (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        R, n = fit_user_rating(rows[lo:hi], ratings, min_n=MIN_ATTEMPTS,
                               slopes=slopes, floors=floors, lapse=lapse)
        if R is not None and n >= MIN_ATTEMPTS:
            spans.append((lo, hi, R))
    if not spans:
        return []
    cov = np.zeros(len(rows))
    for lo, hi, _ in spans:
        cov[lo:hi] += 1
    inv = np.where(cov > 0, 1.0 / np.maximum(cov, 1), 0.0)
    return [(float((times[lo] + times[hi - 1]) / 2), R, float(np.sum(inv[lo:hi])))
            for lo, hi, R in spans]


# ---------------------------------------------------------------------------
# The consensus fit
# ---------------------------------------------------------------------------
def _curve_arrays():
    egf_thr_ranks = list(range(EGF_LO + 1, EGF_HI + 1))
    ys = np.array([float(g) for g in egf_thr_ranks], dtype=float)  # band lower edges
    S = _shape_at(ys)
    dS = np.diff(S)
    return egf_thr_ranks, ys, S, dS


def _bucket(decls):
    """One coupling block per server (EGF included): rating, declared-category index k,
    the de-meaned time feature T, the identity-line target, and the per-row weight w
    (block anchors normalize each declaration's rows to total weight 1; a plain
    4-tuple row defaults to w=1). `g` is a rank BAND index (see rank_surveys.rank_band)."""
    rows_by = defaultdict(list)
    for server, R, g, T, *w in decls:
        rows_by[server].append((R, int(g), T, float(w[0]) if w else 1.0))
    blocks = {}
    for server, rows in rows_by.items():
        if sum(w for *_, w in rows) < MIN_SERVER_SAMPLES:
            continue
        declared = [g for _, g, _, _ in rows]
        g_min = min(declared) - RANK_PADDING
        g_max = max(max(declared) + RANK_PADDING, EXTEND_TO)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        gs = list(range(g_min + 1, g_max + 1))
        blocks[server] = {
            "g_min": g_min, "g_max": g_max,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "identity": np.array([float(g) for g in gs], float),
            "rating": np.array([r for r, _, _, _ in rows], float),
            "T": np.array([t for _, _, t, _ in rows], float),
            "w": np.array([w for _, _, _, w in rows], float),
            "k": np.array([max(0, min(N - 1, g - g_min)) for _, g, _, _ in rows], int)}
    return blocks


def fit_consensus(decls):
    """Jointly fit the rating→consensus curve M, the global time slope α, and the
    per-server couplings (EGF included). decls rows are (server, rating, declared
    scalar, T[, weight]) — weight defaults to 1 for a plain one-row-per-declaration
    anchor. Returns (a, alpha, ys, egf_thr_ranks, server_fits) where `a` are the
    rating values at each consensus boundary `ys`."""
    decls = [d if len(d) == 5 else (*d, 1.0) for d in decls]
    egf_thr_ranks, ys, S, dS = _curve_arrays()
    nseg = len(dS)
    blocks = _bucket(decls)
    if not blocks:
        sys.exit("No servers with enough declarations — cannot fit.")

    # Init the curve affine from a POOLED linear pre-fit over all declarations (no server
    # privileged), then express it in the shape's (α_c, log β) parameterisation.
    all_R = np.array([R for _, R, _, _, _ in decls], float)
    all_g = np.array([g for _, _, g, _, _ in decls], float)
    beta, alpha_c = np.polyfit(all_R, all_g, 1)
    if beta <= 0:
        beta = 1e-3
    prior_rating = (ys - alpha_c) / beta
    b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
    a0 = float(prior_rating[0] - b0 * S[0])

    srv_layout, x0, idx = [], [a0, float(np.log(max(b0, 1e-3)))] + [0.0] * nseg + [0.0], 3 + nseg
    for server, blk in blocks.items():
        n = len(blk["baseline"]) + 1
        srv_layout.append((server, idx, n))
        x0.append(float(blk["g_min"]))
        x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, float)

    def unpack_curve(params):
        beta_ = np.exp(params[1])
        a = np.empty(len(S))
        a[0] = params[0] + beta_ * S[0]
        if nseg:
            a[1:] = a[0] + np.cumsum(beta_ * dS * np.exp(params[2:2 + nseg]))
        return a, params[2 + nseg]   # a, alpha (time slope)

    def objective(params):
        a, alpha = unpack_curve(params)
        nll, pen, w_off, w_tot = 0.0, 0.0, 0.0, 0.0
        for server, st, n in srv_layout:
            blk = blocks[server]
            thr = _unpack_thresholds(params[st:st + n], blk["baseline"])
            x_int = interp_extrap_vec(blk["rating"] + alpha * blk["T"], a, ys)
            nll += ordinal_nll_vec(thr, x_int, blk["k"], blk["w"],
                                   eps=EPS_CONTAMINATION)
            pen += SERVER_LAMBDA * float(np.sum(params[st + 1:st + n] ** 2))
            w = float(np.sum(blk["w"]))     # reliability weight = declaration weight
            w_off += w * float(np.mean(thr - blk["identity"]))
            w_tot += w
        if nseg:
            pen += LAMBDA_SHAPE * float(np.sum(params[2:2 + nseg] ** 2))
        pen += LAMBDA_TIME * float(alpha ** 2)
        pen += GAUGE_WEIGHT * (w_off / w_tot) ** 2         # gauge: n-weighted mean offset → 0
        return nll + pen

    # Intermediate optimizer steps can momentarily collapse the curve (β→0), which
    # divide-by-zeros in interp_extrap_vec's end-slope; the final fit is well-conditioned.
    with np.errstate(divide="ignore", invalid="ignore"):
        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 10000})
    a, alpha = unpack_curve(res.x)
    server_fits = {}
    for server, st, n in srv_layout:
        blk = blocks[server]
        thr = _unpack_thresholds(res.x[st:st + n], blk["baseline"])
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        server_fits[server] = {gs[i]: float(thr[i]) for i in range(len(gs))}
    return a, float(alpha), ys, egf_thr_ranks, server_fits


EXCLUSIONS_CSV = "declaration_exclusions.csv"


def detect_junk_declarations(decls, keys, a, ys, alpha, server_fits):
    """Declarations the fit itself calls junk, separated from the rest by a gap.

    A declaration whose author is describing a different player than the attempt
    log does — a claim typed for effect, a rank carried over from an account that
    is not this one, a registration grade never revisited — is not noise the
    likelihood should average over. The contamination mixture already caps what
    one can pull, and a capped pull is still a pull, applied by rows that are not
    measurements of anything.

    What separates them is not the size of the miss. Servers differ in how far
    apart their categories sit and in how far their populations spread, so a
    six-rank residual is unremarkable in one place and impossible in another;
    the fitted thresholds carry exactly that, which is why the statistic here is
    the mixture's own posterior rather than the residual (`junk_posterior_vec`).
    Each declaration is scored at the attempt-mass-weighted mean of its block
    rows' time-adjusted positions, the position the fit saw.

    The cut is the posterior's own decision boundary: above one half the mixture
    calls a declaration likelier junk than measurement. That bar is severe rather
    than nominal, because the junk prior is small — clearing it takes a declared
    category the fitted thresholds give a likelihood on the order of one in
    thousands, which is a claim no ordinary miss reaches.

    A knee is the tempting alternative and is not used as the cut. The largest
    gap in a sample is the largest gap somewhere even when the sample is one
    smooth population, so cutting there excludes down to wherever the decay
    flattens and reports the result as a discovered boundary; the gap is printed
    (`separated_head`) as a diagnostic of whether a genuinely detached group
    exists, which is a different question from where to stop.

    This runs ONCE against the first fit rather than iterating to convergence:
    each pass would tighten the curve onto the survivors and re-flag whatever now
    sits furthest out, converging on an arbitrarily clean set rather than on the
    truth.

    Returns (dropped, cut) — dropped is a list of dicts, one per excluded
    declaration, ordered by descending posterior; cut is (n, lowest excluded
    posterior, the separated head's gap)."""
    x_by_key, w_by_key, cat_by_key = defaultdict(float), defaultdict(float), {}
    for (server, R, g, T, w), key in zip(decls, keys):
        x_by_key[key] += w * float(interp_extrap_vec(np.array([R + alpha * T]), a, ys)[0])
        w_by_key[key] += w
        cat_by_key[key] = (server, g)

    rows = []
    for key, xw in x_by_key.items():
        server, g = cat_by_key[key]
        thr = server_fits.get(server)
        if not thr or len(thr) < 2:
            continue                      # no coupling: nothing to be junk against
        gs = sorted(thr)
        tau = np.array([thr[k] for k in gs], float)
        x = xw / max(w_by_key[key], 1e-12)
        k = int(np.searchsorted(np.array([float(v) for v in gs]), g))
        p = float(junk_posterior_vec(tau, np.array([x]), np.array([k]),
                                     EPS_CONTAMINATION)[0])
        rows.append({"key": key, "server": server, "declared_band": g,
                     "internal": x, "p_junk": p})
    if not rows:
        return [], (0, float("nan"), 0.0)

    rows.sort(key=lambda r: -r["p_junk"])
    n_cut = sum(1 for r in rows if r["p_junk"] > JUNK_FLOOR)
    n_knee, knee_at, gap = separated_head([r["p_junk"] for r in rows], floor=JUNK_FLOOR)
    print(f"\nDeclaration screening over {len(rows)} declarations: "
          f"{n_cut} above the {JUNK_FLOOR} junk posterior")
    if not n_cut:
        print("  none — nothing excluded")
        return [], (0, float("nan"), 0.0)
    print(f"  separated head ends after {n_knee} (largest gap {gap:.3f} below "
          f"{knee_at:.3f}); the cut is the {JUNK_FLOOR} boundary, at {n_cut}")
    for i, r in enumerate(rows[:n_cut + 2]):
        uid, server, g = r["key"]
        mark = "  cut" if i < n_cut else "  keep"
        head = " <- head" if i < n_knee else ""
        print(f"   {mark:<6}{uid[:8]:<10}{server:<9}declared {value_to_rank(g + 0.5):>4}  "
              f"we say {value_to_rank(r['internal']):>4}  P(junk)={r['p_junk']:.3f}{head}")
    return rows[:n_cut], (n_cut, float(rows[n_cut - 1]["p_junk"]), gap)


def write_exclusions(ratings_dir, dropped, cut, n_declarations):
    """The excluded declarations, for anything that scores the mapping against the
    same survey — an evaluation keeping rows the fit did not see is measuring a
    different comparison. Written every run, empty header included, so a reader
    never has to branch on the file's absence. The `p_junk` column is the value at
    the FIRST fit, the one the cut was taken on."""
    path = os.path.join(ratings_dir, EXCLUSIONS_CSV)
    n_cut, threshold, gap = cut
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_prefix", "server", "declared_band", "declared_rank",
                    "our_internal", "our_rank", "p_junk"])
        for r in dropped:
            uid, server, g = r["key"]
            w.writerow([uid[:8], server, g, value_to_rank(g + 0.5),
                        f"{r['internal']:.4f}", value_to_rank(r["internal"]),
                        f"{r['p_junk']:.6f}"])
    print(f"  wrote {path} ({len(dropped)} of {n_declarations} declarations"
          + (f", lowest excluded {threshold:.3f}; separated head ends "
             f"{gap:.3f} above it)" if dropped else ")"))
    return path


def emit_server_curves(server_fits):
    """Per-server interpolation knots {server: {our, server}} in scalar units — `our` =
    the consensus-rank scalar at each server-rank boundary, `server` the server scalar
    there. EGF is emitted like any other server (no identity special-case). In SERVERS
    order for a stable display."""
    out = {}
    for server in SERVERS:
        thr = server_fits.get(server)
        if not thr or len(thr) < 2:
            continue
        gs = sorted(thr)
        out[server] = {"our": [float(thr[g]) for g in gs],
                       "server": [float(g) for g in gs]}
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--sqlite", default=DB_SQLITE)
    args = ap.parse_args()

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — cannot fit the mapping.")
    # EVERY declaration anchors (weight 1 each; its blocks split that weight) —
    # a re-survey is a new anchor at its own time, not a replacement.
    surveys = surveys[surveys["rank_value"].notna()].copy()
    surveys["t_days"] = (pd.to_datetime(surveys["created_at"], utc=True, format="mixed")
                         .astype("int64") / 86400e9)
    print(f"  {len(surveys)} declarations from {surveys['user_id'].nunique()} users")

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No per-problem ratings found — run train_mirt first.")
    lapse = load_lapse(args.dir)
    # First-time attempts only: the client's rank estimator anchors on first-time
    # attempts, so build the survey anchor from novel problems too — repeats'
    # spaced-repetition success rate would otherwise inflate review-heavy users'
    # 80%-clear ratings. (Both boards land in one time-ordered list; the ratings
    # dicts are merged across sizes, so pooling is implicit.)
    attempts = load_timed_attempts(args.sqlite, set(surveys["user_id"].unique()))
    med_min = load_user_median_solve_minutes(args.sqlite)

    # Per-user career-block ratings (see docstring) + the log10 median solve time.
    print(f"\nEstimating {int(CALIBRATION_P * 100)}%-clear ratings per {BLOCK_DAYS}d "
          f"career block (pooled boards, 4PL, λ={lapse:.4f}) ...")
    user_blocks, user_logT, fallbacks = {}, {}, 0
    for uid in surveys["user_id"].unique():
        times, rows = attempts.get(uid, (np.zeros(0), []))
        R, n = fit_user_rating(rows, merged_ratings, min_n=MIN_ATTEMPTS,
                               slopes=merged_slopes, floors=merged_floors, lapse=lapse)
        if R is None or n < MIN_ATTEMPTS:
            continue
        blks = block_ratings(times, rows, merged_ratings, merged_slopes,
                             merged_floors, lapse)
        if not blks:   # sparse career: never MIN_ATTEMPTS inside one block window
            blks = [(float(np.mean(times)), R, 1.0)]
            fallbacks += 1
        user_blocks[uid] = blks
        pref = uid[:8]
        if pref in med_min and med_min[pref] > 0:
            user_logT[uid] = float(np.log10(med_min[pref]))
    timed = [user_logT[u] for u in user_blocks if u in user_logT]
    center = float(np.mean(timed)) if timed else 0.0   # the time dock's zero point
    nblk = [len(b) for b in user_blocks.values()]
    print(f"  {len(user_blocks)} users with a usable rating ({fallbacks} career "
          f"fallbacks; blocks/user median {int(np.median(nblk))}, max {max(nblk)}); "
          f"{len(timed)} with a time feature (center = log10 median min "
          f"{center:+.3f}, {10 ** center * 60:.0f}s)")

    # Anchor rows: (server, block rating, declared scalar, de-meaned time feature,
    # weight). Each block attaches to the user's nearest-in-time declaration on that
    # server, and a declaration's rows are attempt-mass-normalized to total weight 1.
    # Users without a usable time feature get T=0 (no time term).
    decls, keys = [], []
    for (uid, server), srows in (surveys[surveys["user_id"].isin(user_blocks)]
                                 .groupby(["user_id", "server"])):
        T = (user_logT[uid] - center) if uid in user_logT else 0.0
        dt = srows["t_days"].to_numpy(float)
        dg = np.array([rank_band(v) for v in srows["rank_value"].to_numpy(float)])
        assign = defaultdict(list)
        for tc, R, m in user_blocks[uid]:
            assign[int(np.argmin(np.abs(dt - tc)))].append((R, m))
        for j, brs in assign.items():
            mass = sum(m for _, m in brs)
            for R, m in brs:
                decls.append((server, R, int(dg[j]), T,
                              m / mass if mass > 0 else 1.0 / len(brs)))
                # A declaration is (user, server, declared band); its block rows
                # share one key, so an exclusion drops the declaration whole.
                keys.append((uid, server, int(dg[j])))
    w_egf = sum(d[4] for d in decls if d[0] == "EGF")
    print(f"\nConsensus fit over {len(decls)} anchor rows, "
          f"{sum(d[4] for d in decls):.0f} declaration-weight [{w_egf:.0f} EGF, a "
          f"coupling like any server; cubic shape prior λ_shape={LAMBDA_SHAPE}, "
          f"server λ={SERVER_LAMBDA}, gauge weight={GAUGE_WEIGHT}]")

    a, alpha, ys, egf_thr_ranks, server_fits = fit_consensus(decls)
    print(f"  learned time slope α = {alpha:+.4f} rating per de-meaned log10-min")

    dropped, cut = detect_junk_declarations(decls, keys, a, ys, alpha, server_fits)
    write_exclusions(args.dir, dropped, cut, len(keys))
    if dropped:
        drop_keys = {d["key"] for d in dropped}
        decls = [d for d, k in zip(decls, keys) if k not in drop_keys]
        keys = [k for k in keys if k not in drop_keys]
        print(f"  refitting without them ({len(decls)} anchor rows remain)")
        a, alpha, ys, egf_thr_ranks, server_fits = fit_consensus(decls)
        print(f"  learned time slope α = {alpha:+.4f} rating per de-meaned log10-min")

    # In-sample EGF calibration check (EGF is now forward-mapped through its
    # coupling); weighted so it reads in declaration units, not block-row counts.
    egf = [(R, g, T, w) for srv, R, g, T, w in decls if srv == "EGF"]
    if "EGF" in server_fits and len(egf) >= 2 and len(server_fits["EGF"]) >= 2:
        gs = sorted(server_fits["EGF"])
        our = np.array([server_fits["EGF"][k] for k in gs], float)
        srv = np.array([float(k) for k in gs], float)
        order = np.argsort(our)
        R = np.array([r for r, _, _, _ in egf])
        # Declared BAND -> its centre, so the residual against a boundary-scale
        # prediction reads in the same units it always has.
        gdec = np.array([g + 0.5 for _, g, _, _ in egf], float)
        T = np.array([t for _, _, t, _ in egf]); W = np.array([w for _, _, _, w in egf])
        x_int = interp_extrap_vec(R + alpha * T, a, ys)
        pred = interp_extrap_vec(x_int, our[order], srv[order])
        res = pred - gdec
        print(f"  EGF-coupling calibration: RMSE {np.sqrt(np.average(res ** 2, weights=W)):.2f} | "
              f"bias {np.average(res, weights=W):+.2f} ranks")

    # --- write rating_to_rank.json (both board keys identical) + time_adjust ---
    # Knots sit at consensus-rank band LOWER EDGES, which under this convention are
    # the band integers themselves.
    curve = {"ratings": [float(r) for r in a],
             "ranks": [float(g) for g in egf_thr_ranks]}
    rr = {str(s): curve for s in SIZES}
    rr["time_adjust"] = {"alpha_per_log10_min": round(alpha, 6),
                         "center_log10_min": round(center, 6)}
    server_curves = emit_server_curves(server_fits)

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "rating_to_rank.json"), "w") as f:
        json.dump(rr, f, indent=2)
    with open(os.path.join(args.dir, "rank_map.json"), "w") as f:
        json.dump(server_curves, f, indent=2)

    fitted = [s for s in SERVERS if s in server_curves]
    print(f"\nconsensus range mapped: {value_to_rank(egf_thr_ranks[0])}→rating "
          f"{a[0]:.2f} ... {value_to_rank(egf_thr_ranks[-1])}→rating {a[-1]:.2f}")
    print(f"wrote {args.dir}/rating_to_rank.json (shared 9≡13 + time_adjust) + rank_map.json "
          f"({len(fitted)} server couplings incl EGF: {', '.join(fitted)})\n"
          f"  → publish with: python -m pipeline.package_for_pwa")


if __name__ == "__main__":
    main()
