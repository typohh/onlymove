"""Map the combined cross-board rating to a CONSENSUS rank, and fit the per-server
rank couplings — self-contained, from rank_surveys alone.

The rating model (pipeline.train_rating) puts problem difficulty and user ability on
ONE shared logit scale across both boards, so the rating→rank curve is a SINGLE shared
curve. This step produces, from the survey declarations:

  rating_to_rank.json   per-board (identical) {ratings, rank_indices} + a top-level
                        `time_adjust` block (the jointly-fit solve-time dock).
  rank_map.json         per-server consensus↔server rank couplings — one per server,
                        EGF INCLUDED (EGF is not special; see below).

written to ratings/ (the backend). package_for_pwa is the single writer of the served
pwa/tsumegos/ — it copies these two curve files across.

THE MODEL — a latent CONSENSUS rank, every server (EGF included) a coupling off it:

    time-adjusted rating ──curve M──▶ CONSENSUS rank ──per server──▶ server rank

  * The user's rating is their 80%-CLEAR rating (CALIBRATION_P), pooled across boards
    (`fit_user_rating`, full 4PL with each problem's own slope/floor + the global
    ceiling λ) — the SAME quantity the client computes, so the loop stays closed.
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
    docs/egf_anchor_bias_and_time.md), and no rank-residualization line (the joint
    couplings absorb the rank-correlated part of solve time on their own).
  * M: a MONOTONE per-segment fit with the EGF statistics.txt cubic SHAPE as a PRIOR.
    The gap between consecutive rating-thresholds is `β·ΔS(g)·exp(δ_g)` (>0 ⇒ monotone);
    δ_g=0 reproduces the rigid affine `α_c + β·S`, and `LAMBDA_SHAPE·Σδ_g²` penalises
    deviation. The shape is the ΔAIC-best fit to the even-game win-rate table
    (utils.statistics_shapes) — the right convex backbone for the sparse tails. The
    prior is GENTLE (LAMBDA_SHAPE small): once the curve is anchored by ALL servers it
    is no longer kyu-starved, so a strong prior over-constrains.
  * EGF IS JUST ANOTHER SERVER. Every server, EGF included, gets free ordinal thresholds
    (a cumulative-link coupling, uniform-spacing prior SERVER_LAMBDA). The consensus
    rank is the hub; each server couples off it.
  * THE GAUGE. With no server privileged the consensus scale is only defined up to a
    monotone reparameterisation. We pin it by a VERY WEAK penalty on the
    reliability-weighted (by declaration count) mean coupling offset toward identity
    (GAUGE_WEIGHT). Weak because the likelihood is flat in the gauge direction, so the
    scale has no motivation to drift — a strong pin is unneeded and would shrink the real
    per-server offsets. The data-rich servers set the scale, which lands ≈ EGF, so the
    headline barely moves from the previous EGF-anchored rank.

Run pipeline.fetch_database first to refresh attempts/database.{sql,sqlite} (the survey +
attempts source); this step then needs neither GPU nor the heavy pipeline. Then publish
with pipeline.package_for_pwa.
"""

import argparse
import json
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

from utils.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, latest_per_user_server, load_attempts_for_users,
    load_rank_surveys, scalar_to_rank,
)
from utils.statistics_shapes import fit_shape  # noqa: E402
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec, ordinal_nll_vec,
    _unpack_thresholds,
)
from utils.configurations import ALL_BOARD_SIZES  # noqa: E402
from utils.ratings_io import (  # noqa: E402
    FLOOR_CSV, QUALITY_CSV, RATINGS_CSV, load_lapse, load_value_map,
)

SIZES = list(ALL_BOARD_SIZES)

# --- Model config (baked; this module is production-only, no experimentation) ---
RATING_SHAPE = "cubic"      # statistics.txt shape used as the curve's prior MEAN
LAMBDA_SHAPE = 0.3          # GENTLE shape prior (the all-server latent isn't kyu-starved)
SERVER_LAMBDA = 5.0         # uniform-spacing prior on each server coupling
GAUGE_WEIGHT = 1.0          # VERY WEAK pin on the n-weighted mean coupling offset → 0
LAMBDA_TIME = 0.1           # light ridge keeping the learned time slope bounded
EGF_LO, EGF_HI = -20, 10    # consensus-rank boundary range the curve spans
RANK_PADDING = 2            # extend a server's thresholds past its declared min/max
EXTEND_TO = 9               # force each server range to cover at least up to 9d
MIN_SERVER_SAMPLES = 3      # skip a server with fewer pooled declarations
MIN_ATTEMPTS = 20           # drop survey users with fewer pooled attempts

# Solve-time feature — MUST match the client (pwa/js/population-constants.js
# medianNovelSolveMinutes over the working set): non-repeat, timed, AFK-capped median
# over the most-recent WORKING_SET attempts, in minutes.
AFK_CAP_MS = 180000
MIN_TIMED = 20
WORKING_SET = 3000

# rank_index axis (rating_to_rank.json) is 20k=0 … 1k=19 … 1d=20; the consensus rank
# scalar is 1k=0, 1d=1. So rank_index = scalar + 19.
RANK_INDEX_OFFSET = 19.0


def load_ratings(ratings_dir, size):
    """{tsumego_id: rating} from ratings/<size>/tsumego_ratings_final.csv — the 80%-clear
    black-box scalar package_for_pwa ships."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), RATINGS_CSV), "rating", missing_ok=True)


def load_slopes(ratings_dir, size):
    """{tsumego_id: a_p} from tsumego_quality_final.csv — the raw 2PL discrimination
    (`quality`). fit_user_rating uses these so the survey 80%-clear ratings anchor M with
    the SAME per-problem slope the client estimator uses."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), QUALITY_CSV), "quality", missing_ok=True)


def load_floors(ratings_dir, size):
    """{tsumego_id: c_p} from tsumego_floor_final.csv — the per-problem 4PL guessing floor,
    fed (with the global ceiling λ) through fit_user_rating to match the client's irt.js."""
    return load_value_map(
        os.path.join(ratings_dir, str(size), FLOOR_CSV), "floor", missing_ok=True)


def _shape_at(ys):
    """statistics.txt RATING_SHAPE value at each consensus-boundary scalar in `ys`
    (rank_index = scalar + 19), the curve's prior mean."""
    params, theta = fit_shape(RATING_SHAPE)
    return np.asarray(theta(np.asarray(ys, float) + RANK_INDEX_OFFSET, params), dtype=float)


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


# ---------------------------------------------------------------------------
# The consensus fit
# ---------------------------------------------------------------------------
def _curve_arrays():
    egf_thr_ranks = list(range(EGF_LO + 1, EGF_HI + 1))
    ys = np.array([g - 0.5 for g in egf_thr_ranks], dtype=float)   # consensus scalar
    S = _shape_at(ys)
    dS = np.diff(S)
    return egf_thr_ranks, ys, S, dS


def _bucket(decls):
    """One coupling block per server (EGF included): rating, declared-category index k,
    the de-meaned time feature T, the identity-line target, and n (reliability weight)."""
    rows_by = defaultdict(list)
    for server, R, g, T in decls:
        rows_by[server].append((R, int(round(g)), T))
    blocks = {}
    for server, rows in rows_by.items():
        if len(rows) < MIN_SERVER_SAMPLES:
            continue
        declared = [g for _, g, _ in rows]
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
            "identity": np.array([g - 0.5 for g in gs], float),
            "rating": np.array([r for r, _, _ in rows], float),
            "T": np.array([t for _, _, t in rows], float),
            "k": np.array([max(0, min(N - 1, g - g_min)) for _, g, _ in rows], int)}
    return blocks


def fit_consensus(decls):
    """Jointly fit the rating→consensus curve M, the global time slope α, and the
    per-server couplings (EGF included). Returns (a, alpha, ys, egf_thr_ranks, server_fits)
    where `a` are the rating values at each consensus boundary `ys`."""
    egf_thr_ranks, ys, S, dS = _curve_arrays()
    nseg = len(dS)
    blocks = _bucket(decls)
    if not blocks:
        sys.exit("No servers with enough declarations — cannot fit.")

    # Init the curve affine from a POOLED linear pre-fit over all declarations (no server
    # privileged), then express it in the shape's (α_c, log β) parameterisation.
    all_R = np.array([R for _, R, _, _ in decls], float)
    all_g = np.array([g for _, _, g, _ in decls], float)
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
            nll += ordinal_nll_vec(thr, x_int, blk["k"])
            pen += SERVER_LAMBDA * float(np.sum(params[st + 1:st + n] ** 2))
            w = float(len(blk["k"]))                       # reliability weight = n
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
                       "server": [float(g - 0.5) for g in gs]}
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--sqlite", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    args = ap.parse_args()

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — cannot fit the mapping.")
    surveys = latest_per_user_server(surveys)
    print(f"  {len(surveys)} latest declarations from {surveys['user_id'].nunique()} users")

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No per-problem ratings found — run train_rating first.")
    lapse = load_lapse(args.dir)
    attempts = load_attempts_for_users(args.sql, set(surveys["user_id"].unique()))
    med_min = load_user_median_solve_minutes(args.sqlite)

    # Each user's 80%-clear rating (pooled across boards) + their log10 median solve time.
    print(f"\nEstimating each user's {int(CALIBRATION_P * 100)}%-clear rating "
          f"(pooled, 4PL, λ={lapse:.4f}) ...")
    user_R, user_logT = {}, {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < MIN_ATTEMPTS:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=MIN_ATTEMPTS,
                               slopes=merged_slopes, floors=merged_floors, lapse=lapse)
        if R is not None and n >= MIN_ATTEMPTS:
            user_R[pref] = R
            if pref in med_min and med_min[pref] > 0:
                user_logT[pref] = float(np.log10(med_min[pref]))
    timed = [user_logT[p] for p in user_R if p in user_logT]
    center = float(np.mean(timed)) if timed else 0.0   # the time dock's zero point
    print(f"  {len(user_R)} users with a usable rating; {len(timed)} with a time feature "
          f"(center = log10 median min {center:+.3f}, {10**center*60:.0f}s)")

    # Declarations: (server, rating, declared scalar, de-meaned time feature). Users
    # without a usable time feature get T=0 (no time term).
    decls = []
    for _, row in surveys.iterrows():
        pref = str(row["user_id"])[:8]
        g = row.get("rank_scalar")
        if pref not in user_R or g is None or pd.isna(g):
            continue
        T = (user_logT[pref] - center) if pref in user_logT else 0.0
        decls.append((row["server"], user_R[pref], float(g), T))
    n_egf = sum(1 for d in decls if d[0] == "EGF")
    print(f"\nConsensus fit over {len(decls)} declarations "
          f"[{n_egf} EGF, now a coupling like any server; cubic shape prior "
          f"λ_shape={LAMBDA_SHAPE}, server λ={SERVER_LAMBDA}, gauge weight={GAUGE_WEIGHT}]")

    a, alpha, ys, egf_thr_ranks, server_fits = fit_consensus(decls)
    print(f"  learned time slope α = {alpha:+.4f} rating per de-meaned log10-min")

    # In-sample EGF calibration check (EGF is now forward-mapped through its coupling).
    egf = [(R, g, T) for srv, R, g, T in decls if srv == "EGF"]
    if "EGF" in server_fits and len(egf) >= 2 and len(server_fits["EGF"]) >= 2:
        gs = sorted(server_fits["EGF"])
        our = np.array([server_fits["EGF"][k] for k in gs], float)
        srv = np.array([k - 0.5 for k in gs], float)
        order = np.argsort(our)
        R = np.array([r for r, _, _ in egf]); gdec = np.array([g for _, g, _ in egf])
        T = np.array([t for _, _, t in egf])
        x_int = interp_extrap_vec(R + alpha * T, a, ys)
        pred = interp_extrap_vec(x_int, our[order], srv[order])
        res = pred - gdec
        print(f"  EGF-coupling calibration: RMSE {np.sqrt(np.mean(res ** 2)):.2f} | "
              f"bias {res.mean():+.2f} ranks")

    # --- write rating_to_rank.json (both board keys identical) + time_adjust ---
    curve = {"ratings": [float(r) for r in a],
             "rank_indices": [float((g - 0.5) + RANK_INDEX_OFFSET) for g in egf_thr_ranks]}
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
    print(f"\nconsensus range mapped: {scalar_to_rank(egf_thr_ranks[0])}→rating "
          f"{a[0]:.2f} ... {scalar_to_rank(egf_thr_ranks[-1])}→rating {a[-1]:.2f}")
    print(f"wrote {args.dir}/rating_to_rank.json (shared 9≡13 + time_adjust) + rank_map.json "
          f"({len(fitted)} server couplings incl EGF: {', '.join(fitted)})\n"
          f"  → publish with: python -m pipeline.package_for_pwa")


if __name__ == "__main__":
    main()
