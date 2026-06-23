"""Fit the PWA's population-level model constants from the local D1 log and write
them to ratings/population_constants.json, which package_for_pwa copies into
the served pwa/tsumegos/. The client fetches it at boot and uses these in place
of the hardcoded defaults in game-logic.js (time/review bias) and model.js
(retention constants), so the constants track the real user population instead of
drifting from a one-off manual fit.

Four groups of constants:

1. Overlay time-benefit BIAS (logit per minute). The rushing / insufficient-review
   overlays gate on a per-user solving/review-time coefficient that is biased
   NEGATIVE by residual-difficulty confounding (a problem harder-than-its-rating
   for a user takes longer AND fails more). We estimate the population-mean
   coefficient — the additive bias the client recenters by — under the SAME 3PL
   difficulty control the cold-solve model now uses, in the client's
   logit-per-minute feature space (percentile time → central per-minute slope).
   Solving time carries the bias; review time empirically does not (~0).

2. TIME-ADJUSTED-RANK constants (stats-ui.js, the Rank tab's "time adjusted
   rank"). Practice here is untimed while server ranks reflect timed play, so a
   user who leans on long thinking time clears more than their server rank
   implies. The time feature is RANK-RESIDUALIZED: each user is centered on the
   solving time EXPECTED FOR THEIR RANK (a linear log10(median-min)=g0+g1·rank
   line fit over the active population), NOT a single global mean — so only time
   unexplained by rank moves the rank (a player merely as slow as is typical for
   their level gets no dock; this is the strong-but-deliberate case the old global
   centering over-penalized). We ship alpha_per_log10_min (rating units per
   log10-minute, ≤ 0) plus time_vs_rank_intercept / time_vs_rank_slope (the g0/g1
   line, log10-min per rank index). Alpha is refit from the survey cohort only
   when a cross-fitted check confirms it still helps; otherwise the validated
   default ships. The client reproduces the SAME centering (it has the user's rank
   index): rating' = rating + alpha·(log10(median_minutes) − (g0 + g1·rank_index))
   then maps through M — fit and application share the identical feature, so the
   alpha (the rank-controlled partial time effect) is valid as applied. (A global
   center applied with this partial alpha would be a mismatch; measured no better
   than today and worse-principled — the gain lives entirely in the centering.)

3. RETENTION constants (model.js: KAPPA, Z0, GROWTH, LN_C_PRIOR) for the
   FSRS-style stability map  S = exp(LN_C + KAPPA·(z − Z0) + GROWTH·r),
   R(t) = exp(−t_days / S),  where z = logit(P_cold-solve) is the 3PL cold-solve
   log-odds. Fit by retention NLL over clean post-cutoff repeat chains (repeats
   were only logged from 2026-05-15; a chain whose first attempt predates that has
   unknown hidden reviews, so only post-cutoff chains give trustworthy gaps/r).
   Z0 = logit(0.7) is a fixed reference (LN_C is the base stability at p=0.7).

4. WEAKNESS_RIDGE_LAMBDA (weakness.js): the L2 precision on each per-embedding-axis
   beta of the client's weakness logistic. Fit by the SAME held-out-CV recipe the
   model is regularized under: per-user 60d/30d chunks, 5-fold CV over a lambda grid
   (embedding betas penalized, intercept + problemRank free), shipping the
   per-attempt-NLL-minimizing lambda. The embedding axes are joint across boards.

All constants are board-INDEPENDENT (the cold-solve scale and the embedding axes
are shared across sizes), so a single JSON serves both boards.

Needs the backend ratings/<size>/ 3PL params (tsumego_ratings_final.csv,
tsumego_quality_final.csv, tsumego_floor_final.csv — written by train_rating)
ratings/rating_to_rank.json (M, for the weakness ridge's problemRank feature),
embeddings/joint/per_problem.csv (the weakness ridge's correct-move embeddings),
and attempts/database.sqlite. It reads the backend directly, NOT the served
pwa/tsumegos/ shards: package_for_pwa is the only bridge into pwa/, so nothing
upstream of it depends on the dist artifacts. No GPU, not part of the in-order
pipeline.

Run:
  ./.venv/bin/python3 -m pipeline.fetch_database --rebuild-sqlite   # refresh first
  ./.venv/bin/python3 -m pipeline.fit_population_constants          # reads ratings/
  ./.venv/bin/python3 -m pipeline.package_for_pwa                   # publishes the JSON
"""

import argparse
import bisect
import csv
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rank_surveys import rank_to_scalar  # noqa: E402
from utils.fit_rank_curves import CALIBRATION_P, fit_user_rating, interp_extrap_vec  # noqa: E402
from utils.ratings_io import load_rating_rank_curve, load_size_value_maps  # noqa: E402

# --- Constants matching the client (pwa/js/model.js, rank-decomposition.js) ---
# CALIBRATION_P (the 80%-clear margin) is the single source in utils.fit_rank_curves.
LOGIT_CALIBRATION = np.log(CALIBRATION_P / (1 - CALIBRATION_P))
FLOOR_MAX = 0.78
Z0 = np.log(0.7 / 0.3)                 # fixed retention reference (= logit(0.7))
REPEAT_LOG_CUTOFF = "2026-05-15"       # repeats logged only from here (clean chains)
AFK_CAP_S = 180.0                      # drop AFK solving times before the bias fit

# Current hardcoded defaults — the fallback when data is too sparse to refit.
DEFAULTS = {
    "time_bias_per_min": -0.91,
    "review_bias_per_min": 0.0,
    "fat_finger_rate": 0.01,           # 4PL upper-asymptote slip rate (lapse)
    "retention": {"kappa": 4.22, "z0": float(Z0),
                  "growth": float(np.log(1.54)), "ln_c_prior": 1.55},
    # Time-adjusted rank (rank-RESIDUALIZED: center each user on the time expected
    # for their rank via the log10(median-min)=g0+g1·rank_index line, not a global
    # mean). Fallback values (= the validated 2026-06-14 fit); main() overwrites with
    # the fresh fit when it validates.
    "time_adjust": {"alpha_per_log10_min": -0.45, "time_vs_rank_intercept": -0.7719,
                    "time_vs_rank_slope": 0.005826},
    # L2 precision on each per-embedding-axis beta of the client's weakness logistic
    # (pwa/js/weakness.js RIDGE_LAMBDA). Fallback when the population is too sparse
    # to refit; main() overwrites with the held-out-CV fit when it lands.
    "weakness_ridge_lambda": 200.0,
}
MIN_RETENTION_EVENTS = 400             # below this, keep the retention defaults
MIN_BIAS_USERS = 50                    # below this, keep the bias defaults

# --- Time-adjusted rank fit knobs ---
TA_MIN_TIMED = 20          # timed novel attempts needed for a user's time feature
TA_MIN_PAIRS = 50          # (user, server) survey pairs needed to refit alpha
TA_MIN_SERVER_USERS = 5    # per-server minimum for a Spearman term
TA_MIN_ATTEMPTS = 10       # novel rated attempts needed for a survey user's rating
TA_ALPHA_GRID = np.linspace(-1.5, 0.5, 81)   # rating units per log10-minute
TA_CV_FOLDS = 5
TA_CV_SEEDS = 10

# --- Weakness-model ridge prior (pwa/js/weakness.js RIDGE_LAMBDA) ---
# L2 precision on the per-embedding-axis betas of the client's weakness logistic,
# fit by the SAME held-out-CV recipe the model is regularized under: per-user
# 60d/30d chunks, 5-fold CV over a lambda grid, ship the per-attempt-NLL-minimizing
# lambda. Board-independent (9x9+13x13 pooled), the embedding axes being joint.
WEAKNESS_EMBED_CSV = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")
WEAKNESS_K_EMBED = 64
WEAKNESS_TOTAL_RANKS = 29.0          # problemRank = M(rating) / TOTAL_RANKS (matches the client)
WEAKNESS_WINDOW_DAYS = 60
WEAKNESS_STRIDE_DAYS = 30
WEAKNESS_MIN_ATTEMPTS_PER_CHUNK = 100
WEAKNESS_CV_FOLDS = 5
WEAKNESS_LAMBDA_GRID = [30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]
WEAKNESS_SEED = 0
MIN_WEAKNESS_CHUNKS = 20             # below this, keep the default ridge lambda


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def clear_kappa(c):
    c = np.clip(c, 0.0, FLOOR_MAX)
    q = np.clip((CALIBRATION_P - c) / (1 - c), 1e-6, 1 - 1e-6)
    out = np.log(q / (1 - q))
    return np.where(np.asarray(c) > 0, out, LOGIT_CALIBRATION)


def clear_kappa_scalar(c):
    c = min(max(c, 0.0), FLOOR_MAX)
    if c <= 0:
        return float(LOGIT_CALIBRATION)
    q = min(max((CALIBRATION_P - c) / (1 - c), 1e-6), 1 - 1e-6)
    return float(np.log(q / (1 - q)))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rows(db_path):
    """All attempts, ordered by user/problem/time so chains are easy to walk."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT user_id, problem, success, size, time_used, review_time, "
        "is_repeat, created_at FROM tsumego_reports ORDER BY user_id, problem, created_at"
    ).fetchall()
    con.close()
    return rows


def load_problem_params(ratings_dir, needed):
    """{(size, tid): (rating, slope, floor)} from the backend ratings/<size>/
    CSVs — the SAME source package_for_pwa ships into the shard metadata
    (rating ← tsumego_ratings_final, slope ← tsumego_quality_final's `quality`,
    floor ← tsumego_floor_final). Reading the backend directly keeps the pipeline
    from depending on the served pwa/tsumegos/ artifacts (package_for_pwa is the
    sole bridge into pwa/)."""
    params = {}
    by_size = defaultdict(set)
    for (size, tid) in needed:
        by_size[size].add(tid)
    for size, tids in by_size.items():
        size_dir = os.path.join(ratings_dir, str(size))
        if not os.path.isdir(size_dir):
            print(f"  WARNING: {size_dir} missing; skipping size {size}")
            continue
        ratings, slopes, floors = load_size_value_maps(
            ratings_dir, size, keep=tids, missing_ok=True)
        for tid in tids:
            r, s = ratings.get(tid), slopes.get(tid)
            if s is None or r is None or not np.isfinite(s) or s <= 0 or not np.isfinite(r):
                continue
            params[(size, tid)] = (float(r), float(s),
                                   float(np.clip(floors.get(tid, 0.0), 0, FLOOR_MAX)))
    return params


# ---------------------------------------------------------------------------
# Per-user ability (3PL MLE over novel attempts) — the cold-solve z source.
# ---------------------------------------------------------------------------
def fit_ability(rating, slope, floor, y, iters=50):
    """1-D Newton for ability θ: P = c + (1−c)σ(a·(θ−rating)+κ(c))."""
    kappa = clear_kappa(floor)
    one_mc = 1 - floor
    theta = 0.0
    for _ in range(iters):
        praw = sigmoid(slope * (theta - rating) + kappa)
        P = np.clip(floor + one_mc * praw, 1e-9, 1 - 1e-9)
        g = one_mc * slope * praw * (1 - praw)          # dP/dθ
        score = np.sum((y - P) * g / (P * (1 - P))) - 0.04 * theta   # weak prior
        info = np.sum(g * g / (P * (1 - P))) + 0.04
        step = score / info
        step = max(-2.0, min(2.0, step))
        theta += step
        theta = max(-20.0, min(20.0, theta))
        if abs(step) < 1e-7:
            break
    return float(theta)


def cold_solve_logit(theta, rating, slope, floor):
    """z = logit(P_cold-solve) for the 3PL — what the retention head consumes."""
    praw = sigmoid(slope * (theta - rating) + clear_kappa_scalar(floor))
    P = min(max(floor + (1 - floor) * praw, 1e-9), 1 - 1e-9)
    return float(np.log(P / (1 - P)))


# Fat-finger (lapse) rate λ — the 4PL upper-asymptote slip floor — is no longer fit here.
# pipeline.train_rating LEARNS it jointly with the rating model (a single global ceiling on
# the success curve) and writes it to ratings/rating_constants.json; main() reads it through
# into population_constants.json so the JS-bundle contract is unchanged (see read_rating_lapse).


def read_rating_lapse(ratings_dir):
    """The global ceiling λ produced by the rating model (train_rating →
    ratings/rating_constants.json: {"fat_finger_rate": λ}). Returns float or None."""
    path = os.path.join(ratings_dir, "rating_constants.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return float(json.load(f).get("fat_finger_rate"))
    except (ValueError, TypeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Time-benefit bias (client logit/min units, under the 3PL difficulty control)
# ---------------------------------------------------------------------------
def midrank_pct(sorted_ms, t):
    n = len(sorted_ms)
    lo = bisect.bisect_left(sorted_ms, t)
    hi = bisect.bisect_right(sorted_ms, t)
    return (lo + 0.5 * (hi - lo)) / n


def fit_time_coeff_3pl(rating, slope, floor, pct, y, iters=60):
    """Per-user (ability, time_coeff) under the 3PL with time on the inner logit:
    inner = a·(ability−rating) + κ(floor) + time_coeff·pct. Returns (coeff, se)
    in logit-per-percentile (caller scales to per-minute)."""
    kappa = clear_kappa(floor)
    one_mc = 1 - floor
    w = np.array([0.0, 0.0])
    X = np.column_stack([slope, pct])
    ridge = np.diag([1e-4, 1e-4])
    for _ in range(iters):
        eta = slope * w[0] + pct * w[1] + kappa - slope * rating
        praw = sigmoid(eta)
        P = np.clip(floor + one_mc * praw, 1e-9, 1 - 1e-9)
        g = one_mc * praw * (1 - praw)
        v = g * g / (P * (1 - P))
        s = (y - P) * g / (P * (1 - P))
        H = (X * v[:, None]).T @ X + ridge
        step = np.linalg.solve(H, X.T @ s - np.array([1e-4, 1e-4]) * w)
        step = np.clip(step, -3, 3)
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    cov = np.linalg.inv((X * v[:, None]).T @ X + ridge)
    return float(w[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


def population_time_bias(by_user_rows, params, review=False, min_n=20):
    """Inverse-variance pooled population-mean time/review coefficient (logit/min)
    = the additive bias the client recenters by."""
    coefs, ses = [], []
    for user, rows in by_user_rows.items():
        if review:
            ev = [(r["review_time"], r["success"], params[(r["size"], r["problem"])])
                  for r in rows
                  if r["is_repeat"] and r["review_time"] and r["review_time"] > 0
                  and r["review_time"] < 3600000 and (r["size"], r["problem"]) in params]
        else:
            ev = [(r["time_used"], r["success"], params[(r["size"], r["problem"])])
                  for r in rows
                  if not r["is_repeat"] and r["time_used"] and 0 < r["time_used"] <= AFK_CAP_S * 1000
                  and (r["size"], r["problem"]) in params]
        if len(ev) < min_n:
            continue
        ms = np.array([e[0] for e in ev], float)
        y = np.array([1 if e[1] else 0 for e in ev], float)
        rating = np.array([e[2][0] for e in ev])
        slope = np.array([e[2][1] for e in ev])
        floor = np.clip(np.array([e[2][2] for e in ev]), 0, FLOOR_MAX)
        if y.std() == 0:
            continue
        sm = np.sort(ms)
        n = len(sm)
        tppm = (sm[int(np.ceil(0.55 * (n - 1)))] - sm[int(np.floor(0.45 * (n - 1)))]) / (60000 * 0.10)
        if not (tppm > 0):
            continue
        pct = np.array([midrank_pct(sm, t) for t in ms])
        if pct.std() < 1e-6:
            continue
        c, se = fit_time_coeff_3pl(rating, slope, floor, pct, y)
        if se > 0 and np.isfinite(se):
            coefs.append(c / tppm)
            ses.append(se / tppm)
    if len(coefs) < MIN_BIAS_USERS:
        return None, len(coefs)
    coefs = np.array(coefs)
    w = 1.0 / np.array(ses) ** 2
    return float(np.sum(w * coefs) / np.sum(w)), len(coefs)


# ---------------------------------------------------------------------------
# Time-adjusted rank (alpha per log10-minute of median solve time + centering)
# ---------------------------------------------------------------------------
def load_surveys(db_path):
    """Latest parseable declaration per (user, server): [(user, server, scalar)].
    Empty when the dump carries no rank_surveys table."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT user_id, server, rank, created_at FROM rank_surveys "
            "ORDER BY created_at").fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    latest = {}
    for user, server, rank, _ in rows:
        sc = rank_to_scalar(rank)
        if sc is not None:
            latest[(user, server)] = sc
    return [(u, s, sc) for (u, s), sc in latest.items()]


def load_rating_to_rank(ratings_dir):
    """rating→rank-index interpolator M from ratings/rating_to_rank.json (one
    shared curve; the size-9 key stands in for either board). None if absent/bad —
    the caller then keeps the default time_adjust block."""
    try:
        xs, ys = load_rating_rank_curve(ratings_dir)
    except (ValueError, OSError, KeyError, StopIteration):
        return None
    if xs.size < 2 or ys.size != xs.size:
        return None
    return lambda r: float(interp_extrap_vec(np.asarray([r], float), xs, ys)[0])


def user_median_log10_minutes(urows):
    """log10(median solving minutes) over novel timed attempts (AFK-capped),
    or None below TA_MIN_TIMED — the client computes the same from timeTaken."""
    ts = [r["time_used"] for r in urows
          if not r["is_repeat"] and r["time_used"]
          and 0 < r["time_used"] <= AFK_CAP_S * 1000]
    if len(ts) < TA_MIN_TIMED:
        return None
    return float(np.log10(np.median(ts) / 60000.0))


def _weighted_spearman(pairs, alpha):
    """pairs: [(user, server, rating, rank_scalar, centered_feature)]. Per-server
    Spearman(rating + alpha·feature, declared rank), user-count-weighted mean."""
    by_server = defaultdict(list)
    for _, server, rating, sc, f in pairs:
        by_server[server].append((rating + alpha * f, sc))
    acc, w = 0.0, 0.0
    for rows_ in by_server.values():
        if len(rows_) < TA_MIN_SERVER_USERS:
            continue
        r = np.array([x for x, _ in rows_])
        s = np.array([y for _, y in rows_])
        if np.ptp(r) == 0 or np.ptp(s) == 0:
            continue
        rho = spearmanr(r, s).statistic
        if np.isfinite(rho):
            acc += rho * len(rows_)
            w += len(rows_)
    return acc / w if w else np.nan


def fit_time_adjust(by_user, params, surveys, apply_m, survey_recenter=True):
    """Returns (alpha, time_vs_rank_intercept, time_vs_rank_slope, info); any numeric
    is None when too sparse to fit (caller keeps the default block).

    The time feature is RANK-RESIDUALIZED: instead of centering every user on one
    population-mean solving time, we center on the time EXPECTED FOR THEIR RANK — a
    linear log10(median-min)=g0+g1·rank_index line fit over the active population —
    so only solving time UNEXPLAINED by rank drives the adjustment. The client ships
    g0/g1 and the user's own rank index, reproducing the identical feature, so the
    fitted alpha (the rank-controlled partial time effect, rating units per log10-min)
    is valid as applied. alpha is still refit by the cross-validated survey Spearman
    sweep, clamped ≤0; g1 (the per-rank slope) comes from the OLS time-vs-rank line.

    The intercept g0 is, by default, RE-ANCHORED to the survey cohort (survey_recenter):
    the OLS line is fleet-fit, so its mean residual over survey users is non-zero, which
    leaks a uniform downward rank bias into the dock (measured ~0.25 rank — mean mapped
    error our−declared −0.29→−0.54 with the fleet g0; verified with
    check_mapped_vs_declared_rank --time-adjust). Shifting g0 by that mean residual zeroes
    the dock's mean over the calibration cohort while leaving alpha unchanged (weighted
    Spearman is invariant to a constant feature shift, so the fit below is unaffected) and
    g1 untouched. --no-survey-recenter keeps the raw fleet intercept."""
    raw_ratings = {k: v[0] for k, v in params.items()}
    slopes = {k: v[1] for k, v in params.items()}

    # Time feature + our 80%-clear rating/rank for every user that has both. The raw
    # rating feeds the survey Spearman term (so alpha lands in rating units); the rank
    # index feeds the residualization line. Computed over the FULL population (not just
    # survey users) so the line is a stable population fit.
    feat, rating_of, rank_of = {}, {}, {}
    for user, urows in by_user.items():
        f = user_median_log10_minutes(urows)
        if f is None:
            continue
        feat[user] = f
        if apply_m is None:
            continue
        rated = [r for r in urows
                 if not r["is_repeat"] and (r["size"], r["problem"]) in params]
        if len(rated) < TA_MIN_ATTEMPTS:
            continue
        att = [((r["size"], r["problem"]), r["success"]) for r in rated]
        R, _ = fit_user_rating(att, raw_ratings, slopes=slopes, min_n=TA_MIN_ATTEMPTS)
        if R is not None:
            rating_of[user] = R
            rank_of[user] = apply_m(R)

    pop = [u for u in feat if u in rank_of]
    if len(pop) < MIN_BIAS_USERS:
        return None, None, None, f"only {len(pop)} users with time+rank (<{MIN_BIAS_USERS})"

    # Time-vs-rank line: log10(median min) = g0 + g1·rank_index (OLS over the pop).
    rk = np.array([rank_of[u] for u in pop])
    tt = np.array([feat[u] for u in pop])
    g1, g0 = (float(x) for x in np.polyfit(rk, tt, 1))     # slope, intercept
    resid = {u: feat[u] - (g0 + g1 * rank_of[u]) for u in pop}

    # Survey pairs: rank-residualized time feature; rating term is the raw rating.
    pairs = [(u, server, rating_of[u], sc, resid[u])
             for u, server, sc in surveys if u in resid]
    if len(pairs) < TA_MIN_PAIRS:
        return None, g0, g1, f"only {len(pairs)} survey pairs (<{TA_MIN_PAIRS})"

    base = _weighted_spearman(pairs, 0.0)
    sweep = [_weighted_spearman(pairs, a) for a in TA_ALPHA_GRID]
    a_best = float(TA_ALPHA_GRID[int(np.nanargmax(sweep))])

    # Re-anchor the intercept to the survey cohort (default; see the docstring).
    # One residual per UNIQUE survey user (resid[u] is constant across that user's
    # declarations); adding its mean to g0 demeans the dock over the cohort.
    surv_mbar = float(np.mean(list({p[0]: p[4] for p in pairs}.values())))
    g0_out = g0 + surv_mbar if survey_recenter else g0

    # Cross-fitted guard: alpha picked on train folds only, applied to held-out
    # users; ship the refit alpha only if the assembled score beats alpha=0. (The
    # g0/g1 line is a population nuisance fit on the full pop, not cross-validated.)
    users_list = sorted({p[0] for p in pairs})
    cv = []
    for seed in range(TA_CV_SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(users_list))
        fold = {users_list[j]: i % TA_CV_FOLDS for i, j in enumerate(order)}
        a_of_fold = []
        for k in range(TA_CV_FOLDS):
            train = [p for p in pairs if fold[p[0]] != k]
            sw = [_weighted_spearman(train, a) for a in TA_ALPHA_GRID]
            a_of_fold.append(float(TA_ALPHA_GRID[int(np.nanargmax(sw))]))
        adj = [(u, server, rating + a_of_fold[fold[u]] * f, sc, 0.0)
               for u, server, rating, sc, f in pairs]
        cv.append(_weighted_spearman(adj, 0.0))
    cv_mean = float(np.mean(cv))
    recenter_note = (f"survey-recenter ON (mean resid {surv_mbar:+.4f} → g0 "
                     f"{g0:+.3f}→{g0_out:+.3f})" if survey_recenter
                     else "survey-recenter OFF (fleet g0)")
    info = (f"{len(pairs)} pairs, {len(pop)}-user time-vs-rank line "
            f"(g0={g0:+.3f}, g1={g1:+.4f}); {recenter_note}; weighted Spearman "
            f"base={base:.4f} best={max(sweep):.4f} @ alpha={a_best:+.3f}; "
            f"cross-fitted {cv_mean:.4f}")
    if cv_mean <= base:
        return None, g0_out, g1, info + " — no held-out gain, keeping default alpha"
    # Never adjust in the implausible direction (slow-for-rank ⇒ rated higher).
    return min(a_best, 0.0), g0_out, g1, info


# ---------------------------------------------------------------------------
# Retention constants (KAPPA, GROWTH, LN_C_PRIOR), Z0 fixed.
# ---------------------------------------------------------------------------
def build_retention_events(rows, params, abilities):
    """One event per (prev → cur) consecutive pair in a clean post-cutoff chain:
    y = cur.success, gap_days, r = prev.successSinceFailure, z = cold-solve logit.
    Mirrors model.js scheduling (S from the previous attempt's features)."""
    events = []
    by_up = defaultdict(list)
    for r in rows:
        by_up[(r["user_id"], r["problem"], r["size"])].append(r)
    for (user, problem, size), chain in by_up.items():
        if (size, problem) not in params or user not in abilities:
            continue
        chain.sort(key=lambda r: r["created_at"])
        if chain[0]["created_at"][:10] < REPEAT_LOG_CUTOFF:
            continue   # unknown hidden reviews before the cutoff → corrupt gaps/r
        z = cold_solve_logit(abilities[user], *params[(size, problem)])
        ssf = 0          # successSinceFailure, walked over the chain
        prev_ssf = 0     # the previous attempt's successSinceFailure (r for its gap)
        has_failed = False
        prev = None
        for a in chain:
            if prev is not None and a["is_repeat"]:
                t_prev = datetime.fromisoformat(prev["created_at"].replace("Z", "+00:00"))
                t_cur = datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
                gap_days = (t_cur - t_prev).total_seconds() / 86400.0
                if gap_days > 0:
                    events.append((1 if a["success"] else 0, gap_days, prev_ssf, z))
            # update successSinceFailure AFTER using the previous value
            if not a["success"]:
                has_failed = True
                ssf = 0
            elif has_failed:
                ssf += 1
            prev_ssf = ssf
            prev = a
    return events


def fit_retention(events):
    """Fit KAPPA, GROWTH, LN_C (pooled) by retention NLL. R = exp(−gap/S),
    S = exp(LN_C + KAPPA·(z−Z0) + GROWTH·r)."""
    y = np.array([e[0] for e in events], float)
    gap = np.array([e[1] for e in events], float)
    r = np.array([e[2] for e in events], float)
    z = np.array([e[3] for e in events], float)

    def nll(p):
        kappa, growth, ln_c = p
        lnS = np.clip(ln_c + kappa * (z - Z0) + growth * r, -5, 8)
        S = np.exp(lnS)
        R = np.clip(np.exp(-gap / S), 1e-9, 1 - 1e-9)
        return -np.sum(y * np.log(R) + (1 - y) * np.log(1 - R))

    best = minimize(nll, x0=[DEFAULTS["retention"]["kappa"],
                             DEFAULTS["retention"]["growth"],
                             DEFAULTS["retention"]["ln_c_prior"]],
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 5000})
    kappa, growth, ln_c = best.x
    # Guard against degenerate fits; keep sign/magnitude sane.
    if not best.success or not (0 < kappa < 20) or not (-2 < growth < 4):
        return None
    return {"kappa": float(kappa), "z0": float(Z0),
            "growth": float(growth), "ln_c_prior": float(ln_c)}


# ---------------------------------------------------------------------------
# Weakness-model ridge prior (embedding-axis L2, held-out CV).
# ---------------------------------------------------------------------------
def load_problem_embeddings(needed):
    """{(size, tid): np.ndarray(K)} correct-move joint-ICA embedding from
    embeddings/joint/per_problem.csv, restricted to the attempted problems.
    Empty if the CSV is absent — the caller then keeps the default lambda."""
    if not os.path.exists(WEAKNESS_EMBED_CSV):
        return {}
    cols = [f"embed_{i}" for i in range(WEAKNESS_K_EMBED)]
    out = {}
    with open(WEAKNESS_EMBED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (int(row["size"]), row["tsumego_id"])
            except (TypeError, ValueError, KeyError):
                continue
            if key not in needed:
                continue
            try:
                vec = np.array([float(row[c]) for c in cols], float)
            except (TypeError, ValueError, KeyError):
                continue
            if vec.shape[0] == WEAKNESS_K_EMBED and np.all(np.isfinite(vec)):
                out[key] = vec
    return out


def _ridge_logistic(X, y, lam_vec, beta0=None):
    """L-BFGS fit of an L2-penalized logistic regression. lam_vec is the per-
    coefficient precision (0 = unpenalized). Returns (beta, success)."""
    def nll_grad(beta):
        z = X @ beta
        nll = -np.sum(y * (-np.logaddexp(0.0, -z)) + (1 - y) * (-np.logaddexp(0.0, z)))
        grad = X.T @ (sigmoid(z) - y) + lam_vec * beta
        return nll + 0.5 * float(np.sum(lam_vec * beta * beta)), grad
    if beta0 is None:
        beta0 = np.zeros(X.shape[1])
    res = minimize(nll_grad, beta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7})
    return res.x, res.success


def _weakness_chunks(by_user, rank_feat, embeddings):
    """Yield (rank_vec, embed_matrix, y) for each per-user 60d/30d window holding
    >= WEAKNESS_MIN_ATTEMPTS_PER_CHUNK novel rated+embedded attempts (mirrors the
    chunking the ridge was historically calibrated under)."""
    window = timedelta(days=WEAKNESS_WINDOW_DAYS)
    stride = timedelta(days=WEAKNESS_STRIDE_DAYS)
    for urows in by_user.values():
        ev = []
        for r in urows:
            if r["is_repeat"]:
                continue
            key = (r["size"], r["problem"])
            if key not in rank_feat or key not in embeddings:
                continue
            ev.append((datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                       1.0 if r["success"] else 0.0, rank_feat[key], embeddings[key]))
        if len(ev) < WEAKNESS_MIN_ATTEMPTS_PER_CHUNK:
            continue
        ev.sort(key=lambda e: e[0])
        ts = [e[0] for e in ev]
        start, end_all = ts[0], ts[-1]
        while start <= end_all:
            end = start + window
            idx = [i for i, t in enumerate(ts) if start <= t < end]
            if len(idx) >= WEAKNESS_MIN_ATTEMPTS_PER_CHUNK:
                yield (np.array([ev[i][2] for i in idx], float),
                       np.vstack([ev[i][3] for i in idx]),
                       np.array([ev[i][1] for i in idx], float))
            start += stride


def fit_weakness_ridge(by_user, params, embeddings, apply_m):
    """CV-best L2 precision for the weakness model's per-embedding-axis betas.
    Per chunk a 5-fold CV scores a grid of ridge strengths (embedding betas
    penalized; intercept + problemRank free) and the per-attempt-NLL-minimizing
    lambda summed across chunks is returned. (lambda, info); lambda is None when
    inputs are missing or the population is too sparse (caller keeps the default)."""
    if apply_m is None:
        return None, "no rating_to_rank.json (run map_rating_to_egf first) — keeping default"
    if not embeddings:
        rel = os.path.relpath(WEAKNESS_EMBED_CSV, PROJECT_ROOT)
        return None, f"no embeddings at {rel} (run build_embedding) — keeping default"
    rank_feat = {k: apply_m(v[0]) / WEAKNESS_TOTAL_RANKS for k, v in params.items()}
    rng = np.random.default_rng(WEAKNESS_SEED)
    total_nll = {lam: 0.0 for lam in WEAKNESS_LAMBDA_GRID}
    total_n = {lam: 0 for lam in WEAKNESS_LAMBDA_GRID}
    n_chunks = 0
    for rank_vec, emb, y in _weakness_chunks(by_user, rank_feat, embeddings):
        if y.std() < 1e-3:
            continue
        n_chunks += 1
        n = len(y)
        X = np.column_stack([np.ones(n), rank_vec, emb])   # [1, problemRank, embed_0..K-1]
        d = X.shape[1]
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, WEAKNESS_CV_FOLDS)
        beta_ws = {k: np.zeros(d) for k in range(WEAKNESS_CV_FOLDS)}
        for lam in sorted(WEAKNESS_LAMBDA_GRID, reverse=True):   # warm-start as lambda loosens
            lam_vec = np.zeros(d)
            lam_vec[2:] = lam                                    # penalize embedding betas only
            for k in range(WEAKNESS_CV_FOLDS):
                te = folds[k]
                if len(te) == 0:
                    continue
                tr = np.ones(n, dtype=bool)
                tr[te] = False
                if y[tr].std() < 1e-3:
                    continue
                beta, ok = _ridge_logistic(X[tr], y[tr], lam_vec, beta0=beta_ws[k])
                if not ok or not np.all(np.isfinite(beta)):
                    continue
                beta_ws[k] = beta
                z = X[te] @ beta
                total_nll[lam] += -float(np.sum(
                    y[te] * (-np.logaddexp(0.0, -z)) + (1 - y[te]) * (-np.logaddexp(0.0, z))))
                total_n[lam] += len(te)
    if n_chunks < MIN_WEAKNESS_CHUNKS:
        return None, f"only {n_chunks} chunks (<{MIN_WEAKNESS_CHUNKS}) — keeping default"
    per = {lam: total_nll[lam] / max(1, total_n[lam]) for lam in WEAKNESS_LAMBDA_GRID}
    best = min(WEAKNESS_LAMBDA_GRID, key=lambda lam: per[lam])
    grid = "  ".join(f"{lam:g}:{per[lam]:.5f}" for lam in WEAKNESS_LAMBDA_GRID)
    return float(best), f"{n_chunks} chunks; per-attempt NLL [{grid}] → lambda={best:g}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "ratings", "population_constants.json"))
    ap.add_argument("--no-survey-recenter", dest="survey_recenter", action="store_false",
                    help="Anchor the time-dock intercept g0 on the FLEET instead of the "
                         "survey cohort. Default re-anchors g0 to the survey cohort's mean "
                         "residual so the dock is mean-unbiased on the calibration set "
                         "(alpha + slope g1 unchanged). See the time-adjust note in "
                         "auxiliary/check_mapped_vs_declared_rank.py.")
    ap.set_defaults(survey_recenter=True)
    args = ap.parse_args()

    print(f"Loading attempts from {args.db} ...")
    rows = load_rows(args.db)
    print(f"  {len(rows)} rows")
    needed = {(r["size"], r["problem"]) for r in rows}
    print(f"Loading 3PL params for {len(needed)} distinct problems ...")
    params = load_problem_params(args.ratings_dir, needed)
    print(f"  resolved {len(params)}")

    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)

    # --- Per-user ability (novel attempts) for the cold-solve z ---
    abilities = {}
    for user, urows in by_user.items():
        ev = [(params[(r["size"], r["problem"])], r["success"]) for r in urows
              if not r["is_repeat"] and (r["size"], r["problem"]) in params]
        if len(ev) < 10:
            continue
        rating = np.array([e[0][0] for e in ev])
        slope = np.array([e[0][1] for e in ev])
        floor = np.clip(np.array([e[0][2] for e in ev]), 0, FLOOR_MAX)
        y = np.array([1 if e[1] else 0 for e in ev], float)
        abilities[user] = fit_ability(rating, slope, floor, y)
    print(f"Estimated ability for {len(abilities)} users (≥10 novel attempts)")

    out = dict(DEFAULTS)
    out["retention"] = dict(DEFAULTS["retention"])

    # --- Fat-finger (lapse) rate: PRODUCED by the rating model (train_rating), read through ---
    lam = read_rating_lapse(args.ratings_dir)
    if lam is not None:
        out["fat_finger_rate"] = round(lam, 4)
        print(f"Fat-finger (lapse) rate from the rating model: {out['fat_finger_rate']} "
              f"(ratings/rating_constants.json — train_rating learns it jointly)")
    else:
        print(f"No rating_constants.json (run train_rating first); "
              f"keeping default fat_finger_rate {out['fat_finger_rate']}")

    # --- Time / review bias ---
    print("Fitting time-benefit bias (3PL-controlled, logit/min) ...")
    tb, n_tb = population_time_bias(by_user, params, review=False)
    rb, n_rb = population_time_bias(by_user, params, review=True)
    # The bias correction only ever counteracts NEGATIVE residual-difficulty
    # confounding; a positive population mean is most likely genuine benefit
    # (notably review time, where slower review plausibly helps), so clamp to ≤0 —
    # never recenter in the direction that would suppress a real positive effect.
    if tb is not None:
        out["time_bias_per_min"] = round(min(tb, 0.0), 4)
        print(f"  solving-time bias = {tb:+.3f} logit/min  ({n_tb} users) "
              f"→ shipped {out['time_bias_per_min']:+.3f}")
    else:
        print(f"  solving-time: only {n_tb} usable users (<{MIN_BIAS_USERS}); keeping default")
    if rb is not None:
        out["review_bias_per_min"] = round(min(rb, 0.0), 4)
        print(f"  review-time bias  = {rb:+.3f} logit/min  ({n_rb} users) "
              f"→ shipped {out['review_bias_per_min']:+.3f} (clamped ≤0)")
    else:
        print(f"  review-time: only {n_rb} usable users (<{MIN_BIAS_USERS}); keeping default")

    # --- Time-adjusted rank constants (survey-validated, rank-residualized) ---
    print("Fitting time-adjusted-rank constants (survey-validated, rank-residualized) ...")
    apply_m = load_rating_to_rank(args.ratings_dir)
    if apply_m is None:
        print("  WARNING: ratings/rating_to_rank.json missing (run map_rating_to_egf "
              "first) — keeping default time_adjust block")
    ta_alpha, ta_g0, ta_g1, ta_info = fit_time_adjust(
        by_user, params, load_surveys(args.db), apply_m, args.survey_recenter)
    print(f"  {ta_info}")
    if ta_alpha is not None and ta_g0 is not None and ta_g1 is not None:
        out["time_adjust"] = {
            "alpha_per_log10_min": round(ta_alpha, 4),
            "time_vs_rank_intercept": round(ta_g0, 4),
            "time_vs_rank_slope": round(ta_g1, 6),
        }
    else:
        out["time_adjust"] = dict(DEFAULTS["time_adjust"])
    ta = out["time_adjust"]
    print(f"  shipped alpha={ta['alpha_per_log10_min']:+.3f} rating/log10-min, "
          f"time-vs-rank g0={ta['time_vs_rank_intercept']:+.3f} "
          f"g1={ta['time_vs_rank_slope']:+.5f} log10-min/rank"
          f"{' (defaults)' if ta_alpha is None else ''}")

    # --- Weakness-model ridge prior (embedding-axis L2; held-out CV) ---
    print("Fitting weakness-model ridge prior (embedding-axis L2, held-out CV) ...")
    embeddings = load_problem_embeddings(needed)
    wl, w_info = fit_weakness_ridge(by_user, params, embeddings, apply_m)
    if wl is not None:
        out["weakness_ridge_lambda"] = float(wl)
    print(f"  {w_info}")
    print(f"  shipped weakness_ridge_lambda = {out['weakness_ridge_lambda']:g}"
          f"{' (default)' if wl is None else ''}")

    # --- Retention constants ---
    print("Building retention events from clean post-cutoff chains ...")
    events = build_retention_events(rows, params, abilities)
    print(f"  {len(events)} retention events")
    if len(events) >= MIN_RETENTION_EVENTS:
        fit = fit_retention(events)
        if fit:
            out["retention"] = {k: round(v, 4) for k, v in fit.items()}
            print(f"  KAPPA={fit['kappa']:.3f}  GROWTH={fit['growth']:.3f}  "
                  f"LN_C_PRIOR={fit['ln_c_prior']:.3f}  (Z0={fit['z0']:.3f} fixed)")
        else:
            print("  retention fit degenerate; keeping defaults")
    else:
        print(f"  <{MIN_RETENTION_EVENTS} events; keeping retention defaults")

    out["fit_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out["n_retention_events"] = len(events)
    payload = json.dumps({k: out[k] for k in sorted(out) if k != "version"},
                         sort_keys=True)
    out["version"] = hashlib.sha256(payload.encode()).hexdigest()[:12]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(out, indent=2))
    print("\nRun `package_for_pwa` to publish it into pwa/tsumegos/.")


if __name__ == "__main__":
    main()
