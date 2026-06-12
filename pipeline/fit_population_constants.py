"""Fit the PWA's population-level model constants from the local D1 log and write
them to ratings_v2/population_constants.json, which package_for_pwa copies into
the served pwa/tsumegos/. The client fetches it at boot and uses these in place
of the hardcoded defaults in game-logic.js (time/review bias) and model.js
(retention constants), so the constants track the real user population instead of
drifting from a one-off manual fit.

Three groups of constants:

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
   implies. Validated in auxiliary/validate_time_adjusted_rating.py: adjusting
   rating by the user's median solving time raises the per-server-weighted
   Spearman vs survey-declared ranks (0.84 → 0.87 cross-fitted). We ship
   alpha_per_log10_min (rating units per log10-minute of median solve time,
   ≤ 0) and pop_mean_log10_min (the population centering point, so the average
   user's adjusted rank equals their stable rank). Alpha is refit from the
   survey cohort only when a cross-fitted check confirms it still helps;
   otherwise the validated default ships. The client computes
   rating' = rating + alpha·(log10(median_minutes) − pop_mean) and maps it
   through M.

3. RETENTION constants (model.js: KAPPA, Z0, GROWTH, LN_C_PRIOR) for the
   FSRS-style stability map  S = exp(LN_C + KAPPA·(z − Z0) + GROWTH·r),
   R(t) = exp(−t_days / S),  where z = logit(P_cold-solve) is the 3PL cold-solve
   log-odds. Fit by retention NLL over clean post-cutoff repeat chains (repeats
   were only logged from 2026-05-15; a chain whose first attempt predates that has
   unknown hidden reviews, so only post-cutoff chains give trustworthy gaps/r).
   Z0 = logit(0.7) is a fixed reference (LN_C is the base stability at p=0.7).

All constants are board-INDEPENDENT (the cold-solve scale is shared across sizes),
so a single JSON serves both boards.

Needs the backend ratings_v2/<size>/ 3PL params (tsumego_ratings_final.csv,
tsumego_quality_final.csv, tsumego_floor_final.csv — written by train_rating_v2)
and attempts/database.sqlite. It reads the backend directly, NOT the served
pwa/tsumegos/ shards: package_for_pwa is the only bridge into pwa/, so nothing
upstream of it depends on the dist artifacts. No GPU, not part of the in-order
pipeline.

Run:
  ./.venv/bin/python3 -m pipeline.fetch_database --rebuild-sqlite   # refresh first
  ./.venv/bin/python3 -m pipeline.fit_population_constants          # reads ratings_v2/
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
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import rank_to_scalar  # noqa: E402
from utils.fit_rank_curves import fit_user_rating  # noqa: E402

# --- Constants matching the client (pwa/js/model.js, rank-decomposition.js) ---
CALIBRATION_P = 0.8
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
    # Time-adjusted rank (validated 2026-06: weighted Spearman 0.84 → 0.87
    # cross-fitted; see auxiliary/validate_time_adjusted_rating.py).
    "time_adjust": {"alpha_per_log10_min": -0.41, "pop_mean_log10_min": -0.63},
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


def _read_id_value(path, value_col, keep):
    """{tid: float(value_col)} for tids in `keep` from a 2-col `tsumego_id,<value>`
    CSV. Missing file → empty (the caller treats absent params as unresolvable)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        id_col = reader.fieldnames[0]
        for row in reader:
            tid = row[id_col]
            if tid in keep:
                try:
                    out[tid] = float(row[value_col])
                except (TypeError, ValueError):
                    pass
    return out


def load_problem_params(ratings_dir, needed):
    """{(size, tid): (rating, slope, floor)} from the backend ratings_v2/<size>/
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
        ratings = _read_id_value(os.path.join(size_dir, "tsumego_ratings_final.csv"), "rating", tids)
        slopes = _read_id_value(os.path.join(size_dir, "tsumego_quality_final.csv"), "quality", tids)
        floors = _read_id_value(os.path.join(size_dir, "tsumego_floor_final.csv"), "floor", tids)
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
# pipeline.train_rating_v2 LEARNS it jointly with the rating model (a single global ceiling on
# the success curve) and writes it to ratings_v2/rating_constants.json; main() reads it through
# into population_constants.json so the JS-bundle contract is unchanged (see read_rating_lapse).


def read_rating_lapse(ratings_dir):
    """The global ceiling λ produced by the rating model (train_rating_v2 →
    ratings_v2/rating_constants.json: {"fat_finger_rate": λ}). Returns float or None."""
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


def fit_time_adjust(by_user, params, surveys):
    """Returns (alpha_or_None, pop_mean_or_None, info-string). alpha is refit
    (grid sweep on the survey cohort, clamped ≤0) only when the cross-fitted
    improvement over alpha=0 is positive; pop_mean is the centering point over
    ALL users with a time feature (the average user's rank doesn't move)."""
    feat = {}
    for user, urows in by_user.items():
        f = user_median_log10_minutes(urows)
        if f is not None:
            feat[user] = f
    pop_mean = (float(np.mean(list(feat.values())))
                if len(feat) >= MIN_BIAS_USERS else None)
    if pop_mean is None:
        return None, None, f"only {len(feat)} users with a time feature (<{MIN_BIAS_USERS})"

    raw_ratings = {k: v[0] for k, v in params.items()}
    slopes = {k: v[1] for k, v in params.items()}
    rating_of = {}
    for user in {u for u, _, _ in surveys}:
        if user not in feat:
            continue
        urows = [r for r in by_user.get(user, ())
                 if not r["is_repeat"] and (r["size"], r["problem"]) in params]
        if len(urows) < TA_MIN_ATTEMPTS:
            continue
        att = [((r["size"], r["problem"]), r["success"]) for r in urows]
        R, _ = fit_user_rating(att, raw_ratings, slopes=slopes,
                               min_n=TA_MIN_ATTEMPTS)
        if R is not None:
            rating_of[user] = R

    pairs = [(u, server, rating_of[u], sc, feat[u] - pop_mean)
             for u, server, sc in surveys if u in rating_of]
    if len(pairs) < TA_MIN_PAIRS:
        return None, pop_mean, f"only {len(pairs)} survey pairs (<{TA_MIN_PAIRS})"

    base = _weighted_spearman(pairs, 0.0)
    sweep = [_weighted_spearman(pairs, a) for a in TA_ALPHA_GRID]
    a_best = float(TA_ALPHA_GRID[int(np.nanargmax(sweep))])

    # Cross-fitted guard: alpha picked on train folds only, applied to held-out
    # users; ship the refit alpha only if the assembled score beats alpha=0.
    users_list = sorted(rating_of)
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
    info = (f"{len(pairs)} pairs; weighted Spearman base={base:.4f} "
            f"best={max(sweep):.4f} @ alpha={a_best:+.3f}; "
            f"cross-fitted {cv_mean:.4f}")
    if cv_mean <= base:
        return None, pop_mean, info + " — no held-out gain, keeping default alpha"
    # Never adjust in the implausible direction (slow ⇒ rated higher).
    return min(a_best, 0.0), pop_mean, info


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
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings_v2"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "ratings_v2", "population_constants.json"))
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

    # --- Fat-finger (lapse) rate: PRODUCED by the rating model (train_rating_v2), read through ---
    lam = read_rating_lapse(args.ratings_dir)
    if lam is not None:
        out["fat_finger_rate"] = round(lam, 4)
        print(f"Fat-finger (lapse) rate from the rating model: {out['fat_finger_rate']} "
              f"(ratings_v2/rating_constants.json — train_rating_v2 learns it jointly)")
    else:
        print(f"No rating_constants.json (run train_rating_v2 first); "
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

    # --- Time-adjusted rank constants ---
    print("Fitting time-adjusted-rank constants (survey-validated) ...")
    out["time_adjust"] = dict(DEFAULTS["time_adjust"])
    ta_alpha, ta_pop_mean, ta_info = fit_time_adjust(by_user, params,
                                                     load_surveys(args.db))
    print(f"  {ta_info}")
    if ta_pop_mean is not None:
        out["time_adjust"]["pop_mean_log10_min"] = round(ta_pop_mean, 4)
    if ta_alpha is not None:
        out["time_adjust"]["alpha_per_log10_min"] = round(ta_alpha, 4)
    print(f"  shipped alpha={out['time_adjust']['alpha_per_log10_min']:+.3f} "
          f"rating/log10-min, pop_mean={out['time_adjust']['pop_mean_log10_min']:+.3f} "
          f"log10-min{' (default alpha)' if ta_alpha is None else ''}")

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
