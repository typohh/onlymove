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

2. TIME-ADJUSTED-RANK constants (stats-ui.js / rank-translation.js). NOT fit here —
   READ THROUGH from the rating→rank mapping. pipeline.map_rating_to_rank fits the
   solve-time slope α JOINTLY with the consensus curve + couplings (the time feature
   is per-user, so it is identified) and writes {alpha_per_log10_min, center_log10_min}
   into rating_to_rank.json; this step just surfaces it into population_constants.json
   so the client contract (a `time_adjust` field) is unchanged. A SIMPLE log-time slope
   is all the survey data supports — no rating-interaction, no power-law, and no
   rank-residualization line (the old g0/g1 — the joint couplings absorb the
   rank-correlated part of solve time on their own; see docs/egf_anchor_bias_and_time.md).
   The client docks every SERVER translation by alpha·(log10(median_minutes) − center)
   and leaves the onlymove headline untimed.

3. RETENTION constants (model.js: KAPPA, Z0, GROWTH, LN_C_PRIOR) for the
   FSRS-style stability map  S = exp(LN_C + KAPPA·(z − Z0) + GROWTH·r),
   R(t) = exp(−t_days / S),  where z = logit(P_cold-solve) is the 3PL cold-solve
   log-odds. Fit by retention NLL over clean post-cutoff repeat chains (repeats
   were only logged from 2026-05-15; a chain whose first attempt predates that has
   unknown hidden reviews, so only post-cutoff chains give trustworthy gaps/r).
   Z0 = logit(0.7) is a fixed reference (LN_C is the base stability at p=0.7).

4. WEAKNESS_AXIS_LAMBDA (weakness.js): a per-axis prior precision vector (length K) —
   the L2 precision on each per-embedding-axis beta of the client's weakness logistic.
   Fit by empirical Bayes: the between-user variance tau_k^2 of each axis's weakness
   coefficient is a population variance component, estimated by EM with a Laplace
   E-step — per-user MAP ridge fits under the current per-axis prior, then
   tau_k^2 <- mean_u(beta_uk^2 + posterior var_uk). Axes along which users genuinely
   differ get a loose prior (their coefficients can surface in the Weaknesses pane);
   axes with no between-user signal stay pinned. The offline design has ONE
   coefficient per axis (the correct-move embedding — the identified SUM direction
   beta_correct + beta_picked of the client's augmented model), so the shipped
   per-coefficient lambda_k = 2 / tau_k^2 makes the client's implied prior on that
   sum exactly N(0, tau_k^2). This vector is MANDATORY — there is no scalar fallback:
   if the population is too sparse to fit it, fit_population_constants RAISES rather
   than shipping a fallback, and the client requires it.

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
  ./.venv/bin/python3 -m pipeline.fetch_database          # refresh attempts/database.sqlite
  ./.venv/bin/python3 -m pipeline.fit_population_constants # reads ratings/
  ./.venv/bin/python3 -m pipeline.package_for_pwa          # publishes the JSON
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.fit_rank_curves import CALIBRATION_P, interp_extrap_vec  # noqa: E402
from utils.ratings_io import load_rating_rank_curve, load_size_value_maps  # noqa: E402

# --- Constants matching the client (pwa/js/model.js, rank-decomposition.js) ---
# CALIBRATION_P (the 80%-clear margin) is the single source in utils.fit_rank_curves.
LOGIT_CALIBRATION = np.log(CALIBRATION_P / (1 - CALIBRATION_P))
FLOOR_MAX = 0.78
Z0 = np.log(0.7 / 0.3)                 # fixed retention reference (= logit(0.7))
REPEAT_LOG_CUTOFF = "2026-05-15"       # repeats logged only from here (clean chains)
AFK_CAP_S = 180.0                      # drop AFK solving times before the bias fit

# Current hardcoded defaults. NOT a silent fallback: by default the run CRASHES
# rather than resorting to any of these (see _resort_to_default + main). They are
# used only when --allow-defaults is passed, or as the minimize() init below.
DEFAULTS = {
    "time_bias_per_min": -0.91,
    "review_bias_per_min": 0.0,
    "fat_finger_rate": 0.01,           # 4PL upper-asymptote slip rate (lapse)
    "retention": {"kappa": 4.22, "z0": float(Z0),
                  "growth": float(np.log(1.54)), "ln_c_prior": 1.55,
                  "first_success_step": 0.0},
    # Time-adjusted rank: a single log-time slope + center. PRODUCED by the rating→rank
    # mapping (pipeline.map_rating_to_rank fits α jointly with the curve + couplings) and
    # READ THROUGH here into population_constants.json — not fit here. Fallback only.
    "time_adjust": {"alpha_per_log10_min": -0.16, "center_log10_min": -0.64},
    # NB: no weakness prior default. weakness_axis_lambda is fit here and is
    # MANDATORY — if it can't be fit the run raises rather than shipping a
    # fabricated fallback (there is no scalar weakness_ridge_lambda anymore).
}
MIN_RETENTION_EVENTS = 400             # below this, keep the retention defaults
MIN_RECALL_HOURS = 6.0                 # pwa/js/model.js stability floor, in hours
MIN_BIAS_USERS = 50                    # below this, keep the bias defaults

# --- Weakness-model per-axis prior (pwa/js/weakness.js) — shared inputs ---
# The per-axis L2 precision on the client's weakness betas is fit by empirical
# Bayes (group 4 / fit_weakness_axis_priors). These are the shared inputs.
# Board-independent (9x9+13x13 pooled), the embedding axes being joint.
WEAKNESS_EMBED_CSV = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")


def _weakness_k_embed():
    """Axis count K, from the per_problem.csv header — the axis set grows with
    data (pipeline.build_axes selects K incrementally), so it is never fixed."""
    with open(WEAKNESS_EMBED_CSV, newline="") as f:
        header = csv.DictReader(f).fieldnames or []
    return sum(1 for c in header if c.startswith("embed_"))

# --- Per-axis weakness prior (weakness_axis_lambda; empirical Bayes, see group 5) ---
WEAKNESS_EB_MAX_N = 500              # per-user window, mirrors the client's MODEL_INPUT_HISTORY_LENGTH
WEAKNESS_EB_MIN_N = 100              # attempts a user needs to contribute
WEAKNESS_EB_MIN_USERS = 30           # below this, the fit raises (no fallback — see main)
WEAKNESS_EB_MAX_ITERS = 40
WEAKNESS_EB_TOL = 1e-3               # max relative tau^2 change to declare convergence
# Bounds on the SHIPPED client per-coefficient precision lambda_k = 2/tau_k^2:
# the floor stops one outlier-driven axis from going near-unpenalized; the cap
# keeps a zero-signal axis's precision finite (the client still trains on it).
WEAKNESS_EB_LAMBDA_MIN = 20.0
WEAKNESS_EB_LAMBDA_MAX = 5000.0
# Halt if an axis's pooled per-user coefficient is this far from zero, measured as
# mean^2 / second-moment — the share of what ships as between-user variance that is
# really population displacement. Scale-free, and the only form that cannot be moved
# by rescaling an axis (see the zero-mean check). NOT a significance test: with
# hundreds of users a negligible mean is many standard errors from zero, so a p-value
# would halt on nothing. This bar corresponds to a pooled mean a third of the
# between-user spread, which displaces level one's credibility test by about a fifth
# of its bar — borderline material, and the side to err on.
WEAKNESS_MEAN_SHARE_MAX = 0.10


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
# Time-adjusted rank — READ THROUGH from the mapping (not fit here).
# ---------------------------------------------------------------------------
def read_time_adjust(ratings_dir):
    """{alpha_per_log10_min, center_log10_min} from ratings/rating_to_rank.json's
    `time_adjust` block — the solve-time dock pipeline.map_rating_to_rank fits JOINTLY
    with the rating→rank curve. Read through into population_constants.json so the
    client contract (a `time_adjust` field) is unchanged; the dock is just produced
    upstream now (mirrors how the lapse λ is read through from rating_constants.json).
    Returns the dict or None (caller keeps the default)."""
    path = os.path.join(ratings_dir, "rating_to_rank.json")
    if not os.path.exists(path):
        return None
    try:
        ta = json.load(open(path)).get("time_adjust")
        if ta and np.isfinite(ta.get("alpha_per_log10_min")) \
                and np.isfinite(ta.get("center_log10_min")):
            return {"alpha_per_log10_min": round(float(ta["alpha_per_log10_min"]), 6),
                    "center_log10_min": round(float(ta["center_log10_min"]), 6)}
    except (ValueError, TypeError, KeyError, OSError):
        pass
    return None


def load_rating_to_rank(ratings_dir):
    """rating→rank interpolator M from ratings/rating_to_rank.json (one shared
    curve; the size-9 key stands in for either board), for the weakness ridge's
    problemRank feature. None if absent/bad."""
    try:
        xs, ys = load_rating_rank_curve(ratings_dir)
    except (ValueError, OSError, KeyError, StopIteration):
        return None
    if xs.size < 2 or ys.size != xs.size:
        return None
    return lambda r: float(interp_extrap_vec(np.asarray([r], float), xs, ys)[0])


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
                    events.append((1 if a["success"] else 0, gap_days, prev_ssf, z, user))
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


MIN_STEP_EVENTS_PER_CELL = 5           # per user, each of the two streak cells
MIN_STEP_USERS = 20                    # below this the step keeps its default
STEP_MAX = 2.0                         # sanity rail on the fitted step


def fit_first_success_step(events):
    """The extra log-stability a problem earns from its FIRST success after a
    lapse, over and above the per-success growth already in the ladder.

    The streak counter is zero exactly when the previous attempt FAILED, and one
    when it was the first success after a failure — so rung 0 and rung 1 are the
    post-lapse and post-success states, not two steps of one ladder. Established
    schedulers give a lapse its own formula; this model does not, and the
    shortfall shows up as reviews at rung 1 and above clearing well above the
    0.80 the window opens at, i.e. served before the intended forgetting.

    Fit by MOMENT MATCHING, deliberately not by retention likelihood. The
    likelihood is evaluated on intervals the scheduler itself chose, so it is
    close to uninformative about what a different schedule would do — measured,
    it sizes this step at about a fifth of what the realized rates require.
    The moment condition needs no fitting at all: under R = exp(-t/S), scaling
    an interval by f scales log-retention by f, so the multiplier taking a cell
    from its realized rate p to a target q is ln(q)/ln(p), and the step is its
    logarithm. The target is rung 0's own realized rate — making the ladder
    self-consistent, which is what the client can actually be held to, rather
    than any particular point inside the window.

    The rung contrast is taken WITHIN USER (Mantel-Haenszel, one stratum per
    user). How promptly a user returns once a window opens is a habit that sets
    where in the window their serves land; it is unobservable and stable per
    user, so comparing each user only against themselves removes it.
    """
    by_user = defaultdict(lambda: {0: [0, 0], 1: [0, 0]})
    for y, gap_days, r, _z, user in events:
        if gap_days * 24 < MIN_RECALL_HOURS:
            continue        # under the stability floor: not a schedulable serve
        cell = by_user[user][0 if r == 0 else 1]
        cell[0] += y
        cell[1] += 1

    num = den = 0.0
    k0 = n0 = 0
    strata = 0
    for _user, c in by_user.items():
        if c[0][1] < MIN_STEP_EVENTS_PER_CELL or c[1][1] < MIN_STEP_EVENTS_PER_CELL:
            continue
        A, B = float(c[1][0]), float(c[1][1] - c[1][0])      # rung >= 1
        C, D = float(c[0][0]), float(c[0][1] - c[0][0])      # rung 0
        n = A + B + C + D
        num += A * D / n
        den += B * C / n
        k0 += c[0][0]
        n0 += c[0][1]
        strata += 1
    if strata < MIN_STEP_USERS or den <= 0 or num <= 0 or n0 == 0:
        return None, f"{strata} users with >= {MIN_STEP_EVENTS_PER_CELL} events per cell"

    odds_ratio = num / den
    p0 = k0 / n0                                   # post-lapse realized rate
    if not (0 < p0 < 1):
        return None, "degenerate rung-0 rate"
    odds1 = (p0 / (1 - p0)) * odds_ratio
    p1 = odds1 / (1 + odds1)                       # post-success rate, within user
    if not (0 < p1 < 1) or p1 <= p0:
        return None, f"post-success rate {p1:.3f} not above post-lapse {p0:.3f}"
    step = math.log(math.log(p0) / math.log(p1))   # ln of the interval multiplier
    if not (0 <= step < STEP_MAX):
        return None, f"step {step:.3f} outside [0, {STEP_MAX})"
    return step, (f"OR {odds_ratio:.2f} over {strata} users; post-lapse {p0:.3f} "
                  f"-> post-success {p1:.3f}; interval x{math.exp(step):.2f}")


# ---------------------------------------------------------------------------
# Weakness-model per-axis prior (empirical Bayes).
# ---------------------------------------------------------------------------
def load_problem_embeddings(needed):
    """{(size, tid): np.ndarray(K)} correct-move joint-ICA embedding from
    embeddings/joint/per_problem.csv, restricted to the attempted problems.
    Empty if the CSV is absent — the per-axis fit then raises (no fallback)."""
    if not os.path.exists(WEAKNESS_EMBED_CSV):
        return {}
    k_embed = _weakness_k_embed()
    cols = [f"embed_{i}" for i in range(k_embed)]
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
            if vec.shape[0] == k_embed and np.all(np.isfinite(vec)):
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


def _weakness_user_datasets(by_user, rank_feat, embeddings):
    """One (X, y) design per user for the empirical-Bayes variance fit: the most
    recent WEAKNESS_EB_MAX_N novel rated+embedded attempts (mirroring the client's
    MODEL_INPUT_HISTORY_LENGTH training window), users below WEAKNESS_EB_MIN_N or
    with a degenerate outcome dropped. X = [1, problemRank, embed_0..K-1]."""
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
        if len(ev) < WEAKNESS_EB_MIN_N:
            continue
        ev.sort(key=lambda e: e[0])
        ev = ev[-WEAKNESS_EB_MAX_N:]
        y = np.array([e[1] for e in ev], float)
        if y.std() < 1e-3:
            continue
        X = np.column_stack([np.ones(len(ev)),
                             np.array([e[2] for e in ev], float),
                             np.vstack([e[3] for e in ev])])
        yield X, y


def fit_weakness_axis_priors(by_user, params, embeddings, apply_m,
                             allow_axis_mean=False):
    """Per-axis prior precision vector for the client's weakness betas (group 5 in
    the module docstring): empirical Bayes on the between-user variance component
    tau_k^2 of each axis's weakness coefficient, by EM with a Laplace E-step —
    per-user MAP ridge fits under the current per-axis prior, then
    tau_k^2 <- mean_u(beta_uk^2 + posterior var_uk). The offline design carries one
    coefficient per axis (the identified SUM direction of the client's augmented
    model), so the shipped client per-coefficient lambda_k = 2/tau_k^2 puts a
    N(0, tau_k^2) prior on the client's beta_correct[k] + beta_picked[k].
    Returns (list of K lambdas, info) or (None, info) when data is too sparse."""
    if apply_m is None:
        return None, "no rating_to_rank.json — no per-axis prior"
    if not embeddings:
        return None, "no embeddings — no per-axis prior"
    rank_feat = {k: apply_m(v[0]) for k, v in params.items()}
    data = list(_weakness_user_datasets(by_user, rank_feat, embeddings))
    if len(data) < WEAKNESS_EB_MIN_USERS:
        return None, (f"only {len(data)} users with ≥{WEAKNESS_EB_MIN_N} usable "
                      f"attempts (<{WEAKNESS_EB_MIN_USERS}) — no per-axis prior")
    K = _weakness_k_embed()
    d = 2 + K
    # tau^2 bounds in client-lambda terms (lambda_k = 2/tau_k^2).
    tau2_lo, tau2_hi = 2.0 / WEAKNESS_EB_LAMBDA_MAX, 2.0 / WEAKNESS_EB_LAMBDA_MIN
    # EM seed only (the iteration converges to the data-driven variance): the
    # geometric mean of the bounds, so no scalar prior is needed to start.
    tau2 = np.full(K, float(np.sqrt(tau2_lo * tau2_hi)))
    betas = [np.zeros(d) for _ in data]   # warm starts across EM iterations
    it = 0
    for it in range(WEAKNESS_EB_MAX_ITERS):
        lam_vec = np.zeros(d)
        lam_vec[2:] = 1.0 / tau2          # offline prior: one coefficient per axis
        second = np.zeros(K)              # Σ_u (beta_uk^2 + var_uk)
        first = np.zeros(K)               # Σ_u beta_uk — see the zero-mean check below
        n_ok = 0
        for i, (X, y) in enumerate(data):
            beta, ok = _ridge_logistic(X, y, lam_vec, beta0=betas[i])
            if not ok or not np.all(np.isfinite(beta)):
                continue
            betas[i] = beta
            p = sigmoid(X @ beta)
            w = np.clip(p * (1 - p), 1e-6, None)
            H = (X * w[:, None]).T @ X + np.diag(np.maximum(lam_vec, 1e-6))
            var = np.diag(np.linalg.inv(H))[2:]
            second += beta[2:] ** 2 + np.maximum(var, 0.0)
            first += beta[2:]
            n_ok += 1
        if n_ok < WEAKNESS_EB_MIN_USERS:
            return None, f"only {n_ok} converged per-user fits — no per-axis prior"
        new_tau2 = np.clip(second / n_ok, tau2_lo, tau2_hi)
        rel = float(np.max(np.abs(new_tau2 - tau2) / np.maximum(tau2, 1e-12)))
        tau2 = new_tau2
        if rel < WEAKNESS_EB_TOL:
            break
    lam_axis = 2.0 / tau2

    # ZERO-MEAN CHECK. The M-step above is the right one for the prior actually
    # shipped, N(0, tau^2): it accumulates a second moment about ZERO, not a
    # variance about the pooled mean. That is only the between-user variance if
    # the pooled per-user coefficient really is zero on every axis, which
    # nothing forces -- the client's coefficients are fit per user against the
    # shipped loadings, and a population-level effect that the rank term cannot
    # absorb lands in them. Where the mean is not zero the consequences run the
    # same way on both sides: tau^2 absorbs mean^2 and overstates the spread, so
    # the prior is looser and shrinks less than intended, AND the client shrinks
    # toward a point that is not the population centre while level-one tests
    # credibility against that same wrong point -- turning "everyone finds this
    # axis hard" into a per-user finding. Reported per axis as a share of the
    # second moment, which is the decision-relevant form: it says how much of
    # what ships as variance is really displacement.
    mu = first / n_ok
    raw2 = second / n_ok
    share = mu ** 2 / np.maximum(raw2, 1e-12)
    print("  weakness prior zero-mean check (shipped prior is centred at 0):")
    for k in range(K):
        print(f"    axis {k}: pooled mean {mu[k]:+.4f}  sd {np.sqrt(max(raw2[k] - mu[k] ** 2, 0.0)):.4f}"
              f"  mean^2/second-moment {share[k]:6.1%}")
    print(f"    worst axis: {share.max():.1%} of the shipped second moment is "
          f"pooled mean rather than between-user spread")
    if share.max() > WEAKNESS_MEAN_SHARE_MAX and not allow_axis_mean:
        k_bad = int(np.argmax(share))
        raise RuntimeError(
            f"axis {k_bad}'s pooled per-user coefficient is {share[k_bad]:.1%} of its "
            f"second moment (mean {mu[k_bad]:+.4f} against spread "
            f"{np.sqrt(max(raw2[k_bad] - mu[k_bad] ** 2, 0.0)):.4f}), over the "
            f"{WEAKNESS_MEAN_SHARE_MAX:.0%} bar.\n"
            "A coefficient that is non-zero for EVERYONE is not a weakness — it is "
            "residual difficulty, which build_axes quarantines its population "
            "residual head to keep OUT of the axes. So read this as that quarantine "
            "leaking, and look there first: re-centring the prior here would make the "
            "numbers self-consistent while leaving an axis carrying difficulty that "
            "belongs to the rating model, and the panel would go on reporting it as a "
            "property of people.\n"
            "If the mean is genuinely about users rather than problems, the fix is to "
            "SHIP it (a per-axis mean beside the precision, the client's prior centred "
            "at half of it per slot, and a decision on whether level one starts testing "
            "against the mean) — not to widen this bar. --allow-axis-mean ships the "
            "zero-centred prior anyway, deliberately.")

    info = (f"{len(data)} users, {it + 1} EM iters; per-coefficient lambda "
            f"min/median/max = {lam_axis.min():.0f}/{float(np.median(lam_axis)):.0f}/"
            f"{lam_axis.max():.0f}; zero-mean check worst {share.max():.1%}")
    return [round(float(l), 2) for l in lam_axis], info


# ---------------------------------------------------------------------------
def _resort_to_default(field, reason, allow):
    """Called wherever the fit would otherwise fall back to the DEFAULTS value for
    `field`. Crashes by default so a sparse/missing input is never silently shipped
    as a fabricated constant; --allow-defaults downgrades it to a warning that keeps
    the DEFAULTS value already seeded into `out`."""
    if not allow:
        raise RuntimeError(
            f"would resort to the hardcoded default for '{field}' ({reason}). "
            f"Refusing by default — fix the inputs and re-run, or pass "
            f"--allow-defaults to ship the DEFAULTS value deliberately.")
    print(f"  [allow-defaults] keeping default {field} ({reason})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "ratings", "population_constants.json"))
    ap.add_argument("--allow-defaults", action="store_true",
                    help="ship the hardcoded DEFAULTS for any constant that can't be fit "
                         "(sparse/missing inputs) instead of crashing")
    ap.add_argument("--allow-axis-mean", action="store_true",
                    help="ship the zero-centred weakness prior even if an axis's pooled "
                         "per-user coefficient breaches the zero-mean bar (normally a "
                         "sign that population difficulty has leaked into that axis)")
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
        _resort_to_default("fat_finger_rate",
                           "no rating_constants.json (run train_rating first)",
                           args.allow_defaults)

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
        _resort_to_default("time_bias_per_min",
                           f"only {n_tb} usable users (<{MIN_BIAS_USERS})",
                           args.allow_defaults)
    if rb is not None:
        out["review_bias_per_min"] = round(min(rb, 0.0), 4)
        print(f"  review-time bias  = {rb:+.3f} logit/min  ({n_rb} users) "
              f"→ shipped {out['review_bias_per_min']:+.3f} (clamped ≤0)")
    else:
        _resort_to_default("review_bias_per_min",
                           f"only {n_rb} usable users (<{MIN_BIAS_USERS})",
                           args.allow_defaults)

    # --- Time-adjusted rank: READ THROUGH from the mapping (not fit here) ---
    # pipeline.map_rating_to_rank fits the log-time slope α jointly with the rating→rank
    # curve and writes {alpha_per_log10_min, center_log10_min} into rating_to_rank.json;
    # we just surface it here so the client's population_constants.json contract is kept.
    ta = read_time_adjust(args.ratings_dir)
    if ta is not None:
        out["time_adjust"] = ta
        print(f"Time-adjust dock (read from rating_to_rank.json): "
              f"alpha={ta['alpha_per_log10_min']:+.4f} rating/log10-min, "
              f"center={ta['center_log10_min']:+.4f} log10-min")
    else:
        _resort_to_default("time_adjust",
                           "no time_adjust in rating_to_rank.json (run map_rating_to_rank first)",
                           args.allow_defaults)
        out["time_adjust"] = dict(DEFAULTS["time_adjust"])

    # --- Per-axis weakness prior (empirical-Bayes between-user variance) ---
    # The client's ONLY weakness prior. Mandatory: no scalar fallback, so if it
    # can't be fit we RAISE rather than shipping population_constants.json without
    # it (the client can't score weaknesses without this vector — a silent gap
    # would go unnoticed forward, which is exactly what we refuse to do here).
    apply_m = load_rating_to_rank(args.ratings_dir)   # for the weakness prior's problemRank
    print("Fitting per-axis weakness prior (empirical-Bayes between-user variance) ...")
    embeddings = load_problem_embeddings(needed)
    wa, wa_info = fit_weakness_axis_priors(by_user, params, embeddings, apply_m,
                                          args.allow_axis_mean)
    if wa is None:
        raise RuntimeError(
            f"could not fit weakness_axis_lambda ({wa_info}). It is mandatory and has "
            f"no fallback — fix the inputs (run map_rating_to_rank + build_axes, "
            f"or wait for more attempts) and re-run. Not shipping without it.")
    out["weakness_axis_lambda"] = wa
    print(f"  {wa_info}")

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
            step, step_info = fit_first_success_step(events)
            if step is None:
                _resort_to_default("retention.first_success_step", step_info,
                                   args.allow_defaults)
                out["retention"]["first_success_step"] = \
                    DEFAULTS["retention"]["first_success_step"]
            else:
                out["retention"]["first_success_step"] = round(step, 4)
                print(f"  FIRST_SUCCESS_STEP={step:.3f}  ({step_info})")
        else:
            _resort_to_default("retention", "retention fit degenerate",
                               args.allow_defaults)
    else:
        _resort_to_default("retention",
                           f"<{MIN_RETENTION_EVENTS} retention events",
                           args.allow_defaults)

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
