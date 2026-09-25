"""Fit the PWA's population-level model constants from the local D1 log and write
them to ratings/population_constants.json, which package_for_pwa copies into
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

2. TIME-ADJUSTED-RANK constants (stats-ui.js / rank-translation.js). NOT fit here —
   READ THROUGH from the rating→rank mapping. pipeline.map_rating_to_rank fits the
   solve-time slope α JOINTLY with the consensus curve + couplings (the time feature
   is per-user, so it is identified) and writes {alpha_per_log10_min, center_log10_min}
   into rating_to_rank.json; this step just surfaces it into population_constants.json
   so the client contract (a `time_adjust` field) is unchanged. A SIMPLE log-time slope
   is all the survey data supports — no rating-interaction, no power-law, and no
   rank-residualization line (the old g0/g1 — the joint couplings absorb the
   rank-correlated part of solve time on their own; see docs/methodology.md).
   The client docks every SERVER translation by alpha·(log10(median_minutes) − center)
   and leaves the onlymove headline untimed.

3. RETENTION constants (model.js: KAPPA, Z0, GROWTH, LN_C_PRIOR) for the
   FSRS-style stability map  S = exp(LN_C + KAPPA·(z − Z0) + GROWTH·r),
   R(t) = exp(−t_days / S),  where z = logit(P_cold-solve) is the 3PL cold-solve
   log-odds. Fit by retention NLL over clean post-cutoff repeat chains (repeats
   were only logged from 2026-05-15; a chain whose first attempt predates that has
   unknown hidden reviews, so only post-cutoff chains give trustworthy gaps/r).
   Z0 = logit(0.7) is a fixed reference (LN_C is the base stability at p=0.7).

All constants are board-INDEPENDENT (the cold-solve scale and the embedding axes
are shared across sizes), so a single JSON serves both boards.

Needs the backend ratings/<size>/ 3PL params (tsumego_ratings.csv,
tsumego_slopes.csv, tsumego_floors.csv — written by train_mirt)
ratings/rating_to_rank.json (M, for the rank prior's grid), and db/d1.sqlite.
It reads the backend directly, NOT the served pwa/tsumegos/ shards:
package_for_pwa is the only bridge into pwa/, so nothing upstream of it depends
on the dist artifacts. No GPU, not part of the in-order pipeline.

Run:
  ./.venv/bin/python3 -m pipeline.fetch_database          # refresh db/d1.sqlite
  ./.venv/bin/python3 -m pipeline.fit_population_constants # reads ratings/
  ./.venv/bin/python3 -m pipeline.package_for_pwa          # publishes the JSON
"""

import argparse
import bisect
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

from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec,
)
from utils.rating.ratings_io import load_rating_rank_curve, load_size_value_maps  # noqa: E402
from utils.rating import se_floor  # noqa: E402
from utils.constants import ALL_BOARD_SIZES  # noqa: E402

# --- Constants matching the client (pwa/js/model.js, rank-decomposition.js) ---
# CALIBRATION_P (the 80%-clear margin) is the single source in utils.rating.fit_rank_curves.
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
}
MIN_RETENTION_EVENTS = 400             # below this, keep the retention defaults
MIN_RECALL_HOURS = 6.0                 # pwa/js/model.js stability floor, in hours
MIN_BIAS_USERS = 50                    # below this, keep the bias defaults

# --- Population rank prior (pwa/shared/rank-window.js) ---
# The client's window estimator is a Bayesian posterior over the user's rank, and
# this is its prior: the distribution of rank across the real user population.
# Shipped as a log-density on a fixed rank grid rather than a Gaussian because the
# population is genuinely BIMODAL (a beginner mass and a stronger mass), and a
# unimodal stand-in would pull every early estimate toward a trough where few users
# actually sit.
MIN_PRIOR_ATTEMPTS = 50                # novel rated attempts to place a user in the prior
MIN_PRIOR_USERS = 200                  # below this the prior is too sparse to ship
PRIOR_GRID_STEP = 0.5                  # inlined into the JS bundle — keep it coarse
PRIOR_BANDWIDTH = 1.0                  # KDE width, in stones
# Grid half-width, in population SDs. The bounds are DERIVED, not hardcoded: they
# are this envelope around the fitted user distribution, then narrowed to what the
# corpus can actually price a floor on (utils.rating.se_floor). Both halves matter
# and a fixed pair could satisfy neither — the grid is the one place the rank
# coordinate is bounded, so too tight pins an off-scale user, while too wide hands
# package_for_pwa a knot with no measurement behind it and the run aborts.
PRIOR_GRID_SD = 4.0
# A floor under how far the derivation may narrow — not a clamp on the rank scale,
# which is unbounded by design.
#
# THIS IS WHERE THE CORPUS CAN STILL PRICE A FLOOR, which is what makes it the right
# bound rather than a chosen one. Below 20k the corpus holds almost nothing (a
# rating IS the 80%-clear point — see below), so requiring a floor there would
# demand a MEASUREMENT of an empty region. M's knots reach further down (to cover
# the survey's floor declarations), which does not change what the corpus can price.
#
# The retired value asked for the labelled ladder (clampRankBand's 30k..9d) on the
# reasoning that a legitimately ranked user must be representable. Two things are
# wrong with it. Representability does not need it: an OUTERMOST knot absorbs
# everything past it and seFloorAt clamps to the range, so a user beyond the edge
# is served the edge knot rather than falling off. And the rung it demanded is not
# a real one — a rating IS the 80%-clear point, so a problem priced at 30k is the
# claim that a 30-kyu player finds the single correct move four times in five,
# which the corpus does not contain and plausibly cannot. The old bound was
# satisfiable only while the fit emitted a pathological rating tail; a better-
# behaved fit narrows that tail and fails it, which is the wrong way round.
PRIOR_GRID_MUST_CONTAIN = (-20.0, 9.0)       # 20k .. 10d, the corpus's pricing range

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
        "is_repeat, created_at, move FROM tsumego_reports "
        "ORDER BY user_id, problem, created_at"
    ).fetchall()
    con.close()
    return rows


def load_problem_params(ratings_dir, needed):
    """{(size, tid): (rating, slope, floor)} from the backend ratings/<size>/
    CSVs — the SAME source package_for_pwa ships into the shard metadata
    (rating ← tsumego_ratings.csv, slope ← tsumego_slopes.csv, floor ←
    tsumego_floors.csv). Reading the backend directly keeps the pipeline
    from depending on the served pwa/tsumegos/ artifacts (package_for_pwa is the
    sole bridge into pwa/).

    A needed problem absent from the ratings is a retired one and is left out —
    its attempts outlive it. One that IS rated has its slope and floor by
    construction (load_size_value_maps refuses maps that disagree), and a slope
    that is not positive raises rather than dropping the problem."""
    params = {}
    by_size = defaultdict(set)
    for (size, tid) in needed:
        by_size[size].add(tid)
    for size, tids in by_size.items():
        ratings, slopes, floors = load_size_value_maps(ratings_dir, size, keep=tids)
        for tid in ratings:
            if not slopes[tid] > 0:
                raise ValueError(f"ratings/{size}: slope {slopes[tid]} for {tid} is not positive")
            params[(size, tid)] = (ratings[tid], slopes[tid],
                                   float(np.clip(floors[tid], 0, FLOOR_MAX)))
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
# pipeline.train_mirt LEARNS it jointly with the rating model (a single global ceiling on
# the success curve) and writes it to ratings/rating_constants.json; main() reads it through
# into population_constants.json so the JS-bundle contract is unchanged (see read_rating_lapse).


def read_rating_lapse(ratings_dir):
    """The global ceiling λ produced by the rating model (train_mirt →
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
# Population rank prior (the client window estimator's prior over user rank)
# ---------------------------------------------------------------------------
def fit_rank_prior(by_user, params, apply_m, lapse, ratings_dir, curves):
    """Distribution of user RANK across the population, as a KDE log-density on a
    fixed grid.

    Each user's rank is `fit_user_rating` (the 80%-clear MLE the client mirrors,
    with each problem's own slope and floor plus the global lapse) mapped through
    M — the same two steps the client performs, so the prior lives in exactly the
    coordinate the posterior is taken over.

    Two properties of this estimate the reader has to keep in mind, neither of
    which is corrected for here because any correction would be invented rather
    than measured:

      - It is SURVIVORSHIP-SELECTED. Only users who reached MIN_PRIOR_ATTEMPTS are
        placed, so the population that churns early is absent, and that population
        is likely weaker than the one that stays. The prior is therefore an
        over-estimate of a brand-new user's expected strength, which biases the
        cold-start display UP — the unsafe direction. Raising MIN_PRIOR_ATTEMPTS
        sharpens each user's placement and worsens this; lowering it does the
        reverse.
      - The spread is the OBSERVED one, so it carries per-user estimation noise on
        top of the true population variance. The threshold is set where that noise
        is small against the population sd, not zero.

    Returns (prior_dict, info_str) or (None, reason)."""
    ranks = []
    for user, urows in by_user.items():
        att = [(r["problem"], 1 if r["success"] else 0) for r in urows
               if not r["is_repeat"] and (r["size"], r["problem"]) in params]
        if len(att) < MIN_PRIOR_ATTEMPTS:
            continue
        # params is keyed by (size, tid); tsumego ids are symmetry-canonical and
        # globally unique, so a flat per-user view is lossless.
        flat = {}
        for r in urows:
            key = (r["size"], r["problem"])
            if key in params:
                flat[r["problem"]] = params[key]
        R, n = fit_user_rating(
            att, {t: v[0] for t, v in flat.items()}, target=CALIBRATION_P,
            slopes={t: v[1] for t, v in flat.items()},
            floors={t: v[2] for t, v in flat.items()},
            lapse=lapse, min_n=MIN_PRIOR_ATTEMPTS)
        if R is None or not np.isfinite(R):
            continue
        rk = apply_m(R)
        if np.isfinite(rk):
            ranks.append(rk)
    if len(ranks) < MIN_PRIOR_USERS:
        return None, f"only {len(ranks)} users with >={MIN_PRIOR_ATTEMPTS} rated novel attempts"

    r = np.asarray(ranks, float)

    # Bounds: the population envelope, narrowed to where the corpus can price a
    # floor. package_for_pwa emits se_floor.json on exactly this grid and checks
    # the same criterion, so choosing it here is what keeps the two in step —
    # a grid picked without asking the corpus is how a knot ends up unmeasurable.
    corpus_r, corpus_i = se_floor.corpus_rank_information(
        ratings_dir, curves, lapse, ALL_BOARD_SIZES)
    lo_t, hi_t = r.mean() - PRIOR_GRID_SD * r.std(ddof=1), r.mean() + PRIOR_GRID_SD * r.std(ddof=1)
    grid = se_floor.grid_from_bounds(lo_t, hi_t, PRIOR_GRID_STEP)
    # Walk the bounds inward until the emitter's own read of this grid is
    # clean. Edges absorb and isolated interior knots interpolate, so this
    # normally does nothing; it moves only when the corpus genuinely stops.
    while grid.size > 2:
        try:
            se_floor.mean_info_curve(grid, corpus_r, corpus_i)
            break
        except RuntimeError:
            grid = grid[1:-1]
    must_lo, must_hi = PRIOR_GRID_MUST_CONTAIN
    if grid.size < 3 or grid[0] > must_lo or grid[-1] < must_hi:
        span = f"[{grid[0]:+.1f}, {grid[-1]:+.1f}]" if grid.size else "empty"
        return None, (f"rank grid narrowed to {span}, which does not contain the "
                      f"labelled ladder [{must_lo:+.0f}, {must_hi:+.0f}] — the corpus "
                      f"cannot price a floor across the ranks users actually hold")
    # Gaussian KDE, then normalised to a pmf over the grid. Mass falling outside
    # the grid is folded in by the normalisation rather than dropped, so the tails
    # stay proper — the rank scale is unbounded and the grid is only a resolution
    # choice, never a claim about where users can be.
    dens = np.exp(-0.5 * ((grid[:, None] - r[None, :]) / PRIOR_BANDWIDTH) ** 2).sum(axis=1)
    dens = np.maximum(dens, dens.max() * 1e-9)          # keep log finite everywhere
    dens /= dens.sum()
    prior = {
        "rank_lo": float(grid[0]),
        "rank_step": float(PRIOR_GRID_STEP),
        "log_density": [round(float(v), 4) for v in np.log(dens)],
        "mean": round(float(r.mean()), 4),
        "sd": round(float(r.std(ddof=1)), 4),
        "n_users": int(r.size),
        "min_attempts": int(MIN_PRIOR_ATTEMPTS),
        "bandwidth": float(PRIOR_BANDWIDTH),
    }
    qs = np.percentile(r, [10, 50, 90])
    info = (f"{r.size} users, mean {r.mean():+.2f} sd {r.std(ddof=1):.2f}, "
            f"p10/p50/p90 {qs[0]:+.1f}/{qs[1]:+.1f}/{qs[2]:+.1f}, "
            f"{grid.size} grid points")
    return prior, info


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
    curve; the size-9 key stands in for either board). None if absent/bad."""
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
        lnS = np.clip(ln_c + kappa * (z - Z0) + growth * r, *LN_S_CLIP)
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


# Relative-interval bands for the step's matched strata: a serve's elapsed gap
# over the STEP-LESS stability the fitted constants predict for it — the
# coordinate in which the shipped step is nothing but a horizontal shift of
# where serves land. eval_recall_model imports these, so the ablation's contrast
# and this fit are the same statistic by construction.
STEP_X_BANDS = (0.05, 0.15, 0.30, 0.60, 1.20)
MIN_STEP_EVENTS_PER_CELL = 2           # per (user, band) stratum, each streak cell
MIN_STEP_USERS = 20                    # distinct users among qualifying strata
STEP_MAX = 2.0                         # sanity rail on the fitted step
LN_S_CLIP = (-5.0, 8.0)                # the retention fit's own lnS clip


def fit_first_success_step(events, retention):
    """The extra log-stability a problem earns from its FIRST success after a
    lapse, over and above the per-success growth already in the ladder.

    The streak counter is zero exactly when the previous attempt FAILED, and one
    when it was the first success after a failure — so rung 0 and rung 1 are the
    post-lapse and post-success states, not two steps of one ladder. Established
    schedulers give a lapse its own formula; this model does not, and the
    shortfall shows up as post-success reviews clearing well above their
    post-lapse peers at the same relative interval — more durable than the
    model has them, so served before the intended forgetting.

    Fit by MOMENT MATCHING, deliberately not by retention likelihood. The
    likelihood is evaluated on intervals the scheduler itself chose, so it is
    close to uninformative about what a different schedule would do — measured,
    it sizes this step at about a fifth of what the realized rates require.
    The moment condition needs no fitting at all: under R = exp(-t/S), scaling
    an interval by f scales log-retention by f, so the multiplier taking the
    post-success cells from their rate p to a target q is ln(q)/ln(p), and the
    step is its logarithm. The target is the post-lapse cells' own realized
    rate — making the ladder self-consistent, which is what the client can
    actually be held to, rather than any particular point inside the window.

    The contrast is taken WITHIN (user, matched relative interval) — a
    Mantel-Haenszel odds ratio over strata of user × gap-over-step-less-stability
    band — and both halves of that stratum are load-bearing. Within USER because
    how promptly someone returns once a window opens is a habit, unobservable
    and stable per person, that sets where in the window their serves land.
    Within a MATCHED BAND because this fit reads realized rates off whatever
    schedule is live, and the step it ships stretches exactly the intervals
    those rates are realized on — a contrast pooled over serves its own
    correction produced walks the step down, converging on no correction at
    all. At a matched relative interval the schedule cancels inside the
    stratum, so serves from before and after any schedule change pool into one
    statistic. Matching on the band alone is not enough: who gets served at a
    given ratio moves with the schedule too, and the user half of the stratum
    is what absorbs that.
    """
    kappa, growth, ln_c = (retention["kappa"], retention["growth"],
                           retention["ln_c_prior"])
    strata = defaultdict(lambda: [[0, 0], [0, 0]])
    for y, gap_days, r, z, user in events:
        if gap_days * 24 < MIN_RECALL_HOURS:
            continue        # under the stability floor: not a schedulable serve
        lnS = min(max(ln_c + kappa * (z - Z0) + growth * r, LN_S_CLIP[0]), LN_S_CLIP[1])
        x = gap_days / math.exp(lnS)
        band = next((i for i, (lo, hi) in
                     enumerate(zip(STEP_X_BANDS, STEP_X_BANDS[1:]))
                     if lo <= x < hi), None)
        if band is None:
            continue
        cell = strata[(user, band)][0 if r == 0 else 1]
        cell[0] += y
        cell[1] += 1

    num = den = 0.0
    k0 = n0 = 0
    used = 0
    users = set()
    for (user, _band), c in strata.items():
        if c[0][1] < MIN_STEP_EVENTS_PER_CELL or c[1][1] < MIN_STEP_EVENTS_PER_CELL:
            continue
        A, B = float(c[1][0]), float(c[1][1] - c[1][0])      # rung >= 1
        C, D = float(c[0][0]), float(c[0][1] - c[0][0])      # rung 0
        n = A + B + C + D
        num += A * D / n
        den += B * C / n
        k0 += c[0][0]
        n0 += c[0][1]
        used += 1
        users.add(user)
    if len(users) < MIN_STEP_USERS or den <= 0 or num <= 0 or n0 == 0:
        return None, (f"{used} matched strata over {len(users)} users with >= "
                      f"{MIN_STEP_EVENTS_PER_CELL} events per cell")

    odds_ratio = num / den
    p0 = k0 / n0                                   # post-lapse rate, matched strata
    if not (0 < p0 < 1):
        return None, "degenerate rung-0 rate"
    odds1 = (p0 / (1 - p0)) * odds_ratio
    p1 = odds1 / (1 + odds1)                       # post-success rate, OR-shifted
    if not (0 < p1 < 1) or p1 <= p0:
        return None, f"post-success rate {p1:.3f} not above post-lapse {p0:.3f}"
    step = math.log(math.log(p0) / math.log(p1))   # ln of the interval multiplier
    if not (0 <= step < STEP_MAX):
        return None, f"step {step:.3f} outside [0, {STEP_MAX})"
    return step, (f"matched OR {odds_ratio:.2f} over {used} strata "
                  f"({len(users)} users); post-lapse {p0:.3f} -> post-success "
                  f"{p1:.3f}; interval x{math.exp(step):.2f}")


# ---------------------------------------------------------------------------
# Weakness-model per-axis prior (empirical Bayes).
# ---------------------------------------------------------------------------
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
    ap.add_argument("--db", default=DB_SQLITE)
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "ratings", "population_constants.json"))
    ap.add_argument("--allow-defaults", action="store_true",
                    help="ship the hardcoded DEFAULTS for any constant that can't be fit "
                         "(sparse/missing inputs) instead of crashing")
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

    # --- Fat-finger (lapse) rate: PRODUCED by the rating model (train_mirt), read through ---
    lam = read_rating_lapse(args.ratings_dir)
    if lam is not None:
        out["fat_finger_rate"] = round(lam, 4)
        print(f"Fat-finger (lapse) rate from the rating model: {out['fat_finger_rate']} "
              f"(ratings/rating_constants.json — train_mirt learns it jointly)")
    else:
        _resort_to_default("fat_finger_rate",
                           "no rating_constants.json (run train_mirt first)",
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

    # --- Population rank prior (the window estimator's prior over user rank) ---
    # Mandatory and with no fallback: without it the client's posterior has no
    # anchor, and the failure is invisible — an unanchored estimator still returns
    # a number, it just tracks wherever it last drifted.
    print("Fitting population rank prior ...")
    # fit_rank_prior maps each user's rating through the shipped curve.
    apply_m = load_rating_to_rank(args.ratings_dir)
    # The raw curve file (both board keys) is ALSO passed, not redundantly with
    # apply_m: the grid derivation reads per-size knots to map the corpus into
    # rank space the way se_floor does.
    with open(os.path.join(args.ratings_dir, "rating_to_rank.json")) as _f:
        _curves = json.load(_f)
    rp, rp_info = fit_rank_prior(by_user, params, apply_m, out["fat_finger_rate"],
                                 args.ratings_dir, _curves)
    if rp is None:
        raise RuntimeError(
            f"could not fit rank_prior ({rp_info}). It is mandatory and has no "
            f"fallback — the client's window estimator is a posterior and needs its "
            f"prior. Not shipping without it.")
    out["rank_prior"] = rp
    print(f"  {rp_info}")

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
            step, step_info = fit_first_success_step(events, fit)
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
