"""Fit the PWA's population-level model constants from the local D1 log and write
them to ratings/population_constants.json, which package_for_pwa copies into
the served pwa/tsumegos/. The client fetches it at boot and uses these in place
of the hardcoded defaults in game-logic.js (time/review bias) and model.js
(retention constants), so the constants track the real user population instead of
drifting from a one-off manual fit.

Five groups of constants:

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

4. WEAKNESS_AXIS_LAMBDA (weakness.js): a per-axis prior precision vector (length K) —
   the L2 precision on each per-embedding-axis beta of the client's weakness logistic.
   Fit by empirical Bayes: the between-user variance tau_k^2 of each axis's weakness
   coefficient is a population variance component, estimated by EM with a Laplace
   E-step — per-user MAP ridge fits under the current per-axis prior, then
   tau_k^2 <- mean_u(beta_uk^2 + posterior var_uk). Axes along which users genuinely
   differ get a loose prior (their coefficients can surface in the Weaknesses pane);
   axes with no between-user signal stay pinned. The offline design is the client's own
   AUGMENTED one ([1, problemRank, correct_emb, picked_emb], 2 + 2K columns, both of an
   axis's slots sharing lambda_k), so the two are the same estimator and their
   coefficients are directly comparable; tau_k^2 is taken from the SUM direction, which
   makes the shipped per-coefficient lambda_k = 2 / tau_k^2 put exactly N(0, tau_k^2)
   on beta_correct[k] + beta_picked[k]. This vector is MANDATORY — there is no scalar
   fallback: if the population is too sparse to fit it, fit_population_constants RAISES
   rather than shipping a fallback, and the client requires it.

5. WEAKNESS_CONTRAST_MEAN (weakness.js): the population mean of
   beta_correct[k] - beta_picked[k], length K, fit in the same pass as (4) and equally
   mandatory. Selection scores candidates on that CONTRAST, not on the sum, and the
   contrast is displaced from zero for everyone: across wrong attempts the played cell
   differs from the answer by a systematic corpus offset — a description of what wrong
   moves look like, not of who played them — which the likelihood routes into
   beta_picked. Scoring the raw contrast therefore serves every user the same corner of
   the corpus in the shape of personalisation. Consumers subtract this vector before
   scoring, leaving the part that actually differs between users. Note the deliberate
   asymmetry with (4): a shared SUM coefficient is residual difficulty and HALTS the
   fit, because centring it would paper over a defect belonging upstream in build_axes;
   a shared CONTRAST is measured wrong-move geometry, and centring it removes a known
   nuisance rather than hiding one.

All constants are board-INDEPENDENT (the cold-solve scale and the embedding axes
are shared across sizes), so a single JSON serves both boards.

Needs the backend ratings/<size>/ 3PL params (tsumego_ratings.csv,
tsumego_slopes.csv, tsumego_floors.csv — written by train_rating)
ratings/rating_to_rank.json (M, for the weakness ridge's problemRank feature),
axes/per_problem.csv (the weakness ridge's correct-move embeddings),
axes/per_cell/ (the picked-move block of the same design),
and db/d1.sqlite. It reads the backend directly, NOT the served
pwa/tsumegos/ shards: package_for_pwa is the only bridge into pwa/, so nothing
upstream of it depends on the dist artifacts. No GPU, not part of the in-order
pipeline.

Run:
  ./.venv/bin/python3 -m pipeline.fetch_database          # refresh db/d1.sqlite
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
    # NB: no weakness prior default. weakness_axis_lambda is fit here and is
    # MANDATORY — if it can't be fit the run raises rather than shipping a
    # fabricated fallback (there is no scalar weakness_ridge_lambda anymore).
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
# The grid must always contain the labelled ladder, or a legitimately ranked user
# cannot be represented at all. Not a clamp on the rank scale (which is unbounded
# by design) — a floor under how far the derivation may narrow.
PRIOR_GRID_MUST_CONTAIN = (-30.0, 9.0)       # 30k .. 9d, clampRankBand's range

# --- Weakness-model per-axis prior (pwa/js/weakness.js) — shared inputs ---
# The per-axis L2 precision on the client's weakness betas is fit by empirical
# Bayes (group 4 / fit_weakness_axis_priors). These are the shared inputs.
# Board-independent (9x9+13x13 pooled), the embedding axes being joint.
WEAKNESS_EMBED_CSV = os.path.join(PROJECT_ROOT, "axes", "per_problem.csv")
# Per-CELL axis vectors, for the picked-move block of the client's design. Same
# arrays package_for_pwa fans out into the per-shard .embeddings.bin sidecars, read
# here straight from the build_axes output (nothing upstream of packaging depends on
# pwa/). Keyed in the extraction frame: xys is board-scan order with y counting from
# the BOTTOM, so a reported "C7" is (x=2, y=6) — see _picked_xy.
WEAKNESS_PER_CELL_DIR = os.path.join(PROJECT_ROOT, "axes", "per_cell")
# tsumego_reports.move is written by toBoardNotation (pwa/js/sgf.js), whose column
# alphabet skips I, and always in the CANONICAL orientation-0 frame — the client
# inverse-transforms the click before reporting, so a repeat's random reorientation
# never reaches the log and every row indexes the shard SGF directly.
WEAKNESS_MOVE_COLS = "ABCDEFGHJKLMNOPQRST"


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
# The bar is on whether the population mean EXCEEDS the between-user spread
# (share > 0.5 <=> mu > sigma), not on whether a mean exists at all: a real
# dimension may carry an offset, and the offset is shipped rather than refused.
WEAKNESS_MEAN_SHARE_MAX = 0.50
# Between-user variance must clear this multiple of the mean per-user estimation
# variance. Below it, individual coefficients are not distinguishable from the
# population's and "personalisation" on the axis is fitting noise.
WEAKNESS_AXIS_MIN_VAR_RATIO = 0.25
# Attempts whose resolvable picked cell differs from the answer, pooled over the fit.
# Only these separate beta_correct from beta_picked, so below this the contrast is the
# prior talking and its pooled mean is not a measurement — the fit raises rather than
# shipping a centre that would tilt every client's selection on nothing. Sized so the
# per-axis mean is many times its own standard error at the observed spread.
WEAKNESS_CONTRAST_MIN_ROWS = 2000


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
    if corpus_r.size:
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
    axes/per_problem.csv, restricted to the attempted problems.
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


def _picked_xy(move, size):
    """Reported board notation -> (x, y) in the per_cell/activation frame, or None.

    toBoardNotation is COL_LETTERS[x] + (size - y_sgf) and the extraction frame flips
    the row (y_npz = size - 1 - y_sgf), so y_npz is just the printed row number minus
    one. Verified against correct_idx vs the SGF solution move across the corpus."""
    if not isinstance(move, str):
        return None
    m = move.strip().upper()
    if len(m) < 2 or m[0] not in WEAKNESS_MOVE_COLS:
        return None
    try:
        row = int(m[1:])
    except ValueError:
        return None
    x, y = WEAKNESS_MOVE_COLS.index(m[0]), row - 1
    if not (0 <= x < size and 0 <= y < size):
        return None
    return x, y


def load_picked_embeddings(rows):
    """{(size, tid, move): np.ndarray(K)} for the WRONG novel attempts — the axis
    vector of the cell the user actually played.

    This is the second half of the client's design (pwa/js/weakness.js): its logistic
    carries a beta_correct block on the position's correct-move vector AND a
    beta_picked block on the played cell, and only rows where the two DIFFER identify
    them apart. Fitting the offline prior on the correct block alone therefore cannot
    see the contrast beta_correct - beta_picked at all — which is the quantity
    selection scores on (weaknessScore), so it is the quantity whose population
    displacement has to be measured and shipped.

    Resolved to one vector per (problem, move) pair actually attempted rather than a
    whole per-cell table per problem: the corpus has ~150k problems with tens of legal
    cells each, while the attempt log names only a few tens of thousands of pairs.
    Missing entries are not an error — a row predating move logging (2026-05-15), a
    pass, or a problem whose per_cell file is absent all fall back to picked = correct,
    exactly as the client does when the sidecar fetch has not landed."""
    if not os.path.isdir(WEAKNESS_PER_CELL_DIR):
        return {}
    wanted = defaultdict(set)
    for r in rows:
        if r["is_repeat"] or r["success"] or r["move"] is None:
            continue
        wanted[(int(r["size"]), r["problem"])].add(r["move"])
    out = {}
    for (size, tid), moves in wanted.items():
        path = os.path.join(WEAKNESS_PER_CELL_DIR, str(size), f"{tid}.npz")
        if not os.path.exists(path):
            continue
        try:
            with np.load(path) as d:
                cells = {(int(x), int(y)): i for i, (x, y) in enumerate(d["xys"])}
                emb = d["embeddings"]
                for mv in moves:
                    xy = _picked_xy(mv, size)
                    if xy is None or xy not in cells:
                        continue
                    vec = np.asarray(emb[cells[xy]], dtype=float)
                    if np.all(np.isfinite(vec)):
                        out[(size, tid, mv)] = vec
        except (OSError, ValueError, KeyError):
            continue
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


def _weakness_user_datasets(by_user, rank_feat, embeddings, picked_embeddings):
    """One (X, y) design per user for the empirical-Bayes variance fit: the most
    recent WEAKNESS_EB_MAX_N novel rated+embedded attempts (mirroring the client's
    MODEL_INPUT_HISTORY_LENGTH training window), users below WEAKNESS_EB_MIN_N or
    with a degenerate outcome dropped.

    X = [1, problemRank, correct_0..K-1, picked_0..K-1] — the client's AUGMENTED
    design, column for column. The picked block repeats the correct block on a
    solved attempt (picked == correct by definition) and on any wrong attempt whose
    cell can't be resolved, which is the client's own fallback; such a row informs
    only the sum direction, never the contrast."""
    for urows in by_user.values():
        ev = []
        for r in urows:
            if r["is_repeat"]:
                continue
            key = (r["size"], r["problem"])
            if key not in rank_feat or key not in embeddings:
                continue
            corr = embeddings[key]
            pick = corr
            if not r["success"] and r["move"] is not None:
                pick = picked_embeddings.get((int(r["size"]), r["problem"], r["move"]), corr)
            ev.append((datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                       1.0 if r["success"] else 0.0, rank_feat[key], corr, pick))
        if len(ev) < WEAKNESS_EB_MIN_N:
            continue
        ev.sort(key=lambda e: e[0])
        ev = ev[-WEAKNESS_EB_MAX_N:]
        y = np.array([e[1] for e in ev], float)
        if y.std() < 1e-3:
            continue
        X = np.column_stack([np.ones(len(ev)),
                             np.array([e[2] for e in ev], float),
                             np.vstack([e[3] for e in ev]),
                             np.vstack([e[4] for e in ev])])
        yield X, y


def fit_weakness_axis_priors(by_user, params, embeddings, picked_embeddings, apply_m,
                             allow_axis_mean=False):
    """Per-axis prior precision vector for the client's weakness betas (group 5 in
    the module docstring): empirical Bayes on the between-user variance component
    tau_k^2 of each axis's weakness coefficient, by EM with a Laplace E-step —
    per-user MAP ridge fits under the current per-axis prior, then
    tau_k^2 <- mean_u(beta_uk^2 + posterior var_uk).

    The design is the client's AUGMENTED one — [1, problemRank, correct, picked],
    2 + 2K columns — not a single block on the correct-move vector. Both of an axis's
    slots carry the same lambda_k, exactly as pwa/js/weakness.js penalises them, so
    the offline fit and the client's are the same estimator and their coefficients are
    directly comparable. tau_k^2 is still set from the SUM direction, keeping the
    shipped contract (lambda_k = 2/tau_k^2 puts N(0, tau_k^2) on
    beta_correct[k] + beta_picked[k]) unchanged.

    Fitting the augmented design is what lets this step see the CONTRAST
    beta_correct - beta_picked, which is the direction selection actually scores on
    and which a single-block design cannot represent at all. Its population mean ships
    as weakness_contrast_mean — see the note at the check below for why that one is
    centred and the sum is not.

    Returns (list of K lambdas, list of K contrast means, info), or (None, None, info)
    when the data is too sparse."""
    if apply_m is None:
        return None, None, None, "no rating_to_rank.json — no per-axis prior"
    if not embeddings:
        return None, None, None, "no embeddings — no per-axis prior"
    rank_feat = {k: apply_m(v[0]) for k, v in params.items()}
    data = list(_weakness_user_datasets(by_user, rank_feat, embeddings, picked_embeddings))
    if len(data) < WEAKNESS_EB_MIN_USERS:
        return None, None, None, (f"only {len(data)} users with ≥{WEAKNESS_EB_MIN_N} usable "
                            f"attempts (<{WEAKNESS_EB_MIN_USERS}) — no per-axis prior")
    K = _weakness_k_embed()
    d = 2 + 2 * K
    # Rows where the played cell differs from the correct one are the ONLY rows that
    # separate beta_correct from beta_picked. With too few of them the contrast is
    # prior alone and its pooled mean would be an artefact of the shrinkage rather
    # than a measurement, so refuse instead of shipping a fabricated centre.
    splits = [int(np.count_nonzero(np.any(X[:, 2:2 + K] != X[:, 2 + K:], axis=1)))
              for X, _ in data]
    n_split = sum(splits)
    if n_split < WEAKNESS_CONTRAST_MIN_ROWS:
        return None, None, None, (f"only {n_split} attempts with a resolvable picked cell "
                            f"distinct from the answer (<{WEAKNESS_CONTRAST_MIN_ROWS}) "
                            f"— the beta_correct/beta_picked contrast is unidentified")
    # tau^2 bounds in client-lambda terms (lambda_k = 2/tau_k^2).
    tau2_lo, tau2_hi = 2.0 / WEAKNESS_EB_LAMBDA_MAX, 2.0 / WEAKNESS_EB_LAMBDA_MIN
    # EM seed only (the iteration converges to the data-driven variance): the
    # geometric mean of the bounds, so no scalar prior is needed to start.
    tau2 = np.full(K, float(np.sqrt(tau2_lo * tau2_hi)))
    betas = [np.zeros(d) for _ in data]   # warm starts across EM iterations
    it = 0
    for it in range(WEAKNESS_EB_MAX_ITERS):
        # The SHIPPED lambda_k = 2/tau_k^2 on BOTH of axis k's slots — the client's own
        # penalty lambda_k*(beta_correct_k^2 + beta_picked_k^2), so this loop is
        # literally the fit the client runs. The factor of 2 is what keeps the M-step
        # self-consistent: independent slots at precision 2/tau^2 give the sum variance
        # tau^2, which is exactly the second moment tau^2 is then re-estimated from.
        # Halving it (per-slot precision 1/tau^2) leaves the prior on the sum twice as
        # loose as the quantity it is matched to, and the EM converges to an inflated
        # tau^2 — a looser shipped prior with nothing in the data asking for it.
        lam_vec = np.zeros(d)
        lam_vec[2:2 + K] = 2.0 / tau2
        lam_vec[2 + K:] = 2.0 / tau2
        second_sum = np.zeros(K)          # Σ_u ((bc+bp)^2 + var(bc+bp)) — sets tau^2
        first_sum = np.zeros(K)           # Σ_u (bc+bp) — the SHIPPED sum centre
        var_sum = np.zeros(K)             # Σ_u var(bc+bp) — per-user estimation noise
        # Contrast moments over POINT ESTIMATES only, and only over users whose
        # contrast is identified at all. Both restrictions follow from what the number
        # is for: selection scores the fitted contrast itself, so the posterior
        # variance never reaches the score and does not belong in a statement about how
        # much of the score is shared; and a user with no split row has bc and bp
        # sitting on identical columns under one lambda, so their contrast is exactly
        # zero by construction — averaging those in drags the shipped centre toward
        # zero in proportion to how many such users there are, and the client would
        # then under-subtract for everyone.
        second_con = np.zeros(K)          # Σ_u (bc-bp)^2, identified users
        first_con = np.zeros(K)           # Σ_u (bc-bp)  — the SHIPPED contrast centre
        n_ok = n_con = 0
        for i, (X, y) in enumerate(data):
            beta, ok = _ridge_logistic(X, y, lam_vec, beta0=betas[i])
            if not ok or not np.all(np.isfinite(beta)):
                continue
            betas[i] = beta
            p = sigmoid(X @ beta)
            w = np.clip(p * (1 - p), 1e-6, None)
            H = (X * w[:, None]).T @ X + np.diag(np.maximum(lam_vec, 1e-6))
            cov = np.linalg.inv(H)
            bc, bp = beta[2:2 + K], beta[2 + K:]
            vc, vp = np.diag(cov)[2:2 + K], np.diag(cov)[2 + K:]
            # Off-diagonal of each axis's own 2x2 block. The two slots are strongly
            # anti-correlated (most rows carry picked == correct, which moves them
            # together), so dropping it would misstate BOTH directions' spread —
            # understating the sum's and overstating the contrast's.
            vcp = cov[np.arange(2, 2 + K), np.arange(2 + K, d)]
            s, c = bc + bp, bc - bp
            v_s = np.maximum(vc + vp + 2 * vcp, 0.0)
            second_sum += s ** 2 + v_s
            first_sum += s
            var_sum += v_s
            n_ok += 1
            if splits[i]:
                second_con += c ** 2
                first_con += c
                n_con += 1
        if n_ok < WEAKNESS_EB_MIN_USERS:
            return None, None, None, f"only {n_ok} converged per-user fits — no per-axis prior"
        new_tau2 = np.clip(second_sum / n_ok, tau2_lo, tau2_hi)
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
    mu = first_sum / n_ok
    raw2 = second_sum / n_ok
    v_est = var_sum / n_ok                     # mean per-user estimation variance
    # raw2 is a POSTERIOR second moment (beta^2 + var), so it already estimates
    # E[theta^2] rather than the spread of point estimates. Subtracting mu^2
    # therefore gives the between-user variance with estimation noise removed.
    tau2_c = np.maximum(raw2 - mu ** 2, 0.0)
    share = mu ** 2 / np.maximum(raw2, 1e-12)
    resolve = tau2_c / np.maximum(v_est, 1e-12)
    print("  weakness SUM direction (centre is SHIPPED as weakness_axis_mean):")
    for k in range(K):
        print(f"    axis {k}: pooled mean {mu[k]:+.4f}  between-user sd "
              f"{np.sqrt(tau2_c[k]):.4f}  mean^2/second-moment {share[k]:6.1%}"
              f"  var ratio {resolve[k]:5.1f}x")
    print(f"    worst axis: {share.max():.1%} of the second moment is pooled mean; "
          f"lowest between-user/estimation variance ratio {resolve.min():.1f}x")

    # WHAT THIS REFUSES, AND WHAT IT NO LONGER REFUSES.
    #
    # The harm a non-zero population mean does is real but specific: the client's
    # prior is centred at zero, so tau^2 is inflated by mu^2 (a looser prior than
    # the data asks for) and level-one tests every user against zero, which flags a
    # user holding the population-TYPICAL coefficient as having a finding. Both are
    # removed by shipping the centre, which is what now happens — the same treatment
    # the contrast direction already gets, and the sum/contrast asymmetry goes with
    # it.
    #
    # So a non-zero mean is no longer sufficient to refuse. It cannot be: the
    # statistic mu^2/(mu^2+sigma^2) at a 10% bar demands mu <= 0.33*sigma, and an
    # ordinary between-user dimension is entitled to a population mean a third the
    # size of its spread. Refusing there vetoes real axes — measured: an axis with
    # between-user sd 0.26 against a mean of 0.11, i.e. 84% of its second moment
    # surviving centring, was refused as "difficulty".
    #
    # What genuinely is difficulty in disguise is an axis on which users do not
    # differ at all: every true coefficient equal, the apparent spread all fitting
    # error. That is what tau2_c and the variance ratio measure, and it is what this
    # refuses on. Two independent conditions, because they fail differently — an
    # axis can be mostly offset while still resolvable, or well-centred while no
    # individual coefficient is distinguishable from the population's.
    bad_share = share > WEAKNESS_MEAN_SHARE_MAX
    bad_resolve = resolve < WEAKNESS_AXIS_MIN_VAR_RATIO
    if (bad_share | bad_resolve).any() and not allow_axis_mean:
        k_bad = int(np.argmax(np.where(bad_share | bad_resolve, share, -1.0)))
        why = ("its pooled mean EXCEEDS its between-user spread"
               if bad_share[k_bad] else
               "its between-user variance is not credibly above per-user "
               "estimation noise")
        raise RuntimeError(
            f"axis {k_bad} is a population effect rather than a per-user one: {why} "
            f"(mean {mu[k_bad]:+.4f}, between-user sd {np.sqrt(tau2_c[k_bad]):.4f}, "
            f"mean^2/second-moment {share[k_bad]:.1%} against the "
            f"{WEAKNESS_MEAN_SHARE_MAX:.0%} bar, variance ratio "
            f"{resolve[k_bad]:.1f}x against {WEAKNESS_AXIS_MIN_VAR_RATIO:.1f}x).\n"
            "An axis everyone responds to identically is residual difficulty, which "
            "build_axes quarantines its population residual head to keep OUT of the "
            "axes — so read this as that quarantine leaking and look there first. "
            "Shipping a centre does NOT repair this case: centring an axis with no "
            "between-user variation leaves nothing to personalise on, and the panel "
            "would report the remaining noise as findings.\n"
            "--allow-axis-mean ships anyway, deliberately.")

    # CONTRAST DIRECTION. beta_correct - beta_picked is what problem-loader.js scores
    # candidates on (weaknessScore), and it is displaced for a reason the sum check
    # above cannot see and must not be read as: across wrong attempts the played cell
    # differs from the answer by a systematic corpus offset
    # delta = E[correct_emb - picked_emb | wrong], the same for every user, because it
    # describes what wrong moves look like rather than who played them. The likelihood
    # routes delta into beta_picked, so every user's contrast inherits a common -delta
    # component and the score up-weights whichever problems sit furthest along +delta —
    # the same corner of the corpus for everybody, dressed as personalisation.
    #
    # So this displacement is SHIPPED rather than halted on, and the asymmetry with the
    # sum is the point, not an inconsistency. A shared SUM coefficient means an axis
    # carries difficulty that belongs to the rating model, and centring it would hide
    # that; a shared CONTRAST is a measured property of wrong-move geometry, and
    # centring it removes a known nuisance so what remains is the part that differs
    # between users. Consumers subtract this vector before scoring; the residual is
    # zero-mean by construction, which is the condition the sum check enforces by
    # refusing.
    if n_con < WEAKNESS_EB_MIN_USERS:
        return None, None, None, (f"only {n_con} users have an identified contrast "
                            f"(<{WEAKNESS_EB_MIN_USERS}) — no contrast centre")
    mu_con = first_con / n_con
    raw2_con = second_con / n_con
    share_con = mu_con ** 2 / np.maximum(raw2_con, 1e-12)
    print("  weakness contrast direction (beta_correct - beta_picked; SHIPPED centre):")
    for k in range(K):
        print(f"    axis {k}: pooled mean {mu_con[k]:+.4f}  "
              f"sd {np.sqrt(max(raw2_con[k] - mu_con[k] ** 2, 0.0)):.4f}"
              f"  mean^2/second-moment {share_con[k]:6.1%}")
    print(f"    worst axis: {share_con.max():.1%} of the scored contrast is a common "
          f"offset — shipped as weakness_contrast_mean, from {n_con} users with an "
          f"identified contrast over {n_split} attempts whose pick differs from the answer")

    info = (f"{len(data)} users, {it + 1} EM iters; per-coefficient lambda "
            f"min/median/max = {lam_axis.min():.0f}/{float(np.median(lam_axis)):.0f}/"
            f"{lam_axis.max():.0f}; sum centre worst share {share.max():.1%}, "
            f"lowest var ratio {resolve.min():.1f}x; "
            f"contrast offset worst {share_con.max():.1%} ({n_con} users, {n_split} split rows)")
    return ([round(float(l), 2) for l in lam_axis],
            [round(float(m), 5) for m in mu_con],
            [round(float(m), 5) for m in mu], info)


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
    picked_embeddings = load_picked_embeddings(rows)
    print(f"  resolved {len(picked_embeddings)} picked-move cells from axes/per_cell/")
    wa, wc, wm, wa_info = fit_weakness_axis_priors(by_user, params, embeddings,
                                               picked_embeddings, apply_m,
                                               args.allow_axis_mean)
    if wa is None:
        raise RuntimeError(
            f"could not fit weakness_axis_lambda ({wa_info}). It is mandatory and has "
            f"no fallback — fix the inputs (run map_rating_to_rank + build_axes, "
            f"or wait for more attempts) and re-run. Not shipping without it.")
    out["weakness_axis_lambda"] = wa
    # The centre for the direction selection scores on. Mandatory for the same reason
    # as the precision vector: shipping the lambdas without it leaves consumers
    # scoring an uncentred contrast, which is the whole defect this pair fixes.
    out["weakness_contrast_mean"] = wc
    # The SUM direction's population centre. Shipped for the same reason the
    # contrast's is: the client's prior would otherwise sit at zero while the
    # population sits elsewhere, which both loosens the prior (tau^2 inflated by
    # mu^2) and makes level one test every user against a point the typical user
    # does not occupy. An axis may legitimately carry an offset; what it may not
    # carry is the absence of between-user variation, which the refusal above is
    # what now guards.
    out["weakness_axis_mean"] = wm
    print(f"  {wa_info}")

    # --- Population rank prior (the window estimator's prior over user rank) ---
    # Mandatory for the same reason as the weakness vector: without it the client's
    # posterior has no anchor, and the failure is invisible — an unanchored estimator
    # still returns a number, it just tracks wherever it last drifted.
    print("Fitting population rank prior ...")
    # The raw curve file (both board keys), not apply_m: the grid derivation reads
    # per-size knots to map the corpus into rank space the way se_floor does.
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
