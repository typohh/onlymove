#!/usr/bin/env python3
"""Offline bake-off of live (causal) rank estimators on real users.

Motivation
----------
The PWA derives a *live* user rank two different ways today:

  * Front page  — a hand-tuned momentum recurrence (`pwa/js/user-stats.js`):
    every user cold-starts at 20k (index 0) and ratchets up. It is causal,
    O(1)/attempt, drives both the displayed label AND problem selection, and
    has no 3PL guessing-floor / near-rank handling.
  * Rank tab    — a two-sided time-Gaussian pooled 3PL MLE
    (`pwa/js/rank-decomposition.js`), a *retrospective smoother* (uses future
    data), not usable as a live filter.

This script tests replacing the live estimator with a *causal* MLE whose window
is set by accumulated Fisher INFORMATION rather than by a hand-tuned momentum or
a fixed time bandwidth, plus a state-space (random-walk skill + 3PL) variant. It
replays each user's real attempt stream and scores estimators on the two goals
that trade off against each other:

  (1) Long-term serving quality  — once warmed up, does the live rank track the
      user's true level so served problems stay near the 80%-clear target?
      Measured by steady-phase predictive log-loss/Brier (assumption-free) and
      RMSE of the served rank vs the full-history smoother reference.
  (2) Onboarding speed           — how fast does a brand-new user reach their
      level? Measured by early-phase predictive log-loss and attempts/days to
      converge within a tolerance of the user's final smoothed level.

A good prior (population mean, or the user's rank-survey declaration) buys (2)
cheaply; a tighter information window / lower process noise buys (1) at the cost
of (2). The sweep makes that frontier visible.

Coordinate system mirrors the client exactly: everything is in INTERNAL-RANK
INDEX space (0..29, EGF-anchored), each attempt mapped through its own size's
`rating_to_rank.json` curve M:
    x      = applyM(rawRating)                         (problem rank index)
    slope  = clamp(a_p / |M'(rating)|, [0.05, 3.0])    (logit per internal rank)
    floor  = clampFloor(problemFloor), kappa = clearKappa(floor)
This is byte-for-byte the mapping in `rank-decomposition.js::flattenAttempts`.

Reference "truth"
-----------------
There is no external ground truth, so two reference families are reported:
  * Assumption-free: one-step-ahead predictive log-loss/Brier — at each attempt
    the estimator (seeing only PAST attempts) predicts P(clear) for the served
    problem via its 3PL curve; scored against the actual outcome. This needs no
    notion of a "true" rank and directly measures serving-target accuracy.
  * Smoother-referenced: R*(t), the SAME two-sided wide-kernel 3PL MLE the rank
    tab uses, run over the user's FULL history. Used for convergence/RMSE in
    interpretable rank units. It is a model-defined reference, not gospel — read
    it alongside the log-loss, which makes no such assumption.

Data sources (all gitignored, produced by the pipeline):
  attempts/database.sqlite        first-time attempts + rank_surveys
  ratings/<size>/tsumego_{ratings,quality,floor}_final.csv
  ratings/rating_to_rank.json  M (per size), ratings/rank_map.json (survey)

Usage:
  ./.venv/bin/python3 -m auxiliary.eval_rank_estimators
  ./.venv/bin/python3 -m auxiliary.eval_rank_estimators --min-attempts 60 \
        --new-user-after 2025-09-15 --max-users 400 --csv out.csv
  ./.venv/bin/python3 -m auxiliary.eval_rank_estimators --only momentum,infowin_I8_survey
"""

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- client constants (kept in sync with pwa/js/*.js; see module docstring) ---
TOTAL_RANKS = 29
CALIBRATION_P = 0.8
LOGIT_CALIBRATION = math.log(CALIBRATION_P / (1 - CALIBRATION_P))
FLOOR_MAX = 0.78
SLOPE_LO, SLOPE_HI = 0.05, 3.0
MSLOPE_FLOOR = 1e-3
LOGIT_PER_RATING = 1.6          # fallback per-attempt slope (legacy entries)
LOGIT_PER_RANK_FALLBACK = 0.28
X_CLAMP_LO, X_CLAMP_HI = -15.0, 45.0

HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS
TAU_S_MS = 30 * DAY_MS           # reference smoother (rank-tab wide kernel)
KERNEL_TRUNC_SIGMA = 4

# momentum baseline (user-stats.js / state.js)
DEFAULT_MOMENTUM = 30.0
MOMENTUM_MIN, MOMENTUM_MAX = 0.5, 300.0
MIN_RANK_INCREASE_PER_CORRECT = (1.0 / TOTAL_RANKS) / 10.0


# ---------------------------------------------------------------------------
# Shared 3PL / rank math (ports of pwa/js/irt.js + shared/rank-math.js)
# ---------------------------------------------------------------------------
def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def clamp_floor(c):
    if not isinstance(c, (int, float)):
        return 0.0
    return min(max(c, 0.0), FLOOR_MAX)


def clear_kappa(c):
    if not (c > 0):
        return LOGIT_CALIBRATION
    q = min(max((CALIBRATION_P - c) / (1 - c), 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def p_clear(R, x, slope, floor, kappa, lapse=0.0):
    """4PL P(clear): floor + (1-floor-lapse)*sigmoid(slope*(R-x)+kappa).

    `lapse` is the upper-asymptote slip rate: even at infinite ability P(clear)
    tops out at 1-lapse, so a failure on a trivially-easy problem is no longer a
    near-impossible event (probability lapse, not ~0). lapse=0 is the plain 3PL."""
    praw = sigmoid(slope * (R - x) + kappa)
    p = floor + (1 - floor - lapse) * praw
    return min(max(p, 1e-9), 1 - 1e-9)


def interp_extrap(x, xs, ys):
    n = len(xs)
    if x <= xs[0]:
        return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
    if x >= xs[n - 1]:
        return ys[n - 1] + (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]) * (x - xs[n - 1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        m = (lo + hi) >> 1
        if xs[m] <= x:
            lo = m
        else:
            hi = m
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo])


def rank_label_to_scalar(label):
    s = str(label or "").strip().lower()
    if len(s) < 2 or s[-1] not in "kd" or not s[:-1].isdigit():
        return None
    n = int(s[:-1])
    return (1 - n) if s[-1] == "k" else n


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
class Corpus:
    """Per-problem 3PL params + M curve, keyed by board size."""

    def __init__(self, root):
        self.ratings = {}   # size -> {tid: rating}
        self.slopes = {}    # size -> {tid: a_p}
        self.floors = {}    # size -> {tid: floor}
        self.curve = {}     # size -> (ratings[], rank_indices[])
        rr = json.load(open(os.path.join(root, "ratings", "rating_to_rank.json")))
        for size in ("9", "13"):
            d = os.path.join(root, "ratings", size)
            self.ratings[size] = _read_csv(os.path.join(d, "tsumego_ratings_final.csv"), "rating")
            self.slopes[size] = _read_csv(os.path.join(d, "tsumego_quality_final.csv"), "quality")
            self.floors[size] = _read_csv(os.path.join(d, "tsumego_floor_final.csv"), "floor")
            self.curve[size] = (rr[size]["ratings"], rr[size]["rank_indices"])

    def apply_m(self, rating, size):
        xs, ys = self.curve[size]
        return interp_extrap(rating, xs, ys)

    def m_slope(self, rating, size):
        xs, ys = self.curve[size]
        h = 1e-3
        return (interp_extrap(rating + h, xs, ys) - interp_extrap(rating - h, xs, ys)) / (2 * h)

    def attempt_features(self, tid, size):
        """Return (x, slope_in_rank, floor, kappa, a_p_raw) or None if unknown.

        a_p_raw is the problem's RAW 2PL discrimination (the `quality` column) —
        the momentum baseline needs it because the client's selection weight /
        momentum `quality` is the raw a_p^2 (median ~6, clamped to [0,1]), NOT the
        in-rank slope^2."""
        sz = str(size)
        if sz not in self.ratings or tid not in self.ratings[sz]:
            return None
        rating = self.ratings[sz][tid]
        a_p = self.slopes[sz].get(tid)
        if a_p is None or a_p <= 0:
            a_p = LOGIT_PER_RATING
        x = self.apply_m(rating, sz)
        denom = max(abs(self.m_slope(rating, sz)), MSLOPE_FLOOR)
        slope = a_p / denom
        if not math.isfinite(slope) or slope <= 0:
            slope = LOGIT_PER_RANK_FALLBACK
        slope = min(SLOPE_HI, max(SLOPE_LO, slope))
        floor = clamp_floor(self.floors[sz].get(tid))
        return x, slope, floor, clear_kappa(floor), a_p


def _read_csv(path, value_col):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["tsumego_id"]] = float(row[value_col])
            except (KeyError, ValueError):
                pass
    return out


def parse_ts(s):
    """ISO8601 (with Z) -> epoch ms."""
    return int(datetime.fromisoformat(s.replace("Z", "+00:00"))
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def load_survey_priors(conn, corpus):
    """user_id -> internal-rank-index prior from their LATEST rank survey.

    server scalar (rank_map.json 'server') -> our scalar ('our'); index = our+19
    (index 19 == 1k, matching rankIndexToKey in rank-translation.js). EGF is the
    identity hub. Users with no usable survey are absent (caller falls back to the
    population prior)."""
    rmap = json.load(open(os.path.join(PROJECT_ROOT, "ratings", "rank_map.json")))
    rows = conn.execute(
        "SELECT user_id, server, rank, created_at FROM rank_surveys ORDER BY created_at"
    ).fetchall()
    out = {}
    for user_id, server, rank, _ in rows:   # later rows overwrite -> latest wins
        scalar = rank_label_to_scalar(rank)
        if scalar is None:
            continue
        if server == "EGF":
            our = scalar
        elif server in rmap:
            our = interp_extrap(scalar, rmap[server]["server"], rmap[server]["our"])
        else:
            continue
        out[user_id] = our + 19.0
    return out


def load_users(conn, corpus, new_user_after, min_attempts, max_users, max_attempts):
    """Per-user time-sorted first-time attempts mapped to internal-rank space.

    Keeps only 'new' users (first attempt strictly after new_user_after, so the
    onboarding phase is observed, not left-censored) with >= min_attempts usable
    first-time attempts. Returns {user_id: [attempt, ...]} where each attempt is
    (t_ms, x, slope, floor, kappa, correct)."""
    after_ms = parse_ts(new_user_after if "T" in new_user_after
                         else new_user_after + "T00:00:00Z")
    rows = conn.execute(
        "SELECT user_id, problem, size, success, created_at "
        "FROM tsumego_reports WHERE is_repeat=0 ORDER BY user_id, created_at"
    ).fetchall()

    by_user = defaultdict(list)
    first_seen = {}
    for user_id, problem, size, success, created_at in rows:
        t = parse_ts(created_at)
        if user_id not in first_seen:
            first_seen[user_id] = t
        feat = corpus.attempt_features(problem, size)
        if feat is None:
            continue
        x, slope, floor, kappa, a_p = feat
        by_user[user_id].append((t, x, slope, floor, kappa, 1 if success else 0, a_p))

    users = {}
    onboarded = {}
    for user_id, attempts in by_user.items():
        if first_seen[user_id] <= after_ms:
            continue
        if len(attempts) < min_attempts:
            continue
        attempts.sort(key=lambda a: a[0])
        # Cap to the first max_attempts (onboarding + early steady is what we
        # study, and the reference smoother / causal kernel are O(n^2) per user,
        # so dense power users would dominate runtime).
        if max_attempts and len(attempts) > max_attempts:
            attempts = attempts[:max_attempts]
        users[user_id] = attempts
        onboarded[user_id] = first_seen[user_id]

    if max_users and len(users) > max_users:
        # Deterministic subsample: keep the users with the most attempts (most
        # informative onboarding+steady traces), stable across runs.
        keep = sorted(users, key=lambda u: (-len(users[u]), u))[:max_users]
        users = {u: users[u] for u in keep}
        onboarded = {u: onboarded[u] for u in keep}
    return users, onboarded


def cohort_month(t_ms):
    return datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y-%m")


def serving_drift(attempts, ref):
    """How accurately were this user's problems actually served — i.e. how much
    to TRUST them as a test case, and whether they were onboarded under stale
    ratings.

    Production serves at the 80%-clear target near the user's live rank. We
    measure, under TODAY's ratings, how far each served problem sat from the
    user's contemporaneous ability as estimated by the full-history reference
    smoother R*(t) (itself built from current ratings + outcomes):

        drift_i = x_i - R*(t_i)   (internal-rank units, same applyM space as R*)

    If current ratings are accurate and serving was on-target, problems cluster
    at R* (drift ~ 0, small RMS). A wide/biased drift means served problems are,
    by today's ratings, far from where the user's own results place them — the
    signature of a user onboarded under less-accurate ratings. Weight those
    users down or restrict to recent cohorts. (Uniform global re-rating partly
    cancels because R* shifts with it; per-problem rating noise does not, so RMS
    is the robust staleness/consistency signal.)

    Reference (R*) and momentum-rank space differ by a calibration offset, so we
    deliberately do NOT measure drift against momentum. clear_rate is reported
    for sanity (production targets 0.8) but is ~constant by construction and not
    itself a staleness signal."""
    n = 0
    sum_d = sum_d2 = sum_y = 0.0
    for i, att in enumerate(attempts):
        ri = ref[i]
        if not math.isfinite(ri):
            continue
        d = att[1] - ri
        sum_d += d
        sum_d2 += d * d
        sum_y += att[5]
        n += 1
    if n == 0:
        return {"serve_drift_mean": float("nan"), "serve_drift_rms": float("nan"),
                "clear_rate": float("nan"), "n_attempts": 0}
    return {
        "serve_drift_mean": sum_d / n,
        "serve_drift_rms": math.sqrt(sum_d2 / n),
        "clear_rate": sum_y / n,
        "n_attempts": n,
    }


def run_misclick(users, first_n, strong_rank):
    """Misclick / lapse-rate analysis.

    The 3PL handles the LOW tail (a guessing floor: lucky clears of too-hard
    problems). It has no UPPER asymptote, so a problem far below the user is
    predicted clearable with P -> 1 and a failure there is treated as enormous
    evidence to lower the rank. But some of those failures are just slips
    (misclick, misread, distraction) — a 'lapse rate' the model can't see.

    We isolate them: take each user's first `first_n` first-time attempts (when
    momentum cold-starts everyone at 20k, so a strong player is served problems
    far below their level), compute the EXPECTED failure probability of each from
    the user's LONG-TERM rating (full-history smoother evaluated at their last
    attempt), bin by that expected failure, and compare to the ACTUAL failure
    rate. In the expected-failure -> 0 bins, the actual rate IS the lapse/slip
    floor; expected vs actual >> 1 means the model badly under-predicts failure
    on trivial problems and a lapse parameter is warranted.

    Reports the all-users view and the strong-player subset (long-term internal
    rank >= strong_rank, default index 20 == 1d) separately, since strong players
    give the cleanest near-zero-expected-failure sample."""
    # Long-term rating = smoother at the user's last attempt (one O(n) call).
    rows = []          # (expected_fail, failed(0/1), R_lt)
    rows_strong = []
    for uid, att in users.items():
        ts = [a[0] for a in att]
        R_lt = pooled_3pl_mle(att, att[-1][0], TAU_S_MS, ts)[0]
        if not math.isfinite(R_lt):
            continue
        for a in att[:first_n]:
            x, slope, floor, kappa, y = a[1], a[2], a[3], a[4], a[5]
            exp_fail = 1.0 - p_clear(R_lt, x, slope, floor, kappa)
            rows.append((exp_fail, 1 - y))
            if R_lt >= strong_rank:
                rows_strong.append((exp_fail, 1 - y))

    def report(label, data):
        print(f"\n{label} ({len(data)} attempts)")
        hdr = (f"{'expected_fail <':>16} {'n':>7} {'mean_expected':>13} "
               f"{'actual_fail':>12} {'actual/expected':>16}")
        print(hdr); print("-" * len(hdr))
        for thr in (0.01, 0.03, 0.05, 0.10, 0.20):
            sub = [d for d in data if d[0] < thr]
            if not sub:
                continue
            exp = sum(d[0] for d in sub) / len(sub)
            act = sum(d[1] for d in sub) / len(sub)
            ratio = (act / exp) if exp > 0 else float("inf")
            print(f"{thr:>15.0%} {len(sub):>7} {exp:>12.2%} "
                  f"{act:>11.2%} {ratio:>15.1f}x")

    lbl = f"{strong_rank - 19}d" if strong_rank >= 20 else f"{20 - strong_rank}k"
    print("\n" + "=" * 78)
    print(f"MISCLICK / LAPSE analysis — first {first_n} first-time attempts, "
          f"expected failure\nfrom each user's long-term rating (full-history "
          f"smoother at their last attempt).")
    print("=" * 78)
    report("ALL users", rows)
    report(f"STRONG players (long-term rank >= index {strong_rank} ~ {lbl})", rows_strong)
    print("\nIn the lowest expected-failure bin the actual rate ≈ the irreducible "
          "slip/lapse\nfloor. If actual >> expected, a 3PL upper-asymptote (lapse) "
          "term is warranted so a\nmisclick on a trivial problem stops cratering the "
          "live rank estimate.")


# ---------------------------------------------------------------------------
# Reference smoother R*(t): two-sided wide-kernel pooled 3PL MLE (rank tab)
# ---------------------------------------------------------------------------
def gaussian_weight(dt_over_tau):
    if abs(dt_over_tau) > KERNEL_TRUNC_SIGMA:
        return 0.0
    return math.exp(-0.5 * dt_over_tau * dt_over_tau)


def pooled_3pl_mle(attempts, t0, tau_ms, ts_sorted):
    """Kernel-weighted pooled 3PL MLE for rank R at t0 (port of poolLogistic).
    attempts: list of (t,x,slope,floor,kappa,correct); ts_sorted: the t column."""
    win = KERNEL_TRUNC_SIGMA * tau_ms
    lo = bisect_left(ts_sorted, t0 - win)
    hi = bisect_left(ts_sorted, t0 + win + 1)
    if hi <= lo:
        return float("nan"), 0.0

    sumW = sumWx = 0.0
    for j in range(lo, hi):
        w = gaussian_weight((attempts[j][0] - t0) / tau_ms)
        if w <= 0:
            continue
        sumW += w
        sumWx += w * attempts[j][1]
    if sumW <= 0:
        return float("nan"), 0.0
    R = sumWx / sumW

    for _ in range(15):
        score = info = 0.0
        for j in range(lo, hi):
            t, x, slope, floor, kappa, y = attempts[j][:6]
            w = gaussian_weight((t - t0) / tau_ms)
            if w <= 0:
                continue
            praw = sigmoid(slope * (R - x) + kappa)
            p = min(max(floor + (1 - floor) * praw, 1e-9), 1 - 1e-9)
            g = (1 - floor) * slope * praw * (1 - praw)
            score += w * (y - p) * g / (p * (1 - p))
            info += w * g * g / (p * (1 - p))
        if info < 1e-9:
            break
        step = max(-2.0, min(2.0, score / info))
        R += step
        R = max(X_CLAMP_LO, min(X_CLAMP_HI, R))
        if abs(step) < 1e-6:
            break

    A = B = 0.0
    for j in range(lo, hi):
        t, x, slope, floor, kappa, y = attempts[j][:6]
        w = gaussian_weight((t - t0) / tau_ms)
        if w <= 0:
            continue
        praw = sigmoid(slope * (R - x) + kappa)
        p = min(max(floor + (1 - floor) * praw, 1e-9), 1 - 1e-9)
        g = (1 - floor) * slope * praw * (1 - praw)
        v = g * g / (p * (1 - p))
        A += w * v
        B += w * w * v
    se = math.sqrt(B) / A if A > 0 else float("inf")
    return R, se


def reference_track(attempts):
    ts = [a[0] for a in attempts]
    return [pooled_3pl_mle(attempts, a[0], TAU_S_MS, ts)[0] for a in attempts]


# ---------------------------------------------------------------------------
# Causal estimators. Each: .rank (current index used to SERVE the next problem),
# .predict(x,slope,floor,kappa) -> P(clear) at the current rank, .update(att).
# ---------------------------------------------------------------------------
class Estimator:
    name = "base"
    lapse = 0.0          # 4PL upper-asymptote slip rate (0 = plain 3PL)

    @property
    def rank(self):
        raise NotImplementedError

    def predict(self, x, slope, floor, kappa):
        return p_clear(self.rank, x, slope, floor, kappa, self.lapse)

    def update(self, att):
        raise NotImplementedError


class MomentumEstimator(Estimator):
    """Port of pwa/js/user-stats.js: cold-start 20k, momentum ratchet. Works in
    normalized [0,1]; .rank exposes index = norm * TOTAL_RANKS. quality is the
    a_p^2 selection weight clamped to [0,1] (matching the live clamp).

    *** CONFOUNDED — do NOT treat its metrics as momentum's real accuracy. ***
    Momentum is a CLOSED-LOOP serving controller: production serves problems AT
    the momentum rank, then momentum reacts to the win/loss those problems
    produce. Replaying it on a LOGGED stream breaks the loop — the wins/losses
    here were generated at PRODUCTION's momentum (under the code + ratings live at
    the time), not at this replay's. Empirically the replay diverges hard: with
    the current clamp (sf~=1) it overshoots to ~5d for ~7k users who were served
    ~9k problems and cleared ~79%. Production keeps users near 80% (verified
    db-wide), so its live rank tracked the served level, ~10 ranks below this
    replay. Add the rating-scale drift between serve-time and now and momentum's
    absolute rank is simply not comparable to a current-ratings estimate. Kept
    only as a (heavily caveated) cold-start illustration; the valid comparison is
    among the read-only ability estimators (statespace / infowin / causalkernel),
    which observe ability from current ratings and don't drive serving."""
    name = "momentum"

    def __init__(self, loss_pre_cut=False):
        self.rank_norm = 0.0
        self.momentum = DEFAULT_MOMENTUM
        # loss_pre_cut: take the rank penalty using the PRE-cut momentum (symmetric
        # with the win branch, which uses the pre-growth momentum). The shipped
        # loop shrinks momentum to (m+1)/3 BEFORE penalizing, so the realized
        # penalty is ~2.9x a win instead of the intended 4x, equilibrating serving
        # at ~75% instead of 80%. This flag restores the 4x and should re-center.
        self.loss_pre_cut = loss_pre_cut

    @property
    def rank(self):
        return self.rank_norm * TOTAL_RANKS

    def update(self, att):
        y, a_p = att[5], att[6]
        rank, momentum = self.rank_norm, self.momentum
        # Client `quality` is the RAW discrimination squared (a_p^2, median ~6),
        # clamped to [0,1] -> ~1 for almost every problem. (Earlier this used the
        # IN-RANK slope^2 ~0.25, which made the ratchet climb ~4x too slowly and
        # left strong users stranded far below their level -> spurious huge RMSE.)
        quality = min(1.0, max(0.0, a_p * a_p))
        sf = quality                                   # first-time only -> no 1/5
        if y:
            d = (1.0 - rank) / 9.0 - MIN_RANK_INCREASE_PER_CORRECT
            max_inc = 0.5 * ((1.0 - rank) / 9.0 + MIN_RANK_INCREASE_PER_CORRECT
                             + math.sqrt(d * d + MIN_RANK_INCREASE_PER_CORRECT ** 2))
            inc = sf * momentum / 600.0
            minc = sf * 0.1
            if inc > max_inc:
                minc *= max_inc / inc
                inc = max_inc
            momentum *= 1 + minc
            rank += inc
        elif self.loss_pre_cut:
            rank = max(0.0, rank - 4 * sf * (momentum / 600.0))   # penalize first
            momentum -= sf * (momentum - (momentum + 1) / 3.0)    # then shrink
        else:
            momentum -= sf * (momentum - (momentum + 1) / 3.0)    # shipped: shrink
            rank = max(0.0, rank - 4 * sf * (momentum / 600.0))   # then penalize
        self.momentum = min(MOMENTUM_MAX, max(MOMENTUM_MIN, momentum
                            if math.isfinite(momentum) else DEFAULT_MOMENTUM))
        self.rank_norm = rank


def _newton_3pl(window, prior_mean, prior_info, R0, lapse=0.0):
    """Pooled 4PL MLE over `window` rows (x,slope,floor,kappa,y,weight) with a
    Gaussian prior N(prior_mean, 1/prior_info) on R. lapse is the upper-asymptote
    slip rate (0 = 3PL). Returns (R, total_info).

    With lapse>0 a failure on a far-below-rank problem has p(1-p) -> lapse(1-lapse)
    (bounded away from 0) while its gradient g -> 0, so its score contribution
    vanishes: the fit ignores misclicks on trivial problems instead of lurching."""
    upper = 1.0 - lapse
    R = R0
    for _ in range(25):
        score = prior_info * (prior_mean - R)
        info = prior_info
        for x, slope, floor, kappa, y, w in window:
            praw = sigmoid(slope * (R - x) + kappa)
            p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
            g = (upper - floor) * slope * praw * (1 - praw)
            score += w * (y - p) * g / (p * (1 - p))
            info += w * g * g / (p * (1 - p))
        if info < 1e-12:
            break
        step = max(-2.0, min(2.0, score / info))
        R += step
        R = max(X_CLAMP_LO, min(X_CLAMP_HI, R))
        if abs(step) < 1e-6:
            break
    return R, info


class InfoWindowEstimator(Estimator):
    """Fixed-INFORMATION-window causal 3PL MLE (the proposal).

    Walks back from the newest attempt accumulating per-attempt information
    weight a_p^2 (= slope_in_rank^2 * |M'|^2 ... here we use slope^2 directly, the
    in-rank Fisher proxy that selection already uses) until it reaches I_target,
    optionally also bounded by a recency cap tau_max_days (a pure info window has
    no time localization, so stale data needs a cap). Then MLE over that window
    with a Gaussian prior. SE = 1/sqrt(total info), ~constant by construction."""

    def __init__(self, I_target, prior_mean_fn, prior_info, tau_max_days=None, lapse=0.0):
        self.I_target = I_target
        self.prior_info = prior_info
        self.prior_mean = prior_mean_fn()
        self.tau_max_ms = tau_max_days * DAY_MS if tau_max_days else None
        self.lapse = lapse
        self.hist = []   # (t,x,slope,floor,kappa,y)
        self._R = self.prior_mean
        self.name = "infowin"

    @property
    def rank(self):
        return self._R

    def update(self, att):
        self.hist.append(att)
        t_now = att[0]
        upper = 1.0 - self.lapse
        window = []
        acc_info = 0.0
        R_ref = self._R   # size the window by Fisher info at the current estimate
        for a in reversed(self.hist):
            if self.tau_max_ms is not None and (t_now - a[0]) > self.tau_max_ms:
                break
            _, x, slope, floor, kappa, y = a[:6]
            window.append((x, slope, floor, kappa, y, 1.0))
            # Fisher info of this 4PL observation about R: (dP/dR)^2 / (P(1-P)).
            # (Not slope^2 alone — that omits the P(1-P)~0.16 factor at 80%-clear
            # and undersizes the window ~6x.)
            praw = sigmoid(slope * (R_ref - x) + kappa)
            p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
            g = (upper - floor) * slope * praw * (1 - praw)
            acc_info += g * g / (p * (1 - p))
            if acc_info >= self.I_target:
                break
        self._R, _ = _newton_3pl(window, self.prior_mean, self.prior_info, self._R, self.lapse)


class StateSpaceEstimator(Estimator):
    """Random-walk skill + 3PL observation, causal Laplace/EKF filter.

    The 'fixed information' idea in its principled form: the effective window
    emerges from the ratio process-noise : per-attempt information. q is the
    skill drift variance (rank^2/day); prior is N(prior_mean, prior_var)."""

    def __init__(self, q_per_day, prior_mean_fn, prior_var, lapse=0.0, var_floor=0.0):
        self.q = q_per_day
        self.R = prior_mean_fn()
        self.var = prior_var
        self.lapse = lapse
        self.var_floor = var_floor   # keep the filter nimble: don't let it get so
        self.last_t = None           # confident it can't climb to a far-off ability
        self.name = "statespace"

    @property
    def rank(self):
        return self.R

    def update(self, att):
        t, x, slope, floor, kappa, y = att[:6]
        upper = 1.0 - self.lapse
        if self.last_t is not None:
            self.var += self.q * max(0.0, (t - self.last_t) / DAY_MS)   # process noise
        self.last_t = t
        # Laplace update: maximize prior N(R_pred, var) + 4PL likelihood of this obs.
        R_pred, prec0 = self.R, 1.0 / max(self.var, 1e-9)
        R = R_pred
        for _ in range(25):
            praw = sigmoid(slope * (R - x) + kappa)
            p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
            g = (upper - floor) * slope * praw * (1 - praw)
            score = prec0 * (R_pred - R) + (y - p) * g / (p * (1 - p))
            info = prec0 + g * g / (p * (1 - p))
            step = max(-2.0, min(2.0, score / info))
            R += step
            R = max(X_CLAMP_LO, min(X_CLAMP_HI, R))
            if abs(step) < 1e-6:
                break
        self.R = R
        self.var = max(1.0 / info, self.var_floor)


class CausalKernelEstimator(Estimator):
    """Baseline: the rank-tab wide kernel run ONE-SIDED (past only) — what the
    smoother would give as a live filter at a fixed time bandwidth."""

    def __init__(self, tau_days):
        self.tau_ms = tau_days * DAY_MS
        self.hist = []
        self._R = 0.0
        self.name = "causalkernel"

    @property
    def rank(self):
        return self._R

    def update(self, att):
        self.hist.append(att)
        t0 = att[0]
        # one-sided: only weight past attempts
        window = []
        for a in self.hist:
            w = gaussian_weight((a[0] - t0) / self.tau_ms)
            if w <= 0:
                continue
            window.append((a[1], a[2], a[3], a[4], a[5], w))
        if window:
            self._R, _ = _newton_3pl(window, 0.0, 0.0, self._R)


def run_shock(strong_rank, lapse):
    """Controlled misclick-robustness test (answers: does an MLE filter overreact
    to a slip on a trivial problem, the way momentum is suspected NOT to?).

    Each estimator is first made confident at rank R0 = strong_rank by feeding 60
    at-level problems (x=R0) at the 80% clear rate, hourly. Then a single MISCLICK
    is injected: a failure on a problem `d` ranks below the user (expected failure
    ~0). We report the immediate rank DROP and how many subsequent trivial clears
    it takes to recover, for d in {6, 12} ranks below. A lapse term (4PL upper
    asymptote) is expected to shrink the drop, since the failure stops looking
    near-impossible.

    Momentum is included but ITS warm-up doesn't land at R0 (it has no rank-anchor
    and overshoots — see its class caveat); its drop is reported for context but
    isn't an apples-to-apples rank, just 'how many ranks does one loss cost it'."""
    SL, FL = 0.5, 0.5             # representative in-rank slope + floor
    KA = clear_kappa(FL)
    R0 = float(strong_rank)

    def warm(est):
        t = 0
        for i in range(60):
            y = 0 if (i % 5 == 4) else 1          # 80% clear at-level
            est.update((t, R0, SL, FL, KA, y, 4.0))
            t += HOUR_MS
        return est, t

    def shock(make, d):
        est, t = warm(make())
        before = est.rank
        est.update((t, R0 - d, SL, FL, KA, 0, 4.0)); t += HOUR_MS   # the misclick
        after = est.rank
        rec = 0
        for _ in range(300):                       # recover on trivial clears
            if est.rank >= before - 0.1:
                break
            est.update((t, R0 - d, SL, FL, KA, 1, 4.0)); t += HOUR_MS; rec += 1
        return before, before - after, rec

    cands = [
        ("momentum", lambda: MomentumEstimator()),
        ("statespace (3PL, no lapse)", lambda: StateSpaceEstimator(0.01, lambda: R0, 4.0)),
        (f"statespace (4PL lapse={lapse:g})", lambda: StateSpaceEstimator(0.01, lambda: R0, 4.0, lapse=lapse)),
        ("infowin I8 (3PL, no lapse)", lambda: InfoWindowEstimator(8.0, lambda: R0, 8.0, 120)),
        (f"infowin I8 (4PL lapse={lapse:g})", lambda: InfoWindowEstimator(8.0, lambda: R0, 8.0, 120, lapse=lapse)),
    ]
    print("\n" + "=" * 78)
    print(f"MISCLICK SHOCK TEST — estimator confident at index {strong_rank}, then ONE")
    print("failure on a problem d ranks below (expected failure ~0). Drop & recovery:")
    print("=" * 78)
    exp6 = 1 - p_clear(R0, R0 - 6, SL, FL, KA)
    exp12 = 1 - p_clear(R0, R0 - 12, SL, FL, KA)
    hdr = (f"{'estimator':<32} {'drop@d=6':>9} {'rec@6':>6} {'drop@d=12':>10} {'rec@12':>7}")
    print(f"(expected failure: d=6 -> {exp6:.2%}, d=12 -> {exp12:.3%})")
    print(hdr); print("-" * len(hdr))
    for label, make in cands:
        b6, drop6, rec6 = shock(make, 6)
        b12, drop12, rec12 = shock(make, 12)
        print(f"{label:<32} {drop6:>8.2f}r {rec6:>6} {drop12:>9.2f}r {rec12:>7}")
    print("\nDrop = ranks lost to ONE misclick; rec = trivial clears needed to recover.")
    print("A big drop on d=12 (a ~0%-failure problem) is the overreaction to guard against;")
    print("the lapse term should shrink it toward momentum's bounded, fast-recovering step.")


# ---------------------------------------------------------------------------
# Forward simulation: the estimator DRIVES serving (closed loop)
# ---------------------------------------------------------------------------
# Replaying historically-served problems is invalid for comparing live
# estimators: those problems were picked under the ratings of the time, so scored
# by TODAY's ratings they span a wide difficulty range (~6 ranks RMS from the
# user's level). A production estimator serves at ITS OWN current rank estimate,
# so it only ever sees problems clustered ~1-2 ranks wide — far less information.
# The replay therefore flatters every estimator and can't close the loop momentum
# lives in. Here instead: a simulated user with a KNOWN true ability + the
# measured lapse rate, and the estimator picks each problem from a realistic pool
# near its current estimate (the production policy). This induces the narrow
# distribution the estimator actually gets, and lets momentum run in its own loop.

def load_problem_pool(corpus, size):
    """The FULL corpus as (x, slope, floor, kappa, a_p), sorted by x. Using all
    ~123k problems (not a sample) is essential: the live policy serves the 50
    nearest, which out of 123k is a ~0.01-rank-wide window — a near-point, not the
    1-2 ranks a sparse sample would fake."""
    sz = str(size)
    pool = []
    for tid in corpus.ratings[sz]:
        feat = corpus.attempt_features(tid, size)
        if feat is not None:
            pool.append(feat)            # (x, slope, floor, kappa, a_p)
    pool.sort(key=lambda p: p[0])
    return pool, [p[0] for p in pool]


def simulate_user(est, theta, pool, xs, n_attempts, lapse, serve_k, rng, dt_h=4):
    """Run one simulated user (no problem repeats). Live serving policy: of the
    serve_k problems nearest the estimator's current rank, pick the HIGHEST-slope
    (a_p) one — narrow in rank but maximally informative. The user's outcome is
    drawn from the true 4PL at ability `theta` with slip rate `lapse`. Returns
    per-attempt |R_est - theta|, served x, and outcomes."""
    t = 0
    served = set()
    rest_err, served_x, correct = [], [], []
    W = max(serve_k * 3, 150)            # index half-window to find serve_k unserved
    n = len(pool)
    for _ in range(n_attempts):
        R = est.rank
        j = bisect_left(xs, R)
        lo = max(0, j - W); hi = min(n, j + W)
        cand = [k for k in range(lo, hi) if k not in served]
        cand.sort(key=lambda k: abs(pool[k][0] - R))
        cand = cand[:serve_k]                       # serve_k nearest in rank
        k = max(cand, key=lambda k: pool[k][1])     # highest slope among them
        served.add(k)
        x, slope, floor, kappa, a_p = pool[k]
        p_true = p_clear(theta, x, slope, floor, kappa, lapse)
        y = 1 if rng.random() < p_true else 0
        rest_err.append(abs(R - theta)); served_x.append(x); correct.append(y)
        est.update((t, x, slope, floor, kappa, y, a_p))
        t += dt_h * HOUR_MS
    return rest_err, served_x, correct


def run_simulate(corpus, abilities, reps, n_attempts, lapse, prior, size, seed):
    """Closed-loop forward sim across a grid of true abilities. For each estimator
    × ability, reports onboarding speed (attempts to get within 1 rank of truth),
    steady accuracy (RMSE of estimate vs true ability, last half), the realized
    clear rate (should sit ~80% if serving is on-target), and the served-difficulty
    spread (confirms the narrow distribution, unlike the historical replay)."""
    rng = random.Random(seed)
    random.seed(seed)
    pool, xs = load_problem_pool(corpus, size)
    print(f"\nsimulated problem pool: {len(pool)} problems (FULL corpus, "
          f"x range {xs[0]:.1f}..{xs[-1]:.1f})", file=sys.stderr)

    # Estimator factories. Prior-based ones start at `prior` (the population mean a
    # production app would use); momentum cold-starts at 20k as in production.
    makers = {
        "momentum":                  lambda: MomentumEstimator(),
        "statespace_pop":            lambda: StateSpaceEstimator(0.02, lambda: prior, 9.0),
        # adaptive variants: keep the filter from over-committing so it can still
        # climb to a far-off true ability (the strong-user onboarding stall).
        "statespace_vfloor0.5":      lambda: StateSpaceEstimator(0.02, lambda: prior, 9.0, lapse=lapse, var_floor=0.5),
        "statespace_vfloor1.0":      lambda: StateSpaceEstimator(0.02, lambda: prior, 9.0, lapse=lapse, var_floor=1.0),
        "infowin_pop_lapse":         lambda: InfoWindowEstimator(8.0, lambda: prior, 3.0, 120, lapse=lapse),
    }
    warm = n_attempts // 2

    print("\n" + "=" * 92)
    print(f"FORWARD SIMULATION — estimator drives serving; true user ability known; "
          f"slip rate {lapse:g}.")
    print(f"prior start = index {prior:g}; {reps} reps/ability; {n_attempts} attempts each.")
    print("=" * 92)
    hdr = (f"{'estimator':<22} {'true_rank':>9} {'conv_att':>9} {'steady_RMSE':>12} "
           f"{'clear_rate':>11} {'served_std':>11}")
    print(hdr); print("-" * len(hdr))
    for name, make in makers.items():
        for theta in abilities:
            convs, rmses, clears, spreads = [], [], [], []
            for _ in range(reps):
                est = make()
                err, sx, corr = simulate_user(est, float(theta), pool, xs,
                                              n_attempts, lapse, 50, rng)
                # onboarding: first attempt within 1 rank of truth
                c = next((k for k, e in enumerate(err) if e <= 1.0), n_attempts)
                convs.append(c)
                rmses.append(math.sqrt(sum(e * e for e in err[warm:]) / max(1, len(err[warm:]))))
                clears.append(sum(corr[warm:]) / max(1, len(corr[warm:])))
                mu = sum(sx[warm:]) / max(1, len(sx[warm:]))
                spreads.append(math.sqrt(sum((v - mu) ** 2 for v in sx[warm:]) / max(1, len(sx[warm:]))))
            convs.sort()
            print(f"{name:<22} {theta:>9} {convs[len(convs)//2]:>9} "
                  f"{sum(rmses)/len(rmses):>12.2f} {sum(clears)/len(clears):>11.2%} "
                  f"{sum(spreads)/len(spreads):>11.2f}")
        print()
    print("conv_att = median attempts to first get within 1 rank of TRUE ability.")
    print("steady_RMSE = estimate error over the last half (ranks). clear_rate ~0.80 = on-target.")
    print("served_std = realized spread of served difficulty (ranks) — tiny here (50-nearest of")
    print("the full corpus + highest-slope pick), vs ~6 in the historical replay that")
    print("overstated accuracy. Convergence now rides almost entirely on the clear-rate signal.")


# ---------------------------------------------------------------------------
# Estimator config registry
# ---------------------------------------------------------------------------
def build_configs(pop_mean, survey_priors, lapse=0.0):
    """Return {name: factory(user_id) -> Estimator}. Prior means resolved per
    user: 'pop' = cohort population mean, 'survey' = the user's declaration (pop
    fallback). The two front-runners also get a `_lapse` variant (4PL upper
    asymptote = `lapse`) so the sweep shows the lapse term costs nothing on the
    normal metrics while (per the shock test) curing the misclick fragility."""
    def pop_fn(_uid):
        return lambda: pop_mean

    def survey_fn(uid):
        return lambda: survey_priors.get(uid, pop_mean)

    cfgs = {}
    cfgs["momentum"] = lambda uid: MomentumEstimator()
    cfgs["causalkernel_14d"] = lambda uid: CausalKernelEstimator(14)

    # Information-window sweep × prior. prior_info in pseudo-Fisher units
    # (~slope^2 per pseudo-attempt; typical slope^2 ~ a few).
    for I in (2.0, 4.0, 8.0, 16.0):
        cfgs[f"infowin_I{int(I)}_pop"] = (
            lambda uid, I=I: InfoWindowEstimator(I, pop_fn(uid), prior_info=3.0, tau_max_days=120))
        cfgs[f"infowin_I{int(I)}_survey"] = (
            lambda uid, I=I: InfoWindowEstimator(I, survey_fn(uid), prior_info=8.0, tau_max_days=120))

    # State-space sweep × prior. q in rank^2/day; prior_var in rank^2.
    for q in (0.002, 0.01, 0.05):
        tag = f"{q:g}".replace(".", "p")
        cfgs[f"statespace_q{tag}_pop"] = (
            lambda uid, q=q: StateSpaceEstimator(q, pop_fn(uid), prior_var=9.0))
        cfgs[f"statespace_q{tag}_survey"] = (
            lambda uid, q=q: StateSpaceEstimator(q, survey_fn(uid), prior_var=2.0))

    if lapse > 0:
        cfgs[f"statespace_q0p05_pop_lapse"] = (
            lambda uid: StateSpaceEstimator(0.05, pop_fn(uid), prior_var=9.0, lapse=lapse))
        cfgs[f"infowin_I8_pop_lapse"] = (
            lambda uid: InfoWindowEstimator(8.0, pop_fn(uid), prior_info=3.0, tau_max_days=120, lapse=lapse))
    return cfgs


# ---------------------------------------------------------------------------
# Per-user evaluation
# ---------------------------------------------------------------------------
def eval_user(attempts, ref, estimator, n_early, conv_eps):
    """Replay one user. Returns per-phase metrics dict, or None if too short.

    At attempt i the estimator SERVES using its current rank (PAST only), so we
    score the prediction before folding the outcome in (true one-step-ahead)."""
    n = len(attempts)
    if n < n_early + 5:
        return None
    R_final = ref[-1]
    if not math.isfinite(R_final):
        return None

    early_ll = early_n = 0.0
    steady_ll = steady_n = 0.0
    steady_sq = 0.0          # (R_serve - R*)^2 over steady phase
    conv_attempt = None
    conv_days = None
    t0 = attempts[0][0]

    for i, att in enumerate(attempts):
        x, slope, floor, kappa, y = att[1], att[2], att[3], att[4], att[5]
        p = estimator.predict(x, slope, floor, kappa)
        ll = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        R_serve = estimator.rank
        # convergence to the user's FINAL smoothed level
        if conv_attempt is None and abs(R_serve - R_final) <= conv_eps:
            conv_attempt = i
            conv_days = (att[0] - t0) / DAY_MS
        if i < n_early:
            early_ll += ll
            early_n += 1
        else:
            steady_ll += ll
            steady_n += 1
            ri = ref[i]
            if math.isfinite(ri):
                steady_sq += (R_serve - ri) ** 2
        estimator.update(att)

    return {
        "early_logloss": early_ll / early_n if early_n else float("nan"),
        "steady_logloss": steady_ll / steady_n if steady_n else float("nan"),
        "steady_rmse": math.sqrt(steady_sq / steady_n) if steady_n else float("nan"),
        "conv_attempts": conv_attempt if conv_attempt is not None else n,   # censored = n
        "conv_days": conv_days if conv_days is not None else (attempts[-1][0] - t0) / DAY_MS,
        "converged": conv_attempt is not None,
    }


def aggregate(rows):
    """Mean + median over per-user metric dicts."""
    if not rows:
        return {}
    keys = ("early_logloss", "steady_logloss", "steady_rmse", "conv_attempts", "conv_days")
    out = {"n_users": len(rows),
           "frac_converged": sum(r["converged"] for r in rows) / len(rows)}
    for k in keys:
        vals = sorted(v for r in rows if math.isfinite(v := r[k]))
        if not vals:
            out[f"{k}_mean"] = out[f"{k}_median"] = float("nan")
            continue
        out[f"{k}_mean"] = sum(vals) / len(vals)
        out[f"{k}_median"] = vals[len(vals) // 2]
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--new-user-after", default="2025-09-15",
                    help="keep users whose first attempt is strictly after this (date or ISO)")
    ap.add_argument("--min-attempts", type=int, default=40)
    ap.add_argument("--max-users", type=int, default=400,
                    help="cap (0 = all); keeps the most-active users")
    ap.add_argument("--max-attempts", type=int, default=400,
                    help="truncate each user to their first N attempts (0 = all); "
                         "bounds the O(n^2) smoother on dense power users")
    ap.add_argument("--n-early", type=int, default=20,
                    help="first N attempts counted as the onboarding phase")
    ap.add_argument("--conv-eps", type=float, default=1.0,
                    help="rank tolerance for 'converged to final level' (ranks)")
    ap.add_argument("--only", default="",
                    help="comma-separated estimator names to run (default all)")
    ap.add_argument("--max-serve-rms", type=float, default=0.0,
                    help="trust filter: drop users whose served problems sit > this many "
                         "ranks (RMS) from the momentum target under current ratings "
                         "(0 = keep all). These were served under stale ratings.")
    ap.add_argument("--by-cohort", action="store_true",
                    help="also print the estimator table split by onboarding month")
    ap.add_argument("--misclick", action="store_true",
                    help="run ONLY the misclick/lapse-rate analysis (first-N attempts, "
                         "actual vs expected failure on trivially-easy problems) and exit")
    ap.add_argument("--first-n", type=int, default=30,
                    help="attempts per user counted as onboarding for --misclick")
    ap.add_argument("--strong-rank", type=int, default=20,
                    help="long-term internal-rank index for the 'strong player' subset "
                         "(20 = 1d, 24 = 5d)")
    ap.add_argument("--shock", action="store_true",
                    help="run ONLY the controlled misclick-robustness shock test and exit "
                         "(synthetic; needs no DB)")
    ap.add_argument("--simulate", action="store_true",
                    help="run ONLY the closed-loop forward simulation (estimator drives "
                         "serving; simulated users with known ability) and exit")
    ap.add_argument("--sim-reps", type=int, default=80, help="simulated users per ability")
    ap.add_argument("--sim-attempts", type=int, default=160, help="attempts per simulated user")
    ap.add_argument("--sim-abilities", default="6,11,16,20,24",
                    help="comma-separated true internal-rank indices to simulate")
    ap.add_argument("--sim-prior", type=float, default=13.0,
                    help="prior-start index for the prior-based estimators (population mean)")
    ap.add_argument("--lapse", type=float, default=0.01,
                    help="4PL upper-asymptote slip rate used by the lapse-aware variants "
                         "(shock test + *_lapse configs); ~the measured strong-player floor")
    ap.add_argument("--csv", default="", help="write per-user rows here")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)   # show tables as they print
    except AttributeError:
        pass

    if args.shock:                       # synthetic; no DB / corpus needed
        run_shock(args.strong_rank, args.lapse)
        return

    print(f"loading corpus from {os.path.join(PROJECT_ROOT, 'ratings')} ...", file=sys.stderr)
    corpus = Corpus(PROJECT_ROOT)

    if args.simulate:                    # closed loop; corpus only, no DB
        abilities = [int(s) for s in args.sim_abilities.split(",")]
        run_simulate(corpus, abilities, args.sim_reps, args.sim_attempts,
                     args.lapse, args.sim_prior, 9, seed=12345)
        return

    conn = sqlite3.connect(args.db)

    print(f"loading users (new after {args.new_user_after}, >= {args.min_attempts} attempts) ...",
          file=sys.stderr)
    users, onboarded = load_users(conn, corpus, args.new_user_after, args.min_attempts,
                                  args.max_users, args.max_attempts)
    survey_priors = load_survey_priors(conn, corpus)
    conn.close()
    if not users:
        print("no users matched filters", file=sys.stderr)
        return

    if args.misclick:
        run_misclick(users, args.first_n, args.strong_rank)
        return

    # Reference smoother R*(t) for every loaded user (needed by both the drift
    # diagnostic and the estimator metrics).
    print(f"computing reference smoother for {len(users)} users ...", file=sys.stderr)
    refs = {uid: reference_track(att) for uid, att in users.items()}

    # Serving-accuracy diagnostic (how much to trust each user) + onboarding cohort.
    drift = {uid: serving_drift(att, refs[uid]) for uid, att in users.items()}
    cohort = {uid: cohort_month(onboarded[uid]) for uid in users}

    # --- serving accuracy BY ONBOARDING COHORT (the staleness check) ---
    # Early cohorts were served under less-accurate ratings, so by today's
    # ratings their served problems sit farther from where the user's own results
    # place them. Bigger serve_drift_rms => trust those users less as test cases.
    cmonths = sorted(set(cohort.values()))
    print("\nServing accuracy by onboarding cohort (served rank vs reference ability, "
          "current ratings):")
    chdr = (f"{'cohort':<9} {'users':>5} {'med_drift_rms':>13} {'med_drift_bias':>14} "
            f"{'med_clear_rate':>14}")
    print(chdr); print("-" * len(chdr))
    for cm in cmonths:
        us = [u for u in users if cohort[u] == cm]
        rms = sorted(drift[u]["serve_drift_rms"] for u in us if math.isfinite(drift[u]["serve_drift_rms"]))
        bias = sorted(drift[u]["serve_drift_mean"] for u in us if math.isfinite(drift[u]["serve_drift_mean"]))
        clr = sorted(drift[u]["clear_rate"] for u in us if math.isfinite(drift[u]["clear_rate"]))
        if not rms:
            continue
        print(f"{cm:<9} {len(us):>5} {rms[len(rms)//2]:>13.2f} "
              f"{bias[len(bias)//2]:>14.2f} {clr[len(clr)//2]:>14.2f}")
    print("drift = served problem's current rank - reference ability R*(t) (ranks). "
          "Larger RMS\n=> served problems inconsistent with current ratings "
          "=> onboarded under stale ratings => weaker test case.")

    # Trust filter: optionally drop users whose served stream is too stale.
    if args.max_serve_rms > 0:
        keep = [u for u in users if math.isfinite(drift[u]["serve_drift_rms"])
                and drift[u]["serve_drift_rms"] <= args.max_serve_rms]
        print(f"\ntrust filter --max-serve-rms {args.max_serve_rms}: "
              f"{len(keep)}/{len(users)} users kept", file=sys.stderr)
        users = {u: users[u] for u in keep}
        if not users:
            print("no users left after trust filter", file=sys.stderr)
            return

    # Population prior = median final smoothed level over the kept users.
    finals = sorted(refs[u][-1] for u in users if math.isfinite(refs[u][-1]))
    pop_mean = finals[len(finals) // 2] if finals else 10.0
    n_survey = sum(1 for u in users if u in survey_priors)
    print(f"{len(users)} users | population prior = {pop_mean:.2f} (idx) "
          f"| {n_survey} have a survey prior", file=sys.stderr)

    configs = build_configs(pop_mean, survey_priors, args.lapse)
    if args.only:
        want = set(s.strip() for s in args.only.split(","))
        configs = {k: v for k, v in configs.items() if k in want}
        missing = want - set(configs)
        if missing:
            print(f"unknown estimator(s): {sorted(missing)}", file=sys.stderr)

    # Run every estimator over every user once; keep per-user metric rows so we
    # can aggregate overall AND per cohort without re-running.
    per_user = defaultdict(dict)   # uid -> {name: metrics}
    per_user_rows = []
    for name, factory in configs.items():
        for uid, attempts in users.items():
            est = factory(uid)
            m = eval_user(attempts, refs[uid], est, args.n_early, args.conv_eps)
            if m is None:
                continue
            per_user[uid][name] = m
            if args.csv:
                per_user_rows.append({
                    "estimator": name, "user_id": uid, "onboard_month": cohort[uid],
                    "n_attempts": len(attempts),
                    "serve_drift_rms": round(drift[uid]["serve_drift_rms"], 3),
                    "serve_drift_bias": round(drift[uid]["serve_drift_mean"], 3),
                    "clear_rate": round(drift[uid]["clear_rate"], 3), **m})

    def print_table(title, uid_subset):
        results = {}
        for name in configs:
            rows = [per_user[u][name] for u in uid_subset if name in per_user[u]]
            results[name] = aggregate(rows)
        hdr = (f"{'estimator':<26} {'users':>5} {'early_LL':>9} {'steady_LL':>10} "
               f"{'steady_RMSE':>11} {'conv_att(med)':>13} {'conv_day(med)':>13} {'%conv':>6}")
        print(f"\n{title}\n{hdr}\n" + "-" * len(hdr))
        for name in sorted(results, key=lambda n: results[n].get("steady_logloss_mean", 9e9)):
            r = results[name]
            if not r:
                continue
            print(f"{name:<26} {r['n_users']:>5} "
                  f"{r['early_logloss_mean']:>9.4f} {r['steady_logloss_mean']:>10.4f} "
                  f"{r['steady_rmse_mean']:>11.3f} {r['conv_attempts_median']:>13.0f} "
                  f"{r['conv_days_median']:>13.1f} {100*r['frac_converged']:>5.0f}%")

    print_table("ALL cohorts (sorted by steady log-loss = goal 1):", list(per_user))
    if args.by_cohort:
        for cm in cmonths:
            us = [u for u in per_user if cohort.get(u) == cm]
            if us:
                print_table(f"cohort {cm} ({len(us)} users):", us)

    print("\nGoal 1 (long-term serving): lower steady_LL / steady_RMSE is better.")
    print("Goal 2 (onboarding speed):  lower early_LL / conv_attempts / conv_days is better.")
    print("Log-loss is assumption-free; RMSE/conv are vs the full-history smoother reference.")
    print("Compare cohorts: a config that only wins on stale early cohorts isn't a real win.")
    print("\n*** momentum is a CLOSED-LOOP serving controller and CANNOT be faithfully")
    print("    replayed offline (it overshoots on a logged stream it didn't itself serve);")
    print("    its row is illustrative only. The valid comparison is among the read-only")
    print("    ability estimators (statespace / infowin / causalkernel).")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_user_rows[0].keys()))
            w.writeheader()
            w.writerows(per_user_rows)
        print(f"\nwrote {len(per_user_rows)} rows to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
