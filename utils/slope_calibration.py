"""Client-replica walk-forward slope-scale calibration — the VALIDATION harness.

2026-07-18 finding (docs/scalar_weakness_axes.md follow-ups 2-4): the response
curves are steeper in gap units than the shipped per-problem slope claims —
a single global multiplier s on the inner logit is worth more held-out than
any of the residual models. This module estimates that multiplier under the
CLIENT's operating condition: per user, first-time attempts in time order; at
each EVAL_BLOCK past WARMUP, fit theta by MLE on the trailing WINDOW under the
production 4PL
(pwa/js/irt.js: P = c + (1 - c - LAPSE) * sigmoid(a*(theta - rating) + kappa(c)))
and score the NEXT block. LL(s) is maximized by golden-section. Certificate:
on synthetic outcomes generated from the true parameters the criterion returns
s = 0.993 (unbiased for its estimand).

NO LONGER the production stage (2026-07-25): pipeline.train_rating's phase 2
(fit_global_scales) fits the global slope+floor scales in-sample on the frozen
model with the ability field refit — covering EVERY first-time attempt, where
this walk-forward's causal protocol can only score heavy users' post-warmup
rows (~62% of attempt mass). The two criteria agree within the plateau
(walk-forward 1.667 vs in-sample free-theta 1.71-1.77, <0.1 mnat apart). The
old rationale here — "in-sample refitting re-absorbs the attenuation to s = 1
because the trainer's noisy theta biased discrimination down (EIV)" — is
measured-WRONG on both counts: the multiplier is dropout-induced rating-scale
compression (parametric bootstrap, 2026-07-24), and an in-sample refit does
NOT re-absorb it (frozen-theta 1.62, biased low by fixed-point-estimate
attenuation; free-theta profile likelihood 1.71-1.77, certificate-clean at
1.01-1.02 on synthetic s=1 truth). This module stays as the client-condition
cross-check and the harness for phase/attenuation studies.

Two application modes (docs follow-up: the naive "scale ratings around a
center" is gauge — absorbed exactly by each user's free theta — so the only
distinguishable rating-side variant is re-deriving the anchor):
  mode 'slope':  inner = s*a*(theta - rating) + kappa(c)
                 (shipped rating STAYS the 80%-clear point; apply as slope *= s)
  mode 'rating': inner = s*(a*(theta - rating) + kappa(c))
                 (the 80%-point migrates toward the 50%-point; apply as
                 slope *= s AND rating += kappa(c)*(1 - s)/(s*a))
"""

import json
import os
from multiprocessing import Pool

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Client-replica walk-forward protocol (same constants as the aux axis
# batteries — auxiliary/eval_score_gap_axis.py).
WINDOW = 500
EVAL_BLOCK = 50
WARMUP = 120
MIN_USER_ATTEMPTS = 50
FLOOR_MAX = 0.78          # pwa/js/irt.js clamp
CALIBRATION_P = 0.8

_SEQS = None  # worker global (fork-shared)


def load_client_lapse(fallback):
    """The client's LAPSE (population_constants.fat_finger_rate); falls back to
    the trainer's just-learned ceiling when constants aren't fit yet."""
    path = os.path.join(PROJECT_ROOT, "ratings", "population_constants.json")
    try:
        with open(path) as f:
            return float(json.load(f)["fat_finger_rate"])
    except (OSError, KeyError, ValueError):
        return float(fallback)


def p4pl(theta, R0, A, C, span, kap):
    z = np.clip(A * (theta - R0) + kap, -35, 35)
    return np.clip(C + span / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)


def fit_theta(R0, A, C, span, kap, y, grid):
    P = p4pl(grid[:, None], R0[None, :], A[None, :], C[None, :],
             span[None, :], kap[None, :])
    ll = (y[None, :] * np.log(P) + (1 - y[None, :]) * np.log(1 - P)).sum(axis=1)
    i = int(np.argmax(ll))
    if 0 < i < len(grid) - 1:
        d = (ll[i - 1] - ll[i + 1]) / (2 * (ll[i - 1] - 2 * ll[i] + ll[i + 1]) + 1e-12)
        return float(grid[i] + np.clip(d, -1, 1) * (grid[1] - grid[0]))
    return float(grid[i])


def _build_seqs(df, params_by_tid, lapse):
    """Per-user precomputed arrays for users with >= MIN_USER_ATTEMPTS usable
    first-time attempts, time-ordered. df: [user_id, problem, success,
    created_at]; params_by_tid: tid -> (rating, slope, floor)."""
    df = df[df["problem"].isin(params_by_tid)].sort_values("created_at")
    seqs = []
    for _u, grp in df.groupby("user_id", sort=False):
        if len(grp) < max(MIN_USER_ATTEMPTS, WARMUP + EVAL_BLOCK):
            continue
        pr = [params_by_tid[t] for t in grp["problem"]]
        R0 = np.array([p[0] for p in pr])
        A = np.array([p[1] for p in pr])
        C = np.clip(np.array([p[2] for p in pr]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        y = grp["success"].to_numpy(dtype=float)
        seqs.append((R0, A, C, span, kap, y))
    return seqs


def _init_pool(seqs):
    global _SEQS
    _SEQS = seqs


def _user_ll(args):
    idx, s, mode, glo, ghi = args
    R0, A, C, span, kap, y = _SEQS[idx]
    As = s * A
    kaps = kap if mode == "slope" else s * kap
    grid = np.arange(glo, ghi, 0.05)
    ll = 0.0
    n = 0
    for t0 in range(WARMUP, len(y), EVAL_BLOCK):
        lo = max(0, t0 - WINDOW)
        hi = min(t0 + EVAL_BLOCK, len(y))
        th = fit_theta(R0[lo:t0], As[lo:t0], C[lo:t0], span[lo:t0],
                       kaps[lo:t0], y[lo:t0], grid)
        P = p4pl(th, R0[t0:hi], As[t0:hi], C[t0:hi], span[t0:hi], kaps[t0:hi])
        ll += float((y[t0:hi] * np.log(P) + (1 - y[t0:hi]) * np.log(1 - P)).sum())
        n += hi - t0
    return ll, n


def fit_slope_scale(df, params_by_tid, lapse, mode, lo=0.9, hi=2.0, tol=0.01,
                    workers=None):
    """Golden-section maximize held-out walk-forward LL over the global
    multiplier s. Returns {'s', 'gain_mnat_per_attempt', 'n_heldout',
    'n_users', 'mode'} (gain vs s = 1)."""
    seqs = _build_seqs(df, params_by_tid, lapse)
    if not seqs:
        raise ValueError("no users with enough attempts for slope calibration")
    all_r0 = np.concatenate([s0[0] for s0 in seqs])
    glo, ghi = all_r0.min() - 3.0, all_r0.max() + 3.0
    workers = workers or min(len(seqs), os.cpu_count() or 1)
    pool = Pool(workers, initializer=_init_pool, initargs=(seqs,))
    cache = {}

    def ll_of(s):
        s = round(float(s), 6)
        if s not in cache:
            out = pool.map(_user_ll, [(i, s, mode, glo, ghi)
                                      for i in range(len(seqs))], chunksize=8)
            cache[s] = (sum(o[0] for o in out), sum(o[1] for o in out))
        return cache[s]

    try:
        inv = (np.sqrt(5) - 1) / 2
        a, b = lo, hi
        c1, c2 = b - inv * (b - a), a + inv * (b - a)
        f1, f2 = ll_of(c1)[0], ll_of(c2)[0]
        while b - a > tol:
            if f1 < f2:
                a, c1, f1 = c1, c2, f2
                c2 = a + inv * (b - a)
                f2 = ll_of(c2)[0]
            else:
                b, c2, f2 = c2, c1, f1
                c1 = b - inv * (b - a)
                f1 = ll_of(c1)[0]
        s_star = round((a + b) / 2, 4)
        ll_s, n = ll_of(s_star)
        ll_1, _ = ll_of(1.0)
    finally:
        pool.close()
        pool.join()
    return {"s": s_star, "gain_mnat_per_attempt": (ll_s - ll_1) / n * 1000,
            "n_heldout": n, "n_users": len(seqs), "mode": mode}


def apply_slope_scale(rating, slope, floor, s, mode, lapse):
    """Apply the calibrated multiplier to per-problem arrays; returns
    (rating, slope). mode 'slope' keeps ratings (they stay the 80%-clear
    anchors); mode 'rating' re-derives the 80%-point of the scaled curve:
    inner_B = s*(a*(th - rating) + kap) = kap  =>  th = rating + kap*(1-s)/(s*a)."""
    slope_new = s * np.asarray(slope, dtype=float)
    rating = np.asarray(rating, dtype=float)
    if mode == "rating":
        C = np.clip(np.asarray(floor, dtype=float), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        rating = rating + kap * (1.0 - s) / (s * np.asarray(slope, dtype=float))
    return rating, slope_new
