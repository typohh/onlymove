"""Shared survey-calibration helpers for the rank-curve / rating→rank fits.

This is a stateless helper library (no CLI / no `main`): it provides the pieces
the production rank-mapping (`pipeline.map_rating_to_rank`) and a couple of
auxiliary survey tools share.

Concepts:

  * Each user's rating = the problem RATING at which they clear CALIBRATION_P
    (80%), from a logistic over their (raw_rating, correct) attempts
    (`fit_user_rating`) — the SAME quantity the client computes, using only the
    shipped black-box ratings (never the derivation internals).
  * Per-server couplings are a cumulative-link (ordinal-threshold) fit
    (`ordinal_nll_vec`), with thresholds unpacked from a uniform-spacing
    parameterisation (`_unpack_thresholds`).

The production mapping (`map_rating_to_rank`) is a CONSENSUS model: rating maps
to a latent consensus rank and every server INCLUDING EGF is a free coupling off
it (EGF is not a privileged identity hub). The earlier EGF-hub server-curve knot
builder (`build_server_curves`) was removed as dead.
"""

import os
import sys

import numpy as np
from scipy.special import expit

# Slope of the success curve in RATING units (logit per rating unit). The
# survey user's rating is "the problem rating where they clear CALIBRATION_P",
# estimated by a logistic with this slope; the client (rank-decomposition.js)
# MUST use the same principle and a matching slope, or the loop reopens. Value
# is calibrated to the current rating scale (bots span ~1.5 rating over ~16
# ranks ⇒ ~0.1 rating/rank; with ~0.28 logit/rank that's ~2.8 logit/rating).
LOGIT_PER_RATING = 2.5
CALIBRATION_P = 0.8  # matches the client / momentum: rank = where you clear 80%
# Clamp per-problem floors below CALIBRATION_P so κ(c)=logit((P−c)/(1−c−λ)) stays
# finite (κ→∞ as c→P). Mirrors pwa/js/irt.js clampFloor / FLOOR_MAX exactly.
FLOOR_MAX = 0.78

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def fit_user_rating(attempts, raw_ratings, target=CALIBRATION_P, min_n=1,
                    slopes=None, floors=None, lapse=0.0):
    """The user's rating = the problem RATING at which they clear `target` (80%).

    Full 4PL MLE of R, IDENTICAL to the client's rank estimator
    (pwa/js/rank-decomposition.js via irt.js):

        P(correct) = c_p + (1 − c_p − λ)·σ(a_p·(R − raw_rating) + κ(c_p))
        κ(c) = logit((target − c)/(1 − c − λ))   (anchors P=target at R=raw_rating)

    `raw_rating` is the shipped 80%-clear rating r_p, so the κ offset keeps the
    "clear r_p at target" anchor exact for ANY slope/floor — and because the
    shipped ratings were DERIVED under this same 4PL (per-problem floor c_p +
    global ceiling λ), the ability estimate is consistent with them. `slopes`
    {tid: a_p} gives each attempt its own discrimination (else LOGIT_PER_RATING);
    `floors` {tid: c_p} its guessing floor (else 0); `lapse` is the global ceiling
    λ (population_constants.json fat_finger_rate). With floors=None & lapse=0 this
    reduces EXACTLY to the old 2PL. The floor is what stops a careless failure on
    a far-below-rank problem from yanking R, and matters most for strong users on
    easy problems. M is anchored on THIS quantity, so the client and the anchor
    MUST use the same per-problem a_p/c_p/λ or the loop reopens — keep them in
    lockstep with irt.js.

    Returns (R, n_joined) or (None, 0)."""
    rows = [(raw_ratings[t], s, (slopes.get(t) if slopes else None),
             (floors.get(t) if floors else None))
            for t, s in attempts if t in raw_ratings]
    if len(rows) < min_n:
        return None, 0
    rr = np.array([r for r, _, _, _ in rows], dtype=float)
    sc = np.array([s for _, s, _, _ in rows], dtype=float)
    sl = np.array([(a if (a is not None and np.isfinite(a) and a > 0)
                    else LOGIT_PER_RATING) for _, _, a, _ in rows], dtype=float)
    c = np.array([min(max(f if (f is not None and np.isfinite(f)) else 0.0, 0.0),
                      FLOOR_MAX) for _, _, _, f in rows], dtype=float)
    lam = max(0.0, float(lapse))
    span = np.clip(1.0 - c - lam, 1e-6, 1.0)              # (1 − floor − λ) asymptote span
    q = np.clip((target - c) / span, 1e-6, 1.0 - 1e-6)
    kappa = np.log(q / (1.0 - q))                          # κ(c): P=target at R=raw_rating
    R = float(rr.mean())
    for _ in range(80):
        praw = 1.0 / (1.0 + np.exp(-np.clip(sl * (R - rr) + kappa, -40.0, 40.0)))
        p = np.clip(c + span * praw, 1e-9, 1.0 - 1e-9)
        g = span * sl * praw * (1.0 - praw)               # dP/dR
        denom = p * (1.0 - p)
        score = float(np.sum((sc - p) * g / denom))
        info = float(np.sum(g * g / denom))               # 4PL Fisher info
        if info < 1e-9:
            break
        step = max(-1.0, min(1.0, score / info))
        R += step
        if abs(step) < 1e-6:
            break
    return float(R), len(rows)


def ordinal_nll_vec(tau, x, k):
    """Vectorized cumulative-link negative log-likelihood.

      P(declared = category k | x), thresholds τ (length K, ascending):
        k = 0      : expit(τ_0 − x)
        0 < k < K  : expit(τ_k − x) − expit(τ_{k−1} − x)
        k = K      : 1 − expit(τ_{K−1} − x)

    tau : (K,) thresholds in x units.  x : (n,) abilities.  k : (n,) ints in
    [0, K] (category = declared rank − range minimum)."""
    K = tau.shape[0]
    upper = np.where(k < K, expit(tau[np.minimum(k, K - 1)] - x), 1.0)
    lower = np.where(k > 0, expit(tau[np.maximum(k - 1, 0)] - x), 0.0)
    p = np.clip(upper - lower, 1e-12, 1.0)
    return -float(np.sum(np.log(p)))


def interp_extrap_vec(x, xs, ys):
    """np.interp with linear extrapolation past both ends (vectorized).
    xs strictly ascending, len ≥ 2; x an array."""
    y = np.interp(x, xs, ys)
    sl0 = (ys[1] - ys[0]) / (xs[1] - xs[0])
    sl1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + sl0 * (x - xs[0]), y)
    y = np.where(x > xs[-1], ys[-1] + sl1 * (x - xs[-1]), y)
    return y


def interp_extrap(x, xs, ys):
    """Scalar linear interp with end-slope extrapolation; mirrors the PWA's
    rank-decomposition.js interpExtrap so client and offline tools agree exactly.
    xs ascending, len ≥ 2; x a scalar; xs/ys may be plain lists or arrays."""
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


def _unpack_thresholds(block, baseline):
    """[α, log_e...] + baseline spacings → ascending thresholds (length
    len(baseline)+1).  τ_0 = α; τ_j = α + Σ_{≤j} baseline·exp(log_e)."""
    alpha = block[0]
    if len(baseline) == 0:
        return np.array([alpha])
    spacings = baseline * np.exp(block[1:])
    return np.concatenate([[alpha], alpha + np.cumsum(spacings)])


