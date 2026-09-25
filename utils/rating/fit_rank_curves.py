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

CALIBRATION_P = 0.8  # matches the client / momentum: rank = where you clear 80%
# Clamp per-problem floors below CALIBRATION_P so κ(c)=logit((P−c)/(1−c−λ)) stays
# finite (κ→∞ as c→P). Mirrors pwa/js/irt.js clampFloor / FLOOR_MAX exactly.
FLOOR_MAX = 0.78

from utils.paths import PROJECT_ROOT
# interp_extrap/_vec live in ratings_io, which is deliberately scipy-free so the
# light consumers (package_for_pwa, se_floor) can read a curve without this module.
# Re-exported here because they were defined here and callers import them from here.
from utils.rating.ratings_io import interp_extrap, interp_extrap_vec  # noqa: F401,E402
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def fit_user_rating(attempts, raw_ratings, *, slopes, floors, lapse,
                    target=CALIBRATION_P, min_n=1):
    """The user's rating = the problem RATING at which they clear `target` (80%).

    Full 4PL MLE of R, IDENTICAL to the client's rank estimator
    (pwa/js/rank-decomposition.js via irt.js):

        P(correct) = c_p + (1 − c_p − λ)·σ(a_p·(R − raw_rating) + κ(c_p))
        κ(c) = logit((target − c)/(1 − c − λ))   (anchors P=target at R=raw_rating)

    `raw_rating` is the shipped 80%-clear rating r_p, so the κ offset keeps the
    "clear r_p at target" anchor exact for ANY slope/floor — and because the
    shipped ratings were DERIVED under this same 4PL (per-problem floor c_p +
    global ceiling λ), the ability estimate is consistent with them. `slopes`
    {tid: a_p} gives each attempt its own discrimination, `floors` {tid: c_p} its
    guessing floor, and `lapse` is the global ceiling λ (load_lapse). An attempt
    on a problem absent from `raw_ratings` is dropped (a retired problem keeps its
    attempts); a problem that HAS a rating but no slope or floor, or a slope that
    is not positive, raises — the three come from one fit, and a default in its
    place is a model nobody fitted. The floor is what stops a careless failure on
    a far-below-rank problem from yanking R, and matters most for strong users on
    easy problems. M is anchored on THIS quantity, so the client and the anchor
    MUST use the same per-problem a_p/c_p/λ or the loop reopens — keep them in
    lockstep with irt.js.

    Returns (R, n_joined) or (None, 0)."""
    rows = [(raw_ratings[t], s, slopes[t], floors[t])
            for t, s in attempts if t in raw_ratings]
    if len(rows) < min_n:
        return None, 0
    rr = np.array([r for r, _, _, _ in rows], dtype=float)
    sc = np.array([s for _, s, _, _ in rows], dtype=float)
    sl = np.array([a for _, _, a, _ in rows], dtype=float)
    if not (np.all(np.isfinite(sl)) and np.all(sl > 0)):
        raise ValueError("fit_user_rating: a joined problem's slope is not a positive number")
    c = np.clip(np.array([f for _, _, _, f in rows], dtype=float), 0.0, FLOOR_MAX)
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


def ordinal_nll_vec(tau, x, k, w=None, eps=0.0):
    """Vectorized cumulative-link negative log-likelihood.

      P(declared = category k | x), thresholds τ (length K, ascending):
        k = 0      : expit(τ_0 − x)
        0 < k < K  : expit(τ_k − x) − expit(τ_{k−1} − x)
        k = K      : 1 − expit(τ_{K−1} − x)

    tau : (K,) thresholds in x units.  x : (n,) abilities.  k : (n,) ints in
    [0, K] (category = declared rank − range minimum).  w : optional (n,)
    per-row weights (the block-anchored production mapping normalizes each
    declaration's rows to total weight 1); None = unweighted.

    eps : contamination mixture (robust-IRT). With probability eps the
    declaration is junk — a lie/misclick, uniform over the K+1 categories:
    P_mix = (1−eps)·P_ordinal + eps/(K+1). Caps any single row's NLL at
    −log(eps/(K+1)) and sends its gradient smoothly to 0, so an impossible
    declaration stops steering the fit while merely improbable ones keep
    most of their weight. eps=0 = the plain ordinal likelihood."""
    K = tau.shape[0]
    upper = np.where(k < K, expit(tau[np.minimum(k, K - 1)] - x), 1.0)
    lower = np.where(k > 0, expit(tau[np.maximum(k - 1, 0)] - x), 0.0)
    p = np.clip(upper - lower, 1e-12, 1.0)
    if eps > 0.0:
        p = (1.0 - eps) * p + eps / (K + 1)
    logs = np.log(p)
    return -float(np.sum(logs if w is None else w * logs))


def junk_posterior_vec(tau, x, k, eps):
    """P(this declaration is junk | data) under ordinal_nll_vec's contamination
    mixture — the same P_ordinal, read as a posterior instead of a likelihood:

        junk = eps/(K+1),  P(junk | data) = junk / ((1−eps)·P_ordinal + junk)

    It is the fit's own opinion of a declaration rather than a residual, which is
    what makes it comparable across servers: a two-rank miss on a server whose
    categories are two ranks apart is ordinary, and the same miss where they are
    half a rank apart is not, and the thresholds already carry that. Arguments are
    ordinal_nll_vec's. eps=0 gives zero everywhere (no junk component to assign
    mass to)."""
    K = tau.shape[0]
    upper = np.where(k < K, expit(tau[np.minimum(k, K - 1)] - x), 1.0)
    lower = np.where(k > 0, expit(tau[np.maximum(k - 1, 0)] - x), 0.0)
    p = np.clip(upper - lower, 1e-12, 1.0)
    junk = eps / (K + 1)
    return junk / ((1.0 - eps) * p + junk)


def separated_head(scores, floor=0.5):
    """The head of a sorted-descending score that stands apart from the rest.

    Searched over the entries above `floor` plus the first one below them, so the
    whole candidate set is reachable when the drop lands there. The head ends at
    the LARGEST GAP between consecutive scores.

    This DIAGNOSES separation; it does not decide an exclusion. A gap is the
    largest one somewhere in any sample, including a sample drawn from one smooth
    population, so a caller reading it as a cut excludes down to wherever the
    decay happens to flatten and calls the result a discovered boundary. Read the
    gap SIZE to judge whether a head is real, and cut on a rule that means
    something on its own.

    Returns (n, score, gap): how many entries the head holds, the score of its
    last, and the gap below it. (0, nan, 0.0) when nothing clears the floor."""
    s = np.sort(np.asarray(scores, float))[::-1]
    n_cand = int((s > floor).sum())
    if n_cand == 0:
        return 0, float("nan"), 0.0
    # One past the candidates, so the whole candidate set is a possible cut.
    window = s[:min(n_cand + 1, len(s))]
    if len(window) == 1:
        return 1, float(window[0]), float("inf")
    gaps = window[:-1] - window[1:]
    i = int(np.argmax(gaps))
    return i + 1, float(window[i]), float(gaps[i])


def _unpack_thresholds(block, baseline):
    """[α, log_e...] + baseline spacings → ascending thresholds (length
    len(baseline)+1).  τ_0 = α; τ_j = α + Σ_{≤j} baseline·exp(log_e)."""
    alpha = block[0]
    if len(baseline) == 0:
        return np.array([alpha])
    spacings = baseline * np.exp(block[1:])
    return np.concatenate([[alpha], alpha + np.cumsum(spacings)])


