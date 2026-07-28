#!/usr/bin/env python3
"""Climbing-speed bake-off: history-window rank vs the live momentum loop.

Tests the proposed replacement for the front-page momentum controller
(`pwa/js/momentum-rank.js`): a *history-based* estimator that is

  (1) a CONSTANT-INFORMATION window MLE over recent results — walk back from the
      newest attempt accumulating per-attempt Fisher information about the user's
      rank until a target I is reached, then fit the rank that best explains that
      window (a port of InfoWindowEstimator in eval_rank_estimators.py), AND
  (2) a NORMAL PRIOR that "leans forward" — its mean sits OPTIMISM stones ABOVE
      the current estimate. The forward lean is what makes the system serve
      slightly-harder problems and lets an under-rated / improving user climb fast.

Why the lean gives the behaviour described in the proposal
----------------------------------------------------------
At the fixed point of the lean prior N(R+δ, 1/prior_info), the posterior mode R*
satisfies  prior_info·δ + S(R*) = 0,  where S is the window's log-likelihood
score. So the standing window score is pinned at −prior_info·δ: the user is held
at a level where they slightly UNDER-perform the pure MLE — i.e. permanently
pushed ~δ·(prior_info / window_info) of a stone above their demonstrated level.

  * Hot streak  → the win-only window has no upper pin (likelihood monotone in R),
    so nothing opposes the prior and R climbs rapidly. ("rapid climb on a streak")
  * Failures    → each failure near/above level is highly informative, the window
    score sharpens and overwhelms the fixed prior_info·δ push, pinning R.
    ("once there are multiple failures the prior matters less")

The cost is a STANDING positive bias in the estimate (the overshoot above true
ability). If the same number is the user's displayed rank that over-states them;
if it only drives SERVING it is just desirable difficulty. Both readings are
simulated here:

  optwin_lean      : estimate itself leans +δ (proposal as written; biased display)
  optwin_serve     : estimate is the UNBIASED window MLE, but problems are SERVED
                     at estimate+δ (decoupled — fast & unbiased display, same push)
  plainwin         : window MLE, no optimism (δ=0) — isolates what the lean buys
  momentum         : faithful port of the OBSOLETE momentum loop — replaced in
                     production (2026-06-26) by the linear-information window MLE;
                     kept only as a comparison point, it is NOT the baseline

Methodology — closed-loop forward simulation
--------------------------------------------
Replaying historically-served problems is invalid for comparing live estimators
(they were served under the live rank of the time; see the long note in
eval_rank_estimators.py). So we simulate users of KNOWN true ability whose
outcomes are drawn from the real per-problem 4PL, and let each estimator DRIVE
serving from the full ~123k-problem corpus via the production policy (50 nearest
the serve-rank, highest-slope pick). This induces the narrow served distribution
the estimator actually gets in production and lets momentum run its own loop.

Everything is in INTERNAL-RANK space (0 = the 1k/1d boundary; 1 unit = 1
stone), each problem mapped through its size's rating_to_rank curve exactly as
rank-decomposition.js does. Reuses the corpus loader, momentum port, problem pool
and 4PL math from auxiliary.eval_rank_estimators.

Usage:
  ./.venv/bin/python3 -m auxiliary.eval_history_rank                 # climb + grid
  ./.venv/bin/python3 -m auxiliary.eval_history_rank --optimism 2.0 --i-target 8
  ./.venv/bin/python3 -m auxiliary.eval_history_rank --no-plot       # text only
"""

import argparse
import math
import os
import random
import sys
from bisect import bisect_left

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Reuse the faithful corpus loader, momentum port, 4PL math and problem pool.
from auxiliary import eval_rank_estimators as H  # noqa: E402

HOUR_MS = H.HOUR_MS
CALIBRATION_P = H.CALIBRATION_P     # 0.8 — ratings are the 80%-clear ability, not 50%


def load_lapse():
    """The learned global 4PL upper-asymptote slip rate (fat-finger / ceiling),
    from ratings/rating_constants.json — the SAME constant the client reads
    (irt.js LAPSE). Hardcoding 0.01 was wrong; the fitted value is ~0.0347."""
    import json
    for path in ("rating_constants.json", os.path.join("ratings", "rating_constants.json")):
        p = os.path.join(PROJECT_ROOT, "ratings", "rating_constants.json")
        try:
            ff = json.load(open(p)).get("fat_finger_rate")
            if isinstance(ff, (int, float)) and 0 <= ff < 0.2:
                return float(ff)
        except (OSError, ValueError):
            pass
        break
    return 0.0347


def clear_kappa_lapse(c, lapse):
    """κ(c) anchored so P(clear)=CALIBRATION_P at a ZERO rank gap for the 4PL with
    upper asymptote (1−lapse): κ = logit((P−c)/(1−c−lapse)). This is the client's
    irt.js clearKappa — the lapse is IN the denominator. The reused harness
    clear_kappa omits it (the old 3PL anchor), which with a real lapse leaves the
    at-level clear rate a touch below 80%. With this κ, c + (1−c−lapse)·σ(κ) = P
    EXACTLY, so a problem served at the user's rank is cleared 80% of the time."""
    cc = max(c, 0.0)
    q = min(max((CALIBRATION_P - cc) / (1 - cc - lapse), 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def relapse_pool(pool, lapse):
    """Rebuild each problem's κ with the lapse-corrected anchor (load_problem_pool
    builds it lapse-free via the harness clear_kappa). Pool entries are
    (x, slope, floor, kappa, a_p)."""
    return [(x, slope, floor, clear_kappa_lapse(floor, lapse), a_p)
            for (x, slope, floor, _k, a_p) in pool]


def mci(vals):
    """(mean, half-width of the 95% CI of the mean, sample sd) over a metric's
    per-run values. Half-width = 1.96·sd/√n (normal approx, fine for n≳30). The sd
    is the run-to-run DISTRIBUTION width; the half-CI is our confidence in the mean."""
    v = [x for x in vals if isinstance(x, (int, float)) and math.isfinite(x)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(v) / n
    if n == 1:
        return mean, float("nan"), 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    return mean, 1.96 * sd / math.sqrt(n), sd


def median(vals):
    v = sorted(x for x in vals if isinstance(x, (int, float)) and math.isfinite(x))
    return v[len(v) // 2] if v else float("nan")


def idx_to_label(idx):
    """Internal rank -> kyu/dan label (-0.5 == 1k, 0.5 == 1d)."""
    scalar = idx + 0.5            # band-centre convention of the old printout
    if scalar >= 1:
        return f"{scalar:.0f}d" if abs(scalar - round(scalar)) < 0.05 else f"{scalar:.1f}d"
    n = 1 - scalar                # kyu number
    return f"{n:.0f}k" if abs(n - round(n)) < 0.05 else f"{n:.1f}k"


# ---------------------------------------------------------------------------
# Estimators — uniform (estimate, serve_rank, update) interface so the sim loop
# can treat momentum and the window estimators identically. `estimate` is what
# would be DISPLAYED / scored for accuracy; `serve_rank` is what picks the next
# problem (they differ only for the decoupled serve-offset variant).
# ---------------------------------------------------------------------------
class MomentumDriver:
    """The OBSOLETE momentum loop (port in eval_rank_estimators.MomentumEstimator),
    cold-starting at 20k (index 0). Replaced in production (2026-06-26) by the
    linear-information window MLE (LinWindowDriver); retained only as a comparison
    point — the PRODUCTION/BASELINE estimator is the window MLE, not this."""
    name = "momentum"

    def __init__(self, **_):
        self.m = H.MomentumEstimator()

    @property
    def estimate(self):
        return self.m.rank

    @property
    def serve_rank(self):
        return self.m.rank

    def update(self, att):
        self.m.update(att)


class OptWindowDriver:
    """Constant-information-window 4PL MLE with an optional forward-leaning prior.

    I_target     accumulated Fisher information that sets the window length.
    prior_info   precision of the lean prior (pseudo-Fisher units; ~slope^2 per
                 pseudo-attempt). Larger = the +δ lean pulls harder.
    optimism     δ, stones the prior mean sits ABOVE the current estimate (lean
                 mode) — set 0 for a plain window MLE.
    serve_offset stones the SERVED problem sits above the estimate (decoupled
                 push; keeps the estimate itself unbiased).
    start        cold-start estimate (index 0 == 20k, matching momentum).
    tau_max_days hard recency cap on the window (a pure info window has no time
                 localisation, so very stale wins can't pad it forever).
    """

    def __init__(self, I_target=8.0, prior_info=3.0, optimism=2.0, serve_offset=0.0,
                 start=0.0, lapse=0.0, tau_max_days=120, step_cap=None,
                 fixed_prior_mean=None, ucb_k=0.0, name="optwin"):
        self.I_target = I_target
        self.prior_info = prior_info
        self.optimism = optimism
        self.serve_offset = serve_offset
        # UCB serving: serve at estimate + ucb_k · SE(estimate). The optimism then
        # SCALES WITH UNCERTAINTY — large at cold start / on a streak (push up,
        # explore), vanishing once failures sharpen the posterior. This is the
        # self-limiting form of the proposal's lean: "the prior matters less once
        # failures arrive" falls out of the shrinking SE, with NO standing display
        # bias (the estimate is the regularised MLE). fixed_prior_mean regularises
        # the all-win MLE to a finite value (a weak population anchor, not a lean).
        self.fixed_prior_mean = fixed_prior_mean
        self.ucb_k = ucb_k
        self._last_se = 0.0
        # Steady-state standard-error FLOOR. The info window always converges to
        # holding ~I_target worth of Fisher information plus the prior, so the
        # posterior SE bottoms out at 1/√(prior_info + I_target) — it does NOT go to
        # zero. The UCB bonus below is measured relative to this floor so it vanishes
        # at convergence (see serve_rank).
        denom = self.prior_info + self.I_target
        self.se_floor = (1.0 / math.sqrt(denom)) if denom > 0 else 0.0
        self.lapse = lapse
        self.tau_max_ms = tau_max_days * 24 * HOUR_MS if tau_max_days else None
        # Trust region: max stones the estimate may move in ONE attempt. A window
        # MLE has no natural step bound — on a one-sided (all-win) window it jumps
        # to the clamp, detaching the serve-rank from where informative failures
        # can be generated (the runaway the uncapped lean shows). Capping the step
        # keeps the serving loop coupled, so the clear-rate feedback still works —
        # i.e. it re-introduces the one property that makes momentum stable.
        self.step_cap = step_cap
        self._R = start
        self.hist = []
        self.name = name

    @property
    def estimate(self):
        return self._R

    @property
    def serve_rank(self):
        # UCB exploration bonus measured RELATIVE to the steady-state SE floor, so it
        # is a pure ONBOARDING accelerator: large while the posterior is still loose
        # (cold start / streak), and → 0 once the window fills (SE → se_floor). Without
        # the subtraction, ucb_k·SE leaves a PERMANENT +ucb_k·se_floor over-serve, which
        # pins the steady-state clear rate below the 80% target. Clamped ≥0 so it never
        # serves EASIER than the estimate.
        bonus = self.ucb_k * max(0.0, self._last_se - self.se_floor)
        return self._R + self.serve_offset + bonus

    def update(self, att):
        self.hist.append(att)
        t_now = att[0]
        upper = 1.0 - self.lapse
        window = []
        acc_info = 0.0
        R_ref = self._R                      # size the window by Fisher info at R
        for a in reversed(self.hist):
            if self.tau_max_ms is not None and (t_now - a[0]) > self.tau_max_ms:
                break
            _, x, slope, floor, kappa, y = a[:6]
            window.append((x, slope, floor, kappa, y, 1.0))
            praw = H.sigmoid(slope * (R_ref - x) + kappa)
            p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
            g = (upper - floor) * slope * praw * (1 - praw)
            acc_info += g * g / (p * (1 - p))      # (dP/dR)^2 / P(1-P)
            if acc_info >= self.I_target:
                break
        # Prior mean: a fixed population anchor (UCB mode — no lean, just regularises
        # the all-win MLE to finite) or the forward lean estimate+δ.
        prior_mean = (self.fixed_prior_mean if self.fixed_prior_mean is not None
                      else self._R + self.optimism)
        R_new, info = H._newton_3pl(window, prior_mean, self.prior_info, self._R, self.lapse)
        if self.step_cap is not None:
            R_new = min(self._R + self.step_cap, max(self._R - self.step_cap, R_new))
        self._R = R_new
        self._last_se = 1.0 / math.sqrt(info) if info > 0 else 0.0


class LinWindowDriver:
    """Production-faithful LINEAR-information-ramp window MLE — exactly the shipped
    estimateWindowRank (rank-decomposition.js): walking back from the newest attempt
    the weight ramps 1 → 0 as the accumulated Fisher information (at the current
    estimate) goes 0 → i_target, with a HARD sample cap `win_cap`. Weak prior
    N(R_prev, 1/prior_info); UCB serving at estimate + ucb_k·(se − se_floor).

    Unlike OptWindowDriver (a HARD info window: uniform weight then a sharp cut, no
    sample cap), this reproduces the two knobs the production estimator actually has
    — `i_target` (WIN_ITARGET=16) and `win_cap` (WIN_CAP=500) — so the boundary the
    cap creates (a sample aging out at non-zero weight when the ramp can't reach 0
    within the cap, in the served-at-rank low-information regime) is reproduced."""

    def __init__(self, i_target=16.0, win_cap=500, prior_info=1.0 / 16, ucb_k=1.0,
                 start=0.0, lapse=0.0, name="lin"):
        self.i_target = i_target
        self.win_cap = win_cap
        self.prior_info = prior_info
        self.ucb_k = ucb_k
        denom = prior_info + i_target / 2.0      # steady-state info = prior + ramp area
        self.se_floor = (1.0 / math.sqrt(denom)) if denom > 0 else 0.0
        self.lapse = lapse
        self._R = start
        self._last_se = 0.0
        self.hist = []
        self.name = name

    @property
    def estimate(self):
        return self._R

    @property
    def serve_rank(self):
        return self._R + self.ucb_k * max(0.0, self._last_se - self.se_floor)

    def update(self, att):
        self.hist.append(att)
        upper = 1.0 - self.lapse
        window = []
        acc = 0.0
        R_ref = self._R
        for a in reversed(self.hist):
            if len(window) >= self.win_cap:
                break
            w = 1.0 - acc / self.i_target
            if w <= 0.0:
                break
            _, x, slope, floor, kappa, y = a[:6]
            window.append((x, slope, floor, kappa, y, w))
            praw = H.sigmoid(slope * (R_ref - x) + kappa)
            p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
            g = (upper - floor) * slope * praw * (1 - praw)
            acc += g * g / (p * (1 - p))
        R_new, info = H._newton_3pl(window, self._R, self.prior_info, self._R, self.lapse)
        self._R = R_new
        self._last_se = 1.0 / math.sqrt(info) if info > 0 else 0.0


class LinPosWindowDriver:
    """Window MLE with a PURELY POSITIONAL linear weight — `w = (M − j)/M` for the j-th
    attempt back from the newest (j=0 → w=1, the oldest in-window → w=1/M ≈ 0), where
    M = min(#attempts, cap) is the AVAILABLE window. This is the user's "n/min(samples,500)
    linear weight": the ramp always spans whatever recent history exists, so the oldest
    sample leaves at ~0 weight even cold-start with few attempts (not near-uniform until
    the history fills past the cap).

    Contrast with LinWindowDriver (the shipped info-ramp): there the edge weight only
    reaches 0 once the accumulated Fisher information hits i_target, which in the
    served-at-rank LOW-information regime it never does — so the departing sample leaves
    at ~0.93 and yanks the rank opposite the new answer (the correct→down boundary
    artifact). The positional ramp forces the departing sample to ~0 BY CONSTRUCTION,
    independent of information content, so that artifact cannot occur. Cost: it ignores
    how informative each attempt is (weights by position alone) — fine served at rank
    where per-attempt info is ~constant, but it does NOT auto-widen on genuinely flat
    problems the way a constant-information window would. Effective sample size ~3·M/4
    (Σw=(M+1)/2, Σw²≈M/3).

    Same weak prior N(R_prev, 1/prior_info) and UCB serving as LinWindowDriver; the SE
    floor is the steady-state full-window SE 1/√(prior_info + Σw·i_per) with Σw=(cap+1)/2
    and the per-attempt info i_per tracked online (≈constant served at rank)."""

    def __init__(self, win_cap=500, prior_info=1.0 / 16, ucb_k=1.0,
                 start=0.0, lapse=0.0, i_per_init=5e-3, name="linpos"):
        self.win_cap = win_cap
        self.prior_info = prior_info
        self.ucb_k = ucb_k
        self.sum_w = (win_cap + 1) / 2.0     # Σ_{j=0}^{cap-1}(cap−j)/cap = (cap+1)/2
        self.i_per = i_per_init
        self.lapse = lapse
        self._R = start
        self._last_se = 0.0
        self.hist = []
        self.name = name

    @property
    def estimate(self):
        return self._R

    @property
    def se_floor(self):
        denom = self.prior_info + self.sum_w * self.i_per
        return (1.0 / math.sqrt(denom)) if denom > 0 else 0.0

    @property
    def serve_rank(self):
        return self._R + self.ucb_k * max(0.0, self._last_se - self.se_floor)

    def update(self, att):
        self.hist.append(att)
        upper = 1.0 - self.lapse
        R_ref = self._R
        window = []
        m = min(len(self.hist), self.win_cap)   # ramp spans the AVAILABLE window
        for j, a in enumerate(reversed(self.hist)):
            if j >= self.win_cap:
                break
            w = (m - j) / m
            _, x, slope, floor, kappa, y = a[:6]
            window.append((x, slope, floor, kappa, y, w))
        # online per-attempt info at the newest attempt (≈ served-at-rank constant)
        _, x0, s0, f0, k0, _ = self.hist[-1][:6]
        praw0 = H.sigmoid(s0 * (R_ref - x0) + k0)
        p0 = min(max(f0 + (upper - f0) * praw0, 1e-9), 1 - 1e-9)
        g0 = (upper - f0) * s0 * praw0 * (1 - praw0)
        self.i_per += 0.1 * (g0 * g0 / (p0 * (1 - p0)) - self.i_per)
        R_new, info = H._newton_3pl(window, self._R, self.prior_info, self._R, self.lapse)
        self._R = R_new
        self._last_se = 1.0 / math.sqrt(info) if info > 0 else 0.0


class ExpDecayMLEDriver:
    """Exponentially-weighted (FADING-MEMORY) 4PL MLE — the soft-window cousin of
    OptWindowDriver. Instead of a HARD constant-information window (include recent
    attempts until a fixed Fisher-information target, then cut sharply), EVERY past
    attempt enters the weighted likelihood with weight decay**age (age 0 = newest) —
    a smoothly fading memory. Effective sample size 1/(1-decay); half-life
    ln2 / −ln(decay) attempts. Smaller decay = shorter memory (reacts fast, noisier);
    larger decay = longer memory (steadier, slower to react). The hard window is the
    limit where all in-window weights are 1 and everything older is 0; the decay is
    the soft version of that boundary.

    Serving is the SAME UCB policy as the tuned `ucb` driver: serve at
    estimate + ucb_k·(SE − SE_floor) — an uncertainty-scaled push that is large while
    the posterior is loose (cold start / streak) and vanishes at convergence, so the
    DISPLAYED estimate stays unbiased. The fading-memory steady-state SE floor is
    1/√(prior_info + i_per/(1−decay)), where i_per is the typical per-attempt Fisher
    information at the user's level — tracked online (it is ~constant, since problems
    are always served near the estimate).

    Two decay modes:
      - COUNT (info_scale=0): weight = decay**age. Standard exponential forgetting,
        a fixed effective NUMBER of attempts (1/(1-decay)) regardless of how
        informative they are -- so on FLAT high-rank problems (low Fisher info per
        attempt) it is starved and jitters, where a constant-information window would
        auto-widen.
      - INFO (info_scale>0): weight = exp(-acc_info/info_scale), where acc_info is the
        Fisher information accumulated from the NEWER attempts. The soft-edged
        constant-INFORMATION window -- auto-widens on flat problems exactly like the
        hard window, but with a smooth decay instead of a hard cut. Steady-state info
        ~ prior_info + info_scale, so the SE floor is a clean 1/sqrt(prior_info+info_scale)."""

    def __init__(self, decay=0.9, prior_info=1.0 / 4 ** 2, ucb_k=1.0, serve_offset=0.0,
                 start=0.0, lapse=0.0, weight_eps=1e-2, info_scale=0.0, shape="exp",
                 cap=200, pop_mean=None, pop_sd=6.4, name="exp"):
        self.decay = float(decay)
        # prior_info = the PER-UPDATE REGULARISER precision (1/initial_SD²), centred
        # at the live estimate self._R — it tames the all-win window and bounds the
        # per-step move; initial_SD (was 4) is the knob to sweep.
        self.prior_info = prior_info
        self.ucb_k = ucb_k
        self.serve_offset = serve_offset
        # pop_mean/pop_sd = a SEPARATE, FIXED population anchor N(pop_mean, pop_sd²)
        # ADDED to the per-update prior. It is NOT the cold start: a no-history user
        # still cold-starts at `start` (index 0 == 20k, the shipped `lin` baseline),
        # exactly as production does. The anchor is a constant pull toward pop_mean
        # whose sign flips at the mean — pop_info·(pop_mean − R): while the estimate is
        # BELOW the mean the pull is upward (it SPEEDS the climb toward the mean) and
        # once it passes the mean the pull is downward (it SLOWS progress past it). It
        # is deliberately WEAK (pop_sd = the true population SD ≈ 6.4), so it only bites
        # at cold start when the window is near-empty; once the window fills its
        # ~pop_info/window_info bias is sub-0.05 stone, so it never traps a genuinely
        # weak/strong user at the mean. None = no anchor (plain self._R regulariser).
        #
        # NB the earlier draft SEEDED self._R = pop_mean, which OVERRODE the 20k/SD-4
        # cold start: a true 15k was then served 6k problems (9 stones too hard) and had
        # to climb DOWN — the opposite of "speed toward the mean". The cold start must
        # stay at `start`; the anchor only leans the climb.
        self.pop_mean = pop_mean
        self.pop_info = (1.0 / float(pop_sd) ** 2) if pop_mean is not None else 0.0
        self._R = start
        self.lapse = lapse
        self.weight_eps = weight_eps          # truncate the exp tail where weight < eps
        self.info_scale = float(info_scale)   # >0 ⇒ information-decay mode
        self.shape = shape                    # 'exp' or 'linear' ramp (info mode)
        self.cap = cap                        # hard backstop on #samples (no time cap)
        self.hist = []
        self._last_se = 0.0
        self._i_per = None                    # online EMA of per-attempt info (count mode)
        self.name = name

    @property
    def estimate(self):
        return self._R

    def _se_floor(self):
        if self.info_scale > 0:               # info mode: closed-form steady-state info
            # steady info = prior + Σ weight·i = prior + ∫ weight d(acc_info).
            # exp:    ∫₀^∞ exp(-I/S) dI = S.   linear: ∫₀^S (1-I/S) dI = S/2.
            eff = self.info_scale if self.shape != "linear" else self.info_scale / 2.0
            denom = self.prior_info + eff
            return 1.0 / math.sqrt(denom) if denom > 0 else 0.0
        if self._i_per is None:
            return 0.0
        denom = self.prior_info + self._i_per / max(1e-9, 1.0 - self.decay)
        return 1.0 / math.sqrt(denom) if denom > 0 else 0.0

    @property
    def serve_rank(self):
        bonus = self.ucb_k * max(0.0, self._last_se - self._se_floor())
        return self._R + self.serve_offset + bonus

    def update(self, att):
        self.hist.append(att)
        upper = 1.0 - self.lapse
        window = []
        if self.info_scale > 0:
            # INFO-decay: weight ramps down over ACCUMULATED Fisher info (not sample
            # count), so the window auto-widens on flat problems (info accrues slowly
            # ⇒ weights fade slowly) while the departing attempt sits at ~0 weight.
            #   exp:    weight = exp(-acc/info_scale)        (smooth, infinite tail)
            #   linear: weight = max(0, 1 - acc/info_scale)  (smooth, finite support)
            acc = 0.0
            R_ref = self._R
            for a in reversed(self.hist):
                if self.shape == "linear":
                    w = 1.0 - acc / self.info_scale
                    if w <= 0.0:
                        break
                else:
                    w = math.exp(-acc / self.info_scale)
                    if w < self.weight_eps:
                        break
                if self.cap is not None and len(window) >= self.cap:
                    break
                _, x, slope, floor, kappa, y = a[:6]
                window.append((x, slope, floor, kappa, y, w))
                praw = H.sigmoid(slope * (R_ref - x) + kappa)
                p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
                g = (upper - floor) * slope * praw * (1 - praw)
                acc += g * g / (p * (1 - p))           # accumulate info as we walk back
        else:
            # COUNT-decay: weight = decay**age
            max_age = (int(math.log(self.weight_eps) / math.log(self.decay))
                       if 0.0 < self.decay < 1.0 else len(self.hist))
            for age, a in enumerate(reversed(self.hist)):
                if age > max_age:
                    break
                _, x, slope, floor, kappa, y = a[:6]
                window.append((x, slope, floor, kappa, y, self.decay ** age))
        # prior centred at the current estimate (no lean; optimism lives in serving)
        # Prior = product of two Gaussians: the per-update regulariser N(self._R,
        # 1/prior_info) AND the fixed population anchor N(pop_mean, 1/pop_info).
        # Precisions add; the mean is the precision-weighted average.
        info0 = self.prior_info + self.pop_info
        mean0 = ((self.prior_info * self._R + self.pop_info * (self.pop_mean or 0.0))
                 / info0) if info0 > 0 else self._R
        R_new, info = H._newton_3pl(window, mean0, info0, self._R, self.lapse)
        self._R = R_new
        self._last_se = 1.0 / math.sqrt(info) if info > 0 else 0.0
        # online estimate of the per-attempt info at level, from the newest attempt
        # (served near the estimate ⇒ ~at-level Fisher info ⇒ ~constant).
        _, x, slope, floor, kappa, y = att[:6]
        praw = H.sigmoid(slope * (self._R - x) + kappa)
        p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
        g = (upper - floor) * slope * praw * (1 - praw)
        i_new = g * g / (p * (1 - p))
        self._i_per = i_new if self._i_per is None else 0.85 * self._i_per + 0.15 * i_new


# ---------------------------------------------------------------------------
# Closed-loop forward simulation (estimator drives serving; true ability known).
# ---------------------------------------------------------------------------
def simulate(driver, theta, pool, xs, n_attempts, lapse, serve_k, rng, dt_h=4,
             early_fail_rate=0.0, early_fail_n=0):
    """One simulated user of true ability `theta`. All probabilities are in the
    80%-calibrated logit (each problem's κ = clear_kappa(floor) makes p_clear=0.8
    at x==θ), so an at-level problem is cleared 80% of the time — NOT 50%.

    early_fail_rate / early_fail_n model a NOISY-ONBOARDING user: for the first
    `early_fail_n` attempts an extra `early_fail_rate` fraction of answers are
    forced to FAIL regardless of difficulty (a distracted/misclicking warmup burst,
    decided BEFORE the attempt). This stress-tests whether an estimator mistakes a
    transient early-failure streak for genuine low ability and stalls/under-rates.

    Returns the per-attempt SIGNED estimate trace (the live rank that served each
    problem, recorded BEFORE the outcome is folded in), served difficulties, the
    win/loss stream, and p_true (the TRUE clear probability of each served problem
    — its tail near the guessing floor is the 'no hope' serving-harm signal)."""
    t = 0
    served = set()
    trace, served_x, correct, ptrue, praw = [], [], [], [], []
    W = max(serve_k * 3, 150)
    n = len(pool)
    for ai in range(n_attempts):
        Rs = driver.serve_rank
        j = bisect_left(xs, Rs)
        lo = max(0, j - W); hi = min(n, j + W)
        cand = [k for k in range(lo, hi) if k not in served]
        if not cand:
            break
        cand.sort(key=lambda k: abs(pool[k][0] - Rs))
        cand = cand[:serve_k]
        k = max(cand, key=lambda k: pool[k][1])      # highest slope among nearest
        served.add(k)
        x, slope, floor, kappa, a_p = pool[k]
        pr = H.sigmoid(slope * (theta - x) + kappa)  # logistic part of p_clear
        p_true = H.p_clear(theta, x, slope, floor, kappa, lapse)
        y = 1 if rng.random() < p_true else 0
        if ai < early_fail_n and rng.random() < early_fail_rate:
            y = 0                                    # forced early-onboarding failure
        trace.append(driver.estimate)
        served_x.append(x); correct.append(y); ptrue.append(p_true); praw.append(pr)
        driver.update((t, x, slope, floor, kappa, y, a_p))
        t += dt_h * HOUR_MS
    return trace, served_x, correct, ptrue, praw


# Serving-harm thresholds, reported as COUNTS (the NUMBER of such problems the
# user is actually shown across the run) — the overshoot inflicts a fixed *number*
# of bad problems, which a whole-run % hides.
#   HARD_P    — likelier to FAIL than clear (true clear < 0.5): a demoralising problem.
#   FLOOR_PRAW— "hits the problem's FLOOR": the logistic part praw=σ(slope·(θ−x)+κ)
#               of the served problem is ~0, so the user's true clear prob has
#               collapsed to that problem's OWN guessing floor c_p — skill adds
#               nothing beyond guessing. This is the principled 'no hope', per the
#               problem's own floor, not a fixed probability cut.
HARD_P = 0.50
FLOOR_PRAW = 0.05


def user_metrics(trace, correct, ptrue, praw, theta, settle_tol=1.0):
    """Per-run outcomes. RISE and SETTLE are split on purpose: an estimator that
    overshoots (the lean) RISES to the user's level fast but SETTLES late because
    it has to correct the overshoot — collapsing both into one number hid both the
    lean's real speed AND its real cost.

      rise_att     SPEED to the user's level: first attempt the estimate lands
                   within settle_tol of TRUTH (θ) — what 'converge from 20k' means.
      overshoot    transient cost: peak estimate − θ (stones it sailed ABOVE you).
      n_hard       COUNT of served problems the user is likelier to fail than clear
                   (true clear < HARD_P). Concentrated in the overshoot, so it IS
                   the pre-convergence serving harm — an absolute number of bad
                   problems shown, not a diluted whole-run %.
      n_floor      COUNT of served problems that HIT THE PROBLEM'S FLOOR (praw <
                   FLOOR_PRAW → true clear ≈ the problem's own guessing floor; the
                   user has no real chance). The headline 'hopeless problems shown'.
      settle_att   first attempt after which the estimate STAYS within settle_tol
                   of its settled value (last 25% mean). Late iff it overshoots.
      steady_std   stability once converged: SD of the estimate over the 2nd half.
      steady_bias  settled estimate − θ (the lean's intended standing offset).
      clear        realized clear rate over the 2nd half (on target ≈ 80%).
    """
    n = len(trace)
    tail0 = (3 * n) // 4
    settled = sum(trace[tail0:]) / max(1, n - tail0)
    rise = next((i for i in range(n) if abs(trace[i] - theta) <= settle_tol), n)

    def settle_within(tol):
        # first attempt after which the estimate STAYS within `tol` of its settled
        # value for the entire rest of the run
        for i in range(n):
            if all(abs(trace[j] - settled) <= tol for j in range(i, n)):
                return i
        return n
    settle1 = settle_within(1.0)         # within 1 stone of settled
    settle2 = settle_within(2.0)         # within 2 stones of settled (looser → earlier)
    half = n // 2
    th = trace[half:] or trace
    mu = sum(th) / len(th)
    std = math.sqrt(sum((v - mu) ** 2 for v in th) / len(th))
    ch = correct[half:] or correct
    # Split the steady region (back half) into THIRDS and take each third's mean.
    # If t1 < t2 < t3 (toward truth) the estimate is STILL crawling upward after it
    # nominally settled — slow residual convergence the single steady-state mean hides.
    L = len(th)
    mean3 = lambda s: (sum(s) / len(s)) if s else float("nan")
    t1 = mean3(th[:L // 3])
    t2 = mean3(th[L // 3:2 * L // 3])
    t3 = mean3(th[2 * L // 3:])
    return {
        "rise_att": rise,
        "overshoot": max(trace) - theta,
        "n_hard": sum(1 for p in ptrue if p < HARD_P),
        "n_floor": sum(1 for pr in praw if pr < FLOOR_PRAW),
        "settle_att": settle1,           # back-compat (== settle within 1 stone)
        "settle1": settle1,
        "settle2": settle2,
        "settled_val": settled,          # absolute settled rank index (last-25% mean)
        "steady_t1": t1,                 # back-half thirds: drift detector (t1<t2<t3 = still climbing)
        "steady_t2": t2,
        "steady_t3": t3,
        "steady_std": std,
        "steady_bias": mu - theta,
        "clear": sum(ch) / len(ch),
    }


class KalmanDriver:
    """Local-LINEAR-TREND Kalman filter (EKF / Laplace) — tracks skill LEVEL *and*
    VELOCITY. This is the fix for the old level-only random-walk filter, which had
    NO trajectory component: its only state was the level, so each attempt was a
    lone Bayesian nudge toward the latest result and it crawled ~0.1 stone/attempt,
    never crossing the 20-stone cold-start gap. A trend state carries momentum.

    State s = [level, velocity], per-ATTEMPT transition (Δ = 1 'sample'):
        level'    = level + velocity            (velocity is ranks PER SAMPLE)
        velocity' = vel_decay · velocity        (DAMPED random walk, drift var q_vel)
    F = [[1,1],[0,vel_decay]]; process noise Q = diag(q_level, q_vel). The 4PL obs
    depends on the LEVEL only, so its Jacobian row is [g, 0]; velocity moves only
    through the predict-step covariance coupling (the off-diagonal of P).

    The velocity is SEED-BIASED to fast initial progress: v0 ≈ +2 ranks/sample, so
    a cold-start user climbs ~2 stones/attempt out of 20k (vs the old level-only
    filter's crawl).

    MEAN-REVERTING velocity (OU), the principled self-limit. Each update adds a weak
    prior N(0, 1/vel_revert) on the velocity, so velocity is pulled toward 0 with
    strength vel_revert UNLESS the observations actively evidence a climb. While the
    level sits below ability the served (≈at-level) problems are cleared ABOVE the
    predicted ~80% (the user is stronger than where they're served); that positive
    level-surprise is attributed partly to velocity and holds it up against the
    revert prior, so the climb continues over a path of ANY length. Once the level
    reaches ability the surprise vanishes, nothing opposes the prior, and velocity
    decays to 0 → the level stops and SETTLES. No brake, no arrival latch, no
    failure trigger — so no single-fluke fragility and no win-streak velocity
    rebuild (the two failure modes of the reach-then-brake design). vel_decay (≤1)
    is an optional extra geometric damping in the transition; vel_revert does the
    real work. Variance floors keep the level able to re-accelerate an improving user."""
    name = "kalman"

    def __init__(self, start=0.0, v0=2.0, vel_revert=0.5, vel_decay=1.0,
                 level_var=9.0, vel_var=2.0, q_level=0.02, q_vel=0.03, lapse=0.0,
                 level_var_floor=1.0, vel_var_floor=0.05, level_corr_cap=1.5):
        self.s = np.array([float(start), float(v0)])
        self.P = np.diag([float(level_var), float(vel_var)])
        self.Q = np.diag([float(q_level), float(q_vel)])
        self.vel_revert = float(vel_revert)
        self.vel_decay = float(vel_decay)
        self.lapse = lapse
        self.level_var_floor = level_var_floor
        self.vel_var_floor = vel_var_floor
        # Trust region on the OBSERVATION correction to the level: the local-linear-
        # trend coupling pumps the velocity's uncertainty into the level variance each
        # predict, so the level posterior stays loose and a single surprising outcome
        # yanks the level several stones (the steady-state jitter that blocks settling).
        # Bounding |level − predicted level| per attempt is momentum's bounded-step
        # stability — the velocity-driven PREDICTION still climbs uncapped, only the
        # noise-driven correction is capped.
        self.level_corr_cap = float(level_corr_cap)

    @property
    def estimate(self):
        return float(self.s[0])

    @property
    def serve_rank(self):
        return float(self.s[0])

    def update(self, att):
        _, x, slope, floor, kappa, y = att[:6]
        upper = 1.0 - self.lapse
        F = np.array([[1.0, 1.0], [0.0, self.vel_decay]])
        # --- predict: carry the level forward by the velocity, inflate covariance ---
        s_pred = F @ self.s
        P_pred = F @ self.P @ F.T + self.Q
        Lam = np.linalg.inv(P_pred)                  # prior precision (2x2)
        # --- Laplace/EKF update: Newton on the joint MAP of N(s_pred,P_pred) × 4PL,
        #     plus a velocity→0 mean-reversion prior (precision vel_revert) ---
        s = s_pred.copy()
        Hess = Lam
        for _ in range(25):
            L = s[0]
            praw = H.sigmoid(slope * (L - x) + kappa)
            p = min(max(floor + (upper - floor) * praw, 1e-9), 1 - 1e-9)
            g = (upper - floor) * slope * praw * (1 - praw)   # dP/dlevel
            gl = (y - p) * g / (p * (1 - p))                  # loglik score wrt level
            il = g * g / (p * (1 - p))                        # obs Fisher info (level)
            grad = -Lam @ (s - s_pred)
            grad[0] += gl
            grad[1] += -self.vel_revert * s[1]                # pull velocity toward 0
            Hess = Lam.copy()
            Hess[0, 0] += il
            Hess[1, 1] += self.vel_revert
            step = np.clip(np.linalg.solve(Hess, grad), -2.0, 2.0)
            s = s + step
            s[0] = max(H.X_CLAMP_LO, min(H.X_CLAMP_HI, s[0]))
            if float(np.max(np.abs(step))) < 1e-6:
                break
        # Trust region: cap the level's deviation from its (velocity-driven) prediction.
        s[0] = min(s_pred[0] + self.level_corr_cap,
                   max(s_pred[0] - self.level_corr_cap, s[0]))
        self.s = s
        self.P = np.linalg.inv(Hess)
        self.P[0, 0] = max(self.P[0, 0], self.level_var_floor)
        self.P[1, 1] = max(self.P[1, 1], self.vel_var_floor)


# ---------------------------------------------------------------------------
# Driver factory + parameterised sweep ("evolve") harness.
# A single build_driver(model, params, ...) lets the evolve loop override any one
# parameter while holding the rest at the model's tuned defaults, so a sweep is
# always one-variable-at-a-time (or a 2-D grid) over real knobs — no handpicking.
# ---------------------------------------------------------------------------
DRIVER_DEFAULTS = {
    # window-MLE family (OptWindowDriver): SD = prior stddev (stones), I = info window.
    # Tuned by the evolve sweeps at the 15k & 5d extremes (2026-06-26): the +δ lean
    # at δ=1 (not 2) and a loose SD=3 onboards fast without the +2 overshoot; UCB
    # serves at estimate+k·SE with a gentle k=1 (unbiased estimate, ~momentum hard).
    "lean":   dict(i_target=8.0, sd=3.0, optimism=1.0),
    "serve":  dict(i_target=8.0, sd=3.0, offset=1.0),
    "ucb":    dict(i_target=8.0, sd=4.0, ucb_k=1.0),
    # fading-memory MLE with the same UCB serving. info_scale=0 ⇒ count-decay
    # (weight decay**age, jittery at high rank); info_scale>0 ⇒ information-decay
    # (variable window length set by accumulated info). shape ∈ {exp, linear}.
    "exp":    dict(decay=0.95, sd=4.0, ucb_k=1.0, info_scale=0.0, shape="exp"),
    # linear-in-information decay: smooth zero-weight boundary, info-adaptive window.
    # info_scale=16 ⇒ effective info ≈ 8 (= the hard window's i_target), since the
    # linear ramp's area is info_scale/2.
    "lin":    dict(sd=4.0, ucb_k=1.0, info_scale=16.0, shape="linear"),
    # lin + a FIXED population anchor N(pop_mean=14 ≈6k, pop_sd=6.4 = true population
    # SD) for cold start, PLUS the per-update regulariser whose `sd` (the 'initial SD',
    # was 4) is the knob to sweep. The pop anchor sets where a no-history user starts;
    # the regulariser SD tunes per-step responsiveness vs stability.
    "linpop": dict(sd=4.0, ucb_k=1.0, info_scale=16.0, shape="linear",
                   pop_mean=14.0, pop_sd=6.4),
    # local-linear-trend Kalman
    "kalman": dict(v0=2.0, vel_revert=1.0, vel_decay=1.0, q_level=0.02, q_vel=0.03,
                   level_var=9.0, vel_var=2.0, level_floor=1.0, vel_floor=0.05,
                   level_corr_cap=1.5),
}


def build_driver(model, p, start, lapse):
    """Construct any candidate driver from a full param dict `p` (defaults merged
    with sweep overrides). `model` selects the family; momentum is NOT here — it is
    the fixed production baseline, never swept."""
    if model in ("lean", "serve", "ucb"):
        pinfo = 1.0 / float(p["sd"]) ** 2
        if model == "lean":
            return OptWindowDriver(p["i_target"], pinfo, p["optimism"],
                                   start=start, lapse=lapse, name="lean")
        if model == "serve":
            return OptWindowDriver(p["i_target"], pinfo, 0.0, serve_offset=p["offset"],
                                   start=start, lapse=lapse, name="serve")
        return OptWindowDriver(p["i_target"], pinfo, 0.0, start=start, lapse=lapse,
                               ucb_k=p["ucb_k"], name="ucb")
    if model in ("exp", "lin", "linpop"):
        return ExpDecayMLEDriver(decay=p.get("decay", 0.9),
                                 prior_info=1.0 / float(p["sd"]) ** 2,   # regulariser (initial SD)
                                 ucb_k=p["ucb_k"], info_scale=p.get("info_scale", 0.0),
                                 shape=p.get("shape", "exp"),
                                 pop_mean=p.get("pop_mean", None), pop_sd=p.get("pop_sd", 6.4),
                                 start=start, lapse=lapse, name=model)
    if model == "kalman":
        return KalmanDriver(start=start, v0=p["v0"], vel_revert=p["vel_revert"],
                            vel_decay=p["vel_decay"], level_var=p["level_var"],
                            vel_var=p["vel_var"], q_level=p["q_level"], q_vel=p["q_vel"],
                            lapse=lapse, level_var_floor=p["level_floor"],
                            vel_var_floor=p["vel_floor"], level_corr_cap=p["level_corr_cap"])
    raise ValueError(f"unknown model {model!r}")


def make_drivers(args, start):
    """Factory dict {name: () -> driver}. `start` is the cold-start index shared
    by every window estimator (momentum always cold-starts at 0 internally).

      momentum     the OBSOLETE bounded-step climber — replaced in production by the
                   linear-information window MLE; kept only for comparison, NOT a baseline
      plainwin     window MLE, no optimism (δ=0) — isolates what the lean buys
      optwin_lean  the proposal: +δ lean, prior SD = 1/√prior_info (default 2.5 —
                   a TIGHTER prior overshoots, because a tight lean adds ~δ each
                   step deterministically before the window's failures can rein it
                   in; loosening the prior lets evidence stop the climb sooner)
      optwin_serve unbiased estimate, serve +δ harder (decoupled push)
      kalman       local-linear-trend filter: level + velocity, velocity seeded at
                   +v0 ranks/sample so it onboards fast then self-limits
    """
    # --lin-only: the production way of computing rank IS the linear-ramp window MLE,
    # so `baseline_500/16` (WIN_CAP=500, WIN_ITARGET=16) is the BASELINE. We compare it
    # against the two alternative window CONFIGS under study (expand the cap to 3500;
    # lower the info target to 4 so the ramp reaches 0 by ~the 500th sample). The
    # OBSOLETE MomentumDriver is included only as a legacy comparison point — it is
    # NOT the baseline. One framework run prints the standard climb/grid tables with
    # these as the rows.
    if getattr(args, "lin_only", False):
        pinfo = 1.0 / 16          # WIN_PRIOR_SD = 4 → prior_info = 1/16 (production)
        lw = lambda it, cap, nm: (lambda: LinWindowDriver(
            it, cap, pinfo, 1.0, start=start, lapse=args.lapse, name=nm))
        return {
            "momentum":         lambda: MomentumDriver(),
            "baseline_500/16":  lw(16.0, 500, "baseline_500/16"),
            "expand_3500/16":   lw(16.0, 3500, "expand_3500/16"),
            "linpos_500":       lambda: LinPosWindowDriver(
                500, pinfo, 1.0, start=start, lapse=args.lapse, name="linpos_500"),
        }
    return {
        "momentum":     lambda: MomentumDriver(),
        "plainwin":     lambda: OptWindowDriver(args.i_target, args.prior_info, 0.0,
                                                start=start, lapse=args.lapse, name="plainwin"),
        "optwin_lean":  lambda: OptWindowDriver(args.i_target, args.prior_info, args.optimism,
                                                start=start, lapse=args.lapse, name="optwin_lean"),
        "optwin_serve": lambda: OptWindowDriver(args.i_target, args.prior_info, 0.0,
                                                serve_offset=args.optimism, start=start,
                                                lapse=args.lapse, name="optwin_serve"),
        "kalman":       lambda: KalmanDriver(start=start, v0=args.kal_v0,
                                             level_var=args.kal_level_var,
                                             vel_var=args.kal_vel_var,
                                             q_level=args.kal_q_level, q_vel=args.kal_q_vel,
                                             lapse=args.lapse,
                                             level_var_floor=args.kal_level_floor,
                                             vel_var_floor=args.kal_vel_floor),
    }


def get_pool(args, corpus):
    """Full corpus serving pool with the lapse-corrected (80%-calibrated) κ."""
    pool, xs = H.load_problem_pool(corpus, args.size)
    return relapse_pool(pool, args.lapse), xs


def fmt_mci(vals, pct=False, plus=False, dec=1):
    """'mean±halfCI' (95% CI of the mean) for a metric's per-run values."""
    m, ci, _ = mci(vals)
    scale, suf = (100, "%") if pct else (1, "")
    sign = "+" if plus and m >= 0 else ""
    if not math.isfinite(ci):
        return f"{sign}{m*scale:.{dec}f}{suf}"
    return f"{sign}{m*scale:.{dec}f}±{ci*scale:.{dec}f}{suf}"


def fmt_msd(vals, plus=False, dec=1):
    """'mean±SD' — the run-to-run DISTRIBUTION width (sample sd), not the CI of the
    mean. This is what 'mean ± stddev' on a per-user outcome means: the spread a
    real user would actually land in, not our confidence in the average."""
    m, _, sd = mci(vals)
    sign = "+" if plus and m >= 0 else ""
    if not math.isfinite(sd):
        return f"{sign}{m:.{dec}f}"
    return f"{sign}{m:.{dec}f}±{sd:.{dec}f}"


# ---------------------------------------------------------------------------
# Headline: cold-start climb to a strong true ability.
# ---------------------------------------------------------------------------
def run_climb(args, corpus):
    """The headline 'how quick' test: every estimator cold-starts at 20k (index 0)
    and we watch how fast its estimate climbs to a STRONG true ability. Reports the
    three outcomes of interest as DISTRIBUTIONS (mean ± 95% CI over reps): settling
    speed, steady stability (SD of the estimate), and the no-hope serving rate;
    plus a climb chart (mean estimate vs attempt)."""
    pool, xs = get_pool(args, corpus)
    random.seed(args.seed)
    print(f"\nproblem pool: {len(pool)} problems (size {args.size}, "
          f"x range {xs[0]:.1f}..{xs[-1]:.1f}); lapse={args.lapse:g}", file=sys.stderr)

    cps = [c for c in (5, 20, 80, args.attempts) if c <= args.attempts]
    abilities = [int(s) for s in args.climb_abilities.split(",")]

    traces_for_plot = {}
    for theta in abilities:
        print("\n" + "=" * 104)
        print(f"COLD-START CLIMB to true ability index {theta} (~{idx_to_label(theta)}) "
              f"— cold start 20k; {args.reps} reps (mean ± 95% CI).")
        print("=" * 104)
        hdr = (f"{'estimator':<15} " + " ".join(f"@{c:>4}" for c in cps)
               + f" {'rise':>5} {'overshoot':>10} {'n_hard':>8} {'n_floor':>8} "
                 f"{'settle':>7} {'stabil':>7} {'bias':>10} {'clear':>6}")
        print(hdr); print("-" * len(hdr))

        drivers = make_drivers(args, start=0.0)
        plot_set = {}
        for name, make in drivers.items():
            sum_trace = [0.0] * args.attempts
            mets = []
            for _ in range(args.reps):
                drv = make()
                trace, sx, corr, pt, pr = simulate(drv, float(theta), pool, xs,
                                                   args.attempts, args.lapse, 50, random)
                for i, v in enumerate(trace):
                    sum_trace[i] += v
                mets.append(user_metrics(trace, corr, pt, pr, theta, args.conv_eps))
            mean_trace = [s / args.reps for s in sum_trace]
            plot_set[name] = mean_trace
            cells = " ".join(f"{mean_trace[c - 1]:>5.1f}" for c in cps)
            print(f"{name:<15} {cells} "
                  f"{median(m['rise_att'] for m in mets):>5.0f} "
                  f"{fmt_mci([m['overshoot'] for m in mets], plus=True):>10} "
                  f"{fmt_mci([m['n_hard'] for m in mets]):>8} "
                  f"{fmt_mci([m['n_floor'] for m in mets]):>8} "
                  f"{median(m['settle_att'] for m in mets):>7.0f} "
                  f"{mci([m['steady_std'] for m in mets])[0]:>7.2f} "
                  f"{fmt_mci([m['steady_bias'] for m in mets], plus=True):>10} "
                  f"{mci([m['clear'] for m in mets])[0]:>6.0%}")
        traces_for_plot[theta] = plot_set

    print("\n@N = mean estimated rank index after N attempts (1 idx = 1 stone).")
    print("rise = median attempts to FIRST reach within "
          f"{args.conv_eps:g} of TRUTH (speed to your level — where the lean wins).")
    print("overshoot = peak estimate − truth (stones it sailed above you).")
    print(f"n_hard = NUMBER of problems shown the user is likelier to fail (true clear "
          f"< {int(HARD_P*100)}%).")
    print(f"n_floor = NUMBER of problems shown that HIT THE PROBLEM'S FLOOR (skill adds "
          "nothing beyond guessing) — the headline 'hopeless problems served'.")
    print(f"  Both are COUNTS over the {args.attempts}-attempt run — the overshoot's "
          "concrete cost.")
    print("settle = attempts until it stops moving. stabil = steady SD. "
          "bias = settled − truth. clear target 80%.")

    if not args.no_plot:
        plot_climb(traces_for_plot, args)


def plot_climb(traces_for_plot, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    abilities = list(traces_for_plot)
    fig, axes = plt.subplots(1, len(abilities), figsize=(6.4 * len(abilities), 5.0),
                             squeeze=False)
    colors = {"momentum": "#888888", "plainwin": "#1f77b4",
              "optwin_lean": "#d62728", "optwin_lean_cap": "#ff7f0e",
              "optwin_serve": "#2ca02c", "ucb_serve": "#17becf", "kalman": "#9467bd"}
    for ax, theta in zip(axes[0], abilities):
        plot_set = traces_for_plot[theta]
        for name, tr in plot_set.items():
            xs_a = list(range(1, len(tr) + 1))
            ax.plot(xs_a, tr, label=name, color=colors.get(name), lw=2)
        ax.axhline(theta, ls="--", color="black", lw=1, alpha=0.6,
                   label=f"true ({idx_to_label(theta)})")
        ax.set_title(f"cold-start climb to {idx_to_label(theta)} (idx {theta})")
        ax.set_xlabel("attempt #")
        ax.set_ylabel("estimated rank index (stones)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"History-window (I={args.i_target:g}, prior SD={1/math.sqrt(args.prior_info):.2g}, "
                 f"+{args.optimism:g} stone lean) vs momentum — climbing speed",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(os.path.dirname(__file__), args.plot_out)
    fig.savefig(out, dpi=110)
    print(f"\nwrote climb chart -> {out}", file=sys.stderr)


def _runs(make, theta, pool, xs, args, rng):
    out = []
    for _ in range(args.reps):
        trace, sx, corr, pt, pr = simulate(make(), float(theta), pool, xs,
                                           args.attempts, args.lapse, 50, rng)
        out.append(user_metrics(trace, corr, pt, pr, theta, args.conv_eps))
    return out


# ---------------------------------------------------------------------------
# Grid: the three outcomes across true abilities, as distributions.
# ---------------------------------------------------------------------------
def run_grid(args, corpus):
    """Across a grid of true abilities (cold start each), the three outcomes of
    interest as DISTRIBUTIONS (mean ± 95% CI over reps): settling speed, steady
    stability (SD of the estimate), serving harm (no-hope %), plus settled bias
    and realized clear rate."""
    pool, xs = get_pool(args, corpus)
    abilities = [int(s) for s in args.grid_abilities.split(",")]
    rng = random.Random(args.seed)

    print("\n" + "=" * 100)
    print(f"GRID — cold start 20k to each true ability; {args.reps} reps "
          f"(mean ± 95% CI); lapse={args.lapse:g}.")
    print("=" * 100)
    hdr = (f"{'estimator':<15} {'true':>5} {'rise':>5} {'overshoot':>10} {'n_hard':>8} "
           f"{'n_floor':>8} {'settle':>7} {'stabil':>7} {'bias':>10} {'clear':>6}")
    print(hdr); print("-" * len(hdr))
    drivers = make_drivers(args, start=0.0)
    for name, make in drivers.items():
        for theta in abilities:
            mets = _runs(make, theta, pool, xs, args, rng)
            print(f"{name:<15} {theta:>5} "
                  f"{median(m['rise_att'] for m in mets):>5.0f} "
                  f"{fmt_mci([m['overshoot'] for m in mets], plus=True):>10} "
                  f"{fmt_mci([m['n_hard'] for m in mets]):>8} "
                  f"{fmt_mci([m['n_floor'] for m in mets]):>8} "
                  f"{median(m['settle_att'] for m in mets):>7.0f} "
                  f"{mci([m['steady_std'] for m in mets])[0]:>7.2f} "
                  f"{fmt_mci([m['steady_bias'] for m in mets], plus=True):>10} "
                  f"{mci([m['clear'] for m in mets])[0]:>6.0%}")
        print()
    print(f"rise = median attempts to first reach within {args.conv_eps:g} of TRUTH (speed). "
          "overshoot = peak − truth.")
    print(f"n_hard = NUMBER shown with true clear < {int(HARD_P*100)}%; "
          "n_floor = NUMBER that hit the problem's own guessing floor (counts, overshoot harm).")
    print("settle = attempts until estimate stops moving. stabil = steady SD. clear target 80%.")


# ---------------------------------------------------------------------------
# Focused 5-column report (mean ± SD), one table per true ability.
# ---------------------------------------------------------------------------
def run_five(args, corpus):
    """The headline summary for a single 1d (index 20) cold-start climb, each value
    mean ± SD over reps (SD = run-to-run spread — where a real user actually lands,
    NOT the CI of the mean). Columns:

      name                       the estimator
      samples until steady state attempts until the estimate STOPS MOVING (within
                                 conv_eps of its settled value) — onboarding length
      steady state ⅓₁ / ⅓₂ / ⅓₃ the back-half estimate split into thirds. A rising
                                 t1<t2<t3 means it's STILL slowly converging upward
                                 after it nominally settled; flat = truly converged.
                                 Compare to the true index (20) — 1 idx = 1 stone.
      hard served (over 50%)     NUMBER of problems shown the user is likelier to
                                 FAIL than clear (true clear < 50%)
      impossible served          NUMBER of problems shown that HIT THE PROBLEM'S
                                 FLOOR (logistic part ~0 → true clear ≈ the problem's
                                 own guessing floor; skill buys nothing)"""
    pool, xs = get_pool(args, corpus)
    theta = 20                       # 1d — the single representative target
    rng = random.Random(args.seed)
    print(f"\nproblem pool: {len(pool)} problems (size {args.size}, "
          f"x range {xs[0]:.1f}..{xs[-1]:.1f}); lapse={args.lapse:g}", file=sys.stderr)
    drivers = make_drivers(args, start=0.0)
    print("\n" + "=" * 104)
    print(f"COLD-START CLIMB to 1d (true index {theta}) from 20k; "
          f"{args.reps} reps (mean ± SD).")
    print("=" * 104)
    print("steady ⅓ columns are STONES OFF 1d (signed: + above target, − below; 0.0 = "
          "exactly 1d), NOT an absolute index.")
    hdr = (f"{'name':<14} {'samples→steady':>15} {'off 1d ⅓₁':>11} {'off 1d ⅓₂':>11} "
           f"{'off 1d ⅓₃':>11} {'hard served':>13} {'impossible':>12}")
    print(hdr); print("-" * len(hdr))
    for name, make in drivers.items():
        mets = _runs(make, theta, pool, xs, args, rng)
        print(f"{name:<14} "
              f"{fmt_msd([m['settle_att'] for m in mets], dec=0):>15} "
              f"{fmt_msd([m['steady_t1'] - theta for m in mets], plus=True):>11} "
              f"{fmt_msd([m['steady_t2'] - theta for m in mets], plus=True):>11} "
              f"{fmt_msd([m['steady_t3'] - theta for m in mets], plus=True):>11} "
              f"{fmt_msd([m['n_hard'] for m in mets]):>13} "
              f"{fmt_msd([m['n_floor'] for m in mets]):>12}")
    print(f"\nsamples→steady = attempts until the estimate stops moving (within "
          f"{args.conv_eps:g} of its settled value).")
    print(f"off 1d ⅓₁/⅓₂/⅓₃ = mean estimate MINUS the 1d target ({theta}) over the "
          "first/second/last third of the back half,\n  in stones (1 idx = 1 stone). "
          "Trend toward 0 across the three = STILL converging; flat-nonzero = settled "
          "bias.")
    print(f"hard served = NUMBER of problems shown with true clear < {int(HARD_P*100)}% "
          "(likelier to fail than clear).")
    print(f"impossible served = NUMBER of problems shown that HIT THE PROBLEM'S FLOOR "
          f"(praw<{FLOOR_PRAW:g} → true clear ≈ the problem's own guessing floor).")
    print("All values are mean ± SD over reps (SD = run-to-run spread).")


# ---------------------------------------------------------------------------
# Evolve: sweep ONE (or two) knobs of a candidate model at the two EXTREME
# abilities — 15k (idx 5, a 5-stone climb) and 5d (idx 24, a 24-stone climb).
# Same params must serve both, since the cold-start path length is unknown. The
# comparison row chosen by --baseline is printed first (default the OBSOLETE
# momentum — a legacy reference, NOT the production baseline; the production way of
# computing rank is the linear-information window MLE, --baseline lin); a candidate
# "wins" only if it matches or beats that row's rise / hard / bias / settle at BOTH
# extremes.
# ---------------------------------------------------------------------------
def _agg(mets):
    """Per-scenario summary: (mean, STDDEV) over the per-run metric values for each
    metric. The ± shown in the table is the run-to-run STDDEV — the spread a real user
    actually lands in — NOT the CI of the mean. (For the significance of a difference
    between two means, the CI half-width is stddev·1.96/√reps, far tighter.) rise and
    settle are means here, not medians, so the ± pairs with them. floor is dropped from
    the table — it is 0 everywhere in this corpus (no sharp problems above ~1d)."""
    def ms(key):
        m, _, sd = mci([d[key] for d in mets])
        return (m, sd)
    return dict(
        rise=ms("rise_att"), hard=ms("n_hard"), over=ms("overshoot"),
        bias=ms("steady_bias"), std=ms("steady_std"),
        settle2=ms("settle2"), settle1=ms("settle1"), clear=ms("clear"),
    )


# Shared fixed-width column spec — header AND data rows are both built from this one
# list, so they can never drift out of alignment. Each printed block is one scenario.
# Each cell is mean±stddev; the format lambda takes (mean, stddev).
#   (metric key in _agg, header text, width, formatter(mean, stddev))
_SHOW_COLS = [
    ("rise",   "rise",     8, lambda m, s: f"{m:.0f}±{s:.0f}"),
    ("hard",   "hard",     9, lambda m, s: f"{m:.1f}±{s:.1f}"),
    ("over",   "over",    11, lambda m, s: f"{m:+.1f}±{s:.1f}"),
    ("bias",   "bias",    10, lambda m, s: f"{m:+.1f}±{s:.1f}"),
    ("std",     "std",     9, lambda m, s: f"{m:.1f}±{s:.1f}"),
    ("settle2", "sett<2",  8, lambda m, s: f"{m:.0f}±{s:.0f}"),
    ("settle1", "sett<1",  8, lambda m, s: f"{m:.0f}±{s:.0f}"),
    ("clear",   "clr",     9, lambda m, s: f"{m:.0%}±{s:.0%}"),
]
_RL_W = 22                                          # row-label column width
_HDR = "| " + " ".join(f"{h:>{w}}" for _, h, w, _ in _SHOW_COLS)
_FULL_W = _RL_W + len(_HDR)


def _block_cells(r):
    return "| " + " ".join(f"{fmt(*r[key]):>{w}}" for key, _, w, fmt in _SHOW_COLS)


def _print_block_table(scenarios, rows, sep_after_first=False):
    """STACKED layout: one sub-table per scenario. A side-by-side table (one block per
    scenario) is ~160 cols and wraps in the terminal, which is what made the scenario
    headers look misaligned; stacking keeps each table ≤70 cols and trivially aligned.
    rows: list of (label, {scn_label: agg})."""
    for sc in scenarios:
        print(f"\n  ▸ {sc['label']} — {sc.get('desc', '')}")
        print(f"{'estimator':<{_RL_W}}{_HDR}")
        print("-" * _FULL_W)
        for i, (label, res) in enumerate(rows):
            print(f"{label:<{_RL_W}}{_block_cells(res[sc['label']])}")
            if sep_after_first and i == 0:
                print("-" * _FULL_W)


def _runset_scn(make, scenarios, pool, xs, args, rng):
    """Aggregate metrics for one driver factory across all scenarios (each scenario =
    a true ability plus optional early-failure-burst onboarding noise)."""
    res = {}
    for sc in scenarios:
        mets = []
        for _ in range(args.reps):
            tr, sx, corr, pt, pr = simulate(make(), float(sc["theta"]), pool, xs,
                                            args.attempts, args.lapse, 50, rng,
                                            early_fail_rate=sc.get("ef_rate", 0.0),
                                            early_fail_n=sc.get("ef_n", 0))
            mets.append(user_metrics(tr, corr, pt, pr, sc["theta"], args.conv_eps))
        res[sc["label"]] = _agg(mets)
    return res


def _scenarios(args, with_noise):
    """Test scenarios: the configured abilities (clean onboarding), plus optionally a
    1d user with a 15%-random-failure burst over the first 20 problems."""
    scn = []
    for s in args.evolve_abilities.split(","):
        t = int(s)
        scn.append(dict(theta=t, label=idx_to_label(t), ef_rate=0.0, ef_n=0,
                        desc=f"true {idx_to_label(t)} (idx {t}); {t}-stone climb from 20k"))
    if with_noise:
        scn.append(dict(theta=20, label="1d·n15", ef_rate=0.15, ef_n=20,
                        desc="true 1d (idx 20); 15% random fails over first 20 problems"))
    return scn


def run_evolve(args, corpus):
    pool, xs = get_pool(args, corpus)
    scenarios = _scenarios(args, with_noise=args.with_noise)
    rng = random.Random(args.seed)
    model = args.evolve
    base = dict(DRIVER_DEFAULTS[model])

    def vals(spec):
        return [float(s) for s in spec.split(",")] if spec else []

    if args.p1:
        v1 = vals(args.v1)
        if args.p2:
            combos = [{args.p1: a, args.p2: b} for a in v1 for b in vals(args.v2)]
        else:
            combos = [{args.p1: a} for a in v1]
    else:
        combos = [{}]                                # no sweep: just the defaults

    sweep_desc = (f"{args.p1}×{args.p2}" if args.p2 else args.p1) if args.p1 else "(defaults)"
    print("\n" + "=" * _FULL_W)
    print(f"EVOLVE {model} — sweep {sweep_desc}; {args.reps} reps. "
          f"every cell = mean±stddev (run-to-run spread).")
    print("Beat the baseline row (--baseline): lower rise, lower hard, |bias|→0, lower std/sett.")
    print("=" * _FULL_W)

    if args.baseline == "momentum":
        base_make, base_label = (lambda: MomentumDriver()), "momentum(base)"
    else:
        base_make = lambda: build_driver(args.baseline, dict(DRIVER_DEFAULTS[args.baseline]),
                                         0.0, args.lapse)
        base_label = f"{args.baseline}(base)"
    rows = [(base_label, _runset_scn(base_make, scenarios, pool, xs, args, rng))]
    for c in combos:
        p = {**base, **c}
        make = lambda p=p: build_driver(model, p, 0.0, args.lapse)
        label = ",".join(f"{k}={v:g}" for k, v in c.items()) or "defaults"
        rows.append((label, _runset_scn(make, scenarios, pool, xs, args, rng)))
    _print_block_table(scenarios, rows, sep_after_first=True)
    print(f"\nEvery cell mean±STDDEV over {args.reps} runs (run-to-run spread; CI of the "
          f"mean = stddev·1.96/√{args.reps}). rise=attempts to within conv_eps of TRUTH; "
          "hard=NUMBER served\ntrue clear<50%; over=peak−truth; bias=settled−truth; "
          "std=steady jitter SD; sett<2 / sett<1=attempts until the estimate stays within 2 / 1 stones of its settled value; clr=2nd-half clear (~80%)."
          "'·n15'=15% early-fail burst.")


def run_showdown(args, corpus):
    """Final head-to-head: the --baseline comparison row (default the OBSOLETE
    momentum loop — kept only for reference, NOT the production baseline; the
    production estimator is the linear-information window MLE) vs the best-tuned
    candidate of each family (defaults in DRIVER_DEFAULTS). One row per estimator,
    all metrics, over the two extreme abilities PLUS the noisy-onboarding 1d scenario."""
    pool, xs = get_pool(args, corpus)
    scenarios = _scenarios(args, with_noise=True)
    rng = random.Random(args.seed)

    print("\n" + "=" * _FULL_W)
    print(f"SHOWDOWN — momentum (fixed) vs best-tuned candidates; {args.reps} reps.")
    print("=" * _FULL_W)

    rows = [("momentum", _runset_scn(lambda: MomentumDriver(), scenarios, pool, xs, args, rng))]
    for m in ("ucb", "lin"):
        make = lambda m=m: build_driver(m, dict(DRIVER_DEFAULTS[m]), 0.0, args.lapse)
        rows.append((m, _runset_scn(make, scenarios, pool, xs, args, rng)))
    _print_block_table(scenarios, rows, sep_after_first=True)
    print(f"\nEvery cell mean±STDDEV over {args.reps} runs (run-to-run spread; CI of the "
          f"mean = stddev·1.96/√{args.reps}). rise=attempts to within conv_eps of TRUTH; "
          "hard=NUMBER served\ntrue clear<50%; over=peak−truth; bias=settled−truth; "
          "std=steady jitter SD; sett<2 / sett<1=attempts until the estimate stays within 2 / 1 stones of its settled value; clr=2nd-half clear (~80%)."
          "'1d·n15'=1d user, 15% random fails over first 20 problems.")


# ---------------------------------------------------------------------------
# Sweep: does an APPROPRIATE (prior SD, info window) make the +δ lean converge?
# ---------------------------------------------------------------------------
def run_sweep(args, corpus):
    """The lean's two knobs, swept against each other (the test of whether the +δ
    prior CAN be overridden by results with the right balance): prior SD (stones)
    × information-window length I. For each cell the settled outcome of optwin_lean
    at a mid (idx 11 ~ 9k) and a strong (idx 20 ~ 1d) true ability. A cell is the
    sweet spot iff it settles FAST, near truth+δ (bias ≈ +δ, not +20), STEADY, with
    a LOW no-hope rate. prior_info = 1/SD²; a tight SD (small) at +δ drags hardest."""
    pool, xs = get_pool(args, corpus)
    rng = random.Random(args.seed)
    sds = [float(s) for s in args.sweep_sd.split(",")]
    its = [float(s) for s in args.sweep_i.split(",")]
    abilities = [int(s) for s in args.sweep_abilities.split(",")]

    for theta in abilities:
        print("\n" + "=" * 100)
        print(f"LEAN SWEEP — true ability idx {theta} (~{idx_to_label(theta)}), "
              f"δ=+{args.optimism:g} stones; {args.reps} reps (mean ± 95% CI).")
        print(f"  ideal: settles fast, bias ≈ +{args.optimism:g}, low stabil, low nohope.")
        print("=" * 100)
        hdr = (f"{'prior_SD':>8} {'info_I':>7} {'rise':>5} {'overshoot':>10} "
               f"{'n_hard':>8} {'n_floor':>8} {'settle':>7} {'stabil':>7} {'bias':>10}")
        print(hdr); print("-" * len(hdr))
        for sd in sds:
            for I in its:
                pinfo = 1.0 / (sd * sd)
                make = (lambda sd=sd, I=I, pinfo=pinfo: OptWindowDriver(
                    I, pinfo, args.optimism, start=0.0, lapse=args.lapse,
                    name="lean"))
                mets = _runs(make, theta, pool, xs, args, rng)
                print(f"{sd:>8.2f} {I:>7.0f} "
                      f"{median(m['rise_att'] for m in mets):>5.0f} "
                      f"{fmt_mci([m['overshoot'] for m in mets], plus=True):>10} "
                      f"{fmt_mci([m['n_hard'] for m in mets]):>8} "
                      f"{fmt_mci([m['n_floor'] for m in mets]):>8} "
                      f"{median(m['settle_att'] for m in mets):>7.0f} "
                      f"{mci([m['steady_std'] for m in mets])[0]:>7.2f} "
                      f"{fmt_mci([m['steady_bias'] for m in mets], plus=True):>10}")
            print()
    print(f"info_I is the LAG knob: a smaller window sees failures sooner, so it should cut "
          "the overshoot\n(and n_hard/n_floor) at the cost of steady stabil. The sweet spot "
          "rises fast, overshoots little,\nshows few floor problems, settles near +δ, steady. "
          "prior_info = 1/SD².")


# ---------------------------------------------------------------------------
# How many reps? CI half-width vs N for the outcomes of interest.
# ---------------------------------------------------------------------------
def run_repscurve(args, corpus):
    """Answers 'how many runs for tight bounds?'. Runs a large N once for a couple
    of representative configs, then reports the 95%-CI half-width each outcome
    WOULD have at N ∈ {15,30,60,120,240} (= 1.96·SD/√N from the measured SD). The
    half-width shrinks as 1/√N, so this reads straight off one big sample."""
    pool, xs = get_pool(args, corpus)
    rng = random.Random(args.seed)
    N = max(args.reps, 300)
    configs = [
        ("momentum θ=20", lambda: MomentumDriver(), 20),
        (f"optwin_lean θ=20 (SD=1.5,I=8)",
         lambda: OptWindowDriver(8.0, 1.0 / 1.5**2, args.optimism, start=0.0,
                                 lapse=args.lapse, name="lean"), 20),
    ]
    Ns = [15, 30, 60, 120, 240]
    print("\n" + "=" * 92)
    print(f"REPS vs CONFIDENCE — measured SD over N={N} runs, then CI half-width at "
          f"smaller N (∝1/√N).")
    print("=" * 92)
    args_N = argparse.Namespace(**{**vars(args), "reps": N})
    for label, make, theta in configs:
        mets = _runs(make, theta, pool, xs, args_N, rng)
        print(f"\n{label}:")
        for key in ("rise_att", "overshoot", "settle_att", "n_hard", "n_floor", "steady_std"):
            vals = [m[key] for m in mets]
            _, _, sd = mci(vals)
            row = "  ".join(f"N={n}:±{1.96*sd/math.sqrt(n):.2f}" for n in Ns)
            print(f"  {key:<12} SD={sd:.2f}  ->  {row}")
    print("\nPick N where the half-width is small vs the effect you're comparing. For "
          "settle/bias\n(~stones) and nohope (~%), N≈30 already gives ±a fraction of a "
          "stone / a couple %%; N≈120 halves it.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="9", choices=["9", "13"],
                    help="board size whose corpus drives the simulated serving pool")
    ap.add_argument("--i-target", type=float, default=8.0,
                    help="constant Fisher-information window length")
    ap.add_argument("--prior-info", type=float, default=1.0 / 2.5 ** 2,
                    help="precision of the forward-leaning prior (= 1/SD²; default SD=2.5 "
                         "stones. A TIGHTER prior (smaller SD) adds ~δ deterministically each "
                         "step before the window can react, which is what causes the big "
                         "onboarding overshoot; SD<1 runs the lean clean off to the clamp)")
    ap.add_argument("--optimism", type=float, default=2.0,
                    help="δ: stones the prior mean (lean) / served problem (serve) "
                         "sits above the current estimate")
    ap.add_argument("--step-cap", type=float, default=1.5,
                    help="trust region: max stones the capped-lean estimate may move "
                         "per attempt (keeps the serving loop coupled)")
    ap.add_argument("--kal-v0", type=float, default=2.0,
                    help="kalman initial VELOCITY (ranks/sample) — the fast-progress bias "
                         "that lets a cold-start user climb ~2 stones/attempt")
    ap.add_argument("--kal-level-var", type=float, default=9.0,
                    help="kalman initial LEVEL variance at cold start (stones^2)")
    ap.add_argument("--kal-vel-var", type=float, default=2.0,
                    help="kalman initial VELOCITY variance (stones^2/sample^2)")
    ap.add_argument("--kal-q-level", type=float, default=0.02,
                    help="kalman level process-noise per attempt (stones^2)")
    ap.add_argument("--kal-q-vel", type=float, default=0.05,
                    help="kalman velocity process-noise per attempt (stones^2) — larger "
                         "lets the velocity adapt (decay) faster once failures arrive")
    ap.add_argument("--kal-level-floor", type=float, default=1.0,
                    help="kalman level posterior-variance floor (keeps it nimble)")
    ap.add_argument("--kal-vel-floor", type=float, default=0.05,
                    help="kalman velocity posterior-variance floor (keeps it able to "
                         "re-accelerate for an improving user)")
    ap.add_argument("--ucb-k", type=float, default=3.0,
                    help="ucb_serve: serve at estimate + ucb_k·SE (uncertainty-scaled "
                         "optimism — the self-limiting form of the lean)")
    ap.add_argument("--pop-prior", type=float, default=11.0,
                    help="ucb_serve population prior anchor (index; ~9k), regularises "
                         "the all-win MLE to finite")
    ap.add_argument("--lapse", type=float, default=None,
                    help="4PL upper-asymptote slip rate (default = learned fat_finger_rate "
                         "from ratings/rating_constants.json, ~0.0347)")
    ap.add_argument("--reps", type=int, default=200, help="simulated users per ability")
    ap.add_argument("--attempts", type=int, default=200, help="attempts per simulated user")
    ap.add_argument("--conv-eps", type=float, default=1.0,
                    help="rank tolerance for 'converged' (stones)")
    ap.add_argument("--climb-abilities", default="16,20,24",
                    help="true indices for the headline climb test (16~4k,20~1d,24~5d)")
    ap.add_argument("--grid-abilities", default="6,11,16,20,24",
                    help="true indices for the grid table")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--lin-only", action="store_true",
                    help="run the climb/grid with the BASELINE = the production linear-ramp "
                         "window MLE (baseline 500/16) against the two alternative window "
                         "configs (expand 3500/16, lowinfo 500/4), plus the obsolete momentum "
                         "loop as a legacy reference — needs --attempts > 500 for the cap to bind")
    ap.add_argument("--no-grid", action="store_true", help="skip the grid table")
    ap.add_argument("--no-climb", action="store_true", help="skip the climb test")
    ap.add_argument("--no-plot", action="store_true", help="don't write the climb PNG")
    ap.add_argument("--plot-out", default="history_rank_climb.png")
    ap.add_argument("--sweep", action="store_true",
                    help="run ONLY the (prior SD × info window) lean sweep and exit")
    ap.add_argument("--sweep-sd", default="0.6,1.0,1.5,2.5,4.0",
                    help="prior standard deviations (stones) for --sweep")
    ap.add_argument("--sweep-i", default="1,2,4,8,16,32",
                    help="information-window lengths for --sweep (the lag knob)")
    ap.add_argument("--sweep-abilities", default="11,20",
                    help="true indices for --sweep (mid + strong)")
    ap.add_argument("--reps-curve", action="store_true",
                    help="run ONLY the reps-vs-confidence diagnostic and exit")
    ap.add_argument("--five", action="store_true",
                    help="run ONLY the focused 5-column (mean ± SD) report and exit")
    ap.add_argument("--evolve", choices=["lean", "serve", "ucb", "exp", "lin", "linpop", "kalman"],
                    help="run ONLY the evolve sweep for this candidate model and exit")
    ap.add_argument("--baseline", default="momentum",
                    choices=["momentum", "lin", "ucb", "lean", "serve", "exp", "linpop", "kalman"],
                    help="evolve comparison row to sweep against. NOTE the production baseline is "
                         "the window MLE ('lin'); 'momentum' (the default, for back-compat) is the "
                         "OBSOLETE loop kept only as a legacy reference, not a baseline")
    ap.add_argument("--p1", help="evolve: name of the first parameter to sweep")
    ap.add_argument("--v1", help="evolve: comma-separated values for --p1")
    ap.add_argument("--p2", help="evolve: optional second parameter (2-D grid)")
    ap.add_argument("--v2", help="evolve: comma-separated values for --p2")
    ap.add_argument("--evolve-abilities", default="5,24",
                    help="evolve/showdown: true indices to test (default 5,24 = 15k & 5d extremes)")
    ap.add_argument("--showdown", action="store_true",
                    help="run ONLY the momentum-vs-best-candidates head-to-head and exit")
    ap.add_argument("--with-noise", action="store_true",
                    help="evolve: also test the noisy-onboarding 1d scenario (15% early fails)")
    args = ap.parse_args()

    if args.lapse is None:
        args.lapse = load_lapse()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    print(f"loading corpus from {os.path.join(PROJECT_ROOT, 'ratings')} ... "
          f"(lapse={args.lapse:g})", file=sys.stderr)
    corpus = H.Corpus(PROJECT_ROOT)

    if args.reps_curve:
        run_repscurve(args, corpus)
        return
    if args.five:
        run_five(args, corpus)
        return
    if args.evolve:
        run_evolve(args, corpus)
        return
    if args.showdown:
        run_showdown(args, corpus)
        return
    if args.sweep:
        run_sweep(args, corpus)
        return
    if not args.no_climb:
        run_climb(args, corpus)
    if not args.no_grid:
        run_grid(args, corpus)


if __name__ == "__main__":
    main()
