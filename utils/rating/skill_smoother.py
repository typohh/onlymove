"""The career skill smoother's likelihood, for fitting its population constants.

Mirror of the FORWARD filter in `pwa/js/skill-smoother.js` (`skillLogLik`); the
client's backward smoothing pass has no mirror, since only the filter's one-step-
ahead predictions enter the fit. `tests/test_js_twins_skill_smoother.py` holds the
two to the same answer.

The model, stated once here for the fit's side:

  skill      a random walk in calendar time. The variance a step of `days` adds is
             C·(1 − exp(−q·days / C)): q·days for a short step, saturating at the
             cap C for a long one — change accrues through play, and an absence
             adds a bounded one-off uncertainty rather than one that grows with it.
  form       an offset redrawn from N(0, form_sd²) when a gap of SESSION_GAP_MS or
             more opens a session, and held for the session.
  attempt    P(correct) = c + (1 − c − lapse)·σ(slope·(S + F − x) + κ(c)).
  start      the population rank prior's mean and SD.

`fit` chooses (q, C, form_sd) by maximum likelihood of every first-time attempt's
outcome given the same player's attempts before it. Stdlib + numpy only.
"""

import math

import numpy as np

from utils.rating.ratings_io import apply_M
from utils.rating.se_floor import rank_space_slope

DAY_MS = 24 * 3600 * 1000
# A gap at least this long opens a session (pwa/js/skill-smoother.js SESSION_GAP_MS).
SESSION_GAP_MS = 6 * 3600 * 1000

_CALIBRATION_P = 0.8
_FLOOR_MAX = 0.78


def clear_kappa(c, lapse):
    """κ(c) of pwa/js/irt.js clearKappa: P(clear) = CALIBRATION_P at a zero gap."""
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    q = np.clip((_CALIBRATION_P - c) / (1.0 - c - lapse), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def flatten_attempt(rating, a_p, floor_raw, curve):
    """(x, slope, floor) in internal-rank space — flattenAttempt
    (pwa/shared/rank-window.js) without κ, which the caller adds with its lapse.
    `curve` is (ratings, ranks) for the attempt's board size."""
    cr, ck = curve
    return (apply_M(rating, cr, ck),
            float(rank_space_slope(rating, a_p, cr, ck)),
            float(min(max(floor_raw, 0.0), _FLOOR_MAX)))


def _sigmoid(z):
    """The client's numerically stable logistic (pwa/js/math-utils.js sigmoid)."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


class Population:
    """Every player's attempts, laid out for a filter that steps all players in
    lockstep. Players are ordered longest first, so at attempt index j the active
    players are a PREFIX of that order and step j's inputs are one contiguous slice.

    `users` is a list of per-player dicts of equal-length sequences: t (ms, sorted),
    x, slope, floor, correct (0/1)."""

    def __init__(self, users, lapse):
        users = [u for u in users if len(u["t"])]
        users.sort(key=lambda u: -len(u["t"]))
        self.n_users = len(users)
        self.n_attempts = sum(len(u["t"]) for u in users)
        lens = np.array([len(u["t"]) for u in users])
        max_len = int(lens.max()) if users else 0
        self.active = np.array([(lens > j).sum() for j in range(max_len)], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.active)])

        def lay(key, prev=False):
            out = np.empty(self.n_attempts)
            for j in range(max_len):
                n = self.active[j]
                seg = out[self.offsets[j]:self.offsets[j] + n]
                for i in range(n):
                    seg[i] = users[i][key][j - 1 if prev else j]
            return out

        t = lay("t")
        # Days since the same player's previous attempt (0 at a career's start),
        # and whether that gap opens a session.
        dt = np.zeros(self.n_attempts)
        for j in range(1, max_len):
            a, n = self.offsets[j], self.active[j]
            b = self.offsets[j - 1]
            dt[a:a + n] = t[a:a + n] - t[b:b + n]
        self.days = dt / DAY_MS
        self.reset = dt >= SESSION_GAP_MS
        self.x = lay("x")
        self.slope = lay("slope")
        self.floor = lay("floor")
        self.correct = lay("correct")
        self.span = np.maximum(0.0, 1.0 - self.floor - lapse)
        self.kappa = clear_kappa(self.floor, lapse)


def log_lik(pop, drift_var_per_day, gap_var_cap, form_sd, prior_mean, prior_sd):
    """Summed log predictive probability of every attempt given the same player's
    attempts before it — skillLogLik (pwa/js/skill-smoother.js), per player, summed."""
    n = pop.n_users
    f2 = form_sd * form_sd
    mS = np.full(n, float(prior_mean)); mF = np.zeros(n)
    pSS = np.full(n, float(prior_sd) ** 2); pSF = np.zeros(n); pFF = np.full(n, f2)
    total = 0.0
    for j in range(len(pop.active)):
        k = pop.active[j]
        s = slice(pop.offsets[j], pop.offsets[j] + k)
        mS, mF, pSS, pSF, pFF = mS[:k], mF[:k], pSS[:k], pSF[:k], pFF[:k]
        if j > 0:
            pSS = pSS + (-np.expm1(-drift_var_per_day * pop.days[s] / gap_var_cap)) * gap_var_cap
            r = pop.reset[s]
            mF = np.where(r, 0.0, mF); pSF = np.where(r, 0.0, pSF); pFF = np.where(r, f2, pFF)
        x, sl, c, span, kap, y = (pop.x[s], pop.slope[s], pop.floor[s], pop.span[s],
                                  pop.kappa[s], pop.correct[s])
        mu = mS + mF
        Pu = pSS + 2 * pSF + pFF

        z_pred = (sl * (mu - x) + kap) / np.sqrt(1 + math.pi * sl * sl * Pu / 8)
        p_correct = c + span * _sigmoid(z_pred)
        p_obs = np.where(y > 0, p_correct, 1 - p_correct)
        total += float(np.log(np.maximum(p_obs, 1e-300)).sum())

        # Laplace update of u = S + F: Fisher scoring to the mode, each element
        # stopping on its own at the client's tolerance.
        u = mu.copy(); h = np.zeros(k); live = np.ones(k, dtype=bool)
        for _ in range(30):
            if not live.any():
                break
            praw = _sigmoid(sl * (u - x) + kap)
            p = np.clip(c + span * praw, 1e-9, 1 - 1e-9)
            g = span * sl * praw * (1 - praw)
            hh = g * g / (p * (1 - p))
            score = (y - p) * g / (p * (1 - p)) - (u - mu) / Pu
            step = np.clip(score / (1 / Pu + hh), -2.0, 2.0)
            h = np.where(live, hh, h)
            u = np.where(live, u + step, u)
            live &= np.abs(step) >= 1e-7
        Pu2 = 1 / (1 / Pu + h)
        h_eff = 1 / Pu2 - 1 / Pu
        upd = h_eff > 1e-12
        R = np.where(upd, 1 / np.where(upd, h_eff, 1.0), 1.0)
        zz = R * (u / Pu2 - mu / Pu)
        Sv = Pu + R
        kS = (pSS + pSF) / Sv; kF = (pSF + pFF) / Sv
        innov = zz - mu
        mS = np.where(upd, mS + kS * innov, mS)
        mF = np.where(upd, mF + kF * innov, mF)
        pSS = np.where(upd, pSS - kS * kS * Sv, pSS)
        pSF = np.where(upd, pSF - kS * kF * Sv, pSF)
        pFF = np.where(upd, pFF - kF * kF * Sv, pFF)
    return total


def fit(pop, prior_mean, prior_sd, start=(0.04, 1.0, 1.0)):
    """Maximum-likelihood (drift_var_per_day, gap_var_cap, form_sd): Nelder-Mead
    on their logs, from `start` (a starting point only — the optimum does not
    depend on it). Returns (params dict, log-likelihood, evaluations)."""
    from scipy.optimize import minimize

    def nll(v):
        return -log_lik(pop, math.exp(v[0]), math.exp(v[1]), math.exp(v[2]),
                        prior_mean, prior_sd)

    # An explicit starting simplex half a unit wide in log space: scipy's default
    # perturbs a coordinate by a fraction of its value, which is next to nothing
    # for a log at zero, and the search then stops where it started.
    x0 = np.array([math.log(v) for v in start])
    simplex = np.vstack([x0] + [x0 + 0.5 * np.eye(3)[i] for i in range(3)])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 0.02, "fatol": 0.05, "initial_simplex": simplex})
    if not res.success:
        raise RuntimeError(f"skill smoother fit did not converge: {res.message}")
    q, c, f = (math.exp(v) for v in res.x)
    return ({"drift_var_per_day": q, "gap_var_cap": c, "form_sd": f},
            -float(res.fun), int(res.nfev))
