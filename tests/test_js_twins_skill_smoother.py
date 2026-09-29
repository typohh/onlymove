"""Career skill smoother twins: the likelihood the pipeline fits the smoother's
constants by, against the client filter that uses them.

  log_lik (per player)   utils/rating/skill_smoother.py  <->  skillLogLik (pwa/js/skill-smoother.js)
  flatten_attempt + clear_kappa                          <->  flattenAttempt (pwa/shared/rank-window.js)

The Python filter steps every player in lockstep with array operations, the
client one attempt at a time; both stop each attempt's Laplace iteration at the
same tolerance, so the summed log-likelihoods agree to rounding.
"""

import math
import random

import numpy as np

from utils.rating import skill_smoother as sk

SMOOTHER = "pwa/js/skill-smoother.js"
IRT = "pwa/js/irt.js"
RANK_WINDOW = "pwa/shared/rank-window.js"

HOUR = 3600 * 1000
DAY = 24 * HOUR


def _histories(lapse, seed=5):
    """Synthetic careers whose gaps span a minute to half a year, so every branch —
    within a session, a new session, a step long enough to meet the cap — is hit."""
    rng = random.Random(seed)
    users = []
    for _ in range(12):
        n = rng.randint(1, 120)
        t, skill = 1.7e12, rng.uniform(-20, 5)
        u = {k: [] for k in ("t", "x", "slope", "floor", "correct")}
        for _ in range(n):
            t += rng.choice([60e3, 5 * 60e3, 2 * HOUR, 7 * HOUR, 2 * DAY, 20 * DAY, 180 * DAY])
            skill += rng.gauss(0, 0.1)
            x = skill + rng.gauss(0, 2)
            slope = rng.uniform(0.2, 2.5)
            floor = rng.choice([0.0, rng.uniform(0, 0.7)])
            kap = float(sk.clear_kappa(floor, lapse))
            p = floor + (1 - floor - lapse) / (1 + math.exp(-(slope * (skill - x) + kap)))
            u["t"].append(t); u["x"].append(x); u["slope"].append(slope)
            u["floor"].append(floor); u["correct"].append(1.0 if rng.random() < p else 0.0)
        users.append(u)
    return users


PARAMS = [
    (0.04, 1.0, 0.9, -7.0, 6.4),
    (0.001, 0.05, 0.3, 0.0, 2.0),
    (0.5, 40.0, 2.0, -15.0, 10.0),
    (0.06, 1e9, 0.01, 3.0, 1.0),
]


def test_log_lik_agrees(js):
    lapse = js.const(IRT, "LAPSE")
    users = _histories(lapse)
    calls, expected = [], []
    for q, c, f, pm, ps in PARAMS:
        jp = {"driftVarPerDay": q, "gapVarCap": c, "formSd": f, "priorMean": pm, "priorSd": ps}
        for u in users:
            att = [{"t": u["t"][i], "x": u["x"][i], "slope": u["slope"][i],
                    "floor": u["floor"][i], "correct": u["correct"][i],
                    "kappa": float(sk.clear_kappa(u["floor"][i], lapse))}
                   for i in range(len(u["t"]))]
            calls.append((SMOOTHER, "skillLogLik", (att, jp)))
            expected.append(sk.log_lik(sk.Population([u], lapse), q, c, f, pm, ps))
    got = js.run(calls)
    for j, p in zip(got, expected):
        assert math.isclose(j, p, rel_tol=1e-9, abs_tol=1e-9), (j, p)


def test_log_lik_is_additive_over_players(js):
    """The lockstep layout must not mix players: the whole population's
    likelihood is the sum of each player's alone."""
    lapse = js.const(IRT, "LAPSE")
    users = _histories(lapse, seed=9)
    q, c, f, pm, ps = PARAMS[0]
    whole = sk.log_lik(sk.Population(users, lapse), q, c, f, pm, ps)
    parts = sum(sk.log_lik(sk.Population([u], lapse), q, c, f, pm, ps) for u in users)
    assert math.isclose(whole, parts, rel_tol=1e-12)


def test_flatten_agrees(js):
    rng = random.Random(3)
    ratings = sorted(rng.uniform(-3, 3) for _ in range(12))
    ranks = list(np.cumsum([rng.uniform(0.5, 4) for _ in ratings]) - 25)
    curve = {"ratings": ratings, "ranks": ranks}
    lapse = js.const(IRT, "LAPSE")
    calls, expected = [], []
    for _ in range(60):
        r, a, fl = rng.uniform(-4, 4), rng.uniform(0.1, 3), rng.uniform(-0.1, 0.9)
        calls.append((RANK_WINDOW, "flattenAttempt", (r, a, fl, True, curve)))
        x, s, f = sk.flatten_attempt(r, a, fl, (ratings, ranks))
        expected.append((x, s, f, float(sk.clear_kappa(f, lapse))))
    for j, (x, s, f, k) in zip(js.run(calls), expected):
        for jv, pv in ((j["x"], x), (j["slope"], s), (j["floor"], f), (j["kappa"], k)):
            assert math.isclose(jv, pv, rel_tol=1e-9, abs_tol=1e-12), (j, x, s, f, k)


def test_session_gap_agrees(js):
    assert js.const(SMOOTHER, "SESSION_GAP_MS") == sk.SESSION_GAP_MS
