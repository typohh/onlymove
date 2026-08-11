#!/usr/bin/env python3
"""Is the per-user rank slope (`beta_user`) a real between-user effect, or noise?

The axis model carries a per-user intercept AND a per-user slope on problem rank
alongside the 4PL offset, on the argument that the 4PL gives every user the same
per-problem slope and so cannot express a user whose personal curve is flatter or
steeper than the population's — leaving that variation for the axes to absorb if
nothing else claims it. That is an argument. This measures it.

The test is the one fit_population_constants already applies to an axis before
shipping it: separate the observed spread of the fitted coefficient into a
genuine between-user component and the estimation noise that inflates it. For
each user the design is exactly the model's — the frozen 4PL as an offset, then
[1, rank] by IRLS — and the Fisher information of that fit gives each estimate's
own variance. Then

    var(beta)_observed  =  var(beta)_between  +  mean estimation variance

so the between-user variance is the observed spread minus the noise floor, and
the ratio of the two is what says whether individual users are distinguishable at
all. Below about 1 the coefficient is not identified per user: everyone's
estimate is their own sampling error around a common value, and giving each user
their own parameter costs degrees of freedom to model nothing.

Reported alongside is the same split for the INTERCEPT, which serves as a
positive control — per-user level is unambiguously real (it is most of what
ability means), so an intercept ratio near the slope's would indicate the
estimator, not the effect, is the problem.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_user_rank_slope
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import (  # noqa: E402
    CALIBRATION_P, FLOOR_MAX, MIN_USER_ATTEMPTS, _fit_theta, _irls_off, _p4pl,
    build_axis_features, load_axis_attempts,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402


def main():
    t0 = time.time()
    _feats, rank, params, _ad = build_axis_features()
    lapse = load_client_lapse(0.0328)
    rows_by_user = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in rank:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
    print(f"loaded in {time.time() - t0:.0f}s: {len(users)} users", flush=True)

    coefs, varis, ns = [], [], []
    for u in users:
        a = rows_by_user[u]
        tids = [t for t, _ in a]
        y = np.array([s for _, s in a], dtype=float)
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        th = _fit_theta(R0, A, C, span, kap, y, grid)
        P4 = np.clip(_p4pl(th, R0, A, C, span, kap), 1e-9, 1 - 1e-9)
        off = np.log(P4 / (1 - P4))
        rk = np.array([rank[t] for t in tids])
        X = np.column_stack([np.ones(len(tids)), rk])
        w = _irls_off(X, y, off, ridge=1e-6)
        # Fisher information of the fitted logistic: X' diag(p(1-p)) X. Its
        # inverse diagonal is each coefficient's own estimation variance, which
        # is the noise floor the observed spread has to clear.
        p = 1.0 / (1.0 + np.exp(-np.clip(off + X @ w, -35, 35)))
        W = p * (1.0 - p)
        try:
            cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2))
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
            continue
        coefs.append(w)
        varis.append(np.diag(cov))
        ns.append(len(a))
    coefs = np.array(coefs)
    varis = np.array(varis)
    ns = np.array(ns)
    print(f"{len(coefs)} users with a usable fit "
          f"(median {int(np.median(ns))} attempts)\n")

    print(f"{'':<12}{'mean':>9}{'obs sd':>9}{'noise sd':>10}"
          f"{'between sd':>12}{'var ratio':>11}")
    for j, name in ((0, "intercept"), (1, "rank slope")):
        obs = float(np.var(coefs[:, j]))
        noise = float(np.mean(varis[:, j]))
        between = max(obs - noise, 0.0)
        print(f"{name:<12}{coefs[:, j].mean():>9.3f}{np.sqrt(obs):>9.3f}"
              f"{np.sqrt(noise):>10.3f}{np.sqrt(between):>12.3f}"
              f"{between / noise:>11.2f}")

    # Heavy users are where the coefficient is best identified, so if the effect
    # is real at all it must be clearest there; a ratio that does NOT rise with
    # attempt count is the signature of noise rather than signal.
    print("\nby attempt count (rank slope only):")
    qs = np.quantile(ns, [0.0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (ns >= lo) & (ns <= hi)
        if m.sum() < 10:
            continue
        obs = float(np.var(coefs[m, 1]))
        noise = float(np.mean(varis[m, 1]))
        print(f"  {int(lo):>5}-{int(hi):<6} n={m.sum():>4}  obs sd "
              f"{np.sqrt(obs):.3f}  noise sd {np.sqrt(noise):.3f}  "
              f"var ratio {max(obs - noise, 0.0) / noise:.2f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
