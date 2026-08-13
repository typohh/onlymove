#!/usr/bin/env python3
"""Has an axis started doing the rating's job?

In a compensatory MIRT nothing STRUCTURALLY stops an axis from becoming a second
ability dimension. The logit is

    a_p*(theta_u - b_p) + sum_k g_k(x_p)*v_u[k]

and an axis with g_k ~ a_p and v_k ~ (theta - mean theta) reproduces the main
term while satisfying every penalty in the objective: V_MEAN_PENALTY only asks
that mean_u(v_k) = 0, which a CENTRED copy of ability satisfies exactly, and
ORTHO_WEIGHT only asks that the axis columns be mutually orthonormal, which says
nothing about the rating. What actually holds the split in place is an asymmetry
rather than a constraint — theta is free while v carries V_RIDGE — so ability
goes to theta because it is cheaper, not because it is required to.

WHICH SIDE TO WATCH MATTERS, and the obvious choice is wrong. Do NOT test whether
g_k correlates with difficulty: a loading SHOULD reference difficulty, because
what a problem demands of one skill is only meaningful relative to how hard it is
overall (a 5d problem needing 1d reading is easy for its rating along that axis,
a 5k problem needing 1d reading is hard for its rating). Forbidding that would
forbid the very structure the axes exist to express.

Takeover is a USER-side phenomenon. An axis usurps the rating when its user
loadings duplicate ability, v_k ~ theta — that is a second ability dimension. So
the statistic is corr(v_k, theta) per axis, and the problem side is reported only
as description.

Reads a --dump-params npz.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_takeover_check --dump x.npz
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()
    z = np.load(args.dump, allow_pickle=True)

    v = np.asarray(z["v"], float)                 # (n_users, K)
    users = [str(u) for u in z["user"]]
    theta = np.asarray(z["theta"], float)         # (n_profiles,)
    profs = [str(p) for p in z["profile"]]
    g = np.asarray(z["loading"], float)           # (n_problems, K)
    b = np.asarray(z["difficulty"], float)
    a = np.asarray(z["slope"], float)
    rating = np.asarray(z["rating"], float)
    K = v.shape[1]

    # A profile id is "{user8}_{tag}", so the user is the first 8 characters —
    # never a split on a separator, since the id itself may contain one.
    by_user = {}
    for p, t in zip(profs, theta):
        by_user.setdefault(p[:8], []).append(t)
    th_u = np.array([np.mean(by_user.get(str(u)[:8], [np.nan])) for u in users])
    ok = np.isfinite(th_u) & np.isfinite(v).all(axis=1)
    print(f"{len(users)} users, {ok.sum()} with an ability, K={K}\n")

    print("USER SIDE — the takeover test. |corr| near 1 means the axis IS ability.")
    worst = 0.0
    for k in range(K):
        c = float(np.corrcoef(v[ok, k], th_u[ok])[0, 1])
        worst = max(worst, abs(c))
        sd = float(v[ok, k].std())
        print(f"  axis {k}:  corr(v, theta) {c:+.3f}   sd(v) {sd:.3f}   "
              f"mean(v) {float(v[ok, k].mean()):+.4f}")
    print(f"  worst |corr(v, theta)| = {worst:.3f}")

    print("\nPROBLEM SIDE — description, NOT a failure test. A loading is EXPECTED")
    print("to reference difficulty; that is the contrast the axes exist to carry.")
    fin = np.isfinite(g).all(axis=1) & np.isfinite(b) & np.isfinite(a)
    for k in range(K):
        cb = float(np.corrcoef(g[fin, k], b[fin])[0, 1])
        ca = float(np.corrcoef(g[fin, k], a[fin])[0, 1])
        cr = float(np.corrcoef(g[fin, k], rating[fin])[0, 1])
        print(f"  axis {k}:  corr(g, difficulty) {cb:+.3f}   "
              f"corr(g, slope) {ca:+.3f}   corr(g, rating) {cr:+.3f}")

    print("\nAXIS-AXIS — what ORTHO_WEIGHT actually constrains.")
    C = np.corrcoef(g[fin].T)
    off = C[~np.eye(K, dtype=bool)]
    print(f"  max |corr| between axis columns: {np.abs(off).max():.3f}")


if __name__ == "__main__":
    main()
