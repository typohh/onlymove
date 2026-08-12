#!/usr/bin/env python3
"""Leak-free walk-forward: the global slope scale, on users the fit never saw.

`utils.rating.slope_calibration` runs the walk-forward under the client's
operating condition — per user, first-time attempts in time order, theta fitted
by MLE on the trailing window, the NEXT block scored — and returns s = 1.667,
agreeing with phase 2's in-sample 1.71-1.77. But its per-problem parameters come
from a fit trained on EVERY first-time attempt, including the rows it then
scores. Those predictions are partly memorised, memorised predictions are more
accurate, and more accuracy rewards more confidence — so the optimal steepness
under that protocol is biased UPWARD. The synthetic certificate (s = 0.993 on
outcomes generated from TRUE parameters) does not cover this: it validates the
estimator when the parameters are correct, not when they were fitted on the
scored rows.

This runs the same loop on users held out of training entirely
(`--holdout-user-frac`), so no scored row contributed to any parameter. If s
still comes back near 1.7, phase 2 is real and measures what it claims. If it
comes back near 1.0, the shipped multiplier is fitting its own leak.

The protocol is otherwise identical to slope_calibration's, deliberately: same
WINDOW / WARMUP / EVAL_BLOCK, same mode-'slope' application (the rating stays
the 80%-clear anchor and only steepness moves), so the two numbers differ in
exactly one respect — whether the scored rows helped fit the parameters.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.walk_forward_heldout \\
      --prod-dir /tmp/prod_userho --seed 0 --frac 0.15
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

EPS = 1e-9
WINDOW = 500
WARMUP = 120
EVAL_BLOCK = 50
S_GRID = np.concatenate([np.arange(0.6, 2.61, 0.05)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dir", default=None, help="a ratings/ directory")
    ap.add_argument("--dump", default=None, help="a MIRT --dump-params npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    args = ap.parse_args()

    from auxiliary.rating_dev.holdout_set import held_out_users
    from auxiliary.rating_dev.validate_ratings import clear_logit_vec
    from utils.rating.features import load_first_time_attempts

    hu_list, hu_hash = held_out_users(args.seed, args.frac)
    hu = set(hu_list)
    print(f"held-out USERS hash {hu_hash} | {len(hu_list)} accounts", flush=True)

    if args.prod_dir:
        from auxiliary.rating_dev.validate_ratings import load_ratings
        rate, slope, floor, lam, _th = load_ratings(args.prod_dir)
    else:
        z = np.load(args.dump, allow_pickle=True)
        lam = float(z["lapse"])
        rate = {str(t): float(v) for t, v in zip(z["tid"], z["rating"])}
        slope = {str(t): float(v) for t, v in zip(z["tid"], z["slope"])}
        floor = {str(t): float(v) for t, v in zip(z["tid"], z["floor"])}
    print(f"{len(rate)} problems, lapse {lam:.4f}", flush=True)

    df = load_first_time_attempts()
    df = df[df["size"].isin([9, 13])]
    df = df[df["user_id"].astype(str).isin(hu)]
    df = df.sort_values("created_at")
    by_user = defaultdict(list)
    for u, t, s_ in zip(df["user_id"].astype(str), df["problem"].astype(str),
                        df["success"]):
        if t in rate:
            by_user[u].append((t, int(s_)))
    scored_users = [u for u, a in by_user.items() if len(a) >= WARMUP + EVAL_BLOCK]
    print(f"{len(by_user)} held-out users with rated attempts, "
          f"{len(scored_users)} with >= {WARMUP + EVAL_BLOCK} for the walk-forward",
          flush=True)
    if not scored_users:
        print("nothing scorable — raise --frac or lower WARMUP")
        return

    # theta grid spans the rated range, as slope_calibration does
    allr = np.array([rate[t] for t in rate])
    grid = np.arange(allr.min() - 3.0, allr.max() + 3.0, 0.05)

    def ll_at(s):
        """Total logloss over every scored block at slope scale s. theta is
        refit on the trailing window AT THIS s, which is what makes the estimate
        a profile likelihood rather than a plug-in."""
        tot, n = 0.0, 0
        for u in scored_users:
            a_ = by_user[u]
            tids = [t for t, _ in a_]
            y = np.array([v for _, v in a_], float)
            A = np.array([slope[t] for t in tids]) * s
            C = np.clip(np.array([floor[t] for t in tids]), 0.0, 0.78)
            R = np.array([rate[t] for t in tids])
            span = np.maximum(1.0 - C - lam, 1e-6)
            kap = clear_logit_vec(C, lam)
            for t0 in range(WARMUP, len(a_), EVAL_BLOCK):
                lo, hi = max(0, t0 - WINDOW), min(t0 + EVAL_BLOCK, len(a_))
                zt = A[lo:t0, None] * (grid[None, :] - R[lo:t0, None]) + kap[lo:t0, None]
                P = np.clip(C[lo:t0, None] + span[lo:t0, None] / (1 + np.exp(-zt)),
                            1e-9, 1 - 1e-9)
                ll = (y[lo:t0, None] * np.log(P)
                      + (1 - y[lo:t0, None]) * np.log(1 - P)).sum(axis=0)
                th = grid[int(np.argmax(ll))]
                ze = A[t0:hi] * (th - R[t0:hi]) + kap[t0:hi]
                Pe = np.clip(C[t0:hi] + span[t0:hi] / (1 + np.exp(-ze)),
                             1e-9, 1 - 1e-9)
                tot += -(y[t0:hi] * np.log(Pe)
                         + (1 - y[t0:hi]) * np.log(1 - Pe)).sum()
                n += hi - t0
        return tot / max(n, 1), n

    rows = []
    for s in S_GRID:
        ll, n = ll_at(float(s))
        rows.append((float(s), ll, n))
    best = min(rows, key=lambda r: r[1])
    at1 = [r for r in rows if abs(r[0] - 1.0) < 1e-9]
    print(f"\nWALK-FORWARD on {rows[0][2]} scored attempts, held-out users only")
    for s, ll, _n in rows:
        mark = "  <-- best" if s == best[0] else ""
        if abs(s - round(s, 1)) < 1e-9:
            print(f"    s = {s:.2f}   logloss {ll:.5f}{mark}", flush=True)
    print(f"\n  BEST s = {best[0]:.2f}   logloss {best[1]:.5f}")
    if at1:
        print(f"  s = 1.00 gives {at1[0][1]:.5f}  -> scaling is worth "
              f"{(at1[0][1] - best[1]) * 1000:+.3f} mnat/attempt")
    print(f"  (production ships s = 1.694; its leaky walk-forward returned 1.667)")


if __name__ == "__main__":
    main()
