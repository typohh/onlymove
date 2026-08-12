#!/usr/bin/env python3
"""Held-out-USER axis gain: joint MIRT against build_axes, aggregated identically.

The joint model's axis loadings are fitted on every user unless
--axis-holdout-fold pins a fold's axis positions at zero. Without that flag its
"held-out user" figure is not a holdout at all and cannot be set beside
build_axes, which does genuine two-fold user CV. This script consumes the dumps
from two such fold runs and reproduces build_axes's aggregation exactly:

    pooled   = sum(gain over folds) / sum(n over folds) * 1000
    per-user = sum(rate_f * n_users_f) / sum(n_users_f)

Production's side needs no re-scoring: axes.npz records `gains` and
`gains_per_user` from build_axes's own two-fold CV, computed by this same
axes_ll_gain against this same shipped-4PL baseline. Re-deriving them here would
only add a way to get them wrong.

SCALE. train_axes returns loadings multiplied by v_sd (the per-user position
spread); the joint model's problem_params returns them RMS-normalised only. The
per-user IRLS prior is estimated from the coefficient spread and so adapts to
either, making the gain nearly scale-invariant — but "nearly" is not a basis for
comparing numbers 0.1 apart, so both conventions are reported and the two-stage
one is the one to quote.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.compare_axis_gain \\
      --fold0 /tmp/joint_params_axfold0.npz --fold1 /tmp/joint_params_axfold1.npz
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold0", required=True)
    ap.add_argument("--fold1", required=True)
    ap.add_argument("--axes-npz", default="axes/axes.npz")
    args = ap.parse_args()

    from utils.rating.residual_axes import (
        MIN_USER_ATTEMPTS, _stratified_folds, axes_ll_gain, build_axis_features,
        load_axis_attempts)
    from utils.rating.slope_calibration import load_client_lapse

    feats, rank, params, _ad = build_axis_features()
    tids = sorted(feats)
    ti = {t: i for i, t in enumerate(tids)}
    rbu = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in ti:
            rbu[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rbu.items() if len(a) >= MIN_USER_ATTEMPTS)
    fold_of = _stratified_folds(users, rbu)
    lapse = load_client_lapse(0.0328)
    print(f"{len(users)} axis users | folds "
          f"{sum(1 for u in users if fold_of[u] == 0)}/"
          f"{sum(1 for u in users if fold_of[u] == 1)}")

    def score(path, fold, scale_by_v_sd):
        z = np.load(path, allow_pickle=True)
        load = {str(t): v for t, v in zip(z["tid"], z["loading"])}
        v = z["v"]
        if scale_by_v_sd:
            # v_sd over PARTICIPATING users only — the held-out fold sits at
            # exactly zero and would drag the spread down, which is the very
            # quantity being used as the scale.
            live = v[np.abs(v).sum(axis=1) > 0]
            v_sd = live.std(axis=0) if len(live) > 1 else np.ones(v.shape[1])
        else:
            v_sd = np.ones(v.shape[1])
        k = v.shape[1]
        g = np.zeros((len(tids), k), dtype=np.float32)
        hit = 0
        for t, i in ti.items():
            if t in load:
                g[i] = np.asarray(load[t], dtype=np.float32) * v_sd
                hit += 1
        ev = [u for u in users if fold_of[u] == fold]
        gain, n, gu, nu = axes_ll_gain(rbu, ev, g, rank, params, ti, lapse)
        return gain, n, gu, nu, hit

    for scale in (True, False):
        name = "v_sd-scaled (train_axes convention)" if scale else "RMS-only"
        tot_g = tot_n = 0.0
        pu = []
        per_fold = []
        for fold, path in ((0, args.fold0), (1, args.fold1)):
            gain, n, gu, nu, hit = score(path, fold, scale)
            tot_g += gain
            tot_n += n
            pu.append((gu, nu))
            per_fold.append((fold, gain / n * 1000.0, gu, nu, n, hit))
        pooled = tot_g / tot_n * 1000.0
        per_user = sum(r * m for r, m in pu) / sum(m for _r, m in pu)
        print(f"\nJOINT MIRT, held-out-user axis gain — {name}")
        for fold, rate, gu, nu, n, hit in per_fold:
            print(f"    fold {fold}: pooled {rate:7.3f}  per-user {gu:7.3f}  "
                  f"({nu} users, {n} attempts, {hit}/{len(tids)} priced)")
        print(f"    BOTH FOLDS: pooled {pooled:7.3f}  per-user {per_user:7.3f}")

    z = np.load(args.axes_npz, allow_pickle=True)
    ks = list(z["gains_k"])
    sel = int(z["k"])
    i = ks.index(sel) if sel in ks else 0
    gp, gu = float(z["gains"][i]), float(z["gains_per_user"][i])
    print(f"\nPRODUCTION build_axes, held-out-user 2-fold CV at K={sel}")
    print(f"    BOTH FOLDS: pooled {gp:7.3f}  per-user {gu:7.3f}")
    print(f"\n(compare against the v_sd-scaled joint figures — same convention)")


if __name__ == "__main__":
    main()
