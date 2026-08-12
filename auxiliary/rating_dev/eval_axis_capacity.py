#!/usr/bin/env python3
"""Where between a linear axis map and a full trunk does capacity stop paying?

The ladder that rejected a trunk on the axis path compared two points only:
the shipped linear map at 648 parameters (K=3) and a [128, 64] trunk at ~36k, a
55x step. That is enough to show the far end is worse and not enough to show the
whole range is — with the input dropout in place the trunk arm came within 9% of
the linear map, so an intermediate width could plausibly match or beat it, and a
two-point comparison would never see it.

This fills the interval: single-hidden-layer trunks at 16, 32 and 64 units,
ADDED to the eigen-warm-started linear map rather than replacing it, so capacity
stays separated from initialization (a trunk cannot be handed a direction, and
substituting for the linear map would change two things at once). Every arm runs
at the shipped dropout, since the question is about capacity given regularization
rather than instead of it.

    axis_hidden      params      what it is
    (none)              648      the shipped linear map
    [16]              ~3,500
    [32]              ~7,000
    [64]             ~14,000
    [128, 64]        ~36,000     the rejected point

Runs on `auxiliary.rating_dev.train_axes_staged`, the axis trainer's experiment
surface — production carries no such option and is never edited to run this.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_axis_capacity --seeds 0 1
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import (  # noqa: E402
    MIN_USER_ATTEMPTS, _stratified_folds, axes_ll_gain, build_axis_features,
    eigen_directions, load_axis_attempts,
)
from auxiliary.rating_dev.train_axes_staged import train_axes_staged  # noqa: E402
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

WIDTHS = [None, (16,), (32,), (64,), (128, 64)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    feats, rank, params, _ad = build_axis_features()
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float32)
    lapse = load_client_lapse(0.0328)

    rows_by_user = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in tid_index:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    fold_of = _stratified_folds(users, rows_by_user)
    tu_by_fold = {f: [u for u in users if fold_of[u] != f] for f in (0, 1)}
    ev_by_fold = {f: [u for u in users if fold_of[u] == f] for f in (0, 1)}
    print(f"loaded in {time.time() - t0:.0f}s: {len(users)} users, K={args.k}, "
          f"seeds {args.seeds}", flush=True)

    # shared per fold: every width starts from byte-identical linear directions
    g_init = {f: eigen_directions(rows_by_user, tu_by_fold[f], F, tid_index,
                                  rank, params, lapse, args.k) for f in (0, 1)}
    print(f"  warm starts computed ({time.time() - t0:.0f}s)", flush=True)

    def nparams(h):
        n, d = 0, F.shape[1]
        for w in (h or ()):
            n += d * w + w
            d = w
        return 648 if not h else n + d * args.k + args.k + 648

    res = defaultdict(dict)
    print(f"\n{'axis_hidden':<14}{'params':>9}{'seed':>5}{'pooled':>9}"
          f"{'per-user':>10}{'f0':>8}{'f1':>8}", flush=True)
    for seed in args.seeds:
        for h in WIDTHS:
            tot_g, tot_n, pu, pf = 0.0, 0, [], []
            for f in (0, 1):
                g, _W, _rms, _sd = train_axes_staged(
                    rows_by_user, tu_by_fold[f], F, tid_index, args.k,
                    g_init[f], device, seed=seed, axis_hidden=h)
                gain, n, gu, _nu = axes_ll_gain(rows_by_user, ev_by_fold[f], g,
                                                rank, params, tid_index, lapse)
                tot_g += gain
                tot_n += n
                pu.append(gu)
                pf.append(gain / max(n, 1) * 1000.0)
            p, u = tot_g / max(tot_n, 1) * 1000.0, float(np.mean(pu))
            res[h][seed] = (p, u)
            print(f"{str(h or 'linear'):<14}{nparams(h):>9,}{seed:>5}{p:>9.3f}"
                  f"{u:>10.3f}{pf[0]:>8.3f}{pf[1]:>8.3f}", flush=True)

    print(f"\n===== paired against the linear map, seeds {args.seeds} =====")
    print(f"{'axis_hidden':<14}{'params':>9}{'pooled':>9}{'per-user':>10}"
          f"{'d pooled':>10}{'d per-user':>12}")
    for h in WIDTHS:
        ps = np.array([res[h][s][0] for s in args.seeds])
        us = np.array([res[h][s][1] for s in args.seeds])
        bp = np.array([res[None][s][0] for s in args.seeds])
        bu = np.array([res[None][s][1] for s in args.seeds])
        print(f"{str(h or 'linear'):<14}{nparams(h):>9,}{ps.mean():>9.3f}"
              f"{us.mean():>10.3f}{(ps - bp).mean():>+10.3f}"
              f"{(us - bu).mean():>+12.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
