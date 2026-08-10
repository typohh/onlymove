#!/usr/bin/env python3
"""Fit the axis model at an arbitrary K on ALL users and write inspectable
artifacts, without touching axes/.

The production step selects K itself and ships one answer. For NAMING, the
question is different: at K=3 a single axis may be carrying several things at
once, and whether the parts separate into something recognisable is only visible
by fitting wider and looking. This produces the same two artifacts the inspection
page reads — a per_problem-style CSV and an axes.npz carrying W and the
standardizations the driver tables decompose — into a directory of your choosing,
so the shipped fit is never disturbed.

Fits on all users, like the production final fit, because what is being judged is
the axis set a human would name and ship, not a held-out estimate.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.fit_axes_at_k --k 8 --out /tmp/k8
  ./.venv/bin/python3 -m auxiliary.rating_dev.inspect_residual_axes \\
      --per-problem /tmp/k8/per_problem.csv --axes-npz /tmp/k8/axes.npz \\
      --out /tmp/k8.html
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import (  # noqa: E402
    EPOCHS, MIN_USER_ATTEMPTS, build_axis_features, corpus_tid_size,
    eigen_directions, load_axis_attempts, train_axes,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    feats, rank, params, act_dim = build_axis_features()
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    feat_mu, feat_sd = F.mean(axis=0), F.std(axis=0) + 1e-9
    F = ((F - feat_mu) / feat_sd).astype(np.float32)
    lapse = load_client_lapse(0.0328)

    rows_by_user = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in tid_index:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    print(f"loaded in {time.time() - t0:.0f}s: {len(tids)} problems, "
          f"{len(users)} users; fitting K={args.k}", flush=True)

    g_init = eigen_directions(rows_by_user, users, F, tid_index, rank, params,
                              lapse, args.k)
    g, W, rms, v_sd = train_axes(rows_by_user, users, F, tid_index, args.k,
                                 g_init, device, seed=args.seed,
                                 epochs=args.epochs)
    axis_mu, axis_sd = g.mean(axis=0), g.std(axis=0) + 1e-9
    loadings = ((g - axis_mu) / axis_sd).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    tid_size = corpus_tid_size()
    pp = os.path.join(args.out, "per_problem.csv")
    with open(pp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tsumego_id", "size"]
                   + [f"embed_{i}" for i in range(args.k)])
        for i, t in enumerate(tids):
            w.writerow([t, tid_size[t]] + [f"{v:.5f}" for v in loadings[i]])
    np.savez_compressed(
        os.path.join(args.out, "axes.npz"),
        k=np.array(args.k), act_dim=np.array(act_dim),
        W=W.astype(np.float32), rms=rms.astype(np.float32),
        v_sd=v_sd.astype(np.float32),
        feat_mu=feat_mu.astype(np.float32), feat_sd=feat_sd.astype(np.float32),
        axis_mu=axis_mu.astype(np.float32), axis_sd=axis_sd.astype(np.float32))
    print(f"wrote {pp} and axes.npz ({time.time() - t0:.0f}s)")
    print(f"\nrender with:\n  ./.venv/bin/python3 -m "
          f"auxiliary.rating_dev.inspect_residual_axes --per-problem {pp} "
          f"--axes-npz {os.path.join(args.out, 'axes.npz')} "
          f"--out {args.out.rstrip('/')}.html")


if __name__ == "__main__":
    main()
