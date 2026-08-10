#!/usr/bin/env python3
"""A/B the axis warm start: hand-crafted seeds vs cold random init.

Production (utils.rating.residual_axes) warm-starts the FIRST THREE axes at
ridge-regressed hand-crafted scalar directions (wr gap, score gap, phase ratio)
and leaves every later axis at PyTorch's default init. That asymmetry is
confounded with the K search, which currently stops at K=3 — exactly the number
of seeded axes — so "there is no 4th motif" and "a cold-started 4th axis cannot
escape the v=0 saddle in EPOCHS epochs" are indistinguishable from the shipped
artifacts alone.

This script runs the paired comparison the production path cannot: identical
features, identical user folds, identical eval, ONLY g_init differs. Every arm
of every candidate K is scored on the SAME fold split and the same torch seeds,
so arm differences are not resample noise. Multiple torch seeds are run because
the shipped selection turned on a 3.7% gain gap measured at a single seed, and
nothing establishes that gap is larger than fit noise.

Reported per (K, arm):
  - pooled held-out walk-forward LL gain in mnat/attempt, the production
    selection criterion (utils.rating.residual_axes.axes_ll_gain), plus its
    per-seed spread
  - correlation of each fitted axis's loadings against the three hand-crafted
    scalars — does a COLD fit rediscover them unaided?
  - principal angles between the warm and cold loading SPANS at matching
    (K, seed, fold) — whether the two arms find the same subspace even when
    they disagree about the basis inside it

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_axis_seeding \
      --ks 3 --arms warm,cold --seeds 0,1
"""

import argparse
import hashlib
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import (  # noqa: E402
    MIN_USER_ATTEMPTS, axes_ll_gain, build_axis_features, load_axis_attempts,
    seed_directions, train_axes,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

SCALAR_NAMES = ("wr_gap", "score_gap", "phase")


def _corr_cols(A, B):
    """Column-wise correlation matrix between two (n, k) loading blocks."""
    Az = (A - A.mean(0)) / (A.std(0) + 1e-9)
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return Az.T @ Bz / A.shape[0]


def _principal_angles(A, B):
    """Cosines of the principal angles between the column spans of A and B."""
    Qa, _ = np.linalg.qr((A - A.mean(0)) / (A.std(0) + 1e-9))
    Qb, _ = np.linalg.qr((B - B.mean(0)) / (B.std(0) + 1e-9))
    return np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="3", help="comma-separated axis counts")
    ap.add_argument("--arms", default="warm,cold")
    ap.add_argument("--seeds", default="0", help="comma-separated torch seeds")
    ap.add_argument("--epochs", default="40",
                    help="comma-separated epoch counts; a cold fit starts at a "
                         "saddle, so its deficit may be convergence rather than "
                         "solution quality — training is seconds, so this is "
                         "the cheap way to tell the two apart")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    arms = args.arms.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    epochs_list = [int(x) for x in args.epochs.split(",")]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    t0 = time.time()
    feats, scalars, rank, params, _act_dim = build_axis_features()
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float32)
    S_mat = np.stack([scalars[t] for t in tids])
    lapse = load_client_lapse(0.0328)

    rows_by_user = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in tid_index:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    fold_of = {u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 for u in users}
    n_att = sum(len(rows_by_user[u]) for u in users)
    print(f"features+attempts loaded in {time.time() - t0:.0f}s: "
          f"{len(tids)} problems, {len(users)} users, {n_att} attempts",
          flush=True)

    results = {}          # (k, arm, ep, seed) -> pooled mnat/attempt
    loadings = {}         # (k, arm, ep, seed, fold) -> loadings on the TRAIN fit
    for k in ks:
        g_seed = seed_directions(F, S_mat, min(3, k))
        for ep in epochs_list:
            for seed in seeds:
                for arm in arms:
                    g_init = g_seed if arm == "warm" else None
                    tg, tn = 0.0, 0
                    for fold in (0, 1):
                        tu = [u for u in users if fold_of[u] != fold]
                        eu = [u for u in users if fold_of[u] == fold]
                        ts = time.time()
                        g, _W, _rms, _sd = train_axes(rows_by_user, tu, F,
                                                      tid_index, k, g_init,
                                                      device, seed=seed,
                                                      epochs=ep)
                        t_tr = time.time() - ts
                        ts = time.time()
                        gain, n = axes_ll_gain(rows_by_user, eu, g, rank,
                                               params, tid_index, lapse)
                        loadings[(k, arm, ep, seed, fold)] = g
                        tg += gain
                        tn += n
                        print(f"  K={k} {arm:4s} ep={ep} seed={seed} "
                              f"fold={fold}: {gain / n * 1000:.3f} "
                              f"mnat/attempt over {n} (train {t_tr:.0f}s, "
                              f"eval {time.time() - ts:.0f}s)", flush=True)
                    results[(k, arm, ep, seed)] = tg / tn * 1000.0
                    print(f"K={k} {arm} ep={ep} seed={seed}: POOLED "
                          f"{results[(k, arm, ep, seed)]:.3f} mnat/attempt",
                          flush=True)

    print("\n===== pooled held-out gain (mnat/attempt) =====", flush=True)
    for k in ks:
        for ep in epochs_list:
            for arm in arms:
                v = [results[(k, arm, ep, s)] for s in seeds
                     if (k, arm, ep, s) in results]
                if not v:
                    continue
                spread = (f" (sd {np.std(v):.3f} over {len(v)} seeds)"
                          if len(v) > 1 else "")
                print(f"  K={k:2d} {arm:4s} ep={ep:3d}: {np.mean(v):.3f}{spread}"
                      "  " + " ".join(f"s{s}={results[(k, arm, ep, s)]:.3f}"
                                      for s in seeds
                                      if (k, arm, ep, s) in results))

    print("\n===== corr(fitted loadings, hand-crafted scalars) =====", flush=True)
    for key in sorted(loadings):
        k, arm, ep, seed, fold = key
        C = _corr_cols(loadings[key], S_mat)
        print(f"  K={k} {arm} ep={ep} seed={seed} fold={fold}:")
        for a in range(k):
            print("    axis %d: " % a
                  + "  ".join(f"{SCALAR_NAMES[j]} {C[a, j]:+.3f}" for j in range(3)))

    if "warm" in arms and "cold" in arms:
        print("\n===== warm vs cold: same subspace? =====", flush=True)
        for k in ks:
            for ep in epochs_list:
                for seed in seeds:
                    for fold in (0, 1):
                        kw = (k, "warm", ep, seed, fold)
                        kc = (k, "cold", ep, seed, fold)
                        if kw not in loadings or kc not in loadings:
                            continue
                        ang = _principal_angles(loadings[kw], loadings[kc])
                        C = np.abs(_corr_cols(loadings[kw], loadings[kc]))
                        print(f"  K={k} ep={ep} seed={seed} fold={fold}: "
                              f"principal-angle cosines {np.round(ang, 3)}  |  "
                              f"best |corr| per warm axis {np.round(C.max(1), 3)}")

    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
