#!/usr/bin/env python3
"""Fit K=8 eigen-only, then score every 3-axis SUBSET of that fit.

The motivation is nameability, not gain. A direct K=3 fit has three axes to
absorb everything the residual offers, so each one ends up carrying several
themes at once — correlated themes, but a tail a human cannot write a single
motif name for. Fitting wider separates them, and the question this answers is
what it COSTS to then ship only three of the eight: how much of the wide fit's
gain does a triple retain, and does a triple beat the direct K=3 fit it would
replace.

The subsets are scored with their DIRECTIONS FROZEN — `axes_ll_gain` refits only
the per-user coefficients, exactly as a client would against shipped axes. That
is the honest measurement of the actual proposal (ship these three vectors) and
it is deliberately not a K=3 refit warm-started at them: re-fitting would let the
three travel and re-absorb the themes the wide fit separated, which is the thing
being avoided, and the anchor measurement says starting a fit from already-fitted
directions loses to a plain eigen warm start anyway.

Read the numbers with one caveat in mind. Each of the eight was fitted to be
COMPLEMENTARY to the other seven, so a triple standing alone leaves the five
dropped axes' signal unclaimed rather than re-absorbing it. The triples are
therefore a lower bound on what three axes fitted for the purpose could do, and
the gap to the direct K=3 fit is the price of separation, not evidence that three
axes cannot reach further.

Every fold is fitted and scored on its own basis (fold 0 axis 3 and fold 1 axis 3
are unrelated), so subsets are indexed per fold and the reported figure for a
subset is the two folds' — pooled gain summed over attempts, per-user gain
averaged. A subset that wins in one fold and loses in the other is the case the
per-fold columns exist to show.

For naming, the axes rendered by fit_axes_at_k are an ALL-USER fit and are not
these directions. The cosine table printed at the end maps each fold axis onto
the nearest all-user axis so a triple chosen off that HTML can be located here.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_subset_gain --k 8
"""

import argparse
import itertools
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
    eigen_directions, load_axis_attempts, train_axes,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402


def score(rows_by_user, ev_by_fold, g_by_fold, rank, params, tid_index, lapse,
          idx):
    """Pooled and per-user gain of one column subset, both folds combined."""
    tot_g, tot_n, pu = 0.0, 0, []
    per_fold = []
    for f in (0, 1):
        gain, n, gu, _nu = axes_ll_gain(rows_by_user, ev_by_fold[f],
                                        g_by_fold[f][:, list(idx)], rank,
                                        params, tid_index, lapse)
        tot_g += gain
        tot_n += n
        pu.append(gu)
        per_fold.append((gain / max(n, 1) * 1000.0, gu))
    return tot_g / max(tot_n, 1) * 1000.0, float(np.mean(pu)), per_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="width of the wide fit")
    ap.add_argument("--pick", type=int, default=3, help="subset size to score")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-user-npz", default="/tmp/k8/axes.npz",
                    help="fit_axes_at_k output to match fold axes against")
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
    print(f"loaded in {time.time() - t0:.0f}s: {len(tids)} problems, "
          f"{len(users)} users; fitting K={args.k} per fold", flush=True)

    g_by_fold, W_by_fold = {}, {}
    for f in (0, 1):
        g_init = eigen_directions(rows_by_user, tu_by_fold[f], F, tid_index,
                                  rank, params, lapse, args.k)
        g, W, _rms, _sd = train_axes(rows_by_user, tu_by_fold[f], F, tid_index,
                                     args.k, g_init, device, seed=args.seed,
                                     epochs=args.epochs)
        g_by_fold[f] = g
        W_by_fold[f] = W
        print(f"  fold {f} fitted ({time.time() - t0:.0f}s)", flush=True)

    full = tuple(range(args.k))
    print(f"\n{'subset':<26}{'pooled':>9}{'per-user':>10}"
          f"{'f0 pool':>9}{'f1 pool':>9}{'f0 pu':>8}{'f1 pu':>8}", flush=True)

    def emit(idx, tag=None):
        p, u, pf = score(rows_by_user, ev_by_fold, g_by_fold, rank, params,
                         tid_index, lapse, idx)
        label = tag or ("{" + ",".join(str(i) for i in idx) + "}")
        print(f"{label:<26}{p:>9.3f}{u:>10.3f}{pf[0][0]:>9.3f}{pf[1][0]:>9.3f}"
              f"{pf[0][1]:>8.3f}{pf[1][1]:>8.3f}", flush=True)
        return (label, idx, p, u)

    results = [emit(full, f"ALL {args.k}")]
    for i in range(args.k):
        emit((i,), f"single {i}")
    for idx in itertools.combinations(range(args.k), args.pick):
        results.append(emit(idx))

    subs = [r for r in results if len(r[1]) == args.pick]
    print(f"\n===== {args.pick}-subsets ranked by pooled gain =====", flush=True)
    for label, _i, p, u in sorted(subs, key=lambda r: -r[2]):
        print(f"  {label:<16}{p:>9.3f}{u:>10.3f}")
    print(f"\n===== ranked by per-user gain =====", flush=True)
    for label, _i, p, u in sorted(subs, key=lambda r: -r[3]):
        print(f"  {label:<16}{p:>9.3f}{u:>10.3f}")

    # Fold bases are unrelated to each other and to the all-user fit the naming
    # HTML renders, so print the correspondence rather than leaving a chosen
    # triple unlocatable.
    def cos_table(A, B, an, bn):
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        C = np.abs(A @ B.T)
        print(f"\n{an} vs {bn} — |cos|, best match per row:")
        for i in range(C.shape[0]):
            j = int(np.argmax(C[i]))
            print(f"    {an} {i} -> {bn} {j}  |cos| {C[i, j]:.3f}   "
                  + " ".join(f"{v:.2f}" for v in C[i]))

    cos_table(W_by_fold[0], W_by_fold[1], "fold0", "fold1")
    if os.path.exists(args.all_user_npz):
        Wa = np.load(args.all_user_npz)["W"]
        if Wa.shape[0] == args.k:
            for f in (0, 1):
                cos_table(W_by_fold[f], Wa, f"fold{f}", "alluser")

    out = f"/tmp/axis_subset_k{args.k}.npz"
    np.savez_compressed(out, **{f"g{f}": g_by_fold[f] for f in (0, 1)},
                        **{f"W{f}": W_by_fold[f] for f in (0, 1)})
    print(f"\nwrote {out}\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
