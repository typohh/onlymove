#!/usr/bin/env python3
"""Start wide, drop the weakest axis one at a time, and report the whole path.

The production search grows K from a floor and stops at the first breach, taking
the leading K eigenvectors in order. That shape has two costs this replaces.

It cannot consider a NAMED motif that ranks below the top K by captured
variation — measured, the named vocabulary sits around eigenvector rank 10-16,
so the incremental search never sees any of it. And growing forward judges each
count on a set built from the top down, whereas what one wants to know is which
SUBSET of a wide candidate pool survives, which is a different question whenever
candidates overlap.

So the candidate pool here is every NAMED banked motif plus the leading
eigenvectors, deduplicated, and the fit starts with all of them. At each level
the weakest axis is dropped and the rest refit, down to a floor, printing gain
under both weightings at every K along with what was dropped and why. Nothing is
auto-selected: the path is the output, because the choice between "highest gain"
and "highest gain that still carries named motifs" is a judgement about the
product, not a number.

WEAKEST is measured as how much an axis actually moves a held-out user's score:
the median absolute per-user coefficient times the spread of that axis's loadings
over the scored problems. An axis whose coefficient is shrunk to nothing by the
prior, or whose loadings barely vary, cannot influence anyone whatever its
eigenvalue was. This is one walk-forward per level rather than one per candidate,
which is what makes a 20-deep path affordable; the cost is that it ranks by
influence rather than by held-out gain directly, so a dropped axis is the least
influential rather than provably the least useful.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_backward_select \\
      --eigen 10 --floor 2
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
    MIN_USER_ATTEMPTS, _axis_prior, _stratified_folds, _walk_forward,
    axes_ll_gain, build_axis_features, load_axis_attempts, train_axes,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402
from auxiliary.rating_dev.eval_axis_epoch_curve import (  # noqa: E402
    fold_moment, named_anchor_directions,
)

DEDUP_COS = 0.90


def candidate_pool(F, tid_index, Sigma, n_eigen):
    """Named motifs first, then the leading eigenvectors, deduplicated.

    Named motifs go first so that when a named direction and an eigenvector are
    near-duplicates the NAMED one holds the slot — the whole point being to keep
    a motif that already has a human name rather than mint an equivalent one
    beside it."""
    out = []
    for q, mid, name, e in named_anchor_directions(F, tid_index, Sigma):
        out.append([e, f"{name}", mid, float(q)])
    ev, V = np.linalg.eigh(Sigma)
    order = np.argsort(ev)[::-1][:n_eigen]
    for j, idx in enumerate(order):
        e = V[:, idx]
        out.append([e / (np.linalg.norm(e) + 1e-12), f"eig{j}", None,
                    float(ev[idx])])
    kept = []
    for e, label, mid, q in out:
        if any(abs(float(e @ k[0])) > DEDUP_COS for k in kept):
            continue
        kept.append([e, label, mid, q])
    return kept


def axis_influence(rows_by_user, eval_users, g, rank, params, tid_index, lapse):
    """Per-axis influence on held-out users' scores: median |coefficient| times
    the spread of that axis's loadings over the problems actually scored."""
    _o, _s, _y, _u, c0 = _walk_forward(rows_by_user, eval_users, g, rank,
                                       params, tid_index, lapse)
    _o2, _s2, _y2, _u2, coefs = _walk_forward(rows_by_user, eval_users, g, rank,
                                              params, tid_index, lapse,
                                              penalty=_axis_prior(c0))
    seen = sorted({tid_index[t] for u in eval_users
                   for t, _s in rows_by_user[u] if t in tid_index})
    spread = g[np.array(seen)].std(axis=0)
    return np.median(np.abs(coefs), axis=0) * spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eigen", type=int, default=10,
                    help="leading eigenvectors to add to the named motifs")
    ap.add_argument("--floor", type=int, default=2, help="stop at this K")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    feats, rank, params, _ad = build_axis_features()
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9))
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

    pool = {}
    for f, tu in tu_by_fold.items():
        Sig = fold_moment(rows_by_user, tu, F, tid_index, rank, params, lapse)
        pool[f] = candidate_pool(F, tid_index, Sig, args.eigen)
    print(f"loaded in {time.time() - t0:.0f}s; candidate pool per fold: "
          f"{len(pool[0])} / {len(pool[1])}", flush=True)
    print("  fold 0 pool: " + ", ".join(c[1] for c in pool[0]), flush=True)

    live = {f: list(range(len(pool[f]))) for f in (0, 1)}
    Fk = F.astype(np.float32)
    print(f"\n{'K':>3} {'pooled':>8} {'per-user':>9}  dropped next")
    while len(live[0]) >= args.floor and len(live[1]) >= args.floor:
        k = min(len(live[0]), len(live[1]))
        tot_g, tot_n, pu, infl = 0.0, 0, [], {}
        for f in (0, 1):
            g_init = np.stack([pool[f][i][0] for i in live[f]]).astype(np.float32)
            g, _W, _rms, _sd = train_axes(rows_by_user, tu_by_fold[f], Fk,
                                          tid_index, g_init.shape[0], g_init,
                                          device, seed=args.seed,
                                          epochs=args.epochs)
            gain, n, gu, _nu = axes_ll_gain(rows_by_user, ev_by_fold[f], g,
                                            rank, params, tid_index, lapse)
            tot_g += gain
            tot_n += n
            pu.append(gu)
            infl[f] = axis_influence(rows_by_user, ev_by_fold[f], g, rank,
                                     params, tid_index, lapse)
        drop_label = []
        for f in (0, 1):
            j = int(np.argmin(infl[f]))
            drop_label.append(pool[f][live[f][j]][1])
            live[f].pop(j)
        print(f"{k:>3} {tot_g / tot_n * 1000:>8.3f} {np.mean(pu):>9.3f}  "
              f"{drop_label[0]} / {drop_label[1]}", flush=True)

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
