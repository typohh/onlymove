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
eigenvectors, deduplicated, and the fit starts with all of them. At each level the weakest axis is dropped and the fit CONTINUES from the
previous level's weights with that row removed — level n+1 is genuinely level n
minus one axis, not a fresh fit on a smaller pool. That distinction is the whole
validity of the path: independent fits of this model differ by more than any
single axis is worth (0.65 to 0.93 on identical data, from the draw alone), so a
restart-per-level path reports refit variance and calls it elimination. Nothing is
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
    BATCH, MIN_USER_ATTEMPTS, ORTHO_WEIGHT, _axis_prior, _stratified_folds,
    _walk_forward, axes_ll_gain, build_axis_features, load_axis_attempts,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402
from auxiliary.rating_dev.eval_axis_epoch_curve import (  # noqa: E402
    fold_moment, named_anchor_directions,
)

DEDUP_COS = 0.90


def train_until_settled(rows_by_user, train_users, F, tid_index, g_init, device,
                        seed=0, max_epochs=400, min_epochs=20, tol=2e-4,
                        window=5):
    """residual_axes.train_axes with a CONVERGENCE STOP.

    Kept in step with production except for when it stops. After the first level
    the axis block starts from the previous level's answer minus one row, so it
    is already near a solution; paying a full from-scratch schedule at every
    level throws away the point of carrying the weights.

    Convergence is measured on the DIRECTIONS rather than the loss: mean angular
    movement of the axis rows per epoch, 1 - |cos(W_t, W_t-1)|, averaged over a
    short window. The loss is a minibatch estimate whose per-epoch change is
    mostly sampling noise, and the rows' SCALE is unidentified anyway (the head
    is RMS-normalized), so the angle is the only part that means anything.

    Returns (loadings, W, rms, v_sd, epochs_used)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    uids = sorted(train_users)
    uindex = {u: i for i, u in enumerate(uids)}
    rows = [(uindex[u], tid_index[t], s)
            for u in uids for t, s in rows_by_user[u]]
    U = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    P = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    Y = torch.tensor([float(r[2]) for r in rows], device=device)
    counts = torch.bincount(U, minlength=len(uids)).clamp(min=1).float()
    X = torch.tensor(F, device=device)
    RANK = X[:, -1].clone()

    n_u, n_feat = len(uids), F.shape[1]
    k_axes = g_init.shape[0]
    trunk = nn.Sequential(
        nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    ).to(device)
    head_r = nn.Linear(32, 1).to(device)
    g_lin = nn.Linear(n_feat, k_axes, bias=False).to(device)
    with torch.no_grad():
        g_lin.weight.copy_(torch.tensor(g_init, dtype=torch.float32,
                                        device=device))
    alpha = torch.zeros(n_u, device=device, requires_grad=True)
    beta = torch.zeros(n_u, device=device, requires_grad=True)
    v = torch.zeros(n_u, k_axes, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": list(trunk.parameters()) + list(head_r.parameters()),
         "lr": 1e-3, "weight_decay": 1e-4},
        {"params": list(g_lin.parameters()), "lr": 1e-3},
        {"params": [alpha, beta], "lr": 1e-2},
        {"params": [v], "lr": 1e-2},
    ])
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    n = len(rows)
    prev = g_lin.weight.detach().clone()
    moves, used = [], max_epochs
    for ep in range(1, max_epochs + 1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            u, p, y = U[idx], P[idx], Y[idx]
            h = trunk(X[p])
            rh = head_r(h)
            gh = g_lin(X[p])
            gh = gh / (gh.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
            logit = (alpha[u] + beta[u] * RANK[p] + rh.squeeze(-1)
                     + (gh * v[u]).sum(-1))
            loss = bce(logit, y) \
                + 0.5 * ((v[u] ** 2).sum(dim=1) / counts[u]).sum() / len(idx) \
                + 10.0 * (v.mean(dim=0) ** 2).sum()
            M = torch.cat([gh, rh], dim=1)
            Mc = M - M.mean(dim=0, keepdim=True)
            Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
            Cm = Mn.T @ Mn
            loss = loss + ORTHO_WEIGHT * ((Cm - torch.eye(
                Cm.shape[0], device=Cm.device)) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            cur = g_lin.weight.detach()
            cs = torch.nn.functional.cosine_similarity(cur, prev, dim=1).abs()
            moves.append(float((1.0 - cs).mean()))
            prev = cur.clone()
        if ep >= min_epochs and len(moves) >= window \
                and float(np.mean(moves[-window:])) < tol:
            used = ep
            break
    with torch.no_grad():
        gh = g_lin(X)
        rms = gh.pow(2).mean(dim=0).sqrt() + 1e-6
        v_sd = v.std(dim=0).cpu().numpy()
        g_all = (gh / rms).cpu().numpy() * v_sd[None, :]
        W = g_lin.weight.detach().cpu().numpy()
    return g_all, W, rms.cpu().numpy(), v_sd, used


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
    # The fitted weights are CARRIED FORWARD: each level drops a row from the
    # previous level's answer and continues from there, rather than rebuilding
    # the warm start from the raw candidate directions. Restarting instead makes
    # the path a sequence of INDEPENDENT fits, and independent fits of this model
    # differ by more than any single axis is worth (measured: 0.65 to 0.93 on the
    # same data from the draw alone), so consecutive levels would differ by a
    # refit's variance and the elimination would be reading noise. Carrying the
    # weights makes level n+1 genuinely level n minus one axis.
    carried = {f: None for f in (0, 1)}
    Fk = F.astype(np.float32)
    print(f"\n{'K':>3} {'pooled':>8} {'per-user':>9}  dropped next")
    while len(live[0]) >= args.floor and len(live[1]) >= args.floor:
        k = min(len(live[0]), len(live[1]))
        tot_g, tot_n, pu, infl, used_ep = 0.0, 0, [], {}, []
        for f in (0, 1):
            if carried[f] is None:
                g_init = np.stack([pool[f][i][0] for i in live[f]])
            else:
                g_init = carried[f]
            g_init = (g_init / (np.linalg.norm(g_init, axis=1, keepdims=True)
                                + 1e-12)).astype(np.float32)
            g, W_fit, _rms, _sd, eps = train_until_settled(
                rows_by_user, tu_by_fold[f], Fk, tid_index, g_init, device,
                seed=args.seed, max_epochs=args.epochs)
            carried[f] = W_fit.astype(np.float64)
            used_ep.append(eps)
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
            lab, mid = pool[f][live[f][j]][1], pool[f][live[f][j]][2]
            # motif id alongside the name: the bank holds two motifs named
            # "killing", so a name alone does not identify what was dropped
            drop_label.append(f"{lab}" + (f"[{mid[-4:]}]" if mid else ""))
            live[f].pop(j)
            carried[f] = np.delete(carried[f], j, axis=0)
        print(f"{k:>3} {tot_g / tot_n * 1000:>8.3f} {np.mean(pu):>9.3f}  "
              f"ep {used_ep[0]}/{used_ep[1]}  {drop_label[0]} / {drop_label[1]}",
              flush=True)

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
