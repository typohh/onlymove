#!/usr/bin/env python3
"""Residual MIRT: learn per-problem SECONDARY AXES from human between-user residuals.

Supervised alternative to skew-ICA axes and hand-crafted scalars (see
docs/scalar_weakness_axes.md): on top of a per-user difficulty baseline, a small
network learns K per-problem axis loadings whose per-user positions explain the
between-user response residuals — the axes are DEFINED by validated human
variation instead of hoped-for from activation statistics.

Model (attempt-level):
    logit P(success) = alpha_u + beta_u * rank_p          (per-user baseline)
                       + r(x_p)                           (population residual
                                                            difficulty — real
                                                            signal the 4PL missed;
                                                            kept in its OWN head so
                                                            the axes' capacity isn't
                                                            spent on it: the mean
                                                            signal is ~100x the
                                                            between-user variance)
                       + sum_k g_k(x_p) * v_u[k]          (secondary axes)

    x_p = [joint-ICA emb (64), move evals (20), game-phase RAW components
           (moves-before-position, main-line length — the net can build the
           ratio itself; tightness gaps are already move-eval slots),
           rating, slope, floor, rank]  (z-scored)
    r, g: shared MLP trunk with dropout (the rating backbone's overfit lesson),
    v_u ~ N(0, I) prior, centered per axis (r carries any mean).

Training/eval protocol (per docs/scalar_weakness_axes.md metrics):
  - 2-fold split BY USER. g/r are trained on fold-train users only; held-out
    users never influence the axis directions they are scored on.
  - Evaluation on held-out users reuses the walk-forward harness of
    auxiliary.eval_score_gap_axis: at each block, the user's axis positions are
    re-estimated from the trailing 500 attempts by ridge logistic on the FROZEN
    loadings (exactly what a client would do), and the next block is predicted.
    Reported: kappa of the joint axis prediction (offset difficulty-only
    baseline), kappa conditional on the personalized hand-crafted scalars
    [wr, sc, phase_ratio] (the go/no-go), and the population r head's held-out
    log-loss gain (the "leaked difficulty" bonus, measured).

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.train_residual_axes            # K=4
  ./.venv/bin/python3 -m auxiliary.rating_dev.train_residual_axes --k 2 --epochs 30
Outputs: ratings_dev artifacts under activations/_signal_cache/residual_axes_*.npz
(per-problem loadings + r), evaluation report to stdout.
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.eval_score_gap_axis import (  # noqa: E402
    DB_PATH, WINDOW, EVAL_BLOCK, WARMUP, load_ranks, irls,
)
# Data sources come from the shared production home (utils.rating_features) —
# the same module pipeline.train_rating loads its feature blocks from, so the
# rating trainer and this prototype stay in sync by construction.
from utils.rating_features import (  # noqa: E402
    CACHE_DIR, load_game_phase, load_tightness_gaps, phase_ratio_z, zscore_neglog,
)

PARAMS_BIN = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "problem_params.bin")


# ------------------------------------------------------------------ features
def build_problem_features():
    """tid -> feature vector (z-scored later); plus tid -> rank and the 3 scalars."""
    import csv
    import struct

    emb = {}
    with open(os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")) as f:
        reader = csv.reader(f)
        header = next(reader)
        k_emb = sum(1 for c in header if c.startswith("embed_"))
        for row in reader:
            emb[row[0]] = np.array(row[2:2 + k_emb], dtype=np.float32)

    moves = {}
    for size in (9, 13):
        d = np.load(os.path.join(CACHE_DIR, f"move_evals_{size}.npz"))
        for t, v in zip(d["tids"], d["feats"]):
            moves[str(t)] = v.astype(np.float32)

    # NETWORK inputs carry the RAW game-phase components [moves-before, main-line]
    # (the net can construct main/(prefix+main) itself if useful); the derived
    # scalar forms below feed only the EVAL baseline (conditional kappa vs the
    # personalized hand-crafted axes). Tightness gaps are NOT separate network
    # inputs — their components are already slots of the move-eval block.
    f_sc = zscore_neglog(load_tightness_gaps("score_gap"))
    f_wr = zscore_neglog(load_tightness_gaps("wr_gap"))
    phase = load_game_phase()
    f_ph = phase_ratio_z(phase)

    prank = load_ranks()
    raw = open(PARAMS_BIN, "rb").read()
    rsf = {}
    for off in range(0, len(raw) - 15, 16):
        pid = raw[off:off + 4].hex()
        rsf[pid] = np.array(struct.unpack_from("<fff", raw, off + 4), dtype=np.float32)

    feats, scalars, rank = {}, {}, {}
    for t in emb:
        if t in moves and t in f_sc and t in f_wr and t in f_ph and t in prank and t in rsf:
            ph = phase[t]
            feats[t] = np.concatenate([
                emb[t], moves[t],
                [ph[0] + ph[1], ph[2]],
                rsf[t], [prank[t]],
            ]).astype(np.float32)
            scalars[t] = np.array([f_wr[t], f_sc[t], f_ph[t]], dtype=np.float32)
            rank[t] = prank[t]
    return feats, scalars, rank


def load_attempts():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """SELECT user_id, problem, success FROM tsumego_reports
            WHERE is_repeat = 0 GROUP BY user_id, problem, created_at
            ORDER BY user_id, created_at, id"""
    ).fetchall()
    con.close()
    return rows


# ------------------------------------------------------------------ training
def train_fold(train_users, by_user, F, tid_index, k_axes, epochs, seed, device):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    uids = sorted(train_users)
    uindex = {u: i for i, u in enumerate(uids)}
    rows = []
    for u in uids:
        for tid, s in by_user[u]:
            rows.append((uindex[u], tid_index[tid], s))
    U = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    P = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    Y = torch.tensor([float(r[2]) for r in rows], device=device)
    counts = torch.bincount(U, minlength=len(uids)).clamp(min=1).float()
    X = torch.tensor(F, device=device)
    RANK = X[:, -1].clone()

    n_u, n_feat = len(uids), F.shape[1]
    trunk = nn.Sequential(
        nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    ).to(device)
    head_r = nn.Linear(32, 1).to(device)
    head_g = nn.Linear(32, k_axes).to(device)
    alpha = torch.zeros(n_u, device=device, requires_grad=True)
    beta = torch.zeros(n_u, device=device, requires_grad=True)
    v = torch.zeros(n_u, k_axes, device=device, requires_grad=True)

    opt = torch.optim.Adam([
        {"params": list(trunk.parameters()) + list(head_r.parameters())
                   + list(head_g.parameters()), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": [alpha, beta], "lr": 1e-2},
        {"params": [v], "lr": 1e-2},
    ])
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    n = len(rows)
    batch = 8192
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            u, p, y = U[idx], P[idx], Y[idx]
            h = trunk(X[p])
            logit = (alpha[u] + beta[u] * RANK[p]
                     + head_r(h).squeeze(-1)
                     + (head_g(h) * v[u]).sum(-1))
            # v prior N(0, I): 0.5*||v_u||^2 ONCE per user per epoch — each
            # attempt carries its user's prior share 1/n_u, so shrinkage
            # relaxes as a user's data grows (a fixed-per-attempt prior would
            # over-shrink heavy users). Mean-center penalty keeps the
            # population component in head_r ("difficulty leakage" gets its
            # own slot instead of spending axis capacity).
            loss = bce(logit, y) \
                + 0.5 * ((v[u] ** 2).sum(dim=1) / counts[u]).sum() / len(idx) \
                + 10.0 * (v.mean(dim=0) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs} loss {tot / n:.4f}")

    with torch.no_grad():
        trunk.eval()
        h = trunk(X)
        g_all = head_g(h).cpu().numpy()          # (n_problems, K)
        r_all = head_r(h).squeeze(-1).cpu().numpy()
        # scale axes so v ~ unit variance across train users -> loadings carry
        # the magnitude (kappa is scale-invariant; this is for readability)
        sd = v.std(dim=0).cpu().numpy()
        g_all = g_all * sd[None, :]
    return g_all, r_all, v.detach().cpu().numpy(), uids


# ------------------------------------------------------------------ evaluation
def offset_kappa(off, sc, yy):
    Xk = np.column_stack([np.ones(len(yy)), sc])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(off + Xk @ w, -35, 35)))
        Wd = p * (1 - p) + 1e-9
        step = np.linalg.solve(Xk.T @ (Xk * Wd[:, None]) + 1e-9 * np.eye(2), Xk.T @ (yy - p))
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-np.clip(off + Xk @ w, -35, 35)))
    Wd = p * (1 - p) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv(Xk.T @ (Xk * Wd[:, None]))))[1]
    return w[1], se


def eval_fold(eval_users, by_user, g, r, scalars, rank, tid_index):
    """Walk-forward on held-out users with frozen loadings. Returns pooled arrays."""
    acc = {"off": [], "axes": [], "off_sca": [], "axes_sca": [], "y": [], "r": []}
    for u in eval_users:
        a = by_user[u]
        if len(a) < WARMUP + EVAL_BLOCK:
            continue
        tids = [t for t, _s in a]
        y = np.array([s for _t, s in a], dtype=float)
        rk = np.array([rank[t] for t in tids])
        G = np.stack([g[tid_index[t]] for t in tids])           # (n, K)
        R = np.array([r[tid_index[t]] for t in tids])
        S = np.stack([scalars[t] for t in tids])                # (n, 3)
        X_ax = np.column_stack([np.ones(len(a)), rk, G])
        X_b = X_ax[:, :2]
        X_sca = np.column_stack([np.ones(len(a)), rk, S])
        X_full = np.column_stack([np.ones(len(a)), rk, S, G])
        w_ax = w_b = w_sca = w_full = None
        for t0 in range(WARMUP, len(a), EVAL_BLOCK):
            lo = max(0, t0 - WINDOW)
            hi = min(t0 + EVAL_BLOCK, len(a))
            try:
                w_ax, _ = irls(X_ax[lo:t0], y[lo:t0], w0=w_ax)
                w_b, _ = irls(X_b[lo:t0], y[lo:t0], w0=w_b)
                w_sca, _ = irls(X_sca[lo:t0], y[lo:t0], w0=w_sca)
                w_full, _ = irls(X_full[lo:t0], y[lo:t0], w0=w_full)
            except np.linalg.LinAlgError:
                continue
            off = np.clip(X_b[t0:hi] @ w_b, -35, 35)
            acc["off"].append(off)
            acc["axes"].append(X_ax[t0:hi, 2:] @ w_ax[2:])
            acc["off_sca"].append(np.clip(X_sca[t0:hi] @ w_sca, -35, 35))
            acc["axes_sca"].append(X_full[t0:hi, 5:] @ w_full[5:])
            acc["y"].append(y[t0:hi])
            # population r head: collected for a kappa fit at the fold level
            # (raw addition would conflate scale/centering with signal — the
            # walk-forward baseline already absorbs the served mix's average).
            acc["r"].append(R[t0:hi])
    return {k: np.concatenate(v) for k, v in acc.items() if v}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=4, help="number of secondary axes")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-emb", action="store_true",
                    help="ablation: drop the 64 ICA embedding dims from the features")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    feats, scalars, rank = build_problem_features()
    if args.no_emb:
        feats = {t: v[64:] for t, v in feats.items()}
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
    print(f"problems with full features: {len(tids)}; feature dim {F.shape[1]}")

    rows = load_attempts()
    by_user = defaultdict(list)
    for u, p, s in rows:
        if p in tid_index:
            by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in by_user.items() if len(a) >= 50)
    print(f"users with >= 50 usable attempts: {len(users)} "
          f"({sum(len(by_user[u]) for u in users)} attempts)")

    # 2-fold split by user (stable hash — builtin hash() is process-salted)
    import hashlib
    fold_of = {u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 for u in users}
    results = []
    save = {}
    for fold in (0, 1):
        train_users = [u for u in users if fold_of[u] != fold]
        eval_users = [u for u in users if fold_of[u] == fold]
        print(f"\n--- fold {fold}: train {len(train_users)} users, eval {len(eval_users)} ---")
        g, r, v, uids = train_fold(train_users, by_user, F, tid_index,
                                   args.k, args.epochs, args.seed, device)
        save[f"g_fold{fold}"] = g
        save[f"r_fold{fold}"] = r
        res = eval_fold(eval_users, by_user, g, r, scalars, rank, tid_index)
        if not res:
            continue
        k_j, se_j = offset_kappa(res["off"], res["axes"], res["y"])
        k_c, se_c = offset_kappa(res["off_sca"], res["axes_sca"], res["y"])
        k_r, se_r = offset_kappa(res["off"], res["r"] - res["r"].mean(), res["y"])
        print(f"fold {fold}: held-out n={len(res['y'])}")
        print(f"  kappa(joint axes | rank baseline)          = {k_j:+.3f} +- {se_j:.3f} (z {k_j / se_j:+.1f})")
        print(f"  kappa(axes | personalized wr/sc/phase)     = {k_c:+.3f} +- {se_c:.3f} (z {k_c / se_c:+.1f})  <- go/no-go")
        print(f"  kappa(population r head, centered)         = {k_r:+.3f} +- {se_r:.3f} (z {k_r / se_r:+.1f})")
        results.append((k_j, se_j, k_c, se_c, k_r, se_r))

    if len(results) == 2:
        km = np.array([r[0] for r in results]); ks = np.array([r[1] for r in results])
        cm = np.array([r[2] for r in results]); cs = np.array([r[3] for r in results])
        ivw = lambda m, s: ((m / s**2).sum() / (1 / s**2).sum(),
                            np.sqrt(1 / (1 / s**2).sum()))
        rm = np.array([r[4] for r in results]); rs = np.array([r[5] for r in results])
        kj, kjse = ivw(km, ks); kc, kcse = ivw(cm, cs); kr, krse = ivw(rm, rs)
        print(f"\n=== pooled over folds ===")
        print(f"kappa(joint axes)      = {kj:+.3f} +- {kjse:.3f} (z {kj / kjse:+.1f})")
        print(f"kappa(axes | scalars)  = {kc:+.3f} +- {kcse:.3f} (z {kc / kcse:+.1f})")
        print(f"kappa(population r)    = {kr:+.3f} +- {krse:.3f} (z {kr / krse:+.1f})")

    out = os.path.join(CACHE_DIR, f"residual_axes_k{args.k}_seed{args.seed}.npz")
    np.savez_compressed(out, tids=np.array(tids), **save)
    print(f"\nsaved loadings -> {out}")


if __name__ == "__main__":
    main()
