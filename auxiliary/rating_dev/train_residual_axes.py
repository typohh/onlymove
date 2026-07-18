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

    x_p = [joint-PCA activations (PCA_DIM=128 — pipeline.train_rating's EXACT
           block: same loader, same cache, same seed), bot winrate (common
           columns), move evals (20), game-phase RAW components
           (moves-before-position, main-line length — the net can build the
           ratio itself; tightness gaps are already move-eval slots),
           size_is_13, rating, slope, floor, rank]  (z-scored)
    LITERAL feature parity with the rating trainer by design: every input
    block the rating NN consumes, plus its OUTPUTS (rating/slope/floor/rank).
    The skew-ICA embedding (embeddings/joint/) is deliberately NOT an input —
    the production form of this model must never depend on the motif pipeline.
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
  - A second baseline family runs alongside: the PRODUCTION-FORM 4PL — per
    block a walk-forward theta MLE under the client's exact forward model
    P = floor + (1 - floor - LAPSE)*sigmoid(slope*(theta - rating) + kappa(floor))
    (pwa/js/irt.js; LAPSE from ratings/population_constants.json), used as a
    fixed logit offset for the same IRLS fits. This matters because both r and
    g see rating/slope/floor as inputs, so kappa vs the LINEAR baseline can be
    inflated by mere 4PL curve-shape reconstruction the client already applies.
    kappa(r | 4PL) is the shippable-r gate; kappa(axes | 4PL + scalars) is the
    strict axis go/no-go.

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
    CACHE_DIR, _cache_ids_path, load_backend_problem_params,
    load_joint_activations, load_joint_bot_winrate, load_joint_game_phase,
    load_joint_move_features, load_tightness_gaps, phase_ratio_z, zscore_neglog,
)

# ------------------------------------------------------------------ features
def build_problem_features():
    """tid -> feature vector (z-scored later); plus tid -> rank, the 3 eval
    scalars, and (rating, slope, floor) for the 4PL baseline.

    LITERAL parity with pipeline.train_rating: the same loaders, the same
    joint-PCA cache, the same seed, the same [act | bw | move | phase | size]
    block layout — followed by the rating model's outputs. Membership mirrors
    the trainer too (scorable iff act∩bw; move/phase mean-imputed to zero),
    further gated on the eval-side inputs (scalars, rank, rating params).
    """
    from pipeline.train_rating import PCA_DIM, SEED, SIZES

    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, _k_act = load_joint_activations(SIZES, tids_by_size, PCA_DIM, SEED)
    bw, _nb = load_joint_bot_winrate(SIZES, tids_by_size)
    mv, nmv = load_joint_move_features(SIZES, tids_by_size)
    ph, nph = load_joint_game_phase(SIZES, tids_by_size)
    zero_mv = np.zeros(nmv, dtype=np.float32)
    zero_ph = np.zeros(nph, dtype=np.float32)

    # The derived scalar forms feed only the EVAL baseline (conditional kappa
    # vs the personalized hand-crafted axes), never the network.
    f_sc = zscore_neglog(load_tightness_gaps("score_gap"))
    f_wr = zscore_neglog(load_tightness_gaps("wr_gap"))
    f_ph = phase_ratio_z()

    prank = load_ranks()
    # rating/slope/floor from the BACKEND ratings/ CSVs (the stage before
    # package_for_pwa) — same values packaging copies, but fresh the moment
    # pipeline.train_rating finishes and immune to a stale dist.
    rsf = {t: np.array(v, dtype=np.float32)
           for t, v in load_backend_problem_params().items()}

    feats, scalars, rank, params = {}, {}, {}, {}
    for key in act:
        size, t = key
        t = str(t)
        if (key not in bw or t not in f_sc or t not in f_wr or t not in f_ph
                or t not in prank or t not in rsf):
            continue
        feats[t] = np.concatenate([
            act[key], bw[key], mv.get(key, zero_mv), ph.get(key, zero_ph),
            [1.0 if size == 13 else 0.0],
            rsf[t], [prank[t]],
        ]).astype(np.float32)
        scalars[t] = np.array([f_wr[t], f_sc[t], f_ph[t]], dtype=np.float32)
        rank[t] = prank[t]
        params[t] = rsf[t]  # (rating, slope, floor) for the 4PL baseline
    return feats, scalars, rank, params


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
def load_lapse():
    """Client LAPSE (fat-finger rate) from the backend population constants —
    the same value pwa/js/irt.js bakes in."""
    path = os.path.join(PROJECT_ROOT, "ratings", "population_constants.json")
    with open(path) as f:
        return float(json.load(f)["fat_finger_rate"])


def p4pl(theta, R0, A, C, span, kap):
    """Production 4PL (pwa/js/irt.js): P = c + span*sigmoid(a*(theta-r0)+kappa(c)).
    theta may be scalar or (G, 1) against (n,) problem arrays."""
    z = np.clip(A * (theta - R0) + kap, -35, 35)
    return np.clip(C + span / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)


def fit_theta_4pl(R0, A, C, span, kap, y, grid):
    """Walk-forward theta MLE on one window under the production 4PL: coarse
    grid argmax + one parabolic refinement (1-D, well-behaved)."""
    P = p4pl(grid[:, None], R0[None, :], A[None, :], C[None, :],
             span[None, :], kap[None, :])
    ll = (y[None, :] * np.log(P) + (1 - y[None, :]) * np.log(1 - P)).sum(axis=1)
    i = int(np.argmax(ll))
    if 0 < i < len(grid) - 1:
        d = (ll[i - 1] - ll[i + 1]) / (2 * (ll[i - 1] - 2 * ll[i] + ll[i + 1]) + 1e-12)
        return float(grid[i] + np.clip(d, -1, 1) * (grid[1] - grid[0]))
    return float(grid[i])


def irls_off(X, y, off, ridge=1e-6, w0=None):
    """irls (auxiliary.eval_score_gap_axis) with a fixed logit offset."""
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    for _ in range(100):
        z = np.clip(off + X @ w, -35, 35)
        p = 1 / (1 + np.exp(-z))
        Wd = p * (1 - p) + 1e-9
        H = X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * w)

        def dev(wv):
            zz = np.clip(off + X @ wv, -35, 35)
            pp = np.clip(1 / (1 + np.exp(-zz)), 1e-9, 1 - 1e-9)
            return -(y * np.log(pp) + (1 - y) * np.log(1 - pp)).sum() + 0.5 * ridge * (wv @ wv)

        d0 = dev(w)
        for _h in range(12):
            if dev(w + step) <= d0 + 1e-9:
                break
            step *= 0.5
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


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


def eval_fold(eval_users, by_user, g, r, scalars, rank, params, tid_index, lapse):
    """Walk-forward on held-out users with frozen loadings. Returns pooled arrays."""
    acc = {"off": [], "axes": [], "off_sca": [], "axes_sca": [], "y": [], "r": [],
           "off4": [], "axes4": [], "off4_sca": [], "axes4_sca": []}
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
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
        # production-4PL per-attempt problem params (floor clamped as the
        # client clamps it: [0, FLOOR_MAX=0.78], and below the anchor P)
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, 0.78)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((0.8 - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        X_ax = np.column_stack([np.ones(len(a)), rk, G])
        X_b = X_ax[:, :2]
        X_sca = np.column_stack([np.ones(len(a)), rk, S])
        X_full = np.column_stack([np.ones(len(a)), rk, S, G])
        w_ax = w_b = w_sca = w_full = None
        w4_ax = w4_b = w4_sca = w4_full = None
        for t0 in range(WARMUP, len(a), EVAL_BLOCK):
            lo = max(0, t0 - WINDOW)
            hi = min(t0 + EVAL_BLOCK, len(a))
            try:
                w_ax, _ = irls(X_ax[lo:t0], y[lo:t0], w0=w_ax)
                w_b, _ = irls(X_b[lo:t0], y[lo:t0], w0=w_b)
                w_sca, _ = irls(X_sca[lo:t0], y[lo:t0], w0=w_sca)
                w_full, _ = irls(X_full[lo:t0], y[lo:t0], w0=w_full)
                # production-form baseline: theta MLE on the window, then the
                # 4PL logit as a FIXED offset under the same IRLS fits
                theta = fit_theta_4pl(R0[lo:t0], A[lo:t0], C[lo:t0],
                                      span[lo:t0], kap[lo:t0], y[lo:t0], grid)
                P4 = p4pl(theta, R0[lo:hi], A[lo:hi], C[lo:hi],
                          span[lo:hi], kap[lo:hi])
                off4 = np.log(P4 / (1 - P4))
                o_tr, o_ev = off4[:t0 - lo], off4[t0 - lo:]
                w4_ax = irls_off(X_ax[lo:t0], y[lo:t0], o_tr, w0=w4_ax)
                w4_b = irls_off(X_b[lo:t0], y[lo:t0], o_tr, w0=w4_b)
                w4_sca = irls_off(X_sca[lo:t0], y[lo:t0], o_tr, w0=w4_sca)
                w4_full = irls_off(X_full[lo:t0], y[lo:t0], o_tr, w0=w4_full)
            except np.linalg.LinAlgError:
                continue
            off = np.clip(X_b[t0:hi] @ w_b, -35, 35)
            acc["off"].append(off)
            acc["axes"].append(X_ax[t0:hi, 2:] @ w_ax[2:])
            acc["off_sca"].append(np.clip(X_sca[t0:hi] @ w_sca, -35, 35))
            acc["axes_sca"].append(X_full[t0:hi, 5:] @ w_full[5:])
            acc["off4"].append(np.clip(o_ev + X_b[t0:hi] @ w4_b, -35, 35))
            acc["axes4"].append(X_ax[t0:hi, 2:] @ w4_ax[2:])
            acc["off4_sca"].append(np.clip(o_ev + X_sca[t0:hi] @ w4_sca, -35, 35))
            acc["axes4_sca"].append(X_full[t0:hi, 5:] @ w4_full[5:])
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
    ap.add_argument("--no-act", action="store_true",
                    help="ablation: drop the joint-PCA activation block from the features")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    feats, scalars, rank, params = build_problem_features()
    lapse = load_lapse()
    if args.no_act:
        from pipeline.train_rating import PCA_DIM
        feats = {t: v[PCA_DIM:] for t, v in feats.items()}
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
        res = eval_fold(eval_users, by_user, g, r, scalars, rank, params,
                        tid_index, lapse)
        if not res:
            continue
        rc = res["r"] - res["r"].mean()
        metrics = [
            ("kappa(joint axes | rank baseline)", res["off"], res["axes"]),
            ("kappa(axes | personalized wr/sc/phase)", res["off_sca"], res["axes_sca"]),
            ("kappa(population r head, centered)", res["off"], rc),
            ("kappa(joint axes | 4PL baseline)", res["off4"], res["axes4"]),
            ("kappa(axes | 4PL + scalars)  <- strict go/no-go", res["off4_sca"], res["axes4_sca"]),
            ("kappa(r head | 4PL baseline) <- shippable-r gate", res["off4"], rc),
        ]
        print(f"fold {fold}: held-out n={len(res['y'])}")
        row = []
        for name, off, sc in metrics:
            k, se = offset_kappa(off, sc, res["y"])
            print(f"  {name:<50s} = {k:+.3f} +- {se:.3f} (z {k / se:+.1f})")
            row.append((k, se))
        results.append(row)

    if len(results) == 2:
        ivw = lambda m, s: ((m / s**2).sum() / (1 / s**2).sum(),
                            np.sqrt(1 / (1 / s**2).sum()))
        print(f"\n=== pooled over folds ===")
        names = [m[0] for m in metrics]
        for i, name in enumerate(names):
            m = np.array([r[i][0] for r in results])
            s = np.array([r[i][1] for r in results])
            k, se = ivw(m, s)
            print(f"{name:<52s} = {k:+.3f} +- {se:.3f} (z {k / se:+.1f})")

    out = os.path.join(CACHE_DIR, f"residual_axes_k{args.k}_seed{args.seed}.npz")
    np.savez_compressed(out, tids=np.array(tids), **save)
    print(f"\nsaved loadings -> {out}")


if __name__ == "__main__":
    main()
