#!/usr/bin/env python3
"""Epoch curve for the axis block, from ONE trajectory per (arm, seed, fold).

Training is cumulative — fixed seed, no LR schedule — so a 1600-epoch fit passes
exactly through the state a 40-epoch fit ends at. Fitting each epoch count
separately (as auxiliary.rating_dev.eval_axis_seeding does) therefore re-derives
a prefix already computed. This trains once to the largest checkpoint and scores
at each one along the way.

It also reports, at every checkpoint, the statistic that decides whether a rising
gain curve means anything: the DISPLACEMENT of the per-user axis coefficients.

  gain  — held-out walk-forward LL gain over the production-form 4PL baseline,
          the production selection criterion.
  share — mean^2 / second-moment of the fitted axis coefficients, per axis. This
          is the same form fit_population_constants applies to the shipped
          weakness prior, and it separates the two ways an axis can earn gain.
          An axis carrying genuine between-user WEAKNESS has coefficients spread
          around zero, so share is small. An axis that has drifted onto
          population residual DIFFICULTY has every user's coefficient displaced
          the same way, so share is large — and it still predicts held-out users
          better, because the difficulty is real. Gain alone cannot tell those
          apart, which is why more training raising the gain is not by itself
          evidence that more training is right.
  sign  — fraction of fits whose coefficient agrees in sign with the axis mean, a
          distribution-free companion to `share` (0.5 = no displacement, 1.0 =
          every user the same way).

The coefficients come from the PENALIZED walk-forward pass, and are per
(user, block) rather than per user — so heavy users contribute more of them,
exactly as they dominate the attempt-pooled gain. That parallel is deliberate:
both numbers then describe the same population.

The trainer below MIRRORS utils.rating.residual_axes.train_axes and must be kept
in step with it — it is duplicated only to expose the checkpoint hook, not to
vary the regimen.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_axis_epoch_curve \
      --ks 3 --arms cold,warm --seeds 0,1 --checkpoints 400,800,1600,3200
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
    BATCH, CALIBRATION_P, FLOOR_MAX, MIN_USER_ATTEMPTS, ORTHO_WEIGHT,
    _axis_prior, _fit_theta, _irls_off, _logloss, _p4pl, _walk_forward,
    build_axis_features, load_axis_attempts, seed_directions,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402


def eigen_init(rows_by_user, train_users, F, tid_index, rank, params, lapse, k):
    """Data-derived warm start: the top-k directions of the between-user
    residual, computed from the TRAINING users only.

    At v = 0 the loading gradient is identically zero, so which direction an
    axis escapes along is decided by its random draw. The direction the model
    actually wants is available in closed form: linearizing around v = 0, each
    user pulls the loadings along their own residual score

        s_u = sum_p (y_up - p_hat_up) * x_p

    under the axis-free baseline, and the dominant escape directions are the top
    eigenvectors of the CENTRED second moment of those scores. Centring is what
    keeps this about between-user structure — the uncentred mean direction is
    population residual difficulty, which the axes are forbidden to carry.

    This is a warm start in the same sense the hand-crafted scalars are, with
    the difference that nothing about it is specified by hand: the model picks
    its own directions from what it has not yet explained. Deterministic, so it
    removes the initialization lottery rather than merely damping it. Train
    users only — seeding from the eval fold's residuals would leak."""
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
    scores = []
    for u in train_users:
        a = rows_by_user[u]
        tids = [t for t, _ in a]
        y = np.array([s for _, s in a], dtype=float)
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        th = _fit_theta(R0, A, C, span, kap, y, grid)
        P4 = _p4pl(th, R0, A, C, span, kap)
        off = np.log(P4 / (1 - P4))
        X = np.column_stack([np.ones(len(tids)),
                             np.array([rank[t] for t in tids])])
        w = _irls_off(X, y, off, ridge=1e-6)
        p = 1.0 / (1.0 + np.exp(-np.clip(off + X @ w, -35, 35)))
        idx = np.array([tid_index[t] for t in tids])
        scores.append(F[idx].astype(np.float64).T @ (y - p))
    S = np.stack(scores)
    S = S - S.mean(axis=0, keepdims=True)
    ev, vec = np.linalg.eigh(S.T @ S / S.shape[0])
    top = vec[:, ::-1][:, :k].T                      # (k, n_feat)
    return (top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-9)
            ).astype(np.float32)


def train_axes_checkpointed(rows_by_user, train_users, F, tid_index, k_axes,
                            g_init, device, seed, checkpoints,
                            normalize_g=False, g_lr=1e-3, v_init_sd=0.0):
    """utils.rating.residual_axes.train_axes, yielding (epoch, loadings, row
    norms) at each checkpoint. Body kept literally in step with production
    except for the two knobs below, both defaulting to production behaviour.

    normalize_g projects each row of g_lin.weight back to unit L2 norm after
    every step. This is an OPTIMIZATION fix with no effect on the model: the
    axis output is divided by its own RMS, so a row's scale is already unidenti-
    fied by the loss. That invariance is the problem it addresses. For a
    scale-invariant function the gradient is orthogonal to the parameter, so
    every update is perpendicular to the current weight and the norm grows
    monotonically — |w_t| ~ lr*sqrt(d*t) under Adam's roughly fixed step size.
    The ANGULAR step, which is the only part that changes the model, therefore
    decays like 1/sqrt(t), and the total angle travelled grows only like
    sqrt(T). Unfixed, training does not converge so much as creep, which is
    indistinguishable from "the axes need far more epochs" if you only watch the
    gain. Projecting to the sphere makes the effective angular rate constant.
    Weight decay would also work by setting an equilibrium norm, but it changes
    regularization as well as dynamics; this changes only dynamics."""
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
    trunk = nn.Sequential(
        nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    ).to(device)
    head_r = nn.Linear(32, 1).to(device)
    g_lin = nn.Linear(n_feat, k_axes, bias=False).to(device)
    if g_init is not None:
        with torch.no_grad():
            g_lin.weight[:g_init.shape[0]].copy_(
                torch.tensor(g_init, dtype=torch.float32, device=device))
    alpha = torch.zeros(n_u, device=device, requires_grad=True)
    beta = torch.zeros(n_u, device=device, requires_grad=True)
    # v_init_sd = 0 is production: v starts at EXACTLY zero, which is the saddle
    # — the bilinear term contributes nothing to the logit and the loading
    # gradient is identically zero, so the escape direction is decided entirely
    # by which way g_lin's random init happens to point. Seeding g is one way to
    # choose that direction; breaking the symmetry on the OTHER factor is
    # another, and it lets the residual pick the direction instead of the draw.
    v = (torch.zeros(n_u, k_axes, device=device) if v_init_sd <= 0 else
         torch.randn(n_u, k_axes, device=device) * v_init_sd
         ).requires_grad_(True)

    opt = torch.optim.Adam([
        {"params": list(trunk.parameters()) + list(head_r.parameters()),
         "lr": 1e-3, "weight_decay": 1e-4},
        {"params": list(g_lin.parameters()), "lr": g_lr},
        {"params": [alpha, beta], "lr": 1e-2},
        {"params": [v], "lr": 1e-2},
    ])
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    n = len(rows)
    want = sorted(set(checkpoints))
    for ep in range(1, max(want) + 1):
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
            if normalize_g:
                with torch.no_grad():
                    g_lin.weight.div_(
                        g_lin.weight.norm(dim=1, keepdim=True) + 1e-12)
        if ep in want:
            with torch.no_grad():
                gh = g_lin(X)
                rms = gh.pow(2).mean(dim=0).sqrt() + 1e-6
                g_all = (gh / rms).cpu().numpy() * v.std(dim=0).cpu().numpy()[None, :]
                norms = g_lin.weight.norm(dim=1).cpu().numpy()
            yield ep, g_all, norms


def gain_and_leakage(rows_by_user, eval_users, g, rank, params, tid_index, lapse):
    """residual_axes.axes_ll_gain, additionally returning the per-axis
    coefficient displacement from the penalized pass."""
    _o, _s, _y, c0 = _walk_forward(rows_by_user, eval_users, g, rank, params,
                                   tid_index, lapse)
    off, sc, y, coefs = _walk_forward(rows_by_user, eval_users, g, rank, params,
                                      tid_index, lapse,
                                      penalty=_axis_prior(c0))
    sc = (sc - sc.mean()) / (sc.std() + 1e-9)
    Xk = np.column_stack([np.ones(len(y)), sc])
    w = _irls_off(Xk, y, off, ridge=1e-9)
    gain = _logloss(off, y) - _logloss(off + Xk @ w, y)
    mu = coefs.mean(axis=0)
    second = (coefs ** 2).mean(axis=0)
    share = mu ** 2 / np.maximum(second, 1e-30)
    sign = np.mean(np.sign(coefs) == np.sign(mu)[None, :], axis=0)
    return gain, len(y), share, sign, mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="3")
    ap.add_argument("--arms", default="cold,warm")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--checkpoints", default="400,800,1600,3200")
    ap.add_argument("--normalize-g", action="store_true",
                    help="project g_lin rows to unit norm after each step "
                         "(optimization fix; see train_axes_checkpointed)")
    ap.add_argument("--v-init-sd", type=float, default=0.0,
                    help="init user positions at N(0, sd) instead of exactly 0 "
                         "(0 = production; breaks the bilinear saddle at its "
                         "source so the residual, not the draw, picks the "
                         "escape direction)")
    ap.add_argument("--g-lr", type=float, default=1e-3,
                    help="learning rate for the axis head (production 1e-3; "
                         "note v runs at 1e-2 and the bilinear gradient on each "
                         "factor scales with the other)")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    arms = args.arms.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]
    cps = sorted(int(x) for x in args.checkpoints.split(","))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    feats, scalars, rank, params, _ad = build_axis_features()
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
    print(f"device {device}, loaded in {time.time() - t0:.0f}s: {len(tids)} "
          f"problems, {len(users)} users, checkpoints {cps}", flush=True)

    acc = defaultdict(lambda: [0.0, 0, [], []])   # (k,arm,ep,seed) -> g,n,share,sign
    for k in ks:
        g_seed = seed_directions(F, S_mat, min(3, k))
        for seed in seeds:
            for arm in arms:
                for fold in (0, 1):
                    tu = [u for u in users if fold_of[u] != fold]
                    eu = [u for u in users if fold_of[u] == fold]
                    if arm == "warm":
                        g_init = g_seed
                    elif arm == "eigen":
                        # per FOLD, from that fold's training users only
                        g_init = eigen_init(rows_by_user, tu, F, tid_index,
                                            rank, params, lapse, k)
                    else:
                        g_init = None
                    for ep, g, norms in train_axes_checkpointed(
                            rows_by_user, tu, F, tid_index, k, g_init, device,
                            seed, cps, normalize_g=args.normalize_g,
                            g_lr=args.g_lr, v_init_sd=args.v_init_sd):
                        ts = time.time()
                        gain, n, share, sign, mu = gain_and_leakage(
                            rows_by_user, eu, g, rank, params, tid_index, lapse)
                        a = acc[(k, arm, ep, seed)]
                        a[0] += gain
                        a[1] += n
                        a[2].append(share)
                        a[3].append(sign)
                        print(f"  K={k} {arm:4s} ep={ep:5d} seed={seed} "
                              f"fold={fold}: gain {gain / n * 1000:.3f} "
                              f"share {np.round(share, 3)} sign "
                              f"{np.round(sign, 2)} |g| {np.round(norms, 2)} "
                              f"({time.time() - ts:.0f}s)", flush=True)
                        if a[1] and len(a[2]) == 2:
                            print(f"K={k} {arm} ep={ep} seed={seed}: POOLED "
                                  f"{a[0] / a[1] * 1000:.3f} mnat/attempt  "
                                  f"worst-axis share "
                                  f"{np.max(np.mean(a[2], axis=0)):.3f}",
                                  flush=True)

    print("\n===== epoch curve: gain vs displacement =====", flush=True)
    print("   K arm   epochs   gain(mnat)  sd    worst-share  worst-sign")
    for k in ks:
        for arm in arms:
            for ep in cps:
                vals = [acc[(k, arm, ep, s)] for s in seeds
                        if (k, arm, ep, s) in acc]
                if not vals:
                    continue
                g = [a[0] / a[1] * 1000 for a in vals if a[1]]
                sh = np.max([np.mean(a[2], axis=0) for a in vals])
                sg = np.max([np.mean(a[3], axis=0) for a in vals])
                print(f"  {k:2d} {arm:4s} {ep:6d}   {np.mean(g):8.3f}  "
                      f"{np.std(g):5.3f}      {sh:6.3f}      {sg:6.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
