#!/usr/bin/env python3
"""Production's rating / rating+axis / axis-increment on HELD-OUT PROBLEMS.

Production has no "with axes" logloss of its own — its axes are a per-user
personalisation layer the client consumes, not part of the rating prediction. To
put its axis contribution on the same row set as the joint MIRT's, one is
constructed here in the MIRT's functional form out of production's parts:

    P = c + (1 - c - lambda) * sigmoid( a*(theta - b) + sum_k g_k * v_u[k] )

with (a, b, c, lambda, theta) FROZEN from the held-out-trained 4PL, g from an
axis fit, and v fitted per user. Setting v = 0 recovers production's rating-only
prediction exactly, so the increment is attributable to the axes alone.

EVERY LINK EXCLUDES THE HELD-OUT PROBLEMS:
  4PL          --prod-dir, i.e. train_rating_staged --holdout-problem-frac
  axis features the same directory, via build_axis_features(params_dir=...);
               these supply four feature columns AND the axis baseline, so
               reading the shipped fit would hand the axes rating/slope/floor
               derived from attempts on the very problems being scored
  axis fit     trained only on rows of NON-held-out problems
  per-user v   fitted only on rows of NON-held-out problems

One residual leak, stated rather than hidden: the rank feature column still
comes from the shipped rating->rank curve, a global monotone map fitted on
survey declarations rather than held-out outcomes. Second-order, not zero.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.prod_axis_heldout \\
      --prod-dir /tmp/prod_ho --seed 0 --frac 0.15 --k 3
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

EPS = 1e-6
V_RIDGE = 0.5          # matches the joint model's per-user shrinkage
V_STEPS = 400
V_LR = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dir", default="/tmp/prod_ho")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    import torch
    from auxiliary.rating_dev.holdout_set import held_out_keys
    from auxiliary.rating_dev.validate_ratings import load_ratings, clear_logit_vec
    from utils.rating.features import build_profiles, load_first_time_attempts
    from utils.rating.residual_axes import (
        MIN_USER_ATTEMPTS, build_axis_features, eigen_directions,
        load_axis_attempts, train_axes)
    from utils.rating.slope_calibration import load_client_lapse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ho_keys, ho_hash = held_out_keys(args.seed, args.frac)
    ho_tids = {t for _s, t in ho_keys}
    print(f"held-out hash {ho_hash} | {len(ho_tids)} problems", flush=True)

    rate, slope, floor, lam, theta = load_ratings(args.prod_dir)
    print(f"4PL from {args.prod_dir}: {len(rate)} problems, lapse {lam:.4f}",
          flush=True)

    # --- axis features and fit, both on the held-out-trained 4PL
    feats, rank, params, _ad = build_axis_features(params_dir=args.prod_dir)
    tids = sorted(feats)
    ti = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float32)
    rbu_all = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in ti:
            rbu_all[u].append((p, int(bool(s))))
    # the axis fit sees TRAINING problems only
    rbu_tr = {u: [(p, s) for p, s in a if p not in ho_tids]
              for u, a in rbu_all.items()}
    users = sorted(u for u, a in rbu_tr.items() if len(a) >= MIN_USER_ATTEMPTS)
    print(f"axis fit: {len(users)} users, "
          f"{sum(len(rbu_tr[u]) for u in users)} training rows", flush=True)
    lapse_c = load_client_lapse(0.0328)
    g_init = eigen_directions(rbu_tr, users, F, ti, rank, params, lapse_c,
                              args.k)
    load, _W, _rms, _sd = train_axes(rbu_tr, users, F, ti, args.k, g_init,
                                     device)
    print(f"axes trained: loadings {load.shape}", flush=True)

    # --- rows: production's own profile structure, split by the held-out set
    df = load_first_time_attempts()
    df = df[df["size"].isin([9, 13])]
    prof = build_profiles(df)
    prof = prof[prof["problem"].isin(rate)
                & prof["profile"].astype(str).isin(theta)]
    labs = prof["profile"].astype(str).to_numpy()
    tid_arr = prof["problem"].to_numpy()
    y = prof["success"].to_numpy(float)
    keep = np.array([t in ti for t in tid_arr])
    labs, tid_arr, y = labs[keep], tid_arr[keep], y[keep]
    is_ho = np.array([t in ho_tids for t in tid_arr])
    uid_s = np.array([l[:8] for l in labs])
    uu, uinv = np.unique(uid_s, return_inverse=True)
    gi = np.array([ti[t] for t in tid_arr])
    G = load[gi]
    a = np.array([slope[t] for t in tid_arr])
    c = np.array([floor[t] for t in tid_arr])
    r0 = np.array([rate[t] for t in tid_arr])
    b = r0 - clear_logit_vec(c, lam) / np.maximum(a, 1e-9)
    th = np.array([theta[l] for l in labs])
    base = a * (th - b)                       # production's rating-only logit
    print(f"{len(y)} rows | {int(is_ho.sum())} held-out | {len(uu)} users",
          flush=True)

    # --- per-user v on TRAINING rows only
    tb = torch.tensor(base, device=device, dtype=torch.float32)
    tG = torch.tensor(G, device=device, dtype=torch.float32)
    tc = torch.tensor(c, device=device, dtype=torch.float32)
    ty = torch.tensor(y, device=device, dtype=torch.float32)
    tu = torch.tensor(uinv, device=device, dtype=torch.long)
    tr_i = torch.tensor(np.where(~is_ho)[0], device=device)
    ho_i = torch.tensor(np.where(is_ho)[0], device=device)
    v = torch.zeros(len(uu), args.k, device=device, requires_grad=True)
    opt = torch.optim.Adam([v], lr=V_LR)
    cnt = torch.bincount(tu[tr_i], minlength=len(uu)).clamp(min=1).float()
    for step in range(V_STEPS):
        i = tr_i
        z = tb[i] + (tG[i] * v[tu[i]]).sum(-1)
        p = (tc[i] + (1 - tc[i] - lam) * torch.sigmoid(z)).clamp(EPS, 1 - EPS)
        loss = torch.nn.functional.binary_cross_entropy(p, ty[i])
        loss = loss + V_RIDGE * ((v[tu[i]] ** 2).sum(dim=1)
                                 / cnt[tu[i]]).sum() / len(i)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 100 == 0:
            print(f"    v fit step {step + 1}: train BCE {float(loss):.4f}",
                  flush=True)

    def score(idx, use_v):
        with torch.no_grad():
            z = tb[idx] + ((tG[idx] * v[tu[idx]]).sum(-1) if use_v else 0.0)
            p = (tc[idx] + (1 - tc[idx] - lam)
                 * torch.sigmoid(z)).clamp(EPS, 1 - EPS)
            return float(torch.nn.functional.binary_cross_entropy(p, ty[idx]))

    r_only, r_axis = score(ho_i, False), score(ho_i, True)
    print(f"\nPRODUCTION on {len(ho_i)} held-out rows")
    print(f"  rating only    {r_only:.4f}")
    print(f"  rating + axis  {r_axis:.4f}")
    print(f"  axis adds      {(r_only - r_axis) * 1000:+.3f} mnat/attempt")


if __name__ == "__main__":
    main()
