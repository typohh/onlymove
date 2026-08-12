#!/usr/bin/env python3
"""Production axis fit with a user fold held out, so its loadings can be scored
on those users without leaking.

The shipped `axes/axes.npz` was trained on every user. Using it to score
held-out users would hand production loadings fitted on exactly the accounts the
holdout exists to protect — while the joint MIRT's dumped loadings are clean,
because its `--holdout-user-frac` run never saw them. Comparing the two would
measure the leak, not the models.

So: same `train_axes`, same regimen, same feature construction, with two
exclusions — the held-out users' rows, and the rating-model outputs read from
the matching user-holdout rating fit rather than the shipped one (four feature
columns AND the axis baseline come from there).

Writes an axes.npz in the same shape `project_loadings` reads: W, rms, v_sd,
feat_mu, feat_sd.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.fit_axes_userholdout \\
      --prod-dir /tmp/prod_uh_s0 --seed 0 --frac 0.15 --out /tmp/axes_uh_s0.npz
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from auxiliary.rating_dev.holdout_set import held_out_users
    from utils.rating.residual_axes import (
        MIN_USER_ATTEMPTS, build_axis_features, eigen_directions,
        load_axis_attempts, train_axes)
    from utils.rating.slope_calibration import load_client_lapse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hu_list, hu_hash = held_out_users(args.seed, args.frac)
    hu = set(hu_list)
    print(f"held-out users hash {hu_hash} | {len(hu_list)} accounts", flush=True)

    feats, rank, params, _ad = build_axis_features(params_dir=args.prod_dir)
    tids = sorted(feats)
    ti = {t: i for i, t in enumerate(tids)}
    F_raw = np.stack([feats[t] for t in tids])
    feat_mu, feat_sd = F_raw.mean(0), F_raw.std(0) + 1e-9
    F = ((F_raw - feat_mu) / feat_sd).astype(np.float32)

    rbu = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in ti and str(u) not in hu:          # held-out users excluded
            rbu[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rbu.items() if len(a) >= MIN_USER_ATTEMPTS)
    print(f"axis fit on {len(users)} users, "
          f"{sum(len(rbu[u]) for u in users)} rows (held-out users removed)",
          flush=True)

    lapse = load_client_lapse(0.0328)
    g_init = eigen_directions(rbu, users, F, ti, rank, params, lapse, args.k)
    _load, W, rms, v_sd = train_axes(rbu, users, F, ti, args.k, g_init, device)
    np.savez_compressed(args.out, W=W, rms=rms, v_sd=v_sd,
                        feat_mu=feat_mu.astype(np.float32),
                        feat_sd=feat_sd.astype(np.float32),
                        k=np.array(args.k), holdout_hash=np.array(hu_hash))
    print(f"wrote {args.out}  (W {W.shape}, K={args.k})", flush=True)


if __name__ == "__main__":
    main()
