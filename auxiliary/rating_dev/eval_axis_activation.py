#!/usr/bin/env python3
"""SwiGLU against ReLU in the axis fit's residual-difficulty trunk.

The trunk that feeds `r` is the only nonlinearity in the shipped axis model —
the axis map itself is linear and measured to want to stay that way. Its
activation has never been chosen, only inherited, so this asks whether the
default is costing anything.

SwiGLU replaces `[Linear, ReLU]` with `silu(xW) * xV`, which carries TWICE the
parameters at the same width because the gate is its own projection. That makes a
same-width comparison a capacity comparison as much as an activation one, so both
are run: same width, and width scaled by 2/3 to match parameter count. Reading
only the same-width arm would credit the activation for capacity it was handed.

Everything else is production, and the ReLU arm is bit-identical to the shipped
trainer, so any difference is the activation and its width.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_axis_activation --seeds 0 1
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
    MIN_USER_ATTEMPTS, _stratified_folds, axes_ll_gain, build_axis_features,
    eigen_directions, load_axis_attempts,
)
from auxiliary.rating_dev.train_axes_staged import train_axes_staged  # noqa: E402
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

# (label, act, trunk_hidden)
# NOTE on "matched": scaling the widths by 2/3 does NOT match parameters here.
# The input is 216 wide so the first layer dominates, and (43,21) comes to 20,532
# against ReLU's 16,001 — 28% larger, not equal. (32,16) at ~14,961 is the honest
# match, and it is the only arm that can attribute a difference to the GATE
# rather than to the capacity that came with it.
ARMS = [
    ("relu (64,32) shipped", "relu", (64, 32)),      # 16,001
    ("swiglu (32,16) matched", "swiglu", (32, 16)),  # 14,961 — the real match
    ("swiglu (43,21)", "swiglu", (43, 21)),          # 20,532 — 2/2 win, unmatched
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
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
    tu = {f: [u for u in users if fold_of[u] != f] for f in (0, 1)}
    ev = {f: [u for u in users if fold_of[u] == f] for f in (0, 1)}
    print(f"loaded in {time.time() - t0:.0f}s: {len(users)} users, seeds "
          f"{args.seeds}", flush=True)
    g_init = {f: eigen_directions(rows_by_user, tu[f], F, tid_index, rank,
                                  params, lapse, args.k) for f in (0, 1)}
    print(f"  warm starts computed ({time.time() - t0:.0f}s)", flush=True)

    def trunk_params(act, hid):
        n, d = 0, F.shape[1]
        for w in hid:
            n += (2 if act == "swiglu" else 1) * (d * w + w)
            d = w
        return n + d + 1

    res = defaultdict(dict)
    print(f"\n{'arm':<24}{'trunk p':>9}{'seed':>5}{'pooled':>9}{'per-user':>10}"
          f"{'f0':>8}{'f1':>8}", flush=True)
    for seed in args.seeds:
        for label, act, hid in ARMS:
            tot_g, tot_n, pu, pf = 0.0, 0, [], []
            for f in (0, 1):
                g, _W, _r, _s = train_axes_staged(
                    rows_by_user, tu[f], F, tid_index, args.k, g_init[f],
                    device, seed=seed, act=act, trunk_hidden=hid)
                gain, n, gu, _nu = axes_ll_gain(rows_by_user, ev[f], g, rank,
                                                params, tid_index, lapse)
                tot_g += gain; tot_n += n; pu.append(gu)
                pf.append(gain / max(n, 1) * 1000.0)
            p, u = tot_g / max(tot_n, 1) * 1000.0, float(np.mean(pu))
            res[label][seed] = (p, u)
            print(f"{label:<24}{trunk_params(act, hid):>9,}{seed:>5}{p:>9.3f}"
                  f"{u:>10.3f}{pf[0]:>8.3f}{pf[1]:>8.3f}", flush=True)

    base = ARMS[0][0]
    print(f"\n===== paired against '{base}', seeds {args.seeds} =====")
    for label, act, hid in ARMS:
        ps = np.array([res[label][s][0] for s in args.seeds])
        us = np.array([res[label][s][1] for s in args.seeds])
        bp = np.array([res[base][s][0] for s in args.seeds])
        bu = np.array([res[base][s][1] for s in args.seeds])
        print(f"  {label:<24}{ps.mean():>9.3f}{us.mean():>10.3f}"
              f"   d {(ps - bp).mean():+.3f} / {(us - bu).mean():+.3f}"
              f"   wins {int(np.sum(ps > bp))}/{len(ps)}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
