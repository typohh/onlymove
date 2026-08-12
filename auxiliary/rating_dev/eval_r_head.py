#!/usr/bin/env python3
"""Is the residual head's trunk the right size, and why does it lack input dropout?

Two asymmetries in the shipped axis fit that were inherited rather than chosen.

SIZE. The r head spends ~16,000 parameters to emit ONE scalar per problem, while
the axis map spends 648 to emit K. Some asymmetry is justified — r is a
population quantity fitted from every attempt in the fold, whereas an axis's
value reaches the objective through per-user coefficients each estimated on a few
hundred attempts, so the axes overfit at a capacity r tolerates. That argues for
r being larger; it does not argue for (64, 32) in particular, and capacity on the
axis path has just been measured actively harmful, so the number deserves a test
rather than an assumption.

INPUT DROPOUT. Production column-drops the AXIS map's input and leaves the
residual trunk's input raw — the reverse of pipeline.train_rating, where input
column dropout over the wide region is the headline anti-memorization finding.
r is the one head here that can fingerprint an individual problem, and it is the
one without that guard. An r that memorizes explains too much in sample and
leaves the axes a thinner residual to be defined against.

Both arms are scored the same way as everything else, so they sit on one scale
with the dropout and capacity results. The precision floor from float rounding
alone is ~0.03 on this trainer, so anything smaller is not a result.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_r_head --seeds 0 1
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

# (label, trunk_hidden, trunk_input_dropout)
ARMS = [
    ("shipped (64,32)", (64, 32), 0.0),
    ("(64,32) +in-drop .3", (64, 32), 0.3),
    ("(32,16)", (32, 16), 0.0),
    ("(16,)", (16,), 0.0),
    ("linear r head", (), 0.0),
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
    ti = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float32)
    lapse = load_client_lapse(0.0328)

    rbu = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in ti:
            rbu[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rbu.items() if len(a) >= MIN_USER_ATTEMPTS)
    fold_of = _stratified_folds(users, rbu)
    tu = {f: [u for u in users if fold_of[u] != f] for f in (0, 1)}
    ev = {f: [u for u in users if fold_of[u] == f] for f in (0, 1)}
    print(f"loaded in {time.time() - t0:.0f}s: {len(users)} users, seeds "
          f"{args.seeds}", flush=True)
    g_init = {f: eigen_directions(rbu, tu[f], F, ti, rank, params, lapse,
                                  args.k) for f in (0, 1)}
    print(f"  warm starts computed ({time.time() - t0:.0f}s)", flush=True)

    def rparams(hid):
        n, d = 0, F.shape[1]
        for w in hid:
            n += d * w + w
            d = w
        return n + d + 1

    res = defaultdict(dict)
    print(f"\n{'arm':<22}{'r params':>10}{'seed':>5}{'pooled':>9}"
          f"{'per-user':>10}{'f0':>8}{'f1':>8}", flush=True)
    for seed in args.seeds:
        for label, hid, tdrop in ARMS:
            tot_g, tot_n, pu, pf = 0.0, 0, [], []
            for f in (0, 1):
                g, _W, _r, _s = train_axes_staged(
                    rbu, tu[f], F, ti, args.k, g_init[f], device, seed=seed,
                    trunk_hidden=hid, trunk_input_dropout=tdrop)
                gain, n, gu, _nu = axes_ll_gain(rbu, ev[f], g, rank, params,
                                                ti, lapse)
                tot_g += gain; tot_n += n; pu.append(gu)
                pf.append(gain / max(n, 1) * 1000.0)
            p, u = tot_g / max(tot_n, 1) * 1000.0, float(np.mean(pu))
            res[label][seed] = (p, u)
            print(f"{label:<22}{rparams(hid):>10,}{seed:>5}{p:>9.3f}{u:>10.3f}"
                  f"{pf[0]:>8.3f}{pf[1]:>8.3f}", flush=True)

    base = ARMS[0][0]
    print(f"\n===== paired against '{base}', seeds {args.seeds} "
          f"(float-rounding floor ~0.03) =====")
    for label, hid, _t in ARMS:
        ps = np.array([res[label][s][0] for s in args.seeds])
        us = np.array([res[label][s][1] for s in args.seeds])
        bp = np.array([res[base][s][0] for s in args.seeds])
        bu = np.array([res[base][s][1] for s in args.seeds])
        print(f"  {label:<22}{rparams(hid):>10,}{ps.mean():>9.3f}{us.mean():>10.3f}"
              f"   d {(ps - bp).mean():+.3f} / {(us - bu).mean():+.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
