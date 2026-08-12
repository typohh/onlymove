#!/usr/bin/env python3
"""Is the residual head earning its place, and how would we know?

The hypothesis is that `r` does almost nothing — that it re-derives structure the
rating model already captured, since the rating model's own outputs (rating,
slope, floor, rank) are four of the features `r` reads. Capacity arms support it
as far as they go: halving the trunk changes held-out gain by less than the
float-rounding floor.

But GAIN IS THE WRONG PLACE TO LOOK, and reading only that column would answer a
different question than the one asked. `r` exists to keep population difficulty
OUT of the axes — an axis that loads on "hard problems" is not a skill, and
fit_population_constants refuses to ship one. So r can be load-bearing precisely
by being invisible in gain: it absorbs the difficulty the axes would otherwise
take, and the fit scores the same either way while the axes come out clean rather
than contaminated.

This therefore reports two things per arm:

  GAIN            held-out walk-forward, the usual scale (floor ~0.03).
  DISPLACEMENT    mean(coef)^2 / mean(coef^2) over the per-user axis
                  coefficients, per axis. An axis carrying real between-user
                  variation has coefficients spread around zero and scores low.
                  An axis carrying population difficulty gives every user a
                  coefficient of the same sign, and it approaches 1. This is the
                  contamination measure, and it is what the ship-time zero-mean
                  check is testing for.

An arm that holds gain and holds displacement means r is genuinely redundant. An
arm that holds gain while displacement RISES means r was doing its job all along
and the gain column simply cannot see it.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_r_ablation --seeds 0 1
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
    axes_ll_gain, build_axis_features, eigen_directions, load_axis_attempts,
)
from auxiliary.rating_dev.train_axes_staged import train_axes_staged  # noqa: E402
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

# (label, trunk_hidden, use_r_head)
ARMS = [
    ("shipped r (64,32)", (64, 32), True),
    ("linear r", (), True),
    ("NO r head", (64, 32), False),
]


def displacement(rbu, ev_users, g, rank, params, ti, lapse):
    """Per-axis mean^2 / second-moment of the per-user coefficients — low means
    the axis separates users, near 1 means it points the same way for everyone,
    which is population difficulty wearing a skill's clothes."""
    _o, _s, _y, _u, c0 = _walk_forward(rbu, ev_users, g, rank, params, ti, lapse)
    _o2, _s2, _y2, _u2, coefs = _walk_forward(rbu, ev_users, g, rank, params, ti,
                                              lapse, penalty=_axis_prior(c0))
    return (coefs.mean(axis=0) ** 2) / (np.mean(coefs ** 2, axis=0) + 1e-12)


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

    res = defaultdict(dict)
    print(f"\n{'arm':<20}{'seed':>5}{'pooled':>9}{'per-user':>10}"
          f"   displacement per axis", flush=True)
    for seed in args.seeds:
        for label, hid, use_r in ARMS:
            tot_g, tot_n, pu, disp = 0.0, 0, [], []
            for f in (0, 1):
                g, _W, _r, _s = train_axes_staged(
                    rbu, tu[f], F, ti, args.k, g_init[f], device, seed=seed,
                    trunk_hidden=hid, use_r_head=use_r)
                gain, n, gu, _nu = axes_ll_gain(rbu, ev[f], g, rank, params,
                                                ti, lapse)
                tot_g += gain; tot_n += n; pu.append(gu)
                disp.append(displacement(rbu, ev[f], g, rank, params, ti, lapse))
            p, u = tot_g / max(tot_n, 1) * 1000.0, float(np.mean(pu))
            dmean = np.mean(np.stack(disp), axis=0)
            res[label][seed] = (p, u, dmean)
            print(f"{label:<20}{seed:>5}{p:>9.3f}{u:>10.3f}   "
                  + "  ".join(f"{d:.3f}" for d in dmean), flush=True)

    base = ARMS[0][0]
    print(f"\n===== paired against '{base}' (gain floor ~0.03) =====")
    for label, _h, _r in ARMS:
        ps = np.array([res[label][s][0] for s in args.seeds])
        us = np.array([res[label][s][1] for s in args.seeds])
        ds = np.stack([res[label][s][2] for s in args.seeds]).mean(axis=0)
        bp = np.array([res[base][s][0] for s in args.seeds])
        bu = np.array([res[base][s][1] for s in args.seeds])
        bd = np.stack([res[base][s][2] for s in args.seeds]).mean(axis=0)
        print(f"  {label:<20}{ps.mean():>9.3f}{us.mean():>10.3f}"
              f"   d {(ps - bp).mean():+.3f} / {(us - bu).mean():+.3f}"
              f"   displacement " + "  ".join(f"{d:.3f}" for d in ds)
              + "   (d " + " ".join(f"{x:+.3f}" for x in (ds - bd)) + ")")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
