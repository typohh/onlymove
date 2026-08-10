#!/usr/bin/env python3
"""Are the axes tailored to a handful of heavy users?

The production warm start takes the leading eigenvectors of the CENTRED
between-user residual score moment (residual_axes.eigen_directions), so "which
users determine an axis" is not an inference — each user's contribution to
eigenvector k is exactly (s_u · e_k)^2, and those contributions sum to the
eigenvalue. This reports how that sum is distributed.

The headline is the PARTICIPATION RATIO, (Σ c)^2 / Σ c^2, which is the effective
number of users behind a direction: N if all contribute alike, 1 if a single
account supplies the whole thing. It is reported beside the raw top-k shares
because the two fail differently — a participation ratio can look healthy while
one account still holds a fifth of the mass, and a top-1 share can look modest
while the rest is held by five more.

Two concentrations are distinguished, and conflating them is the trap:

  ATTEMPT concentration — how lopsided the attempt distribution is. This is a
    property of the userbase and is not by itself evidence of anything wrong.
  DIRECTION concentration — how lopsided each axis's support is. This is the
    one that matters, because an axis supported by three accounts describes
    those accounts rather than a skill, and it will be shipped to everyone.

The two need not agree: a heavy user contributes many attempts to the score sum
but their residual can still point in an ordinary direction, in which case they
inflate the attempt count without dominating any axis. That is exactly the case
this script exists to distinguish from the alternative.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_user_concentration [--k 8]
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import (  # noqa: E402
    EIGEN_WEIGHTING, EVAL_BLOCK, MIN_USER_ATTEMPTS, WARMUP,
    _user_score_matrix, build_axis_features, load_axis_attempts,
    normalize_scores,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402


def participation(c):
    """Effective number of contributors: (Σc)^2 / Σc^2."""
    s = float(c.sum())
    q = float((c ** 2).sum())
    return (s * s / q) if q > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fitted", action="store_true",
                    help="also measure the SHIPPED axis directions (axes.npz W) "
                         "rather than only the eigenvectors they warm-start "
                         "from. Training moves them, so broad support at "
                         "initialisation does not by itself establish broad "
                         "support in what ships")
    args = ap.parse_args()

    feats, _s, rank, params, _ad = build_axis_features()
    tids = sorted(feats)
    ti = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float64)
    lapse = load_client_lapse(0.0328)

    rbu = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in ti:
            rbu[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rbu.items() if len(a) >= MIN_USER_ATTEMPTS)
    scored = [u for u in users if len(rbu[u]) >= WARMUP + EVAL_BLOCK]

    n_att = np.array([len(rbu[u]) for u in users], dtype=float)
    print(f"{len(users)} users in the axis fit, {int(n_att.sum())} attempts; "
          f"{len(scored)} of them scorable (>= {WARMUP + EVAL_BLOCK})")
    for name, sel in (("all fitted", users), ("scored only", scored)):
        a = np.array([len(rbu[u]) for u in sel], dtype=float)
        o = np.sort(a)[::-1]
        cum = np.cumsum(o) / o.sum()
        half = int(np.searchsorted(cum, 0.5) + 1)
        print(f"  ATTEMPT concentration [{name}]: top-1 {100*o[0]/o.sum():.1f}%  "
              f"top-3 {100*cum[2]:.1f}%  top-10 {100*cum[9]:.1f}%  "
              f"users holding half the attempts: {half}  "
              f"participation {participation(a):.1f}/{len(sel)}")

    # scores built by the SAME helper the production warm start uses, so this
    # diagnostic cannot drift from what actually ships
    S_raw, n_by_user = _user_score_matrix(rbu, users, F, ti, rank, params, lapse)

    rng = np.random.default_rng(0)
    R = rng.normal(size=(F.shape[1], 32))
    R /= np.linalg.norm(R, axis=0, keepdims=True)

    for weighting in ("sum", "rootn", "user"):
        S = normalize_scores(S_raw, n_by_user, weighting)
        S = S - S.mean(axis=0, keepdims=True)
        ev, vec = np.linalg.eigh(S.T @ S / S.shape[0])
        order = np.argsort(ev)[::-1]
        base = [participation((S @ R[:, j]) ** 2) for j in range(R.shape[1])]
        print(f"\nDIRECTION concentration, weighting={weighting!r} "
              f"(random-direction baseline effN median {np.median(base):.1f})")
        print("  axis   effN   top-1%  top-3%  top-5%  top-10%   corr(contrib, attempts)")
        for k in range(min(args.k, len(order))):
            e = vec[:, order[k]]
            c = (S @ e) ** 2
            o = np.sort(c)[::-1]
            cum = np.cumsum(o) / o.sum()
            r = np.corrcoef(c, n_by_user)[0, 1]
            print(f"  {k:4d}  {participation(c):6.1f}  "
                  f"{100*cum[0]:6.1f}  {100*cum[2]:6.1f}  {100*cum[4]:6.1f}  "
                  f"{100*cum[9]:7.1f}   {r:+.3f}")


    if args.fitted:
        path = os.path.join(PROJECT_ROOT, "axes", "axes.npz")
        if not os.path.exists(path):
            print("\n(--fitted: no axes/axes.npz)")
            return
        d = np.load(path, allow_pickle=False)
        W = d["W"]                                   # (K, n_feat), same space as S
        S = normalize_scores(S_raw, n_by_user, EIGEN_WEIGHTING)
        S = S - S.mean(axis=0, keepdims=True)
        base = [participation((S @ R[:, j]) ** 2) for j in range(R.shape[1])]
        print(f"\nSHIPPED axes (axes.npz, K={W.shape[0]}, weighting="
              f"{EIGEN_WEIGHTING!r}; random baseline effN "
              f"{np.median(base):.1f})")
        print("  axis   effN   top-1%  top-3%  top-5%  top-10%   corr(contrib, attempts)")
        for k in range(W.shape[0]):
            w = W[k] / (np.linalg.norm(W[k]) + 1e-12)
            c = (S @ w) ** 2
            o = np.sort(c)[::-1]
            cum = np.cumsum(o) / o.sum()
            r = np.corrcoef(c, n_by_user)[0, 1]
            print(f"  {k:4d}  {participation(c):6.1f}  "
                  f"{100*cum[0]:6.1f}  {100*cum[2]:6.1f}  {100*cum[4]:6.1f}  "
                  f"{100*cum[9]:7.1f}   {r:+.3f}")


if __name__ == "__main__":
    main()
