#!/usr/bin/env python3
"""What is each candidate axis direction MADE OF, before anything is fitted?

The production warm start takes the leading eigenvectors of the centred,
per-user-normalised residual score moment and uses the top K in order. That
ordering is by how much between-user variation a direction carries, which is the
right default but says nothing about what the direction is ABOUT — so a
lower-ranked eigenvector that loads cleanly on one interpretable block is
invisible to it, and there is no way to ask for the phase direction or the
score-margin direction specifically.

This profiles them. Each eigenvector is a unit vector over the standardized
feature columns, so the sum of its squared weights within a block is exactly
that block's share of the direction — no fitting, no tail sampling, seconds to
compute. Feed the interesting indices to
auxiliary.rating_dev.eval_axis_epoch_curve --eigen-idx to fit them and see what
they are worth and what composes with them.

Two cautions on reading the shares. The ACTIVATION block is 128 of 216 columns,
so it starts with a large share by construction and a direction is only
activation-DOMINATED if its share clearly exceeds that baseline; the printout
gives the baseline share for comparison. And a block with few columns (phase has
two) reaching even a modest share is concentrated in a way a wide block never
has to be, so the same percentage does not mean the same thing across blocks —
the per-column figure is what makes them comparable.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_direction_profile [--n 24]
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
    MIN_USER_ATTEMPTS, _user_score_matrix, build_axis_features,
    load_axis_attempts,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402
from auxiliary.inspect_axis_inputs import column_names  # noqa: E402

BLOCK_ORDER = ["activation", "bot winrate", "move eval", "phase", "size",
               "rating model"]


def eigen_basis(rows_by_user, users, F, tid_index, rank, params, lapse):
    """The same construction eigen_directions uses, returning every eigenvector
    and eigenvalue rather than the leading k."""
    S, n = _user_score_matrix(rows_by_user, users, F, tid_index, rank, params,
                              lapse)
    S = S / n[:, None]
    S = S - S.mean(axis=0, keepdims=True)
    ev, vec = np.linalg.eigh(S.T @ S / S.shape[0])
    order = np.argsort(ev)[::-1]
    return ev[order], vec[:, order].T          # (n_feat,), (n_feat, n_feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="directions to profile")
    ap.add_argument("--top", type=int, default=3, help="named columns per row")
    args = ap.parse_args()

    feats, rank, params, act_dim = build_axis_features()
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
    names, blocks = column_names(F.shape[1], int(act_dim))
    idx_of = {b: [i for i, bb in enumerate(blocks) if bb == b] for b in BLOCK_ORDER}

    ev, V = eigen_basis(rbu, users, F, ti, rank, params, lapse)
    tot_ev = float(ev[ev > 0].sum())
    print(f"{len(users)} users, {F.shape[1]} feature columns")
    print("block sizes / share of a direction spread evenly over all columns:")
    for b in BLOCK_ORDER:
        print(f"    {b:<14}{len(idx_of[b]):>4} cols   even-spread share "
              f"{len(idx_of[b]) / F.shape[1]:5.1%}")
    print()
    hdr = "  dir  eigval  var%  " + "".join(f"{b[:9]:>10}" for b in BLOCK_ORDER)
    print(hdr + "   dominant named columns")
    for j in range(min(args.n, V.shape[0])):
        e = V[j]
        w2 = e ** 2
        shares = {b: float(w2[idx_of[b]].sum()) for b in BLOCK_ORDER}
        named = [i for i, b in enumerate(blocks) if b != "activation"]
        top = sorted(named, key=lambda i: -abs(e[i]))[:args.top]
        cells = "".join(f"{shares[b]:>9.1%} " for b in BLOCK_ORDER)
        cols = ", ".join(f"{names[i][:26]}{'+' if e[i] >= 0 else '-'}" for i in top)
        print(f"  {j:>4}{ev[j]:>8.3g}{ev[j] / tot_ev:>6.1%}  {cells}  {cols}")

    # Per-column concentration is what makes a narrow block comparable to a wide
    # one: phase reaching 10% across two columns is a far stronger statement than
    # the activation block reaching 40% across 128.
    print("\nmost CONCENTRATED directions per block (share / column, "
          "excluding activations):")
    for b in BLOCK_ORDER:
        if b == "activation" or not idx_of[b]:
            continue
        per = [(float((V[j][idx_of[b]] ** 2).sum()) / len(idx_of[b]), j)
               for j in range(min(args.n, V.shape[0]))]
        per.sort(reverse=True)
        best = ", ".join(f"dir {j} ({v * len(idx_of[b]):.0%} of it)"
                         for v, j in per[:3])
        print(f"    {b:<14}{best}")


if __name__ == "__main__":
    main()
