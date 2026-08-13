#!/usr/bin/env python3
"""Do the axes need history before they pay? Breaks the held-out-user walk down
by how many attempts PRECEDE each scored one.

The standing partition — seen vs never-trained — describes the PROBLEM: whether
anyone's attempts on it reached training. It says nothing about how much is known
about the USER, and the user side is where the axes have to earn their place: a
held-out user's ability and axis position are BOTH estimated from their own
prefix, from nothing at attempt 1.

They are not estimated equally well, and that is the reason to look. Ability is a
single number under an exact grid posterior and converges fast. An axis position
is K numbers under a Laplace posterior, each informed only by the component of a
problem's loading along its own axis, so it should need substantially more
history — and if the axes only pay for long-history users, "the axes are worth
+2 mnat" is an average over a population most of whom get nothing.

Both models are scored on identical rows with identical bins, so the comparison
per bin is paired even though the bins differ in size.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_by_history \\
      --dumps /tmp/gate/trunkcoupl.npz /tmp/q2/base_f1.npz \\
      --prod-dirs /tmp/prod_uh_s0 /tmp/prod_uh_s1 \\
      --prod-axes /tmp/axes_uh_s0.npz /tmp/axes_uh_s1.npz --seeds 0 1
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.rating_dev.eval_user_holdout import (  # noqa: E402
    PREFIX_BINS, _load, forward_walk)
from auxiliary.rating_dev.holdout_set import held_out_users  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402

ROWS = ["all"] + [lbl for _lo, _hi, lbl in PREFIX_BINS]


def score(prod_dir, dump, axes_npz, seed, frac, df_cache):
    hu_list, hu_hash = held_out_users(seed, frac)
    hu = set(hu_list)
    rate, slope, floor, lam, load = _load(prod_dir, dump, axes_npz)
    df = df_cache
    is_held = df["user_id"].astype(str).isin(hu)
    seen = set(df.loc[~is_held, "problem"].astype(str))
    seqs = defaultdict(list)
    for u, t, ok in zip(df.loc[is_held, "user_id"].astype(str),
                        df.loc[is_held, "problem"].astype(str),
                        df.loc[is_held, "success"]):
        if t in rate:
            seqs[u].append((t, int(ok)))
    k = len(next(iter(load.values()))) if load else 0
    base = forward_walk(rate, slope, floor, lam, seqs, seen)
    ax = (forward_walk(rate, slope, floor, lam, seqs, seen, loadings=load,
                       k_axes=k) if k else None)
    return hu_hash, base, ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="*", default=[])
    ap.add_argument("--prod-dirs", nargs="*", default=[])
    ap.add_argument("--prod-axes", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--frac", type=float, default=0.15)
    args = ap.parse_args()

    df = load_first_time_attempts()
    df = df[df["size"].isin([9, 13])].sort_values("created_at")

    models = []
    for i, sd in enumerate(args.seeds):
        if i < len(args.dumps):
            models.append((f"MIRT f{sd}", None, args.dumps[i], None, sd))
        if i < len(args.prod_dirs):
            ax = args.prod_axes[i] if i < len(args.prod_axes) else None
            models.append((f"prod f{sd}", args.prod_dirs[i], None, ax, sd))

    # POOL the folds rather than averaging their logloss: they hold different
    # users and different bin counts, so the pooled figure is total loss over
    # total attempts, sum(logloss*n) / sum(n).
    pooled = {}
    for name, pd_, dp, ax, sd in models:
        h, base, wax = score(pd_, dp, ax, sd, args.frac, df)
        fam = name.split()[0]
        acc = pooled.setdefault(fam, {})
        for r in ROWS:
            for key, d in (("base", base), ("axis", wax)):
                if d and r in d:
                    t, n, e, c = acc.setdefault((r, key), [0.0, 0, 0.0, 0.0])
                    acc[(r, key)] = [t + d[r]["logloss"] * d[r]["n"],
                                     n + d[r]["n"],
                                     e + d[r]["mae"] * d[r]["n"],
                                     c + d[r]["acc"] * d[r]["n"]]
        print(f"scored {name} (hash {h})", flush=True)

    def val(fam, r, key, field=0):
        e = pooled[fam].get((r, key))
        return (e[field] / e[1], e[1]) if e and e[1] else (None, 0)

    fams = list(pooled)
    w = 14
    print("\nBOTH FOLDS POOLED (total loss / total attempts; the folds hold "
          "different users, so their logloss cannot simply be averaged)\n")
    print(" " * 15 + "".join(r.center(w) for r in ROWS))
    print(" " * 15 + "".join(f"n={val(fams[0], r, 'base')[1]}".center(w)
                             for r in ROWS))
    for label, key in (("rating", "base"), ("rating+axis", "axis")):
        print(f"  {label}")
        for fam in fams:
            cells = "".join(
                (f"{val(fam, r, key)[0]:.4f}" if val(fam, r, key)[0] is not None
                 else "—").center(w) for r in ROWS)
            print(f"    {fam:11}{cells}")
    print("  axis adds (mnat, higher better)")
    for fam in fams:
        cells = ""
        for r in ROWS:
            b, _ = val(fam, r, "base")
            a, _ = val(fam, r, "axis")
            cells += (f"{(b - a) * 1000:+.2f}" if (b is not None and a is not None)
                      else "—").center(w)
        print(f"    {fam:11}{cells}")

    # Probability space: how far the predicted chance of success sits from what
    # happened, in percentage points. Same comparison, readable units.
    print("\n  mean |predicted P(correct) - outcome|, percentage points")
    for label, key in (("  rating", "base"), ("  +axis ", "axis")):
        for fam in fams:
            cells = "".join(
                (f"{val(fam, r, key, 2)[0] * 100:.2f}"
                 if val(fam, r, key, 2)[0] is not None else "—").center(w)
                for r in ROWS)
            print(f"  {label} {fam:9}{cells}")
    print("  axis effect (percentage points, negative = closer to truth)")
    for fam in fams:
        cells = ""
        for r in ROWS:
            b, _ = val(fam, r, "base", 2)
            a, _ = val(fam, r, "axis", 2)
            cells += (f"{(a - b) * 100:+.3f}" if (b is not None and a is not None)
                      else "—").center(w)
        print(f"    {fam:11}{cells}")
    print("\n  accuracy at the 0.5 threshold, %")
    for label, key in (("  rating", "base"), ("  +axis ", "axis")):
        for fam in fams:
            cells = "".join(
                (f"{val(fam, r, key, 3)[0] * 100:.2f}"
                 if val(fam, r, key, 3)[0] is not None else "—").center(w)
                for r in ROWS)
            print(f"  {label} {fam:9}{cells}")


if __name__ == "__main__":
    main()
