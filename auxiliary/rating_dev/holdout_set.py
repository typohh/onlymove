"""Deterministic held-out PROBLEM set, shared by the rating tests and the validation bench.

The universe is every problem with >=1 first-time user attempt — the only problems that carry
a training/eval signal (unattempted problems get a feature-prior rating but no held-out
measurement). Given a (seed, fraction) it draws a held-out subset and a short content hash, so a
training run that EXCLUDES the set (auxiliary.rating_dev.train_rating_staged --holdout-problem-frac)
and a validation bench that EVALUATES on it (auxiliary.rating_dev.validate_ratings) can confirm, by
hash, that they assumed the IDENTICAL set. The hash also implicitly guards against attempts-DB drift
between the two (a changed universe → changed set → changed hash).

CLI:
  ./.venv/bin/python3 -m auxiliary.rating_dev.holdout_set --seed 0 --frac 0.1   # print count + hash
  ./.venv/bin/python3 -m auxiliary.rating_dev.holdout_set --seed 0 --frac 0.1 --out s.txt # also dump the list
"""

import argparse
import hashlib
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import load_first_time_attempts  # noqa: E402

SIZES = [9, 13]


def attempted_keys():
    """Sorted list of (size, tid) with >=1 first-time attempt — the held-out universe.

    Deterministic (a sorted set), so the seeded draw in held_out_keys is reproducible for a fixed
    attempts DB. Restricted to SIZES; mirrors the size/problem columns the trainer reads."""
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)]
    return sorted(set(zip(df["size"].astype(int).tolist(), df["problem"].astype(str).tolist())))


def hash_keys(keys):
    """Order-independent 16-hex content hash of a set of (size, tid)."""
    h = hashlib.sha256()
    for s, t in sorted(keys):
        h.update(f"{int(s)}:{t}\n".encode())
    return h.hexdigest()[:16]


def held_out_keys(seed, frac, universe=None):
    """Return (sorted held-out [(size, tid)], hash). Deterministic in (seed, frac, universe).

    Draws round(frac * |universe|) members without replacement via np.random.default_rng(seed) over
    the SORTED universe, so the selection is index-stable regardless of dict/iteration order."""
    keys = sorted(universe) if universe is not None else attempted_keys()
    n = len(keys)
    n_ho = int(round(frac * n))
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=n_ho, replace=False)
    ho = sorted(keys[i] for i in sel)
    return ho, hash_keys(ho)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frac", type=float, required=True, help="Fraction of attempted problems to hold out.")
    ap.add_argument("--out", default=None, help="If set, write the held-out 'size,tid' list here.")
    args = ap.parse_args()
    universe = attempted_keys()
    ho, h = held_out_keys(args.seed, args.frac, universe)
    by_size = {s: sum(1 for sz, _ in ho if sz == s) for s in SIZES}
    print(f"universe (attempted problems): {len(universe)} "
          f"({', '.join(f'{s}:{sum(1 for sz, _ in universe if sz == s)}' for s in SIZES)})")
    print(f"held out: {len(ho)} ({args.frac:g}) "
          f"({', '.join(f'{s}:{by_size[s]}' for s in SIZES)}) | seed {args.seed} | hash {h}")
    if args.out:
        with open(args.out, "w") as f:
            for s, t in ho:
                f.write(f"{s},{t}\n")
        print(f"wrote {len(ho)} held-out keys to {args.out}")


if __name__ == "__main__":
    main()
