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


def all_users():
    """Sorted user ids over first-time attempts — the user-holdout universe.

    Deliberately NOT restricted to users who pass a trainer's MIN_ATTEMPTS
    filter: those filters differ between train_rating_staged and the joint MIRT,
    so drawing from a filtered set would make the two models hold out DIFFERENT
    users from the same seed. Drawing from everyone keeps the split identical
    across models; each trainer then applies its own filter to whatever remains,
    and the walk-forward simply scores whichever held-out users have the history
    to be scored."""
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)]
    return sorted(set(df["user_id"].astype(str).tolist()))


def all_rows():
    """Sorted (user_id, problem) pairs — the SAMPLE-holdout universe.

    The pair is a unique key because these are FIRST-TIME attempts: one row per
    user per problem, verified 0 duplicates. Holding out rows this way leaves
    each held-out problem with its other attempts still in training, which is
    the client's actual condition — parameters informed by other people's
    attempts, predicting this user's next one — and is the protocol a problem
    holdout (parameters from the feature prior alone) and a time-ordered
    walk-forward (parameters fitted on the very rows being scored) each miss
    from opposite sides."""
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)]
    return sorted(zip(df["user_id"].astype(str).tolist(),
                      df["problem"].astype(str).tolist()))


def _draw(keys, seed, frac, fmt):
    n_ho = int(round(frac * len(keys)))
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(keys), size=n_ho, replace=False)
    ho = sorted(keys[i] for i in sel)
    h = hashlib.sha256()
    for k in ho:
        h.update(fmt(k).encode())
    return ho, h.hexdigest()[:16]


def held_out_users(seed, frac, universe=None):
    """(sorted held-out user ids, hash). Whole users, for the WALK-FORWARD:
    parameters that never saw them, then theta fitted causally on their trailing
    window and the next block scored. That is production's walk-forward with its
    leak removed — utils.rating.slope_calibration scores rows whose problems'
    parameters were fitted ON those same rows, which rewards confidence and
    biases the slope scale upward."""
    return _draw(sorted(universe) if universe is not None else all_users(),
                 seed, frac, lambda u: f"{u}\n")


def held_out_rows(seed, frac, universe=None):
    """(sorted held-out (user, problem) pairs, hash) — the SAMPLE holdout."""
    return _draw(sorted(universe) if universe is not None else all_rows(),
                 seed, frac, lambda k: f"{k[0]}:{k[1]}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("problem", "user", "row"),
                    default="problem", help="what to hold out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frac", type=float, required=True, help="Fraction of attempted problems to hold out.")
    ap.add_argument("--out", default=None, help="If set, write the held-out 'size,tid' list here.")
    args = ap.parse_args()
    if args.kind == "user":
        universe = all_users()
        ho, h = held_out_users(args.seed, args.frac, universe)
        print(f"universe (users): {len(universe)}")
        print(f"held out: {len(ho)} ({args.frac:g}) | seed {args.seed} | hash {h}")
        return
    if args.kind == "row":
        universe = all_rows()
        ho, h = held_out_rows(args.seed, args.frac, universe)
        print(f"universe (first-time attempts): {len(universe)}")
        print(f"held out: {len(ho)} ({args.frac:g}) | seed {args.seed} | hash {h}")
        return
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
