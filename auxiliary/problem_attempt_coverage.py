"""How many problems have enough first-time attempts from rank-estimable users?

Question: for the local D1 dump (attempts/database.sql), how many problems have
X or more first-time attempts, counting only attempts by users who have answered
enough to estimate a rank (>= Y correct AND >= Y incorrect first-time answers)?

Definitions
-----------
- First-time attempt: is_repeat = 0, further deduplicated to the earliest row
  per (user_id, problem) so a user contributes at most one attempt per problem
  (the dump has a few hundred (user,problem) pairs with multiple is_repeat=0
  rows; we keep the first).
- Qualified (rank-estimable) user: across their first-time attempts (both board
  sizes pooled, since rank is shared across sizes here), at least Y problems
  answered correctly AND at least Y answered incorrectly. Y guarantees signal on
  both tails — a user who only ever succeeds (or fails) can't be ranked.
- Coverage count for a problem: number of distinct qualified users with a
  first-time attempt at it.

The output is a matrix: rows are Y (the qualification threshold), columns are X
(the coverage cutoff), cells are the number of problems with >= X qualified
first-time attempts. Printed for all problems and split by board size.

Usage:
  python -m auxiliary.problem_attempt_coverage
  python -m auxiliary.problem_attempt_coverage --min 5 10 20 --x 1 2 3 5 10
"""

import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating_features import load_first_time_attempts  # noqa: E402


def qualified_users(df, y):
    """Set of user_ids with >= y first-time correct AND >= y first-time incorrect."""
    g = df.groupby("user_id")["success"].agg(correct="sum", total="count")
    g["incorrect"] = g["total"] - g["correct"]
    return set(g.index[(g["correct"] >= y) & (g["incorrect"] >= y)])


def coverage_counts(df, users):
    """Per-problem distinct-qualified-user count, as a DataFrame [problem, size, n]."""
    sub = df[df["user_id"].isin(users)]
    # Each (user, problem) is already unique, so a size-aware group count == distinct users.
    return (sub.groupby(["problem", "size"]).size()
               .reset_index(name="n"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min", type=int, nargs="+", default=[3, 5, 10, 20],
                        help="Y thresholds: min correct AND min incorrect "
                             "first-time answers to qualify a user (default 3 5 10 20).")
    parser.add_argument("--x", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50],
                        help="X coverage cutoffs: count problems with >= X "
                             "qualified attempts (default 1 2 3 5 10 20 50).")
    args = parser.parse_args()

    df = load_first_time_attempts()
    n_users = df["user_id"].nunique()
    n_problems = df["problem"].nunique()
    print(f"First-time attempts: {len(df)} | users: {n_users} | "
          f"distinct problems attempted: {n_problems}")
    sz = df.groupby("size")["problem"].nunique().to_dict()
    print(f"  distinct problems attempted by size: "
          f"{ {int(k): int(v) for k, v in sz.items()} }\n")

    xs = sorted(set(args.x))
    scopes = [("all", None)] + [(f"{s}x{s}", s) for s in sorted(df["size"].unique())]

    for y in sorted(set(args.min)):
        users = qualified_users(df, y)
        cov = coverage_counts(df, users)
        # How many qualified users, and how many first-time answers they account for.
        n_qual_attempts = int(df["user_id"].isin(users).sum())
        print(f"=== Y = {y} (>= {y} correct AND >= {y} incorrect first-time answers) ===")
        print(f"  qualified users: {len(users)}/{n_users}  "
              f"({100 * len(users) / n_users:.1f}%); "
              f"their first-time attempts: {n_qual_attempts}")
        header = "  " + f"{'scope':>7s}" + "".join(f"{'>=' + str(x):>9s}" for x in xs)
        print(header)
        for label, size in scopes:
            c = cov if size is None else cov[cov["size"] == size]
            counts = c["n"].to_numpy()
            row = "".join(f"{int((counts >= x).sum()):>9d}" for x in xs)
            print(f"  {label:>7s}{row}")
        print()


if __name__ == "__main__":
    main()
