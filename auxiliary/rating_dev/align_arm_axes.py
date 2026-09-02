#!/usr/bin/env python3
"""Align the axes of several panel arms to one reference arm and write each as
an inspectable dump, so the SAME slot on every page is the same direction.

A free fit's axis index means nothing across runs — the objective is invariant
to rotating the axis space, so axis 3 of one seed and axis 3 of another are
unrelated — and reading two free fits' pages side by side by index compares
arbitrary columns. This matches every arm's axes one-to-one to the reference's
(Hungarian on |correlation| of the per-problem loadings over ATTEMPTED problems,
the population the axes are identified on), permutes and sign-flips the columns
so slot k on every page is the reference's axis k, and titles each slot with
the match it rests on. The boards then decide whether the slot carries the same
motif — the number only orders what to look at first.

Each arm becomes a `--dump` npz for `auxiliary.rating_dev.inspect_axes`: `tid`,
`size`, `loading` (aligned), `rating` (the ARM'S OWN ratings, so rank buckets
and `--rank-max` read the fit that produced these axes) and `labels`.

    ./.venv/bin/python3 -m auxiliary.rating_dev.align_arm_axes \\
        --panel axis_space --reference untaught_i0 \\
        --arms untaught_i1 untaught_i2 untaught_i3 --out-dir <dir>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402
from utils.rating.model_constants import SIZES  # noqa: E402
from utils.rating.ratings_io import RATINGS_CSV, load_value_map  # noqa: E402

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")


def load_arm(panel, arm):
    """(loadings DataFrame keyed "<size>:<tid>", {tid: rating}) for one arm."""
    arm_dir = os.path.join(DEV_RUNS, panel, arm)
    path = os.path.join(arm_dir, "axes", "per_problem.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}")
    df = pd.read_csv(path, dtype={"tsumego_id": str})
    cols = [c for c in df.columns if c.startswith("embed_")]
    df.index = df["size"].astype(str) + ":" + df["tsumego_id"]
    ratings = {}
    for sz in ALL_BOARD_SIZES:
        ratings.update(load_value_map(
            os.path.join(arm_dir, "ratings", str(sz), RATINGS_CSV), "rating",
            missing_ok=True))
    return df, cols, ratings


def attempted_keys():
    att = load_first_time_attempts()
    att = att[att["size"].isin(SIZES)]
    return set((att["size"].astype(str) + ":" + att["problem"]).unique())


def match(ref_z, arm_z):
    """(perm, signs, |r|) so that arm_z[:, perm[k]] * signs[k] is the arm's
    best one-to-one match for the reference's axis k."""
    corr = ref_z.T @ arm_z / len(ref_z)
    rows, cols = linear_sum_assignment(-np.abs(corr))
    perm = np.empty(len(rows), dtype=int)
    signs = np.empty(len(rows))
    agree = np.empty(len(rows))
    for i, j in zip(rows, cols):
        perm[i] = j
        signs[i] = 1.0 if corr[i, j] >= 0 else -1.0
        agree[i] = abs(corr[i, j])
    return perm, signs, agree


def write_dump(path, df, cols, ratings, perm, signs, labels):
    tids = df["tsumego_id"].to_numpy(str)
    sizes = df["size"].to_numpy(int)
    M = df[cols].to_numpy(float)[:, perm] * signs[None, :]
    rating = np.array([ratings.get(t, np.nan) for t in tids])
    keep = np.isfinite(rating)
    dropped = int((~keep).sum())
    np.savez(path, tid=tids[keep], size=sizes[keep], loading=M[keep],
             rating=rating[keep], labels=np.array(labels))
    return int(keep.sum()), dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--reference", required=True,
                    help="the arm whose axis order every other arm is aligned to")
    ap.add_argument("--arms", nargs="+", required=True,
                    help="arms to align (the reference itself is written too, "
                         "unpermuted)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref_df, cols, ref_ratings = load_arm(args.panel, args.reference)
    keys = attempted_keys() & set(ref_df.index)
    arms = {}
    for arm in args.arms:
        arms[arm] = load_arm(args.panel, arm)
        keys &= set(arms[arm][0].index)
    keys = sorted(keys)
    print(f"matching over {len(keys)} attempted problems shared by every arm")

    def z(df):
        A = df.loc[keys, cols].to_numpy(float)
        return (A - A.mean(0)) / A.std(0)

    ref_z = z(ref_df)
    K = len(cols)
    ident = np.arange(K)
    # "A/B" captions both tails (inspect_axes names each end from its own half),
    # and both ends of a free axis are equally the thing being judged.
    labels = [f"slot {k}: reference {args.reference} axis {k}/"
              f"slot {k}: reference {args.reference} axis {k}" for k in range(K)]
    n, dropped = write_dump(os.path.join(args.out_dir, f"{args.reference}.npz"),
                            ref_df, cols, ref_ratings, ident, np.ones(K), labels)
    print(f"  {args.reference}: reference, {n} problems written"
          + (f", {dropped} without a rating dropped" if dropped else ""))

    print(f"\n{'arm':<20}" + "".join(f"{f'slot {k}':>10}" for k in range(K))
          + "      mean")
    for arm, (df, acols, ratings) in arms.items():
        if acols != cols:
            raise SystemExit(f"{arm} carries axes {acols}, the reference {cols}")
        perm, signs, agree = match(ref_z, z(df))
        labels = []
        for k in range(K):
            desc = (f"slot {k}: {arm} axis {perm[k]}"
                    f"{' flipped' if signs[k] < 0 else ''}, |r| {agree[k]:.2f} "
                    f"vs {args.reference} axis {k}")
            labels.append(f"{desc}/{desc}")
        n, dropped = write_dump(os.path.join(args.out_dir, f"{arm}.npz"),
                                df, cols, ratings, perm, signs, labels)
        cells = "".join(f"{f'{perm[k]}{chr(8722) if signs[k] < 0 else chr(43)} {agree[k]:.2f}':>10}"
                        for k in range(K))
        print(f"{arm:<20}{cells}{agree.mean():>10.2f}"
              + (f"   ({dropped} unrated dropped)" if dropped else ""))
    print("\ncells are `<own axis><sign> |r|`: which of the arm's axes fills the "
          "slot, whether it was flipped, and its agreement with the reference")


if __name__ == "__main__":
    main()
