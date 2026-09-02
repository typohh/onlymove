#!/usr/bin/env python3
"""What the axis channel is worth BY HISTORY LENGTH, for panel arms and the
shipped fit alike: the `eval_user_holdout` forward walk — ability and axis
position both estimated causally from the user's own prefix — over a drawn set
of accounts, rating-only against rating+axis, binned by how many attempts
precede the scored one.

This is the reading `methodology.md` §1.14 rests its "the axes pay for
long-history users and nobody else" claim on, pointed at ARM records so two
ways of fitting the axes — with the rules taught, with the pull off, with a
mutual penalty — can be compared on the population that has enough history for
a position to be detectable at all. A whole-fit held-out level cannot make that
comparison: the axis increment on held-out PROBLEMS is near zero whichever way
the axes were fitted, because a held-out problem's loadings come from the trunk
alone and most scored users have short histories.

THE ARMS HELD OUT PROBLEMS, NOT USERS, so the drawn accounts' attempts did reach
every arm's per-problem parameters. That is mildly optimistic, identically for
every arm scored on one draw, so a contrast BETWEEN arms is paired and quotable
while an arm's absolute figure is not the clean user-holdout number the
methodology quotes for the shipped fit.

Each arm is read as a `--dump`: its own ratings, slopes, floors and lapse from
its `ratings/` directory and its loadings from `axes/per_problem.csv`, so the
walk scores the fit that produced the axes and nothing else.

    ./.venv/bin/python3 -m auxiliary.rating_dev.panel_axis_by_history \\
        --arms axis_space/untaught_i0 axis_space/taught_i0 shipped --seeds 0 1
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.rating_dev.axis_by_history import score  # noqa: E402
from auxiliary.rating_dev.eval_user_holdout import PREFIX_BINS  # noqa: E402
from auxiliary.rating_dev.validate_ratings import load_ratings  # noqa: E402
from utils.rating.features import load_first_time_attempts  # noqa: E402

DEV_RUNS = os.path.join(PROJECT_ROOT, "ablations", "dev_runs")
BINS = ["all"] + [lbl for _lo, _hi, lbl in PREFIX_BINS]


def arm_dirs(arm):
    """`shipped` -> the production fit; `<panel>/<arm>` -> that arm's record."""
    if arm == "shipped":
        return os.path.join(PROJECT_ROOT, "ratings"), os.path.join(PROJECT_ROOT, "axes")
    if "/" not in arm:
        raise SystemExit(f"{arm!r}: name an arm as <panel>/<arm>, or `shipped`")
    panel, name = arm.split("/", 1)
    base = os.path.join(DEV_RUNS, panel, name)
    if not os.path.isdir(base):
        raise SystemExit(f"no arm directory {base}")
    return os.path.join(base, "ratings"), os.path.join(base, "axes")


def arm_dump(arm, cache_dir):
    """The arm as an eval_user_holdout `--dump` npz, built once and cached."""
    path = os.path.join(cache_dir, f"dump_{arm.replace('/', '__')}.npz")
    if os.path.exists(path):
        return path
    rdir, adir = arm_dirs(arm)
    rate, slope, floor, lam, _ = load_ratings(rdir)
    pp = pd.read_csv(os.path.join(adir, "per_problem.csv"), dtype={"tsumego_id": str})
    cols = [c for c in pp.columns if c.startswith("embed_")]
    load = dict(zip(pp["tsumego_id"], pp[cols].to_numpy(float)))
    tids = [t for t in rate if t in slope and t in floor and t in load]
    if not tids:
        raise SystemExit(f"{arm}: no problem carries a rating, slope, floor and loading")
    np.savez(path, tid=np.array(tids), rating=np.array([rate[t] for t in tids]),
             slope=np.array([slope[t] for t in tids]),
             floor=np.array([floor[t] for t in tids]), lapse=np.array(lam),
             loading=np.stack([load[t] for t in tids]))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", required=True,
                    help="<panel>/<arm> names, or `shipped`")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="account-draw seeds (held_out_users)")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--cache-dir", default=os.path.join(DEV_RUNS, "_walk_dumps"),
                    help="where the per-arm dump npz files are kept")
    ap.add_argument("--json", default=None, help="write every bin's figures here")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    df = load_first_time_attempts()
    df = df[df["size"].isin([9, 13])].sort_values("created_at")
    results = {}
    for arm in args.arms:
        dump = arm_dump(arm, args.cache_dir)
        for sd in args.seeds:
            h, base, ax = score(None, dump, None, sd, args.frac, df)
            results[(arm, sd)] = (h, base, ax)
            print(f"scored {arm} draw {sd} ({h})", flush=True)

    print("\naxis adds, millinats (rating-only minus rating+axis one-step-ahead "
          "logloss; higher = the axes help), by attempts preceding the scored one")
    print(f"{'arm':<28}{'draw':>5}" + "".join(f"{b:>14}" for b in BINS))
    for (arm, sd), (h, base, ax) in results.items():
        cells = "".join(
            f"{1e3 * (base[b]['logloss'] - ax[b]['logloss']):>+14.2f}" if b in base
            else f"{'—':>14}" for b in BINS)
        print(f"{arm:<28}{sd:>5}{cells}")
    for sd in args.seeds:
        first = next(v[1] for (a, s), v in results.items() if s == sd)
        print(f"{f'n (attempts), draw {sd}':<33}"
              + "".join(f"{first[b]['n']:>14}" for b in BINS if b in first))
    if args.json:
        json.dump({f"{a}|{s}": {b: {"n": base[b]["n"], "base": base[b]["logloss"],
                                    "axis": ax[b]["logloss"]}
                                for b in BINS if b in base}
                   for (a, s), (h, base, ax) in results.items()},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
