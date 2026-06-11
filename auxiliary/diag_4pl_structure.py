"""Structural diagnostics for a ratings_v2-style 4PL artifact (docs/rating_4pl_structure_log.md).

Reads <dir>/<size>/tsumego_{ratings,quality,floor}_final.csv (rating / slope a_p / floor c_p)
and compares against bot-ladder ground truth from attempts/<size>/b18c384nbt-humanv0_<rank>.txt:

  - empirical bot floor  = min pass-rate over the whole humanv0 ladder (per problem),
  - empirical steepness  = 1 / (rank-width of the ladder curve's 25%→75% transition), computed
    only where the ladder actually transitions (range ≥ --min-range) — bot curves are flatter
    than human curves in absolute terms (the ~50% overdispersion), so use it for ORDERING
    (Spearman), never as an absolute slope target.

Prints per-rating-decile structure (clamp%, slope/floor percentiles, floor vs 0.70·bot-floor,
within-decile floor↔bot-floor and slope↔steepness Spearmans) plus the global correlations and
the motivating two-candidate hard case (bot floor 0.4–0.6, hardest decile).

Usage:
  ./.venv/bin/python3 -m auxiliary.diag_4pl_structure --dir ratings_v2_20k9d_erm [--dir B ...]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SIZES = [9, 13]


def rank_scalar(r):
    return -int(r[:-1]) if r.endswith("k") else int(r[:-1])


def load_ladder(size):
    """{tid: {rank_scalar: pass_rate}} over every humanv0 rung on disk for this size."""
    out = {}
    for fn in glob.glob(os.path.join(PROJECT_ROOT, "attempts", str(size),
                                     "b18c384nbt-humanv0_*.txt")):
        m = re.search(r"humanv0_([0-9]+[kd])\.txt$", fn)
        if not m:
            continue
        rs = rank_scalar(m.group(1))
        with open(fn) as fh:
            for line in fh:
                pp = line.split()
                if len(pp) != 2:
                    continue
                try:
                    out.setdefault(pp[0], {})[rs] = float(pp[1])
                except ValueError:
                    pass
    return out


def ladder_stats(curve, min_range):
    """(bot_floor, steepness | nan) from one problem's {rank: pass_rate} ladder. Steepness =
    1 / rank-width of the 25%→75% crossing of the min→max range (monotone-smoothed)."""
    ranks = np.array(sorted(curve))
    p = np.array([curve[r] for r in ranks], dtype=np.float64)
    floor = float(p.min())
    p_mono = np.maximum.accumulate(p)            # smooth out non-monotone jitter
    lo, hi = p_mono[0], p_mono[-1]
    if hi - lo < min_range or len(ranks) < 5:
        return floor, float("nan")
    q = (p_mono - lo) / (hi - lo)

    def cross(level):
        i = int(np.searchsorted(q, level))
        if i == 0:
            return float(ranks[0])
        if i >= len(ranks):
            return float(ranks[-1])
        f = (level - q[i - 1]) / max(q[i] - q[i - 1], 1e-9)
        return float(ranks[i - 1] + f * (ranks[i] - ranks[i - 1]))

    width = cross(0.75) - cross(0.25)
    return floor, (1.0 / width if width > 0.25 else float("nan"))


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return float("nan")
    xr = np.argsort(np.argsort(x[ok])); yr = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(xr, yr)[0, 1])


def load_artifact(d):
    rows = []
    for s in SIZES:
        sd = os.path.join(PROJECT_ROOT, d, str(s))
        if not os.path.isdir(sd):
            continue
        r = pd.read_csv(os.path.join(sd, "tsumego_ratings_final.csv"))
        a = pd.read_csv(os.path.join(sd, "tsumego_quality_final.csv"))
        c = pd.read_csv(os.path.join(sd, "tsumego_floor_final.csv"))
        df = r.merge(a, on="tsumego_id").merge(c, on="tsumego_id")
        df["size"] = s
        rows.append(df)
    df = pd.concat(rows, ignore_index=True).rename(columns={"quality": "slope"})
    sf = float("nan")
    pt = os.path.join(PROJECT_ROOT, d, "model_v2.pt")
    if os.path.exists(pt):
        try:
            import torch
            sf = float(torch.load(pt, map_location="cpu",
                                  weights_only=False)["args"].get("slope_floor", float("nan")))
        except Exception as e:                                   # diagnostics-only, stay soft
            print(f"  (could not read slope_floor from model_v2.pt: {e})")
    return df, sf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True,
                    help="Artifact dir (repeatable; relative to repo root).")
    ap.add_argument("--min-range", type=float, default=0.3,
                    help="Min ladder min→max range to trust a steepness estimate.")
    args = ap.parse_args()

    ladders = {s: load_ladder(s) for s in SIZES}
    print(f"bot ladder rungs: " + ", ".join(
        f"size {s}: {len({r for c in ladders[s].values() for r in c})} ranks, "
        f"{len(ladders[s])} problems" for s in SIZES))

    for d in args.dir:
        df, sf = load_artifact(d)
        bf, st = [], []
        for tid, s in zip(df["tsumego_id"], df["size"]):
            cur = ladders[s].get(tid)
            if cur is None or len(cur) < 5:
                bf.append(float("nan")); st.append(float("nan")); continue
            f_, k_ = ladder_stats(cur, args.min_range)
            bf.append(f_); st.append(k_)
        df["bot_floor"] = bf; df["bot_steep"] = st
        clamp_thr = sf * 1.02 if np.isfinite(sf) else df["slope"].min() * 1.02
        df["dec"] = pd.qcut(df["rating"], 10, labels=False, duplicates="drop")

        print(f"\n=== {d}  (n {len(df)}, slope_floor {sf}) ===")
        print(f"  global Spearman: rating↔floor {spear(df.rating, df.floor):+.2f} | "
              f"rating↔slope {spear(df.rating, df.slope):+.2f} | "
              f"floor↔bot_floor {spear(df.floor, df.bot_floor):+.2f} | "
              f"slope↔bot_steep {spear(df.slope, df.bot_steep):+.2f} "
              f"(steepness known for {int(np.isfinite(df.bot_steep).sum())} problems)")
        print(f"  {'dec':>3} {'rating_mid':>10} {'clamp%':>6} {'slope p10/50/90':>17} "
              f"{'floor p10/50/90':>17} {'0.7·botf':>8} {'ρ(c,botf)':>9} {'ρ(a,steep)':>10}")
        for g, sub in df.groupby("dec"):
            sl = np.percentile(sub.slope, [10, 50, 90])
            fl = np.percentile(sub.floor, [10, 50, 90])
            print(f"  {int(g):>3} {sub.rating.median():>10.2f} "
                  f"{100 * float((sub.slope <= clamp_thr).mean()):>5.1f}% "
                  f"{sl[0]:>5.2f}/{sl[1]:>4.2f}/{sl[2]:>4.2f} "
                  f"   {fl[0]:>5.2f}/{fl[1]:>4.2f}/{fl[2]:>4.2f} "
                  f"{0.7 * np.nanmean(sub.bot_floor):>8.2f} "
                  f"{spear(sub.floor, sub.bot_floor):>+9.2f} {spear(sub.slope, sub.bot_steep):>+10.2f}")
        hard = df[(df.dec == df.dec.max()) & df.bot_floor.between(0.4, 0.6)]
        if len(hard):
            print(f"  two-candidate hard case (bot floor .4–.6, hardest decile, n {len(hard)}): "
                  f"floor p10/50/90 {np.percentile(hard.floor, [10, 50, 90]).round(2)} | "
                  f"clamp% {100 * float((hard.slope <= clamp_thr).mean()):.1f}")


if __name__ == "__main__":
    main()
