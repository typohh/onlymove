"""Map the v2 combined rating (b_p, in ability-θ units) to EGF rank space.

The v2 model (pipeline.train_rating_v2) puts problem difficulty b_p and user ability θ on
ONE shared logit scale across both boards. So the rating→rank curve is a SINGLE curve:

  1. Aggregate the model's per-(user,2-month) θ into a per-user ability.
  2. Anchor θ → EGF rank using rank_surveys: EGF declarations map directly; other servers
     are converted to EGF via the existing server couplings (rank_map.json). Fit a robust
     linear θ→EGF map E(θ)=α+βθ (EGF rank is ~linear in logit ability).
  3. Emit rating_to_rank knots by inverting: at each EGF boundary g, rating = (g−α)/β and
     rank_index = g + 18.5 (the exact convention fit_rank_curves uses). Because the scale is
     shared, the SAME knots are written for both board keys ("9" and "13").

Writes to the BACKEND v2 location (does NOT touch the served pwa/tsumegos/):
  ratings_v2/rating_to_rank.json   per-board (identical) {ratings, rank_indices}
  ratings_v2/rank_map.json         server couplings (copied from the existing fit; server↔
                                   EGF is model-independent)

Deploying v2 = run package_for_pwa --ratings-root ratings_v2 (ships the zips+metadata into
the SAME pwa/tsumegos/), then copy these two files onto the served pwa/tsumegos/ paths.

Run pipeline.train_rating_v2 first (produces ratings_v2/user_theta.csv).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, latest_per_user_server, load_rank_surveys, scalar_to_rank,
)

SIZES = [9, 13]


def server_to_egf(server, scalar, rank_map):
    """Convert a (server, server-rank-scalar) declaration to EGF/internal-rank scalar."""
    if server == "EGF":
        return scalar
    knots = rank_map.get(server)
    if not knots:
        return None
    return float(np.interp(scalar, knots["server"], knots["our"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-dir", default=os.path.join(PROJECT_ROOT, "ratings_v2"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--rank-map", default=os.path.join(
        PROJECT_ROOT, "pwa", "tsumegos", "rank_map.json"),
        help="Existing server↔EGF couplings (model-independent); copied into v2-dir.")
    ap.add_argument("--g-lo", type=int, default=-19, help="Lowest EGF boundary (≈20k).")
    ap.add_argument("--g-hi", type=int, default=10, help="Highest EGF boundary (≈10d).")
    args = ap.parse_args()

    theta_df = pd.read_csv(os.path.join(args.v2_dir, "user_theta.csv"))
    # profile = user8_YYYY-MM → per-user attempt-weighted mean ability
    theta_df["user8"] = theta_df["profile"].str.rsplit("_", n=1).str[0]
    theta_df["w"] = theta_df["n_attempts"].clip(lower=1)
    theta_df["wt"] = theta_df["theta"] * theta_df["w"]
    per_user = theta_df.groupby("user8").agg(wt=("wt", "sum"), w=("w", "sum"))
    user_theta = (per_user["wt"] / per_user["w"]).to_dict()

    rank_map = json.load(open(args.rank_map))
    surveys = load_rank_surveys(args.sql)
    if surveys is None or len(surveys) == 0:
        sys.exit("No rank_surveys rows found — cannot anchor θ→EGF.")
    surveys = latest_per_user_server(surveys)

    # anchor points (θ_user, EGF scalar)
    th, eg = [], []
    for _, r in surveys.iterrows():
        u8 = str(r["user_id"])[:8]
        if u8 not in user_theta or r.get("rank_scalar") is None or pd.isna(r["rank_scalar"]):
            continue
        g = server_to_egf(r["server"], float(r["rank_scalar"]), rank_map)
        if g is None:
            continue
        th.append(user_theta[u8])
        eg.append(g)
    th, eg = np.asarray(th, float), np.asarray(eg, float)
    if len(th) < 5:
        sys.exit(f"Only {len(th)} survey anchors with a matching θ — too few to fit.")

    # robust linear E(θ)=α+βθ: one trim pass on residuals (drops gross outliers)
    beta, alpha = np.polyfit(th, eg, 1)
    resid = eg - (alpha + beta * th)
    keep = np.abs(resid) <= 2.5 * np.std(resid)
    beta, alpha = np.polyfit(th[keep], eg[keep], 1)
    pred = alpha + beta * th[keep]
    ss = 1.0 - np.sum((eg[keep] - pred) ** 2) / np.sum((eg[keep] - eg[keep].mean()) ** 2)
    print(f"θ→EGF anchors: {len(th)} ({int(keep.sum())} after trim) | "
          f"EGF = {alpha:+.3f} {beta:+.3f}·θ | R² {ss:.3f}")
    if beta <= 0:
        sys.exit("Non-positive θ→EGF slope — anchoring failed (check surveys).")

    # invert to rating-at-EGF-boundary knots; shared across boards
    gs = list(range(args.g_lo, args.g_hi + 1))
    ratings = [float((g - alpha) / beta) for g in gs]
    rank_indices = [float(g + 18.5) for g in gs]
    curve = {"ratings": ratings, "rank_indices": rank_indices}
    rr = {str(s): curve for s in SIZES}      # one shared curve, both board keys

    os.makedirs(args.v2_dir, exist_ok=True)
    with open(os.path.join(args.v2_dir, "rating_to_rank.json"), "w") as f:
        json.dump(rr, f, indent=2)
    with open(os.path.join(args.v2_dir, "rank_map.json"), "w") as f:
        json.dump(rank_map, f, indent=2)
    print(f"rating range mapped: {scalar_to_rank(args.g_lo)}→rating {ratings[0]:.2f} ... "
          f"{scalar_to_rank(args.g_hi)}→rating {ratings[-1]:.2f}")
    print(f"wrote {args.v2_dir}/rating_to_rank.json (shared 9≡13) + rank_map.json")


if __name__ == "__main__":
    main()
