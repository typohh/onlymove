"""Per-account rank error for STRONG (EGF 4d+) survey users, across rating configs.

`estimate_user_rank.py` answers "what rank for user X?". This generalises it to the
whole EGF cohort and compares several rating backends side by side, with the focus on
the strong tail where the 4PL rewrite over-rates (see the project memory).

For each `label=dir` config it reproduces EXACTLY what `pipeline.map_rating_to_egf`
does, end to end, so the comparison includes the rating→rank mapping (NOT a stale
shipped curve):
  1. load that config's per-problem ratings/slopes from <dir>/<size>/*.csv,
  2. fit every survey user's 80%-clear v2-rating (`fit_user_rating` @ CALIBRATION_P,
     pooled across boards) — the same estimator the client + EGF anchor use,
  3. JOINTLY refit the shared rating→rank curve + per-server couplings over all
     declarations (`fit_joint_shared`, map's exact defaults),
  4. map each EGF declaration's rating through that fresh curve → predicted rank,
  5. residual = predicted − declared, reported overall AND for the declared-≥4d cohort.

So each config is scored against ITS OWN freshly-fit mapping — re-run a model into a
fresh --out-dir and point a config at it; no separate map step needed.

Reads the LOCAL DB (`attempts/database.sql` via utils.rank_surveys) + each config's
`ratings_v2*` dir. No network, no GPU.

Usage:
  ./.venv/bin/python3 -m auxiliary.eval_strong_user_error \
      legacy=ratings_v2_legacy 5k=ratings_v2_5k no-anchor=ratings_v2_noanchor
  # --min-declared 4 = 4d (default); --min-declared 3 widens to 3d+.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, latest_per_user_server, load_attempts_for_users,
    load_rank_surveys,
)
from pipeline.map_rating_to_egf import (  # noqa: E402
    SIZES, fit_joint_shared, load_v2_ratings, load_v2_slopes,
)
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec,
)

# map_rating_to_egf argparse defaults — keep in lockstep so this scores the
# SAME curve the pipeline would ship for each config.
MAP_DEFAULTS = dict(egf_lo=-20, egf_hi=10, lam=5.0, rank_padding=2,
                    extend_to=9, min_server_samples=3)
MIN_ATTEMPTS = 20


def scalar_label(s):
    """Rank scalar → kyu/dan label (1k = 0, 1d = 1). Rounds for display."""
    k = int(round(s))
    return f"{1 - k}k" if k <= 0 else f"{k}d"


def user_ratings_for_config(v2_dir, surveys, attempts):
    """{user_pref: 80%-clear v2-rating} for every survey user with ≥MIN_ATTEMPTS
    pooled attempts, using this config's per-problem ratings/slopes."""
    merged_ratings, merged_slopes = {}, {}
    for s in SIZES:
        merged_ratings.update(load_v2_ratings(v2_dir, s))
        merged_slopes.update(load_v2_slopes(v2_dir, s))
    if not merged_ratings:
        return None
    out = {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < MIN_ATTEMPTS:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=MIN_ATTEMPTS,
                               slopes=merged_slopes)
        if R is not None and n >= MIN_ATTEMPTS:
            out[pref] = R
    return out


def evaluate(v2_dir, surveys, attempts):
    """Reproduce map_rating_to_egf and return per-EGF-declaration predicted vs
    declared as a DataFrame, plus the fitted curve span."""
    user_rating = user_ratings_for_config(v2_dir, surveys, attempts)
    if user_rating is None:
        return None, None

    decls = []  # (server, R, declared_scalar)
    for _, row in surveys.iterrows():
        pref = str(row["user_id"])[:8]
        g = row.get("rank_scalar")
        if pref not in user_rating or g is None or pd.isna(g):
            continue
        decls.append((row["server"], pref, user_rating[pref], float(g)))

    rating_knots, egf_thr_ranks, _ = fit_joint_shared(
        [(srv, R, g) for srv, _, R, g in decls], **MAP_DEFAULTS)
    ys = np.array([g - 0.5 for g in egf_thr_ranks], dtype=float)

    egf = [(pref, R, g) for srv, pref, R, g in decls if srv == "EGF"]
    if not egf:
        return None, None
    R_arr = np.array([R for _, R, _ in egf])
    g_arr = np.array([g for _, _, g in egf])
    pred = interp_extrap_vec(R_arr, rating_knots, ys)
    df = pd.DataFrame({
        "user": [p for p, _, _ in egf],
        "rating": R_arr,
        "declared": g_arr,
        "predicted": pred,
        "residual": pred - g_arr,
    }).sort_values("declared", ascending=False).reset_index(drop=True)
    span = (float(rating_knots[0]), float(rating_knots[-1]),
            float(ys[0]), float(ys[-1]))
    return df, span


def summarize(res):
    """(RMSE, bias, mean|res|, n) over a residual array."""
    r = res.to_numpy() if hasattr(res, "to_numpy") else np.asarray(res)
    if len(r) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.sqrt(np.mean(r ** 2))), float(r.mean()),
            float(np.mean(np.abs(r))), len(r))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+",
                    help="label=dir pairs, e.g. legacy=ratings_v2_legacy")
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--min-declared", type=float, default=4.0,
                    help="strong-cohort threshold in rank scalar (4 = 4d).")
    ap.add_argument("--no-per-user", action="store_true",
                    help="skip the per-user 4d+ table")
    args = ap.parse_args()

    configs = []
    for c in args.configs:
        if "=" not in c:
            sys.exit(f"config '{c}' must be label=dir")
        label, d = c.split("=", 1)
        d = d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)
        if not os.path.exists(os.path.join(d, "9")) and \
           not os.path.exists(os.path.join(d, "13")):
            sys.exit(f"{d}: no <size>/ subdir with ratings CSVs")
        configs.append((label, d))

    surveys = latest_per_user_server(load_rank_surveys(args.sql))
    attempts = load_attempts_for_users(args.sql,
                                       set(surveys["user_id"].unique()))
    thr = args.min_declared
    thr_label = scalar_label(thr)
    print(f"Strong-cohort threshold: declared ≥ {thr_label} (scalar ≥ {thr:g})\n")

    table = []
    per_config_df = {}
    for label, d in configs:
        df, span = evaluate(d, surveys, attempts)
        if df is None:
            print(f"=== {label} ({d}) ===  no usable EGF declarations\n")
            continue
        per_config_df[label] = df
        strong = df[df["declared"] >= thr]
        all_s = summarize(df["residual"])
        str_s = summarize(strong["residual"])
        print(f"=== {label} ({d}) ===")
        print(f"  curve span: rating {span[0]:.2f}..{span[1]:.2f}  "
              f"(scalar {span[2]:+.1f}..{span[3]:+.1f})")
        print(f"  all EGF   (n={all_s[3]:>3}): "
              f"RMSE {all_s[0]:.2f}  bias {all_s[1]:+.2f}  MAE {all_s[2]:.2f}")
        print(f"  {thr_label}+ cohort (n={str_s[3]:>3}): "
              f"RMSE {str_s[0]:.2f}  bias {str_s[1]:+.2f}  MAE {str_s[2]:.2f}")
        if not args.no_per_user and len(strong):
            for _, r in strong.iterrows():
                print(f"      {r['user']:<9} declared {scalar_label(r['declared']):>3}"
                      f"  rating {r['rating']:>6.2f}  pred {r['predicted']:+5.2f} "
                      f"({scalar_label(r['predicted']):>3})  res {r['residual']:+5.2f}")
        print()
        table.append((label, all_s, str_s))

    if len(table) > 1:
        print(f"=== comparison ({thr_label}+ cohort) ===")
        print(f"{'config':<14}{'all RMSE':>10}{'all bias':>10}"
              f"{thr_label+' RMSE':>10}{thr_label+' bias':>10}{thr_label+' MAE':>10}")
        for label, all_s, str_s in table:
            print(f"{label:<14}{all_s[0]:>10.2f}{all_s[1]:>+10.2f}"
                  f"{str_s[0]:>10.2f}{str_s[1]:>+10.2f}{str_s[2]:>10.2f}")

        # Per-user residuals side by side (strong cohort), so a single over-rated
        # account is visible across configs rather than hidden in the aggregate.
        labels = [t[0] for t in table]
        strong_users = sorted(
            {u for lbl in labels
             for u in per_config_df[lbl].loc[
                 per_config_df[lbl]["declared"] >= thr, "user"]})
        if not args.no_per_user and strong_users:
            print(f"\n=== per-user residual ({thr_label}+), predicted−declared ===")
            print(f"{'user':<10}{'declared':>9}" + "".join(f"{l:>10}" for l in labels))
            for u in strong_users:
                decl = None
                cells = []
                for lbl in labels:
                    sub = per_config_df[lbl]
                    row = sub[sub["user"] == u]
                    if len(row):
                        decl = scalar_label(row.iloc[0]["declared"])
                        cells.append(f"{row.iloc[0]['residual']:>+10.2f}")
                    else:
                        cells.append(f"{'—':>10}")
                print(f"{u:<10}{decl or '?':>9}" + "".join(cells))


if __name__ == "__main__":
    main()
