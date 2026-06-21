"""Compare user-declared server ranks against our internal rating.

Inputs (no network calls — re-run `pipeline.fetch_database` first to refresh
`attempts/database.sql`):
  - attempts/database.sql              (rank_surveys + tsumego_reports)
  - ratings/<size>/tsumego_parameters_final.csv  (per-problem mixture params)
  - ratings/<size>/user_profiles_final.csv       (humanv0 anchor θ ladder)

We fit θ per (user, size) directly from this user's attempts against the
per-problem mixture params — that way recently-added users still get a
rating even though `user_profiles_final.csv` (pipeline output) hasn't been
refit since. Per-problem params are properties of the problems, not the
users, so they don't go stale the way the user-month rating snapshot does.

For each user that submitted at least one rank-survey response:
  1. Fit θ_size per board size from this user's attempts. Convert θ to a
     kyu/dan number by piecewise-linear interp on the humanv0 anchor ladder
     in user_profiles_final.csv; the anchor y-values come from
     calibrate_handicap_ranks.RANK_PRIOR (handicap-game-derived rank),
     not nominal humanv0 labels — go ranks are defined by handicap and
     humanv0 drifts from its labels outside ~15k-5d. Take the max θ across
     sizes as "our rating" (1d is 1d regardless of which size proved it).
  2. Convert each declared server rank (e.g. "5d", "12k") to the same
     kyu/dan scalar (1k=0, 1d=1, 25k=-24, 9d=9). Server self-declarations
     stay on the nominal scale; only the humanv0 anchors are calibrated.
  3. Print a per-user table and a bucketed comparison.

Usage:
  ./.venv/bin/python3 -m auxiliary.compare_rank_surveys
  ./.venv/bin/python3 -m auxiliary.compare_rank_surveys --csv out.csv
  ./.venv/bin/python3 -m auxiliary.compare_rank_surveys --sizes 9
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.calibrate_handicap_ranks import RANK_PRIOR as CALIBRATED_RANK  # noqa: E402
# Survey/attempts loaders, the rank↔scalar convention, and the SQL row regexes are shared with
# the pipeline; they live in utils.rank_surveys and are re-exported here for this module's
# other callers.
from utils.rank_surveys import (  # noqa: E402
    FULL_ATTEMPT_RE, RANK_SURVEY_RE, SERVERS, SQL_PATH_DEFAULT,
    latest_per_user_server, load_attempts_for_users, load_rank_surveys,
    rank_to_scalar, scalar_to_rank,
)

RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")

HUMANV0_RE = re.compile(r"humanv0_(\d{1,2})([kd])$")
# Strong-net (kata1 b18c384nbt) at fixed temperature corresponds to spec_t<temp>
# in calibrate_handicap_ranks BOT_CONFIGS. user_profiles_final.csv writes the
# temp without trailing zeros ("_t1" not "_t1.0"), so we normalize on lookup.
SPECNET_RE = re.compile(r"kata1-b18c384nbt-.*_t([\d.]+)$")


# ---------------------------------------------------------------------------
# Internal-rating loader + θ → kyu/dan interp
# ---------------------------------------------------------------------------

def _calibration_key(bot_id):
    """Map a user_profiles bot_id to its calibrate_handicap_ranks
    BOT_CONFIGS label, or None if no calibration entry exists."""
    m = HUMANV0_RE.search(bot_id)
    if m:
        return f"{m.group(1)}{m.group(2)}"  # "5k", "3d"
    m = SPECNET_RE.search(bot_id)
    if m:
        temp = m.group(1)
        if "." not in temp:  # "1" → "1.0", "2" → "2.0"
            temp = f"{temp}.0"
        return f"spec_t{temp}"
    return None


def build_anchor_interp(profiles_df):
    """Build a monotone θ → calibrated kyu/dan scalar interpolator from the
    user_profiles bots that also appear in calibrate_handicap_ranks
    BOT_CONFIGS.

    Anchors used:
      - humanv0_Xk / humanv0_Yd (the "rank" type in BOT_CONFIGS) — the
        20k..9d ladder, anchored at humanv0_3k = 0.
      - kata1-...t<temp> spec-net variants ("spec_temp" type in
        BOT_CONFIGS, temps 1.0/0.8/0.6/0.4/0.1) — extend the strong end
        from +8.55 (9d) up to +12.45, so users above humanv0_9d's θ
        interpolate within the calibration ladder rather than
        extrapolating off the 8d-9d slope.

    A go rank IS defined by the ability to give handicap stones, so
    RANK_PRIOR is the correct y-axis; nominal humanv0 labels are just
    training tags and drift outside the ~15k-5d band. Anchors without
    a calibration entry (e.g. high-temp humanv0_20k_t1.1, uncalibrated
    spec temps t0.2/t0.05) are skipped rather than mixing scales."""
    pairs = []
    for bot_id, theta in zip(profiles_df["bot_id"], profiles_df["rating"]):
        key = _calibration_key(bot_id)
        if key is None or key not in CALIBRATED_RANK:
            continue
        pairs.append((float(theta), float(CALIBRATED_RANK[key])))
    if len(pairs) < 2:
        return None, None
    # Sort by θ, drop duplicates on θ (anchor table is monotone in rank but
    # rare flat ties would break np.interp's monotonicity).
    pairs.sort()
    theta_arr = np.array([p[0] for p in pairs])
    scalar_arr = np.array([p[1] for p in pairs])
    keep = np.concatenate(([True], np.diff(theta_arr) > 1e-9))
    return theta_arr[keep], scalar_arr[keep]


def theta_to_kyudan(theta, theta_arr, scalar_arr):
    """Piecewise-linear interp, extrapolating linearly past the ends using
    the slope of the nearest two anchors."""
    if theta_arr is None or len(theta_arr) < 2:
        return None
    if theta <= theta_arr[0]:
        slope = (scalar_arr[1] - scalar_arr[0]) / (theta_arr[1] - theta_arr[0])
        return scalar_arr[0] + slope * (theta - theta_arr[0])
    if theta >= theta_arr[-1]:
        slope = ((scalar_arr[-1] - scalar_arr[-2])
                 / (theta_arr[-1] - theta_arr[-2]))
        return scalar_arr[-1] + slope * (theta - theta_arr[-1])
    return float(np.interp(theta, theta_arr, scalar_arr))


# Stop-gap curve from pipeline/package_for_pwa.py — the PWA's displayed-rank
# mapping. Anchored at humanv0_8k (rank_index 12.5) and humanv0_3k (17.5)
# only, with k=0.12. The current full handi-anchored curve in
# package_for_pwa.fit_rating_curve under-curves the dan range until the new
# calibrate_handicap_ranks anchors land, so the site uses this stop-gap and
# we mirror it here to match what users actually see.
STOPGAP_K = 0.12
STOPGAP_X_8K = 12.5  # rank_index for 8k on the PWA's 0..28 axis
STOPGAP_X_3K = 17.5  # rank_index for 3k
RATING_FLOOR_MARGIN = 0.1


def load_stopgap_curve(size):
    """Read humanv0_8k and humanv0_3k from bot_ratings_final.csv at this size
    and fit r = A·exp(k·x_index) + C with k fixed and the two anchors
    pinning A, C. Same construction as package_for_pwa.process_size."""
    csv_path = os.path.join(RATINGS_DIR, str(size), "bot_ratings_final.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    try:
        y_8k = float(df.loc[df["bot_id"] == "b18c384nbt-humanv0_8k",
                            "rating"].values[0])
        y_3k = float(df.loc[df["bot_id"] == "b18c384nbt-humanv0_3k",
                            "rating"].values[0])
    except (IndexError, KeyError):
        return None
    k = STOPGAP_K
    A = (y_3k - y_8k) / (np.exp(k * STOPGAP_X_3K) - np.exp(k * STOPGAP_X_8K))
    C = y_8k - A * np.exp(k * STOPGAP_X_8K)
    return {"A": A, "k": k, "C": C}


def theta_to_stopgap_scalar(theta, curve):
    """θ → kyu/dan scalar via the PWA stop-gap exponential.

    PWA's getDescriptiveRank does floor(rank_index): rank_index 19 -> 1k
    (= scalar 0), rank_index 20 -> 1d (= scalar 1). So in continuous form,
    scalar = rank_index - 19."""
    if curve is None:
        return None
    r_floor = curve["C"] + RATING_FLOOR_MARGIN
    r = max(float(theta), r_floor)
    rank_index = np.log((r - curve["C"]) / curve["A"]) / curve["k"]
    return float(rank_index - 19.0)


def load_problem_params(size):
    """Return (params_df indexed by tsumego_id, anchor_theta, anchor_scalar,
    stopgap_curve)."""
    params_path = os.path.join(RATINGS_DIR, str(size),
                               "tsumego_parameters_final.csv")
    profiles_path = os.path.join(RATINGS_DIR, str(size),
                                 "user_profiles_final.csv")
    if not os.path.exists(params_path) or not os.path.exists(profiles_path):
        return None, None, None, None
    params = pd.read_csv(params_path).set_index("tsumego_id")
    profiles = pd.read_csv(profiles_path)
    theta_arr, scalar_arr = build_anchor_interp(profiles)
    stopgap = load_stopgap_curve(size)
    return params, theta_arr, scalar_arr, stopgap


def fit_theta(attempts, params, theta_lo=-3.0, theta_hi=10.0):
    """Fit θ for one user via 1D MLE on the 3-expert mixture model.

    P(solve | θ, prob) = Σ_k w_k · σ(α_k · (θ - β_k))

    Returns (theta, n_attempts_joined). Drops attempts whose tsumego_id isn't
    in params (legacy / since-removed problems). Returns (None, 0) if too
    few joined attempts."""
    rows = []
    for tid, success in attempts:
        if tid in params.index:
            rows.append((tid, success))
    if not rows:
        return None, 0

    df = pd.DataFrame(rows, columns=["tsumego_id", "success"])
    joined = df.join(params, on="tsumego_id", how="inner")
    if joined.empty:
        return None, 0

    alphas = joined[[f"discrimination_{k+1}" for k in range(3)]].to_numpy(
        dtype=float)
    betas = joined[[f"difficulty_{k+1}" for k in range(3)]].to_numpy(
        dtype=float)
    weights = joined[[f"weight_{k+1}" for k in range(3)]].to_numpy(dtype=float)
    success = joined["success"].to_numpy(dtype=float)

    def neg_ll(theta):
        diffs = alphas * (theta - betas)
        # Clip to keep np.exp from overflowing on extrapolated θ.
        diffs = np.clip(diffs, -40.0, 40.0)
        p_expert = 1.0 / (1.0 + np.exp(-diffs))
        p = np.sum(weights * p_expert, axis=1)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -float(np.sum(success * np.log(p)
                             + (1 - success) * np.log(1 - p)))

    res = minimize_scalar(neg_ll, bounds=(theta_lo, theta_hi),
                          method="bounded", options={"xatol": 1e-4})
    return float(res.x), len(joined)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def build_per_user_table(surveys, attempts_by_user_size, params_by_size,
                         sizes, min_attempts):
    """One row per user: ratings per size + declared rank per server.

    Two internal-rating columns per size:
      rating_pwa_<size>     — what the PWA actually shows (stop-gap curve)
      rating_anchor_<size>  — interp through full humanv0 ladder (raw θ)
    The PWA column is the apples-to-apples comparison with self-declared
    server ranks; the anchor column shows where the user sits on the raw
    rating axis (useful for spotting the over-rating bias documented in
    auxiliary/analyze_user_data.py)."""
    rows = []
    user_ids = sorted(surveys["user_id"].unique())
    for uid in user_ids:
        prefix = uid[:8]
        row = {"user_id": uid, "prefix": prefix}
        thetas = []
        scalars_pwa = []
        scalars_anchor = []
        for size in sizes:
            attempts = attempts_by_user_size.get((prefix, size), [])
            row[f"attempts_{size}"] = len(attempts)
            params, theta_arr, scalar_arr, stopgap = params_by_size.get(
                size, (None, None, None, None))
            if params is None or len(attempts) < min_attempts:
                row[f"theta_{size}"] = None
                row[f"rating_pwa_{size}"] = None
                row[f"rating_anchor_{size}"] = None
                continue
            theta, n_used = fit_theta(attempts, params)
            if theta is None or n_used < min_attempts:
                row[f"theta_{size}"] = None
                row[f"rating_pwa_{size}"] = None
                row[f"rating_anchor_{size}"] = None
                continue
            s_pwa = theta_to_stopgap_scalar(theta, stopgap)
            s_anchor = theta_to_kyudan(theta, theta_arr, scalar_arr)
            row[f"theta_{size}"] = theta
            row[f"rating_pwa_{size}"] = s_pwa
            row[f"rating_anchor_{size}"] = s_anchor
            thetas.append(theta)
            if s_pwa is not None:
                scalars_pwa.append(s_pwa)
            if s_anchor is not None:
                scalars_anchor.append(s_anchor)

        # "our rating" = max across sizes for each scale (a player's true
        # rank is a max-like quantity over their playing repertoire).
        row["our_rating_pwa"] = max(scalars_pwa) if scalars_pwa else None
        row["our_rating_anchor"] = (max(scalars_anchor)
                                     if scalars_anchor else None)
        row["our_label_pwa"] = scalar_to_rank(row["our_rating_pwa"])
        row["our_label_anchor"] = scalar_to_rank(row["our_rating_anchor"])
        row["our_theta_max"] = max(thetas) if thetas else None

        u_surv = surveys[surveys["user_id"] == uid]
        for server in SERVERS:
            hit = u_surv[u_surv["server"] == server]
            if hit.empty:
                row[server] = None
                row[f"{server}_scalar"] = None
            else:
                row[server] = hit.iloc[0]["rank"]
                row[f"{server}_scalar"] = hit.iloc[0]["rank_scalar"]
        row["latest_survey"] = u_surv["created_at"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(surveys, table, sizes):
    print(f"\n{'='*70}")
    print("RANK SURVEY SUMMARY")
    print(f"{'='*70}")
    print(f"Unique users surveyed:      {surveys['user_id'].nunique()}")
    print(f"Total survey rows:          {len(surveys)}")
    print(f"Users with ≥1 sized rating: "
          f"{(table['our_rating_pwa'].notna()).sum()} / {len(table)}")
    print()

    print("Declarations per server (one row per (user, server) latest):")
    counts = surveys.groupby("server").size().reindex(SERVERS, fill_value=0)
    for s in SERVERS:
        print(f"  {s:<8}  {counts[s]:>5d}")
    print()

    for size in sizes:
        col = f"attempts_{size}"
        with_attempts = (table[col] > 0).sum()
        with_rating = (table[f"rating_pwa_{size}"].notna()).sum()
        print(f"Size {size}: {with_attempts} users with any attempts; "
              f"{with_rating} with a usable rating")
    print()


def print_per_user_table(table, sizes, limit=None):
    print(f"\n{'='*120}")
    print("PER-USER COMPARISON")
    print("  θ<size> = raw IRT rating fit at that board size.")
    print("  pwa = what the PWA shows (stop-gap exponential, anchored at")
    print("        humanv0_8k/3k, nominal labels).")
    print("  raw = piecewise-linear interp through the humanv0 anchor")
    print("        ladder, with y-axis from calibrate_handicap_ranks")
    print("        (handicap-game-derived rank; trustworthy ~15k-5d,")
    print("        extrapolates outside that range).")
    print("  pwa/raw label columns are max-across-size.")
    print(f"{'='*120}")

    header = (
        f"{'prefix':<10}"
        + "".join(f"{f'a{s}':>6}" for s in sizes)
        + "".join(f"{f'θ{s}':>8}" for s in sizes)
        + f"{'pwa':>6}{'raw':>6}"
        + "".join(f"{s:>6}" for s in SERVERS)
    )
    print(header)
    print("-" * len(header))

    # Sort by PWA-displayed rating (None last), then by total attempts.
    def sort_key(r):
        rating = (r["our_rating_pwa"]
                  if r["our_rating_pwa"] is not None else -1e9)
        attempts = sum((r.get(f"attempts_{s}", 0) or 0) for s in sizes)
        return (-rating, -attempts)

    rows = table.to_dict("records")
    rows.sort(key=sort_key)
    if limit:
        rows = rows[:limit]

    for r in rows:
        line = f"{r['prefix']:<10}"
        for s in sizes:
            n = r.get(f"attempts_{s}") or 0
            line += f"{n:>6d}"
        for s in sizes:
            t = r.get(f"theta_{s}")
            if t is None or not np.isfinite(t):
                line += f"{'-':>8}"
            else:
                line += f"{t:>8.2f}"
        line += f"{r['our_label_pwa']:>6}"
        line += f"{r['our_label_anchor']:>6}"
        for server in SERVERS:
            v = r.get(server)
            line += f"{(v if v else '-'):>6}"
        print(line)


# Bucketing: span what users actually fall into, in θ-wide bands. Bucketing by
# raw θ (not kyu/dan scalar) means a single table covers both the PWA stop-gap
# and the raw humanv0 anchor scales — they just become two columns sharing the
# same rows.
def make_theta_buckets(values, width=0.5):
    """Return list of (lo, hi, label) bands in θ space."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return []
    lo_all = min(finite)
    hi_all = max(finite)
    start = float(np.floor(lo_all / width) * width)
    bands = []
    cur = start
    while cur <= hi_all:
        bands.append((cur, cur + width,
                      f"{cur:+.2f}..{cur+width:+.2f}"))
        cur += width
    return bands


def _fmt_bucket_cell(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{len(values)}/{scalar_to_rank(values[0])}"
    med = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return (f"{len(values)}/{scalar_to_rank(med)}"
            f"/{scalar_to_rank(q1)}-{scalar_to_rank(q3)}")


def print_bucketed_comparison(table, size):
    """θ-bucketed comparison for a single board size. pwa + raw 'Ours'
    columns sit side-by-side; only users with a fit θ at this size appear,
    so the size-9 and size-13 tables can have different membership."""
    theta_col = f"theta_{size}"
    pwa_col = f"rating_pwa_{size}"
    raw_col = f"rating_anchor_{size}"
    rated = table[table[theta_col].notna()].copy()
    if rated.empty:
        print(f"\nSize {size}: no rated users with surveys; bucketed "
              f"comparison skipped.")
        return

    buckets = make_theta_buckets(rated[theta_col].tolist(), width=0.5)
    if not buckets:
        return

    print(f"\n{'='*120}")
    print(f"RATING-BUCKETED RANK COMPARISON — size {size} "
          f"(bucketed by θ at this size)")
    print("Cell: n / median / IQR  (declared scalar rank within bucket).")
    print("Ours(pwa) = stop-gap exponential (nominal humanv0 labels).")
    print("Ours(raw) = humanv0 anchor interp on the handicap-calibrated")
    print("rank axis (calibrate_handicap_ranks.RANK_PRIOR).")
    print(f"{'='*120}")

    cols = ["Ours(pwa)", "Ours(raw)"] + SERVERS
    header = f"{'θ bucket':<16}{'n':>5}"
    for c in cols:
        header += f"{c:>14}"
    print(header)
    print("-" * len(header))

    for lo, hi, blabel in buckets:
        mask = (rated[theta_col] >= lo) & (rated[theta_col] < hi)
        sub = rated[mask]
        if sub.empty:
            continue
        line = f"{blabel:<16}{len(sub):>5d}"
        line += f"{_fmt_bucket_cell(sub[pwa_col].tolist()):>14}"
        line += f"{_fmt_bucket_cell(sub[raw_col].tolist()):>14}"
        for server in SERVERS:
            line += f"{_fmt_bucket_cell(sub[f'{server}_scalar'].tolist()):>14}"
        print(line)


def print_offsets(table):
    """One offsets block per internal-rating column so it's obvious which
    scale produces the cleanest agreement with each server."""
    print(f"\n{'='*70}")
    print("MEAN SIGNED OFFSET PER SERVER  (declared − ours, in ranks)")
    print(f"{'='*70}")
    print(f"{'server':<8}{'n':>5}"
          f"{'pwa_mean':>10}{'pwa_med':>9}{'pwa_std':>9}  |  "
          f"{'raw_mean':>10}{'raw_med':>9}{'raw_std':>9}")

    def stats(diffs):
        if not diffs:
            return ("-", "-", "-")
        diffs = np.array(diffs)
        return (f"{diffs.mean():+.2f}",
                f"{np.median(diffs):+.2f}",
                f"{diffs.std(ddof=0):.2f}")

    for server in SERVERS:
        diffs_pwa, diffs_raw = [], []
        for _, r in table.iterrows():
            ds = r.get(f"{server}_scalar")
            if ds is None or not np.isfinite(ds):
                continue
            if r["our_rating_pwa"] is not None:
                diffs_pwa.append(float(ds) - float(r["our_rating_pwa"]))
            if r["our_rating_anchor"] is not None:
                diffs_raw.append(float(ds) - float(r["our_rating_anchor"]))
        n = max(len(diffs_pwa), len(diffs_raw))
        m_p, md_p, s_p = stats(diffs_pwa)
        m_r, md_r, s_r = stats(diffs_raw)
        print(f"{server:<8}{n:>5d}"
              f"{m_p:>10}{md_p:>9}{s_p:>9}  |  "
              f"{m_r:>10}{md_r:>9}{s_r:>9}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare rank-survey declarations vs internal rating.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--sql", default=SQL_PATH_DEFAULT,
        help="Path to D1 SQL dump (re-run pipeline.fetch_database first)")
    parser.add_argument(
        "--sizes", default="9,13",
        help="Comma-separated board sizes to include")
    parser.add_argument(
        "--min-attempts", type=int, default=20,
        help="Skip per-size rating if user has fewer attempts at that size")
    parser.add_argument(
        "--csv", default=None,
        help="If set, write the per-user comparison table here")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap per-user table to N rows (sorted by rating desc)")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys.empty:
        print("  No rank_surveys rows found. If the dump was taken before"
              " the feature shipped, re-run `pipeline.fetch_database`.")
        return
    surveys = latest_per_user_server(surveys)
    print(f"  {len(surveys)} latest-per-(user,server) rank declarations "
          f"from {surveys['user_id'].nunique()} users")

    print(f"Loading attempts for {surveys['user_id'].nunique()} surveyed "
          f"users ...")
    attempts_by_user_size = load_attempts_for_users(
        args.sql, set(surveys["user_id"].unique()))
    n_pairs = len(attempts_by_user_size)
    n_attempts = sum(len(v) for v in attempts_by_user_size.values())
    print(f"  {n_attempts} attempts across {n_pairs} (user, size) pairs")

    params_by_size = {}
    for size in sizes:
        params, theta_arr, scalar_arr, stopgap = load_problem_params(size)
        if params is None:
            print(f"  size {size}: missing tsumego_parameters_final.csv or "
                  f"user_profiles_final.csv — skipping size")
        else:
            sg_state = ("stop-gap ready" if stopgap
                        else "stop-gap MISSING (no bot_ratings_final.csv)")
            print(f"  size {size}: {len(params)} problem params, "
                  f"{len(theta_arr)} calibrated anchors, {sg_state}")
        params_by_size[size] = (params, theta_arr, scalar_arr, stopgap)

    table = build_per_user_table(surveys, attempts_by_user_size,
                                 params_by_size, sizes, args.min_attempts)

    print_summary(surveys, table, sizes)
    print_per_user_table(table, sizes, limit=args.limit)
    for size in sizes:
        print_bucketed_comparison(table, size)
    print_offsets(table)

    if args.csv:
        out_cols = (["user_id", "prefix"]
                    + [c for s in sizes for c in
                       (f"attempts_{s}", f"theta_{s}",
                        f"rating_pwa_{s}", f"rating_anchor_{s}")]
                    + ["our_theta_max",
                       "our_rating_pwa", "our_label_pwa",
                       "our_rating_anchor", "our_label_anchor"]
                    + [c for s in SERVERS for c in (s, f"{s}_scalar")]
                    + ["latest_survey"])
        table[out_cols].to_csv(args.csv, index=False)
        print(f"\nPer-user table written to {args.csv}")


if __name__ == "__main__":
    main()
