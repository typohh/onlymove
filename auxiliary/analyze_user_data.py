"""Diagnose calibration of the PWA displayed-rank pipeline using
unanchored human user data.

We can't tell each user's true rank (no labels). But the human attempt
data + the rating model's per-problem mixture parameters do let us ask:

  1. Where do users sit in the model's θ-space relative to the named-bot
     ladder? If most users cluster above humanv0_9d, the model is
     systematically over-rating users vs bots at equivalent task
     performance — the core suspicion for why a 4d player displays as 14d.

  2. Do users solve at the rates the model predicts? Per-attempt residuals
     binned by model-predicted P(solve) reveal whether human solve rates
     follow the mixture-of-sigmoids shape the model assumes, and in which
     direction the bias runs.

  3. Is the model's effective slope on humans steeper or shallower than
     on bots? Fit a single global slope multiplier K so that
     P(user solves) = σ_mix(K · α_p · (θ_u − β_p)) best matches observed
     solves. K=1 means the model's per-stone discrimination is right for
     humans; K > 1 means humans see sharper transitions per r-unit (= fewer
     handi-stones per logit of solve-rate change than bots).

  4. Within-user dispersion (kept simple here): the std of per-bucket
     observed solve rate around the model's prediction, pooled across
     users. Tells us roughly how noisy the rating model's per-problem r
     is on human data.

Run:
  ./.venv/bin/python3 -m auxiliary.analyze_user_data            # both sizes
  ./.venv/bin/python3 -m auxiliary.analyze_user_data --size 9   # one size
"""

import argparse
import os
import sys
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES  # noqa: E402

USER_ID_RE = re.compile(r"^[a-z0-9]{8}_\d{4}-\d{2}$")
NUM_EXPERTS = 3

SQL_PATTERN = re.compile(
    r"INSERT INTO [\"`]?tsumego_reports[\"`]?\s*(?:\([^)]*\))?\s*VALUES\("
    r"\d+,'([^']+)','([^']+)',([01]),'([^']+)'"
    r"(?:,(\d+))?(?:,[^)]*)?\);"
)


def load_raw_user_attempts(sql_path, target_size):
    """Same parsing logic as rank_tsumegos_and_bots.load_data, minus the
    profile expansion. Returns (user_id, tsumego_id, success) tuples
    filtered to target_size."""
    if not os.path.exists(sql_path):
        return pd.DataFrame()
    rows = []
    with open(sql_path, "r") as f:
        for line in f:
            m = SQL_PATTERN.match(line)
            if not m:
                continue
            user_id, tid, success, ts, sz = m.groups()
            row_size = int(sz) if sz else 9
            if row_size != target_size:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            # Use the *current-month* profile only. The rating pipeline
            # also duplicates each attempt into the previous month for
            # overlap, but for per-attempt residuals we want each attempt
            # counted once against the profile that was actually fit on it.
            cur = f"{user_id[:8]}_{dt.year}-{dt.month:02d}"
            rows.append({
                "user_id": cur,
                "tsumego_id": tid,
                "success": int(success),
            })
    return pd.DataFrame(rows)


def mixture_predict(theta, alphas, betas, weights, slope_mult=1.0):
    """P(solve) for a participant at θ on a problem with mixture params.
    `slope_mult` multiplies α (=1 reproduces the model exactly)."""
    diffs = slope_mult * alphas * (theta - betas)
    expert = 1.0 / (1.0 + np.exp(-diffs))
    return float(np.sum(weights * expert))


def mixture_predict_vec(theta_vec, alphas_mat, betas_mat, weights_mat,
                        slope_mult=1.0):
    """Vectorized mixture predict over many attempts.
    Shapes: theta (N,), alphas/betas/weights (N, K). Returns (N,)."""
    diffs = slope_mult * alphas_mat * (theta_vec[:, None] - betas_mat)
    expert = 1.0 / (1.0 + np.exp(-diffs))
    return np.sum(weights_mat * expert, axis=1)


def analyze_size(size, min_attempts=50):
    ratings_dir = os.path.join(PROJECT_ROOT, "ratings", str(size))
    if not os.path.isdir(ratings_dir):
        print(f"  no ratings dir at {ratings_dir}; skipping size {size}")
        return

    profiles_path = os.path.join(ratings_dir, "user_profiles_final.csv")
    params_path = os.path.join(ratings_dir, "tsumego_parameters_final.csv")
    sql_path = os.path.join(PROJECT_ROOT, "attempts", "database.sql")
    missing = [p for p in (profiles_path, params_path) if not os.path.exists(p)]
    if missing:
        print(f"  missing inputs for size {size}: {missing}")
        return

    print(f"\n{'='*70}")
    print(f"Board {size}")
    print(f"{'='*70}")

    profiles = pd.read_csv(profiles_path)
    params = pd.read_csv(params_path)

    is_user = profiles["bot_id"].str.match(USER_ID_RE)
    user_profiles = profiles[is_user].copy()
    bot_profiles = profiles[~is_user].copy()

    print(f"\nProfiles loaded: {len(user_profiles)} user-month buckets, "
          f"{len(bot_profiles)} bots")

    sized_users = user_profiles[user_profiles["attempts"] >= min_attempts].copy()
    print(f"User profiles with ≥{min_attempts} attempts: {len(sized_users)}")
    if len(sized_users) == 0:
        print("  not enough data; skipping rest of analysis")
        return

    # --- 1. User θ distribution vs named-bot ladder ----------------------
    print(f"\n--- 1. User θ distribution vs named-bot ladder ---")
    theta_user = sized_users["rating"].values
    quantiles = [5, 25, 50, 75, 95]
    qvals = np.percentile(theta_user, quantiles)
    print(f"  User θ percentiles: " +
          "  ".join(f"p{q}={v:+.2f}" for q, v in zip(quantiles, qvals)))

    # Named-bot θ ladder for reference. Build {label: θ} from the humanv0
    # entries in bot_profiles.
    named_bots = {}
    for _, row in bot_profiles.iterrows():
        m = re.match(r"b18c384nbt-humanv0_(\d{1,2}[kd])$", row["bot_id"])
        if m:
            named_bots[m.group(1)] = float(row["rating"])

    ordered_names = ["20k", "15k", "10k", "5k", "3k", "1k",
                     "1d", "3d", "5d", "7d", "9d"]
    print(f"\n  Named-bot θ + fraction of users above each:")
    print(f"  {'bot':<6}  {'θ':>7s}  {'% users above':>14s}")
    for n in ordered_names:
        if n in named_bots:
            theta_b = named_bots[n]
            pct = (theta_user > theta_b).mean() * 100
            print(f"  {n:<6}  {theta_b:+7.3f}  {pct:13.1f}%")

    if "9d" in named_bots:
        n_above_9d = (theta_user > named_bots["9d"]).sum()
        print(f"\n  {n_above_9d}/{len(theta_user)} users "
              f"({n_above_9d/len(theta_user)*100:.1f}%) "
              f"sit above humanv0_9d's θ ({named_bots['9d']:+.3f}).")

    # --- 2. Per-attempt model-predicted P vs observed solves -------------
    raw = load_raw_user_attempts(sql_path, size)
    if raw.empty:
        print("\n  no raw user attempts (need attempts/database.sql); "
              "skipping residual + slope analyses")
        return

    theta_lookup = dict(zip(profiles["bot_id"], profiles["rating"]))
    # Index params by tsumego_id for fast lookup.
    params_indexed = params.set_index("tsumego_id")

    # Vectorize: build (theta_u, alphas, betas, weights, success) arrays.
    merged = raw.merge(params, on="tsumego_id", how="inner")
    merged["theta"] = merged["user_id"].map(theta_lookup)
    merged = merged.dropna(subset=["theta"])

    if len(merged) == 0:
        print("\n  no user attempts joined to problem params; skipping")
        return

    theta_arr = merged["theta"].to_numpy(dtype=float)
    alphas = np.stack([merged[f"discrimination_{k+1}"].to_numpy(dtype=float)
                       for k in range(NUM_EXPERTS)], axis=1)
    betas = np.stack([merged[f"difficulty_{k+1}"].to_numpy(dtype=float)
                      for k in range(NUM_EXPERTS)], axis=1)
    weights = np.stack([merged[f"weight_{k+1}"].to_numpy(dtype=float)
                        for k in range(NUM_EXPERTS)], axis=1)
    success = merged["success"].to_numpy(dtype=float)

    preds = mixture_predict_vec(theta_arr, alphas, betas, weights, slope_mult=1.0)

    print(f"\n--- 2. Predicted vs observed solve rates (model assumes K=1) ---")
    print(f"  {len(merged)} attempt rows joined to params.")
    print(f"  Overall observed solve rate: {success.mean():.3f}")
    print(f"  Overall mean predicted P:    {preds.mean():.3f}")
    print(f"  Mean residual (obs − pred):  {(success - preds).mean():+.3f}")
    print()
    print(f"  Binned by predicted P:")
    print(f"    {'pred range':>11s}  {'n':>7s}  {'pred':>8s}  {'obs':>8s}  {'residual':>10s}")
    bin_edges = np.linspace(0, 1, 11)
    for i in range(len(bin_edges) - 1):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i+1] if i < 9
                                          else preds <= bin_edges[i+1])
        if mask.sum() > 0:
            pm = preds[mask].mean()
            om = success[mask].mean()
            print(f"    {bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}  "
                  f"{mask.sum():>7d}  {pm:>8.3f}  {om:>8.3f}  "
                  f"{om - pm:>+10.3f}")

    # --- 3. Pooled slope-multiplier K fit --------------------------------
    print(f"\n--- 3. Slope-multiplier fit: K=1 means humans match bot-derived"
          f" discrimination ---")

    def neg_ll(K):
        if K <= 0:
            return 1e9
        p = mixture_predict_vec(theta_arr, alphas, betas, weights, slope_mult=K)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(np.mean(success * np.log(p) + (1 - success) * np.log(1 - p)))

    base_ll = neg_ll(1.0)
    result = minimize_scalar(neg_ll, bounds=(0.05, 10.0), method="bounded",
                             options={"xatol": 1e-4})
    K_fit = result.x
    fit_ll = result.fun
    print(f"  Base BCE (K=1):    {base_ll:.6f}")
    print(f"  Best BCE:          {fit_ll:.6f}  at K = {K_fit:.3f}")
    delta_ll = base_ll - fit_ll
    print(f"  ΔBCE vs K=1:       {delta_ll:.6f}  "
          f"({'humans sharper' if K_fit > 1 else 'humans blunter'} than model)")
    # Rough scan around K=1 to show the curve.
    print()
    print(f"  BCE at sample K values:")
    for K in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        print(f"    K={K:>4.2f}:  BCE = {neg_ll(K):.6f}")

    # --- 4. Per-attempt residual pooled stats ----------------------------
    print(f"\n--- 4. Residual dispersion (single-attempt; mostly Bernoulli noise) ---")
    resid = success - preds
    print(f"  std(residual):                 {resid.std():.4f}")
    # Theoretical Bernoulli noise floor: each attempt's residual variance is
    # p(1-p). Mean of these is the "irreducible" noise floor.
    pq = preds * (1 - preds)
    print(f"  mean Bernoulli noise (p(1−p)): {pq.mean():.4f}")
    print(f"  excess variance ratio:         {(resid.std() ** 2) / pq.mean():.3f}")
    print(f"  (≥1 means residuals are wider than pure Bernoulli — typically "
          f"because attempts within a user are correlated by hidden 'effort'\n   "
          f"or time-of-day; <1 would be over-fitting on the user side)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, choices=ALL_BOARD_SIZES,
                        help="Board size to analyze (default: all configured)")
    parser.add_argument("--min-attempts", type=int, default=50,
                        help="Minimum attempts to include a user profile")
    args = parser.parse_args()

    sizes = [args.size] if args.size else ALL_BOARD_SIZES
    for size in sizes:
        analyze_size(size, min_attempts=args.min_attempts)


if __name__ == "__main__":
    main()
