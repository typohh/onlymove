"""Held-out human prediction harness for the rating model.

The pipeline fits a per-tsumego item response function (IRF) — currently a
mixture of `NUM_EXPERTS` logistic experts — and then collapses each problem to
`(rating, quality)`. This script measures the thing we actually care about: how
well the IRF predicts *held-out human* outcomes once a single per-user ability
(theta) is estimated by maximum likelihood.

Protocol (matches how the model is used in production):
  1. Human attempts are bucketed into 2-month overlapping profiles
     (`{user[:8]}_{YYYY-MM}` + the previous month), the same windowing
     rank_tsumegos_and_bots.load_data builds, so a profile's ability is roughly
     stationary. Repeat attempts (is_repeat=1) are dropped.
  2. Item parameters are taken as fixed (these are what we'd ship). For each
     profile we hold out a random fraction of its attempts, MLE theta on the
     rest, and score the held-out attempts.
  3. The random hold-out is *resampled* (Monte-Carlo cross-validation) and the
     aggregate held-out log-loss is accumulated until its 95% CI half-width
     drops below `--ci-tol` (or `--max-repeats` is reached) — i.e. we bootstrap
     the hold-out split until the result stabilizes.

It is model-agnostic in the item file: point `--params` at any
`tsumego_parameters_*.csv` with `difficulty_k/discrimination_k/weight_k`
columns (K inferred). So a 1-expert (2PL) / 2-expert / 3-expert fit can be
compared head-to-head on identical human splits — the direct test of whether
the mixture's extra shape flexibility (e.g. plateau-then-spike difficulty)
earns its keep on humans.

Usage:
  ./.venv/bin/python3 -m pipeline.evaluate_rating_model            # current size's shipped params
  ./.venv/bin/python3 -m pipeline.evaluate_rating_model --params ratings/13/tsumego_parameters_2pl.csv
"""

import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from tqdm import tqdm

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import BOARD_SIZE

EPS = 1e-6

# Mirrors the INSERT parser in rank_tsumegos_and_bots.load_data — keep in sync.
INSERT_PATTERN = re.compile(
    r"INSERT INTO [\"`]?tsumego_reports[\"`]?"
    r"(?:\s*\([^)]*\))?"
    r"\s*VALUES\s*\("
    r"\d+,'([^']+)','([^']+)',([01]),'([^']+)'"   # id, user_id, problem, success, created_at
    r"(?:,(\d+|NULL))?"                            # optional size
    r"(.*?)"                                       # optional move/time/review/is_repeat tail
    r"\);"
)


def load_human_attempts(size):
    """Parse attempts/database.sql into 2-month profile rows for one board size.

    Returns a DataFrame with columns [profile, tid, correct]. Each physical
    attempt contributes to two overlapping profiles (its month + the previous
    month), exactly like rank_tsumegos_and_bots.load_data.
    """
    sql_file = os.path.join(PROJECT_ROOT, "attempts", "database.sql")
    if not os.path.exists(sql_file):
        raise FileNotFoundError(f"{sql_file} not found; run pipeline.fetch_database first.")

    rows = []
    with open(sql_file, "r") as f:
        for line in tqdm(f, desc="Parsing SQL"):
            m = INSERT_PATTERN.match(line)
            if not m:
                continue
            user_id, tid, success, ts, size_str, tail = m.groups()

            tail_fields = [x for x in tail.split(",") if x != ""]
            if len(tail_fields) >= 4 and tail_fields[-1] == "1":  # is_repeat
                continue

            row_size = int(size_str) if size_str not in (None, "NULL") else 9
            if row_size != size:
                continue

            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue

            uid = user_id[:8]
            prev_m, prev_y = dt.month - 1, dt.year
            if prev_m == 0:
                prev_m, prev_y = 12, prev_y - 1
            for prof in (f"{uid}_{dt.year}-{dt.month:02d}", f"{uid}_{prev_y}-{prev_m:02d}"):
                rows.append((prof, tid, int(success)))

    df = pd.DataFrame(rows, columns=["profile", "tid", "correct"])
    print(f"Loaded {len(df)} human attempt-rows across {df['profile'].nunique()} profiles "
          f"(size {size}).")
    return df


def load_irf_params(path):
    """Load a tsumego_parameters CSV into per-tid (beta, alpha, weight) arrays.

    K (number of experts) is inferred from the column count.
    """
    p = pd.read_csv(path)
    k = sum(c.startswith("difficulty_") for c in p.columns)
    if k == 0:
        raise ValueError(f"No difficulty_* columns in {path}")
    beta = p[[f"difficulty_{i+1}" for i in range(k)]].to_numpy(np.float64)
    alpha = p[[f"discrimination_{i+1}" for i in range(k)]].to_numpy(np.float64)
    weight = p[[f"weight_{i+1}" for i in range(k)]].to_numpy(np.float64)
    params = {tid: (beta[i], alpha[i], weight[i]) for i, tid in enumerate(p["tsumego_id"])}
    print(f"Loaded IRF params for {len(params)} tsumegos (K={k} experts) from {os.path.basename(path)}.")
    return params, k


def prob_irf(theta, beta, alpha, weight):
    """Mixture-of-logistics success probability. theta scalar, arrays (n,K)."""
    return np.sum(weight * expit(alpha * (theta - beta)), axis=1)


def fit_theta(beta, alpha, weight, y, lo, hi):
    """Bounded MLE of ability theta for one profile's fit attempts."""
    def nll(theta):
        p = np.clip(prob_irf(theta, beta, alpha, weight), EPS, 1 - EPS)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    return res.x


def build_profiles(df, params):
    """Group attempts by profile into stacked (beta, alpha, weight, y) arrays.

    Only attempts whose tsumego has IRF params are kept.
    """
    df = df[df["tid"].isin(params)].copy()
    profiles = []
    for _, g in df.groupby("profile", sort=False):
        beta = np.stack([params[t][0] for t in g["tid"]])
        alpha = np.stack([params[t][1] for t in g["tid"]])
        weight = np.stack([params[t][2] for t in g["tid"]])
        y = g["correct"].to_numpy(np.float64)
        profiles.append((beta, alpha, weight, y))
    return profiles


def evaluate(profiles, theta_lo, theta_hi, holdout_frac, min_fit,
             max_repeats, min_repeats, ci_tol, seed):
    """Monte-Carlo CV: resample hold-out splits until held-out log-loss stabilizes."""
    rng = np.random.default_rng(seed)

    # Keep only profiles big enough to split.
    usable = [p for p in profiles if len(p[3]) >= min_fit + 1]
    print(f"{len(usable)}/{len(profiles)} profiles have >= {min_fit + 1} attempts (usable).")
    if not usable:
        raise SystemExit("No usable profiles.")

    per_repeat_logloss = []
    # Reliability accumulators (10 predicted-prob bins): count, sum_pred, sum_actual.
    bin_n = np.zeros(10); bin_p = np.zeros(10); bin_y = np.zeros(10)
    n_holdout_total = 0
    sum_y = 0.0
    # Pooled held-out (logit, y) for a single global temperature recalibration —
    # the cheapest possible "tune on humans": flatten every curve by one T.
    all_logit, all_y = [], []

    pbar = tqdm(range(max_repeats), desc="MC repeats")
    for r in pbar:
        ll_sum, ll_cnt = 0.0, 0
        for beta, alpha, weight, y in usable:
            n = len(y)
            n_hold = max(1, int(round(n * holdout_frac)))
            if n - n_hold < min_fit:
                n_hold = n - min_fit
            idx = rng.permutation(n)
            hold, fit = idx[:n_hold], idx[n_hold:]

            theta = fit_theta(beta[fit], alpha[fit], weight[fit], y[fit], theta_lo, theta_hi)
            p = np.clip(prob_irf(theta, beta[hold], alpha[hold], weight[hold]), EPS, 1 - EPS)
            yh = y[hold]

            ll_sum += -np.sum(yh * np.log(p) + (1 - yh) * np.log(1 - p))
            ll_cnt += n_hold

            if r == 0:
                n_holdout_total += n_hold
                sum_y += yh.sum()
            b = np.clip((p * 10).astype(int), 0, 9)
            np.add.at(bin_n, b, 1)
            np.add.at(bin_p, b, p)
            np.add.at(bin_y, b, yh)

            all_logit.append(np.log(p / (1 - p)))
            all_y.append(yh)

        per_repeat_logloss.append(ll_sum / ll_cnt)
        arr = np.array(per_repeat_logloss)
        mean = arr.mean()
        half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.inf
        pbar.set_postfix(logloss=f"{mean:.5f}", ci=f"{half:.5f}")
        if len(arr) >= min_repeats and half < ci_tol:
            break

    base_rate = sum_y / n_holdout_total
    base_ll = -(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate))

    # Fit one global temperature T on the pooled held-out logits (1 param over
    # millions of points -> leakage negligible). T>1 flattens the curves; the
    # recalibrated log-loss is the achievable floor if we only fix global steepness.
    logit = np.concatenate(all_logit); yy = np.concatenate(all_y)

    def ll_of(p):
        p = np.clip(p, EPS, 1 - EPS)
        return -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))

    # (a) global temperature only (steepness / overdispersion fix, symmetric).
    res_t = minimize_scalar(lambda t: ll_of(expit(logit / t)), bounds=(0.2, 8.0), method="bounded")
    # (b) guessing floor only: p' = c + (1-c)*p  (asymmetric low-tail lift).
    res_c = minimize_scalar(lambda c: ll_of(c + (1 - c) * expit(logit)), bounds=(0.0, 0.6), method="bounded")
    # (c) floor + temperature jointly (a real 3PL recalibration).
    res_ct = minimize(lambda x: ll_of(x[0] + (1 - x[0]) * expit(logit / x[1])),
                      x0=[0.2, 1.0], bounds=[(0.0, 0.6), (0.2, 8.0)], method="L-BFGS-B")

    return {
        "logloss_mean": mean,
        "logloss_ci_half": half,
        "repeats": len(per_repeat_logloss),
        "base_rate": base_rate,
        "base_logloss": base_ll,
        "n_usable_profiles": len(usable),
        "reliability": (bin_n, bin_p, bin_y),
        "raw_pooled_logloss": ll_of(expit(logit)),
        "temp_only": (res_t.x, ll_of(expit(logit / res_t.x))),
        "floor_only": (res_c.x, ll_of(res_c.x + (1 - res_c.x) * expit(logit))),
        "floor_temp": (res_ct.x[0], res_ct.x[1], res_ct.fun),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=None,
                    help="Path to tsumego_parameters CSV (default: ratings/<size>/tsumego_parameters_final.csv)")
    ap.add_argument("--size", type=int, default=BOARD_SIZE)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--min-fit", type=int, default=5,
                    help="Minimum fit attempts required to estimate a profile's theta.")
    ap.add_argument("--max-repeats", type=int, default=200)
    ap.add_argument("--min-repeats", type=int, default=15)
    ap.add_argument("--ci-tol", type=float, default=5e-4,
                    help="Stop resampling once the 95%% CI half-width on mean log-loss < this.")
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    params_path = args.params or os.path.join(
        PROJECT_ROOT, "ratings", str(args.size), "tsumego_parameters_final.csv")

    # theta search bounds from bot rating range (same idea as the pipeline's inversion).
    bots = pd.read_csv(os.path.join(PROJECT_ROOT, "ratings", str(args.size), "bot_ratings_final.csv"))
    lo, hi = bots["rating"].min(), bots["rating"].max()
    rng_ = hi - lo
    theta_lo, theta_hi = lo - rng_, hi + rng_

    params, k = load_irf_params(params_path)
    df = load_human_attempts(args.size)
    profiles = build_profiles(df, params)

    res = evaluate(profiles, theta_lo, theta_hi, args.holdout_frac, args.min_fit,
                   args.max_repeats, args.min_repeats, args.ci_tol, args.seed)

    print("\n" + "=" * 60)
    print(f"Model:           {os.path.basename(params_path)}  (K={k})")
    print(f"Board size:      {args.size}")
    print(f"Usable profiles: {res['n_usable_profiles']}")
    print(f"MC repeats:      {res['repeats']}")
    print("-" * 60)
    print(f"Held-out log-loss:   {res['logloss_mean']:.5f}  (+/- {res['logloss_ci_half']:.5f})")
    print(f"Base-rate log-loss:  {res['base_logloss']:.5f}   (base rate {res['base_rate']:.3f})")
    improve = (res["base_logloss"] - res["logloss_mean"]) / res["base_logloss"] * 100
    print(f"Improvement over base rate: {improve:.1f}%")
    print("-" * 60)
    print(f"Pooled held-out recalibration ('tune on humans'):  raw={res['raw_pooled_logloss']:.5f}")
    base = res["base_logloss"]
    t_x, t_ll = res["temp_only"]
    c_x, c_ll = res["floor_only"]
    fc, ft, ft_ll = res["floor_temp"]
    def pct(ll):
        return (base - ll) / base * 100
    print(f"  temperature only:  T={t_x:.3f}                 ll={t_ll:.5f}  ({pct(t_ll):.1f}% over base)")
    print(f"  floor only:        c={c_x:.3f}                 ll={c_ll:.5f}  ({pct(c_ll):.1f}% over base)")
    print(f"  floor + temp:      c={fc:.3f}, T={ft:.3f}      ll={ft_ll:.5f}  ({pct(ft_ll):.1f}% over base)")
    print("-" * 60)
    print("Reliability (predicted vs empirical, by predicted-prob decile):")
    bin_n, bin_p, bin_y = res["reliability"]
    print(f"  {'bin':>8} {'n':>10} {'pred':>8} {'actual':>8}")
    ece = 0.0
    tot = bin_n.sum()
    for i in range(10):
        if bin_n[i] == 0:
            continue
        pred = bin_p[i] / bin_n[i]
        act = bin_y[i] / bin_n[i]
        ece += bin_n[i] / tot * abs(pred - act)
        print(f"  {i/10:.1f}-{(i+1)/10:.1f} {int(bin_n[i]):>10} {pred:>8.3f} {act:>8.3f}")
    print(f"  ECE: {ece:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
