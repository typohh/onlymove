"""How accurate is bot-derived tsumego difficulty at predicting REAL users?

Run AFTER fitting a bot-only rating model:
    ./.venv/bin/python3 -m pipeline.rank_tsumegos_and_bots --no-users
which writes a bot-only IRF (tsumego_parameters_final.csv) to ratings/<size>/.
Because the IRF was fit without any user attempts, scoring it against real users
is a fully out-of-sample test of "do the bots predict human difficulty?".

Method
------
Each problem's difficulty is the 3-sigmoid mixture IRF from the bot-only fit:
    P(success | theta) = Σ_k w_k · σ(α_k · (theta − β_k))
on the bot skill scale theta (same scale as the tsumego `rating`). We:

1. Load real users' first-time attempts (is_repeat=0, deduped) for the board.
2. For each user with >= Y correct AND >= Y incorrect, MLE-fit a single skill
   theta_u against the bot-only IRF over their attempts (places the user on the
   bot scale; the IRF itself never saw users).
3. Predict p_hat = IRF_p(theta_u) for every attempt and report:
   - CALIBRATION (bias): observed vs predicted success across p_hat bins. Flat on
     the diagonal ⇒ no systematic mean shift (what you'd expect if only noise).
   - NOISE (overdispersion): group attempts by problem; under "rating is exact"
     the per-problem count variance equals the Poisson-binomial floor
     Σ p_hat(1−p_hat). Pooled excess variance, divided by the squared local IRF
     slope, estimates σ_ε — the std of per-problem difficulty misalignment in
     rating units. Compared against the spread of the ratings themselves.

σ_ε is an aggregate (one number for the whole board), not a per-problem error.
theta_u is fit in-sample on the user's own attempts, which mildly *shrinks*
per-problem residuals, so σ_ε here is a slight lower bound on true noise.

Usage:
  python -m auxiliary.eval_bot_difficulty            # uses BOARD_SIZE
  python -m auxiliary.eval_bot_difficulty --size 13 --min 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402

EPS = 1e-9


def load_irf(size):
    """Return {tid: (w(3), alpha(3), beta(3))}, the bot rating range, and the
    std of the bot-only tsumego ratings (the reference difficulty spread)."""
    rd = os.path.join(PROJECT_ROOT, "ratings", str(size))
    params = pd.read_csv(os.path.join(rd, "tsumego_parameters_final.csv"))
    irf = {}
    for _, r in params.iterrows():
        w = np.array([r.weight_1, r.weight_2, r.weight_3])
        a = np.array([r.discrimination_1, r.discrimination_2, r.discrimination_3])
        b = np.array([r.difficulty_1, r.difficulty_2, r.difficulty_3])
        irf[r.tsumego_id] = (w, a, b)
    bots = pd.read_csv(os.path.join(rd, "bot_ratings_final.csv"))
    lo, hi = bots.rating.min(), bots.rating.max()
    rng = hi - lo
    ratings = pd.read_csv(os.path.join(rd, "tsumego_ratings_final.csv"))
    return irf, (lo - rng, hi + rng), float(ratings.rating.std())


def irf_prob(theta, w, a, b):
    """P(success) at scalar theta for stacked params w/a/b each (n, 3) -> (n,)."""
    s = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    return (w * s).sum(axis=1)


def irf_slope(theta, w, a, b):
    """dP/dtheta at theta -> (n,). Local sensitivity of success to skill."""
    s = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    return (w * a * s * (1.0 - s)).sum(axis=1)


def fit_theta(w, a, b, y, bounds):
    """MLE the user's skill theta against their attempts (Bernoulli likelihood)."""
    def neg_ll(theta):
        p = np.clip(irf_prob(theta, w, a, b), EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    return minimize_scalar(neg_ll, bounds=bounds, method="bounded").x


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=None,
                        help="Board size (default: BOARD_SIZE from configurations).")
    parser.add_argument("--min", type=int, default=5,
                        help="Y: min correct AND min incorrect first-time answers "
                             "for a user to be skill-estimable (default 5).")
    parser.add_argument("--bins", type=int, default=10,
                        help="Calibration bins over predicted probability (default 10).")
    parser.add_argument("--boot", type=int, default=300,
                        help="Bootstrap resamples (over problems) for the σ_ε CI.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size is None:
        from utils.configurations import BOARD_SIZE
        args.size = BOARD_SIZE

    irf, bounds, rating_std = load_irf(args.size)
    print(f"Bot-only IRF: {len(irf)} problems | theta search {bounds[0]:.2f}..{bounds[1]:.2f}"
          f" | rating std (difficulty spread) = {rating_std:.3f}")

    df = load_first_time_attempts()
    df = df[df["size"] == args.size].copy()
    # Keep only attempts on problems that survived the bot-only fit (monotonic).
    in_irf = df["problem"].isin(irf.keys())
    n_drop = int((~in_irf).sum())
    df = df[in_irf]
    print(f"Real-user first-time attempts on size {args.size}: {len(df)} "
          f"({n_drop} dropped: problem not in bot-only IRF)")

    # Qualified, skill-estimable users.
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    qual = set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)])
    df = df[df["user_id"].isin(qual)]
    print(f"Skill-estimable users (>= {args.min} correct & incorrect): {len(qual)} "
          f"| attempts from them: {len(df)}")

    # Fit each user's theta, then predict every attempt.
    rng = np.random.default_rng(args.seed)
    p_hat = np.empty(len(df))
    slope = np.empty(len(df))
    y = df["success"].to_numpy(float)
    probs = df["problem"].to_numpy()
    rows_by_user = df.groupby("user_id").indices
    for uid, idx in rows_by_user.items():
        tids = probs[idx]
        w = np.stack([irf[t][0] for t in tids])
        a = np.stack([irf[t][1] for t in tids])
        b = np.stack([irf[t][2] for t in tids])
        yi = y[idx]
        theta = fit_theta(w, a, b, yi, bounds)
        p_hat[idx] = irf_prob(theta, w, a, b)
        slope[idx] = irf_slope(theta, w, a, b)
    p_hat = np.clip(p_hat, EPS, 1 - EPS)

    # --- Calibration (bias) ---
    print(f"\nCalibration ({args.bins} equal-width predicted-prob bins):")
    print(f"  {'p_hat range':>15s} {'n':>7s} {'pred':>7s} {'obs':>7s} {'obs-pred':>9s}")
    edges = np.linspace(0, 1, args.bins + 1)
    bin_idx = np.clip(np.digitize(p_hat, edges) - 1, 0, args.bins - 1)
    for k in range(args.bins):
        m = bin_idx == k
        if not m.any():
            continue
        print(f"  {edges[k]:.2f}-{edges[k+1]:.2f}      {int(m.sum()):>7d} "
              f"{p_hat[m].mean():>7.3f} {y[m].mean():>7.3f} {y[m].mean()-p_hat[m].mean():>+9.3f}")
    brier = np.mean((y - p_hat) ** 2)
    base = np.mean((y - y.mean()) ** 2)
    ll = -np.mean(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))
    ll_base = -np.mean(y * np.log(y.mean()) + (1 - y) * np.log(1 - y.mean()))
    print(f"  overall: predicted={p_hat.mean():.3f} observed={y.mean():.3f} "
          f"(global mean shift {y.mean()-p_hat.mean():+.3f})")
    print(f"  Brier: {brier:.4f} (predict-global-mean baseline {base:.4f})")
    print(f"  LogLoss: {ll:.4f} (baseline {ll_base:.4f})")

    # --- Overdispersion (noise) ---
    # Per problem: O−E, Poisson-binomial floor, and (Σ slope)². σ_ε² = pooled
    # excess variance / pooled squared slope, in theta (=rating) units.
    dd = pd.DataFrame({"problem": probs, "y": y, "p": p_hat,
                       "v": p_hat * (1 - p_hat), "s": slope})
    grp = dd.groupby("problem")
    n_p = grp.size().to_numpy()
    O = grp["y"].sum().to_numpy()
    E = grp["p"].sum().to_numpy()
    Vfloor = grp["v"].sum().to_numpy()
    S = grp["s"].sum().to_numpy()
    rep = n_p >= 2  # only replicated problems inform overdispersion
    a_term = (O - E) ** 2 - Vfloor          # excess count variance per problem
    b_term = S ** 2                         # slope² maps θ-noise → count variance
    num, den = a_term[rep].sum(), b_term[rep].sum()
    var_eps = num / den if den > 0 else float("nan")
    sigma_eps = np.sqrt(var_eps) if var_eps > 0 else float("nan")

    # Grouped overdispersion ratio φ on replicated problems (1.0 ⇒ no excess).
    phi = (O - E)[rep] ** 2
    phi = phi.sum() / Vfloor[rep].sum()

    # Bootstrap CI over replicated problems.
    ar, br = a_term[rep], b_term[rep]
    boots = []
    m = len(ar)
    for _ in range(args.boot):
        j = rng.integers(0, m, m)
        v = ar[j].sum() / br[j].sum()
        boots.append(np.sqrt(v) if v > 0 else 0.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"\nNoise / overdispersion (replicated problems with >=2 attempts: {int(rep.sum())}):")
    print(f"  grouped overdispersion φ = {phi:.3f}  (1.0 = exactly Poisson-binomial, no excess)")
    print(f"  σ_ε (per-problem rating misalignment) = {sigma_eps:.3f} rating units "
          f"[95% CI {lo:.3f}..{hi:.3f}]")
    print(f"  for reference, difficulty spread (rating std) = {rating_std:.3f}  "
          f"→ σ_ε is {sigma_eps / rating_std * 100:.0f}% of the rating spread")


if __name__ == "__main__":
    main()
