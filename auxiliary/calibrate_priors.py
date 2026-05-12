"""
Calibrate PWA logit-model priors against the production attempts database.

What we can calibrate from the DB
---------------------------------
The reports table is (user_id, tsumego_id, success, created_at, size) with a
UNIQUE(user_id, problem) constraint, so we only see one row per user/problem.
That means we can only inform priors on features that are derivable from rating
and PC-embedding alone:

  - LOGIT_PER_RANK_PRIOR + LOGIT_PER_RANK_PRIOR_SIGMA   (pwa/js/model.js)
      Population mean +/- std of |beta_difficulty|, where beta_diff is the
      slope of success vs. problem rating in a per-chunk logistic fit.
  - L2_REGULARIZATION_TARGETS[PROBLEM_DIFFICULTY]       (pwa/js/model.js)
      Population mean of the signed beta_difficulty (in logit/rating-unit).
  - RIDGE_LAMBDA                                        (pwa/js/weakness.js)
      L2 strength on the 64-PC beta_k's. Picked by minimizing held-out NLL
      (5-fold within each chunk), summed across chunks.
  - Empirical-Bayes cross-check: at the chosen lambda, the pooled std of
      beta_k across chunks should satisfy 1/std^2 ~ lambda.

What we cannot calibrate
------------------------
TIME_SOLVING, REVIEW_TIME, IS_NOVEL, RECALL_FORMULA priors need repeat /
timing data, which the DB does not carry.

Per the PWA's per-size split being lazy state plumbing rather than a real
population claim, we pool 9x9 and 13x13 attempts.

Chunking
--------
Each user's attempts are split into 60-day windows with a 30-day stride;
chunks with < MIN_ATTEMPTS_PER_CHUNK attempts are dropped. Overlap mainly
serves to multiply the count of "stable" profiles per long-tenured user.
Overlapping chunks share data, so they are not independent for population
statistics; we report this caveat in the output.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_DB = os.path.join(PROJECT_ROOT, "attempts", "database.sql")
EMBED_CSV = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")

WINDOW_DAYS = 60
STRIDE_DAYS = 30
MIN_ATTEMPTS_PER_CHUNK = 100  # overridden by --min-attempts
CV_FOLDS = 5
LAMBDA_GRID = [30.0, 60.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0]
K_EMBED = 64
SEED = 0

INSERT_RE = re.compile(r"VALUES\((\d+),'([^']+)','([^']+)',(\d+),'([^']+)',(\d+)\)")


def parse_db(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = INSERT_RE.search(line)
            if not m:
                continue
            _id, user, prob, succ, ts, size = m.groups()
            rows.append((
                user, prob, int(succ),
                datetime.fromisoformat(ts.replace("Z", "+00:00")),
                int(size),
            ))
    return pd.DataFrame(rows, columns=["user_id", "tsumego_id", "success", "ts", "size"])


def pwa_rescale(raw_rating, bot_ratings_df, k=0.12, x1=12.5, x2=17.5):
    """Apply package_for_pwa's exponential-then-normalize transform.

    Matches pipeline/package_for_pwa.py (lines ~230-280):
      A, C from anchors (humanv0_8k @ x1, humanv0_3k @ x2),
      rank_index = log((raw - C) / A) / k,
      pwa_rating = clip(rank_index / 29, 0, 1).
    The PWA's logistic models see this clipped [0, 1] value as `problemRating`,
    so beta_diff fit on this scale is directly comparable to model.js's
    L2_REGULARIZATION_TARGETS[PROBLEM_DIFFICULTY] (= -10) and to
    LOGIT_PER_RANK_PRIOR after multiplying by 1/29.
    """
    y1 = bot_ratings_df.loc[bot_ratings_df.bot_id == "b18c384nbt-humanv0_8k", "rating"].values[0]
    y2 = bot_ratings_df.loc[bot_ratings_df.bot_id == "b18c384nbt-humanv0_3k", "rating"].values[0]
    A = (y2 - y1) / (np.exp(k * x2) - np.exp(k * x1))
    C = y1 - A * np.exp(k * x1)
    val = np.clip((raw_rating - C) / A, 1e-9, None)
    rank_idx = np.log(val) / k
    return np.clip(rank_idx / 29.0, 0.0, 1.0)


def load_problem_features():
    embed_cols = [f"embed_{i}" for i in range(K_EMBED)]
    emb = pd.read_csv(EMBED_CSV, usecols=["tsumego_id", "size"] + embed_cols)
    rating_frames = []
    for sz in (9, 13):
        r = pd.read_csv(os.path.join(PROJECT_ROOT, "ratings", str(sz), "tsumego_ratings_final.csv"))
        bots = pd.read_csv(os.path.join(PROJECT_ROOT, "ratings", str(sz), "bot_ratings_final.csv"))
        r["pwa_rating"] = pwa_rescale(r["rating"].values, bots)
        r["raw_rating"] = r["rating"]
        r["rating"] = r["pwa_rating"]
        r["size"] = sz
        rating_frames.append(r[["tsumego_id", "size", "rating", "raw_rating"]])
    ratings = pd.concat(rating_frames, ignore_index=True)
    return emb.merge(ratings, on=["tsumego_id", "size"], how="inner"), embed_cols


def chunk_user(times):
    if len(times) == 0:
        return
    start = times[0]
    end_all = times[-1]
    window = np.timedelta64(WINDOW_DAYS, "D")
    stride = np.timedelta64(STRIDE_DAYS, "D")
    while start <= end_all:
        end = start + window
        idx = np.where((times >= start) & (times < end))[0]
        if len(idx) >= MIN_ATTEMPTS_PER_CHUNK:
            yield idx
        start = start + stride


def nll_and_grad(beta, X, y, lam_vec):
    z = X @ beta
    # numerically stable log-likelihood
    log_p = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0, z)
    nll = -np.sum(y * log_p + (1.0 - y) * log_1mp)
    p = expit(z)
    grad = X.T @ (p - y) + lam_vec * beta
    reg = 0.5 * float(np.sum(lam_vec * beta * beta))
    return nll + reg, grad


def fit_logistic(X, y, lam_vec, beta0=None):
    d = X.shape[1]
    if beta0 is None:
        beta0 = np.zeros(d)
    res = minimize(
        nll_and_grad, beta0, args=(X, y, lam_vec),
        jac=True, method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7},
    )
    return res.x, res.success


def held_out_nll(beta, X, y):
    z = X @ beta
    log_p = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0, z)
    return -float(np.sum(y * log_p + (1.0 - y) * log_1mp))


def build_chunks(df, embed_cols):
    chunks = []
    for user, grp in df.groupby("user_id", sort=False):
        grp = grp.sort_values("ts")
        times = grp.ts.values
        rating = grp.rating.values.astype(np.float64)
        embed = grp[embed_cols].values.astype(np.float64)
        y = grp.success.values.astype(np.float64)
        for idx in chunk_user(times):
            chunks.append({
                "user": user,
                "n": len(idx),
                "start": pd.Timestamp(times[idx[0]]),
                "end": pd.Timestamp(times[idx[-1]]),
                "y": y[idx],
                "rating": rating[idx],
                "embed": embed[idx],
            })
    return chunks


def analysis_a(chunks):
    """Per-chunk logistic( success ~ 1 + rating ). Aggregate slope stats."""
    print("\n[A] Per-chunk fit: success ~ 1 + rating")
    rows = []
    for c in chunks:
        if c["y"].std() < 1e-3 or c["rating"].std() < 1e-3:
            continue
        X = np.column_stack([np.ones(c["n"]), c["rating"]])
        beta, ok = fit_logistic(X, c["y"], lam_vec=np.zeros(2))
        if not ok or not np.all(np.isfinite(beta)):
            continue
        if abs(beta[1]) > 200.0:  # implausibly steep PWA-unit slope
            continue
        rows.append((c["user"], beta[0], beta[1], c["n"]))
    bd = pd.DataFrame(rows, columns=["user", "intercept", "beta_diff", "n"])
    abs_slope = bd.beta_diff.abs()
    per_rank_signed = bd.beta_diff / 29.0
    per_rank_abs = abs_slope / 29.0
    print(f"  usable chunks: {len(bd)}")
    print(f"  in logit/PWA-unit space (the scale problemRating uses in the PWA):")
    print(f"    signed beta_diff: mean={bd.beta_diff.mean():+.3f}  median={bd.beta_diff.median():+.3f}  std={bd.beta_diff.std():.3f}")
    print(f"    |beta_diff|:      mean={abs_slope.mean():.3f}  median={abs_slope.median():.3f}  std={abs_slope.std():.3f}")
    print(f"  in logit/rank space (multiply by 1/29):")
    print(f"    signed beta_diff: mean={per_rank_signed.mean():+.4f}  median={per_rank_signed.median():+.4f}  std={per_rank_signed.std():.4f}")
    print(f"    |beta_diff|:      mean={per_rank_abs.mean():.4f}  median={per_rank_abs.median():.4f}  std={per_rank_abs.std():.4f}")
    print(f"  --> suggested LOGIT_PER_RANK_PRIOR       = {per_rank_abs.mean():.3f}    (currently 0.28)")
    print(f"  --> suggested LOGIT_PER_RANK_PRIOR_SIGMA = {per_rank_abs.std():.3f}    (currently 0.1)")
    print(f"  --> suggested L2 target for beta_diff    = {bd.beta_diff.mean():+.2f}   (PWA-unit space; currently -10)")
    return {
        "n_chunks": int(len(bd)),
        "logit_per_rank_prior":            float(per_rank_abs.mean()),
        "logit_per_rank_prior_sigma":      float(per_rank_abs.std()),
        "logit_per_rank_prior_median":     float(per_rank_abs.median()),
        "difficulty_l2_target_signed_mean_pwa_unit":   float(bd.beta_diff.mean()),
        "difficulty_l2_target_signed_median_pwa_unit": float(bd.beta_diff.median()),
        "beta_diff_pwa_unit_std":          float(bd.beta_diff.std()),
    }


def analysis_b(chunks):
    """5-fold CV per chunk over a grid of L2 on the 64 PC beta_k's only."""
    print(f"\n[B] 5-fold CV over lambda grid: {LAMBDA_GRID}")
    rng = np.random.default_rng(SEED)
    total_nll = {lam: 0.0 for lam in LAMBDA_GRID}
    total_n = {lam: 0 for lam in LAMBDA_GRID}

    for ci, c in enumerate(chunks):
        n = c["n"]
        if c["y"].std() < 1e-3:
            continue
        X = np.column_stack([np.ones(n), c["rating"], c["embed"]])
        y = c["y"]
        idx = np.arange(n)
        rng.shuffle(idx)
        folds = np.array_split(idx, CV_FOLDS)
        d = X.shape[1]
        # warm-starts across lambda values (descending lambda -> looser, beta grows)
        beta_ws = {k: np.zeros(d) for k in range(CV_FOLDS)}
        for lam in sorted(LAMBDA_GRID, reverse=True):
            lam_vec = np.zeros(d)
            lam_vec[2:] = lam  # intercept (0) and rating (1) unpenalized
            for k in range(CV_FOLDS):
                te = folds[k]
                tr_mask = np.ones(n, dtype=bool)
                tr_mask[te] = False
                Xtr, ytr = X[tr_mask], y[tr_mask]
                Xte, yte = X[te], y[te]
                if ytr.std() < 1e-3 or len(te) == 0:
                    continue
                beta, ok = fit_logistic(Xtr, ytr, lam_vec, beta0=beta_ws[k])
                if not ok or not np.all(np.isfinite(beta)):
                    continue
                beta_ws[k] = beta
                total_nll[lam] += held_out_nll(beta, Xte, yte)
                total_n[lam] += len(te)
        if (ci + 1) % 25 == 0 or ci + 1 == len(chunks):
            print(f"  ... {ci+1}/{len(chunks)} chunks")

    print(f"\n  {'lambda':>8}  {'total NLL':>12}  {'per-attempt NLL':>16}  {'held-out attempts':>18}")
    summary = []
    for lam in LAMBDA_GRID:
        n_tot = total_n[lam]
        per = total_nll[lam] / max(1, n_tot)
        summary.append((lam, total_nll[lam], n_tot, per))
        print(f"  {lam:>8.3g}  {total_nll[lam]:>12.2f}  {per:>16.5f}  {n_tot:>18d}")
    best = min(summary, key=lambda r: r[3])
    print(f"  --> best lambda = {best[0]:g}  (per-attempt NLL = {best[3]:.5f})")
    print(f"  --> suggested RIDGE_LAMBDA = {best[0]:g}")
    return {
        "lambda_grid": LAMBDA_GRID,
        "per_attempt_nll": {str(r[0]): float(r[3]) for r in summary},
        "ridge_lambda_cv": float(best[0]),
    }


def analysis_d(chunks, lam):
    """Option-C identifiability diagnostics.

    The PWA's weakness model is sigma(alpha + gamma*rating + sum_k beta_k*embed_k).
    The PCs are constructed to be rating-orthogonal across the corpus, but within
    one user's chunk the rating distribution is non-random (selector targets
    near user rank), so (embed_k, rating) sample correlation can be > 0 and
    rating signal can leak into beta_k. We check:
      - within-chunk |corr(embed_k, rating)| distribution (leakage risk)
      - SE(gamma) and |gamma|/SE   (is gamma well-identified to soak up rating?)
      - stratification by chunk size and rating span (cold-start failure mode)
    """
    print(f"\n[D] Option-C identifiability diagnostics (lambda={lam:g}, gamma unpenalized)")
    rows = []
    for c in chunks:
        n = c["n"]
        rating = c["rating"]
        embed = c["embed"]
        y = c["y"]
        if rating.std() < 1e-6 or y.std() < 1e-3:
            continue
        corrs = []
        for k in range(K_EMBED):
            ek = embed[:, k]
            if ek.std() < 1e-6:
                continue
            corrs.append(abs(np.corrcoef(rating, ek)[0, 1]))
        if not corrs:
            continue
        X = np.column_stack([np.ones(n), rating, embed])
        lam_vec = np.zeros(X.shape[1])
        lam_vec[2:] = lam
        beta, ok = fit_logistic(X, y, lam_vec)
        if not ok or not np.all(np.isfinite(beta)):
            continue
        p = expit(X @ beta)
        XtWX = (X * (p * (1 - p))[:, None]).T @ X
        H = XtWX + np.diag(lam_vec)
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue
        se_gamma = float(np.sqrt(max(cov[1, 1], 1e-12)))
        rows.append({
            "n": int(n),
            "rating_span": float(rating.max() - rating.min()),
            "max_abs_corr": float(max(corrs)),
            "mean_abs_corr": float(np.mean(corrs)),
            "gamma": float(beta[1]),
            "se_gamma": se_gamma,
            "abs_z_gamma": abs(beta[1]) / se_gamma,
        })

    if not rows:
        print("  no usable chunks")
        return {}
    df = pd.DataFrame(rows)
    print(f"  chunks analyzed: {len(df)}")
    print(f"\n  Within-chunk |corr(embed_k, rating)| across 64 PCs:")
    print(f"    per-chunk max:  mean={df.max_abs_corr.mean():.3f}  median={df.max_abs_corr.median():.3f}  p90={df.max_abs_corr.quantile(0.9):.3f}")
    print(f"    per-chunk mean: mean={df.mean_abs_corr.mean():.3f}  median={df.mean_abs_corr.median():.3f}")
    print(f"\n  Gamma (rating slope) identifiability in PWA-unit space:")
    print(f"    SE(gamma):           mean={df.se_gamma.mean():.2f}  median={df.se_gamma.median():.2f}  p90={df.se_gamma.quantile(0.9):.2f}")
    print(f"    |gamma| / SE(gamma): mean={df.abs_z_gamma.mean():.2f}  median={df.abs_z_gamma.median():.2f}  p10={df.abs_z_gamma.quantile(0.1):.2f}")
    print(f"    fraction with |z(gamma)| > 2 (decent identification): {(df.abs_z_gamma > 2).mean():.2%}")
    print(f"\n  Stratified by chunk size:")
    for lo, hi in [(100, 200), (200, 500), (500, 10**9)]:
        sub = df[(df.n >= lo) & (df.n < hi)]
        if len(sub) == 0:
            continue
        print(f"    n in [{lo},{hi if hi < 10**9 else 'inf'}): {len(sub):>3d} chunks  "
              f"max|corr| median={sub.max_abs_corr.median():.3f}  "
              f"SE(gamma) median={sub.se_gamma.median():.2f}  "
              f"|z(gamma)| median={sub.abs_z_gamma.median():.2f}  "
              f"rating span median={sub.rating_span.median():.3f}")
    print(f"\n  Stratified by within-chunk rating span (PWA units):")
    for lo, hi in [(0.0, 0.1), (0.1, 0.2), (0.2, 1.0)]:
        sub = df[(df.rating_span >= lo) & (df.rating_span < hi)]
        if len(sub) == 0:
            continue
        print(f"    span in [{lo:.2f},{hi:.2f}): {len(sub):>3d} chunks  "
              f"max|corr| median={sub.max_abs_corr.median():.3f}  "
              f"SE(gamma) median={sub.se_gamma.median():.2f}  "
              f"|z(gamma)| median={sub.abs_z_gamma.median():.2f}")
    return {
        "n_chunks": int(len(df)),
        "max_abs_corr_median": float(df.max_abs_corr.median()),
        "max_abs_corr_p90":    float(df.max_abs_corr.quantile(0.9)),
        "gamma_se_median":     float(df.se_gamma.median()),
        "abs_z_gamma_median":  float(df.abs_z_gamma.median()),
        "frac_z_gamma_gt_2":   float((df.abs_z_gamma > 2).mean()),
    }


def analysis_c(chunks, best_lambda):
    """At chosen lambda, fit each chunk and inspect dispersion of PC betas."""
    print(f"\n[C] Empirical-Bayes consistency at lambda={best_lambda:g}")
    Bs = []
    for c in chunks:
        n = c["n"]
        if c["y"].std() < 1e-3:
            continue
        X = np.column_stack([np.ones(n), c["rating"], c["embed"]])
        lam_vec = np.zeros(X.shape[1])
        lam_vec[2:] = best_lambda
        beta, ok = fit_logistic(X, c["y"], lam_vec)
        if not ok or not np.all(np.isfinite(beta)):
            continue
        Bs.append(beta[2:])
    B = np.array(Bs)
    pooled_std = float(B.std(ddof=1))
    per_dim_std = B.std(axis=0, ddof=1)
    median_dim_std = float(np.median(per_dim_std))
    implied_lambda_pooled = 1.0 / (pooled_std ** 2)
    implied_lambda_median = 1.0 / (median_dim_std ** 2)
    print(f"  chunks fit: {len(B)}")
    print(f"  pooled std(beta_k) over all (chunk,k) = {pooled_std:.4f}  -> implies lambda ~ {implied_lambda_pooled:.2f}")
    print(f"  median per-dim std(beta_k)            = {median_dim_std:.4f}  -> implies lambda ~ {implied_lambda_median:.2f}")
    if not 0.5 <= implied_lambda_pooled / best_lambda <= 2.0:
        print("  WARNING: CV-best lambda and empirical-Bayes lambda disagree by > 2x.")
        print("  Posterior shrinkage is biasing one estimate or the prior is mis-shaped.")
    return {
        "ridge_lambda_eb_pooled": float(implied_lambda_pooled),
        "ridge_lambda_eb_median_per_dim": float(implied_lambda_median),
        "pooled_std_beta_k": pooled_std,
        "median_per_dim_std_beta_k": median_dim_std,
        "chunks_fit": int(len(B)),
    }


def main():
    global MIN_ATTEMPTS_PER_CHUNK
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "auxiliary", "calibrate_priors.json"),
    )
    ap.add_argument("--min-attempts", type=int, default=MIN_ATTEMPTS_PER_CHUNK)
    args = ap.parse_args()
    MIN_ATTEMPTS_PER_CHUNK = args.min_attempts

    print("loading attempts...")
    df = parse_db(args.db)
    print(f"  {len(df)} attempts, {df.user_id.nunique()} users")

    print("loading problem features (rating + 64-d embedding)...")
    feats, embed_cols = load_problem_features()
    print(f"  {len(feats)} problems with both rating and embedding")

    df = df.merge(feats, on=["tsumego_id", "size"], how="inner")
    print(f"  joined: {len(df)} attempts, {df.user_id.nunique()} users")

    chunks = build_chunks(df, embed_cols)
    print(f"  {len(chunks)} chunks (>= {MIN_ATTEMPTS_PER_CHUNK} attempts, "
          f"{WINDOW_DAYS}d window, {STRIDE_DAYS}d stride)")
    if not chunks:
        print("no chunks; nothing to calibrate")
        sys.exit(0)

    a = analysis_a(chunks)
    b = analysis_b(chunks)
    c = analysis_c(chunks, b["ridge_lambda_cv"])
    d = analysis_d(chunks, b["ridge_lambda_cv"])

    out = {
        "config": {
            "window_days": WINDOW_DAYS,
            "stride_days": STRIDE_DAYS,
            "min_attempts_per_chunk": MIN_ATTEMPTS_PER_CHUNK,
            "cv_folds": CV_FOLDS,
            "k_embed": K_EMBED,
            "lambda_grid": LAMBDA_GRID,
            "n_total_chunks": len(chunks),
        },
        "caveat": (
            "Overlapping chunks (60d window, 30d stride) are not independent. "
            "Population std reported here is an upper bound; use the CV-best "
            "lambda as the primary recommendation."
        ),
        "analysis_a_difficulty_slope": a,
        "analysis_b_cv_ridge": b,
        "analysis_c_eb_consistency": c,
        "analysis_d_identifiability": d,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
