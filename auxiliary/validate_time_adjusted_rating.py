"""Can the per-user solving-time signal (the quantity fit_population_constants
derives) be used to ADJUST a user's rating — and does that adjustment improve
agreement with survey-declared ranks?

Hypothesis: onlymove is untimed while server ranks reflect timed play, so a user
who leans on long thinking time should clear more problems here than their
server rank implies (over-rated), and a fast solver the reverse. If true, a
correction  rating' = rating + alpha * time_feature  should raise the Spearman
correlation between user ratings and declared ranks.

Time features tested (per survey user, novel timed attempts only):
  med_time   — median solving time in minutes (AFK-capped at 180 s), log10.
               The "absolute pace" signal: between-user time usage.
  time_coeff — the per-user 3PL-controlled time coefficient in logit/min
               (fit_time_coeff_3pl from pipeline.fit_population_constants, the
               exact quantity whose population mean becomes time_bias_per_min).
               Within-user signal: does THIS user's success co-move with time.

Validation metric (what the user asked for): per-server Spearman between user
rating and declared rank scalar, combined as a user-count-weighted mean over
servers with >= --min-server-users declarations. The declared rank is the most
recent per (user, server).

Three views, increasingly honest:
  1. Diagnostic: weighted partial Spearman of (declared rank, feature)
     controlling for rating — is there ANY residual time signal in the ranks?
  2. Full-data alpha sweep: weighted Spearman of rating + alpha*feature over an
     alpha grid (feature standardized; alpha in rating units per feature-sd).
     Upper bound — alpha is chosen on the evaluation data.
  3. Cross-fitted: K-fold over users, alpha picked on train folds only, each
     user adjusted with an alpha that never saw them, weighted Spearman over the
     assembled set; repeated over fold seeds. This is the deployable estimate.

Reads attempts/database.sqlite + backend ratings_v2/ (same sources as
pipeline.fit_population_constants). No GPU, no network.

Run:
  ./.venv/bin/python3 -m auxiliary.validate_time_adjusted_rating
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import rank_to_scalar  # noqa: E402
from pipeline.fit_population_constants import (  # noqa: E402
    AFK_CAP_S, FLOOR_MAX, fit_ability, fit_time_coeff_3pl, load_problem_params,
    midrank_pct,
)
from utils.fit_rank_curves import fit_user_rating  # noqa: E402

DB_DEFAULT = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
V2_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "ratings_v2")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_surveys(con):
    """Latest declaration per (user, server) with a parseable rank.
    Returns list of (user_id, server, rank_scalar)."""
    rows = con.execute(
        "SELECT user_id, server, rank, created_at FROM rank_surveys "
        "ORDER BY created_at"
    ).fetchall()
    latest = {}
    for user, server, rank, _ in rows:
        sc = rank_to_scalar(rank)
        if sc is not None:
            latest[(user, server)] = sc
    return [(u, s, sc) for (u, s), sc in latest.items()]


def load_novel_attempts(con, users):
    """{user: [(size, tid, success, time_used_ms)]} over first-time attempts."""
    qmarks = ",".join("?" * len(users))
    rows = con.execute(
        f"SELECT user_id, problem, success, size, time_used FROM tsumego_reports "
        f"WHERE is_repeat = 0 AND user_id IN ({qmarks}) ORDER BY created_at",
        list(users),
    ).fetchall()
    out = defaultdict(list)
    for user, tid, success, size, time_used in rows:
        out[user].append((size, tid, int(success), time_used))
    return out


# ---------------------------------------------------------------------------
# Per-user features
# ---------------------------------------------------------------------------
def user_time_features(attempts, params, min_timed):
    """(median_log10_minutes, time_coeff_logit_per_min) — either may be NaN."""
    timed = [(t_ms, y, params[(size, tid)])
             for size, tid, y, t_ms in attempts
             if t_ms and 0 < t_ms <= AFK_CAP_S * 1000 and (size, tid) in params]
    if len(timed) < min_timed:
        return np.nan, np.nan
    ms = np.array([t for t, _, _ in timed], float)
    med_log_min = float(np.log10(np.median(ms) / 60000.0))

    y = np.array([y for _, y, _ in timed], float)
    rating = np.array([p[0] for _, _, p in timed])
    slope = np.array([p[1] for _, _, p in timed])
    floor = np.clip(np.array([p[2] for _, _, p in timed]), 0, FLOOR_MAX)
    coeff = np.nan
    if y.std() > 0:
        sm = np.sort(ms)
        n = len(sm)
        # central 10-percentile width in minutes -> percentile units per minute
        tppm = (sm[int(np.ceil(0.55 * (n - 1)))]
                - sm[int(np.floor(0.45 * (n - 1)))]) / (60000 * 0.10)
        pct = np.array([midrank_pct(list(sm), t) for t in ms])
        if tppm > 0 and pct.std() > 1e-6:
            c, se = fit_time_coeff_3pl(rating, slope, floor, pct, y)
            if np.isfinite(se) and se > 0:
                coeff = c / tppm
    return med_log_min, coeff


# ---------------------------------------------------------------------------
# Weighted Spearman
# ---------------------------------------------------------------------------
def weighted_spearman(table, ratings, min_server_users):
    """table: list of (user_idx, server, rank_scalar). ratings: per-user array.
    Per-server Spearman(rating, rank), weighted mean by user count."""
    by_server = defaultdict(list)
    for ui, server, sc in table:
        by_server[server].append((ratings[ui], sc))
    total_w, acc, per_server = 0.0, 0.0, {}
    for server, pairs in by_server.items():
        if len(pairs) < min_server_users:
            continue
        r = np.array([p[0] for p in pairs])
        s = np.array([p[1] for p in pairs])
        if np.ptp(r) == 0 or np.ptp(s) == 0:
            continue
        rho = stats.spearmanr(r, s).statistic
        if not np.isfinite(rho):
            continue
        per_server[server] = (rho, len(pairs))
        acc += rho * len(pairs)
        total_w += len(pairs)
    return (acc / total_w if total_w else np.nan), per_server


def partial_spearman_pooled(table, ratings, feature, min_server_users):
    """Weighted pooled partial Spearman corr(rank, feature | rating), per server
    on rank-transformed data; weights n-3; rough p via Fisher z."""
    by_server = defaultdict(list)
    for ui, server, sc in table:
        if np.isfinite(feature[ui]):
            by_server[server].append((ratings[ui], sc, feature[ui]))
    zs, ws = [], []
    for server, rows in by_server.items():
        if len(rows) < min_server_users:
            continue
        arr = np.array(rows, float)
        rr = stats.rankdata(arr[:, 0])
        rs = stats.rankdata(arr[:, 1])
        rf = stats.rankdata(arr[:, 2])
        if np.ptp(rs) == 0 or np.ptp(rf) == 0 or np.ptp(rr) == 0:
            continue
        # residualize rank & feature on rating (in rank space), then correlate
        def resid(v):
            b = np.polyfit(rr, v, 1)
            return v - np.polyval(b, rr)
        e_s, e_f = resid(rs), resid(rf)
        if e_s.std() < 1e-9 or e_f.std() < 1e-9:
            continue
        rho = float(np.corrcoef(e_s, e_f)[0, 1])
        rho = np.clip(rho, -0.999, 0.999)
        zs.append(np.arctanh(rho))
        ws.append(len(rows) - 3)
    if not zs:
        return np.nan, np.nan, 0
    zs, ws = np.array(zs), np.array(ws, float)
    z_pool = float(np.sum(ws * zs) / np.sum(ws))
    se = 1.0 / np.sqrt(np.sum(ws))
    p = 2 * (1 - stats.norm.cdf(abs(z_pool) / se))
    return float(np.tanh(z_pool)), float(p), int(np.sum(ws + 3))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--ratings-dir", default=V2_DIR_DEFAULT)
    ap.add_argument("--min-attempts", type=int, default=10,
                    help="min novel attempts for a usable baseline rating")
    ap.add_argument("--min-timed", type=int, default=20,
                    help="min timed novel attempts for the time features")
    ap.add_argument("--min-server-users", type=int, default=5)
    ap.add_argument("--alpha-max", type=float, default=0.5,
                    help="sweep half-range, rating units per feature sd")
    ap.add_argument("--alpha-steps", type=int, default=51)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-seeds", type=int, default=20)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    surveys = load_surveys(con)
    users = sorted({u for u, _, _ in surveys})
    print(f"{len(surveys)} (user, server) declarations from {len(users)} users")

    attempts = load_novel_attempts(con, users)
    con.close()
    needed = {(size, tid) for rows in attempts.values()
              for size, tid, _, _ in rows}
    params = load_problem_params(args.ratings_dir, needed)
    print(f"3PL params resolved for {len(params)}/{len(needed)} problems")

    # --- per-user baseline rating + time features ---
    raw_ratings = {k: v[0] for k, v in params.items()}
    slopes = {k: v[1] for k, v in params.items()}
    user_idx, R, f_med, f_coef = {}, [], [], []
    for u in users:
        rows = [r for r in attempts.get(u, []) if (r[0], r[1]) in params]
        if len(rows) < args.min_attempts:
            continue
        att = [((size, tid), y) for size, tid, y, _ in rows]
        rating, n = fit_user_rating(att, raw_ratings, slopes=slopes,
                                    min_n=args.min_attempts)
        if rating is None:
            continue
        med, coef = user_time_features(rows, params, args.min_timed)
        user_idx[u] = len(R)
        R.append(rating)
        f_med.append(med)
        f_coef.append(coef)
    R = np.array(R)
    feats = {"med_time(log10 min)": np.array(f_med),
             "time_coeff(logit/min)": np.array(f_coef)}
    print(f"{len(R)} survey users with >= {args.min_attempts} rated novel attempts")
    for name, f in feats.items():
        ok = np.isfinite(f)
        print(f"  {name}: {ok.sum()} users with feature "
              f"(mean {np.nanmean(f):+.3f}, sd {np.nanstd(f):.3f})")

    table = [(user_idx[u], server, sc) for u, server, sc in surveys
             if u in user_idx]
    print(f"{len(table)} usable (user, server) pairs")

    base, per_server = weighted_spearman(table, R, args.min_server_users)
    print(f"\nBASELINE weighted Spearman = {base:.4f}")
    for server, (rho, n) in sorted(per_server.items(), key=lambda kv: -kv[1][1]):
        print(f"  {server:8s} rho={rho:+.3f}  (n={n})")

    alphas = np.linspace(-args.alpha_max, args.alpha_max, args.alpha_steps)
    rng_global = np.random.default_rng(0)

    for name, f_raw in feats.items():
        ok = np.isfinite(f_raw)
        sd = np.nanstd(f_raw)
        f = np.where(ok, (f_raw - np.nanmean(f_raw)) / (sd if sd > 0 else 1), 0.0)

        rho_pf, p_pf, n_pf = partial_spearman_pooled(table, R, f_raw,
                                                     args.min_server_users)
        print(f"\n=== {name} ===")
        print(f"  partial Spearman(rank, feature | rating) = {rho_pf:+.3f} "
              f"(p={p_pf:.3f}, n={n_pf}) "
              f"[{'feature carries residual rank signal' if p_pf < 0.05 else 'no significant residual signal'}]")

        # full-data sweep (optimistic upper bound)
        curve = np.array([weighted_spearman(table, R + a * f,
                                            args.min_server_users)[0]
                          for a in alphas])
        i_best = int(np.nanargmax(curve))
        print(f"  full-data sweep: best alpha={alphas[i_best]:+.3f} "
              f"rating/sd -> weighted Spearman {curve[i_best]:.4f} "
              f"(delta {curve[i_best] - base:+.4f}); alpha=0 -> {curve[len(alphas)//2]:.4f}")

        # cross-fitted (honest) estimate
        n_users = len(R)
        cv_scores = []
        for seed in range(args.cv_seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_users)
            fold_of = np.empty(n_users, int)
            for k in range(args.folds):
                fold_of[perm[k::args.folds]] = k
            adj = R.copy()
            for k in range(args.folds):
                train_users = set(np.where(fold_of != k)[0])
                sub = [(ui, s, sc) for ui, s, sc in table if ui in train_users]
                sweep = [weighted_spearman(sub, R + a * f,
                                           args.min_server_users)[0]
                         for a in alphas]
                a_star = alphas[int(np.nanargmax(sweep))]
                test = fold_of == k
                adj[test] = R[test] + a_star * f[test]
            cv_scores.append(weighted_spearman(table, adj,
                                               args.min_server_users)[0])
        cv_scores = np.array(cv_scores)
        frac_better = float(np.mean(cv_scores > base))
        print(f"  cross-fitted ({args.folds}-fold x {args.cv_seeds} seeds): "
              f"weighted Spearman {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} "
              f"(delta {cv_scores.mean() - base:+.4f}; "
              f"{frac_better:.0%} of seeds beat baseline)")

    print("\nInterpretation guide: the cross-fitted delta is the deployable "
          "improvement; the full-data sweep is an upper bound (alpha tuned on "
          "the evaluation data); the partial Spearman tells you whether the "
          "feature has ANY residual association with declared rank at all.")


if __name__ == "__main__":
    main()
