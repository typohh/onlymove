"""Paired held-out CV: does the consensus rating→rank fit gain from separating the
9x9 / 13x13 solve-time effects?

Production (pipeline.map_rating_to_rank) time-adjusts each anchor's rating by
`alpha * (log10 pooled median solve min - center)` — the median pools both board
sizes. But 13x13 problems take ~1.3x longer per user (paired log10 gap +0.11,
sd 0.22 over dual-size survey users) and users' size mix spans 0..1, so the pooled
median partially reads "plays 13x13" as "slow player". Variants (all on the
production blocks-mass anchors; they differ ONLY in the time features):

  notime         no time adjustment (control: how much does time help at all?)
  pooled         production baseline — one pooled log10 median, one alpha.
  size-adj       one feature, one alpha: per-size log10 medians each de-meaned by
                 its own population center, combined per user by timed-attempt-count
                 weights. Same dof as pooled, size-mix confound removed.
  two-alpha      two features/alphas: alpha9*T9 + alpha13*T13 (a missing size's
                 slot is 0 = population center).
  pooled+frac13  pooled time + the user's 13x13 timed-attempt fraction (de-meaned)
                 as a second covariate — isolates whether the size MIX itself
                 carries rank signal beyond the pooled time.

Feature availability mirrors production: a user is "timed" iff their pooled
working-set count >= MIN_TIMED (20); per-size features additionally need
MIN_TIMED_SIZE (10) timed attempts on that size. Centers are computed once on the
full user set (production behavior; identical across variants, the small leak
cancels in paired diffs). Untimed users get all-zero features in every variant.

Protocol: identical to auxiliary.eval_rank_map_cv — 5-fold CV by USER x reps,
every variant fit and scored on the SAME foldings (paired). Held-out latest
declarations scored point-wise with the career rating and the variant's own
alphas/features. Baseline for the paired diffs is `pooled` (production).

Usage: ./.venv/bin/python3 -m auxiliary.eval_rank_map_time_by_size [--reps 6] [--folds 5]
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.eval_rank_map_cv import (  # noqa: E402
    RatingFitter, block_masses, load_attempts_sqlite, user_blocks,
)
from pipeline.map_rating_to_rank import (  # noqa: E402
    AFK_CAP_MS, GAUGE_WEIGHT, LAMBDA_SHAPE, LAMBDA_TIME, MIN_ATTEMPTS,
    MIN_SERVER_SAMPLES, MIN_TIMED, RANK_PADDING, EXTEND_TO, SERVER_LAMBDA, SIZES,
    WORKING_SET, _curve_arrays, load_floors, load_ratings, load_slopes,
)
from utils.fit_rank_curves import interp_extrap_vec, _unpack_thresholds  # noqa: E402
from utils.rank_surveys import SQL_PATH_DEFAULT, load_rank_surveys  # noqa: E402
from utils.ratings_io import load_lapse  # noqa: E402

MIN_TIMED_SIZE = 10


# ---------------------------------------------------------------------------
# Per-user solve-time stats, split by board size (same working-set window +
# filters as production load_user_median_solve_minutes, but keeping `size`).
# ---------------------------------------------------------------------------
def load_user_time_stats(sqlite_path):
    """{user_prefix8: {"pooled": log10 med min or None,
                       "by_size": {9: (log10 med min, n_timed), 13: ...},
                       "frac13": fraction of timed attempts on 13x13}}
    over each user's most-recent WORKING_SET attempts; pooled needs MIN_TIMED,
    a size slot needs MIN_TIMED_SIZE."""
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.execute("SELECT user_id, is_repeat, time_used, size FROM tsumego_reports "
                          "ORDER BY created_at, id")
        windows = defaultdict(lambda: deque(maxlen=WORKING_SET))
        for uid, rep, t, sz in cur:
            windows[uid[:8]].append((rep, t, sz))
    finally:
        con.close()
    out = {}
    for pref, rows in windows.items():
        ts = [(float(t), sz) for (rep, t, sz) in rows
              if not rep and t is not None and 0 < float(t) <= AFK_CAP_MS]
        if len(ts) < MIN_TIMED:
            continue
        allts = [t for t, _ in ts]
        by_size = {}
        for sz in SIZES:
            sts = [t for t, s in ts if s == sz]
            if len(sts) >= MIN_TIMED_SIZE:
                by_size[sz] = (float(np.log10(np.median(sts) / 60000.0)), len(sts))
        out[pref] = {"pooled": float(np.log10(np.median(allts) / 60000.0)),
                     "by_size": by_size,
                     "frac13": float(np.mean([s == 13 for _, s in ts]))}
    return out


def build_feature_defs(users, time_stats):
    """Per-variant per-user feature vectors. Returns {variant: ({uid: np.array(F)}, F)}.
    Centers are population means over the timed users (full-data, like production)."""
    timed = [u for u in users if u[:8] in time_stats]
    c_pool = float(np.mean([time_stats[u[:8]]["pooled"] for u in timed])) if timed else 0.0
    c_size = {}
    for sz in SIZES:
        vals = [time_stats[u[:8]]["by_size"][sz][0] for u in timed
                if sz in time_stats[u[:8]]["by_size"]]
        c_size[sz] = float(np.mean(vals)) if vals else 0.0
    f13 = [time_stats[u[:8]]["frac13"] for u in timed]
    c_f13 = float(np.mean(f13)) if f13 else 0.0

    T_pool, T_adj, T_two, T_pf = {}, {}, {}, {}
    for u in users:
        st = time_stats.get(u[:8])
        if st is None:
            T_pool[u] = np.zeros(1); T_adj[u] = np.zeros(1)
            T_two[u] = np.zeros(2); T_pf[u] = np.zeros(2)
            continue
        T_pool[u] = np.array([st["pooled"] - c_pool])
        dem = {sz: st["by_size"][sz][0] - c_size[sz] for sz in st["by_size"]}
        if dem:
            wts = np.array([st["by_size"][sz][1] for sz in st["by_size"]], float)
            T_adj[u] = np.array([np.average(list(dem.values()), weights=wts)])
        else:
            T_adj[u] = np.zeros(1)
        T_two[u] = np.array([dem.get(sz, 0.0) for sz in SIZES])
        T_pf[u] = np.array([st["pooled"] - c_pool, st["frac13"] - c_f13])
    zero1 = {u: np.zeros(1) for u in users}
    return {
        "notime":        (zero1, 1),
        "pooled":        (T_pool, 1),
        "size-adj":      (T_adj, 1),
        "two-alpha":     (T_two, 2),
        "pooled+frac13": (T_pf, 2),
    }, dict(c_pool=c_pool, c_size=c_size, c_f13=c_f13, n_timed=len(timed))


# ---------------------------------------------------------------------------
# Consensus fit with a VECTOR time feature (se=0 path of eval_rank_map_cv's
# fit_consensus_cv, alpha generalized to length-F; F=1 + pooled T reproduces it).
# ---------------------------------------------------------------------------
def _bucket_vec(decls, F):
    rows_by = defaultdict(list)
    for server, R, g, T, w in decls:
        rows_by[server].append((R, int(round(g)), T, w))
    blocks = {}
    for server, rows in rows_by.items():
        if sum(w for *_, w in rows) < MIN_SERVER_SAMPLES:
            continue
        declared = [g for _, g, _, _ in rows]
        g_min = min(declared) - RANK_PADDING
        g_max = max(max(declared) + RANK_PADDING, EXTEND_TO)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        gs = list(range(g_min + 1, g_max + 1))
        blocks[server] = {
            "g_min": g_min, "g_max": g_max,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "identity": np.array([g - 0.5 for g in gs], float),
            "rating": np.array([r for r, _, _, _ in rows], float),
            "T": np.array([t for _, _, t, _ in rows], float).reshape(len(rows), F),
            "w": np.array([w for _, _, _, w in rows], float),
            "k": np.array([max(0, min(N - 1, g - g_min)) for _, g, _, _ in rows], int)}
    return blocks


def fit_consensus_vec(decls, F):
    """decls: [(server, R, g, Tvec, w)]. Returns (a, alpha_vec, ys, server_fits)."""
    egf_thr_ranks, ys, S, dS = _curve_arrays()
    nseg = len(dS)
    blocks = _bucket_vec(decls, F)
    if not blocks:
        return None

    all_R = np.array([R for _, R, _, _, _ in decls], float)
    all_g = np.array([g for _, _, g, _, _ in decls], float)
    beta, alpha_c = np.polyfit(all_R, all_g, 1)
    if beta <= 0:
        beta = 1e-3
    prior_rating = (ys - alpha_c) / beta
    b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
    a0 = float(prior_rating[0] - b0 * S[0])

    head = [a0, float(np.log(max(b0, 1e-3)))] + [0.0] * nseg + [0.0] * F
    srv_layout, x0, idx = [], head, len(head)
    for server, blk in blocks.items():
        n = len(blk["baseline"]) + 1
        srv_layout.append((server, idx, n))
        x0.append(float(blk["g_min"]))
        x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, float)

    def unpack_curve(params):
        beta_ = np.exp(params[1])
        a = np.empty(len(S))
        a[0] = params[0] + beta_ * S[0]
        if nseg:
            a[1:] = a[0] + np.cumsum(beta_ * dS * np.exp(params[2:2 + nseg]))
        return a, params[2 + nseg:2 + nseg + F]

    def objective(params):
        a, alpha = unpack_curve(params)
        nll, pen, w_off, w_tot = 0.0, 0.0, 0.0, 0.0
        for server, st, n in srv_layout:
            blk = blocks[server]
            thr = _unpack_thresholds(params[st:st + n], blk["baseline"])
            x_int = interp_extrap_vec(blk["rating"] + blk["T"] @ alpha, a, ys)
            K = thr.shape[0]
            kk = blk["k"]
            upper = np.where(kk < K, expit(thr[np.minimum(kk, K - 1)] - x_int), 1.0)
            lower = np.where(kk > 0, expit(thr[np.maximum(kk - 1, 0)] - x_int), 0.0)
            p = np.clip(upper - lower, 1e-12, 1.0)
            nll += -float(np.sum(blk["w"] * np.log(p)))
            pen += SERVER_LAMBDA * float(np.sum(params[st + 1:st + n] ** 2))
            wsrv = float(np.sum(blk["w"]))
            w_off += wsrv * float(np.mean(thr - blk["identity"]))
            w_tot += wsrv
        if nseg:
            pen += LAMBDA_SHAPE * float(np.sum(params[2:2 + nseg] ** 2))
        pen += LAMBDA_TIME * float(np.sum(alpha ** 2))
        pen += GAUGE_WEIGHT * (w_off / w_tot) ** 2
        return nll + pen

    with np.errstate(divide="ignore", invalid="ignore"):
        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 10000})
    a, alpha = unpack_curve(res.x)
    server_fits = {}
    for server, st, n in srv_layout:
        blk = blocks[server]
        thr = _unpack_thresholds(res.x[st:st + n], blk["baseline"])
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        server_fits[server] = {gs[i]: float(thr[i]) for i in range(len(gs))}
    return a, np.asarray(alpha, float), ys, server_fits


def evaluate(fitres, eval_rows):
    """eval_rows: [(server, R, g, Tvec)]. Returns (nll[], err[], g[])."""
    a, alpha, ys, server_fits = fitres
    nlls, errs, gdecl = [], [], []
    for server, R, g, T in eval_rows:
        thr = server_fits.get(server)
        if not thr or len(thr) < 2:
            continue
        gs = sorted(thr)
        tau = np.array([thr[gg] for gg in gs], float)
        g_min = gs[0] - 1
        N = len(gs) + 1
        k = max(0, min(N - 1, int(round(g)) - g_min))
        x = float(interp_extrap_vec(np.array([R + float(T @ alpha)]), a, ys)[0])
        K = tau.shape[0]
        upper = expit(tau[min(k, K - 1)] - x) if k < K else 1.0
        lower = expit(tau[max(k - 1, 0)] - x) if k > 0 else 0.0
        nlls.append(-np.log(max(upper - lower, 1e-12)))
        order = np.argsort(tau)
        srv_scalar = np.array([gg - 0.5 for gg in gs], float)
        pred = float(interp_extrap_vec(np.array([x]), tau[order], srv_scalar[order])[0])
        errs.append(pred - g)
        gdecl.append(g)
    return np.array(nlls), np.array(errs), np.array(gdecl)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--sqlite", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print("Loading surveys + ratings + attempts ...")
    surveys = load_rank_surveys(args.sql)
    surveys = surveys[surveys["rank_scalar"].notna()].copy()
    surveys["t"] = (pd.to_datetime(surveys["created_at"], utc=True, format="mixed")
                    .astype("int64") / 86400e9)
    surveys = surveys.sort_values("t")
    latest = surveys.groupby(["user_id", "server"], as_index=False).tail(1)

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    lapse = load_lapse(args.dir)
    fitter = RatingFitter(merged_ratings, merged_slopes, merged_floors, lapse)

    users_all = sorted(surveys["user_id"].unique())
    att = load_attempts_sqlite(args.sqlite, users_all)
    time_stats = load_user_time_stats(args.sqlite)

    career, blocks_by_user = {}, {}
    for uid in users_all:
        if uid not in att:
            continue
        fit = fitter.fit(att[uid][1])
        if fit is None:
            continue
        career[uid] = fit
        blocks_by_user[uid] = user_blocks(att, fitter, uid)
    users = sorted(career)

    feats, info = build_feature_defs(users, time_stats)
    n_both = sum(1 for u in users if len(time_stats.get(u[:8], {}).get("by_size", {})) == 2)
    n_913 = {sz: sum(1 for u in users if sz in time_stats.get(u[:8], {}).get("by_size", {}))
             for sz in SIZES}
    print(f"  {len(users)} usable users | timed {info['n_timed']} "
          f"(9x9 slot {n_913[9]}, 13x13 slot {n_913[13]}, both {n_both}) | centers: "
          f"pooled {info['c_pool']:+.3f}, 9x9 {info['c_size'][9]:+.3f}, "
          f"13x13 {info['c_size'][13]:+.3f} log10-min; frac13 {info['c_f13']:.3f}")

    latest = latest[latest["user_id"].isin(career)].reset_index(drop=True)
    surveys = surveys[surveys["user_id"].isin(career)]

    def block_decls(uids, T_of):
        """Production blocks-mass anchors with the variant's feature vector."""
        out = []
        for uid in uids:
            n_att = len(att[uid][1])
            blks = blocks_by_user[uid] or [(0.0, career[uid][0], career[uid][1], 0, n_att)]
            masses = block_masses(blks, n_att)
            udecl = surveys[surveys["user_id"] == uid]
            for server, srows in udecl.groupby("server"):
                srows = srows.sort_values("t")
                dt = srows["t"].to_numpy()
                dg = srows["rank_scalar"].to_numpy(float)
                assign = defaultdict(list)
                for (tc, R, se, _, _), m in zip(blks, masses):
                    j = int(np.argmin(np.abs(dt - tc)))
                    assign[j].append((R, m))
                for j, brs in assign.items():
                    mass_tot = sum(m for _, m in brs)
                    for R, m in brs:
                        w = m / mass_tot if mass_tot > 0 else 1.0 / len(brs)
                        out.append((server, R, float(dg[j]), T_of[uid], w))
        return out

    VARIANTS = list(feats.keys())

    # Full-data fit per variant: report the learned alphas.
    print("\nFull-data fitted time slopes (rating per de-meaned log10-min):")
    for vname in VARIANTS:
        T_of, F = feats[vname]
        fitres = fit_consensus_vec(block_decls(users, T_of), F)
        alpha = fitres[1]
        label = {"notime": "(unused feature)", "pooled": "alpha_pooled",
                 "size-adj": "alpha_sizeadj", "two-alpha": "alpha9, alpha13",
                 "pooled+frac13": "alpha_pooled, beta_frac13(rating/unit-frac)"}[vname]
        print(f"  {vname:<14} {label}: {np.round(alpha, 4)}")

    # --- CV -------------------------------------------------------------------
    fold_means = defaultdict(list)
    pooled_errs = defaultdict(lambda: ([], []))
    n_folds_total = 0
    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(users)
        for f, test_users in enumerate(np.array_split(perm, args.folds)):
            test_set = set(test_users.tolist())
            train_users = [u for u in users if u not in test_set]
            n_folds_total += 1
            test_latest = latest[latest["user_id"].isin(test_set)]
            for vname in VARIANTS:
                T_of, F = feats[vname]
                fitres = fit_consensus_vec(block_decls(train_users, T_of), F)
                if fitres is None:
                    continue
                eval_rows = [(row["server"], career[row["user_id"]][0],
                              float(row["rank_scalar"]), T_of[row["user_id"]])
                             for _, row in test_latest.iterrows()]
                nll, err, gdecl = evaluate(fitres, eval_rows)
                if len(nll) == 0:
                    continue
                fold_means[(vname, "nll")].append(float(np.mean(nll)))
                fold_means[(vname, "mae")].append(float(np.mean(np.abs(err))))
                fold_means[(vname, "rmse")].append(float(np.sqrt(np.mean(err ** 2))))
                pooled_errs[vname][0].extend(err.tolist())
                pooled_errs[vname][1].extend(gdecl.tolist())
        print(f"  rep {rep + 1}/{args.reps} done")

    def paired(vname, metric, base="pooled"):
        a = np.array(fold_means[(vname, metric)])
        b = np.array(fold_means[(base, metric)])
        if len(a) != len(b) or len(a) == 0:
            return None
        d = a - b
        se_d = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        return float(d.mean()), float(se_d)

    print(f"\n===== paired CV over {n_folds_total} foldings "
          f"({args.folds}-fold x {args.reps} reps), {len(latest)} eval declarations, "
          f"baseline = pooled (production) =====")
    print(f"{'variant':<14} {'NLL':>8} {'MAE':>7} {'RMSE':>7} | "
          f"{'ΔNLL vs pooled':>17} {'ΔMAE':>14}")
    for vname in VARIANTS:
        nll = np.mean(fold_means[(vname, "nll")])
        mae = np.mean(fold_means[(vname, "mae")])
        rmse = np.mean(fold_means[(vname, "rmse")])
        dn = paired(vname, "nll")
        dm = paired(vname, "mae")
        dn_s = f"{dn[0]:+.4f}±{dn[1]:.4f}" if dn else "-"
        dm_s = f"{dm[0]:+.4f}±{dm[1]:.4f}" if dm else "-"
        print(f"{vname:<14} {nll:8.4f} {mae:7.4f} {rmse:7.4f} | {dn_s:>17} {dm_s:>14}")

    print("\n--- band breakdown (pooled held-out errors) ---")
    print(f"{'variant':<14} {'dan MAE':>8} {'dan bias':>9} {'kyu MAE':>8} {'kyu bias':>9} "
          f"{'<=10k MAE':>9}")
    for vname in VARIANTS:
        errs, gdecl = pooled_errs[vname]
        errs, gdecl = np.array(errs), np.array(gdecl)
        dan, kyu, deep = gdecl >= 1, gdecl < 1, gdecl <= -9
        def mae_bias(m):
            if m.sum() == 0:
                return "     -", "      -"
            return f"{np.mean(np.abs(errs[m])):6.3f}", f"{np.mean(errs[m]):+7.3f}"
        dm_, db = mae_bias(dan)
        km, kb = mae_bias(kyu)
        pm, _ = mae_bias(deep)
        print(f"{vname:<14} {dm_:>8} {db:>9} {km:>8} {kb:>9} {pm:>9}")

    # dual-size-user slice: the only users the size split can actually move
    both_users = {u for u in users
                  if len(time_stats.get(u[:8], {}).get("by_size", {})) == 2}
    print(f"\n--- dual-size users only ({len(both_users)} users; pooled held-out) ---")
    both_ids = latest["user_id"].isin(both_users)
    print(f"{'variant':<14} {'MAE':>7} {'bias':>8}  (single held-out fit: train = "
          f"non-dual users, score = dual users' declarations)")
    for vname in VARIANTS:
        T_of, F = feats[vname]
        train_users = [u for u in users if u not in both_users]
        fitres = fit_consensus_vec(block_decls(train_users, T_of), F)
        if fitres is None:
            continue
        eval_rows = [(row["server"], career[row["user_id"]][0],
                      float(row["rank_scalar"]), T_of[row["user_id"]])
                     for _, row in latest[both_ids].iterrows()]
        nll, err, gdecl = evaluate(fitres, eval_rows)
        print(f"{vname:<14} {np.mean(np.abs(err)):7.4f} {np.mean(err):+8.4f}")


if __name__ == "__main__":
    main()
