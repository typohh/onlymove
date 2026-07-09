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
  norm-fit       production aggregation unchanged (ONE pooled median, one alpha),
                 but each 13x13 attempt's time is divided by a DATA-DERIVED size
                 multiplier before the median (the within-user paired median gap,
                 fit over ALL DB users with both sizes timed, not just survey users).
  norm-2x        same, with a fixed 2x multiplier for 13x13 (kept for the record; a
                 2x correction overshoots the real ~1.28x gap and halves the learned
                 slope — flawed by construction, excluded from default runs).
  norm-add       same aggregation, ADDITIVE correction: subtract the data-derived
                 constant (median within-user med13-med9 gap, ~3.6s — the "scan the
                 whole board once" story; the free affine fit med13 ≈ 3-5s +
                 (1.10-1.20)*med9 sits between pure-additive and pure-multiplicative)
                 from each 13x13 attempt time, clamped at 1s so the median/log stay
                 positive.

Metrics: held-out ordinal NLL, MAE/RMSE of predicted vs declared server rank, and
SPEARMAN of predicted vs declared (per folding, paired diffs vs pooled). The band
breakdown slices pooled errors by declared band AND by the user's timed-size
profile (9-only / 13-only / both / untimed) — the size corrections mostly move
single-size users (a dual user's pooled median barely shifts), so 13-only vs
9-only relative bias is the slice where a size effect must show up.

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
from scipy.stats import spearmanr

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
def load_user_timed_lists(sqlite_path):
    """{user_prefix8: [(time_ms, size), ...]} — the timed non-repeat attempts inside
    each user's most-recent WORKING_SET window (ALL DB users; production filters)."""
    con = sqlite3.connect(sqlite_path)
    try:
        cur = con.execute("SELECT user_id, is_repeat, time_used, size FROM tsumego_reports "
                          "ORDER BY created_at, id")
        windows = defaultdict(lambda: deque(maxlen=WORKING_SET))
        for uid, rep, t, sz in cur:
            windows[uid[:8]].append((rep, t, sz))
    finally:
        con.close()
    return {pref: [(float(t), sz) for (rep, t, sz) in rows
                   if not rep and t is not None and 0 < float(t) <= AFK_CAP_MS]
            for pref, rows in windows.items()}


def fit_size_multiplier(timed_lists):
    """The data-derived 13x13→9x9 time factor: mean within-user paired
    log10(median13/median9) over every DB user with MIN_TIMED_SIZE timed attempts
    on both sizes. Returns (multiplier, n_users)."""
    gaps = []
    for ts in timed_lists.values():
        by = {sz: [t for t, s in ts if s == sz] for sz in SIZES}
        if all(len(by[sz]) >= MIN_TIMED_SIZE for sz in SIZES):
            gaps.append(np.log10(np.median(by[13])) - np.log10(np.median(by[9])))
    if not gaps:
        return 1.0, 0
    return float(10 ** np.mean(gaps)), len(gaps)


def user_time_stats(timed_lists):
    """{user_prefix8: {"pooled": log10 med min, "by_size": {sz: (log10 med min, n)},
    "frac13": ...}} — pooled needs MIN_TIMED, a size slot needs MIN_TIMED_SIZE."""
    out = {}
    for pref, ts in timed_lists.items():
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


def fit_size_offset_ms(timed_lists):
    """The data-derived ADDITIVE 13x13→9x9 correction: median within-user
    (median13 - median9) gap in ms over the dual-size users. Returns (ms, n)."""
    gaps = []
    for ts in timed_lists.values():
        by = {sz: [t for t, s in ts if s == sz] for sz in SIZES}
        if all(len(by[sz]) >= MIN_TIMED_SIZE for sz in SIZES):
            gaps.append(np.median(by[13]) - np.median(by[9]))
    if not gaps:
        return 0.0, 0
    return float(np.median(gaps)), len(gaps)


def normalized_pooled_medians(timed_lists, mult13=1.0, sub13_ms=0.0):
    """{user_prefix8: log10 med min} with each 13x13 time divided by mult13 and/or
    shifted down by sub13_ms (clamped at 1s) before the ONE pooled median —
    production aggregation, size-normalized times."""
    out = {}
    for pref, ts in timed_lists.items():
        if len(ts) < MIN_TIMED:
            continue
        # clamp only on the additive path (subtraction can cross 0; pure division
        # cannot, and norm-fit stays bit-identical to its earlier definition)
        norm = [(max(t - sub13_ms, 1000.0) if sub13_ms > 0 else t) / mult13
                if sz == 13 else t for t, sz in ts]
        out[pref] = float(np.log10(np.median(norm) / 60000.0))
    return out


def build_feature_defs(users, time_stats, norm_medians=None):
    """Per-variant per-user feature vectors. Returns {variant: ({uid: np.array(F)}, F)}.
    Centers are population means over the timed users (full-data, like production).
    norm_medians: {name: {pref8: log10 med}} adds one single-feature variant each
    (the size-normalized pooled medians; same user availability as pooled)."""
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
    variants = {
        "notime":        (zero1, 1),
        "pooled":        (T_pool, 1),
        "size-adj":      (T_adj, 1),
        "two-alpha":     (T_two, 2),
        "pooled+frac13": (T_pf, 2),
    }
    for name, med in (norm_medians or {}).items():
        c = float(np.mean([med[u[:8]] for u in timed if u[:8] in med])) if timed else 0.0
        variants[name] = ({u: (np.array([med[u[:8]] - c]) if u[:8] in med
                               else np.zeros(1)) for u in users}, 1)
    return variants, dict(c_pool=c_pool, c_size=c_size, c_f13=c_f13, n_timed=len(timed))


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
    """eval_rows: [(server, R, g, Tvec, tag)]. Returns (nll[], err[], g[], tags[])
    over the rows whose server has a fitted coupling; tag is opaque caller context
    (the user's timed-size profile)."""
    a, alpha, ys, server_fits = fitres
    nlls, errs, gdecl, tags = [], [], [], []
    for server, R, g, T, tag in eval_rows:
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
        tags.append(tag)
    return np.array(nlls), np.array(errs), np.array(gdecl), tags


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--sqlite", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--variants", default=None,
                    help="comma-list of variant names to run (default: all). Foldings "
                         "are seed-deterministic, so runs with the same --reps/--folds "
                         "are paired-comparable across invocations via the pooled "
                         "baseline.")
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
    timed_lists = load_user_timed_lists(args.sqlite)
    time_stats = user_time_stats(timed_lists)
    mult13, n_mult = fit_size_multiplier(timed_lists)
    off13, n_off = fit_size_offset_ms(timed_lists)
    print(f"  size corrections (within-user paired medians, ALL DB users with both "
          f"sizes): multiplicative {mult13:.3f}x, additive {off13 / 1000:.1f}s "
          f"({n_mult} users)")
    norm_medians = {"norm-fit": normalized_pooled_medians(timed_lists, mult13=mult13),
                    "norm-2x": normalized_pooled_medians(timed_lists, mult13=2.0),
                    "norm-add": normalized_pooled_medians(timed_lists, sub13_ms=off13)}

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

    feats, info = build_feature_defs(users, time_stats, norm_medians)
    if args.variants:
        keep = set(args.variants.split(","))
    else:
        keep = set(feats) - {"norm-2x"}   # flawed by construction; opt-in only
    keep.add("pooled")   # the paired-diff baseline is always needed
    feats = {k: v for k, v in feats.items() if k in keep}

    def size_profile(uid):
        by = time_stats.get(uid[:8], {}).get("by_size", {})
        if len(by) == 2:
            return "both"
        if 9 in by:
            return "9-only"
        if 13 in by:
            return "13-only"
        return "untimed"
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
                 "pooled+frac13": "alpha_pooled, beta_frac13(rating/unit-frac)",
                 }.get(vname, "alpha")
        print(f"  {vname:<14} {label}: {np.round(alpha, 4)}")

    # --- CV -------------------------------------------------------------------
    fold_means = defaultdict(list)
    pooled_errs = defaultdict(lambda: ([], [], []))
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
                              float(row["rank_scalar"]), T_of[row["user_id"]],
                              size_profile(row["user_id"]))
                             for _, row in test_latest.iterrows()]
                nll, err, gdecl, tags = evaluate(fitres, eval_rows)
                if len(nll) == 0:
                    continue
                fold_means[(vname, "nll")].append(float(np.mean(nll)))
                fold_means[(vname, "mae")].append(float(np.mean(np.abs(err))))
                fold_means[(vname, "rmse")].append(float(np.sqrt(np.mean(err ** 2))))
                if len(gdecl) >= 3 and np.std(gdecl) > 0:
                    pred = err + gdecl
                    fold_means[(vname, "spearman")].append(
                        float(spearmanr(pred, gdecl).statistic))
                pooled_errs[vname][0].extend(err.tolist())
                pooled_errs[vname][1].extend(gdecl.tolist())
                pooled_errs[vname][2].extend(tags)
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
    print(f"{'variant':<14} {'NLL':>8} {'MAE':>7} {'RMSE':>7} {'SPEAR':>7} | "
          f"{'ΔNLL vs pooled':>17} {'ΔMAE':>14} {'ΔSPEAR':>16}")
    for vname in VARIANTS:
        nll = np.mean(fold_means[(vname, "nll")])
        mae = np.mean(fold_means[(vname, "mae")])
        rmse = np.mean(fold_means[(vname, "rmse")])
        sp = np.mean(fold_means[(vname, "spearman")])
        dn = paired(vname, "nll")
        dm = paired(vname, "mae")
        ds = paired(vname, "spearman")
        dn_s = f"{dn[0]:+.4f}±{dn[1]:.4f}" if dn else "-"
        dm_s = f"{dm[0]:+.4f}±{dm[1]:.4f}" if dm else "-"
        ds_s = f"{ds[0]:+.4f}±{ds[1]:.4f}" if ds else "-"
        print(f"{vname:<14} {nll:8.4f} {mae:7.4f} {rmse:7.4f} {sp:7.4f} | "
              f"{dn_s:>17} {dm_s:>14} {ds_s:>16}")

    print("\n--- band breakdown (pooled held-out errors) ---")
    print(f"{'variant':<14} {'dan MAE':>8} {'dan bias':>9} {'kyu MAE':>8} {'kyu bias':>9} "
          f"{'<=10k MAE':>9}")
    for vname in VARIANTS:
        errs, gdecl, _ = pooled_errs[vname]
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

    # The slice that matters for a SIZE effect: single-size users. A size correction
    # barely moves a dual user's pooled median; it shifts 13-only users relative to
    # 9-only users. Look at the RELATIVE bias (13-only minus 9-only).
    profiles = ["9-only", "13-only", "both", "untimed"]
    _, _, tags0 = pooled_errs[VARIANTS[0]]
    cnt = {p: tags0.count(p) for p in profiles}
    print(f"\n--- timed-size-profile breakdown (pooled held-out errors; "
          f"declaration rows: {cnt}) ---")
    print(f"{'variant':<14}" + "".join(f" {p + ' MAE':>12} {p + ' bias':>13}"
                                       for p in profiles[:3])
          + f" {'rel bias 13-9':>14}")
    for vname in VARIANTS:
        errs, _, tags = pooled_errs[vname]
        errs = np.array(errs)
        tags = np.array(tags)
        cells, bias = [], {}
        for p in profiles[:3]:
            m = tags == p
            if m.sum() == 0:
                cells.append(f"{'-':>12} {'-':>13}")
                continue
            bias[p] = float(np.mean(errs[m]))
            cells.append(f"{np.mean(np.abs(errs[m])):12.3f} {bias[p]:+13.3f}")
        rel = (f"{bias['13-only'] - bias['9-only']:+14.3f}"
               if "13-only" in bias and "9-only" in bias else f"{'-':>14}")
        print(f"{vname:<14}" + "".join(cells) + rel)

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
                      float(row["rank_scalar"]), T_of[row["user_id"]], None)
                     for _, row in latest[both_ids].iterrows()]
        nll, err, gdecl, _ = evaluate(fitres, eval_rows)
        print(f"{vname:<14} {np.mean(np.abs(err)):7.4f} {np.mean(err):+8.4f}")


if __name__ == "__main__":
    main()
