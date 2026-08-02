"""Paired held-out CV for the consensus rating→rank fit (pipeline.map_rating_to_rank):
does the mapping gain from (a) errors-in-variables over each user's rating CI, or
(b) time-segmented user ratings?

Variants (all share the exact production objective/penalties; they differ only in the
user-rating covariate and in whether the fit integrates over its standard error):

  career        production baseline — one pooled all-career 80%-clear rating per user,
                treated as exact (se = 0).
  career+eiv    same rating, but each declaration's ordinal likelihood is averaged over
                Normal(rating, se^2) via Gauss-Hermite quadrature; se = the Fisher
                standard error of fit_user_rating's MLE.
  win120[+eiv]  trailing-120-day rating at declaration time (fallback to career when the
                window has < MIN_ATTEMPTS rated attempts). This is the DRIFT PROBE: per
                user guidance we assume rank does NOT drift (surveys only began
                2026-05-18, so declarations carry no early history) — if this variant
                beats career, that assumption is wrong.
  blocks[+eiv]  overlapping 2-month blocks (60d window, 30d step) over the user's whole
                career; every block with >= MIN_ATTEMPTS rated attempts becomes its own
                training row anchored to the user's declared rank (nearest-in-time
                declaration per server when they filed several — "no drift unless a
                survey says otherwise"). Rows are weighted EQUALLY per block, normalized
                so each declaration contributes total weight 1 regardless of career
                length. NOTE: equal-per-block underweights the career's edge eras — an
                interior attempt sits in 2 overlapping blocks, the first/last ~30 days
                in only 1 — see blocks-mass.
  blocks-mass   blocks, but each block is weighted by the ATTEMPT MASS it holds: every
                attempt's unit mass is split equally over the blocks containing it, a
                block's weight is the mass sum, and the declaration is normalized to
                total weight 1. Every attempt then counts exactly once regardless of
                the overlap coverage (fixes the edge-era underweighting above).
  blocks-raw    blocks with NO per-declaration weight normalization (each block weight 1)
                — probes whether heavy users' extra pseudo-rows help or bias.
  *+rob         same declarations, but FIT with the production contamination mixture
                (ordinal_nll_vec eps=EPS_CONTAMINATION: with probability eps a
                declaration is junk, uniform over categories). Scoring stays the plain
                ordinal NLL for every variant, so the paired diffs isolate the fit
                change.

Protocol: 5-fold CV by USER, repeated with several seeds; every variant is fit and
scored on the SAME foldings (paired). Held-out declarations (latest per user-server,
same as production) are scored point-wise — the client maps a point rating through M —
under two covariate lenses: the career rating and the win120 rating. Metrics: mean
ordinal NLL (via the fitted server coupling's thresholds) and MAE/RMSE of the
continuous predicted server rank vs declared. The time feature T and its center are
computed once from the full data (production behavior; identical across variants, so
the small leak cancels in paired diffs).

Sanity: with se=0 and unit weights this module's fit reproduces pipeline
fit_consensus exactly (checked at startup on the full data).

Usage: ./.venv/bin/python3 -m auxiliary.eval_rank_map_cv [--reps 6] [--folds 5]
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline.map_rating_to_rank import (  # noqa: E402
    EPS_CONTAMINATION, GAUGE_WEIGHT, LAMBDA_SHAPE, LAMBDA_TIME, MIN_ATTEMPTS,
    MIN_SERVER_SAMPLES, RANK_PADDING, EXTEND_TO, SERVER_LAMBDA, SIZES,
    _curve_arrays, fit_consensus,
    load_floors, load_ratings, load_slopes, load_user_median_solve_minutes,
)
from utils.rating.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, FLOOR_MAX, LOGIT_PER_RATING, fit_user_rating,
    interp_extrap_vec, _unpack_thresholds,
)
from utils.rating.rank_surveys import SQL_PATH_DEFAULT, load_rank_surveys  # noqa: E402
from utils.rating.ratings_io import load_lapse  # noqa: E402

BLOCK_DAYS = 60
BLOCK_STEP = 30
WIN_DAYS = 120
GH_NODES = 7

_ghz, _ghw = np.polynomial.hermite_e.hermegauss(GH_NODES)
_ghw = _ghw / _ghw.sum()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_attempts_sqlite(sqlite_path, user_ids):
    """{user_id: (times_days ascending np.array, [(tid, success), ...] aligned)} —
    first-time attempts only, mirroring load_attempts_for_users(first_time_only=True)."""
    ids = sorted(user_ids)
    con = sqlite3.connect(sqlite_path)
    try:
        q = ("SELECT user_id, problem, success, created_at FROM tsumego_reports "
             "WHERE (is_repeat IS NULL OR is_repeat != 1) AND user_id IN ({})"
             .format(",".join("?" * len(ids))))
        df = pd.read_sql_query(q, con, params=ids)
    finally:
        con.close()
    df["t"] = pd.to_datetime(df["created_at"], utc=True, format="mixed").astype("int64") / 86400e9
    df = df.sort_values("t")
    out = {}
    for uid, grp in df.groupby("user_id"):
        out[uid] = (grp["t"].to_numpy(),
                    list(zip(grp["problem"].tolist(), grp["success"].astype(int).tolist())))
    return out


# ---------------------------------------------------------------------------
# Rating + Fisher SE (mirrors fit_user_rating's internal 4PL exactly)
# ---------------------------------------------------------------------------
class RatingFitter:
    def __init__(self, ratings, slopes, floors, lapse):
        self.ratings, self.slopes, self.floors, self.lapse = ratings, slopes, floors, lapse

    def fit(self, rows):
        """(R, se, n_joined) or None. se = 1/sqrt(Fisher info) at the MLE."""
        R, n = fit_user_rating(rows, self.ratings, min_n=MIN_ATTEMPTS,
                               slopes=self.slopes, floors=self.floors, lapse=self.lapse)
        if R is None or n < MIN_ATTEMPTS:
            return None
        joined = [(self.ratings[t], (self.slopes.get(t) if self.slopes else None),
                   (self.floors.get(t) if self.floors else None)) for t, _ in rows
                  if t in self.ratings]
        rr = np.array([r for r, _, _ in joined], float)
        sl = np.array([(a if (a is not None and np.isfinite(a) and a > 0)
                        else LOGIT_PER_RATING) for _, a, _ in joined], float)
        c = np.array([min(max(f if (f is not None and np.isfinite(f)) else 0.0, 0.0),
                          FLOOR_MAX) for _, _, f in joined], float)
        lam = max(0.0, float(self.lapse))
        span = np.clip(1.0 - c - lam, 1e-6, 1.0)
        q = np.clip((CALIBRATION_P - c) / span, 1e-6, 1.0 - 1e-6)
        kappa = np.log(q / (1.0 - q))
        praw = expit(np.clip(sl * (R - rr) + kappa, -40.0, 40.0))
        p = np.clip(c + span * praw, 1e-9, 1.0 - 1e-9)
        g = span * sl * praw * (1.0 - praw)
        info = float(np.sum(g * g / (p * (1.0 - p))))
        se = 1.0 / np.sqrt(info) if info > 1e-9 else None
        if se is None:
            return None
        return float(R), float(se), n


def window_rows(att, uid, t_end, days):
    times, rows = att[uid]
    hi = int(np.searchsorted(times, t_end, side="right"))
    lo = int(np.searchsorted(times, t_end - days, side="left"))
    return rows[lo:hi]


def user_blocks(att, fitter, uid):
    """[(t_center, R, se, lo, hi)] for every overlapping BLOCK_DAYS block with a
    fittable rating ([lo, hi) = the attempt-index range, for coverage-mass weights);
    empty careers fall back to nothing (caller falls back to career)."""
    times, rows = att[uid]
    if len(times) == 0:
        return []
    out, seen = [], set()
    starts = np.arange(times[0], times[-1] + 1e-9, BLOCK_STEP)
    for s in starts:
        lo = int(np.searchsorted(times, s, side="left"))
        hi = int(np.searchsorted(times, s + BLOCK_DAYS, side="right"))
        if hi - lo < MIN_ATTEMPTS or (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        fit = fitter.fit(rows[lo:hi])
        if fit is not None:
            out.append((float((times[lo] + times[hi - 1]) / 2), fit[0], fit[1], lo, hi))
    return out


def block_masses(blks, n_attempts):
    """Attempt-mass per block: each attempt's unit mass is split equally over the
    blocks covering it; a block's mass is the sum over its attempts. Attempts no
    kept block covers (e.g. sparse gaps below MIN_ATTEMPTS) simply carry no mass."""
    cov = np.zeros(n_attempts)
    for _, _, _, lo, hi in blks:
        cov[lo:hi] += 1
    inv = np.where(cov > 0, 1.0 / np.maximum(cov, 1), 0.0)
    return [float(np.sum(inv[lo:hi])) for _, _, _, lo, hi in blks]


# ---------------------------------------------------------------------------
# Consensus fit with optional per-row se (EIV via Gauss-Hermite) and weights.
# Copy of pipeline fit_consensus extended with `se`/`w` arrays; se=0, w=1
# reproduces production exactly (verified at startup).
# ---------------------------------------------------------------------------
def _bucket_w(decls):
    rows_by = defaultdict(list)
    for server, R, g, T, se, w in decls:
        rows_by[server].append((R, int(round(g)), T, se, w))
    blocks = {}
    for server, rows in rows_by.items():
        # production gate counts declarations; blocks variants multiply rows, so gate
        # on total weight (== declaration count when w=1)
        if sum(w for *_, w in rows) < MIN_SERVER_SAMPLES:
            continue
        declared = [g for _, g, _, _, _ in rows]
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
            "rating": np.array([r for r, _, _, _, _ in rows], float),
            "T": np.array([t for _, _, t, _, _ in rows], float),
            "se": np.array([s for _, _, _, s, _ in rows], float),
            "w": np.array([w for _, _, _, _, w in rows], float),
            "k": np.array([max(0, min(N - 1, g - g_min)) for _, g, _, _, _ in rows], int)}
    return blocks


def fit_consensus_cv(decls, eps=0.0):
    """decls: [(server, R, g, T, se, w)]. Returns (a, alpha, ys, server_fits).
    eps: contamination-mixture weight on a uniform-over-categories junk component
    (matches ordinal_nll_vec's eps; production fits with EPS_CONTAMINATION)."""
    egf_thr_ranks, ys, S, dS = _curve_arrays()
    nseg = len(dS)
    blocks = _bucket_w(decls)
    if not blocks:
        return None

    all_R = np.array([R for _, R, _, _, _, _ in decls], float)
    all_g = np.array([g for _, _, g, _, _, _ in decls], float)
    beta, alpha_c = np.polyfit(all_R, all_g, 1)
    if beta <= 0:
        beta = 1e-3
    prior_rating = (ys - alpha_c) / beta
    b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
    a0 = float(prior_rating[0] - b0 * S[0])

    srv_layout, x0, idx = [], [a0, float(np.log(max(b0, 1e-3)))] + [0.0] * nseg + [0.0], 3 + nseg
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
        return a, params[2 + nseg]

    def objective(params):
        a, alpha = unpack_curve(params)
        nll, pen, w_off, w_tot = 0.0, 0.0, 0.0, 0.0
        for server, st, n in srv_layout:
            blk = blocks[server]
            thr = _unpack_thresholds(params[st:st + n], blk["baseline"])
            x_mid = blk["rating"] + alpha * blk["T"]
            if np.any(blk["se"] > 0):
                x_eff = x_mid[:, None] + blk["se"][:, None] * _ghz[None, :]
                x_int = interp_extrap_vec(x_eff.ravel(), a, ys).reshape(len(x_mid), -1)
                gh_w = _ghw
            else:
                x_int = interp_extrap_vec(x_mid, a, ys)[:, None]
                gh_w = np.array([1.0])
            K = thr.shape[0]
            kk = blk["k"]
            upper = np.where((kk < K)[:, None],
                             expit(thr[np.minimum(kk, K - 1)][:, None] - x_int), 1.0)
            lower = np.where((kk > 0)[:, None],
                             expit(thr[np.maximum(kk - 1, 0)][:, None] - x_int), 0.0)
            p = np.clip(upper - lower, 1e-12, 1.0) @ gh_w
            if eps > 0.0:
                p = (1.0 - eps) * p + eps / (K + 1)
            nll += -float(np.sum(blk["w"] * np.log(np.clip(p, 1e-12, 1.0))))
            pen += SERVER_LAMBDA * float(np.sum(params[st + 1:st + n] ** 2))
            wsrv = float(np.sum(blk["w"]))
            w_off += wsrv * float(np.mean(thr - blk["identity"]))
            w_tot += wsrv
        if nseg:
            pen += LAMBDA_SHAPE * float(np.sum(params[2:2 + nseg] ** 2))
        pen += LAMBDA_TIME * float(alpha ** 2)
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
    return a, float(alpha), ys, server_fits


# ---------------------------------------------------------------------------
# Held-out scoring
# ---------------------------------------------------------------------------
def evaluate(fitres, eval_rows):
    """eval_rows: [(server, R, g, T)] point covariates. Returns (nll[], err[], g[])
    over the rows whose server has a fitted coupling (>= 2 thresholds)."""
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
        x = float(interp_extrap_vec(np.array([R + alpha * T]), a, ys)[0])
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
    ap.add_argument("--variants", default=None,
                    help="comma-list of variant names to run (default: all). Foldings "
                         "are seed-deterministic, so runs with the same --reps/--folds "
                         "are paired-comparable across invocations via the career "
                         "baseline.")
    ap.add_argument("--rob-eps", default=None,
                    help="comma-list of extra contamination eps values to sweep; each "
                         "adds a blocks-mass+rob@<eps> variant (fit with that eps, "
                         "scored plain like everything else).")
    ap.add_argument("--drop-eval", default=None,
                    help="comma-list of userprefix8:server declarations to drop from "
                         "the EVAL set only (training keeps them). For scoring on "
                         "honest declarations when the eval set is known to contain "
                         "junk — e.g. the contamination-posterior>0.5 rows from "
                         "auxiliary.check_mapped_vs_declared_rank. Applied identically "
                         "to every variant, so paired diffs stay fair.")
    args = ap.parse_args()

    print("Loading surveys + ratings + attempts ...")
    surveys = load_rank_surveys(args.sql)
    surveys = surveys[surveys["rank_value"].notna()].copy()
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
    med_min = load_user_median_solve_minutes(args.sqlite)

    # --- per-user covariates ---------------------------------------------------
    career, win, blocks_by_user = {}, {}, {}
    win_fallbacks = 0
    for uid in users_all:
        if uid not in att:
            continue
        rows = att[uid][1]
        fit = fitter.fit(rows)
        if fit is None:
            continue
        career[uid] = fit  # (R, se, n)
        blocks_by_user[uid] = user_blocks(att, fitter, uid)
        t_latest = float(surveys[surveys["user_id"] == uid]["t"].max())
        wfit = fitter.fit(window_rows(att, uid, t_latest, WIN_DAYS))
        if wfit is None:
            wfit = fit
            win_fallbacks += 1
        win[uid] = wfit

    timed = {u: np.log10(med_min[u[:8]]) for u in career
             if u[:8] in med_min and med_min[u[:8]] > 0}
    center = float(np.mean(list(timed.values()))) if timed else 0.0
    T_of = {u: (timed[u] - center if u in timed else 0.0) for u in career}

    users = sorted(career)
    ses = np.array([career[u][1] for u in users])
    ns = np.array([career[u][2] for u in users])
    nblocks = np.array([max(1, len(blocks_by_user[u])) for u in users])
    print(f"  {len(users)} usable users | career se p10/50/90 = "
          f"{np.percentile(ses, 10):.3f}/{np.percentile(ses, 50):.3f}/{np.percentile(ses, 90):.3f} "
          f"rating | attempts p10/50/90 = {np.percentile(ns, 10):.0f}/{np.percentile(ns, 50):.0f}/"
          f"{np.percentile(ns, 90):.0f} | blocks/user p50/90/max = "
          f"{np.percentile(nblocks, 50):.0f}/{np.percentile(nblocks, 90):.0f}/{nblocks.max()} | "
          f"win120 fallbacks {win_fallbacks}")

    latest = latest[latest["user_id"].isin(career)].reset_index(drop=True)
    surveys = surveys[surveys["user_id"].isin(career)]

    # --- declaration builders ---------------------------------------------------
    def point_decls(uids, cov, eiv):
        out = []
        for _, row in latest[latest["user_id"].isin(uids)].iterrows():
            uid = row["user_id"]
            R, se, _ = cov[uid]
            out.append((row["server"], R, float(row["rank_value"]), T_of[uid],
                        se if eiv else 0.0, 1.0))
        return out

    def block_decls(uids, eiv, mode):
        """mode: 'equal' (1/n per block), 'mass' (attempt-mass, normalized to 1 per
        declaration), 'raw' (weight 1 per block, no normalization)."""
        out = []
        for uid in uids:
            n_att = len(att[uid][1])
            blks = blocks_by_user[uid] or [(0.0, career[uid][0], career[uid][1], 0, n_att)]
            masses = block_masses(blks, n_att)
            udecl = surveys[surveys["user_id"] == uid]
            for server, srows in udecl.groupby("server"):
                srows = srows.sort_values("t")
                dt = srows["t"].to_numpy()
                dg = srows["rank_value"].to_numpy(float)
                # nearest-in-time declaration anchors each block (no drift otherwise)
                assign = defaultdict(list)
                for (tc, R, se, _, _), m in zip(blks, masses):
                    j = int(np.argmin(np.abs(dt - tc)))
                    assign[j].append((R, se, m))
                for j, brs in assign.items():
                    mass_tot = sum(m for _, _, m in brs)
                    for R, se, m in brs:
                        w = (1.0 if mode == "raw" else
                             1.0 / len(brs) if mode == "equal" else
                             (m / mass_tot if mass_tot > 0 else 1.0 / len(brs)))
                        out.append((server, R, float(dg[j]), T_of[uid],
                                    se if eiv else 0.0, w))
        return out

    # eiv was measurably worse across every covariate in the first full run
    # (see the memory/report); keep only career+eiv as the recorded control.
    # +rob = fit with the production contamination mixture (EPS_CONTAMINATION);
    # scoring stays the plain ordinal NLL for every variant so diffs are paired.
    VARIANTS = [
        ("career",          lambda u: point_decls(u, career, False), 0.0),
        ("career+eiv",      lambda u: point_decls(u, career, True), 0.0),
        ("career+rob",      lambda u: point_decls(u, career, False), EPS_CONTAMINATION),
        ("win120",          lambda u: point_decls(u, win, False), 0.0),
        ("blocks",          lambda u: block_decls(u, False, "equal"), 0.0),
        ("blocks-mass",     lambda u: block_decls(u, False, "mass"), 0.0),
        ("blocks-mass+rob", lambda u: block_decls(u, False, "mass"), EPS_CONTAMINATION),
        ("blocks-raw",      lambda u: block_decls(u, False, "raw"), 0.0),
    ]
    if args.variants:
        keep = set(args.variants.split(","))
        VARIANTS = [v for v in VARIANTS if v[0] in keep]
    if args.rob_eps:
        for tok in args.rob_eps.split(","):
            VARIANTS.append((f"blocks-mass+rob@{tok}",
                             lambda u: block_decls(u, False, "mass"), float(tok)))
    LENSES = [("career-lens", career), ("win120-lens", win)]

    # --- sanity: se=0/w=1/eps=EPS_CONTAMINATION reproduces production fit_consensus
    prod_decls = [(s, R, g, T) for (s, R, g, T, _, _) in point_decls(users, career, False)]
    a_p, al_p, ys_p, _, _ = fit_consensus(prod_decls)
    a_c, al_c, _, _ = fit_consensus_cv(point_decls(users, career, False),
                                       eps=EPS_CONTAMINATION)
    print(f"  sanity vs production fit: max|Δcurve| = {np.max(np.abs(a_p - a_c)):.2e}, "
          f"|Δalpha| = {abs(al_p - al_c):.2e}")

    # --- CV ---------------------------------------------------------------------
    drop_eval = set()
    if args.drop_eval:
        drop_eval = {tuple(tok.split(":", 1)) for tok in args.drop_eval.split(",")}
        n_dropped = sum((row["user_id"][:8], row["server"]) in drop_eval
                        for _, row in latest.iterrows())
        print(f"  eval-only drop: {n_dropped} declarations matched of "
              f"{len(drop_eval)} requested (training keeps them)")
    fold_means = defaultdict(list)   # (variant, lens, metric) -> per-folding means
    pooled = defaultdict(lambda: ([], []))  # (variant, lens) -> (errs, gdecl) pooled
    n_folds_total = 0
    for rep in range(args.reps):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(users)
        for f, test_users in enumerate(np.array_split(perm, args.folds)):
            test_set = set(test_users.tolist())
            train_users = [u for u in users if u not in test_set]
            n_folds_total += 1
            eval_rows = {}
            for lens_name, cov in LENSES:
                eval_rows[lens_name] = [
                    (row["server"], cov[row["user_id"]][0], float(row["rank_value"]),
                     T_of[row["user_id"]])
                    for _, row in latest[latest["user_id"].isin(test_set)].iterrows()
                    if (row["user_id"][:8], row["server"]) not in drop_eval]
            for vname, builder, veps in VARIANTS:
                fitres = fit_consensus_cv(builder(train_users), eps=veps)
                if fitres is None:
                    continue
                for lens_name, _ in LENSES:
                    nll, err, gdecl = evaluate(fitres, eval_rows[lens_name])
                    if len(nll) == 0:
                        continue
                    fold_means[(vname, lens_name, "nll")].append(float(np.mean(nll)))
                    fold_means[(vname, lens_name, "mae")].append(float(np.mean(np.abs(err))))
                    fold_means[(vname, lens_name, "rmse")].append(
                        float(np.sqrt(np.mean(err ** 2))))
                    pooled[(vname, lens_name)][0].extend(err.tolist())
                    pooled[(vname, lens_name)][1].extend(gdecl.tolist())
        print(f"  rep {rep + 1}/{args.reps} done")

    # --- report -------------------------------------------------------------
    def paired(vname, lens, metric, base="career"):
        a = np.array(fold_means[(vname, lens, metric)])
        b = np.array(fold_means[(base, lens, metric)])
        if len(a) != len(b) or len(a) == 0:
            return None
        d = a - b
        se_d = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        return float(d.mean()), float(se_d)

    print(f"\n===== paired CV over {n_folds_total} foldings "
          f"({args.folds}-fold x {args.reps} reps), {len(latest)} eval declarations =====")
    for lens_name, _ in LENSES:
        print(f"\n--- lens: {lens_name} (held-out point covariate) ---")
        print(f"{'variant':<16} {'NLL':>8} {'MAE':>7} {'RMSE':>7} | "
              f"{'ΔNLL vs career':>17} {'ΔMAE':>14}")
        for vname, _, _ in VARIANTS:
            key = (vname, lens_name)
            nll = np.mean(fold_means[(vname, lens_name, 'nll')])
            mae = np.mean(fold_means[(vname, lens_name, 'mae')])
            rmse = np.mean(fold_means[(vname, lens_name, 'rmse')])
            dn = paired(vname, lens_name, "nll")
            dm = paired(vname, lens_name, "mae")
            dn_s = f"{dn[0]:+.4f}±{dn[1]:.4f}" if dn else "-"
            dm_s = f"{dm[0]:+.4f}±{dm[1]:.4f}" if dm else "-"
            print(f"{vname:<16} {nll:8.4f} {mae:7.4f} {rmse:7.4f} | {dn_s:>17} {dm_s:>14}")

    # direct paired diff of each +rob variant against its eps=0 twin
    names = [v[0] for v in VARIANTS]
    rob_pairs = [(n, n.split("+rob")[0]) for n in names
                 if "+rob" in n and n.split("+rob")[0] in names]
    if rob_pairs:
        print("\n--- robust (contamination eps) vs plain, paired ---")
        for rob, base in rob_pairs:
            for lens_name, _ in LENSES:
                dn = paired(rob, lens_name, "nll", base=base)
                dm = paired(rob, lens_name, "mae", base=base)
                if dn is None:
                    continue
                print(f"{rob:<16} vs {base:<12} [{lens_name}]  "
                      f"ΔNLL {dn[0]:+.4f}±{dn[1]:.4f}  ΔMAE {dm[0]:+.4f}±{dm[1]:.4f}")

    print("\n--- band breakdown (pooled held-out errors, career-lens) ---")
    print(f"{'variant':<16} {'dan MAE':>8} {'dan bias':>9} {'kyu MAE':>8} {'kyu bias':>9} "
          f"{'<=10k MAE':>9}")
    for vname, _, _ in VARIANTS:
        errs, gdecl = pooled[(vname, "career-lens")]
        errs, gdecl = np.array(errs), np.array(gdecl)
        dan, kyu, deep = gdecl >= 1, gdecl < 1, gdecl <= -9
        def mae_bias(m):
            if m.sum() == 0:
                return "     -", "      -"
            return f"{np.mean(np.abs(errs[m])):6.3f}", f"{np.mean(errs[m]):+7.3f}"
        dm, db = mae_bias(dan)
        km, kb = mae_bias(kyu)
        pm, _ = mae_bias(deep)
        print(f"{vname:<16} {dm:>8} {db:>9} {km:>8} {kb:>9} {pm:>9}")


if __name__ == "__main__":
    main()
