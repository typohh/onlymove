"""Does a per-user TIME-BENEFIT slope improve the survey rank calibration?

Hypothesis (2026-07-06): the shipped median-time dock assumes every user converts
extra thinking time into accuracy at the same rate — a single global alpha on
log10(median solve min). But users differ in how much additional time actually
helps them (and in whether they would use their clock in real games), so a
per-user "time benefit" B_u could carry signal about declared server ranks
beyond (rating, median time): a user whose accuracy is time-INSENSITIVE holds
the same rating at any time control (their dock should be ~0), while a user who
grinds accuracy out of the clock overstates their timed-server strength exactly
in proportion to the time they lean on.

B_u estimation: over the user's first-time timed attempts, hold their fitted
80%-clear rating R_u fixed and MLE the slope b_u in

    P_i = c_i + (1 − c_i − λ)·σ(a_i·(R_u + b_u·x_i − r_i) + κ_i),
    x_i = log10(t_i) − median_u(log10 t)

— the exact fit_user_rating 4PL with a per-attempt time term (rating units per
decade of time) — then empirical-Bayes shrink each b̂_u toward the population
mean by its Fisher information. NOTE the within-user time↔success confound:
users think longer on problems that are harder FOR THEM than the population
difficulty says, which biases every b̂_u downward — B_u is a RELATIVE trait
measure, not a causal time-value curve; the test is whether the trait predicts
declared ranks, not its absolute level.

Evaluation mirrors the survey yardstick (auxiliary.rating_dev.validate_ratings
_time_adjusted_rho, which validated the shipped dock): per-server Spearman of
the adjusted rating vs declared rank, n-weighted over servers, with adjustment
coefficients chosen by grid on train folds and applied CROSS-FITTED
(grouped-by-user folds; the SAME fold assignments for every candidate so
per-declaration errors pair — unpaired CV deltas are resample noise). Candidates:

    base       R                       (no time adjustment)
    medtime    R + α·T                 (the current production dock's form)
    benefit    R + β·B̃                 (benefit level only)
    med+ben    R + α·T + β·B̃
    inter      R + β·(B·T)             (the hypothesis: dock ∝ benefit × time)
    med+inter  R + α·T + β·(B·T)

  T  = log10 median solve min, residualized on rating (the framework's feature —
       only time UNEXPLAINED by skill drives the dock);
  B̃  = shrunk benefit residualized on rating, unit-SD (incremental-signal form);
  B·T uses the RAW shrunk benefit (B=0 ⇒ no dock — the hypothesis's zero point),
       standardized to unit SD after the product.

--describe adds the existence checks: B_u distribution, split-half reliability
(is there a stable trait at all?), correlations with rating/median time, and a
user-cluster-bootstrapped OLS of the mapping residual (declared internal rank −
M(R), through the SHIPPED curve + couplings) on the features.

Print-only research harness — writes nothing; production stays
pipeline.map_rating_to_rank.

Per-attempt times are MAGNITUDE-based (they enter the 4PL directly, unlike the
rank-based Spearman), so they get a hard AFK cap: anything beyond ~5 minutes on
one problem is almost certainly an idle tab, not thinking (user, 2026-07-06).
The median-time feature keeps the production 180s cap for framework parity.

  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_time_benefit --describe
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_time_benefit
  ./.venv/bin/python3 -m auxiliary.rating_dev.eval_time_benefit --benefit-cap-s 180
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, latest_per_user_server, load_rank_surveys,
)
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, FLOOR_MAX, LOGIT_PER_RATING, fit_user_rating,
)
from utils.ratings_io import load_lapse  # noqa: E402
from pipeline.map_rating_to_rank import (  # noqa: E402
    SIZES, load_floors, load_ratings, load_slopes,
)

# Same knobs as the retired fit_population_constants time-adjust fit (the values
# validate_ratings' _time_adjusted_rho was written against).
MIN_ATTEMPTS = 20        # attempts needed for a user's rating (production mapper)
TA_MIN_TIMED = 20        # timed novel attempts needed for the median-time feature
MIN_TIMED_B = 30         # timed novel attempts needed for an individual b̂_u
TA_MIN_SERVER_USERS = 5  # per-server minimum for a Spearman term
CV_FOLDS = 5
CV_SEEDS = 10
ALPHA_GRID = np.linspace(-1.5, 0.5, 41)    # rating units per log10-minute
BETA_GRID = np.linspace(-1.5, 1.5, 61)     # rating units per unit-SD feature
RIDGE_B = 1.0            # weak ridge (1/τ0²) stabilizing each per-user b̂ MLE
B_CLAMP = 5.0            # |b̂| clamp, rating units per decade of time
MED_TIME_CAP_S = 180.0   # production AFK cap for the MEDIAN-time feature
BENEFIT_CAP_S = 300.0    # per-attempt cap for the benefit slope (≥5 min ≈ AFK)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_user_attempts(sqlite_path, user_ids):
    """{prefix8: [(tid, success, time_ms_or_None)]} — first-time attempts of the
    given (full-id) survey users, in created_at order."""
    con = sqlite3.connect(sqlite_path)
    try:
        qmarks = ",".join("?" * len(user_ids))
        rows = con.execute(
            "SELECT user_id, problem, success, time_used FROM tsumego_reports "
            f"WHERE user_id IN ({qmarks}) AND (is_repeat IS NULL OR is_repeat=0) "
            "ORDER BY created_at, id", list(user_ids)).fetchall()
    finally:
        con.close()
    out = defaultdict(list)
    for uid, tid, success, t in rows:
        out[uid[:8]].append((tid, int(success), float(t) if t else None))
    return out


def fit_time_benefit(rows, R, ratings, slopes, floors, lapse, afk_cap_ms,
                     slow_side=False):
    """MLE of b in P = c + (1−c−λ)·σ(a·(R + b·x − r) + κ) over the user's timed
    first-time attempts, R held fixed, x = log10(t) − median. Ridge RIDGE_B.
    slow_side replaces x with max(0, x) — the literal "does ABOVE-their-norm
    time convert into accuracy" form. Returns (b_hat, fisher_info, n_timed) or
    (None, None, n_timed)."""
    tr = [(ratings[t], s, tm, slopes.get(t), floors.get(t)) for t, s, tm in rows
          if t in ratings and tm is not None and 0 < tm <= afk_cap_ms]
    n = len(tr)
    if n < MIN_TIMED_B:
        return None, None, n
    rr = np.array([r for r, *_ in tr])
    sc = np.array([s for _, s, *_ in tr], float)
    lt = np.log10(np.array([tm for _, _, tm, *_ in tr]) / 60000.0)
    x = lt - np.median(lt)
    if slow_side:
        x = np.maximum(x, 0.0)
    sl = np.array([(a if (a is not None and np.isfinite(a) and a > 0)
                    else LOGIT_PER_RATING) for *_, a, _ in tr])
    c = np.array([min(max(f if (f is not None and np.isfinite(f)) else 0.0, 0.0),
                      FLOOR_MAX) for *_, f in tr])
    lam = max(0.0, float(lapse))
    span = np.clip(1.0 - c - lam, 1e-6, 1.0)
    q = np.clip((CALIBRATION_P - c) / span, 1e-6, 1.0 - 1e-6)
    kappa = np.log(q / (1.0 - q))
    b = 0.0
    for _ in range(80):
        praw = 1.0 / (1.0 + np.exp(-np.clip(sl * (R + b * x - rr) + kappa, -40, 40)))
        p = np.clip(c + span * praw, 1e-9, 1.0 - 1e-9)
        g = span * sl * praw * (1.0 - praw) * x          # dP/db
        denom = p * (1.0 - p)
        score = float(np.sum((sc - p) * g / denom)) - RIDGE_B * b
        info = float(np.sum(g * g / denom)) + RIDGE_B
        step = max(-1.0, min(1.0, score / info))
        b = max(-B_CLAMP, min(B_CLAMP, b + step))
        if abs(step) < 1e-6:
            break
    return float(b), float(info - RIDGE_B), n


def build_users(args):
    """Fit every survey user's rating R, median-time feature T (log10 min), and
    time-benefit b̂ (+ split-half b̂s for the reliability check).
    Returns dict prefix8 -> {R, T, b, info, b_even, b_odd, n_timed}."""
    surveys = latest_per_user_server(load_rank_surveys(args.sql))
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows — run pipeline.fetch_database.")
    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No per-problem ratings — run train_rating first.")
    lapse = load_lapse(args.dir)
    attempts = load_user_attempts(args.sqlite, set(surveys["user_id"].unique()))
    med_cap_ms = MED_TIME_CAP_S * 1000.0     # production median-feature cap
    ben_cap_ms = args.benefit_cap_s * 1000.0  # per-attempt cap for the slope

    users = {}
    for pref, rows in attempts.items():
        pairs = [(t, s) for t, s, _ in rows]
        R, n = fit_user_rating(pairs, merged_ratings, min_n=MIN_ATTEMPTS,
                               slopes=merged_slopes, floors=merged_floors,
                               lapse=lapse)
        if R is None or n < MIN_ATTEMPTS:
            continue
        timed = [tm for t, s, tm in rows
                 if t in merged_ratings and tm is not None and 0 < tm <= med_cap_ms]
        T = (float(np.log10(np.median(timed) / 60000.0))
             if len(timed) >= TA_MIN_TIMED else None)
        b, info, n_timed = fit_time_benefit(rows, R, merged_ratings, merged_slopes,
                                            merged_floors, lapse, ben_cap_ms,
                                            slow_side=args.slow_side)
        b_ev, _, _ = fit_time_benefit(rows[0::2], R, merged_ratings, merged_slopes,
                                      merged_floors, lapse, ben_cap_ms,
                                      slow_side=args.slow_side)
        b_od, _, _ = fit_time_benefit(rows[1::2], R, merged_ratings, merged_slopes,
                                      merged_floors, lapse, ben_cap_ms,
                                      slow_side=args.slow_side)
        users[pref] = {"R": R, "T": T, "b": b, "info": info,
                       "b_even": b_ev, "b_odd": b_od, "n_timed": n_timed}
    return users, surveys


def eb_shrink(users):
    """Empirical-Bayes shrink each b̂_u toward the population mean by Fisher info.
    The between-user variance τ² is the MARGINAL-LIKELIHOOD maximizer of
    b̂_u ~ N(b̄, τ² + 1/info_u) (profile over b̄, grid+refine over τ) — the naive
    method-of-moments var(b̂) − mean(1/info) collapses to 0 here because a few
    low-info users inflate mean(1/info), while the split-half reliability shows
    a real trait. Users without a b̂ get exactly the population mean (⇒ the
    average dock, i.e. today's behavior). Sets u['B'] on every user; returns
    (b_bar, tau2)."""
    have = [u for u in users.values() if u["b"] is not None and u["info"]]
    if len(have) < 5:
        for u in users.values():
            u["B"] = 0.0
        return 0.0, 0.0
    bs = np.array([u["b"] for u in have])
    vs = 1.0 / np.array([u["info"] for u in have])

    def neg_marg_ll(tau2):
        var = tau2 + vs
        mu = float(np.sum(bs / var) / np.sum(1.0 / var))
        return 0.5 * float(np.sum(np.log(var) + (bs - mu) ** 2 / var)), mu

    grid = np.concatenate([[0.0], np.geomspace(1e-4, 4.0, 200)])
    lls = [neg_marg_ll(t)[0] for t in grid]
    tau2 = float(grid[int(np.argmin(lls))])
    _, b_bar = neg_marg_ll(tau2)
    tau2 = max(tau2, 1e-6)
    for u in users.values():
        if u["b"] is None or not u["info"]:
            u["B"] = b_bar
        else:
            prec = 1.0 / tau2
            u["B"] = float((u["info"] * u["b"] + prec * b_bar)
                           / (u["info"] + prec))
    return b_bar, tau2


def build_pairs(users, surveys):
    """[(pref, server, R, rank_scalar)] for users with BOTH a rating and a
    median-time feature — the framework's timed comparison set (identical for
    every candidate)."""
    pairs = []
    for _, row in surveys.iterrows():
        pref = str(row["user_id"])[:8]
        g = row.get("rank_scalar")
        u = users.get(pref)
        if u is None or u["T"] is None or g is None or np.isnan(float(g)):
            continue
        pairs.append((pref, row["server"], u["R"], float(g)))
    return pairs


def make_features(users, pairs):
    """Per-pair feature columns. T residualized on rating; B̃ = shrunk benefit
    residualized on rating, unit SD; BT = RAW shrunk benefit × T_resid, unit SD
    (the standalone hypothesis form — B=0 ⇒ no dock); BTc = CENTERED benefit
    (B − b̄) × T_resid, unit SD (near-orthogonal to T, for joint models — raw
    B·T is ≈ b̄·T + small, i.e. collinear with T). Standardization is over the
    pairs' UNIQUE USERS (not declarations) so a multi-server user doesn't
    dominate the scale."""
    prefs = sorted({p for p, *_ in pairs})
    R = np.array([users[p]["R"] for p in prefs])
    T = np.array([users[p]["T"] for p in prefs])
    B = np.array([users[p]["B"] for p in prefs])
    b_bar = float(B.mean())
    g1, g0 = np.polyfit(R, T, 1)
    Tres = {p: users[p]["T"] - (g0 + g1 * users[p]["R"]) for p in prefs}
    h1, h0 = np.polyfit(R, B, 1)
    Bres = {p: users[p]["B"] - (h0 + h1 * users[p]["R"]) for p in prefs}
    sd_b = float(np.std([Bres[p] for p in prefs])) or 1.0
    BT = {p: users[p]["B"] * Tres[p] for p in prefs}
    sd_bt = float(np.std(list(BT.values()))) or 1.0
    BTc = {p: (users[p]["B"] - b_bar) * Tres[p] for p in prefs}
    sd_btc = float(np.std(list(BTc.values()))) or 1.0
    feats = {p: {"T": Tres[p], "B": Bres[p] / sd_b, "BT": BT[p] / sd_bt,
                 "BTc": BTc[p] / sd_btc}
             for p in prefs}
    return feats


# ---------------------------------------------------------------------------
# Weighted Spearman (vectorized over a coefficient grid)
# ---------------------------------------------------------------------------
def weighted_spearman_grid(pairs, feat_cols, grids):
    """pairs: [(pref, server, R, g)]; feat_cols: list of per-pair feature arrays
    (len = n features); grids: list of 1-D coefficient grids (same length).
    Returns an array shaped like the meshgrid of `grids` (C-order) of the
    n-weighted per-server Spearman of (R + Σ coef_j·feat_j) vs declared rank."""
    mesh = np.meshgrid(*grids, indexing="ij") if grids else [np.zeros(1)]
    coefs = np.stack([m.ravel() for m in mesh], axis=0) if grids else \
        np.zeros((0, 1))
    G = coefs.shape[1]
    by_server = defaultdict(list)
    for i, (_, server, _, _) in enumerate(pairs):
        by_server[server].append(i)
    acc = np.zeros(G)
    wtot = 0.0
    Rv = np.array([p[2] for p in pairs])
    gv = np.array([p[3] for p in pairs])
    F = (np.stack(feat_cols, axis=0) if feat_cols
         else np.zeros((0, len(pairs))))
    for idx in by_server.values():
        if len(idx) < TA_MIN_SERVER_USERS:
            continue
        idx = np.array(idx)
        y = gv[idx]
        if np.ptp(y) == 0:
            continue
        adj = Rv[idx][None, :] + coefs.T @ F[:, idx]     # (G, n)
        rx = rankdata(adj, method="average", axis=1)
        ry = rankdata(y, method="average")
        rx = rx - rx.mean(axis=1, keepdims=True)
        ry = ry - ry.mean()
        den = np.sqrt((rx ** 2).sum(axis=1) * (ry ** 2).sum())
        with np.errstate(invalid="ignore"):
            rho = np.where(den > 0, rx @ ry / den, np.nan)
        ok = np.isfinite(rho)
        if not ok.all():
            rho = np.where(ok, rho, 0.0)
        acc += rho * len(idx)
        wtot += len(idx)
    out = acc / wtot if wtot else np.full(G, np.nan)
    return out.reshape([len(g) for g in grids]) if grids else float(out[0])


CANDIDATES = [
    ("base", []),
    ("medtime  R+aT (prod form)", ["T"]),
    ("benefit  R+bB", ["B"]),
    ("med+ben  R+aT+bB", ["T", "B"]),
    ("inter    R+b(B*T)", ["BT"]),
    ("med+int  R+aT+b(Bc*T)", ["T", "BTc"]),
]


def grid_for(col):
    return ALPHA_GRID if col == "T" else BETA_GRID


def cross_fitted_eval(pairs, feats, seeds=CV_SEEDS, folds=CV_FOLDS, boot=1000):
    """Paired cross-fitted comparison: for every candidate, coefficients are
    grid-picked on train folds and applied to the held-out fold's declarations —
    with the SAME fold assignments across candidates. Reports each candidate's
    pooled cross-fitted weighted Spearman (mean over seeds) and a user-cluster
    bootstrap CI on its delta vs the medtime baseline."""
    prefs = sorted({p for p, *_ in pairs})
    cols_of = {name: cols for name, cols in CANDIDATES}
    # adjusted rating per (candidate, seed, pair) — cross-fitted
    adj = {name: np.zeros((seeds, len(pairs))) for name, _ in CANDIDATES}
    picked = {name: [] for name, _ in CANDIDATES}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(prefs))
        fold = {prefs[j]: i % folds for i, j in enumerate(order)}
        for kf in range(folds):
            tr_idx = [i for i, p in enumerate(pairs) if fold[p[0]] != kf]
            te_idx = [i for i, p in enumerate(pairs) if fold[p[0]] == kf]
            if not te_idx or len(tr_idx) < 10:
                continue
            tr_pairs = [pairs[i] for i in tr_idx]
            for name, cols in CANDIDATES:
                fcols = [np.array([feats[pairs[i][0]][c] for i in tr_idx])
                         for c in cols]
                if cols:
                    surf = weighted_spearman_grid(
                        tr_pairs, fcols, [grid_for(c) for c in cols])
                    best = np.unravel_index(int(np.nanargmax(surf)), surf.shape)
                    coef = [float(grid_for(c)[best[j]])
                            for j, c in enumerate(cols)]
                else:
                    coef = []
                picked[name].append(coef)
                for i in te_idx:
                    p = pairs[i]
                    adj[name][seed, i] = p[2] + sum(
                        cf * feats[p[0]][c] for cf, c in zip(coef, cols))

    def pooled_rho(name, pair_idx, seed_list):
        vals = []
        for seed in seed_list:
            sub = [(pairs[i][0], pairs[i][1], adj[name][seed, i], pairs[i][3])
                   for i in pair_idx]
            vals.append(weighted_spearman_grid(sub, [], []))
        return float(np.mean(vals))

    full = list(range(len(pairs)))
    rho = {name: pooled_rho(name, full, range(seeds)) for name, _ in CANDIDATES}

    # user-cluster bootstrap on the delta vs medtime (paired: same resample, same
    # cross-fitted adjustments)
    by_pref = defaultdict(list)
    for i, p in enumerate(pairs):
        by_pref[p[0]].append(i)
    REF = CANDIDATES[1][0]
    deltas = {name: [] for name, _ in CANDIDATES if name != REF}
    for bnum in range(boot):
        rng = np.random.default_rng(50000 + bnum)
        pick = rng.choice(len(prefs), len(prefs), replace=True)
        idx = [i for j in pick for i in by_pref[prefs[j]]]
        ref = pooled_rho(REF, idx, range(seeds))
        for name, _ in CANDIDATES:
            if name != REF:
                deltas[name].append(pooled_rho(name, idx, range(seeds)) - ref)

    print(f"\nPaired cross-fitted survey-Spearman ({len(pairs)} declarations, "
          f"{len(prefs)} timed users; {seeds}x{folds}-fold grouped by user, "
          f"same folds per candidate; {boot} user-bootstrap runs)")
    print(f"{'candidate':<28}{'rho_cv':>8}{'Δ vs medtime [5,95]':>26}"
          f"{'P(>med)':>9}   median picked coefs")
    for name, cols in CANDIDATES:
        med_coef = ", ".join(
            f"{c}={np.median([cf[j] for cf in picked[name]]):+.2f}"
            for j, c in enumerate(cols)) or "-"
        if name == REF:
            print(f"{name:<28}{rho[name]:>8.4f}{'(reference)':>26}{'':>9}   "
                  f"{med_coef}")
        else:
            d = np.array(deltas[name])
            lo, mid, hi = np.percentile(d, [5, 50, 95])
            pb = float(np.mean(d > 0))
            print(f"{name:<28}{rho[name]:>8.4f}"
                  f"{f'{mid:+.4f} [{lo:+.4f},{hi:+.4f}]':>26}{pb:>9.2f}   "
                  f"{med_coef}")


# ---------------------------------------------------------------------------
# Descriptive stage
# ---------------------------------------------------------------------------
def describe(users, surveys, pairs, feats, args, boot=1000):
    have_b = {p: u for p, u in users.items() if u["b"] is not None}
    bs = np.array([u["b"] for u in have_b.values()])
    print(f"\n--- time-benefit b̂_u (rating units per decade of solve time; "
          f"afk cap {args.benefit_cap_s:.0f}s) ---")
    print(f"users: {len(users)} rated, {len(have_b)} with b̂ "
          f"(≥{MIN_TIMED_B} timed novel attempts)")
    if len(bs):
        q = np.percentile(bs, [5, 25, 50, 75, 95])
        print(f"b̂ distribution: mean {bs.mean():+.2f}  "
              f"[5/25/50/75/95%: {q[0]:+.2f} {q[1]:+.2f} {q[2]:+.2f} "
              f"{q[3]:+.2f} {q[4]:+.2f}]")
    both = [(u["b_even"], u["b_odd"]) for u in have_b.values()
            if u["b_even"] is not None and u["b_odd"] is not None]
    if len(both) >= 10:
        be = np.array([x for x, _ in both]); bo = np.array([y for _, y in both])
        r_half = float(np.corrcoef(be, bo)[0, 1])
        r_full = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
        print(f"split-half reliability (n={len(both)}): r={r_half:+.2f} "
              f"(Spearman-Brown full-length {r_full:+.2f})")
    Rv = np.array([u["R"] for u in have_b.values()])
    Tv = np.array([u["T"] if u["T"] is not None else np.nan
                   for u in have_b.values()])
    ok = np.isfinite(Tv)
    print(f"corr(b̂, rating) = {np.corrcoef(bs, Rv)[0, 1]:+.2f}   "
          f"corr(b̂, log10 med-min) = {np.corrcoef(bs[ok], Tv[ok])[0, 1]:+.2f}")

    # Mapping residual (through the SHIPPED curve + couplings) vs features.
    from auxiliary.check_mapped_vs_declared_rank import (
        apply_M, declared_to_internal, load_M, load_server_curves)
    m_r, m_i = load_M(args.dir)
    curves = load_server_curves(args.dir)
    rows = []
    for pref, server, R, g in pairs:
        resid = declared_to_internal(server, g, curves) - apply_M(R, m_r, m_i)
        f = feats[pref]
        rows.append((pref, resid, f["T"], f["B"], f["BTc"]))
    y = np.array([r[1] for r in rows])
    X = np.column_stack([np.ones(len(rows)),
                         [r[2] for r in rows], [r[3] for r in rows],
                         [r[4] for r in rows]])
    by_user = defaultdict(list)
    for r in rows:
        by_user[r[0]].append(r)
    keys = list(by_user)
    print(f"\nOLS of mapping residual (declared internal − M(R), ranks) on the "
          f"features ({len(rows)} declarations, {len(keys)} users), "
          f"user-cluster bootstrap [5/50/95]:")
    samp = defaultdict(list)
    for bnum in range(boot):
        rng = np.random.default_rng(bnum)
        pick = rng.choice(len(keys), len(keys), replace=True)
        br = [r for j in pick for r in by_user[keys[j]]]
        yb = np.array([r[1] for r in br])
        Xb = np.column_stack([np.ones(len(br)), [r[2] for r in br],
                              [r[3] for r in br], [r[4] for r in br]])
        try:
            beta, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        except np.linalg.LinAlgError:
            continue
        for k, v in zip(("const", "T_resid", "B̃ (per SD)", "Bc·T (per SD)"), beta):
            samp[k].append(v)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - pred) ** 2)) / ss if ss > 0 else float("nan")
    for k in ("const", "T_resid", "B̃ (per SD)", "Bc·T (per SD)"):
        v = np.array(samp[k])
        lo, mid, hi = np.percentile(v, [5, 50, 95])
        psign = max(float(np.mean(v > 0)), float(np.mean(v < 0)))
        print(f"  {k:<14} {mid:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"P(consistent sign) {psign:.2f}")
    print(f"  in-sample R² {r2:+.3f}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--sqlite",
                    default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--benefit-cap-s", type=float, default=BENEFIT_CAP_S,
                    help="AFK cap (s) on the per-attempt times feeding the "
                         "benefit slope — magnitude-based time use, so ≥~5 min "
                         "is almost certainly AFK, not thinking. The median-time "
                         f"feature always keeps the production {MED_TIME_CAP_S:.0f}s cap.")
    ap.add_argument("--slow-side", action="store_true",
                    help="Fit the benefit slope on max(0, x) — above-their-norm "
                         "time only (the literal 'additional time' form).")
    ap.add_argument("--describe", action="store_true",
                    help="Print the b̂ distribution / reliability / residual-"
                         "regression existence checks (skips the CV eval).")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--cv-seeds", type=int, default=CV_SEEDS)
    args = ap.parse_args()

    users, surveys = build_users(args)
    b_bar, tau2 = eb_shrink(users)
    print(f"{len(users)} survey users rated; population benefit mean "
          f"b̄={b_bar:+.3f} rating/decade, between-user τ={np.sqrt(tau2):.3f}")
    pairs = build_pairs(users, surveys)
    feats = make_features(users, pairs)

    if args.describe:
        describe(users, surveys, pairs, feats, args, boot=args.boot)
    else:
        cross_fitted_eval(pairs, feats, seeds=args.cv_seeds, boot=args.boot)
    print("\n(no files written — research harness; production stays "
          "pipeline.map_rating_to_rank)")


if __name__ == "__main__":
    main()
