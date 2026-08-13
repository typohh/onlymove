"""EXPERIMENTAL copy of pipeline.map_rating_to_rank — research only, writes nothing
by default (so it can never clobber the shipped ratings/ curve).

Two questions this harness answers, both raised 2026-06-29:

  (A) MEDIAN SOLVE TIME AS A LEARNED PARAMETER.  Today the time signal lives
      OUTSIDE the mapping: pipeline.fit_population_constants fits a rank-residualized
      time dock by a separate Spearman sweep, and the client applies it only to the
      SERVER translations (the onlymove header rank stays raw).  Here we instead fold
      a single learned scalar `gamma` (rating units per de-meaned log10-minute of a
      user's median solve time) straight INTO the joint survey fit: every user's
      effective rating becomes `R + gamma * (logT_u - mean logT)`, and `gamma` is
      learned jointly with the rating→EGF curve + the server couplings by the same
      ordinal likelihood.  `--time {off,all,server}` controls whether the time term
      enters the EGF anchor too (`all`) or only the non-EGF server declarations
      (`server`, matching the production philosophy).

  (B) STOP OTHER SERVERS CONTAMINATING THE EGF ANCHOR.  EGF is the calibration hub,
      but in the production joint fit the SHARED rating→EGF curve `a` appears in BOTH
      the EGF likelihood AND every server's likelihood (each server maps its users
      through `a` to get their internal rank), so a server with inflated/deflated
      declarations PULLS the shared curve, and its bias leaks into where EGF users land.
      MEASURED (the `--sweep` per-system table): EGF carries a systematic bias of -1.27
      (the worst of all 8 systems) while every server sits near zero, because every
      server has FREE coupling thresholds that absorb its own offset but EGF is the
      identity hub with NO offset of its own, so the shared-curve residual lands on EGF.
      That confirms "EGF soaks up the bias." Two levers to prioritize the EGF anchor:
        --egf-weight W      multiply the EGF likelihood by W (soft; W=1 = production).
        --curve-egf-only    fit the curve + gamma on EGF declarations ALONE, FREEZE it,
                            then fit each server coupling against the frozen curve
                            (hard: a server can no longer move the EGF mapping at all).
      Both DO drive EGF's bias to ~0, but only by starving the kyu range (EGF is
      kyu-sparse; the servers are its only kyu anchor), so the kyu band blows up to
      +8..+12 and overall RMSE degrades. The trade is fundamental (relaxing --lambda-shape
      does NOT decouple it; see docs/methodology.md).

  (C) METHODOLOGY (user, 2026-06-29): assigned-vs-declared is compared on the SHIPPED
      production TIME-ADJUSTED rating R_prod (the default), not the raw rating, because
      every declared server/EGF rank reflects timed play. `--raw-rating` reverts to R_raw.

FRAMING DISCIPLINE (see the project memory / docs):  there is NO overall population
bias in this mapping.  The only real effect is the dan MINORITY sitting ~2 stones
below their declared rank, and MOST of that is UPSTREAM rating compression (kyu and
dan 80%-clear ratings overlap, and a single-valued curve cannot separate overlapping
groups) which NO mapper change can fix.  So judge the levers by the EGF-anchor bias
PER BAND and the per-system table, under grouped-by-user CV (in-sample EGF bias is
optimistic), NOT by a single cohort-mean number.

Run pipeline.fetch_database first (refreshes db/d1.{sql,sqlite}).  Then e.g.:

  ./.venv/bin/python3 -m auxiliary.map_rating_to_rank_dev --sweep
  ./.venv/bin/python3 -m auxiliary.map_rating_to_rank_dev --sweep --cv
  ./.venv/bin/python3 -m auxiliary.map_rating_to_rank_dev --egf-weight 10 --time all
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, latest_per_user_server, load_attempts_for_users,
    load_rank_surveys,
)
from utils.rating.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec, ordinal_nll_vec,
    _unpack_thresholds,
)
from utils.rating.ratings_io import load_lapse, load_rating_rank_curve  # noqa: E402
from utils.rating.rank_curve_shape import fit_shape, rank_to_axis  # noqa: E402
from pipeline.map_rating_to_rank import (  # noqa: E402
    RATING_SHAPE, SIZES, load_floors, load_ratings, load_slopes,
)


def _shape_at(ys, shape=RATING_SHAPE):
    """even_game_winrates.txt SHAPE value at each EGF-boundary scalar in `ys`
    (rank_index = scalar + 19). `shape` is matched as a prefix of the
    rank_curve_shape models — 'cubic' (production), 'linear', 'exp', ..."""
    params, theta = fit_shape(shape)
    return np.asarray(theta(rank_to_axis(ys), params), float)
from auxiliary.check_mapped_vs_declared_rank import (  # noqa: E402
    apply_M, declared_to_internal, load_M, load_server_curves,
    load_time_adjust, load_user_median_solve_minutes,
)

# EGF declared-scalar bands for the per-band bias report (1k=0, 1d=1; no zero gap).
EGF_BANDS = [
    (-1e9, -5, "10k+"),
    (-5, 0, "5k-1k"),
    (0, 3, "1d-3d"),
    (3, 1e9, "4d+"),
]


def _time_feature(T, form):
    """Transform the de-meaned log10-median-minutes feature T for the joint fit.

      linear   : T as-is — a SYMMETRIC dock (slow→down, fast→up). With a free curve
                 intercept this only ROTATES users around the time centroid (fixes
                 within-group spread, the level is re-absorbed; see the docs).
      onesided : max(0, −T) = "how much FASTER than the population-median solver this
                 user is", 0 for slower-than-median. With gamma>0 it BOOSTS the fast
                 (whose untimed rating is censored-from-below by rushing) and never
                 debits the slow — the only form the prior work found moves the LEVEL.
    """
    T = np.asarray(T, float)
    return np.maximum(0.0, -T) if form == "onesided" else T


# ---------------------------------------------------------------------------
# Feature assembly (shared by every config so they're compared on one dataset)
# ---------------------------------------------------------------------------
def build_decls(args):
    """Return (decls, meta) where decls is a list of (server, R_raw, R_prod, g, T):

      R_raw  : the user's 80%-clear pooled v2-rating (same estimator the client +
               production mapper use).
      R_prod : R_raw + the SHIPPED production time dock — alpha·(log10 median-min −
               (g0 + g1·rank_index)), the rank-residualized adjustment
               population_constants.json already ships, with the user's native
               rank_index taken through the SHIPPED M.  This is the "time-adjusted
               rating" the assigned-vs-declared comparison should use: practice here
               is untimed but every declared server/EGF rank reflects timed play, so a
               solver slower than typical for their rank reads weaker on a timed ladder.
               R_prod == R_raw for a user without a usable median-time feature.
      T      : (log10 median solve minutes − population mean), or 0.0 — the de-meaned
               feature the LEARNED gamma uses (an alternative to the fixed prod dock).

    The fit/eval pick R_raw or R_prod via the `prod_time` flag, so raw / production-
    time-adjusted / learned-gamma are all compared on one declaration set."""
    surveys = latest_per_user_server(load_rank_surveys(args.sql))
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows — run pipeline.fetch_database.")

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No v2 per-problem ratings — run train_mirt first.")
    lapse = load_lapse(args.dir)

    # Shipped M (rating → rank_index, 20k=0 axis) + production time dock constants,
    # used ONLY to compute each user's fixed R_prod (NOT the curve being fit).
    m_xs, m_idx = load_rating_rank_curve(args.dir, size=str(SIZES[0]), offset=0.0)
    order = np.argsort(m_xs)
    m_xs, m_idx = m_xs[order], m_idx[order]
    # load_time_adjust returns the CONSENSUS dock (alpha, center) — the old
    # rank-residualized (alpha, g0, g1) form is retired, so express the shipped
    # dock in the old machinery as g0=center, g1=0 (no rank line).
    ta = load_time_adjust(args.dir)               # (alpha, center) or None
    if ta is None:
        print("  WARNING: no time_adjust block — R_prod falls back to R_raw")
    alpha, g0, g1 = (ta[0], ta[1], 0.0) if ta else (0.0, 0.0, 0.0)

    # first_time_only: mirror the production anchor + the client's first-time-only
    # estimator; repeats would inflate the 80%-clear rating this harness fits.
    attempts = load_attempts_for_users(args.sql, set(surveys["user_id"].unique()),
                                       first_time_only=True)
    med_min = load_user_median_solve_minutes(args.sqlite)   # {pref8: median minutes}

    user_R, user_T, user_dock = {}, {}, {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < args.min_attempts:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=args.min_attempts,
                               slopes=merged_slopes, floors=merged_floors, lapse=lapse)
        if R is None or n < args.min_attempts:
            continue
        user_R[pref] = R
        mm = med_min.get(pref)
        if mm is not None and mm > 0:
            user_T[pref] = float(np.log10(mm))
            native_idx = float(interp_extrap_vec(np.array([R]), m_xs, m_idx)[0])
            user_dock[pref] = alpha * (np.log10(mm) - (g0 + g1 * native_idx))

    # De-mean the time feature over the users that BOTH have a rating and a time
    # feature (so a uniform shift is absorbed by the curve's intercept, leaving
    # gamma to capture only the fast/slow-AT-FIXED-RATING differential).
    timed = [user_T[p] for p in user_R if p in user_T]
    mean_logT = float(np.mean(timed)) if timed else 0.0

    decls = []
    for _, row in surveys.iterrows():
        pref = str(row["user_id"])[:8]
        g = row.get("rank_value")
        if pref not in user_R or g is None or pd.isna(g):
            continue
        T = (user_T[pref] - mean_logT) if pref in user_T else 0.0
        R_raw = user_R[pref]
        R_prod = R_raw + user_dock.get(pref, 0.0)
        decls.append((row["server"], R_raw, R_prod, float(g), T))
    meta = {"mean_logT": mean_logT, "n_timed": len(timed),
            "alpha": alpha, "g0": g0, "g1": g1}
    return decls, meta


# ---------------------------------------------------------------------------
# Joint fit (curve + gamma + server couplings), with the two new levers
# ---------------------------------------------------------------------------
def _boxcox(m, d):
    """Box-Cox of median solve minutes: log(m) at d->0 (the production log-time form),
    linear (m-1) at d=1 — so the fitted exponent says WHERE on that spectrum time wants
    to sit."""
    m = np.asarray(m, float)
    return np.log(m) if abs(d) < 1e-9 else (np.power(m, d) - 1.0) / d


def _fit_linear_time(y, Rc, m, interaction, d=0.0):
    """resid_rank = a0 + (a1 + a2*Rc)*boxcox(m; d), linear least squares at FIXED d
    (d=0 => log time, the production form; d=1 => linear-in-minutes). interaction=False
    drops a2. Returns (params, sse, r2)."""
    h = _boxcox(m, d)
    cols = [np.ones_like(y), h] + ([Rc * h] if interaction else [])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    sse = float(np.sum((y - pred) ** 2))
    sstot = float(np.sum((y - y.mean()) ** 2))
    return ({"a0": float(beta[0]), "a1": float(beta[1]),
             "a2": float(beta[2]) if interaction else 0.0, "d": float(d)},
            sse, 1 - sse / sstot if sstot > 0 else float("nan"))


def _fit_full_time(y, Rc, m):
    """resid_rank = a0 + (a1 + a2*Rc)*boxcox(m;d), free time exponent d."""
    from scipy.optimize import least_squares

    def resid(p):
        a0, a1, a2, d = p
        return a0 + (a1 + a2 * Rc) * _boxcox(m, d) - y

    res = least_squares(resid, [0.0, 0.0, 0.0, 0.0],
                        bounds=([-np.inf, -np.inf, -np.inf, -1.0],
                                [np.inf, np.inf, np.inf, 2.0]), max_nfev=10000)
    a0, a1, a2, d = res.x
    sse = float(np.sum(resid(res.x) ** 2))
    sstot = float(np.sum((y - y.mean()) ** 2))
    return ({"a0": float(a0), "a1": float(a1), "a2": float(a2), "d": float(d)},
            sse, 1 - sse / sstot if sstot > 0 else float("nan"))


def time_form_study(args, boot=300):
    """Discover the FORM the time adjustment wants: regress the rank residual
    (declared_internal - M(rating)) on a rating-modulated, Box-Cox-shaped time term
    resid = a0 + (a1 + a2*(R-Rbar))*boxcox(median_min; d), and user-cluster-bootstrap
    to get each parameter's distribution. a1 = mean time slope (rank units per Box-Cox
    unit; curve-relative level), a2 = rating-interaction (does time matter more for the
    strong?), d = time shape (0 = log-time, the production form; 1 = linear)."""
    surveys = latest_per_user_server(load_rank_surveys(args.sql))
    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    lapse = load_lapse(args.dir)
    m_ratings, m_internal = load_M(args.dir)
    server_curves = load_server_curves(args.dir)
    # first_time_only: mirror the production anchor + the client's first-time-only
    # estimator; repeats would inflate the 80%-clear rating this harness fits.
    attempts = load_attempts_for_users(args.sql, set(surveys["user_id"].unique()),
                                       first_time_only=True)
    med_min = load_user_median_solve_minutes(args.sqlite)

    user_R = {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < args.min_attempts:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=args.min_attempts,
                               slopes=merged_slopes, floors=merged_floors, lapse=lapse)
        if R is not None and n >= args.min_attempts:
            user_R[pref] = R

    rows = []   # (user, resid_rank, R, median_min)
    for _, r in surveys.iterrows():
        pref = str(r["user_id"])[:8]
        g = r.get("rank_value")
        if pref not in user_R or pref not in med_min or g is None or pd.isna(g):
            continue
        dec_int = declared_to_internal(r["server"], float(g), server_curves)
        resid = dec_int - apply_M(user_R[pref], m_ratings, m_internal)
        rows.append((pref, float(resid), user_R[pref], med_min[pref]))
    users = sorted({u for u, *_ in rows})
    by_user = defaultdict(list)
    for row in rows:
        by_user[row[0]].append(row)
    y = np.array([r[1] for r in rows]); Rv = np.array([r[2] for r in rows])
    mv = np.array([r[3] for r in rows]); Rbar = float(Rv.mean())
    print(f"{len(rows)} declarations from {len(users)} timed users; "
          f"mean rating {Rbar:+.2f}, median solve {np.median(mv)*60:.0f}s\n")

    keys = list(by_user)
    models = (("log (d=0), no interaction", lambda yy, RR, mm: _fit_linear_time(yy, RR, mm, False, 0.0)),
              ("linear (d=1), no interaction", lambda yy, RR, mm: _fit_linear_time(yy, RR, mm, False, 1.0)),
              ("log + rating-interaction", lambda yy, RR, mm: _fit_linear_time(yy, RR, mm, True, 0.0)),
              ("linear + rating-interaction", lambda yy, RR, mm: _fit_linear_time(yy, RR, mm, True, 1.0)),
              ("Box-Cox (free d) + interaction", _fit_full_time))

    print("Model fits — IN-SAMPLE vs HELD-OUT R^2 (grouped-by-user CV) with a "
          "user-bootstrap CI on the held-out R^2, plus a PAIRED held-out delta vs LOG")
    print("(same CV splits for every model, so the per-declaration errors pair). "
          "A delta CI clear of 0 = a real held-out difference.\n")

    # One pass of fixed CV splits, predicting every held-out declaration under EVERY
    # model (shared splits => predictions are paired per declaration).
    SEEDS, FOLDS, NB = 20, 5, 500
    inst_u, inst_y, P = [], [], {name: [] for name, _ in models}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        fof = {keys[order[i]]: i % FOLDS for i in range(len(keys))}
        for f in range(FOLDS):
            tr = [r for k in keys if fof[k] != f for r in by_user[k]]
            te = [(k, r) for k in keys if fof[k] == f for r in by_user[k]]
            if len(tr) < 5 or not te:
                continue
            ytr = np.array([r[1] for r in tr]); Rtr = np.array([r[2] for r in tr]) - Rbar
            mtr = np.array([r[3] for r in tr])
            fitted = {}
            for name, fitter in models:
                try:
                    fitted[name] = fitter(ytr, Rtr, mtr)[0]
                except Exception:
                    fitted[name] = None
            for k, r in te:
                inst_u.append(k); inst_y.append(r[1])
                for name, _ in models:
                    p = fitted[name]
                    P[name].append(np.nan if p is None else
                                   p["a0"] + (p["a1"] + p["a2"] * (r[2] - Rbar))
                                   * _boxcox(r[3], p["d"]))
    inst_u = np.array(inst_u); inst_y = np.array(inst_y)
    P = {name: np.array(v) for name, v in P.items()}
    uniq = sorted(set(inst_u))
    uidx = {u: np.where(inst_u == u)[0] for u in uniq}

    def r2(idx, name):
        yy = inst_y[idx]; pr = P[name][idx]; ok = np.isfinite(pr)
        sstot = float(np.sum((yy - yy.mean()) ** 2))
        return (1 - float(np.sum((yy[ok] - pr[ok]) ** 2)) / sstot) if sstot > 0 else np.nan

    full = np.arange(len(inst_y))
    boot_idx = []
    for b in range(NB):
        rng = np.random.default_rng(70000 + b)
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        boot_idx.append(np.concatenate([uidx[uniq[j]] for j in pick]))
    REF = models[0][0]   # log, no interaction
    for name, fitter in models:
        p, _, r2in = fitter(y, Rv - Rbar, mv)
        r2cv = r2(full, name)
        bs = np.array([r2(ix, name) for ix in boot_idx])
        lo, hi = np.percentile(bs, [5, 95])
        if name == REF:
            extra = "  (reference)"
        else:
            dif = np.array([r2(ix, name) - r2(ix, REF) for ix in boot_idx])
            dlo, dmid, dhi = np.percentile(dif, [5, 50, 95])
            pbet = float(np.mean(dif > 0))
            extra = (f"  Δvs-log {dmid:+.3f} [{dlo:+.3f},{dhi:+.3f}] "
                     f"P(>log) {pbet:.2f}")
        print(f"  {name:<31} in {r2in:+.3f}  held-out {r2cv:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]{extra}")

    # User-cluster bootstrap. Two models: the full Box-Cox (does d/the shape want to
    # move off log-time?) and the well-identified log model (clean a1/a2, since a1 in
    # Box-Cox is entangled with d's scale).

    def boot_model(fitter):
        samp = defaultdict(list)
        for b in range(boot):
            rng = np.random.default_rng(b)
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            br = [row for j in pick for row in by_user[keys[j]]]
            yb = np.array([r[1] for r in br]); Rb = np.array([r[2] for r in br])
            mb = np.array([r[3] for r in br])
            try:
                p, _, _ = fitter(yb, Rb - Rbar, mb)
            except Exception:
                continue
            for k in p:
                samp[k].append(p[k])
        return samp

    def report(samp, labels):
        for k, lbl in labels:
            v = np.array(samp.get(k, []))
            if v.size == 0:
                continue
            lo, mid, hi = np.percentile(v, [5, 50, 95])
            psign = max(float(np.mean(v > 0)), float(np.mean(v < 0)))
            print(f"  {lbl}: {mid:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
                  f"P(consistent sign) {psign:.2f}")

    print(f"\nUser-cluster bootstrap ({boot} runs), LOG model d=0 (well-identified) "
          "[5/50/95, P(sign)]:")
    report(boot_model(lambda y_, Rc_, m_: _fit_linear_time(y_, Rc_, m_, True, 0.0)),
           [("a1", "a1 time slope (rank/log_e-min) "),
            ("a2", "a2 rating-interaction         ")])
    print(f"\nUser-cluster bootstrap ({boot} runs), LINEAR model d=1 (rank per minute) "
          "[5/50/95, P(sign)]:")
    report(boot_model(lambda y_, Rc_, m_: _fit_linear_time(y_, Rc_, m_, True, 1.0)),
           [("a1", "a1 time slope (rank/minute)    "),
            ("a2", "a2 rating-interaction         ")])
    print(f"\nUser-cluster bootstrap ({boot} runs), BOX-COX model (does the shape move?) "
          "[5/50/95, P(sign)]:")
    report(boot_model(_fit_full_time),
           [("a1", "a1 time slope (rank/BoxCox)    "),
            ("a2", "a2 rating-interaction         "),
            ("d", "d  exponent (0=log, 1=linear)  ")])
    print("\n(no files written — research harness)")


def _curve_arrays(egf_lo, egf_hi, shape=RATING_SHAPE):
    egf_thr_ranks = list(range(egf_lo + 1, egf_hi + 1))
    ys = np.array([g - 0.5 for g in egf_thr_ranks], dtype=float)
    S = _shape_at(ys, shape)
    dS = np.diff(S)
    return egf_thr_ranks, ys, S, dS


def _decl_rating(d, prod_time):
    """Pick R_prod (time-adjusted) or R_raw from a (server, R_raw, R_prod, g, T) decl."""
    return d[2] if prod_time else d[1]


def _bucket(decls, egf_lo, egf_hi, rank_padding, extend_to, min_server_samples,
            time_form="linear", prod_time=False):
    """EGF arrays + per-server blocks (each carrying its rows' rating, k, and the
    form-transformed time feature T)."""
    N_egf = egf_hi - egf_lo + 1
    egf_R, egf_k, egf_T = [], [], []
    server_rows = defaultdict(list)
    for d in decls:
        server, g, T = d[0], d[3], d[4]
        R = _decl_rating(d, prod_time)
        gi = int(round(g))
        Tf = float(_time_feature(T, time_form))
        if server == "EGF":
            egf_R.append(R)
            egf_k.append(max(0, min(N_egf - 1, gi - egf_lo)))
            egf_T.append(Tf)
        else:
            server_rows[server].append((R, gi, Tf))
    egf = (np.asarray(egf_R, float), np.asarray(egf_k, int), np.asarray(egf_T, float))

    blocks = {}
    for server, rows in server_rows.items():
        if len(rows) < min_server_samples:
            continue
        declared = [g for _, g, _ in rows]
        g_min = min(declared) - rank_padding
        g_max = max(max(declared) + rank_padding, extend_to)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        blocks[server] = {
            "g_min": g_min, "g_max": g_max,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "rating": np.array([r for r, _, _ in rows], float),
            "T": np.array([t for _, _, t in rows], float),
            "k": np.array([max(0, min(N - 1, g - g_min)) for _, g, _ in rows], int)}
    return egf, blocks


def _fit_curve_params(egf, S, dS, x0_rating, lam_shape, lam_time, time_on,
                      egf_weight=1.0, blocks=None, ys=None, lam=0.0):
    """Fit [alpha, log beta, delta_1..nseg, gamma] (+ optionally server blocks).
    When `blocks`/`ys`/`lam` are given the server likelihoods are included in the
    SAME minimize (the production joint fit, with the EGF term scaled by egf_weight).
    When they are None this is the EGF-ONLY curve fit (the --curve-egf-only stage 1).
    Returns (a, gamma, server_param_slices, full_x)."""
    egf_R, egf_k, egf_T = egf
    nseg = len(dS)

    srv_layout = []
    x0 = list(x0_rating) + [0.0] * nseg + [0.0]   # + gamma
    idx = 2 + nseg + 1
    if blocks:
        for server, blk in blocks.items():
            n = len(blk["baseline"]) + 1
            srv_layout.append((server, idx, n))
            x0.append(float(blk["g_min"]))
            x0.extend([0.0] * len(blk["baseline"]))
            idx += n
    x0 = np.asarray(x0, float)

    def unpack(params):
        beta = np.exp(params[1])
        a = np.empty(len(S))
        a[0] = params[0] + beta * S[0]
        if nseg:
            gaps = beta * dS * np.exp(params[2:2 + nseg])
            a[1:] = a[0] + np.cumsum(gaps)
        gamma = params[2 + nseg] if time_on else 0.0
        return a, gamma

    def objective(params):
        a, gamma = unpack(params)
        nll = egf_weight * ordinal_nll_vec(a, egf_R + gamma * egf_T, egf_k)
        pen = 0.0
        if blocks:
            for server, st, n in srv_layout:
                blk = blocks[server]
                thr = _unpack_thresholds(params[st:st + n], blk["baseline"])
                x_int = interp_extrap_vec(blk["rating"] + gamma * blk["T"], a, ys)
                nll += ordinal_nll_vec(thr, x_int, blk["k"])
                pen += lam * float(np.sum(params[st + 1:st + n] ** 2))
        if nseg:
            pen += lam_shape * float(np.sum(params[2:2 + nseg] ** 2))
        pen += lam_time * float(gamma ** 2)
        return nll + pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 2000})
    a, gamma = unpack(res.x)
    return a, gamma, srv_layout, res.x


def fit(decls, egf_lo=-20, egf_hi=10, lam=5.0, lam_shape=3.0, lam_time=0.1,
        rank_padding=2, extend_to=9, min_server_samples=3,
        time_on=True, egf_weight=1.0, curve_egf_only=False, time_form="linear",
        prod_time=False, curve_shape=RATING_SHAPE):
    """Returns (a, gamma, ys, egf_thr_ranks, server_fits)."""
    egf_thr_ranks, ys, S, dS = _curve_arrays(egf_lo, egf_hi, shape=curve_shape)
    egf, blocks = _bucket(decls, egf_lo, egf_hi, rank_padding, extend_to,
                          min_server_samples, time_form=time_form, prod_time=prod_time)
    egf_R, egf_k, _ = egf
    if egf_R.size == 0:
        sys.exit("No EGF declarations — cannot anchor the internal-rank scale.")

    # init the affine from a robust EGF linear pre-fit (as production does)
    beta, alpha = np.polyfit(egf_R, egf_k.astype(float) + egf_lo, 1)
    if beta <= 0:
        beta = max(beta, 1e-3)
    prior_rating = (ys - alpha) / beta
    b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
    a0 = float(prior_rating[0] - b0 * S[0])
    x0_rating = [a0, float(np.log(max(b0, 1e-3)))]

    if curve_egf_only:
        # Stage 1: curve + gamma on EGF alone.  Stage 2: each server vs frozen curve.
        a, gamma, _, _ = _fit_curve_params(egf, S, dS, x0_rating, lam_shape,
                                           lam_time, time_on)
        server_fits = _fit_servers_frozen(blocks, a, ys, gamma, lam)
    else:
        a, gamma, srv_layout, full_x = _fit_curve_params(
            egf, S, dS, x0_rating, lam_shape, lam_time, time_on,
            egf_weight=egf_weight, blocks=blocks, ys=ys, lam=lam)
        server_fits = {}
        for server, st, n in srv_layout:
            blk = blocks[server]
            thr = _unpack_thresholds(full_x[st:st + n], blk["baseline"])
            gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
            server_fits[server] = {gs[i]: float(thr[i]) for i in range(len(gs))}
    return a, gamma, ys, egf_thr_ranks, server_fits


def _fit_servers_frozen(blocks, a, ys, gamma, lam):
    """Fit each server coupling against a FROZEN curve (the --curve-egf-only path)."""
    fits = {}
    for server, blk in blocks.items():
        baseline = blk["baseline"]
        n = len(baseline) + 1
        x_int = interp_extrap_vec(blk["rating"] + gamma * blk["T"], a, ys)

        def obj(p):
            thr = _unpack_thresholds(p, baseline)
            return ordinal_nll_vec(thr, x_int, blk["k"]) + lam * float(np.sum(p[1:] ** 2))

        x0 = np.concatenate([[float(blk["g_min"])], np.zeros(len(baseline))])
        res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 2000})
        thr = _unpack_thresholds(res.x, baseline)
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        fits[server] = {gs[i]: float(thr[i]) for i in range(len(gs))}
    return fits


# ---------------------------------------------------------------------------
# Consensus model: rating -> LATENT rank -> every server INCLUDING EGF (each with
# its own free coupling). EGF is no longer the identity hub; the latent scale is
# pinned by a ridge of EGF's coupling toward identity (egf_anchor): egf_anchor -> inf
# recovers today's EGF-is-identity model, finite lets EGF absorb its own systematic
# offset in its coupling (so it no longer dumps the cross-server residual onto the
# shared curve), egf_anchor -> 0 is a fully democratic consensus.
# ---------------------------------------------------------------------------
def _bucket_all(decls, rank_padding, extend_to, min_server_samples,
                time_form="linear", prod_time=False):
    """One coupling block per server INCLUDING EGF (rating, k, time feature, plus the
    identity-line target the latent ridge pulls EGF toward)."""
    server_rows = defaultdict(list)
    for d in decls:
        server, g, T = d[0], d[3], d[4]
        server_rows[server].append((_decl_rating(d, prod_time), int(round(g)),
                                    float(_time_feature(T, time_form))))
    blocks = {}
    for server, rows in server_rows.items():
        if len(rows) < min_server_samples:
            continue
        declared = [g for _, g, _ in rows]
        g_min = min(declared) - rank_padding
        g_max = max(max(declared) + rank_padding, extend_to)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        gs = list(range(g_min + 1, g_max + 1))
        blocks[server] = {
            "g_min": g_min, "g_max": g_max,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "identity": np.array([g - 0.5 for g in gs], float),   # latent==server line
            "rating": np.array([r for r, _, _ in rows], float),
            "T": np.array([t for _, _, t in rows], float),
            "k": np.array([max(0, min(N - 1, g - g_min)) for _, g, _ in rows], int)}
    return blocks


def fit_consensus(decls, egf_lo=-20, egf_hi=10, lam=5.0, lam_shape=3.0, lam_time=0.1,
                  rank_padding=2, extend_to=9, min_server_samples=3, time_on=True,
                  time_form="linear", prod_time=False, egf_anchor=30.0,
                  curve_shape=RATING_SHAPE, anchor_mode="egf"):
    """Returns (a, gamma, ys, fits) where `a` maps rating -> LATENT consensus rank and
    `fits` carries a coupling for every server (EGF included).

    anchor_mode pins the otherwise-free latent scale (gauge fix):
      "egf" : ridge ONLY EGF's coupling toward identity (latent ≈ EGF; EGF privileged).
      "all" : ridge EVERY server's coupling toward identity equally with `egf_anchor` as
              the shared weight — SYMMETRIC consensus, no server special. The latent is
              then the point that minimizes total deviation from all servers."""
    egf_thr_ranks, ys, S, dS = _curve_arrays(egf_lo, egf_hi, shape=curve_shape)
    nseg = len(dS)
    blocks = _bucket_all(decls, rank_padding, extend_to, min_server_samples,
                         time_form=time_form, prod_time=prod_time)
    if "EGF" not in blocks:
        sys.exit("No EGF block — cannot anchor the latent scale.")

    # init curve affine from an EGF linear pre-fit (mirrors the production init)
    egf = blocks["EGF"]
    beta, alpha = np.polyfit(egf["rating"], egf["k"] + egf["g_min"], 1)
    if beta <= 0:
        beta = 1e-3
    prior_rating = (ys - alpha) / beta
    b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
    a0 = float(prior_rating[0] - b0 * S[0])

    srv_layout = []
    x0 = [a0, float(np.log(max(b0, 1e-3)))] + [0.0] * nseg + [0.0]   # +gamma
    idx = 2 + nseg + 1
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
        gamma = params[2 + nseg] if time_on else 0.0
        return a, gamma

    def objective(params):
        a, gamma = unpack_curve(params)
        nll, pen, off_sum, w_sum, w_tot = 0.0, 0.0, 0.0, 0.0, 0.0
        for server, st, n in srv_layout:
            blk = blocks[server]
            thr = _unpack_thresholds(params[st:st + n], blk["baseline"])
            x_int = interp_extrap_vec(blk["rating"] + gamma * blk["T"], a, ys)
            nll += ordinal_nll_vec(thr, x_int, blk["k"])
            pen += lam * float(np.sum(params[st + 1:st + n] ** 2))
            # Gauge fix. "egf": ridge EGF's whole coupling to identity (EGF special).
            # "all": ridge every coupling (symmetric but SHRINKS real offsets — degrades).
            # "mean": pin only the AVERAGE coupling location to identity (symmetric,
            #         EGF non-special, leaves individual offsets free — no shrinkage).
            # "wmean": same but RELIABILITY-WEIGHTED by each server's declaration count,
            #          so the data-rich servers set the scale (lands ≈ EGF, no rank jump).
            if anchor_mode == "all" or (anchor_mode == "egf" and server == "EGF"):
                pen += egf_anchor * float(np.sum((thr - blk["identity"]) ** 2))
            off = float(np.mean(thr - blk["identity"]))
            off_sum += off
            w = float(len(blk["k"]))          # reliability weight = n declarations
            w_sum += w * off
            w_tot += w
        if anchor_mode == "mean":
            pen += egf_anchor * off_sum ** 2          # pin the gauge (avg offset → 0)
        if anchor_mode == "wmean":
            pen += egf_anchor * (w_sum / w_tot) ** 2  # pin the n-weighted avg offset → 0
        if nseg:
            pen += lam_shape * float(np.sum(params[2:2 + nseg] ** 2))
        pen += lam_time * float(gamma ** 2)
        return nll + pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 2000})
    a, gamma = unpack_curve(res.x)
    fits = {}
    for server, st, n in srv_layout:
        blk = blocks[server]
        thr = _unpack_thresholds(res.x[st:st + n], blk["baseline"])
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        fits[server] = {gs[i]: float(thr[i]) for i in range(len(gs))}
    return a, gamma, ys, fits


def forward_rmse(a, gamma, ys, fits, decls, time_form="linear", prod_time=False,
                 egf_is_coupling=True):
    """Pooled forward error: predict each declarer's SERVER rank from their rating
    (rating -> latent -> server coupling) and compare to declared, all in server rank
    units (comparable across servers). Returns (rmse, n)."""
    res = []
    for d in decls:
        server, g = d[0], d[3]
        R = _decl_rating(d, prod_time)
        T = float(_time_feature(d[4], time_form))
        x_int = float(interp_extrap_vec(np.array([R + gamma * T]), a, ys)[0])
        if server == "EGF" and not egf_is_coupling:
            pred = x_int
        elif server in fits and len(fits[server]) >= 2:
            gs = sorted(fits[server])
            our = np.array([fits[server][k] for k in gs], float)
            srv = np.array([k - 0.5 for k in gs], float)
            o = np.argsort(our)
            pred = float(interp_extrap_vec(np.array([x_int]), our[o], srv[o])[0])
        else:
            pred = x_int
        if np.isfinite(pred):
            res.append(pred - g)
    res = np.array(res, float)
    return float(np.sqrt(np.mean(res ** 2))), len(res)


def cv_forward_compare(decls, base, prod_time, egf_anchor=30.0, folds=5, seeds=3):
    """Grouped-by-user held-out forward server-RMSE + held-out EGF bias for the
    EGF-identity model vs the consensus model — the honest "no accuracy cost" check.
    Held-out declarations are forward-predicted from a fit on the other users."""
    keyed = defaultdict(list)
    for d in decls:
        keyed[(round(d[1], 6), round(d[4], 6))].append(d)
    keys = list(keyed)
    out = {"identity": {"se": [], "egf": []}, "consensus": {"se": [], "egf": []}}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        fold_of = {keys[order[i]]: i % folds for i in range(len(keys))}
        for f in range(folds):
            train = [d for k in keys if fold_of[k] != f for d in keyed[k]]
            test = [d for k in keys if fold_of[k] == f for d in keyed[k]]
            if not test or not any(d[0] == "EGF" for d in train):
                continue
            for name, is_coup in (("identity", False), ("consensus", True)):
                try:
                    if is_coup:
                        a, gm, ys, fits = fit_consensus(
                            train, prod_time=prod_time, time_on=False,
                            egf_anchor=egf_anchor, **base)
                    else:
                        a, gm, ys, _, fits = fit(train, prod_time=prod_time,
                                                 time_on=False, **base)
                except SystemExit:
                    continue
                for d in test:
                    R = _decl_rating(d, prod_time)
                    x = float(interp_extrap_vec(np.array([R]), a, ys)[0])
                    if d[0] == "EGF" and not is_coup:
                        pred = x
                    elif d[0] in fits and len(fits[d[0]]) >= 2:
                        gs = sorted(fits[d[0]])
                        our = np.array([fits[d[0]][k] for k in gs], float)
                        srv = np.array([k - 0.5 for k in gs], float)
                        o = np.argsort(our)
                        pred = float(interp_extrap_vec(np.array([x]), our[o], srv[o])[0])
                    else:
                        pred = x
                    if not np.isfinite(pred):
                        continue
                    out[name]["se"].append((pred - d[3]) ** 2)
                    if d[0] == "EGF":
                        out[name]["egf"].append(pred - d[3])
    res = {}
    for name in out:
        se, egf = np.array(out[name]["se"]), np.array(out[name]["egf"])
        res[name] = (float(np.sqrt(se.mean())) if se.size else float("nan"),
                     float(egf.mean()) if egf.size else float("nan"))
    return res   # {name: (forward_rmse, egf_bias)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def egf_stats(a, gamma, ys, decls, time_form="linear", prod_time=False):
    """EGF-anchor calibration: predicted internal rank vs declared EGF rank.
    Returns dict with overall n/bias/rmse/r2/within1 and a per-band bias list."""
    egf = [d for d in decls if d[0] == "EGF"]
    R = np.array([_decl_rating(d, prod_time) for d in egf], float)
    g = np.array([d[3] for d in egf], float)
    T = _time_feature([d[4] for d in egf], time_form)
    if R.size < 2:
        return None
    pred = interp_extrap_vec(R + gamma * T, a, ys)
    res = pred - g
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    bands = []
    for lo, hi, label in EGF_BANDS:
        m = (g >= lo) & (g < hi)
        bands.append((label, int(m.sum()),
                      float(res[m].mean()) if m.any() else float("nan")))
    return {"n": int(R.size), "bias": float(res.mean()),
            "rmse": float(np.sqrt(np.mean(res ** 2))), "r2": r2,
            "within1": float(np.mean(np.abs(res) <= 1.0)), "bands": bands}


def server_bias_report(a, gamma, ys, server_fits, decls, time_form="linear",
                       prod_time=False, egf_is_coupling=False):
    """In-sample bias PER ranking system, in that system's own rank units — the
    direct test of "does EGF soak up more of the misfit than the other servers?".

    In the EGF-identity model EGF has no coupling, so its bias is whatever the shared
    curve can't absorb — while every server's free thresholds soak up their own offset,
    so a small server bias + a large EGF bias is the asymmetry.  In the CONSENSUS model
    (egf_is_coupling=True) EGF is forward-mapped through its own coupling like the rest,
    so it should join them near zero.  Returns [(server, n, bias_mean, bias_abs_mean)]."""
    out = []
    for server in ["EGF"] + sorted(s for s in server_fits if s != "EGF"):
        rows = [d for d in decls if d[0] == server]
        if not rows:
            continue
        R = np.array([_decl_rating(d, prod_time) for d in rows], float)
        g = np.array([d[3] for d in rows], float)
        T = _time_feature([d[4] for d in rows], time_form)
        x_int = interp_extrap_vec(R + gamma * T, a, ys)
        if server == "EGF" and not egf_is_coupling:
            pred = x_int                                   # identity hub
        else:
            thr = server_fits[server]
            gs = sorted(thr)
            our = np.array([thr[k] for k in gs], float)    # internal at each boundary
            srv = np.array([k - 0.5 for k in gs], float)   # server scalar there
            o = np.argsort(our)
            pred = interp_extrap_vec(x_int, our[o], srv[o])
        res = pred - g
        out.append((server, len(rows), float(res.mean()), float(np.mean(np.abs(res)))))
    return out


def cv_egf_bias(decls, fit_kwargs, folds=5, seeds=5):
    """Grouped-by-user 5-fold CV of the held-out EGF-anchor bias/rmse.

    User identity isn't in `decls`, so we group by the (R_raw, T) pair — unique per
    survey user — which is the same grouping fit_user_rating produces."""
    prod_time = fit_kwargs.get("prod_time", False)
    time_form = fit_kwargs.get("time_form", "linear")
    keyed = defaultdict(list)
    for d in decls:
        keyed[(round(d[1], 6), round(d[4], 6))].append(d)
    keys = list(keyed)
    egf_keys = [k for k in keys if any(d[0] == "EGF" for d in keyed[k])]
    biases, rmses, n_bad = [], [], 0
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        fold_of = {keys[order[i]]: i % folds for i in range(len(keys))}
        for f in range(folds):
            train = [d for k in keys if fold_of[k] != f for d in keyed[k]]
            test_egf = [d for k in egf_keys if fold_of[k] == f
                        for d in keyed[k] if d[0] == "EGF"]
            if len(test_egf) < 2 or not any(d[0] == "EGF" for d in train):
                continue
            try:
                a, gamma, ys, _, _ = fit(train, **fit_kwargs)
            except SystemExit:
                continue
            R = np.array([_decl_rating(d, prod_time) for d in test_egf], float)
            g = np.array([d[3] for d in test_egf], float)
            T = _time_feature([d[4] for d in test_egf], time_form)
            res = interp_extrap_vec(R + gamma * T, a, ys) - g
            # A degenerate tiny-fold curve (β underflows ⇒ adjacent rating knots
            # collapse ⇒ interp divides by zero) yields non-finite residuals; skip
            # those folds (and count them, since a fragile config dropping folds is
            # itself a finding).
            if not np.all(np.isfinite(res)):
                n_bad += 1
                continue
            biases.append(float(res.mean()))
            rmses.append(float(np.sqrt(np.mean(res ** 2))))
    if not biases:
        return None
    return {"bias": float(np.mean(biases)), "rmse": float(np.mean(rmses)),
            "n_folds": len(biases), "n_bad": n_bad}


# ---------------------------------------------------------------------------
def _row(label, st, gamma, cv=None):
    if st is None:
        print(f"{label:<26}  (no EGF data)")
        return
    band = {b[0]: b[2] for b in st["bands"]}
    cvs = ""
    if cv:
        bad = f" ({cv['n_bad']} degenerate)" if cv.get("n_bad") else ""
        cvs = f"  cvBias {cv['bias']:+.2f} cvRMSE {cv['rmse']:.2f}{bad}"
    print(f"{label:<26} gamma {gamma:+.3f} | EGF bias {st['bias']:+.2f} "
          f"rmse {st['rmse']:.2f} R2 {st['r2']:+.2f} ≤1rk {100*st['within1']:.0f}% | "
          f"band 10k+ {band.get('10k+', float('nan')):+.1f} "
          f"5k-1k {band.get('5k-1k', float('nan')):+.1f} "
          f"1d-3d {band.get('1d-3d', float('nan')):+.1f} "
          f"4d+ {band.get('4d+', float('nan')):+.1f}{cvs}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sqlite",
                    default=DB_SQLITE)
    ap.add_argument("--min-attempts", type=int, default=20)
    ap.add_argument("--min-server-samples", type=int, default=3)
    ap.add_argument("--rank-padding", type=int, default=2)
    ap.add_argument("--extend-to", type=int, default=9)
    ap.add_argument("--egf-lo", type=int, default=-20)
    ap.add_argument("--egf-hi", type=int, default=10)
    ap.add_argument("--lambda", type=float, default=5.0, dest="lam")
    ap.add_argument("--lambda-shape", type=float, default=0.3, dest="lam_shape",
                    help="Cubic-shape prior strength. Default 0.3 (the --shape-study "
                         "held-out optimum for the all-server consensus latent; "
                         "production map_rating_to_rank still ships 3.0, which over-"
                         "constrains once the curve isn't EGF-kyu-starved). Keep cubic "
                         "(--curve-shape): convexity edges linear at matched strength.")
    ap.add_argument("--lambda-time", type=float, default=0.1, dest="lam_time",
                    help="Ridge on gamma (keeps the learned time coef bounded; the "
                         "term is identifiable only through fast/slow-at-fixed-rating "
                         "covariance, so a light prior avoids tail blowup).")
    ap.add_argument("--time", choices=["off", "all", "server"], default="all",
                    help="Where the learned time term enters: off (no gamma), all "
                         "(EGF anchor + servers), server (non-EGF declarations only).")
    ap.add_argument("--time-form", choices=["linear", "onesided"], default="linear",
                    help="linear = symmetric dock (fixes spread, level re-absorbed); "
                         "onesided = boost the faster-than-median only (the form that "
                         "moves the level — see the docs).")
    ap.add_argument("--egf-weight", type=float, default=1.0,
                    help="Up-weight the EGF likelihood (soft EGF-priority lever).")
    ap.add_argument("--curve-egf-only", action="store_true",
                    help="Hard EGF-priority: fit the curve+gamma on EGF alone, freeze "
                         "it, then fit couplings — servers cannot move the EGF mapping.")
    ap.add_argument("--raw-rating", dest="prod_time", action="store_false",
                    help="Map/evaluate on RAW ratings. Default uses the SHIPPED "
                         "production time-adjusted rating R_prod (assigned-vs-declared "
                         "should compare timed declarations against a timed rating).")
    ap.set_defaults(prod_time=True)
    ap.add_argument("--sweep", action="store_true",
                    help="Run a preset grid of configs and print a comparison table.")
    ap.add_argument("--consensus", action="store_true",
                    help="CONSENSUS model: rating->latent rank->every server INCLUDING "
                         "EGF (each a free coupling). Sweeps the EGF anchor ridge (inf = "
                         "today's EGF-is-identity ... 0 = democratic) and reports per-"
                         "system bias + pooled forward server-RMSE.")
    ap.add_argument("--egf-anchor", type=float, default=30.0,
                    help="Ridge pulling EGF's coupling toward identity in --consensus "
                         "(high = EGF stays the anchor; low = EGF absorbs its offset).")
    ap.add_argument("--curve-shape", default=RATING_SHAPE,
                    help="even_game_winrates.txt shape used as the curve's prior MEAN "
                         "(cubic/linear/exp/quadratic; prefix-matched).")
    ap.add_argument("--shape-study", action="store_true",
                    help="In the CONSENSUS model (now anchored by ALL servers), compare "
                         "curve shapes x prior strength on held-out forward RMSE — is the "
                         "cubic prior still earning its keep, or does a gentle/linear one "
                         "fit better now the curve isn't EGF-kyu-starved?")
    ap.add_argument("--time-form-study", action="store_true",
                    help="Discover the FORM the time adjustment wants: fit "
                         "resid_rank = a0 + (a1 + a2*rating)*boxcox(median_min; d) and "
                         "user-cluster-bootstrap each parameter's distribution.")
    ap.add_argument("--boot", type=int, default=300,
                    help="Bootstrap runs for --time-form-study (default 300).")
    ap.add_argument("--cv", action="store_true",
                    help="Add grouped-by-user 5-fold held-out EGF bias to the report.")
    ap.add_argument("--cv-seeds", type=int, default=3,
                    help="Repeats of the 5-fold split for the CV estimate (default 3).")
    args = ap.parse_args()
    # Degenerate tiny-fold curves (β underflow) divide-by-zero in interp_extrap_vec;
    # cv_egf_bias catches the resulting non-finite folds explicitly, so silence the
    # numpy noise here.
    np.seterr(divide="ignore", invalid="ignore")

    decls, meta = build_decls(args)
    # --time server: learn gamma from non-EGF declarations only by zeroing the EGF
    # time feature (the EGF anchor stays untimed, matching the production philosophy
    # where the onlymove header rank ignores time).
    if args.time == "server":
        decls = [(s, Rr, Rp, g, (0.0 if s == "EGF" else T))
                 for (s, Rr, Rp, g, T) in decls]
    n_egf = sum(1 for d in decls if d[0] == "EGF")
    print(f"{len(decls)} declarations ({n_egf} EGF), {meta['n_timed']} users with a "
          f"time feature; mean log10(median min) = {meta['mean_logT']:+.3f} "
          f"(median {10**meta['mean_logT']*60:.0f}s)")
    print(f"production time dock: alpha={meta['alpha']:+.3f} rating/log10-min, "
          f"g0={meta['g0']:+.3f} g1={meta['g1']:+.5f}")
    print(f"shape='{RATING_SHAPE}' prior λ_shape={args.lam_shape}, server λ={args.lam}, "
          f"λ_time={args.lam_time} | rating = "
          f"{'PRODUCTION time-adjusted (R_prod)' if args.prod_time else 'RAW (R_raw)'}\n")

    base = dict(egf_lo=args.egf_lo, egf_hi=args.egf_hi, lam=args.lam,
                lam_shape=args.lam_shape, lam_time=args.lam_time,
                rank_padding=args.rank_padding, extend_to=args.extend_to,
                min_server_samples=args.min_server_samples,
                curve_shape=args.curve_shape)

    if args.time_form_study:
        time_form_study(args, boot=args.boot)
        return

    if args.shape_study:
        # Now the latent rank is anchored by ALL servers (dense across kyu AND dan),
        # not the kyu-sparse EGF — so the cubic prior may no longer be load-bearing.
        # Compare shape x prior-strength on held-out forward RMSE (consensus, anchor 30).
        print("SHAPE STUDY (consensus model, egf_anchor=30) — is the cubic prior still "
              "earning its keep now the curve is anchored by all servers?\n")
        print(f"{'shape':<9}{'λ_shape':>8}{'fwdRMSE(in)':>13}{'fwdRMSE(cv)':>13}"
              f"{'cvEGFbias':>11}")
        configs = [("cubic", 3.0), ("cubic", 1.0), ("cubic", 0.3), ("cubic", 0.0),
                   ("linear", 3.0), ("linear", 1.0), ("linear", 0.3)]
        for shape, lsh in configs:
            b = dict(base, curve_shape=shape, lam_shape=lsh)
            a, gm, ys, fits = fit_consensus(decls, time_on=False, prod_time=args.prod_time,
                                            egf_anchor=args.egf_anchor, **b)
            fin, _ = forward_rmse(a, gm, ys, fits, decls, prod_time=args.prod_time)
            cvr = cv_forward_compare(decls, b, args.prod_time,
                                     egf_anchor=args.egf_anchor, seeds=args.cv_seeds)
            fcv, ecv = cvr["consensus"]
            print(f"{shape:<9}{lsh:>8.2g}{fin:>13.2f}{fcv:>13.2f}{ecv:>+11.2f}")
        print("\n(in = in-sample, cv = grouped-by-user held-out; lower fwdRMSE better)")
        print("(no files written — research harness; production stays "
              "pipeline.map_rating_to_rank)")
        return

    def run(label, prod_time, **over):
        kw = dict(base, prod_time=prod_time, **over)
        a, gamma, ys, _, server_fits = fit(decls, **kw)
        st = egf_stats(a, gamma, ys, decls, time_form=kw.get("time_form", "linear"),
                       prod_time=prod_time)
        cv = cv_egf_bias(decls, kw, seeds=args.cv_seeds) if args.cv else None
        _row(label, st, gamma, cv)
        return a, gamma, ys, server_fits

    if args.consensus:
        # rating -> LATENT consensus rank -> every server INCLUDING EGF (free coupling).
        # Loosening the EGF anchor lets EGF absorb its OWN offset in its coupling instead
        # of dumping it on the shared curve — the test is whether EGF's systematic bias
        # drops toward the servers' ~0 WITHOUT the pooled forward RMSE degrading (i.e.
        # without the kyu collateral the egf-weight lever paid).
        print("CONSENSUS model — rating -> latent rank -> every server incl EGF (each a "
              "free coupling).")
        print("egf_anchor: high = EGF pinned to identity (today's model); low = EGF "
              "absorbs its own offset.\n")
        ckw = dict(base, time_on=False, time_form="linear", prod_time=args.prod_time)
        # Honest baseline: TODAY's EGF-is-identity model (the fit() path, EGF scored
        # directly against the shared curve with no coupling).
        a_b, g_b, ys_b, _, sf_b = fit(decls, prod_time=args.prod_time, time_on=False, **base)
        frmse_b, _ = forward_rmse(a_b, g_b, ys_b, sf_b, decls, prod_time=args.prod_time,
                                  egf_is_coupling=False)
        rep_b = server_bias_report(a_b, g_b, ys_b, sf_b, decls, prod_time=args.prod_time)
        egf_b = next(r for r in rep_b if r[0] == "EGF")
        print(f"{'egf_anchor':>11}{'fwdRMSE':>9}{'EGFbias':>9}{'EGF|b|':>8}"
              f"{'meanSrv|b|':>11}")
        print(f"{'identity':>11}{frmse_b:>9.2f}{egf_b[2]:>+9.2f}{egf_b[3]:>8.2f}"
              f"{float(np.mean([ab for s,n,b,ab in rep_b if s!='EGF'])):>11.2f}"
              "  (today's model: EGF = identity hub)")
        for anch in (100.0, 30.0, 10.0, 3.0, 1.0, 0.3):
            a, gamma, ys, fits = fit_consensus(decls, egf_anchor=anch, **ckw)
            frmse, _ = forward_rmse(a, gamma, ys, fits, decls, prod_time=args.prod_time)
            rep = server_bias_report(a, gamma, ys, fits, decls, prod_time=args.prod_time,
                                     egf_is_coupling=True)
            egf = next(r for r in rep if r[0] == "EGF")
            mean_srv_ab = float(np.mean([ab for s, n, b, ab in rep if s != "EGF"]))
            print(f"{anch:>11.4g}{frmse:>9.2f}{egf[2]:>+9.2f}{egf[3]:>8.2f}"
                  f"{mean_srv_ab:>11.2f}")
        a, gamma, ys, fits = fit_consensus(decls, egf_anchor=args.egf_anchor, **ckw)
        print(f"\nPer-system bias at egf_anchor={args.egf_anchor:g} (EGF forward-mapped "
              "through its own coupling, like the rest):")
        print(f"{'system':<8}{'n':>5}{'bias':>9}{'|bias|':>9}")
        for srv, n, b, ab in server_bias_report(a, gamma, ys, fits, decls,
                                                prod_time=args.prod_time,
                                                egf_is_coupling=True):
            print(f"{srv:<8}{n:>5}{b:>+9.2f}{ab:>9.2f}")
        if args.cv:
            print(f"\nHeld-out forward server-RMSE (grouped-by-user CV, {args.cv_seeds} "
                  f"seeds; the honest no-cost check, egf_anchor={args.egf_anchor:g}):")
            cvres = cv_forward_compare(decls, base, args.prod_time,
                                       egf_anchor=args.egf_anchor, seeds=args.cv_seeds)
            for name in ("identity", "consensus"):
                fr, eb = cvres[name]
                print(f"  {name:<10} fwdRMSE {fr:.2f}  held-out EGF bias {eb:+.2f}")
    elif args.sweep:
        print("config                       learned   ── in-sample EGF anchor ──        "
              "── per-band EGF bias ──" + ("   ── held-out CV ──" if args.cv else ""))
        print("# rating = RAW (reference)")
        run("raw baseline (W=1)", prod_time=False, time_on=False, egf_weight=1.0)
        print("# rating = PRODUCTION time-adjusted (R_prod) — the assigned-vs-declared view")
        a0, g0_, ys0, sf0 = run("prod-time baseline (W=1)", prod_time=True, time_on=False)
        for W in (3.0, 10.0, 30.0):
            run(f"prod-time egf-weight {W:g}", prod_time=True, time_on=False, egf_weight=W)
        run("prod-time curve-egf-only", prod_time=True, time_on=False, curve_egf_only=True)
        print("# rating = RAW + LEARNED gamma (goal 1: time learned IN the mapping)")
        run("learned linear (W=1)", prod_time=False, time_on=True, egf_weight=1.0)
        run("learned onesided (W=1)", prod_time=False, time_on=True, time_form="onesided")
        run("learned linear + egf-weight10", prod_time=False, time_on=True, egf_weight=10.0)

        # The user's claim: EGF (identity hub, no offset of its own) soaks up more of
        # the cross-server misfit than the servers, whose free thresholds absorb it.
        print("\nPer-system in-sample bias (prod-time baseline, that system's own rank "
              "units) — does EGF carry more than the servers?")
        print(f"{'system':<8}{'n':>5}{'bias':>9}{'|bias|':>9}")
        for srv, n, b, ab in server_bias_report(a0, g0_, ys0, sf0, decls,
                                                prod_time=True):
            print(f"{srv:<8}{n:>5}{b:>+9.2f}{ab:>9.2f}")
    else:
        time_on = args.time != "off"
        lbl = (f"time={args.time}/{args.time_form}, egf-weight={args.egf_weight:g}"
               f"{', curve-egf-only' if args.curve_egf_only else ''}"
               f", rating={'prod' if args.prod_time else 'raw'}")
        a, gamma, ys, sf = run(lbl, prod_time=args.prod_time, time_on=time_on,
                               egf_weight=args.egf_weight,
                               curve_egf_only=args.curve_egf_only,
                               time_form=args.time_form)
        print(f"\n{'system':<8}{'n':>5}{'bias':>9}{'|bias|':>9}")
        for srv, n, b, ab in server_bias_report(a, gamma, ys, sf, decls,
                                                time_form=args.time_form,
                                                prod_time=args.prod_time):
            print(f"{srv:<8}{n:>5}{b:>+9.2f}{ab:>9.2f}")

    print("\n(no files written — research harness; production stays pipeline.map_rating_to_rank)")


if __name__ == "__main__":
    main()
