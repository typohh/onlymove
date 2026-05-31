"""Rank-mapping experiment harness (see docs/rank_mapping_experiments.md).

Compares ways of generating the survey-fit rating→EGF curve (+ server couplings)
by OUT-OF-SAMPLE error: grouped 5-fold CV by user, user-level bootstrap CIs, and
a per-rank-band breakdown. A user's 80%-clear rating is computed from attempts
(fold-independent); only the curve+coupling fit sees the train split.

Curve methods for the SHARED v2-style rating→EGF curve (`--methods`):
  free      monotone piecewise-linear thresholds, uniform-spacing prior (= v2-new)
  linear    R = α + β·x            (straight line in rank; 2 params)
  cubic     R = α + β·θ_stats(x)   (EGF statistics.txt cubic shape; 2 params)   [v3]
  exp       R = α + β·θ_exp(x)     (EGF statistics.txt free-exp shape; 2 params)
  quadratic R = α + β·θ_quad(x)    (EGF statistics.txt quadratic shape; 2 params)
The non-`free` methods fix the SHAPE from EGF data (fit_statistics_shapes) and fit
only the affine (α, β) to surveys — principled, low-variance. Server couplings are
the same ordinal-threshold fit (uniform prior) in every method.

`v1` is also evaluated for reference (pipeline.fit_rank_curves, per-board).

Usage:
  ./.venv/bin/python3 -m auxiliary.rank_mapping_lab --methods free linear cubic exp
  ./.venv/bin/python3 -m auxiliary.rank_mapping_lab --methods cubic --folds 5 --boot 3000
"""
import argparse, json, os, pickle, sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import (  # noqa: E402
    load_rank_surveys, latest_per_user_server, load_attempts_for_users,
    load_problem_params, SQL_PATH_DEFAULT)
from auxiliary.fit_statistics_shapes import fit_shape  # noqa: E402
from pipeline.fit_rank_curves import (  # noqa: E402
    fit_user_rating, load_raw_ratings, fit_joint, build_server_curves,
    _cubic_curve, ordinal_nll_vec, interp_extrap_vec, _unpack_thresholds)
from pipeline.map_rating_v2_to_egf import load_v2_ratings  # noqa: E402

SIZES = [9, 13]
EGF_LO, EGF_HI = -20, 10
CACHE = "/tmp/rank_lab_cache.pkl"

# stats-shape rank_index axis: 20k=0 … 1k=19 … 1d=20.  EGF scalar g: 1k=0, 1d=1.
# So rank_index(g) = g + 19.
def _shape_fn(name):
    """Return S(g_array) -> shape value at EGF scalar g, from statistics.txt.
    `name` may carry a `_prior` suffix (hybrid: shape used as ridge prior)."""
    base = name[:-6] if name.endswith("_prior") else name
    key = {"linear": "linear", "cubic": "cubic", "exp": "exp  A",
           "quadratic": "quadratic"}.get(base)
    if key is None:
        raise ValueError(name)
    params, theta = fit_shape(key)
    return lambda g: theta(np.asarray(g, float) + 19.0, params)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _cubic_by_size():
    out = {}
    for s in SIZES:
        *_, stop = load_problem_params(s)
        c = _cubic_curve(stop)
        if c is not None:
            out[s] = c
    return out


def load_data(sql=SQL_PATH_DEFAULT, use_cache=True):
    if use_cache and os.path.exists(CACHE):
        data = pickle.load(open(CACHE, "rb"))
        data["cubic_by_size"] = _cubic_by_size()   # not picklable; recompute (cheap)
        return data
    surveys = latest_per_user_server(load_rank_surveys(sql))
    users = list(surveys.user_id.unique())
    att = load_attempts_for_users(sql, set(users))
    v1r = {s: load_raw_ratings(s) for s in SIZES}
    v2m = {}
    for s in SIZES:
        v2m.update(load_v2_ratings(os.path.join(PROJECT_ROOT, "ratings_v2"), s))
    Rv1, Rv2 = {}, {}
    for uid in users:
        p = uid[:8]
        for s in SIZES:
            a = att.get((p, s), [])
            if len(a) >= 20 and v1r[s]:
                R, n = fit_user_rating(a, v1r[s], min_n=20)
                if R is not None:
                    Rv1[(p, s)] = R
        pool = []
        for s in SIZES:
            pool += att.get((p, s), [])
        if len(pool) >= 20:
            R, n = fit_user_rating(pool, v2m, min_n=20)
            if R is not None:
                Rv2[p] = R
    recs = defaultdict(list)
    for _, r in surveys.iterrows():
        g = r.rank_scalar
        if g is None or pd.isna(g):
            continue
        recs[str(r.user_id)[:8]].append((r.server, float(g)))
    data = dict(recs=dict(recs), Rv1=Rv1, Rv2=Rv2)
    pickle.dump(data, open(CACHE, "wb"))
    data["cubic_by_size"] = _cubic_by_size()
    return data


# ---------------------------------------------------------------------------
# shared rating→EGF curve (+ couplings), pluggable rating-curve parameterization
# ---------------------------------------------------------------------------
def fit_shared(decls, method, shape=None, rating_lam=1.0, lam=10.0,
               rank_padding=2, extend_to=9, min_server_samples=3):
    """decls: [(server, R, g)]. Returns (curve, server_curves) where
    curve = (rating_thresholds, ys) usable as interp_extrap_vec(R, *curve)."""
    N_egf = EGF_HI - EGF_LO + 1
    egf_thr_ranks = list(range(EGF_LO + 1, EGF_HI + 1))
    ys = np.array([g - 0.5 for g in egf_thr_ranks], float)

    egf_R, egf_k = [], []
    srv_rows = defaultdict(list)
    for server, R, g in decls:
        gi = int(round(g))
        if server == "EGF":
            egf_R.append(R); egf_k.append(max(0, min(N_egf - 1, gi - EGF_LO)))
        else:
            srv_rows[server].append((R, gi))
    egf_R = np.asarray(egf_R, float); egf_k = np.asarray(egf_k, int)
    if len(egf_R) == 0:
        raise SystemExit("no EGF declarations to anchor the scale")

    # linear pre-fit for init / uniform baseline
    beta0, alpha0 = np.polyfit(egf_R, egf_k.astype(float) + EGF_LO, 1)
    if beta0 <= 0:
        beta0 = 0.1
    prior_rating = (ys - alpha0) / beta0
    rating_baseline = np.diff(prior_rating)

    # rating-curve parameterization
    if method == "free" or method.endswith("_prior"):
        # free monotone thresholds; baseline spacings = the SHAPE prior's spacings
        # (uniform/linear for `free`, statistics.txt shape for `<shape>_prior`),
        # ridge-penalized toward it. rating_lam->inf => pure shape; ->0 => free.
        if method.endswith("_prior"):
            S = shape(ys)
            b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
            base_rating = prior_rating[0] + b0 * (S - S[0])     # shape pinned affinely
            baseline = np.diff(base_rating)
            init0 = float(base_rating[0])
        else:
            baseline = rating_baseline
            init0 = float(prior_rating[0])
        n_rating = len(baseline) + 1
        x0_rating = [init0] + [0.0] * len(baseline)
        def rating_thr(pr):
            return _unpack_thresholds(np.asarray(pr), baseline)
        def rating_pen(pr):
            return rating_lam * float(np.sum(np.asarray(pr[1:]) ** 2))
    else:                                   # pure shape: thr = a + b*S(ys), b>0
        S = shape(ys)
        b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
        a0 = float(prior_rating[0] - b0 * S[0])
        n_rating = 2
        x0_rating = [a0, float(np.log(max(b0, 1e-3)))]
        def rating_thr(pr):
            return pr[0] + np.exp(pr[1]) * S
        def rating_pen(pr):
            return 0.0                      # shape is fixed; nothing to penalize

    # server blocks
    blocks = {}
    for server, rows in srv_rows.items():
        if len(rows) < min_server_samples:
            continue
        decl = [g for _, g in rows]
        g_min = min(decl) - rank_padding
        g_max = max(max(decl) + rank_padding, extend_to)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        blocks[server] = dict(
            g_min=g_min, g_max=g_max,
            baseline=np.ones(K - 1) if K > 1 else np.array([]),
            rating=np.array([r for r, _ in rows], float),
            k=np.array([max(0, min(N - 1, g - g_min)) for _, g in rows], int))

    layout = [("rating", None, 0, n_rating)]
    x0 = list(x0_rating)
    idx = n_rating
    for server, blk in blocks.items():
        n = len(blk["baseline"]) + 1
        layout.append(("srv", server, idx, n))
        x0.append(float(blk["g_min"])); x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, float)

    def unpack(p):
        a = None; b = {}; logs = {}
        for kind, key, st, n in layout:
            if kind == "rating":
                a = rating_thr(p[st:st + n]); logs["rating"] = p[st:st + n]
            else:
                b[key] = _unpack_thresholds(p[st:st + n], blocks[key]["baseline"])
                logs[("srv", key)] = p[st + 1:st + n]
        return a, b, logs

    def obj(p):
        a, b, logs = unpack(p)
        nll = ordinal_nll_vec(a, egf_R, egf_k)
        for server, blk in blocks.items():
            x_int = interp_extrap_vec(blk["rating"], a, ys)
            nll += ordinal_nll_vec(b[server], x_int, blk["k"])
        pen = rating_pen(logs["rating"])
        for server in blocks:
            pen += lam * float(np.sum(logs[("srv", server)] ** 2))
        return nll + pen

    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 10000})
    a, b, _ = unpack(res.x)
    server_fits = {}
    for server, blk in blocks.items():
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        server_fits[server] = ({gs[i]: float(b[server][i]) for i in range(len(gs))}, 0, 0)
    return (a, ys), build_server_curves(server_fits)


def fit_v1(decls_v1, cubic_by_size):
    rr, sf = fit_joint(decls_v1, cubic_by_size, SIZES, EGF_LO, EGF_HI,
                       rating_lam=1.0, lam=10.0, rank_padding=2, extend_to=9,
                       min_server_samples=3)
    curves = {}
    for s, thr in rr.items():
        gs = sorted(thr)
        curves[s] = (np.array([thr[g] for g in gs]),
                     np.array([g - 0.5 for g in gs], float))
    return curves, build_server_curves(sf)


def srv2egf(server, g, rmap):
    if server == "EGF":
        return g
    k = rmap.get(server)
    return None if not k else float(interp_extrap_vec(np.array([g]), np.array(k["server"]), np.array(k["our"]))[0])


# ---------------------------------------------------------------------------
# CV + metrics
# ---------------------------------------------------------------------------
def metrics(pred, targ):
    pred = np.asarray(pred, float); targ = np.asarray(targ, float)
    m = np.isfinite(pred) & np.isfinite(targ)
    pred, targ = pred[m], targ[m]
    res = pred - targ
    sst = np.sum((targ - targ.mean()) ** 2)
    return dict(n=len(pred), R2=1 - np.sum(res ** 2) / sst,
                RMSE=float(np.sqrt(np.mean(res ** 2))),
                MAE=float(np.mean(np.abs(res))), bias=float(res.mean()))


def run(methods, folds=5, boot=3000, seed=0, rating_lam=1.0, server_lam=10.0):
    rng = np.random.default_rng(seed)
    d = load_data()
    recs, Rv1, Rv2, cubic = d["recs"], d["Rv1"], d["Rv2"], d["cubic_by_size"]
    shapes = {m: (_shape_fn(m) if m not in ("free",) else None)
              for m in methods if m != "v1"}

    prefs = list(recs.keys()); rng.shuffle(prefs)
    fold = {p: i % folds for i, p in enumerate(prefs)}

    # OOF rows per method: (pref, server, declared_g, pred_egf, target_egf)
    oof = {m: [] for m in methods}
    for i in range(folds):
        train = [p for p in prefs if fold[p] != i]
        test = [p for p in prefs if fold[p] == i]
        # build decls for shared (v2) methods and for v1
        d_shared = [(srv, Rv2[p], g) for p in train if p in Rv2 for srv, g in recs[p]]
        d_v1 = [(s, srv, Rv1[(p, s)], g) for p in train for srv, g in recs[p]
                for s in SIZES if (p, s) in Rv1]
        fitted = {}
        for m in methods:
            if m == "v1":
                fitted[m] = ("v1",) + fit_v1(d_v1, cubic)
            else:
                fitted[m] = ("shared",) + fit_shared(d_shared, m, shapes[m],
                                                      rating_lam=rating_lam,
                                                      lam=server_lam)
        for p in test:
            for m in methods:
                kind, curve, rmap = fitted[m]
                if kind == "v1":
                    es = [float(interp_extrap_vec(np.array([Rv1[(p, s)]]), *curve[s])[0])
                          for s in SIZES if (p, s) in Rv1 and s in curve]
                    pe = float(np.mean(es)) if es else None
                else:
                    pe = (float(interp_extrap_vec(np.array([Rv2[p]]), *curve)[0])
                          if p in Rv2 else None)
                for srv, g in recs[p]:
                    te = srv2egf(srv, g, rmap)
                    oof[m].append((p, srv, g, pe, te))
    frames = {m: pd.DataFrame(oof[m], columns=["p", "srv", "g", "pe", "te"])
              for m in methods}

    print(f"\n{'='*78}\nOUT-OF-SAMPLE ({folds}-fold CV by user)\n{'='*78}")
    print("\n--- EGF declarers only (clean rating→rank curve) ---")
    for m in methods:
        e = frames[m][frames[m].srv == "EGF"]
        mm = metrics(e.pe, e.g)
        print(f"  {m:10s} n={mm['n']:3d}  R2={mm['R2']:+.3f}  RMSE={mm['RMSE']:.2f}  MAE={mm['MAE']:.2f}  bias={mm['bias']:+.2f}")
    print("\n--- all declarations (own couplings) ---")
    for m in methods:
        mm = metrics(frames[m].pe, frames[m].te)
        print(f"  {m:10s} n={mm['n']:3d}  R2={mm['R2']:+.3f}  RMSE={mm['RMSE']:.2f}  MAE={mm['MAE']:.2f}  bias={mm['bias']:+.2f}")

    bands = [(-20, -10, "~20k-11k"), (-10, -5, "10k-6k"), (-5, -1, "5k-2k"),
             (-1, 1, "1k-1d"), (1, 4, "1d-3d"), (4, 11, "4d+")]
    print(f"\n--- per-rank-band bias (binned by declared EGF target) ---")
    hdr = "  " + f"{'band':9}" + "".join(f"{m:>14}" for m in methods)
    print(hdr)
    band_rmse = {m: [] for m in methods}
    for lo, hi, lab in bands:
        cells = []
        for m in methods:
            s = frames[m][(frames[m].te >= lo) & (frames[m].te < hi)]
            r = (s.pe - s.te).dropna()
            if len(r):
                rmse = float(np.sqrt(np.mean(r ** 2))); band_rmse[m].append(rmse)
                cells.append(f"{rmse:4.1f}/{r.mean():+4.1f}")
            else:
                cells.append("   -  ")
        n = len(frames[methods[0]][(frames[methods[0]].te >= lo) & (frames[methods[0]].te < hi)])
        print(f"  {lab:9}" + "".join(f"{c:>14}" for c in cells) + f"   (n={n})")
    # macro (band-balanced) RMSE + worst-band RMSE: de-weights the over-represented
    # middle so the tails count. The decision-relevant criterion for THIS goal.
    print("\n--- band-balanced error (equal weight per band) ---")
    for m in methods:
        br = np.array(band_rmse[m])
        print(f"  {m:11s} macro-RMSE={br.mean():.2f}  worst-band={br.max():.2f}")

    # bootstrap on all-declaration RMSE, paired vs first method
    print(f"\n--- bootstrap RMSE 95% CI (all decls), paired vs '{methods[0]}' ---")
    base = methods[0]
    bf = frames[base]
    ups = bf.p.unique()
    by = {m: {p: frames[m][frames[m].p == p] for p in ups} for m in methods}
    stat = {m: [] for m in methods}; diff = {m: [] for m in methods if m != base}
    for _ in range(boot):
        pick = rng.choice(ups, len(ups), replace=True)
        for m in methods:
            samp = pd.concat([by[m][p] for p in pick])
            stat[m].append(metrics(samp.pe, samp.te)["RMSE"])
        for m in methods:
            if m != base:
                diff[m].append(stat[m][-1] - stat[base][-1])
    for m in methods:
        lo, hi = np.percentile(stat[m], [2.5, 97.5])
        extra = ""
        if m != base:
            dl, dh = np.percentile(diff[m], [2.5, 97.5])
            extra = f"   Δvs{base} [{dl:+.2f},{dh:+.2f}]  P(worse)={np.mean(np.array(diff[m])>0):.2f}"
        print(f"  {m:10s} RMSE [{lo:.2f},{hi:.2f}]{extra}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+",
                    default=["free", "linear", "cubic", "exp", "v1"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rating-lam", type=float, default=1.0,
                    help="Ridge strength for free/_prior curve deviations.")
    ap.add_argument("--server-lam", type=float, default=10.0,
                    help="Uniform-spacing prior strength for the server couplings.")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if args.no_cache and os.path.exists(CACHE):
        os.remove(CACHE)
    run(args.methods, args.folds, args.boot, args.seed, args.rating_lam,
        args.server_lam)
