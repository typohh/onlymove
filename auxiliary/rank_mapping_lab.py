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
    SQL_PATH_DEFAULT)
from auxiliary.fit_statistics_shapes import fit_shape  # noqa: E402
from utils.fit_rank_curves import (  # noqa: E402
    fit_user_rating, build_server_curves,
    ordinal_nll_vec, interp_extrap_vec, _unpack_thresholds)
from pipeline.map_rating_to_egf import load_v2_ratings  # noqa: E402

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
def load_data(sql=SQL_PATH_DEFAULT, use_cache=True):
    if use_cache and os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    surveys = latest_per_user_server(load_rank_surveys(sql))
    users = list(surveys.user_id.unique())
    att = load_attempts_for_users(sql, set(users))
    v2m = {}
    for s in SIZES:
        v2m.update(load_v2_ratings(os.path.join(PROJECT_ROOT, "ratings_v2"), s))
    Rv2 = {}
    for uid in users:
        p = uid[:8]
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
    data = dict(recs=dict(recs), Rv2=Rv2)
    pickle.dump(data, open(CACHE, "wb"))
    return data


# ---------------------------------------------------------------------------
# shared rating→EGF curve (+ couplings), pluggable rating-curve parameterization
# ---------------------------------------------------------------------------
def fit_shared(decls, method, shape=None, rating_lam=1.0, lam=10.0,
               rank_padding=2, extend_to=9, min_server_samples=3,
               coupling2018=None, prior2018_lam=0.1):
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

    # server blocks. mode 'affine' (2-param affine on the 2018 SHAPE: thresholds
    # = α + e^logβ·τ⁰, absorbing per-server temporal drift) when a 2018 curve is
    # available; else 'free' (uniform-spacing-prior thresholds). τ⁰ is the 2018
    # EGF value at each of this server's rank boundaries.
    coupling2018 = coupling2018 or {}
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
        blk = dict(
            g_min=g_min, g_max=g_max,
            rating=np.array([r for r, _ in rows], float),
            k=np.array([max(0, min(N - 1, g - g_min)) for _, g in rows], int))
        bounds = np.array([g - 0.5 for g in range(g_min + 1, g_max + 1)], float)
        if server in coupling2018:
            c = coupling2018[server]
            blk["mode"] = "affine"
            blk["tau0"] = interp_extrap_vec(bounds, np.array(c["server"], float),
                                            np.array(c["our"], float))
            blk["nparam"] = 2
        else:
            blk["mode"] = "free"
            blk["baseline"] = np.ones(K - 1) if K > 1 else np.array([])
            blk["nparam"] = len(blk["baseline"]) + 1
        blocks[server] = blk

    layout = [("rating", None, 0, n_rating)]
    x0 = list(x0_rating)
    idx = n_rating
    for server, blk in blocks.items():
        layout.append(("srv", server, idx, blk["nparam"]))
        if blk["mode"] == "affine":
            x0.extend([0.0, 0.0])                      # α=0, logβ=0 ⇒ 2018 shape
        else:
            x0.append(float(blk["g_min"])); x0.extend([0.0] * (blk["nparam"] - 1))
        idx += blk["nparam"]
    x0 = np.asarray(x0, float)

    def srv_thr(server, pr):
        blk = blocks[server]
        if blk["mode"] == "affine":
            return pr[0] + np.exp(pr[1]) * blk["tau0"]
        return _unpack_thresholds(pr, blk["baseline"])

    def unpack(p):
        a = None; b = {}; logs = {}
        for kind, key, st, n in layout:
            if kind == "rating":
                a = rating_thr(p[st:st + n]); logs["rating"] = p[st:st + n]
            else:
                b[key] = srv_thr(key, p[st:st + n]); logs[("srv", key)] = p[st:st + n]
        return a, b, logs

    def obj(p):
        a, b, logs = unpack(p)
        nll = ordinal_nll_vec(a, egf_R, egf_k)
        pen = rating_pen(logs["rating"])
        for server, blk in blocks.items():
            x_int = interp_extrap_vec(blk["rating"], a, ys)
            nll += ordinal_nll_vec(b[server], x_int, blk["k"])
            lg = logs[("srv", server)]
            if blk["mode"] == "affine":   # mild prior toward (α,logβ)=(0,0): trust 2018
                pen += prior2018_lam * float(lg[0] ** 2 + lg[1] ** 2)
            else:
                pen += lam * float(np.sum(lg[1:] ** 2))
        return nll + pen

    res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": 10000})
    a, b, _ = unpack(res.x)
    server_fits = {}
    for server, blk in blocks.items():
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        server_fits[server] = ({gs[i]: float(b[server][i]) for i in range(len(gs))}, 0, 0)
    return (a, ys), build_server_curves(server_fits)


def srv2egf(server, g, rmap):
    if server == "EGF":
        return g
    k = rmap.get(server)
    return None if not k else float(interp_extrap_vec(np.array([g]), np.array(k["server"]), np.array(k["our"]))[0])


def egf2srv(server, egf, rmap):
    """Forward: internal/EGF rank -> declared server rank (the non-circular eval
    direction). Identity for EGF."""
    if server == "EGF":
        return egf
    k = rmap.get(server)
    return None if not k else float(interp_extrap_vec(np.array([egf]), np.array(k["our"]), np.array(k["server"]))[0])


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


def run(methods, folds=5, boot=3000, seed=0, rating_lam=1.0, server_lam=10.0,
        coupling_prior=None):
    rng = np.random.default_rng(seed)
    d = load_data()
    recs, Rv2 = d["recs"], d["Rv2"]
    shapes = {m: (_shape_fn(m) if m not in ("free",) else None)
              for m in methods}
    c2018 = None
    if coupling_prior in ("2018", "2024"):
        from auxiliary.fit_2018_couplings import fit_2018_couplings, fit_2024_couplings
        c2018 = fit_2018_couplings() if coupling_prior == "2018" else fit_2024_couplings()
        print(f"{coupling_prior} coupling prior on: {[s for s in c2018 if s != 'EGF']}")

    prefs = list(recs.keys()); rng.shuffle(prefs)
    fold = {p: i % folds for i, p in enumerate(prefs)}

    # OOF rows per method: (pref, server, declared_g, pred_egf, target_egf)
    oof = {m: [] for m in methods}
    for i in range(folds):
        train = [p for p in prefs if fold[p] != i]
        test = [p for p in prefs if fold[p] == i]
        # build decls for the shared (v2) rating→EGF methods
        d_shared = [(srv, Rv2[p], g) for p in train if p in Rv2 for srv, g in recs[p]]
        fitted = {}
        for m in methods:
            fitted[m] = fit_shared(d_shared, m, shapes[m],
                                   rating_lam=rating_lam,
                                   lam=server_lam,
                                   coupling2018=c2018)
        for p in test:
            for m in methods:
                curve, rmap = fitted[m]
                pe = (float(interp_extrap_vec(np.array([Rv2[p]]), *curve)[0])
                      if p in Rv2 else None)
                for srv, g in recs[p]:
                    te = srv2egf(srv, g, rmap)
                    psr = egf2srv(srv, pe, rmap) if pe is not None else None
                    oof[m].append((p, srv, g, pe, te, psr))
    frames = {m: pd.DataFrame(oof[m], columns=["p", "srv", "g", "pe", "te", "psr"])
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

    # NON-CIRCULAR forward metric: predict the user's declared SERVER rank from R
    # (R -> curve -> EGF -> coupling -> server rank), vs observed. The coupling is
    # used FORWARD against an observed target, so coupling variants are comparable
    # (unlike 'te', which is coupling-defined). Per-server macro de-weights big servers.
    print("\n--- forward metric: predicted declared SERVER rank vs observed ---")
    for m in methods:
        f = frames[m].dropna(subset=["psr"])
        ns = f[f.srv != "EGF"]
        per = ns.groupby("srv").apply(
            lambda x: np.sqrt(np.mean((x.psr - x.g) ** 2)), include_groups=False)
        overall = np.sqrt(np.mean((ns.psr - ns.g) ** 2))
        print(f"  {m:11s} server-RMSE overall={overall:.2f}  "
              f"macro(per-server)={per.mean():.2f}  worst={per.max():.2f} "
              f"({per.idxmax()})")

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
                    default=["free", "linear", "cubic", "exp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rating-lam", type=float, default=1.0,
                    help="Ridge strength for free/_prior curve deviations.")
    ap.add_argument("--server-lam", type=float, default=10.0,
                    help="Uniform-spacing prior strength for the server couplings.")
    ap.add_argument("--coupling-prior", choices=["2018", "2024"], default=None,
                    help="Use a cross-server survey as an affine (offset+scale) shape "
                         "prior on the couplings (servers it covers).")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if args.no_cache and os.path.exists(CACHE):
        os.remove(CACHE)
    run(args.methods, args.folds, args.boot, args.seed, args.rating_lam,
        args.server_lam, args.coupling_prior)
