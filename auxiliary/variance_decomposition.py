"""Variance-source decomposition of attempt-outcome residuals.

Separates the residual variance around the production 4PL success model into
its structurally distinct sources, using the fact that each source correlates
residuals along a different slice of (user, problem, time):

  1. PROBLEM-RANK ERROR   — user-general, problem-specific, time-stable.
     Measured by a split-half design: users are randomly halved, each problem's
     rating offset is estimated independently in each half, and the cross-half
     covariance of those offsets is the true item-error variance (independent
     noise cancels). Reported in rating units and rank units.
  2. FORM ERROR           — user-general, problem-general, a function of the
     ability-difficulty gap only. Measured as the pooled calibration curve
     binned by (ability − problem rating), overall and by slope tercile.
  3. STABLE INDIVIDUAL SKILL COMPONENTS — user-specific, feature-specific,
     time-stable. Measured as same-user residual co-movement that depends on
     the embedding similarity of the two problems at LONG time gaps.
  4. EBB AND FLOW         — user-specific, feature-general, time-local.
     Measured as same-user residual co-movement that decays with the time gap
     at fixed similarity. Session-position and relative-solve-time covariates
     give observable state proxies.

Residuals are Pearson residuals (observed − predicted)/√(P(1−P)) from the SAME
4PL the client uses (utils.fit_rank_curves.fit_user_rating's curve: per-problem
rating/slope/floor + global lapse, κ-anchored at CALIBRATION_P), evaluated at a
per-attempt LEAVE-ONE-OUT kernel-local ability (Gaussian kernel in days over
the user's other first-time attempts, weak prior toward the career ability).
The local trend removes each user's learning curve, so analyses 3/4 see only
fluctuation FASTER than the trend bandwidth.

EVERY analysis is calibrated against a MODEL-TRUE NULL (--null-sims): outcomes
re-simulated from a constant per-user ability under the fitted 4PL, with the
whole pipeline (career fit, leave-one-out trend, residuals, analyses) re-run on
the simulated outcomes. The null absorbs the pipeline's own artifacts — the
trend fit's leakage (spurious negative same-user covariance), errors-in-
variables flattening of the calibration curve, split-half noise — so the
REAL − NULL excess is the estimate to read.

Same-user co-movements are converted to ability units by a ratio estimator:
E[r_i·r_j] ≈ cov_ability · s_i·s_j with s = (dP/dability)/√(P(1−P)), so
Σ r_i r_j / Σ s_i s_j estimates the implied ability covariance (rating units²)
for that (time-gap × similarity) cell. Standard errors are jackknifed over
users (pairs sharing a user are dependent).

Caveats: adaptive problem selection is ignorable because residuals condition
on the model's own per-attempt prediction; solve time is co-determined with
the outcome (state PROXY, not cause); the item analysis is powered only on the
frequently-attempted head of the corpus; first-time rows only (repeats are
memory, not skill).

Usage:
    ./.venv/bin/python3 -m auxiliary.variance_decomposition [--help]

Reads attempts/database.sqlite + ratings/<size>/ + embeddings/joint/per_problem.csv.
Stdout report only; nothing is written.
"""

import argparse
import csv
import os
import sqlite3
import sys
import time
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES  # noqa: E402
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, FLOOR_MAX, LOGIT_PER_RATING,
)
from utils.ratings_io import (  # noqa: E402
    load_lapse, load_rating_rank_curve, load_size_value_maps,
)

DAY_SECONDS = 86400.0

# Time-gap bin edges in days (labels below); same-user pair analyses.
TIME_EDGES = np.array([0.0, 1 / 48.0, 0.25, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0])
TIME_LABELS = ["<30min", "30m-6h", "6-24h", "1-3d", "3-7d", "7-14d", "14-30d", "30-60d"]
LONG_GAP_LABELS = ("7-14d", "14-30d", "30-60d")

# Ability-difficulty gap bins (rating units) for the form-error calibration.
GAP_EDGES = np.array([-1.5, -1.0, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.0, 1.5])

# Within-session position bins (1-based attempt index inside a session).
SESSION_POS_EDGES = np.array([1, 2, 6, 16, 41])  # → 1, 2-5, 6-15, 16-40, 41+
SESSION_POS_LABELS = ["1st", "2-5", "6-15", "16-40", "41+"]
SESSION_GAP_MINUTES = 30.0

# Relative solve time bins: log10(minutes) − user's median log10(minutes).
RELTIME_EDGES = np.array([-0.6, -0.3, -0.1, 0.1, 0.3, 0.6])
RELTIME_LABELS = ["<0.25x", "0.25-0.5x", "0.5-0.8x", "0.8-1.25x", "1.25-2x", "2-4x", ">4x"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--db", default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"),
                    help="ratings artifact dir (per-size CSVs + curve + lapse)")
    ap.add_argument("--embeddings",
                    default=os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv"),
                    help="joint correct-move embedding CSV (for pair similarity)")
    ap.add_argument("--trend-bandwidth-days", type=float, default=30.0,
                    help="Gaussian kernel bandwidth of the per-user ability trend")
    ap.add_argument("--prior-info", type=float, default=4.0,
                    help="ridge info pulling the local ability toward the career fit")
    ap.add_argument("--min-user-attempts", type=int, default=30,
                    help="min joined first-time attempts for a user to enter at all")
    ap.add_argument("--min-vario-attempts", type=int, default=100,
                    help="min attempts for the same-user variogram analyses")
    ap.add_argument("--min-half", type=int, default=4,
                    help="min attempts per user-half for a problem in the split-half")
    ap.add_argument("--max-gap-days", type=float, default=60.0)
    ap.add_argument("--max-pairs-per-user", type=int, default=300_000)
    ap.add_argument("--null-sims", type=int, default=2,
                    help="model-true null re-simulations (0 = raw numbers only)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- data loading

def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() / DAY_SECONDS


def load_attempts(db_path):
    """Per-(user, problem) FIRST attempt rows: is_repeat=0, deduped by earliest
    created_at (defensive against historical server-side duplicate rows)."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT user_id, problem, success, created_at, time_used FROM tsumego_reports "
        "WHERE is_repeat = 0 ORDER BY user_id, created_at").fetchall()
    con.close()
    out, seen = [], set()
    for uid, tid, succ, ts, time_used in rows:
        key = (uid, tid)
        if key in seen:
            continue
        seen.add(key)
        out.append((uid, tid, int(bool(succ)), parse_ts(ts), time_used))
    return out


def load_problem_maps(ratings_dir):
    ratings, slopes, floors = {}, {}, {}
    for size in ALL_BOARD_SIZES:
        r, s, f = load_size_value_maps(ratings_dir, size, missing_ok=True)
        ratings.update(r)
        slopes.update(s)
        floors.update(f)
    return ratings, slopes, floors


def load_embeddings(path):
    """{tsumego_id: unit-normalized float32 embedding} from per_problem.csv."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        k0 = header.index("embed_0")
        for row in reader:
            v = np.array(row[k0:], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                out[row[0]] = v / n
    return out


# ------------------------------------------------------------------- 4PL terms

def fourpl(R, rr, sl, kappa, c, span):
    """P and dP/dability at ability R (broadcastable) for attempts (rr, sl, ...)."""
    z = np.clip(sl * (R - rr) + kappa, -40.0, 40.0)
    praw = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(c + span * praw, 1e-9, 1.0 - 1e-9)
    g = span * sl * praw * (1.0 - praw)
    return p, g


def newton_career(rr, sc, sl, kappa, c, span):
    """Unweighted career ability MLE — same iteration as fit_user_rating."""
    R = float(rr.mean())
    for _ in range(80):
        p, g = fourpl(R, rr, sl, kappa, c, span)
        denom = p * (1.0 - p)
        score = float(np.sum((sc - p) * g / denom))
        info = float(np.sum(g * g / denom))
        if info < 1e-9:
            break
        step = max(-1.0, min(1.0, score / info))
        R += step
        if abs(step) < 1e-6:
            break
    return R


def loo_local_abilities(t, rr, sc, sl, kappa, c, span, theta0, bandwidth, prior_info):
    """Per-attempt leave-one-out kernel-local ability + kernel mass.

    For each attempt i, Newton-fit ability over the user's OTHER attempts with
    Gaussian time weights exp(−½((t_j−t_i)/h)²) plus a ridge of `prior_info`
    toward the career ability. Vectorized over blocks of attempts."""
    n = len(t)
    R = np.full(n, theta0)
    mass = np.zeros(n)
    block = max(16, int(3e6 / max(n, 1)))
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        idx = np.arange(lo, hi)
        W = np.exp(-0.5 * ((t[idx, None] - t[None, :]) / bandwidth) ** 2)
        W[idx - lo, idx] = 0.0  # leave the attempt itself out
        mass[idx] = W.sum(axis=1)
        Rb = np.full(hi - lo, theta0)
        for _ in range(60):
            p, g = fourpl(Rb[:, None], rr[None, :], sl[None, :], kappa[None, :],
                          c[None, :], span[None, :])
            denom = p * (1.0 - p)
            score = (W * (sc[None, :] - p) * g / denom).sum(axis=1) \
                + prior_info * (theta0 - Rb)
            info = (W * g * g / denom).sum(axis=1) + prior_info
            step = np.clip(score / info, -1.0, 1.0)
            Rb += step
            if float(np.abs(step).max()) < 1e-5:
                break
        R[idx] = Rb
    return R, mass


# ------------------------------------------------------------------- jackknife

def jk_ratio(num, den):
    """Elementwise ratio Σnum/Σden over axis 0 (users) + leave-one-user-out
    jackknife SE. num/den: (U, ...). Cells with <2 contributing users → nan SE."""
    N = num.sum(axis=0)
    D = den.sum(axis=0)
    est = np.where(D != 0, N / np.where(D == 0, 1.0, D), np.nan)
    contrib = den != 0
    loo_den = D - den
    loo = np.where(loo_den != 0, (N - num) / np.where(loo_den == 0, 1.0, loo_den), np.nan)
    loo = np.where(contrib, loo, np.nan)
    m = contrib.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean_loo = np.nanmean(loo, axis=0)
        var = np.nansum((loo - mean_loo) ** 2, axis=0)
    se = np.where(m >= 2, np.sqrt(np.maximum(m - 1, 1) / np.maximum(m, 1) * var), np.nan)
    return est, se, m


# ------------------------------------------------------- fit + residual builder

def build_users(static, y_by_user, bandwidth, prior_info):
    """Fit career + leave-one-out local abilities and residuals for every user.

    static: {uid: dict of fixed arrays (tid,t,rr,sl,kappa,c,span,tu,emb,has_emb)}
    y_by_user: {uid: outcome array} (real or simulated)."""
    users = {}
    masses, trend_sds = [], []
    for uid, st in static.items():
        sc = y_by_user[uid]
        theta0 = newton_career(st["rr"], sc, st["sl"], st["kappa"], st["c"], st["span"])
        R, mass = loo_local_abilities(st["t"], st["rr"], sc, st["sl"], st["kappa"],
                                      st["c"], st["span"], theta0, bandwidth, prior_info)
        p, g = fourpl(R, st["rr"], st["sl"], st["kappa"], st["c"], st["span"])
        denom = p * (1.0 - p)
        users[uid] = dict(st, y=sc, p=p, g=g, denom=denom, R=R, theta0=theta0,
                          r=(sc - p) / np.sqrt(denom), s=g / np.sqrt(denom))
        masses.append(np.median(mass))
        trend_sds.append(np.std(R))
    return users, float(np.median(masses)), float(np.median(trend_sds))


def simulate_outcomes(users, rng):
    """Model-true null outcomes: constant career ability, fitted 4PL."""
    out = {}
    for uid, d in users.items():
        p0, _ = fourpl(d["theta0"], d["rr"], d["sl"], d["kappa"], d["c"], d["span"])
        out[uid] = (rng.random(len(p0)) < p0).astype(float)
    return out


# ------------------------------------------------------------------- analyses

def item_split_half(users, rng, min_half, n_boot=500):
    """Cross-half covariance of per-problem one-step rating offsets."""
    uids = sorted(users.keys())
    half = {u: int(rng.integers(0, 2)) for u in uids}
    acc = {}  # tid -> [score0, info0, n0, score1, info1, n1]
    for u in uids:
        d = users[u]
        h = half[u]
        num = -(d["y"] - d["p"]) * d["g"] / d["denom"]   # dNLL/d(rating offset)
        info = d["g"] ** 2 / d["denom"]
        for tid, sc_v, in_v in zip(d["tid"], num, info):
            a = acc.setdefault(tid, [0.0, 0.0, 0, 0.0, 0.0, 0])
            a[3 * h] += sc_v
            a[3 * h + 1] += in_v
            a[3 * h + 2] += 1
    rows = [(s0 / i0, i0, s1 / i1, i1)
            for (s0, i0, n0, s1, i1, n1) in acc.values()
            if n0 >= min_half and n1 >= min_half and i0 > 0 and i1 > 0]
    if not rows:
        return None
    e0, i0, e1, i1 = map(np.array, zip(*rows))
    w = i0 * i1 / (i0 + i1)

    def stats(sel):
        ws = w[sel] / w[sel].sum()
        m0 = float(np.sum(ws * e0[sel]))
        m1 = float(np.sum(ws * e1[sel]))
        cov = float(np.sum(ws * (e0[sel] - m0) * (e1[sel] - m1)))
        v0 = float(np.sum(ws * (e0[sel] - m0) ** 2))
        v1 = float(np.sum(ws * (e1[sel] - m1) ** 2))
        corr = cov / np.sqrt(v0 * v1) if v0 > 0 and v1 > 0 else np.nan
        return cov, corr

    cov, corr = stats(np.arange(len(w)))
    boots = np.array([stats(rng.integers(0, len(w), len(w))) for _ in range(n_boot)])
    return {
        "n_problems": len(rows),
        "cov": cov, "cov_ci": np.percentile(boots[:, 0], [2.5, 97.5]),
        "corr": corr, "corr_ci": np.percentile(boots[:, 1], [2.5, 97.5]),
        "noise_var": float(np.median(1.0 / i0 + 1.0 / i1) / 2),
    }


def form_calibration(users, uids):
    """Gap-binned calibration sums (per user, for jackknifing) + slope terciles."""
    B = len(GAP_EDGES) + 1
    sums = {k: np.zeros((len(uids), B)) for k in ("y", "p", "r", "n")}
    slope_all = np.concatenate([users[u]["sl"] for u in uids])
    terc = np.percentile(slope_all, [33.3, 66.7])
    tsum = {k: np.zeros((len(uids), 3, B)) for k in ("y", "p", "n")}
    for ui, u in enumerate(uids):
        d = users[u]
        b = np.searchsorted(GAP_EDGES, d["R"] - d["rr"])
        np.add.at(sums["y"][ui], b, d["y"])
        np.add.at(sums["p"][ui], b, d["p"])
        np.add.at(sums["r"][ui], b, d["r"])
        np.add.at(sums["n"][ui], b, 1.0)
        tb = np.searchsorted(terc, d["sl"])
        np.add.at(tsum["y"][ui], (tb, b), d["y"])
        np.add.at(tsum["p"][ui], (tb, b), d["p"])
        np.add.at(tsum["n"][ui], (tb, b), 1.0)
    return {"sums": sums, "tsum": tsum, "terc": terc}


def variogram(users, rng, max_gap, cap_pairs, min_attempts, sim_edges=None):
    """Same-user pair accumulation: (time-gap bin × similarity bin) sums of
    r_i·r_j and s_i·s_j, per user (for jackknifing). Similarity quartile edges
    are computed from a pooled pair sample unless passed in (the null sims
    reuse the real run's edges so cells align)."""
    uids = sorted(u for u in users if len(users[u]["y"]) >= min_attempts)
    T = len(TIME_LABELS)

    def sample_pairs(d, cap):
        t = d["t"]
        n = len(t)
        hi = np.searchsorted(t, t + max_gap)
        cnt = np.maximum(hi - np.arange(n) - 1, 0)
        total = int(cnt.sum())
        if total == 0:
            return None, None
        cum = np.concatenate([[0], np.cumsum(cnt)])
        pos = np.arange(total) if total <= cap else rng.integers(0, total, cap)
        i = np.searchsorted(cum, pos, side="right") - 1
        j = i + 1 + (pos - cum[i])
        return i, j

    if sim_edges is None:
        sims_sample = []
        for u in uids:
            d = users[u]
            i, j = sample_pairs(d, 400)
            if i is None:
                continue
            ok = d["has_emb"][i] & d["has_emb"][j]
            if ok.any():
                sims_sample.append(np.einsum("ij,ij->i", d["emb"][i[ok]], d["emb"][j[ok]]))
        if not sims_sample:
            return None
        sim_edges = np.percentile(np.concatenate(sims_sample), [25, 50, 75])
    S = 4  # quartile cells; index S = missing-embedding pairs

    sum_rr = np.zeros((len(uids), T, S + 1))
    sum_ss = np.zeros((len(uids), T, S + 1))
    cnt = np.zeros((len(uids), T, S + 1))
    for ui, u in enumerate(uids):
        d = users[u]
        i, j = sample_pairs(d, cap_pairs)
        if i is None:
            continue
        dt = d["t"][j] - d["t"][i]
        tb = np.clip(np.searchsorted(TIME_EDGES[1:-1], dt, side="right"), 0, T - 1)
        sb = np.full(len(i), S)
        ok = d["has_emb"][i] & d["has_emb"][j]
        if ok.any():
            sims = np.einsum("ij,ij->i", d["emb"][i[ok]], d["emb"][j[ok]])
            sb[ok] = np.searchsorted(sim_edges, sims)
        np.add.at(sum_rr[ui], (tb, sb), d["r"][i] * d["r"][j])
        np.add.at(sum_ss[ui], (tb, sb), d["s"][i] * d["s"][j])
        np.add.at(cnt[ui], (tb, sb), 1.0)
    return {"sum_rr": sum_rr, "sum_ss": sum_ss, "cnt": cnt, "sim_edges": sim_edges,
            "n_users": len(uids)}


def session_positions(t):
    """1-based within-session attempt positions (sessions split at 30-min gaps)."""
    pos = np.ones(len(t), dtype=int)
    for k in range(1, len(t)):
        pos[k] = 1 if (t[k] - t[k - 1]) * DAY_SECONDS / 60.0 > SESSION_GAP_MINUTES \
            else pos[k - 1] + 1
    return pos


def state_proxies(users, uids):
    P = len(SESSION_POS_LABELS)
    RT = len(RELTIME_LABELS)
    pos_r = np.zeros((len(uids), P))
    pos_n = np.zeros((len(uids), P))
    rt_r = np.zeros((len(uids), RT))
    rt_n = np.zeros((len(uids), RT))
    for ui, u in enumerate(uids):
        d = users[u]
        pos = session_positions(d["t"])
        pb = np.clip(np.searchsorted(SESSION_POS_EDGES, pos, side="right") - 1, 0, P - 1)
        np.add.at(pos_r[ui], pb, d["r"])
        np.add.at(pos_n[ui], pb, 1.0)
        tu = d["tu"]
        valid = np.isfinite(tu) & (tu > 500)
        if valid.sum() >= 20:
            lt = np.log10(tu[valid] / 60000.0)
            rb = np.searchsorted(RELTIME_EDGES, lt - np.median(lt))
            np.add.at(rt_r[ui], rb, d["r"][valid])
            np.add.at(rt_n[ui], rb, 1.0)
    return {"pos_r": pos_r, "pos_n": pos_n, "rt_r": rt_r, "rt_n": rt_n}


def run_analyses(users, args, rng, sim_edges=None):
    uids = sorted(users.keys())
    all_r = np.concatenate([users[u]["r"] for u in uids])
    all_p = np.concatenate([users[u]["p"] for u in uids])
    all_y = np.concatenate([users[u]["y"] for u in uids])
    return {
        "overall": {"obs": float(all_y.mean()), "pred": float(all_p.mean()),
                    "mean_r": float(all_r.mean()), "var_r": float(all_r.var())},
        "item": item_split_half(users, rng, args.min_half),
        "form": form_calibration(users, uids),
        "vario": variogram(users, rng, args.max_gap_days, args.max_pairs_per_user,
                           args.min_vario_attempts, sim_edges=sim_edges),
        "proxies": state_proxies(users, uids),
    }


# --------------------------------------------------------------------- report

def fmt(v, digits=3, width=8):
    return f"{'--':>{width}}" if v is None or not np.isfinite(v) else f"{v:>{width}.{digits}f}"


def null_mean(nulls, path):
    """Mean of a (possibly array) value extracted from each null result."""
    vals = []
    for n in nulls:
        v = n
        for key in path:
            v = v[key] if v is not None else None
        if v is not None:
            vals.append(v)
    return np.mean(np.array(vals, dtype=float), axis=0) if vals else None


def print_report(real, nulls, args, ranks_per_rating, med_rating, n_users, n_att,
                 med_mass, med_trend_sd):
    def gap_labels():
        return ([f"< {GAP_EDGES[0]:+.1f}"]
                + [f"{a:+.1f}..{b:+.1f}" for a, b in zip(GAP_EDGES[:-1], GAP_EDGES[1:])]
                + [f"> {GAP_EDGES[-1]:+.1f}"])

    print()
    print("=" * 78)
    print("0. DATA + OVERALL CALIBRATION")
    print("=" * 78)
    o = real["overall"]
    print(f"  users {n_users}  attempts {n_att}  |  trend bandwidth "
          f"{args.trend_bandwidth_days:.0f}d  median kernel mass {med_mass:.0f}  "
          f"median within-user trend SD {med_trend_sd:.3f} rating")
    print(f"  observed {o['obs']:.4f}  predicted {o['pred']:.4f}  "
          f"mean Pearson residual {o['mean_r']:+.4f}  residual variance {o['var_r']:.4f}")
    nv = null_mean(nulls, ["overall", "var_r"])
    if nv is not None:
        print(f"  null residual variance {nv:.4f}  (excess over model-true: "
              f"{o['var_r'] - nv:+.4f})")
    print(f"  unit conversion: {ranks_per_rating:.2f} ranks per rating unit at the "
          f"median served rating ({med_rating:.2f})")

    print()
    print("=" * 78)
    print("1. PROBLEM-RANK ERROR (user split-half, per-problem rating offsets)")
    print("=" * 78)
    ish = real["item"]
    if ish is None:
        print("  not enough multi-attempt problems.")
    else:
        ncov = null_mean(nulls, ["item", "cov"])
        cov_x = ish["cov"] - (ncov or 0.0)
        sig = np.sqrt(max(cov_x, 0.0))
        lo, hi = ish["cov_ci"]
        print(f"  problems with >= {args.min_half} attempts per half: {ish['n_problems']}")
        print(f"  cross-half offset covariance (true item-error variance):")
        print(f"    real {ish['cov']:+.4f} rating^2  [boot {lo:+.4f}, {hi:+.4f}]"
              + (f"   null {ncov:+.4f}" if ncov is not None else ""))
        print(f"    excess -> sigma_item = {sig:.3f} rating units "
              f"= {sig * ranks_per_rating:.2f} ranks")
        print(f"  cross-half correlation {ish['corr']:+.3f} "
              f"[{ish['corr_ci'][0]:+.3f}, {ish['corr_ci'][1]:+.3f}]")
        print(f"  median per-problem sampling noise variance {ish['noise_var']:.4f} rating^2")

    print()
    print("=" * 78)
    print("2. FORM ERROR (calibration vs ability-difficulty gap; 0 anchors P=0.8)")
    print("=" * 78)
    f = real["form"]
    obs, _, _ = jk_ratio(f["sums"]["y"], f["sums"]["n"])
    pred, _, _ = jk_ratio(f["sums"]["p"], f["sums"]["n"])
    resid, rse, _ = jk_ratio(f["sums"]["r"], f["sums"]["n"])
    n_obs_null = null_mean(nulls, ["form", "sums", "y"])
    n_n_null = null_mean(nulls, ["form", "sums", "n"])
    n_p_null = null_mean(nulls, ["form", "sums", "p"])
    null_dev = None
    if n_obs_null is not None:
        with np.errstate(invalid="ignore"):
            null_dev = (n_obs_null.sum(0) / n_n_null.sum(0)
                        - n_p_null.sum(0) / n_n_null.sum(0))
    print(f"  {'gap bin':>12} {'n':>7} {'observed':>9} {'predicted':>9} {'obs-pred':>9}"
          f" {'null':>7} {'excess':>7}   by slope tercile (lo/mid/hi)")
    B = len(GAP_EDGES) + 1
    for b in range(B):
        n_b = int(f["sums"]["n"].sum(axis=0)[b])
        if n_b == 0:
            continue
        parts = []
        for k in range(3):
            nn = f["tsum"]["n"][:, k, b].sum()
            parts.append(f"{(f['tsum']['y'][:, k, b].sum() / nn - f['tsum']['p'][:, k, b].sum() / nn):+.3f}"
                         if nn >= 200 else "   --")
        dv = obs[b] - pred[b]
        nl = null_dev[b] if null_dev is not None else np.nan
        print(f"  {gap_labels()[b]:>12} {n_b:>7} {fmt(obs[b], 4, 9)} {fmt(pred[b], 4, 9)}"
              f" {fmt(dv, 4, 9)} {fmt(nl, 3, 7)} {fmt(dv - nl, 3, 7)}"
              f"   {' / '.join(parts)}")
    print(f"  (slope terciles at a_p <= {f['terc'][0]:.2f} / <= {f['terc'][1]:.2f} / above;"
          f" null = model-true pipeline artifact; excess = real form error)")

    print()
    print("=" * 78)
    print("3+4. SAME-USER VARIOGRAM (implied ability covariance, rating units^2)")
    print("=" * 78)
    vg = real["vario"]
    if vg is None:
        print("  no usable pairs.")
        return
    nvg_rr = null_mean(nulls, ["vario", "sum_rr"])
    nvg_ss = null_mean(nulls, ["vario", "sum_ss"])

    def cell_est(rr_sums, ss_sums, sel_time=None, sel_sim=None):
        rr, ss = rr_sums, ss_sums
        if sel_sim is not None:
            rr, ss = rr[:, :, sel_sim], ss[:, :, sel_sim]
        if sel_time is not None:
            rr, ss = rr[:, sel_time], ss[:, sel_time]
        return rr, ss

    print(f"  users {vg['n_users']} (>= {args.min_vario_attempts} attempts); pairs "
          f"capped at {args.max_pairs_per_user}/user, gap <= {args.max_gap_days:.0f}d")
    print(f"  similarity quartile edges: {vg['sim_edges'][0]:+.3f} / "
          f"{vg['sim_edges'][1]:+.3f} / {vg['sim_edges'][2]:+.3f}")

    # Time profile (similarity-marginal) -> EBB AND FLOW
    num, den = vg["sum_rr"].sum(axis=2), vg["sum_ss"].sum(axis=2)
    est, se, _ = jk_ratio(num, den)
    nprof = None
    if nvg_rr is not None:
        with np.errstate(invalid="ignore"):
            nprof = nvg_rr.sum(axis=(0, 2)) / nvg_ss.sum(axis=(0, 2))
    n_t = vg["cnt"].sum(axis=2).sum(axis=0)
    print("\n  TIME PROFILE (all similarities pooled) — decay of the excess = ebb and flow:")
    print(f"  {'gap':>8} {'pairs':>10} {'real':>9} {'se':>8} {'null':>9} {'excess':>9}"
          f" {'implied SD':>11}")
    for k in range(len(TIME_LABELS)):
        nl = nprof[k] if nprof is not None else np.nan
        ex = est[k] - nl
        sd = np.sqrt(ex) if np.isfinite(ex) and ex > 0 else np.nan
        tail = "" if not np.isfinite(sd) else f"  ({sd * ranks_per_rating:.2f} ranks)"
        print(f"  {TIME_LABELS[k]:>8} {int(n_t[k]):>10} {fmt(est[k], 4, 9)}"
              f" {fmt(se[k], 4, 8)} {fmt(nl, 4, 9)} {fmt(ex, 4, 9)} {fmt(sd, 3, 11)}{tail}")

    # Similarity profile at long gaps -> STABLE COMPONENTS
    long_mask = np.array([lbl in LONG_GAP_LABELS for lbl in TIME_LABELS])
    est_s, se_s, _ = jk_ratio(vg["sum_rr"][:, long_mask, :4].sum(axis=1),
                              vg["sum_ss"][:, long_mask, :4].sum(axis=1))
    nsim = None
    if nvg_rr is not None:
        with np.errstate(invalid="ignore"):
            nsim = (nvg_rr[:, long_mask, :4].sum(axis=(0, 1))
                    / nvg_ss[:, long_mask, :4].sum(axis=(0, 1)))
    print("\n  SIMILARITY PROFILE at long gaps (>= 7d) — rise of the excess with")
    print("  similarity = stable individual skill components:")
    n_s = vg["cnt"][:, long_mask, :4].sum(axis=(0, 1))
    sim_labels = ["Q1 (least similar)", "Q2", "Q3", "Q4 (most similar)"]
    print(f"  {'':>20} {'pairs':>10} {'real':>9} {'se':>8} {'null':>9} {'excess':>9}")
    for k in range(4):
        nl = nsim[k] if nsim is not None else np.nan
        print(f"  {sim_labels[k]:>20} {int(n_s[k]):>10} {fmt(est_s[k], 4, 9)}"
              f" {fmt(se_s[k], 4, 8)} {fmt(nl, 4, 9)} {fmt(est_s[k] - nl, 4, 9)}")

    # Full matrix (real - null), compact
    est_m, _, _ = jk_ratio(vg["sum_rr"][:, :, :4], vg["sum_ss"][:, :, :4])
    if nvg_rr is not None:
        with np.errstate(invalid="ignore"):
            null_m = nvg_rr[:, :, :4].sum(axis=0) / nvg_ss[:, :, :4].sum(axis=0)
        est_m = est_m - null_m
        title = "FULL MATRIX, REAL - NULL"
    else:
        title = "FULL MATRIX, RAW"
    print(f"\n  {title} (ability cov x1000; rows time gap, cols sim quartile):")
    print(f"  {'':>8} " + " ".join(f"{q:>8}" for q in ["Q1", "Q2", "Q3", "Q4"]))
    for k in range(len(TIME_LABELS)):
        print(f"  {TIME_LABELS[k]:>8} "
              + " ".join(fmt(est_m[k, q] * 1000, 1, 8) for q in range(4)))

    print()
    print("=" * 78)
    print("5. STATE PROXIES (mean Pearson residual; positive = better than model)")
    print("=" * 78)
    pr = real["proxies"]
    est, se, _ = jk_ratio(pr["pos_r"], pr["pos_n"])
    npos = None
    if nulls:
        n_r = null_mean(nulls, ["proxies", "pos_r"])
        n_n = null_mean(nulls, ["proxies", "pos_n"])
        with np.errstate(invalid="ignore"):
            npos = n_r.sum(0) / n_n.sum(0)
    print("  within-session position:            n      real ± se       null")
    for k in range(len(SESSION_POS_LABELS)):
        nl = npos[k] if npos is not None else np.nan
        print(f"    {SESSION_POS_LABELS[k]:>6} {int(pr['pos_n'].sum(axis=0)[k]):>16} "
              f"{fmt(est[k], 4, 9)} ± {fmt(se[k], 4, 6).strip():<7} {fmt(nl, 4, 8)}")
    est, se, _ = jk_ratio(pr["rt_r"], pr["rt_n"])
    nrt = None
    if nulls:
        n_r = null_mean(nulls, ["proxies", "rt_r"])
        n_n = null_mean(nulls, ["proxies", "rt_n"])
        with np.errstate(invalid="ignore"):
            nrt = n_r.sum(0) / n_n.sum(0)
    print("  solve time relative to own median:")
    for k in range(len(RELTIME_LABELS)):
        nl = nrt[k] if nrt is not None else np.nan
        print(f"    {RELTIME_LABELS[k]:>10} {int(pr['rt_n'].sum(axis=0)[k]):>12} "
              f"{fmt(est[k], 4, 9)} ± {fmt(se[k], 4, 6).strip():<7} {fmt(nl, 4, 8)}")

    print()
    print("Read the REAL - NULL excess everywhere: the null re-simulates outcomes at")
    print("constant per-user ability under the fitted 4PL and re-runs the whole")
    print("pipeline, so it absorbs trend-fit leakage, errors-in-variables flattening,")
    print("and split-half noise. Remaining caveats: solve time is co-determined with")
    print("the outcome; item analysis covers only the frequently-attempted head.")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading artifacts ...")
    ratings, slopes, floors = load_problem_maps(args.dir)
    lapse = load_lapse(args.dir)
    embeddings = load_embeddings(args.embeddings)
    xs, ys = load_rating_rank_curve(args.dir)  # rating knots → rank-index knots
    attempts = load_attempts(args.db)
    print(f"  {len(attempts)} first-time attempts, {len(ratings)} rated problems, "
          f"{len(embeddings)} embeddings, lapse={lapse:.4f}")

    by_user = {}
    for uid, tid, succ, t, time_used in attempts:
        if tid in ratings:
            by_user.setdefault(uid, []).append((tid, succ, t, time_used))
    by_user = {u: v for u, v in by_user.items() if len(v) >= args.min_user_attempts}
    n_att = sum(len(v) for v in by_user.values())
    print(f"  {len(by_user)} users with >= {args.min_user_attempts} joined attempts "
          f"({n_att} attempts)")

    # Static per-user arrays (everything that doesn't depend on outcomes).
    static, y_real = {}, {}
    for uid, rows in by_user.items():
        rows.sort(key=lambda r: r[2])
        tid = [r[0] for r in rows]
        rr = np.array([ratings[x] for x in tid])
        sl = np.array([slopes.get(x) or LOGIT_PER_RATING for x in tid])
        sl = np.where(np.isfinite(sl) & (sl > 0), sl, LOGIT_PER_RATING)
        c = np.array([min(max(floors.get(x) or 0.0, 0.0), FLOOR_MAX) for x in tid])
        span = np.clip(1.0 - c - max(0.0, lapse), 1e-6, 1.0)
        q = np.clip((CALIBRATION_P - c) / span, 1e-6, 1.0 - 1e-6)
        static[uid] = {
            "tid": tid,
            "t": np.array([r[2] for r in rows], dtype=float),
            "tu": np.array([r[3] if r[3] is not None else np.nan for r in rows],
                           dtype=float),
            "rr": rr, "sl": sl, "c": c, "span": span,
            "kappa": np.log(q / (1.0 - q)),
            "has_emb": np.array([x in embeddings for x in tid]),
            "emb": np.stack([embeddings.get(x, np.zeros(64, dtype=np.float32))
                             for x in tid]),
        }
        y_real[uid] = np.array([r[1] for r in rows], dtype=float)

    t0 = time.time()
    print("Fitting per-user leave-one-out local ability trends (real data) ...")
    users, med_mass, med_trend_sd = build_users(static, y_real,
                                                args.trend_bandwidth_days,
                                                args.prior_info)
    print(f"  done in {time.time() - t0:.0f}s")
    real = run_analyses(users, args, rng)
    sim_edges = real["vario"]["sim_edges"] if real["vario"] else None

    nulls = []
    for k in range(args.null_sims):
        t0 = time.time()
        print(f"Null simulation {k + 1}/{args.null_sims} (constant-ability re-fit) ...")
        y_sim = simulate_outcomes(users, rng)
        sim_users, _, _ = build_users(static, y_sim, args.trend_bandwidth_days,
                                      args.prior_info)
        nulls.append(run_analyses(sim_users, args, rng, sim_edges=sim_edges))
        print(f"  done in {time.time() - t0:.0f}s")

    uids = sorted(users.keys())
    med_rating = float(np.median(np.concatenate([users[u]["rr"] for u in uids])))
    eps = 0.05
    ranks_per_rating = float((np.interp(med_rating + eps, xs, ys)
                              - np.interp(med_rating - eps, xs, ys)) / (2 * eps))
    print_report(real, nulls, args, ranks_per_rating, med_rating,
                 len(users), n_att, med_mass, med_trend_sd)


if __name__ == "__main__":
    main()
