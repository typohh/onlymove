#!/usr/bin/env python3
"""Between-user predictive power of the best-vs-second-best SCORE GAP axis.

Hypothesis: the score margin between the solving move and the best failing
alternative (from the move-eval cache: solution score minus best-alt score,
side-to-move perspective) is a "does precise counting matter here" axis. Users
could plausibly differ on it AFTER rating — how reliably they calculate the
score, and whether they bother to calculate at all — making it a candidate
single, non-KataGo-embedding weakness axis.

Feature per problem: f = z(-log1p(max(score_gap, 0))), pooled across boards —
positive f = TIGHT margin (counting matters), negative = blow-out margin.
(--feature wr_gap swaps in the logit-winrate gap as a comparator; that one is
mostly plain difficulty, so rating should absorb it — if score-gap shows more
between-user structure than wr-gap, the counting story is specific.)

All users, all novel (is_repeat = 0) attempts, no timeframe limit. Per-user
model: success ~ alpha_u + gamma_u * problemRank + delta_u * f.

Tests:
  1. POPULATION effect — inverse-variance-weighted mean of delta_u: does the
     axis carry a shared miscalibration after rating? (context, not the target)
  2. BETWEEN-USER structure — chronological split-half: delta_u fit separately
     on each half; noise-corrected between-user SD
     tau^2 = Var(delta_hat) - mean(SE^2), split-half correlation vs its
     attenuation ceiling, and the weighted delta_2 ~ delta_1 slope (a
     shrinkage-free forward test: does a user's first-half coefficient predict
     their second-half performance pattern?).
  3. FORWARD kappa — walk-forward (trailing 500-attempt window, refit every 50):
     held-out success ~ offset(baseline logit) + kappa * (delta_u_window * f).
     kappa ~ 1 = the personalized axis predicts at face value; 0 = noise.
  4. WHO calculates — across users, correlation of delta_u with log median
     solve time (partial on rank): do slower, more deliberate players do
     relatively better on tight-margin problems?

--feature joint fits BOTH axes per user (they correlate ~0.5 across the corpus)
and reports each coefficient's split-half structure plus the walk-forward kappa
of the SCORE term over a personalized wr-gap baseline — i.e. whether a
score-specific counting skill exists beyond general tightness.

Usage:
  ./.venv/bin/python3 -m auxiliary.eval_score_gap_axis
  ./.venv/bin/python3 -m auxiliary.eval_score_gap_axis --feature wr_gap
  ./.venv/bin/python3 -m auxiliary.eval_score_gap_axis --feature joint
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")

WINDOW = 500
EVAL_BLOCK = 50
WARMUP = 120
RANK_NORM = 29.0


# The feature/cache implementations live in utils.rating_features (the production
# data-source home shared with pipeline.train_rating and auxiliary.rating_dev.*)
# — this script keeps thin aliases so its CLI and importers are unchanged.
from utils.rating_features import (  # noqa: E402
    load_backend_problem_params, load_backend_rank_curve,
    load_game_phase, load_tightness_gaps, zscore_neglog,
)

load_feature = load_tightness_gaps


def load_ranks():
    """tid -> normalized problemRank from the BACKEND rating outputs (ratings/
    CSVs + ratings/rating_to_rank.json — the stage before package_for_pwa), so
    the axis batteries and the residual-axes trainer always see the freshly
    trained ratings, never a stale/half-packaged dist. (Client-replica
    harnesses read pwa/tsumegos on purpose; this is model-development code.)"""
    params = load_backend_problem_params()
    curve = load_backend_rank_curve()
    xs = np.array(curve["ratings"]); ys = np.array(curve["rank_indices"])
    return {t: float(np.interp(v[0], xs, ys)) / RANK_NORM for t, v in params.items()}


def irls(X, y, ridge=1e-6, w0=None):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    for _ in range(100):
        z = np.clip(X @ w, -35, 35)
        p = 1 / (1 + np.exp(-z))
        Wd = p * (1 - p) + 1e-9
        H = X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * w)
        # step-halving on the penalized deviance
        def dev(wv):
            zz = np.clip(X @ wv, -35, 35)
            pp = np.clip(1 / (1 + np.exp(-zz)), 1e-9, 1 - 1e-9)
            return -(y * np.log(pp) + (1 - y) * np.log(1 - pp)).sum() + 0.5 * ridge * (wv @ wv)
        d0 = dev(w)
        for _h in range(12):
            if dev(w + step) <= d0 + 1e-9:
                break
            step *= 0.5
        w = w + step
        if np.max(np.abs(step)) < 1e-9:
            break
    z = np.clip(X @ w, -35, 35)
    p = 1 / (1 + np.exp(-z))
    Wd = p * (1 - p) + 1e-9
    cov = np.linalg.inv(X.T @ (X * Wd[:, None]) + ridge * np.eye(X.shape[1]))
    return w, np.sqrt(np.diag(cov))


def run_joint(min_attempts):
    """Both axes per user: success ~ a + g*rank + d_wr*f_wr + d_sc*f_sc."""
    f_sc = zscore_neglog(load_feature("score_gap"))
    f_wr = zscore_neglog(load_feature("wr_gap"))
    prank = load_ranks()
    common = [t for t in f_sc if t in f_wr]
    print("corr(f_score, f_wr) over corpus: %+.3f"
          % np.corrcoef([f_sc[t] for t in common], [f_wr[t] for t in common])[0, 1])

    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """SELECT user_id, problem, success FROM tsumego_reports
            WHERE is_repeat = 0 GROUP BY user_id, problem, created_at
            ORDER BY user_id, created_at, id""").fetchall()
    con.close()
    by_user = defaultdict(list)
    for u, p, s in rows:
        if p in f_sc and p in f_wr and p in prank:
            by_user[u].append((prank[p], f_wr[p], f_sc[p], int(bool(s))))

    print(f"\n=== split-half, joint model (users with >= {min_attempts}) ===")
    stats = {"score | wr": [], "wr | score": []}
    for u, a in by_user.items():
        if len(a) < min_attempts:
            continue
        A = np.array(a, float)
        X = np.column_stack([np.ones(len(A)), A[:, 0], A[:, 1], A[:, 2]])
        y = A[:, 3]
        h = len(A) // 2
        try:
            wa, sa = irls(X[:h], y[:h])
            wb, sb = irls(X[h:], y[h:])
        except np.linalg.LinAlgError:
            continue
        if not (np.isfinite(sa[3]) and np.isfinite(sb[3]) and sa[3] < 5 and sb[3] < 5):
            continue
        stats["score | wr"].append((wa[3], wb[3], sa[3]))
        stats["wr | score"].append((wa[2], wb[2], sa[2]))
    for name, tr in stats.items():
        a, b, sea = (np.array(v) for v in zip(*tr))
        tau2 = a.var() - (sea**2).mean()
        r = np.corrcoef(a, b)[0, 1]
        ac = a - a.mean()
        slope = (ac * (b - b.mean())).mean() / (ac**2).mean()
        print(f"  {name}: n={len(a)}  split-half corr {r:+.3f}, fwd slope {slope:+.3f}, "
              f"tau {np.sqrt(max(tau2, 0)):.3f}")

    # walk-forward kappa of the SCORE term over a personalized wr-gap baseline
    off_all, sc_all, y_all = [], [], []
    for u, a in by_user.items():
        if len(a) < WARMUP + EVAL_BLOCK:
            continue
        A = np.array(a, float)
        Xf = np.column_stack([np.ones(len(A)), A[:, 0], A[:, 1], A[:, 2]])
        Xb = Xf[:, :3]
        y = A[:, 3]
        wf = wb = None
        for t0 in range(WARMUP, len(A), EVAL_BLOCK):
            lo = max(0, t0 - WINDOW); hi = min(t0 + EVAL_BLOCK, len(A))
            try:
                wf, _ = irls(Xf[lo:t0], y[lo:t0], w0=wf)
                wb, _ = irls(Xb[lo:t0], y[lo:t0], w0=wb)
            except np.linalg.LinAlgError:
                continue
            off_all.append(np.clip(Xb[t0:hi] @ wb, -35, 35))
            sc_all.append(wf[3] * A[t0:hi, 2])
            y_all.append(y[t0:hi])
    off = np.concatenate(off_all); sc = np.concatenate(sc_all); yy = np.concatenate(y_all)
    Xk = np.column_stack([np.ones(len(yy)), sc])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(off + Xk @ w, -35, 35)))
        Wd = p * (1 - p) + 1e-9
        step = np.linalg.solve(Xk.T @ (Xk * Wd[:, None]) + 1e-9 * np.eye(2), Xk.T @ (yy - p))
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-np.clip(off + Xk @ w, -35, 35)))
    Wd = p * (1 - p) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv(Xk.T @ (Xk * Wd[:, None]))))[1]
    print(f"\nwalk-forward kappa of SCORE term given personalized wr baseline: "
          f"{w[1]:+.3f} +- {se:.3f} (z {w[1] / se:+.1f}), n={len(yy)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--feature",
                    choices=("score_gap", "wr_gap", "joint", "game_phase", "phase_ratio"),
                    default="score_gap")
    ap.add_argument("--rebuild-phase-cache", action="store_true")
    ap.add_argument("--min-attempts", type=int, default=400,
                    help="split-half needs this many usable attempts (default 400)")
    args = ap.parse_args()

    if args.feature == "joint":
        run_joint(args.min_attempts)
        return

    prank = load_ranks()
    if args.feature == "phase_ratio":
        # main/(prefix+main): the share of the game still ahead as continuation.
        # THE combination that carries the person-signal the raw total cancels
        # (prefix/main correlate -0.53 across the corpus while users load on
        # them same-signed; see the game_phase finding). Keeps kappa ~ +0.17
        # (z 5.6) beyond personalized wr/score tightness.
        phase = load_game_phase(args.rebuild_phase_cache)
        ratio = {t: v[2] / max(v[0] + v[1] + v[2], 1) for t, v in phase.items()}
        fz = {}
        for size in (9, 13):
            ids = [t for t, v in phase.items() if v[3] == size]
            vals = np.array([ratio[t] for t in ids])
            fz.update({t: (ratio[t] - vals.mean()) / vals.std() for t in ids})
        gap_raw = ratio
        print("phase ratio main/(prefix+main) quantiles: "
              f"{np.percentile(list(ratio.values()), [5, 25, 50, 75, 95]).round(2)}")
        f_label = "f > 0 = continuation is a larger share (position arose earlier)"
    elif args.feature == "game_phase":
        # total game length at resolution = setup stones + prefix moves + main
        # line; z-scored WITHIN each board size (raw counts aren't comparable
        # across 9x9/13x13), positive = later in the game.
        phase = load_game_phase(args.rebuild_phase_cache)
        totals = {t: v[0] + v[1] + v[2] for t, v in phase.items()}
        fz = {}
        for size in (9, 13):
            ids = [t for t, v in phase.items() if v[3] == size]
            vals = np.array([totals[t] for t in ids])
            mu, sd = vals.mean(), vals.std()
            fz.update({t: (totals[t] - mu) / sd for t in ids})
        gap_raw = totals
        comp = np.array([phase[t][:3] for t in phase])
        print("game_phase components (setup, prefix, main) means: "
              f"{comp.mean(axis=0).round(1)}; total quantiles "
              f"{np.percentile(list(totals.values()), [5, 25, 50, 75, 95]).round(0)}")
        f_label = "f > 0 = later in the game"
    else:
        gap_raw = load_feature(args.feature)

    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """SELECT user_id, problem, success, time_used
             FROM tsumego_reports
            WHERE is_repeat = 0
            GROUP BY user_id, problem, created_at
            ORDER BY user_id, created_at, id"""
    ).fetchall()
    con.close()

    if args.feature not in ("game_phase", "phase_ratio"):
        # pooled z of -log1p(gap) (clip tiny negatives: solution picked by winrate,
        # so a rare negative score gap is real but off-hypothesis; fold to 0)
        vals = np.array([-np.log1p(max(v, 0.0)) for v in gap_raw.values()])
        mu, sd = vals.mean(), vals.std()
        fz = {t: (-np.log1p(max(v, 0.0)) - mu) / sd for t, v in gap_raw.items()}
        f_label = "f > 0 = tight margin, counting matters"
        g_arr = np.array(list(gap_raw.values()))
        print(f"raw gap quantiles: {np.percentile(g_arr, [5, 25, 50, 75, 95]).round(2)}")
    print(f"feature = {args.feature}; corpus coverage {len(gap_raw)} problems")
    rk = np.array([prank[t] for t in gap_raw if t in prank])
    fv = np.array([fz[t] for t in gap_raw if t in prank])
    print(f"corr(f, problemRank) = {np.corrcoef(fv, rk)[0, 1]:+.3f}  ({f_label})\n")

    by_user = defaultdict(list)
    for user, problem, success, t_used in rows:
        if problem in fz and problem in prank:
            by_user[user].append((prank[problem], fz[problem], int(bool(success)),
                                  t_used if t_used and t_used > 0 else np.nan))

    # ---------------- split-half between-user structure ---------------------------
    print(f"=== 2. split-half between-user structure (users with >= {args.min_attempts}) ===")
    d1s, d2s, se1s, se2s, users, deltas, dses, medtimes, ranks_u = [], [], [], [], [], [], [], [], []
    for user, a in by_user.items():
        if len(a) < args.min_attempts:
            continue
        A = np.array(a, dtype=float)
        X = np.column_stack([np.ones(len(A)), A[:, 0], A[:, 1]])
        y = A[:, 2]
        h = len(A) // 2
        try:
            w1, se1 = irls(X[:h], y[:h])
            w2, se2 = irls(X[h:], y[h:])
            wall, seall = irls(X, y)
        except np.linalg.LinAlgError:
            continue
        if not (np.isfinite(se1[2]) and np.isfinite(se2[2]) and se1[2] < 5 and se2[2] < 5):
            continue
        users.append(user)
        d1s.append(w1[2]); d2s.append(w2[2]); se1s.append(se1[2]); se2s.append(se2[2])
        deltas.append(wall[2]); dses.append(seall[2])
        medtimes.append(np.nanmedian(A[:, 3]))
        ranks_u.append(np.median(A[:, 0]))
    d1s, d2s = np.array(d1s), np.array(d2s)
    se1s, se2s = np.array(se1s), np.array(se2s)
    deltas, dses = np.array(deltas), np.array(dses)
    print(f"users: {len(users)}")

    # 1. population effect
    ivw = (deltas / dses**2).sum() / (1 / dses**2).sum()
    ivw_se = np.sqrt(1 / (1 / dses**2).sum())
    print(f"\n=== 1. population effect (IVW mean delta) ===")
    print(f"delta_pop = {ivw:+.4f} +- {ivw_se:.4f} (z = {ivw / ivw_se:+.1f})  "
          f"[logit per SD of f; negative = tight-margin problems are harder than rating implies]")

    # noise-corrected between-user variance (on the centered full-fit deltas)
    exc = deltas.var() - (dses**2).mean()
    tau = np.sqrt(max(exc, 0.0))
    print(f"\nbetween-user: Var(delta_hat) = {deltas.var():.4f}, mean SE^2 = {(dses**2).mean():.4f}"
          f"  =>  tau (true between-user SD) = {tau:.3f} logits/SD")
    # split-half correlation vs attenuation ceiling
    r = np.corrcoef(d1s, d2s)[0, 1]
    ceil1 = tau**2 / (tau**2 + (se1s**2).mean())
    ceil2 = tau**2 / (tau**2 + (se2s**2).mean())
    ceiling = np.sqrt(max(ceil1, 0) * max(ceil2, 0))
    print(f"split-half corr(delta_1, delta_2) = {r:+.3f}   "
          f"(attenuation ceiling if tau real: {ceiling:.3f})")
    # weighted forward slope delta2 ~ delta1
    wts = 1 / (se2s**2)
    d1c = d1s - np.average(d1s, weights=wts)
    slope = np.average(d1c * (d2s - np.average(d2s, weights=wts)), weights=wts) / np.average(d1c**2, weights=wts)
    # bootstrap SE over users
    rng = np.random.default_rng(0)
    sl = []
    for _ in range(2000):
        i = rng.integers(0, len(d1s), len(d1s))
        w_ = wts[i]; a_ = d1s[i] - np.average(d1s[i], weights=w_)
        b_ = d2s[i] - np.average(d2s[i], weights=w_)
        va = np.average(a_**2, weights=w_)
        if va > 0:
            sl.append(np.average(a_ * b_, weights=w_) / va)
    print(f"forward slope delta_2 ~ delta_1 = {slope:+.3f} "
          f"[{np.percentile(sl, 2.5):+.3f}, {np.percentile(sl, 97.5):+.3f}] "
          f"(expected ~ shrinkage {ceil1:.2f} if tau real, 0 if noise)")

    # ---------------- 3. forward kappa (walk-forward) -----------------------------
    print(f"\n=== 3. walk-forward kappa (window {WINDOW}, refit every {EVAL_BLOCK}) ===")
    off_all, sc_all, y_all = [], [], []
    for user, a in by_user.items():
        if len(a) < WARMUP + EVAL_BLOCK:
            continue
        A = np.array(a, dtype=float)
        X = np.column_stack([np.ones(len(A)), A[:, 0], A[:, 1]])
        Xb = X[:, :2]
        y = A[:, 2]
        wf = wb = None
        for t0 in range(WARMUP, len(A), EVAL_BLOCK):
            lo = max(0, t0 - WINDOW)
            hi = min(t0 + EVAL_BLOCK, len(A))
            try:
                wf, _ = irls(X[lo:t0], y[lo:t0], w0=wf)
                wb, _ = irls(Xb[lo:t0], y[lo:t0], w0=wb)
            except np.linalg.LinAlgError:
                continue
            off_all.append(np.clip(Xb[t0:hi] @ wb, -35, 35))
            sc_all.append(wf[2] * A[t0:hi, 1])       # the personalized axis prediction
            y_all.append(y[t0:hi])
    off = np.concatenate(off_all); sc = np.concatenate(sc_all); yy = np.concatenate(y_all)
    Xk = np.column_stack([np.ones(len(yy)), sc])
    w = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(off + Xk @ w, -35, 35)))
        Wd = p * (1 - p) + 1e-9
        H = Xk.T @ (Xk * Wd[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, Xk.T @ (yy - p))
        w += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-np.clip(off + Xk @ w, -35, 35)))
    Wd = p * (1 - p) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv(Xk.T @ (Xk * Wd[:, None]))))[1]
    print(f"held-out attempts: {len(yy)};  kappa = {w[1]:+.3f} +- {se:.3f} (z = {w[1] / se:+.1f}; "
          f"1 = face value, 0 = noise)")

    # ---------------- 4. who calculates? -------------------------------------------
    print("\n=== 4. does delta_u track deliberateness (log median solve time)? ===")
    mt = np.log10(np.array(medtimes) / 1000.0)
    ru = np.array(ranks_u)
    ok = np.isfinite(mt)
    # partial corr of delta with log time, controlling rank (weights: 1/SE^2)
    wts = 1 / dses[ok]**2
    def wresid(v, c):
        c1 = np.column_stack([np.ones(c.shape[0]), c])
        beta = np.linalg.lstsq(c1 * np.sqrt(wts)[:, None], v * np.sqrt(wts), rcond=None)[0]
        return v - c1 @ beta
    dr = wresid(deltas[ok], ru[ok, None])
    tr = wresid(mt[ok], ru[ok, None])
    num = np.average(dr * tr, weights=wts)
    pc = num / np.sqrt(np.average(dr**2, weights=wts) * np.average(tr**2, weights=wts))
    rng = np.random.default_rng(1)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, ok.sum(), ok.sum())
        w_ = wts[i]
        a_ = dr[i] - np.average(dr[i], weights=w_)
        b_ = tr[i] - np.average(tr[i], weights=w_)
        den = np.sqrt(np.average(a_**2, weights=w_) * np.average(b_**2, weights=w_))
        if den > 0:
            bs.append(np.average(a_ * b_, weights=w_) / den)
    print(f"partial corr(delta_u, log median time | rank) = {pc:+.3f} "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  "
          f"(positive = slower players relatively better on tight-margin problems)")


if __name__ == "__main__":
    main()
