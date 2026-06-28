#!/usr/bin/env python3
"""Pick SLOPE_PRIOR_SD for the Rank-tab graph's local-LINEAR boundary fix, and
measure how much it removes the right-edge lag.

The flaw
--------
The Rank-tab skill line is a 30-day-Gaussian-kernel pooled 4PL MLE that is
local-CONSTANT in time (`poolLogistic` in pwa/js/rank-decomposition.js). At the
RIGHT edge (now) the kernel is one-sided, so the reported "current rank" is a
backward-weighted average that TRAILS a trending user by ~0.8·τ (~24 days). The
fix is to go local-LINEAR in time (`poolLinear`), reporting the intercept of an
a+b·u fit so the lag cancels — with the slope ridge-shrunk (SLOPE_PRIOR_SD) so a
thin/one-sided window degrades back to local-constant instead of extrapolating.

The held-out test (no external ground truth needed)
---------------------------------------------------
At an INTERIOR time t_k the TWO-SIDED kernel (data on both sides of t_k) has no
boundary lag — it is the lag-free reference for the user's skill at t_k. We then
emulate "as-if-now" endpoints: restrict to attempts with t ≤ t_k (exactly what
the real right edge sees) and compute

    past_const(t_k)   — today's behaviour at the edge (local-constant, past-only)
    past_lin(t_k; SD) — the proposed fix (local-linear, past-only), per SD

and score each against ref(t_k) = two-sided local-constant on ALL data:

    signed error = estimate − ref     (lag shows as a NEGATIVE mean for improvers)
    |error|, RMS                       (overall accuracy)
    SE                                 (band half-width — the variance cost)

A good SD drives the signed lag toward 0 (especially on TRENDING stretches) and
the RMS down, without blowing up the SE. As SD→0 the linear fit IS the constant
fit; as SD→∞ it is free (max de-bias, max variance).

Convergence check (flaw #2)
---------------------------
At the TRUE endpoint t1 we also print, per user, the local-constant endpoint, the
local-linear endpoint, and the live front-page window MLE (estimateWindowRank).
The chart endpoint and the headline rank disagree today; the fix should pull the
chart endpoint TOWARD the window MLE.

Reads the local DB (attempts/database.sqlite). No network.

Usage:
  ./.venv/bin/python3 -m auxiliary.eval_rank_endpoint_bias
  ./.venv/bin/python3 -m auxiliary.eval_rank_endpoint_bias --users <redacted-user-id>,<redacted-user-id> --sds 0,1,1.5,2,2.5,3,5
"""

import argparse
import math
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.recreate_rank_graph import (  # noqa: E402
    RankingSystem, build_features, load_rows, Smoother,
    DB_DEFAULT, V2_DIR_DEFAULT, POP_DEFAULT, load_pop_constants, lapse_from,
    TAU_S_MS, DAY_MS, KERNEL_TRUNC_SIGMA, SUPPORT_EFFN, MIN_ATTEMPTS,
    index_to_label, X_CLAMP_LO, X_CLAMP_HI,
)

_sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))

# Production live-rank window MLE constants (rank-decomposition.js).
WIN_CAP = 500
WIN_PRIOR_INFO = 1.0 / (4.0 * 4.0)


def window_mle(sm, lapse):
    """One-shot port of estimateWindowRank (rank-decomposition.js windowMLE): a
    positional linear ramp w=(M−j)/M over the last min(N,500) attempts, 4PL Newton
    warm-started at the recent-tail mean with a weak N(ref,1/WIN_PRIOR_INFO) prior.
    `sm` is the full time-sorted Smoother."""
    n = sm.n
    M = min(n, WIN_CAP)
    lo = max(0, n - WIN_CAP)
    idx = np.arange(n - 1, lo - 1, -1)            # newest first
    w = (M - np.arange(len(idx))) / M             # positional ramp 1 → ~0
    x = sm.x[idx]; sp = sm.slope[idx]; fl = sm.floor[idx]
    ka = sm.kappa[idx]; y = sm.y[idx]; span = sm.span[idx]
    tail = sm.x[max(0, n - 20):]
    ref = float(tail.mean()) if len(tail) else 0.0
    R = ref
    for _ in range(15):
        praw = _sigmoid(sp * (R - x) + ka)
        p = np.clip(fl + span * praw, 1e-9, 1 - 1e-9)
        g = span * sp * praw * (1 - praw)
        denom = p * (1 - p)
        score = WIN_PRIOR_INFO * (ref - R) + float((w * (y - p) * g / denom).sum())
        info = WIN_PRIOR_INFO + float((w * g * g / denom).sum())
        if info < 1e-9:
            break
        step = score / info
        step = 2.0 if step > 2 else (-2.0 if step < -2 else step)
        R += step
        R = X_CLAMP_LO if R < X_CLAMP_LO else (X_CLAMP_HI if R > X_CLAMP_HI else R)
        if abs(step) < 1e-6:
            break
    return R


def eval_points(sm, n_grid=60, future_ratio=1.4):
    """Pick interior times with both past support (effN ≥ SUPPORT_EFFN past-only)
    and genuine future data (two-sided effN ≥ future_ratio× the past-only effN), so
    the two-sided reference is actually two-sided. Returns [(t_k, past_lo_index)]."""
    t0, t1 = float(sm.ts[0]), float(sm.ts[-1])
    if t1 <= t0:
        return []
    out = []
    for t in np.linspace(t0, t1, n_grid):
        k = int(np.searchsorted(sm.ts, t, side="right"))   # # attempts with ts ≤ t
        if k < MIN_ATTEMPTS:
            continue
        _, _, eff_two = sm.at(float(t))                     # two-sided effN
        past = Smoother(sm.ts[:k], sm.x[:k], sm.slope[:k], sm.floor[:k],
                        sm.kappa[:k], sm.y[:k], sm._lapse)
        _, _, eff_past = past.at(float(t))
        if eff_past < SUPPORT_EFFN or eff_two < future_ratio * eff_past:
            continue
        out.append((float(t), past))
    return out


def trend_at(sm, t, span_days=15):
    """Lag-free local slope (stones per 30 days) from the two-sided reference."""
    d = span_days * DAY_MS
    hi, _, _ = sm.at(t + d)
    lo, _, _ = sm.at(t - d)
    if not (math.isfinite(hi) and math.isfinite(lo)):
        return float("nan")
    return (hi - lo) * (30.0 / (2 * span_days))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--dir", default=V2_DIR_DEFAULT)
    ap.add_argument("--pop-constants", default=POP_DEFAULT)
    ap.add_argument("--users", default=None,
                    help="comma list of user-id prefixes (default: a built-in panel)")
    ap.add_argument("--sds", default="0.0,1.0,1.5,2.0,2.5,3.0,5.0",
                    help="SLOPE_PRIOR_SD values to sweep (0 == local-constant)")
    ap.add_argument("--trend-thresh", type=float, default=0.5,
                    help="|stones/30d| above which an eval point is 'trending'")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found — run pipeline.fetch_database first.")

    pc = load_pop_constants(args.pop_constants)
    lapse = lapse_from(pc)
    rs = RankingSystem(args.dir, lapse)
    sds = [float(s) for s in args.sds.split(",")]

    panel = (args.users.split(",") if args.users else
             ["<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>",
              "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>", "<redacted-user-id>"])

    # per-(SD or const) accumulators, split all / trending
    keys = ["const"] + [f"lin{sd:g}" for sd in sds]
    acc = {k: {"err": [], "abs": [], "se": [], "terr": [], "tabs": []} for k in keys}

    print(f"LAPSE={lapse:.4f}  ranking dir={os.path.relpath(args.dir, PROJECT_ROOT)}")
    print(f"trending = |trend| ≥ {args.trend_thresh} stones/30d\n")
    print("Per-user endpoint (t1):  const → lin(SD=2.0) vs live window-MLE  "
          "[internal index, 19=1k]")
    print(f"{'user':<10}{'n':>7}{'const':>8}{'lin2.0':>8}{'winMLE':>8}"
          f"{'lin-const':>10}{'lin→win?':>9}")
    print("-" * 60)

    for uid in panel:
        rows, _surv, _na = load_rows(args.db, uid)
        if not rows:
            continue
        ts, x, slope, floor, kappa, y, size, _nu = build_features(rows, rs)
        if len(ts) < MIN_ATTEMPTS:
            continue
        sm = Smoother(ts, x, slope, floor, kappa, y, lapse)
        sm._lapse = lapse   # stash for eval_points' past-smoother rebuilds

        # ---- endpoint convergence (flaw #2) ----
        t1 = float(sm.ts[-1])
        end_const, _, _ = sm.at(t1, linear=False)
        end_lin, _, _ = sm.at(t1, linear=True, slope_prior_sd=2.0)
        end_win = window_mle(sm, lapse)
        moved_toward = "yes" if abs(end_lin - end_win) < abs(end_const - end_win) else "no"
        print(f"{uid:<10}{sm.n:>7}{end_const:>8.2f}{end_lin:>8.2f}{end_win:>8.2f}"
              f"{end_lin - end_const:>+10.2f}{moved_toward:>9}")

        # ---- interior held-out lag test ----
        for t_k, past in eval_points(sm):
            ref, _, _ = sm.at(t_k, linear=False)            # two-sided, lag-free
            if not math.isfinite(ref):
                continue
            tr = trend_at(sm, t_k)
            is_trend = math.isfinite(tr) and abs(tr) >= args.trend_thresh
            # past-only local-constant (today's edge behaviour)
            c, cse, _ = past.at(t_k, linear=False)
            if math.isfinite(c):
                acc["const"]["err"].append(c - ref)
                acc["const"]["abs"].append(abs(c - ref))
                acc["const"]["se"].append(cse)
                if is_trend:
                    acc["const"]["terr"].append(c - ref)
                    acc["const"]["tabs"].append(abs(c - ref))
            # past-only local-linear, per SD
            for sd in sds:
                k = f"lin{sd:g}"
                if sd == 0.0:
                    val, vse = c, cse                       # SD=0 ≡ local-constant
                else:
                    val, vse, _ = past.at(t_k, linear=True, slope_prior_sd=sd)
                if not math.isfinite(val):
                    continue
                acc[k]["err"].append(val - ref)
                acc[k]["abs"].append(abs(val - ref))
                acc[k]["se"].append(vse)
                if is_trend:
                    acc[k]["terr"].append(val - ref)
                    acc[k]["tabs"].append(abs(val - ref))

    # ----------------------------- summary -----------------------------------
    def stat(v, f):
        v = [a for a in v if math.isfinite(a)]
        return f(v) if v else float("nan")
    rms = lambda v: math.sqrt(sum(a * a for a in v) / len(v))
    mean = lambda v: sum(v) / len(v)

    n_all = len(acc["const"]["err"])
    n_tr = len(acc["const"]["terr"])
    print(f"\nInterior held-out lag test: {n_all} eval points "
          f"({n_tr} trending) pooled across {len(panel)} users")
    print("(signed err <0 ⇒ estimate LAGS below the lag-free reference)\n")
    print(f"{'estimator':<10}{'signed':>9}{'|err|':>8}{'rms':>7}"
          f"{'  | ':>4}{'tr.signed':>10}{'tr.rms':>8}{'  | ':>4}{'meanSE':>8}")
    print("-" * 72)
    for k in keys:
        a = acc[k]
        row = (f"{k:<10}"
               f"{stat(a['err'], mean):>+9.3f}"
               f"{stat(a['abs'], mean):>8.3f}"
               f"{stat(a['err'], rms):>7.3f}"
               f"{'  | ':>4}"
               f"{stat(a['terr'], mean):>+10.3f}"
               f"{stat(a['terr'], rms):>8.3f}"
               f"{'  | ':>4}"
               f"{stat(a['se'], mean):>8.3f}")
        print(row)
    print("\nRead: pick the smallest |tr.signed| & tr.rms whose meanSE hasn't "
          "ballooned. SD=0 row == today's local-constant baseline.")


if __name__ == "__main__":
    main()
