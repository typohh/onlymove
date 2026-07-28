"""Walk-forward evaluation of the client weakness model's training history:
window length and recency weighting.

The client (pwa/js/weakness.js) trains an UNWEIGHTED ridge logistic on the last
MODEL_INPUT_HISTORY_LENGTH=500 novel embedded attempts. The recall model and the
front-page rank window both weight recent attempts linearly; this script asks
whether the weakness model should too, and what window length the population
data actually supports.

Design (mirrors pipeline.fit_population_constants' offline sum-form replica —
X = [1, problemRank, embed_0..K-1], one coefficient per axis = the identified
sum beta_correct+beta_picked of the client's augmented model):

1. WALK-FORWARD PREDICTION (the headline): per user, at every STRIDE-th attempt
   index t0 >= WARMUP, fit each config on the trailing window before t0 and
   score the held-out log-loss of attempts t0..t0+STRIDE-1. All configs score
   the SAME prediction points (paired), including a rank-only null (intercept +
   problemRank, no axes) that measures whether the axes add predictive value at
   all. Weights are mean-normalized so the data/prior balance is identical
   across configs (only the recency TILT varies).

2. DRIFT DIAGNOSTIC: for users with >= 2*HALF_MIN usable attempts, fit the
   axis coefficients on (a) first half vs last half (temporal split) and
   (b) even vs odd attempts (interleaved split). Both splits have the same
   sampling noise; only the temporal split sees drift. The gap between the two
   split-half correlations is the drift signal — if weaknesses were stable,
   the two would match.

Read-only over attempts/database.sqlite + embeddings/joint/per_problem.csv +
ratings/. No GPU. Run: ./.venv/bin/python3 -m auxiliary.eval_weakness_history

RESULTS 2026-07-20 (659 users >=60 usable novel attempts, 230k predicted
attempts; deltas in mnats/attempt, paired bootstrap over users):
- The axes add real predictive value: unw500 (the shipped config) beats the
  rank-only null by +1.45 [+1.08, +1.89].
- Window: 500 sits on the plateau. 125/250 are worse (-2.99 / +0.35 vs null);
  1000/2000 unweighted are NOT better than 500 (+0.16 n.s. / +0.00).
- LINEAR positional recency (the recall/rank models' scheme) HURTS:
  lin500 vs unw500 = -1.10 [-1.45, -0.77]. Do not adopt it for alignment.
- Mild EXPONENTIAL decay over a LONGER window wins: exp half-life 300 over a
  1000-attempt window = +0.56 [+0.34, +0.75] vs unw500; the half-life plateau
  is broad (300/450/600 all +0.5-ish; window 2000 adds nothing). Decay and the
  long window only work together: unw1000 alone +0.16 n.s., exp300 truncated
  to the 500 window only +0.18 [+0.05, +0.31].
- Drift diagnostic corroborates: axes 0/1/4 show real temporal drift
  (interleaved split-half r exceeds temporal r by ~0.14-0.17); axes 2/3 are
  stable. Weaknesses drift moderately -> gentle decay, not a linear ramp.
- An earlier run with GUARD_LAM=1e-3 showed lin500 at -7 mnats; that was a
  separation artifact (four users' degenerate early windows pushed the MAP
  intercept to |beta|~30, single ~50-nat catastrophes). Guard kept at 0.05.
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from pipeline.fit_population_constants import (  # noqa: E402
    _weakness_k_embed, load_problem_embeddings,
    load_problem_params, load_rating_to_rank, load_rows)

WARMUP = 40          # first predicted index (>= client's MIN_TRAINING_ATTEMPTS=20)
STRIDE = 10          # refit every STRIDE attempts, predict the next STRIDE
MIN_EVAL_N = 60      # user must have this many usable novel attempts to enter the eval
HALF_MIN = 120       # per-half minimum for the drift diagnostic (user needs 2x)
GUARD_LAM = 0.05     # ridge on intercept + rank (prior sd ~4.5 logits — a separation
                     # guard, identical across configs; at 1e-3 the MAP fit let
                     # degenerate early windows push the intercept to |30|, and a
                     # handful of ~50-nat single-prediction catastrophes dominated
                     # the config deltas. The client's Adam-from-warm-start never
                     # reaches such params, so the strong guard is the more
                     # faithful offline stand-in.)
BOOT_REPS = 2000
SEED = 0


def weight_fn(kind, half_life=None):
    """Per-row weights over a length-n training window (oldest..newest),
    mean-normalized to 1 so the effective data mass vs the prior is constant."""
    def w(n):
        if kind == "unw":
            return np.ones(n)
        if kind == "lin":
            v = np.arange(1, n + 1, dtype=float)
        elif kind == "exp":
            v = 0.5 ** ((n - 1 - np.arange(n, dtype=float)) / half_life)
        else:
            raise ValueError(kind)
        return v / v.mean()
    return w


# (name, window, weightfn, use_axes)
def build_configs(lam_axis):
    K = len(lam_axis)
    cfgs = [
        ("rank-only", 500, weight_fn("unw"), False),
        ("unw125",    125, weight_fn("unw"), True),
        ("unw250",    250, weight_fn("unw"), True),
        ("unw500",    500, weight_fn("unw"), True),   # current client
        ("unw1000",  1000, weight_fn("unw"), True),
        ("unw2000",  2000, weight_fn("unw"), True),
        ("lin250",    250, weight_fn("lin"), True),
        ("lin500",    500, weight_fn("lin"), True),   # aligned with recall/rank models
        ("lin1000",  1000, weight_fn("lin"), True),
        ("exp150",   1000, weight_fn("exp", 150), True),
        ("exp300",   1000, weight_fn("exp", 300), True),
        # Truncated at the client's existing 500-entry embedding-retention
        # window — the variants implementable with no storage-policy change.
        ("exp150w500", 500, weight_fn("exp", 150), True),
        ("exp300w500", 500, weight_fn("exp", 300), True),
    ]
    lam_full = np.concatenate([[GUARD_LAM, GUARD_LAM], np.asarray(lam_axis, float)])
    lam_rank = np.array([GUARD_LAM, GUARD_LAM])
    return cfgs, lam_full, lam_rank


def fit_logistic(X, y, w, lam_vec, beta0):
    """Damped Newton for the weighted ridge logistic. Deterministic, warm-started."""
    beta = beta0.copy()
    for _ in range(60):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        g = X.T @ (w * (p - y)) + lam_vec * beta
        Wd = w * np.clip(p * (1 - p), 1e-9, None)
        H = (X * Wd[:, None]).T @ X
        H[np.diag_indices_from(H)] += lam_vec + 1e-9
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, -5.0, 5.0)   # separation guard
        beta = beta - step
        if not np.all(np.isfinite(beta)):
            return beta0.copy()
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


def loglik(X, y, beta):
    z = np.clip(X @ beta, -35, 35)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(np.where(y > 0.5, p, 1 - p))


def load_user_designs():
    """{user: (X_full, y)} usable novel attempts in time order.
    X_full = [1, problemRank, embed_0..K-1]."""
    rows = load_rows(os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"))
    needed = {(r["size"], r["problem"]) for r in rows}
    params = load_problem_params(os.path.join(PROJECT_ROOT, "ratings"), needed)
    apply_m = load_rating_to_rank(os.path.join(PROJECT_ROOT, "ratings"))
    if apply_m is None:
        raise SystemExit("no ratings/rating_to_rank.json")
    embeddings = load_problem_embeddings(needed)
    if not embeddings:
        raise SystemExit("no embeddings/joint/per_problem.csv")
    rank_feat = {k: apply_m(v[0]) for k, v in params.items()}

    by_user = defaultdict(list)
    for r in rows:
        if r["is_repeat"]:
            continue
        key = (r["size"], r["problem"])
        if key not in rank_feat or key not in embeddings:
            continue
        by_user[r["user_id"]].append(
            (datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
             1.0 if r["success"] else 0.0, rank_feat[key], embeddings[key]))

    out = {}
    for user, ev in by_user.items():
        ev.sort(key=lambda e: e[0])
        y = np.array([e[1] for e in ev])
        X = np.column_stack([np.ones(len(ev)),
                             np.array([e[2] for e in ev]),
                             np.vstack([e[3] for e in ev])])
        out[user] = (X, y)
    return out


def walk_forward(designs, cfgs, lam_full, lam_rank):
    """Per config: per-user summed held-out LL over the shared prediction points.
    Returns (per_user_ll {name: {user: ll}}, per_user_n {user: n_pred})."""
    per_user_ll = {name: {} for name, *_ in cfgs}
    per_user_n = {}
    users = [u for u, (X, y) in designs.items() if len(y) >= MIN_EVAL_N]
    for ui, user in enumerate(sorted(users)):
        X, y = designs[user]
        n = len(y)
        warm = {name: np.zeros(X.shape[1] if ax else 2)
                for name, _, _, ax in cfgs}
        ll_sum = {name: 0.0 for name, *_ in cfgs}
        n_pred = 0
        for t0 in range(WARMUP, n, STRIDE):
            hi = min(t0 + STRIDE, n)
            for name, window, wfn, use_axes in cfgs:
                lo = max(0, t0 - window)
                Xt = X[lo:t0] if use_axes else X[lo:t0, :2]
                Xp = X[t0:hi] if use_axes else X[t0:hi, :2]
                w = wfn(t0 - lo)
                lam = lam_full if use_axes else lam_rank
                beta = fit_logistic(Xt, y[lo:t0], w, lam, warm[name])
                warm[name] = beta
                ll_sum[name] += float(np.sum(loglik(Xp, y[t0:hi], beta)))
            n_pred += hi - t0
        for name, *_ in cfgs:
            per_user_ll[name][user] = ll_sum[name]
        per_user_n[user] = n_pred
        if (ui + 1) % 100 == 0:
            print(f"  walk-forward: {ui + 1}/{len(users)} users", flush=True)
    return per_user_ll, per_user_n


def paired_delta(per_user_ll, per_user_n, a, b, rng):
    """Mean (LL_a - LL_b) in mnats/attempt with a paired bootstrap CI over users."""
    users = sorted(per_user_ll[a])
    d = np.array([per_user_ll[a][u] - per_user_ll[b][u] for u in users])
    n = np.array([per_user_n[u] for u in users], float)
    point = 1000.0 * d.sum() / n.sum()
    idx = rng.integers(0, len(users), size=(BOOT_REPS, len(users)))
    boot = 1000.0 * d[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


def drift_diagnostic(designs, lam_full):
    """Split-half axis-coefficient correlations: temporal (first vs last half)
    vs interleaved (even vs odd). Same noise, only the temporal split sees
    drift."""
    K = len(lam_full) - 2
    temporal, interleaved = [], []
    for user, (X, y) in designs.items():
        n = len(y)
        if n < 2 * HALF_MIN:
            continue
        halves = {
            "temporal": (np.arange(n) < n // 2, np.arange(n) >= n - n // 2),
            "interleaved": (np.arange(n) % 2 == 0, np.arange(n) % 2 == 1),
        }
        fits = {}
        ok = True
        for split, (m1, m2) in halves.items():
            bs = []
            for m in (m1, m2):
                if y[m].std() < 1e-3:
                    ok = False
                    break
                b = fit_logistic(X[m], y[m], np.ones(int(m.sum())), lam_full,
                                 np.zeros(X.shape[1]))
                bs.append(b[2:])
            if not ok:
                break
            fits[split] = bs
        if not ok:
            continue
        temporal.append(np.concatenate([fits["temporal"][0], fits["temporal"][1]]))
        interleaved.append(np.concatenate([fits["interleaved"][0], fits["interleaved"][1]]))
    if len(temporal) < 10:
        return None
    T = np.array(temporal)   # (U, 2K): first-half betas ++ second-half betas
    I = np.array(interleaved)
    out = {"n_users": len(temporal), "per_axis": []}
    for k in range(K):
        rt = np.corrcoef(T[:, k], T[:, K + k])[0, 1]
        ri = np.corrcoef(I[:, k], I[:, K + k])[0, 1]
        out["per_axis"].append((rt, ri))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    rng = np.random.default_rng(SEED)

    import json
    pc = json.load(open(os.path.join(PROJECT_ROOT, "ratings", "population_constants.json")))
    lam_axis = pc["weakness_axis_lambda"]
    K = _weakness_k_embed()
    assert len(lam_axis) == K, (len(lam_axis), K)
    cfgs, lam_full, lam_rank = build_configs(lam_axis)

    print("Loading designs ...", flush=True)
    designs = load_user_designs()
    usable = {u: len(y) for u, (X, y) in designs.items()}
    n_eval = sum(1 for v in usable.values() if v >= MIN_EVAL_N)
    print(f"  {len(designs)} users with any usable attempt; "
          f"{sum(1 for v in usable.values() if v >= 20)} with >=20 (model active); "
          f"{n_eval} with >={MIN_EVAL_N} (in eval); "
          f"{sum(v for v in usable.values() if v >= MIN_EVAL_N)} attempts in eval cohort")

    print("Walk-forward ...", flush=True)
    per_user_ll, per_user_n = walk_forward(designs, cfgs, lam_full, lam_rank)
    total_n = sum(per_user_n.values())
    print(f"\n{total_n} predicted attempts, {len(per_user_n)} users")
    print(f"{'config':<12}{'LL/att (mnats)':>16}{'delta vs rank-only':>22}{'95% CI':>22}")
    base = "rank-only"
    for name, *_ in cfgs:
        ll = 1000.0 * sum(per_user_ll[name].values()) / total_n
        if name == base:
            print(f"{name:<12}{ll:>16.1f}{'—':>22}{'—':>22}")
        else:
            d, lo, hi = paired_delta(per_user_ll, per_user_n, name, base, rng)
            print(f"{name:<12}{ll:>16.1f}{d:>+22.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>22}")
    print("\nHead-to-head (mnats/attempt, paired bootstrap):")
    for a, b in [("lin500", "unw500"), ("exp150", "unw500"), ("exp300", "unw500"),
                 ("exp300w500", "unw500"), ("exp150w500", "unw500"),
                 ("exp300w500", "exp300"), ("unw250", "unw500"),
                 ("unw1000", "unw500"), ("lin1000", "unw500")]:
        d, lo, hi = paired_delta(per_user_ll, per_user_n, a, b, rng)
        print(f"  {a} vs {b}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    print("\nDrift diagnostic (split-half corr of axis coefficients):", flush=True)
    dd = drift_diagnostic(designs, lam_full)
    if dd is None:
        print("  too few users")
    else:
        print(f"  {dd['n_users']} users with >= {2 * HALF_MIN} attempts")
        print(f"  {'axis':<6}{'temporal r':>12}{'interleaved r':>15}{'drift gap':>12}")
        for k, (rt, ri) in enumerate(dd["per_axis"]):
            print(f"  {k:<6}{rt:>12.3f}{ri:>15.3f}{ri - rt:>12.3f}")


if __name__ == "__main__":
    main()
