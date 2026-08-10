#!/usr/bin/env python3
"""How many axis directions does the between-user residual actually support?

The production K search answers this by FITTING each candidate count (minutes
per candidate, and — as auxiliary.rating_dev.eval_axis_seeding measures — with a
cold-init lottery whose seed spread is twice the margin the stop rule turns on).
This script answers the same question without fitting anything, by asking how
many directions of between-user residual variation clear the noise floor.

Construction. Under the production baseline (the 4PL forward model as a fixed
offset, plus a per-user intercept and rank slope — the SAME baseline the axis
selection scores against), each user's residual score vector is

    s_u = Σ_p (y_up - p̂_up) · x_p

the gradient the axis block would feel at that user if an axis pointed along
x. Between-user axis structure is exactly a low-rank component in the second
moment Σ_u s_u s_uᵀ, so its eigenvalue spectrum is the quantity of interest and
the number of eigenvalues clearing the null band is the effective rank.

Two details are load-bearing, and getting either wrong moves the answer by tens
of directions.

THE SCORES ARE CENTRED ACROSS USERS. The uncentred moment is dominated by
E[s_u] — the direction in which the baseline mispredicts for EVERYONE — which
is population residual difficulty, not a skill. The production model quarantines
exactly that in its own head (and pulls the axis positions to mean zero) because
the population signal runs far larger than the between-user variance, so an
uncentred spectrum measures mostly the quantity the axes are forbidden to carry.

THE NULL REFITS THE BASELINE. The null band is a PARAMETRIC BOOTSTRAP —
resample each outcome from its own fitted baseline probability, which destroys
axis structure while preserving every attempt's difficulty and every user's
problem set. But the observed residuals have had a per-user ability, intercept
and rank slope fitted IN SAMPLE, which removes variance the null would keep, so
a null that skips the refit is inflated along exactly those directions (visibly:
its leading eigenvalue lands ABOVE the observed one). Each replicate therefore
re-runs the whole per-user baseline on its simulated outcomes. A within-user
permutation is not a substitute — it destroys the problem-difficulty association
too, inflating the null for a second reason and understating the rank.

Two scalings are reported because neither is obviously right and they answer
slightly different questions: UNWEIGHTED lets a heavy user dominate the moment
in proportion to their attempt count (which is also how much they inform the
axes), while ROOT-N normalizes each user's score to a common scale so a
direction has to be shared across users rather than carried by a few.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_residual_spectrum --boots 5
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import (  # noqa: E402
    FLOOR_MAX, CALIBRATION_P, MIN_USER_ATTEMPTS, _fit_theta, _irls_off, _p4pl,
    build_axis_features, load_axis_attempts,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402


def user_baseline(pre, y, grid):
    """Production-form 4PL offset + per-user intercept and rank slope, fit on
    `y`. Returns fitted probabilities under the axis-free baseline. Called on
    the real outcomes AND on every bootstrap replicate, so the null carries the
    same in-sample fitting the observed residuals do."""
    R0, A, C, span, kap, X = pre
    th = _fit_theta(R0, A, C, span, kap, y, grid)
    P4 = _p4pl(th, R0, A, C, span, kap)
    off = np.log(P4 / (1 - P4))
    w = _irls_off(X, y, off, ridge=1e-6)
    z = np.clip(off + X @ w, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def spectra(scores_by_user, n_by_user):
    """(unweighted, root-n) eigenvalue spectra of the CENTRED score second
    moment — centring is what restricts this to between-user structure, the
    only kind an axis is allowed to carry."""
    S = np.stack(scores_by_user)
    n = np.array(n_by_user, dtype=float)
    out = []
    for Sx in (S, S / np.sqrt(n)[:, None]):
        Sc = Sx - Sx.mean(axis=0, keepdims=True)
        out.append(np.linalg.eigvalsh(Sc.T @ Sc / Sc.shape[0])[::-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boots", type=int, default=5,
                    help="parametric-bootstrap replicates for the null band")
    ap.add_argument("--show", type=int, default=25,
                    help="leading eigenvalues to print")
    args = ap.parse_args()

    t0 = time.time()
    feats, _scalars, rank, params, _ad = build_axis_features()
    tids_all = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids_all)}
    F = np.stack([feats[t] for t in tids_all])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float64)
    lapse = load_client_lapse(0.0328)

    rows_by_user = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in tid_index:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
    print(f"loaded in {time.time() - t0:.0f}s: {len(tids_all)} problems, "
          f"{len(users)} users, feature dim {F.shape[1]}", flush=True)

    # per-user constants, reused by the observed pass and every replicate
    pre_by_user, idx_by_user, ns = {}, {}, []
    for u in users:
        a = rows_by_user[u]
        tids = [t for t, _ in a]
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        rk = np.array([rank[t] for t in tids])
        pre_by_user[u] = (R0, A, C, span, kap,
                          np.column_stack([np.ones(len(tids)), rk]))
        idx_by_user[u] = np.array([tid_index[t] for t in tids])
        ns.append(len(a))

    def scores(y_by_user):
        out = []
        for u in users:
            p = user_baseline(pre_by_user[u], y_by_user[u], grid)
            out.append(F[idx_by_user[u]].T @ (y_by_user[u] - p))
        return out

    y_obs = {u: np.array([s for _, s in rows_by_user[u]], dtype=float)
             for u in users}
    ev_u, ev_r = spectra(scores(y_obs), ns)
    print(f"observed spectra computed ({time.time() - t0:.0f}s)", flush=True)

    # the null resamples from the OBSERVED fit, then refits on the simulation
    p_obs = {u: user_baseline(pre_by_user[u], y_obs[u], grid) for u in users}
    rng = np.random.default_rng(0)
    null_u, null_r = [], []
    for b in range(args.boots):
        y_sim = {u: rng.binomial(1, p_obs[u]).astype(float) for u in users}
        a, c = spectra(scores(y_sim), ns)
        null_u.append(a)
        null_r.append(c)
        print(f"  bootstrap {b + 1}/{args.boots} ({time.time() - t0:.0f}s)",
              flush=True)
    null_u = np.max(np.stack(null_u), axis=0)   # max over replicates = the band
    null_r = np.max(np.stack(null_r), axis=0)

    for name, ev, nl in (("UNWEIGHTED", ev_u, null_u), ("ROOT-N", ev_r, null_r)):
        above = ev > nl
        n_above = int(np.sum(above))
        # A raw count is the wrong quantity to size a fit from: it says how many
        # directions are detectable, not where the structure lives. The excess
        # over the null is what an axis block can actually capture, so report how
        # fast it accumulates — that is what says whether starting wide pays.
        excess = np.maximum(ev - nl, 0.0)
        share = np.cumsum(excess) / (excess.sum() + 1e-30)
        print(f"\n===== {name} =====")
        print(f"  eigenvalues clearing the null band: {n_above}")
        for k in (3, 5, 8, 12, 16, 24):
            if k <= len(share):
                print(f"    top {k:2d} directions hold {share[k - 1] * 100:5.1f}%"
                      " of the above-null excess")
        print("   i   observed        null(max)     ratio")
        for i in range(min(args.show, len(ev))):
            print(f"  {i:2d}   {ev[i]:12.4g}  {nl[i]:12.4g}  "
                  f"{ev[i] / (nl[i] + 1e-30):8.2f}"
                  + ("   <- above" if above[i] else ""))

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
