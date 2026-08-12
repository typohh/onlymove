#!/usr/bin/env python3
"""PAIRED comparison of two rating models on survey rho, with a user bootstrap.

Two models scored on the same declarations produce two numbers, and the gap
between them is routinely quoted without an interval — which is unsafe here,
because both are averages of per-server Spearmans over 184 users, and the
differences at stake are ~0.001-0.007. An unpaired eyeball cannot tell those from
noise.

The comparison is PAIRED at the user level: both models rate the SAME users from
the SAME declarations, so resampling users (not declarations) and recomputing
both models on each replicate cancels the shared sampling variation. That is the
whole reason a 0.001 difference can be resolvable at all — the two models' errors
on a given user are strongly correlated, so the DIFFERENCE has far less variance
than either rho.

Scope: BASE rho only. The time-adjusted figure cross-fits its own alpha inside
`_time_adjusted_rho`, so an honest bootstrap of it would refit alpha within every
replicate — around 20s each, hours for a useful number. The base difference is
reported with an interval; the time-adjusted difference is reported as a point
estimate and must be quoted as such.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.compare_survey_rho \\
      --joint /tmp/joint_params_whole_s0.npz --prod-dir /tmp/prod_ho
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MIN_ATTEMPTS = 20
MIN_SERVER = 4


def _user_ratings(ratemap, slopemap, surveys, attempts, sizes):
    """Each survey user's MLE 80%-clear rating under one model's parameters."""
    from utils.rating.fit_rank_curves import fit_user_rating
    out = {}
    for uid in surveys["user_id"].unique():
        pooled = []
        for s in sizes:
            pooled.extend(attempts.get((uid[:8], s), []))
        if len(pooled) < MIN_ATTEMPTS:
            continue
        R, nj = fit_user_rating(pooled, ratemap, min_n=MIN_ATTEMPTS, slopes=slopemap)
        if R is not None and nj >= MIN_ATTEMPTS:
            out[uid[:8]] = R
    return out


def _weighted_rho(by_server):
    """Per-server Spearman, averaged with user-count weights — the shipped
    definition. Servers below MIN_SERVER pairs are dropped, as there."""
    num = den = 0.0
    for pairs in by_server.values():
        if len(pairs) < MIN_SERVER:
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
            continue
        r = float(np.corrcoef(np.argsort(np.argsort(x)),
                              np.argsort(np.argsort(y)))[0, 1])
        if np.isfinite(r):
            num += r * len(pairs)
            den += len(pairs)
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True, help="npz from --dump-params")
    ap.add_argument("--prod-dir", required=True, help="a ratings/ directory")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from auxiliary.rating_dev.validate_ratings import load_ratings, SIZES
    from utils.rating.rank_surveys import (
        load_rank_surveys, latest_per_user_server, load_attempts_for_users,
        SQL_PATH_DEFAULT)

    z = np.load(args.joint, allow_pickle=True)
    joint_rate = {str(t): float(r) for t, r in zip(z["tid"], z["rating"])}
    joint_slope = {str(t): float(a) for t, a in zip(z["tid"], z["slope"])}
    prod_rate, prod_slope, _fl, _lam, _th = load_ratings(args.prod_dir)
    print(f"joint {len(joint_rate)} problems | production {len(prod_rate)}")

    surveys = latest_per_user_server(load_rank_surveys(SQL_PATH_DEFAULT))
    attempts = load_attempts_for_users(SQL_PATH_DEFAULT,
                                       set(surveys["user_id"].unique()),
                                       first_time_only=True)
    ur_j = _user_ratings(joint_rate, joint_slope, surveys, attempts, SIZES)
    ur_p = _user_ratings(prod_rate, prod_slope, surveys, attempts, SIZES)
    # Restrict to users BOTH models rate, so the comparison is exactly paired —
    # a user scored by only one model would otherwise shift that model's rho for
    # a reason unrelated to its quality.
    common = sorted(set(ur_j) & set(ur_p))
    print(f"users rated by joint {len(ur_j)}, production {len(ur_p)}, "
          f"both {len(common)}")

    rows = []
    for _, row in surveys.iterrows():
        g = row["rank_value"]
        pref = str(row["user_id"])[:8]
        if g is None or (isinstance(g, float) and np.isnan(g)):
            continue
        if pref in ur_j and pref in ur_p:
            rows.append((pref, row["server"], float(g)))
    per_user = defaultdict(list)
    for pref, server, g in rows:
        per_user[pref].append((server, g))
    print(f"{len(rows)} declarations over {len(per_user)} users")

    def rho_for(users, table):
        by_server = defaultdict(list)
        for u in users:
            for server, g in per_user[u]:
                by_server[server].append((table[u], g))
        return _weighted_rho(by_server)

    users = sorted(per_user)
    rj, rp = rho_for(users, ur_j), rho_for(users, ur_p)
    print(f"\nbase rho   joint {rj:.4f}   production {rp:.4f}   "
          f"delta {rj - rp:+.4f}  (joint minus production)")

    rng = np.random.default_rng(args.seed)
    d = []
    for _ in range(args.boot):
        samp = [users[i] for i in rng.integers(0, len(users), len(users))]
        a, b = rho_for(samp, ur_j), rho_for(samp, ur_p)
        if np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_joint_better = float((d > 0).mean())
    print(f"paired user bootstrap ({len(d)} replicates)")
    print(f"  delta mean {d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(joint > production) = {p_joint_better:.3f}")
    verdict = ("production better" if hi < 0 else
               "joint better" if lo > 0 else
               "INDISTINGUISHABLE — the interval spans zero")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    main()
