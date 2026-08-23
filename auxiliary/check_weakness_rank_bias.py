#!/usr/bin/env python3
"""Is the weakness pane's per-axis centre rank-dependent?

`fit_population_constants` ships ONE scalar centre per axis for each identified
direction — `weakness_axis_mean` for the SUM, `weakness_contrast_mean` for the
CONTRAST — and the client subtracts it to decide whether a user is weak/strong
and under/overplaying. `_weakness_user_datasets` discards the user's rank, so
those centres are rank-independent by construction.

That is only sound if the population coefficient does not move with rank. Motif
PREVALENCE demonstrably does (tenuki fires on a few percent of the corpus at the
weak end and a third of it at the strong end), so this script measures the
consequence directly: fit each user's weakness model with the SAME design,
prior and ridge the client uses, then ask whether the fitted sum and contrast
trend with the user's own rank — and what share of each rank band the shipped
scalar centre flags.

A slope indistinguishable from zero means the scalar centre is fine. A slope
that carries the population across the flag gate between rank bands means the
pane is reporting the user's RANK, not their tendency.

Usage:  ./.venv/bin/python3 -m auxiliary.check_weakness_rank_bias [--csv out.csv]
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline.fit_population_constants import (          # noqa: E402
    DB_SQLITE, FLOOR_MAX, WEAKNESS_EB_MAX_N, WEAKNESS_EB_MIN_N,
    load_rows, load_problem_params, load_problem_embeddings,
    load_picked_embeddings, load_rating_to_rank, fit_ability, sigmoid,
)

# The client's flag gate (pwa/js/constants.js MOTIF_Z).
MOTIF_Z = 1.0


def fit_user(X, y, lam_vec, prior_mean, iters=60):
    """Newton on the client's penalized logistic. Returns (beta, cov) or None."""
    beta = prior_mean.copy()
    for _ in range(iters):
        z = X @ beta
        p = sigmoid(z)
        g = X.T @ (p - y) + lam_vec * (beta - prior_mean)
        W = p * (1 - p)
        H = (X.T * W) @ X + np.diag(lam_vec)
        try:
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    z = X @ beta
    p = sigmoid(z)
    H = (X.T * (p * (1 - p))) @ X + np.diag(lam_vec)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    return beta, cov


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_SQLITE)
    ap.add_argument("--ratings-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--constants", default=os.path.join(PROJECT_ROOT, "ratings",
                                                        "population_constants.json"))
    ap.add_argument("--csv", default=None, help="write the per-user fits here")
    args = ap.parse_args()

    consts = json.load(open(args.constants))
    lam = np.array(consts["weakness_axis_lambda"], float)
    sum_mean = np.array(consts["weakness_axis_mean"], float)
    con_mean = np.array(consts["weakness_contrast_mean"], float)
    K = len(lam)
    tau = np.sqrt(2.0 / lam)

    print(f"Loading attempts from {args.db} ...")
    rows = load_rows(args.db)
    needed = {(r["size"], r["problem"]) for r in rows}
    params = load_problem_params(args.ratings_dir, needed)
    curves = load_rating_to_rank(args.ratings_dir)
    apply_m = curves[0] if isinstance(curves, tuple) else curves
    embeddings = load_problem_embeddings(needed)
    picked = load_picked_embeddings(rows)
    print(f"  {len(rows)} rows, {len(params)} rated problems, "
          f"{len(embeddings)} embedded, {len(picked)} picked cells")

    by_user = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)

    rank_feat = {k: apply_m(v[0]) for k, v in params.items()}

    # The client's prior: slot means (sum +- contrast)/2, so BOTH identified
    # directions sit on their shipped centre.
    numP = 2 + 2 * K
    lam_vec = np.full(numP, 1e-6)
    prior_mean = np.zeros(numP)
    for k in range(K):
        lam_vec[2 + k] = lam_vec[2 + K + k] = lam[k]
        prior_mean[2 + k] = (sum_mean[k] + con_mean[k]) / 2
        prior_mean[2 + K + k] = (sum_mean[k] - con_mean[k]) / 2

    recs = []
    for user, urows in by_user.items():
        ev = []
        for r in urows:
            if r["is_repeat"]:
                continue
            key = (r["size"], r["problem"])
            if key not in rank_feat or key not in embeddings:
                continue
            corr = embeddings[key]
            pick = corr
            if not r["success"] and r["move"] is not None:
                pick = picked.get((int(r["size"]), r["problem"], r["move"]), corr)
            ev.append((datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")),
                       1.0 if r["success"] else 0.0, rank_feat[key], corr, pick))
        if len(ev) < WEAKNESS_EB_MIN_N:
            continue
        ev.sort(key=lambda e: e[0])
        ev = ev[-WEAKNESS_EB_MAX_N:]
        y = np.array([e[1] for e in ev], float)
        if y.std() < 1e-3:
            continue
        X = np.column_stack([np.ones(len(ev)),
                             np.array([e[2] for e in ev], float),
                             np.vstack([e[3] for e in ev]),
                             np.vstack([e[4] for e in ev])])
        # The user's own rank, from all their novel rated attempts.
        av = [(params[(r["size"], r["problem"])], r["success"]) for r in urows
              if not r["is_repeat"] and (r["size"], r["problem"]) in params]
        if len(av) < 10:
            continue
        th = fit_ability(np.array([a[0][0] for a in av]),
                         np.array([a[0][1] for a in av]),
                         np.clip(np.array([a[0][2] for a in av]), 0, FLOOR_MAX),
                         np.array([1.0 if a[1] else 0.0 for a in av]))
        fit = fit_user(X, y, lam_vec, prior_mean)
        if fit is None:
            continue
        beta, cov = fit
        rec = {"user": user, "rank": float(apply_m(th)), "n": len(ev)}
        for k in range(K):
            bc, bp = beta[2 + k], beta[2 + K + k]
            vc, vp = cov[2 + k, 2 + k], cov[2 + K + k, 2 + K + k]
            cv = cov[2 + k, 2 + K + k]
            rec[f"sum{k}"] = bc + bp - sum_mean[k]
            rec[f"con{k}"] = bc - bp - con_mean[k]
            rec[f"sum_se{k}"] = np.sqrt(max(vc + vp + 2 * cv, 1e-12))
            rec[f"con_se{k}"] = np.sqrt(max(vc + vp - 2 * cv, 1e-12))
        recs.append(rec)

    print(f"\nFitted {len(recs)} users with >= {WEAKNESS_EB_MIN_N} embedded novel attempts\n")
    if len(recs) < 30:
        print("Too few users to say anything.")
        return

    rank = np.array([r["rank"] for r in recs])
    names = consts.get("_axis_names") or [f"axis{k}" for k in range(K)]
    meta = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "embedding_meta.json")
    if os.path.exists(meta):
        comps = json.load(open(meta)).get("labels", {}).get("components", [])
        names = [(c.get("positive") or f"axis{k}") for k, c in enumerate(comps)] or names

    # Rank bands, wide enough to hold users.
    edges = [-np.inf, -8, -5, -3, -1, 1, np.inf]
    band_names = ["<8k", "8k-5k", "5k-3k", "3k-1k", "1k-1d", "1d+"]

    for label, pre in (("CONTRAST (underplay <-> overplay)", "con"),
                       ("SUM (weak <-> strong)", "sum")):
        print(f"=== {label} — centred on the shipped scalar; 0 = the average player ===")
        print(f"{'axis':16s} {'slope/rank':>12s} {'+-SE':>8s} {'t':>6s}   "
              + "".join(f"{b:>9s}" for b in band_names))
        for k in range(K):
            v = np.array([r[f"{pre}{k}"] for r in recs])
            A = np.column_stack([np.ones(len(rank)), rank])
            coef, *_ = np.linalg.lstsq(A, v, rcond=None)
            resid = v - A @ coef
            s2 = resid @ resid / max(len(rank) - 2, 1)
            se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
            means = []
            for i in range(len(edges) - 1):
                m = (rank >= edges[i]) & (rank < edges[i + 1])
                means.append(f"{v[m].mean():+9.3f}" if m.sum() >= 5 else f"{'  n<5':>9s}")
            print(f"{names[k][:16]:16s} {coef[1]:+12.4f} {se:8.4f} "
                  f"{coef[1]/se if se > 0 else 0:6.1f}   " + "".join(means))
        print()

    # The actionable number: what share of each band the scalar centre FLAGS.
    print("=== Share of each rank band FLAGGED by the shipped scalar centre "
          f"(|z| > {MOTIF_Z}) ===")
    print("(a rank-independent centre should flag a similar share in every band)")
    for k in range(K):
        v = np.array([r[f"con{k}"] for r in recs])
        se = np.array([r[f"con_se{k}"] for r in recs])
        z = np.divide(v, se, out=np.zeros_like(v), where=se > 0)
        print(f"\n  {names[k]}  (contrast)")
        print(f"    {'band':>8s} {'n':>5s} {'underplay':>11s} {'overplay':>10s}")
        for i in range(len(edges) - 1):
            m = (rank >= edges[i]) & (rank < edges[i + 1])
            if m.sum() < 5:
                continue
            under = (z[m] < -MOTIF_Z).mean()
            over = (z[m] > MOTIF_Z).mean()
            print(f"    {band_names[i]:>8s} {m.sum():5d} {under:10.0%} {over:9.0%}")

    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
