#!/usr/bin/env python3
"""Split-half reliability of the tendency reading: arm A vs arm B projection.

tail_conditioning_sensitivity shows the within-problem sign structure both
tendency halves are built from is strongly conditioning-dependent — which says
the two projections DISAGREE, not which one is right. Neither has ground truth
(no supervision exists at alternative cells), so this asks the data the only
askable way: which projection's per-user readings are more STABLE?

Method: for every user with enough recorded wrong first-time attempts, split
their errors into odd/even halves by their own error sequence (interleaved, so
drift and serving-era effects land in both halves). Per half, per axis, per
arm, compute exactly the shipped tendency statistic (motif-stats.js): each
error contributes the sign of quantized correct-minus-picked and its problem's
availability-null toward-probability; the half's reading is the
Poisson-binomial z = (nOver - sumP) / sqrt(sumPQ). Then correlate z_odd vs
z_even across users, per axis, for each arm, on the users valid in BOTH arms
and both halves — fully paired, so the comparison is the projection and
nothing else. A bootstrap over users gives the CI on the per-axis difference.

Arm A is always the SHIPPED fit on its full input. Arm B is one of two things:

  default — the SAME fit with the tail's wide columns dropped to 0 (their
  pooled mean) and the surviving activation block scaled by 1/(1-INPUT_DROPOUT),
  the training-side presentation of survivors beside dropped columns
  (--no-input-scale for the eval-convention variant). This asks which
  CONDITIONING of one model reads more stably.

  --arm-b-model CKPT --arm-b-features BLOCKS — a DIFFERENT fit, read on its own
  full input: its trunk and axis head fed the activation PCA plus exactly the
  per-problem blocks it was trained on, nothing masked and nothing rescaled.
  This asks which MODEL reads more stably, so a feature ablation that shows no
  loss on the rating can be scored on the readings it exists to produce. The
  two fits' axes correspond BY INDEX, which holds only because both were taught
  the same rule bank in the same order — there is no post-hoc matching here.

Everything after the projection is identical between the modes: the same users,
the same halves, the same statistic, the same pairing.

CAVEAT: reliability rewards stability, not truth. A distortion that is stable
within a user — carried, say, by the rank band they are served — replicates
across interleaved halves too. A clear win here is evidence for an arm, not
proof; a tie says the arms are observationally equivalent at current volume.
"""

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import not_ignored_sql  # noqa: E402
from utils.katago.coords import gtp_to_xy  # noqa: E402
from utils.rating.model_constants import INPUT_DROPOUT, K_AXES, SIZES  # noqa: E402
from utils.rating.fitted_projection import (  # noqa: E402
    FEATURE_BLOCKS, FittedProjection, ShippedProjection, problem_arrays,
    quantize, variation_npz_rows)

# The shipped gate: a half's reading exists only on this many directional
# errors (MIN_ERRORS in pwa/js/motif-stats.js).
MIN_ERRORS_HALF = 10


def load_user_errors(min_errors):
    """{(size, tid): [(user_idx, half, x, y)]} plus the user count — every
    recorded wrong first-time attempt of every user with at least min_errors
    of them, halved odd/even along the user's own error sequence."""
    db = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
    if not os.path.exists(db):
        raise SystemExit(f"missing {db} — run pipeline.fetch_database first")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT user_id, size, problem, move FROM tsumego_reports "
            f"WHERE success = 0 AND is_repeat = 0 AND move IS NOT NULL AND move != '' AND {not_ignored_sql()} "
            "ORDER BY user_id, created_at, id").fetchall()
    finally:
        con.close()
    by_user = {}
    for uid, size, tid, move in rows:
        xy = gtp_to_xy(move)
        if xy is None:
            continue
        by_user.setdefault(uid, []).append((int(size), str(tid),
                                            int(xy[0]), int(xy[1])))
    by_problem, n_users = {}, 0
    for uid in sorted(by_user):
        errs = by_user[uid]
        if len(errs) < min_errors:
            continue
        ui = n_users
        n_users += 1
        for seq, (size, tid, x, y) in enumerate(errs):
            by_problem.setdefault((size, tid), []).append((ui, seq % 2, x, y))
    return by_problem, n_users


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def spearman(a, b):
    """Rank correlation — the guard against a few tail users driving the
    Pearson figure."""
    rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
    return pearson(rank(np.asarray(a)), rank(np.asarray(b)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-errors", type=int, default=2 * MIN_ERRORS_HALF,
                    help="errors a user needs to enter at all (halved between "
                         "the two splits)")
    ap.add_argument("--no-input-scale", action="store_true",
                    help="masked arm at eval convention: surviving columns at "
                         "scale 1 instead of 1/(1-INPUT_DROPOUT). Meaningless "
                         "with --arm-b-model, which masks nothing")
    ap.add_argument("--arm-b-model", metavar="CKPT",
                    help="make arm B ANOTHER FIT read on its own full input "
                         "(e.g. dev_runs/actbot_s3.pt) instead of the masked "
                         "shipped fit. Axes are paired by index")
    ap.add_argument("--arm-b-features", choices=FEATURE_BLOCKS,
                    help="the per-problem block layout --arm-b-model was "
                         "FITTED on; it is not recoverable from the file")
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap resamples over users for the Δr CI")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if bool(args.arm_b_model) != bool(args.arm_b_features):
        ap.error("--arm-b-model and --arm-b-features go together")

    import torch
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = ShippedProjection(device)
    # Arm B: another fit on its own input, or the same fit with the tail masked.
    Q = (FittedProjection(device, args.arm_b_model, args.arm_b_features)
         if args.arm_b_model else None)
    bname = "dev" if Q is not None else "mask"
    act_scale = 1.0 if args.no_input_scale else 1.0 / (1.0 - INPUT_DROPOUT)

    by_problem, n_users = load_user_errors(args.min_errors)
    corpus = {(s, t) for s in SIZES for t in P.tids_by_size[s]}
    todo = sorted(k for k in by_problem if k in corpus)
    n_att = sum(len(v) for k, v in by_problem.items() if k in corpus)
    arm_b = (f"{args.arm_b_model} on its own full input "
             f"(blocks={args.arm_b_features}, fdim {Q.fdim})" if Q is not None
             else f"the shipped fit with the tail masked, activation scale "
                  f"{act_scale:.4f}")
    print(f"{n_users} users >= {args.min_errors} errors, {n_att} attempts over "
          f"{len(todo)} problems, device {device}\n"
          f"arm A (full): the shipped fit on its own full input "
          f"(fdim {P.fdim})\narm B ({bname}): {arm_b}", flush=True)

    K = K_AXES
    # (arm, user, half, axis) accumulators of the shipped statistic.
    shape = (2, n_users, 2, K)
    nOver = np.zeros(shape)
    nDir = np.zeros(shape)
    sumP = np.zeros(shape)
    sumPQ = np.zeros(shape)
    n_skip_problem, n_unresolved = 0, 0

    for done, (size, tid) in enumerate(todo, 1):
        cand, xys, correct_idx = problem_arrays(size, tid)
        var_rows = variation_npz_rows(size, tid, xys)
        if var_rows is None or correct_idx not in var_rows or len(var_rows) < 2:
            n_skip_problem += 1
            continue
        xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
        row_to_local = {r: i for i, r in enumerate(var_rows)}

        rows = np.array(var_rows, dtype=np.intp)
        Z = P.pca(cand[rows])
        tail_full = P.tail(size, tid)
        n = len(rows)
        inp_a = np.concatenate([Z, np.tile(tail_full, (n, 1))], axis=1)
        if Q is None:
            tail_mask = tail_full.copy()
            tail_mask[:P.n_wide] = 0.0         # size flag kept
            inp_b = np.concatenate([Z * act_scale,
                                    np.tile(tail_mask, (n, 1))], axis=1)
            # One forward pass over both arms: same fit, so this is also what
            # keeps the default reproducible against the committed figures.
            g = P.axes(np.concatenate([inp_a, inp_b], axis=0).astype(np.float32))
            q = (quantize(g[:n]), quantize(g[n:]))
        else:
            # Arm B's own basis, its own tail blocks, its own head. The basis is
            # the same joint PCA (same corpus, PCA_DIM, SEED), recomputed rather
            # than assumed.
            inp_b = np.concatenate([Q.pca(cand[rows]),
                                    np.tile(Q.tail(size, tid), (n, 1))], axis=1)
            q = (quantize(P.axes(inp_a.astype(np.float32))),
                 quantize(Q.axes(inp_b.astype(np.float32))))
        ci = var_rows.index(correct_idx)

        # Per arm: each alternative cell's sign vs correct, and the problem's
        # null toward-probability per axis (strict comparisons only, exactly
        # computeAltStats + the motif-stats accumulation).
        arm_sign, arm_p = [], []
        for a in range(2):
            s = np.sign(q[a][ci][None, :] - q[a])          # d = correct - cell
            above = (s < 0).sum(axis=0)                    # cell above correct
            below = (s > 0).sum(axis=0)
            with np.errstate(invalid="ignore"):
                p = np.where(above + below > 0,
                             above / np.maximum(above + below, 1), np.nan)
            arm_sign.append(s)
            arm_p.append(p)

        for ui, half, x, y in by_problem[(size, tid)]:
            li = row_to_local.get(xy_to_row.get((x, y)))
            if li is None:
                n_unresolved += 1
                continue
            for a in range(2):
                s, p = arm_sign[a][li], arm_p[a]
                for k in range(K):
                    if s[k] == 0 or not np.isfinite(p[k]):
                        continue
                    pk = float(p[k])
                    nDir[a, ui, half, k] += 1
                    sumP[a, ui, half, k] += pk
                    sumPQ[a, ui, half, k] += pk * (1.0 - pk)
                    if s[k] < 0:                           # erred TOWARD
                        nOver[a, ui, half, k] += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    names = [str(c) for c in json.load(open(labels_path))["components"]] \
        if os.path.exists(labels_path) else [f"axis{i}" for i in range(K)]
    print(f"\nproblems skipped: {n_skip_problem}; attempts with no variation "
          f"cell: {n_unresolved}", flush=True)

    rng = np.random.default_rng(args.seed)
    # Pearson on z mixes lean with VOLUME (z grows with sqrt(n) and per-user n
    # varies widely, so heavy users dominate the covariance); the rank
    # correlation is the primary metric, Pearson kept for context.
    print(f"\n{'axis':<24}{'users':>6}{'r_full':>8}{'r_' + bname:>8}{'dr':>8}"
          f"{'dr 95% CI':>18}{'rho_full':>9}{'rho_' + bname:>9}{'drho':>8}"
          f"{'drho 95% CI':>18}")
    valid = (nDir >= MIN_ERRORS_HALF) & (sumPQ > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (nOver - sumP) / np.sqrt(np.where(sumPQ > 0, sumPQ, np.nan))
    dr_all = []
    for k in range(K):
        ok = valid[:, :, :, k].all(axis=(0, 2))            # both arms, both halves
        # Two-step indexing: combining the boolean with the integer k in one
        # subscript is "advanced indices separated by a slice", which moves the
        # user axis to the front.
        zk = z[..., k][:, ok, :]                           # (arm, users, half)
        nu = int(ok.sum())
        if nu < 3:
            print(f"{names[k][:23]:<24}{nu:>6}  (too few users)")
            continue
        rA = pearson(zk[0, :, 0], zk[0, :, 1])
        rB = pearson(zk[1, :, 0], zk[1, :, 1])
        sA = spearman(zk[0, :, 0], zk[0, :, 1])
        sB = spearman(zk[1, :, 0], zk[1, :, 1])
        boots = np.empty(args.boot)
        sboots = np.empty(args.boot)
        for b in range(args.boot):
            idx = rng.integers(0, nu, size=nu)
            boots[b] = (pearson(zk[1, idx, 0], zk[1, idx, 1])
                        - pearson(zk[0, idx, 0], zk[0, idx, 1]))
            sboots[b] = (spearman(zk[1, idx, 0], zk[1, idx, 1])
                         - spearman(zk[0, idx, 0], zk[0, idx, 1]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        slo, shi = np.percentile(sboots, [2.5, 97.5])
        dr_all.append(sB - sA)
        print(f"{names[k][:23]:<24}{nu:>6}{rA:>8.3f}{rB:>8.3f}{rB - rA:>8.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{sA:>9.3f}{sB:>9.3f}"
              f"{sB - sA:>8.3f}{f'[{slo:+.3f}, {shi:+.3f}]':>18}", flush=True)
    if dr_all:
        print(f"\nmean drho ({bname} - full) across axes: {np.mean(dr_all):+.3f}")
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
