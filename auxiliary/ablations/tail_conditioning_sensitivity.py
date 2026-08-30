#!/usr/bin/env python3
"""How tail-conditioned are the per-cell axis readings' within-problem signs?

The per-cell projection (pipeline.write_mirt_per_cell) feeds every legal cell
the SAME per-problem tail (bot winrate, move evals, ownership, phase — all
describing the problem via its answer) beside that cell's own activations. The
client's tendency test and its availability null read only WITHIN-PROBLEM sign
structure referenced to the correct cell, so a tail that acted as a monotone
contraction toward the correct cell's reading would cancel out of both. The
trunk is nonlinear, so nothing guarantees that; this measures it.

Method: for a seeded problem sample, project each problem's SGF root-variation
cells (the exact cell set the shipped null is computed over) through the fitted
trunk twice —

  full arm:    the shipped input (activation PCA + real tail + size flag)
  masked arm:  the tail's wide columns at 0 (their pooled mean — every column
               is z-scored), size flag kept. The tail's drop rate goes from
               INPUT_DROPOUT in training to 1 here — past the eval convention,
               not toward it — so the surviving dropout-region columns (the
               activation block) are scaled by 1/(1-INPUT_DROPOUT), presenting
               them the way training presented survivors beside dropped
               columns. --no-input-scale measures the eval-convention variant
               (all surviving columns at scale 1) instead.

--population picked swaps the uniform cell population for the wrong moves
players actually recorded (db/d1.sqlite, weighted by attempt counts) — the
population the tendency test's observed side runs on — and weights each
problem's null shift by how often it was really failed.

and compare, per axis, at the client's own resolution (fp16 storage cast +
quantizeEmbedding's round-to-0.01):

  pairs        alternative cells with a strict sign vs the correct cell in
               BOTH arms
  flip%        of those, how many changed side — the tendency test's observed
               direction and the null's counts both live on this sign
  tie-ch%      cells strict in exactly one arm (quantization ties made/broken)
  mean/p90|dp| per-problem shift of the null toward-probability
               above/(above+below)
  |d|mask/full unquantized within-problem delta magnitude ratio (the tail's
               net amplification/attenuation of the contrast the serve-side
               weakness score reads)
  corr-shift   mean |Δ| of the correct cell's own reading (level context)

Low flip% would vindicate the shipped null against the tail-amalgamation
concern; high flip% means the within-problem structure is substantially a
property of the conditioning, not of the cells alone.
"""

import argparse
import json
import os
import sqlite3
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago.coords import gtp_to_xy  # noqa: E402
from utils.rating.model_constants import INPUT_DROPOUT, K_AXES, SIZES  # noqa: E402
from auxiliary.ablations._projection import (  # noqa: E402
    ShippedProjection, problem_arrays, quantize, variation_npz_rows)


def load_picked_errors():
    """{(size, tid): [(x, y, n_attempts)]} — recorded wrong first-time attempts'
    move coordinates from the local DB copy, GTP-decoded. This is the population
    the tendency test's OBSERVED side actually runs on, so in --population
    picked the flip statistics weight each (problem, cell) by how often real
    players picked it."""
    db = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
    if not os.path.exists(db):
        raise SystemExit(f"missing {db} — run pipeline.fetch_database first")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT size, problem, move, COUNT(*) FROM tsumego_reports "
            "WHERE success = 0 AND is_repeat = 0 AND move IS NOT NULL AND move != '' "
            "GROUP BY size, problem, move").fetchall()
    finally:
        con.close()
    out = {}
    for size, tid, move, cnt in rows:
        xy = gtp_to_xy(move)
        if xy is None:
            continue
        out.setdefault((int(size), str(tid)), []).append(
            (int(xy[0]), int(xy[1]), int(cnt)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-per-size", type=int, default=1000,
                    help="problems sampled per board (variations mode only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-input-scale", action="store_true",
                    help="masked arm at eval convention: surviving columns at "
                         "scale 1 instead of 1/(1-INPUT_DROPOUT)")
    ap.add_argument("--population", choices=("variations", "picked"),
                    default="variations",
                    help="variations: uniform over sampled problems' variation "
                         "cells (the null's own population). picked: every "
                         "problem with recorded wrong attempts, observed-side "
                         "cells weighted by real error counts from the DB — "
                         "the population the tendency test's observed "
                         "direction runs on")
    args = ap.parse_args()

    import torch
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = ShippedProjection(device)
    tids_by_size = P.tids_by_size
    act_scale = 1.0 if args.no_input_scale else 1.0 / (1.0 - INPUT_DROPOUT)

    picks = None
    if args.population == "picked":
        picks = load_picked_errors()
        corpus = {(s, t) for s in SIZES for t in tids_by_size[s]}
        sample = sorted(k for k in picks if k in corpus)
        print(f"{len(sample)} failed problems from the DB "
              f"({sum(c for v in picks.values() for *_xy, c in v)} wrong "
              f"attempts), device {device}, masked-arm activation scale "
              f"{act_scale:.4f}", flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        sample = []
        for size in SIZES:
            tids = tids_by_size[size]
            pick = rng.choice(len(tids), size=min(args.n_per_size, len(tids)),
                              replace=False)
            sample += [(size, tids[i]) for i in sorted(pick)]
        print(f"{len(sample)} problems sampled (seed {args.seed}), device "
              f"{device}, masked-arm activation scale {act_scale:.4f}",
              flush=True)

    K = K_AXES
    n_pairs = np.zeros(K)
    n_flip = np.zeros(K)
    n_strict_any = np.zeros(K)
    n_tie_change = np.zeros(K)
    p_pairs = [[] for _ in range(K)]
    absd_full = np.zeros(K)
    absd_mask = np.zeros(K)
    corr_shift = np.zeros(K)
    n_corr = 0
    n_skip = 0
    n_outside, n_at_correct, n_no_obs = 0, 0, 0

    for done, (size, tid) in enumerate(sample, 1):
        cand, xys, correct_idx = problem_arrays(size, tid)
        var_rows = variation_npz_rows(size, tid, xys)
        if var_rows is None or correct_idx not in var_rows or len(var_rows) < 2:
            n_skip += 1
            continue
        xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}

        rows = np.array(var_rows, dtype=np.intp)
        Z = P.pca(cand[rows])
        tail_full = P.tail(size, tid)
        tail_mask = tail_full.copy()
        tail_mask[:P.n_wide] = 0.0             # size flag kept
        n = len(rows)
        inp = np.concatenate([
            np.concatenate([Z, np.tile(tail_full, (n, 1))], axis=1),
            np.concatenate([Z * act_scale, np.tile(tail_mask, (n, 1))], axis=1),
        ], axis=0).astype(np.float32)
        g = P.axes(inp)
        gA, gB = g[:n], g[n:]
        ci = var_rows.index(correct_idx)
        qA, qB = quantize(gA), quantize(gB)

        # Observed-side cells: the whole variation set (uniform), or the
        # really-picked wrong cells weighted by their attempt counts.
        if picks is None:
            obs_idx = [i for i in range(n) if i != ci]
            obs_w = np.ones(len(obs_idx))
        else:
            row_to_local = {r: i for i, r in enumerate(var_rows)}
            obs_idx, wlist = [], []
            for x, y, cnt in picks[(size, tid)]:
                li = row_to_local.get(xy_to_row.get((x, y)))
                if li is None:
                    n_outside += cnt       # no embeddable variation cell —
                    continue               # the client gets no vector either
                if li == ci:
                    n_at_correct += cnt    # failed deeper in the tree; the
                    continue               # sign test drops it as a tie
                obs_idx.append(li)
                wlist.append(cnt)
            if not obs_idx:
                n_no_obs += 1
                continue
            obs_w = np.array(wlist, dtype=np.float64)

        corr_shift += np.abs(gA[ci] - gB[ci])
        n_corr += 1
        wcol = obs_w[:, None]
        dA = qA[ci][None, :] - qA[obs_idx]
        dB = qB[ci][None, :] - qB[obs_idx]
        sA, sB = np.sign(dA), np.sign(dB)
        strictA, strictB = sA != 0, sB != 0
        n_pairs += ((strictA & strictB) * wcol).sum(axis=0)
        n_flip += (((sA * sB) < 0) * wcol).sum(axis=0)
        n_strict_any += ((strictA | strictB) * wcol).sum(axis=0)
        n_tie_change += ((strictA ^ strictB) * wcol).sum(axis=0)
        absd_full += (np.abs(gA[ci][None, :] - gA[obs_idx]) * wcol).sum(axis=0)
        absd_mask += (np.abs(gB[ci][None, :] - gB[obs_idx]) * wcol).sum(axis=0)
        # The null stays over the problem's whole variation set (that IS the
        # null), entering once per observed error — matching the shipped
        # Poisson-binomial, where each error contributes its problem's p.
        nul = [i for i in range(n) if i != ci]
        nA = np.sign(qA[ci][None, :] - qA[nul])
        nB = np.sign(qB[ci][None, :] - qB[nul])
        # One problem one vote in the uniform draw; in picked mode a problem
        # votes once per real error, matching the Poisson-binomial's sum of p.
        w_prob = float(obs_w.sum()) if picks is not None else 1.0
        for k in range(K):
            a_above = int((nA[:, k] < 0).sum())
            a_below = int((nA[:, k] > 0).sum())
            b_above = int((nB[:, k] < 0).sum())
            b_below = int((nB[:, k] > 0).sum())
            if a_above + a_below > 0 and b_above + b_below > 0:
                p_pairs[k].append((a_above / (a_above + a_below),
                                   b_above / (b_above + b_below), w_prob))
        if done % 400 == 0:
            print(f"  {done}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    names = [str(c) for c in json.load(open(labels_path))["components"]] \
        if os.path.exists(labels_path) else [f"axis{i}" for i in range(K)]

    extra = (f", {n_outside} picked outside the variation set, "
             f"{n_at_correct} at the correct cell, {n_no_obs} problems with "
             f"no usable pick" if picks is not None else "")
    print(f"\nproblems used: {n_corr}  (skipped: {n_skip}{extra})", flush=True)
    print(f"\n{'axis':<24}{'pairs':>8}{'flip%':>8}{'tie-ch%':>9}"
          f"{'mean|dp|':>10}{'p90|dp|':>9}{'|d|mask/full':>13}{'corr-shift':>11}")
    for k in range(K):
        dp = np.repeat(np.array([abs(a - b) for a, b, _ in p_pairs[k]]),
                       np.array([int(w) for *_a, w in p_pairs[k]], dtype=np.int64))
        ratio = absd_mask[k] / max(absd_full[k], 1e-12)
        print(f"{names[k][:23]:<24}{int(n_pairs[k]):>8}"
              f"{100.0 * n_flip[k] / max(n_pairs[k], 1):>8.2f}"
              f"{100.0 * n_tie_change[k] / max(n_strict_any[k], 1):>9.2f}"
              f"{dp.mean() if len(dp) else float('nan'):>10.4f}"
              f"{np.percentile(dp, 90) if len(dp) else float('nan'):>9.4f}"
              f"{ratio:>13.3f}"
              f"{corr_shift[k] / max(n_corr, 1):>11.3f}", flush=True)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
