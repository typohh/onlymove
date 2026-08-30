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
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf  # noqa: E402

from utils.katago.coords import gtp_to_xy, sgf_to_gtp  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf  # noqa: E402
from utils.rating.mirt_net import build_trunk  # noqa: E402
from utils.rating.model_constants import (  # noqa: E402
    INPUT_DROPOUT, K_AXES, PCA_DIM, SEED, SIZES)
from utils.rating.features import (  # noqa: E402
    SIZE_FLAG_COLS, corpus_tids_by_size, joint_feature_tail,
    load_joint_activation_basis, load_joint_tail_blocks)

MODEL_PT = os.path.join(PROJECT_ROOT, "ratings", "mirt_model.pt")


def quantize(a):
    """Client resolution: fp16 storage cast, then quantizeEmbedding's 0.01 rounding."""
    return np.round(a.astype(np.float16).astype(np.float64) * 100.0) / 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-per-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-input-scale", action="store_true",
                    help="masked arm at eval convention: surviving columns at "
                         "scale 1 instead of 1/(1-INPUT_DROPOUT)")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PT):
        raise SystemExit(f"no {MODEL_PT} — run pipeline.train_mirt first")
    tids_by_size = corpus_tids_by_size(SIZES)
    basis = load_joint_activation_basis(SIZES, tids_by_size, PCA_DIM, SEED)
    m1, s1 = basis["m1"], basis["s1"]
    C, pmean = basis["C"], basis["pca_mean"]
    m2, s2 = basis["m2"], basis["s2"]

    st = torch.load(MODEL_PT, map_location=device)
    wide, n_wide = load_joint_tail_blocks(SIZES, tids_by_size)
    fdim = PCA_DIM + n_wide + SIZE_FLAG_COLS
    trunk, d = build_trunk(fdim, dropout=0.0)
    trunk = trunk.to(device)
    trunk.load_state_dict(st["trunk"])
    trunk.eval()
    g_lin = nn.Linear(d, K_AXES, bias=False).to(device)
    g_lin.load_state_dict(st["g_lin"])
    g_rms = st["g_rms"].to(device).view(1, -1)
    act_scale = 1.0 if args.no_input_scale else 1.0 / (1.0 - INPUT_DROPOUT)

    rng = np.random.default_rng(args.seed)
    sample = []
    for size in SIZES:
        tids = tids_by_size[size]
        pick = rng.choice(len(tids), size=min(args.n_per_size, len(tids)),
                          replace=False)
        sample += [(size, tids[i]) for i in sorted(pick)]
    print(f"{len(sample)} problems sampled (seed {args.seed}), device {device}, "
          f"masked-arm activation scale {act_scale:.4f}", flush=True)

    K = K_AXES
    n_pairs = np.zeros(K, dtype=np.int64)
    n_flip = np.zeros(K, dtype=np.int64)
    n_strict_any = np.zeros(K, dtype=np.int64)
    n_tie_change = np.zeros(K, dtype=np.int64)
    p_pairs = [[] for _ in range(K)]
    absd_full = np.zeros(K)
    absd_mask = np.zeros(K)
    corr_shift = np.zeros(K)
    n_corr = 0
    n_skip_parse, n_skip_cells = 0, 0

    with torch.no_grad():
        for done, (size, tid) in enumerate(sample, 1):
            sgf_path = os.path.join(PROJECT_ROOT, "tsumegos", str(size), tid + ".sgf")
            npz_path = os.path.join(PROJECT_ROOT, "activations", str(size), tid + ".npz")
            try:
                with open(sgf_path, "rb") as f:
                    game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
                _, variation_nodes = parse_tsumego_sgf(game)
            except Exception:
                n_skip_parse += 1
                continue
            with np.load(npz_path, allow_pickle=False) as dnpz:
                cand = np.asarray(dnpz["candidates"], dtype=np.float32)
                xys = np.asarray(dnpz["xys"], dtype=np.int64)
                correct_idx = int(dnpz["correct_idx"])
            xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
            var_rows = []
            ok = True
            for node in variation_nodes:
                _, mb = node.get_raw_move()
                xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
                if xy is None:
                    continue                       # pass — never embedded
                r = xy_to_row.get((int(xy[0]), int(xy[1])))
                if r is None:
                    ok = False
                    break
                var_rows.append(r)
            if not ok or correct_idx not in var_rows or len(var_rows) < 2:
                n_skip_cells += 1
                continue

            rows = np.array(var_rows, dtype=np.intp)
            Z = ((((cand[rows] - m1) / s1 - pmean) @ C.T - m2) / s2)
            tail_full = joint_feature_tail((size, tid), wide).astype(np.float32)
            tail_mask = tail_full.copy()
            tail_mask[:n_wide] = 0.0               # size flag kept
            n = len(rows)
            inp = np.concatenate([
                np.concatenate([Z, np.tile(tail_full, (n, 1))], axis=1),
                np.concatenate([Z * act_scale, np.tile(tail_mask, (n, 1))], axis=1),
            ], axis=0)
            g = g_lin(trunk(torch.from_numpy(inp).to(device))) / g_rms
            g = g.cpu().numpy()
            gA, gB = g[:n], g[n:]
            ci = var_rows.index(correct_idx)
            qA, qB = quantize(gA), quantize(gB)

            corr_shift += np.abs(gA[ci] - gB[ci])
            n_corr += 1
            alt = [i for i in range(n) if i != ci]
            dA = qA[ci][None, :] - qA[alt]
            dB = qB[ci][None, :] - qB[alt]
            sA, sB = np.sign(dA), np.sign(dB)
            strictA, strictB = sA != 0, sB != 0
            both = strictA & strictB
            n_pairs += both.sum(axis=0)
            n_flip += ((sA * sB) < 0).sum(axis=0)
            n_strict_any += (strictA | strictB).sum(axis=0)
            n_tie_change += (strictA ^ strictB).sum(axis=0)
            absd_full += np.abs(gA[ci][None, :] - gA[alt]).sum(axis=0)
            absd_mask += np.abs(gB[ci][None, :] - gB[alt]).sum(axis=0)
            for k in range(K):
                a_above = int((sA[:, k] < 0).sum())
                a_below = int((sA[:, k] > 0).sum())
                b_above = int((sB[:, k] < 0).sum())
                b_below = int((sB[:, k] > 0).sum())
                if a_above + a_below > 0 and b_above + b_below > 0:
                    p_pairs[k].append((a_above / (a_above + a_below),
                                       b_above / (b_above + b_below)))
            if done % 400 == 0:
                print(f"  {done}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    names = [str(c) for c in json.load(open(labels_path))["components"]] \
        if os.path.exists(labels_path) else [f"axis{i}" for i in range(K)]

    print(f"\nproblems used: {n_corr}  (skipped: {n_skip_parse} parse, "
          f"{n_skip_cells} cell-map)", flush=True)
    print(f"\n{'axis':<24}{'pairs':>8}{'flip%':>8}{'tie-ch%':>9}"
          f"{'mean|dp|':>10}{'p90|dp|':>9}{'|d|mask/full':>13}{'corr-shift':>11}")
    for k in range(K):
        dp = np.array([abs(a - b) for a, b in p_pairs[k]])
        ratio = absd_mask[k] / max(absd_full[k], 1e-12)
        print(f"{names[k][:23]:<24}{n_pairs[k]:>8}"
              f"{100.0 * n_flip[k] / max(n_pairs[k], 1):>8.2f}"
              f"{100.0 * n_tie_change[k] / max(n_strict_any[k], 1):>9.2f}"
              f"{dp.mean() if len(dp) else float('nan'):>10.4f}"
              f"{np.percentile(dp, 90) if len(dp) else float('nan'):>9.4f}"
              f"{ratio:>13.3f}"
              f"{corr_shift[k] / max(n_corr, 1):>11.3f}", flush=True)
    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
