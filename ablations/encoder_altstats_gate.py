"""Gate 1 for the activation-encoder migration: does the 512-dim store move
the availability null?

The shipped alt-stats (lookup.altstats.bin) count, per axis, how many of a
problem's offered alternatives load strictly above/below the answer's loading.
Those loadings come from the per-cell chain: raw activation -> joint-PCA basis
-> trunk -> g head. After the migration the chain's input is the encoded
store, so the question is whether encoder truncation flips any of those strict
comparisons — measured here WITHOUT any refit, by running the SAME fitted
chain twice per cell: once on the raw vector, once on the encoded store's
vector reconstructed back to raw space (y @ C + pca_mean, un-standardized).

Context for the numbers: emit_alt_stats records that merely referencing a
nominally equivalent vector from a different projection moves a count on
~1.8-4.6% of problems, and the migration replaces that entire class of
discrepancy with one pinned source. The gate passes if the encoder's flip rate
sits at or below that order.

Usage:
  ./.venv/bin/python3 -m ablations.encoder_altstats_gate [--n-per-size 1500]
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf
import torch
import torch.nn as nn

from utils.katago.coords import gtp_to_xy, sgf_to_gtp
from utils.katago.sgf_parsing import parse_tsumego_sgf
from utils.rating.activation_encoder import fingerprint, load_encoder
from utils.rating.features import (
    SIZE_FLAG_COLS, corpus_tids_by_size, joint_feature_tail,
    load_joint_activation_basis, load_joint_tail_blocks)
from utils.rating.mirt_net import build_trunk
from utils.rating.model_constants import K_AXES, PCA_DIM, SEED, SIZES

MODEL_PT = os.path.join(PROJECT_ROOT, "ratings", "mirt_model.pt")
ENC_DIR = os.path.join(PROJECT_ROOT, "activations", "enc")
RAW_DIR = os.path.join(PROJECT_ROOT, "activations")


def variation_xys(size, sgf_path):
    """The offered alternatives in SGF order: (x, y) per non-pass root
    variation, answer first (generation pins it)."""
    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    _, variation_nodes = parse_tsumego_sgf(game)
    xys = []
    for node in variation_nodes:
        _, mb = node.get_raw_move()
        if mb is None:
            continue
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is not None:
            xys.append(xy)
    return xys


def rows_for(xys_wanted, xys_all):
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(xys_all)}
    return [idx[xy] for xy in xys_wanted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-size", type=int, default=1500)
    ap.add_argument("--basis-cache", default=None,
                    help="joint-PCA cache npz to load the basis from — pass the "
                         "file the CURRENT mirt_model.pt was trained with (the "
                         "keyed default refits when the corpus has moved, which "
                         "is both heavy and a chain the trunk never saw)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = load_encoder()
    fp = fingerprint(enc)
    dec_C = torch.tensor(enc["C"], dtype=torch.float32, device=device)
    dec_mean = torch.tensor(enc["pca_mean"], dtype=torch.float32, device=device)
    dec_m1 = torch.tensor(enc["m1"], dtype=torch.float32, device=device)
    dec_s1 = torch.tensor(enc["s1"], dtype=torch.float32, device=device)

    tids_by_size = corpus_tids_by_size(SIZES)
    if args.basis_cache:
        with np.load(args.basis_cache) as d:
            basis = {"m1": d["basis_m1"], "s1": d["basis_s1"],
                     "pca_mean": d["basis_pca_mean"], "C": d["basis_C"],
                     "m2": d["basis_m2"], "s2": d["basis_s2"]}
        print(f"basis from {args.basis_cache}")
    else:
        basis = load_joint_activation_basis(SIZES, tids_by_size, PCA_DIM, SEED)
    m1 = torch.tensor(basis["m1"], dtype=torch.float32, device=device)
    s1 = torch.tensor(basis["s1"], dtype=torch.float32, device=device)
    pmean = torch.tensor(basis["pca_mean"], dtype=torch.float32, device=device)
    C = torch.tensor(basis["C"], dtype=torch.float32, device=device)
    m2 = torch.tensor(basis["m2"], dtype=torch.float32, device=device)
    s2 = torch.tensor(basis["s2"], dtype=torch.float32, device=device)

    st = torch.load(MODEL_PT, map_location=device)
    wide, n_wide = load_joint_tail_blocks(SIZES, tids_by_size)
    trunk, d = build_trunk(PCA_DIM + n_wide + SIZE_FLAG_COLS, dropout=0.0)
    trunk = trunk.to(device); trunk.load_state_dict(st["trunk"]); trunk.eval()
    g_lin = nn.Linear(d, K_AXES, bias=False).to(device)
    g_lin.load_state_dict(st["g_lin"])
    g_rms = st["g_rms"].to(device).view(1, -1)

    def loadings(x_raw, key):
        z = ((((x_raw - m1) / s1) - pmean) @ C.T - m2) / s2
        tail = torch.tensor(
            np.tile(joint_feature_tail(key, wide).astype(np.float32)[None, :],
                    (x_raw.shape[0], 1)), device=device)
        with torch.no_grad():
            return (g_lin(trunk(torch.cat([z, tail], dim=1))) / g_rms).cpu().numpy()

    n_prob = flip_prob = n_skip = 0
    n_cmp = flip_cmp = 0
    corr_num = corr_den = 0.0
    all_margins, flip_margins = [], []
    for size in SIZES:
        tids = rng.choice(tids_by_size[size],
                          size=min(args.n_per_size, len(tids_by_size[size])),
                          replace=False)
        for tid in tids:
            if any((size, tid) not in block for _, block in wide):
                # Corpus has moved since the caches were built; a gate sample
                # only needs coverage, not completeness.
                n_skip += 1
                continue
            xys = variation_xys(size, os.path.join(
                PROJECT_ROOT, "tsumegos", str(size), tid + ".sgf"))
            if len(xys) < 2:
                continue
            with np.load(os.path.join(RAW_DIR, str(size), tid + ".npz")) as z:
                rows = rows_for(xys, np.asarray(z["xys"]))
                x_raw = torch.tensor(np.asarray(z["candidates"])[rows],
                                     dtype=torch.float32, device=device)
            with np.load(os.path.join(ENC_DIR, str(size), tid + ".npz")) as z:
                assert int(z["encoder_fp"]) == fp
                y = torch.tensor(np.asarray(z["candidates"])[rows],
                                 dtype=torch.float32, device=device)
            x_rec = ((y @ dec_C) + dec_mean) * dec_s1 + dec_m1

            g_a = loadings(x_raw, (size, tid))       # (n_var, K), answer row 0
            g_b = loadings(x_rec, (size, tid))
            corr_num += float((g_a * g_b).sum())
            corr_den += float(np.sqrt((g_a ** 2).sum() * (g_b ** 2).sum()))

            marg_a = g_a[1:] - g_a[0]                # signed margin vs answer
            sign_a = np.sign(marg_a)                 # strict above/below
            sign_b = np.sign(g_b[1:] - g_b[0])
            flipped = sign_a != sign_b
            flips = int(flipped.sum())
            n_cmp += sign_a.size; flip_cmp += flips
            n_prob += 1; flip_prob += flips > 0
            all_margins.append(np.abs(marg_a).ravel())
            if flips:
                flip_margins.append(np.abs(marg_a)[flipped].ravel())

    print(f"problems: {n_prob}; with >=1 flipped comparison: {flip_prob} "
          f"({100 * flip_prob / n_prob:.2f}%)")
    print(f"strict comparisons: {n_cmp}; flipped: {flip_cmp} "
          f"({100 * flip_cmp / n_cmp:.3f}%)")
    print(f"loading cosine (pooled): {corr_num / corr_den:.5f}")
    am = np.concatenate(all_margins); fm = np.concatenate(flip_margins)
    print(f"median |margin|: all {np.median(am):.4f}, flipped {np.median(fm):.4f}")
    for q in (5, 10, 25):
        thr = np.percentile(am, q)
        print(f"  flips with |margin| below the all-comparison p{q} "
              f"({thr:.4f}): {100 * (fm < thr).mean():.1f}%")


if __name__ == "__main__":
    main()
