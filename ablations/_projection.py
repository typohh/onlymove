"""Shared loading for the ablation scripts: the shipped fit and its per-cell
projection, bundled so every script here projects EXACTLY the shipped
computation. The model and the feature assembly stay single-sourced in
utils.rating.mirt_net / utils.rating.features — this module only gathers the
loading idiom (the same one pipeline.write_mirt_per_cell uses) so sibling
scripts cannot restate it differently.
"""

import os

import numpy as np
import sgfmill.sgf

from utils.katago.coords import gtp_to_xy, sgf_to_gtp
from utils.katago.sgf_parsing import parse_tsumego_sgf
from utils.rating.mirt_net import build_trunk
from utils.rating.model_constants import K_AXES, PCA_DIM, SEED, SIZES
from utils.rating.features import (
    SIZE_FLAG_COLS, corpus_tids_by_size, joint_feature_tail,
    load_joint_activation_basis, load_joint_tail_blocks)
from utils.paths import PROJECT_ROOT

MODEL_PT = os.path.join(PROJECT_ROOT, "ratings", "mirt_model.pt")


def quantize(a):
    """Client resolution: fp16 storage cast, then quantizeEmbedding's 0.01 rounding."""
    return np.round(a.astype(np.float16).astype(np.float64) * 100.0) / 100.0


class ShippedProjection:
    """The fitted trunk + axis head and everything needed to feed it: the
    joint-PCA basis, the per-problem tail blocks, the corpus listing."""

    def __init__(self, device):
        import torch
        import torch.nn as nn
        if not os.path.exists(MODEL_PT):
            raise SystemExit(f"no {MODEL_PT} — run pipeline.train_mirt first")
        self._torch = torch
        self.device = device
        self.tids_by_size = corpus_tids_by_size(SIZES)
        basis = load_joint_activation_basis(SIZES, self.tids_by_size, PCA_DIM, SEED)
        self.m1, self.s1 = basis["m1"], basis["s1"]
        self.C, self.pmean = basis["C"], basis["pca_mean"]
        self.m2, self.s2 = basis["m2"], basis["s2"]
        st = torch.load(MODEL_PT, map_location=device)
        self.wide, self.n_wide = load_joint_tail_blocks(SIZES, self.tids_by_size)
        fdim = PCA_DIM + self.n_wide + SIZE_FLAG_COLS
        trunk, d = build_trunk(fdim, dropout=0.0)
        self.trunk = trunk.to(device)
        self.trunk.load_state_dict(st["trunk"])
        self.trunk.eval()
        self.g_lin = nn.Linear(d, K_AXES, bias=False).to(device)
        self.g_lin.load_state_dict(st["g_lin"])
        self.g_rms = st["g_rms"].to(device).view(1, -1)

    def tail(self, size, tid):
        return joint_feature_tail((size, tid), self.wide).astype(np.float32)

    def pca(self, cand_rows):
        """Stored activation rows -> the trunk's activation block."""
        return (((cand_rows - self.m1) / self.s1 - self.pmean) @ self.C.T
                - self.m2) / self.s2

    def axes(self, inp):
        """(n, fdim) float32 -> (n, K_AXES) numpy, the g head's output."""
        torch = self._torch
        with torch.no_grad():
            g = self.g_lin(self.trunk(torch.from_numpy(inp).to(self.device)))
            return (g / self.g_rms).cpu().numpy()


def problem_arrays(size, tid):
    """(candidates float32, xys int64, correct_idx) from the activation store."""
    npz_path = os.path.join(PROJECT_ROOT, "activations", str(size), tid + ".npz")
    with np.load(npz_path, allow_pickle=False) as d:
        return (np.asarray(d["candidates"], dtype=np.float32),
                np.asarray(d["xys"], dtype=np.int64),
                int(d["correct_idx"]))


def variation_npz_rows(size, tid, xys):
    """The SGF root variations' non-pass cells as activation-store row indices,
    in variation order — the exact cell set the shipped per-cell bin publishes.
    Returns None on a parse failure or a variation cell missing from the store."""
    sgf_path = os.path.join(PROJECT_ROOT, "tsumegos", str(size), tid + ".sgf")
    try:
        with open(sgf_path, "rb") as f:
            game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
        _, variation_nodes = parse_tsumego_sgf(game)
    except Exception:
        return None
    xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
    rows = []
    for node in variation_nodes:
        _, mb = node.get_raw_move()
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is None:
            continue                       # pass — never embedded
        r = xy_to_row.get((int(xy[0]), int(xy[1])))
        if r is None:
            return None
        rows.append(r)
    return rows
