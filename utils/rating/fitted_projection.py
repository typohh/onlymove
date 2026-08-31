"""Shared loading for the ablation and auxiliary/rating_dev/ scripts: a fitted
trunk and its per-cell projection, bundled so every caller projects EXACTLY the
computation the checkpoint was fitted with. The model and the feature assembly
stay single-sourced in utils.rating.mirt_net / utils.rating.features — this
module only gathers the loading idiom (the same one pipeline.write_mirt_per_cell
uses) so those scripts cannot restate it differently.

`ShippedProjection` is the production fit at the shipped block layout and is
what an ablation quoting a production figure uses. `FittedProjection` is the
same machinery pointed at any checkpoint and block layout, so a dev fit trained
on a different feature set can be projected as a COMPARISON ARM — read on its
own full input, never as a masked variant of something else.
"""

import os

import numpy as np

from utils.rating.mirt_net import build_trunk
from utils.rating.model_constants import (
    K_AXES, PCA_DIM, PRODUCTION_BLOCKS, SEED, SIZES)
from utils.rating.features import (
    SIZE_FLAG_COLS, corpus_tids_by_size, joint_feature_tail,
    load_joint_activation_basis, load_joint_blocks)
from utils.paths import PROJECT_ROOT

MODEL_PT = os.path.join(PROJECT_ROOT, "ratings", "mirt_model.pt")

# The block layouts a checkpoint here can have been fitted on, named exactly as
# load_joint_mirt_features' `blocks` argument names them — a checkpoint trained
# with `--features X` is projected with blocks="X" and nothing else. The layouts
# themselves are utils.rating.features.load_joint_blocks; this tuple is only the
# subset these scripts compare.
FEATURE_BLOCKS = ("all", "activations+bot")


def quantize(a):
    """Client resolution: fp16 storage cast, then quantizeEmbedding's 0.01 rounding."""
    return np.round(a.astype(np.float16).astype(np.float64) * 100.0) / 100.0


class FittedProjection:
    """A fitted trunk + axis head and everything needed to feed it: the
    joint-PCA basis, the per-problem tail blocks for ITS layout, the corpus.

    `blocks` is the layout the checkpoint was fitted on (see FEATURE_BLOCKS),
    defaulting to the shipped one. It is not inferable from the file, and getting
    it wrong is caught by the trunk's first Linear refusing the width rather than
    silently absorbed — which is why a caller pointing at a dev checkpoint states
    it.

    The joint-PCA basis is shared across checkpoints by construction: it is
    keyed on the corpus, PCA_DIM and SEED, none of which a feature ablation
    moves, so two fits' activation blocks are the same columns.
    """

    def __init__(self, device, model_path=MODEL_PT, blocks=PRODUCTION_BLOCKS):
        import torch
        import torch.nn as nn
        if not os.path.exists(model_path):
            hint = (" — run pipeline.train_mirt first" if model_path == MODEL_PT
                    else "")
            raise SystemExit(f"no {model_path}{hint}")
        self._torch = torch
        self.device = device
        self.model_path = model_path
        self.blocks = blocks
        self.tids_by_size = corpus_tids_by_size(SIZES)
        basis = load_joint_activation_basis(SIZES, self.tids_by_size, PCA_DIM, SEED)
        self.m1, self.s1 = basis["m1"], basis["s1"]
        self.C, self.pmean = basis["C"], basis["pca_mean"]
        self.m2, self.s2 = basis["m2"], basis["s2"]
        st = torch.load(model_path, map_location=device)
        self.wide, self.n_wide = load_joint_blocks(blocks, SIZES, self.tids_by_size)
        self.fdim = fdim = PCA_DIM + self.n_wide + SIZE_FLAG_COLS
        trunk, d = build_trunk(fdim, dropout=0.0)
        self.trunk = trunk.to(device)
        self.trunk.load_state_dict(st["trunk"])
        self.trunk.eval()
        k = int(st.get("k", K_AXES))
        if k != K_AXES:
            raise SystemExit(
                f"{model_path} carries {k} axes, K_AXES is {K_AXES} — axes are "
                f"compared BY INDEX here, so a differing count has no alignment")
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


class ShippedProjection(FittedProjection):
    """The production fit at the shipped block layout — what a figure quoting
    the deployed model is measured on."""

    def __init__(self, device):
        super().__init__(device, MODEL_PT, PRODUCTION_BLOCKS)


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
    Returns None on a parse failure or a variation cell missing from the store.

    The SGF reader and the coordinate conversions are imported INSIDE the
    function, following utils.rating.motif_rules: this module is read by the
    whole rating side and importing it must not drag in the KataGo half of
    utils/."""
    import sgfmill.sgf

    from utils.katago.coords import gtp_to_xy, sgf_to_gtp
    from utils.katago.sgf_parsing import parse_tsumego_sgf

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
