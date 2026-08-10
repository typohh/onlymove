#!/usr/bin/env python3
"""The hand-crafted axis warm start, kept OUT of production.

Ridge-regresses three chosen scalars — win-rate tightness, score tightness and
game phase — onto the z-scored features to make starting directions for the
first three axes, leaving every later axis at its random init.

This is the alternative the production eigenvector start replaced, and it lives
here rather than in utils.rating.residual_axes because production carries one
implementation of anything, not a switch. It is retained only so the comparison
that retired it can be re-run: the asymmetry it creates — as many warm-started
axes as there are chosen scalars, the rest cold — is what confounded the axis
count with the initialisation, since a count one above the number of scalars was
judged on an axis that began with no information.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import (  # noqa: E402
    load_tightness_gaps, phase_ratio_z, zscore_neglog,
)

SEED_RIDGE = 10.0


def hand_scalars(tids):
    """(n_tids, 3) array of [wr_gap, score_gap, phase_ratio], z-scored."""
    f_wr = zscore_neglog(load_tightness_gaps("wr_gap"))
    f_sc = zscore_neglog(load_tightness_gaps("score_gap"))
    f_ph = phase_ratio_z()
    return np.stack([
        np.array([f_wr[t] for t in tids], dtype=np.float32),
        np.array([f_sc[t] for t in tids], dtype=np.float32),
        np.array([f_ph[t] for t in tids], dtype=np.float32),
    ], axis=1)


def seed_directions(F, S_mat, k_seed):
    """Ridge-regress each hand-crafted scalar onto F; unit rows (k_seed, n_feat)."""
    G0 = np.linalg.solve(F.T @ F + SEED_RIDGE * np.eye(F.shape[1]),
                         F.T @ S_mat[:, :k_seed])
    return (G0 / (np.linalg.norm(G0, axis=0, keepdims=True) + 1e-9)).T
