"""Shared rating-model data/feature loaders for the rating-dev harness.

Re-exports the cross-board profile/feature builders from `utils.rating.features`
(`build_profiles`, `load_joint_activations`, `load_joint_bot_winrate`,
`load_joint_move_features`) that `rating_dev/{train_rating_staged,
validate_ratings}.py` import, plus `load_problem_features` (unsupervised
activation-PCA features for a problem set).

History: this was `auxiliary/compare_rating_models.py`, the held-out human-outcome
model-comparison CLI behind the `--even-error` DRO study in
`docs/methodology.md`. The comparison/fitting machinery
(`RatingModel`, `fit_eval`, `run_*`, `even_error`, …) was used only by that CLI
and has been removed — recover it from git history if it's ever needed again.
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import (  # noqa: F401,E402  (build_profiles/load_joint_* re-exported for the trainers/validator)
    N_CONTRAST_BLOCKS,
    N_FEATS,
    _cache_ids_path,
    _cache_vecs_path,
    _contrastive_cache_path,
    build_profiles,
    load_joint_activations,
    load_joint_bot_winrate,
    load_joint_alt_activations,
    load_joint_continuation,
    load_joint_game_phase,
    load_joint_move_features,
)


def load_problem_features(size, tids, pca_dim, contrastive, seed):
    """Return ({tid: feature_vec}, k) = PCA-reduced (correct-move [+contrastive])
    activations, fit unsupervised on the given attempted-problem set.

    Merged in from the former auxiliary.predict_user_results_from_activations."""
    ids = json.load(open(_cache_ids_path(size)))
    row = {t: i for i, t in enumerate(ids)}
    vecs = np.load(_cache_vecs_path(size), mmap_mode="r")
    cvecs = None
    if contrastive:
        cpath = _contrastive_cache_path(size)
        if not os.path.exists(cpath):
            sys.exit(f"missing {cpath}; the contrastive-cache builder was retired with the "
                     f"ICA embedding — recover it from git history to rebuild this arm")
        cvecs = np.load(cpath, mmap_mode="r")

    usable = [t for t in tids if t in row]
    rows = np.array([row[t] for t in usable])
    width = N_FEATS * (1 + (N_CONTRAST_BLOCKS if contrastive else 0))
    X = np.empty((len(usable), width), dtype=np.float32)
    X[:, :N_FEATS] = vecs[rows]
    if contrastive:
        X[:, N_FEATS:] = cvecs[rows]
    finite = np.isfinite(X).all(axis=1)
    usable = [t for t, f in zip(usable, finite) if f]
    X = X[finite]

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    k = min(pca_dim, X.shape[1], X.shape[0])
    Z = PCA(n_components=k, random_state=seed).fit_transform(X)
    Z = StandardScaler().fit_transform(Z)  # unit-variance PCs for a sane ridge scale
    return {t: Z[i] for i, t in enumerate(usable)}, k
