"""Which contrastive (non-correct-move) blocks actually help predict user results?

Block ablation for the linear activation→user-outcome model. Same Rasch setup
(logit P = user intercept + difficulty), GroupKFold by problem (memorization-proof).
Each block is PCA'd separately so its contribution is isolable:
    correct : the correct move's activation (the position itself)
    mean    : per-channel mean over the OTHER legal moves
    std     : per-channel std over the other moves
    max     : per-channel max over the other moves
    zscore  : the correct move's per-channel z-score within the board, (correct-mean)/std

(There is no `min` block — the contrastive cache was built with max only. Say the
word and it can be added.)

Two views, both on held-out user attempts:
  A. Standalone — skill + a single block: does the block carry difficulty signal at all?
  B. Marginal over correct — skill + correct + block: does it add beyond the correct move?

Usage:
  python -m auxiliary.user_results_block_ablation --size 9
  python -m auxiliary.user_results_block_ablation --size 13 --k-correct 128 --k-block 64
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.inspect_activation_signals import (  # noqa: E402
    N_FEATS, _cache_ids_path, _cache_vecs_path,
)
from auxiliary.predict_rating_from_activations import (  # noqa: E402
    CONTRAST_BLOCK_NAMES, _contrastive_cache_path,
)
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402


def pca_block(raw, k, seed):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    raw = StandardScaler().fit_transform(raw)
    k = min(k, raw.shape[1], raw.shape[0])
    Z = PCA(n_components=k, random_state=seed).fit_transform(raw)
    return StandardScaler().fit_transform(Z)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--min", type=int, default=5)
    parser.add_argument("--k-correct", type=int, default=128)
    parser.add_argument("--k-block", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.size is None:
        from utils.configurations import BOARD_SIZE
        args.size = BOARD_SIZE

    df = load_first_time_attempts()
    df = df[df["size"] == args.size].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    qual = set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)])
    df = df[df["user_id"].isin(qual)].copy()

    # Load correct + contrastive blocks for the attempted problems.
    ids = json.load(open(_cache_ids_path(args.size)))
    rowmap = {t: i for i, t in enumerate(ids)}
    vecs = np.load(_cache_vecs_path(args.size), mmap_mode="r")
    cvecs = np.load(_contrastive_cache_path(args.size), mmap_mode="r")
    usable = [t for t in df["problem"].unique() if t in rowmap]
    rows = [rowmap[t] for t in usable]
    correct = np.asarray(vecs[rows], np.float32)
    contrast = np.asarray(cvecs[rows], np.float32)
    finite = np.isfinite(correct).all(1) & np.isfinite(contrast).all(1)
    usable = [t for t, f in zip(usable, finite) if f]
    correct, contrast = correct[finite], contrast[finite]

    raw_blocks = {"correct": correct}
    for bi, name in enumerate(CONTRAST_BLOCK_NAMES):
        raw_blocks[name] = contrast[:, bi * N_FEATS:(bi + 1) * N_FEATS]

    print(f"size {args.size}: PCA per block "
          f"(correct→{args.k_correct}, others→{args.k_block})...", flush=True)
    feat = {}  # block -> {tid: pca_vec}
    for name, raw in raw_blocks.items():
        k = args.k_correct if name == "correct" else args.k_block
        Z = pca_block(raw, k, args.seed)
        feat[name] = {t: Z[i] for i, t in enumerate(usable)}

    keep = df["problem"].isin(set(usable))
    df = df[keep].reset_index(drop=True)
    y = df["success"].to_numpy(float)
    users = pd.Categorical(df["user_id"])
    U = sparse.csr_matrix((np.ones(len(df)), (np.arange(len(df)), users.codes)),
                          shape=(len(df), len(users.categories)))
    groups = pd.Categorical(df["problem"]).codes
    probs = df["problem"].to_numpy()

    def block_feats(name):
        return np.stack([feat[name][t] for t in probs])

    blockmats = {name: block_feats(name) for name in raw_blocks}
    print(f"  {len(df)} attempts | users {df['user_id'].nunique()} | "
          f"problems {df['problem'].nunique()} | GroupKFold by problem\n", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    folds = list(GroupKFold(n_splits=args.folds).split(y, groups=groups))

    def evaluate(parts):
        """parts: list of block names to include (besides user intercepts)."""
        X = U if not parts else sparse.hstack(
            [U] + [sparse.csr_matrix(blockmats[p]) for p in parts]).tocsr()
        oof = np.empty(len(y))
        for tr, te in folds:
            clf = LogisticRegression(C=args.C, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        oof = np.clip(oof, 1e-6, 1 - 1e-6)
        return log_loss(y, oof), roc_auc_score(y, oof)

    ll_skill, auc_skill = evaluate([])
    ll_corr, auc_corr = evaluate(["correct"])

    print("A. Standalone (skill + one block):")
    print(f"  {'block':>10s} {'LogLoss':>9s} {'AUC':>7s} {'ΔLL vs skill':>13s}")
    print(f"  {'skill_only':>10s} {ll_skill:>9.4f} {auc_skill:>7.4f} {0.0:>+13.4f}")
    for name in raw_blocks:
        ll, auc = evaluate([name])
        print(f"  {name:>10s} {ll:>9.4f} {auc:>7.4f} {ll_skill - ll:>+13.4f}")

    print("\nB. Marginal over correct (skill + correct + block):")
    print(f"  {'added':>10s} {'LogLoss':>9s} {'AUC':>7s} {'ΔLL vs correct':>15s}")
    print(f"  {'(none)':>10s} {ll_corr:>9.4f} {auc_corr:>7.4f} {0.0:>+15.4f}")
    for name in CONTRAST_BLOCK_NAMES:
        ll, auc = evaluate(["correct", name])
        print(f"  {name:>10s} {ll:>9.4f} {auc:>7.4f} {ll_corr - ll:>+15.4f}")
    ll_all, auc_all = evaluate(["correct"] + list(CONTRAST_BLOCK_NAMES))
    print(f"  {'ALL':>10s} {ll_all:>9.4f} {auc_all:>7.4f} {ll_corr - ll_all:>+15.4f}")


if __name__ == "__main__":
    main()
