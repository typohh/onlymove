"""Can KataGo activations predict REAL-USER results better than the bot rating?

The prior steps showed (a) activations linearly predict the bot-derived rating at
R²≈0.6 and (b) the bot rating predicts users unbiased but ~50% overdispersed
(σ_ε≈0.5 rating units). This asks the synthesis question directly: fit human
success/failure and see how much of it the activations explain — and whether they
beat the bot-benchmark rating as a human-difficulty signal.

Model (Rasch-style, additive in logit):
    logit P(success_{u,p}) = α_u  +  (difficulty term)
α_u is a free per-user skill intercept (one-hot). Four nested difficulty terms are
compared on the SAME cross-validation folds (over attempts):
    skill_only : (none)                          — user ability ceiling
    bot_rating : β · rating_p                     — the bot-difficulty predictor
    activations: w · PCA(correct-move act_p)      — KataGo activations
    both       : β · rating_p + w · PCA(act_p)    — does activation add over rating?

Activation PCA is unsupervised (fit on the attempted-problem activation matrix), so
problem overlap across folds is not leakage — every difficulty term is a *shared*
function of features, with no per-problem free parameters; only user intercepts are
per-entity, and qualified users appear in every train fold.

Reported per size: out-of-fold LogLoss / Brier / AUC for each model. The deltas are
the story — activations beating bot_rating ⇒ the raw trunk carries human-difficulty
signal the bot-solve rating loses.

NOTE: reads ratings/<size>/tsumego_ratings_final.csv as the bot rating. If you ran
`rank_tsumegos_and_bots --no-users`, that is the clean bot-only difficulty (ideal
here); otherwise it is the production with-user rating.

Usage:
  python -m auxiliary.predict_user_results_from_activations --size 13
  python -m auxiliary.predict_user_results_from_activations --size 9 --pca-dim 256
  python -m auxiliary.predict_user_results_from_activations --size 13 --contrastive
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
    N_CONTRAST_BLOCKS, _contrastive_cache_path,
)
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402


def load_problem_features(size, tids, pca_dim, contrastive, seed):
    """Return {tid: feature_vec} = PCA-reduced (correct-move [+contrastive])
    activations, fit unsupervised on the given attempted-problem set."""
    ids = json.load(open(_cache_ids_path(size)))
    row = {t: i for i, t in enumerate(ids)}
    vecs = np.load(_cache_vecs_path(size), mmap_mode="r")
    cvecs = None
    if contrastive:
        cpath = _contrastive_cache_path(size)
        if not os.path.exists(cpath):
            sys.exit(f"missing {cpath}; run predict_rating_from_activations --contrastive first")
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=None,
                        help="Board size (default: BOARD_SIZE from configurations).")
    parser.add_argument("--min", type=int, default=5,
                        help="Y: min correct AND incorrect first-time answers to keep a user.")
    parser.add_argument("--pca-dim", type=int, default=256,
                        help="Activation PCA components used as difficulty features.")
    parser.add_argument("--contrastive", action="store_true",
                        help="Add mean/std/max/zscore-over-other-moves blocks before PCA.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0, help="Inverse L2 strength.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size is None:
        from utils.configurations import BOARD_SIZE
        args.size = BOARD_SIZE

    df = load_first_time_attempts()
    df = df[df["size"] == args.size].copy()

    # Keep skill-estimable users.
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    qual = set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)])
    df = df[df["user_id"].isin(qual)].copy()

    # Bot rating per problem.
    rpath = os.path.join(PROJECT_ROOT, "ratings", str(args.size), "tsumego_ratings_final.csv")
    rating = pd.read_csv(rpath).set_index("tsumego_id")["rating"].to_dict()

    # Activation features for the attempted problems.
    feat, k = load_problem_features(args.size, df["problem"].unique(),
                                    args.pca_dim, args.contrastive, args.seed)

    # An attempt is usable only if it has BOTH a rating and activation features,
    # so all four models are compared on an identical attempt set.
    keep = df["problem"].map(lambda t: t in rating and t in feat)
    n_drop = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    print(f"size {args.size}: {len(df)} attempts ({n_drop} dropped: missing rating/activation) "
          f"| users {df['user_id'].nunique()} | problems {df['problem'].nunique()} "
          f"| activation PCs {k}{' +contrastive' if args.contrastive else ''}")

    y = df["success"].to_numpy(float)
    # Design blocks.
    users = pd.Categorical(df["user_id"])
    U = sparse.csr_matrix(
        (np.ones(len(df)), (np.arange(len(df)), users.codes)),
        shape=(len(df), len(users.categories)))
    rat = df["problem"].map(rating).to_numpy(float)
    rat = ((rat - rat.mean()) / rat.std()).reshape(-1, 1)
    act = np.stack(df["problem"].map(feat).to_numpy())

    blocks = {
        "skill_only":  None,
        "bot_rating":  rat,
        "activations": act,
        "both":        np.hstack([rat, act]),
    }

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    folds = list(kf.split(y))

    print(f"\n  {'model':>12s} {'LogLoss':>9s} {'Brier':>8s} {'AUC':>7s} {'ΔLL vs skill':>13s}")
    base_ll = None
    for name, feats in blocks.items():
        X = U if feats is None else sparse.hstack([U, sparse.csr_matrix(feats)]).tocsr()
        oof = np.empty(len(y))
        for tr, te in folds:
            clf = LogisticRegression(C=args.C, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        oof = np.clip(oof, 1e-6, 1 - 1e-6)
        ll = log_loss(y, oof)
        br = brier_score_loss(y, oof)
        auc = roc_auc_score(y, oof)
        if base_ll is None:
            base_ll = ll
        print(f"  {name:>12s} {ll:>9.4f} {br:>8.4f} {auc:>7.4f} {base_ll - ll:>+13.4f}")


if __name__ == "__main__":
    main()
