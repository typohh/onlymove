"""Does NON-LINEARITY help predict user results — without memorization leakage?

Companion to predict_user_results_from_activations.py (linear). Same Rasch-style
setup — logit P(success_{u,p}) = α_u + difficulty(features_p) — but difficulty is
a small dense net, and the evaluation is hardened against memorization.

The memorization trap: a flexible net can map a problem's unique activation
fingerprint to that problem's empirical difficulty (an implicit per-problem lookup).
Under attempt-level CV (problems shared across folds) that inflates the score
without any generalization. So here EVERY fold is split by PROBLEM (GroupKFold on
problem id): test problems are never seen in training, so memorizing a fingerprint
buys nothing on the held-out set. Additional guards: PCA bottleneck (K ≪ #problems),
weight decay, dropout, and early stopping on a problem-grouped inner validation.
User intercepts stay estimable because each user spans many problems across folds.

Configs compared on identical problem-grouped folds:
    bot_rating (lin) : difficulty = β·rating
    act (lin)        : difficulty = w·PCA(act)            [linear ceiling, group-CV]
    act (mlp)        : difficulty = MLP(PCA(act))         [does non-linearity help?]
    act+rating (mlp) : difficulty = MLP([PCA(act), rating])

If act(mlp) ≈ act(lin) on held-out problems, the activation→human-difficulty map is
essentially linear; a gap means real non-linear structure (not memorization, since
test problems are unseen).

Usage:
  python -m auxiliary.predict_user_results_mlp --size 13
  python -m auxiliary.predict_user_results_mlp --size 9 --pca-dim 256 --hidden 256 64
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.predict_user_results_from_activations import load_problem_features  # noqa: E402
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402


class DiffModel(nn.Module):
    """logit = user_bias[u] + difficulty_net(features). hidden=[] → linear difficulty."""

    def __init__(self, n_users, in_dim, hidden, dropout):
        super().__init__()
        self.user_bias = nn.Embedding(n_users, 1)
        nn.init.zeros_(self.user_bias.weight)
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.diff = nn.Sequential(*layers)

    def forward(self, u, x):
        return self.user_bias(u).squeeze(-1) + self.diff(x).squeeze(-1)


def train_predict(Xtr, utr, ytr, Xva, uva, yva, Xte, ute,
                  n_users, hidden, dropout, wd, lr, max_epochs, patience, device):
    """Train with early stopping on the (problem-grouped) val split; return test probs."""
    model = DiffModel(n_users, Xtr.shape[1], hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr, utr, ytr = Xtr.to(device), utr.to(device), ytr.to(device)
    Xva, uva, yva = Xva.to(device), uva.to(device), yva.to(device)

    best_val, best_state, bad = float("inf"), None, 0
    n = len(ytr)
    bs = 8192
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            j = perm[s:s + bs]
            opt.zero_grad()
            loss = loss_fn(model(utr[j], Xtr[j]), ytr[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(uva, Xva), yva).item()
        if vl < best_val - 1e-5:
            best_val, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(ute.to(device), Xte.to(device))).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--min", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--contrastive", action="store_true")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 32],
                        help="MLP hidden layer sizes (default 128 32).")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size is None:
        from utils.configurations import BOARD_SIZE
        args.size = BOARD_SIZE
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_first_time_attempts()
    df = df[df["size"] == args.size].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    qual = set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)])
    df = df[df["user_id"].isin(qual)].copy()

    rpath = os.path.join(PROJECT_ROOT, "ratings", str(args.size), "tsumego_ratings_final.csv")
    rating = pd.read_csv(rpath).set_index("tsumego_id")["rating"].to_dict()
    feat, k = load_problem_features(args.size, df["problem"].unique(),
                                    args.pca_dim, args.contrastive, args.seed)
    keep = df["problem"].map(lambda t: t in rating and t in feat)
    df = df[keep].reset_index(drop=True)
    print(f"size {args.size}: {len(df)} attempts | users {df['user_id'].nunique()} | "
          f"problems {df['problem'].nunique()} | PCs {k}"
          f"{' +contrastive' if args.contrastive else ''} | device {device.type} | "
          f"CV: GroupKFold by problem (memorization-proof)")

    y = df["success"].to_numpy(np.float32)
    ucodes = pd.Categorical(df["user_id"]).codes.astype(np.int64)
    n_users = int(ucodes.max()) + 1
    rat = df["problem"].map(rating).to_numpy(float)
    rat = ((rat - rat.mean()) / rat.std()).astype(np.float32).reshape(-1, 1)
    act = np.stack(df["problem"].map(feat).to_numpy()).astype(np.float32)
    groups = pd.Categorical(df["problem"]).codes

    configs = [
        ("bot_rating (lin)", rat, []),
        ("act (lin)",        act, []),
        ("act (mlp)",        act, args.hidden),
        ("act+rating (mlp)", np.hstack([act, rat]), args.hidden),
    ]

    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=args.folds)
    outer = list(gkf.split(y, groups=groups))
    rng = np.random.default_rng(args.seed)

    print(f"\n  {'model':>18s} {'LogLoss':>9s} {'Brier':>8s} {'AUC':>7s} {'ΔLL vs rating':>14s}")
    rating_ll = None
    for name, X, hidden in configs:
        oof = np.empty(len(y))
        for tr, te in outer:
            # Inner problem-grouped val split for early stopping (15% of train problems).
            tr_groups = groups[tr]
            uniq = np.unique(tr_groups)
            va_g = set(rng.choice(uniq, size=max(1, int(0.15 * len(uniq))), replace=False))
            va_mask = np.array([gp in va_g for gp in tr_groups])
            tr_idx, va_idx = tr[~va_mask], tr[va_mask]
            oof[te] = train_predict(
                torch.tensor(X[tr_idx]), torch.tensor(ucodes[tr_idx]), torch.tensor(y[tr_idx]),
                torch.tensor(X[va_idx]), torch.tensor(ucodes[va_idx]), torch.tensor(y[va_idx]),
                torch.tensor(X[te]), torch.tensor(ucodes[te]),
                n_users, hidden, args.dropout, args.wd, args.lr,
                args.max_epochs, args.patience, device)
        oof = np.clip(oof, 1e-6, 1 - 1e-6)
        ll = log_loss(y, oof)
        if rating_ll is None:
            rating_ll = ll
        print(f"  {name:>18s} {ll:>9.4f} {brier_score_loss(y, oof):>8.4f} "
              f"{roc_auc_score(y, oof):>7.4f} {rating_ll - ll:>+14.4f}")


if __name__ == "__main__":
    main()
