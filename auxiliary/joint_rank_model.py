"""Prototype: place tsumegos directly into the internal (EGF) rank space from
their KataGo activations — one joint model, no rating space, no evaluation bots.

A single IRT-style fit over ALL real-user first-time attempts (both boards pooled)
plus the survey rank declarations:

    P(user u solves problem p) = σ( slope · ( θ_u − d_p ) )
        d_p = w · PCA(correct-move activations_p) + β · size_is_13   (rank units)
        θ_u = per-user ability (rank units), one per user, ranked OR not
    anchor (absolute scale):  θ_u ≈ internal_rank(user's declared ranks)
        EGF = identity; other servers converted via the existing rank_map.json knots

Everyone's outcomes pin the *relative* ordering of users and tsumegos and the
activation→difficulty map w; the ~80 survey users pin the *absolute* location on
the EGF scale. d_p = w·activations means any problem (even one no user has seen)
gets placed from its position alone — which is exactly what removes the evaluation
bots and the per-board rating scale. Unranked users still help: they constrain the
tsumego ordering.

This is a PROTOTYPE / validation harness, run alongside the production pipeline (it
writes nothing the PWA consumes). It checks the design's three load-bearing claims:
  1. the activation placement generalizes to unseen problems (GroupKFold by problem);
  2. the fitted abilities reproduce the survey ranks (absolute anchor holds);
  3. the placed ranks are consistent with the current bot-rating ordering.

Simplifications vs the production target (documented in
docs/joint_rank_methodology.md): the survey anchor is an L2 pull to the internal
rank rather than the joint ordinal cumulative-link with the shape prior on the
server couplings; the outcome link is a plain logistic rather than the
statistics.txt win-rate-vs-rank-gap shape. Both are noted as the next refinements.

Usage:
  ./.venv/bin/python3 -m auxiliary.joint_rank_model
  ./.venv/bin/python3 -m auxiliary.joint_rank_model --pca-dim 128 --min-attempts 10
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, latest_per_user_server, load_rank_surveys,
)
from auxiliary.fit_statistics_shapes import fit_shape  # noqa: E402
from auxiliary.inspect_activation_signals import (  # noqa: E402
    _cache_ids_path, _cache_vecs_path,
)
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402

# Internal-rank scalar (1d=+1, 1k=0, 10k=-9) → the statistics.txt rank_index axis
# (20k=0, 1k=19, 1d=20) used by the cubic shape.
RANK_INDEX_OFFSET = 19.0
SOLVE_OFFSET = float(np.log(0.8 / 0.2))  # rank = where you clear 80%, so P=0.8 at θ=d


def load_joint_features(sizes, tids_by_size, pca_dim, seed):
    """Joint PCA of raw correct-move activations across both boards.
    Returns {tid: feat (K,)} and K. Raw (not rating-residualized) — we WANT the
    difficulty signal that build_embedding deliberately removes."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    mats, tid_order = [], []
    for size in sizes:
        ids = json.load(open(_cache_ids_path(size)))
        rowmap = {t: i for i, t in enumerate(ids)}
        vecs = np.load(_cache_vecs_path(size), mmap_mode="r")
        want = [t for t in tids_by_size[size] if t in rowmap]
        rows = [rowmap[t] for t in want]
        if not rows:
            continue
        X = np.asarray(vecs[rows], np.float32)
        finite = np.isfinite(X).all(1)
        mats.append(X[finite])
        tid_order += [t for t, f in zip(want, finite) if f]
    X = np.vstack(mats)
    X = StandardScaler().fit_transform(X)
    k = min(pca_dim, X.shape[1], X.shape[0])
    Z = PCA(n_components=k, random_state=seed).fit_transform(X)
    Z = StandardScaler().fit_transform(Z)
    return {t: Z[i] for i, t in enumerate(tid_order)}, k


def load_anchors():
    """{user_id: internal_rank} from survey declarations, EGF = identity, other
    servers mapped through the existing rank_map.json knots. A user with several
    declarations is averaged."""
    surveys = latest_per_user_server(load_rank_surveys(SQL_PATH_DEFAULT))
    rm_path = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rank_map.json")
    rm = json.load(open(rm_path)) if os.path.exists(rm_path) else {}
    by_user = {}
    for _, r in surveys.iterrows():
        g = r["rank_scalar"]
        if g is None or not np.isfinite(g):
            continue
        srv = r["server"]
        if srv == "EGF":
            internal = float(g)
        elif srv in rm and len(rm[srv]["our"]) >= 2:
            internal = float(np.interp(g, rm[srv]["server"], rm[srv]["our"]))
        else:
            continue
        by_user.setdefault(r["user_id"], []).append(internal)
    return {u: float(np.mean(v)) for u, v in by_user.items()}


class JointModel(nn.Module):
    """logit P(solve) = cubic(θ_u) − cubic(d_p) + logit(0.8), with cubic = the
    FIXED statistics.txt rank→win-logit shape. Fixing the link (no free slope)
    is the point: the model can't pack users together to fit relative outcomes —
    to reproduce ~80% it must place each user the rank-gap the shape demands above
    the problems they see, and the cubic's curvature spreads the kyu end instead
    of compressing it the way a flat/exponential slope would. θ_u and d_p are in
    internal-rank scalar units; difficulty d_p = w·activations + board FE."""

    def __init__(self, n_users, k, cubic_params):
        super().__init__()
        self.theta = nn.Embedding(n_users, 1)
        nn.init.zeros_(self.theta.weight)
        self.w = nn.Linear(k, 1, bias=False)
        nn.init.zeros_(self.w.weight)
        self.board = nn.Parameter(torch.zeros(1))
        a, b, d = [float(p) for p in cubic_params]
        self.register_buffer("ca", torch.tensor(a))
        self.register_buffer("cb", torch.tensor(b))
        self.register_buffer("cd", torch.tensor(d))

    def difficulty(self, feat, size13):
        return self.w(feat).squeeze(-1) + self.board * size13

    def _cubic(self, rank):
        # rank scalar → rank_index → fixed cubic in win-logit units. Clamp the
        # index to a sane range so w·activations can't drive x³ to ±inf.
        x = (rank + RANK_INDEX_OFFSET).clamp(-10.0, 40.0)
        return self.ca * x + self.cb * x * x + self.cd * x * x * x

    def forward(self, u, feat, size13):
        return (self._cubic(self.theta(u).squeeze(-1))
                - self._cubic(self.difficulty(feat, size13)) + SOLVE_OFFSET)


def train(model, feat, u, s13, y, anc_u, anc_v, device,
          epochs, lr, ridge, anchor_w, theta_l2):
    model.to(device)
    feat, u, s13, y = feat.to(device), u.to(device), s13.to(device), y.to(device)
    anc_u, anc_v = anc_u.to(device), anc_v.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = bce(model(u, feat, s13), y)
        loss = loss + anchor_w * ((model.theta(anc_u).squeeze(-1) - anc_v) ** 2).mean()
        loss = loss + ridge * model.w.weight.pow(2).sum()
        loss = loss + theta_l2 * model.theta.weight.pow(2).mean()
        loss.backward()
        opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="9,13")
    ap.add_argument("--min-attempts", type=int, default=10,
                    help="Keep users with at least this many first-time attempts.")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--anchor-w", type=float, default=1.0,
                    help="Weight of the L2 pull of θ toward declared internal rank.")
    ap.add_argument("--theta-l2", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "auxiliary",
                                                  "joint_rank_placements.csv"))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed link shape: the statistics.txt cubic rank→win-logit curve. The same
    # curve governs EVERY anchored user (all servers), not just EGF.
    cubic_params, cubic_theta = fit_shape("cubic")
    def _slope_at(scalar):  # local steepness of the link, logit per rank
        x = scalar + RANK_INDEX_OFFSET
        a, b, d = cubic_params
        return a + 2 * b * x + 3 * d * x * x
    print(f"Link: FIXED cubic shape (statistics.txt). Local steepness "
          f"{_slope_at(-9):.3f} logit/rank @10k, {_slope_at(0):.3f} @1k, "
          f"{_slope_at(5):.3f} @5d (vs the old fitted flat ~0.18).")

    df = load_first_time_attempts()
    df = df[df["size"].isin(sizes)].copy()
    # keep users with enough attempts (pooled across boards: ability is shared)
    counts = df.groupby("user_id").size()
    df = df[df["user_id"].isin(counts.index[counts >= args.min_attempts])].copy()

    tids_by_size = {s: df[df["size"] == s]["problem"].unique() for s in sizes}
    feat, k = load_joint_features(sizes, tids_by_size, args.pca_dim, args.seed)
    df = df[df["problem"].isin(feat.keys())].reset_index(drop=True)

    anchors = load_anchors()
    users = pd.Categorical(df["user_id"])
    uidx = users.codes.astype(np.int64)
    n_users = len(users.categories)
    ucat = {u: i for i, u in enumerate(users.categories)}

    anc_pairs = [(ucat[u], v) for u, v in anchors.items() if u in ucat]
    anc_u = np.array([a for a, _ in anc_pairs], np.int64)
    anc_v = np.array([v for _, v in anc_pairs], np.float32)

    X = np.stack(df["problem"].map(feat).to_numpy()).astype(np.float32)
    s13 = (df["size"] == 13).to_numpy(np.float32)
    y = df["success"].to_numpy(np.float32)
    groups = pd.Categorical(df["problem"]).codes

    print(f"attempts {len(df)} | users {n_users} (anchored {len(anc_pairs)}) | "
          f"problems {df['problem'].nunique()} | PCs {k} | device {device.type}")
    n9 = int((s13 == 0).sum()); n13 = int((s13 == 1).sum())
    print(f"  board mix: 9x9 {n9} attempts, 13x13 {n13} attempts\n")

    tX = torch.tensor(X); tu = torch.tensor(uidx)
    ts = torch.tensor(s13); ty = torch.tensor(y)
    tanu = torch.tensor(anc_u); tanv = torch.tensor(anc_v)

    common = dict(epochs=args.epochs, lr=args.lr, ridge=args.ridge,
                  anchor_w=args.anchor_w, theta_l2=args.theta_l2)

    # ---- Validation 1: outcome generalization, GroupKFold by problem ----
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold
    oof = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=args.folds).split(y, groups=groups):
        m = JointModel(n_users, k, cubic_params)
        train(m, tX[tr], tu[tr], ts[tr], ty[tr], tanu, tanv, device, **common)
        m.eval()
        with torch.no_grad():
            logit = m(tu[te].to(device), tX[te].to(device), ts[te].to(device))
            oof[te] = torch.sigmoid(logit).cpu().numpy()
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    print("Validation 1 — outcome generalization (GroupKFold by problem, "
          "memorization-proof):")
    print(f"  LogLoss {log_loss(y, oof):.4f} | AUC {roc_auc_score(y, oof):.4f} "
          f"(linear probe baseline ≈0.63)\n")

    from scipy.stats import pearsonr, spearmanr

    # ---- Validation 2: HELD-OUT anchor recovery, stratified by strength ----
    # Anchor on most survey users, leave a fold UN-anchored, then read the
    # left-out users' abilities (fit from their OUTCOMES only) and compare to
    # their declared rank. Folds are STRATIFIED by declared rank so each spans
    # the strength range, and the error is broken down per rank band to show
    # WHERE recovery is worst (the in-sample version is circular — the L2 pull
    # guarantees a match).
    from sklearn.model_selection import StratifiedKFold
    declared = np.array([v for _, v in anc_pairs])
    auser = np.array([a for a, _ in anc_pairs])
    nfold = min(5, len(anc_pairs))
    nbin = min(nfold, len(np.unique(declared)))
    strata = pd.qcut(declared, q=nbin, labels=False, duplicates="drop")
    oof_theta = np.full(len(anc_pairs), np.nan)
    for tr_a, te_a in StratifiedKFold(nfold, shuffle=True, random_state=args.seed).split(
            np.arange(len(anc_pairs)), strata):
        m = JointModel(n_users, k, cubic_params)
        train(m, tX, tu, ts, ty, tanu[tr_a], tanv[tr_a], device, **common)
        with torch.no_grad():
            th = m.theta.weight.squeeze(-1).cpu().numpy()
        oof_theta[te_a] = th[auser[te_a]]
    rmse = float(np.sqrt(np.mean((oof_theta - declared) ** 2)))
    print("Validation 2 — HELD-OUT anchor recovery (stratified by declared rank; "
          f"{len(anc_pairs)} survey users; rank scalar: 1d=+1, 1k=0, 10k=-9):")
    print(f"  overall: Spearman {spearmanr(oof_theta, declared).statistic:.3f} | "
          f"Pearson {pearsonr(oof_theta, declared)[0]:.3f} | RMSE {rmse:.2f} ranks")
    bands = pd.qcut(declared, q=min(5, len(np.unique(declared))), duplicates="drop")
    bdf = pd.DataFrame({"declared": declared, "recovered": oof_theta, "band": bands})
    print(f"  {'declared band':>16s} {'n':>3s} {'mean_decl':>9s} {'mean_rec':>9s} "
          f"{'bias':>7s} {'MAE':>6s} {'RMSE':>6s}")
    for b, gdf in bdf.groupby("band", observed=True):
        e = gdf.recovered - gdf.declared
        print(f"  {f'{b.left:+.1f}..{b.right:+.1f}':>16s} {len(gdf):>3d} "
              f"{gdf.declared.mean():>+9.2f} {gdf.recovered.mean():>+9.2f} "
              f"{e.mean():>+7.2f} {e.abs().mean():>6.2f} "
              f"{np.sqrt((e**2).mean()):>6.2f}")
    print()

    # ---- Full fit for placement ----
    model = JointModel(n_users, k, cubic_params)
    train(model, tX, tu, ts, ty, tanu, tanv, device, **common)
    model.eval()
    board = float(model.board.item())
    print(f"Full fit: fixed cubic link | board FE (13x13) {board:+.3f} rank")

    # ---- Placement: d_p for every problem ----
    with torch.no_grad():
        allfeat = torch.tensor(np.stack(list(feat.values())).astype(np.float32)).to(device)
        # placement uses board=9x9 frame (size13=0); board FE reported separately
        d = model.difficulty(allfeat, torch.zeros(len(feat), device=device)).cpu().numpy()
    tid_list = list(feat.keys())
    size_of = {}
    for s in sizes:
        for t in tids_by_size[s]:
            size_of.setdefault(t, s)
    placed = pd.DataFrame({"tsumego_id": tid_list,
                           "size": [size_of.get(t, -1) for t in tid_list],
                           "internal_rank": d})

    # ---- Validation 3: consistency with current bot-rating ordering ----
    print("\nValidation 3 — placed internal_rank vs current bot rating (Spearman):")
    for s in sizes:
        rp = os.path.join(PROJECT_ROOT, "ratings", str(s), "tsumego_ratings_final.csv")
        if not os.path.exists(rp):
            continue
        rat = pd.read_csv(rp).set_index("tsumego_id")["rating"]
        sub = placed[placed["size"] == s].set_index("tsumego_id").join(rat, how="inner")
        rho = spearmanr(sub["internal_rank"], sub["rating"]).statistic
        print(f"  size {s}: ρ={rho:.3f} over {len(sub)} problems | "
              f"placed rank range [{sub['internal_rank'].min():.1f}, "
              f"{sub['internal_rank'].max():.1f}]")

    placed.sort_values("internal_rank").to_csv(args.out, index=False)
    print(f"\nWrote {len(placed)} placed tsumego ranks → {args.out}")


if __name__ == "__main__":
    main()
