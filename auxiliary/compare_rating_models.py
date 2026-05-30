"""Head-to-head: can an ACTIVATION-based rating beat the humanv0/bot rating at
predicting held-out HUMAN outcomes — in our own rating space, on the SAME
held-out problems, with a fair guessing floor on both?

This is the decision test for replacing the bot-benchmark difficulty with a
shared activation→difficulty map. It deliberately drops the survey/rank axis:
both models live in their own internal logit-difficulty space, and any later
map to displayed ranks is a monotone reparametrisation that does not change a
log-loss comparison.

What makes the comparison fair
------------------------------
1. HELD OUT BY PROBLEM (GroupKFold on problem id). Test problems are never seen
   in training, so a per-problem free parameter buys nothing — the only way to
   score a test problem is a function of its features. This is the regime that
   matters: rating a brand-new problem with no human (or bot) attempts on it.
   Attempt-level CV would flatter the incumbent, whose shipped rating distils a
   per-problem fit.
2. Both difficulties are SHARED FUNCTIONS of that model's features, no
   per-problem freedom:
       bot_rating : d_p = β · rating_p            (the humanv0/bot scalar)
       activations: d_p = w · PCA(act_p)          (KataGo trunk, correct move)
       both       : d_p = β·rating + w·PCA(act)
3. 2-MONTH OVERLAPPING PROFILES as the ability unit (same windowing as
   pipeline.evaluate_rating_model / rank_tsumegos_and_bots.load_data): a profile
   is `{user[:8]}_{YYYY-MM}` holding that month + the next, so a profile's
   ability θ is roughly stationary and real improvement-over-time doesn't
   confound the fit. θ is a per-profile intercept; profiles span many problems
   so they stay estimable across problem folds.
4. A LEARNABLE GUESSING FLOOR on BOTH models (prior work: the dominant human
   miss is a missing 3PL floor c≈0.34, not the rank):
       P(success) = c_p + (1 − c_p) · σ(θ_profile + d_p)
   Three floor modes are reported for every difficulty model:
       none    : c_p = 0                          (pure 2PL — the old setup)
       global  : c_p = σ(scalar)                  (isolates the DIFFICULTY race)
       feature : c_p = σ(g · features_p)          (per-problem floor, generalises
                 via the same features — tests whether activations also predict
                 the human floor better than the bot scalar can)
   `feature` is the closest generalisable form of a "per-problem learnable
   floor": a free per-problem c can't be evaluated on a held-out problem, so it
   is parametrised through each model's own features.

Everything (θ profiles, d-head, floor-head) is fit jointly per fold by Adam on
the training problems; the held-out problems' human attempts are scored. Reports
out-of-fold LogLoss / Brier / AUC / ECE.

Usage:
  ./.venv/bin/python3 -m auxiliary.compare_rating_models --size 13
  ./.venv/bin/python3 -m auxiliary.compare_rating_models --size 9 --pca-dim 128
  ./.venv/bin/python3 -m auxiliary.compare_rating_models --size 13 --contrastive
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.predict_user_results_from_activations import (  # noqa: E402
    load_problem_features,
    _cache_ids_path,
    _cache_vecs_path,
)
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402

EPS = 1e-6


def build_profiles(df):
    """Explode first-time attempts into 2-month overlapping profile rows.

    Each attempt at month M contributes to profile M and profile M-1, so
    profile K holds months {K, K+1} — a 2-month window overlapping its
    neighbour by one month (identical to evaluate_rating_model). Returns the
    same df with one row per (profile, attempt); `profile` replaces the raw
    user as the ability unit."""
    out = []
    for uid, tid, succ, ts in zip(df["user_id"], df["problem"],
                                  df["success"], df["created_at"]):
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        pm, py = dt.month - 1, dt.year
        if pm == 0:
            pm, py = 12, py - 1
        u8 = uid[:8]
        for prof in (f"{u8}_{dt.year}-{dt.month:02d}", f"{u8}_{py}-{pm:02d}"):
            out.append((prof, tid, int(succ)))
    return pd.DataFrame(out, columns=["profile", "problem", "success"])


def load_bot_winrate(size, tids, clip=1e-4):
    """Return ({tid: logit-winrate vector over every bot}, n_bot_cols).

    The RAW bot signal that the shipped mixture-of-sigmoids rating distils into
    a single scalar, fed here directly so the bot side of the comparison doesn't
    depend on that fit. For each problem the feature is the per-bot success
    probability across every `attempts/<size>/*.txt` file (humanv0 kyu/dan
    ladder + kata strength sweeps + heuristic/sgf_move), clipped to
    [clip, 1-clip] and logit-transformed — the additive-logit space the rating
    model lives in, so a LINEAR head has a fair shot (raw p compresses all the
    action into the tails). Bot files cover different problem subsets, so a
    missing (bot, problem) cell is mean-imputed in logit space per column;
    zero-variance columns are dropped; columns are then z-scored for a sane
    ridge scale (mirrors load_problem_features' PC standardisation)."""
    import glob
    adir = os.path.join(PROJECT_ROOT, "attempts", str(size))
    files = sorted(glob.glob(os.path.join(adir, "*.txt")))
    if not files:
        return {}, 0
    tidset = set(tids)
    lo, hi = clip, 1.0 - clip
    cols = []  # one {tid: logit(prob)} per bot file
    for f in files:
        d = {}
        with open(f) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 2:
                    continue
                t, p = parts
                if t not in tidset:
                    continue
                try:
                    pv = min(max(float(p), lo), hi)
                except ValueError:
                    continue
                d[t] = float(np.log(pv / (1.0 - pv)))
        cols.append(d)
    usable = [t for t in tids if any(t in c for c in cols)]
    ridx = {t: i for i, t in enumerate(usable)}
    X = np.full((len(usable), len(cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(cols):
        for t, v in c.items():
            X[ridx[t], j] = v
    colmean = np.nanmean(X, axis=0)
    nan = np.where(np.isnan(X))
    X[nan] = np.take(colmean, nan[1])           # mean-impute missing cells
    keep = np.nanstd(X, axis=0) > 1e-9
    X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)              # z-score columns
    return {t: X[i].astype(np.float32) for i, t in enumerate(usable)}, int(keep.sum())


def load_joint_activations(sizes, tids_by_size, pca_dim, seed):
    """ONE PCA basis shared across boards (so a pooled model's activation weights mean
    the same thing on 9×9 and 13×13). Loads each board's raw correct-move activation
    cache for its attempted problems, stacks, fits a single StandardScaler+randomized
    PCA, z-scores. Returns {(size, tid): vec}, k. (Distinct from embeddings/joint, which
    residualises rating OUT — useless as a difficulty feature.)"""
    import json as _json
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xparts, keys = [], []
    for size in sizes:
        ids = _json.load(open(_cache_ids_path(size)))
        row = {t: i for i, t in enumerate(ids)}
        vecs = np.load(_cache_vecs_path(size), mmap_mode="r")
        usable = [t for t in tids_by_size[size] if t in row]
        rows = np.array([row[t] for t in usable])
        X = np.asarray(vecs[rows], dtype=np.float32)
        finite = np.isfinite(X).all(axis=1)
        usable = [t for t, f in zip(usable, finite) if f]
        Xparts.append(X[finite])
        keys += [(size, t) for t in usable]
    X = np.vstack(Xparts)
    X = StandardScaler().fit_transform(X)
    k = min(pca_dim, X.shape[1], X.shape[0])
    Z = PCA(n_components=k, random_state=seed, svd_solver="randomized").fit_transform(X)
    Z = StandardScaler().fit_transform(Z)
    return {key: Z[i].astype(np.float32) for i, key in enumerate(keys)}, k


def load_joint_bot_winrate(sizes, tids_by_size, clip=1e-4):
    """Bot winrate in a shared column layout across boards (the bots are the same;
    a bot's logit pass-rate means the same thing on either board). Common file set =
    intersection across boards; clip→logit→pooled mean-impute→drop pooled-zero-var→
    pooled z-score. Returns {(size, tid): vec}, n_cols."""
    import glob
    filesets = []
    for size in sizes:
        adir = os.path.join(PROJECT_ROOT, "attempts", str(size))
        filesets.append({os.path.basename(f)
                         for f in glob.glob(os.path.join(adir, "*.txt"))})
    common = sorted(set.intersection(*filesets))
    lo, hi = clip, 1.0 - clip
    keys, mat = [], []
    for size in sizes:
        adir = os.path.join(PROJECT_ROOT, "attempts", str(size))
        tidset = set(tids_by_size[size])
        cols = []
        for fn in common:
            d = {}
            with open(os.path.join(adir, fn)) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) != 2:
                        continue
                    t, v = p
                    if t not in tidset:
                        continue
                    try:
                        pv = min(max(float(v), lo), hi)
                    except ValueError:
                        continue
                    d[t] = float(np.log(pv / (1.0 - pv)))
            cols.append(d)
        usable = [t for t in tids_by_size[size] if any(t in c for c in cols)]
        for t in usable:
            keys.append((size, t))
            mat.append([cols[j].get(t, np.nan) for j in range(len(common))])
    X = np.array(mat, dtype=np.float64)
    colmean = np.nanmean(X, axis=0)
    nan = np.where(np.isnan(X))
    X[nan] = np.take(colmean, nan[1])
    keep = np.nanstd(X, axis=0) > 1e-9
    X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, int(keep.sum())


def run_pooled(args, device):
    """ONE model for BOTH boards. Shared per-user ability θ (profile = user_YYYY-MM,
    no size — a user has one skill), shared feature space (joint-PCA activations ⊕
    common-column bot winrate). MLP head, 2PL (slope on, floor off — the floor is dead,
    see --floor-gap). Test A = bw_act; Test B = bw_act ⊕ size_is_13 flag. Reports overall
    + per-board, so we see whether pooling matches per-board models and whether the size
    flag (a board offset, like the production recall model's SIZE_IS_13) earns anything."""
    from sklearn.metrics import log_loss, roc_auc_score
    sizes = [9, 13]
    dh = tuple(args.diff_hidden)
    df = load_first_time_attempts()
    df = df[df["size"].isin(sizes)].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    qual = set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)])
    df = df[df["user_id"].isin(qual)].copy()
    tids_by_size = {s: df[df["size"] == s]["problem"].unique() for s in sizes}

    act, k = load_joint_activations(sizes, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(sizes, tids_by_size)
    prob2size = dict(zip(df["problem"], df["size"]))
    keep = df["problem"].map(lambda t: (prob2size[t], t) in act
                             and (prob2size[t], t) in bw)
    df = df[keep].copy()

    prof_df = build_profiles(df)            # profile = user_YYYY-MM (NO size → shared θ)
    pc = prof_df["profile"].value_counts()
    prof_df = prof_df[prof_df["profile"].isin(
        set(pc.index[pc >= args.min_profile]))].reset_index(drop=True)

    profiles = pd.Categorical(prof_df["profile"])
    prof_idx = profiles.codes.astype(np.int64)
    n_profiles = len(profiles.categories)
    y = prof_df["success"].to_numpy(float)
    groups = pd.Categorical(prof_df["problem"]).codes
    row_size = prof_df["problem"].map(prob2size).to_numpy()
    bwact = np.stack([np.concatenate([bw[(prob2size[t], t)], act[(prob2size[t], t)]])
                      for t in prof_df["problem"]]).astype(np.float32)
    size_flag = (row_size == 13).astype(np.float32).reshape(-1, 1)
    bwact_sz = np.hstack([bwact, size_flag]).astype(np.float32)

    print(f"POOLED 9+13 | joint PCA k={k} | bot cols {nb} | profiles {n_profiles} "
          f"| problems {prof_df['problem'].nunique()} | rows {len(prof_df)} "
          f"({(row_size==13).mean()*100:.0f}% are 13×13) | MLP {dh} | 2PL (slope on, "
          f"floor off) | {args.seeds} seeds | device {device.type}")
    for s in sizes:
        sb = y[row_size == s]
        print(f"  size {s}: {len(sb)} rows, base rate {sb.mean():.3f}")

    def seed_avg_oof(feat):
        acc = None
        for sd in range(args.seeds):
            o = fit_eval(prof_idx, feat, y, groups, n_profiles, "none", args.folds,
                         args.epochs, args.lr, args.ridge, device, sd, slope=True,
                         diff_hidden=dh, dropout=args.dropout, mlp_wd=args.mlp_wd)
            acc = o if acc is None else acc + o
        return acc / args.seeds

    print(f"\n  {'test':>20s} {'scope':>7s} {'LogLoss':>9s} {'AUC':>7s} {'n':>8s}")
    print("  " + "-" * 56)
    for name, feat in (("bw_act", bwact), ("bw_act+size_flag", bwact_sz)):
        oof = seed_avg_oof(feat)
        for scope, semask in ([("overall", np.ones(len(y), bool))]
                              + [(f"size {s}", row_size == s) for s in sizes]):
            m = semask & ~np.isnan(oof)
            print(f"  {name:>20s} {scope:>7s} "
                  f"{log_loss(y[m], np.clip(oof[m], EPS, 1-EPS)):>9.4f} "
                  f"{roc_auc_score(y[m], oof[m]):>7.4f} {int(m.sum()):>8d}")
        print()


class RatingModel(nn.Module):
    """Per-problem IRF with all three 3PL components driven by features (so they
    generalise to held-out problems — no per-problem free parameter):

        P = c_p + (1 − c_p) · σ( a_p · θ_profile + d_p )
        d_p = w·feat                       (difficulty)
        a_p = softplus(0.5413 + s·feat)    (slope / discrimination, ≈1 baseline;
              clamp [0.2,5])  — only when slope=True
        c_p per floor_mode                 (guessing floor)

    slope=False fixes a_p=1 (the 1PL/2PL-by-difficulty model)."""

    def __init__(self, n_profiles, fdim, floor_mode, slope=False,
                 diff_hidden=(), dropout=0.0):
        super().__init__()
        self.theta = nn.Embedding(n_profiles, 1)
        nn.init.zeros_(self.theta.weight)
        self.has_diff = fdim > 0
        self.mlp = bool(diff_hidden) and self.has_diff
        if self.mlp:
            # Non-linear difficulty head: MLP(PCA(act)) -> scalar. Memorisation-safe
            # under GroupKFold-by-problem (it's a shared function, no per-problem
            # param) + PCA bottleneck + dropout + ridge. Last layer zero-init so it
            # starts at the linear baseline's 0 and grows only if it earns held-out gain.
            layers, d = [], fdim
            for h in diff_hidden:
                layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            head = nn.Linear(d, 1, bias=False)
            nn.init.zeros_(head.weight)
            self.diff = nn.Sequential(*layers, head)
        elif self.has_diff:
            self.w = nn.Linear(fdim, 1, bias=False)
            nn.init.zeros_(self.w.weight)
        self.use_slope = slope and self.has_diff
        if self.use_slope:
            self.sg = nn.Linear(fdim, 1, bias=False)
            nn.init.zeros_(self.sg.weight)
            self.register_buffer("slope_bias", torch.tensor(0.5413))
        self.floor_mode = floor_mode
        if floor_mode == "global":
            # init at logit(0.05) so the floor starts small and grows if earned.
            self.c_logit = nn.Parameter(torch.tensor([-2.94]))
        elif floor_mode == "feature":
            self.cg = nn.Linear(fdim, 1)
            nn.init.zeros_(self.cg.weight)
            nn.init.constant_(self.cg.bias, -2.94)

    def forward(self, prof, feat):
        th = self.theta(prof).squeeze(-1)
        if self.use_slope:
            a = torch.nn.functional.softplus(self.slope_bias + self.sg(feat).squeeze(-1))
            th = a.clamp(0.2, 5.0) * th
        logit = th
        if self.mlp:
            logit = logit + self.diff(feat).squeeze(-1)
        elif self.has_diff:
            logit = logit + self.w(feat).squeeze(-1)
        p = torch.sigmoid(logit)
        if self.floor_mode == "global":
            c = torch.sigmoid(self.c_logit)
            p = c + (1 - c) * p
        elif self.floor_mode == "feature":
            c = torch.sigmoid(self.cg(feat).squeeze(-1))
            p = c + (1 - c) * p
        return p.clamp(EPS, 1 - EPS)

    def reg(self, ridge):
        # MLP diff head is regularised by decoupled weight-decay (optimizer) +
        # dropout, NOT this sum-of-squares term — adding ridge·Σw² over its ~10^4
        # params to a MEAN bce swamps the data and collapses the head to 0.
        r = 0.0
        if self.has_diff and not self.mlp:
            r = r + ridge * self.w.weight.pow(2).sum()
        if self.use_slope:
            r = r + ridge * self.sg.weight.pow(2).sum()
        if self.floor_mode == "feature":
            r = r + ridge * self.cg.weight.pow(2).sum()
        return r


def fit_eval(prof_idx, feat, y, groups, n_profiles, floor_mode, folds, epochs,
             lr, ridge, device, seed, slope=False, diff_hidden=(), dropout=0.0,
             mlp_wd=1e-4):
    """GroupKFold-by-problem OOF prediction for one (feature set, slope, floor) model."""
    from sklearn.model_selection import GroupKFold
    oof = np.full(len(y), np.nan)
    fdim = feat.shape[1] if feat is not None else 0
    ty = torch.tensor(y, dtype=torch.float32, device=device)
    tprof = torch.tensor(prof_idx, dtype=torch.long, device=device)
    tfeat = (torch.tensor(feat, dtype=torch.float32, device=device)
             if feat is not None else None)
    for tr, te in GroupKFold(folds).split(y, groups=groups):
        torch.manual_seed(seed)
        m = RatingModel(n_profiles, fdim, floor_mode, slope=slope,
                        diff_hidden=diff_hidden, dropout=dropout).to(device)
        # Only profiles present in TRAIN get a usable intercept; drop test rows
        # whose profile never appears in train (no ability estimate).
        train_profs = set(prof_idx[tr].tolist())
        te = te[np.isin(prof_idx[te], list(train_profs))]
        if m.mlp:
            # decoupled weight decay on the MLP diff head only (see reg()).
            mlp_p = set(map(id, m.diff.parameters()))
            other = [p for p in m.parameters() if id(p) not in mlp_p]
            opt = torch.optim.AdamW([
                {"params": other, "weight_decay": 0.0},
                {"params": list(m.diff.parameters()), "weight_decay": mlp_wd},
            ], lr=lr)
        else:
            opt = torch.optim.Adam(m.parameters(), lr=lr)
        bce = nn.BCELoss()
        if m.mlp:
            # Early stopping on a held-out-BY-PROBLEM inner split of TRAIN — the
            # MLP otherwise overfits into overconfidence (great AUC, ruined
            # log-loss). The inner-val problems are unseen by the head, and only
            # inner-val rows whose profile appears in inner-train (so it has an
            # ability estimate) are scored. Keeps the best-inner-val state.
            rng = np.random.default_rng(seed)
            tr_probs = np.unique(groups[tr])
            rng.shuffle(tr_probs)
            n_val = max(1, int(0.2 * len(tr_probs)))
            val_probs = set(tr_probs[:n_val].tolist())
            in_val = np.isin(groups[tr], list(val_probs))
            itr, iva = tr[~in_val], tr[in_val]
            itr_profs = set(prof_idx[itr].tolist())
            iva = iva[np.isin(prof_idx[iva], list(itr_profs))]
            itrp, itrf, itry = tprof[itr], tfeat[itr], ty[itr]
            ivap, ivaf, ivay = tprof[iva], tfeat[iva], ty[iva]
            best_ll, best_state, bad = np.inf, None, 0
            for ep in range(epochs):
                m.train(); opt.zero_grad()
                loss = bce(m(itrp, itrf), itry) + m.reg(ridge)
                loss.backward(); opt.step()
                if ep % 5 == 0 or ep == epochs - 1:
                    m.eval()
                    with torch.no_grad():
                        vll = bce(m(ivap, ivaf), ivay).item()
                    if vll < best_ll - 1e-5:
                        best_ll, bad = vll, 0
                        best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= 6:  # patience (×5 epochs)
                            break
            if best_state is not None:
                m.load_state_dict(best_state)
        else:
            trp, trf, try_ = tprof[tr], (tfeat[tr] if tfeat is not None else None), ty[tr]
            for _ in range(epochs):
                m.train()
                opt.zero_grad()
                loss = bce(m(trp, trf), try_) + m.reg(ridge)
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            oof[te] = m(tprof[te], tfeat[te] if tfeat is not None else None).cpu().numpy()
    return oof


def metrics(y, p, mask):
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    y, p = y[mask], np.clip(p[mask], EPS, 1 - EPS)
    ll = log_loss(y, p)
    br = brier_score_loss(y, p)
    auc = roc_auc_score(y, p)
    b = np.clip((p * 10).astype(int), 0, 9)
    ece = 0.0
    for i in range(10):
        sel = b == i
        if sel.any():
            ece += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return ll, br, auc, ece


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--min", type=int, default=5,
                    help="Min first-time correct AND incorrect to keep a user.")
    ap.add_argument("--min-profile", type=int, default=8,
                    help="Min attempts for a 2-month profile to be kept.")
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--contrastive", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--bins", type=int, default=8,
                    help="Quantile bins for the by-rating-range error breakdown.")
    ap.add_argument("--three-pl", action="store_true",
                    help="Full 3PL: each feature set also gets a per-problem slope "
                         "(discrimination) head + a per-problem (feature) floor, so "
                         "the comparison is floor+slope+difficulty vs floor+slope+"
                         "difficulty. Reported alongside the floor sweep.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matrix", action="store_true",
                    help="Run the head-to-head GRID for one size: feature source "
                         "{bot_rating, bot_winrate, activations, both} × head {lin, mlp} "
                         "× 3PL {off, on}, single table. 3PL off = σ(θ+d) (difficulty "
                         "only); 3PL on = feature-driven floor+slope+difficulty. For the "
                         ".md results doc.")
    ap.add_argument("--matrix-feats", nargs="*",
                    default=["bot_rating", "bot_rating_nousers", "bot_winrate",
                             "activations", "both"],
                    help="Which feature sources the --matrix / --fair-mlp grid covers "
                         "(bot_rating_nousers is skipped if the bot-only CSV is absent).")
    ap.add_argument("--fair-mlp", action="store_true",
                    help="Capacity-matched, multi-seed MLP SOURCE comparison: PCA every "
                         "wide source to --common-dim (bot_rating is irreducibly 1-D), "
                         "identical MLP head, floor=feature slope=off, over --seeds seeds. "
                         "Removes the input-width confound that makes --matrix's MLP rows "
                         "unfair across 1-D/61-D/128-D inputs. Reports each source's "
                         "seed-invariant LINEAR reference alongside.")
    ap.add_argument("--common-dim", type=int, default=32,
                    help="Common PCA width for --fair-mlp wide sources.")
    ap.add_argument("--head3pl", action="store_true",
                    help="Factorial per source: {lin,mlp}×{3PL off,on}, MLP multi-seeded "
                         "(native width). Answers 'does MLP and/or 3PL add value?' "
                         "directly. Use --matrix-feats to pick sources.")
    ap.add_argument("--rank-error", action="store_true",
                    help="Per-RANK held-out log-loss for the leak-free configs (incl. the "
                         "bot_rating_nousers_q = rating+quality 3PL test). Seed-averaged "
                         "OOF, binned by internal rank index (easy→hard), with Δskill.")
    ap.add_argument("--pool", action="store_true",
                    help="ONE pooled model for BOTH boards (shared per-user θ, joint-PCA "
                         "activations ⊕ common-column bot winrate), MLP 2PL. Reports "
                         "bw_act and bw_act+size_is_13 flag, overall + per-board. Ignores "
                         "--size.")
    ap.add_argument("--floor-gap", action="store_true",
                    help="Test whether the user-ability↔tsumego-rating DISTANCE governs "
                         "1PL-vs-3PL: bin held-out attempts by the 1PL predicted P (the "
                         "gap axis) and compare 1PL / +floor / 3PL log-loss per bin "
                         "(linear heads). Source = first --rank-feats or bot_rating_nousers.")
    ap.add_argument("--rank-feats", nargs="*", default=None,
                    help="Override the --rank-error source list (each run as mlp/3PL-on). "
                         "e.g. 'bot_winrate activations bw_act' for the leak-free hybrid; "
                         "'activations' with --pca-dim 256 for the double-PCA test.")
    ap.add_argument("--no-floor-sweep", action="store_true",
                    help="Skip the floor-sweep table + its rating breakdown (use with "
                         "--nonlinear to run only the linear-vs-MLP test, e.g. on 9x9 "
                         "where the full sweep is slow).")
    ap.add_argument("--nonlinear", action="store_true",
                    help="Also test a NON-LINEAR (MLP) activation difficulty head vs "
                         "the linear one, on identical held-out-problem folds, across "
                         "--seeds seeds (MLP init/dropout are the only seed-varying part, "
                         "so the spread is the reliability check the earlier single-seed "
                         "predict_user_results_mlp test lacked).")
    ap.add_argument("--diff-hidden", type=int, nargs="*", default=[128, 64],
                    help="MLP hidden sizes for the non-linear difficulty head.")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mlp-wd", type=float, default=1e-4,
                    help="Decoupled weight decay on the MLP difficulty head.")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Seeds for the --nonlinear stability check.")
    ap.add_argument("--nl-floor", choices=("none", "global", "feature"),
                    default="feature",
                    help="Floor mode used in the MLP-vs-linear test. Default 'feature' "
                         "gives BOTH a learnable per-problem floor so the MLP's gain is "
                         "attributable to non-linear DIFFICULTY, not to it implicitly "
                         "reconstructing a guessing floor the linear-no-floor model lacks. "
                         "Use 'none' for the floor-off isolation.")
    args = ap.parse_args()
    if args.size is None:
        from utils.configurations import BOARD_SIZE
        args.size = BOARD_SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.pool:
        run_pooled(args, device)
        return

    df = load_first_time_attempts()
    df = df[df["size"] == args.size].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    qual = set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)])
    df = df[df["user_id"].isin(qual)].copy()

    rpath = os.path.join(PROJECT_ROOT, "ratings", str(args.size),
                         "tsumego_ratings_final.csv")
    rating = pd.read_csv(rpath).set_index("tsumego_id")["rating"].to_dict()
    # Optional bot-ONLY rating (rank_tsumegos_and_bots --no-users), to isolate the
    # human-outcome leakage the with-users tsumego_ratings_final.csv carries.
    nupath = os.path.join(PROJECT_ROOT, "ratings", str(args.size),
                          "tsumego_ratings_botonly.csv")
    rating_nu = (pd.read_csv(nupath).set_index("tsumego_id")["rating"].to_dict()
                 if os.path.exists(nupath) else None)
    # Bot-only QUALITY (rank_tsumegos_and_bots --no-users), to test whether the
    # per-problem quality measure helps the 3PL slope/floor on top of the rating.
    qupath = os.path.join(PROJECT_ROOT, "ratings", str(args.size),
                          "tsumego_quality_botonly.csv")
    quality_nu = (pd.read_csv(qupath).set_index("tsumego_id")["quality"].to_dict()
                  if os.path.exists(qupath) else None)
    feat, k = load_problem_features(args.size, df["problem"].unique(),
                                    args.pca_dim, args.contrastive, args.seed)
    botw, n_bots = load_bot_winrate(args.size, df["problem"].unique())
    keep = df["problem"].map(lambda t: t in rating and t in feat and t in botw
                             and (rating_nu is None or t in rating_nu)
                             and (quality_nu is None or t in quality_nu))
    df = df[keep].copy()

    prof_df = build_profiles(df)
    pc = prof_df["profile"].value_counts()
    big = set(pc.index[pc >= args.min_profile])
    prof_df = prof_df[prof_df["profile"].isin(big)].reset_index(drop=True)

    profiles = pd.Categorical(prof_df["profile"])
    prof_idx = profiles.codes.astype(np.int64)
    n_profiles = len(profiles.categories)
    y = prof_df["success"].to_numpy(float)
    groups = pd.Categorical(prof_df["problem"]).codes
    rat = prof_df["problem"].map(rating).to_numpy(float)
    rat = ((rat - rat.mean()) / rat.std()).reshape(-1, 1).astype(np.float32)
    act = np.stack(prof_df["problem"].map(feat).to_numpy()).astype(np.float32)
    bw = np.stack(prof_df["problem"].map(botw).to_numpy()).astype(np.float32)
    rat_nu = None
    if rating_nu is not None:
        rat_nu = prof_df["problem"].map(rating_nu).to_numpy(float)
        rat_nu = ((rat_nu - rat_nu.mean()) / rat_nu.std()).reshape(-1, 1).astype(np.float32)
    rat_nu_q = None
    if rating_nu is not None and quality_nu is not None:
        q = prof_df["problem"].map(quality_nu).to_numpy(float)
        q = ((q - q.mean()) / q.std()).reshape(-1, 1).astype(np.float32)
        rat_nu_q = np.hstack([rat_nu, q]).astype(np.float32)  # [rating, quality]

    print(f"size {args.size}: {len(prof_df)} profile-rows | profiles {n_profiles} "
          f"(>= {args.min_profile} attempts) | problems {prof_df['problem'].nunique()} "
          f"| activation PCs {k}{' +contrastive' if args.contrastive else ''} "
          f"| bot cols {n_bots} | device {device.type}")
    base = float(y.mean())
    base_ll = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    print(f"base rate {base:.3f} | base-rate log-loss {base_ll:.4f}\n")

    feats = {"skill_only": None, "bot_rating": rat, "bot_winrate": bw,
             "activations": act, "both": np.hstack([rat, act]).astype(np.float32)}
    if rat_nu is not None:
        feats["bot_rating_nousers"] = rat_nu
    if rat_nu_q is not None:
        feats["bot_rating_nousers_q"] = rat_nu_q
    # leak-free hybrid: raw bot winrate ⊕ activations (NOT the production `both`,
    # which uses the leaky with-users bot_rating).
    feats["bw_act"] = np.hstack([bw, act]).astype(np.float32)
    floor_modes = ["none", "global", "feature"]

    if args.matrix:
        run_matrix(prof_idx, feats, y, groups, n_profiles, args, device)
        return
    if args.fair_mlp:
        fair_mlp(prof_idx, feats, y, groups, n_profiles, args, device)
        return
    if args.head3pl:
        head3pl_grid(prof_idx, feats, y, groups, n_profiles, args, device)
        return
    if args.rank_error:
        rank_error(prof_idx, feats, y, groups, n_profiles, prof_df, rating, args, device)
        return
    if args.floor_gap:
        floor_gap(prof_idx, feats, y, groups, n_profiles, args, device)
        return

    hdr = f"  {'model':>11s} {'floor':>8s} {'LogLoss':>9s} {'Brier':>8s} {'AUC':>7s} {'ECE':>7s} {'n_eval':>9s}"
    if not args.no_floor_sweep:
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        oof_none = {}  # per-model OOF preds at floor=none, for the rating-range breakdown
        for fname, fmat in feats.items():
            for fm in floor_modes:
                if fname == "skill_only" and fm == "feature":
                    continue  # no features to drive a floor
                oof = fit_eval(prof_idx, fmat, y, groups, n_profiles, fm,
                               args.folds, args.epochs, args.lr, args.ridge,
                               device, args.seed)
                mask = ~np.isnan(oof)
                ll, br, auc, ece = metrics(y, oof, mask)
                print(f"  {fname:>11s} {fm:>8s} {ll:>9.4f} {br:>8.4f} {auc:>7.4f} "
                      f"{ece:>7.4f} {int(mask.sum()):>9d}")
                if fm == "none":
                    oof_none[fname] = oof
            print()
        report_by_rating(prof_df, rating, y, oof_none, args.bins)

    if args.three_pl:
        print("\nFull 3PL (per-problem difficulty + slope + feature floor, all "
              "feature-driven):")
        print(hdr.replace(" floor ", " irf  "))
        print("  " + "-" * (len(hdr) - 2))
        oof_3pl = {}
        for fname, fmat in feats.items():
            if fname == "skill_only":
                continue
            oof = fit_eval(prof_idx, fmat, y, groups, n_profiles, "feature",
                           args.folds, args.epochs, args.lr, args.ridge,
                           device, args.seed, slope=True)
            mask = ~np.isnan(oof)
            ll, br, auc, ece = metrics(y, oof, mask)
            print(f"  {fname:>11s} {'3PL':>8s} {ll:>9.4f} {br:>8.4f} {auc:>7.4f} "
                  f"{ece:>7.4f} {int(mask.sum()):>9d}")
            oof_3pl[fname] = oof
        print()
        report_by_rating(prof_df, rating, y, oof_3pl, args.bins)

    if args.nonlinear:
        nonlinear_compare(prof_idx, feats, y, groups, n_profiles, prof_df,
                          rating, args, device)


def run_matrix(prof_idx, feats, y, groups, n_profiles, args, device):
    """Grid: feature source × head {lin, mlp} × 3PL {off, on}, one OOF table.

    3PL off = σ(θ + d_p)                  (difficulty-only, slope=1, no floor)
    3PL on  = c_p + (1-c_p)·σ(a_p·θ + d_p) (feature-driven floor + slope + diff)
    Identical held-out-by-problem folds for every cell. MLP uses the
    inner-validation early-stopping path; single seed (--seed) for the headline
    table — the multi-seed MLP stability check lives in --nonlinear."""
    dh = tuple(args.diff_hidden)
    base = float(y.mean())
    base_ll = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    print(f"\nMATRIX (size {args.size}) base-rate log-loss {base_ll:.4f} | "
          f"MLP {dh} dropout {args.dropout} | single seed {args.seed}")
    hdr = (f"  {'source':>12s} {'head':>4s} {'3PL':>4s} {'LogLoss':>9s} "
           f"{'Brier':>8s} {'AUC':>7s} {'ECE':>7s} {'n_eval':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for src in args.matrix_feats:
        if src not in feats:
            print(f"  (skip unknown source {src})")
            continue
        fmat = feats[src]
        for head in ("lin", "mlp"):
            hidden = dh if head == "mlp" else ()
            for tpl in ("off", "on"):
                floor = "feature" if tpl == "on" else "none"
                slope = (tpl == "on")
                oof = fit_eval(prof_idx, fmat, y, groups, n_profiles, floor,
                               args.folds, args.epochs, args.lr, args.ridge,
                               device, args.seed, slope=slope, diff_hidden=hidden,
                               dropout=args.dropout, mlp_wd=args.mlp_wd)
                mask = ~np.isnan(oof)
                ll, br, auc, ece = metrics(y, oof, mask)
                print(f"  {src:>12s} {head:>4s} {tpl:>4s} {ll:>9.4f} {br:>8.4f} "
                      f"{auc:>7.4f} {ece:>7.4f} {int(mask.sum()):>9d}")
                rows.append((src, head, tpl, ll, br, auc, ece, int(mask.sum())))
            print()
    return rows


def fair_mlp(prof_idx, feats, y, groups, n_profiles, args, device):
    """Capacity-matched, multi-seed MLP source comparison.

    Each wide source is PCA-reduced to --common-dim (unsupervised, same as
    load_problem_features — no label leak; bot_rating is irreducibly 1-D and
    stays so), then the *identical* MLP diff head (floor=feature, slope=off) is
    fit over --seeds seeds on the same held-out-by-problem folds. The linear
    reference (seed-invariant) is shown beside each. This removes --matrix's
    input-width confound: there the one MLP config was tuned for 1-D bot_rating
    and under-fit the 61-/128-D inputs. If bot_rating still dominates here it is
    NOT capacity — suspect the with-users rating fit leaking held-out human
    outcomes (bot_winrate is the clean bot-only baseline)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, roc_auc_score
    dh = tuple(args.diff_hidden)
    print(f"\nFAIR-MLP (size {args.size}) common-dim {args.common_dim} | MLP {dh} "
          f"dropout {args.dropout} mlp_wd {args.mlp_wd} | floor=feature slope=off | "
          f"{args.seeds} seeds")
    hdr = (f"  {'source':>12s} {'dim':>4s} {'lin LL':>8s} {'lin AUC':>8s} "
           f"{'mlp LogLoss(mean±std)':>23s} {'mlp AUC':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for src in args.matrix_feats:
        if src not in feats or feats[src] is None:
            continue
        fmat = feats[src]
        if fmat.shape[1] > args.common_dim:
            Z = StandardScaler().fit_transform(fmat)
            Z = PCA(n_components=args.common_dim,
                    random_state=args.seed).fit_transform(Z)
            Z = StandardScaler().fit_transform(Z).astype(np.float32)
        else:
            Z = fmat
        used = Z.shape[1]
        ol = fit_eval(prof_idx, Z, y, groups, n_profiles, "feature", args.folds,
                      args.epochs, args.lr, args.ridge, device, args.seed)
        ml = ~np.isnan(ol)
        ll_l = log_loss(y[ml], np.clip(ol[ml], EPS, 1 - EPS))
        auc_l = roc_auc_score(y[ml], ol[ml])
        lls, aucs = [], []
        for s in range(args.seeds):
            om = fit_eval(prof_idx, Z, y, groups, n_profiles, "feature", args.folds,
                          args.epochs, args.lr, args.ridge, device, s,
                          diff_hidden=dh, dropout=args.dropout, mlp_wd=args.mlp_wd)
            mm = ~np.isnan(om)
            lls.append(log_loss(y[mm], np.clip(om[mm], EPS, 1 - EPS)))
            aucs.append(roc_auc_score(y[mm], om[mm]))
        print(f"  {src:>12s} {used:>4d} {ll_l:>8.4f} {auc_l:>8.4f} "
              f"{np.mean(lls):>15.4f}±{np.std(lls):<7.4f} {np.mean(aucs):>8.4f}   "
              "[" + " ".join(f"{x:.4f}" for x in lls) + "]")


_SHORT = {"bot_rating_nousers": "bratNU", "bot_rating_nousers_q": "bratNU+q",
          "bot_winrate": "bwin", "activations": "act", "both": "both",
          "bot_rating": "brat", "bw_act": "bwin+act"}


def _rank_of(prof_df, rating, size):
    """Map each profile-row's problem to its internal RANK INDEX via the served
    rating->rank curve (pwa/tsumegos/rating_to_rank.json). Higher = harder. The
    production (with-users) rating is fine as a binning axis even though it's a
    leaky *feature* — here it only orders problems; the models being scored are
    leak-free."""
    import json
    knots = json.load(open(os.path.join(PROJECT_ROOT, "pwa", "tsumegos",
                                         "rating_to_rank.json")))[str(size)]
    xs = np.asarray(knots["ratings"], float)
    ys = np.asarray(knots["rank_indices"], float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    raw = prof_df["problem"].map(rating).to_numpy(float)
    return np.interp(raw, xs, ys)


def rank_error(prof_idx, feats, y, groups, n_profiles, prof_df, rating, args, device):
    """Per-RANK held-out error for the leak-free configs (incl. the +quality 3PL
    test). Seed-averaged OOF per config, then log-loss within equal-count rank
    bins to see whether error is flat across ranks or concentrated at a tail."""
    from sklearn.metrics import log_loss, roc_auc_score
    dh = tuple(args.diff_hidden)
    if args.rank_feats:
        want = [(s, "mlp", "on") for s in args.rank_feats]
    else:
        want = [("bot_rating_nousers", "lin", "on"),
                ("bot_rating_nousers", "mlp", "on"),
                ("bot_rating_nousers_q", "lin", "on"),
                ("bot_rating_nousers_q", "mlp", "on"),
                ("bot_winrate", "mlp", "on"),
                ("activations", "mlp", "on")]
    want = [(s, h, t) for (s, h, t) in want if s in feats and feats[s] is not None]
    print(f"\nRANK-ERROR (size {args.size}) | MLP {dh} dropout {args.dropout} | "
          f"{args.seeds} seeds | 3PL=on (feature floor+slope)")
    print(f"  {'config':>16s} {'feat_dim':>8s} {'LogLoss':>9s} {'AUC':>7s}")
    print("  " + "-" * 46)
    oof = {}
    for (src, head, tpl) in want:
        floor = "feature" if tpl == "on" else "none"
        slope = (tpl == "on")
        hidden = dh if head == "mlp" else ()
        seeds = list(range(args.seeds)) if head == "mlp" else [args.seed]
        acc = None
        for s in seeds:
            o = fit_eval(prof_idx, feats[src], y, groups, n_profiles, floor,
                         args.folds, args.epochs, args.lr, args.ridge, device, s,
                         slope=slope, diff_hidden=hidden, dropout=args.dropout,
                         mlp_wd=args.mlp_wd)
            acc = o if acc is None else acc + o
        o = acc / len(seeds)
        key = f"{_SHORT.get(src, src)}/{head}"
        oof[key] = o
        m = ~np.isnan(o)
        print(f"  {key:>16s} {feats[src].shape[1]:>8d} "
              f"{log_loss(y[m], np.clip(o[m], EPS, 1-EPS)):>9.4f} "
              f"{roc_auc_score(y[m], o[m]):>7.4f}")
    rank_breakdown(prof_df, rating, y, oof, args.size, args.bins)


def rank_breakdown(prof_df, rating, y, oof, size, n_bins):
    rank = _rank_of(prof_df, rating, size)
    models = list(oof)
    valid = ~np.isnan(oof[models[0]])
    edges = np.quantile(rank[valid], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6

    def ll(yy, pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return -np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    print("\nHeld-out log-loss by RANK INDEX (easy→hard); "
          "Δskill = LL − bin base-rate LL (<0 beats base):")
    head = (f"  {'rank bin':>13s} {'n':>7s} {'base':>5s}"
            + "".join(f"{m:>11s}" for m in models)
            + "".join(f"{'Δ'+m:>11s}" for m in models) + f"  {'worst':>10s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i in range(n_bins):
        sel = valid & (rank >= edges[i]) & (rank < edges[i + 1])
        if sel.sum() < 50:
            continue
        yy = y[sel]
        base = yy.mean()
        bll = -(base*np.log(max(base, EPS)) + (1-base)*np.log(max(1-base, EPS)))
        lls = {m: ll(yy, oof[m][sel]) for m in models}
        worst = max(lls, key=lls.get)
        print(f"  {edges[i]:5.1f}..{edges[i+1]:5.1f} {int(sel.sum()):>7d} {base:>5.2f}"
              + "".join(f"{lls[m]:>11.4f}" for m in models)
              + "".join(f"{lls[m]-bll:>+11.4f}" for m in models)
              + f"  {worst:>10s}")


def floor_gap(prof_idx, feats, y, groups, n_profiles, args, device):
    """Does the user-ability ↔ tsumego-rating DISTANCE govern how 1PL vs 3PL compare?

    The guessing floor c only moves predictions where P is low (problem much harder
    than the user, logit θ+d ≪ 0). For the LINEAR 1PL, logit(oof) = θ_profile + d_p
    exactly, and d_p = w·rating, so that logit is monotone in (tsumego_rating −
    user_ability) — the very 'distance' in question. So: bin held-out attempts by the
    1PL predicted P (= position on that gap axis), show the DISTRIBUTION (how much mass
    sits in the hard-for-user regime), and compare 1PL vs +floor vs 3PL log-loss per
    bin. Hypothesis: the bins cluster at high P (problem≈user, adaptive selection), and
    1PL/+floor/3PL only diverge in the sparse low-P (hard) bins. Linear heads → the
    floor/slope are the only differences and the runs are seed-invariant."""
    src = next((s for s in (args.rank_feats or ["bot_rating_nousers"])
                if s in feats and feats[s] is not None), None) or "bot_winrate"
    fmat = feats[src]

    def oof_of(floor, slope):
        return fit_eval(prof_idx, fmat, y, groups, n_profiles, floor, args.folds,
                        args.epochs, args.lr, args.ridge, device, args.seed, slope=slope)

    p1 = oof_of("none", False)      # 1PL: σ(θ+d), no floor, unit slope
    pf = oof_of("feature", False)   # +floor: c+(1-c)σ(θ+d)
    p3 = oof_of("feature", True)    # 3PL: feature floor + slope
    m = ~(np.isnan(p1) | np.isnan(pf) | np.isnan(p3))
    p1, pf, p3, yy = p1[m], pf[m], p3[m], y[m]

    def ll(a, b):
        b = np.clip(b, EPS, 1 - EPS)
        return -np.mean(a * np.log(b) + (1 - a) * np.log(1 - b))

    print(f"\nFLOOR-GAP (size {args.size}) source={src} (linear heads) | "
          f"n_eval {m.sum()} | base rate {yy.mean():.3f}")
    print(f"  overall LogLoss  1PL {ll(yy,p1):.4f}  +floor {ll(yy,pf):.4f}  "
          f"3PL {ll(yy,p3):.4f}")
    # The floor's reach: how low does each model predict? (lift on hard-for-user probs)
    qs = [0.1, 1, 5, 50]
    print("  predicted-P percentiles (the floor lifts the low tail):")
    print("     pct " + " ".join(f"{q:>7}" for q in qs))
    print("     1PL " + " ".join(f"{np.percentile(p1,q):>7.3f}" for q in qs))
    print("   +floor " + " ".join(f"{np.percentile(pf,q):>7.3f}" for q in qs))
    frac_hard = float((p1 < 0.5).mean())
    print(f"  share of held-out attempts with 1PL P<0.5 (problem harder than user): "
          f"{frac_hard*100:.1f}%   P<0.35: {float((p1<0.35).mean())*100:.1f}%")

    # FIXED predicted-P bins so the per-bin COUNT shows the distribution directly.
    edges = [0.0, 0.45, 0.60, 0.70, 0.80, 0.88, 0.94, 1.0001]
    print("\n  Binned by 1PL predicted P (low P = problem hard for user → where a floor acts):")
    print(f"  {'P bin':>12s} {'n':>7s} {'share':>6s} {'actual':>7s} "
          f"{'1PL':>8s} {'+floor':>8s} {'3PL':>8s} {'Δfloor':>8s} {'Δ3PL':>8s}")
    print("  " + "-" * 86)
    for i in range(len(edges) - 1):
        sel = (p1 >= edges[i]) & (p1 < edges[i + 1])
        if sel.sum() < 50:
            continue
        a = yy[sel]
        l1, lf, l3 = ll(a, p1[sel]), ll(a, pf[sel]), ll(a, p3[sel])
        print(f"  {edges[i]:.2f}..{edges[i+1] if edges[i+1]<=1 else 1.0:.2f} "
              f"{int(sel.sum()):>7d} {sel.mean()*100:>5.1f}% {a.mean():>7.3f} "
              f"{l1:>8.4f} {lf:>8.4f} {l3:>8.4f} {lf-l1:>+8.4f} {l3-l1:>+8.4f}")


def head3pl_grid(prof_idx, feats, y, groups, n_profiles, args, device):
    """Factorial: per source, {lin, mlp} × {3PL off, on}, MLP multi-seeded.

    Isolates the two questions directly:
      - does the MLP head add value?  -> compare lin vs mlp within a (source, 3PL) row
      - does 3PL add value?           -> compare 3PL off vs on within a (source, head) col
    3PL off = σ(θ+d) (floor=none, slope=off); 3PL on = feature floor + slope.
    Linear cells are seed-invariant (zero-init, fixed folds) so single-run; MLP
    cells are mean±std over --seeds. Native width (no PCA bottleneck)."""
    from sklearn.metrics import log_loss, roc_auc_score
    dh = tuple(args.diff_hidden)

    def run(fmat, floor, slope, hidden, seed):
        oof = fit_eval(prof_idx, fmat, y, groups, n_profiles, floor, args.folds,
                       args.epochs, args.lr, args.ridge, device, seed,
                       slope=slope, diff_hidden=hidden, dropout=args.dropout,
                       mlp_wd=args.mlp_wd)
        m = ~np.isnan(oof)
        return (log_loss(y[m], np.clip(oof[m], EPS, 1 - EPS)),
                roc_auc_score(y[m], oof[m]))

    print(f"\nHEAD×3PL GRID (size {args.size}) | MLP {dh} dropout {args.dropout} | "
          f"{args.seeds} seeds | base-rate LL "
          f"{-(lambda b: b*np.log(b)+(1-b)*np.log(1-b))(float(y.mean())):.4f}")
    hdr = (f"  {'source':>18s} {'3PL':>4s} {'lin LL':>8s} {'lin AUC':>8s} "
           f"{'mlp LogLoss(mean±std)':>23s} {'mlp AUC':>8s} {'mlp−lin':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for src in args.matrix_feats:
        if src not in feats or feats[src] is None:
            continue
        fmat = feats[src]
        for tpl in ("off", "on"):
            floor = "feature" if tpl == "on" else "none"
            slope = (tpl == "on")
            ll_l, auc_l = run(fmat, floor, slope, (), args.seed)
            lls, aucs = [], []
            for s in range(args.seeds):
                ll_m, auc_m = run(fmat, floor, slope, dh, s)
                lls.append(ll_m); aucs.append(auc_m)
            print(f"  {src:>18s} {tpl:>4s} {ll_l:>8.4f} {auc_l:>8.4f} "
                  f"{np.mean(lls):>15.4f}±{np.std(lls):<7.4f} {np.mean(aucs):>8.4f} "
                  f"{np.mean(lls)-ll_l:>+8.4f}")
        print()


def nonlinear_compare(prof_idx, feats, y, groups, n_profiles, prof_df, rating,
                      args, device):
    """Linear vs MLP activation difficulty on identical folds, multi-seed.

    floor=none, slope=off → isolates the difficulty non-linearity. The linear
    head is seed-invariant (zero init, fixed folds), so its single number is the
    bar; the MLP is run over `--seeds` seeds and reported as mean ± std — if the
    mean doesn't clear the linear bar by more than the std, non-linearity isn't
    buying reliable generalisation (the likely outcome given rank/difficulty have
    repeatedly probed as linear)."""
    from sklearn.metrics import log_loss, roc_auc_score
    fl = args.nl_floor
    print(f"\nNon-linear difficulty test (MLP {args.diff_hidden}, dropout "
          f"{args.dropout}; floor={fl}, slope=off; {args.seeds} seeds):")
    print(f"  {'model':>14s} {'LogLoss(mean±std)':>22s} {'AUC':>7s}")
    print("  " + "-" * 46)
    oof_for_bins = {}
    # bot_rating linear reference (the incumbent), so this block is self-contained
    # even with --no-floor-sweep.
    ob = fit_eval(prof_idx, feats["bot_rating"], y, groups, n_profiles, "none",
                  args.folds, args.epochs, args.lr, args.ridge, device, args.seed)
    mb = ~np.isnan(ob)
    print(f"  {'bot_rating':>14s} {log_loss(y[mb], np.clip(ob[mb], EPS, 1-EPS)):>16.4f}"
          f"{'':>6s} {roc_auc_score(y[mb], ob[mb]):>7.4f}")
    oof_for_bins["bot_rating"] = ob
    for fname in ("activations", "both"):
        fmat = feats[fname]
        # linear baseline (seed-invariant)
        ol = fit_eval(prof_idx, fmat, y, groups, n_profiles, fl, args.folds,
                      args.epochs, args.lr, args.ridge, device, args.seed)
        ml = ~np.isnan(ol)
        ll_l = log_loss(y[ml], np.clip(ol[ml], EPS, 1 - EPS))
        auc_l = roc_auc_score(y[ml], ol[ml])
        print(f"  {fname+' (lin)':>14s} {ll_l:>16.4f}{'':>6s} {auc_l:>7.4f}")
        oof_for_bins[fname] = ol  # linear = the reference in the bin breakdown
        # MLP across seeds
        lls, aucs, last = [], [], None
        for s in range(args.seeds):
            om = fit_eval(prof_idx, fmat, y, groups, n_profiles, fl,
                          args.folds, args.epochs, args.lr, args.ridge, device,
                          s, diff_hidden=tuple(args.diff_hidden),
                          dropout=args.dropout, mlp_wd=args.mlp_wd)
            mm = ~np.isnan(om)
            lls.append(log_loss(y[mm], np.clip(om[mm], EPS, 1 - EPS)))
            aucs.append(roc_auc_score(y[mm], om[mm]))
            last = om
        print(f"  {fname+' (mlp)':>14s} {np.mean(lls):>16.4f}±{np.std(lls):<5.4f} "
              f"{np.mean(aucs):>7.4f}   [seed LLs: "
              + " ".join(f"{x:.4f}" for x in lls) + "]")
        oof_for_bins[fname + "_mlp"] = last
    print()
    # Where does the MLP change the error profile vs linear act and vs the bot rating?
    bin_models = {"bot_rating": oof_for_bins["bot_rating"],
                  "activations": oof_for_bins["activations"],
                  "activations_mlp": oof_for_bins["activations_mlp"]}
    report_by_rating(prof_df, rating, y, bin_models, args.bins)


def report_by_rating(prof_df, rating, y, oof_none, n_bins):
    """Where does each model's error live across the rating range?

    Bins held-out attempts by the BOT rating (a neutral external axis — the
    activation model never sees it) into equal-count quantile bins, then reports
    each model's held-out log-loss within the bin. An even error profile is
    preferable to one spiking at a tail, so the last column flags which model is
    worst in each bin. Absolute log-loss falls at the easy tail (base rate→1), so
    also shown is each model's log-loss MINUS the bin base-rate log-loss (skill
    over base) — the tail-fair comparison."""
    raw = prof_df["problem"].map(rating).to_numpy(float)
    models = [m for m in ("bot_rating", "activations", "activations_mlp", "both")
              if m in oof_none] or list(oof_none)
    valid = ~np.isnan(oof_none[models[0]])  # common eval mask (same folds)
    edges = np.quantile(raw[valid], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6

    def ll_of(yy, pp):
        pp = np.clip(pp, EPS, 1 - EPS)
        return -np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

    print("Held-out log-loss by BOT-RATING range (easy→hard); "
          "Δskill = log-loss − bin base-rate log-loss (lower=better, <0=beats base):")
    def short(m):
        return (m.replace("bot_rating", "bot").replace("activations", "act")
                 .replace("_mlp", "·mlp"))
    head = (f"  {'rating bin':>16s} {'n':>7s} {'base':>6s}"
            + "".join(f"{short(m):>10s}" for m in models)
            + "".join(f"{'Δ' + short(m):>10s}" for m in models) + f"  {'worst':>11s}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i in range(n_bins):
        sel = valid & (raw >= edges[i]) & (raw < edges[i + 1])
        if sel.sum() < 50:
            continue
        yy = y[sel]
        base = yy.mean()
        base_ll = -(base * np.log(max(base, EPS)) + (1 - base) * np.log(max(1 - base, EPS)))
        lls = {m: ll_of(yy, oof_none[m][sel]) for m in models}
        worst = max(lls, key=lls.get)
        row = (f"  {edges[i]:+6.1f}..{edges[i+1]:+6.1f} {int(sel.sum()):>7d} {base:>6.2f}"
               + "".join(f"{lls[m]:>10.4f}" for m in models)
               + "".join(f"{lls[m]-base_ll:>+10.4f}" for m in models)
               + f"  {short(worst):>11s}")
        print(row)


if __name__ == "__main__":
    main()
