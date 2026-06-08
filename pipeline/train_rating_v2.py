"""Cross-board anchored feature→4PL rating model — the rating pipeline.

ONE model for 9x9 AND 13x13. A SHARED feature→4PL network maps each problem's features
to (difficulty b_p, slope a_p, floor c_p); per-(user, 2-month) ability θ_u is hard-gauged
to a unit Gaussian; a single GLOBAL ceiling (fat-finger / lapse) λ is learned.

    P(correct) = c_p + (1 − c_p − λ)·σ( a_p·(θ_u − b_p) )
    (b_p, a_p, c_p) = net(feat_p)        # shared net; a_p ≥ 0.01, c_p ∈ [0, 1 − 2λ]
    feat_p = joint-PCA(correct-move activations, both boards) ++ common-column bot winrate (bw_act)

There are NO per-problem residuals: the shared net IS the model. Every problem — including
the never-attempted and the one-sided — gets its 4PL straight from features. Two humanv0 bot
"users" (default 20k & 9d) join as extra examinees with LEARNED ability and SOFT-target
(pass-rate) responses on EVERY problem; that coverage anchors difficulty AND slope on the
empty / one-sided regions the sparse, ~80%-clear-served human attempts cannot identify
(two strengths ⇒ two points on each problem's curve). Validated to match the previous
residual+DRO pipeline's survey-rank accuracy with none of the per-problem human fitting —
see auxiliary/test_lapse.py (research sandbox) and auxiliary/train_rating_v2_legacy.py
(the preserved residual+DRO+fixed-bot-floor model, for future development).

Ability is hard-gauged (θ whitened to mean 0 / std 1 each step): the population genuinely
follows N(0, 1), and pinning the scale stops slopes inflating to compensate for compressed
ability. The shipped RATING is the ability at which the user clears the problem 80% of the
time (CALIBRATION_P) — NOT the 50%-point — so a flat low-discrimination problem correctly
needs more ability to clear at 80% than its midpoint implies:

    rating_p = b_p + logit((0.8 − c_p) / (1 − c_p − λ)) / a_p

The global ceiling λ is produced HERE (written to ratings_v2/rating_constants.json);
pipeline.fit_population_constants reads it through and no longer fits it posthoc.

Outputs (parallel; nothing in pwa/tsumegos/ is touched — package_for_pwa ships these):
  ratings_v2/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= 80%-clear rating)
  ratings_v2/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= raw slope a_p)
  ratings_v2/<size>/tsumego_floor_final.csv     tsumego_id, floor    (= learned c_p)
  ratings_v2/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings_v2/rating_constants.json              {"fat_finger_rate": λ}  (the learned ceiling)
  ratings_v2/model_v2.pt                        weights + run metadata

The defaults reproduce the production result; the knobs are for experiments only.

Usage:
  ./.venv/bin/python3 -m pipeline.train_rating_v2 [--epochs 300] [--anchor-weight 0.2] ...
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

# NOTE (prototype tech-debt): these live in auxiliary/ today. When this backend stabilizes,
# promote load_joint_*/build_profiles + the activation-cache helpers into utils/.
from auxiliary.compare_rating_models import (  # noqa: E402
    build_profiles, load_joint_activations, load_joint_bot_winrate,
)
from auxiliary.inspect_activation_signals import _cache_ids_path  # noqa: E402
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402
from utils.fit_rank_curves import CALIBRATION_P  # noqa: E402

EPS = 1e-6
SIZES = [9, 13]
# logit of the serving/anchoring success rate (80%): rating = the ability θ at which
# P(clear) = CALIBRATION_P. Single source of truth shared with the PWA + map_rating_v2_to_egf.
CLEAR_LOGIT = float(np.log(CALIBRATION_P / (1.0 - CALIBRATION_P)))


def clear_logit_vec(c, lapse=0.0):
    """Per-problem 80%-clear logit anchor under the 4PL: at the rating P(clear)=CALIBRATION_P, so
    σ(a(θ−b))=(P*−c)/(1−c−λ) and rating = b + logit((P*−c)/(1−c−λ))/a. Vector over problems;
    reduces to the scalar CLEAR_LOGIT where c=0 and lapse=0. `lapse` is the global 4PL slip rate;
    it must match the client's clearKappa so the shipped rating is the true 80%-clear point."""
    c = np.asarray(c, dtype=np.float64)
    q = np.clip((CALIBRATION_P - c) / (1.0 - c - lapse), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def load_anchor_winrates(keys, ranks):
    """Per-problem humanv0 pass-rate for each anchor rank, aligned to `keys` (the model's problem
    order). Returns float (n_keys, n_ranks), NaN where a rank lacks a value for that problem. These
    are the soft targets for the bot 'users': value = P(the rank-conditioned humanv0 bot plays the
    correct first move) — the same first-move-correct semantics as a human attempt."""
    out = np.full((len(keys), len(ranks)), np.nan, dtype=np.float32)
    key_row = {k: i for i, k in enumerate(keys)}
    for sz in sorted({s for s, _ in keys}):
        adir = os.path.join(PROJECT_ROOT, "attempts", str(sz))
        for j, r in enumerate(ranks):
            fn = os.path.join(adir, f"b18c384nbt-humanv0_{r}.txt")
            if not os.path.exists(fn):
                continue
            with open(fn) as fh:
                for line in fh:
                    pp = line.split()
                    if len(pp) != 2:
                        continue
                    row = key_row.get((sz, pp[0]))
                    if row is None:
                        continue
                    try:
                        out[row, j] = float(pp[1])
                    except ValueError:
                        pass
    return out


def survey_weighted_rho(ratemap, slopemap, min_attempts=20, min_server=4):
    """Composite survey objective (calibration to real ranks). For each survey user, fit their MLE
    80%-clear rating (fit_user_rating over their attempts with these per-problem ratings/slopes); then,
    PER SERVER, the Spearman against the declared rank, and the n-weighted average over servers.

    Per-server Spearman is offset- AND scale-invariant, so it is robust to server-population imbalance
    (EGF skews strong, OGS skews weak) — unlike mean-centered pooling, which only removes a server's
    offset and still mixes their differing within-server spread. The surveys are NEVER used in
    training; they enter only here, as an external validation signal. Returns (weighted_rho,
    {server: (n, rho)})."""
    from collections import defaultdict
    from auxiliary.compare_rank_surveys import (
        load_rank_surveys, latest_per_user_server, load_attempts_for_users, SQL_PATH_DEFAULT)
    from utils.fit_rank_curves import fit_user_rating
    surveys = latest_per_user_server(load_rank_surveys(SQL_PATH_DEFAULT))
    attempts = load_attempts_for_users(SQL_PATH_DEFAULT, set(surveys["user_id"].unique()))
    user_rating = {}
    for uid in surveys["user_id"].unique():
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((uid[:8], s), []))
        if len(pooled) < min_attempts:
            continue
        R, nj = fit_user_rating(pooled, ratemap, min_n=min_attempts, slopes=slopemap)
        if R is not None and nj >= min_attempts:
            user_rating[uid[:8]] = R
    serv = defaultdict(list)
    for _, row in surveys.iterrows():
        g = row["rank_scalar"]; pref = str(row["user_id"])[:8]
        if g is None or (isinstance(g, float) and np.isnan(g)) or pref not in user_rating:
            continue
        serv[row["server"]].append((user_rating[pref], float(g)))

    def spear(pairs):
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
    per = {s: (len(p), spear(p)) for s, p in serv.items() if len(p) >= min_server}
    if not per:
        return float("nan"), {}
    wn = sum(n for n, _ in per.values())
    return sum(n * r for n, r in per.values()) / wn, per


class RatingNet(nn.Module):
    """Shared feature→4PL net + per-user ability (hard unit gauge) + one global learned ceiling λ.

        P = c + (1 − c − λ)·σ( a·(θ_u − b) ),  (b, a, c) = net(feat)

    The shared net has NO per-problem parameters, so every problem's 4PL comes from its features —
    a zero-interaction problem still gets a prediction. a = softplus(·).clamp(min=0.01) (lower bound
    only, no upper cap); the floor is smoothly bounded to [0, 1 − 2λ] (ceiling-coupled: it stays a
    margin λ below the 1 − λ upper asymptote, so the active band 1 − c − λ ≥ λ > 0)."""

    def __init__(self, fdim, n_users, hidden=(128, 64), dropout=0.1, lapse_cap=0.05, lapse_init=0.01,
                 slope_floor=0.01):
        super().__init__()
        self.slope_floor = float(slope_floor)
        self.theta = nn.Embedding(n_users, 1)
        nn.init.normal_(self.theta.weight, std=1.0)
        layers, d = [], fdim
        for h in hidden:                                       # decreasing layers, ReLU between
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 3)]                            # → (difficulty b, slope a, floor c)
        self.net = nn.Sequential(*layers)
        self.lapse_cap = float(lapse_cap)
        li = float(np.clip(lapse_init / max(lapse_cap, EPS), EPS, 1 - EPS))
        self.lapse_raw = nn.Parameter(torch.tensor(float(np.log(li / (1 - li)))))

    def lapse(self):
        """Global 4PL upper-asymptote slip rate λ ∈ (0, lapse_cap)."""
        return self.lapse_cap * torch.sigmoid(self.lapse_raw)

    def params(self, feat):
        o = self.net(feat)
        b = o[:, 0]
        a = torch.nn.functional.softplus(o[:, 1]).clamp(min=self.slope_floor)  # lower bound only
        cap = (1.0 - 2.0 * self.lapse()).clamp(min=EPS)
        c = cap * torch.sigmoid(o[:, 2])
        return b, a, c

    def ability(self, u):
        """Per-user ability, hard-whitened to mean 0 / std 1 each call (unit-Gaussian gauge)."""
        w = self.theta.weight
        return ((self.theta(u) - w.mean()) / (w.std() + EPS)).squeeze(-1)

    def forward(self, u, feat):
        b, a, c = self.params(feat)
        praw = torch.sigmoid(a * (self.ability(u) - b))
        return (c + (1.0 - c - self.lapse()) * praw).clamp(EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "ratings_v2"))
    ap.add_argument("--pca-dim", type=int, default=128, help="Joint activation PCs.")
    ap.add_argument("--diff-hidden", type=int, nargs="*", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--min", type=int, default=5,
                    help="Min first-time correct AND incorrect to keep a user.")
    ap.add_argument("--min-profile", type=int, default=8)
    ap.add_argument("--winsor", type=float, default=1.0,
                    help="Winsorize final ratings to [p, 100-p] percentile (legacy robustness for "
                         "the low-slope b + logit/a tail on the unattempted extrapolation).")
    ap.add_argument("--slope-floor", type=float, default=0.2,
                    help="Lower bound on the per-problem slope a_p in the MODEL (softplus floor) — the "
                         "real knob behind the low-slope rating tail, tuned by the objectives, NOT by "
                         "how the rating histogram looks. Default 0.2 is the compromise across seeds: "
                         "weighted survey ρ ≈ 0.80 (matches legacy 0.802, vs 0.792 at 0.3), held-out-"
                         "problem logloss ≈ the 0.01 level (0.3 is ~0.005 better there), and the "
                         "positive rating tail stays sane (~14). 0.01 overfits flat slopes AND blows "
                         "the rating to ~229; the bound is thus a consequence of the objective, not a cap.")
    ap.add_argument("--holdout-problem-frac", type=float, default=0.0,
                    help="If >0, hold out this fraction of ATTEMPTED problems (and their anchor "
                         "rows) from training and report held-out-PROBLEM logloss — the feature "
                         "prior's generalization to unseen problems. Eval mode: skips writing CSVs.")
    ap.add_argument("--anchors", action=argparse.BooleanOptionalAction, default=True,
                    help="Add humanv0 bot 'users' (--anchor-ranks) as extra examinees with learned "
                         "ability + soft-target responses on every problem (coverage prior). On by "
                         "default; --no-anchors trains on human attempts only.")
    ap.add_argument("--anchor-ranks", default="20k,9d",
                    help="comma-separated humanv0 ranks used as bot users (default 20k,9d).")
    ap.add_argument("--anchor-weight", type=float, default=0.2,
                    help="per-row weight on bot-anchor responses (human rows = 1.0). Low = minimal "
                         "prior: humans own the shape where attested, bots fill the empty regions.")
    ap.add_argument("--lapse-cap", type=float, default=0.05,
                    help="Upper bound on the learned global ceiling λ.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- features (joint-PCA correct-move activations ⊕ common-column bot winrate) ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]                  # (size, tid); scorable iff both blocks
    key_idx = {key: i for i, key in enumerate(keys)}
    fdim = k + nb + 1                                         # + size_is_13 board indicator
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        feat[i] = np.concatenate([act[key], bw[key], [1.0 if key[0] == 13 else 0.0]])
    print(f"scorable problems {len(keys)} (9:{sum(s == 9 for s, _ in keys)} "
          f"13:{sum(s == 13 for s, _ in keys)}) | feat dim {fdim} (act {k} + bot {nb} + size_is_13)")

    # --- HUMAN first-time attempts, pooled, shared profiles (no size in profile id) ---
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    df = df[df["user_id"].isin(
        set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)]))].copy()
    prob2size = dict(zip(df["problem"], df["size"]))
    prof_df = build_profiles(df)                              # profile, problem, success
    pc = prof_df["profile"].value_counts()
    prof_df = prof_df[prof_df["profile"].isin(set(pc.index[pc >= args.min_profile]))]
    prof_df = prof_df[prof_df["problem"].map(
        lambda t: (prob2size.get(t), t) in key_idx)].reset_index(drop=True)
    profiles = pd.Categorical(prof_df["profile"])
    prof_idx = profiles.codes.astype(np.int64)
    n_profiles = len(profiles.categories)
    y = prof_df["success"].to_numpy(np.float32)
    prob_idx = prof_df["problem"].map(
        lambda t: key_idx[(prob2size[t], t)]).to_numpy(np.int64)
    print(f"training: {len(prof_df)} human attempt-rows | profiles {n_profiles} | "
          f"attempted problems {prof_df['problem'].nunique()} | base rate {y.mean():.3f} "
          f"| device {device.type}")

    # --- optional held-out PROBLEM set: the feature prior's generalization to UNSEEN problems
    #     (held-out problems are excluded from BOTH human and anchor training rows, so the model
    #     has zero signal on them — its b/a/c there come purely from features). ---
    ho_mask_h = np.zeros(len(y), bool)
    ho_problems = set()
    if args.holdout_problem_frac > 0:
        att_probs = np.unique(prob_idx)
        rng = np.random.default_rng(args.seed)
        n_ho = int(round(args.holdout_problem_frac * len(att_probs)))
        ho_problems = set(rng.choice(att_probs, size=n_ho, replace=False).tolist())
        ho_mask_h = np.isin(prob_idx, list(ho_problems))
        print(f"HELD-OUT eval: {len(ho_problems)}/{len(att_probs)} attempted problems held out "
              f"({int(ho_mask_h.sum())} human rows) — feature-prior generalization test")

    # --- bot 'anchor users': extra examinees (learned ability) with soft-target responses on every
    #     problem. Coverage is the prior; two strengths ⇒ b AND a identified everywhere. ---
    n_anchor, ranks = 0, []
    au = ap_ = at = np.empty(0)
    if args.anchors:
        ranks = [r.strip() for r in args.anchor_ranks.split(",") if r.strip()]
        aw = load_anchor_winrates(keys, ranks)
        n_anchor = len(ranks)
        uu, pp_, tt_ = [], [], []
        for j in range(n_anchor):
            ok = np.where(np.isfinite(aw[:, j]))[0]
            uu.append(np.full(len(ok), n_profiles + j, np.int64))
            pp_.append(ok.astype(np.int64))
            tt_.append(aw[ok, j].astype(np.float32))
        au, ap_, at = np.concatenate(uu), np.concatenate(pp_), np.concatenate(tt_)
        cover = [int(np.isfinite(aw[:, j]).sum()) for j in range(n_anchor)]
        print(f"anchors: bot users {ranks} | coverage {cover}/{len(keys)} | "
              f"row-weight {args.anchor_weight} | abilities LEARNED")

    # by-row val (only in eval mode): among the kept human rows, hold out 15% of ATTEMPTS on
    # SEEN problems — generalization to new attempts, distinct from the new-problem held-out set.
    byrow_val = np.zeros(len(y), bool)
    if args.holdout_problem_frac > 0:
        kept_idx = np.where(~ho_mask_h)[0]
        rng2 = np.random.default_rng(args.seed + 1)
        sel = rng2.choice(len(kept_idx), size=int(0.15 * len(kept_idx)), replace=False)
        byrow_val[kept_idx[sel]] = True

    # --- assemble training rows: human (w=1) ⊕ anchor (w=anchor_weight), MINUS held-out problems
    #     and the by-row val rows ---
    train_h = (~ho_mask_h) & (~byrow_val)
    u_all = [prof_idx[train_h]]; p_all = [prob_idx[train_h]]; t_all = [y[train_h]]
    w_all = [np.ones(int(train_h.sum()), np.float32)]
    if n_anchor:
        keep_a = ~np.isin(ap_, list(ho_problems)) if ho_problems else np.ones(len(ap_), bool)
        u_all.append(au[keep_a]); p_all.append(ap_[keep_a]); t_all.append(at[keep_a])
        w_all.append(np.full(int(keep_a.sum()), args.anchor_weight, np.float32))
    u_all = np.concatenate(u_all); p_all = np.concatenate(p_all)
    t_all = np.concatenate(t_all); w_all = np.concatenate(w_all)

    tfeat = torch.tensor(feat, device=device)
    tu = torch.tensor(u_all, device=device)
    tp = torch.tensor(p_all, device=device)
    tt = torch.tensor(t_all, device=device)
    tw = torch.tensor(w_all, device=device)
    thu = torch.tensor(prof_idx, device=device)              # human-only, for the train-BCE trace
    thp = torch.tensor(prob_idx, device=device)
    thy = torch.tensor(y, device=device)
    n = len(u_all)

    # --- fit ---
    torch.manual_seed(args.seed)
    m = RatingNet(fdim, n_profiles + n_anchor, tuple(args.diff_hidden),
                  dropout=args.dropout, lapse_cap=args.lapse_cap,
                  slope_floor=args.slope_floor).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    epoch_records = []                                        # (pass, epoch, human_train_bce, λ)
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(n, device=device)
        for b0 in range(0, n, args.batch):
            bi = perm[b0:b0 + args.batch]
            opt.zero_grad()
            p = m(tu[bi], tfeat[tp[bi]])
            bce = torch.nn.functional.binary_cross_entropy(p, tt[bi], reduction="none")
            loss = (bce * tw[bi]).sum() / tw[bi].sum().clamp_min(EPS)
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            ll = torch.nn.functional.binary_cross_entropy(m(thu, tfeat[thp]), thy).item()
        lam_ep = float(m.lapse())
        epoch_records.append((1, ep, ll, lam_ep))
        if ep % args.log_every == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:4d}  human train BCE {ll:.4f}  λ {lam_ep:.5f}")

    # --- score the whole corpus ---
    m.eval()
    with torch.no_grad():
        b_all, a_all, c_all = m.params(tfeat)
        b_arr = b_all.cpu().numpy(); a = a_all.cpu().numpy(); c = c_all.cpu().numpy()
        theta_full = m.ability(torch.arange(n_profiles + n_anchor, device=device)).cpu().numpy()
        lam = float(m.lapse())
    theta = theta_full[:n_profiles]
    bot_theta = theta_full[n_profiles:]
    rating = b_arr + clear_logit_vec(c, lam) / a             # 80%-clear ability under the 4PL
    lo, hi = np.percentile(rating, [args.winsor, 100 - args.winsor])
    rating = np.clip(rating, lo, hi)                          # percentile winsor (legacy robustness)
    print(f"  λ {lam:.5f} | floor mean {c.mean():.3f} p[5,50,95] "
          f"{np.percentile(c, [5, 50, 95]).round(3)} | slope p[1,50,99] "
          f"{np.percentile(a, [1, 50, 99]).round(2)} | rating winsor [{lo:.2f},{hi:.2f}] "
          f"| ability std {theta.std():.3f}")
    if n_anchor:
        print("  bot-user learned ability θ: "
              + ", ".join(f"{ranks[j]}={bot_theta[j]:+.2f}" for j in range(n_anchor)))

    if args.holdout_problem_frac > 0:
        # --- objective eval (no CSVs written): three independent objectives ---
        def _ll(mask):
            with torch.no_grad():
                pp = m(torch.tensor(prof_idx[mask], device=device),
                       tfeat[torch.tensor(prob_idx[mask], device=device)]).cpu().numpy()
            yy = y[mask]; pp = np.clip(pp, EPS, 1 - EPS)
            return float(-(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)).mean())
        byrow_ll = _ll(byrow_val)                    # new attempts on SEEN problems
        ho_ll = _ll(ho_mask_h)                       # new (UNSEEN) problems — feature-prior transfer
        ratemap = {keys[i][1]: float(rating[i]) for i in range(len(keys))}
        slopemap = {keys[i][1]: float(a[i]) for i in range(len(keys))}
        wrho, per = survey_weighted_rho(ratemap, slopemap)
        print(f"\n=== OBJECTIVES | slope_floor {args.slope_floor} | feat {fdim} | "
              f"holdout {args.holdout_problem_frac} | seed {args.seed} ===")
        print(f"  1. by-row val logloss      : {byrow_ll:.4f}  "
              f"({int(byrow_val.sum())} held-out attempts, seen problems)")
        print(f"  2. held-out-PROBLEM logloss: {ho_ll:.4f}  "
              f"({int(ho_mask_h.sum())} attempts on {len(ho_problems)} UNSEEN problems)")
        print(f"  3. survey rho (weighted)   : {wrho:.4f}  "
              f"(per-server Spearman, n-weighted: "
              + " ".join(f"{s}:{r:.2f}(n{n})" for s, (n, r) in sorted(per.items(), key=lambda kv: -kv[1][0]))
              + ")")
        return

    # --- write outputs ---
    os.makedirs(args.out_dir, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(args.out_dir, str(s))
        os.makedirs(sdir, exist_ok=True)
        pd.DataFrame({"tsumego_id": tids, "rating": rating[sel]}).sort_values(
            "rating", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_ratings_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "quality": a[sel]}).sort_values(
            "quality", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_quality_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "floor": c[sel]}).sort_values(
            "floor", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_floor_final.csv"), index=False)
        print(f"  size {s}: wrote {len(tids)} ratings | rating "
              f"[{rating[sel].min():.2f},{rating[sel].max():.2f}] slope [{a[sel].min():.2f},"
              f"{a[sel].max():.2f}] | floor mean {c[sel].mean():.3f}")

    natt = pd.Series(prof_idx).value_counts().sort_index()
    pd.DataFrame({"profile": list(profiles.categories), "theta": theta,
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_profiles)]}).to_csv(
        os.path.join(args.out_dir, "user_theta.csv"), index=False)
    # The learned global ceiling λ — produced HERE; fit_population_constants reads it through.
    with open(os.path.join(args.out_dir, "rating_constants.json"), "w") as f:
        json.dump({"fat_finger_rate": round(lam, 4)}, f, indent=2)
    torch.save({"state_dict": m.state_dict(), "keys": keys, "fdim": fdim, "k": k,
                "n_bot_cols": nb, "n_anchor": n_anchor, "anchor_ranks": ranks,
                "lapse": lam, "args": vars(args)},
               os.path.join(args.out_dir, "model_v2.pt"))
    pd.DataFrame(epoch_records, columns=["pass", "epoch", "train_bce", "lapse"]).to_csv(
        os.path.join(args.out_dir, "epoch_log.csv"), index=False)
    print(f"wrote user_theta.csv ({n_profiles} profiles) + rating_constants.json "
          f"(fat_finger_rate λ={lam:.5f}) + model_v2.pt to {args.out_dir}")


if __name__ == "__main__":
    main()
