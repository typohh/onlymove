"""Combined cross-board MLP-2PL difficulty model — the v2 rating pipeline.

PARALLEL to pipeline.rank_tsumegos_and_bots: it writes only under ratings_v2/ and
touches nothing the current backend reads. Eventual drop-in replacement (output format
matches what package_for_pwa + the PWA expect, see below), so no frontend changes.

ONE model for 9x9 AND 13x13:

    P(correct) = sigmoid( a_p * theta_u + d_p )
    d_p = diff_head(feat_p) + delta_p              # feature PRIOR + per-problem residual
    a_p = softplus(b0 + slope_head(feat_p) + eta_p)  # feature-prior slope + residual

  - feat_p = joint-PCA(correct-move activations, both boards) ++ common-column bot winrate
    (bw_act). The bot data enters ONLY through these features, so the feature heads are a
    bot+activation PRIOR; the HUMAN first-time attempts (is_repeat=0) are the training
    target and add per-problem residuals delta_p / eta_p ON TOP of the prior. Ridge on the
    residuals = empirical-Bayes shrinkage: sparsely / never-attempted problems stay at the
    feature prior (delta=eta=0), human-attempted problems pull away from it.
  - theta_u is a single shared per-(user, 2-month) ability across boards (one skill).

The 2PL SLOPE a_p replaces the old `quality`: a high-discrimination (sharp) problem is the
"high quality" one the selector should up-weight. The RATING that maps to rank is the
ability at which the user CLEARS the problem 80% of the time (CALIBRATION_P) — the same
margin the PWA serves and anchors at, NOT the 50%-point:

    rating_p = (logit(0.8) - d_p) / a_p  =  b_p + logit(0.8)/a_p   (b_p = -d_p/a_p = 50%-pt)

The slope-dependent 80% correction is load-bearing: under it a user clears exactly 80% on
problems whose rating == their ability theta for EVERY problem regardless of slope (a flat
low-discrimination problem genuinely needs more ability to clear at 80% than its 50%-point
implies). A monotone rating->rank map cannot recover this per-problem reordering, so it has
to be built here. The EGF mapping (pipeline.map_rating_v2_to_egf) then anchors each user's
80%-clear rating -> EGF against these ratings (self-consistent — it re-derives the curve
from whatever scalar we ship).

Every corpus problem is scorable (bots benchmark all, activations are cached for all), so
this assigns a rating to the whole set; unattempted problems fall back to the prior.

Outputs (parallel; nothing in ratings/ or pwa/tsumegos/ is touched):
  ratings_v2/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= b_p)
  ratings_v2/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= raw slope a_p; package
                                                normalizes to [0,1] as it already does)
  ratings_v2/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings_v2/model_v2.pt                        weights + run metadata

Usage:
  ./.venv/bin/python3 -m pipeline.train_rating_v2 [--epochs 600] [--pca-dim 128] ...
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

# NOTE (prototype tech-debt): these live in auxiliary/ today. When v2 becomes the main
# backend, promote load_joint_*/build_profiles + the activation-cache helpers into utils/.
from auxiliary.compare_rating_models import (  # noqa: E402
    build_profiles, load_joint_activations, load_joint_bot_winrate,
)
from auxiliary.inspect_activation_signals import _cache_ids_path  # noqa: E402
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402
from pipeline.fit_rank_curves import CALIBRATION_P  # noqa: E402

EPS = 1e-6
SIZES = [9, 13]
# logit of the serving/anchoring success rate (80%): the rating is the ability θ at which
# P(clear) = CALIBRATION_P, i.e. (CLEAR_LOGIT − d)/a, NOT the 50%-point −d/a. Single source
# of truth shared with the PWA + map_rating_v2_to_egf via fit_rank_curves.CALIBRATION_P.
CLEAR_LOGIT = float(np.log(CALIBRATION_P / (1.0 - CALIBRATION_P)))


class RatingV2(nn.Module):
    """P = sigmoid(a_p * theta_u + d_p); d_p / a_p = feature prior + per-problem residual."""

    def __init__(self, n_profiles, n_problems, fdim, diff_hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.theta = nn.Embedding(n_profiles, 1)
        nn.init.zeros_(self.theta.weight)
        layers, d = [], fdim
        for h in diff_hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        head = nn.Linear(d, 1)              # bias = global difficulty offset
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.diff_head = nn.Sequential(*layers, head)
        self.slope_head = nn.Linear(fdim, 1, bias=False)
        nn.init.zeros_(self.slope_head.weight)
        self.register_buffer("slope_bias", torch.tensor(0.5413))  # softplus(.5413)≈1
        self.delta = nn.Embedding(n_problems, 1)   # per-problem difficulty residual
        self.eta = nn.Embedding(n_problems, 1)     # per-problem slope residual
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.eta.weight)

    def diff_slope(self, prob_idx, feat):
        d = self.diff_head(feat).squeeze(-1) + self.delta(prob_idx).squeeze(-1)
        a = torch.nn.functional.softplus(
            self.slope_bias + self.slope_head(feat).squeeze(-1)
            + self.eta(prob_idx).squeeze(-1)).clamp(0.2, 12.0)  # floor avoids b=-d/a blowup
        return d, a

    def forward(self, prof_idx, prob_idx, feat):
        d, a = self.diff_slope(prob_idx, feat)
        return torch.sigmoid(a * self.theta(prof_idx).squeeze(-1) + d).clamp(EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "ratings_v2"))
    ap.add_argument("--pca-dim", type=int, default=128, help="Joint activation PCs.")
    ap.add_argument("--diff-hidden", type=int, nargs="*", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--min", type=int, default=5,
                    help="Min first-time correct AND incorrect to keep a user.")
    ap.add_argument("--min-profile", type=int, default=8)
    ap.add_argument("--lambda-resid", type=float, default=0.5,
                    help="Ridge on per-problem residuals delta/eta = prior precision "
                         "(empirical-Bayes shrinkage toward the feature prior). With "
                         "~5 attempts/problem the prior should dominate, so keep this "
                         "fairly strong. The key knob to calibrate.")
    ap.add_argument("--winsor", type=float, default=1.0,
                    help="Winsorize final ratings to [p, 100-p] percentile (robustness to "
                         "the residual b=-d/a tail for low-slope problems).")
    ap.add_argument("--lambda-theta", type=float, default=0.01)
    ap.add_argument("--lambda-w", type=float, default=1e-4,
                    help="Ridge on the feature-head weights.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dro-skill", action=argparse.BooleanOptionalAction, default=True,
                    help="ON by default: spread held-out error evenly across the rating "
                         "range with a base-rate-centered group-DRO objective (replaces "
                         "mean BCE on the human-attempt term). Two-pass: pass 1 trains with "
                         "mean BCE to get the rating, which defines the rating bins; pass 2 "
                         "retrains with DRO over those bins. `--no-dro-skill` reverts to "
                         "plain ERM (mean BCE), which is overall-LogLoss-optimal but "
                         "concentrates error at the difficulty tails. See the --even-error "
                         "study in docs/rating_model_comparison_results.md.")
    ap.add_argument("--dro-eta", type=float, default=0.2,
                    help="Group-DRO exponentiated-gradient step on per-bin base-centered "
                         "loss. 0.2 is the flatness↔cost knee (η≥~1.6 collapses q onto one "
                         "bin; <0.1 under-flattens the U-shaped 9×9 board).")
    ap.add_argument("--dro-bins", type=int, default=10,
                    help="Equal-count rating-quantile bins the DRO objective groups over.")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # --- scorable corpus = problems with a cached activation vector (per size) ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    print("activation-cache problems: " + ", ".join(
        f"{s}:{len(tids_by_size[s])}" for s in SIZES))
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    # a problem is scorable iff it has BOTH feature blocks
    keys = [key for key in act if key in bw]                 # (size, tid)
    key_idx = {key: i for i, key in enumerate(keys)}
    fdim = k + nb
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        feat[i] = np.concatenate([act[key], bw[key]])
    print(f"scorable problems {len(keys)} (9:{sum(s==9 for s,_ in keys)} "
          f"13:{sum(s==13 for s,_ in keys)}) | feat dim {fdim} (act {k} + bot {nb})")

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
    prof_df = prof_df[prof_df["profile"].isin(
        set(pc.index[pc >= args.min_profile]))]
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

    # --- fit ---
    tfeat = torch.tensor(feat, device=device)
    tprof = torch.tensor(prof_idx, device=device)
    tprob = torch.tensor(prob_idx, device=device)
    ty = torch.tensor(y, device=device)
    bce = nn.BCELoss()
    n = len(y)

    def fit_model(dro=None):
        """Train one RatingV2. dro=(rank_bin_tensor, base_g, G, eta) swaps the human
        BCE term for a base-rate-centered online group-DRO over rank bins (Sagawa et
        al., centered to target REDUCIBLE skill-over-base, not the aleatoric base-rate
        floor — see docs/rating_model_comparison_results.md --even-error). The ridge
        priors are unchanged. q persists across minibatches/epochs."""
        torch.manual_seed(args.seed)
        m = RatingV2(n_profiles, len(keys), fdim, tuple(args.diff_hidden),
                     args.dropout).to(device)
        feat_params = list(m.diff_head.parameters()) + list(m.slope_head.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        if dro is not None:
            rb, base_g, G, eta = dro
            q = torch.ones(G, device=device) / G
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(n, device=device)
            for b0 in range(0, n, args.batch):
                bi = perm[b0:b0 + args.batch]
                opt.zero_grad()
                p = m(tprof[bi], tprob[bi], tfeat[tprob[bi]])
                if dro is None:
                    data_loss = bce(p, ty[bi])
                else:
                    l = torch.nn.functional.binary_cross_entropy(
                        p, ty[bi], reduction="none")
                    gb = rb[bi]
                    gl = torch.zeros(G, device=device).index_add(0, gb, l)
                    cnt = torch.zeros(G, device=device).index_add(
                        0, gb, torch.ones_like(l))
                    present = cnt > 0
                    Lg = gl[present] / cnt[present]
                    with torch.no_grad():           # update q on CENTERED bin loss
                        q[present] = q[present] * torch.exp(
                            eta * (Lg.detach() - base_g[present]))
                        q[~present] = 0.0
                        q /= q.sum().clamp(min=EPS)
                    data_loss = (q[present] * Lg).sum()
                loss = data_loss + args.lambda_resid * (m.delta.weight.pow(2).mean()
                                                        + m.eta.weight.pow(2).mean())
                loss = loss + args.lambda_theta * m.theta.weight.pow(2).mean()
                loss = loss + args.lambda_w * sum(w.pow(2).sum() for w in feat_params)
                loss.backward()
                opt.step()
            if ep % 50 == 0 or ep == args.epochs - 1:
                m.eval()
                with torch.no_grad():
                    ll = bce(m(tprof, tprob, tfeat[tprob]), ty).item()
                print(f"  [{'dro' if dro else 'mean'}] epoch {ep:4d}  train BCE {ll:.4f}")
        return m

    def score(m):
        m.eval()
        with torch.no_grad():
            allprob = torch.arange(len(keys), device=device)
            d, a = m.diff_slope(allprob, tfeat)
        return d.cpu().numpy(), a.cpu().numpy()

    m = fit_model()
    if args.dro_skill:
        # Pass 1 done. Bin TRAIN rows by their problem's pass-1 rating b_p (the shipped
        # rating, pooled across boards in shared ability units), equal-count quantiles;
        # per-bin base-rate loss = the bin's human-success entropy (the aleatoric floor).
        d0, a0 = score(m)
        row_b = ((CLEAR_LOGIT - d0) / a0)[prob_idx]   # bin by the shipped 80%-clear rating
        edges = np.quantile(row_b, np.linspace(0, 1, args.dro_bins + 1))
        edges[-1] += 1e-6
        rb = np.clip(np.searchsorted(edges, row_b, side="right") - 1,
                     0, args.dro_bins - 1).astype(np.int64)
        br = np.array([y[rb == g].mean() if (rb == g).any() else 0.5
                       for g in range(args.dro_bins)])
        br = np.clip(br, EPS, 1 - EPS)
        base_g = torch.tensor(-(br * np.log(br) + (1 - br) * np.log(1 - br)),
                              dtype=torch.float32, device=device)
        rbt = torch.tensor(rb, dtype=torch.long, device=device)
        print(f"  dro_skill: {args.dro_bins} rating bins, eta {args.dro_eta} — refitting")
        m = fit_model((rbt, base_g, args.dro_bins, args.dro_eta))

    # --- score the whole corpus ---
    d, a = score(m)
    b = (CLEAR_LOGIT - d) / a            # rating = ability at which P = CALIBRATION_P (80%)
    lo, hi = np.percentile(b, [args.winsor, 100 - args.winsor])
    b = np.clip(b, lo, hi)                        # tame the low-slope b=-d/a tail
    dres = m.delta.weight.detach().cpu().numpy().ravel()
    eres = m.eta.weight.detach().cpu().numpy().ravel()
    print(f"  rating winsorized to [{lo:.2f},{hi:.2f}] | slope p[1,50,99] "
          f"{np.percentile(a,[1,50,99]).round(2)} | |delta| mean "
          f"{np.abs(dres).mean():.3f} (frac>0.5: {(np.abs(dres)>0.5).mean():.2f}) | "
          f"|eta| mean {np.abs(eres).mean():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(args.out_dir, str(s))
        os.makedirs(sdir, exist_ok=True)
        pd.DataFrame({"tsumego_id": tids, "rating": b[sel]}).sort_values(
            "rating", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_ratings_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "quality": a[sel]}).sort_values(
            "quality", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_quality_final.csv"), index=False)
        print(f"  size {s}: wrote {len(tids)} ratings | rating "
              f"[{b[sel].min():.2f},{b[sel].max():.2f}] slope [{a[sel].min():.2f},"
              f"{a[sel].max():.2f}]")

    theta = m.theta.weight.detach().cpu().numpy().ravel()
    natt = pd.Series(prof_idx).value_counts().sort_index()
    pd.DataFrame({"profile": list(profiles.categories), "theta": theta,
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_profiles)]}).to_csv(
        os.path.join(args.out_dir, "user_theta.csv"), index=False)
    torch.save({"state_dict": m.state_dict(), "keys": keys, "fdim": fdim, "k": k,
                "n_bot_cols": nb, "args": vars(args)},
               os.path.join(args.out_dir, "model_v2.pt"))
    print(f"wrote user_theta.csv ({n_profiles} profiles) + model_v2.pt to {args.out_dir}")


if __name__ == "__main__":
    main()
