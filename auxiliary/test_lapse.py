"""Held-out A/B for a GLOBAL 4PL lapse (fat-finger) rate fit INSIDE train_rating_v2's
RatingV2, vs the current pure-3PL fit (lapse off).

Two questions (the ones the user asked):
  1. Does a free-fit global lapse find a REASONABLE value without being forced — i.e. does it
     converge near the posthoc plateau (~0.0026) regardless of init/seed, rather than collapsing
     to 0 or running to the cap?
  2. Does it IMPROVE held-out prediction — overall, and specifically in the saturated regime
     (problems trivially easy for the user) where the cap is supposed to act?

Same data prep + regularization as pipeline.train_rating_v2.main(); the ONLY difference between
arms is RatingV2(learn_lapse=...). Single-pass mean-BCE (no DRO) for speed — the lapse is
identified by the data term, present in both passes. Held out by attempt ROW (85/15), evaluated
on val rows whose profile is in train (so ability is estimable; identical disadvantage to both
arms, so the comparison is fair).

Run:  ./.venv/bin/python3 -m auxiliary.test_lapse
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rating_models import (  # noqa: E402
    build_profiles, load_joint_activations, load_joint_bot_winrate)
from auxiliary.inspect_activation_signals import _cache_ids_path  # noqa: E402
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402
from pipeline.train_rating_v2 import (  # noqa: E402
    CALIBRATION_P, EPS, SIZES, clear_logit_vec)
from auxiliary.train_rating_v2_legacy import (  # noqa: E402  (the preserved residual+DRO model)
    MAX_FLOOR, ZCLIP, RatingV2, dyadic_tail_bins, load_bot_floor)


class Stage1Shared(torch.nn.Module):
    """Stage-1 (the design you described): a SHARED feature→3PL network — three decreasing layers
    with ReLU between, final layer producing 3 outputs (difficulty, slope, floor) — plus per-user
    ability. NO per-problem parameters: every problem's 3PL comes from the shared net applied to
    its features, so a zero-interaction problem still gets a prediction (nothing 'sits at init').
    Only regularizer: the unit-variance ability gauge.
        P = c(feat) + (1 − c(feat))·σ( a(feat)·(θ_u − b(feat)) )
    """

    def __init__(self, fdim, n_users, hidden=(128, 64), max_floor=0.5, dropout=0.0,
                 standardize_ability=False, lapse_cap=0.05, lapse_init=0.01):
        super().__init__()
        self.theta = torch.nn.Embedding(n_users, 1)
        torch.nn.init.normal_(self.theta.weight, std=1.0)
        self.standardize_ability = standardize_ability
        layers, d = [], fdim
        for h in hidden:                                    # decreasing layers, ReLU between
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
            if dropout > 0:
                layers += [torch.nn.Dropout(dropout)]
            d = h
        layers += [torch.nn.Linear(d, 3)]                   # final layer → (difficulty, slope, floor)
        self.net = torch.nn.Sequential(*layers)
        self.max_floor = float(max_floor)
        # Global learned 4PL ceiling (fat-finger / lapse), shared across problems.
        self.lapse_cap = float(lapse_cap)
        li = float(np.clip(lapse_init / max(lapse_cap, EPS), EPS, 1 - EPS))
        self.lapse_raw = torch.nn.Parameter(torch.tensor(float(np.log(li / (1 - li)))))

    def lapse(self):
        return self.lapse_cap * torch.sigmoid(self.lapse_raw)

    def params(self, feat):
        o = self.net(feat)
        b = o[:, 0]
        a = torch.nn.functional.softplus(o[:, 1]).clamp(min=0.01)   # lower bound ONLY, no upper cap
        # floor SMOOTHLY bounded to [0, 1 − 2λ]: ceiling-coupled, so it stays a margin λ below the
        # 1−λ upper asymptote (the active band 1−c−λ ≥ λ > 0). The cap rises as the lapse shrinks.
        cap = (1.0 - 2.0 * self.lapse()).clamp(min=EPS)
        c = cap * torch.sigmoid(o[:, 2])
        return b, a, c

    def ability(self, u):
        """Per-user ability. With standardize_ability the embedding is whitened to mean 0 / std 1
        BY CONSTRUCTION each call (a hard unit-Gaussian gauge), so item slopes can't inflate to
        compensate for compressed ability — instead of a soft N(0,1) prior that lets it drift."""
        if self.standardize_ability:
            w = self.theta.weight
            return ((self.theta(u) - w.mean()) / (w.std() + EPS)).squeeze(-1)
        return self.theta(u).squeeze(-1)

    def forward(self, u, pfeat):
        b, a, c = self.params(pfeat)
        praw = torch.sigmoid(a * (self.ability(u) - b))
        return (c + (1.0 - c - self.lapse()) * praw).clamp(EPS, 1 - EPS)


class Stage2(torch.nn.Module):
    """Stage-2: per-problem difficulty/slope RESIDUALS (ridge-shrunk to the stage-1 feature priors)
    + re-fit per-user ability, on the human attempts. The stage-1 floor c and global ceiling λ are
    frozen (they're the pooled estimates stage 1 owns). This is the empirical-Bayes second stage:
        b = b_prior + δb_p ,  a = a_prior·exp(δa_p) ,  c = c_prior ,
        P = c + (1 − c − λ)·σ( a·(θ_u − b) )         δb,δa ~ N(0, 1/lambda_resid)
    """

    def __init__(self, b_prior, a_prior, c_prior, lapse, theta_init, standardize_ability=True):
        super().__init__()
        self.register_buffer("b_prior", torch.as_tensor(b_prior, dtype=torch.float32))
        self.register_buffer("a_prior", torch.as_tensor(a_prior, dtype=torch.float32))
        self.register_buffer("c_prior", torch.as_tensor(c_prior, dtype=torch.float32))
        self.register_buffer("lapse", torch.as_tensor(float(lapse), dtype=torch.float32))
        n_prob, n_users = len(b_prior), len(theta_init)
        self.delta_b = torch.nn.Embedding(n_prob, 1); torch.nn.init.zeros_(self.delta_b.weight)
        self.delta_a = torch.nn.Embedding(n_prob, 1); torch.nn.init.zeros_(self.delta_a.weight)
        self.theta = torch.nn.Embedding(n_users, 1)
        self.theta.weight.data = torch.as_tensor(theta_init, dtype=torch.float32).view(-1, 1).clone()
        self.standardize_ability = standardize_ability

    def ability(self, u):
        if self.standardize_ability:
            w = self.theta.weight
            return ((self.theta(u) - w.mean()) / (w.std() + EPS)).squeeze(-1)
        return self.theta(u).squeeze(-1)

    def eff(self, p):
        b = self.b_prior[p] + self.delta_b(p).squeeze(-1)
        a = (self.a_prior[p] * torch.exp(self.delta_a(p).squeeze(-1))).clamp(0.05, 15.0)
        return b, a, self.c_prior[p]

    def forward(self, u, p):
        b, a, c = self.eff(p)
        praw = torch.sigmoid(a * (self.ability(u) - b))
        return (c + (1.0 - c - self.lapse) * praw).clamp(EPS, 1 - EPS)


def load_data(args, device):
    tids_by_size = {s: __import__("json").load(open(_cache_ids_path(s))) for s in SIZES}
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]
    key_idx = {key: i for i, key in enumerate(keys)}
    fdim = k + nb
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        feat[i] = np.concatenate([act[key], bw[key]])

    floor_c, _ = load_bot_floor(keys)

    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    df = df[df["user_id"].isin(set(g.index[(g["c"] >= args.min) & (g["w"] >= args.min)]))].copy()
    prob2size = dict(zip(df["problem"], df["size"]))
    prof_df = build_profiles(df)
    pc = prof_df["profile"].value_counts()
    prof_df = prof_df[prof_df["profile"].isin(set(pc.index[pc >= args.min_profile]))]
    prof_df = prof_df[prof_df["problem"].map(
        lambda t: (prob2size.get(t), t) in key_idx)].reset_index(drop=True)

    profiles = pd.Categorical(prof_df["profile"])
    prof_idx = profiles.codes.astype(np.int64)
    n_profiles = len(profiles.categories)
    y = prof_df["success"].to_numpy(np.float32)
    prob_idx = prof_df["problem"].map(lambda t: key_idx[(prob2size[t], t)]).to_numpy(np.int64)
    print(f"rows {len(y)} | profiles {n_profiles} | problems {prof_df['problem'].nunique()} "
          f"| base rate {y.mean():.4f} | feat {fdim} (act {k}+bot {nb})")
    return dict(
        feat=torch.tensor(feat, device=device), floor_c=floor_c, fdim=fdim,
        n_keys=len(keys), n_profiles=n_profiles, keys=keys,
        profiles=list(profiles.categories), prob2size=prob2size,
        prof_idx=prof_idx, prob_idx=prob_idx, y=y)


def load_anchor_winrates(keys, ranks):
    """Per-problem humanv0 pass-rate for each anchor rank, aligned to `keys` (the model's problem
    order). Returns float (n_keys, n_ranks), NaN where a rank lacks a value for that problem.
    These are the soft targets for the bot 'users': value = P(the rank-conditioned humanv0 bot
    plays the correct first move) — same semantics as a human first-move success."""
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


def fit(data, args, device, learn_lapse, lapse_init, seed, trace=False, freeze=False,
        free_floor_scale=False, floor_scale_init=1.0, free_floor=False):
    """Train one RatingV2 on the train split; return (model, val_pred, eval_mask, lapse).
    freeze=True holds the lapse FIXED at lapse_init (tests forcing the posthoc value) instead
    of learning it. free_floor_scale=True replaces the prior-anchored floor head with a FREE
    global floor scale γ (init floor_scale_init) — the 'does the floor collapse?' test.
    With args.dro the human term is the production two-pass base-rate-centered 2D group-DRO
    (rating × relative-gap dyadic bins) ported from train_rating_v2.main — the objective that
    up-weights the saturated far-above-rank cells where the lapse acts (the fair test)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(data["y"])
    perm = np.random.permutation(n)
    n_va = int(args.val_frac * n)
    va_rows = np.zeros(n, bool); va_rows[perm[:n_va]] = True
    tr_rows = ~va_rows
    tr_profiles = set(np.unique(data["prof_idx"][tr_rows]).tolist())
    tr_problems = set(np.unique(data["prob_idx"][tr_rows]).tolist())
    # eval only val rows whose profile/problem were seen in train (estimable θ; identical to both arms)
    eval_mask = va_rows & np.isin(data["prof_idx"], list(tr_profiles)) \
                        & np.isin(data["prob_idx"], list(tr_problems))

    prof_idx, prob_idx, y = data["prof_idx"], data["prob_idx"], data["y"]
    tprof = torch.tensor(prof_idx, device=device)
    tprob = torch.tensor(prob_idx, device=device)
    ty = torch.tensor(y, device=device)
    tfeat = data["feat"]
    tr_idx = torch.tensor(np.where(tr_rows)[0], device=device)
    ntr = int(tr_rows.sum())
    prior_a = 1.9
    slope_bias_val = float(np.log(np.expm1(prior_a)))

    def new_model():
        torch.manual_seed(seed)
        m = RatingV2(data["n_profiles"], data["n_keys"], data["fdim"],
                     tuple(args.diff_hidden), args.dropout, slope_bias=slope_bias_val,
                     floor_c=data["floor_c"], floor_scale=args.floor_scale,
                     learn_floor_head=(not free_floor_scale and not free_floor), max_floor=MAX_FLOOR,
                     learn_lapse=learn_lapse, lapse_cap=args.lapse_cap, lapse_init=lapse_init,
                     learn_floor_scale=free_floor_scale, floor_scale_init=floor_scale_init,
                     learn_floor_free=free_floor, floor_free_init=args.floor_free_init).to(device)
        if learn_lapse and freeze:
            m.lapse_raw.requires_grad_(False)        # hold lapse FIXED at lapse_init
        return m

    def train_one(dro, tag):
        m = new_model()
        feat_params = list(m.diff_head.parameters()) + list(m.slope_head.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        if dro is not None:
            rbt, base_g, G = dro
            q = torch.ones(G, device=device) / G
        for ep in range(args.epochs):
            m.train()
            pm = tr_idx[torch.randperm(ntr, device=device)]
            for b0 in range(0, ntr, args.batch):
                bi = pm[b0:b0 + args.batch]
                opt.zero_grad()
                p = m(tprof[bi], tprob[bi], tfeat[tprob[bi]])
                if dro is None:
                    data_loss = torch.nn.functional.binary_cross_entropy(p, ty[bi])
                else:
                    lvec = torch.nn.functional.binary_cross_entropy(p, ty[bi], reduction="none")
                    gb = rbt[bi]
                    gl = torch.zeros(G, device=device).index_add(0, gb, lvec)
                    cnt = torch.zeros(G, device=device).index_add(0, gb, torch.ones_like(lvec))
                    present = cnt > 0
                    Lg = gl[present] / cnt[present]
                    with torch.no_grad():
                        q[present] = q[present] * torch.exp(args.dro_eta * (Lg.detach() - base_g[present]))
                        q[~present] = 0.0
                        q /= q.sum().clamp(min=EPS)
                    data_loss = (q[present] * Lg).sum()
                loss = data_loss + args.lambda_resid * (m.delta.weight.pow(2).mean()
                                                        + m.eta.weight.pow(2).mean())
                th = m.theta.weight
                loss = loss + (0.5 * th.pow(2)).sum() / ntr + args.lambda_theta_mean * th.mean().pow(2)
                loss = loss + args.lambda_w * sum(w.pow(2).sum() for w in feat_params)
                if m.learn_floor_head:
                    loss = loss + args.lambda_floor * m.floor_resid.weight.pow(2).mean()
                    loss = loss + args.lambda_w * m.floor_head.weight.pow(2).sum()
                if m.learn_floor_free:                  # ridge on the per-problem floor resid only
                    loss = loss + args.lambda_floor * m.floor_free_resid.weight.pow(2).mean()
                loss.backward()
                opt.step()
            if trace and (ep % args.log_every == 0 or ep == args.epochs - 1):
                m.eval()
                with torch.no_grad():
                    pv = m(tprof, tprob, tfeat[tprob]).cpu().numpy()
                    extra = ""
                    if free_floor_scale:
                        allk = torch.arange(data["n_keys"], device=device)
                        fl = m.floor_for(allk, tfeat).cpu().numpy()
                        nz = data["floor_c"] > 0
                        extra = (f"  gamma {float(m.floor_gamma()):.4f}  "
                                 f"mean_floor(c_bot>0) {fl[nz].mean():.4f}")
                    elif free_floor:
                        allk = torch.arange(data["n_keys"], device=device)
                        fl = m.floor_for(allk, tfeat).cpu().numpy()
                        att = np.unique(data["prob_idx"])           # problems with human attempts
                        extra = (f"  floor: mean {fl.mean():.4f}  std {fl.std():.4f}  "
                                 f"mean@attempted {fl[att].mean():.4f}  "
                                 f"p[50,95] {np.percentile(fl, [50, 95]).round(3)}")
                    else:
                        extra = f"  lapse {float(m.lapse()) if learn_lapse else 0.0:.5f}"
                print(f"      [{tag}] ep {ep:4d}  val_ll "
                      f"{logloss(y[eval_mask], pv[eval_mask]):.5f}{extra}")
        if dro is not None:
            m._q_final = q.detach().cpu().numpy()      # converged DRO bin weights
        return m

    m = train_one(None, "mean")
    if args.dro:
        # Bin TRAIN rows by pass-1 80%-clear rating (dyadic) × relative-gap z (dyadic), exactly
        # like train_rating_v2.main; base-rate loss per cell on TRAIN rows only (no val leak).
        m.eval()
        with torch.no_grad():
            allk = torch.arange(data["n_keys"], device=device)
            d0t, a0t = m.diff_slope(allk, tfeat)
            c0 = m.floor_for(allk, tfeat).cpu().numpy() if m.use_floor else np.zeros(data["n_keys"])
            lam0 = float(m.lapse()) if learn_lapse else 0.0
            theta0 = m.theta.weight.detach().cpu().numpy().ravel()
        d0, a0 = d0t.cpu().numpy(), a0t.cpu().numpy()
        clv0 = clear_logit_vec(c0, lam0)
        rating0 = (clv0 - d0) / a0
        row_b = rating0[prob_idx]
        rbin_all, nrb = dyadic_tail_bins(row_b, args.dro_r_depth)
        z_row = np.clip(a0[prob_idx] * theta0[prof_idx] + d0[prob_idx] - clv0[prob_idx], -ZCLIP, ZCLIP)
        zbin_all, nzb = dyadic_tail_bins(z_row, args.dro_z_depth)
        G = nrb * nzb
        rb = (zbin_all * nrb + rbin_all).astype(np.int64)
        br = np.array([y[tr_rows & (rb == g)].mean() if (tr_rows & (rb == g)).any() else 0.5
                       for g in range(G)])
        br = np.clip(br, EPS, 1 - EPS)
        base_g = torch.tensor(-(br * np.log(br) + (1 - br) * np.log(1 - br)),
                              dtype=torch.float32, device=device)
        rbt = torch.tensor(rb, dtype=torch.long, device=device)
        if trace:
            print(f"      dro bins: {nzb}z×{nrb}r = {G}")
        m = train_one((rbt, base_g, G), "dro")
        m._dro = dict(rb=rb, zbin=zbin_all, nrb=nrb, nzb=nzb, z_row=z_row,
                      base_g=br, G=G, tr_rows=tr_rows)

    m.eval()
    with torch.no_grad():
        pred = m(tprof, tprob, tfeat[tprob]).cpu().numpy()
        lam = float(m.lapse()) if learn_lapse else 0.0
    return m, pred, eval_mask, lam


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def text_hist(name, vals, lo=None, hi=None, nbins=20, width=46):
    """Print a text histogram of `vals` over [lo, hi]; lo/hi default to the p0.5/p99.5 of the data
    (so heavy tails are shown, not clipped at an arbitrary fixed floor). Out-of-range → end bins."""
    if lo is None:
        lo = float(np.floor(np.percentile(vals, 0.5)))
    if hi is None:
        hi = float(np.ceil(np.percentile(vals, 99.5)))
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(vals, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins)
    mx = counts.max() or 1
    print(f"  {name}  (n={len(vals)}, mean {vals.mean():.3f}, median {np.median(vals):.3f}, "
          f"std {vals.std():.3f})")
    for i in range(nbins):
        bar = "#" * int(round(width * counts[i] / mx))
        print(f"    [{edges[i]:6.2f},{edges[i+1]:6.2f})  {counts[i]:7d} {bar}")


def posthoc_plateau(m, data, device, thresh=0.99):
    """Reproduce fit_population_constants.fit_fat_finger on the SAME fit: the raw failure rate
    on rows whose INNER sigmoid (modeled clear before floor/lapse) ≥ thresh — the 'trivially
    easy for the user' set. This is the number the posthoc estimator ships (~0.0026); compare it
    to the MLE global lapse."""
    with torch.no_grad():
        allprob = torch.tensor(data["prob_idx"], device=device)
        prof = torch.tensor(data["prof_idx"], device=device)
        dd, aa = m.diff_slope(allprob, data["feat"][allprob])
        praw = torch.sigmoid(aa * m.theta(prof).squeeze(-1) + dd).cpu().numpy()
    easy = praw >= thresh
    n = int(easy.sum())
    fail = float((1 - data["y"][easy]).mean()) if n else float("nan")
    return fail, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--diff-hidden", type=int, nargs="*", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--min", type=int, default=5)
    ap.add_argument("--min-profile", type=int, default=8)
    ap.add_argument("--lambda-resid", type=float, default=0.5)
    ap.add_argument("--lambda-theta-mean", type=float, default=10.0)
    ap.add_argument("--lambda-w", type=float, default=1e-4)
    ap.add_argument("--lambda-floor", type=float, default=0.5)
    ap.add_argument("--floor-scale", type=float, default=0.70)
    ap.add_argument("--max-floor", type=float, default=MAX_FLOOR)
    ap.add_argument("--lapse-cap", type=float, default=0.05)
    ap.add_argument("--dro", action="store_true",
                    help="Use the production two-pass base-rate-centered 2D group-DRO objective "
                         "(rating × gap dyadic bins) instead of plain mean BCE — the fair test, "
                         "since 2D-DRO up-weights the saturated cells the lapse acts on.")
    ap.add_argument("--dro-eta", type=float, default=0.2)
    ap.add_argument("--dro-r-depth", type=int, default=3)
    ap.add_argument("--dro-z-depth", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--inits", type=float, nargs="*", default=[0.0005, 0.005, 0.02])
    ap.add_argument("--trace", action="store_true",
                    help="Fit ONE off + ONE on model with periodic val-logloss/lapse logging "
                         "(check convergence) instead of the seed/init grid.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--fixed", type=float, nargs="*", default=None,
                    help="Held-out A/B over FIXED (frozen) lapse values, e.g. --fixed 0 0.0026 "
                         "0.005 0.01 — tests whether FORCING the posthoc value (vs free-fit) "
                         "improves held-out prediction.")
    ap.add_argument("--dro-diag", action="store_true",
                    help="Anatomy of why DRO doesn't keep the lapse: fit free-lapse DRO, then per "
                         "z-bin report n, empirical vs predicted clear, loss vs base-rate entropy "
                         "(centered), the converged DRO weight on the bin, and the catastrophic "
                         "(pred≥0.999 & failed) fraction.")
    ap.add_argument("--free-floor-scale", action="store_true",
                    help="Replace the prior-anchored floor with a FREE global floor scale γ on the "
                         "bot floor (c_p = γ·c_bot, no prior/ridge) and trace γ — does the floor "
                         "LEVEL collapse to ~0 when determined freely? Inits γ from --floor-inits.")
    ap.add_argument("--floor-inits", type=float, nargs="*", default=[1.0, 0.70, 0.3],
                    help="γ init values for --free-floor-scale (init-independence check).")
    ap.add_argument("--free-floor", action="store_true",
                    help="TRULY FREE per-problem floor (no bot prior): c_p = σ(free_bias + "
                         "resid_p), free global bias + ridge-shrunk per-problem residual. Does the "
                         "free average floor grow (data wants a floor) or collapse, and do the "
                         "per-problem floors differentiate? Traces floor mean/std + far-below fit.")
    ap.add_argument("--floor-free-init", type=float, default=0.1,
                    help="Initial floor level for --free-floor (the free bias starts here).")
    ap.add_argument("--empirical-curve", action="store_true",
                    help="Model-stable raw empirical clear rate vs current-rating gap (unclipped, "
                         "fine bins + deep-tail probes): does far-below plateau (real floor) and "
                         "does far-above hit 1 (no real ceiling)?")
    ap.add_argument("--room-sweep", type=float, nargs="*", default=None,
                    help="Sweep lambda_resid (per-problem rating/slope 'room') with a FREE floor to "
                         "test whether more room makes the floor used MORE (co-adaptation) or LESS "
                         "(slopes mimic it). e.g. --room-sweep 0.1 0.5 2.0")
    ap.add_argument("--hard-unit-ability", action="store_true",
                    help="For --stage1: HARD-enforce N(0,1) user abilities by whitening the ability "
                         "embedding (mean 0 / std 1) each step, instead of the soft N(0,1) prior — "
                         "stops slope inflation / ability compression.")
    ap.add_argument("--stage2", action="store_true",
                    help="STAGE 2: load ratings_v2/stage1.npz priors, fit per-problem difficulty/"
                         "slope residuals (ridge --lambda-resid → stage-1 prior) + re-fit ability on "
                         "the human attempts; report held-out lift over the prior alone.")
    ap.add_argument("--stage1", action="store_true",
                    help="Two-stage model, STAGE 1 only: clean per-problem 3PL IRT (free a,b,c + "
                         "per-user ability, plain MLE, unit-variance ability gauge, NO feature heads, "
                         "NO item ridge, NO DRO). c starts at ~0; report the fitted 3PL distributions "
                         "— does the floor stay 0 or does the data pull it up where identifiable?")
    ap.add_argument("--mixed-only", action="store_true",
                    help="STAGE 1: train the shared net ONLY on two-sided problems (≥1 clear AND "
                         "≥1 fail); one-sided problems can't identify a slope. The net still PREDICTS "
                         "a 4PL for every problem; inspect whether the one-sided/unattempted tails tame.")
    ap.add_argument("--anchors", action="store_true",
                    help="STAGE 1: add bot 'users' (humanv0 ranks, --anchor-ranks) as extra users "
                         "with LEARNED ability and SOFT-target (pass-rate) responses on EVERY problem. "
                         "Coverage regularizes the empty/one-sided feature regions (two strengths ⇒ two "
                         "points per curve ⇒ b AND a identified everywhere). Human val is unchanged.")
    ap.add_argument("--anchor-ranks", default="20k,9d",
                    help="comma-separated humanv0 ranks to use as bot users (default 20k,9d).")
    ap.add_argument("--anchor-weight", type=float, default=0.2,
                    help="per-row weight on bot-anchor responses (human rows = 1.0). Low = minimal "
                         "prior: humans own the shape where attested, bots fill the empty regions.")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device.type}")
    data = load_data(args, device)
    y = data["y"]

    if args.empirical_curve:
        args.dro = True
        print("\n=== model-independent EMPIRICAL success curve by current-rating gap ===")
        print("    one fixed production-config model (floor head, no lapse) defines the gap axis;")
        print("    emp_clr is the RAW human clear rate per gap bin (unclipped) — the true shape.")
        m, pred, mask, _ = fit(data, args, device, False, 0.0, args.seed)
        with torch.no_grad():
            allk = torch.arange(data["n_keys"], device=device)
            dd, aa = m.diff_slope(allk, data["feat"])
            cc = m.floor_for(allk, data["feat"]).cpu().numpy()
            theta = m.theta.weight.detach().cpu().numpy().ravel()
        dd, aa = dd.cpu().numpy(), aa.cpu().numpy()
        clv = clear_logit_vec(cc, 0.0)
        rating = (clv - dd) / aa
        gap = theta[data["prof_idx"]] - rating[data["prob_idx"]]      # rating units, UNCLIPPED
        print(f"  gap range [{gap.min():.2f}, {gap.max():.2f}]  median {np.median(gap):.2f}")
        nb = 24
        edges = np.quantile(gap, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-6
        b = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0, nb - 1)
        print(f"  {'bin':>3} {'n':>6} {'gap_lo':>7} {'gap_hi':>7} {'mean_gap':>8} "
              f"{'emp_clr':>8} {'pred':>7} {'floorless_σ':>11}")
        for bi in range(nb):
            sel = b == bi; n = int(sel.sum())
            if not n:
                continue
            # floorless 2PL sigmoid at this model's slope (what "no floor" predicts for the gap)
            inner = np.clip(aa[data["prob_idx"][sel]] * gap[sel], -30, 30)
            fl_sig = float((1.0 / (1.0 + np.exp(-inner))).mean())
            print(f"  {bi:>3} {n:>6} {edges[bi]:>7.2f} {edges[bi+1]:>7.2f} {gap[sel].mean():>8.2f} "
                  f"{y[sel].mean():>8.4f} {pred[sel].mean():>7.4f} {fl_sig:>11.4f}")
        # deep-tail asymptote probes (does far-below plateau, does far-above hit 1?)
        for pct, lab in [(0.5, "below p0.5"), (1, "below p1"), (2, "below p2")]:
            thr = np.percentile(gap, pct); s = gap <= thr
            print(f"  {lab:>11} (gap≤{thr:5.2f}): n {int(s.sum()):5d}  emp_clr {y[s].mean():.4f}")
        for pct, lab in [(99.5, "above p99.5"), (99, "above p99"), (98, "above p98")]:
            thr = np.percentile(gap, pct); s = gap >= thr
            print(f"  {lab:>11} (gap≥{thr:5.2f}): n {int(s.sum()):5d}  emp_clr {y[s].mean():.4f}")
        return

    if args.stage1:
        print("\n=== STAGE 1: SHARED feature→4PL net (3 layers ReLU → 3 outputs: difficulty/slope/"
              "floor) + per-user ability + learned global ceiling λ. floor cap = 1−2λ (ceiling-"
              "coupled); slope lower-bounded at 0.01 (no upper cap); unit-ability gauge; no ridge/DRO ===")
        prof_idx_f, prob_idx_f, yf = data["prof_idx"], data["prob_idx"], y
        n_users, n_prob = data["n_profiles"], data["n_keys"]
        if args.mixed_only:
            # drop problems whose first-time attempts are ALL clear or ALL fail (one-sided ⇒ slope
            # unidentifiable); the shared net still PREDICTS a 4PL for every problem via features.
            succ = np.bincount(prob_idx_f, weights=y, minlength=n_prob)
            cntp = np.bincount(prob_idx_f, minlength=n_prob)
            mixed = (succ > 0) & (succ < cntp)
            keep = mixed[prob_idx_f]
            n_att = len(np.unique(prob_idx_f))
            prof_idx_f, prob_idx_f, yf = prof_idx_f[keep], prob_idx_f[keep], y[keep]
            print(f"  MIXED-ONLY: trained on {int(keep.sum())}/{len(keep)} rows from "
                  f"{int(mixed.sum())} two-sided problems (dropped {n_att - int(mixed.sum())} "
                  f"one-sided attempted problems)")
        nh = len(yf)
        # held-out split on HUMAN rows only (the metric that matters; anchors are a train-only prior)
        np.random.seed(args.seed)
        va_h = np.zeros(nh, bool); va_h[np.random.permutation(nh)[:int(0.15 * nh)]] = True

        # optional bot 'anchor users': extra users with LEARNED ability + soft-target (pass-rate)
        # responses on EVERY problem. Coverage is the prior; their θ is fit like any user (no fixed
        # rank). Two strengths ⇒ two points on each problem's curve ⇒ b AND a identified everywhere.
        n_anchor = 0
        if args.anchors:
            ranks = [r.strip() for r in args.anchor_ranks.split(",") if r.strip()]
            aw = load_anchor_winrates(data["keys"], ranks)            # (n_prob, n_anchor) pass-rate
            n_anchor = len(ranks)
            au, ap, at = [], [], []
            for j in range(n_anchor):
                ok = np.where(np.isfinite(aw[:, j]))[0]
                au.append(np.full(len(ok), n_users + j, np.int64))
                ap.append(ok.astype(np.int64))
                at.append(aw[ok, j].astype(np.float32))
            au, ap, at = np.concatenate(au), np.concatenate(ap), np.concatenate(at)
            cover = [int(np.isfinite(aw[:, j]).sum()) for j in range(n_anchor)]
            print(f"  ANCHORS: {n_anchor} bot users {ranks} | coverage {cover}/{n_prob} | "
                  f"row-weight {args.anchor_weight} | abilities LEARNED (model-determined)")

        # train = human-train ⊕ all anchor rows; per-row weights: human 1.0, anchor = anchor_weight
        u_tr, p_tr, t_tr = [prof_idx_f[~va_h]], [prob_idx_f[~va_h]], [yf[~va_h]]
        w_tr = [np.ones(int((~va_h).sum()), np.float32)]
        if n_anchor:
            u_tr.append(au); p_tr.append(ap); t_tr.append(at)
            w_tr.append(np.full(len(au), args.anchor_weight, np.float32))
        u_tr = torch.tensor(np.concatenate(u_tr), device=device)
        p_tr = torch.tensor(np.concatenate(p_tr), device=device)
        t_tr = torch.tensor(np.concatenate(t_tr), device=device)
        w_tr = torch.tensor(np.concatenate(w_tr), device=device)
        ntr = len(u_tr)
        tfeat = data["feat"]
        thu = torch.tensor(prof_idx_f, device=device)   # human-only tensors for comparable logloss
        thp = torch.tensor(prob_idx_f, device=device)
        torch.manual_seed(args.seed)
        hard = args.hard_unit_ability
        print(f"  ability gauge: {'HARD (whiten θ to mean0/std1 each step)' if hard else 'SOFT N(0,1) prior'}")
        m = Stage1Shared(data["fdim"], n_users + n_anchor, hidden=tuple(args.diff_hidden),
                         max_floor=args.max_floor, dropout=args.dropout,
                         standardize_ability=hard).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            m.train()
            pm = torch.randperm(ntr, device=device)
            for b0 in range(0, ntr, args.batch):
                bi = pm[b0:b0 + args.batch]
                opt.zero_grad()
                p = m(u_tr[bi], tfeat[p_tr[bi]])
                bce = torch.nn.functional.binary_cross_entropy(p, t_tr[bi], reduction="none")
                loss = (bce * w_tr[bi]).sum() / w_tr[bi].sum().clamp_min(EPS)
                if not hard:                                  # soft N(0,1) prior + mean-pin gauge
                    th = m.theta.weight
                    loss = loss + (0.5 * th.pow(2)).sum() / ntr + args.lambda_theta_mean * th.mean().pow(2)
                loss.backward(); opt.step()
            if ep % args.log_every == 0 or ep == args.epochs - 1:
                m.eval()
                with torch.no_grad():
                    pvh = m(thu, tfeat[thp]).cpu().numpy()
                    cm = m.params(tfeat)[2].mean().item()
                print(f"  ep {ep:4d}  train_ll {logloss(yf[~va_h], pvh[~va_h]):.5f}  "
                      f"val_ll {logloss(yf[va_h], pvh[va_h]):.5f}  mean_floor(corpus) {cm:.4f}")
        m.eval()
        with torch.no_grad():
            b_all, a_all, c_all = m.params(tfeat)
            a, b, c = a_all.cpu().numpy(), b_all.cpu().numpy(), c_all.cpu().numpy()
            theta = m.ability(torch.arange(n_users, device=device)).cpu().numpy()
            bot_theta = (m.ability(torch.arange(n_users, n_users + n_anchor, device=device)).cpu().numpy()
                         if n_anchor else np.array([]))
            lam = float(m.lapse())
        full_prob_idx = data["prob_idx"]   # inspect over ALL attempted (comparable across runs)
        att = np.unique(full_prob_idx); cnt = np.bincount(full_prob_idx, minlength=n_prob)
        pcts = [5, 25, 50, 75, 95]
        # rating = the 80%-clear ability under the 4PL: σ(a(θ−b)) = (0.8−c)/(1−c−λ)
        q = np.clip((CALIBRATION_P - c) / np.clip(1.0 - c - lam, EPS, None), EPS, 1 - EPS)
        rating = b + np.log(q / (1 - q)) / a
        floor_cap = 1.0 - 2.0 * lam
        print(f"\n  learned global lapse (ceiling) λ = {lam:.5f}  |  floor cap = 1−2λ = {floor_cap:.4f}")
        print(f"  ability std {theta.std():.2f} | the shared net predicts a 4PL for ALL "
              f"{n_prob} problems (attempted: {len(att)})")
        if n_anchor:
            rk = [r.strip() for r in args.anchor_ranks.split(",") if r.strip()]
            print(f"  bot-user learned ability θ (human-gauged): "
                  + ", ".join(f"{rk[j]}={bot_theta[j]:+.2f}" for j in range(n_anchor))
                  + f"  [human θ range {theta.min():+.2f}..{theta.max():+.2f}]")
        print(f"  slope  a  p{pcts}: {np.percentile(a, pcts).round(2)}  (corpus)")
        print(f"  diff   b  p{pcts}: {np.percentile(b, pcts).round(2)}")
        print(f"  FLOOR  c  p{pcts}: {np.percentile(c, pcts).round(3)}  mean {c.mean():.3f}  "
              f"frac>0.05 {(c > 0.05).mean():.2f}  frac>0.2 {(c > 0.2).mean():.2f}")
        print(f"  RATING (80%-clear) p[0.1,1,5,50,95,99,99.9]: "
              f"{np.percentile(rating[att], [0.1,1,5,50,95,99,99.9]).round(1)}  "
              f"min {rating[att].min():.1f} max {rating[att].max():.1f}")
        print("\n  --- distributions over ATTEMPTED problems (auto-ranged: p0.5..p99.5) ---")
        text_hist("slope a ", a[att], 0.0, 6.0)
        text_hist("diff  b ", b[att])                 # auto-range: shows the real degeneracy tail
        text_hist("RATING  ", rating[att])
        text_hist("FLOOR c ", c[att], 0.0, float(floor_cap))
        print("\n  floor by attempt count (feature-based ⇒ thin & thick should look alike):")
        print(f"  {'attempts':>10} {'n_prob':>7} {'mean_floor':>11} {'mean_slope':>11} {'mean_diff':>10}")
        for lo, hi in [(0, 1), (1, 5), (5, 20), (20, 100), (100, 10**9)]:
            grp = (cnt >= lo) & (cnt < hi)
            sel = np.where(grp)[0]
            if len(sel) == 0:
                continue
            label = f"{lo}-{hi-1}" if hi < 10**9 else f"{lo}+"
            print(f"  {label:>10} {len(sel):>7} {c[sel].mean():>11.3f} {a[sel].mean():>11.2f} "
                  f"{b[sel].mean():>10.2f}")
        cr = np.corrcoef(c[att], a[att])[0, 1]
        print(f"\n  corr(floor, slope) over attempted = {cr:+.3f}")
        # --- save stage-1 outputs for stage 2 (per-problem priors aligned to keys + abilities + λ) ---
        out_path = os.path.join(PROJECT_ROOT, "ratings_v2", "stage1.npz")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sizes_arr = np.array([s for s, _ in data["keys"]], dtype=np.int64)
        tids_arr = np.array([t for _, t in data["keys"]])
        np.savez(out_path, b_prior=b, a_prior=a, c_prior=c, rating=rating,
                 theta=theta, lapse=np.float32(lam), max_floor=np.float32(floor_cap),
                 sizes=sizes_arr, tids=tids_arr,
                 profiles=np.array(data["profiles"]), hard_unit=np.array(hard))
        print(f"\n  wrote stage-1 outputs (b/a/c/rating priors, θ, λ) → {out_path}")
        return

    if args.stage2:
        print("\n=== STAGE 2: per-problem difficulty/slope residuals (ridge→stage-1 prior) + re-fit "
              "ability on human attempts; floor c & ceiling λ frozen from stage 1 ===")
        s1 = np.load(os.path.join(PROJECT_ROOT, "ratings_v2", "stage1.npz"), allow_pickle=True)
        b_prior, a_prior, c_prior = s1["b_prior"], s1["a_prior"], s1["c_prior"]
        lam, theta_init = float(s1["lapse"]), s1["theta"]
        n_prob, n_users, n = data["n_keys"], data["n_profiles"], len(y)
        assert len(b_prior) == n_prob and len(theta_init) == n_users, \
            "stage-1 outputs misaligned with current data — rerun --stage1 first (same data/seed)"
        prof_idx, prob_idx = data["prof_idx"], data["prob_idx"]
        tprof = torch.tensor(prof_idx, device=device)
        tprob = torch.tensor(prob_idx, device=device)
        ty = torch.tensor(y, device=device)
        np.random.seed(args.seed)                                  # SAME split as stage 1
        va = np.zeros(n, bool); va[np.random.permutation(n)[:int(0.15 * n)]] = True
        tr_idx = torch.tensor(np.where(~va)[0], device=device); ntr = len(tr_idx)
        torch.manual_seed(args.seed)
        m = Stage2(b_prior, a_prior, c_prior, lam, theta_init, standardize_ability=True).to(device)
        m.eval()
        with torch.no_grad():
            ll0 = logloss(y[va], m(tprof, tprob).cpu().numpy()[va])   # prior-only (δ=0): stage-1 alone
        print(f"  loaded stage-1 priors (λ={lam:.5f}) | prior-only held-out logloss {ll0:.5f} | "
              f"ridge λ_resid {args.lambda_resid}")
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            m.train()
            pm = tr_idx[torch.randperm(ntr, device=device)]
            for b0 in range(0, ntr, args.batch):
                bi = pm[b0:b0 + args.batch]
                opt.zero_grad()
                p = m(tprof[bi], tprob[bi])
                loss = (torch.nn.functional.binary_cross_entropy(p, ty[bi])
                        + args.lambda_resid * (m.delta_b.weight.pow(2).mean()
                                               + m.delta_a.weight.pow(2).mean()))
                loss.backward(); opt.step()
            if ep % args.log_every == 0 or ep == args.epochs - 1:
                m.eval()
                with torch.no_grad():
                    pv = m(tprof, tprob).cpu().numpy()
                print(f"  ep {ep:4d}  train_ll {logloss(y[~va], pv[~va]):.5f}  "
                      f"val_ll {logloss(y[va], pv[va]):.5f}")
        m.eval()
        with torch.no_grad():
            ll1 = logloss(y[va], m(tprof, tprob).cpu().numpy()[va])
            allp = torch.arange(n_prob, device=device)
            b_eff, a_eff, c_eff = m.eff(allp)
            b_e, a_e, c_e = b_eff.cpu().numpy(), a_eff.cpu().numpy(), c_eff.cpu().numpy()
            theta = m.ability(torch.arange(n_users, device=device)).cpu().numpy()
        att = np.unique(prob_idx)
        db = m.delta_b.weight.detach().squeeze(-1).cpu().numpy()
        da = m.delta_a.weight.detach().squeeze(-1).cpu().numpy()
        print(f"\n  prior-only val {ll0:.5f}  →  stage-2 val {ll1:.5f}   "
              f"Δ {ll1 - ll0:+.5f} ({'better' if ll1 < ll0 else 'worse'})")
        print(f"  residual sizes (attempted): |δb| mean {np.abs(db[att]).mean():.3f}  "
              f"|δlog_a| mean {np.abs(da[att]).mean():.3f}  | ability std {theta.std():.2f}")
        q = np.clip((CALIBRATION_P - c_e) / np.clip(1.0 - c_e - lam, EPS, None), EPS, 1 - EPS)
        rating = b_e + np.log(q / (1 - q)) / a_e
        out_path = os.path.join(PROJECT_ROOT, "ratings_v2", "stage2.npz")
        np.savez(out_path, rating=rating, slope=a_e, floor=c_e, lapse=np.float32(lam),
                 sizes=s1["sizes"], tids=s1["tids"])
        print(f"  wrote stage-2 outputs (rating/slope/floor + λ) → {out_path}")
        return

    if args.room_sweep:
        args.dro = True
        print("\n=== floor vs per-problem 'room' (lambda_resid): does MORE room use the floor MORE "
              "(co-adaptation hypothesis) or LESS (slopes-mimic hypothesis)? ===")
        print("    free per-problem floor (no bot prior) at each room budget; bot-pinned floor as the")
        print("    generalization reference. lower lambda_resid = MORE per-problem rating/slope room.")
        print(f"  {'lambda_resid':>12} {'free_floor_mean':>15} {'std':>7} {'free_val':>9} {'pinned_val':>10}")
        for lr in args.room_sweep:
            args.lambda_resid = lr
            mf, pf, maskf, _ = fit(data, args, device, False, 0.0, args.seed, free_floor=True)
            mp, pp, maskp, _ = fit(data, args, device, False, 0.0, args.seed)   # bot-pinned (production)
            with torch.no_grad():
                allk = torch.arange(data["n_keys"], device=device)
                floors = mf.floor_for(allk, data["feat"]).cpu().numpy()
            print(f"  {lr:>12.2f} {floors.mean():>15.4f} {floors.std():>7.4f} "
                  f"{logloss(y[maskf], pf[maskf]):>9.5f} {logloss(y[maskp], pp[maskp]):>10.5f}")
        return

    if args.free_floor:
        args.dro = True
        print("\n=== TRULY FREE per-problem floor (no bot prior), 2D-DRO ===")
        print(f"    free global bias init {args.floor_free_init}; per-problem resid ridge "
              f"λ={args.lambda_floor}. Does the avg floor grow (data wants one) or collapse?")
        m, pred, mask, _ = fit(data, args, device, False, 0.0, args.seed, trace=True, free_floor=True)
        info = m._dro
        zbin, nzb, nrb, tr, z_row = info["zbin"], info["nzb"], info["nrb"], info["tr_rows"], info["z_row"]
        with torch.no_grad():
            allk = torch.arange(data["n_keys"], device=device)
            floors = m.floor_for(allk, data["feat"]).cpu().numpy()
        att = np.unique(data["prob_idx"])
        print(f"\n  final free floor: mean {floors.mean():.4f}  mean@attempted {floors[att].mean():.4f}"
              f"  std {floors.std():.4f}  p[5,50,95] {np.percentile(floors, [5,50,95]).round(3)}")
        print("  far-below fit (does the free floor match emp_clr better than pinned 0.36?):")
        print(f"  {'zbin':>4} {'mean_z':>7} {'emp_clr':>8} {'pred':>7} {'mean_floor':>10}")
        for zb in range(min(4, nzb)):                   # the 4 most far-below z-bins
            sel = tr & (zbin == zb); n = int(sel.sum())
            if not n:
                continue
            fl_sel = floors[data["prob_idx"][sel]]
            print(f"  {zb:>4} {z_row[sel].mean():>7.2f} {y[sel].mean():>8.4f} "
                  f"{pred[sel].mean():>7.4f} {fl_sel.mean():>10.4f}")
        return

    if args.dro_diag:
        args.dro = True
        print("\n=== why DRO doesn't keep the lapse: saturated-bin anatomy (free lapse) ===")
        m, pred, mask, lam = fit(data, args, device, True, args.inits[len(args.inits) // 2], args.seed)
        info, q = m._dro, m._q_final
        rb, zbin, nrb, nzb = info["rb"], info["zbin"], info["nrb"], info["nzb"]
        tr, z_row = info["tr_rows"], info["z_row"]
        print(f"  converged lapse = {lam:.6f}  (collapsed despite DRO)")
        print(f"  {nzb} z-bins (relative gap) × {nrb} rating-bins; per z-bin over TRAIN rows, "
              f"low→high gap.\n  q_sum = converged DRO weight on the z-bin (uniform would be "
              f"{1.0/nzb:.3f}); centered = loss − base_rate_entropy (≈0 ⇒ calibrated, no DRO signal).")
        hdr = (f"  {'zbin':>4} {'n':>7} {'mean_z':>7} {'emp_clr':>8} {'pred':>7} "
               f"{'loss':>7} {'base_H':>7} {'center':>8} {'q_sum':>6} {'cat%':>6}")
        print(hdr)
        for zb in range(nzb):
            sel = tr & (zbin == zb)
            n = int(sel.sum())
            if n == 0:
                continue
            ys = y[sel]; ps = np.clip(pred[sel], 1e-7, 1 - 1e-7)
            emp, mp = ys.mean(), ps.mean()
            loss = float(-(ys * np.log(ps) + (1 - ys) * np.log(1 - ps)).mean())
            be = float(-(emp * np.log(max(emp, 1e-9)) + (1 - emp) * np.log(max(1 - emp, 1e-9))))
            qsum = float(q[[zb * nrb + r for r in range(nrb)]].sum())
            cat = float(((ps >= 0.999) & (ys == 0)).mean() * 100)
            print(f"  {zb:>4} {n:>7} {z_row[sel].mean():>7.2f} {emp:>8.4f} {mp:>7.4f} "
                  f"{loss:>7.4f} {be:>7.4f} {loss-be:>+8.4f} {qsum:>6.3f} {cat:>6.3f}")
        return

    if args.free_floor_scale:
        obj = "2D-DRO" if args.dro else "mean-BCE"
        print(f"\n=== FREE global floor scale γ (c_p = γ·c_bot, no anchor) | objective {obj} ===")
        print("    production pins γ=0.70 externally; does a FREE γ hold there or collapse to ~0?")
        for g0 in args.floor_inits:
            print(f"\n  init γ={g0}:")
            _, p, mask, _ = fit(data, args, device, False, 0.0, args.seed, trace=True,
                                free_floor_scale=True, floor_scale_init=g0)
        return

    if args.trace:
        print("\n=== convergence trace (seed 0) ===")
        print("  [OFF / 3PL]")
        _, p_off, mask, _ = fit(data, args, device, False, 0.0, 0, trace=True)
        print("  [ON / 4PL free]")
        _, p_on, mask2, lam = fit(data, args, device, True, args.inits[len(args.inits)//2], 0, trace=True)
        yv = y[mask]
        print(f"\n  final fitted lapse {lam:.5f} | overall val logloss "
              f"off {logloss(yv, p_off[mask]):.5f}  on {logloss(yv, p_on[mask]):.5f}  "
              f"Δ {logloss(yv, p_on[mask])-logloss(yv, p_off[mask]):+.5f}")
        for t in [0.95, 0.98, 0.99]:
            sat = mask & (p_off >= t); ns = int(sat.sum())
            if ns:
                ys = y[sat]
                print(f"    pred≥{t:.2f}: n {ns:6d} emp_fail {(1-ys).mean():.4f} "
                      f"ll off {logloss(ys, p_off[sat]):.5f} on {logloss(ys, p_on[sat]):.5f} "
                      f"Δ {logloss(ys, p_on[sat])-logloss(ys, p_off[sat]):+.5f}")
        return

    if args.fixed is not None:
        print("\n=== FIXED (frozen) lapse: the narrow-regime tradeoff (NOT global accuracy) ===")
        print("    overall      = mean val logloss (the small global cost we trade away)")
        print("    CAT_fail     = mean loss on the CATASTROPHIC events: held-out FAILURES the")
        print("                   λ=0 model called clear≥0.99 (a slip the estimator over-trusts)")
        print("    sat_success  = mean loss on the saturated SUCCESSES (the cost of the cap)")
        print("    worst, p99.9 = max / 99.9-pct single-row val loss (leverage the cap bounds)")
        print("    (catastrophic/saturated sets are FIXED from the λ=0 fit so rows don't move)")

        def rowloss(yy, pp):
            pp = np.clip(pp, 1e-7, 1 - 1e-7)
            return -(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))

        for seed in args.seeds:
            print(f"\n  seed {seed}:")
            cat = sat = None
            for val in args.fixed:
                lz = val > 0
                _, p, mask, _ = fit(data, args, device, lz, max(val, 1e-9), seed, freeze=True)
                if cat is None:                       # λ=0 baseline defines the fixed sets
                    sat99 = mask & (p >= 0.99)
                    cat = sat99 & (y == 0)            # catastrophic: confident-but-failed
                    sat = sat99 & (y == 1)            # saturated successes (the cost side)
                rl = rowloss(y[mask], p[mask])
                print(f"    λ={val:.4f}: overall {logloss(y[mask], p[mask]):.5f} | "
                      f"CAT_fail {rowloss(y[cat], p[cat]).mean():.3f} (n {int(cat.sum())}) | "
                      f"sat_success {rowloss(y[sat], p[sat]).mean():.4f} (n {int(sat.sum())}) | "
                      f"worst {rl.max():.2f}  p99.9 {np.percentile(rl, 99.9):.2f}")
        return

    sat_thresh = [0.95, 0.98, 0.99]
    print("\n=== free-fit lapse: init/seed independence (is it forced?) ===")
    for seed in args.seeds:
        for init in args.inits:
            _, _, _, lam = fit(data, args, device, True, init, seed)
            print(f"  seed {seed}  init {init:.4f}  ->  fitted lapse = {lam:.5f}")

    print("\n=== held-out A/B (lapse OFF = 3PL  vs  lapse ON = 4PL, free) ===")
    for seed in args.seeds:
        m_off, p_off, mask, _ = fit(data, args, device, False, 0.0, seed)
        _, p_on, mask2, lam = fit(data, args, device, True, args.inits[len(args.inits)//2], seed)
        assert np.array_equal(mask, mask2)
        yv = y[mask]
        ll_off, ll_on = logloss(yv, p_off[mask]), logloss(yv, p_on[mask])
        plateau, n_easy = posthoc_plateau(m_off, data, device)
        print(f"\n  seed {seed} | val rows {mask.sum()} | fitted lapse {lam:.5f} | "
              f"posthoc plateau (inner≥0.99, in-sample) {plateau:.5f} (n={n_easy})")
        print(f"    overall val logloss   off {ll_off:.5f}   on {ll_on:.5f}   "
              f"Δ {ll_on-ll_off:+.5f} ({'better' if ll_on<ll_off else 'worse'})")
        for t in sat_thresh:
            sat = mask & (p_off >= t)   # saturated regime defined by the OFF model
            ns = int(sat.sum())
            if ns == 0:
                continue
            ys = y[sat]
            print(f"    pred≥{t:.2f}: n {ns:6d}  emp_fail {(1-ys).mean():.4f}  "
                  f"logloss off {logloss(ys, p_off[sat]):.5f}  on {logloss(ys, p_on[sat]):.5f}  "
                  f"Δ {logloss(ys, p_on[sat])-logloss(ys, p_off[sat]):+.5f}")


if __name__ == "__main__":
    main()
