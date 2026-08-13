#!/usr/bin/env python3
"""ONE model producing the rating and the K skill axes together — the production
trainer, replacing `pipeline.train_rating` + `pipeline.build_axes`.

    P = c_p + (1 - c_p - lambda) * sigmoid( a_p*(theta_u - b_p)
                                            + sum_k g_k(x_p)*v_u[k] )

        a_p, b_p, c_p   per-problem discrimination, difficulty, guessing floor
        g_k(x_p)        per-problem loading on axis k
        theta_u         per-user ability
        v_u[k]          per-user position on axis k
        lambda          one global slip rate

Under the standard multidimensional-IRT reading, fitting these in two stages is
artificial: the axes are factors 2..K+1 of one model and the rating is factor 1.
ONE TRUNK produces every output, axes included — a Linear's output rows are
independent, so a shared head emitting (b, a, c, g_1..g_K) is exactly this.

PRODUCTION ONLY. No argparse, no flags, config baked in as constants, always seed
0, always writes `ratings/` and `axes/`. The experiment surface — every ablation,
sweep, holdout form and alternative architecture — lives in
`auxiliary.rating_dev.train_joint_mirt`, so a test tweak cannot reach the shipped
fit. Measurements behind every constant here: `docs/mirt_benchmark.md`.

Outputs, matching what the two-stage pair wrote so downstream is unchanged:

  ratings/<size>/tsumego_ratings.csv    tsumego_id, rating (the 80%-clear point)
  ratings/<size>/tsumego_slopes.csv     tsumego_id, slope
  ratings/<size>/tsumego_floors.csv     tsumego_id, floor
  ratings/user_theta.csv                profile, theta, n_attempts
  ratings/rating_constants.json         {"fat_finger_rate": lambda, "slope_scale": s}
  axes/per_problem.csv                  tsumego_id, size, embed_0..embed_{K-1}
  axes/mirt_axes.npz                    the model + the content-hash embedding_version

NOT axes/axes.npz: that is tracked and holds build_axes' FEATURE-space map, whose
readers require keys (rms, v_sd, feat_mu, feat_sd) that describe a standardization
this model does not have. Same name, different object — writing there would be a
silent format break while the two-stage path is still needed.

WHAT THIS DOES NOT DO, and must be run alongside until it does:
  - `axes/per_cell/` — a K-vector for every legal cell, needed because the client
    scores the move the user PICKED and not only the answer. The two-stage axis
    map was linear on features so a cell cost one dot product; this map reads the
    trunk, so a cell costs a trunk pass, about 7.7M corpus-wide.
  - K selection. `build_axes` grows K with the data; K is fixed here.
  - `axes/labels.json`, which a human writes against a fit.

SLOPE_SCALE IS 1.0, within noise of this model's MEASURED optimum. Swept on
held-out users and split by whether training saw the problem — the split is
necessary, since a never-trained problem is priced from the feature prior alone
and honestly wants a shallower slope, so pooling those rows in drags the optimum
down and confounds a leak with a pricing difficulty:

    best s        all     SEEN    unseen
      production  0.700   0.800   0.700
      MIRT        0.900   0.900   0.800

The dilution is worth ~0.1 in s, against the ~0.9 separating 0.8 from 1.694. So
train_rating's multiplier really was fitting its own leak — on problems training
DID see, its optimum is 0.80 and shipping 1.694 costs 19.9 mnat, handing the
client 1.694^2 ~ 2.87x the Fisher information it earned, so se runs too tight and
rank convergence too fast. This model's optimum is 0.90; 1.0 costs 0.2 mnat
against it, inside the 0.85 mnat seed noise.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline.train_rating import (  # noqa: E402
    DIFF_HIDDEN, DROPOUT, EPS, FLOOR_CAP, INPUT_DROPOUT, LAPSE_CAP, LAPSE_INIT,
    PCA_DIM, SEED, SIZES, SLOPE_FLOOR, clear_logit_vec)
from utils.rating.features import (  # noqa: E402
    _cache_ids_path, load_first_time_attempts, load_joint_activations,
    load_joint_bot_winrate, load_joint_game_phase, load_joint_move_features)

K_AXES = 3                       # inherited from build_axes' selection; see docstring
EPOCHS = 10000                   # runaway failsafe; the patience rule ends the run
CHECK_EVERY = 10
FLAT_PATIENCE = 5                # checks without a new best before stopping
BATCH = 65536
# SPLIT RATES, and the split is mandatory. What the grid measures is the RATIO
# lr_field/lr_net, monotone and saturating near 30x: below 10x the fit loses
# 2.5-5 mnat. Given a ratio past 30x a slower net buys the last 0.8; 1e-4 and
# 3e-4 tie, and 3e-4 converges in half the epochs.
LR_NET = 3e-4
LR_FIELD = 1e-2
V_RIDGE = 0.5                    # not binding: 0.05/0.15/0.5 score alike
V_MEAN_PENALTY = 10.0            # stops an axis carrying POPULATION difficulty
ORTHO_WEIGHT = 0.5
GAUGE_MOM = 0.01
# ONE barrier, on the slope, one-sided, acting on the PRE-SOFTPLUS output. The
# rating is b + kappa(c)/a, so a slope collapsing toward its floor sends it to
# infinity. Without this, slopes fall to a 0.94 median and ratings blow out to a
# range of [-225, +146] — a 22 mnat loss. There is no floor barrier: the floor is
# bounded structurally by cap = (1 - 2*lambda).
SHALLOW_TAU = 0.1
SHALLOW_W = 0.3
# Epochs with v pinned at zero before the axis warm start reads the model's own
# residuals. A FIXED count beats running to the axis-free fit's own convergence,
# which sounds strictly better but is seed-dependent (380 and 550 on two seeds)
# and the outcome is epoch-sensitive, so the criterion inherits that variance.
BURN_IN_EPOCHS = 300
# IDENTIFIABILITY, the only inclusion rule. Under the 4PL an all-correct record's
# likelihood rises monotonically to (1-lambda)^n as ability grows without bound,
# and an all-wrong one to (1-c)^n as it falls, so neither has an interior
# maximum. One of each outcome is the condition for the ability to be estimable
# at all; anything above it is a judgement about how WELL, which the likelihood
# already makes by giving a thin record few terms.
MIN_BOTH_OUTCOMES = 1
STOP_ROW_FRAC = 0.03             # rows held aside purely to detect convergence
OUT_DIR = os.path.join(PROJECT_ROOT, "ratings")
AXES_DIR = os.path.join(PROJECT_ROOT, "axes")


class JointMIRT:
    """Compensatory M4PL with per-problem parameters from a shared trunk.

    The one unavoidable departure from textbook MIRT is that a_p, b_p, c_p and
    g_k come from a FEATURE NET rather than being free per item: a third of the
    corpus has no attempts at all, so free per-item parameters cannot price it.
    """

    def __init__(self, fdim, n_users, input_dim, device, seed):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.input_dim = int(input_dim)

        # theta before the trunk so the RNG draw order is fixed.
        self.theta = nn.Embedding(n_users, 1).to(device)
        nn.init.normal_(self.theta.weight, std=1.0)

        class SwiGLU(nn.Module):
            """silu(xW) * xV, replacing [Linear, ReLU]."""

            def __init__(self, d_in, d_out):
                super().__init__()
                self.w = nn.Linear(d_in, d_out)
                self.v = nn.Linear(d_in, d_out)

            def forward(self, x):
                return torch.nn.functional.silu(self.w(x)) * self.v(x)

        layers, d = [], fdim
        for h in DIFF_HIDDEN:            # 128x64, an interior optimum both ways
            layers += [SwiGLU(d, h), nn.Dropout(DROPOUT)]
            d = h
        self.trunk = nn.Sequential(*layers).to(device)
        self.head = nn.Linear(d, 3).to(device)          # difficulty, slope, floor
        # The axis map, on the TRUNK's representation. Same linear map, different
        # input: it adds no capacity (fewer parameters than reading the 212
        # features) and the axis objective reaches the trunk, so the trunk learns
        # what the AXES need and not only what the rating needs. Detaching would
        # make this feature reuse rather than a joint fit.
        self.g_lin = nn.Linear(d, K_AXES, bias=False).to(device)
        self.v = torch.zeros(n_users, K_AXES, device=device, requires_grad=True)
        li = float(np.clip(LAPSE_INIT / LAPSE_CAP, EPS, 1.0 - EPS))
        self.lapse_raw = torch.tensor(float(np.log(li / (1.0 - li))),
                                      device=device, requires_grad=True)
        # Rating-gauge EMA, on DIFFICULTY — see problem_params.
        self.b_mu = torch.zeros((), device=device)
        self.b_sd = torch.ones((), device=device)
        self.training = True

    def params(self):
        return (list(self.trunk.parameters()) + list(self.head.parameters())
                + list(self.g_lin.parameters()) + list(self.theta.parameters())
                + [self.v, self.lapse_raw])

    def param_groups(self):
        """SPLIT RATES. The network must train an order of magnitude slower than
        the free per-user fields; under one shared rate this fit diverges late,
        in eval mode, in every arrangement tried. The rating trainer gets away
        with a single rate because its only per-user parameter is one ability
        embedding; this model has that plus a K-dimensional axis position."""
        net = (list(self.trunk.parameters()) + list(self.head.parameters())
               + list(self.g_lin.parameters()))
        field = list(self.theta.parameters()) + [self.v, self.lapse_raw]
        return [{"params": net, "lr": LR_NET},
                {"params": field, "lr": LR_FIELD}]

    def train(self, on):
        self.training = on
        self.trunk.train(on)

    def lapse(self):
        return LAPSE_CAP * self.torch.sigmoid(self.lapse_raw)

    def _drop(self, feat):
        """COLUMN dropout over the wide input region as ONE mask — activations,
        bot winrate and move evals together. The two game-phase columns and the
        size indicator sit past that region and are deliberately exempt, so a new
        wide block belongs INSIDE input_dim and is then masked automatically."""
        torch = self.torch
        if not self.training or INPUT_DROPOUT <= 0.0 or self.input_dim <= 0:
            return feat
        keep = (torch.rand(len(feat), self.input_dim, device=feat.device)
                >= INPUT_DROPOUT).float() / (1.0 - INPUT_DROPOUT)
        return torch.cat([feat[:, :self.input_dim] * keep,
                          feat[:, self.input_dim:]], dim=1)

    def slope_logit(self, feat):
        """Pre-softplus slope output — the space the barrier acts in, where the
        penalty falls away properly for healthy slopes."""
        return self.head(self.trunk(feat))[:, 1]

    def problem_params(self, feat):
        """(difficulty b, discrimination a, floor c, axis loadings g).

        THE GAUGE MUST SIT ON DIFFICULTY, NOT ON AN INTERCEPT. Two freedoms have
        to be removed, a shift of ability and a rescaling of it, and whitening an
        ADDITIVE intercept removes only the first: ability*s with a/s leaves an
        additive intercept untouched, so its whitening is satisfied and nothing
        objects, while difficulty is MULTIPLIED by the slope and must rescale
        with it. The shipped rating is an ability, so under the unblocked
        rescaling every rating scales and the whole axis floats."""
        torch, nn = self.torch, self.nn
        h = self.trunk(self._drop(feat))
        o = self.head(h)
        b = o[:, 0]
        if self.training:
            mu, sd = b.mean(), b.std() + EPS
            with torch.no_grad():
                self.b_mu.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * mu)
                self.b_sd.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * sd)
        else:
            mu, sd = self.b_mu, self.b_sd
        b = (b - mu) / sd
        a = nn.functional.softplus(o[:, 1]).clamp(min=SLOPE_FLOOR)
        lam = self.lapse()
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=FLOOR_CAP)
        c = cap * torch.sigmoid(o[:, 2])
        g = self.g_lin(h)
        g = g / (g.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
        return b, a, c, g

    def trunk_out(self, feat):
        """The axis map's input basis with dropout off — the space the warm
        start's residual directions must live in, since that is what g_lin is
        parameterized over."""
        torch = self.torch
        was = self.training
        self.train(False)
        with torch.no_grad():
            out = self.trunk(feat)
        self.train(was)
        return out

    def forward(self, uid, feat):
        torch = self.torch
        b, a, c, g = self.problem_params(feat)
        th = self.theta(uid).squeeze(-1)
        lam = self.lapse()
        z = (a * (th - b) + (g * self.v[uid]).sum(-1)).clamp(-35, 35)
        return c + (1.0 - c - lam) * torch.sigmoid(z), g


def burn_in_directions(m, tfeat, tp, tu, ty, tr, n_users):
    """Top-k escape directions from the model's OWN axis-free residuals.

    At v = 0 the loading gradient is identically zero — which is why a warm start
    is needed at all — so the burn-in trains EXACTLY the axis-free baseline the
    residual scores are defined against. It is a schedule point inside one fit,
    not a stage: those epochs are the rating heads' own.

        s_u = sum_p (y_up - p_hat_up) * h_p / n_u

    CENTRING across users is load-bearing: the uncentred lead direction is the
    one in which the baseline mispredicts for EVERYONE, i.e. population residual
    difficulty, a property of problems rather than people. DIVISION BY n_u is
    too: s_u is a SUM, so its weight in a second moment grows with n_u squared,
    and on the raw form a single heavy account supplied a third of every axis.
    """
    torch = m.torch
    xf = m.trunk_out(tfeat)
    S = torch.zeros(n_users, xf.shape[1], dtype=torch.float64, device=xf.device)
    n = torch.zeros(n_users, dtype=torch.float64, device=xf.device)
    was = m.training
    m.train(False)
    with torch.no_grad():
        for i in range(0, len(tr), BATCH):
            bi = tr[i:i + BATCH]
            pred, _g = m.forward(tu[bi], tfeat[tp[bi]])
            resid = (ty[bi] - pred).double()
            S.index_add_(0, tu[bi], resid[:, None] * xf[tp[bi]].double())
            n.index_add_(0, tu[bi], torch.ones_like(resid))
    m.train(was)
    S, n = S.cpu().numpy(), n.cpu().numpy()
    seen = n > 0
    S = S[seen] / n[seen, None]
    S = S - S.mean(axis=0, keepdims=True)
    _ev, vec = np.linalg.eigh(S.T @ S / S.shape[0])
    top = vec[:, ::-1][:, :K_AXES].T
    return (top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-9)
            ).astype(np.float32), int(seen.sum())


def main():
    import pandas as pd
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, k_act = load_joint_activations(SIZES, tids_by_size, PCA_DIM, SEED)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    mv, nmv = load_joint_move_features(SIZES, tids_by_size)
    ph, nph = load_joint_game_phase(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]
    key_idx = {key: i for i, key in enumerate(keys)}
    zero_mv, zero_ph = np.zeros(nmv, np.float32), np.zeros(nph, np.float32)
    input_dim = k_act + nb + nmv
    fdim = input_dim + nph + 1
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        feat[i] = np.concatenate([act[key], bw[key], mv.get(key, zero_mv),
                                  ph.get(key, zero_ph),
                                  [1.0 if key[0] == 13 else 0.0]])
    print(f"loaded in {time.time() - t0:.0f}s: {len(keys)} problems, "
          f"feat {fdim} (dropout region {input_dim})", flush=True)

    # ONE ability unit per user, over their whole history. Measured best against
    # every windowed arrangement, and it makes the ability and axis-position
    # granularities coincide — removing the mixed-granularity question the
    # two-stage design has to answer rather than answering it.
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    keep = set(g.index[(g["c"] >= MIN_BOTH_OUTCOMES)
                       & (g["w"] >= MIN_BOTH_OUTCOMES)])
    df = df[df["user_id"].isin(keep)].copy()
    prob2size = dict(zip(df["problem"], df["size"]))
    df = df[df["problem"].map(lambda t: (prob2size.get(t), t) in key_idx)]
    users = pd.Categorical(df["user_id"].astype(str))
    user_idx = users.codes.astype(np.int64)
    n_users = len(users.categories)
    y = df["success"].to_numpy(np.float32)
    prob_idx = df["problem"].map(
        lambda t: key_idx[(prob2size[t], t)]).to_numpy(np.int64)
    print(f"  {len(df)} rows | {n_users} users | base rate {y.mean():.3f}",
          flush=True)

    tfeat = torch.tensor(feat, device=device)
    tu = torch.tensor(user_idx, device=device)
    ty = torch.tensor(y, device=device)
    tp = torch.tensor(prob_idx, device=device)

    # A convergence detector, not a criterion: every row spent here is a row the
    # model does not learn from, so it is small. Stopping on TRAINING loss alone
    # would pick the wrong epoch — held-out quality peaks hundreds of epochs
    # before the training loss flattens.
    rng = np.random.default_rng(SEED)
    is_stop = rng.random(len(df)) < STOP_ROW_FRAC
    tr = torch.tensor(np.where(~is_stop)[0], device=device)
    i_stop = torch.tensor(np.where(is_stop)[0], device=device)
    print(f"  stop signal: {int(is_stop.sum())} rows held aside", flush=True)

    m = JointMIRT(fdim, n_users, input_dim, device, SEED)
    opt = torch.optim.Adam(m.param_groups())
    o_tau = float(np.log(np.expm1(SHALLOW_TAU)))
    n = len(tr)
    best, best_state, flat, best_epoch = float("inf"), None, 0, 0
    in_burn, stopped = True, False
    m.v.requires_grad_(False)
    print(f"\n=== joint fit | K={K_AXES} | burn-in {BURN_IN_EPOCHS} ===",
          flush=True)

    for ep in range(EPOCHS):
        m.train(True)
        perm = tr[torch.randperm(n, device=device)]
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            pred, gh = m.forward(tu[bi], tfeat[tp[bi]])
            pred = pred.clamp(EPS, 1 - EPS)
            loss = torch.nn.functional.binary_cross_entropy(
                pred, ty[bi], reduction="none").sum() / len(bi)
            if not in_burn:
                # While v is pinned these are identically zero or act on a g the
                # warm start is about to overwrite.
                loss = loss + V_RIDGE * (m.v[tu[bi]] ** 2).sum() / len(bi)
                loss = loss + V_MEAN_PENALTY * (m.v.mean(dim=0) ** 2).sum()
                Mc = gh - gh.mean(dim=0, keepdim=True)
                Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
                Cm = Mn.T @ Mn
                loss = loss + ORTHO_WEIGHT * (
                    (Cm - torch.eye(K_AXES, device=device)) ** 2).sum()
            m.train(False)          # the barrier reads the corpus, dropout off
            short = torch.nn.functional.softplus(
                o_tau - m.slope_logit(tfeat)).pow(2).mean()
            loss = loss + SHALLOW_W * short
            m.train(True)
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep + 1) % CHECK_EVERY == 0:
            m.train(False)
            with torch.no_grad():
                pr, _ = m.forward(tu, tfeat[tp])
                pr = pr.clamp(EPS, 1 - EPS)
                bce_tr = float(torch.nn.functional.binary_cross_entropy(
                    pr[tr], ty[tr]))
                bce_st = float(torch.nn.functional.binary_cross_entropy(
                    pr[i_stop], ty[i_stop]))
            improved = bce_st < best - 1e-6
            if improved:
                best, best_epoch = bce_st, ep + 1
                best_state = [q.detach().clone() for q in m.params()]
            flat = 0 if improved else flat + 1
            print(f"  {'[burn] ' if in_burn else ''}epoch {ep + 1:4d}  "
                  f"train {bce_tr:.4f}  stop {bce_st:.4f}  "
                  f"best {best:.4f}@{best_epoch}  stale {flat}/{FLAT_PATIENCE}",
                  flush=True)
            if in_burn:
                if (ep + 1) >= BURN_IN_EPOCHS:
                    # Taken IN PLACE, not from the axis-free best: rewinding the
                    # parameters while Adam's moments describe a later trajectory
                    # resumes the fit with one epoch's weights under another's
                    # momentum, which costs 1.4 mnat.
                    gi, n_seen = burn_in_directions(m, tfeat, tp, tu, ty, tr,
                                                    n_users)
                    with torch.no_grad():
                        m.g_lin.weight.copy_(torch.tensor(gi, device=device))
                    m.v.requires_grad_(True)
                    best, best_state, flat, best_epoch = float("inf"), None, 0, 0
                    in_burn = False
                    print(f"  warm start from {n_seen} users' own residuals; "
                          f"v released at epoch {ep + 1}", flush=True)
                continue            # never END the run during the burn-in
            if flat >= FLAT_PATIENCE:
                print(f"  stopped at epoch {ep + 1}; restoring best "
                      f"{best:.4f}@{best_epoch}", flush=True)
                stopped = True
                break

    if not stopped:
        raise SystemExit(
            f"EPOCH CAP HIT: ran {EPOCHS} epochs without the patience rule "
            f"firing. The fit never plateaued, so this is not a converged "
            f"solution and its numbers are not comparable with anything.")
    with torch.no_grad():
        for q, b_ in zip(m.params(), best_state):
            q.copy_(b_)

    # --- outputs
    m.train(False)
    with torch.no_grad():
        b_all, a_all, c_all, g_all = m.problem_params(tfeat)
        lam = float(m.lapse())
    b_np, a_np = b_all.cpu().numpy(), a_all.cpu().numpy()
    c_np, g_np = c_all.cpu().numpy(), g_all.cpu().numpy()
    # The ability at which the curve reaches the calibration point. AN EXTREME
    # RATING IS NOT A SYMPTOM: the corpus holds genuinely trivial problems and a
    # KataGo-hard tail nobody clears, and the rank coordinate is unbounded. What
    # distinguishes a real difficulty from a collapsed slope is the SLOPE
    # distribution and the share at the clamp, printed below.
    rating = b_np + clear_logit_vec(c_np, lam) / np.maximum(a_np, 1e-6)
    theta = m.theta.weight.detach().cpu().numpy().ravel()

    print(f"  lambda {lam:.5f} | slope p[1,50,99] "
          f"{np.percentile(a_np, [1, 50, 99]).round(3)} | at floor "
          f"{np.mean(a_np <= SLOPE_FLOOR * 1.01):.2%} | rating p[1,50,99] "
          f"{np.percentile(rating, [1, 50, 99]).round(2)}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(OUT_DIR, str(s))
        os.makedirs(sdir, exist_ok=True)
        for name, arr in (("tsumego_ratings.csv", rating), ("tsumego_slopes.csv", a_np),
                          ("tsumego_floors.csv", c_np)):
            col = name.split("_")[1].split(".")[0].rstrip("s")
            pd.DataFrame({"tsumego_id": tids, col: arr[sel]}).sort_values(
                col, ascending=False).to_csv(os.path.join(sdir, name), index=False)
        print(f"  size {s}: wrote {len(tids)} ratings | rating "
              f"[{rating[sel].min():.2f},{rating[sel].max():.2f}] | floor mean "
              f"{c_np[sel].mean():.3f}", flush=True)

    natt = pd.Series(user_idx).value_counts()
    pd.DataFrame({"profile": list(users.categories), "theta": theta,
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_users)]}
                 ).to_csv(os.path.join(OUT_DIR, "user_theta.csv"), index=False)
    with open(os.path.join(OUT_DIR, "rating_constants.json"), "w") as f:
        json.dump({"fat_finger_rate": round(lam, 4), "slope_scale": 1.0,
                   "scale_fit": "none_shipped_as_fitted"}, f, indent=2)

    os.makedirs(AXES_DIR, exist_ok=True)
    rows = {"tsumego_id": [k[1] for k in keys], "size": [k[0] for k in keys]}
    for k in range(K_AXES):
        rows[f"embed_{k}"] = g_np[:, k].round(5)
    pd.DataFrame(rows).to_csv(os.path.join(AXES_DIR, "per_problem.csv"),
                              index=False)
    # The content hash the PWA's embedding-invalidation gate rides: it must move
    # when and only when the projection would give different vectors.
    h = hashlib.sha256()
    for arr in (m.g_lin.weight.detach().cpu().numpy(),
                m.head.weight.detach().cpu().numpy()):
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    ev = h.hexdigest()[:16]
    # DELIBERATELY NOT axes/axes.npz. That file is tracked, and build_axes writes
    # a FEATURE-space map there — W is (K, 216) with rms/v_sd/feat_mu/feat_sd
    # describing the standardization it was fitted under, and readers such as
    # eval_user_holdout.project_loadings require those keys. This model's map is
    # (K, trunk_width) over the trunk's output: a different object under the same
    # name, so writing it there would not be an upgrade but a silent format
    # break, and the two-stage path is still needed for per_cell and labels.
    np.savez_compressed(os.path.join(AXES_DIR, "mirt_axes.npz"),
                        W=m.g_lin.weight.detach().cpu().numpy(),
                        k=np.array(K_AXES),
                        trunk_width=np.array(m.g_lin.weight.shape[1]),
                        embedding_version=np.array(ev))
    print(f"  wrote axes/per_problem.csv + mirt_axes.npz (K={K_AXES}, "
          f"embedding_version {ev})", flush=True)

    # RETIRE ANY per_cell/ FROM THE PREVIOUS AXIS MAP. package_for_pwa reads the
    # correct-move vectors from per_problem.csv and the picked-move vectors from
    # per_cell/, and has no way to tell they came from different fits — it would
    # ship a shard whose two embedding sources disagree, which no client check
    # can catch because both files are individually well-formed. Moving it aside
    # makes packaging fail loudly instead.
    pc = os.path.join(AXES_DIR, "per_cell")
    if os.path.isdir(pc):
        stale = os.path.join(AXES_DIR, f"per_cell.stale-{int(time.time())}")
        os.rename(pc, stale)
        print(f"  RETIRED the previous per_cell/ -> {os.path.basename(stale)}. "
              f"It was projected through the axis map this fit replaces, so "
              f"shipping it beside these correct-move vectors would mix two "
              f"fits. Packaging needs a per-cell pass against THIS model before "
              f"it can run; delete the stale directory once that exists.",
              flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
