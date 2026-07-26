"""Cross-board feature→4PL rating model — the PRODUCTION rating trainer.

This is the production-only entry point: it trains the shipped model and writes the rating
artifacts, and does NOTHING else. It takes NO options (CLI or internal) — the shipped configuration
is hardcoded as module constants below — and carries no eval, held-out, DRO, anchor, hierarchical,
per-problem, or other experiment machinery. The full experiment + evaluation surface (every knob,
the held-out / survey-ρ eval suite, alternative objectives) lives in
auxiliary/rating_dev/train_rating_staged.py, which can replicate this output. Do experiments
THERE, so a test tweak can never break the shipped ratings.

ONE model for 9x9 AND 13x13. A SHARED feature→4PL network maps each problem's features to
(difficulty b_p, slope a_p, floor c_p); per-(user, 2-month) ability θ_u is a free point estimate;
a single GLOBAL ceiling (fat-finger / lapse) λ is learned.

    P(correct) = c_p + (1 − c_p − λ)·σ( a_p·(θ_u − b_p) )
    (b_p, a_p, c_p) = net(feat_p)        # shared net; a_p ≥ SLOPE_FLOOR, c_p ∈ [0, 1 − 2λ]
    feat_p = joint-PCA(correct-move activations) ++ common-column bot winrate ++ candidate-move evals
             ++ game-phase raw components
             (bw_act + move + phase; the move block needs pipeline.cache_move_features run first)

There are NO per-problem residuals: the shared net IS the model, so every problem — including the
never-attempted and the one-sided — gets its 4PL straight from features. The only training target is
the HUMAN first-time attempts (bot data enters only through the features). The feature trunk uses 0.3
COLUMN dropout on each of the activation, bot-winrate and candidate-move-eval blocks (every feature
column dropped independently), and 0.3 dropout between the trunk's own hidden layers. The activation
block used BLOCK dropout (all columns on/off together) until 2026-07-26; see the ACT_DROPOUT note.

The fit is a SINGLE JOINT pass (replaced the two-phase staged coordinate descent, 2026-07-24): all
three per-problem heads and the global ceiling train together, made well-posed by three measured
ingredients (validated in the gauge/penalty experiment series — see the CLAUDE.md rating section):

  1. RATING GAUGE — the scale freedom (rating×s, slope/s, θ×s) is pinned on the PROBLEM side: the
     difficulty head b is hard-whitened to mean 0 / std 1 (BatchNorm-style: train batches use their
     own attempt-weighted stats and update an EMA; eval uses the EMA, so scored ratings live in the
     gauge θ was trained against). θ is left free (its spread settles ≈1.2).
  2. SHALLOW-SLOPE PENALTY — w_a·mean(1/a_raw²) over all problems, on the PRE-clamp slope so the
     gradient survives at the SLOPE_FLOOR rail. 1/a is the curve's transition width; with the corpus
     difficulty spread pinned to 1 this one-sided, threshold-free term is exactly the factor the
     80%-clear rating diverges through on no-win/unattempted problems (the flat-curve collapse).
     Deliberately floor/ceiling-free: a penalty on the DERIVED rating is gameable through the floor
     head (measured: floors inflate to cancel b and held-out calibration collapses).
  3. HIGH-FLOOR PENALTY — the easy-side mirror: the rating diverges to −∞ as the floor approaches
     CALIBRATION_P, so w_c·mean(softplus(o_c − o_div)²) in the floor head's LOGIT space, centered at
     the raw output where c = 0.8 (derived from the serve margin, not hand-set; λ detached). Smooth
     and one-sided, with live gradient even past 0.8 (a q-space penalty clamp-dies exactly there).

Both weights were selected by held-out logloss sweeps (the tail comes in at ~zero cost; an order of
magnitude higher starts homogenizing real floor/slope structure — don't raise them casually).

PHASE 2 — after the joint pass, the SAME response model runs a second time with the trunk frozen
(fit_global_scales): every per-problem (rating, slope, floor) is an eval-mode constant, the phase-2
ability field is refit free (per user × non-overlapping 2-month block, profile likelihood — REFIT,
not plugged in: fixed-theta plug-in is attenuation-biased, certificate-measured 2026-07-25), and two
global scalars train that phase 1 held at unity: s (slope scale) and f (floor scale). Pure BCE — no
dropout (nothing stochastic remains), no barriers (they act on the frozen heads). Scored in the
CLIENT's response form pivoting at the shipped rating (mode-'slope' semantics: ratings
stay the 80%-clear anchors; quality ships as s·slope, floor as f·floor). This replaced the held-out
walk-forward calibration (2026-07-25): the walk-forward's causal protocol culled ~38% of attempt
mass (heavy users only), while phase 2 scores every first-time attempt; the two criteria agree
within the plateau (<0.1 mnat), the free-ability profile likelihood is certificate-clean (synthetic
s=1 truth returns 1.01–1.02 at this granularity), and the multiplier is dropout-induced rating-scale
compression, NOT errors-in-variables — an in-sample refit does NOT re-absorb it to 1 (measured:
frozen-theta 1.62, free-theta 1.71–1.77 vs walk-forward 1.667). utils/slope_calibration.py keeps
the walk-forward as the client-condition validation harness.

The shipped RATING is the ability at which the user clears the problem 80% of the time
(CALIBRATION_P) — NOT the 50%-point — so a flat low-discrimination problem correctly needs more
ability to clear at 80% than its midpoint implies:

    rating_p = b_p + logit((0.8 − c_p) / (1 − c_p − λ)) / a_p

The global ceiling λ is produced HERE (written to ratings/rating_constants.json);
pipeline.fit_population_constants reads it through.

Outputs (nothing in pwa/tsumegos/ is touched — package_for_pwa ships these):
  ratings/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= 80%-clear rating)
  ratings/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= s·a_p, phase-2-calibrated)
  ratings/<size>/tsumego_floor_final.csv     tsumego_id, floor    (= f·c_p, phase-2-calibrated)
  ratings/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings/rating_constants.json              {"fat_finger_rate": λ, "slope_scale": s, "floor_scale": f, ...}
  ratings/model.pt                        weights + run metadata

Usage (run pipeline.cache_move_features first — it builds the candidate-move-eval cache the
feature trunk reads):
  ./.venv/bin/python3 -m pipeline.cache_move_features
  ./.venv/bin/python3 -m pipeline.train_rating
"""

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

from utils.rating_features import (  # noqa: E402
    _cache_ids_path, build_profiles, load_first_time_attempts,
    load_joint_activations, load_joint_bot_winrate, load_joint_game_phase,
    load_joint_move_features,
)
from utils.slope_calibration import load_client_lapse  # noqa: E402
from utils.fit_rank_curves import CALIBRATION_P  # noqa: E402
from utils.configurations import ALL_BOARD_SIZES  # noqa: E402

EPS = 1e-6
SIZES = list(ALL_BOARD_SIZES)

# ── PRODUCTION configuration (the shipped single-pass rating-gauge fit) ───────────────────────────
# To re-derive or experiment with any of these, use auxiliary/rating_dev/train_rating_staged.py
# (--production-flags reproduces this config).
OUT_DIR = os.path.join(PROJECT_ROOT, "ratings")
SEED = 0
PCA_DIM = 128                    # joint-PCA activation feature dimensionality
DIFF_HIDDEN = (128, 64)          # shared feature→4PL trunk hidden layers
DROPOUT = 0.3                    # trunk dropout (0.3 uniformly — the backbone-overfit fix)
# COLUMN dropout on the activation block (train only) — each PC dropped independently, the same
# convention as the bot-winrate and move blocks. Replaced BLOCK dropout (one keep bit per row for all
# PCA_DIM columns at once) on 2026-07-26 after a paired 3-seed measurement: column granularity is
# worth ~0.006-0.012 nats of held-out-PROBLEM logloss (every seed, every cluster-bootstrap CI
# excluding zero) AND shrinks the phase-2 correction (gain 8.0 -> 5.5 mnat, s 1.63 -> 1.52 mean).
# Why the finer grain wins, when the block variant was introduced precisely to break per-problem
# fingerprint memorization: block dropout leaves the fingerprint FULLY INTACT on 1-p of rows, so the
# net can learn a present/absent regime pair and memorize freely in the majority one. Column dropout
# never presents the same PCA_DIM-wide code twice, so the exact CONJUNCTION memorization needs is
# unreliable on every row, while the smooth low-order signal survives everywhere. The dose response
# is a broad plateau (0.3 and 0.5 tie, 0.7 slightly worse) and only degrades past ~0.9, where so few
# PCs survive that it reverts to the block regime — which is also the mechanism's own prediction.
ACT_DROPOUT = 0.3
BW_DROPOUT = 0.3                 # column dropout on the bot-winrate columns (train only)
MOVE_DROPOUT = 0.3               # column dropout on the candidate-move-eval columns (train only)
LR = 0.01
BATCH = 65536
EPOCHS = 200                     # single joint pass (matches the old 2×100 staged budget)
MIN_ATTEMPTS = 5                 # keep users with ≥5 correct AND ≥5 incorrect first-time attempts
MIN_PROFILE = 8                  # keep 2-month profiles with ≥8 attempts
SLOPE_FLOOR = 0.01               # lower clamp on the per-problem slope a_p (numerical guard)
LAPSE_CAP = 0.05                 # upper bound on the global ceiling slip λ
LAPSE_INIT = 0.02                # λ init (λ is learned)
FLOOR_CAP = 1.0                  # hard upper bound on the guessing floor c_p
GAUGE_MOM = 0.01                 # EMA momentum of the difficulty-whitening (rating gauge) buffers
SHALLOW_SLOPE_PENALTY = 1e-4     # w_a on mean(1/a_raw²) — the hard-tail (flat-curve) barrier
HIGH_FLOOR_PENALTY = 0.01        # w_c on mean(softplus(o_c − o_div)²) — the easy-tail barrier
P2_LR = 0.05                     # phase-2 (global scale fit) full-batch Adam learning rate
P2_STEPS = 2500                  # phase-2 full-batch steps (deterministic: no dropout, no minibatch)
P2_BIN_MONTHS = 2                # phase-2 ability unit: user × non-overlapping 2-month block
CLIENT_FLOOR_MAX = 0.78          # pwa/js/irt.js floor clamp — phase 2 scores the client's curve


def clear_logit_vec(c, lapse=0.0):
    """Per-problem 80%-clear logit anchor under the 4PL: at the rating P(clear)=CALIBRATION_P, so
    σ(a(θ−b))=(P*−c)/(1−c−λ) and rating = b + logit((P*−c)/(1−c−λ))/a. Vector over problems.
    `lapse` is the global 4PL slip rate; it must match the client's clearKappa so the shipped rating
    is the true 80%-clear point."""
    c = np.asarray(c, dtype=np.float64)
    q = np.clip((CALIBRATION_P - c) / (1.0 - c - lapse), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def fit_global_scales(df, params_by_tid, lapse, device):
    """PHASE 2 of the two-phase production fit: the global slope scale s and floor scale f (both
    held at unity during phase 1) by in-sample MLE, everything else frozen except the ability field.

    The same response model run a second time: per-problem (rating, slope, floor) are phase 1's
    eval-mode outputs, now constants; the phase-2 ability field is REFIT free — one theta per
    user × non-overlapping P2_BIN_MONTHS calendar block, i.e. trainer-granularity profile
    likelihood. Refit is load-bearing, not a luxury: plugging in FIXED abilities attenuates the
    fitted scale (error in the regressor), while the refit-at-every-s profile likelihood is
    certificate-clean (synthetic data generated at true s = 1 returns 1.01–1.02 at this
    granularity; 2026-07-25 series). Loss is pure BCE — no dropout (nothing stochastic remains),
    no barriers (they act on the frozen heads) — over EVERY first-time attempt on a scorable
    problem (no user cull; the walk-forward this replaced scored only heavy users' post-warmup
    rows, ~62% of attempt mass).

    Scored in the CLIENT's response form (pwa/js/irt.js), pivoting at the SHIPPED
    rating — mode-'slope' semantics: the rating stays the 80%-clear anchor of the unscaled curve,
    and the scales ship as quality = s·slope, floor = f·floor. (Scaling around the trainer's
    internal 50%-point instead would rescale the rating axis itself.) kappa is recomputed from the
    scaled floor inside the loss, exactly as the client will compute it from the shipped floor.

    Runs twice — scales frozen at unity (theta-only baseline), then scales free, warm-started from
    the baseline thetas — so the reported gain is like-for-like. Fully deterministic: zero/mean
    inits, full-batch Adam, no RNG. Returns {'s', 'f', 'gain_mnat_per_attempt', 'n_rows',
    'n_units', 'bce_unity', 'bce_scaled'}."""
    df = df[df["problem"].isin(params_by_tid)].copy()
    ts = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()]
    mon = ts.dt.year.to_numpy() * 12 + (ts.dt.month.to_numpy() - 1)
    unit = pd.Categorical(df["user_id"].astype(str) + "|" + (mon // P2_BIN_MONTHS).astype(str))
    codes = unit.codes.astype(np.int64)
    n_units = len(unit.categories)
    pr = [params_by_tid[t] for t in df["problem"]]
    R0 = np.array([p[0] for p in pr], dtype=np.float64)
    A = np.array([p[1] for p in pr], dtype=np.float64)
    C = np.array([p[2] for p in pr], dtype=np.float64)
    y = df["success"].to_numpy(dtype=np.float64)
    n = len(y)
    # warm start: each unit's theta at its attempt-mean rating (serving is difficulty-matched)
    th0 = np.bincount(codes, weights=R0, minlength=n_units) / \
        np.maximum(np.bincount(codes, minlength=n_units), 1)

    tR0 = torch.tensor(R0, device=device)
    tA = torch.tensor(A, device=device)
    tC = torch.tensor(C, device=device)
    ty = torch.tensor(y, device=device)
    tcodes = torch.tensor(codes, device=device)

    def run(free_scales, th_init):
        th = nn.Parameter(torch.tensor(th_init, device=device, dtype=torch.float64))
        ls = nn.Parameter(torch.zeros((), device=device, dtype=torch.float64))
        lf = nn.Parameter(torch.zeros((), device=device, dtype=torch.float64))
        opt = torch.optim.Adam([th, ls, lf] if free_scales else [th], lr=P2_LR)
        for step in range(P2_STEPS):
            s = torch.exp(ls) if free_scales else 1.0
            f = torch.exp(lf) if free_scales else 1.0
            c2 = (f * tC).clamp(0.0, CLIENT_FLOOR_MAX)
            span = (1.0 - c2 - lapse).clamp(min=EPS)
            q = ((CALIBRATION_P - c2) / span).clamp(EPS, 1.0 - EPS)
            kap = torch.log(q / (1.0 - q))
            z = (s * tA * (th[tcodes] - tR0) + kap).clamp(-35.0, 35.0)
            P = (c2 + span * torch.sigmoid(z)).clamp(EPS, 1.0 - EPS)
            loss = -(ty * torch.log(P) + (1.0 - ty) * torch.log(1.0 - P)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if free_scales and (step % 500 == 0 or step == P2_STEPS - 1):
                print(f"    phase-2 step {step:4d}  BCE {float(loss):.5f}  "
                      f"s {float(torch.exp(ls)):.3f}  f {float(torch.exp(lf)):.3f}", flush=True)
        return (float(loss), float(torch.exp(ls)), float(torch.exp(lf)),
                th.detach().cpu().numpy())

    bce_unity, _, _, th_base = run(False, th0)
    bce_scaled, s, f, _ = run(True, th_base)
    return {"s": round(s, 4), "f": round(f, 4),
            "gain_mnat_per_attempt": (bce_unity - bce_scaled) * 1000,
            "n_rows": n, "n_units": n_units,
            "bce_unity": bce_unity, "bce_scaled": bce_scaled}


class RatingNet(nn.Module):
    """Shared feature→4PL net + free per-user ability + one learned global ceiling λ, with the
    scale gauge pinned on the DIFFICULTY head (rating gauge).

        P = c + (1 − c − λ)·σ( a·(θ_u − b) ),   (b, a, c) = net(feat)

    b is hard-whitened to mean 0 / std 1 — batch stats while training (differentiable, EMA
    updated), the EMA at eval — so scored ratings live in the same gauge θ trained against.
    θ is a free embedding (no whitening; its spread is identified by the b pin).
    a = softplus(·).clamp(min=SLOPE_FLOOR); c is ceiling-coupled to [0, 1 − 2λ]."""

    def __init__(self, fdim, n_users, act_dim, bw_dim, move_dim):
        super().__init__()
        self.act_dim = int(act_dim)         # activation block = cols [:act_dim]  (column-dropped)
        self.bw_dim = int(bw_dim)           # bot-winrate block = cols [act_dim:+bw_dim] (col-drop)
        self.move_dim = int(move_dim)       # move-eval block = cols [act_dim+bw_dim:+move_dim]
        self.register_buffer("b_mu", torch.zeros(()))   # rating-gauge EMA (train stats → eval)
        self.register_buffer("b_sd", torch.ones(()))
        # (theta is constructed BEFORE the trunk so the RNG draw order is fixed.)
        self.theta = nn.Embedding(n_users, 1)
        nn.init.normal_(self.theta.weight, std=1.0)
        layers, d = [], fdim
        for h in DIFF_HIDDEN:                                # decreasing layers, ReLU + dropout
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        layers += [nn.Linear(d, 3)]                          # → (difficulty b, slope a, floor c)
        self.net = nn.Sequential(*layers)
        li = float(np.clip(LAPSE_INIT / LAPSE_CAP, EPS, 1.0 - EPS))
        self.lapse_raw = nn.Parameter(torch.tensor(float(np.log(li / (1.0 - li)))))

    def lapse(self):
        """Global 4PL upper-asymptote slip rate λ ∈ (0, LAPSE_CAP)."""
        return LAPSE_CAP * torch.sigmoid(self.lapse_raw)

    def _dropped(self, feat):
        """Train-time column dropout on the feature blocks (eval passes through)."""
        if self.training and self.act_dim > 0:               # column-drop the activation block
            keep = (torch.rand(len(feat), self.act_dim, device=feat.device)
                    >= ACT_DROPOUT).float() / (1.0 - ACT_DROPOUT)
            feat = torch.cat([feat[:, :self.act_dim] * keep, feat[:, self.act_dim:]], dim=1)
        if self.training and self.bw_dim > 0:                # column-drop the bot-winrate block
            d0 = self.act_dim
            keep = (torch.rand(len(feat), self.bw_dim, device=feat.device)
                    >= BW_DROPOUT).float() / (1.0 - BW_DROPOUT)
            feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.bw_dim] * keep,
                              feat[:, d0 + self.bw_dim:]], dim=1)
        if self.training and self.move_dim > 0:              # column-drop the move-eval block
            d0 = self.act_dim + self.bw_dim
            keep = (torch.rand(len(feat), self.move_dim, device=feat.device)
                    >= MOVE_DROPOUT).float() / (1.0 - MOVE_DROPOUT)
            feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.move_dim] * keep,
                              feat[:, d0 + self.move_dim:]], dim=1)
        return feat

    def params(self, feat):
        """(b, a, c) for the rows of `feat`; b in the whitened rating gauge."""
        o = self.net(self._dropped(feat))
        b = o[:, 0]
        if self.training:
            mu, sd = b.mean(), b.std() + EPS
            with torch.no_grad():
                self.b_mu.mul_(1.0 - GAUGE_MOM).add_(GAUGE_MOM * mu)
                self.b_sd.mul_(1.0 - GAUGE_MOM).add_(GAUGE_MOM * sd)
        else:
            mu, sd = self.b_mu, self.b_sd
        b = (b - mu) / sd
        a = torch.nn.functional.softplus(o[:, 1]).clamp(min=SLOPE_FLOOR)  # lower bound only
        lam = self.lapse()
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=FLOOR_CAP)
        c = cap * torch.sigmoid(o[:, 2])
        return b, a, c

    def slope_raw(self, feat):
        """PRE-clamp per-problem slope — the shallow-slope penalty acts on this (the clamp zeroes
        gradients exactly where collapsed slopes need lifting)."""
        return torch.nn.functional.softplus(self.net(feat)[:, 1]).clamp_min(1e-4)

    def floor_logit(self, feat):
        """The floor head's raw logit — the high-floor barrier acts on this (a q-space penalty
        loses its gradient exactly where floors pass CALIBRATION_P)."""
        return self.net(feat)[:, 2]

    def ability(self, u):
        """Per-user ability (free point estimate; the scale pin lives on the difficulty head)."""
        return self.theta(u).squeeze(-1)

    def forward(self, u, feat):
        b, a, c = self.params(feat)
        th = self.ability(u)
        lam = self.lapse()
        return (c + (1.0 - c - lam) * torch.sigmoid(a * (th - b))).clamp(EPS, 1 - EPS)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- features: joint-PCA correct-move activations ⊕ common-column bot winrate ⊕ size_is_13 ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, k = load_joint_activations(SIZES, tids_by_size, PCA_DIM, SEED)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]                  # (size, tid); scorable iff act∩bw
    key_idx = {key: i for i, key in enumerate(keys)}
    # candidate-move-eval block (pipeline.cache_move_features); appended AFTER bot, BEFORE the size
    # indicator so the [act | bw | move | size] layout keeps the per-block dropouts' column ranges
    # valid. Doesn't gate the scorable set (act∩bw); a problem missing a move row is mean-imputed
    # (zero after the pooled z-score).
    mv, nmv = load_joint_move_features(SIZES, tids_by_size)
    zero_mv = np.zeros(nmv, dtype=np.float32)
    # game-phase raw components [moves_before_position, main_line_len] (shared source with the
    # secondary-axis work — utils.rating_features.load_joint_game_phase). Included by FEATURE-PARITY
    # POLICY (see the old staged trainer's notes); no dedicated dropout (2 smooth columns cannot
    # fingerprint problems).
    ph, nph = load_joint_game_phase(SIZES, tids_by_size)
    zero_ph = np.zeros(nph, dtype=np.float32)
    fdim = k + nb + nmv + nph + 1                             # act ⊕ bw ⊕ move ⊕ phase ⊕ size_is_13
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        feat[i] = np.concatenate([act[key], bw[key], mv.get(key, zero_mv), ph.get(key, zero_ph),
                                  [1.0 if key[0] == 13 else 0.0]])
    n_mv_miss = sum(1 for key in keys if key not in mv)
    n_ph_miss = sum(1 for key in keys if key not in ph)
    print(f"scorable problems {len(keys)} (9:{sum(s == 9 for s, _ in keys)} "
          f"13:{sum(s == 13 for s, _ in keys)}) | feat dim {fdim} "
          f"(act {k} + bot {nb} + move {nmv} ({n_mv_miss} imputed) + phase {nph} "
          f"({n_ph_miss} imputed) + size_is_13)")

    # --- HUMAN first-time attempts, pooled, shared 2-month profiles (no size in profile id) ---
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    df = df[df["user_id"].isin(
        set(g.index[(g["c"] >= MIN_ATTEMPTS) & (g["w"] >= MIN_ATTEMPTS)]))].copy()
    prob2size = dict(zip(df["problem"], df["size"]))
    prof_df = build_profiles(df)                              # profile, problem, success
    pc = prof_df["profile"].value_counts()
    prof_df = prof_df[prof_df["profile"].isin(set(pc.index[pc >= MIN_PROFILE]))]
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

    tfeat = torch.tensor(feat, device=device)
    tu = torch.tensor(prof_idx, device=device)
    tp = torch.tensor(prob_idx, device=device)
    ty = torch.tensor(y, device=device)
    n = len(prof_idx)

    # ---- single joint pass: BCE + the two divergence barriers (see the module docstring) ----
    torch.manual_seed(SEED)
    m = RatingNet(fdim, n_profiles, act_dim=k, bw_dim=nb, move_dim=nmv).to(device)
    print(f"\n=== JOINT rating-gauge fit | {EPOCHS} epochs | dropout {DROPOUT} uniform | "
          f"shallow-slope w {SHALLOW_SLOPE_PENALTY:g} | high-floor w {HIGH_FLOOR_PENALTY:g} | "
          f"ceiling LEARNED ===")
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for ep in range(EPOCHS):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            pred = m(tu[bi], tfeat[tp[bi]])
            # explicit none→sum/N, NOT reduction="mean": the fused mean kernel differs from this
            # reduction at non-power-of-two batch sizes (the last, smaller minibatch), and over the
            # epochs that ULP difference compounds in the model's flat slope/floor directions.
            loss = torch.nn.functional.binary_cross_entropy(
                pred, ty[bi], reduction="none").sum() / pred.shape[0]
            # divergence barriers, over the FULL corpus with dropout off (clean per-problem read)
            m.eval()
            loss = loss + SHALLOW_SLOPE_PENALTY * m.slope_raw(tfeat).pow(-2).mean()
            lam_d = m.lapse().detach()                       # detached: the barrier must not train λ
            cap_d = (1.0 - 2.0 * lam_d).clamp(min=EPS, max=FLOOR_CAP)
            ratio = (CALIBRATION_P / cap_d).clamp(EPS, 1.0 - EPS)
            o_div = torch.log(ratio / (1.0 - ratio))
            loss = loss + HIGH_FLOOR_PENALTY * \
                torch.nn.functional.softplus(m.floor_logit(tfeat) - o_div).pow(2).mean()
            m.train()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == EPOCHS - 1:
            m.eval()
            with torch.no_grad():
                hb = float(torch.nn.functional.binary_cross_entropy(m(tu, tfeat[tp]), ty))
            print(f"  epoch {ep:4d}  human train BCE {hb:.4f}  λ {float(m.lapse()):.5f}")

    # ---- score the whole corpus (final model) ----
    m.eval()
    with torch.no_grad():
        b_all, a_all, c_all = m.params(tfeat)
        theta = m.ability(torch.arange(n_profiles, device=device)).cpu().numpy()
        lam = float(m.lapse())
    b_all, a_all, c_all = b_all.cpu().numpy(), a_all.cpu().numpy(), c_all.cpu().numpy()
    rating = b_all + clear_logit_vec(c_all, lam) / a_all     # 80%-clear ability under the 4PL
    with np.printoptions(precision=2, suppress=False):
        # tail-sanity visibility: the divergence barriers keep these finite (~[-4, +13]); a blowup
        # here means a barrier regression, not something to clip after the fact (the percentile
        # winsor that used to sit here was removed 2026-07-25 — redundant under the barriers, and
        # it erased the ordering of the top/bottom 1% of the corpus)
        print(f"  rating percentiles [.1, 1, 50, 99, 99.9]: "
              f"{np.percentile(rating, [0.1, 1, 50, 99, 99.9])}")

    # ---- PHASE 2: global slope/floor scale fit (replaced the walk-forward stage 2026-07-25) ----
    # The per-problem values come out scale-compressed toward the feature-conditional mean
    # (dropout-induced — NOT errors-in-variables; bootstrap-proven, 2026-07-24 series), so the true
    # response curves are steeper per shipped-rating unit than the phase-1 slope claims. Phase 2
    # measures the correction as two global scalars on the frozen model, in-sample over EVERY
    # first-time attempt with the ability field refit (see fit_global_scales — refit is what keeps
    # the criterion unbiased). s is recalculated every run, never hardcoded; ordering matters:
    # rating is derived from the UNSCALED slope/floor above (it anchors the curve phase 2 scales).
    cal = fit_global_scales(
        load_first_time_attempts(),
        {keys[i][1]: (float(rating[i]), float(a_all[i]), float(c_all[i]))
         for i in range(len(keys))},
        load_client_lapse(lam), device)
    print(f"  phase-2 scales: s = {cal['s']:.3f}, f = {cal['f']:.3f} "
          f"(+{cal['gain_mnat_per_attempt']:.2f} mnat/attempt in-sample, "
          f"{cal['n_units']} ability units / {cal['n_rows']} attempts)")
    a_all = cal["s"] * a_all
    c_all = cal["f"] * c_all
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(OUT_DIR, str(s)); os.makedirs(sdir, exist_ok=True)
        pd.DataFrame({"tsumego_id": tids, "rating": rating[sel]}).sort_values(
            "rating", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_ratings_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "quality": a_all[sel]}).sort_values(
            "quality", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_quality_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "floor": c_all[sel]}).sort_values(
            "floor", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_floor_final.csv"), index=False)
        print(f"  size {s}: wrote {len(tids)} ratings | rating "
              f"[{rating[sel].min():.2f},{rating[sel].max():.2f}] slope "
              f"[{a_all[sel].min():.2f},{a_all[sel].max():.2f}] | floor mean {c_all[sel].mean():.3f}")
    natt = pd.Series(prof_idx).value_counts().sort_index()
    pd.DataFrame({"profile": list(profiles.categories), "theta": theta,
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_profiles)]}).to_csv(
        os.path.join(OUT_DIR, "user_theta.csv"), index=False)
    with open(os.path.join(OUT_DIR, "rating_constants.json"), "w") as f:
        json.dump({"fat_finger_rate": round(lam, 4),
                   "slope_scale": cal["s"], "floor_scale": cal["f"],
                   "scale_fit": "two_phase_insample",
                   "scale_gain_mnat": round(cal["gain_mnat_per_attempt"], 3)},
                  f, indent=2)
    config = {"features": "bw_act", "move_features": True, "phase_features": True,
              "pca_dim": PCA_DIM, "diff_hidden": list(DIFF_HIDDEN), "dropout": DROPOUT,
              "act_dropout": ACT_DROPOUT, "act_dropout_mode": "column",
              "bw_dropout": BW_DROPOUT, "move_dropout": MOVE_DROPOUT,
              "lr": LR, "batch": BATCH, "epochs": EPOCHS,
              "gauge": "rating", "gauge_mom": GAUGE_MOM,
              "shallow_slope_penalty": SHALLOW_SLOPE_PENALTY,
              "high_floor_penalty": HIGH_FLOOR_PENALTY,
              "p2_lr": P2_LR, "p2_steps": P2_STEPS, "p2_bin_months": P2_BIN_MONTHS,
              "min": MIN_ATTEMPTS, "min_profile": MIN_PROFILE,
              "slope_floor": SLOPE_FLOOR, "lapse_cap": LAPSE_CAP,
              "lapse_init": LAPSE_INIT, "floor_cap": FLOOR_CAP, "seed": SEED}
    torch.save({"state_dict": m.state_dict(), "keys": keys, "fdim": fdim, "k": k,
                "n_bot_cols": nb, "n_move_cols": nmv, "n_anchor": 0, "anchor_ranks": [],
                "lapse": lam, "args": config}, os.path.join(OUT_DIR, "model.pt"))
    print(f"\n=== JOINT complete === wrote ratings + user_theta.csv ({n_profiles} profiles) + "
          f"rating_constants.json (λ={lam:.5f}) + model.pt to {OUT_DIR}")


if __name__ == "__main__":
    main()
