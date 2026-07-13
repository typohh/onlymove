"""Cross-board feature→4PL rating model — the PRODUCTION rating trainer.

This is the production-only entry point: it trains the shipped model and writes the rating
artifacts, and does NOTHING else. It takes NO options (CLI or internal) — the shipped configuration
is hardcoded as module constants below — and carries no eval, held-out, DRO, anchor, hierarchical,
per-problem, or other experiment machinery. The full experiment + evaluation surface (every knob,
the held-out / survey-ρ eval suite, alternative objectives) lives in
auxiliary/rating_dev/train_rating_staged.py, which can replicate this output. Do experiments
THERE, so a test tweak can never break the shipped ratings.

ONE model for 9x9 AND 13x13. A SHARED feature→4PL network maps each problem's features to
(difficulty b_p, slope a_p, floor c_p); per-(user, 2-month) ability θ_u is hard-gauged to a unit
Gaussian; a single GLOBAL ceiling (fat-finger / lapse) λ is learned.

    P(correct) = c_p + (1 − c_p − λ)·σ( a_p·(θ_u − b_p) )
    (b_p, a_p, c_p) = net(feat_p)        # shared net; a_p ≥ SLOPE_FLOOR, c_p ∈ [0, 1 − 2λ]
    feat_p = joint-PCA(correct-move activations) ++ common-column bot winrate ++ candidate-move evals
             (bw_act + move; the move block needs pipeline.cache_move_features run first)

There are NO per-problem residuals: the shared net IS the model, so every problem — including the
never-attempted and the one-sided — gets its 4PL straight from features. The only training target is
the HUMAN first-time attempts (bot data enters only through the features). The feature trunk uses 0.3
block dropout on the activation columns, 0.3 column dropout on each of the bot-winrate and
candidate-move-eval columns, and 0.3 dropout between the trunk's own hidden layers (the
backbone-overfit fix — 0.3 dropout uniformly, blocks and trunk alike).

The fit is a TWO-PHASE coordinate descent on the shared trunk:
  phase 1 — fit the per-problem difficulty (rating) and simultaneously LEARN a single global slope +
            global floor + global ceiling (auto-finding the self-consistent globals, no hand
            fixed-point search, so it stays calibrated as the corpus drifts);
  phase 2 — individualize the per-problem slope + floor and keep training the ceiling, while the
            difficulty is held at its phase-1 values by an L2 consistency penalty.

Ability is hard-gauged (θ whitened to mean 0 / std 1 each step): the population follows N(0, 1), and
pinning the scale stops slopes inflating to compensate for compressed ability. The shipped RATING is
the ability at which the user clears the problem 80% of the time (CALIBRATION_P) — NOT the 50%-point
— so a flat low-discrimination problem correctly needs more ability to clear at 80% than its midpoint
implies:

    rating_p = b_p + logit((0.8 − c_p) / (1 − c_p − λ)) / a_p

The global ceiling λ is produced HERE (written to ratings/rating_constants.json);
pipeline.fit_population_constants reads it through.

Outputs (nothing in pwa/tsumegos/ is touched — package_for_pwa ships these):
  ratings/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= 80%-clear rating)
  ratings/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= raw slope a_p)
  ratings/<size>/tsumego_floor_final.csv     tsumego_id, floor    (= learned c_p)
  ratings/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings/rating_constants.json              {"fat_finger_rate": λ}  (the learned ceiling)
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
from utils.fit_rank_curves import CALIBRATION_P  # noqa: E402
from utils.configurations import ALL_BOARD_SIZES  # noqa: E402

EPS = 1e-6
SIZES = list(ALL_BOARD_SIZES)

# ── PRODUCTION configuration (the shipped staged self-consistent fixed point) ─────────────────────
# To re-derive or experiment with any of these, use auxiliary/rating_dev/train_rating_staged.py.
OUT_DIR = os.path.join(PROJECT_ROOT, "ratings")
SEED = 0
PCA_DIM = 128                    # joint-PCA activation feature dimensionality
DIFF_HIDDEN = (128, 64)          # shared feature→4PL trunk hidden layers
DROPOUT = 0.3                    # trunk dropout (swept 0.0–0.5: 0.3 peaks survey-ρ + held-out flatness and sits at the held-out-LL floor; 0.1 was undertuned, costing ~0.011 held-out LL)
ACT_DROPOUT = 0.3                # block dropout on the activation feature block (train only)
BW_DROPOUT = 0.3                 # column dropout on the bot-winrate columns (train only)
MOVE_DROPOUT = 0.3               # column dropout on the candidate-move-eval columns (train only)
LR = 0.01
BATCH = 65536
STAGE_EPOCHS = 100               # epochs per coordinate-descent phase
STAGE_PIN_WEIGHT = 10.0          # L2 penalty holding a frozen head at its stored per-problem values
MIN_ATTEMPTS = 5                 # keep users with ≥5 correct AND ≥5 incorrect first-time attempts
MIN_PROFILE = 8                  # keep 2-month profiles with ≥8 attempts
WINSOR = 1.0                     # percentile winsor on the shipped rating (robustness)
SLOPE_FLOOR = 0.01               # lower clamp on the per-problem slope a_p
LAPSE_CAP = 0.05                 # upper bound on the global ceiling slip λ
LAPSE_INIT = 0.02                # λ init (λ is learned every phase)
FLOOR_CAP = 1.0                  # hard upper bound on the guessing floor c_p
GLOBAL_SLOPE_INIT = 1.0          # phase-1 learned-global slope init
GLOBAL_FLOOR_INIT = 0.25         # phase-1 learned-global floor init
# Two-phase plan: (free-head set, tag). Heads: 0=rating/b, 1=slope/a, 2=floor/c. The global ceiling
# λ is trained in BOTH phases.
STAGE_PLAN = [({0}, "rating+ceiling"), ({1, 2}, "slope+floor+ceiling")]
HEAD_NAME = {0: "rating", 1: "slope", 2: "floor"}


def clear_logit_vec(c, lapse=0.0):
    """Per-problem 80%-clear logit anchor under the 4PL: at the rating P(clear)=CALIBRATION_P, so
    σ(a(θ−b))=(P*−c)/(1−c−λ) and rating = b + logit((P*−c)/(1−c−λ))/a. Vector over problems.
    `lapse` is the global 4PL slip rate; it must match the client's clearKappa so the shipped rating
    is the true 80%-clear point."""
    c = np.asarray(c, dtype=np.float64)
    q = np.clip((CALIBRATION_P - c) / (1.0 - c - lapse), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


class RatingNet(nn.Module):
    """Shared feature→4PL net + per-user ability (hard unit gauge) + one learned global ceiling λ.

        P = c + (1 − c − λ)·σ( a·(θ_u − b) ),   (b, a, c) = net(feat)

    The shared trunk has NO per-problem parameters, so every problem's 4PL comes from its features.
    The fit is staged: phase 1 fits b while a and c are held at a single LEARNED global slope /
    global floor (gslope_raw / gfloor_raw, trained jointly with b and the ceiling), phase 2
    individualizes a and c (b held at its phase-1 value by a consistency penalty in the loss).
    a = softplus(·).clamp(min=SLOPE_FLOOR); c is ceiling-coupled to [0, 1 − 2λ]."""

    def __init__(self, fdim, n_users, n_problems, act_dim, bw_dim, move_dim):
        super().__init__()
        self.act_dim = int(act_dim)         # activation block = cols [:act_dim]  (block-dropped, train)
        self.bw_dim = int(bw_dim)           # bot-winrate block = cols [act_dim:act_dim+bw_dim] (col-drop)
        self.move_dim = int(move_dim)       # move-eval block = cols [act_dim+bw_dim:+move_dim] (col-drop)
        # head indices 0=rating/b, 1=slope/a, 2=floor/c. A phase frees a SET of heads (live data BCE);
        # already-trained heads are frozen at their stored per-problem pin buffers by the loss penalty;
        # not-yet-trained heads sit at the learned global scalar (gslope/gfloor) until their phase.
        self.trained_heads = set()
        self.free_heads = set()
        # phase-1 learned globals (trained jointly with rating + the ceiling)
        self.gslope_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(GLOBAL_SLOPE_INIT)))))
        self.gfloor_raw = nn.Parameter(
            torch.tensor(float(np.log(GLOBAL_FLOOR_INIT / (1.0 - GLOBAL_FLOOR_INIT)))))
        # per-problem pin buffers: seeded, then overwritten with the live head values when that head's
        # phase completes; only pin_b is read as a penalty target (rating, frozen in phase 2).
        self.register_buffer("pin_b", torch.zeros(n_problems))
        self.register_buffer("pin_a", torch.ones(n_problems))
        self.register_buffer("pin_c", torch.full((n_problems,), 1.0 / 3.0))
        # (theta is constructed BEFORE the trunk so the RNG draw order is fixed.)
        self.theta = nn.Embedding(n_users, 1)
        nn.init.normal_(self.theta.weight, std=1.0)
        layers, d = [], fdim
        for h in DIFF_HIDDEN:                                # decreasing layers, ReLU + dropout between
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        layers += [nn.Linear(d, 3)]                          # → (difficulty b, slope a, floor c)
        self.net = nn.Sequential(*layers)
        li = float(np.clip(LAPSE_INIT / LAPSE_CAP, EPS, 1.0 - EPS))
        self.lapse_raw = nn.Parameter(torch.tensor(float(np.log(li / (1.0 - li)))))

    def lapse(self):
        """Global 4PL upper-asymptote slip rate λ ∈ (0, LAPSE_CAP)."""
        return LAPSE_CAP * torch.sigmoid(self.lapse_raw)

    def params(self, feat):
        """(b, a, c) for the rows of `feat`. Not-yet-individualized heads return the learned global
        scalar; trained/free heads come from the shared trunk."""
        if self.training and self.act_dim > 0:               # block-drop the whole activation block
            keep = (torch.rand(len(feat), 1, device=feat.device)
                    >= ACT_DROPOUT).float() / (1.0 - ACT_DROPOUT)
            feat = torch.cat([feat[:, :self.act_dim] * keep, feat[:, self.act_dim:]], dim=1)
        if self.training and self.bw_dim > 0:                # column-drop the bot-winrate block
            d0 = self.act_dim
            keep = (torch.rand(len(feat), self.bw_dim, device=feat.device)
                    >= BW_DROPOUT).float() / (1.0 - BW_DROPOUT)
            feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.bw_dim] * keep,
                              feat[:, d0 + self.bw_dim:]], dim=1)
        if self.training and self.move_dim > 0:              # column-drop the candidate-move-eval block
            d0 = self.act_dim + self.bw_dim
            keep = (torch.rand(len(feat), self.move_dim, device=feat.device)
                    >= MOVE_DROPOUT).float() / (1.0 - MOVE_DROPOUT)
            feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.move_dim] * keep,
                              feat[:, d0 + self.move_dim:]], dim=1)
        o = self.net(feat)
        b = o[:, 0]
        a = torch.nn.functional.softplus(o[:, 1]).clamp(min=SLOPE_FLOOR)  # lower bound only
        lam = self.lapse()
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=FLOOR_CAP)
        c = cap * torch.sigmoid(o[:, 2])
        # NOT-YET-individualized heads → learned global scalar (held out of the prediction until their
        # phase); frozen (trained) heads stay backbone-produced and are held at pin_* by the LOSS.
        not_yet = {1, 2} - self.trained_heads - self.free_heads
        if 1 in not_yet:
            a = torch.nn.functional.softplus(self.gslope_raw).expand_as(a)
        if 2 in not_yet:
            c = torch.sigmoid(self.gfloor_raw).expand_as(c)
        return b, a, c

    def ability(self, u):
        """Per-user ability, hard-whitened to mean 0 / std 1 each call (unit-Gaussian gauge)."""
        w = self.theta.weight
        return ((self.theta(u) - w.mean()) / (w.std() + EPS)).squeeze(-1)

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
    # secondary-axis work — utils.rating_features.load_joint_game_phase). Metric-neutral on the
    # held-out suite (paired A/B 2026-07-13: logloss 0.4964->0.4969, survey-rho within jitter) but
    # included by FEATURE-PARITY POLICY: the rating model and the secondary-axis models train from
    # the same data sources. Appended after move, before the size indicator (dropout ranges valid);
    # no dedicated dropout (2 smooth columns cannot fingerprint problems).
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

    def score_corpus(m):
        """(b, a, c, per-profile θ, λ) for every problem/user, on cpu — the final outputs."""
        m.eval()
        with torch.no_grad():
            b_all, a_all, c_all = m.params(tfeat)
            theta_full = m.ability(torch.arange(n_profiles, device=device)).cpu().numpy()
            lam = float(m.lapse())
        return (b_all.cpu().numpy(), a_all.cpu().numpy(), c_all.cpu().numpy(), theta_full, lam)

    # ---- staged coordinate descent over the per-problem 4PL heads (rating → slope+floor) ----
    torch.manual_seed(SEED)
    m = RatingNet(fdim, n_profiles, len(keys), act_dim=k, bw_dim=nb, move_dim=nmv).to(device)
    print(f"\n=== STAGED | shared trunk + pin-weight {STAGE_PIN_WEIGHT} | "
          f"plan {[sorted(HEAD_NAME[h] for h in s) + ['ceiling'] for s, _ in STAGE_PLAN]} | "
          f"{STAGE_EPOCHS} ep/stage | dropout {ACT_DROPOUT}/{BW_DROPOUT} | ceiling LEARNED ===")

    def train_stage(free_set, epochs):
        """Free a SET of heads (data BCE) while already-trained heads are held at their stored
        per-problem values by a consistency penalty. Frozen heads stay LIVE in the prediction but
        are DETACHED from the BCE — their only gradient is the penalty (head→pin buffer), forcing the
        shared trunk to keep emitting them. The global ceiling λ trains every phase."""
        m.free_heads = set(free_set)
        frozen = m.trained_heads - m.free_heads
        m.lapse_raw.requires_grad_(True)
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(n, device=device)
            for i in range(0, n, BATCH):
                bi = perm[i:i + BATCH]
                b, a, c = m.params(tfeat[tp[bi]])            # live (not-yet heads already global)
                th = m.ability(tu[bi]); lam = m.lapse()
                b_p = b.detach() if 0 in frozen else b       # frozen heads: used live, no BCE grad
                a_p = a.detach() if 1 in frozen else a
                c_p = c.detach() if 2 in frozen else c
                pred = (c_p + (1.0 - c_p - lam) * torch.sigmoid(a_p * (th - b_p))).clamp(EPS, 1 - EPS)
                # explicit none→sum/N, NOT reduction="mean": the fused mean kernel differs from this
                # reduction at non-power-of-two batch sizes (the last, smaller minibatch), and over the
                # epochs that ULP difference compounds in the model's flat slope/floor directions.
                bce = torch.nn.functional.binary_cross_entropy(
                    pred, ty[bi], reduction="none").sum() / pred.shape[0]
                pen = torch.zeros((), device=device)
                if 0 in frozen:                              # frozen heads → maintain pin buffer
                    pen = pen + ((b - m.pin_b[tp[bi]]) ** 2).mean()
                if 1 in frozen:
                    pen = pen + ((a - m.pin_a[tp[bi]]) ** 2).mean()
                if 2 in frozen:
                    pen = pen + ((c - m.pin_c[tp[bi]]) ** 2).mean()
                loss = bce + STAGE_PIN_WEIGHT * pen
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                # human train BCE for the trace
            m.eval()
            return float(torch.nn.functional.binary_cross_entropy(m(tu, tfeat[tp]), ty))

    for free_set, tag in STAGE_PLAN:
        hb = train_stage(free_set, STAGE_EPOCHS)
        freed = sorted(HEAD_NAME[h] for h in free_set) + ["ceiling"]
        print(f"--- {tag} trained (free {freed}); human train BCE {hb:.4f} ---")
        with torch.no_grad():                                # store the just-trained heads' LIVE
            b_all, a_all, c_all = m.params(tfeat)            # per-problem values for the penalty
            for h, buf, live in [(0, m.pin_b, b_all), (1, m.pin_a, a_all), (2, m.pin_c, c_all)]:
                if h in free_set:
                    buf.copy_(live)
        m.trained_heads |= set(free_set)

    # ---- write the final artifacts to OUT_DIR ----
    b_all, a_all, c_all, theta_full, lam = score_corpus(m)
    rating = b_all + clear_logit_vec(c_all, lam) / a_all     # 80%-clear ability under the 4PL
    lo, hi = np.percentile(rating, [WINSOR, 100 - WINSOR])
    rating = np.clip(rating, lo, hi)                          # percentile winsor (robustness)
    theta = theta_full[:n_profiles]
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
        json.dump({"fat_finger_rate": round(lam, 4)}, f, indent=2)
    config = {"features": "bw_act", "move_features": True, "pca_dim": PCA_DIM,
              "diff_hidden": list(DIFF_HIDDEN), "dropout": DROPOUT, "act_dropout": ACT_DROPOUT,
              "bw_dropout": BW_DROPOUT, "move_dropout": MOVE_DROPOUT,
              "lr": LR, "batch": BATCH, "stage_epochs": STAGE_EPOCHS,
              "stage_pin_weight": STAGE_PIN_WEIGHT, "min": MIN_ATTEMPTS, "min_profile": MIN_PROFILE,
              "winsor": WINSOR, "slope_floor": SLOPE_FLOOR, "lapse_cap": LAPSE_CAP,
              "lapse_init": LAPSE_INIT, "floor_cap": FLOOR_CAP,
              "stage_slope_const": GLOBAL_SLOPE_INIT, "stage_floor_const": GLOBAL_FLOOR_INIT,
              "seed": SEED}
    torch.save({"state_dict": m.state_dict(), "keys": keys, "fdim": fdim, "k": k,
                "n_bot_cols": nb, "n_move_cols": nmv, "n_anchor": 0, "anchor_ranks": [],
                "lapse": lam, "args": config}, os.path.join(OUT_DIR, "model.pt"))
    print(f"\n=== STAGED complete === wrote ratings + user_theta.csv ({n_profiles} profiles) + "
          f"rating_constants.json (λ={lam:.5f}) + model.pt to {OUT_DIR}")


if __name__ == "__main__":
    main()
