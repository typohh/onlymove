"""Cross-board anchored feature→4PL rating model — the PRODUCTION rating trainer.

This is the stripped, production-only entry point: it trains the shipped model with the
fixed shipped configuration and writes the rating artifacts, and does nothing else (no
held-out eval, no survey-ρ / time-adjusted-ρ metrics, no DRO / gap-balance / kNN / feature-
dropout knobs). The ONLY flags are --out-dir and --seed. The full experiment + evaluation
surface — every flag, the held-out logloss / survey-ρ objectives, and the alternative
objectives (even-error DRO, gap-balance, kNN reweight, per-problem, global-slope, …) —
lives in auxiliary/train_rating_v2_research.py. Do experiments there, not here, so a test
tweak can never break the shipped ratings. (This file is the bare `-m pipeline.train_rating_v2`
zero-flag config that auxiliary/train_rating_v2_research.py also reproduces at zero flags.)

ONE model for 9x9 AND 13x13. A SHARED feature→4PL network maps each problem's features
to (difficulty b_p, slope a_p, floor c_p); per-(user, 2-month) ability θ_u is hard-gauged
to a unit Gaussian; a single GLOBAL ceiling (fat-finger / lapse) λ is learned.

    P(correct) = c_p + (1 − c_p − λ)·σ( a_p·(θ_u − b_p) )
    (b_p, a_p, c_p) = net(feat_p)        # shared net; a_p ≥ 0.01, c_p ∈ [0, 1 − 2λ]
    feat_p = joint-PCA(correct-move activations, both boards) ++ common-column bot winrate (bw_act)

The shared net carries ridge-shrunk per-problem floor/slope/difficulty residuals δ_p, fit in
a detached one-stage pass (the backbone fits the residual-free optimum; δ_p gets its own
gradient + ridge). Two humanv0 bot "users" (20k & 9d) join as extra examinees with LEARNED
ability and SOFT-target (pass-rate) responses on EVERY problem; that coverage anchors
difficulty AND slope on the empty / one-sided regions the sparse, ~80%-clear-served human
attempts cannot identify (two strengths ⇒ two points on each problem's curve). The log-slope
head is shrunk toward a fixed hierarchical prior (μ=0.8, σ=0.8). See auxiliary/test_lapse.py
and auxiliary/train_rating_v2_legacy.py for the research sandboxes.

Ability is hard-gauged (θ whitened to mean 0 / std 1 each step): the population genuinely
follows N(0, 1), and pinning the scale stops slopes inflating to compensate for compressed
ability. The shipped RATING is the ability at which the user clears the problem 80% of the
time (CALIBRATION_P) — NOT the 50%-point — so a flat low-discrimination problem correctly
needs more ability to clear at 80% than its midpoint implies:

    rating_p = b_p + logit((0.8 − c_p) / (1 − c_p − λ)) / a_p

The HUMAN-attempt term is plain weighted BCE (ERM) — the shipped objective. The global ceiling
λ is produced HERE (written to ratings_v2/rating_constants.json); pipeline.fit_population_constants
reads it through and no longer fits it posthoc.

Outputs (parallel; nothing in pwa/tsumegos/ is touched — package_for_pwa ships these):
  ratings_v2/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= 80%-clear rating)
  ratings_v2/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= raw slope a_p)
  ratings_v2/<size>/tsumego_floor_final.csv     tsumego_id, floor    (= learned c_p)
  ratings_v2/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings_v2/rating_constants.json              {"fat_finger_rate": λ}  (the learned ceiling)
  ratings_v2/model_v2.pt                        weights + run metadata

Usage:
  ./.venv/bin/python3 -m pipeline.train_rating_v2 [--out-dir ratings_v2] [--seed 0]
"""

import argparse
import json
import os
import re
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
    load_joint_move_features,
)
from auxiliary.inspect_activation_signals import _cache_ids_path  # noqa: E402
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402
from utils.fit_rank_curves import CALIBRATION_P  # noqa: E402

EPS = 1e-6
SIZES = [9, 13]
# The bot-anchor ladder size --anchor-weight is calibrated at (20k, 9d). With --anchor-fixed-mass
# the per-rung weight is rescaled by this/len(ladder) so a DENSER ladder sharpens slope
# identification (more rungs ⇒ more points on each problem's success curve) without growing the
# total bot mass and drowning the human attempts.
ANCHOR_REF_RUNGS = 2
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


def _rank_scalar(r):
    """'20k'→-20, '1d'→1 (for sorting the humanv0 ladder)."""
    return -int(r[:-1]) if r.endswith("k") else int(r[:-1])


def discover_humanv0_ranks():
    """All humanv0 benchmark ranks present on disk (attempts/<size>/...humanv0_<rank>.txt),
    weakest→strongest. Used to build the bot-floor distribution prior (min correct-move
    rate over the WHOLE ladder per problem)."""
    import glob
    rs = set()
    for sz in SIZES:
        for fn in glob.glob(os.path.join(PROJECT_ROOT, "attempts", str(sz),
                                         "b18c384nbt-humanv0_*.txt")):
            m = re.search(r"humanv0_([0-9]+[kd])\.txt$", fn)
            if m:
                rs.add(m.group(1))
    return sorted(rs, key=_rank_scalar)


def anchor_filename(token):
    """An --anchor-ranks token is either a humanv0 rank ('9d' → the humanv0 ladder file), a
    full attempt-file stem (e.g. 'kata1-b18c384nbt-s9996604416-d4316597426_t0.02' — any bot
    benchmarked by pipeline.benchmark_bots can anchor, incl. the general KataGo engine and the
    near-random 20k temperature variants), or a synthetic 'const<p>' examinee (e.g. 'const0.5':
    target pass rate p on EVERY problem — no file; one mid-curve point per problem, a
    blowup-taming pivot for regimes no real examinee observes)."""
    if re.fullmatch(r"[0-9]+[kd]", token):
        return f"b18c384nbt-humanv0_{token}.txt"
    return f"{token}.txt"


def load_anchor_winrates(keys, ranks):
    """Per-problem pass-rate for each anchor token (see anchor_filename), aligned to `keys` (the
    model's problem order). Returns float (n_keys, n_ranks), NaN where a token lacks a value for
    that problem. These are the soft targets for the bot 'users': value = P(the bot plays the
    correct first move) — the same first-move-correct semantics as a human attempt."""
    out = np.full((len(keys), len(ranks)), np.nan, dtype=np.float32)
    for j, r in enumerate(ranks):
        m = re.fullmatch(r"const([0-9.]+)", r)
        if m:
            out[:, j] = float(m.group(1))  # synthetic examinee: no file (the file loop skips it)
    key_row = {k: i for i, k in enumerate(keys)}
    for sz in sorted({s for s, _ in keys}):
        adir = os.path.join(PROJECT_ROOT, "attempts", str(sz))
        for j, r in enumerate(ranks):
            fn = os.path.join(adir, anchor_filename(r))
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


class RatingNet(nn.Module):
    """Shared feature→4PL net + per-user ability (hard unit gauge) + one global learned ceiling λ.

        P = c + (1 − c − λ)·σ( a·(θ_u − b) ),  (b, a, c) = net(feat)

    The shared net has NO per-problem parameters, so every problem's 4PL comes from its features —
    a zero-interaction problem still gets a prediction. a = softplus(·).clamp(min=0.01) (lower bound
    only, no upper cap); the floor is smoothly bounded to [0, 1 − 2λ] (ceiling-coupled: it stays a
    margin λ below the 1 − λ upper asymptote, so the active band 1 − c − λ ≥ λ > 0)."""

    def __init__(self, fdim, n_users, hidden=(128, 64), dropout=0.1, lapse_cap=0.05, lapse_init=0.01,
                 slope_floor=0.01, floor_cap=1.0, n_anchor=0, slope_discount=False,
                 hier_slope=False, hier_floor=False, global_slope=False,
                 global_slope_fixed=None,
                 mech_floor_botf=None, mech_floor_scale_fixed=None,
                 floor_resid=False, n_problems=0, per_problem=False, lapse_human_only=False,
                 lapse_fixed=None, act_dropout=0.0, act_dim=0, bw_dropout=0.0, bw_dim=0,
                 move_dropout=0.0, move_dim=0):
        super().__init__()
        self.slope_floor = float(slope_floor)
        self.floor_cap = float(floor_cap)   # hard upper bound on the guessing floor c_p
        # Hierarchical empirical-Bayes priors on the per-problem heads (--hier-slope/--hier-floor):
        # log a_p ~ N(μ_a, σ_a²), logit-floor o_c ~ N(μ_c, σ_c²). Two-pass EM with FROZEN pass-1
        # robust moments (see main): pass 1 fits free, then μ = median, σ = IQR/1.349 of the
        # model's OWN unpenalized spectrum become a fixed quadratic-shrinkage target for pass 2.
        # Two simpler schemes are verified-degenerate: gradient-learned σ collapses via the
        # unbounded N·log σ reward (Neyman-Scott — every problem got the identical 4PL, survey
        # rho −0.11), and per-step stop-grad moment matching cold-start-locks (net outputs start
        # near-constant ⇒ σ̂≈0 ⇒ shrinkage force ≫ data gradient forever). A frozen target is
        # stationary — no ratchet, no lock-in — and still answer-free: the target is the corpus'
        # own fitted distribution, nothing imported from legacy.
        self.hier_slope, self.hier_floor = bool(hier_slope), bool(hier_floor)
        # --act-dropout: drop the ENTIRE activation feature block per row (train only), scaled
        # 1/(1−p). The activation PCs are near-unique per-problem fingerprints — the measured
        # memorization channel (Findings F, S: the net attaches attempted problems' base rates
        # to their signatures; removing the block entirely improves unseen-problem transfer).
        # Block dropout regularizes that reliance without losing the features at eval time:
        # the net must keep working when the fingerprint is absent, so base-rate information
        # routes through the transferable bot-winrate columns instead.
        self.act_dropout, self.act_dim = float(act_dropout), int(act_dim)
        # --bw-dropout: elementwise dropout on the bot-winrate columns (train only, scaled).
        # Per-bot pass rates are ~50% overdispersed vs human difficulty — column dropout keeps
        # the net from leaning on any single rung's idiosyncratic value and pushes it toward
        # ladder consensus.
        self.bw_dropout, self.bw_dim = float(bw_dropout), int(bw_dim)
        # --move-dropout: elementwise dropout on the candidate-move-eval columns (train only,
        # scaled). The winning-move + 9-alt (logit-winrate, score) block is largely redundant with
        # the bot-winrate ladder and the activation PCs, so unregularized it adds residual-head
        # capacity that overfits attempted problems and worsens unseen-problem transfer (measured:
        # held-out-problem logloss and new-problem flatness both regress). Column dropout forces the
        # prior to treat it as corroborating, not load-bearing. Block sits at cols
        # [act_dim+bw_dim : act_dim+bw_dim+move_dim] (before the trailing size indicator).
        self.move_dropout, self.move_dim = float(move_dropout), int(move_dim)
        # --global-slope: ONE learned discrimination for the whole corpus (3PL+lapse reference
        # model). Motivated by an (accidental but clean) finding: a constant-slope fit had the
        # best held-out-problem transfer and honest-val calibration of any healthy run — much of
        # the unconstrained net's per-problem slope variation is overfit. This is the explicit,
        # reproducible version; the net's slope column is ignored.
        self.global_slope = None
        if global_slope_fixed is not None:
            # --global-slope-fixed: a PINNED constant discrimination (not learned) — deletes the
            # hier machinery's regime-sensitive cached (μ, σ) target outright. Per-problem slope
            # stays available only through the data-attested η_p residual column.
            self.global_slope = nn.Parameter(
                torch.tensor(float(np.log(np.expm1(global_slope_fixed)))), requires_grad=False)
        elif global_slope:
            self.global_slope = nn.Parameter(torch.tensor(float(np.log(np.expm1(2.0)))))
        # --mech-floor: the guessing floor is MEASURED, not learned. c_p = s·bot_floor_p where
        # bot_floor_p = min pass-rate over the standard-temperature humanv0 ladder (the empirical
        # guess rate; same convention as auxiliary/diag_4pl_structure.py) and s is ONE global
        # human-vs-bot scalar (independently measured ≈0.70; legacy shipped exactly 0.70·bot).
        # Rationale: the floor is the one 4PL parameter with a physical meaning a measurement can
        # pin (chance of guessing the move among plausible candidates). Removing the learned head
        # closes the floor⇄difficulty swap BY CONSTRUCTION (no per-problem floor direction left to
        # relocate difficulty into) and bounds the easy end with the bot floor (the universal
        # easy-tail inflation wart no learned config fixed). Per-problem floor freedom survives
        # only as the detached/frozen δ_floor residual where human attempts attest it.
        self.mech_floor_scale = None
        if mech_floor_botf is not None:
            self.register_buffer("botf", mech_floor_botf)
            init_s = 0.70 if mech_floor_scale_fixed is None else float(mech_floor_scale_fixed)
            self.mech_floor_scale = nn.Parameter(
                torch.tensor(float(np.log(np.expm1(init_s)))),
                requires_grad=mech_floor_scale_fixed is None)
        # --floor-residual: per-problem additive logit δ_p on the floor head, ridge-shrunk to 0.
        # The Finding-O architecture test: floors are locally identifiable (~0.04 nats by-row)
        # but NOT feature-predictable beyond f(rating) (smooth-family models transfer far
        # better), so the only way to get both is prior-from-features + attested local residual
        # — unattested/held-out problems keep δ=0 and fall back to the feature prior.
        # Frozen-pass per-problem residuals (gate: --floor-residual; columns: --resid-cols).
        # One (n_problems, 3) table of additive corrections (Δdifficulty, Δraw-slope, Δfloor-
        # logit), masked to the requested columns — fit ONLY in the frozen pass (see main).
        # Generalizes the original floor-only residual after the per-problem reference model
        # showed ~0.06 nats of real local heterogeneity beyond any pooled fit.
        self.floor_resid = None
        if floor_resid:
            self.floor_resid = nn.Embedding(n_problems, 3)
            nn.init.zeros_(self.floor_resid.weight)
            self.register_buffer("resid_mask", torch.zeros(3))
        # --per-problem: a unique FREE 3PL per tsumego (no feature net at all) — the maximal-
        # local reference model. Identification rests entirely on examinee coverage (run with
        # --anchor-ranks all = the full 61-bot battery, down-weighted); without it this is the
        # Finding-R swap waiting to happen. Held-out problems' params are never trained, so the
        # held-out-problem metric is MEANINGLESS under this flag.
        self.per_problem = bool(per_problem)
        self.pp = None
        if per_problem:
            self.pp = nn.Embedding(n_problems, 3)             # (difficulty b, raw slope, raw floor)
            with torch.no_grad():
                self.pp.weight[:, 0] = 0.0
                self.pp.weight[:, 1] = float(np.log(np.expm1(2.0)))   # softplus⁻¹(2): slope init 2
                self.pp.weight[:, 2] = 0.0                            # floor init mid-cap
        # --lapse-human-only: the global ceiling models HUMAN slips; slip-free bot examinees
        # (e.g. kata1 t0.02) otherwise anchor it to zero (Finding T). Bot rows get lapse 0 —
        # their miss process is already in their learned ability/temperature.
        self.lapse_human_only = bool(lapse_human_only)
        self.theta = nn.Embedding(n_users, 1)
        nn.init.normal_(self.theta.weight, std=1.0)
        # Per-bot steepness discount (--anchor-slope-discount): bot examinees see each
        # problem's curve flattened by a learned per-bot factor (softplus, init 1.0). Bot
        # response curves across the ladder are FLATTER than human curves (the ~50%
        # overdispersion of bot-derived difficulty), so anchoring many rungs as plain
        # examinees forces bot-flat slopes/floors into the SHIPPED human-scale params
        # (verified: dense-20 ladder → 97% hard-tail slope clamp, floors → 0; see
        # docs/rating_4pl_structure_log.md). The discount absorbs that mismatch per bot:
        # bots keep teaching difficulty location and RELATIVE shape across problems;
        # humans alone pin the absolute steepness scale. Slope-side analogue of legacy's
        # human-recalibrated (0.70·) bot floor, but learned.
        self.n_human = n_users - n_anchor
        self.slope_discount = None
        if slope_discount and n_anchor > 0:
            self.slope_discount = nn.Parameter(
                torch.full((n_anchor,), float(np.log(np.expm1(1.0)))))  # softplus⁻¹(1)
        if per_problem:
            self.net = None                                    # no feature net at all
        else:
            layers, d = [], fdim
            for h in hidden:                                   # decreasing layers, ReLU between
                layers += [nn.Linear(d, h), nn.ReLU()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
                d = h
            layers += [nn.Linear(d, 3)]                        # → (difficulty b, slope a, floor c)
            self.net = nn.Sequential(*layers)
        self.lapse_cap = float(lapse_cap)
        # --lapse-fixed: freeze the global ceiling at an externally calibrated value (the
        # posthoc plateau λ̂) instead of trusting the trained MLE, which under-estimates λ
        # whenever floors are free (Finding Q) — folds the posthoc rescoring stage into the fit.
        if lapse_fixed is not None:
            assert 0.0 < lapse_fixed < lapse_cap, "--lapse-fixed must lie in (0, lapse_cap)"
            lapse_init = lapse_fixed
        li = float(np.clip(lapse_init / max(lapse_cap, EPS), EPS, 1 - EPS))
        self.lapse_raw = nn.Parameter(torch.tensor(float(np.log(li / (1 - li)))))
        if lapse_fixed is not None:
            self.lapse_raw.requires_grad_(False)

    def lapse(self):
        """Global 4PL upper-asymptote slip rate λ ∈ (0, lapse_cap)."""
        return self.lapse_cap * torch.sigmoid(self.lapse_raw)

    def params(self, feat, pidx=None, use_resid=True, detach_backbone=False, pin=None):
        """pidx = problem indices (key order) for the rows of `feat`; None ⇒ feat IS the full
        corpus in key order (the score_corpus / prior-block convention).
        use_resid=False evaluates the bare backbone (no δ_p); detach_backbone=True cuts the
        gradient to everything upstream of the δ table — together they implement the ONE-STAGE
        detached residual mode (--floor-resid-mode detached): the backbone trains residual-free
        while δ_p trains against a frozen-in-the-graph copy of it, reproducing the two-stage
        frozen fixed point in a single optimization pass."""
        if self.pp is not None:
            if pidx is None:
                pidx = torch.arange(len(feat), device=feat.device)
            o = self.pp(pidx)
        else:
            if self.training and self.act_dropout > 0 and self.act_dim > 0:
                keep = (torch.rand(len(feat), 1, device=feat.device)
                        >= self.act_dropout).float() / (1.0 - self.act_dropout)
                feat = torch.cat([feat[:, :self.act_dim] * keep, feat[:, self.act_dim:]], dim=1)
            if self.training and self.bw_dropout > 0 and self.bw_dim > 0:
                d0 = self.act_dim
                keep = (torch.rand(len(feat), self.bw_dim, device=feat.device)
                        >= self.bw_dropout).float() / (1.0 - self.bw_dropout)
                feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.bw_dim] * keep,
                                  feat[:, d0 + self.bw_dim:]], dim=1)
            if self.training and self.move_dropout > 0 and self.move_dim > 0:
                d0 = self.act_dim + self.bw_dim
                keep = (torch.rand(len(feat), self.move_dim, device=feat.device)
                        >= self.move_dropout).float() / (1.0 - self.move_dropout)
                feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.move_dim] * keep,
                                  feat[:, d0 + self.move_dim:]], dim=1)
            o = self.net(feat)
        if self.global_slope is not None:
            # substitute the single global slope into the raw-slope column so per-problem slope
            # residuals (and the detach below) compose with it like with the net's slope head
            gs = self.global_slope.view(1, 1).expand(len(o), 1)
            o = torch.cat([o[:, :1], gs, o[:, 2:]], dim=1)
        if self.mech_floor_scale is not None:
            # substitute logit(s·bot_floor_p) into the floor column so the cap·sigmoid pathway,
            # δ_floor residuals, detach and head-pin all compose unchanged (same pattern as the
            # global-slope substitution above)
            pid = pidx if pidx is not None else torch.arange(len(o), device=o.device)
            s = torch.nn.functional.softplus(self.mech_floor_scale)
            cf = (s * self.botf[pid]).clamp(EPS, 1.0 - EPS)
            o = torch.cat([o[:, :2], torch.log(cf / (1.0 - cf)).unsqueeze(1)], dim=1)
        if detach_backbone:
            o = o.detach()
        if pin:
            # --head-pin-dropout: this batch sees the pinned head(s) replaced by a corpus-
            # median constant — no gradient reaches the pinned head, the others must fit the
            # rows alone (head-level dropout; the user's compensation-learning test).
            o = o.clone()
            for col_, v_ in pin.items():
                o[:, col_] = v_
        if use_resid and self.floor_resid is not None:
            if pidx is None:
                pidx = torch.arange(len(o), device=o.device)
            o = o + self.floor_resid(pidx) * self.resid_mask
        b = o[:, 0]
        a = torch.nn.functional.softplus(o[:, 1]).clamp(min=self.slope_floor)  # lower bound only
        lam = self.lapse()
        if detach_backbone:
            lam = lam.detach()
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=self.floor_cap)
        c = cap * torch.sigmoid(o[:, 2])
        return b, a, c

    def slope_unclamped(self, feat):
        """Per-problem slope BEFORE the slope_floor clamp — the distribution prior penalizes
        this (the clamp zeroes gradients exactly where collapsed slopes need lifting)."""
        return torch.nn.functional.softplus(self.net(feat)[:, 1]).clamp_min(1e-4)

    def hier_nll(self, feat, target):
        """Mean per-problem quadratic shrinkage toward the FROZEN pass-1 robust moments in
        `target` ({head: (μ, σ)}; see __init__). Uses the PRE-clamp slope (gradient survives at
        the slope_floor rail) and the raw floor logit."""
        o = self.net(feat)
        loss = torch.zeros((), device=o.device)
        if "log-slope" in target:
            mu, sig = target["log-slope"]
            la = torch.log(torch.nn.functional.softplus(o[:, 1]).clamp_min(1e-4))
            loss = loss + ((la - mu).pow(2) / (2 * sig * sig)).mean()
        if "floor-logit" in target:
            mu, sig = target["floor-logit"]
            loss = loss + ((o[:, 2] - mu).pow(2) / (2 * sig * sig)).mean()
        return loss

    def cond_diversity_nll(self, feat, tgt):
        """Continuous conditional diversity+decorrelation objective (user proposal, kept as a
        falsifiable TEST): residualize both heads on the frozen pass-1 conditional means
        μ(rating) in `tgt` ({head: (μ_vec, σ_resid)}), then (a) pull each head's residual spread
        toward its pass-1 value (diversity), (b) penalize the residual cross-correlation
        (decorrelation given rating). Prediction on record: structure tables improve while
        slope↔bot_steep stays ≈0 — diversity rewards manufacture ASSIGNMENT noise along the
        flat directions of the likelihood (Findings G+I)."""
        o = self.net(feat)
        vals = {"log-slope": torch.log(torch.nn.functional.softplus(o[:, 1]).clamp_min(1e-4)),
                "floor-logit": o[:, 2]}
        pen = torch.zeros((), device=o.device)
        resid = {}
        for head, v in vals.items():
            mu, sig = tgt[head]
            r = v - mu
            pen = pen + (r.std() / sig - 1.0).pow(2)
            resid[head] = r - r.mean()
        ra, rc = resid["log-slope"], resid["floor-logit"]
        corr = (ra * rc).mean() / (ra.std() * rc.std() + 1e-6)
        return pen + corr.pow(2)

    def hier_moments(self, feat):
        """Current corpus moments the shrinkage is matched to (for reporting)."""
        with torch.no_grad():
            o = self.net(feat)
            out = {}
            if self.hier_slope:
                la = torch.log(torch.nn.functional.softplus(o[:, 1]).clamp_min(1e-4))
                out["log-slope"] = (float(la.mean()), float(la.std()))
            if self.hier_floor:
                out["floor-logit"] = (float(o[:, 2].mean()), float(o[:, 2].std()))
        return out

    def ability(self, u):
        """Per-user ability, hard-whitened to mean 0 / std 1 each call (unit-Gaussian gauge)."""
        w = self.theta.weight
        return ((self.theta(u) - w.mean()) / (w.std() + EPS)).squeeze(-1)

    def forward(self, u, feat, pidx=None, use_resid=True, detach_backbone=False, pin=None):
        b, a, c = self.params(feat, pidx, use_resid=use_resid, detach_backbone=detach_backbone,
                              pin=pin)
        th = self.ability(u)
        lam = self.lapse()
        if detach_backbone:
            th, lam = th.detach(), lam.detach()
        g = a * (th - b)
        if self.slope_discount is not None:
            is_bot = u >= self.n_human
            disc = torch.nn.functional.softplus(
                self.slope_discount)[(u - self.n_human).clamp(min=0)]
            if detach_backbone:
                disc = disc.detach()
            g = torch.where(is_bot, g * disc, g)
        praw = torch.sigmoid(g)
        if self.lapse_human_only:
            lam = lam * (u < self.n_human).float()
        return (c + (1.0 - c - lam) * praw).clamp(EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "ratings_v2"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Production configuration ──────────────────────────────────────────────
    # These were tunable flags; the shipped run uses these fixed values (the bare
    # zero-flag config). The full experiment + evaluation surface — every flag, the
    # held-out / survey-ρ objectives, and the alternative objectives (even-error DRO,
    # gap-balance, kNN reweight, per-problem, global-slope, …) — lives in
    # auxiliary/train_rating_v2_research.py. Do experiments THERE, not here.
    args.features = "bw_act"            # activations ⊕ common-column bot winrate
    args.move_features = False
    args.pca_dim = 128                  # joint activation PCs
    args.diff_hidden = [128, 64]        # shared feature-net hidden widths
    args.dropout = 0.1
    args.act_dropout = 0.0
    args.bw_dropout = 0.0
    args.move_dropout = 0.0
    args.epochs = 300
    args.lr = 0.01
    args.batch = 65536
    args.log_every = 50
    args.min = 5                        # min first-time correct AND incorrect to keep a user
    args.min_profile = 8
    args.winsor = 1.0                   # winsorize final ratings to [1, 99] pct
    args.slope_floor = 0.01
    args.lapse_cap = 0.05
    args.lapse_fixed = 0.02
    args.floor_cap = 1.0
    args.mech_floor = True              # c_p = s·bot_floor_p over the humanv0 ladder
    args.mech_floor_scale_fixed = 0.70  # fixed floor scale s
    args.hier_slope = True              # hierarchical EB shrinkage on the log-slope head
    args.hier_slope_mu = 0.8            # FIXED shrinkage target (single-pass; no pass-1 moments)
    args.hier_slope_sigma = 0.8
    args.hier_weight = 1.0
    args.hier_floor = False
    args.floor_residual = True          # ridge-shrunk per-problem residuals δ_p
    args.floor_resid_mode = "detached"  # one-stage detached backbone (δ fits the clean optimum)
    args.floor_resid_l2 = 4.0
    args.resid_cols = "diff,slope,floor"
    args.anchors = True                 # humanv0 bot "users" as coverage examinees
    args.anchor_ranks = "20k,9d"
    args.anchor_weight = 0.2
    args.anchor_fixed_mass = True
    args.anchor_slope_discount = False
    args.global_slope = False
    args.global_slope_fixed = None
    args.per_problem = False
    args.lapse_human_only = False

    # --- features (joint-PCA correct-move activations ⊕ common-column bot winrate
    #     [⊕ candidate-move evaluations]) ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]                  # (size, tid); scorable iff both blocks
    key_idx = {key: i for i, key in enumerate(keys)}
    use_act = args.features in ("bw_act", "act")
    use_bw = args.features in ("bw_act", "bw")
    use_mv = args.move_features
    # The move block is appended AFTER the bot columns and BEFORE the size indicator, so the
    # [act | bw | move | size] layout keeps --act-dropout (cols :k) and --bw-dropout (cols
    # k:k+nb) valid. Move features don't gate the scorable set (it stays act∩bw — see --features);
    # a problem missing a row is mean-imputed (zero after the pooled z-score).
    mv, nmv = (load_joint_move_features(SIZES, tids_by_size) if use_mv else ({}, 0))
    zero_mv = np.zeros(nmv, dtype=np.float32)
    fdim = (k if use_act else 0) + (nb if use_bw else 0) + nmv + 1  # + size_is_13 board indicator
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        blocks = ([act[key]] if use_act else []) + ([bw[key]] if use_bw else [])
        if use_mv:
            blocks.append(mv.get(key, zero_mv))
        feat[i] = np.concatenate(blocks + [[1.0 if key[0] == 13 else 0.0]])
    n_mv_miss = sum(1 for key in keys if key not in mv) if use_mv else 0
    print(f"scorable problems {len(keys)} (9:{sum(s == 9 for s, _ in keys)} "
          f"13:{sum(s == 13 for s, _ in keys)}) | feat dim {fdim} "
          f"[{args.features}{'+move' if use_mv else ''}] (act {k if use_act else 0} + bot "
          f"{nb if use_bw else 0}{f' + move {nmv} ({n_mv_miss} imputed)' if use_mv else ''} "
          f"+ size_is_13)")

    botf_t = None
    if args.mech_floor:
        ladder = discover_humanv0_ranks()
        lw = load_anchor_winrates(keys, ladder)               # (n_keys, n_rungs), NaN where absent
        botf = np.nanmin(lw, axis=1)
        n_nan = int(np.isnan(botf).sum())
        botf = np.where(np.isnan(botf), np.nanmedian(botf), botf)
        botf_t = torch.tensor(botf.astype(np.float32), device=device)
        print(f"mech floor: c_p = s·bot_floor_p over {len(ladder)}-rung humanv0 ladder | "
              f"bot_floor p[5,50,95] {np.percentile(botf, [5, 50, 95]).round(3)} | "
              f"{n_nan} problems lack ladder data (median-filled) | "
              f"s {'fixed ' + str(args.mech_floor_scale_fixed) if args.mech_floor_scale_fixed is not None else 'learned (init 0.70)'}")

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

    # --- bot 'anchor users': extra examinees (learned ability) with soft-target responses on every
    #     problem. Coverage is the prior; each rung is a point on every problem's success curve, so
    #     the SLOPE a_p is only identified where adjacent rungs separate the problem — a 2-rung
    #     (20k,9d) ladder floors the slope wherever those two don't (→ rating=b+clear_logit/a blows
    #     up). A denser --anchor-ranks adds separating rungs; --anchor-fixed-mass keeps total mass. ---
    n_anchor, ranks = 0, []
    au = ap_ = at = np.empty(0)
    per_bot_weight = args.anchor_weight
    if args.anchors:
        if args.anchor_ranks.strip().lower() == "all":
            # every attempt file common to both sizes (the full bot battery, 61 examinees)
            import glob as _glob
            stems = [set(os.path.basename(f)[:-4]
                         for f in _glob.glob(os.path.join(PROJECT_ROOT, "attempts", str(s), "*.txt")))
                     for s in SIZES]
            ranks = sorted(set.intersection(*stems))
        else:
            ranks = [r.strip() for r in args.anchor_ranks.split(",") if r.strip()]
        aw = load_anchor_winrates(keys, ranks)
        n_anchor = len(ranks)
        if args.anchor_fixed_mass and n_anchor:
            per_bot_weight = args.anchor_weight * ANCHOR_REF_RUNGS / n_anchor
        uu, pp_, tt_ = [], [], []
        for j in range(n_anchor):
            ok = np.where(np.isfinite(aw[:, j]))[0]
            uu.append(np.full(len(ok), n_profiles + j, np.int64))
            pp_.append(ok.astype(np.int64))
            tt_.append(aw[ok, j].astype(np.float32))
        au, ap_, at = np.concatenate(uu), np.concatenate(pp_), np.concatenate(tt_)
        cover = [int(np.isfinite(aw[:, j]).sum()) for j in range(n_anchor)]
        print(f"anchors: {n_anchor} bot users {ranks} | coverage {cover}/{len(keys)} | "
              f"per-rung weight {per_bot_weight:.4f} (total bot mass "
              f"{'FIXED' if args.anchor_fixed_mass else 'grows'}) | abilities LEARNED")

    # --- assemble training rows: every human attempt (w=1) ⊕ anchor coverage (w=anchor_weight) ---
    train_h = np.ones(len(y), bool)
    u_all = [prof_idx[train_h]]; p_all = [prob_idx[train_h]]; t_all = [y[train_h]]
    w_all = [np.ones(int(train_h.sum()), np.float32)]
    if n_anchor:
        u_all.append(au); p_all.append(ap_); t_all.append(at)
        w_all.append(np.full(len(ap_), per_bot_weight, np.float32))
    n_human = int(train_h.sum())                             # human rows lead; anchors follow
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

    # --- fit: weighted BCE + detached residuals + fixed hier-slope shrinkage (single pass) ---
    epoch_records = []                                        # (pass, epoch, human_train_bce, λ)

    def score_corpus(m):
        """(b, a, c, all-user-θ, λ) for every problem/user, on cpu — used for the DRO binning
        (pass-1) and the final outputs (pass-2)."""
        m.eval()
        with torch.no_grad():
            b_all, a_all, c_all = m.params(tfeat)
            theta_full = m.ability(
                torch.arange(n_profiles + n_anchor, device=device)).cpu().numpy()
            lam = float(m.lapse())
        return (b_all.cpu().numpy(), a_all.cpu().numpy(), c_all.cpu().numpy(),
                theta_full, lam)

    def fit_model(hier_target=None):
        """Train one RatingNet on weighted BCE (human attempts w=1 ⊕ bot-anchor coverage
        w=anchor_weight), with the detached one-stage floor-residual term and the fixed
        hierarchical log-slope shrinkage prior. Single pass — the shipped objective. The
        DRO / gap-balance / kNN alternatives + the held-out convergence trace live in
        auxiliary/train_rating_v2_research.py."""
        torch.manual_seed(args.seed)
        m = RatingNet(fdim, n_profiles + n_anchor, tuple(args.diff_hidden),
                      dropout=args.dropout, lapse_cap=args.lapse_cap,
                      slope_floor=args.slope_floor, floor_cap=args.floor_cap,
                      n_anchor=n_anchor,
                      slope_discount=args.anchor_slope_discount,
                      hier_slope=args.hier_slope, hier_floor=args.hier_floor,
                      global_slope=args.global_slope,
                      global_slope_fixed=args.global_slope_fixed,
                      mech_floor_botf=botf_t,
                      mech_floor_scale_fixed=args.mech_floor_scale_fixed,
                      floor_resid=args.floor_residual, n_problems=len(keys),
                      per_problem=args.per_problem,
                      lapse_human_only=args.lapse_human_only,
                      lapse_fixed=args.lapse_fixed,
                      act_dropout=args.act_dropout,
                      act_dim=k if use_act else 0,
                      bw_dropout=args.bw_dropout,
                      bw_dim=nb if use_bw else 0,
                      move_dropout=args.move_dropout,
                      move_dim=nmv if use_mv else 0).to(device)
        # which heads carry the ridge-shrunk per-problem residual δ_p (production: all three)
        ci = {"diff": 0, "slope": 1, "floor": 2}
        with torch.no_grad():
            for cn in args.resid_cols.split(","):
                m.resid_mask[ci[cn.strip()]] = 1.0
        # detached residuals: keep δ_p OUT of the main optimizer; it trains in its own param
        # group (lr 0.05) against the DETACHED backbone, so the backbone fits exactly what a
        # residual-free run would and δ_p reaches the frozen two-stage solution's ridge optimum.
        main_params = [p for nm, p in m.named_parameters() if nm != "floor_resid.weight"]
        opt = torch.optim.Adam([{"params": main_params},
                                {"params": [m.floor_resid.weight], "lr": 0.05}], lr=args.lr)
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(n, device=device)
            for b0 in range(0, n, args.batch):
                bi = perm[b0:b0 + args.batch]
                opt.zero_grad()
                # backbone term WITHOUT δ_p (use_resid=False): the backbone fits the clean optimum
                p = m(tu[bi], tfeat[tp[bi]], tp[bi], use_resid=False)
                l = torch.nn.functional.binary_cross_entropy(p, tt[bi], reduction="none")
                w = tw[bi]
                loss = (l * w).sum() / w.sum().clamp_min(EPS)
                # one-stage residual term: backbone detached in the graph, so ONLY δ_p receives
                # this gradient, plus its ridge — converging to the frozen two-stage solution.
                m.eval()      # dropout off: δ fits the clean backbone
                pr = m(tu[bi], tfeat[tp[bi]], tp[bi], detach_backbone=True)
                m.train()
                lres = torch.nn.functional.binary_cross_entropy(pr, tt[bi], reduction="none")
                loss = loss + ((lres * w).sum() / w.sum().clamp_min(EPS)
                               + args.floor_resid_l2 * m.floor_resid.weight.pow(2).mean())
                if hier_target is not None:
                    # Fixed hierarchical log-slope shrinkage prior (dropout off for a clean read).
                    # Bayes-correct scale: in the per-row-mean units the loss is in, the prior
                    # contributes (n_problems / total row mass)·mean-NLL — a derived constant.
                    m.eval()
                    hier_scale = args.hier_weight * len(keys) / float(tw.sum())
                    loss = loss + hier_scale * m.hier_nll(tfeat, hier_target)
                    m.train()
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                ll = torch.nn.functional.binary_cross_entropy(m(thu, tfeat[thp], thp), thy).item()
            lam_ep = float(m.lapse())
            epoch_records.append((1, ep, ll, lam_ep))
            if ep % args.log_every == 0 or ep == args.epochs - 1:
                print(f"  [mean] epoch {ep:4d}  human train BCE {ll:.4f}  λ {lam_ep:.5f}")
        return m

    # FIXED hierarchical log-slope shrinkage target (μ=0.8, σ=0.8) → straight into the single fit.
    # The shrinkage equilibrium barely depends on a free pass-1 fit (Finding G/K), so the target is
    # cached rather than re-derived per run. It IS regime-sensitive (Finding DD): re-derive it (the
    # two-pass mode in auxiliary/train_rating_v2_research.py) whenever the anchor or feature set
    # changes, and update args.hier_slope_mu/sigma above.
    hier_target = {"log-slope": (torch.full((len(keys),), args.hier_slope_mu, device=device),
                                 args.hier_slope_sigma)}
    print(f"  hier shrinkage target FIXED: log-slope μ {args.hier_slope_mu:+.3f} "
          f"σ {args.hier_slope_sigma:.3f}")
    m = fit_model(hier_target=hier_target)

    # detached one-stage residual diagnostic (δ_p spread per head)
    if args.floor_residual:
        with torch.no_grad():
            d_ = (m.floor_resid.weight * m.resid_mask).cpu().numpy()
        names = ["diff", "slope", "floor"]
        stats = " | ".join(
            f"δ_{names[j]} std {d_[:, j].std():.3f} |>0.1|: "
            f"{100 * float((np.abs(d_[:, j]) > 0.1).mean()):.0f}%"
            for j in range(3) if float(m.resid_mask[j]) > 0)
        print(f"  detached one-stage residuals (cols {args.resid_cols}): {stats}")

    # --- score the whole corpus (final model) ---
    b_arr, a, c, theta_full, lam = score_corpus(m)
    theta = theta_full[:n_profiles]
    bot_theta = theta_full[n_profiles:]
    rating = b_arr + clear_logit_vec(c, lam) / a             # 80%-clear ability under the 4PL
    lo, hi = np.percentile(rating, [args.winsor, 100 - args.winsor])
    rating = np.clip(rating, lo, hi)                          # percentile winsor (legacy robustness)
    if m.mech_floor_scale is not None:
        print(f"  mech-floor scale s (c_p = s·bot_floor_p): "
              f"{float(torch.nn.functional.softplus(m.mech_floor_scale)):.3f}"
              f"{' (fixed)' if not m.mech_floor_scale.requires_grad else ''}")
    print(f"  λ {lam:.5f} | floor mean {c.mean():.3f} p[5,50,95] "
          f"{np.percentile(c, [5, 50, 95]).round(3)} | slope p[1,50,99] "
          f"{np.percentile(a, [1, 50, 99]).round(2)} | rating winsor [{lo:.2f},{hi:.2f}] "
          f"| ability std {theta.std():.3f}")
    if hier_target is not None:
        print("  hier shrinkage: target (fixed) "
              + " | ".join(f"{k} μ~{float(np.median(mu.cpu().numpy())):+.3f} σ {sg:.3f}"
                           for k, (mu, sg) in hier_target.items())
              + "  →  final corpus moments "
              + " | ".join(f"{k} μ {mu:+.3f} σ {sg:.3f}"
                           for k, (mu, sg) in m.hier_moments(tfeat).items()))
    if n_anchor:
        print("  bot-user learned ability θ: "
              + ", ".join(f"{ranks[j]}={bot_theta[j]:+.2f}" for j in range(n_anchor)))

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
