"""STAGED (coordinate-descent) fork of the rating trainer — an experiment harness.

The production superset: every flag the shipped configuration was selected under, plus a `--staged`
mode that fits the per-problem 4PL heads ONE AT A TIME (rating → slope → floor) instead of jointly
(`--production-flags` reproduces what pipeline.train_rating bakes in). Each staged run
frees a single head (still backbone/feature-predicted, with 0.3 act/bw dropout), trains it, then
FREEZES it at its learned PER-PROBLEM values (stored in pin_b/pin_a/pin_c buffers, not reverted to a
global mean) before the next head is freed. Slope/floor start pinned to constants (1.0 / 0.333). The
backbone trunk is shared, so a frozen head is held by substituting its stored per-problem value in
(b,a,c) space — the trunk's gradient reaches only the free head each stage. After every stage it
writes per-size CSVs + runs auxiliary.rating_dev.validate_ratings, so each head's MARGINAL
contribution (1PL → 2PL → 3PL ladder) is visible. The non-staged path is the plain joint fit — same
data prep, anchors and eval, no staging. Run: `--staged --stage-epochs 100
--no-mech-floor --act-dropout 0.3 --bw-dropout 0.3 --holdout-problem-frac 0.1 --out-dir /tmp/...`.

Original docstring follows.
======================================================================================================
Cross-board anchored feature→4PL rating model — the rating pipeline.

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
see auxiliary/test_lapse.py (research sandbox) and auxiliary/train_rating_legacy.py
(the preserved residual+DRO+fixed-bot-floor model, for future development).

Ability is hard-gauged (θ whitened to mean 0 / std 1 each step): the population genuinely
follows N(0, 1), and pinning the scale stops slopes inflating to compensate for compressed
ability. The shipped RATING is the ability at which the user clears the problem 80% of the
time (CALIBRATION_P) — NOT the 50%-point — so a flat low-discrimination problem correctly
needs more ability to clear at 80% than its midpoint implies:

    rating_p = b_p + logit((0.8 − c_p) / (1 − c_p − λ)) / a_p

The global ceiling λ is produced HERE (written to ratings/rating_constants.json);
pipeline.fit_population_constants reads it through and no longer fits it posthoc.

OBJECTIVE — plain mean BCE (ERM). An optional base-rate-centered even-error group-DRO is available
behind `--dro-skill`; it is OFF by default and is NOT part of the shipped config (production
pipeline.train_rating runs plain BCE). The paragraph below describes that opt-in objective.
Ported from the legacy residual model. Plain mean BCE is gradient-dominated by the dense mid-ranks, so it under-fits
the sparse difficulty TAILS — exactly where strong/grinder users live, who then over-rate. So the
HUMAN-attempt term is a two-pass online group-DRO (Sagawa et al.) over rating(×gap) bins: pass 1
trains mean BCE to get the rating that defines the bins, pass 2 reweights bins by their loss
EXCESS over each bin's base-rate entropy (the aleatoric floor — centering is load-bearing, raw-loss
DRO chases that floor and loses to ERM). The bot-anchor coverage rows keep their flat weighted BCE.
`--dro-skill` opts in; the default (and production) is plain weighted BCE (ERM). A per-bin
train-logloss table prints under either objective (`--report-bins`) so a DRO and an ERM run compare
bin-for-bin. See auxiliary/train_rating_legacy.py.

Outputs (parallel; nothing in pwa/tsumegos/ is touched — package_for_pwa ships these):
  ratings/<size>/tsumego_ratings.csv         tsumego_id, rating   (= 80%-clear rating)
  ratings/<size>/tsumego_slopes.csv          tsumego_id, slope    (= raw slope a_p)
  ratings/<size>/tsumego_floors.csv          tsumego_id, floor    (= learned c_p)
  ratings/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings/rating_constants.json              {"fat_finger_rate": λ}  (the learned ceiling)
  ratings/model.pt                        weights + run metadata

Pass --production-flags to reproduce the shipped config; the other knobs are for experiments.

Usage:
  ./.venv/bin/python3 -m pipeline.train_rating [--epochs 300] [--anchor-weight 0.2] ...
  ./.venv/bin/python3 -m pipeline.train_rating --no-dro-skill   # plain-BCE ERM baseline
  # compare objectives without writing CSVs (per-bin logloss + held-out + survey ρ):
  ./.venv/bin/python3 -m pipeline.train_rating --holdout-problem-frac 0.1 [--no-dro-skill]
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# NOTE (prototype tech-debt): these live in auxiliary/ today. When this backend stabilizes,
# promote load_joint_*/build_profiles + the activation-cache helpers into utils/.
from auxiliary.rating_dev.model_comparison import (  # noqa: E402
    build_profiles, load_joint_activations, load_joint_alt_activations,
    load_joint_bot_winrate, load_joint_continuation, load_joint_game_phase,
    load_joint_move_features,
)
from utils.rating.features import _cache_ids_path, load_first_time_attempts  # noqa: E402
from utils.rating.fit_rank_curves import CALIBRATION_P  # noqa: E402
from utils.rating.ratings_io import FLOOR_CSV, RATINGS_CSV, SLOPE_CSV  # noqa: E402

EPS = 1e-6
SIZES = [9, 13]
# EMA momentum for the --gauge rating difficulty-whitening buffers (BatchNorm-style: train uses
# batch stats, eval uses the EMA — so the score-time gauge matches the one θ was trained in).
GAUGE_MOM = 0.01
# The bot-anchor ladder size --anchor-weight is calibrated at (20k, 9d). With --anchor-fixed-mass
# the per-rung weight is rescaled by this/len(ladder) so a DENSER ladder sharpens slope
# identification (more rungs ⇒ more points on each problem's success curve) without growing the
# total bot mass and drowning the human attempts.
ANCHOR_REF_RUNGS = 2
# logit of the serving/anchoring success rate (80%): rating = the ability θ at which
# P(clear) = CALIBRATION_P. Single source of truth shared with the PWA + map_rating_to_rank.
CLEAR_LOGIT = float(np.log(CALIBRATION_P / (1.0 - CALIBRATION_P)))
# Effective support of the success curve in gap (= slope·(ability−rating)) logits: past ±ZCLIP the
# sigmoid is within ~0.7% of its asymptote, so a larger gap carries no curve information (and extreme
# gaps are broken-ability artifacts). Bounds the gap-DRO bin range — a structural constant, not tuned.
ZCLIP = 5.0


def dyadic_tail_bins(vals, depth):
    """Bin `vals` so cells are mass-balanced (quantile edges, never near-empty) BUT resolution
    concentrates at the two EXTREMES: start from a median split (2 bins), then each level splits the
    current outermost bin on each side at its own median. The depth-d extreme cell holds the
    most-extreme 2^-(d+1) of the mass — shrinking but always a real fraction. depth=0 → 2 bins;
    n_bins = 2·depth+2. Favors the tails (where the structural biases live) without the empty-cell
    blowup of equal-width bins or the no-extreme-resolution of plain quantiles. Returns
    (bin_index_per_value, n_bins)."""
    qpts = np.array(sorted({0.0, 0.5, 1.0}
                           | {0.5 ** (k + 1) for k in range(1, depth + 1)}
                           | {1.0 - 0.5 ** (k + 1) for k in range(1, depth + 1)}))
    edges = np.quantile(vals, qpts); edges[-1] += 1e-6
    nb = len(edges) - 1
    return np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, nb - 1), nb


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


# The survey-ρ / time-adjusted-ρ metrics moved to auxiliary.rating_dev.validate_ratings (the single home of
# rating validation). This file trains + writes; run_validation (the bench) judges the output.


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
                 move_dropout=0.0, move_dim=0, cont_dropout=0.0, cont_dim=0,
                 separate_trunks=False, gauge="theta",
                 theta_posterior=False, act_dropout_mode="block"):
        super().__init__()
        # --theta-posterior: per-profile posterior sd + learned population prior sd (see the flag
        # help). theta itself doubles as the posterior MEAN m_u; sampling happens in forward().
        self.theta_posterior = bool(theta_posterior)
        if theta_posterior:
            self.theta_logsd = nn.Embedding(n_users, 1)
            nn.init.constant_(self.theta_logsd.weight, float(np.log(0.5)))
            self.pop_logsd = nn.Parameter(torch.tensor(float(np.log(1.2))))
        # --gauge: which population pins the 4PL scale (the θ↔b↔a scale freedom needs exactly one
        # pin). "theta" = the shipped hard unit-Gaussian gauge on ability. "rating" = whiten the
        # DIFFICULTY head b to mean 0 / std 1 instead and leave θ free — BatchNorm-style (train
        # batches use their own attempt-weighted stats and update an EMA; eval uses the EMA, so
        # scored ratings live in the same gauge θ was trained against).
        self.gauge = gauge
        self.register_buffer("b_mu", torch.zeros(()))
        self.register_buffer("b_sd", torch.ones(()))
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
        # --act-dropout-mode: the shipped "block" mode removes the fingerprint AS A UNIT, so the net
        # must stay predictive with no activation signal at all and base rates have to route through
        # the transferable bot-winrate columns. "column" instead drops each PC independently — at
        # p=0.3 over 128 PCs essentially every row keeps ~90 of them, so the problem stays
        # identifiable and the constraint degrades to input jitter (a ridge-like variance penalty)
        # rather than a routing constraint. Which of the two effects the transfer gain comes from is
        # the measurement this flag exists for.
        self.act_dropout, self.act_dim = float(act_dropout), int(act_dim)
        self.act_dropout_mode = str(act_dropout_mode)
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
        # Continuation block (--cont-features), at cols
        # [act_dim+bw_dim+move_dim : +cont_dim], i.e. straight after the move block and still
        # before the phase columns and the trailing size indicator. Production masks all four
        # wide blocks as ONE region (pipeline.train_rating's INPUT_DROPOUT); the per-block flags
        # are kept here because ablating a single block is what this trainer exists for.
        self.cont_dropout, self.cont_dim = float(cont_dropout), int(cont_dim)
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
        # guess rate) and s is ONE global
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
        # --staged: coordinate-descent over the per-problem 4PL heads (rating b → slope a → floor c).
        # The backbone still produces all three from the shared trunk, but at any stage only ONE head
        # is FREE; the others are pinned in (b, a, c) output space — to a global CONSTANT before that
        # head has been trained (slope const_a, floor const_c), and to its STORED PER-PROBLEM value
        # once it has (pin_b/pin_a/pin_c, snapshotted over the whole corpus at the stage that trained
        # it). So when the trunk is freed to fit floor, rating and slope hold the per-problem values
        # they were trained to — not a global mean. Inert unless self.staged (set by the driver).
        self.staged = False
        # head indices: 0=rating/b, 1=slope/a, 2=floor/c. A stage frees a SET of heads; already-
        # trained heads are frozen (held at pin_b/pin_a/pin_c by the loss penalty), and not-yet-
        # trained heads sit at their global constant (const_a / const_c) until their stage.
        self.trained_heads = set()  # heads frozen at their stored per-problem values (penalty-held)
        self.free_heads = set()     # heads trainable this stage (live, data BCE)
        self.const_a = 1.0          # global slope before the slope stage
        self.const_c = 1.0 / 3.0    # global floor before the floor stage (0.333)
        # --global-stage1: instead of HOLDING a not-yet-individualized head at a fixed constant,
        # hold it at a LEARNABLE GLOBAL scalar that trains in phase 1 (jointly with the per-problem
        # rating + the global ceiling). This auto-finds the self-consistent global slope/floor that
        # the fixed-point search did by hand; phase 2 then frees the per-problem heads (which start
        # from the net's own output, not the global). Gated by --global-stage1.
        self.global_stage1 = False
        self.gslope_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(self.const_a)))))
        self.gfloor_raw = nn.Parameter(torch.tensor(float(np.log(self.const_c / (1 - self.const_c)))))
        if n_problems > 0:
            self.register_buffer("pin_b", torch.zeros(n_problems))
            self.register_buffer("pin_a", torch.ones(n_problems))
            self.register_buffer("pin_c", torch.full((n_problems,), 1.0 / 3.0))
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
        # docs/methodology.md). The discount absorbs that mismatch per bot:
        # bots keep teaching difficulty location and RELATIVE shape across problems;
        # humans alone pin the absolute steepness scale. Slope-side analogue of legacy's
        # human-recalibrated (0.70·) bot floor, but learned.
        self.n_human = n_users - n_anchor
        self.slope_discount = None
        if slope_discount and n_anchor > 0:
            self.slope_discount = nn.Parameter(
                torch.full((n_anchor,), float(np.log(np.expm1(1.0)))))  # softplus⁻¹(1)
        def _trunk(out_dim):
            layers, d = [], fdim
            for h in hidden:                                   # decreasing layers, ReLU between
                layers += [nn.Linear(d, h), nn.ReLU()]
                if dropout > 0:
                    layers += [nn.Dropout(dropout)]
                d = h
            layers += [nn.Linear(d, out_dim)]
            return nn.Sequential(*layers)
        # --separate-trunks (staged only): an INDEPENDENT trunk per output (b, a, c) instead of one
        # shared trunk → Linear(·,3). With separate trunks, freezing a head = freezing ITS trunk's
        # params, so its output is maintained EXACTLY (zero drift) while the other heads train — the
        # shared-trunk leak (a moving trunk drags the frozen readout) is gone by construction.
        self.separate = bool(separate_trunks)
        self.net = self.net_b = self.net_a = self.net_c = None
        if per_problem:
            pass                                               # no feature net at all
        elif self.separate:
            self.net_b, self.net_a, self.net_c = _trunk(1), _trunk(1), _trunk(1)
        else:
            self.net = _trunk(3)                               # → (difficulty b, slope a, floor c)
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
                # --act-dropout-mode: "block" draws ONE keep bit per row and broadcasts it over the
                # whole activation block (the shipped behavior — the block is present or absent);
                # "column" draws an independent bit per PC, like the bw/move blocks.
                ncol = 1 if self.act_dropout_mode == "block" else self.act_dim
                keep = (torch.rand(len(feat), ncol, device=feat.device)
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
            if self.training and self.cont_dropout > 0 and self.cont_dim > 0:
                d0 = self.act_dim + self.bw_dim + self.move_dim
                keep = (torch.rand(len(feat), self.cont_dim, device=feat.device)
                        >= self.cont_dropout).float() / (1.0 - self.cont_dropout)
                feat = torch.cat([feat[:, :d0], feat[:, d0:d0 + self.cont_dim] * keep,
                                  feat[:, d0 + self.cont_dim:]], dim=1)
            if self.separate:                                  # one independent trunk per output
                o = torch.cat([self.net_b(feat), self.net_a(feat), self.net_c(feat)], dim=1)
            else:
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
        resid = None
        if use_resid and self.floor_resid is not None:
            if pidx is None:
                pidx = torch.arange(len(o), device=o.device)
            resid = self.floor_resid(pidx) * self.resid_mask
            # slope/floor residual columns act in raw head space (unwhitened heads); the DIFFICULTY
            # column is deferred to AFTER the rating-gauge whitening — a pre-whitening additive δ is
            # scale-free under the gauge (the trunk shrinks its raw spread, the whitening divides by
            # it, and the ridge is evaded by amplification: measured b_sd collapse 0.22→0.07 with
            # sweep weight, the 2026-07-24 ho_ebdd degeneracy), so the ridge must bind in the
            # whitened/shipped units the residual actually moves.
            o = o + torch.cat([torch.zeros_like(resid[:, :1]), resid[:, 1:]], dim=1)
        b = o[:, 0]
        if self.gauge == "rating":
            # hard-whiten difficulty instead of θ (see __init__). Differentiable through the batch
            # stats, like the θ gauge; EMA buffers carry the training gauge into eval scoring.
            if self.training:
                mu, sd = b.mean(), b.std() + EPS
                with torch.no_grad():
                    self.b_mu.mul_(1.0 - GAUGE_MOM).add_(GAUGE_MOM * mu)
                    self.b_sd.mul_(1.0 - GAUGE_MOM).add_(GAUGE_MOM * sd)
            else:
                mu, sd = self.b_mu, self.b_sd
            b = (b - mu) / sd
            if resid is not None:
                b = b + resid[:, 0]
        elif resid is not None:
            b = b + resid[:, 0]
        a = torch.nn.functional.softplus(o[:, 1]).clamp(min=self.slope_floor)  # lower bound only
        lam = self.lapse()
        if detach_backbone:
            lam = lam.detach()
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=self.floor_cap)
        c = cap * torch.sigmoid(o[:, 2])
        if self.staged:
            # NOT-YET-TRAINED heads (neither frozen nor free this stage) → global constant, kept out
            # of the prediction until their stage. Frozen (trained) and free heads stay LIVE/backbone-
            # produced; the frozen ones are held at their stored per-problem values by a consistency
            # penalty in the LOSS (not here), so the final model is ONE coherent forward pass.
            not_yet = {1, 2} - self.trained_heads - self.free_heads
            # held at a LEARNABLE GLOBAL scalar (--global-stage1, trains in phase 1) or a fixed const
            g_a = (torch.nn.functional.softplus(self.gslope_raw) if self.global_stage1
                   else b.new_tensor(self.const_a))
            g_c = (torch.sigmoid(self.gfloor_raw) if self.global_stage1
                   else b.new_tensor(self.const_c))
            if 1 in not_yet:                                          # slope not yet individualized
                a = g_a.expand_as(a)
            if 2 in not_yet:                                          # floor not yet individualized
                c = g_c.expand_as(c)
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
        """Per-user ability. Gauge "theta": hard-whitened to mean 0 / std 1 each call (unit-Gaussian
        gauge). Gauge "rating": raw embedding — the scale pin lives on the difficulty head instead."""
        if self.gauge == "rating":
            return self.theta(u).squeeze(-1)
        w = self.theta.weight
        return ((self.theta(u) - w.mean()) / (w.std() + EPS)).squeeze(-1)

    def forward(self, u, feat, pidx=None, use_resid=True, detach_backbone=False, pin=None):
        b, a, c = self.params(feat, pidx, use_resid=use_resid, detach_backbone=detach_backbone,
                              pin=pin)
        th = self.ability(u)
        if self.theta_posterior and self.training:
            # reparameterized draw from q(θ_u) — the ELBO's sampled-θ likelihood term
            th = th + torch.exp(self.theta_logsd(u)).squeeze(-1) * torch.randn_like(th)
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


# Model-defining flags that reproduce the shipped pipeline.train_rating output (its hardcoded
# module constants, mapped to this trainer's argparse dests). --production-flags installs these as
# DEFAULTS, so any explicitly-passed flag still wins (e.g. `--production-flags --dropout 0.5`).
# Experiment-control flags (seed, holdout-problem-frac, out-dir, validate, eval-every, log-every)
# are deliberately NOT included — layer them on top. Keep in sync with pipeline/train_rating.py.
PRODUCTION_FLAGS = {
    "staged": False,                     # SINGLE joint pass (replaced staged descent 2026-07-24)
    "gauge": "rating",                   # scale gauge on the difficulty head; θ free
    "shallow_slope_penalty": 1e-4,       # hard-tail (flat-curve) divergence barrier
    "high_floor_penalty": 0.01,          # easy-tail (floor→0.8) divergence barrier
    "free_lapse": True,                  # ceiling LEARNED in the joint pass
    "epochs": 200,
    "features": "bw_act",
    "move_features": True,               # bw_act ⊕ candidate-move-eval block
    "cont_features": False,              # continuation block: measured-rejected, see --cont-features
    "phase_features": True,              # ⊕ game-phase raw components (feature-parity policy)
    "pca_dim": 128,
    "diff_hidden": [128, 64],
    "dropout": 0.3, "act_dropout": 0.3, "bw_dropout": 0.3, "move_dropout": 0.3,
    "cont_dropout": 0.3,                 # rate the block takes when an experiment turns it on
    "act_dropout_mode": "column",        # per-PC, not whole-block (2026-07-26; see the flag help)
    "anchors": False,                    # no bot-anchor examinees
    "hier_slope": False, "mech_floor": False, "floor_residual": False,  # no M0b machinery
    # Pinned even though the argparse default already matches production: these are the knobs an
    # experiment is most likely to flip, and PRODUCTION_FLAGS must not depend on a default staying
    # put. (dro_skill in particular: the module docstring above spent a long time claiming DRO was
    # on by default when the default was False — pin the value, don't trust the prose.)
    "dro_skill": False,                  # plain BCE/ERM — production runs no DRO objective
    "weight_decay": 0.0,                 # production uses plain Adam, no decoupled decay
    "lr_schedule": "constant",           # production has no warmup/anneal
    "lr": 0.01, "batch": 65536,
    "min": 5, "min_profile": 8,
    "winsor": 1.0, "slope_floor": 0.01,
    "lapse_cap": 0.05, "lapse_init": 0.02, "floor_cap": 1.0,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--production-flags", action="store_true",
                    help="Set every model-defining flag to the shipped pipeline.train_rating "
                         "production config (single joint pass, --gauge rating, the two divergence "
                         "barriers --shallow-slope-penalty 1e-4 / --high-floor-penalty 0.01, "
                         "bw_act+move+phase features, uniform 0.3 dropout at COLUMN granularity "
                         "on every feature block, plain-BCE objective, no anchors, learned "
                         "ceiling). Installed as DEFAULTS so explicit flags still override; "
                         "experiment-control flags (--seed, --holdout-problem-frac, --out-dir, "
                         "...) are NOT touched. Typical: `--production-flags "
                         "--holdout-problem-frac 0.1 --seed 1 --out-dir /tmp/rv2_ho/s1`.")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--synthetic-outcomes", default=None,
                    help="Parametric-bootstrap hook: CSV with columns user_id,problem,success; "
                         "replaces every first-time attempt's real outcome by the synthetic one "
                         "(merged on the unique (user_id, problem) key) before profile explosion. "
                         "Rows absent from the CSV keep their real outcome. Everything else — row "
                         "structure, timestamps, serving pattern, features — stays identical, so "
                         "a refit on data generated from a known static-θ 4PL measures the "
                         "machinery's own bias (slope attenuation, tail behavior) with truth known.")
    ap.add_argument("--pca-dim", type=int, default=128, help="Joint activation PCs.")
    ap.add_argument("--features", choices=["bw_act", "act", "bw", "move"], default="bw_act",
                    help="Feature ablation: bw_act = activations ⊕ bot-winrate columns (the "
                         "validated production set), act = activations only (tests whether the "
                         "bot ladder columns are a convenient-but-flawed SHORTCUT for the shape "
                         "heads — bot per-problem signals are ~50%% overdispersed vs humans, and "
                         "a free floor head loses to f(rating) on unseen problems despite having "
                         "the bot floor available), bw = bot columns only (attribution "
                         "reference), move = candidate-move-eval columns only (implies "
                         "--move-features; the all-combinations singleton). The scorable problem "
                         "set stays the SAME act∩bw intersection in all modes — only the fed "
                         "columns change.")
    ap.add_argument("--move-features", action=argparse.BooleanOptionalAction, default=False,
                    help="Append the candidate-move evaluation block to the --features set: for "
                         "each problem the winning move's win-prob+score plus those of its 9 best "
                         "alternatives, oriented to the side to move (logit win-prob + clipped "
                         "score per move, 20 cols). A sharp gap to the alternatives = an obvious "
                         "best move (easy); many near-solution rivals = ambiguous (hard) — a "
                         "difficulty signal orthogonal to the activations and bot ladder. Cache "
                         "it first with `python -m pipeline.cache_rating_features`. The block sits "
                         "between the bot-winrate columns and the size indicator, so --act-dropout "
                         "/ --bw-dropout are unaffected. The scorable problem set is unchanged "
                         "(problems missing a move-feature row are mean-imputed).")
    ap.add_argument("--alt-act-features", action=argparse.BooleanOptionalAction, default=False,
                    help="Add the BEST-ALTERNATIVE cell's trunk activations, projected through "
                         "the SAME joint-PCA basis as the correct-move block (no refit, so the "
                         "runner-up lands in the correct move's coordinate system and the trunk "
                         "can read the two together). Sits immediately after the activation "
                         "block and is covered by --act-dropout as one activation region, since "
                         "it is the same kind of near-unique per-problem code. No separate "
                         "contrast feature is needed: [correct, alt] and [correct, alt-correct] "
                         "span the same functions through a linear layer. Needs "
                         "utils.rating.features.build_alt_vec_cache run for each size.")
    ap.add_argument("--cont-features", action=argparse.BooleanOptionalAction, default=False,
                    help="Add the CONTINUATION block (pipeline.cache_rating_features' `cont` "
                         "array): the principal line after the answer, CONT_PLIES deep and "
                         "CONT_MOVES wide, plus the opponent's other tries at its first node "
                         "each with the solver's best answers — all raw (logit win-prob, score) "
                         "oriented to the SOLVER. Covers what the root block cannot: the "
                         "single-solution gate leaves exactly one winning first move with every "
                         "rival decisively lost, so the root block's alternative slots are "
                         "saturated at the win-prob clip by construction, while the branching a "
                         "solver actually has to read lives past move 1. Appended after the move "
                         "block, before phase.")
    ap.add_argument("--cont-dropout", type=float, default=0.0,
                    help="Elementwise dropout on the continuation columns (--cont-features), "
                         "same rate and rationale as --move-dropout: the block is wide enough "
                         "to fingerprint a problem and shares its SGF with the move block, so "
                         "unmasked it hands the net a second uncorrupted copy of a partly shared "
                         "signal. No-op without --cont-features. 0 = off.")
    ap.add_argument("--phase-features", action=argparse.BooleanOptionalAction, default=False,
                    help="Append the game-phase RAW-component block "
                         "[moves_before_position, main_line_len] "
                         "(utils.rating.features.load_joint_game_phase; z-scored, 2 cols) after "
                         "the move block. Motivated by the residual-MIRT r-head finding "
                         "(kappa +0.109 of predictable residual difficulty on a surface "
                         "including these; auxiliary.rating_dev.train_residual_axes). Not in the "
                         "production config; no dedicated dropout (2 smooth columns cannot "
                         "fingerprint problems).")
    ap.add_argument("--diff-hidden", type=int, nargs="*", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--act-dropout", type=float, default=0.0,
                    help="Probability of dropping the ENTIRE activation feature block per row "
                         "during training (scaled 1/(1−p); eval uses all features). Targets the "
                         "measured memorization channel (Findings F, S): the activation PCs are "
                         "near-unique per-problem fingerprints the net uses to memorize attempted "
                         "problems' base rates — which inflates by-row fit but pollutes the "
                         "feature prior that unseen problems receive. Block dropout forces "
                         "base-rate information through the transferable bot-winrate columns "
                         "(soft version of the bw-backbone/act-local split). No-op for "
                         "--features bw. 0 = off.")
    ap.add_argument("--act-dropout-mode", choices=["block", "column"], default="block",
                    help="Granularity of --act-dropout. 'block' (shipped) draws ONE keep bit per "
                         "row for the WHOLE activation block, so the per-problem fingerprint is "
                         "present or absent as a unit and the net must stay predictive without it "
                         "— base-rate information is forced through the transferable bot-winrate "
                         "columns. 'column' draws an independent bit per PC (the bw/move "
                         "convention): at p=0.3 over 128 PCs a row keeps ~90 of them, so the "
                         "problem remains identifiable and the regularizer degrades to input "
                         "jitter. Measures whether the transfer gain comes from the BLOCK "
                         "structure (routing) or merely from noise on the activation columns.")
    ap.add_argument("--bw-dropout", type=float, default=0.0,
                    help="Elementwise dropout on the bot-winrate feature columns during "
                         "training (scaled; eval uses all). Per-bot pass rates are ~50%% "
                         "overdispersed vs human difficulty, so this regularizes reliance on "
                         "any single rung's idiosyncratic value toward ladder consensus. "
                         "No-op for --features act. 0 = off.")
    ap.add_argument("--move-dropout", type=float, default=0.0,
                    help="Elementwise dropout on the candidate-move-eval columns (--move-features) "
                         "during training (scaled 1/(1−p); eval uses all). The winning-move + 9-alt "
                         "(winrate, score) block is largely redundant with the bot-winrate ladder "
                         "and activations, so unregularized it overfits attempted problems and "
                         "worsens unseen-problem transfer; column dropout forces the prior to treat "
                         "it as corroborating, not load-bearing. No-op without --move-features. "
                         "0 = off.")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="DECOUPLED (AdamW) weight decay on the feature trunk's Linear WEIGHT "
                         "matrices only — the alternative-regularizer probe: can weight shrinkage "
                         "replace the 0.3 dropout stack? Motivation: dropout is the measured cause "
                         "of the phase-2 slope-scale compression (s≈1.73), so a regularizer that "
                         "bought the same unseen-problem transfer with less scale compression "
                         "would shrink the global fudge factor. Deliberately EXCLUDES (a) the "
                         "theta embedding — decaying ability is the measured theta-compression "
                         "failure (~0.6x ability squash + rating-dependent miscalibration), and "
                         "under --gauge rating theta carries the free scale, so decaying it "
                         "silently redefines the gauge; (b) biases and the scalar lapse/global "
                         "heads — no memorization capacity, and decaying the lapse logit just "
                         "drags the ceiling toward its sigmoid midpoint; (c) the per-problem "
                         "residual/pp tables, which have their own explicit L2 knobs. Decoupled "
                         "(AdamW), not Adam's coupled L2, so the shrinkage rate is not divided by "
                         "the per-parameter Adam denominator. Joint-pass only (asserts against "
                         "--staged). 0 = off (plain Adam, byte-for-byte the previous path).")
    ap.add_argument("--head-pin-dropout", type=float, default=0.0,
                    help="Per-batch head-level dropout (user compensation-learning test): with "
                         "this probability, ONE head from --head-pin-heads is pinned to an EMA "
                         "of its corpus-median raw output for the batch's backbone loss — the "
                         "remaining heads must fit the rows alone. Prediction on record: acts "
                         "like shrinkage (robustness-to-pinning is maximized by not using the "
                         "head per problem), so per-problem floor spread/meaning shrink toward "
                         "the smooth-family equilibrium rather than assignment improving. 0 = off.")
    ap.add_argument("--head-pin-heads", default="slope,floor",
                    help="Comma subset of {diff,slope,floor} eligible for --head-pin-dropout.")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--staged", action="store_true",
                    help="Coordinate-descent fit of the per-problem 4PL heads instead of the joint "
                         "fit. Order rating→slope→floor: free one head (backbone-predicted, 0.3 "
                         "dropout), train --stage-epochs, then FREEZE it at its learned PER-PROBLEM "
                         "values (not a global mean) and free the next. Slope/floor start pinned to "
                         "--stage-slope-const / --stage-floor-const. Implies no mech-floor / no hier "
                         "/ no floor-residual (the heads are governed by the staging). Writes + "
                         "validates after EACH stage (out-dir/stage{1,2,3}_*) so each head's marginal "
                         "contribution is visible. Honors --act-dropout/--bw-dropout, --seed, "
                         "--holdout-problem-frac, --anchors.")
    ap.add_argument("--stage-epochs", type=int, default=100,
                    help="Epochs per coordinate-descent stage (--staged). Default 100 (the new "
                         "single-stage budget); 3 stages ⇒ 300 backbone-epochs total.")
    ap.add_argument("--stage-slope-const", type=float, default=1.0,
                    help="Global slope a the model is pinned to before the slope stage (--staged).")
    ap.add_argument("--stage-floor-const", type=float, default=1.0 / 3.0,
                    help="Global floor c the model is pinned to before the floor stage (--staged).")
    ap.add_argument("--lapse-init", type=float, default=0.02,
                    help="Initial global lapse (1−ceiling) — the value the ceiling is held at before "
                         "it is freed (the ceiling analogue of --stage-slope-const/--stage-floor-const "
                         "for the fixed-point search). Default 0.02.")
    ap.add_argument("--stage-pin-weight", type=float, default=1.0,
                    help="Weight on the consistency penalty that holds each FROZEN head's live output "
                         "at its stored per-problem value while a later head trains (--staged). The "
                         "frozen heads are detached from the data BCE, so this is their ONLY signal — "
                         "it forces the shared trunk to keep emitting the learned rating/slope as it "
                         "fits floor. Higher = stricter maintenance (less trunk capacity for the new "
                         "head). The per-stage drift readout shows how well it held. (With "
                         "--separate-trunks this is vacuous — freezing the head's own trunk maintains "
                         "it exactly, so the penalty term is ~0.)")
    ap.add_argument("--separate-trunks", action="store_true",
                    help="--staged variant: give EACH output (rating/slope/floor) its OWN independent "
                         "trunk instead of one shared trunk → Linear(·,3). Freezing a trained head "
                         "then just freezes its trunk's params, maintaining its output EXACTLY (zero "
                         "drift) while the other heads train — removes the shared-trunk leak. Triples "
                         "the trunk parameters.")
    ap.add_argument("--two-stage", action="store_true",
                    help="--staged variant: TWO phases instead of three — phase 1 trains rating "
                         "(slope/floor const), phase 2 frees slope AND floor TOGETHER (rating frozen). "
                         "Pair with --free-lapse to also train the ceiling in phase 2.")
    ap.add_argument("--global-stage1", action="store_true",
                    help="--staged: hold a not-yet-individualized head at a LEARNABLE GLOBAL scalar "
                         "(trained) instead of a fixed const. With phase 1 = rating(+ceiling) and "
                         "phase 2 = slope+floor, this trains a global slope + global floor + global "
                         "ceiling jointly with the per-problem rating in phase 1 (auto-finding the "
                         "self-consistent globals the fixed-point search did by hand), then phase 2 "
                         "individualizes slope/floor. Use with --stage-plan 'rating+ceiling,slope+floor'.")
    ap.add_argument("--stage-plan", default=None,
                    help="--staged: explicit phase ORDER, overriding the default/--two-stage plan. "
                         "Comma-separated phases, each a '+'-joined subset of "
                         "{rating,slope,floor,ceiling}, e.g. 'rating,slope,floor+ceiling' (default 3-"
                         "stage with a learned ceiling) or 'rating,floor+ceiling,slope' (floor+ceiling "
                         "before slope). A phase containing 'ceiling' trains the global lapse that "
                         "phase; the lapse is freed automatically if any phase trains it.")
    ap.add_argument("--free-lapse", action="store_true",
                    help="Train the global ceiling (lapse λ, upper asymptote 1−λ) instead of pinning "
                         "it at --lapse-fixed. Under --staged it is trained only in the phase(s) that "
                         "free a head other than rating (so phase 1 keeps it at init 0.02). Without "
                         "--staged it is free for the whole joint fit. (Expect it to drift low — the "
                         "asymptote-free-fit collapse; this flag is to MEASURE that.)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                    help="Per-epoch LR schedule for the MAIN fit (the detached δ group rides the "
                         "same multiplicative factor, preserving its lr ratio). constant = the "
                         "historical flat Adam lr (default, behavior-preserving). cosine = linear "
                         "warmup for --warmup-epochs from lr·--lr-warmup-init-frac up to lr, then "
                         "cosine anneal to 0 over the remaining epochs. The frozen residual pass "
                         "(--floor-resid-mode frozen) is unaffected.")
    ap.add_argument("--warmup-epochs", type=int, default=10,
                    help="Linear-warmup length (epochs) for --lr-schedule cosine. Stabilizes the "
                         "early high-lr Adam steps before the cosine decay (the LR was the DRO "
                         "instability knob — see docs); 0 = straight into cosine.")
    ap.add_argument("--lr-warmup-init-frac", type=float, default=0.1,
                    help="Warmup starts at lr·this fraction (cosine schedule).")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Diagnostic (holdout mode only): every N epochs print held-out-PROBLEM "
                         "logloss (the unseen-problem backbone-transfer metric) so convergence vs "
                         "epoch is visible in ONE run — used to pick a reasonable --epochs / cosine "
                         "horizon. 0 = off (no per-epoch eval cost).")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--min", type=int, default=5,
                    help="Min first-time correct AND incorrect to keep a user.")
    ap.add_argument("--min-profile", type=int, default=8)
    ap.add_argument("--winsor", type=float, default=1.0,
                    help="Winsorize final ratings to [p, 100-p] percentile (legacy robustness for "
                         "the low-slope b + logit/a tail on the unattempted extrapolation).")
    ap.add_argument("--slope-floor", type=float, default=0.01,
                    help="Lower bound on the per-problem slope a_p in the MODEL (softplus floor). "
                         "Under the default hier-slope shrinkage this is a numerical guard, not a "
                         "clamp — the hard 0.2 clamp era (pileup at the rail, U-shaped by rating "
                         "decile) is obsolete (Findings D/K); 0.01 is safe whenever any sound "
                         "slope constraint is active, and only a fully FREE slope head blows the "
                         "rating tail through it (Finding UU).")
    ap.add_argument("--holdout-problem-frac", type=float, default=0.0,
                    help="If >0, hold out this fraction of ATTEMPTED problems (and their anchor "
                         "rows) from training (via auxiliary.rating_dev.holdout_set, seed-keyed, hash-printed) "
                         "so the feature prior's generalization to those UNSEEN problems can be "
                         "measured. Writes the (subset-trained) CSVs to --out-dir (refuses the "
                         "production ratings/); validate with auxiliary.rating_dev.validate_ratings or "
                         "--validate.")
    ap.add_argument("--holdout-user-frac", type=float, default=0.0,
                    help="If >0, hold out this fraction of USERS (whole accounts, via "
                         "auxiliary.rating_dev.holdout_set, seed-keyed, hash-printed). Their rows "
                         "leave training entirely, so the fitted parameters never saw them — which "
                         "is what makes a WALK-FORWARD on those users leak-free. "
                         "utils.rating.slope_calibration's walk-forward scores rows whose problems' "
                         "parameters were fitted ON those same rows; that rewards confidence and "
                         "biases the slope scale upward.")
    ap.add_argument("--holdout-row-frac", type=float, default=0.0,
                    help="If >0, hold out this fraction of ATTEMPTS, keyed on (user, problem) which "
                         "is unique for first-time attempts. This is the CLIENT's condition: a "
                         "held-out problem keeps its other attempts in training, so its parameters "
                         "are informed by other people exactly as in deployment, and only this "
                         "user's next attempt is unseen. Between the problem holdout (parameters "
                         "from the feature prior alone) and a time-ordered walk-forward (parameters "
                         "fitted on the scored rows), this is the one that matches.")
    ap.add_argument("--validate", action="store_true",
                    help="After writing, run auxiliary.rating_dev.validate_ratings.run_validation on --out-dir "
                         "for the same (seed, held-out frac) — held-out logloss + Δskill flatness + "
                         "survey-ρ, hash-confirmed — so you needn't chain the bench by hand.")
    ap.add_argument("--mixed-only", action=argparse.BooleanOptionalAction, default=False,
                    help="Train only on human rows whose problem has BOTH successes and failures "
                         "and whose user does too (iterated to a fixed point; eval rows are not "
                         "filtered). Conditional-MLE-style drop of perfect/zero score patterns, "
                         "whose monotone likelihood exerts unbounded outward pull on difficulty — "
                         "tests whether the anchors' Finding-OO location job is only needed "
                         "because of one-sided rows.")
    ap.add_argument("--holdout-keep-anchors", action=argparse.BooleanOptionalAction, default=False,
                    help="Keep the BOT anchor rows for held-out problems (only their HUMAN rows "
                         "are excluded). Changes what metric 2 measures: bot→human per-problem "
                         "transfer (how well bot-fitted curves predict unseen humans) instead of "
                         "pure feature-prior transfer. The right mode for --per-problem, where "
                         "the default exclusion leaves held-out params at init.")
    ap.add_argument("--anchors", action=argparse.BooleanOptionalAction, default=True,
                    help="Add humanv0 bot 'users' (--anchor-ranks) as extra examinees with learned "
                         "ability + soft-target responses on every problem (coverage prior). On by "
                         "default; --no-anchors trains on human attempts only.")
    ap.add_argument("--anchor-ranks", default="20k,9d",
                    help="comma-separated anchor tokens (default 20k,9d). A token is a humanv0 "
                         "rank ('9d') or a full attempts/<size>/ file stem (any benchmarked bot, "
                         "e.g. 'kata1-b18c384nbt-s9996604416-d4316597426_t0.02' for the general "
                         "KataGo engine, or 'b18c384nbt-humanv0_20k_t10' for a near-random "
                         "easy-tail rung).")
    ap.add_argument("--anchor-weight", type=float, default=0.2,
                    help="per-row weight on bot-anchor responses (human rows = 1.0), calibrated at "
                         "the 2-rung (20k,9d) ladder. Low = minimal prior: humans own the shape "
                         "where attested, bots fill the empty regions.")
    ap.add_argument("--anchor-slope-discount", action=argparse.BooleanOptionalAction, default=False,
                    help="Give each anchor bot ONE learned steepness-discount scalar (init 1.0): "
                         "for bot rows P = c + (1−c−λ)·σ(discount·a·(θ−b)); human rows unchanged. "
                         "Absorbs 'bot curves are flatter than human curves' so a dense ladder "
                         "teaches difficulty location + relative shape without forcing bot-flat "
                         "slopes/floors into the shipped params.")
    ap.add_argument("--anchor-fixed-mass", action=argparse.BooleanOptionalAction, default=True,
                    help="Hold TOTAL bot mass fixed as the ladder grows: each rung is weighted "
                         "anchor_weight·2/len(ladder), so a denser --anchor-ranks adds slope-"
                         "identifying rungs without overwhelming the human attempts. --no-anchor-"
                         "fixed-mass keeps every rung at anchor_weight (mass grows with the ladder).")
    ap.add_argument("--lapse-cap", type=float, default=0.05,
                    help="Upper bound on the learned global ceiling λ.")
    ap.add_argument("--floor-cap", type=float, default=1.0,
                    help="Hard upper bound on the per-problem guessing floor c_p. Default 1.0 is a "
                         "no-op (the floor stays ceiling-coupled at 1−2λ≈0.97). Set 0.5 to forbid a "
                         "guessing rate above 'pick one of two moves' — caps the high easy-end floors "
                         "(0.9+) the unconstrained 4PL otherwise learns.")
    ap.add_argument("--l2-floor", type=float, default=0.0,
                    help="Outlier penalty on the per-problem floor: λ·Var(c_p) over all problems "
                         "(= L2 of each floor toward the corpus mean). Quadratic, so it pulls in the "
                         "collapsed-to-~0 outliers (the floor↔slope-ridge degeneracy in the hard "
                         "tail) without a per-problem target. GLOBAL only — does not import the bot "
                         "floor's over-extreme tails. 0 = off.")
    ap.add_argument("--slope-dist-prior", type=float, default=0.0,
                    help="Weight on a DISTRIBUTION prior over the corpus slope spectrum: sort all "
                         "problems' UNCLAMPED log-slopes and L2-match them to the quantiles of a "
                         "target normal (--slope-dist-mu/sigma, i.e. log-normal slopes). Unlike "
                         "--l2-slope (variance shrinkage to the mean) this pins location, spread "
                         "AND shape — it penalizes both the clamp pileup and the runaway 7–12 "
                         "slopes, while leaving WHICH problem gets which slope entirely to the "
                         "data. Acts on pre-clamp slopes so the gradient survives at the clamp. "
                         "0 = off.")
    ap.add_argument("--slope-dist-mu", type=float, default=0.50,
                    help="Target log-slope mean (default from the validated legacy artifact's "
                         "slope distribution: median slope ≈ 1.65).")
    ap.add_argument("--slope-dist-sigma", type=float, default=0.45,
                    help="Target log-slope std (legacy: 0.45; the live 4PL artifact shows 1.27 — "
                         "i.e. ~3x over-dispersed, clamp pileup + runaway tail).")
    ap.add_argument("--hier-slope", action=argparse.BooleanOptionalAction, default=True,
                    help="Hierarchical empirical-Bayes shrinkage on the slope head: log a_p ~ "
                         "N(μ, σ²), two-pass with μ, σ FROZEN at the pass-1 fit's robust corpus "
                         "moments (median, IQR/1.349; gradient-learned σ and per-step moment "
                         "matching are both verified-degenerate — see RatingNet), weighted at "
                         "the Bayes-correct rate of one prior draw per problem vs one BCE term per "
                         "data row (scale = n_problems / total row mass — no tuned knob). Answer-"
                         "free alternative to --slope-dist-prior (which imports legacy's μ/σ): the "
                         "corpus chooses its own slope distribution; intended to replace the hard "
                         "--slope-floor clamp (run with --slope-floor 0.01).")
    ap.add_argument("--hier-slope-mu", type=float, default=0.8,
                    help="FIXED hier-slope target mean (log-slope). With --hier-slope-sigma, "
                         "replaces the frozen pass-1 robust moments with a constant target, so "
                         "the free pass-1 fit is SKIPPED entirely (one optimization pass). "
                         "Justified by Finding G/K: the shrinkage equilibrium is insensitive to "
                         "the pass-1 target's σ (1.20 → 0.034 spread), and pass-1 medians land "
                         "at 0.63–0.92 across configs — μ 0.8 / σ 0.8 reproduces the stack's "
                         "target within that band. Two ordinary hyperparameters in exchange for "
                         "a whole training pass.")
    ap.add_argument("--hier-slope-sigma", type=float, default=0.8,
                    help="FIXED hier-slope target std (see --hier-slope-mu). The default 0.8/0.8 "
                         "is a CACHED pass-1 calibrated on the anchored bw_act regime and is "
                         "regime-SENSITIVE (Finding DD): re-derive it whenever the anchor set or "
                         "feature set changes by passing a NEGATIVE value (e.g. "
                         "--hier-slope-mu -1 --hier-slope-sigma -1), which restores the two-pass "
                         "mode (free pass 1 → frozen robust moments).")
    ap.add_argument("--lapse-fixed", type=float, default=0.02,
                    help="Freeze the global ceiling λ at this value instead of learning it. The "
                         "trained MLE under-estimates λ whenever floors are free (Finding Q; "
                         "plateau λ̂ ≈ 0.0225 vs trained 0.013 on the final stack) — fixing it at "
                         "the externally calibrated plateau value folds the posthoc λ rescoring "
                         "stage into the fit (ratings come out consistent in one pass).")
    ap.add_argument("--hier-floor", action=argparse.BooleanOptionalAction, default=False,
                    help="Same hierarchical prior on the floor head's raw logit (pre-sigmoid). "
                         "Targets the unidentified easy-end floors (no examinee sits below an easy "
                         "problem's rating, so c_p is free there and inflates to 0.9+): they shrink "
                         "to the learned corpus floor distribution instead.")
    ap.add_argument("--per-problem", action=argparse.BooleanOptionalAction, default=False,
                    help="Unique FREE 3PL per tsumego — no feature net (the maximal-local "
                         "reference model; the bot ladder IS the model, humans correct it where "
                         "attested). Run with --anchor-ranks all and --lapse-human-only. "
                         "Held-out-problem LL is MEANINGLESS under this flag (no features to "
                         "transfer through). Incompatible with the head priors/--global-slope/"
                         "--floor-residual.")
    ap.add_argument("--lapse-human-only", action=argparse.BooleanOptionalAction, default=False,
                    help="Apply the global ceiling λ to HUMAN rows only. λ models human slips; "
                         "a slip-free bot examinee otherwise anchors it to zero (Finding T, "
                         "kata1 t0.02 → λ 0.0004).")
    ap.add_argument("--floor-residual", action=argparse.BooleanOptionalAction, default=True,
                    help="Per-problem additive logit residual δ_p on the floor head, ridge-shrunk "
                         "to 0 (--floor-resid-l2). The Finding-O architecture test: floors are "
                         "locally identifiable (~0.04 nats by-row) but not feature-predictable "
                         "beyond f(rating), so prior-from-features + attested residual should win "
                         "by-row AND held-out-problem simultaneously (held-out problems keep δ=0 "
                         "= the feature prior).")
    ap.add_argument("--floor-resid-l2", type=float, default=4.0,
                    help="Ridge weight on mean(δ_p²). 4.0 (prior σ≈0.23) is the chain14 ship "
                         "knee: matches floor-only by-row with the best calibration; looser "
                         "values buy by-row at a real rho/calibration tax (Finding W).")
    ap.add_argument("--resid-cols", default="diff,slope,floor",
                    help="Comma subset of {diff,slope,floor}: which per-problem corrections the "
                         "frozen residual pass fits (gate: --floor-residual). The per-problem "
                         "reference model showed ~0.06 nats of real local heterogeneity; "
                         "diff+floor on a frozen backbone extracts it without the geometry "
                         "damage (legacy's δ_p/η_p, rebuilt on the measured-best backbone).")
    ap.add_argument("--floor-resid-mode", choices=["frozen", "joint", "detached"], default="detached",
                    help="frozen: fit the model normally, then FREEZE net/θ/λ and fit "
                         "only δ_p — local corrections on a fixed backbone, so the global "
                         "geometry can't drift. joint: train δ_p with everything else — "
                         "verified-degenerate (floor⇄difficulty swap: floors→0.74 mean, ratings "
                         "compressed, survey rho 0.07; the model explains difficulty AS floor). "
                         "detached: ONE-STAGE gradient-isolated residuals — the backbone loss is "
                         "computed residual-free while δ_p trains in the same pass against a "
                         "detached copy of the backbone (+ its own ridge); same fixed point as "
                         "frozen, no second pass, and the joint-mode swap is impossible because "
                         "the backbone never sees δ.")
    ap.add_argument("--floor-resid-epochs", type=int, default=100,
                    help="Epochs for the frozen residual pass (δ is per-problem and converges "
                         "fast against a fixed backbone).")
    ap.add_argument("--hier-conditional", action=argparse.BooleanOptionalAction, default=False,
                    help="Make the hier shrinkage targets RATING-CONDITIONAL: shrink log-slope / "
                         "floor-logit toward smooth pass-1 conditional medians μ(rating) (20 "
                         "quantile bins, piecewise-linear) instead of a global μ. Conditional "
                         "means are identified by POOLED data even where per-problem deviations "
                         "are not, so the real rating-coupled structure (e.g. easy ⇒ high floor) "
                         "is preserved and only residual spread is shrunk — fixes the global-"
                         "target floor-crushing of the plain hier prior.")
    ap.add_argument("--cond-diversity", type=float, default=0.0,
                    help="Weight on the continuous conditional diversity+decorrelation objective "
                         "(TEST cell): maintain each head's pass-1 residual spread around "
                         "μ(rating) and penalize residual cross-correlation given rating. "
                         "Prediction on record: improves structure tables without improving "
                         "slope↔bot_steep / heavy-val calibration (manufactured assignment). "
                         "0 = off.")
    ap.add_argument("--global-slope", action=argparse.BooleanOptionalAction, default=False,
                    help="ONE learned discrimination a for the whole corpus (3PL+lapse reference "
                         "model; the net keeps per-problem difficulty + floor). An accidental "
                         "constant-slope fit had the best held-out-problem LL and honest-val "
                         "calibration of any healthy run — this makes that model explicit: the "
                         "ceiling on how much per-problem slope variation is worth shipping.")
    ap.add_argument("--global-slope-fixed", type=float, default=None,
                    help="PIN the corpus discrimination to this constant (e.g. 2.2 — the hier "
                         "equilibrium level) instead of learning it; deletes the hier machinery's "
                         "regime-sensitive cached target. η_p residuals still compose on top.")
    ap.add_argument("--mech-floor", action=argparse.BooleanOptionalAction, default=True,
                    help="MEASURED floor: c_p = s·bot_floor_p (min standard-temp humanv0 ladder "
                         "pass rate, the empirical guess rate) with s ONE global human-vs-bot "
                         "scalar (fixed at 0.70 by default; see --mech-floor-scale-fixed). "
                         "Replaces the learned floor head; closes the floor⇄difficulty swap by "
                         "construction. δ_floor residuals still compose on top where attempts "
                         "attest. --no-mech-floor restores the learned floor head.")
    ap.add_argument("--mech-floor-scale-fixed", type=float, default=0.70,
                    help="Freeze the --mech-floor scale s at this value — zero learned floor "
                         "parameters (Finding QQ: pinning at the measured 0.70 strictly improves "
                         "every config vs learning s). Pass a NEGATIVE value to LEARN s instead "
                         "(init 0.70).")
    ap.add_argument("--hier-weight", type=float, default=1.0,
                    help="Multiplier on the Bayes-correct hierarchical-prior scale (1.0 = exactly "
                         "one prior draw per problem). A robustness knob, not a tuning target.")
    ap.add_argument("--gap-balance", action=argparse.BooleanOptionalAction, default=False,
                    help="Two-pass regime-balanced likelihood: pass 1 trains plain mean BCE, then "
                         "each human row's gap z = a·(θ−rating) (success-curve logits, pass-1 "
                         "model) is binned EQUAL-WIDTH over ±ZCLIP and rows are reweighted by "
                         "inverse bin mass (capped, total human mass preserved). Makes the floor "
                         "(far-below), slope (transition) and ceiling (far-above) regimes count in "
                         "the gradient instead of being drowned by the near-rank bulk — the "
                         "deterministic, non-adversarial fix for the 'fitted curve globally too "
                         "flat' miscalibration (gap-DRO's dynamics lost to ERM; this keeps ERM and "
                         "just fixes the regime weighting). No target values injected.")
    ap.add_argument("--gap-balance-bins", type=int, default=10,
                    help="Equal-width gap bins over ±ZCLIP for --gap-balance.")
    ap.add_argument("--gap-balance-max", type=float, default=10.0,
                    help="Cap on the per-row inverse-mass upweight for --gap-balance (keeps a "
                         "near-empty extreme bin from dominating the gradient).")
    ap.add_argument("--gap-balance-power", type=float, default=1.0,
                    help="Exponent on the inverse-mass weight (w ∝ (1/bin mass)^p). 1.0 = full "
                         "regime balance; <1 softens it — full balance amplifies the outcome-"
                         "contaminated tail rows of sparse profiles (see the train-vs-val "
                         "calibration discrepancy) and over-sharpens the tails.")
    ap.add_argument("--l2-slope", type=float, default=0.0,
                    help="Optional companion shrinkage on slope spread: λ·Var(log a_p) over all "
                         "problems. May be unnecessary once the floor prior holds (a sound floor "
                         "absorbs the flat hard-tail performance the slope was floored to mimic). "
                         "0 = off.")
    ap.add_argument("--theta-quadrature", type=int, default=0,
                    help="TRUE marginal-ML over ability via Gauss-Hermite quadrature: each user's "
                         "likelihood is integrated over θ ~ N(0, σ_pop²) (σ_pop LEARNED) with this "
                         "many nodes — log ∫ Π_rows p(row|θ) dθ per USER, the integral INSIDE the "
                         "log. This is the correct de-attenuation (Neyman-Scott) fix the sampling "
                         "ELBO is not: expected-log-loss under sampled θ INJECTS position noise "
                         "and flattens slopes (measured: s 1.47→1.65), while the marginal keeps "
                         "sharp conditional curves viable because the mixture flattening happens "
                         "in the probability, not the parameters. Trains in user-grouped batches "
                         "(a profile's rows must share nodes); no per-profile θ parameters exist — "
                         "scoring writes each profile's posterior MEAN into the θ table so every "
                         "artifact keeps its shape. Composes with the divergence barriers; "
                         "requires --gauge rating, no anchors/DRO/gap/knn/staged/certainty "
                         "weighting. 21 is a good node count. 0 = off.")
    ap.add_argument("--mixup-alpha", type=float, default=0.0,
                    help="Mixup row synthesis: every training point is a convex combination of "
                         "two attempt rows — problem features, ability values, and the soft "
                         "target interpolated with one per-row Beta(α,α) weight (1.0 = uniform) "
                         "— never a raw row. Interpolation-consistency regularizer probed as a "
                         "less scale-compressing alternative/complement to dropout (the "
                         "divergence barriers still read real corpus features). Plain joint-fit "
                         "path only (no staged/DRO/θ-posterior/quadrature/residual/certainty).")
    ap.add_argument("--certainty-weight-k", type=float, default=0.0,
                    help="Naive errors-in-variables mitigation: weight each HUMAN row by its "
                         "profile's ability-certainty proxy n/(n+k) (n = the profile's row count), "
                         "so poorly-localized profiles vote less on the SHARED parameters — the "
                         "attenuation channel — while their own ability estimate is untouched (a "
                         "uniform per-user scaling leaves that user's own argmax invariant). The "
                         "smooth generalization of the MIN_ATTEMPTS cutoff; k = the half-weight "
                         "profile size. Watch for heavy-user composition bias (their curves are "
                         "flatter). 0 = off.")
    ap.add_argument("--theta-posterior", action=argparse.BooleanOptionalAction, default=False,
                    help="Variational per-profile ability posterior: q(θ_u) = N(m_u, s_u²) with a "
                         "LEARNED per-profile sd, trained by sampled-θ ELBO (reparameterized; one "
                         "draw per row — unbiased since the row expectation is linear) plus the KL "
                         "to a learned population prior N(0, σ_pop²) at the Bayes per-row scale. "
                         "Marginal-ML replacement for the point-θ joint MLE — the Neyman-Scott "
                         "attenuation fix: integrating θ noise during training should de-flatten "
                         "the fitted slopes (the errors-in-variables channel behind the posthoc "
                         "slope-scale s ≈ 1.3–1.6). Requires --gauge rating (the θ-whitening gauge "
                         "would fight the prior; the scale pin lives on b). Eval/scoring uses the "
                         "posterior mean m_u, so artifacts and validation are unchanged in form.")
    ap.add_argument("--high-floor-penalty", type=float, default=0.0,
                    help="Weight w on the easy-side divergence barrier — the floor mirror of "
                         "--shallow-slope-penalty: the shipped rating diverges to −∞ as the floor "
                         "approaches CALIBRATION_P (0.8), so penalize w·mean(softplus(o_c − "
                         "o_div)²) in the floor head's LOGIT space, o_div = the raw output where "
                         "c = 0.8 (center derived from the serve margin, not hand-set). One-sided "
                         "(exponentially vanishing below), smooth, and with live gradient even for "
                         "floors already past 0.8 — a q-space penalty would clamp-die exactly "
                         "there (the slope-rail lesson). Floor-head only (not gameable via slope "
                         "or λ; λ is detached in the center). 0 = off.")
    ap.add_argument("--shallow-slope-penalty", type=float, default=0.0,
                    help="Weight w on a one-sided anti-flat-curve penalty: w·mean(1/a_raw²) over "
                         "all problems, a_raw = the PRE-clamp slope (gradient alive at the rail). "
                         "1/a is the curve's transition width; under --gauge rating the corpus "
                         "difficulty spread is pinned to 1, so this penalizes transition regions "
                         "that dwarf the whole corpus scale — the exact term the 80%%-clear rating "
                         "diverges through. Deliberately floor/ceiling-free (not gameable via c or "
                         "λ — the --l2-rating lesson) and threshold-free (1/a² is ~10⁴ at the rail "
                         "vs ~0.7 at the median slope, so it self-targets shallow). 0 = off.")
    ap.add_argument("--l2-rating", type=float, default=0.0,
                    help="Weight on a direct penalty on the DERIVED rating magnitude: "
                         "w·mean(rating_p²) over all problems, rating_p = b + clear_logit(c, λ)/a "
                         "(the shipped 80%%-clear point). Only coherent WITH a scale gauge (else "
                         "global shrinkage is a free escape); under --gauge rating the b spread is "
                         "pinned, so the penalty can only act by pulling in the extreme tail — "
                         "chiefly lifting collapsed slopes out of the logit/a blowup. The heavy "
                         "tail makes the quadratic self-targeting (a p99.9≈150 problem outweighs a "
                         "bulk problem ~10⁴:1), so no explicit 'extreme' threshold is needed. "
                         "0 = off.")
    # --- continuous kNN inverse-density reweighting (--gap-balance's bin-free analogue) ---
    ap.add_argument("--knn-reweight", action=argparse.BooleanOptionalAction, default=False,
                    help="Continuous kNN inverse-density analogue of --gap-balance over a CHOICE of "
                         "the {rating, ability, gap} axes (--knn-axes) and their pairs. Each human "
                         "row is upweighted ∝ (k-th NN distance in the z-scored axis space)^beta, "
                         "capped + mass-preserving — reaching the sparse extremes coarse bins smear. "
                         "Pass-1 model defines the axes; intended standalone (ERM + these weights).")
    ap.add_argument("--knn-axes", default="rating+gap",
                    help="'+'-joined subset of {rating, ability, gap} for --knn-reweight.")
    ap.add_argument("--knn-k", type=int, default=50,
                    help="Neighbours for the kNN density estimate (--knn-reweight).")
    ap.add_argument("--knn-beta", type=float, default=1.0,
                    help="Upweight sparse rows ∝ (k-th NN distance)^beta (0 = uniform).")
    ap.add_argument("--knn-max", type=float, default=10.0,
                    help="Cap on the per-row kNN upweight (mirrors --gap-balance-max).")
    ap.add_argument("--knn-count-normalize", action=argparse.BooleanOptionalAction, default=False,
                    help="Sample-count-normalize the kNN density: estimate it over per-USER points "
                         "(one per user = their mean axis position) → user rarity W_u, then split "
                         "W_u across the user's rows (w_row = W_u/n_u), capping per user. Fixes the "
                         "ability-axis collapse where per-ATTEMPT density is dominated by who "
                         "attempted most (heavy users spike it → zeroed; rare-ability users blow "
                         "up). Exact for ability-only; per-user mean is approximate when "
                         "rating/gap (which vary within a user) are in --knn-axes.")
    # --- even-error group-DRO (ported from the legacy residual model; ON by default) ---
    ap.add_argument("--dro-skill", action=argparse.BooleanOptionalAction, default=False,
                    help="Spread held-out error evenly across the rating range with a base-rate-"
                         "centered group-DRO objective (replaces mean BCE on the HUMAN-attempt term; "
                         "bot anchors keep their flat coverage BCE). Two-pass: pass 1 trains with "
                         "mean BCE to get the rating, which defines the bins; pass 2 retrains with "
                         "DRO over those bins. OFF by default: in its first honest test ERM beat "
                         "DRO on all three objectives at fixed anchors (Round 1, the 4PL "
                         "investigation) — plain weighted BCE (ERM) is the production objective.")
    ap.add_argument("--dro-eta", type=float, default=0.2,
                    help="Group-DRO exponentiated-gradient step on per-bin base-centered loss. 0.2 "
                         "is the flatness↔cost knee (η≥~1.6 collapses q onto one bin).")
    ap.add_argument("--dro-bins", type=int, default=10,
                    help="Equal-count rating-quantile bins the DRO objective groups over "
                         "(--no-dro-dyadic).")
    ap.add_argument("--dro-2d", action=argparse.BooleanOptionalAction, default=True,
                    help="Group-DRO over (relative-gap z × absolute-rating) instead of rating alone "
                         "— resolves the far-below/above-rank gap regions, otherwise drowned by the "
                         "near-rank bulk.")
    ap.add_argument("--dro-z-bins", type=int, default=5,
                    help="Relative-gap z bins for --dro-2d (× the rating bins), non-dyadic modes.")
    ap.add_argument("--dro-dyadic", action=argparse.BooleanOptionalAction, default=True,
                    help="Bin BOTH DRO axes by dyadic-tail-by-mass (--dro-z-depth / --dro-r-depth): "
                         "mass-balanced cells with resolution concentrated at the extremes. "
                         "--no-dro-dyadic uses plain quantile/uniform with --dro-bins / --dro-z-bins.")
    ap.add_argument("--dro-z-depth", type=int, default=4,
                    help="Dyadic tail depth for the gap axis (n_bins = 2·depth+2).")
    ap.add_argument("--dro-r-depth", type=int, default=3,
                    help="Dyadic tail depth for the absolute-rating axis (n_bins = 2·depth+2).")
    ap.add_argument("--dro-z-uniform", action=argparse.BooleanOptionalAction, default=True,
                    help="Bin the gap axis equal-WIDTH over ±ZCLIP (non-dyadic) so --dro-z-bins is a "
                         "real lever on the far-from-rank extremes (quantile bins crowd the bulk).")
    ap.add_argument("--report-gap-bins", type=int, default=10,
                    help="Equal-width gap bins (gap = slope·(ability−rating), clipped to ±ZCLIP) "
                         "for the gap-binned CALIBRATION diagnostic: mean predicted vs realized "
                         "success per bin. The curve-SHAPE metric the three objectives barely see "
                         "(a globally-too-flat fit shows as +Δ far below rank / −Δ far above) — "
                         "printed on train rows always, and on the by-row val rows in eval mode. "
                         "0 disables.")
    ap.add_argument("--report-bins", type=int, default=10,
                    help="Equal-count rating-quantile bins for the final per-bin train-logloss "
                         "diagnostic (printed regardless of --dro-skill, so a DRO run and an ERM run "
                         "can be compared bin-for-bin). 0 disables the report.")
    ap.add_argument("--gauge", choices=["theta", "rating"], default="theta",
                    help="Which population pins the 4PL scale gauge. theta (default, shipped): "
                         "ability hard-whitened to N(0,1) each step. rating: the DIFFICULTY head b "
                         "is whitened to mean 0 / std 1 instead (BatchNorm-style batch stats + EMA "
                         "for eval) and θ is left FREE — tests whether a fully joint unconstrained "
                         "fit (all three per-problem heads + the ceiling at once) settles when the "
                         "gauge sits on the problem side instead of the population side.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    # --production-flags: install the shipped config as defaults BEFORE the real parse, so any
    # explicitly-passed flag still overrides it (argparse: CLI args beat set_defaults).
    if ap.parse_known_args()[0].production_flags:
        ap.set_defaults(**PRODUCTION_FLAGS)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The M0b production config is the DEFAULT (zero flags = ship config). Flags that replace a
    # default-on component auto-disable it with a note instead of asserting, so research
    # invocations like --global-slope-fixed or --per-problem keep working unchanged.
    if args.mech_floor_scale_fixed is not None and args.mech_floor_scale_fixed < 0:
        args.mech_floor_scale_fixed = None                      # negative sentinel: LEARN s
    if not args.mech_floor:
        args.mech_floor_scale_fixed = None
    if args.per_problem:
        for f in ("hier_slope", "mech_floor", "floor_residual"):
            if getattr(args, f):
                print(f"NOTE: --per-problem disables the default --{f.replace('_', '-')}")
                setattr(args, f, False)
        args.mech_floor_scale_fixed = None
        assert not (args.hier_floor or args.cond_diversity > 0
                    or args.global_slope or args.global_slope_fixed is not None
                    or args.slope_dist_prior > 0 or args.l2_floor > 0 or args.l2_slope > 0), \
            "--per-problem has no feature heads; the head priors/residual flags do not apply"
        if args.holdout_problem_frac > 0 and not args.holdout_keep_anchors:
            print("NOTE: --per-problem without --holdout-keep-anchors ⇒ held-out-problem "
                  "logloss is MEANINGLESS (held-out problems' free params are never trained); "
                  "use --holdout-keep-anchors to measure bot→human per-problem transfer")
    if (args.global_slope_fixed is not None or args.global_slope) and args.hier_slope:
        print("NOTE: --global-slope(-fixed) replaces the hier slope machinery "
              "(default --hier-slope disabled)")
        args.hier_slope = False
    assert not (args.global_slope and args.global_slope_fixed is not None), \
        "--global-slope and --global-slope-fixed are mutually exclusive"
    if args.mech_floor:
        assert not args.hier_floor, "--mech-floor replaces the floor head; no hier-floor prior"
    if args.theta_posterior:
        assert args.gauge == "rating", \
            "--theta-posterior needs --gauge rating (the θ-whitening gauge conflicts with the prior)"

    # --- features (joint-PCA correct-move activations ⊕ common-column bot winrate
    #     [⊕ candidate-move evaluations]) ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]                  # (size, tid); scorable iff both blocks
    key_idx = {key: i for i, key in enumerate(keys)}
    use_act = args.features in ("bw_act", "act")
    use_bw = args.features in ("bw_act", "bw")
    use_alt = args.alt_act_features and use_act
    alt, nalt = (load_joint_alt_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
                 if use_alt else ({}, 0))
    zero_alt = np.zeros(nalt, dtype=np.float32)
    use_mv = args.move_features or args.features == "move"   # --features move = move block only
    # The move block is appended AFTER the bot columns and BEFORE the size indicator, so the
    # [act | bw | move | size] layout keeps --act-dropout (cols :k) and --bw-dropout (cols
    # k:k+nb) valid. Move features don't gate the scorable set (it stays act∩bw — see --features);
    # a problem missing a row is mean-imputed (zero after the pooled z-score).
    mv, nmv = (load_joint_move_features(SIZES, tids_by_size) if use_mv else ({}, 0))
    zero_mv = np.zeros(nmv, dtype=np.float32)
    # --phase-features: appended AFTER move and BEFORE the size indicator, so every existing
    # block-dropout column range (act :k, bw, move — all indexed from the front) stays valid.
    # --cont-features: the continuation block, appended AFTER move and BEFORE phase, so every
    # block-dropout column range ahead of it (act :k, bw, move) stays valid.
    use_ct = args.cont_features
    ct, nct = (load_joint_continuation(SIZES, tids_by_size) if use_ct else ({}, 0))
    zero_ct = np.zeros(nct, dtype=np.float32)
    use_ph = args.phase_features
    ph, nph = (load_joint_game_phase(SIZES, tids_by_size) if use_ph else ({}, 0))
    zero_ph = np.zeros(nph, dtype=np.float32)
    # The alt block sits directly after the activations and is folded into act_dim, so it takes
    # the activation dropout as one region and every later block's column range shifts with it.
    act_block = (k if use_act else 0) + (nalt if use_alt else 0)
    fdim = act_block + (nb if use_bw else 0) + nmv + nct + nph + 1  # + size_is_13
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        blocks = ([act[key]] if use_act else [])
        if use_alt:
            blocks.append(alt.get(key, zero_alt))
        blocks += ([bw[key]] if use_bw else [])
        if use_mv:
            blocks.append(mv.get(key, zero_mv))
        if use_ct:
            blocks.append(ct.get(key, zero_ct))
        if use_ph:
            blocks.append(ph.get((key[0], key[1]), zero_ph))
        feat[i] = np.concatenate(blocks + [[1.0 if key[0] == 13 else 0.0]])
    n_mv_miss = sum(1 for key in keys if key not in mv) if use_mv else 0
    n_ct_miss = sum(1 for key in keys if key not in ct) if use_ct else 0
    n_ph_miss = sum(1 for key in keys if key not in ph) if use_ph else 0
    print(f"scorable problems {len(keys)} (9:{sum(s == 9 for s, _ in keys)} "
          f"13:{sum(s == 13 for s, _ in keys)}) | feat dim {fdim} "
          f"[{args.features}{'+alt' if use_alt else ''}{'+move' if use_mv else ''}"
          f"{'+cont' if use_ct else ''}"
          f"{'+phase' if use_ph else ''}] "
          f"(act {k if use_act else 0}{f' + alt {nalt}' if use_alt else ''} + bot "
          f"{nb if use_bw else 0}{f' + move {nmv} ({n_mv_miss} imputed)' if use_mv else ''}"
          f"{f' + cont {nct} ({n_ct_miss} imputed)' if use_ct else ''}"
          f"{f' + phase {nph} ({n_ph_miss} imputed)' if use_ph else ''} "
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

    if args.l2_floor > 0 or args.l2_slope > 0:
        print(f"param outlier penalty: λ_floor {args.l2_floor} (Var(c_p) → shrink outlier "
              f"floors to the mean) | λ_slope {args.l2_slope} (Var(log a_p))")
    slope_target_q = None
    if args.slope_dist_prior > 0:
        pp = (torch.arange(len(keys), dtype=torch.float32, device=device) + 0.5) / len(keys)
        slope_target_q = (args.slope_dist_mu
                          + args.slope_dist_sigma * float(np.sqrt(2.0)) * torch.erfinv(2 * pp - 1))
        print(f"slope distribution prior: weight {args.slope_dist_prior}, target log-slope ~ "
              f"N({args.slope_dist_mu}, {args.slope_dist_sigma}²) quantile-matched over "
              f"{len(keys)} problems (pre-clamp)")
    if args.hier_slope or args.hier_floor:
        heads = [h for h, on in (("log-slope", args.hier_slope),
                                 ("floor-logit", args.hier_floor)) if on]
        print(f"hierarchical EB shrinkage on {'+'.join(heads)}: two-pass, frozen pass-1 robust "
              f"moments, Bayes scale = n_problems/row-mass (× --hier-weight {args.hier_weight})")

    # --- HUMAN first-time attempts, pooled, shared profiles (no size in profile id) ---
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    if args.synthetic_outcomes:
        syn = pd.read_csv(args.synthetic_outcomes)
        df = df.merge(syn.rename(columns={"success": "success_syn"}),
                      on=["user_id", "problem"], how="left")
        n_rep = int(df["success_syn"].notna().sum())
        df["success"] = df["success_syn"].fillna(df["success"]).astype(int)
        df.drop(columns=["success_syn"], inplace=True)
        print(f"SYNTHETIC outcomes: replaced {n_rep}/{len(df)} attempts from "
              f"{args.synthetic_outcomes} | synthetic base rate {df['success'].mean():.4f}")
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
    #     has zero signal on them — its b/a/c there come purely from features). The set + its hash
    #     come from the SHARED generator (auxiliary.rating_dev.holdout_set) so a held-out training run and the
    #     validation bench (auxiliary.rating_dev.validate_ratings) provably assume the identical set; the hash
    #     is printed for cross-confirmation. Held-out runs WRITE CSVs (see the eval block) so the
    #     bench can score them. ---
    ho_mask_h = np.zeros(len(y), bool)
    ho_problems = set()
    ho_hash = None
    if args.holdout_problem_frac > 0:
        from auxiliary.rating_dev.holdout_set import held_out_keys
        ho_keys, ho_hash = held_out_keys(args.seed, args.holdout_problem_frac)
        ho_problems = set(key_idx[k] for k in ho_keys if k in key_idx)  # scorable subset, as indices
        ho_mask_h = np.isin(prob_idx, list(ho_problems))
        print(f"HELD-OUT eval: held-out-set hash {ho_hash} | {len(ho_keys)} attempted problems held "
              f"out ({len(ho_problems)} scorable, {int(ho_mask_h.sum())} human rows) — feature-prior "
              f"generalization test (confirm via auxiliary.rating_dev.validate_ratings --expect-hash {ho_hash})")

    # USER and ROW holdouts ride the SAME mask as the problem holdout, so they
    # are excluded from training (train_h = ~ho_mask_h) and scored as held-out
    # without any further plumbing. `ho_problems` is deliberately NOT extended:
    # it drives the ANCHOR exclusion, which is a statement about problems, and a
    # held-out user says nothing about whether a problem's anchor rows belong.
    if args.holdout_user_frac > 0:
        from auxiliary.rating_dev.holdout_set import held_out_users
        hu_list, hu_hash = held_out_users(args.seed, args.holdout_user_frac)
        hu = set(hu_list)
        m_u = np.array([str(p)[:8] in hu for p in prof_df["profile"]], bool)
        ho_mask_h = ho_mask_h | m_u
        print(f"HELD-OUT USERS: hash {hu_hash} | {len(hu_list)} accounts held out "
              f"({int(m_u.sum())} rows) — parameters never see them, so a "
              f"walk-forward on these users is leak-free")
    if args.holdout_row_frac > 0:
        from auxiliary.rating_dev.holdout_set import held_out_rows
        hr_list, hr_hash = held_out_rows(args.seed, args.holdout_row_frac)
        hr = set(hr_list)
        m_r = np.array([(str(p)[:8], str(t)) in hr
                        for p, t in zip(prof_df["profile"], prof_df["problem"])], bool)
        ho_mask_h = ho_mask_h | m_r
        print(f"HELD-OUT ROWS: hash {hr_hash} | {len(hr_list)} attempts held out "
              f"({int(m_r.sum())} profile rows; overlapping windows duplicate each "
              f"attempt, and BOTH copies are masked)")

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

    # --- assemble training rows: human (w=1) ⊕ anchor (w=anchor_weight), MINUS held-out problems
    #     (held-out PROBLEMS get NO training rows, so the bench measures pure feature-prior transfer;
    #     held-out VALIDATION itself now lives in auxiliary.rating_dev.validate_ratings, not here). ---
    train_h = ~ho_mask_h
    if args.mixed_only:
        # Conditional-MLE-style filter (user direction): train only on rows whose PROBLEM has both
        # outcomes and whose USER has both outcomes among the kept training rows. One-sided score
        # patterns have monotone likelihood (an all-success problem's BCE only improves as
        # difficulty → −∞) — the classical Rasch perfect/zero-score drop — and their outward pull
        # is a candidate cause of the no-anchor difficulty-axis warp (Finding OO). Iterated to a
        # fixed point (each pass can re-expose one-sided units). Eval rows (by-row val, held-out
        # problems) are deliberately NOT filtered: the treatment is training-only, so metrics stay
        # comparable across cells; dropped users' θ never trains (stays at init) — their val rows
        # are counted below.
        before = int(train_h.sum())
        for _ in range(50):
            ps = np.zeros(len(keys)); pf = np.zeros(len(keys))
            np.add.at(ps, prob_idx[train_h], y[train_h])
            np.add.at(pf, prob_idx[train_h], 1.0 - y[train_h])
            us = np.zeros(n_profiles); uf = np.zeros(n_profiles)
            np.add.at(us, prof_idx[train_h], y[train_h])
            np.add.at(uf, prof_idx[train_h], 1.0 - y[train_h])
            ok = (((ps > 0) & (pf > 0))[prob_idx] & ((us > 0) & (uf > 0))[prof_idx] & train_h)
            if int(ok.sum()) == int(train_h.sum()):
                break
            train_h = ok
        trained_users = np.unique(prof_idx[train_h])
        print(f"mixed-only filter: train rows {before} → {int(train_h.sum())} "
              f"({int(train_h.sum()) / before:.1%}) | problems "
              f"{len(np.unique(prob_idx[train_h]))} | profiles {len(trained_users)}/{n_profiles}")
    u_all = [prof_idx[train_h]]; p_all = [prob_idx[train_h]]; t_all = [y[train_h]]
    w_all = [np.ones(int(train_h.sum()), np.float32)]
    if n_anchor:
        keep_a = (np.ones(len(ap_), bool) if (args.holdout_keep_anchors or not ho_problems)
                  else ~np.isin(ap_, list(ho_problems)))
        u_all.append(au[keep_a]); p_all.append(ap_[keep_a]); t_all.append(at[keep_a])
        w_all.append(np.full(int(keep_a.sum()), per_bot_weight, np.float32))
    n_human = int(train_h.sum())                             # human rows lead; anchors follow
    u_all = np.concatenate(u_all); p_all = np.concatenate(p_all)
    t_all = np.concatenate(t_all); w_all = np.concatenate(w_all)
    if args.certainty_weight_k > 0:
        # inverse-uncertainty row weights on the human rows (see --certainty-weight-k); profile
        # size n from the FULL kept profile (pc), the ability-precision proxy
        n_u = pc.reindex(profiles.categories).to_numpy(float)
        cw = (n_u / (n_u + args.certainty_weight_k))[u_all[:n_human]]
        w_all[:n_human] *= cw.astype(np.float32)
        print(f"certainty weights: k {args.certainty_weight_k:g} | row weight p[5,50,95] "
              f"{np.percentile(cw, [5, 50, 95]).round(3)} | effective human mass "
              f"{cw.mean():.1%}")

    tfeat = torch.tensor(feat, device=device)
    tu = torch.tensor(u_all, device=device)
    tp = torch.tensor(p_all, device=device)
    tt = torch.tensor(t_all, device=device)
    tw = torch.tensor(w_all, device=device)
    thu = torch.tensor(prof_idx, device=device)              # human-only, for the train-BCE trace
    thp = torch.tensor(prob_idx, device=device)
    thy = torch.tensor(y, device=device)
    n = len(u_all)

    # --- fit (optionally two-pass even-error group-DRO; see fit_model / the DRO block below) ---
    is_human = np.zeros(n, bool); is_human[:n_human] = True   # anchors gated out of the DRO term
    t_is_human = torch.tensor(is_human, device=device)
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

    if args.staged:
        # ---- coordinate-descent over the per-problem 4PL heads (rating → slope → floor) ----
        if args.holdout_problem_frac > 0 and os.path.abspath(args.out_dir) == os.path.join(
                PROJECT_ROOT, "ratings"):
            sys.exit("--staged + --holdout writes SUBSET-trained ratings; pass a temp --out-dir.")
        from auxiliary.rating_dev.validate_ratings import run_validation
        HN = {0: "rating", 1: "slope", 2: "floor"}
        HEAD_TOK = {"rating": 0, "slope": 1, "floor": 2}
        # stage plan: list of (free-head set, tag, train-ceiling?). --stage-plan gives an explicit
        # ordering, comma-separated phases of "+"-joined tokens from {rating,slope,floor,ceiling},
        # e.g. "rating,slope,floor+ceiling" or "rating,floor+ceiling,slope". Else --two-stage =
        # rating then slope+floor; else default rating → slope → floor (ceiling in the floor phase
        # if --free-lapse). A phase containing "ceiling" trains the global lapse that phase; the
        # lapse is a free parameter whenever ANY phase trains it (or --free-lapse).
        if args.stage_plan:
            plan = []
            for i, phase in enumerate(args.stage_plan.split(",")):
                toks = [t.strip() for t in phase.split("+")]
                heads = {HEAD_TOK[t] for t in toks if t in HEAD_TOK}
                plan.append((heads, f"stage{i + 1}_" + "+".join(toks), "ceiling" in toks))
        elif args.two_stage:
            plan = [({0}, "stage1_rating", False), ({1, 2}, "stage2_slope+floor", args.free_lapse)]
        else:
            plan = [({0}, "stage1_rating", False), ({1}, "stage2_slope", False),
                    ({2}, "stage3_floor", args.free_lapse)]
        free_lapse = args.free_lapse or any(tl for _, _, tl in plan)
        torch.manual_seed(args.seed)
        m = RatingNet(fdim, n_profiles + n_anchor, tuple(args.diff_hidden),
                      dropout=args.dropout, lapse_cap=args.lapse_cap,
                      slope_floor=args.slope_floor, floor_cap=args.floor_cap, n_anchor=n_anchor,
                      hier_slope=False, hier_floor=False, global_slope=False,
                      global_slope_fixed=None, mech_floor_botf=None, mech_floor_scale_fixed=None,
                      floor_resid=False, n_problems=len(keys), per_problem=False,
                      lapse_human_only=args.lapse_human_only, lapse_init=args.lapse_init,
                      lapse_fixed=(None if free_lapse else args.lapse_fixed),
                      act_dropout=args.act_dropout, act_dim=act_block,
                      bw_dropout=args.bw_dropout, bw_dim=nb if use_bw else 0,
                      move_dropout=args.move_dropout, move_dim=nmv if use_mv else 0,
                      cont_dropout=args.cont_dropout, cont_dim=nct if use_ct else 0,
                      separate_trunks=args.separate_trunks, gauge=args.gauge,
                      act_dropout_mode=args.act_dropout_mode).to(device)
        m.staged = True
        m.const_a = float(args.stage_slope_const)
        m.const_c = float(args.stage_floor_const)
        m.global_stage1 = args.global_stage1
        if args.global_stage1:                                    # init the global scalars at the consts
            with torch.no_grad():
                m.gslope_raw.fill_(float(np.log(np.expm1(m.const_a))))
                m.gfloor_raw.fill_(float(np.log(m.const_c / (1 - m.const_c))))
        print(f"\n=== STAGED | {'SEPARATE per-head trunks' if m.separate else 'SHARED trunk + pin-weight ' + str(args.stage_pin_weight)}"
              f" | plan {[sorted(HN[h] for h in s) + (['ceiling'] if tl else []) for s, _, tl in plan]}"
              f" | slope const {m.const_a:g} floor const {m.const_c:g} | {args.stage_epochs} ep/stage"
              f" | dropout {args.act_dropout}/{args.bw_dropout} | ceiling "
              f"{'LEARNED' if free_lapse else 'fixed ' + str(args.lapse_fixed)} ===")

        def train_stage(free_set, train_lapse, epochs):
            """Free a SET of heads (data BCE) while already-trained heads are held at their stored
            per-problem values by a consistency penalty. Frozen heads stay LIVE in the prediction but
            are DETACHED from the BCE — their only gradient is the penalty (head→pin buffer), forcing
            the shared trunk to keep emitting them. train_lapse frees the ceiling this stage."""
            m.free_heads = set(free_set)
            # FROZEN = trained but NOT free this phase. A head re-freed in a later phase (e.g. a
            # rating REFIT after slope/floor/ceiling are in place) is in both trained_heads and
            # free_heads → it is treated as FREE (re-fit on the BCE), not held by the penalty.
            frozen = m.trained_heads - m.free_heads
            if free_lapse:
                m.lapse_raw.requires_grad_(bool(train_lapse))
            opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=args.lr)
            for ep in range(epochs):
                m.train()
                perm = torch.randperm(n, device=device)
                for i in range(0, n, args.batch):
                    bi = perm[i:i + args.batch]
                    pid = tp[bi]
                    b, a, c = m.params(tfeat[pid], pid)          # live (not-yet heads already const)
                    th = m.ability(tu[bi]); lam = m.lapse()
                    b_p = b.detach() if 0 in frozen else b       # frozen heads: used live, no BCE grad
                    a_p = a.detach() if 1 in frozen else a
                    c_p = c.detach() if 2 in frozen else c
                    pred = (c_p + (1.0 - c_p - lam) * torch.sigmoid(a_p * (th - b_p))).clamp(EPS, 1 - EPS)
                    bce = (torch.nn.functional.binary_cross_entropy(pred, tt[bi], reduction="none")
                           * tw[bi]).sum() / tw[bi].sum()
                    pen = torch.zeros((), device=device)
                    if 0 in frozen:                              # frozen heads → maintain pin buffer
                        pen = pen + ((b - m.pin_b[pid]) ** 2).mean()
                    if 1 in frozen:
                        pen = pen + ((a - m.pin_a[pid]) ** 2).mean()
                    if 2 in frozen:
                        pen = pen + ((c - m.pin_c[pid]) ** 2).mean()
                    loss = bce + args.stage_pin_weight * pen
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():                                # human train BCE for the trace
                m.eval()
                hb = float(torch.nn.functional.binary_cross_entropy(m(thu, tfeat[thp], thp), thy))
            return hb

        def write_stage(tag):
            b_all, a_all, c_all, theta_full, lam = score_corpus(m)
            # drift readout: how far the LIVE frozen heads ended from their stored per-problem values
            # (small = the penalty held them, so the single coherent forward still reflects each stage)
            for h, live, buf in [(0, b_all, m.pin_b), (1, a_all, m.pin_a), (2, c_all, m.pin_c)]:
                if h in m.trained_heads:
                    print(f"   {HN[h]}-maintain drift: mean|Δ| "
                          f"{np.abs(live - buf.cpu().numpy()).mean():.4f}")
            rating = b_all + clear_logit_vec(c_all, lam) / a_all
            lo, hi = np.percentile(rating, [args.winsor, 100 - args.winsor])
            rating = np.clip(rating, lo, hi)
            theta = theta_full[:n_profiles]
            sdir0 = os.path.join(args.out_dir, tag)
            for s in SIZES:
                sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
                tids = [keys[i][1] for i in sel]
                d = os.path.join(sdir0, str(s)); os.makedirs(d, exist_ok=True)
                pd.DataFrame({"tsumego_id": tids, "rating": rating[sel]}).to_csv(
                    os.path.join(d, RATINGS_CSV), index=False)
                pd.DataFrame({"tsumego_id": tids, "slope": a_all[sel]}).to_csv(
                    os.path.join(d, SLOPE_CSV), index=False)
                pd.DataFrame({"tsumego_id": tids, "floor": c_all[sel]}).to_csv(
                    os.path.join(d, FLOOR_CSV), index=False)
            pd.DataFrame({"profile": list(profiles.categories), "theta": theta}).to_csv(
                os.path.join(sdir0, "user_theta.csv"), index=False)
            with open(os.path.join(sdir0, "rating_constants.json"), "w") as f:
                json.dump({"fat_finger_rate": round(lam, 4)}, f)
            print(f"  [{tag}] ceiling 1−λ {1 - lam:.4f} (λ {lam:.4f}) | "
                  f"rating [{rating.min():.2f},{rating.max():.2f}] "
                  f"slope [{a_all.min():.2f},{a_all.max():.2f}] floor mean {c_all.mean():.3f}")
            run_validation(sdir0, args.seed, args.holdout_problem_frac, expect_hash=ho_hash)

        for free_set, tag, train_lapse in plan:
            hb = train_stage(free_set, train_lapse, args.stage_epochs)
            freed = sorted(HN[h] for h in free_set) + (["ceiling"] if train_lapse else [])
            if args.global_stage1:
                freed += ["GLOBAL-slope", "GLOBAL-floor"] if any(h in ({1, 2} - m.trained_heads - free_set)
                                                                  for h in (1, 2)) else []
            print(f"\n--- {tag} trained (free {freed}); human train BCE {hb:.4f} ---")
            if args.global_stage1:
                print(f"    learned globals: slope {float(torch.nn.functional.softplus(m.gslope_raw)):.3f}"
                      f"  floor {float(torch.sigmoid(m.gfloor_raw)):.3f}")
            write_stage(tag)
            with torch.no_grad():                                # store the just-trained heads' LIVE
                b_all, a_all, c_all = m.params(tfeat)            # per-problem values for the penalty
                for h, buf, live in [(0, m.pin_b, b_all), (1, m.pin_a, a_all), (2, m.pin_c, c_all)]:
                    if h in free_set:
                        buf.copy_(live)
            if m.separate:                                       # freeze the freed heads' own trunks →
                for h in free_set:
                    [m.net_b, m.net_a, m.net_c][h].requires_grad_(False)  # exact maintenance, no leak
            m.trained_heads |= set(free_set)
        print("\n=== STAGED complete ===")
        return

    def fit_model(dro=None, hier_target=None, cond_div=None):
        """Train one RatingNet. dro=(bin_per_row, base_g, G, eta) swaps the HUMAN mean-BCE term for
        a base-rate-centered online group-DRO over rating(/gap) bins (Sagawa et al.): each minibatch
        aggregates the human rows' BCE into per-bin means Lg, the exponentiated-gradient weight q is
        nudged up on bins whose loss EXCEEDS their base-rate entropy base_g, and the human loss is
        the q-weighted bin mean. Centering on base_g targets REDUCIBLE skill-over-base, not the
        aleatoric base-rate floor — raw-loss DRO chases that floor and underperforms ERM (see
        docs/methodology.md, --even-error). The bot-anchor rows keep their plain
        weighted-BCE coverage term; only the human term is reweighted, and the batch human:anchor
        balance is preserved. q persists across minibatches/epochs. dro=None ⇒ plain weighted BCE."""
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
                      lapse_init=args.lapse_init,
                      lapse_fixed=(None if args.free_lapse else args.lapse_fixed),
                      act_dropout=args.act_dropout,
                      act_dim=act_block,
                      bw_dropout=args.bw_dropout,
                      bw_dim=nb if use_bw else 0,
                      move_dropout=args.move_dropout,
                      move_dim=nmv if use_mv else 0,
                      cont_dropout=args.cont_dropout,
                      cont_dim=nct if use_ct else 0,
                      separate_trunks=args.separate_trunks,
                      gauge=args.gauge,
                      act_dropout_mode=args.act_dropout_mode,
                      theta_posterior=args.theta_posterior).to(device)
        if args.floor_residual:
            ci = {"diff": 0, "slope": 1, "floor": 2}
            with torch.no_grad():
                for cn in args.resid_cols.split(","):
                    m.resid_mask[ci[cn.strip()]] = 1.0
        if args.theta_quadrature > 0:
            # ---- per-user Gauss-Hermite marginal likelihood (see --theta-quadrature): the
            # integral sits INSIDE the log, so a user's rows must be evaluated together on a
            # shared node grid — user-grouped batches replace row batches. No per-profile θ
            # parameter trains; σ_pop is learned and the posterior means are written into the θ
            # table afterwards so score_corpus/artifacts are unchanged in shape. ----
            assert args.gauge == "rating" and n_anchor == 0 and dro is None \
                and not args.gap_balance and not args.knn_reweight \
                and not args.certainty_weight_k and not args.floor_residual \
                and not args.theta_posterior, \
                "--theta-quadrature: --gauge rating + plain ERM config only " \
                "(no anchors/DRO/weights/residuals; the θ-whitening gauge would distort the " \
                "posterior means written to the θ table)"
            K = args.theta_quadrature
            gh_x, gh_w = np.polynomial.hermite.hermgauss(K)
            gh_x_t = torch.tensor(gh_x, dtype=torch.float32, device=device)
            log_gh_w = torch.tensor(np.log(gh_w / np.sqrt(np.pi)), dtype=torch.float32,
                                    device=device)
            log_sigma = torch.nn.Parameter(torch.tensor(float(np.log(1.2)), device=device))
            qopt = torch.optim.Adam(
                [p for p in m.parameters() if p.requires_grad] + [log_sigma], lr=args.lr)
            np.random.seed(args.seed)
            order = np.argsort(u_all, kind="stable")
            u_sorted = u_all[order]
            uniq_u, starts = np.unique(u_sorted, return_index=True)
            ends = np.append(starts[1:], len(u_sorted))
            UB = 512                                          # users per batch

            def user_batch(sel):
                """(row-index tensor, within-batch user position tensor) for user indices sel."""
                rows = np.concatenate([order[starts[i]:ends[i]] for i in sel])
                upos = np.repeat(np.arange(len(sel)), (ends[sel] - starts[sel]))
                return (torch.tensor(rows, device=device), torch.tensor(upos, device=device),
                        len(sel), len(rows))

            def node_loglik(tr, tup, n_users_b, th_nodes):
                """Per-user per-node log-likelihood L (n_users_b, K) for the rows tr."""
                b, a, c = m.params(tfeat[tp[tr]], tp[tr])
                lam = m.lapse()
                P = (c.unsqueeze(1) + (1.0 - c - lam).unsqueeze(1)
                     * torch.sigmoid(a.unsqueeze(1) * (th_nodes.unsqueeze(0) - b.unsqueeze(1)))
                     ).clamp(EPS, 1 - EPS)
                yb = tt[tr].unsqueeze(1)
                lp = yb * torch.log(P) + (1.0 - yb) * torch.log(1.0 - P)
                return torch.zeros(n_users_b, len(th_nodes), device=device).index_add(0, tup, lp)

            for ep in range(args.epochs):
                m.train()
                uperm = np.random.permutation(len(uniq_u))
                tot_nll, tot_rows = 0.0, 0
                for i0 in range(0, len(uniq_u), UB):
                    tr, tup, nub, nrows = user_batch(uperm[i0:i0 + UB])
                    th_nodes = float(np.sqrt(2.0)) * torch.exp(log_sigma) * gh_x_t
                    L = node_loglik(tr, tup, nub, th_nodes)
                    marg = torch.logsumexp(log_gh_w.unsqueeze(0) + L, dim=1)
                    loss = -marg.sum() / nrows               # per-row units
                    m.eval()                                 # corpus penalties, dropout off
                    if args.shallow_slope_penalty > 0:
                        loss = loss + args.shallow_slope_penalty * \
                            m.slope_unclamped(tfeat).pow(-2).mean()
                    if args.high_floor_penalty > 0:
                        lam_d = m.lapse().detach()
                        cap_d = (1.0 - 2.0 * lam_d).clamp(min=EPS, max=args.floor_cap)
                        ratio = (CALIBRATION_P / cap_d).clamp(EPS, 1.0 - EPS)
                        o_div = torch.log(ratio / (1.0 - ratio))
                        loss = loss + args.high_floor_penalty * torch.nn.functional.softplus(
                            m.net(tfeat)[:, 2] - o_div).pow(2).mean()
                    m.train()
                    qopt.zero_grad(); loss.backward(); qopt.step()
                    tot_nll -= float(marg.sum().detach()); tot_rows += nrows
                if ep % args.log_every == 0 or ep == args.epochs - 1:
                    print(f"  [quad] epoch {ep:4d}  marginal NLL/row {tot_nll / tot_rows:.4f}  "
                          f"λ {float(m.lapse()):.5f}  σ_pop {float(torch.exp(log_sigma)):.3f}")
            m.eval()                                          # posterior means → θ table
            with torch.no_grad():
                th_nodes = float(np.sqrt(2.0)) * torch.exp(log_sigma) * gh_x_t
                for i0 in range(0, len(uniq_u), UB):
                    sel = np.arange(i0, min(i0 + UB, len(uniq_u)))
                    tr, tup, nub, _ = user_batch(sel)
                    L = node_loglik(tr, tup, nub, th_nodes)
                    w_post = torch.softmax(log_gh_w.unsqueeze(0) + L, dim=1)
                    means = (w_post * th_nodes.unsqueeze(0)).sum(dim=1)
                    m.theta.weight[torch.tensor(uniq_u[sel], device=device), 0] = means
            print(f"  [quad] done | σ_pop {float(torch.exp(log_sigma)):.3f} | posterior-mean θ "
                  f"std {float(m.theta.weight[:, 0].std()):.3f} (written to the θ table)")
            return m
        # frozen-mode residuals are fit in their own pass AFTER the backbone; exclude the
        # δ table from the main optimizer so the backbone never sees it. detached-mode
        # residuals DO train here, but in their own param group at the frozen pass's δ lr
        # (their gradient comes only from the detached branch below).
        detached = args.floor_residual and args.floor_resid_mode == "detached"
        main_params = [p for nm, p in m.named_parameters()
                       if not (args.floor_residual and args.floor_resid_mode != "joint"
                               and nm == "floor_resid.weight")]
        # --weight-decay: decoupled shrinkage on the TRUNK LINEAR WEIGHTS ONLY. The decay set is
        # named explicitly (net/net_b/net_a/net_c "<idx>.weight") rather than by "everything except
        # a deny-list", so a future parameter is unregularized until someone opts it in. theta,
        # biases, lapse_raw and the per-problem tables stay in the plain group (see the flag help).
        if args.weight_decay > 0:
            assert not args.staged, "--weight-decay is joint-pass only (not wired into --staged)"
            decay_names = {nm for nm, _ in m.named_parameters()
                           if re.match(r"^net(_[bac])?\.\d+\.weight$", nm)}
            assert decay_names, "--weight-decay set but the model has no feature trunk"
            named_main = [(nm, p) for nm, p in m.named_parameters()
                          if any(p is q for q in main_params)]
            groups = [{"params": [p for nm, p in named_main if nm in decay_names],
                       "weight_decay": args.weight_decay},
                      {"params": [p for nm, p in named_main if nm not in decay_names],
                       "weight_decay": 0.0}]
            print(f"  [wd] decoupled weight decay {args.weight_decay:g} on "
                  f"{len(groups[0]['params'])} trunk weight matrices "
                  f"({sorted(decay_names)}); {len(groups[1]['params'])} params undecayed")
        else:
            groups = [{"params": main_params}]
        if detached:
            groups.append({"params": [m.floor_resid.weight], "lr": 0.05})
        opt = (torch.optim.AdamW(groups, lr=args.lr) if args.weight_decay > 0
               else torch.optim.Adam(groups, lr=args.lr))

        def _lr_factor(ep):
            """Per-epoch multiplicative LR factor, applied to ALL param groups (so the detached-δ
            group's 0.05/0.01 lr ratio is preserved). constant ⇒ 1.0 every epoch (the historical
            flat Adam lr, behavior-preserving). cosine ⇒ linear warmup from --lr-warmup-init-frac
            up to 1.0 over --warmup-epochs, then a half-cosine anneal to 0 across the remaining
            epochs (no min floor — cosine is already flat at the tail, so a floor saves nothing)."""
            if args.lr_schedule != "cosine":
                return 1.0
            w = max(args.warmup_epochs, 0)
            if w > 0 and ep < w:
                f0 = args.lr_warmup_init_frac
                return f0 + (1.0 - f0) * (ep + 1) / w
            prog = min(max((ep - w + 1) / max(args.epochs - w, 1), 0.0), 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_factor)
        # held-out-PROBLEM rows (unseen-problem transfer) for the --eval-every convergence trace
        ho_idx_t = (torch.tensor(np.where(ho_mask_h)[0], device=device)
                    if (args.eval_every and args.holdout_problem_frac > 0 and ho_mask_h.any())
                    else None)
        pass_tag = 1 if dro is None else 2
        if dro is not None:
            rbt, base_g, G, eta = dro
            q = torch.ones(G, device=device) / G
        pin_cols, pin_ema = [], {}
        if args.head_pin_dropout > 0:
            assert m.net is not None, "--head-pin-dropout needs the feature net"
            hci = {"diff": 0, "slope": 1, "floor": 2}
            pin_cols = [hci[h.strip()] for h in args.head_pin_heads.split(",") if h.strip()]
        if args.mixup_alpha > 0:
            assert (dro is None and not args.theta_posterior and not args.theta_quadrature
                    and not args.floor_residual and not args.certainty_weight_k
                    and not args.head_pin_dropout and not args.staged), \
                "--mixup-alpha needs the plain joint fit"
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(n, device=device)
            for b0 in range(0, n, args.batch):
                bi = perm[b0:b0 + args.batch]
                opt.zero_grad()
                pin = None
                if pin_cols:
                    with torch.no_grad():      # EMA of each pinnable head's batch median
                        m.eval()
                        o_nb = m.net(tfeat[tp[bi]])
                        m.train()
                        for c_ in pin_cols:
                            med = float(o_nb[:, c_].median())
                            pin_ema[c_] = (med if c_ not in pin_ema
                                           else 0.99 * pin_ema[c_] + 0.01 * med)
                    if float(torch.rand(())) < args.head_pin_dropout:
                        c_ = pin_cols[int(torch.randint(len(pin_cols), ()).item())]
                        pin = {c_: pin_ema[c_]}
                # detached mode: the backbone term is evaluated WITHOUT δ_p (use_resid=False) so
                # the backbone fits exactly what a residual-free run would; δ gets its own
                # likelihood term against the detached backbone further down.
                if args.mixup_alpha > 0:
                    # Mixup row synthesis (see --mixup-alpha): every training point is a convex
                    # combination of two attempt rows — problem features, ability VALUES, and the
                    # soft target share one per-row λ — never a raw row. The distribution
                    # priors/barriers below still read the real corpus features.
                    bj = bi[torch.randperm(bi.shape[0], device=bi.device)]
                    if args.mixup_alpha == 1.0:
                        lmb = torch.rand(bi.shape[0], device=device)
                    else:
                        lmb = torch.distributions.Beta(
                            args.mixup_alpha, args.mixup_alpha).sample((bi.shape[0],)).to(device)
                    fmix = (lmb[:, None] * tfeat[tp[bi]]
                            + (1.0 - lmb[:, None]) * tfeat[tp[bj]])
                    b_m, a_m, c_m = m.params(fmix)
                    th_m = lmb * m.ability(tu[bi]) + (1.0 - lmb) * m.ability(tu[bj])
                    p = (c_m + (1.0 - c_m - m.lapse())
                         * torch.sigmoid(a_m * (th_m - b_m))).clamp(EPS, 1 - EPS)
                    ymix = lmb * tt[bi] + (1.0 - lmb) * tt[bj]
                    w = tw[bi]
                    loss = torch.nn.functional.binary_cross_entropy(
                        p, ymix, reduction="none").mean()
                elif dro is None:
                    p = m(tu[bi], tfeat[tp[bi]], tp[bi], use_resid=not detached, pin=pin)
                    l = torch.nn.functional.binary_cross_entropy(p, tt[bi], reduction="none")
                    w = tw[bi]
                    loss = (l * w).sum() / w.sum().clamp_min(EPS)
                else:
                    p = m(tu[bi], tfeat[tp[bi]], tp[bi], use_resid=not detached, pin=pin)
                    l = torch.nn.functional.binary_cross_entropy(p, tt[bi], reduction="none")
                    w = tw[bi]
                    hm = t_is_human[bi]
                    lh = l[hm]                                # human rows in this batch
                    gb = rbt[bi][hm]
                    gl = torch.zeros(G, device=device).index_add(0, gb, lh)
                    cnt = torch.zeros(G, device=device).index_add(0, gb, torch.ones_like(lh))
                    present = cnt > 0
                    Lg = gl[present] / cnt[present]
                    with torch.no_grad():                    # update q on base-centered bin loss
                        q[present] = q[present] * torch.exp(eta * (Lg.detach() - base_g[present]))
                        q[~present] = 0.0
                        q /= q.sum().clamp(min=EPS)
                    human_loss = (q[present] * Lg).sum()      # macro-mean over bins (the DRO term)
                    wa = w[~hm]                               # bot-anchor coverage term stays flat
                    anchor_loss = ((l[~hm] * wa).sum() / wa.sum().clamp_min(EPS)
                                   if wa.numel() else torch.zeros((), device=device))
                    fh = w[hm].sum() / w.sum().clamp_min(EPS)  # preserve the human:anchor balance
                    loss = fh * human_loss + (1.0 - fh) * anchor_loss
                if args.theta_posterior:
                    # ELBO KL term: Σ_u KL(N(m_u, s_u²) || N(0, σ_pop²)), in per-row loss units
                    # (÷ total row mass — the same Bayes-scale convention as the hier prior)
                    mw = m.theta.weight.squeeze(-1)
                    ls = m.theta_logsd.weight.squeeze(-1)
                    kl = 0.5 * ((torch.exp(2 * ls) + mw * mw) / torch.exp(2 * m.pop_logsd)
                                - 1.0 + 2 * m.pop_logsd - 2 * ls).sum()
                    loss = loss + kl / float(tw.sum())
                if detached:
                    # one-stage residual term: same rows/weights as the frozen pass would see,
                    # but the backbone (net, θ, λ, global slope) is detached in the graph — only
                    # δ_p receives this gradient, plus its ridge. At convergence the fixed point
                    # is the frozen two-stage solution (backbone at its residual-free optimum,
                    # δ at the ridge optimum given the backbone).
                    m.eval()      # dropout off: δ fits the clean backbone, like the frozen pass
                    pr = m(tu[bi], tfeat[tp[bi]], tp[bi], detach_backbone=True)
                    m.train()
                    lres = torch.nn.functional.binary_cross_entropy(pr, tt[bi], reduction="none")
                    loss = loss + ((lres * w).sum() / w.sum().clamp_min(EPS)
                                   + args.floor_resid_l2 * m.floor_resid.weight.pow(2).mean())
                if (args.l2_floor > 0 or args.l2_slope > 0 or args.l2_rating > 0
                        or args.shallow_slope_penalty > 0 or args.high_floor_penalty > 0
                        or args.slope_dist_prior > 0
                        or hier_target is not None or cond_div is not None
                        or args.floor_residual):
                    # global distribution priors over ALL problems' params (dropout off for a clean
                    # read of the per-problem floor/slope the net emits)
                    m.eval()
                    if cond_div is not None:
                        loss = loss + args.cond_diversity * m.cond_diversity_nll(tfeat, cond_div)
                    if m.floor_resid is not None and not detached:   # detached adds its own ridge
                        loss = loss + args.floor_resid_l2 * m.floor_resid.weight.pow(2).mean()
                    if hier_target is not None:
                        # Bayes-correct scale: the full objective is Σ_rows w·BCE + Σ_problems
                        # prior-NLL; in the per-row-mean units the loss is expressed in, the prior
                        # contributes (n_problems / total row mass)·mean-NLL — a derived constant,
                        # not a tuned knob.
                        hier_scale = args.hier_weight * len(keys) / float(tw.sum())
                        loss = loss + hier_scale * m.hier_nll(tfeat, hier_target)
                    if args.l2_floor > 0 or args.l2_slope > 0 or args.l2_rating > 0:
                        b_pen, a_pen, c_pen = m.params(tfeat)
                        if args.l2_floor > 0:                   # L2 outlier penalty: floor → its mean
                            loss = loss + args.l2_floor * c_pen.var(unbiased=False)
                        if args.l2_slope > 0:                   # shrink log-slope spread to its mean
                            loss = loss + args.l2_slope * torch.log(
                                a_pen.clamp_min(1e-4)).var(unbiased=False)
                        if args.l2_rating > 0:
                            # penalty on the DERIVED rating (the shipped 80%-clear point), the
                            # quantity that actually blows up (see --l2-rating). λ is detached so
                            # the penalty opens no new ceiling-training channel (the const-examinee
                            # sweeps' λ-collapse lesson).
                            lam_d = m.lapse().detach()
                            q_pen = ((CALIBRATION_P - c_pen)
                                     / (1.0 - c_pen - lam_d)).clamp(EPS, 1 - EPS)
                            r_pen = b_pen + torch.log(q_pen / (1.0 - q_pen)) / a_pen
                            loss = loss + args.l2_rating * r_pen.pow(2).mean()
                    if args.shallow_slope_penalty > 0:
                        # one-sided width² penalty on the PRE-clamp slope (see --shallow-slope-
                        # penalty); slope_unclamped keeps the gradient alive at the slope_floor rail
                        loss = loss + args.shallow_slope_penalty * \
                            m.slope_unclamped(tfeat).pow(-2).mean()
                    if args.high_floor_penalty > 0:
                        # easy-side divergence barrier in the floor head's raw logit space (see
                        # --high-floor-penalty); center = the raw output where c = CALIBRATION_P,
                        # λ detached so the barrier can't train the ceiling
                        lam_d = m.lapse().detach()
                        cap_d = (1.0 - 2.0 * lam_d).clamp(min=EPS, max=args.floor_cap)
                        ratio = (CALIBRATION_P / cap_d).clamp(EPS, 1.0 - EPS)
                        o_div = torch.log(ratio / (1.0 - ratio))
                        o_c = m.net(tfeat)[:, 2]
                        loss = loss + args.high_floor_penalty * \
                            torch.nn.functional.softplus(o_c - o_div).pow(2).mean()
                    if args.slope_dist_prior > 0:
                        # Quantile-match the corpus UNCLAMPED log-slope spectrum to the target
                        # normal: sorted log-slopes vs the target's inverse CDF on the plotting
                        # grid. Pins location/spread/shape of the distribution; assignment of
                        # slopes to problems stays free.
                        la_sorted = torch.sort(torch.log(m.slope_unclamped(tfeat))).values
                        loss = loss + args.slope_dist_prior * (
                            la_sorted - slope_target_q).pow(2).mean()
                    m.train()
                loss.backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                ll = torch.nn.functional.binary_cross_entropy(m(thu, tfeat[thp], thp), thy).item()
            lam_ep = float(m.lapse())
            maxq = float(q.max().item()) if dro is not None else float("nan")
            epoch_records.append((pass_tag, ep, ll, lam_ep))
            if ep % args.log_every == 0 or ep == args.epochs - 1:
                tag = "dro" if dro is not None else "mean"
                print(f"  [{tag}] epoch {ep:4d}  human train BCE {ll:.4f}  λ {lam_ep:.5f}"
                      + (f"  max_q {maxq:.3f}" if dro is not None else "")
                      + (f"  lr {opt.param_groups[0]['lr']:.5f}"
                         if args.lr_schedule != "constant" else ""))
            if ho_idx_t is not None and (ep % args.eval_every == 0 or ep == args.epochs - 1):
                with torch.no_grad():                            # m already in eval mode here
                    php = m(thu[ho_idx_t], tfeat[thp[ho_idx_t]], thp[ho_idx_t])
                    ho_bce = torch.nn.functional.binary_cross_entropy(php, thy[ho_idx_t]).item()
                print(f"  [eval] {'dro' if dro is not None else 'mean'} epoch {ep:4d}  "
                      f"held-out-PROBLEM logloss {ho_bce:.4f}  lr {opt.param_groups[0]['lr']:.5f}")
            sched.step()
        return m

    # --hier-slope-mu/sigma: a FIXED shrinkage target makes the free pass-1 fit unnecessary —
    # the target goes straight into the first (and then only) fit. Finding G/K ground this:
    # the shrinkage equilibrium barely depends on the pass-1 moments, so deriving them each run
    # buys a whole training pass for nothing.
    # Negative mu/sigma = sentinel for "derive the target from a free pass 1" (two-pass mode) —
    # needed since the cached 0.8/0.8 target became the DEFAULT; it is regime-sensitive
    # (Finding DD) and must be re-derived (one two-pass run) whenever the anchor or feature
    # set changes.
    fixed_hier = (args.hier_slope and args.hier_slope_mu is not None
                  and args.hier_slope_sigma is not None
                  and args.hier_slope_mu >= 0 and args.hier_slope_sigma >= 0)
    hier_target = cond_div_tgt = None
    if fixed_hier:
        assert not (args.hier_floor or args.hier_conditional or args.cond_diversity > 0), \
            "--hier-slope-mu/sigma implement a fixed SLOPE target only"
        hier_target = {"log-slope": (
            torch.full((len(keys),), args.hier_slope_mu, device=device),
            args.hier_slope_sigma)}
        print(f"  hier shrinkage target FIXED: log-slope μ {args.hier_slope_mu:+.3f} "
              f"σ {args.hier_slope_sigma:.3f} (pass-1 skipped)")
    m = fit_model(hier_target=hier_target)
    if (args.hier_slope or args.hier_floor or args.cond_diversity > 0) and not fixed_hier:
        # --- frozen pass-1 robust moments: the shrinkage target is the model's OWN unpenalized
        #     spectrum (median + IQR/1.349, so the clamp pileup / runaway tail don't poison it).
        #     See RatingNet.__init__ for why the target must be frozen (learned σ and per-step
        #     moment matching are both verified-degenerate). With --hier-conditional (and always
        #     for --cond-diversity) μ is RATING-conditional: smooth pass-1 conditional medians
        #     μ(rating) over 20 quantile bins, so the real rating-coupled structure is kept and
        #     only residual spread is shrunk/judged. ---
        m.eval()
        with torch.no_grad():
            o1 = m.net(tfeat).cpu().numpy().astype(np.float64)
        head_vals = {"log-slope": np.log(np.maximum(np.log1p(np.exp(np.minimum(o1[:, 1], 30.0))),
                                                    1e-4)),
                     "floor-logit": o1[:, 2]}
        rating1 = None
        if args.hier_conditional or args.cond_diversity > 0:
            b1, a1, c1, _, lam1 = score_corpus(m)
            rating1 = b1 + clear_logit_vec(c1, lam1) / a1

        def head_target(v, conditional):
            if conditional:
                edges = np.quantile(rating1, np.linspace(0, 1, 21)); edges[-1] += 1e-6
                bi = np.clip(np.searchsorted(edges, rating1, side="right") - 1, 0, 19)
                ok = [g for g in range(20) if (bi == g).any()]
                centers = np.array([np.median(rating1[bi == g]) for g in ok])
                meds = np.array([np.median(v[bi == g]) for g in ok])
                mu = np.interp(rating1, centers, meds)
            else:
                mu = np.full_like(v, np.median(v))
            r = v - mu
            sig = max(float((np.percentile(r, 75) - np.percentile(r, 25)) / 1.349), 0.1)
            return torch.tensor(mu, dtype=torch.float32, device=device), sig

        def tgt_str(tgt):
            return " | ".join(f"{k} μ~{float(np.median(mu.cpu().numpy())):+.3f} σ {sg:.3f}"
                              for k, (mu, sg) in tgt.items())

        if args.hier_slope or args.hier_floor:
            hier_target = {h: head_target(head_vals[h], args.hier_conditional)
                           for h, on in (("log-slope", args.hier_slope),
                                         ("floor-logit", args.hier_floor)) if on}
            print(f"  hier shrinkage target (pass-1 robust moments"
                  f"{', rating-conditional' if args.hier_conditional else ''}): "
                  + tgt_str(hier_target))
        if args.cond_diversity > 0:
            cond_div_tgt = {h: head_target(v, True) for h, v in head_vals.items()}
            print(f"  cond-diversity target (pass-1, rating-conditional, weight "
                  f"{args.cond_diversity}): " + tgt_str(cond_div_tgt))
    if args.gap_balance:
        # --- regime-balanced pass 2: reweight each HUMAN row by the inverse mass of its gap bin
        #     (gap z = a·(θ−rating) under the pass-1 model, equal-width bins over ±ZCLIP). The
        #     floor regime (far below), the transition band (slope) and the ceiling regime (far
        #     above) then pull on the gradient with the same total force as the near-rank bulk,
        #     instead of being outvoted by it — likelihood pressure, not injected targets. Weights
        #     are capped (--gap-balance-max) and renormalized so total human mass is unchanged
        #     (the human:anchor balance is preserved). Composes with --dro-skill (DRO pass bins
        #     from this model and keeps these row weights), but the intended use is ERM. ---
        bg_, ag_, cg_, thg_full, lamg = score_corpus(m)
        rating_g = bg_ + clear_logit_vec(cg_, lamg) / ag_
        hp_g = p_all[:n_human]; hu_g = u_all[:n_human]
        z_g = np.clip(ag_[hp_g] * (thg_full[hu_g] - rating_g[hp_g]), -ZCLIP, ZCLIP)
        gedges = np.linspace(-ZCLIP, ZCLIP, args.gap_balance_bins + 1); gedges[-1] += 1e-6
        gb = np.clip(np.searchsorted(gedges, z_g, side="right") - 1, 0, args.gap_balance_bins - 1)
        cnt = np.bincount(gb, minlength=args.gap_balance_bins).astype(np.float64)
        wmul = (n_human / (args.gap_balance_bins * np.maximum(cnt, 1.0))) ** args.gap_balance_power
        wmul = np.minimum(wmul, args.gap_balance_max)[gb]
        wmul *= n_human / wmul.sum()                          # preserve total human mass
        w_all[:n_human] = wmul.astype(np.float32)
        tw = torch.tensor(w_all, device=device)
        with np.printoptions(precision=2, suppress=True):
            print(f"  gap_balance: {args.gap_balance_bins} equal-width gap bins over ±{ZCLIP}, "
                  f"counts {cnt.astype(int)},\n               row weights "
                  f"{np.array([wmul[gb == g].mean() if (gb == g).any() else 0 for g in range(args.gap_balance_bins)])} "
                  f"(cap {args.gap_balance_max}, power {args.gap_balance_power})")
    if args.knn_reweight:
        # Continuous kNN analogue of --gap-balance over a choice of {rating, ability, gap}.
        # Each human row's weight ∝ (distance to its k-th NN in the z-scored axis space)^beta,
        # so the sparse EXTREMES (which equal-width/quantile bins smear into a neighbour) get
        # upweighted smoothly; capped (--knn-max) and renormalized to preserve total human mass.
        # Pass-1 model defines the axes (as gap_balance/dro_skill). Standalone (ERM + weights).
        from sklearn.neighbors import NearestNeighbors
        bk_, ak_, ck_, thk_full, lamk = score_corpus(m)
        rating_k = bk_ + clear_logit_vec(ck_, lamk) / ak_
        hp_k = p_all[:n_human]; hu_k = u_all[:n_human]
        ax = {"rating": rating_k[hp_k], "ability": thk_full[hu_k],
              "gap": np.clip(ak_[hp_k] * (thk_full[hu_k] - rating_k[hp_k]), -ZCLIP, ZCLIP)}
        names = [a.strip() for a in args.knn_axes.split("+") if a.strip()]
        if any(nm not in ax for nm in names):
            raise SystemExit(f"--knn-axes must be a '+'-subset of {list(ax)}; got {args.knn_axes}")
        X = np.column_stack([(ax[nm] - ax[nm].mean()) / (ax[nm].std() + 1e-9) for nm in names])
        if args.knn_count_normalize:
            # Density over per-USER points (mean axis position), so heavy users don't spike it;
            # split each user's rarity mass W_u across their rows. See --knn-count-normalize.
            uniq, inv, cnt_u = np.unique(hu_k, return_inverse=True, return_counts=True)
            sums = np.zeros((len(uniq), X.shape[1])); np.add.at(sums, inv, X)
            Xu = sums / cnt_u[:, None]
            du, _ = NearestNeighbors(n_neighbors=min(args.knn_k + 1, len(uniq))).fit(Xu).kneighbors(Xu)
            Wu = np.power(np.maximum(du[:, -1], 1e-9), args.knn_beta)
            Wu = np.minimum(Wu / Wu.mean(), args.knn_max)     # cap per-USER rarity
            wmul = (Wu / cnt_u)[inv]                           # split across that user's rows
        else:
            dist, _ = NearestNeighbors(n_neighbors=args.knn_k + 1).fit(X).kneighbors(X)
            wmul = np.power(np.maximum(dist[:, -1], 1e-9), args.knn_beta)
            wmul = np.minimum(wmul / wmul.mean(), args.knn_max)
        wmul *= n_human / wmul.sum()                          # preserve total human mass
        w_all[:n_human] = wmul.astype(np.float32)
        tw = torch.tensor(w_all, device=device)
        print(f"  knn_reweight: axes [{'+'.join(names)}] k={args.knn_k} beta={args.knn_beta} "
              f"cap {args.knn_max}{' count-norm' if args.knn_count_normalize else ''} — row wt "
              f"p99/p50/p1 {np.percentile(wmul, 99):.2f}/{np.percentile(wmul, 50):.2f}/"
              f"{np.percentile(wmul, 1):.2f}")
    if (hier_target is not None and not fixed_hier) or cond_div_tgt is not None or args.gap_balance \
            or args.knn_reweight:
        print("  refitting (pass 2: "
              + " + ".join(s for s, on in (("hier shrinkage", hier_target is not None),
                                           ("cond-diversity", cond_div_tgt is not None),
                                           ("gap-balanced weights", args.gap_balance),
                                           ("knn-reweight weights", args.knn_reweight)) if on) + ")")
        m = fit_model(hier_target=hier_target, cond_div=cond_div_tgt)
    if args.dro_skill:
        # Pass 1 done. Bin the HUMAN train rows by their problem's pass-1 80%-clear rating (shared
        # ability units, pooled across boards), optionally × a relative-gap z axis; per-bin base-
        # rate loss = the bin's human-success entropy (the aleatoric floor the centering subtracts).
        b0_, a0_, c0_, theta0_full, lam0 = score_corpus(m)
        rating0 = b0_ + clear_logit_vec(c0_, lam0) / a0_      # per-problem rating (key order)
        hp = p_all[:n_human]; hu = u_all[:n_human]            # problem/user index per human train row
        row_b = rating0[hp]
        if args.dro_dyadic:
            rbin, nrb = dyadic_tail_bins(row_b, args.dro_r_depth)
        else:
            r_edges = np.quantile(row_b, np.linspace(0, 1, args.dro_bins + 1)); r_edges[-1] += 1e-6
            rbin = np.clip(np.searchsorted(r_edges, row_b, side="right") - 1, 0, args.dro_bins - 1)
            nrb = args.dro_bins
        if args.dro_2d:
            # gap z = a·(θ−rating) in success-curve logits; clip the broken-ability artifact tail.
            z_row = np.clip(a0_[hp] * (theta0_full[hu] - rating0[hp]), -ZCLIP, ZCLIP)
            if args.dro_dyadic:
                zbin, nzb = dyadic_tail_bins(z_row, args.dro_z_depth)
            elif args.dro_z_uniform:
                z_edges = np.linspace(-ZCLIP, ZCLIP, args.dro_z_bins + 1); z_edges[-1] += 1e-6
                zbin = np.clip(np.searchsorted(z_edges, z_row, side="right") - 1,
                               0, args.dro_z_bins - 1); nzb = args.dro_z_bins
            else:
                z_edges = np.quantile(z_row, np.linspace(0, 1, args.dro_z_bins + 1)); z_edges[-1] += 1e-6
                zbin = np.clip(np.searchsorted(z_edges, z_row, side="right") - 1,
                               0, args.dro_z_bins - 1); nzb = args.dro_z_bins
            G = nrb * nzb
            rb_h = (zbin * nrb + rbin).astype(np.int64)
            tag = f"2D {nzb}z×{nrb}r" + ("(dyadic)" if args.dro_dyadic else "")
        else:
            G = nrb
            rb_h = rbin.astype(np.int64)
            tag = f"{nrb} rating" + ("(dyadic)" if args.dro_dyadic else "")
        yh = t_all[:n_human]
        br = np.array([yh[rb_h == g].mean() if (rb_h == g).any() else 0.5 for g in range(G)])
        br = np.clip(br, EPS, 1 - EPS)
        base_g = torch.tensor(-(br * np.log(br) + (1 - br) * np.log(1 - br)),
                              dtype=torch.float32, device=device)
        rb_full = np.zeros(n, np.int64); rb_full[:n_human] = rb_h  # anchors → bin 0 (gated out)
        print(f"  dro_skill: {tag} bins, eta {args.dro_eta} — refitting (pass 2)")
        m = fit_model((torch.tensor(rb_full, device=device), base_g, G, args.dro_eta),
                      hier_target=hier_target, cond_div=cond_div_tgt)

    if args.floor_residual and args.floor_resid_mode == "frozen":
        # --- frozen-backbone residual pass: freeze net/θ/λ, fit ONLY the ridge-shrunk δ_p.
        #     Local corrections on a fixed geometry — the joint variant let the optimizer explain
        #     difficulty AS floor (verified collapse, --floor-resid-mode joint). Held-out problems
        #     have no training rows, so their δ stays 0 = pure feature prior. ---
        for p_ in m.parameters():
            p_.requires_grad_(False)
        m.floor_resid.weight.requires_grad_(True)
        ropt = torch.optim.Adam([m.floor_resid.weight], lr=0.05)
        m.eval()                                              # dropout off: backbone is fixed
        for ep in range(args.floor_resid_epochs):
            perm = torch.randperm(n, device=device)
            for b0 in range(0, n, args.batch):
                bi = perm[b0:b0 + args.batch]
                ropt.zero_grad()
                p = m(tu[bi], tfeat[tp[bi]], tp[bi])
                l = torch.nn.functional.binary_cross_entropy(p, tt[bi], reduction="none")
                w = tw[bi]
                loss = ((l * w).sum() / w.sum().clamp_min(EPS)
                        + args.floor_resid_l2 * m.floor_resid.weight.pow(2).mean())
                loss.backward()
                ropt.step()
        with torch.no_grad():
            d_ = (m.floor_resid.weight * m.resid_mask).cpu().numpy()
        names = ["diff", "slope", "floor"]
        stats = " | ".join(
            f"δ_{names[j]} std {d_[:, j].std():.3f} |>0.1|: "
            f"{100 * float((np.abs(d_[:, j]) > 0.1).mean()):.0f}%"
            for j in range(3) if float(m.resid_mask[j]) > 0)
        print(f"  frozen residual pass ({args.floor_resid_epochs} epochs, cols "
              f"{args.resid_cols}): {stats}")
    elif args.floor_residual and args.floor_resid_mode == "detached":
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
    # pre-winsor tail readout: the blowup diagnostic (rating = b + logit/a explodes exactly where
    # the free slope collapses toward the floor) — the winsor below hides it from the CSVs.
    with np.printoptions(precision=2, suppress=False):
        print(f"  PRE-winsor rating percentiles [0, .1, 1, 50, 99, 99.9, 100]: "
              f"{np.percentile(rating, [0, 0.1, 1, 50, 99, 99.9, 100])}")
        print(f"  slope percentiles [0, .1, 1, 50, 99, 99.9, 100]: "
              f"{np.percentile(a, [0, 0.1, 1, 50, 99, 99.9, 100])} | "
              f"at slope_floor rail ({m.slope_floor}): {float((a <= m.slope_floor * 1.01).mean()):.2%}")
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
        print("  hier shrinkage: target (pass-1) "
              + " | ".join(f"{k} μ~{float(np.median(mu.cpu().numpy())):+.3f} σ {sg:.3f}"
                           for k, (mu, sg) in hier_target.items())
              + "  →  final corpus moments "
              + " | ".join(f"{k} μ {mu:+.3f} σ {sg:.3f}"
                           for k, (mu, sg) in m.hier_moments(tfeat).items()))
    if n_anchor:
        print("  bot-user learned ability θ: "
              + ", ".join(f"{ranks[j]}={bot_theta[j]:+.2f}" for j in range(n_anchor)))
        if args.anchor_slope_discount:
            with torch.no_grad():
                disc = torch.nn.functional.softplus(m.slope_discount).cpu().numpy()
            print("  bot steepness discount (1.0 = human-equal): "
                  + ", ".join(f"{ranks[j]}={disc[j]:.2f}" for j in range(n_anchor)))

    # --- per-bin train-logloss diagnostic: HUMAN rows grouped into equal-count FINAL-rating bins.
    #     Printed for ERM and DRO alike (so two runs compare bin-for-bin), it shows WHERE the
    #     objective spends its error across the rating range. `excess` = logloss − the bin's base-
    #     rate entropy (the aleatoric floor); a low std of excess across bins = flat, even error
    #     (what the DRO targets), a high std = error piled at the tails (the ERM failure mode). ---
    if args.report_bins and args.report_bins > 1:
        with torch.no_grad():
            ph = np.clip(m(thu, tfeat[thp], thp).cpu().numpy(), EPS, 1 - EPS)
        bce_h = -(y * np.log(ph) + (1 - y) * np.log(1 - ph))
        r_h = rating[prob_idx]                               # each human row's problem FINAL rating
        edges = np.quantile(r_h, np.linspace(0, 1, args.report_bins + 1)); edges[-1] += 1e-6
        b_of = np.clip(np.searchsorted(edges, r_h, side="right") - 1, 0, args.report_bins - 1)
        obj = "DRO" if args.dro_skill else "ERM/BCE"
        print(f"\n  per-bin train logloss [{args.report_bins} equal-count rating bins | obj {obj}]:")
        print(f"    {'bin':>3} {'lo':>7} {'hi':>7} {'n':>8} {'base':>6} {'logloss':>8} {'excess':>8}")
        excess = []
        for g in range(args.report_bins):
            sel = b_of == g
            if not sel.any():
                continue
            brg = float(np.clip(y[sel].mean(), EPS, 1 - EPS))
            ent = -(brg * np.log(brg) + (1 - brg) * np.log(1 - brg))
            llg = float(bce_h[sel].mean()); ex = llg - ent
            excess.append(ex)
            print(f"    {g:>3} {edges[g]:>7.2f} {edges[g + 1]:>7.2f} {int(sel.sum()):>8} "
                  f"{brg:>6.3f} {llg:>8.4f} {ex:>+8.4f}")
        excess = np.array(excess)
        print(f"    overall {float(bce_h.mean()):.4f} | excess-over-base across bins: mean "
              f"{excess.mean():+.4f} std {excess.std():.4f} "
              f"[{excess.min():+.4f},{excess.max():+.4f}]  (lower std = flatter error)")

    # --- gap-binned CALIBRATION: predicted vs realized success by user-problem gap (the curve-
    #     shape metric; see --report-gap-bins). Δ = predicted − realized; a globally-too-flat fit
    #     over-predicts far below rank (+Δ, floor too high / slope too flat) and under-predicts
    #     far above (−Δ, ceiling too low). The client rank estimator consumes exactly this curve. ---
    def gap_calibration(mask, label):
        idx = np.where(mask)[0]
        if args.report_gap_bins < 2 or len(idx) == 0:
            return
        with torch.no_grad():
            tpi = torch.tensor(prob_idx[idx], device=device)
            pp = m(torch.tensor(prof_idx[idx], device=device), tfeat[tpi], tpi).cpu().numpy()
        z = np.clip(a[prob_idx[idx]] * (theta[prof_idx[idx]] - rating[prob_idx[idx]]),
                    -ZCLIP, ZCLIP)
        gedges = np.linspace(-ZCLIP, ZCLIP, args.report_gap_bins + 1); gedges[-1] += 1e-6
        bo = np.clip(np.searchsorted(gedges, z, side="right") - 1, 0, args.report_gap_bins - 1)
        print(f"\n  gap-binned calibration [{label} | gap = slope·(ability−rating), ±{ZCLIP}]:")
        print(f"    {'bin':>3} {'gap_lo':>7} {'gap_hi':>7} {'n':>8} {'pred':>6} {'real':>6} {'Δ':>7}")
        absd, ws = 0.0, 0
        for g_ in range(args.report_gap_bins):
            sel = bo == g_
            if not sel.any():
                continue
            pr, re_ = float(pp[sel].mean()), float(y[idx][sel].mean())
            absd += abs(pr - re_) * int(sel.sum()); ws += int(sel.sum())
            print(f"    {g_:>3} {gedges[g_]:>7.2f} {gedges[g_ + 1]:>7.2f} {int(sel.sum()):>8} "
                  f"{pr:>6.3f} {re_:>6.3f} {pr - re_:>+7.3f}")
        print(f"    n-weighted mean |Δ|: {absd / max(ws, 1):.4f}  (lower = truer curve shape)")

    # Two-way slice: corpus rating deciles × gap regimes. This is the PREDICTIVE form of the
    # structure tables' floor question — where humans actually visit a problem's floor regime
    # (z < −2), does predicted success match realized, conditioned on where the problem sits on
    # the rating axis? An empty far-below cell at an easy decile means the regime is
    # UNOBSERVABLE there (criterion vacuous, not passed); the floor shipped for such cells is a
    # prior choice no human row can falsify.
    def gap_rating_calibration(mask, label):
        idx = np.where(mask)[0]
        if args.report_gap_bins < 2 or len(idx) == 0:
            return
        with torch.no_grad():
            tpi = torch.tensor(prob_idx[idx], device=device)
            pp = m(torch.tensor(prof_idx[idx], device=device), tfeat[tpi], tpi).cpu().numpy()
        z = a[prob_idx[idx]] * (theta[prof_idx[idx]] - rating[prob_idx[idx]])
        dec_edges = np.quantile(rating, np.linspace(0, 1, 11)); dec_edges[-1] += 1e-6
        dec = np.clip(np.searchsorted(dec_edges, rating[prob_idx[idx]], side="right") - 1, 0, 9)
        regimes = [("z<-2", z < -2.0), ("-2..0", (z >= -2.0) & (z < 0.0)),
                   ("0..3", (z >= 0.0) & (z < 3.0)), ("z>3", z >= 3.0)]
        print(f"\n  rating-decile × gap-regime calibration [{label}] (Δ = pred − real):")
        print("    " + f"{'dec':>3} {'r_lo':>6} {'r_hi':>6}"
              + "".join(f" | {nm:^25}" for nm, _ in regimes))
        print("    " + " " * 17
              + "".join(f" | {'n':>7} {'pred':>5} {'real':>5} {'Δ':>4}" for _ in regimes))
        fb_absd, fb_n = 0.0, 0
        for d_ in range(10):
            row = f"    {d_:>3} {dec_edges[d_]:>6.2f} {dec_edges[d_ + 1]:>6.2f}"
            for ri, (nm, rm) in enumerate(regimes):
                sel = (dec == d_) & rm
                if sel.any():
                    pr, re_ = float(pp[sel].mean()), float(y[idx][sel].mean())
                    row += f" | {int(sel.sum()):>7} {pr:>5.3f} {re_:>5.3f} {pr - re_:>+5.2f}"
                    if ri == 0:
                        fb_absd += abs(pr - re_) * int(sel.sum()); fb_n += int(sel.sum())
                else:
                    row += f" | {0:>7} {'—':>5} {'—':>5} {'—':>5}"
            print(row)
        if fb_n:
            print(f"    far-below (z<-2) n-weighted mean |Δ| across deciles: "
                  f"{fb_absd / fb_n:.4f} (n {fb_n})")

    # Heavy-user variants bound the two opposite measurement biases: TRAIN tails are outcome-
    # contaminated (a sparse profile's row outcome is baked into the θ/gap it is binned by — a
    # leave-one-out effect; verified: baseline train far-below realized 0.23–0.28 vs val 0.38–0.46
    # at the same gap), while VAL tails are attenuation-biased (gap estimation noise flattens the
    # realized curve, over-stating "model too sharp"). For ≥100-train-row profiles both effects
    # shrink; if heavy-train and heavy-val converge, that shared table is the honest curve shape.
    train_mask = (np.ones(len(y), bool) if args.holdout_problem_frac == 0
                  else ~ho_mask_h)
    prof_cnt = np.bincount(prof_idx[train_mask], minlength=n_profiles)
    heavy_row = prof_cnt[prof_idx] >= 100
    gap_calibration(train_mask, "train rows")
    gap_calibration(train_mask & heavy_row, "train rows, heavy users (≥100 train rows)")
    gap_rating_calibration(train_mask & heavy_row,
                           "train rows, heavy users — outcome-contaminated, read A/B diffs only")

    # Held-out runs train on a SUBSET, so refuse to overwrite the production ratings/ — pass an
    # explicit --out-dir (e.g. a temp dir). The standalone bench (auxiliary.rating_dev.validate_ratings) scores
    # the written CSVs; --validate runs it automatically after the write below.
    if args.holdout_problem_frac > 0 and os.path.abspath(args.out_dir) == os.path.join(PROJECT_ROOT, "ratings"):
        sys.exit("--holdout-problem-frac writes a SUBSET-trained model; refusing to overwrite the "
                 "production ratings/. Pass an explicit --out-dir (e.g. /tmp/rv2_holdout).")

    # --- write outputs ---
    os.makedirs(args.out_dir, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(args.out_dir, str(s))
        os.makedirs(sdir, exist_ok=True)
        pd.DataFrame({"tsumego_id": tids, "rating": rating[sel]}).sort_values(
            "rating", ascending=False).to_csv(
            os.path.join(sdir, RATINGS_CSV), index=False)
        pd.DataFrame({"tsumego_id": tids, "slope": a[sel]}).sort_values(
            "slope", ascending=False).to_csv(
            os.path.join(sdir, SLOPE_CSV), index=False)
        pd.DataFrame({"tsumego_id": tids, "floor": c[sel]}).sort_values(
            "floor", ascending=False).to_csv(
            os.path.join(sdir, FLOOR_CSV), index=False)
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
               os.path.join(args.out_dir, "model.pt"))
    pd.DataFrame(epoch_records, columns=["pass", "epoch", "train_bce", "lapse"]).to_csv(
        os.path.join(args.out_dir, "epoch_log.csv"), index=False)
    print(f"wrote user_theta.csv ({n_profiles} profiles) + rating_constants.json "
          f"(fat_finger_rate λ={lam:.5f}) + model.pt to {args.out_dir}")

    if args.validate:
        # Convenience: run the standalone bench on what we just wrote (no manual chaining). It
        # re-derives the held-out set from (seed, frac) and confirms it matches the hash we trained
        # on (ho_hash), so a mismatch — e.g. an attempts-DB drift mid-run — fails loud.
        from auxiliary.rating_dev.validate_ratings import run_validation
        print("\n--validate: scoring the written ratings via auxiliary.rating_dev.validate_ratings ...")
        # If USERS were held out, the prequential walk-forward must score THOSE
        # users and no others. Passing everyone measures the model on accounts it
        # trained on: measured, that reads 27.6 mnat better than the truth, which
        # is eight times the effect sizes this bench is used to judge — and it
        # looks entirely plausible while doing it. So the held-out set is derived
        # here rather than left to the caller to remember.
        pq_users = None
        if args.holdout_user_frac > 0:
            from auxiliary.rating_dev.holdout_set import held_out_users
            pq_users, _h = held_out_users(args.seed, args.holdout_user_frac)
        run_validation(args.out_dir, args.seed, args.holdout_problem_frac,
                       expect_hash=ho_hash, prequential_users=pq_users)


if __name__ == "__main__":
    main()
