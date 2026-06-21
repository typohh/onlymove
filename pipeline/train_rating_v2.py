"""Cross-board feature→4PL rating model — the PRODUCTION rating trainer (STAGED self-consistent).

The production-only entry point: it trains the shipped model and writes the rating artifacts, and
does nothing else (no held-out eval, no survey-ρ metrics, no DRO / kNN / experiment knobs). The
ONLY flags are --out-dir and --seed; the shipped configuration is forced in main()'s production
block. The full experiment + evaluation surface lives in
auxiliary/rating_dev/train_rating_v2_staged.py (and …_research.py). Do experiments THERE, not here,
so a test tweak can never break the shipped ratings.

The shipped model is a TWO-PHASE coordinate-descent fit on a SHARED feature→4PL trunk (act ⊕ bw,
0.3 dropout): phase 1 fits the per-problem difficulty (rating) and simultaneously LEARNS a single
GLOBAL slope + GLOBAL floor + GLOBAL ceiling (`--global-stage1`); phase 2 then individualizes the
per-problem slope + per-problem floor and keeps refitting the global ceiling, while the difficulty is
held at its phase-1 values by a consistency penalty. Learning the globals jointly with the rating
AUTO-FINDS the self-consistent anchors (no hand fixed-point search), so it stays calibrated as the
corpus drifts — replacing the legacy mechanistic floor (0.70·bot_floor) and the hierarchical-slope
prior with self-consistency (no externally-imposed floor, no slope shrinkage). The learned globals
land near slope ~1.3 / floor ~0.15-0.20 / ceiling ~0.97; the `--stage-*-const`/`--lapse-init` values
are only the (barely-relevant) inits. Difficulty↔slope coupling means the per-problem slope mean ends
~0.1 above the learned global (an intrinsic residual, not a bug).

ONE model for 9x9 AND 13x13. A SHARED feature→4PL network maps each problem's features
to (difficulty b_p, slope a_p, floor c_p); per-(user, 2-month) ability θ_u is hard-gauged

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

OBJECTIVE — base-rate-centered even-error group-DRO (ON by default, ported from the legacy
residual model). Plain mean BCE is gradient-dominated by the dense mid-ranks, so it under-fits
the sparse difficulty TAILS — exactly where strong/grinder users live, who then over-rate. So the
HUMAN-attempt term is a two-pass online group-DRO (Sagawa et al.) over rating(×gap) bins: pass 1
trains mean BCE to get the rating that defines the bins, pass 2 reweights bins by their loss
EXCESS over each bin's base-rate entropy (the aleatoric floor — centering is load-bearing, raw-loss
DRO chases that floor and loses to ERM). The bot-anchor coverage rows keep their flat weighted BCE.
`--no-dro-skill` reverts to plain weighted BCE (ERM). A per-bin train-logloss table prints under
either objective (`--report-bins`) so a DRO and an ERM run compare bin-for-bin. See
docs/rating_model_comparison_results.md (--even-error) and auxiliary/train_rating_v2_legacy.py.

Outputs (parallel; nothing in pwa/tsumegos/ is touched — package_for_pwa ships these):
  ratings_v2/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= 80%-clear rating)
  ratings_v2/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= raw slope a_p)
  ratings_v2/<size>/tsumego_floor_final.csv     tsumego_id, floor    (= learned c_p)
  ratings_v2/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings_v2/rating_constants.json              {"fat_finger_rate": λ}  (the learned ceiling)
  ratings_v2/model_v2.pt                        weights + run metadata

Defaults are the production config (even-error DRO on); the other knobs are for experiments.

Usage:
  ./.venv/bin/python3 -m pipeline.train_rating_v2 [--epochs 300] [--anchor-weight 0.2] ...
  ./.venv/bin/python3 -m pipeline.train_rating_v2 --no-dro-skill   # plain-BCE ERM baseline
  # compare objectives without writing CSVs (per-bin logloss + held-out + survey ρ):
  ./.venv/bin/python3 -m pipeline.train_rating_v2 --holdout-problem-frac 0.1 [--no-dro-skill]
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


# The survey-ρ / time-adjusted-ρ metrics moved to auxiliary.validate_ratings (the single home of
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
                 move_dropout=0.0, move_dim=0, separate_trunks=False):
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
        # global_stage1: hold a not-yet-individualized head at a LEARNABLE GLOBAL scalar (trains in
        # phase 1, jointly with rating + the global ceiling) instead of a fixed const — auto-finds
        # the self-consistent global slope/floor; phase 2 then individualizes the per-problem heads.
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
        # docs/rating_4pl_structure_log.md). The discount absorbs that mismatch per bot:
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
        if self.staged:
            # NOT-YET-TRAINED heads (neither frozen nor free this stage) → global constant, kept out
            # of the prediction until their stage. Frozen (trained) and free heads stay LIVE/backbone-
            # produced; the frozen ones are held at their stored per-problem values by a consistency
            # penalty in the LOSS (not here), so the final model is ONE coherent forward pass.
            not_yet = {1, 2} - self.trained_heads - self.free_heads
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

    # ── PRODUCTION configuration (staged self-consistent fixed point) ──────────
    # The shipped model is a TWO-PHASE coordinate-descent fit on a SHARED feature trunk
    # (act ⊕ bw, 0.3 dropout): phase 1 fits rating with slope/floor/ceiling held at the
    # SELF-CONSISTENT constants below; phase 2 frees slope + floor + the global ceiling
    # together (rating held by a consistency penalty). The constants are the FIXED POINT
    # of that fit (the converged head means equal the inits) — they replace the old
    # mechanistic floor (0.70·bot_floor) and hierarchical-slope prior with self-consistency.
    # Re-derive them (auxiliary/rating_dev/train_rating_v2_staged.py: sweep --stage-slope-const
    # / --stage-floor-const / --lapse-init until final == init) whenever the corpus/features
    # change. The full experiment + eval surface lives in that research file. Do experiments THERE.
    args.features = "bw_act"; args.move_features = False; args.pca_dim = 128
    args.diff_hidden = [128, 64]; args.dropout = 0.1
    args.act_dropout = 0.3; args.bw_dropout = 0.3; args.move_dropout = 0.0
    args.lr = 0.01; args.batch = 65536; args.log_every = 50; args.epochs = 300
    args.min = 5; args.min_profile = 8; args.winsor = 1.0; args.slope_floor = 0.01
    args.lapse_cap = 0.05; args.lapse_fixed = 0.02; args.floor_cap = 1.0
    # staged 2-phase, with --global-stage1: phase 1 LEARNS a global slope + global floor + global
    # ceiling jointly with the per-problem rating (auto-finding the self-consistent globals — no hand
    # fixed-point search, so it stays calibrated as the corpus drifts); phase 2 individualizes the
    # per-problem slope + floor and keeps training the ceiling.
    args.staged = True; args.two_stage = False; args.global_stage1 = True
    args.stage_plan = "rating+ceiling,slope+floor+ceiling"
    args.free_lapse = True; args.separate_trunks = False
    args.stage_epochs = 100; args.stage_pin_weight = 10.0
    # INITS for the learned globals (auto-tuned to ≈ slope 1.26 / floor 0.20 / ceiling 0.98); the
    # exact init barely matters since phase 1 trains them.
    args.stage_slope_const = 1.0; args.stage_floor_const = 0.25; args.lapse_init = 0.02
    # priors OFF (replaced by the self-consistent fixed point)
    args.mech_floor = False; args.mech_floor_scale_fixed = 0.70
    args.hier_slope = False; args.hier_slope_mu = 0.8; args.hier_slope_sigma = 0.8
    args.hier_weight = 1.0; args.hier_floor = False; args.hier_conditional = False
    args.floor_residual = False; args.floor_resid_mode = "detached"
    args.floor_resid_l2 = 4.0; args.floor_resid_epochs = 100; args.resid_cols = "diff,slope,floor"
    args.global_slope = False; args.global_slope_fixed = None
    args.per_problem = False; args.lapse_human_only = False
    # bot-anchor coverage examinees OFF: the 20k/9d anchors STEEPEN the per-problem slope (mean
    # ~1.38 vs ~1.23 human-only) without any held-out-logloss benefit (Δ ~+0.003, within seed
    # noise) — i.e. they impose a bot-derived slope bias the human data doesn't support. They do
    # prop up the survey-ρ rank metric (~+0.028) via that steeper slope, but the honest
    # generalization metric is indifferent, so the model is fit on human attempts alone.
    args.anchors = False; args.anchor_ranks = "20k,9d"; args.anchor_weight = 0.2
    args.anchor_fixed_mass = True; args.anchor_slope_discount = False
    # production: no held-out, no eval, plain ERM (all research objectives OFF)
    args.holdout_problem_frac = 0.0; args.holdout_keep_anchors = False
    args.mixed_only = False; args.validate = False; args.eval_every = 0
    args.report_bins = 10; args.report_gap_bins = 10
    args.dro_skill = False; args.dro_eta = 0.2; args.dro_bins = 10; args.dro_2d = True
    args.dro_z_bins = 5; args.dro_dyadic = True; args.dro_z_depth = 4; args.dro_r_depth = 3
    args.dro_z_uniform = True
    args.gap_balance = False; args.gap_balance_bins = 10; args.gap_balance_max = 10.0
    args.gap_balance_power = 1.0
    args.knn_reweight = False; args.knn_axes = "rating+gap"; args.knn_k = 50
    args.knn_beta = 1.0; args.knn_max = 10.0; args.knn_count_normalize = False
    args.slope_dist_prior = 0.0; args.slope_dist_mu = 0.50; args.slope_dist_sigma = 0.45
    args.cond_diversity = 0.0; args.l2_floor = 0.0; args.l2_slope = 0.0
    args.head_pin_dropout = 0.0; args.head_pin_heads = "slope,floor"
    args.lr_schedule = "constant"; args.warmup_epochs = 10; args.lr_warmup_init_frac = 0.1
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

    # --- features (joint-PCA correct-move activations ⊕ common-column bot winrate
    #     [⊕ candidate-move evaluations]) ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    keys = [key for key in act if key in bw]                  # (size, tid); scorable iff both blocks
    key_idx = {key: i for i, key in enumerate(keys)}
    use_act = args.features in ("bw_act", "act")
    use_bw = args.features in ("bw_act", "bw")
    use_mv = args.move_features or args.features == "move"   # --features move = move block only
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
    #     come from the SHARED generator (auxiliary.holdout_set) so a held-out training run and the
    #     validation bench (auxiliary.validate_ratings) provably assume the identical set; the hash
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
              f"generalization test (confirm via auxiliary.validate_ratings --expect-hash {ho_hash})")

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
    #     held-out VALIDATION itself now lives in auxiliary.validate_ratings, not here). ---
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
                PROJECT_ROOT, "ratings_v2"):
            sys.exit("--staged + --holdout writes SUBSET-trained ratings; pass a temp --out-dir.")
        if args.validate:
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
                      act_dropout=args.act_dropout, act_dim=k if use_act else 0,
                      bw_dropout=args.bw_dropout, bw_dim=nb if use_bw else 0,
                      move_dropout=args.move_dropout, move_dim=nmv if use_mv else 0,
                      separate_trunks=args.separate_trunks).to(device)
        m.staged = True
        m.const_a = float(args.stage_slope_const)
        m.const_c = float(args.stage_floor_const)
        m.global_stage1 = args.global_stage1
        if args.global_stage1:                                    # init the learned globals at the consts
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
                    os.path.join(d, "tsumego_ratings_final.csv"), index=False)
                pd.DataFrame({"tsumego_id": tids, "quality": a_all[sel]}).to_csv(
                    os.path.join(d, "tsumego_quality_final.csv"), index=False)
                pd.DataFrame({"tsumego_id": tids, "floor": c_all[sel]}).to_csv(
                    os.path.join(d, "tsumego_floor_final.csv"), index=False)
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
            print(f"\n--- {tag} trained (free {freed}); human train BCE {hb:.4f} ---")
            if args.validate:                                    # research per-stage eval only
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

        # ---- PRODUCTION: write the final (last-phase) artifacts to --out-dir ----
        b_all, a_all, c_all, theta_full, lam = score_corpus(m)
        rating = b_all + clear_logit_vec(c_all, lam) / a_all     # 80%-clear ability under the 4PL
        lo, hi = np.percentile(rating, [args.winsor, 100 - args.winsor])
        rating = np.clip(rating, lo, hi)                          # percentile winsor (legacy robustness)
        theta = theta_full[:n_profiles]
        os.makedirs(args.out_dir, exist_ok=True)
        for s in SIZES:
            sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
            tids = [keys[i][1] for i in sel]
            sdir = os.path.join(args.out_dir, str(s)); os.makedirs(sdir, exist_ok=True)
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
            os.path.join(args.out_dir, "user_theta.csv"), index=False)
        with open(os.path.join(args.out_dir, "rating_constants.json"), "w") as f:
            json.dump({"fat_finger_rate": round(lam, 4)}, f, indent=2)
        torch.save({"state_dict": m.state_dict(), "keys": keys, "fdim": fdim, "k": k,
                    "n_bot_cols": nb, "n_anchor": n_anchor, "anchor_ranks": ranks,
                    "lapse": lam, "args": vars(args)}, os.path.join(args.out_dir, "model_v2.pt"))
        print(f"\n=== STAGED complete === wrote user_theta.csv ({n_profiles} profiles) + "
              f"rating_constants.json (λ={lam:.5f}) + model_v2.pt to {args.out_dir}")
        return

    def fit_model(dro=None, hier_target=None, cond_div=None):
        """Train one RatingNet. dro=(bin_per_row, base_g, G, eta) swaps the HUMAN mean-BCE term for
        a base-rate-centered online group-DRO over rating(/gap) bins (Sagawa et al.): each minibatch
        aggregates the human rows' BCE into per-bin means Lg, the exponentiated-gradient weight q is
        nudged up on bins whose loss EXCEEDS their base-rate entropy base_g, and the human loss is
        the q-weighted bin mean. Centering on base_g targets REDUCIBLE skill-over-base, not the
        aleatoric base-rate floor — raw-loss DRO chases that floor and underperforms ERM (see
        docs/rating_model_comparison_results.md, --even-error). The bot-anchor rows keep their plain
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
                      act_dim=k if use_act else 0,
                      bw_dropout=args.bw_dropout,
                      bw_dim=nb if use_bw else 0,
                      move_dropout=args.move_dropout,
                      move_dim=nmv if use_mv else 0).to(device)
        if args.floor_residual:
            ci = {"diff": 0, "slope": 1, "floor": 2}
            with torch.no_grad():
                for cn in args.resid_cols.split(","):
                    m.resid_mask[ci[cn.strip()]] = 1.0
        # frozen-mode residuals are fit in their own pass AFTER the backbone; exclude the
        # δ table from the main optimizer so the backbone never sees it. detached-mode
        # residuals DO train here, but in their own param group at the frozen pass's δ lr
        # (their gradient comes only from the detached branch below).
        detached = args.floor_residual and args.floor_resid_mode == "detached"
        main_params = [p for nm, p in m.named_parameters()
                       if not (args.floor_residual and args.floor_resid_mode != "joint"
                               and nm == "floor_resid.weight")]
        groups = [{"params": main_params}]
        if detached:
            groups.append({"params": [m.floor_resid.weight], "lr": 0.05})
        opt = torch.optim.Adam(groups, lr=args.lr)

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
                p = m(tu[bi], tfeat[tp[bi]], tp[bi], use_resid=not detached, pin=pin)
                l = torch.nn.functional.binary_cross_entropy(p, tt[bi], reduction="none")
                w = tw[bi]
                if dro is None:
                    loss = (l * w).sum() / w.sum().clamp_min(EPS)
                else:
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
                if (args.l2_floor > 0 or args.l2_slope > 0 or args.slope_dist_prior > 0
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
                    if args.l2_floor > 0 or args.l2_slope > 0:
                        _, a_pen, c_pen = m.params(tfeat)
                        if args.l2_floor > 0:                   # L2 outlier penalty: floor → its mean
                            loss = loss + args.l2_floor * c_pen.var(unbiased=False)
                        if args.l2_slope > 0:                   # shrink log-slope spread to its mean
                            loss = loss + args.l2_slope * torch.log(
                                a_pen.clamp_min(1e-4)).var(unbiased=False)
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

    # Held-out runs train on a SUBSET, so refuse to overwrite the production ratings_v2/ — pass an
    # explicit --out-dir (e.g. a temp dir). The standalone bench (auxiliary.validate_ratings) scores
    # the written CSVs; --validate runs it automatically after the write below.
    if args.holdout_problem_frac > 0 and os.path.abspath(args.out_dir) == os.path.join(PROJECT_ROOT, "ratings_v2"):
        sys.exit("--holdout-problem-frac writes a SUBSET-trained model; refusing to overwrite the "
                 "production ratings_v2/. Pass an explicit --out-dir (e.g. /tmp/rv2_holdout).")

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

    if args.validate:
        # Convenience: run the standalone bench on what we just wrote (no manual chaining). It
        # re-derives the held-out set from (seed, frac) and confirms it matches the hash we trained
        # on (ho_hash), so a mismatch — e.g. an attempts-DB drift mid-run — fails loud.
        from auxiliary.rating_dev.validate_ratings import run_validation
        print("\n--validate: scoring the written ratings via auxiliary.validate_ratings ...")
        run_validation(args.out_dir, args.seed, args.holdout_problem_frac, expect_hash=ho_hash)


if __name__ == "__main__":
    main()
