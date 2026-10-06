#!/usr/bin/env python3
"""ONE model producing the rating and the K skill axes together: compensatory
M4PL with per-problem parameters predicted from features.

    P = c_p + (1 - c_p - lambda) * sigmoid( a_p*(theta_u - b_p)
                                            + sum_k g_k(x_p)*v_u[k] )

        a_p, b_p, c_p   per-problem discrimination, difficulty, guessing floor
        g_k(x_p)        per-problem loading on axis k
        theta_u         per-user ability
        v_u[k]          per-user position on axis k
        lambda          one global slip rate

The axes are factors 2..K+1 of this model and the rating is factor 1. ONE TRUNK
produces every output, axes included — a Linear's output rows are independent, so
a shared head emitting (b, a, c, g_1..g_K) is exactly this. The axis path is
COUPLED: the axis objective reaches the trunk, so the trunk learns what the axes
need and not only what the rating needs.

The one unavoidable departure from textbook MIRT is that a_p, b_p, c_p and g_k
come from a FEATURE NET rather than being free per item: a third of the corpus
has no attempts at all, so free per-item parameters cannot price it.

PRODUCTION ONLY. No argparse, no flags, config baked in as constants, always seed
0. The experiment surface — every ablation, sweep, holdout form and alternative
architecture — lives in `auxiliary.rating_dev.train_joint_mirt`, so a test tweak
cannot reach the shipped fit. The measurement behind every constant here:
`docs/mirt_benchmark.md`.

THE TRUNK SEES `PRODUCTION_BLOCKS` (utils.rating.model_constants): the joint-PCA
correct-cell activations, the bot pass-rate block, and the board-size flag past the
dropout region. The answer-derived blocks — move evals, ownership swing, game phase
— are not trunk inputs: this same trunk is fanned out over every legal cell by
`pipeline.write_mirt_per_cell`, where a block describing the problem through its
ANSWER conditions every alternative cell on the correct move. They remain
prerequisites, one step further back, because `pipeline.build_motif_proxies` is
written over them and its verdicts are what the axes are taught onto.

Run `pipeline.extract_activations` then `pipeline.cache_rating_features` first; the
correct-cell activation cache is a hard prerequisite. So are the motif proxies,
`axes/motif_proxies.npz` from `pipeline.build_motif_proxies` — see below.

THE AXES ARE TAUGHT ONTO FIXED DIRECTIONS, which is what gives a name a chance to
outlive a refit. The objective is exactly invariant to rotating the axis space — the
likelihood reads the loadings and the user positions only through their product,
and V_RIDGE is rotation-invariant too, as is centring v — so
the fit has a continuum of equal solutions and no preference among them. Left to
itself it lands wherever initialization drops it: a 2% data change re-rolls the
basis as hard as a different seed does, which would silently reattach every name
in `axes/labels.json` to a direction it was not written for. So the axis map is
not re-derived. It is pulled toward the target by an auxiliary MSE at a constant
TEACH_WEIGHT for the whole run, while the trunk trains normally.

THE TARGET IS A PROXY RULE, NOT A PREVIOUSLY-FITTED DIRECTION.
`pipeline.build_motif_proxies` labels every problem against the hand-written
criteria in `utils.rating.motif_rules`, and `axes/labels.json`'s `components`
pairs those rules onto axes in axis order, `"A/B"` putting rule A at an axis's
positive tail and rule B at its negative; the first K_AXES entries are taught, so
that list is the shipping decision. Teaching from previously FITTED directions is
the alternative and it only half worked: they return at 0.43-0.73 agreement, and
inspection of axes at exactly those figures found four of six poles mutated or
unnameable — a name held numerically and lost its meaning. What settles it is that
the round trip buys no stability either: a stored direction is itself only an
approximation of the rule it was taught from, and chaining through it holds an axis
no better than teaching from that rule directly, so it adds a reference that moves
for nothing. A rule is a fixed STARTING POINT, not a guarantee — the axis is what
the model lands on while held toward the rule, and report_bank_identity below is
what measures how close that landed.

Binding by rule NAME is safe here in a way it is not for the dev motif store
(`axes/label_anchors.npz`, keyed by an `ax_<hex>` content hash): a rule name is
unique by construction, checked in motif_rules, while the store holds several
directions under each of these names. Nothing in production reads that store.

The per-axis SCALE is ignored — a rescaling of the loadings is absorbed by v — so
only DIRECTION is taught.

Writes:

  ratings/<size>/tsumego_ratings.csv    tsumego_id, rating (the 80%-clear point)
  ratings/<size>/tsumego_slopes.csv     tsumego_id, slope
  ratings/<size>/tsumego_floors.csv     tsumego_id, floor
  ratings/user_theta.csv                profile, theta, n_attempts
  ratings/rating_constants.json         {"fat_finger_rate": lambda, "slope_scale"}
  axes/per_problem.csv                  tsumego_id, size, embed_0..embed_{K-1}
  axes/axes.npz                         the axis map + its content-hash version

SLOPE_SCALE IS 1.0, within noise of this model's measured optimum. Swept on
held-out users and split by whether training saw the problem — the split is
necessary, since a never-trained problem is priced from the feature prior alone
and honestly wants a shallower slope, so pooling those rows in drags the optimum
down. Measured optimum 0.90 on both `all` and the seen partition; 1.0 costs 0.2
mnat against it, inside the 0.85 mnat seed noise. No post-hoc multiplier is
applied: slopes ship as fitted.

Two things this does NOT produce. `axes/per_cell/` — a K-vector for every legal
cell, needed because the client scores the move the user PICKED and not only the
answer — comes from `pipeline.write_mirt_per_cell`, a separate pass because it
reads the trunk once per cell rather than folding into a matmul. `axes/labels.json`
is hand-maintained: a human names an axis by eyeballing its extremes, and an
unnamed axis ships and works, the client just stays silent about it.

It retires any existing `axes/per_cell/` by renaming it: per-cell vectors
projected through a different axis map than `per_problem.csv` give a shard whose
two embedding sources disagree, and both files are individually well-formed, so
nothing downstream can catch it.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.model_constants import (  # noqa: E402
    CALIBRATION_P, DROPOUT, EPS, GAUGE_MOM, INPUT_DROPOUT, LAPSE_CAP,
    LAPSE_INIT,
    K_AXES, PCA_DIM, PRODUCTION_BLOCKS, SEED, SIZES, SLOPE_FLOOR,
    clear_logit_vec)
from utils.rating.features import (  # noqa: E402
    load_answer_choice, load_first_time_attempts, load_joint_mirt_features,
    load_legal_cell_counts)
# The trunk's layer stack, stated once — pipeline.write_mirt_per_cell rebuilds it
# to project every legal cell, and a stack that matches by shape while differing
# in its activation loads cleanly and projects through the wrong function.
from utils.rating.mirt_net import build_trunk  # noqa: E402
# The shared tail weight, in numpy. `_tail_w` below is the same function in torch;
# the objective, the teaching target and the identity report are all this shape, so
# they cannot drift apart.
from utils.rating.tail_agreement import (  # noqa: E402
    TAIL_HINGE, hinged_r, tail_weight as tail_weight_np)

# The fit runs a FIXED budget under a cosine decay of both rates to zero, and the
# final epoch is what ships. Nothing is selected on a stop split, which is why the
# shipped fit holds none aside (STOP_ROW_FRAC): a comparison arm passes a fraction
# and gets the split measured and printed as the record of where it got to. The decay is what makes the budget a real choice rather
# than a stopping rule in disguise. Under a constant rate the fit wanders inside a
# band at its end and the shipped value is whichever point of that band the last
# epoch lands on, so a patience rule reading the split selects an epoch out of
# the noise rather than out of the fit; the decay freezes the fit wherever it
# stands as the rate reaches zero — which cuts both ways, since a budget long
# enough for the held-out-problem score to have turned locks the overfit in. The
# budget is chosen on held-out PROBLEMS over a sweep (docs/mirt_benchmark.md).
# The row split cannot choose it: a held-out row shares its users and problems
# with training, so it keeps improving past the point where generalization to
# unseen problems has flattened.
EPOCHS = 7000
LR_SCHEDULE = "cosine"
CHECK_EVERY = 10
BATCH = 65536
# SPLIT RATES, and the split is mandatory. What the grid measures is the RATIO
# lr_field/lr_net, monotone and saturating near 30x: below 10x the fit loses
# 2.5-5 mnat. Given a ratio past 30x a slower net buys the last 0.8; 1e-4 and
# 3e-4 tie, and 3e-4 converges in half the epochs.
LR_NET = 3e-4
LR_FIELD = 1e-2
# The free per-user fields, BY PARAMETER NAME. Named rather than assembled by
# hand so the split cannot fall out of step with the module's own parameters:
# anything not listed here is "the net" and gets the slow rate, so a parameter
# added and forgotten lands on the safe side.
FIELD_PARAMS = ("theta.weight", "v", "lapse_raw", "link_power_raw",
                "floor_prior_mu", "floor_prior_logsd")
LOCAL_STEEP_SCALE = 0.25         # curve_param="local" only: initial steepness scale
V_RIDGE = 0.5                    # not binding: 0.05/0.15/0.5 score alike
ORTHO_RATING = 0.5               # keeps each TAIL off rating, slope and floor
# ONE barrier, on the slope, one-sided, acting on the PRE-SOFTPLUS output. The
# rating is b + kappa(c)/a, so a slope collapsing toward its floor sends it to
# infinity. Without this, slopes fall to a 0.94 median and ratings blow out to a
# range of [-225, +146] — a 22 mnat loss. There is no floor barrier: the floor is
# bounded structurally by cap = (1 - 2*lambda).
SHALLOW_TAU = 0.1
SHALLOW_W = 0.3
# The barrier reads the CORPUS rather than the batch, and has to: a problem
# nobody attempted has no likelihood term, so its slope is held off the floor by
# this and nothing else. It does not have to read ALL of it every step. A uniform
# sample with replacement is an UNBIASED estimate of the same corpus mean, and a
# run takes tens of thousands of steps, so every problem is drawn thousands of
# times over the fit — what the sample costs is variance in one step's gradient,
# not coverage.
#
# Reading all of it dominates the step, because the corpus is several times a
# batch and a step's cost is a trunk pass over whatever it is handed. So this
# sits well under BATCH, and moving it trades one number against the other in
# the obvious direction: more samples, steadier barrier gradient, slower epoch.
SHALLOW_SAMPLES = 20000
# ONE SCHEDULE, and the loop is uniform because of it. Holding the per-user axis
# positions at zero for an opening span is the obvious companion to teaching —
# while they are zero the likelihood's gradient with respect to the loadings is
# identically zero (the two appear only as a product), so the bank would shape
# the map with nothing competing, and the axis penalties would stay off a map
# that is random at epoch 0. It buys nothing. Measured head to head at 300
# epochs against none, same bank and seed: per-axis agreement with the bank comes
# back identical to three decimals on all three axes and the tails differ by at
# most ten problems in a thousand, while the loss is a hair BETTER without it.
# The pin does cost ~2 mnat through the first few hundred epochs — free positions
# fitting against a random projection, then unlearning it — but that is a
# transient, fully repaid by epoch 1000. So there is no pinned span, no release
# event, and no gating of the axis penalties on one; the ortho penalty in
# particular is now on from the first step rather than switching on at full
# weight against a map it has never seen, which removes an abrupt objective
# change rather than merely tolerating one.
#
# The pull is CONSTANT for the whole run, and it is what fixes the basis. The
# likelihood reads the loadings and the user positions only through their
# product, so rotating the axis space leaves the objective exactly unchanged and
# nothing else in it prefers one basis to another. Without a standing pull the
# axes drift off the directions they were taught, and the names in
# axes/labels.json stop describing what they are attached to.
#
# The magnitude is not a second lever on identity. Retention against the rule
# columns SATURATES — it does not respond to the pull across a wide range of
# weights — so what binds is the shape of the TARGET, not the strength of the
# pull; see docs/mirt_benchmark.md.
TEACH_WEIGHT = 0.25
# MUTUAL ORTHOGONALITY, ON THE DETACHED READOUT. The pull fixes which direction
# each axis is and prices nothing about two axes carrying one direction; the
# ridge on the positions rewards exactly that, being isotropic over axes that
# are each unit-RMS, so a direction split over two axes at half the position
# each makes every prediction one axis would and pays half the ridge. This term
# asks the batch correlation matrix of the loadings to be the identity. It reads
# loadings computed from a DETACHED copy of the trunk output — the same values,
# with no path back into the network the rating shares — so it can only
# re-weight what the axes read out of the representation and never reshape the
# representation itself. The detachment is what makes it affordable: reaching
# the trunk, the same term costs the rating held-out likelihood; detached, it
# buys distinct axes at a slightly better level with identity unchanged
# (docs/mirt_benchmark.md, the detach panel). The weight is the one point
# measured and is unswept.
ORTHO_AXES = 0.25
ORTHO_AXES_DETACH = True
# IDENTIFIABILITY, the only inclusion rule. Under the 4PL an all-correct record's
# likelihood rises monotonically to (1-lambda)^n as ability grows without bound,
# and an all-wrong one to (1-c)^n as it falls, so neither has an interior
# maximum. One of each outcome is the condition for the ability to be estimable
# at all; anything above it is a judgement about how WELL, which the likelihood
# already makes by giving a thin record few terms.
MIN_BOTH_OUTCOMES = 1

# TWO SYNTHETIC USERS, each holding one row on EVERY corpus problem, at the two
# extremes real players do not reach. They are ordinary users in every respect —
# rows in the same shuffled stream, a fitted ability, fitted axis positions, the
# same dropout, counted in the same batch statistics — and differ from a real
# user only in where their outcomes come from, which enter as an EXPECTATION
# rather than as sampled answers.
#   THE RANDOM USER picks uniformly among a problem's legal moves, so is right
#   with probability one over their count. It is what tells the floor apart
#   from the difficulty where no weak player has tried a problem: attempted
#   only by players who solve it, an easy problem and a high floor explain the
#   outcomes equally, nothing in the likelihood breaks the tie, and the floor
#   is free to run up to where the 80%-clear rating stops existing.
#   THE STRONG USER is right with the probability a chooser reading the stored
#   evaluations plays the answer (the move-eval cache's `answer_q`) — near
#   certain where the answer alone wins, lower where the search itself left it
#   unclear. Where every real attempt on a problem failed, a hard problem and
#   an unboundedly hard one explain the outcomes equally; this user's success
#   is what bounds it. Its outcome is scaled by the fit's own ceiling,
#   (1 - lapse) * q with the lapse held as a constant there, because an outcome
#   above the ceiling cannot be met by any ability.
# A WEIGHT is the share of the user's rows drawn into each epoch, afresh each
# time: 1 is the user as stated, less is the same user having answered that
# fraction of the corpus. Neither carries a rank or is part of any row the fit
# is scored on. Their fitted abilities are written to user_theta.csv under the
# profiles `synthetic:random` and `synthetic:strong` with n_attempts 0 — no real
# user id has that form, and a reader summing attempts is not moved by them.
RANDOM_USER_WEIGHT = 0.0
STRONG_USER_WEIGHT = 0.0
STOP_ROW_FRAC = 0.0              # the shipped fit trains on every row: the budget
                                 # is fixed, so a held-aside split decides nothing
                                 # and only withholds data. Comparison arms pass
                                 # their own fraction (ablations.ablation_mirt)
# The `spread` holdout (an ablation path; the shipped fit holds out rows). A
# per-problem departure from the fitted curve is visible only as outcomes that
# AGREE across different users on one problem, so it is drawn among problems
# carrying at least this many training rows, whole problems for one set and half
# of each problem's rows for the other — see the draw in run().
SPREAD_MIN_ATTEMPTS = 4
SPREAD_FRAC = 0.10               # of the eligible problems, per set
OUT_DIR = os.path.join(PROJECT_ROOT, "ratings")
AXES_DIR = os.path.join(PROJECT_ROOT, "axes")
MOTIF_PROXIES = os.path.join(AXES_DIR, "motif_proxies.npz")
LABELS = os.path.join(AXES_DIR, "labels.json")


RAMP_SHARPNESS = 10.0
# THE FLOOR IS A ROW OF THE TRUNK'S HEAD, like difficulty and slope. On its own
# that floor lands somewhere different every seed — served attempts sit near
# each problem's 80% point and say almost nothing about the curve below it —
# but under the ability prior and the base-rate start below it reproduces. A
# linear map of the inputs also reproduces, by having no room to drift, but it
# lets a heavy user be parked far below every problem and explained by floors.
# "linear", "bot" and the other sources remain for comparison.
FLOOR_MODE = "trunk"
# ABILITY HAS A POPULATION PRIOR WHOSE VARIANCE IS THE DATA'S OWN (Laplace-EM:
# each epoch sigma^2 = mean over users of (theta - mean)^2 + 1/I_u), and the fit
# STARTS AT THE POPULATION'S CLEARANCE (one ability shift at initialisation,
# so the floor is not the cheapest early intercept). Without the prior nothing
# prices a user's distance from everyone and a user can be explained by floors;
# without the initialisation the floor overshoots early and settles wherever
# its weak gradient leaves it. Together they make the curve's global shape
# reproduce across seeds. Evidence: auxiliary/rating_dev/exploration_tf3_refit.md.
THETA_LAPLACE = True
INIT_BASE_RATE = True


def _orthogonalize(g, steps=5):
    """Newton-Schulz iteration towards the orthogonal factor of `g` (the
    quintic of the Muon optimizer): every singular value pushed near one."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        xx = x @ x.T
        x = a * x + (b * xx + c * xx @ xx) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """EXPERIMENT: momentum whose update is orthogonalized per weight matrix,
    for the trunk's matrices only; everything else stays on Adam."""

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("buf", torch.zeros_like(p))
                buf.mul_(grp["momentum"]).add_(p.grad)
                upd = _orthogonalize(p.grad + grp["momentum"] * buf)
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if grp["weight_decay"]:
                    p.mul_(1.0 - grp["lr"] * grp["weight_decay"])
                p.add_(upd, alpha=-grp["lr"] * scale)


class _Both:
    """Two optimizers stepped as one, their groups listed together so the
    learning-rate schedule reaches all of them."""

    def __init__(self, *opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self):
        for o in self.opts:
            o.zero_grad()

    def step(self):
        for o in self.opts:
            o.step()


class JointMIRT(nn.Module):
    """Compensatory M4PL with per-problem parameters from a shared trunk.

    The one unavoidable departure from textbook MIRT is that a_p, b_p, c_p and
    g_k come from a FEATURE NET rather than being free per item: a third of the
    corpus has no attempts at all, so free per-item parameters cannot price it.
    """

    def __init__(self, fdim, n_users, input_dim, seed, k_axes=K_AXES,
                 dropout=DROPOUT, input_dropout=INPUT_DROPOUT, link_power=False,
                 link="sigmoid", odds_sharpness=1.0, blend_frac=None, stage_frac=None, stage_lapse=False, head_zero_init=False, curve_lr_mult=None, floor_init=None, curve_param="ab", lapse_fixed=None, gauge="sd",
                 floor_prior=False, floor_rel=False,
                 slope_height=False, floor_mode=FLOOR_MODE,
                 slope_mode="trunk", floor_cols=None,
                 hidden=None, lapse_mode="global"):
        super().__init__()
        self.k_axes = int(k_axes)
        # The two dropout sites are separate parameters because they suppress
        # different failure modes and are reasoned about separately (1.5); the
        # trunk's is passed on, the input region's is read by _drop below.
        self.input_dropout = float(input_dropout)
        torch.manual_seed(seed)
        self.input_dim = int(input_dim)

        # theta before the trunk so the RNG draw order is fixed. nn.Embedding
        # initializes to N(0, 1) itself, so there is nothing to re-draw.
        self.theta = nn.Embedding(n_users, 1)
        # EXPERIMENT (None = the shipped DIFF_HIDDEN): the trunk's hidden
        # widths; () makes every head a linear map of the inputs.
        self.trunk, d = (build_trunk(fdim, dropout=dropout) if hidden is None
                         else build_trunk(fdim, dropout=dropout,
                                          hidden=tuple(hidden)))
        self.head = nn.Linear(d, 3)                     # difficulty, slope, floor
        if head_zero_init:
            # EXPERIMENT: the slope and floor rows start at zero, weights and
            # bias, so every seed starts from the same flat curve shape and the
            # per-problem values move only where a gradient pushes them. Zeroed
            # after the default draw, so the RNG stream is unchanged.
            with torch.no_grad():
                self.head.weight[1:3].zero_()
                self.head.bias[1:3].zero_()
        # The axis map, on the TRUNK's representation. Same linear map, different
        # input: it adds no capacity (fewer parameters than reading the 212
        # features) and the axis objective reaches the trunk, so the trunk learns
        # what the AXES need and not only what the rating needs. Detaching would
        # make this feature reuse rather than a joint fit.
        self.g_lin = nn.Linear(d, self.k_axes, bias=False)
        self.v = nn.Parameter(torch.zeros(n_users, self.k_axes))
        li = float(np.clip(LAPSE_INIT / LAPSE_CAP, EPS, 1.0 - EPS))
        self.lapse_raw = nn.Parameter(
            torch.tensor(float(np.log(li / (1.0 - li)))))
        # EXPERIMENT (off as shipped): the main slope as sigmoid(z) ** power,
        # one fitted global exponent starting at 1. Above 1 the curve leaves the
        # floor more slowly than it reaches the ceiling. Only the training
        # prediction reads it; the written rating and the held-out diagnostics
        # still assume the symmetric curve.
        self.link_power = bool(link_power)
        # EXPERIMENT (off as shipped): the floor read in the SLOPE's units. The
        # head's floor output becomes a distance in ability, and the floor is
        # what the main slope would give a player that far from the problem:
        # logit(floor / cap) = slope * output. A floor can then not move
        # without saying how it relates to the slope beside it.
        self.floor_rel = bool(floor_rel)
        # EXPERIMENT (off as shipped): the slope output read as the curve's
        # steepness IN PROBABILITY at its midpoint. The 4PL's midpoint
        # derivative is (1 - floor - lapse) * slope / 4, so a raised floor
        # flattens the curve even with the slope untouched, and the two can
        # stand in for each other. Dividing the logistic slope by the height
        # (1 - floor - lapse) makes the midpoint derivative output / 4 whatever
        # the floor. The published triple is still an ordinary 4PL: the slope
        # written is the logistic one.
        self.slope_height = bool(slope_height)
        # EXPERIMENT ("trunk" as shipped): where the floor comes from. "global"
        # is one fitted floor for every problem; "bot" is a linear map from the
        # bot win-rate columns and the board flag alone (everything past the
        # activations), so the floor cannot draw on the activation trunk.
        # "linear" is a linear map from every input column, activations
        # included: low capacity like "bot", without restricting what it reads.
        # "weak_human" reads only the human-imitation bot at KYU strengths
        # (`floor_cols`, indices into the columns past the activations): the
        # floor as a calibration of how often a weak human-like player finds
        # the answer.
        # "own_trunk" gives the floor a trunk of its own, the same shape as the
        # shared one but sharing no weights with difficulty, slope or axes.
        if floor_mode not in ("trunk", "global", "bot", "linear", "weak_human",
                              "own_trunk"):
            raise ValueError(f"unknown floor_mode {floor_mode!r}")
        self.floor_mode = floor_mode
        # EXPERIMENT ("global" as shipped): "linear" gives every problem its own
        # lapse, a linear map of every input column under the same cap.
        if lapse_mode not in ("global", "linear"):
            raise ValueError(f"unknown lapse_mode {lapse_mode!r}")
        self.lapse_mode = lapse_mode
        if lapse_mode == "linear":
            self.lapse_lin = nn.Linear(int(fdim), 1)
            nn.init.zeros_(self.lapse_lin.weight)
            nn.init.constant_(self.lapse_lin.bias, float(np.log(li / (1.0 - li))))
        # EXPERIMENT ("trunk" as shipped): where the slope comes from, as with
        # the floor — "bot" a linear map of the bot columns and board flag,
        # "linear" a linear map of every input column, "global" one slope
        # shared by every problem.
        if slope_mode not in ("trunk", "bot", "linear", "global"):
            raise ValueError(f"unknown slope_mode {slope_mode!r}")
        self.slope_mode = slope_mode
        if slope_mode == "global":
            self.slope_global = nn.Parameter(torch.zeros(()))
        elif slope_mode != "trunk":
            self.slope_lin = nn.Linear(
                int(fdim) - (PCA_DIM if slope_mode == "bot" else 0), 1)
            nn.init.zeros_(self.slope_lin.weight)
        # `blend_frac`: a curriculum from one global floor and slope to the
        # per-problem ones. Each is used as (1 - w) * global + w * local in its
        # pre-activation, w rising linearly from 0 to 1 over that fraction of
        # the epochs, so the fit settles abilities and difficulty under a
        # shared curve shape before any problem may depart from it. The end
        # state is the ordinary model; only the path differs.
        self.blend_frac = blend_frac
        # EXPERIMENT ("ab" as shipped): how the trunk's difficulty and slope
        # outputs define the curve. "local" makes them the rating itself (the
        # 80% point) and the curve's steepness AT that point, and derives the
        # sigmoid's midpoint and slope from those and the floor. Served
        # attempts sit near the 80% point, so the data there measures exactly
        # these two; the floor then moves only the curve's tail and cannot
        # stand in for the slope, because changing it leaves the curve's level
        # and steepness at the 80% point where they were. Floors are held below
        # 0.8, where an 80% point exists.
        if curve_param not in ("ab", "local"):
            raise ValueError(f"unknown curve_param {curve_param!r}")
        self.curve_param = curve_param
        if blend_frac:
            self.slope_g = nn.Parameter(torch.zeros(()))
            self.floor_g = nn.Parameter(torch.zeros(()))
            self.register_buffer("blend_w", torch.zeros(()))
        # `stage_frac`: two stages. Until that fraction of the epochs one global
        # floor and one global slope serve every problem; then the globals are
        # frozen and the per-problem outputs take over, CENTRED the way the
        # difficulty gauge centres b — each batch's mean pre-activation is
        # subtracted and the frozen global added back (an EMA of the mean at
        # eval) — so the per-problem floors and slopes may spread but their
        # mean stays where stage one put it.
        self.stage_frac = stage_frac
        # `stage_lapse`: freeze the global lapse at the switch too, so stage two
        # changes only the per-problem departures.
        self.stage_lapse = bool(stage_lapse)
        if stage_frac:
            self.slope_g = nn.Parameter(torch.zeros(()))
            self.floor_g = nn.Parameter(torch.zeros(()))
            for nm in ("stage2", "slope_g_frozen", "floor_g_frozen",
                       "slope_mu", "floor_mu"):
                self.register_buffer(nm, torch.zeros(()))
        if floor_mode == "global":
            self.floor_global = nn.Parameter(torch.zeros(()))
        elif floor_mode == "own_trunk":
            self.floor_trunk, d_f = build_trunk(fdim, dropout=dropout)
            self.floor_out = nn.Linear(d_f, 1)
        elif floor_mode == "weak_human":
            self.register_buffer("floor_cols", torch.as_tensor(
                list(floor_cols), dtype=torch.long))
            self.floor_lin = nn.Linear(len(floor_cols), 1)
        elif floor_mode in ("bot", "linear"):
            self.floor_lin = nn.Linear(
                int(fdim) - (PCA_DIM if floor_mode == "bot" else 0), 1)
            nn.init.zeros_(self.floor_lin.weight)
            nn.init.zeros_(self.floor_lin.bias)
        # EXPERIMENT (off as shipped): an EMPIRICAL-BAYES prior on the floor's
        # raw output, normal with a FITTED mean and spread. A floor only the
        # weakest players' attempts can inform is otherwise set by wherever the
        # trunk drifts; the prior draws it toward the population's floors by as
        # much as the population's own spread says, with nothing tuned.
        self.floor_prior = bool(floor_prior)
        if self.floor_prior:
            self.floor_prior_mu = nn.Parameter(torch.zeros(()))
            self.floor_prior_logsd = nn.Parameter(torch.zeros(()))
        # EXPERIMENT ("sd" as shipped): what the difficulty gauge holds at one.
        # A standard deviation can be met by a few far outliers while the bulk
        # collapses; "iqr" holds the interquartile range, on the scale where a
        # normal's would equal its standard deviation, and centres on the median.
        if gauge not in ("sd", "iqr"):
            raise ValueError(f"unknown gauge {gauge!r}")
        self.gauge = gauge
        # EXPERIMENT (None as shipped: the lapse is fitted): hold it at a value.
        self.lapse_fixed = None if lapse_fixed is None else float(lapse_fixed)
        # EXPERIMENT (sigmoid as shipped): the main slope's shape between the
        # floor and the ceiling. The two ramps are the sigmoid's tangent at its
        # centre, a straight line in probability: "ramp_hard" clamps it at both
        # levels, so the levels are truly flat and pass no gradient; "ramp_soft"
        # rounds the two corners over a tenth of the ramp.
        # "max_floor": the main slope runs from 0 to the ceiling and the floor
        # is a level the curve cannot fall below, P = max(floor, main). Only
        # attempts where the floor is the larger term train it, and only those
        # where the main slope is larger train the slope and difficulty.
        # "max_ramp" is the same with the hard ramp as the main slope: the
        # hockey stick with the floor and the slope fully separated.
        # "clamp" / "clamp_ramp" extend the same to the lapse: the main slope
        # runs from 0 to 1 and the curve is clamped between the floor and the
        # ceiling, P = min(1 - lapse, max(floor, main)), so attempts on the
        # slope train neither level and its midpoint steepness is slope / 4.
        # "odds_clamp" is the clamp made smooth in ODDS: the floor's odds and
        # the main slope's odds ADD (a soft maximum, logit P ~ max(logit floor,
        # z)), and the lapse's odds add to the failure odds (a soft minimum
        # against the ceiling). Each level takes the gradient in proportion to
        # its share of the odds, so near the slope's middle the floor and lapse
        # get almost none, and nothing sets the width but the logit scale the
        # sigmoid already has. `odds_sharpness` k sharpens both soft extremes
        # to the k-norm of the odds — log odds = logsumexp(k * terms) / k —
        # so a level's gradient share at a given log-odds distance d from the
        # slope is sigmoid(-k d) instead of sigmoid(-d); k = 1 is the plain
        # odds sum and k -> infinity is the hard clamp in odds.
        if link not in ("sigmoid", "ramp_soft", "ramp_hard", "max_floor",
                        "max_ramp", "clamp", "clamp_ramp", "odds_clamp"):
            raise ValueError(f"unknown link {link!r}")
        self.link = link
        self.odds_sharpness = float(odds_sharpness)
        if self.link_power:
            self.link_power_raw = nn.Parameter(
                torch.tensor(float(np.log(np.expm1(1.0)))))
        # THE TWO GAUGES, both the same shape: batch statistics while training,
        # an EMA of them at eval. BUFFERS, so they travel with the weights
        # wherever the weights go — mirt_model.pt, which write_mirt_per_cell
        # reads, and any state_dict a caller saves or restores. The rating scale
        # rides on b's whitening and every shipped axis vector on g's divisor, so
        # weights beside a gauge from a different epoch price both by statistics
        # belonging to a different network.
        self.register_buffer("b_mu", torch.zeros(()))
        self.register_buffer("b_sd", torch.ones(()))
        self.register_buffer("g_rms", torch.ones(1, self.k_axes))
        # EXPERIMENT (None as shipped): slope and floor read from their own
        # head on the trunk's representation, trained at `curve_lr_mult` times
        # the network rate, so ability and difficulty settle first and the
        # curve shape follows — the two-stage ordering without a switch.
        # Built last so every other module draws the same initial weights.
        self.curve_lr_mult = curve_lr_mult
        if curve_lr_mult:
            self.head_cs = nn.Linear(d, 2)
        if floor_init is not None:
            # EXPERIMENT: start every floor near `floor_init` instead of near
            # (1 - 2*lapse)/2 — the floor row's bias is shifted by the logit
            # of the target, so the per-problem spread at initialisation is
            # unchanged. Applied after construction, so no initial weight draw
            # moves.
            with torch.no_grad():
                shift = float(np.log(floor_init / (1.0 - floor_init)))
                if curve_lr_mult:
                    self.head_cs.bias[1] += shift
                else:
                    self.head.bias[2] += shift

    def param_groups(self, lr_net=LR_NET, lr_field=LR_FIELD):
        """SPLIT RATES: the network trains an order of magnitude slower than the
        free per-user fields.

        The divergence this split was adopted against belonged to an earlier
        arrangement of the model and does NOT reproduce on the shipping one —
        measured, one shared rate stays finite and costs nothing that survives a
        second seed. What the fit is sensitive to across the range tried is the
        NETWORK's rate alone. So the split stands as the arrangement everything
        else was tuned inside, not as a bound either rate has been shown to
        need; the panel behind that is `ablations/panels/lr.json`."""
        net, field = [], []
        for name, prm in self.named_parameters():
            (field if name in FIELD_PARAMS else net).append(prm)
        groups = [{"params": net, "lr": lr_net},
                  {"params": field, "lr": lr_field}]
        if self.curve_lr_mult:
            cs = {id(q) for q in self.head_cs.parameters()}
            groups[0]["params"] = [q for q in net if id(q) not in cs]
            groups.append({"params": list(self.head_cs.parameters()),
                           "lr": lr_net * self.curve_lr_mult})
        return groups

    def lapse(self):
        if self.lapse_fixed is not None:
            return self.lapse_raw.detach() * 0.0 + self.lapse_fixed
        return LAPSE_CAP * torch.sigmoid(self.lapse_raw)

    def user_axis(self, uid):
        """Per-user axis positions, CENTRED over the population.

        An axis multiplied by a non-zero population mean is a global difficulty
        shift wearing an axis's clothes, and difficulty belongs in the rating,
        which is the channel the client's rank path actually reads. Removing the
        mean by construction makes that impossible where a penalty only made it
        expensive, and it costs nothing to do: the penalty read every user's
        position each step for its mean, which is exactly what centring reads.

        It leaves the parameter redundant by one direction — adding a fixed row
        to every user changes no prediction. Nothing acts on that direction:
        V_RIDGE shrinks these centred positions rather than the raw parameter,
        precisely so it is spent on what the likelihood reads. The redundancy is
        harmless because the gradient along it is analytically zero, and it is
        measured to stay near zero over a full run."""
        return (self.v - self.v.mean(dim=0, keepdim=True))[uid]

    def _drop(self, feat):
        """COLUMN dropout over the wide input region as ONE mask — activations and
        bot winrate together, every block measured about the PROBLEM. Only the
        board-size indicator sits past that region, since it identifies the board
        rather than the problem, so a new wide block belongs INSIDE input_dim and
        is then masked automatically."""
        if (not self.training or self.input_dropout <= 0.0
                or self.input_dim <= 0):
            return feat
        keep = (torch.rand(len(feat), self.input_dim, device=feat.device)
                >= self.input_dropout).float() / (1.0 - self.input_dropout)
        return torch.cat([feat[:, :self.input_dim] * keep,
                          feat[:, self.input_dim:]], dim=1)

    def slope_logit(self, feat):
        """Pre-softplus slope output — the space the barrier acts in, where the
        penalty falls away properly for healthy slopes."""
        if self.curve_param == "local":
            a = self.problem_params(feat)[1].clamp(min=1e-4)
            return a + torch.log(-torch.expm1(-a))   # inverse softplus, stable
        if self.slope_mode == "bot":
            return self.slope_lin(feat[:, PCA_DIM:])[:, 0]
        if self.slope_mode == "linear":
            return self.slope_lin(feat)[:, 0]
        if self.slope_mode == "global":
            return self.slope_global.expand(feat.shape[0])
        if self.curve_lr_mult:
            return self.head_cs(self.trunk(feat))[:, 0]
        return self.head(self.trunk(feat))[:, 1]

    def _staged(self, raw, g, frozen, mu_buf):
        if not bool(self.stage2):
            return g.expand_as(raw)
        if self.training:
            mu = raw.mean()
            with torch.no_grad():
                mu_buf.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * mu)
        else:
            mu = mu_buf
        return frozen + raw - mu

    def start_stage2(self):
        with torch.no_grad():
            self.slope_g_frozen.copy_(self.slope_g)
            self.floor_g_frozen.copy_(self.floor_g)
            self.stage2.fill_(1.0)
        self.slope_g.requires_grad_(False)
        self.floor_g.requires_grad_(False)
        if self.stage_lapse:
            self.lapse_raw.requires_grad_(False)

    def _blend_floor(self, cap, raw):
        if self.blend_frac:
            raw = (1 - self.blend_w) * self.floor_g + self.blend_w * raw
        if self.stage_frac:
            raw = self._staged(raw, self.floor_g, self.floor_g_frozen, self.floor_mu)
        return cap * torch.sigmoid(raw)

    def problem_params(self, feat, pin_gauge=False):
        """(difficulty b, discrimination a, floor c, axis loadings g).

        THE GAUGE MUST SIT ON DIFFICULTY, NOT ON AN INTERCEPT. Two freedoms have
        to be removed, a shift of ability and a rescaling of it, and whitening an
        ADDITIVE intercept removes only the first: ability*s with a/s leaves an
        additive intercept untouched, so its whitening is satisfied and nothing
        objects, while difficulty is MULTIPLIED by the slope and must rescale
        with it. The shipped rating is an ability, so under the unblocked
        rescaling every rating scales and the whole axis floats."""
        h = self.trunk(self._drop(feat))
        o = self.head(h)
        b = o[:, 0]
        if self.training:
            if self.gauge == "iqr":
                q = torch.quantile(b, torch.tensor(
                    [0.25, 0.5, 0.75], device=b.device, dtype=b.dtype))
                mu, sd = q[1], (q[2] - q[0]) / 1.349 + EPS
            else:
                mu, sd = b.mean(), b.std() + EPS
            with torch.no_grad():
                self.b_mu.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * mu)
                self.b_sd.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * sd)
        else:
            mu, sd = self.b_mu, self.b_sd
        b = (b - mu) / sd
        if self.slope_mode == "bot":
            so = self.slope_lin(feat[:, PCA_DIM:])[:, 0]
        elif self.slope_mode == "linear":
            so = self.slope_lin(self._drop(feat))[:, 0]
        elif self.slope_mode == "global":
            so = self.slope_global.expand_as(b)
        else:
            so = self.head_cs(h)[:, 0] if self.curve_lr_mult else o[:, 1]
        if self.blend_frac:
            so = (1 - self.blend_w) * self.slope_g + self.blend_w * so
        if self.stage_frac:
            so = self._staged(so, self.slope_g, self.slope_g_frozen, self.slope_mu)
        a = nn.functional.softplus(so).clamp(min=SLOPE_FLOOR)
        if self.lapse_mode == "linear":
            lam = LAPSE_CAP * torch.sigmoid(self.lapse_lin(self._drop(feat))[:, 0])
        else:
            lam = self.lapse()
        self._last_lam = lam
        cap = 1.0 - 2.0 * lam
        if self.curve_param == "local":
            cap = cap.clamp(max=CALIBRATION_P) * (1.0 - 1e-3)
        if self.floor_mode == "global":
            c = cap * torch.sigmoid(self.floor_global).expand_as(b)
        elif self.floor_mode == "bot":
            c = cap * torch.sigmoid(self.floor_lin(feat[:, PCA_DIM:])[:, 0])
        elif self.floor_mode == "own_trunk":
            c = cap * torch.sigmoid(
                self.floor_out(self.floor_trunk(self._drop(feat)))[:, 0])
        elif self.floor_mode == "weak_human":
            c = cap * torch.sigmoid(self.floor_lin(
                feat[:, PCA_DIM:][:, self.floor_cols])[:, 0])
        elif self.floor_mode == "linear":
            c = self._blend_floor(cap, self.floor_lin(self._drop(feat))[:, 0])
        else:
            fr = self.head_cs(h)[:, 1] if self.curve_lr_mult else o[:, 2]
            c = self._blend_floor(cap, a * fr if self.floor_rel else fr)
        if self.slope_height:
            a = a / (1.0 - c - lam)
        if self.curve_param == "local":
            # b holds the rating r and a the steepness dP/dtheta at r (scaled
            # so the initial output sits near the fitted range); solve for the
            # sigmoid's slope and midpoint.
            r_loc, steep = b, LOCAL_STEEP_SCALE * a
            q = ((CALIBRATION_P - c) / (1.0 - c - lam)).clamp(EPS, 1.0 - EPS)
            a = (steep / ((1.0 - c - lam) * q * (1.0 - q))).clamp(min=SLOPE_FLOOR)
            b = r_loc - torch.log(q / (1.0 - q)) / a
        self._last_a = a
        g = self.g_lin(h)
        # Kept for `detached_loadings`: the same representation and, below, the
        # same divisor this batch's loadings use, so a detached readout is the
        # batch's loadings exactly, minus the path back into the trunk.
        self._last_h = h
        # The axis gauge, run exactly like the difficulty one above: the batch's
        # own per-axis RMS while training, its EMA at eval. Eval is what makes it
        # a gauge rather than a side effect — the divisor stops depending on what
        # was passed last, and the per-cell fanout reads the same number out of
        # the checkpoint, so a cell and the correct-move vector in
        # per_problem.csv cannot end up on different scales.
        #
        # `pin_gauge` overwrites it with THIS batch's RMS, for the corpus-wide
        # pass that writes the outputs — see the call in main().
        rms = (g.pow(2).mean(dim=0, keepdim=True).sqrt() + EPS).detach()
        if pin_gauge:
            with torch.no_grad():
                self.g_rms.copy_(rms)
        elif self.training:
            with torch.no_grad():
                self.g_rms.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * rms)
        else:
            rms = self.g_rms
        self._last_rms = rms
        return b, a, c, g / rms

    def detached_loadings(self):
        """The last batch's loadings computed from a DETACHED copy of the trunk
        output: identical values, but a penalty on them reaches the axis readout
        alone and never the network the rating shares. No second trunk pass and
        no gauge update. The shipped mutual term reads this; the pull and the
        rating-orthogonality term read `forward`'s loadings, whose gradient
        does reach the trunk, and both are measured to need it."""
        return self.g_lin(self._last_h.detach()) / self._last_rms

    def forward(self, uid, feat):
        b, a, c, g = self.problem_params(feat)
        th = self.theta(uid).squeeze(-1)
        lam = self._last_lam
        z = (a * (th - b) + (g * self.user_axis(uid)).sum(-1)).clamp(-35, 35)
        # The SHIPPED per-problem triple, returned DETACHED for the rating-
        # orthogonality penalty: rating, slope, floor — exactly what packaging
        # writes and the client's rank path reads, and exactly what an axis
        # correlated with them would be duplicating through a channel the client
        # cannot use.
        #
        # Recomputing problem_params to get these would run the trunk twice and,
        # worse, tick the difficulty gauge's EMA twice per batch — the gauge is
        # defined over the attempt stream and an extra update would quietly move
        # the rating's units.
        q = ((CALIBRATION_P - c) / (1.0 - c - lam)).clamp(EPS, 1.0 - EPS)
        r = b + torch.log(q / (1.0 - q)) / a.clamp(min=1e-6)
        if self.link == "odds_clamp":
            lc = torch.log(c.clamp(EPS, 1 - EPS)) - torch.log1p(-c.clamp(EPS, 1 - EPS))
            ll = torch.log(lam) - torch.log1p(-lam)
            k = self.odds_sharpness
            low = torch.logaddexp(k * lc, k * z) / k     # log odds, floor + slope
            lp = -torch.logaddexp(-k * low, k * ll) / k  # log odds after the lapse
            return (torch.sigmoid(lp), g, torch.stack([r, a, c], dim=1).detach())
        if self.link in ("max_floor", "max_ramp", "clamp", "clamp_ramp"):
            main = (torch.sigmoid(z) if self.link in ("max_floor", "clamp")
                    else (0.5 + z / 4.0).clamp(0.0, 1.0))
            if self.link.startswith("max"):
                p = torch.maximum(c, (1.0 - lam) * main)
            else:
                p = torch.minimum(torch.maximum(c, main), 1.0 - lam)
            return (p, g, torch.stack([r, a, c], dim=1).detach())
        if self.link == "sigmoid":
            sz = torch.sigmoid(z)
        else:
            u = 0.5 + z / 4.0
            if self.link == "ramp_hard":
                sz = u.clamp(0.0, 1.0)
            else:
                sz = (nn.functional.softplus(RAMP_SHARPNESS * u)
                      - nn.functional.softplus(RAMP_SHARPNESS * (u - 1.0))
                      ) / RAMP_SHARPNESS
        if self.link_power:
            sz = sz ** nn.functional.softplus(self.link_power_raw)
        return (c + (1.0 - c - lam) * sz, g,
                torch.stack([r, a, c], dim=1).detach())


def _tail_w(torch, g, at):
    """The shared tail weight in torch, differentiable: z-score, then `z` above
    `at` and `at*exp((z-at)/at)` below. The same function the measurement uses
    (utils.rating.tail_agreement.tail_weight), so the objective and the report are
    the same shape.

    The exponent is CLAMPED because torch.where evaluates both branches: a large
    positive z overflows the exponential branch to inf even though the result is
    discarded, and the backward pass turns that into NaN through the zero
    gradient. Clamping leaves the kept branch exactly unchanged — where z >= at
    the clamped exponent is zero, so exp is 1.
    """
    z = (g - g.mean(dim=0, keepdim=True)) / (g.std(dim=0, keepdim=True) + 1e-6)
    zc = torch.clamp(z, max=at)
    return torch.where(z >= at, z, at * torch.exp((zc - at) / at))


def load_bank(key_idx, k_axes=K_AXES):
    """The proxy directions this fit's axes are taught onto, one row per problem of
    `key_idx` in that order, plus a mask of which rows are supplied.

    THE SEED IS A RULE, NOT A PREVIOUSLY-FITTED DIRECTION. `pipeline.build_motif_proxies`
    labels every problem against hand-written criteria; `axes/labels.json`'s
    `components` pairs those rules onto axes, `"A/B"` putting rule A at an axis's
    positive tail and rule B at its negative.

    That choice is measured, not stylistic. The objective is invariant to rotating
    the axis space, so a free fit lands on a different basis every seed — 0.34
    agreement between two seeds of one configuration. Teaching from previously
    FITTED directions was the answer to that and only half worked: they return at
    0.43-0.73 agreement, and inspection of axes at exactly those figures found four
    of six poles mutated or unnameable. A rule cannot drift, because it states what
    the direction IS rather than recording where a fit once landed.

    Columns are built as `f(A) - f(B)` with the shared tail weight, so a problem at
    neither pole contributes ~0 rather than cancelling two plateaus. A one-sided
    `"A"` is allowed and simply has no negative pole. The column is z-scored because
    scale is arbitrary — rescaling a loading is absorbed by v — so direction is the
    only part that means anything.

    Returns (tgt, ok, spec) — the spec so the caller can stamp exactly what was
    taught.
    """
    for path in (MOTIF_PROXIES, LABELS):
        if not os.path.exists(path):
            raise SystemExit(
                f"NO MOTIF PROXIES at {path}. They are the directions the axes are "
                f"taught onto, without which this fit would re-roll the basis and "
                f"every name in axes/labels.json would land on a direction it was "
                f"not written for. Build them with "
                f"`python -m pipeline.build_motif_proxies`.")
    meta = json.load(open(LABELS))
    spec = [str(x) for x in meta.get("components", [])]
    if len(spec) < k_axes:
        raise SystemExit(
            f"axes/labels.json lists {len(spec)} components, this fit ships "
            f"K_AXES={k_axes}. `components` is the shipping decision: one entry "
            f"per axis, each a rule name or `\"A/B\"` to put two rules on one "
            f"axis's two tails.")
    # NESTED BY CONSTRUCTION: a smaller count keeps the first k rules and drops
    # the rest, so a sweep over the count varies only the count.
    spec = spec[:k_axes]

    z = np.load(MOTIF_PROXIES, allow_pickle=False)
    rules = [str(x) for x in z["rules"]]
    lab = np.asarray(z["labels"], bool)
    row = {f"{sz}:{t}": i for i, (sz, t) in
           enumerate(zip(z["size"], z["tids"] if "tids" in z.files else z["tid"]))}
    wanted = {r for col in spec for r in col.split("/")}
    missing = sorted(wanted - set(rules))
    if missing:
        raise SystemExit(
            f"axes/labels.json names rules the proxies do not carry: {missing}. "
            f"Available: {rules}. Rebuild with pipeline.build_motif_proxies, or "
            f"correct `components`.")

    tgt = np.zeros((len(key_idx), k_axes), np.float64)
    ok = np.zeros(len(key_idx), bool)
    src = np.full((len(key_idx), len(rules)), np.nan)
    for (sz, t), j in key_idx.items():
        i = row.get(f"{sz}:{t}")
        if i is not None:
            src[j] = lab[i]
            ok[j] = True
    if not ok.any():
        raise SystemExit(
            f"MOTIF PROXIES SHARE NO PROBLEMS with this corpus ({len(row)} labelled "
            f"against {len(key_idx)} here) — rebuild them against this corpus.")

    for k, col in enumerate(spec):
        poles = col.split("/")
        w = tail_weight_np(src[ok, rules.index(poles[0])])
        if len(poles) == 2:
            w = w - tail_weight_np(src[ok, rules.index(poles[1])])
        tgt[ok, k] = (w - w.mean()) / (w.std() + 1e-12)
    print(f"  BANK from the motif proxies: {', '.join(spec)} "
          f"({ok.sum()}/{len(key_idx)} problems taught)", flush=True)
    return tgt, ok, spec


def report_bank_identity(g_np, tgt, ok, prob_idx, k_axes=K_AXES):
    """How far each shipped axis moved off the bank — the check that says whether
    `axes/labels.json` still describes what it is attached to.

    HINGED TAIL AGREEMENT, not correlation. A name is a claim about one END of an
    axis, and the axis is a trunk output rather than a linear map, so its extremes
    and its ordering of the middle are free to diverge — measured, correlation
    reverses which of two fits held identity better and reports a bipolar column
    collapsing when its poles are both intact. Each direction is z-scored and
    floored at TAIL_HINGE, then correlated; the result is exactly zero under
    independence, so there is no chance baseline to quote. Bulk correlation is
    printed alongside as context, never as the verdict.

    BOTH ENDS are scored separately: an axis can hold the end its name is on and
    lose the other, and one number averaging the two hides exactly that. A tail
    that is constant after hinging scores NaN rather than 0 — an empty tail makes
    no claim and must not read as "measured, independent".

    Scored on ATTEMPTED problems only. A third of the corpus is never attempted;
    its loadings are unconstrained extrapolation, so their extremes churn freely
    between fits and would be counted here as a failure to reproduce. They are
    also ENRICHED in the tails rather than merely present, so including them
    understates what the model actually holds.
    """
    att = np.zeros(len(g_np), bool)
    att[np.unique(prob_idx)] = True
    sel = np.where(att & ok)[0]
    print(f"  BANK IDENTITY over {len(sel)} attempted+banked problems "
          f"(tail-weighted agreement at z >= {TAIL_HINGE}, 0 = independent):",
          flush=True)
    rows = []
    for k in range(k_axes):
        x, y = g_np[sel, k], tgt[sel, k]
        pos = hinged_r(x, y)
        neg = hinged_r(-x, -y)
        bulk = float(np.corrcoef(x, y)[0, 1])
        print(f"    axis {k}: +tail {pos:6.3f}  -tail {neg:6.3f}"
              f"   (bulk r {bulk:+.3f})", flush=True)
        rows.append({"axis": k, "pos_tail": float(pos), "neg_tail": float(neg),
                     "bulk_r": bulk})
    return rows


def run(*, epochs=EPOCHS, check_every=CHECK_EVERY, batch=BATCH, seed=SEED,
        lr_net=LR_NET, lr_field=LR_FIELD,
        v_ridge=V_RIDGE, ortho_rating=ORTHO_RATING, ortho_mode="per_tail",
        shallow_tau=SHALLOW_TAU,
        shallow_w=SHALLOW_W, shallow_samples=SHALLOW_SAMPLES,
        random_user_weight=RANDOM_USER_WEIGHT,
        strong_user_weight=STRONG_USER_WEIGHT,
        teach_weight=TEACH_WEIGHT, stop_row_frac=STOP_ROW_FRAC,
        pca_dim=PCA_DIM, blocks=PRODUCTION_BLOCKS, act_source="store",
        stop_unit="row",
        split_seed=None, pca_seed=None, init_seed=None, stream_seed=None,
        snapshot_epochs=None, stop_epoch=None,
        lr_schedule=LR_SCHEDULE,
        lr_warmup_epochs=0, ortho_axes=ORTHO_AXES,
        teach_detach=False, ortho_detach=False,
        ortho_axes_detach=ORTHO_AXES_DETACH,
        v_ridge_ladder=1.0,
        k_axes=K_AXES, dropout=DROPOUT, input_dropout=INPUT_DROPOUT,
        link_power=False, link="sigmoid", odds_sharpness=1.0, blend_frac=None, stage_frac=None, stage_lapse=False, head_zero_init=False, curve_lr_mult=None, floor_init=None, theta_laplace=THETA_LAPLACE, init_base_rate=INIT_BASE_RATE, slope_pin=None, slope_pin_w=0.01, theta_moments=None, rating_anchor=None, anchor_skip=30, curve_param="ab", lapse_fixed=None, gauge="sd",
        floor_prior=False, floor_rel=False, weight_avg=None,
        optimizer="adam", muon_lr=0.02, muon_wd=0.0, slope_height=False,
        floor_mode=FLOOR_MODE, slope_mode="trunk", hidden=None,
        lapse_mode="global",
        avg_start_frac=0.5,
        ema_decay=0.99,
        weight_decay=0.0, weight_decay_fields=None, exclude_users=None,
        out_dir=OUT_DIR, axes_dir=AXES_DIR):
    """The shipped fit, callable. Every keyword defaults to the production
    constant above, so `run()` IS the production run — `main()` passes nothing,
    keeping the trainer flag-free — and `ablations.ablation_mirt` is the only
    caller that passes anything else, always with `out_dir`/`axes_dir`
    redirected away from `ratings/` and `axes/`. The per_cell retirement acts on
    `axes_dir` itself, so a redirected run cannot touch the production store.
    Returns a metrics dict (row counts, the convergence trace, the output
    distributions, the bank-identity report) for the ablation ledger."""
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # THE FEATURE BASIS HAS ITS OWN SEED. The joint PCA is randomized and keyed
    # on the seed it is given, so one seed re-rolls the basis, the network's
    # initial weights and the minibatch order together. Production runs them all
    # off one seed and is self-consistent; a REPLICATE that varies the seed is
    # varying three things, and separating them is the only way to say which one
    # a spread between runs comes from.
    feat, keys, key_idx, input_dim = load_joint_mirt_features(
        SIZES, pca_dim, seed if pca_seed is None else pca_seed, blocks=blocks,
        act_source=act_source)
    fdim = feat.shape[1]
    print(f"loaded in {time.time() - t0:.0f}s: {len(keys)} problems, "
          f"feat {fdim} (dropout region {input_dim})", flush=True)

    # ONE ability unit per user, over their whole history. Measured best against
    # every windowed arrangement, and it makes the ability and axis-position
    # granularities coincide — removing the mixed-granularity question the
    # two-stage design has to answer rather than answering it.
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    if exclude_users:
        # Diagnostic only: drop named users (id prefixes) to see how much of a
        # fit's shape they carry. Changes the held-out draw, so an arm run with
        # it is not paired with one run without.
        uid = df["user_id"].astype(str)
        drop = np.zeros(len(df), dtype=bool)
        for pre in exclude_users:
            drop |= uid.str.startswith(pre).to_numpy()
        print(f"  exclude_users: dropping {int(drop.sum())} rows", flush=True)
        df = df[~drop].copy()
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
    # The synthetic users' rows, appended after the real ones so a real row's
    # index never moves: (weight, first row, user index) per injected user, the
    # strong user's rows marked for the ceiling scaling.
    synthetic = []
    tids_by_size = {s: [t for sz, t in keys if sz == s] for s in SIZES}
    cells = load_legal_cell_counts(SIZES, tids_by_size)
    every = torch.arange(len(keys), device=device)
    tu_all, tp_all, ty_all = [tu], [tp], [ty]
    strong_from = None
    for weight, strong in ((random_user_weight, False), (strong_user_weight, True)):
        if weight <= 0:
            continue
        if strong:
            aq = load_answer_choice(SIZES, tids_by_size)
            target = [aq[k] for k in keys]
            strong_from = sum(len(t) for t in tu_all)
        else:
            target = [1.0 / cells[k] for k in keys]
        uid = n_users + len(synthetic)
        synthetic.append((float(weight), sum(len(t) for t in tu_all), uid))
        tu_all.append(torch.full((len(keys),), uid, device=device))
        tp_all.append(every)
        ty_all.append(torch.tensor(target, dtype=torch.float32, device=device))
    tu_all, tp_all, ty_all = (torch.cat(t) for t in (tu_all, tp_all, ty_all))
    # Each real row's place in its user's history (0 = first attempt), for the
    # rating anchor; synthetic rows get 0 and are excluded there anyway.
    seq = df.groupby("user_id")["created_at"].rank(method="first").to_numpy() - 1
    t_seq = torch.zeros(len(tu_all), device=device)
    t_seq[:len(df)] = torch.tensor(seq, dtype=torch.float32, device=device)

    # A measurement for comparison arms, never a criterion: every row spent here
    # is a row the model does not learn from, so the shipped fit spends none.
    # THE HELD-ASIDE DRAW AND THE INITIALIZATION ARE SEPARABLE. One seed drives
    # both by default, which is right for production but conflates two things in
    # a comparison: two runs at different seeds differ in the fit AND in the rows
    # the fit is scored on, so the difference between their scores is not a
    # property of the fit alone. Pinning `split_seed` across arms holds the rows
    # fixed and leaves the initialization as the only thing moving.
    rng = np.random.default_rng(seed if split_seed is None else split_seed)
    if stop_unit == "problem":
        # Held-out PROBLEMS rather than held-out rows: every row of a chosen
        # problem is set aside together, so its parameters come from the trunk
        # alone and the score reads generalization to unseen problems. Only an
        # amortized model can be asked this — free per-item parameters would
        # leave a held-out problem undefined rather than merely unfitted. It is
        # what a claim about the FEATURE side has to be scored on; the row split
        # below, which ships, is a convergence detector and leaves every problem
        # partly trained.
        held_problem = rng.random(len(keys)) < stop_row_frac
        is_stop = held_problem[prob_idx]
    elif stop_unit == "spread":
        # TWO DISJOINT HELD-OUT SETS, both among problems attempted often
        # enough that several different users' outcomes on one problem can be
        # compared — a problem held out with one attempt says nothing about its
        # own departure from the curve, only about the population's.
        #   set 0: whole problems out, so their parameters come from the
        #          features alone — the position of every never-attempted one.
        #   set 1: half of each problem's rows out and half kept, so the fit
        #          has seen the problem — what its own attempts correct.
        # The users stay in through their other attempts, so their abilities
        # are fitted without the scored rows; that is what an after-the-fact
        # holdout cannot give. Row-level draws stay on `rng`, so the draw is a
        # function of the split seed alone.
        cnt = np.bincount(prob_idx, minlength=len(keys))
        dense = rng.permutation(np.where(cnt >= SPREAD_MIN_ATTEMPTS)[0])
        n_set = int(round(SPREAD_FRAC * len(dense)))
        whole, halved = dense[:n_set], dense[n_set:2 * n_set]
        spread_set = np.full(len(df), -1, np.int8)
        spread_set[np.isin(prob_idx, whole)] = 0
        rows_by_p = pd.Series(np.arange(len(df))).groupby(prob_idx)
        for p in halved:
            r = rows_by_p.get_group(p).to_numpy()
            spread_set[rng.choice(r, len(r) // 2, replace=False)] = 1
        is_stop = spread_set >= 0
        held_problem = np.zeros(len(keys), bool)
        held_problem[whole] = True
    else:
        is_stop = rng.random(len(df)) < stop_row_frac
        held_problem = np.zeros(len(keys), bool)
    # A problem held out WHOLE is held out of the synthetic users too: one row
    # of theirs on it and its parameters are no longer the trunk's alone.
    seen = torch.tensor(np.where(~held_problem)[0], device=device)
    tr = torch.tensor(np.where(~is_stop)[0], device=device)
    i_stop = torch.tensor(np.where(is_stop)[0], device=device)
    has_stop = bool(is_stop.any())
    print(f"  stop signal: {int(is_stop.sum())} rows held aside", flush=True)

    bank_tgt, bank_ok, bank_spec = load_bank(key_idx, k_axes)
    t_tgt = torch.tensor(bank_tgt, dtype=torch.float32, device=device)
    t_ok = torch.tensor(bank_ok, device=device)
    print(f"  axis bank: {int(bank_ok.sum())}/{len(keys)} problems carry banked "
          f"directions, taught at a constant weight of {teach_weight}",
          flush=True)

    floor_cols = None
    if floor_mode == "weak_human":
        import re
        from utils.rating.features import bot_winrate_column_names
        names = bot_winrate_column_names(SIZES)
        if len(names) + 1 != fdim - PCA_DIM:
            raise ValueError(f"{len(names)} bot files against "
                             f"{fdim - PCA_DIM - 1} bot columns")
        floor_cols = [i for i, n in enumerate(names)
                      if re.search(r"humanv0_\d+k(_t[0-9.]+)?\.txt$", n)]
        print(f"  floor reads {len(floor_cols)} weak human-bot columns",
              flush=True)
    m = JointMIRT(fdim, n_users + len(synthetic), input_dim,
                  seed if init_seed is None else init_seed, k_axes,
                  dropout=dropout, input_dropout=input_dropout,
                  link_power=link_power, link=link, odds_sharpness=odds_sharpness,
                  blend_frac=blend_frac, stage_frac=stage_frac, stage_lapse=stage_lapse,
                  curve_lr_mult=curve_lr_mult, floor_init=floor_init,
                  head_zero_init=head_zero_init,
                  curve_param=curve_param,
                  lapse_fixed=lapse_fixed, gauge=gauge,
                  floor_prior=floor_prior, floor_rel=floor_rel,
                  slope_height=slope_height, floor_mode=floor_mode,
                  slope_mode=slope_mode, floor_cols=floor_cols,
                  hidden=hidden, lapse_mode=lapse_mode).to(device)
    # The training stream — minibatch order and dropout masks — runs on the
    # global RNG the model's initialisation left behind, so by default it is a
    # function of the init seed. `stream_seed` re-seeds it after construction,
    # separating the two for the replicate study.
    if stream_seed is not None:
        torch.manual_seed(stream_seed)
    if init_base_rate:
        # EXPERIMENT: start at the population's own clearance. One shift is
        # added to every real user's ability, found by bisection so that the
        # mean predicted success over a batch of training rows equals the
        # observed base rate; otherwise the fit starts near 0.5 against data
        # near 0.8 and the floor is the cheapest intercept to close the gap.
        # Run in training mode (the gauge standardises b as in training) with
        # the RNG state and the gauge buffers restored afterwards, so the
        # training stream is unchanged.
        g_init = torch.Generator(device=device).manual_seed(0)
        rows0 = tr[torch.randperm(len(tr), generator=g_init, device=device)[:batch]]
        target = float(ty_all[tr].mean())
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state() if device.type == "cuda" else None
        bufs = {k: v.clone() for k, v in m.named_buffers()}
        th0 = m.theta.weight.data[:n_users].clone()
        lo, hi = -10.0, 10.0
        with torch.no_grad():
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                m.theta.weight.data[:n_users] = th0 + mid
                torch.set_rng_state(cpu_state)
                if cuda_state is not None:
                    torch.cuda.set_rng_state(cuda_state)
                pm = float(m(tu_all[rows0], tfeat[tp_all[rows0]])[0].mean())
                lo, hi = (mid, hi) if pm < target else (lo, mid)
            m.theta.weight.data[:n_users] = th0 + 0.5 * (lo + hi)
            for k, v in m.named_buffers():
                v.copy_(bufs[k])
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        print(f"  init_base_rate: ability shift {0.5 * (lo + hi):+.3f} "
              f"(predicted {pm:.3f}, target {target:.3f})", flush=True)
    if theta_laplace:
        # EXPERIMENT: a population prior on ability whose variance is the
        # data's own, by a Laplace-EM step each epoch: sigma^2 = mean over
        # users of (theta - mean)^2 + 1 / I_u, with I_u each user's Fisher
        # information summed over their rows in the previous epoch. A sparse
        # user's large 1/I keeps sigma^2 from collapsing (the failure of
        # maximising jointly over abilities and their variance), heavy users
        # are barely touched, and a user parked far from everyone is priced.
        # No weight is set; sigma^2 is detached (the M-step).
        info_acc = torch.zeros(n_users, device=device)
        sig2 = None
    prior_ratio = len(keys) / len(df)
    # Adam with no decay is what ships; the decay knob exists so 1.5's claim
    # that a norm penalty does not substitute for dropout can be re-measured,
    # and at 0 this is the same optimizer it always was.
    # Weight decay is set PER GROUP. Adam's single argument reaches the free
    # per-user fields too — an L2 on ability, which the shipped design leaves
    # unpenalized, and which re-gauges the whole fit (abilities shrink, slopes
    # inflate to compensate). `weight_decay_fields` exists so an ablation can
    # decay the network alone; None means the fields follow `weight_decay`,
    # which is what Adam's own argument would have done.
    groups = m.param_groups(lr_net, lr_field)
    groups[0]["weight_decay"] = weight_decay
    for extra in groups[2:]:
        extra["weight_decay"] = weight_decay
    groups[1]["weight_decay"] = (weight_decay if weight_decay_fields is None
                                 else weight_decay_fields)
    if optimizer == "muon":
        mats = [prm for n, prm in m.trunk.named_parameters() if prm.ndim == 2]
        ids = {id(q) for q in mats}
        for grp in groups:
            grp["params"] = [q for q in grp["params"] if id(q) not in ids]
        opt = _Both(torch.optim.Adam(groups),
                    Muon(mats, lr=muon_lr, weight_decay=muon_wd))
    elif optimizer == "adam":
        opt = torch.optim.Adam(groups)
    elif optimizer == "adamw":
        # Decoupled decay: each step shrinks the weights by lr * decay directly,
        # instead of adding decay * weight to a gradient Adam then rescales.
        opt = torch.optim.AdamW(groups)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    o_tau = float(np.log(np.expm1(shallow_tau)))
    # THE COSINE DECAY SHIPS; `constant` is the comparison arm and leaves every
    # step at the base rate. A constant rate keeps taking full-size steps long
    # after the fit has arrived, which is what a run wandering inside a wide band
    # at its end looks like, and the shipped value is then whichever point of
    # the band the last epoch lands on. The decay removes that draw: the fit is
    # frozen where it stands as the rate reaches zero.
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}")
    base_lrs = [g["lr"] for g in opt.param_groups]
    n = len(tr)
    trace = []
    print(f"\n=== joint fit | K={k_axes} | teach weight {teach_weight} "
          f"(constant) ===", flush=True)

    # THE LEARNING-RATE INTEGRAL, in constant-rate epochs. A decayed run does
    # less total optimization than its epoch count suggests — a cosine to zero
    # averages half the base rate, so it delivers what half as many constant
    # epochs would. Comparing two schedules at equal EPOCHS therefore compares
    # them at unequal work, and against a fit still descending that reads as the
    # schedule being worse when it was only shorter. Recorded so a panel can
    # match on it.
    # NO WARMUP SHIPS. The ramp is an ablation knob and nothing else: with the
    # axis objective indifferent to a rotation, where a free fit lands is decided
    # by its early path, and a gentler opening is the obvious candidate for
    # making two runs land closer. At zero the schedule below is exactly the
    # shipped one — the multiplier is applied only inside the ramp, so the
    # decay itself is untouched and the recorded rate integral says what the
    # ramp cost in work.
    if lr_warmup_epochs < 0:
        raise ValueError(f"lr_warmup_epochs must be >= 0, got {lr_warmup_epochs}")
    lr_epochs = 0.0
    # EXPERIMENT (None as shipped): WEIGHT AVERAGING, so what ships is not the
    # last epoch's weights but an average along the run. "swa" is the uniform
    # mean of the end-of-epoch weights from avg_start_frac of the budget on;
    # "ema" is an exponential moving average of them at ema_decay per epoch.
    # Every float tensor of the state is averaged, buffers included.
    if weight_avg not in (None, "swa", "ema"):
        raise ValueError(f"unknown weight_avg {weight_avg!r}")
    avg_state, avg_n = None, 0
    avg_from = int(avg_start_frac * epochs)

    def _swap_in(state):
        live = {k: v.detach().clone() for k, v in m.state_dict().items()}
        m.load_state_dict(state)
        return live

    for ep in range(epochs):
        if theta_laplace:
            if ep > 0:
                with torch.no_grad():
                    th = m.theta.weight[:n_users, 0]
                    sig2 = float(((th - th.mean()) ** 2
                                  + 1.0 / info_acc.clamp(min=1e-3)).mean())
                if ep % 1000 == 1:
                    print(f"  laplace prior: sigma {sig2 ** 0.5:.3f}, ability sd "
                          f"{float(th.std()):.3f}", flush=True)
            info_acc.zero_()
        if stop_epoch is not None and ep >= stop_epoch:
            # Diagnostic: end the run here with the schedule still that of the
            # full budget, so the path so far is the full run's path.
            break
        if blend_frac:
            m.blend_w.fill_(min(1.0, ep / max(blend_frac * epochs, 1.0)))
        if snapshot_epochs and ep in snapshot_epochs:
            # Diagnostic: the per-problem triple at this epoch, read at eval
            # (the gauges' EMAs, nothing ticked), for following when two
            # replicates' floors and slopes part ways.
            m.train(False)
            with torch.no_grad():
                sb, sa, sc, _ = m.problem_params(tfeat)
                sl = float(m._last_lam.mean()) if torch.is_tensor(m._last_lam) else float(m._last_lam)
            m.train(True)
            os.makedirs(out_dir, exist_ok=True)
            np.savez(os.path.join(out_dir, f"snap_{ep}.npz"),
                     b=sb.cpu().numpy(), a=sa.cpu().numpy(),
                     c=sc.cpu().numpy(), lam=sl,
                     theta=m.theta.weight.detach().cpu().numpy().ravel())
        if stage_frac and ep == int(stage_frac * epochs):
            m.start_stage2()
            print(f"  stage 2 from epoch {ep}: global floor "
                  f"{float(torch.sigmoid(m.floor_g)):.3f} (pre-cap), slope "
                  f"{float(nn.functional.softplus(m.slope_g)):.3f}", flush=True)
        if lr_schedule == "cosine":
            # Standard cosine from the base rate to zero over the whole budget,
            # applied to BOTH groups so their ratio — the thing 1.8 measures —
            # is held fixed all the way down.
            scale = 0.5 * (1.0 + float(np.cos(np.pi * ep / max(epochs - 1, 1))))
        else:
            scale = 1.0
        if ep < lr_warmup_epochs:
            # Linear from base/warmup at the first epoch to base at the ramp's
            # end; a zero-rate epoch would be a wasted one.
            scale *= (ep + 1) / lr_warmup_epochs
        for grp, base in zip(opt.param_groups, base_lrs):
            grp["lr"] = base * scale
        lr_epochs += scale
        m.train(True)
        rows = [tr]
        for weight, first, _uid in synthetic:
            take = int(round(weight * len(seen)))
            rows.append(first + seen[torch.randperm(len(seen), device=device)[:take]])
        perm = torch.cat(rows)
        perm = perm[torch.randperm(len(perm), device=device)]
        n = len(perm)
        # THE REMAINDER IS DROPPED, after the shuffle, so every step is a full
        # batch and a different handful of rows sits out each epoch. The gauges
        # and several penalties are this batch's own statistics and every step
        # is taken at full size, so a stub of a few thousand rows is one noisy
        # full-weight step per epoch — and how small the stub is depends on
        # nothing but the row count modulo the batch size.
        for i in range(0, (n // batch) * batch or n, batch):
            bi = perm[i:i + batch]
            pred, gh, rsf = m(tu_all[bi], tfeat[tp_all[bi]])
            pred = pred.clamp(EPS, 1 - EPS)
            yb = ty_all[bi]
            if strong_from is not None:
                yb = torch.where(bi >= strong_from,
                                 (1.0 - m.lapse().detach()) * yb, yb)
            loss = torch.nn.functional.binary_cross_entropy(
                pred, yb, reduction="none").sum() / len(bi)
            if theta_laplace:
                with torch.no_grad():
                    a_r, c_r = rsf[:, 1], rsf[:, 2]
                    lam_r = m._last_lam
                    span = (1.0 - c_r - lam_r).clamp(min=1e-6)
                    sz = ((pred - c_r) / span).clamp(1e-6, 1 - 1e-6)
                    dp = span * a_r * sz * (1 - sz)
                    fi = dp ** 2 / (pred * (1 - pred))
                    uu = tu_all[bi]
                    real = uu < n_users
                    info_acc.index_add_(0, uu[real], fi[real])
                if sig2 is not None:
                    th = m.theta.weight[:n_users, 0]
                    loss = loss + (0.5 * (th - th.mean().detach()) ** 2
                                   ).sum() / sig2 / n
            if rating_anchor:
                # EXPERIMENT: each attempt pulls its user's ability toward the
                # problem's RATING (the 80% point, held fixed here), so a user
                # is drawn to the average rating of what they were served —
                # where a user clearing about 80% belongs — whatever the slope
                # and floor do. Per attempt, so the pull grows with a user's
                # attempts exactly as the likelihood does; a parked user, far
                # from every problem they played, pays quadratically. Rows
                # whose problem has no 80% point inside the curve are left out, and
                # so are the synthetic users, who belong far from the problems,
                # and each user's first `anchor_skip` attempts: before serving
                # has found a user's level, what they are served says little
                # about where they are (clearance settles near 0.78 from about
                # the 30th first-time attempt on).
                r_row, c_row = rsf[:, 0], rsf[:, 2]
                q_row = (CALIBRATION_P - c_row) / (1.0 - c_row - m._last_lam)
                ok = ((q_row > 0.01) & (q_row < 0.99) & (tu_all[bi] < n_users)
                      & (t_seq[bi] >= anchor_skip))
                th_row = m.theta(tu_all[bi]).squeeze(-1)
                loss = loss + rating_anchor * (
                    (th_row - r_row) ** 2 * ok).sum() / len(bi)
            if slope_pin:
                # EXPERIMENT: a gentle pull of the batch's mean log slope toward
                # log(slope_pin), so floor and slope cannot trade the slope's
                # centre away, and an ability penalty cannot be met by shrinking
                # the ability scale while the slopes grow to compensate.
                loss = loss + slope_pin_w * (
                    torch.log(m._last_a.clamp(min=1e-6)).mean()
                    - float(np.log(slope_pin))) ** 2
            if theta_moments:
                # EXPERIMENT: a squared penalty pulling the abilities' mean and
                # spread toward zero, which the likelihood resists — no target
                # spread is asserted. Nothing else prices an ability's distance
                # from everyone, so a user can otherwise be parked arbitrarily
                # far below every problem and explained by the floors; under
                # this a far outlier inflates the spread quadratically and is
                # held near the edge, where the fit can still pull it back.
                # mean^2 + var is the population's second moment, i.e. a ridge
                # of the same weight on every user. (A learned per-user spread
                # was tried first and collapses every ability to one value.)
                th_all = m.theta.weight[:n_users, 0]
                loss = loss + theta_moments * (th_all.mean() ** 2
                                               + th_all.var(unbiased=False))
            # SHRINK WHAT THE LIKELIHOOD READS. user_axis, not the raw
            # parameter: the two differ by the population mean, which is a
            # direction no prediction depends on, so a ridge on raw v spends
            # itself on a gauge instead of on the positions the model uses.
            # Measured, that is not a nicety — under the raw form the fitted
            # positions come out collinear across users and some axes are
            # starved to zero spread, while shrinking the centred positions
            # leaves the axes distinct, the spread uniform across all of them,
            # and retention and never-trained pricing both better.
            #
            # The mean is then unconstrained by anything, and does not need to
            # be: the gradient along it is analytically zero, and it stays at
            # ~0.002-0.009 over a full run against a per-axis spread an order of
            # magnitude larger.
            if v_ridge_ladder == 1.0:
                loss = loss + v_ridge * (m.user_axis(tu_all[bi]) ** 2).sum() / len(bi)
            else:
                # ANISOTROPIC, an ablation path: a distinct ridge per axis,
                # geometric in the axis index, breaks the objective's
                # indifference to rotating the axis space — the least-penalized
                # axis takes the direction of largest between-user variance —
                # which is the one in-objective candidate for identity without
                # a taught target. It does not price duplication: two axes
                # sharing a direction still pay less than one carrying it
                # unless their ridges differ threefold.
                ladder = v_ridge * v_ridge_ladder ** torch.arange(
                    k_axes, device=device, dtype=torch.float32)
                loss = loss + (ladder * m.user_axis(tu_all[bi]) ** 2).sum() / len(bi)
            # DECORRELATE THE AXES FROM DIFFICULTY. Nothing else does: centring
            # v (user_axis) removes only the POPULATION mean, which blocks an
            # axis acting as a global difficulty shift but not one tracking
            # difficulty per problem. Left free, a fit spends an axis on a rating-side property
            # and that axis is unnameable when it does.
            #
            # b is DETACHED, so this can only push the axes off difficulty and
            # never degrade the rating to satisfy itself. The direction matters:
            # the client's rank path reads rating, slope and floor and never an
            # axis, so difficulty living in an axis is signal the product cannot
            # reach.
            #
            # THE SHIPPED TRIPLE, not difficulty alone. Stopping at b only
            # DISPLACES the leak: measured, closing b sent one axis from -0.04 to
            # -0.43 on the floor head and left another at +0.62 with slope, since
            # each axis simply finds a different rating parameter to absorb.
            # Closing all three removes it — every correlation under 0.13, slope
            # and floor under 0.04 — and on a FREE fit costs nothing: 6.4 mnat
            # BETTER on rating than b-only, with the axes still worth 3.7 mnat.
            #
            # THE BANK MUST BE RATING-ORTHOGONAL FOR THIS TO WORK. Teaching
            # directions that are themselves rating-correlated while pushing the
            # axes off every rating channel is over-determined, and the axes lose
            # to the conflict rather than to the constraint — measured at 0.16
            # mnat against 4.41. Directions taken from a fit under this same
            # penalty carry no such conflict.
            #
            # Each column is centred and normalised, so the term is a sum of
            # squared cosines: floor lives on (0, 1-2*lambda) and rating on its own
            # scale, and an unnormalised penalty would be dominated by whichever
            # happens to be widest. The targets are DETACHED, so this can only
            # push the axes off the rating side and never degrade the rating to
            # satisfy itself.
            # DETACHED READOUTS: each of the three terms on the loadings can be
            # pointed at the readout alone, so that it steers which directions
            # of the trunk's representation the axes read out without reshaping
            # the representation the rating also reads. The mutual term ships
            # that way; the pull and the rating-orthogonality penalty are
            # measured to need the trunk, and their detachments are ablation
            # paths.
            gh_d = m.detached_loadings() if (teach_detach or ortho_detach
                                             or ortho_axes_detach) else None
            gh_o = gh_d if ortho_detach else gh
            gh_m = gh_d if ortho_axes_detach else gh
            gh_t = gh_d if teach_detach else gh
            tz = rsf - rsf.mean(dim=0, keepdim=True)
            tz = tz / (tz.norm(dim=0, keepdim=True) + 1e-9)
            # PER TAIL, not whole-axis. A whole-axis correlation is a LINEAR
            # statistic and cannot see a symmetric dependence: an axis whose two
            # tails are both rating-laden in opposite directions averages to zero
            # and satisfies the penalty completely, while both NAMED tails — the
            # only part the client reads — are laden. Measured on a free fit whose
            # whole-axis correlations sat at 0.02-0.09, the tails carried rating at
            # 0.27-0.31; penalising per tail cuts the worst tail from 0.291 to
            # 0.105. The whole-axis figure rises slightly under this and that is
            # expected — it is no longer the statistic being optimised, and it was
            # never the one that mattered.
            if ortho_mode in ("per_tail", "rating_only"):
                # `rating_only` keeps the per-tail form and narrows the TARGET
                # to the rating column alone (rsf is stacked rating, slope,
                # floor). It is the comparison arm for 1.5's claim that closing
                # the rating channel by itself only displaces the leak, each
                # axis then finding a different shipped quantity to absorb —
                # which is the whole reason the penalty names all three.
                tgt = tz if ortho_mode == "per_tail" else tz[:, :1]
                for _sg in (1.0, -1.0):
                    W = _tail_w(torch, _sg * gh_o, TAIL_HINGE)
                    W = W - W.mean(dim=0, keepdim=True)
                    W = W / (W.norm(dim=0, keepdim=True) + 1e-9)
                    loss = loss + ortho_rating * (W.T @ tgt).pow(2).sum()
            elif ortho_mode == "whole":
                # The whole-axis form, kept ONLY as a comparison arm: a linear
                # statistic that a symmetric tail dependence satisfies at zero,
                # which is the failure the per-tail form above exists to close.
                W = gh_o - gh_o.mean(dim=0, keepdim=True)
                W = W / (W.norm(dim=0, keepdim=True) + 1e-9)
                loss = loss + ortho_rating * (W.T @ tz).pow(2).sum()
            elif ortho_mode != "off":
                raise ValueError(f"unknown ortho_mode {ortho_mode!r}")
            # MUTUAL ORTHOGONALITY, on the detached readout in the shipped fit
            # (ORTHO_AXES / ORTHO_AXES_DETACH above). Nothing above asks the
            # axes to differ from EACH OTHER, and the pull alone does not hold
            # seven distinct directions. Whole-axis, unlike the rating term: a
            # duplicated axis IS a whole-axis correlation of one, with or
            # without a sign, so the linear statistic is the right one here.
            if ortho_axes > 0:
                Wm = gh_m - gh_m.mean(dim=0, keepdim=True)
                Wm = Wm / (Wm.norm(dim=0, keepdim=True) + 1e-9)
                Cm = Wm.T @ Wm
                loss = loss + ortho_axes * (
                    Cm - torch.eye(k_axes, device=Cm.device)).pow(2).sum()
            # TEACHING, ON THE ATTEMPT STREAM, reusing the gh this batch already
            # produced — no second forward and no second sampling distribution.
            #
            # Both properties are load-bearing. Sampling problems separately
            # makes the trunk adapt to a MIXTURE of two input distributions
            # rather than to the one the rest of the objective sees; riding gh
            # means no second distribution exists at all.
            # It also matches the population the axes are DEFINED on: loadings
            # are identified by between-user residuals, which exist only where
            # somebody attempted the problem. The never-attempted third of the
            # corpus is left alone deliberately — its banked values are the
            # previous fit's extrapolation into a region that never carried
            # signal, so teaching them would spend trunk capacity, in competition
            # with the rating, reproducing an arbitrary guess.
            mk = t_ok[tp_all[bi]]
            if bool(mk.any()):
                # Both sides standardized, because the bank's scale is
                # arbitrary and gh carries this fit's own — only direction is
                # being compared. The two standardizations are over DIFFERENT
                # populations on purpose and should not be reconciled: gh is
                # z-scored over the batch, so attempt-weighted and repeating a
                # popular problem, while the target was z-scored over the
                # shared corpus unweighted. That leaves the MSE a small
                # irreducible floor and pulls g toward a slightly shifted
                # bank, which costs nothing downstream — problem_params
                # divides by g_rms, so any residual scale is removed before
                # anything reads a loading, and v absorbs the rest.
                gz = (gh_t - gh_t.mean(0)) / (gh_t.std(0) + 1e-6)
                loss = loss + teach_weight * (
                    (gz[mk] - t_tgt[tp_all[bi]][mk]) ** 2).mean()
            m.train(False)          # the barrier reads corpus rows, dropout off
            si = torch.randint(len(tfeat), (shallow_samples,), device=device)
            if m.floor_prior:
                o_s = m.head(m.trunk(tfeat[si]))
                short = torch.nn.functional.softplus(
                    o_tau - o_s[:, 1]).pow(2).mean()
                # Negative log prior per problem, weighted problems-to-rows so
                # that over an epoch the prior counts each problem once against
                # each row's likelihood counting once.
                zf = (o_s[:, 2] - m.floor_prior_mu) * torch.exp(
                    -m.floor_prior_logsd)
                loss = loss + prior_ratio * (
                    0.5 * zf.pow(2) + m.floor_prior_logsd).mean()
            else:
                short = torch.nn.functional.softplus(
                    o_tau - m.slope_logit(tfeat[si])).pow(2).mean()
            loss = loss + shallow_w * short
            m.train(True)
            opt.zero_grad(); loss.backward()
            opt.step()

        if weight_avg is not None:
            with torch.no_grad():
                cur = m.state_dict()
                if weight_avg == "ema":
                    if avg_state is None:
                        avg_state = {k: v.detach().clone()
                                     for k, v in cur.items()}
                    else:
                        for k, v in cur.items():
                            if v.is_floating_point():
                                avg_state[k].mul_(ema_decay).add_(
                                    v.detach(), alpha=1.0 - ema_decay)
                elif ep >= avg_from:
                    avg_n += 1
                    if avg_state is None:
                        avg_state = {k: v.detach().clone()
                                     for k, v in cur.items()}
                    else:
                        for k, v in cur.items():
                            if v.is_floating_point():
                                avg_state[k].add_(
                                    (v.detach() - avg_state[k]) / avg_n)

        if (ep + 1) % check_every == 0:
            m.train(False)
            with torch.no_grad():
                pr, _, _ = m(tu, tfeat[tp])
                pr = pr.clamp(EPS, 1 - EPS)
                bce_tr = float(torch.nn.functional.binary_cross_entropy(
                    pr[tr], ty[tr]))
                bce_st = float(torch.nn.functional.binary_cross_entropy(
                    pr[i_stop], ty[i_stop])) if has_stop else None
                bce_avg = None
                if avg_state is not None and has_stop:
                    live = _swap_in(avg_state)
                    pa, _, _ = m(tu[i_stop], tfeat[tp[i_stop]])
                    bce_avg = float(torch.nn.functional.binary_cross_entropy(
                        pa.clamp(EPS, 1 - EPS), ty[i_stop]))
                    m.load_state_dict(live)
            print(f"  epoch {ep + 1:4d}/{epochs}  train {bce_tr:.4f}"
                  + (f"  stop {bce_st:.4f}" if has_stop else "")
                  + (f"  stop(avg) {bce_avg:.4f}" if bce_avg is not None
                     else ""), flush=True)
            trace.append({"epoch": ep + 1, "train_bce": bce_tr,
                          "stop_bce": bce_st, "stop_bce_avg": bce_avg})

    if avg_state is not None:
        # Ship the averaged weights. The gauge buffers are averaged with them
        # and kept: they are training-mode statistics, and re-measuring them in
        # eval mode would rescale the difficulty under abilities and slopes
        # fitted against the training-mode scale.
        m.load_state_dict(avg_state)
        print(f"  shipped the {weight_avg} average of the weights", flush=True)

    # WHAT THE AXES ADD, on the held-aside rows, measured before the gauge is
    # pinned so both halves come from the state the last trace entry describes.
    # Zeroing the axis term rather than refitting without it is deliberate: the
    # question is what the axis channel contributes to THIS fit's predictions,
    # and a separate K=0 fit would answer a different one, its rating having
    # absorbed whatever the axes were carrying.
    #
    # Reported and never optimized. The increment rises with the axis count
    # whether or not the model predicts better, since more axes mechanically
    # absorb more residual, so it is a diagnostic beside the combined score and
    # never a criterion for choosing the count.
    m.train(False)
    bce_stop_no_axes, slope_sweep = None, []
    with torch.no_grad():
        # Scored only where rows were held aside; the shipped fit holds none.
        if has_stop:
            b_s, a_s, c_s, _ = m.problem_params(tfeat[tp[i_stop]])
            th_s = m.theta(tu[i_stop]).squeeze(-1)
            lam_s = m.lapse()
            z_s = (a_s * (th_s - b_s)).clamp(-35, 35)
            p_noax = (c_s + (1.0 - c_s - lam_s) * torch.sigmoid(z_s)).clamp(
                EPS, 1 - EPS)
            bce_stop_no_axes = float(torch.nn.functional.binary_cross_entropy(
                p_noax, ty[i_stop]))

            # WHAT A GLOBAL SLOPE MULTIPLIER WOULD BUY (1.10), swept here because
            # the sweep is a rescoring of this fit rather than a fit of its own.
            # The multiplier PIVOTS AT THE RATING, so the 80%-clear anchor is held
            # and only the steepness around it moves — at s=1 the expression is
            # algebraically the model's own, which is what makes the curve's
            # minimum comparable to the shipped value.
            #
            # Scored on the same held-aside rows as everything else: those rows are
            # out of the fit entirely, while the abilities they are scored with come
            # from each user's OTHER attempts, which is the position the client is
            # in when it predicts an attempt from a history. Reported, never applied.
            g_s = m.user_axis(tu[i_stop])
            _, _, _, gl_s = m.problem_params(tfeat[tp[i_stop]])
            ax_s = (gl_s * g_s).sum(-1)
            kap_s = torch.log(
                ((CALIBRATION_P - c_s) / (1.0 - c_s - lam_s)).clamp(EPS, 1 - EPS)
                / (1.0 - ((CALIBRATION_P - c_s) / (1.0 - c_s - lam_s)).clamp(
                    EPS, 1 - EPS)))
            rat_s = b_s + kap_s / a_s.clamp(min=1e-6)
            slope_sweep = []
            for mult in [round(0.60 + 0.05 * i, 2) for i in range(17)]:
                z = ((a_s * mult) * (th_s - rat_s) + kap_s + ax_s).clamp(-35, 35)
                pz = (c_s + (1.0 - c_s - lam_s) * torch.sigmoid(z)).clamp(
                    EPS, 1 - EPS)
                slope_sweep.append(
                    {"mult": mult,
                     "stop_bce": float(torch.nn.functional.binary_cross_entropy(
                         pz, ty[i_stop]))})

    # THE SPREAD HOLDOUT'S ROWS, for the producer that reads them. The full
    # prediction, axes included, and its derivative in the logit — a problem
    # offset u moves P by about w*u, which is what converts a covariance of
    # outcomes into a variance on the logit. Taken before the gauge is pinned,
    # from the same state the stop score describes.
    spread_rows = None
    if stop_unit == "spread":
        with torch.no_grad():
            b_h, a_h, c_h, g_h = m.problem_params(tfeat[tp[i_stop]])
            lam_h = m.lapse()
            th_h = m.theta(tu[i_stop]).squeeze(-1)
            z_h = (a_h * (th_h - b_h)
                   + (g_h * m.user_axis(tu[i_stop])).sum(-1)).clamp(-35, 35)
            sg_h = torch.sigmoid(z_h)
            p_h = (c_h + (1.0 - c_h - lam_h) * sg_h)
            w_h = (1.0 - c_h - lam_h) * sg_h * (1.0 - sg_h)
        stop_np = i_stop.cpu().numpy()
        spread_rows = {"set": spread_set[stop_np],
                       "tid": np.array([keys[i][1] for i in prob_idx[stop_np]]),
                       "user_id": np.asarray(users.categories)[user_idx[stop_np]].astype(str),
                       "y": y[stop_np],
                       "z": z_h.cpu().numpy(), "floor": c_h.cpu().numpy(),
                       "lapse": np.float32(float(lam_h)),
                       "p": p_h.cpu().numpy(), "w": w_h.cpu().numpy()}

    # --- outputs
    m.train(False)
    if blend_frac:
        m.blend_w.fill_(1.0)
    with torch.no_grad():
        # PIN THE AXIS GAUGE TO THE CORPUS before anything is written. Its EMA
        # tracks the attempt stream, which over-weights whatever is popular,
        # while the vectors being shipped — per_problem.csv here and every legal
        # cell in write_mirt_per_cell — cover the corpus. That is the population
        # their scale should be defined on, and writing it into the buffer is
        # what carries the same divisor to the fanout. The scale itself is
        # arbitrary (v absorbs it), so what matters is that both ends agree and
        # that the number is stated rather than left over from the last call.
        b_all, a_all, c_all, g_all = m.problem_params(tfeat, pin_gauge=True)
        lam = float(m.lapse())
    b_np, a_np = b_all.cpu().numpy(), a_all.cpu().numpy()
    c_np, g_np = c_all.cpu().numpy(), g_all.cpu().numpy()
    # The ability at which the curve reaches the calibration point. AN EXTREME
    # RATING IS NOT A SYMPTOM: the corpus holds genuinely trivial problems and a
    # KataGo-hard tail nobody clears, and the rank coordinate is unbounded. What
    # distinguishes a real difficulty from a collapsed slope is the SLOPE
    # distribution and the share at the clamp, printed below.
    rating = b_np + clear_logit_vec(c_np, lam) / np.maximum(a_np, 1e-6)
    theta_all = m.theta.weight.detach().cpu().numpy().ravel()
    theta = theta_all[:n_users]
    theta_synthetic = {("strong" if first == strong_from else "random"):
                       float(theta_all[uid]) for _w, first, uid in synthetic}

    print(f"  lambda {lam:.5f} | slope p[1,50,99] "
          f"{np.percentile(a_np, [1, 50, 99]).round(3)} | at floor "
          f"{np.mean(a_np <= SLOPE_FLOOR * 1.01):.2%} | rating p[1,50,99] "
          f"{np.percentile(rating, [1, 50, 99]).round(2)}", flush=True)
    identity = report_bank_identity(g_np, bank_tgt, bank_ok, prob_idx, k_axes)

    os.makedirs(out_dir, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(out_dir, str(s))
        os.makedirs(sdir, exist_ok=True)
        for name, arr in (("tsumego_ratings.csv", rating), ("tsumego_slopes.csv", a_np),
                          ("tsumego_floors.csv", c_np)):
            col = name.split("_")[1].split(".")[0].rstrip("s")
            pd.DataFrame({"tsumego_id": tids, col: arr[sel]}).sort_values(
                col, ascending=False).to_csv(os.path.join(sdir, name), index=False)
        print(f"  size {s}: wrote {len(tids)} ratings | rating "
              f"[{rating[sel].min():.2f},{rating[sel].max():.2f}] | floor mean "
              f"{c_np[sel].mean():.3f}", flush=True)

    if spread_rows is not None:
        np.savez_compressed(os.path.join(out_dir, "spread_rows.npz"),
                            **spread_rows)

    natt = pd.Series(user_idx).value_counts()
    pd.DataFrame({"profile": list(users.categories)
                  + [f"synthetic:{k}" for k in theta_synthetic],
                  "theta": list(theta) + list(theta_synthetic.values()),
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_users)]
                  + [0] * len(theta_synthetic)}
                 ).to_csv(os.path.join(out_dir, "user_theta.csv"), index=False)
    with open(os.path.join(out_dir, "rating_constants.json"), "w") as f:
        json.dump({"fat_finger_rate": round(lam, 4), "slope_scale": 1.0,
                   "scale_fit": "none_shipped_as_fitted"}, f, indent=2)

    os.makedirs(axes_dir, exist_ok=True)
    rows = {"tsumego_id": [k[1] for k in keys], "size": [k[0] for k in keys]}
    for k in range(k_axes):
        rows[f"embed_{k}"] = g_np[:, k].round(5)
    pd.DataFrame(rows).to_csv(os.path.join(axes_dir, "per_problem.csv"),
                              index=False)
    # The content hash the PWA's embedding-invalidation gate rides: it must move
    # when and only when the projection would give different vectors. So it is
    # taken over EXACTLY what the projection reads — the trunk, the axis map and
    # the RMS the loadings are divided by (write_mirt_per_cell reruns all three
    # per cell) — and over nothing else. `head` is not in that path at all, and
    # the trunk was missing from it: two fits sharing a g_lin and differing in
    # the trunk give different vectors under one version, which no client check
    # can catch. Keys are sorted so the digest does not ride dict order.
    h = hashlib.sha256()
    proj = dict(m.trunk.state_dict())
    proj["g_lin.weight"] = m.g_lin.weight
    proj["g_rms"] = m.g_rms
    for key in sorted(proj):
        h.update(key.encode())
        h.update(np.ascontiguousarray(
            proj[key].detach().cpu().numpy(), dtype=np.float32).tobytes())
    ev = h.hexdigest()[:16]
    # axes/axes.npz is where packaging reads the shipped embedding_version, so
    # this must write THERE: a version stamped from one fit onto another fit's
    # embeddings is exactly the mismatch the per_cell retirement above prevents,
    # and it would ship silently.
    # The bank is recorded, not just used: the axes ARE its directions and the
    # labels are attached on that basis, so a fit has to be able to say which
    # bank it was held to. The SELECTED ids are what is stamped, not the store
    # file — the store accumulates motifs this fit never saw, so hashing it whole
    # would move the stamp on an addition that changed nothing about this fit,
    # and identity is exactly what the stamp exists to pin.
    bh = hashlib.sha256("\n".join(bank_spec).encode()).hexdigest()[:16]
    np.savez_compressed(os.path.join(axes_dir, "axes.npz"),
                        W=m.g_lin.weight.detach().cpu().numpy(),
                        k=np.array(k_axes),
                        trunk_width=np.array(m.g_lin.weight.shape[1]),
                        g_rms=m.g_rms.cpu().numpy().ravel(),
                        axis_bank=np.array(bh),
                        embedding_version=np.array(ev))
    # The trunk too: the per-cell fanout has to run it, and unlike the two-stage
    # map this projection does not fold into a matrix (see write_mirt_per_cell).
    torch.save({"trunk": m.trunk.state_dict(), "head": m.head.state_dict(),
                "g_lin": m.g_lin.state_dict(),
                "floor_mode": floor_mode,
                "floor_lin": (m.floor_lin.state_dict()
                              if hasattr(m, "floor_lin") else None),
                "b_mu": m.b_mu, "b_sd": m.b_sd,
                "g_rms": m.g_rms.cpu()},
               os.path.join(out_dir, "mirt_model.pt"))
    print(f"  wrote axes/per_problem.csv + axes.npz (K={k_axes}, "
          f"embedding_version {ev}, bank {bh})", flush=True)

    # RETIRE ANY per_cell/ FROM THE PREVIOUS AXIS MAP. package_for_pwa reads the
    # correct-move vectors from per_problem.csv and the picked-move vectors from
    # per_cell/, and has no way to tell they came from different fits — it would
    # ship a shard whose two embedding sources disagree, which no client check
    # can catch because both files are individually well-formed. Moving it aside
    # makes packaging fail loudly instead.
    pc = os.path.join(axes_dir, "per_cell")
    if os.path.isdir(pc):
        stale = os.path.join(axes_dir, f"per_cell.stale-{int(time.time())}")
        os.rename(pc, stale)
        print(f"  RETIRED the previous per_cell/ -> {os.path.basename(stale)}. "
              f"It was projected through the axis map this fit replaces, so "
              f"shipping it beside these correct-move vectors would mix two "
              f"fits. Packaging needs a per-cell pass against THIS model before "
              f"it can run; delete the stale directory once that exists.",
              flush=True)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s", flush=True)
    return {
        "rows": int(len(df)), "users": int(n_users),
        "base_rate": float(y.mean()), "stop_rows": int(is_stop.sum()),
        "bank_taught": int(bank_ok.sum()), "bank_spec": list(bank_spec),
        "lambda": lam, "elapsed_s": round(elapsed, 1),
        "link_power": (float(nn.functional.softplus(m.link_power_raw))
                       if link_power else None),
        "floor_prior": ([float(m.floor_prior_mu),
                         float(torch.exp(m.floor_prior_logsd))]
                        if floor_prior else None),
        "epochs": int(epochs), "trace": trace,
        "k_axes": int(k_axes),
        "random_user_weight": float(random_user_weight),
        "strong_user_weight": float(strong_user_weight),
        "theta_synthetic": theta_synthetic,
        "lr_schedule": str(lr_schedule),
        "lr_warmup_epochs": int(lr_warmup_epochs),
        "floor_mode": str(floor_mode),
        "theta_laplace": bool(theta_laplace),
        "init_base_rate": bool(init_base_rate),
        "ortho_axes": float(ortho_axes),
        "teach_detach": bool(teach_detach),
        "ortho_detach": bool(ortho_detach),
        "ortho_axes_detach": bool(ortho_axes_detach),
        "v_ridge_ladder": float(v_ridge_ladder),
        "lr_integral_epochs": round(lr_epochs, 3),
        "split_seed": int(seed if split_seed is None else split_seed),
        "pca_seed": int(seed if pca_seed is None else pca_seed),
        "init_seed": int(seed if init_seed is None else init_seed),
        "stream_seed": None if stream_seed is None else int(stream_seed),
        "stop_bce_no_axes": bce_stop_no_axes,
        "slope_scale_sweep": slope_sweep,
        "slope_pct": {str(p): float(np.percentile(a_np, p))
                      for p in (0.1, 1, 5, 50, 95, 99, 99.9)},
        "slope_at_floor_share": float(np.mean(a_np <= SLOPE_FLOOR * 1.01)),
        "rating_pct": {str(p): float(np.percentile(rating, p))
                       for p in (0.1, 1, 5, 50, 95, 99, 99.9)},
        "rating_min": float(rating.min()), "rating_max": float(rating.max()),
        "floor_pct": {str(p): float(np.percentile(c_np, p))
                      for p in (50, 99, 99.9)},
        "identity": identity, "embedding_version": ev,
    }


def main():
    run()


if __name__ == "__main__":
    main()
