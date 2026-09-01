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
    load_first_time_attempts, load_joint_mirt_features)
# The trunk's layer stack, stated once — pipeline.write_mirt_per_cell rebuilds it
# to project every legal cell, and a stack that matches by shape while differing
# in its activation loads cleanly and projects through the wrong function.
from utils.rating.mirt_net import build_trunk  # noqa: E402
# The shared tail weight, in numpy. `_tail_w` below is the same function in torch;
# the objective, the teaching target and the identity report are all this shape, so
# they cannot drift apart.
from utils.rating.tail_agreement import (  # noqa: E402
    TAIL_HINGE, hinged_r, tail_weight as tail_weight_np)

# The fit runs a FIXED budget. The stop split is flat well before the end — it
# settles into a band a thousandth wide and then wanders inside it — so a
# patience rule reading it selects an epoch out of that noise rather than out of
# the fit, and the epoch it lands on moves by thousands of epochs between runs
# that are otherwise identical. A fixed budget removes that source of variance
# entirely; the split is still measured and printed, as the record of where the
# fit got to, but nothing is selected on it.
EPOCHS = 3000
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
FIELD_PARAMS = ("theta.weight", "v", "lapse_raw")
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
MOTIF_PROXIES = os.path.join(AXES_DIR, "motif_proxies.npz")
LABELS = os.path.join(AXES_DIR, "labels.json")


class JointMIRT(nn.Module):
    """Compensatory M4PL with per-problem parameters from a shared trunk.

    The one unavoidable departure from textbook MIRT is that a_p, b_p, c_p and
    g_k come from a FEATURE NET rather than being free per item: a third of the
    corpus has no attempts at all, so free per-item parameters cannot price it.
    """

    def __init__(self, fdim, n_users, input_dim, seed, k_axes=K_AXES,
                 dropout=DROPOUT, input_dropout=INPUT_DROPOUT):
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
        self.trunk, d = build_trunk(fdim, dropout=dropout)
        self.head = nn.Linear(d, 3)                     # difficulty, slope, floor
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
        return [{"params": net, "lr": lr_net},
                {"params": field, "lr": lr_field}]

    def lapse(self):
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
        return self.head(self.trunk(feat))[:, 1]

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
            mu, sd = b.mean(), b.std() + EPS
            with torch.no_grad():
                self.b_mu.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * mu)
                self.b_sd.mul_(1 - GAUGE_MOM).add_(GAUGE_MOM * sd)
        else:
            mu, sd = self.b_mu, self.b_sd
        b = (b - mu) / sd
        a = nn.functional.softplus(o[:, 1]).clamp(min=SLOPE_FLOOR)
        lam = self.lapse()
        cap = 1.0 - 2.0 * lam
        c = cap * torch.sigmoid(o[:, 2])
        g = self.g_lin(h)
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
        return b, a, c, g / rms

    def forward(self, uid, feat):
        b, a, c, g = self.problem_params(feat)
        th = self.theta(uid).squeeze(-1)
        lam = self.lapse()
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
        return (c + (1.0 - c - lam) * torch.sigmoid(z), g,
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
        teach_weight=TEACH_WEIGHT, stop_row_frac=STOP_ROW_FRAC,
        pca_dim=PCA_DIM, blocks=PRODUCTION_BLOCKS, stop_unit="row",
        split_seed=None, pca_seed=None, lr_schedule="constant",
        k_axes=K_AXES, dropout=DROPOUT, input_dropout=INPUT_DROPOUT,
        weight_decay=0.0, out_dir=OUT_DIR, axes_dir=AXES_DIR):
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
        SIZES, pca_dim, seed if pca_seed is None else pca_seed, blocks=blocks)
    fdim = feat.shape[1]
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
        is_stop = (rng.random(len(keys)) < stop_row_frac)[prob_idx]
    else:
        is_stop = rng.random(len(df)) < stop_row_frac
    tr = torch.tensor(np.where(~is_stop)[0], device=device)
    i_stop = torch.tensor(np.where(is_stop)[0], device=device)
    print(f"  stop signal: {int(is_stop.sum())} rows held aside", flush=True)

    bank_tgt, bank_ok, bank_spec = load_bank(key_idx, k_axes)
    t_tgt = torch.tensor(bank_tgt, dtype=torch.float32, device=device)
    t_ok = torch.tensor(bank_ok, device=device)
    print(f"  axis bank: {int(bank_ok.sum())}/{len(keys)} problems carry banked "
          f"directions, taught at a constant weight of {teach_weight}",
          flush=True)

    m = JointMIRT(fdim, n_users, input_dim, seed, k_axes,
                  dropout=dropout, input_dropout=input_dropout).to(device)
    # Adam with no decay is what ships; the decay knob exists so 1.5's claim
    # that a norm penalty does not substitute for dropout can be re-measured,
    # and at 0 this is the same optimizer it always was.
    opt = torch.optim.Adam(m.param_groups(lr_net, lr_field),
                           weight_decay=weight_decay)
    o_tau = float(np.log(np.expm1(shallow_tau)))
    # A COSINE DECAY IS THE ALTERNATIVE TO A CONSTANT RATE, and what it is being
    # asked is whether the settled band narrows: a constant rate keeps taking
    # full-size steps long after the fit has arrived, which is what a run
    # wandering inside a wide band at its end looks like. `constant` is what
    # ships and leaves every step exactly as it was.
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown lr_schedule {lr_schedule!r}")
    base_lrs = [g["lr"] for g in opt.param_groups]
    n = len(tr)
    trace = []
    print(f"\n=== joint fit | K={k_axes} | teach weight {teach_weight} "
          f"(constant) ===", flush=True)

    for ep in range(epochs):
        if lr_schedule == "cosine":
            # Standard cosine from the base rate to zero over the whole budget,
            # applied to BOTH groups so their ratio — the thing 1.8 measures —
            # is held fixed all the way down.
            scale = 0.5 * (1.0 + float(np.cos(np.pi * ep / max(epochs - 1, 1))))
            for grp, base in zip(opt.param_groups, base_lrs):
                grp["lr"] = base * scale
        m.train(True)
        perm = tr[torch.randperm(n, device=device)]
        for i in range(0, n, batch):
            bi = perm[i:i + batch]
            pred, gh, rsf = m(tu[bi], tfeat[tp[bi]])
            pred = pred.clamp(EPS, 1 - EPS)
            loss = torch.nn.functional.binary_cross_entropy(
                pred, ty[bi], reduction="none").sum() / len(bi)
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
            loss = loss + v_ridge * (m.user_axis(tu[bi]) ** 2).sum() / len(bi)
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
                    W = _tail_w(torch, _sg * gh, TAIL_HINGE)
                    W = W - W.mean(dim=0, keepdim=True)
                    W = W / (W.norm(dim=0, keepdim=True) + 1e-9)
                    loss = loss + ortho_rating * (W.T @ tgt).pow(2).sum()
            elif ortho_mode == "whole":
                # The whole-axis form, kept ONLY as a comparison arm: a linear
                # statistic that a symmetric tail dependence satisfies at zero,
                # which is the failure the per-tail form above exists to close.
                W = gh - gh.mean(dim=0, keepdim=True)
                W = W / (W.norm(dim=0, keepdim=True) + 1e-9)
                loss = loss + ortho_rating * (W.T @ tz).pow(2).sum()
            elif ortho_mode != "off":
                raise ValueError(f"unknown ortho_mode {ortho_mode!r}")
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
            mk = t_ok[tp[bi]]
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
                gz = (gh - gh.mean(0)) / (gh.std(0) + 1e-6)
                loss = loss + teach_weight * (
                    (gz[mk] - t_tgt[tp[bi]][mk]) ** 2).mean()
            m.train(False)          # the barrier reads corpus rows, dropout off
            si = torch.randint(len(tfeat), (shallow_samples,), device=device)
            short = torch.nn.functional.softplus(
                o_tau - m.slope_logit(tfeat[si])).pow(2).mean()
            loss = loss + shallow_w * short
            m.train(True)
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep + 1) % check_every == 0:
            m.train(False)
            with torch.no_grad():
                pr, _, _ = m(tu, tfeat[tp])
                pr = pr.clamp(EPS, 1 - EPS)
                bce_tr = float(torch.nn.functional.binary_cross_entropy(
                    pr[tr], ty[tr]))
                bce_st = float(torch.nn.functional.binary_cross_entropy(
                    pr[i_stop], ty[i_stop]))
            print(f"  epoch {ep + 1:4d}/{epochs}  train {bce_tr:.4f}  "
                  f"stop {bce_st:.4f}", flush=True)
            trace.append({"epoch": ep + 1, "train_bce": bce_tr,
                          "stop_bce": bce_st})

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
    with torch.no_grad():
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

    # --- outputs
    m.train(False)
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
    theta = m.theta.weight.detach().cpu().numpy().ravel()

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

    natt = pd.Series(user_idx).value_counts()
    pd.DataFrame({"profile": list(users.categories), "theta": theta,
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_users)]}
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
        "epochs": int(epochs), "trace": trace,
        "k_axes": int(k_axes),
        "lr_schedule": str(lr_schedule),
        "split_seed": int(seed if split_seed is None else split_seed),
        "pca_seed": int(seed if pca_seed is None else pca_seed),
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
