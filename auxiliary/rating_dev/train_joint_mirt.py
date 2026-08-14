#!/usr/bin/env python3
"""ONE model producing the rating and the K skill axes together: compensatory
M4PL with per-problem parameters predicted from features.

The shipped arrangement fits these in two stages with two regimens — the rating
from a trunk under one dropout scheme, the axes afterwards from a linear map
under another, against a frozen 4PL, with a quarantined residual head to absorb
what the first stage left. Under the standard multidimensional-IRT reading that
split is artificial: the axes are simply factors 2..K+1 of one model, and the
rating is factor 1.

    P = c_p + (1 - c_p - lambda) * sigmoid( a_p*(theta_u - b_p)
                                            + sum_k g_k(x_p)*v_u[k] )

        a_p, b_p    per-problem discrimination and difficulty
        g_k(x_p)    per-problem loading on axis k
        theta_u     per-profile ability
        v_u[k]      per-user position on axis k
        c_p         per-problem guessing floor
        lambda      one global slip rate

Written with difficulty rather than a bare intercept, and that is a GAUGE
requirement rather than a stylistic one — see problem_params.

This is textbook compensatory M4PL. The one unavoidable departure is that A_p,
d_p and c_p come from a FEATURE NET rather than being free per item: a third of
the corpus has no attempts at all, so free per-item parameters cannot price it.

WHAT THE ARCHITECTURE INHERITS FROM MEASUREMENT, not from taste:

  The AXIS block is a LINEAR map on the features with column dropout on its
  input. Measured on the two-stage trainer: giving the axis path a trunk costs
  -0.430 pooled / -0.413 per-user even with the warm start preserved and no
  sharing, while input dropout on the linear map gains (+0.164 per-user, 9/9
  seeds x rates). Capacity there does not pay and regularization does.

  The RATING heads keep a trunk, with the rating trainer's dropout regimen.
  Their per-problem parameters are informed by every attempt on the problem, so
  they tolerate capacity the axes cannot — the signal each factor sees falls off
  with its index, which is the single fact most of these asymmetries follow from.

  `r` IS GONE, on the measurement and NOT on the argument first given here. That
  argument — r buys back freedom lost when the intercept is forced to factor as
  -a_p*b_p, so a free intercept supplies it directly — is void: for any (a, d)
  with a > 0 the factorization reproduces it exactly at b = -d/a, so the two
  forms carry identical parameter counts and neither is freer. They differ only
  in which quantity the gauge can constrain, which is why this file now uses the
  difficulty form. What survives is r's measured size: removing its capacity
  costs 0.05 pooled / 0.18 per-user, and 7.5k parameters saturate it against the
  shipped 16k. A correction that small and that bounded is what a per-problem
  difficulty fitted jointly with the axes should absorb.

  THERE IS ONE BARRIER, on the slope, in a one-sided form; see SHALLOW_TAU. It
  is not 4PL-specific: the rating is derived as b + kappa(c)/a, so a slope
  collapsing toward its floor sends that ratio to infinity, and the barrier acts
  on the PRE-SOFTPLUS slope because the clamp zeroes the gradient exactly where a
  collapsed slope needs lifting. Omitting it produced a 99th-percentile rating of
  172 against an expected ceiling near 13 — the failure is loud, but only in a
  percentile nobody would print by default. The rating trainer's counterpart on
  the easy tail, a high-floor barrier, is NOT carried here.

  The AXIS WARM START is kept: at v = 0 the loading gradient is identically
  zero, so a randomly-started axis escapes along its draw rather than the data.

  FEATURES EXCLUDE the rating model's own outputs. The two-stage axis fit reads
  rating/slope/floor/rank as four of its columns, which is only sound because
  those come from an earlier, frozen fit. Here they would be circular.

TWO CRITERIA, REPORTED SEPARATELY, because the two halves face different novelty
in deployment: the rating must price problems nobody has attempted (a third of
the shipped corpus), while the axes must describe users nobody fitted on. Held-out
PROBLEM logloss answers the first, held-out USER gain the second. Merging them
into one number would require an exchange rate nobody can justify.

The axis gain is scored by the SAME harness and against the SAME shipped-4PL
baseline as every other axis measurement, so its number sits on one scale with
them. The float-rounding floor on that scale is ~0.03.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.train_joint_mirt --k 3 --seeds 0
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from datetime import datetime  # noqa: E402

from utils.rating.model_constants import (  # noqa: E402
    DIFF_HIDDEN, DROPOUT, EPS, FLOOR_CAP, INPUT_DROPOUT,
    LAPSE_CAP, LAPSE_INIT,
    PCA_DIM, SEED, SIZES, SLOPE_FLOOR, clear_logit_vec,
)
from utils.rating.features import (  # noqa: E402
    _cache_ids_path, load_first_time_attempts, load_joint_activations,
    load_joint_bot_winrate, load_joint_game_phase, load_joint_move_features,
)
from utils.rating.residual_axes import (  # noqa: E402
    AXIS_INPUT_DROPOUT, MIN_USER_ATTEMPTS, ORTHO_WEIGHT, _stratified_folds,
    axes_ll_gain, build_axis_features, eigen_directions, load_axis_attempts,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

from auxiliary.rating_dev.burn_in_warm_start import (  # noqa: E402
    burn_in_directions,
)

# A RUNAWAY FAILSAFE, not a budget. Convergence is decided by FLAT_PATIENCE on
# the stop split, and reaching this cap is FATAL — the run is discarded rather
# than reported, since a fit cut off mid-descent is not the fit the stopping rule
# describes. So the cap must sit far above any epoch a healthy run reaches
# (whole-history fits stop in the mid-hundreds); at a value that can plausibly
# bind, every configuration change risks failing for want of budget rather than
# for anything about the model, and the natural repair — raise it and re-run — is
# indistinguishable from tuning the stop.
EPOCHS = 10000
CHECK_EVERY = 10
# NO-NEW-BEST patience, not a relative-improvement threshold. The stop split
# holds ~27k rows, so its BCE carries roughly +/-0.004 of sampling noise; a
# threshold small enough to mean "converged" sits far below that and fires on
# noise, while one above it fires immediately. Patience on the best value is
# robust to exactly this: noise yields occasional lucky checks but not a
# sustained run of them.
FLAT_PATIENCE = 5                # checks without a new best before stopping
# (no MIN_EPOCHS: patience cannot fire before FLAT_PATIENCE*CHECK_EVERY = 50
#  epochs anyway, so a separate floor below that was never reachable.)
BATCH = 65536
# Split rates — see JointMIRT.param_groups. A single 0.01 for both the network
# and the free per-user fields diverges late in every arrangement measured.
# SWEPT as a 4x3 grid on held-out users (2026-08-13, whole history, one barrier,
# 1-of-each filter). What the grid actually measures is the RATIO lr_field/lr_net,
# monotone and saturating near 30x: ratio 1-3.3x scores 0.4982-0.5009, 10x scores
# 0.4953-0.4978, and 30x upward is flat at 0.4928-0.4941. So "an order of
# magnitude" is NOT enough — 10x is 2.5 mnat down. Given a ratio past 30x, a
# slower net buys the last 0.8: lr_net 1e-4 and 3e-4 tie at 0.4928 while 1e-3
# gives 0.4953 and 3e-3 gives 0.4982.
#
# 3e-4 over 1e-4 on the two things that separate a tie: better axes at identical
# rating (+0.54 vs -0.65 mnat, a comparison only legitimate because the rating
# is equal), and 380 epochs against 850.
LR_NET = 3e-4
LR_FIELD = 1e-2
V_RIDGE = 0.5
V_MEAN_PENALTY = 10.0
GAUGE_MOM = 0.01
# ONE-SIDED shallow-slope barrier, replacing the rating trainer's mean(1/a^2).
# That form is unbounded below in a, so its gradient runs as -2/a^3 — at a slope
# of 0.01 that is -2e6, and at the shipped weight it still contributes -200 per
# element. It does not say "do not collapse", it says "larger is always better",
# and only the likelihood pushes back. Measured here: with it on, the median
# slope reached 929 and the 99th percentile 1.2e7 while held-out BCE worsened
# 75%; with it off, 7.6% of slopes sat on the clamp and the derived ratings
# diverged. Neither is usable, and the weight that balances it in the rating
# trainer does not transfer.
#
# The replacement is the floor barrier's own form, applied on the other side and
# in the SAME SPACE. The floor acts on its raw pre-sigmoid logit, which is why a
# healthy floor sits several units below the divergence point and softplus
# genuinely vanishes there. Comparing slopes directly does not have that
# property: at the median slope of ~0.4 with tau = 0.1, softplus(tau - a) is
# still 0.55 and the penalty pushes healthy slopes up as hard as the form it
# replaced. So this acts on the PRE-SOFTPLUS slope logit, where the scale is
# comparable and the penalty falls away properly.
#
#   slope:  softplus(o_tau  - o_a  )^2      below a threshold
#
# The floor barrier the slope form was modelled on — softplus(o_c - o_div)^2,
# holding the floor below the point where the 80%-clear rating stops existing —
# is NOT carried here. There is one barrier, on the slope.
SHALLOW_TAU = 0.1                # threshold expressed as a SLOPE, converted below
# Measured, 30-epoch sweep at 0 / 0.3 / 1 / 3. Both failure modes are real and
# they are not mirror images. Too weak and slopes collapse onto the clamp,
# sending derived ratings to +/-10^3. Too strong and the RATING RANGE compresses
# toward nothing — at 3 the whole corpus spans -2.2 to +1.1, which destroys the
# ordering the rating exists to provide, and a few slopes run away to 10^5 while
# the median stays near one. 0.3 is best on held-out logloss and carries the
# widest plausible rating range with no slope on the clamp.
SHALLOW_W = 0.3
HELD_OUT_PROBLEM_FRAC = 0.15
# Seed for the SHARED held-out generator (auxiliary.rating_dev.holdout_set), the
# same one train_rating_staged --holdout-problem-frac and validate_ratings use.
# Changing it makes this model's numbers incomparable with every production
# measurement, which is the entire point of not drawing a split locally.
HELD_OUT_SEED = 0
# Of the held-out problems, this share is used ONLY to decide when to stop; the
# remainder is the reported criterion. Stopping on the same problems the score is
# read from would select the epoch on the eval set and make every arm optimistic
# by an unknown amount — the trunk experiment showed held-out peaking hundreds of
# epochs before the training loss flattened, so the stop has to watch a held-out
# signal, and that signal must not be the one being reported.
STOP_SPLIT_FRAC = 0.33
# Share of TRAINING rows set aside purely as a stopping signal when there is
# no problem holdout. Small: it is a convergence detector, not a criterion,
# and every row spent on it is a row the model does not learn from.
STOP_ROW_FRAC = 0.03
# Epochs trained with v pinned at zero before --warm-start burnin reads the axis
# directions off the model's own residuals. Not free-standing: it is a schedule
# point inside one fit, not a stage, and the epochs are the rating heads' own.
#
# Measured on held-out users, two seeds each (axis mnat, and its seed spread is
# the point):
#     100          0.4944 +/- 0.0021   +0.22 +/- 1.20   residuals carry rating ERROR
#     300          0.4933 +/- 0.0004   -0.13 +/- 0.23   <-- shipped
#     500          0.4942 +/- 0.0007   -1.15 +/- 0.07   residuals carry rating OVERFIT
#     converge     0.4954 +/- 0.0020   -0.63 +/- 1.46
#
# A FIXED COUNT BEATS RUNNING TO CONVERGENCE, which is not the obvious answer and
# is worth stating in full. Handing over at the axis-free fit's own patience-rule
# convergence sounds strictly better — it is what the two-stage pipeline does —
# but that epoch is itself seed-dependent (380 on one seed, 550 on the other),
# and the outcome is epoch-sensitive, so the criterion INHERITS that variance:
# spread 1.46 against a fixed count's 0.23. A fixed handover is therefore not a
# magic number standing in for a criterion; it removes a variance source the
# criterion introduces. Both ends still corrupt the residuals as the table shows,
# so 300 is an interior optimum, not a floor or a ceiling.
BURN_IN_EPOCHS = 300
# THE ONLY INCLUSION RULE. It replaces both of the rating trainer's — the 5-of-
# each user filter (MIN_ATTEMPTS) and the 8-attempt profile floor (MIN_PROFILE) —
# because those are judgements about how WELL a parameter is estimated, while
# this is the condition for it to be estimable at all, and only the second kind
# belongs in a filter. A thin profile is not bad data; it is data carrying little
# information, which the likelihood already weighs correctly by giving it few
# terms. Discarding it instead throws away the attempts too.
#
# IDENTIFIABILITY: under the 4PL an all-correct record has likelihood
# rising monotonically to (1-lambda)^n as ability grows without bound, and an
# all-wrong one rises monotonically to (1-c)^n as it falls, so neither has an
# interior maximum and the ability is not determined by the data at all. One of
# each outcome is exactly the condition for an interior optimum to exist, and
# anything above it is a judgement about how well ability should be estimated
# rather than whether it can be. Applied to USERS (their axis position) and again
# to PROFILES (their ability), since a user with one of each need not have one of
# each inside every window.
MIN_BOTH_OUTCOMES = 1


def build_profiles_at(df, window_months, overlap=True):
    """Ability units over a window of `window_months`, sliding one month at a
    time when `overlap` (production's arrangement), or in disjoint blocks of that
    width otherwise. A dev copy of utils.rating.features.build_profiles, which is
    fixed at production's setting — production carries no knob and is not edited
    to run a sweep.

    WINDOW WIDTH AND STEP ARE SEPARATE, and conflating them is what made an
    earlier sweep measure nothing. Production keys profiles by MONTH and writes
    each attempt to month M and M-1, so a profile spans 2 months and the windows
    step by 1. Keying by a `window_months`-wide bucket instead and writing to
    that bucket and its predecessor gives a window of TWICE the requested width
    stepping by the width — so every arm of such a sweep is mislabelled by 2x and
    the production setting is not among them.

    Here an attempt in month M is written to profiles M, M-1, ... M-(W-1), so
    profile P spans months {P .. P+W-1}: width exactly W, step exactly 1, and
    W=2 reproduces production. `overlap=False` instead keys by a disjoint
    W-wide bucket, which is a genuinely different arrangement rather than a
    different width.

    window_months=0 means ONE UNIT PER USER: the whole history, no windowing.
    That is the simplest arrangement available and it makes the ability and
    axis-position granularities coincide, removing the mixed-granularity
    question rather than answering it.

    The id format is kept as "{user8}_{tag}" so the user is always the first
    eight characters — the joint fit recovers it that way, and a separator-based
    split silently fits axis positions per BLOCK instead of per user."""
    out = []
    for uid, tid, succ, ts in zip(df["user_id"], df["problem"],
                                  df["success"], df["created_at"]):
        u8 = str(uid)[:8]
        if window_months <= 0:
            out.append((f"{u8}_all", tid, int(succ)))
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        idx = dt.year * 12 + (dt.month - 1)
        tags = ([idx - k for k in range(window_months)] if overlap
                else [idx // window_months])
        for t in tags:
            out.append((f"{u8}_{t}", tid, int(succ)))
    import pandas as pd
    return pd.DataFrame(out, columns=["profile", "problem", "success"])


class JointMIRT:
    """Feature net -> (intercept, discrimination, floor) + a linear axis map,
    with free per-profile ability and free per-user axis positions."""

    def __init__(self, fdim, n_profiles, n_users, k_axes, input_dim, device,
                 seed, g_init=None, global_slope=False, act="swiglu",
                 hidden=DIFF_HIDDEN, axis_dropout=AXIS_INPUT_DROPOUT,
                 axis_reads_rating=True, axis_from_trunk=True,
                 axis_trunk_coupled=True):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch, self.nn, self.k = torch, nn, k_axes
        self.input_dim = int(input_dim)
        self.axis_dropout = float(axis_dropout)
        # The axis map reads the rating head's outputs (difficulty, slope,
        # floor) as extra input columns. See problem_params.
        self.axis_from_trunk = bool(axis_from_trunk)
        self.axis_trunk_coupled = bool(axis_trunk_coupled)
        # Reading the trunk supersedes reading its 3 scalar outputs.
        self.axis_reads_rating = bool(axis_reads_rating) and not self.axis_from_trunk
        self.axis_extra = 3 if self.axis_reads_rating else 0

        # theta is constructed BEFORE the trunk so the RNG draw order is fixed,
        # matching the rating trainer.
        self.theta = nn.Embedding(n_profiles, 1).to(device)
        nn.init.normal_(self.theta.weight, std=1.0)
        class SwiGLU(nn.Module):
            """silu(xW) * xV, replacing [Linear, ReLU]. Carries twice the
            parameters at the same width, since the gate is its own
            projection — so a same-width swap is also a capacity increase."""

            def __init__(self, d_in, d_out):
                super().__init__()
                self.w = nn.Linear(d_in, d_out)
                self.v = nn.Linear(d_in, d_out)

            def forward(self, x):
                return torch.nn.functional.silu(self.w(x)) * self.v(x)

        layers, d = [], fdim
        for h in hidden:
            layers += ([SwiGLU(d, h)] if act == "swiglu"
                       else [nn.Linear(d, h), nn.ReLU()])
            layers += [nn.Dropout(DROPOUT)]
            d = h
        self.trunk = nn.Sequential(*layers).to(device)
        self.head = nn.Linear(d, 3).to(device)          # (intercept, slope, floor)
        self.global_slope = bool(global_slope)
        # One corpus-wide slope, learned. Not a constant: the scale still has to
        # be fit, it just cannot vary per problem.
        self.a_raw = torch.tensor(0.0, device=device, requires_grad=True)
        # The axis map is LINEAR on the features, deliberately — see the module
        # docstring. It is NOT fed by the trunk.
        # From the TRUNK's representation, or from the raw features. The
        # rejection on record — a trunk on the axis path costs 0.43 mnat — is
        # for a SEPARATE trunk beside the linear map, i.e. added capacity. This
        # adds none: it reads the representation the rating heads already
        # compute, which is strictly more than the 3 scalars they emit.
        g_in_dim = d if self.axis_from_trunk else fdim + self.axis_extra
        self.g_lin = nn.Linear(g_in_dim, k_axes, bias=False).to(device)
        if g_init is not None:
            if g_init.shape[1] > self.g_lin.weight.shape[1]:
                raise SystemExit(
                    f"warm-start directions are {g_init.shape[1]}-dimensional but "
                    f"the axis map reads {self.g_lin.weight.shape[1]} inputs. "
                    "These are not the same coordinates and one cannot be "
                    "truncated into the other: --warm-start eigen builds its "
                    "directions in the axis FEATURE space, while --axis-from-"
                    "trunk parameterizes the map over the trunk's output. Use "
                    "--warm-start burnin, which takes its residuals in whichever "
                    "space the map actually lives in, or --axis-from-features.")
            with torch.no_grad():
                # g_init may be narrower than g_lin's input (the eigen form is
                # built on the feature matrix alone); seed those columns and
                # leave any extra ones at their init.
                self.g_lin.weight[:g_init.shape[0], :g_init.shape[1]].copy_(
                    torch.tensor(g_init, dtype=torch.float32, device=device))
        self.v = torch.zeros(n_users, k_axes, device=device, requires_grad=True)
        li = float(np.clip(LAPSE_INIT / LAPSE_CAP, EPS, 1.0 - EPS))
        self.lapse_raw = torch.tensor(float(np.log(li / (1.0 - li))),
                                      device=device, requires_grad=True)
        self.register = [self.lapse_raw]
        # Rating-gauge EMA, on DIFFICULTY. See problem_params for why it cannot
        # sit on the intercept.
        self.b_mu = torch.zeros((), device=device)
        self.b_sd = torch.ones((), device=device)
        self.training = True

    def params(self):
        return (list(self.trunk.parameters()) + list(self.head.parameters())
                + list(self.g_lin.parameters()) + list(self.theta.parameters())
                + [self.v, self.lapse_raw, self.a_raw])

    def param_groups(self, lr_net, lr_field):
        """SPLIT RATES. The network must train an order of magnitude slower than
        the free per-user fields. Under one shared rate this fit DIVERGES LATE:
        eval-mode training loss — so not a dropout artifact — rises from 0.460 at
        epoch 70 to 0.521 at epoch 90 and never recovers, and the same happens in
        every window arrangement tried, at epochs 40 to 110. It is the failure
        already on record for the two-stage axis trainer, whose network likewise
        sits beside a free per-user field.

        The gauge parameterization is NOT the cause and does not mitigate it,
        measured directly: the difficulty-whitened fit diverges at epoch 80 where
        the intercept-whitened one lasted to 110. The rating trainer gets away
        with a single rate because its only per-user parameter is one ability
        embedding; this model has that plus a K-dimensional axis position per
        user, so it does not inherit the constant."""
        net = (list(self.trunk.parameters()) + list(self.head.parameters())
               + list(self.g_lin.parameters()))
        # theta, v and the two global scalars are all free point estimates fitted
        # directly against the data rather than through the net, so they take the
        # field rate together.
        field = (list(self.theta.parameters())
                 + [self.v, self.lapse_raw, self.a_raw])
        return [{"params": net, "lr": lr_net},
                {"params": field, "lr": lr_field}]

    def train(self, on):
        self.training = on
        self.trunk.train(on)

    def lapse(self):
        return LAPSE_CAP * self.torch.sigmoid(self.lapse_raw)

    def _drop(self, feat, rate):
        torch = self.torch
        if not self.training or rate <= 0.0 or self.input_dim <= 0:
            return feat
        keep = (torch.rand(len(feat), self.input_dim, device=feat.device)
                >= rate).float() / (1.0 - rate)
        return torch.cat([feat[:, :self.input_dim] * keep,
                          feat[:, self.input_dim:]], dim=1)

    def slope_logit(self, feat):
        """Pre-softplus slope output — the space the shallow barrier acts in,
        mirroring the floor barrier acting on its own pre-sigmoid logit."""
        o = self.head(self.trunk(feat))
        return (self.a_raw.expand(o.shape[0]) if self.global_slope
                else o[:, 1])

    def slope_raw(self, feat):
        """PRE-clamp slope. The shallow-slope barrier acts here: past the clamp
        the gradient is zero exactly where a collapsed slope needs lifting."""
        o = self.head(self.trunk(feat))
        a_src = (self.a_raw.expand(o.shape[0]) if self.global_slope
                 else o[:, 1])
        return self.nn.functional.softplus(a_src).clamp_min(1e-4)

    def floor_logit(self, feat):
        """Raw floor logit. No barrier acts on it — the floor is bounded
        structurally instead, by `cap` in problem_params."""
        return self.head(self.trunk(feat))[:, 2]

    def problem_params(self, feat):
        """(difficulty b, discrimination a, floor c, axis loadings g).

        b carries the GAUGE, whitened to mean 0 / sd 1 on batch statistics while
        training and on their EMA at eval, exactly as the rating trainer does.

        THE GAUGE MUST SIT ON DIFFICULTY, NOT ON THE INTERCEPT. Two freedoms have
        to be removed — a shift of ability and a rescaling of it — and whitening
        an ADDITIVE intercept removes only the first.

          shift:  ability + t  needs  intercept - a*t  to compensate, which moves
                  the intercept's mean off zero. Blocked either way.
          scale:  ability * s  with  a / s  leaves an additive intercept
                  UNTOUCHED, so its whitening is satisfied and nothing objects.
                  Blocked only when difficulty is the whitened quantity, because
                  it is MULTIPLIED by the slope and so must rescale with it.

        The shipped rating is an ability, so under the unblocked rescaling every
        rating scales with it and the whole axis floats. In the intercept form
        the only term with any preference about that scale is the shallow-slope
        barrier, i.e. a guard against a numerical failure was setting the units
        of the rating — and it was selected on held-out logloss, which is exactly
        invariant to the scale it was choosing."""
        torch, nn = self.torch, self.nn
        h = self.trunk(self._drop(feat, INPUT_DROPOUT))
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
        a_src = (self.a_raw.expand(o.shape[0]) if self.global_slope
                 else o[:, 1])
        a = nn.functional.softplus(a_src).clamp(min=SLOPE_FLOOR)
        lam = self.lapse()
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=FLOOR_CAP)
        c = cap * torch.sigmoid(o[:, 2])
        # THE AXIS MAP READS THE RATING HEAD'S OUTPUTS. A residual only has a
        # meaning relative to a baseline: the same miss carries different
        # information on an easy problem than on a hard one, so a loading that
        # cannot condition on difficulty is describing the wrong thing. The map
        # is LINEAR and the difficulty is a NONLINEAR function of the features
        # (the trunk computes it), so it cannot recover b from feat even in
        # principle — the columns have to be supplied. Production's axis fit does
        # exactly this, reading rating/slope/floor as three of its own input
        # columns; withholding them here while keeping the map linear inherits
        # the "linear is enough" result from a setting that had them.
        #
        # This is NOT the circularity the module docstring warns about. Building
        # FEATURES out of a previous fit's outputs and treating them as exogenous
        # is circular; reading your own intermediate activations further down the
        # same forward pass is ordinary composition, with no fixed point and no
        # leakage. DETACHED, so the axis objective cannot reshape the rating
        # head — which is the frozen-fit semantics production gets for free — and
        # appended AFTER the dropout mask, matching production's exemption of
        # these columns (EXEMPT_TAIL_COLS).
        if self.axis_from_trunk:
            # COUPLED by default: the axis objective reaches the trunk, so the
            # trunk learns representations the AXES need and not only the ones
            # the rating needs. Detaching would make this feature reuse rather
            # than a joint fit — the axes taking whatever a rating-optimised
            # representation happens to offer — which is the two-stage split
            # again, one layer down, in a model whose premise is that the split
            # is artificial.
            #
            # Measured indistinguishable (rating 0.4930 vs 0.4928, axis +1.93 vs
            # +1.91), so this is a choice about what the model IS, not a score.
            # The cost is that nothing structurally stops the axes reshaping what
            # the rating depends on; V_RIDGE is the only brake, and the check is
            # corr(v, theta) — measured 0.110 coupled against 0.097 detached, the
            # predicted direction and small.
            g_in = h if self.axis_trunk_coupled else h.detach()
        else:
            g_in = self._drop(feat, self.axis_dropout)
            if self.axis_reads_rating:
                g_in = torch.cat([g_in, b.detach()[:, None], a.detach()[:, None],
                                  c.detach()[:, None]], dim=1)
        g = self.g_lin(g_in)
        g = g / (g.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
        return b, a, c, g

    def axis_input(self, feat):
        """The axis map's input basis with dropout off — what the warm start's
        residual scores must be taken against, so its directions live in the
        same space the map is parameterized over."""
        torch = self.torch
        was = self.training
        self.train(False)
        if self.axis_from_trunk:
            with torch.no_grad():
                out = self.trunk(feat)
            self.train(was)
            return out
        if not self.axis_reads_rating:
            self.train(was)
            return feat
        with torch.no_grad():
            b, a, c, _g = self.problem_params(feat)
        self.train(was)
        return torch.cat([feat, b[:, None], a[:, None], c[:, None]], dim=1)

    def forward(self, prof, uid, feat):
        torch = self.torch
        b, a, c, g = self.problem_params(feat)
        th = self.theta(prof).squeeze(-1)
        lam = self.lapse()
        z = (a * (th - b) + (g * self.v[uid]).sum(-1)).clamp(-35, 35)
        return (c + (1.0 - c - lam) * torch.sigmoid(z)).clamp(EPS, 1 - EPS), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="runaway failsafe, not a budget — reaching it is fatal. "
                         "Stopping is FLAT_PATIENCE on the stop split. Lower it "
                         "only to bound a deliberately broken diagnostic.")
    ap.add_argument("--shallow-w", type=float, default=SHALLOW_W,
                    help="weight on the one-sided shallow-slope barrier "
                         "softplus(tau - a)^2, the floor barrier's own form; "
                         "0 disables it")
    ap.add_argument("--no-barriers", action="store_true",
                    help="disable the shallow-slope divergence barrier "
                         "(diagnostic). Equivalent to --shallow-w 0.")
    ap.add_argument("--holdout-problem-frac", type=float,
                    default=HELD_OUT_PROBLEM_FRAC,
                    help="fraction of PROBLEMS held out; 0 disables it, which is "
                         "what a pure row- or user-holdout run wants. Left at the "
                         "benchmark value by default so existing comparisons are "
                         "unaffected.")
    ap.add_argument("--init-directions", default=None,
                    help="FROZEN GAUGE. An npz of per-problem loadings (any "
                         "--dump-params output) to initialize the axis map "
                         "toward, instead of this fit's own residual eigen-"
                         "directions. The bank is stored in PROBLEM space, not "
                         "trunk space, because the trunk is refitted every run "
                         "and its coordinates mean nothing across fits; g_lin is "
                         "recovered by least squares against the new trunk. The "
                         "likelihood is invariant to rotating the axis space, so "
                         "fixing the basis this way costs no accuracy by "
                         "construction — it is a gauge choice, like the "
                         "difficulty gauge.")
    ap.add_argument("--lr-warmup-epochs", type=int, default=0,
                    help="linearly ramp both learning rates from 0 to their full "
                         "value over this many epochs. The claim under test is "
                         "ENDPOINT CONSISTENCY: the axis objective is invariant to "
                         "rotating the latent space, so the endpoint is decided by "
                         "the path, and a less perturbation-sensitive path should "
                         "land two runs closer together.")
    ap.add_argument("--init-from", default=None,
                    help="continue from a previous fit's saved model (--save-model). "
                         "The problem-side modules load directly; theta and the axis "
                         "positions are scattered by profile/user NAME, so accounts "
                         "that did not exist in that fit start at zero. Implies no "
                         "burn-in: the carried axes ARE the warm start.")
    ap.add_argument("--save-model", default=None,
                    help="write the fitted model (post best-epoch restore) so a "
                         "later run can --init-from it")
    ap.add_argument("--drop-recent-frac", type=float, default=0.0,
                    help="drop the most recent fraction of ATTEMPTS by date, "
                         "making the run an earlier refit of itself. Compare "
                         "its axes against the full fit at the same seed to "
                         "measure what one refit's worth of new data moves.")
    ap.add_argument("--holdout-user-frac", type=float, default=0.0,
                    help="hold out this fraction of USERS entirely (shared "
                         "generator, hash-printed) so a walk-forward on them is "
                         "leak-free")
    ap.add_argument("--holdout-row-frac", type=float, default=0.0,
                    help="hold out this fraction of ATTEMPTS, keyed (user, "
                         "problem) — the client's condition: the problem keeps "
                         "its other attempts, only this one is unseen")
    ap.add_argument("--holdout-seed", type=int, default=HELD_OUT_SEED,
                    help="seed for the SHARED held-out problem generator. Must "
                         "match the production run it is compared against; the "
                         "hash printed by both confirms it.")
    ap.add_argument("--axis-holdout-fold", type=int, default=-1,
                    help="hold this user fold OUT of the axis fit (0 or 1; -1 = "
                         "none). Their rows still train difficulty/slope/floor "
                         "and their own ability, but their axis positions are "
                         "pinned at zero so they contribute no gradient to the "
                         "loadings — the same protocol build_axes uses, where "
                         "the 4PL saw every user and only the axis stage does "
                         "CV. WITHOUT this there is no held-out-user axis "
                         "measurement and none may be reported.")
    ap.add_argument("--v-ridge", type=float, default=V_RIDGE,
                    help="ridge on the per-user axis positions. THE AXES' WHOLE "
                         "MAGNITUDE LIVES HERE: g is RMS-normalised per column, "
                         "so v alone sets how much an axis can contribute, and "
                         "too large a value pins the axes flat and reads as "
                         "'the axes are not predictive'.")
    ap.add_argument("--v-mean-penalty", type=float, default=V_MEAN_PENALTY,
                    help="penalty on mean_u(v_k). Stops an axis carrying "
                         "POPULATION difficulty, which would load every user the "
                         "same sign; it does not stop one carrying individual "
                         "ability, since a centred copy satisfies it exactly.")
    ap.add_argument("--axis-from-features", action="store_true",
                    help="read the raw features instead of the trunk, which is "
                         "what this model did before 2026-08-13. ONE trunk now "
                         "produces every output. That adds no capacity — the same "
                         "linear map on a different input, and fewer parameters, "
                         "Linear(64,K) against Linear(215,K) — so the measured "
                         "rejection of a SEPARATE axis trunk beside the linear "
                         "map, which IS a capacity increase, does not cover it. "
                         "Measured indistinguishable, and adopted because a "
                         "single trunk feeding every head is what the model "
                         "claims to be; it also makes AXIS_INPUT_DROPOUT inert "
                         "and subsumes feeding the axis map b, a and c.")
    ap.add_argument("--axis-trunk-detached", action="store_true",
                    help="with --axis-from-trunk, block axis gradients from the "
                         "shared trunk. NOT the default: it makes the trunk "
                         "learn for the rating alone and hands the axes whatever "
                         "that produces, which is feature reuse rather than a "
                         "joint fit. Kept as the ablation for whether the axes "
                         "shaping the representation costs the rating anything.")
    ap.add_argument("--axis-blind", action="store_true",
                    help="withhold the rating head's outputs from the axis map, "
                         "which is what this model did before 2026-08-13. A "
                         "residual is defined against a baseline, so a loading "
                         "that cannot see difficulty cannot say what it is a "
                         "residual OF; and since the map is linear and "
                         "difficulty is a nonlinear function of the features, it "
                         "cannot rederive it. Kept as the ablation.")
    ap.add_argument("--axis-dropout", type=float, default=AXIS_INPUT_DROPOUT,
                    help="column dropout on the AXIS map's input. Inherited at "
                         "0.3 from train_axes, where the axes train against a "
                         "FROZEN 4PL; here they train against a moving one and "
                         "have more to chase, and the stopping diagnostic shows "
                         "the axis block overfitting ~2.5x faster than the "
                         "rating heads. So the inherited value has no claim to "
                         "being right in this model.")
    ap.add_argument("--axis-params-dir", default=None,
                    help="read the rating model's outputs for the EIGEN warm "
                         "start from this ratings/ directory instead of the "
                         "shipped one. Under a holdout the shipped fit saw the "
                         "rows being scored, so a clean eigen measurement needs "
                         "the matching held-out fit here. Irrelevant to "
                         "--warm-start burnin, which reads no external fit.")
    ap.add_argument("--burn-in", type=int, default=BURN_IN_EPOCHS,
                    help="epochs with v pinned at zero before --warm-start "
                         "burnin reads its directions off the model's own "
                         "residuals. Those residuals are only skill structure to "
                         "the extent the axis-free baseline has CONVERGED; taken "
                         "early they are mostly rating error, which centring "
                         "cannot remove because it is user-specific rather than "
                         "shared.")
    ap.add_argument("--flat-patience", type=int, default=FLAT_PATIENCE,
                    help="checks without a new best stop-split before stopping. "
                         "Set far above any reachable check count to disable "
                         "early stopping, which only the diagnostic wants.")
    ap.add_argument("--dump-every", type=int, default=0,
                    help="also dump params every N epochs, as <dump>.epNNNNN.npz "
                         "(needs --dump-params). DIAGNOSTIC MODE: it also makes "
                         "reaching --epochs an endpoint rather than a fatal "
                         "error, since outrunning the stopping rule is the point "
                         "of it. The question it answers is whether the stop "
                         "split picks the epoch the held-out USERS would: the "
                         "two criteria are anti-correlated across the learning-"
                         "rate grid, and the stop split cannot see population "
                         "overfitting because it shares the population.")
    ap.add_argument("--warm-start", choices=("burnin", "eigen", "none"),
                    default="burnin",
                    help="how the axis map is initialized. DEFAULT 'burnin' "
                         "derives it inside the pass from the model's own "
                         "residuals at v = 0, over TRAINING ROWS ONLY — no "
                         "shipped ratings/, no shipped rank curve, nothing from "
                         "a held-out user, so the model is self-contained and "
                         "the leak class is gone by construction. It is chosen "
                         "on STRUCTURE, not on score: with its own leaks removed "
                         "'eigen' measures +0.17 +/- 0.30 against burnin's "
                         "-0.13 +/- 0.23, a difference of 0.30 +/- 0.52 paired, "
                         "i.e. none that two seeds can resolve. 'eigen' borrows "
                         "the two-stage warm start and needs --axis-params-dir "
                         "to be honest under a holdout. 'none' is the ablation, "
                         "and it costs ~2.8 mnat of axis gain.")
    ap.add_argument("--dump-params", default=None,
                    help="write per-problem (rating, difficulty, slope, floor, "
                         "loading) plus abilities and axis positions to this "
                         ".npz, so downstream questions cost a load rather than "
                         "a re-run")
    ap.add_argument("--fit-only", action="store_true",
                    help="stop after --dump-params, skipping the survey rho and "
                         "axis-criterion evaluations. Those run on CPU and can "
                         "outlast the fit itself, so a QUEUE of fits spends most "
                         "of its wall clock with the GPU idle; the dump is "
                         "written before them and scorers read it separately.")
    ap.add_argument("--no-survey", action="store_true",
                    help="skip the survey-rho evaluation (~1 min); it is the "
                         "only criterion comparable against pipeline.train_rating")
    ap.add_argument("--lr-net", type=float, default=LR_NET,
                    help="learning rate for the trunk, heads and axis map")
    ap.add_argument("--lr-field", type=float, default=LR_FIELD,
                    help="learning rate for the free per-user fields. Must run "
                         "well above --lr-net; equal rates diverge late")
    ap.add_argument("--window-months", type=int, default=0,
                    help="ability window WIDTH in months, sliding one month at a "
                         "time. DEFAULT 0 = one unit per user, the whole history: "
                         "measured best (both seeds, both criteria, monotone in "
                         "how much windowing there is) and the simplest object. "
                         "Pass 2 only to reproduce production's ability structure "
                         "for a specific diagnostic — NOT for benchmarking. Each "
                         "model is compared at ITS OWN best; production's eval "
                         "rows duplicate each attempt across two overlapping "
                         "profiles, and that denominator difference is corrected "
                         "at SCORING time by comparing per attempt, never by "
                         "handicapping this model into a worse window.")
    ap.add_argument("--disjoint-blocks", action="store_true",
                    help="disjoint blocks of that width instead of a sliding "
                         "window. Production slides, which writes every attempt "
                         "into `--window-months` separate units")
    ap.add_argument("--hidden", type=int, nargs="+", default=list(DIFF_HIDDEN),
                    help="trunk widths (default: the rating trainer's). Widen "
                         "the relu control with this to parameter-match SwiGLU, "
                         "which doubles each layer.")
    ap.add_argument("--act", choices=("swiglu", "relu"), default="swiglu",
                    help="trunk activation. SwiGLU by default; it doubles the "
                         "trunk's parameters at a given width, so read it "
                         "against the relu control rather than alone.")
    ap.add_argument("--global-slope", action="store_true",
                    help="one discrimination for the whole corpus on dimension "
                         "1 (the Rasch reduction), difficulty still per-problem. "
                         "Tests whether the per-problem slope head earns its "
                         "place: methodology 1.11 finds most of the phase-2 "
                         "multiplier is rating-axis gauge rather than a "
                         "statement about the slope, and per-problem slope has "
                         "been recorded as largely assignment noise. If "
                         "held-out PROBLEM logloss is unchanged, the head is "
                         "fitting noise and the model gets simpler.")
    args = ap.parse_args()

    import torch
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # --- features: the RATING trainer's set. The axis trainer's set additionally
    # carries the rating model's own outputs, which would be circular here.
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
    print(f"loaded in {time.time() - t0:.0f}s: {len(keys)} problems, feat "
          f"{fdim} (dropout region {input_dim})", flush=True)

    # --- attempts, with BOTH groupings: profiles for ability (it drifts),
    # whole-user for the axis positions (a skill profile should not, and a
    # 2-month block carries far too few attempts to fit one).
    df = load_first_time_attempts()
    df = df[df["size"].isin(SIZES)].copy()
    if args.drop_recent_frac > 0:
        # SIMULATE THE PREVIOUS REFIT. Cutting by DATE is what makes this an
        # earlier run of this same script rather than an ablation: it removes
        # the accounts that had not arrived yet AND truncates the histories of
        # the ones that had, which is both halves of what a refit adds back.
        # Dropping a random fraction of USERS instead is a different and much
        # harsher perturbation — it takes long-history accounts, and those are
        # the ones the axes are fitted from (the axis block pays nothing below
        # ~500 attempts), while the users a refit actually gains are new and
        # short. The cut sits BEFORE the outcome filter so the filter applies to
        # the truncated data, exactly as the earlier run would have applied it.
        ts = df["created_at"].astype(str)
        cut = ts.quantile(1.0 - args.drop_recent_frac, interpolation="lower")
        before = len(df)
        df = df[ts <= cut].copy()
        print(f"  TIME CUT: dropped the most recent {args.drop_recent_frac:.0%} "
              f"of attempts (> {cut}) — {before} -> {len(df)} rows, "
              f"{df['user_id'].nunique()} users remain", flush=True)
    g = df.groupby("user_id")["success"].agg(c="sum", n="count")
    g["w"] = g["n"] - g["c"]
    df = df[df["user_id"].isin(set(g.index[(g["c"] >= MIN_BOTH_OUTCOMES)
                                           & (g["w"] >= MIN_BOTH_OUTCOMES)]))].copy()
    prob2size = dict(zip(df["problem"], df["size"]))
    prof_df = build_profiles_at(df, args.window_months,
                                overlap=not args.disjoint_blocks)
    # The SAME condition at the level the ability actually lives at. One win and
    # one loss for the user does not give one of each in every window, and it is
    # the profile that carries theta.
    pg = prof_df.groupby("profile")["success"].agg(c="sum", n="count")
    pg["w"] = pg["n"] - pg["c"]
    keep_prof = set(pg.index[(pg["c"] >= MIN_BOTH_OUTCOMES)
                             & (pg["w"] >= MIN_BOTH_OUTCOMES)])
    prof_df = prof_df[prof_df["profile"].isin(keep_prof)]
    prof_df = prof_df[prof_df["problem"].map(
        lambda t: (prob2size.get(t), t) in key_idx)].reset_index(drop=True)
    profiles = pd.Categorical(prof_df["profile"])
    prof_idx = profiles.codes.astype(np.int64)
    # build_profiles names a profile "{user_id[:8]}_{YYYY-MM}", so the user is
    # the first 8 characters — NOT a split on a separator, since the id itself
    # may contain one. Getting this wrong silently fits the axis positions per
    # 2-MONTH BLOCK instead of per user, on roughly eight attempts each, which
    # is far too little to estimate a skill profile and is the exact granularity
    # mismatch the two-stage design avoids.
    user_of_profile = [str(p)[:8] for p in profiles.categories]
    uids = sorted(set(user_of_profile))
    uindex = {u: i for i, u in enumerate(uids)}
    prof_user = np.array([uindex[u] for u in user_of_profile], dtype=np.int64)
    user_idx = prof_user[prof_idx]
    y = prof_df["success"].to_numpy(np.float32)
    prob_idx = prof_df["problem"].map(
        lambda t: key_idx[(prob2size[t], t)]).to_numpy(np.int64)
    print(f"  {len(prof_df)} rows | {len(profiles.categories)} profiles | "
          f"{len(uids)} users | base rate {y.mean():.3f}", flush=True)

    # --- held-out PROBLEMS for the rating criterion (the axes get their own
    # held-out-USER criterion below; the two answer different questions)
    # The SHARED held-out generator, not a local draw. auxiliary.rating_dev.
    # holdout_set is what train_rating_staged --holdout-problem-frac excludes and
    # what validate_ratings evaluates on, and it emits a content hash so two runs
    # can confirm by hash that they assumed the identical set. Using it here is
    # the whole reason a joint-model number can be put beside a production one.
    #
    # Its universe is problems with >=1 first-time attempt, NOT every problem
    # with features — an unattempted problem gets a feature-prior rating but
    # carries no held-out measurement, so including it would dilute the metric
    # with rows that cannot move it. A local draw over all 153,720 keys is a
    # different universe and its numbers are not comparable to anything.
    from auxiliary.rating_dev.holdout_set import held_out_keys
    if args.holdout_problem_frac > 0:
        ho_keys, ho_hash = held_out_keys(args.holdout_seed,
                                         args.holdout_problem_frac)
    else:
        ho_keys, ho_hash = [], "none"
    ho_set = set(ho_keys)
    ho = np.array([k in ho_set for k in keys])
    # The stop split is carved OUT of the held-out set by a separate deterministic
    # draw, so the reported split is never the one the stopping rule watched.
    # Production has no early stopping and so needs no such carve-out; the
    # like-for-like comparison is therefore production's whole held-out set
    # against this model's REPORT split, both being random subsets of the same
    # universe.
    sub = np.random.default_rng(args.holdout_seed + 1000).random(len(keys))
    stop_p = ho & (sub < STOP_SPLIT_FRAC)
    rep_p = ho & ~stop_p
    # USER and ROW holdouts, ORed with the problem holdout. All three exclude
    # rows from TRAINING; which of them a given evaluation should read back is
    # the evaluator's business, not the trainer's.
    extra_ho = np.zeros(len(prof_df), bool)
    uid_arr = np.array(uids)
    hu = set()
    if args.holdout_user_frac > 0:
        from auxiliary.rating_dev.holdout_set import held_out_users
        hu_list, hu_hash = held_out_users(args.holdout_seed,
                                          args.holdout_user_frac)
        hu = set(hu_list)
        extra_ho |= np.array([u in hu for u in uid_arr[user_idx]], bool)
        print(f"  HELD-OUT USERS: hash {hu_hash} | {len(hu_list)} accounts, "
              f"{int(extra_ho.sum())} rows", flush=True)
    if args.holdout_row_frac > 0:
        from auxiliary.rating_dev.holdout_set import held_out_rows
        hr_list, hr_hash = held_out_rows(args.holdout_seed,
                                         args.holdout_row_frac)
        hr = set(hr_list)
        m_r = np.array([(uid_arr[ui], str(t)) in hr for ui, t
                        in zip(user_idx, prof_df["problem"])], bool)
        extra_ho |= m_r
        print(f"  HELD-OUT ROWS: hash {hr_hash} | {len(hr_list)} attempts, "
              f"{int(m_r.sum())} profile rows (both window copies masked)",
              flush=True)
    # WHICH ROWS THE STOPPING RULE READS — and it must be rows whose ABILITY
    # THIS MODEL FITTED. A held-out user's theta embedding gets no gradient, so
    # it sits at its random init; scoring their rows measures theta-at-init, not
    # the model, and stopping on that signal stops on noise. (Observed: a
    # user-holdout run's stop split read 0.5313 where every comparable run read
    # ~0.50.) The EVALUATION is unaffected — eval_user_holdout fits ability
    # online from zero, which is the point of it — but the stopping rule cannot.
    #
    # So the stop signal always comes from rows the model can actually score:
    # the problem holdout when there is one, otherwise a small extra draw of
    # TRAINING-user rows set aside for this and excluded from training.
    tr_rows = (~ho[prob_idx]) & (~extra_ho)
    if args.holdout_problem_frac > 0:
        ho_rows, stop_rows, rep_rows = ho[prob_idx], stop_p[prob_idx], rep_p[prob_idx]
    else:
        sub_r = np.random.default_rng(args.holdout_seed + 2000).random(len(prof_df))
        stop_rows = tr_rows & (sub_r < STOP_ROW_FRAC)
        tr_rows = tr_rows & ~stop_rows
        rep_rows = extra_ho          # reported by eval_user_holdout, not here
        ho_rows = extra_ho
        print(f"  stop signal: {int(stop_rows.sum())} TRAINING-user rows held "
              f"aside (their abilities ARE fitted, so the signal is real); "
              f"the held-out users' {int(extra_ho.sum())} rows are scored "
              f"separately by eval_user_holdout", flush=True)
    if args.holdout_problem_frac > 0:
        print(f"  held-out PROBLEMS hash {ho_hash} | {len(ho_keys)} held out, "
              f"{int(ho.sum())} featured -> stop {int(stop_p.sum())} / report "
              f"{int(rep_p.sum())}; train rows {int(tr_rows.sum())}", flush=True)
    else:
        # Printing "stop 0 / report 0" here reads like a broken run when the
        # evaluation set is simply coming from a different holdout; the carve-out
        # a few lines above already reported the real split.
        print(f"  problem holdout OFF; train rows {int(tr_rows.sum())}",
              flush=True)

    # WINDOWING-SENSITIVE USERS — the only ones a window-length arm can reach.
    # A user whose attempts fall inside a single calendar month gets, under EVERY
    # arrangement (production's sliding window, a disjoint block, the whole
    # history), ability units holding exactly that user's attempts; the arms
    # differ only in how many times those rows are counted, never in content. So
    # their rows carry no information about window length while contributing full
    # variance to a pooled criterion. Measured on the local DB: two-thirds of
    # fitted users are single-month and carry 41% of attempt mass, and averaging
    # a minority effect over that many inert rows is close to the worst available
    # instrument. The subgroup is defined from the RAW attempt times, so it is
    # identical across arms — defining it per-arm would compare different row
    # sets and answer nothing.
    _u8 = df["user_id"].astype(str).str[:8]
    _span = df["created_at"].astype(str).str[:7].groupby(_u8).nunique()
    _multi = set(_span.index[_span > 1])
    sens_user = np.array([u in _multi for u in uids], dtype=bool)
    sens_row = sens_user[user_idx]
    print(f"  windowing-sensitive: {int(sens_user.sum())}/{len(uids)} users, "
          f"{sens_row.mean():.0%} of rows", flush=True)

    tfeat = torch.tensor(feat, device=device)
    tp = torch.tensor(prob_idx, device=device)
    tprof = torch.tensor(prof_idx, device=device)
    tu = torch.tensor(user_idx, device=device)
    ty = torch.tensor(y, device=device)
    tr = torch.tensor(np.where(tr_rows)[0], device=device)
    i_stop = torch.tensor(np.where(stop_rows)[0], device=device)
    i_rep = torch.tensor(np.where(rep_rows)[0], device=device)
    i_rep_s = torch.tensor(np.where(rep_rows & sens_row)[0],
                           device=device)
    # The FULL held-out set — production's evaluation row set exactly, since it
    # carves out no stop split. Mildly optimistic here because a third of it
    # steered the stopping rule, so quote the report split as this model's honest
    # number and this one only when putting it beside production.
    i_ho = torch.tensor(np.where(ho_rows)[0], device=device)

    # --- axis warm start, from the two-stage machinery (same construction), on
    # the axis trainer's feature matrix restricted to these columns' semantics.
    # --axis-params-dir reads the rating model's outputs from a named fit rather
    # than the shipped ratings/, which was trained on EVERY user and every
    # problem. Leaving it unset under a holdout means the warm start is derived
    # from parameters informed by the rows about to be scored.
    feats_ax, rank_ax, params_ax, _ad = build_axis_features(
        params_dir=args.axis_params_dir)
    tids_ax = sorted(feats_ax)
    ti_ax = {t: i for i, t in enumerate(tids_ax)}
    Fax = np.stack([feats_ax[t] for t in tids_ax])
    Fax = ((Fax - Fax.mean(0)) / (Fax.std(0) + 1e-9)).astype(np.float32)
    # HELD-OUT USERS ARE EXCLUDED HERE TOO, and this is a fix rather than an
    # option. --holdout-user-frac reached only the SGD row mask, so the warm
    # start was still built from the held-out accounts' own attempts — measured
    # at 17.2% and 13.6% of its users on the two folds. Protecting the training
    # rows and not the initialization protects nothing.
    rbu = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in ti_ax and str(u) not in hu:
            rbu[u].append((p, int(bool(s))))
    ax_users = sorted(u for u, a in rbu.items() if len(a) >= MIN_USER_ATTEMPTS)
    if hu:
        print(f"  warm-start users: {len(ax_users)} (held-out accounts "
              f"excluded)", flush=True)
    lapse_c = load_client_lapse(0.0328)
    fold_of = _stratified_folds(ax_users, rbu)

    for seed in args.seeds:
        # the warm start is built on TRAIN-fold users only, as production does
        # The warm start is built on TRAIN-fold users only. That was already
        # fold-clean, which is exactly why the leak below went unnoticed for so
        # long: protecting the INITIALISATION does nothing about the fit.
        ax_fold = args.axis_holdout_fold
        held_fold = ax_fold if ax_fold in (0, 1) else 0
        tu_ax = [u for u in ax_users if fold_of[u] != held_fold]
        # K=0 is the ABLATION: no axis block at all, i.e. the joint fit reduced
        # to its rating heads. It answers whether the axes cost the rating, which
        # no amount of staring at the axis-on/axis-off difference can, since that
        # difference is measured with the axes already having shaped the fit.
        if args.k > 0 and args.warm_start == "eigen":
            gi_full = eigen_directions(rbu, tu_ax, Fax, ti_ax, rank_ax,
                                       params_ax, lapse_c, args.k)
            gi = gi_full[:, :fdim] if gi_full.shape[1] >= fdim else None
        else:
            gi = None
        m = JointMIRT(fdim, len(profiles.categories), len(uids), args.k,
                      input_dim, device, seed, g_init=gi,
                      global_slope=args.global_slope, act=args.act,
                      hidden=tuple(args.hidden),
                      axis_dropout=args.axis_dropout,
                      axis_reads_rating=not args.axis_blind,
                      axis_from_trunk=not args.axis_from_features,
                      axis_trunk_coupled=not args.axis_trunk_detached)
        if args.init_from:
            # CONTINUE the previous fit rather than rediscovering a basis. The
            # axis objective is invariant to rotating the K-dim latent space, so
            # an independent fit has no reason to land in the previous fit's
            # basis and measurably does not; carrying the map over is what makes
            # a motif NAME survive a refit. Problem-side modules transfer whole.
            # theta and v are per-account, so they scatter BY NAME and accounts
            # that did not exist yet start at zero — which is what they would
            # have been initialized to anyway.
            st = torch.load(args.init_from, map_location=device)
            if int(st["k"]) != args.k:
                raise SystemExit(f"--init-from has K={st['k']}, this run K={args.k}")
            m.trunk.load_state_dict(st["trunk"])
            m.head.load_state_dict(st["head"])
            m.g_lin.load_state_dict(st["g_lin"])
            with torch.no_grad():
                m.lapse_raw.copy_(st["lapse_raw"].to(device))
                m.a_raw.copy_(st["a_raw"].to(device))
                m.b_mu.copy_(st["b_mu"].to(device))
                m.b_sd.copy_(st["b_sd"].to(device))
                pidx = {str(x): i for i, x in enumerate(st["profiles"])}
                th, nth = st["theta"].to(device), 0
                for j, pr_ in enumerate(profiles.categories):
                    i = pidx.get(str(pr_))
                    if i is not None:
                        m.theta.weight[j].copy_(th[i]); nth += 1
                uidx = {str(x): i for i, x in enumerate(st["users"])}
                vv, nv = st["v"].to(device), 0
                for j, u_ in enumerate(uids):
                    i = uidx.get(str(u_))
                    if i is not None:
                        m.v[j].copy_(vv[i]); nv += 1
            print(f"  CONTINUED from {os.path.basename(args.init_from)}: "
                  f"trunk/heads/axis map whole, {nth}/{len(profiles.categories)} "
                  f"profiles and {nv}/{len(uids)} users carried, the rest at zero",
                  flush=True)
        tr_eff = tr
        if ax_fold in (0, 1):
            # HOLDING A USER OUT MEANS HOLDING THEM OUT. An earlier version also
            # offered a "mask" mode that kept their rows and only pinned their
            # axis positions at zero, on the theory that it matched build_axes
            # (whose 4PL saw every user and whose axis stage alone does CV). It
            # does not: in a JOINT fit the rating heads co-adapt with the axes,
            # so those rows still reach the loadings indirectly through a channel
            # the two-stage pipeline structurally cannot have. It is removed
            # rather than defaulted off — a leaky protocol available behind a
            # flag is one forgotten flag away from a wrong number that looks fine.
            mask = np.array([0.0 if fold_of.get(u, -1) == ax_fold else 1.0
                             for u in uids], dtype=np.float32)
            keep = mask[user_idx] > 0
            tr_eff = torch.tensor(np.where(tr_rows & keep)[0], device=device)
            print(f"  AXIS HOLDOUT: fold {ax_fold}, {int((mask == 0).sum())}/"
                  f"{len(uids)} users excluded entirely — {len(tr)} -> "
                  f"{len(tr_eff)} training rows. Note this also removes them "
                  f"from the RATING fit, which build_axes does not do to its "
                  f"users; the resulting axis gain is biased AGAINST this model.",
                  flush=True)

        opt = torch.optim.Adam(m.param_groups(args.lr_net, args.lr_field))
        base_lrs = [gp["lr"] for gp in opt.param_groups]

        def _extract():
            """Per-problem parameters as of NOW, always in eval mode — the gauge
            reads its EMA there, and a dropout sample is not the model."""
            was = m.training
            m.train(False)
            with torch.no_grad():
                b_a, a_a, c_a, g_a = m.problem_params(tfeat)
                lm = float(m.lapse())
            m.train(was)
            b_, a_ = b_a.cpu().numpy(), a_a.cpu().numpy()
            c_, g_ = c_a.cpu().numpy(), g_a.cpu().numpy()
            r_ = b_ + clear_logit_vec(c_, lm) / np.maximum(a_, 1e-6)
            return b_, a_, c_, g_, lm, r_

        def _write_dump(path):
            b_, a_, c_, g_, lm, r_ = _extract()
            tids_o = [t for (sz, t) in key_idx]
            szs_o = [sz for (sz, t) in key_idx]
            idxs = [key_idx[(sz, t)] for sz, t in zip(szs_o, tids_o)]
            np.savez_compressed(
                path, tid=np.array(tids_o), size=np.array(szs_o),
                rating=r_[idxs], difficulty=b_[idxs], slope=a_[idxs],
                floor=c_[idxs], loading=g_[idxs], lapse=np.array(lm),
                theta=m.theta.weight.detach().cpu().numpy().ravel(),
                profile=np.array([str(p) for p in profiles.categories]),
                v=m.v.detach().cpu().numpy(), user=np.array(uids))

        # threshold in the slope head's own (pre-softplus) space
        o_tau = float(np.log(np.expm1(SHALLOW_TAU)))
        n = len(tr_eff)
        best, best_state, flat, best_epoch = float("inf"), None, 0, 0
        stopped_early = False
        # BURN-IN: epochs trained with v pinned at zero, after which the warm
        # start is read off the model's OWN residuals. Not a separate stage —
        # at v = 0 the loading gradient is identically zero, so these epochs are
        # the axis-free baseline the residual scores are defined against, and
        # they are the rating heads' own fit either way.
        #
        # ITS LENGTH IS A CONVERGENCE CRITERION, NOT A CONSTANT (--burn-in 0).
        # Both ends of a fixed count corrupt the residuals, in opposite ways and
        # measurably: too early and they are dominated by rating ERROR, too late
        # by rating OVERFIT, since the baseline has its own optimum to pass. So
        # the burn-in runs under the same patience rule as everything else and
        # hands over at its own best epoch, restored — which is exactly what the
        # two-stage pipeline does, train_rating converging before build_axes
        # reads it, and is why the borrowed warm start works at all.
        burnin_on = (args.k > 0 and args.warm_start == "burnin"
                     and not args.init_from)
        burn_fixed = args.burn_in if (burnin_on and args.burn_in > 0) else 0
        in_burn = burnin_on
        if in_burn:
            m.v.requires_grad_(False)
        print(f"\n=== joint fit seed {seed} | {args.epochs} epochs | K={args.k} "
              f" | warm start {args.warm_start}"
              + (f" after {burn_fixed} burn-in epochs" if burn_fixed
                 else " after a burn-in to convergence" if in_burn else "")
              + " ===", flush=True)
        for ep in range(args.epochs):
            if args.lr_warmup_epochs:
                # Ramp from one step's worth up to full, never from exactly 0 —
                # a first epoch at lr 0 is an epoch of pure bookkeeping.
                scale = min(1.0, (ep + 1) / args.lr_warmup_epochs)
                for gp, b in zip(opt.param_groups, base_lrs):
                    gp["lr"] = b * scale
            m.train(True)
            perm = tr_eff[torch.randperm(n, device=device)]
            for i in range(0, n, BATCH):
                bi = perm[i:i + BATCH]
                pred, gh = m.forward(tprof[bi], tu[bi], tfeat[tp[bi]])
                loss = torch.nn.functional.binary_cross_entropy(
                    pred, ty[bi], reduction="none").sum() / len(bi)
                # While v is pinned at zero the axis terms are either identically
                # zero or act on a g that the warm start is about to overwrite.
                if not in_burn:
                    loss = loss + args.v_ridge * (m.v[tu[bi]] ** 2).sum() / len(bi)
                    loss = loss + args.v_mean_penalty * (m.v.mean(dim=0) ** 2).sum()
                    Mc = gh - gh.mean(dim=0, keepdim=True)
                    Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
                    Cm = Mn.T @ Mn
                    loss = loss + ORTHO_WEIGHT * ((Cm - torch.eye(
                        args.k, device=device)) ** 2).sum()
                # divergence barrier, over the FULL corpus with dropout off
                if not args.no_barriers:
                  m.train(False)
                  short = torch.nn.functional.softplus(
                      o_tau - m.slope_logit(tfeat)).pow(2).mean()
                  loss = loss + args.shallow_w * short
                  m.train(True)
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % CHECK_EVERY == 0 or ep == args.epochs - 1:
                m.train(False)
                with torch.no_grad():
                    pr, _ = m.forward(tprof, tu, tfeat[tp])
                    bce_tr = float(torch.nn.functional.binary_cross_entropy(
                        pr[tr], ty[tr]))
                    bce_st = float(torch.nn.functional.binary_cross_entropy(
                        pr[i_stop], ty[i_stop]))
                improved = bce_st < best - 1e-6
                if improved:
                    best = bce_st
                    best_epoch = ep + 1
                    best_state = [q.detach().clone() for q in m.params()]
                flat = 0 if improved else flat + 1
                print(f"  {'[burn] ' if in_burn else ''}epoch {ep + 1:4d}  "
                      f"train {bce_tr:.4f}  stop-split {bce_st:.4f}  "
                      f"best {best:.4f}@{best_epoch}  "
                      f"stale {flat}/{args.flat_patience}", flush=True)
                if in_burn:
                    # The burn-in ends at its fixed count, or — with none given —
                    # at its OWN convergence under the same patience rule. A
                    # fixed count rounds up to the next check.
                    done = ((ep + 1) >= burn_fixed if burn_fixed
                            else flat >= args.flat_patience)
                    if done:
                        # NO REWIND to the axis-free best, though it is tempting
                        # by analogy with the two-stage pipeline. There the fit
                        # ENDS at its best epoch, so there is no optimizer left
                        # to keep consistent; here training continues, and
                        # restoring params while Adam's moments still describe
                        # the later trajectory resumes the fit with one epoch's
                        # weights driven by another's momentum. Measured: the
                        # rewind costs the axis column 1.4 mnat and quadruples
                        # its seed spread against handing over in place.
                        if args.init_directions:
                            bank = np.load(args.init_directions,
                                           allow_pickle=False)
                            brow = {f"{sz}:{t}": i for i, (sz, t) in
                                    enumerate(zip(bank["size"], bank["tid"]))}
                            BL = np.asarray(bank["loading"], np.float64)
                            if BL.shape[1] != args.k:
                                raise SystemExit(
                                    f"--init-directions has K={BL.shape[1]}, "
                                    f"this run K={args.k}")
                            rows, tgt = [], []
                            for (sz, t), j in key_idx.items():
                                i = brow.get(f"{sz}:{t}")
                                if i is not None:
                                    rows.append(j); tgt.append(BL[i])
                            H = m.axis_input(tfeat)[rows].cpu().numpy().astype(
                                np.float64)
                            Y = np.asarray(tgt)
                            # The bank's SCALE is arbitrary — a per-axis rescale
                            # is absorbed by v — so match direction only.
                            Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
                            W, *_ = np.linalg.lstsq(H, Y, rcond=None)
                            r2 = [1 - ((Y[:, kk] - H @ W[:, kk]) ** 2).sum()
                                  / (Y[:, kk] ** 2).sum() for kk in range(args.k)]
                            with torch.no_grad():
                                m.g_lin.weight.copy_(torch.tensor(
                                    W.T, dtype=torch.float32, device=device))
                            m.v.requires_grad_(True)
                            print(f"  FROZEN GAUGE at epoch {ep + 1}: axis map "
                                  f"fitted to {len(rows)} banked loadings from "
                                  f"{os.path.basename(args.init_directions)}, "
                                  f"per-axis R^2 "
                                  f"{' '.join(f'{v:.3f}' for v in r2)}; "
                                  f"v released", flush=True)
                        else:
                            gi_bi, n_seen = burn_in_directions(
                                m, tfeat, tp, tprof, tu, ty, tr_eff, args.k,
                                len(uids), batch=BATCH,
                                xfeat=m.axis_input(tfeat))
                            with torch.no_grad():
                                m.g_lin.weight.copy_(
                                    torch.tensor(gi_bi, device=device))
                            m.v.requires_grad_(True)
                            print(f"  warm start taken IN PLACE at epoch {ep + 1} "
                                  f"(axis-free fit converged, best {best:.4f}@"
                                  f"{best_epoch}) over {n_seen} training users' own "
                                  f"residuals; v released", flush=True)
                        # The burn-in's convergence says nothing about the joint
                        # fit's, so its patience is discarded rather than carried.
                        best, best_state, flat, best_epoch = \
                            float("inf"), None, 0, 0
                        in_burn = False
                    continue        # never END the run during the burn-in
                if args.dump_every and args.dump_params and \
                        (ep + 1) % args.dump_every == 0:
                    _write_dump(f"{args.dump_params[:-4]}.ep{ep + 1:05d}.npz")
                if flat >= args.flat_patience:
                    print(f"  stopped at epoch {ep + 1}; best stop-split "
                          f"{best:.4f} at epoch {best_epoch}; restoring it",
                          flush=True)
                    stopped_early = True
                    break
        if not stopped_early:
            # Under --dump-every, running to the cap is the POINT: the diagnostic
            # deliberately outruns the stopping rule to see what lies past it, so
            # the cap is an endpoint there rather than a failure.
            if args.dump_every:
                print(f"  reached epoch {args.epochs} with stopping disabled "
                      f"(diagnostic mode); best stop-split {best:.4f} at epoch "
                      f"{best_epoch}", flush=True)
            else:
                raise SystemExit(
                    f"EPOCH CAP HIT: ran the full {args.epochs} epochs without "
                    f"the patience rule firing (best stop-split {best:.4f} at "
                    f"epoch {best_epoch}). The fit never plateaued, so this is "
                    f"not a converged solution and its numbers are not "
                    f"comparable with anything in docs/mirt_benchmark.md. "
                    f"Raise --epochs and re-run.")
        if best_state is not None:
            with torch.no_grad():
                for q, b in zip(m.params(), best_state):
                    q.copy_(b)

        if args.save_model:
            torch.save({"trunk": m.trunk.state_dict(),
                        "head": m.head.state_dict(),
                        "g_lin": m.g_lin.state_dict(),
                        "theta": m.theta.weight.detach().cpu(),
                        "profiles": [str(x) for x in profiles.categories],
                        "v": m.v.detach().cpu(),
                        "users": [str(x) for x in uids],
                        "lapse_raw": m.lapse_raw.detach().cpu(),
                        "a_raw": m.a_raw.detach().cpu(),
                        "b_mu": m.b_mu.cpu(), "b_sd": m.b_sd.cpu(),
                        "k": args.k}, args.save_model)
            print(f"  wrote model -> {args.save_model}", flush=True)

        # --- criterion 1: held-out problems (the rating side), on the split
        # the stopping rule never saw
        m.train(False)
        with torch.no_grad():
            pr, _ = m.forward(tprof, tu, tfeat[tp])
            bce_rep = float(torch.nn.functional.binary_cross_entropy(
                pr[i_rep], ty[i_rep]))
        # The axes' own contribution ON HELD-OUT PROBLEMS: the same rows scored
        # with the axis term zeroed. This is the criterion the two-stage harness
        # structurally cannot produce — axes_ll_gain walks a user's own history,
        # so every problem it scores is by construction one somebody attempted,
        # and an axis that is accurate where people have played and noise
        # everywhere else passes it. The loadings ship for the WHOLE corpus,
        # about a third of which has never been attempted and sits in a
        # systematically different region of feature space, so this is the test
        # that speaks to what actually ships.
        with torch.no_grad():
            v_keep = m.v.detach().clone()
            m.v.zero_()
            pr0, _ = m.forward(tprof, tu, tfeat[tp])
            bce_rep0 = float(torch.nn.functional.binary_cross_entropy(
                pr0[i_rep], ty[i_rep]))
            m.v.copy_(v_keep)
        # THREE separate quantities; quoting the full-model BCE as "the rating
        # criterion" conflates the first two, because better axes improve it too.
        #   RATING     = axes off. What the rating is asked to do alone, and what
        #                the client's rank estimator actually reads — it uses
        #                rating/slope/floor and never touches the axes.
        #   AXIS-PROB  = the difference. The axes' marginal value on problems
        #                nobody attempted.
        #   (AXIS-USER is reported below against the SHIPPED 4PL baseline, so it
        #    is comparable with every two-stage measurement but says nothing
        #    about THIS model's rating.)
        with torch.no_grad():
            bce_rep0_s = float(torch.nn.functional.binary_cross_entropy(
                pr0[i_rep_s], ty[i_rep_s]))
            bce_ho0 = float(torch.nn.functional.binary_cross_entropy(
                pr0[i_ho], ty[i_ho]))
            bce_ho = float(torch.nn.functional.binary_cross_entropy(
                pr[i_ho], ty[i_ho]))
        print(f"  FULL held-out set (production's row set): axes off "
              f"{bce_ho0:.4f} | axes on {bce_ho:.4f}  -> axis "
              f"{(bce_ho0 - bce_ho) * 1000:+.3f} mnat   [{len(i_ho)} rows]",
              flush=True)
        print(f"  RATING criterion  (held-out problems, axes off): "
              f"{bce_rep0:.4f}", flush=True)
        # The comparison a WINDOW-LENGTH arm should be read on. Identical row set
        # across arms; the pooled figure above dilutes it with users no window
        # choice can affect.
        print(f"  RATING criterion  (held-out problems, axes off, "
              f"windowing-sensitive users only): {bce_rep0_s:.4f}  "
              f"[{len(i_rep_s)}/{len(i_rep)} rows]", flush=True)
        print(f"  AXIS criterion    (held-out problems, marginal): "
              f"{(bce_rep0 - bce_rep) * 1000:+.3f} mnat/attempt   "
              f"[full model {bce_rep:.4f}]", flush=True)
        # One extraction path, shared with the periodic dumps — the rating is the
        # ability at which the curve reaches CALIBRATION_P, and with the gauge on
        # difficulty that is a whitened head output PLUS a slope-dependent
        # correction, so the slope perturbs the rating rather than scaling it.
        b_np, a_np, c_np, g_np, lam, rating = _extract()
        # AN EXTREME RATING IS NOT A SYMPTOM AND MUST NOT BE READ AS ONE. The
        # corpus holds genuinely trivial problems and a KataGo-hard tail nobody
        # clears, and the rank coordinate is deliberately unbounded, so extreme
        # values at both ends are what a CORRECT model of extreme problems
        # produces. A magnitude cannot distinguish a real difficulty from a
        # collapsed slope, so it is not evidence for either.
        #
        # The quantity that CAN distinguish them is the slope distribution and
        # the share pinned at the clamp, so that is what to read. The rating
        # min/max below is printed for a different purpose entirely: it is the
        # numerical RANGE that downstream solves inherit — axes_ll_gain builds a
        # theta grid spanning it and feeds it to a per-user IRLS as a design
        # column — so it is diagnostic of CONDITIONING, not of the model.
        pct = np.percentile(rating, [0.1, 1, 50, 99, 99.9]).round(2)
        apct = np.percentile(a_np, [0.1, 1, 50, 99]).round(4)
        at_floor = float(np.mean(a_np <= SLOPE_FLOOR * 1.01))
        print(f"  rating percentiles [.1,1,50,99,99.9]: {pct}   "
              f"range [{rating.min():.1f}, {rating.max():.1f}] (conditioning, "
              f"not a symptom)", flush=True)
        print(f"  slope percentiles  [.1,1,50,99]: {apct}   "
              f"at floor: {at_floor:.2%}"
              + ("   <-- slopes collapsing, not difficulty" if at_floor > 0.01
                 else ""), flush=True)

        # Persist the per-problem parameters. The model is otherwise thrown away
        # at exit, so every downstream question — a paired bootstrap against
        # production's survey rho, the dropout-compression multiplier, whether an
        # extreme rating comes from a small slope or a large difficulty — has
        # cost a full re-run. It does not, once this file exists.
        if args.dump_params:
            _write_dump(args.dump_params)
            print(f"  wrote per-problem params -> {args.dump_params}", flush=True)
        if args.fit_only:
            print(f"  fit-only: skipping post-fit evaluation "
                  f"(total {time.time() - t0:.0f}s)", flush=True)
            continue

        # --- SURVEY RHO: the ONLY criterion on which this model and
        # pipeline.train_rating can be compared fairly.
        #
        # Held-out problem logloss cannot compare them. The production trainer
        # fits every problem in the corpus, so there is no split on which it is
        # out-of-sample without retraining it — and a retrained production model
        # is not the shipped one. Survey declarations, by contrast, are external
        # to BOTH fits: neither model ever sees them, so scoring both against
        # them is like-for-like.
        #
        # Reuses auxiliary.rating_dev.validate_ratings.survey_weighted_rho
        # verbatim — the single definition of this statistic, per-server Spearman
        # averaged with user-count weights. Per-server is offset AND scale
        # invariant, so a gauge difference between the two models cannot affect
        # it; pooling across servers would not have that property, and the two
        # models are on different gauges by construction.
        #
        # QUOTE THE TIME-ADJUSTED FIGURE. Every server coupling was fitted
        # against time-adjusted ratings, so the untimed number compares two sides
        # in different units — and it fails silently, because the wrong value is
        # still a plausible one. The framework cross-fits its own alpha rather
        # than reading the shipped one, which would be stale for a fresh fit.
        if not args.no_survey:
            from auxiliary.rating_dev.validate_ratings import survey_weighted_rho
            ratemap = {t: float(rating[i]) for (sz, t), i in key_idx.items()}
            slopemap = {t: float(a_np[i]) for (sz, t), i in key_idx.items()}
            t_s = time.time()
            wrho, per, ta = survey_weighted_rho(ratemap, slopemap)
            print(f"  SURVEY rho (base, per-server n-weighted): {wrho:.4f}   "
                  f"[{sum(n for n, _ in per.values())} declarations, "
                  f"{len(per)} servers, {time.time() - t_s:.0f}s]", flush=True)
            if ta and "note" not in ta:
                print(f"  SURVEY rho (TIME-ADJUSTED, cross-fitted): "
                      f"{ta['rho']:.4f}   [base {ta['rho_base']:.4f}, in-sample "
                      f"{ta['rho_insample']:.4f}, alpha {ta['alpha']:+.3f}, "
                      f"{ta.get('n_users')} users]", flush=True)
            elif ta:
                print(f"  SURVEY time-adjust: {ta['note']}", flush=True)
            for s, (n, r) in sorted(per.items(), key=lambda kv: -kv[1][0]):
                print(f"      {s:<12} n={n:<5} rho={r:+.4f}", flush=True)

        # --- criterion 2: held-out users, scored by the SAME harness and the
        # SAME shipped-4PL baseline as every other axis measurement, so the
        # number is comparable rather than merely internally consistent.
        if args.k == 0:
            print("  K=0: no axis block — axis criteria skipped", flush=True)
            print(f"\ntotal {time.time() - t0:.0f}s")
            return
        gmap = {}
        for (sz, t), i in key_idx.items():
            gmap[t] = g_np[i]
        garr = np.zeros((len(tids_ax), args.k), dtype=np.float32)
        hit = 0
        for t, i in ti_ax.items():
            if t in gmap:
                garr[i] = gmap[t]
                hit += 1
        # Evaluate on the fold that was pinned out of the axis fit. When no fold
        # was held out, fold 0 is still scored — but those users DID shape the
        # loadings, so the figures below are in-sample and are labelled as such.
        ev_ax = [u for u in ax_users if fold_of[u] == held_fold]
        held = ax_fold in (0, 1)
        tag = (f"HELD-OUT USERS (fold {ax_fold})" if held
               else "IN-SAMPLE USERS — NOT a holdout")
        # PRIMARY: score against THIS MODEL'S OWN rating/slope/floor. The axes
        # were fitted to explain the residual against it, so scoring them
        # against a different 4PL asks them to explain a residual they were
        # never fitted for — and splitting rating from axes at evaluation time
        # reintroduces exactly the staging a joint fit exists to remove.
        # The baseline design is [1, difficulty] with a FREE per-user
        # coefficient, and its job is to let the baseline own a per-user
        # difficulty slope so that an axis merely loading on difficulty earns
        # zero gain and fails selection. The column therefore has to be
        # difficulty; it does not have to be RANK. Using this model's own rating
        # keeps it internal, where mapping through the shipped rating->rank curve
        # would import a curve fitted against different ratings to produce a
        # column whose units the free coefficient absorbs anyway.
        params_own, rank_own = {}, {}
        for (sz, t), i in key_idx.items():
            params_own[t] = (float(rating[i]), float(a_np[i]), float(c_np[i]))
            rank_own[t] = float(rating[i])
        have = [t for t in ti_ax if t in params_own]
        gain_o, n_o, gu_o, nu_o = axes_ll_gain(
            rbu, ev_ax, garr, rank_own, params_own, ti_ax, float(lam))
        print(f"  AXIS criterion    ({tag}, vs THIS MODEL's own 4PL): "
              f"pooled {gain_o / n_o * 1000:.3f}  per-user {gu_o:.3f}  "
              f"({nu_o} users, {len(have)} problems priced)", flush=True)
        # SECONDARY: the shipped 4PL as a fixed reference, which holds the
        # baseline constant across models and so isolates LOADING quality —
        # comparable with every two-stage measurement, but not a statement about
        # this model as a whole.
        gain, nn_, gu, nu = axes_ll_gain(rbu, ev_ax, garr, rank_ax, params_ax,
                                         ti_ax, lapse_c)
        print(f"  axis loadings mapped for {hit}/{len(tids_ax)} problems")
        print(f"  reference         ({tag}, vs SHIPPED 4PL, fixed across "
              f"models): pooled {gain / nn_ * 1000:.3f}  "
              f"per-user {gu:.3f}  ({nu} users)", flush=True)
        if not held:
            print("      ^ these users SHAPED the loadings. Comparable with "
                  "build_axes only under --axis-holdout-fold.", flush=True)

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
