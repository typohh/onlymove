#!/usr/bin/env python3
"""Skill axes fitted the way pipeline.train_rating fits ratings: shared trunk,
column dropout over the wide input region, one joint pass, then a clean-pass
phase 2. Scored by the SAME harness as utils.rating.residual_axes so the two are
directly comparable.

WHY. The production axis map is a bare `Linear(fdim, K)` with no weight decay and
no input dropout, sitting in the same loop as a trunk that has both. Nothing
constrains the weights WITHIN an axis — the ortho penalty, the RMS normalisation
and the v-ridge all constrain axes relative to each other — so on a design whose
columns are near-collinear (adjacent rungs of a bot ladder, neighbouring PCA
components) the fit can put a large positive weight on one column and a large
negative on its neighbour at no cost to the loss. That shows up directly in the
inspection tables as ladders whose signs alternate and whose contributions
cancel. The rating trainer's single largest measured finding is that column
dropout at 0.3 over the wide region is what stops feature-fingerprint
memorization; the axis block never received it.

THE MODEL. The 4PL is an INPUT here, not something being fit:

    logit P = off_4PL(u, p) + alpha_u + beta_u * rank_p + r(x_p)
              + sum_k g_k(x_p) * v_u[k]

`off_4PL` is the frozen production response curve at a per-user theta fitted once
before training, so what the net sees is the residue of the shipped rating model
— the same baseline `axes_ll_gain` scores gain over, which is why training and
evaluation now optimise the same quantity. `r` is population residual difficulty
and is NEVER shipped; it exists so the axes do not have to absorb difficulty the
rating model missed. `alpha_u`/`beta_u` mirror the per-user design the client
fits ([1, rank, axes]).

WHAT IS COPIED FROM train_rating: the shared trunk with decreasing hidden layers
and ReLU+Dropout between them; ONE column-dropout region covering the wide
per-problem blocks with the smooth tail exempt; a single joint pass under Adam at
one learning rate; a phase 2 that freezes the trunk, turns dropout off, refits
the free field and reports its correction; deterministic phase 2 (zero/mean
inits, full-batch, no RNG).

WHAT IS DROPPED, and why each is 4PL-specific rather than an omission: the three
4PL heads and the learned ceiling (there is no response curve here — the curve is
the frozen input); the two divergence barriers (they act on slope and floor); the
rating gauge (it pins the rating/ability scale freedom, which does not exist once
ratings are an input — the axes' own indeterminacy is rotational, and the ortho
penalty is what addresses it).

TWO IDENTIFICATION NOTES, because they decide what phase 2 can and cannot do, and
getting them wrong would ship a vestigial correction.

A GLOBAL SCALE ON THE AXIS BLOCK IS NOT IDENTIFIED. The axis contribution is
`sum_k g_k * v_u[k]`, so scaling g by s and v by 1/s leaves every prediction
unchanged; with the field free there is no fact for such a scalar to measure. Nor
is one needed downstream: the client refits its own per-user coefficients, and
`axes_ll_gain` refits coefficients, estimates its shrinkage prior from those same
fitted coefficients and standardizes the pooled score, so a column rescale is
absorbed three times over. This is the substantive difference from the rating
model, where the slope scale IS identified because the rating axis is held fixed,
and it is why phase 2 here is not a copy of fit_global_slope_scale.

A SCALE ON `r` IS IDENTIFIED, and it is the true analogue. `r` is a per-problem
term with no per-user multiplier to absorb it, so dropout compression of `r`
is measurable exactly as slope compression is in the rating model. It matters
beyond diagnostics: an `r` that under-explains leaves population difficulty on
the table for the axes to absorb, which is the quarantine leak
fit_population_constants' zero-mean check exists to catch. Phase 2 cannot undo
phase-1 contamination, but a large correction here is the signal that `r` needs
more capacity or that the axes should be refit against the corrected offset.

So PHASE 2 is: trunk frozen in eval mode (no dropout, nothing stochastic left),
the per-user field (alpha, beta, v) refit FREE, and one global scalar on `r` that
phase 1 held at unity — pure BCE, full-batch, deterministic. It reports that
scalar, the in-sample gain, and the per-axis ratio of clean-refit coefficient
spread to phase-1 spread, which is the dropout compression of the field itself.
The refit-rather-than-plug-in choice is carried over deliberately: the rating
model's certificate showed fixed abilities attenuate the fitted scale.

The shipped loading convention matches production exactly — g normalised to unit
RMS per column, then scaled by the per-user coefficient spread — except that the
spread is taken from the phase-2 CLEAN refit rather than the dropout-contaminated
phase-1 field.

This is a DEV script and writes nothing to axes/. It fits per fold and scores
with axes_ll_gain, so its numbers sit on the same scale as every other variant
measured on this harness. Reference points, all single-seed:

    eigen warm start, K=3      1.071 pooled / 1.262 per-user
    named-anchor init, K=3     0.995 / 0.944
    eigen warm start, K=8      0.886 / 0.903

Those are orientation, not a control. A verdict needs the current trainer re-run
under the same seeds as this one — --seed is per-run for that reason, and the
repo's bar for a change like this is a 3-seed paired A/B on identical splits.

Usage:
  ./.venv/bin/python3 -m auxiliary.rating_dev.train_axes_trunk --k 3 --seed 0
  ./.venv/bin/python3 -m auxiliary.rating_dev.train_axes_trunk --k 3 --no-phase2
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

from utils.rating.features import load_joint_game_phase  # noqa: E402
from utils.rating.residual_axes import (  # noqa: E402
    CALIBRATION_P, FLOOR_MAX, MIN_USER_ATTEMPTS, _cache_ids_path, _fit_theta,
    _p4pl, _stratified_folds, axes_ll_gain, build_axis_features,
    eigen_directions, load_axis_attempts,
)
from utils.rating.slope_calibration import load_client_lapse  # noqa: E402

# --- mirrored from pipeline.train_rating, same names and same values where the
# quantity is the same one. AXES_HIDDEN matches DIFF_HIDDEN: the trunk is doing
# the same job on the same features.
AXES_HIDDEN = (128, 64)
DROPOUT = 0.3                    # between the trunk's hidden layers
INPUT_DROPOUT = 0.3              # ONE column mask over the wide input region
# residual_axes' split: the network trains an order of magnitude slower than the
# per-user field. train_rating has a single 0.01 because it has no comparable
# split to make — its only field parameter is the ability embedding.
NET_LR = 1e-3
FIELD_LR = 1e-2
# NOT train_rating's 65536. That value is sized for a corpus of over a million
# first-time attempts; this fit sees ~136k rows per fold, where 65536 gives THREE
# minibatches per epoch and 200 epochs comes to 600 optimizer steps against the
# residual-axes trainer's ~6800. Matching the STEP COUNT is what makes the two
# comparable, not matching the batch constant.
BATCH = 8192
EPOCHS = 400
# Convergence stop (opt-in via --flat-tol). CHECK_EVERY is short enough that a
# divergence is caught within a few checks; the threshold is on RELATIVE
# improvement per check, since the same absolute delta means different things at
# BCE 0.47 and 0.40. Reference for calibrating it: at net LR 1e-3 the observed
# per-50-epoch relative improvement ran 2.8% early and was still 0.9% at epoch
# 399 — so a tolerance anywhere near 1% would stop a fit that is plainly still
# descending, and the useful range is well below that.
CHECK_EVERY = 25
FLAT_PATIENCE = 3
MIN_EPOCHS = 100
P2_LR = 0.05
P2_STEPS = 2500
EPS = 1e-6

# --- carried over from utils.rating.residual_axes, which applies ONE penalty to
# the whole [g | r] correlation matrix. It is split here because the two blocks
# are not the same case and only one of them is load-bearing.
#
# AXIS-AXIS is the droppable half. Redundancy between axes is already against the
# objective's interest — two identical axes are a SADDLE, not a minimum, since
# the duplicate buys nothing and any separation toward unexplained signal lowers
# the loss — so the likelihood diversifies without being told to. Measured
# against it: the K=8 fit ran at 0.5 and came out heavily redundant regardless
# (singletons summing ~2.4 pooled against 0.886 for all eight together), because
# this term decorrelates PROBLEM LOADINGS while the redundancy that costs gain is
# between the per-user COEFFICIENTS, which orthogonal loadings do not prevent.
# What it still buys is conditioning the training objective cannot price: the
# client and the eval fit [1, rank, K axes] by IRLS on a trailing WINDOW, where
# mild correlation that is harmless over a full history goes near-singular.
#
# AXIS-R is not droppable. There the likelihood pulls the WRONG way: an axis that
# duplicates r and takes a per-user coefficient centred near 1 fits at least as
# well as r alone, and strictly better if users differ at all in how much
# population residual difficulty costs them. That is the quarantine leak
# fit_population_constants refuses at ship time, and V_MEAN_PENALTY only pins the
# mean coefficient, not the correlation.
ORTHO_AXES = 0.5                 # off-diagonal g-g correlation
ORTHO_R = 0.5                    # g-r correlation (the quarantine)
V_MEAN_PENALTY = 10.0            # pulls each axis's population mean to zero
V_RIDGE = 0.5                    # per-user shrinkage, scaled by 1/n_u


def four_pl_offset(rows_by_user, users, rank, params, lapse):
    """Frozen production 4PL offset per attempt row, at a per-user theta fitted
    once over that user's rows.

    Theta is fitted rather than free because what the offset must carry is the
    SHAPE of the user's difficulty profile — which problems are hard for them —
    not its level; the free alpha_u absorbs any additive error in the level, so
    leaving theta free would only duplicate it."""
    all_r0 = np.array([v[0] for v in params.values()])
    grid = np.arange(all_r0.min() - 3.0, all_r0.max() + 3.0, 0.05)
    off_by_user = {}
    for u in users:
        a = rows_by_user[u]
        tids = [t for t, _ in a]
        y = np.array([s for _, s in a], dtype=float)
        R0 = np.array([params[t][0] for t in tids])
        A = np.array([params[t][1] for t in tids])
        C = np.clip(np.array([params[t][2] for t in tids]), 0.0, FLOOR_MAX)
        span = np.maximum(1.0 - C - lapse, 1e-6)
        q = np.clip((CALIBRATION_P - C) / span, 1e-6, 1 - 1e-6)
        kap = np.log(q / (1 - q))
        th = _fit_theta(R0, A, C, span, kap, y, grid)
        P4 = np.clip(_p4pl(th, R0, A, C, span, kap), 1e-9, 1 - 1e-9)
        off_by_user[u] = np.log(P4 / (1 - P4))
    return off_by_user


class AxesNet:
    """Shared feature→(axes, residual difficulty) trunk plus a free per-user
    field, built lazily so torch is imported only when a fit runs.

    The axis output is normalised to unit RMS per column exactly as production
    does. That is not cosmetic: it makes the ortho penalty and the v-ridge
    scale-free, and it keeps the emitted loadings in the convention the eval
    harness and package_for_pwa already expect."""

    def __init__(self, fdim, n_users, k_axes, input_dim, device, seed,
                 user_main_effects=False, g_init=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.input_dim = int(input_dim)
        layers, d = [], fdim
        for h in AXES_HIDDEN:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROPOUT)]
            d = h
        self.trunk = nn.Sequential(*layers).to(device)
        self.head_g = nn.Linear(d, k_axes).to(device)
        self.head_r = nn.Linear(d, 1).to(device)
        # OPTIONAL EIGEN WARM START, as a LINEAR SKIP rather than an init of the
        # trunk. The axis block sits at a bilinear saddle when v = 0 — the
        # loading gradient is identically zero there — so residual_axes warm
        # starts its linear map at eigen_directions, the closed-form escape
        # directions of that same linearization. Measured against a cold start in
        # that model: 0.973 vs 0.830 mnat, seed sd 0.027 vs 0.126, with one draw
        # in three collapsing to ~0.65. A trunk cannot take a direction init the
        # same way, so the skip path carries it: at initialisation g IS the eigen
        # directions plus a small random head, and the trunk contributes
        # nonlinear correction as it trains. If the trunk adds nothing the model
        # degrades toward the incumbent rather than toward noise.
        #
        # The skip reads the DROPPED features, not the raw ones — regularizing
        # the linear axis map is the whole reason this experiment exists, and a
        # skip outside the mask would reinstate exactly the unregularized
        # Linear(fdim, K) whose collinear-column cancellation prompted it.
        self.skip = None
        if g_init is not None:
            self.skip = nn.Linear(fdim, k_axes, bias=False).to(device)
            with torch.no_grad():
                self.skip.weight.copy_(torch.tensor(
                    g_init, dtype=torch.float32, device=device))
        # v IS THE ONLY FREE PER-USER PARAMETER, and that is the point rather
        # than an economy. train_rating produces every per-problem quantity from
        # features and carries exactly one free per-user parameter — the ability
        # — which is the quantity the model exists to estimate, not a correction
        # to something already modelled. Free per-problem RESIDUAL parameters
        # were tried there in several configurations and are measured-rejected.
        #
        # A free per-user intercept and rank slope are that same rejected
        # pattern transposed: fix-ups sitting on top of a response the frozen 4PL
        # has already predicted. The residue the axes fit against is COMPUTED —
        # it is what the shipped rating model leaves over at a per-user ability
        # fitted once — not whatever is left after free parameters have taken
        # what they wanted. r stays because it is not of that kind: the trunk
        # produces it from features, so it is feature-conditional exactly as
        # train_rating's per-problem heads are.
        #
        # user_main_effects=True restores residual_axes.train_axes' alpha/beta
        # for a like-for-like arm against the incumbent. It is not free: only v
        # is ridged below, so where a residual could be explained by a user's
        # rank slope OR by an axis, the rank slope takes it unpenalized — and
        # problem rank correlates with the features the axes read.
        self.user_main = bool(user_main_effects)
        self.alpha = torch.zeros(n_users, device=device,
                                 requires_grad=self.user_main)
        self.beta = torch.zeros(n_users, device=device,
                                requires_grad=self.user_main)
        self.v = torch.zeros(n_users, k_axes, device=device, requires_grad=True)
        # phase 1 holds this at unity, exactly as train_rating holds the slope
        # scale; phase 2 frees it. exp() keeps it positive without a clamp.
        self.log_r_scale = torch.zeros((), device=device, requires_grad=True)

    def net_params(self):
        return (list(self.trunk.parameters()) + list(self.head_g.parameters())
                + list(self.head_r.parameters())
                + (list(self.skip.parameters()) if self.skip is not None
                   else []))

    def field_params(self):
        """Only what actually trains — with user_main_effects off, alpha and beta
        stay at zero and contribute nothing, so handing them to the optimizer
        would be a silent no-op rather than an honest one."""
        return [p for p in (self.alpha, self.beta, self.v) if p.requires_grad]

    def train_mode(self, on):
        self.trunk.train(on)

    def _dropped(self, feat):
        """Train-time column dropout over the wide input region; eval passes
        through. One mask across activations, bot winrate and move evals
        together — a per-block mask has to be extended by hand for every block
        added, and a block landing outside every range is regularized by nothing
        with no symptom at the call site."""
        torch = self.torch
        if self.trunk.training and self.input_dim > 0:
            keep = (torch.rand(len(feat), self.input_dim, device=feat.device)
                    >= INPUT_DROPOUT).float() / (1.0 - INPUT_DROPOUT)
            feat = torch.cat([feat[:, :self.input_dim] * keep,
                              feat[:, self.input_dim:]], dim=1)
        return feat

    def heads(self, feat):
        """(g, r) for the rows of `feat`; g normalised to unit RMS per column
        over the rows it is given."""
        # ONE dropout draw shared by the trunk and the skip. Drawing separately
        # would give the two paths different views of the same problem in the
        # same step, which is not the regularizer either is supposed to be.
        x = self._dropped(feat)
        h = self.trunk(x)
        g = self.head_g(h)
        if self.skip is not None:
            g = g + self.skip(x)
        # RMS AFTER the sum: the two paths are one axis map, and normalising them
        # apart would fix their relative weight by construction instead of
        # letting the fit decide how much the trunk contributes over the warm
        # start.
        g = g / (g.pow(2).mean(dim=0, keepdim=True).sqrt() + EPS)
        return g, self.head_r(h).squeeze(-1)


def _ortho_penalty(torch, g, r, w_axes, w_r):
    """Two separately-weighted decorrelations over the batch: axis-against-axis
    and axis-against-r. See the constants block for why only the second is
    load-bearing.

    Note what neither of them does: orthogonality is preserved under any rotation
    of the axis subspace, so this does NOT resolve the rotational indeterminacy
    that makes an axis hard to name. It stops an axis from re-describing
    population difficulty; it does not choose a basis."""
    M = torch.cat([g, r.unsqueeze(1)], dim=1)
    Mc = M - M.mean(dim=0, keepdim=True)
    Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
    C = Mn.T @ Mn
    k = g.shape[1]
    gg = C[:k, :k]
    pen = 0.0
    if w_axes:
        pen = pen + w_axes * ((gg - torch.eye(k, device=C.device)) ** 2).sum()
    if w_r:
        pen = pen + w_r * 2.0 * (C[:k, k] ** 2).sum()
    return pen


def fit_phase1(m, tfeat, tu, tp, ty, toff, counts, rank_col, epochs, device,
               w_axes=ORTHO_AXES, w_r=ORTHO_R, lr=NET_LR,
               field_lr=FIELD_LR, flat_tol=0.0, check_every=CHECK_EVERY,
               patience=FLAT_PATIENCE, min_epochs=MIN_EPOCHS,
               score_fn=None, score_every=0):
    """One joint pass: BCE on the residual of the frozen 4PL, plus the axes'
    own three penalties. The r scale is held at unity.

    With flat_tol > 0 the epoch count becomes a CAP and the run stops when the
    fit stops improving, which is what a fixed budget cannot get right: 400
    epochs is past the plateau at 1e-2 (fold 0 diverged at ~250 and never
    recovered) and nowhere near it at 1e-3 (still falling 0.9% per 50 epochs at
    epoch 399). Any single number is therefore wrong for some rate on the sweep,
    and picking one per rate is guessing at the answer the sweep exists to find.

    The stop is on RELATIVE improvement per check, needing `patience` consecutive
    flat checks — an absolute threshold would mean different things at different
    BCE levels. Every check prints its relative delta, because the last time a
    convergence stop went into this codebase it fired immediately against a
    tolerance orders of magnitude too loose and the log gave no way to see why.

    BEST STATE IS RESTORED, not the last. Divergence is a live failure mode here
    rather than a hypothetical, and patience alone would stop a few checks INTO a
    divergence and score the damaged model."""
    # CONSTANT rate, as train_rating runs one — deliberately not a decay
    # schedule. Both are ways to stop the late degradation measured at LR 0.01
    # (mild with the free intercept and slope, 0.4082 -> 0.4174; severe without
    # them, 0.4163 -> 0.4570, never recovering), but a schedule confounds the two
    # explanations it would be testing: it settles the fit AND lowers the average
    # rate, so a run that improves under it says nothing about which did the
    # work. A lower constant rate isolates that, and keeps the analogy to the
    # rating trainer intact. LR is per-run for exactly this reason.
    torch = m.torch
    # SPLIT RATES, as residual_axes uses (network 1e-3, per-user field 1e-2) and
    # NOT train_rating's single 0.01. Collapsing them to one rate is what the
    # first arms did, and it trains the trunk at ten times the rate the incumbent
    # uses for its network — which matches the late divergence those runs showed.
    # A single rate cannot be fixed by lowering it either: the network and the
    # field want rates an order of magnitude apart, so one knob starves whichever
    # it is not sized for.
    opt = torch.optim.Adam(
        [{"params": m.net_params(), "lr": lr}]
        + ([{"params": m.field_params(), "lr": field_lr}]
           if m.field_params() else []))
    n = len(ty)

    def clean_bce():
        m.train_mode(False)
        with torch.no_grad():
            g, r = m.heads(tfeat[tp])
            logit = (toff + m.alpha[tu] + m.beta[tu] * rank_col[tp] + r
                     + (g * m.v[tu]).sum(-1))
            return float(torch.nn.functional.binary_cross_entropy_with_logits(
                logit, ty))

    def snapshot():
        return ({k: v.detach().clone() for k, v in
                 zip(range(len(m.net_params())), m.net_params())},
                m.alpha.detach().clone(), m.beta.detach().clone(),
                m.v.detach().clone())

    def restore(snap):
        nets, a, b, v = snap
        with torch.no_grad():
            for i, p in enumerate(m.net_params()):
                p.copy_(nets[i])
            m.alpha.copy_(a); m.beta.copy_(b); m.v.copy_(v)

    best, best_snap, flat, prev = float("inf"), None, 0, None
    for ep in range(epochs):
        m.train_mode(True)
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            u, p = tu[bi], tp[bi]
            g, r = m.heads(tfeat[p])
            logit = (toff[bi] + m.alpha[u] + m.beta[u] * rank_col[p] + r
                     + (g * m.v[u]).sum(-1))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, ty[bi], reduction="none").sum() / len(bi)
            loss = loss + V_RIDGE * ((m.v[u] ** 2).sum(dim=1)
                                     / counts[u]).sum() / len(bi)
            loss = loss + V_MEAN_PENALTY * (m.v.mean(dim=0) ** 2).sum()
            loss = loss + _ortho_penalty(torch, g, r, w_axes, w_r)
            opt.zero_grad(); loss.backward(); opt.step()
        due = (ep + 1) % (check_every if flat_tol > 0 else 50) == 0
        if due or ep == epochs - 1:
            bce = clean_bce()
            rel = float("inf") if prev is None else (prev - bce) / max(prev, EPS)
            if flat_tol > 0:
                if bce < best:
                    best, best_snap = bce, snapshot()
                flat = flat + 1 if rel < flat_tol else 0
                held = ""
                if score_fn and score_every and (ep + 1) % score_every == 0:
                    # HELD-OUT gain during training, not only at the end. The
                    # training curve cannot stand in for it: measured on this
                    # model, going from 600 to 6400 steps lowered training BCE
                    # throughout while pooled held-out gain FELL (0.341 ->
                    # 0.326). A fit can therefore plateau long after its
                    # held-out peak, and an end-of-run score alone cannot tell
                    # the two apart.
                    pg, ug = score_fn(m)
                    held = f"  || held-out pooled {pg:+.3f} per-user {ug:+.3f}"
                print(f"    epoch {ep + 1:5d}  train BCE {bce:.4f}  "
                      f"rel {rel:+.4%}  flat {flat}/{patience}{held}",
                      flush=True)
                prev = bce
                if ep + 1 >= min_epochs and flat >= patience:
                    print(f"    settled at epoch {ep + 1} "
                          f"(best BCE {best:.4f}); restoring best", flush=True)
                    break
            else:
                print(f"    epoch {ep:4d}  train BCE {bce:.4f}", flush=True)
    if flat_tol > 0 and best_snap is not None:
        restore(best_snap)


def fit_phase2(m, tfeat, tu, tp, ty, toff, counts, rank_col, device):
    """PHASE 2: trunk frozen in eval mode, the per-user field refit FREE, and the
    one global scalar phase 1 held at unity — the scale on r — trained alongside
    it. Pure BCE: no dropout, nothing stochastic, full batch, no RNG.

    Runs twice, scale pinned then free and warm-started from the pinned solution,
    so the reported gain is like-for-like. Returns the scale, the gain and the
    per-axis ratio of refit coefficient spread to phase-1 spread — that ratio is
    the dropout compression of the field, and it is what sets the emitted loading
    scale."""
    torch = m.torch
    m.train_mode(False)
    with torch.no_grad():
        g_all, r_all = m.heads(tfeat)          # clean pass, computed once
    v0_sd = m.v.detach().std(dim=0).cpu().numpy()
    g_rows, r_rows = g_all[tp], r_all[tp]

    def run(free_scale, init):
        # alpha/beta follow phase 1: if they were not free there, freeing them
        # here would refit the model under a wider user side than the one whose
        # compression is being measured.
        a = init[0].clone().requires_grad_(m.user_main)
        b = init[1].clone().requires_grad_(m.user_main)
        v = init[2].clone().requires_grad_(True)
        ls = torch.zeros((), device=device, requires_grad=True)
        params = ([a, b] if m.user_main else []) + [v] \
            + ([ls] if free_scale else [])
        opt = torch.optim.Adam(params, lr=P2_LR)
        for step in range(P2_STEPS):
            s = torch.exp(ls) if free_scale else 1.0
            logit = (toff + a[tu] + b[tu] * rank_col[tp] + s * r_rows
                     + (g_rows * v[tu]).sum(-1))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, ty, reduction="none").sum() / len(ty)
            loss = loss + V_RIDGE * ((v[tu] ** 2).sum(dim=1)
                                     / counts[tu]).sum() / len(ty)
            # THE ZERO-MEAN PENALTY MUST BE CARRIED OVER FROM PHASE 1, or the
            # r-scale this phase exists to measure is not identified: with each
            # axis's population mean free, mean(v)·g is itself a per-problem
            # term, so r can be scaled up and the mean shifted down for exactly
            # the same predictions. The symptom is a scale that drifts while the
            # loss does not move at all.
            loss = loss + V_MEAN_PENALTY * (v.mean(dim=0) ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if free_scale and (step % 500 == 0 or step == P2_STEPS - 1):
                print(f"    phase-2 step {step:4d}  BCE {float(loss):.5f}  "
                      f"r-scale {float(torch.exp(ls)):.3f}", flush=True)
        return float(loss), float(torch.exp(ls)), (a.detach(), b.detach(),
                                                   v.detach())

    zeros = (torch.zeros_like(m.alpha), torch.zeros_like(m.beta),
             torch.zeros_like(m.v))
    bce_unity, _s, base = run(False, zeros)
    bce_scaled, s, fin = run(True, base)
    v_sd = fin[2].std(dim=0).cpu().numpy()
    return {"r_scale": s, "gain_mnat": (bce_unity - bce_scaled) * 1000.0,
            "v_sd": v_sd, "compression": v_sd / np.maximum(v0_sd, 1e-9)}


def loadings(m, tfeat, v_sd):
    """Per-problem loadings in the production convention: unit-RMS columns times
    the per-user coefficient spread."""
    torch = m.torch
    m.train_mode(False)
    with torch.no_grad():
        g, _r = m.heads(tfeat)
    return g.cpu().numpy() * v_sd[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="axis count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--no-phase2", action="store_true",
                    help="skip the clean-pass refit; emit phase-1 spread")
    ap.add_argument("--flat-tol", type=float, default=0.0,
                    help="stop when relative train-BCE improvement per check "
                         "falls below this for --flat-patience checks; --epochs "
                         "becomes a CAP. 0 (default) = fixed budget. Best state "
                         "is restored, not last.")
    ap.add_argument("--flat-patience", type=int, default=FLAT_PATIENCE)
    ap.add_argument("--score-every", type=int, default=0,
                    help="also score held-out gain every N epochs during "
                         "training (costly: a full walk-forward per fold), so a "
                         "held-out peak that precedes the training plateau is "
                         "visible rather than missed")
    ap.add_argument("--check-every", type=int, default=CHECK_EVERY)
    ap.add_argument("--no-offset", action="store_true",
                    help="drop the frozen 4PL offset (the incumbent's shape: "
                         "learn the whole baseline). Implies "
                         "--user-main-effects, since otherwise nothing carries "
                         "per-user level at all")
    ap.add_argument("--warm-start", action="store_true",
                    help="add an eigen-initialised linear skip alongside the "
                         "trunk (residual_axes' warm start; the axis block sits "
                         "at a saddle when v=0 and a cold start is measured to "
                         "cost 0.973->0.830 with 4x the seed spread)")
    ap.add_argument("--lr", type=float, default=NET_LR,
                    help="phase-1 NETWORK learning rate, constant "
                         "(residual_axes uses 1e-3; train_rating's single 0.01 "
                         "diverges late here)")
    ap.add_argument("--field-lr", type=float, default=FIELD_LR,
                    help="phase-1 per-user field learning rate "
                         "(residual_axes uses 1e-2)")
    ap.add_argument("--user-main-effects", action="store_true",
                    help="restore residual_axes' free per-user intercept and "
                         "rank slope (the incumbent's user side; NOT "
                         "train_rating's, which has no residual fix-ups)")
    ap.add_argument("--ortho-axes", type=float, default=ORTHO_AXES,
                    help="axis-against-axis decorrelation weight (0 to ablate — "
                         "the objective already disfavours duplicate axes)")
    ap.add_argument("--ortho-r", type=float, default=ORTHO_R,
                    help="axis-against-r decorrelation weight (the quarantine; "
                         "0 lets an axis absorb population difficulty)")
    ap.add_argument("--out", default=None, help="npz to write the loadings to")
    args = ap.parse_args()
    if args.no_offset:
        args.user_main_effects = True

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    feats, rank, params, act_dim = build_axis_features()
    tids = sorted(feats)
    tid_index = {t: i for i, t in enumerate(tids)}
    F = np.stack([feats[t] for t in tids])
    F = ((F - F.mean(0)) / (F.std(0) + 1e-9)).astype(np.float32)
    lapse = load_client_lapse(0.0328)

    # build_axis_features lays the columns out as
    #   [ act | bot | move | phase | size_is_13 | rating-model (3) | rank ]
    # The wide region is the first three blocks, as in the rating trainer; the
    # tail is exempt because those columns are smooth and cannot fingerprint a
    # problem — and `rank` in particular must survive, since the model reads it
    # structurally as beta_u's regressor. Only the phase width is not implied by
    # what build_axis_features returns, so it is read from its own loader.
    from utils.rating.model_constants import SIZES
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    _ph, nph = load_joint_game_phase(SIZES, tids_by_size)
    exempt = nph + 1 + 3 + 1
    input_dim = F.shape[1] - exempt

    rows_by_user = defaultdict(list)
    for u, p, s in load_axis_attempts():
        if p in tid_index:
            rows_by_user[u].append((p, int(bool(s))))
    users = sorted(u for u, a in rows_by_user.items()
                   if len(a) >= MIN_USER_ATTEMPTS)
    fold_of = _stratified_folds(users, rows_by_user)
    print(f"loaded in {time.time() - t0:.0f}s: {len(tids)} problems, "
          f"{len(users)} users | feat dim {F.shape[1]} "
          f"(act {act_dim} + ... | input-dropout region {input_dim}, "
          f"{exempt} exempt) | K={args.k} seed={args.seed} device {device.type}",
          flush=True)

    tfeat = torch.tensor(F, device=device)
    # THE STANDARDIZED COLUMN, exactly as residual_axes takes it
    # (RANK = X[:, -1]) — NOT the raw rank from build_axis_features. The raw
    # scale has mean -7.2, sd 11.1 and reaches -175, so beta_user's regressor
    # would be ~11x wider than the incumbent's with a large offset in it: beta
    # trains an order of magnitude more aggressively, and its intercept
    # entangles with alpha. Using the raw values invalidated the first
    # user-main-effects comparison, which scored the arm CARRYING beta against
    # one that disables it — handicapping the design under test with an
    # implementation defect.
    rank_col = tfeat[:, -1].clone()
    tot_g, tot_n, pu, g_by_fold = 0.0, 0, [], {}
    for f in (0, 1):
        tu_list = [u for u in users if fold_of[u] != f]
        ev_list = [u for u in users if fold_of[u] == f]
        uindex = {u: i for i, u in enumerate(tu_list)}
        off_by_user = four_pl_offset(rows_by_user, tu_list, rank, params, lapse)
        ui, pi, yy, oo = [], [], [], []
        for u in tu_list:
            off = off_by_user[u]
            for j, (t, s) in enumerate(rows_by_user[u]):
                ui.append(uindex[u]); pi.append(tid_index[t])
                yy.append(float(s)); oo.append(off[j])
        tu = torch.tensor(ui, dtype=torch.long, device=device)
        tp = torch.tensor(pi, dtype=torch.long, device=device)
        ty = torch.tensor(yy, dtype=torch.float32, device=device)
        toff = torch.tensor(np.asarray(oo, dtype=np.float32), device=device)
        if args.no_offset:
            # ABLATION: drop the frozen 4PL offset entirely, as the incumbent
            # does — residual_axes learns its whole baseline (alpha + beta*rank
            # + r) from scratch and never reads the shipped curve as an input.
            # This is the ONE structural difference between the two trainers that
            # nothing else in this script isolates, and with no free per-user
            # intercept there is nothing to absorb a miscalibrated offset, so v
            # would have to — corrupting exactly what is being measured. Forces
            # the user main effects back on, because without an offset AND
            # without them the model has no per-user level term at all.
            toff = torch.zeros_like(toff)
        counts = torch.bincount(tu, minlength=len(tu_list)).clamp(min=1).float()

        print(f"\n=== fold {f}: {len(tu_list)} train users / {len(ui)} rows "
              f"| {args.epochs} epochs | dropout {DROPOUT} / input "
              f"{INPUT_DROPOUT} ===", flush=True)
        g_init = None
        if args.warm_start:
            g_init = eigen_directions(rows_by_user, tu_list, F, tid_index, rank,
                                      params, lapse, args.k)
            print(f"  eigen warm start: {g_init.shape[0]} directions",
                  flush=True)
        def _score(mm):
            gg = loadings(mm, tfeat, mm.v.detach().std(dim=0).cpu().numpy())
            gain, nn_, gu, _nu = axes_ll_gain(rows_by_user, ev_list, gg, rank,
                                              params, tid_index, lapse)
            return gain / max(nn_, 1) * 1000.0, gu

        m = AxesNet(F.shape[1], len(tu_list), args.k, input_dim, device,
                    args.seed, user_main_effects=args.user_main_effects,
                    g_init=g_init)
        fit_phase1(m, tfeat, tu, tp, ty, toff, counts, rank_col, args.epochs,
                   device, w_axes=args.ortho_axes, w_r=args.ortho_r,
                   lr=args.lr, field_lr=args.field_lr,
                   flat_tol=args.flat_tol, score_every=args.score_every,
                   score_fn=(lambda mm: _score(mm)) if args.score_every else None,
                   check_every=args.check_every,
                   patience=args.flat_patience)
        if args.no_phase2:
            v_sd = m.v.detach().std(dim=0).cpu().numpy()
        else:
            p2 = fit_phase2(m, tfeat, tu, tp, ty, toff, counts, rank_col,
                            device)
            v_sd = p2["v_sd"]
            print(f"  phase 2: r-scale {p2['r_scale']:.3f} "
                  f"(+{p2['gain_mnat']:.2f} mnat/attempt in-sample) | per-axis "
                  "field compression "
                  + " ".join(f"{c:.2f}" for c in p2["compression"]), flush=True)
        g = loadings(m, tfeat, v_sd)
        g_by_fold[f] = g
        gain, n, gu, nu = axes_ll_gain(rows_by_user, ev_list, g, rank, params,
                                       tid_index, lapse)
        tot_g += gain; tot_n += n; pu.append(gu)
        print(f"  fold {f}: pooled {gain / n * 1000:.3f} per-user {gu:.3f} "
              f"({nu} scored users, {n} rows)", flush=True)

    print(f"\nK={args.k} seed={args.seed}  POOLED {tot_g / tot_n * 1000:.3f}  "
          f"PER-USER {float(np.mean(pu)):.3f}", flush=True)
    # The default path carries EVERY knob that changes the fit, not just k and
    # seed. Keyed on those two alone, two arms of the same comparison overwrite
    # each other silently and the loadings needed to explain a surprising result
    # are gone by the time the result appears.
    tag = (f"k{args.k}_s{args.seed}_lr{args.lr:g}"
           f"_flr{args.field_lr:g}_oa{args.ortho_axes:g}_or{args.ortho_r:g}"
           f"{f'_ft{args.flat_tol:g}' if args.flat_tol else ''}"
           f"{'_noff' if args.no_offset else ''}"
           f"{'_ws' if args.warm_start else ''}"
           f"{'_ume' if args.user_main_effects else ''}"
           f"{'_nop2' if args.no_phase2 else ''}_e{args.epochs}")
    out = args.out or f"/tmp/axes_trunk_{tag}.npz"
    np.savez_compressed(out, **{f"g{f}": g_by_fold[f] for f in (0, 1)},
                        tids=np.array(tids))
    print(f"wrote {out}\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
