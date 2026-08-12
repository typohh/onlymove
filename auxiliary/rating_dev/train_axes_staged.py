#!/usr/bin/env python3
"""The axis trainer's EXPERIMENT SURFACE — every knob production does not have.

`utils.rating.residual_axes.train_axes` is production: no options, one regimen,
the values it ships. Testing an alternative there means editing production and
reverting it, which is how a probe hook ends up committed by accident and how a
"temporary" parameter becomes permanent. This module is the counterpart, and the
arrangement mirrors `pipeline.train_rating` versus
`auxiliary.rating_dev.train_rating_staged`.

`train_axes_staged` is a parameterized copy of the production loop. EVERY default
is imported from `residual_axes` rather than restated, so an unset knob
reproduces the shipped fit exactly and a production constant that moves is
picked up here automatically — a restated default would silently drift and every
A/B run afterwards would be against a control that is no longer the shipped one.

What is exposed, and what each was added to answer:

  axis_input_dropout  the rate and its presence. Shipped at 0.3 on a 9/9
                      per-user result; 0.0 recovers the unregularized head.
  axis_hidden         a trunk on the axis path, ADDED to the linear map so
                      capacity stays separate from initialization (a trunk
                      cannot be handed a warm-start direction; substituting for
                      the linear map would change two things at once).
  ortho_weight        the mutual axis-against-axis decorrelation.
  ortho_r_weight      the axis-against-r quarantine, separately weighted.
                      None = follow ortho_weight, which reproduces production's
                      single combined penalty exactly (verified numerically).
  trunk_hidden        the residual head's own trunk widths.
  act                 "relu" (production) or "swiglu". SwiGLU replaces
                      [Linear, ReLU] with silu(xW)*xV and so carries TWICE the
                      parameters at the same width — a same-width arm is
                      therefore also a capacity arm, and a param-matched one
                      needs the widths scaled by roughly 2/3. Run both.
  use_r_head          False removes the residual head entirely. Read on axis
                      CONTAMINATION as well as gain: r exists to keep difficulty
                      out of the axes, so it can be load-bearing while
                      contributing nothing measurable to held-out gain.
  trunk_input_dropout column dropout on the RESIDUAL trunk's input. Production
                      has none, which is the reverse of the rating trainer.
  lr_net / lr_field   the split rates. The network trains an order of magnitude
                      slower than the per-user field; collapsing them to one
                      rate trains the net ~10x too fast and diverges late.
  v_ridge /
  v_mean_penalty      the per-user shrinkage and the zero-mean pull.
  epochs / seed

Returns exactly what production returns — (loadings, W, rms, v_sd) — so anything
that scores `train_axes` output scores this identically.

Usage (as a library):
  from auxiliary.rating_dev.train_axes_staged import train_axes_staged
  g, W, rms, v_sd = train_axes_staged(..., axis_hidden=(32,))
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Defaults are IMPORTED, never restated — see the module docstring.
from utils.rating.residual_axes import (  # noqa: E402
    AXIS_INPUT_DROPOUT, BATCH, EPOCHS, EXEMPT_TAIL_COLS, ORTHO_WEIGHT, SEED,
)

TRUNK_HIDDEN = (64, 32)          # the residual head's trunk, as production builds it
TRUNK_DROPOUT = 0.3
NET_LR = 1e-3
FIELD_LR = 1e-2
V_RIDGE = 0.5
V_MEAN_PENALTY = 10.0
ACT = "relu"                     # production's activation; "swiglu" is the arm
# Production applies column dropout to the AXIS map's input and none to the
# residual trunk's, which is the reverse of pipeline.train_rating, where input
# column dropout is the headline anti-memorization finding. r is the one head
# here that can fingerprint a problem, and it is the one without that guard.
TRUNK_INPUT_DROPOUT = 0.0        # production; 0.3 matches the rating trainer


def train_axes_staged(rows_by_user, train_users, F, tid_index, k_axes, g_init,
                      device, seed=SEED, epochs=EPOCHS,
                      ortho_weight=ORTHO_WEIGHT, ortho_r_weight=None,
                      axis_input_dropout=AXIS_INPUT_DROPOUT, axis_hidden=None,
                      trunk_hidden=TRUNK_HIDDEN, trunk_dropout=TRUNK_DROPOUT,
                      lr_net=NET_LR, lr_field=FIELD_LR, v_ridge=V_RIDGE,
                      v_mean_penalty=V_MEAN_PENALTY, act=ACT,
                      trunk_input_dropout=TRUNK_INPUT_DROPOUT,
                      use_r_head=True):
    """Production's fit with every constant exposed. Defaults = the shipped fit.

    Returns (loadings (n_problems, k) scaled to unit-variance v, raw weight
    matrix W (k, fdim), rms (k,), v_sd (k,)) — production's contract exactly."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    uids = sorted(train_users)
    uindex = {u: i for i, u in enumerate(uids)}
    rows = [(uindex[u], tid_index[t], s)
            for u in uids for t, s in rows_by_user[u]]
    U = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)
    P = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
    Y = torch.tensor([float(r[2]) for r in rows], device=device)
    counts = torch.bincount(U, minlength=len(uids)).clamp(min=1).float()
    X = torch.tensor(F, device=device)
    RANK = X[:, -1].clone()

    n_u, n_feat = len(uids), F.shape[1]
    input_dim = n_feat - EXEMPT_TAIL_COLS

    class SwiGLU(nn.Module):
        """Swish-gated linear unit: silu(xW) * xV, replacing [Linear, ReLU].

        It carries TWICE the parameters of the Linear+ReLU it replaces, because
        the gate is its own projection — so a same-width comparison is also a
        capacity comparison, and a param-matched one needs the width scaled by
        about 2/3. Both are worth running; neither alone answers the question."""

        def __init__(self, d_in, d_out):
            super().__init__()
            self.w = nn.Linear(d_in, d_out)
            self.v = nn.Linear(d_in, d_out)

        def forward(self, x):
            return torch.nn.functional.silu(self.w(x)) * self.v(x)

    def block(d_in, d_out):
        return ([SwiGLU(d_in, d_out)] if act == "swiglu"
                else [nn.Linear(d_in, d_out), nn.ReLU()])

    layers, d = [], n_feat
    for hdim in trunk_hidden:
        layers += block(d, hdim) + [nn.Dropout(trunk_dropout)]
        d = hdim
    trunk = nn.Sequential(*layers).to(device)
    head_r = nn.Linear(d, 1).to(device)
    # use_r_head=False removes the population residual-difficulty term outright.
    # Its purpose is not gain — it is the quarantine that stops an axis absorbing
    # difficulty the rating model missed — so an ablation must be read on the
    # axes' contamination as well as on held-out gain. With no r there is also no
    # r column in the correlation matrix, so the decorrelation becomes
    # axis-against-axis only, which is correct rather than a confound.

    g_lin = nn.Linear(n_feat, k_axes, bias=False).to(device)
    if g_init is not None:
        with torch.no_grad():
            g_lin.weight[:g_init.shape[0]].copy_(
                torch.tensor(g_init, dtype=torch.float32, device=device))
    g_net = None
    if axis_hidden:
        gl, d2 = [], n_feat
        for hdim in axis_hidden:
            gl += block(d2, hdim) + [nn.Dropout(trunk_dropout)]
            d2 = hdim
        gl += [nn.Linear(d2, k_axes)]
        g_net = nn.Sequential(*gl).to(device)

    alpha = torch.zeros(n_u, device=device, requires_grad=True)
    beta = torch.zeros(n_u, device=device, requires_grad=True)
    v = torch.zeros(n_u, k_axes, device=device, requires_grad=True)

    opt = torch.optim.Adam([
        {"params": list(trunk.parameters()) + list(head_r.parameters()),
         "lr": lr_net, "weight_decay": 1e-4},
        {"params": (list(g_lin.parameters())
                    + (list(g_net.parameters()) if g_net is not None else [])),
         "lr": lr_net},
        {"params": [alpha, beta], "lr": lr_field},
        {"params": [v], "lr": lr_field},
    ])
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    w_r = ortho_weight if ortho_r_weight is None else ortho_r_weight
    n = len(rows)
    for _ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            u, p, y = U[idx], P[idx], Y[idx]
            xt = X[p]
            if trunk_input_dropout > 0.0:
                kt = (torch.rand(len(idx), input_dim, device=X.device)
                      >= trunk_input_dropout).float() / (1.0 - trunk_input_dropout)
                xt = torch.cat([xt[:, :input_dim] * kt, xt[:, input_dim:]], dim=1)
            h = trunk(xt)
            rh = head_r(h)
            xg = X[p]
            if axis_input_dropout > 0.0:
                keep = (torch.rand(len(idx), input_dim, device=X.device)
                        >= axis_input_dropout).float() / (1.0 - axis_input_dropout)
                xg = torch.cat([xg[:, :input_dim] * keep, xg[:, input_dim:]],
                               dim=1)
            gh = g_lin(xg) + (g_net(xg) if g_net is not None else 0)
            gh = gh / (gh.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6)
            logit = (alpha[u] + beta[u] * RANK[p]
                     + (rh.squeeze(-1) if use_r_head else 0.0)
                     + (gh * v[u]).sum(-1))
            loss = bce(logit, y) \
                + v_ridge * ((v[u] ** 2).sum(dim=1) / counts[u]).sum() / len(idx) \
                + v_mean_penalty * (v.mean(dim=0) ** 2).sum()
            M = torch.cat([gh, rh], dim=1) if use_r_head else gh
            Mc = M - M.mean(dim=0, keepdim=True)
            Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
            Cm = Mn.T @ Mn
            if not use_r_head:
                loss = loss + ortho_weight * ((Cm - torch.eye(
                    Cm.shape[0], device=Cm.device)) ** 2).sum()
            elif ortho_r_weight is None:
                # The DEFAULT path takes production's single combined form
                # verbatim, not an algebraically-equal split. The split sums in a
                # different order, and float32 rounding then makes the control
                # differ from the shipped fit at ~1e-8 and compounding — harmless
                # against gains of order 1e-3, but a control that is merely
                # nearly the shipped fit is a control nobody can fully trust.
                loss = loss + ortho_weight * ((Cm - torch.eye(
                    Cm.shape[0], device=Cm.device)) ** 2).sum()
            else:
                # Weighted separately, for asking whether the two halves differ.
                # At equal weights this equals the branch above up to the
                # Cm[k,k] diagonal term, which is ~0 by construction.
                if ortho_weight:
                    gg = Cm[:k_axes, :k_axes]
                    loss = loss + ortho_weight * ((gg - torch.eye(
                        k_axes, device=Cm.device)) ** 2).sum()
                if w_r:
                    loss = loss + w_r * 2.0 * (Cm[:k_axes, k_axes] ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

    if g_net is not None:
        g_net.eval()
    with torch.no_grad():
        gh = g_lin(X) + (g_net(X) if g_net is not None else 0)
        rms = gh.pow(2).mean(dim=0).sqrt() + 1e-6
        g_all = (gh / rms).cpu().numpy()
        v_sd = v.std(dim=0).cpu().numpy()
        g_all = g_all * v_sd[None, :]
        W = g_lin.weight.detach().cpu().numpy()
    return g_all, W, rms.cpu().numpy(), v_sd
