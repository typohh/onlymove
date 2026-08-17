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

Run `pipeline.extract_activations` then `pipeline.cache_rating_features` first;
the move-eval cache is a hard prerequisite. So is the motif store,
`axes/{label_anchors.npz,labels.json}` — see below.

THE AXES ARE TAUGHT FROM A FROZEN BANK, which is what lets a name outlive a
refit. The objective is exactly invariant to rotating the axis space — the
likelihood reads the loadings and the user positions only through their product,
and V_RIDGE, V_MEAN_PENALTY and ORTHO_WEIGHT are all rotation-invariant too — so
the fit has a continuum of equal solutions and no preference among them. Left to
itself it lands wherever initialization drops it: a 2% data change re-rolls the
basis as hard as a different seed does, which would silently reattach every name
in `axes/labels.json` to a direction it was not written for. So the axis map is
not re-derived. It is pulled toward the motif store by an auxiliary MSE,
which decays linearly to zero over TEACH_DECAY_EPOCHS while the trunk trains
normally. Measured across a real refit: without it the new axes share 0.20 min
canonical correlation with the old ones, with it 0.899, and the loss is if
anything slightly better.

THE BANK IS THE MOTIF STORE, SELECTED BY NAME. `axes/label_anchors.npz` holds
every direction ever kept, keyed by an `ax_<hex>` content hash, and
`axes/labels.json`'s `components` lists which ids this fit ships, in axis order;
the first K_AXES are taught, so that list is the shipping decision. Binding by id
rather than by column position is the point — a bank file orders its directions
implicitly while labels.json is indexed against that order, so reordering it
renames every motif onto a direction it was not written for. Directions enter the
store from a `--dump-params` npz via `auxiliary.rating_dev.update_motif_store`.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.model_constants import (  # noqa: E402
    DIFF_HIDDEN, DROPOUT, EPS, FLOOR_CAP, INPUT_DROPOUT, LAPSE_CAP, LAPSE_INIT,
    K_AXES, PCA_DIM, SEED, SIZES, SLOPE_FLOOR, clear_logit_vec)
from utils.rating.features import (  # noqa: E402
    load_first_time_attempts, load_joint_mirt_features)

EPOCHS = 10000                   # runaway failsafe; the patience rule ends the run
CHECK_EVERY = 10
# Checks without a new best before stopping. SIZED FROM THE GAPS BETWEEN
# IMPROVEMENTS, not chosen: the stop split keeps improving in bursts separated
# by long flat stretches, and the largest gap before the final optimum measured
# 25 checks on one seed and 10 on the other. Anything under that stops in a
# stretch rather than at the end — at 5 both seeds halted around epoch 350 while
# their true optima sat at 850 and 580, costing 0.5 and 0.7 mnat of RATING (a
# paired comparison, same seed either side, so it is not the across-seed noise).
# The axes are indifferent to this: 490 extra epochs move them 0.86 canonical,
# far less than a seed change does, so raising it buys rating accuracy and NOT
# axis stability. The cost is runtime, ~9 minutes to ~25.
FLAT_PATIENCE = 30
BATCH = 65536
# SPLIT RATES, and the split is mandatory. What the grid measures is the RATIO
# lr_field/lr_net, monotone and saturating near 30x: below 10x the fit loses
# 2.5-5 mnat. Given a ratio past 30x a slower net buys the last 0.8; 1e-4 and
# 3e-4 tie, and 3e-4 converges in half the epochs.
LR_NET = 3e-4
LR_FIELD = 1e-2
V_RIDGE = 0.5                    # not binding: 0.05/0.15/0.5 score alike
V_MEAN_PENALTY = 10.0            # stops an axis carrying POPULATION difficulty
ORTHO_WEIGHT = 0.5
ORTHO_RATING = 0.5               # keeps the axes off DIFFICULTY (see the loop)
GAUGE_MOM = 0.01
# ONE barrier, on the slope, one-sided, acting on the PRE-SOFTPLUS output. The
# rating is b + kappa(c)/a, so a slope collapsing toward its floor sends it to
# infinity. Without this, slopes fall to a 0.94 median and ratings blow out to a
# range of [-225, +146] — a 22 mnat loss. There is no floor barrier: the floor is
# bounded structurally by cap = (1 - 2*lambda).
SHALLOW_TAU = 0.1
SHALLOW_W = 0.3
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
# The decay length is a pick on a FRONTIER, not an optimum: identity rises with
# it and saturates around here while the loss cost rises monotonically, and the
# two never cross. Hold long, then let go — never releasing costs another 1.1
# mnat for no further identity, and releasing ABRUPTLY does not stick at all.
#
# IT MUST ALSO END BEFORE THE FIT DOES, which is what sets it here rather than
# the frontier alone. FLAT_PATIENCE counts staleness against the objective, and
# while the decay is still running that objective is still changing — so a
# schedule outlasting convergence stops the run with the bank still holding the
# map, and the axes are then scored before they were ever released. A length
# chosen on a corpus that converged late does not transfer to one that converges
# early; this value leaves room on the corpus as it stands.
TEACH_WEIGHT = 1.0
TEACH_DECAY_EPOCHS = 1000
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
MOTIF_STORE = os.path.join(AXES_DIR, "label_anchors.npz")
LABELS = os.path.join(AXES_DIR, "labels.json")


class JointMIRT:
    """Compensatory M4PL with per-problem parameters from a shared trunk.

    The one unavoidable departure from textbook MIRT is that a_p, b_p, c_p and
    g_k come from a FEATURE NET rather than being free per item: a third of the
    corpus has no attempts at all, so free per-item parameters cannot price it.
    """

    def __init__(self, fdim, n_users, input_dim, device, seed):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.input_dim = int(input_dim)

        # theta before the trunk so the RNG draw order is fixed.
        self.theta = nn.Embedding(n_users, 1).to(device)
        nn.init.normal_(self.theta.weight, std=1.0)

        class SwiGLU(nn.Module):
            """silu(xW) * xV, replacing [Linear, ReLU]."""

            def __init__(self, d_in, d_out):
                super().__init__()
                self.w = nn.Linear(d_in, d_out)
                self.v = nn.Linear(d_in, d_out)

            def forward(self, x):
                return torch.nn.functional.silu(self.w(x)) * self.v(x)

        layers, d = [], fdim
        for h in DIFF_HIDDEN:            # 128x64, an interior optimum both ways
            layers += [SwiGLU(d, h), nn.Dropout(DROPOUT)]
            d = h
        self.trunk = nn.Sequential(*layers).to(device)
        self.head = nn.Linear(d, 3).to(device)          # difficulty, slope, floor
        # The axis map, on the TRUNK's representation. Same linear map, different
        # input: it adds no capacity (fewer parameters than reading the 212
        # features) and the axis objective reaches the trunk, so the trunk learns
        # what the AXES need and not only what the rating needs. Detaching would
        # make this feature reuse rather than a joint fit.
        self.g_lin = nn.Linear(d, K_AXES, bias=False).to(device)
        self.v = torch.zeros(n_users, K_AXES, device=device, requires_grad=True)
        li = float(np.clip(LAPSE_INIT / LAPSE_CAP, EPS, 1.0 - EPS))
        self.lapse_raw = torch.tensor(float(np.log(li / (1.0 - li))),
                                      device=device, requires_grad=True)
        # Rating-gauge EMA, on DIFFICULTY — see problem_params.
        self.b_mu = torch.zeros((), device=device)
        self.b_sd = torch.ones((), device=device)
        self.training = True

    def params(self):
        return (list(self.trunk.parameters()) + list(self.head.parameters())
                + list(self.g_lin.parameters()) + list(self.theta.parameters())
                + [self.v, self.lapse_raw])

    def param_groups(self):
        """SPLIT RATES. The network must train an order of magnitude slower than
        the free per-user fields; under one shared rate this fit diverges late,
        in eval mode, in every arrangement tried. The rating trainer gets away
        with a single rate because its only per-user parameter is one ability
        embedding; this model has that plus a K-dimensional axis position."""
        net = (list(self.trunk.parameters()) + list(self.head.parameters())
               + list(self.g_lin.parameters()))
        field = list(self.theta.parameters()) + [self.v, self.lapse_raw]
        return [{"params": net, "lr": LR_NET},
                {"params": field, "lr": LR_FIELD}]

    def train(self, on):
        self.training = on
        self.trunk.train(on)

    def lapse(self):
        return LAPSE_CAP * self.torch.sigmoid(self.lapse_raw)

    def _drop(self, feat):
        """COLUMN dropout over the wide input region as ONE mask — activations,
        bot winrate and move evals together. The two game-phase columns and the
        size indicator sit past that region and are deliberately exempt, so a new
        wide block belongs INSIDE input_dim and is then masked automatically."""
        torch = self.torch
        if not self.training or INPUT_DROPOUT <= 0.0 or self.input_dim <= 0:
            return feat
        keep = (torch.rand(len(feat), self.input_dim, device=feat.device)
                >= INPUT_DROPOUT).float() / (1.0 - INPUT_DROPOUT)
        return torch.cat([feat[:, :self.input_dim] * keep,
                          feat[:, self.input_dim:]], dim=1)

    def slope_logit(self, feat):
        """Pre-softplus slope output — the space the barrier acts in, where the
        penalty falls away properly for healthy slopes."""
        return self.head(self.trunk(feat))[:, 1]

    def problem_params(self, feat):
        """(difficulty b, discrimination a, floor c, axis loadings g).

        THE GAUGE MUST SIT ON DIFFICULTY, NOT ON AN INTERCEPT. Two freedoms have
        to be removed, a shift of ability and a rescaling of it, and whitening an
        ADDITIVE intercept removes only the first: ability*s with a/s leaves an
        additive intercept untouched, so its whitening is satisfied and nothing
        objects, while difficulty is MULTIPLIED by the slope and must rescale
        with it. The shipped rating is an ability, so under the unblocked
        rescaling every rating scales and the whole axis floats."""
        torch, nn = self.torch, self.nn
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
        cap = (1.0 - 2.0 * lam).clamp(min=EPS, max=FLOOR_CAP)
        c = cap * torch.sigmoid(o[:, 2])
        g = self.g_lin(h)
        # Per-axis RMS over the batch. When called on the whole corpus this is a
        # corpus constant, and the per-cell fanout MUST reuse it: a cell scaled
        # by its own batch's RMS would not be comparable with the correct-move
        # vector in per_problem.csv, and nothing downstream could tell.
        self.g_rms = (g.pow(2).mean(dim=0, keepdim=True).sqrt() + 1e-6).detach()
        return b, a, c, g / self.g_rms

    def forward(self, uid, feat):
        torch = self.torch
        b, a, c, g = self.problem_params(feat)
        th = self.theta(uid).squeeze(-1)
        lam = self.lapse()
        z = (a * (th - b) + (g * self.v[uid]).sum(-1)).clamp(-35, 35)
        # b is returned DETACHED for the difficulty-orthogonality penalty.
        # Recomputing problem_params to get it would run the trunk twice and,
        # worse, tick the difficulty gauge's EMA twice per batch — the gauge is
        # defined over the attempt stream and an extra update would quietly move
        # the rating's units.
        return c + (1.0 - c - lam) * torch.sigmoid(z), g, b.detach()


def load_bank(key_idx):
    """The frozen directions this fit's axes are taught onto, one row per problem
    of `key_idx` in that order, plus a mask of which rows the bank supplies.

    THE BANK IS THE MOTIF STORE, SELECTED BY NAME. `axes/label_anchors.npz` holds
    every direction ever kept, keyed by an `ax_<hex>` content hash, and
    `axes/labels.json`'s `components` list names which of them this fit ships, in
    axis order. Binding by id rather than by column position is what makes the
    labels safe: a bank file carrying the directions in an implicit order that
    labels.json is indexed against can be reordered, and every motif silently
    renames onto a direction it was not written for.

    Each column is z-scored over the problems the store and this corpus share,
    because the stored per-axis scale is arbitrary — rescaling a loading is
    absorbed by v — so direction is the only part of it that means anything.

    A problem the bank does not carry is left unconstrained rather than taught to
    zero: absent is not a value. Legacy directions predate the current corpus and
    are NaN outside their own coverage, so a row is taught only where EVERY
    selected column is finite — a partially-known target would teach one axis
    from data and its neighbour from nothing, on the same problem.

    Returns (tgt, ok, ids) — the ids so the caller can stamp exactly what was
    taught rather than the whole store, which changes whenever an unrelated
    motif is added.
    """
    for path in (MOTIF_STORE, LABELS):
        if not os.path.exists(path):
            raise SystemExit(
                f"NO MOTIF STORE at {path}. It holds the frozen directions the "
                f"axes are taught onto, without which this fit would re-roll the "
                f"basis and every name in axes/labels.json would land on a "
                f"direction it was not written for. Add directions to it with "
                f"`auxiliary.rating_dev.update_motif_store`, and list the ones "
                f"to ship under `components` in axes/labels.json.")
    meta = json.load(open(LABELS))
    want = [str(x) for x in meta.get("components", [])]
    if len(want) < K_AXES:
        raise SystemExit(
            f"axes/labels.json lists {len(want)} components, this fit ships "
            f"K_AXES={K_AXES}. `components` is the shipping decision: an ordered "
            f"list of motif ids from the store's `bank`, longest-lived first. "
            f"Pick them by inspecting the directions with "
            f"`auxiliary.rating_dev.inspect_axes`.")
    want = want[:K_AXES]

    z = np.load(MOTIF_STORE, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    missing = [a for a in want if a not in ids]
    if missing:
        raise SystemExit(
            f"axes/labels.json names components not in the store: {missing}. "
            f"The store has {len(ids)} directions; a name with no direction "
            f"behind it cannot be taught.")
    M = np.asarray(z["M"], np.float64)[[ids.index(a) for a in want]]

    row = {f"{sz}:{t}": i for i, (sz, t) in
           enumerate(zip(z["sizes"], z["tids"]))}
    tgt = np.zeros((len(key_idx), K_AXES), np.float64)
    ok = np.zeros(len(key_idx), bool)
    for (sz, t), j in key_idx.items():
        i = row.get(f"{sz}:{t}")
        if i is not None and np.isfinite(M[:, i]).all():
            tgt[j] = M[:, i]
            ok[j] = True
    if not ok.any():
        raise SystemExit(
            f"MOTIF STORE SHARES NO PROBLEMS with this corpus ({len(row)} stored "
            f"against {len(key_idx)} here). Its keys are size:tsumego_id — a "
            f"store from a different corpus generation teaches nothing.")
    tgt[ok] = (tgt[ok] - tgt[ok].mean(0)) / (tgt[ok].std(0) + 1e-12)
    names = [meta["bank"].get(a, {}).get("positive") or "(unnamed)" for a in want]
    print(f"  BANK from the motif store: {', '.join(names)} "
          f"({ok.sum()}/{len(key_idx)} problems taught)", flush=True)
    return tgt, ok, want


def report_bank_identity(g_np, tgt, ok, prob_idx):
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
    for k in range(K_AXES):
        x, y = g_np[sel, k], tgt[sel, k]
        pos = hinged_r(x, y)
        neg = hinged_r(-x, -y)
        bulk = float(np.corrcoef(x, y)[0, 1])
        print(f"    axis {k}: +tail {pos:6.3f}  -tail {neg:6.3f}"
              f"   (bulk r {bulk:+.3f})", flush=True)


def main():
    import pandas as pd
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    feat, keys, key_idx, input_dim = load_joint_mirt_features(SIZES, PCA_DIM, SEED)
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
    rng = np.random.default_rng(SEED)
    is_stop = rng.random(len(df)) < STOP_ROW_FRAC
    tr = torch.tensor(np.where(~is_stop)[0], device=device)
    i_stop = torch.tensor(np.where(is_stop)[0], device=device)
    print(f"  stop signal: {int(is_stop.sum())} rows held aside", flush=True)

    bank_tgt, bank_ok, bank_ids = load_bank(key_idx)
    t_tgt = torch.tensor(bank_tgt, dtype=torch.float32, device=device)
    t_ok = torch.tensor(bank_ok, device=device)
    print(f"  axis bank: {int(bank_ok.sum())}/{len(keys)} problems carry banked "
          f"directions, taught with weight {TEACH_WEIGHT} decaying to zero over "
          f"{TEACH_DECAY_EPOCHS} epochs", flush=True)

    m = JointMIRT(fdim, n_users, input_dim, device, SEED)
    opt = torch.optim.Adam(m.param_groups())
    o_tau = float(np.log(np.expm1(SHALLOW_TAU)))
    n = len(tr)
    best, best_state, flat, best_epoch = float("inf"), None, 0, 0
    stopped = False
    print(f"\n=== joint fit | K={K_AXES} | teach decays over "
          f"{TEACH_DECAY_EPOCHS} epochs ===", flush=True)

    for ep in range(EPOCHS):
        m.train(True)
        tw = TEACH_WEIGHT * max(0.0, 1.0 - ep / TEACH_DECAY_EPOCHS)
        perm = tr[torch.randperm(n, device=device)]
        for i in range(0, n, BATCH):
            bi = perm[i:i + BATCH]
            pred, gh, bh = m.forward(tu[bi], tfeat[tp[bi]])
            pred = pred.clamp(EPS, 1 - EPS)
            loss = torch.nn.functional.binary_cross_entropy(
                pred, ty[bi], reduction="none").sum() / len(bi)
            loss = loss + V_RIDGE * (m.v[tu[bi]] ** 2).sum() / len(bi)
            loss = loss + V_MEAN_PENALTY * (m.v.mean(dim=0) ** 2).sum()
            Mc = gh - gh.mean(dim=0, keepdim=True)
            Mn = Mc / (Mc.norm(dim=0, keepdim=True) + 1e-9)
            Cm = Mn.T @ Mn
            loss = loss + ORTHO_WEIGHT * (
                (Cm - torch.eye(K_AXES, device=device)) ** 2).sum()
            # DECORRELATE THE AXES FROM DIFFICULTY. Nothing else does:
            # ORTHO_WEIGHT is K x K over the loadings alone, and V_MEAN_PENALTY
            # pins only the POPULATION mean of v, which blocks an axis acting as
            # a global difficulty shift but not one tracking difficulty per
            # problem. Left free, a fit spends an axis on a rating-side property
            # and that axis is unnameable when it does.
            #
            # b is DETACHED, so this can only push the axes off difficulty and
            # never degrade the rating to satisfy itself. The direction matters:
            # the client's rank path reads rating, slope and floor and never an
            # axis, so difficulty living in an axis is signal the product cannot
            # reach.
            #
            # DIFFICULTY ALONE, not the shipped triple. Penalising rating, slope
            # and floor together does decorrelate — and leaves the axes worth
            # 0.16 mnat on held-out users against 4.41 here, because three
            # constraints on K=3 axes leave room to satisfy the penalty or to
            # carry signal, not both. The cost of stopping at b is that the leak
            # displaces rather than vanishes, most of it into the floor head.
            bz = bh - bh.mean()
            bz = bz / (bz.norm() + 1e-9)
            loss = loss + ORTHO_RATING * (Mn.T @ bz).pow(2).sum()
            # TEACHING, ON THE ATTEMPT STREAM, reusing the gh this batch already
            # produced — no second forward and no second sampling distribution.
            #
            # Both properties are load-bearing. Sampling problems separately
            # makes the trunk adapt to a MIXTURE, and when the teaching weight
            # decays that half of its input distribution disappears; the epochs
            # of attempt-only training that follow overwrite whatever served it,
            # taking the taught directions along. Riding gh means no second
            # distribution exists, so the decay moves a scalar and nothing else.
            # It also matches the population the axes are DEFINED on: loadings
            # are identified by between-user residuals, which exist only where
            # somebody attempted the problem. The never-attempted third of the
            # corpus is left alone deliberately — its banked values are the
            # previous fit's extrapolation into a region that never carried
            # signal, so teaching them would spend trunk capacity, in competition
            # with the rating, reproducing an arbitrary guess.
            if tw > 0.0:
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
                    loss = loss + tw * (
                        (gz[mk] - t_tgt[tp[bi]][mk]) ** 2).mean()
            m.train(False)          # the barrier reads the corpus, dropout off
            short = torch.nn.functional.softplus(
                o_tau - m.slope_logit(tfeat)).pow(2).mean()
            loss = loss + SHALLOW_W * short
            m.train(True)
            opt.zero_grad(); loss.backward(); opt.step()

        if (ep + 1) % CHECK_EVERY == 0:
            m.train(False)
            with torch.no_grad():
                pr, _, _ = m.forward(tu, tfeat[tp])
                pr = pr.clamp(EPS, 1 - EPS)
                bce_tr = float(torch.nn.functional.binary_cross_entropy(
                    pr[tr], ty[tr]))
                bce_st = float(torch.nn.functional.binary_cross_entropy(
                    pr[i_stop], ty[i_stop]))
            improved = bce_st < best - 1e-6
            if improved:
                best, best_epoch = bce_st, ep + 1
                best_state = [q.detach().clone() for q in m.params()]
            flat = 0 if improved else flat + 1
            print(f"  epoch {ep + 1:4d}  train {bce_tr:.4f}  "
                  f"stop {bce_st:.4f}  best {best:.4f}@{best_epoch}  "
                  f"teach {tw:.2f}  stale {flat}/{FLAT_PATIENCE}", flush=True)
            # NOTE the stopping rule runs against an objective the decay is still
            # changing, so a fit that plateaus before the decay ends stops while
            # the bank is still partly holding the map. That is the observed case
            # rather than a hypothetical.
            if flat >= FLAT_PATIENCE:
                print(f"  stopped at epoch {ep + 1}; restoring best "
                      f"{best:.4f}@{best_epoch}", flush=True)
                stopped = True
                break

    if not stopped:
        raise SystemExit(
            f"EPOCH CAP HIT: ran {EPOCHS} epochs without the patience rule "
            f"firing. The fit never plateaued, so this is not a converged "
            f"solution and its numbers are not comparable with anything.")
    with torch.no_grad():
        for q, b_ in zip(m.params(), best_state):
            q.copy_(b_)

    # --- outputs
    m.train(False)
    with torch.no_grad():
        b_all, a_all, c_all, g_all = m.problem_params(tfeat)
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
    report_bank_identity(g_np, bank_tgt, bank_ok, prob_idx)

    os.makedirs(OUT_DIR, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(OUT_DIR, str(s))
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
                 ).to_csv(os.path.join(OUT_DIR, "user_theta.csv"), index=False)
    with open(os.path.join(OUT_DIR, "rating_constants.json"), "w") as f:
        json.dump({"fat_finger_rate": round(lam, 4), "slope_scale": 1.0,
                   "scale_fit": "none_shipped_as_fitted"}, f, indent=2)

    os.makedirs(AXES_DIR, exist_ok=True)
    rows = {"tsumego_id": [k[1] for k in keys], "size": [k[0] for k in keys]}
    for k in range(K_AXES):
        rows[f"embed_{k}"] = g_np[:, k].round(5)
    pd.DataFrame(rows).to_csv(os.path.join(AXES_DIR, "per_problem.csv"),
                              index=False)
    # The content hash the PWA's embedding-invalidation gate rides: it must move
    # when and only when the projection would give different vectors.
    h = hashlib.sha256()
    for arr in (m.g_lin.weight.detach().cpu().numpy(),
                m.head.weight.detach().cpu().numpy()):
        h.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
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
    bh = hashlib.sha256("\n".join(bank_ids).encode()).hexdigest()[:16]
    np.savez_compressed(os.path.join(AXES_DIR, "axes.npz"),
                        W=m.g_lin.weight.detach().cpu().numpy(),
                        k=np.array(K_AXES),
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
               os.path.join(OUT_DIR, "mirt_model.pt"))
    print(f"  wrote axes/per_problem.csv + axes.npz (K={K_AXES}, "
          f"embedding_version {ev}, bank {bh})", flush=True)

    # RETIRE ANY per_cell/ FROM THE PREVIOUS AXIS MAP. package_for_pwa reads the
    # correct-move vectors from per_problem.csv and the picked-move vectors from
    # per_cell/, and has no way to tell they came from different fits — it would
    # ship a shard whose two embedding sources disagree, which no client check
    # can catch because both files are individually well-formed. Moving it aside
    # makes packaging fail loudly instead.
    pc = os.path.join(AXES_DIR, "per_cell")
    if os.path.isdir(pc):
        stale = os.path.join(AXES_DIR, f"per_cell.stale-{int(time.time())}")
        os.rename(pc, stale)
        print(f"  RETIRED the previous per_cell/ -> {os.path.basename(stale)}. "
              f"It was projected through the axis map this fit replaces, so "
              f"shipping it beside these correct-move vectors would mix two "
              f"fits. Packaging needs a per-cell pass against THIS model before "
              f"it can run; delete the stale directory once that exists.",
              flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
