"""Combined cross-board MLP-2PL difficulty model — the rating pipeline.

Writes under ratings_v2/, which package_for_pwa ships (output format matches what
the PWA expects, see below — no frontend changes). This is the only rating backend.

ONE model for 9x9 AND 13x13:

    P(correct) = c_p + (1 - c_p) * sigmoid( a_p * theta_u + d_p )
    c_p = FIXED per-problem guessing floor from the bot ladder (load_bot_floor; --bot-floor)
    d_p = diff_head(feat_p) + delta_p              # feature PRIOR + per-problem residual
    a_p = softplus(b0 + slope_head(feat_p) + eta_p)  # feature-prior slope + residual

The floor c_p is the lowest clear rate across the standard-temperature humanv0 bot ladder —
the bots attempt far-above-their-strength problems, sampling the floor regime that human
first-time attempts (served ~80%-clear) never reach. Supplying it FIXED during training stops
a_p from deflating to mimic a floor (the deflation that fattened the far-from-rank tail and
corrupted the client rank estimator), so a_p and the rating are fit on top of the real tail.

  - feat_p = joint-PCA(correct-move activations, both boards) ++ common-column bot winrate
    (bw_act). The bot data enters ONLY through these features, so the feature heads are a
    bot+activation PRIOR; the HUMAN first-time attempts (is_repeat=0) are the training
    target and add per-problem residuals delta_p / eta_p ON TOP of the prior. Ridge on the
    residuals = empirical-Bayes shrinkage: sparsely / never-attempted problems stay at the
    feature prior (delta=eta=0), human-attempted problems pull away from it.
  - theta_u is a single shared per-(user, 2-month) ability across boards (one skill).

The 2PL SLOPE a_p replaces the old `quality`: a high-discrimination (sharp) problem is the
"high quality" one the selector should up-weight. The RATING that maps to rank is the
ability at which the user CLEARS the problem 80% of the time (CALIBRATION_P) — the same
margin the PWA serves and anchors at, NOT the 50%-point:

    rating_p = (logit(0.8) - d_p) / a_p  =  b_p + logit(0.8)/a_p   (b_p = -d_p/a_p = 50%-pt)

The slope-dependent 80% correction is load-bearing: under it a user clears exactly 80% on
problems whose rating == their ability theta for EVERY problem regardless of slope (a flat
low-discrimination problem genuinely needs more ability to clear at 80% than its 50%-point
implies). A monotone rating->rank map cannot recover this per-problem reordering, so it has
to be built here. The EGF mapping (pipeline.map_rating_v2_to_egf) then anchors each user's
80%-clear rating -> EGF against these ratings (self-consistent — it re-derives the curve
from whatever scalar we ship).

Every corpus problem is scorable (bots benchmark all, activations are cached for all), so
this assigns a rating to the whole set; unattempted problems fall back to the prior.

Outputs (parallel; nothing in ratings/ or pwa/tsumegos/ is touched):
  ratings_v2/<size>/tsumego_ratings_final.csv   tsumego_id, rating   (= 80%-clear rating)
  ratings_v2/<size>/tsumego_quality_final.csv   tsumego_id, quality  (= raw slope a_p)
  ratings_v2/<size>/tsumego_floor_final.csv     tsumego_id, floor    (= c_p, shipped per problem)
  ratings_v2/user_theta.csv                     profile, theta, n_attempts (for EGF mapping)
  ratings_v2/model_v2.pt                        weights + run metadata

Usage:
  ./.venv/bin/python3 -m pipeline.train_rating_v2 [--epochs 600] [--pca-dim 128] ...
"""

import argparse
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

# NOTE (prototype tech-debt): these live in auxiliary/ today. When v2 becomes the main
# backend, promote load_joint_*/build_profiles + the activation-cache helpers into utils/.
from auxiliary.compare_rating_models import (  # noqa: E402
    build_profiles, load_joint_activations, load_joint_bot_winrate,
)
from auxiliary.inspect_activation_signals import _cache_ids_path  # noqa: E402
from auxiliary.problem_attempt_coverage import load_first_time_attempts  # noqa: E402
from utils.fit_rank_curves import CALIBRATION_P  # noqa: E402

EPS = 1e-6
SIZES = [9, 13]
# logit of the serving/anchoring success rate (80%): the rating is the ability θ at which
# P(clear) = CALIBRATION_P, i.e. (CLEAR_LOGIT − d)/a, NOT the 50%-point −d/a. Single source
# of truth shared with the PWA + map_rating_v2_to_egf via fit_rank_curves.CALIBRATION_P.
CLEAR_LOGIT = float(np.log(CALIBRATION_P / (1.0 - CALIBRATION_P)))
# Max per-problem guessing floor. Must stay < CALIBRATION_P so the 80%-clear anchor
# logit((P*-c)/(1-c)) is finite; 0.78 leaves headroom below 0.80.
MAX_FLOOR = 0.78
# Effective support of the success curve in gap (= slope·(ability−rating)) logits: past ±ZCLIP
# the sigmoid is within ~0.7% of its asymptote, so a larger gap carries no curve information
# (and extreme gaps are broken-ability artifacts). Bounds both the artifact clip and the
# equal-width gap-DRO bin range — a structural constant, not a tuned one.
ZCLIP = 5.0


def dyadic_tail_bins(vals, depth):
    """Bin `vals` so cells are mass-balanced (quantile edges, never near-empty) BUT resolution
    concentrates at the two EXTREMES: start from a median split (2 bins), then each level splits
    the current outermost bin on each side at its own median. The depth-d extreme cell holds the
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


def clear_logit_vec(c):
    """Per-problem 80%-clear logit anchor under the 3PL: at the rating P(clear)=CALIBRATION_P,
    so σ(aθ+d)=(P*-c)/(1-c) and rating=(logit((P*-c)/(1-c))-d)/a. Vector over problems;
    reduces to the scalar CLEAR_LOGIT where c=0."""
    c = np.asarray(c, dtype=np.float64)
    q = np.clip((CALIBRATION_P - c) / (1.0 - c), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def load_bot_floor(keys, max_floor=MAX_FLOOR):
    """Per-problem guessing floor c_p = the LOWEST clear rate across the standard-temperature
    humanv0 bot ladder (humanv0_Xk / Xd; excludes the _t* temperature variants, kata,
    heuristic, sgf_move). The weak bots attempt problems far above their strength, sampling
    the far-below-the-problem regime that human first-time attempts never reach (selection
    serves ~80%-clear) — so this identifies the floor the human data structurally can't.
    Aligned to `keys` ((size, tid) order); missing problems -> 0. Clipped to [0, max_floor]."""
    import glob
    by_key = {}
    for size in SIZES:
        adir = os.path.join(PROJECT_ROOT, "attempts", str(size))
        for f in sorted(glob.glob(os.path.join(adir, "*.txt"))):
            b = os.path.basename(f)
            if "humanv0_" not in b or not (b.endswith("k.txt") or b.endswith("d.txt")):
                continue
            with open(f) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) != 2:
                        continue
                    try:
                        v = float(p[1])
                    except ValueError:
                        continue
                    kk = (size, p[0])
                    if kk not in by_key or v < by_key[kk]:
                        by_key[kk] = v
    c = np.zeros(len(keys), dtype=np.float32)
    miss = 0
    for i, key in enumerate(keys):
        if key in by_key:
            c[i] = by_key[key]
        else:
            miss += 1
    return np.clip(c, 0.0, max_floor), miss


class RatingV2(nn.Module):
    """P = c_p + (1-c_p)·sigmoid(a_p * theta_u + d_p); c_p is the FIXED per-problem bot
    guessing floor (load_bot_floor), d_p / a_p = feature prior + per-problem residual."""

    def __init__(self, n_profiles, n_problems, fdim, diff_hidden=(128, 64), dropout=0.1,
                 slope_bias=0.5413, floor_c=None, floor_scale=1.0, learn_floor_head=False,
                 max_floor=MAX_FLOOR):
        super().__init__()
        self.theta = nn.Embedding(n_profiles, 1)
        nn.init.zeros_(self.theta.weight)
        layers, d = [], fdim
        for h in diff_hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        head = nn.Linear(d, 1)              # bias = global difficulty offset
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.diff_head = nn.Sequential(*layers, head)
        self.slope_head = nn.Linear(fdim, 1, bias=False)
        nn.init.zeros_(self.slope_head.weight)
        # softplus(slope_bias) is the prior slope a_p (sparse problems fall back here).
        # 1PL/2PL default 0.5413 → a≈1; with the bot floor carrying the tail we raise it
        # (prior a≈1.9) so sparsely attested HARD problems don't deflate to a flat curve.
        self.register_buffer("slope_bias", torch.tensor(float(slope_bias)))
        self.delta = nn.Embedding(n_problems, 1)   # per-problem difficulty residual
        self.eta = nn.Embedding(n_problems, 1)     # per-problem slope residual
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.eta.weight)
        # 3PL floor P = c_p + (1-c_p)·σ(aθ+d), c_p = floor_scale·c_bot, FIXED (not learned).
        # The bot ladder (load_bot_floor) gives a per-problem floor that PREDICTS the human
        # far-below-rank clear rate monotonically (corr +0.36) but over-states its LEVEL by a
        # stable ~1.4×: human floor ≈ 0.70·bot floor (auxiliary/floortest.py — e.g. bot 0.76 →
        # human ~0.54, genuine high floors on hard "coin-flip between two unreadable moves"
        # problems). The level CANNOT be learned jointly: d_p/a_p are free per problem, so the
        # human data (served ~80%-clear, sparse far-below) can't separate "hard+high floor"
        # from "easy+no floor" — any learnable floor scale collapses to ~0 and is absorbed
        # into difficulty (verified: a learnable gamma drove the floor to 0 and rescaled every
        # rating). So the floor is FIXED (the original design that breaks the confound) but at
        # the externally-calibrated HUMAN level floor_scale=0.70, not the raw bot 1.0. The
        # per-problem SHAPE (which the bots get right) is preserved.
        self.use_floor = floor_c is not None
        # max_floor caps the HUMAN floor c_p. 0.5 is the principled "guess between two
        # unreadable options" ceiling (values above that aren't a guess floor, they're the
        # problem being easy — a regime far-below attempts rarely test); a lower cap also gives
        # a learnable floor head less room to flatten curves, shrinking the floor↔ability trade.
        self.max_floor = max_floor
        cb = np.clip(floor_scale * (floor_c if floor_c is not None
                                    else np.zeros(n_problems, np.float32)), EPS, max_floor)
        # floor_scale·c_bot is the PRIOR MEAN (the validated fixed floor), held in logit space.
        self.register_buffer("floor_prior_logit", torch.logit(torch.as_tensor(cb, dtype=torch.float32)))
        # Optional learnable floor HEAD: a feature deviation (generalizes the human-vs-bot floor
        # correction to problems with no far-below data) + a per-problem residual, both ridge-
        # shrunk toward the prior. The far-below human attempts (up-weighted by 2D-DRO over z)
        # identify the deviation; the prior covers the rest. Init zero ⇒ starts AT the fixed
        # floor_scale·c_bot, so this only ADDS structure the data supports.
        self.learn_floor_head = learn_floor_head and floor_c is not None
        if self.learn_floor_head:
            self.floor_head = nn.Linear(fdim, 1, bias=False)
            nn.init.zeros_(self.floor_head.weight)
            self.floor_resid = nn.Embedding(n_problems, 1)
            nn.init.zeros_(self.floor_resid.weight)

    def floor_for(self, prob_idx, feat=None):
        """Per-problem human floor c_p. Fixed = floor_scale·c_bot (sigmoid of the prior logit);
        with the learnable head, += feature deviation + per-problem residual (prior-anchored).
        Clamped [0, MAX_FLOOR]; zeros when the floor is disabled (2PL)."""
        if not self.use_floor:
            return torch.zeros(prob_idx.shape, device=prob_idx.device)
        logit = self.floor_prior_logit[prob_idx]
        if self.learn_floor_head:
            logit = logit + self.floor_head(feat).squeeze(-1) + self.floor_resid(prob_idx).squeeze(-1)
        return torch.sigmoid(logit).clamp(0.0, self.max_floor)

    def diff_slope(self, prob_idx, feat):
        d = self.diff_head(feat).squeeze(-1) + self.delta(prob_idx).squeeze(-1)
        a = torch.nn.functional.softplus(
            self.slope_bias + self.slope_head(feat).squeeze(-1)
            + self.eta(prob_idx).squeeze(-1)).clamp(0.2, 12.0)  # floor avoids b=-d/a blowup
        return d, a

    def forward(self, prof_idx, prob_idx, feat):
        d, a = self.diff_slope(prob_idx, feat)
        p = torch.sigmoid(a * self.theta(prof_idx).squeeze(-1) + d)
        if self.use_floor:
            c = self.floor_for(prob_idx, feat)
            p = c + (1.0 - c) * p
        return p.clamp(EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "ratings_v2"))
    ap.add_argument("--pca-dim", type=int, default=128, help="Joint activation PCs.")
    ap.add_argument("--diff-hidden", type=int, nargs="*", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.01,
                    help="0.01 is the stabilized rate: at 0.03 the min-max DRO pass "
                         "oscillated and diverged late (seed-to-seed log-loss spread "
                         "0.10); 0.01 makes the DRO pass track the mean pass, collapses "
                         "seed variance to ~0.001, and converges lower. See the LR sweep "
                         "in docs/rating_model_comparison_results.md.")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--min", type=int, default=5,
                    help="Min first-time correct AND incorrect to keep a user.")
    ap.add_argument("--min-profile", type=int, default=8)
    ap.add_argument("--lambda-resid", type=float, default=0.5,
                    help="Ridge on per-problem residuals delta/eta = prior precision "
                         "(empirical-Bayes shrinkage toward the feature prior). With "
                         "~5 attempts/problem the prior should dominate, so keep this "
                         "fairly strong. The key knob to calibrate.")
    ap.add_argument("--winsor", type=float, default=1.0,
                    help="Winsorize final ratings to [p, 100-p] percentile (robustness to "
                         "the residual b=-d/a tail for low-slope problems).")
    ap.add_argument("--bot-floor", action=argparse.BooleanOptionalAction, default=True,
                    help="3PL with a FIXED per-problem guessing floor c_p from the bot "
                         "ladder (load_bot_floor = min clear rate over standard-temperature "
                         "humanv0 bots): P=c_p+(1-c_p)·σ(aθ+d). The bots sample the "
                         "far-below-the-problem regime human attempts never reach, so c_p is "
                         "the real floor; supplying it keeps a_p from deflating to mimic one "
                         "(the deflation that fattened the far-from-rank tail and corrupted "
                         "the rank estimator). Also raises the slope prior to a≈1.9. "
                         "--no-bot-floor reverts to the 2PL (c_p=0).")
    ap.add_argument("--floor-scale", type=float, default=0.70,
                    help="FIXED probability-space scale on the bot floor: c_human = "
                         "floor_scale·c_bot. The bot floor ranks the per-problem human floor "
                         "correctly (corr +0.36) but over-states its level by a stable ~1.4× "
                         "(human ≈ 0.70·bot; auxiliary/floortest.py + gamma.py). The level is "
                         "NOT learnable jointly — d_p/a_p absorb it and any learnable scale "
                         "collapses to 0 (verified) — so it's set here, externally calibrated. "
                         "1.0 = raw bot floor (pre-fix, for A/B).")
    ap.add_argument("--learn-floor-head", action=argparse.BooleanOptionalAction, default=True,
                    help="Learn a per-problem floor DEVIATION from the floor_scale·c_bot prior: "
                         "c_p = σ(logit(floor_scale·c_bot) + floor_head(feat) + floor_resid_p), "
                         "ridge-shrunk to the prior (--lambda-floor). The far-below human "
                         "attempts (up-weight with --dro-2d) identify per-problem human-vs-bot "
                         "floor deviations the global scale misses; the feature head "
                         "generalizes them to problems with no far-below data. Init zero = the "
                         "fixed floor_scale·c_bot.")
    ap.add_argument("--lambda-floor", type=float, default=0.5,
                    help="Ridge on the per-problem floor residual (shrink to the bot-floor prior).")
    ap.add_argument("--max-floor", type=float, default=MAX_FLOOR,
                    help="Cap on the human floor c_p. 0.5 = the 'guess between two unreadable "
                         "options' ceiling (principled; also limits a learnable floor head's "
                         "room to flatten curves and trade against ability). Must stay < 0.8.")
    ap.add_argument("--dro-2d", action=argparse.BooleanOptionalAction, default=True,
                    help="Group-DRO over (relative-gap z × absolute-rating) instead of rating "
                         "alone. The z dimension up-weights far-below cells that IDENTIFY the "
                         "floor (otherwise drowned by the near-rank bulk); pairs with "
                         "--learn-floor-head. z is clipped to ±5 (artifact guard).")
    ap.add_argument("--dro-z-bins", type=int, default=5,
                    help="Number of relative-gap z quantile bins for --dro-2d (× --dro-bins).")
    ap.add_argument("--dro-dyadic", action=argparse.BooleanOptionalAction, default=True,
                    help="Bin BOTH DRO axes by dyadic-tail-by-mass (--dro-z-depth / --dro-r-depth): "
                         "mass-balanced cells (never near-empty) with resolution recursively "
                         "concentrated at the extremes. Combines quantile's stable cells with "
                         "equal-width's extreme resolution, no placed edges. --no-dro-dyadic uses "
                         "plain quantile/uniform with --dro-bins / --dro-z-bins.")
    ap.add_argument("--dro-z-depth", type=int, default=4,
                    help="Dyadic tail depth for the gap axis (n_bins = 2·depth+2; depth-d extreme "
                         "cell = most-extreme 2^-(d+1) of mass). The 'how fine at the gap tails' knob.")
    ap.add_argument("--dro-r-depth", type=int, default=3,
                    help="Dyadic tail depth for the absolute-rating axis (n_bins = 2·depth+2).")
    ap.add_argument("--dro-z-uniform", action=argparse.BooleanOptionalAction, default=True,
                    help="Bin the gap axis equal-WIDTH over the success curve's saturation range "
                         "(default) rather than equal-count (quantile). Equal-width makes "
                         "--dro-z-bins a real lever on the far-below extremes (quantile bins "
                         "collapse the whole far-below into one cell). No placed edges.")
    ap.add_argument("--lambda-theta", type=float, default=0.01,
                    help="Fixed θ-ridge precision (only used with --no-hier-theta).")
    ap.add_argument("--hier-theta", action=argparse.BooleanOptionalAction, default=True,
                    help="Unit-variance (standard IRT) ability identification: ability ~ N(0,1) "
                         "with variance FIXED, 1/N-scaled to the mean-BCE term, plus a mean-pin "
                         "gauge (--lambda-theta-mean). No learned variance ⇒ no log-σ→0 collapse "
                         "(the failure that needed a clamp); the scale lives in the slopes and "
                         "the spread is data-driven. Replaces the fixed-precision ridge that "
                         "compressed ability (see project_theta_compression). --no-hier-theta "
                         "reverts to lambda_theta·mean(ability²) for A/B.")
    ap.add_argument("--lambda-theta-mean", type=float, default=10.0,
                    help="Gauge anchor pinning mean ability to 0 under --hier-theta (the prior's "
                         "own location pull is weak). Constrains only the 1-d.o.f. mean, not the "
                         "spread.")
    ap.add_argument("--lambda-w", type=float, default=1e-4,
                    help="Ridge on the feature-head weights.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dro-skill", action=argparse.BooleanOptionalAction, default=True,
                    help="ON by default: spread held-out error evenly across the rating "
                         "range with a base-rate-centered group-DRO objective (replaces "
                         "mean BCE on the human-attempt term). Two-pass: pass 1 trains with "
                         "mean BCE to get the rating, which defines the rating bins; pass 2 "
                         "retrains with DRO over those bins. `--no-dro-skill` reverts to "
                         "plain ERM (mean BCE), which is overall-LogLoss-optimal but "
                         "concentrates error at the difficulty tails. See the --even-error "
                         "study in docs/rating_model_comparison_results.md.")
    ap.add_argument("--dro-eta", type=float, default=0.2,
                    help="Group-DRO exponentiated-gradient step on per-bin base-centered "
                         "loss. 0.2 is the flatness↔cost knee (η≥~1.6 collapses q onto one "
                         "bin; <0.1 under-flattens the U-shaped 9×9 board).")
    ap.add_argument("--dro-bins", type=int, default=10,
                    help="Equal-count rating-quantile bins the DRO objective groups over.")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    # Prior slope a_p (sparse problems fall back here): with the bot floor carrying the
    # tail, hard problems should sit near the empirically-fit discrimination (~1.9), not
    # the flat a≈1 the floorless fit needs. slope_bias = softplus^{-1}(prior_a).
    prior_a = 1.9 if args.bot_floor else 1.0
    slope_bias_val = float(np.log(np.expm1(prior_a)))

    # --- scorable corpus = problems with a cached activation vector (per size) ---
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    print("activation-cache problems: " + ", ".join(
        f"{s}:{len(tids_by_size[s])}" for s in SIZES))
    act, k = load_joint_activations(SIZES, tids_by_size, args.pca_dim, args.seed)
    bw, nb = load_joint_bot_winrate(SIZES, tids_by_size)
    # a problem is scorable iff it has BOTH feature blocks
    keys = [key for key in act if key in bw]                 # (size, tid)
    key_idx = {key: i for i, key in enumerate(keys)}
    fdim = k + nb
    feat = np.empty((len(keys), fdim), dtype=np.float32)
    for key, i in key_idx.items():
        feat[i] = np.concatenate([act[key], bw[key]])
    print(f"scorable problems {len(keys)} (9:{sum(s==9 for s,_ in keys)} "
          f"13:{sum(s==13 for s,_ in keys)}) | feat dim {fdim} (act {k} + bot {nb})")

    # --- per-problem bot guessing floor c_p (fixed input, aligned to keys) ---
    if args.bot_floor:
        floor_c, fmiss = load_bot_floor(keys)
        nz = floor_c > 0
        print(f"bot floor c_p: {nz.sum()}/{len(keys)} problems have a floor "
              f"(missing {fmiss}) | mean {floor_c[nz].mean():.3f} "
              f"p[5,50,95] {np.percentile(floor_c[nz],[5,50,95]).round(3)} "
              f"max {floor_c.max():.3f}")
    else:
        floor_c = np.zeros(len(keys), dtype=np.float32)
        print("bot floor DISABLED (--no-bot-floor): 2PL, c_p=0")
    # The clear-anchor logit per problem (rating = (clv - d)/a so P(clear)=80% at rating) now
    # depends on the LEARNED floor, so it's recomputed from the trained model (model_clv), not
    # fixed up front.

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
    prof_df = prof_df[prof_df["profile"].isin(
        set(pc.index[pc >= args.min_profile]))]
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

    # --- fit ---
    tfeat = torch.tensor(feat, device=device)
    tprof = torch.tensor(prof_idx, device=device)
    tprob = torch.tensor(prob_idx, device=device)
    ty = torch.tensor(y, device=device)
    bce = nn.BCELoss()
    n = len(y)

    epoch_records = []   # (pass, epoch, mean_train_BCE, max_dro_q) per epoch, for stability analysis

    def fit_model(dro=None):
        """Train one RatingV2. dro=(rank_bin_tensor, base_g, G, eta) swaps the human
        BCE term for a base-rate-centered online group-DRO over rank bins (Sagawa et
        al., centered to target REDUCIBLE skill-over-base, not the aleatoric base-rate
        floor — see docs/rating_model_comparison_results.md --even-error). The ridge
        priors are unchanged. q persists across minibatches/epochs."""
        torch.manual_seed(args.seed)
        m = RatingV2(n_profiles, len(keys), fdim, tuple(args.diff_hidden),
                     args.dropout, slope_bias=slope_bias_val,
                     floor_c=(floor_c if args.bot_floor else None),
                     floor_scale=args.floor_scale,
                     learn_floor_head=args.learn_floor_head,
                     max_floor=args.max_floor).to(device)
        feat_params = list(m.diff_head.parameters()) + list(m.slope_head.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        if dro is not None:
            rb, base_g, G, eta = dro
            q = torch.ones(G, device=device) / G
        for ep in range(args.epochs):
            m.train()
            perm = torch.randperm(n, device=device)
            for b0 in range(0, n, args.batch):
                bi = perm[b0:b0 + args.batch]
                opt.zero_grad()
                p = m(tprof[bi], tprob[bi], tfeat[tprob[bi]])
                if dro is None:
                    data_loss = bce(p, ty[bi])
                else:
                    l = torch.nn.functional.binary_cross_entropy(
                        p, ty[bi], reduction="none")
                    gb = rb[bi]
                    gl = torch.zeros(G, device=device).index_add(0, gb, l)
                    cnt = torch.zeros(G, device=device).index_add(
                        0, gb, torch.ones_like(l))
                    present = cnt > 0
                    Lg = gl[present] / cnt[present]
                    with torch.no_grad():           # update q on CENTERED bin loss
                        q[present] = q[present] * torch.exp(
                            eta * (Lg.detach() - base_g[present]))
                        q[~present] = 0.0
                        q /= q.sum().clamp(min=EPS)
                    data_loss = (q[present] * Lg).sum()
                loss = data_loss + args.lambda_resid * (m.delta.weight.pow(2).mean()
                                                        + m.eta.weight.pow(2).mean())
                if args.hier_theta:
                    # Unit-variance (standard IRT / factor-analysis) ability identification:
                    # ability ~ N(0, 1) with the variance FIXED, not learned. The scale lives
                    # in the slopes. There is no σ parameter, hence no log-σ→0 collapse
                    # degeneracy (a LEARNED variance is rewarded for shrinking ability to nothing
                    # once the floors/difficulties fit the per-problem averages — that's what
                    # forced the magic clamp). The cross-entropy rewards using ability to fit
                    # per-user differences (which a per-problem floor cannot), so the spread is
                    # data-driven; the unit prior only anchors the scale and pools sparse users.
                    # 1/N-scaled to the mean-BCE term; "1" is the unit, not a tuned knob. A firm
                    # mean-pin sets the ability gauge (the prior's own location pull is weak).
                    th = m.theta.weight
                    thm = th.mean()
                    loss = loss + (0.5 * th.pow(2)).sum() / n + args.lambda_theta_mean * thm.pow(2)
                else:
                    loss = loss + args.lambda_theta * m.theta.weight.pow(2).mean()
                loss = loss + args.lambda_w * sum(w.pow(2).sum() for w in feat_params)
                if m.learn_floor_head:                  # shrink the floor head/residual to the prior
                    loss = loss + args.lambda_floor * m.floor_resid.weight.pow(2).mean()
                    loss = loss + args.lambda_w * m.floor_head.weight.pow(2).sum()
                loss.backward()
                opt.step()
            # Record the UNWEIGHTED mean train BCE every epoch (consistent metric across passes
            # and seeds) + the DRO weight concentration (max q; high = the DRO piling onto one
            # cell = the suspected instability source), for the per-epoch stability trace.
            m.eval()
            with torch.no_grad():
                ll = bce(m(tprof, tprob, tfeat[tprob]), ty).item()
            maxq = float(q.max().item()) if dro is not None else float("nan")
            epoch_records.append((1 if dro is None else 2, ep, ll, maxq))
            if ep % 50 == 0 or ep == args.epochs - 1:
                print(f"  [{'dro' if dro else 'mean'}] epoch {ep:4d}  train BCE {ll:.4f}"
                      + (f"  max_q {maxq:.3f}" if dro is not None else ""))
        return m

    def score(m):
        m.eval()
        with torch.no_grad():
            allprob = torch.arange(len(keys), device=device)
            d, a = m.diff_slope(allprob, tfeat)
        return d.cpu().numpy(), a.cpu().numpy()

    def model_clv(m):
        """Learned per-problem floor c_p and its 80%-clear anchor logit, from the trained
        model (floor_for). The rating = (clv - d)/a uses these, so the anchor moves with the
        learned human floor instead of the raw bot floor."""
        with torch.no_grad():
            c = m.floor_for(torch.arange(len(keys), device=device), tfeat).cpu().numpy()
        return c, clear_logit_vec(c)

    m = fit_model()
    if args.dro_skill:
        # Pass 1 done. Bin TRAIN rows by their problem's pass-1 rating b_p (the shipped
        # rating, pooled across boards in shared ability units), equal-count quantiles;
        # per-bin base-rate loss = the bin's human-success entropy (the aleatoric floor).
        d0, a0 = score(m)
        _, clv0 = model_clv(m)                               # 80%-clear anchor at the learned floor
        rating0 = (clv0 - d0) / a0
        row_b = rating0[prob_idx]                            # per-problem 80%-clear rating
        # Each axis bins either by dyadic-tail-by-mass (resolution concentrated at the extremes,
        # cells mass-balanced so never near-empty — depth is the "how fine" knob), or plain
        # equal-count quantiles (--no-dro-dyadic). Both axes can be tuned independently.
        if args.dro_dyadic:
            rbin, nrb = dyadic_tail_bins(row_b, args.dro_r_depth)
        else:
            r_edges = np.quantile(row_b, np.linspace(0, 1, args.dro_bins + 1)); r_edges[-1] += 1e-6
            rbin = np.clip(np.searchsorted(r_edges, row_b, side="right") - 1, 0, args.dro_bins - 1)
            nrb = args.dro_bins
        if args.dro_2d:
            # 2D group = (relative-gap z bin) × (absolute-rating bin). z = a·θ+d−clv = a·(θ−rating)
            # is the user-problem gap in success-curve logits. Clip z to tame the artifact tail
            # (broken/clamped θ on easy problems → fake extreme gaps).
            theta0 = m.theta.weight.detach().cpu().numpy().ravel()
            z_row = np.clip(a0[prob_idx] * theta0[prof_idx] + d0[prob_idx] - clv0[prob_idx], -ZCLIP, ZCLIP)
            if args.dro_dyadic:
                zbin, nzb = dyadic_tail_bins(z_row, args.dro_z_depth)
            elif args.dro_z_uniform:
                z_edges = np.linspace(-ZCLIP, ZCLIP, args.dro_z_bins + 1); z_edges[-1] += 1e-6
                zbin = np.clip(np.searchsorted(z_edges, z_row, side="right") - 1, 0, args.dro_z_bins - 1); nzb = args.dro_z_bins
            else:
                z_edges = np.quantile(z_row, np.linspace(0, 1, args.dro_z_bins + 1)); z_edges[-1] += 1e-6
                zbin = np.clip(np.searchsorted(z_edges, z_row, side="right") - 1, 0, args.dro_z_bins - 1); nzb = args.dro_z_bins
            G = nrb * nzb
            rb = (zbin * nrb + rbin).astype(np.int64)
            tag = f"2D {nzb}z×{nrb}r" + ("(dyadic)" if args.dro_dyadic else "")
        else:
            G = nrb
            rb = rbin.astype(np.int64)
            tag = f"{nrb} rating" + ("(dyadic)" if args.dro_dyadic else "")
        br = np.array([y[rb == g].mean() if (rb == g).any() else 0.5 for g in range(G)])
        br = np.clip(br, EPS, 1 - EPS)
        base_g = torch.tensor(-(br * np.log(br) + (1 - br) * np.log(1 - br)),
                              dtype=torch.float32, device=device)
        rbt = torch.tensor(rb, dtype=torch.long, device=device)
        print(f"  dro_skill: {tag} bins, eta {args.dro_eta} — refitting")
        m = fit_model((rbt, base_g, G, args.dro_eta))

    # --- score the whole corpus ---
    d, a = score(m)
    floor_learned, clv = model_clv(m)    # learned human floor c_p + its 80%-clear anchor
    if args.bot_floor:
        nz = floor_c > 0
        head = "head" if args.learn_floor_head else "fixed"
        print(f"  floor ({head}, scale {args.floor_scale:.2f}): bot->human "
              f"mean {floor_c[nz].mean():.3f}->{floor_learned[nz].mean():.3f} "
              f"p[5,50,95] {np.percentile(floor_c[nz],[5,50,95]).round(3)}->"
              f"{np.percentile(floor_learned[nz],[5,50,95]).round(3)}")
        if args.learn_floor_head:
            prior = np.clip(args.floor_scale * floor_c, EPS, args.max_floor)
            dev = floor_learned - prior      # head deviation from the floor_scale·c_bot prior
            print(f"    floor-head deviation from {args.floor_scale:.2f}·c_bot: "
                  f"mean |Δ| {np.abs(dev[nz]).mean():.3f}, p[5,95] "
                  f"{np.percentile(dev[nz],[5,95]).round(3)}")
    b = (clv - d) / a                    # rating = ability at which P = CALIBRATION_P (80%),
                                         # using each problem's floor-adjusted clear anchor
    lo, hi = np.percentile(b, [args.winsor, 100 - args.winsor])
    b = np.clip(b, lo, hi)                        # tame the low-slope b=-d/a tail
    dres = m.delta.weight.detach().cpu().numpy().ravel()
    eres = m.eta.weight.detach().cpu().numpy().ravel()
    print(f"  rating winsorized to [{lo:.2f},{hi:.2f}] | slope p[1,50,99] "
          f"{np.percentile(a,[1,50,99]).round(2)} | |delta| mean "
          f"{np.abs(dres).mean():.3f} (frac>0.5: {(np.abs(dres)>0.5).mean():.2f}) | "
          f"|eta| mean {np.abs(eres).mean():.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    for s in SIZES:
        sel = [i for i, (sz, _) in enumerate(keys) if sz == s]
        tids = [keys[i][1] for i in sel]
        sdir = os.path.join(args.out_dir, str(s))
        os.makedirs(sdir, exist_ok=True)
        pd.DataFrame({"tsumego_id": tids, "rating": b[sel]}).sort_values(
            "rating", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_ratings_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "quality": a[sel]}).sort_values(
            "quality", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_quality_final.csv"), index=False)
        pd.DataFrame({"tsumego_id": tids, "floor": floor_learned[sel]}).sort_values(
            "floor", ascending=False).to_csv(
            os.path.join(sdir, "tsumego_floor_final.csv"), index=False)
        print(f"  size {s}: wrote {len(tids)} ratings | rating "
              f"[{b[sel].min():.2f},{b[sel].max():.2f}] slope [{a[sel].min():.2f},"
              f"{a[sel].max():.2f}] | floor mean {floor_learned[sel].mean():.3f}")

    theta = m.theta.weight.detach().cpu().numpy().ravel()
    if args.hier_theta:
        print(f"  unit-variance ability prior | ability std {theta.std():.3f} "
              f"p[1,50,99] {np.percentile(theta,[1,50,99]).round(2)}")
    natt = pd.Series(prof_idx).value_counts().sort_index()
    pd.DataFrame({"profile": list(profiles.categories), "theta": theta,
                  "n_attempts": [int(natt.get(i, 0)) for i in range(n_profiles)]}).to_csv(
        os.path.join(args.out_dir, "user_theta.csv"), index=False)
    torch.save({"state_dict": m.state_dict(), "keys": keys, "fdim": fdim, "k": k,
                "n_bot_cols": nb, "args": vars(args)},
               os.path.join(args.out_dir, "model_v2.pt"))
    pd.DataFrame(epoch_records, columns=["pass", "epoch", "train_bce", "max_q"]).to_csv(
        os.path.join(args.out_dir, "epoch_log.csv"), index=False)
    print(f"wrote user_theta.csv ({n_profiles} profiles) + model_v2.pt to {args.out_dir}")


if __name__ == "__main__":
    main()
