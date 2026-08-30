#!/usr/bin/env python3
"""Split-half reliability of the tendency reading under CANDIDATE ERROR NULLS.

`tendency_split_half` holds the null fixed and asks which PROJECTION reads more
stably. This asks the other half of the same question: hold the projection
fixed, and ask which NULL does. The client's tendency reading is a sign test of
the picked cell's axis loading against the answer's, scored against the
availability null -- and what ships is UNIFORM over the alternatives the board
offered. `eval_error_null_models` shows the uniform null is not the best
predictor of which wrong variation actually gets picked; a weighted one is. That
says the weighted null is a better description of an error. It does NOT say the
per-user statistic built on it is a better MEASUREMENT, which is what this
answers.

Method, identical to `tendency_split_half` except for what varies: for every
user with enough recorded wrong first-time attempts, split their errors into
odd/even halves by their own error sequence (interleaved, so drift and
serving-era effects land in both halves). Per half, per axis, per NULL VARIANT,
compute exactly the shipped statistic (pwa/js/motif-stats.js): each error
contributes the sign of quantized correct-minus-picked, and its problem's own
null toward-probability; the half's reading is the Poisson-binomial z =
(nOver - sumP) / sqrt(sumPQ). Then correlate z_odd against z_even across users,
per axis, per variant, on the users valid in EVERY variant and both halves --
fully paired, so the comparison is the null and nothing else. A bootstrap over
users gives the CI on each variant's difference from uniform.

The variant changes ONLY each problem's p. The observed side -- which errors
count, and which way each one leans -- is byte-identical across variants,
because it never touches a weight.

PARAMETER-FREE VARIANTS ONLY. A fitted null would have to ship its parameters
with the corpus and refit with it, which is a doctrine decision this measurement
does not get to make; the fitted forms stay in `eval_error_null_models`, where
they are scored on prediction rather than on reliability. The weights come from
the SGF's own variation comments through `eval_error_null_models.variation_cells`
-- the same numbers phase 1 scores, read by the same scanner.

`fixed-0.5` IS THE CONTROL, NOT A CANDIDATE. It is the null the client used
before the availability null shipped: p = 0.5 on every problem and every axis,
no per-problem computation at all, so the reading collapses to the raw
toward-share. It is in the default table because it is what makes the rest of
the table readable. Under it, the per-problem MENU -- the answer to a tsumego is
by construction not a tenuki, so nearly every alternative loads higher on that
axis -- is left inside the statistic instead of being divided out, and a menu is
a property of the rank band a user is served, which is stable within a user and
therefore replicates across interleaved halves exactly as a real lean would. So
if `fixed-0.5` scores as the MOST reliable arm, that is not an argument for it:
it is the demonstration that split-half reliability is gameable by removing
information, and that this measurement can only ORDER candidates that have
already passed the validity question (the fixed null was retired on a measured
confound -- it flagged tenuki over-reach for ~94% of users -- not on
reliability). Read the whole table with that row as the yardstick.

The default projection is the SHIPPED fit on its full input. The reading-level
work currently stands on the activations+bot dev fit instead, which is:

    ./.venv/bin/python3 -m ablations.tendency_null_variants \\
        --model dev_runs/actbot_s3.pt --features activations+bot

CAVEAT, the same one the sibling carries: reliability rewards stability, not
truth. A null that is stable within a user replicates across interleaved halves
whether or not it describes the error better. A clear win here is evidence for a
variant, not proof; a tie says the variants are observationally equivalent at
current volume.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.model_constants import K_AXES  # noqa: E402
from ablations._projection import (  # noqa: E402
    FEATURE_BLOCKS, FittedProjection, ShippedProjection, problem_arrays,
    quantize)
from ablations.eval_error_null_models import EPS_WR, variation_cells  # noqa: E402
# The user halving, the two correlations and the shipped per-half gate are the
# sibling's, imported rather than restated so the two scripts cannot end up
# scoring different users.
from ablations.tendency_split_half import (  # noqa: E402
    MIN_ERRORS_HALF, load_user_errors, pearson, spearman)


def _lcb_fixed(w, v):
    """Binomial lower confidence bound on the winrate at its own visit count,
    at a conventional z of 1 -- no fitted radius."""
    p = np.clip(w, 0.0, 1.0)
    return np.maximum(p - np.sqrt(p * (1.0 - p) / np.maximum(v, 1.0)), EPS_WR)


# Each entry maps a problem's alternatives to WEIGHTS. Arguments are the arrays
# the SGF supplies per cell: mover-perspective winrate `w`, the visit count `v`
# behind that estimate, the cell's position `r` among the alternatives (0-based,
# SGF variation order) and their count `n`. Only ratios matter -- the weights
# are normalised away by the toward-probability -- so no entry needs to sum to
# anything.
VARIANTS = {
    "uniform": lambda w, v, r, n: np.ones_like(w),
    "winrate-linear": lambda w, v, r, n: np.maximum(w, EPS_WR),
    "visits-linear": lambda w, v, r, n: np.maximum(v, 1.0),
    "winrate*visits": lambda w, v, r, n: np.maximum(w, EPS_WR) * np.maximum(v, 1.0),
    "lcb-fixed": lambda w, v, r, n: _lcb_fixed(w, v),
    "1/order": lambda w, v, r, n: 1.0 / (r + 1.0),
    "linear-order": lambda w, v, r, n: np.maximum(n - r, 1.0),
}

# Nulls that are not a weighting of the alternatives at all: p is asserted, and
# the problem is never consulted. Only the retired client null lives here.
CONSTANT_P = {"fixed-0.5": 0.5}

# The shipped null, the four parameter-free forms that lead
# eval_error_null_models' phase 1 (in its order), and the retired fixed null as
# the control. Kept short on purpose: every added variant is another row per
# axis in a table meant to be read, and the two left out (winrate*visits,
# linear-order) are the two that predict the picked move WORST of the
# parameter-free set.
DEFAULT_VARIANTS = ("uniform,visits-linear,winrate-linear,1/order,lcb-fixed,"
                    "fixed-0.5")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default=DEFAULT_VARIANTS,
                    help=f"comma-separated, from "
                         f"{sorted(list(VARIANTS) + list(CONSTANT_P))}. "
                         f"'uniform' is the reference every delta is taken "
                         f"against and is added if omitted")
    ap.add_argument("--min-errors", type=int, default=2 * MIN_ERRORS_HALF,
                    help="errors a user needs to enter at all (halved between "
                         "the two splits)")
    ap.add_argument("--model", metavar="CKPT",
                    help="project through THIS checkpoint instead of the shipped "
                         "fit (e.g. dev_runs/actbot_s3.pt)")
    ap.add_argument("--features", choices=FEATURE_BLOCKS,
                    help="the per-problem block layout --model was FITTED on; it "
                         "is not recoverable from the file")
    ap.add_argument("--boot", type=int, default=2000,
                    help="bootstrap resamples over users for the delta CIs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if bool(args.model) != bool(args.features):
        ap.error("--model and --features go together")

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    known = set(VARIANTS) | set(CONSTANT_P)
    unknown = [v for v in names if v not in known]
    if unknown:
        ap.error(f"unknown variant(s) {unknown}; choose from {sorted(known)}")
    if "uniform" in names:
        names.remove("uniform")
    names = ["uniform"] + names
    V = len(names)

    import torch
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # A problem's variation cells are a few dozen rows, and at that size the
        # intra-op thread pool costs more in synchronisation than the matmuls
        # themselves (measured ~13x slower at 4 threads than at 1).
        torch.set_num_threads(1)
    P = (FittedProjection(device, args.model, args.features) if args.model
         else ShippedProjection(device))
    which = (f"{args.model} on its own full input (blocks={args.features})"
             if args.model else "the shipped fit on its own full input")

    by_problem, n_users = load_user_errors(args.min_errors)
    corpus = {(s, t) for s in P.tids_by_size for t in P.tids_by_size[s]}
    todo = sorted(k for k in by_problem if k in corpus)
    n_att = sum(len(v) for k, v in by_problem.items() if k in corpus)
    print(f"{n_users} users >= {args.min_errors} errors, {n_att} attempts over "
          f"{len(todo)} problems, device {device}\n"
          f"projection: {which} (fdim {P.fdim})\n"
          f"null variants: {', '.join(names)}", flush=True)

    K = K_AXES
    # (variant, user, half, axis) accumulators of the shipped statistic.
    shape = (V, n_users, 2, K)
    nOver = np.zeros(shape)
    nDir = np.zeros(shape)
    sumP = np.zeros(shape)
    sumPQ = np.zeros(shape)
    n_skip_problem = n_unresolved = n_order_fault = 0

    for done, (size, tid) in enumerate(todo, 1):
        cells = variation_cells(size, tid)
        if not cells or len(cells) < 2:
            n_skip_problem += 1
            continue
        cand, xys, correct_idx = problem_arrays(size, tid)
        xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
        rows = [xy_to_row.get(c["xy"]) for c in cells]
        if any(r is None for r in rows):
            n_skip_problem += 1
            continue
        if rows[0] != correct_idx:
            # variation_cells[0] IS the answer by generation's own ordering
            # invariant; a disagreement with the activation store's correct_idx
            # is a corpus fault, reported rather than worked around.
            n_order_fault += 1
            continue

        idx = np.array(rows, dtype=np.intp)
        inp = np.concatenate([P.pca(cand[idx]),
                              np.tile(P.tail(size, tid), (len(idx), 1))], axis=1)
        q = quantize(P.axes(inp.astype(np.float32)))
        # d = correct - cell; d < 0 means the cell loaded HIGHER = "toward".
        s = np.sign(q[0][None, :] - q)                      # (n_cells, K)

        n_alt = len(cells) - 1
        w = np.array([c["w"] for c in cells])
        vis = np.array([c["v"] for c in cells])
        # Rank among the ALTERNATIVES: cell 0 is the answer and ties with itself
        # on every axis, so its own rank is never used.
        rank = np.maximum(np.arange(len(cells)) - 1, 0).astype(np.float64)

        var_p = []
        for name in names:
            const = CONSTANT_P.get(name)
            if const is not None:
                # Asserted, never computed: the problem's alternatives are not
                # consulted at all. Finite everywhere, so it admits exactly the
                # errors every other variant admits -- an error is directional
                # or not, and that is decided by the sign alone.
                var_p.append(np.full(K, float(const)))
                continue
            wt = np.asarray(VARIANTS[name](w, vis, rank, float(n_alt)),
                            dtype=np.float64)
            above = (wt[:, None] * (s < 0)).sum(axis=0)
            below = (wt[:, None] * (s > 0)).sum(axis=0)
            tot = above + below
            with np.errstate(invalid="ignore"):
                var_p.append(np.where(tot > 1e-12, above / np.maximum(tot, 1e-12),
                                      np.nan))

        local = {c["xy"]: i for i, c in enumerate(cells)}
        for ui, half, x, y in by_problem[(size, tid)]:
            li = local.get((x, y))
            if li is None:
                n_unresolved += 1
                continue
            sl = s[li]
            for a in range(V):
                p = var_p[a]
                for k in range(K):
                    if sl[k] == 0 or not np.isfinite(p[k]):
                        continue
                    pk = float(p[k])
                    nDir[a, ui, half, k] += 1
                    sumP[a, ui, half, k] += pk
                    sumPQ[a, ui, half, k] += pk * (1.0 - pk)
                    if sl[k] < 0:                          # erred TOWARD
                        nOver[a, ui, half, k] += 1
        if done % 2000 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    axis_names = [str(c) for c in json.load(open(labels_path))["components"]] \
        if os.path.exists(labels_path) else [f"axis{i}" for i in range(K)]
    print(f"\nproblems skipped: {n_skip_problem}; answer-not-first faults: "
          f"{n_order_fault}; attempts with no variation cell: {n_unresolved}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    valid = (nDir >= MIN_ERRORS_HALF) & (sumPQ > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (nOver - sumP) / np.sqrt(np.where(sumPQ > 0, sumPQ, np.nan))

    # Pearson on z mixes lean with VOLUME (z grows with sqrt(n) and per-user n
    # varies widely, so heavy users dominate the covariance); the rank
    # correlation is the primary metric, Pearson kept for context.
    print(f"\n{'axis':<18}{'variant':<16}{'users':>6}{'r':>8}{'dr':>8}"
          f"{'dr 95% CI':>18}{'rho':>9}{'drho':>8}{'drho 95% CI':>18}")
    print("-" * 109)
    mean_drho = {n: [] for n in names}
    for k in range(K):
        ok = valid[:, :, :, k].all(axis=(0, 2))            # every variant, both halves
        # Two-step indexing: combining the boolean with the integer k in one
        # subscript is "advanced indices separated by a slice", which moves the
        # user axis to the front.
        zk = z[..., k][:, ok, :]                           # (variant, users, half)
        nu = int(ok.sum())
        if nu < 3:
            print(f"{axis_names[k][:17]:<18}{'':<16}{nu:>6}  (too few users)")
            continue
        r = [pearson(zk[a, :, 0], zk[a, :, 1]) for a in range(V)]
        rho = [spearman(zk[a, :, 0], zk[a, :, 1]) for a in range(V)]
        bd = np.empty((args.boot, V))
        bs = np.empty((args.boot, V))
        for b in range(args.boot):
            # ONE resample scored by every variant: the comparison is paired.
            sel = rng.integers(0, nu, size=nu)
            for a in range(V):
                bd[b, a] = pearson(zk[a, sel, 0], zk[a, sel, 1])
                bs[b, a] = spearman(zk[a, sel, 0], zk[a, sel, 1])
        for a in range(V):
            lo, hi = np.percentile(bd[:, a] - bd[:, 0], [2.5, 97.5])
            slo, shi = np.percentile(bs[:, a] - bs[:, 0], [2.5, 97.5])
            head = axis_names[k][:17] if a == 0 else ""
            dr = "" if a == 0 else f"{r[a] - r[0]:>8.3f}"
            drho = "" if a == 0 else f"{rho[a] - rho[0]:>8.3f}"
            dci = "" if a == 0 else f"[{lo:+.3f}, {hi:+.3f}]"
            sci = "" if a == 0 else f"[{slo:+.3f}, {shi:+.3f}]"
            mean_drho[names[a]].append(rho[a] - rho[0])
            print(f"{head:<18}{names[a]:<16}{nu:>6}{r[a]:>8.3f}{dr:>8}"
                  f"{dci:>18}{rho[a]:>9.3f}{drho:>8}{sci:>18}", flush=True)
    print("\nr / rho: odd-half against even-half tendency z across users, per axis.\n"
          "dr / drho: that variant minus 'uniform' on the SAME users and the SAME\n"
          "bootstrap resamples. Positive = that null reads more stably than the\n"
          "uniform one the client ships.")
    if "fixed-0.5" in names:
        print("\n'fixed-0.5' is the RETIRED client null and a CONTROL, not a\n"
              "candidate: p = 0.5 everywhere, so the problem is never consulted\n"
              "and the per-problem menu stays inside the statistic. A menu is a\n"
              "property of the rank band a user is served, so it is stable within\n"
              "a user and replicates across halves like a real lean. If that row\n"
              "wins, the finding is about this METRIC -- reliability is gameable\n"
              "by removing information -- and not about that null, which was\n"
              "retired on a measured confound rather than on stability.")
    print("\nmean drho across axes:")
    for n in names[1:]:
        vals = mean_drho[n]
        if vals:
            print(f"  {n:<16}{np.mean(vals):+.3f}")
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
