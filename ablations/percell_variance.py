#!/usr/bin/env python3
"""How much of a per-cell axis reading is MOVE-LOCAL, and how much is the
problem wearing different cells' clothes?

Both client readings are within-problem comparisons. The tendency sign test
scores the picked cell against the ANSWER'S cell on the same board, and the
strength contrast is beta(correct) - beta(picked) on the same board. So the only
component either one can consume is the variation of an axis ACROSS the cells of
one problem. A component that is constant within a problem cancels in both
statistics exactly.

Nothing in the fit forces that component to be large. Training supervises one
target per problem, evaluated at the correct cell; every other cell of every
problem is unsupervised, reached only by fanning the same trunk over a different
activation row. A fit is therefore free to emit an axis that is essentially one
number per POSITION, repeated at each of its cells with only slight jitter --
well-formed, rating-relevant, and carrying almost nothing the client can read.
This measures which of those two things each fit actually emitted, per axis.

Method: draw the same seeded problem sample for every arm (--n-per-size per
board, --seed), take each problem's SGF root-variation cells (the exact cell set
the shipped per-cell bin publishes, via _projection.variation_npz_rows; the pass
is never embedded and so never enters), and decompose each axis over those cells
with the PROBLEM as the unit:

    within  W = mean over problems of the problem's own cell variance (ddof=1)
    between B = variance over problems of the problem's cell mean (ddof=1)

Both UNWEIGHTED over problems -- a 30-cell problem and an 80-cell problem count
the same, because the question is about a problem's reading, not about a corpus
average over cells. within-share = W / (W + B). The arms share one problem set
by construction: a problem enters only if EVERY arm can produce its cells, so
the tables are paired row for row and the counts are identical across arms.

B carries a sampling component -- each problem mean is estimated from n_p cells,
so it is inflated by W * mean(1/n_p) -- and the footnote line prints that term
as a share of B. At corpus cell counts it is a couple of percent and moves no
conclusion; it is printed rather than silently subtracted.

The supplementary row asks a different question of the same cells: does the
ANSWER sit off-centre among its own alternatives? Per problem, the correct
cell's delta from that problem's cell mean, divided by that problem's own within
sd, averaged over problems -- a standardized off-centre effect, one number per
axis. Near zero means the answer is a typical cell of its board on that axis;
far from zero means the axis separates the answer from the alternatives by
construction, which is the regime the tendency test's null exists to divide out.

Arms. The DEFAULT arm is the production per-cell store (axes/per_cell/, or the
retired sibling train_mirt renamed it to -- --store resolves whichever is on
disk, and takes an explicit path so the same script re-runs against a copy or
against a freshly written store). Store values are read as published, fp16 and
otherwise unquantized -- the client's 0.01 rounding is deliberately NOT applied,
since a rounding step can only shrink the within component and this measures
what the fit emitted. A dev checkpoint enters only as an explicitly named
comparison arm:

    ./.venv/bin/python3 -m ablations.percell_variance \\
        --arm dev_runs/actbot_s3.pt:activations+bot \\
        --arm dev_runs/featprod_s3.pt:all

Axes correspond BY INDEX across arms, which holds only because every arm here
was taught the same rule bank in the same order; there is no post-hoc matching.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.model_constants import K_AXES, SIZES  # noqa: E402
from ablations._projection import (  # noqa: E402
    FEATURE_BLOCKS, FittedProjection, problem_arrays, variation_npz_rows)


def resolve_store():
    """The per-cell store on disk: axes/per_cell/, else the sibling train_mirt
    renamed the previous fit's store to. A directory counts only if it holds a
    subdirectory per board size. Newest mtime wins among the renamed siblings,
    the exact name wins outright."""
    exact = os.path.join(PROJECT_ROOT, "axes", "per_cell")
    cands = [p for p in glob.glob(os.path.join(PROJECT_ROOT, "axes", "per_cell*"))
             if os.path.isdir(p)
             and all(os.path.isdir(os.path.join(p, str(s))) for s in SIZES)]
    if exact in cands:
        return exact
    return max(cands, key=os.path.getmtime) if cands else None


def sample_problems(tids_by_size, n_per_size, seed):
    """The seeded problem draw, shared by every arm. One generator advanced
    across sizes in SIZES order, so the sample is a function of (seed,
    n_per_size, corpus) alone."""
    rng = np.random.default_rng(seed)
    out = []
    for size in SIZES:
        tids = tids_by_size[size]
        n = min(n_per_size, len(tids))
        idx = np.sort(rng.choice(len(tids), size=n, replace=False))
        out.extend((size, tids[int(i)]) for i in idx)
    return out


class StoreArm:
    """The published per-cell store read as-is: fp16 rows keyed by board point."""

    def __init__(self, path):
        self.path = path
        rel = os.path.relpath(path, PROJECT_ROOT)
        self.name = f"store {path if rel.startswith('..') else rel}"
        self.detail = "the per-cell store as published (fp16, unquantized)"

    def cells(self, size, tid, var_xys, cand):
        npz = os.path.join(self.path, str(size), tid + ".npz")
        if not os.path.exists(npz):
            return None
        with np.load(npz, allow_pickle=False) as d:
            emb = np.asarray(d["embeddings"], dtype=np.float64)
            xys = np.asarray(d["xys"], dtype=np.int64)
        xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
        rows = []
        for x, y in var_xys:
            r = xy_to_row.get((int(x), int(y)))
            if r is None:
                return None
            rows.append(r)
        out = emb[np.array(rows, dtype=np.intp)]
        # The pass placeholder is an all-zeros row; the variation cells here are
        # non-pass by construction (variation_npz_rows skips it), so this is a
        # check on that, not a filter that should ever fire.
        if (out == 0.0).all(axis=1).any():
            return None
        return out


class ModelArm:
    """A fitted trunk projected over the same cells, on its own full input."""

    def __init__(self, device, ckpt, blocks):
        self.P = FittedProjection(device, ckpt, blocks)
        self.name = f"{os.path.relpath(ckpt, PROJECT_ROOT)} ({blocks})"
        self.detail = f"projected on its own full input, fdim {self.P.fdim}"

    def cells(self, size, tid, var_xys, cand):
        tail = self.P.tail(size, tid)
        Z = self.P.pca(cand)
        inp = np.concatenate([Z, np.tile(tail, (len(Z), 1))], axis=1)
        return self.P.axes(inp.astype(np.float32)).astype(np.float64)


def decompose(per_problem, per_problem_correct):
    """(within, between, off-centre mean, off-centre sd) per axis from the list
    of per-problem (n_cells, K) blocks and each problem's correct-cell index."""
    means = np.array([b.mean(axis=0) for b in per_problem])          # (P, K)
    wvar = np.array([b.var(axis=0, ddof=1) for b in per_problem])    # (P, K)
    W = wvar.mean(axis=0)
    B = means.var(axis=0, ddof=1)
    wsd = np.sqrt(wvar)
    delta = np.array([b[ci] for b, ci in zip(per_problem, per_problem_correct)]) - means
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.where(wsd > 0, delta / np.where(wsd > 0, wsd, 1.0), np.nan)
    return W, B, np.nanmean(std, axis=0), np.nanstd(std, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=None,
                    help="per-cell store directory for the default arm; "
                         "resolved from axes/per_cell* when omitted, 'none' to "
                         "run the model arms alone")
    ap.add_argument("--arm", action="append", default=[], metavar="CKPT:BLOCKS",
                    help="comparison arm: a dev checkpoint and the block layout "
                         f"it was FITTED on ({'|'.join(FEATURE_BLOCKS)}), which "
                         "is not recoverable from the file. Repeatable")
    ap.add_argument("--n-per-size", type=int, default=1500,
                    help="problems sampled per board size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    t0 = time.time()
    device = torch.device("cpu")

    arms = []
    store = args.store if args.store is not None else resolve_store()
    if store is not None and store != "none":
        if not os.path.isdir(store):
            raise SystemExit(f"no per-cell store at {store}")
        arms.append(StoreArm(store))
    elif args.store is None:
        print("no per-cell store found under axes/per_cell* — model arms only")
    for spec in args.arm:
        ckpt, _, blocks = spec.rpartition(":")
        if not ckpt or blocks not in FEATURE_BLOCKS:
            ap.error(f"--arm wants CKPT:BLOCKS with BLOCKS in {FEATURE_BLOCKS}, "
                     f"got {spec!r}")
        arms.append(ModelArm(device, ckpt, blocks))
    if not arms:
        ap.error("no arms — give a --store, an --arm, or both")

    # The corpus listing, not an artifact listing (see corpus_tids_by_size); the
    # model arms already hold it, and the store-only run needs it too.
    if any(isinstance(a, ModelArm) for a in arms):
        tids_by_size = next(a.P.tids_by_size for a in arms if isinstance(a, ModelArm))
    else:
        from utils.rating.features import corpus_tids_by_size
        tids_by_size = corpus_tids_by_size(SIZES)
    sample = sample_problems(tids_by_size, args.n_per_size, args.seed)

    print(f"per-cell axis variance: within-problem (move-local, the only "
          f"component the client's readings consume) vs between-problem\n"
          f"{len(sample)} problems sampled, {args.n_per_size}/size, seed "
          f"{args.seed}, device {device}", flush=True)
    for i, a in enumerate(arms):
        print(f"  arm {i}: {a.name} — {a.detail}")
    print(flush=True)

    acc = [[] for _ in arms]
    correct = []
    n_skip_sgf = n_skip_arm = 0
    for done, (size, tid) in enumerate(sample, 1):
        cand, xys, correct_idx = problem_arrays(size, tid)
        var_rows = variation_npz_rows(size, tid, xys)
        if var_rows is None or correct_idx not in var_rows or len(var_rows) < 2:
            n_skip_sgf += 1
            continue
        rows = np.array(var_rows, dtype=np.intp)
        var_xys = xys[rows]
        blocks = [a.cells(size, tid, var_xys, cand[rows]) for a in arms]
        # Paired by construction: one arm failing drops the problem everywhere,
        # so the tables share a problem set and the counts are comparable.
        if any(b is None for b in blocks):
            n_skip_arm += 1
            continue
        for lst, b in zip(acc, blocks):
            lst.append(b)
        correct.append(var_rows.index(correct_idx))
        if done % 500 == 0:
            print(f"  {done}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    if len(correct) < 3:
        raise SystemExit("too few problems survived to decompose")
    labels_path = os.path.join(PROJECT_ROOT, "axes", "labels.json")
    names = [str(c) for c in json.load(open(labels_path))["components"]] \
        if os.path.exists(labels_path) else [f"axis{i}" for i in range(K_AXES)]

    n_prob = len(correct)
    n_cells = sum(len(b) for b in acc[0])
    inv_n = float(np.mean([1.0 / len(b) for b in acc[0]]))
    print(f"\n{n_prob} problems decomposed ({n_skip_sgf} skipped on the SGF, "
          f"{n_skip_arm} on an arm), {n_cells} variation cells, "
          f"{n_cells / n_prob:.1f} cells/problem\n")

    for lst, a in zip(acc, arms):
        W, B, offm, offsd = decompose(lst, correct)
        share = W / (W + B)
        print(f"=== {a.name} ===")
        print("within-share = W/(W+B), W = mean over problems of the problem's "
              "own cell variance,\nB = variance over problems of the problem's "
              "cell mean; both unweighted over problems.")
        print(f"{'axis':<22}{'within%':>9}{'sd_within':>11}{'sd_between':>12}"
              f"{'n_prob':>9}{'n_cells':>9}")
        for k in range(K_AXES):
            print(f"{names[k][:21]:<22}{100.0 * share[k]:>8.1f}%"
                  f"{np.sqrt(W[k]):>11.4f}{np.sqrt(B[k]):>12.4f}"
                  f"{n_prob:>9}{n_cells:>9}")
        infl = W * inv_n / B
        print(f"  sampling inflation of B (W*mean(1/n_cells), as a share of B): "
              f"{', '.join(f'{v:.1%}' for v in infl)}")
        print(f"\n  answer off-centre: (correct cell - its problem's cell mean) "
              f"in that problem's own within sd")
        print(f"  {'axis':<20}{'mean':>9}{'sd':>9}")
        for k in range(K_AXES):
            print(f"  {names[k][:19]:<20}{offm[k]:>9.3f}{offsd[k]:>9.3f}")
        print(flush=True)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
