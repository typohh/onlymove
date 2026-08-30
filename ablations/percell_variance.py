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

B is then SPLIT. "Should the readings be normalized by rating?" is a question
about B alone: the rating is one number per problem, so it is constant across
that problem's cells and lives entirely in the between component. **The within
component cannot be rating-confounded by construction**, and no normalization
can change a within-share -- but a large B may be difficulty in disguise rather
than position-global motif content. So each axis's per-problem cell means are
regressed (OLS, with intercept) on that ARM'S OWN fitted triple -- rating, slope
and floor -- splitting

    B = B_triple (explained by the arm's own difficulty parameters)
      + B_resid  (position-global structure that is not the triple)

and the table reports the three-way shares of total variance W + B, plus
B_triple / B. Each arm is regressed on ITS OWN fit's parameters and never on
another's: the store arm on the params the packaging run shipped beside it
(--store-params), a dev checkpoint on its sibling .npz dump.

CAVEAT, and it is the same one ORTHO_RATING is built around: this is a
WHOLE-AXIS LINEAR statistic, and a linear statistic cannot see a symmetric or
tail-only dependence -- an axis whose BOTH tails are laden with difficulty has
no linear correlation with it at all. So B_triple is a LOWER BOUND on difficulty
entanglement, never a measurement of its absence. The finer instrument is the
per-tail weighting in utils/rating/tail_agreement, which is what the fit itself
is penalised on.

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

A dev arm's triple comes from the checkpoint's sibling dump (dev_runs/<name>.npz
beside dev_runs/<name>.pt), which is what makes "its own fit's parameters"
automatic rather than a flag a caller can get wrong.

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


def load_packed_params(path):
    """tid -> (rating, slope, floor) from a packed problem_params.bin: 16 bytes
    per record, a 4-byte id followed by three little-endian float32. The format
    is pipeline.package_for_pwa.emit_problem_params, and reading the SHIPPED file
    rather than the ratings CSVs is what lets an arm be scored against the triple
    that was published beside its own per-cell store after a later refit has
    overwritten ratings/ in place."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 16:
        raise SystemExit(f"{path} is not a multiple of 16 bytes")
    recs = raw.reshape(-1, 16)
    ids = [b.tobytes().hex() for b in recs[:, :4]]
    vals = recs[:, 4:].copy().view("<f4").reshape(-1, 3)
    return {t: (float(v[0]), float(v[1]), float(v[2])) for t, v in zip(ids, vals)}


def load_dump_params(path):
    """tid -> (rating, slope, floor) from a train_mirt dump npz, the same keys
    auxiliary.rating_dev.eval_user_holdout reads."""
    with np.load(path, allow_pickle=True) as z:
        for k in ("tid", "rating", "slope", "floor"):
            if k not in z.files:
                raise SystemExit(f"{path} has no '{k}' — not a train_mirt dump")
        return {str(t): (float(r), float(s), float(f))
                for t, r, s, f in zip(z["tid"], z["rating"], z["slope"],
                                      z["floor"])}


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

    def __init__(self, path, params_path=None):
        self.path = path
        rel = os.path.relpath(path, PROJECT_ROOT)
        self.name = f"store {path if rel.startswith('..') else rel}"
        self.detail = "the per-cell store as published (fp16, unquantized)"
        if params_path is None:
            from utils.rating.features import load_backend_problem_params
            self.params = load_backend_problem_params(tuple(SIZES))
            self.params_detail = "the live ratings/ CSVs"
        elif os.path.isdir(params_path):
            from utils.rating.features import load_backend_problem_params
            self.params = load_backend_problem_params(tuple(SIZES),
                                                      ratings_dir=params_path)
            self.params_detail = f"the ratings CSVs under {params_path}"
        else:
            self.params = load_packed_params(params_path)
            self.params_detail = f"packed params {params_path}"

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
        # ITS OWN fit's triple, by the trainer's own naming: a dev arm scored
        # against another fit's parameters would be a category error the caller
        # cannot see, so it is not a flag.
        dump = os.path.splitext(ckpt)[0] + ".npz"
        if not os.path.exists(dump):
            raise SystemExit(f"no dump beside {ckpt} — expected {dump}, which "
                             f"carries this fit's own rating/slope/floor")
        self.params = load_dump_params(dump)
        self.params_detail = f"its own dump {os.path.relpath(dump, PROJECT_ROOT)}"

    def cells(self, size, tid, var_xys, cand):
        tail = self.P.tail(size, tid)
        Z = self.P.pca(cand)
        inp = np.concatenate([Z, np.tile(tail, (len(Z), 1))], axis=1)
        return self.P.axes(inp.astype(np.float32)).astype(np.float64)


def decompose(per_problem, per_problem_correct, triple):
    """(W, B, B_triple, off-centre mean, off-centre sd) per axis from the list of
    per-problem (n_cells, K) blocks, each problem's correct-cell index, and the
    (P, 3) rating/slope/floor of THIS arm's own fit.

    B_triple is the part of the between component an OLS on the triple explains,
    so B_resid = B - B_triple. The regressors are centred before the solve, which
    with the intercept dropped is the same fit and better conditioned (rating
    runs around 7, floor around 0.1)."""
    means = np.array([b.mean(axis=0) for b in per_problem])          # (P, K)
    wvar = np.array([b.var(axis=0, ddof=1) for b in per_problem])    # (P, K)
    W = wvar.mean(axis=0)
    B = means.var(axis=0, ddof=1)

    X = triple - triple.mean(axis=0)
    Y = means - means.mean(axis=0)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    # Fitted and residual are orthogonal and both centred, so the sums of
    # squares add exactly and B splits without a remainder.
    B_triple = (X @ beta).var(axis=0, ddof=1)

    wsd = np.sqrt(wvar)
    delta = np.array([b[ci] for b, ci in zip(per_problem, per_problem_correct)]) - means
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.where(wsd > 0, delta / np.where(wsd > 0, wsd, 1.0), np.nan)
    return W, B, B_triple, np.nanmean(std, axis=0), np.nanstd(std, axis=0)


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
    ap.add_argument("--store-params", default=None, metavar="PATH",
                    help="the store arm's OWN triple: a packed problem_params.bin "
                         "(as package_for_pwa shipped beside that store) or a "
                         "ratings/ directory. Defaults to the live ratings/ CSVs, "
                         "which are correct ONLY for a store the current fit "
                         "wrote — a later refit overwrites ratings/ in place")
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
        arms.append(StoreArm(store, args.store_params))
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
    # A problem enters only if EVERY arm holds its own triple for it, so the
    # three-way split is computed on the same rows as the two-way one.
    n_drawn = len(sample)
    sample = [(s, t) for s, t in sample if all(t in a.params for a in arms)]
    n_no_params = n_drawn - len(sample)

    print(f"per-cell axis variance: within-problem (move-local, the only "
          f"component the client's readings consume) vs between-problem, with "
          f"the between component split against each arm's OWN fitted triple\n"
          f"{n_drawn} problems sampled, {args.n_per_size}/size, seed "
          f"{args.seed}, device {device}"
          + (f"; {n_no_params} dropped for a missing triple" if n_no_params
             else ""), flush=True)
    for i, a in enumerate(arms):
        print(f"  arm {i}: {a.name} — {a.detail}; triple from {a.params_detail}")
    print(flush=True)

    acc = [[] for _ in arms]
    correct, kept = [], []
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
        kept.append(tid)
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
        triple = np.array([a.params[t] for t in kept], dtype=np.float64)
        W, B, Btrip, offm, offsd = decompose(lst, correct, triple)
        total = W + B
        Bres = B - Btrip
        print(f"=== {a.name} ===")
        print("shares of TOTAL variance W+B. W = mean over problems of the "
              "problem's own cell variance,\nB = variance over problems of the "
              "problem's cell mean; both unweighted over problems.\n"
              "B splits by OLS of the cell means on this arm's OWN "
              "(rating, slope, floor):\nB_triple = explained (a LOWER BOUND on "
              "difficulty entanglement — a linear statistic\ncannot see a "
              "symmetric or tail-only dependence), B_resid = the rest. The "
              "WITHIN\ncomponent cannot be triple-confounded at all: the triple "
              "is constant across a\nproblem's cells, so no normalization can "
              "move a within-share.")
        print(f"{'axis':<22}{'within%':>9}{'B_resid%':>10}{'B_triple%':>11}"
              f"{'Btrip/B':>10}{'sd_within':>11}{'sd_between':>12}"
              f"{'n_prob':>9}{'n_cells':>9}")
        for k in range(K_AXES):
            print(f"{names[k][:21]:<22}{100.0 * W[k] / total[k]:>8.1f}%"
                  f"{100.0 * Bres[k] / total[k]:>9.1f}%"
                  f"{100.0 * Btrip[k] / total[k]:>10.1f}%"
                  f"{100.0 * Btrip[k] / B[k]:>9.1f}%"
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
