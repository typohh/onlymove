#!/usr/bin/env python3
"""Is "this cell is the answer" a cheap linear read in the trunk's INPUT space?

A per-cell supervision pass would train against recorded picks: the answer cell
labelled one way, the cell the player actually chose the other. Whether that can
use a CENSORED-OUTCOME loss (the player's pick is a lower bound on the answer's
score, not a labelled negative) depends entirely on how separable the two cells
already are BEFORE any head is fitted. If a trivial detector exists in the input
representation, a censored loss is satisfied by that detector alone: every pair
clears its margin from the first epoch, the gradient goes to zero and the pass
learns nothing about the cells it was meant to order. If picked-wrong cells sit
in the same region as answers, the censored form has real work to do and is safe.

So this measures the ceiling of the cheap detector directly, with no fit of our
own in the way: a plain logistic probe on the 128-dim joint-PCA activation block
— the exact activation columns pipeline.write_mirt_per_cell feeds the trunk,
loaded through _projection so it cannot drift from what ships. The per-problem
TAIL blocks are deliberately excluded: they are constant across a problem's
cells, so they carry no cell information at all and would only hand the probe
problem identity.

Three cell classes per problem, drawn only from problems with a recorded error:

  correct   the answer cell (the npz correct_idx; generation pins the answer as
            the first root variation)
  picked    the wrong cells players really chose (db/d1.sqlite, first-time
            failures with a recorded move, deduplicated to distinct cells).
            Split by whether the cell is one of the SGF root variations, since
            a pick KataGo never explored is separable for a reason that has
            nothing to do with the player
  random    seeded draw of --n-random other root-variation cells, excluding the
            answer and every picked cell — the CONTRAST condition, the cells a
            uniform alternative draw would have offered

Two probes, plus the contrast: correct vs picked is the decision-relevant one,
correct vs random the expected-easier reference. Both are sklearn's default L2
logistic regression, class-balanced, standardized inside each fold, scored under
5-fold GroupKFold GROUPED BY PROBLEM — a problem's answer and its wrong cells
must never straddle the split, or the probe scores problem-identity components
of the PCA block instead of anything about the move.

Every probe is run twice:

  raw       the vectors as the trunk sees them
  centered  each problem's sampled cells minus their own mean, which removes the
            problem-global component exactly. Separability that survives is
            MOVE-LOCAL — the only kind a per-cell head can use without the
            problem-global shortcut, and the only kind the client's
            within-problem readings could ever consume

CAVEAT: a linear probe is a LOWER bound on separability. A low AUC says no cheap
linear detector exists, not that no detector exists; the planned pass's own head
is nonlinear and sits on top of the same block. And AUC here is a property of
this corpus's error population — the cells players pick on problems they were
actually served, not of the alternative set at large.
"""

import argparse
import os
import sqlite3
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago.coords import gtp_to_xy  # noqa: E402
from utils.rating.model_constants import PCA_DIM, SEED, SIZES  # noqa: E402
from ablations._projection import (  # noqa: E402
    ShippedProjection, problem_arrays, variation_npz_rows)

# Cell classes, one row of the collected matrix each.
CLS_CORRECT, CLS_PICKED_VAR, CLS_PICKED_OUT, CLS_RANDOM = 0, 1, 2, 3
CLS_NAMES = ("correct", "picked-in-variation", "picked-outside", "random")

N_SPLITS = 5
MAX_ITER = 2000


def load_picked_cells():
    """{(size, tid): {(x, y): n_attempts}} — the DISTINCT wrong cells recorded
    against each problem by first-time failures, with how often each was picked.
    Distinct cells rather than attempts: the probe asks whether a cell is
    separable, and repeating a popular cell would weight the fit by popularity
    instead."""
    db = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")
    if not os.path.exists(db):
        raise SystemExit(f"missing {db} — run pipeline.fetch_database first")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT size, problem, move, COUNT(*) FROM tsumego_reports "
            "WHERE success = 0 AND is_repeat = 0 AND move IS NOT NULL AND move != '' "
            "GROUP BY size, problem, move").fetchall()
    finally:
        con.close()
    out = {}
    for size, tid, move, cnt in rows:
        xy = gtp_to_xy(move)
        if xy is None:
            continue
        cells = out.setdefault((int(size), str(tid)), {})
        key = (int(xy[0]), int(xy[1]))
        cells[key] = cells.get(key, 0) + int(cnt)
    return out


def collect_rows(P, picks, n_random, seed):
    """One pass over the activation store: the projected activation block of
    every sampled cell, plus its class, its problem and its board size.

    Cells are addressed by (x, y) against the npz's own xys — never
    positionally, since the npz is a board scan and the SGF variations are a
    pruned subset in a different order."""
    corpus = {(s, t) for s in SIZES for t in P.tids_by_size[s]}
    todo = sorted(k for k in picks if k in corpus)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    Xs, cls, grp, szs = [], [], [], []
    n_skip_problem = n_unmapped = n_at_correct = n_no_pick = 0
    gid = 0
    for done, (size, tid) in enumerate(todo, 1):
        cand, xys, correct_idx = problem_arrays(size, tid)
        var_rows = variation_npz_rows(size, tid, xys)
        if var_rows is None or correct_idx not in var_rows or len(var_rows) < 2:
            n_skip_problem += 1
            continue
        xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}
        var_set = set(var_rows)

        picked = []
        for (x, y), cnt in picks[(size, tid)].items():
            r = xy_to_row.get((x, y))
            if r is None:
                n_unmapped += 1        # not a legal cell of this board scan
                continue
            if r == correct_idx:
                n_at_correct += 1      # failed deeper in the tree, not a wrong
                continue               # first move — no wrong cell to read
            picked.append((r, CLS_PICKED_VAR if r in var_set else CLS_PICKED_OUT))
        if not picked:
            n_no_pick += 1
            continue

        taken = {correct_idx} | {r for r, _ in picked}
        avail = [r for r in var_rows if r not in taken]
        n_draw = min(n_random, len(avail))
        drawn = (rng.choice(np.array(avail, dtype=np.intp), size=n_draw,
                            replace=False) if n_draw else np.empty(0, dtype=np.intp))

        rows = ([correct_idx] + [r for r, _ in picked] + [int(r) for r in drawn])
        kinds = ([CLS_CORRECT] + [c for _, c in picked] + [CLS_RANDOM] * n_draw)
        Xs.append(P.pca(cand[np.array(rows, dtype=np.intp)]).astype(np.float32))
        cls += kinds
        grp += [gid] * len(rows)
        szs += [size] * len(rows)
        gid += 1
        if done % 4000 == 0:
            print(f"  {done}/{len(todo)} problems ({time.time()-t0:.0f}s)",
                  flush=True)

    print(f"\nproblems with a recorded error: {len(todo)}; used {gid}\n"
          f"  skipped (unreadable variations / answer not among them): "
          f"{n_skip_problem}\n"
          f"  skipped (no wrong first move left): {n_no_pick}\n"
          f"  picked cells dropped: {n_unmapped} off the board scan, "
          f"{n_at_correct} at the answer cell", flush=True)
    return (np.vstack(Xs), np.array(cls, dtype=np.int8),
            np.array(grp, dtype=np.int64), np.array(szs, dtype=np.int16))


def probe(X, y, groups):
    """5-fold GroupKFold logistic probe. Returns (auc, acc, bacc) as
    (mean, sd) pairs over folds, standardizing inside each fold so the raw and
    centered arms meet the same L2 penalty at the same effective scale."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 roc_auc_score)
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    auc, acc, bacc = [], [], []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=MAX_ITER))
        clf.fit(X[tr], y[tr])
        s = clf.decision_function(X[te])
        auc.append(roc_auc_score(y[te], s))
        pred = (s > 0.0).astype(np.int8)          # the 0.5 probability threshold
        acc.append(accuracy_score(y[te], pred))
        bacc.append(balanced_accuracy_score(y[te], pred))
    f = lambda v: (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2
    return f(auc), f(acc), f(bacc)


def run(name, X, cls, grp, pos_kinds, neg_kinds, rows_mask=None):
    """One probe over the rows of the named classes, restricted to problems
    that carry BOTH sides (so a probe is never fed unpaired answer cells)."""
    sel = np.isin(cls, pos_kinds + neg_kinds)
    if rows_mask is not None:
        sel &= rows_mask
    if not sel.any():
        return None
    y = np.isin(cls[sel], pos_kinds).astype(np.int8)
    g, Xs = grp[sel], X[sel]
    have_pos = set(g[y == 1].tolist())
    have_neg = set(g[y == 0].tolist())
    both = have_pos & have_neg
    keep = np.fromiter((v in both for v in g.tolist()), dtype=bool, count=len(g))
    if keep.sum() == 0:
        return None
    y, g, Xs = y[keep], g[keep], Xs[keep]
    a, c, b = probe(Xs, y, g)
    return (name, len(both), int((y == 1).sum()), int((y == 0).sum()), a, c, b)


def table(title, results):
    print(f"\n--- {title} ---")
    print(f"{'probe':<34}{'probs':>7}{'n_pos':>8}{'n_neg':>8}"
          f"{'AUC':>16}{'acc@.5':>16}{'bacc@.5':>16}")
    for r in results:
        if r is None:
            continue
        name, nprob, npos, nneg, a, c, b = r
        print(f"{name:<34}{nprob:>7}{npos:>8}{nneg:>8}"
              f"{f'{a[0]:.3f}+-{a[1]:.3f}':>16}"
              f"{f'{c[0]:.3f}+-{c[1]:.3f}':>16}"
              f"{f'{b[0]:.3f}+-{b[1]:.3f}':>16}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-random", type=int, default=2,
                    help="random other root-variation cells per problem (the "
                         "contrast class)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rows-cache", default=None,
                    help="npz of the collected cell rows; written on the first "
                         "run and reused after, so the probe side re-runs "
                         "without a second pass over the activation store")
    args = ap.parse_args()

    cache = args.rows_cache or os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"answer_separability_rows_r{args.n_random}_s{args.seed}.npz")

    t0 = time.time()
    if os.path.exists(cache):
        print(f"cell rows from {cache}", flush=True)
        with np.load(cache, allow_pickle=False) as d:
            X, cls, grp, szs = d["X"], d["cls"], d["grp"], d["size"]
    else:
        import torch
        # ShippedProjection is loaded for its joint-PCA basis alone: the probe
        # runs on the trunk's INPUT, so the trunk itself is deliberately unused.
        device = torch.device("cpu")
        P = ShippedProjection(device)
        picks = load_picked_cells()
        X, cls, grp, szs = collect_rows(P, picks, args.n_random, args.seed)
        np.savez(cache, X=X, cls=cls, grp=grp, size=szs)
        print(f"cell rows -> {cache}", flush=True)

    print(f"\n{len(X)} cells over {len(np.unique(grp))} problems, "
          f"{X.shape[1]} dims (joint-PCA activation block, PCA_DIM={PCA_DIM}), "
          f"seed {args.seed}, {args.n_random} random cells/problem")
    print("  " + " | ".join(f"{CLS_NAMES[k]} {int((cls == k).sum())}"
                            for k in range(4)))
    print("  " + " | ".join(f"{s}x{s} {int((szs == s).sum())} cells"
                            for s in SIZES))
    # The variation split is only worth two extra table rows when it separates
    # anything: a corpus whose SGF roots branch at every legal cell puts every
    # pick inside the variation set, and the split rows then restate the all row.
    has_outside = bool((cls == CLS_PICKED_OUT).any())
    if not has_outside:
        print("  every recorded pick is one of the problem's root variations, "
              "so the in-variation/outside split is not shown")

    # Problem-mean centering over each problem's own sampled cells: the
    # problem-global component leaves exactly, what remains is move-local.
    order = np.argsort(grp, kind="stable")
    Xc = np.empty_like(X)
    for g in np.unique(grp):
        idx = order[np.searchsorted(grp[order], g, "left"):
                    np.searchsorted(grp[order], g, "right")]
        Xc[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)

    print("\nAUC / acc: mean +- sd over "
          f"{N_SPLITS} GroupKFold folds GROUPED BY PROBLEM — a problem's answer\n"
          "and its wrong cells never straddle train/test, so no probe can score\n"
          "problem identity. acc@.5 is the 0.5-probability threshold of a\n"
          "class-balanced L2 fit, bacc@.5 its class-balanced twin. AUC 0.500 =\n"
          "the two cell classes are linearly indistinguishable in this space;\n"
          "1.000 = a free 'not the answer' detector, which is what would make a\n"
          "censored-outcome loss saturate.")

    for arm, M in (("raw vectors, as the trunk sees them", X),
                   ("within-problem centered (move-local component only)", Xc)):
        for scope, mask in ([("pooled", None)]
                            + [(f"{s}x{s}", szs == s) for s in SIZES]):
            res = [run("correct vs picked (all)", M, cls, grp,
                       [CLS_CORRECT], [CLS_PICKED_VAR, CLS_PICKED_OUT], mask)]
            if has_outside:
                res += [
                    run("correct vs picked (in-variation)", M, cls, grp,
                        [CLS_CORRECT], [CLS_PICKED_VAR], mask),
                    run("correct vs picked (outside)", M, cls, grp,
                        [CLS_CORRECT], [CLS_PICKED_OUT], mask),
                ]
            res += [run("correct vs random alternative", M, cls, grp,
                        [CLS_CORRECT], [CLS_RANDOM], mask)]
            table(f"{arm} | {scope}", res)

    print(f"\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
