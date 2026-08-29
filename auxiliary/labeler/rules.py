"""Seed the labeler from hand-written rules instead of from clicks.

A motif you can STATE is worth more as a rule than as fifty clicks: the rule
labels every pool problem at once, positives AND negatives, which is what the
probe actually needs — hand labelling produces mostly-unconsidered rows, so a
label sits at a handful of positives against an unknown negative set for a long
time before it separates at all.

The rule is not the deliverable. What ships to `axes/axis_bank.npz` is the
PROBE's per-problem value, and the probe is fitted in the MIRT feature space.

READ THE --report AUC FOR WHAT IT IS, because three of these four rules are
computed from quantities the probe can already see, and each measures something
different as a result:

  life and death  the eight ownership parts ARE feature columns. The rule is a
                  CONJUNCTION over them (one part dominant, the other three
                  negligible), which a linear reader cannot express, so its AUC
                  measures the SHAPE gap — how much a linear probe loses on a
                  rule whose every input it holds.
  counting        the `move_evals` block is the winning move and its best
                  alternatives, evaluated; the rule is a comparison between two
                  of those. Near-perfect AUC here is the probe reading the
                  rule's own inputs, not evidence the motif is real.
  opening         `game_phase` ships the move prefix (summed with the setup
                  count). Same caveat, more so: the rule is one threshold on one
                  feature.
  ko              nothing in the feature matrix encodes a ko shape. This is the
                  only one of the four whose AUC answers the interesting
                  question — is the motif visible in the trunk's space at all.

A high AUC on the first three says the labels are self-consistent; a LOW one on
them says the rule's shape is beyond a linear reader, which is a fact about the
probe and points at a derived column rather than at the motif.

Every threshold below is a STARTING POINT, not a measurement. Run with --report
to see prevalence and how separable each rule is in feature space, move the
numbers, re-run; or take the rule labels into the labeler UI and correct them by
hand, which is the same loop with a human in it.

  ./.venv/bin/python3 -m auxiliary.labeler.rules --report
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf  # noqa: E402

from utils.katago.add_variations import parse_eval_comment  # noqa: E402
from utils.rating.model_constants import SIZES  # noqa: E402
from utils.katago.board_render import parse_sgf, simulate  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
LABELS_PATH = os.path.join(HERE, "labels.json")
SIGNAL_CACHE = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")
OWNERSHIP_CSV = os.path.join(SIGNAL_CACHE, "ownership_swing.csv")

# ---------------------------------------------------------------- parameters
# life and death: ONE outcome, no trade. The four DIRECTIONAL parts are the ones
# that change a group's status; the other four ("secured", "deadened") describe
# stones that end the same way either side of the choice and are ignored here.
# The rule logic itself lives in utils/rating/motif_rules.py, imported here so the
# pipeline step, this labelling CLI and the dev inspection tools cannot drift onto
# different thresholds. Only the CLI and its pool/report plumbing stay in this file.
from utils.rating.motif_rules import (  # noqa: E402,F401
    LABEL_SET, LD_FOCUS, LD_OTHER, LD_MIN_STONES, LD_TRADE_RATIO,
    COUNT_SCORE_MAX, COUNT_WR_MIN, KO_DEPTH, OPENING_MAX_PREFIX,
    SHAPE_WR_GAP, TERR_BASE, TERR_PER_STONE, OWNERSHIP_CSV, SIGNAL_CACHE,
    _subtree_is_resolved, _mover_eval, _line_moves,
    rule_life_and_death, rule_semeai, rule_territory, rule_shape, rule_counting,
    rule_ko, rule_opening)

def _sgf_facts(args):
    """Everything the SGF-side rules need, from ONE read of the file.

    Two parsers on purpose: sgfmill for the variation TREE (the variation
    ORDERING and the evaluation comments are what the rules read off nodes), and
    board_render.parse_sgf for the setup,
    because it yields SGF coordinates in the form `simulate` consumes."""
    size, tid = args
    path = os.path.join(PROJECT_ROOT, "tsumegos", str(size), f"{tid}.sgf")
    with open(path, "rb") as f:
        raw = f.read()
    game = sgfmill.sgf.Sgf_game.from_bytes(raw)
    _, variation_nodes = parse_tsumego_sgf(game)
    _, setup_sgf, _ = parse_sgf(raw.decode("utf-8", "replace"))
    evals = [e for e in (_mover_eval(n) for n in variation_nodes) if e is not None]
    return tid, evals, rule_ko(setup_sgf, variation_nodes, size)


def build(pool, ownership, phase, workers):
    from multiprocessing import Pool as ProcPool

    tids = pool["tids"]
    sizes = pool["sizes"]

    jobs = list(zip(sizes, tids))
    facts = {}
    with ProcPool(workers) as p:
        for i, rec in enumerate(p.imap_unordered(_sgf_facts, jobs, chunksize=50), 1):
            facts[rec[0]] = rec[1:]
            if i % 1000 == 0:
                print(f"  parsed {i}/{len(jobs)}", flush=True)

    ann, counts = {}, Counter()
    skipped = []
    for tid, size in zip(tids, sizes):
        own = ownership.get(tid)
        ph = phase.get(tid)
        if own is None or ph is None or tid not in facts:
            skipped.append(tid)
            continue
        evals, has_ko = facts[tid]
        hits = []
        if rule_life_and_death(own):
            hits.append("life and death")
        if rule_semeai(own):
            hits.append("semeai")
        if rule_counting(evals):
            hits.append("counting")
        if rule_territory(own):
            hits.append("territory")
        if rule_shape(evals):
            hits.append("shape")
        if has_ko:
            hits.append("ko")
        if rule_opening(ph[1], size):
            hits.append("opening")
        for h in hits:
            counts[h] += 1
        # `known` is the FULL label set: a rule evaluates every problem, so every
        # label is decided here, positives and negatives alike. That is the part
        # hand labelling cannot supply cheaply, and it is what lets a probe fit at
        # all — without negatives there is nothing to separate from.
        ann[tid] = {"labels": hits, "known": list(LABEL_SET), "ts": 0.0}
    return ann, counts, skipped


def load_ownership():
    keep = ("own_dead_revived", "own_alive_killed",
            "opp_alive_killed", "opp_dead_revived", "empty_swing_abs",
            # the territory rule's inputs
            "score_correct", "score_alt", "stone_swing_abs")
    out = {}
    with open(OWNERSHIP_CSV, newline="") as fh:
        for rec in csv.DictReader(fh):
            if not rec.get("alt_move"):
                continue
            out[rec["tid"]] = {k: rec[k] for k in keep}
    return out


def load_phase(sizes):
    out = {}
    for size in sizes:
        with open(os.path.join(SIGNAL_CACHE, f"game_phase_{size}.json")) as fh:
            out.update(json.load(fh))
    return out


def report(ann, pool):
    """How separable is each rule in the MIRT feature space the probe reads?

    Stratified 4-fold CV AUC of the same elastic-net logistic the server fits. AUC
    rather than accuracy because the rules are unbalanced, and because what the
    bank consumes is a RANKING over problems, not a decision."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    F = np.load(os.path.join(CACHE_DIR, "features.npy")).astype(np.float64)
    F = (F - F.mean(0)) / np.maximum(F.std(0), 1e-6)   # the server's own scaling
    row = {t: i for i, t in enumerate(pool["tids"])}

    print(f"\n{'label':<18} {'pos':>6} {'prevalence':>11} {'CV AUC':>8}")
    print("-" * 47)
    for label in LABEL_SET:
        idx = [row[t] for t in ann if t in row]
        y = np.array([1 if label in ann[t]["labels"] else 0
                      for t in ann if t in row])
        pos = int(y.sum())
        if pos < 10 or pos == len(y):
            print(f"{label:<18} {pos:>6} {pos/len(y):>10.1%} {'--':>8}  (too few to score)")
            continue
        X = F[idx]
        aucs = []
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=0).split(X, y):
            clf = LogisticRegression(penalty="elasticnet", solver="saga",
                                     l1_ratio=0.5, C=0.3, max_iter=3000, tol=1e-3)
            clf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
        print(f"{label:<18} {pos:>6} {pos/len(y):>10.1%} {np.mean(aucs):>8.3f}"
              f"   (folds {', '.join(f'{a:.3f}' for a in aucs)})")


def fit_and_cache():
    """Fit the probes on the labels just written and persist them, so the server
    serves predictions from its first request.

    Doing it here rather than leaving it to the server is the difference between
    "the labeler is ready" and "the labeler is ready in two minutes": every label
    a rule writes clears CV_MIN_CLASS, so all of them take the CV branch at once,
    and until that finishes the grid can only offer random positions.
    """
    import time
    from auxiliary.labeler import server

    print("\nfitting probes so the server starts with a model...", flush=True)
    t0 = time.time()
    server.load_cache()
    server.load_store()
    model = server.train()
    server.save_model(model)
    modes = Counter(m for m in model["mode"].values() if m)
    print(f"  model v{model['version']} cached in {time.time()-t0:.0f}s "
          f"({dict(modes)}) -> {os.path.relpath(server.MODEL_NPZ, PROJECT_ROOT)}")
    print("  the labeler is ready: ./.venv/bin/python3 -m auxiliary.labeler.server")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="also score each rule's separability in feature space")
    ap.add_argument("--no-train", action="store_true",
                    help="skip fitting/caching the probes (the server will fit on boot)")
    ap.add_argument("--corpus", default=None, metavar="OUT.json",
                    help="label the WHOLE corpus instead of the pool and write "
                         "there. This is ground truth for scoring a fitted axis's "
                         "tail — the rules are pure functions of corpus "
                         "artifacts, so they answer for problems no labeller ever "
                         "saw. Deliberately NOT labels.json: precompute_features "
                         "pins every annotated problem into the pool, so writing "
                         "160k annotations there would pin the entire corpus and "
                         "the pool could never be resampled.")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=LABELS_PATH)
    args = ap.parse_args()

    if args.corpus:
        # The corpus stands in for the pool: same rules, same code path, every
        # problem instead of a sample. The opening cutoff is still a per-size
        # percentile, now over the whole board rather than over the sample.
        tids, sizes = [], []
        for size in sorted(SIZES):
            d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
            for f in sorted(os.listdir(d)):
                if f.endswith(".sgf"):
                    tids.append(f[:-4])
                    sizes.append(size)
        pool = {"tids": tids, "sizes": sizes}
    else:
        with open(os.path.join(CACHE_DIR, "pool.json")) as fh:
            pool = json.load(fh)
    print(f"{'corpus' if args.corpus else 'pool'}: {len(pool['tids'])} problems, "
          f"{dict(Counter(pool['sizes']))}")

    ownership = load_ownership()
    phase = load_phase(sorted(set(pool["sizes"])))
    print(f"  ownership rows {len(ownership)}, game-phase rows {len(phase)}")

    ann, counts, skipped = build(pool, ownership, phase, args.workers)
    n = len(ann)
    print(f"\nlabelled {n} pool problems ({len(skipped)} skipped for missing inputs)")
    for label in LABEL_SET:
        print(f"  {label:<18} {counts[label]:>5}  ({counts[label]/max(n,1):6.2%})")
    overlap = Counter(len(r["labels"]) for r in ann.values())
    print(f"  labels per problem: {dict(sorted(overlap.items()))}")

    out = args.corpus or args.out
    with open(out, "w") as fh:
        json.dump({"label_set": LABEL_SET, "annotations": ann}, fh, indent=2)
    print(f"\n-> {out}")

    # Both of these read the POOL's feature cache, which a corpus run has no
    # counterpart for and does not need: it produces ground truth, not a probe.
    if args.corpus:
        return
    if args.report:
        report(ann, pool)
    if not args.no_train:
        fit_and_cache()


if __name__ == "__main__":
    main()
