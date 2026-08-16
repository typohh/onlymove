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
LD_FOCUS = ("own_dead_revived", "opp_alive_killed")   # fail-to-live, fail-to-kill
LD_OTHER = ("own_alive_killed", "opp_dead_revived")   # sacrifice, let-out
LD_MIN_STONES = 5.0    # the winning part must move at least this many stones
# Trades are judged RELATIVE to the outcome, not against a fixed stone count. A
# 30-stone group living or dying is not a trade because a quarter-stone drifted
# somewhere else, and an absolute cap says it is — it gets stricter the larger
# the problem, which is backwards. What "no trade" means is that nothing else
# comes close to the thing being decided.
LD_TRADE_RATIO = 0.20  # every OTHER directional part, as a fraction of the winner

# counting: the top two first moves finish within a hair of each other on score
# while the win probability separates hard — the position turns on a margin you
# have to count rather than on a status you can see.
COUNT_SCORE_MAX = 1.0   # points between best and second best
COUNT_WR_MIN = 50.0     # percentage points of winrate between the same two

# ko: replay each resolved line this many plies and ask whether a ko shape stands.
# Depth 0 is the root position itself.
KO_DEPTH = 3

# opening: few moves led to the position, per board size.
OPENING_PERCENTILE = 5.0

LABEL_SET = ["life and death", "counting", "ko", "opening"]


# ---------------------------------------------------------------- helpers

def _subtree_has_te(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.has_property("TE"):
            return True
        stack.extend(list(n))
    return False


def _mover_eval(node):
    """(winrate, score) in the MOVER's frame, or None. Comments are stored from
    BLACK's perspective, so a white mover flips both."""
    colour, _ = node.get_raw_move()
    if colour is None:
        return None
    c = node.get("C") if node.has_property("C") else None
    parsed = parse_eval_comment(c)
    if parsed is None:
        return None
    wr_black, score_black = parsed
    return ((wr_black, score_black) if colour == "b"
            else (100.0 - wr_black, -score_black))


def _line_moves(node, depth):
    """The first `depth` plies at and under `node`, as (COLOR, sgf_coord) pairs in
    board_render's setup format. Follows the first child at each step — the
    generator writes the line it read, so the principal continuation is the line."""
    out = []
    cur = node
    while cur is not None and len(out) < depth:
        colour, raw = cur.get_raw_move()
        if colour is not None and raw and raw != b"tt":
            out.append((colour.upper(), raw.decode("utf-8")))
        cur = cur[0] if len(cur) else None
    return out


# ---------------------------------------------------------------- the rules

def rule_life_and_death(own, min_stones=None, trade_ratio=None):
    """ONE status change, big enough to be the point of the problem, and nothing
    else close to it.

    `own` is the raw ownership row (stone counts, NOT the z-scored feature
    columns — a threshold in stones is only a threshold in stones before the
    pooled standardization). The dominant directional part must be a fail-to-live
    or a fail-to-kill; the other three are judged as a FRACTION of it, so a
    problem where you save one group by giving up another is excluded while
    incidental drift beside a large group is not.
    """
    min_stones = LD_MIN_STONES if min_stones is None else min_stones
    trade_ratio = LD_TRADE_RATIO if trade_ratio is None else trade_ratio
    parts = {k: abs(float(own[k])) for k in (*LD_FOCUS, *LD_OTHER)}
    top = max(parts, key=parts.get)
    if top not in LD_FOCUS or parts[top] < min_stones:
        return False
    bar = trade_ratio * parts[top]
    return all(v <= bar for k, v in parts.items() if k != top)


def rule_counting(evals):
    """Score says the two best first moves are equivalent; winrate says they are
    not. That gap is the signature of a position decided by a margin.

    `evals` is [(winrate, score)] over the root variations, mover-framed. Ranked
    by WINRATE, because that is what names the answer (findBestMoveFromChildren),
    so "best and second best" is the answer and its closest rival."""
    if len(evals) < 2:
        return False
    top = sorted(evals, key=lambda t: (t[0], t[1]), reverse=True)[:2]
    (w1, s1), (w2, s2) = top
    return (s1 - s2) <= COUNT_SCORE_MAX and (w1 - w2) >= COUNT_WR_MIN


def rule_ko(base, variation_nodes, size):
    """A ko shape stands within KO_DEPTH plies of a line the generator resolved.

    Checked at the ROOT first, then along each TE-marked root variation — the
    root case is the position that already sits in a ko, the line case is the
    position that becomes one once the fight starts. Only resolved branches are
    walked, for the same reason the ownership baseline only uses them: an
    abandoned stub is a line nobody played out.

    `base` is the setup in SGF COORDINATES ("pd"), which is what `simulate`
    parses — NOT the GTP form ("D4") that `parse_tsumego_sgf` hands back. Feeding
    it GTP fails silently and completely: every stone misses the range check, the
    board stays empty, and no position ever holds a ko."""
    if simulate(base, size)[2] is not None:
        return True
    if KO_DEPTH <= 0:
        return False
    for node in variation_nodes:
        if not _subtree_has_te(node):
            continue
        moves = _line_moves(node, KO_DEPTH)
        for i in range(1, len(moves) + 1):
            if simulate(base + moves[:i], size)[2] is not None:
                return True
    return False


def rule_opening(prefix, cutoff):
    """Few moves led to the contested position."""
    return prefix <= cutoff


# ---------------------------------------------------------------- driver

def _sgf_facts(args):
    """Everything the SGF-side rules need, from ONE read of the file.

    Two parsers on purpose: sgfmill for the variation TREE (the TE marks and the
    evaluation comments live on nodes), and board_render.parse_sgf for the setup,
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

    # opening cutoff per board size, over the POOL (the population the labels
    # describe), so "bottom 5%" means bottom 5% of what a labeller will be shown.
    cutoff = {}
    for size in sorted(set(sizes)):
        pref = [phase[t][1] for t, s in zip(tids, sizes)
                if s == size and t in phase]
        cutoff[size] = float(np.percentile(pref, OPENING_PERCENTILE))
        print(f"  opening cutoff, size {size}: prefix <= {cutoff[size]:.0f} moves "
              f"({OPENING_PERCENTILE}th pct of {len(pref)} pool problems)")

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
        if rule_counting(evals):
            hits.append("counting")
        if has_ko:
            hits.append("ko")
        if rule_opening(ph[1], cutoff[size]):
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
            "opp_alive_killed", "opp_dead_revived", "empty_swing_abs")
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
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=LABELS_PATH)
    args = ap.parse_args()

    with open(os.path.join(CACHE_DIR, "pool.json")) as fh:
        pool = json.load(fh)
    print(f"pool: {len(pool['tids'])} problems, "
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

    with open(args.out, "w") as fh:
        json.dump({"label_set": LABEL_SET, "annotations": ann}, fh, indent=2)
    print(f"\n-> {args.out}")

    if args.report:
        report(ann, pool)

    if not args.no_train:
        fit_and_cache()


if __name__ == "__main__":
    main()
