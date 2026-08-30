#!/usr/bin/env python3
"""Label every problem with the motif PROXY rules — the directions train_mirt teaches onto.

    ./.venv/bin/python3 -m pipeline.build_motif_proxies          # incremental
    ./.venv/bin/python3 -m pipeline.build_motif_proxies --force  # rebuild from scratch

WHY PRODUCTION SEEDS FROM RULES AND NOT FROM PREVIOUSLY-FITTED DIRECTIONS. The axis
space is defined only up to rotation, so a free fit lands on a different basis every
seed — measured, two seeds of one configuration agree on their directions at 0.34.
Teaching from a bank of previously-fitted directions was the answer to that, and it
half-works: the directions come back at 0.43-0.73 agreement, but INSPECTION of those
same axes found four of six poles mutated or unnameable at exactly those figures. A
name held numerically and lost its meaning.

A rule is a fixed, re-readable STARTING POINT instead. It states a criterion over
the board and the engine's own evaluation, so every refit seeds from the same
reference. It does NOT guarantee the fitted axis means the rule — the axis is
whatever the model lands on when pulled toward it and released, and the hope is that
it lands close enough to represent it; the fit's identity report is what measures
that rather than assuming it. What decides against the alternative is that the round
trip buys nothing: teaching from a direction an earlier fit produced — itself only an
approximation of the rule behind it — stabilises an axis no better than teaching from
that rule directly, so it adds a moving reference for no gain. The supporting evidence
points the same way: of the motifs that reproduced across two entirely different model
generations, both — `opening` and `counting` — are rule-shaped.

So the axis bank is a DEV tool now (auxiliary/rating_dev/build_bank_from_dumps.py,
the motif store in axes/label_anchors.npz), used to explore what a fit finds. What
production teaches is this file's output.

INCREMENTAL BY TID, and reconciling in BOTH directions. New problems are labelled and
added; problems that have left the corpus are pruned. "Up to date" means the stored
tid set EQUALS the corpus, never just "nothing new to add" — a stale entry is a
direction seeded on a problem that ships nowhere, and a missing one silently drops a
problem out of every proxy column. `--force` re-labels everything, which is needed
after a rule or threshold changes, since nothing in a tid can reveal that.

Writes `axes/motif_proxies.npz`:
    tid    (n,)          problem ids, sorted
    size   (n,)          9 or 13
    labels (n, n_rules)  bool, one column per rule in LABEL_SET order
    rules  (n_rules,)    the rule names, so a reader never assumes the order

EVERY RULE IS A THRESHOLD ON THE PROBLEM'S OWN NUMBERS, which is what makes an
incremental labelling sound: a stored row can never be made wrong by a problem
added later. `opening` used to be a percentile of the corpus and therefore moved
as the corpus grew, silently re-labelling problems that had not changed; it is now
an absolute prefix length per board. If a rule is ever added that depends on the
population, incrementality has to be revisited with it.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf  # noqa: E402

from utils.katago.board_render import parse_sgf  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf  # noqa: E402
from utils.rating.model_constants import SIZES  # noqa: E402
from utils.rating.motif_rules import (  # noqa: E402
    LABEL_SET, OWNERSHIP_CSV, _mover_eval, _subtree_is_resolved,
    answer_opponent_distance, best_refuted_node, reply_distance,
    reply_kind, rule_fast, rule_gote, rule_ko,
    rule_life_and_death, rule_opening, rule_sacrifice, rule_yose, rule_semeai,
    rule_sente, rule_shape, rule_tenuki, rule_territory)

OUT = os.path.join(PROJECT_ROOT, "axes", "motif_proxies.npz")
# The ownership columns the rules read. A projection, not the whole row — the CSV
# carries thirty-odd fields and the rules use eight.
OWN_COLS = ("own_dead_revived", "own_alive_killed", "opp_alive_killed",
            "opp_dead_revived", "empty_swing_abs",
            "score_correct", "score_alt", "stone_swing_abs")

_OWN = {}
_PHASE = {}


def _init(own, phase):
    """Worker-side globals: the ownership and phase tables are ~160k rows each and
    pickling them per task would dominate the run."""
    global _OWN, _PHASE
    _OWN, _PHASE = own, phase


def label_one(args):
    """Every rule's verdict for one problem, or None if its inputs are missing."""
    size, tid = args
    own = _OWN.get(tid)
    ph = _PHASE.get(tid)
    if own is None or ph is None:
        return None
    path = os.path.join(PROJECT_ROOT, "tsumegos", str(size), f"{tid}.sgf")
    try:
        raw = open(path, "rb").read()
        game = sgfmill.sgf.Sgf_game.from_bytes(raw)
        _, variation_nodes = parse_tsumego_sgf(game)
        _, setup_sgf, correct = parse_sgf(raw.decode("utf-8", "replace"))
    except Exception:
        return None
    # The answer NODE, matched by coordinate to the winrate-chosen correct move
    # rather than taken as variation_nodes[0]. The two agree by construction
    # (the answer is written first and generation discards any file whose
    # evaluations name another move) but matching costs nothing and does not
    # depend on that invariant holding for every file.
    answer_node = None
    if correct:
        for _n in variation_nodes:
            _c, _raw = _n.get_raw_move()
            if _raw and _raw.decode("utf-8") == correct[1]:
                answer_node = _n
                break
    # Both distances ride THIS parse -- the geometry rules need no board replay
    # and no second pass over the corpus.
    opp_dist = answer_opponent_distance(setup_sgf, correct)
    # The SAME measurement twice — how close the opponent replies to the move
    # just made — asked of the answer and of the best refuted try. rule_sente
    # wants the first to be a local answer and the second a tenuki.
    reply_dist = reply_distance(answer_node)
    refuted_dist = reply_distance(best_refuted_node(variation_nodes, correct))
    # Whether a ko stands in the ROOT position, computed once: rule_ko needs it
    # as its first test and rule_tenuki needs it as a disqualifier.
    from utils.katago.board_render import simulate
    root_board, _last, _ko = simulate(setup_sgf, size)
    root_ko = _ko is not None
    evals = [e for e in (_mover_eval(n) for n in variation_nodes) if e is not None]
    # `shape` ranks only RESOLVED alternatives: an abandoned line is not a rival,
    # it is a line nobody played out.
    resolved = [e for e in (_mover_eval(n) for n in variation_nodes
                            if _subtree_is_resolved(n)) if e is not None]
    hit = {
        "life and death": rule_life_and_death(own),
        # The complement of `life and death` over the same two ownership parts —
        # both statuses decided together rather than one of them — and disjoint
        # from it by construction, so no problem seeds both.
        "semeai": rule_semeai(own),
        "yose": rule_yose(evals),
        "ko": True if root_ko else rule_ko(setup_sgf, variation_nodes, size),
        "opening": rule_opening(ph[1], size),
        "territory": rule_territory(own),
        "shape": rule_shape(resolved),
        "tenuki": rule_tenuki(opp_dist, root_ko),
        "sente": rule_sente(reply_dist, refuted_dist),
        # The mirror contrast wants the answer's reply KIND as well: a pass in
        # reply to the answer is gote's strongest form, and reply_distance's
        # None cannot say it happened.
        "gote": rule_gote(reply_kind(answer_node)[0], reply_dist, refuted_dist),
        "fast": rule_fast(root_board, correct),
        "sacrifice": rule_sacrifice(own),
    }
    return tid, size, [bool(hit[k]) for k in LABEL_SET]


def load_ownership():
    out = {}
    with open(OWNERSHIP_CSV, newline="") as fh:
        for rec in csv.DictReader(fh):
            if rec.get("alt_move"):
                out[rec["tid"]] = {k: rec[k] for k in OWN_COLS}
    return out


def load_phase(sizes):
    """The game-phase table, through the loader that OWNS and heals that cache.

    `utils.rating.features.load_game_phase` is incremental in both directions —
    it parses the corpus tids the cache lacks and prunes departed ones — whereas
    a plain read of game_phase_<size>.json returns whatever was last written. A
    plain read is what silently costs a corpus top-up its labels: the new
    problems have no phase row, `label_one` returns None for every one of them,
    and the proxies ship covering only the problems that happened to be in the
    corpus at some earlier run. Reading through the healer is what makes "the
    stored tid set EQUALS the corpus" true rather than aspirational."""
    from utils.rating.features import load_game_phase
    return load_game_phase(sizes=sizes)


def corpus():
    out = []
    for size in SIZES:
        d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        out += [(size, f[:-4]) for f in os.listdir(d) if f.endswith(".sgf")]
    return sorted(out, key=lambda t: t[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="re-label every problem. REQUIRED after a rule or a "
                         "threshold changes: incremental keys on the tid, and "
                         "nothing in a tid reveals that the rule moved under it.")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    t0 = time.time()
    have = corpus()
    want = {t for _, t in have}
    print(f"corpus: {len(have)} problems", flush=True)

    prev = {}
    if os.path.exists(OUT) and not args.force:
        z = np.load(OUT, allow_pickle=False)
        if [str(x) for x in z["rules"]] != LABEL_SET:
            raise SystemExit(
                f"{OUT} was built for rules {[str(x) for x in z['rules']]}, this "
                f"run has {LABEL_SET}. The rule SET changed, so every stored row "
                f"is a different question — re-run with --force.")
        prev = {str(t): (int(s), row) for t, s, row in
                zip(z["tid"], z["size"], z["labels"])}
        gone = len(set(prev) - want)
        if gone:
            print(f"  pruning {gone} problems no longer in the corpus", flush=True)
            prev = {t: v for t, v in prev.items() if t in want}

    todo = [(s, t) for s, t in have if t not in prev]
    print(f"  {len(prev)} carried over, {len(todo)} to label", flush=True)

    rows = dict(prev)
    if todo:
        own, phase = load_ownership(), load_phase(sorted(SIZES))
        print(f"  ownership {len(own)}, phase {len(phase)}", flush=True)
        from multiprocessing import Pool
        skipped = 0
        with Pool(args.workers, initializer=_init,
                  initargs=(own, phase)) as p:
            for i, r in enumerate(p.imap_unordered(label_one, todo, chunksize=200)):
                if r is None:
                    skipped += 1
                else:
                    tid, size, lab = r
                    rows[tid] = (size, np.array(lab, bool))
                if (i + 1) % 20000 == 0:
                    print(f"    {i+1}/{len(todo)}", flush=True)
        if skipped:
            # ABORT rather than write a short file. These labels are the
            # directions train_mirt teaches its axes onto, and a problem with no
            # row is simply never taught -- which does not fail anywhere, it just
            # quietly shrinks the taught set and leaves the axes shaped by
            # whatever subset happened to have inputs. Every live problem is
            # supposed to have an ownership row (the sweep emits one even when
            # n_te_alts = 0) and a parseable SGF, so a skip means an upstream
            # block is behind the corpus, and naming it here is far cheaper than
            # inferring it later from an axis that came out wrong.
            raise SystemExit(
                f"\n{skipped} of {len(todo)} problems have MISSING INPUTS and "
                f"were not labelled. The proxies are only correct when the "
                f"stored tid set EQUALS the corpus, so this run wrote nothing. "
                f"Ownership rows: {len(own)}, phase rows: {len(phase)}, corpus: "
                f"{len(todo)}. Bring the short one up to date "
                f"(`pipeline.extract_ownership_swing` for ownership; phase heals "
                f"itself through utils.rating.features.load_game_phase) and "
                f"re-run.")

    tid = np.array(sorted(rows))
    size = np.array([rows[t][0] for t in tid], np.int16)
    labels = np.stack([rows[t][1] for t in tid]).astype(bool)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, tid=tid, size=size, labels=labels,
                        rules=np.array(LABEL_SET))
    print(f"\n  {'rule':<18}{'problems':>10}{'share':>9}")
    for j, name in enumerate(LABEL_SET):
        n = int(labels[:, j].sum())
        print(f"  {name:<18}{n:>10}{n/len(tid):>9.2%}")
    per = np.bincount(labels.sum(1))
    print(f"  labels per problem: {dict(enumerate(per.tolist()))}")
    print(f"-> {OUT}  ({len(tid)} problems, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
