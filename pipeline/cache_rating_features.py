"""Cache candidate-move evaluation features (the winning move + its 9 best alternatives) for
the rating model.

Every tsumego SGF embeds, on each candidate first-move node, a KataGo evaluation comment
"<winrate>%/<score>/<visits>" (Black-perspective win rate, Black-minus-White score lead, and the
search depth behind them; only the first two are read here) — see the
"PWA SGF format carries embedded analysis" note in CLAUDE.md. The generator evaluates (almost)
every legal first move, so a problem carries dozens of these. This script reads them for every
problem, orients each to the SIDE TO MOVE (so "win prob" means "the solver's win prob" on either
colour), sorts the moves by win probability, and keeps the WINNING move plus the 9 next-best
alternatives. For each of those 10 moves it stores (logit win-prob, clipped score): a fixed
20-dim raw feature vector per problem describing how sharply the best move stands out from its
rivals. This block is OUTSIDE the shipped trunk layout (model_constants.PRODUCTION_BLOCKS): it
describes the problem through its ANSWER, which is what the trunk cannot carry once it is fanned
out over every legal cell. Two motif rules turn on the same quantity, but they read the SGF's own
evaluation comments directly, so pipeline.build_motif_proxies does not depend on this cache.

It ALSO reads the CONTINUATION — the principal line after the answer, plus the opponent's other
tries at its first node, every node's explored children kept as (logit win-prob, score) oriented
to the SOLVER. That is where the board's remaining branching lives: how many continuations still
hold the win, how far the losing ones fall, and how hard each of the opponent's tries is to meet.
The root cannot answer any of it, because the generator's single-solution gate leaves exactly one
winning first move with every rival decisively lost (measured: the 5th percentile of every
alternative slot sits AT the win-prob clip), so the block above is saturated by construction on
its alternatives.

Output (one per board size), read through utils.rating.features.load_joint_move_features +
load_joint_continuation (the "all" comparison layout), load_tightness_gaps (the axis analyses)
and build_alt_vec_cache (which takes its runner-up cell from `alt_xy`):
  activations/_signal_cache/move_evals_<size>.npz
      tids   (n,)      <U8   canonical problem id (orientation-invariant hash)
      feats  (n, 20)   f32   RAW [logit win-prob, score] × 10 moves (solution, then 9 alts)
      cont   (n, 75)   f32   RAW [logit win-prob, score], solver-oriented: the principal line
                             (CONT_PLIES × CONT_MOVES), then the opponent's other tries
                             (CONT_TRIES × CONT_ANSWERS), then the per-node explored-child
                             counts (CONT_COUNTS)

Both blocks come out of ONE parse per SGF (the parse dominates this step), and a row counts as
cached only when both are present, so the two can never drift apart.

The npz stores RAW features; the loader pools + z-scores them across boards at train time, so
the cache is reusable across tid subsets. The tid set is the activation id cache for the size
(activations/_signal_cache/correct_ids_<size>.json) — exactly the problems train_mirt can
score.

This step ALSO refreshes that activation cache (utils.rating.features.build_correct_vec_cache),
because it is the other roll-up of extract_activations' per-problem .npz output and this step's
own tid set comes from it. So one command brings both aggregate caches forward after a tsumego
top-up; neither can be left stale behind the other. Run once per size (both cached by default):

  ./.venv/bin/python3 -m pipeline.cache_rating_features [--sizes 9,13] [--workers N] [--force]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import sgfmill.sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import (  # noqa: E402
    CACHE_DIR, _cache_ids_path, build_correct_vec_cache,
)
from utils.katago.coords import sgf_to_gtp  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf  # noqa: E402
from utils.katago.coords import gtp_to_xy  # noqa: E402

# Feature layout: the winning move + N_ALT best alternatives, each as (logit win-prob, score).
N_ALT = 9
N_MOVES = 1 + N_ALT
N_FEATS = 2 * N_MOVES                 # = 20
WP_CLIP = 1e-3                        # win-prob clip before the logit (KataGo emits exact 0/100%)
SCORE_CLIP = 50.0                     # tame the rare blow-out leads before pooled z-scoring
CHUNK = 500                           # SGFs per worker task (amortizes pickling over the parse)
# "100.0%/34.1/1000" / "44.6%/-0.3/30" — Black-perspective win rate %, then Black−White score
# lead, then the visits behind the estimate. Unanchored and reading the first two fields only:
# the trailing "/<visits>" falls outside the match, and a pre-field comment still reads.
CMT_RE = re.compile(r"(\d+\.?\d*)%/([-]?\d+\.?\d*)")

# Continuation layout: the principal line AFTER the answer, CONT_PLIES deep, each ply's
# explored children kept CONT_MOVES wide as (logit win-prob, score). Same raw treatment as
# the block above — the evaluations are handed over as they stand, not reduced to derived
# scalars, so the net decides what to make of them.
CONT_PLIES = 6
CONT_MOVES = 4
# ...plus the opponent's OTHER tries at its first node, each with the solver's best answers.
# Reading a tsumego means answering every try the opponent has, not just its best one, and the
# principal line walks into that best try alone — so on its own the walk sees how many tries the
# opponent has but nothing about how hard each is to meet. Try 0 IS the principal line and is
# already covered at ply 2, so these start at try 1. Measured: every try the generator records
# has its answers explored (100%), ~3-4 of them, and tries 1/2/3 exist in 92/77/56% of 9x9 and
# 94/85/72% of 13x13 problems.
CONT_TRIES = 3
CONT_ANSWERS = 3
# ...plus one column per node/try-group holding how many children the generator actually explored.
# A node's children are the branches the search read to FREE_BRANCH_VISITS, so a MISSING move
# is one the search declined to spend visits on — evidence that the best move dominates, not that
# the move is unknown. Measured on the principal line: the best-vs-second gap shrinks monotonically
# as children multiply (9x9 solver nodes 2.87 -> 0.89 mover-logit from 2 to 7 children; opponent
# nodes 1.08 -> 0.40), so at one child the true gap is WIDER than any observed. Two consequences,
# and both are needed — the count alone still leaves a wrong number in the slot, and the pad alone
# leaves "explored and losing" indistinguishable from "never explored":
#   - a missing slot pads at the extreme that DISFAVOURS THE MOVER, never by repeating the last
#     real child (that asserts a gap of zero, the maximum error in the wrong direction);
#   - the count ships as its own column, so the net can tell a pad from a measurement.
CONT_COUNTS = CONT_PLIES + CONT_TRIES
N_CONT = 2 * (CONT_PLIES * CONT_MOVES + CONT_TRIES * CONT_ANSWERS) + CONT_COUNTS  # = 75

# Every array in the cache is one row per tid, and a row counts as cached only when ALL of them
# are present — so adding an array here re-extracts the corpus rather than leaving half-filled
# rows that nothing downstream can detect.
CACHE_ARRAYS = ("feats", "cont", "alt_xy")


def _cache_path(size):
    return os.path.join(CACHE_DIR, f"move_evals_{size}.npz")


def _logit(p):
    p = min(max(float(p), WP_CLIP), 1.0 - WP_CLIP)
    return float(np.log(p / (1.0 - p)))


def _node_eval(node):
    """(win-prob, score) from a node's evaluation comment, BLACK's perspective, or None.

    A comment-less move node is normal (not every node carries an eval), so callers skip
    just that node. get_raw_list("C") would raise KeyError on such a node, so the
    has_property gate is expected-data handling, NOT error tolerance: a genuinely
    unreadable SGF still propagates out of _process_chunk and aborts.
    """
    if not node.has_property("C"):
        return None
    for cb in node.get_raw_list("C"):
        m = CMT_RE.search(cb.decode("utf-8"))
        if m:
            return float(m.group(1)) / 100.0, float(m.group(2))
    return None


def _evaluated_children_of(nodes):
    """The move-carrying nodes of an iterable — so the root walk and the continuation walk
    agree on what counts as a move node."""
    return [c for c in nodes if c.has_property("B") or c.has_property("W")]


def extract_move_features(game):
    """The 20-dim raw feature vector for one SGF, or None if it has no usable evaluations.

    All candidate first-move nodes share the side to move; each comment is read and oriented to
    that side (win prob and score from the solver's perspective). Moves are sorted best→worst by
    (win-prob, score) so slot 0 is the best-evaluated move — which is the answer, though the
    answer is CONTRACTUALLY variation 0 and this sort is a feature layout, not a lookup —
    and slots 1..9 are the
    best alternatives. The rare problem with <10 evaluated moves forward-fills its last (worst)
    move into the empty slots (gap-to-a-missing-alternative = 0)."""
    _setup, vnodes = parse_tsumego_sgf(game)
    moves = []                                       # (win-prob_stm, score_stm)
    for node in _evaluated_children_of(vnodes):
        color, _mb = node.get_raw_move()
        black_to_move = color == "b"
        e = _node_eval(node)
        if e is None:
            continue
        wr, sc = e
        moves.append((wr, sc) if black_to_move else (1.0 - wr, -sc))
    if not moves:
        return None
    moves.sort(key=lambda t: (t[0], t[1]), reverse=True)
    feat = np.empty(N_FEATS, dtype=np.float32)
    last = moves[-1]
    for i in range(N_MOVES):
        wp, sc = moves[i] if i < len(moves) else last
        feat[2 * i] = _logit(wp)
        feat[2 * i + 1] = float(np.clip(sc, -SCORE_CLIP, SCORE_CLIP))
    return feat


def _ranked_children(node, solver_black):
    """A node's evaluated children as [(solver_wp, solver_score, child)], ordered by what the
    MOVER would pick — highest win prob for the mover, score breaking ties (also how slot 0
    of the block above is chosen).

    Ordering by the mover while recording for the solver is the whole split: slot 0 has to be
    the move that actually gets played, or the line walked here is not the line the solver
    would face, while the VALUE has to be solver-oriented or the opponent's nodes sit at the
    win-prob clip on every slot (past the answer the opponent is lost in all lines) — the
    degenerate shape the root block's alternative slots already have.
    """
    scored = []
    for c in _evaluated_children_of(node):
        e = _node_eval(c)
        if e is None:
            continue
        wr, sc = e                                   # Black's perspective
        mover_black = c.get_raw_move()[0] == "b"
        mover = (wr, sc) if mover_black else (1.0 - wr, -sc)
        solver = (wr, sc) if solver_black else (1.0 - wr, -sc)
        scored.append((mover, solver, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [(s[0], s[1], c) for _m, s, c in scored]


def _slots(ranked, width, pad, solver_to_move):
    """`width` (logit win-prob, score) pairs from ranked children, padding the empty slots at
    the extreme that DISFAVOURS THE MOVER — solver to move: the solver loses; opponent to move:
    the opponent's try fails and the solver wins.

    A move absent from the SGF is one the search never read to FREE_BRANCH_VISITS, i.e. one
    it judged not worth pursuing, so its absence says the moves present dominate it. The
    rejected alternative is forward-filling the worst real child (the convention the root block
    uses, where every legal move IS evaluated and nothing is missing): here it would claim the
    unexplored moves are as good as the worst explored one — a gap of zero, in the one direction
    the evidence rules out.

    `pad` is used only when the node has NO explored children at all, which is a different
    statement: the line ended, so the position stands at that value with nothing left to read.
    The count column tells the two cases apart."""
    vals = [(_logit(wp), float(np.clip(sc, -SCORE_CLIP, SCORE_CLIP))) for wp, sc, _c in ranked]
    if not vals:
        return [pad] * width
    # Score has no natural extreme to pad with, so it holds at the worst explored child's value;
    # the win-prob column carries the direction.
    dead = (_logit(WP_CLIP) if solver_to_move else _logit(1.0 - WP_CLIP), vals[-1][1])
    return [vals[i] if i < len(vals) else dead for i in range(width)]


def extract_continuation_features(game):
    """The 75-dim raw continuation vector for one SGF, or None if it has no answer move.

    Three parts, all deliberately raw exactly like the block above — the counting is left to the
    net:

    1. THE PRINCIPAL LINE (CONT_PLIES x CONT_MOVES): walking on from the answer, each node's
       explored children as (logit win-prob, score) oriented to the SOLVER. The two node kinds
       carry the two halves of the question. On the solver's own nodes the slots FALL away
       from slot 0, so the fall-off is how many continuations still hold the win and how
       sharply the losing ones are punished. On the opponent's nodes they RISE from slot 0, so
       the rise is how many tries the opponent genuinely has.

    2. THE OPPONENT'S OTHER TRIES (CONT_TRIES x CONT_ANSWERS): for each of the opponent's
       non-principal replies to the answer, the solver's best answers under it. Part 1 walks
       into the opponent's BEST try alone, so by itself it shows how many tries the opponent
       has but nothing about how hard each one is to meet — and meeting all of them is what
       reading the problem consists of. Try 0 is the principal line, already covered at ply 2,
       so these start at try 1.

    3. THE EXPLORED-CHILD COUNTS (CONT_COUNTS): how many children each of those nodes actually
       had, so a padded slot is never read as a measured one. See the CONT_COUNTS note for why
       missingness here is informative rather than unknown.

    Orientation is to the SOLVER while ORDERING is by the mover: slot 0 has to be the move that
    actually gets played, or the line walked is not the line the solver would face, while the
    value has to be solver-oriented or the opponent's nodes sit at the win-prob clip on every
    slot (past the answer the opponent is lost in all lines) — the degenerate shape the root
    block's alternative slots already have.
    """
    _setup, vnodes = parse_tsumego_sgf(game)
    answer = None
    for c in _evaluated_children_of(vnodes):
        e = _node_eval(c)
        if e is None:
            continue
        mover_black = c.get_raw_move()[0] == "b"
        key = (e[0], e[1]) if mover_black else (1.0 - e[0], -e[1])
        if answer is None or key > answer[0]:
            answer = (key, c)
    if answer is None:
        return None
    solver_black = answer[1].get_raw_move()[0] == "b"

    feat = np.empty(N_CONT, dtype=np.float32)
    counts_at = 2 * (CONT_PLIES * CONT_MOVES + CONT_TRIES * CONT_ANSWERS)
    # The line ends -> the position stands at the value it reached, nothing left to read. Distinct
    # from a slot pad, and told apart by a count of 0.
    pad = (_logit(answer[0][0]), float(np.clip(answer[0][1], -SCORE_CLIP, SCORE_CLIP)))

    # --- 1. the principal line ---
    node, first_opp = answer[1], []
    for d in range(CONT_PLIES):
        ranked = _ranked_children(node, solver_black) if node is not None else []
        if d == 0:
            first_opp = ranked                       # the opponent's tries, kept for part 2
        # Ply 0 is the opponent's reply to the answer, so the solver moves on odd plies.
        row = _slots(ranked, CONT_MOVES, pad, solver_to_move=(d % 2 == 1))
        for m, v in enumerate(row):
            feat[(d * CONT_MOVES + m) * 2] = v[0]
            feat[(d * CONT_MOVES + m) * 2 + 1] = v[1]
        feat[counts_at + d] = float(len(ranked))
        node = ranked[0][2] if ranked else None
        if ranked:
            pad = (row[0][0], row[0][1])             # the line's standing value if it ends here

    # --- 2. the opponent's other tries, each with the solver's best answers ---
    base = CONT_PLIES * CONT_MOVES
    for i in range(CONT_TRIES):
        t = i + 1                                    # try 0 is the principal line
        ranked = _ranked_children(first_opp[t][2], solver_black) if t < len(first_opp) else []
        # A try the opponent does not have is one less thing to answer, so an absent GROUP pads
        # solver-favourably — the same "disfavour the mover" rule one level up, the mover here
        # being the opponent choosing whether such a try exists.
        row = _slots(ranked, CONT_ANSWERS,
                     (_logit(1.0 - WP_CLIP), pad[1]), solver_to_move=True)
        for m, v in enumerate(row):
            feat[(base + i * CONT_ANSWERS + m) * 2] = v[0]
            feat[(base + i * CONT_ANSWERS + m) * 2 + 1] = v[1]
        feat[counts_at + CONT_PLIES + i] = float(len(ranked))
    return feat


def best_alternative_xy(game):
    """KataGo (x, y) of the BEST ALTERNATIVE first move — the runner-up by the mover's own
    preference — or (-1, -1) if the problem has none.

    Addresses the activation .npz, whose rows are every legal cell in board-scan order, so the
    coordinate has to be built the way extraction built it: KataGo (x, y) with y counted from the
    BOTTOM, via gtp_to_xy(sgf_to_gtp(...)). Reading the SGF letters directly gives the row from
    the TOP and silently selects a mirrored cell — a wrong-but-plausible activation vector with
    nothing downstream able to tell.

    PASS is skipped rather than counted: extraction embeds only legal cells, so the pass
    variation every problem carries has no row to point at, and the runner-up that matters here
    is a move a solver could actually play.
    """
    _setup, vnodes = parse_tsumego_sgf(game)
    size = game.get_size()
    ranked = []
    for c in _evaluated_children_of(vnodes):
        e = _node_eval(c)
        if e is None:
            continue
        _color, mb = c.get_raw_move()
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is None:
            continue                                  # pass: no embedded cell
        mover_black = c.get_raw_move()[0] == "b"
        key = (e[0], e[1]) if mover_black else (1.0 - e[0], -e[1])
        ranked.append((key, xy))
    if len(ranked) < 2:
        return (-1, -1)
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked[1][1]                               # rank 0 is the answer


def _process_chunk(task):
    """Worker: feature vectors for a chunk of tids (module scope so it pickles).

    Fail-loud: an unreadable/corrupt SGF, or a problem with no usable move
    evaluations, RAISES and aborts the whole build (the exception surfaces in
    the parent at fu.result()). Dropping such a problem would silently desync
    the cache from correct_ids — train_mirt would then score a different
    problem set than it thinks — so a bad input must be fixed, not skipped."""
    size, tids = task
    tdir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    out = np.empty((len(tids), N_FEATS), dtype=np.float32)
    cont = np.empty((len(tids), N_CONT), dtype=np.float32)
    axy = np.empty((len(tids), 2), dtype=np.int32)
    for i, tid in enumerate(tids):
        # ONE parse feeds both blocks — the sgfmill parse dominates this step, so the
        # continuation costs essentially nothing on top of the root evaluations.
        try:
            with open(os.path.join(tdir, f"{tid}.sgf"), "rb") as fh:
                game = sgfmill.sgf.Sgf_game.from_bytes(fh.read())
            f = extract_move_features(game)
            c = extract_continuation_features(game)
            a = best_alternative_xy(game)
        except Exception as e:
            raise RuntimeError(f"size {size}: failed to read/parse {tid}.sgf") from e
        if f is None or c is None:
            raise RuntimeError(
                f"size {size}: {tid}.sgf has no usable move evaluations "
                f"(missing '<winrate>%/<score>/<visits>' comments)")
        out[i] = f
        cont[i] = c
        axy[i] = a
    return tids, out, cont, axy


def build_size(size, workers, force, rebuild_activations=False):
    out_path = _cache_path(size)
    ids_path = _cache_ids_path(size)

    # Refresh the correct-vector cache FIRST: correct_ids is this step's tid set and the
    # rating model's scorable set, and both roll up the per-problem activation .npz files
    # that extract_activations writes. Refreshing here means one command brings both
    # aggregate caches forward after a tsumego top-up, instead of a stale correct_ids
    # silently truncating this cache and the rated set alike. It is a cheap no-op when
    # already current. A size with no activations at all is skipped, not an error.
    #
    # --force does NOT reach it: forcing a move-feature re-extraction is a minutes-long
    # SGF reparse, while forcing the activation cache is a full re-read of every .npz in
    # the corpus. --rebuild-activation-cache asks for that separately.
    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    if not os.path.isdir(act_dir):
        print(f"size {size}: no activations dir ({act_dir}); skipping")
        return
    build_correct_vec_cache(size, workers=workers, force=rebuild_activations)

    if not os.path.exists(ids_path):
        print(f"size {size}: no activation id cache ({ids_path}); skipping")
        return
    tids = json.load(open(ids_path))

    # INCREMENTAL: each row is a pure per-problem function of its SGF's embedded
    # KataGo eval (no training, no cross-problem dependency), so reusing the already-
    # cached rows and extracting only the tids missing from the cache is byte-identical
    # to a full rebuild — at a fraction of the cost. This also makes the no-force path
    # self-updating: a top-up of new tsumegos is picked up automatically instead of the
    # old "skip if the file exists" (which silently left the cache stale). --force still
    # re-extracts every row.
    #
    # A row is reusable only if BOTH blocks are cached for it. A cache written before the
    # continuation block existed carries `feats` alone, and reusing those rows would leave
    # the continuation half unfilled with nothing saying so, so such a cache re-extracts
    # whole (one SGF reparse of the corpus) rather than being read as current.
    cached = {name: {} for name in CACHE_ARRAYS}
    if os.path.exists(out_path) and not force:
        with np.load(out_path) as d:
            have = [n for n in CACHE_ARRAYS if n in d.files]
            if len(have) == len(CACHE_ARRAYS):
                ids = d["tids"].tolist()
                for name in CACHE_ARRAYS:
                    cached[name] = {t: r for t, r in zip(ids, d[name])}
            else:
                miss = [n for n in CACHE_ARRAYS if n not in have]
                print(f"size {size}: cache is missing {miss} — re-extracting")
    to_extract = tids if force else [t for t in tids if t not in cached["feats"]]
    # Surplus = cached rows whose tid has since LEFT correct_ids (a corpus purge). The write
    # path below emits in correct_ids order and so prunes them, but only if we get there:
    # short-circuiting on `not to_extract` alone would strand them, since nothing new needs
    # extracting. The cache is up to date only when its tid set EQUALS the target set.
    target = set(tids)
    surplus = [t for t in cached["feats"] if t not in target]
    if not to_extract and not surplus:
        print(f"size {size}: cache up to date ({len(cached['feats'])} problems)")
        return
    if surplus:
        print(f"size {size}: pruning {len(surplus)} rows no longer in correct_ids "
              f"(e.g. {surplus[:3]})")

    idx = {t: i for i, t in enumerate(to_extract)}
    new = {"feats": np.full((len(to_extract), N_FEATS), np.nan, dtype=np.float32),
           "cont": np.full((len(to_extract), N_CONT), np.nan, dtype=np.float32),
           "alt_xy": np.full((len(to_extract), 2), -1, dtype=np.int32)}
    chunks = [(size, to_extract[i:i + CHUNK]) for i in range(0, len(to_extract), CHUNK)]
    t0 = time.time()
    done = 0
    # A prune-only run has nothing to extract, so `chunks` is empty; skip the worker pool
    # and fall straight through to the write, which is what does the pruning.
    if chunks:
        print(f"size {size}: {len(to_extract)} to extract ({len(cached['feats'])} cached), "
              f"{len(chunks)} chunks, {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_chunk, c) for c in chunks]
            for k, fu in enumerate(as_completed(futs)):
                ctids, cout, ccont, caxy = fu.result()
                for t, row, crow, arow in zip(ctids, cout, ccont, caxy):
                    new["feats"][idx[t]] = row
                    new["cont"][idx[t]] = crow
                    new["alt_xy"][idx[t]] = arow
                done += len(ctids)
                if (k + 1) % 50 == 0 or done == len(to_extract):
                    print(f"  {done}/{len(to_extract)} "
                          f"({done / max(time.time() - t0, 1e-9):.0f}/s)")
    for name in CACHE_ARRAYS:
        for t, row in zip(to_extract, new[name]):
            cached[name][t] = row

    # Emit in correct_ids order. Every tid is guaranteed present and finite — a
    # failed extraction aborts the run above — so nothing is dropped and the
    # cache tid set matches correct_ids exactly. The finite check is a loud
    # backstop (replacing the old silent drop): a corrupt cache must not ship.
    out_tids = list(tids)
    out = {name: np.array([cached[name][t] for t in tids]) for name in CACHE_ARRAYS}
    bad = [t for i, t in enumerate(tids)
           if not (np.isfinite(out["feats"][i]).all() and np.isfinite(out["cont"][i]).all())]
    if bad:
        raise RuntimeError(
            f"size {size}: {len(bad)} non-finite feature rows (e.g. {bad[:5]}); "
            f"refusing to write a corrupt cache")
    n_noalt = int((out["alt_xy"][:, 0] < 0).sum())
    if n_noalt:
        print(f"size {size}: {n_noalt} problems have no non-pass alternative first move")
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}.npz"
    np.savez(tmp, tids=np.array(out_tids), **out)
    os.replace(tmp, out_path)
    print(f"size {size}: wrote {len(out_tids)} rows to {os.path.basename(out_path)} "
          f"in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="9,13", help="comma-separated board sizes to cache")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true",
                    help="re-extract every move-feature row (leaves the activation cache "
                         "on its incremental path)")
    ap.add_argument("--rebuild-activation-cache", action="store_true",
                    help="also re-read every activation .npz into correct_vecs (hours; "
                         "only needed if that cache is corrupt)")
    args = ap.parse_args()
    for s in [int(x) for x in args.sizes.split(",") if x.strip()]:
        build_size(s, max(1, args.workers), args.force, args.rebuild_activation_cache)


if __name__ == "__main__":
    main()
