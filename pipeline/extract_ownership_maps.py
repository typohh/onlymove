"""Per-intersection ownership after each of a problem's leading first moves and a
pass → activations/_signal_cache/ownership_maps/<size>/part_<n>.npz.

This is the ENGINE half of every ownership-derived motif rule. It runs KataGo
once and records whole maps, so a rule — or a threshold inside one — is changed
by re-reading the cache rather than by re-running the GPU. The layout, the slot
meanings and the BLACK-positive frame convention are stated once, in the reader
(utils.rating.ownership_maps); this module only fills them.

Per problem it evaluates (one maxVisits query each, issued as the full MOVE
SEQUENCE under the corpus rules — see extract_ownership_swing on why a stone
snapshot scores territory wrongly):
  - a PASS by the solver, whether or not the SGF carries a pass variation
  - the answer, variation_nodes[0]
  - the next TOP_ALTS stone-move root variations in SGF ORDER, so the first five
    moves including the answer. That order is the generator's own ranking of its
    candidates, so the second branch is practically always the best alternative

Several candidates rather than one baseline because a motif is often legible only
in the contrast: two leading moves sharing a purpose against a third and a pass
that do not. The root itself is not evaluated — the engine prices it as though
the answer is found, so its map is practically the answer's.

Moves the SGF names that are not legal
on the replayed board abort the problem rather than being dropped: they mean the
SGF and the board disagree, which no rule downstream can repair.

ALL EIGHT BOARD SYMMETRIES ARE AVERAGED (rootNumSymmetriesToSample). A one-visit
query otherwise evaluates under ONE RANDOM symmetry, and the eight disagree:
measured, two runs of the same query differ by 0.01 ownership per point on
average and by more than 0.2 somewhere on ~4% of problems — exactly where a rule
thresholding at 0.95 or 0.4-0.6 flips its verdict. The average is deterministic,
and it matches the corpus's own identity, which is D4-canonical.

PROVENANCE IS STAMPED and checked. Every part carries the model, visit budget,
rules, komi and layout version; a resume under any other value aborts, since a
cache mixing two networks or two budgets is well-formed and wrong. --force wipes
the size's cache and starts over.

INCREMENTAL BY ID and reconciled both ways: problems already present are skipped,
and a part holding an id no longer in the corpus (a retired problem) is rewritten
without it. Each part is written to a temp name and renamed, so a kill loses at
most the part in flight.

Launch detached (a few hours over the corpus at maxVisits=1 on one GPU):
    setsid nohup ./.venv/bin/python3 -u -m pipeline.extract_ownership_maps \\
        > /tmp/ownership_maps.log 2>&1 < /dev/null &
"""

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import sgfmill.sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils.katago.torch_model  # noqa: F401,E402  -- adds vendored katago/ to sys.path

# ONLY LIGHT IMPORTS AT MODULE LEVEL. The parse workers are SPAWNED, so each
# re-imports this module: the vendored Board is ~25 MB, while GameState (its
# feature tables) is ~877 MB and the engine side is heavier still. Those are
# imported inside run_size, which only the parent runs.
from katago.game.board import Board  # noqa: E402

from utils.constants import ALL_BOARD_SIZES, KOMI, RULES  # noqa: E402
from utils.katago.add_variations import (  # noqa: E402
    parse_eval_comment, parse_eval_visits, subtree_reaches_conclusion)
from utils.katago.coords import gtp_to_xy, sgf_to_gtp, xy_to_gtp  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf  # noqa: E402
from utils.rating import ownership_maps as OM  # noqa: E402

# THE CPU HALF IS IN PROCESSES, THE GPU HALF IS BATCHED. Run in one interpreter,
# SGF parsing and board replay hold the GIL and starve the engine — measured: one
# core at ~75% and the GPU idle, 60 problems/s. So parsing runs in PARSE_WORKERS
# spawned processes, one chunk ahead of the engine, and every query of a chunk is
# put in flight at once so KataGo's batches stay full.
ENGINE_THREADS = 128     # KataGo analysis threads = queries evaluated concurrently
NN_BATCH = 128
QUERY_THREADS = 128      # in-flight queries from this side
PARSE_WORKERS = 6
CHUNK = 2000             # problems per part file

PASS = "pass"


def provenance(visits, symmetries):
    from utils.katago.models import DEFAULT_MODEL
    return {"model": DEFAULT_MODEL, "visits": visits, "symmetries": symmetries,
            "rules": RULES, "komi": KOMI,
            "layout": OM.LAYOUT_VERSION, "top_alts": OM.TOP_ALTS}


def node_eval(node):
    """(winrate 0..1, score, visits) from a node's comment, BLACK frame, or
    (nan, nan, 0) when it carries none."""
    comment = node.get("C") if node.has_property("C") else None
    parsed = parse_eval_comment(comment)
    if parsed is None:
        return np.nan, np.nan, 0
    wr, sc = parsed
    return wr / 100.0, sc, parse_eval_visits(comment)


def variation_move(node, size):
    """A root variation's move: an (x, y) cell, PASS, or None for a node with no
    move. Same reading as extract_ownership_swing.variation_move — repeated
    because that module imports GameState at load, which a spawned worker cannot
    afford."""
    colour, move_bytes = node.get_raw_move()
    if colour is None:
        return None
    if not move_bytes:
        return PASS
    gtp = sgf_to_gtp(move_bytes.decode("utf-8"), size)
    return PASS if gtp == PASS else gtp_to_xy(gtp)


def replay(setup_moves, size):
    """The root board, replayed move by move with captures (GameState.play is
    exactly Board.play plus a history this step does not need)."""
    board = Board(size=size)
    for colour, gtp in setup_moves:
        xy = gtp_to_xy(gtp)
        if xy is None:
            continue
        board.play(Board.BLACK if colour.lower() == "b" else Board.WHITE, board.loc(*xy))
    return board


def select_candidates(variation_nodes, size, legal):
    """[(slot, move, node_or_None, var_idx)] for every slot this problem fills.
    `move` is an (x, y) cell or PASS; `var_idx` is the node's index among the root
    variations, -1 for a pass the SGF has no variation for.

    The alternatives are the stone moves that FOLLOW the answer in document order,
    not a re-ranking of them: the generator writes its candidates best-first, so
    that order is its own verdict on them, reached with the search it ran."""
    correct = variation_move(variation_nodes[0], size)
    if correct is None or correct is PASS:
        raise ValueError("the answer is not a stone move")
    if correct not in legal:
        raise ValueError(f"answer {correct} is not legal on the replayed board")

    pass_node, pass_idx, alts = None, -1, []
    for i, node in enumerate(variation_nodes[1:], start=1):
        mv = variation_move(node, size)
        if mv is None:
            continue
        if mv is PASS:
            if pass_node is None:
                pass_node, pass_idx = node, i
            continue
        if mv not in legal:
            raise ValueError(f"root variation {mv} is not legal on the replayed board")
        if len(alts) < OM.TOP_ALTS:
            alts.append((node, mv, i))

    out = [(OM.SLOT_PASS, PASS, pass_node, pass_idx),
           (OM.SLOT_CORRECT, correct, variation_nodes[0], 0)]
    out += [(OM.SLOT_ALT0 + k, mv, node, i) for k, (node, mv, i) in enumerate(alts)]
    return out


def prepare_problem(path):
    """The CPU half, run in a worker process: everything but the engine's answers,
    plus the move list each slot's query plays. Raises on anything malformed, and
    the parent re-raises it — an unreadable problem aborts the run."""
    with open(path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    setup_moves, variation_nodes = parse_tsumego_sgf(game)
    colour, _ = variation_nodes[0].get_raw_move()
    pla = Board.BLACK if colour.lower() == "b" else Board.WHITE
    board = replay(setup_moves, size)

    stones = np.zeros((size, size), dtype=np.int8)
    legal = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            loc = board.loc(x, y)
            c = board.board[loc]
            stones[y, x] = OM.BLACK if c == Board.BLACK else OM.WHITE if c == Board.WHITE else OM.EMPTY
            legal[y, x] = c == Board.EMPTY and board.would_be_legal(pla, loc)
    legal_cells = {(x, y) for y in range(size) for x in range(size) if legal[y, x]}

    S = OM.SLOTS
    rec = {
        "tid": os.path.basename(path)[:-4],
        "solver": np.int8(OM.BLACK if pla == Board.BLACK else OM.WHITE),
        "stones": stones, "legal": legal,
        "own": np.full((S, size, size), np.nan, dtype=np.float16),
        "move": np.full(S, OM.MOVE_NONE, dtype=np.int16),
        "var_idx": np.full(S, -1, dtype=np.int16),
        "sgf_wr": np.full(S, np.nan, dtype=np.float32),
        "sgf_score": np.full(S, np.nan, dtype=np.float32),
        "sgf_visits": np.zeros(S, dtype=np.int32),
        "resolved": np.zeros(S, dtype=bool),
        "eng_wr": np.full(S, np.nan, dtype=np.float32),
        "eng_score": np.full(S, np.nan, dtype=np.float32),
    }
    # The full MOVE SEQUENCE, never a stone snapshot: territory scoring reads the
    # prisoner count, which only the sequence carries (extract_ownership_swing).
    seq = [[c.upper(), gtp] for c, gtp in setup_moves]
    colour_s = "B" if pla == Board.BLACK else "W"
    queries = []
    for slot, mv, node, var_idx in select_candidates(variation_nodes, size, legal_cells):
        rec["move"][slot] = OM.MOVE_PASS if mv is PASS else mv[1] * size + mv[0]
        rec["var_idx"][slot] = var_idx
        if node is not None:
            rec["sgf_wr"][slot], rec["sgf_score"][slot], rec["sgf_visits"][slot] = node_eval(node)
            rec["resolved"][slot] = subtree_reaches_conclusion(node)
        queries.append((slot, seq + [[colour_s, PASS if mv is PASS else xy_to_gtp(*mv)]]))
    return size, rec, queries


def evaluate(ctl, size, recs_queries, visits, symmetries, pool):
    """The GPU half: every query of the chunk in flight at once."""
    jobs = [(rec, slot, moves) for rec, queries in recs_queries for slot, moves in queries]

    def one(job):
        rec, slot, moves = job
        r = ctl.send_query({
            "initialStones": [], "moves": moves,
            "rules": RULES, "komi": KOMI, "boardXSize": size, "boardYSize": size,
            "maxVisits": visits, "includeOwnership": True, "includePolicy": False,
            "overrideSettings": {"rootNumSymmetriesToSample": symmetries},
        })
        # Engine ownership is BLACK-positive and row-major from the TOP; the cache
        # indexes y from the bottom, so flip rows and keep the sign.
        rec["own"][slot] = np.asarray(r["ownership"], dtype=np.float32).reshape(size, size)[::-1]
        rec["eng_wr"][slot] = r["rootInfo"]["winrate"]
        rec["eng_score"][slot] = r["rootInfo"]["scoreLead"]

    for _ in pool.map(one, jobs):
        pass
    return [rec for rec, _ in recs_queries]


def write_part(path, recs, meta):
    tmp = path + ".tmp.npz"
    np.savez(tmp, meta=np.array(json.dumps(meta, sort_keys=True)),
             **{k: np.stack([np.asarray(r[k]) for r in recs]) for k in OM.ARRAYS})
    os.replace(tmp, path)


def reconcile(size, corpus, meta):
    """Existing ids, after rewriting any part that holds ids no longer in the
    corpus. Aborts on a part stamped with other provenance."""
    done = set()
    for p in OM.part_paths(size):
        with np.load(p) as d:
            m = json.loads(str(d["meta"]))
            if m != meta:
                raise SystemExit(f"{p} was written under {m}, this run is {meta} — "
                                 "re-run with --force to rebuild this size")
            arrays = {k: d[k] for k in OM.ARRAYS}
        keep = np.isin(arrays["tid"], list(corpus))
        if not keep.all():
            print(f"[size {size}] {p}: dropping {int((~keep).sum())} retired ids", flush=True)
            if keep.any():
                tmp = p + ".tmp.npz"
                np.savez(tmp, meta=np.array(json.dumps(meta, sort_keys=True)),
                         **{k: v[keep] for k, v in arrays.items()})
                os.replace(tmp, p)
            else:
                os.remove(p)
        done.update(arrays["tid"][keep].tolist())
    return done


def run_size(size, visits, symmetries, force, limit):
    in_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    out_dir = OM.size_dir(size)
    if force and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for stale in os.listdir(out_dir):
        if stale.endswith(".tmp.npz"):
            os.remove(os.path.join(out_dir, stale))

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".sgf"))
    if limit:
        files = files[:limit]
    meta = provenance(visits, symmetries)
    done = reconcile(size, {f[:-4] for f in files}, meta)
    todo = [f for f in files if f[:-4] not in done]
    print(f"[size {size}] {len(files)} problems, {len(done)} cached, {len(todo)} to do "
          f"-> {out_dir}", flush=True)
    if not todo:
        return

    existing = [int(os.path.basename(p)[5:-4]) for p in OM.part_paths(size)]
    next_part = max(existing, default=-1) + 1
    from utils.katago.controller import KatagoController
    from utils.katago.models import DEFAULT_MODEL, get_model
    ctl = KatagoController()
    ctl.start_engine(get_model(DEFAULT_MODEL), size,
                     extra_overrides=f"nnMaxBatchSize={NN_BATCH},numAnalysisThreads={ENGINE_THREADS}")
    qpool = ThreadPoolExecutor(max_workers=QUERY_THREADS)
    ppool = ProcessPoolExecutor(max_workers=PARSE_WORKERS,
                                mp_context=multiprocessing.get_context("spawn"))
    chunks = [todo[s:s + CHUNK] for s in range(0, len(todo), CHUNK)]

    def submit(chunk):
        return [ppool.submit(prepare_problem, os.path.join(in_dir, f)) for f in chunk]

    t0 = time.time()
    n = 0
    try:
        ahead = submit(chunks[0])
        for k, chunk in enumerate(chunks):
            # AN UNREADABLE PROBLEM ABORTS THE RUN, as in every step: a rule read
            # off this cache aborts on a missing problem, so skipping one here only
            # moves the halt to a later step and a worse place. .result() re-raises.
            prepared = [fut.result() for fut in ahead]
            ahead = submit(chunks[k + 1]) if k + 1 < len(chunks) else []
            for got_size, _, _ in prepared:
                if got_size != size:
                    raise ValueError(f"board size {got_size} under tsumegos/{size}")
            recs = evaluate(ctl, size, [(rec, q) for _, rec, q in prepared],
                            visits, symmetries, qpool)
            write_part(os.path.join(out_dir, f"part_{next_part:05d}.npz"), recs, meta)
            next_part += 1
            n += len(chunk)
            rate = n / (time.time() - t0)
            print(f"[size {size}] {n}/{len(todo)}  {rate:.1f} probs/s  "
                  f"ETA {(len(todo) - n) / rate / 3600:.2f} h", flush=True)
    finally:
        ppool.shutdown(wait=False, cancel_futures=True)
        qpool.shutdown(wait=False)
        ctl.stop_engine()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", default=",".join(map(str, ALL_BOARD_SIZES)))
    ap.add_argument("--visits", type=int, default=1,
                    help="maxVisits per query (default 1, the budget extract_ownership_swing uses)")
    ap.add_argument("--symmetries", type=int, default=8,
                    help="board symmetries the root eval averages (default 8, all of D4)")
    ap.add_argument("--force", action="store_true", help="wipe and rebuild the cache for these sizes")
    ap.add_argument("--limit", type=int, default=0, help="first N problems per size (smoke tests)")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    for size in map(int, args.sizes.split(",")):
        run_size(size, args.visits, args.symmetries, args.force, args.limit)


if __name__ == "__main__":
    main()
