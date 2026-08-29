#!/usr/bin/env python3
"""Rebuild an existing tsumego SGF's annotation from scratch at honest depth.

WHAT IS WRONG WITH THE FILES. Every "<winrate>%/<score>" comment in the corpus
is a 30-visit estimate. Generation ran its deep passes at PLAYOUTS=30 and then
let `add_shallow_variations` overwrite the decision node unconditionally, while
`problem_finder` had already verified the position at 1000 visits and thrown that
away — KataGo rebuilds its search tree per query, so the deep result is not
inherited, only its neural-net evaluations are cached. 14.5% of 13x13 answers and
12.2% of 9x9 answers record a winrate below the 80% floor generation refuses to
admit, which is impossible as a property of the position.

WHAT THIS WRITES. Generation's annotation phase, re-run at a real budget: the
answer's continuation and the refutations each entered at DEEP_ENTRY_PLAYOUTS,
followed down by deep_follow_playouts, then the shallow sweep over whatever the
searches declined to read. Every node records the visits behind its estimate as a
third comment field, so this can never go unnoticed again.

REBUILT, NOT PATCHED. The old annotation is discarded and the file is
reconstructed through generation's own `construct_initial_sgf`, so a repaired
file is structurally what generation would produce today rather than a hybrid of
two eras. That is safe because the corpus root carries only FF/CA/GM/KM/RU/SZ and
its trunk nodes carry nothing but moves — verified across the corpus, not
assumed. What is preserved is what identifies the problem: the trunk, and the
answer as variation 0.

WHY THE ANSWER NEEDS ITS OWN WRITE. The answer is nobody's `moveInfos` entry —
the decision-node pass excludes it, precisely so the budget reaches the
refutations instead of being swallowed by the move that wins. Its comment
therefore comes from `write_parent_eval`: the root evaluation of the search of
the position it leads to, which is what "how good is this move" means and what
the shallow pass was approximating at 30 visits.

THE ANSWER STAYS AT INDEX 0 BY CONSTRUCTION, not by enforcement.
`construct_initial_sgf` hangs it as the crux's only child; the first search
starts AFTER it, so it can only add children beneath it; the second passes
`exclude_move`, so the answer is never in its `moveInfos` and its alternates can
only be appended after it; and the shallow sweep looks the answer up with
`get_child_move`, which FINDS the existing node rather than creating one, while
`set_eval_comment` refuses to let its 30-visit value displace the answer's deep
one. Nothing in this pass can reorder, which matters because every backend stage
reads `variation_nodes[0]` POSITIONALLY — a reordering would feed a losing move's
activations into the ratings while the app still played correctly.

WHAT THE TWO CHECKS ARE FOR is AGREEMENT, not position. Under 1000 visits a
problem admitted on 30-visit numbers can stop being well posed:
`first_move_is_well_formed` catches it losing its single winner, and
`winning_first_move` catches the new numbers preferring some other first move.
Either way the file is positionally correct but internally contradictory — the
backend would read index 0 while the client's `findBestMoveFromChildren` picked
something else.

SUCH A FILE IS STILL WRITTEN. The searches are already paid for and their
variations are strictly better than the 30-visit ones on disk, so withholding
them would leave the corpus's worst-annotated problems as its only old-format
ones. The id goes to the flag file instead, and retirement is a separate decision
taken over the whole set. The check deliberately uses `winning_first_move`, which
only ASKS, rather than `promote_winning_first_move`, which would silently resolve
the contradiction by reordering — generation is allowed to promote a late winner,
a repair is not, because the id, the ratings and the attempt history are all tied
to the answer this file already has.

Resumable and atomic: a done-list is appended after each file, and each SGF is
written to a temp file and renamed, so an interrupted run leaves no half-file.

  ./.venv/bin/python3 -m auxiliary.repair_sgf_evals --size 13 --limit 20 --dry-run
  ./.venv/bin/python3 -m auxiliary.repair_sgf_evals --size 13
"""

import argparse
import os
import queue
import re
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf  # noqa: E402

from utils.constants import ALL_BOARD_SIZES, DEFAULT_MODEL  # noqa: E402
from utils.katago.add_variations import (  # noqa: E402
    DEEP_ENTRY_PLAYOUTS, VariationDepthExceeded, add_shallow_variations,
    construct_initial_sgf, deep_follow_playouts, first_move_is_well_formed,
    generate_variations_directly_to_sgf, winning_first_move)
from utils.katago.controller import KatagoController  # noqa: E402
from utils.katago.coords import sgf_to_gtp  # noqa: E402
from utils.katago.models import get_model  # noqa: E402
from utils.katago.queries import KatagoQueries  # noqa: E402


def default_workers():
    """numAnalysisThreads from the tuned config — the number of positions KataGo
    will actually work on at once. Asking for more just queues; asking for fewer
    leaves analysis threads idle."""
    cfg = os.path.join(PROJECT_ROOT, "bots", "analysis.cfg")
    try:
        m = re.search(r"^\s*numAnalysisThreads\s*=\s*(\d+)", open(cfg).read(),
                      re.M)
        if m:
            return max(1, int(m.group(1)))
    except OSError:
        pass
    return 8


def crux_and_answer(game, size):
    """(moves to the decision node, answer move, crux node) in GTP coordinates.

    The decision node is where the trunk stops being a single line. The answer is
    its FIRST child — the corpus contract, enforced at generation by
    promote_winning_first_move."""
    node, moves = game.get_root(), []
    while True:
        kids = list(node)
        if len(kids) != 1:
            break
        node = kids[0]
        _c, mv = node.get_raw_move()
        if mv is None:
            break
        moves.append(sgf_to_gtp(mv.decode(), size))
    kids = list(node)
    if not kids:
        return None, None, None
    _c, mv = kids[0].get_raw_move()
    if mv is None:
        return None, None, None
    return moves, sgf_to_gtp(mv.decode(), size), node


def repair_one(queries, path, size, dry_run):
    """Rebuild one file's annotation. Returns (status, detail)."""
    old = sgfmill.sgf.Sgf_game.from_bytes(open(path, "rb").read())
    moves, answer_gtp, _crux = crux_and_answer(old, size)
    if not moves or answer_gtp is None:
        return "skipped", "no decision node"

    # The trunk and the answer are everything the old file contributes.
    game, node, answer = construct_initial_sgf(moves, answer_gtp, size)

    try:
        generate_variations_directly_to_sgf(
            queries, answer, moves + [answer_gtp], True,
            playouts=DEEP_ENTRY_PLAYOUTS, playouts_fn=deep_follow_playouts,
            write_parent_eval=True)
        generate_variations_directly_to_sgf(
            queries, node, moves, True, answer_gtp,
            playouts=DEEP_ENTRY_PLAYOUTS, playouts_fn=deep_follow_playouts)
        # Last, and it fills only what the two searches left bare — it decides
        # per move BEFORE querying, so the moves they covered cost nothing here.
        add_shallow_variations(queries, node, moves)
    except VariationDepthExceeded as e:
        # `.moves` is the move LIST; args[0] is a formatted string, and taking
        # len() of that reports characters — which read as a depth of 5334
        # against a cap of 800 until it was checked.
        return "failed", f"depth exceeded at {len(getattr(e, 'moves', []))} plies"
    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"

    # Both checks are ADVISORY: they classify the problem, they do not gate the
    # write. The searches are already paid for and the variations they produced
    # are strictly better than the 30-visit ones on disk, so withholding them
    # would leave the corpus's worst-annotated problems as its ONLY
    # old-format files. The id goes to the flag file instead, and retiring is a
    # separate decision made over the whole set.
    status, detail = "ok", f"{len(list(node))} root variations"
    if not first_move_is_well_formed(node):
        status, detail = "not-well-formed", "no longer single-solution under deeper eval"
    elif winning_first_move(node) is not answer:
        # ASKED, not applied. Generation reorders a late winner; a repair must
        # not, because the id, the ratings and the attempt history are all tied
        # to the answer this file already has.
        status, detail = "answer-changed", "a different first move now evaluates as the winner"

    if dry_run:
        return f"{status}(dry)", detail

    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(game.serialise())
    os.replace(tmp, path)
    return status, detail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, required=True, choices=ALL_BOARD_SIZES)
    ap.add_argument("--corpus-dir", default=None,
                    help="directory of SGFs to repair (default tsumegos/<size>). "
                         "tsumegos/new/<size> holds problems generated but not yet "
                         "promoted, and they carry the same shallow evaluations, so "
                         "they need the same pass — promotion is a git move and "
                         "does not re-annotate. `broken/` is quarantined and stays "
                         "out of the corpus, so it is deliberately NOT repaired.")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files")
    ap.add_argument("--dry-run", action="store_true",
                    help="do the work and the checks, write nothing")
    ap.add_argument("--done-file", default=None,
                    help="resume list (default repaired_<size>.txt beside the "
                         "corpus). APPENDED BY THIS MACHINE ONLY — it is what the "
                         "sync-back uses to decide which files this box actually "
                         "repaired, so it must never contain another box's work.")
    ap.add_argument("--flag-file", default=None,
                    help="durable record of every problem written with a "
                         "non-ok status (default flagged_<size>.txt beside the "
                         "corpus), as id/status/reason. These files ARE "
                         "repaired — the flag says the problem stopped being "
                         "well-posed once it was evaluated properly, which is a "
                         "retirement decision to make over the whole set later. "
                         "It exists because the run log is the only other record "
                         "and a relaunch truncates it.")
    ap.add_argument("--exclude-file", default=None,
                    help="additional ids to skip, read-only. This is how a "
                         "RE-SHARD avoids duplicate work: push the union of every "
                         "box's done-list here, and a box that inherits a shard "
                         "another box already finished skips those files without "
                         "claiming them as its own. Kept separate from --done-file "
                         "precisely because the sync-back reads that one.")
    ap.add_argument("--workers", type=int, default=0,
                    help="concurrent files (default: numAnalysisThreads from "
                         "bots/analysis.cfg). One engine serves them all; the "
                         "point is to keep its analysis threads saturated, since "
                         "a single file's recursion is sequential and leaves the "
                         "GPU idle between queries.")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the corpus across N machines; each takes every "
                         "Nth id from the sorted list. Deterministic and needs no "
                         "coordination — the boxes share no state, so their "
                         "resume lists stay independent and the repaired files "
                         "are disjoint and merge cleanly on sync-back.")
    ap.add_argument("--shard", default="0",
                    help="which shard(s) this machine takes, 0-based, comma "
                         "separated. A LIST because the machines are not equally "
                         "fast: with --shards 10 a card measured at 3x another's "
                         "rate takes 7 shards to its 3, so both finish together "
                         "instead of the fast one idling while the slow one runs "
                         "on. Split by measured files/h, not by machine count.")
    ap.add_argument("--engines", type=int, default=1,
                    help="KataGo processes sharing the GPU. ONE ENGINE DOES NOT "
                         "SATURATE A FAST CARD — the generator needs 2-3 depending "
                         "on the machine, and vast_orchestration caches the tuned "
                         "count per GPU model (4070 Ti: 2, 5090: 3). Workers are "
                         "split round-robin across them.")
    ap.add_argument("--nn-batch", type=int, default=0,
                    help="override nnMaxBatchSize. The tuner picks a batch for "
                         "INTERACTIVE search; a bulk sweep runs many positions at "
                         "once and wants a far larger one, or the GPU is handed "
                         "batches smaller than the number of threads feeding it.")
    ap.add_argument("--analysis-threads", type=int, default=0,
                    help="override numAnalysisThreads. Raise with --workers and "
                         "--nn-batch together: threads without batch just queue.")
    ap.add_argument("--restart-every", type=int, default=5000,
                    help="restart the engine every N files. HYGIENE ONLY, and "
                         "deliberately large: nnCacheSizePowerOfTwo makes the NN "
                         "cache a FIXED-SIZE LRU, so it does not grow and there is "
                         "nothing to reclaim. A restart costs the ~20s model load "
                         "AND a drained worker pool, because the batch joins before "
                         "the next engine starts and the last wave runs with most "
                         "threads idle. At 200 that overhead was hours across the "
                         "corpus; the only reason to keep any restart is to survive "
                         "an engine that wedges on a multi-day run.")
    args = ap.parse_args()

    corpus = args.corpus_dir or os.path.join(PROJECT_ROOT, "tsumegos",
                                             str(args.size))
    if not os.path.isabs(corpus):
        corpus = os.path.join(PROJECT_ROOT, corpus)
    if not os.path.isdir(corpus):
        sys.exit(f"no such corpus directory: {corpus}")
    # The resume list lives BESIDE the corpus it describes, so a second directory
    # cannot be credited with the first one's completed work.
    done_file = args.done_file or os.path.join(
        os.path.dirname(corpus.rstrip("/")),
        f"repaired_{os.path.basename(corpus.rstrip('/'))}.txt")
    flag_file = args.flag_file or os.path.join(
        os.path.dirname(corpus.rstrip("/")),
        f"flagged_{os.path.basename(corpus.rstrip('/'))}.txt")
    done = set()
    if os.path.exists(done_file):
        done = {l.strip() for l in open(done_file) if l.strip()}
    skip = set(done)
    if args.exclude_file and os.path.exists(args.exclude_file):
        extra = {l.strip() for l in open(args.exclude_file) if l.strip()}
        skip |= extra
        print(f"excluding {len(extra)} ids repaired elsewhere "
              f"({len(extra - done)} not in this box's own list)", flush=True)
    # find, never a glob: 100k+ files blow past ARG_MAX
    tids = sorted(f[:-4] for f in os.listdir(corpus) if f.endswith(".sgf"))
    mine = {int(x) for x in str(args.shard).split(",") if x.strip() != ""}
    if args.shards > 1:
        # Interleaved, not contiguous blocks: any systematic ordering in the id
        # list (they are content hashes, so there should be none, but the corpus
        # has been through id surgery) would otherwise hand one machine a
        # different KIND of problem than another and make their rates
        # incomparable.
        bad = {x for x in mine if not 0 <= x < args.shards}
        if bad:
            sys.exit(f"--shard {sorted(bad)} outside 0..{args.shards - 1}")
        tids = [t for i, t in enumerate(tids) if i % args.shards in mine]
    todo = [t for t in tids if t not in skip]
    if args.limit:
        todo = todo[:args.limit]
    print(f"flagged/failed problems recorded in {flag_file}", flush=True)
    print(f"corpus {corpus}: {len(tids)} files"
          + (f" (shards {sorted(mine)} of {args.shards})" if args.shards > 1 else "")
          + f", {len(done)} already repaired, {len(todo)} to do"
          + (" (DRY RUN)" if args.dry_run else ""), flush=True)
    if not todo:
        return

    ov = ["requireMaxBoardSize=false"]
    if args.nn_batch:
        ov.append(f"nnMaxBatchSize={args.nn_batch}")
    if args.analysis_threads:
        ov.append(f"numAnalysisThreads={args.analysis_threads}")
    overrides = ",".join(ov)
    per_engine = args.analysis_threads or default_workers()
    workers = args.workers or per_engine * args.engines
    print(f"{workers} worker threads across {args.engines} engine(s) "
          f"({per_engine}/engine), overrides: {overrides}", flush=True)

    counts, lock, t0, done_n = {}, threading.Lock(), time.time(), [0]

    def worker(q, queries):
        while True:
            try:
                tid = q.get_nowait()
            except queue.Empty:
                return
            try:
                status, detail = repair_one(queries,
                                            os.path.join(corpus, f"{tid}.sgf"),
                                            args.size, args.dry_run)
            except Exception as e:                      # never let one file end a batch
                status, detail = "failed", f"{type(e).__name__}: {e}"
            with lock:
                counts[status] = counts.get(status, 0) + 1
                done_n[0] += 1
                n = done_n[0]
                # A file is DONE once it has been written, whatever its
                # status says about the problem — only a hard failure leaves the
                # old annotation on disk, and only that is worth retrying.
                if not args.dry_run and not status.startswith("failed"):
                    with open(done_file, "a") as fh:
                        fh.write(tid + "\n")
                if not status.startswith("ok"):
                    print(f"  {tid}: {status} — {detail}", flush=True)
                    if not args.dry_run:
                        with open(flag_file, "a") as fh:
                            fh.write(f"{tid}\t{status}\t{detail}\n")
                if n % 50 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  {el / n:.2f}s/file  "
                          f"eta {(len(todo) - n) * el / n / 3600:.1f}h  {counts}",
                          flush=True)

    # BATCHED, and the batch boundary is where the engine is recycled. One engine
    # serves every worker — KatagoController fans responses out by query id, so
    # concurrent send_query calls are safe, and running WHOLE FILES in parallel is
    # what keeps KataGo busy: a single file's recursion is query-wait-query, which
    # would leave the GPU idle between every step.
    #
    # The boundary is not free — the pool drains before the next engine starts, so
    # the final wave of each batch runs with most threads idle — which is why
    # --restart-every is large. It is hygiene against a wedged engine, not cache
    # management; the NN cache is a fixed-size LRU.
    for start in range(0, len(todo), args.restart_every):
        batch = todo[start:start + args.restart_every]
        ctrls = []
        try:
            for _ in range(args.engines):
                c = KatagoController()
                c.start_engine(get_model(DEFAULT_MODEL), board_size=args.size,
                               extra_overrides=overrides)
                ctrls.append(c)
            qs = [KatagoQueries(c, args.size) for c in ctrls]
            # start_engine only spawns; the model load lands on the first query,
            # so pay it here — in parallel, or N engines serialise their loads.
            warm = [threading.Thread(target=lambda x=x: x.query([], playouts=1))
                    for x in qs]
            for t in warm:
                t.start()
            for t in warm:
                t.join()
            q = queue.Queue()
            for tid in batch:
                q.put(tid)
            # Round-robin so each engine gets the same number of workers. The
            # queue is shared, so a slow file on one engine does not starve the
            # others — they simply take the next id.
            threads = [threading.Thread(target=worker,
                                        args=(q, qs[i % len(qs)]), daemon=True)
                       for i in range(workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            for c in ctrls:
                try:
                    c.stop_engine()
                except Exception:
                    pass
    print(f"done in {(time.time() - t0) / 60:.1f} min: {counts}", flush=True)


if __name__ == "__main__":
    main()
