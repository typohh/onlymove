"""Give every tsumego a LOSING LINE that is played out to the end.

WHAT IS MISSING
---------------
A problem's decision node lists the answer plus every refuted first move. The
answer is driven to a DOUBLE PASS at generation time; the refutations are not. On 9x9 the generator pursued a refutation to a conclusion
in only 57% of problems, so in the rest a solver who plays a losing move is
shown a refutation that stops mid-line, and the client falls back to querying
the analysis server for every ply after it.

Note the root PASS variation does not count. `pass` answered by `pass` IS a
double pass, so scoring it would call almost every problem covered; it is also
not a move the client can offer. Same exclusion as
auxiliary/audit_solution_lines.py.

WHAT THIS ADDS
--------------
Exactly one line per problem: the BEST losing first move — highest winrate for
the side to move, score breaking exact ties, which is the client's
findBestMoveFromChildren rule applied to the candidates that are not the answer
— played out with KataGo's own best move for both sides until both sides pass.

Only the PRINCIPAL line is added (see BRANCH_SHARE): one conclusion is what the
problem lacks, and a branching subtree costs both engine time and shard bytes on
a branch the solver reaches only by being wrong.

The added line terminates in a DOUBLE PASS exactly as generation ends the
answer's, which is what `subtree_reaches_conclusion` (and its client twin
`subtreeReachesConclusion`) reads to tell a played-out refutation from an
abandoned stub. Nothing is annotated: the structure carries the fact.

WHAT IT LEAVES ALONE
--------------------
The decision node's children — neither their order nor their count changes, so
`variation_nodes[0]` is still the answer and the per-cell embedding bin still
lines up with the SGF root variations. Nothing upstream of the decision node is
touched, so the problem's identity hash is unchanged.

The file is rewritten through sgfmill, which orders each node's properties its
own way. SGF property order is not significant and every reader here is
order-blind — verified for WGo, the client's parser, as well as sgfmill and the
regex reader in audit_solution_lines. What IS significant is the order of the
decision node's CHILDREN, which this script leaves untouched.

USAGE
-----
    ./.venv/bin/python3 -m auxiliary.add_losing_lines 9 --dry-run
    ./.venv/bin/python3 -m auxiliary.add_losing_lines 9 --ids 000030e6 \\
        --out-dir /tmp/losing_test          # one problem, writes nothing in place
    ./.venv/bin/python3 -m auxiliary.add_losing_lines 9

`tsumegos/` is tracked in git, so `git checkout -- tsumegos/9` reverts a run.

The run is RESUMABLE with no state file, by whichever record of the work exists:
in place, a problem that already has a played-out losing line is skipped — the
same test that selected it; under `--out-dir`, the written file is the record,
since the input it was made from is left untouched. Either way an interrupted
run costs only the problems in flight, which is what makes `--out-dir` the right
mode on a rented remote: the corpus it reads never changes, so the output
directory can be pulled down repeatedly while the run continues.
"""
import argparse
import os
import queue
import sys
import threading
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf  # noqa: E402

# The corpus audit already defines each of these rules once; importing them
# keeps "a line that ended", "the answer by evaluation" and "a pass" identical
# to what the audit and the client read out of the same files.
from auxiliary.check_tsumego_sequences import (  # noqa: E402
    black_eval,
    find_problem_node,
    is_pass,
    pick_winning_move,
    subtree_has_complete_line,
    to_solver,
)
from utils.constants import ALL_BOARD_SIZES, DEBUG, DEFAULT_MODEL  # noqa: E402
from utils.katago.add_variations import (  # noqa: E402
    VariationDepthExceeded,
    generate_variations_directly_to_sgf,
)
from utils.katago.controller import KatagoController  # noqa: E402
from utils.katago.coords import sgf_to_gtp  # noqa: E402
from utils.katago.models import get_model  # noqa: E402
from utils.katago.queries import KatagoQueries  # noqa: E402

# Threads per unit of KataGo's batch size. Same ratio as
# pipeline.generate_tsumegos: python workers have to keep the analysis queue
# full despite the GIL serializing their own SGF work.
THREAD_FACTOR = 3

# A progress line every PROGRESS_EVERY problems OR every PROGRESS_SECONDS,
# whichever comes first. The time-based half is what makes a run watchable: a
# count-based line alone goes quiet for however long a slow stretch takes, which
# reads exactly like a dead engine.
PROGRESS_EVERY = 200
PROGRESS_SECONDS = 60


class Skip(Exception):
    """This problem cannot be processed; the reason is the message."""


def read_problem(game):
    """(decision_node, gtp_moves) for the position the solver faces.

    The setup is a linear chain and the decision node is the first node that
    branches; find_problem_node applies that rule and the walk below only
    re-reads the moves along the way to it. `gtp_moves` is the flat
    move list KataGo is queried with, so it must alternate from Black: the
    engine wrapper assigns colours by position, and a sequence that does not
    alternate would be queried as a different position than the file holds.
    """
    size = game.get_size()
    root = game.get_root()
    if root.has_property("AB") or root.has_property("AW"):
        raise Skip("setup stones (AB/AW) — colours are not positional")

    decision_node = find_problem_node(game)
    node = root
    moves = []
    while node is not decision_node:
        node = node[0]                  # single-child by construction, above here
        colour, raw = node.get_raw_move()
        if colour is None:
            raise Skip("a property-only node in the setup chain")
        raw = raw.decode()
        moves.append((colour, "pass" if raw in ("", "tt") else sgf_to_gtp(raw, size)))

    for i, (colour, _move) in enumerate(moves):
        if colour != ("b" if i % 2 == 0 else "w"):
            raise Skip(f"setup colours do not alternate from black at ply {i + 1}")
    return decision_node, [move for _colour, move in moves]


def split_candidates(decision_node, solver_is_black):
    """(answer, [(eval, losing_child)]) among the decision node's children.

    `eval` is the solver's-perspective (winrate, score) the file stores, so the
    losing moves sort by exactly what the client ranks first moves by. Passes
    are dropped: the client cannot offer one, and pass-answered-by-pass would
    make every problem look already covered.
    """
    children = list(decision_node)
    _idx, answer = pick_winning_move(children, solver_is_black)
    if answer is None:
        raise Skip("no child carries an evaluation comment")
    if is_pass(answer):
        raise Skip("the best-evaluated first move is a pass")

    losing = []
    for child in children:
        if child is answer or is_pass(child):
            continue
        ev = to_solver(black_eval(child), solver_is_black)
        if ev is None:
            continue
        losing.append((ev, child))
    if not losing:
        raise Skip("no evaluated losing first move")
    return answer, losing


def needs_losing_line(losing):
    """True when no losing first move has a line that reaches a double pass."""
    return not any(subtree_has_complete_line(child) for _ev, child in losing)


def process_one(path, queries, out_dir):
    """-> a status string. Writes the file only when a line was actually added."""
    if out_dir and os.path.exists(os.path.join(out_dir, os.path.basename(path))):
        # An --out-dir run leaves its input untouched, so the "already covered"
        # test below would re-do every problem a killed run had finished. The
        # written file is the record of that work; treat it as the resume point.
        return "already_written"
    with open(path, "rb") as fh:
        game = sgfmill.sgf.Sgf_game.from_bytes(fh.read())
    if game.get_size() != queries.board_size:
        raise Skip(f"board size {game.get_size()} in a {queries.board_size} run")

    decision_node, moves = read_problem(game)
    solver_is_black = len(moves) % 2 == 0
    _answer, losing = split_candidates(decision_node, solver_is_black)
    if not needs_losing_line(losing):
        return "already_covered"

    # max() keeps the first maximal element, matching the client's strict-`>`
    # scan, so an exact tie on both axes leaves the earlier variation in front.
    _ev, node = max(losing, key=lambda item: item[0])
    _colour, raw = node.get_raw_move()
    move = sgf_to_gtp(raw.decode(), queries.board_size)

    # Default budget, no playouts_fn: every node's own search is PLAYOUTS, so
    # only its top child clears FREE_BRANCH_VISITS. That buys the principal line
    # and nothing else — one query per ply, which is all this pass wants.
    reached_end = generate_variations_directly_to_sgf(
        queries, node, moves + [move], True)
    # The return value and the file's own tree have to agree, and it is the
    # file that every consumer reads. A branch KataGo would not play out to a
    # conclusion is left as it was rather than written back half-explored.
    if not (reached_end and subtree_has_complete_line(node)):
        return "no_conclusion"

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, os.path.basename(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(game.serialise(None).decode())
    os.replace(tmp, path)
    return "added"


def worker(paths, queries, out_dir, counts, lock, state, stop):
    while not stop.is_set():
        try:
            path = paths.get_nowait()
        except queue.Empty:
            return
        try:
            status = process_one(path, queries, out_dir)
        except Skip as exc:
            status = "skipped"
            with lock:
                print(f"[skip] {os.path.basename(path)}: {exc}", flush=True)
        except VariationDepthExceeded as exc:
            status = "runaway"
            with lock:
                print(f"[skip] {os.path.basename(path)}: runaway variation "
                      f"({len(exc.moves)} moves)", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            # One bad problem must cost one problem, not the worker: a thread
            # that dies takes its share of the throughput with it for the rest
            # of a run measured in hours.
            status = "error"
            with lock:
                print(f"[error] {os.path.basename(path)}: {exc}", flush=True)
                traceback.print_exc()
        with lock:
            counts[status] = counts.get(status, 0) + 1
            state["done"] += 1
            if (state["done"] % PROGRESS_EVERY == 0
                    or time.time() - state["last_print"] >= PROGRESS_SECONDS):
                report_progress(counts, state)


def report_progress(counts, state):
    done, total = state["done"], state["total"]
    state["last_print"] = time.time()
    elapsed = time.time() - state["start"]
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else float("inf")
    print(f"  {done}/{total}  {rate:.2f}/s  eta {eta / 3600:.1f}h  "
          + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)


def corpus_paths(size, ids, limit):
    directory = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    if ids:
        paths = [os.path.join(directory, f"{tid}.sgf") for tid in ids]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            sys.exit(f"not in tsumegos/{size}/: "
                     + ", ".join(os.path.basename(p) for p in missing))
    else:
        with os.scandir(directory) as it:
            paths = sorted(e.path for e in it if e.name.endswith(".sgf"))
    return paths[:limit] if limit else paths


def dry_run(paths, size):
    """Count what a real run would do, without starting KataGo."""
    counts = {}
    for i, path in enumerate(paths, 1):
        try:
            with open(path, "rb") as fh:
                game = sgfmill.sgf.Sgf_game.from_bytes(fh.read())
            decision_node, moves = read_problem(game)
            _answer, losing = split_candidates(decision_node, len(moves) % 2 == 0)
            status = "needs_line" if needs_losing_line(losing) else "already_covered"
        except Skip as exc:
            status = "skipped"
            print(f"[skip] {os.path.basename(path)}: {exc}", flush=True)
        except Exception as exc:                                  # noqa: BLE001
            status = "error"
            print(f"[error] {os.path.basename(path)}: {exc}", flush=True)
        counts[status] = counts.get(status, 0) + 1
        if i % 5000 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    print(f"\ntsumegos/{size}/: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("size", type=int, choices=ALL_BOARD_SIZES,
                    help="board size to repair (one of %(choices)s)")
    ap.add_argument("--ids", default=None,
                    help="comma-separated tsumego ids instead of the whole corpus")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many problems have been examined")
    ap.add_argument("--out-dir", default=None,
                    help="write modified SGFs here instead of over the corpus")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"worker threads (default: KataGo batch size x {THREAD_FACTOR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many problems need a line; starts no engine")
    args = ap.parse_args()

    ids = [t.strip() for t in args.ids.split(",") if t.strip()] if args.ids else None
    paths = corpus_paths(args.size, ids, args.limit)
    print(f"examining {len(paths)} problems in tsumegos/{args.size}/")
    if args.dry_run:
        dry_run(paths, args.size)
        return

    controller = None
    try:
        print("Starting KataGo engine...")
        controller = KatagoController()
        controller.start_engine(get_model(DEFAULT_MODEL), board_size=args.size,
                                verbose=DEBUG)
        queries = KatagoQueries(controller, args.size)
        workers = args.workers or (1 if DEBUG else controller.get_batch_size() * THREAD_FACTOR)
        print(f"KataGo started; {workers} worker threads.")

        work = queue.Queue()
        for path in paths:
            work.put(path)
        counts, lock, stop = {}, threading.Lock(), threading.Event()
        state = {"done": 0, "total": len(paths),
                 "start": time.time(), "last_print": time.time()}
        threads = [threading.Thread(target=worker,
                                    args=(work, queries, args.out_dir, counts,
                                          lock, state, stop))
                   for _ in range(workers)]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt; letting workers finish their problem.")
            stop.set()
            for thread in threads:
                thread.join()
        report_progress(counts, state)
    finally:
        if controller:
            print("Stopping KataGo engine...")
            controller.stop_engine()


if __name__ == "__main__":
    main()
