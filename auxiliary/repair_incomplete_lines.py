"""Play the winning line to the end of the game for tsumegos whose principal
variation stops early.

auxiliary.check_tsumego_sequences reports problems whose winning sequence — the
winning first move followed by the first child at every step — never reaches the
double pass that ends a game.  Two shapes exist in the corpus:

  * the line ends on a SINGLE pass.  Variation generation used to stop there
    (the two-consecutive-pass guard came in with 56655d8887, 2025-09-02), so the
    opponent's reply to the pass was never queried.  One move completes these.
  * the line ends on a real move, because the KataGo query behind it came back
    without moveInfos and generate_variations_directly_to_sgf returned quietly.
    These need genuine play-out.

Both are repaired the same way: re-run the generator's own
generate_variations_directly_to_sgf from the truncated leaf, as the primary
line, so the completed tail is built by the same code, at the same playout
count, with the same "{winrate}%/{score}" comments.  Nothing else about the file
is touched: sgfmill round-trips these SGFs byte-for-byte, so a repaired file
differs from the original only by the appended nodes.

ONLY the main branch is completed — no side variations are added.  The tail
being repaired is the settled end of the game (for the single-pass files, the
game has already ended), so candidate subtrees there would analyse moves nobody
plays.

Before overwriting, every file must still satisfy:
  * the setup move sequence is unchanged,
  * the problem node's first-move variations are unchanged in content AND order
    (package_for_pwa addresses per-cell embeddings by SGF variation index),
  * the previously existing part of the winning line is unchanged,
  * the winning line now ends the game.
A file failing any of these is left exactly as it was.  Writes go through a temp
file + os.replace, so an interrupted run cannot leave a half-written SGF.

The repair APPENDS variations, so it cannot change a problem's identity: the
orientation-invariant id is hashed from the setup position alone, which the
first check above pins.

Usage:
    ./.venv/bin/python3 -m auxiliary.repair_incomplete_lines --dry-run
    ./.venv/bin/python3 -m auxiliary.repair_incomplete_lines --sizes 9
    ./.venv/bin/python3 -m auxiliary.repair_incomplete_lines --dir /tmp/copy
"""

import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from sgfmill import sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.check_tsumego_sequences import (  # noqa: E402
    ends_the_game,
    find_problem_node,
    pick_winning_move,
    principal_line,
)
from utils.add_variations import (  # noqa: E402
    VariationDepthExceeded,
    generate_variations_directly_to_sgf,
)
from utils.configurations import ALL_BOARD_SIZES, DEBUG, DEFAULT_MODEL  # noqa: E402
from utils.coords import sgf_to_gtp, sgfmill_point_to_sgf  # noqa: E402
from utils.katago_controller import KatagoController  # noqa: E402
from utils.katago_queries import KatagoQueries  # noqa: E402
from utils.models import get_model  # noqa: E402

print_lock = threading.Lock()

# generate_variations_directly_to_sgf breaks out of its move loop after the first
# (primary) child once the cumulative playout share drops below MIN_PLAYOUT_PERC.
# Starting below that threshold therefore extends the main branch to the double
# pass and adds NOTHING else, which is exactly the repair wanted here.
START_PLAYOUT_PERC = 0.0


def gtp_move(node, size):
    """The node's move in KataGo coordinates, "pass", or None for a setup node."""
    colour, point = node.get_move()
    if colour is None:
        return None
    if point is None:
        return "pass"
    return sgf_to_gtp(sgfmill_point_to_sgf(point, size), size)


def describe(game):
    """(problem node, winning line, moves from the root to the line's leaf).

    The move list is what KataGo needs to analyse the position at the leaf, so it
    spans the setup game and the whole winning line.
    """
    size = game.get_size()
    moves = []
    node = game.get_root()
    move = gtp_move(node, size)
    if move:
        moves.append(move)
    while len(node) == 1:
        node = node[0]
        move = gtp_move(node, size)
        if move:
            moves.append(move)
    problem = node

    children = list(problem)
    solver_colour = next((c.get_move()[0] for c in children if c.get_move()[0] is not None), None)
    winner_idx, winner = pick_winning_move(children, solver_colour == "b")
    if winner is None:
        return problem, None, None, None
    line = principal_line(winner)
    for n in line:
        move = gtp_move(n, size)
        if move:
            moves.append(move)
    return problem, winner_idx, line, moves


def fingerprint(game):
    """The parts of the file the repair must not disturb."""
    size = game.get_size()
    setup = []
    node = game.get_root()
    while len(node) == 1:
        node = node[0]
        setup.append(gtp_move(node, size))
    problem = node
    first_moves = [gtp_move(c, size) for c in problem]
    _, winner_idx, line, _ = describe(game)
    return {
        "setup": setup,
        "first_moves": first_moves,
        "winner_idx": winner_idx,
        "line": [gtp_move(n, size) for n in line] if line else None,
    }


def repair_one(path, queries, stats):
    with open(path, "rb") as f:
        raw = f.read()
    game = sgf.Sgf_game.from_bytes(raw)
    size = game.get_size()
    if size != queries.board_size:
        return "skipped-size"

    before = fingerprint(game)
    problem, winner_idx, line, moves = describe(game)
    if line is None:
        return "no-winning-move"
    if winner_idx != 0:
        # The corpus contract is that variation_nodes[0] IS the winning branch;
        # repairing a file that violates it would bake the disagreement in.
        return "winner-not-first-child"
    if ends_the_game(line):
        return "already-complete"

    leaf = line[-1]
    generate_variations_directly_to_sgf(queries, leaf, moves, START_PLAYOUT_PERC, True, None)

    after = fingerprint(game)
    if after["setup"] != before["setup"]:
        return "rejected-setup-changed"
    if after["first_moves"] != before["first_moves"]:
        return "rejected-first-moves-changed"
    if after["line"][:len(before["line"])] != before["line"]:
        return "rejected-line-rewritten"
    _, _, new_line, _ = describe(game)
    if not ends_the_game(new_line):
        return "still-incomplete"

    tmp = path + ".repair.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(game.serialise(None).decode())
    os.replace(tmp, path)
    stats["added_nodes"] += len(new_line) - len(line)
    return "repaired"


def worker(path, queries, stats, counts):
    try:
        outcome = repair_one(path, queries, stats)
    except VariationDepthExceeded as e:
        outcome = "failed-runaway-depth"
        with print_lock:
            print(f"  [skip] {os.path.basename(path)}: runaway line ({len(e.moves)} moves)")
    except Exception as e:
        outcome = "failed-error"
        with print_lock:
            print(f"  [skip] {os.path.basename(path)}: {type(e).__name__}: {e}")
            if DEBUG:
                traceback.print_exc()
    with print_lock:
        counts[outcome] = counts.get(outcome, 0) + 1
        done = sum(counts.values())
        if done % 10 == 0 or outcome not in ("repaired", "already-complete"):
            print(f"  {done} processed: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    return outcome


def scan_one(path):
    """(path, size, winner_idx, line length) when the winning line is incomplete."""
    try:
        with open(path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        _, winner_idx, line, _ = describe(game)
        if line is None or ends_the_game(line):
            return None
        return (path, game.get_size(), winner_idx, len(line), None)
    except Exception as e:
        return (path, None, None, None, f"{type(e).__name__}: {e}")


def find_incomplete(paths, workers):
    """The subset whose winning line does not end the game, grouped by board size.

    Parsing dominates, so this fans out over processes — the corpus is 160k+
    files and a serial scan would take longer than the repair itself.
    """
    by_size = {}
    unreadable = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(scan_one, paths, chunksize=64):
            if result is None:
                continue
            path, size, winner_idx, length, error = result
            if error:
                unreadable.append((path, error))
            else:
                by_size.setdefault(size, []).append((path, winner_idx, length))
    for entries in by_size.values():
        entries.sort()
    return by_size, unreadable


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(ALL_BOARD_SIZES),
                        choices=list(ALL_BOARD_SIZES))
    parser.add_argument("--dir", default=None,
                        help="Scan this directory instead of tsumegos/<size> (board size "
                             "is read from each SGF, so a mixed directory is fine)")
    parser.add_argument("--files", default=None,
                        help="Read the paths to check from this file, one per line")
    parser.add_argument("--limit", type=int, default=None, help="Repair at most this many files")
    parser.add_argument("--workers", type=int, default=None,
                        help="Threads per engine (default: KataGo batch size x3, as the generator)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be repaired; never starts KataGo")
    args = parser.parse_args()

    if args.files:
        paths = [l.strip() for l in open(args.files) if l.strip()]
    elif args.dir:
        paths = sorted(os.path.join(args.dir, n) for n in os.listdir(args.dir)
                       if n.endswith(".sgf"))
    else:
        paths = []
        for size in args.sizes:
            d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
            if os.path.isdir(d):
                paths.extend(sorted(os.path.join(d, n) for n in os.listdir(d)
                                    if n.endswith(".sgf")))

    scan_workers = args.workers or os.cpu_count() or 1
    print(f"Checking {len(paths)} SGF files for incomplete winning lines "
          f"({scan_workers} workers)...")
    by_size, unreadable = find_incomplete(paths, scan_workers)
    total = sum(len(v) for v in by_size.values())
    for size in sorted(by_size):
        entries = by_size[size]
        lens = sorted(e[2] for e in entries)
        not_first = sum(1 for e in entries if e[1] != 0)
        print(f"  size {size}: {len(entries)} incomplete "
              f"(winning-line length min {lens[0]}, median {lens[len(lens) // 2]}, max {lens[-1]}"
              + (f"; {not_first} where the winner is not variation_nodes[0] — those are SKIPPED"
                 if not_first else "") + ")")
    if unreadable:
        print(f"  {len(unreadable)} unreadable: " + ", ".join(os.path.basename(p) for p, _ in unreadable[:5]))
    if not total:
        print("Nothing to repair.")
        return
    if args.dry_run:
        print("\n--dry-run: files that would be repaired")
        for size in sorted(by_size):
            for path, _, _ in by_size[size]:
                print(f"  {os.path.relpath(path, PROJECT_ROOT)}")
        return

    stats = {"added_nodes": 0}
    grand = {}
    for size in sorted(by_size):
        targets = [p for p, _, _ in by_size[size]]
        if args.limit:
            targets = targets[:args.limit]
        print(f"\nStarting KataGo for {size}x{size} ({len(targets)} files)...")
        controller = KatagoController()
        try:
            controller.start_engine(get_model(DEFAULT_MODEL), board_size=size, verbose=DEBUG)
            queries = KatagoQueries(controller, size)
            workers = args.workers or int(controller.get_batch_size() * 3)
            counts = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda p: worker(p, queries, stats, counts), targets))
            for k, v in counts.items():
                grand[k] = grand.get(k, 0) + v
        finally:
            controller.stop_engine()

    print("\n=== Result ===")
    for k, v in sorted(grand.items()):
        print(f"  {k}: {v}")
    if grand.get("repaired"):
        print(f"  nodes appended to winning lines: {stats['added_nodes']} "
              f"({stats['added_nodes'] / grand['repaired']:.1f} per repaired file)")
    print("\nRe-verify with: ./.venv/bin/python3 -m auxiliary.check_tsumego_sequences")


if __name__ == "__main__":
    main()
