"""Corpus-wide tsumego id integrity scan.

Recomputes every SGF's orientation-invariant canonical hash (the same
`get_orientation_invariant_hash` the generator names files with, taken at the
problem position = after the single-child setup chain) and reports:

  1. MISNAMED files — filename id != canonical hash truncated to 8 hex. These
     break the generator's filename-existence dedup (`tsumego_id_taken`) for
     their position, which is how the same position can later be re-generated
     under its correct id and ship as a duplicate.
  2. DUPLICATE POSITIONS — groups of files (across both sizes) whose FULL
     canonical hashes are identical: the same position, up to D4 symmetry,
     served under more than one id. Violates the symmetry-aware single-copy
     invariant.
  3. TRUNCATION COLLISIONS — different positions sharing the same 8-hex id
     across sizes (the known grandfathered kind; listed for completeness).

Usage:
    ./.venv/bin/python3 -m auxiliary.check_corpus_ids [--sizes 9 13] [--workers N]

Read-only; scans tsumegos/<size>/*.sgf.
"""

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sgfmill import sgf, boards  # noqa: E402

from pipeline.generate_tsumegos import get_orientation_invariant_hash  # noqa: E402
from utils.configurations import ALL_BOARD_SIZES  # noqa: E402


def problem_hash(path):
    """(tsumego_id, full canonical hash) for one SGF, or (id, None) on parse
    failure. The problem position is after the single-child setup chain (the
    variation root), matching where the generator hashed it."""
    tid = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        board = boards.Board(game.get_size())
        root = game.get_root()
        for color, points in (("b", root.get_setup_stones()[0]),
                              ("w", root.get_setup_stones()[1])):
            for (r, c) in points:
                board.play(r, c, color)
        node = root
        while len(node) == 1:
            node = node[0]
            color, move = node.get_move()
            if move is not None:
                board.play(move[0], move[1], color)
        return tid, get_orientation_invariant_hash(board)
    except Exception as e:  # noqa: BLE001 — report, don't abort the sweep
        return tid, f"PARSE_ERROR:{e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sizes", type=int, nargs="*", default=list(ALL_BOARD_SIZES))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    paths = []
    for size in args.sizes:
        d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        with os.scandir(d) as it:
            paths.extend((size, e.path) for e in it
                         if e.name.endswith(".sgf") and e.is_file())
    print(f"Scanning {len(paths)} SGFs with {args.workers} workers ...")

    results = []  # (size, tid, full_hash)
    with Pool(processes=args.workers) as pool:
        for k, ((size, path), (tid, h)) in enumerate(zip(
                paths, pool.imap(problem_hash, (p for _, p in paths),
                                 chunksize=256))):
            results.append((size, tid, h))
            if (k + 1) % 20000 == 0:
                print(f"  {k + 1}/{len(paths)}")

    errors = [(s, t, h) for s, t, h in results if h is None or h.startswith("PARSE_ERROR")]
    ok = [(s, t, h) for s, t, h in results if not (h is None or h.startswith("PARSE_ERROR"))]

    print()
    print(f"parsed {len(ok)} / {len(results)}  (errors: {len(errors)})")
    for s, t, h in errors[:20]:
        print(f"  ERROR {s}/{t}: {h}")

    misnamed = [(s, t, h) for s, t, h in ok if h[:8] != t]
    print()
    print(f"1. MISNAMED (id != canonical hash prefix): {len(misnamed)}")
    for s, t, h in sorted(misnamed):
        print(f"  {s:>2}/{t}.sgf  position hashes to {h[:8]}  (full {h[:16]}...)")

    by_full = defaultdict(list)
    for s, t, h in ok:
        by_full[h].append((s, t))
    dups = {h: v for h, v in by_full.items() if len(v) > 1}
    print()
    print(f"2. DUPLICATE POSITIONS (same full canonical hash, >1 file): {len(dups)}")
    for h, v in sorted(dups.items()):
        print(f"  {h[:16]}...: " + ", ".join(f"{s}/{t}" for s, t in sorted(v)))

    by_tid = defaultdict(list)
    for s, t, h in ok:
        by_tid[t].append((s, h))
    coll = {t: v for t, v in by_tid.items()
            if len(v) > 1 and len({h for _, h in v}) > 1}
    print()
    print(f"3. TRUNCATION COLLISIONS (same id, different positions): {len(coll)}")
    for t, v in sorted(coll.items()):
        print(f"  {t}: " + ", ".join(f"size {s} -> {h[:12]}" for s, h in sorted(v)))


if __name__ == "__main__":
    main()
