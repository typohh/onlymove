"""
Check that existing tsumegos contain at least one complete variation —
a path from the root to a leaf whose last two moves are both passes.

The current generator (pipeline/generate_tsumegos.py) builds two such
complete lines (regular + CALM_UTILITY_OVERRIDE), but older runs may have
produced files that never reached two consecutive passes on any branch.
This script scans tsumegos/<board_size>/ and reports incomplete files.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from sgfmill import sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Some generated trees are deep; allow plenty of headroom for recursion.
sys.setrecursionlimit(10000)


def _walk(node, prev_was_pass, prev_prev_was_pass):
    """DFS for a leaf reached by two consecutive passes.

    prev_was_pass: True iff the most recent ancestor move was a pass.
    prev_prev_was_pass: True iff the move before that was a pass.
    Setup-only nodes (no move) don't update these.
    """
    color, move = node.get_move()
    if color is not None:
        prev_prev_was_pass = prev_was_pass
        prev_was_pass = move is None  # None == pass in sgfmill (incl. "tt" on size<=19)

    children = list(node)
    if not children:
        return prev_was_pass and prev_prev_was_pass

    for child in children:
        if _walk(child, prev_was_pass, prev_prev_was_pass):
            return True
    return False


def has_complete_variation(game):
    return _walk(game.get_root(), False, False)


def check_file(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        game = sgf.Sgf_game.from_bytes(data)
    except Exception as e:
        return path, "error", f"{type(e).__name__}: {e}"
    return path, "ok" if has_complete_variation(game) else "incomplete", None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, default=9,
                        help="Tsumego board size folder under tsumegos/ (default: 9)")
    parser.add_argument("--dir", default=None,
                        help="Override directory to scan (default: tsumegos/<board_size>)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes")
    parser.add_argument("--list-incomplete", action="store_true",
                        help="Print every incomplete filename (otherwise only a sample)")
    parser.add_argument("--sample", type=int, default=20,
                        help="How many incomplete filenames to print when --list-incomplete is off")
    args = parser.parse_args()

    target_dir = args.dir or os.path.join(PROJECT_ROOT, "tsumegos", str(args.board_size))
    if not os.path.isdir(target_dir):
        print(f"Directory not found: {target_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        os.path.join(target_dir, name)
        for name in os.listdir(target_dir)
        if name.endswith(".sgf")
    )
    total = len(files)
    print(f"Scanning {total} SGF files in {target_dir} with {args.workers} workers...")

    incomplete = []
    errors = []
    ok_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_file, p) for p in files]
        progress_step = max(1, total // 50)
        for i, fut in enumerate(as_completed(futures), 1):
            path, status, err = fut.result()
            if status == "ok":
                ok_count += 1
            elif status == "incomplete":
                incomplete.append(path)
            else:
                errors.append((path, err))
            if i % progress_step == 0 or i == total:
                print(f"  {i}/{total} ({100*i/total:.1f}%) — ok={ok_count}, "
                      f"incomplete={len(incomplete)}, errors={len(errors)}",
                      end="\r", flush=True)
    print()

    print()
    print("=== Results ===")
    print(f"Total:       {total}")
    print(f"Complete:    {ok_count}")
    print(f"Incomplete:  {len(incomplete)}")
    print(f"Parse errors:{len(errors)}")

    if incomplete:
        to_show = incomplete if args.list_incomplete else incomplete[: args.sample]
        label = "All incomplete files" if args.list_incomplete else f"First {len(to_show)} incomplete files"
        print(f"\n{label}:")
        for p in to_show:
            print(f"  {os.path.basename(p)}")
        if not args.list_incomplete and len(incomplete) > len(to_show):
            print(f"  ... ({len(incomplete) - len(to_show)} more)")

    if errors:
        print("\nParse errors (first 10):")
        for p, err in errors[:10]:
            print(f"  {os.path.basename(p)}: {err}")


if __name__ == "__main__":
    main()
