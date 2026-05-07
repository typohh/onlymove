"""
Identify 13x13 tsumegos that place a stone on the 1st or 2nd line within
the first 4 moves of the main line, and move them into tsumegos/broken/13/.

Such early-edge stones are unrealistic in real games and almost always
indicate a malformed puzzle setup.
"""

import argparse
import os
import shutil
import sys

from sgfmill import sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "tsumegos", "13")
DEST_DIR = os.path.join(PROJECT_ROOT, "tsumegos", "broken", "13")
BOARD_SIZE = 13
NUM_OPENING_MOVES = 4


def is_edge_move(move, board_size=BOARD_SIZE):
    """True if (row, col) is on the 1st or 2nd line of the board."""
    row, col = move
    edge_dist = min(row, board_size - 1 - row, col, board_size - 1 - col)
    return edge_dist <= 1


def first_edge_move(path):
    """Return (move_index, color, (row, col)) of the first edge move within
    the opening, or None if no early edge move exists. Walks main line only."""
    with open(path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    if game.get_size() != BOARD_SIZE:
        return None

    node = game.get_root()
    move_idx = 0
    while move_idx < NUM_OPENING_MOVES:
        children = list(node)
        if not children:
            return None
        node = children[0]
        color, move = node.get_move()
        if color is None:
            continue
        move_idx += 1
        if move is None:
            continue
        if is_edge_move(move):
            return move_idx, color, move
    return None


def coord_to_label(move):
    """Convert sgfmill (row, col) to human-readable label like 'A1' (row 1
    is bottom). Skip 'I' to match standard Go notation."""
    row, col = move
    letters = "ABCDEFGHJKLMNOPQRST"
    return f"{letters[col]}{row + 1}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="List offending files but do not move them")
    parser.add_argument("--source", default=SOURCE_DIR,
                        help=f"Source directory (default: {SOURCE_DIR})")
    parser.add_argument("--dest", default=DEST_DIR,
                        help=f"Destination directory (default: {DEST_DIR})")
    args = parser.parse_args()

    if not os.path.isdir(args.source):
        print(f"Source directory not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        os.path.join(args.source, name)
        for name in os.listdir(args.source)
        if name.endswith(".sgf")
    )
    total = len(files)
    print(f"Scanning {total} SGF files in {args.source}...")

    offenders = []
    errors = []
    for path in files:
        try:
            result = first_edge_move(path)
        except Exception as e:
            errors.append((path, f"{type(e).__name__}: {e}"))
            continue
        if result is not None:
            offenders.append((path, result))

    print(f"\nFound {len(offenders)} offending file(s):")
    for path, (idx, color, move) in offenders:
        print(f"  {os.path.basename(path)}  (move {idx} {color.upper()}{coord_to_label(move)})")

    if errors:
        print(f"\n{len(errors)} parse error(s):")
        for path, err in errors[:10]:
            print(f"  {os.path.basename(path)}: {err}")
        if len(errors) > 10:
            print(f"  ... ({len(errors) - 10} more)")

    if args.dry_run:
        print("\nDry run: no files moved.")
        return

    if not offenders:
        return

    os.makedirs(args.dest, exist_ok=True)
    moved = 0
    for path, _ in offenders:
        dest_path = os.path.join(args.dest, os.path.basename(path))
        shutil.move(path, dest_path)
        moved += 1
    print(f"\nMoved {moved} file(s) to {args.dest}")


if __name__ == "__main__":
    main()
