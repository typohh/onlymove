import os
from sgfmill import sgf


def variation_ends_in_pass(node, primary):
    """
    Recursively checks the main variation from this node to see if it ends in a pass.
    """
    # A leaf node has no children.
    if len(node) == 0:
        # Per your instruction, checking the raw move value for a pass.
        _color, raw_move = node.get_raw_move()
        if raw_move is not None and raw_move == b"tt":
            return True, primary
        return False, False

    # Per your instruction, only follow the first child node.
    for n in node:
        found, is_primary = variation_ends_in_pass(n, primary)
        if found:
            return found, is_primary
        primary = False

    return False, False


def analyze_sgf_directory(directory):
    """
    Analyzes all SGF files in a directory.
    """
    print(f"Analyzing SGF files in '{directory}'...", flush=True)

    files_with_pass = 0
    files_with_primary = 0
    total_files = 0
    # Per your instruction, restoring the power-of-two progress prints.
    next_milestone = 1

    try:
        with os.scandir(directory) as it:
            for entry in it:
                if not (entry.is_file() and entry.name.endswith('.sgf')):
                    continue

                total_files += 1

                if total_files == next_milestone:
                    percentage = (files_with_pass / total_files) * 100
                    print(f"{files_with_primary} {files_with_pass} out of {total_files} files ({percentage:.2f}%) contain a variation ending in a pass.", flush=True)
                    next_milestone *= 2

                with open(entry.path, 'rb') as f:
                    try:
                        game = sgf.Sgf_game.from_bytes(f.read())
                    except Exception:
                        continue

                root = game.get_root()

                # Per your instruction, assuming one main variation from the root.
                if len(root) > 0:
                    found, primary = variation_ends_in_pass(root[0], True)
                    if found:
                        files_with_pass += 1
                        if primary:
                            files_with_primary += 1

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.", flush=True)
        return

    if not total_files:
        print(f"No .sgf files found in '{directory}'.", flush=True)
        return

    print("\nAnalysis complete.", flush=True)
    percentage = (files_with_pass / total_files) * 100
    print(f"{files_with_primary} {files_with_pass} out of {total_files} files ({percentage:.2f}%) contain a variation ending in a pass.", flush=True)


def main():
    target_directory = "tsumegos"
    analyze_sgf_directory(target_directory)


if __name__ == "__main__":
    main()
