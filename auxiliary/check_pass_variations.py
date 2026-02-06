import os

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sgfmill import sgf


def variation_ends_in_pass(node, primary):
    """
    Iteratively checks the main variation from this node to see if it ends in a pass.
    """
    current_node = node
    current_primary = primary

    while True:
        # A leaf node has no children.
        if len(current_node) == 0:
            # Per your instruction, checking the raw move value for a pass.
            _color, raw_move = current_node.get_raw_move()
            if raw_move is not None and raw_move == b"tt":
                return True, current_primary
            return False, False

        # Per your instruction, only follow the first child node.
        # However, we must preserve the 'primary' flag logic.
        # The original code would set primary = False after the first child was visited in the recursive call.
        # But wait, original code:
        # for n in node:
        #     found, is_primary = variation_ends_in_pass(n, primary)
        #     if found: return found, is_primary
        #     primary = False
        # This actually means it checks ALL children, but only the first child (and ITS descendants) can be 'primary'.
        # Since the instruction said "only follow the first child node", the loop in the original code was redundant if we only care about the first child.
        # Let's re-read the original:
        # 17:     # Per your instruction, only follow the first child node.
        # 18:     for n in node:
        # 19:         found, is_primary = variation_ends_in_pass(n, primary)
        # 20:         if found:
        # 21:             return found, is_primary
        # 22:         primary = False
        
        # If we ONLY follow the first child, then:
        next_node = current_node[0]
        current_node = next_node
        # current_primary stays the same because we only follow the first child of the first child, etc.



def analyze_sgf_directory(directory):
    """
    Analyzes all SGF files in a directory.
    """
    print(f"Analyzing SGF files in '{directory}'...", flush=True)

    files_with_pass = 0
    files_with_primary = 0
    total_files = 0
    files_to_move = []
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

                try:
                    # Per your instruction, assuming one main variation from the root.
                    if len(root) > 0:
                        found, primary = variation_ends_in_pass(root[0], True)
                        if found:
                            files_with_pass += 1
                            if primary:
                                files_with_primary += 1
                        
                        if not primary:
                            files_to_move.append(entry.path)
                except Exception as e:
                    print(f"\nCrash occurred while processing: {entry.path}")
                    raise e

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.", flush=True)
        return []

    if not total_files:
        print(f"No .sgf files found in '{directory}'.", flush=True)
        return []

    print("\nAnalysis complete.", flush=True)
    percentage = (files_with_pass / total_files) * 100
    print(f"{files_with_primary} {files_with_pass} out of {total_files} files ({percentage:.2f}%) contain a variation ending in a pass.", flush=True)

    return files_to_move


def main():
    target_directory = os.path.join(PROJECT_ROOT, "tsumegos")
    old_directory = os.path.join(PROJECT_ROOT, "old_tsumegos")
    
    files_to_move = analyze_sgf_directory(target_directory)
    
    if files_to_move:
        print(f"\nMoving {len(files_to_move)} files to '{old_directory}'...", flush=True)
        if not os.path.exists(old_directory):
            os.makedirs(old_directory)
        
        for file_path in files_to_move:
            filename = os.path.basename(file_path)
            target_path = os.path.join(old_directory, filename)
            try:
                os.rename(file_path, target_path)
            except Exception as e:
                print(f"Error moving {filename}: {e}", flush=True)
        
        print("Move complete.", flush=True)


if __name__ == "__main__":
    main()
