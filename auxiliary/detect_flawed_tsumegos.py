import os
import sys
import re
import glob
from sgfmill import sgf

import shutil

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "tsumegos")
FAILED_DIR = os.path.join(PROJECT_ROOT, "failed_tsumegos")

def parse_comment(comment):
    # Expecting format like "99.9%/1.6"
    if not comment:
        return None, None
        
    # Look for the pattern. Note: sometimes comments might have extra text? 
    # The current generator produces "99.9%/1.6" or similar.
    match = re.search(r"([\d\.]+)%/([-\d\.]+)", comment)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            return None, None
    return None, None

def check_tsumego(file_path):
    try:
        with open(file_path, "rb") as f:
            content = f.read()
            # Basic validation to ensure it looks like SGF before parsing to avoid massive logs on binary garbage
            if not content.strip().startswith(b"("):
                 return None
            game = sgf.Sgf_game.from_bytes(content)
    except Exception as e:
        # print(f"Failed to parse {file_path}: {e}")
        return None

    mainline = game.get_main_sequence()
    
    # Defaults
    last_winrate = None
    last_score = None
    
    stays_positive = True
    
    # Iterate through all nodes in the mainline
    for node in mainline:
        try:
            comment = node.get("C")
        except KeyError:
            comment = None
            
        if comment:
            winrate, score = parse_comment(comment)
            if winrate is not None and score is not None:
                last_winrate = winrate
                last_score = score
                
                # Check if Black is winning (Winrate > 50% and Score > 0)
                is_winning = winrate > 50.0 and score > 0.0
                if not is_winning:
                    stays_positive = False

    # Determine validation results
    mainline_wins = True
    if last_winrate is not None:
        mainline_wins = last_winrate > 50.0 and last_score > 0.0
    
    return mainline_wins, stays_positive

def main():
    print(f"Scanning SGF files in {TSUMEGOS_DIR}...")
    sgf_files = glob.glob(os.path.join(TSUMEGOS_DIR, "*.sgf"))
    print(f"Found {len(sgf_files)} SGF files.")
    
    if not os.path.exists(FAILED_DIR):
        os.makedirs(FAILED_DIR)
        print(f"Created failed directory: {FAILED_DIR}")

    flawed_result_count = 0
    flawed_intermediate_count = 0
    moved_count = 0
    
    processed = 0
    
    import time
    start_time = time.time()

    for file_path in sgf_files:
        res = check_tsumego(file_path)
        if res is None:
            continue
            
        wins, stays_positive = res
        filename = os.path.basename(file_path)
        is_flawed = False
        
        if not wins:
            flawed_result_count += 1
            is_flawed = True
        
        if not stays_positive:
            flawed_intermediate_count += 1
            is_flawed = True
            
        if is_flawed:
            dest_path = os.path.join(FAILED_DIR, filename)
            try:
                shutil.move(file_path, dest_path)
                moved_count += 1
            except Exception as e:
                print(f"Failed to move {filename}: {e}")
            
        processed += 1
        if processed % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {processed} files in {elapsed:.1f}s...")

    print("\n--- Analysis Complete ---")
    print(f"Total files checked: {processed}")
    print(f"Files where mainline final result is NOT a win: {flawed_result_count}")
    print(f"Files where mainline drops below winning criteria at some point: {flawed_intermediate_count}")
    print(f"Total files moved to {FAILED_DIR}: {moved_count}")

if __name__ == "__main__":
    main()
