import json
import os
import subprocess
import random
import sys
import time
import re

from katago_controller import KATAGO_PATH, CONFIG_FILE, RL_MODEL_PATH, SL_MODEL_PATH


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
# CONFIG_FILE = os.path.join(BASE_DIR, "analysis.cfg")
# RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")
# SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")


# --- Helper Function for Query Generation ---

def generate_random_query(query_id: str) -> dict:
    """
    Generates a query with 2 black and 2 white stones placed randomly
    on a 9x9 board.
    """
    board_size = 9
    cols = "ABCDEFGHJ"  # No 'I' in Go coordinates
    all_coords = [f"{c}{r}" for c in cols for r in range(1, board_size + 1)]
    try:
        positions = random.sample(all_coords, 4)
    except ValueError:
        return {}

    initial_stones = [
        ["B", positions[0]], ["B", positions[1]],
        ["W", positions[2]], ["W", positions[3]],
    ]

    return {
        "id": query_id, "moves": [], "initialStones": initial_stones,
        "rules": "tromp-taylor", "komi": 7.5, "boardXSize": board_size,
        "boardYSize": board_size, "maxVisits": 1, "includePolicy": True,
        "overrideSettings": {"humanSLProfile": "rank_2d", "ignorePreRootHistory": False, "rootNumSymmetriesToSample": 1
                             }
    }


def update_config_file(config_path: str, optimal_size: int):
    """
    Updates the nnMaxBatchSize and numAnalysisThreads in the given config file.
    """
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        keys_to_update = {"nnMaxBatchSize", "numAnalysisThreads"}
        keys_found = set()

        for line in lines:
            stripped_line = line.strip()
            # Find lines that are not comments and define one of our keys
            if not stripped_line.startswith('#'):
                match = re.match(r"^\s*(\w+)\s*=", stripped_line)
                if match:
                    key = match.group(1)
                    if key in keys_to_update:
                        updated_lines.append(f"{key} = {optimal_size}\n")
                        keys_found.add(key)
                        continue  # Skip appending the old line
            # If the line doesn't match, add it as-is
            updated_lines.append(line)

        # If any key was not found in the file, add it to the end
        for key in keys_to_update:
            if key not in keys_found:
                updated_lines.append(f"\n# Added by performance test\n{key} = {optimal_size}\n")

        with open(config_path, 'w') as f:
            f.writelines(updated_lines)

        print(f"\n--- Configuration Updated ---")
        print(f"Successfully set nnMaxBatchSize and numAnalysisThreads to {optimal_size} in:")
        print(f"'{config_path}'")

    except Exception as e:
        print(f"\nError: Failed to update config file: {e}", file=sys.stderr)


# --- Main Test Logic ---

def main():
    """
    Main function to run the batch size test.
    """
    for path in [KATAGO_PATH, CONFIG_FILE, RL_MODEL_PATH, SL_MODEL_PATH]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'", file=sys.stderr)
            sys.exit(1)

    batch_size = 1
    best_batch_size = 0
    best_queries_per_second = 0

    print("--- Starting KataGo Throughput Test ---")
    while True:
        # One-line status message for the current test
        print(f"Testing Batch Size: {batch_size:<4} ... ", end='', flush=True)

        command = [
            KATAGO_PATH, "analysis",
            "-config", CONFIG_FILE,
            "-override-config", f"numAnalysisThreads={batch_size},nnMaxBatchSize={batch_size},numSearchThreadsPerAnalysisThread=1",
            "-model", RL_MODEL_PATH, "-human-model", SL_MODEL_PATH
        ]

        process = None
        try:
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1
            )
            num_queries_per_round = 2 * batch_size

            # Round 1: Warm-up
            warmup_queries = [generate_random_query(f"w_{i}") for i in range(max(1,int(num_queries_per_round/4)))]
            for query in warmup_queries:
                process.stdin.write(json.dumps(query) + "\n")
            process.stdin.flush()
            for i in range(max(1,int(num_queries_per_round/4))):
                if not process.stdout.readline():
                    raise RuntimeError("exited during warm-up")

            # Round 2: Timed Test
            timed_queries = [generate_random_query(f"t_{i}") for i in range(num_queries_per_round)]
            start_time = time.perf_counter()
            for query in timed_queries:
                process.stdin.write(json.dumps(query) + "\n")
            process.stdin.flush()
            for i in range(num_queries_per_round):
                if not process.stdout.readline():
                    raise RuntimeError("exited during timed test")
            end_time = time.perf_counter()

            queries_per_second = num_queries_per_round / (end_time - start_time)
            print(f"SUCCESS | Throughput: {queries_per_second:.2f} q/s")
            if queries_per_second < 0.9 * best_queries_per_second:
                print("Signficant throughput drop.")
                break
            if queries_per_second > best_queries_per_second:
                best_batch_size = batch_size
                best_queries_per_second = queries_per_second

            batch_size *= 2


        except (RuntimeError, BrokenPipeError) as e:
            print(f"FAILURE | KataGo process {e}.")
            if process and process.stderr:
                stderr = process.stderr.read()
                if "out of memory" in stderr.lower():
                    print("          (Likely cause: Out of VRAM)")
            break
        except Exception as e:
            print(f"FAILURE | An unexpected Python error occurred: {e}")
            break
        finally:
            if process:
                process.terminate()
                process.communicate()

    print("\n--- Test Complete ---")
    if best_queries_per_second > 0:
        print(f"The optimal throughput was achieved with batch size: {best_batch_size}")
        update_config_file(CONFIG_FILE, best_batch_size)
    else:
        print("Could not complete a single run. Please check your KataGo setup and file paths.")


if __name__ == "__main__":
    main()
