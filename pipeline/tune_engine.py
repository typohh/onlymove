import json
import os
import subprocess
import random
import sys
import threading
import time
import re
import argparse
import traceback

from utils.katago_controller import KATAGO_PATH, CONFIG_FILE
from utils.models import get_model, MODEL_9X9, MODEL_HUMAN, MODEL_BEST_SMALL


verbose = False
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
        "boardYSize": board_size, "maxVisits": 1,
        # "includePolicy": True, "overrideSettings": {"humanSLProfile": "rank_2d", "ignorePreRootHistory": False, "rootNumSymmetriesToSample": 1 }
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


def run_test(batch_size, best_queries_per_second):
    # One-line status message for the current test
    print(f"Testing Batch Size: {batch_size:<4} ... ", end='', flush=True)

    command = [
        KATAGO_PATH, "analysis",
        "-config", CONFIG_FILE,
        "-override-config", f"numAnalysisThreads={batch_size},nnMaxBatchSize={batch_size},numSearchThreadsPerAnalysisThread=1",
        # "-model", get_model(MODEL_9X9), "-human-model", get_model(MODEL_HUMAN)
        "-model", get_model(MODEL_9X9)
        # "-model", get_model(MODEL_BEST_SMALL)
    ]

    process = None
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1
        )

        if verbose:
            def read_stderr(process):
                for line in process.stderr:
                    print(f"KataGo STDERR: {line.strip()}")

            stderr_thread = threading.Thread(target=read_stderr, args=(process,))
            stderr_thread.daemon = True  # Allow main program to exit even if thread is still running
            stderr_thread.start()

        num_queries_per_round = 2 * batch_size

        # Round 1: Warm-up
        warmup_queries = [generate_random_query(f"w_{i}") for i in range(max(1, int(num_queries_per_round / 4)))]
        for query in warmup_queries:
            process.stdin.write(json.dumps(query) + "\n")
        process.stdin.flush()
        for _ in range(len(warmup_queries)):
            if not process.stdout.readline():
                raise RuntimeError("exited during warm-up")

        # Round 2: Timed Test
        timed_queries = [generate_random_query(f"t_{i}") for i in range(num_queries_per_round)]
        start_time = time.perf_counter()
        for query in timed_queries:
            process.stdin.write(json.dumps(query) + "\n")
        process.stdin.flush()

        predicted_total_time = num_queries_per_round / best_queries_per_second
        predicted_time_limit = predicted_total_time * 1.5

        has_stalled = False
        for i in range(num_queries_per_round):
            if predicted_time_limit > 0:
                elapsed_time = time.perf_counter() - start_time
                if elapsed_time > predicted_time_limit:
                    print(f"STALLED | Time ({elapsed_time:.2f}s) exceeded 150% of predicted time ({predicted_total_time:.2f}s). Halting.")
                    has_stalled = True
                    break
            if not process.stdout.readline():
                raise RuntimeError("exited during timed test")

        if has_stalled:
            return 0

        end_time = time.perf_counter()
        queries_per_second = num_queries_per_round / (end_time - start_time)
        print(f"SUCCESS | Throughput: {queries_per_second:.2f} q/s")

        return queries_per_second

    except (RuntimeError, BrokenPipeError) as e:
        print(f"FAILURE | KataGo process {e}.")
        if process and process.stderr:
            stderr = process.stderr.read()
            if "out of memory" in stderr.lower():
                print("          (Likely cause: Out of VRAM)")
        return 0
    except Exception as e:
        print(f"FAILURE | An unexpected Python error occurred: {e}")
        traceback.print_exc()
        return 0
    finally:
        if process:
            process.terminate()
            process.communicate()


def run_test_in_thread(batch_size, best_queries_per_second):
    result = [0]  # Use list to store result from thread

    def thread_target():
        result[0] = run_test(batch_size, best_queries_per_second)

    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join(timeout=5 + 2 * batch_size / best_queries_per_second if best_queries_per_second > 0 else None)
    if thread.is_alive():
        return 0  # Abort if thread is still running after timeout
    return result[0]


# --- Main Test Logic ---

def main():
    """
    Main function to run the batch size test.
    """
    parser = argparse.ArgumentParser(description='KataGo batch size optimization')
    parser.add_argument('--batchsize', type=int, help='Set specific batch size and skip testing')
    parser.add_argument('--verbose', action='store_true', help='set verbose mode')
    args = parser.parse_args()

    if args.verbose:
        global verbose
        verbose = True

    for path in [KATAGO_PATH, CONFIG_FILE, get_model(MODEL_9X9), get_model(MODEL_HUMAN)]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'", file=sys.stderr)
            sys.exit(1)

    if args.batchsize is not None:
        print(f"Setting batch size to {args.batchsize} from command line argument")
        update_config_file(CONFIG_FILE, args.batchsize)
        return

    batch_size = 1
    best_queries_per_second = 0.01
    all_results = []

    print("--- Starting KataGo Throughput Test ---")
    while True:

        queries_per_second = run_test_in_thread(batch_size, best_queries_per_second)
        all_results.append((batch_size, queries_per_second))

        if queries_per_second == 0:
            break

        if queries_per_second > best_queries_per_second:
            best_queries_per_second = queries_per_second

        batch_size *= 2

    print("\n--- Test Complete ---")
    if all_results:
        # Find the absolute maximum throughput from all successful runs
        max_batch_size, max_throughput = max(all_results, key=lambda item: item[1])
        print(f"Peak throughput found: {max_throughput:.2f} q/s at batch size {max_batch_size}.")

        # Calculate the 85% target
        target_throughput = max_throughput * 0.85

        # Find the smallest batch size that meets the target
        # all_results is already sorted by batch size, so we can iterate
        balanced_batch_size = 1
        for bs, throughput in all_results:
            if throughput >= target_throughput:
                balanced_batch_size = bs
                break

        print(f"Selecting balanced batch size of: {balanced_batch_size}")
        update_config_file(CONFIG_FILE, balanced_batch_size)
    else:
        print("Could not complete a single run. Please check your KataGo setup and file paths.")
    os._exit(0)


if __name__ == "__main__":
    main()
