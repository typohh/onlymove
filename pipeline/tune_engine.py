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

from utils.configurations import BOARD_SIZE, DEFAULT_MODEL
from utils.katago_controller import KATAGO_PATH, CONFIG_FILE
from utils.models import get_model, MODEL_HUMAN, MODEL_BEST_SMALL


verbose = False

# Discard this many batches' worth of queries before timing — long enough that
# the slow model-load + first-batch warmup is fully absorbed.
WARMUP_BATCHES = 20
# Time this many batches' worth of queries — large enough to be stable but
# bounded so each iteration finishes in seconds.
TIMED_BATCHES = 100
# numAnalysisThreads : nnMaxBatchSize. > 1 keeps the inference batch full when
# any single thread stalls. KataGo recommends ≈1.5–2×.
ANALYSIS_THREAD_FACTOR = 1.5
# --- Helper Function for Query Generation ---

def generate_random_query(query_id: str) -> dict:
    """
    Generates a query with 2 black and 2 white stones placed randomly
    on a board of size BOARD_SIZE.
    """
    cols = "ABCDEFGHJKLMNOPQRST"[:BOARD_SIZE]  # No 'I' in Go coordinates
    all_coords = [f"{c}{r}" for c in cols for r in range(1, BOARD_SIZE + 1)]
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
        "rules": "tromp-taylor", "komi": 7.5, "boardXSize": BOARD_SIZE,
        "boardYSize": BOARD_SIZE, "maxVisits": 1,
        # "includePolicy": True, "overrideSettings": {"humanSLProfile": "rank_2d", "ignorePreRootHistory": False, "rootNumSymmetriesToSample": 1 }
    }


def update_config_file(config_path: str, batch_size: int):
    """
    Updates nnMaxBatchSize (= batch_size) and numAnalysisThreads
    (= ANALYSIS_THREAD_FACTOR × batch_size) in the given config file.
    """
    target_values = {
        "nnMaxBatchSize": batch_size,
        "numAnalysisThreads": max(1, int(batch_size * ANALYSIS_THREAD_FACTOR)),
    }
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        keys_found = set()

        for line in lines:
            stripped_line = line.strip()
            # Find lines that are not comments and define one of our keys
            if not stripped_line.startswith('#'):
                match = re.match(r"^\s*(\w+)\s*=", stripped_line)
                if match:
                    key = match.group(1)
                    if key in target_values:
                        updated_lines.append(f"{key} = {target_values[key]}\n")
                        keys_found.add(key)
                        continue  # Skip appending the old line
            # If the line doesn't match, add it as-is
            updated_lines.append(line)

        # If any key was not found in the file, add it to the end
        for key, value in target_values.items():
            if key not in keys_found:
                updated_lines.append(f"\n# Added by performance test\n{key} = {value}\n")

        with open(config_path, 'w') as f:
            f.writelines(updated_lines)

        print(f"\n--- Configuration Updated ---")
        for key, value in target_values.items():
            print(f"  {key} = {value}")
        print(f"in: '{config_path}'")

    except Exception as e:
        print(f"\nError: Failed to update config file: {e}", file=sys.stderr)


def run_test(batch_size, best_queries_per_second):
    # One-line status message for the current test
    print(f"Testing Batch Size: {batch_size:<4} ... ", end='', flush=True)

    analysis_threads = max(1, int(batch_size * ANALYSIS_THREAD_FACTOR))
    command = [
        KATAGO_PATH, "analysis",
        "-config", CONFIG_FILE,
        "-override-config",
        f"numAnalysisThreads={analysis_threads},nnMaxBatchSize={batch_size},"
        f"numSearchThreadsPerAnalysisThread=1,"
        f"maxBoardXSizeForNNBuffer={BOARD_SIZE},maxBoardYSizeForNNBuffer={BOARD_SIZE}",
        # "-model", get_model(DEFAULT_MODEL), "-human-model", get_model(MODEL_HUMAN)
        "-model", get_model(DEFAULT_MODEL)
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

        warmup_count = WARMUP_BATCHES * batch_size
        timed_count = TIMED_BATCHES * batch_size

        # Warmup absorbs model load (~20–40s on the b18c384nbt) plus first-batch
        # latency. Send all queries up front and drain the responses; the model
        # load happens transparently while we're waiting.
        warmup_queries = [generate_random_query(f"w_{i}") for i in range(warmup_count)]
        for query in warmup_queries:
            process.stdin.write(json.dumps(query) + "\n")
        process.stdin.flush()
        for _ in range(warmup_count):
            if not process.stdout.readline():
                raise RuntimeError("exited during warm-up")

        # Timed window: now at steady state. No per-iteration stall check —
        # the outer thread join already bounds total time.
        timed_queries = [generate_random_query(f"t_{i}") for i in range(timed_count)]
        start_time = time.perf_counter()
        for query in timed_queries:
            process.stdin.write(json.dumps(query) + "\n")
        process.stdin.flush()
        for _ in range(timed_count):
            if not process.stdout.readline():
                raise RuntimeError("exited during timed test")
        end_time = time.perf_counter()

        queries_per_second = timed_count / (end_time - start_time)
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
    if best_queries_per_second > 0:
        # 60s buffer for KataGo startup + model load, then 2× the expected
        # query time at the prior throughput. The previous formula
        # (5 + 2*batch/qps) didn't include startup at all and could
        # falsely time out high batches before the model finished loading.
        expected_query_seconds = (
            (WARMUP_BATCHES + TIMED_BATCHES) * batch_size / best_queries_per_second
        )
        timeout = 60 + expected_query_seconds * 2
    else:
        timeout = None  # First iteration: no prior estimate.
    thread.join(timeout=timeout)
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

    for path in [KATAGO_PATH, CONFIG_FILE, get_model(DEFAULT_MODEL), get_model(MODEL_HUMAN)]:
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

    # Stop the doubling search when a step yields less than this fractional
    # throughput gain. At saturation each doubling brings near-zero benefit;
    # anything below ~5% is dominated by measurement noise, not real headroom.
    min_improvement = 0.05

    print("--- Starting KataGo Throughput Test ---")
    while True:

        queries_per_second = run_test_in_thread(batch_size, best_queries_per_second)
        all_results.append((batch_size, queries_per_second))

        if queries_per_second == 0:
            # OOM, stall, or model-load timeout. End the search.
            break

        # Stop when doubling stops paying off (within a noise margin).
        if len(all_results) >= 2:
            prev_qps = all_results[-2][1]
            if prev_qps > 0 and queries_per_second < prev_qps * (1 + min_improvement):
                print(f"  No meaningful improvement (+{(queries_per_second/prev_qps-1)*100:.1f}%). Stopping search.")
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
