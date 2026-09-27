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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.katago.binary import CONFIG_FILE, resolve_katago  # noqa: E402
from utils.katago.models import DEFAULT_MODEL, get_model  # noqa: E402


verbose = False

# Discard this many batches' worth of queries before timing — long enough that
# the slow model-load + first-batch warmup is fully absorbed.
WARMUP_BATCHES = 20
# Time this many batches' worth of queries — large enough to be stable but
# bounded so each iteration finishes in seconds.
TIMED_BATCHES = 100
# numAnalysisThreads : nnMaxBatchSize. With Python's GIL holding back per-process
# query rate, we want generous oversubscription on the KataGo side so the NN
# batch is full whenever the GIL is held. 3× costs a few hundred extra OS
# threads (each cheap) and ensures inference batches don't run half-empty.
# generate_tsumegos.py's worker_count should match this so python workers
# fill the queue katago is willing to accept.
ANALYSIS_THREAD_FACTOR = 3.0
# --- Helper Function for Query Generation ---

def generate_random_query(query_id: str, board_size: int) -> dict:
    """
    Generates a query with 2 black and 2 white stones placed randomly
    on a board of size board_size.
    """
    cols = "ABCDEFGHJKLMNOPQRST"[:board_size]  # No 'I' in Go coordinates
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


def run_test(batch_size, board_size, proc_holder=None):
    # One-line status message for the current test
    print(f"Testing Batch Size: {batch_size:<4} ... ", end='', flush=True)

    analysis_threads = max(1, int(batch_size * ANALYSIS_THREAD_FACTOR))
    katago_path, env = resolve_katago()
    command = [
        katago_path, "analysis",
        "-config", CONFIG_FILE,
        "-override-config",
        f"numAnalysisThreads={analysis_threads},nnMaxBatchSize={batch_size},"
        f"numSearchThreadsPerAnalysisThread=1,"
        f"maxBoardXSizeForNNBuffer={board_size},maxBoardYSizeForNNBuffer={board_size}",
        "-model", get_model(DEFAULT_MODEL),
    ]

    process = None
    try:
        # Only keep stderr as a readable pipe when we're going to drain it
        # (verbose mode). Otherwise DEVNULL it: nothing reads stderr in the
        # steady path, so a PIPE would deadlock KataGo the moment it writes
        # more than a pipe-buffer's worth (e.g. if logToStderr is enabled).
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if verbose else subprocess.DEVNULL,
            text=True, bufsize=1, env=env,
        )
        if proc_holder is not None:
            proc_holder[0] = process  # publish so the watchdog can kill on timeout

        if verbose:
            def read_stderr(process):
                for line in process.stderr:
                    print(f"KataGo STDERR: {line.strip()}")

            stderr_thread = threading.Thread(target=read_stderr, args=(process,))
            stderr_thread.daemon = True  # Allow main program to exit even if thread is still running
            stderr_thread.start()

        warmup_count = WARMUP_BATCHES * batch_size
        timed_count = TIMED_BATCHES * batch_size
        total_count = warmup_count + timed_count

        # Concurrent writer thread to avoid pipe-buffer deadlock: at higher
        # batch sizes, total query bytes (~250B × N) and response bytes
        # easily exceed Linux's ~64KB default pipe size, so writing all
        # queries before reading any responses can deadlock both sides.
        all_queries = [generate_random_query(f"q_{i}", board_size) for i in range(total_count)]
        write_error = [None]

        def writer():
            try:
                for q in all_queries:
                    process.stdin.write(json.dumps(q) + "\n")
                process.stdin.flush()
            except Exception as e:
                write_error[0] = e

        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()

        # Reader: drain warmup responses (absorbs model load + first-batch
        # warmup), then time the next chunk at steady state.
        for _ in range(warmup_count):
            if not process.stdout.readline():
                raise RuntimeError("exited during warm-up")

        start_time = time.perf_counter()
        for _ in range(timed_count):
            if not process.stdout.readline():
                raise RuntimeError("exited during timed test")
        end_time = time.perf_counter()

        writer_thread.join(timeout=10)
        if write_error[0] is not None:
            raise write_error[0]

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


def run_test_in_thread(batch_size, best_queries_per_second, board_size):
    result = [0]  # shared between threads
    proc_holder = [None]  # the thread publishes its subprocess here

    def thread_target():
        result[0] = run_test(batch_size, board_size, proc_holder=proc_holder)

    # Daemon so a run whose cleanup wedges (a KataGo that ignores kill) can
    # never block interpreter exit — main() returns normally instead of
    # needing os._exit to escape a lingering non-daemon thread.
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    if best_queries_per_second > 0:
        # 120s buffer for KataGo startup + model load + GPU init (the
        # b18c384nbt + CUDA combo is consistently 30–60s, with occasional
        # spikes), then 2× the expected query time at the prior throughput.
        expected_query_seconds = (
            (WARMUP_BATCHES + TIMED_BATCHES) * batch_size / best_queries_per_second
        )
        timeout = 120 + expected_query_seconds * 2
    else:
        # First iteration: no prior throughput estimate to scale by. Model
        # load + GPU init + a batch-size-1 warmup is a couple of minutes, so
        # cap generously rather than waiting forever (None) or the ~6.7h a
        # nonzero seed estimate used to imply.
        timeout = 600
    thread.join(timeout=timeout)
    if thread.is_alive():
        # Kill the subprocess so it doesn't keep eating GPU while the next
        # iteration tries to start.
        proc = proc_holder[0]
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        thread.join(timeout=5)
        print(f"TIMED OUT after {timeout:.1f}s")
        return 0
    return result[0]


# --- Main Test Logic ---

def main():
    """
    Main function to run the batch size test.
    """
    parser = argparse.ArgumentParser(description='KataGo batch size optimization')
    parser.add_argument('size', type=int, choices=ALL_BOARD_SIZES,
                        help='Board size to tune the engine for (one of %(choices)s)')
    parser.add_argument('--batchsize', type=int, help='Set specific batch size and skip testing')
    parser.add_argument('--verbose', action='store_true', help='set verbose mode')
    args = parser.parse_args()

    if args.verbose:
        global verbose
        verbose = True

    # get_model(DEFAULT_MODEL) downloads it if absent (run_test needs it).
    # Don't pull MODEL_HUMAN here — tune_engine never loads a human model, so
    # listing it only triggered a large pointless ~hundreds-of-MB download.
    for path in [resolve_katago()[0], CONFIG_FILE, get_model(DEFAULT_MODEL)]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at '{path}'", file=sys.stderr)
            sys.exit(1)

    if args.batchsize is not None:
        print(f"Setting batch size to {args.batchsize} from command line argument")
        update_config_file(CONFIG_FILE, args.batchsize)
        return

    batch_size = 1
    # Start at 0 so the first run_test_in_thread hits the "no prior estimate"
    # timeout branch; a nonzero seed made that branch dead and imposed a
    # multi-hour timeout on the very first run.
    best_queries_per_second = 0.0
    all_results = []

    # Stop the doubling search when a step yields less than this fractional
    # throughput gain. At saturation each doubling brings near-zero benefit;
    # anything below ~5% is dominated by measurement noise, not real headroom.
    min_improvement = 0.05

    print("--- Starting KataGo Throughput Test ---")
    while True:

        queries_per_second = run_test_in_thread(batch_size, best_queries_per_second, args.size)
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
    # Only successful runs may drive selection. A failed run still records
    # (batch_size, 0.0); if we selected from those, a run where EVERY attempt
    # OOM'd/stalled would pick batch size 1 and overwrite a previously
    # hardware-tuned bots/analysis.cfg — so filter them out and leave the
    # config untouched when nothing succeeded.
    successful = [(bs, qps) for bs, qps in all_results if qps > 0]
    if successful:
        # Find the absolute maximum throughput from all successful runs
        max_batch_size, max_throughput = max(successful, key=lambda item: item[1])
        print(f"Peak throughput found: {max_throughput:.2f} q/s at batch size {max_batch_size}.")

        # Calculate the 85% target
        target_throughput = max_throughput * 0.85

        # Find the smallest batch size that meets the target
        # successful is already sorted by batch size, so we can iterate
        balanced_batch_size = 1
        for bs, throughput in successful:
            if throughput >= target_throughput:
                balanced_batch_size = bs
                break

        print(f"Selecting balanced batch size of: {balanced_batch_size}")
        update_config_file(CONFIG_FILE, balanced_batch_size)
    else:
        print("No run succeeded; leaving bots/analysis.cfg unchanged. "
              "Please check your KataGo setup and file paths.")


if __name__ == "__main__":
    main()
