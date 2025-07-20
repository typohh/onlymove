import os
import argparse
import traceback
import uuid
import concurrent.futures
import time
import threading

from Stats import STATS, DEBUG
from katago_queries import KatagoQueries
from problem_finder import ProblemFinder
from katago_controller import KatagoController

# Define paths - adjust these if your KataGo executable or models are elsewhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")
OUTPUT_DIR = "./tsumegos"



def worker_task(worker_id: int, katago_queries: KatagoQueries, problem_finder: ProblemFinder):
    """
    The task for a single worker thread.
    Generates one game and finds all problems within it.
    """
    try:
        problem_finder.create_problem(OUTPUT_DIR)

    except Exception as e:
        print(f"An error occurred in worker {worker_id}: {e}")
        traceback.print_exc()


def print_stats():
    while not DEBUG:
        time.sleep(10)
        print("\n--- Statistics ---")
        current_stats = STATS.get_stats()
        for key, value in current_stats.items():
            if key == "Rejection Reasons":
                if value:
                    print("Rejection Reasons:")
                    for reason, count in value.items():
                        print(f"  - {reason}: {count}")
            else:
                print(f"{key}: {value}")
        print("------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Generate Go problems using KataGo in parallel.")
    parser.add_argument("--verbose", action="store_true", help="Show KataGo engine output")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker threads to run")
    args = parser.parse_args()

    if DEBUG:
        args.num_workers = 1

    controller = None
    try:
        print("Starting KataGo engines...")
        controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
        controller.start_engine(verbose=args.verbose)
        print("KataGo engines started.")

        queries = KatagoQueries(controller)
        problem_finder = ProblemFinder(queries)

        # Start a thread to print stats periodically
        stats_thread = threading.Thread(target=print_stats, args=(), daemon=True)
        stats_thread.start()

        # Use a ThreadPoolExecutor to manage a pool of worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            print(f"Starting {args.num_workers} worker threads for continuous problem generation.")

            futures = {
                executor.submit(worker_task, i, queries, problem_finder): i for i in range(args.num_workers)
            }

            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    worker_id = futures[future]
                    del futures[future]
                    new_worker_id = worker_id
                    new_future = executor.submit(worker_task, new_worker_id, queries, problem_finder)
                    futures[new_future] = new_worker_id

    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down.")
    except Exception as e:
        print(f"A critical error occurred in the main thread: {e}")
        traceback.print_exc()
    finally:
        print("Stopping KataGo engines...")
        if controller:
            controller.stop_engine()
        print("KataGo engines stopped.")


if __name__ == "__main__":
    main()
