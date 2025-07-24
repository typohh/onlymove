import argparse
import concurrent.futures
import os
import threading
import time
import traceback
import uuid

from stats import STATS, DEBUG
from katago_controller import KatagoController
from katago_queries import KatagoQueries
from problem_finder import ProblemFinder, problem_to_sgf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "tsumegos")

print_lock = threading.Lock()

def worker_task(problem_finder: ProblemFinder):
    """
    The task for a single worker thread.
    Generates one game and finds all problems within it.
    """
    try:
        problem = problem_finder.create_problem()
        if problem:
            sgf_str = problem_to_sgf(problem)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filename = os.path.join(OUTPUT_DIR, f"{uuid.uuid4()}.sgf")
            with open(filename, "wb") as f:
                f.write(sgf_str.encode("utf-8"))
            with print_lock:
                print_stats()
                print(problem["move_rank"], problem["game_rank"])
                print(sgf_str)

    except Exception as e:
        print(e)
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
    controller = None
    try:
        print("Starting KataGo engines...")
        controller = KatagoController()
        controller.start_engine()
        print("KataGo engines started.")

        queries = KatagoQueries(controller)
        problem_finder = ProblemFinder(queries)

        worker_count = 1 if DEBUG else controller.get_batch_size()

        # Use a ThreadPoolExecutor to manage a pool of worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            print(f"Starting {worker_count} worker threads for continuous problem generation.")

            futures = {
                executor.submit(worker_task, problem_finder): i for i in range(worker_count)
            }

            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    worker_id = futures[future]
                    del futures[future]
                    new_worker_id = worker_id
                    new_future = executor.submit(worker_task, problem_finder)
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
