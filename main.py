import os
import shutil
import threading
import time
import traceback
import hashlib

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
    Generates problems continuously.
    """
    try:
        while True:  # Keep working continuously
            problem = problem_finder.create_problem()
            if problem:
                sgf_str = problem_to_sgf(problem)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename = hashlib.sha256(sgf_str.encode('utf-8')).hexdigest()[:8] + ".sgf"
                with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                    f.write(sgf_str.encode("utf-8"))

                with print_lock:
                    print_stats()

    except Exception as e:
        print(e)
        traceback.print_exc()


def print_stats():
    print("\n--- Statistics ---")
    current_stats = STATS.get_stats()
    for key, value in current_stats.items():
        print(f"{key}: {value}")
    print("------------------\n")


def main():
    controller = None
    threads = []
    try:
        print("Starting KataGo engines...")
        controller = KatagoController()
        controller.start_engine()
        print("KataGo engines started.")

        queries = KatagoQueries(controller)
        problem_finder = ProblemFinder(queries)

        worker_count = 1 if DEBUG else controller.get_batch_size()

        print(f"Starting {worker_count} worker threads for continuous problem generation.")

        # Create and start worker threads
        for i in range(worker_count):
            thread = threading.Thread(target=worker_task, args=(problem_finder,))
            thread.daemon = True  # Dies when main thread dies
            thread.start()
            threads.append(thread)

        # Keep main thread alive
        while True:
            time.sleep(1)

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
