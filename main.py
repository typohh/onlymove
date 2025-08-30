import hashlib
import os
import threading
import time
import traceback

from katago_controller import KatagoController
from katago_queries import KatagoQueries
from models import MODEL_9X9, get_model
from problem_finder import ProblemFinder, problem_to_sgf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "tsumegos")

print_lock = threading.Lock()

DEBUG = False


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of differences from the mean

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def stddev(self):
        return (self.M2 / (self.n - 1))**0.5 if self.n > 1 else float("nan")

    @property
    def get_mean(self):
        return self.mean

class Stats:
    def __init__(self):
        self.problems_found = 0
        self.failures_to_create = 0
        self.number_of_moves = RunningStats()
        self.start_time = time.time()

    def add_problem_found(self, number_of_moves):
        self.problems_found += 1
        self.number_of_moves.update( number_of_moves )

    def add_failure_to_create_problem(self):
        self.failures_to_create += 1

    def get_stats(self):
        elapsed_time = time.time() - self.start_time
        tsumegos_per_minute = (self.problems_found / elapsed_time) * 60 if elapsed_time > 0 else 0
        stats = {
            "Running Time": f"{elapsed_time:.2f}s",
            "Games processed": self.problems_found + self.failures_to_create,
            "Success rate": f"{(100*self.problems_found/max(1,self.problems_found + self.failures_to_create)):.1f}%",
            "Avg moves": f"{self.number_of_moves.mean:.1f}±{self.number_of_moves.stddev:.1f}",
            "Tsumegos per Minute": f"{tsumegos_per_minute:.2f}",
        }
        return stats


STATS = Stats()

def worker_task(problem_finder: ProblemFinder):
    """
    The task for a single worker thread.
    Generates problems continuously.
    """
    try:
        while True:  # Keep working continuously
            problem = problem_finder.create_problem()
            if problem:
                STATS.add_problem_found(len(problem["moves"]))
                sgf_str = problem_to_sgf(problem)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename = hashlib.sha256(sgf_str.encode('utf-8')).hexdigest()[:8] + ".sgf"
                with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
                    f.write(sgf_str.encode("utf-8"))
                with print_lock:
                    print_stats()
            else:
                STATS.add_failure_to_create_problem()



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
        controller.start_engine(get_model(MODEL_9X9), verbose=DEBUG)
        print("KataGo engines started.")

        queries = KatagoQueries(controller)
        problem_finder = ProblemFinder(queries)
        problem_finder.debug = DEBUG

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
