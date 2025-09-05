import hashlib
import os
import threading
import time
import traceback

from add_variations import add_shallow_variations, generate_variations_directly_to_sgf
from configurations import DEBUG, debug
from helper import construct_initial_sgf
from katago_controller import KatagoController
from katago_queries import KatagoQueries
from models import MODEL_9X9, get_model
from problem_finder import ProblemFinder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "tsumegos")

print_lock = threading.Lock()


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
        return (self.M2 / (self.n - 1)) ** 0.5 if self.n > 1 else float("nan")

    @property
    def get_mean(self):
        return self.mean


class Stats:
    def __init__(self):
        self.problems_found = 0
        self.failures_to_create = 0
        self.number_of_moves = RunningStats()
        self.start_time = time.time()
        self.last_state_change_time = time.time()
        self.current_state = "other"
        self.time_spent_in_states = {
            "problem_creation": 0.0,
            "shallow_variations": 0.0,
            "deep_variations": 0.0,
            "other": 0.0,
        }
        self.last_print_time = 0  # Track when stats were last printed

    def set_state(self, new_state):
        now = time.time()
        time_in_previous_state = now - self.last_state_change_time
        if self.current_state in self.time_spent_in_states:
            self.time_spent_in_states[self.current_state] += time_in_previous_state

        self.current_state = new_state
        self.last_state_change_time = now

    def add_problem_found(self, number_of_moves):
        self.problems_found += 1
        self.number_of_moves.update(number_of_moves)

    def add_failure_to_create_problem(self):
        self.failures_to_create += 1

    def should_print_stats(self):
        """Check if enough time has passed to print stats again (at most once per minute)"""
        now = time.time()
        if now - self.last_print_time >= 60:  # 60 seconds = 1 minute
            self.last_print_time = now
            return True
        return False

    def get_stats(self, active_threads=0):
        # Make a copy to avoid modifying the original during calculation
        time_spent = self.time_spent_in_states.copy()
        now = time.time()
        time_in_current_state = now - self.last_state_change_time
        if self.current_state in time_spent:
            time_spent[self.current_state] += time_in_current_state

        elapsed_time = time.time() - self.start_time
        tsumegos_per_minute = (self.problems_found / elapsed_time) * 60 if elapsed_time > 0 else 0

        stats = {
            "Running Time": f"{elapsed_time:.2f}s",
            "Games processed": self.problems_found + self.failures_to_create,
            "Success rate": f"{(100 * self.problems_found / max(1, self.problems_found + self.failures_to_create)):.1f}%",
            "Avg moves": f"{self.number_of_moves.mean:.1f}±{self.number_of_moves.stddev:.1f}",
            "Tsumegos per Minute": f"{tsumegos_per_minute:.2f}",
            "Active Worker Threads": active_threads,
        }

        # Add time tracking stats
        total_tracked_time = sum(time_spent.values())
        if total_tracked_time > 0:
            for state, duration in time_spent.items():
                percentage = (duration / total_tracked_time) * 100
                stats[f"Time in {state}"] = f"{duration:.2f}s ({percentage:.1f}%)"

        return stats


STATS = Stats()

def worker_task(problem_finder: ProblemFinder, queries: KatagoQueries, threads_list: list):
    """
    The task for a single worker thread.
    Generates problems continuously.
    """
    try:
        while True:  # Keep working continuously
            STATS.set_state("problem_creation")
            moves,solution = problem_finder.create_problem()
            if solution:
                STATS.add_problem_found(len(moves))

                STATS.set_state("other")
                # 1. Construct initial SGF
                game,node,answer = construct_initial_sgf(moves, solution)

                STATS.set_state("deep_variations")
                generate_variations_directly_to_sgf(queries, answer, moves + [solution], 1.0, True,None)
                generate_variations_directly_to_sgf(queries, node, moves, 1.0, False, solution)

                STATS.set_state("shallow_variations")
                add_shallow_variations(queries, node, moves)

                STATS.set_state("other")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filename = hashlib.sha256(str(moves).encode()).hexdigest()[:8] + ".sgf"
                with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                    f.write(game.serialise(None).decode())
                with print_lock:
                    debug(filename)
                    # Print stats at most once per minute
                    if STATS.should_print_stats():
                        # Count active and alive threads
                        active_threads = sum(1 for t in threads_list if t.is_alive())
                        print_stats(active_threads)
            else:
                STATS.add_failure_to_create_problem()

    except Exception as e:
        print(e)
        traceback.print_exc()


def print_stats(active_threads=0):
    print("--- Statistics ---")
    current_stats = STATS.get_stats(active_threads)
    for key, value in current_stats.items():
        print(f"{key}: {value}")
    print("------------------")


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

        worker_count = 1 if DEBUG else int(controller.get_batch_size() * 1.5)

        print(f"Starting {worker_count} worker threads for continuous problem generation.")

        # Create and start worker threads
        for i in range(worker_count):
            thread = threading.Thread(target=worker_task, args=(problem_finder, queries, threads)) # Pass threads list
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
