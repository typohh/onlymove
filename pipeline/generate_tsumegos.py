import argparse
import hashlib
import os
import signal
import threading
import time
import traceback

from utils.add_variations import (
    CALM_UTILITY_OVERRIDE,
    VariationDepthExceeded,
    add_shallow_variations,
    generate_variations_directly_to_sgf,
)
from utils.configurations import BOARD_SIZE, DEBUG, DEFAULT_MODEL, debug
from utils.helper import construct_initial_sgf
from utils.katago_controller import KatagoController
from utils.katago_queries import KatagoQueries, katago_to_sgf_coords
from utils.models import get_model
from utils.problem_finder import ProblemFinder
from sgfmill import boards

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tsumegos", str(BOARD_SIZE))

print_lock = threading.Lock()
stop_event = threading.Event()
max_problems = None


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

def transform_point(r, c, trans, size):
    if trans == 0:  # identity
        return r, c
    elif trans == 1:  # 90 clockwise
        return c, size - 1 - r
    elif trans == 2:  # 180
        return size - 1 - r, size - 1 - c
    elif trans == 3:  # 270 clockwise
        return size - 1 - c, r
    elif trans == 4:  # horizontal reflect
        return size - 1 - r, c
    elif trans == 5:  # vertical reflect
        return r, size - 1 - c
    elif trans == 6:  # main diagonal
        return c, r
    elif trans == 7:  # anti-diagonal
        return size - 1 - c, size - 1 - r

def get_transformed_board(board, trans):
    size = board.side
    new_board = boards.Board(size)
    for r in range(size):
        for c in range(size):
            stone = board.get(r, c)
            if stone:
                nr, nc = transform_point(r, c, trans, size)
                new_board.play(nr, nc, stone)
    return new_board

def get_orientation_invariant_hash(board):
    hashes = []
    for trans in range(8):
        tb = get_transformed_board(board, trans)
        h = hashlib.sha256(str(tb.board).encode()).hexdigest()
        hashes.append(h)
    return min(hashes)

def worker_task(problem_finder: ProblemFinder, queries: KatagoQueries, threads_list: list):
    """
    The task for a single worker thread.
    Generates problems continuously.
    """
    try:
        while not stop_event.is_set():
            try:
                STATS.set_state("problem_creation")
                moves,solution = problem_finder.create_problem()
                if solution:
                    STATS.add_problem_found(len(moves))

                    STATS.set_state("other")
                    # Compute board state for hashing
                    board = boards.Board(BOARD_SIZE)
                    player = 'b'
                    for move in moves:
                        if move != "pass":
                            row, col = katago_to_sgf_coords(move, BOARD_SIZE)
                            board.play(row, col, player)
                        player = 'w' if player == 'b' else 'b'
                    # 1. Construct initial SGF
                    game,node,answer = construct_initial_sgf(moves, solution)

                    STATS.set_state("deep_variations")
                    generate_variations_directly_to_sgf(queries, answer, moves + [solution], 1.0, True, None)
                    generate_variations_directly_to_sgf(
                        queries, answer, moves + [solution], 1.0, True, None,
                        override_settings=CALM_UTILITY_OVERRIDE,
                    )
                    generate_variations_directly_to_sgf(queries, node, moves, 1.0, True, solution)

                    STATS.set_state("shallow_variations")
                    add_shallow_variations(queries, node, moves)

                    STATS.set_state("other")
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    board_hash = get_orientation_invariant_hash(board)
                    filename = board_hash[:8] + ".sgf"
                    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                        f.write(game.serialise(None).decode())
                    with print_lock:
                        debug(filename)
                        # Print stats at most once per minute
                        if STATS.should_print_stats():
                            # Count active and alive threads
                            active_threads = sum(1 for t in threads_list if t.is_alive())
                            print_stats(active_threads)
                    if max_problems is not None and STATS.problems_found >= max_problems:
                        stop_event.set()
                else:
                    STATS.add_failure_to_create_problem()
            except VariationDepthExceeded as e:
                with print_lock:
                    print(f"[skip] runaway variation ({len(e.moves)} moves); ignoring this problem")
                    print(f"  sequence: {e.moves}")
                STATS.set_state("other")
                continue

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
    parser = argparse.ArgumentParser(description='Generate tsumegos')
    parser.add_argument('--max-problems', type=int, default=None,
                        help='Stop after this many problems have been generated')
    args = parser.parse_args()

    global max_problems
    max_problems = args.max_problems

    # Treat SIGHUP / SIGTERM as KeyboardInterrupt so the main try/finally runs
    # and KataGo gets stop_engine'd. tmux kill-session sends SIGHUP; without
    # this, Python exits without running cleanup and katago is orphaned.
    def _signal_to_kbi(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGHUP, _signal_to_kbi)
    signal.signal(signal.SIGTERM, _signal_to_kbi)

    controller = None
    threads = []
    try:
        print("Starting KataGo engines...")
        controller = KatagoController()
        controller.start_engine(get_model(DEFAULT_MODEL), verbose=DEBUG)
        print("KataGo engines started.")

        queries = KatagoQueries(controller)
        problem_finder = ProblemFinder(queries)
        problem_finder.debug = DEBUG

        # Match ANALYSIS_THREAD_FACTOR in pipeline.tune_engine — python workers
        # need to keep KataGo's analysis queue full despite GIL serialization.
        worker_count = 1 if DEBUG else int(controller.get_batch_size() * 3)

        print(f"Starting {worker_count} worker threads for continuous problem generation.")

        # Create and start worker threads
        for i in range(worker_count):
            thread = threading.Thread(target=worker_task, args=(problem_finder, queries, threads)) # Pass threads list
            thread.daemon = True  # Dies when main thread dies
            thread.start()
            threads.append(thread)

        # Keep main thread alive
        while not stop_event.is_set():
            time.sleep(1)

        if max_problems is not None:
            print(f"\nReached max_problems={max_problems}; shutting down.")

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
