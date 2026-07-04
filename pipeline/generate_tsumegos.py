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
from utils.configurations import ALL_BOARD_SIZES, DEBUG, DEFAULT_MODEL, debug
from utils.helper import construct_initial_sgf
from utils.katago_controller import KatagoController
from utils.katago_queries import KatagoQueries
from utils.coords import gtp_to_sgfmill_point
from utils.models import get_model
from utils.problem_finder import ProblemFinder
from sgfmill import boards

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print_lock = threading.Lock()
stop_event = threading.Event()
max_problems = None
# Set in main() from the mandatory size arg: tsumegos/<size>/.
OUTPUT_DIR = None


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


class Stats:
    def __init__(self):
        self.problems_found = 0
        self.failures_to_create = 0
        self.duplicate_ids_skipped = 0
        self.worker_errors = 0
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

    def add_duplicate_id_skipped(self):
        self.duplicate_ids_skipped += 1

    def add_worker_error(self):
        self.worker_errors += 1

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
        if self.duplicate_ids_skipped:
            stats["Duplicate ids skipped"] = self.duplicate_ids_skipped
        if self.worker_errors:
            stats["Worker errors (skipped)"] = self.worker_errors

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

def tsumego_id_taken(filename):
    """True if a tsumego with this id already exists on ANY board size.

    Never overwrite an existing id: attempts/, activations/, ratings/ and the
    PWA all key rows by the 8-hex id, so replacing the SGF would silently
    re-point every one of those artifacts at different content. Hits happen two
    ways — the truncated hash is only 32 bits, so distinct positions collide
    (birthday math: ~2 expected at the current 9x9 corpus size), and the
    generator can legitimately rediscover an already-generated position. The id
    namespace is checked across ALL sizes because downstream id-keyed consumers
    share it (the corpus already carries 4 cross-size collisions from before
    this guard; those are grandfathered, not license to add more)."""
    return any(
        os.path.exists(os.path.join(PROJECT_ROOT, "tsumegos", str(s), filename))
        for s in ALL_BOARD_SIZES)

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
                    STATS.set_state("other")
                    # Compute board state for hashing
                    board = boards.Board(queries.board_size)
                    player = 'b'
                    for move in moves:
                        if move != "pass":
                            row, col = gtp_to_sgfmill_point(move, queries.board_size)
                            board.play(row, col, player)
                        player = 'w' if player == 'b' else 'b'
                    # Identity check BEFORE the expensive variation queries: a
                    # duplicate id costs only the position search, not the
                    # minutes of KataGo analysis below.
                    board_hash = get_orientation_invariant_hash(board)
                    filename = board_hash[:8] + ".sgf"
                    if tsumego_id_taken(filename):
                        STATS.add_duplicate_id_skipped()
                        with print_lock:
                            print(f"[skip] {filename}: id already exists "
                                  f"(this or another size); not overwriting")
                        continue

                    STATS.add_problem_found(len(moves))

                    # 1. Construct initial SGF
                    game,node,answer = construct_initial_sgf(moves, solution, queries.board_size)

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
                # Any other transient failure (a KataGo query hiccup, a parse
                # error, ...) must skip ONE problem, not kill the worker for
                # the rest of the run. Without this the outer handler returned
                # and the thread died permanently, so workers dropped off one
                # by one and throughput silently decayed toward zero.
                STATS.add_worker_error()
                STATS.set_state("other")
                with print_lock:
                    print(f"[skip] worker error: {e}")
                    traceback.print_exc()
                continue

    except Exception as e:
        # Backstop only — the inner loop now handles per-problem errors, so
        # reaching here means the loop scaffolding itself failed.
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
    parser.add_argument('size', type=int, choices=ALL_BOARD_SIZES,
                        help='Board size to generate tsumegos for (one of %(choices)s)')
    parser.add_argument('--max-problems', type=int, default=None,
                        help='Stop after this many problems have been generated')
    args = parser.parse_args()

    global max_problems, OUTPUT_DIR
    max_problems = args.max_problems
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tsumegos", str(args.size))

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
        controller.start_engine(get_model(DEFAULT_MODEL), board_size=args.size, verbose=DEBUG)
        print("KataGo engines started.")

        queries = KatagoQueries(controller, args.size)
        problem_finder = ProblemFinder(queries)

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
