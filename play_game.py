import os
import threading
import time
from scipy import stats

from katago_controller import KatagoController
from katago_queries import KatagoQueries

NUM_WORKERS = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATAGO_PATH = os.path.join(BASE_DIR, "KataGo", "cpp", "katago")
SL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "b18c384nbt-humanv0.bin.gz")
RL_MODEL_PATH = os.path.join(BASE_DIR, "weights", "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")


def play_game(rank1, rank2, komi, queries):
    """
    Plays a 9x9 game and returns whether black won and the number of moves.
    """
    moves = []
    ranks = [rank1, rank2]
    current_player = 0
    while True:
        rank = ranks[current_player]
        move = queries.query_weighted_random_move(moves, rank, komi)
        moves.append(move)
        if len(moves) > 1 and moves[-1] == "pass" and moves[-2] == "pass":
            break
        current_player = 1 - current_player
    score, _ = queries.get_position_info(moves)
    # Return both the result and the number of moves played
    return score > 0, len(moves)


def worker_loop(queries, shared_data, lock, shutdown_event, start_time):
    """
    Main loop for each worker thread. Plays games and reports stats.
    """
    while not shutdown_event.is_set():
        komi_for_this_game = shared_data["current_komi"]

        black_won, num_moves = play_game("rank_1d", "rank_2d", komi_for_this_game + 0.5, queries)

        with lock:
            # --- Update State ---
            shared_data["komis"].append(komi_for_this_game)
            shared_data["total_games"] += 1
            shared_data["total_moves"] += num_moves

            if black_won:
                shared_data["current_komi"] -= 1.0
            else:
                shared_data["current_komi"] += 1.0

            # --- Calculate Stats ---
            komis_data = shared_data["komis"]

            warmup_cutoff = round(stats.tmean(komis_data))
            if warmup_cutoff not in komis_data:
                relevant_games = komis_data[len(komis_data) // 2:]
            else:
                start_index = komis_data.index(warmup_cutoff)
                relevant_games = komis_data[start_index:]

            mean_val = stats.tmean(relevant_games)
            if len(set(relevant_games)) == 1:
                std_dev = stats.tstd(relevant_games)
            else:
                std_dev = 0

            # --- Calculate Performance Metrics ---
            elapsed_seconds = time.time() - start_time
            elapsed_minutes = elapsed_seconds / 60.0
            games_per_minute = shared_data["total_games"] / elapsed_minutes
            moves_per_minute = shared_data["total_moves"] / elapsed_minutes

            # --- Print Combined Output ---
            print(f"Games: {len(relevant_games):>4} | Mean Komi: {mean_val:>6.3f} | Std Dev: {std_dev:>5.3f} | "
                  f"Games/min: {games_per_minute:>6.1f} | Moves/min: {moves_per_minute:>7.1f}")


if __name__ == "__main__":
    controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
    controller.start_engine()
    queries = KatagoQueries(controller)
    lock = threading.Lock()
    shutdown_event = threading.Event()

    # Record the start time for performance calculations
    start_time = time.time()

    # Add new counters to shared data
    shared_data = {
        "current_komi": 6.0,
        "komis": [],
        "total_games": 0,
        "total_moves": 0,
    }

    threads = []
    for i in range(NUM_WORKERS):
        thread = threading.Thread(target=worker_loop, args=(queries, shared_data, lock, shutdown_event, start_time))
        thread.start()
        threads.append(thread)

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        shutdown_event.set()
        for thread in threads: thread.join()
        controller.stop_engine()
        print("Done.")