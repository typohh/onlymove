import os
import threading
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

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
    model = LogisticRegression()

    previous_komi = 7.5

    while not shutdown_event.is_set():
        # --- Phase 1: Analyze, Print, and Plan (locked) ---
        with lock:
            # Fit model on all available historical data
            wins = shared_data["wins"]
            losses = shared_data["losses"]
            all_komis = wins + losses

            X = np.array(all_komis).reshape(-1, 1)
            y = np.array([1] * len(wins) + [0] * len(losses))
            model.fit(X, y)

            intercept = model.intercept_[0]
            coef = model.coef_[0][0]

            # Periodically print the current stats from a single thread
            current_time = time.time()
            if current_time - shared_data["last_print_time"] > 2.0:
                fair_komi = -intercept / coef
                mean_val = np.mean(all_komis)
                std_dev = np.std(all_komis)

                elapsed_seconds = current_time - start_time
                elapsed_minutes = elapsed_seconds / 60.0
                games_per_minute = shared_data["total_games"] / elapsed_minutes
                moves_per_minute = shared_data["total_moves"] / elapsed_minutes

                print(f"Games: {len(all_komis):>5} | Fair Komi (LR): {fair_komi:>6.3f} | "
                      f"Mean Komi: {mean_val:>6.3f} | Std Dev: {std_dev:>5.3f} | "
                      f"Games/min: {games_per_minute:>6.1f} | Moves/min: {moves_per_minute:>7.1f}"
                      f" | Last Game Komi: {previous_komi:>6.1f}")

                shared_data["last_print_time"] = current_time

            # Sample a new komi from the calculated distribution for the next game
            target_prob = np.random.uniform(0.2, 0.8)
            logit_p = np.log(target_prob / (1 - target_prob))
            komi_for_this_game = (logit_p - intercept) / coef
            komi_for_this_game = round(komi_for_this_game-0.5) + 0.5
            komi_for_this_game = max(-149.5, min( komi_for_this_game, 149.5))
            previous_komi = komi_for_this_game

        # --- Phase 2: Play the Game (unlocked) ---
        black_won, num_moves = play_game("rank_1d", "rank_2d", komi_for_this_game, queries)

        # --- Phase 3: Update Results (locked) ---
        with lock:
            if black_won:
                shared_data["wins"].append(komi_for_this_game)
            else:
                shared_data["losses"].append(komi_for_this_game)

            shared_data["total_games"] += 1
            shared_data["total_moves"] += num_moves


if __name__ == "__main__":
    controller = KatagoController(KATAGO_PATH, RL_MODEL_PATH, human_model_path=SL_MODEL_PATH)
    controller.start_engine()
    queries = KatagoQueries(controller)
    lock = threading.Lock()
    shutdown_event = threading.Event()

    start_time = time.time()

    # Initialize with two wins and two losses to provide a stable starting point
    # for the logistic regression model, centered around a komi of 7.5.
    shared_data = {
        "wins": [-2.5],
        "losses": [17.5],
        "total_games": 4,
        "total_moves": 0,
        "last_print_time": start_time,
    }

    threads = []
    for i in range(NUM_WORKERS):
        thread = threading.Thread(target=worker_loop, args=(queries, shared_data, lock, shutdown_event, start_time))
        thread.start()
        threads.append(thread)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        shutdown_event.set()
        for thread in threads:
            thread.join()
        controller.stop_engine()
        print("Done.")