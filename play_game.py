import os
import threading
import time

import numpy as np
from sklearn.isotonic import IsotonicRegression

from katago_controller import KatagoController
from katago_queries import KatagoQueries

NUM_WORKERS = 2


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


def get_komis(shared_data):
    wins = shared_data["wins"]
    losses = shared_data["losses"]
    all_komis = wins + losses

    X = np.array(all_komis).reshape(-1, 1)
    y = np.array([1] * len(wins) + [0] * len(losses))

    # Fit an Isotonic Regression model
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso_reg.fit(X.flatten(), y)

    unique_komis = sorted(list(set(all_komis)))
    x_points = np.array(unique_komis)
    y_points = iso_reg.predict(x_points)

    target_prob = np.random.uniform(0.2, 0.8)
    komi_for_this_game, komi_minus_1_std, fair_komi, komi_plus_1_std = np.interp([target_prob, 0.5 - 0.341, 0.5, 0.5 + 0.341], y_points, x_points)
    komi_std = (komi_plus_1_std - komi_minus_1_std) / 2

    komi_for_this_game = round(komi_for_this_game - 0.5) + 0.5
    komi_for_this_game = max(-149.5, min(komi_for_this_game, 149.5))

    return fair_komi, komi_std, komi_for_this_game, len(all_komis)


def worker_loop(queries, shared_data, lock, shutdown_event, start_time):
    """
    Main loop for each worker thread. Plays games and reports stats.
    """

    while not shutdown_event.is_set():
        # --- Phase 1: Analyze, Print, and Plan (locked) ---
        with lock:
            # Fit model on all available historical data

            fair_komi, komi_std, komi_for_this_game, game_count = get_komis(shared_data)

            current_time = time.time()

            elapsed_seconds = current_time - start_time
            elapsed_minutes = elapsed_seconds / 60.0
            games_per_minute = shared_data["total_games"] / elapsed_minutes
            moves_per_minute = shared_data["total_moves"] / elapsed_minutes

            print(f"Games: {game_count:>5} | Komi : {fair_komi:>6.1f}±{komi_std} | "
                  f"Games/min: {games_per_minute:>6.1f} | Moves/min: {moves_per_minute:>7.1f}"
                  f" | Next Komi: {komi_for_this_game:>6.1f}")

        # --- Phase 2: Play the Game (unlocked) ---
        black_won, num_moves = play_game("rank_2d", "rank_2d", komi_for_this_game, queries)

        # --- Phase 3: Update Results (locked) ---
        with lock:
            if black_won:
                shared_data["wins"].append(komi_for_this_game)
            else:
                shared_data["losses"].append(komi_for_this_game)

            shared_data["total_games"] += 1
            shared_data["total_moves"] += num_moves


if __name__ == "__main__":
    controller = KatagoController(KATAGO_PATH, model_path= SL_MODEL_PATH, human_model_path=SL_MODEL_PATH, batch_size=NUM_WORKERS, search_threads=1)
    controller.start_engine(verbose=True)
    queries = KatagoQueries(controller)
    lock = threading.Lock()
    shutdown_event = threading.Event()

    time.sleep(10)

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
