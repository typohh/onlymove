import threading
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
from typing import Dict, List, Tuple

from katago_controller import KatagoController
from katago_queries import KatagoQueries, get_rank_str
from problem_finder import save_problem_as_sgf

BLACK_RANK = 0.4
WHITE_RANK = 0.6


def play_game(black_rank: float, white_rank: float, komi, queries: KatagoQueries):
    """
    Plays a 9x9 game and returns whether black won and the number of moves.
    """
    moves = []
    ranks = [black_rank, white_rank]
    current_player = 0
    while True:
        move = queries.query_weighted_random_move(moves, ranks[current_player], komi)
        moves.append(move)
        if len(moves) > 1 and moves[-1] == "pass" and moves[-2] == "pass":
            break
        current_player = 1 - current_player
    score, winrate = queries.get_position_info(moves)
    sgf_str = save_problem_as_sgf({
        "moves": moves,
    }, komi=komi).decode()
    return winrate > 0.5, len(moves), sgf


def get_komis(shared_data: Dict[str, List[float]]) -> Tuple[float, float, float, int]:
    """
    Calculates fair komi, distribution std dev, and a random sample from the 20%-80% win probability range.
    """
    wins = shared_data["wins"]
    losses = shared_data["losses"]

    all_komis_list = wins + losses

    # Fit a logistic regression model to the game data.
    X = np.array(all_komis_list).reshape(-1, 1)
    y = np.array([1] * len(wins) + [0] * len(losses))
    model = LogisticRegression(C=1e10)
    model.fit(X, y)

    intercept = model.intercept_[0]
    coefficient = model.coef_[0][0]

    # Calculate fair komi, where win probability is 50%.
    fair_komi = -intercept / coefficient

    # Calculate the standard deviation of the fitted probability distribution.
    # For a logistic distribution, the standard deviation is scale * pi / sqrt(3),
    # where the scale parameter is 1 / coefficient.
    scale_param = 1 / abs(coefficient)
    komi_std_of_distribution = scale_param * np.pi / np.sqrt(3)

    if len(all_komis_list) == 2:
        komi_std_of_distribution = 30

    # Sample a komi corresponding to a random win probability between 0.2 and 0.8.
    random_probability = random.uniform(0.2, 0.8)
    logit_p = np.log(random_probability / (1 - random_probability))
    komi_for_this_game = fair_komi + logit_p * (max(3, komi_std_of_distribution) * np.sqrt(3) / np.pi)

    komi_for_this_game = round(komi_for_this_game + 0.5) - 0.5
    komi_for_this_game = max(-80.5, min(komi_for_this_game, 80.5))

    return fair_komi, komi_std_of_distribution, komi_for_this_game, len(all_komis_list)


def worker_loop(queries, shared_data, lock, shutdown_event, start_time):
    """
    Main loop for each worker thread. Plays games and reports stats.
    """

    while not shutdown_event.is_set():
        # --- Phase 1: Analyze, Print, and Plan (locked) ---
        with lock:
            # Fit model on all available historical data

            fair_komi, komi_std, komi_for_this_game, game_count = get_komis(shared_data)
            game_count -= 2

            current_time = time.time()

            elapsed_seconds = current_time - start_time
            elapsed_minutes = elapsed_seconds / 60.0
            games_per_minute = shared_data["total_games"] / elapsed_minutes
            moves_per_minute = shared_data["total_moves"] / elapsed_minutes

            if game_count > 0:
                print("wins:", sorted(shared_data["wins"]))
                print("losses:", sorted(shared_data["losses"]))

                print(f"Games: {game_count} | Komi : {fair_komi:.1f}±{komi_std:.1f} | "
                      f"Games/min: {games_per_minute:.1f} | Moves/min: {moves_per_minute:.1f}"
                      f" | Next Komi: {komi_for_this_game:.1f}")

        # --- Phase 2: Play the Game (unlocked) ---
        black_won, num_moves, sgf = play_game(BLACK_RANK, WHITE_RANK, komi_for_this_game, queries)

        # --- Phase 3: Update Results (locked) ---
        with lock:

            print(sgf)

            if black_won:
                shared_data["wins"].append(komi_for_this_game)
            else:
                shared_data["losses"].append(komi_for_this_game)

            shared_data["total_games"] += 1
            shared_data["total_moves"] += num_moves


if __name__ == "__main__":
    controller = KatagoController()
    controller.start_engine()
    queries = KatagoQueries(controller)
    lock = threading.Lock()
    shutdown_event = threading.Event()

    time.sleep(10)

    start_time = time.time()

    # Initialize with two wins and two losses to provide a stable starting point
    # for the logistic regression model, centered around a komi of 7.5.
    shared_data = {
        "wins": [-81.5],
        "losses": [96.5],
        "total_games": 2,
        "total_moves": 0,
        "last_print_time": start_time,
    }

    threads = []
    for i in range(controller.get_batch_size()):
    # for i in range(1):
        thread = threading.Thread(target=worker_loop, args=(queries, shared_data, lock, shutdown_event, start_time))
        thread.start()
        threads.append(thread)

    print("started", len(threads), "worker threads")
    print("black:", get_rank_str(BLACK_RANK), "white:", get_rank_str(WHITE_RANK))

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
