import threading
import time

from utils.helper import construct_initial_sgf
from utils.katago_controller import KatagoController
from utils.katago_queries import KatagoQueries
from utils.models import get_model, MODEL_FULL, MODEL_9X9

DEBUG = True
MAX_GAMES = 2000


def play_game(black_playouts: int, white_playouts: int, queries_black: KatagoQueries, queries_white: KatagoQueries):
    moves = []
    playouts = [black_playouts, white_playouts]

    current_player = 0
    while True:
        if current_player == 0:
            move, winrate, score, _ = queries_black.get_best_move(queries_black.query(moves, playouts=playouts[current_player]))
        else:
            move, winrate, score, _ = queries_white.get_best_move(queries_white.query(moves, playouts=playouts[current_player]))

        moves.append(move)
        if len(moves) > 1 and moves[-1] == "pass" and moves[-2] == "pass":
            break
        current_player = 1 - current_player

    # Get position info from either engine (using the full model one)
    score_full, winrate_full = queries_white.get_position_info(moves)
    score_9x9, winrate_9x9 = queries_white.get_position_info(moves)

    sgf,_,_ = construct_initial_sgf(moves, None)
    print(sgf.serialise(None))

    if (score_full > 0) != (score_9x9 > 0):
        raise Exception("disagreement score: " + str(score_full) + " vs " + str(score_9x9))
    if (winrate_full > 0.5) != (winrate_9x9 > 0.5):
        raise Exception("disagreement winrate: " + str(winrate_full) + " vs " + str(winrate_9x9))

    print(winrate_full > 0.5)

    return winrate > 0.5, len(moves)


from scipy.stats import binomtest


def ci_95(wins, losses):
    k = wins
    n = wins + losses
    p_hat = k / n
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method='wilson')
    return p_hat, ci.low, ci.high


def worker_loop(queries_black, queries_white, shared_data, lock, shutdown_event, start_time):
    """
    Main loop for each worker thread. Plays games and reports stats.
    """

    while not shutdown_event.is_set():
        # --- Phase 1: Analyze, Print, and Plan (locked) ---
        with lock:
            # Fit model on all available historical data

            current_time = time.time()

            total_games = shared_data["won_games"] + shared_data["lost_games"]
            winrate = shared_data["won_games"] / total_games if total_games > 0 else 0.5
            elapsed_seconds = current_time - start_time
            elapsed_minutes = elapsed_seconds / 60.0
            games_per_minute = total_games / elapsed_minutes
            moves_per_minute = shared_data["total_moves"] / elapsed_minutes

            if moves_per_minute != 0:
                _, low, high = ci_95(shared_data["won_games"], shared_data["lost_games"])
                print(f"Games: {total_games} | Winrate: {(winrate * 100):.1f}% | 95% CI {100 * low:.1f}%-{100 * high:.1f}% | "
                      f"Games/min: {games_per_minute:.1f} | Moves/min: {moves_per_minute:.1f}")

            if total_games > MAX_GAMES:
                return

        black_won, num_moves = play_game(shared_data["black_playouts"], shared_data["white_playouts"],
                                         queries_black, queries_white)

        with lock:

            if black_won:
                shared_data["won_games"] += 1
            else:
                shared_data["lost_games"] += 1

            shared_data["total_moves"] += num_moves


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Play game with specified parameters')
    parser.add_argument('--black-playouts', type=int, default=10, help='Number of playouts for black moves.')
    parser.add_argument('--white-playouts', type=int, default=10, help='Number of playouts for white moves.')

    args = parser.parse_args()

    print(f"Black playouts: {args.black_playouts}, White playouts: {args.white_playouts}")

    # Create two controllers - one for each model
    controller_full = KatagoController()
    controller_9x9 = KatagoController()

    # Start engines with different models
    controller_full.start_engine(get_model(MODEL_FULL), human=False, verbose=DEBUG)
    controller_9x9.start_engine(get_model(MODEL_9X9), human=False, verbose=DEBUG)

    # Create queries objects for each controller
    queries_black = KatagoQueries(controller_9x9)
    queries_white = KatagoQueries(controller_full)

    lock = threading.Lock()
    shutdown_event = threading.Event()

    start_time = time.time()

    shared_data = {
        "black_playouts": args.black_playouts,
        "white_playouts": args.white_playouts,
        "won_games": 0,
        "lost_games": 0,
        "total_moves": 0,
    }

    # Determine batch size from one of the controllers
    batch_size = controller_full.get_batch_size()

    threads = []
    for i in range(1 if DEBUG else batch_size * 2):
        thread = threading.Thread(target=worker_loop, args=(queries_black, queries_white, shared_data, lock, shutdown_event, start_time))
        thread.start()
        threads.append(thread)

    print("started", len(threads), "worker threads")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        shutdown_event.set()
        for thread in threads:
            thread.join()
        controller_full.stop_engine()
        controller_9x9.stop_engine()
        print("Done.")
