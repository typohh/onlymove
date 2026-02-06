import threading
import time

from utils.katago_controller import KatagoController
from utils.katago_queries import KatagoQueries, get_rank_number
from utils.models import MODEL_BEST_SMALL, get_model, MODEL_FULL
from utils.problem_finder import problem_to_sgf

DEBUG = False
MAX_GAMES = 2000


def play_game(black_rank: float, black_temperature: float, black_handicap, white_rank: float, white_temperature: float, queries: KatagoQueries):
    moves = []
    ranks = [get_rank_number(black_rank), get_rank_number(white_rank)]
    temperatures = [black_temperature,white_temperature]

    # 95% CI 38.3%-46.0%
    # ranks = [get_rank_number('20k'),get_rank_number('14k')]
    # temperatures = [1,1]

    # # 95% CI 48.4%-55.9%
    # ranks = [get_rank_number('20k'),get_rank_number('14k')]
    # temperatures = [0.8,1]

    # 95% CI 31.3%-39.6%
    # ranks = [get_rank_number('14k'),get_rank_number('7k')]
    # temperatures = [0.9,1]

    # 95% CI 40.6%-50.0%
    # ranks = [get_rank_number('14k'),get_rank_number('7k')]
    # temperatures = [0.6,1]

    # 95% CI 84.2%-91.0%
    # ranks = [get_rank_number('14k'),get_rank_number('7k')]
    # temperatures = [1,2]

    # 95% CI 51.2%-62.1%
    # ranks = [get_rank_number('14k'),get_rank_number('7k')]
    # temperatures = [1,1.5]

    # 95% CI 37.3%-48.5%
    # ranks = [get_rank_number('7k'),get_rank_number('1k')]
    # temperatures = [1,1.2]

    # 95% CI 37.4%-45.7%
    # ranks = [get_rank_number('14k'),get_rank_number('7k')]
    # temperatures = [0.8,1]

    # 95% CI 51.9%-59.5%
    # ranks = [get_rank_number('2k'),get_rank_number('5d')]
    # temperatures = [1,1]

    # 95% CI 42.3%-51.6%
    # ranks = [get_rank_number('3k'),get_rank_number('4d')]
    # temperatures = [1,1]

    # 95% CI 41.6%-48.8%
    # ranks = [get_rank_number('4k'),get_rank_number('3d')]
    # temperatures = [1,1]

    # 95% CI 73.3%-82.7
    # ranks = [get_rank_number('3d'),get_rank_number('9d')]
    # temperatures = [1,1]

    # 95% CI 49.8%-55.9%
    # ranks = [get_rank_number('3d'),get_rank_number('9d')]
    # temperatures = [1,0.2]

    # 95% CI 42.4%-53.3%
    # ranks = [get_rank_number('17k'),get_rank_number('11k')]
    # temperatures = [1,1.2]

    # 95% CI 32.2%-43.1%
    # ranks = [get_rank_number('10k'),get_rank_number('4k')]
    # temperatures = [1,1.2]

    # 95% CI 30.6%-41.3%
    # ranks = [get_rank_number('10k'),get_rank_number('4k')]
    # temperatures = [1,1]

    # 95% CI 48.1%-52.5%
    # ranks = [get_rank_number('3d'),get_rank_number('9d')]
    # temperatures = [1,0.1]

    # 95% CI 45.7%-50.0%
    # ranks = [get_rank_number('4k'),get_rank_number('3d')]
    # human_temperature = [0.9,1]

    # 95% CI 47.9%-52.3%
    # ranks = [get_rank_number('10k'),get_rank_number('4k')]
    # temperatures = [0.5,0.9]

    # 95% CI 46.2%-50.6%
    # ranks = [get_rank_number('20k'),get_rank_number('14k')]
    # temperatures = [0.1,0.5]

    # 95% CI 49.1%-53.4%
    # ranks = [get_rank_number('14k'),get_rank_number('8k')]
    # temperatures = [0.5,0.95]

    # 95% CI 46.5%-50.8%
    # ranks = [get_rank_number('8k'),get_rank_number('2k')]
    # temperatures = [0.95,1.2]

    # 95% CI 46.3%-50.6%
    # ranks = [get_rank_number('2k'),get_rank_number('5d')]
    # temperatures = [1.2,1.15]

    current_player = 0
    while True:
        if black_handicap and len(moves) == 1:
            move = "pass"
        else:
            move,_ = queries.query_human_temperature(moves, ranks[current_player], temperatures[current_player])
        moves.append(move)
        if len(moves) > 1 and moves[-1] == "pass" and moves[-2] == "pass":
            break
        current_player = 1 - current_player

    score, winrate = queries.get_position_info(moves)

    sgf = problem_to_sgf({
        "moves": moves
    })
    print(winrate > 0.5)
    print(sgf)

    return winrate > 0.5, len(moves)


from scipy.stats import binomtest


def ci_95(wins, losses):
    k = wins
    n = wins + losses
    p_hat = k / n
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method='wilson')
    return p_hat, ci.low, ci.high


def worker_loop(queries, shared_data, lock, shutdown_event, start_time):
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

        black_won, num_moves = play_game(shared_data["black_rank"], shared_data["black_temperature"], shared_data["black_handicap"], shared_data["white_rank"],
                                         shared_data["white_temperature"], queries)

        with lock:

            if black_won:
                shared_data["won_games"] += 1
            else:
                shared_data["lost_games"] += 1

            shared_data["total_moves"] += num_moves


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Play game with specified parameters')
    parser.add_argument('--black-temperature', type=float, default=1.0, help='Temperature of black moves.')
    parser.add_argument('--white-temperature', type=float, default=1.0, help='Temperature of white moves.')
    parser.add_argument('--black-rank', type=str, default="3d", help='Rank for black player')
    parser.add_argument('--white-rank', type=str, default="9d", help='Rank for white player')
    parser.add_argument('--no-handicap', action='store_true', default=False, help='Enable handicap for black player')

    args = parser.parse_args()

    print(args.black_rank, args.black_temperature, not args.no_handicap, args.white_rank, args.white_temperature)

    controller = KatagoController()
    controller.start_engine(get_model(MODEL_BEST_SMALL), human=True, verbose=DEBUG)
    queries = KatagoQueries(controller)
    lock = threading.Lock()
    shutdown_event = threading.Event()

    start_time = time.time()

    shared_data = {
        "black_rank": args.black_rank,
        "white_rank": args.white_rank,
        "black_temperature": args.black_temperature,
        "white_temperature": args.white_temperature,
        "black_handicap": not args.no_handicap,
        "won_games": 0,
        "lost_games": 0,
        "total_moves": 0,
    }

    threads = []
    for i in range(1 if DEBUG else controller.get_batch_size() * 2):
        thread = threading.Thread(target=worker_loop, args=(queries, shared_data, lock, shutdown_event, start_time))
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
        controller.stop_engine()
        print("Done.")

