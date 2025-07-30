import threading
import time

from katago_controller import KatagoController
from katago_queries import KatagoQueries, get_rank_str

BLACK_HANDICAP = True
BLACK_RANK = 0.5
BLACK_PLAYOUTS = 1
WHITE_RANK = 0.5
WHITE_PLAYOUTS = 30
DEBUG = False
MAX_GAMES = 300


def play_game(black_rank: float, black_playouts: int, white_rank: float, white_playouts: int, queries: KatagoQueries):
    """
    Plays a 9x9 game and returns whether black won and the number of moves.
    """
    moves = []
    ranks = [black_rank, white_rank]
    playouts = [black_playouts, white_playouts]
    current_player = 0
    while True:
        if BLACK_HANDICAP and len(moves) == 1:
            move = "pass"
        else:
            move = queries.query_human(moves, ranks[current_player], playouts[current_player])
        moves.append(move)
        if len(moves) > 1 and moves[-1] == "pass" and moves[-2] == "pass":
            break
        current_player = 1 - current_player
    score, winrate = queries.get_position_info(moves)
    return winrate > 0.5, len(moves)


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
                print(f"Games: {total_games} | Winrate: {(winrate * 100):.1f}% | "
                      f"Games/min: {games_per_minute:.1f} | Moves/min: {moves_per_minute:.1f}")

            if total_games > MAX_GAMES:
                return

        black_won, num_moves = play_game(BLACK_RANK, BLACK_PLAYOUTS, WHITE_RANK, WHITE_PLAYOUTS, queries)

        with lock:

            if black_won:
                shared_data["won_games"] += 1
            else:
                shared_data["lost_games"] += 1

            shared_data["total_moves"] += num_moves


if __name__ == "__main__":
    print("black:", get_rank_str(BLACK_RANK), BLACK_PLAYOUTS, "white:", get_rank_str(WHITE_RANK), WHITE_PLAYOUTS)

    controller = KatagoController()
    controller.start_engine()
    queries = KatagoQueries(controller)
    lock = threading.Lock()
    shutdown_event = threading.Event()

    time.sleep(10)

    start_time = time.time()

    shared_data = {
        "won_games": 0,
        "lost_games": 0,
        "total_moves": 0,
    }

    threads = []
    for i in range(1 if DEBUG else controller.get_batch_size()):
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

# 20 kyu + 1 stone handicap vs 16 kyu = 55%
# 5 dan + 1 stone handicap vs 9 dan = 88%
# 5 dan 50 playouts = 9 dan