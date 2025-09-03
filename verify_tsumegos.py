import os
import queue
import shutil
import threading
import time
import traceback
from decimal import Decimal

import sgfmill.sgf

from katago_controller import KatagoController
from katago_queries import KatagoQueries
from models import MODEL_9X9, get_model
from test_tsumegos_on_bots import sgf_to_gtp, point_to_sgf, parse_tsumego_sgf, PropagatingThread

TSUMEGO_DIR = "tsumegos"
DEBUG = False

cache = {}


def get_file_content(filename):
    if filename in cache:
        return cache[filename]
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        game = sgfmill.sgf.Sgf_game.from_string(content)
        cache[filename] = game
        return game


def process_file(file_path, query, x):
    filename = os.path.basename(file_path)
    game = get_file_content(file_path)
    setup_moves, _ = parse_tsumego_sgf(game)
    if len(setup_moves) < 6:
        return False
    # For move 5
    moves_up_to_4 = [m[1] for m in setup_moves[:4]]
    response = query.query(moves_up_to_4, playouts=1)
    policy = response["policy"]
    prob5 = query.get_move_probability(policy, setup_moves[4][1])
    # For move 6
    moves_up_to_5 = [m[1] for m in setup_moves[:5]]
    response = query.query(moves_up_to_5, playouts=1)
    policy = response["policy"]
    prob6 = query.get_move_probability(policy, setup_moves[5][1])
    return prob5 < x or prob6 < x

def worker(file_queue, query, lock, total_processed, fails, fail_files, x):
    while True:
        try:
            filename = file_queue.get_nowait()
        except queue.Empty:
            break
        file_path = os.path.join(TSUMEGO_DIR, filename)
        result = process_file(file_path, query, x)
        with lock:
            total_processed[0] += 1
            if result:
                fails[0] += 1
                fail_files.append(filename[:-4])
                pct = (fails[0] / total_processed[0]) * 100
                print(f"FAIL: {filename[:-4]} | Fails so far: {fails[0]}/{total_processed[0]} ({pct:.2f}%)")
        file_queue.task_done()


if __name__ == "__main__":
    controller = KatagoController()
    controller.start_engine(model=get_model(MODEL_9X9), verbose=DEBUG)
    query = KatagoQueries(controller)
    x = 0.0001

    file_queue = queue.Queue()
    lock = threading.Lock()
    total_processed = [0]
    fails = [0]
    fail_files = []
    start_time = time.time()

    for filename in os.listdir(TSUMEGO_DIR):
        if filename.endswith(".sgf"):
            file_queue.put(filename)
            if DEBUG and file_queue.qsize() >= 10:  # limit for debug
                break

    if file_queue.empty():
        print("No files to process")
        controller.stop_engine()
        exit(0)

    print(f"Processing {file_queue.qsize()} tsumegos")

    num_threads = controller.get_batch_size() * 2 if not DEBUG else 2
    threads = []
    for _ in range(num_threads):
        thread = PropagatingThread(target=worker, args=(file_queue, query, lock, total_processed, fails, fail_files, x))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    elapsed_time = time.time() - start_time
    print(f"Processed in {elapsed_time:.2f} seconds")
    print(f"Tsumegos per second: {file_queue.qsize() / elapsed_time:.2f}")

    # Create broken folder and move failed tsumegos
    broken_dir = os.path.join(TSUMEGO_DIR, "broken")
    os.makedirs(broken_dir, exist_ok=True)
    for fail_file in fail_files:
        src = os.path.join(TSUMEGO_DIR, fail_file + ".sgf")
        dst = os.path.join(broken_dir, fail_file + ".sgf")
        if os.path.exists(src):
            shutil.move(src, dst)

    print(f"Moved {len(fail_files)} failed tsumegos to {broken_dir}")

    controller.stop_engine()
