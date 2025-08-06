import os
import queue
import subprocess
import threading
import time

import sgfmill.sgf

from katago_controller import KatagoController
from katago_queries import KatagoQueries, get_all_ranks, get_rank_str
from weights import WEIGHT_DIRECTORY, WEIGHT_9X9, WEIGHTS, WEIGHTS_CORRUPTED

TSUMEGO_DIR = "tsumegos"
DEBUG = False


def sgf_to_gtp(sgf_coord: str, size: int) -> str:
    """
    Convert SGF coordinate (e.g. 'dd') to GTP move (e.g. 'D16').
    size is board size, typically 19.
    """
    if sgf_coord == "tt":
        return "pass"

    col = ord(sgf_coord[0]) - ord('a')  # 0-based from left
    row = ord(sgf_coord[1]) - ord('a')  # 0-based from top

    # GTP column letter map ('A'...'T', skipping 'I')
    def col_to_gtp(c):
        letter = chr(ord('A') + c)
        if letter >= 'I':
            letter = chr(ord(letter) + 1)  # skip 'I'
        return letter

    gtp_col = col_to_gtp(col)
    gtp_row = size - row  # because SGF row=0 is top → GTP row=size
    gtp_coord = f"{gtp_col}{gtp_row}"
    if gtp_row < 0:
        raise Exception(f"Invalid SGF coordinate: {sgf_coord}->{gtp_coord}")
    return gtp_coord


def get_main_variation(sgf_content):
    """Extracts the main move sequence from an SGF file."""
    game = sgfmill.sgf.Sgf_game.from_string(sgf_content)
    main_sequence = []
    for node in game.get_main_sequence():
        if node.has_property("B") or node.has_property("W"):
            move = node.get_raw_move()
            color, move_coords = move[0], sgf_to_gtp(move[1].decode(), 9)
            main_sequence.append((color, move_coords))
    return main_sequence


class Pachi:

    def __init__(self, playouts):

        args = ["Pachi-12.86/pachi", "-t", "=" + str(playouts), "--nodcnn", "--nojoseki", "--nopatterns"]

        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False
        )
        self._send_gtp_command(f"boardsize 9")
        self._send_gtp_command(f"komi 7.5")
        self._playouts = playouts

    def _send_gtp_command(self, command):
        """Sends a GTP command to the Pachi process and returns the response."""
        self._process.stdin.write((command + '\n').encode('utf-8'))
        self._process.stdin.flush()

        # Read response until an empty line is encountered
        response = ""
        while True:
            line = self._process.stdout.readline().decode('utf-8').strip()
            if not line:
                break
            response += line + "\n"

        response = response.strip()

        if response.startswith("?"):
            raise Exception(f"Error from Pachi: {command} -> {response[1:].strip()}")
        return response[1:].strip()

    def reset(self):
        self._send_gtp_command("clear_board")

    def play(self, color, move):
        self._send_gtp_command(f"play {color} {move}")

    def get_probability(self, color, move):
        played_move = self._send_gtp_command(f"genmove {color}")
        return float(played_move == move)

    def get_batch_size(self):
        return 8

    def copy(self):
        return Pachi(self._playouts)


class GnuGo:

    def __init__(self, level):

        args = ["bots/gnugo", "--mode", "gtp", "--silent", "--level", str(level), "--boardsize", "9", "--komi", "7.5", "--color", "white", "--chinese-rules",
                "--nofusekidb", "--nojosekidb"]

        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False
        )
        self._send_gtp_command(f"boardsize 9")
        self._send_gtp_command(f"komi 7.5")
        self._level = level

    def _send_gtp_command(self, command):
        self._process.stdin.write((command + '\n').encode('utf-8'))
        self._process.stdin.flush()

        # Read response until an empty line is encountered
        response = ""
        while True:
            line = self._process.stdout.readline().decode('utf-8').strip()
            if not line:
                break
            response += line + "\n"

        response = response.strip()

        if response.startswith("?"):
            raise Exception(f"Error from Pachi: {command} -> {response[1:].strip()}")
        return response[1:].strip()

    def reset(self):
        self._send_gtp_command("clear_board")

    def play(self, color, move):
        self._send_gtp_command(f"play {color} {move}")

    def get_probability(self, color, move):
        played_move = self._send_gtp_command(f"genmove {color}")
        return float(played_move == move)

    def get_batch_size(self):
        return 8

    def copy(self):
        return GnuGo(self._level)


class KataGo:

    def __init__(self, weights, playouts, human_rank=None):
        self._controller = KatagoController()
        self._controller.start_engine(weight=weights, verbose=DEBUG, human=human_rank is not None)
        self._query = KatagoQueries(self._controller)
        self._moves = []
        self._playouts = playouts
        self._human_rank = human_rank

    def reset(self):
        self._moves = []

    def play(self, color, move):
        self._moves.append(move)

    def get_probability(self, color, move):
        if self._human_rank is None:
            response = self._query.query(self._moves, playouts=self._playouts)
            played_move, _, _ = self._query.get_best_move(response)
            return float(move == played_move)
        else:
            response = self._query.query_policy(self._moves, self._human_rank)
            return response["humanPolicy"][self._query.get_idx_from_move(move)]

    def get_batch_size(self):
        return self._controller.get_batch_size()

    def stop(self):
        self._controller.stop_engine()

    def copy(self):
        return self


cache = {}


def get_file_content(filename):
    if cache.get(filename):
        return cache[filename]
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        moves = get_main_variation(content)
        cache[filename] = moves
        return moves


def process_file(filename, bot):
    for i in range(3):
        try:
            all_moves = get_file_content(filename)
            moves = all_moves[:-2]
            next_move_color, next_move_coords = all_moves[-2]

            bot.reset()

            # Set up the board position by playing the moves
            for color, move in moves:
                bot.play(color, move)

            # Ask bot to predict the next move
            return bot.get_probability(next_move_color, next_move_coords)

        except Exception as e:
            if i == 2:
                raise e

    raise Exception("We should never ever get here!")


def worker(file_queue, bot, lock, probabilities):
    """
    Worker function for threads.
    Processes files from the queue until it's empty.
    """

    worker_bot = bot.copy()
    start_time = time.time()

    while True:
        try:
            filename = file_queue.get_nowait()
            queue_size = file_queue.qsize()
        except queue.Empty:
            break

        # if queue_size % 10000 == 0:
        #     time_since_start = time.time() - start_time
        #     print(time_since_start, queue_size)

        file_path = os.path.join(TSUMEGO_DIR, filename)
        probability = process_file(file_path, worker_bot)

        if probability is None:
            print(f"Error processing {filename}")
        else:
            with lock:
                probabilities[filename[:-4]] = probability

        file_queue.task_done()


def list_outcomes():
    outcomes = []
    for filename in os.listdir(ATTEMPTS_DIRECTORY):
        if not filename.endswith('.txt'):
            continue

        total = 0
        count = 0
        filepath = os.path.join(ATTEMPTS_DIRECTORY, filename)

        with open(filepath, 'r') as f:
            for line in f:
                _, value = line.strip().split()
                total += float(value)
                count += 1

        if count > 0:
            outcomes.append((filename, 100 * total / count))

    outcomes.sort(key=lambda x: x[1], reverse=True)
    for filename, score in outcomes:
        print(f"{filename}: {score:.2f}%")


def test_bot(bot, attempts_file):
    file_queue = queue.Queue()
    lock = threading.Lock()
    probabilities = {}
    start_time = time.time()

    print("Generating attempts for " + attempts_file)

    # Populate the queue with SGF files
    for filename in os.listdir(TSUMEGO_DIR):
        if filename.endswith(".sgf"):
            file_queue.put(filename)
            if DEBUG:
                if file_queue.qsize() >= 5000:
                    break

    threads = []
    for _ in range(1 if DEBUG else bot.get_batch_size() * 2):
        thread = threading.Thread(target=worker, args=(file_queue, bot, lock, probabilities))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish

    for thread in threads:
        thread.join()

    total_count = len(probabilities)
    solved_count = sum(probabilities.values())
    elapsed_time = time.time() - start_time

    print(f"Generating: {attempts_file}")
    print(f"Processed: {total_count}")
    print(f"Accuracy: {(100 * solved_count / total_count):.2f}%")
    print(f"Tsumegos per second: {total_count / elapsed_time:.2f}")
    print(f"Time: {time.time() - start_time:.2f} seconds")
    print()

    # Write results to file
    if DEBUG:
        exit(-1)
    with open(attempts_file, 'w') as f:
        for key, value in probabilities.items():
            f.write(f"{key} {value}\n")


ATTEMPTS_DIRECTORY = "attempts"

if __name__ == "__main__":

    os.makedirs(ATTEMPTS_DIRECTORY, exist_ok=True)

    for weight, percentage in WEIGHTS_CORRUPTED.items():
        for playouts in [1]:
            filename = os.path.join(ATTEMPTS_DIRECTORY, f"katago_corrupt_{percentage}_{playouts}.txt")
            if not os.path.exists(filename):
                bot = KataGo(os.path.join(WEIGHT_DIRECTORY, weight), playouts)
                test_bot(bot, filename)
                bot.stop()
            else:
                print(f"skipping {filename} because it already exists")

    for w in reversed(WEIGHTS.keys()):
        # for p in [1, 10, 100]:
        for p in [1]:
            filename = os.path.join(ATTEMPTS_DIRECTORY, f"katago_{WEIGHTS[w]}_{p}.txt")
            if not os.path.exists(filename):
                bot = KataGo(os.path.join(WEIGHT_DIRECTORY, w), p)
                test_bot(bot, filename)
                bot.stop()

    for rank in get_all_ranks():
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"katago_{get_rank_str(rank)}.txt")
        if not os.path.exists(filename):
            bot = KataGo(os.path.join(WEIGHT_DIRECTORY, WEIGHT_9X9), 1, rank)
            test_bot(bot, filename)
            bot.stop()

    # for level in [1, 5, 10]:
    #     filename = os.path.join(ATTEMPTS_DIRECTORY, f"gnugo_{level}.txt")
    #     if not os.path.exists(filename):
    #         bot = GnuGo(level)
    #         test_bot(bot, filename)
    #         bot.reset()
    #
    # for playouts in [6, 60, 600]:
    #     filename = os.path.join(ATTEMPTS_DIRECTORY, f"pachi_{playouts}.txt")
    #     if not os.path.exists(filename):
    #         bot = Pachi(playouts)
    #         test_bot(bot, filename)
    #         bot.reset()

    list_outcomes()
