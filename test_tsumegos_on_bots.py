import os
import queue
import re
import threading
import time
import traceback
from decimal import Decimal

import sgfmill.sgf

from katago_controller import KatagoController
from katago_queries import KatagoQueries, get_all_ranks, get_rank_number, get_rank_str, _get_rank_str
from models import MODEL_9X9, MODEL_BEST_SMALL, MODEL_FULL, get_corrupted_model, get_model, get_filename_without_extension, MODELS, MODEL_HUMAN

TSUMEGO_DIR = "tsumegos"
DEBUG = False

class PropagatingThread(threading.Thread):
    def run(self):
        self.exc = None
        try:
            super().run()
        except Exception as e:
            self.exc = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc


def sgf_to_gtp(sgf_coord: str, size: int) -> str:
    if sgf_coord == "tt":
        return "pass"
    col = ord(sgf_coord[0]) - ord('a')
    row = ord(sgf_coord[1]) - ord('a')

    def col_to_gtp(c):
        letter = chr(ord('A') + c)
        if letter >= 'I':
            letter = chr(ord(letter) + 1)
        return letter

    gtp_col = col_to_gtp(col)
    gtp_row = size - row
    gtp_coord = f"{gtp_col}{gtp_row}"
    if gtp_row < 0:
        raise Exception(f"Invalid SGF coordinate: {sgf_coord}->{gtp_coord}")
    return gtp_coord


def point_to_sgf(point):
    """Converts a (row, col) tuple from sgfmill to an SGF coordinate string like 'pd'."""
    row, col = point
    return f"{chr(ord('a') + col)}{chr(ord('a') + row)}"


def parse_tsumego_sgf(game: sgfmill.sgf.Sgf_game):
    """
    Correctly parses a tsumego SGF file by finding the node where the path branches.
    Uses the correct, list-like API for sgfmill Tree_node objects.
    """
    setup_moves = []
    root_node = game.get_root()
    size = game.get_size()

    for prop, color_char in [("AB", "B"), ("AW", "W")]:
        if root_node.has_property(prop):
            for point_tuple in root_node.get(prop):
                sgf_string = point_to_sgf(point_tuple)
                setup_moves.append((color_char, sgf_to_gtp(sgf_string, size)))

    current_node = root_node
    while len(current_node) == 1:
        if current_node.has_property("B") or current_node.has_property("W"):
            color, move_bytes = current_node.get_raw_move()
            move_sgf = move_bytes.decode('utf-8')
            setup_moves.append((color, sgf_to_gtp(move_sgf, size)))
        current_node = current_node[0]

    if current_node.has_property("B") or current_node.has_property("W"):
        color, move_bytes = current_node.get_raw_move()
        move_sgf = move_bytes.decode('utf-8')
        setup_moves.append((color, sgf_to_gtp(move_sgf, size)))

    variation_nodes = list(current_node)
    if not variation_nodes:
        raise ValueError("SGF file is linear and contains no variations to analyze.")

    return setup_moves, variation_nodes


class HeuristicBot:
    def __init__(self, accuracy):
        self.accuracy = accuracy

    def get_probability(self, game: sgfmill.sgf.Sgf_game) -> float:
        _, variation_nodes = parse_tsumego_sgf(game)
        scores = []
        for node in variation_nodes:
            if not (node.has_property("B") or node.has_property("W")):
                continue

            score_found = False
            for comment_bytes in node.get_raw_list("C"):
                comment = comment_bytes.decode('utf-8')
                match = re.search(r"Score: ([-]?\d+\.?\d*)", comment)
                if match:
                    scores.append(float(match.group(1)))
                    score_found = True
                    break
            if not score_found:
                color, move_bytes = node.get_raw_move()
                raise ValueError(f"SGF variation for move '{color.upper()} {move_bytes.decode('utf-8')}' is missing a 'Score:' comment.")

        num_scores = len(scores)
        if num_scores < 3:
            return 1.0
        scores.sort(reverse=True)
        best_score, second_best_score = scores[0], scores[1]
        epsilon = 0.5
        if num_scores >= 5:
            nth_best_score = scores[4]
        elif num_scores == 4:
            nth_best_score = scores[3]
        else:
            nth_best_score = scores[2]
        difficulty = 1 - (epsilon + best_score - second_best_score) / (epsilon + best_score - nth_best_score)
        return 1.0 if difficulty < self.accuracy else 0.0

    def copy(self):
        return HeuristicBot(self.accuracy)

    def stop(self):
        pass

    def get_batch_size(self):
        return os.cpu_count()


class KataGo:
    def __init__(self, weights, playouts=1, human_rank=None, temperature=1.0):
        self._controller = KatagoController()
        self._controller.start_engine(model=weights, verbose=DEBUG, human=human_rank is not None)
        self._query = KatagoQueries(self._controller)
        self._playouts = playouts
        self._human_rank = human_rank
        self._temperature = temperature

    def get_probability(self, game):
        for i in range(3):
            try:
                setup_moves, variation_nodes = parse_tsumego_sgf(game)
                correct_node = variation_nodes[0]
                _, next_move_bytes = correct_node.get_raw_move()
                next_move_sgf = next_move_bytes.decode('utf-8')
                next_move_gtp = sgf_to_gtp(next_move_sgf, game.get_size())
                setup_moves_gtp = [move[1] for move in setup_moves]

                if self._human_rank:
                    response = self._query.query(setup_moves_gtp,human_rank=self._human_rank)
                    policy = response["humanPolicy"]
                else:
                    response = self._query.query(setup_moves_gtp, playouts=self._playouts)
                    policy = response["policy"]

                policy = self._query.adjust_temperature(policy,self._temperature)
                return self._query.get_move_probability(policy, next_move_gtp)
            except Exception as e:
                traceback.print_exc()
                exit(-1)
                print(f"KataGo attempt {i + 1}/3 failed. Retrying... Error: {e}")
                if i == 2:
                    print(f"FATAL: KataGo failed after 3 attempts. Raising exception.")
                    raise e
        raise Exception("KataGo failed all retry attempts.")

    def get_batch_size(self):
        return self._controller.get_batch_size()

    def stop(self):
        self._controller.stop_engine()

    def copy(self):
        return self

class KataGoFullGame:
    def __init__(self, weights, human_rank):
        self._controller = KatagoController()
        self._controller.start_engine(model=weights, verbose=DEBUG, human=True)
        self._query = KatagoQueries(self._controller)
        self._human_rank = human_rank

    def get_probability(self, game):
        wins = 0
        losses = 0
        for i in range(5):
            try:
                setup_moves, variation_nodes = parse_tsumego_sgf(game)
                moves = [move[1] for move in setup_moves]
                black_turn = True
                resign_ready_count = 0
                while ( moves[-1] != 'pass' or moves[-2] != 'pass' ) and resign_ready_count < 3:
                    if black_turn:
                        move,_ = self._query.query_human(moves, self._human_rank)
                    else:
                        response = self._query.query(moves, playouts=1)
                        move, winrate, score, _ = self._query.get_best_move(response)
                        if winrate < 0.01:
                            resign_ready_count += 1
                        else:
                            resign_ready_count = 0
                    moves.append(move)
                    black_turn = not black_turn
                score,winrate = self._query.get_position_info(moves)
                if winrate > 0.5:
                    wins += 1
                else :
                    losses += 1
            except Exception as e:
                print(f"KataGo attempt {i + 1}/5 failed. Retrying... Error: {e}")

        if wins == 0 and losses == 0:
            raise Exception("KataGo failed all retry attempts.")
        return wins / (wins + losses)

    def get_batch_size(self):
        return self._controller.get_batch_size()

    def stop(self):
        self._controller.stop_engine()

    def copy(self):
        return self

cache = {}


def get_file_content(filename):
    if filename in cache:
        return cache[filename]
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        game = sgfmill.sgf.Sgf_game.from_string(content)
        cache[filename] = game
        return game


def process_file(filename, bot):
    game = get_file_content(filename)
    return bot.get_probability(game)




def worker(file_queue, bot, lock, probabilities):
    worker_bot = bot.copy()
    while True:
        try:
            filename = file_queue.get_nowait()
        except queue.Empty:
            break
        file_path = os.path.join(TSUMEGO_DIR, filename)
        probability = process_file(file_path, worker_bot)
        with lock:
            probabilities[filename[:-4]] = probability
        file_queue.task_done()



def list_outcomes():
    outcomes = []
    for filename in os.listdir(ATTEMPTS_DIRECTORY):
        if not filename.endswith('.txt'):
            continue
        total, count = 0, 0
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
    if os.path.exists(attempts_file):
        with open(attempts_file, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                probabilities[key] = float(value)
    for filename in os.listdir(TSUMEGO_DIR):
        if filename.endswith(".sgf"):
            key = filename[:-4]
            if key not in probabilities:
                file_queue.put(filename)
                if DEBUG and file_queue.qsize() >= 5000:
                    break
    if file_queue.empty():
        return
    print(f"generating {file_queue.qsize()} attempts for {attempts_file}")
    threads = []
    num_threads = 1 if DEBUG else (bot.get_batch_size() * 2 if isinstance(bot, KataGo) else 8)
    for _ in range(num_threads):
        thread = PropagatingThread(target=worker, args=(file_queue, bot, lock, probabilities))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    total_count = len(probabilities)
    solved_count = sum(probabilities.values())
    elapsed_time = time.time() - start_time
    print(f"Saving: {attempts_file}")
    print(f"Processed: {total_count}")
    if total_count > 0:
        print(f"Accuracy: {(100 * solved_count / total_count):.2f}%")
    if elapsed_time > 0:
        print(f"Tsumegos per second: {total_count / elapsed_time:.2f}")
    print(f"Time: {time.time() - start_time:.2f} seconds")
    print()
    if DEBUG:
        exit(-1)
    with open(attempts_file, 'w') as f:
        for key, value in sorted(probabilities.items()):
            f.write(f"{key} {value}\n")


ATTEMPTS_DIRECTORY = "attempts"
SKIP_IF_EXISTS = True

if __name__ == "__main__":

    os.makedirs(ATTEMPTS_DIRECTORY, exist_ok=True)

    for human_rank in get_all_ranks():
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_{get_rank_str(human_rank)}.txt")
        if SKIP_IF_EXISTS and os.path.exists(filename):
            continue
        bot = KataGo(get_model(MODEL_BEST_SMALL), 1 , human_rank=human_rank, temperature=1.0)
        test_bot(bot, filename)
        bot.stop()

    # for model in MODELS[0:]:
    #     model_filename = get_model(model)
    #     filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(model_filename)}.txt")
    #     if SKIP_IF_EXISTS and os.path.exists(filename):
    #         continue
    #     bot = KataGo(model_filename)
    #     test_bot(bot, filename)
    #     bot.stop()
    #
    for model in [MODEL_FULL,MODEL_9X9]:
        for temperature in [1,0.8,0.6,0.4,0.2,0.1,0.05,0.02]:
            model_filename = get_model(model)
            filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(model_filename)}_t{format(Decimal(str(temperature)).normalize(), 'f') }.txt")
            if SKIP_IF_EXISTS and os.path.exists(filename):
                continue
            bot = KataGo(model_filename, temperature=temperature)
            test_bot(bot, filename)
            bot.stop()

    for temperature in [1.1,1.2,1.3,1.4,1.5,1.7,2,3,5,10,50]:
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_20k_t{format(Decimal(str(temperature)).normalize(), 'f')}.txt")
        if SKIP_IF_EXISTS and os.path.exists(filename):
            continue
        bot = KataGo(get_model(MODEL_BEST_SMALL), human_rank=get_all_ranks()[0], temperature=temperature)
        test_bot(bot, filename)
        bot.stop()

    # for human_rank in get_all_ranks():
    #     model_filename = get_model(MODEL_FULL)
    #     filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(model_filename)}_{playouts}.txt")
    #     if SKIP_IF_EXISTS and os.path.exists(filename):
    #         continue
    #     bot = KataGoFullGame(model_filename,human_rank=human_rank)
    #     test_bot(bot, filename)
    #     bot.stop()

    # for accuracy in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05 ]:
    #     filename = os.path.join(ATTEMPTS_DIRECTORY, f"heuristic_{str(accuracy)[2:]}.txt")
    #     if SKIP_IF_EXISTS and os.path.exists(filename):
    #         continue
    #     bot = HeuristicBot(accuracy)
    #     test_bot(bot, filename)
    #     bot.stop()

    # for c in range(1, 120):
    #     corruption = c / 200
    #     for playouts in [1, 3, 10, 30]:
    #         model_filename = get_corrupted_model(corruption)
    #         filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(model_filename) + f'_{playouts}.txt'}")
    #         if SKIP_IF_EXISTS and os.path.exists(filename):
    #             continue
    #         bot = KataGo(model_filename, playouts)
    #         test_bot(bot, filename)
    #         bot.stop()


    list_outcomes()