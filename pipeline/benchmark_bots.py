import math
import os
import queue
import re
import sys
import threading
import time
import traceback
from decimal import Decimal

import sgfmill.sgf

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago_controller import KatagoController
from utils.katago_queries import KatagoQueries, get_all_ranks, get_rank_number, get_rank_str, _get_rank_str
from utils.models import MODEL_9X9, MODEL_BEST_SMALL, MODEL_FULL, get_corrupted_model, get_model, get_filename_without_extension, MODELS, MODEL_HUMAN

TSUMEGO_DIR = os.path.join(PROJECT_ROOT, "tsumegos")
ATTEMPTS_DIRECTORY = os.path.join(PROJECT_ROOT, "attempts")
DEBUG = False
BATCH_SIZE = 10000

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



class SgfMoveBot:
    def __init__(self):
        pass

    def get_probability(self, game):
        setup_moves, variation_nodes = parse_tsumego_sgf(game)
        
        # In YoseTsumego, variation_nodes[0] is the correct move
        correct_node = variation_nodes[0]
        
        if correct_node.has_property("C"):
            for comment_bytes in correct_node.get_raw_list("C"):
                comment = comment_bytes.decode('utf-8')
                match = re.search(r"(\d+\.?\d*)%/([-]?\d+\.?\d*)", comment)
                if match:
                    winrate = float(match.group(1)) / 100.0
                    return max(winrate, 1.0 - winrate)
                    
        raise ValueError("Could not find winrate comment in SGF for the correct move.")

    def get_batch_size(self):
        return 1000

    def stop(self):
        pass

    def copy(self):
        return self


class HeuristicBot:
    def __init__(self):
        # Tuned parameters optimized for ~6-13 kyu players
        self.score_decay = 0.9808
        self.moves_decay = 3.4782
        self.score_weight = 2.3564
        self.moves_weight = 2.7749
        self.base_prob = 0.5968

    def _get_moves_remaining(self, game):
        _, variation_nodes = parse_tsumego_sgf(game)
        if not variation_nodes:
            return 0
        
        count = 1
        current_node = variation_nodes[0]
        while len(current_node) > 0:
            current_node = current_node[0]
            count += 1
        return count

    def get_probability(self, game: sgfmill.sgf.Sgf_game) -> float:
        _, variation_nodes = parse_tsumego_sgf(game)
        scores = []
        for node in variation_nodes:
            if not (node.has_property("B") or node.has_property("W")):
                continue

            score_found = False
            for comment_bytes in node.get_raw_list("C"):
                comment = comment_bytes.decode('utf-8')
                # Format: "100.0%/34.1"
                match = re.search(r"(\d+\.?\d*)%/([-]?\d+\.?\d*)", comment)
                if match:
                    scores.append(float(match.group(2)))
                    score_found = True
                    break
            if not score_found:
                color, move_bytes = node.get_raw_move()
                raise ValueError(f"SGF variation for move '{color.upper()} {move_bytes.decode('utf-8')}' is missing a probability/score comment.")

        if len(scores) < 2:
            score_diff = 100.0
        else:
            scores.sort(reverse=True)
            score_diff = scores[0] - scores[1]

        # Compute probability components
        p_score = 1.0 - math.exp(-score_diff / self.score_decay)
        moves_remaining = self._get_moves_remaining(game)
        p_moves = math.exp(-moves_remaining / self.moves_decay)

        # Weighted combination with base probability
        w_sum = self.score_weight + self.moves_weight
        combined = (self.score_weight * p_score + self.moves_weight * p_moves) / w_sum
        return self.base_prob + (1.0 - self.base_prob) * combined

    def copy(self):
        return HeuristicBot()

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
        

def get_all_tsumego_ids():
    """Returns a sorted list of all tsumego IDs in the tsumego directory."""
    ids = []
    for filename in os.listdir(TSUMEGO_DIR):
        if filename.endswith(".sgf"):
            ids.append(filename[:-4])
    ids.sort()
    return ids


def test_bot(bot, attempts_file, batch_ids):
    """
    Process a batch of tsumegos specified by their IDs.
    
    Args:
        bot: The bot to use for processing
        attempts_file: Path to the file storing attempt results
        batch_ids: Set of tsumego IDs to process in this batch
    
    Returns:
        None
    """
    file_queue = queue.Queue()
    lock = threading.Lock()
    probabilities = {}
    start_time = time.time()
    if os.path.exists(attempts_file):
        with open(attempts_file, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                probabilities[key] = float(value)
    
    # Queue only files from the batch that haven't been processed yet
    queued_count = 0
    for tsumego_id in batch_ids:
        if tsumego_id not in probabilities:
            file_queue.put(tsumego_id + ".sgf")
            queued_count += 1
            if DEBUG and queued_count >= 5000:
                break
    
    if file_queue.empty():
        return
    
    print(f"Generating {file_queue.qsize()} attempts for {attempts_file}")
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
    print(f"Processed: {queued_count} new out of {total_count} total")
    if total_count > 0:
        print(f"Accuracy: {(100 * solved_count / total_count):.2f}%")
    if elapsed_time > 0:
        print(f"Tsumegos per second: {queued_count / elapsed_time:.2f}")
    print(f"Time: {time.time() - start_time:.2f} seconds")
    print()
    if DEBUG:
        exit(-1)
    with open(attempts_file, 'w') as f:
        for key, value in sorted(probabilities.items()):
            f.write(f"{key} {value}\n")



# ATTEMPTS_DIRECTORY is defined at the top
SKIP_IF_EXISTS = False


def get_bot_configs():
    """
    Returns a list of (bot_factory, attempts_filename) tuples.
    bot_factory is a callable that creates the bot when called.
    """
    configs = []
    
    # Human rank bots with temperature 1.0
    for human_rank in get_all_ranks():
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_{get_rank_str(human_rank)}.txt")
        configs.append((
            lambda hr=human_rank: KataGo(get_model(MODEL_BEST_SMALL), 1, human_rank=hr, temperature=1.0),
            filename
        ))
    
    # Human rank bots with temperature 0.5, they are better players, but much worse at predicting human play.
    if False:
        for human_rank in get_all_ranks():
            filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_{get_rank_str(human_rank)}_t0.5.txt")
            configs.append((
                lambda hr=human_rank: KataGo(get_model(MODEL_BEST_SMALL), 1, human_rank=hr, temperature=0.5),
                filename
            ))
    
    # Model + temperature combinations
    for model in [MODEL_FULL, MODEL_9X9]:
        for temperature in [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]:
            model_filename = get_model(model)
            filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(model_filename)}_t{format(Decimal(str(temperature)).normalize(), 'f')}.txt")
            configs.append((
                lambda mf=model_filename, t=temperature: KataGo(mf, temperature=t),
                filename
            ))
    
    # High temperature variants for 20k rank
    for temperature in [1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2, 3, 5, 10, 50]:
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_20k_t{format(Decimal(str(temperature)).normalize(), 'f')}.txt")
        configs.append((
            lambda t=temperature: KataGo(get_model(MODEL_BEST_SMALL), human_rank=get_all_ranks()[0], temperature=t),
            filename
        ))
    
    # Heuristic bot
    configs.append((
        lambda: HeuristicBot(),
        os.path.join(ATTEMPTS_DIRECTORY, "heuristic.txt")
    ))

    # SGF Move Bot
    configs.append((
        lambda: SgfMoveBot(),
        os.path.join(ATTEMPTS_DIRECTORY, "sgf_move.txt")
    ))

    return configs


if __name__ == "__main__":
    os.makedirs(ATTEMPTS_DIRECTORY, exist_ok=True)
    
    bot_configs = get_bot_configs()
    
    # Filter out configs where SKIP_IF_EXISTS applies
    if SKIP_IF_EXISTS:
        bot_configs = [(factory, filename) for factory, filename in bot_configs if not os.path.exists(filename)]
    
    if not bot_configs:
        print("All bots already have complete attempt files.")
        list_outcomes()
        exit(0)
    
    all_ids = get_all_tsumego_ids()
    total_count = len(all_ids)
    
    iteration = 0
    while True:
        # Calculate batch for this iteration
        start_idx = iteration * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_ids = set(all_ids[start_idx:end_idx])
        
        if not batch_ids:
            print("\nAll tsumegos processed for all bots.")
            break
        
        remaining = total_count - end_idx
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}: Processing tsumegos {start_idx}-{min(end_idx, total_count) - 1} ({len(batch_ids)} files, {max(0, remaining)} remaining)")
        print(f"{'='*60}\n")
        
        for bot_factory, attempts_file in bot_configs:
            bot = bot_factory()
            try:
                test_bot(bot, attempts_file, batch_ids)
            finally:
                bot.stop()
        
        # Clear the cache between iterations to free memory
        print(f"\nClearing cache ({len(cache)} entries)...")
        cache.clear()
        
        iteration += 1
    
    list_outcomes()