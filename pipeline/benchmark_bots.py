import argparse
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

from utils.configurations import ALL_BOARD_SIZES
from utils.katago_controller import KatagoController
from utils.katago_queries import KatagoQueries, get_all_ranks, get_rank_str
from utils.models import MODEL_BEST_SMALL, MODEL_FULL, get_model, get_filename_without_extension, MODEL_HUMAN

# Set in main() from the mandatory size arg: tsumegos/<size>/ and attempts/<size>/.
TSUMEGO_DIR = None
ATTEMPTS_DIRECTORY = None
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


def point_to_sgf(point, size):
    """Converts an sgfmill (row, col) point to an SGF coordinate string like 'pd'.

    sgfmill points count rows from the BOTTOM ((0,0) = lower-left); an SGF
    coordinate's row letter counts from the TOP. Flip the row (size-1-row) so
    AB/AW setup stones resolve to the same square as move-node coordinates,
    which come straight from the raw SGF (top-origin). Without the flip the
    setup stones were vertically mirrored against the moves — latent only
    because current tsumego SGFs place setup via move nodes and carry no
    AB/AW."""
    row, col = point
    return f"{chr(ord('a') + col)}{chr(ord('a') + (size - 1 - row))}"


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
                sgf_string = point_to_sgf(point_tuple, size)
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


class KataGo:
    def __init__(self, weights, board_size, playouts=1, human_rank=None, temperature=1.0):
        self._controller = KatagoController()
        self._controller.start_engine(model=weights, board_size=board_size, verbose=DEBUG, human=human_rank is not None)
        self._query = KatagoQueries(self._controller, board_size)
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



def list_outcomes():
    outcomes = []
    for filename in os.listdir(ATTEMPTS_DIRECTORY):
        if not filename.endswith('.txt'):
            continue
        total, count = 0, 0
        filepath = os.path.join(ATTEMPTS_DIRECTORY, filename)
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue  # skip blank / legacy-truncated lines
                total += float(parts[1])
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
                parts = line.split()
                if len(parts) != 2:
                    continue  # skip blank / legacy-truncated lines
                probabilities[parts[0]] = float(parts[1])
    
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
    # Join every worker even if one raised: a single worker's failure (e.g.
    # KataGo exhausting its retries) must not discard the results the other
    # workers already wrote into `probabilities`. Remember the first failure
    # and re-raise it only AFTER persisting the partial progress below.
    worker_exc = None
    for thread in threads:
        try:
            thread.join()
        except Exception as e:
            if worker_exc is None:
                worker_exc = e
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
    # Atomic write: a kill mid-write must not truncate the attempts file (a
    # half-written final line would crash the next run's parse) nor lose the
    # bot's accumulated results. Write a sibling temp file, then os.replace.
    tmp_path = f"{attempts_file}.tmp"
    with open(tmp_path, 'w') as f:
        for key, value in sorted(probabilities.items()):
            f.write(f"{key} {value}\n")
    os.replace(tmp_path, attempts_file)
    if worker_exc is not None:
        raise worker_exc


def batch_has_missing(attempts_file, batch_ids):
    """True iff some id in batch_ids is not yet recorded in attempts_file — i.e. this
    bot has work to do for this batch. Lets main() skip the (expensive) engine spin-up
    for batch×bot cells with nothing to benchmark: a no-op test_bot otherwise still
    pays a full KataGo model-load + OpenCL init. On an incremental top-up (most cells
    have zero missing) this makes the run cost proportional to the new work, not to the
    corpus size. Cheap — reads only the key column of the .txt."""
    if not os.path.exists(attempts_file):
        return True  # nothing benchmarked yet → all of batch_ids is missing
    done = set()
    with open(attempts_file) as f:
        for line in f:
            parts = line.split()
            if parts:
                done.add(parts[0])
    return any(tid not in done for tid in batch_ids)


# TSUMEGO_DIR / ATTEMPTS_DIRECTORY are set in main() from the size arg.
SKIP_IF_EXISTS = False


def get_bot_configs(board_size):
    """
    Returns a list of (bot_factory, attempts_filename) tuples.
    bot_factory is a callable that creates the bot when called.
    """
    configs = []

    # Human rank bots with temperature 1.0
    for human_rank in get_all_ranks():
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_{get_rank_str(human_rank)}.txt")
        configs.append((
            lambda hr=human_rank: KataGo(get_model(MODEL_BEST_SMALL), board_size, 1, human_rank=hr, temperature=1.0),
            filename
        ))

    # Model + temperature combinations
    for model in [MODEL_FULL]:
        for temperature in [2.0, 1.5, 1.2, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]:
            model_filename = get_model(model)
            filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(model_filename)}_t{format(Decimal(str(temperature)).normalize(), 'f')}.txt")
            configs.append((
                lambda mf=model_filename, t=temperature: KataGo(mf, board_size, temperature=t),
                filename
            ))

    # High temperature variants for 20k rank
    for temperature in [1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2, 3, 5, 10, 50]:
        filename = os.path.join(ATTEMPTS_DIRECTORY, f"{get_filename_without_extension(MODEL_HUMAN)}_20k_t{format(Decimal(str(temperature)).normalize(), 'f')}.txt")
        configs.append((
            lambda t=temperature: KataGo(get_model(MODEL_BEST_SMALL), board_size, human_rank=get_all_ranks()[0], temperature=t),
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


def main():
    parser = argparse.ArgumentParser(description='Benchmark bots on generated tsumegos')
    parser.add_argument('size', type=int, choices=ALL_BOARD_SIZES,
                        help='Board size to benchmark (one of %(choices)s)')
    args = parser.parse_args()

    global TSUMEGO_DIR, ATTEMPTS_DIRECTORY
    TSUMEGO_DIR = os.path.join(PROJECT_ROOT, "tsumegos", str(args.size))
    ATTEMPTS_DIRECTORY = os.path.join(PROJECT_ROOT, "attempts", str(args.size))

    os.makedirs(ATTEMPTS_DIRECTORY, exist_ok=True)

    bot_configs = get_bot_configs(args.size)

    # Filter out configs where SKIP_IF_EXISTS applies
    if SKIP_IF_EXISTS:
        bot_configs = [(factory, filename) for factory, filename in bot_configs if not os.path.exists(filename)]

    if not bot_configs:
        print("All bots already have complete attempt files.")
        list_outcomes()
        return

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
        
        ran = 0
        for bot_factory, attempts_file in bot_configs:
            # Skip the engine spin-up when this bot already covers the whole batch.
            # KataGo startup (model load + OpenCL) dominates and is otherwise paid for
            # every batch×bot cell even when test_bot would no-op — so an incremental
            # top-up costs as much as a full rebuild. This makes it proportional to the
            # new work. (See batch_has_missing.)
            if not batch_has_missing(attempts_file, batch_ids):
                continue
            ran += 1
            bot = bot_factory()
            try:
                test_bot(bot, attempts_file, batch_ids)
            finally:
                bot.stop()
        print(f"  batch {iteration}: {ran}/{len(bot_configs)} bots had new work "
              f"({len(bot_configs) - ran} engine startups skipped)")

        # Clear the cache between iterations to free memory
        print(f"\nClearing cache ({len(cache)} entries)...")
        cache.clear()
        
        iteration += 1

    list_outcomes()


if __name__ == "__main__":
    main()