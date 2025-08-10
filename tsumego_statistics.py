import os
import shutil
from pathlib import Path
from collections import defaultdict
import random

# --- Configuration ---
# The directory containing the bot performance files.
ATTEMPTS_DIR = "attempts"
# The Tsumego ID you want to analyze.
# TSUMEGO_TO_ANALYZE = "5180d01a"
# TSUMEGO_TO_ANALYZE = "78db4e47"
TSUMEGO_TO_ANALYZE = "659d7b7f"


def get_bot_type_from_filename(filename: str) -> str:
    """
    Extracts the bot type from a filename (e.g., 'botA_engine.txt' -> 'engine').
    """
    stem = Path(filename).stem
    if '_' in stem:
        return stem.rsplit('_', 1)[1]
    return "unknown"


def load_and_process_data(directory: str) -> tuple[dict, list] or tuple[None, None]:
    """
    Loads data from all .txt files in the specified directory.

    Args:
        directory: The path to the directory containing bot files.

    Returns:
        A tuple containing:
        - A dictionary of all bots' data.
        - A list of bot items sorted by strength in descending order.
        Returns (None, None) if the directory or files are not found.
    """
    base_path = Path(directory)
    if not base_path.is_dir():
        print(f"Error: Directory '{directory}' not found.")
        return None, None

    files = list(base_path.glob("*.txt"))
    if not files:
        print(f"Error: No .txt files found in '{directory}'.")
        return None, None

    bots_data = {}
    for file_path in files:
        bot_name = file_path.stem
        bot_type = get_bot_type_from_filename(file_path.name)

        failures = set()
        total_count = 0
        success_count = 0

        with open(file_path, 'r') as f:
            for line in f:
                try:
                    tsumego_id, result_str = line.strip().split()
                    total_count += 1
                    if float(result_str) == 1.0:
                        success_count += 1
                    else:
                        failures.add(tsumego_id)
                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed line in {file_path.name}")
                    continue

        if total_count > 0:
            strength = success_count / total_count
            bots_data[bot_name] = {
                'type': bot_type,
                'strength': strength,
                'failures': failures,
                'total_solved': success_count,
                'total_tsumegos': total_count
            }

    # Sort bots by strength (strongest first)
    sorted_bots_list = sorted(
        bots_data.items(),
        key=lambda item: item[1]['strength'],
        reverse=True
    )

    return bots_data, sorted_bots_list


def analyze_tsumego_difficulty(tsumego_id: str, sorted_bots: list):
    """
    Analyzes a specific tsumego to find failure points and estimate difficulty.
    """
    print(f"\n--- Analysis for Tsumego ID: {tsumego_id} ---")

    strongest_failing_bot = None
    weakest_passing_bot = None

    # Find the strongest bot that failed
    for bot_name, data in sorted_bots:
        if tsumego_id in data['failures']:
            strongest_failing_bot = (bot_name, data['strength'])
            break  # Found the strongest one, no need to look further

    # Find the weakest bot that passed by iterating from the bottom up
    for bot_name, data in reversed(sorted_bots):
        if tsumego_id not in data['failures']:
            weakest_passing_bot = (bot_name, data['strength'])
            break  # Found the weakest one

    # --- Report Results ---
    if strongest_failing_bot:
        name, strength = strongest_failing_bot
        print(f"-> Strongest bot that FAILS it: '{name}' (Strength: {strength:.2%})")
    else:
        print("-> This tsumego was SOLVED by all bots.")

    if weakest_passing_bot:
        name, strength = weakest_passing_bot
        print(f"-> Weakest bot that SOLVES it: '{name}' (Strength: {strength:.2%})")
    else:
        print("-> This tsumego was FAILED by all bots.")

    # --- Estimate 50% Crossover Strength ---
    if strongest_failing_bot and weakest_passing_bot:
        s_fail = strongest_failing_bot[1]
        s_pass = weakest_passing_bot[1]
        # The crossover point is between these two bots. A good approximation is the average.
        midpoint_strength = (s_fail + s_pass) / 2
        print(f"\n=> Approximate 50% failure rate occurs at strength level: {midpoint_strength:.2%}")
    elif not strongest_failing_bot:
        print("\n=> Cannot estimate 50% level; all bots passed.")
    elif not weakest_passing_bot:
        print("\n=> Cannot estimate 50% level; all bots failed.")


def analyze_by_bot_type(tsumego_id: str, all_bots_data: dict):
    """
    Analyzes tsumego failure rates grouped by bot type.
    """
    print(f"\n--- Failure Rate by Bot Type for Tsumego ID: {tsumego_id} ---")

    type_stats = defaultdict(lambda: {'fails': 0, 'total': 0})

    for bot_name, data in all_bots_data.items():
        bot_type = data['type']
        type_stats[bot_type]['total'] += 1
        if tsumego_id in data['failures']:
            type_stats[bot_type]['fails'] += 1

    if not type_stats:
        print("No bot types found to analyze.")
        return

    for bot_type, stats in sorted(type_stats.items()):
        fail_rate = (stats['fails'] / stats['total']) if stats['total'] > 0 else 0
        print(f"- Type '{bot_type}': {stats['fails']} of {stats['total']} bots failed ({fail_rate:.2%})")


def create_dummy_files(directory: str, num_bots: int, num_tsumegos: int):
    """
    Generates a directory with dummy bot performance files for demonstration.
    """
    print("--- Setting up a dummy 'attempts' directory for demonstration ---")
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    bot_types = ['mcts', 'neural', 'hybrid']
    tsumego_ids = [f"tsumego_{i}" for i in range(1, num_tsumegos + 1)]

    for i in range(1, num_bots + 1):
        bot_type = random.choice(bot_types)
        bot_name = f"Bot_{i:02d}_{bot_type}"
        file_path = Path(directory) / f"{bot_name}.txt"

        # Bots with higher index are stronger
        base_success_prob = 0.2 + 0.8 * (i / num_bots)

        with open(file_path, 'w') as f:
            for t_id in tsumego_ids:
                # Make some tsumegos harder than others
                tsumego_difficulty = int(t_id.split('_')[1]) / num_tsumegos
                success = 1.0 if (base_success_prob * (1 - tsumego_difficulty)) > random.random() else 0.0
                f.write(f"{t_id} {success}\n")
    print(f"Created {num_bots} dummy bot files in '{directory}'.\n")


def main():
    """Main execution function."""
    # To run with real data, comment out or delete this call to create_dummy_files()
    create_dummy_files(ATTEMPTS_DIR, num_bots=20, num_tsumegos=100)

    all_bots, sorted_bots_list = load_and_process_data(ATTEMPTS_DIR)

    if not all_bots:
        return  # Exit if data loading failed

    # 1. Print overall bot strength ranking
    print("--- Bot Strength Ranking (Overall Success Rate) ---")
    for bot_name, data in sorted_bots_list:
        print(
            f"- {bot_name:<20} (Type: {data['type']:<8}) "
            f"| Strength: {data['strength']:>7.2%} "
            f"({data['total_solved']}/{data['total_tsumegos']} solved)"
        )

    # 2. Analyze the specific tsumego across all bots
    analyze_tsumego_difficulty(TSUMEGO_TO_ANALYZE, sorted_bots_list)

    # 3. Analyze the specific tsumego by bot type
    analyze_by_bot_type(TSUMEGO_TO_ANALYZE, all_bots)


if __name__ == "__main__":
    main()