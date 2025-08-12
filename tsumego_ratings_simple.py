from collections import defaultdict
from pathlib import Path
import numpy as np

from katago_queries import get_rank_number

ATTEMPTS_DIR = "attempts"


def get_tsumego_ratings():
    """
    Generates tsumego ratings based purely on bot attempt data.

    The method is as follows:
    1.  Parse all attempt files to build a success/failure matrix for each bot.
    2.  Rank bots from strongest to weakest based on their total number of solved problems.
    3.  For each tsumego, determine its difficulty by finding the rank of the 5th
        strongest bot that FAILED it. If fewer than 5 bots failed it, the weakest
        failing bot is used. If no bots failed it, it's considered very easy.
    4.  Sort the tsumegos based on this difficulty metric.
    5.  Assign a final, uniformly distributed rank from 0.0 (easiest) to 1.0 (hardest).

    Returns:
        A dictionary mapping each tsumego_id (str) to its calculated rank (float).
    """
    attempts_path = Path(ATTEMPTS_DIR)
    if not attempts_path.is_dir():
        raise FileNotFoundError(f"Attempts directory not found at '{attempts_path}'")

    # --- 1. Parse all attempts ---
    # Structure: {bot_id: {tsumego_id: success_bool}}
    bot_results = defaultdict(dict)
    all_tsumegos = set()

    print("--- Parsing Bot Attempts ---")
    for file_path in attempts_path.glob("*.txt"):
        bot_id = file_path.stem
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    tsumego_id, result = line.strip().split()
                    is_success = float(result) == 1.0
                    bot_results[bot_id][tsumego_id] = is_success
                    all_tsumegos.add(tsumego_id)
                except (ValueError, IndexError):
                    continue

    print(f"Parsed data for {len(bot_results)} bots and {len(all_tsumegos)} unique tsumegos.")

    # --- 2. Rank bots by success ---
    bot_solves = {bot: sum(results.values()) for bot, results in bot_results.items()}
    # Sort bot_ids from strongest (most solves) to weakest (fewest solves)
    ranked_bots = sorted(bot_solves.keys(), key=lambda bot: bot_solves[bot], reverse=True)
    bot_strength_rank = {bot_id: i for i, bot_id in enumerate(ranked_bots)}

    # --- 3. Determine tsumego difficulty based on failing bots ---
    tsumego_difficulty_scores = {}
    print("\n--- Calculating Tsumego Difficulty ---")

    for tsumego_id in all_tsumegos:
        # Start with strongest bots and look for up to 5 failing ones
        found_failing_bots = []
        for bot in ranked_bots:
            if not bot_results.get(bot, {}).get(tsumego_id, True):
                found_failing_bots.append(bot)
                if len(found_failing_bots) >= 5:
                    break

        if not found_failing_bots:
            # No bot failed it -> Easiest. Assign a rank equal to weakest bot.
            failure_bot = ranked_bots[-1]
        else:
            # 5th strongest failing bot
            failure_bot = found_failing_bots[-1]

        tsumego_difficulty_scores[tsumego_id] = bot_strength_rank[failure_bot]

    # --- 4. Sort tsumegos by their calculated difficulty ---
    # A lower score means a stronger bot failed it -> harder problem.
    # We sort by score descending to get easiest problems first.
    sorted_tsumegos = sorted(
        tsumego_difficulty_scores.keys(),
        key=lambda tid: tsumego_difficulty_scores[tid],
        reverse=True
    )

    # --- 5. Assign final uniform ranks ---
    uniform_ratings = {}
    total_tsumegos = len(sorted_tsumegos)
    for i, tsumego_id in enumerate(sorted_tsumegos):
        # Assign rank from 0.0 (weakest/easiest) to 1.0 (strongest/hardest)
        rank = i / (total_tsumegos - 1)
        uniform_ratings[tsumego_id] = rank

    anchors = {
        1.0: 1.0,
        get_rank_number("2k"): get_rank_number("4d"),
        0.0: 0.0
    }

    # Convert anchor points into sorted arrays for np.interp
    anchor_x = np.array(sorted(anchors.keys()))
    anchor_y = np.array([anchors[x] for x in anchor_x])

    # Create interpolated ratings
    interpolated_ratings = {}
    for tsumego_id, rating in uniform_ratings.items():
        interpolated_ratings[tsumego_id] = np.interp(rating, anchor_x, anchor_y)

    # Calculate average difference between ratings
    differences = [(interpolated_ratings[tid] - uniform_ratings[tid]) for tid in uniform_ratings.keys()]
    avg_difference = sum(differences) / len(differences)
    print(f"\nAverage difference between interpolated and final ratings: {avg_difference:.4f}")

    return interpolated_ratings
