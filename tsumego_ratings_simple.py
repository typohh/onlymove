from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np

from katago_queries import get_rank_number

ATTEMPTS_DIR = "attempts"

import numpy as np

import numpy as np


def find_bot_50_percent_rating(bot_id: str, uniform_ratings: dict, bot_results: dict) -> float:
    """
    Finds the tsumego rating where a bot's success rate is predicted to be 50%
    by using a library-based linear regression.

    This method uses `numpy.polyfit` to fit a line to the bot's performance data
    (1 for success, 0 for failure) across the full spectrum of tsumego ratings.
    It then calculates the rating at which this line of best fit crosses the 0.5 (50%) mark.

    Args:
        bot_id: The ID of the bot to analyze.
        uniform_ratings: Dictionary mapping tsumego_id to its uniform rank (0.0-1.0).
        bot_results: Dictionary containing the solve data for all bots.

    Returns:
        The uniform tsumego rating where the bot's success rate is predicted to be 50%.
    """
    results_for_bot = bot_results.get(bot_id)
    if not results_for_bot:
        raise ValueError(f"No results found for bot: {bot_id}")

    # Create a list of (rating, success_bool) tuples and sort by rating
    rated_successes = sorted(
        ((r, results_for_bot[t_id]) for t_id, r in uniform_ratings.items() if t_id in results_for_bot),
        key=lambda x: x[0]
    )

    num_problems = len(rated_successes)
    # Require a minimum number of data points for a meaningful regression
    if num_problems < 20:
        raise ValueError(
            f"Not enough data points ({num_problems}) for bot '{bot_id}' to perform a reliable regression."
        )

    # X is the independent variable (tsumego rating)
    # y is the dependent variable (success: 1 or 0)
    X = np.array([r for r, s in rated_successes])
    y = np.array([s for r, s in rated_successes])

    # --- Use numpy.polyfit to perform the linear regression ---
    # `np.polyfit(X, y, 1)` fits a 1st-degree polynomial (a line) and returns
    # the coefficients [m, b] for the equation y = m*x + b.
    m, b = np.polyfit(X, y, 1)

    # The slope (m) should be negative, as success rate should decrease as problem
    # difficulty (rating) increases. If it's positive, the data is likely too noisy
    # or the bot's performance is erratic.
    if m >= 0:
        # Fall back to a simple check on the overall success rate.
        # If average success is < 50%, the bot is weak (anchor at 0.0). Otherwise, strong (anchor at 1.0).
        return 0.0 if np.mean(y) < 0.5 else 1.0

    # Solve for x where y = 0.5
    # 0.5 = m*x + b  =>  x = (0.5 - b) / m
    rating_at_50_percent = (0.5 - b) / m

    # Clamp the result between 0.0 and 1.0. A value outside this range means the
    # 50% crossing point is predicted to be easier than the easiest problem or
    # harder than the hardest problem.
    # return float(np.clip(rating_at_50_percent, 0.0, 1.0))
    return 1.0 / (1.0 + np.exp(-4.0 * (rating_at_50_percent - 0.5)))


def get_tsumego_ratings():
    """
    Generates tsumego ratings based on bot attempt data and calibrates them
    using a specific bot's 50% performance point as an anchor.

    The method is as follows:
    1.  Parse all attempt files to build a success/failure matrix for each bot.
    2.  Rank bots from strongest to weakest based on their total number of solved problems.
    3.  For each tsumego, determine its difficulty by finding the rank of the 5th
        strongest bot that FAILED it. This establishes a relative ordering.
    4.  Sort the tsumegos based on this difficulty metric.
    5.  Assign a preliminary, uniformly distributed rank from 0.0 (easiest) to 1.0 (hardest).
    6.  Select an "anchor bot" and find the uniform rank where it solves ~50% of problems.
    7.  Create anchor points mapping this 50% point to a desired real-world rank (e.g., 4d).
    8.  Interpolate the final ratings based on these dynamic anchors.

    Returns:
        A dictionary mapping each tsumego_id (str) to its calculated, calibrated rank (float).
    """
    attempts_path = Path(ATTEMPTS_DIR)
    if not attempts_path.is_dir():
        raise FileNotFoundError(f"Attempts directory not found at '{attempts_path}'")

    # --- 1. Parse all attempts ---
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

    # REVISED: Fail loudly if no data was loaded. This prevents the script from
    # producing meaningless results or crashing with obscure errors later on.
    if not all_tsumegos:
        raise ValueError(
            "Parsing completed, but no tsumego problems were found. "
            f"Check the attempts directory ('{ATTEMPTS_DIR}') and file formats."
        )

    print(f"Parsed data for {len(bot_results)} bots and {len(all_tsumegos)} unique tsumegos.")

    # --- 2. Rank bots by success ---
    bot_solves = {bot: sum(results.values()) for bot, results in bot_results.items()}
    ranked_bots = sorted(bot_solves.keys(), key=lambda bot: bot_solves[bot], reverse=True)
    bot_strength_rank = {bot_id: i for i, bot_id in enumerate(ranked_bots)}

    if not ranked_bots:
        raise ValueError("No bots found after parsing results.")

    # --- 3. Determine tsumego difficulty based on 6th strongest failing bot ---
    tsumego_difficulty_scores = {}
    print("\n--- Calculating Tsumego Difficulty ---")
    for tsumego_id in all_tsumegos:
        found_failing_bots = []
        for bot in ranked_bots:
            if not bot_results.get(bot, {}).get(tsumego_id, True):
                found_failing_bots.append(bot)
                if len(found_failing_bots) >= 6:
                    break

        failure_bot = found_failing_bots[-1] if found_failing_bots else ranked_bots[-1]
        tsumego_difficulty_scores[tsumego_id] = bot_strength_rank[failure_bot]

    # --- 4. Sort tsumegos by their calculated difficulty ---
    sorted_tsumegos = sorted(
        tsumego_difficulty_scores.keys(),
        key=lambda tid: tsumego_difficulty_scores[tid],
        reverse=True
    )

    # --- 5. Assign final uniform ranks ---
    uniform_ratings = {}
    total_tsumegos = len(sorted_tsumegos)
    for i, tsumego_id in enumerate(sorted_tsumegos):
        rank = i / (total_tsumegos - 1) if total_tsumegos > 1 else 0.0
        uniform_ratings[tsumego_id] = rank

    # -- assign rank to each bot ---

    bot_ratings = {}

    for bot_id in bot_results.keys():
        bot_ratings[bot_id] = find_bot_50_percent_rating(bot_id, uniform_ratings, bot_results)

    sorted(bot_ratings)
    for bot_id in sorted(bot_ratings, key=bot_ratings.get, reverse=True):
        print(bot_id, bot_ratings[bot_id])

    # --- 6. Define bot-based anchors for interpolation ---
    print("\n--- Calibrating Ratings Using Bot Performance Anchor ---")

    anchors = {
        1.0: 1.0,
        find_bot_50_percent_rating("kata1-b6c96-s175395328-d26788732_corrupt21_1", uniform_ratings, bot_results): get_rank_number("4d"),
        0.0: 0.0
    }

    pprint(anchors)

    # --- 7. Interpolate final ratings based on anchors ---
    anchor_x = np.array(sorted(anchors.keys()))
    anchor_y = np.array([anchors[x] for x in anchor_x])

    interpolated_ratings = {}
    for tsumego_id, rating in uniform_ratings.items():
        interpolated_ratings[tsumego_id] = np.interp(rating, anchor_x, anchor_y)

    differences = [(interpolated_ratings[tid] - uniform_ratings[tid]) for tid in uniform_ratings.keys()]
    avg_difference = sum(differences) / len(differences)
    print(f"\nAverage difference between interpolated and uniform ratings: {avg_difference:.4f}")

    return interpolated_ratings


def reading_statistics():
    """
    Analyzes how additional reading (playouts) affects bot performance.
    
    This function examines bot IDs that contain playout information in the format
    'name_x' where 'x' is the number of playouts, and determines:
    1. How many bots show performance changes with additional reading
    2. How much reading is needed to see significant performance improvements
    """
    attempts_path = Path(ATTEMPTS_DIR)
    if not attempts_path.is_dir():
        raise FileNotFoundError(f"Attempts directory not found at '{attempts_path}'")

    # Parse all attempts
    bot_results = defaultdict(dict)
    all_tsumegos = set()

    print("--- Parsing Bot Attempts for Reading Analysis ---")
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

    if not all_tsumegos:
        raise ValueError("No tsumego problems found in attempts directory.")

    # Group bots by base name (without playout count)
    bot_groups = defaultdict(list)
    for bot_id in bot_results.keys():
        # Extract playout count from bot ID (format: name_x where x is playouts)
        if '_' in bot_id:
            parts = bot_id.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                base_name = '_'.join(parts[:-1])
                playouts = int(parts[-1])
                bot_groups[base_name].append((playouts, bot_id))
            else:
                # Bot ID doesn't follow the expected pattern, skip for this analysis
                continue
        # Bots without playout info are not included in this analysis

    # Analyze reading effects for each bot group
    print(f"\n--- Reading Statistics for {len(bot_groups)} Bot Groups ---")
    
    total_bots_with_reading_data = 0
    bots_showing_improvement = 0
    bots_with_insufficient_data = 0
    
    for base_name, bots in bot_groups.items():
        if len(bots) < 2:
            # Need at least 2 bots with different playouts to compare
            bots_with_insufficient_data += 1
            continue
            
        total_bots_with_reading_data += 1
        
        # Sort bots by playout count
        sorted_bots = sorted(bots, key=lambda x: x[0])
        playout_counts = [b[0] for b in sorted_bots]
        bot_ids = [b[1] for b in sorted_bots]
        
        # Calculate success rates for each bot
        success_rates = []
        for bot_id in bot_ids:
            results = bot_results[bot_id]
            if results:
                success_rate = sum(results.values()) / len(results)
                success_rates.append(success_rate)
            else:
                success_rates.append(0.0)
        
        # Check if additional reading improves performance
        shows_improvement = False
        for i in range(1, len(success_rates)):
            if success_rates[i] > success_rates[i-1]:
                shows_improvement = True
                break
        
        if shows_improvement:
            bots_showing_improvement += 1
            
        # Print detailed statistics for this bot group
        print(f"\nBot Group: {base_name}")
        print(f"  Playouts -> Success Rate")
        for playouts, success_rate in zip(playout_counts, success_rates):
            print(f"  {playouts:>8} -> {success_rate:.4f}")
        
        if shows_improvement:
            print("  Status: IMPROVEMENT with additional reading")
        else:
            print("  Status: NO IMPROVEMENT with additional reading")
    
    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Total bot groups with sufficient reading data: {total_bots_with_reading_data}")
    print(f"Bot groups showing improvement with more reading: {bots_showing_improvement}")
    
    if total_bots_with_reading_data > 0:
        improvement_percentage = (bots_showing_improvement / total_bots_with_reading_data) * 100
        print(f"Percentage of bots improved with reading: {improvement_percentage:.1f}%")
    else:
        print("No bots with sufficient reading data found for comparison.")
        
    if bots_with_insufficient_data > 0:
        print(f"Bots with insufficient data for reading analysis: {bots_with_insufficient_data}")


if __name__ == "__main__":
    get_tsumego_ratings()
    reading_statistics()