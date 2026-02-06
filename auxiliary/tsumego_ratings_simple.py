import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from pprint import pprint

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago_queries import get_rank_number

ATTEMPTS_DIR = os.path.join(PROJECT_ROOT, "attempts")


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
    3.  For each tsumego, determine cutoff points where cumulative success reaches 95% and 50% among ranked bots, then combine with 90% weight on 95% cutoff and 10% on 50% cutoff.
    4.  Sort the tsumegos based on this combined difficulty metric.
    5.  Assign a preliminary, uniformly distributed rank from 0.0 (easiest) to 1.0 (hardest).
    6.  Select an "anchor bot" and find the uniform rank where it solves ~50% of problems.
    7.  Create anchor points mapping this 50% point to a desired real-world rank (e.g., 4d).
    8.  Interpolate the final ratings based on these dynamic anchors.

    Returns:
        A tuple of (difficulty_scores, quality_scores, sorted_ratings, sorted_quality).
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

    # Filter to only include tsumegos with SGF files
    all_tsumegos = {tid for tid in all_tsumegos if Path(os.path.join(PROJECT_ROOT, "tsumegos", f"{tid}.sgf")).exists()}

    # REVISED: Fail loudly if no data was loaded. This prevents the script from
    # producing meaningless results or crashing with obscure errors later on.
    if not all_tsumegos:
        raise ValueError(
            "Parsing completed, but no tsumego problems were found. "
            f"Check the attempts directory ('{ATTEMPTS_DIR}') and file formats."
        )

    print(f"Parsed data for {len(bot_results)} bots and {len(all_tsumegos)} unique tsumegos (with SGF files).")

    # --- 2. Rank bots by success ---
    bot_solves = {bot: sum(results.values()) for bot, results in bot_results.items()}
    ranked_bots = sorted(bot_solves.keys(), key=lambda bot: bot_solves[bot], reverse=False)

    if not ranked_bots:
        raise ValueError("No bots found after parsing results.")

    def find_cutoff(requirement, successes):
        cumulative = 0
        cutoff = 0
        for i, success in enumerate(reversed(successes)):
            cumulative += success
            average = cumulative / (i + 1)
            if average >= requirement:
                cutoff = i
        return len(successes)-(1+cutoff)

    # --- 3. Determine tsumego difficulty based on cutoff points ---
    tsumego_difficulty_scores = {}
    tsumego_quality_scores = {}
    print("\n--- Calculating Tsumego Difficulty ---")
    for tsumego_id in all_tsumegos:
        successes = [bot_results[bot][tsumego_id] for bot in ranked_bots]
        cutoff_95 = find_cutoff(0.95, successes)
        cutoff_55 = find_cutoff(0.55, successes[:cutoff_95])
        rating = (0.9 * cutoff_95 + 0.1 * cutoff_55) / len(successes) # mainly what rank is required to have 95% confidence, with the rest a tiebreaker
        difference = (cutoff_95 - cutoff_55)
        quality = (len(successes) - difference) / (len(successes) + difference) # how quickly do we go from 55% to 95% confidence
        tsumego_difficulty_scores[tsumego_id] = rating
        tsumego_quality_scores[tsumego_id] = quality

    # --- 4. Sort tsumegos by their calculated difficulty ---
    sorted_ratings = sorted(
        tsumego_difficulty_scores.keys(),
        key=lambda tid: tsumego_difficulty_scores[tid],
        reverse=True
    )
    sorted_quality = sorted(
        tsumego_quality_scores.keys(),
        key=lambda tid: tsumego_quality_scores[tid],
        reverse=True
    )
    return tsumego_difficulty_scores, tsumego_quality_scores, sorted_ratings, sorted_quality


if __name__ == "__main__":
    difficulty_scores, quality_scores, sorted_ratings, sorted_quality = get_tsumego_ratings()
    samples = 10
    sorted_by_rating_asc = sorted(difficulty_scores.keys(), key=lambda tid: difficulty_scores[tid])
    for i in range(samples):
        idx = int(len(sorted_by_rating_asc) * (i + 0.5) / samples)
        tsumego_id = sorted_by_rating_asc[idx]
        rating = difficulty_scores[tsumego_id]
        quality = quality_scores[tsumego_id]
        print(f"{tsumego_id} {rating:.5f} {quality:.5f} http://localhost:8000/?id={tsumego_id}")
