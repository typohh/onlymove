ATTEMPTS_DIR = "attempts"
TSUMEGOS_TO_ANALYZE = ["5180d01a", "78db4e47", "659d7b7f", "10d40a14", "822b297a", "ce1e7413"]
NTH_STRONGEST_FAILER = 5
RATINGS_DIR = "ratings"
RATINGS_FILE = "tsumego_ratings_final.csv"
BOT_RATINGS_FILE = "bot_ratings_final.csv"

from pathlib import Path
import csv


def load_bot_ratings() -> dict:
    """
    Loads bot ratings from the CSV file.
    Returns a dictionary mapping bot names to their ratings.
    """
    ratings_path = Path(RATINGS_DIR) / BOT_RATINGS_FILE
    if not ratings_path.is_file():
        print(f"Warning: Bot ratings file '{ratings_path}' not found.")
        return {}

    ratings = {}
    try:
        with open(ratings_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    bot_name = row[0].strip()
                    try:
                        rating = float(row[1].strip())
                        ratings[bot_name] = rating
                    except ValueError:
                        # Skip rows with invalid ratings
                        continue
        return ratings
    except Exception as e:
        print(f"Warning: Error reading bot ratings file: {e}")
        return {}


def load_and_process_data(directory: str, bot_ratings: dict) -> list | None:
    """
    Loads data from all .txt files, calculates bot strength, and returns a list
    of bots sorted by strength in descending order.
    """
    base_path = Path(directory)
    if not base_path.is_dir():
        print(f"Error: Directory '{directory}' not found. Please check 'ATTEMPTS_DIR'.")
        return None

    files = list(base_path.glob("*.txt"))
    if not files:
        print(f"Error: No .txt files found in '{directory}'.")
        return None

    bots_data = {}
    for file_path in files:
        bot_name = file_path.stem
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
                    print(f"Warning: Skipping malformed line in {file_path.name}: '{line.strip()}'")
                    continue

        if total_count > 0:
            strength = success_count / total_count
            # Get bot rating if available
            bot_rating = bot_ratings.get(bot_name, None)
            bots_data[bot_name] = {
                'strength': strength,
                'failures': failures,
                'rating': bot_rating
            }

    sorted_bots_list = sorted(
        bots_data.items(),
        key=lambda item: item[1]['strength'],
        reverse=True
    )

    return sorted_bots_list


def load_tsumego_ratings() -> dict:
    """
    Loads tsumego ratings from the CSV file.
    Returns a dictionary mapping tsumego IDs to their ratings.
    """
    ratings_path = Path(RATINGS_DIR) / RATINGS_FILE
    if not ratings_path.is_file():
        print(f"Warning: Ratings file '{ratings_path}' not found.")
        return {}

    ratings = {}
    try:
        with open(ratings_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2:
                    tsumego_id = row[0].strip()
                    try:
                        rating = float(row[1].strip())
                        ratings[tsumego_id] = rating
                    except ValueError:
                        # Skip rows with invalid ratings
                        continue
        return ratings
    except Exception as e:
        print(f"Warning: Error reading ratings file: {e}")
        return {}


def analyze_tsumego_globally(tsumego_id: str, sorted_bots: list, ratings: dict):
    """
    Analyzes tsumego performance to find the strongest failing, weakest passing,
    and Nth strongest failing bot.
    """
    print(f"--- Tsumego Global Analysis for ID: {tsumego_id} ---\n")

    # Display tsumego rating if available
    if tsumego_id in ratings:
        print(f"  -> Tsumego Rating: {ratings[tsumego_id]:.2f}")
    else:
        print("  -> Tsumego Rating: Not available")
    print()  # Add a blank line for better formatting

    total_bots = len(sorted_bots)
    strongest_failing_bot_details = None
    weakest_passing_bot_details = None
    nth_strongest_failing_bot_details = None
    failures_found_count = 0

    # --- Find Key Bots ---

    # Find the strongest failing bot (first failure in the sorted list)
    for i, (bot_name, data) in enumerate(sorted_bots):
        if tsumego_id in data['failures']:
            # This bot failed the tsumego
            # Since we iterate from strongest to weakest, the first failure we find
            # is the strongest failing bot.
            if strongest_failing_bot_details is None:
                strongest_failing_bot_details = {
                    "name": bot_name,
                    "strength": data['strength'],
                    "rank": i + 1,
                    "rating": data.get('rating', None)
                }
            break

    # Find the weakest bot that passed (by iterating from the weakest)
    for i, (bot_name, data) in enumerate(reversed(sorted_bots)):
        if tsumego_id not in data['failures']:
            original_index = total_bots - 1 - i
            weakest_passing_bot_details = {
                "name": bot_name,
                "strength": data['strength'],
                "rank": original_index + 1,
                "rating": data.get('rating', None)
            }
            break

    # Find the Nth strongest failing bot
    for i, (bot_name, data) in enumerate(sorted_bots):
        if tsumego_id in data['failures']:
            # This bot failed the tsumego
            failures_found_count += 1
            if failures_found_count == NTH_STRONGEST_FAILER:
                nth_strongest_failing_bot_details = {
                    "name": bot_name,
                    "strength": data['strength'],
                    "rank": i + 1,
                    "rating": data.get('rating', None)
                }
                break

    # --- Report Findings ---
    if strongest_failing_bot_details:
        s = strongest_failing_bot_details
        rating_info = f", Rating: {s['rating']:.2f}" if s['rating'] is not None else ""
        print(f"  -> Strongest bot that FAILS: '{s['name']}' "
              f"(Strength: {s['strength']:.2%}, Rank: {s['rank']}/{total_bots}{rating_info})")
    else:
        print("  -> All bots SOLVED the tsumego.")

    if weakest_passing_bot_details:
        w = weakest_passing_bot_details
        rating_info = f", Rating: {w['rating']:.2f}" if w['rating'] is not None else ""
        print(f"  -> Weakest bot that SOLVES:  '{w['name']}' "
              f"(Strength: {w['strength']:.2%}, Rank: {w['rank']}/{total_bots}{rating_info})")
    else:
        print("  -> All bots FAILED the tsumego.")

    # Report on the Nth strongest failing bot
    if nth_strongest_failing_bot_details:
        n = nth_strongest_failing_bot_details
        rating_info = f", Rating: {n['rating']:.2f}" if n['rating'] is not None else ""
        print(f"  -> {NTH_STRONGEST_FAILER}th strongest bot that FAILS: '{n['name']}' "
              f"(Strength: {n['strength']:.2%}, Rank: {n['rank']}/{total_bots}{rating_info})")
    elif failures_found_count > 0:
        print(f"  -> Fewer than {NTH_STRONGEST_FAILER} bots FAILED the tsumego (only {failures_found_count} failed it).")

    # --- Estimate 50% crossover strength ---
    if strongest_failing_bot_details and weakest_passing_bot_details:
        s_fail = strongest_failing_bot_details['strength']
        s_pass = weakest_passing_bot_details['strength']
        midpoint_strength = (s_fail + s_pass) / 2
        print(f"\n  => Approx. 50% failure rate is at strength: {midpoint_strength:.2%}")
    else:
        print(f"\n  => Cannot estimate 50% failure rate.")


def main():
    """Main execution function."""
    # Load bot ratings
    bot_ratings = load_bot_ratings()
    
    # Load and process bot data with ratings
    sorted_bots_list = load_and_process_data(ATTEMPTS_DIR, bot_ratings)

    if not sorted_bots_list:
        print("Processing aborted due to data loading errors.")
        return

    # Load tsumego ratings
    ratings = load_tsumego_ratings()

    # Perform the global analysis for each tsumego ID in the list
    for i, tsumego_id in enumerate(TSUMEGOS_TO_ANALYZE):
        if i > 0:
            # Print a separator for readability between reports
            print("\n" + "=" * 60 + "\n")
        analyze_tsumego_globally(tsumego_id, sorted_bots_list, ratings)


if __name__ == "__main__":
    main()