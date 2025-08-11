import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
ATTEMPTS_DIR = "attempts"
RATINGS_DIR = "ratings"
TSUMEGO_RATINGS_FILE = "tsumego_ratings_final.csv"

# --- ADD OUTLIER THRESHOLDS ---
# Define a sane range for ratings. Anything outside this is an outlier.
MIN_RATING_THRESHOLD = -100.0
MAX_RATING_THRESHOLD = 100.0


def main():
    """
    Main function to calculate and print the correlation coefficient,
    IGNORING statistical outliers.
    """
    # --- Load Data using Pandas for easier filtering ---
    ratings_path = Path(RATINGS_DIR) / TSUMEGO_RATINGS_FILE
    attempts_path = Path(ATTEMPTS_DIR)

    if not ratings_path.is_file():
        print(f"Error: Ratings file not found at '{ratings_path}'.")
        return

    # Load ratings into a pandas DataFrame
    tsumego_ratings_df = pd.read_csv(ratings_path)
    tsumego_ratings_df = tsumego_ratings_df.rename(columns={"tsumego_id": "id", "rating": "rating"})

    # Count solves
    solve_counts = {}
    for file_path in attempts_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    tsumego_id, result = line.strip().split()
                    if float(result) == 1.0:
                        solve_counts[tsumego_id] = solve_counts.get(tsumego_id, 0) + 1
                except (ValueError, IndexError):
                    continue

    solve_counts_df = pd.DataFrame(list(solve_counts.items()), columns=['id', 'solves'])

    # Combine the two dataframes
    combined_df = pd.merge(tsumego_ratings_df, solve_counts_df, on='id')

    if combined_df.empty:
        print("Error: No common tsumegos found.")
        return

    # --- THIS IS THE CRITICAL STEP ---
    # Filter out the extreme outliers before calculating correlation
    filtered_df = combined_df[
        (combined_df['rating'] > MIN_RATING_THRESHOLD) &
        (combined_df['rating'] < MAX_RATING_THRESHOLD)
        ]

    print(f"Original data points: {len(combined_df)}")
    print(f"Data points after filtering outliers: {len(filtered_df)}")

    if len(filtered_df) < 2:
        print("Not enough data left after filtering to calculate correlation.")
        return

    # Calculate correlation on the CLEANED data
    correlation = filtered_df['solves'].corr(filtered_df['rating'])

    # --------------------------------

    if pd.notna(correlation):
        print(f"\nCorrelation on FILTERED data: {correlation:.8f}")
    else:
        print("\nError: Could not calculate correlation.")


if __name__ == "__main__":
    main()