import torch
import random
from collections import defaultdict
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
import time
import numpy as np

# --- Configuration ---
# Directory containing the raw bot attempt data (e.g., 'bot1.txt', 'bot2.txt')
ATTEMPTS_DIR = "attempts"

# The different values of 'N' (for "Nth strongest bot to fail") to test.
N_VALUES_TO_TEST = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]

# Number of random trials to run for each N value.
NUM_TRIALS = 10

# Seed for the random number generator to ensure the bot splits are reproducible.
RANDOM_SEED = 42

# --- GPU/Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")


def calculate_ratings_gpu(
        bot_indices_subset: torch.Tensor,
        full_results_tensor: torch.Tensor,
        n_value: int,
) -> np.ndarray:
    """
    Calculates tsumego ratings for a subset of bots using GPU-accelerated tensor operations.

    Args:
        bot_indices_subset: A 1D tensor of bot indices for this group.
        full_results_tensor: The complete [bots x tsumegos] boolean tensor on the GPU.
        n_value: The 'N' for the "Nth strongest bot to fail" heuristic.

    Returns:
        A NumPy array of final tsumego ratings (0.0 to 1.0).
    """
    num_bots_in_subset = len(bot_indices_subset)

    # 1. Select the subset of results for this group
    subset_tensor = full_results_tensor[bot_indices_subset]

    # 2. Rank bots within the SUBSET by summing their successes (across tsumego dimension)
    solve_counts = subset_tensor.sum(dim=1, dtype=torch.int32)
    # Get the indices that would sort the bots from strongest to weakest
    strength_ranking = torch.argsort(solve_counts, descending=True)

    # 3. Reorder the subset tensor according to bot strength
    # Now, row 0 is the strongest bot in the subset, row 1 is the second strongest, etc.
    ranked_subset_tensor = subset_tensor[strength_ranking]

    # 4. Calculate difficulty scores for all tsumegos in parallel
    # A failure is 1, a success is 0.
    failures_tensor = 1 - ranked_subset_tensor.to(torch.int8)
    total_failures_per_tsumego = failures_tensor.sum(dim=0)

    # Use cumulative sum to find the rank of the Nth failing bot for each tsumego.
    # cumsum() will show, for each tsumego, how many bots have failed up to that rank.
    # We find the first rank where the cumulative count is >= n_value.
    cumsum_failures = torch.cumsum(failures_tensor, dim=0)
    # argmax on a boolean tensor returns the index of the *first* True value.
    # This is our primary difficulty score (the rank of the Nth failing bot).
    # `clamp(max=1)` prevents argmax from picking the wrong index on all-success columns.
    nth_fail_rank = torch.argmax((cumsum_failures >= n_value).to(torch.int8).clamp(max=1), dim=0)

    # Handle case where fewer than N bots fail: use the weakest failing bot's rank.
    # To find the last fail rank, we flip the tensor, find the first failure, and adjust index.
    last_fail_rank = (num_bots_in_subset - 1) - torch.argmax(
        torch.flip(failures_tensor, dims=[0]), dim=0
    )

    # Choose the appropriate rank based on failure count
    difficulty_score = torch.where(
        total_failures_per_tsumego < n_value, last_fail_rank, nth_fail_rank
    )

    # Handle case where no bots fail: assign a score making it the easiest problem.
    difficulty_score = torch.where(
        total_failures_per_tsumego == 0, num_bots_in_subset, difficulty_score
    )

    # 5. Assign final uniform ranks
    # argsort gives the indices that would sort the array. This is the final ranking.
    # A lower difficulty score (stronger bot failed) means a harder problem.
    # We sort ascending by score to get a mapping from rank -> tsumego index.
    sorted_indices = torch.argsort(difficulty_score)  # Easier problems (high score) will be at the end

    # Create the final ranks tensor (0.0 to 1.0, hardest to easiest)
    num_tsumegos = len(difficulty_score)
    final_ranks = torch.empty_like(difficulty_score, dtype=torch.float32)
    ranks_values = torch.linspace(1.0, 0.0, num_tsumegos, device=device)  # Hardest=1.0, Easiest=0.0
    final_ranks[sorted_indices] = ranks_values

    return final_ranks.cpu().numpy()


def main():
    print("--- Tsumego Rating Stability Test (GPU Accelerated) ---")
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # --- 1. Load data from files (CPU) ---
    print(f"Loading data from '{ATTEMPTS_DIR}'...")
    attempts_path = Path(ATTEMPTS_DIR)
    file_paths = list(attempts_path.glob("*.txt"))

    bot_map = {path.stem: i for i, path in enumerate(file_paths)}
    all_tsumegos = set()
    records = []

    for file_path in tqdm(file_paths, desc="Parsing Bot Files"):
        bot_id = file_path.stem
        bot_idx = bot_map[bot_id]
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    tsumego_id, result = line.strip().split()
                    records.append((bot_idx, tsumego_id, float(result) == 1.0))
                    all_tsumegos.add(tsumego_id)
                except (ValueError, IndexError):
                    continue

    tsumego_list = sorted(list(all_tsumegos))
    tsumego_map = {name: i for i, name in enumerate(tsumego_list)}
    num_bots = len(bot_map)
    num_tsumegos = len(tsumego_map)

    print(f"Loaded data for {num_bots} bots and {num_tsumegos} tsumegos.")

    # --- 2. Create the master tensor and move to GPU ---
    print("Creating master results tensor and moving to GPU...")
    # Initialize with False (default to fail if no data exists)
    results_tensor = torch.zeros((num_bots, num_tsumegos), dtype=torch.bool, device=device)
    for bot_idx, tsumego_id, is_success in tqdm(records, desc="Populating Tensor"):
        tsumego_idx = tsumego_map[tsumego_id]
        if is_success:
            results_tensor[bot_idx, tsumego_idx] = True

    print(f"Tensor created on '{results_tensor.device}'. Starting stability test...")

    # --- 3. Run the stability test ---
    results_summary = []
    all_bot_indices = list(range(num_bots))
    start_time = time.time()

    print(f"\nWill test N values: {N_VALUES_TO_TEST}")
    print(f"Will run {NUM_TRIALS} random trials for each N value.\n")

    for n_value in N_VALUES_TO_TEST:
        correlations = []
        n_start_time = time.time()

        for trial_num in range(NUM_TRIALS):
            # Split bots into two random, non-overlapping groups
            random.shuffle(all_bot_indices)
            midpoint = num_bots // 2
            group_a_indices = torch.tensor(all_bot_indices[:midpoint], device=device)
            group_b_indices = torch.tensor(all_bot_indices[midpoint:], device=device)

            # Calculate ratings for each group in parallel on the GPU
            ratings_a = calculate_ratings_gpu(group_a_indices, results_tensor, n_value)
            ratings_b = calculate_ratings_gpu(group_b_indices, results_tensor, n_value)

            # Spearman correlation is calculated on the CPU
            corr, _ = spearmanr(ratings_a, ratings_b)
            correlations.append(corr)

        avg_correlation = sum(correlations) / len(correlations)
        n_time = time.time() - n_start_time
        results_summary.append({"N": n_value, "Avg Correlation": avg_correlation})

        # --- Intermediate Report ---
        print(f"--- N = {n_value} Complete ({n_time:.1f}s) ---")
        print(f"Average Spearman Correlation: {avg_correlation:.6f}\n")

    total_time = time.time() - start_time
    print(f"--- Test Finished ({total_time:.1f} seconds) ---")
    print("\nFinal Results Summary (higher correlation is more stable):")
    print("---------------------------------")
    print("| N Value | Avg. Correlation  |")
    print("|---------|-------------------|")
    for result in results_summary:
        print(f"|  {result['N']:<6} |    {result['Avg Correlation']:.6f}     |")
    print("---------------------------------\n")
    print("Look for the 'N' value where the correlation score stops increasing significantly.")


if __name__ == "__main__":
    main()