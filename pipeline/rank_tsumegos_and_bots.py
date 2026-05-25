import glob
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import random
import re
import scipy.optimize as opt
from datetime import datetime

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import BOARD_SIZE

# --- Configuration ---
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "attempts", str(BOARD_SIZE))
TSUMEGOS_DIRECTORY = os.path.join(PROJECT_ROOT, "tsumegos", str(BOARD_SIZE))
RATINGS_DIRECTORY = os.path.join(PROJECT_ROOT, "ratings", str(BOARD_SIZE))
BATCH_SIZE = 8192
LEARNING_RATE = 0.01
LOSS_DROP_THRESHOLD = 0.001  # Stop training if loss improves by less than 0.1%
MIN_BATCHES = 50  # Minimum number of batches before checking loss drop
PROB_TARGET = 0.95  # Target probability for tsumego rating calculation
NUM_EXPERTS = 3  # Number of sigmoid experts in the mixture model
# This constant represents 1 standard deviation of the latent ability in a standard logistic distribution.
DIFFERENCE_FOR_QUALITY = -np.pi / np.sqrt(3)  # Approx. 1.8138

# --- Global Epsilon for numerical stability ---
EPSILON = 1e-9

# Set a fixed seed for reproducibility
torch.manual_seed(43)
np.random.seed(43)


# --- Data Loading ---
def load_data(data_dir, include_users=True):
    """Loads and preprocesses tsumego attempt data from text files and database.sql.

    With include_users=False the database.sql user attempts are skipped entirely,
    so the IRF is fit on bot (+ humanv0 model) data only — used to evaluate how
    well bot-derived difficulty predicts real users out-of-sample.
    """
    print("Loading and preprocessing data...")

    # --- Temporary storage for raw data ---
    all_participant_ids = set()
    all_tsumego_ids = set()
    records_to_process = []
    attempt_counts = {}

    # --- 1. Load Bot Data ---
    bot_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not bot_files:
        print(f"Warning: No bot data files (.txt) found in '{data_dir}'")
    else:
        for bot_file in tqdm(bot_files, desc="Loading Bot Files"):
            bot_name = os.path.splitext(os.path.basename(bot_file))[0]
            all_participant_ids.add(bot_name)
            lines = []
            with open(bot_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        tsumego_name, success_rate = parts
                        all_tsumego_ids.add(tsumego_name)
                        records_to_process.append((bot_name, tsumego_name, float(success_rate)))
                        lines.append(line)
            attempt_counts[bot_name] = len(lines)

    # --- 2. Load User Data from database.sql ---
    # database.sql is the global D1 dump at attempts/database.sql (cross-size).
    # The schema has (id, user_id, problem, success, created_at, size, move,
    # time_used, review_time, is_repeat). Only user_id/problem/success/created_at
    # and size are needed here; per-attempt move/time fields are ignored.
    # Repeat attempts (is_repeat=1) are skipped: the user has already seen the
    # problem, so they are easier and would inflate the skill estimate.
    # Newer `wrangler d1 export` dumps prefix each INSERT with the column list
    # (`("id",...) VALUES(...)`); the regex tolerates that and legacy rows where
    # size and the per-attempt columns are absent (treat missing size as 9).
    sql_file = os.path.join(PROJECT_ROOT, 'attempts', 'database.sql')
    user_attempts = []
    if not include_users:
        print("Skipping user data load: include_users=False (bot-only fit).")
    elif not os.path.exists(sql_file):
        print(f"Skipping user data load: {sql_file} not found.")
    else:
        print(f"Loading user data from {sql_file}...")
        insert_pattern = re.compile(
            r"INSERT INTO [\"`]?tsumego_reports[\"`]?"
            r"(?:\s*\([^)]*\))?"                          # optional column list (newer D1 dumps)
            r"\s*VALUES\s*\("
            r"\d+,'([^']+)','([^']+)',([01]),'([^']+)'"  # id, user_id, problem, success, created_at
            r"(?:,(\d+|NULL))?"                          # optional size
            r"(.*?)"                                     # optional move/time_used/review_time/is_repeat tail
            r"\);"
        )

        with open(sql_file, 'r') as f:
            for line in tqdm(f, desc="Parsing SQL"):
                match = insert_pattern.match(line)
                if match:
                    user_id, tsumego_id, success, timestamp_str, size_str, tail = match.groups()

                    # Drop repeat attempts. When the per-attempt columns are
                    # present, is_repeat is the last value (after size: move,
                    # time_used, review_time, is_repeat).
                    tail_fields = [f for f in tail.split(',') if f != '']
                    if len(tail_fields) >= 4 and tail_fields[-1] == '1':
                        continue

                    row_size = int(size_str) if size_str not in (None, 'NULL') else 9
                    if row_size != BOARD_SIZE:
                        continue

                    # Parse timestamp
                    # Format example: '2025-08-20T23:04:55.761Z'
                    # We only need year and month.
                    try:
                        # Taking the first 7 chars is risky if format changes, but safe for ISO-8601 YYYY-MM
                         # More robust parsing:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                        profiles_to_add = []

                        # 1. Current month profile
                        current_profile = f"{user_id[:8]}_{dt.year}-{dt.month:02d}"
                        profiles_to_add.append(current_profile)

                        # 2. Previous month profile (for overlap)
                        prev_month = dt.month - 1
                        prev_year = dt.year
                        if prev_month == 0:
                            prev_month = 12
                            prev_year -= 1
                        prev_profile = f"{user_id[:8]}_{prev_year}-{prev_month:02d}"
                        profiles_to_add.append(prev_profile)

                        for profile_id in profiles_to_add:
                            user_attempts.append({'user_id': profile_id, 'tsumego_id': tsumego_id, 'success': int(success)})

                    except ValueError as e:
                        print(f"Skipping malformed timestamp: {timestamp_str} ({e})")
                        continue

    # Capture the raw attempts before aggregation
    raw_user_attempts_df = pd.DataFrame(user_attempts)
    # user_id is already formatted as {id[:8]}_{YYYY-MM}, so no further truncation needed.

    if user_attempts:
        user_df = pd.DataFrame(user_attempts)
        # user_df['user_id'] = user_df['user_id'].str[:8]  # REMOVED: Truncation handled during profile generation

        # There is no UNIQUE(user_id, problem) constraint, but we drop repeat
        # attempts (is_repeat=1) above, so each (user, problem) keeps only its
        # first attempt. We just rename 'success' to 'success_rate' for
        # consistency with the bot data format.
        user_df.rename(columns={'success': 'success_rate'}, inplace=True)
        user_success_rates = user_df

        # Add all user_ids to participants, even if they have no valid attempts
        unique_users = user_success_rates['user_id'].unique()
        for user_id in unique_users:
            all_participant_ids.add(user_id)

        # Calculate attempt counts for users (number of valid tsumegos attempted)
        user_attempt_counts = user_success_rates.groupby('user_id').size().to_dict()
        for user_id, count in user_attempt_counts.items():
            attempt_counts[user_id] = count

        # Add user data to processing lists
        for _, row in tqdm(user_success_rates.iterrows(), total=len(user_success_rates), desc="Processing Users"):
            all_tsumego_ids.add(row['tsumego_id'])
            records_to_process.append((row['user_id'], row['tsumego_id'], row['success_rate']))

    # --- 3. Create Mappings and Final DataFrame ---
    if not records_to_process:
        raise ValueError("No records found from bots or users. Cannot proceed.")

    participant_ids = sorted(list(all_participant_ids))
    participant_map = {name: i for i, name in enumerate(participant_ids)}

    tsumego_ids_unfiltered = sorted(list(all_tsumego_ids))
    # Filter to only include tsumegos with SGF files
    tsumego_ids = [tid for tid in tsumego_ids_unfiltered if os.path.exists(os.path.join(TSUMEGOS_DIRECTORY, f"{tid}.sgf"))]
    tsumego_map = {name: i for i, name in enumerate(tsumego_ids)}

    tqdm.write(f"Discovered {len(participant_map)} participants (bots/users) and {len(tsumego_map)} valid tsumegos.")

    records = []
    for participant_name, tsumego_name, success_rate in tqdm(records_to_process, desc="Finalizing Records"):
        if tsumego_name in tsumego_map:
            participant_idx = participant_map[participant_name]
            tsumego_idx = tsumego_map[tsumego_name]
            records.append((participant_idx, tsumego_idx, success_rate))

    df = pd.DataFrame(records, columns=["bot_id", "tsumego_id", "success_rate"])
    tqdm.write(f"Loaded {len(df)} total solve attempts.")

    return df, participant_map, tsumego_map, tsumego_ids, attempt_counts, raw_user_attempts_df


# --- Preliminary Bot Ranking ---
def get_preliminary_bot_ranking(df, bot_map):
    """
    Assigns a preliminary rank to each bot based on their overall average success rate.
    Returns a DataFrame with bot_id and preliminary_rank (lower rank = better bot).
    """
    print("\nCalculating preliminary bot rankings...")
    bot_avg_success = df.groupby('bot_id')['success_rate'].mean().reset_index()
    bot_avg_success = bot_avg_success.sort_values(by='success_rate', ascending=False).reset_index(drop=True)
    bot_avg_success['preliminary_rank'] = bot_avg_success.index + 1  # Rank 1 is best bot

    # Map bot_id (int index) to its rank
    bot_rank_map = bot_avg_success.set_index('bot_id')['preliminary_rank'].to_dict()
    print("Preliminary bot rankings calculated.")
    return bot_rank_map


# --- Identify Non-Monotonic Tsumegos ---
def identify_non_monotonic_tsumegos(df, tsumego_map, bot_rank_map):
    """
    Identifies tsumegos that exhibit a negative slope with bot preliminary rankings.
    This function is vectorized for performance.
    Returns a set of non-monotonic tsumego IDs.
    """
    print("\nIdentifying non-monotonic tsumegos...")

    # Create a DataFrame with bot preliminary ranks
    df_ranked_bots = df.copy()
    df_ranked_bots['bot_prelim_rank'] = df_ranked_bots['bot_id'].map(bot_rank_map)

    # Convert tsumego_id (int) back to tsumego_name
    rev_tsumego_map = {i: name for name, i in tsumego_map.items()}
    df_ranked_bots['tsumego_name'] = df_ranked_bots['tsumego_id'].map(rev_tsumego_map)

    # Define skill groups thresholds
    num_bots_total = len(bot_rank_map)
    strong_bots_threshold = num_bots_total // 3
    weak_bots_threshold = num_bots_total * 2 // 3

    # Calculate average success rates for each tsumego for strong and weak bots
    strong_bot_avg = df_ranked_bots[df_ranked_bots['bot_prelim_rank'] <= strong_bots_threshold].groupby('tsumego_name')['success_rate'].mean().rename(
        'strong_bot_success')
    weak_bot_avg = df_ranked_bots[df_ranked_bots['bot_prelim_rank'] > weak_bots_threshold].groupby('tsumego_name')['success_rate'].mean().rename(
        'weak_bot_success')

    # Combine the average success rates into a single DataFrame
    tsumego_stats = pd.concat([strong_bot_avg, weak_bot_avg], axis=1)

    # Drop rows where a tsumego was not attempted by either strong or weak bots
    tsumego_stats.dropna(inplace=True)

    # Identify non-monotonic tsumegos
    non_monotonic_mask = tsumego_stats['strong_bot_success'] < tsumego_stats['weak_bot_success']
    non_monotonic_tsumego_ids = set(tsumego_stats[non_monotonic_mask].index)

    print(f"Identified {len(non_monotonic_tsumego_ids)} non-monotonic tsumegos.")
    return non_monotonic_tsumego_ids


# --- Rating Model (Mixture of Sigmoids) ---
class RatingModel(nn.Module):
    """A Mixture of Experts model where each expert is a 2-parameter logistic function."""

    def __init__(self, num_bots, num_tsumegos, num_experts):
        super().__init__()
        self.num_bots = num_bots
        self.num_tsumegos = num_tsumegos
        self.num_experts = num_experts

        # Bot Parameters
        self.bot_ratings = nn.Embedding(num_bots, 1)  # θ_b (skill)

        # Tsumego Parameters for each expert
        self.tsumego_difficulties_raw = nn.Embedding(num_tsumegos, num_experts)  # β_t,k
        self.tsumego_discrimination_raw = nn.Embedding(num_tsumegos, num_experts)  # α_t,k
        self.tsumego_weights_raw = nn.Embedding(num_tsumegos, num_experts)  # w_t,k (logits)

        # Initialize parameters
        nn.init.normal_(self.bot_ratings.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_difficulties_raw.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_discrimination_raw.weight, mean=0.0, std=0.5)
        nn.init.constant_(self.tsumego_weights_raw.weight, 0.0)  # Equal weights initially

    def forward(self, bot_indices, tsumego_indices):
        """Calculates the predicted success rate."""
        # Shape: (batch_size, 1)
        theta = self.bot_ratings(bot_indices)

        # Get raw parameters for each expert. Shape: (batch_size, num_experts)
        beta = self.tsumego_difficulties_raw(tsumego_indices)
        alpha_raw = self.tsumego_discrimination_raw(tsumego_indices)
        weights_raw = self.tsumego_weights_raw(tsumego_indices)

        # Transform raw parameters to be valid
        # Alpha (discrimination) must be positive
        alpha = func.softplus(alpha_raw) + EPSILON
        # Weights must be positive and sum to 1
        weights = func.softmax(weights_raw, dim=1)

        # Calculate expert outputs
        # theta is (batch_size, 1), beta is (batch_size, num_experts)
        # Broadcasting makes (theta - beta) shape (batch_size, num_experts)
        diff = alpha * (theta - beta)
        expert_probs = torch.sigmoid(diff)

        # Combine experts using learned weights
        # (batch_size, num_experts) * (batch_size, num_experts) -> sum -> (batch_size)
        final_prob = torch.sum(weights * expert_probs, dim=1)

        return final_prob


# --- Quality Factor Optimization ---
def find_optimal_quality_adjustment(raw_user_attempts, user_ratings_df, tsumego_ratings_df, tsumego_quality_df):
    """
    Finds the optimal quality adjustment factor by driving the correlation between
    prediction error and problem quality to zero for human players.
    """
    print("\nOptimizing quality adjustment factor for human players...")

    # 1. Prepare the data by merging everything into one DataFrame.
    # This will crash if there are no human players or data is misaligned.
    human_ids = raw_user_attempts['user_id'].unique()
    human_ratings_df = user_ratings_df[user_ratings_df['bot_id'].isin(human_ids)]

    df = pd.merge(raw_user_attempts, human_ratings_df, left_on='user_id', right_on='bot_id')
    df = pd.merge(df, tsumego_ratings_df, on='tsumego_id')
    df = pd.merge(df, tsumego_quality_df, on='tsumego_id')
    df.rename(columns={'rating_x': 'user_rating', 'rating_y': 'tsumego_rating', 'success': 'is_correct'}, inplace=True)

    # 2. Define the objective function for the optimizer.
    # The goal is to find a q_factor that makes problem quality uncorrelated
    # with the success/failure prediction error.
    def get_quality_error_correlation(q_factor, data):
        # Calculate adjusted rating and the skill delta for each attempt
        data['adjusted_rating'] = data['tsumego_rating'] - q_factor * (1 - data['quality'])
        data['delta'] = data['user_rating'] - data['adjusted_rating']

        # Model the probability of success using a standard logistic function of the delta.
        predicted_prob = 1.0 / (1.0 + np.exp(-data['delta']))

        # The error (residual) is the difference between the actual outcome and the prediction.
        residuals = data['is_correct'] - predicted_prob

        # Calculate the Pearson correlation between problem quality and the prediction error.
        correlation = data['quality'].corr(residuals)

        # If correlation is NaN (e.g., if quality or residuals have no variance),
        # the script will fail later, but we can return a large value for the optimizer.
        if pd.isna(correlation):
            return 1e9

        # The loss is how far the correlation is from zero. We want to minimize this.
        return abs(correlation)

    # 3. Run the optimization to find the q_factor that minimizes the absolute correlation.
    result = opt.minimize_scalar(
        fun=get_quality_error_correlation,
        args=(df,),
        bounds=(0, 10),  # Search for a factor between 0 and 10
        method='bounded'
    )

    optimal_q_factor = result.x
    final_corr = get_quality_error_correlation(optimal_q_factor, df)
    print(f"Optimization complete. Optimal Q_factor found: {optimal_q_factor:.4f} (Final absolute correlation: {final_corr:.6f})")

    return optimal_q_factor


# --- Calculate Tsumego Ratings Numerically ---
def wrapper_calculate_tsumego_ratings(all_tsumego_names, attempt_counts, q_factor=1.5, produce_unadjusted_for_optim=False):
    """
    Recalculates tsumego ratings and quality from the saved parameters CSV file
    using numerical methods. This function will fail if required files are not found.
    """
    if produce_unadjusted_for_optim:
        print("\nCalculating initial UNADJUSTED tsumego ratings and quality...")
    else:
        print("\nCalculating FINAL tsumego ratings and quality...")

    # This will now raise FileNotFoundError if files are missing, as requested.
    params_df = pd.read_csv(os.path.join(RATINGS_DIRECTORY, 'tsumego_parameters_final.csv'))
    bots_df = pd.read_csv(os.path.join(RATINGS_DIRECTORY, 'bot_ratings_final.csv'))

    # Use the empirical range of bot ratings to define a safe search space for theta
    bot_rating_min = bots_df['rating'].min()
    bot_rating_max = bots_df['rating'].max()
    bot_rating_range = bot_rating_max - bot_rating_min

    # Define the search range for theta, extended to be safe
    theta_search_min = bot_rating_min - bot_rating_range
    theta_search_max = bot_rating_max + bot_rating_range

    def predict_prob_for_tsumego(theta, params):
        """Calculates success probability for a single tsumego at a given theta."""
        diffs = params['discriminations'] * (theta - params['difficulties'])
        expert_probs = 1 / (1 + np.exp(-diffs))
        return np.sum(params['weights'] * expert_probs)

    def find_rating_for_tsumego(params, target_prob):
        """Performs a binary search to find the theta for a target probability."""
        low, high = theta_search_min, theta_search_max
        for _ in range(20):  # 20 iterations is more than enough for high precision
            mid = (low + high) / 2
            prob = predict_prob_for_tsumego(mid, params)
            if prob < target_prob:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    final_results = []
    desc = "Calculating UNADJUSTED Ratings" if produce_unadjusted_for_optim else "Calculating FINAL Ratings"
    for _, row in tqdm(params_df.iterrows(), total=len(params_df), desc=desc):
        tsumego_id = row['tsumego_id']

        # Extract parameters for the current tsumego
        params = {
            'difficulties': np.array([row[f'difficulty_{i + 1}'] for i in range(NUM_EXPERTS)]),
            'discriminations': np.array([row[f'discrimination_{i + 1}'] for i in range(NUM_EXPERTS)]),
            'weights': np.array([row[f'weight_{i + 1}'] for i in range(NUM_EXPERTS)])
        }

        # 1. Calculate unadjusted Rating
        rating = find_rating_for_tsumego(params, PROB_TARGET)

        # 2. Calculate Quality
        theta_for_quality = rating + DIFFERENCE_FOR_QUALITY
        prob_at_quality_theta = predict_prob_for_tsumego(theta_for_quality, params)
        if DIFFERENCE_FOR_QUALITY > 0:
            quality = (prob_at_quality_theta - PROB_TARGET) / (1 - PROB_TARGET)
        else:
            quality = (PROB_TARGET - prob_at_quality_theta) / PROB_TARGET

        # 3. Apply adjustment if this is the final run
        if produce_unadjusted_for_optim:
            adjusted_rating = rating
        else:
            adjusted_rating = rating - q_factor * (1 - quality)

        final_results.append({
            "tsumego_id": tsumego_id,
            "rating": adjusted_rating,
            "quality": quality
        })

    # --- Save files ---
    # Choose filename based on run type
    if produce_unadjusted_for_optim:
        rating_filename = os.path.join(RATINGS_DIRECTORY, 'tsumego_ratings_unadjusted.csv')
    else:
        rating_filename = os.path.join(RATINGS_DIRECTORY, 'tsumego_ratings_final.csv')

    # Save ratings
    tsumego_rating_df = pd.DataFrame(final_results)[['tsumego_id', 'rating']].sort_values(by="rating", ascending=False)
    tsumego_rating_df.to_csv(rating_filename, index=False)
    print(f"\nRatings saved to '{rating_filename}'")

    # Save quality and user profiles on both runs, as they don't change
    tsumego_quality_df = pd.DataFrame(final_results)[['tsumego_id', 'quality']].sort_values(by="quality", ascending=False)
    tsumego_quality_df.to_csv(os.path.join(RATINGS_DIRECTORY, 'tsumego_quality_final.csv'), index=False)
    print(f"Quality saved to '{os.path.join(RATINGS_DIRECTORY, 'tsumego_quality_final.csv')}'")

    # Create user_profiles_final.csv only on the final run for clarity
    if not produce_unadjusted_for_optim:
        user_profiles_df = bots_df.copy()
        user_profiles_df['attempts'] = user_profiles_df['bot_id'].map(attempt_counts).fillna(0).astype(int)
        user_profiles_df.to_csv(os.path.join(RATINGS_DIRECTORY, 'user_profiles_final.csv'), index=False)
        print(f"User profiles with attempts saved to '{os.path.join(RATINGS_DIRECTORY, 'user_profiles_final.csv')}'")


# --- Save ratings snapshot ---
def save_ratings_snapshot(model, bot_map, tsumego_map, all_tsumego_names,
                          monotonic_tsumego_map):
    """
    Calculates and saves the bot and tsumego parameters to CSV files.
    """
    model.eval()
    bot_filename = os.path.join(RATINGS_DIRECTORY, "bot_ratings_final.csv")
    tsumego_parameters_filename = os.path.join(RATINGS_DIRECTORY, "tsumego_parameters_final.csv")

    os.makedirs(RATINGS_DIRECTORY, exist_ok=True)

    with torch.no_grad():
        bot_ratings_np = model.bot_ratings.weight.cpu().numpy().flatten()

        # Get final parameters for monotonic tsumegos
        monotonic_difficulties = model.tsumego_difficulties_raw.weight.cpu().numpy()
        monotonic_discrimination_raw = model.tsumego_discrimination_raw.weight.cpu()
        monotonic_weights_raw = model.tsumego_weights_raw.weight.cpu()

        # Apply transformations
        monotonic_discrimination = (func.softplus(monotonic_discrimination_raw) + EPSILON).numpy()
        monotonic_weights = func.softmax(monotonic_weights_raw, dim=1).numpy()

    rev_bot_map = {i: name for name, i in bot_map.items()}
    rev_monotonic_tsumego_map = {i: name for name, i in monotonic_tsumego_map.items()}

    # --- Combine all tsumego parameters ---
    param_records = []
    for i in range(len(monotonic_difficulties)):
        record = {"tsumego_id": rev_monotonic_tsumego_map[i]}
        for k in range(model.num_experts):
            record[f'difficulty_{k + 1}'] = monotonic_difficulties[i, k]
            record[f'discrimination_{k + 1}'] = monotonic_discrimination[i, k]
            record[f'weight_{k + 1}'] = monotonic_weights[i, k]
        param_records.append(record)

    # All tsumegos in the trained model are monotonic, so this list is complete
    tsumego_parameters_df = pd.DataFrame(param_records).sort_values(by="tsumego_id")

    # Save bot ratings
    bot_df = pd.DataFrame({
        "bot_id": [rev_bot_map[i] for i in range(len(bot_ratings_np))],
        "rating": bot_ratings_np
    }).sort_values(by="rating", ascending=False)
    bot_df.to_csv(bot_filename, index=False)

    # Save tsumego parameters
    tsumego_parameters_df.to_csv(tsumego_parameters_filename, index=False)

    print(f"\nBot ratings saved to '{bot_filename}'")
    print(f"Tsumego parameters saved to '{tsumego_parameters_filename}'")


# --- Training ---
def train_model(model, data_loader, device):
    """
    Trains the model, stopping when loss improvement is less than a threshold.
    """
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-8)

    epoch = 0
    previous_loss = float('inf')
    batches_since_check = 0
    min_batches_before_check = len(data_loader)  # Check once per epoch

    print("\nStarting model training...")
    while True:
        model.train()
        total_loss = 0
        count = 0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}") as progress_bar:
            for bot_batch, tsumego_batch, rate_batch in progress_bar:
                bot_batch = bot_batch.to(device)
                tsumego_batch = tsumego_batch.to(device)
                rate_batch = rate_batch.to(device).float()

                optimizer.zero_grad()
                predicted_rates = model(bot_batch, tsumego_batch)

                # Clamp predictions to avoid log(0) or log(1) with BCELoss
                predicted_rates = torch.clamp(predicted_rates, EPSILON, 1 - EPSILON)

                loss = loss_fn(predicted_rates, rate_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
                progress_bar.set_postfix(loss=f"{(total_loss / count):.4f}")
                batches_since_check += 1

        avg_loss = total_loss / count

        # Check for early stopping
        if batches_since_check >= min_batches_before_check:
            relative_loss_drop = (previous_loss - avg_loss) / (previous_loss + EPSILON)
            print(f"\nEpoch {epoch + 1} complete. Average Loss: {avg_loss:.6f}. Relative Loss Drop: {relative_loss_drop:.4%}")
            if relative_loss_drop < LOSS_DROP_THRESHOLD and epoch > 0:
                print(f"Loss improvement ({relative_loss_drop:.4%}) is below threshold ({LOSS_DROP_THRESHOLD:.2%}). Stopping training.")
                break
            previous_loss = avg_loss
            batches_since_check = 0
        epoch += 1
        if epoch > 100:  # Max epochs safeguard
            print("Reached maximum epochs (100). Stopping training.")
            break

    print("\nTraining complete.")


def rank_tsumegos_and_bots(df_full, bot_map, tsumego_map, all_tsumego_names):
    rev_tsumego_map = {i: name for name, i in tsumego_map.items()}

    # --- Preliminary Bot Ranking ---
    bot_prelim_rank_map = get_preliminary_bot_ranking(df_full, bot_map)

    # --- Identify and Discard Non-Monotonic Tsumegos ---
    non_monotonic_tsumego_names = identify_non_monotonic_tsumegos(df_full, tsumego_map, bot_prelim_rank_map)

    monotonic_tsumego_names = sorted([name for name in all_tsumego_names if name not in non_monotonic_tsumego_names])
    monotonic_tsumego_map = {name: i for i, name in enumerate(monotonic_tsumego_names)}

    # Filter DataFrame to include only monotonic tsumegos
    df_monotonic = df_full[df_full['tsumego_id'].map(lambda x: rev_tsumego_map[x] in monotonic_tsumego_map)].copy()

    # Re-map tsumego_ids in df_monotonic to their new compact indices
    df_monotonic['tsumego_id_monotonic'] = df_monotonic['tsumego_id'].map(lambda x: monotonic_tsumego_map[rev_tsumego_map[x]])

    print(f"\nTraining on {len(monotonic_tsumego_map)} monotonic tsumegos.")
    print(f"Removed {len(non_monotonic_tsumego_names)} non-monotonic tsumegos from training set.")

    if len(df_monotonic) == 0:
        print("No monotonic data left to train on. Exiting.")
        return

    # Create dataset and DataLoader for monotonic tsumegos
    dataset_monotonic = TensorDataset(
        torch.LongTensor(df_monotonic.bot_id.values),
        torch.LongTensor(df_monotonic.tsumego_id_monotonic.values),
        torch.Tensor(df_monotonic.success_rate.values)
    )
    # Determine num_workers. This will raise a TypeError if os.cpu_count() returns None.
    num_workers = min(os.cpu_count(), 4)
    data_loader_monotonic = DataLoader(dataset_monotonic, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

    # --- Model Initialization ---
    model = RatingModel(
        num_bots=len(bot_map),
        num_tsumegos=len(monotonic_tsumego_map),
        num_experts=NUM_EXPERTS
    )

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on: {device}")
    model.to(device)

    # --- Training ---
    train_model(
        model,
        data_loader_monotonic,
        device
    )

    # --- Final Parameter Save ---
    save_ratings_snapshot(
        model,
        bot_map,
        tsumego_map,
        monotonic_tsumego_names,
        monotonic_tsumego_map
    )


# --- Main Execution ---
if __name__ == '__main__':
    # "cd pwa && npx wrangler d1 export DB --remote --output=../attempts/database.sql"

    import argparse
    parser = argparse.ArgumentParser(description="Fit the mixture-of-sigmoids rating model.")
    parser.add_argument("--no-users", action="store_true",
                        help="Fit on bot data only (skip database.sql user attempts and "
                             "the q-factor optimization). Use to evaluate bot-derived "
                             "difficulty against real users out-of-sample.")
    cli_args = parser.parse_args()

    # Step 1: Load all data from files.
    # Will raise ValueError if no data is found.
    df_full, bot_map, tsumego_map, all_tsumego_names_with_sgf, attempt_counts, raw_user_attempts = load_data(
        DATA_DIRECTORY, include_users=not cli_args.no_users)

    # Step 2: Train the model and save the learned parameters.
    rank_tsumegos_and_bots(df_full, bot_map, tsumego_map, all_tsumego_names_with_sgf)

    # Step 3: Generate UNADJUSTED ratings and quality from the saved parameters.
    # This will raise FileNotFoundError if the previous step failed to produce output.
    wrapper_calculate_tsumego_ratings(all_tsumego_names_with_sgf, attempt_counts, produce_unadjusted_for_optim=True)

    # Step 4: Find the optimal quality adjustment factor.
    # Requires real user attempts to fit the q_factor; skip when there's no user data
    # (e.g. new board sizes that have no user attempts yet) and fall back to default.
    if raw_user_attempts.empty:
        print("\nSkipping quality-adjustment optimization: no user attempts available.")
        optimal_factor = 1.5  # default in wrapper_calculate_tsumego_ratings
    else:
        user_ratings = pd.read_csv(os.path.join(RATINGS_DIRECTORY, 'bot_ratings_final.csv'))
        tsumego_ratings_unadjusted = pd.read_csv(os.path.join(RATINGS_DIRECTORY, 'tsumego_ratings_unadjusted.csv'))
        tsumego_quality = pd.read_csv(os.path.join(RATINGS_DIRECTORY, 'tsumego_quality_final.csv'))

        optimal_factor = find_optimal_quality_adjustment(
            raw_user_attempts,
            user_ratings,
            tsumego_ratings_unadjusted,
            tsumego_quality
        )

    # Step 5: Recalculate the FINAL ratings using the optimal factor.
    print(f"\n--- Recalculating final ratings with Q_factor: {optimal_factor:.4f} ---")
    wrapper_calculate_tsumego_ratings(
        all_tsumego_names_with_sgf,
        attempt_counts,
        q_factor=optimal_factor
    )

    print("\n\nProcess finished successfully.")

    # 2 sigmoids : 0.317173
    # 3 sigmoids : 0.317043