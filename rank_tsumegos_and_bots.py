import glob
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import random  # Keep for potential future sampling/testing
import re

# --- Configuration ---
DATA_DIRECTORY = "attempts"
BATCH_SIZE = 8192
LEARNING_RATE = 0.01
LOSS_DROP_THRESHOLD = 0.001  # Stop training if loss improves by less than 0.1%
MIN_BATCHES = 50  # Minimum number of batches before checking loss drop
PROB_TARGET = 0.95  # Target probability for tsumego rating calculation
NUM_EXPERTS = 2  # Number of sigmoid experts in the mixture model
# This constant represents 1 standard deviation of the latent ability in a standard logistic distribution.
DIFFERENCE_FOR_QUALITY = -np.pi / np.sqrt(3)  # Approx. 1.8138

# --- Global Epsilon for numerical stability ---
EPSILON = 1e-9

# Set a fixed seed for reproducibility
torch.manual_seed(43)
np.random.seed(43)


# --- Data Loading ---
def load_data(data_dir):
    """Loads and preprocesses tsumego attempt data from text files and database.sql."""
    print("Loading and preprocessing data...")

    # --- Temporary storage for raw data ---
    all_participant_ids = set()
    all_tsumego_ids = set()
    records_to_process = []

    # --- 1. Load Bot Data ---
    bot_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not bot_files:
        print(f"Warning: No bot data files (.txt) found in '{data_dir}'")
    else:
        for bot_file in tqdm(bot_files, desc="Loading Bot Files"):
            bot_name = os.path.splitext(os.path.basename(bot_file))[0]
            all_participant_ids.add(bot_name)
            with open(bot_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        tsumego_name, success_rate = parts
                        all_tsumego_ids.add(tsumego_name)
                        records_to_process.append((bot_name, tsumego_name, float(success_rate)))

    # --- 2. Load User Data from database.sql ---
    sql_file = 'database.sql'
    user_attempts = []
    if os.path.exists(sql_file):
        print(f"Loading user data from {sql_file}...")
        insert_pattern = re.compile(r"INSERT INTO tsumego_reports VALUES\(\d+,'([^']+)','([^']+)',([01]),'.*?'\);")
        with open(sql_file, 'r') as f:
            for line in tqdm(f, desc="Parsing SQL"):
                match = insert_pattern.match(line)
                if match:
                    user_id, tsumego_id, success = match.groups()
                    user_attempts.append({'user_id': user_id, 'tsumego_id': tsumego_id, 'success': int(success)})

        if user_attempts:
            user_df = pd.DataFrame(user_attempts)
            user_df['user_id'] = user_df['user_id'].str[:8]  # Truncate user IDs

            # Calculate success rate for each user on each tsumego
            user_success_rates = user_df.groupby(['user_id', 'tsumego_id'])['success'].mean().reset_index()
            user_success_rates.rename(columns={'success': 'success_rate'}, inplace=True)

            # Add user data to processing lists
            for _, row in tqdm(user_success_rates.iterrows(), total=len(user_success_rates), desc="Processing Users"):
                all_participant_ids.add(row['user_id'])
                all_tsumego_ids.add(row['tsumego_id'])
                records_to_process.append((row['user_id'], row['tsumego_id'], row['success_rate']))
    else:
        print(f"Warning: {sql_file} not found. Skipping user data loading.")


    # --- 3. Create Mappings and Final DataFrame ---
    if not records_to_process:
         raise ValueError("No records found from bots or users. Cannot proceed.")

    participant_ids = sorted(list(all_participant_ids))
    participant_map = {name: i for i, name in enumerate(participant_ids)}

    tsumego_ids_unfiltered = sorted(list(all_tsumego_ids))
    # Filter to only include tsumegos with SGF files
    tsumego_ids = [tid for tid in tsumego_ids_unfiltered if os.path.exists(os.path.join("tsumegos", f"{tid}.sgf"))]
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
    # The 'bot_map' is now a 'participant_map', but we'll keep the variable name for compatibility
    return df, participant_map, tsumego_map, tsumego_ids


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


# --- Calculate Tsumego Ratings Numerically ---
def wrapper_calculate_tsumego_ratings(all_tsumego_names):
    """
    Recalculates tsumego ratings and quality from the saved parameters CSV file
    using numerical methods.
    """
    print("\nCalculating final tsumego ratings and quality from parameters...")
    try:
        params_df = pd.read_csv('ratings/tsumego_parameters_final.csv')
        bots_df = pd.read_csv('ratings/bot_ratings_final.csv')
    except FileNotFoundError as e:
        print(f"Error: Missing required file for calculation: {e.filename}")
        print("Please ensure the model has been trained and parameters have been saved.")
        return

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
    # Process all tsumegos (all are monotonic in the final parameter file)
    for _, row in tqdm(params_df.iterrows(), total=len(params_df), desc="Calculating Ratings"):
        tsumego_id = row['tsumego_id']

        # Extract parameters for the current tsumego
        params = {
            'difficulties': np.array([row[f'difficulty_{i + 1}'] for i in range(NUM_EXPERTS)]),
            'discriminations': np.array([row[f'discrimination_{i + 1}'] for i in range(NUM_EXPERTS)]),
            'weights': np.array([row[f'weight_{i + 1}'] for i in range(NUM_EXPERTS)])
        }

        # 1. Calculate Rating
        rating = find_rating_for_tsumego(params, PROB_TARGET)

        # 2. Calculate Quality
        # Use the theoretical standard deviation of the logistic distribution for skill difference
        theta_for_quality = rating + DIFFERENCE_FOR_QUALITY
        prob_at_quality_theta = predict_prob_for_tsumego(theta_for_quality, params)
        if DIFFERENCE_FOR_QUALITY > 0:
            quality = (prob_at_quality_theta - PROB_TARGET) / (1 - PROB_TARGET)
        else:
            quality = (PROB_TARGET - prob_at_quality_theta) / PROB_TARGET

        final_results.append({
            "tsumego_id": tsumego_id,
            "rating": rating + 2.5 * (quality-0.5), # difficult to guess feel almost 2 stones more difficult on average..
            "quality": quality
        })

    # Save ratings
    tsumego_rating_df = pd.DataFrame(final_results)[['tsumego_id', 'rating']].sort_values(by="rating", ascending=False)
    tsumego_rating_df.to_csv('ratings/tsumego_ratings_final.csv', index=False)

    # Save quality
    tsumego_quality_df = pd.DataFrame(final_results)[['tsumego_id', 'quality']].sort_values(by="quality", ascending=False)
    tsumego_quality_df.to_csv('ratings/tsumego_quality_final.csv', index=False)

    print("\nRecalculated ratings saved to 'ratings/tsumego_ratings_final.csv'")
    print("Recalculated quality saved to 'ratings/tsumego_quality_final.csv'")


# --- Save ratings snapshot ---
def save_ratings_snapshot(model, bot_map, tsumego_map, all_tsumego_names,
                          monotonic_tsumego_map):
    """
    Calculates and saves the bot and tsumego parameters to CSV files.
    """
    model.eval()
    bot_filename = "ratings/bot_ratings_final.csv"
    tsumego_parameters_filename = "ratings/tsumego_parameters_final.csv"

    os.makedirs("ratings", exist_ok=True)

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
                predicted_rates = model(bot_batch, tsumego_batch)  # Squeeze is not needed anymore

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
    # Determine num_workers based on CPU count, capped to avoid issues
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
        monotonic_tsumego_names,  # Pass only the names of tsumegos that were trained
        monotonic_tsumego_map
    )


# --- Main Execution ---
if __name__ == '__main__':
    # "cd pwa && npx wrangler d1 export DB --remote --output=../attempts/database.sql"

    df_full, bot_map, tsumego_map, all_tsumego_names_with_sgf = load_data(DATA_DIRECTORY)

    # Step 1: Train the model and save the learned parameters
    # rank_tsumegos_and_bots(df_full, bot_map, tsumego_map, all_tsumego_names_with_sgf)

    # Step 2: Load the saved parameters and calculate final ratings and quality
    # We pass all original names so it can handle non-monotonic ones if needed in the future
    # (Currently, they are just excluded from the parameter file)
    wrapper_calculate_tsumego_ratings(all_tsumego_names_with_sgf)
