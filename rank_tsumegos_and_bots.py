import glob
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func  # For softplus
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import random  # Keep for potential future sampling/testing

# --- Configuration ---
DATA_DIRECTORY = "attempts"
BATCH_SIZE = 8192
LEARNING_RATE = 0.01
LOSS_DROP_THRESHOLD = 0.001  # Stop training if loss improves by less than 0.1%
MIN_BATCHES = 50  # Minimum number of batches before checking loss drop
PROB_TARGET = 0.8  # Target probability for tsumego rating calculation

# --- Global Epsilon for numerical stability ---
EPSILON = 1e-9

# Set a fixed seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# --- Data Loading ---
def load_data(data_dir):
    """Loads and preprocesses tsumego attempt data from text files."""
    print("Loading and preprocessing data...")
    bot_files = glob.glob(os.path.join(data_dir, "*.txt"))

    if not bot_files:
        raise FileNotFoundError(f"No bot data files (.txt) found in '{data_dir}'")

    bot_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in bot_files])
    bot_map = {name: i for i, name in enumerate(bot_ids)}

    all_tsumego_ids = set()
    for bot_file in tqdm(bot_files, desc="Discovering Tsumegos"):
        with open(bot_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    all_tsumego_ids.add(parts[0])

    tsumego_ids = sorted(list(all_tsumego_ids))
    tsumego_map = {name: i for i, name in enumerate(tsumego_ids)}

    tqdm.write(f"Discovered {len(bot_map)} bots and {len(tsumego_map)} tsumegos.")

    records = []
    for bot_file in tqdm(bot_files, desc="Loading Records"):
        bot_name = os.path.splitext(os.path.basename(bot_file))[0]
        bot_idx = bot_map[bot_name]
        with open(bot_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tsumego_name, success_rate = parts
                    # Only add records for tsumegos we've mapped
                    if tsumego_name in tsumego_map:
                        tsumego_idx = tsumego_map[tsumego_name]
                        records.append((bot_idx, tsumego_idx, float(success_rate)))

    df = pd.DataFrame(records, columns=["bot_id", "tsumego_id", "success_rate"])
    tqdm.write(f"Loaded {len(df)} total solve attempts.")
    return df, bot_map, tsumego_map, tsumego_ids  # Return tsumego_ids for convenience


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
    strong_bot_avg = df_ranked_bots[df_ranked_bots['bot_prelim_rank'] <= strong_bots_threshold].groupby('tsumego_name')['success_rate'].mean().rename('strong_bot_success')
    weak_bot_avg = df_ranked_bots[df_ranked_bots['bot_prelim_rank'] > weak_bots_threshold].groupby('tsumego_name')['success_rate'].mean().rename('weak_bot_success')

    # Combine the average success rates into a single DataFrame
    tsumego_stats = pd.concat([strong_bot_avg, weak_bot_avg], axis=1)

    # Drop rows where a tsumego was not attempted by either strong or weak bots
    tsumego_stats.dropna(inplace=True)

    # Identify non-monotonic tsumegos
    non_monotonic_mask = tsumego_stats['strong_bot_success'] < tsumego_stats['weak_bot_success']
    non_monotonic_tsumego_ids = set(tsumego_stats[non_monotonic_mask].index)

    print(f"Identified {len(non_monotonic_tsumego_ids)} non-monotonic tsumegos.")
    return non_monotonic_tsumego_ids



# --- Rating Model (3PL IRT) ---
class RatingModel(nn.Module):
    """A 3-Parameter Logistic Item Response Theory model."""

    def __init__(self, num_bots, num_tsumegos):
        super().__init__()
        self.num_bots = num_bots  # Store for later use in save_ratings_snapshot
        self.num_tsumegos = num_tsumegos  # Store for later use

        self.bot_ratings = nn.Embedding(num_bots, 1)  # θ_b (skill)
        self.tsumego_difficulties_raw = nn.Embedding(num_tsumegos, 1)  # β_t (difficulty)
        self.tsumego_discrimination_raw = nn.Embedding(num_tsumegos, 1)  # α_t (discrimination)
        self.tsumego_guessing_raw = nn.Embedding(num_tsumegos, 1)  # c_t (guessing)

        # Initialize parameters
        nn.init.normal_(self.bot_ratings.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_difficulties_raw.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_discrimination_raw.weight, mean=1.0, std=0.1)
        nn.init.constant_(self.tsumego_guessing_raw.weight, -2.2)  # Corresponds to sigmoid(-2.2) approx 0.1

    def forward(self, bot_indices, tsumego_indices):
        """Calculates the predicted success rate."""
        theta = self.bot_ratings(bot_indices)

        # Apply softplus to discrimination parameter and add epsilon to ensure positivity and avoid zero
        alpha = func.softplus(self.tsumego_discrimination_raw(tsumego_indices)) + EPSILON * 10

        beta = self.tsumego_difficulties_raw(tsumego_indices)

        # Apply sigmoid to guessing parameter raw output
        c = torch.sigmoid(self.tsumego_guessing_raw(tsumego_indices))

        diff = alpha * (theta - beta)
        non_guess_success_prob = torch.sigmoid(diff)

        return c + (1 - c) * non_guess_success_prob


# --- Save ratings snapshot ---
def save_ratings_snapshot(model, bot_map, tsumego_map, all_tsumego_names,
                          monotonic_tsumego_map, non_monotonic_tsumego_ids, output_suffix):
    """
    Calculates and saves the bot and tsumego ratings to CSV files.
    Non-monotonic tsumegos are added back with a rating equal to the hardest monotonic tsumego.
    """
    model.eval()
    bot_filename = f"ratings/bot_ratings_{output_suffix}.csv"
    tsumego_filename = f"ratings/tsumego_ratings_{output_suffix}.csv"

    os.makedirs("ratings", exist_ok=True)

    # Get parameters for monotonic tsumegos
    with torch.no_grad():
        bot_ratings_np = model.bot_ratings.weight.cpu().numpy().flatten()
        monotonic_tsumego_difficulties = model.tsumego_difficulties_raw.weight.cpu().numpy().flatten()
        # Apply softplus and epsilon as in forward pass for alpha
        monotonic_tsumego_discrimination = (func.softplus(model.tsumego_discrimination_raw.weight) + EPSILON * 10).cpu().numpy().flatten()
        monotonic_tsumego_guessing_logit = model.tsumego_guessing_raw.weight.cpu().numpy().flatten()
        monotonic_tsumego_c = 1 / (1 + np.exp(-monotonic_tsumego_guessing_logit))

    rev_bot_map = {i: name for name, i in bot_map.items()}

    # --- Calculate Monotonic Tsumego Ratings ---
    # Argument for the logit function: (P - c) / (1 - c)
    p_minus_c = PROB_TARGET - monotonic_tsumego_c
    one_minus_c = 1 - monotonic_tsumego_c
    logit_arg = np.clip(p_minus_c / (one_minus_c + EPSILON), EPSILON, 1 - EPSILON)
    logit_val = np.log(logit_arg / (1 - logit_arg))

    # Full 3PL formula to solve for rating (theta)
    monotonic_tsumego_rating = monotonic_tsumego_difficulties + logit_val / (monotonic_tsumego_discrimination + EPSILON)

    # Map monotonic tsumego indices back to their original names
    rev_monotonic_tsumego_map = {i: name for name, i in monotonic_tsumego_map.items()}
    monotonic_tsumego_df = pd.DataFrame({
        "tsumego_id": [rev_monotonic_tsumego_map[i] for i in range(len(monotonic_tsumego_rating))],
        "rating": monotonic_tsumego_rating
    })

    # Find the hardest monotonic tsumego rating
    hardest_monotonic_rating = monotonic_tsumego_df['rating'].max()
    print(f"\nHardest monotonic tsumego rating: {hardest_monotonic_rating:.2f}")

    # --- Combine all tsumego ratings ---
    final_tsumego_ratings = []
    for tsumego_name in all_tsumego_names:
        if tsumego_name in non_monotonic_tsumego_ids:
            # Assign rating of the hardest monotonic tsumego
            final_tsumego_ratings.append({"tsumego_id": tsumego_name, "rating": hardest_monotonic_rating})
        else:
            # Get rating from the trained monotonic model
            rating_row = monotonic_tsumego_df[monotonic_tsumego_df['tsumego_id'] == tsumego_name]
            final_tsumego_ratings.append({"tsumego_id": tsumego_name, "rating": rating_row['rating'].iloc[0]})

    tsumego_df_final = pd.DataFrame(final_tsumego_ratings).sort_values(by="rating", ascending=False)

    bot_df = pd.DataFrame({
        "bot_id": [rev_bot_map[i] for i in range(len(bot_ratings_np))],
        "rating": bot_ratings_np
    }).sort_values(by="rating", ascending=False)

    tsumego_df_final.to_csv(tsumego_filename, index=False)
    bot_df.to_csv(bot_filename, index=False)
    print(f"\nFinal ratings saved to '{bot_filename}' and '{tsumego_filename}'")


# --- Training ---
def train_model(model, data_loader, device):
    """
    Trains the IRT model, stopping when loss improvement is less than a threshold.
    """
    loss_fn = nn.BCELoss()
    # Add a small L2 regularization (weight_decay) to all parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    epoch = 0
    previous_loss = float('inf')
    batches_processed = 0

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
                predicted_rates = model(bot_batch, tsumego_batch).squeeze()

                # Clamp predictions to avoid log(0) or log(1) with BCELoss
                predicted_rates = torch.clamp(predicted_rates, EPSILON, 1 - EPSILON)

                loss = loss_fn(predicted_rates, rate_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
                progress_bar.set_postfix(loss=f"{(total_loss / count):.4f}")
                batches_processed += 1

        avg_loss = total_loss / count

        # Check for early stopping only after a minimum number of batches
        if batches_processed > MIN_BATCHES:
            relative_loss_drop = (previous_loss - avg_loss) / previous_loss
            print(f"\nEpoch {epoch + 1} complete. Average Loss: {avg_loss:.6f}. Relative Loss Drop: {relative_loss_drop:.4%}")
            if relative_loss_drop < LOSS_DROP_THRESHOLD:
                print(f"Loss improvement ({relative_loss_drop:.4%}) is below threshold ({LOSS_DROP_THRESHOLD:.2%}). Stopping training.")
                break
            previous_loss = avg_loss
            batches_processed = 0
        epoch += 1

    print("\nTraining complete.")


# --- Main Execution ---
if __name__ == '__main__':
    # --- Data Loading ---
    df_full, bot_map, tsumego_map, all_tsumego_names = load_data(DATA_DIRECTORY)

    # Define rev_tsumego_map here so it's accessible globally
    rev_tsumego_map = {i: name for name, i in tsumego_map.items()}

    # --- Preliminary Bot Ranking ---
    bot_prelim_rank_map = get_preliminary_bot_ranking(df_full, bot_map)

    # --- Identify Non-Monotonic Tsumegos ---
    # Passing all_tsumego_names is important here to check every original tsumego
    non_monotonic_tsumego_ids = identify_non_monotonic_tsumegos(df_full, tsumego_map, bot_prelim_rank_map)

    # --- Filter Data for Monotonic Training ---
    monotonic_tsumego_names = [name for name in all_tsumego_names if name not in non_monotonic_tsumego_ids]

    # Create a new map for monotonic tsumegos for training
    monotonic_tsumego_map = {name: i for i, name in enumerate(monotonic_tsumego_names)}

    # Filter the DataFrame to include only monotonic tsumegos
    # Now rev_tsumego_map is correctly defined in this scope
    df_monotonic = df_full[df_full['tsumego_id'].map(lambda x: rev_tsumego_map[x] in monotonic_tsumego_map)].copy()

    # Re-map tsumego_ids in df_monotonic to their new monotonic_tsumego_map indices
    # This line was also causing an issue because it was recreating rev_tsumego_map
    # which already exists, and the mapping logic was a bit convoluted.
    # The correct way is to map the original tsumego_id (integer) to its name,
    # then map that name to its new monotonic_tsumego_map index.
    df_monotonic['tsumego_id_monotonic'] = df_monotonic['tsumego_id'].map(lambda x: monotonic_tsumego_map[rev_tsumego_map[x]])

    print(f"\nTraining on {len(monotonic_tsumego_map)} monotonic tsumegos.")
    print(f"Removed {len(non_monotonic_tsumego_ids)} non-monotonic tsumegos from training set.")

    # Create dataset and DataLoader for monotonic tsumegos
    dataset_monotonic = TensorDataset(
        torch.LongTensor(df_monotonic.bot_id.values),
        torch.LongTensor(df_monotonic.tsumego_id_monotonic.values),  # Use new monotonic tsumego IDs
        torch.Tensor(df_monotonic.success_rate.values)
    )
    data_loader_monotonic = DataLoader(dataset_monotonic, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Model Initialization ---
    model = RatingModel(num_bots=len(bot_map), num_tsumegos=len(monotonic_tsumego_map))

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

    # --- Final Rating Save (including non-monotonic tsumegos) ---
    save_ratings_snapshot(
        model,
        bot_map,
        tsumego_map,  # Pass the original tsumego_map for full name lookup
        all_tsumego_names,
        monotonic_tsumego_map,  # Pass the monotonic map for indexing trained parameters
        non_monotonic_tsumego_ids,
        "final"
    )

    print("\nScript finished.")