import glob
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np


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

    print(f"Discovered {len(bot_map)} bots and {len(tsumego_map)} tsumegos.")

    records = []
    for bot_file in tqdm(bot_files, desc="Loading Records"):
        bot_name = os.path.splitext(os.path.basename(bot_file))[0]
        bot_idx = bot_map[bot_name]
        with open(bot_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tsumego_name, success_rate = parts
                    if tsumego_name in tsumego_map:
                        tsumego_idx = tsumego_map[tsumego_name]
                        records.append((bot_idx, tsumego_idx, float(success_rate)))

    df = pd.DataFrame(records, columns=["bot_id", "tsumego_id", "success_rate"])
    print(f"Loaded {len(df)} total solve attempts.")
    return df, bot_map, tsumego_map


# --- Rating Model (3PL IRT) ---
class RatingModel(nn.Module):
    """A 3-Parameter Logistic Item Response Theory model."""

    def __init__(self, num_bots, num_tsumegos):
        super().__init__()
        self.bot_ratings = nn.Embedding(num_bots, 1)  # θ_b (skill)
        self.tsumego_difficulties = nn.Embedding(num_tsumegos, 1)  # β_t (difficulty)
        self.tsumego_discrimination = nn.Embedding(num_tsumegos, 1)  # α_t (discrimination)
        self.tsumego_guessing = nn.Embedding(num_tsumegos, 1)  # c_t (guessing)

        # Initialize parameters to reasonable starting values
        nn.init.normal_(self.bot_ratings.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_difficulties.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_discrimination.weight, mean=1.0, std=0.1)
        nn.init.constant_(self.tsumego_guessing.weight, -2.2)  # Corresponds to sigmoid(-2.2) ≈ 0.1

    def forward(self, bot_indices, tsumego_indices):
        """Calculates the predicted success rate."""
        theta = self.bot_ratings(bot_indices)
        beta = self.tsumego_difficulties(tsumego_indices)
        alpha = self.tsumego_discrimination(tsumego_indices)
        c = torch.sigmoid(self.tsumego_guessing(tsumego_indices))

        diff = alpha * (theta - beta)
        non_guess_success_prob = torch.sigmoid(diff)

        return c + (1 - c) * non_guess_success_prob


# --- Save ratings snapshot ---
def save_ratings_snapshot(model, bot_map, tsumego_map, epoch_num_str):
    """Calculates and saves the bot and tsumego ratings to CSV files."""
    model.eval()
    output_suffix = f"epoch_{epoch_num_str}" if isinstance(epoch_num_str, int) else epoch_num_str
    bot_filename = f"ratings/bot_ratings_{output_suffix}.csv"
    tsumego_filename = f"ratings/tsumego_ratings_{output_suffix}.csv"

    os.makedirs("ratings", exist_ok=True)

    with torch.no_grad():
        bot_ratings = model.bot_ratings.weight.cpu().numpy().flatten()
        tsumego_difficulties = model.tsumego_difficulties.weight.cpu().numpy().flatten()
        tsumego_discrimination = model.tsumego_discrimination.weight.cpu().numpy().flatten()
        # Get the "logit of the guess" parameter and apply sigmoid to get the guess probability `c`
        tsumego_guessing_logit = model.tsumego_guessing.weight.cpu().numpy().flatten()
        tsumego_c = 1 / (1 + np.exp(-tsumego_guessing_logit)) # This is the sigmoid function

    rev_bot_map = {i: name for name, i in bot_map.items()}
    rev_tsumego_map = {i: name for name, i in tsumego_map.items()}

    # --- Corrected 3PL Rating Calculation ---
    # This finds the skill level (theta) needed for a 50% success rate
    PROB_TARGET = 0.5
    epsilon = 1e-9 # A small value to prevent division by zero or log(0)

    # Argument for the logit function: (P - c) / (1 - c)
    # We clip the value to avoid math errors if c is very close to P
    p_minus_c = PROB_TARGET - tsumego_c
    one_minus_c = 1 - tsumego_c
    logit_arg = np.clip(p_minus_c / (one_minus_c + epsilon), epsilon, 1 - epsilon)

    # Logit function: log(p / (1 - p))
    logit_val = np.log(logit_arg / (1 - logit_arg))

    # Full 3PL formula to solve for rating (theta): beta + logit_val / alpha
    tsumego_rating = tsumego_difficulties + logit_val / (tsumego_discrimination + epsilon)

    # --- DataFrame creation with original format ---
    tsumego_df = pd.DataFrame({
        "tsumego_id": [rev_tsumego_map[i] for i in range(len(tsumego_rating))],
        "rating": tsumego_rating
    }).sort_values(by="rating", ascending=False)

    bot_df = pd.DataFrame({
        "bot_id": [rev_bot_map[i] for i in range(len(bot_ratings))], # Corrected typo here
        "rating": bot_ratings
    }).sort_values(by="rating", ascending=False)

    tsumego_df.to_csv(tsumego_filename, index=False)
    bot_df.to_csv(bot_filename, index=False)
    print(f"\nFinal ratings saved to '{bot_filename}' and '{tsumego_filename}'")


# --- Training ---
def train_model(model, data_loader, device, learning_rate, loss_drop_threshold, max_epochs):
    """
    Trains the IRT model, stopping when loss improvement is less than a threshold.
    """
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)

    epoch = 0
    previous_loss = float('inf')

    while epoch < max_epochs:
        model.train()
        total_loss = 0
        count = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")
        for bot_batch, tsumego_batch, rate_batch in progress_bar:
            bot_batch = bot_batch.to(device)
            tsumego_batch = tsumego_batch.to(device)
            rate_batch = rate_batch.to(device).float()

            optimizer.zero_grad()

            predicted_rates = model(bot_batch, tsumego_batch).squeeze()
            predicted_rates = torch.clamp(predicted_rates, 1e-9, 1 - 1e-9)

            loss = loss_fn(predicted_rates, rate_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            progress_bar.set_postfix(loss=f"{(total_loss / count):.4f}")

        avg_loss = total_loss / count

        if epoch > 0:
            relative_loss_drop = (previous_loss - avg_loss) / previous_loss
            print(f"\nEpoch {epoch + 1} complete. Average Loss: {avg_loss:.6f}. Relative Loss Drop: {relative_loss_drop:.4%}")
            if relative_loss_drop < loss_drop_threshold:
                print(f"Loss improvement ({relative_loss_drop:.4%}) is below threshold ({loss_drop_threshold:.2%}). Stopping training.")
                break
        else:
            print(f"\nEpoch {epoch + 1} complete. Average Loss: {avg_loss:.6f}.")

        previous_loss = avg_loss
        epoch += 1

    if epoch == max_epochs:
        print(f"\nTraining stopped after reaching the maximum of {max_epochs} epochs.")

    print("\nTraining complete.")


# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIRECTORY = "attempts"
    BATCH_SIZE = 8192
    LEARNING_RATE = 0.01
    LOSS_DROP_THRESHOLD = 0.02  # Stop training if loss improves by less than 2%
    MAX_EPOCHS = 100  # Maximum number of epochs to run as a safeguard

    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Data Loading ---
    df, bot_map, tsumego_map = load_data(DATA_DIRECTORY)

    dataset = TensorDataset(
        torch.LongTensor(df.bot_id.values),
        torch.LongTensor(df.tsumego_id.values),
        torch.Tensor(df.success_rate.values)
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Model Initialization ---
    model = RatingModel(num_bots=len(bot_map), num_tsumegos=len(tsumego_map))

    # --- Device Configuration ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nCUDA is available. Moving model to GPU...")
        model.to(device)
    else:
        device = torch.device("cpu")
        print("\nCUDA not available. Running on CPU.")

    # --- Training and Final Save ---
    train_model(
        model,
        data_loader,
        device,
        learning_rate=LEARNING_RATE,
        loss_drop_threshold=LOSS_DROP_THRESHOLD,
        max_epochs=MAX_EPOCHS
    )
    save_ratings_snapshot(model, bot_map, tsumego_map, "final")

    print("\nScript finished.")