import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# --- 2. Data Loading and Preprocessing ---
def load_data(data_dir):
    """
    Loads all bot data from the specified directory.
    It dynamically discovers all bots (from filenames) and all tsumegos
    (from the content of the files).
    """
    print("Loading and preprocessing data...")
    bot_files = glob.glob(os.path.join(data_dir, "*.txt"))
    if not bot_files:
        raise FileNotFoundError(f"No bot data files (.txt) found in '{data_dir}'")

    # Dynamically discover bots from filenames
    bot_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in bot_files])
    bot_map = {name: i for i, name in enumerate(bot_ids)}

    # Dynamically discover all unique tsumegos by reading the files
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

    # Create records for observed interactions.
    # This inherently handles missing data, as non-existent pairs are simply not added.
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

# --- 3. The Rating Model (Using Logistic Distribution) ---
class RatingModel(nn.Module):
    """
    A PyTorch module for the logistic-based rating system (Elo-style).
    """
    def __init__(self, num_bots, num_tsumegos):
        super().__init__()
        # Bot ratings (theta)
        self.bot_ratings = nn.Embedding(num_bots, 1)
        # Tsumego difficulties (beta)
        self.tsumego_difficulties = nn.Embedding(num_tsumegos, 1)

        # Initialize ratings from a standard normal distribution.
        # The optimization will then find their optimal values.
        nn.init.normal_(self.bot_ratings.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_difficulties.weight, mean=0, std=1)

    def forward(self, bot_indices, tsumego_indices):
        bot_rating = self.bot_ratings(bot_indices)
        tsumego_difficulty = self.tsumego_difficulties(tsumego_indices)

        # The core of the logistic model (Elo system)
        # P(solve) = sigmoid(bot_rating - tsumego_difficulty)
        diff = bot_rating - tsumego_difficulty
        return torch.sigmoid(diff)

# --- 4. The Training Process ---
def train_model(model, data_loader, epochs=5, learning_rate=0.01):
    """
    Trains the rating model using gradient descent.
    """
    # Binary Cross Entropy is the correct loss function for a probabilistic
    # model with a sigmoid output. It is the negative log-likelihood.
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for bot_batch, tsumego_batch, rate_batch in progress_bar:
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass: get predicted success probabilities
            predicted_rates = model(bot_batch, tsumego_batch).squeeze()

            # Clamp predictions to avoid log(0) issues in the loss function
            predicted_rates = torch.clamp(predicted_rates, 1e-9, 1 - 1e-9)

            # Calculate loss between predictions and actual success rates
            loss = loss_fn(predicted_rates, rate_batch.float())

            # Backward pass: compute gradients
            loss.backward()

            # Update ratings based on gradients
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{(total_loss / len(progress_bar)):.4f}")

    print("Training complete.")

# --- Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    # The only thing you need to specify is the directory with your bot data.
    DATA_DIRECTORY = "attempts"
    BATCH_SIZE = 4096  # Increased for potentially faster training on large datasets
    EPOCHS = 5
    LEARNING_RATE = 0.01

    # --- Step 2: Load and prepare the data dynamically ---
    df, bot_map, tsumego_map = load_data(DATA_DIRECTORY)

    # Create a PyTorch DataLoader for efficient mini-batching
    dataset = TensorDataset(
        torch.LongTensor(df.bot_id.values),
        torch.LongTensor(df.tsumego_id.values),
        torch.Tensor(df.success_rate.values)
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Step 3: Initialize the model with dynamically determined sizes ---
    model = RatingModel(num_bots=len(bot_map), num_tsumegos=len(tsumego_map))

    # --- Step 4: Train the model ---
    train_model(model, data_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # --- Step 5: Extract, format, and display the final ratings ---
    model.eval()
    with torch.no_grad(): # Ensure no gradients are calculated during extraction
        bot_ratings = model.bot_ratings.weight.cpu().numpy().flatten()
        tsumego_difficulties = model.tsumego_difficulties.weight.cpu().numpy().flatten()

    # Create a reverse mapping from index back to original ID for reporting
    rev_bot_map = {i: name for name, i in bot_map.items()}
    rev_tsumego_map = {i: name for name, i in tsumego_map.items()}

    bot_ratings_df = pd.DataFrame({
        "bot_id": [rev_bot_map.get(i) for i in range(len(bot_ratings))],
        "rating": bot_ratings
    }).sort_values(by="rating", ascending=False)

    tsumego_ratings_df = pd.DataFrame({
        "tsumego_id": [rev_tsumego_map.get(i) for i in range(len(tsumego_difficulties))],
        "difficulty": tsumego_difficulties
    }).sort_values(by="difficulty", ascending=False)

    print("\n--- Top 15 Bots by Rating ---")
    print(bot_ratings_df.head(15))

    print("\n--- Top 15 Most Difficult Tsumegos ---")
    print(tsumego_ratings_df.head(15))

    # Save the full results to CSV files
    bot_ratings_df.to_csv("bot_ratings_logistic.csv", index=False)
    tsumego_ratings_df.to_csv("tsumego_ratings_logistic.csv", index=False)
    print("\nFull ratings saved to 'bot_ratings_logistic.csv' and 'tsumego_ratings_logistic.csv'")