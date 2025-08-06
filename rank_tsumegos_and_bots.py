import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

APPROVED = [
    "gnugo_1.txt",
    "gnugo_6.txt",
    "katago_1710_100.txt",
    "katago_12k.txt",
    "katago_3d.txt",
    "katago_9d.txt",
    "katago_9958_10.txt",
    "katago_9958_100.txt"
]


# --- 2. Data Loading and Preprocessing ---
def load_data(data_dir):
    """
    Loads all bot data from the specified directory.
    It dynamically discovers all bots (from filenames) and all tsumegos
    (from the content of the files).
    """
    print("Loading and preprocessing data...")
    # bot_files = glob.glob(os.path.join(data_dir, "*.txt"))
    # bot_files = [os.path.join(data_dir, filename) for filename in APPROVED if os.path.exists(os.path.join(data_dir, filename))]
    bot_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if "corrupt" in f and f.endswith('.txt')]
    bot_files.extend([os.path.join(data_dir, filename) for filename in APPROVED if os.path.exists(os.path.join(data_dir, filename))])

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
        nn.init.normal_(self.bot_ratings.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_difficulties.weight, mean=0, std=1)

    def forward(self, bot_indices, tsumego_indices):
        bot_rating = self.bot_ratings(bot_indices)
        tsumego_difficulty = self.tsumego_difficulties(tsumego_indices)
        # The core of the logistic model (Elo system)
        diff = bot_rating - tsumego_difficulty
        return torch.sigmoid(diff)


# --- NEW: Function to save a snapshot of the ratings ---
def save_ratings_snapshot(model, bot_map, tsumego_map, epoch_num_str):
    """
    Extracts ratings from the model and saves them to CSV files.
    'epoch_num_str' can be an integer or a string like 'final'.
    """
    # Set model to evaluation mode for consistent results
    model.eval()

    output_suffix = f"epoch_{epoch_num_str}" if isinstance(epoch_num_str, int) else epoch_num_str
    bot_filename = f"ratings/bot_ratings_{output_suffix}.csv"
    tsumego_filename = f"ratings/tsumego_ratings_{output_suffix}.csv"

    print(f"\nSaving snapshot for epoch {epoch_num_str}...")

    with torch.no_grad():  # Ensure no gradients are calculated during extraction
        bot_ratings = model.bot_ratings.weight.cpu().numpy().flatten()
        tsumego_difficulties = model.tsumego_difficulties.weight.cpu().numpy().flatten()

    # Create reverse mappings from index back to original ID for reporting
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

    # Save the full results to CSV files
    bot_ratings_df.to_csv(bot_filename, index=False)
    tsumego_ratings_df.to_csv(tsumego_filename, index=False)
    print(f"Ratings saved to '{bot_filename}' and '{tsumego_filename}'")


# --- 4. The Training Process (MODIFIED) ---
def train_model(model, data_loader, bot_map, tsumego_map, device, epochs=5, learning_rate=0.01):
    """
    Trains the rating model and saves results after each epoch.
    """
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for bot_batch, tsumego_batch, rate_batch in progress_bar:
            # --- FIX: Move data batches to the correct device ---
            bot_batch = bot_batch.to(device)
            tsumego_batch = tsumego_batch.to(device)
            rate_batch = rate_batch.to(device)

            optimizer.zero_grad()
            predicted_rates = model(bot_batch, tsumego_batch).squeeze()
            predicted_rates = torch.clamp(predicted_rates, 1e-9, 1 - 1e-9)
            loss = loss_fn(predicted_rates, rate_batch.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            progress_bar.set_postfix(loss=f"{(total_loss / count):.4f}")

        save_ratings_snapshot(model, bot_map, tsumego_map, epoch + 1)

    print("\nTraining complete.")


# --- Main Execution (MODIFIED) ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIRECTORY = "attempts"
    BATCH_SIZE = 2 * 4096
    EPOCHS = 5
    LEARNING_RATE = 0.01

    # --- Step 1: Load and prepare the data dynamically ---
    df, bot_map, tsumego_map = load_data(DATA_DIRECTORY)

    dataset = TensorDataset(
        torch.LongTensor(df.bot_id.values),
        torch.LongTensor(df.tsumego_id.values),
        torch.Tensor(df.success_rate.values)
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- Step 2: Initialize the model and define the device ---
    model = RatingModel(num_bots=len(bot_map), num_tsumegos=len(tsumego_map))

    # --- FIX: Determine device and move model ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nCUDA is available. Moving model to GPU...")
    else:
        device = torch.device("cpu")
        print("\nCUDA not available. Running on CPU.")
    model.to(device)

    # --- Step 3: Train the model ---
    # --- FIX: Pass the device to the training function ---
    train_model(model, data_loader, bot_map, tsumego_map, device, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # --- Step 4: Save a final, clearly marked version of the ratings ---
    save_ratings_snapshot(model, bot_map, tsumego_map, "final")

    print("\nScript finished.")
