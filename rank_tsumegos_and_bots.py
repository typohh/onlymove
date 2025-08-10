import glob
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# --- Data Loading ---
def load_data(data_dir):
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
    def __init__(self, num_bots, num_tsumegos):
        super().__init__()
        self.bot_ratings = nn.Embedding(num_bots, 1)           # θ_b
        self.tsumego_difficulties = nn.Embedding(num_tsumegos, 1)  # β_t
        self.tsumego_discrimination = nn.Embedding(num_tsumegos, 1) # α_t
        self.tsumego_guessing = nn.Embedding(num_tsumegos, 1)       # c_t

        nn.init.normal_(self.bot_ratings.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_difficulties.weight, mean=0, std=1)
        nn.init.normal_(self.tsumego_discrimination.weight, mean=1.0, std=0.1)
        nn.init.constant_(self.tsumego_guessing.weight, -2.2)  # sigmoid ~ 0.1

    def forward(self, bot_indices, tsumego_indices):
        theta = self.bot_ratings(bot_indices)                # (batch, 1)
        beta = self.tsumego_difficulties(tsumego_indices)    # (batch, 1)
        alpha = self.tsumego_discrimination(tsumego_indices) # (batch, 1)
        c = torch.sigmoid(self.tsumego_guessing(tsumego_indices))  # ensure (0,1)
        diff = alpha * (theta - beta)
        return c + (1 - c) * torch.sigmoid(diff)


# --- Save ratings snapshot (only 80% rating) ---
def save_ratings_snapshot(model, bot_map, tsumego_map, epoch_num_str):
    model.eval()
    output_suffix = f"epoch_{epoch_num_str}" if isinstance(epoch_num_str, int) else epoch_num_str
    bot_filename = f"ratings/bot_ratings_{output_suffix}.csv"
    tsumego_filename = f"ratings/tsumego_ratings_{output_suffix}.csv"

    with torch.no_grad():
        bot_ratings = model.bot_ratings.weight.cpu().numpy().flatten()
        tsumego_difficulties = model.tsumego_difficulties.weight.cpu().numpy().flatten()
        tsumego_discrimination = model.tsumego_discrimination.weight.cpu().numpy().flatten()
        tsumego_guessing = torch.sigmoid(model.tsumego_guessing.weight).cpu().numpy().flatten()

    rev_bot_map = {i: name for name, i in bot_map.items()}
    rev_tsumego_map = {i: name for name, i in tsumego_map.items()}

    # Tsumego: rating needed for 80% success
    tsumego_rating_80 = tsumego_difficulties + (
        np.log((0.8 - tsumego_guessing) / (0.2 - 0.2 * tsumego_guessing))
        / tsumego_discrimination
    )

    tsumego_df = pd.DataFrame({
        "tsumego_id": [rev_tsumego_map[i] for i in range(len(tsumego_rating_80))],
        "rating": tsumego_rating_80
    }).sort_values(by="rating", ascending=False)

    # Bot: rating where they'd succeed 80% on a "standard" tsumego (β=0, α=1, c=0)
    bot_rating_80 = bot_ratings - 1.386  # This is arbitrary; may not be meaningful for 3PL

    bot_df = pd.DataFrame({
        "bot_id": [rev_bot_map[i] for i in range(len(bot_rating_80))],
        "rating": bot_rating_80
    }).sort_values(by="rating", ascending=False)

    tsumego_df.to_csv(tsumego_filename, index=False)
    bot_df.to_csv(bot_filename, index=False)
    print(f"80% ratings saved to '{bot_filename}' and '{tsumego_filename}'")


# --- Training ---
def train_model(model, data_loader, bot_map, tsumego_map, device, epochs=5, learning_rate=0.01):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for bot_batch, tsumego_batch, rate_batch in progress_bar:
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


# --- Main ---
if __name__ == '__main__':
    import numpy as np
    DATA_DIRECTORY = "attempts"
    BATCH_SIZE = 2 * 4096
    EPOCHS = 7
    LEARNING_RATE = 0.01

    df, bot_map, tsumego_map = load_data(DATA_DIRECTORY)

    dataset = TensorDataset(
        torch.LongTensor(df.bot_id.values),
        torch.LongTensor(df.tsumego_id.values),
        torch.Tensor(df.success_rate.values)
    )
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = RatingModel(num_bots=len(bot_map), num_tsumegos=len(tsumego_map))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("\nCUDA is available. Moving model to GPU...")
    else:
        device = torch.device("cpu")
        print("\nCUDA not available. Running on CPU.")
    model.to(device)

    train_model(model, data_loader, bot_map, tsumego_map, device, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    save_ratings_snapshot(model, bot_map, tsumego_map, "final")

    print("\nScript finished.")
