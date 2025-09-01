import os
import random

import numpy as np
import pandas as pd

from katago_queries import get_rank_number


def intepolate(points, rating):
    """
    Performs linear interpolation to find a new value based on a rating.

    Args:
        points: A list of tuples, where each tuple is (new_value, old_value).
        rating: The rating (old_value) to interpolate from.

    Returns:
        The interpolated new value.
    """
    points.sort(key=lambda p: p[1])

    above = next((p for p in points if p[1] >= rating), None)
    below = next((p for p in reversed(points) if p[1] <= rating), None)

    if above is None and below is None: return 0.5
    if above is None: return below[0]
    if below is None: return above[0]
    if above[1] == below[1]: return above[0]

    t = (rating - below[1]) / (above[1] - below[1])
    return below[0] + t * (above[0] - below[0])

RATINGS_DIR = "ratings"

def get_tsumego_ratings():
    """
    Calculates final tsumego ratings based on bot and tsumego performance data.

    This function loads raw ratings, creates a normalized mapping based on tsumego
    quantiles, and then uses a set of anchor points to interpolate a final,
    calibrated rank for each problem.

    Args:
        ratings_dir: Path to the directory containing the ratings .csv files.

    Returns:
        A dictionary mapping tsumego_id (str) to its final calculated rank (float).
    """

    final_tsumego_ratings = {}

    bot_ratings_path = os.path.join(RATINGS_DIR, 'bot_ratings_final.csv')
    tsumego_ratings_path = os.path.join(RATINGS_DIR, 'tsumego_ratings_final.csv')

    # --- Load Data ---
    bot_ratings_df = pd.read_csv(bot_ratings_path, header=0, names=['id', 'rating'])
    tsumego_ratings_df = pd.read_csv(tsumego_ratings_path, header=0, names=['id', 'rating'])
    bot_ratings_dict = dict(zip(bot_ratings_df['id'], bot_ratings_df['rating']))

    # --- Normalization Logic ---
    tsumego_only_map_df = tsumego_ratings_df.copy()
    tsumego_only_map_df['quantile_rank'] = tsumego_only_map_df['rating'].rank(method='average', pct=True)
    tsumego_only_map_df.sort_values('rating', inplace=True)

    original_ratings_for_map = tsumego_only_map_df['rating'].values
    quantile_ranks_for_map = tsumego_only_map_df['quantile_rank'].values
    tsumego_quantile_lookup = dict(zip(tsumego_only_map_df['id'], tsumego_only_map_df['quantile_rank']))

    # --- Define Anchors ---
    min_tsumego_id = tsumego_only_map_df.iloc[0]['id']
    max_tsumego_id = tsumego_only_map_df.iloc[-1]['id']

    anchors = {
        max_tsumego_id: 1.0,
        # "b18c384nbt-humanv0_4d": get_rank_number("4d"),
        # "b18c384nbt-humanv0_1k": get_rank_number("1k"),
        # "b18c384nbt-humanv0_5k": get_rank_number("5k"),
        # "b18c384nbt-humanv0_10k": get_rank_number("10k"),
        # "b18c384nbt-humanv0_15k": get_rank_number("15k"),
        min_tsumego_id: 0.0
    }

    rating_map = []
    # print("--- Anchor Transformation ---")
    for anchor_id, target_rank in anchors.items():
        if anchor_id in bot_ratings_dict:
            original_bot_rating = bot_ratings_dict[anchor_id]
            source_rank = np.interp(original_bot_rating, original_ratings_for_map, quantile_ranks_for_map)
        elif anchor_id in tsumego_quantile_lookup:
            source_rank = tsumego_quantile_lookup[anchor_id]
        else:
            raise Exception(f"Anchor ID {anchor_id} not found in bot or tsumego ratings.")
        rating_map.append((target_rank, source_rank))

    # --- Process Tsumego Ratings ---
    # Iterate directly over the tsumegos for which we have rating data.
    for tsumego_id, source_quantile_rank in tsumego_quantile_lookup.items():
        final_rank = intepolate(rating_map, source_quantile_rank)
        final_tsumego_ratings[tsumego_id] = final_rank

    return final_tsumego_ratings


def sample_tsumegos_with_links(sample_size=50):
    """
    Randomly samples tsumegos and prints their ratings with links for easier testing.
    
    Args:
        sample_size: Number of tsumegos to sample (default: 50)
    """
    # Load the ratings data
    bot_ratings_path = os.path.join(RATINGS_DIR, 'bot_ratings_final.csv')
    tsumego_ratings_path = os.path.join(RATINGS_DIR, 'tsumego_ratings_final.csv')
    
    if not os.path.exists(bot_ratings_path) or not os.path.exists(tsumego_ratings_path):
        print("Rating files not found. Please run the rating calculation first.")
        return
    
    # --- Load Data ---
    bot_ratings_df = pd.read_csv(bot_ratings_path, header=0, names=['id', 'rating'])
    tsumego_ratings_df = pd.read_csv(tsumego_ratings_path, header=0, names=['id', 'rating'])
    
    # Get all tsumego IDs
    all_tsumego_ids = tsumego_ratings_df['id'].tolist()
    
    # Sample without replacement (or use all if fewer than sample_size)
    sample_count = min(sample_size, len(all_tsumego_ids))
    sampled_ids = random.sample(all_tsumego_ids, sample_count)
    
    # Create a list of (tsumego_id, rating) pairs for sorting
    sampled_tsumegos = []
    for tsumego_id in sampled_ids:
        rating_row = tsumego_ratings_df[tsumego_ratings_df['id'] == tsumego_id]
        if not rating_row.empty:
            rating = rating_row.iloc[0]['rating']
            sampled_tsumegos.append((tsumego_id, rating))
    
    # Sort by rating (descending order - higher ratings first)
    sampled_tsumegos.sort(key=lambda x: x[1], reverse=True)
    
    # Print sampled tsumegos with ratings, closest bot, and links
    print(f"\n--- Randomly Sampled {sample_count} Tsumegos for Testing (Sorted by Rating) ---")
    for tsumego_id, tsumego_rating_val in sampled_tsumegos:
        # Find the bot with the closest rating
        closest_bot = None
        min_rating_diff = float('inf')
        
        for _, bot_row in bot_ratings_df.iterrows():
            bot_id = bot_row['id']
            bot_rating_val = bot_row['rating']
            rating_diff = abs(tsumego_rating_val - bot_rating_val)
            if rating_diff < min_rating_diff:
                min_rating_diff = rating_diff
                closest_bot = bot_id
        
        print(f"Rating: {tsumego_rating_val:.2f} (Closest bot: {closest_bot}) - http://localhost:8000/?id={tsumego_id}")