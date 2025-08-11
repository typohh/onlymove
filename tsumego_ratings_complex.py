import os

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
        "kata1-b6c96-s175395328-d26788732_corrupt3_1": get_rank_number("4d"),
        "kata1-b6c96-s175395328-d26788732_corrupt5_1": get_rank_number("19k"),
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