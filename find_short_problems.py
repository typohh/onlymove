
import os
import sys
import math
import pandas as pd

# Add sgfmill to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'docs'))

import sgfmill.sgf

TSUMEGO_DIR = "tsumegos"


def get_main_variation_move_count(game: sgfmill.sgf.Sgf_game):
    """
    Counts the number of moves in the main variation of a tsumego SGF file,
    including setup stones (AB/AW) and moves along the single-child path
    until a branch is encountered.
    """
    move_count = 0
    root_node = game.get_root()

    # Count setup stones
    if root_node.has_property("AB"):
        move_count += len(root_node.get("AB"))
    if root_node.has_property("AW"):
        move_count += len(root_node.get("AW"))

    current_node = root_node
    # Traverse the main variation (single-child nodes)
    while len(current_node) == 1:
        if current_node.has_property("B") or current_node.has_property("W"):
            move_count += 1
        current_node = current_node[0]

    # Count the move in the node that branches, if it exists
    if current_node.has_property("B") or current_node.has_property("W"):
        move_count += 1

    return move_count


def get_descriptive_rank(numeric_rank):
    """
    Converts a numeric rank (0-1) to a descriptive rank (e.g., '10 kyu', '3 dan').
    Logic is ported from pwa/js/utils.js.
    """
    if numeric_rank < 0: numeric_rank = 0
    if numeric_rank > 1: numeric_rank = 1

    total_ranks = 29
    rank_index = math.floor(numeric_rank * total_ranks)

    if rank_index > 29:
        rank_index = 29

    if rank_index < 20:
        return f"{20 - rank_index} kyu"
    else:
        # This can result in "10 dan" if rank_index is 29, consistent with the JS code.
        return f"{rank_index - 19} dan"


def load_rating_data():
    """
    Loads and processes tsumego ratings and quality from CSV files,
    returning maps from tsumego_id to rank and quality.
    The rank calculation is ported from prepare_tsumegos.py.
    """
    # Load and process ratings
    tsumego_ratings_df = pd.read_csv("ratings/tsumego_ratings_final.csv")
    tsumego_ratings_df = tsumego_ratings_df.sort_values(by='rating', ascending=True).reset_index(drop=True)

    total = len(tsumego_ratings_df)
    bottom_1_count = int(0.01 * total)
    middle_end = int(0.90 * total)
    top_count = total - middle_end

    if bottom_1_count > 1:
        tsumego_ratings_df.loc[:bottom_1_count - 1, 'rating'] = (tsumego_ratings_df.index[:bottom_1_count] / (
                    bottom_1_count - 1)) * (1 / 29)
    elif bottom_1_count > 0:
        tsumego_ratings_df.loc[:bottom_1_count - 1, 'rating'] = 0.0

    min_middle = tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'].min()
    max_middle = tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'].max()
    if max_middle > min_middle:
        tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'] = (tsumego_ratings_df.loc[
                                                                    bottom_1_count:middle_end - 1, 'rating'] - min_middle) / (
                                                                                       max_middle - min_middle) * (
                                                                                       28 / 29 - 1 / 29) + 1 / 29
    else:
        tsumego_ratings_df.loc[bottom_1_count:middle_end - 1, 'rating'] = 1 / 29

    if top_count > 1:
        tsumego_ratings_df.loc[middle_end:, 'rating'] = ((tsumego_ratings_df.index[middle_end:] - middle_end) / (
                    top_count - 1)) * (1 - 28 / 29) + 28 / 29
    elif top_count > 0:
        tsumego_ratings_df.loc[middle_end:, 'rating'] = 28 / 29

    tsumego_ratings_map = dict(zip(tsumego_ratings_df['tsumego_id'], tsumego_ratings_df['rating']))

    # Load and process quality
    tsumego_quality_df = pd.read_csv("ratings/tsumego_quality_final.csv")
    min_quality = tsumego_quality_df['quality'].min()
    max_quality = tsumego_quality_df['quality'].max()
    if max_quality > min_quality:
        tsumego_quality_df['quality'] = (tsumego_quality_df['quality'] - min_quality) / (max_quality - min_quality)
    else:
        tsumego_quality_df['quality'] = 0.5  # Default value if all qualities are the same

    tsumego_quality_map = dict(zip(tsumego_quality_df['tsumego_id'], tsumego_quality_df['quality']))

    return tsumego_ratings_map, tsumego_quality_map


def find_and_print_short_problems():
    """
    Goes through all SGF files, finds short problems (<= 6 moves),
    and prints their rank, quality, and a link.
    """
    if not os.path.isdir(TSUMEGO_DIR):
        print(f"Error: Directory '{TSUMEGO_DIR}' not found.", file=sys.stderr)
        return

    try:
        tsumego_ratings_map, tsumego_quality_map = load_rating_data()
    except FileNotFoundError as e:
        print(f"Error loading rating/quality data: {e}", file=sys.stderr)
        return

    for filename in sorted(os.listdir(TSUMEGO_DIR)):
        if not filename.endswith(".sgf"):
            continue

        file_path = os.path.join(TSUMEGO_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            game = sgfmill.sgf.Sgf_game.from_string(content)
            move_count = get_main_variation_move_count(game)

            if move_count <= 6:
                tsumego_id = filename[:-4]
                numeric_rank = tsumego_ratings_map.get(tsumego_id)
                quality = tsumego_quality_map.get(tsumego_id)

                if numeric_rank is not None and quality is not None:
                    descriptive_rank = get_descriptive_rank(numeric_rank)
                    print(
                        f"{descriptive_rank}, quality: {quality:.2f} - https://onlymove.net/?id={tsumego_id}")
                else:
                    print(f"Rank/quality not found for {tsumego_id} - https://onlymove.net/?id={tsumego_id}")

        except Exception as e:
            print(f"Error processing {filename}: {e}", file=sys.stderr)


if __name__ == "__main__":
    find_and_print_short_problems()
