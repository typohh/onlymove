import hashlib
import io
import json
import os
import random
import re

from katago_queries import get_rank_number, get_rank_str, get_all_ranks


def fix_tsumegos(tsumegos_dir):
    sgf_files = [f for f in os.listdir(tsumegos_dir) if f.endswith('.sgf')]
    for filename in sgf_files:
        filepath = os.path.join(tsumegos_dir, filename)
        with open(filepath, 'rb') as f:
            file_bytes = f.read()
            new_filename = hashlib.sha256(file_bytes).hexdigest()[:8] + ".sgf"
            os.rename(filepath, os.path.join(tsumegos_dir, new_filename))


def get_heuristic_rank(content):
    """Find difference between two highest scores in content."""
    pattern = r'Score: (-?\d+\.?\d*)'
    matches = re.findall(pattern, content)
    scores = sorted([float(score) for score in matches], reverse=True)
    moves = len(set([round(score) for score in scores])) - 2
    avg_moves = 7
    return moves ** avg_moves / (moves ** avg_moves + avg_moves ** avg_moves)


# 7 : heuristic_    0.165    0.479    1.000


def extract_difficulty(content):
    """Extract move and game ranks from the GN tag and return a dict with components and average."""
    pattern = r"GN\[Move: (\d+[kd]), Game: (\d+[kd])\]"
    match = re.search(pattern, content)

    if not match:
        raise Exception(f"Error: Could not find difficulty in {content}")

    move_rank = get_rank_number(match.group(1))
    game_rank = get_rank_number(match.group(2))
    heuristic_rank = get_heuristic_rank(content)

    average_rank = (move_rank + game_rank + heuristic_rank) / 3

    return {
        "rank": average_rank,
        "move_rank": move_rank,
        "game_rank": game_rank,
        "heuristic_rank": heuristic_rank
    }


def intepolate(points, rating):
    # Find points above and below the rating
    above = None
    below = None
    for new_val, old_val in points:
        if old_val >= rating:
            above = (new_val, old_val)
        if old_val <= rating and below is None:
            below = (new_val, old_val)
        if above is not None and below is not None:
            break

    if above is None:
        return 1
    if below is None:
        return 0

    # If points are the same, return either value
    if above[1] == below[1]:
        return above[0]

    # Linear interpolation
    t = (rating - below[1]) / (above[1] - below[1])
    return below[0] + t * (above[0] - below[0])


if __name__ == "__main__":

    tsumegos_dir = "tsumegos"
    if not os.path.exists(tsumegos_dir):
        print(f"Error: {tsumegos_dir} directory not found")
        exit(1)

    metadata = []
    rank_distribution = []
    rank_components_data = {}  # To store data for correlation analysis

    # Load tsumego ratings
    import pandas as pd

    # Load and normalize bot ratings
    bot_ratings_df = pd.read_csv('ratings/bot_ratings_final.csv', header=0, names=['id', 'rating'])
    bot_ratings_dict = dict(zip(bot_ratings_df['id'], bot_ratings_df['rating']))

    max_rating = max(bot_ratings_dict.items(), key=lambda x: x[1])[0]
    min_rating = min(bot_ratings_dict.items(), key=lambda x: x[1])[0]

    anchors = {
        max_rating: 1.0,
        "katago_9958_10": get_rank_number("8d"),
        "katago_corrupt_28_1": get_rank_number("4d"),
        "katago_corrupt_50_1": get_rank_number("19k"),
        min_rating: 0.0
    }

    print(bot_ratings_dict)

    print(anchors)

    anchors_old = {key: bot_ratings_dict[key] for key in anchors.keys()}

    rating_map = list(zip(anchors.values(), anchors_old.values()))

    normalized_bot_ratings = {}
    for bot, rating in bot_ratings_dict.items():
        normalized_bot_ratings[bot] = intepolate(rating_map, rating)

    print("\nNormalized bot ratings:")
    for bot, rating in normalized_bot_ratings.items():
        print(f"{bot}: {bot_ratings_dict[bot]:.3f} -> {rating:.3f} {get_rank_str(rating)}")

    ratings_df = pd.read_csv('ratings/tsumego_ratings_final.csv', header=0, names=['id', 'rating'])
    ratings_dict = dict(zip(ratings_df['id'], ratings_df['rating']))

    for filename in os.listdir(tsumegos_dir):
        if filename.endswith(".sgf"):
            file_path = os.path.join(tsumegos_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            rank_data = {}

            # Add true rank from ratings file
            tsumego_id = filename.split('.')[0]
            if ratings_dict.get(tsumego_id) is None:
                print(f"Warning: No rating found for {tsumego_id}")
                continue

            rank_data["rank"] = intepolate(rating_map, ratings_dict.get(tsumego_id))

            if rank_data is None or "rank" not in rank_data:
                raise Exception(f"Warning: Could not determine difficulty for {filename}")

            # spread out extremes for about half a rank, so we can order them into zips properly
            if rank_data["rank"] == 1.0:
                rank_data["rank"] -= random.Random(tsumego_id).uniform(0, 1 / 60)
            if rank_data["rank"] == 0.0:
                rank_data["rank"] += random.Random(tsumego_id).uniform(0, 1 / 60)

            # Add randomness to the main rank
            # rank_data["rank"] = rank_data["rank"] * (29 / 30) + (1 / 30) * random.random()

            # Add filename to the rank data
            rank_data["filename"] = filename

            metadata.append(rank_data)
            rank_distribution.append(rank_data["rank"])

            # Collect data for correlation analysis (excluding "rank" and "filename")
            for key, value in rank_data.items():
                if key not in ["filename"]:
                    if key not in rank_components_data:
                        rank_components_data[key] = []
                    rank_components_data[key].append(value)

            # if len(metadata)>5000:
            #     break

    # Sort metadata by rank and assign groups
    metadata.sort(key=lambda x: x["rank"])

    # spread out evenly the ranks
    # for i, item in enumerate(metadata):
    #     item["rank"] = i / (len(metadata) - 1)

    group_count = 150

    for i, item in enumerate(metadata):
        item["group"] = int(group_count * i / len(metadata))

    # Create zip files for each group with embedded metadata
    import zipfile

    # Delete any pre-existing zip files in the directory
    zip_dir = os.path.join("pwa", "tsumegos")
    if os.path.exists(zip_dir):
        for filename in os.listdir(zip_dir):
            if filename.endswith(".zip"):
                os.remove(os.path.join(zip_dir, filename))

    zip_metadata = []  # For the main metadata file

    for group in range(group_count):
        group_items = [item for item in metadata if item["group"] == group]
        group_files = [item["filename"] for item in group_items]

        # Calculate min and max rank for this group
        if group_items:
            min_rank = min(item["rank"] for item in group_items)
            max_rank = max(item["rank"] for item in group_items)
        else:
            min_rank = max_rank = 0

        # Create zip content and calculate hash
        zip_content = io.BytesIO()
        with zipfile.ZipFile(zip_content, 'w', zipfile.ZIP_DEFLATED) as temp_zip:
            # Add tsumego files
            for filename in group_files:
                file_path = os.path.join(tsumegos_dir, filename)
                with open(file_path, 'rb') as f:
                    temp_zip.writestr(filename, f.read())

            # Create and add metadata.json for this zip (excluding group field)
            group_metadata = []
            for item in group_items:
                # Copy item but exclude the "group" field
                item_copy = {k: v for k, v in item.items() if k != "group"}
                group_metadata.append(item_copy)
            temp_zip.writestr("metadata.json", json.dumps(group_metadata, indent=2))

        # Calculate hash of zip content
        zip_content.seek(0)
        zip_hash = hashlib.sha256(zip_content.getvalue()).hexdigest()[:16]
        zip_filename = f"{zip_hash}.zip"
        zip_path = os.path.join("pwa", "tsumegos", zip_filename)

        # Write the actual zip file
        with open(zip_path, 'wb') as f:
            f.write(zip_content.getvalue())

        # Add to main metadata
        zip_metadata.append({
            "filename": zip_filename,
            "min_rank": min_rank,
            "max_rank": max_rank
        })

    # Write main metadata file with zip information
    metadata_path = os.path.join("pwa", "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(zip_metadata, f, indent=2)

    # Print rank distribution sorted by rank_num
    rank_distribution = sorted(rank_distribution)
    print("\nRank Distribution:")
    previous_rank = 0
    for rank in get_all_ranks():
        min_rank = (rank + previous_rank) / 2
        max_rank = rank + rank - previous_rank
        # print aproximately how many of rank_distrubitn is between min_rank and max_rank
        count = sum(1 for x in rank_distribution if min_rank <= x <= max_rank)
        print(f"{get_rank_str(rank)}: {count} problems")
        previous_rank = rank
