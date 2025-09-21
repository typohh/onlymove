import hashlib
import io
import json
import os
import zipfile
import pandas as pd

from katago_queries import get_all_ranks, get_rank_str


def create_packages(metadata, tsumegos_dir="tsumegos", output_dir="pwa"):
    """Sorts, groups, and packages tsumegos into zip files based on their rating."""

    metadata.sort(key=lambda x: x["rating"])
    group_count = int(len(metadata)**0.5)
    for i, item in enumerate(metadata):
        item["group"] = int(group_count * i / len(metadata))

    zip_dir = os.path.join(output_dir, "tsumegos")
    if os.path.exists(zip_dir):
        for f in os.listdir(zip_dir):
            if f.endswith(".zip"):
                os.remove(os.path.join(zip_dir, f))
    else:
        os.makedirs(zip_dir)

    zip_metadata = []
    lookup_entries = []  # To store tsumego_id to zip_id mappings
    for group in range(group_count):
        group_items = [item for item in metadata if item["group"] == group]
        if not group_items:
            continue

        min_rating = min(item["rating"] for item in group_items)
        max_rating = max(item["rating"] for item in group_items)

        zip_content = io.BytesIO()
        with zipfile.ZipFile(zip_content, 'w', zipfile.ZIP_DEFLATED) as temp_zip:
            # Add SGF files to zip
            for item in group_items:
                file_path = os.path.join(tsumegos_dir, item["filename"])
                with open(file_path, 'rb') as f:
                    temp_zip.writestr(item["filename"], f.read())

            # Add metadata.json to zip
            group_metadata_content = [{k: v for k, v in item.items() if k != "group"} for item in group_items]
            temp_zip.writestr("metadata.json", json.dumps(group_metadata_content, indent=2))

        zip_content.seek(0)
        zip_hash = hashlib.sha256(zip_content.getvalue()).hexdigest()[:8]
        zip_filename = f"{zip_hash}.zip"
        zip_path = os.path.join(zip_dir, zip_filename)

        with open(zip_path, 'wb') as f:
            f.write(zip_content.getvalue())

        zip_metadata.append({"filename": zip_filename, "min_rating": min_rating, "max_rating": max_rating})

        # Add entries to lookup for each tsumego in this group
        for item in group_items:
            tsumego_id = item["filename"].split('.')[0]
            lookup_entries.append((tsumego_id, zip_hash))

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(zip_metadata, f, indent=2)

    # Create lookup.txt file sorted by tsumego_id
    lookup_entries.sort(key=lambda x: x[0])  # Sort by tsumego_id
    lookup_path = os.path.join(output_dir, "lookup.txt")
    with open(lookup_path, 'w', encoding='utf-8') as f:
        for tsumego_id, zip_id in lookup_entries:
            f.write(f"{tsumego_id} {zip_id}\n")


def print_rating_distribution(metadata):
    """Prints the final distribution of problems by rating."""
    if not metadata:
        print("No metadata to generate distribution from.")
        return

    rating_distribution = sorted([item['rating'] for item in metadata])

    print("\n--- Final Rating Distribution ---")
    previous_rank = 0.0
    total_problems = len(rating_distribution)
    for rank in get_all_ranks():
        min_r = (rank + previous_rank) / 2
        max_r = rank + (rank - previous_rank) / 2

        count = sum(1 for x in rating_distribution if min_r <= x < max_r)
        percentage = (count / total_problems * 100) if total_problems > 0 else 0
        qualities_in_range = [item['quality'] for item in metadata if min_r <= item['rating'] < max_r]
        avg_quality = sum(qualities_in_range) / len(qualities_in_range) if qualities_in_range else 0.0
        raw_ratings_in_range = [item['raw_rating'] for item in metadata if min_r <= item['rating'] < max_r]
        avg_raw_rating = sum(raw_ratings_in_range) / len(raw_ratings_in_range) if raw_ratings_in_range else 0.0
        print(f"{get_rank_str(rank):>4s}: {count:>5d} problems ({percentage:5.1f}%, avg quality: {avg_quality:.3f}, avg rating: {avg_raw_rating:.3f})")
        previous_rank = rank


if __name__ == "__main__":

    TSUMEGOS_DIR = "tsumegos"
    OUTPUT_DIR = "pwa"

    # 1. Get the ratings map (tsumego_id -> rating).
    tsumego_ratings_df = pd.read_csv("ratings/tsumego_ratings_final.csv")

    # Store raw ratings before scaling
    tsumego_ratings_df['raw_rating'] = tsumego_ratings_df['rating']

    # Apply order..
    tsumego_ratings_df = tsumego_ratings_df.sort_values(by='rating', ascending=True).reset_index(drop=True)

    total = len(tsumego_ratings_df)
    # Read bot ratings
    bot_ratings_df = pd.read_csv("ratings/bot_ratings_final.csv")
    bot_8k = bot_ratings_df[bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_8k']['rating'].values[0]
    bot_3d = bot_ratings_df[bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_3d']['rating'].values[0]
    rank_adjustment = + 1.5
    f1 = ( 12.5 + rank_adjustment ) / 29
    f2 = ( 22.5 + rank_adjustment ) / 29
    a = (f2 - f1) / (bot_3d - bot_8k)
    b = f1 - a * bot_8k
    tsumego_ratings_df['rating'] = a * tsumego_ratings_df['raw_rating'] + b
    bottom = tsumego_ratings_df[tsumego_ratings_df['rating'] < 1/29].index
    for i, idx in enumerate(sorted(bottom, key=lambda x: tsumego_ratings_df.loc[x, 'raw_rating'])):
        tsumego_ratings_df.loc[idx, 'rating'] = i / (len(bottom) - 1) * (1/29) if len(bottom) > 1 else 0
    top = tsumego_ratings_df[tsumego_ratings_df['rating'] > 28/29].index
    for i, idx in enumerate(sorted(top, key=lambda x: tsumego_ratings_df.loc[x, 'raw_rating'])):
        tsumego_ratings_df.loc[idx, 'rating'] = (i / (len(top) - 1)) * (1 - 28/29) + 28/29 if len(top) > 1 else 28/29


    tsumego_ratings_map = dict(zip(tsumego_ratings_df['tsumego_id'], zip(tsumego_ratings_df['rating'], tsumego_ratings_df['raw_rating'])))

    # 2. Get the quality map (tsumego_id -> quality).
    tsumego_quality_df = pd.read_csv("ratings/tsumego_quality_final.csv")
    # Normalize qualities to 0-1
    min_quality = tsumego_quality_df['quality'].min()
    max_quality = tsumego_quality_df['quality'].max()
    tsumego_quality_df['quality'] = (tsumego_quality_df['quality'] - min_quality) / (max_quality - min_quality)
    # tsumego_quality_df['quality'] = 1 - tsumego_quality_df['quality'] # test out how low quality tsumegos compare
    tsumego_quality_map = dict(zip(tsumego_quality_df['tsumego_id'], tsumego_quality_df['quality']))

    # 3. Build the final metadata list by matching files with their ratings and quality.
    print(f"Found {len(tsumego_ratings_map)} rating values in ratings/tsumego_ratings_final.csv" )
    print(f"Found {len(tsumego_ratings_map)} quality values in ratings/tsumego_quality_final.csv" )
    all_metadata = []
    sgf_files = [f for f in os.listdir(TSUMEGOS_DIR) if f.endswith(".sgf")]
    print(f"Found {len(sgf_files)} SGF files in '{TSUMEGOS_DIR}'.")
    skipped_count = 0
    for filename in sgf_files:
        tsumego_id = filename.split('.')[0]
        if tsumego_id in tsumego_ratings_map:
            rating, raw_rating = tsumego_ratings_map[tsumego_id]
            quality = tsumego_quality_map.get(tsumego_id, 0.0)
            all_metadata.append({"filename": filename, "rating": rating, "raw_rating": raw_rating, "quality": quality})
        else:
            skipped_count += 1

    print(f"\nSuccessfully created metadata for {len(all_metadata)} tsumegos.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} SGF files with no matching rating.")

    # 3. Create the zip packages and main metadata file.
    print("\n--- Creating Tsumego Packages ---")
    create_packages(all_metadata, TSUMEGOS_DIR, OUTPUT_DIR)
    print(f"Created packages in '{os.path.join(OUTPUT_DIR, 'tsumegos')}'")
    print(f"Created main metadata file at '{os.path.join(OUTPUT_DIR, 'metadata.json')}'")

    print("\nProcessing complete.")

    # 4. Print the final distribution report.
    print_rating_distribution(all_metadata)