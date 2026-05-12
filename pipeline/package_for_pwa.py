import hashlib
import io
import json
import os
import sys
import zipfile
import pandas as pd

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np
from utils.configurations import BOARD_SIZE
from utils.katago_queries import get_all_ranks, get_rank_str


def create_packages(metadata, board_size, tsumegos_dir="tsumegos", output_dir="pwa"):
    """Sorts, groups, and packages tsumegos into zip files based on their rating.

    Output layout:
        {output_dir}/tsumegos/{board_size}/{hash}.zip
        {output_dir}/tsumegos/{board_size}/metadata.json
        {output_dir}/lookup.txt   (merged across sizes; rows: "id zipHash size")
    """

    metadata.sort(key=lambda x: x["rating"])
    group_count = int(len(metadata)**0.5)
    for i, item in enumerate(metadata):
        item["group"] = int(group_count * i / len(metadata))

    zip_dir = os.path.join(output_dir, "tsumegos", str(board_size))
    if os.path.exists(zip_dir):
        for f in os.listdir(zip_dir):
            if f.endswith(".zip") or f == "metadata.json":
                os.remove(os.path.join(zip_dir, f))
    else:
        os.makedirs(zip_dir)

    zip_metadata = []
    new_lookup_entries = []  # entries for the size we are (re)building now
    for group in range(group_count):
        group_items = [item for item in metadata if item["group"] == group]
        if not group_items:
            continue

        min_rating = min(item["rating"] for item in group_items)
        max_rating = max(item["rating"] for item in group_items)

        zip_content = io.BytesIO()
        with zipfile.ZipFile(zip_content, 'w', zipfile.ZIP_DEFLATED) as temp_zip:
            for item in group_items:
                file_path = os.path.join(tsumegos_dir, item["filename"])
                with open(file_path, 'rb') as f:
                    temp_zip.writestr(item["filename"], f.read())

            group_metadata_content = [{k: v for k, v in item.items() if k != "group"} for item in group_items]
            temp_zip.writestr("metadata.json", json.dumps(group_metadata_content, indent=2))

        zip_content.seek(0)
        zip_hash = hashlib.sha256(zip_content.getvalue()).hexdigest()[:8]
        zip_filename = f"{zip_hash}.zip"
        zip_path = os.path.join(zip_dir, zip_filename)

        with open(zip_path, 'wb') as f:
            f.write(zip_content.getvalue())

        zip_metadata.append({"filename": zip_filename, "min_rating": min_rating, "max_rating": max_rating})

        for item in group_items:
            tsumego_id = item["filename"].split('.')[0]
            new_lookup_entries.append((tsumego_id, zip_hash, board_size))

    # Per-size metadata.json sits inside the size subdirectory.
    metadata_path = os.path.join(zip_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(zip_metadata, f, indent=2)

    # Stale root metadata.json (legacy single-size layout) is no longer used.
    legacy_root_metadata = os.path.join(output_dir, "metadata.json")
    if os.path.exists(legacy_root_metadata):
        os.remove(legacy_root_metadata)

    # Merge new entries into the global lookup, replacing any rows for this size.
    lookup_path = os.path.join(output_dir, "lookup.txt")
    merged = []
    if os.path.exists(lookup_path):
        with open(lookup_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    tsumego_id, zip_id, size_str = parts
                    try:
                        size_int = int(size_str)
                    except ValueError:
                        continue
                    if size_int != board_size:
                        merged.append((tsumego_id, zip_id, size_int))
                # Rows shorter than 3 cols are pre-multi-size legacy data and
                # are dropped — they belong to the size we are rebuilding now
                # (only one size existed before this change).

    merged.extend(new_lookup_entries)
    merged.sort(key=lambda x: x[0])
    with open(lookup_path, 'w', encoding='utf-8') as f:
        for tsumego_id, zip_id, size_int in merged:
            f.write(f"{tsumego_id} {zip_id} {size_int}\n")


def print_rating_distribution(metadata):
    """Prints the final distribution of problems by rating."""
    if not metadata:
        print("No metadata to generate distribution from.")
        return

    rating_distribution = sorted([item['rating'] for item in metadata])

    print("\n--- Final Rating Distribution ---")
    print("\n--- Final Rating Distribution ---")
    
    total_problems = len(rating_distribution)
    num_ranks = len(get_all_ranks())
    
    for i, rank in enumerate(get_all_ranks()):
        min_r = i / num_ranks
        max_r = (i + 1) / num_ranks
        
        # Handle floating point issues at boundaries slightly? 
        # Actually standard half-open [min, max) is fine, except for the very last one which should include 1.0
        if i == num_ranks - 1:
             count = sum(1 for x in rating_distribution if min_r <= x <= max_r + 1e-9)
             qualities_in_range = [item['quality'] for item in metadata if min_r <= item['rating'] <= max_r + 1e-9]
             raw_ratings_in_range = [item['raw_rating'] for item in metadata if min_r <= item['rating'] <= max_r + 1e-9]
        else:
             count = sum(1 for x in rating_distribution if min_r <= x < max_r)
             qualities_in_range = [item['quality'] for item in metadata if min_r <= item['rating'] < max_r]
             raw_ratings_in_range = [item['raw_rating'] for item in metadata if min_r <= item['rating'] < max_r]

        percentage = (count / total_problems * 100) if total_problems > 0 else 0
        avg_quality = sum(qualities_in_range) / len(qualities_in_range) if qualities_in_range else 0.0
        avg_raw_rating = sum(raw_ratings_in_range) / len(raw_ratings_in_range) if raw_ratings_in_range else 0.0
        print(f"{get_rank_str(rank):>4s}: {count:>5d} problems ({percentage:5.1f}%, avg quality: {avg_quality:.3f}, avg rating: {avg_raw_rating:.3f})")


def load_embeddings(board_size):
    """Load per-problem PCA scores filtered to this board size, plus the
    shared embedding metadata (model is fit jointly across sizes).

    Returns (embedding_map, embedding_meta) where:
      embedding_map: tsumego_id -> list of K floats (or None if not available)
      embedding_meta: dict with K, embedding_version, labels.
                     The PWA only reads version (for cache invalidation) and
                     labels (for stats display); fields like explained-variance
                     and extreme_problems are kept locally in embeddings/joint/
                     as inspection aids and not shipped.
    """
    joint_dir = os.path.join(PROJECT_ROOT, "embeddings", "joint")
    legacy_dir = os.path.join(PROJECT_ROOT, "embeddings", str(board_size))

    # Prefer the joint model. Fall back to per-size for backward compat
    # if the joint pipeline hasn't been run yet.
    if os.path.exists(os.path.join(joint_dir, "per_problem.csv")):
        embed_dir = joint_dir
        is_joint = True
    elif os.path.exists(os.path.join(legacy_dir, "per_problem.csv")):
        embed_dir = legacy_dir
        is_joint = False
    else:
        print(f"  no embeddings at {joint_dir} or {legacy_dir}; skipping embedding fields")
        return None, None

    print(f"  reading embeddings from {embed_dir}")
    df = pd.read_csv(os.path.join(embed_dir, "per_problem.csv"))
    if is_joint and "size" in df.columns:
        df = df[df["size"] == board_size]
    embed_cols = sorted(
        [c for c in df.columns if c.startswith("embed_")],
        key=lambda c: int(c.split("_")[1]),
    )
    K = len(embed_cols)
    embedding_map = {}
    for _, row in df.iterrows():
        embedding_map[row["tsumego_id"]] = [float(row[c]) for c in embed_cols]

    labels_path = os.path.join(embed_dir, "labels.json")
    pca_path = os.path.join(embed_dir, "pca.npz")
    labels = None
    embedding_version = None
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    # The version travels with the model state; the PWA reads it from
    # embedding_meta.json to auto-invalidate localStorage on refits.
    if os.path.exists(pca_path):
        try:
            arc = np.load(pca_path, allow_pickle=False)
            if "embedding_version" in arc.files:
                embedding_version = str(arc["embedding_version"].item())
        except Exception as e:
            print(f"  couldn't read embedding_version from pca.npz: {e}")

    embedding_meta = {
        "K": K,
        "embedding_version": embedding_version,
        "labels": labels,
    }
    return embedding_map, embedding_meta


if __name__ == "__main__":

    TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "tsumegos", str(BOARD_SIZE))
    RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings", str(BOARD_SIZE))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pwa")

    # 1. Get the ratings map (tsumego_id -> rating).
    tsumego_ratings_df = pd.read_csv(os.path.join(RATINGS_DIR, "tsumego_ratings_final.csv"))

    # Store raw ratings before scaling
    tsumego_ratings_df['raw_rating'] = tsumego_ratings_df['rating']

    # Apply order..
    tsumego_ratings_df = tsumego_ratings_df.sort_values(by='rating', ascending=True).reset_index(drop=True)

    total = len(tsumego_ratings_df)
    # Read bot ratings
    bot_ratings_df = pd.read_csv(os.path.join(RATINGS_DIR, "bot_ratings_final.csv"))
    # --- Exponential Mapping Implementation ---
    # 1. Get Anchor Ratings
    # 8k is index 12 in 0-based 20k-9d scale. Center of bin is 12.5.
    # 3k is index 17 in 0-based 20k-9d scale. Center of bin is 17.5.
    bot_8k_rating = bot_ratings_df[bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_8k']['rating'].values[0]
    bot_3k_rating = bot_ratings_df[bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_3k']['rating'].values[0]
    
    x1 = 12.5  # 8k index
    x2 = 17.5  # 3k index
    y1 = bot_8k_rating
    y2 = bot_3k_rating
    k = 0.12   # Variance Decay Constant

    # 2. Derive Constants A and C
    # System:
    # y1 = A * e^(k * x1) + C
    # y2 = A * e^(k * x2) + C
    # y2 - y1 = A * (e^(k * x2) - e^(k * x1))
    
    A = (y2 - y1) / (np.exp(k * x2) - np.exp(k * x1))
    C = y1 - A * np.exp(k * x1)

    print(f"Exponential Mapping Constants: A={A:.6f}, C={C:.6f}")

    # 3. Apply Inverse Function: Rating -> Rank Index (x)
    # y = A * e^(k * x) + C  =>  (y - C) / A = e^(k * x)  =>  ln((y - C) / A) / k = x
    def rating_to_rank_index(r):
        # Safety check for log domain
        val = (r - C) / A
        if val <= 0:
            return 0.0 # Should be covered by clamping, but safe guard against math errors
        return np.log(val) / k

    # Vectorized application
    # Note: If A is negative (which it shouldn't be for standard rating scales where rating increases with skill), 
    # we need to be careful. In this system, rating increases with skill, so y2 > y1 and exp(kx2) > exp(kx1), so A > 0.
    
    # Calculate x (rank index)
    # We use a lambda but we need to handle potential domain errors if ratings are extremely low.
    # However, since we clamp afterwards, we can just clip the input to log.
    
    tsumego_ratings_df['rank_index'] = (tsumego_ratings_df['rating'] - C) / A
    # Clip to small positive to avoid log(<=0)
    tsumego_ratings_df['rank_index'] = tsumego_ratings_df['rank_index'].clip(lower=1e-9)
    tsumego_ratings_df['rank_index'] = np.log(tsumego_ratings_df['rank_index']) / k

    # 4. Normalize to [0, 1] PWA scale
    # PWA scale is 0..1 representing roughly 20k to 9d (indices 0 to 28)
    tsumego_ratings_df['pwa_rating'] = tsumego_ratings_df['rank_index'] / 29.0
    
    # 5. Clamp to valid range [0, 1]
    tsumego_ratings_df['rating'] = tsumego_ratings_df['pwa_rating'].clip(0.0, 1.0)
    
    # Drop temp columns
    tsumego_ratings_df.drop(columns=['rank_index', 'pwa_rating'], inplace=True)



    tsumego_ratings_map = dict(zip(tsumego_ratings_df['tsumego_id'], zip(tsumego_ratings_df['rating'], tsumego_ratings_df['raw_rating'])))

    # 2. Get the quality map (tsumego_id -> quality).
    tsumego_quality_df = pd.read_csv(os.path.join(RATINGS_DIR, "tsumego_quality_final.csv"))
    # Normalize qualities to 0-1
    min_quality = tsumego_quality_df['quality'].min()
    max_quality = tsumego_quality_df['quality'].max()
    tsumego_quality_df['quality'] = (tsumego_quality_df['quality'] - min_quality) / (max_quality - min_quality)
    # tsumego_quality_df['quality'] = 1 - tsumego_quality_df['quality'] # test out how low quality tsumegos compare
    tsumego_quality_map = dict(zip(tsumego_quality_df['tsumego_id'], tsumego_quality_df['quality']))

    # 3. Build the final metadata list by matching files with their ratings and quality.
    print(f"Found {len(tsumego_ratings_map)} rating values in {os.path.join(RATINGS_DIR, 'tsumego_ratings_final.csv')}")
    print(f"Found {len(tsumego_quality_map)} quality values in {os.path.join(RATINGS_DIR, 'tsumego_quality_final.csv')}")
    print("\n--- Loading embeddings (if available) ---")
    embedding_map, embedding_meta = load_embeddings(BOARD_SIZE)

    all_metadata = []
    sgf_files = [f for f in os.listdir(TSUMEGOS_DIR) if f.endswith(".sgf")]
    print(f"Found {len(sgf_files)} SGF files in '{TSUMEGOS_DIR}'.")
    skipped_count = 0
    embedding_missing = 0
    for filename in sgf_files:
        tsumego_id = filename.split('.')[0]
        if tsumego_id in tsumego_ratings_map:
            rating, raw_rating = tsumego_ratings_map[tsumego_id]
            quality = tsumego_quality_map.get(tsumego_id, 0.0)
            item = {"filename": filename, "rating": rating, "raw_rating": raw_rating, "quality": quality}
            if embedding_map is not None:
                emb = embedding_map.get(tsumego_id)
                if emb is None:
                    embedding_missing += 1
                else:
                    item["embedding"] = emb
            all_metadata.append(item)
        else:
            skipped_count += 1

    print(f"\nSuccessfully created metadata for {len(all_metadata)} tsumegos.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} SGF files with no matching rating.")
    if embedding_map is not None:
        n_with = sum(1 for m in all_metadata if "embedding" in m)
        print(f"Embeddings: {n_with}/{len(all_metadata)} entries have one "
              f"(missing for {embedding_missing}).")

    # 3. Create the zip packages and main metadata file.
    print("\n--- Creating Tsumego Packages ---")
    create_packages(all_metadata, BOARD_SIZE, TSUMEGOS_DIR, OUTPUT_DIR)
    size_dir = os.path.join(OUTPUT_DIR, 'tsumegos', str(BOARD_SIZE))
    print(f"Created packages in '{size_dir}'")
    print(f"Created per-size metadata file at '{os.path.join(size_dir, 'metadata.json')}'")
    print(f"Updated global lookup at '{os.path.join(OUTPUT_DIR, 'lookup.txt')}'")

    if embedding_meta is not None:
        # embedding_meta is joint across board sizes (PCA + ICA fit jointly),
        # so it lives at pwa/tsumegos/embedding_meta.json rather than under
        # each size's directory. Re-running for a second size overwrites with
        # identical content, which is fine.
        embedding_meta_path = os.path.join(OUTPUT_DIR, 'tsumegos', 'embedding_meta.json')
        with open(embedding_meta_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_meta, f, indent=2)
        print(f"Created joint embedding metadata at '{embedding_meta_path}' "
              f"(K={embedding_meta['K']})")

    print("\nProcessing complete.")

    # 4. Print the final distribution report.
    print_rating_distribution(all_metadata)