import hashlib
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import sgfmill.sgf

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES
from utils.katago_queries import get_all_ranks, get_rank_str
from pipeline.benchmark_bots import parse_tsumego_sgf, sgf_to_gtp


# Per-cell embedding sidecar parameters. Keep these in sync with whatever the
# PWA-side decoder expects in pwa/tsumegos/embedding_meta.json.
EMBEDDING_DTYPE = np.float16
EMBEDDING_DTYPE_NAME = "float16"
EMBEDDING_BYTES_PER_FLOAT = 2

GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # I-skipped, matches KataGo / GTP convention


def gtp_to_xy(gtp_coord):
    if gtp_coord.lower() == "pass":
        return None
    col_letter = gtp_coord[0].upper()
    x = GTP_COLS.index(col_letter)
    y = int(gtp_coord[1:]) - 1
    return x, y


def project_cells_for_problem(per_cell_path, sgf_path):
    """Load pre-projected per-cell embeddings for one tsumego and reorder to
    SGF variation document order. Returns (fp16 (n_cells, K) array, n_cells),
    or (None, 0) on missing/corrupt input.

    Reads embeddings/joint/per_cell/<size>/<tid>.npz, written upstream by
    build_embedding.write_per_cell_projections. The file already contains the
    K-dim projections in extraction order — packaging just reorders rows by
    the (x, y) → variation_index map. No PCA matmul, no raw 6912-dim
    activation read; ~7.5 KB read per problem at K=64, n_cells≈60.

    The PWA does a parallel walk over the SGF's root variations to compute a
    picked-move's cell index; layouts must agree, so reordering is keyed off
    the (x, y) → row map saved alongside the projected embeddings.
    """
    if not os.path.exists(per_cell_path):
        return None, 0

    try:
        with np.load(per_cell_path) as data:
            embeddings = np.asarray(data["embeddings"])  # (n_cells, K) fp16
            xys = np.asarray(data["xys"])                # (n_cells, 2)
    except Exception as e:
        print(f"  [warn] corrupt per_cell {per_cell_path}: {type(e).__name__}; "
              f"skipping per-cell sidecar for this problem", flush=True)
        return None, 0

    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    _, variation_nodes = parse_tsumego_sgf(game)

    xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}

    ordered_rows = []
    for node in variation_nodes:
        _, mb = node.get_raw_move()
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is None:
            continue  # pass: not embedded
        row = xy_to_row.get(xy)
        if row is None:
            continue  # rare: SGF has a cell the extraction didn't
        ordered_rows.append(embeddings[row])

    if not ordered_rows:
        return None, 0

    return np.stack(ordered_rows, axis=0).astype(EMBEDDING_DTYPE), len(ordered_rows)


def create_packages(metadata, board_size, tsumegos_dir="tsumegos", output_dir="pwa",
                    per_cell_dir=None):
    """Sorts, groups, and packages tsumegos into zip files based on rating.

    Output layout:
        {output_dir}/tsumegos/{board_size}/{hash}.zip
        {output_dir}/tsumegos/{board_size}/{hash}.embeddings.bin   (if per_cell_dir)
        {output_dir}/tsumegos/{board_size}/metadata.json
        {output_dir}/lookup.txt   (merged across sizes; rows: "id zipHash size")

    When per_cell_dir is supplied (pre-projected per-problem .npz files written
    by build_embedding.write_per_cell_projections), each shard also gets:
      - a sibling .embeddings.bin with fp16 per-cell K-dim vectors concatenated
        in SGF variation order, one tsumego after another;
      - embeddings_index.json inside the zip listing {tid: {offset, n_cells}}.
    Without that arg, no sidecar is written — useful for "just rezip the
    SGFs" runs.
    """

    metadata.sort(key=lambda x: x["rating"])
    group_count = int(len(metadata) ** 0.5)
    for i, item in enumerate(metadata):
        item["group"] = int(group_count * i / len(metadata))

    zip_dir = os.path.join(output_dir, "tsumegos", str(board_size))
    if os.path.exists(zip_dir):
        for f in os.listdir(zip_dir):
            if (f.endswith(".zip") or f.endswith(".embeddings.bin")
                    or f == "metadata.json"):
                os.remove(os.path.join(zip_dir, f))
    else:
        os.makedirs(zip_dir)

    embed_cells_enabled = per_cell_dir is not None and os.path.isdir(per_cell_dir)

    zip_metadata = []
    new_lookup_entries = []  # entries for the size we are (re)building now
    n_with_cells = n_without_cells = 0
    for group in range(group_count):
        group_items = [item for item in metadata if item["group"] == group]
        if not group_items:
            continue

        min_rating = min(item["rating"] for item in group_items)
        max_rating = max(item["rating"] for item in group_items)

        # Build the per-shard per-cell embedding sidecar + index ahead of the
        # zip so we can drop embeddings_index.json inside the same zip below.
        bin_buffer = bytearray()
        embeddings_index = {}
        if embed_cells_enabled:
            for item in group_items:
                tid = item["filename"].split(".")[0]
                per_cell_path = os.path.join(per_cell_dir, tid + ".npz")
                sgf_path = os.path.join(tsumegos_dir, item["filename"])
                cells, n_cells = project_cells_for_problem(per_cell_path, sgf_path)
                if cells is None:
                    n_without_cells += 1
                    continue
                offset = len(bin_buffer)
                bin_buffer.extend(cells.tobytes())  # fp16 little-endian
                embeddings_index[tid] = {"offset": offset, "n_cells": int(n_cells)}
                n_with_cells += 1

        zip_content = io.BytesIO()
        with zipfile.ZipFile(zip_content, 'w', zipfile.ZIP_DEFLATED) as temp_zip:
            for item in group_items:
                file_path = os.path.join(tsumegos_dir, item["filename"])
                with open(file_path, 'rb') as f:
                    temp_zip.writestr(item["filename"], f.read())

            group_metadata_content = [
                {k: v for k, v in item.items() if k != "group"}
                for item in group_items
            ]
            temp_zip.writestr("metadata.json",
                              json.dumps(group_metadata_content, indent=2))

            if embed_cells_enabled:
                temp_zip.writestr(
                    "embeddings_index.json",
                    json.dumps({"problems": embeddings_index}),
                )

        zip_content.seek(0)
        zip_hash = hashlib.sha256(zip_content.getvalue()).hexdigest()[:8]
        zip_filename = f"{zip_hash}.zip"
        zip_path = os.path.join(zip_dir, zip_filename)

        with open(zip_path, 'wb') as f:
            f.write(zip_content.getvalue())

        if embed_cells_enabled and bin_buffer:
            bin_filename = f"{zip_hash}.embeddings.bin"
            bin_path = os.path.join(zip_dir, bin_filename)
            with open(bin_path, 'wb') as f:
                f.write(bytes(bin_buffer))

        zip_metadata.append({
            "filename": zip_filename,
            "min_rating": min_rating,
            "max_rating": max_rating,
        })

        for item in group_items:
            tsumego_id = item["filename"].split('.')[0]
            new_lookup_entries.append((tsumego_id, zip_hash, board_size))

    if embed_cells_enabled:
        print(f"  per-cell embeddings: {n_with_cells} problems shipped, "
              f"{n_without_cells} skipped (no .npz / no rating / no SGF match)")

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


def compute_ratings_hash(metadata):
    """Stable content hash over (tsumego_id, rating) pairs for one size.

    The PWA reads this from embedding_meta.json (ratings_version[size]) and
    refreshes answerHistory.problemRating values from per-shard metadata.json
    whenever the hash changes. Rounded to 6 decimals so floating-point noise
    doesn't trigger spurious backfills."""
    pairs = sorted(
        (item["filename"].split(".")[0], round(float(item["rating"]), 6))
        for item in metadata
    )
    payload = json.dumps(pairs, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


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


def process_size(board_size, output_dir):
    """Run the full per-size packaging — ratings load, exponential rank
    mapping, metadata assembly, zip + per-cell sidecar emission. Returns
    (embedding_meta, ratings_hash); either element may be None if the
    corresponding artifact wasn't available. The multi-size loop aggregates
    both into the single shared embedding_meta.json at the end."""
    print(f"\n========== Packaging board size {board_size} ==========")

    TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "tsumegos", str(board_size))
    RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings", str(board_size))
    PER_CELL_DIR = os.path.join(
        PROJECT_ROOT, "embeddings", "joint", "per_cell", str(board_size))

    if not os.path.isdir(TSUMEGOS_DIR):
        print(f"  no tsumegos dir at {TSUMEGOS_DIR}; skipping")
        return None, None
    ratings_csv = os.path.join(RATINGS_DIR, "tsumego_ratings_final.csv")
    if not os.path.exists(ratings_csv):
        print(f"  no ratings at {ratings_csv}; skipping")
        return None, None

    # 1. Ratings map (raw + normalized PWA scale).
    tsumego_ratings_df = pd.read_csv(ratings_csv)
    tsumego_ratings_df['raw_rating'] = tsumego_ratings_df['rating']
    tsumego_ratings_df = tsumego_ratings_df.sort_values(
        by='rating', ascending=True).reset_index(drop=True)

    bot_ratings_df = pd.read_csv(os.path.join(RATINGS_DIR, "bot_ratings_final.csv"))
    # Exponential mapping anchored on humanv0_8k and humanv0_3k bots.
    # 8k is index 12 (center 12.5) and 3k is index 17 (center 17.5) on the
    # 0-based 20k..9d scale.
    bot_8k_rating = bot_ratings_df[bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_8k']['rating'].values[0]
    bot_3k_rating = bot_ratings_df[bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_3k']['rating'].values[0]
    x1, x2 = 12.5, 17.5
    y1, y2 = bot_8k_rating, bot_3k_rating
    k = 0.12  # variance decay constant
    A = (y2 - y1) / (np.exp(k * x2) - np.exp(k * x1))
    C = y1 - A * np.exp(k * x1)
    print(f"  exponential mapping: A={A:.6f}, C={C:.6f}")

    # Rating -> rank index -> [0, 1] PWA scale, clamped.
    tsumego_ratings_df['rank_index'] = (tsumego_ratings_df['rating'] - C) / A
    tsumego_ratings_df['rank_index'] = tsumego_ratings_df['rank_index'].clip(lower=1e-9)
    tsumego_ratings_df['rank_index'] = np.log(tsumego_ratings_df['rank_index']) / k
    tsumego_ratings_df['pwa_rating'] = tsumego_ratings_df['rank_index'] / 29.0
    tsumego_ratings_df['rating'] = tsumego_ratings_df['pwa_rating'].clip(0.0, 1.0)
    tsumego_ratings_df.drop(columns=['rank_index', 'pwa_rating'], inplace=True)

    tsumego_ratings_map = dict(zip(
        tsumego_ratings_df['tsumego_id'],
        zip(tsumego_ratings_df['rating'], tsumego_ratings_df['raw_rating']),
    ))
    raw_rating_map = dict(zip(
        tsumego_ratings_df['tsumego_id'],
        tsumego_ratings_df['raw_rating'].astype(float),
    ))

    # raw_rating_map is no longer needed for packaging — projection happens
    # upstream in build_embedding.write_per_cell_projections. Kept the
    # raw-rating extraction above only for the ratings_map metadata.json field.
    del raw_rating_map

    # 2. Quality map (normalized to [0, 1]).
    tsumego_quality_df = pd.read_csv(os.path.join(RATINGS_DIR, "tsumego_quality_final.csv"))
    min_q = tsumego_quality_df['quality'].min()
    max_q = tsumego_quality_df['quality'].max()
    tsumego_quality_df['quality'] = (tsumego_quality_df['quality'] - min_q) / (max_q - min_q)
    tsumego_quality_map = dict(zip(
        tsumego_quality_df['tsumego_id'], tsumego_quality_df['quality']))

    print(f"  ratings: {len(tsumego_ratings_map)}, qualities: {len(tsumego_quality_map)}")
    print(f"  loading joint embeddings (correct-cell field for metadata.json) ...")
    embedding_map, embedding_meta = load_embeddings(board_size)

    # 3. Final per-problem metadata.
    all_metadata = []
    sgf_files = [f for f in os.listdir(TSUMEGOS_DIR) if f.endswith(".sgf")]
    print(f"  found {len(sgf_files)} SGFs in {TSUMEGOS_DIR}")
    skipped_count = 0
    embedding_missing = 0
    for filename in sgf_files:
        tsumego_id = filename.split('.')[0]
        if tsumego_id in tsumego_ratings_map:
            rating, raw_rating = tsumego_ratings_map[tsumego_id]
            quality = tsumego_quality_map.get(tsumego_id, 0.0)
            item = {
                "filename": filename,
                "rating": rating,
                "raw_rating": raw_rating,
                "quality": quality,
            }
            if embedding_map is not None:
                emb = embedding_map.get(tsumego_id)
                if emb is None:
                    embedding_missing += 1
                else:
                    item["embedding"] = emb
            all_metadata.append(item)
        else:
            skipped_count += 1

    print(f"  metadata built for {len(all_metadata)} tsumegos "
          f"(skipped {skipped_count} without ratings)")
    if embedding_map is not None:
        n_with = sum(1 for m in all_metadata if "embedding" in m)
        print(f"  correct-cell embeddings on metadata.json: {n_with}/{len(all_metadata)} "
              f"(missing for {embedding_missing})")

    per_cell_arg = PER_CELL_DIR if os.path.isdir(PER_CELL_DIR) else None
    if per_cell_arg is None:
        print(f"  no per_cell/ at {PER_CELL_DIR} — skipping sidecar for this size "
              f"(run build_embedding to generate)")

    print(f"  building shards ...")
    create_packages(
        all_metadata, board_size, TSUMEGOS_DIR, output_dir,
        per_cell_dir=per_cell_arg,
    )
    size_dir = os.path.join(output_dir, 'tsumegos', str(board_size))
    print(f"  packages -> {size_dir}")

    print_rating_distribution(all_metadata)
    ratings_hash = compute_ratings_hash(all_metadata)
    print(f"  ratings_version[{board_size}]={ratings_hash}")
    return embedding_meta, ratings_hash


if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pwa")

    joint_embedding_meta = None
    ratings_versions = {}
    for size in ALL_BOARD_SIZES:
        meta, ratings_hash = process_size(size, OUTPUT_DIR)
        if meta is not None:
            joint_embedding_meta = meta  # joint across sizes; last write wins
        if ratings_hash is not None:
            ratings_versions[str(size)] = ratings_hash

    # Single shared embedding_meta.json. Carries both joint embedding info and
    # per-size ratings_version (content hash) so the PWA can independently
    # detect embedding refits vs rating refreshes.
    if joint_embedding_meta is not None or ratings_versions:
        meta_out = joint_embedding_meta if joint_embedding_meta is not None else {}
        K = meta_out.get("K") or 0
        meta_out["dtype"] = EMBEDDING_DTYPE_NAME
        meta_out["bytes_per_cell"] = int(K) * EMBEDDING_BYTES_PER_FLOAT
        if ratings_versions:
            meta_out["ratings_version"] = ratings_versions
        embedding_meta_path = os.path.join(OUTPUT_DIR, 'tsumegos', 'embedding_meta.json')
        os.makedirs(os.path.dirname(embedding_meta_path), exist_ok=True)
        with open(embedding_meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_out, f, indent=2)
        print(f"\nWrote joint embedding metadata at {embedding_meta_path} "
              f"(K={K}, dtype={EMBEDDING_DTYPE_NAME}, "
              f"bytes_per_cell={meta_out['bytes_per_cell']}, "
              f"ratings_version={ratings_versions or 'none'})")

    print("\nAll sizes done.")