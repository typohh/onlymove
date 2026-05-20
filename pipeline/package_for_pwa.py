import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import sgfmill.sgf
from scipy.optimize import curve_fit

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES
from utils.katago_queries import get_all_ranks
from pipeline.benchmark_bots import parse_tsumego_sgf, sgf_to_gtp
from auxiliary.fit_statistics_shapes import CubicRatingCurve


# Per-cell embedding sidecar parameters. Keep these in sync with whatever the
# PWA-side decoder expects in pwa/tsumegos/embedding_meta.json.
EMBEDDING_DTYPE = np.float16
EMBEDDING_DTYPE_NAME = "float16"
EMBEDDING_BYTES_PER_FLOAT = 2

# Rating-to-rank mapping. Each problem's raw rating r (natural-log-odds skill,
# from pipeline.rank_tsumegos_and_bots) is mapped to a true-handicap-stones
# axis via r = A·exp(k·x_handi) + C, then linearly to the PWA's 0..1 axis
# where x = -17 (true 20k) -> 0 and x = +11 (true 9d) -> 1. The PWA axis is
# thus linear in handicap stones, which is the underlying definition of rank.
#
# The rating model in rank_tsumegos_and_bots is scale-invariant under joint
# additive shifts of (bot ratings, problem ratings), so the absolute level
# of r drifts every time the model is refit on a new corpus. We therefore
# can't hardcode (A, k, C) — they need to be re-fit at packaging time from
# the *current* bot_ratings_final.csv against a stable handicap-stone
# anchor. BOT_HANDI_RANK below is that anchor: the bots' 19x19 handicap-
# stone strength from auxiliary/calibrate_handicap_ranks.py --report-only,
# which is independent of the tsumego corpus and stable across rating
# refits. Per-bot SE in the handi calibration is ~0.2-0.6 stones; refresh
# the table if calibration_handicap_sgf/ grows substantially.
BOT_HANDI_RANK = {
    "20k": -14.59, "19k": -14.15, "18k": -13.00, "17k": -13.01,
    "16k": -12.21, "15k": -12.06, "14k": -10.86, "13k": -10.54,
    "12k": -9.66,  "11k": -8.85,  "10k":  -8.39, "9k":   -7.63,
    "8k":   -6.07, "7k":   -5.11, "6k":   -4.02, "5k":   -2.77,
    "4k":   -1.19, "3k":    0.00, "2k":   +1.36, "1k":   +2.54,
    "1d":   +3.76, "2d":   +5.12, "3d":   +5.93, "4d":   +6.85,
    "5d":   +7.24, "6d":   +7.63, "7d":   +7.69, "8d":   +8.01,
    "9d":   +8.56,
}
PWA_X0 = -17.0   # true-handi-stones at PWA axis 0 (nominal 20k)
PWA_X1 = +11.0   # true-handi-stones at PWA axis 1 (nominal 9d)
PWA_SPAN = PWA_X1 - PWA_X0
# Margin above the fit's asymptote C; keeps ln((r - C)/A) well-conditioned
# for the rare ultra-easy problem that ends up within an r-hair of C. At
# 0.1, neither current corpus actually clamps; raise (e.g. to 0.2) if a
# future regenerated corpus starts producing problems with r ≤ C + 0.1.
RATING_FLOOR_MARGIN = 0.1


# Per-board rating→internal(EGF)-rank curves fitted by pipeline.fit_rank_curves
# (EGF survey declarations as data, the cubic only as a shape prior). When this
# file is present it is the source of truth for the displayed rank, so the rank
# the PWA shows and pwa/tsumegos/rank_map.json agree by construction. When it is
# absent we fall back to the bare cubic prior (see process_size).
RATING_RANK_MAP_PATH = os.path.join(PROJECT_ROOT, "ratings", "rating_to_rank.json")
_RATING_RANK_MAP = None


def _load_rating_rank_map():
    global _RATING_RANK_MAP
    if _RATING_RANK_MAP is None:
        if os.path.exists(RATING_RANK_MAP_PATH):
            with open(RATING_RANK_MAP_PATH, "r", encoding="utf-8") as f:
                _RATING_RANK_MAP = json.load(f)
        else:
            _RATING_RANK_MAP = {}
    return _RATING_RANK_MAP


def _rating_to_rank_index(ratings, knot_ratings, knot_rank_indices):
    """Map raw ratings → 0..28 rank_index by linear interp over the fitted
    curve's knots, extrapolating past either end with the end slope."""
    xs = np.asarray(knot_ratings, dtype=float)
    ys = np.asarray(knot_rank_indices, dtype=float)
    r = np.asarray(ratings, dtype=float)
    out = np.interp(r, xs, ys)  # clamps beyond the ends; fix up below
    lo = r < xs[0]
    hi = r > xs[-1]
    if lo.any():
        out[lo] = ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (r[lo] - xs[0])
    if hi.any():
        out[hi] = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (r[hi] - xs[-1])
    return out


def fit_rating_curve(ratings_dir):
    """Fit r = A·exp(k·x_handi) + C against the current humanv0 bot ratings.

    Reads ratings_dir/bot_ratings_final.csv, joins each humanv0 bot (no
    temperature variant) against BOT_HANDI_RANK, and least-squares-fits the
    three-parameter exponential. Returns (A, k, C). The fit is the anchor
    between the freely-drifting tsumego rating scale and the stable
    handicap-stone scale, and must be recomputed every time the rating
    model is refit (i.e. every packaging run)."""
    bot_ratings_df = pd.read_csv(os.path.join(ratings_dir, "bot_ratings_final.csv"))
    xs, ys = [], []
    for _, row in bot_ratings_df.iterrows():
        name = row["bot_id"]
        if not name.startswith("b18c384nbt-humanv0_"):
            continue
        label = name[len("b18c384nbt-humanv0_"):]
        if "_t" in label:  # temperature variants share a base bot's name suffix
            continue
        if label not in BOT_HANDI_RANK:
            continue
        xs.append(BOT_HANDI_RANK[label])
        ys.append(float(row["rating"]))

    if len(xs) < 5:
        raise RuntimeError(
            f"not enough humanv0 bots matched to BOT_HANDI_RANK at {ratings_dir} "
            f"({len(xs)} found); check bot_id naming")

    xs, ys = np.array(xs), np.array(ys)
    # Initial guess scaled to the data range; curve_fit's default Levenberg-
    # Marquardt walks home from here in <50 iters on either corpus.
    y_range = ys.max() - ys.min()
    x_range = xs.max() - xs.min()
    p0 = [y_range / 2.0, float(np.log(2.0)) / x_range, ys.min() - y_range / 2.0]
    (A, k, C), _ = curve_fit(
        lambda x, A, k, C: A * np.exp(k * x) + C, xs, ys,
        p0=p0, maxfev=10000,
    )
    rmse = float(np.sqrt(np.mean((ys - (A * np.exp(k * xs) + C)) ** 2)))
    return float(A), float(k), float(C), rmse

GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # I-skipped, matches KataGo / GTP convention


def gtp_to_xy(gtp_coord):
    if gtp_coord.lower() == "pass":
        return None
    col_letter = gtp_coord[0].upper()
    x = GTP_COLS.index(col_letter)
    y = int(gtp_coord[1:]) - 1
    return x, y


def project_cells_for_problem(per_cell_path, sgf_path, reorder_by_sgf=True):
    """Load pre-projected per-cell embeddings for one tsumego in the order the
    PWA's cell index expects. Returns (fp16 (n_cells, K) array, n_cells), or
    (None, 0) on missing/corrupt/empty input.

    Reads embeddings/joint/per_cell/<size>/<tid>.npz, written upstream by
    build_embedding.write_per_cell_projections. Those rows are in *extraction
    order*, which is board-scan order (sorted by (y, x)) — NOT the order the PWA
    uses. The PWA recovers a picked move's cell index via
    children.indexOf(moveNode), i.e. SGF root-variation document order, then
    Range-fetches that slice of the .embeddings.bin. So the bin MUST be written
    in SGF-variation order; otherwise every picked-move lookup reads the wrong
    cell's embedding. The reorder below is therefore load-bearing.

    reorder_by_sgf=True (default) re-parses the SGF and reorders rows by the
    (x, y) → variation_index map. The sgfmill parse dominates packaging time
    (~60× the npz read), so it's the hot path — parallelize the caller, don't
    skip this. reorder_by_sgf=False ships the npz rows as-is and is ONLY correct
    if the .npz has been pre-reordered into SGF-variation order upstream; on the
    current build_embedding output (board-scan order) it produces a corrupt bin.
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

    if not reorder_by_sgf:
        if embeddings.shape[0] == 0:
            return None, 0
        return embeddings.astype(EMBEDDING_DTYPE), int(embeddings.shape[0])

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


# Fixed DOS-epoch timestamp for every zip entry. zipfile.writestr stamps each
# member with the current wall-clock time by default, which makes the zip bytes
# (and thus the content hash used as the shard filename + referenced in
# lookup.txt) change on every run even when nothing changed — forcing the PWA to
# re-download every shard. Pinning the timestamp makes archives reproducible:
# identical content -> identical bytes -> stable filename -> no spurious refetch.
ZIP_ENTRY_DATE_TIME = (1980, 1, 1, 0, 0, 0)  # 1980 is the zip-format minimum


def _zip_writestr(zf, name, data):
    """writestr with a pinned timestamp and explicit DEFLATE, so byte output is
    a deterministic function of (name, data) alone. A bare ZipInfo defaults to
    STORED, so we set compress_type; external_attr matches zipfile's own
    file-from-arcname default (0o600) to avoid any other run-to-run drift."""
    info = zipfile.ZipInfo(filename=name, date_time=ZIP_ENTRY_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    zf.writestr(info, data)


def _build_group_shard(task):
    """Worker: build one shard's zip bytes (+ per-cell bin) entirely in memory.

    Each group is independent — its zip, hash, and embedding bin depend only on
    its own items — so this is the unit of parallelism. The parent process owns
    all disk writes; this returns the bytes plus the metadata it needs. Lives at
    module scope so ProcessPoolExecutor can pickle it.
    """
    (group_index, group_items, tsumegos_dir, per_cell_dir,
     embed_cells_enabled, reorder_by_sgf) = task

    min_rating = min(item["rating"] for item in group_items)
    max_rating = max(item["rating"] for item in group_items)

    # Build the per-shard per-cell embedding sidecar + index ahead of the
    # zip so we can drop embeddings_index.json inside the same zip below.
    bin_buffer = bytearray()
    embeddings_index = {}
    n_with_cells = n_without_cells = 0
    if embed_cells_enabled:
        for item in group_items:
            tid = item["filename"].split(".")[0]
            per_cell_path = os.path.join(per_cell_dir, tid + ".npz")
            sgf_path = os.path.join(tsumegos_dir, item["filename"])
            cells, n_cells = project_cells_for_problem(
                per_cell_path, sgf_path, reorder_by_sgf=reorder_by_sgf)
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
                _zip_writestr(temp_zip, item["filename"], f.read())

        group_metadata_content = [
            {k: v for k, v in item.items() if k != "group"}
            for item in group_items
        ]
        _zip_writestr(temp_zip, "metadata.json",
                      json.dumps(group_metadata_content, indent=2))

        if embed_cells_enabled:
            _zip_writestr(temp_zip, "embeddings_index.json",
                          json.dumps({"problems": embeddings_index}))

    zip_bytes = zip_content.getvalue()
    zip_hash = hashlib.sha256(zip_bytes).hexdigest()[:8]

    return {
        "group_index": group_index,
        "zip_hash": zip_hash,
        "zip_bytes": zip_bytes,
        "bin_bytes": bytes(bin_buffer) if (embed_cells_enabled and bin_buffer) else None,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "tsumego_ids": [item["filename"].split(".")[0] for item in group_items],
        "n_with_cells": n_with_cells,
        "n_without_cells": n_without_cells,
    }


def create_packages(metadata, board_size, tsumegos_dir="tsumegos", output_dir="pwa",
                    per_cell_dir=None, reorder_by_sgf=True, n_workers=None):
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

    Shards are built in parallel across n_workers processes (default
    os.cpu_count()); the parent serializes only the disk writes. reorder_by_sgf
    is forwarded to project_cells_for_problem (see there).
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

    grouped = {}
    for item in metadata:
        grouped.setdefault(item["group"], []).append(item)
    tasks = [
        (gi, items, tsumegos_dir, per_cell_dir, embed_cells_enabled, reorder_by_sgf)
        for gi, items in sorted(grouped.items())
    ]

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, len(tasks))) if tasks else 1

    zip_metadata = []  # (group_index, entry) — sorted to group order before write
    new_lookup_entries = []  # entries for the size we are (re)building now
    n_with_cells = n_without_cells = 0

    def handle_result(res):
        nonlocal n_with_cells, n_without_cells
        zip_filename = f"{res['zip_hash']}.zip"
        with open(os.path.join(zip_dir, zip_filename), 'wb') as f:
            f.write(res["zip_bytes"])
        if res["bin_bytes"] is not None:
            bin_filename = f"{res['zip_hash']}.embeddings.bin"
            with open(os.path.join(zip_dir, bin_filename), 'wb') as f:
                f.write(res["bin_bytes"])
        zip_metadata.append((res["group_index"], {
            "filename": zip_filename,
            "min_rating": res["min_rating"],
            "max_rating": res["max_rating"],
        }))
        for tsumego_id in res["tsumego_ids"]:
            new_lookup_entries.append((tsumego_id, res["zip_hash"], board_size))
        n_with_cells += res["n_with_cells"]
        n_without_cells += res["n_without_cells"]

    if n_workers == 1:
        for task in tasks:
            handle_result(_build_group_shard(task))
    else:
        # as_completed (not map): write each shard as soon as its worker
        # finishes so we never hold more than ~n_workers shards' bytes in the
        # parent at once. Group order is restored by sorting zip_metadata below.
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_build_group_shard, task) for task in tasks]
            for fut in as_completed(futures):
                handle_result(fut.result())

    zip_metadata.sort(key=lambda x: x[0])
    zip_metadata = [entry for _, entry in zip_metadata]

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
    """Prints the final distribution of problems by named rank.

    Ratings can extend beyond [0, 1] because the rating-to-rank curve doesn't
    clamp. Bucket index i = floor(rating * num_ranks); i in [0, num_ranks-1]
    covers 20k..9d, i < 0 covers ranks weaker than 20k (21k, 22k, ...), and
    i >= num_ranks covers ranks stronger than 9d (10d, 11d, ...).
    """
    if not metadata:
        print("No metadata to generate distribution from.")
        return

    num_ranks = len(get_all_ranks())
    total_problems = len(metadata)

    buckets = {}
    for item in metadata:
        i = int(np.floor(item['rating'] * num_ranks))
        buckets.setdefault(i, []).append(item)

    if not buckets:
        return

    print("\n--- Final Rating Distribution ---")
    for i in range(min(buckets), max(buckets) + 1):
        items = buckets.get(i, [])
        count = len(items)
        percentage = (count / total_problems * 100) if total_problems > 0 else 0
        avg_quality = sum(x['quality'] for x in items) / count if items else 0.0
        avg_raw_rating = sum(x['raw_rating'] for x in items) / count if items else 0.0
        label = f"{20 - i}k" if i < 20 else f"{i - 19}d"
        print(f"{label:>4s}: {count:>5d} problems ({percentage:5.1f}%, "
              f"avg quality: {avg_quality:.3f}, avg rating: {avg_raw_rating:.3f})")


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


def process_size(board_size, output_dir, reorder_by_sgf=True, n_workers=None):
    """Run the full per-size packaging — ratings load, exponential rank
    mapping, metadata assembly, zip + per-cell sidecar emission. Returns
    (embedding_meta, ratings_hash); either element may be None if the
    corresponding artifact wasn't available. The multi-size loop aggregates
    both into the single shared embedding_meta.json at the end.

    reorder_by_sgf and n_workers are forwarded to create_packages."""
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

    # Rating → rank mapping. Source of truth is the per-board curve fitted by
    # pipeline.fit_rank_curves from EGF survey declarations (internal rank = EGF
    # rank), so the displayed rank and the server rank-translation map
    # (pwa/tsumegos/rank_map.json) agree by construction. If that file is absent
    # we fall back to the bare statistics.txt cubic prior (the same curve
    # fit_rank_curves uses as its prior). rank_index is the 0..28 named axis
    # (20k=0, 1k=19, 9d=28); /29 keeps the existing 0..1 scale so
    # print_rating_distribution and the PWA's label logic are unchanged.
    raw = tsumego_ratings_df['rating'].to_numpy()
    rrm = _load_rating_rank_map().get(str(board_size))
    if rrm is not None:
        rank_index = _rating_to_rank_index(
            raw, rrm["ratings"], rrm["rank_indices"])
        print(f"  rating→rank via fitted EGF-anchored curve "
              f"({RATING_RANK_MAP_PATH})")
    else:
        bot_ratings_df = pd.read_csv(
            os.path.join(RATINGS_DIR, "bot_ratings_final.csv"))
        bot_8k_rating = float(bot_ratings_df.loc[
            bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_8k', 'rating'].values[0])
        bot_3k_rating = float(bot_ratings_df.loc[
            bot_ratings_df['bot_id'] == 'b18c384nbt-humanv0_3k', 'rating'].values[0])
        cubic_curve = CubicRatingCurve(bot_8k_rating, bot_3k_rating)
        rank_index = cubic_curve.inverse(raw)
        print(f"  rating→rank via statistics.txt cubic prior — no fitted curve "
              f"at {RATING_RANK_MAP_PATH} (run pipeline.fit_rank_curves)")
    tsumego_ratings_df['rating'] = rank_index / 29.0

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
        per_cell_dir=per_cell_arg, reorder_by_sgf=reorder_by_sgf,
        n_workers=n_workers,
    )
    size_dir = os.path.join(output_dir, 'tsumegos', str(board_size))
    print(f"  packages -> {size_dir}")

    print_rating_distribution(all_metadata)
    ratings_hash = compute_ratings_hash(all_metadata)
    print(f"  ratings_version[{board_size}]={ratings_hash}")
    return embedding_meta, ratings_hash


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package rated tsumegos for the PWA.")
    parser.add_argument(
        "--no-sgf-reorder", dest="reorder_by_sgf", action="store_false",
        help="ship per-cell embeddings in raw npz (board-scan) order instead of "
             "reordering to SGF-variation order. ONLY correct if the .npz files "
             "have been pre-reordered upstream; on current build_embedding "
             "output this produces a corrupt picked-move sidecar. Default off.")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="parallel shard-builder processes (default: os.cpu_count())")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pwa")

    joint_embedding_meta = None
    ratings_versions = {}
    for size in ALL_BOARD_SIZES:
        meta, ratings_hash = process_size(
            size, OUTPUT_DIR,
            reorder_by_sgf=args.reorder_by_sgf, n_workers=args.workers)
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

    # Ship the rating→rank map (M) so the client can convert the raw ratings it
    # also receives into ranks itself — the displayed rank is then computed in
    # rating space and mapped with the same survey-calibrated M that labels the
    # problems, closing the rank loop. (Source: pipeline.fit_rank_curves.)
    if os.path.exists(RATING_RANK_MAP_PATH):
        dst = os.path.join(OUTPUT_DIR, 'tsumegos', 'rating_to_rank.json')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(RATING_RANK_MAP_PATH, 'r', encoding='utf-8') as fsrc:
            payload = fsrc.read()
        with open(dst, 'w', encoding='utf-8') as fdst:
            fdst.write(payload)
        print(f"Copied rating→rank map → {dst}")
    else:
        print(f"\n[warn] {RATING_RANK_MAP_PATH} missing — client will fall back to "
              f"rank-space skill (run pipeline.fit_rank_curves first)")

    print("\nAll sizes done.")