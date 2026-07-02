import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import sgfmill.sgf

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES
from utils.katago_queries import gtp_to_xy
from utils.ratings_io import FLOOR_CSV, QUALITY_CSV, RATINGS_CSV, load_size_value_maps, load_value_map
from pipeline.benchmark_bots import parse_tsumego_sgf, sgf_to_gtp


# Per-cell embedding sidecar parameters. Keep these in sync with whatever the
# PWA-side decoder expects in pwa/tsumegos/embedding_meta.json.
EMBEDDING_DTYPE = np.float16
EMBEDDING_DTYPE_NAME = "float16"
EMBEDDING_BYTES_PER_FLOAT = 2

# Backend rating set this packages. v2 (combined cross-board MLP-2PL, written by
# pipeline.train_rating + pipeline.map_rating_to_rank) is the only rating
# backend — package_for_pwa is the single writer of pwa/tsumegos/ and always
# reads ratings/. RATING_MODEL_TAG is stamped into embedding_meta.json as
# `rating_model` for provenance (the PWA does not branch on it).
RATINGS_ROOT = "ratings"
RATING_MODEL_TAG = "v2"

# package_for_pwa is rank-agnostic: it ships one raw black-box rating per
# problem and never converts to ranks. The rating→rank curve (M) is fitted and
# written into ratings/rating_to_rank.json by pipeline.map_rating_to_rank
# (and copied across by this script); the client fetches it and does all
# rating→rank conversion itself.


def project_cells_for_problem(per_cell_path, sgf_path, reorder_by_sgf=True):
    """Load pre-projected per-cell embeddings for one tsumego in the order the
    PWA's cell index expects. Returns (fp16 (n_cells, K) array, n_cells). A missing,
    corrupt, or SGF-inconsistent .npz RAISES — packaging requires a per-cell projection
    for every problem (build_embedding fills the whole corpus), so a gap is a hard
    error, never silently skipped.

    Reads embeddings/joint/per_cell/<size>/<tid>.npz, written upstream by
    build_embedding.write_per_cell_projections. Those rows are in *extraction
    order*, which is board-scan order (sorted by (y, x)) — NOT the order the PWA
    uses. The PWA recovers a picked move's cell index via
    children.indexOf(moveNode), i.e. SGF root-variation document order, then
    Range-fetches that slice of the .embeddings.bin. So the bin MUST be written
    in SGF-variation order; otherwise every picked-move lookup reads the wrong
    cell's embedding. The reorder below is therefore load-bearing.

    The pass variation (present in every problem's SGF, never embedded by
    extraction) gets an ALL-ZEROS placeholder row so that bin row i == SGF child
    index i exactly. The PWA's children.indexOf counts the pass child, so
    skipping it here would shift every later cell's row by one — for the ~half
    of problems where the pass isn't the last variation, that mis-read a
    neighboring cell's embedding for every picked move after it (BIN_FORMAT_REV
    2 fixed this). The zeros row is never fetched: a board click can't select
    the pass node, and backfill's coordinate match never resolves to it.

    reorder_by_sgf=True (default) re-parses the SGF and reorders rows by the
    (x, y) → variation_index map. The sgfmill parse dominates packaging time
    (~60× the npz read), so it's the hot path — parallelize the caller, don't
    skip this. reorder_by_sgf=False ships the npz rows as-is and is ONLY correct
    if the .npz has been pre-reordered into SGF-variation order upstream; on the
    current build_embedding output (board-scan order) it produces a corrupt bin.
    """
    if not os.path.exists(per_cell_path):
        raise FileNotFoundError(
            f"missing per-cell projection {per_cell_path} — corpus and per_cell/ are "
            f"out of sync; run build_embedding")

    # A file that exists but won't load is corruption, not a coverage gap — let it
    # propagate and abort the run rather than silently dropping the sidecar.
    with np.load(per_cell_path) as data:
        embeddings = np.asarray(data["embeddings"])  # (n_cells, K) fp16
        xys = np.asarray(data["xys"])                # (n_cells, 2)

    if not reorder_by_sgf:
        if embeddings.shape[0] == 0:
            raise RuntimeError(f"per_cell {per_cell_path} has zero embedded cells")
        return embeddings.astype(EMBEDDING_DTYPE), int(embeddings.shape[0])

    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    _, variation_nodes = parse_tsumego_sgf(game)

    xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}

    ordered_rows = []
    n_real_cells = 0
    zero_row = np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
    for node in variation_nodes:
        _, mb = node.get_raw_move()
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is None:
            # Pass: never embedded — placeholder row for index parity (see docstring).
            ordered_rows.append(zero_row)
            continue
        row = xy_to_row.get(xy)
        if row is None:
            # The npz is meant to be a superset of the SGF's non-pass variations, so a
            # move with no embedded cell means stale/mismatched extraction. Bailing here
            # is load-bearing: a silently-dropped cell shifts every later cell's bin
            # index, corrupting picked-move lookups for the whole shard.
            raise RuntimeError(
                f"per_cell {per_cell_path} is missing cell {xy} present in {sgf_path} "
                f"(stale extraction?) — refusing to ship a misaligned sidecar")
        ordered_rows.append(embeddings[row])
        n_real_cells += 1

    if n_real_cells == 0:
        raise RuntimeError(
            f"per_cell {per_cell_path}: no non-pass variation matched {sgf_path}")

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
    n_with_cells = 0
    if embed_cells_enabled:
        for item in group_items:
            tid = item["filename"].split(".")[0]
            per_cell_path = os.path.join(per_cell_dir, tid + ".npz")
            sgf_path = os.path.join(tsumegos_dir, item["filename"])
            # project_cells_for_problem RAISES on any missing/corrupt/
            # SGF-inconsistent npz (a gap is a hard error, never silently
            # skipped), so there is no "no cells" case to count here.
            cells, n_cells = project_cells_for_problem(
                per_cell_path, sgf_path, reorder_by_sgf=reorder_by_sgf)
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

    # Tie-break equal ratings by filename so grouping (and thus the shard
    # contents/hashes) is deterministic — otherwise os.listdir order leaks in
    # and the reproducible-zip effort is undone for same-rating problems.
    metadata.sort(key=lambda x: (x["rating"], x["filename"]))
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
    n_with_cells = 0

    def handle_result(res):
        nonlocal n_with_cells
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
        print(f"  per-cell embeddings: {n_with_cells} problems shipped")

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
                if not line.strip():
                    continue  # tolerate blank/trailing lines only
                parts = line.split()
                # Every real row is "id zipHash size". A malformed row is corruption
                # (the legacy single-size format is long gone) — abort, don't drop.
                if len(parts) != 3:
                    raise ValueError(
                        f"malformed lookup.txt row (expected 'id zipHash size'): {line!r}")
                tsumego_id, zip_id, size_str = parts
                size_int = int(size_str)  # non-integer size aborts
                if size_int != board_size:
                    merged.append((tsumego_id, zip_id, size_int))

    merged.extend(new_lookup_entries)
    merged.sort(key=lambda x: x[0])
    with open(lookup_path, 'w', encoding='utf-8') as f:
        for tsumego_id, zip_id, size_int in merged:
            f.write(f"{tsumego_id} {zip_id} {size_int}\n")


def compute_ratings_hash(metadata):
    """Stable content hash over (tsumego_id, rating, slope, floor) tuples for one size.

    The PWA reads this from embedding_meta.json (ratings_version[size]) and
    refreshes answerHistory.problemRating / problemSlope / problemFloor from
    per-shard metadata.json whenever the hash changes. Slope and floor are included
    so a rating-stable re-fit that only moved the discriminations or guessing floors
    still bumps the version and backfills them. Rounded to 6 decimals so
    floating-point noise doesn't trigger spurious backfills."""
    tuples = sorted(
        (item["filename"].split(".")[0], round(float(item["rating"]), 6),
         round(float(item.get("slope", 0.0)), 6),
         round(float(item.get("floor", 0.0)), 6))
        for item in metadata
    )
    payload = json.dumps(tuples, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def print_rating_distribution(metadata):
    """Prints the final distribution of problems by raw rating (0.5-wide bins).
    package_for_pwa is rank-agnostic, so this diagnostic stays in rating units —
    the client maps ratings to ranks with the fitted curve."""
    if not metadata:
        print("No metadata to generate distribution from.")
        return

    total_problems = len(metadata)
    BIN = 0.5
    buckets = {}
    for item in metadata:
        i = int(np.floor(item['rating'] / BIN))
        buckets.setdefault(i, []).append(item)

    if not buckets:
        return

    print("\n--- Final Rating Distribution (raw rating, 0.5-wide bins) ---")
    for i in range(min(buckets), max(buckets) + 1):
        items = buckets.get(i, [])
        count = len(items)
        percentage = (count / total_problems * 100) if total_problems > 0 else 0
        avg_slope = sum(x['slope'] for x in items) / count if items else 0.0
        avg_rating = sum(x['rating'] for x in items) / count if items else 0.0
        label = f"[{i * BIN:+.1f},{(i + 1) * BIN:+.1f})"
        print(f"{label:>14s}: {count:>5d} problems ({percentage:5.1f}%, "
              f"avg slope: {avg_slope:.3f}, avg rating: {avg_rating:.3f})")


def _embeddings_dir(board_size):
    """The embeddings source dir for this size: prefer the joint model, fall back
    to the legacy per-size one. Returns (dir, is_joint) or (None, False)."""
    joint_dir = os.path.join(PROJECT_ROOT, "embeddings", "joint")
    legacy_dir = os.path.join(PROJECT_ROOT, "embeddings", str(board_size))
    if os.path.exists(os.path.join(joint_dir, "per_problem.csv")):
        return joint_dir, True
    if os.path.exists(os.path.join(legacy_dir, "per_problem.csv")):
        return legacy_dir, False
    return None, False


def _read_embedding_version(embed_dir):
    """The content-hash embedding_version stamped into pca.npz by build_embedding
    (the PWA reads it to auto-invalidate localStorage on refits), or None."""
    pca_path = os.path.join(embed_dir, "pca.npz")
    if not os.path.exists(pca_path):
        return None
    # A present pca.npz that won't load is corruption — let it abort rather than
    # silently shipping embedding_meta with a null version (which disables the PWA's
    # embedding-cache invalidation).
    with np.load(pca_path, allow_pickle=False) as arc:
        if "embedding_version" in arc.files:
            return str(arc["embedding_version"].item())
    return None


# Bump when the .embeddings.bin / embeddings_index.json layout changes in a way
# the client must react to. The rev is folded into the SHIPPED embedding_version
# (below), so a format change wipes + re-derives clients' stored embeddings via
# the standing version gate, and into size_inputs_hash, so the shards rebuild on
# the next run without --force. Rev history:
#   1 (implicit): SGF-variation order MINUS the pass — bin rows after a non-last
#     pass were off by one vs the PWA's children.indexOf cell index.
#   2: pass variations get an all-zeros placeholder row (index parity).
BIN_FORMAT_REV = 2


def shipped_embedding_version(embed_dir):
    """embedding_version as shipped to the client: the pca.npz content hash with
    the bin-format rev appended. The pca hash alone can't see sidecar-layout
    changes (same PCA, different bin bytes), so both invalidation consumers —
    the PWA's embedding wipe and the shard-build inputs hash — must use THIS,
    never the raw hash."""
    raw = _read_embedding_version(embed_dir)
    if raw is None:
        return None
    return f"{raw}-b{BIN_FORMAT_REV}"


def load_embeddings(board_size, meta_only=False):
    """Load per-problem PCA scores filtered to this board size, plus the
    shared embedding metadata (model is fit jointly across sizes).

    Returns (embedding_map, embedding_meta) where:
      embedding_map: tsumego_id -> list of K floats (None if unavailable or
                     meta_only — the per-row map is the only costly part, built
                     only when shards are actually being (re)built).
      embedding_meta: dict with K, embedding_version, labels.
                     The PWA only reads version (for cache invalidation) and
                     labels (for stats display); fields like explained-variance
                     and extreme_problems are kept locally in embeddings/joint/
                     as inspection aids and not shipped.
    """
    embed_dir, is_joint = _embeddings_dir(board_size)
    if embed_dir is None:
        # The whole embeddings artifact is missing (build_embedding never ran) — a
        # misconfigured run, not a per-problem coverage gap. Abort.
        raise FileNotFoundError(
            f"no embeddings under embeddings/{{joint,{board_size}}}/per_problem.csv "
            f"— run build_embedding")

    per_problem = os.path.join(embed_dir, "per_problem.csv")
    # meta_only: read just the header to count K columns, skip the row map.
    df = pd.read_csv(per_problem, nrows=0 if meta_only else None)
    embed_cols = sorted(
        [c for c in df.columns if c.startswith("embed_")],
        key=lambda c: int(c.split("_")[1]),
    )
    K = len(embed_cols)
    embedding_map = None
    if not meta_only:
        print(f"  reading embeddings from {embed_dir}")
        if is_joint and "size" in df.columns:
            df = df[df["size"] == board_size]
        embedding_map = {}
        for _, row in df.iterrows():
            embedding_map[row["tsumego_id"]] = [float(row[c]) for c in embed_cols]

    labels_path = os.path.join(embed_dir, "labels.json")
    labels = None
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    embedding_meta = {
        "K": K,
        "embedding_version": shipped_embedding_version(embed_dir),
        "labels": labels,
    }
    return embedding_map, embedding_meta


def _hash_file_into(h, path):
    """Feed a file's path + contents into hash `h` (chunked); 'MISSING' if absent."""
    h.update(path.encode())
    if os.path.exists(path):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    else:
        h.update(b"\x00MISSING")


def size_inputs_hash(board_size, reorder_by_sgf):
    """Content hash of everything that determines a size's shards (zips + .bin +
    metadata.json). When this is unchanged the shard build can be skipped — the
    expensive SGF reparse + zip/bin emission. Inputs:
      - tsumego_ratings_final.csv (rating drives grouping + metadata + min/max)
      - tsumego_quality_final.csv (metadata `quality`)
      - tsumego_floor_final.csv (metadata `floor`; absent on older backends)
      - embeddings/joint/per_problem.csv (correct-cell `embedding` in metadata)
      - shipped embedding_version (covers per_cell/*.npz content, which is
        regenerated with the embedding refit — those files are tid-named but
        their CONTENTS change on refit, so the version, not the filename set,
        is the signal — AND the bin-format rev, so a sidecar-layout change
        rebuilds shards without --force)
      - the SGF filename set (filenames ARE orientation-invariant content hashes,
        so a content change cannot occur without a filename change → the set hash
        detects add/remove/content)
      - reorder flag + ratings_root (output semantics)."""
    h = hashlib.sha256()
    rdir = os.path.join(PROJECT_ROOT, RATINGS_ROOT, str(board_size))
    _hash_file_into(h, os.path.join(rdir, "tsumego_ratings_final.csv"))
    _hash_file_into(h, os.path.join(rdir, "tsumego_quality_final.csv"))
    _hash_file_into(h, os.path.join(rdir, "tsumego_floor_final.csv"))
    embed_dir, _ = _embeddings_dir(board_size)
    if embed_dir:
        _hash_file_into(h, os.path.join(embed_dir, "per_problem.csv"))
        h.update((shipped_embedding_version(embed_dir) or "noemb").encode())
    tdir = os.path.join(PROJECT_ROOT, "tsumegos", str(board_size))
    names = sorted(f for f in os.listdir(tdir)
                   if f.endswith(".sgf")) if os.path.isdir(tdir) else []
    h.update(("\n".join(names)).encode())
    h.update(f"|reorder={reorder_by_sgf}|root={RATINGS_ROOT}".encode())
    return h.hexdigest()[:16]


def copy_curve(output_dir):
    """Copy the rating→rank curve (rating_to_rank.json + rank_map.json) from the
    backend RATINGS_ROOT into the served pwa/tsumegos/. package_for_pwa is the
    SINGLE writer of pwa/tsumegos/: map_rating_to_rank writes the curve into
    ratings/, and this copies it across. Returns the list of copied filenames.
    Every source is REQUIRED — the client cannot rank a problem without the curve and
    aborts boot without the population constants — so a missing source aborts the run
    rather than shipping a partial bundle."""
    src_dir = os.path.join(PROJECT_ROOT, RATINGS_ROOT)
    served = os.path.join(output_dir, "tsumegos")
    os.makedirs(served, exist_ok=True)
    copied = []
    # rating_to_rank.json + rank_map.json: the survey-calibrated rating→rank curve M
    # (map_rating_to_rank). population_constants.json: the fitted model constants the
    # client requires (fit_population_constants). All three are mandatory.
    required = {
        "rating_to_rank.json": "map_rating_to_rank",
        "rank_map.json": "map_rating_to_rank",
        "population_constants.json": "fit_population_constants",
    }
    for name, producer in required.items():
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"missing required {name} in {RATINGS_ROOT}/ — run {producer} first")
        shutil.copyfile(src, os.path.join(served, name))
        copied.append(name)
    return copied


def load_rating_artifacts(ratings_dir, ratings_csv):
    """The three raw per-problem rating artifacts package_for_pwa ships into each
    shard's metadata.json — returns (ratings_map, slope_map, floor_map):
      - ratings_map ← tsumego_ratings_final.csv `rating`: the raw black-box rating
        (the client converts it to a rank via the survey-calibrated curve; no rank
        baked in here).
      - slope_map   ← tsumego_quality_final.csv `quality`: the raw 2PL slope a_p,
        shipped RAW (the client uses it as the success-curve slope and as the
        selection weight ∝ a_p²).
      - floor_map   ← tsumego_floor_final.csv `floor`: the raw 3PL guessing floor
        c_p (lower asymptote of the per-attempt success curve).
    Uses the shared utils.ratings_io reader, so the shipped floats are
    correctly-rounded and consistent with the rest of the pipeline's loaders.
    (These values can differ from an older pandas-parsed bundle by ≤1 ULP, which
    re-hashes the shards — fine: this only runs when the shard set is rebuilt
    anyway, i.e. when a new ratings/SGF set has changed size_inputs_hash.)"""
    ratings_map = load_value_map(ratings_csv, "rating")

    quality_csv = os.path.join(ratings_dir, QUALITY_CSV)
    if not os.path.exists(quality_csv):
        raise FileNotFoundError(f"no slopes at {quality_csv} — run train_rating")
    slope_map = load_value_map(quality_csv, "quality")

    floor_csv = os.path.join(ratings_dir, FLOOR_CSV)
    if not os.path.exists(floor_csv):
        raise FileNotFoundError(f"no floors at {floor_csv} — run train_rating")
    floor_map = load_value_map(floor_csv, "floor")

    print(f"  ratings: {len(ratings_map)}, slopes: {len(slope_map)}, "
          f"floors: {len(floor_map)}")
    return ratings_map, slope_map, floor_map


def assemble_metadata(sgf_files, board_size, ratings_map, slope_map, floor_map,
                      embedding_map):
    """Join each SGF to its (rating, slope, floor, correct-cell embedding) into the
    per-problem metadata list. Every SGF must resolve all four; any gap means the
    corpus and the rating/embedding artifacts are out of sync, which aborts (rather
    than shipping a partial bundle) with a per-field tally of the orphans."""
    all_metadata = []
    missing_rating, missing_slope, missing_floor, missing_emb = [], [], [], []
    for filename in sgf_files:
        tsumego_id = filename.split('.')[0]
        if tsumego_id not in ratings_map:
            missing_rating.append(tsumego_id)
            continue
        if tsumego_id not in slope_map:
            missing_slope.append(tsumego_id)
            continue
        if tsumego_id not in floor_map:
            missing_floor.append(tsumego_id)
            continue
        emb = embedding_map.get(tsumego_id)
        if emb is None:
            missing_emb.append(tsumego_id)
            continue
        all_metadata.append({
            "filename": filename,
            "rating": ratings_map[tsumego_id],
            "slope": slope_map[tsumego_id],
            "floor": floor_map[tsumego_id],
            "embedding": emb,
        })

    def _sample(ids, n=5):
        return ", ".join(ids[:n]) + (" ..." if len(ids) > n else "")
    gaps = []
    for label, ids in (("rating", missing_rating), ("slope", missing_slope),
                       ("floor", missing_floor), ("embedding", missing_emb)):
        if ids:
            gaps.append(f"{len(ids)} without a {label} ({_sample(ids)})")
    if gaps:
        raise RuntimeError(
            f"[size {board_size}] corpus/artifact mismatch — " + "; ".join(gaps)
            + ". Re-run the rating + embedding pipeline (or remove the orphan SGFs).")
    return all_metadata


def process_size(board_size, output_dir, reorder_by_sgf=True, n_workers=None,
                 prev=None, force=False):
    """Run the full per-size packaging — ratings load, metadata assembly, zip +
    per-cell sidecar emission. Returns (embedding_meta, ratings_hash, inputs_hash);
    the first two may be None if the artifact wasn't available. The multi-size loop
    aggregates them into the single shared embedding_meta.json + the manifest.

    CHANGE DETECTION: if `prev` (this size's last-run manifest entry) carries an
    inputs_hash equal to the current size_inputs_hash AND the output zips still
    exist, the shard build is SKIPPED (the expensive SGF reparse + zip/bin write)
    and the cached ratings_hash is reused — so a curve-only or no-op re-run is near
    instant. `force` bypasses the skip. If embeddings/ is absent (e.g. the GPU
    artifacts were cleaned up) the shards can't be rebuilt: existing shards are kept
    as-is (returns (None, None, None), leaving meta + manifest untouched) so a
    curve/constants-only publish still works; only missing shards are then fatal.
    reorder_by_sgf/n_workers → create_packages."""
    print(f"\n========== Packaging board size {board_size} ==========")

    TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "tsumegos", str(board_size))
    # The backend rating set is always ratings/ (combined MLP-2PL). Output
    # lands in the served pwa/tsumegos/ structure.
    RATINGS_DIR = os.path.join(PROJECT_ROOT, RATINGS_ROOT, str(board_size))
    PER_CELL_DIR = os.path.join(
        PROJECT_ROOT, "embeddings", "joint", "per_cell", str(board_size))

    # Every size in ALL_BOARD_SIZES is expected to have a full set of artifacts; a
    # missing one is a misconfigured run, not something to skip past.
    if not os.path.isdir(TSUMEGOS_DIR):
        raise FileNotFoundError(f"no tsumegos dir at {TSUMEGOS_DIR}")
    ratings_csv = os.path.join(RATINGS_DIR, RATINGS_CSV)
    if not os.path.exists(ratings_csv):
        raise FileNotFoundError(f"no ratings at {ratings_csv} — run train_rating")

    inputs_hash = size_inputs_hash(board_size, reorder_by_sgf)
    zip_dir = os.path.join(output_dir, "tsumegos", str(board_size))
    # A shard set counts as present only with BOTH zips AND its metadata.json.
    # create_packages removes metadata.json first and rewrites it last, so
    # "zips exist but no metadata.json" means a crashed rebuild left a
    # half-built dir — require both so the skip / embeddings-absent-sentinel
    # paths below never validate (and ship) that partial dir.
    have_zips = (
        os.path.isdir(zip_dir)
        and os.path.exists(os.path.join(zip_dir, "metadata.json"))
        and any(f.endswith(".zip") for f in os.listdir(zip_dir))
    )

    # Embeddings absent: the shards CANNOT be (re)built. They may, however, already
    # exist from a prior full run on a machine where the heavy embeddings/ artifacts
    # have since been cleaned up. In that case leave the shards, the served
    # embedding_meta.json and the manifest entry UNTOUCHED (sentinel return) so a
    # curve/constants-only re-run still succeeds — the rating→rank curve and the
    # population constants are published separately (copy_curve) and don't depend on
    # the shard build. Only a missing shard set with no embeddings to build from is
    # fatal. This also sidesteps size_inputs_hash dropping its embedding terms when
    # embeddings/ is gone, which would otherwise change the hash and force a doomed
    # rebuild instead of the skip the unchanged shards deserve.
    if _embeddings_dir(board_size)[0] is None:
        if have_zips:
            print("  embeddings/ absent but shards present — keeping shards, "
                  "embedding_meta and manifest untouched (curve/constants-only run)")
            return None, None, None
        raise FileNotFoundError(
            f"no embeddings under embeddings/{{joint,{board_size}}}/ AND no existing "
            f"shards in {zip_dir} — run build_embedding first")

    if (not force and prev and have_zips
            and prev.get("inputs_hash") == inputs_hash
            and prev.get("ratings_hash")):
        print(f"  inputs unchanged (hash {inputs_hash}) and zips present — "
              f"SKIP shard rebuild; reusing ratings_version "
              f"{prev['ratings_hash']}")
        _, embedding_meta = load_embeddings(board_size, meta_only=True)
        return embedding_meta, prev["ratings_hash"], inputs_hash

    # Raw per-problem rating/slope/floor maps shipped into metadata.json (read
    # bit-stably with pandas — see load_rating_artifacts).
    ratings_map, slope_map, floor_map = load_rating_artifacts(RATINGS_DIR, ratings_csv)
    print(f"  loading joint embeddings (correct-cell field for metadata.json) ...")
    embedding_map, embedding_meta = load_embeddings(board_size)

    # Final per-problem metadata — every SGF joined to its rating, slope, floor and
    # correct-cell embedding (aborts on any gap; see assemble_metadata).
    sgf_files = [f for f in os.listdir(TSUMEGOS_DIR) if f.endswith(".sgf")]
    print(f"  found {len(sgf_files)} SGFs in {TSUMEGOS_DIR}")
    if not sgf_files:
        raise RuntimeError(f"no SGFs in {TSUMEGOS_DIR}")
    all_metadata = assemble_metadata(
        sgf_files, board_size, ratings_map, slope_map, floor_map, embedding_map)
    print(f"  metadata built for all {len(all_metadata)} tsumegos (no gaps)")

    if not os.path.isdir(PER_CELL_DIR):
        raise FileNotFoundError(
            f"no per_cell dir at {PER_CELL_DIR} — run build_embedding")
    per_cell_arg = PER_CELL_DIR

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
    return embedding_meta, ratings_hash, inputs_hash


def compute_data_version(output_dir):
    """Content hash over the per-size directory metadata.json files — the PWA's
    `data_version`, the single value it cache-busts metadata.json + lookup.txt by.
    Each metadata.json lists every shard's content-hash filename, so ANY data change
    (new tsumegos, re-rated problems, refit embeddings → a new shard hash) perturbs
    it. It gates the FETCH cache ONLY — new problems become serveable lazily — while
    the history backfill stays gated on ratings_version / embedding_version (those
    drive the immediate re-derive). Stable across pure code deploys, so a JS-only
    ship doesn't needlessly re-pull the (multi-MB) lookup.txt."""
    h = hashlib.sha256()
    for size in ALL_BOARD_SIZES:
        p = os.path.join(output_dir, "tsumegos", str(size), "metadata.json")
        try:
            with open(p, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(f"missing-{size}".encode())
    # Fold the merged problem_params.bin (per-problem rating/slope/floor for both
    # sizes) so a ratings-only change perturbs data_version and the PWA re-fetches it.
    pp = os.path.join(output_dir, "tsumegos", "problem_params.bin")
    try:
        with open(pp, "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(b"missing-problem-params")
    return h.hexdigest()[:16]


def emit_problem_params(output_dir):
    """Single packed file for BOTH board sizes: tsumego_id -> (rating, slope, floor).
    16 bytes/problem — a 4-byte id (the 8-hex tsumego_id) followed by 3 little-endian
    float32 (rating, slope, floor). The PWA Worker (leaderboard rank recompute) and
    the client ratings-change backfill both read it into one Map for O(1) rating
    resolution, so neither has to walk the shard zips. ~2.6MB for ~165k problems.

    Re-reads the ratings CSVs directly (not the per-size in-memory maps, which are
    absent on a skipped-shard run), so it always covers every size."""
    ratings_root = os.path.join(PROJECT_ROOT, RATINGS_ROOT)
    chunks = []
    for size in ALL_BOARD_SIZES:
        ratings, slopes, floors = load_size_value_maps(ratings_root, size, missing_ok=True)
        for tid, rating in ratings.items():
            if len(tid) != 8 or tid not in slopes or tid not in floors:
                continue
            try:
                id_bytes = bytes.fromhex(tid)
            except ValueError:
                continue
            vals = np.array([rating, slopes[tid], floors[tid]], dtype="<f4").tobytes()
            chunks.append(id_bytes + vals)
    out_path = os.path.join(output_dir, "tsumegos", "problem_params.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"".join(chunks))
    print(f"  wrote problem_params.bin ({len(chunks)} problems, "
          f"{len(chunks) * 16} bytes) at {out_path}")
    return out_path


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
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild every size's shards even if the change-detection manifest says "
             "their inputs are unchanged (the curve + embedding_meta are always "
             "refreshed regardless).")
    args = parser.parse_args()

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "pwa")
    MANIFEST_PATH = os.path.join(OUTPUT_DIR, "tsumegos", ".package_manifest.json")

    # package_for_pwa is the SINGLE writer of pwa/tsumegos/. A manifest of per-size
    # input content-hashes lets it skip rebuilding shards whose inputs are unchanged
    # (e.g. a curve-only re-run), so the heavy SGF reparse only happens when ratings
    # / embeddings / the SGF set actually changed.
    prev_manifest = {}
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH) as f:
                prev_manifest = json.load(f)
        except Exception as e:
            print(f"[warn] couldn't read {MANIFEST_PATH}: {e}; rebuilding all")
    prev_sizes = prev_manifest.get("sizes", {})

    joint_embedding_meta = None
    ratings_versions = {}
    # A size returns (None, None, None) on the embeddings-absent sentinel path
    # (its shards are left frozen at an older rating era). Track it so we don't
    # re-derive problem_params.bin from newer CSVs and desync it from them.
    any_sentinel = False
    # Seed with the prior manifest so a size that no-ops this run (unchanged shards,
    # or embeddings/ absent) keeps its recorded entry instead of being dropped.
    size_manifest = dict(prev_sizes)
    for size in ALL_BOARD_SIZES:
        meta, ratings_hash, inputs_hash = process_size(
            size, OUTPUT_DIR,
            reorder_by_sgf=args.reorder_by_sgf, n_workers=args.workers,
            prev=prev_sizes.get(str(size)), force=args.force)
        if meta is not None:
            joint_embedding_meta = meta  # joint across sizes; last write wins
        if ratings_hash is not None:
            ratings_versions[str(size)] = ratings_hash
            size_manifest[str(size)] = {
                "inputs_hash": inputs_hash, "ratings_hash": ratings_hash}
        else:
            any_sentinel = True

    # Copy the rating→rank curve from the backend ratings/ into the served path
    # (single-writer rule). Cheap; always refreshed so a curve-only edit ships even
    # when every size's shards were skipped above.
    print("\n--- curve ---")
    copied = copy_curve(OUTPUT_DIR)
    if copied:
        print(f"  copied {', '.join(copied)} from {RATINGS_ROOT}/ → pwa/tsumegos/")

    # Merged per-problem params (rating/slope/floor) for the leaderboard Worker and
    # the client ratings-change backfill. Emitted every run (cheap, deterministic)
    # and BEFORE compute_data_version so its bytes are folded into data_version.
    print("\n--- problem_params ---")
    params_path = os.path.join(OUTPUT_DIR, "tsumegos", "problem_params.bin")
    if any_sentinel and os.path.exists(params_path):
        # A size took the embeddings-absent sentinel path: its shards are frozen
        # at an older rating era. Re-deriving from the CURRENT ratings CSVs could
        # desync problem_params.bin from those frozen shards, so keep the
        # existing file (which matches the shards on disk).
        print("  keeping existing problem_params.bin (a size took the "
              "embeddings-absent sentinel path; shards frozen)")
    else:
        emit_problem_params(OUTPUT_DIR)

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
        meta_out["rating_model"] = RATING_MODEL_TAG
        # data_version: hash of the served metadata.json files — the PWA's cache-bust
        # key for metadata.json + lookup.txt (computed AFTER the curve copy so every
        # size's metadata.json is final on disk, including sizes whose shard rebuild
        # was skipped this run).
        meta_out["data_version"] = compute_data_version(OUTPUT_DIR)
        embedding_meta_path = os.path.join(OUTPUT_DIR, 'tsumegos', 'embedding_meta.json')
        os.makedirs(os.path.dirname(embedding_meta_path), exist_ok=True)
        with open(embedding_meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_out, f, indent=2)
        print(f"\nWrote joint embedding metadata at {embedding_meta_path} "
              f"(K={K}, dtype={EMBEDDING_DTYPE_NAME}, "
              f"bytes_per_cell={meta_out['bytes_per_cell']}, "
              f"ratings_version={ratings_versions or 'none'})")

    # Persist the change-detection manifest for the next run.
    if size_manifest:
        manifest = {"ratings_root": RATINGS_ROOT,
                    "model_tag": RATING_MODEL_TAG, "sizes": size_manifest}
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)

    print("\nAll sizes done.")