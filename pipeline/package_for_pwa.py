import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
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

from utils.paths import tsumego_dir
from utils.constants import ALL_BOARD_SIZES
from utils.katago.coords import gtp_to_xy, sgf_to_gtp
from utils.rating.ratings_io import FLOOR_CSV, RATINGS_CSV, SLOPE_CSV, load_size_value_maps, load_value_map
from utils.rating import se_floor  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf
from utils.katago.add_variations import parse_eval_visits
from utils.rating.motif_rules import LABEL_SET, RULE_LABELS


# Per-cell embedding sidecar parameters. Keep these in sync with whatever the
# PWA-side decoder expects in pwa/tsumegos/embedding_meta.json.
# Reserved point code for the pass variation in the published cell map. Board
# points encode as `y * size + x`, so the largest real code is 168 on 13x13 and
# 255 can never collide with one.
PASS_POINT_CODE = 255

EMBEDDING_DTYPE = np.float16
EMBEDDING_DTYPE_NAME = "float16"
EMBEDDING_BYTES_PER_FLOAT = 2

# The availability null (lookup.altstats.bin) carries a MASS per axis and side,
# not a count: each alternative is weighed by the visits its evaluation rests on,
# so the quantity is a sum over alternatives rather than how many there are. The
# masses are INTEGER sums of integer visit counts, which is what lets this
# producer and the client's live path agree exactly rather than nearly — no
# rounding stands between them. uint32 because a mass is bounded by
# (visits per variation) x (variations on the board), well past a uint8 count.
ALTSTATS_DTYPE = np.dtype("<u4")
ALTSTATS_DTYPE_NAME = "uint32"
ALTSTATS_BYTES_PER_QUANTITY = 4

# Backend rating set this packages. v2 (combined cross-board MLP-2PL, written by
# pipeline.train_mirt + pipeline.map_rating_to_rank) is the only rating
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


def project_cells_for_problem(per_cell_path, sgf_path):
    """Load pre-projected per-cell embeddings for one tsumego in the order the
    PWA's cell index expects. Returns (fp16 (n_cells, K) array, n_cells, point codes).
    The point codes are the same variations' board points as uint8 `y * size + x`
    (255 for the pass), which is what lets a reader that holds no SGF turn a
    reported move coordinate back into a variation index. A missing,
    corrupt, or SGF-inconsistent .npz RAISES — packaging requires a per-cell projection
    for every problem (write_mirt_per_cell fills the whole corpus), so a gap is a hard
    error, never silently skipped.

    Reads axes/per_cell/<size>/<tid>.npz, written upstream by
    pipeline.write_mirt_per_cell. Those rows are in *extraction
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

    The sgfmill parse dominates packaging time (~60× the npz read), so this is
    the hot path; create_packages parallelizes it across processes.
    """
    if not os.path.exists(per_cell_path):
        raise FileNotFoundError(
            f"missing per-cell projection {per_cell_path} — corpus and per_cell/ are "
            f"out of sync; run pipeline.write_mirt_per_cell")

    # A file that exists but won't load is corruption, not a coverage gap — let it
    # propagate and abort the run rather than silently dropping the sidecar.
    with np.load(per_cell_path) as data:
        embeddings = np.asarray(data["embeddings"])  # (n_cells, K) fp16
        xys = np.asarray(data["xys"])                # (n_cells, 2)

    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    _, variation_nodes = parse_tsumego_sgf(game)

    xy_to_row = {(int(x), int(y)): i for i, (x, y) in enumerate(xys)}

    ordered_rows = []
    point_codes = bytearray()
    n_real_cells = 0
    n_pass = 0
    zero_row = np.zeros(embeddings.shape[1], dtype=embeddings.dtype)
    for node in variation_nodes:
        _, mb = node.get_raw_move()
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is None:
            # Pass: never embedded — placeholder row for index parity (see docstring).
            ordered_rows.append(zero_row)
            point_codes.append(PASS_POINT_CODE)
            n_pass += 1
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
        point_codes.append(int(xy[1]) * size + int(xy[0]))
        n_real_cells += 1

    if n_real_cells == 0:
        raise RuntimeError(
            f"per_cell {per_cell_path}: no non-pass variation matched {sgf_path}")
    # Every problem carries exactly one pass variation (it holds the refutation
    # line). Two would mean two placeholder rows sharing one reserved code, and
    # none would mean the tree is not the shape everything downstream assumes.
    if n_pass != 1:
        raise RuntimeError(
            f"{sgf_path}: {n_pass} pass variations, expected exactly 1 — the "
            f"variation list is not the shape the published layout assumes")

    return (np.stack(ordered_rows, axis=0).astype(EMBEDDING_DTYPE),
            len(ordered_rows), bytes(point_codes))


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
     embed_cells_enabled) = task

    min_rating = min(item["rating"] for item in group_items)
    max_rating = max(item["rating"] for item in group_items)

    # Build the per-shard per-cell embedding sidecar + index ahead of the
    # zip so we can drop embeddings_index.json inside the same zip below.
    bin_buffer = bytearray()
    cells_buffer = bytearray()
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
            cells, n_cells, point_codes = project_cells_for_problem(
                per_cell_path, sgf_path)
            offset = len(bin_buffer)
            bin_buffer.extend(cells.tobytes())  # fp16 little-endian
            # The cell map runs in lockstep with the embedding block at one byte
            # per cell, so it needs no offsets of its own: a problem's codes
            # start at offset // bytes_per_cell and run n_cells bytes.
            cells_buffer.extend(point_codes)
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
        "cells_bytes": bytes(cells_buffer) if (embed_cells_enabled and cells_buffer) else None,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "tsumego_ids": [item["filename"].split(".")[0] for item in group_items],
        "n_with_cells": n_with_cells,
    }


def create_packages(metadata, board_size, tsumegos_dir="tsumegos", output_dir="pwa",
                    per_cell_dir=None, n_workers=None):
    """Sorts, groups, and packages tsumegos into zip files based on rating.

    Output layout:
        {output_dir}/tsumegos/{board_size}/{hash}.zip
        {output_dir}/tsumegos/{board_size}/{hash}.embeddings.bin   (if per_cell_dir)
        {output_dir}/tsumegos/{board_size}/{hash}.cells.bin        (if per_cell_dir)
        {output_dir}/tsumegos/{board_size}/metadata.json
        {output_dir}/lookup.txt   (merged across sizes; rows: "id zipHash size")

    When per_cell_dir is supplied (pre-projected per-problem .npz files written
    by pipeline.write_mirt_per_cell), each shard also gets:
      - a sibling .embeddings.bin with fp16 per-cell K-dim vectors concatenated
        in SGF variation order, one tsumego after another;
      - embeddings_index.json inside the zip listing {tid: {offset, n_cells}}.
    Without that arg, no sidecar is written — useful for "just rezip the
    SGFs" runs.

    Shards are built in parallel across n_workers processes (default
    os.cpu_count()); the parent serializes only the disk writes.
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
                    or f.endswith(".cells.bin") or f == "metadata.json"):
                os.remove(os.path.join(zip_dir, f))
    else:
        os.makedirs(zip_dir)

    embed_cells_enabled = per_cell_dir is not None and os.path.isdir(per_cell_dir)

    grouped = {}
    for item in metadata:
        grouped.setdefault(item["group"], []).append(item)
    tasks = [
        (gi, items, tsumegos_dir, per_cell_dir, embed_cells_enabled)
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
        if res["cells_bytes"] is not None:
            cells_filename = f"{res['zip_hash']}.cells.bin"
            with open(os.path.join(zip_dir, cells_filename), 'wb') as f:
                f.write(res["cells_bytes"])
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


def _axes_dir():
    """The skill-axis source dir (axes/, written by pipeline.train_mirt and pipeline.write_mirt_per_cell), or None
    if it hasn't been built (the axes-absent sentinel — the size's shards are then
    left frozen at their older rating era). Returns (dir, is_joint) or (None, False);
    is_joint is always True for a built fit — the axes are fit jointly across board
    sizes, and there is deliberately no per-size fallback, so a stale per-size
    directory cannot silently stand in for the joint fit."""
    axes_dir = os.path.join(PROJECT_ROOT, "axes")
    if os.path.exists(os.path.join(axes_dir, "per_problem.csv")):
        return axes_dir, True
    return None, False


def axis_count_to_ship(output_dir):
    """The axis count K this run will publish, and where it came from.

    Two sources, in the order the run itself resolves them: a built
    axes/per_problem.csv (a fresh or unchanged axis fit — its
    embed_* column count IS K), else the K already stamped in the served
    embedding_meta.json (the embeddings-absent sentinel path, where the shards
    stay frozen and only the curve + constants are republished). Returns
    (K, source_description), or (None, reason) when neither exists — a run that
    process_size will abort on anyway."""
    embed_dir, _ = _axes_dir()
    if embed_dir is not None:
        header = pd.read_csv(os.path.join(embed_dir, "per_problem.csv"), nrows=0)
        return (sum(1 for c in header.columns if c.startswith("embed_")),
                "axes/per_problem.csv")
    served = os.path.join(output_dir, "tsumegos", "embedding_meta.json")
    if os.path.exists(served):
        with open(served, encoding="utf-8") as f:
            k = json.load(f).get("K")
        if isinstance(k, int) and k > 0:
            return k, "the served embedding_meta.json (shards frozen)"
    return None, "no axes/ and no served embedding_meta.json"


def _read_embedding_version(embed_dir):
    """The content-hash embedding_version stamped into axes.npz by
    pipeline.train_mirt (the PWA reads it to auto-invalidate localStorage on
    refits), or None."""
    axes_path = os.path.join(embed_dir, "axes.npz")
    if not os.path.exists(axes_path):
        return None
    # A present axes.npz that won't load is corruption — let it abort rather than
    # silently shipping embedding_meta with a null version (which disables the PWA's
    # embedding-cache invalidation).
    with np.load(axes_path, allow_pickle=False) as arc:
        if "embedding_version" in arc.files:
            return str(arc["embedding_version"].item())
    return None


# Bump when the published per-cell layout changes in a way the client must react
# to — the shard .embeddings.bin / embeddings_index.json, or the lookup-aligned
# alt-stats read off them, since a stored altAbove/altBelow is fit-derived data
# governed by the same gate as the vectors (EMBEDDING_VECTOR_FIELDS in
# pwa/js/constants.js). The rev is folded into the SHIPPED embedding_version
# (below), so a format change wipes + re-derives clients' stored embeddings via
# the standing version gate, and into size_inputs_hash, so the shards rebuild on
# the next run without --force. Rev history:
#   1 (implicit): SGF-variation order MINUS the pass — bin rows after a non-last
#     pass were off by one vs the PWA's children.indexOf cell index.
#   2: pass variations get an all-zeros placeholder row (index parity).
#   3: the availability null weighs each alternative by its visits instead of
#     counting them, so every stored altAbove/altBelow means something new.
BIN_FORMAT_REV = 3


def shipped_embedding_version(embed_dir):
    """embedding_version as shipped to the client: the axes.npz content hash with
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
                     labels (for stats display); the rest of the fit (the
                     loadings in axes.npz, the per-problem values) stays in
                     axes/ and is not shipped.
    """
    embed_dir, is_joint = _axes_dir()
    if embed_dir is None:
        # The whole embeddings artifact is missing (train_mirt never ran) — a
        # misconfigured run, not a per-problem coverage gap. Abort.
        raise FileNotFoundError(
            "no embeddings at axes/per_problem.csv — run pipeline.train_mirt")

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

    # Labels are OPTIONAL to ship. An axis with neither tail named is published
    # as a working dimension that says nothing about itself: it steers selection
    # and weakness detection exactly like a named one, and the client prints no
    # sentence for a tail it has no name for (stats-weaknesses.js skips such an
    # axis, motif-bubble.js skips such a tail). Naming it in axes/labels.json is
    # what turns it into a readable finding, so report which axes are waiting.
    # `components` is a list of PROXY RULE names — the same list
    # pipeline.train_mirt teaches from, `"A/B"` putting rule A at an axis's
    # positive tail and rule B at its negative — so the axis a name ships on is
    # the axis that name was taught onto, by construction. The names resolve
    # through utils.rating.motif_rules, which owns both the criteria and the
    # display text, and NEVER through labels.json's `bank`: that is the dev motif
    # store, and it holds several directions under each of these names, so a name
    # lookup there is ambiguous in both the direction and its sign. Flatten to the
    # {positive, negative, i18n} records the client expects (backfill.js reads
    # meta.labels.components and nothing else); embedding_meta.json is INLINED
    # into the JS bundle, so ship the text and nothing else.
    comps = []
    for spec in (labels or {}).get("components", []):
        poles = [p.strip() for p in str(spec).split("/")]
        unknown = [p for p in poles if p not in LABEL_SET]
        if unknown:
            raise SystemExit(
                f"axes/labels.json component {spec!r} names rules that do not "
                f"exist: {unknown}. Known rules: {LABEL_SET}. A component is the "
                f"shipping decision and must name what train_mirt actually "
                f"taught, so a typo here is a mislabeled axis, not a missing one.")
        rec = {}
        for side, pole in zip(("positive", "negative"), poles):
            rec[side] = pole
            rec[side + "_i18n"] = RULE_LABELS.get(pole)
        comps.append({k: rec.get(k) for k in
                      ("positive", "positive_i18n", "negative", "negative_i18n")})
    unlabeled = [i for i, c in enumerate(comps)
                 if not (c.get("positive") or c.get("negative"))]
    if len(comps) != K or unlabeled:
        print(f"  NOTE: axis labels incomplete ({len(comps)} components for "
              f"K={K}; unlabeled axes {unlabeled}) — those axes ship and work, "
              f"but stay silent on the client until named in axes/labels.json")

    embedding_meta = {
        "K": K,
        "embedding_version": shipped_embedding_version(embed_dir),
        "labels": {"components": comps},
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


def size_inputs_hash(board_size):
    """Content hash of everything that determines a size's shards (zips + .bin +
    metadata.json). When this is unchanged the shard build can be skipped — the
    expensive SGF reparse + zip/bin emission. Inputs:
      - tsumego_ratings.csv (rating drives grouping + metadata + min/max)
      - tsumego_slopes.csv (metadata `slope`)
      - tsumego_floors.csv (metadata `floor`; absent on older backends)
      - axes/per_problem.csv (correct-cell `embedding` in metadata)
      - shipped embedding_version (covers per_cell/*.npz content, which is
        regenerated with the embedding refit — those files are tid-named but
        their CONTENTS change on refit, so the version, not the filename set,
        is the signal — AND the bin-format rev, so a sidecar-layout change
        rebuilds shards without --force)
      - the SGF filename set (filenames ARE orientation-invariant content hashes,
        so a content change cannot occur without a filename change → the set hash
        detects add/remove/content)
      - ratings_root (output semantics)."""
    h = hashlib.sha256()
    rdir = os.path.join(PROJECT_ROOT, RATINGS_ROOT, str(board_size))
    _hash_file_into(h, os.path.join(rdir, RATINGS_CSV))
    _hash_file_into(h, os.path.join(rdir, SLOPE_CSV))
    _hash_file_into(h, os.path.join(rdir, FLOOR_CSV))
    embed_dir, _ = _axes_dir()
    if embed_dir:
        _hash_file_into(h, os.path.join(embed_dir, "per_problem.csv"))
        h.update((shipped_embedding_version(embed_dir) or "noemb").encode())
    tdir = tsumego_dir(board_size)
    names = sorted(f for f in os.listdir(tdir)
                   if f.endswith(".sgf")) if os.path.isdir(tdir) else []
    h.update(("\n".join(names)).encode())
    h.update(f"|root={RATINGS_ROOT}".encode())
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
      - ratings_map ← tsumego_ratings.csv `rating`: the raw black-box rating
        (the client converts it to a rank via the survey-calibrated curve; no rank
        baked in here).
      - slope_map   ← tsumego_slopes.csv `slope`: the raw 2PL slope a_p,
        shipped RAW (the client uses it as the success-curve slope and as the
        selection weight ∝ a_p²).
      - floor_map   ← tsumego_floors.csv `floor`: the raw 3PL guessing floor
        c_p (lower asymptote of the per-attempt success curve).
    Uses the shared utils.rating.ratings_io reader, so the shipped floats are
    correctly-rounded and consistent with the rest of the pipeline's loaders.
    (These values can differ from an older pandas-parsed bundle by ≤1 ULP, which
    re-hashes the shards — fine: this only runs when the shard set is rebuilt
    anyway, i.e. when a new ratings/SGF set has changed size_inputs_hash.)"""
    ratings_map = load_value_map(ratings_csv, "rating")

    slope_csv = os.path.join(ratings_dir, SLOPE_CSV)
    if not os.path.exists(slope_csv):
        raise FileNotFoundError(f"no slopes at {slope_csv} — run train_mirt")
    slope_map = load_value_map(slope_csv, "slope")

    floor_csv = os.path.join(ratings_dir, FLOOR_CSV)
    if not os.path.exists(floor_csv):
        raise FileNotFoundError(f"no floors at {floor_csv} — run train_mirt")
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


def process_size(board_size, output_dir, n_workers=None,
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
    n_workers → create_packages."""
    print(f"\n========== Packaging board size {board_size} ==========")

    corpus_dir = tsumego_dir(board_size)
    # The backend rating set is always ratings/ (combined MLP-2PL). Output
    # lands in the served pwa/tsumegos/ structure.
    RATINGS_DIR = os.path.join(PROJECT_ROOT, RATINGS_ROOT, str(board_size))
    PER_CELL_DIR = os.path.join(
        PROJECT_ROOT, "axes", "per_cell", str(board_size))

    # Every size in ALL_BOARD_SIZES is expected to have a full set of artifacts; a
    # missing one is a misconfigured run, not something to skip past.
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"no tsumegos dir at {corpus_dir}")
    ratings_csv = os.path.join(RATINGS_DIR, RATINGS_CSV)
    if not os.path.exists(ratings_csv):
        raise FileNotFoundError(f"no ratings at {ratings_csv} — run train_mirt")

    inputs_hash = size_inputs_hash(board_size)
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
    if _axes_dir()[0] is None:
        if have_zips:
            print("  embeddings/ absent but shards present — keeping shards, "
                  "embedding_meta and manifest untouched (curve/constants-only run)")
            return None, None, None
        raise FileNotFoundError(
            f"no embeddings at axes/ AND no existing "
            f"shards in {zip_dir} — run pipeline.train_mirt first")

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
    sgf_files = [f for f in os.listdir(corpus_dir) if f.endswith(".sgf")]
    print(f"  found {len(sgf_files)} SGFs in {corpus_dir}")
    if not sgf_files:
        raise RuntimeError(f"no SGFs in {corpus_dir}")
    all_metadata = assemble_metadata(
        sgf_files, board_size, ratings_map, slope_map, floor_map, embedding_map)
    print(f"  metadata built for all {len(all_metadata)} tsumegos (no gaps)")

    if not os.path.isdir(PER_CELL_DIR):
        raise FileNotFoundError(
            f"no per_cell dir at {PER_CELL_DIR} — run pipeline.write_mirt_per_cell")
    per_cell_arg = PER_CELL_DIR

    print(f"  building shards ...")
    create_packages(
        all_metadata, board_size, corpus_dir, output_dir,
        per_cell_dir=per_cell_arg, n_workers=n_workers,
    )
    size_dir = os.path.join(output_dir, 'tsumegos', str(board_size))
    print(f"  packages -> {size_dir}")

    print_rating_distribution(all_metadata)
    ratings_hash = compute_ratings_hash(all_metadata)
    print(f"  ratings_version[{board_size}]={ratings_hash}")
    return embedding_meta, ratings_hash, inputs_hash


# Folded into data_version ahead of the files. Packaging is deterministic, so
# re-running it over unchanged inputs reproduces the same data_version — which is
# right until a deploy has served DIFFERENT files under that value, after which
# clients hold the wrong metadata.json / lookup.txt under a key that will never
# move again by itself. Bumping this and repackaging is the one way to make every
# client re-fetch from identical inputs. Mirrored as DATA_VERSION_EPOCH in
# pwa/scripts/data-check.mjs.
DATA_VERSION_EPOCH = b"epoch-2"


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
    h.update(DATA_VERSION_EPOCH)
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
    # Fold the lookup-aligned correct-move embedding sidecar for the same reason —
    # and because its row alignment against lookup.txt is only guaranteed within
    # one packaging run, both must always cache-bust together.
    ce = os.path.join(output_dir, "tsumegos", "lookup.embeddings.bin")
    try:
        with open(ce, "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(b"missing-correct-embeddings")
    # The two other lookup-aligned sidecars, for the same reason: their row
    # alignment against lookup.txt holds only within one packaging run, and the
    # alt-stats masses are referenced to the correct-move vectors above, so all
    # three have to cache-bust together or a client pairs one run's masses with
    # another run's answer vector.
    for name in ("lookup.offsets.bin", "lookup.altstats.bin"):
        q = os.path.join(output_dir, "tsumegos", name)
        try:
            with open(q, "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(f"missing-{name}".encode())
    # And the SE-floor curve, which is derived from the same ratings + M: it feeds
    # the shown rank directly, so a change to it has to reach clients on the same
    # cache-bust as the ratings that moved it.
    sf = os.path.join(output_dir, "tsumegos", "se_floor.json")
    try:
        with open(sf, "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(b"missing-se-floor")
    return h.hexdigest()[:16]


def emit_problem_params(output_dir):
    """Single packed file for BOTH board sizes: tsumego_id -> (rating, slope, floor).
    16 bytes/problem — a 4-byte id (the 8-hex tsumego_id) followed by 3 little-endian
    float32 (rating, slope, floor). The PWA Worker (leaderboard rank recompute) and
    the client ratings-change backfill both read it into one Map for O(1) rating
    resolution, so neither has to walk the shard zips. ~2.6MB for ~165k problems.

    Re-reads the ratings CSVs directly (not the per-size in-memory maps, which are
    absent on a skipped-shard run), so it always covers every size. Every rated
    problem is written: a size without CSVs, a problem missing a slope or floor
    (load_size_value_maps) or an id that is not 8 hex digits raises, because a
    problem absent here is one the client re-derive and the Worker cannot price."""
    ratings_root = os.path.join(PROJECT_ROOT, RATINGS_ROOT)
    chunks = []
    for size in ALL_BOARD_SIZES:
        ratings, slopes, floors = load_size_value_maps(ratings_root, size)
        for tid, rating in ratings.items():
            try:
                id_bytes = bytes.fromhex(tid)
            except ValueError:
                id_bytes = b""
            if len(tid) != 8 or len(id_bytes) != 4:
                raise ValueError(f"ratings/{size}: {tid!r} is not an 8-hex tsumego id")
            vals = np.array([rating, slopes[tid], floors[tid]], dtype="<f4").tobytes()
            chunks.append(id_bytes + vals)
    out_path = os.path.join(output_dir, "tsumegos", "problem_params.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"".join(chunks))
    print(f"  wrote problem_params.bin ({len(chunks)} problems, "
          f"{len(chunks) * 16} bytes) at {out_path}")
    return out_path


# --- SE-floor curve --------------------------------------------------------------
# The rank estimator's display and serve offsets both scale with the EXCESS posterior
# se over an irreducible floor: the se a full window of AT-the-user's-rank problems
# would still leave. That floor is a property of the CORPUS at a rank — the local mean
# per-attempt Fisher information — so it is measured here, once, over every problem
# this run ships, and read straight off a curve by the client and the Worker.
#
# The alternative is for each reader to measure it from the problems it happens to
# hold, which is what makes it a shipped artifact rather than a runtime scan: the
# client holds only the shards loaded this session (clustered around the SERVE rank,
# which sits above the mean the floor is priced at) while the Worker holds the whole
# params map, so the two disagree on the same history — and the client's own value
# jumps as shards land. Both are display-visible: the shown rank is mean − 2·excess,
# so an error in the floor is twice that in stones, and a floor that moves under a
# The SE-floor geometry (kernel width, the measurability bar, the corpus read and
# the edge/interpolation rules) lives in utils.rating.se_floor, because
# fit_population_constants CHOOSES the rank grid with exactly these and this step
# EMITS the curve on it. One definition, so the two cannot drift apart.


def emit_se_floor_curve(output_dir):
    """pwa/tsumegos/se_floor.json — the corpus's mean per-attempt information as a
    function of internal rank, on the SAME grid as population_constants' rank_prior.

    That grid is not a coincidence and not a tunable: windowMLE evaluates the posterior
    on the prior's grid and returns a weighted mean over it, so the rank handed to the
    floor lookup can never fall outside it. Sharing the grid means the client's
    interpExtrap never extrapolates, and there is no end-behaviour convention (clamp?
    extend?) for anyone to get wrong — a linear extension of interpolated information
    can go negative, and a non-positive information yields an INFINITE floor, which
    silently collapses the excess to zero and shows the raw posterior mean.

    Ships mean INFORMATION, not the floor: information is the quantity that averages,
    and seFloorForMeanInfo's 1/sqrt is applied by the reader after interpolation.
    Emitted from the same ratings CSVs as problem_params.bin, every run, before
    compute_data_version so its bytes cache-bust with everything else.

    The curve's shape IS the floor's shape: information is a_p²/M'², so it is highest
    (the floor tightest) wherever a rank spans the most rating — M's two ends — and
    lowest where ranks are cheapest. Whatever structure M has is reproduced here
    faithfully rather than smoothed away: the per-attempt information the estimator
    itself accumulates has the same structure, so flattening it here would make the
    floor disagree with the se it is a floor ON. Do not widen the kernel to hide any
    of it — see se_floor.SE_FLOOR_MIN_EFFECTIVE."""
    ratings_root = os.path.join(PROJECT_ROOT, RATINGS_ROOT)
    curve_path = os.path.join(output_dir, "tsumegos", "rating_to_rank.json")
    with open(curve_path) as f:
        curves = json.load(f)
    pop_path = os.path.join(output_dir, "tsumegos", "population_constants.json")
    with open(pop_path) as f:
        pop = json.load(f)
    prior = pop["rank_prior"]
    lapse = float(pop["fat_finger_rate"])
    rank_lo, rank_step = float(prior["rank_lo"]), float(prior["rank_step"])
    n_knots = len(prior["log_density"])

    # The client's own per-attempt transform (shared/rank-window.js
    # slopeInRankSpace + js/irt.js anchorInfo), evaluated so the shipped curve is
    # in the units the reader accumulates.
    r, inf = se_floor.corpus_rank_information(
        ratings_root, curves, lapse, ALL_BOARD_SIZES)
    if r.size == 0:
        raise RuntimeError("no rated problems found — cannot emit se_floor.json")

    grid = rank_lo + rank_step * np.arange(n_knots)
    # Raises if an edge is unmeasurable even absorbing, or an interior gap is too
    # wide to bridge. fit_population_constants derived this grid against the same
    # call, so a failure here means the two artifacts are out of step — re-run it.
    mean_info, n_eff, interpolated = se_floor.mean_info_curve(grid, r, inf)
    for rank_x, ne_x in interpolated:
        print(f"    NOTE knot {rank_x:+.1f} had only {ne_x:.1f} effective problems; "
              f"interpolated from its neighbours")
    out = {
        "rank_lo": rank_lo,
        "rank_step": rank_step,
        "mean_info": [float(v) for v in mean_info],
        # Provenance only; no reader branches on these.
        "bandwidth": se_floor.SE_FLOOR_BANDWIDTH,
        "n_problems": int(len(r)),
    }
    out_path = os.path.join(output_dir, "tsumegos", "se_floor.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # The floor itself, for the eyeball check — it is what the offsets are built from.
    floors = 1.0 / np.sqrt(_win_prior_info(prior) + _WIN_STEADY_WEIGHT
                           * np.asarray(mean_info))
    print(f"  wrote se_floor.json ({n_knots} knots, rank {grid[0]:.1f}..{grid[-1]:.1f} "
          f"step {rank_step}, {len(r)} problems)")
    print(f"    floor {floors.min():.3f}..{floors.max():.3f}, "
          f"effective sample per knot {min(n_eff):.0f}..{max(n_eff):.0f}")
    return out_path


# Mirrors of the client's math, so the emitted curve is in the reader's units.
# pwa/shared/rank-window.js: WIN_CAP / the positional ramp's steady-state weight,
# and the per-attempt slope conversion; pwa/js/irt.js: anchorInfo.
_WIN_CAP = 250
_WIN_STEADY_WEIGHT = (_WIN_CAP + 1) / 2
_CALIBRATION_P = 0.8


def _win_prior_info(prior):
    sd = float(prior["sd"])
    return 1.0 / (sd * sd)


def emit_correct_embeddings(output_dir):
    """Correct-move embedding sidecar ALIGNED TO lookup.txt: row i is the K-dim
    fp16 correct-cell embedding of the problem on lookup.txt line i (lookup is
    sorted by id and covers both sizes, so the line index doubles as the row
    address). The PWA already fetches + parses lookup.txt in every flow that
    needs embeddings, so a backfill Range-fetches the one ~K*2-byte row per
    problem instead of downloading the ~2MB shard zip whose metadata.json
    carries the same vector. No keys, no offsets, no per-shard state — and no
    shard rebuild: like problem_params.bin this is re-emitted from the source
    CSV every run.

    Row alignment is the whole contract: MUST be emitted after the final
    lookup.txt write of the run, and both files fold into data_version so a
    client can never cache a mismatched pair (the client additionally checks
    the file size against its line count via Content-Range). An id missing
    from per_problem.csv (frozen-shard corpus drift) gets an ALL-ZEROS row,
    which the client treats as absent."""
    lookup_path = os.path.join(output_dir, "lookup.txt")
    with open(lookup_path, "r", encoding="utf-8") as f:
        lookup_ids = [line.split()[0] for line in f if line.strip()]

    per_problem = os.path.join(PROJECT_ROOT, "axes", "per_problem.csv")
    df = pd.read_csv(per_problem)
    embed_cols = sorted(
        [c for c in df.columns if c.startswith("embed_")],
        key=lambda c: int(c.split("_")[1]),
    )
    vec_by_id = {
        tid: np.asarray(vals, dtype=EMBEDDING_DTYPE)
        for tid, vals in zip(df["tsumego_id"], df[embed_cols].to_numpy())
    }

    K = len(embed_cols)
    zeros = np.zeros(K, dtype=EMBEDDING_DTYPE)
    missing = 0
    rows = []
    for tid in lookup_ids:
        vec = vec_by_id.get(tid)
        if vec is None:
            vec = zeros
            missing += 1
        rows.append(vec.tobytes())

    out_path = os.path.join(output_dir, "tsumegos", "lookup.embeddings.bin")
    with open(out_path, "wb") as f:
        f.write(b"".join(rows))
    print(f"  wrote lookup.embeddings.bin ({len(rows)} rows x {K * EMBEDDING_BYTES_PER_FLOAT} B"
          f"{f', {missing} missing ids zero-filled' if missing else ''}) at {out_path}")
    return out_path


# --- lookup-aligned shard-block sidecars -----------------------------------
#
# Two more files row-aligned to lookup.txt, alongside lookup.embeddings.bin.
# Both exist so a reader that holds NO shard can still resolve a problem: the
# per-cell byte offsets otherwise live only in embeddings_index.json INSIDE the
# 2MB zip, which made "read one 780-byte block" cost a whole shard download.
#
# They are rebuilt from the shards on disk every run rather than carried in
# memory out of create_packages, because packaging is incremental PER SIZE: a
# run that rebuilds only 9x9 must still emit rows for 13x13, and the shards it
# did not touch are the only remaining record of their own offsets.

OFFSETS_BYTES_PER_ROW = 5   # <u4 offset + u1 n_cells


def _read_shard_index(output_dir, size, zip_hash):
    """embeddings_index.json out of one shard zip: {tid: {offset, n_cells}}.
    Returns None when the shard has no index (a `just rezip the SGFs` run built
    it with embed_cells disabled). Reads the one member, not the archive."""
    zip_path = os.path.join(output_dir, "tsumegos", str(size), f"{zip_hash}.zip")
    try:
        with zipfile.ZipFile(zip_path) as z:
            return json.loads(z.read("embeddings_index.json"))["problems"]
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _read_shard_variation_visits(output_dir, size, zip_hash, tids):
    """Per-root-variation VISIT COUNTS for `tids`, out of the shard's own SGFs.

    Returns {tid: int64 array}, one entry per root variation in exactly the order
    project_cells_for_problem wrote the per-cell block — parse_tsumego_sgf's
    variation_nodes, pass row included — so visits[i] weighs block row i.

    The SGFs are read from the shard ZIP, not from tsumegos/<size>/: the zip is
    the artifact the client holds, and the live-path null reads its comments. Two
    producers reading different copies of a comment could disagree while both
    look correct, which is the one failure this sidecar exists to rule out.

    Visits go through utils.katago.add_variations.parse_eval_visits, which owns
    the convention that a comment carrying no third field is a 30-visit value —
    never a regex here, or that convention would have two homes.

    A tid whose SGF is not in the zip raises: the shard's own index named it, so
    an absent member is corruption, not a coverage gap.
    """
    zip_path = os.path.join(output_dir, "tsumegos", str(size), f"{zip_hash}.zip")
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for tid in tids:
            game = sgfmill.sgf.Sgf_game.from_bytes(z.read(tid + ".sgf"))
            _, variation_nodes = parse_tsumego_sgf(game)
            visits = []
            for node in variation_nodes:
                try:
                    comment = node.get("C")
                except KeyError:
                    comment = None
                visits.append(parse_eval_visits(comment))
            out[tid] = np.asarray(visits, dtype=np.int64)
    return out


def _lookup_rows(output_dir):
    """lookup.txt as [(tsumego_id, zip_hash, size)] in file (id-sorted) order —
    the row order every lookup-aligned sidecar is addressed by."""
    lookup_path = os.path.join(output_dir, "lookup.txt")
    rows = []
    with open(lookup_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tid, zip_hash, size_str = line.split()
            rows.append((tid, zip_hash, int(size_str)))
    return rows


def _group_rows_by_shard(rows):
    """Map (size, zip_hash) -> [(row_index, tsumego_id)], so each shard is
    opened once however many of its problems the caller wants."""
    by_shard = {}
    for i, (tid, zip_hash, size) in enumerate(rows):
        by_shard.setdefault((size, zip_hash), []).append((i, tid))
    return by_shard


def emit_lookup_offsets(output_dir):
    """Where each problem's per-cell block lives, ALIGNED TO lookup.txt: row i
    is (uint32 byte offset, uint8 n_cells) for the problem on lookup.txt line i,
    inside its own shard's .embeddings.bin. The shard is already named by
    lookup.txt's zipHash column, so offset + cell count is all that was missing.

    5 bytes/problem. A reader holding this plus lookup.txt can address any
    problem's block without opening a single zip — which is the whole point,
    since the in-zip embeddings_index.json costs a ~2MB download to read.

    An id whose shard carries no index gets an ALL-ZERO row (offset 0,
    n_cells 0); n_cells == 0 is the absent sentinel, since every real problem
    has at least one variation."""
    rows = _lookup_rows(output_dir)
    buf = bytearray(len(rows) * OFFSETS_BYTES_PER_ROW)
    missing = 0
    shards_without_index = 0
    for (size, zip_hash), items in _group_rows_by_shard(rows).items():
        index = _read_shard_index(output_dir, size, zip_hash)
        if index is None:
            shards_without_index += 1
            missing += len(items)
            continue
        for row_i, tid in items:
            entry = index.get(tid)
            if entry is None:
                missing += 1
                continue
            offset = int(entry["offset"])
            n_cells = int(entry["n_cells"])
            # Both widths are load-bearing: a silent wraparound here addresses
            # a stranger's bytes rather than failing, so bound them explicitly.
            if not 0 <= offset < 2 ** 32:
                raise ValueError(f"{tid}: block offset {offset} exceeds uint32")
            if not 0 < n_cells < 256:
                raise ValueError(f"{tid}: n_cells {n_cells} outside 1..255")
            struct.pack_into("<IB", buf, row_i * OFFSETS_BYTES_PER_ROW,
                             offset, n_cells)

    if shards_without_index and shards_without_index == len(_group_rows_by_shard(rows)):
        print("  skipping lookup.offsets.bin (no shard carries an embeddings "
              "index — per-cell embeddings were not built this run)")
        return None

    out_path = os.path.join(output_dir, "tsumegos", "lookup.offsets.bin")
    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"  wrote lookup.offsets.bin ({len(rows)} rows x {OFFSETS_BYTES_PER_ROW} B"
          f"{f', {missing} absent zero-filled' if missing else ''}) at {out_path}")
    return out_path


def _js_quantize(a):
    """quantizeEmbedding (pwa/js/math-utils.js) exactly: round to 2 decimals.

    JS Math.round is floor(x + 0.5) — it breaks ties toward +infinity, where
    numpy.round breaks them to even. fp16 carries plenty of values that land on
    an exact .5 after scaling (0.125 -> 12.5), and the counts below are strict
    per-axis comparisons, so a tie resolved the other way moves a count. Do the
    arithmetic in float64, which is what a JS Number is."""
    return np.floor(np.asarray(a, dtype=np.float64) * 100.0 + 0.5) / 100.0


def alt_stat_masses(q, visits):
    """The availability null for ONE problem: per axis, the summed VISITS of the
    alternatives loading strictly above / below the answer's own loading.

    `q` is the problem's per-cell block already quantized to the client's own
    resolution — (n_cells, K), SGF root-variation order — and `visits` is one
    visit count per those same rows. Returns (above, below), int64 of length K.

    An alternative is weighed by the visits its evaluation rests on rather than
    counted, because the moves a click could pick are not equally pickable and
    the search spends its depth where the position invited it. The weights are
    the integer visit counts themselves, so both masses are exact integers: the
    client's live path sums the same integers off the same comments, and the two
    producers can only agree or differ — never drift apart by a rounding.

    Row 0 IS the answer and is the reference. It ties with itself on every axis,
    so the strict comparisons drop it with no special case — and every other tie
    drops the same way, WITH its weight: a tie carries no direction, so it
    belongs to neither side rather than to both. The pass placeholder (the
    all-zeros row) is excluded before weighing; a click cannot select it.
    """
    ref = q[0]
    keep = ~np.all(q == 0.0, axis=1)
    kept = q[keep]
    w = np.asarray(visits, dtype=np.int64)[keep][:, None]
    above = ((kept > ref) * w).sum(axis=0)
    below = ((kept < ref) * w).sum(axis=0)
    return above, below


def emit_alt_stats(output_dir):
    """The availability null per problem, ALIGNED TO lookup.txt: row i holds,
    for the problem on lookup.txt line i, the summed VISITS of the moves a click
    could pick that load strictly ABOVE then strictly BELOW the correct move's
    loading, per axis. 2*K uint32 per problem.

    The quantities are MASSES, not counts: an alternative weighs the visits its
    own evaluation rests on (the third field of its SGF comment; a comment with
    no third field is a 30-visit value by the convention parse_eval_visits owns).
    A reader still forms p = above / (above + below) and reads it as before —
    what changed is which alternatives that ratio gives weight to.

    This is computeAltStats (pwa/js/picked-embedding.js) evaluated at packaging
    time. It is a property of the CORPUS, not of the player — the masses depend
    only on the problem and its answer — so computing it here costs the client
    the whole shard walk it currently pays to rediscover the same numbers.

    The reference is the block's OWN first row — the correct move's cell in the
    same per-cell projection the alternatives are drawn from. The nominally
    equivalent vectors in per_problem.csv (hence lookup.embeddings.bin and each
    shard's metadata.json) are not bit-identical to it, and referencing one of
    those compares cells across two projections, moving counts on a few percent
    of problems (`ablations.motif_null_figures` measures both rates against the
    shipped artifacts). Row 0 also matches what the client computes on a live
    wrong attempt, which reads the answer's own row out of the same block, so
    the two producers agree exactly rather than nearly.

    Row 0 IS the answer: generation enforces the answer as the first root
    variation and every backend stage reads it positionally. The pass-row
    assertion below is the cheap corpus-wide check of that — a problem whose
    first variation is the pass would have a misordered SGF, and the masses
    would be referenced to a placeholder.

    Ties carry no direction and are outside both masses, mirroring the
    observed-side tie drop — a tie drops WITH its weight, rather than being
    split between the sides — which is also why the correct row needs no special
    exclusion: it ties with itself on every axis and falls out of the strict
    comparisons. The pass placeholder (all-zeros row) is skipped: a click cannot
    select the pass. An id whose shard or block is absent gets an ALL-ZERO row,
    which the client already reads as `no null available` (altAbove + altBelow
    == 0 fails motif-stats' nAvail guard).

    The weights are read from each shard's OWN SGFs, so this pays an sgfmill
    parse per problem that the shard build already paid separately. Reading the
    zip rather than tsumegos/<size>/ is what keeps the weights identical to the
    ones the client will read out of the archive it downloaded."""
    rows = _lookup_rows(output_dir)
    correct_path = os.path.join(output_dir, "tsumegos", "lookup.embeddings.bin")
    n_floats = os.path.getsize(correct_path) // EMBEDDING_BYTES_PER_FLOAT
    if n_floats % len(rows) != 0:
        raise ValueError(
            f"lookup.embeddings.bin holds {n_floats} floats, not a whole number "
            f"of rows for {len(rows)} lookup lines — refusing to emit an "
            f"alt-stats sidecar against a lookup it is not aligned to")
    K = n_floats // len(rows)

    out = np.zeros((len(rows), 2 * K), dtype=ALTSTATS_DTYPE)
    missing = 0
    for (size, zip_hash), items in _group_rows_by_shard(rows).items():
        index = _read_shard_index(output_dir, size, zip_hash)
        bin_path = os.path.join(output_dir, "tsumegos", str(size),
                                f"{zip_hash}.embeddings.bin")
        if index is None or not os.path.exists(bin_path):
            missing += len(items)
            continue
        blob = np.fromfile(bin_path, dtype=EMBEDDING_DTYPE)
        visits_by_tid = _read_shard_variation_visits(
            output_dir, size, zip_hash, [tid for _, tid in items if tid in index])
        for row_i, tid in items:
            entry = index.get(tid)
            if entry is None:
                missing += 1
                continue
            start = int(entry["offset"]) // EMBEDDING_BYTES_PER_FLOAT
            n_cells = int(entry["n_cells"])
            block = blob[start:start + n_cells * K]
            if block.size != n_cells * K:
                raise ValueError(
                    f"{tid}: shard {zip_hash} bin is short for its own index "
                    f"({block.size} floats, expected {n_cells * K}) — refusing "
                    f"to ship masses read past the end of a block")
            visits = visits_by_tid[tid]
            # The weights are addressed positionally against the block, so a
            # length disagreement means the SGF in the zip is not the tree the
            # block was projected from — every weight would then be on the wrong
            # cell while both files stayed individually well-formed.
            if visits.size != n_cells:
                raise ValueError(
                    f"{tid}: shard SGF has {visits.size} root variations against "
                    f"{n_cells} rows in its per-cell block — refusing to weigh "
                    f"cells by visits that belong to different variations")
            q = _js_quantize(block.reshape(n_cells, K))
            if np.all(q[0] == 0.0):
                raise ValueError(
                    f"{tid}: first root variation is the pass placeholder — the "
                    f"SGF's variations are misordered and the null would be "
                    f"referenced to a row that is not a move")
            above, below = alt_stat_masses(q, visits)
            # A mass is (visits per variation) x (variations), so unlike the
            # counts it replaced it is not bounded by the uint8 cell count —
            # a silent wraparound here would ship a null pointing the wrong way.
            if max(above.max(initial=0), below.max(initial=0)) > 2 ** 32 - 1:
                raise ValueError(f"{tid}: alt mass exceeds uint32")
            out[row_i, :K] = above
            out[row_i, K:] = below

    if missing == len(rows):
        print("  skipping lookup.altstats.bin (no shard carries per-cell blocks)")
        return None

    out_path = os.path.join(output_dir, "tsumegos", "lookup.altstats.bin")
    out.tofile(out_path)
    print(f"  wrote lookup.altstats.bin ({len(rows)} rows x "
          f"{2 * K * ALTSTATS_BYTES_PER_QUANTITY} B, {ALTSTATS_DTYPE_NAME} masses"
          f"{f', {missing} absent zero-filled' if missing else ''}) at {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package rated tsumegos for the PWA.")
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
            size, OUTPUT_DIR, n_workers=args.workers,
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

    # The corpus's information-vs-rank curve, which the rank estimator's display and
    # serve offsets are measured against. Reads the curve + population constants just
    # copied above, and the same ratings CSVs as problem_params — so it follows the
    # same sentinel rule: with a size's shards frozen at an older rating era, a curve
    # re-derived from the current CSVs would describe a corpus that isn't shipped.
    print("\n--- SE-floor curve ---")
    se_floor_path = os.path.join(OUTPUT_DIR, "tsumegos", "se_floor.json")
    if any_sentinel and os.path.exists(se_floor_path):
        print("  keeping existing se_floor.json (a size took the "
              "embeddings-absent sentinel path; shards frozen)")
    else:
        emit_se_floor_curve(OUTPUT_DIR)

    # Correct-move embedding sidecar, row-aligned to the (now final) lookup.txt.
    # Same sentinel rule: with embeddings/ absent the source CSV is gone and
    # lookup.txt is unchanged from the prior run, so the existing sidecar still
    # aligns — keep it rather than emitting a mismatched one.
    print("\n--- correct-move embeddings ---")
    ce_path = os.path.join(OUTPUT_DIR, "tsumegos", "lookup.embeddings.bin")
    if _axes_dir()[0] is None or any_sentinel:
        print("  keeping existing lookup.embeddings.bin (embeddings/ absent or "
              "a size took the sentinel path)"
              if os.path.exists(ce_path) else
              "  skipping lookup.embeddings.bin (embeddings/ absent, no existing file)")
    else:
        emit_correct_embeddings(OUTPUT_DIR)

    # Where each problem's per-cell block sits, and the availability null read off
    # it. Both are row-aligned to the same final lookup.txt as the correct-move
    # sidecar. Offsets are rebuilt from the shard zips on disk, so they stay right
    # for a size whose shards were not rebuilt this run; the alt-stats masses are
    # referenced to the correct-move sidecar, so they follow ITS sentinel rule —
    # keep the pair in lockstep or the masses describe a different answer vector
    # than the one shipped beside them.
    print("\n--- per-cell block offsets + availability null ---")
    if _axes_dir()[0] is None or any_sentinel:
        print("  keeping existing lookup.offsets.bin / lookup.altstats.bin "
              "(embeddings/ absent or a size took the sentinel path)")
    else:
        emit_lookup_offsets(OUTPUT_DIR)
        emit_alt_stats(OUTPUT_DIR)

    # Single shared embedding_meta.json. Carries both joint embedding info and
    # per-size ratings_version (content hash) so the PWA can independently
    # detect embedding refits vs rating refreshes.
    if joint_embedding_meta is not None or ratings_versions:
        meta_out = joint_embedding_meta if joint_embedding_meta is not None else {}
        K = meta_out.get("K") or 0
        meta_out["dtype"] = EMBEDDING_DTYPE_NAME
        meta_out["bytes_per_cell"] = int(K) * EMBEDDING_BYTES_PER_FLOAT
        # Layout of the two block sidecars, so a reader never hardcodes either
        # width: lookup.offsets.bin is <u4 offset + u1 n_cells per problem,
        # lookup.altstats.bin is K little-endian uint32 above-masses then K
        # below-masses (summed visits, not counts — see alt_stat_masses).
        meta_out["offsets_bytes_per_row"] = OFFSETS_BYTES_PER_ROW
        meta_out["altstats_bytes_per_row"] = (
            2 * int(K) * ALTSTATS_BYTES_PER_QUANTITY)
        meta_out["altstats_dtype"] = ALTSTATS_DTYPE_NAME
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