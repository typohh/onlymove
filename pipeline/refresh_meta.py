"""Recompute joint metadata (ratings_version) from existing per-shard
metadata.json files inside pwa/tsumegos/<size>/*.zip — no full pipeline
re-run required.

Use this when you want to trigger client-side answerHistory.problemRating
backfill without repackaging zips. Two scenarios:

1. Bootstrapping the feature: ratings_version was just added to the
   embedding_meta.json schema, but the production zips were built before
   the field existed. Running this once injects the field; clients then
   start tracking it on their next load.

2. Manual re-trigger: if you've edited ratings out-of-band (rare) or want
   to force every client to re-fetch their answerHistory ratings, run with
   --salt <token> to perturb the hash without touching shard contents.

Normal pipeline runs (package_for_pwa.py) already write ratings_version
automatically — this script is the surgical alternative."""
import argparse
import hashlib
import json
import os
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES
from pipeline.package_for_pwa import compute_ratings_hash


def load_size_metadata(output_dir, board_size):
    zip_dir = os.path.join(output_dir, "tsumegos", str(board_size))
    if not os.path.isdir(zip_dir):
        return []
    items = []
    for f in sorted(os.listdir(zip_dir)):
        if not f.endswith(".zip"):
            continue
        with zipfile.ZipFile(os.path.join(zip_dir, f), "r") as zf:
            try:
                with zf.open("metadata.json") as mf:
                    items.extend(json.load(mf))
            except KeyError:
                continue
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salt", default=None,
                        help="Optional string appended to each per-size hash "
                             "input. Use to force a client refresh without "
                             "actually changing ratings.")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "pwa")
    meta_path = os.path.join(output_dir, "tsumegos", "embedding_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    ratings_versions = {}
    for size in ALL_BOARD_SIZES:
        items = load_size_metadata(output_dir, size)
        if not items:
            print(f"  size {size}: no shards found, skipping")
            continue
        h = compute_ratings_hash(items)
        if args.salt:
            h = hashlib.sha256(f"{h}|{args.salt}".encode()).hexdigest()[:16]
        ratings_versions[str(size)] = h
        print(f"  size {size}: {len(items)} problems, ratings_version={h}")

    if not ratings_versions:
        print("nothing to write")
        return

    meta["ratings_version"] = ratings_versions
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nWrote {meta_path}")


if __name__ == "__main__":
    main()
