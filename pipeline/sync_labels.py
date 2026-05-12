"""Refresh the labels block inside pwa/tsumegos/embedding_meta.json
without rebuilding any zips. Use this after editing embeddings/joint/labels.json.

A full `package_for_pwa` run rezips every problem; this script touches only
the joint metadata file, which is what the PWA reads for label display.
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():
    joint_dir = os.path.join(PROJECT_ROOT, "embeddings", "joint")
    labels_path = os.path.join(joint_dir, "labels.json")
    if not os.path.exists(labels_path):
        raise SystemExit(f"no labels file at {labels_path}")
    with open(labels_path, "r") as f:
        labels = json.load(f)

    meta_path = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "embedding_meta.json")
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"no joint embedding metadata at {meta_path} — run package_for_pwa first"
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["labels"] = labels
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"updated labels in {meta_path}")


if __name__ == "__main__":
    main()
