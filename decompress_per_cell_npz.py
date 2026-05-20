"""TEMPORARY one-off: rewrite per-cell embedding .npz files from DEFLATE to
STORED (uncompressed), in place, without recomputing the projections.

The projections are expensive (GPU activations -> PCA/ICA); the compression
codec is not. build_embedding now writes STORED; this back-converts the
already-shipped files so packaging stops paying inflate on ~20k loads.

Idempotent: files already fully STORED are skipped. Atomic per file
(temp + os.replace), so an interrupt can't leave a half-written .npz.

    ./.venv/bin/python3 decompress_per_cell_npz.py            # all sizes
    ./.venv/bin/python3 decompress_per_cell_npz.py --dry-run

Delete this script once the corpus is converted.
"""
import argparse
import glob
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

PER_CELL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "embeddings", "joint", "per_cell",
)


def already_stored(path):
    """True if every member of the npz is uncompressed (compress_type 0)."""
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        return bool(infos) and all(i.compress_type == zipfile.ZIP_STORED
                                   for i in infos)


def convert_one(path):
    """Re-save one npz as STORED. Returns (path, status)."""
    try:
        if already_stored(path):
            return path, "skip"
        with np.load(path) as d:
            arrays = {k: d[k] for k in d.files}
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)  # np.savez == ZIP_STORED
        os.replace(tmp, path)
        return path, "ok"
    except Exception as e:
        return path, f"fail:{type(e).__name__}:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts/sizes, write nothing")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(PER_CELL_ROOT, "*", "*.npz")))
    if not files:
        print(f"no .npz under {PER_CELL_ROOT}")
        return

    before = sum(os.path.getsize(f) for f in files)
    print(f"{len(files)} files, {before/1e6:.1f} MB on disk")

    if args.dry_run:
        n_stored = sum(already_stored(f) for f in files)
        print(f"dry-run: {n_stored} already STORED, "
              f"{len(files) - n_stored} would be converted")
        return

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, f) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            path, status = fut.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "fail":
                failures.append((path, status))
            if i % 1000 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  "
                      f"ok={counts['ok']} skip={counts['skip']} "
                      f"fail={counts['fail']}", flush=True)

    after = sum(os.path.getsize(f) for f in files)
    print(f"done: ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")
    print(f"size {before/1e6:.1f} MB -> {after/1e6:.1f} MB "
          f"({100*(after-before)/before:+.1f}%)")
    if failures:
        print("failures:", file=sys.stderr)
        for path, status in failures[:20]:
            print(f"  {path}: {status}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
