"""Re-encode the raw activation store through the frozen encoder.

Reads raw activations/<size>/<tid>.npz (candidates (n, raw_dim) fp16) and
writes the encoded store to --dst/<size>/<tid>.npz with:

    candidates   (n, ACT_ENCODER_DIM) float16 — encoder-projected vectors
    xys          (n, 2) int32                 — unchanged
    correct_idx  ()  int32                    — unchanged
    encoder_fp   ()  uint32                   — the encoder's 4-byte fingerprint

This is the BOOTSTRAP path, for a store whose raw files already exist; the
steady-state producer of encoded files is extraction itself, which projects at
write time so the raw vectors never touch disk. The raw source is left in
place — it is the input to the acceptance gates, and removing it afterwards is
a deliberate, separate act.

Resumable: a destination file whose stored encoder_fp matches the current
encoder is skipped; one written under a DIFFERENT encoder is a hard error, not
a skip — re-encoding into a store that mixes fingerprints would poison every
reader that trusts the directory to be one encoder's output (delete the stale
destination tree and re-run instead). Any unreadable source file aborts the
run: a silently missing problem would desync every downstream cache.

Usage:
  ./.venv/bin/python3 -m pipeline.encode_activations [--dst activations/enc]
      [--src activations] [--workers 8]
"""

import argparse
import os
import sys
from multiprocessing import Pool

# One BLAS thread per worker, set BEFORE numpy loads OpenBLAS: the per-file
# GEMMs are small, and N workers each spawning a full team of spinning BLAS
# threads oversubscribes every core into a measured order-of-magnitude stall.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import ALL_BOARD_SIZES
from utils.rating.activation_encoder import apply_encoder, fingerprint, load_encoder

_ENC = None
_FP = None


def _init(encoder_path):
    global _ENC, _FP
    _ENC = load_encoder(encoder_path)
    _FP = fingerprint(_ENC)


def _encode_one(job):
    src, dst = job
    if os.path.exists(dst):
        with np.load(dst) as existing:
            fp = int(existing["encoder_fp"])
        if fp == _FP:
            return "skipped"
        raise RuntimeError(
            f"{dst}: written under encoder {fp:08x}, current is {_FP:08x} — "
            f"a mixed-fingerprint store must not exist; delete the stale "
            f"destination tree and re-run")
    with np.load(src) as z:
        enc_cand = apply_encoder(_ENC, np.asarray(z["candidates"]))
        xys = np.asarray(z["xys"])
        ci = np.int32(z["correct_idx"])
    # The temp name must end in .npz — np.savez appends it to a name that
    # lacks it, and the atomic rename then targets a file never created.
    tmp = dst + ".tmp.npz"
    np.savez_compressed(tmp, candidates=enc_cand, xys=xys, correct_idx=ci,
                        encoder_fp=np.uint32(_FP))
    os.replace(tmp, dst)
    return "written"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(PROJECT_ROOT, "activations"))
    ap.add_argument("--dst", default=os.path.join(PROJECT_ROOT, "activations", "enc"))
    ap.add_argument("--encoder", default=None,
                    help="encoder npz (default: <src>/encoder.npz)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    encoder_path = args.encoder or os.path.join(args.src, "encoder.npz")
    _init(encoder_path)
    print(f"encoder {encoder_path}: fingerprint {_FP:08x}, "
          f"M={_ENC['C'].shape[0]} from raw {_ENC['C'].shape[1]}")

    jobs = []
    for size in ALL_BOARD_SIZES:
        sdir = os.path.join(args.src, str(size))
        ddir = os.path.join(args.dst, str(size))
        if not os.path.isdir(sdir):
            print(f"size {size}: no source dir ({sdir}); skipping")
            continue
        os.makedirs(ddir, exist_ok=True)
        n = 0
        for e in os.scandir(sdir):
            if e.name.endswith(".npz"):
                jobs.append((e.path, os.path.join(ddir, e.name)))
                n += 1
        print(f"size {size}: {n} problems")

    written = skipped = 0
    with Pool(args.workers, initializer=_init, initargs=(encoder_path,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_encode_one, jobs, chunksize=64)):
            written += result == "written"
            skipped += result == "skipped"
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(jobs)} ({written} written, {skipped} up to date)",
                      flush=True)
    print(f"done: {written} written, {skipped} already up to date, {len(jobs)} total")


if __name__ == "__main__":
    main()
