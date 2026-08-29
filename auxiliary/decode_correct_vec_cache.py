"""Arm B of the encoder-migration parity gate: a correct-vec cache decoded
from the 512-dim store.

Reads _signal_cache/correct_ids_<size>.json (the row order is the cache
contract) and writes a correct_vecs_<size>.npy of the SAME shape and dtype as
the raw one, but with every row reconstructed from the encoded store
(y @ C + pca_mean, un-standardized back to raw space). Swapping this file in
for the raw cache and re-running the trainer measures exactly "what if only
the encoded store existed" — the joint-PCA cache re-keys itself automatically
(it fingerprints the vec-cache files), and every other code path is untouched.

Usage:
  ./.venv/bin/python3 auxiliary/decode_correct_vec_cache.py --out-dir DIR
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import ALL_BOARD_SIZES
from utils.rating.activation_encoder import fingerprint, load_encoder

CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")
ENC_DIR = os.path.join(PROJECT_ROOT, "activations", "enc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="directory to write correct_vecs_<size>.npy into "
                         "(NOT _signal_cache — the swap is explicit)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    enc = load_encoder()
    fp = fingerprint(enc)
    C, pmean = enc["C"], enc["pca_mean"]
    m1, s1 = enc["m1"], enc["s1"]

    for size in ALL_BOARD_SIZES:
        ids = json.load(open(os.path.join(CACHE_DIR, f"correct_ids_{size}.json")))
        out = np.empty((len(ids), C.shape[1]), dtype=np.float16)
        for i, tid in enumerate(ids):
            path = os.path.join(ENC_DIR, str(size), f"{tid}.npz")
            with np.load(path) as z:
                if int(z["encoder_fp"]) != fp:
                    raise SystemExit(f"{path}: fingerprint mismatch")
                y = np.asarray(z["candidates"][int(z["correct_idx"])],
                               dtype=np.float32)
            out[i] = ((y @ C) + pmean) * s1 + m1
            if (i + 1) % 20000 == 0:
                print(f"  size {size}: {i + 1}/{len(ids)}", flush=True)
        dst = os.path.join(args.out_dir, f"correct_vecs_{size}.npy")
        np.save(dst, out)
        print(f"size {size}: wrote {dst} {out.shape}")


if __name__ == "__main__":
    main()
