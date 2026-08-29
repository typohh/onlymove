"""Fit the frozen activation encoder (raw trunk vectors -> ACT_ENCODER_DIM).

The fit set is the distribution the encoder must represent — the cells that
ship or get compared against, weighted toward what players actually face and
choose — in three equal thirds:

  uniform  the correct cell of problems drawn uniformly over the corpus
  served   the correct cell of problems drawn by attempt count (the served
           distribution; sampled without replacement, weighted)
  picked   distinct human-picked wrong cells from tsumego_reports, resolved
           from their GTP coordinate against the npz (x, y) map

Rows are read from raw per-problem npz files (candidates wider than
ACT_ENCODER_DIM — an encoded store aborts the fit). The BOOTSTRAP fit reads
the full raw store; a future REFRESH extracts a raw subset somewhere and
points --act-dir at it — the frames restrict themselves to the files that
exist, so it is the same code path. Either way the result overwrites
activations/encoder.npz (TRACKED — with the raw store gone, nothing can
regenerate it), and every re-encode of the corpus must follow, because a store
written under the previous encoder no longer matches the new fingerprint.

Usage:
  ./.venv/bin/python3 -m pipeline.fit_activation_encoder [--n-third 20000]
      [--act-dir activations] [--out activations/encoder.npz]
"""

import argparse
import os
import sqlite3
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.constants import ALL_BOARD_SIZES
from utils.katago.coords import gtp_to_xy
from utils.rating.activation_encoder import (
    ENCODER_PATH, fingerprint_hex, fit_encoder, save_encoder)
from utils.rating.model_constants import ACT_ENCODER_DIM, SEED

DB_PATH = os.path.join(PROJECT_ROOT, "db", "d1.sqlite")

# A picked coordinate that is not a legal cell of the current SGF can only be a
# report predating corpus surgery on its problem. A trickle is history; a flood
# means the frame and the report log have diverged and the fit must not proceed.
MAX_UNRESOLVED_FRACTION = 0.01


def build_manifest(act_dirs, n_third, rng):
    """[(size, tid, xy-or-None)] — xy None means the problem's correct cell."""
    frame = []
    for size, d in act_dirs.items():
        if os.path.isdir(d):
            frame.extend((size, e.name[:-4]) for e in os.scandir(d)
                         if e.name.endswith(".npz"))
    if not frame:
        raise SystemExit(f"no activation npz files under {sorted(act_dirs.values())}")
    have = set(frame)
    print(f"frame: {len(frame)} problems ("
          + ", ".join(f"{s}: {sum(1 for z in frame if z[0] == s)}" for s in act_dirs) + ")")

    con = sqlite3.connect(DB_PATH)
    attempts, picked = {}, []
    for tid, size, n in con.execute(
            "SELECT problem, size, COUNT(*) FROM tsumego_reports GROUP BY problem, size"):
        if (size, tid) in have:
            attempts[(size, tid)] = n
    for tid, size, move in con.execute(
            "SELECT DISTINCT problem, size, move FROM tsumego_reports "
            "WHERE success=0 AND move IS NOT NULL AND move != ''"):
        xy = gtp_to_xy(move)  # None = a pass pick, which has no cell row
        if xy is not None and (size, tid) in have:
            picked.append((size, tid, xy))
    con.close()
    print(f"attempted problems in frame: {len(attempts)}; distinct wrong picks: {len(picked)}")

    manifest = []
    idx = rng.choice(len(frame), size=min(n_third, len(frame)), replace=False)
    manifest += [(*frame[i], None) for i in idx]

    akeys = list(attempts)
    if akeys:
        w = np.array([attempts[k] for k in akeys], dtype=np.float64)
        idx = rng.choice(len(akeys), size=min(n_third, len(akeys)),
                         replace=False, p=w / w.sum())
        manifest += [(*akeys[i], None) for i in idx]

    if picked:
        idx = rng.choice(len(picked), size=min(n_third, len(picked)), replace=False)
        manifest += [picked[i] for i in idx]

    return manifest


def read_rows(act_dirs, manifest):
    """One npz open per problem; a missing or unreadable file is a hard error
    (the manifest was built from the same directory listing moments ago)."""
    by_file = {}
    for size, tid, xy in manifest:
        by_file.setdefault((size, tid), []).append(xy)

    rows, unresolved = [], 0
    t0 = time.time()
    for n_done, ((size, tid), wants) in enumerate(by_file.items()):
        path = os.path.join(act_dirs[size], f"{tid}.npz")
        with np.load(path) as z:
            cand, xys, ci = z["candidates"], z["xys"], int(z["correct_idx"])
            if cand.shape[1] <= ACT_ENCODER_DIM:
                raise SystemExit(
                    f"{path}: candidates width {cand.shape[1]} <= {ACT_ENCODER_DIM} — "
                    f"this is an ENCODED store; the encoder fits on raw activations only")
            for xy in wants:
                if xy is None:
                    rows.append(cand[ci])
                else:
                    hit = np.where((xys[:, 0] == xy[0]) & (xys[:, 1] == xy[1]))[0]
                    if len(hit) == 0:
                        unresolved += 1
                    else:
                        rows.append(cand[int(hit[0])])
        if (n_done + 1) % 5000 == 0:
            rate = (n_done + 1) / (time.time() - t0)
            print(f"  read {n_done + 1}/{len(by_file)} files ({rate:.0f}/s, "
                  f"eta {(len(by_file) - n_done - 1) / rate / 60:.1f} min)", flush=True)

    if unresolved > MAX_UNRESOLVED_FRACTION * len(manifest):
        raise SystemExit(
            f"{unresolved}/{len(manifest)} picked coordinates resolve to no legal "
            f"cell — the report log and the activation frame have diverged")
    if unresolved:
        print(f"skipped {unresolved} picked coordinates with no legal cell "
              f"(reports predating corpus surgery)")
    return np.array(rows, dtype=np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-third", type=int, default=20000)
    ap.add_argument("--act-dir", default=os.path.join(PROJECT_ROOT, "activations"),
                    help="directory holding raw <size>/<tid>.npz (a refresh points "
                         "this at its freshly extracted subset)")
    ap.add_argument("--out", default=ENCODER_PATH)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    act_dirs = {s: os.path.join(args.act_dir, str(s)) for s in ALL_BOARD_SIZES}
    manifest = build_manifest(act_dirs, args.n_third, rng)
    print(f"manifest: {len(manifest)} rows")

    X = read_rows(act_dirs, manifest)
    print(f"fit matrix: {X.shape[0]} x {X.shape[1]}; fitting M={ACT_ENCODER_DIM}...",
          flush=True)
    enc = fit_encoder(X.astype(np.float32), ACT_ENCODER_DIM, seed=SEED)
    save_encoder(enc, args.out)
    print(f"wrote {args.out}")
    print(f"  fingerprint: {fingerprint_hex(enc)}")
    print(f"  cumulative explained variance at M={ACT_ENCODER_DIM}: {enc['evr']:.4f}")


if __name__ == "__main__":
    main()
