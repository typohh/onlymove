"""Extract per-block trunk activations at every legal cell for each tsumego.

For each SGF in tsumegos/<size>/:
  - Parse SGF, replay setup moves to obtain board state.
  - Walk root-level variation nodes — utils/add_variations.add_shallow_variations
    emits one variation per legal move via queries.get_legal_moves, so SGF
    document order IS the legal-move enumeration. Using that order means the
    PWA can index cells by variation position without recomputing legality.
  - Run one forward pass through KataGo's b18 net.
  - Capture post-skip-add output of all 18 residual blocks (rconv1.out..rconv18.out).
  - Slice each block at every variation's first-move (x, y); concatenate blocks
    -> one 6912-float vector per legal cell.
  - Save to activations/<size>/<id>.npz with:
      candidates  (n_legal, 6912) float16  -- per-cell vector, SGF variation order
      xys         (n_legal, 2)    int32    -- KataGo (x, y) per row (matches order)
      correct_idx ()              int32    -- index of the correct cell; always 0

Downstream (build_embedding / package_for_pwa) decides which candidates to ship.

Resumable: skips problems where the output .npz already exists. Legacy
correct-cell-only .npy files are not auto-skipped by this script — they live
alongside the new .npz until manually cleaned up. build_embedding.py reads
either format (.npz preferred).
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import sgfmill.sgf
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils.katago_torch  # noqa: F401  -- adds vendored katago/ to sys.path

from katago.game.board import Board
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.train.load_model import load_model
from katago.train.model_pytorch import ExtraOutputs

from pipeline.benchmark_bots import parse_tsumego_sgf, sgf_to_gtp
from utils.configurations import ALL_BOARD_SIZES

DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "models",
    "kata1-b18c384nbt-s9996604416-d4316597426",
    "model.ckpt",
)

GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # I-skipped, matching KataGo / GTP convention


def gtp_to_xy(gtp_coord: str):
    if gtp_coord.lower() == "pass":
        return None
    col_letter = gtp_coord[0].upper()
    x = GTP_COLS.index(col_letter)
    y = int(gtp_coord[1:]) - 1  # GTP row 1 -> y=0 (bottom)
    return x, y


def build_state(setup_moves, size: int) -> GameState:
    state = GameState(size, GameState.RULES_TT.copy())
    for color_char, gtp in setup_moves:
        xy = gtp_to_xy(gtp)
        if xy is None:
            continue
        x, y = xy
        loc = state.board.loc(x, y)
        pla = Board.BLACK if color_char.lower() == "b" else Board.WHITE
        state.play(pla, loc)
    return state


def extract_block_activations(state: GameState, model, block_names):
    features = Features(model.config, model.pos_len)
    bin_input, global_input = state.get_input_features(features)
    extra = ExtraOutputs(block_names)
    with torch.no_grad():
        model(
            torch.tensor(bin_input, dtype=torch.float32, device=model.device),
            torch.tensor(global_input, dtype=torch.float32, device=model.device),
            input_meta=None,
            extra_outputs=extra,
        )
    missing = [n for n in block_names if n not in extra.returned]
    if missing:
        raise RuntimeError(
            f"Block outputs not captured: {missing}. "
            f"First few available: {extra.available[:6]}"
        )
    return {name: extra.returned[name][0].cpu().numpy() for name in block_names}


def vecs_at_cells(activations_by_name, block_names, xys) -> np.ndarray:
    """Returns (n_cells, sum_C) float16: per-cell vector with block channels
    concatenated in `block_names` order (same layout as the previous
    single-cell vec_at_cell so PCA loadings stay compatible). fp16 halves
    disk relative to fp32; precision loss is well below the z-scoring noise
    in build_embedding.
    """
    xs = np.array([x for x, _ in xys], dtype=np.int64)
    ys = np.array([y for _, y in xys], dtype=np.int64)
    parts = []
    for name in block_names:
        a = activations_by_name[name]  # (C, H, W); H=y, W=x
        parts.append(a[:, ys, xs])  # (C, n_cells)
    return np.concatenate(parts, axis=0).T.astype(np.float16)  # (n_cells, sum_C)


def process_one_sgf(sgf_path: str, model, block_names):
    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    setup_moves, variation_nodes = parse_tsumego_sgf(game)

    # The generator emits one root variation per legal move (see
    # utils/add_variations.add_shallow_variations). Iterate in SGF document
    # order so downstream cell indexing matches what the client sees when
    # parsing the same SGF — no client-side legality recomputation needed.
    candidate_xys = []
    for i, node in enumerate(variation_nodes):
        _, mb = node.get_raw_move()
        xy = gtp_to_xy(sgf_to_gtp(mb.decode("utf-8"), size))
        if xy is None:
            # Pass variation: never embedded. Packaging (project_cells_for_problem)
            # also maps cells by (x, y) and drops the pass, so skip it here rather than
            # failing extraction — consumers read by correct_idx / (x, y), not position.
            continue
        candidate_xys.append(xy)
    # variation_nodes[0] is the correct move by generator convention.
    correct_idx = 0

    state = build_state(setup_moves, size)
    activations = extract_block_activations(state, model, block_names)
    vecs = vecs_at_cells(activations, block_names, candidate_xys)  # (n_legal, 6912)
    xys = np.array(candidate_xys, dtype=np.int32)
    return vecs, xys, correct_idx, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N problems (debugging).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the output .npy exists.")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: cuda if available).")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help="Path to KataGo PyTorch checkpoint.")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)  # silence KataGo's verbose config dump

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    block_names = [f"rconv{i}.out" for i in range(1, 19)]
    print(f"Capturing {len(block_names)} block outputs (rconv1..rconv18)", flush=True)
    print(f"Processing sizes: {list(ALL_BOARD_SIZES)}", flush=True)

    totals = {"done": 0, "skipped": 0, "errored": 0}
    for size in ALL_BOARD_SIZES:
        in_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        out_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
        if not os.path.isdir(in_dir):
            print(f"\n[size {size}] no tsumegos dir at {in_dir}; skipping", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)

        sgf_files = sorted(f for f in os.listdir(in_dir) if f.endswith(".sgf"))
        if args.limit is not None:
            sgf_files = sgf_files[: args.limit]
        print(f"\n[size {size}] Found {len(sgf_files)} SGFs in {in_dir}", flush=True)

        print(f"[size {size}] Loading model on {device}: {args.ckpt}", flush=True)
        model, _, _ = load_model(args.ckpt, use_swa=False, device=device,
                                 pos_len=size, verbose=False)
        model.eval()

        n_done = n_skipped = n_errored = 0
        t0 = time.time()
        for fname in sgf_files:
            tsumego_id = fname[:-4]
            out_path = os.path.join(out_dir, tsumego_id + ".npz")
            if not args.force and os.path.exists(out_path):
                n_skipped += 1
                continue
            try:
                vecs, xys, correct_idx, _ = process_one_sgf(
                    os.path.join(in_dir, fname), model, block_names)
            except Exception as e:
                n_errored += 1
                print(f"  [err] {fname}: {e}", flush=True)
                continue
            np.savez_compressed(
                out_path,
                candidates=vecs,
                xys=xys,
                correct_idx=np.int32(correct_idx),
            )
            n_done += 1
            if n_done <= 3 or n_done % 500 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                # Accumulate stats in fp32 — reducing fp16 directly overflows.
                stat_mean = vecs.mean(dtype=np.float32)
                stat_std = vecs.std(dtype=np.float32)
                print(f"  [size {size}] {n_done} done, {n_skipped} skipped, "
                      f"{n_errored} errored ({rate:.1f}/s); last: {fname} -> "
                      f"{vecs.shape[0]} cells × {vecs.shape[1]} feats "
                      f"correct_idx={correct_idx} "
                      f"sample stats mean={stat_mean:+.3f} std={stat_std:.3f}",
                      flush=True)

        print(f"[size {size}] Done: {n_done} extracted, {n_skipped} skipped, "
              f"{n_errored} errored.", flush=True)
        totals["done"] += n_done
        totals["skipped"] += n_skipped
        totals["errored"] += n_errored

        # Free GPU memory before loading the next size's model.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll sizes done: {totals['done']} extracted, "
          f"{totals['skipped']} skipped, {totals['errored']} errored.")


if __name__ == "__main__":
    main()
