"""Extract per-block trunk activations at every legal cell for each tsumego.

For each SGF in tsumegos/<size>/:
  - Parse SGF, replay setup moves to obtain the board state the solver faces.
  - Enumerate EVERY legal cell for the player to move (the color of the correct
    variation), in board-scan order — sorted by (y, x). This is a *superset* of
    the SGF's root variations (the generator prunes those down), so the cell set
    is independent of SGF document order. Downstream maps cells by (x, y) and the
    correct cell by correct_idx, never by row position — so order need only be
    self-consistent with xys/correct_idx, which scan order guarantees.
  - Run one forward pass through KataGo's b18 net.
  - Capture post-skip-add output of all 18 residual blocks (rconv1.out..rconv18.out).
  - Slice each block at every legal cell's (x, y); concatenate blocks
    -> one 6912-float vector per cell.
  - Save to activations/<size>/<id>.npz with:
      candidates  (n_legal, 6912) float16  -- per-cell vector, board-scan order
      xys         (n_legal, 2)    int32    -- KataGo (x, y) per row (scan order)
      correct_idx ()              int32    -- scan-order index of the correct cell

Downstream consumers: build_embedding reads the correct-cell row via
candidates[correct_idx]; package_for_pwa re-parses the SGF and remaps cells into
SGF-variation order by (x, y) (the npz being a superset of the SGF's non-pass
variations is what makes that remap total).

Resumable: skips problems where the output .npz already exists (--force overrides).
Fail-loud: a problem that fails to process is a hard error that aborts the run,
never skipped — a missing id would silently desync every downstream cache from
the tsumego set.
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
from utils.katago_queries import gtp_to_xy

DEFAULT_CKPT = os.path.join(
    PROJECT_ROOT, "models",
    "kata1-b18c384nbt-s9996604416-d4316597426",
    "model.ckpt",
)


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


def enumerate_legal_cells(board, pla, size):
    """Every empty point that is a legal move for `pla`, in board-scan order
    (y outer, x inner → sorted by (y, x)). This is the cell set the corpus
    embeds: a superset of the SGF's pruned root variations and independent of
    SGF document order, so downstream consumers index by (x, y) / correct_idx,
    never by row position."""
    cells = []
    for y in range(size):
        for x in range(size):
            loc = board.loc(x, y)
            if board.board[loc] == Board.EMPTY and board.would_be_legal(pla, loc):
                cells.append((x, y))
    return cells


def process_one_sgf(sgf_path: str, model, block_names):
    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    setup_moves, variation_nodes = parse_tsumego_sgf(game)

    # variation_nodes[0] is the correct move by generator convention; its color
    # is the player to move for the whole problem (and thus whose legal moves
    # define the candidate cells).
    correct_color, correct_mb = variation_nodes[0].get_raw_move()
    pla = Board.BLACK if correct_color.lower() == "b" else Board.WHITE
    correct_xy = gtp_to_xy(sgf_to_gtp(correct_mb.decode("utf-8"), size))
    if correct_xy is None:
        raise ValueError(f"{sgf_path}: correct variation is a pass; cannot embed")

    state = build_state(setup_moves, size)

    # Embed EVERY legal cell (a superset of the SGF variations), in scan order,
    # and record where the correct move lands in that scan. Note this is
    # independent of the SGF's variation set/order — the client recovers a
    # picked move's cell via its (x, y), and build_embedding via correct_idx.
    candidate_xys = enumerate_legal_cells(state.board, pla, size)
    try:
        correct_idx = candidate_xys.index(correct_xy)
    except ValueError:
        raise ValueError(
            f"{sgf_path}: correct move {correct_xy} is not a legal cell "
            f"(setup-replay or coordinate mismatch)")

    activations = extract_block_activations(state, model, block_names)
    vecs = vecs_at_cells(activations, block_names, candidate_xys)  # (n_legal, 6912)
    xys = np.array(candidate_xys, dtype=np.int32)
    return vecs, xys, correct_idx, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N problems (debugging).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the output .npz exists.")
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

    totals = {"done": 0, "skipped": 0}
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

        n_done = n_skipped = 0
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
                # A malformed / unembeddable problem is a hard error: abort the
                # whole run rather than silently dropping it from the activation
                # corpus (a missing id desyncs every downstream cache — the move
                # cache, correct_ids, the embedding — from the tsumego set).
                raise RuntimeError(
                    f"[size {size}] failed to extract activations for {fname}") from e
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
                print(f"  [size {size}] {n_done} done, {n_skipped} skipped "
                      f"({rate:.1f}/s); last: {fname} -> "
                      f"{vecs.shape[0]} cells × {vecs.shape[1]} feats "
                      f"correct_idx={correct_idx} "
                      f"sample stats mean={stat_mean:+.3f} std={stat_std:.3f}",
                      flush=True)

        print(f"[size {size}] Done: {n_done} extracted, {n_skipped} skipped.",
              flush=True)
        totals["done"] += n_done
        totals["skipped"] += n_skipped

        # Free GPU memory before loading the next size's model.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll sizes done: {totals['done']} extracted, "
          f"{totals['skipped']} skipped.")


if __name__ == "__main__":
    main()
