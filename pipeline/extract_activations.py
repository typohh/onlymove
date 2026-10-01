"""Extract the activation network's trunk state at every legal cell of each tsumego.

For each SGF in tsumegos/<size>/:
  - Parse SGF, replay setup moves to obtain the board state the solver faces.
  - Enumerate EVERY legal cell for the player to move (the color of the correct
    variation), in board-scan order — sorted by (y, x). This is a *superset* of
    the SGF's root variations (the generator prunes those down), so the cell set
    is independent of SGF document order. Downstream maps cells by (x, y) and the
    correct cell by correct_idx, never by row position — so order need only be
    self-consistent with xys/correct_idx, which scan order guarantees.
  - Run one forward pass through the activation network
    (utils.katago.models.ACTIVATION_NETWORK), under the corpus's own rules
    and komi (utils.constants RULES/KOMI, the values the generator stamps into
    every SGF and queries the analysis engine with). Rules reach the net through
    its global inputs — komi, scoring, ko, suicide and tax — so reading a
    position under rules it was not evaluated with changes the activations
    while leaving every array shape intact, i.e. silently.
  - Capture the FINAL TRUNK STATE (`trunkfinal`): every block's increment summed
    onto the stem, then the final norm and activation — the tensor the
    network's own heads read. A block's reported output is its increment alone,
    which is why no single block stands in for it.
  - Slice it at every legal cell's (x, y) and save to
    activations/<size>/<id>.npz with:
      candidates  (n_legal, ACT_STORE_DIM) float16 -- per-cell trunk channels,
                                                      board-scan order
      xys         (n_legal, 2) int32   -- KataGo (x, y) per row (scan order)
      correct_idx ()           int32   -- scan-order index of the correct cell
      source_fp   ()           uint32  -- which network and tensor wrote the file

Downstream consumers: pipeline.train_mirt reads the correct-cell row (through utils.rating.features) via
candidates[correct_idx]; package_for_pwa re-parses the SGF and remaps cells into
SGF-variation order by (x, y) (the npz being a superset of the SGF's non-pass
variations is what makes that remap total).

Resumable: skips problems where the output .npz already exists (--force overrides).
A CHANGE OF NETWORK IS TOTAL: every reader trusts the directory to be one
source's output, a resumed run aborts on a file stamped by another, and the
id-keyed caches downstream cannot see a same-width change — so it ends with
`cache_rating_features --rebuild-activation-cache`.
Fail-loud: a problem that fails to process is a hard error that aborts the run,
never skipped — a missing id would silently desync every downstream cache from
the tsumego set.
"""

import argparse
import hashlib
import logging
import os
import queue
import sys
import threading
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import sgfmill.sgf
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils.katago.torch_model  # noqa: F401  -- adds vendored katago/ to sys.path

from katago.game.board import Board
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.train.load_model import load_model
from katago.train.model_pytorch import ExtraOutputs

from utils.katago.sgf_parsing import parse_tsumego_sgf
from utils.constants import ALL_BOARD_SIZES, KOMI, RULES
from utils.katago.coords import gtp_to_xy, sgf_to_gtp
from utils.katago.models import ACTIVATION_NETWORK, get_checkpoint
from utils.rating.model_constants import ACT_STORE_DIM

TRUNK_OUTPUT = "trunkfinal"


def source_fingerprint(network=ACTIVATION_NETWORK, tensor=TRUNK_OUTPUT) -> int:
    """What a stored file was read from, as a uint32: the network and the tensor."""
    return int.from_bytes(
        hashlib.sha256(f"{network}:{tensor}".encode()).digest()[:4], "big")


# The vendored net takes rules as a dict, while the rest of the pipeline names
# them as the analysis engine's strings (utils.constants). Map rather than
# restate, so the one place RULES/KOMI are set governs extraction too, and fail
# loud on a name with no preset instead of silently reading the corpus under
# rules it was never evaluated with.
_RULES_PRESETS = {
    "japanese": GameState.RULES_JAPANESE,
    "chinese": GameState.RULES_CHINESE,
    "tromp-taylor": GameState.RULES_TT,
}


def katago_rules() -> dict:
    """`utils.constants` RULES/KOMI in the vendored GameState's dict form."""
    preset = _RULES_PRESETS.get(RULES)
    if preset is None:
        raise ValueError(
            f"No vendored GameState preset for RULES={RULES!r}; "
            f"known: {sorted(_RULES_PRESETS)}")
    return {**preset, "whiteKomi": KOMI}


def build_state(setup_moves, size: int) -> GameState:
    # Setup stones are REPLAYED AS MOVES, matching how the analysis engine is
    # handed the same position (utils.katago.queries.query sends them as `moves`
    # with an empty `initialStones`). Under territory scoring the net's komi
    # input is offset by the per-colour move counts, so the two agree only
    # because both sides count these stones the same way — do not switch this to
    # a stone-placement setup without re-checking that offset.
    state = GameState(size, katago_rules())
    for color_char, gtp in setup_moves:
        xy = gtp_to_xy(gtp)
        if xy is None:
            continue
        x, y = xy
        loc = state.board.loc(x, y)
        pla = Board.BLACK if color_char.lower() == "b" else Board.WHITE
        state.play(pla, loc)
    return state


def batch_forward(model, block_names, bins, globs):
    """One forward pass for a BATCH of problems; returns the captured block
    outputs as device tensors, (B, C, H, W) each. Batching is the first of the
    three things that make extraction GPU-bound instead of CPU-bound (with
    on-device slicing and threaded parse/write): a b18 forward is
    milliseconds, so per-problem calls spend the run in Python overhead."""
    extra = ExtraOutputs(block_names)
    with torch.no_grad():
        model(
            torch.tensor(np.concatenate(bins), dtype=torch.float32,
                         device=model.device),
            torch.tensor(np.concatenate(globs), dtype=torch.float32,
                         device=model.device),
            input_meta=None,
            extra_outputs=extra,
        )
    missing = [n for n in block_names if n not in extra.returned]
    if missing:
        raise RuntimeError(
            f"Block outputs not captured: {missing}. "
            f"First few available: {extra.available[:6]}"
        )
    return [extra.returned[name] for name in block_names]


def vecs_at_cells_gpu(trunk, j, xys) -> np.ndarray:
    """Problem j's per-cell vectors, sliced ON DEVICE so only the legal cells
    cross to the CPU. Returns float16 (n_cells, channels). fp16 precision loss
    is well below the z-scoring noise in pipeline.train_mirt."""
    dev = trunk.device
    xs = torch.tensor([x for x, _ in xys], dtype=torch.long, device=dev)
    ys = torch.tensor([y for _, y in xys], dtype=torch.long, device=dev)
    return trunk[j][:, ys, xs].T.float().half().cpu().numpy()


def _prepare_task(task):
    return prepare_problem(*task)


def prepare_problem(sgf_path: str, model_config, pos_len):
    """The CPU half of one problem — parse, replay, enumerate cells, build the
    net inputs. Runs in reader PROCESSES: it is pure Python, and as threads the
    readers hold the interpreter against the forward pass, whose many short
    calls each wait to take it back."""
    with open(sgf_path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    setup_moves, variation_nodes = parse_tsumego_sgf(game)

    correct_color, correct_mb = variation_nodes[0].get_raw_move()
    pla = Board.BLACK if correct_color.lower() == "b" else Board.WHITE
    correct_xy = gtp_to_xy(sgf_to_gtp(correct_mb.decode("utf-8"), size))
    if correct_xy is None:
        raise ValueError(f"{sgf_path}: correct variation is a pass; cannot embed")

    state = build_state(setup_moves, size)
    candidate_xys = enumerate_legal_cells(state.board, pla, size)
    try:
        correct_idx = candidate_xys.index(correct_xy)
    except ValueError:
        raise ValueError(
            f"{sgf_path}: correct move {correct_xy} is not a legal cell "
            f"(setup-replay or coordinate mismatch)")

    bin_input, global_input = state.get_input_features(
        Features(model_config, pos_len))
    return bin_input, global_input, candidate_xys, correct_idx


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


BATCH = 16                       # problems per forward pass
PARSE_WORKERS = 4                # SGF parse + input-feature processes
WRITE_WORKERS = 4                # compressed-npz writer threads
PREFETCH = 64                    # prepared problems the readers may run ahead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N problems (debugging).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the output .npz exists.")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: cuda if available).")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "activations"),
                        help="Root to write <size>/<id>.npz under.")
    args = parser.parse_args()

    ckpt = get_checkpoint(ACTIVATION_NETWORK)
    src_fp = source_fingerprint()
    print(f"activation network {ACTIVATION_NETWORK}, {TRUNK_OUTPUT}: "
          f"fingerprint {src_fp:08x}", flush=True)

    logging.disable(logging.CRITICAL)  # silence KataGo's verbose config dump

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    block_names = [TRUNK_OUTPUT]
    print(f"Processing sizes: {list(ALL_BOARD_SIZES)}", flush=True)

    totals = {"done": 0, "skipped": 0}
    for size in ALL_BOARD_SIZES:
        in_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        out_dir = os.path.join(args.out_dir, str(size))
        if not os.path.isdir(in_dir):
            print(f"\n[size {size}] no tsumegos dir at {in_dir}; skipping", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)

        sgf_files = sorted(f for f in os.listdir(in_dir) if f.endswith(".sgf"))
        if args.limit is not None:
            sgf_files = sgf_files[: args.limit]
        print(f"\n[size {size}] Found {len(sgf_files)} SGFs in {in_dir}", flush=True)

        print(f"[size {size}] Loading model on {device}: {ckpt}", flush=True)
        model, _, _ = load_model(ckpt, use_swa=False, device=device,
                                 pos_len=size, verbose=False)
        model.eval()

        # Split the worklist by existence up front, then fingerprint-verify the
        # existing files in a background thread WHILE extraction runs: a
        # resumed run must never leave a file from another network in the
        # store — every reader trusts the directory to be ONE source's output
        # — but the check is pure reading and need not serialize with the GPU.
        todo, existing = [], []
        for fname in sgf_files:
            out_path = os.path.join(out_dir, fname[:-4] + ".npz")
            if not args.force and os.path.exists(out_path):
                existing.append(out_path)
            else:
                todo.append(fname)
        verify_err = []

        def _verify_existing():
            for path in existing:
                with np.load(path) as d:
                    old_fp = int(d["source_fp"]) if "source_fp" in d.files else None
                if old_fp != src_fp:
                    verify_err.append(
                        f"{path}: written from source "
                        f"{'unstamped' if old_fp is None else f'{old_fp:08x}'}, "
                        f"current is {src_fp:08x} — remove the stale store "
                        f"before resuming")
                    return

        verifier = None
        if existing:
            verifier = threading.Thread(target=_verify_existing, daemon=True)
            verifier.start()
        print(f"[size {size}] {len(todo)} to extract, {len(existing)} existing"
              + (" (fingerprints verifying in background)" if verifier else ""),
              flush=True)

        # Reader processes run the CPU half (parse, replay, input features)
        # ahead of the GPU, PREFETCH problems at a time so the lookahead stays
        # bounded; a writer pool compresses and renames behind it. Any
        # failure anywhere is a hard abort, never a skipped problem — a missing
        # id desyncs every downstream cache from the tsumego set.
        q = queue.Queue(maxsize=PREFETCH)

        def _read_all():
            try:
                with ProcessPoolExecutor(
                        max_workers=PARSE_WORKERS,
                        mp_context=multiprocessing.get_context("spawn")) as ex:
                    for s in range(0, len(todo), PREFETCH):
                        chunk = todo[s:s + PREFETCH]
                        tasks = [(os.path.join(in_dir, f), model.config,
                                  model.pos_len) for f in chunk]
                        for fname, item in zip(chunk, ex.map(
                                _prepare_task, tasks, chunksize=4)):
                            q.put((fname, item))
                q.put(None)
            except BaseException as e:  # surfaces in the consumer, fail-loud
                q.put(e)

        threading.Thread(target=_read_all, daemon=True).start()

        n_done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as writers:
            pending = []

            def _write(out_path, vecs, xys, correct_idx):
                tmp = out_path + ".tmp.npz"
                np.savez_compressed(tmp, candidates=vecs,
                                    xys=np.array(xys, dtype=np.int32),
                                    correct_idx=np.int32(correct_idx),
                                    source_fp=np.uint32(src_fp))
                os.replace(tmp, out_path)

            batch = []
            drained = False
            while not drained or batch:
                if not drained and len(batch) < BATCH:
                    got = q.get()
                    if isinstance(got, BaseException):
                        raise RuntimeError(
                            f"[size {size}] failed to prepare a problem") from got
                    if got is None:
                        drained = True
                    else:
                        batch.append(got)
                        continue
                if not batch:
                    break
                trunk = batch_forward(
                    model, block_names,
                    [b[1][0] for b in batch], [b[1][1] for b in batch])[0]
                if trunk.shape[1] != ACT_STORE_DIM:
                    raise RuntimeError(
                        f"{TRUNK_OUTPUT} has {trunk.shape[1]} channels, the "
                        f"store is {ACT_STORE_DIM} wide")
                for j, (fname, (_, _, xys, ci)) in enumerate(batch):
                    vecs = vecs_at_cells_gpu(trunk, j, xys)
                    out_path = os.path.join(out_dir, fname[:-4] + ".npz")
                    pending.append(writers.submit(_write, out_path, vecs, xys, ci))
                    n_done += 1
                    if n_done <= 3 or n_done % 500 == 0:
                        rate = n_done / max(time.time() - t0, 1e-9)
                        stat_mean = vecs.mean(dtype=np.float32)
                        stat_std = vecs.std(dtype=np.float32)
                        print(f"  [size {size}] {n_done}/{len(todo)} "
                              f"({rate:.1f}/s); last: {fname} -> "
                              f"{vecs.shape[0]} cells × {vecs.shape[1]} feats "
                              f"correct_idx={ci} "
                              f"sample stats mean={stat_mean:+.3f} "
                              f"std={stat_std:.3f}", flush=True)
                batch = []
                if verify_err:
                    raise RuntimeError(verify_err[0])
                done, pending = [f for f in pending if f.done()], \
                    [f for f in pending if not f.done()]
                for f in done:
                    f.result()      # re-raise a writer failure promptly
            for f in pending:
                f.result()
        if verifier is not None:
            verifier.join()
            if verify_err:
                raise RuntimeError(verify_err[0])

        print(f"[size {size}] Done: {n_done} extracted, {len(existing)} "
              f"skipped.", flush=True)
        totals["done"] += n_done
        totals["skipped"] += len(existing)

        # Free GPU memory before loading the next size's model.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll sizes done: {totals['done']} extracted, "
          f"{totals['skipped']} skipped.")


if __name__ == "__main__":
    main()
