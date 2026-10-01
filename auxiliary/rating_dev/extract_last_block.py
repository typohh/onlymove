#!/usr/bin/env python3
"""Single-layer correct-cell activations, and the position-global vectors the
per-cell store never holds, for the layer comparison arms
(`--act-source` in ablations.ablation_mirt).

The production store holds every block's channels through the frozen encoder.
A block's reported output is its INCREMENT to the trunk, so the encoded store is
eighteen increments side by side and "the last layer" has two readings, both
written here at the correct cell, raw:

  trunk_final   the trunk state the heads read — every increment summed, then
                the final norm and activation
  last_block    the final block's increment alone

each as the two-file roll-up the joint PCA reads
(activations/_signal_cache_<source>/{correct_vecs,correct_ids}_<size>), over
EXACTLY the production roll-up's id list, so an arm on any source rates the
same problems. The same pass writes three per-POSITION vectors beside them, in
activations/_signal_cache_global/ and row-aligned to the same ids — the final
trunk's board mean and board max, and the value head's hidden layer — which no
arm reads yet. The forward pass, the position replay and the cell enumeration
are pipeline.extract_activations' own.

    ./.venv/bin/python3 -u -m auxiliary.rating_dev.extract_last_block

A size whose roll-ups all cover the production id list is skipped; a problem
that fails to prepare aborts the run.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pipeline.extract_activations import (  # noqa: E402
    BATCH, DEFAULT_CKPT, PARSE_WORKERS, batch_forward, prepare_problem)
from katago.train.load_model import load_model  # noqa: E402
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.features import (  # noqa: E402
    ACT_SOURCE_DIRS, _cache_ids_path, _cache_vecs_path)

GLOBAL_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache_global")
GLOBALS = ("trunk_mean", "trunk_max", "value_hidden")
# Problems the readers prepare before the GPU takes them. The two alternate
# rather than overlap: reader threads left running through the forward passes
# hold the interpreter, and the pass starves behind them.
CHUNK = BATCH * 16


def _paths(size, last_name):
    cell = {"trunk_final": _cache_vecs_path(size, "trunk_final"),
            last_name: _cache_vecs_path(size, "last_block")}
    glob = {g: os.path.join(GLOBAL_DIR, f"{g}_{size}.npy") for g in GLOBALS}
    ids = [_cache_ids_path(size, "trunk_final"),
           _cache_ids_path(size, "last_block"),
           os.path.join(GLOBAL_DIR, f"ids_{size}.json")]
    return cell, glob, ids


def main():
    logging.disable(logging.CRITICAL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for d in (ACT_SOURCE_DIRS["trunk_final"], ACT_SOURCE_DIRS["last_block"],
              GLOBAL_DIR):
        os.makedirs(d, exist_ok=True)
    for size in ALL_BOARD_SIZES:
        ids = json.load(open(_cache_ids_path(size)))
        in_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        model, _, _ = load_model(DEFAULT_CKPT, use_swa=False, device=device,
                                 pos_len=size, verbose=False)
        model.eval()
        last_name = f"rconv{len(model.blocks)}.out"
        cell, glob, id_paths = _paths(size, last_name)
        outputs = list(cell.values()) + list(glob.values())
        if (all(os.path.exists(p) for p in outputs + id_paths)
                and all(json.load(open(p)) == ids for p in id_paths)):
            print(f"[size {size}] up to date ({len(ids)} problems)", flush=True)
            continue
        print(f"[size {size}] {len(ids)} problems: trunkfinal, {last_name}, "
              f"{', '.join(GLOBALS)}", flush=True)

        hidden = []
        hook = model.value_head.act2.register_forward_hook(
            lambda _m, _i, o: hidden.append(o))
        bufs, start, t0 = {}, 0, time.time()

        def put(name, rows):
            rows = rows.float().half().cpu().numpy()
            if name not in bufs:
                bufs[name] = np.empty((len(ids), rows.shape[1]), np.float16)
            bufs[name][start:start + len(rows)] = rows

        with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as ex:
            for c in range(0, len(ids), CHUNK):
                items = list(ex.map(
                    lambda t: prepare_problem(
                        os.path.join(in_dir, t + ".sgf"), model.config,
                        model.pos_len), ids[c:c + CHUNK]))
                for b in range(0, len(items), BATCH):
                    batch = items[b:b + BATCH]
                    hidden.clear()
                    final, last = batch_forward(
                        model, ["trunkfinal", last_name],
                        [x[0] for x in batch], [x[1] for x in batch])
                    xy = [x[2][x[3]] for x in batch]
                    j = torch.arange(len(batch), device=final.device)
                    xs = torch.tensor([x for x, _ in xy], device=final.device)
                    ys = torch.tensor([y for _, y in xy], device=final.device)
                    put("trunk_final", final[j, :, ys, xs])
                    put(last_name, last[j, :, ys, xs])
                    put("trunk_mean", final.float().mean(dim=(2, 3)))
                    put("trunk_max", final.float().amax(dim=(2, 3)))
                    put("value_hidden", hidden[0])
                    start += len(batch)
                print(f"  [size {size}] {start}/{len(ids)} "
                      f"({start / (time.time() - t0):.0f}/s)", flush=True)
        hook.remove()
        if start != len(ids) or not all(
                np.isfinite(v).all() for v in bufs.values()):
            raise RuntimeError(f"size {size}: incomplete or non-finite roll-up")
        for name, path in {**cell, **glob}.items():
            tmp = path + ".tmp.npy"
            np.save(tmp, bufs[name])
            os.replace(tmp, path)
            print(f"[size {size}] wrote {name} {bufs[name].shape} -> {path}",
                  flush=True)
        for p in id_paths:
            with open(p, "w") as f:
                json.dump(ids, f)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
