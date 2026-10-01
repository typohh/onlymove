#!/usr/bin/env python3
"""Correct-cell activations of the LAST residual block alone, for the layer
comparison arm (`--act-source last_block` in ablations.ablation_mirt).

The production store holds every block's channels through the frozen encoder;
this writes the final block's raw channels at the correct cell and nothing
else, as the same two-file roll-up the joint PCA reads —
activations/_signal_cache_last_block/{correct_vecs,correct_ids}_<size>.{npy,json}
— over EXACTLY the id list of the production roll-up, so an arm on either
source rates the same problems. The forward pass, the position replay and the
cell enumeration are pipeline.extract_activations' own.

    ./.venv/bin/python3 -u -m auxiliary.rating_dev.extract_last_block

A size whose roll-up already covers the production id list is skipped; a
problem that fails to prepare aborts the run.
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
from utils.rating.features import _cache_ids_path, _cache_vecs_path  # noqa: E402

LAST_BLOCK = "rconv18.out"
SOURCE = "last_block"


def main():
    logging.disable(logging.CRITICAL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for size in ALL_BOARD_SIZES:
        ids = json.load(open(_cache_ids_path(size)))
        vecs_path = _cache_vecs_path(size, SOURCE)
        ids_path = _cache_ids_path(size, SOURCE)
        if (os.path.exists(vecs_path) and os.path.exists(ids_path)
                and json.load(open(ids_path)) == ids):
            print(f"[size {size}] up to date ({len(ids)} problems)", flush=True)
            continue
        os.makedirs(os.path.dirname(vecs_path), exist_ok=True)
        in_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        model, _, _ = load_model(DEFAULT_CKPT, use_swa=False, device=device,
                                 pos_len=size, verbose=False)
        model.eval()
        print(f"[size {size}] {len(ids)} problems, {LAST_BLOCK}", flush=True)

        out = None
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=PARSE_WORKERS) as ex:
            prepared = ex.map(
                lambda t: prepare_problem(os.path.join(in_dir, t + ".sgf"),
                                          model.config, model.pos_len),
                ids, chunksize=4)
            batch, start = [], 0
            for i, item in enumerate(prepared):
                batch.append(item)
                if len(batch) < BATCH and i + 1 < len(ids):
                    continue
                act = batch_forward(model, [LAST_BLOCK],
                                    [b[0] for b in batch],
                                    [b[1] for b in batch])[0]
                xy = [b[2][b[3]] for b in batch]
                j = torch.arange(len(batch), device=act.device)
                xs = torch.tensor([x for x, _ in xy], device=act.device)
                ys = torch.tensor([y for _, y in xy], device=act.device)
                rows = act[j, :, ys, xs].float().half().cpu().numpy()
                if out is None:
                    out = np.empty((len(ids), rows.shape[1]), dtype=np.float16)
                out[start:start + len(batch)] = rows
                start += len(batch)
                batch = []
                if start % (BATCH * 250) == 0:
                    print(f"  [size {size}] {start}/{len(ids)} "
                          f"({start / (time.time() - t0):.0f}/s)", flush=True)
        if start != len(ids) or not np.isfinite(out).all():
            raise RuntimeError(f"size {size}: incomplete or non-finite roll-up")
        tmp = vecs_path + ".tmp.npy"
        np.save(tmp, out)
        os.replace(tmp, vecs_path)
        with open(ids_path, "w") as f:
            json.dump(ids, f)
        print(f"[size {size}] wrote {out.shape} -> {vecs_path}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
