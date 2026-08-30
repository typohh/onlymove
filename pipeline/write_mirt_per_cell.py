#!/usr/bin/env python3
"""Fan the axis projection out to every legal cell of every problem.

The client scores the move the user PICKED, not only the answer, so every legal
cell needs its own K-vector. Output matches what packaging reads:

    axes/per_cell/<size>/<tid>.npz    embeddings (n_cells, K) fp16 + xys

WHY THE TWO-STAGE SHORTCUT IS UNAVAILABLE. When the axis map is LINEAR on
features, a cell's shipped value is `problem_value + (z_cell - z_correct) @
delta`, and taking that difference annihilates every additive term — the
activation mean, the PCA mean, the post-PCA mean alike — while the rest is two
diagonal scalings and a rotation, all linear and composing. The entire projection
then folds into ONE matrix of K columns applied to the stored activations.

`pipeline.train_mirt` puts a TRUNK between the features and the loadings. It is
nonlinear, so differences no longer annihilate and nothing folds: every cell
needs the full stored-vector -> PCA_DIM projection and then the trunk.

THAT SOUNDS EXPENSIVE AND IS NOT. The projection over every cell of the corpus
is a batched matmul the GPU finishes in seconds; the run is bound by reading
and decompressing the activation store (the encoded per-problem npz), which is
CPU work. So the optimization that matters is READ PARALLELISM, and the GPU
side needs nothing clever.

  READER THREADS, not one. npz decompression releases the GIL, so threads scale
  on it, and a single reader leaves the GPU idle almost all the time.

  fp16 for the projection matmul — this GPU has no fast fp16 path, so the gain
  is bandwidth, not arithmetic.

  NOT IMPLEMENTED, and deliberately: the non-activation tail (bot winrate and
  the board-size flag) is constant per problem, so its contribution to trunk
  layer 1 could be computed once per problem instead of once per cell. That
  would mean bypassing nn.Sequential to reach inside the first layer, to shave a
  few percent off a term that is already not the bottleneck.

THAT TAIL IS SHORT BECAUSE OF THIS PASS. `PRODUCTION_BLOCKS` keeps the blocks
that describe the problem and drops the ones that describe its ANSWER (move
evals, ownership swing, game phase), because here the trunk is evaluated at every
legal cell against a tail that is the CORRECT move's — so an answer-derived block
conditions each alternative cell on the answer instead of on itself. Reading the
layout from the constant rather than naming one here is what keeps this pass and
`pipeline.train_mirt` on the same columns.

Run after `pipeline.train_mirt`, which writes the model this reads.
"""

import os
import queue
import sys
import threading
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.model_constants import (  # noqa: E402
    K_AXES, PCA_DIM, PRODUCTION_BLOCKS, SEED, SIZES)
from utils.rating.features import (  # noqa: E402
    SIZE_FLAG_COLS, corpus_tids_by_size, joint_feature_tail,
    load_joint_activation_basis, load_joint_blocks)
# The trunk's layer stack, stated once — it has to be rebuilt here EXACTLY as it
# was fitted, and a stack that matches by shape while differing in its activation
# loads cleanly and projects every cell through the wrong function.
from utils.rating.mirt_net import build_trunk  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "axes", "per_cell")
MODEL_PT = os.path.join(PROJECT_ROOT, "ratings", "mirt_model.pt")
PROBLEM_BATCH = 64               # problems per GPU batch; ~3-6k cells
READ_AHEAD = 8                   # batches the reader may run in front
READ_WORKERS = max(4, (os.cpu_count() or 4))   # npz decompression is the bottleneck


def _load_one(args):
    size, tid, act_dir = args
    path = os.path.join(act_dir, tid + ".npz")
    try:
        with np.load(path, allow_pickle=False) as d:
            return (size, tid, np.asarray(d["candidates"]),
                    np.asarray(d["xys"], dtype=np.int32))
    except FileNotFoundError:
        # MISSING DATA for a live corpus problem, not a skippable gap. Returning
        # None here shipped a shard whose per_cell/ silently omitted the problem
        # while per_problem.csv carried it.
        raise SystemExit(
            f"size {size}: {path} missing for a tsumego that IS in the corpus. "
            f"Run pipeline.extract_activations until it covers every SGF, then re-run.")


def _reader(tasks, act_dirs, q):
    """Fill the queue from a THREAD POOL. This is the whole performance story:
    the projection is seconds of GPU time and the run is bound by decompressing
    the activation store's npz. That work releases the GIL, so threads scale on
    it, and a single reader would leave the GPU idle for essentially the entire
    run."""
    from concurrent.futures import ThreadPoolExecutor
    batch = []
    with ThreadPoolExecutor(max_workers=READ_WORKERS) as ex:
        for r in ex.map(_load_one,
                        ((s, t, act_dirs[s]) for s, t in tasks),
                        chunksize=32):
            if r is None:
                continue
            batch.append(r)
            if len(batch) >= PROBLEM_BATCH:
                q.put(batch); batch = []
    if batch:
        q.put(batch)
    q.put(None)


def main():
    import torch
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PT):
        raise SystemExit(f"no {MODEL_PT} — run pipeline.train_mirt first")

    tids_by_size = corpus_tids_by_size(SIZES)
    basis = load_joint_activation_basis(SIZES, tids_by_size, PCA_DIM, SEED)
    m1 = torch.tensor(basis["m1"], dtype=torch.float32, device=device)
    s1 = torch.tensor(basis["s1"], dtype=torch.float32, device=device)
    pmean = torch.tensor(basis["pca_mean"], dtype=torch.float32, device=device)
    C = torch.tensor(basis["C"], dtype=torch.float32, device=device)      # (K_pca, raw)
    m2 = torch.tensor(basis["m2"], dtype=torch.float32, device=device)
    s2 = torch.tensor(basis["s2"], dtype=torch.float32, device=device)
    Ch = C.t().half()                                                     # (raw, K_pca)

    st = torch.load(MODEL_PT, map_location=device)
    import torch.nn as nn

    # Shared tail — the SAME layout the trunk was fitted with, from the same
    # loader, with the same fail-loud-on-missing rule. Only the activation block
    # differs here (per CELL rather than the correct cell), which is exactly why
    # the rest must not be restated locally: a layout named here would be a second
    # switch, and feeding the trunk columns it was not fitted with produces a
    # well-formed shard.
    wide, n_wide = load_joint_blocks(PRODUCTION_BLOCKS, SIZES, tids_by_size)
    fdim = PCA_DIM + n_wide + SIZE_FLAG_COLS
    # Dropout is not in the state_dict, so 0.0 here still loads weights fitted
    # at DROPOUT; this pass is eval-only anyway.
    trunk, d = build_trunk(fdim, dropout=0.0)
    trunk = trunk.to(device)
    trunk.load_state_dict(st["trunk"])
    trunk.eval()
    g_lin = nn.Linear(d, K_AXES, bias=False).to(device)
    g_lin.load_state_dict(st["g_lin"])
    g_rms = st["g_rms"].to(device).view(1, -1)

    act_dirs = {s: os.path.join(PROJECT_ROOT, "activations", str(s)) for s in SIZES}
    # tids_by_size IS the corpus listing, and corpus_tids_by_size asserts an id
    # names at most one problem across every board, so there is nothing to filter
    # against: a per-tid size map built from the same listdir agrees by
    # construction. (The rule it enforced still holds — size membership comes
    # from the SGF corpus, never from a cache id list that can be stale.)
    tasks = [(s, t) for s in SIZES for t in tids_by_size[s]]
    # RESUMABLE by file existence, the same idiom extract_activations uses. Every
    # write goes through a tmp file and os.replace, so a file that exists is
    # complete — an interrupted run loses at most the batch in flight. Worth
    # having: this is a long pass over the whole activation store, and it is the
    # last step before packaging, so it is exactly the one that gets interrupted.
    n_all = len(tasks)
    tasks = [(s, t) for s, t in tasks
             if not os.path.exists(os.path.join(OUT_DIR, str(s), t + ".npz"))]
    if len(tasks) < n_all:
        print(f"  resuming: {n_all - len(tasks)} already written, "
              f"{len(tasks)} to go", flush=True)
    for s in SIZES:
        os.makedirs(os.path.join(OUT_DIR, str(s)), exist_ok=True)
    print(f"{len(tasks)} problems to project, K={K_AXES}, device {device}",
          flush=True)

    q = queue.Queue(maxsize=READ_AHEAD)
    threading.Thread(target=_reader, args=(tasks, act_dirs, q),
                     daemon=True).start()

    done = cells = 0
    with torch.no_grad():
        while True:
            batch = q.get()
            if batch is None:
                break
            # One GPU call for the whole batch: the per-cell matmul is
            # bandwidth-bound, so per-problem calls would spend their time in
            # launch overhead.
            counts = [b[2].shape[0] for b in batch]
            raw = torch.from_numpy(np.concatenate([b[2] for b in batch])).to(
                device, non_blocking=True)
            x1 = (raw.float() - m1) / s1
            z = (((x1 - pmean).half() @ Ch).float() - m2) / s2       # (n, PCA_DIM)
            tail = torch.from_numpy(np.concatenate([
                np.tile(joint_feature_tail((b[0], b[1]), wide)
                        .astype(np.float32)[None, :], (n, 1))
                for b, n in zip(batch, counts)])).to(device)
            g = g_lin(trunk(torch.cat([z, tail], dim=1))) / g_rms
            g = g.cpu().numpy().astype(np.float16)
            off = 0
            for (size, tid, _c, xys), n in zip(batch, counts):
                out = os.path.join(OUT_DIR, str(size))
                tmp = os.path.join(out, f".{tid}.tmp.npz")
                np.savez_compressed(tmp, embeddings=g[off:off + n], xys=xys)
                os.replace(tmp, os.path.join(out, tid + ".npz"))
                off += n
            done += len(batch); cells += int(sum(counts))
            if done % 6400 < PROBLEM_BATCH:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} problems, {cells/1e6:.2f}M cells, "
                      f"{el/60:.1f} min, {done/max(el,1):.0f} problems/s",
                      flush=True)
    print(f"per-cell: {done} problems, {cells/1e6:.2f}M cells in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
