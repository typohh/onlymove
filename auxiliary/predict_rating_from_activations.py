"""Linear probe: how much of a tsumego's rating is recoverable from the
correct-move KataGo trunk activations?

Fits a ridge regression `rating ~ activations` on the 6912-dim correct-move
vector (the same vector build_embedding.py feeds into its PCA) and reports
K-fold cross-validated R² / RMSE / Pearson r. This is a ceiling on what any
*linear* read-off of the trunk knows about difficulty — a high R² means the
black-box rating is largely a linear function of the activations; a low one
means the rating carries information the trunk doesn't expose linearly (or the
activations are just noisy w.r.t. difficulty).

With --contrastive, the feature vector is augmented with the move's *context*:
per-channel mean / std / max over all the OTHER legal moves, plus the per-channel
z-score of the correct move within the board ((correct - mean) / std). The idea
is that difficulty is relational — a move is hard to find when it doesn't stand
out from tempting alternatives — so giving the regressor "compared to what"
should add signal beyond the lone correct-move vector. (For a *linear* model the
explicit `correct - mean` delta is redundant with including `correct` and `mean`
separately, so it is not added; std / max / z-score are nonlinear and genuinely
new.) Blocks: [correct, mean, std, max, zscore], each 6912-dim → 5×6912=34560.

Ratings are board-specific black-box scalars (different scale per size), so each
size is fit and scored independently — there is no cross-size pooling here.

Reuses the correct-vector memmap cache built by inspect_activation_signals.py
(activations/_signal_cache/correct_vecs_<size>.npy). --contrastive builds a
sibling contrastive_vecs_<size>.npy (mean/std/max/zscore blocks) the first time,
aligned to the same id ordering. The first run for a size scans the .npz corpus
to build these; later runs are instant. Pass --rebuild-cache to force a rescan.

Usage:
  python -m auxiliary.predict_rating_from_activations
  python -m auxiliary.predict_rating_from_activations --size 13 --alpha 100
  python -m auxiliary.predict_rating_from_activations --alpha 10 100 1000   # sweep
  python -m auxiliary.predict_rating_from_activations --max-problems 0       # use all
  python -m auxiliary.predict_rating_from_activations --pca-dim 256          # reduce first
  python -m auxiliary.predict_rating_from_activations --contrastive          # +context
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.inspect_activation_signals import (  # noqa: E402
    CACHE_DIR, N_FEATS, _build_cache, _cache_ids_path, _cache_vecs_path,
)

# Contrastive cache: per problem, the per-channel aggregates over the OTHER
# legal moves, concatenated as [mean, std, max, zscore] (4 × N_FEATS). The
# correct-move block lives in the existing correct-vec cache; the two are
# concatenated at eval time and share the same id ordering.
N_CONTRAST_BLOCKS = 4
CONTRAST_BLOCK_NAMES = ("mean", "std", "max", "zscore")
ALL_BLOCK_NAMES = ("correct",) + CONTRAST_BLOCK_NAMES
_Z_CLIP = 10.0  # clamp z-score so std≈0 channels don't overflow fp16 to inf


def _contrastive_cache_path(size):
    return os.path.join(CACHE_DIR, f"contrastive_vecs_{size}.npy")


def _load_contrastive_chunk(task):
    """Worker: per-channel mean/std/max over the other legal moves + the correct
    move's z-score, for a chunk of problems. Returns (start, (chunk, 4*N_FEATS)
    fp16). Problems without a usable .npz get a NaN row (dropped by the caller).
    """
    start, act_dir, items = task
    out = np.full((len(items), N_CONTRAST_BLOCKS * N_FEATS), np.nan, dtype=np.float16)
    for i, (tid, npz_name) in enumerate(items):
        if not npz_name:
            continue  # legacy correct-only .npy has no candidate stack
        try:
            with np.load(os.path.join(act_dir, npz_name)) as s:
                cand = np.asarray(s["candidates"], dtype=np.float32)  # (n_legal, D)
                ci = int(s["correct_idx"])
            correct = cand[ci]
            others = np.delete(cand, ci, axis=0) if cand.shape[0] > 1 else cand
            mean = others.mean(axis=0)
            std = others.std(axis=0)
            mx = others.max(axis=0)
            z = np.clip((correct - mean) / (std + 1e-6), -_Z_CLIP, _Z_CLIP)
            out[i] = np.concatenate([mean, std, mx, z]).astype(np.float16)
        except Exception:
            pass  # leave NaN row
    return start, out


def _build_contrastive_cache(size, workers):
    """Build contrastive_vecs_<size>.npy aligned to the correct-vec cache ids.

    Requires the correct-vec cache first (for the canonical id ordering), then
    re-reads each problem's .npz candidate stack to compute the aggregates.
    """
    ids_path = _cache_ids_path(size)
    if not os.path.exists(ids_path):
        _build_cache(size, workers)
    with open(ids_path) as f:
        ids = json.load(f)

    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    npz = {f[:-4]: f for f in os.listdir(act_dir) if f.endswith(".npz")}
    n = len(ids)
    n_no_npz = sum(1 for t in ids if t not in npz)
    if n_no_npz:
        print(f"  [size {size}] {n_no_npz}/{n} ids have no .npz "
              f"(legacy correct-only); contrastive rows will be NaN/dropped")

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = _contrastive_cache_path(size) + ".tmp"
    mm = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.float16,
        shape=(n, N_CONTRAST_BLOCKS * N_FEATS))

    chunk = 512
    tasks = [(s, act_dir, [(t, npz.get(t)) for t in ids[s:s + chunk]])
             for s in range(0, n, chunk)]
    nbytes = n * N_CONTRAST_BLOCKS * N_FEATS * 2
    print(f"  [size {size}] building contrastive cache: {n} problems "
          f"-> {_contrastive_cache_path(size)} ({nbytes / 1e9:.2f} GB)", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s, arr in ex.map(_load_contrastive_chunk, tasks):
            mm[s:s + arr.shape[0]] = arr
            done += arr.shape[0]
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            print(f"\r    {done}/{n} ({rate:.0f}/s)", end="", flush=True)
    print(flush=True)
    mm.flush()
    del mm
    os.replace(tmp_path, _contrastive_cache_path(size))


def load_size(size, max_problems, seed, rebuild, workers, contrastive=False):
    """Return (X float32 (n, D), y float64 (n,), block_names) for one board size.

    D is N_FEATS (correct-move only) or 5×N_FEATS with --contrastive (correct +
    mean/std/max/zscore over the other legal moves). Loads from the memmap
    cache(s), joins the raw rating, drops failed (NaN) rows, and optionally
    subsamples to `max_problems` (0 = all) so the ridge fit stays tractable.
    """
    vecs_path = _cache_vecs_path(size)
    ids_path = _cache_ids_path(size)
    if rebuild or not (os.path.exists(vecs_path) and os.path.exists(ids_path)):
        _build_cache(size, workers)

    with open(ids_path) as f:
        ids = json.load(f)
    vecs = np.load(vecs_path, mmap_mode="r")  # (n, N_FEATS) fp16
    if vecs.shape[1] != N_FEATS or vecs.shape[0] != len(ids):
        raise SystemExit(
            f"cache for size {size} is stale ({vecs.shape} vs {len(ids)} ids); "
            f"rerun with --rebuild-cache")

    cvecs = None
    if contrastive:
        cpath = _contrastive_cache_path(size)
        if rebuild or not os.path.exists(cpath):
            _build_contrastive_cache(size, workers)
        cvecs = np.load(cpath, mmap_mode="r")  # (n, 4*N_FEATS) fp16
        if cvecs.shape != (len(ids), N_CONTRAST_BLOCKS * N_FEATS):
            raise SystemExit(
                f"contrastive cache for size {size} is stale ({cvecs.shape}); "
                f"rerun with --rebuild-cache")

    ratings_path = os.path.join(PROJECT_ROOT, "ratings", str(size),
                                "tsumego_ratings_final.csv")
    if not os.path.exists(ratings_path):
        raise SystemExit(f"no ratings file for size {size}: {ratings_path}")
    rmap = (pd.read_csv(ratings_path)[["tsumego_id", "rating"]]
            .set_index("tsumego_id")["rating"].to_dict())

    keep = np.array([t in rmap for t in ids])
    n_no_rating = int((~keep).sum())
    kept_pos = np.flatnonzero(keep)
    if n_no_rating:
        print(f"  [size {size}] {n_no_rating} problems without a rating dropped")

    rng = np.random.default_rng(seed)
    if max_problems and len(kept_pos) > max_problems:
        kept_pos = np.sort(rng.choice(kept_pos, size=max_problems, replace=False))
        print(f"  [size {size}] subsampled to {max_problems} problems "
              f"(seed {seed}); pass --max-problems 0 for all")

    n_blocks = 1 + (N_CONTRAST_BLOCKS if contrastive else 0)
    ncols = n_blocks * N_FEATS
    block_names = ALL_BLOCK_NAMES if contrastive else ("correct",)
    print(f"  [size {size}] loading {len(kept_pos)} × {ncols} features "
          f"({'+'.join(block_names)}) into memory "
          f"({len(kept_pos) * ncols * 4 / 1e9:.2f} GB f32)...", flush=True)
    # Preallocate and fill per block to avoid an extra full-size concat copy.
    X = np.empty((len(kept_pos), ncols), dtype=np.float32)
    X[:, :N_FEATS] = vecs[kept_pos]
    if contrastive:
        X[:, N_FEATS:] = cvecs[kept_pos]
    y = np.array([rmap[ids[i]] for i in kept_pos], dtype=np.float64)

    # Failed extractions are stored as NaN rows in the cache(s); drop any.
    finite = np.isfinite(X).all(axis=1)
    n_nan = int((~finite).sum())
    if n_nan:
        print(f"  [size {size}] dropping {n_nan} NaN (failed-load) rows")
        X = X[finite]
        y = y[finite]
    return X, y, block_names


def evaluate(X, y, alpha, n_splits, seed, pca_dim, jobs):
    """K-fold CV of a (standardize → [PCA] → ridge) pipeline.

    Returns (r2, rmse, pearson_r) computed on the out-of-fold predictions, so
    every score is on data the model never saw in that fold.
    """
    from scipy.stats import pearsonr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("scale", StandardScaler())]
    if pca_dim:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=seed)))
    steps.append(("ridge", Ridge(alpha=alpha)))
    model = Pipeline(steps)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X, y, cv=kf, n_jobs=jobs)
    r2 = r2_score(y, oof)
    rmse = math.sqrt(mean_squared_error(y, oof))
    r = float(pearsonr(oof, y)[0])
    return r2, rmse, r


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="all",
                        help="9, 13, or 'all' (default; each size fit separately).")
    parser.add_argument("--alpha", type=float, nargs="+", default=[100.0],
                        help="Ridge regularization strength(s). Multiple values "
                             "are swept and reported (default: 100).")
    parser.add_argument("--max-problems", type=int, default=40000,
                        help="Cap problems per size to keep the fit fast "
                             "(0 = use all; default 40000).")
    parser.add_argument("--pca-dim", type=int, default=0,
                        help="Reduce to this many PCA components before ridge "
                             "(0 = none; speeds up the 6912-dim fit).")
    parser.add_argument("--contrastive", action="store_true",
                        help="Augment the correct-move vector with per-channel "
                             "mean/std/max over the OTHER legal moves + the "
                             "correct move's z-score (5×6912 features).")
    parser.add_argument("--splits", type=int, default=5,
                        help="K for K-fold cross-validation (default 5).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel CV folds. >1 duplicates X per worker "
                             "(memory-heavy on the full corpus); default 1.")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Force a rescan of activations/<size> into the cache.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Cache-build worker processes (default: all CPUs).")
    args = parser.parse_args()

    sizes = [9, 13] if args.size == "all" else [int(args.size)]

    for size in sizes:
        print(f"\n=== size {size}x{size}{' (contrastive)' if args.contrastive else ''} ===",
              flush=True)
        X, y, block_names = load_size(size, args.max_problems, args.seed,
                                      args.rebuild_cache, args.workers,
                                      contrastive=args.contrastive)
        n = len(y)
        baseline_rmse = float(np.std(y))  # RMSE of predicting the mean rating
        print(f"  {n} problems | rating mean={y.mean():.3f} std={baseline_rmse:.3f} "
              f"min={y.min():.3f} max={y.max():.3f}")
        print(f"  baseline RMSE (predict mean) = {baseline_rmse:.3f}, R²=0 by definition")
        print(f"  feature blocks: {'+'.join(block_names)}")
        if args.pca_dim:
            print(f"  reducing to {args.pca_dim} PCA components before ridge")
        print(f"  {args.splits}-fold CV, features={X.shape[1]}\n", flush=True)

        print(f"  {'alpha':>10s} {'R²':>8s} {'RMSE':>9s} {'RMSE/base':>10s} {'pearson r':>10s}")
        for alpha in args.alpha:
            t0 = time.time()
            r2, rmse, r = evaluate(X, y, alpha, args.splits, args.seed,
                                   args.pca_dim, args.jobs)
            print(f"  {alpha:>10.4g} {r2:>8.4f} {rmse:>9.4f} "
                  f"{rmse / baseline_rmse:>10.4f} {r:>10.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
