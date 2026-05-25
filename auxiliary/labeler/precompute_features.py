"""Precompute the residualized correct-move feature cache for the labeler.

Samples a working pool of problems from embeddings/joint/per_problem.csv, loads
each problem's correct-move 6912-d KataGo trunk activation, and applies the SAME
rating + size_is_13 residualization that build_embedding.py fits — reused
verbatim from embeddings/joint/pca.npz (drop dead channels, z-score, subtract
the OLS offset). It stops *before* the PCA projection / ICA rotation, so the
probe sees the full residual feature space rather than the 64-dim embedding
(which was rotated for skewness and may discard label-relevant variance).

Output (auxiliary/labeler/cache/):
  features.npy   (N_pool, D) float32   residualized correct-move features
  pool.json      manifest: tids, sizes, ratings, zip_ids, embedding_version, D

The labeler server loads this once at startup. Re-run if pca.npz changes
(embedding_version mismatch) or to refresh / enlarge the pool.

Usage:
  ./.venv/bin/python3 -m auxiliary.labeler.precompute_features --pool 5000
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def load_resid_params(pca_path):
    a = np.load(pca_path, allow_pickle=False)
    return {
        "kept": a["kept_feature_mask"].astype(bool),
        "fmean": a["feature_means"].astype(np.float64),
        "fstd": a["feature_stds"].astype(np.float64),
        "rslope": a["rating_slope"].astype(np.float64),
        "sslope": a["size_slope"].astype(np.float64),
        "intercept": a["ols_intercept"].astype(np.float64),
        "version": str(a["embedding_version"]),
    }


def residualize(vec, rating, size_is_13, p):
    """Mirror build_embedding.project_to_embedding steps 1-4 (mask, z-score,
    residualize) WITHOUT the PCA/ICA projection. Returns (D_kept,) float32."""
    v = np.asarray(vec, dtype=np.float64)[p["kept"]]
    z = (v - p["fmean"]) / p["fstd"]
    offset = p["rslope"] * rating + p["sslope"] * size_is_13 + p["intercept"]
    return (z - offset).astype(np.float32)


def load_correct_vec(act_dir, tid):
    """Return the correct-move 6912-float vector, npz (preferred) or legacy npy."""
    npz_path = os.path.join(act_dir, tid + ".npz")
    if os.path.exists(npz_path):
        with np.load(npz_path) as s:
            return s["candidates"][int(s["correct_idx"])]
    npy_path = os.path.join(act_dir, tid + ".npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def load_ratings(size):
    path = os.path.join(PROJECT_ROOT, "ratings", str(size), "tsumego_ratings_final.csv")
    df = pd.read_csv(path)[["tsumego_id", "rating"]]
    return dict(zip(df["tsumego_id"], df["rating"]))


def load_lookup():
    """tid -> zip_id, from pwa/lookup.txt (needed by the server to fetch SGFs)."""
    out = {}
    with open(os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = parts[1]
    return out


def interp_extrap(x, xs, ys):
    """Linear interp with end-slope extrapolation; mirrors rank-decomposition.js
    interpExtrap so the rank filter matches the PWA exactly. xs ascending."""
    n = len(xs)
    if x <= xs[0]:
        return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
    if x >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])
    i = int(np.searchsorted(xs, x)) - 1
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


def parse_rank(label):
    """Rank label → internal rank index (getDescriptiveRank inverse).
    '1d' -> 20, '1k' -> 19, '5k' -> 15, '2d' -> 21. A bare number passes through."""
    label = str(label).strip().lower()
    if label.endswith("k"):
        return 20 - int(label[:-1])
    if label.endswith("d"):
        return 19 + int(label[:-1])
    return float(label)


def build_eligible(per_problem_csv, sizes, max_rank_index):
    """Join ratings onto per_problem, derive each problem's internal rank index
    via the per-board rating->rank curve, and keep only rank_index <= threshold."""
    curve = json.load(open(os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json")))
    df = pd.read_csv(per_problem_csv, usecols=["tsumego_id", "size"])
    df = df[df["size"].isin(sizes)]
    frames = []
    for s in sizes:
        sub = df[df["size"] == s].copy()
        sub["rating"] = sub["tsumego_id"].map(load_ratings(s))
        sub = sub.dropna(subset=["rating"])
        c = curve[str(s)]
        xs, ys = np.asarray(c["ratings"], float), np.asarray(c["rank_indices"], float)
        sub["rank_index"] = sub["rating"].map(lambda r: interp_extrap(r, xs, ys))
        kept = sub[sub["rank_index"] <= max_rank_index]
        print(f"  size {s}: {len(kept)}/{len(sub)} problems at rank_index <= "
              f"{max_rank_index:g}")
        frames.append(kept)
    return pd.concat(frames, ignore_index=True)


def sample_pool(elig, pool, seed):
    """Stratified sample over the eligible set: as even as possible across sizes.
    Returns a list of (size, tid, rating)."""
    rng = np.random.default_rng(seed)
    sizes = sorted(elig["size"].unique())
    per = max(1, pool // len(sizes))
    picked = []
    for s in sizes:
        sub = elig[elig["size"] == s]
        take = min(per, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        rows = sub.iloc[idx]
        picked.extend(zip(rows["size"].astype(int), rows["tsumego_id"], rows["rating"].astype(float)))
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=int, default=5000, help="Total working-pool size.")
    ap.add_argument("--sizes", default="9,13", help="Comma-separated board sizes.")
    ap.add_argument("--max-rank", default="1d",
                    help="Exclude problems harder than this internal rank "
                         "(e.g. '1d', '2k', or a bare index). Default 1d (=index 20).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    max_rank_index = parse_rank(args.max_rank)
    pca_path = os.path.join(PROJECT_ROOT, "embeddings", "joint", "pca.npz")
    pp_path = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_problem.csv")
    if not os.path.exists(pca_path):
        raise SystemExit(f"missing {pca_path}; run pipeline.build_embedding first")
    if not os.path.exists(pp_path):
        raise SystemExit(f"missing {pp_path}; run pipeline.build_embedding first")

    params = load_resid_params(pca_path)
    print(f"filtering to rank <= {args.max_rank} (rank_index <= {max_rank_index:g}):")
    elig = build_eligible(pp_path, sizes, max_rank_index)
    picked = sample_pool(elig, args.pool, args.seed)
    print(f"sampled {len(picked)} problems from {len(elig)} eligible across sizes {sizes}")

    lookup = load_lookup()

    D = int(params["kept"].sum())
    feats = np.empty((len(picked), D), dtype=np.float32)
    tids, out_sizes, out_ratings, out_zips = [], [], [], []
    n_ok = n_skip = 0
    for size, tid, rating in picked:
        zip_id = lookup.get(tid)
        if zip_id is None:
            n_skip += 1
            continue
        act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
        vec = load_correct_vec(act_dir, tid)
        if vec is None:
            n_skip += 1
            continue
        feats[n_ok] = residualize(vec, float(rating), 1.0 if size == 13 else 0.0, params)
        tids.append(tid)
        out_sizes.append(int(size))
        out_ratings.append(float(rating))
        out_zips.append(zip_id)
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  {n_ok}/{len(picked)}", flush=True)

    feats = feats[:n_ok]
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, "features.npy"), feats)
    with open(os.path.join(CACHE_DIR, "pool.json"), "w") as f:
        json.dump({
            "tids": tids,
            "sizes": out_sizes,
            "ratings": out_ratings,
            "zip_ids": out_zips,
            "embedding_version": params["version"],
            "D": D,
            "max_rank": args.max_rank,
            "max_rank_index": max_rank_index,
        }, f)
    print(f"cached {n_ok} feature rows ({n_skip} skipped) → {CACHE_DIR}/")
    print(f"  features.npy: {feats.shape} {feats.dtype}")
    print(f"  embedding_version: {params['version']}")


if __name__ == "__main__":
    main()
