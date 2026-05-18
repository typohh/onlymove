"""Fit a rating-residualized PCA (optionally rotated by FastICA) jointly on
per-problem KataGo trunk activations across all configured board sizes.

Inputs (loads from each size that has data):
  activations/<size>/<id>.npy
  ratings/<size>/tsumego_ratings_final.csv

Outputs (single shared model — embeddings/joint/):
  pca.npz                 — full model state, sufficient to project a new problem
  per_problem.csv         — tsumego_id, size, embed_0..K-1
  explained_variance.csv  — component, var_ratio, cumulative
  extreme_problems.json   — per component, top-K and bottom-K problems (across sizes)
  labels.json             — manually-curated human labels per PC per side
                            (preserved across re-runs; new K's get null entries)

Per-feature residualization controls for both rating and a 9-vs-13 board-size
indicator before PCA, so the PCA's variance budget isn't spent on either.

PC scores are standardized to unit variance per component so coefficients
in the downstream weakness regression are comparable across components.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, IncrementalPCA

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

SIZES = [9, 13]


def _chunked_mean_std(X, chunk_rows):
    """One-pass per-feature mean and std without (N, D) temporaries.

    np.std materializes (X - mean) as a same-shape float64 buffer internally
    when dtype=float64; on the joint matrix that's ~8 GB for nothing. This
    streams float64 sum and sum-of-squares accumulators over chunks instead,
    so peak transient memory is just one chunk_rows × D × 8 buffer.
    """
    N, D = X.shape
    sum_ = np.zeros(D, dtype=np.float64)
    sum_sq = np.zeros(D, dtype=np.float64)
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        chunk64 = X[s:e].astype(np.float64)
        sum_ += chunk64.sum(axis=0)
        sum_sq += (chunk64 * chunk64).sum(axis=0)
    mean = sum_ / N
    var = np.maximum(sum_sq / N - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def _chunked_design_T_X(X_design, X, chunk_rows):
    """Compute X_design.T @ X (shape (P, D)) without a (N, D) temp.

    X_design (N, P) and X (N, D) are float32 here; the matmul is dispatched to
    sgemm per chunk and accumulated.
    """
    P = X_design.shape[1]
    D = X.shape[1]
    out = np.zeros((P, D), dtype=np.float32)
    N = X.shape[0]
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        out += X_design[s:e].T @ X[s:e]
    return out


def _chunked_residualize(X, X_design, coefs, chunk_rows):
    """In-place X[s:e] -= X_design[s:e] @ coefs across chunks.

    The eager form X -= X_design @ coefs allocates a single (N, D) float32
    temp (~4 GB on the joint matrix); chunking reduces it to chunk_rows × D.
    """
    N = X.shape[0]
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        X[s:e] -= X_design[s:e] @ coefs


def _load_correct_vec(act_dir: str, tid: str, by_ext: dict) -> np.ndarray:
    """Return the correct-move 6912-float vector for a single problem,
    handling both the new .npz (all-legal-cells, with correct_idx) and the
    legacy .npy (correct-cell-only) layouts. On a corrupt .npz, falls back
    to .npy if present — silently for fitting, since the correct cell is
    the only vector we need here regardless of format.
    """
    if tid in by_ext.get("npz", {}):
        npz_path = os.path.join(act_dir, by_ext["npz"][tid])
        try:
            with np.load(npz_path) as s:
                return s["candidates"][int(s["correct_idx"])]
        except Exception as e:
            if tid not in by_ext.get("npy", {}):
                raise RuntimeError(
                    f"corrupt .npz at {npz_path} and no .npy fallback ({e})"
                )
            print(f"  [warn] corrupt npz {npz_path}: {type(e).__name__}; "
                  f"falling back to .npy", flush=True)
    path = os.path.join(act_dir, by_ext["npy"][tid])
    return np.load(path)


def load_size(size: int):
    """Returns (ids, X, ratings) for a single size, or (None, None, None) if no data."""
    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    ratings_path = os.path.join(PROJECT_ROOT, "ratings", str(size), "tsumego_ratings_final.csv")
    if not os.path.isdir(act_dir):
        print(f"  [size {size}] no activations dir; skipping")
        return None, None, None
    if not os.path.exists(ratings_path):
        print(f"  [size {size}] no ratings file; skipping")
        return None, None, None

    by_ext = {
        "npz": {f[:-4]: f for f in os.listdir(act_dir) if f.endswith(".npz")},
        "npy": {f[:-4]: f for f in os.listdir(act_dir) if f.endswith(".npy")},
    }
    # Prefer .npz when both exist for the same id.
    ids = sorted(set(by_ext["npz"]) | set(by_ext["npy"]))
    if not ids:
        print(f"  [size {size}] no activation files; skipping")
        return None, None, None
    n_npz = sum(1 for tid in ids if tid in by_ext["npz"])
    n_npy = len(ids) - n_npz
    if n_npy:
        print(f"  [size {size}] {n_npz} .npz + {n_npy} legacy .npy "
              f"(legacy entries fitted from correct-cell-only vector)")

    sample = _load_correct_vec(act_dir, ids[0], by_ext)
    n, d = len(ids), sample.shape[0]
    print(f"  [size {size}] loading {n} activations × {d} features...")
    X = np.empty((n, d), dtype=np.float32)
    for i, tid in enumerate(ids):
        X[i] = _load_correct_vec(act_dir, tid, by_ext)
        if (i + 1) % 20000 == 0:
            print(f"    [size {size}]   {i + 1}/{n}", flush=True)

    ratings_df = pd.read_csv(ratings_path)[["tsumego_id", "rating"]]
    ratings_map = dict(zip(ratings_df["tsumego_id"], ratings_df["rating"]))
    keep = np.array([i in ratings_map for i in ids])
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  [size {size}] dropping {n_drop} problems with no rating")
    ids = [i for i, k in zip(ids, keep) if k]
    X = X[keep]
    ratings = np.array([ratings_map[i] for i in ids], dtype=np.float64)
    print(f"  [size {size}] kept {len(ids)} problems with ratings")
    return ids, X, ratings


def _project_to_full_space(components: np.ndarray, mask: np.ndarray, full_d: int) -> np.ndarray:
    """Embed loading vectors of shape (K, kept_d) back into the full feature space
    (K, full_d) by zero-filling dropped features. Lets us compute meaningful
    cosine similarity between two PCA fits whose dead-channel masks differ.
    """
    out = np.zeros((components.shape[0], full_d), dtype=np.float64)
    out[:, mask] = components
    return out


def remap_labels_by_cosine(
    old_labels: dict,
    old_components: np.ndarray, old_mask: np.ndarray,
    new_components: np.ndarray, new_mask: np.ndarray,
    full_d: int,
    threshold: float,
) -> dict:
    """Carry labeled components forward from a previous PCA fit to the current one.

    Each old labeled PC is matched to the new PC with maximum |cosine similarity|
    in the full feature space. Above `threshold`, the label is copied. Negative
    cosine flips the sign of the axis, so positive/negative label sides swap.

    Returns a fresh `components` dict {str(k): {"positive": ..., "negative": ...,
    "auto_mapped_from": int, "auto_mapped_cos": float}} with all K_new entries.
    """
    K_old = old_components.shape[0]
    K_new = new_components.shape[0]

    old_full = _project_to_full_space(old_components, old_mask, full_d)
    new_full = _project_to_full_space(new_components, new_mask, full_d)
    # Loading vectors should already be unit-norm but renormalize defensively
    # (zero-fill outside the kept set has no effect on dot products since the
    # missing features were also zero in the original computation).
    old_norms = np.linalg.norm(old_full, axis=1, keepdims=True)
    new_norms = np.linalg.norm(new_full, axis=1, keepdims=True)
    old_full = old_full / np.where(old_norms < 1e-12, 1.0, old_norms)
    new_full = new_full / np.where(new_norms < 1e-12, 1.0, new_norms)
    sim = old_full @ new_full.T  # (K_old, K_new), in [-1, 1]

    old_components_dict = old_labels.get("components", {}) if old_labels else {}

    # For each old PC that has a non-null label on either side, find the best
    # new PC and (if confident) carry the label across.
    proposals = []  # (old_k, new_k, signed_cos, |cos|, sign)
    for k_str, entry in old_components_dict.items():
        if not entry:
            continue
        if entry.get("positive") is None and entry.get("negative") is None:
            continue
        k_old = int(k_str)
        if k_old >= K_old:
            continue
        row = sim[k_old]
        k_new = int(np.argmax(np.abs(row)))
        c = float(row[k_new])
        if abs(c) < threshold:
            continue
        proposals.append((k_old, k_new, c))

    # If two old PCs claim the same new PC, keep the higher-confidence one.
    by_new = {}
    for k_old, k_new, c in proposals:
        prev = by_new.get(k_new)
        if prev is None or abs(c) > abs(prev[1]):
            by_new[k_new] = (k_old, c)

    new_components_dict = {}
    for k_new in range(K_new):
        slot = {"positive": None, "negative": None}
        if k_new in by_new:
            k_old, c = by_new[k_new]
            old_entry = old_components_dict.get(str(k_old), {})
            pos = old_entry.get("positive")
            neg = old_entry.get("negative")
            if c < 0:
                pos, neg = neg, pos  # axis flipped — swap sides
            slot["positive"] = pos
            slot["negative"] = neg
            slot["auto_mapped_from"] = k_old
            slot["auto_mapped_cos"] = round(c, 4)
        new_components_dict[str(k_new)] = slot
    return new_components_dict


def _skew_contrast(x):
    """FastICA contrast targeting the 3rd cumulant (skewness) instead of the
    symmetric 4th-cumulant proxy that 'logcosh'/'exp'/'cube' approximate.

    With G(u) = u^3/3, g(u) = G'(u) = u^2 and g'(u) = 2u. Because g'(u) is odd,
    the FastICA update simplifies (the E[g'(w·X)]·w term has zero expectation
    on whitened, mean-zero input). The recovered components are structurally
    one-sided — positive tail concentrates the rare motif, negative tail is
    Gaussian-like baseline — so labels are unambiguous (don't label negative).
    """
    return x ** 2, (2.0 * x).mean(axis=-1)


_LABELS_DOC = (
    "Human-curated semantic labels per PCA component. Each side independently — "
    "set to null when no clear pattern. Auto-remapped across PCA refits via "
    "cosine similarity in the original feature space; auto_mapped_cos records "
    "the match strength."
)


def write_or_remap_labels(
    labels_path: str,
    new_components: np.ndarray,
    new_mask: np.ndarray,
    full_d: int,
    K: int,
    prior_components=None,
    prior_mask=None,
    cosine_threshold: float = 0.8,
):
    """Write or update labels.json across a PCA refit.

    - No existing file → fresh stub with K null entries.
    - Existing file + prior loadings → cosine-similarity remap (sign-flipping as
      needed), survivors transferred, others reset to null.
    - Existing file but no prior loadings → preserve labels by index, extend to K.
    """
    if not os.path.exists(labels_path):
        labels = {
            "version": 2,
            "_doc": _LABELS_DOC,
            "components": {str(k): {"positive": None, "negative": None} for k in range(K)},
        }
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"  wrote stub labels.json (K={K}, all entries null)")
        return

    with open(labels_path, "r") as f:
        old_labels = json.load(f)

    if prior_components is None or prior_mask is None:
        old_dict = old_labels.get("components", {})
        labels = {
            "version": old_labels.get("version", 2),
            "_doc": old_labels.get("_doc", _LABELS_DOC),
            "components": {
                str(k): old_dict.get(str(k), {"positive": None, "negative": None})
                for k in range(K)
            },
        }
        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"  preserved labels by index (no prior loadings to align)")
        return

    remapped = remap_labels_by_cosine(
        old_labels, prior_components, prior_mask,
        new_components, new_mask, full_d, cosine_threshold,
    )
    labels = {
        "version": 2,
        "_doc": old_labels.get("_doc", _LABELS_DOC),
        "components": remapped,
    }
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    n_carried = sum(
        1 for v in remapped.values()
        if v.get("positive") is not None or v.get("negative") is not None
    )
    n_old_nonempty = sum(
        1 for v in (old_labels.get("components", {}) or {}).values()
        if v and (v.get("positive") is not None or v.get("negative") is not None)
    )
    print(f"  remapped labels: {n_carried}/{n_old_nonempty} old labels survived "
          f"(cosine threshold {cosine_threshold})")


def project_to_embedding(activations, rating, size_is_13, pca_arc):
    """Project raw 6912-dim trunk activations to the K-dim joint embedding.

    Inverse-application of the fit pipeline in main(): drop dead channels,
    z-score per feature, residualize on (rating, size_is_13), recenter, PCA,
    standardize by component_stds, then optionally rotate by the stored ICA
    unmixing matrix. The same vector passed through fit and through this
    function should round-trip (modulo fp16/fp64 noise).

    Parameters
    ----------
    activations : np.ndarray
        Either (D,) for a single cell or (N, D) for N cells of one problem.
        D must match the full pre-mask dimension (6912 for b18 × 18 blocks).
        Accepted dtypes: anything castable to float64 (fp16 / fp32 / fp64).
    rating, size_is_13 : float
        Shared scalars for the problem these cells belong to. The fit OLS
        used rating in its native (raw) scale and size_is_13 ∈ {0, 1}.
    pca_arc : np.lib.npyio.NpzFile
        Already-opened embeddings/joint/pca.npz (allow_pickle=False).

    Returns
    -------
    np.ndarray
        (K,) or (N, K) float32, with shape matching the input's leading dim.
    """
    kept = pca_arc["kept_feature_mask"].astype(bool)
    feature_means = pca_arc["feature_means"].astype(np.float64)
    feature_stds = pca_arc["feature_stds"].astype(np.float64)
    rating_slope = pca_arc["rating_slope"].astype(np.float64)
    size_slope = pca_arc["size_slope"].astype(np.float64)
    ols_intercept = pca_arc["ols_intercept"].astype(np.float64)
    pca_components = pca_arc["pca_components"].astype(np.float64)
    pca_mean = pca_arc["pca_mean"].astype(np.float64)
    component_stds = pca_arc["component_stds"].astype(np.float64)
    component_stds = np.where(component_stds < 1e-9, 1.0, component_stds)

    V = np.asarray(activations, dtype=np.float64)
    single = (V.ndim == 1)
    if single:
        V = V[None, :]
    V_kept = V[:, kept]                                       # (N, D_kept)
    Z = (V_kept - feature_means) / feature_stds               # (N, D_kept)
    # Rating + size offsets are constant across cells of one problem.
    offset = rating_slope * rating + size_slope * size_is_13 + ols_intercept
    R = Z - offset                                            # (N, D_kept)
    centered = R - pca_mean
    pca_scores = (centered @ pca_components.T) / component_stds  # (N, K)

    if "ica_unmixing" in pca_arc.files:
        ica_unmixing = pca_arc["ica_unmixing"].astype(np.float64)
        ica_stds = pca_arc["ica_stds"].astype(np.float64)
        ica_stds = np.where(ica_stds < 1e-9, 1.0, ica_stds)
        out = (pca_scores @ ica_unmixing.T) / ica_stds       # (N, K)
    else:
        out = pca_scores

    out = out.astype(np.float32)
    return out[0] if single else out


# --- Per-cell projection: a post-fit pass that projects every problem's full
# (n_cells, 6912) activation block through the freshly-fit PCA/ICA and saves
# the resulting (n_cells, K) fp16 array. This makes package_for_pwa cheap —
# no PCA matmul or raw-activation reads at packaging time, just a small
# pre-projected file per problem.
#
# Parallelized via ProcessPoolExecutor. Each worker loads pca.npz once and
# then processes its share of problems. Embarrassingly parallel — no shared
# mutable state, modest per-task memory.

_PCA_ARC_WORKER = None


def _init_per_cell_worker(pca_path):
    """ProcessPoolExecutor initializer: open pca.npz once per worker."""
    global _PCA_ARC_WORKER
    _PCA_ARC_WORKER = np.load(pca_path, allow_pickle=False)


def _project_one_for_per_cell(task):
    """Worker function: read one problem's .npz, project all cells, save
    pre-projected (n_cells, K) fp16 + xys to the per_cell output dir.
    Returns (tid, status_str) for logging."""
    size, tid, raw_rating, in_dir, out_dir = task
    npz_path = os.path.join(in_dir, tid + ".npz")
    if not os.path.exists(npz_path):
        return tid, "missing_npz"
    try:
        with np.load(npz_path) as data:
            candidates = np.asarray(data["candidates"])  # (n_cells, 6912) fp16
            xys = np.asarray(data["xys"])                # (n_cells, 2)
    except Exception as e:
        return tid, f"load_failed:{type(e).__name__}"

    size_is_13 = 1.0 if size == 13 else 0.0
    try:
        projected = project_to_embedding(candidates, raw_rating, size_is_13,
                                         _PCA_ARC_WORKER)
    except Exception as e:
        return tid, f"project_failed:{type(e).__name__}"

    out_path = os.path.join(out_dir, tid + ".npz")
    np.savez_compressed(
        out_path,
        embeddings=projected.astype(np.float16),
        xys=xys.astype(np.int32),
    )
    return tid, "ok"


def write_per_cell_projections(parts, pca_path, n_workers=8):
    """Project every problem's full cell stack through the saved PCA and write
    embeddings/joint/per_cell/<size>/<tid>.npz with the K-dim embeddings.

    `parts` is the list of (size, ids, X, ratings) tuples already loaded by
    main() for the PCA fit — same ids and ratings, no re-walk of the .npz files
    needed at scheduling time. Workers re-open each .npz to read the full
    (n_cells, 6912) array (which load_size dropped after slicing the correct
    row), then project.
    """
    out_root = os.path.join(PROJECT_ROOT, "embeddings", "joint", "per_cell")
    os.makedirs(out_root, exist_ok=True)

    tasks = []
    for size, ids, _X, ratings in parts:
        in_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
        out_dir = os.path.join(out_root, str(size))
        os.makedirs(out_dir, exist_ok=True)
        for tid, rating in zip(ids, ratings):
            tasks.append((size, tid, float(rating), in_dir, out_dir))

    if not tasks:
        print("  per_cell projection: no tasks", flush=True)
        return

    print(f"\nProjecting per-cell embeddings: {len(tasks)} problems → "
          f"{out_root}/<size>/<tid>.npz", flush=True)
    print(f"  parallel workers: {n_workers}", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_per_cell_worker,
        initargs=(pca_path,),
    ) as ex:
        for i, (tid, status) in enumerate(
                ex.map(_project_one_for_per_cell, tasks, chunksize=200)):
            if status == "ok":
                n_ok += 1
            elif status == "missing_npz":
                n_skip += 1
            else:
                n_err += 1
                if n_err <= 10:
                    print(f"  [err] {tid}: {status}", flush=True)
            done = i + 1
            if done % 20000 == 0 or done == len(tasks):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{len(tasks)}  ({rate:.0f}/s)", flush=True)
    print(f"Per-cell projection done: {n_ok} ok, {n_skip} skipped, "
          f"{n_err} errored  ({time.time() - t0:.0f}s total)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-components", type=int, default=64,
                        help="K — pin post-hoc from the variance curve.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--std-floor", type=float, default=1e-3,
                        help="Drop features whose std across the dataset is below this.")
    parser.add_argument("--top-k-extreme", type=int, default=8,
                        help="High/low problems recorded per component for inspection.")
    parser.add_argument("--rotation", choices=["none", "ica"], default="ica",
                        help="Post-PCA rotation. 'ica' applies FastICA within the K-dim "
                             "PCA-whitened subspace to find statistically independent / "
                             "non-Gaussian axes (recommended when motifs spread across "
                             "several PCs). 'none' keeps raw PCA axes.")
    parser.add_argument("--ica-fun", choices=["logcosh", "exp", "cube", "skew"],
                        default="skew",
                        help="FastICA contrast (non-Gaussianity proxy). "
                             "'logcosh'/'exp'/'cube' are symmetric → find both "
                             "one-sided and contrast (motif-vs-motif) axes. "
                             "'skew' uses g(u)=u^2 → axes are structurally "
                             "one-sided (positive tail = motif, negative tail = "
                             "baseline). Default is 'skew' for cleaner labels.")
    parser.add_argument("--ica-max-iter", type=int, default=2000,
                        help="FastICA max iterations.")
    parser.add_argument("--ica-tol", type=float, default=1e-4,
                        help="FastICA convergence tolerance.")
    parser.add_argument("--chunk-rows", type=int, default=8192,
                        help="Rows per chunk for streaming ops (mean/std, OLS, "
                             "residualize, IncrementalPCA). Smaller = lower "
                             "peak RAM, marginally slower. 8192 rows × 6912 "
                             "features × 4 bytes ≈ 220 MB per chunk.")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "embeddings", "joint")
    os.makedirs(out_dir, exist_ok=True)

    # Snapshot the prior model (if any) before we overwrite — needed below
    # to remap labels across the PCA refit.
    pca_npz_path = os.path.join(out_dir, "pca.npz")
    labels_path = os.path.join(out_dir, "labels.json")
    prior_components = None
    prior_mask = None
    if os.path.exists(pca_npz_path):
        try:
            arc = np.load(pca_npz_path, allow_pickle=False)
            if "pca_components" in arc.files and "kept_feature_mask" in arc.files:
                prior_pca = arc["pca_components"]
                prior_mask = arc["kept_feature_mask"].astype(bool).copy()
                # If the prior fit included an ICA rotation, the labeled axes are
                # the ICA-rotated directions in residual feature space — compose
                # them so cosine matching happens in the same coordinate system
                # as the new fit's effective directions.
                if "ica_unmixing" in arc.files and "component_stds" in arc.files:
                    prior_W = arc["ica_unmixing"].astype(np.float64)
                    prior_stds = arc["component_stds"].astype(np.float64)
                    prior_stds = np.where(prior_stds < 1e-9, 1.0, prior_stds)
                    prior_components = (
                        prior_W @ (prior_pca / prior_stds[:, None])
                    ).copy()
                else:
                    prior_components = prior_pca.copy()
                print(f"  found prior pca.npz; will remap labels post-refit")
        except Exception as e:
            print(f"  prior pca.npz unreadable, continuing without label remap: {e}")

    print("Loading per-size activations + ratings...")
    parts = []
    for size in SIZES:
        ids, X, ratings = load_size(size)
        if ids is None:
            continue
        parts.append((size, ids, X, ratings))

    if not parts:
        raise SystemExit("no per-size data available; nothing to fit")

    sizes_used = sorted(p[0] for p in parts)
    print(f"\n  joint fit will use sizes: {sizes_used}")

    all_ids = []
    all_sizes = []
    Xs = []
    rs = []
    for size, ids, X, ratings in parts:
        all_ids.extend(ids)
        all_sizes.append(np.full(len(ids), size, dtype=np.int32))
        Xs.append(X)
        rs.append(ratings)
    # Stack into a single float32 matrix and immediately release the per-size
    # chunks (both Xs and the X refs held in `parts`). Staying in float32 here
    # halves the data-matrix footprint vs float64 — randomized SVD does not
    # need float64 and OLS on (rating, size_is_13, intercept) is well-conditioned
    # at this N.
    X = np.vstack(Xs)
    del Xs
    parts = [(size, ids, None, r) for size, ids, _, r in parts]
    sizes_col = np.concatenate(all_sizes)
    ratings = np.concatenate(rs)
    print(f"\n  joint matrix: {X.shape[0]} problems × {X.shape[1]} features")
    unique_counts = dict(zip(*[a.tolist() for a in np.unique(sizes_col, return_counts=True)]))
    print(f"  size composition: {unique_counts}")

    chunk_rows = max(1024, int(args.chunk_rows))
    print(f"\n  streaming chunk size: {chunk_rows} rows", flush=True)

    print("\nZ-scoring per feature...", flush=True)
    # Streaming mean/std avoids np.std's (N, D) float64 temp (~8 GB).
    fm64, fs64 = _chunked_mean_std(X, chunk_rows)
    feature_means = fm64.astype(np.float32)
    feature_stds = fs64.astype(np.float32)
    keep_feat = feature_stds >= args.std_floor
    n_dead = int((~keep_feat).sum())
    print(f"  dropping {n_dead}/{len(keep_feat)} dead channels (std < {args.std_floor})")
    feature_means = feature_means[keep_feat]
    feature_stds = feature_stds[keep_feat]
    if n_dead > 0:
        # Fancy-indexing column mask copies the kept slab into a fresh (N, D_kept)
        # buffer; the old X is GC'd once reassigned. Brief ~2× transient.
        X = X[:, keep_feat]
    # In-place broadcast subtract / divide; no temp.
    for s in range(0, X.shape[0], chunk_rows):
        e = min(s + chunk_rows, X.shape[0])
        X[s:e] -= feature_means
        X[s:e] /= feature_stds

    print("Residualizing rating + size_is_13 jointly per feature (vectorized OLS)...",
          flush=True)
    size_is_13 = (sizes_col == 13).astype(np.float32)
    X_design = np.column_stack([
        ratings.astype(np.float32), size_is_13,
        np.ones_like(ratings, dtype=np.float32),
    ])  # (N, 3) float32
    XtX_inv = np.linalg.inv((X_design.T @ X_design).astype(np.float64)).astype(np.float32)
    coefs = XtX_inv @ _chunked_design_T_X(X_design, X, chunk_rows)  # (3, D) float32
    _chunked_residualize(X, X_design, coefs, chunk_rows)
    _, residual_std = _chunked_mean_std(X, chunk_rows)
    print(f"  residual feature norms: mean={residual_std.mean():.3f} "
          f"min={residual_std.min():.3f} max={residual_std.max():.3f}")

    K = args.n_components
    N = X.shape[0]
    print(f"\nFitting IncrementalPCA n_components={K} batch_size={chunk_rows}...",
          flush=True)
    # IncrementalPCA accumulates an incremental SVD over fixed-size batches —
    # peak transient memory is one batch's worth instead of the (N, D) copies
    # randomized PCA materializes internally. copy=False lets it mean-center
    # each batch in place (we discard X right after transforming).
    ipca = IncrementalPCA(n_components=K, copy=False, batch_size=chunk_rows)
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        ipca.partial_fit(X[s:e])

    raw_scores = np.empty((N, K), dtype=np.float32)
    pca_mean_f32 = ipca.mean_.astype(np.float32)
    components_T_f32 = ipca.components_.astype(np.float32).T  # (D, K)
    for s in range(0, N, chunk_rows):
        e = min(s + chunk_rows, N)
        # Manual mean-center + project keeps the per-chunk temp at chunk_rows × D
        # and avoids sklearn's transform() allocating a same-size copy.
        chunk = X[s:e] - pca_mean_f32
        raw_scores[s:e] = chunk @ components_T_f32
    del X, components_T_f32

    # Shim onto the rest of the pipeline (which expects sklearn PCA attribute
    # names). IncrementalPCA exposes the same attributes; alias to keep edits
    # contained.
    pca = ipca
    component_stds = np.sqrt(pca.explained_variance_).astype(np.float64)
    component_stds[component_stds < 1e-9] = 1.0
    scores = (raw_scores / component_stds).astype(np.float32)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  top-{K} cumulative explained variance: {cumvar[-1]:.4f}")
    print(f"  per-component scores standardized to unit variance "
          f"(divisors: min={component_stds.min():.3f} max={component_stds.max():.3f})")

    # Optional FastICA rotation within the whitened K-dim subspace. Preserves the
    # linear span (so any downstream linear model gets identical predictive power)
    # but seeks axes whose projected scores are maximally non-Gaussian — useful
    # when one motif is split across several PCs.
    W_ica = None
    ica_stds = None
    ica_excess_kurt = None
    ica_skewness = None
    if args.rotation == "ica":
        fun = _skew_contrast if args.ica_fun == "skew" else args.ica_fun
        print(f"\nFitting FastICA n_components={K} on whitened PCA scores "
              f"(fun={args.ica_fun!r}, whiten=False)...", flush=True)
        # n_components is intentionally omitted: with whiten=False sklearn
        # ignores it and uses the input dim, which is already K from PCA.
        ica = FastICA(
            whiten=False,
            fun=fun,
            max_iter=args.ica_max_iter,
            tol=args.ica_tol,
            random_state=args.seed,
        )
        ica_raw = ica.fit_transform(scores.astype(np.float64))  # (N, K)
        if ica.n_iter_ >= args.ica_max_iter:
            print(f"  WARNING: FastICA hit max_iter={args.ica_max_iter}; "
                  "consider raising --ica-max-iter or changing --seed.", flush=True)

        # Canonicalize order and sign so re-runs at the same seed produce
        # stable axes. Sort by whichever non-Gaussianity moment ICA actually
        # maximized (skewness for fun='skew', kurtosis otherwise), then flip
        # each component so its skewness is non-negative — heavy tail on +.
        centered = ica_raw - ica_raw.mean(axis=0)
        var = (centered ** 2).mean(axis=0)
        sd = np.sqrt(np.maximum(var, 1e-12))
        skew = (centered ** 3).mean(axis=0) / (sd ** 3 + 1e-12)
        excess_kurt = (centered ** 4).mean(axis=0) / (var ** 2 + 1e-12) - 3.0

        sort_score = np.abs(skew) if args.ica_fun == "skew" else np.abs(excess_kurt)
        order = np.argsort(-sort_score)
        signs = np.where(skew[order] < 0, -1.0, 1.0)
        W_ica = (signs[:, None] * ica.components_[order]).astype(np.float64)
        ica_raw = ica_raw[:, order] * signs[None, :]
        excess_kurt = excess_kurt[order]
        skew = skew[order] * signs

        # FastICA may converge to slightly off-unit variance; restandardize so
        # downstream regression coefficients stay comparable across components.
        ica_stds = ica_raw.std(axis=0)
        ica_stds[ica_stds < 1e-9] = 1.0
        scores = (ica_raw / ica_stds).astype(np.float32)
        ica_excess_kurt = excess_kurt
        ica_skewness = skew
        primary = skew if args.ica_fun == "skew" else excess_kurt
        primary_name = "|skewness|" if args.ica_fun == "skew" else "|excess kurt|"
        print(f"  ICA converged in {ica.n_iter_} iterations; "
              f"{primary_name} range: {np.abs(primary).min():.3f}.."
              f"{np.abs(primary).max():.3f}")
        if args.ica_fun == "skew":
            print("  (negative tails are baseline by construction — do not "
                  "label the 'negative' side unless |score| ≫ skew expectation)")

    # Persist model state (everything needed to project a new problem).
    save_kwargs = dict(
        kept_feature_mask=keep_feat,
        feature_means=feature_means.astype(np.float32),
        feature_stds=feature_stds.astype(np.float32),
        # OLS coefficients on [rating, size_is_13, intercept]:
        rating_slope=coefs[0].astype(np.float32),
        size_slope=coefs[1].astype(np.float32),
        ols_intercept=coefs[2].astype(np.float32),
        rating_mean=np.array([ratings.mean()], dtype=np.float64),
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        pca_explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        component_stds=component_stds.astype(np.float32),
        rotation=np.array(args.rotation),
    )
    if args.rotation == "ica":
        save_kwargs["ica_unmixing"] = W_ica.astype(np.float32)
        save_kwargs["ica_stds"] = ica_stds.astype(np.float32)
        save_kwargs["ica_excess_kurtosis"] = ica_excess_kurt.astype(np.float32)
        save_kwargs["ica_skewness"] = ica_skewness.astype(np.float32)

    # Content-hash the model state so the PWA can detect refits automatically
    # (read by package_for_pwa.py and surfaced via embedding_meta.json). Only
    # parameters that actually affect projected embeddings are hashed; the
    # diagnostic arrays (variance ratios, kurtosis, skewness) are excluded so
    # cosmetic changes don't invalidate user localStorage caches.
    h = hashlib.sha256()
    h.update(args.rotation.encode())
    fingerprint_keys = [
        "kept_feature_mask", "feature_means", "feature_stds",
        "rating_slope", "size_slope", "ols_intercept",
        "pca_components", "pca_mean", "component_stds",
    ]
    if args.rotation == "ica":
        fingerprint_keys += ["ica_unmixing", "ica_stds"]
    for key in fingerprint_keys:
        h.update(np.ascontiguousarray(save_kwargs[key]).tobytes())
    embedding_version = h.hexdigest()[:16]
    save_kwargs["embedding_version"] = np.array(embedding_version)
    print(f"  embedding_version: {embedding_version}")

    np.savez(os.path.join(out_dir, "pca.npz"), **save_kwargs)

    # Project every problem's full cell stack through the freshly-saved PCA
    # and write the (n_cells, K) fp16 result as embeddings/joint/per_cell/
    # <size>/<tid>.npz. Done here (with the activations already cached on disk
    # and workers operating in parallel) so package_for_pwa just blits the
    # pre-projected bytes into shard sidecars — no PCA matmul or 6912-dim
    # activation reads at packaging time.
    write_per_cell_projections(parts, pca_path=os.path.join(out_dir, "pca.npz"))

    df = pd.DataFrame(scores, columns=[f"embed_{i}" for i in range(K)])
    df.insert(0, "tsumego_id", all_ids)
    df.insert(1, "size", sizes_col)
    df.to_csv(os.path.join(out_dir, "per_problem.csv"), index=False)

    pd.DataFrame({
        "component": np.arange(K),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": cumvar,
    }).to_csv(os.path.join(out_dir, "explained_variance.csv"), index=False)

    extremes = {}
    for k in range(K):
        col = scores[:, k]
        top_idx = np.argsort(col)[-args.top_k_extreme:][::-1]
        bot_idx = np.argsort(col)[:args.top_k_extreme]
        entry = {
            "high": [
                {"tsumego_id": all_ids[i], "size": int(sizes_col[i]), "score": float(col[i])}
                for i in top_idx
            ],
            "low": [
                {"tsumego_id": all_ids[i], "size": int(sizes_col[i]), "score": float(col[i])}
                for i in bot_idx
            ],
        }
        if args.rotation == "ica":
            # Per-component variance ratio is meaningless after rotation
            # (scores standardized to unit variance); record non-Gaussianity
            # stats instead — these are what ICA actually optimized for.
            entry["excess_kurtosis"] = float(ica_excess_kurt[k])
            entry["skewness"] = float(ica_skewness[k])
        else:
            entry["explained_variance_ratio"] = float(pca.explained_variance_ratio_[k])
        extremes[f"component_{k}"] = entry
    with open(os.path.join(out_dir, "extreme_problems.json"), "w") as f:
        json.dump(extremes, f, indent=2)

    full_d = len(keep_feat)  # full feature dim before dropping dead channels
    if args.rotation == "ica":
        # Gradient direction of each ICA score in residualized feature space.
        # cosine remap normalizes rows internally, so the per-row magnitudes
        # introduced by the 1/sqrt(λ) whitening factor don't matter.
        new_components_for_labels = W_ica @ (pca.components_ / component_stds[:, None])
    else:
        new_components_for_labels = pca.components_
    write_or_remap_labels(
        labels_path,
        new_components=new_components_for_labels,
        new_mask=keep_feat,
        full_d=full_d,
        K=K,
        prior_components=prior_components,
        prior_mask=prior_mask,
    )

    if args.rotation == "ica":
        sort_label = "skewness" if args.ica_fun == "skew" else "|excess kurtosis|"
        print(f"\nTop {min(K, 16)} ICA components (sorted by {sort_label}):")
        print(f"  {'k':>3s} {'skew':>10s} {'|kurt|':>10s}")
        for k in range(min(K, 16)):
            print(f"  {k:>3d} {ica_skewness[k]:>+10.3f} "
                  f"{abs(ica_excess_kurt[k]):>10.3f}")
    else:
        print("\nExplained variance (first 16 components):")
        print(f"  {'k':>3s} {'ratio':>8s} {'cumulative':>10s}")
        for k in range(min(K, 16)):
            print(f"  {k:>3d} {pca.explained_variance_ratio_[k]:>8.4f} {cumvar[k]:>10.4f}")
    print(f"\nWrote outputs to {out_dir}/")


if __name__ == "__main__":
    main()
