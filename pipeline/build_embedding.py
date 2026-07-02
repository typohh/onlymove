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
  labels.json             — manually-curated human motif label per PC (positive
                            side only; skew-ICA negative tail is baseline). v3:
                            {bank, components} — labels are anchored (see below)
                            so they survive refits without transitive drift.
  label_anchors.npz       — frozen per-motif direction registry that labels.json's
                            bank keys into; matched against on every refit.

The FastICA rotation warm-starts from the prior fit's unmixing (mapped into the
new PCA basis) so axes converge back to the same directions unless the data
moved enough to relocate them — reproducible refits, minimal PWA cache churn.

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA, IncrementalPCA
from threadpoolctl import threadpool_limits  # sklearn dep; pin BLAS in workers

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.configurations import ALL_BOARD_SIZES  # noqa: E402

SIZES = list(ALL_BOARD_SIZES)


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
    # ratings/ is the rating backend; the embedding residualizes on these ratings so
    # it stays on the same scale as packaging/the client.
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
    # Each file is np.savez_compressed, so reading the correct row still
    # decompresses the whole candidates block (zlib, GIL-released). The reads
    # are independent and write to distinct rows of X, so fan them out across
    # threads — byte-for-byte the same X as the serial fill, just ~cores× faster.
    X = np.empty((n, d), dtype=np.float32)
    n_workers = min(8, (os.cpu_count() or 4))

    def _fill(idx_tid):
        i, tid = idx_tid
        X[i] = _load_correct_vec(act_dir, tid, by_ext)

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for _ in ex.map(_fill, enumerate(ids)):
            done += 1
            if done % 20000 == 0:
                print(f"    [size {size}]   {done}/{n}", flush=True)

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


def compute_ica_warm_start(prior_components, prior_mask,
                           new_pca_components, new_mask,
                           component_stds, full_d):
    """Map the prior fit's IC directions into the new PCA-whitened basis to seed
    FastICA, so it converges back to the same axes unless the data moved enough
    to relocate them. Returns w_init (K, K) float64, or None when the prior is
    shape-incompatible (different K or feature dim → random init).

    The k-th old IC direction, expressed in the new whitened-score coordinates,
    becomes the k-th initial unmixing row. For an old feature-space direction d
    and new (orthonormal) PCA loadings P with whitening stds s, the new whitened
    score is score_j = (x·P_j)/s_j, so
        w_init[k, j] = (d_k · P_j) * s_j .
    FastICA symmetric-decorrelates w_init, so absolute row scale washes out; we
    still row-normalize for conditioning.
    """
    K_old = prior_components.shape[0]
    K_new = new_pca_components.shape[0]
    if K_old != K_new:
        return None
    if prior_mask.shape[0] != full_d or new_mask.shape[0] != full_d:
        return None
    prior_full = _project_to_full_space(prior_components, prior_mask, full_d)   # (K, full_d)
    new_full = _project_to_full_space(new_pca_components, new_mask, full_d)     # (K, full_d)
    C = prior_full @ new_full.T                        # (K, K): old IC k onto new PCA j
    w_init = C * np.asarray(component_stds)[None, :]   # into whitened-score coords
    norms = np.linalg.norm(w_init, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (w_init / norms).astype(np.float64)


def _full_unit_dirs(components, mask, full_d):
    """Project (K, D_kept) loadings to full feature space and unit-normalize rows."""
    full = _project_to_full_space(components, mask, full_d)
    norms = np.linalg.norm(full, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return full / norms


def _load_anchor_store(path):
    """Return (directions list[(full_d,)], motif_ids list[str]); ([], []) if absent."""
    if not os.path.exists(path):
        return [], []
    arc = np.load(path, allow_pickle=False)
    dirs = [row for row in arc["directions"].astype(np.float64)]
    ids = [str(x) for x in arc["motif_ids"]]
    return dirs, ids


def _save_anchor_store(path, directions, motif_ids):
    """Persist the motif registry's frozen directions. Removes the file when the
    registry is empty so retired motifs can't resurrect on the next run."""
    if not motif_ids:
        if os.path.exists(path):
            os.remove(path)
        return
    np.savez(
        path,
        directions=np.asarray(directions, dtype=np.float32),
        motif_ids=np.array([str(m) for m in motif_ids]),
    )


def _next_motif_ids(existing_ids, n):
    """Generate n fresh 'mNNNN' ids that don't collide with existing_ids."""
    nums = [int(m[1:]) for m in existing_ids if m.startswith("m") and m[1:].isdigit()]
    start = (max(nums) + 1) if nums else 0
    return [f"m{start + i:04d}" for i in range(n)]


def _skew_contrast(x):
    """FastICA contrast targeting the 3rd cumulant (skewness) instead of the
    symmetric 4th-cumulant proxy that 'logcosh'/'exp'/'cube' approximate.

    With G(u) = u^3/3, g(u) = G'(u) = u^2 and g'(u) = 2u. Because g'(u) is odd,
    the FastICA update simplifies (the E[g'(w·X)]·w term has zero expectation
    on whitened, mean-zero input). The recovered components are structurally
    one-sided — positive tail concentrates the rare motif, negative tail is
    Gaussian-like baseline — so only the positive side is ever labeled.
    """
    return x ** 2, (2.0 * x).mean(axis=-1)


_LABELS_DOC = (
    "Human-curated semantic motif label per PCA/ICA component (skew-ICA: positive "
    "tail = motif, negative tail = baseline, so only the positive side is labeled). "
    "Set positive=null when no clear pattern. Labels are anchored: each motif owns "
    "a frozen direction in label_anchors.npz keyed by motif_id, and every refit "
    "matches the new axes to those anchors by cosine (never to the last refit), so "
    "labels can't drift transitively across refits. 'bank' is the full motif "
    "registry (may exceed K); 'components' is the current fit's axes keyed by PC "
    "index (positive / positive_i18n are what ships). anchor_cos = match strength. "
    "Edit a component's positive text to relabel; set it to null to retire the "
    "motif. To assign a new label, fill positive on an unmatched component."
)


def update_labels_and_bank(
    labels_path: str,
    anchors_path: str,
    new_components_for_labels: np.ndarray,
    new_mask: np.ndarray,
    full_d: int,
    K: int,
    prior_components=None,
    prior_mask=None,
    cosine_threshold: float = 0.8,
):
    """Refit-stable labeling via an anchored motif bank (see _LABELS_DOC).

    Phase 1 banks the prior fit's human labels: their text is lifted into
    bank[motif_id] and, the first time a motif is seen, its anchor direction is
    frozen from the prior fit's loadings (later refits keep that direction,
    refreshing only the text). Clearing a component (positive=null while its
    motif_id persists and the motif is used nowhere else) retires the motif.

    Phase 2 matches each new axis to the nearest *frozen* anchor by signed cosine
    (>= threshold, greedy 1-1). Matching never compares to the last refit, so
    labels can't drift transitively; the bank may hold more motifs than K (absent
    ones persist for a later fit).

    Writes labels.json v3 ({version, _doc, bank, components}, components keyed by
    current PC index for downstream) plus the binary anchor store.
    """
    old_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            old_labels = json.load(f)
    old_components = old_labels.get("components", {}) or {}
    bank = dict(old_labels.get("bank", {}) or {})

    anchor_rows, anchor_ids = _load_anchor_store(anchors_path)
    have = set(anchor_ids)

    # Active vs explicitly-cleared motif ids referenced by the prior components.
    cleared, active = set(), set()
    for entry in old_components.values():
        mid = entry.get("motif_id") if entry else None
        if not mid:
            continue
        (active if entry.get("positive") is not None else cleared).add(mid)

    # --- Phase 1: bank the prior fit's human labels (text + frozen anchor). ---
    can_bank = (
        prior_components is not None and prior_mask is not None
        and prior_mask.shape[0] == full_d
    )
    if can_bank:
        prior_dirs = _full_unit_dirs(prior_components, prior_mask, full_d)
        pending = []  # (k_old, entry) for labeled prior components
        for k_str, entry in old_components.items():
            if not entry or entry.get("positive") is None:
                continue
            k_old = int(k_str)
            if k_old < prior_dirs.shape[0]:
                pending.append((k_old, entry))
        fresh = _next_motif_ids(
            list(have) + list(bank.keys()),
            sum(1 for _, e in pending if not e.get("motif_id")),
        )
        fi = 0
        for k_old, entry in pending:
            mid = entry.get("motif_id")
            if not mid:
                mid = fresh[fi]
                fi += 1
            text = {"positive": entry.get("positive")}
            if entry.get("positive_i18n") is not None:
                text["positive_i18n"] = entry["positive_i18n"]
            bank[mid] = text  # refresh (possibly hand-edited) text
            if mid not in have:  # freeze the anchor direction only once
                have.add(mid)
                anchor_ids.append(mid)
                anchor_rows.append(prior_dirs[k_old])

    # Retire explicitly-cleared motifs (label removed, used nowhere else).
    retire = cleared - active
    if retire:
        for mid in retire:
            bank.pop(mid, None)
        kept = [i for i, mid in enumerate(anchor_ids) if mid not in retire]
        anchor_rows = [anchor_rows[i] for i in kept]
        anchor_ids = [anchor_ids[i] for i in kept]

    # --- Phase 2: match every new axis to the nearest frozen anchor. ---
    new_dirs = _full_unit_dirs(new_components_for_labels, new_mask, full_d)  # (K, full_d)
    components = {str(k): {"positive": None, "motif_id": None} for k in range(K)}
    if anchor_ids:
        sim = new_dirs @ np.asarray(anchor_rows).T  # (K, M) signed cosine
        triples = []
        for k in range(sim.shape[0]):
            for m in range(sim.shape[1]):
                c = float(sim[k, m])
                if c >= cosine_threshold:  # positive + confident
                    triples.append((c, k, m))
        triples.sort(reverse=True)
        used_axis, used_motif = set(), set()
        for c, k, m in triples:
            if k in used_axis or m in used_motif:
                continue
            used_axis.add(k)
            used_motif.add(m)
            mid = anchor_ids[m]
            slot = {
                "positive": bank.get(mid, {}).get("positive"),
                "motif_id": mid,
                "anchor_cos": round(c, 4),
            }
            i18n = bank.get(mid, {}).get("positive_i18n")
            if i18n is not None:
                slot["positive_i18n"] = i18n
            components[str(k)] = slot

    labels = {
        "version": 3,
        "_doc": _LABELS_DOC,
        "bank": bank,
        "components": components,
    }
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    _save_anchor_store(anchors_path, anchor_rows, anchor_ids)

    n_matched = sum(1 for v in components.values() if v.get("positive") is not None)
    print(f"  labels: {n_matched}/{K} axes matched to bank "
          f"({len(anchor_ids)} anchors, {len(bank)} bank entries, "
          f"cosine >= {cosine_threshold})")


def _decode_pca_arc(pca_arc):
    """Decode pca.npz once into the float64 arrays + precomputed constants that
    _project_decoded needs, so a caller projecting many problems pays the npz
    read / astype / standardization-fold once instead of per call.

    The transforms _project_decoded applies are algebraically equal to the
    explicit z-score → residualize → center → standardize sequence in main();
    folding constants here only reassociates the float ops (negligible rounding,
    well under the fp16 the embeddings ship in):
      - inv_stds = 1/feature_stds replaces the per-feature division;
      - affine_const = -(feature_means/feature_stds) - pca_mean is the
        problem-independent part of the centering (the per-problem part is the
        OLS offset, subtracted in _project_decoded);
      - comp_scaled_T folds /component_stds into the (transposed, contiguous)
        PCA loadings; ica_scaled_T folds /ica_stds into the unmixing.
    """
    kept = pca_arc["kept_feature_mask"].astype(bool)
    feature_means = pca_arc["feature_means"].astype(np.float64)
    feature_stds = pca_arc["feature_stds"].astype(np.float64)
    inv_stds = 1.0 / feature_stds  # kept features all have std >= std_floor
    pca_mean = pca_arc["pca_mean"].astype(np.float64)
    pca_components = pca_arc["pca_components"].astype(np.float64)
    component_stds = pca_arc["component_stds"].astype(np.float64)
    component_stds = np.where(component_stds < 1e-9, 1.0, component_stds)
    P = {
        "all_kept": bool(kept.all()),
        "kept": kept,
        "inv_stds": inv_stds,
        "affine_const": -(feature_means * inv_stds) - pca_mean,
        "rating_slope": pca_arc["rating_slope"].astype(np.float64),
        "size_slope": pca_arc["size_slope"].astype(np.float64),
        "ols_intercept": pca_arc["ols_intercept"].astype(np.float64),
        "comp_scaled_T": np.ascontiguousarray(
            (pca_components / component_stds[:, None]).T),   # (D_kept, K)
    }
    if "ica_unmixing" in pca_arc.files:
        ica_unmixing = pca_arc["ica_unmixing"].astype(np.float64)
        ica_stds = pca_arc["ica_stds"].astype(np.float64)
        ica_stds = np.where(ica_stds < 1e-9, 1.0, ica_stds)
        P["ica_scaled_T"] = np.ascontiguousarray(
            (ica_unmixing / ica_stds[:, None]).T)            # (K, K)
    return P


def _project_decoded(activations, rating, size_is_13, P):
    """Project raw activations to the K-dim embedding using arrays already
    decoded by _decode_pca_arc. See project_to_embedding for the contract."""
    V = np.asarray(activations, dtype=np.float64)
    single = (V.ndim == 1)
    if single:
        V = V[None, :]
    # When no channels were dropped (the common case) the boolean fancy-index
    # is an exact copy of V — skip it.
    V_kept = V if P["all_kept"] else V[:, P["kept"]]         # (N, D_kept)
    # offset = OLS residualization shift (constant across a problem's cells).
    offset = P["rating_slope"] * rating + P["size_slope"] * size_is_13 + P["ols_intercept"]
    centered = V_kept * P["inv_stds"]                         # (N, D_kept)
    centered += P["affine_const"] - offset
    pca_scores = centered @ P["comp_scaled_T"]               # (N, K), /std folded in
    out = pca_scores @ P["ica_scaled_T"] if "ica_scaled_T" in P else pca_scores
    out = out.astype(np.float32)
    return out[0] if single else out


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
    return _project_decoded(activations, rating, size_is_13, _decode_pca_arc(pca_arc))


# --- Per-cell projection: a post-fit pass that projects every problem's full
# (n_cells, 6912) activation block through the freshly-fit PCA/ICA and saves
# the resulting (n_cells, K) fp16 array. This makes package_for_pwa cheap —
# no PCA matmul or raw-activation reads at packaging time, just a small
# pre-projected file per problem.
#
# Parallelized via ProcessPoolExecutor. Each worker decodes pca.npz once and
# then processes its share of problems. Embarrassingly parallel — no shared
# mutable state, modest per-task memory.

_PCA_DECODED_WORKER = None
_THREAD_LIMITER = None


def _init_per_cell_worker(pca_path):
    """ProcessPoolExecutor initializer: decode pca.npz once per worker and pin
    BLAS to a single thread.

    Pinning matters: with N worker processes each defaulting to all cores, the
    small per-problem matmuls oversubscribe the CPU and run several-fold slower
    (measured ~4× on an 8-core box). One thread per process lets N processes
    cleanly fill N cores. The limiter is stashed in a module global so it stays
    in effect for the worker's lifetime.
    """
    global _PCA_DECODED_WORKER, _THREAD_LIMITER
    try:
        _THREAD_LIMITER = threadpool_limits(limits=1)
    except Exception:
        _THREAD_LIMITER = None
    with np.load(pca_path, allow_pickle=False) as arc:
        _PCA_DECODED_WORKER = _decode_pca_arc(arc)


def _project_one_for_per_cell(task):
    """Worker function: read one problem's .npz, project all cells, save
    pre-projected (n_cells, K) fp16 + xys to the per_cell output dir.
    Returns (tid, status_str) for logging."""
    size, tid, raw_rating, in_dir, out_dir = task
    out_path = os.path.join(out_dir, tid + ".npz")

    def _fail(status):
        # Never leave a PRIOR fit's projection behind on failure: packaging
        # would ship that stale file under the NEW embedding_version — exactly
        # the staleness the version gate exists to prevent. Deleting it makes
        # packaging's missing-file check fire loudly instead of shipping stale.
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return tid, status

    npz_path = os.path.join(in_dir, tid + ".npz")
    if not os.path.exists(npz_path):
        return _fail("missing_npz")
    try:
        with np.load(npz_path) as data:
            candidates = np.asarray(data["candidates"])  # (n_cells, 6912) fp16
            xys = np.asarray(data["xys"])                # (n_cells, 2)
    except Exception as e:
        return _fail(f"load_failed:{type(e).__name__}")

    size_is_13 = 1.0 if size == 13 else 0.0
    try:
        projected = _project_decoded(candidates, raw_rating, size_is_13,
                                     _PCA_DECODED_WORKER)
    except Exception as e:
        return _fail(f"project_failed:{type(e).__name__}")

    # Uncompressed (STORED): the payload is ~94% fp16 embedding data, which is
    # high-entropy and only deflates ~8%. Not worth the inflate cost paid on
    # every load during packaging (package_for_pwa reads all ~20k per run).
    # Write to a temp file then os.replace so a crash mid-write can't leave a
    # torn npz that a later packaging run reads as valid.
    tmp_path = out_path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(
                fh,
                embeddings=projected.astype(np.float16),
                xys=xys.astype(np.int32),
            )
        os.replace(tmp_path, out_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return _fail(f"save_failed:{type(e).__name__}")
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


def rotate_with_ica(scores, args, pca, keep_feat, component_stds,
                    prior_components, prior_mask):
    """FastICA rotation within the whitened K-dim PCA subspace (the
    args.rotation == 'ica' path). Preserves the linear span but seeks maximally
    non-Gaussian axes, then canonicalizes their order + sign and restandardizes
    to unit variance. Returns (scores, ica_unmixing, ica_stds, ica_excess_kurt,
    ica_skewness) with `scores` replaced by the rotated scores."""
    fun = _skew_contrast if args.ica_fun == "skew" else args.ica_fun
    print(f"\nFitting FastICA n_components={scores.shape[1]} on whitened PCA scores "
          f"(fun={args.ica_fun!r}, whiten=False)...", flush=True)
    # Warm-start from the prior fit's unmixing (mapped into the new PCA basis)
    # so axes converge back to the same directions unless the data moved —
    # reproducible refits + far less PWA cache churn. Random init otherwise.
    w_init = None
    if args.ica_warm_start == "auto" and prior_components is not None:
        w_init = compute_ica_warm_start(
            prior_components, prior_mask,
            pca.components_, keep_feat, component_stds, len(keep_feat),
        )
        print("  warm-start: seeded FastICA from prior unmixing"
              if w_init is not None else
              "  warm-start: prior fit shape-incompatible; using random init")
    # n_components is intentionally omitted: with whiten=False sklearn
    # ignores it and uses the input dim, which is already K from PCA.
    ica = FastICA(
        whiten=False,
        fun=fun,
        max_iter=args.ica_max_iter,
        tol=args.ica_tol,
        w_init=w_init,
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
    primary = skew if args.ica_fun == "skew" else excess_kurt
    primary_name = "|skewness|" if args.ica_fun == "skew" else "|excess kurt|"
    print(f"  ICA converged in {ica.n_iter_} iterations; "
          f"{primary_name} range: {np.abs(primary).min():.3f}.."
          f"{np.abs(primary).max():.3f}")
    if args.ica_fun == "skew":
        print("  (negative tails are baseline by construction — only the "
              "positive tail carries a motif, so labels are positive-only)")
    return scores, W_ica, ica_stds, excess_kurt, skew


def write_score_artifacts(out_dir, scores, all_ids, sizes_col, K, args,
                          pca, cumvar, ica_excess_kurt, ica_skewness):
    """Write the score-derived output files: per_problem.csv (consumed by
    packaging + fit_population_constants), explained_variance.csv and
    extreme_problems.json (inspection)."""
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
    parser.add_argument("--ica-warm-start", choices=["auto", "off"], default="auto",
                        help="Seed FastICA from the prior fit's unmixing (remapped "
                             "into the new PCA basis) so axes stay put across refits "
                             "unless the data moved. 'auto' warm-starts when a "
                             "shape-compatible prior exists; 'off' uses random init.")
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
    anchors_path = os.path.join(out_dir, "label_anchors.npz")
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
    W_ica = ica_stds = ica_excess_kurt = ica_skewness = None
    if args.rotation == "ica":
        scores, W_ica, ica_stds, ica_excess_kurt, ica_skewness = rotate_with_ica(
            scores, args, pca, keep_feat, component_stds,
            prior_components, prior_mask)

    # Persist model state (everything needed to project a new problem).
    save_kwargs = dict(
        kept_feature_mask=keep_feat,
        feature_means=feature_means.astype(np.float32),
        feature_stds=feature_stds.astype(np.float32),
        # OLS coefficients on [rating, size_is_13, intercept]:
        rating_slope=coefs[0].astype(np.float32),
        size_slope=coefs[1].astype(np.float32),
        ols_intercept=coefs[2].astype(np.float32),
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

    write_score_artifacts(out_dir, scores, all_ids, sizes_col, K, args,
                          pca, cumvar, ica_excess_kurt, ica_skewness)

    full_d = len(keep_feat)  # full feature dim before dropping dead channels
    if args.rotation == "ica":
        # Gradient direction of each ICA score in residualized feature space.
        # anchor matching normalizes rows internally, so the per-row magnitudes
        # introduced by the 1/sqrt(λ) whitening factor don't matter.
        new_components_for_labels = W_ica @ (pca.components_ / component_stds[:, None])
    else:
        new_components_for_labels = pca.components_
    update_labels_and_bank(
        labels_path,
        anchors_path=anchors_path,
        new_components_for_labels=new_components_for_labels,
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
