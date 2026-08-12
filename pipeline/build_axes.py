"""Build the per-problem SKILL AXES shipped to the PWA (replaces the retired
skew-ICA build_embedding pipeline, 2026-07-18 — validation in
docs/methodology.md).

Trains the supervised residual-axes model (utils.rating.residual_axes: fixed regimen,
literal rating-trainer feature parity, incremental K selected by held-out
walk-forward LL gain), then writes the same artifact contract the packaging
step consumes — so pipeline.package_for_pwa and the PWA client are unchanged
in format, only K shrinks from 64 to the selected axis count:

  axes/per_problem.csv        tsumego_id, size, embed_0..embed_{K-1}
  axes/per_cell/<size>/<tid>.npz   embeddings (n_cells, K) fp16 + xys
  axes/axes.npz               model + content-hash embedding_version
  axes/labels.json            v4 — TWO-TAILED labels (see below)
  axes/label_anchors.npz      frozen per-motif loading anchors

Per-cell values: the axes are linear in the trainer features and only the
joint-PCA activation block varies per cell, so a cell's axis value is the
problem's shipped value plus a weighted delta of (cell activation z − correct
cell activation z) — every other feature block is position-level and constant.

Labels are TWO-TAILED (v4): each component may name its positive tail, its
negative tail, or both — some axes only have a nameable motif on one side.
Axes keep their identity across refits via frozen per-motif loading anchors
(corr >= 0.8 over shared problems, greedy 1-1); an axis matching an anchor
with NEGATIVE correlation is sign-flipped to the anchor's canonical
orientation before anything is written, so labels never swap sides. Unmatched
axes get fresh motif ids with null labels; those ship and work like any other
axis, and the client simply says nothing about a tail it has no name for, so
naming one in labels.json is what makes it readable rather than shippable.

Usage: ./.venv/bin/python3 -m pipeline.build_axes     (no flags — production)
"""

import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.residual_axes import select_and_fit  # noqa: E402
from utils.rating.features import (  # noqa: E402
    _cache_ids_path, load_joint_activation_basis,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "axes")
ANCHOR_MATCH_MIN = 0.8
LABELS_SPACE = "residual_axes_v1"
PER_CELL_WORKERS = os.cpu_count() or 4

_W = None  # per-cell worker state


# ------------------------------------------------------------------ anchors/labels
def _load_labels_and_anchors():
    """Current-space labels.json + anchors; an ICA-era labels.json is archived
    and a fresh bank started (the spaces are incommensurable)."""
    lpath = os.path.join(OUT_DIR, "labels.json")
    apath = os.path.join(OUT_DIR, "label_anchors.npz")
    labels = None
    if os.path.exists(lpath):
        with open(lpath) as f:
            labels = json.load(f)
        if labels.get("space") != LABELS_SPACE:
            legacy = os.path.join(OUT_DIR, "labels_legacy_ica.json")
            os.replace(lpath, legacy)
            print(f"  archived pre-axes labels.json -> {legacy}")
            # The anchors belong to the space being retired, so they cannot be
            # matched against — but they are still the only record of those
            # motifs' directions, and the fresh bank below would otherwise let
            # the write overwrite them. Move them aside with their labels.
            if os.path.exists(apath):
                legacy_a = os.path.join(OUT_DIR, "label_anchors_legacy_ica.npz")
                os.replace(apath, legacy_a)
                print(f"  archived pre-axes anchors -> {legacy_a}")
            labels = None
    if labels is None:
        labels = {"version": 4, "space": LABELS_SPACE,
                  "_doc": ("Two-tailed axis labels. components[k] names the "
                           "k-th shipped axis: positive/negative label either "
                           "tail's motif (null = unnamed; an unnamed tail is "
                           "shipped and used, the client just says nothing "
                           "about it). bank persists labels for motifs absent "
                           "from the current fit. Anchors (label_anchors.npz) "
                           "freeze each motif's loading direction; edit TEXT "
                           "here, never anchors."),
                  "bank": {}, "components": []}
    anchors = {"tids": None, "ids": [], "M": None}
    if os.path.exists(apath) and labels["bank"]:
        d = np.load(apath, allow_pickle=False)
        if "tids" in d.files:
            anchors = {"tids": [str(t) for t in d["tids"]],
                       "ids": [str(i) for i in d["ids"]], "M": d["M"]}
    return labels, anchors


def _match_axes_to_anchors(loadings, tids, anchors):
    """Greedy 1-1 |corr| matching of new axes to frozen anchors on shared
    problems. Returns per-axis (motif_id or None, sign)."""
    k = loadings.shape[1]
    if anchors["M"] is None or not anchors["ids"]:
        return [(None, 1.0)] * k
    tid_index = {t: i for i, t in enumerate(tids)}
    common = [(i, tid_index[t]) for i, t in enumerate(anchors["tids"])
              if t in tid_index]
    ai = np.array([c[0] for c in common])
    ni = np.array([c[1] for c in common])
    A = anchors["M"][:, ai]                     # (n_anchors, n_common)
    N = loadings[ni, :].T                       # (k, n_common)
    Az = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
    Nz = (N - N.mean(1, keepdims=True)) / (N.std(1, keepdims=True) + 1e-9)
    C = Nz @ Az.T / Az.shape[1]                 # (k, n_anchors) corr
    out = [(None, 1.0)] * k
    used = set()
    order = np.dstack(np.unravel_index(
        np.argsort(-np.abs(C), axis=None), C.shape))[0]
    for kk, aa in order:
        if abs(C[kk, aa]) < ANCHOR_MATCH_MIN:
            break
        if out[kk][0] is not None or aa in used:
            continue
        out[kk] = (anchors["ids"][aa], 1.0 if C[kk, aa] >= 0 else -1.0)
        used.add(aa)
    return out


def _update_labels(labels, anchors, loadings, tids, matches):
    """Materialize components for this fit; bank new motifs; extend anchors
    with the new motifs' frozen directions. Returns (labels, anchors)."""
    prev_by_motif = {c.get("motif_id"): c for c in labels.get("components", [])
                     if c.get("motif_id")}
    # bank the previous components' text (curation done since last run)
    for mid, comp in prev_by_motif.items():
        b = labels["bank"].setdefault(mid, {})
        for f in ("positive", "positive_i18n", "negative", "negative_i18n"):
            b[f] = comp.get(f)
    components = []
    new_ids = []
    for k in range(loadings.shape[1]):
        mid, _sign = matches[k]
        if mid is None:
            mid = "ax_" + hashlib.sha256(
                loadings[:, k].tobytes()).hexdigest()[:8]
            labels["bank"].setdefault(
                mid, {"positive": None, "positive_i18n": None,
                      "negative": None, "negative_i18n": None})
            new_ids.append((mid, k))
        b = labels["bank"][mid]
        components.append({"motif_id": mid,
                           "positive": b.get("positive"),
                           "positive_i18n": b.get("positive_i18n"),
                           "negative": b.get("negative"),
                           "negative_i18n": b.get("negative_i18n")})
    labels["components"] = components
    # extend anchors with the new motifs, frozen at THIS fit's direction
    # (existing anchor rows are NEVER rewritten — the anti-drift invariant)
    if new_ids:
        if anchors["M"] is None:
            anchors = {"tids": list(tids), "ids": [],
                       "M": np.zeros((0, len(tids)), dtype=np.float32)}
        tid_index = {t: i for i, t in enumerate(tids)}
        idx = [tid_index.get(t) for t in anchors["tids"]]
        rows = []
        for mid, k in new_ids:
            rows.append(np.array(
                [loadings[i, k] if i is not None else 0.0 for i in idx],
                dtype=np.float32))
            anchors["ids"].append(mid)
        anchors["M"] = np.vstack([anchors["M"]] + [r[None, :] for r in rows])
    return labels, anchors


# ------------------------------------------------------------------ per-cell fanout
def _init_worker(delta_full, prob_vals, act_dir_by_size):
    global _W
    _W = (delta_full, prob_vals, act_dir_by_size)


def _project_one(task):
    size, tid = task
    delta_full, prob_vals, act_dir_by_size = _W
    src = os.path.join(act_dir_by_size[size], tid + ".npz")
    out_dir = os.path.join(OUT_DIR, "per_cell", str(size))
    with np.load(src, allow_pickle=False) as d:
        cand = np.asarray(d["candidates"], dtype=np.float32)
        xys = np.asarray(d["xys"], dtype=np.int32)
        ci = int(d["correct_idx"])
    # raw activations straight to axis space — the basis means cancel in the
    # difference and its scalings are folded into delta_full (write_per_cell)
    delta = (cand - cand[ci][None, :]) @ delta_full         # (n_cells, K)
    vals = (prob_vals[tid][None, :] + delta).astype(np.float16)
    tmp = os.path.join(out_dir, f".{tid}.tmp.npz")
    np.savez_compressed(tmp, embeddings=vals, xys=xys)
    os.replace(tmp, os.path.join(out_dir, tid + ".npz"))
    return 1


def write_per_cell(fit, tid_size):
    """Fan the axis projection out to every legal cell of every problem."""
    from pipeline.train_rating import PCA_DIM, SEED as PCA_SEED, SIZES
    tids_by_size = {s: json.load(open(_cache_ids_path(s))) for s in SIZES}
    load_joint_activation_basis(SIZES, tids_by_size, PCA_DIM, PCA_SEED)
    # basis is cached in the joint-PCA npz; workers re-open it directly
    from utils.rating.features import _joint_pca_cache_path
    basis_path = _joint_pca_cache_path(SIZES, tids_by_size, PCA_DIM, PCA_SEED)

    act_dim = fit["act_dim"]
    # delta weights: d(shipped value)/d(activation z), per axis —
    # W acts on F standardized by feat_sd; loadings scale by v_sd/rms; the
    # shipped value further standardizes by axis_sd.
    scale = fit["v_sd"] / (fit["rms"] * fit["axis_sd"])       # (K,)
    delta_w = (fit["W"][:, :act_dim] / fit["feat_sd"][None, :act_dim]).T \
        * scale[None, :]                                      # (act_dim, K)

    # Fold the ENTIRE per-cell projection into one matrix, applied to the RAW
    # activations. A cell ships prob_value + (z(cell) - z(correct cell)) @
    # delta_w, and taking that difference annihilates every additive term in z
    # — the activation mean, the PCA mean and the post-PCA mean alike — while
    # what remains (two diagonal scalings and the PCA rotation) is linear and
    # composes. So the per-problem work becomes ONE matmul, K columns wide
    # instead of PCA_DIM, with no full-width intermediates: worth doing because
    # the PCA_DIM-wide matmul is small enough that threaded BLAS spends longer
    # coordinating than computing, and the fanout runs one process per core.
    d = np.load(basis_path, allow_pickle=False)
    C, s1, s2 = d["basis_C"], d["basis_s1"], d["basis_s2"]    # C is (act_dim, raw)
    if C.shape[0] != act_dim:
        raise RuntimeError(
            f"activation block is {act_dim} wide but the joint-PCA basis is "
            f"{C.shape[0]}; the fold below assumes the activation features ARE "
            "the PCA output.")
    delta_full = (((C.T / s1[:, None]) / s2[None, :])
                  @ delta_w).astype(np.float32)               # (raw, K)
    prob_vals = {t: fit["loadings"][i]
                 for i, t in enumerate(fit["tids"])}
    act_dir_by_size = {s: os.path.join(PROJECT_ROOT, "activations", str(s))
                       for s in SIZES}
    tasks = [(tid_size[t], t) for t in fit["tids"]]
    for s in SIZES:
        os.makedirs(os.path.join(OUT_DIR, "per_cell", str(s)), exist_ok=True)
    done = 0
    with ProcessPoolExecutor(
            max_workers=PER_CELL_WORKERS, initializer=_init_worker,
            initargs=(delta_full, prob_vals, act_dir_by_size)) as ex:
        for _ in ex.map(_project_one, tasks, chunksize=200):
            done += 1
            if done % 20000 == 0:
                print(f"  per-cell: {done}/{len(tasks)}", flush=True)
    print(f"  per-cell: {done}/{len(tasks)} written")


# ------------------------------------------------------------------ main
def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit = select_and_fit(device=device)
    tids = fit["tids"]

    labels, anchors = _load_labels_and_anchors()
    # Motif identity is APPEND-ONLY, and these two snapshots are what proves it
    # at the write below. A motif absent from this fit keeps its banked label
    # text and its frozen anchor row so a later fit can recognize it again; the
    # anchor row is the ONLY record of what that motif's direction was, so
    # dropping one is unrecoverable by re-running anything.
    prior_anchor_ids = set(anchors["ids"])
    prior_bank_ids = set(labels["bank"])
    matches = _match_axes_to_anchors(fit["loadings"], tids, anchors)
    # canonical orientation: flip any axis that matched an anchor negatively,
    # BEFORE anything is written — labels never swap sides
    for k, (mid, sign) in enumerate(matches):
        if sign < 0:
            fit["loadings"][:, k] *= -1.0
            fit["W"][k, :] *= -1.0
            matches[k] = (mid, 1.0)
    labels, anchors = _update_labels(labels, anchors, fit["loadings"], tids,
                                     matches)

    # Checked BEFORE the first write, not beside the write it guards: an abort
    # partway through leaves the artifacts disagreeing with each other, and a
    # rewritten axes.npz next to the previous labels and anchors is exactly the
    # inconsistent state this whole guard exists to prevent.
    lost_anchors = sorted(prior_anchor_ids - set(anchors["ids"]))
    lost_bank = sorted(prior_bank_ids - set(labels["bank"]))
    if lost_anchors or lost_bank:
        raise RuntimeError(
            "axis identity would be LOST by this run — refusing before any "
            "artifact is touched.\n"
            f"  anchor directions dropped: {lost_anchors}\n"
            f"  banked labels dropped    : {lost_bank}\n"
            "Anchors and bank entries are append-only: a motif missing from "
            "the current fit must keep both so a later fit can recognize and "
            "re-adopt it. A frozen anchor cannot be regenerated — it is the "
            "only surviving record of that motif's direction.")

    os.makedirs(OUT_DIR, exist_ok=True)
    # per_problem.csv — same contract packaging always read. tid->size comes
    # from the SGF corpus (corpus_tid_size), NEVER the cache id lists — a stale
    # cache id in the other size's list would mis-size the row (000d0d75).
    from utils.rating.residual_axes import corpus_tid_size
    tid_size = corpus_tid_size()
    import csv
    k_sel = fit["k_sel"]
    pp = os.path.join(OUT_DIR, "per_problem.csv")
    with open(pp + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tsumego_id", "size"] + [f"embed_{i}" for i in range(k_sel)])
        for i, t in enumerate(tids):
            w.writerow([t, tid_size[t]]
                       + [f"{v:.5f}" for v in fit["loadings"][i]])
    os.replace(pp + ".tmp", pp)

    # axes.npz — the model + the content-hash version packaging ships
    h = hashlib.sha256()
    for arr in (fit["W"], fit["feat_mu"], fit["feat_sd"], fit["axis_mu"],
                fit["axis_sd"], fit["rms"], fit["v_sd"]):
        h.update(np.ascontiguousarray(arr).tobytes())
    version = h.hexdigest()[:12]
    np.savez_compressed(
        os.path.join(OUT_DIR, "axes.npz"),
        embedding_version=np.array(version),
        k=np.array(k_sel), act_dim=np.array(fit["act_dim"]),
        W=fit["W"], rms=fit["rms"], v_sd=fit["v_sd"],
        feat_mu=fit["feat_mu"], feat_sd=fit["feat_sd"],
        axis_mu=fit["axis_mu"], axis_sd=fit["axis_sd"],
        gains_k=np.array(sorted(fit["gains"])),
        gains=np.array([fit["gains"][k] for k in sorted(fit["gains"])]),
        # the same candidates under per-USER weighting, recorded beside the
        # attempt-pooled figures the selection reads (residual_axes.axes_ll_gain)
        gains_per_user=np.array([fit["per_user_gains"][k]
                                 for k in sorted(fit["gains"])]),
        motif_ids=np.array([m for m, _ in matches]))

    with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT_DIR, "label_anchors.npz"),
        tids=np.array(anchors["tids"]), ids=np.array(anchors["ids"]),
        M=anchors["M"])

    write_per_cell(fit, tid_size)

    unlabeled = [i for i, c in enumerate(labels["components"])
                 if not (c.get("positive") or c.get("negative"))]
    print(f"\nbuild_axes complete: K={k_sel}, embedding_version={version}")
    if unlabeled:
        print(f"  NOTE: axes {unlabeled} are UNLABELED — they ship and steer "
              f"selection, but name a tail in labels.json to have the client "
              f"surface them as findings")


if __name__ == "__main__":
    main()
