"""The frozen activation encoder: raw trunk vectors -> the stored M-dim encoding.

One immutable linear map — per-channel standardization folded together with a
PCA projection — that defines what `activations/<size>/<id>.npz` stores in
place of the raw 6912-dim trunk vectors. Everything downstream (the joint-PCA
feature basis, the per-cell axis projection) fits INSIDE the encoded space, so
the encoder is the one artifact that cannot be regenerated once the raw
activations are gone: it stays tracked in git, and every encoded file carries
its 4-byte fingerprint so a store written under one encoder can never be read
through another silently.

Fit-set policy, refresh procedure and the M=512 measurement live with the
fitting step (pipeline.fit_activation_encoder); this module is only the pure
math shared by the fitter, the converter and every reader, and deliberately
imports nothing from utils.katago.
"""

import hashlib
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODER_PATH = os.path.join(PROJECT_ROOT, "activations", "encoder.npz")

_ARRAYS = ("m1", "s1", "pca_mean", "C")  # fingerprint hashes these, in this order


def fingerprint(enc) -> int:
    """First 4 bytes of sha256 over the encoder's arrays, as a uint32.
    Stamped into every encoded npz; collision odds are irrelevant at the
    handful of encoders this repo will ever mint."""
    h = hashlib.sha256()
    for name in _ARRAYS:
        h.update(np.ascontiguousarray(enc[name], dtype=np.float32).tobytes())
    return int.from_bytes(h.digest()[:4], "big")


def fingerprint_hex(enc) -> str:
    return f"{fingerprint(enc):08x}"


def fit_encoder(rows: np.ndarray, m: int, seed: int = 0) -> dict:
    """Fit the encoder on (n_rows, raw_dim) rows (any float dtype; fp16 input
    stays fp16 — one fp32 working buffer is the only full-size allocation, so
    peak memory is rows + that buffer, not the several transient copies the
    naive astype/standardize/PCA chain makes). Returns the encoder dict plus
    the fit's cumulative explained-variance at m under "evr"."""
    from sklearn.decomposition import PCA

    n, d = rows.shape
    chunk = 4096
    total = np.zeros(d, dtype=np.float64)
    total_sq = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        c = rows[i:i + chunk].astype(np.float32)
        total += c.sum(axis=0, dtype=np.float64)
        total_sq += np.square(c, out=c).sum(axis=0, dtype=np.float64)
    m1 = (total / n).astype(np.float32)
    s1 = np.sqrt(np.maximum(total_sq / n - (total / n) ** 2, 0.0)).astype(np.float32)
    s1[s1 < 1e-6] = 1e-6

    Z = np.empty((n, d), dtype=np.float32)
    for i in range(0, n, chunk):
        Z[i:i + chunk] = (rows[i:i + chunk].astype(np.float32) - m1) / s1
    pca = PCA(n_components=m, random_state=seed, svd_solver="randomized",
              copy=False).fit(Z)
    del Z
    return {
        "m1": m1.astype(np.float32),
        "s1": s1.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "C": pca.components_.astype(np.float32),
        "evr": float(np.sum(pca.explained_variance_ratio_)),
    }


def apply_encoder(enc, X_raw: np.ndarray) -> np.ndarray:
    """(n, raw_dim) raw trunk vectors -> (n, M) float16 stored encoding."""
    X = X_raw.astype(np.float32, copy=False)
    if X.shape[1] != enc["C"].shape[1]:
        raise ValueError(
            f"encoder expects raw dim {enc['C'].shape[1]}, got {X.shape[1]} — "
            f"this input is not raw trunk activations")
    Z = (X - enc["m1"]) / enc["s1"]
    return ((Z - enc["pca_mean"]) @ enc["C"].T).astype(np.float16)


def save_encoder(enc, path: str = ENCODER_PATH):
    # np.savez appends ".npz" to any filename that lacks it, so the temp name
    # must already end in .npz or the atomic rename finds nothing.
    tmp = path + ".tmp.npz"
    np.savez(tmp, fingerprint=np.uint32(fingerprint(enc)),
             **{k: enc[k] for k in _ARRAYS})
    os.replace(tmp, path)


def load_encoder(path: str = ENCODER_PATH) -> dict:
    """Load and verify: a stored fingerprint that no longer matches the arrays
    means a corrupt or hand-edited artifact, which must never quietly encode."""
    with np.load(path) as d:
        enc = {k: d[k] for k in _ARRAYS}
        stored = int(d["fingerprint"])
    actual = fingerprint(enc)
    if stored != actual:
        raise RuntimeError(
            f"encoder at {path} is corrupt: stored fingerprint {stored:08x} "
            f"!= recomputed {actual:08x}")
    return enc
