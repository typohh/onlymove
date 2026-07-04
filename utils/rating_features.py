"""Shared rating-model input assembly — the data + features the rating trainer consumes.

This is the production home for the helpers that load the rating model's training data
(first-time human attempts), turn it into the ability unit (2-month overlapping profiles),
and assemble the per-problem feature trunk (joint-PCA correct-move activations, common-column
bot winrate, candidate-move evaluations). It also owns the activation "signal cache" path
helpers those feature loaders read.

These functions used to live in auxiliary/ (the research area) and were imported back into the
pipeline; they are leaf utilities (no auxiliary/ or pipeline/ imports), so both the production
trainer (pipeline.train_rating) and the experiment scripts (auxiliary.rating_dev.*) import
them from here. The auxiliary modules that historically defined them now re-export from here so
their other callers are unaffected.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SQL_PATH = os.path.join(PROJECT_ROOT, "attempts", "database.sql")
SQLITE_PATH = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")

# Raw KataGo trunk activation layout (18 residual blocks × 384 channels), captured at
# the correct move by pipeline.extract_activations. Feature index = layer*384 + channel.
N_LAYERS = 18
CHANNELS_PER_LAYER = 384
N_FEATS = N_LAYERS * CHANNELS_PER_LAYER  # 6912

# Number of contrastive blocks: per-channel [mean, std, max, zscore] aggregates over the
# OTHER legal moves (N_CONTRAST_BLOCKS × N_FEATS), id-aligned with the correct-vec cache.
N_CONTRAST_BLOCKS = 4

# The activation "signal cache": correct-move trunk vectors written once per board by
# auxiliary.inspect_activation_signals._build_cache and read directly by load_joint_activations.
# The optional contrastive cache (the other-move aggregates) is written by
# auxiliary.predict_rating_from_activations --contrastive into the same directory.
CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")


def _cache_vecs_path(size):
    return os.path.join(CACHE_DIR, f"correct_vecs_{size}.npy")


def _cache_ids_path(size):
    return os.path.join(CACHE_DIR, f"correct_ids_{size}.json")


def _contrastive_cache_path(size):
    return os.path.join(CACHE_DIR, f"contrastive_vecs_{size}.npy")


def load_first_time_attempts():
    """Return a DataFrame of first-time attempts: one row per (user_id, problem),
    columns [user_id, problem, success, size, created_at].

    Reads the canonical binary attempts/database.sqlite directly (opened
    read-only) instead of replaying the ~290k-INSERT text dump into an in-memory
    DB on every call: fetch_database keeps the two artifacts in sync and treats
    the binary as the merge target, so this is the cheaper, equivalent source."""
    if not os.path.exists(SQLITE_PATH):
        raise SystemExit(
            f"Missing {SQLITE_PATH}. Run `python -m pipeline.fetch_database` first.")
    con = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT user_id, problem, success, size, created_at "
            "FROM tsumego_reports WHERE is_repeat = 0",
            con)
    finally:
        con.close()
    # Deduplicate to the earliest attempt per (user, problem).
    df = (df.sort_values("created_at")
            .drop_duplicates(["user_id", "problem"], keep="first")
            .reset_index(drop=True))
    df["success"] = df["success"].astype(int)
    return df


def build_profiles(df):
    """Explode first-time attempts into 2-month overlapping profile rows.

    Each attempt at month M contributes to profile M and profile M-1, so
    profile K holds months {K, K+1} — a 2-month window overlapping its
    neighbour by one month (identical to evaluate_rating_model). Returns the
    same df with one row per (profile, attempt); `profile` replaces the raw
    user as the ability unit."""
    out = []
    for uid, tid, succ, ts in zip(df["user_id"], df["problem"],
                                  df["success"], df["created_at"]):
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        pm, py = dt.month - 1, dt.year
        if pm == 0:
            pm, py = 12, py - 1
        u8 = uid[:8]
        for prof in (f"{u8}_{dt.year}-{dt.month:02d}", f"{u8}_{py}-{pm:02d}"):
            out.append((prof, tid, int(succ)))
    return pd.DataFrame(out, columns=["profile", "problem", "success"])


def _joint_pca_cache_path(sizes, tids_by_size, pca_dim, seed):
    """Deterministic cache file for the joint-PCA reduction. Keyed on (sizes, pca_dim, seed),
    the raw activation-cache file fingerprints (basename+size+mtime — so a re-extraction
    invalidates it), AND the exact tid subset requested (a caller fitting on a different subset
    gets a different basis, hence a different file). Lives beside the raw activation cache."""
    import hashlib
    h = hashlib.sha256()
    h.update(repr((list(sizes), int(pca_dim), int(seed))).encode())
    for size in sizes:
        for p in (_cache_ids_path(size), _cache_vecs_path(size)):
            st = os.stat(p)
            h.update(f"{os.path.basename(p)}:{st.st_size}:{st.st_mtime_ns}".encode())
        h.update("\x00".join(tids_by_size[size]).encode())
    cdir = os.path.dirname(_cache_vecs_path(sizes[0]))
    return os.path.join(cdir, f"joint_pca_{h.hexdigest()[:16]}.npz")


def load_joint_activations(sizes, tids_by_size, pca_dim, seed, cache=True):
    """ONE PCA basis shared across boards (so a pooled model's activation weights mean
    the same thing on 9×9 and 13×13). Loads each board's raw correct-move activation
    cache for its attempted problems, stacks, fits a single StandardScaler+randomized
    PCA, z-scores. Returns {(size, tid): vec}, k. (Distinct from embeddings/joint, which
    residualises rating OUT — useless as a difficulty feature.)

    The randomized-PCA fit over the stacked ~N×6912 activation matrix is the dominant
    startup cost and is fully determined by its inputs, so the reduced output is memoized to
    joint_pca_<hash>.npz (atomic write, fingerprinted on the raw-cache files). cache=False
    forces a recompute (e.g. to rebuild a corrupt cache)."""
    import json as _json
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    cpath = _joint_pca_cache_path(sizes, tids_by_size, pca_dim, seed) if cache else None
    if cpath and os.path.exists(cpath):
        d = np.load(cpath, allow_pickle=False)
        Z, k = d["Z"], int(d["k"])
        keys = list(zip(d["key_size"].tolist(), d["key_tid"].tolist()))
        print(f"  joint-PCA: cache hit {os.path.basename(cpath)} ({Z.shape[0]} problems, k={k})")
        return {key: Z[i] for i, key in enumerate(keys)}, k
    Xparts, keys = [], []
    for size in sizes:
        ids = _json.load(open(_cache_ids_path(size)))
        row = {t: i for i, t in enumerate(ids)}
        vecs = np.load(_cache_vecs_path(size), mmap_mode="r")
        usable = [t for t in tids_by_size[size] if t in row]
        # Force integer dtype: an empty `usable` yields np.array([]) which is
        # float64, and a float array can't index the memmap (IndexError). With
        # intp an empty index just selects zero rows, so a size with no usable
        # ids contributes an empty block instead of crashing the whole fit.
        rows = np.array([row[t] for t in usable], dtype=np.intp)
        X = np.asarray(vecs[rows], dtype=np.float32)
        finite = np.isfinite(X).all(axis=1)
        usable = [t for t, f in zip(usable, finite) if f]
        Xparts.append(X[finite])
        keys += [(size, t) for t in usable]
    X = np.vstack(Xparts)
    X = StandardScaler().fit_transform(X)
    k = min(pca_dim, X.shape[1], X.shape[0])
    Z = PCA(n_components=k, random_state=seed, svd_solver="randomized").fit_transform(X)
    Z = StandardScaler().fit_transform(Z).astype(np.float32)
    if cpath:
        tmp = f"{cpath}.tmp{os.getpid()}.npz"
        np.savez(tmp, Z=Z, k=k,
                 key_size=np.array([s for s, _ in keys], dtype=np.int16),
                 key_tid=np.array([t for _, t in keys]))
        os.replace(tmp, cpath)
        print(f"  joint-PCA: computed + cached {os.path.basename(cpath)} "
              f"({Z.shape[0]} problems, k={k})")
    return {key: Z[i] for i, key in enumerate(keys)}, k


def load_joint_bot_winrate(sizes, tids_by_size, clip=1e-4):
    """Bot winrate in a shared column layout across boards (the bots are the same;
    a bot's logit pass-rate means the same thing on either board). Common file set =
    intersection across boards; clip→logit→pooled mean-impute→drop pooled-zero-var→
    pooled z-score. Returns {(size, tid): vec}, n_cols."""
    import glob
    filesets = []
    for size in sizes:
        adir = os.path.join(PROJECT_ROOT, "attempts", str(size))
        filesets.append({os.path.basename(f)
                         for f in glob.glob(os.path.join(adir, "*.txt"))})
    common = sorted(set.intersection(*filesets))
    lo, hi = clip, 1.0 - clip
    keys, mat = [], []
    for size in sizes:
        adir = os.path.join(PROJECT_ROOT, "attempts", str(size))
        tidset = set(tids_by_size[size])
        cols = []
        for fn in common:
            d = {}
            with open(os.path.join(adir, fn)) as fh:
                for line in fh:
                    p = line.split()
                    if len(p) != 2:
                        continue
                    t, v = p
                    if t not in tidset:
                        continue
                    try:
                        pv = min(max(float(v), lo), hi)
                    except ValueError:
                        continue
                    d[t] = float(np.log(pv / (1.0 - pv)))
            cols.append(d)
        usable = [t for t in tids_by_size[size] if any(t in c for c in cols)]
        for t in usable:
            keys.append((size, t))
            mat.append([cols[j].get(t, np.nan) for j in range(len(common))])
    X = np.array(mat, dtype=np.float64)
    colmean = np.nanmean(X, axis=0)
    nan = np.where(np.isnan(X))
    X[nan] = np.take(colmean, nan[1])
    keep = np.nanstd(X, axis=0) > 1e-9
    X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, int(keep.sum())


def load_joint_move_features(sizes, tids_by_size):
    """Candidate-move evaluation features (the winning move + its 9 best alternatives), pooled
    and z-scored across boards (a feature means the same on 9×9 and 13×13). Reads the per-size
    RAW caches written by pipeline.cache_move_features and applies the pooled column z-score
    here — so the cache is reusable across tid subsets, like load_joint_bot_winrate. Returns
    {(size, tid): vec}, n_cols. Raises with a build hint if a cache is missing."""
    keys, rows, ncol = [], [], None
    for size in sizes:
        path = os.path.join(CACHE_DIR, f"move_evals_{size}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"move-feature cache {path} missing — run "
                f"`./.venv/bin/python3 -m pipeline.cache_move_features` first")
        d = np.load(path, allow_pickle=False)
        feats = d["feats"]                                # decompress ONCE (NpzFile re-reads per access)
        ncol = int(feats.shape[1])
        row = {t: i for i, t in enumerate(d["tids"].tolist())}
        for t in tids_by_size[size]:
            i = row.get(t)
            if i is not None:
                keys.append((size, t))
                rows.append(feats[i].copy())              # detach from `feats`'s base so it can free
    X = np.array(rows, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # pooled column z-score
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, ncol
