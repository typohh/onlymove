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


# --------------------------------------------------------------------------------
# Scalar problem axes: tightness gaps (from the move-eval cache) and game phase
# (from the SGFs, cached). These are the forward-validated per-user axes of
# docs/scalar_weakness_axes.md AND candidate rating-model features (the residual-
# MIRT prototype's r head found predictable residual difficulty on exactly this
# surface — see auxiliary.rating_dev.train_residual_axes). Living here keeps the
# production trainer (pipeline.train_rating), the experiment surface
# (auxiliary.rating_dev.*) and the axis analyses (auxiliary.eval_score_gap_axis)
# on ONE set of definitions: same caches, same transforms, in sync by construction.


def _game_phase_cache_path(size):
    return os.path.join(CACHE_DIR, f"game_phase_{size}.json")


def _game_phase_one(path):
    """(n_setup, prefix_moves, main_len) for one tsumego SGF.

    prefix = the linear move chain from root to the branch point (the game that
    led to the contested position); main_len = non-pass moves on the principal
    line (first-child chain) of the CORRECT first move — the child with the best
    winrate comment, score breaking ties (the same choice as the PWA's
    findBestMoveFromChildren / cache_move_features slot 0). The principal line
    terminates in passes by construction."""
    import re as _re
    import sgfmill.sgf
    cmt_re = _re.compile(rb"(\d+\.?\d*)%/([-]?\d+\.?\d*)")
    with open(path, "rb") as f:
        game = sgfmill.sgf.Sgf_game.from_bytes(f.read())
    root = game.get_root()
    n_setup = sum(len(root.get(p)) for p in ("AB", "AW") if root.has_property(p))
    node = root
    prefix = 0
    while len(node) == 1:
        if node.has_property("B") or node.has_property("W"):
            prefix += 1
        node = node[0]
    if node is not root and (node.has_property("B") or node.has_property("W")):
        prefix += 1                     # the branch node itself carries a move
    best, best_key = None, None
    for child in node:
        if not (child.has_property("B") or child.has_property("W")):
            continue
        m = cmt_re.search((child.get_raw("C") if child.has_property("C") else b"") or b"")
        if not m:
            continue
        key = (float(m.group(1)), float(m.group(2)))
        if best_key is None or key > best_key:
            best, best_key = child, key
    main_len = 0
    cur = best
    while cur is not None:
        if cur.has_property("B") or cur.has_property("W"):
            _color, mv = cur.get_raw_move()
            if mv is not None and mv != b"" and mv != b"tt":
                main_len += 1
        cur = cur[0] if len(cur) else None
    return n_setup, prefix, main_len


def _game_phase_chunk(paths):
    out = {}
    for p in paths:
        tid = os.path.basename(p)[:-4]
        try:
            out[tid] = _game_phase_one(p)
        except Exception:
            out[tid] = None
    return out


def load_game_phase(rebuild=False, sizes=(9, 13)):
    """tid -> (n_setup, prefix_moves, main_len, size); cached per size in the
    signal cache (multiprocess SGF parse on first build)."""
    import json as _json
    from concurrent.futures import ProcessPoolExecutor, as_completed
    res = {}
    for size in sizes:
        cache = _game_phase_cache_path(size)
        if os.path.exists(cache) and not rebuild:
            res.update({t: (*v, size) for t, v in _json.load(open(cache)).items()})
            continue
        sgf_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        paths = [os.path.join(sgf_dir, n) for n in os.listdir(sgf_dir) if n.endswith(".sgf")]
        print(f"parsing {len(paths)} SGFs for size {size} (cached to {cache}) ...")
        got = {}
        with ProcessPoolExecutor() as ex:
            futs = [ex.submit(_game_phase_chunk, paths[i:i + 500]) for i in range(0, len(paths), 500)]
            for fu in as_completed(futs):
                got.update(fu.result())
        got = {t: v for t, v in got.items() if v is not None}
        _json.dump(got, open(cache, "w"))
        res.update({t: (*v, size) for t, v in got.items()})
    return res


def load_tightness_gaps(kind, sizes=(9, 13)):
    """tid -> RAW best-vs-second-best gap from the move-eval cache.
    kind='score_gap': solution score − best-alt score (points, side to move);
    kind='wr_gap': solution logit-winprob − best-alt logit-winprob."""
    gap = {}
    for size in sizes:
        path = os.path.join(CACHE_DIR, f"move_evals_{size}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"move-feature cache {path} missing — run "
                f"`./.venv/bin/python3 -m pipeline.cache_move_features` first")
        d = np.load(path, allow_pickle=False)
        feats = d["feats"]
        g = feats[:, 1] - feats[:, 3] if kind == "score_gap" else feats[:, 0] - feats[:, 2]
        for t, v in zip(d["tids"].tolist(), g):
            gap[str(t)] = float(v)
    return gap


def zscore_neglog(gap):
    """The tightness transform: z of −log1p(max(gap, 0)) — positive = TIGHT margin.
    (Negative score gaps are real but off-hypothesis; folded to 0.)"""
    v = np.array([-np.log1p(max(x, 0.0)) for x in gap.values()])
    mu, sd = v.mean(), v.std()
    return {t: (-np.log1p(max(x, 0.0)) - mu) / sd for t, x in gap.items()}


def phase_ratio_z(phase=None):
    """tid -> z (per board size) of main/(prefix+main) — the share of the game
    still ahead as continuation; the game-phase form that carries the per-user
    signal (the raw total self-cancels; see docs/scalar_weakness_axes.md)."""
    if phase is None:
        phase = load_game_phase()
    ratio = {t: v[2] / max(v[0] + v[1] + v[2], 1) for t, v in phase.items()}
    fz = {}
    for size in (9, 13):
        ids = [t for t, v in phase.items() if v[3] == size]
        vals = np.array([ratio[t] for t in ids])
        fz.update({t: (ratio[t] - vals.mean()) / vals.std() for t in ids})
    return fz


def load_joint_game_phase(sizes, tids_by_size):
    """Game-phase RAW COMPONENTS as a trainer-convention feature block:
    [moves_before_position (setup stones + prefix moves), main_line_len] per
    problem, pooled column z-score like the other load_joint_* blocks. Returns
    {(size, tid): vec(2)}, n_cols. Deliberately the raw components, NOT the
    derived main/(prefix+main) ratio — a network can construct the ratio (or a
    better combination) itself, and the raw sum/level also carries the board-
    fullness information the ratio drops. (The score/winrate tightness gaps are
    NOT a block here for the same reason: their components are already slots of
    load_joint_move_features.) The derived scalar forms used by the linear axis
    analyses live in zscore_neglog / phase_ratio_z below."""
    phase = load_game_phase(sizes=sizes)
    keys, rows = [], []
    for size in sizes:
        for t in tids_by_size[size]:
            v = phase.get(t)
            if v is not None:
                keys.append((size, t))
                rows.append([v[0] + v[1], v[2]])
    X = np.array(rows, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # pooled column z-score
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, 2


# --------------------------------------------------------------------------------
# Backend rating outputs (ratings/<size>/*.csv — the stage BEFORE package_for_pwa).
# Model-development consumers (auxiliary.rating_dev.*, the axis batteries) read
# THESE, not pwa/tsumegos/problem_params.bin: same values (packaging just copies),
# but available the moment training finishes and immune to a stale/half-packaged
# dist. Client-replica harnesses (leaderboard replay, eval_weakness_forward) keep
# reading the shipped files on purpose — bit-fidelity to what clients see IS
# their job.

RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")


def load_backend_problem_params(sizes=(9, 13)):
    """tid -> (rating, slope, floor) from ratings/<size>/tsumego_{ratings,quality,
    floor}_final.csv. Fails loud on a missing file (run pipeline.train_rating)."""
    out = {}
    for size in sizes:
        d = os.path.join(RATINGS_DIR, str(size))
        frames = {}
        for name, col in (("tsumego_ratings_final.csv", "rating"),
                          ("tsumego_quality_final.csv", "quality"),
                          ("tsumego_floor_final.csv", None)):
            path = os.path.join(d, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing — run pipeline.train_rating first")
            frames[name] = pd.read_csv(path)
        r = frames["tsumego_ratings_final.csv"].set_index("tsumego_id")["rating"]
        q = frames["tsumego_quality_final.csv"].set_index("tsumego_id")["quality"]
        fdf = frames["tsumego_floor_final.csv"].set_index("tsumego_id")
        f = fdf[fdf.columns[0]]
        for tid in r.index:
            if tid in q.index and tid in f.index:
                out[str(tid)] = (float(r[tid]), float(q[tid]), float(f[tid]))
    return out


def load_backend_rank_curve():
    """The backend rating->rank curve (ratings/rating_to_rank.json, one shared
    curve written to both size keys by pipeline.map_rating_to_rank)."""
    import json as _json
    path = os.path.join(RATINGS_DIR, "rating_to_rank.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing — run pipeline.map_rating_to_rank first")
    m = _json.load(open(path))
    return m["9"] if "9" in m else next(iter(m.values()))
