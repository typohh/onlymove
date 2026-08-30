"""Shared rating-model input assembly — the data + features the rating trainer consumes.

This is the production home for the helpers that load the rating model's training data
(first-time human attempts), turn it into the ability unit (2-month overlapping profiles),
and assemble the per-problem feature trunk. Every per-problem block lives here, whether or
not the shipped layout (model_constants.PRODUCTION_BLOCKS) feeds it to the trunk: the ones
outside it — candidate-move evaluations, ownership swing, game phase — are what the motif
rules and the axis analyses are written over. It also owns the activation "signal cache"
path helpers those loaders read.

These functions used to live in auxiliary/ (the research area) and were imported back into the
pipeline; they are leaf utilities (no auxiliary/ or pipeline/ imports), so both the production
trainer (pipeline.train_mirt) and the experiment scripts (auxiliary.rating_dev.*) import
them from here. The auxiliary modules that historically defined them now re-export from here so
their other callers are unaffected.
"""

import csv
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from utils.rating.model_constants import ACT_ENCODER_DIM, PRODUCTION_BLOCKS
from utils.rating.ratings_io import FLOOR_CSV, RATINGS_CSV, SLOPE_CSV

from utils.constants import RETIRED_IDS_PATH, read_retired_ids
from utils.paths import DB_SQL, DB_SQLITE, PROJECT_ROOT

SQL_PATH = DB_SQL
SQLITE_PATH = DB_SQLITE

# Raw KataGo trunk activation layout (18 residual blocks × 384 channels), captured at
# the correct move by pipeline.extract_activations. Feature index = layer*384 + channel.
# The activation store holds the frozen encoder's output, not raw trunk
# channels (18 blocks x 384 = 6912 exist only transiently inside extraction —
# see utils.rating.activation_encoder). Every buffer and shape check below
# follows this width, so a stale raw-width cache fails validation and rebuilds.
N_FEATS = ACT_ENCODER_DIM

# Number of contrastive blocks: per-channel [mean, std, max, zscore] aggregates over the
# OTHER legal moves (N_CONTRAST_BLOCKS × N_FEATS), id-aligned with the correct-vec cache.
N_CONTRAST_BLOCKS = 4

# The activation "signal cache": correct-move trunk vectors written per board by
# build_correct_vec_cache below and read directly by load_joint_activations. The optional
# contrastive cache (the other-move aggregates) is a legacy sibling in the same directory,
# read if present and no longer written by anything in the repo.
CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")


def _cache_vecs_path(size):
    return os.path.join(CACHE_DIR, f"correct_vecs_{size}.npy")


def _cache_ids_path(size):
    return os.path.join(CACHE_DIR, f"correct_ids_{size}.json")


def _contrastive_cache_path(size):
    return os.path.join(CACHE_DIR, f"contrastive_vecs_{size}.npy")


def load_correct_vec(act_dir, tid, by_ext):
    """The correct-move N_FEATS vector for one problem, read from its activation
    .npz (all legal cells, addressed by correct_idx — NEVER row 0; the npz is in
    board-scan order).

    This is the per-problem primitive build_correct_vec_cache calls once per tid,
    which is what makes that build incremental: a row depends on nothing but its
    own .npz.

    A corrupt/unreadable .npz is a HARD ERROR here, so on-disk corruption of the
    activation corpus surfaces instead of being silently masked.
    """
    npz_path = os.path.join(act_dir, by_ext["npz"][tid])
    try:
        with np.load(npz_path) as s:
            return s["candidates"][int(s["correct_idx"])]
    except Exception as e:
        raise RuntimeError(f"corrupt/unreadable .npz at {npz_path}") from e


def _load_correct_vec_chunk(task):
    """Worker: the correct-move vectors for a chunk of problems.

    Returns (start_index, (chunk, N_FEATS) float16). An unreadable .npz raises out
    of load_correct_vec and aborts the whole build — deliberately. The alternative
    (a NaN row) is invisible: the tid stays in correct_ids while _fit_joint_pca's
    finite mask drops the row, so the problem silently leaves the rated set.
    """
    start, act_dir, items = task
    out = np.empty((len(items), N_FEATS), dtype=np.float16)
    for i, (tid, npz_name) in enumerate(items):
        by_ext = {"npz": {tid: npz_name} if npz_name else {}}
        out[i] = load_correct_vec(act_dir, tid, by_ext).astype(np.float16)
    return start, out


def _assert_activations_subset_of_corpus(size, ids):
    """Every activation .npz must belong to a problem still in tsumegos/<size>/.

    ONE-DIRECTIONAL on purpose. Corpus ⊅ activations is an error; corpus ⊋ activations is
    normal — extraction is resumable, so a fresh tsumego top-up legitimately has SGFs with
    no .npz yet, and asserting that direction would break the ordinary workflow.

    The subset direction has no legitimate cause: pipeline.extract_activations writes one
    .npz per SGF it reads, so an orphan can only mean the SGF was removed without its
    artifacts. That is worth a hard failure because correct_ids is derived from THIS
    listing, so an orphan silently promotes a problem that no longer exists into the rated
    set — it gets features, a rating, a slope and a floor, and nothing downstream can tell
    it apart from a real one. It ships nowhere (packaging works off tsumegos/), so the only
    place the contradiction is visible is right here. Found in production 2026-08-02 as a
    single 13x13 .npz left behind by a pre-canonical-id removal; it had been feeding a
    phantom problem through the whole rating chain since May.
    """
    corpus_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    if not os.path.isdir(corpus_dir):
        raise SystemExit(
            f"size {size}: activations exist but the corpus dir is missing ({corpus_dir}); "
            f"refusing to build a cache whose ids cannot be checked against real problems")
    corpus = {f[:-4] for f in os.listdir(corpus_dir) if f.endswith(".sgf")}
    orphans = sorted(t for t in ids if t not in corpus)
    if orphans:
        raise SystemExit(
            f"size {size}: {len(orphans)} activation .npz with no SGF in {corpus_dir} "
            f"(e.g. {orphans[:5]}). Each would enter correct_ids as a problem that does "
            f"not exist. Delete them (keeping a backup — with the SGF gone the .npz is the "
            f"last trace of that position), or restore the SGFs, then re-run.")


def build_correct_vec_cache(size, workers=None, force=False):
    """Build/refresh one board's correct-cell activation cache — correct_vecs_<size>.npy
    plus correct_ids_<size>.json — the aggregate that load_joint_activations reads.

    pipeline.extract_activations writes one .npz per problem; this rolls those up into
    the single memmap the feature loaders address by row, and correct_ids becomes the
    definitive "problems the rating model can score" set (pipeline.cache_rating_features
    takes its tid set straight from it). So a stale cache after a tsumego top-up drops
    the new problems from the rated set silently, which is why every consumer refreshes
    it rather than assuming it is current.

    INCREMENTAL: each row is a pure per-problem read of that problem's .npz, so reusing
    the rows already cached and loading only the tids missing from them is identical to a
    full rebuild while skipping the per-problem disk reads that dominate. force=True
    re-reads every .npz.

    The .npz set must be a SUBSET of the corpus — see _assert_activations_subset_of_corpus,
    checked before any work.
    """
    workers = workers or os.cpu_count() or 1
    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    if not os.path.isdir(act_dir):
        raise SystemExit(f"no activations dir for size {size}: {act_dir}")
    files = os.listdir(act_dir)
    npz = {f[:-4]: f for f in files if f.endswith(".npz")}
    ids = sorted(npz)
    if not ids:
        raise SystemExit(f"no .npz activation files in {act_dir}")
    _assert_activations_subset_of_corpus(size, ids)
    n = len(ids)

    # Reuse the existing cache where it covers the same tid (in any order).
    vecs_path, ids_path = _cache_vecs_path(size), _cache_ids_path(size)
    old_row, old_vecs = {}, None
    if not force and os.path.exists(vecs_path) and os.path.exists(ids_path):
        old_ids = json.load(open(ids_path))
        cand = np.load(vecs_path, mmap_mode="r")
        if cand.shape == (len(old_ids), N_FEATS):
            old_vecs, old_row = cand, {t: i for i, t in enumerate(old_ids)}
        if old_ids == ids and old_vecs is not None:
            print(f"[size {size}] correct-vector cache up to date ({n} problems)", flush=True)
            return

    missing = [t for t in ids if t not in old_row]
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[size {size}] correct-vector cache: {n} problems "
          f"({len(missing)} to load, {n - len(missing)} reused) "
          f"-> {vecs_path} ({n * N_FEATS * 2 / 1e9:.2f} GB)", flush=True)

    # Load only the missing tids' correct vectors (parallel); keep by tid for assembly.
    missing_rows = {}
    chunk = 1024
    tasks = [(s, act_dir,
              [(t, npz[t]) for t in missing[s:s + chunk]])
             for s in range(0, len(missing), chunk)]
    t0, done = time.time(), 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for (_, arr), (_, _, items) in zip(ex.map(_load_correct_vec_chunk, tasks), tasks):
                for j, (tid, _) in enumerate(items):
                    missing_rows[tid] = arr[j]
                done += len(items)
                rate = done / (time.time() - t0) if time.time() > t0 else 0.0
                print(f"\r  {done}/{len(missing)} ({rate:.0f}/s)", end="", flush=True)
        print(flush=True)

    # Assemble the full cache in `ids` order: reused rows from the old memmap, the rest
    # from this run's loads. Written to a tmp memmap, then atomically swapped in.
    #
    # Every row is checked finite as it lands. Freshly loaded rows cannot be otherwise —
    # an unreadable .npz already aborted the worker — but a REUSED row was written by an
    # older build that tolerated failures, so this is what retires those. A non-finite row
    # is a hard abort, not a drop: its tid would stay in correct_ids while _fit_joint_pca's
    # finite mask removed it from the fit, i.e. the problem leaves the rated set with
    # nothing anywhere saying so.
    tmp_path = vecs_path + ".tmp"
    try:
        mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(n, N_FEATS))
        try:
            bad = []
            for i, t in enumerate(ids):
                row = old_vecs[old_row[t]] if t in old_row else missing_rows[t]
                if not np.isfinite(row).all():
                    bad.append(t)
                mm[i] = row
            if bad:
                raise RuntimeError(
                    f"size {size}: {len(bad)} non-finite correct-vector rows (e.g. {bad[:5]}); "
                    f"refusing to write a corrupt cache. Re-extract those problems' "
                    f"activations, then rebuild with force=True.")
            mm.flush()
        finally:
            del mm
            del old_vecs  # release the mmaps before replacing the file underneath them
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # never leave a rejected build to be mistaken for a cache
        raise
    os.replace(tmp_path, vecs_path)
    with open(ids_path, "w") as f:
        json.dump(ids, f)


def _cache_alt_vecs_path(size):
    return os.path.join(CACHE_DIR, f"alt_vecs_{size}.npy")


def _load_alt_vec_chunk(task):
    """Worker: the BEST-ALTERNATIVE cell's vector for a chunk of problems.

    The alternative is addressed by (x, y) against the npz's own `xys`, never by row
    position — the npz is board-scan order over every legal cell, so a positional guess
    reads a different move's activations while looking entirely healthy.

    A problem with no non-pass alternative gets a NaN row, which the loader turns into a
    dropped key so the trainer mean-imputes it like any other missing block row. That is
    the one absence here that is genuinely uninformative: filling it with the correct
    cell's own vector would make the contrast exactly zero, i.e. "the runner-up is
    indistinguishable from the answer" — the most confusable reading possible, and the
    opposite of what having no alternative means.
    """
    start, act_dir, items = task
    out = np.empty((len(items), N_FEATS), dtype=np.float16)
    for i, (tid, npz_name, xy) in enumerate(items):
        if xy[0] < 0:
            out[i] = np.nan
            continue
        npz_path = os.path.join(act_dir, npz_name)
        try:
            with np.load(npz_path) as s:
                xys = s["xys"]
                row = np.flatnonzero((xys[:, 0] == xy[0]) & (xys[:, 1] == xy[1]))
                if not len(row):
                    raise RuntimeError(
                        f"best-alternative cell {tuple(xy)} is not an embedded cell")
                out[i] = s["candidates"][int(row[0])].astype(np.float16)
        except Exception as e:
            raise RuntimeError(f"corrupt/unreadable .npz at {npz_path} ({tid})") from e
    return start, out


def build_alt_vec_cache(size, workers=None, force=False):
    """Build/refresh one board's BEST-ALTERNATIVE cell activation cache — alt_vecs_<size>.npy,
    row-aligned with correct_ids_<size>.json exactly like correct_vecs_<size>.npy.

    The alternative cell's coordinate comes from the move-eval cache's `alt_xy` array
    (pipeline.cache_rating_features), which ranks the root moves the way the mover would and
    takes the runner-up. Sharing that one ranking is what keeps this block addressing the same
    move the 20-column move block calls slot 1.

    INCREMENTAL on the same terms as build_correct_vec_cache: each row is a pure per-problem
    read, so rows already cached are reused and only missing tids are loaded. force=True
    re-reads every .npz (hours — the whole point of the incremental path).
    """
    workers = workers or os.cpu_count() or 1
    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    if not os.path.isdir(act_dir):
        raise SystemExit(f"no activations dir for size {size}: {act_dir}")
    ids_path = _cache_ids_path(size)
    if not os.path.exists(ids_path):
        raise SystemExit(f"no activation id cache ({ids_path}) — run cache_rating_features")
    ids = json.load(open(ids_path))

    mv_path = os.path.join(CACHE_DIR, f"move_evals_{size}.npz")
    if not os.path.exists(mv_path):
        raise SystemExit(f"move-eval cache {mv_path} missing — run cache_rating_features")
    with np.load(mv_path) as d:
        if "alt_xy" not in d.files:
            raise SystemExit(
                f"{mv_path} has no alt_xy — re-run `python -m pipeline.cache_rating_features`")
        alt_xy = {t: tuple(int(v) for v in xy)
                  for t, xy in zip(d["tids"].tolist(), d["alt_xy"])}
    missing_xy = [t for t in ids if t not in alt_xy]
    if missing_xy:
        raise SystemExit(
            f"size {size}: {len(missing_xy)} tids in correct_ids have no alt_xy row "
            f"(e.g. {missing_xy[:5]}); the two caches are out of step — re-run "
            f"pipeline.cache_rating_features")

    npz = {f[:-4]: f for f in os.listdir(act_dir) if f.endswith(".npz")}
    _assert_activations_subset_of_corpus(size, sorted(npz))
    n = len(ids)

    vecs_path = _cache_alt_vecs_path(size)
    old_row, old_vecs = {}, None
    if not force and os.path.exists(vecs_path):
        cand = np.load(vecs_path, mmap_mode="r")
        # The file carries no id list of its own: it is defined as row-aligned to correct_ids,
        # so a length change means correct_ids moved and every row's identity is in doubt.
        # Rebuilding is the only safe reading of that.
        if cand.shape == (n, N_FEATS):
            old_vecs, old_row = cand, {t: i for i, t in enumerate(ids)}
            print(f"[size {size}] alt-vector cache up to date ({n} problems)", flush=True)
            return
        print(f"[size {size}] alt-vector cache is {cand.shape[0]} rows against {n} "
              f"problems — rebuilding", flush=True)

    todo = [(t, npz.get(t), alt_xy[t]) for t in ids]
    absent = [t for t, f, _ in todo if f is None]
    if absent:
        raise SystemExit(
            f"size {size}: {len(absent)} problems in correct_ids have no .npz "
            f"(e.g. {absent[:5]}) — run pipeline.extract_activations")
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[size {size}] alt-vector cache: {n} problems -> {vecs_path} "
          f"({n * N_FEATS * 2 / 1e9:.2f} GB)", flush=True)

    tmp_path = vecs_path + ".tmp"
    chunk = 1024
    tasks = [(s, act_dir, todo[s:s + chunk]) for s in range(0, n, chunk)]
    t0, done, n_nan = time.time(), 0, 0
    try:
        mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16,
                                       shape=(n, N_FEATS))
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for start, arr in ex.map(_load_alt_vec_chunk, tasks):
                    mm[start:start + len(arr)] = arr
                    n_nan += int((~np.isfinite(arr)).all(axis=1).sum())
                    done += len(arr)
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"\r  {done}/{n} ({rate:.0f}/s)", end="", flush=True)
            print(flush=True)
            mm.flush()
        finally:
            del mm
            del old_vecs
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # never leave a rejected build to be mistaken for a cache
        raise
    os.replace(tmp_path, vecs_path)
    print(f"[size {size}] wrote {n} alt rows in {time.time() - t0:.0f}s "
          f"({n_nan} with no alternative -> imputed at load)", flush=True)


def load_joint_alt_activations(sizes, tids_by_size, pca_dim, seed):
    """The BEST-ALTERNATIVE cell's activations, projected through the SAME joint-PCA basis the
    correct-move block uses. Returns {(size, tid): vec}, k.

    Same basis, not a refit: the point of the block is that the runner-up lands in the correct
    move's coordinate system, so the trunk can read the two together. A separate basis would put
    the two blocks on different axes and make any comparison between them meaningless.

    The post-PCA standardization (m2, s2) is likewise the correct-cell one and is deliberately
    NOT re-fit here, so the alternative's OFFSET from where correct cells sit is preserved
    rather than centred away — that offset is the signal. Re-standardizing costs nothing in
    expressiveness (a linear first layer spans the same functions either way) but it would
    quietly rescale the one comparison the block exists to support.

    Problems with no non-pass alternative carry a NaN row in the cache and are simply absent
    from the returned map, which routes them onto the trainer's ordinary mean-imputation path.
    """
    basis = load_joint_activation_basis(sizes, tids_by_size, pca_dim, seed)
    m1, s1, C = basis["m1"], basis["s1"], basis["C"]
    pca_mean, m2, s2 = basis["pca_mean"], basis["m2"], basis["s2"]
    out, k = {}, C.shape[0]
    for size in sizes:
        path = _cache_alt_vecs_path(size)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"alt-activation cache {path} missing — build it with "
                f"utils.rating.features.build_alt_vec_cache({size})")
        ids = json.load(open(_cache_ids_path(size)))
        vecs = np.load(path, mmap_mode="r")
        if vecs.shape != (len(ids), N_FEATS):
            raise RuntimeError(
                f"{path} is {vecs.shape} against {len(ids)} problems in correct_ids; the two "
                f"are row-aligned by definition — rebuild the alt cache")
        row = {t: i for i, t in enumerate(ids)}
        want = [t for t in tids_by_size[size] if t in row]
        rows = np.array([row[t] for t in want], dtype=np.intp)
        for s0 in range(0, len(rows), 4096):                # chunked: the raw block is GB-scale
            sl = rows[s0:s0 + 4096]
            X = np.asarray(vecs[sl], dtype=np.float32)
            finite = np.isfinite(X).all(axis=1)
            Z = (((X - m1) / s1 - pca_mean) @ C.T - m2) / s2
            for j, t in enumerate(want[s0:s0 + 4096]):
                if finite[j]:
                    out[(size, t)] = Z[j].astype(np.float32)
    return out, int(k)


def _no_alternative_tids(sizes):
    """{size: {tid}} for the problems whose move-eval cache records NO non-pass
    alternative (`alt_xy` = (-1, -1)).

    This is what separates a real state from missing data in the alt cache. A NaN
    alt row is written by _load_alt_vec_chunk in exactly one situation — the
    coordinate says there is no alternative — because every other failure raises
    out of the worker. That makes the NaN safe to READ as "no alternative", and
    reading it that way without checking would be assuming an invariant this
    module elsewhere checks (see _assert_ids_unique). So the two are cross-checked
    at load: a NaN row for a problem that HAS a runner-up coordinate is cache
    corruption, not a legitimate absence, and it must not be quietly encoded as one.
    """
    out = {}
    for size in sizes:
        path = os.path.join(CACHE_DIR, f"move_evals_{size}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"move-feature cache {path} missing — run "
                f"`./.venv/bin/python3 -m pipeline.cache_rating_features` first")
        d = np.load(path, allow_pickle=False)
        if "alt_xy" not in d.files:
            raise FileNotFoundError(
                f"{path} has no alt_xy — re-run "
                f"`./.venv/bin/python3 -m pipeline.cache_rating_features`")
        out[size] = {str(t) for t, xy in zip(d["tids"].tolist(), d["alt_xy"])
                     if int(xy[0]) < 0}
    return out


def load_joint_act_delta(sizes, tids_by_size, pca_dim, seed):
    """CANDIDATE BLOCK, DEV ARMS ONLY (see the last paragraph). The correct-vs-
    best-alternative activation CONTRAST: the correct cell's joint-PCA vector minus
    the runner-up cell's, plus a has_alt indicator column. Returns
    {(size, tid): vec(k + 1)}, k + 1.

    ONE basis for both cells — the correct-move block's, through
    load_joint_activations and load_joint_alt_activations, which already share it.
    The subtraction commutes with the whole linear chain except its additive
    centres, which cancel outright:

        z_c - z_a = ( ((x_c - m1)/s1 - pca_mean) @ C.T - m2 )/s2
                  - ( ((x_a - m1)/s1 - pca_mean) @ C.T - m2 )/s2
                  = ( ((x_c - x_a)/s1) @ C.T ) / s2

    so m1, pca_mean and m2 leave no trace and only the two scalings survive. The
    block is therefore the raw difference of the two stored vectors placed on the
    shared axes, and it is NOT re-standardized afterwards, for load_joint_alt_-
    activations' reason: the magnitude of the contrast is the quantity of interest
    (how far the runner-up sits from the answer), and re-centring a difference
    against the population would price a tight problem relative to the average
    tightness rather than stating it.

    WHY A DIFFERENCE RATHER THAN THE ALTERNATIVE'S OWN VECTOR. The concatenated
    form was measured and rejected (methodology.md: worse on both measures), and
    the reason recorded there is WIDTH — a second full activation block gives the
    difficulty head a second chance to fingerprint the individual problem, which is
    also why a LOW correlation between the two blocks is not evidence of
    complementary signal. The difference is the one form that objection does not
    reach in the same way: the correct cell and the runner-up sit on the same board
    and share their position-global components, so subtracting cancels the identity
    those blocks were memorizing and leaves the move-local part — the tightness the
    two cells differ by. That is a claim the battery tests, not one this loader
    asserts; the width cost is still paid and still has to be earned.

    THE has_alt COLUMN. A problem with no non-pass alternative is a real state, not
    missing data, so it cannot be dropped (that removes a live problem from the
    rated set) — but neither can the fact be encoded by the delta itself. A ZERO
    delta says the runner-up is indistinguishable from the answer, i.e. maximal
    confusability, which is the OPPOSITE of what having no runner-up means; a large
    delta would be an invented magnitude. So the block is two parts: the delta,
    zero-filled where there is no alternative, and an indicator that states the
    fact. The zero is then inert behind a column that says why it is zero, rather
    than a value the trunk must read as a measurement. Where the corpus has no such
    problem the indicator is simply constant and the trunk ignores it.

    A tid absent from the correct-cell map is absent here too, so it reaches
    joint_feature_tail's _missing_block_error like any other gap.

    NEVER PRODUCTION_BLOCKS. This block is answer-derived twice over — the correct
    cell is the answer, and the runner-up is defined by ranking against it — so
    under pipeline.write_mirt_per_cell's fan-out, where the trunk is evaluated at
    every legal cell with this tail held fixed, it hands every alternative cell a
    description of the correct move. That is the contamination the tail-conditioning
    ablation measures (a large share of alternative cells change SIGN when the wide
    tail moves), and it is why PRODUCTION_BLOCKS keeps only bot winrate. If the
    contrast ever earns a place it would feed the difficulty heads alone, which is a
    separate architecture change, not a layout change.
    """
    missing = [s for s in sizes if not os.path.exists(_cache_alt_vecs_path(s))]
    if missing:
        sizes_arg = ", ".join(str(s) for s in missing)
        raise FileNotFoundError(
            f"alt-activation cache missing for size(s) {sizes_arg} "
            f"({', '.join(_cache_alt_vecs_path(s) for s in missing)}). The delta "
            f"block cannot be assembled without it. Build it with:\n"
            f"  ./.venv/bin/python3 -c 'from utils.rating.features import "
            f"build_alt_vec_cache; "
            f"[build_alt_vec_cache(s) for s in ({sizes_arg},)]'\n"
            f"That reads every problem's activation .npz once (long, disk-bound, "
            f"resumable only in the sense that a complete file is reused whole), "
            f"and it must run AFTER pipeline.cache_rating_features, whose alt_xy "
            f"array supplies the runner-up coordinate.")
    corr, k = load_joint_activations(sizes, tids_by_size, pca_dim, seed)
    alt, k_alt = load_joint_alt_activations(sizes, tids_by_size, pca_dim, seed)
    if k_alt != k:
        raise RuntimeError(
            f"correct-cell block is k={k} but the alternative block is k={k_alt}; "
            f"they are defined to share one basis, so the difference between them "
            f"is not a contrast — delete the joint-PCA cache and reload")
    no_alt = _no_alternative_tids(sizes)
    zero = np.zeros(k, dtype=np.float32)
    has = np.ones(1, dtype=np.float32)
    hasnt = np.zeros(1, dtype=np.float32)
    out = {}
    for size in sizes:
        for t in tids_by_size[size]:
            key = (size, t)
            c = corr.get(key)
            if c is None:
                continue                      # the activation block's own gap; it raises there
            a = alt.get(key)
            if a is None:
                if t not in no_alt[size]:
                    raise RuntimeError(
                        f"alt-activation cache has a non-finite row for {t} (size "
                        f"{size}), but its move-eval alt_xy names a real runner-up "
                        f"cell — the row is CORRUPT, not a problem without an "
                        f"alternative, and encoding it as one would state a fact "
                        f"that is not true. Rebuild with "
                        f"build_alt_vec_cache({size}, force=True).")
                out[key] = np.concatenate([zero, hasnt])
            else:
                out[key] = np.concatenate([(c - a).astype(np.float32), has])
    return out, int(k) + 1


def load_first_time_attempts():
    """Return a DataFrame of first-time attempts: one row per (user_id, problem),
    columns [user_id, problem, success, size, created_at].

    Reads the canonical binary db/d1.sqlite directly (opened
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
    PCA, z-scores. Returns {(size, tid): vec}, k. (Distinct from axes/, which
    residualises rating OUT — useless as a difficulty feature.)

    The randomized-PCA fit over the stacked ~N×N_FEATS activation matrix is the dominant
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
    return _fit_joint_pca(sizes, tids_by_size, pca_dim, seed, cpath)[:2]


def load_joint_activation_basis(sizes, tids_by_size, pca_dim, seed):
    """The full joint-PCA TRANSFORM (not just the projected values): stored N_FEATS-dim
    activation vector -> standardize (m1, s1) -> PCA components C -> standardize
    (m2, s2). Needed to project arbitrary (non-correct) CELLS into the same
    activation space — pipeline.write_mirt_per_cell. Reads the basis from
    the joint-PCA cache; a pre-basis cache file (older format) is refit in place."""
    cpath = _joint_pca_cache_path(sizes, tids_by_size, pca_dim, seed)
    if os.path.exists(cpath):
        d = np.load(cpath, allow_pickle=False)
        if "basis_m1" in d.files:
            return {"m1": d["basis_m1"], "s1": d["basis_s1"], "C": d["basis_C"],
                    "pca_mean": d["basis_pca_mean"],
                    "m2": d["basis_m2"], "s2": d["basis_s2"]}
        print("  joint-PCA: cache predates basis storage — refitting to capture it")
    return _fit_joint_pca(sizes, tids_by_size, pca_dim, seed, cpath)[2]


def _fit_joint_pca(sizes, tids_by_size, pca_dim, seed, cpath):
    """Fit + cache the joint PCA; returns ({key: vec}, k, basis dict)."""
    import json as _json
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
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
    sc1 = StandardScaler().fit(X)
    X = sc1.transform(X)
    k = min(pca_dim, X.shape[1], X.shape[0])
    pca = PCA(n_components=k, random_state=seed, svd_solver="randomized").fit(X)
    P = pca.transform(X)
    sc2 = StandardScaler().fit(P)
    Z = sc2.transform(P).astype(np.float32)
    # The full transform chain (pca.transform subtracts pca.mean_ first):
    #   z = ( ((x - m1)/s1 - pca_mean) @ C.T - m2 ) / s2
    basis = {"m1": sc1.mean_.astype(np.float32),
             "s1": sc1.scale_.astype(np.float32),
             "C": pca.components_.astype(np.float32),
             "pca_mean": pca.mean_.astype(np.float32),
             "m2": sc2.mean_.astype(np.float32),
             "s2": sc2.scale_.astype(np.float32)}
    if cpath:
        tmp = f"{cpath}.tmp{os.getpid()}.npz"
        np.savez(tmp, Z=Z, k=k,
                 key_size=np.array([s for s, _ in keys], dtype=np.int16),
                 key_tid=np.array([t for _, t in keys]),
                 basis_m1=basis["m1"], basis_s1=basis["s1"], basis_C=basis["C"],
                 basis_m2=basis["m2"], basis_s2=basis["s2"],
                 basis_pca_mean=basis["pca_mean"])
        os.replace(tmp, cpath)
        print(f"  joint-PCA: computed + cached {os.path.basename(cpath)} "
              f"({Z.shape[0]} problems, k={k})")
    return {key: Z[i] for i, key in enumerate(keys)}, k, basis


def load_joint_bot_winrate(sizes, tids_by_size, clip=1e-4):
    """Bot winrate in a shared column layout across boards (the bots are the same;
    a bot's logit pass-rate means the same thing on either board). Common file set =
    intersection across boards; clip→logit→drop pooled-zero-var→pooled z-score.
    Returns {(size, tid): vec}, n_cols.

    PARTIAL COVERAGE RAISES rather than being filled — see the check below for why
    the column mean is the one filler that cannot be allowed here."""
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
    # A BLANK CELL RAISES; IT IS NEVER FILLED. Every other block in this module
    # already refuses a gap (see _missing_block_error) and this was the one that
    # did not: it substituted the column mean, which is the worst available
    # filler precisely because it is the most plausible one. After the pooled
    # z-score the column mean IS the population's value, so a problem no bot was
    # asked about enters the fit asserting that the bot found it exactly average
    # — and nothing downstream can tell that apart from a measurement. The tid
    # keeps its rating, its slope, its floor and its axis vector, priced in part
    # on a number that was invented here.
    #
    # Note the two gap shapes are caught in different places and both are loud:
    # a problem NO bot attempted never reaches this matrix (`usable` drops it)
    # and joint_feature_tail raises on it; a problem SOME bots attempted lands
    # here as a hole in an otherwise real row, which is this check.
    if np.isnan(X).any():
        rows, colsi = np.where(np.isnan(X))
        per_bot = {common[c]: int((colsi == c).sum())
                   for c in sorted(set(colsi.tolist()))}
        examples = [f"{keys[r][0]}/{keys[r][1]}" for r in rows[:5].tolist()]
        raise ValueError(
            f"bot_winrate has {len(rows)} MISSING CELLS across "
            f"{len(per_bot)} bot(s) — partial coverage, which this block refuses "
            f"to impute. Per bot (missing attempt rows): {per_bot}. Example "
            f"problems: {examples}. Fix the DATA, not the reader: either run "
            f"`./.venv/bin/python3 -m pipeline.benchmark_bots <size>` so the "
            f"bot covers the whole corpus (it is incremental — each bot only "
            f"solves the tids missing from its own attempt file), or, if the bot "
            f"is RETIRED and nothing regenerates it, delete its attempts/<size>/ "
            f"file on BOTH boards so its column leaves the pooled layout "
            f"entirely. A retired bot's partial file is not a cheap extra "
            f"feature; it is a column that is mostly invented.")
    keep = np.nanstd(X, axis=0) > 1e-9
    X = X[:, keep]
    X = (X - X.mean(0)) / X.std(0)
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, int(keep.sum())


def load_joint_move_features(sizes, tids_by_size):
    """Candidate-move evaluation features (the winning move + its 9 best alternatives), pooled
    and z-scored across boards (a feature means the same on 9×9 and 13×13). Reads the per-size
    RAW caches written by pipeline.cache_rating_features and applies the pooled column z-score
    here — so the cache is reusable across tid subsets, like load_joint_bot_winrate. Returns
    {(size, tid): vec}, n_cols. Raises with a build hint if a cache is missing."""
    keys, rows, ncol = [], [], None
    for size in sizes:
        path = os.path.join(CACHE_DIR, f"move_evals_{size}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"move-feature cache {path} missing — run "
                f"`./.venv/bin/python3 -m pipeline.cache_rating_features` first")
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


def load_joint_continuation(sizes, tids_by_size):
    """Continuation evaluations — the principal line AFTER the answer, each node's explored
    children as (logit win-prob, score) oriented to the SOLVER — pooled and z-scored across
    boards. Reads the `cont` array of the same per-size cache load_joint_move_features reads
    (pipeline.cache_rating_features writes both from one parse). Returns {(size, tid): vec},
    n_cols.

    This is the block that carries branching PAST the first move: how many continuations
    still hold the win, and how sharply the losing ones are punished. The move block above
    cannot carry it — the corpus has exactly one winning first move by construction, so its
    alternative slots are pinned at the loss clip. Kept RAW and separate for the same reason
    every other block is: the counting is the net's job, and a distinct column range is what
    lets it take its own dropout.

    Raises with a build hint if a cache is missing, and treats a cache without the array as
    missing too — a pre-continuation cache must be rebuilt, not silently read as zero-width."""
    keys, rows, ncol = [], [], None
    for size in sizes:
        path = os.path.join(CACHE_DIR, f"move_evals_{size}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"move-feature cache {path} missing — run "
                f"`./.venv/bin/python3 -m pipeline.cache_rating_features` first")
        d = np.load(path, allow_pickle=False)
        if "cont" not in d.files:
            raise FileNotFoundError(
                f"{path} predates the continuation block — re-run "
                f"`./.venv/bin/python3 -m pipeline.cache_rating_features` to add it")
        cont = d["cont"]                                  # decompress ONCE (NpzFile re-reads)
        ncol = int(cont.shape[1])
        row = {t: i for i, t in enumerate(d["tids"].tolist())}
        for t in tids_by_size[size]:
            i = row.get(t)
            if i is not None:
                keys.append((size, t))
                rows.append(cont[i].copy())               # detach so `cont`'s base can free
    X = np.array(rows, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # pooled column z-score
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, ncol


# --------------------------------------------------------------------------------
# Scalar problem axes: tightness gaps (from the move-eval cache) and game phase
# (from the SGFs, cached). These are the forward-validated per-user axes of
# docs/methodology.md, and they are what the MOTIF RULES are written over
# (utils.rating.motif_rules, evaluated by pipeline.build_motif_proxies, which reads
# load_game_phase below) rather than trunk inputs. Living here keeps the rule
# builder, the experiment surface (auxiliary.rating_dev.*) and the axis analyses
# (ablations.eval_score_gap_axis) on ONE set of definitions: same caches, same
# transforms, in sync by construction.


def _game_phase_cache_path(size):
    return os.path.join(CACHE_DIR, f"game_phase_{size}.json")


def _game_phase_one(path):
    """(n_setup, prefix_moves, main_len) for one tsumego SGF.

    prefix = the linear move chain from root to the branch point (the game that
    led to the contested position); main_len = non-pass moves on the principal
    line (first-child chain) of the CORRECT first move — here the child with the
    best winrate comment, score breaking ties, matching cache_rating_features
    slot 0. The principal line terminates in passes by construction."""
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
    signal cache (multiprocess SGF parse).

    INCREMENTAL and RECONCILING IN BOTH DIRECTIONS, like the other roll-ups in
    the signal cache: each entry is a pure per-SGF parse, so reusing the cached
    entries and parsing only the corpus tids missing from them is identical to a
    full rebuild while skipping the parses that dominate. "Up to date" means the
    cached tid set EQUALS the corpus, never just "nothing new to add", and BOTH
    directions are load-bearing here:

      a tid the cache LACKS is not a neutral omission — it is missing data for a
      live problem, and every consumer RAISES on it: pipeline.build_motif_proxies
      cannot label the problem, so it aborts rather than shrinking the taught set,
      and joint_feature_tail raises for a fit assembling the "all" layout. A zero
      vector in its place is exactly the population mean after the pooled z-score,
      so the problem would be labelled, rated and shipped on an imputed phase block
      with nothing anywhere recording it. Keeping this cache in step with the corpus
      is what stops that error being reached; it is not what stops it being silent.

      a DEPARTED tid left behind is not inert either — phase_ratio_z takes its
      z-scores over this cache's whole population, so a purged problem shifts
      the feature for every problem that remains.

    rebuild=True re-parses every SGF (for a corrupt cache, not routine)."""
    import json as _json
    from concurrent.futures import ProcessPoolExecutor, as_completed
    res = {}
    for size in sizes:
        cache = _game_phase_cache_path(size)
        sgf_dir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        ids = sorted(n[:-4] for n in os.listdir(sgf_dir) if n.endswith(".sgf"))
        old = {}
        if os.path.exists(cache) and not rebuild:
            old = _json.load(open(cache))
        keep = {t: old[t] for t in ids if t in old}      # prunes the departed
        missing = [t for t in ids if t not in keep]
        departed = len(old) - len(keep)
        if missing:
            print(f"[size {size}] game-phase cache: {len(ids)} problems "
                  f"({len(missing)} to parse, {len(keep)} reused"
                  f"{f', {departed} departed pruned' if departed else ''}) "
                  f"-> {cache}", flush=True)
            paths = [os.path.join(sgf_dir, t + ".sgf") for t in missing]
            got = {}
            with ProcessPoolExecutor() as ex:
                futs = [ex.submit(_game_phase_chunk, paths[i:i + 500])
                        for i in range(0, len(paths), 500)]
                for fu in as_completed(futs):
                    got.update(fu.result())
            # A parse failure stays OUT of the cache, so it is retried next run
            # rather than cached as absent — but it is announced, because an
            # absent tid is exactly the silent split described above.
            failed = sorted(t for t, v in got.items() if v is None)
            if failed:
                print(f"  [size {size}] {len(failed)} SGFs failed to parse and are "
                      f"NOT cached (e.g. {failed[:5]}) — train_mirt will impute "
                      f"them", flush=True)
            keep.update({t: got[t] for t in missing
                         if got.get(t) is not None})
        elif departed:
            print(f"[size {size}] game-phase cache: pruning {departed} departed "
                  f"tids ({len(keep)} problems)", flush=True)
        if rebuild or set(keep) != set(old):
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache + ".tmp", "w") as f:
                _json.dump(keep, f)
            os.replace(cache + ".tmp", cache)
        res.update({t: (*v, size) for t, v in keep.items()})
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
                f"`./.venv/bin/python3 -m pipeline.cache_rating_features` first")
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
    signal (the raw total self-cancels; see docs/methodology.md)."""
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


# ONE file for both boards, keyed by the `size` column, beside the other
# feature-block caches. Per-size FILES are the convention for arrays loaded one
# board at a time (correct_vecs_<size>.npy, move_evals_<size>.npz); a per-problem
# TABLE follows axes/per_problem.csv instead, which is single-file with a size
# column — splitting it would duplicate in the filename what the column says.
OWNERSHIP_CSV = os.path.join(CACHE_DIR, "ownership_swing.csv")

# The swing decomposition pipeline.extract_ownership_swing writes, minus its
# roll-up columns: each roll-up is the exact sum of two parts below and
# stone_swing_abs the sum of all eight, so including them would ship exact linear
# dependencies into the trunk. empty_swing_abs is kept because it is the one
# quantity NOT implied by the eight — the territory the move moves rather than the
# stones, which is what separates a trade from a pure life-and-death problem.
OWNERSHIP_COLS = (
    "own_dead_revived", "own_alive_secured", "own_alive_killed", "own_dead_deadened",
    "opp_alive_killed", "opp_dead_deadened", "opp_dead_revived", "opp_alive_secured",
    "empty_swing_abs",
)


def load_joint_ownership(sizes, tids_by_size):
    """Ownership-swing features — how many fractional stones the correct move moves,
    against the best REFUTATION THAT WAS READ TO SCORING — pooled and z-scored across
    boards like the other blocks. Returns {(size, tid): vec}, n_cols.

    A row whose `alt_move` is blank (no refutation was read to a conclusion, so
    the swing is undefined) is NOT emitted, and is therefore missing rather than zero: a zero swing
    positively asserts the problem has no life-and-death content, which is a different
    claim from having failed to measure it. That absence is a hard error wherever the
    block is consumed — pipeline.build_motif_proxies, which writes the two ownership
    rules and aborts rather than shipping a short label file, and joint_feature_tail
    for a fit assembling the "all" layout.
    """
    if not os.path.exists(OWNERSHIP_CSV):
        raise FileNotFoundError(
            f"ownership cache {OWNERSHIP_CSV} missing — run "
            f"`./.venv/bin/python3 -m pipeline.extract_ownership_swing` first")
    have = {}
    with open(OWNERSHIP_CSV, newline="") as fh:
        for rec in csv.DictReader(fh):
            if not rec.get("alt_move"):
                continue
            have[rec["tid"]] = [float(rec[c]) for c in OWNERSHIP_COLS]
    keys, rows = [], []
    for size in sizes:
        for t in tids_by_size[size]:
            v = have.get(t)
            if v is not None:
                keys.append((size, t))
                rows.append(v)
    X = np.array(rows, dtype=np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)            # pooled column z-score
    return {k: X[i].astype(np.float32) for i, k in enumerate(keys)}, len(OWNERSHIP_COLS)


# The board-size indicator joint_feature_tail appends after every wide block. It is
# the ONE column INPUT_DROPOUT does not mask, since it identifies the board rather
# than the problem, so callers size the trunk as `k_act + n_wide + SIZE_FLAG_COLS`.
SIZE_FLAG_COLS = 1


def load_joint_tail_blocks(sizes, tids_by_size):
    """The "all" layout: every per-problem block, loaded once, in THE order the
    trunk is trained on. Returns (wide, n_wide).

    EVERY BLOCK IS MASKED — bot winrate, move evals, ownership and game phase all
    sit inside the INPUT_DROPOUT region; the board-size flag past them is the one
    exception and is SIZE_FLAG_COLS above, not a block.

    This exists so the order is stated ONCE. Consumers build trunk input from the
    same blocks in different places — the trainer here, pipeline.write_mirt_per_cell
    (which substitutes per-CELL activations for the correct-cell block) and
    utils.rating.residual_axes (which appends columns of its own) — and a copy of
    the concatenation agrees only by convention: a block added to one and not the
    others lands in a different column than the trunk was fitted with, and both
    files stay well-formed. Which of these blocks a given fit actually sees is
    load_joint_blocks below, not this function.
    """
    bw, nb = load_joint_bot_winrate(sizes, tids_by_size)
    mv, nmv = load_joint_move_features(sizes, tids_by_size)
    ow, now = load_joint_ownership(sizes, tids_by_size)
    ph, nph = load_joint_game_phase(sizes, tids_by_size)
    wide = (("bot_winrate", bw), ("move_evals", mv), ("ownership", ow),
            ("game_phase", ph))
    return wide, nb + nmv + now + nph


# The block LAYOUTS a fit can be assembled on. Which one ships is
# model_constants.PRODUCTION_BLOCKS; the rest are comparison arms.
FEATURE_BLOCKS = ("all", "activations", "activations+bot", "activations+bot+delta")

# Layouts that may be MEASURED but never SHIPPED. A layout lands here when it is
# answer-derived and the trunk it would feed is fanned out per cell — the fan-out
# holds the wide tail fixed at the correct move's values, so such a block describes
# every alternative cell by the answer. Stated as a set rather than a comment so the
# constraint is checked instead of remembered: PRODUCTION_BLOCKS naming one of these
# is a mistake that produces two well-formed files and no error anywhere downstream.
DEV_ONLY_BLOCKS = ("activations+bot+delta",)

if PRODUCTION_BLOCKS in DEV_ONLY_BLOCKS:
    raise RuntimeError(
        f"model_constants.PRODUCTION_BLOCKS is {PRODUCTION_BLOCKS!r}, which is a "
        f"DEV-ONLY layout (DEV_ONLY_BLOCKS): it is answer-derived, so "
        f"pipeline.write_mirt_per_cell would condition every alternative cell on the "
        f"correct move. A dev-only block can only ship by feeding the difficulty heads "
        f"alone, which is an architecture change, not a layout change.")


def load_joint_blocks(blocks, sizes, tids_by_size, pca_dim=None, seed=None):
    """(wide, n_wide) for ONE layout: the per-problem blocks between the activation
    block and the board-size flag, in trunk order.

    THE ONLY place a layout name becomes a block list. Every consumer that feeds the
    trunk — the trainer, the per-cell fanout, the ablation projections — comes through
    here, because a second switch elsewhere feeds a checkpoint columns it was not
    fitted with, and the two arms are both well-formed matrices.

    "activations+bot" keeps the one wide block that does NOT describe the problem
    through its ANSWER, which is what makes it usable as per-cell conditioning: the
    trunk is fanned out over every legal cell, so an answer-derived block hands every
    alternative cell a description of the correct move. "activations" drops every wide
    block. "all" is every block load_joint_tail_blocks states. The size flag is not a
    block and stays in every layout (it is the board the problem is on, not a property
    measured about it), so the layouts differ in columns and in nothing else.

    "activations+bot+delta" is a DEV ARM (DEV_ONLY_BLOCKS): "activations+bot" plus the
    correct-vs-alternative activation contrast, which is answer-derived and so unusable
    as per-cell conditioning for the reason just given. `pca_dim` and `seed` exist for
    it — the contrast is taken in the joint-PCA basis, which those two identify — and
    every other layout ignores them.
    """
    if blocks == "all":
        return load_joint_tail_blocks(sizes, tids_by_size)
    if blocks == "activations+bot":
        bw, nb = load_joint_bot_winrate(sizes, tids_by_size)
        return (("bot_winrate", bw),), nb
    if blocks == "activations+bot+delta":
        if pca_dim is None or seed is None:
            raise ValueError(
                "the 'activations+bot+delta' layout needs pca_dim and seed: the "
                "contrast is taken in the joint-PCA basis, and a basis identified by "
                "different settings than the activation block's is a different "
                "coordinate system — call load_joint_blocks(..., pca_dim=..., seed=...)")
        bw, nb = load_joint_bot_winrate(sizes, tids_by_size)
        dl, nd = load_joint_act_delta(sizes, tids_by_size, pca_dim, seed)
        return (("bot_winrate", bw), ("act_delta", dl)), nb + nd
    if blocks == "activations":
        return (), 0
    raise ValueError(f"blocks must be one of {FEATURE_BLOCKS}, got {blocks!r}")


def corpus_tids_by_size(sizes):
    """THE problem set: the tids with an SGF in tsumegos/<size>/. The corpus is the
    only ground truth for what a problem IS — every other id list is an artifact
    listing that can be stale in either direction.

    Not correct_ids_<size>.json (derived from the activation .npz listing), because
    that is wrong in both directions and each direction is a silent fault:

      an id it has that the corpus LACKS is a PHANTOM — the SGF was removed while its
      .npz, bot attempts and ratings row survived, so a problem that no longer exists
      gets rated, given an axis vector and a floor;

      an id the corpus has that IT lacks is MISSING DATA — a tsumego that was never
      extracted, silently dropped from the rated set instead of failing. Resolving a
      gap by omission is no better than resolving it by imputation; both are
      invisible. Extraction being resumable is why THAT STEP can be re-run, not a
      licence to train on whatever happens to be on disk.

    A retired id needs no filtering: it has no SGF by construction, because
    tsumego_id_taken refuses to reissue one. That is an invariant maintained by
    the generator, so it is CHECKED here rather than assumed — see
    _assert_ids_unique.
    """
    out = {}
    for size in sizes:
        d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        if not os.path.isdir(d):
            raise SystemExit(f"corpus dir missing: {d}")
        out[size] = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".sgf"))
    _assert_ids_unique(out)
    return out


def _assert_ids_unique(tids_by_size):
    """The 8-hex id must name at most one problem across the WHOLE namespace —
    every board size and the retired list together.

    Uniqueness is enforced at generation time (generate_tsumegos.tsumego_id_taken
    checks all sizes, staging and the retired list) but nothing verifies it
    afterwards, and the id is a TRUNCATED hash: only 32 bits, so distinct
    positions collide at a rate growing with the SQUARE of the corpus. The
    generator discards a colliding candidate; what this catches is the enforcement
    having been bypassed — a corpus assembled by hand, a restore from a stale
    backup, a purge that put SGFs back, an id remap applied to one size only.

    Both failures are silent and neither is repairable after the fact, which is
    why they are worth a hard stop before a fit rather than a warning:

      SAME ID ON BOTH BOARDS — every id-keyed artifact (activations, attempts,
      ratings, per_cell, the shipped lookup which spans both sizes) would have
      two problems fighting over one row, and the client's binary search on
      lookup.txt would resolve to whichever landed there.

      A RETIRED ID BACK IN THE CORPUS — the problem's attempt rows were kept when
      it left, so the new occupant silently inherits a stranger's history: it is
      rated on other people's answers, and nothing anywhere records it.
    """
    seen = {}
    clashes = []
    for size, tids in tids_by_size.items():
        for t in tids:
            if t in seen:
                clashes.append(f"{t}: in tsumegos/{seen[t]}/ AND tsumegos/{size}/")
            else:
                seen[t] = size
    retired = read_retired_ids()
    for t in sorted(set(seen) & retired):
        clashes.append(f"{t}: in tsumegos/{seen[t]}/ AND listed in "
                       f"{os.path.relpath(RETIRED_IDS_PATH, PROJECT_ROOT)}")
    if clashes:
        shown = "\n  ".join(clashes[:20])
        more = f"\n  ... and {len(clashes) - 20} more" if len(clashes) > 20 else ""
        raise SystemExit(
            f"TSUMEGO ID COLLISION — {len(clashes)} id(s) name more than one problem:\n"
            f"  {shown}{more}\n"
            f"Refusing to build a problem set on an ambiguous namespace. Resolve each "
            f"before fitting: a cross-board clash needs one side re-hashed or removed "
            f"(auxiliary/check_corpus_ids.py --emit-map), and a retired id back in the "
            f"corpus needs the SGF removed unless you intend it to inherit that id's "
            f"existing attempt rows.")


def _missing_block_error(name, key):
    """Classify a gap before raising, because the two causes need opposite fixes."""
    size, tid = key
    corpus = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    if not os.path.exists(os.path.join(corpus, f"{tid}.sgf")):
        return KeyError(
            f"{name} has no row for {tid} (size {size}), and NEITHER DOES THE CORPUS — "
            f"{corpus}/{tid}.sgf is gone, so this is a PHANTOM: the problem was removed "
            f"but its activations/{size}/{tid}.npz survived and carried it into "
            f"correct_ids. Delete the orphan .npz (keep a backup — with the SGF gone it "
            f"is the last trace of that position) and re-run pipeline.cache_rating_features, "
            f"which asserts activations are a subset of the corpus.")
    return KeyError(
        f"{name} has no row for {tid} (size {size}), which IS a live problem in {corpus}. "
        f"MISSING DATA for a real tsumego — rebuild that block (extract_activations, "
        f"benchmark_bots, cache_rating_features or extract_ownership_swing, per the block "
        f"named above) so it covers the whole corpus. It is neither zero-filled nor "
        f"dropped: a zero row is the population mean after the pooled z-score, and a "
        f"dropped row removes the problem from the rated set — both silently.")


def joint_feature_tail(key, wide):
    """The per-problem columns after the activation block, plus the board-size flag.

    A block that lacks `key` RAISES. There is deliberately no default: an absent row
    is either a phantom or a genuine gap (see _missing_block_error), and both are
    invisible once imputed.
    """
    parts = []
    for name, block in wide:
        try:
            parts.append(block[key])
        except KeyError:
            raise _missing_block_error(name, key) from None
    parts.append(np.array([1.0 if key[0] == 13 else 0.0], dtype=np.float32))
    return np.concatenate(parts)


def load_joint_mirt_features(sizes, pca_dim, seed, blocks=PRODUCTION_BLOCKS):
    """THE per-problem feature matrix the MIRT trunk sees — the joint-PCA activation
    block, the wide blocks `blocks` selects, then the board-size flag — assembled once
    here so every consumer sees the same columns in the same order.

    Returns (feat (n_problems, fdim) float32, keys [(size, tid)], key_idx {key: row},
    input_dim). `input_dim` is the leading block INPUT_DROPOUT masks: the activation
    block plus every wide per-problem block, excluding only the board-size flag.

    THE PROBLEM SET IS THE CORPUS, and EVERY block must cover it — activations
    included. A tsumego with no activation vector is missing data exactly like one with
    no move-eval row, and is raised on rather than quietly left out of the fit. The
    earlier set (correct_ids ∩ bot winrate) silently resolved both faults by omission:
    an unextracted problem simply vanished from the rated set, and a removed problem
    stayed in it.

    Second caller is the motif labeler (auxiliary/labeler), whose whole point is to
    probe THIS space — a direction it finds in some other feature space need not be one
    the trunk can represent, so the axis bank it builds would be teaching the fit
    something unreachable. Copying the assembly there would let the two drift silently,
    which is the failure this single source exists to prevent.

    `blocks` selects WHICH per-problem blocks are assembled (load_joint_blocks holds
    the layouts), and exists so a feature ablation is expressed HERE rather than by
    slicing the returned matrix at a caller — a slice has to restate where the
    activation block ends, which is the one number this function exists to state once.
    It defaults to PRODUCTION_BLOCKS, so a caller that means "what the shipped trunk
    sees" gets it without naming a layout.
    """
    # Membership only, so a typo fails before the joint-PCA load rather than after
    # it; the layouts themselves are load_joint_blocks', which is the one switch.
    if blocks not in FEATURE_BLOCKS:
        raise ValueError(f"blocks must be one of {FEATURE_BLOCKS}, got {blocks!r}")
    tids_by_size = corpus_tids_by_size(sizes)
    act, k_act = load_joint_activations(sizes, tids_by_size, pca_dim, seed)
    # pca_dim/seed are passed for the layouts that take a block in the joint-PCA basis
    # (the dev delta arm); every other layout ignores them, so the shipped layouts are
    # assembled exactly as before.
    wide, n_wide = load_joint_blocks(blocks, sizes, tids_by_size, pca_dim, seed)
    keys = [(s, t) for s in sizes for t in tids_by_size[s]]
    key_idx = {key: i for i, key in enumerate(keys)}
    input_dim = k_act + n_wide
    feat = np.empty((len(keys), input_dim + SIZE_FLAG_COLS), dtype=np.float32)
    for key, i in key_idx.items():
        try:
            act_vec = act[key]
        except KeyError:
            raise _missing_block_error("activations", key) from None
        feat[i] = np.concatenate([act_vec, joint_feature_tail(key, wide)])
    return feat, keys, key_idx, input_dim


# --------------------------------------------------------------------------------
# Backend rating outputs (ratings/<size>/*.csv — the stage BEFORE package_for_pwa).
# Model-development consumers (auxiliary.rating_dev.*, the axis batteries) read
# THESE, not pwa/tsumegos/problem_params.bin: same values (packaging just copies),
# but available the moment training finishes and immune to a stale/half-packaged
# dist. Client-replica harnesses (leaderboard replay, user_weakness_report) keep
# reading the shipped files on purpose — bit-fidelity to what clients see IS
# their job.

RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")


def load_backend_problem_params(sizes=(9, 13), ratings_dir=None):
    """tid -> (rating, slope, floor) from ratings/<size>/tsumego_{ratings,quality,
    tsumego_{ratings,slopes,floors}.csv. Fails loud on a missing file (run
    pipeline.train_mirt).

    `ratings_dir` overrides the default RATINGS_DIR and exists for HELD-OUT
    evaluation: these values feed the axis features and the axis baseline, so
    reading the shipped fit while measuring on a held-out problem set hands the
    axes rating/slope/floor derived from attempts on exactly the problems being
    scored. Production passes nothing and is unaffected."""
    root = ratings_dir or RATINGS_DIR
    out = {}
    for size in sizes:
        d = os.path.join(root, str(size))
        frames = {}
        for name, col in ((RATINGS_CSV, "rating"),
                          (SLOPE_CSV, "slope"),
                          (FLOOR_CSV, None)):
            path = os.path.join(d, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing — run pipeline.train_mirt first")
            frames[name] = pd.read_csv(path)
        r = frames[RATINGS_CSV].set_index("tsumego_id")["rating"]
        q = frames[SLOPE_CSV].set_index("tsumego_id")["slope"]
        fdf = frames[FLOOR_CSV].set_index("tsumego_id")
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
