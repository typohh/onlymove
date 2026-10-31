"""A fingerprint of the feature space the labeler's cached pool was built in.

`precompute_features` stamps one into pool.json; the server recomputes it at
startup and REFUSES TO RUN if it differs. The failure this exists to stop is
silent: every block in `load_joint_mirt_features` is pooled z-scored over the
whole corpus and the activation block is a joint PCA over it, so a corpus top-up,
a re-extraction or a new feature block changes every number in features.npy
without changing its shape. A probe fitted on the old numbers still fits, still
predicts, and is a function of a space the trunk no longer reads — which is
exactly the drift the single-source assembly exists to prevent, arriving through
the cache instead of through a copied assembly.

It is deliberately made of CHEAP STATS — file size, mtime, and the corpus id
listing — rather than content hashes of multi-GB caches, so the check costs a
fraction of a second at startup. That over-invalidates: re-running a step that
produces identical bytes moves an mtime and trips it. The cost of that is one
`precompute_features` run, against a stale-space probe that nothing else can
detect, so the trade goes this way deliberately.
"""

import hashlib
import os

from utils.paths import tsumego_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")


def _stat(path):
    """(bytes, whole-second mtime), or None when the file is absent — absent is a
    value here, so a block appearing or disappearing registers as a change."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return [st.st_size, int(st.st_mtime)]


def _digest(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:16]


def _corpus_digest(size):
    """Over the id LISTING, not the file count: a purge that removes one problem
    and a top-up that adds another leaves the count identical while every pooled
    z-score and the PCA basis move."""
    d = tsumego_dir(size)
    if not os.path.isdir(d):
        return None
    tids = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".sgf"))
    return [len(tids), _digest(tids)]


def _bot_digest(size):
    d = os.path.join(PROJECT_ROOT, "attempts", str(size))
    if not os.path.isdir(d):
        return None
    files = sorted((f, *_stat(os.path.join(d, f)))
                   for f in os.listdir(d) if f.endswith(".txt"))
    return [len(files), _digest(files)]


def feature_fingerprint(sizes, pca_dim, seed):
    """One flat dict, so a mismatch can be reported key by key rather than as an
    opaque hash inequality."""
    fp = {
        "pca_dim": int(pca_dim),
        "seed": int(seed),
        "sizes": sorted(int(s) for s in sizes),
        "ownership": _stat(os.path.join(CACHE_DIR, "ownership_swing.csv")),
    }
    for size in sorted(sizes):
        fp[f"corpus_{size}"] = _corpus_digest(size)
        fp[f"bots_{size}"] = _bot_digest(size)
        fp[f"activations_{size}"] = [
            _stat(os.path.join(CACHE_DIR, f"correct_vecs_{size}.npy")),
            _stat(os.path.join(CACHE_DIR, f"correct_ids_{size}.json")),
        ]
        fp[f"move_evals_{size}"] = _stat(os.path.join(CACHE_DIR, f"move_evals_{size}.npz"))
        fp[f"game_phase_{size}"] = _stat(os.path.join(CACHE_DIR, f"game_phase_{size}.json"))
    return fp


def describe_mismatch(want, have):
    """The keys that differ, most-explanatory first. Returns [] when they agree."""
    out = []
    for key in sorted(set(want) | set(have)):
        a, b = want.get(key), have.get(key)
        if a != b:
            out.append(f"  {key}: pool built with {a!r}, now {b!r}")
    return out
