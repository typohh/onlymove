"""Precompute the labeler's working pool — THE MIRT'S OWN PER-PROBLEM FEATURES.

The features here are `utils.rating.features.load_joint_mirt_features`, byte-for-byte
the matrix `pipeline.train_mirt` feeds its trunk: joint-PCA correct-move activations,
bot winrate, candidate-move evaluations, game phase, board-size flag. They are NOT
assembled here — the assembly is single-sourced in utils so the two cannot drift.

That is the whole point of the tool. A label probe fitted in this space is a function
of exactly what the trunk reads, so a direction it finds is one the trunk can
represent and therefore one `axes/axis_bank.npz` can teach the fit. A probe fitted on
some other view of the position (raw trunk activations, a hand-built residual space)
can find a direction the MIRT has no way to reproduce, and the bank would then be
teaching the fit toward something unreachable.

The pool is a rank-filtered sample: a human cannot label a position they cannot read,
so problems harder than --max-rank are excluded before sampling. Every problem
labels.json already carries an annotation for is pinned in regardless, so rebuilding
the pool never costs hand-entered work.

Output (auxiliary/labeler/cache/):
  features.npy   (N_pool, fdim) float32   MIRT feature rows for the pool
  pool.json      manifest: tids, sizes, ratings, zip_ids, fdim, pca_dim, seed

The labeler server loads this once at startup. Re-run after a corpus top-up or a
`pipeline.cache_rating_features` rebuild, or to refresh / enlarge the pool — the
server ENFORCES that rather than trusting it, refusing to serve a pool whose
`feature_fingerprint` no longer matches the artifacts the features are built from
(fingerprint.py). Rebuilding costs nothing but this run: labels.json is keyed by
problem id, and every annotated problem still in the corpus is pinned straight
back into the new pool.

Usage:
  ./.venv/bin/python3 -m auxiliary.labeler.precompute_features --pool 5000
"""

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.labeler.fingerprint import feature_fingerprint  # noqa: E402
from utils.katago.board_render import load_lookup  # noqa: E402
from utils.rating.features import load_joint_mirt_features  # noqa: E402
from utils.rating.fit_rank_curves import interp_extrap  # noqa: E402
from utils.rating.model_constants import PCA_DIM, SEED, SIZES  # noqa: E402
from utils.rating.rank_surveys import rank_to_value  # noqa: E402
from utils.rating.ratings_io import RATINGS_CSV, load_value_map  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
LABELS_PATH = os.path.join(HERE, "labels.json")


def load_ratings(size):
    return load_value_map(
        os.path.join(PROJECT_ROOT, "ratings", str(size), RATINGS_CSV), "rating")


def parse_max_rank(label):
    """Rank cutoff -> a value on the project rank scale. A kyu/dan label means that
    band's LOWER edge (rank_to_value gives the centre), so '1d' admits everything
    below the 1k/1d boundary and nothing inside the dan range. A bare number passes
    through as a raw rank value."""
    v = rank_to_value(label)
    if v is not None:
        return v - 0.5
    return float(label)


def load_curve():
    with open(os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json")) as f:
        return json.load(f)


def annotated_tids():
    """Problems the labeler already holds human labels for. They are pinned into every
    rebuilt pool: a pool is a fresh sample, and the eligible set moves under it with
    every corpus top-up and refit, so sampling alone would drop hand-entered work that
    nothing can regenerate."""
    if not os.path.exists(LABELS_PATH):
        return set()
    with open(LABELS_PATH) as f:
        return set(json.load(f).get("annotations", {}))


def renderable_rows(keys, max_rank, pinned):
    """Rows of the feature matrix the labeler can work with: rated, and present in the
    shipped lookup so the board can be rendered. Returns (candidates, forced) — both
    lists of (row, size, tid, rating, zip_id).

    `forced` is the already-annotated rows, exempt from the difficulty cutoff. The
    cutoff bounds what a human is ASKED to label, not what the probe may learn from,
    and a labeled row that drifted past it is still evidence.

    A pinned problem that cannot be pinned back is REPORTED BY REASON, not counted
    into one bucket. Hand-entered labels are the one thing here nothing can
    regenerate, so "gone from the corpus" (the label is stranded for good) and
    "rated but absent from lookup.txt" or "in the corpus but unrated" (a packaging
    or ratings step is simply behind, and the label returns when it catches up)
    call for opposite responses and must not read the same.
    """
    curve = load_curve()
    lookup = load_lookup()
    ratings = {s: load_ratings(s) for s in SIZES}
    cand, forced = [], []
    n_unrated = n_nozip = n_hard = 0
    lost = {"unrated": [], "no_zip": []}
    for i, (size, tid) in enumerate(keys):
        rating = ratings.get(size, {}).get(tid)
        if rating is None:
            n_unrated += 1
            if tid in pinned:
                lost["unrated"].append(tid)
            continue
        entry = lookup.get(tid)
        if entry is None:
            n_nozip += 1
            if tid in pinned:
                lost["no_zip"].append(tid)
            continue
        row = (i, int(size), tid, float(rating), entry[0])
        if tid in pinned:
            forced.append(row)
            continue
        c = curve[str(size)]
        if interp_extrap(rating, c["ratings"], c["ranks"]) > max_rank:
            n_hard += 1
            continue
        cand.append(row)
    print(f"  candidates {len(cand)}/{len(keys)} "
          f"(dropped {n_hard} harder than the cutoff, {n_nozip} not in lookup.txt, "
          f"{n_unrated} unrated)")

    seen = {p[2] for p in forced} | set(lost["unrated"]) | set(lost["no_zip"])
    gone = sorted(pinned - seen)
    print(f"  pinned {len(forced)}/{len(pinned)} already-annotated problems")
    for reason, tids in (("gone from the corpus", gone),
                         ("in the corpus but unrated", sorted(lost["unrated"])),
                         ("rated but not in lookup.txt", sorted(lost["no_zip"]))):
        if tids:
            shown = ", ".join(tids[:8]) + (" ..." if len(tids) > 8 else "")
            print(f"    {len(tids)} {reason}: {shown}")
    if gone:
        print(f"    (their labels stay in labels.json — harmless, and they are pinned "
              f"back automatically if the problems ever return)")
    return cand, forced


def sample_pool(cand, forced, pool, seed):
    """Every pinned row, then a stratified fill from the candidates — as even as
    possible across board sizes, over whatever room the pinned rows leave."""
    rng = np.random.default_rng(seed)
    sizes = sorted({e[1] for e in cand})
    per = max(0, (pool - len(forced))) // max(len(sizes), 1)
    picked = list(forced)
    for s in sizes:
        sub = [e for e in cand if e[1] == s]
        take = min(per, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        picked.extend(sub[j] for j in idx)
        print(f"  size {s}: sampled {take}/{len(sub)}")
    rng.shuffle(picked)
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=int, default=5000, help="Total working-pool size.")
    ap.add_argument("--max-rank", default="1d",
                    help="Exclude problems harder than this rank (e.g. '1d', '2k', or "
                         "a bare rank value). Default 1d (= the 1k/1d boundary).")
    ap.add_argument("--seed", type=int, default=0, help="Pool sampling seed.")
    args = ap.parse_args()

    max_rank = parse_max_rank(args.max_rank)
    print(f"loading the MIRT feature matrix (sizes {list(SIZES)}, "
          f"pca_dim {PCA_DIM}, seed {SEED})...", flush=True)
    feat, keys, _, input_dim = load_joint_mirt_features(SIZES, PCA_DIM, SEED)
    print(f"  {feat.shape[0]} problems x {feat.shape[1]} features "
          f"(dropout region {input_dim})")

    print(f"filtering to rank <= {args.max_rank} (rank value <= {max_rank:g}):")
    cand, forced = renderable_rows(keys, max_rank, annotated_tids())
    if not (cand or forced):
        raise SystemExit("no eligible problems; check ratings/ and pwa/lookup.txt")
    picked = sample_pool(cand, forced, args.pool, args.seed)

    rows = [p[0] for p in picked]
    out = feat[rows].copy()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, "features.npy"), out)
    with open(os.path.join(CACHE_DIR, "pool.json"), "w") as f:
        json.dump({
            "tids": [p[2] for p in picked],
            "sizes": [p[1] for p in picked],
            "ratings": [p[3] for p in picked],
            "zip_ids": [p[4] for p in picked],
            "fdim": int(out.shape[1]),
            "input_dim": int(input_dim),
            "pca_dim": int(PCA_DIM),
            "seed": int(SEED),
            "max_rank": args.max_rank,
            "max_rank_value": max_rank,
            # What the server checks this cache against before it will serve — see
            # fingerprint.py. The feature values move under a corpus or block change
            # without the shape moving, so shape alone cannot detect a stale pool.
            "feature_fingerprint": feature_fingerprint(SIZES, PCA_DIM, SEED),
        }, f)
    print(f"cached {len(picked)} feature rows → {CACHE_DIR}/")
    print(f"  features.npy: {out.shape} {out.dtype}")


if __name__ == "__main__":
    main()
