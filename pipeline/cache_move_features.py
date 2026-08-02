"""Cache candidate-move evaluation features (the winning move + its 9 best alternatives) for
the rating model.

Every tsumego SGF embeds, on each candidate first-move node, a KataGo evaluation comment
"<winrate>%/<score>" (Black-perspective win rate, Black-minus-White score lead) — see the
"PWA SGF format carries embedded analysis" note in CLAUDE.md. The generator evaluates (almost)
every legal first move, so a problem carries dozens of these. This script reads them for every
problem, orients each to the SIDE TO MOVE (so "win prob" means "the solver's win prob" on either
colour), sorts the moves by win probability, and keeps the WINNING move plus the 9 next-best
alternatives. For each of those 10 moves it stores (logit win-prob, clipped score): a fixed
20-dim raw feature vector per problem describing how sharply the best move stands out from its
rivals — a difficulty signal orthogonal to the trunk activations and the bot-ladder win rates
that train_rating already feeds.

Output (one per board size), consumed by train_rating --move-features through
utils.rating.features.load_joint_move_features:
  activations/_signal_cache/move_evals_<size>.npz
      tids   (n,)      <U8   canonical problem id (orientation-invariant hash)
      feats  (n, 20)   f32   RAW [logit win-prob, score] × 10 moves (solution, then 9 alts)

The npz stores RAW features; the loader pools + z-scores them across boards at train time, so
the cache is reusable across tid subsets. The tid set is the activation id cache for the size
(activations/_signal_cache/correct_ids_<size>.json) — exactly the problems train_rating can
score.

This step ALSO refreshes that activation cache (utils.rating.features.build_correct_vec_cache),
because it is the other roll-up of extract_activations' per-problem .npz output and this step's
own tid set comes from it. So one command brings both aggregate caches forward after a tsumego
top-up; neither can be left stale behind the other. Run once per size (both cached by default):

  ./.venv/bin/python3 -m pipeline.cache_move_features [--sizes 9,13] [--workers N] [--force]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import sgfmill.sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import (  # noqa: E402
    CACHE_DIR, _cache_ids_path, build_correct_vec_cache,
)
from pipeline.benchmark_bots import parse_tsumego_sgf  # noqa: E402

# Feature layout: the winning move + N_ALT best alternatives, each as (logit win-prob, score).
N_ALT = 9
N_MOVES = 1 + N_ALT
N_FEATS = 2 * N_MOVES                 # = 20
WP_CLIP = 1e-3                        # win-prob clip before the logit (KataGo emits exact 0/100%)
SCORE_CLIP = 50.0                     # tame the rare blow-out leads before pooled z-scoring
CHUNK = 500                           # SGFs per worker task (amortizes pickling over the parse)
# "100.0%/34.1" / "44.6%/-0.3" — Black-perspective win rate %, then Black−White score lead.
CMT_RE = re.compile(r"(\d+\.?\d*)%/([-]?\d+\.?\d*)")


def _cache_path(size):
    return os.path.join(CACHE_DIR, f"move_evals_{size}.npz")


def _logit(p):
    p = min(max(float(p), WP_CLIP), 1.0 - WP_CLIP)
    return float(np.log(p / (1.0 - p)))


def extract_move_features(sgf_bytes):
    """The 20-dim raw feature vector for one SGF, or None if it has no usable evaluations.

    All candidate first-move nodes share the side to move; each comment is read and oriented to
    that side (win prob and score from the solver's perspective). Moves are sorted best→worst by
    (win-prob, score) so slot 0 IS the winning move (the same choice as the PWA's
    findBestMoveFromChildren — highest win rate, score breaking ties), and slots 1..9 are the
    best alternatives. The rare problem with <10 evaluated moves forward-fills its last (worst)
    move into the empty slots (gap-to-a-missing-alternative = 0)."""
    game = sgfmill.sgf.Sgf_game.from_bytes(sgf_bytes)
    _setup, vnodes = parse_tsumego_sgf(game)
    moves = []                                       # (win-prob_stm, score_stm)
    for node in vnodes:
        if not (node.has_property("B") or node.has_property("W")):
            continue
        color, _mb = node.get_raw_move()
        black_to_move = color == "b"
        wr = sc = None
        # A comment-less move node is normal (not every node carries an eval),
        # so skip just that node (wr stays None → continue). get_raw_list("C")
        # would raise KeyError on such a node, so gate on has_property first —
        # this is expected-data handling, NOT error tolerance: a genuinely
        # unreadable SGF still propagates out of _process_chunk and aborts.
        if node.has_property("C"):
            for cb in node.get_raw_list("C"):
                m = CMT_RE.search(cb.decode("utf-8"))
                if m:
                    wr = float(m.group(1)) / 100.0
                    sc = float(m.group(2))
                    break
        if wr is None:
            continue
        moves.append((wr, sc) if black_to_move else (1.0 - wr, -sc))
    if not moves:
        return None
    moves.sort(key=lambda t: (t[0], t[1]), reverse=True)
    feat = np.empty(N_FEATS, dtype=np.float32)
    last = moves[-1]
    for i in range(N_MOVES):
        wp, sc = moves[i] if i < len(moves) else last
        feat[2 * i] = _logit(wp)
        feat[2 * i + 1] = float(np.clip(sc, -SCORE_CLIP, SCORE_CLIP))
    return feat


def _process_chunk(task):
    """Worker: feature vectors for a chunk of tids (module scope so it pickles).

    Fail-loud: an unreadable/corrupt SGF, or a problem with no usable move
    evaluations, RAISES and aborts the whole build (the exception surfaces in
    the parent at fu.result()). Dropping such a problem would silently desync
    the cache from correct_ids — train_rating would then score a different
    problem set than it thinks — so a bad input must be fixed, not skipped."""
    size, tids = task
    tdir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    out = np.empty((len(tids), N_FEATS), dtype=np.float32)
    for i, tid in enumerate(tids):
        try:
            with open(os.path.join(tdir, f"{tid}.sgf"), "rb") as fh:
                f = extract_move_features(fh.read())
        except Exception as e:
            raise RuntimeError(f"size {size}: failed to read/parse {tid}.sgf") from e
        if f is None:
            raise RuntimeError(
                f"size {size}: {tid}.sgf has no usable move evaluations "
                f"(missing '<winrate>%/<score>' comments)")
        out[i] = f
    return tids, out


def build_size(size, workers, force, rebuild_activations=False):
    out_path = _cache_path(size)
    ids_path = _cache_ids_path(size)

    # Refresh the correct-vector cache FIRST: correct_ids is this step's tid set and the
    # rating model's scorable set, and both roll up the per-problem activation .npz files
    # that extract_activations writes. Refreshing here means one command brings both
    # aggregate caches forward after a tsumego top-up, instead of a stale correct_ids
    # silently truncating this cache and the rated set alike. It is a cheap no-op when
    # already current. A size with no activations at all is skipped, not an error.
    #
    # --force does NOT reach it: forcing a move-feature re-extraction is a minutes-long
    # SGF reparse, while forcing the activation cache is a full re-read of every .npz in
    # the corpus. --rebuild-activation-cache asks for that separately.
    act_dir = os.path.join(PROJECT_ROOT, "activations", str(size))
    if not os.path.isdir(act_dir):
        print(f"size {size}: no activations dir ({act_dir}); skipping")
        return
    build_correct_vec_cache(size, workers=workers, force=rebuild_activations)

    if not os.path.exists(ids_path):
        print(f"size {size}: no activation id cache ({ids_path}); skipping")
        return
    tids = json.load(open(ids_path))

    # INCREMENTAL: each row is a pure per-problem function of its SGF's embedded
    # KataGo eval (no training, no cross-problem dependency), so reusing the already-
    # cached rows and extracting only the tids missing from the cache is byte-identical
    # to a full rebuild — at a fraction of the cost. This also makes the no-force path
    # self-updating: a top-up of new tsumegos is picked up automatically instead of the
    # old "skip if the file exists" (which silently left the cache stale). --force still
    # re-extracts every row.
    cached = {}
    if os.path.exists(out_path) and not force:
        with np.load(out_path) as d:
            cached = {t: r for t, r in zip(d["tids"].tolist(), d["feats"])}
    to_extract = tids if force else [t for t in tids if t not in cached]
    # Surplus = cached rows whose tid has since LEFT correct_ids (a corpus purge). The write
    # path below emits in correct_ids order and so prunes them, but only if we get there:
    # short-circuiting on `not to_extract` alone would strand them, since nothing new needs
    # extracting. The cache is up to date only when its tid set EQUALS the target set.
    target = set(tids)
    surplus = [t for t in cached if t not in target]
    if not to_extract and not surplus:
        print(f"size {size}: cache up to date ({len(cached)} problems)")
        return
    if surplus:
        print(f"size {size}: pruning {len(surplus)} rows no longer in correct_ids "
              f"(e.g. {surplus[:3]})")

    idx = {t: i for i, t in enumerate(to_extract)}
    new_feats = np.full((len(to_extract), N_FEATS), np.nan, dtype=np.float32)
    chunks = [(size, to_extract[i:i + CHUNK]) for i in range(0, len(to_extract), CHUNK)]
    t0 = time.time()
    done = 0
    # A prune-only run has nothing to extract, so `chunks` is empty; skip the worker pool
    # and fall straight through to the write, which is what does the pruning.
    if chunks:
        print(f"size {size}: {len(to_extract)} to extract ({len(cached)} cached), "
              f"{len(chunks)} chunks, {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_chunk, c) for c in chunks]
            for k, fu in enumerate(as_completed(futs)):
                ctids, cout = fu.result()
                for t, row in zip(ctids, cout):
                    new_feats[idx[t]] = row
                done += len(ctids)
                if (k + 1) % 50 == 0 or done == len(to_extract):
                    print(f"  {done}/{len(to_extract)} "
                          f"({done / max(time.time() - t0, 1e-9):.0f}/s)")
    for t, row in zip(to_extract, new_feats):
        cached[t] = row

    # Emit in correct_ids order. Every tid is guaranteed present and finite — a
    # failed extraction aborts the run above — so nothing is dropped and the
    # cache tid set matches correct_ids exactly. The finite check is a loud
    # backstop (replacing the old silent drop): a corrupt cache must not ship.
    out_tids = list(tids)
    out_rows = np.array([cached[t] for t in tids],
                        dtype=np.float32).reshape(len(tids), N_FEATS)
    if not np.isfinite(out_rows).all():
        bad = [t for t, r in zip(tids, out_rows) if not np.isfinite(r).all()]
        raise RuntimeError(
            f"size {size}: {len(bad)} non-finite feature rows (e.g. {bad[:5]}); "
            f"refusing to write a corrupt cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}.npz"
    np.savez(tmp, tids=np.array(out_tids), feats=out_rows)
    os.replace(tmp, out_path)
    print(f"size {size}: wrote {len(out_tids)} rows to {os.path.basename(out_path)} "
          f"in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="9,13", help="comma-separated board sizes to cache")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true",
                    help="re-extract every move-feature row (leaves the activation cache "
                         "on its incremental path)")
    ap.add_argument("--rebuild-activation-cache", action="store_true",
                    help="also re-read every activation .npz into correct_vecs (hours; "
                         "only needed if that cache is corrupt)")
    args = ap.parse_args()
    for s in [int(x) for x in args.sizes.split(",") if x.strip()]:
        build_size(s, max(1, args.workers), args.force, args.rebuild_activation_cache)


if __name__ == "__main__":
    main()
