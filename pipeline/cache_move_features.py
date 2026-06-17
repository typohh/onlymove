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
that train_rating_v2 already feeds.

Output (one per board size), consumed by train_rating_v2 --move-features through
auxiliary.compare_rating_models.load_joint_move_features:
  activations/_signal_cache/move_evals_<size>.npz
      tids   (n,)      <U8   canonical problem id (orientation-invariant hash)
      feats  (n, 20)   f32   RAW [logit win-prob, score] × 10 moves (solution, then 9 alts)

The npz stores RAW features; the loader pools + z-scores them across boards at train time, so
the cache is reusable across tid subsets. The tid set is the activation id cache for the size
(activations/_signal_cache/correct_ids_<size>.json) — exactly the problems train_rating_v2 can
score. Run once per size (both cached by default):

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

from auxiliary.inspect_activation_signals import CACHE_DIR, _cache_ids_path  # noqa: E402
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
    """Worker: feature vectors for a chunk of tids (module scope so it pickles). A problem that
    fails to parse / has no comments stays a NaN row; the parent drops it."""
    size, tids = task
    tdir = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
    out = np.full((len(tids), N_FEATS), np.nan, dtype=np.float32)
    for i, tid in enumerate(tids):
        try:
            with open(os.path.join(tdir, f"{tid}.sgf"), "rb") as fh:
                f = extract_move_features(fh.read())
            if f is not None:
                out[i] = f
        except Exception:
            pass
    return tids, out


def build_size(size, workers, force):
    out_path = _cache_path(size)
    if os.path.exists(out_path) and not force:
        print(f"size {size}: cache exists ({os.path.basename(out_path)}); --force to rebuild")
        return
    ids_path = _cache_ids_path(size)
    if not os.path.exists(ids_path):
        print(f"size {size}: no activation id cache ({ids_path}); skipping")
        return
    tids = json.load(open(ids_path))
    idx = {t: i for i, t in enumerate(tids)}
    feats = np.full((len(tids), N_FEATS), np.nan, dtype=np.float32)
    chunks = [(size, tids[i:i + CHUNK]) for i in range(0, len(tids), CHUNK)]
    print(f"size {size}: {len(tids)} problems, {len(chunks)} chunks, {workers} workers")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_chunk, c) for c in chunks]
        for k, fu in enumerate(as_completed(futs)):
            ctids, cout = fu.result()
            for t, row in zip(ctids, cout):
                feats[idx[t]] = row
            done += len(ctids)
            if (k + 1) % 50 == 0 or done == len(tids):
                print(f"  {done}/{len(tids)} ({done / max(time.time() - t0, 1e-9):.0f}/s)")
    good = np.isfinite(feats).all(axis=1)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{out_path}.tmp{os.getpid()}.npz"
    np.savez(tmp, tids=np.array([t for t, g in zip(tids, good) if g]), feats=feats[good])
    os.replace(tmp, out_path)
    print(f"size {size}: wrote {int(good.sum())} rows to {os.path.basename(out_path)} "
          f"({int((~good).sum())} skipped) in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="9,13", help="comma-separated board sizes to cache")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = ap.parse_args()
    for s in [int(x) for x in args.sizes.split(",") if x.strip()]:
        build_size(s, max(1, args.workers), args.force)


if __name__ == "__main__":
    main()
