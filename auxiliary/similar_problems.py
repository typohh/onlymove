"""Find each sampled problem's nearest partner in the whole corpus, three ways.

    ./.venv/bin/python3 -m auxiliary.similar_problems [--n 100] [--seed 0]

Queries are a random sample of N problems (pooled over both sizes); partners are
searched over EVERY problem of the query's size (sizes never match each other).

  stones  — the position before the answer, encoded relative to the side to
            move (+1 own, -1 opponent), compared under all 8 D4 transforms of
            the candidate. Score = matching stones / stones in the union (a
            colour-aware Jaccard), so empty-board agreement counts for nothing.
  katago  — cosine between the activation network's trunk state at the ANSWER
            cell (`correct_vecs_<size>.npy`, standardized per size). The trunk
            is (near-)equivariant, so this needs no orientation search.
  combined— each score z-scored over the whole corpus for that query, then
            averaged: a partner has to be unusual on BOTH to win.

For each criterion it prints every query's partner, then the single most
similar pair in the sample as onlymove links. The partner's link carries the
`o=` orientation that lays it over the query (the query is shown at o=0).

The board states are replayed from the SGFs once and cached beside the other
signal caches (`activations/_signal_cache/board_states_<size>.npz`).
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import tsumego_dir  # noqa: E402
from utils.katago.board_render import parse_sgf, simulate  # noqa: E402

CACHE_DIR = os.path.join(PROJECT_ROOT, "activations", "_signal_cache")
SITE = "https://onlymove.net"


def _transform(x, y, t, size):
    """`transformCoordinates` (pwa/js/sgf.js): the client's orientation t."""
    n = size - 1
    return [(x, y), (y, n - x), (n - x, n - y), (n - y, x),
            (x, n - y), (n - x, y), (y, x), (n - y, n - x)][t]


def _perms(size):
    """perm[t][c] = cell that canonical cell c lands on when shown at orientation t."""
    out = np.empty((8, size * size), dtype=np.int64)
    for t in range(8):
        for y in range(size):
            for x in range(size):
                tx, ty = _transform(x, y, t, size)
                out[t, y * size + x] = ty * size + tx
    return out


def _board_state(args):
    tid, size = args
    with open(os.path.join(tsumego_dir(size), f"{tid}.sgf")) as f:
        sz, setup, correct = parse_sgf(f.read())
    if sz != size or correct is None:
        raise ValueError(f"{tid}: size {sz} / answer {correct}")
    board, _, _ = simulate(setup, size)
    mover = correct[0]
    v = np.zeros(size * size, dtype=np.int8)
    for y in range(size):
        for x in range(size):
            if board[y][x] != ".":
                v[y * size + x] = 1 if board[y][x] == mover else -1
    ax, ay = ord(correct[1][0]) - ord("a"), ord(correct[1][1]) - ord("a")
    return v, ay * size + ax


def load_boards(size, ids):
    path = os.path.join(CACHE_DIR, f"board_states_{size}.npz")
    if os.path.exists(path):
        d = np.load(path)
        if list(d["ids"]) == ids and "answers" in d:
            return d["boards"], d["answers"]
        print(f"  board cache {path} is stale — rebuilding")
    print(f"  replaying {len(ids)} {size}x{size} SGFs ...", flush=True)
    with Pool() as pool:
        states = pool.map(_board_state, [(t, size) for t in ids], chunksize=256)
    boards = np.stack([v for v, _ in states])
    answers = np.array([a for _, a in states], dtype=np.int64)
    np.savez(path, ids=np.array(ids), boards=boards, answers=answers)
    return boards, answers


def stone_scores(q, boards, perms, q_ans=None, answers=None):
    """Best colour-aware Jaccard of query board q against every candidate, over
    the 8 orientations of the candidate. Returns (score[N], best_t[N]) where
    best_t is the orientation t at which the candidate matches q. With q_ans and
    answers given, an orientation that puts the candidate's answer on the
    query's answer cell is not allowed to count."""
    own = (boards == 1).astype(np.float32)
    opp = (boards == -1).astype(np.float32)
    n_c = own.sum(1) + opp.sum(1)
    n_q = float((q != 0).sum())
    best = np.full(len(boards), -1.0, dtype=np.float32)
    best_t = np.zeros(len(boards), dtype=np.int64)
    for t in range(8):
        # Candidate shown at orientation t: its canonical cell c sits at perm[t][c],
        # so the query cell it is compared with is q[perm[t][c]].
        qt = q[perms[t]]
        match = own @ (qt == 1).astype(np.float32) + opp @ (qt == -1).astype(np.float32)
        s = match / np.maximum(n_c + n_q - match, 1.0)
        if q_ans is not None:
            s[perms[t][answers] == q_ans] = -1.0
        better = s > best
        best[better], best_t[better] = s[better], t
    return best, best_t


def zscore(s, self_idx):
    m = np.ones(len(s), bool)
    m[self_idx] = False
    return (s - s[m].mean()) / (s[m].std() + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--different-answer", action="store_true",
                    help="stones: a partner only counts at an orientation where its "
                         "answer lands on a DIFFERENT cell from the query's")
    args = ap.parse_args()

    data = {}
    for size in (9, 13):
        ids = json.load(open(os.path.join(CACHE_DIR, f"correct_ids_{size}.json")))
        vecs = np.load(os.path.join(CACHE_DIR, f"correct_vecs_{size}.npy")).astype(np.float32)
        if len(vecs) != len(ids):
            raise SystemExit(f"correct_vecs_{size} / correct_ids_{size} disagree in length")
        vecs = (vecs - vecs.mean(0)) / (vecs.std(0) + 1e-6)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        boards, answers = load_boards(size, ids)
        data[size] = dict(ids=ids, vecs=vecs, boards=boards, answers=answers, perms=_perms(size))

    pool = [(s, i) for s in (9, 13) for i in range(len(data[s]["ids"]))]
    rng = np.random.default_rng(args.seed)
    sample = [pool[k] for k in rng.choice(len(pool), args.n, replace=False)]

    rows = {"stones": [], "katago": [], "combined": []}
    for size, qi in sample:
        d = data[size]
        st, st_t = stone_scores(
            d["boards"][qi].astype(np.int64), d["boards"], d["perms"],
            *((d["answers"][qi], d["answers"]) if args.different_answer else ()))
        kg = d["vecs"] @ d["vecs"][qi]
        cb = (zscore(st, qi) + zscore(kg, qi)) / 2
        for name, s in (("stones", st), ("katago", kg), ("combined", cb)):
            s = s.copy()
            s[qi] = -np.inf
            pj = int(np.argmax(s))
            rows[name].append(dict(size=size, q=d["ids"][qi], p=d["ids"][pj], score=float(s[pj]),
                                   stones=float(st[pj]), katago=float(kg[pj]), o=int(st_t[pj])))

    for name, rs in rows.items():
        print(f"\n=== {name} ===")
        print(f"{'size':>4} {'query':>8} {'partner':>8} {'score':>7} {'stones':>7} {'katago':>7}")
        for r in sorted(rs, key=lambda r: -r["score"]):
            print(f"{r['size']:>4} {r['q']:>8} {r['p']:>8} {r['score']:7.3f} {r['stones']:7.3f} {r['katago']:7.3f}")
    print()
    for name, rs in rows.items():
        r = max(rs, key=lambda r: r["score"])
        # Explicit o=0 on the query too: a bare id in the viewer's recall window
        # is served at a random orientation.
        print(f"{name:>8}: {SITE}/{r['size']}?id={r['q']}&o=0  <->  {SITE}/{r['size']}?id={r['p']}&o={r['o']}"
              f"   (stones {r['stones']:.3f}, katago cos {r['katago']:.3f})")


if __name__ == "__main__":
    main()
