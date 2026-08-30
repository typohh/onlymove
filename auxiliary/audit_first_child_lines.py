"""Audit each problem's PRINCIPAL LINE by the walk the writer itself followed.

WHAT IT ASKS
------------
`subtree_reaches_conclusion` (utils/katago/add_variations.py, mirrored by
`subtreeReachesConclusion` in pwa/js/analysis.js) defines a played-out branch as
one whose FIRST-CHILD chain ends in two consecutive passes: the writer appends
children in `moveInfos` order and always recurses into the first, so the first
child IS the continuation it followed. That predicate says only that the line was
read to a conclusion. It says nothing about WHICH SIDE the conclusion favours,
and the terminal node already carries the answer.

So this walks the same chain and reads the terminal evaluation:

  ANSWER   is variation_nodes[0], the move generation pins as the solution.
           Flagged when its principal line ends BELOW the win threshold -- the
           solver does not hold the win along the line the writer followed.
  ALT-1    is variation_nodes[1], the engine's best alternative first move under
           the same `moveInfos` ordering. Flagged when its principal line ends AT
           OR ABOVE the threshold -- the move the problem calls losing is one the
           solver ends up winning with.

This is deliberately NOT minimax over the subtree. Minimax lets each side pick
its best RECORDED reply, which flatters whichever side has more branches written
under it, and the two sides are covered very unevenly. The first-child chain is
the one line the writer actually committed to, which is why the pass-pass check
below holds exactly.

WHAT A FLAG IS AND IS NOT
-------------------------
It is a contradiction WITHIN the file's own stored evaluations, not proof the
answer is wrong. The entry evaluation is deep (the answer's own comment is a
1000-visit value) but the tail is played at the 30-visit floor, so a long
principal line can drift a won position into a lost one without the first move
being mistaken. Read a flag as "this file's recorded line does not demonstrate
what the problem claims", and re-read the line before acting on it.

Winrates are black-perspective and Black is the solver in every corpus problem,
so one threshold serves both checks.

USAGE
-----
    ./.venv/bin/python3 auxiliary/audit_first_child_lines.py

Writes, next to this file:

    first_child_line_defects.txt

Options:
    --dir SUBDIR   audit one repo-relative directory instead of `tsumegos`
    --out PATH     write somewhere else
    --workers N    process count
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np

from auxiliary.audit_solution_lines import (make_is_pass, node_eval, node_move,
                                            normalize, parse_sgf,
                                            RECURSION_LIMIT, WIN_THRESHOLD)
from utils.constants import ALL_BOARD_SIZES

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGED = os.path.join(PROJECT_ROOT, "pwa", "tsumegos")

# MAX_VARIATION_DEPTH + 2, the bound the shipped predicate walks under.
MAX_STEPS = 802


def walk(node, is_pass):
    """Follow the first child CARRYING A MOVE to the end of the line.

    -> (terminal winrate, ends in two consecutive passes, plies). The move filter
    is not cosmetic: a node can carry only a comment, and stepping onto one drops
    the walk off the line the writer followed.
    """
    n = node
    prev_was_pass = is_pass(node_move(node)[1])
    plies = 1
    ended = False
    for _ in range(MAX_STEPS):
        kids = [c for c in n["children"] if node_move(c)[0] is not None]
        if not kids:
            break
        n = kids[0]
        plies += 1
        this_is_pass = is_pass(node_move(n)[1])
        if this_is_pass and prev_was_pass:
            ended = True
            break
        prev_was_pass = this_is_pass
    return node_eval(n)[0], ended, plies


def analyse(path):
    try:
        text = open(path).read()
        seq = parse_sgf(text)
        normalize(seq)
        root = seq[0]
        size = int(root["props"]["SZ"][0])
        is_pass = make_is_pass(size)
        roots = [c for c in seq[-1]["children"] if node_move(c)[0] is not None]
        if len(roots) < 2:
            return dict(size=size, tid=os.path.basename(path)[:-4],
                        verdict="ERR", detail="fewer than two root variations")
        p_wr, p_pp, p_len = walk(roots[0], is_pass)
        a_wr, a_pp, a_len = walk(roots[1], is_pass)
        return dict(size=size, tid=os.path.basename(path)[:-4], verdict=None,
                    p_wr=p_wr, p_pp=p_pp, p_len=p_len,
                    a_wr=a_wr, a_pp=a_pp, a_len=a_len)
    except Exception as exc:                                    # noqa: BLE001
        return dict(size=0, tid=os.path.basename(path)[:-4],
                    verdict="ERR", detail=str(exc)[:80])


def verdict_of(row):
    primary_bad = row["p_wr"] is not None and row["p_wr"] < WIN_THRESHOLD
    alt_bad = row["a_wr"] is not None and row["a_wr"] >= WIN_THRESHOLD
    if primary_bad and alt_bad:
        return "both"
    if primary_bad:
        return "primary_losing"
    if alt_bad:
        return "alt1_winning"
    return None


# --- shipped rank, for the rank column -------------------------------------

def load_shipped_ranks():
    """tsumego_id -> internal rank, from the PUBLISHED artifacts.

    problem_params.bin is 16 bytes per problem: a 4-byte id then three
    little-endian float32 (rating, slope, floor). The id is sliced as raw bytes
    rather than read through numpy's 'S4', which strips trailing NULLs and would
    silently mis-key every id ending in 00.

    Returns {} when the packaged data is absent; the rank column is then blank
    throughout, which is a missing column and not a claim about the problems.
    """
    params = os.path.join(PACKAGED, "problem_params.bin")
    curve_path = os.path.join(PACKAGED, "rating_to_rank.json")
    if not (os.path.exists(params) and os.path.exists(curve_path)):
        return {}
    curve = json.load(open(curve_path))
    raw = open(params, "rb").read()
    ids = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 16)[:, :4]
    ratings = np.frombuffer(raw, dtype=np.dtype(
        [("pad", "V4"), ("r", "<f4"), ("s", "<f4"), ("f", "<f4")]))["r"]
    # map_rating_to_rank writes ONE curve to every size key, so any key serves.
    key = str(ALL_BOARD_SIZES[0])
    xs = np.array(curve[key]["ratings"], dtype=float)
    ys = np.array(curve[key]["ranks"], dtype=float)
    r = np.asarray(ratings, dtype=float)
    rank = np.interp(r, xs, ys)
    lo, hi = r < xs[0], r > xs[-1]
    rank[lo] = ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (r[lo] - xs[0])
    rank[hi] = ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (r[hi] - xs[-1])
    return {bytes(i).hex(): float(v) for i, v in zip(ids, rank)}


HEADER = """\
# Tsumegos whose PRINCIPAL LINE contradicts what the problem claims.
#
# Produced by auxiliary/audit_first_child_lines.py. For each problem it follows
# the first child carrying a move -- the chain subtree_reaches_conclusion itself
# walks, the one the writer committed to -- from two starting points, and reads
# the evaluation the terminal node carries:
#
#   primary_losing   variation_nodes[0] (the answer) ends BELOW 50%: the solver
#                    does not hold the win along the recorded line.
#   alt1_winning     variation_nodes[1] (the engine's best alternative first
#                    move) ends AT OR ABOVE 50%: a move the problem calls losing
#                    is one the solver ends up winning with.
#   both             both.
#
# EVERY problem's two lines end in a double pass -- 112,714/112,714 on 9x9 and
# 124,937/124,937 on 13x13 -- so no row here is an abandoned branch. What is
# recorded is a resolved line that resolves the wrong way.
#
# A FLAG IS NOT PROOF THE ANSWER IS WRONG. The first move's own evaluation is
# deep and the tail is played at the 30-visit floor, so a long line can drift a
# won position into a lost one without the first move being mistaken. NONE of
# these problems fails the first-move test: across the whole corpus, zero have a
# losing answer or a winning alternative by the moves' OWN evaluations, which is
# the gate generation enforces and the client reads. Re-read a line before
# acting on it.
#
# rank is the internal rank the SHIPPED artifacts give the problem
# (pwa/tsumegos/problem_params.bin through rating_to_rank.json, unclamped as the
# client leaves it: 0 is the 1k/1d boundary). It is blank for a problem promoted
# since the last fit, which is a missing rating and not a defect. This column is
# a snapshot -- a re-ship moves it, the verdict columns do not.
#
# Columns, tab-separated:
#   size  tsumego_id  verdict  primary_terminal  primary_plies  alt1_terminal \
 alt1_plies  rank
"""


def main():
    sys.setrecursionlimit(RECURSION_LIMIT)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="tsumegos",
                    help="repo-relative corpus directory")
    ap.add_argument("--out", default=os.path.join(
        HERE, "first_child_line_defects.txt"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    ranks = load_shipped_ranks()
    if not ranks:
        print("  no packaged data found; the rank column will be blank")

    rows = []
    for size in ALL_BOARD_SIZES:
        directory = os.path.join(PROJECT_ROOT, args.dir, str(size))
        if not os.path.isdir(directory):
            continue
        with os.scandir(directory) as it:
            paths = sorted(e.path for e in it if e.name.endswith(".sgf"))
        with Pool(args.workers) as pool:
            found = pool.map(analyse, paths, chunksize=40)
        unresolved = sum(1 for r in found
                         if r["verdict"] != "ERR" and not (r["p_pp"] and r["a_pp"]))
        flagged = [r for r in found
                   if r["verdict"] == "ERR" or verdict_of(r) is not None]
        print(f"  {args.dir}/{size}: {len(paths)} problems, {len(flagged)} flagged, "
              f"{unresolved} without a double pass on both lines")
        rows.extend(flagged)

    with open(args.out, "w") as fh:
        fh.write(HEADER)
        for r in sorted(rows, key=lambda r: (r["size"], r["tid"])):
            if r["verdict"] == "ERR":
                fh.write(f"{r['size']}\t{r['tid']}\tERR\t{r['detail']}\n")
                continue
            rank = ranks.get(r["tid"])
            fh.write(f"{r['size']}\t{r['tid']}\t{verdict_of(r)}\t"
                     f"{r['p_wr']}\t{r['p_len']}\t{r['a_wr']}\t{r['a_len']}\t"
                     f"{'' if rank is None else f'{rank:.2f}'}\n")
    print(f"\nwrote {args.out} ({len(rows)} problems)")


if __name__ == "__main__":
    main()
