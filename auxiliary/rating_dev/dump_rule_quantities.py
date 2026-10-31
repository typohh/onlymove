#!/usr/bin/env python3
"""Dump the RAW per-problem quantities behind the geometry rules, so their
thresholds can be swept without reparsing the corpus per candidate.

rule_tenuki / rule_sente / rule_gote are each a threshold on one or two of the
distances measured here, and `pipeline.build_motif_proxies` bakes only the
verdicts at the shipped bar. A boundary sweep re-asks the same verdicts at
other bars, which needs the distances, not the booleans — one corpus parse
here, then every candidate bar is a numpy comparison.

    ./.venv/bin/python3 -m auxiliary.rating_dev.dump_rule_quantities \
        --out axes/rule_quantities.csv [--workers N]

Columns per (size, tid):
    opp_dist       answer to the opponent's last setup move (rule_tenuki input)
    root_ko        1 if a ko stands in the root position
    answer_kind    move|pass|none — the opponent's recorded reply to the answer
    answer_reply   its distance when answer_kind == move, else blank
    refuted_kind / refuted_reply — the same, for the best refuted try
Blank means "not measurable" — exactly the cases the rules decline on.
"""

import argparse
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import sgfmill.sgf  # noqa: E402

from utils.paths import tsumego_dir  # noqa: E402
from utils.katago.board_render import parse_sgf, simulate  # noqa: E402
from utils.katago.sgf_parsing import parse_tsumego_sgf  # noqa: E402
from utils.rating.model_constants import SIZES  # noqa: E402
from utils.rating.motif_rules import (  # noqa: E402
    answer_opponent_distance, best_refuted_node, reply_kind)

FIELDS = ["tid", "size", "opp_dist", "root_ko", "adj_stones",
          "answer_kind", "answer_reply", "refuted_kind", "refuted_reply"]


def one(args):
    size, tid = args
    path = os.path.join(tsumego_dir(size), f"{tid}.sgf")
    try:
        raw = open(path, "rb").read()
        game = sgfmill.sgf.Sgf_game.from_bytes(raw)
        _, variation_nodes = parse_tsumego_sgf(game)
        _, setup_sgf, correct = parse_sgf(raw.decode("utf-8", "replace"))
    except Exception:
        return None
    answer_node = None
    if correct:
        for n in variation_nodes:
            _c, r = n.get_raw_move()
            if r and r.decode("utf-8") == correct[1]:
                answer_node = n
                break
    ak, ad = reply_kind(answer_node)
    rk, rd = reply_kind(best_refuted_node(variation_nodes, correct))
    opp = answer_opponent_distance(setup_sgf, correct)
    board, _last, ko = simulate(setup_sgf, size)
    adj = ""
    if correct:
        x, y = ord(correct[1][0]) - 97, ord(correct[1][1]) - 97
        if 0 <= x < size and 0 <= y < size:
            adj = sum(1 for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
                      if 0 <= nx < size and 0 <= ny < size
                      and board[ny][nx] != ".")
    return {"tid": tid, "size": size,
            "opp_dist": "" if opp is None else opp,
            "root_ko": int(ko is not None),
            "adj_stones": adj,
            "answer_kind": ak or "none",
            "answer_reply": "" if ad is None else ad,
            "refuted_kind": rk or "none",
            "refuted_reply": "" if rd is None else rd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "axes",
                                                  "rule_quantities.csv"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    todo = []
    for size in SIZES:
        d = tsumego_dir(size)
        todo += [(size, f[:-4]) for f in os.listdir(d) if f.endswith(".sgf")]
    print(f"{len(todo)} problems, {args.workers} workers")

    t0 = time.time()
    from multiprocessing import Pool
    rows, skipped = [], 0
    with Pool(args.workers) as p:
        for i, r in enumerate(p.imap_unordered(one, todo, chunksize=200), 1):
            if r is None:
                skipped += 1
            else:
                rows.append(r)
            if i % 20000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    rows.sort(key=lambda r: (r["size"], r["tid"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows ({skipped} unparseable) -> {args.out} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
