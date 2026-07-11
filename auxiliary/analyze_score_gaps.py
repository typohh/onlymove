"""
Distribution of the score gap between the best and second-best first move.

For every generated tsumego we read the problem's decision node — the first
node on the main line that branches (its children are the candidate first
moves, i.e. the solution plus the shallow all-legal-move variations that
`add_shallow_variations` attaches). Each candidate carries a
"<winrate>%/<score>" comment from BLACK's perspective (bots/analysis.cfg sets
reportAnalysisWinratesAs = BLACK); we flip the score to the mover's perspective
(score_black for Black to move, -score_black for White), rank the candidates by
that mover score, and record

    gap = best_mover_score - second_best_mover_score

i.e. how many points the solution beats the best *wrong* first move by. A large
gap is a sharp, unambiguous tsumego; a gap near ~1 means the second-best move
is nearly as good.

The histogram uses fine 0.1-wide buckets in a window around 1 (so you can see
0.9-1.0 vs 1.0-1.1 etc.) and coarser buckets in the tail. Score gaps are
generally well above 1, so the fine region surfaces the borderline problems.

Usage:
    ./.venv/bin/python3 -m auxiliary.analyze_score_gaps                 # both sizes
    ./.venv/bin/python3 -m auxiliary.analyze_score_gaps --sizes 13
    ./.venv/bin/python3 -m auxiliary.analyze_score_gaps --fine-hi 4 --examples 30
"""

import argparse
import bisect
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sgfmill import sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generated trees are deep; sgfmill parses lazily but walks can recurse.
sys.setrecursionlimit(10000)

# "<winrate>%/<score>", both black-perspective (see module docstring).
_EVAL_RE = re.compile(r"(-?[\d.]+)%/(-?[\d.]+)")


def parse_eval_comment(comment):
    """(winrate_black, score_black) from a "{winrate}%/{score}" comment, else None."""
    if not comment:
        return None
    m = _EVAL_RE.search(comment)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def decision_node(game):
    """First main-line node that branches — its children are the candidate first moves.

    The setup is a linear chain (each node has exactly one child); the problem
    position is the first node with >1 children (matches `node` in
    utils.helper.construct_initial_sgf, whose children are answer + shallow
    variations). Returns None for a file that never branches.
    """
    node = game.get_root()
    while len(node) == 1:
        node = node[0]
    return node if len(node) > 1 else None


def score_gap(game):
    """best - second-best mover score among the candidate first moves.

    Returns (gap, n_candidates) or (None, n_candidates) if fewer than two
    candidates carry a parseable score.
    """
    node = decision_node(game)
    if node is None:
        return None, 0

    mover_scores = []
    black_to_move = None
    for child in node:
        colour, _ = child.get_move()
        if colour is None:  # property-only node
            continue
        ev = parse_eval_comment(child.get("C") if child.has_property("C") else None)
        if ev is None:
            continue
        _, score_black = ev
        if black_to_move is None:
            black_to_move = colour == "b"
        # All candidates share the side to move; flip to the mover's view so a
        # higher mover score is always the better move for whoever is to play.
        mover_scores.append(score_black if colour == "b" else -score_black)

    if len(mover_scores) < 2:
        return None, len(mover_scores)
    mover_scores.sort(reverse=True)
    return mover_scores[0] - mover_scores[1], len(mover_scores)


def process_file(path):
    tid = os.path.basename(path)[:-4]  # strip ".sgf"
    try:
        with open(path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception as e:  # noqa: BLE001 — report, don't crash the whole scan
        return tid, None, 0, f"{type(e).__name__}: {e}"
    gap, n = score_gap(game)
    return tid, gap, n, None


def build_edges(fine_lo, fine_hi, fine_width, coarse_hi, coarse_width):
    """Tiered bucket edges: fine buckets around the interesting region, coarse tail.

    Anything < fine_lo is underflow; anything >= coarse_hi is overflow.
    """
    edges = []
    e = fine_lo
    while e < fine_hi - 1e-9:
        edges.append(round(e, 6))
        e += fine_width
    e = fine_hi
    while e < coarse_hi - 1e-9:
        edges.append(round(e, 6))
        e += coarse_width
    edges.append(round(coarse_hi, 6))
    return edges


def print_histogram(gaps_by_size, sizes, edges):
    """Tiered histogram with per-size + total columns, percentages, cumulative, bar."""
    all_gaps = np.concatenate([gaps_by_size[s] for s in sizes]) if sizes else np.array([])
    total = len(all_gaps)
    if total == 0:
        print("  (no gaps)")
        return

    # bucket index: -1 underflow, len(edges)-1 overflow, else fine/coarse bucket.
    def bucketize(arr):
        counts = [0] * (len(edges) + 1)  # +1 for underflow at index 0 shift
        under = 0
        over = 0
        body = [0] * (len(edges) - 1)
        for g in arr:
            if g < edges[0]:
                under += 1
            elif g >= edges[-1]:
                over += 1
            else:
                body[bisect.bisect_right(edges, g) - 1] += 1
        return under, body, over

    per_size = {s: bucketize(gaps_by_size[s]) for s in sizes}
    t_under, t_body, t_over = bucketize(all_gaps)

    size_cols = " ".join(f"{('n' + str(s)):>8}" for s in sizes)
    print(f"  {'bucket':>14} {size_cols} {'total':>8} {'pct':>7} {'cum%':>7}  bar")
    max_body = max(t_body) if t_body else 1
    max_cell = max(max_body, t_under, t_over, 1)

    def row(label, idx_or_flag):
        if idx_or_flag == "under":
            t = t_under
            per = {s: per_size[s][0] for s in sizes}
        elif idx_or_flag == "over":
            t = t_over
            per = {s: per_size[s][2] for s in sizes}
        else:
            t = t_body[idx_or_flag]
            per = {s: per_size[s][1][idx_or_flag] for s in sizes}
        cells = " ".join(f"{per[s]:>8}" for s in sizes)
        bar = "#" * int(round(40 * t / max_cell)) if t else ""
        pct = 100 * t / total
        row.cum += t
        cum = 100 * row.cum / total
        print(f"  {label:>14} {cells} {t:>8} {pct:>6.2f}% {cum:>6.2f}%  {bar}")

    row.cum = 0
    if t_under or any(per_size[s][0] for s in sizes):
        row(f"< {edges[0]:g}", "under")
    for i in range(len(edges) - 1):
        row(f"{edges[i]:g}-{edges[i + 1]:g}", i)
    row(f">= {edges[-1]:g}", "over")
    print(f"  {'TOTAL':>14} {' '.join(f'{len(gaps_by_size[s]):>8}' for s in sizes)} {total:>8}")


def print_stats(gaps_by_size, sizes):
    all_gaps = np.concatenate([gaps_by_size[s] for s in sizes]) if sizes else np.array([])
    print(f"\n  {'':>6} {'n':>8} {'min':>7} {'p1':>7} {'p10':>7} {'median':>8} "
          f"{'mean':>7} {'p90':>7} {'p99':>7} {'max':>7}")
    for s in list(sizes) + (["all"] if len(sizes) > 1 else []):
        g = all_gaps if s == "all" else gaps_by_size[s]
        if len(g) == 0:
            continue
        q = np.percentile(g, [1, 10, 50, 90, 99])
        print(f"  {str(s):>6} {len(g):>8} {g.min():>7.2f} {q[0]:>7.2f} {q[1]:>7.2f} "
              f"{q[2]:>8.2f} {g.mean():>7.2f} {q[3]:>7.2f} {q[4]:>7.2f} {g.max():>7.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", type=int, default=[9, 13],
                    help="Board sizes to scan (default: 9 13)")
    ap.add_argument("--dir", default=None,
                    help="Override single directory to scan (implies one size)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only scan the first N files per size (quick sanity check)")
    ap.add_argument("--fine-lo", type=float, default=0.0)
    ap.add_argument("--fine-hi", type=float, default=3.0,
                    help="Upper edge of the fine 0.1-wide region (default: 3.0)")
    ap.add_argument("--fine-width", type=float, default=0.1)
    ap.add_argument("--coarse-hi", type=float, default=12.0,
                    help="Upper edge of the coarse region; >= this is overflow (default: 12)")
    ap.add_argument("--coarse-width", type=float, default=0.5)
    ap.add_argument("--examples", type=int, default=20,
                    help="List the N smallest-gap problems per size (borderline cases)")
    args = ap.parse_args()

    if args.dir:
        jobs = [(None, args.dir)]
    else:
        jobs = [(s, os.path.join(PROJECT_ROOT, "tsumegos", str(s))) for s in args.sizes]

    gaps_by_size = {}
    insufficient = {}
    errors = {}
    for size, target in jobs:
        key = size if size is not None else os.path.basename(target.rstrip("/"))
        if not os.path.isdir(target):
            print(f"Directory not found: {target}", file=sys.stderr)
            continue
        files = sorted(
            os.path.join(target, n) for n in os.listdir(target) if n.endswith(".sgf")
        )
        if args.limit:
            files = files[: args.limit]
        total = len(files)
        print(f"Scanning {total} SGFs in {target} with {args.workers} workers...")

        gaps = []
        small = []  # (gap, tid) for the examples list
        n_insufficient = 0
        errs = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_file, p) for p in files]
            step = max(1, total // 50)
            for i, fut in enumerate(as_completed(futures), 1):
                tid, gap, n, err = fut.result()
                if err is not None:
                    errs.append((tid, err))
                elif gap is None:
                    n_insufficient += 1
                else:
                    gaps.append(gap)
                    small.append((gap, tid))
                if i % step == 0 or i == total:
                    print(f"  {i}/{total} ({100 * i / total:.1f}%)", end="\r", flush=True)
        print()
        gaps_by_size[key] = np.array(gaps, dtype=float)
        insufficient[key] = n_insufficient
        errors[key] = errs
        small.sort()
        gaps_by_size.setdefault("_examples", {})[key] = small[: args.examples]

    example_bank = gaps_by_size.pop("_examples", {})
    sizes = [k for k in gaps_by_size]

    edges = build_edges(args.fine_lo, args.fine_hi, args.fine_width,
                        args.coarse_hi, args.coarse_width)

    print("\n=== Score gap (best - 2nd-best first move), point difference ===")
    for k in sizes:
        print(f"  size {k}: {len(gaps_by_size[k])} problems, "
              f"{insufficient.get(k, 0)} with <2 scored candidates, "
              f"{len(errors.get(k, []))} parse errors")

    print_stats(gaps_by_size, sizes)
    print()
    print_histogram(gaps_by_size, sizes, edges)

    if args.examples:
        print(f"\n=== {args.examples} smallest-gap problems per size (sharpest alternatives) ===")
        for k in sizes:
            print(f"  size {k}:")
            for gap, tid in example_bank.get(k, []):
                print(f"    {tid}  gap={gap:.2f}")

    for k in sizes:
        if errors.get(k):
            print(f"\nParse errors (size {k}, first 5):")
            for tid, err in errors[k][:5]:
                print(f"  {tid}: {err}")


if __name__ == "__main__":
    main()
