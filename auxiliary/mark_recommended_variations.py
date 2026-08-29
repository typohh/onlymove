"""RETIRED -- the `TE[1]` marker this wrote is gone. Only `--unmark` still runs.

WHY IT IS RETIRED
-----------------
The mark existed to stop the client DEDUCING which continuations were played
out, back when its test was a single pass anywhere in the subtree -- a test that
could offer a line resolving to the solver losing. Both the deduction and the
mark are now replaced by an exact structural test: walk first children and ask
whether the chain ends in TWO CONSECUTIVE PASSES
(utils/katago/add_variations.subtree_reaches_conclusion, mirrored by
subtreeReachesConclusion in pwa/js/analysis.js). The writer creates the second
pass only where its recursion reached the terminal, so the mark recorded nothing
the file did not already say, and the corpus carries no `TE` any more.

The marking path below therefore REFUSES to run: re-introducing marks would put
a second, divergent source of truth beside the structure. `--unmark --apply` is
kept, to strip marks from any file still holding them (tsumegos/broken/).

The historical rationale for what got marked is kept below, because
auxiliary/apply_solution_line_surgery.py still describes its inputs in these
terms.

WHAT GETS MARKED
----------------
At every node, for the side to move, each evaluated child is valued by minimax
over the branches that reach a DOUBLE PASS (see auxiliary/audit_solution_lines.py
for that machinery). A child is CLASS A when it has such a value AND that value
is the correct outcome for the first move whose subtree it sits in:

    under the answer        -> black winning
    under a refuted move    -> black losing

ONLY CLASS A IS EVER MARKED. Where no child qualifies, the node gets no mark at
all and the client falls back. There is therefore no way for a mark to lead to
a wrong result, and no priority rule between classes to get wrong.

Among class A, children are ranked lexicographically:

  1. INCONSISTENCY -- along the minimax principal variation from that child to
     its double-pass terminal, the number of nodes whose winrate predicts the
     wrong winner. Counts every node on the path regardless of colour, the
     child itself included. Fewer is better, so the marked line is the one that
     does not dip and contradict its own result midway.
  2. The move's own winrate -- highest for black, lowest for white.

Black gets one mark per node. White gets up to two: the winrate pick and the
score pick, deduped by coordinate -- so the two ghost stones the client already
draws are preserved, and appear exactly when those two differ.

THE ROOT IS AN EXCEPTION. At the decision node only the ANSWER may be marked,
and only if it is class A. Class A there also contains every refuted move that
correctly loses, and ranking by winrate among them would tag a losing first move
as the move to find whenever the answer itself fails to resolve correctly.

The edit is an insertion of `TE[1]` after the node's `;`, leaving every other
byte untouched; property order within a node is not significant in SGF.

USAGE
-----
    ./.venv/bin/python3 auxiliary/mark_recommended_variations.py            # dry run
    ./.venv/bin/python3 auxiliary/mark_recommended_variations.py --apply
    ./.venv/bin/python3 auxiliary/mark_recommended_variations.py --unmark --apply

`tsumegos/` is tracked in git, so `git checkout -- tsumegos/` reverts it.
`tsumegos/broken/` is NOT tracked -- back it up before applying there.
"""
import argparse
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.audit_solution_lines import (           # noqa: E402
    WIN_THRESHOLD, MAX_DEPTH, RECURSION_LIMIT,
    parse_sgf, normalize, node_move, node_eval, has_eval, make_is_pass,
    pick_answer)

MARK = 'TE[1]'
MARK_RE = re.compile(r'TE\[[^\]]*\]')


def evaluate(node, is_pass, prev_was_pass, expect_black_win, depth):
    """Store (value, inconsistency) on every node of this subtree, post-order.

    value          minimax over double-pass-resolved branches, or None
    inconsistency  nodes on the principal variation from here whose winrate
                   predicts the wrong winner, this node included

    Among children achieving the optimal value, the one with the fewer
    inconsistencies is taken, so the principal variation is deterministic and
    is the cleanest realisation of that value.
    """
    winrate, _ = node_eval(node)
    wrong = 0
    if winrate is not None:
        predicts_black = winrate >= WIN_THRESHOLD
        wrong = 1 if predicts_black != expect_black_win else 0

    _, coord = node_move(node)
    this_is_pass = is_pass(coord)

    if depth > MAX_DEPTH:
        node['_val'], node['_inc'] = None, None
        return None, None

    if this_is_pass and prev_was_pass:
        node['_val'] = winrate
        node['_inc'] = wrong if winrate is not None else None
        return node['_val'], node['_inc']

    colour, _ = node_move(node)
    black_to_play = (colour == 'W')

    best_val, best_inc = None, None
    for child in node['children']:
        if not has_eval(child):
            continue
        val, inc = evaluate(child, is_pass, this_is_pass, expect_black_win,
                            depth + 1)
        if val is None:
            continue
        if best_val is None:
            best_val, best_inc = val, inc
        else:
            better = (val > best_val) if black_to_play else (val < best_val)
            if better or (val == best_val and inc < best_inc):
                best_val, best_inc = val, inc

    node['_val'] = best_val
    node['_inc'] = None if best_inc is None else best_inc + wrong
    return node['_val'], node['_inc']


def class_a(children, expect_black_win):
    """Children that resolve, to the outcome this subtree is supposed to have."""
    out = []
    for child in children:
        if not has_eval(child):
            continue
        val = child.get('_val')
        if val is None:
            continue
        if (val >= WIN_THRESHOLD) == expect_black_win:
            out.append(child)
    return out


def choose(candidates, black_to_play):
    """-> the nodes to mark: one for black, up to two for white.

    Lexicographic: inconsistency first, then the move's own evaluation --
    highest winrate for black, lowest winrate / lowest score for white.
    """
    if not candidates:
        return []
    if black_to_play:
        best = min(candidates,
                   key=lambda c: (c['_inc'], -(node_eval(c)[0] or -1e9)))
        return [best]

    by_winrate = min(candidates,
                     key=lambda c: (c['_inc'], node_eval(c)[0]
                                    if node_eval(c)[0] is not None else 1e9))
    by_score = min(candidates,
                   key=lambda c: (c['_inc'], node_eval(c)[1]
                                  if node_eval(c)[1] is not None else 1e9))
    picks, seen = [], set()
    for child in (by_winrate, by_score):
        _, coord = node_move(child)
        if coord in seen:
            continue
        seen.add(coord)
        picks.append(child)
    return picks


def mark_subtree(node, expect_black_win, marks):
    """Walk every evaluated node, recording which children to mark.

    Descends into ALL evaluated children, not just the marked ones: a user can
    explore any branch, and each node they can reach needs its own marks.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        children = [c for c in current['children'] if has_eval(c)]
        if not children:
            continue
        colour, _ = node_move(current)
        black_to_play = (colour == 'W')
        for pick in choose(class_a(children, expect_black_win), black_to_play):
            marks.add(pick['pos'])
        stack.extend(children)


def plan(path):
    """-> (path, sorted offsets to mark, stats) or None."""
    try:
        text = open(path).read()
        size = int(re.search(r'SZ\[(\d+)\]', text).group(1))
        is_pass = make_is_pass(size)
        seq = parse_sgf(text)
        normalize(seq)
        roots = [c for c in seq[-1]['children'] if has_eval(c)]
        if not roots:
            return None
        answer = pick_answer(roots)

        marks = set()
        answer_marked = False
        for root in roots:
            expect_black_win = (root is answer)
            evaluate(root, is_pass, False, expect_black_win, 1)
            # Root exception: only the answer is eligible for a mark here.
            if root is answer:
                val = root.get('_val')
                if val is not None and val >= WIN_THRESHOLD:
                    marks.add(root['pos'])
                    answer_marked = True
            mark_subtree(root, expect_black_win, marks)

        return (path, sorted(marks), {
            'size': size,
            'answer_marked': answer_marked,
            'n_marks': len(marks),
            'had_marks': bool(MARK_RE.search(text)),
        })
    except Exception as exc:                                    # noqa: BLE001
        return (path, None, {'size': 'ERR', 'error': str(exc)[:80],
                             'answer_marked': False, 'n_marks': 0,
                             'had_marks': False})


def apply_marks(job):
    path, offsets, stats = job
    if offsets is None:
        return stats
    text = open(path).read()
    if stats['had_marks']:
        text = MARK_RE.sub('', text)
        # offsets were computed against the marked text; re-plan on the clean one
        tmp = path + '.__clean__'
        with open(tmp, 'w') as fh:
            fh.write(text)
        _, offsets, stats2 = plan(tmp)
        os.unlink(tmp)
        stats['n_marks'] = stats2['n_marks']
        stats['answer_marked'] = stats2['answer_marked']
        if offsets is None:
            return stats
    out = []
    prev = 0
    for off in offsets:
        out.append(text[prev:off + 1])      # through the ';'
        out.append(MARK)
        prev = off + 1
    out.append(text[prev:])
    with open(path, 'w') as fh:
        fh.write(''.join(out))
    return stats


def strip_marks(path):
    text = open(path).read()
    if 'TE[' not in text:
        return 0
    with open(path, 'w') as fh:
        fh.write(MARK_RE.sub('', text))
    return 1


def gather(subdirs):
    paths = []
    for subdir in subdirs:
        for size in (9, 13):
            directory = os.path.join(PROJECT_ROOT, subdir, str(size))
            if not os.path.isdir(directory):
                continue
            with os.scandir(directory) as it:
                paths.extend(sorted(e.path for e in it
                                    if e.name.endswith('.sgf')))
    return paths


def main():
    sys.setrecursionlimit(RECURSION_LIMIT)
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='write the files (default is a dry run)')
    ap.add_argument('--unmark', action='store_true',
                    help='remove every TE property (the only mode still live)')
    ap.add_argument('--dirs', nargs='+', default=['tsumegos', 'tsumegos/broken'])
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    if not args.unmark:
        raise SystemExit(
            "RETIRED: the TE[1] marker is no longer part of the corpus, and "
            "writing it back would stand a second source of truth beside the "
            "structural test that replaced it (subtree_reaches_conclusion, "
            "utils/katago/add_variations.py). Only --unmark --apply still runs.")

    paths = gather(args.dirs)
    print(f"{len(paths)} SGFs from {', '.join(args.dirs)}", flush=True)

    if args.unmark:
        if not args.apply:
            print("dry run: --unmark needs --apply")
            return
        with Pool(args.workers) as pool:
            changed = sum(pool.map(strip_marks, paths, chunksize=200))
        print(f"stripped TE from {changed} files")
        return

    with Pool(args.workers) as pool:
        jobs = [j for j in pool.map(plan, paths, chunksize=40) if j]

    per_size = {}
    errors = 0
    for _, offsets, st in jobs:
        if st['size'] == 'ERR':
            errors += 1
            continue
        acc = per_size.setdefault(st['size'], Counter())
        acc['files'] += 1
        acc['marks'] += st['n_marks']
        acc['answer_marked'] += 1 if st['answer_marked'] else 0
        acc['no_marks'] += 1 if st['n_marks'] == 0 else 0
        acc['already'] += 1 if st['had_marks'] else 0

    for size, acc in sorted(per_size.items()):
        print(f"\n  {size}x{size}: {acc['files']} files")
        print(f"     marks to write          : {acc['marks']}")
        print(f"     answer marked at root   : {acc['answer_marked']} "
              f"({100 * acc['answer_marked'] / acc['files']:.2f}%)")
        print(f"     files with NO mark at all: {acc['no_marks']}")
        if acc['already']:
            print(f"     files already carrying TE: {acc['already']} (re-marked)")
    if errors:
        print(f"\n  {errors} files failed to plan")

    if not args.apply:
        print("\ndry run -- nothing written. Re-run with --apply.")
        return

    with Pool(args.workers) as pool:
        pool.map(apply_marks, jobs, chunksize=40)
    print(f"\nwrote {len(jobs)} files")


if __name__ == '__main__':
    main()
