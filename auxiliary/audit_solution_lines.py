"""Audit every tsumego's recorded continuation by minimax over the played-out tree.

WHAT IT ASKS
------------
A problem asserts two things: the answer wins, and every other first move loses.
Both are checked against the SGF's own stored evaluations.

Only branches that reach a DOUBLE PASS are considered. A double pass is where
the generator played a line out to a conclusion; anything else is an evaluation
it started and abandoned, and carries no verdict either way. This replaces the
retired ply-count floor in `broken_review_paths.txt`, which could not tell a
resolved short line from an unresolved long one and therefore removed problems
on the strength of abandoned stubs.

Within that pruned tree the value is exact minimax on the Black-perspective
winrate stored in each node's comment:

    black to play    -> the child with the highest value
    white to play    -> the child with the lowest value
    double-pass leaf -> its own winrate
    anything else    -> no value; it is not part of the pruned tree

  ANSWER   is broken when its value is below WIN_THRESHOLD: inside the
           played-out data, black cannot hold the win against white's best.
  REFUTED  moves are broken when their value reaches WIN_THRESHOLD: the move
           the problem calls losing is one black can win with.

The root PASS variation is scored but held in its own column. It is not a move
the client can offer -- the per-cell embeddings sidecar has no row for a pass --
so letting it drive the refuted-move verdict would misreport the corpus.

The two sides are NOT symmetric in coverage. A refuted move only gets a value
if some line under it was played out, and the generator pursues the answer far
more often than it pursues refutations, so refuted-move counts are a floor
while answer counts are complete.

USAGE
-----
    ./.venv/bin/python3 auxiliary/audit_solution_lines.py

Audits `tsumegos/` and `tsumegos/broken/` and writes, next to this file:

    broken_solution_lines.txt   defects found in the live corpus
    quarantine_cleared.txt      quarantined problems this audit finds sound

Options:
    --dir SUBDIR      audit one directory instead (repo-relative)
    --out PATH        write the raw per-problem TSV here
    --sample N        N random problems per board size (0 = all)
    --workers N       process count
"""
import argparse
import os
import random
import re
import sys
from collections import Counter
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))

# A winrate at or above this counts as a win for black. Comments are
# Black-perspective throughout (bots/analysis.cfg sets
# reportAnalysisWinratesAs = BLACK), so one threshold serves both checks.
WIN_THRESHOLD = 50.0

# Guards against a pathological tree; neither is reached by the shipped corpus.
MAX_DEPTH = 600
RECURSION_LIMIT = 10000

WINRATE_RE = re.compile(r'([\d.]+)%/(-?[\d.]+)')


# --------------------------------------------------------------------------
# Minimal SGF reader. sgfmill would do, but it is an order slower on a corpus
# this size and this only needs moves plus the winrate comment.
# --------------------------------------------------------------------------

def parse_sgf(text):
    """Parse into a sequence of node dicts; the last holds the variations."""
    i, n = 0, len(text)

    def skip_ws():
        nonlocal i
        while i < n and text[i] in ' \t\r\n':
            i += 1

    def parse_node():
        nonlocal i
        props = {}
        while True:
            skip_ws()
            if i >= n or text[i] in ';()':
                break
            m = re.match(r'([A-Za-z]+)', text[i:])
            if not m:
                break
            key = m.group(1)
            i += len(key)
            vals = []
            while True:
                skip_ws()
                if i >= n or text[i] != '[':
                    break
                i += 1
                buf = []
                while i < n and text[i] != ']':
                    if text[i] == '\\':
                        i += 1
                        if i < n:
                            buf.append(text[i])
                            i += 1
                        continue
                    buf.append(text[i])
                    i += 1
                i += 1
                vals.append(''.join(buf))
            props[key] = vals
        return {'props': props, 'children': []}

    def parse_tree():
        nonlocal i
        i += 1                                  # consume '('
        seq = []
        while True:
            skip_ws()
            if i >= n:
                break
            if text[i] == ';':
                start = i               # offset of the ';', for in-place edits
                i += 1
                node = parse_node()
                node['pos'] = start
                seq.append(node)
            elif text[i] == '(':
                subs = []
                while True:
                    skip_ws()
                    if i < n and text[i] == '(':
                        subs.append(parse_tree())
                    else:
                        break
                seq[-1]['children'] = subs
                break
            elif text[i] == ')':
                i += 1
                return seq
            else:
                i += 1
        skip_ws()
        if i < n and text[i] == ')':
            i += 1
        return seq

    skip_ws()
    return parse_tree()


def normalize(seq):
    """Chain a parsed sequence so every node's `children` holds NODES.

    `(;A;B(;C)(;D))` parses as [A, B] with B carrying two sub-sequences; after
    this, A.children == [B] and B.children == [C, D].
    """
    last = seq[-1]
    last['children'] = [normalize(s) for s in last['children']]
    for i in range(len(seq) - 2, -1, -1):
        seq[i]['children'] = [seq[i + 1]]
    return seq[0]


def node_move(nd):
    props = nd['props']
    for colour in ('B', 'W'):
        if colour in props:
            return colour, props[colour][0]
    return None, None


def node_eval(nd):
    """-> (winrate, score) from the node's comment, or (None, None)."""
    comment = nd['props'].get('C', [None])[0]
    if not comment:
        return None, None
    m = WINRATE_RE.search(comment)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def has_eval(nd):
    """A node the engine evaluated: a move plus a winrate comment.

    Nodes without one are user-explored or structural and are not candidates.
    """
    _, coord = node_move(nd)
    comment = nd['props'].get('C', [None])[0]
    return coord is not None and bool(comment) and '%/' in comment


def make_is_pass(board_size):
    """SGF passes are `[]` or a coordinate off the board (`tt` on 9x9/13x13)."""
    def is_pass(coord):
        if coord is None:
            return False
        if coord in ('', 'tt'):
            return True
        return not (len(coord) == 2
                    and all(ord('a') <= ord(c) < ord('a') + board_size
                            for c in coord))
    return is_pass


# --------------------------------------------------------------------------
# Minimax over the played-out tree
# --------------------------------------------------------------------------

def minimax(node, is_pass, prev_was_pass, depth):
    """-> (value, depth_of_the_leaf_it_came_from), or (None, None).

    None means no line under this node reaches a double pass, so the node is
    outside the pruned tree and contributes nothing to either side's choice.
    """
    if depth > MAX_DEPTH:
        return None, None
    _, coord = node_move(node)
    this_is_pass = is_pass(coord)
    if this_is_pass and prev_was_pass:
        winrate, _ = node_eval(node)
        return (winrate, depth) if winrate is not None else (None, None)

    children = [c for c in node['children'] if has_eval(c)]
    if not children:
        return None, None

    # `node` was just played, so the other colour moves next.
    colour, _ = node_move(node)
    black_to_play = (colour == 'W')

    best_value, best_depth = None, None
    for child in children:
        value, leaf_depth = minimax(child, is_pass, this_is_pass, depth + 1)
        if value is None:
            continue
        if (best_value is None
                or (black_to_play and value > best_value)
                or (not black_to_play and value < best_value)):
            best_value, best_depth = value, leaf_depth
    return best_value, best_depth


def pick_answer(roots):
    """The answer as the EVALUATIONS name it: highest winrate, score breaks ties.

    Deliberately not the corpus contract (variation 0, which is what the client
    and every pipeline stage read). This is an audit: picking the other way is
    what lets it report where a shipped file's comments disagree with its order.
    """
    best, best_w, best_s = None, -float('inf'), -float('inf')
    for child in roots:
        winrate, score = node_eval(child)
        if winrate is None:
            continue
        tie = score if score is not None else -float('inf')
        if winrate > best_w or (winrate == best_w and tie > best_s):
            best, best_w, best_s = child, winrate, tie
    return best


def analyse(path):
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
        answer_wr, _ = node_eval(answer)

        win_value, win_depth = minimax(answer, is_pass, False, 1)
        answer_broken = win_value is not None and win_value < WIN_THRESHOLD

        worst_refuted = None          # the refuted move that gets closest to a win
        refuted_broken = 0
        refuted_valued = 0
        pass_root_wins = 0
        for root in roots:
            if root is answer:
                continue
            value, leaf_depth = minimax(root, is_pass, False, 1)
            if value is None:
                continue
            _, coord = node_move(root)
            if is_pass(coord):
                if value >= WIN_THRESHOLD:
                    pass_root_wins += 1
                continue
            refuted_valued += 1
            if value >= WIN_THRESHOLD:
                refuted_broken += 1
            if worst_refuted is None or value > worst_refuted[0]:
                worst_refuted = (value, leaf_depth)

        if answer_broken and refuted_broken:
            verdict = 'both_broken'
        elif answer_broken:
            verdict = 'answer_broken'
        elif refuted_broken:
            verdict = 'refuted_broken'
        else:
            verdict = 'clean'

        return dict(
            size=size, tid=os.path.basename(path)[:-4], verdict=verdict,
            answer_wr=answer_wr, n_roots=len(roots),
            answer_value=win_value, answer_depth=win_depth,
            refuted_valued=refuted_valued, refuted_broken=refuted_broken,
            worst_refuted=worst_refuted[0] if worst_refuted else None,
            worst_refuted_depth=worst_refuted[1] if worst_refuted else None,
            pass_root_wins=pass_root_wins)
    except Exception as exc:                                # noqa: BLE001
        return dict(size='ERR', tid=os.path.basename(path)[:-4], verdict='ERR',
                    answer_wr=str(exc)[:60], n_roots=-1, answer_value=None,
                    answer_depth=None, refuted_valued=-1, refuted_broken=-1,
                    worst_refuted=None, worst_refuted_depth=None,
                    pass_root_wins=-1)


COLUMNS = ['size', 'tid', 'verdict', 'answer_wr', 'n_roots', 'answer_value',
           'answer_depth', 'refuted_valued', 'refuted_broken', 'worst_refuted',
           'worst_refuted_depth', 'pass_root_wins']


def collect(subdir, sample, workers):
    paths = []
    for size in (9, 13):
        directory = os.path.join(PROJECT_ROOT, subdir, str(size))
        if not os.path.isdir(directory):
            continue
        with os.scandir(directory) as it:
            found = sorted(e.path for e in it if e.name.endswith('.sgf'))
        if sample:
            random.seed(4321 + size)
            found = random.sample(found, min(sample, len(found)))
        paths.extend(found)
    print(f"  {subdir}/: {len(paths)} problems", flush=True)
    with Pool(workers) as pool:
        return [r for r in pool.map(analyse, paths, chunksize=40) if r]


def write_tsv(rows, path):
    with open(path, 'w') as fh:
        fh.write('\t'.join(COLUMNS) + '\n')
        for row in rows:
            fh.write('\t'.join(str(row[c]) for c in COLUMNS) + '\n')


def summarise(name, rows):
    print(f"\n{name}")
    for size in (9, 13):
        subset = [r for r in rows if r['size'] == size]
        if not subset:
            continue
        counts = Counter(r['verdict'] for r in subset)
        print(f"  {size}x{size}  n={len(subset)}")
        for key in ('clean', 'answer_broken', 'refuted_broken', 'both_broken', 'ERR'):
            if counts[key]:
                print(f"     {key:16s} {counts[key]:7d} "
                      f"({100 * counts[key] / len(subset):6.3f}%)")


def load_prior_claim():
    """Ids the retired depth-floor audit named as broken, if its file is present."""
    claimed = {}
    path = os.path.join(HERE, 'broken_review_paths.txt')
    if not os.path.exists(path):
        return claimed
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4:
            claimed[(int(parts[0]), parts[1])] = (parts[2], parts[3])
    return claimed


def main():
    sys.setrecursionlimit(RECURSION_LIMIT)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=None,
                    help='audit one repo-relative directory instead of both')
    ap.add_argument('--out', default=None, help='raw per-problem TSV')
    ap.add_argument('--sample', type=int, default=0,
                    help='N random problems per board size (0 = all)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    if args.dir:
        rows = collect(args.dir, args.sample, args.workers)
        summarise(args.dir, rows)
        if args.out:
            write_tsv(rows, args.out)
            print(f"\nwrote {args.out}")
        return

    print("auditing:")
    live = collect('tsumegos', args.sample, args.workers)
    quarantined = collect('tsumegos/broken', args.sample, args.workers)
    summarise('LIVE CORPUS', live)
    summarise('QUARANTINE', quarantined)
    if args.out:
        write_tsv(live + quarantined, args.out)

    claimed = load_prior_claim()

    broken = sorted((r for r in live if r['verdict'] != 'clean'),
                    key=lambda r: (r['size'], r['tid']))
    out_broken = os.path.join(HERE, 'broken_solution_lines.txt')
    with open(out_broken, 'w') as fh:
        fh.write(
            "# Live-corpus tsumegos whose recorded continuation contradicts the\n"
            "# problem: the answer cannot hold the win, or a move the problem\n"
            "# calls losing can be won with.\n"
            "#\n"
            "# Produced by auxiliary/audit_solution_lines.py -- exact minimax over\n"
            "# the branches that reach a double pass, on the winrate comments the\n"
            "# SGF already stores. Branches that never reach a double pass were\n"
            "# started and abandoned by the generator and are excluded; they are\n"
            "# not evidence in either direction.\n"
            "#\n"
            "# answer_broken   the answer's minimax value is below the win threshold\n"
            "# refuted_broken  some non-answer first move reaches the win threshold\n"
            "# both_broken     both\n"
            "#\n"
            "# Answer coverage is complete. REFUTED-MOVE COVERAGE IS NOT: a refuted\n"
            "# move is only judged when some line under it was played out, and the\n"
            "# generator pursues refutations far less often on 9x9 than on 13x13,\n"
            "# so refuted_broken is a floor rather than a count.\n"
            "#\n"
            "# The root pass variation is excluded from the verdict and reported in\n"
            "# pass_root_wins; the client cannot offer a pass as an answer.\n"
            "#\n"
            "# Columns, tab-separated:\n"
            "#   size  tsumego_id  verdict  answer_value  answer_depth  "
            "worst_refuted  refuted_valued  pass_root_wins\n")
        for r in broken:
            fh.write(f"{r['size']}\t{r['tid']}\t{r['verdict']}\t{r['answer_value']}\t"
                     f"{r['answer_depth']}\t{r['worst_refuted']}\t"
                     f"{r['refuted_valued']}\t{r['pass_root_wins']}\n")

    cleared = sorted((r for r in quarantined if r['verdict'] == 'clean'),
                     key=lambda r: (r['size'], r['tid']))
    out_cleared = os.path.join(HERE, 'quarantine_cleared.txt')
    with open(out_cleared, 'w') as fh:
        fh.write(
            "# Quarantined tsumegos that this audit finds SOUND: inside the\n"
            "# played-out data the answer holds the win and no refuted move can be\n"
            "# won with. They are candidates to return to the corpus.\n"
            "#\n"
            "# Produced by auxiliary/audit_solution_lines.py. See\n"
            "# broken_solution_lines.txt for the method and its coverage limits.\n"
            "#\n"
            "# prior_claim names why the problem was removed:\n"
            "#   review_path  the retired depth-floor audit in broken_review_paths.txt\n"
            "#                named it. That audit scored ABANDONED lines and applied a\n"
            "#                ply-count floor in place of a played-out test, so a\n"
            "#                clean verdict here contradicts it directly.\n"
            "#   other        removed by an earlier purge whose reason is not recorded\n"
            "#                here; this audit tests the solution line only, so a clean\n"
            "#                verdict does not clear whatever that purge was about.\n"
            "#\n"
            "# prior_worst / prior_depth are what the review-path audit recorded, for\n"
            "# comparison against answer_value / answer_depth.\n"
            "#\n"
            "# Columns, tab-separated:\n"
            "#   size  tsumego_id  prior_claim  prior_worst  prior_depth  "
            "answer_value  answer_depth  refuted_valued\n")
        for r in cleared:
            key = (r['size'], r['tid'])
            if key in claimed:
                claim, prior = 'review_path', claimed[key]
            else:
                claim, prior = 'other', ('-', '-')
            fh.write(f"{r['size']}\t{r['tid']}\t{claim}\t{prior[0]}\t{prior[1]}\t"
                     f"{r['answer_value']}\t{r['answer_depth']}\t"
                     f"{r['refuted_valued']}\n")

    print(f"\nwrote {out_broken}  ({len(broken)} problems)")
    print(f"wrote {out_cleared}  ({len(cleared)} problems)")


if __name__ == '__main__':
    main()
