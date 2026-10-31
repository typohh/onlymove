"""Audit the generated tsumego corpus' variation lines against the properties
the generator is supposed to guarantee.

For every SGF in tsumegos/<size>/ this measures:

  1. Does the WINNING sequence (the winning first move's principal line — first
     child at every step) run to the end of the game, i.e. finish on two
     consecutive passes?  Also: whether ANY line in that subtree does.
  2. Does at least one NON-winning first move have a sequence running to the end?
  3. Does the completed winning line actually end in a WIN for the solver, and do
     the completed losing lines end in a LOSS?
  4. Does the winning line stay winning at EVERY node along the way — checked
     separately by winrate and by score?  (Same, mirrored, for losing lines.)
  5. How many nodes carry a SELF-CONTRADICTORY evaluation — winrate above 50%
     with the score below 0, or the reverse — at the first-move level, on the
     principal lines, and anywhere in the tree.

THE ANSWER IS variation_nodes[0] — the corpus contract every consumer reads,
the client's isCorrectMove included.  This audit deliberately picks the winner
the OTHER way, by evaluation (best winrate among the first moves, score as the
tie-break), precisely so the two can be compared: the count below is how often a
shipped file's own comments name a first move other than the one generation
verified, which is a corpus-health signal and nothing else.  A disagreement is
never resolved by reordering; the answer's position is fixed when the tree is
built.  Also reported is a census of the stricter
"exactly one first move with winrate > 50% AND score > 0" rule that
utils.katago.add_variations.first_move_is_well_formed gates new problems on.

PERSPECTIVE.  Every "{winrate}%/{score}/{visits}" comment is BLACK-perspective
(bots/analysis.cfg sets reportAnalysisWinratesAs = BLACK; pwa/js/sgf.js reads
them the same way), so the values do NOT alternate with the side to move.  The
report prints a self-check of exactly that — the mean gap between consecutive
nodes on a line against the mean of |w_i + w_next - 100| — so the assumption is
verified by the data rather than assumed.  Values are converted to the solver's
perspective, which is a no-op unless a problem has a White solver; the solver
colour census in the report says whether any does.

Winrate and score are stored rounded to one decimal, so a genuine 0.04-point
lead reads as "0.0"; exact ties on either axis are counted and reported
separately rather than folded into the failures.

Usage:
    ./.venv/bin/python3 -m auxiliary.check_tsumego_sequences
    ./.venv/bin/python3 -m auxiliary.check_tsumego_sequences --sizes 13 --limit 2000
    ./.venv/bin/python3 -m auxiliary.check_tsumego_sequences --csv /tmp/seq.csv
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

from sgfmill import sgf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import tsumego_dir  # noqa: E402
from utils.katago.add_variations import parse_eval_comment  # noqa: E402
from utils.constants import ALL_BOARD_SIZES  # noqa: E402

# A conflicting node only counts as MATERIAL when both axes are clear of the
# rounding grain and of the coin-flip band around 50%.
MATERIAL_WINRATE_MARGIN = 5.0
MATERIAL_SCORE_MARGIN = 0.5
STRONG_WINRATE_MARGIN = 20.0
STRONG_SCORE_MARGIN = 2.0


def find_problem_node(game):
    """The position the solver faces: walk the single-child chain from the root.

    Same rule as pipeline.benchmark_bots.parse_tsumego_sgf — the setup game is a
    linear sequence and the problem position is the first node that branches.
    """
    node = game.get_root()
    while len(node) == 1:
        node = node[0]
    return node


def is_pass(node):
    colour, move = node.get_move()
    return colour is not None and move is None


def principal_line(node):
    """The node plus its first-child chain down to a leaf."""
    line = [node]
    cur = node
    while len(cur):
        cur = cur[0]
        line.append(cur)
    return line


def ends_the_game(line):
    """True iff the line's last two moves are both passes (the game ended)."""
    moves = [n for n in line if n.get_move()[0] is not None]
    return len(moves) >= 2 and is_pass(moves[-1]) and is_pass(moves[-2])


def subtree_has_complete_line(node):
    """True iff some leaf below (and including) `node` is reached by two passes.

    Iterative DFS — the trees run to MAX_VARIATION_DEPTH (800) plies, well past
    Python's recursion limit.
    """
    # stack entries: (node, prev_was_pass, prev_prev_was_pass)
    stack = [(node, False, False)]
    while stack:
        cur, prev_pass, prev2_pass = stack.pop()
        colour, move = cur.get_move()
        if colour is not None:
            prev2_pass, prev_pass = prev_pass, move is None
        if not len(cur):
            if prev_pass and prev2_pass:
                return True
            continue
        for child in cur:
            stack.append((child, prev_pass, prev2_pass))
    return False


def black_eval(node):
    """(winrate, score) after `node`'s move, BLACK perspective — as stored."""
    comment = node.get("C") if node.has_property("C") else None
    return parse_eval_comment(comment)


def to_solver(ev, solver_is_black):
    """Black-perspective (winrate, score) -> the solver's perspective."""
    if ev is None:
        return None
    return ev if solver_is_black else (100.0 - ev[0], -ev[1])


def is_conflicting(ev):
    """Winrate and score disagree on who is ahead (perspective-invariant)."""
    if ev is None:
        return False
    winrate, score = ev
    return (winrate > 50.0 and score < 0.0) or (winrate < 50.0 and score > 0.0)


def conflict_strength(ev):
    """How far a conflicting node sits from the two decision boundaries."""
    return abs(ev[0] - 50.0), abs(ev[1])


def pick_winning_move(children, solver_is_black):
    """The winning first move, by EVALUATION — deliberately not by position.

    Best solver-perspective winrate among the first moves, score as the tie-break.
    Returns (index, node) or (None, None) when nothing carries an evaluation.
    """
    best_idx, best_key = None, None
    for i, child in enumerate(children):
        if child.get_move()[0] is None:  # a pass alternative is not a candidate
            continue
        ev = to_solver(black_eval(child), solver_is_black)
        if ev is None:
            continue
        if best_key is None or ev > best_key:
            best_key, best_idx = ev, i
    return best_idx, (children[best_idx] if best_idx is not None else None)


def summarize_line(line, solver_is_black):
    """Evaluation extremes, terminal value and conflicts along one line."""
    evals = [to_solver(black_eval(n), solver_is_black) for n in line]
    present = [e for e in evals if e is not None]
    terminal = next((e for e in reversed(evals) if e is not None), None)
    # Where the line first stops being a win — index 0 means the winning move
    # itself is not evaluated as winning, anything later is a mid-line collapse.
    first_dip_wr = next((i for i, e in enumerate(evals) if e is not None and e[0] <= 50.0), None)
    first_dip_sc = next((i for i, e in enumerate(evals) if e is not None and e[1] <= 0.0), None)
    # Perspective self-check: consecutive nodes alternate colour, so if the
    # stored values were mover-perspective these two means would swap places.
    pairs = [(a, b) for a, b in zip(present, present[1:])]
    return {
        "n_nodes": len(line),
        "n_uncommented": sum(1 for e in evals if e is None),
        "min_winrate": min(e[0] for e in present) if present else None,
        "max_winrate": max(e[0] for e in present) if present else None,
        "min_score": min(e[1] for e in present) if present else None,
        "max_score": max(e[1] for e in present) if present else None,
        "terminal_winrate": terminal[0] if terminal else None,
        "terminal_score": terminal[1] if terminal else None,
        "first_dip_winrate_idx": first_dip_wr,
        "first_dip_score_idx": first_dip_sc,
        "n_conflicting": sum(1 for e in present if is_conflicting(e)),
        "adjacent_pairs": len(pairs),
        "adjacent_abs_delta": sum(abs(b[0] - a[0]) for a, b in pairs),
        "adjacent_abs_sum_gap": sum(abs(a[0] + b[0] - 100.0) for a, b in pairs),
    }


def scan_tree_conflicts(root):
    """Census of self-contradictory evaluations over the whole variation tree."""
    total = conflicting = material = strong = 0
    worst_winrate = worst_score = 0.0
    stack = [root]
    while stack:
        cur = stack.pop()
        ev = black_eval(cur)
        if ev is not None:
            total += 1
            if is_conflicting(ev):
                conflicting += 1
                dw, ds = conflict_strength(ev)
                if dw >= MATERIAL_WINRATE_MARGIN and ds >= MATERIAL_SCORE_MARGIN:
                    material += 1
                if dw >= STRONG_WINRATE_MARGIN and ds >= STRONG_SCORE_MARGIN:
                    strong += 1
                worst_winrate = max(worst_winrate, dw)
                worst_score = max(worst_score, ds)
        stack.extend(cur)
    return {
        "tree_n_evaluated": total,
        "tree_n_conflicting": conflicting,
        "tree_n_conflicting_material": material,
        "tree_n_conflicting_strong": strong,
        "tree_worst_conflict_winrate_margin": worst_winrate,
        "tree_worst_conflict_score_margin": worst_score,
    }


def check_file(path):
    try:
        with open(path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}

    try:
        problem = find_problem_node(game)
        children = list(problem)
        if not children:
            return {"path": path, "error": "no first-move variations"}

        solver_colour = next((c.get_move()[0] for c in children if c.get_move()[0] is not None), None)
        if solver_colour is None:
            return {"path": path, "error": "no first move carries a colour"}
        solver_is_black = solver_colour == "b"

        winner_idx, winner = pick_winning_move(children, solver_is_black)
        if winner is None:
            return {"path": path, "error": "no first move carries an evaluation"}

        # Census over the first-move candidates: the strict single-solution rule
        # (winrate > 50% AND score > 0) and the self-contradictory ones.
        strict_winners = conflicting_first = evaluated_first = 0
        for child in children:
            if child.get_move()[0] is None:
                continue
            ev_black = black_eval(child)
            if ev_black is None:
                continue
            evaluated_first += 1
            if is_conflicting(ev_black):
                conflicting_first += 1
            ev = to_solver(ev_black, solver_is_black)
            if ev[0] > 50.0 and ev[1] > 0.0:
                strict_winners += 1

        win_line = principal_line(winner)
        row = {
            "path": path,
            "error": None,
            "solver": solver_colour,
            "n_first_moves": len(children),
            "n_first_moves_evaluated": evaluated_first,
            "winner_idx": winner_idx,
            "winner_is_first_child": winner_idx == 0,
            "n_strict_winners": strict_winners,
            "n_conflicting_first_moves": conflicting_first,
            "sol_principal_complete": ends_the_game(win_line),
            "sol_any_complete": subtree_has_complete_line(winner),
        }
        row.update({f"sol_{k}": v for k, v in summarize_line(win_line, solver_is_black).items()})
        row.update(scan_tree_conflicts(problem))

        alt_principal_complete = 0
        alt_any_complete = 0
        alt_lines = []
        for i, child in enumerate(children):
            if i == winner_idx or child.get_move()[0] is None:
                continue
            line = principal_line(child)
            if ends_the_game(line):
                alt_principal_complete += 1
                alt_lines.append(summarize_line(line, solver_is_black))
            if subtree_has_complete_line(child):
                alt_any_complete += 1

        row["n_alt_principal_complete"] = alt_principal_complete
        row["n_alt_any_complete"] = alt_any_complete
        # Aggregate the completed losing lines: a wrong first move must stay lost,
        # so the SOLVER's best moment along the line is the binding value.
        if alt_lines:
            row["alt_worst_terminal_winrate"] = max(a["terminal_winrate"] for a in alt_lines)
            row["alt_worst_terminal_score"] = max(a["terminal_score"] for a in alt_lines)
            row["alt_max_winrate"] = max(a["max_winrate"] for a in alt_lines)
            row["alt_max_score"] = max(a["max_score"] for a in alt_lines)
            row["alt_n_terminal_loss"] = sum(
                1 for a in alt_lines
                if a["terminal_winrate"] < 50.0 and a["terminal_score"] < 0.0
            )
            row["alt_n_stays_lost_winrate"] = sum(1 for a in alt_lines if a["max_winrate"] < 50.0)
            row["alt_n_stays_lost_score"] = sum(1 for a in alt_lines if a["max_score"] < 0.0)
        else:
            for key in ("alt_worst_terminal_winrate", "alt_worst_terminal_score",
                        "alt_max_winrate", "alt_max_score", "alt_n_terminal_loss",
                        "alt_n_stays_lost_winrate", "alt_n_stays_lost_score"):
                row[key] = None
        return row
    except Exception as e:
        return {"path": path, "error": f"{type(e).__name__}: {e}"}


def pct(num, den):
    return f"{num:7d}/{den:<7d} ({100.0 * num / den:6.2f}%)" if den else f"{num:7d}/{0:<7d} (   n/a)"


def report(size, rows, errors):
    total = len(rows) + len(errors)
    print(f"\n{'=' * 78}")
    print(f"Board {size}: {total} SGF files ({len(errors)} unreadable/malformed)")
    print(f"{'=' * 78}")
    if not rows:
        return
    n = len(rows)

    print(f"\n0. Sanity of the reading conventions")
    colours = {}
    for r in rows:
        colours[r["solver"]] = colours.get(r["solver"], 0) + 1
    print(f"  solver colour: " + ", ".join(f"{k}={v}" for k, v in sorted(colours.items()))
          + "   (the Black-perspective -> solver flip only ever fires on a White solver)")
    pairs = sum(r["sol_adjacent_pairs"] for r in rows)
    if pairs:
        d = sum(r["sol_adjacent_abs_delta"] for r in rows) / pairs
        s = sum(r["sol_adjacent_abs_sum_gap"] for r in rows) / pairs
        print(f"  consecutive nodes on the winning line (they alternate colour):")
        print(f"    mean |w_next - w_cur|      = {d:6.2f}   <- small => ONE fixed perspective")
        print(f"    mean |w_cur + w_next - 100| = {s:6.2f}   <- small would mean mover-perspective")

    print(f"\n   Identifying the winning first move (by evaluation, for comparison)")
    not_first = sum(1 for r in rows if not r["winner_is_first_child"])
    print(f"  winner is NOT variation_nodes[0]:      {pct(not_first, n)}")
    strict = {}
    for r in rows:
        strict[r["n_strict_winners"]] = strict.get(r["n_strict_winners"], 0) + 1
    print(f"  first moves with winrate>50% AND score>0 (should be exactly 1):")
    for k in sorted(strict):
        print(f"    {k} such move(s): {pct(strict[k], n)}")
    winner_strict = sum(1 for r in rows if r["sol_min_winrate"] is not None)
    del winner_strict

    print(f"\n1. Winning sequence runs to the end of the game (pass, pass)")
    sol_pc = sum(1 for r in rows if r["sol_principal_complete"])
    sol_any = sum(1 for r in rows if r["sol_any_complete"])
    print(f"  principal line (best play) ends the game:  {pct(sol_pc, n)}")
    print(f"  some line in the winning subtree ends it:  {pct(sol_any, n)}")

    print(f"\n2. A non-winning first move runs to the end of the game")
    alt_pc = sum(1 for r in rows if r["n_alt_principal_complete"] > 0)
    alt_any = sum(1 for r in rows if r["n_alt_any_complete"] > 0)
    print(f"  >=1 alternative whose principal line ends the game: {pct(alt_pc, n)}")
    print(f"  >=1 alternative with some line ending the game:     {pct(alt_any, n)}")
    counts = sorted(r["n_alt_principal_complete"] for r in rows)
    print(f"  alternatives-with-a-complete-principal-line per problem: "
          f"median {counts[n // 2]}, mean {sum(counts) / n:.2f}, max {counts[-1]}")

    print(f"\n3. Terminal result of a completed sequence")
    done = [r for r in rows if r["sol_principal_complete"] and r["sol_terminal_winrate"] is not None]
    d = len(done)
    win_wr = sum(1 for r in done if r["sol_terminal_winrate"] > 50.0)
    win_sc = sum(1 for r in done if r["sol_terminal_score"] > 0.0)
    win_both = sum(1 for r in done if r["sol_terminal_winrate"] > 50.0 and r["sol_terminal_score"] > 0.0)
    tie_wr = sum(1 for r in done if r["sol_terminal_winrate"] == 50.0)
    tie_sc = sum(1 for r in done if r["sol_terminal_score"] == 0.0)
    print(f"  winning line ends in a WIN for the solver (of {d} completed lines)")
    print(f"    by winrate (>50%):     {pct(win_wr, d)}   [exactly 50.0%: {tie_wr}]")
    print(f"    by score   (>0):       {pct(win_sc, d)}   [exactly 0.0: {tie_sc}]")
    print(f"    by both:               {pct(win_both, d)}")

    alt_done = [r for r in rows if r["alt_n_terminal_loss"] is not None]
    ad = len(alt_done)
    lines_total = sum(r["n_alt_principal_complete"] for r in alt_done)
    lines_loss = sum(r["alt_n_terminal_loss"] for r in alt_done)
    all_loss = sum(1 for r in alt_done if r["alt_n_terminal_loss"] == r["n_alt_principal_complete"])
    print(f"  losing lines end in a LOSS for the solver (winrate<50% AND score<0)")
    print(f"    per completed losing line:  {pct(lines_loss, lines_total)}")
    print(f"    problems where every completed losing line loses: {pct(all_loss, ad)}")

    print(f"\n4. Winning sequence stays winning at EVERY node along the line ({d} completed lines)")
    mono_wr = sum(1 for r in done if r["sol_min_winrate"] > 50.0)
    mono_sc = sum(1 for r in done if r["sol_min_score"] > 0.0)
    mono_both = sum(1 for r in done if r["sol_min_winrate"] > 50.0 and r["sol_min_score"] > 0.0)
    tie_wr_line = sum(1 for r in done if r["sol_min_winrate"] == 50.0)
    tie_sc_line = sum(1 for r in done if r["sol_min_score"] == 0.0)
    print(f"    winrate > 50% throughout:  {pct(mono_wr, d)}   [dips to exactly 50.0%: {tie_wr_line}]")
    print(f"    score   > 0   throughout:  {pct(mono_sc, d)}   [dips to exactly 0.0: {tie_sc_line}]")
    print(f"    both throughout:           {pct(mono_both, d)}")
    dips_wr = sorted(r["sol_min_winrate"] for r in done if r["sol_min_winrate"] <= 50.0)
    dips_sc = sorted(r["sol_min_score"] for r in done if r["sol_min_score"] <= 0.0)
    if dips_wr:
        print(f"    winrate dips: median {dips_wr[len(dips_wr) // 2]:.1f}%, "
              f"p10 {dips_wr[max(0, len(dips_wr) // 10)]:.1f}%, min {dips_wr[0]:.1f}%")
    if dips_sc:
        print(f"    score dips:   median {dips_sc[len(dips_sc) // 2]:.1f}, "
              f"p10 {dips_sc[max(0, len(dips_sc) // 10)]:.1f}, min {dips_sc[0]:.1f}")
    at_move_one = sum(1 for r in done if r["sol_first_dip_winrate_idx"] == 0
                      or r["sol_first_dip_score_idx"] == 0)
    dipped = [r for r in done if r["sol_first_dip_winrate_idx"] is not None
              or r["sol_first_dip_score_idx"] is not None]
    recovered = sum(1 for r in dipped
                    if r["sol_terminal_winrate"] > 50.0 and r["sol_terminal_score"] > 0.0)
    print(f"    of the {len(dipped)} lines that dip: {at_move_one} dip at the winning move itself, "
          f"{recovered} recover to a win by the end")
    uncommented = sum(1 for r in done if r["sol_n_uncommented"])
    if uncommented:
        print(f"    lines with at least one uncommented (unevaluated) node: {uncommented}")

    print(f"\n   Losing sequences stay lost at every node ({lines_total} completed losing lines)")
    lost_wr = sum(r["alt_n_stays_lost_winrate"] for r in alt_done)
    lost_sc = sum(r["alt_n_stays_lost_score"] for r in alt_done)
    print(f"    winrate < 50% throughout:  {pct(lost_wr, lines_total)}")
    print(f"    score   < 0   throughout:  {pct(lost_sc, lines_total)}")

    print(f"\n   Combined: winning line complete AND ends in a win AND stays winning on both axes")
    clean = sum(1 for r in done
                if r["sol_terminal_winrate"] > 50.0 and r["sol_terminal_score"] > 0.0
                and r["sol_min_winrate"] > 50.0 and r["sol_min_score"] > 0.0)
    print(f"    {pct(clean, n)} of all problems")

    print(f"\n5. Nodes whose winrate and score disagree (winrate>50% with score<0, or the reverse)")
    fm_nodes = sum(r["n_first_moves_evaluated"] for r in rows)
    fm_conf = sum(r["n_conflicting_first_moves"] for r in rows)
    fm_probs = sum(1 for r in rows if r["n_conflicting_first_moves"] > 0)
    print(f"  among FIRST MOVES (what first_move_is_well_formed gates on):")
    print(f"    conflicting candidate moves:  {pct(fm_conf, fm_nodes)}")
    print(f"    problems with >=1:            {pct(fm_probs, n)}")
    line_conf = sum(1 for r in rows if r["sol_n_conflicting"] > 0)
    print(f"  on the WINNING LINE: problems with >=1 conflicting node: {pct(line_conf, n)}")
    tree_nodes = sum(r["tree_n_evaluated"] for r in rows)
    tree_conf = sum(r["tree_n_conflicting"] for r in rows)
    tree_mat = sum(r["tree_n_conflicting_material"] for r in rows)
    tree_strong = sum(r["tree_n_conflicting_strong"] for r in rows)
    tree_probs = sum(1 for r in rows if r["tree_n_conflicting"] > 0)
    tree_probs_mat = sum(1 for r in rows if r["tree_n_conflicting_material"] > 0)
    print(f"  ANYWHERE in the variation tree ({tree_nodes} evaluated nodes):")
    print(f"    conflicting nodes:            {pct(tree_conf, tree_nodes)}")
    print(f"    ... material (|winrate-50|>={MATERIAL_WINRATE_MARGIN:.0f} and "
          f"|score|>={MATERIAL_SCORE_MARGIN}):  {pct(tree_mat, tree_nodes)}")
    print(f"    ... strong   (|winrate-50|>={STRONG_WINRATE_MARGIN:.0f} and "
          f"|score|>={STRONG_SCORE_MARGIN}):  {pct(tree_strong, tree_nodes)}")
    print(f"    problems with >=1 conflicting node:          {pct(tree_probs, n)}")
    print(f"    problems with >=1 MATERIAL conflicting node: {pct(tree_probs_mat, n)}")
    worst_w = sorted((r["tree_worst_conflict_winrate_margin"] for r in rows), reverse=True)[:1]
    worst_s = sorted((r["tree_worst_conflict_score_margin"] for r in rows), reverse=True)[:1]
    if worst_w:
        # Independent maxima over conflicting nodes — not necessarily the same node.
        print(f"    worst margins on a conflicting node: |winrate-50| up to {worst_w[0]:.1f} "
              f"points, |score| up to {worst_s[0]:.1f}")

    if errors:
        print(f"\n  First errors:")
        for e in errors[:5]:
            print(f"    {os.path.basename(e['path'])}: {e['error']}")


CSV_FIELDS = [
    "path", "error", "solver", "n_first_moves", "n_first_moves_evaluated",
    "winner_idx", "winner_is_first_child", "n_strict_winners", "n_conflicting_first_moves",
    "sol_principal_complete", "sol_any_complete", "sol_n_nodes", "sol_n_uncommented",
    "sol_min_winrate", "sol_max_winrate", "sol_min_score", "sol_max_score",
    "sol_terminal_winrate", "sol_terminal_score",
    "sol_first_dip_winrate_idx", "sol_first_dip_score_idx", "sol_n_conflicting",
    "n_alt_principal_complete", "n_alt_any_complete",
    "alt_worst_terminal_winrate", "alt_worst_terminal_score",
    "alt_max_winrate", "alt_max_score", "alt_n_terminal_loss",
    "alt_n_stays_lost_winrate", "alt_n_stays_lost_score",
    "tree_n_evaluated", "tree_n_conflicting", "tree_n_conflicting_material",
    "tree_n_conflicting_strong", "tree_worst_conflict_winrate_margin",
    "tree_worst_conflict_score_margin",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(ALL_BOARD_SIZES),
                        choices=list(ALL_BOARD_SIZES), help="Board sizes to scan")
    parser.add_argument("--dir", default=None,
                        help="Scan this directory instead (implies a single size)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only check this many files per size (random sample)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for --limit")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--csv", default=None, help="Write the per-problem table here")
    args = parser.parse_args()

    targets = ([(args.dir, args.dir)] if args.dir
               else [(s, tsumego_dir(s)) for s in args.sizes])

    csv_writer = csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["size"] + CSV_FIELDS, extrasaction="ignore")
        csv_writer.writeheader()

    for size, directory in targets:
        if not os.path.isdir(directory):
            print(f"Skipping {directory} (not found)", file=sys.stderr)
            continue
        files = [os.path.join(directory, name) for name in os.listdir(directory)
                 if name.endswith(".sgf")]
        files.sort()
        if args.limit and len(files) > args.limit:
            random.Random(args.seed).shuffle(files)
            files = sorted(files[:args.limit])
        total = len(files)
        print(f"Scanning {total} files in {directory} with {args.workers} workers...")

        rows, errors = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(check_file, files, chunksize=32), 1):
                if row.get("error"):
                    errors.append(row)
                else:
                    rows.append(row)
                if csv_writer:
                    csv_writer.writerow({"size": size, **row})
                if i % 5000 == 0 or i == total:
                    print(f"  {i}/{total} ({100.0 * i / total:.1f}%)", flush=True)
        report(size, rows, errors)

    if csv_file:
        csv_file.close()
        print(f"\nPer-problem table written to {args.csv}")


if __name__ == "__main__":
    main()
