"""Single source of truth for Go board parsing, simulation and SVG rendering.

The auxiliary inspection tools (auxiliary/rating_dev/inspect_residual_axes.py)
import from here, so there is exactly one copy of the board-position logic in
Python.

The board is shown the way the solver sees it (mirrors the PWA board.js
markers):
  - last move      → contrasting circle on the stone   (board.js CR marker)
  - ko-illegal pt  → square on the empty intersection  (board.js SQ marker)
  - correct move   → translucent phantom stone + red ring (PWA hides this
                     during solving; inspection tools reveal it)

Ko detection mirrors the classic simple-ko shape: the last move captured
exactly one stone and is itself a lone stone in atari whose sole liberty is
the captured point — the opponent may not immediately recapture.
"""

import os
import re

from utils.paths import PROJECT_ROOT, tsumego_dir

_STAR_POINTS = {
    9: [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)],
    13: [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)],
    19: [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)],
}


# ---- SGF parsing -------------------------------------------------------

# Setup-chain move: ";B[ab]" / ";W[ab]" with a REQUIRED leading ";" so an
# AB[..]/AW[..] setup-stone property is never mistaken for a move.
_SETUP_MOVE_RE = re.compile(r";\s*([BW])\[([a-z]{0,2})\]")
# A move inside a variation's first node (the leading ";" has been stripped).
_VAR_MOVE_RE = re.compile(r"([BW])\[([a-z]{0,2})\]")
# Embedded analysis comment: a Black-perspective "<winrate>%/<score>/<visits>",
# e.g. "86.4%/3.1/1000". Every candidate first move carries one. NOT anchored on
# the closing bracket — the visit count was appended after this regex was
# written, and anchoring is what would have made it silently match nothing.
_CMT_RE = re.compile(r"C\[(\d+\.?\d*)%/(-?\d+\.?\d*)")


def _top_level_variations(s):
    """The content of each top-level ``(...)`` group in ``s`` (which begins at the
    first ``(``). Tracks ``[...]`` property values so a paren inside a comment
    can't skew the paren depth, and honours SGF's ``\\]`` escape."""
    groups, i, n = [], 0, len(s)
    while i < n:
        if s[i] != "(":
            i += 1
            continue
        depth, j, in_brackets = 0, i, False
        while j < n:
            c = s[j]
            if in_brackets:
                if c == "\\":
                    j += 2
                    continue
                if c == "]":
                    in_brackets = False
                j += 1
                continue
            if c == "[":
                in_brackets = True
            elif c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    groups.append(s[i + 1:j])
                    j += 1
                    break
            j += 1
        i = j
    return groups


def _select_correct_move(variations_region):
    """The correct first move: the FIRST top-level variation that plays a stone.

    That is the corpus contract, the same one `isCorrectMove` (pwa/js/sgf.js) and
    every backend consumer read — construct_initial_sgf writes the verified
    solution as the decision node's first child and every annotation pass only
    appends, so variation 0 is the answer. This used to re-derive it from the
    embedded winrates instead; that is a shallower annotation than the search
    that verified the solution, and re-deriving it here made this a second,
    independent definition of the answer that could drift from the contract.

    A pass is skipped rather than returned — it is never the solution, and it can
    appear among the variations. Returns (color, "ab"), or None if no top-level
    variation plays a stone."""
    cands = []  # (color, coord, winrate|None, score|None)
    for content in _top_level_variations(variations_region):
        head = content[1:] if content.startswith(";") else content
        end = re.search(r"[;(]", head)   # the first node ends at the next node / child
        first_node = head[:end.start()] if end else head
        mv = _VAR_MOVE_RE.search(first_node)
        if not mv or not mv.group(2) or mv.group(2) == "tt":
            continue                     # skip a pass — never the solution
        cm = _CMT_RE.search(first_node)
        wr = float(cm.group(1)) if cm else None
        sc = float(cm.group(2)) if cm else None
        cands.append((mv.group(1), mv.group(2), wr, sc))
    if not cands:
        return None
    return (cands[0][0], cands[0][1])


def parse_sgf(sgf_text):
    """Return (size, setup_moves, correct_move).

    setup_moves: list of (color, "ab") played in the linear chain before the
    first variation node. correct_move: (color, "ab") of the CORRECT first move —
    the first stone-playing top-level variation, the corpus contract (see
    _select_correct_move). None when there are no variations.
    """
    size_m = re.search(r"SZ\[(\d+)\]", sgf_text)
    size = int(size_m.group(1)) if size_m else 9

    # Strip the leading outer "(" so we can find the first INNER "(".
    body = sgf_text.lstrip()
    if body.startswith("("):
        body = body[1:]
    first_inner = body.find("(")
    if first_inner == -1:
        setup_part, variations_region = body, ""
    else:
        setup_part, variations_region = body[:first_inner], body[first_inner:]

    setup = [(c, p) for c, p in _SETUP_MOVE_RE.findall(setup_part) if p and p != "tt"]
    correct = _select_correct_move(variations_region)
    return size, setup, correct


# ---- board simulation --------------------------------------------------

def _neighbors(x, y, size):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            yield nx, ny


def _group(board, x, y, size):
    """Flood-fill the group at (x, y); return (cells, liberty_set)."""
    color = board[y][x]
    seen, group, libs, stack = set(), [], set(), [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in seen:
            continue
        seen.add((cx, cy))
        group.append((cx, cy))
        for nx, ny in _neighbors(cx, cy, size):
            v = board[ny][nx]
            if v == ".":
                libs.add((nx, ny))
            elif v == color and (nx, ny) not in seen:
                stack.append((nx, ny))
    return group, libs


def simulate(setup, size):
    """Replay setup with capture rules. Return (board, last_move, ko).

    last_move = (color, x, y) of the final legally-played stone, or None.
    ko_point  = (x, y) the opponent may not retake, or None.
    """
    board = [["." for _ in range(size)] for _ in range(size)]
    last_move = None
    last_captured = []
    for color, coord in setup:
        x = ord(coord[0]) - ord("a")
        y = ord(coord[1]) - ord("a")
        if not (0 <= x < size and 0 <= y < size) or board[y][x] != ".":
            continue
        board[y][x] = color
        opp = "W" if color == "B" else "B"
        captured = []
        for nx, ny in _neighbors(x, y, size):
            if board[ny][nx] == opp:
                grp, libs = _group(board, nx, ny, size)
                if not libs:
                    for gx, gy in grp:
                        board[gy][gx] = "."
                    captured.extend(grp)
        grp, libs = _group(board, x, y, size)
        if not libs:  # suicide (no capture above saved it) — undo only the
            board[y][x] = "."  # just-placed stone, NOT the whole friendly
            continue           # group it joined (those were legal before this)
        last_move = (color, x, y)
        last_captured = captured

    ko = None
    if last_move and len(last_captured) == 1:
        _, lx, ly = last_move
        grp, libs = _group(board, lx, ly, size)
        if len(grp) == 1 and len(libs) == 1 and last_captured[0] in libs:
            ko = last_captured[0]
    return board, last_move, ko


# ---- SVG rendering -----------------------------------------------------

def render_svg(board, size, correct, last_move=None, ko=None, cell=30, margin=18):
    """Render the board to an SVG string. `correct`/`last_move`/`ko` are
    optional markers — pass None to omit any of them."""
    side = 2 * margin + (size - 1) * cell
    out = [f'<svg viewBox="0 0 {side} {side}" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<rect width="{side}" height="{side}" fill="#dcb35c"/>')
    out.append('<g stroke="#222" stroke-width="1">')
    lo, hi = margin, margin + (size - 1) * cell
    for i in range(size):
        v = margin + i * cell
        out.append(f'<line x1="{v}" y1="{lo}" x2="{v}" y2="{hi}"/>')
        out.append(f'<line x1="{lo}" y1="{v}" x2="{hi}" y2="{v}"/>')
    out.append('</g>')
    for sx, sy in _STAR_POINTS.get(size, []):
        out.append(f'<circle cx="{margin+sx*cell}" cy="{margin+sy*cell}" r="2.5" fill="#222"/>')

    r = cell * 0.46
    for y in range(size):
        for x in range(size):
            if board[y][x] != ".":
                cx, cy = margin + x * cell, margin + y * cell
                fill = "#111" if board[y][x] == "B" else "#fafafa"
                out.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="{fill}" '
                           f'stroke="#000" stroke-width="0.8"/>')

    # last move: contrasting circle on the stone
    if last_move:
        _, lx, ly = last_move
        cx, cy = margin + lx * cell, margin + ly * cell
        stroke = "#fafafa" if last_move[0] == "B" else "#111"
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r*0.5:.2f}" fill="none" '
                   f'stroke="{stroke}" stroke-width="2"/>')

    # ko-illegal point: square on the empty intersection
    if ko:
        kx, ky = ko
        cx, cy = margin + kx * cell, margin + ky * cell
        s = cell * 0.42
        out.append(f'<rect x="{cx-s/2:.2f}" y="{cy-s/2:.2f}" width="{s:.2f}" '
                   f'height="{s:.2f}" fill="none" stroke="#222" stroke-width="2"/>')

    # correct move: translucent phantom stone + red ring
    if correct:
        color, coord = correct
        cxc = ord(coord[0]) - ord("a")
        cyc = ord(coord[1]) - ord("a")
        cx, cy = margin + cxc * cell, margin + cyc * cell
        fill = "#111" if color == "B" else "#fafafa"
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="{fill}" '
                   f'fill-opacity="0.45" stroke="#e63946" stroke-width="2.5"/>')

    out.append('</svg>')
    return "".join(out)


# ---- lookup & SGF retrieval -------------------------------------------

def load_lookup(path=None):
    """Load lookup.txt → {tsumego_id: (zip_id, size)}."""
    if path is None:
        path = os.path.join(PROJECT_ROOT, "pwa", "lookup.txt")
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                out[parts[0]] = (parts[1], int(parts[2]))
    return out


def fetch_sgf(tsumego_id, zip_id=None, size=None):
    """The problem's SGF, read from the CORPUS — tsumegos/<size>/<tid>.sgf.

    NOT from pwa/tsumegos/<size>/<zip_id>.zip, which is what this used to do and
    which was backwards in two ways. The corpus is the ground truth every stage
    reads; the shards are one consumer's packaging of it, written by a step that
    runs LAST. Every caller here is an inspection tool, and the labelling workflow
    puts inspection BEFORE packaging — refit, eyeball the extremes, name the axes
    in labels.json, then package — so sourcing boards from the packager's output
    meant the tool that gates packaging could not run until packaging had already
    happened. Worse, it raced: shard ids are content-derived, so a rebuild in
    flight replaces the very files lookup.txt names and every board fails at once.

    Nothing is lost by reading the corpus: packaging copies these bytes verbatim
    into the zip (_zip_writestr with the file's own contents), so the two are
    identical for every purpose, and the corpus additionally covers problems that
    have not been packaged yet.

    `zip_id` is accepted and ignored, so existing callers need no change.
    """
    path = os.path.join(tsumego_dir(size), f"{tsumego_id}.sgf")
    with open(path, encoding="utf-8") as f:
        return f.read()


def render_problem(tid, zip_id, size):
    """Fetch + parse + simulate + render a single problem.

    Returns {svg, to_move ("Black"/"White"/"?"), board, correct, last_move,
    ko, has_ko}. Raises FileNotFoundError if the corpus SGF is missing.
    """
    sgf = fetch_sgf(tid, zip_id, size)
    sz, setup, correct = parse_sgf(sgf)
    board, last_move, ko = simulate(setup, sz)
    svg = render_svg(board, sz, correct, last_move, ko)
    to_move = {"B": "Black", "W": "White"}.get(correct[0] if correct else "?", "?")
    return {
        "svg": svg, "to_move": to_move, "board": board, "correct": correct,
        "last_move": last_move, "ko": ko, "has_ko": ko is not None,
    }
