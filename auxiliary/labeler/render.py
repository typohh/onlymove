"""Labeler board rendering = shared board renderer + rank label.

The board-position logic (SGF parse, capture/ko simulation, SVG rendering with
last-move / ko / correct-move markers) now lives in utils/board_render.py so it
is not duplicated across the inspection tools. This module only adds the rank
label the labeler shows on top.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.board_render import render_problem  # noqa: E402


# ---- rank label (mirrors rank-decomposition.js + i18n.getDescriptiveRank) ----

def interp_extrap(x, xs, ys):
    n = len(xs)
    if x <= xs[0]:
        return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0])
    if x >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        md = (lo + hi) // 2
        if xs[md] <= x:
            lo = md
        else:
            hi = md
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo])


def load_curve():
    with open(os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json")) as f:
        return json.load(f)


def rank_label(rating, size, curve):
    c = curve.get(str(size))
    if not c:
        return "?"
    idx = math.floor(interp_extrap(rating, c["ratings"], c["rank_indices"]))
    return f"{20 - idx} kyu" if idx < 20 else f"{idx - 19} dan"


def build(tid, zip_id, size, rating, curve):
    """Fetch + render a single problem. Returns {svg, to_move, rank, has_ko}."""
    rp = render_problem(tid, zip_id, size)
    return {
        "svg": rp["svg"],
        "to_move": rp["to_move"],
        "rank": rank_label(rating, size, curve),
        "has_ko": rp["has_ko"],
    }
