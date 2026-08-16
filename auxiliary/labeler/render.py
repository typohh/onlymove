"""Labeler board rendering = shared board renderer + rank label.

The board-position logic (SGF parse, capture/ko simulation, SVG rendering with
last-move / ko / correct-move markers) lives in utils/katago/board_render.py so it
is not duplicated across the inspection tools. This module only adds the rank
label the labeler shows on top.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.katago.board_render import render_problem  # noqa: E402
from utils.rating.fit_rank_curves import interp_extrap  # noqa: E402
from utils.rating.rank_surveys import value_to_rank  # noqa: E402


# ---- rank label -------------------------------------------------------

def load_curve():
    with open(os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json")) as f:
        return json.load(f)


def rank_label(rating, size, curve):
    c = curve.get(str(size))
    if not c:
        return "?"
    lab = value_to_rank(interp_extrap(rating, c["ratings"], c["ranks"]))
    return lab.replace("k", " kyu").replace("d", " dan")


def build(tid, zip_id, size, rating, curve):
    """Fetch + render a single problem. Returns {svg, to_move, rank, has_ko}."""
    rp = render_problem(tid, zip_id, size)
    return {
        "svg": rp["svg"],
        "to_move": rp["to_move"],
        "rank": rank_label(rating, size, curve),
        "has_ko": rp["has_ko"],
    }
