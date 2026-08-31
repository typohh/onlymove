#!/usr/bin/env python3
"""Extremes of the published per-problem parameters, and the floor's approach to
the calibration probability.

Reads `pwa/tsumegos/problem_params.bin` — the (rating, slope, floor) triple per
published problem, the same file the Worker's leaderboard recompute and the
client's ratings-change backfill read.

The 80%-clear rating exists only while a problem's floor sits below the
calibration probability: at or above it the problem is cleared at least four
times in five by any ability, so no ability brings it down to 80% and the
derivation's epsilon clip returns a large negative SENTINEL in place of a rating.
Nothing bars the fit from that region — the floor's structural ceiling is
1 − 2·lapse, far above the calibration probability — so how far in it goes is a
property of a particular fit and is measured here rather than assumed: the count
above the threshold, the rank span those problems land in (which is where a
sentinel is visible, since it drops them to the bottom of the published scale),
the count in the approach band just below the threshold, and the floor at the
99.9th percentile, which is what says the bulk of the set is nowhere near it.

The read checks what every consumer assumes: 16-byte rows, finite values, floors
inside [0, 1), slopes strictly positive, and a row for exactly the problems
`lookup.txt` publishes. The rank span goes through the PUBLISHED curve, the one
the client applies.

    ./.venv/bin/python3 -m ablations.published_params_figures
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.rating.fit_rank_curves import FLOOR_MAX  # noqa: E402
from utils.rating.model_constants import CALIBRATION_P, SLOPE_FLOOR  # noqa: E402
from utils.rating.ratings_io import apply_M, load_M, read_problem_params  # noqa: E402

TIER = "methodology"

PWA_DIR = os.path.join(PROJECT_ROOT, "pwa")
TSUMEGOS_DIR = os.path.join(PWA_DIR, "tsumegos")
PARAMS_PATH = os.path.join(TSUMEGOS_DIR, "problem_params.bin")
LOOKUP_PATH = os.path.join(PWA_DIR, "lookup.txt")

# The approach band: floors this close to the calibration probability without
# crossing it. Its width is a reporting choice, stated here so the figure and the
# sentence quoting it cannot drift apart.
NEAR_CALIBRATION_LO = 0.75


def measure():
    p = read_problem_params(PARAMS_PATH)
    n = len(p["id"])
    require(n > 0, "problem_params.bin is empty")
    for col in ("rating", "slope", "floor"):
        require(np.all(np.isfinite(p[col])), f"problem_params.bin: non-finite {col}")
    require(np.all((p["floor"] >= 0.0) & (p["floor"] < 1.0)), "problem_params.bin: floor outside [0, 1)")
    require(np.all(p["slope"] > 0.0), "problem_params.bin: non-positive slope")
    require(os.path.exists(LOOKUP_PATH), f"no {LOOKUP_PATH}")
    with open(LOOKUP_PATH) as f:
        published = {line.split()[0] for line in f if line.strip()}
    require(published == set(p["id"]), "problem_params.bin ids differ from lookup.txt")

    above = p["floor"] > CALIBRATION_P
    near = (p["floor"] >= NEAR_CALIBRATION_LO) & ~above
    # The client clamps floors at FLOOR_MAX, just under the calibration
    # probability, to keep the response curve's offset finite. A floor in
    # (FLOOR_MAX, CALIBRATION_P] is clamped despite its rating being perfectly
    # defined — the case the "catches the sentinel problems and nothing else"
    # claim asserts empty, which only holds while this count is zero.
    between = (p["floor"] > FLOOR_MAX) & ~above
    m_ratings, m_internal = load_M(TSUMEGOS_DIR)
    require(len(m_ratings) >= 2, "the published rating→rank curve has too few knots")
    ranks = np.array([apply_M(x, m_ratings, m_internal) for x in p["rating"][above]]) \
        if above.any() else np.empty(0)
    return {
        "n": n,
        "floor_max": float(p["floor"].max()),
        "floor_p999": float(np.percentile(p["floor"], 99.9)),
        "n_above": int(above.sum()),
        "n_near": int(near.sum()),
        "n_between_clamp": int(between.sum()),
        "rank_weakest": float(ranks.min()) if ranks.size else float("nan"),
        "rank_strongest": float(ranks.max()) if ranks.size else float("nan"),
        "slope_min": float(p["slope"].min()),
        "rating_min": float(p["rating"].min()),
        "rating_max": float(p["rating"].max()),
    }


def figures():
    m = measure()
    figs = [
        Figure("published.floor_max", m["floor_max"], ".3f",
               "largest guessing floor among the published problems"),
        Figure("published.floor_p999", m["floor_p999"], ".2f",
               "guessing floor at the 99.9th percentile of the published set"),
        Figure("published.floor_above_calibration", m["n_above"], "d",
               f"published problems whose floor exceeds the calibration probability "
               f"{CALIBRATION_P}, so their rating is a sentinel rather than an 80%-clear point"),
        Figure("published.floor_near_calibration", m["n_near"], "d",
               f"published problems with a floor in [{NEAR_CALIBRATION_LO}, {CALIBRATION_P}] — "
               f"the approach band, below the threshold"),
        Figure("published.floor_between_clamp_and_calibration", m["n_between_clamp"], "d",
               f"published problems with a floor in ({FLOOR_MAX}, {CALIBRATION_P}] — clamped "
               f"by the client despite a defined rating; the clamp catches only the "
               f"sentinel problems exactly while this is zero"),
        Figure("published.slope_min", m["slope_min"], ".2f",
               "smallest published discrimination slope"),
    ]
    if m["n_above"]:
        figs += [
            Figure("published.floor_above_calibration_rank_weakest", m["rank_weakest"], "rank",
                   "weakest rank a sentinel rating maps to through the published curve"),
            Figure("published.floor_above_calibration_rank_strongest", m["rank_strongest"], "rank",
                   "strongest rank a sentinel rating maps to"),
        ]
    return figs


def main():
    from utils.rating.rank_surveys import value_to_rank
    m = measure()
    print(f"{m['n']:,d} published problems\n")
    print(f"{'floor max':>26}{m['floor_max']:>10.4f}")
    print(f"{'floor p99.9':>26}{m['floor_p999']:>10.4f}")
    print(f"{'slope min':>26}{m['slope_min']:>10.4f}   "
          f"{m['slope_min'] / SLOPE_FLOOR:.0f}x the {SLOPE_FLOOR} clamp")
    print(f"{'rating range':>26}{m['rating_min']:>10.2f} .. {m['rating_max']:.2f}")
    print(f"\nfloor vs the {CALIBRATION_P} calibration probability")
    print(f"{'in [' + str(NEAR_CALIBRATION_LO) + ', ' + str(CALIBRATION_P) + ']':>26}"
          f"{m['n_near']:>10,d}   approaching, rating still defined")
    print(f"{'in (' + str(FLOOR_MAX) + ', ' + str(CALIBRATION_P) + ']':>26}"
          f"{m['n_between_clamp']:>10,d}   clamped by the client with a defined rating")
    print(f"{'above it':>26}{m['n_above']:>10,d}   rating is a SENTINEL "
          f"(epsilon clip), landing at "
          f"{value_to_rank(m['rank_weakest'])}..{value_to_rank(m['rank_strongest'])}")


if __name__ == "__main__":
    main()
