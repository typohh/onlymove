"""Rank-scale twins: the curve interpolator, the band and label conversions, the
server couplings, and the per-attempt transforms the SE floor is built in.

  interp_extrap           utils/rating/ratings_io.py  (scalar)
  interp_extrap           utils/rating/se_floor.py    (vectorized)
  interpExtrap            pwa/shared/rank-math.js
  apply_M                 utils/rating/ratings_io.py  <->  applyM (pwa/shared/rank-window.js)
  rank_band, rank_to_value, value_to_rank, gor_to_value, aga_rating_to_value
                          utils/rating/rank_surveys.py
  rankBand, rankLabelToRank, rankToLabel   pwa/shared/rank-math.js
  gorToRank, agaRatingToRank               pwa/src/leaderboard.js
  anchor_info, rank_space_slope            utils/rating/se_floor.py
  anchorInfo (pwa/js/irt.js), slopeInRankSpace (pwa/shared/rank-window.js)

The interpolators evaluate the same formula in a different operation order, so
floats compare to a relative 1e-12 (1e-9 for the slope, a finite difference
of them); everything that yields a band, a label or
an integer compares exactly.
"""

import math
import random

import numpy as np
import pytest

from pipeline import fit_population_constants
from utils.rating import fit_rank_curves, se_floor
from utils.rating.ratings_io import apply_M, interp_extrap
from utils.rating.rank_surveys import (aga_rating_to_value, gor_to_value, rank_band,
                                       rank_to_value, value_to_rank)

RANK_MATH = "pwa/shared/rank-math.js"
RANK_WINDOW = "pwa/shared/rank-window.js"
IRT = "pwa/js/irt.js"
LEADERBOARD = "pwa/src/leaderboard.js"


def _close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _curves():
    """Random monotone curves shaped like M: a few to a few dozen knots."""
    rng = random.Random(17)
    curves = []
    for _ in range(40):
        n = rng.randint(2, 30)
        xs = sorted(rng.sample(range(-2000, 2000), n))
        xs = [x / 1000 for x in xs]
        ys = np.cumsum([rng.uniform(0.05, 3.0) for _ in range(n)]) - 20
        curves.append((xs, ys.tolist()))
    return curves


def _probe_points(xs, rng):
    lo, hi = xs[0], xs[-1]
    pts = list(xs)                                          # exactly on knots
    pts += [rng.uniform(lo, hi) for _ in range(20)]         # interior
    pts += [lo - rng.uniform(0, 3) for _ in range(5)]       # extrapolated low
    pts += [hi + rng.uniform(0, 3) for _ in range(5)]       # extrapolated high
    return pts


def test_interp_extrap_agrees(js):
    rng = random.Random(1)
    calls, expected = [], []
    for xs, ys in _curves():
        pts = _probe_points(xs, rng)
        vec = se_floor.interp_extrap(pts, xs, ys)
        for x, v in zip(pts, vec):
            calls.append((RANK_MATH, "interpExtrap", (x, xs, ys)))
            expected.append((x, interp_extrap(x, xs, ys), float(v)))
    got = js.run(calls)
    for (x, scalar, vectorized), j in zip(expected, got):
        assert _close(j, scalar), ("ratings_io", x, j, scalar)
        assert _close(j, vectorized), ("se_floor", x, j, vectorized)


def test_apply_m_agrees(js):
    rng = random.Random(2)
    calls, expected = [], []
    for xs, ys in _curves():
        for r in _probe_points(xs, rng):
            calls.append((RANK_WINDOW, "applyM", (r, {"ratings": xs, "ranks": ys})))
            expected.append(apply_M(r, np.array(xs), np.array(ys)))
    for e, j in zip(expected, js.run(calls)):
        assert _close(j, e)


BAND_VALUES = [-30.5, -12.0, -1.0, -0.5, -1e-9, -0.0, 0.0, 1e-9, 0.5, 0.999999, 1.0,
               2.5, 8.0, 8.999, 11.3, -45.25]


def test_rank_band_and_label_agree(js):
    got = js.run([(RANK_MATH, "rankBand", (v,)) for v in BAND_VALUES]
                 + [(RANK_MATH, "rankToLabel", (v,)) for v in BAND_VALUES])
    bands, labels = got[:len(BAND_VALUES)], got[len(BAND_VALUES):]
    for v, b, lab in zip(BAND_VALUES, bands, labels):
        assert b == rank_band(v), v
        assert lab == value_to_rank(v), v


LABELS = ["1k", "5k", "30k", "1d", "3d", "9d", "12d", "01k", " 3D ", "3D", "5 k",
          "0k", "0d", "100k", "k", "", "1p", "abc", "3dd"]


def test_rank_label_to_value_agrees(js):
    got = js.run([(RANK_MATH, "rankLabelToRank", (s,)) for s in LABELS])
    for s, j in zip(LABELS, got):
        assert j == rank_to_value(s), repr(s)


SERVER_VALUES = [2000.0, 2100.0, 2389.0, 2389.5, 1.0, 0.0, -50.0, 2750.0, math.nan,
                 math.inf, -math.inf]
AGA_VALUES = [5.3, 5.0, 1.0, 0.999, 0.0, -0.5, -1.0, -1.001, -12.7, 9.9, math.nan,
              math.inf]


def test_server_couplings_agree(js):
    got = js.run([(LEADERBOARD, "gorToRank", (g,)) for g in SERVER_VALUES]
                 + [(LEADERBOARD, "agaRatingToRank", (a,)) for a in AGA_VALUES])
    gor, aga = got[:len(SERVER_VALUES)], got[len(SERVER_VALUES):]
    for g, j in zip(SERVER_VALUES, gor):
        assert _close(j, gor_to_value(g)), g
    for a, j in zip(AGA_VALUES, aga):
        py = aga_rating_to_value(a)
        # A non-finite rating is refused on both sides; Python returns None,
        # and so must the client.
        assert _close(j, py), a


def test_floor_and_calibration_constants_agree(js):
    floor_max, calibration_p = js.run([(IRT, "FLOOR_MAX", None), (IRT, "CALIBRATION_P", None)])
    assert fit_rank_curves.FLOOR_MAX == floor_max
    assert fit_population_constants.FLOOR_MAX == floor_max
    assert se_floor._FLOOR_MAX == floor_max
    assert fit_rank_curves.CALIBRATION_P == calibration_p
    assert se_floor._CALIBRATION_P == calibration_p


def test_anchor_info_agrees(js):
    floors = [-0.2, 0.0, 0.05, 0.2, 0.37, 0.5, 0.7, 0.77, 0.78, 0.79, 0.95, 1.0]
    lapse = js.const(IRT, "LAPSE")
    got = js.run([(IRT, "anchorInfo", (c,)) for c in floors])
    for c, j in zip(floors, got):
        assert _close(j, float(se_floor.anchor_info(c, lapse))), c


def test_rank_space_slope_agrees(js):
    rng = random.Random(4)
    calls, expected = [], []
    for xs, ys in _curves():
        curve = {"ratings": xs, "ranks": ys}
        for r in _probe_points(xs, rng)[:12]:
            # Includes slopes that hit both clamp ends of the slope range.
            for a in (0.01, 0.4, 1.6, 2.5, 40.0):
                calls.append((RANK_WINDOW, "slopeInRankSpace", (r, a, curve)))
                expected.append(float(se_floor.rank_space_slope(np.array([r]), a, xs, ys)[0]))
    # The slope is a central difference over h = 1e-3, which scales the
    # interpolators' last-bit disagreement by 1/(2h).
    for e, j in zip(expected, js.run(calls)):
        assert _close(j, e, tol=1e-9)


@pytest.mark.parametrize("module,name", [(RANK_MATH, "interpExtrap"), (RANK_WINDOW, "applyM")])
def test_extrapolation_is_unclamped(js, module, name):
    """The rank is never clamped: far past the end knots, both sides follow the
    end slopes rather than pinning to the end values."""
    xs, ys = [0.0, 1.0, 2.0], [0.0, 1.0, 3.0]
    args = (50.0, xs, ys) if name == "interpExtrap" else (50.0, {"ratings": xs, "ranks": ys})
    assert js.call(module, name, *args) == interp_extrap(50.0, xs, ys) == 99.0
