"""Fit a named candidate rating→rank SHAPE to KataGo's even-game win-rate table.

`statistics.txt` is KataGo's tabulated weak-side win rate for even games at each rank gap; the
ΔAIC-best fit to it gives the rating→rank curve's SHAPE (the EGF cubic), with only the affine
(α, β) left for the survey fit downstream. The rank-calibration stage (pipeline.map_rating_to_rank
via utils.fit_rank_curves) and the auxiliary rank-mapping experiments both reuse the same fitted
shape, so the fit lives here (a dependency-free leaf — no auxiliary/ or pipeline/ imports).

`fit_shape(name)` returns (params, theta_func); `name` is matched as a prefix of the families in
make_models(), so "cubic" picks the cubic family. theta_func(x, params) maps this table's own
axis x (20k=0 … 1k=19 … 6d=25; see rank_to_axis) to θ in win-model logit units.
"""

import os
import re
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from utils.rank_surveys import rank_to_value

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.path.join(PROJECT_ROOT, "data", "statistics.txt")

# This table has its OWN axis: 20k = 0, 1k = 19, 1d = 20. It is deliberately NOT
# the project rank scale (pwa/shared/rank-math.js), because every candidate shape
# below is a polynomial with no constant term — the ORIGIN is part of the model, so
# translating x would refit a different curve. Callers convert at the boundary with
# rank_to_axis(); nothing else in the project speaks this coordinate.
AXIS_ZERO_RANK = -19.5   # the project rank this axis calls 0 (the 20k band centre)

KNEE_6K = 14.0  # 6k  (axis position of the 6k/5k boundary)
KNEE_1D = 20.0  # 1d  (axis position of the 1k/1d boundary)


def rank_to_axis(rank):
    """Project rank (rank_surveys.rank_to_value convention) -> this table's axis."""
    return np.asarray(rank, dtype=float) - AXIS_ZERO_RANK


def rank_index(label):
    """Kyu/dan label -> this table's axis: 20k → 0, 1k → 19, 1d → 20, 6d → 25."""
    return float(rank_to_axis(rank_to_value(label)))


def parse_even_table(path):
    """Return cells [{grade, x, n, Nw, Ng}] from the first (even-games,
    weak-side win%) table only.

    The table is ragged: at the sparse high-grade/high-n corner the Nw column
    is sometimes blank (only Ng and Pw printed). We tokenize each row and walk
    left→right: every group ends at the Pw token (the one containing '.'); the
    integers seen since the previous group are [Nw, Ng] when two are present,
    or [Ng] alone when Nw was omitted (then Nw is recovered from Pw·Ng)."""
    cells = []
    started = False
    with open(path) as f:
        for line in f:
            if "Winning Statistics - Even Games" in line:
                started = True
                continue
            if not started:
                continue
            if "Statistics of Even Games" in line:
                break  # reached the next section
            m = re.match(r"\s*(\d{1,2}[kd])\b(.*)", line)
            if not m:
                continue
            grade, rest = m.group(1), m.group(2)
            x = rank_index(grade)
            ints = []
            n = 0
            for tok in rest.split():
                if "." in tok:
                    pw = float(tok)
                    n += 1
                    if len(ints) >= 2:
                        Nw, Ng = ints[-2], ints[-1]
                    elif len(ints) == 1:
                        Ng = ints[-1]
                        Nw = int(round(pw * Ng / 100.0))
                    else:
                        ints = []
                        continue
                    if Ng > 0:
                        cells.append({"grade": grade, "x": x, "n": n,
                                      "Nw": Nw, "Ng": Ng})
                    ints = []
                else:
                    ints.append(int(tok))
    return cells


def _piecewise(knee):
    """Continuous 2-slope curve: slope p[0] up to `knee`, slope p[1] after."""
    return lambda x, p: (p[0] * np.minimum(x, knee)
                         + p[1] * np.maximum(0.0, x - knee))


def make_models():
    return [
        ("linear  a·x",
         [0.3], [(1e-4, 2.0)],
         lambda x, p: p[0] * x),
        ("exp k=0.12  (current PWA shape)",
         [0.1], [(1e-4, 100.0)],
         lambda x, p: p[0] * np.exp(0.12 * x)),
        ("exp  A·e^(k·x)  (k free)",
         [0.1, 0.12], [(1e-4, 100.0), (0.005, 0.4)],
         lambda x, p: p[0] * np.exp(p[1] * x)),
        ("quadratic  a·x + b·x²",
         [0.3, 0.0], [(-1.0, 2.0), (-0.05, 0.05)],
         lambda x, p: p[0] * x + p[1] * x * x),
        ("cubic  a·x + b·x² + d·x³",
         [0.3, 0.0, 0.0], [(-1.0, 2.0), (-0.1, 0.1), (-0.01, 0.01)],
         lambda x, p: p[0] * x + p[1] * x * x + p[2] * x ** 3),
        ("piecewise 2-slope @6k",
         [0.2, 0.4], [(0.0, 2.0), (0.0, 2.0)], _piecewise(KNEE_6K)),
        ("piecewise 2-slope @1d",
         [0.2, 0.4], [(0.0, 2.0), (0.0, 2.0)], _piecewise(KNEE_1D)),
    ]


def fit_model(theta, init, bounds, X, N, Nw, Ng):
    def neg_ll(params):
        z = theta(X, params) - theta(X + N, params)
        p = np.clip(expit(z), 1e-9, 1 - 1e-9)
        return -float(np.sum(Nw * np.log(p) + (Ng - Nw) * np.log(1 - p)))

    res = minimize(neg_ll, init, method="L-BFGS-B", bounds=bounds)
    return res.x, -res.fun


@lru_cache(maxsize=None)
def fit_shape(model_name, stats_path=STATS_PATH):
    """Fit one named candidate shape to the statistics.txt even-game table and
    return (params, theta_func). theta_func(x, params) takes this table's own axis x
    (20k=0 … 1k=19 … 6d=25; see rank_to_axis) and returns θ in win-model logit units.

    Lets other modules reuse a fitted shape without re-implementing the fit —
    e.g. fit_rank_curves's `--prior cubic`. `model_name` is matched as a prefix
    of the names in make_models() (so "cubic" picks the cubic family). Cached:
    callers in hot loops (per-rank table cells) hit the same fit, not a refit."""
    cells = parse_even_table(stats_path)
    if not cells:
        raise RuntimeError(f"no even-game cells parsed from {stats_path}")
    X = np.array([c["x"] for c in cells], float)
    N = np.array([c["n"] for c in cells], float)
    Nw = np.array([c["Nw"] for c in cells], float)
    Ng = np.array([c["Ng"] for c in cells], float)
    for name, init, bounds, theta in make_models():
        if name.startswith(model_name):
            params, _ = fit_model(theta, init, bounds, X, N, Nw, Ng)
            return params, theta
    raise ValueError(f"no candidate shape matching {model_name!r}")
