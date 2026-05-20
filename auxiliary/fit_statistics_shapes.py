#!/usr/bin/env python3
"""Fit candidate rank→rating shapes to the EGF even-game win-rate table in
statistics.txt — and check whether the PWA stop-gap exponential
(θ = A·exp(0.12·rank_index) + C) misfits, especially at the weak (DDK) end.

Signal
------
The "Winning Statistics - Even Games" table (top of statistics.txt) gives, for
each declared grade G and opponents n = 1..4 stones stronger, the wins Nw and
games Ng of the *weak* player. Under a Bradley-Terry / logistic win model with
a latent rating θ(x), x = rank_index (20k=0 … 1k=19 … 1d=20 … 6d=25):

    P(weak grade-x player beats an opponent n stones stronger)
        = σ( θ(x) − θ(x+n) )

θ is identified only up to an additive constant (only differences enter) and
its overall scale is absorbed into each candidate's parameters, so this fits
the *shape* of rating-vs-grade in natural logit units — directly comparable to
the PWA exponential's curvature. Candidates are fit by maximizing the binomial
log-likelihood over every table cell, then compared by total LL / AIC and,
crucially, by where their deviance and signed bias land (DDK / SDK / dan).

A signed residual (predicted_Pw − observed_Pw) > 0 means the model thinks those
grades are *closer together* than the data says — i.e. it compresses that part
of the scale. That is the exact symptom we are hunting for at the DDK end.

Usage:
  ./.venv/bin/python3 -m auxiliary.fit_statistics_shapes
  ./.venv/bin/python3 -m auxiliary.fit_statistics_shapes --plot
"""

import argparse
import os
import re
import sys

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.path.join(PROJECT_ROOT, "statistics.txt")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def rank_index(label):
    """20k → 0, 1k → 19, 1d → 20, 6d → 25 (the PWA rank_index axis)."""
    m = re.fullmatch(r"(\d{1,2})([kd])", label)
    n = int(m.group(1))
    scalar = (1 - n) if m.group(2) == "k" else n
    return scalar + 19


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


# ---------------------------------------------------------------------------
# Candidate shapes.  Each: (name, init, bounds, theta(x, params)).
# The additive constant of θ is dropped (only differences enter the model);
# every other degree of freedom — including overall scale — is free, so this
# is a fair shape-only comparison.
# ---------------------------------------------------------------------------

KNEE_6K = 14.0  # 6k  (rank_index for the 6k/5k boundary)
KNEE_1D = 20.0  # 1d  (rank_index for the 1k/1d boundary)


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


# ---------------------------------------------------------------------------
# Fit + diagnostics
# ---------------------------------------------------------------------------

def band_of(x):
    if x <= 10:
        return "DDK 20k-10k"
    if x <= 19:
        return "SDK  9k-1k"
    return "dan  1d-6d"


BANDS = ["DDK 20k-10k", "SDK  9k-1k", "dan  1d-6d"]


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
    return (params, theta_func). theta_func(x, params) takes rank_index x
    (20k=0 … 1k=19 … 6d=25) and returns θ in win-model logit units.

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


# Default named-rank anchors on the 0..28 axis (20k=0 … 1k=19 … 9d=28). These
# mirror the PWA stop-gap convention (humanv0_8k=12.5, 3k=17.5) so the cubic is
# a drop-in for the old exponential curve in both fit_rank_curves and
# package_for_pwa.
ANCHOR_X_8K = 12.5
ANCHOR_X_3K = 17.5


class CubicRatingCurve:
    """The statistics.txt-fit cubic rank shape pinned onto a rating scale.

    `forward(rank_index) -> rating` and `inverse(rating) -> rank_index`, where
    rank_index is the 0..28 named-rank axis (20k=0, 1k=19, 1d=20, 9d=28). The
    cubic SHAPE comes from fit_shape("cubic"); two ratings (`y_8k`, `y_3k` at
    rank_index `x_8k`/`x_3k`) pin its scale + offset, exactly the way the old
    exponential stop-gap was pinned — so this is a drop-in with a better DDK
    tail. Single source of truth for both the fit_rank_curves cubic prior and
    the package_for_pwa rating→rank mapping.

    The cubic is monotone over the whole plausible range, so `inverse` is a
    dense-grid linear interpolation (built lazily — forward-only callers, e.g.
    the per-rank comparison table, pay nothing for it)."""

    def __init__(self, y_8k, y_3k, x_8k=ANCHOR_X_8K, x_3k=ANCHOR_X_3K):
        self._params, self._theta = fit_shape("cubic")
        g8 = float(self._theta(np.array([x_8k]), self._params)[0])
        g3 = float(self._theta(np.array([x_3k]), self._params)[0])
        if g3 == g8:
            raise ValueError("degenerate cubic anchors (x_8k == x_3k)")
        self.scale = (y_3k - y_8k) / (g3 - g8)
        self.offset = y_8k - self.scale * g8
        self._grid_x = None
        self._grid_r = None

    def forward(self, rank_index):
        x = np.asarray(rank_index, dtype=float)
        return self.offset + self.scale * self._theta(x, self._params)

    def inverse(self, rating):
        if self._grid_x is None:
            # Covers ~40k (x=-20) .. ~17d (x=45); cubic is monotone so the
            # forward grid is strictly increasing and np.interp-invertible.
            # Wide enough that even the easiest/hardest tsumego ratings land
            # inside it rather than clamping at a grid end.
            self._grid_x = np.arange(-20.0, 45.0, 0.005)
            self._grid_r = self.forward(self._grid_x)
        return np.interp(np.asarray(rating, dtype=float),
                         self._grid_r, self._grid_x)


def cell_stats(theta, params, X, N, Nw, Ng):
    """Per-cell predicted Pw, binomial deviance, signed Pw residual."""
    z = theta(X, params) - theta(X + N, params)
    p = np.clip(expit(z), 1e-9, 1 - 1e-9)
    obs = Nw / Ng
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(Nw > 0, Nw * np.log(obs / p), 0.0)
        t2 = np.where(Ng - Nw > 0, (Ng - Nw) * np.log((1 - obs) / (1 - p)), 0.0)
    dev = 2.0 * (t1 + t2)
    resid = p - obs            # >0 ⇒ model over-predicts weak win ⇒ compresses
    return p, dev, resid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true",
                    help="Also write a PNG comparing shapes + residuals")
    args = ap.parse_args()

    cells = parse_even_table(STATS_PATH)
    if not cells:
        print(f"No even-game cells parsed from {STATS_PATH}")
        return
    X = np.array([c["x"] for c in cells], float)
    N = np.array([c["n"] for c in cells], float)
    Nw = np.array([c["Nw"] for c in cells], float)
    Ng = np.array([c["Ng"] for c in cells], float)
    bands = np.array([band_of(c["x"]) for c in cells])

    print(f"Parsed {len(cells)} (grade, n-stones) cells from statistics.txt "
          f"even-games table")
    print(f"  total games = {int(Ng.sum())}, "
          f"grades {cells[0]['grade']}..{cells[-1]['grade']}, n = 1..4 stones")
    for b in BANDS:
        msk = bands == b
        print(f"    {b}:  {int(msk.sum()):>2d} cells, "
              f"{int(Ng[msk].sum()):>6d} games")

    models = make_models()
    fitted = []
    for name, init, bounds, theta in models:
        params, ll = fit_model(theta, init, bounds, X, N, Nw, Ng)
        k = len(params)
        aic = 2 * k - 2 * ll
        p, dev, resid = cell_stats(theta, params, X, N, Nw, Ng)
        fitted.append({"name": name, "theta": theta, "params": params,
                       "ll": ll, "k": k, "aic": aic,
                       "dev": dev, "resid": resid})

    best_aic = min(f["aic"] for f in fitted)

    # ---- overall fit quality ------------------------------------------------
    print(f"\n{'='*78}")
    print("OVERALL FIT (binomial LL over all cells; ΔAIC vs best, lower = better)")
    print(f"{'='*78}")
    print(f"{'shape':<34}{'params':>7}{'logLik':>10}{'AIC':>9}{'ΔAIC':>8}")
    print("-" * 78)
    for f in sorted(fitted, key=lambda d: d["aic"]):
        pstr = ", ".join(f"{v:.3g}" for v in f["params"])
        print(f"{f['name']:<34}{f['k']:>7d}{f['ll']:>10.1f}"
              f"{f['aic']:>9.1f}{f['aic']-best_aic:>8.1f}")
        print(f"{'    └ '+pstr:<34}")

    # ---- where does the deviance land? --------------------------------------
    print(f"\n{'='*78}")
    print("DEVIANCE BY BAND  (sum of binomial deviance; higher = worse fit "
          "in that band)")
    print(f"{'='*78}")
    header = f"{'shape':<34}" + "".join(f"{b:>13}" for b in BANDS) + f"{'tot':>8}"
    print(header)
    print("-" * len(header))
    for f in sorted(fitted, key=lambda d: d["aic"]):
        line = f"{f['name']:<34}"
        for b in BANDS:
            line += f"{f['dev'][bands == b].sum():>13.1f}"
        line += f"{f['dev'].sum():>8.1f}"
        print(line)

    # ---- signed bias by band (the compression diagnostic) -------------------
    print(f"\n{'='*78}")
    print("GAME-WEIGHTED SIGNED BIAS BY BAND  (Σ Ng·(pred_Pw − obs_Pw) / Σ Ng)")
    print("  >0 ⇒ model predicts the weak side wins more often than reality ⇒")
    print("       it places those grades too CLOSE together (compresses them).")
    print(f"{'='*78}")
    print(f"{'shape':<34}" + "".join(f"{b:>13}" for b in BANDS))
    print("-" * (34 + 13 * len(BANDS)))
    for f in sorted(fitted, key=lambda d: d["aic"]):
        line = f"{f['name']:<34}"
        for b in BANDS:
            msk = bands == b
            wbias = np.sum(Ng[msk] * f["resid"][msk]) / Ng[msk].sum()
            line += f"{wbias*100:>+12.2f}%"
        print(line)

    # ---- consecutive-grade spacing: empirical vs each model -----------------
    # Uses only the n=1 cells: emp spacing(x) = logit of weak-loss odds.
    n1 = {c["x"]: c for c in cells if c["n"] == 1}
    xs = sorted(n1)
    print(f"\n{'='*78}")
    print("CONSECUTIVE-GRADE θ-SPACING  θ(x+1)−θ(x), logit units  (from n=1 col)")
    print("  emp = log((1−Pw)/Pw) of the weak-vs-+1-stone cell.")
    print(f"{'='*78}")
    short = {"linear  a·x": "lin", "exp k=0.12  (current PWA shape)": "exp.12",
             "exp  A·e^(k·x)  (k free)": "expfit", "quadratic  a·x + b·x²": "quad",
             "cubic  a·x + b·x² + d·x³": "cubic",
             "piecewise 2-slope @6k": "pw6k",
             "piecewise 2-slope @1d": "pw1d"}
    cols = [short[f["name"]] for f in fitted]
    print(f"{'grade':>6}{'n':>6}{'emp':>8}" + "".join(f"{c:>8}" for c in cols))
    print("-" * (20 + 8 * len(cols)))
    for x in xs:
        c = n1[x]
        obs = c["Nw"] / c["Ng"]
        emp = np.log((1 - obs) / obs) if 0 < obs < 1 else float("nan")
        line = f"{c['grade']:>6}{c['Ng']:>6d}{emp:>8.3f}"
        for f in fitted:
            sp = float(f["theta"](np.array([x + 1.0]), f["params"])[0]
                       - f["theta"](np.array([x]), f["params"])[0])
            line += f"{sp:>8.3f}"
        print(line)

    print(f"\nReading: a pure exponential forces spacing to grow by a constant "
          f"factor e^k each\ngrade. If the empirical spacing is roughly flat "
          f"through the kyu range and only\nrises in the dan range, the "
          f"exponential must under-spread (compress) the weak\nend to pay for "
          f"its dan-range acceleration — visible as a positive DDK bias.")

    if args.plot:
        make_plot(cells, fitted, n1, xs, short)


def make_plot(cells, fitted, n1, xs, short):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[plot skipped: matplotlib unavailable: {e}]")
        return

    # Empirical cumulative θ from n=1 spacings, anchored θ(8k)=0 (x=11).
    all_x = np.arange(0, 26)
    emp_theta = {}
    cum = 0.0
    emp_theta[min(xs)] = 0.0
    for x in range(min(xs), max(xs)):
        c = n1.get(x)
        if c is None:
            sp = np.nan
        else:
            obs = c["Nw"] / c["Ng"]
            sp = np.log((1 - obs) / obs) if 0 < obs < 1 else np.nan
        cum += sp
        emp_theta[x + 1] = cum
    ex = np.array([x for x in emp_theta if np.isfinite(emp_theta[x])])
    ey = np.array([emp_theta[x] for x in ex])
    anchor = 11.0
    ey = ey - np.interp(anchor, ex, ey)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ex, ey, "ko", label="empirical (cumsum of n=1 log-odds)")
    for f in fitted:
        ty = f["theta"](all_x.astype(float), f["params"])
        ty = ty - f["theta"](np.array([anchor]), f["params"])[0]
        ax1.plot(all_x, ty, label=short[f["name"]])
    ax1.axvspan(0, 10, color="orange", alpha=0.08)
    ax1.set_xlabel("rank_index (20k=0, 1k=19, 1d=20, 6d=25)")
    ax1.set_ylabel("θ (logit units, anchored θ(8k)=0)")
    ax1.set_title("Rank→rating shape vs EGF even-game data")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    X = np.array([c["x"] for c in cells], float)
    bands_x = sorted(set(int(x) for x in X))
    width = 0.8 / len(fitted)
    for i, f in enumerate(fitted):
        per_grade = []
        for gx in bands_x:
            m = X == gx
            per_grade.append(f["dev"][m].sum())
        ax2.bar(np.array(bands_x) + i * width, per_grade, width,
                label=short[f["name"]])
    ax2.axvspan(-0.5, 10.5, color="orange", alpha=0.08)
    ax2.set_xlabel("rank_index (weak player's grade)")
    ax2.set_ylabel("binomial deviance (summed over n)")
    ax2.set_title("Per-grade misfit (shaded = DDK)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    out = os.path.join(PROJECT_ROOT, "auxiliary", "statistics_shape_fit.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
