#!/usr/bin/env python3
"""Recreate a single user's *client Rank-tab graph* offline, under a given
ranking system, and print monthly rank distributions.

What "client graph" means
--------------------------
The PWA's Rank tab plots the user's rank over time as the **skill curve S(t)**
from `pwa/js/rank-decomposition.js` — a two-sided, 30-day-Gaussian-kernel pooled
4PL MLE in EGF-anchored INTERNAL-RANK space, computed on the user's NEAR-RANK
attempts only (the far-from-rank near-filter; see below). That curve — not the
front-page momentum number (`momentum-rank.js`) — is the literal graph the user
sees. This script ports it byte-for-faithful and reports, per calendar month,
the distribution of that recreated rank.

Server views (`--server`)
--------------------------
The Rank tab can show the line as it would read ON A SERVER. `--server OnlyMove`
(default) is the raw internal rank. `--server EGF` (or Fox/GoQuest/OGS/… from
rank_map.json) applies, exactly as `stats-ui.js::buildRatingHistoryChart` does:
  1. a TIME ADJUSTMENT (untimed practice here vs timed server play): per point
     delta = alpha·(log10(med_solve_min(t)) − (g0 + g1·S(t)))  [RATING units],
     med_solve_min(t) the 30-day-kernel median solve time, shrunk toward the
     whole-history median in log space; alpha/g0/g1 = population_constants.time_adjust;
  2. a rank-map translation: S' = M(M⁻¹(S) + delta), then through that server's
     coupling curve. EGF is the identity hub, so "EGF" vs "OnlyMove" differ ONLY
     by the time adjustment.

Near-rank filter on/off (`--both-filters`)
------------------------------------------
The official client curve drops far-from-rank attempts (a strong user's lucky
above-rank clears over-credit skill; the guessing floor inflates far-below
clears). `--both-filters` overlays BOTH the filtered (official) and unfiltered
curves on one graph / in the monthly table, so the filter's effect is visible.

"With a given ranking system"
-----------------------------
Every per-attempt feature (x = M(rating), slope = a_p/|M'|, floor) is looked up
from a ranking-system dir (`--dir`, default `ratings/`) + its
`rating_to_rank.json` + `rank_map.json`, and the fat-finger LAPSE from
`pwa/tsumegos/population_constants.json`. Point them elsewhere to re-render the
same user under a different ranking system.

Faithful to the client (ports of irt.js + rank-decomposition.js + stats-ui.js):
the 4PL anchored at P=0.8, per-attempt features through each attempt's own size
curve, first-time attempts only (repeats/linked excluded), pooled across sizes;
near filter |slope·(S0−x)| ≤ logit(0.95)−logit(0.8) ≈ 1.56. The wide kernel is
smooth at daily scale, so S0/S are evaluated on a daily grid (vectorised numpy)
and interpolated to attempt times — precisely how the chart polyline is drawn.

Reads the LOCAL DB copy (`attempts/database.sqlite`; see CLAUDE.md). No network.

Usage (the user id is positional and mandatory; pick one from the local DB):
  ./.venv/bin/python3 -m auxiliary.recreate_rank_graph <id>
  ./.venv/bin/python3 -m auxiliary.recreate_rank_graph <id> --server EGF --both-filters --plot g.png
"""

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.fit_rank_curves import interp_extrap  # noqa: E402
from utils.ratings_io import load_size_value_maps  # noqa: E402

# --- client constants (ports of pwa/js/{irt,rank-decomposition,rank-translation}.js) ---
SIZES = ("9", "13")
RANK_BAND_LO, RANK_BAND_HI = -30.0, 8.0   # labelled ladder: 30k .. 9d
CALIBRATION_P = 0.8
LOGIT_CALIBRATION = math.log(CALIBRATION_P / (1 - CALIBRATION_P))
FLOOR_MAX = 0.78
SLOPE_LO, SLOPE_HI = 0.05, 3.0
MSLOPE_FLOOR = 1e-3
LOGIT_PER_RATING = 1.6           # fallback per-attempt slope (legacy/no a_p)
LOGIT_PER_RANK_FALLBACK = 0.28
X_CLAMP_LO, X_CLAMP_HI = -15.0, 45.0

HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS
TAU_S_MS = 30 * DAY_MS            # skill (slow) kernel — the rank-tab wide kernel
KERNEL_TRUNC_SIGMA = 4
MIN_ATTEMPTS = 20
SUPPORT_EFFN = 10                 # effN floor for "supported" (line drawn)
# Local-linear wide-kernel slope ridge (ranks per τ_S); see Smoother.at(linear=).
# Mirrors SLOPE_PRIOR_SD in pwa/js/rank-decomposition.js (tuned via
# auxiliary/eval_rank_endpoint_bias.py).
SLOPE_PRIOR_SD = 2.0

NEAR_RANK_P_HI = 0.95
LOGIT_80 = math.log(0.8 / 0.2)
NEAR_RANK_L = math.log(NEAR_RANK_P_HI / (1 - NEAR_RANK_P_HI)) - LOGIT_80  # ~1.558

# time adjustment (stats-ui.js makeMedianSolveMinutesAt / population-constants.js)
TIME_ADJUST_AFK_CAP_MS = 180000
TIME_ADJUST_MIN_TIMED = 20
TIME_ADJUST_TAU_MS = 30 * DAY_MS
TIME_ADJUST_SHRINK_ESS = 8

ONLYMOVE = "OnlyMove"
DB_DEFAULT = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
V2_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "ratings")
POP_DEFAULT = os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "population_constants.json")


# ---------------------------------------------------------------------------
# 3PL/4PL + rank math (ports of irt.js + shared/rank-math.js)
# ---------------------------------------------------------------------------
def load_pop_constants(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def lapse_from(pc):
    ff = pc.get("fat_finger_rate")
    if isinstance(ff, (int, float)) and 0 <= ff < 1 - FLOOR_MAX:
        return float(ff)
    return 0.0026


def clamp_floor(c):
    if not isinstance(c, (int, float)):
        return 0.0
    return min(max(float(c), 0.0), FLOOR_MAX)


def clear_kappa(c, lapse):
    cc = max(c, 0.0)
    q = min(max((CALIBRATION_P - cc) / (1 - cc - lapse), 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def scalar_to_label(v):
    """rank -> label (rankToLabel; 0 is the 1k/1d boundary, integers are edges)."""
    b = math.floor(v)
    return f"{-b}k" if b < 0 else f"{b + 1}d"


def index_to_label(idx):
    return scalar_to_label(idx)


def parse_ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00"))
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def month_of(t_ms):
    return datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Ranking system: per-problem 3PL params + M curve + server couplings
# ---------------------------------------------------------------------------
class RankingSystem:
    def __init__(self, v2_dir, lapse):
        self.lapse = lapse
        self.ratings, self.slopes, self.floors, self.curve = {}, {}, {}, {}
        rr = json.load(open(os.path.join(v2_dir, "rating_to_rank.json")))
        for size in SIZES:
            self.ratings[size], self.slopes[size], self.floors[size] = \
                load_size_value_maps(v2_dir, size, missing_ok=True)
            if size in rr:
                self.curve[size] = (rr[size]["ratings"], rr[size]["ranks"])
        # one shared curve (both keys identical); used for the time-adjust remap
        self.curve9 = self.curve.get("9") or next(iter(self.curve.values()))
        try:
            self.rank_map = json.load(open(os.path.join(v2_dir, "rank_map.json")))
        except (OSError, ValueError):
            self.rank_map = {}

    def _m_slope(self, rating, xs, ys):
        h = 1e-3
        return (interp_extrap(rating + h, xs, ys) - interp_extrap(rating - h, xs, ys)) / (2 * h)

    def features(self, tid, size):
        """(x, slope_in_rank, floor, kappa) or None — port of flattenAttempts."""
        sz = str(size)
        if sz not in self.curve or sz not in self.ratings or tid not in self.ratings[sz]:
            return None
        xs, ys = self.curve[sz]
        rating = self.ratings[sz][tid]
        a_p = self.slopes[sz].get(tid)
        if a_p is None or a_p <= 0:
            a_p = LOGIT_PER_RATING
        x = interp_extrap(rating, xs, ys)
        denom = max(abs(self._m_slope(rating, xs, ys)), MSLOPE_FLOOR)
        slope = a_p / denom
        if not math.isfinite(slope) or slope <= 0:
            slope = LOGIT_PER_RANK_FALLBACK
        slope = min(SLOPE_HI, max(SLOPE_LO, slope))
        floor = clamp_floor(self.floors[sz].get(tid))
        return x, slope, floor, clear_kappa(floor, self.lapse)

    # --- server display (rank-translation.js + stats-ui.js skillToDisplay) ----
    def rank_index_after_rating_delta(self, idx, delta):
        xs, ys = self.curve9  # (ratings, ranks)
        rating = interp_extrap(idx, ys, xs)        # invert M: index -> rating
        return interp_extrap(rating + delta, xs, ys)

    def to_server_scalar(self, idx, server):
        our = idx
        if server == "EGF":
            return min(max(our, RANK_BAND_LO), RANK_BAND_HI)
        c = self.rank_map.get(server)
        if not c or "our" not in c or "server" not in c or len(c["our"]) < 2:
            return float("nan")
        s = interp_extrap(our, c["our"], c["server"])
        return min(max(s, RANK_BAND_LO), RANK_BAND_HI)

    def skill_to_display(self, idx, delta, server):
        """Internal skill index -> display value: index for OnlyMove, else the
        time-adjusted, server-translated scalar (port of skillToDisplay)."""
        if server == ONLYMOVE:
            return idx
        adj = self.rank_index_after_rating_delta(idx, delta) \
            if (math.isfinite(delta) and delta != 0) else idx
        if not math.isfinite(adj):
            adj = idx
        return self.to_server_scalar(adj, server)

    def display_label(self, val, server):
        return index_to_label(val) if server == ONLYMOVE else scalar_to_label(val)


# ---------------------------------------------------------------------------
# Vectorised wide-kernel pooled 4PL MLE — port of poolLogistic / evaluateAt
# ---------------------------------------------------------------------------
def _sigmoid(z):
    z = np.clip(z, -700, 700)
    return 1.0 / (1.0 + np.exp(-z))


class Smoother:
    def __init__(self, ts, x, slope, floor, kappa, y, lapse):
        order = np.argsort(ts, kind="stable")
        self.ts = ts[order]
        self.x = x[order]
        self.slope = slope[order]
        self.floor = floor[order]
        self.kappa = kappa[order]
        self.y = y[order]
        self.span = np.maximum(0.0, 1.0 - self.floor - lapse)
        self.n = len(ts)

    def at(self, t0, tau=TAU_S_MS, linear=False, slope_prior_sd=SLOPE_PRIOR_SD):
        """Port of pwa/js/rank-decomposition.js poolLogistic. linear=False is the
        local-CONSTANT kernel MLE; linear=True is the local-LINEAR-in-time fit (port
        of poolLinear) that removes the right-edge lag, reporting the intercept (level
        at t0) with the slope ridge-shrunk by SLOPE_PRIOR_INFO=1/slope_prior_sd²."""
        win = KERNEL_TRUNC_SIGMA * tau
        lo = int(np.searchsorted(self.ts, t0 - win, side="left"))
        hi = int(np.searchsorted(self.ts, t0 + win + 1, side="left"))
        if hi <= lo:
            return float("nan"), float("inf"), 0.0
        sl = slice(lo, hi)
        dt = (self.ts[sl] - t0) / tau
        w = np.where(np.abs(dt) > KERNEL_TRUNC_SIGMA, 0.0, np.exp(-0.5 * dt * dt))
        sumW = float(w.sum())
        if sumW <= 0:
            return float("nan"), float("inf"), 0.0
        x, sp, fl, ka, y, span = (self.x[sl], self.slope[sl], self.floor[sl],
                                  self.kappa[sl], self.y[sl], self.span[sl])
        effN = sumW * sumW / float((w * w).sum())
        if linear:
            return self._at_linear(t0, tau, w, dt, x, sp, fl, ka, y, span, sumW,
                                   effN, slope_prior_sd)
        R = float((w * x).sum() / sumW)
        for _ in range(15):
            praw = _sigmoid(sp * (R - x) + ka)
            p = np.clip(fl + span * praw, 1e-9, 1 - 1e-9)
            g = span * sp * praw * (1 - praw)
            denom = p * (1 - p)
            info = float((w * g * g / denom).sum())
            if info < 1e-9:
                break
            step = float((w * (y - p) * g / denom).sum()) / info
            step = 2.0 if step > 2 else (-2.0 if step < -2 else step)
            R += step
            R = X_CLAMP_LO if R < X_CLAMP_LO else (X_CLAMP_HI if R > X_CLAMP_HI else R)
            if abs(step) < 1e-6:
                break
        praw = _sigmoid(sp * (R - x) + ka)
        p = np.clip(fl + span * praw, 1e-9, 1 - 1e-9)
        g = span * sp * praw * (1 - praw)
        v = g * g / (p * (1 - p))
        A = float((w * v).sum())
        B = float((w * w * v).sum())
        se = math.sqrt(B) / A if A > 0 else float("inf")
        return R, se, effN

    def _at_linear(self, t0, tau, w, u, x, sp, fl, ka, y, span, sumW, effN, sd):
        """Local-linear fit R(t)=a+b·u, u=(t−t0)/τ; report intercept a + 2×2 sandwich
        SE. Port of poolLinear. The slope b is ridge-shrunk by prior_info=1/sd²."""
        prior_info = 1.0 / (sd * sd)
        a = float((w * x).sum() / sumW)
        b = 0.0
        for _ in range(25):
            praw = _sigmoid(sp * (a + b * u - x) + ka)
            p = np.clip(fl + span * praw, 1e-9, 1 - 1e-9)
            g = span * sp * praw * (1 - praw)
            denom = p * (1 - p)
            wv = w * g * g / denom
            sc = w * (y - p) * g / denom
            Sa = float(sc.sum()); Sb = float((sc * u).sum()) - prior_info * b
            Iaa = float(wv.sum()); Iab = float((wv * u).sum())
            Ibb = float((wv * u * u).sum()) + prior_info
            det = Iaa * Ibb - Iab * Iab
            if abs(det) < 1e-12:
                break
            da = (Ibb * Sa - Iab * Sb) / det
            db = (Iaa * Sb - Iab * Sa) / det
            da = 2.0 if da > 2 else (-2.0 if da < -2 else da)
            db = 2.0 if db > 2 else (-2.0 if db < -2 else db)
            a += da; b += db
            a = X_CLAMP_LO if a < X_CLAMP_LO else (X_CLAMP_HI if a > X_CLAMP_HI else a)
            if abs(da) < 1e-6 and abs(db) < 1e-6:
                break
        praw = _sigmoid(sp * (a + b * u - x) + ka)
        p = np.clip(fl + span * praw, 1e-9, 1 - 1e-9)
        g = span * sp * praw * (1 - praw)
        v = g * g / (p * (1 - p))
        wv = w * v; w2v = w * w * v
        Baa = float(wv.sum()); Bab = float((wv * u).sum())
        Bbb = float((wv * u * u).sum()) + prior_info
        Maa = float(w2v.sum()); Mab = float((w2v * u).sum()); Mbb = float((w2v * u * u).sum())
        detB = Baa * Bbb - Bab * Bab
        se = float("inf")
        if detB > 0:
            cov00 = (Bbb * Bbb * Maa - 2 * Bab * Bbb * Mab + Bab * Bab * Mbb) / (detB * detB)
            se = math.sqrt(cov00) if cov00 > 0 else float("inf")
        return a, se, effN

    def grid(self, ts_grid, tau=TAU_S_MS, linear=False, slope_prior_sd=SLOPE_PRIOR_SD):
        R = np.empty(len(ts_grid)); se = np.empty(len(ts_grid)); eff = np.empty(len(ts_grid))
        for i, t in enumerate(ts_grid):
            R[i], se[i], eff[i] = self.at(t, tau, linear, slope_prior_sd)
        return R, se, eff


# ---------------------------------------------------------------------------
# Time-adjustment median (port of stats-ui.js makeMedianSolveMinutesAt)
# ---------------------------------------------------------------------------
def make_median_solve_minutes_at(time_pts):
    """time_pts: sorted [(t_ms, time_used_ms)] non-repeat, 0<t<=cap. Returns a
    callable t->median solve MINUTES (30d kernel, shrunk to global), or None."""
    pts = [(t, v) for t, v in time_pts if v and 0 < v <= TIME_ADJUST_AFK_CAP_MS]
    if len(pts) < TIME_ADJUST_MIN_TIMED:
        return None
    pts.sort(key=lambda p: p[0])
    ts = np.array([p[0] for p in pts], float)
    vs = np.array([p[1] for p in pts], float)
    global_med = float(np.median(vs))
    global_log = math.log10(global_med / 60000.0)

    def at(t):
        d = (t - ts) / TIME_ADJUST_TAU_MS
        m = np.abs(d) <= 4
        if not m.any():
            return 10 ** global_log
        w = np.exp(-0.5 * d[m] * d[m])
        W = float(w.sum())
        if W <= 0 or m.sum() < 2:
            return 10 ** global_log
        ess = W * W / float((w * w).sum())
        # weighted median in minute space
        v = vs[m]
        o = np.argsort(v)
        vv, ww = v[o], w[o]
        cum = np.cumsum(ww)
        target = 0.5 * W
        k = int(np.searchsorted(cum, target, side="left"))
        if k == 0:
            wmed = vv[0]
        else:
            wmed = vv[k - 1] + (vv[k] - vv[k - 1]) * ((target - cum[k - 1]) / ww[k])
        local_log = math.log10(wmed / 60000.0)
        wl = ess / (ess + TIME_ADJUST_SHRINK_ESS)
        return 10 ** (wl * local_log + (1 - wl) * global_log)

    return at


# ---------------------------------------------------------------------------
def load_rows(db_path, user_id):
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT problem, size, success, created_at, time_used FROM tsumego_reports "
            "WHERE user_id LIKE ? AND is_repeat=0 ORDER BY created_at",
            (user_id + "%",),
        ).fetchall()
        surveys = con.execute(
            "SELECT server, rank, created_at FROM rank_surveys "
            "WHERE user_id LIKE ? ORDER BY created_at", (user_id + "%",),
        ).fetchall()
        n_all = con.execute(
            "SELECT count(*), sum(is_repeat) FROM tsumego_reports WHERE user_id LIKE ?",
            (user_id + "%",)).fetchone()
    finally:
        con.close()
    return rows, surveys, n_all


def build_features(rows, rs):
    ts, x, slope, floor, kappa, y, size = [], [], [], [], [], [], []
    n_unmatched = 0
    for problem, sz, success, created_at, _t in rows:
        feat = rs.features(problem, sz)
        if feat is None:
            n_unmatched += 1
            continue
        fx, fs, ff, fk = feat
        ts.append(parse_ts(created_at))
        x.append(fx); slope.append(fs); floor.append(ff); kappa.append(fk)
        y.append(1 if success else 0); size.append(int(sz))
    arr = lambda v, d=float: np.asarray(v, dtype=d)
    return (arr(ts), arr(x), arr(slope), arr(floor), arr(kappa),
            arr(y), arr(size, int), n_unmatched)


def compute_variant(full, grid, apply_filter, lapse, linear=True):
    """Return (smoother, S_grid, seS_grid, effN_grid, S_at, effN_at, used, n_kept)
    for the near-rank-filtered (or unfiltered) skill curve in INTERNAL-RANK space.
    `linear` selects the displayed skill curve's time model — True (default) matches
    the shipped local-LINEAR wide kernel (rank-decomposition.js); the near-rank
    filter's S0 pre-pass stays local-constant (it only decides which attempts to
    keep)."""
    if apply_filter:
        S0_grid, _, _ = full.grid(grid)
        S0_at = np.interp(full.ts, grid, S0_grid)
        keep = np.abs(full.slope * (S0_at - full.x)) <= NEAR_RANK_L
        n_kept = int(keep.sum())
        if n_kept >= MIN_ATTEMPTS:
            sm = Smoother(full.ts[keep], full.x[keep], full.slope[keep],
                          full.floor[keep], full.kappa[keep], full.y[keep], lapse)
            used = "near-rank filtered"
        else:
            sm, used = full, "UNFILTERED (too few survived the filter)"
            n_kept = full.n
    else:
        sm, used, n_kept = full, "no filter", full.n
    S_grid, seS_grid, effN_grid = sm.grid(grid, linear=linear)
    S_at = np.interp(full.ts, grid, S_grid)
    effN_at = np.interp(full.ts, grid, effN_grid)
    return sm, S_grid, seS_grid, effN_grid, S_at, effN_at, used, n_kept


def time_deltas(S_vals, t_vals, med_at, ta, server):
    """Per-point time-adjust delta (rating units); 0 for OnlyMove / no time data."""
    out = np.zeros(len(S_vals))
    if server == ONLYMOVE or med_at is None or ta is None:
        return out
    for i, (S, t) in enumerate(zip(S_vals, t_vals)):
        m = med_at(float(t))
        if m and m > 0 and math.isfinite(S):
            expected = ta["g0"] + ta["g1"] * S
            out[i] = ta["alpha"] * (math.log10(m) - expected)
    return out


def to_display(S_vals, deltas, rs, server):
    return np.array([rs.skill_to_display(float(S), float(d), server)
                     for S, d in zip(S_vals, deltas)])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user_id")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--dir", default=V2_DIR_DEFAULT)
    ap.add_argument("--pop-constants", default=POP_DEFAULT)
    ap.add_argument("--server", default=ONLYMOVE,
                    help="OnlyMove (default), EGF, or a rank_map.json server")
    ap.add_argument("--both-filters", action="store_true",
                    help="overlay near-rank-filter ON and OFF on one graph/table")
    ap.add_argument("--local-constant", action="store_true",
                    help="render the OLD local-constant wide kernel (pre-boundary-fix) "
                         "instead of the shipped local-linear one")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found — run pipeline.fetch_database first.")

    pc = load_pop_constants(args.pop_constants)
    lapse = lapse_from(pc)
    rs = RankingSystem(args.dir, lapse)
    server = args.server
    if server != ONLYMOVE and server != "EGF" and server not in rs.rank_map:
        sys.exit(f"Unknown server '{server}'. Known: OnlyMove, EGF, "
                 + ", ".join(sorted(rs.rank_map)))

    rows, surveys, n_all = load_rows(args.db, args.user_id)
    if not rows:
        sys.exit(f"No first-time attempts for user prefix '{args.user_id}'.")

    ts, x, slope, floor, kappa, y, size, n_unmatched = build_features(rows, rs)
    if len(ts) < MIN_ATTEMPTS:
        sys.exit(f"Only {len(ts)} matched first-time attempts (< {MIN_ATTEMPTS}).")
    full = Smoother(ts, x, slope, floor, kappa, y, lapse)

    # time-adjust support (uses ALL first-time attempts' times, matched or not)
    tad = pc.get("time_adjust") or {}
    ta = None
    if all(k in tad for k in ("alpha_per_log10_min", "time_vs_rank_intercept",
                              "time_vs_rank_slope")):
        ta = {"alpha": tad["alpha_per_log10_min"], "g0": tad["time_vs_rank_intercept"],
              "g1": tad["time_vs_rank_slope"]}
    med_at = make_median_solve_minutes_at(
        [(parse_ts(r[3]), r[4]) for r in rows]) if server != ONLYMOVE else None

    t0, t1 = float(full.ts[0]), float(full.ts[-1])
    grid = np.linspace(t0, t1, max(2, int((t1 - t0) / DAY_MS) + 1))

    variants = [("filter ON", True)]
    if args.both_filters:
        variants.append(("filter OFF", False))

    # ----------------------------- header ------------------------------------
    n_rows_all, n_rep = int(n_all[0]), int(n_all[1] or 0)
    print(f"User '{args.user_id}'  —  recreated client Rank-tab graph  "
          f"[server = {server}]")
    print(f"  ranking system: {os.path.relpath(args.dir, PROJECT_ROOT)}   LAPSE={lapse:.4f}")
    if server != ONLYMOVE:
        if ta and med_at:
            print(f"  time-adjust: alpha={ta['alpha']:+.3f}/log10min  "
                  f"(timed attempts available)")
        else:
            print("  time-adjust: UNAVAILABLE (constants or timed data missing) "
                  "-> server view == identity")
    print(f"  reports in DB: {n_rows_all} ({n_rep} repeats); first-time used: "
          f"{len(ts)}  (unmatched/dropped: {n_unmatched})")
    by_sz = {s: int((size == s).sum()) for s in (9, 13)}
    print(f"  by size: 9x9={by_sz[9]}  13x13={by_sz[13]}")
    print(f"  span: {rows[0][3]}  ->  {rows[-1][3]}")
    if surveys:
        latest = {srv: rk for srv, rk, _ in surveys}
        print("  declared survey rank(s) (latest): "
              + "  ".join(f"{s}:{r}" for s, r in latest.items()))

    # --------------------------- per-variant ---------------------------------
    lab = rs.display_label
    results = {}
    for name, do_filter in variants:
        (sm, S_grid, seS_grid, effN_grid, S_at, effN_at,
         used, n_kept) = compute_variant(full, grid, do_filter, lapse,
                                          linear=not args.local_constant)
        # transform skill (+ band edges) to display space
        d_grid = time_deltas(S_grid, grid, med_at, ta, server)
        disp_grid = to_display(S_grid, d_grid, rs, server)
        disp_up = to_display(S_grid + seS_grid, d_grid, rs, server)
        disp_dn = to_display(S_grid - seS_grid, d_grid, rs, server)
        d_at = time_deltas(S_at, full.ts, med_at, ta, server)
        disp_at = to_display(S_at, d_at, rs, server)
        results[name] = dict(disp_grid=disp_grid, disp_up=disp_up, disp_dn=disp_dn,
                             effN_grid=effN_grid, disp_at=disp_at, effN_at=effN_at,
                             used=used, n_kept=n_kept)
        cur = disp_grid[-1]
        print(f"\n  [{name}] near-rank: kept {n_kept}/{len(ts)} [{used}]")
        print(f"    current rank (end of graph): {cur:+.2f} = {lab(cur, server)}")

    # --------------------- monthly rank distributions ------------------------
    by_month = defaultdict(list)
    for i in range(len(full.ts)):
        by_month[month_of(full.ts[i])].append(i)

    print(f"\nMonthly {server} rank distribution (recreated graph value -> label):")
    cols = "".join(f"{n+' p25/med/p75':>26}" for n, _ in variants)
    print(f"{'month':<9}{'n':>6}{'clear%':>8}{cols}")
    print("-" * (23 + 26 * len(variants)))
    for m in sorted(by_month):
        idx = np.array(by_month[m])
        clear = float(full.y[idx].mean()) * 100
        line = f"{m:<9}{len(idx):>6}{clear:>7.1f}%"
        for name, _ in variants:
            sv = results[name]["disp_at"][idx]
            sv = sv[np.isfinite(sv)]
            if len(sv) == 0:
                line += f"{'-':>26}"; continue
            p25, p50, p75 = np.percentile(sv, [25, 50, 75])
            cell = f"{p25:+.1f}/{p50:+.1f}({lab(p50, server)})/{p75:+.1f}"
            line += f"{cell:>26}"
        print(line)
    print(f"\nValues are {server} rank ("
          + ("internal index" if server == ONLYMOVE else "kyu/dan scalar: 1k=0, 1d=1")
          + "); med shown with its rounded label.")

    # ----------------------------- artefacts ---------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            head = ["date"]
            for name, _ in variants:
                tag = name.replace(" ", "_")
                head += [f"{tag}_rank", f"{tag}_label", f"{tag}_effN"]
            wr.writerow(head)
            for i in range(len(grid)):
                dstr = datetime.fromtimestamp(grid[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                row = [dstr]
                for name, _ in variants:
                    r = results[name]
                    row += [f"{r['disp_grid'][i]:.4f}", lab(r['disp_grid'][i], server),
                            f"{r['effN_grid'][i]:.1f}"]
                wr.writerow(row)
        print(f"\nwrote daily graph -> {args.csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.ticker import FuncFormatter, MultipleLocator
        except ImportError:
            print("matplotlib not available; skipping --plot", file=sys.stderr)
        else:
            dts = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in grid]
            colors = {"filter ON": "#2563eb", "filter OFF": "#e07b00"}
            fig, ax = plt.subplots(figsize=(11, 4.8))
            for name, _ in variants:
                r = results[name]
                sup = (r["effN_grid"] >= SUPPORT_EFFN) & np.isfinite(r["disp_grid"])
                col = colors.get(name, "#444")
                line = np.where(sup, r["disp_grid"], np.nan)
                up = np.where(sup, r["disp_up"], np.nan)
                dn = np.where(sup, r["disp_dn"], np.nan)
                ax.fill_between(dts, dn, up, alpha=0.13, color=col)
                ax.plot(dts, line, color=col, lw=2,
                        label=f"{name} (±1 se)")
            ax.set_ylabel(f"{server} rank")
            # One tick per whole grade so each label is distinct (half-integer
            # ticks would round two neighbours to the same kyu/dan label).
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: lab(v, server)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            flt = "near-rank filter ON vs OFF" if args.both_filters else "near-rank filtered"
            ax.set_title(f"{args.user_id} — recreated Rank-tab graph — "
                         f"{server} rank — {flt}")
            ax.grid(alpha=0.3); ax.legend(loc="best")
            fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"wrote plot -> {args.plot}")


if __name__ == "__main__":
    main()
