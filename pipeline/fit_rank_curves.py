"""Jointly fit, from rank_surveys, the per-board rating→rank curve AND the
per-server rank couplings — one model, internal(EGF) rank as the shared scale.

Model (internal rank = EGF rank is the hub / common unit):

    rating θ ──per board, m_s──▶ INTERNAL(EGF) RANK ──per server──▶ server rank

  Both arrows are estimated TOGETHER over every declaration (fit_joint).

  * INTERNAL RANK IS EGF RANK. Each user's rating = the problem RATING at which
    they clear CALIBRATION_P (80%), from a logistic over their (raw_rating,
    correct) attempts (fit_user_rating) — the SAME quantity the client computes,
    using only the shipped black-box ratings (never the derivation internals).
    The per-board map m_s(rating) → internal rank is a monotone piecewise-linear
    curve defined by EGF rating-thresholds, with the cubic
    (auxiliary.fit_statistics_shapes, pinned to that board's humanv0_8k/3k bots)
    as a SHAPE PRIOR (--rating-lambda). package_for_pwa ships these per-board
    curves (ratings/rating_to_rank.json) AND the raw ratings, and the client
    converts rating→rank with the same curve — so the displayed rank, the
    problem ranks, and this map all agree by construction (the rank loop closes).
  * Each non-EGF server has thresholds in internal-rank units:

        P(declared ≤ g | internal_rank x) = expit(τ_{server,g} − x)

    with a uniform-spacing shape prior (--lambda; sparse servers collapse to a
    pure offset α, the offset itself unregularized). EGF feeds the rating-curve
    DIRECTLY, never through a server offset, so it stays the identity hub; every
    other server's declaration is evidence about BOTH its own offset AND the
    shape of m_s where that declaration lands.

  Why joint, not two stages: previously m_s was fit on EGF declarers alone
  (essentially all dan), then FROZEN, and the server couplings fit on top. The
  kyu range of m_s was then uncalibrated cubic extrapolation that over-rated
  weak players; those inflated internal ranks dragged the dan thresholds through
  the pooled coupling fit and floated EGF to the top of the map (shown as the
  most lenient server — the opposite of reality). Fitting both directions at
  once lets the abundant non-EGF kyu declarations pin the kyu end of m_s, so
  neither range is traded off against the other.

A both-boards user contributes a declaration on each board; both observations
enter the joint objective on the one internal-rank scale.

Output: pwa/tsumegos/rank_map.json = {our_rank_label: {server: server_rank}},
board-independent (the per-board difference lives in rating→internal-rank, i.e.
the user's displayed rank, not in the translation). Re-run
pipeline.fetch_database first to refresh attempts/database.sql.

Usage:
  ./.venv/bin/python3 -m pipeline.fit_rank_curves
  ./.venv/bin/python3 -m pipeline.fit_rank_curves --lambda 1.0
  ./.venv/bin/python3 -m pipeline.fit_rank_curves --rank-padding 4
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

# Slope of the success curve in RATING units (logit per rating unit). The
# survey user's rating is "the problem rating where they clear CALIBRATION_P",
# estimated by a logistic with this slope; the client (rank-decomposition.js)
# MUST use the same principle and a matching slope, or the loop reopens. Value
# is calibrated to the current rating scale (bots span ~1.5 rating over ~16
# ranks ⇒ ~0.1 rating/rank; with ~0.28 logit/rank that's ~2.8 logit/rating).
LOGIT_PER_RATING = 2.5
CALIBRATION_P = 0.8  # matches the client / momentum: rank = where you clear 80%

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import (  # noqa: E402
    SERVERS,
    SQL_PATH_DEFAULT,
    STOPGAP_X_3K,
    STOPGAP_X_8K,
    latest_per_user_server,
    load_attempts_for_users,
    load_problem_params,
    load_rank_surveys,
    scalar_to_rank,
)


# ---------------------------------------------------------------------------
# rating θ → our-rank, per board. The cubic shape (best fit to the
# statistics.txt even-game win-rate table, auxiliary.fit_statistics_shapes) is
# pinned to this board's humanv0_8k/3k bot ratings — identical to the mapping
# package_for_pwa uses to display ranks, so the our-rank here matches the rank
# the user actually sees.
# ---------------------------------------------------------------------------

_CUBIC_CURVE_CACHE = {}


def _cubic_curve(stopgap_curve):
    """A CubicRatingCurve pinned to the humanv0_8k/3k anchors recovered from
    `stopgap_curve` (which passes exactly through those two points), or None if
    the cubic fit / anchors are unavailable. Cached per anchor pair."""
    if stopgap_curve is None:
        return None
    A, k, C = stopgap_curve["A"], stopgap_curve["k"], stopgap_curve["C"]
    y_8k = A * np.exp(k * STOPGAP_X_8K) + C
    y_3k = A * np.exp(k * STOPGAP_X_3K) + C
    key = (round(y_8k, 6), round(y_3k, 6))
    if key not in _CUBIC_CURVE_CACHE:
        try:
            from auxiliary.fit_statistics_shapes import CubicRatingCurve
            _CUBIC_CURVE_CACHE[key] = CubicRatingCurve(
                y_8k, y_3k, STOPGAP_X_8K, STOPGAP_X_3K)
        except Exception as e:  # noqa: BLE001
            print(f"  (cubic curve unavailable: {e})")
            _CUBIC_CURVE_CACHE[key] = None
    return _CUBIC_CURVE_CACHE[key]


def cubic_prior_for_rating_fit(curve, g_min, g_max):
    """(init_alpha, baseline_spacings) that anchors the joint fit's rating→EGF-
    rank curve in RATING space with the cubic as the shape prior. Threshold g
    sits at the rating the cubic assigns to the boundary just below rank g —
    our-rank scalar g−0.5, i.e. rank_index (g−0.5)+19. baseline_spacings are the
    cubic's rating deltas between consecutive thresholds; init_alpha is the cubic
    rating at the first threshold. With log_e=0 this reproduces the cubic shape
    exactly, so λ pulls the fitted curve toward the cubic."""
    gs = list(range(g_min + 1, g_max + 1))           # the K threshold ranks
    boundary_ri = np.array([(g - 0.5) + 19.0 for g in gs])
    prior_rating = np.asarray(curve.forward(boundary_ri), dtype=float)
    init_alpha = float(prior_rating[0])
    baseline = np.diff(prior_rating) if len(prior_rating) > 1 else np.array([])
    return init_alpha, baseline


def load_raw_ratings(size):
    """{tsumego_id: raw rating} from ratings/<size>/tsumego_ratings_final.csv —
    the same black-box scalar the PWA ships and the client estimates against."""
    path = os.path.join(PROJECT_ROOT, "ratings", str(size),
                        "tsumego_ratings_final.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["tsumego_id"], df["rating"].astype(float)))


def fit_user_rating(attempts, raw_ratings, target=CALIBRATION_P, min_n=1):
    """The user's rating = the problem RATING at which they clear `target` (80%).

    MLE of R in  P(correct) = σ(LOGIT_PER_RATING·(R − raw_rating) + logit(target))
    over the user's (raw_rating, correct) attempts. This is the in-loop estimator
    — it never touches the rating-derivation internals, only the shipped ratings —
    and it is the SAME quantity the client computes; M is then anchored on it.

    Returns (R, n_joined) or (None, 0)."""
    rows = [(raw_ratings[t], s) for t, s in attempts if t in raw_ratings]
    if len(rows) < min_n:
        return None, 0
    rr = np.array([r for r, _ in rows], dtype=float)
    sc = np.array([s for _, s in rows], dtype=float)
    off = float(np.log(target / (1 - target)))
    R = float(rr.mean())
    for _ in range(80):
        z = np.clip(LOGIT_PER_RATING * (R - rr) + off, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        score = float(np.sum(LOGIT_PER_RATING * (sc - p)))
        info = float(np.sum(LOGIT_PER_RATING ** 2 * p * (1 - p)))
        if info < 1e-9:
            break
        step = max(-1.0, min(1.0, score / info))
        R += step
        if abs(step) < 1e-6:
            break
    return float(R), len(rows)


# ---------------------------------------------------------------------------
# Joint fit: rating → internal(EGF) rank (per board) AND internal rank →
# server rank (per server), estimated TOGETHER over all declarations.
#
# Latent ability = internal rank = EGF rank. Per board s the curve
# m_s(rating) → internal rank is a monotone piecewise-linear map defined by EGF
# rating-thresholds a_{s,g} (rating at the boundary just below EGF rank g), with
# the cubic as a shape prior (λ_rating). Each non-EGF server gets thresholds
# b_{server,h} in internal-rank units (uniform-spacing shape prior, λ). The
# likelihood couples them:
#
#   EGF declaration    (board s, rating r, declared g):
#       P(declared ≤ g | r) = expit(a_{s,g} − r)
#   server declaration (board s, rating r, server, declared h):
#       x = m_s(r) = interp(r; a_{s,·} → g−0.5)
#       P(declared ≤ h | r) = expit(b_{server,h} − x)
#
# Because every server term runs through m_s, a kyu OGS/Fox declaration is now
# evidence about the SHAPE of the rating→rank curve at the kyu end — not just
# the server's offset. EGF anchors the absolute internal-rank scale (it feeds
# a_{s,·} directly, never through a server offset), so EGF stays the hub /
# identity; the other servers pin the parts of the curve EGF never reaches.
# Contrast the old pipeline: there a_{s,·} was fit on EGF alone, FROZEN, then
# server offsets were fit on top — so kyu-heavy non-EGF cohorts (whose internal
# ranks were extrapolated cubic) could never correct the axis, and their
# inflated internal ranks dragged the dan thresholds, floating EGF to the top.
# ---------------------------------------------------------------------------

def ordinal_nll_vec(tau, x, k):
    """Vectorized cumulative-link negative log-likelihood.

      P(declared = category k | x), thresholds τ (length K, ascending):
        k = 0      : expit(τ_0 − x)
        0 < k < K  : expit(τ_k − x) − expit(τ_{k−1} − x)
        k = K      : 1 − expit(τ_{K−1} − x)

    tau : (K,) thresholds in x units.  x : (n,) abilities.  k : (n,) ints in
    [0, K] (category = declared rank − range minimum)."""
    K = tau.shape[0]
    upper = np.where(k < K, expit(tau[np.minimum(k, K - 1)] - x), 1.0)
    lower = np.where(k > 0, expit(tau[np.maximum(k - 1, 0)] - x), 0.0)
    p = np.clip(upper - lower, 1e-12, 1.0)
    return -float(np.sum(np.log(p)))


def interp_extrap_vec(x, xs, ys):
    """np.interp with linear extrapolation past both ends (vectorized).
    xs strictly ascending, len ≥ 2; x an array."""
    y = np.interp(x, xs, ys)
    sl0 = (ys[1] - ys[0]) / (xs[1] - xs[0])
    sl1 = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + sl0 * (x - xs[0]), y)
    y = np.where(x > xs[-1], ys[-1] + sl1 * (x - xs[-1]), y)
    return y


def _unpack_thresholds(block, baseline):
    """[α, log_e...] + baseline spacings → ascending thresholds (length
    len(baseline)+1).  τ_0 = α; τ_j = α + Σ_{≤j} baseline·exp(log_e)."""
    alpha = block[0]
    if len(baseline) == 0:
        return np.array([alpha])
    spacings = baseline * np.exp(block[1:])
    return np.concatenate([[alpha], alpha + np.cumsum(spacings)])


def fit_joint(decls, cubic_by_size, sizes, egf_lo, egf_hi,
              rating_lam, lam, rank_padding, extend_to, min_server_samples):
    """Jointly fit per-board rating→internal(EGF)-rank curves and per-server
    internal-rank→server-rank couplings over all `decls`.

    decls : iterable of (size, server, rating, g_scalar).
    Returns (rating_rank_fits, server_fits) where
      rating_rank_fits : {size: {egf_g: rating at boundary below g}}
      server_fits      : {server: ({declared_g: τ_internal}, nll, dev)}
    Each board needs ≥1 EGF declaration to anchor its absolute scale; boards
    without one keep the pure cubic curve (and any server declarations on that
    board contribute via the cubic-fixed internal rank)."""
    boards = [s for s in sizes if cubic_by_size.get(s) is not None]
    N_egf = egf_hi - egf_lo + 1
    egf_thr_ranks = list(range(egf_lo + 1, egf_hi + 1))   # K = N_egf-1 ranks
    ys_by_board = {s: np.array([g - 0.5 for g in egf_thr_ranks], dtype=float)
                   for s in boards}

    # --- bucket declarations ---
    egf_data = {s: {"x": [], "k": []} for s in boards}
    server_rows = defaultdict(list)   # server -> [(size, rating, g_int)]
    for size, server, rating, g in decls:
        if size not in boards:
            continue
        gi = int(round(g))
        if server == "EGF":
            egf_data[size]["x"].append(rating)
            egf_data[size]["k"].append(max(0, min(N_egf - 1, gi - egf_lo)))
        else:
            server_rows[server].append((size, rating, gi))
    for s in boards:
        egf_data[s]["x"] = np.asarray(egf_data[s]["x"], dtype=float)
        egf_data[s]["k"] = np.asarray(egf_data[s]["k"], dtype=int)

    # Boards with their own EGF data fit a_{s,·}; the rest stay on the cubic.
    fit_boards = [s for s in boards if len(egf_data[s]["x"]) > 0]
    cubic_a = {}   # board -> fixed cubic thresholds (ratings) for non-fit boards
    for s in boards:
        if s not in fit_boards:
            ia, base = cubic_prior_for_rating_fit(cubic_by_size[s], egf_lo, egf_hi)
            cubic_a[s] = _unpack_thresholds(
                np.concatenate([[ia], np.zeros(len(base))]), base)

    # --- per-server blocks (internal-rank thresholds) ---
    server_blocks = {}
    for server, rows in server_rows.items():
        if len(rows) < min_server_samples:
            continue
        declared = [g for _, _, g in rows]
        g_min = min(declared) - rank_padding
        g_max = max(max(declared) + rank_padding, extend_to)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        by_board = defaultdict(lambda: {"rating": [], "k": []})
        for size, rating, g in rows:
            by_board[size]["rating"].append(rating)
            by_board[size]["k"].append(max(0, min(N - 1, g - g_min)))
        for s in by_board:
            by_board[s]["rating"] = np.asarray(by_board[s]["rating"], dtype=float)
            by_board[s]["k"] = np.asarray(by_board[s]["k"], dtype=int)
        server_blocks[server] = {
            "g_min": g_min, "g_max": g_max, "K": K,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "by_board": dict(by_board)}

    # --- parameter layout: EGF rating-thresholds, then per-server thresholds ---
    layout = []   # (kind, key, start, n, baseline)
    x0 = []
    idx = 0
    for s in fit_boards:
        ia, base = cubic_prior_for_rating_fit(cubic_by_size[s], egf_lo, egf_hi)
        base = np.asarray(base, dtype=float)
        n = len(base) + 1
        layout.append(("egf", s, idx, n, base))
        x0.append(ia)
        x0.extend([0.0] * len(base))
        idx += n
    for server, blk in server_blocks.items():
        n = len(blk["baseline"]) + 1
        layout.append(("srv", server, idx, n, blk["baseline"]))
        x0.append(float(blk["g_min"]))
        x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, dtype=float)

    def unpack(params):
        a = dict(cubic_a)   # board -> rating thresholds
        b = {}              # server -> internal-rank thresholds
        logs = {}           # (kind, key) -> log deviations
        for kind, key, st, n, base in layout:
            thr = _unpack_thresholds(params[st:st + n], base)
            logs[(kind, key)] = params[st + 1:st + n]
            (a if kind == "egf" else b)[key] = thr
        return a, b, logs

    def objective(params):
        a, b, logs = unpack(params)
        nll = 0.0
        for s in fit_boards:
            d = egf_data[s]
            if len(d["x"]):
                nll += ordinal_nll_vec(a[s], d["x"], d["k"])
        for server, blk in server_blocks.items():
            for s, dd in blk["by_board"].items():
                x_int = interp_extrap_vec(dd["rating"], a[s], ys_by_board[s])
                nll += ordinal_nll_vec(b[server], x_int, dd["k"])
        pen = 0.0
        for (kind, key), lg in logs.items():
            pen += (rating_lam if kind == "egf" else lam) * float(np.sum(lg ** 2))
        return nll + pen

    res = minimize(objective, x0, method="L-BFGS-B",
                   options={"maxiter": 10000})
    a, b, logs = unpack(res.x)

    rating_rank_fits = {s: {g: float(a[s][i])
                            for i, g in enumerate(egf_thr_ranks)}
                        for s in fit_boards}
    server_fits = {}
    for server, blk in server_blocks.items():
        thr = b[server]
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        thresholds = {gs[i]: float(thr[i]) for i in range(len(gs))}
        nll = 0.0
        for s, dd in blk["by_board"].items():
            x_int = interp_extrap_vec(dd["rating"], a[s], ys_by_board[s])
            nll += ordinal_nll_vec(thr, x_int, dd["k"])
        dev = float(np.sum(logs[("srv", server)] ** 2))
        server_fits[server] = (thresholds, nll, dev)
    return rating_rank_fits, server_fits


# ---------------------------------------------------------------------------
# Board-independent server rank-translation map.
# ---------------------------------------------------------------------------

def _interp_extrap(x, xs, ys):
    """np.interp with linear extrapolation past both ends using the end slope.
    xs strictly ascending, len ≥ 2."""
    if x <= xs[0]:
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return ys[0] + slope * (x - xs[0])
    if x >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    return float(np.interp(x, xs, ys))


def build_rank_map(fits, our_lo=-29, our_hi=9):
    """Map our rank → equivalent declared rank on each fitted server.

    Board-independent: keyed only by our-rank label (30k..9d). Each server's
    thresholds τ_g are in our-rank units; we invert them (our-rank at the
    boundary below g is τ_g; the server scalar there is g−0.5), linearly
    extrapolate past the surveyed band, and clamp to the real rank range so
    extreme extrapolation can't print impossible labels like "15 dan".

    Returns {our_rank_label: {server: server_rank_label}}."""
    server_xy = {}
    for server, (thresholds, _, _) in fits.items():
        if len(thresholds) < 2:
            continue
        gs = sorted(thresholds)
        server_xy[server] = ([thresholds[g] for g in gs],
                             [g - 0.5 for g in gs])
    out = {}
    for s in range(our_lo, our_hi + 1):
        row = {}
        for server in SERVERS:
            if server == "EGF":
                # Internal rank IS EGF rank, so EGF is identity by definition.
                row[server] = scalar_to_rank(s)
                continue
            if server not in server_xy:
                continue
            xs, ys = server_xy[server]
            srv = max(our_lo, min(our_hi,
                                  int(round(_interp_extrap(float(s), xs, ys)))))
            row[server] = scalar_to_rank(srv)
        out[scalar_to_rank(s)] = row
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit a board-independent server↔our-rank coupling from "
                    "rank-survey declarations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sql", default=SQL_PATH_DEFAULT,
                        help="Path to D1 SQL dump")
    parser.add_argument("--sizes", default="9,13",
                        help="Comma-separated board sizes whose ratings feed "
                             "the common our-rank scale")
    parser.add_argument("--min-attempts", type=int, default=20,
                        help="Drop users with fewer attempts at a size")
    parser.add_argument("--min-server-samples", type=int, default=3,
                        help="Skip a server with fewer pooled declarations")
    parser.add_argument("--rank-padding", type=int, default=2,
                        help="Extend the threshold range this many ranks past "
                             "the min/max declared rank for a server")
    parser.add_argument("--extend-to", type=int, default=9,
                        help="Force the threshold range to cover at least up to "
                             "this rank scalar (9 = 9d) so strong-end "
                             "translation is defined")
    parser.add_argument("--lambda", type=float, default=10.0, dest="lam",
                        help="Cubic-shape prior strength for the SERVER couplings "
                             "(internal rank → server rank): L2 weight on the "
                             "per-rank log-deviations from uniform spacing. Kept "
                             "strong so sparse servers stay at a clean offset.")
    parser.add_argument("--rating-lambda", type=float, default=1.0, dest="rating_lam",
                        help="Cubic-shape prior strength for the per-board "
                             "rating→rank fit. Weaker than --lambda so the EGF "
                             "survey can bend the dan range off the (nominal-"
                             "anchored) cubic where it over-rates; ≈1 lets the "
                             "declarations correct the curve, higher pins it to "
                             "the cubic (dan over-rating returns).")
    parser.add_argument(
        "--map-out",
        default=os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rank_map.json"),
        help="Write the board-independent server rank-translation map (our "
             "rank → equivalent rank per server) here. Empty string skips it.")
    parser.add_argument(
        "--rating-map-out",
        default=os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rating_to_rank.json"),
        help="Write the per-board rating→internal(EGF)-rank curves straight to "
             "the PWA's served path. The client fetches this and converts every "
             "problem's raw rating to a rank itself — package_for_pwa ships only "
             "raw ratings and never touches ranks. Empty string skips it.")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys.empty:
        print("  No rank_surveys rows found.")
        return
    surveys = latest_per_user_server(surveys)
    print(f"  {len(surveys)} latest declarations from "
          f"{surveys['user_id'].nunique()} users")

    attempts_by_user_size = load_attempts_for_users(
        args.sql, set(surveys["user_id"].unique()))

    raw_ratings_by_size = {}
    cubic_by_size = {}
    for size in sizes:
        _params, _theta_arr, _scalar_arr, stopgap = load_problem_params(size)
        raw_ratings = load_raw_ratings(size)
        if not raw_ratings:
            print(f"  size {size}: skipped (missing tsumego_ratings_final.csv)")
            continue
        raw_ratings_by_size[size] = raw_ratings
        curve = _cubic_curve(stopgap)
        if curve is not None:
            cubic_by_size[size] = curve
        print(f"  size {size}: {len(raw_ratings)} problem ratings, "
              f"cubic {'ready' if curve is not None else 'MISSING'}")

    print(f"\nEstimating each user's {int(CALIBRATION_P*100)}%-clear rating per "
          f"(user, size) in rating space ...")
    user_rating = defaultdict(dict)  # user_id -> {size: 80%-clear rating}
    for uid in sorted(surveys["user_id"].unique()):
        prefix = uid[:8]
        for size, raw_ratings in raw_ratings_by_size.items():
            if cubic_by_size.get(size) is None:
                continue
            attempts = attempts_by_user_size.get((prefix, size), [])
            if len(attempts) < args.min_attempts:
                continue
            r, n = fit_user_rating(attempts, raw_ratings, min_n=args.min_attempts)
            if r is not None and n >= args.min_attempts:
                user_rating[uid][size] = r

    # ---- Joint fit: rating → internal(EGF) rank per board AND internal rank →
    # server rank per server, estimated TOGETHER (see fit_joint). EGF anchors
    # the internal-rank scale; every other server's declarations also bend the
    # rating→rank curve where they land — so a kyu-heavy cohort can no longer
    # be stamped with an extrapolated-cubic internal rank that drags the dan
    # thresholds (the failure mode of the old frozen-Stage-A pipeline).
    egf_lo, egf_hi = -20, 10  # 21k..10d

    decls = []  # (size, server, rating, declared_scalar)
    for _, row in surveys.iterrows():
        uid, server, g = row["user_id"], row["server"], row["rank_scalar"]
        if g is None or not np.isfinite(g):
            continue
        for size, rating in user_rating.get(uid, {}).items():
            decls.append((size, server, rating, float(g)))

    n_egf = sum(1 for _sz, srv, _, _ in decls if srv == "EGF")
    print(f"\nJoint fit over {len(decls)} (declaration × board) observations "
          f"[{n_egf} EGF anchoring the scale; rating→rank prior λ = "
          f"{args.rating_lam}, server prior λ = {args.lam}, "
          f"≥{args.min_server_samples} samples, padding = ±{args.rank_padding}]")

    rating_rank_fits, fits = fit_joint(
        decls, cubic_by_size, sizes, egf_lo, egf_hi,
        rating_lam=args.rating_lam, lam=args.lam,
        rank_padding=args.rank_padding, extend_to=args.extend_to,
        min_server_samples=args.min_server_samples)

    for size in sorted(rating_rank_fits):
        n = sum(1 for sz, srv, _, _ in decls if srv == "EGF" and sz == size)
        print(f"  size {size}: rating→internal-rank curve fit "
              f"({n} EGF declarations anchoring it)")

    # Declaration counts for the per-server report below (one per board, so a
    # both-boards user counts twice — both feed the joint fit).
    by_server = defaultdict(list)              # server -> [declared scalar]
    per_board_counts = defaultdict(lambda: defaultdict(int))  # server -> size -> n
    for size, server, _rating, g in decls:
        if server == "EGF":
            continue
        by_server[server].append(int(round(g)))
        per_board_counts[server][size] += 1

    skipped = sorted((srv, len(gs)) for srv, gs in by_server.items()
                     if srv not in fits and gs)
    if skipped:
        print("\nSkipped (below sample threshold):")
        for server, n in skipped:
            print(f"  {server}: {n} obs")

    # Per-server table: declared server rank → our-rank threshold τ_g. dev is
    # Σ log_e_g² — how far this server bends off the cubic shape (0 = exactly
    # the cubic, offset only).
    for server in SERVERS:
        if server not in fits:
            continue
        thresholds, nll, dev = fits[server]
        board_bits = ", ".join(f"n{sz}={per_board_counts[server][sz]}"
                               for sz in sizes)
        print(f"\n{'=' * 60}")
        print(f"{server}  —  {len(by_server[server])} obs ({board_bits}), "
              f"NLL {nll:.2f}, shape-dev {dev:.3f}")
        print(f"{'=' * 60}")
        counts = defaultdict(int)
        for g in by_server[server]:
            counts[g] += 1
        print(f"  {'declared':>9}  {'n':>4}  {'τ (our-rank)':>13}")
        print("  " + "-" * 32)
        for g in sorted(thresholds):
            print(f"  {scalar_to_rank(g):>9}  {counts.get(g, 0):>4d}  "
                  f"{thresholds[g]:>+13.3f}")

    print("\nτ_g = our-rank at the boundary just below declared server rank g.")
    print("shape-dev = Σ log_e_g²  (0 ⇒ server matches the cubic shape exactly,")
    print("            i.e. a pure offset; larger ⇒ data pulled a per-rank bend).")

    if args.map_out:
        rank_map = build_rank_map(fits)
        n_servers = len(next(iter(rank_map.values()))) if rank_map else 0
        os.makedirs(os.path.dirname(args.map_out), exist_ok=True)
        with open(args.map_out, "w", encoding="utf-8") as f:
            json.dump(rank_map, f, indent=2)
        print(f"\nWrote board-independent rank map → {args.map_out}")
        print(f"  {len(rank_map)} our-ranks × {n_servers} servers")

    # Per-board rating→internal(EGF)-rank curves for package_for_pwa. Each
    # threshold g sits at rank_index (g−0.5)+19, so a problem's raw rating
    # interpolates straight onto the 0..28 display axis the PWA already uses.
    if args.rating_map_out and rating_rank_fits:
        rr = {}
        for size, thr in rating_rank_fits.items():
            gs = sorted(thr)
            rr[str(size)] = {
                "ratings": [float(thr[g]) for g in gs],
                "rank_indices": [float((g - 0.5) + 19.0) for g in gs],
            }
        os.makedirs(os.path.dirname(args.rating_map_out), exist_ok=True)
        with open(args.rating_map_out, "w", encoding="utf-8") as f:
            json.dump(rr, f, indent=2)
        print(f"\nWrote per-board rating→rank curves → {args.rating_map_out}")
        print(f"  sizes: {', '.join(sorted(rr))}")


if __name__ == "__main__":
    main()
