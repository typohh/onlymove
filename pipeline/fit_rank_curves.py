"""Fit a board-independent server↔rank coupling from rank_surveys.

Model (our singular rank scale is the hub / common unit):

    server rank  ──one coupling per server──▶  OUR RANK  ◀──cubic, per board──  rating θ

  * OUR RANK IS EGF RANK. rating θ → our rank is fit per board with the same
    cumulative-link machinery used for the server couplings below, but in
    RATING space: the cubic (auxiliary.fit_statistics_shapes, pinned to that
    board's humanv0_8k/3k bots) is only the SHAPE PRIOR — its per-rank rating
    deltas are the baseline spacings — while the EGF survey declarations are
    the data and λ pulls the curve toward the cubic where EGF data is thin.
    package_for_pwa consumes these per-board curves (ratings/rating_to_rank.json)
    to display ranks, so the displayed rank and this map agree by construction.
  * server rank → our rank is ONE coupling per (non-EGF) server, board-
    independent. We convert every surveyed user's per-board θ to our(EGF)-rank
    via the right per-board curve, then pool all (our_rank, declared_server_rank)
    pairs across both boards and fit a per-server proportional-odds
    cumulative-link model in our-rank space. EGF itself is the hub: identity.

        P(declared ≤ g | our_rank) = expit(τ_{s,g} − our_rank)

    The thresholds τ live in our-rank units. The prior baseline is UNIFORM
    spacing (1 server rank = 1 our rank) — which in this cubic-linearized
    our-rank space IS the cubic shape — so each server keeps the cubic shape by
    default and the per-rank deviations log_e (regularized toward 0 with weight
    λ) only let it bend where the data demands. Survey data is sparse and noisy
    (≈3–34 declarations/server, several ranks of scatter), so λ is set strong:
    sparse servers collapse to a pure offset α (cubic shape, shifted), and only
    data-rich servers pick up a per-rank bend. The offset α is unregularized.

A user with both boards contributes their 9×9 our-rank AND 13×13 our-rank as
two pooled observations of the same declaration — both are on the one our-rank
scale, so this just adds data.

Output: pwa/tsumegos/rank_map.json = {our_rank_label: {server: server_rank}},
board-independent (the per-board difference lives in rating→our-rank, i.e. the
user's displayed rank, not in the translation). Re-run pipeline.fetch_database
first to refresh attempts/database.sql.

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
from scipy.optimize import minimize
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.compare_rank_surveys import (  # noqa: E402
    SERVERS,
    SQL_PATH_DEFAULT,
    STOPGAP_X_3K,
    STOPGAP_X_8K,
    fit_theta,
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


def theta_to_our_rank(theta, curve):
    """rating θ → our-rank scalar (1k=0, 1d=1, 20k=−19, 9d=9) via this board's
    cubic. rank_index = cubic⁻¹(θ); our-rank scalar = rank_index − 19. Used only
    as the fallback when a board has too little EGF data to fit rating→EGF."""
    return float(curve.inverse(np.array([theta]))[0]) - 19.0


def cubic_prior_for_rating_fit(curve, g_min, g_max):
    """(init_alpha, baseline_spacings) that puts fit_cumulative_link in RATING
    space with the cubic as the shape prior. Threshold g sits at the rating the
    cubic assigns to the boundary just below rank g — our-rank scalar g−0.5, i.e.
    rank_index (g−0.5)+19. baseline_spacings are the cubic's rating deltas
    between consecutive thresholds; init_alpha is the cubic rating at the first
    threshold. With log_e=0 this reproduces the cubic shape exactly, so λ pulls
    the fitted curve toward the cubic."""
    gs = list(range(g_min + 1, g_max + 1))           # the K threshold ranks
    boundary_ri = np.array([(g - 0.5) + 19.0 for g in gs])
    prior_rating = np.asarray(curve.forward(boundary_ri), dtype=float)
    init_alpha = float(prior_rating[0])
    baseline = np.diff(prior_rating) if len(prior_rating) > 1 else np.array([])
    return init_alpha, baseline


def rating_to_internal_rank(theta, thresholds):
    """Invert fitted rating-space thresholds to an internal(EGF)-rank scalar.
    thresholds {g: rating at the boundary just below rank g}. That boundary sits
    at internal-rank scalar g−0.5; linearly interp/extrapolate in rating."""
    gs = sorted(thresholds)
    xs = [thresholds[g] for g in gs]   # ascending ratings
    ys = [g - 0.5 for g in gs]         # internal-rank scalars at those boundaries
    return _interp_extrap(float(theta), xs, ys)


# ---------------------------------------------------------------------------
# Per-server cumulative-link fit in our-rank space.
#
# Ability = our_rank (continuous), category = declared server rank. Thresholds
# τ_g live in our-rank units. The baseline is UNIFORM spacing (1 our-rank per
# server-rank) — which in this cubic-linearized our-rank space IS the cubic
# shape — so the per-rank deviations log_e measure how the server bends away
# from the cubic. They're L2-regularized toward 0 with weight lambda_dev: a
# strong prior keeps a server pinned to the cubic shape (sparse servers
# collapse to a pure offset α), and only relaxes into per-rank shape where the
# data genuinely supports it. The offset α is unregularized.
# ---------------------------------------------------------------------------

def fit_cumulative_link(observations, g_min, g_max, lambda_dev,
                        baseline_spacings=None, init_alpha=None):
    """One proportional-odds (cumulative-link) fit.

      P(declared ≤ g | x) = expit(τ_g − x)
      spacings_g  = baseline_spacings_g · exp(log_e_g)
      τ_{g_min+1} = α
      τ_g         = α + Σ_{g'≤g} spacings_{g'}

    observations : list of (x, declared_g_scalar) tuples, where x is the
        continuous ability axis the thresholds live on.
    g_min, g_max : integer declared-rank range to fit thresholds over
    lambda_dev   : L2 weight on Σ log_e_g² (strength of the shape prior)
    baseline_spacings : prior gap between consecutive thresholds (length K-1).
        None ⇒ uniform spacing of 1. In cubic-linearized our-rank space uniform
        IS the cubic shape (server couplings). For the rating→rank direction the
        axis is rating (not rank-linear), so pass the cubic's per-rank rating
        deltas to make the cubic the shape prior there.
    init_alpha   : prior/init for the first threshold. None ⇒ g_min.

    Returns ({declared_g: τ_g}, nll, deviation) with τ in x units, or None if
    infeasible."""
    N_ranks = g_max - g_min + 1
    K = N_ranks - 1
    if K < 1 or len(observations) < 3:
        return None
    if baseline_spacings is None:
        baseline_spacings = np.ones(K - 1) if K > 1 else np.array([])
    else:
        baseline_spacings = np.asarray(baseline_spacings, dtype=float)

    our_ranks = np.array([t for t, _ in observations], dtype=float)
    rank_indices = np.array(
        [max(0, min(N_ranks - 1, int(round(g)) - g_min))
         for _, g in observations], dtype=int)

    def unpack(params):
        alpha = params[0]
        if K == 1:
            return np.array([alpha])
        log_e = params[1:]
        spacings = baseline_spacings * np.exp(log_e)
        return np.concatenate([[alpha], alpha + np.cumsum(spacings)])

    def neg_log_likelihood(tau):
        ps = np.empty(len(our_ranks))
        for i in range(len(our_ranks)):
            x = our_ranks[i]
            k = rank_indices[i]
            if k == 0:
                ps[i] = expit(tau[0] - x)
            elif k == N_ranks - 1:
                ps[i] = 1.0 - expit(tau[K - 1] - x)
            else:
                ps[i] = expit(tau[k] - x) - expit(tau[k - 1] - x)
        ps = np.clip(ps, 1e-12, 1.0)
        return -float(np.sum(np.log(ps)))

    def deviation(params):
        if K <= 1:
            return 0.0
        return float(np.sum(params[1:] ** 2))

    def objective(params):
        return (neg_log_likelihood(unpack(params))
                + lambda_dev * deviation(params))

    if init_alpha is None:
        init_alpha = float(g_min)
    x0 = (np.concatenate([[init_alpha], np.zeros(K - 1)]) if K > 1
          else np.array([init_alpha]))
    try:
        res = minimize(objective, x0, method="L-BFGS-B",
                       options={"maxiter": 2000})
    except Exception:
        return None
    if not np.isfinite(res.fun):
        return None

    tau = unpack(res.x)
    return ({g_min + k + 1: float(tau[k]) for k in range(K)},
            float(neg_log_likelihood(tau)),
            float(deviation(res.x)))


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
                        help="Strength of the cubic-shape prior: L2 weight on "
                             "the per-rank log-deviations from uniform spacing. "
                             "Large (≳30) ⇒ servers pinned to the cubic shape "
                             "(offset only); small (≲3) ⇒ free per-server bend "
                             "that overfits sparse data. The default keeps the "
                             "real bend on data-rich servers while leaving "
                             "sparse ones at a clean offset")
    parser.add_argument(
        "--map-out",
        default=os.path.join(PROJECT_ROOT, "pwa", "tsumegos", "rank_map.json"),
        help="Write the board-independent server rank-translation map (our "
             "rank → equivalent rank per server) here. Empty string skips it.")
    parser.add_argument(
        "--rating-map-out",
        default=os.path.join(PROJECT_ROOT, "ratings", "rating_to_rank.json"),
        help="Write the per-board rating→internal(EGF)-rank curves here for "
             "package_for_pwa to consume. Empty string skips it.")
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

    params_by_size = {}
    cubic_by_size = {}
    for size in sizes:
        params, _theta_arr, _scalar_arr, stopgap = load_problem_params(size)
        if params is None:
            print(f"  size {size}: skipped (missing rating files)")
            continue
        params_by_size[size] = params
        curve = _cubic_curve(stopgap)
        if curve is not None:
            cubic_by_size[size] = curve
        print(f"  size {size}: {len(params)} problem params, "
              f"cubic {'ready' if curve is not None else 'MISSING'}")

    print("\nFitting θ per (user, size) ...")
    user_theta = defaultdict(dict)  # user_id -> {size: raw rating θ}
    for uid in sorted(surveys["user_id"].unique()):
        prefix = uid[:8]
        for size, params in params_by_size.items():
            if cubic_by_size.get(size) is None:
                continue
            attempts = attempts_by_user_size.get((prefix, size), [])
            if len(attempts) < args.min_attempts:
                continue
            theta, n = fit_theta(attempts, params)
            if theta is not None and n >= args.min_attempts:
                user_theta[uid][size] = theta

    # ---- Stage A: per-board rating → internal rank, where INTERNAL RANK IS
    # EGF RANK. Same cumulative-link as the server couplings below, but in
    # RATING space with the cubic as the shape prior (cubic per-rank rating
    # deltas as baseline spacings); the EGF survey declarations are the data and
    # λ pulls the curve toward the cubic where EGF data is thin. Fit over a wide
    # fixed rank band so the curve is defined across the whole display range.
    egf_lo, egf_hi = -20, 10  # 21k..10d
    egf_surveys = surveys[surveys["server"] == "EGF"]
    rating_rank_fits = {}  # size -> {egf_g: τ_rating}
    print(f"\nFitting rating → internal(EGF) rank per board "
          f"(cubic-shape prior λ = {args.lam}):")
    for size in sizes:
        curve = cubic_by_size.get(size)
        if curve is None:
            continue
        obs = []
        for _, row in egf_surveys.iterrows():
            g = row["rank_scalar"]
            th = user_theta.get(row["user_id"], {}).get(size)
            if th is not None and g is not None and np.isfinite(g):
                obs.append((th, float(g)))
        if len(obs) < args.min_server_samples:
            print(f"  size {size}: only {len(obs)} EGF declarations with θ — "
                  f"falling back to the raw cubic for this board")
            continue
        init_alpha, baseline = cubic_prior_for_rating_fit(curve, egf_lo, egf_hi)
        result = fit_cumulative_link(obs, egf_lo, egf_hi, args.lam,
                                     baseline_spacings=baseline,
                                     init_alpha=init_alpha)
        if result is None:
            print(f"  size {size}: rating→rank fit failed — raw cubic fallback")
            continue
        thresholds, nll, dev = result
        rating_rank_fits[size] = thresholds
        print(f"  size {size}: {len(obs)} EGF declarations, NLL {nll:.2f}, "
              f"shape-dev {dev:.3f}")

    def theta_to_internal_rank(theta, size):
        thr = rating_rank_fits.get(size)
        if thr is not None:
            return rating_to_internal_rank(theta, thr)
        return theta_to_our_rank(theta, cubic_by_size[size])  # cubic fallback

    user_our_rank = defaultdict(dict)  # user_id -> {size: internal(EGF)-rank scalar}
    for uid, by_size in user_theta.items():
        for size, theta in by_size.items():
            user_our_rank[uid][size] = theta_to_internal_rank(theta, size)
    both = sum(1 for r in user_our_rank.values() if len(r) == 2)
    print(f"  internal(EGF) rank for {len(user_our_rank)} users "
          f"({both} have both sizes)")

    # Pool (internal_rank, declared_server_rank) across every user × board. A
    # both-boards user contributes their 9×9 AND 13×13 internal rank — both on
    # the one EGF scale, so this just adds evidence for the same declaration.
    # EGF itself is the hub (identity), so it gets no coupling fit.
    by_server = defaultdict(list)
    per_board_counts = defaultdict(lambda: defaultdict(int))  # server -> size -> n
    for _, row in surveys.iterrows():
        uid, server, g = row["user_id"], row["server"], row["rank_scalar"]
        if server == "EGF" or g is None or not np.isfinite(g):
            continue
        for size, our_rank in user_our_rank.get(uid, {}).items():
            by_server[server].append((our_rank, float(g)))
            per_board_counts[server][size] += 1

    print(f"\nFitting one cumulative-link per server in internal(EGF)-rank space "
          f"(cubic-shape prior λ = {args.lam}, ≥{args.min_server_samples} "
          f"pooled samples, rank padding = ±{args.rank_padding}):")

    fits = {}  # server -> (thresholds, nll, dev)
    skipped = []
    for server in SERVERS:
        obs = by_server.get(server, [])
        if len(obs) < args.min_server_samples:
            if obs:
                skipped.append((server, len(obs)))
            continue
        declared = [int(round(g)) for _, g in obs]
        g_min = min(declared) - args.rank_padding
        g_max = max(max(declared) + args.rank_padding, args.extend_to)
        result = fit_cumulative_link(obs, g_min, g_max, args.lam)
        if result is None:
            print(f"  {server:<8}: fit failed")
            continue
        fits[server] = result

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
        for _, g in by_server[server]:
            counts[int(round(g))] += 1
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
