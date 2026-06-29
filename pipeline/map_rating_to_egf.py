"""Map the v2 combined rating (b_p, in ability-θ units) to EGF rank space, and
fit the server↔EGF rank couplings — SELF-CONTAINED, no v1 artifacts required.

The v2 model (pipeline.train_rating) puts problem difficulty b_p and user
ability on ONE shared logit scale across both boards, so the rating→rank curve
is a SINGLE shared curve. This step produces, from rank_surveys alone:

  rating_to_rank.json   per-board (identical) {ratings, rank_indices}
  rank_map.json         per-server internal(EGF)↔server rank couplings

written to ratings/ (the backend). package_for_pwa is the SINGLE writer of the
served pwa/tsumegos/ — it copies these two curve files across (and change-detects
the rest). Both come out of ONE joint fit (fit_joint_shared) — v2 no longer reads
v1's rank_map.

WHY 80%, not raw θ (the calibration fix):
  The PWA derives a user's rank as the rating level they clear at CALIBRATION_P
  = 80% (pwa/js/rank-decomposition.js, model.js, == utils.fit_rank_curves),
  then maps it through rating_to_rank.json. Raw θ is the 2PL 50%-point
  (P = σ(a_p·(θ − b_p)), so θ = b_p ⇒ P = 0.5). Anchoring raw θ → EGF therefore
  calibrates the curve at 50% while the client measures users at 80% — a
  systematic offset in every displayed rank. So we instead estimate each survey
  user's 80%-CLEAR rating against v2's OWN per-problem ratings (the same
  fit_user_rating the v1 fitter uses, pooled across boards since b_p is one
  shared scale) and anchor THAT → EGF. The curve then speaks the PWA's language.

Model (internal rank ≡ EGF rank is the hub / identity), estimated TOGETHER:

    rating R ──shared curve m──▶ INTERNAL(EGF) RANK ──per server──▶ server rank

  * R = the v2-rating at which a user clears 80% (fit_user_rating, pooled).
  * m: a MONOTONE per-segment fit with the EGF statistics.txt cubic SHAPE as a
    PRIOR. The gap between consecutive rating-thresholds is `β·ΔS(g)·exp(δ_g)`
    (>0 ⇒ monotone); δ_g=0 reproduces the old rigid affine `a_g = α + β·S(g)`, and
    `λ_shape·Σδ_g²` penalises deviation from the shape. S = the ΔAIC-best fit to the
    even-game win-rate table (utils.statistics_shapes). The shape is NOT a magic
    constant: EGF data rejects a linear rank↔rating link (ΔAIC 1652 vs cubic 0),
    and the bot Elo ladder + our human R(rank) independently show the same convex
    acceleration — so it's the right EXTRAPOLATION backbone for the sparse tails.
    But as a HARD 2-param affine it under-fit the dan anchors by ~3 ranks (the EGF
    declarers are the only dan-heavy cohort, and the data's real shape there isn't
    the borrowed cubic). Softening it to a prior lets the dense kyu–dan EGF data
    bend the curve to the anchors (held-out CV picks λ_shape at a real bias-variance
    minimum — see docs/rank_mapping_experiments.md), halving the EGF residual; the
    irreducible remainder is the rating model's strong-player compression, not m.
    (NOT v1's cubic, pinned to v1's scale via humanv0 bots; this fits v2's b_p.)
  * Each non-EGF server: P(declared ≤ h | internal rank x) = expit(τ_{s,h} − x),
    free ordinal thresholds, uniform-spacing prior (--lambda). EGF feeds the
    rating curve directly (the identity hub); other servers couple in on top.

Deploying v2 = run this (writes the curve to ratings/), then package_for_pwa
(now defaults to v2) — package_for_pwa is the single PWA writer: it copies the
curve across and change-detects ratings/embeddings. No manual copy.

Run pipeline.fetch_database first to refresh attempts/database.sql (the survey +
attempts source); this step then needs neither GPU nor the heavy pipeline.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, latest_per_user_server, load_attempts_for_users,
    load_rank_surveys, scalar_to_rank,
)
from utils.statistics_shapes import fit_shape  # noqa: E402
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, build_server_curves, fit_user_rating, interp_extrap_vec,
    ordinal_nll_vec, _unpack_thresholds,
)
from utils.configurations import ALL_BOARD_SIZES  # noqa: E402
from utils.ratings_io import (  # noqa: E402
    FLOOR_CSV, QUALITY_CSV, RATINGS_CSV, load_lapse, load_value_map,
)

SIZES = list(ALL_BOARD_SIZES)

# stats-shape rank_index axis: 20k=0 … 1k=19 … 1d=20. EGF scalar g: 1k=0, 1d=1.
# So rank_index(g) = g + 19. The rating→EGF curve uses the EGF statistics.txt
# cubic SHAPE (ΔAIC-best fit to the even-game win-rate table; see
# docs/rank_mapping_experiments.md), with only the affine (α, β) fit to surveys.
RATING_SHAPE = "cubic"


def load_ratings(v2_dir, size):
    """{tsumego_id: b_p rating} from ratings/<size>/tsumego_ratings_final.csv
    — the same shared-scale black-box scalar package_for_pwa ships for v2."""
    return load_value_map(
        os.path.join(v2_dir, str(size), RATINGS_CSV), "rating", missing_ok=True)


def load_slopes(v2_dir, size):
    """{tsumego_id: a_p} from ratings/<size>/tsumego_quality_final.csv — the raw
    2PL discrimination (the `quality` column). fit_user_rating uses these as the
    per-problem success-curve slope so the survey-user 80%-clear ratings (and thus
    M) are anchored with the SAME per-problem a_p the PWA's estimator uses."""
    return load_value_map(
        os.path.join(v2_dir, str(size), QUALITY_CSV), "quality", missing_ok=True)


def load_floors(v2_dir, size):
    """{tsumego_id: c_p} from ratings/<size>/tsumego_floor_final.csv — the per-problem
    4PL guessing floor. fit_user_rating uses these (with the global ceiling λ) so the
    survey-user 80%-clear ratings that anchor M match the client's 4PL rank estimator
    (pwa/js/irt.js) exactly — same c_p/λ or the anchor↔client loop reopens."""
    return load_value_map(
        os.path.join(v2_dir, str(size), FLOOR_CSV), "floor", missing_ok=True)


def _shape_at(ys):
    """The EGF statistics.txt RATING_SHAPE value at each EGF-boundary scalar in
    `ys` (rank_index = g + 19). Monotone increasing — used as the fixed shape of
    the rating→EGF curve, scaled by a 2-param affine fit to surveys."""
    params, theta = fit_shape(RATING_SHAPE)
    return np.asarray(theta(np.asarray(ys, float) + 19.0, params), dtype=float)


def fit_joint_shared(decls, egf_lo, egf_hi, lam, lam_shape,
                     rank_padding, extend_to, min_server_samples):
    """Jointly fit ONE shared rating→internal(EGF)-rank curve and per-server
    internal-rank→server-rank couplings over all `decls`.

    The rating→EGF curve is a MONOTONE per-segment fit with the EGF statistics.txt
    cubic shape as a PRIOR (not a hard constraint): the gap between consecutive
    rating-thresholds is `β·ΔS(g)·exp(δ_g)` (strictly positive ⇒ monotone), so δ_g=0
    reproduces the old rigid affine `a_g = α + β·S(g)` exactly, and `lam_shape·Σδ_g²`
    penalises each segment's slope for deviating from the shape. Where the EGF
    declarations are dense (kyu–dan) the likelihood overrides the prior and the
    curve bends to the anchors (the old 2-param affine under-fit the dan anchors by
    ~3 ranks); where they are sparse (the tails) nothing overrides it, so
    extrapolation stays on the cubic shape. `lam_shape` is held-out-CV-selected (a
    real bias-variance minimum; see docs/rank_mapping_experiments.md). A free LINEAR
    curve was decisively rejected by the same EGF data (ΔAIC 1652) — the shape prior
    keeps the tails from that tilt. Server couplings stay free ordinal thresholds
    with a uniform-spacing prior (`lam`).

    decls : iterable of (server, rating R, declared g_scalar). There is no board
    split — b_p is one shared scale, so a user contributes ONE R (pooled across
    boards) and one declaration per server.

    Returns (rating_knots, egf_thr_ranks, server_fits):
      rating_knots : (K,) fitted rating at each EGF-rank boundary g−0.5
      egf_thr_ranks: [egf_lo+1 .. egf_hi]  (the K threshold ranks)
      server_fits  : {server: ({declared_g: τ_internal}, nll, dev)}
    Needs ≥1 EGF declaration to anchor the absolute internal-rank scale.
    """
    N_egf = egf_hi - egf_lo + 1
    egf_thr_ranks = list(range(egf_lo + 1, egf_hi + 1))   # K = N_egf-1 ranks
    ys = np.array([g - 0.5 for g in egf_thr_ranks], dtype=float)  # EGF scalar
    S = _shape_at(ys)                                      # cubic shape (prior MEAN)
    dS = np.diff(S)                  # shape's per-segment slope (>0, S monotone)
    nseg = len(dS)                   # = K_egf-1 free per-segment shape-deviations

    # --- bucket declarations ---
    egf_R, egf_k = [], []
    server_rows = defaultdict(list)   # server -> [(rating, g_int)]
    for server, R, g in decls:
        gi = int(round(g))
        if server == "EGF":
            egf_R.append(R)
            egf_k.append(max(0, min(N_egf - 1, gi - egf_lo)))
        else:
            server_rows[server].append((R, gi))
    egf_R = np.asarray(egf_R, dtype=float)
    egf_k = np.asarray(egf_k, dtype=int)
    if len(egf_R) == 0:
        sys.exit("No EGF declarations — cannot anchor the internal-rank scale.")

    # --- init the curve's affine (α, β) from a robust linear EGF pre-fit ---
    beta, alpha = np.polyfit(egf_R, egf_k.astype(float) + egf_lo, 1)
    if beta <= 0:
        sys.exit("Non-positive rating→EGF slope in pre-fit — check surveys.")
    prior_rating = (ys - alpha) / beta            # rating at each EGF boundary
    b0 = (prior_rating[-1] - prior_rating[0]) / (S[-1] - S[0])
    a0 = float(prior_rating[0] - b0 * S[0])       # rating-thresholds = a0 + b0·S
    x0_rating = [a0, float(np.log(max(b0, 1e-3)))]   # [α, log β]  (β>0 ⇒ monotone)

    # --- per-server blocks (internal-rank thresholds), as in fit_rank_curves ---
    server_blocks = {}
    for server, rows in server_rows.items():
        if len(rows) < min_server_samples:
            continue
        declared = [g for _, g in rows]
        g_min = min(declared) - rank_padding
        g_max = max(max(declared) + rank_padding, extend_to)
        K = g_max - g_min
        if K < 1:
            continue
        N = g_max - g_min + 1
        rating = np.array([r for r, _ in rows], dtype=float)
        kk = np.array([max(0, min(N - 1, g - g_min)) for _, g in rows], dtype=int)
        server_blocks[server] = {
            "g_min": g_min, "g_max": g_max,
            "baseline": np.ones(K - 1) if K > 1 else np.array([]),
            "rating": rating, "k": kk}

    # --- parameter layout: rating curve [α, log β, δ_1..δ_nseg], then per-server
    #     blocks. δ init 0 ⇒ the fit STARTS exactly at today's α+β·S affine; the
    #     shape penalty (lam_shape) pulls it back there wherever the EGF data is too
    #     sparse to override. ---
    srv_layout = []   # (server, start, n)
    x0 = list(x0_rating) + [0.0] * nseg
    idx = 2 + nseg
    for server, blk in server_blocks.items():
        n = len(blk["baseline"]) + 1
        srv_layout.append((server, idx, n))
        x0.append(float(blk["g_min"]))
        x0.extend([0.0] * len(blk["baseline"]))
        idx += n
    x0 = np.asarray(x0, dtype=float)

    def unpack(params):
        # Rating thresholds as α plus a running sum of STRICTLY POSITIVE gaps, so the
        # curve is monotone by construction. Each gap is the shape's local slope β·dS
        # times exp(δ), so δ=0 ⇒ a = α + β·S exactly (today's curve) and the EGF data
        # can stretch/compress any segment (δ≠0) at the cost of the shape penalty.
        beta = np.exp(params[1])
        a = np.empty(len(S))
        a[0] = params[0] + beta * S[0]
        if nseg:
            gaps = beta * dS * np.exp(params[2:2 + nseg])   # > 0 ⇒ strictly increasing
            a[1:] = a[0] + np.cumsum(gaps)
        b, logs = {}, {}
        for server, st, n in srv_layout:
            b[server] = _unpack_thresholds(params[st:st + n],
                                           server_blocks[server]["baseline"])
            logs[server] = params[st + 1:st + n]
        return a, b, logs

    def objective(params):
        a, b, logs = unpack(params)
        nll = ordinal_nll_vec(a, egf_R, egf_k)
        for server, blk in server_blocks.items():
            x_int = interp_extrap_vec(blk["rating"], a, ys)
            nll += ordinal_nll_vec(b[server], x_int, blk["k"])
        pen = lam * sum(float(np.sum(lg ** 2)) for lg in logs.values())
        # Shape prior: penalise each segment's slope for deviating from the cubic
        # shape (δ=0). High lam_shape ⇒ rigid α+β·S; low ⇒ free monotone fit.
        if nseg:
            pen += lam_shape * float(np.sum(params[2:2 + nseg] ** 2))
        return nll + pen

    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 10000})
    a, b, logs = unpack(res.x)

    server_fits = {}
    for server, blk in server_blocks.items():
        thr = b[server]
        gs = list(range(blk["g_min"] + 1, blk["g_max"] + 1))
        thresholds = {gs[i]: float(thr[i]) for i in range(len(gs))}
        x_int = interp_extrap_vec(blk["rating"], a, ys)
        nll = ordinal_nll_vec(thr, x_int, blk["k"])
        dev = float(np.sum(logs[server] ** 2))
        server_fits[server] = (thresholds, nll, dev)
    return a, egf_thr_ranks, server_fits


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(PROJECT_ROOT, "ratings"))
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--min-attempts", type=int, default=20,
                    help="Drop survey users with fewer pooled attempts.")
    ap.add_argument("--min-server-samples", type=int, default=3,
                    help="Skip a server with fewer pooled declarations.")
    ap.add_argument("--rank-padding", type=int, default=2,
                    help="Extend a server's threshold range this many ranks "
                         "past its min/max declared rank.")
    ap.add_argument("--extend-to", type=int, default=9,
                    help="Force each server range to cover at least up to this "
                         "rank scalar (9 = 9d).")
    ap.add_argument("--lambda", type=float, default=5.0, dest="lam",
                    help="Uniform-spacing prior strength for the SERVER couplings. "
                         "Default 5 (CV-selected: beats 10 on band-balanced error "
                         "at every seed — it lets the couplings down-rate online "
                         "dan inflation; see docs/rank_mapping_experiments.md). "
                         "Sparse servers still stay near a clean offset.")
    ap.add_argument("--lambda-shape", type=float, default=3.0, dest="lam_shape",
                    help="Prior strength pulling the rating→EGF curve's per-segment "
                         "slopes toward the fixed cubic shape. HIGH ⇒ today's rigid "
                         "2-param affine (under-fits the dan anchors); LOW ⇒ free "
                         "monotone fit to the EGF declarations. Where the EGF data is "
                         "dense the likelihood overrides the prior (kills the dan "
                         "under-fit by interpolation); where it is sparse (the tails) "
                         "nothing overrides it, so extrapolation stays on the shape. "
                         "CV-selected — see docs/rank_mapping_experiments.md.")
    ap.add_argument("--egf-lo", type=int, default=-20,
                    help="Lowest EGF boundary (egf_lo+1 = weakest threshold).")
    ap.add_argument("--egf-hi", type=int, default=10, help="Highest EGF boundary.")
    args = ap.parse_args()

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — cannot anchor the curve.")
    surveys = latest_per_user_server(surveys)
    print(f"  {len(surveys)} latest declarations from "
          f"{surveys['user_id'].nunique()} users")

    ratings_by_size = {s: load_ratings(args.dir, s) for s in SIZES}
    slopes_by_size = {s: load_slopes(args.dir, s) for s in SIZES}
    floors_by_size = {s: load_floors(args.dir, s) for s in SIZES}
    merged_ratings = {}
    merged_slopes = {}
    merged_floors = {}
    for s in SIZES:
        if not ratings_by_size[s]:
            print(f"  size {s}: no v2 ratings at {args.dir}/{s}/ (skipped)")
        merged_ratings.update(ratings_by_size[s])   # b_p is one shared scale
        merged_slopes.update(slopes_by_size[s])     # a_p, same shared scale
        merged_floors.update(floors_by_size[s])     # c_p, 4PL guessing floor
    if not merged_ratings:
        sys.exit("No v2 per-problem ratings found — run train_rating first.")
    lapse = load_lapse(args.dir)                     # global 4PL ceiling λ (= client's)

    attempts = load_attempts_for_users(args.sql, set(surveys["user_id"].unique()))

    # Each user's 80%-CLEAR v2-rating, POOLED across boards (shared scale). This
    # is the same quantity the PWA computes per user (fit_user_rating @ 80%, full
    # 4PL with per-problem floor c_p + ceiling λ — same as the client's irt.js),
    # so anchoring it → EGF makes the displayed rank match the survey rank.
    print(f"\nEstimating each user's {int(CALIBRATION_P * 100)}%-clear v2-rating "
          f"(pooled across boards; 4PL, λ={lapse:.4f}) ...")
    user_rating = {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < args.min_attempts:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=args.min_attempts,
                               slopes=merged_slopes, floors=merged_floors,
                               lapse=lapse)
        if R is not None and n >= args.min_attempts:
            user_rating[pref] = R
    print(f"  {len(user_rating)} users with a usable rating "
          f"(≥{args.min_attempts} attempts)")

    decls = []  # (server, R, declared_scalar)
    for _, row in surveys.iterrows():
        pref = str(row["user_id"])[:8]
        g = row.get("rank_scalar")
        if pref not in user_rating or g is None or pd.isna(g):
            continue
        decls.append((row["server"], user_rating[pref], float(g)))
    n_egf = sum(1 for srv, _, _ in decls if srv == "EGF")
    print(f"\nJoint fit over {len(decls)} declarations "
          f"[{n_egf} EGF anchoring the scale; rating curve = monotone fit with a "
          f"'{RATING_SHAPE}' statistics.txt shape PRIOR (λ_shape = {args.lam_shape}); "
          f"server prior λ = {args.lam}, ≥{args.min_server_samples} samples]")

    rating_knots, egf_thr_ranks, server_fits = fit_joint_shared(
        decls, args.egf_lo, args.egf_hi, lam=args.lam, lam_shape=args.lam_shape,
        rank_padding=args.rank_padding, extend_to=args.extend_to,
        min_server_samples=args.min_server_samples)

    # In-sample calibration check on the EGF anchors (predicted vs declared EGF).
    ys = np.array([g - 0.5 for g in egf_thr_ranks], dtype=float)
    egf_R = np.array([R for srv, R, _ in decls if srv == "EGF"])
    egf_g = np.array([g for srv, _, g in decls if srv == "EGF"])
    if len(egf_R) >= 2:
        pred = interp_extrap_vec(egf_R, rating_knots, ys)
        res = pred - egf_g
        ss_tot = float(np.sum((egf_g - egf_g.mean()) ** 2))
        r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        print(f"  EGF-anchor calibration: R² {r2:.3f} | "
              f"RMSE {np.sqrt(np.mean(res ** 2)):.2f} | bias {res.mean():+.2f} ranks")

    # --- write the shared rating→rank curve (both board keys, identical) ---
    gs = egf_thr_ranks
    curve = {
        "ratings": [float(r) for r in rating_knots],
        "rank_indices": [float((g - 0.5) + 19.0) for g in gs],
    }
    rr = {str(s): curve for s in SIZES}
    server_curves = build_server_curves(server_fits)   # server↔EGF couplings, built here

    # Write ONLY to the backend (ratings/). package_for_pwa is the single writer of
    # the served pwa/tsumegos/ — run `python -m pipeline.package_for_pwa` to publish
    # (it copies these two curve files across and change-detects the rest).
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "rating_to_rank.json"), "w") as f:
        json.dump(rr, f, indent=2)
    with open(os.path.join(args.dir, "rank_map.json"), "w") as f:
        json.dump(server_curves, f, indent=2)

    # report
    print(f"\nrating range mapped: {scalar_to_rank(gs[0])}→rating "
          f"{rating_knots[0]:.2f} ... {scalar_to_rank(gs[-1])}→rating "
          f"{rating_knots[-1]:.2f}")
    fitted = [s for s in SERVERS if s in server_curves and s != "EGF"]
    print(f"wrote {args.dir}/rating_to_rank.json (shared 9≡13) + rank_map.json "
          f"({len(fitted)} servers + EGF identity: {', '.join(fitted)})\n"
          f"  → publish with: python -m pipeline.package_for_pwa  (defaults to v2)")


if __name__ == "__main__":
    main()
