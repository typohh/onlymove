#!/usr/bin/env python3
"""What the rank mapping's two anchoring decisions are worth, measured by running
the mapping's OWN fit as comparison arms.

Two claims in the mapping's design carry a magnitude, and both are answered by
re-running `pipeline.map_rating_to_rank`'s fit on a changed input rather than by
restating any of its machinery. Nothing here writes to `ratings/`: every arm is
fitted in memory from the same anchor rows the production step builds, and the
shipped arm is checked against the published curve before anything is quoted.

THE SCREEN (`rankmap.screen_*`). The mapping fits once, scores every declaration
by the contamination mixture's own junk posterior, drops those past one half and
refits. The UNSCREENED arm is therefore free: it is the first fit, the one the
screen is taken against, and the shipped arm is the second.

  rankmap.screen_curve_shift  the largest horizontal displacement between the two
                              curves AT EQUAL RANK, over the labelled ladder the
                              curve spans, in rating units. "Horizontal" because
                              the curve is stored as a rating per rank knot and
                              both arms share the rank grid, so this is a
                              difference of ratings at a common rank.
  rankmap.screen_rank_shift   the same displacement read the other way round —
                              the largest gap between the two curves' ranks AT
                              EQUAL RATING, over the rating span both cover, in
                              ranks. A rating unit is worth wildly different
                              numbers of ranks at different points of the ladder,
                              so the rating-unit figure alone does not say
                              whether anyone's shown rank moves.
  rankmap.screen_likelihood_bar   the LOOSEST likelihood bar clearing the junk
                              posterior's one-half boundary implies, as a
                              one-in-N. Exactly: the posterior is
                              junk/((1−eps)·P + junk) with junk = eps/(K+1) over
                              a server's K+1 categories, so it passes one half
                              exactly when P < eps/((K+1)(1−eps)). Computed per
                              server off the first fit's own couplings — the ones
                              the screen scored against — and reported at the
                              server where the bar sits highest, so it is the
                              weakest form of the claim.
  rankmap.screen_mildest_miss the smallest miss among the declarations the screen
                              removes: how far the mapping places that user from
                              the band they declared, in ranks. What makes the
                              bar severe rather than nominal is that even the
                              mildest row clearing it is a miss of this size.

THE FEDERATION ANCHOR (`rankmap.*_dan_bias`). The latent is a consensus rank with
every server, the European federation included, a FREE coupling off it. The arm
is the same fit with that one coupling HELD AT IDENTITY, which is what "anchored
on the federation alone" means: the latent stops being a consensus and becomes
that federation's own ladder, every other server couples off it, and it is the
one server with no coupling of its own to absorb its offset — so whatever the
shared curve cannot satisfy for everybody lands on it. Every declaration still
enters, and everything else (the shape prior, the spacing prior, the
contamination mixture, the screen-and-refit) is left exactly as production runs
it, so the two arms differ in the anchoring and in nothing else. Anchoring on the
federation's DECLARATIONS alone is a different arm and not this one: dropping
every other server takes the kyu end's only data with it, so what it would
measure is a starved curve rather than a privileged hub.

  rankmap.federation_dan_shift  what the arm would SHOW: where it puts the
  rankmap.federation_kyu_shift  surveyed users the shipped mapping places in the
                              dan bands, and in the kyu bands, against where the
                              shipped mapping puts them, in ranks. This is the
                              headline moving onto the federation's ladder, and
                              it is uneven along the ladder, which is what makes
                              it an idiosyncrasy rather than a relabelling.
  rankmap.federation_dan_bias   the arm's mean signed error against declarations
                              in the dan band, in the declaring server's own
                              units — negative means the mapping places a
                              declarer weaker than they say they are.
  rankmap.consensus_dan_bias    the same error under the shipped mapping. The two
                              are the accuracy half of the comparison, and they
                              come out level: the anchoring decides what the
                              headline number MEANS, not how well the fit
                              predicts what anyone declares.

The errors are read the one way a coupling permits: forward, into the server's
own units, on the time-adjusted rating, since a coupling's thresholds are free in
location and an error taken on the latent scale would mean a different thing on
every server. Both arms are scored on the rows BOTH kept and bucketed by one
shared placement, so the comparison cannot be a change of cohort; the run also
prints each arm on its own rows at its own bucketing.

Every input is checked on the way: the anchor rows reproduce the shipped curve to
floating point, the screen reproduces the published exclusion list exactly, the
arm's federation coupling really is the identity, and both arms' couplings form.

    ./.venv/bin/python3 -m ablations.rank_map_arm_figures
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from pipeline import map_rating_to_rank as mapping  # noqa: E402
from utils.rating.fit_rank_curves import fit_user_rating  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SQL_PATH_DEFAULT, load_rank_surveys, rank_band, value_to_rank,
)
from utils.rating.ratings_io import (  # noqa: E402
    internal_to_server, interp_extrap_vec, load_declaration_exclusions,
    load_lapse, load_rating_rank_curve,
)

TIER = "about"

RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
# The one server the federation arm anchors on, named as the survey names it.
FEDERATION = "EGF"
# The dan band the mapping's accuracy is read over, half-open on the one internal
# rank scale: 0 is the 1k/1d boundary and every rank is one unit wide, so this is
# 1d through 4d — the same band the shipped mapping's calibration table reports.
DAN_BAND = (0.0, 4.0)
# The shipped curve is refitted here from the same inputs, so it must come back
# identical up to the optimizer's own arithmetic; anything larger means the
# published mapping was not produced from the artifacts now on disk.
CURVE_MATCH_TOL = 1e-6


# --- the anchor rows -------------------------------------------------------
def _anchor_rows(ratings_dir, sql_path, sqlite_path):
    """`map_rating_to_rank`'s anchor rows, rebuilt so an arm can be fitted on a
    subset of them. Every quantity is computed by the production step's own
    helpers — this restates its assembly order and nothing else — and the
    reproduction is checked afterwards against the published curve.

    Returns (decls, keys, users): the anchor rows and their per-declaration keys
    exactly as `fit_consensus` and `detect_junk_declarations` take them, plus
    {user_id: (pooled career rating, de-meaned time feature)} for reading a
    fitted curve at where each surveyed user actually sits."""
    surveys = load_rank_surveys(sql_path)
    require(surveys is not None and not surveys.empty,
            f"no rank_surveys rows in {sql_path} — run pipeline.fetch_database")
    surveys = surveys[surveys["rank_value"].notna()].copy()
    require(len(surveys) > 0, "no declarations carry a parseable rank")
    surveys["t_days"] = (pd.to_datetime(surveys["created_at"], utc=True, format="mixed")
                         .astype("int64") / 86400e9)

    ratings, slopes, floors = {}, {}, {}
    for size in mapping.SIZES:
        ratings.update(mapping.load_ratings(ratings_dir, size))
        slopes.update(mapping.load_slopes(ratings_dir, size))
        floors.update(mapping.load_floors(ratings_dir, size))
    require(ratings, f"no per-problem ratings under {ratings_dir} — run pipeline.train_mirt")
    lapse = load_lapse(ratings_dir)

    attempts = mapping.load_timed_attempts(sqlite_path, set(surveys["user_id"].unique()))
    med_min = mapping.load_user_median_solve_minutes(sqlite_path)

    blocks, logT, pooled = {}, {}, {}
    for uid in surveys["user_id"].unique():
        times, rows = attempts.get(uid, (np.zeros(0), []))
        R, n = fit_user_rating(rows, ratings, min_n=mapping.MIN_ATTEMPTS,
                               slopes=slopes, floors=floors, lapse=lapse)
        if R is None or n < mapping.MIN_ATTEMPTS:
            continue
        blks = mapping.block_ratings(times, rows, ratings, slopes, floors, lapse)
        if not blks:
            blks = [(float(np.mean(times)), R, 1.0)]
        blocks[uid], pooled[uid] = blks, R
        pref = uid[:8]
        if pref in med_min and med_min[pref] > 0:
            logT[uid] = float(np.log10(med_min[pref]))
    require(blocks, "no surveyed user has enough attempts for a rating")
    timed = [logT[u] for u in blocks if u in logT]
    center = float(np.mean(timed)) if timed else 0.0

    decls, keys = [], []
    for (uid, server), srows in (surveys[surveys["user_id"].isin(blocks)]
                                 .groupby(["user_id", "server"])):
        T = (logT[uid] - center) if uid in logT else 0.0
        dt = srows["t_days"].to_numpy(float)
        dg = np.array([rank_band(v) for v in srows["rank_value"].to_numpy(float)])
        assign = defaultdict(list)
        for tc, R, m in blocks[uid]:
            assign[int(np.argmin(np.abs(dt - tc)))].append((R, m))
        for j, brs in assign.items():
            mass = sum(m for _, m in brs)
            for R, m in brs:
                decls.append((server, R, int(dg[j]), T, m / mass if mass > 0 else 1.0 / len(brs)))
                keys.append((uid, server, int(dg[j])))
    require(decls, "no anchor rows formed")
    users = {uid: (pooled[uid], (logT[uid] - center) if uid in logT else 0.0)
             for uid in blocks}
    return decls, keys, users


def _screen_and_refit(decls, keys, label, **fit_kw):
    """The mapping's own fit-screen-refit, run on whatever rows it is handed.

    Returns (first, final, kept, dropped): the unscreened fit, the shipped-shape
    fit, the rows the second one saw, and the declarations the screen removed.
    Both fits are (a, alpha, ys, rank_knots, server_fits)."""
    print(f"\n--- {label}: {len(decls)} anchor rows, {len(set(keys))} declarations")
    first = mapping.fit_consensus(decls, **fit_kw)
    a, alpha, ys, _, server_fits = first
    dropped, _ = mapping.detect_junk_declarations(decls, keys, a, ys, alpha, server_fits)
    if not dropped:
        return first, first, list(decls), dropped
    drop_keys = {d["key"] for d in dropped}
    kept = [d for d, k in zip(decls, keys) if k not in drop_keys]
    return first, mapping.fit_consensus(kept, **fit_kw), kept, dropped


def _rank_gap(a_ref, a_arm, ys):
    """(grid, arm rank − reference rank) over the rating span both curves cover.

    Both curves are piecewise linear in the rating, so their difference is too and
    its extremes sit on the union of their knots — evaluating there is exact, not
    a sampling."""
    lo, hi = max(a_ref[0], a_arm[0]), min(a_ref[-1], a_arm[-1])
    require(hi > lo, "the two curves cover no common rating range")
    grid = np.unique(np.concatenate([a_ref, a_arm, [lo, hi]]))
    grid = grid[(grid >= lo) & (grid <= hi)]
    return grid, (interp_extrap_vec(grid, a_arm, ys)
                  - interp_extrap_vec(grid, a_ref, ys))


def _couplings(server_fits):
    """The fitted couplings in the shape `internal_to_server` reads: {server:
    (latent knots ascending, that server's own rank at each)}. The fit stores a
    threshold per server-rank band EDGE, and an edge is the band integer itself, so
    each pair is already a continuous latent↔server correspondence."""
    out = {}
    for server, thr in server_fits.items():
        if not thr or len(thr) < 2:
            continue
        gs = sorted(thr)
        out[server] = (np.array([thr[g] for g in gs], float),
                       np.array([float(g) for g in gs], float))
    require(out, "the fit produced no usable coupling")
    return out


def _band_error(arm, rows, band, band_by=None):
    """The mean signed error against declarations inside one rank band, in that
    server's OWN units — the only comparable direction, since a coupling's
    thresholds are free in location and an error read on the latent scale would
    mean something different on every server.

    Per row: map the time-adjusted rating through the arm's curve and that server's
    coupling, and subtract the declared band's centre. Rows are bucketed by the
    arm's own UNDOCKED rank, the placement the band label refers to, unless
    `band_by` supplies one placement per row so two arms can be scored on the same
    rows. Weights are the anchor weights, so the answer reads per declaration
    rather than per career block."""
    a, alpha, ys, _, server_fits = arm
    maps = _couplings(server_fits)
    err, wts = [], []
    for i, (server, R, g, T, w) in enumerate(rows):
        if server not in maps:
            continue
        pos = float(interp_extrap_vec(np.array([R]), a, ys)[0]) if band_by is None \
            else float(band_by[i])
        if not (band[0] <= pos < band[1]):
            continue
        x = float(interp_extrap_vec(np.array([R + alpha * T]), a, ys)[0])
        err.append(internal_to_server(server, x, maps) - (g + 0.5))
        wts.append(w)
    require(err, f"no declarations land in the {band} band")
    err, wts = np.asarray(err, float), np.asarray(wts, float)
    require(np.all(np.isfinite(err)), "a coupling failed to translate a declaration")
    return (float(np.average(err, weights=wts)),
            float(np.average(np.abs(err), weights=wts)),
            float(np.sum(wts)), len(err))


def _placements(arm, rows):
    """Each row's undocked rank under one arm — what `_band_error` buckets on when
    two arms have to be scored over the same rows."""
    a, _, ys, _, _ = arm
    return interp_extrap_vec(np.array([R for _, R, _, _, _ in rows], float), a, ys)


def _likelihood_bar(server_fits, eps):
    """{server: one-in-N} — the ordinal likelihood a declared category has to fall
    below for the junk posterior to pass one half, under that server's fitted
    coupling. The posterior is junk/((1−eps)·P + junk) with junk = eps/(K+1), so
    it clears one half exactly at P = eps/((K+1)(1−eps))."""
    out = {}
    for server, thr in server_fits.items():
        if not thr or len(thr) < 2:
            continue
        n_cat = len(thr) + 1              # K thresholds cut the line into K+1 categories
        out[server] = float(n_cat * (1.0 - eps) / eps)
    require(out, "no server coupling carries a category count")
    return out


def measure():
    decls, keys, users = _anchor_rows(RATINGS_DIR, SQL_PATH_DEFAULT, DB_SQLITE)
    print(f"anchor rows: {len(decls)} from {len(set(keys))} declarations "
          f"by {len(users)} users")

    first, shipped, ship_kept, dropped = _screen_and_refit(
        decls, keys, "every server a free coupling (production)")
    a_unscreened, alpha_unscreened, ys = first[0], first[1], first[2]
    a_ship, alpha_ship = shipped[0], shipped[1]
    require(dropped, "the screen removed nothing — there is no arm to compare")

    # The reproduction is only evidence if it IS the shipped mapping.
    xs_pub, ys_pub = load_rating_rank_curve(RATINGS_DIR)
    require(xs_pub.shape == a_ship.shape and ys_pub.shape == ys.shape,
            "the published curve has a different number of knots than this fit")
    require(float(np.max(np.abs(ys_pub - ys))) == 0.0,
            "the published curve sits on different rank knots than this fit")
    curve_err = float(np.max(np.abs(xs_pub - a_ship)))
    require(curve_err < CURVE_MATCH_TOL,
            f"refitting the mapping from the artifacts on disk misses the published "
            f"curve by {curve_err:.3g} rating units — ratings/ and db/ are not the "
            f"snapshot rating_to_rank.json was fitted on; re-run "
            f"pipeline.map_rating_to_rank")
    published_cut = load_declaration_exclusions(RATINGS_DIR)
    reproduced = {(uid[:8], server, band) for uid, server, band in
                  (d["key"] for d in dropped)}
    require(reproduced == published_cut,
            f"the screen reproduced here removes {len(reproduced)} declarations and "
            f"the published exclusion list holds {len(published_cut)}; they differ on "
            f"{sorted(reproduced ^ published_cut)}")

    # --- what the screen moves ---
    curve_shift = float(np.max(np.abs(a_ship - a_unscreened)))
    grid, gap = _rank_gap(a_unscreened, a_ship, ys)
    rank_shift = float(np.max(np.abs(gap)))
    rank_shift_at = float(interp_extrap_vec(np.array([grid[int(np.argmax(np.abs(gap)))]]),
                                            a_ship, ys)[0])
    user_shift = np.array([
        float(interp_extrap_vec(np.array([R + alpha_ship * T]), a_ship, ys)[0]
              - interp_extrap_vec(np.array([R + alpha_unscreened * T]), a_unscreened, ys)[0])
        for R, T in users.values()])
    bars = _likelihood_bar(first[4], mapping.EPS_CONTAMINATION)
    misses = [abs(r["internal"] - (r["declared_band"] + 0.5)) for r in dropped]

    # --- the federation-anchored arm ---
    require(any(d[0] == FEDERATION for d in decls),
            f"no {FEDERATION} declarations — nothing to anchor the arm on")
    _, fed, fed_kept, fed_dropped = _screen_and_refit(
        decls, keys, f"{FEDERATION} the identity hub (comparison arm)",
        identity_servers=(FEDERATION,))
    a_fed, alpha_fed, ys_fed, _, sf_fed = fed
    require(np.array_equal(ys_fed, ys), "the arm was fitted on a different rank grid")
    require(FEDERATION in sf_fed, f"the arm carries no {FEDERATION} coupling")
    pin_err = float(np.max([abs(v - g) for g, v in sf_fed[FEDERATION].items()]))
    require(pin_err == 0.0,
            f"the arm's {FEDERATION} coupling is {pin_err:.3g} off identity — it was "
            f"fitted, not pinned, so its latent is not that federation's ladder")

    # The claim is about the mapping's residual against DECLARATIONS in the dan
    # band, each arm scored on the rows it was fitted on (a row an arm screened out
    # would measure that arm's exclusion, not its mapping) and bucketed by that
    # arm's own placement, which is what a band label names.
    ship_bias, ship_mae, ship_w, ship_n = _band_error(shipped, ship_kept, DAN_BAND)
    fed_bias, fed_mae, fed_w, fed_n = _band_error(fed, fed_kept, DAN_BAND)
    # …and the same rows under both, so the difference cannot be a change of cohort.
    common = [d for d, k in zip(decls, keys)
              if k not in {x["key"] for x in dropped}
              and k not in {x["key"] for x in fed_dropped}]
    at_ship = _placements(shipped, common)
    paired_ship = _band_error(shipped, common, DAN_BAND, band_by=at_ship)
    paired_fed = _band_error(fed, common, DAN_BAND, band_by=at_ship)

    # What the two arms would SHOW the surveyed users the shipped mapping calls dan.
    rank_ship, rank_fed = [], []
    for R, T in users.values():
        rank_ship.append(float(interp_extrap_vec(np.array([R + alpha_ship * T]), a_ship, ys)[0]))
        rank_fed.append(float(interp_extrap_vec(np.array([R + alpha_fed * T]), a_fed, ys)[0]))
    rank_ship, rank_fed = np.asarray(rank_ship), np.asarray(rank_fed)
    dan = rank_ship >= DAN_BAND[0]
    require(int(dan.sum()) >= 5,
            f"only {int(dan.sum())} surveyed users sit in the dan bands — too few to "
            f"read a dan-range displacement")

    return {
        "n_rows": len(decls), "n_declarations": len(set(keys)), "n_users": len(users),
        "curve_err": curve_err,
        "n_screened": len(dropped),
        "curve_shift": curve_shift,
        "rank_shift": rank_shift, "rank_shift_at": rank_shift_at,
        "user_shift_max": float(np.max(np.abs(user_shift))),
        "user_shift_mean": float(np.mean(np.abs(user_shift))),
        "alpha_shift": float(alpha_ship - alpha_unscreened),
        "bars": bars,
        "bar_loosest": float(min(bars.values())),
        "bar_loosest_at": min(bars, key=lambda s: bars[s]),
        "bar_tightest": float(max(bars.values())),
        "mildest_miss": float(min(misses)), "widest_miss": float(max(misses)),
        "fed_screened": len(fed_dropped),
        "fed_alpha": float(alpha_fed), "ship_alpha": float(alpha_ship),
        "ship_dan_bias": ship_bias, "ship_dan_mae": ship_mae,
        "ship_dan_weight": ship_w, "ship_dan_rows": ship_n,
        "fed_dan_bias": fed_bias, "fed_dan_mae": fed_mae,
        "fed_dan_weight": fed_w, "fed_dan_rows": fed_n,
        "paired_ship_bias": paired_ship[0], "paired_ship_mae": paired_ship[1],
        "paired_fed_bias": paired_fed[0], "paired_fed_mae": paired_fed[1],
        "paired_weight": paired_ship[2], "paired_rows": paired_ship[3],
        "dan_users": int(dan.sum()), "kyu_users": int((~dan).sum()),
        "dan_shift": float(np.mean((rank_fed - rank_ship)[dan])),
        "dan_shift_median": float(np.median((rank_fed - rank_ship)[dan])),
        "kyu_shift": float(np.mean((rank_fed - rank_ship)[~dan])),
    }


def figures():
    m = measure()
    return [
        Figure("rankmap.screen_curve_shift", m["curve_shift"], ".3f",
               "largest displacement between the screened and unscreened rating→rank "
               "curves at equal rank, over the ladder they span, in rating units"),
        Figure("rankmap.screen_rank_shift", m["rank_shift"], ".2f",
               "the same displacement at equal RATING, in ranks — the largest gap "
               "between the two curves anywhere on the ladder"),
        Figure("rankmap.screen_likelihood_bar", m["bar_loosest"], "~1000",
               "loosest one-in-N ordinal likelihood a declared category must fall "
               "below for the mixture's junk posterior to pass one half, over the "
               "servers' fitted couplings"),
        Figure("rankmap.screen_mildest_miss", m["mildest_miss"], ".1f",
               "smallest miss among the declarations the screen removes: how far the "
               "mapping places that user from the band they declared, in ranks"),
        Figure("rankmap.federation_dan_shift", m["dan_shift"], "+.1f",
               "how far a latent anchored on the federation alone moves the shown "
               "rank of the surveyed users the shipped mapping places in the dan "
               "bands, in ranks (positive = shown stronger)"),
        Figure("rankmap.federation_kyu_shift", m["kyu_shift"], "+.1f",
               "…and of those it places in the kyu bands"),
        Figure("rankmap.federation_dan_bias", m["paired_fed_bias"], "+.2f",
               "mean signed error against dan-band declarations, in the declaring "
               "server's own units, with the federation the identity hub the latent "
               "is defined by (negative = placed weaker than they declare)"),
        Figure("rankmap.consensus_dan_bias", m["paired_ship_bias"], "+.2f",
               "…the same error over the same rows under the shipped mapping, where "
               "the federation is one coupling among many"),
    ]


def main():
    m = measure()
    print("\n" + "=" * 78)
    print(f"anchor rows {m['n_rows']} · declarations {m['n_declarations']} · "
          f"surveyed users with a rating {m['n_users']} "
          f"(shipped curve reproduced to {m['curve_err']:.2g} rating units)")

    print("\nTHE SCREEN — the first fit (unscreened) against the shipped refit")
    print(f"  removed                 {m['n_screened']} declarations; mildest miss "
          f"{m['mildest_miss']:.2f} ranks, widest {m['widest_miss']:.2f}")
    print(f"  likelihood bar at 1/2   one in {m['bar_loosest']:,.0f} "
          f"({m['bar_loosest_at']}, the loosest) to one in {m['bar_tightest']:,.0f}")
    print(f"  curve moves             {m['curve_shift']:.4f} rating units at worst, at "
          f"equal rank over the ladder")
    print(f"  read in ranks           {m['rank_shift']:.3f} ranks at worst, at equal "
          f"rating (near {value_to_rank(m['rank_shift_at'])})")
    print(f"  per surveyed user       {m['user_shift_mean']:.3f} ranks mean absolute, "
          f"{m['user_shift_max']:.3f} worst (each arm at its own time dock; "
          f"α moves {m['alpha_shift']:+.4f})")

    print(f"\nTHE FEDERATION ANCHOR — the same fit with {FEDERATION} held at identity, "
          f"so the latent IS that ladder")
    print(f"  its own screen removed  {m['fed_screened']} declarations (the shipped fit "
          f"removes {m['n_screened']})")
    print(f"  time dock               α {m['fed_alpha']:+.4f} against the shipped "
          f"{m['ship_alpha']:+.4f} rating per de-meaned log10-min")
    print(f"  what each would SHOW    {m['dan_shift']:+.2f} ranks over the "
          f"{m['dan_users']} surveyed users the shipped mapping calls dan (median "
          f"{m['dan_shift_median']:+.2f}), {m['kyu_shift']:+.2f} over the "
          f"{m['kyu_users']} it calls kyu — positive = shown stronger")
    print("\n  error against declarations in the 1d–4d band, in each declaring "
          "server's own units")
    print("  (negative = the mapping places a declarer weaker than they declare)")
    print(f"    {'arm':<26}{'bias':>8}{'MAE':>8}{'rows':>7}{'weight':>9}")
    print(f"    {'federation as the hub':<26}{m['paired_fed_bias']:>+8.2f}"
          f"{m['paired_fed_mae']:>8.2f}{m['paired_rows']:>7d}{m['paired_weight']:>9.1f}")
    print(f"    {'one coupling among many':<26}{m['paired_ship_bias']:>+8.2f}"
          f"{m['paired_ship_mae']:>8.2f}{m['paired_rows']:>7d}{m['paired_weight']:>9.1f}")
    print(f"    (the rows both arms kept, bucketed by the shipped placement. Each arm "
          f"on its OWN rows at its own bucketing: {m['fed_dan_bias']:+.2f} over "
          f"{m['fed_dan_rows']} against {m['ship_dan_bias']:+.2f} over "
          f"{m['ship_dan_rows']} — the arm calls more of the population dan, which is "
          f"the shift above, not a different fit)")


if __name__ == "__main__":
    main()
