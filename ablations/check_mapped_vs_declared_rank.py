"""How closely does our mapped rank match each user's self-declared rank?

This is the end-to-end calibration check on the SHIPPED v2 mapping (NOT a refit):
it takes the curves `map_rating_to_rank` already wrote and asks, per survey
user, "does the rank we'd show them line up with what they say they are?".

Pipeline mirrored from the PWA / map_rating_to_rank (so the numbers are what a
user actually sees):

  1. R = the user's 80%-clear v2-rating, pooled across boards
     (`utils.rating.fit_rank_curves.fit_user_rating` @ CALIBRATION_P, with each
     problem's own a_p slope) — the same estimator the client uses.
  2. our_internal = applyM(R): interpolate R through ratings/rating_to_rank.json,
     which is already on the one internal rank scale (0 = the 1k/1d boundary,
     integers are band edges; pwa/shared/rank-math.js).
  3. For each declared (server, rank), invert that server's coupling in
     ratings/rank_map.json (interp the declared SERVER scalar back through the
     {server→our} knots) to get declared_internal — the internal rank the user is
     claiming. Under the consensus mapping EGF is one coupling among the rest, so
     it is inverted like any other server rather than passed through.

  signed error  e = our_internal − declared_internal   (internal-rank units,
  same on every server so they pool). e > 0 ⇒ we rate the user STRONGER than
  they rate themselves; e < 0 ⇒ weaker.

Everything is in internal-rank units so servers are directly comparable. We
report the error distribution (a) bucketed by the user's own (our) rank band and
(b) per server. A per-server forward-mapped column (predicted server rank −
declared, in that server's own units) is also printed since that's the literal
on-screen discrepancy.

VERIFIED categories. `rank_surveys` also carries the weekly MACHINE heartbeats
behind the leaderboard's "Verified" column, and they are reported here as two more
categories alongside the self-reports:

  EGD   the player's EGD GoR, converted through the standard EGF grade
        linearization (100 GoR/rank, 1k = 2000) and then treated exactly like an
        EGF declaration — same ladder, just scaled, and continuous instead of
        snapped to a band centre.
  AGAR  the player's AGA rating on the AGA coupling (no such rows yet).

They are ordinary categories in every table below — same error, same units. The
one difference is which disqualifier applies: a self-report can be a lie (the
contamination posterior), a verified rating can be stale (the same EGF activity
criterion the leaderboard gates on), and both drop out of the "excl junk" row.

Where a user has BOTH an EGF self-report and an EGD GoR, the two are also compared
head-to-head — that measures self-report bias with our estimate out of the loop
entirely.

Inputs (run `pipeline.fetch_database` first to refresh the dump):
  db/d1.sql                         rank_surveys + tsumego_reports
  ratings/<size>/tsumego_ratings.csv   b_p (shipped rating)
  ratings/<size>/tsumego_slopes.csv    a_p (shipped slope)
  ratings/rating_to_rank.json                M  (shared rating→rank curve)
  ratings/rank_map.json                      per-server internal↔server knots

Server ranks are always time-adjusted (the onlymove header rank stays raw), the
same dock the PWA shows: a solver slower than typical for their rank reads weaker
on a timed server. Users without enough timed novel attempts keep the raw rank.

Usage:
  ./.venv/bin/python3 -m ablations.check_mapped_vs_declared_rank --list
  ./.venv/bin/python3 -m ablations.check_mapped_vs_declared_rank --csv out.csv
  ./.venv/bin/python3 -m ablations.check_mapped_vs_declared_rank --min-attempts 30
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import t as student_t, ttest_1samp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, VERIFIED_SERVERS, is_verified_active,
    latest_per_user_server, load_attempts_for_users, load_rank_surveys, rank_band,
    value_to_rank, verified_rank_value, verified_tier_rank,
    TIME_ADJUST_AFK_CAP_MS, TIME_ADJUST_MIN_TIMED, WORKING_SET,
    load_user_median_solve_minutes,
)
from pipeline.map_rating_to_rank import (  # noqa: E402
    EPS_CONTAMINATION, SIZES, load_floors, load_ratings, load_slopes,
)
from utils.rating.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec,
)
from utils.rating.ratings_io import (  # noqa: E402
    load_declaration_exclusions,
    apply_M, declared_to_internal, load_M, load_lapse, load_rank_map,
    load_time_adjust,
)

V2_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "ratings")


# --- Time adjustment (always on, mirrors the client) -----------------------
# The PWA docks the *server-translated* rank (header onlymove rank stays raw):
# practice here is untimed but server ranks are timed, so a solver slower than
# typical FOR THEIR RANK overstates their timed strength. See
# pwa/js/population-constants.js (timeAdjustDeltaRating) + rank-decomposition.js
# (rankAfterRatingDelta). Feature must match user_median_log10_minutes in
# fit_population_constants: non-repeat, time_used in (0, AFK cap], ≥MIN_TIMED.
# The client takes this median over its WORKING SET — the unified most-recent
# ANSWER_HISTORY_WORKING_SET attempts across both sizes (getMergedAnswerHistory),
# repeats included — NOT whole history, so we mirror that window here.


def internal_to_server(server, internal_scalar, server_curves):
    """Forward map: internal rank scalar → predicted SERVER rank scalar (what the
    PWA would display for this user on that server)."""
    if server not in server_curves:
        return float(internal_scalar)
    our, srv = server_curves[server]
    return float(interp_extrap_vec(np.array([internal_scalar]), our, srv)[0])


def junk_posterior(server, declared_scalar, x, server_curves):
    """P(this declaration is junk | data) under the fit's contamination mixture.

    The rank_map knots ARE the fitted ordinal thresholds (`our` = τ at each
    server category boundary, `server` = the band-EDGE scalars), so the
    declared category's likelihood is reconstructed exactly:
    P_ordinal = expit(τ_up − x) − expit(τ_lo − x), and the posterior is
    (ε/N) / ((1−ε)·P_ordinal + ε/N) with N = len(τ)+1 categories and x the
    time-adjusted internal rank (what the fit anchored on). None for a server
    without a coupling."""
    if server not in server_curves:
        return None
    tau, srv = server_curves[server]
    g = rank_band(declared_scalar)       # declared scalars are band CENTRES; floor, never round
    K = len(tau)
    k = int(np.searchsorted(srv, g, side="right"))   # edges at or below the declared band
    upper = expit(tau[k] - x) if k < K else 1.0
    lower = expit(tau[k - 1] - x) if k > 0 else 0.0
    p = max(float(upper - lower), 1e-12)
    junk = EPS_CONTAMINATION / (K + 1)
    return junk / ((1.0 - EPS_CONTAMINATION) * p + junk)


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

def _stats(errs):
    e = np.asarray([x for x in errs if x is not None and np.isfinite(x)], float)
    if e.size == 0:
        return None
    return {
        "n": int(e.size),
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "std": float(e.std(ddof=0)),
        "mae": float(np.mean(np.abs(e))),
        "q1": float(np.percentile(e, 25)),
        "q3": float(np.percentile(e, 75)),
        "within1": float(np.mean(np.abs(e) <= 1.0)),
        "within2": float(np.mean(np.abs(e) <= 2.0)),
    }


def _print_stat_row(label, st, width=18):
    if st is None:
        print(f"{label:<{width}}{'-':>6}")
        return
    print(f"{label:<{width}}{st['n']:>6}"
          f"{st['mean']:>9.2f}{st['median']:>9.2f}{st['std']:>8.2f}"
          f"{st['mae']:>8.2f}{st['q1']:>+8.2f}{st['q3']:>+8.2f}"
          f"{100*st['within1']:>9.0f}{100*st['within2']:>8.0f}")


def _stat_header(first="bucket", width=18):
    print(f"{first:<{width}}{'n':>6}{'mean':>9}{'med':>9}{'std':>8}"
          f"{'mae':>8}{'q1':>8}{'q3':>8}{'≤1rk%':>9}{'≤2rk%':>8}")
    print("-" * (width + 6 + 9 + 9 + 8 + 8 + 8 + 8 + 9 + 8))


def _print_full_listing(df):
    """Per server: onlymove rank, declared rank, predicted rank, err (ranks).

    The verified categories print the raw external rating they came from (the GoR /
    the AGA rating) alongside the continuous rank it converts to, since that number
    — not a band label — is the thing being compared."""
    for server in SERVERS + list(VERIFIED_SERVERS):
        sub = df[df["server"] == server].sort_values("declared_scalar")
        if sub.empty:
            continue
        system = VERIFIED_SERVERS.get(server)
        if system is None:
            print(f"\n{server}  (n={len(sub)})")
            print(f"{'user':<10}{'rating':>8}{'time':>8}{'onlymove':>9}{'declared':>9}"
                  f"{'predicted':>10}{'err':>7}  junk")
            for _, r in sub.iterrows():
                tstr = f"{r['median_min']*60:.0f}s" if pd.notna(r['median_min']) else "-"
                jstr = ("!" if r["p_junk"] > 0.5 else "") if pd.notna(r["p_junk"]) else ""
                print(f"{r['prefix']:<10}{r['our_R']:>+8.2f}{tstr:>8}{r['our_label']:>9}"
                      f"{r['declared_rank']:>9}{r['pred_server_label']:>10}"
                      f"{r['err_server']:>+7.1f}  {jstr}")
            continue
        raw = "GoR" if server == "EGD" else "rating"
        print(f"\n{server} → {system}  (n={len(sub)}, verified; ranks are {system})")
        print(f"{'user':<10}{'rating':>8}{'time':>8}{'onlymove':>9}{raw:>9}"
              f"{'as rank':>9}{'predicted':>10}{'err':>7}  last game")
        for _, r in sub.iterrows():
            tstr = f"{r['median_min']*60:.0f}s" if pd.notna(r['median_min']) else "-"
            print(f"{r['prefix']:<10}{r['our_R']:>+8.2f}{tstr:>8}{r['our_label']:>9}"
                  f"{r['declared_rank']:>9}{r['declared_scalar']:>+9.2f}"
                  f"{r['pred_server_scalar']:>+10.2f}{r['err_server']:>+7.1f}"
                  f"  {r['last_game_at'] or '-'}")


# Rank tolerances for the matched contrast below, in ranks. Several are printed
# rather than one chosen: a tight caliper controls the fit residual better and a
# loose one averages more self-reporters, and agreement across the range is the
# evidence that neither the tolerance nor the matching is driving the answer.
MATCH_CALIPERS = (0.5, 1.0, 2.0)


def _print_rank_matched_contrast(df):
    """Self-report displacement estimated WITHOUT requiring one user to hold both.

    The residual against a declaration carries three things — the bias the fit
    absorbed from the declarations it was fitted on, its own residual on this kind of
    user, and the declaration's displacement:

        err_verified = b_absorbed + eps(rank)
        err_survey   = b_absorbed + eps(rank) - b_self

    so a level answers nothing, and differencing two users MATCHED ON RANK cancels
    both nuisance terms and leaves the displacement. Three specifics carry that:

      * Match on OUR estimate, never on the declared rank. Differencing against a
        quantity that appears inside the difference manufactures a slope — regressing
        (predicted - declared) on declared has expectation corr*sd_pred/sd_declared - 1
        whatever the truth is — and matching on it is the same error in bins.
      * The self-report arm EXCLUDES every user who has a verified rating, because
        those are the users whose profile is public and checkable and who therefore
        have the least room to misreport. Comparing a verified rating against the same
        user's self-report (the function above) is the exact-match limit of this
        estimator and reads cleanest, but it is selected on the very trait being
        measured, so it is the least representative pair available — an upper bound on
        honesty rather than an estimate of it.
      * One row per user, their declarations averaged, so a user who declares on five
        servers does not count five times.

    Sign follows the tables above: positive means the self-report claims more strength
    than a verified rating at the same fitted rank would.

    Returns the printed estimates so a caller can read them (None where the contrast
    cannot be formed)."""
    ver = df[df["kind"] == "verified"]
    if ver.empty:
        return None
    ver_users = set(ver["prefix"])
    srv = df[(df["kind"] == "survey") & (~df["prefix"].isin(ver_users))]
    if srv.empty:
        return None

    def per_user(d):
        return d.groupby("prefix").agg(err=("err_server", "mean"),
                                       rank=("our_internal", "first")).reset_index()

    v, s = per_user(ver), per_user(srv)
    print(f"\nrank-matched self-report displacement  "
          f"({len(v)} verified users vs {len(s)} self-reporters holding no verified "
          f"rating; + = self-reports claim stronger)")
    print(f"{'caliper':<14}{'pairs':>6}{'estimate':>10}{'se':>7}"
          f"{'95% CI':>18}{'p':>7}   unmatched")
    out = {"n_verified": int(len(v)), "n_selfreport": int(len(s)), "calipers": []}
    for caliper in MATCH_CALIPERS:
        diffs, unmatched = [], 0
        for _, r in v.iterrows():
            m = s[(s["rank"] - r["rank"]).abs() <= caliper]
            if m.empty:
                unmatched += 1
                continue
            diffs.append(r["err"] - m["err"].mean())
        d = np.asarray(diffs, float)
        if d.size < 3:
            print(f"±{caliper:<13.1f}{d.size:>6}   too few matched pairs")
            continue
        se = d.std(ddof=1) / np.sqrt(d.size)
        half = student_t.ppf(0.975, d.size - 1) * se
        print(f"±{caliper:<13.1f}{d.size:>6}{d.mean():>+10.2f}{se:>7.2f}"
              f"{f'[{d.mean()-half:+.2f}, {d.mean()+half:+.2f}]':>18}"
              f"{ttest_1samp(d, 0.0).pvalue:>7.2f}   {unmatched}")
        out["calipers"].append({
            "caliper": float(caliper), "pairs": int(d.size),
            "estimate": float(d.mean()), "se": float(se),
            "ci_lo": float(d.mean() - half), "ci_hi": float(d.mean() + half),
            "p": float(ttest_1samp(d, 0.0).pvalue), "unmatched": int(unmatched),
        })
    print("  matched on the fitted rank, so the fit's own residual and any bias it "
          "absorbed cancel;\n  the binding constraint is the verified count, not the "
          "self-reporters.")
    return out


def _print_selfreport_vs_verified(surveys):
    """Self-reported rank vs the same player's VERIFIED rating on that same system.

    Our estimate is out of this entirely: an EGF self-report and an EGD GoR are the
    same ladder (the GoR just resolves within the band), so the gap between them is
    pure self-report bias. Read it as the exact-match limit of the rank-matched
    contrast below, and as an upper bound rather than an estimate: these are exactly
    the users whose declared rank is publicly checkable. Not restricted to users with
    a fitted rating, since nothing here needs one.

    Returns {verified server: statistics} for a caller, including how many declarations
    name the band their verified rating actually falls in and how far outside it the
    rest sit — a declaration is a BAND, so a miss is a distance to that band's EDGE and
    not the gap between the two numbers."""
    latest = latest_per_user_server(surveys)
    out = {}
    for server, system in VERIFIED_SERVERS.items():
        v = latest[latest["server"] == server]
        d = latest[latest["server"] == system]
        pairs, inactive = [], 0
        for _, r in v.iterrows():
            g = verified_rank_value(server, r["rank"])
            match = d[d["user_id"] == r["user_id"]]
            if g is None or match.empty:
                continue
            m = match.iloc[0]
            if m["rank_value"] is None or pd.isna(m["rank_value"]):
                continue
            # Same activity gate as the main tables: a rating too old to describe
            # the player can't measure how honest their current self-report is.
            if not is_verified_active(
                    verified_tier_rank(server, r["rank"], r["rank_label"]),
                    r["last_game_at"]):
                inactive += 1
                continue
            pairs.append({
                "prefix": str(r["user_id"])[:8],
                "declared": m["rank"],
                # the declared BAND's centre, so the band itself is [floor, floor+1)
                "declared_scalar": float(m["rank_value"]),
                "raw": r["rank"],
                "verified_scalar": g,
                # + = the self-report claims more strength than the rating shows
                "diff": float(m["rank_value"]) - g,
                "last_game_at": r["last_game_at"],
            })
        if not pairs:
            continue
        p = pd.DataFrame(pairs).sort_values("verified_scalar", ascending=False)
        raw = "GoR" if server == "EGD" else "rating"
        print(f"\nself-reported {system} vs verified {server}  (n={len(p)} users; "
              f"ranks, + = self-report claims stronger"
              + (f"; {inactive} dropped as inactive)" if inactive else ")"))
        print(f"{'user':<10}{'declared':>9}{raw:>9}{'as rank':>9}{'diff':>7}"
              f"  last game")
        for _, r in p.iterrows():
            print(f"{r['prefix']:<10}{r['declared']:>9}{r['raw']:>9}"
                  f"{r['verified_scalar']:>+9.2f}{r['diff']:>+7.2f}"
                  f"  {r['last_game_at'] or '-'}")
        _stat_header(first="")
        _print_stat_row("", _stats(p["diff"]))

        band = np.floor(p["declared_scalar"].to_numpy(float))
        vs = p["verified_scalar"].to_numpy(float)
        diff = p["diff"].to_numpy(float)
        in_band = np.floor(vs) == band
        miss = np.where(vs < band, band - vs,
                        np.where(vs >= band + 1.0, vs - (band + 1.0), 0.0))
        outside = np.flatnonzero(~in_band)
        widest = outside[np.argsort(-miss[outside])][:2]
        total = float(diff.sum())
        out[server] = {
            "system": system, "n": int(len(p)),
            "mean": float(diff.mean()), "sd": float(diff.std(ddof=0)),
            "within1": int(np.sum(np.abs(diff) <= 1.0)),
            "in_band": int(in_band.sum()),
            "misses": sorted(float(x) for x in miss[outside]),
            # what the two widest-missing rows carry of the mean between them
            "widest_share": float(diff[widest].sum() / total) if total else None,
        }
    return out


def _median_fitted_rank(sub):
    """Median fitted rank over the USERS in a slice, one row per user."""
    r = sub.groupby("prefix")["our_internal"].first()
    return float(r.median()) if len(r) else None


def _server_residual_means(df):
    """Per-server mean residual with its standard error and t.

    A coupling's thresholds are free in location, so a bias shared by everyone
    declaring on one server is absorbed into that server's own thresholds and leaves
    no residual behind — which is exactly why these means measure the fit's residual
    on a cohort and never anyone's honesty. Reported so that can be seen rather than
    assumed."""
    out = []
    for server in SERVERS:
        e = df[df["server"] == server]["err_server"].to_numpy(float)
        if e.size < 2:
            continue
        se = float(e.std(ddof=1) / np.sqrt(e.size))
        out.append({"server": server, "n": int(e.size), "mean": float(e.mean()),
                    "se": se, "t": float(e.mean() / se) if se > 0 else 0.0})
    return out


def _residual_spreads(df):
    """SPREAD of the residual on self-reports against the spread on verified ratings.

    A separate question from the level the contrasts above estimate: symmetric noise
    in a declaration attenuates the fit rather than displacing it, so a wider spread
    on self-reports is declaration noise and not dishonesty. Restricted to the
    verified rows' own fitted-rank range, since the two arms otherwise sit on
    different parts of the ladder and the residual is not homoscedastic along it."""
    ver = df[df["kind"] == "verified"]
    if len(ver) < 2:
        return None
    lo, hi = float(ver["our_internal"].min()), float(ver["our_internal"].max())
    in_range = df[(df["our_internal"] >= lo) & (df["our_internal"] <= hi)]
    pooled = in_range[in_range["kind"] == "survey"]
    egf = pooled[pooled["system"] == "EGF"]

    def sd(rows):
        e = rows["err_server"].to_numpy(float)
        return (float(e.std(ddof=0)), int(e.size)) if e.size >= 2 else (None, int(e.size))

    out = {"rank_lo": lo, "rank_hi": hi}
    for name, rows in (("verified", ver), ("survey_egf", egf), ("survey_pooled", pooled)):
        out[name], out["n_" + name] = sd(rows)
    return out


# Internal-rank bands (scalar: 1k=0, 1d=1). Half-open [lo, hi).
RANK_BANDS = [
    (-1e9, -14, "≤15k"),
    (-14, -9, "14k–10k"),
    (-9, -4, "9k–5k"),
    (-4, 0, "4k–1k"),
    (0, 4, "1d–4d"),
    (4, 1e9, "≥5d"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--dir", default=V2_DIR_DEFAULT)
    ap.add_argument("--min-attempts", type=int, default=20,
                    help="Drop survey users with fewer pooled attempts.")
    ap.add_argument("--csv", default=None,
                    help="If set, write the per-declaration error table here.")
    ap.add_argument("--list", action="store_true",
                    help="Print the full per-user predicted-vs-declared list, "
                         "grouped by ranking system (in each server's own rank "
                         "units), in addition to the aggregate stats.")
    ap.add_argument("--time-recenter", action="store_true",
                    help="Demean the time-dock over the survey cohort (the fleet-fit "
                         "g0,g1 line has a non-zero mean residual on survey users, "
                         "leaking a uniform bias). Keeps the within-cohort precision "
                         "gain, drops the mean shift.")
    ap.add_argument("--sqlite",
                    default=DB_SQLITE,
                    help="SQLite DB for per-user median solve time (--time-adjust).")
    args = ap.parse_args(argv)

    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — run pipeline.fetch_database.")
    surveys = latest_per_user_server(surveys)

    m_ratings, m_internal = load_M(args.dir)
    server_curves = load_rank_map(args.dir)

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No v2 per-problem ratings — run train_mirt first.")
    lapse = load_lapse(args.dir)   # global 4PL ceiling λ — match the client / anchor

    # first_time_only: match the production anchor (and the client's first-time-only
    # rank estimator) — repeats' spaced-repetition success would inflate the 80%-clear
    # rating this diagnostic compares against declared ranks.
    attempts = load_attempts_for_users(args.sql, set(surveys["user_id"].unique()),
                                       first_time_only=True)

    # user_internal: the raw onlymove rank (the header value); never time-adjusted.
    # user_internal_adj: the rank that drives the SERVER translations — the onlymove
    #   rank plus the time dock. Mirrors the PWA (rank-translation.js): the onlymove
    #   header stays raw, every server map (EGF included) translates from the docked
    #   index, so a slow solver reads weaker on a (timed) server than on onlymove.
    user_internal = {}      # prefix8 -> raw onlymove rank scalar
    user_internal_adj = {}  # prefix8 -> docked rank feeding the server maps
    user_R = {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < args.min_attempts:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=args.min_attempts,
                               slopes=merged_slopes, floors=merged_floors, lapse=lapse)
        if R is not None and n >= args.min_attempts:
            user_R[pref] = R
            user_internal[pref] = apply_M(R, m_ratings, m_internal)
            user_internal_adj[pref] = user_internal[pref]  # default: no dock

    # Time adjustment (always on): dock each user's SERVER rank by
    # alpha·(log10 median solve min − expected-for-their-rank). Users without
    # ≥MIN_TIMED timed novel attempts keep the raw rank (matching the client).
    ta = load_time_adjust(args.dir)
    if ta is None:
        sys.exit("No time_adjust block in population_constants.json.")
    alpha, center = ta
    med_min = load_user_median_solve_minutes(args.sqlite)
    deltas = {}
    for pref in user_R:
        mm = med_min.get(pref)
        if mm is None or mm <= 0:
            continue
        deltas[pref] = alpha * (np.log10(mm) - center)
    if args.time_recenter and deltas:
        mean_delta = float(np.mean(list(deltas.values())))
        deltas = {p: d - mean_delta for p, d in deltas.items()}
    for pref, d in deltas.items():
        user_internal_adj[pref] = apply_M(user_R[pref] + d, m_ratings, m_internal)

    # --- one row per (user, server) declaration or verified rating ---
    # A machine heartbeat rides the coupling of the system it speaks (EGD → the EGF
    # curve, AGAR → the AGA curve) rather than one of its own: it IS that ladder,
    # only scaled, so `system` — not `server` — is what indexes rank_map.
    rows = []
    inactive = 0
    screened = 0
    # Declarations map_rating_to_rank screened out before fitting. Scoring one
    # against a curve that excluded it measures the exclusion, not the mapping.
    excluded = load_declaration_exclusions(args.dir)
    for _, r in surveys.iterrows():
        pref = str(r["user_id"])[:8]
        if pref not in user_internal:
            continue
        rv = r.get("rank_value")
        if rv is not None and pd.notna(rv) and \
                (pref, r["server"], rank_band(float(rv))) in excluded:
            screened += 1
            continue
        verified = r["server"] in VERIFIED_SERVERS
        # A verified rating only describes the player while it's current, so an
        # inactive one is dropped outright rather than carried and excluded later:
        # a GoR whose last tournament predates the EGF window for its tier says
        # nothing about the strength we're predicting today.
        if verified and not is_verified_active(
                verified_tier_rank(r["server"], r["rank"], r["rank_label"]),
                r["last_game_at"]):
            inactive += 1
            continue
        system = VERIFIED_SERVERS.get(r["server"], r["server"])
        g = (verified_rank_value(r["server"], r["rank"]) if verified
             else r.get("rank_value"))
        if g is None or pd.isna(g):
            continue
        g = float(g)
        ours = user_internal[pref]              # raw header rank
        ours_adj = user_internal_adj[pref]      # time-adjusted; drives server maps
        declared_internal = declared_to_internal(system, g, server_curves)
        # Server prediction translates from the TIME-ADJUSTED internal, so a timed
        # server (EGF included) differs from the raw header rank by the dock.
        pred_server = internal_to_server(system, ours_adj, server_curves)
        # Contamination is a model of LYING, so it applies to self-reports only; a
        # verified rating is instead gated on being current enough to still describe
        # the player (the leaderboard's EGF activity criterion).
        p_junk = None if verified else junk_posterior(system, g, ours_adj,
                                                      server_curves)
        rows.append({
            "p_junk": p_junk,
            "prefix": pref,
            "kind": "verified" if verified else "survey",
            "server": r["server"],
            "system": system,
            "last_game_at": r["last_game_at"],
            "declared_rank": r["rank"],
            "declared_scalar": g,
            "our_R": user_R[pref],
            "median_min": med_min.get(pref),   # working-set median novel solve min, or None
            "our_internal": ours,
            "our_label": value_to_rank(ours),
            "our_internal_adj": ours_adj,
            "declared_internal": declared_internal,
            # internal-rank error (pools across servers): raw ours − theirs
            "err_internal": ours - declared_internal,
            # forward-mapped predicted rank in the server's own rank units
            "pred_server_scalar": pred_server,
            "pred_server_label": value_to_rank(pred_server),
            "err_server": pred_server - g,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No (user, server) declarations joined to a rated user.")

    n_ver = int((df["kind"] == "verified").sum())
    no_coupling = sorted(set(df["system"]) - set(server_curves))
    note = f"  ({', '.join(no_coupling)}: no rank_map coupling)" if no_coupling else ""
    print(f"\n{len(df)} ranks, {df['prefix'].nunique()} users, "
          f"{df['server'].nunique()} categories  "
          f"[{len(deltas)} time-adjusted, {len(user_R) - len(deltas)} raw]" + note)
    if screened:
        print(f"  {screened} declarations screened out by the mapping "
              f"(ratings/declaration_exclusions.csv) and not scored here")
    print(f"  of which {n_ver} verified machine ratings, the rest self-reported"
          + (f"; {inactive} more dropped as inactive (last tournament outside the "
             f"EGF window for their tier)" if inactive else ""))

    # Suspect declarations under the fit's contamination mixture: posterior
    # P(junk) > 0.5 means the fitted model finds the declaration likelier to be
    # a lie/misclick than a real rank. They stay in the CSV/listing but are
    # excluded from the "excl junk" aggregate row. (A verified rating has no such
    # test — it can't lie, and an out-of-date one never made it this far.)
    suspect = df["p_junk"].fillna(0.0) > 0.5

    # err = predicted server rank − declared server rank (ranks; + = we rate stronger)
    print("\nerr = predicted − declared  (ranks; + = we rate them stronger)")
    _stat_header()
    _print_stat_row("OVERALL", _stats(df["err_server"]))
    _print_stat_row("OVERALL excl junk", _stats(df[~suspect]["err_server"]))
    if suspect.any():
        print(f"\nsuspect declarations (contamination posterior > 0.5, "
              f"eps={EPS_CONTAMINATION:.3f}):")
        for _, r in df[suspect].sort_values("p_junk", ascending=False).iterrows():
            print(f"  {r['prefix']:<10}{r['server']:<9}declared {r['declared_rank']:>4}"
                  f"  onlymove {r['our_label']:>4}  err {r['err_server']:>+6.1f}"
                  f"  P(junk)={r['p_junk']:.2f}")

    print("\nby onlymove rank band")
    _stat_header()
    band_stats = {}
    for lo, hi, label in RANK_BANDS:
        sub = df[(df["our_internal"] >= lo) & (df["our_internal"] < hi)]
        band_stats[label] = _stats(sub["err_server"])
        _print_stat_row(label, band_stats[label])

    print("\nby server  (EGD/AGAR = the verified machine rating, on that system's coupling)")
    _stat_header(first="server")
    for server in SERVERS + list(VERIFIED_SERVERS):
        sub = df[df["server"] == server]
        label = f"{server}→{VERIFIED_SERVERS[server]}" if server in VERIFIED_SERVERS \
            else server
        if not sub.empty or server in VERIFIED_SERVERS:
            _print_stat_row(label, _stats(sub["err_server"]))

    matched = _print_rank_matched_contrast(df)
    exact = _print_selfreport_vs_verified(surveys)

    if args.list:
        _print_full_listing(df)

    if args.csv:
        df.sort_values(["server", "our_internal"]).to_csv(args.csv, index=False)
        print(f"\nPer-declaration table written to {args.csv}")

    return {
        "n_rows": int(len(df)),
        "n_screened": int(screened),
        "band": band_stats,
        "servers": _server_residual_means(df),
        "spreads": _residual_spreads(df),
        "median_rank": {k: _median_fitted_rank(df[df["kind"] == k])
                        for k in ("verified", "survey")},
        "matched": matched,
        "exact_match": exact,
    }


if __name__ == "__main__":
    main()
