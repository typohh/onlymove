"""How closely does our mapped rank match each user's self-declared rank?

This is the end-to-end calibration check on the SHIPPED v2 mapping (NOT a refit):
it takes the curves `map_rating_to_egf` already wrote and asks, per survey
user, "does the rank we'd show them line up with what they say they are?".

Pipeline mirrored from the PWA / map_rating_to_egf (so the numbers are what a
user actually sees):

  1. R = the user's 80%-clear v2-rating, pooled across boards
     (`utils.fit_rank_curves.fit_user_rating` @ CALIBRATION_P, with each
     problem's own a_p slope) — the same estimator the client uses.
  2. our_internal = applyM(R): interpolate R through ratings_v2/rating_to_rank.json
     (rank_index axis 20k=0..1k=19..1d=20), shifted to the internal/EGF scalar
     (1k=0, 1d=1) by −19.
  3. For each declared (server, rank), invert that server's coupling in
     ratings_v2/rank_map.json (interp the declared SERVER scalar back through the
     {server→our} knots) to get declared_internal — the internal rank the user is
     claiming. EGF is the identity hub, so declared_internal = declared scalar.

  signed error  e = our_internal − declared_internal   (internal-rank units,
  same on every server so they pool). e > 0 ⇒ we rate the user STRONGER than
  they rate themselves; e < 0 ⇒ weaker.

Everything is in internal-rank units so servers are directly comparable. We
report the error distribution (a) bucketed by the user's own (our) rank band and
(b) per server. A per-server forward-mapped column (predicted server rank −
declared, in that server's own units) is also printed since that's the literal
on-screen discrepancy.

Inputs (run `pipeline.fetch_database` first to refresh the dump):
  attempts/database.sql                         rank_surveys + tsumego_reports
  ratings_v2/<size>/tsumego_ratings_final.csv   b_p (shipped rating)
  ratings_v2/<size>/tsumego_quality_final.csv   a_p (shipped slope)
  ratings_v2/rating_to_rank.json                M  (shared rating→rank curve)
  ratings_v2/rank_map.json                      per-server internal↔server knots

Usage:
  ./.venv/bin/python3 -m auxiliary.check_mapped_vs_declared_rank
  ./.venv/bin/python3 -m auxiliary.check_mapped_vs_declared_rank --csv out.csv
  ./.venv/bin/python3 -m auxiliary.check_mapped_vs_declared_rank --min-attempts 30
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, latest_per_user_server, load_attempts_for_users,
    load_rank_surveys, scalar_to_rank,
)
from pipeline.map_rating_to_egf import (  # noqa: E402
    SIZES, load_v2_ratings, load_v2_slopes,
)
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec,
)

V2_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "ratings_v2")

# rank_index axis (rating_to_rank.json) is 20k=0 .. 1k=19 .. 1d=20; the internal
# (EGF) rank scalar is 1k=0, 1d=1. So internal_scalar = rank_index − 19.
RANK_INDEX_OFFSET = 19.0


def load_M(v2_dir):
    """Shared rating→internal-rank-scalar knots (both board keys are identical).
    Returns (ratings_asc, internal_scalars_asc) for interp_extrap_vec."""
    with open(os.path.join(v2_dir, "rating_to_rank.json")) as f:
        rr = json.load(f)
    curve = rr[str(SIZES[0])]
    ratings = np.asarray(curve["ratings"], dtype=float)
    internal = np.asarray(curve["rank_indices"], dtype=float) - RANK_INDEX_OFFSET
    order = np.argsort(ratings)
    return ratings[order], internal[order]


def load_server_curves(v2_dir):
    """{server: (our_knots, server_knots)} from rank_map.json, both ascending in
    internal-rank scalar / server-rank scalar."""
    with open(os.path.join(v2_dir, "rank_map.json")) as f:
        rm = json.load(f)
    out = {}
    for server, v in rm.items():
        our = np.asarray(v["our"], dtype=float)
        srv = np.asarray(v["server"], dtype=float)
        order = np.argsort(our)
        out[server] = (our[order], srv[order])
    return out


def apply_M(R, m_ratings, m_internal):
    """rating R → internal(EGF) rank scalar."""
    return float(interp_extrap_vec(np.array([R]), m_ratings, m_internal)[0])


# --- Time adjustment (optional, mirrors the client) ------------------------
# The PWA docks the *server-translated* rank (header onlymove rank stays raw):
# practice here is untimed but server ranks are timed, so a solver slower than
# typical FOR THEIR RANK overstates their timed strength. See
# pwa/js/population-constants.js (timeAdjustDeltaRating) + rank-decomposition.js
# (rankIndexAfterRatingDelta). Feature must match user_median_log10_minutes in
# fit_population_constants: non-repeat, time_used in (0, AFK cap], ≥MIN_TIMED.
# The client takes this median over its WORKING SET — the unified most-recent
# ANSWER_HISTORY_WORKING_SET attempts across both sizes (getMergedAnswerHistory),
# repeats included — NOT whole history, so we mirror that window here.
TIME_ADJUST_AFK_CAP_MS = 180000
TIME_ADJUST_MIN_TIMED = 20
WORKING_SET = 3000  # ANSWER_HISTORY_WORKING_SET in pwa/js/constants.js


def load_time_adjust(v2_dir):
    """(alpha, g0, g1) from the shipped population_constants.json, or None."""
    with open(os.path.join(v2_dir, "population_constants.json")) as f:
        ta = json.load(f).get("time_adjust")
    if not ta:
        return None
    return (float(ta["alpha_per_log10_min"]),
            float(ta["time_vs_rank_intercept"]),
            float(ta["time_vs_rank_slope"]))


def load_user_median_solve_minutes(sqlite_path, working_set=WORKING_SET):
    """{user_prefix8: median novel solve minutes} over each user's WORKING-SET
    window — the most-recent `working_set` attempts unified across both sizes —
    reproducing the client's medianNovelSolveMinutes() on getMergedAnswerHistory().

    The window is taken over ALL attempts (repeats included, like the client's
    working set), ordered by created_at so backfilled rows land at their true
    historical position, and ONLY THEN filtered to non-repeat + timed (0, AFK cap]
    before the median — matching the client, which slices the last `working_set`
    raw attempts first and filters inside that slice."""
    con = sqlite3.connect(sqlite_path)
    try:
        # created_at is ISO-8601 TEXT (sorts chronologically); id breaks ties.
        # Order by id, not created_at, would pull backfilled-historical rows
        # (recent id, old timestamp) into the recent window — see the fetch
        # cursor note in CLAUDE.md.
        cur = con.execute(
            "SELECT user_id, is_repeat, time_used FROM tsumego_reports "
            "ORDER BY created_at, id")
        windows = defaultdict(lambda: deque(maxlen=working_set))
        for uid, is_repeat, t in cur:
            windows[uid[:8]].append((is_repeat, t))
    finally:
        con.close()
    out = {}
    for pref, rows in windows.items():
        ts = [float(t) for (rep, t) in rows
              if not rep and t is not None and 0 < float(t) <= TIME_ADJUST_AFK_CAP_MS]
        if len(ts) >= TIME_ADJUST_MIN_TIMED:
            out[pref] = float(np.median(ts)) / 60000.0
    return out


def declared_to_internal(server, declared_scalar, server_curves):
    """Invert a server coupling: declared SERVER scalar → internal rank scalar.
    Interpolate through the {server → our} knots (server is the x-axis here).
    EGF / unknown servers are the identity hub."""
    if server not in server_curves:
        return float(declared_scalar)
    our, srv = server_curves[server]
    # srv must be ascending for np.interp; it is (sorted with `our`, monotone fit).
    return float(interp_extrap_vec(np.array([declared_scalar]), srv, our)[0])


def internal_to_server(server, internal_scalar, server_curves):
    """Forward map: internal rank scalar → predicted SERVER rank scalar (what the
    PWA would display for this user on that server)."""
    if server not in server_curves:
        return float(internal_scalar)
    our, srv = server_curves[server]
    return float(interp_extrap_vec(np.array([internal_scalar]), our, srv)[0])


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


# Internal-rank bands (scalar: 1k=0, 1d=1). Half-open [lo, hi).
RANK_BANDS = [
    (-1e9, -14, "≤15k"),
    (-14, -9, "14k–10k"),
    (-9, -4, "9k–5k"),
    (-4, 0, "4k–1k"),
    (0, 4, "1d–4d"),
    (4, 1e9, "≥5d"),
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sql", default=SQL_PATH_DEFAULT)
    ap.add_argument("--v2-dir", default=V2_DIR_DEFAULT)
    ap.add_argument("--min-attempts", type=int, default=20,
                    help="Drop survey users with fewer pooled attempts.")
    ap.add_argument("--csv", default=None,
                    help="If set, write the per-declaration error table here.")
    ap.add_argument("--time-adjust", action="store_true",
                    help="Apply the client's time adjustment to the displayed "
                         "(server) rank — docks solvers slower than typical for "
                         "their rank, the same correction the PWA shows. Median "
                         f"solve time is taken over the working set (last {WORKING_SET} "
                         "attempts), matching getMergedAnswerHistory.")
    ap.add_argument("--time-recenter", action="store_true",
                    help="With --time-adjust: demean the time-dock over the survey "
                         "cohort (the fleet-fit g0,g1 line has a non-zero mean "
                         "residual on survey users, leaking a uniform bias). Keeps "
                         "the within-cohort precision gain, drops the mean shift.")
    ap.add_argument("--sqlite",
                    default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"),
                    help="SQLite DB for per-user median solve time (--time-adjust).")
    args = ap.parse_args()

    print(f"Reading {args.sql} ...")
    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — run pipeline.fetch_database.")
    surveys = latest_per_user_server(surveys)
    print(f"  {len(surveys)} latest declarations from "
          f"{surveys['user_id'].nunique()} users")

    m_ratings, m_internal = load_M(args.v2_dir)
    server_curves = load_server_curves(args.v2_dir)
    print(f"  M: {len(m_ratings)} knots, rating "
          f"[{m_ratings[0]:.2f},{m_ratings[-1]:.2f}] → internal "
          f"[{m_internal[0]:.1f},{m_internal[-1]:.1f}]")
    print(f"  rank_map servers: {', '.join(server_curves)}")

    merged_ratings, merged_slopes = {}, {}
    for s in SIZES:
        merged_ratings.update(load_v2_ratings(args.v2_dir, s))
        merged_slopes.update(load_v2_slopes(args.v2_dir, s))
    if not merged_ratings:
        sys.exit("No v2 per-problem ratings — run train_rating first.")

    attempts = load_attempts_for_users(args.sql, set(surveys["user_id"].unique()))

    print(f"\nEstimating each user's {int(CALIBRATION_P*100)}%-clear v2-rating "
          f"(pooled across boards) ...")
    user_internal = {}  # prefix8 -> our internal rank scalar
    user_R = {}
    for uid in surveys["user_id"].unique():
        pref = uid[:8]
        pooled = []
        for s in SIZES:
            pooled.extend(attempts.get((pref, s), []))
        if len(pooled) < args.min_attempts:
            continue
        R, n = fit_user_rating(pooled, merged_ratings, min_n=args.min_attempts,
                               slopes=merged_slopes)
        if R is not None and n >= args.min_attempts:
            user_R[pref] = R
            user_internal[pref] = apply_M(R, m_ratings, m_internal)
    print(f"  {len(user_internal)} users with a usable rating "
          f"(≥{args.min_attempts} attempts)")

    if args.time_adjust:
        ta = load_time_adjust(args.v2_dir)
        if ta is None:
            sys.exit("--time-adjust: no time_adjust block in population_constants.json")
        alpha, g0, g1 = ta
        med_min = load_user_median_solve_minutes(args.sqlite)
        # delta in RATING units per user (skip users missing the feature — the
        # client shows them the raw rank). Center on the time expected for THIS
        # user's (base) rank index; the curve's native axis is
        # internal_scalar + RANK_INDEX_OFFSET.
        deltas = {}
        for pref in user_R:
            mm = med_min.get(pref)
            if mm is None or mm <= 0:
                continue
            native_idx = user_internal[pref] + RANK_INDEX_OFFSET
            deltas[pref] = alpha * (np.log10(mm) - (g0 + g1 * native_idx))
        if args.time_recenter and deltas:
            mean_delta = float(np.mean(list(deltas.values())))
            deltas = {p: d - mean_delta for p, d in deltas.items()}
        for pref, d in deltas.items():
            user_internal[pref] = apply_M(user_R[pref] + d, m_ratings, m_internal)
        print(f"  TIME ADJUSTMENT applied to {len(deltas)}/{len(user_R)} users "
              f"(alpha={alpha}, g0={g0}, g1={g1:g}"
              f"{', survey-recentered' if args.time_recenter else ''}); the rest "
              f"lack ≥{TIME_ADJUST_MIN_TIMED} timed novel attempts (raw rank shown)")

    # --- one row per (user, server) declaration ---
    rows = []
    for _, r in surveys.iterrows():
        pref = str(r["user_id"])[:8]
        g = r.get("rank_scalar")
        if pref not in user_internal or g is None or pd.isna(g):
            continue
        ours = user_internal[pref]
        declared_internal = declared_to_internal(r["server"], float(g),
                                                  server_curves)
        pred_server = internal_to_server(r["server"], ours, server_curves)
        rows.append({
            "prefix": pref,
            "server": r["server"],
            "declared_rank": r["rank"],
            "declared_scalar": float(g),
            "our_R": user_R[pref],
            "our_internal": ours,
            "our_label": scalar_to_rank(ours),
            "declared_internal": declared_internal,
            # internal-rank error (pools across servers): ours − theirs
            "err_internal": ours - declared_internal,
            # forward-mapped error in the server's own rank units
            "pred_server_scalar": pred_server,
            "err_server": pred_server - float(g),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No (user, server) declarations joined to a rated user.")

    print(f"\n{len(df)} declarations from {df['prefix'].nunique()} rated users "
          f"across {df['server'].nunique()} servers.")
    no_coupling = sorted(set(df["server"]) - set(server_curves))
    if no_coupling:
        print(f"  NOTE: {', '.join(no_coupling)} have no coupling in rank_map.json "
              f"(too few survey samples) — scored against the identity hub, so "
              f"their error reflects the raw internal scale, not a fitted offset.")
    print("\nerror e = our_internal − declared_internal  (internal-rank units; "
          ">0 ⇒ we rate them STRONGER than they self-declare)")

    overall = _stats(df["err_internal"])
    print(f"\n{'='*92}")
    print("OVERALL")
    print(f"{'='*92}")
    _stat_header()
    _print_stat_row("all declarations", overall)

    print(f"\n{'='*92}")
    print("ERROR BY OUR RANK BAND  (bucketed on our_internal estimate)")
    print(f"{'='*92}")
    _stat_header()
    for lo, hi, label in RANK_BANDS:
        sub = df[(df["our_internal"] >= lo) & (df["our_internal"] < hi)]
        _print_stat_row(label, _stats(sub["err_internal"]))

    print(f"\n{'='*92}")
    print("ERROR BY SERVER  (internal-rank units; pooled-comparable)")
    print(f"{'='*92}")
    _stat_header(first="server")
    for server in SERVERS:
        sub = df[df["server"] == server]
        if not sub.empty:
            _print_stat_row(server, _stats(sub["err_internal"]))

    print(f"\n{'='*92}")
    print("ERROR BY SERVER  (forward-mapped: predicted − declared, in that "
          "server's OWN rank units)")
    print("  = the literal on-screen discrepancy for that server's players.")
    print(f"{'='*92}")
    _stat_header(first="server")
    for server in SERVERS:
        sub = df[df["server"] == server]
        if not sub.empty:
            _print_stat_row(server, _stats(sub["err_server"]))

    if args.csv:
        df.sort_values(["server", "our_internal"]).to_csv(args.csv, index=False)
        print(f"\nPer-declaration table written to {args.csv}")


if __name__ == "__main__":
    main()
