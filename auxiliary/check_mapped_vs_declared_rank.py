"""How closely does our mapped rank match each user's self-declared rank?

This is the end-to-end calibration check on the SHIPPED v2 mapping (NOT a refit):
it takes the curves `map_rating_to_rank` already wrote and asks, per survey
user, "does the rank we'd show them line up with what they say they are?".

Pipeline mirrored from the PWA / map_rating_to_rank (so the numbers are what a
user actually sees):

  1. R = the user's 80%-clear v2-rating, pooled across boards
     (`utils.fit_rank_curves.fit_user_rating` @ CALIBRATION_P, with each
     problem's own a_p slope) — the same estimator the client uses.
  2. our_internal = applyM(R): interpolate R through ratings/rating_to_rank.json
     (rank_index axis 20k=0..1k=19..1d=20), shifted to the internal/EGF scalar
     (1k=0, 1d=1) by −19.
  3. For each declared (server, rank), invert that server's coupling in
     ratings/rank_map.json (interp the declared SERVER scalar back through the
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
  ratings/<size>/tsumego_ratings_final.csv   b_p (shipped rating)
  ratings/<size>/tsumego_quality_final.csv   a_p (shipped slope)
  ratings/rating_to_rank.json                M  (shared rating→rank curve)
  ratings/rank_map.json                      per-server internal↔server knots

Server ranks are always time-adjusted (the onlymove header rank stays raw), the
same dock the PWA shows: a solver slower than typical for their rank reads weaker
on a timed server. Users without enough timed novel attempts keep the raw rank.

Usage:
  ./.venv/bin/python3 -m auxiliary.check_mapped_vs_declared_rank --list
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
from scipy.special import expit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rank_surveys import (  # noqa: E402
    SERVERS, SQL_PATH_DEFAULT, latest_per_user_server, load_attempts_for_users,
    load_rank_surveys, scalar_to_rank,
)
from pipeline.map_rating_to_rank import (  # noqa: E402
    EPS_CONTAMINATION, SIZES, load_floors, load_ratings, load_slopes,
)
from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec,
)
from utils.ratings_io import load_lapse, load_rating_rank_curve  # noqa: E402

V2_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "ratings")

# rank_index axis (rating_to_rank.json) is 20k=0 .. 1k=19 .. 1d=20; the internal
# (EGF) rank scalar is 1k=0, 1d=1. So internal_scalar = rank_index − 19.
RANK_INDEX_OFFSET = 19.0


def load_M(v2_dir):
    """Shared rating→internal-rank-scalar knots (both board keys are identical).
    Returns (ratings_asc, internal_scalars_asc) for interp_extrap_vec."""
    ratings, internal = load_rating_rank_curve(
        v2_dir, size=SIZES[0], offset=RANK_INDEX_OFFSET)
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


# --- Time adjustment (always on, mirrors the client) -----------------------
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
    """(alpha, center) from the shipped population_constants.json, or None — the simple
    log-time dock alpha·(log10 median min − center) the consensus mapping produces."""
    with open(os.path.join(v2_dir, "population_constants.json")) as f:
        ta = json.load(f).get("time_adjust")
    if not ta:
        return None
    return (float(ta["alpha_per_log10_min"]), float(ta["center_log10_min"]))


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


def junk_posterior(server, declared_scalar, x, server_curves):
    """P(this declaration is junk | data) under the fit's contamination mixture.

    The rank_map knots ARE the fitted ordinal thresholds (`our` = τ at each
    server category boundary, `server` = the g−0.5 boundary scalars), so the
    declared category's likelihood is reconstructed exactly:
    P_ordinal = expit(τ_up − x) − expit(τ_lo − x), and the posterior is
    (ε/N) / ((1−ε)·P_ordinal + ε/N) with N = len(τ)+1 categories and x the
    time-adjusted internal rank (what the fit anchored on). None for a server
    without a coupling."""
    if server not in server_curves:
        return None
    tau, srv = server_curves[server]
    g = int(round(declared_scalar))
    K = len(tau)
    k = int(np.searchsorted(srv, g))     # boundaries below the declared category
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
    """Per server: onlymove rank, declared rank, predicted rank, err (ranks)."""
    for server in SERVERS:
        sub = df[df["server"] == server].sort_values("declared_scalar")
        if sub.empty:
            continue
        print(f"\n{server}  (n={len(sub)})")
        print(f"{'user':<10}{'rating':>8}{'time':>8}{'onlymove':>9}{'declared':>9}"
              f"{'predicted':>10}{'err':>7}  junk")
        for _, r in sub.iterrows():
            tstr = f"{r['median_min']*60:.0f}s" if pd.notna(r['median_min']) else "-"
            jstr = ("!" if r["p_junk"] > 0.5 else "") if pd.notna(r["p_junk"]) else ""
            print(f"{r['prefix']:<10}{r['our_R']:>+8.2f}{tstr:>8}{r['our_label']:>9}"
                  f"{r['declared_rank']:>9}{r['pred_server_label']:>10}"
                  f"{r['err_server']:>+7.1f}  {jstr}")


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
                    default=os.path.join(PROJECT_ROOT, "attempts", "database.sqlite"),
                    help="SQLite DB for per-user median solve time (--time-adjust).")
    args = ap.parse_args()

    surveys = load_rank_surveys(args.sql)
    if surveys is None or surveys.empty:
        sys.exit("No rank_surveys rows found — run pipeline.fetch_database.")
    surveys = latest_per_user_server(surveys)

    m_ratings, m_internal = load_M(args.dir)
    server_curves = load_server_curves(args.dir)

    merged_ratings, merged_slopes, merged_floors = {}, {}, {}
    for s in SIZES:
        merged_ratings.update(load_ratings(args.dir, s))
        merged_slopes.update(load_slopes(args.dir, s))
        merged_floors.update(load_floors(args.dir, s))
    if not merged_ratings:
        sys.exit("No v2 per-problem ratings — run train_rating first.")
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

    # --- one row per (user, server) declaration ---
    rows = []
    for _, r in surveys.iterrows():
        pref = str(r["user_id"])[:8]
        g = r.get("rank_scalar")
        if pref not in user_internal or g is None or pd.isna(g):
            continue
        ours = user_internal[pref]              # raw header rank
        ours_adj = user_internal_adj[pref]      # time-adjusted; drives server maps
        declared_internal = declared_to_internal(r["server"], float(g),
                                                  server_curves)
        # Server prediction translates from the TIME-ADJUSTED internal, so a timed
        # server (EGF included) differs from the raw header rank by the dock.
        pred_server = internal_to_server(r["server"], ours_adj, server_curves)
        p_junk = junk_posterior(r["server"], float(g), ours_adj, server_curves)
        rows.append({
            "p_junk": p_junk,
            "prefix": pref,
            "server": r["server"],
            "declared_rank": r["rank"],
            "declared_scalar": float(g),
            "our_R": user_R[pref],
            "median_min": med_min.get(pref),   # working-set median novel solve min, or None
            "our_internal": ours,
            "our_label": scalar_to_rank(ours),
            "our_internal_adj": ours_adj,
            "declared_internal": declared_internal,
            # internal-rank error (pools across servers): raw ours − theirs
            "err_internal": ours - declared_internal,
            # forward-mapped predicted rank in the server's own rank units
            "pred_server_scalar": pred_server,
            "pred_server_label": scalar_to_rank(pred_server),
            "err_server": pred_server - float(g),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No (user, server) declarations joined to a rated user.")

    no_coupling = sorted(set(df["server"]) - set(server_curves))
    note = f"  ({', '.join(no_coupling)}: no rank_map coupling)" if no_coupling else ""
    print(f"\n{len(df)} declarations, {df['prefix'].nunique()} users, "
          f"{df['server'].nunique()} servers  "
          f"[{len(deltas)} time-adjusted, {len(user_R) - len(deltas)} raw]" + note)

    # Suspect declarations under the fit's contamination mixture: posterior
    # P(junk) > 0.5 means the fitted model finds the declaration likelier to be
    # a lie/misclick than a real rank. They stay in the CSV/listing but are
    # excluded from the "excl junk" aggregate rows.
    suspect = df["p_junk"].fillna(0.0) > 0.5
    clean = df[~suspect]

    # err = predicted server rank − declared server rank (ranks; + = we rate stronger)
    print("\nerr = predicted − declared  (ranks; + = we rate them stronger)")
    _stat_header()
    _print_stat_row("OVERALL", _stats(df["err_server"]))
    _print_stat_row("OVERALL excl junk", _stats(clean["err_server"]))
    if suspect.any():
        print(f"\nsuspect declarations (contamination posterior > 0.5, "
              f"eps={EPS_CONTAMINATION:.3f}):")
        for _, r in df[suspect].sort_values("p_junk", ascending=False).iterrows():
            print(f"  {r['prefix']:<10}{r['server']:<9}declared {r['declared_rank']:>4}"
                  f"  onlymove {r['our_label']:>4}  err {r['err_server']:>+6.1f}"
                  f"  P(junk)={r['p_junk']:.2f}")

    print("\nby onlymove rank band")
    _stat_header()
    for lo, hi, label in RANK_BANDS:
        sub = df[(df["our_internal"] >= lo) & (df["our_internal"] < hi)]
        _print_stat_row(label, _stats(sub["err_server"]))

    print("\nby server")
    _stat_header(first="server")
    for server in SERVERS:
        sub = df[df["server"] == server]
        if not sub.empty:
            _print_stat_row(server, _stats(sub["err_server"]))

    if args.list:
        _print_full_listing(df)

    if args.csv:
        df.sort_values(["server", "our_internal"]).to_csv(args.csv, index=False)
        print(f"\nPer-declaration table written to {args.csv}")


if __name__ == "__main__":
    main()
