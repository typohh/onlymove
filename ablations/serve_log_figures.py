#!/usr/bin/env python3
"""What the production attempt log says about serving, in two halves.

MODEL-FREE. What a first-time attempt clears at, and how far back the rank
estimator's window reaches in time. Both are raw means over the local snapshot of
the production attempt table — no ratings, no fitted model, no corpus and no GPU.
That is the point of quoting them: the clear rate is the one check on the serve
target that survives whatever the rating fit is doing, and the window span is a
property of how people play rather than of anything the system computed.

RATINGS-JOINED. The same attempts, each carrying the gap between the player and
the problem they were served: how the gap opens on a fresh account and closes,
what the clear rate does across the gap, and how wide the gap in the recorded
history has become. This half reads the SHIPPED per-problem parameter table and
the SHIPPED rating→rank curve, so what it measures is the serving a player
actually met rather than a backend intermediate.

WHAT A FIRST-TIME ATTEMPT IS HERE. Rows the client flagged as first-time
(`is_repeat = 0`), reduced to the EARLIEST attempt per (user, problem). The
reduction matters: a device that has lost its local history re-serves a problem
its owner has already seen and flags it first-time again, so the flag alone
admits attempts that are repeats in fact. Earliest-per-pair is also the rule the
rating fit's own attempt table uses, so the rate quoted here and the rate the fit
is calibrated against describe the same rows.

ORDER. A user's history is ordered by event time (`created_at`, arrival `id`
breaking ties), which is the order the player experienced and the order the
client's own window walks. Ordering by arrival id instead moves nothing here
(backfilled rows enter at the id frontier carrying an earlier event time), but
event order is what makes a window's span non-negative by construction rather
than by luck.

THE PLAYER'S RATING is the anchor quantity: the problem rating at which they
clear the calibration probability, a full four-parameter maximum likelihood over
their own first-time attempts using each attempt's own slope and floor plus the
global lapse, pooled across board sizes. It is the same estimator the client runs
and the same one the rank map is anchored on, imported rather than restated. Two
response patterns have no such rating and are DROPPED rather than carried: fewer
joined attempts than the anchor's own minimum, and a pattern that is all correct
or all wrong — there the likelihood is monotone in ability, the calibration point
sits at infinity, and what a finite solve returns is its own step budget rather
than a measurement.

WHAT JOINS. An attempt carries a gap only if its problem still ships. A retired
problem keeps its attempt rows and has no published rating, so those rows sit out
of every ratings-joined figure below; the run checks that retirement is what
accounts for the loss, because a shipped table that had drifted from the log
would shrink these figures with nothing to show for it.

A GAP IN RANKS is the difference of two MAPPED ranks — the player's rating and
the problem's rating each pushed through the shipped curve, then differenced —
never a rating gap multiplied by a local gradient. The curve's steepness changes
several-fold along the ladder, so the two readings disagree wherever a gap spans
a knot, and the mapped difference is the one the client shows.

    ./.venv/bin/python3 -m ablations.serve_log_figures
"""

import bisect
import datetime
import os
import re
import sqlite3
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import numpy as np  # noqa: E402

from ablations._figures import AblationError, Figure, require  # noqa: E402
# The anchor's own attempt minimum, taken from the step that fits the rank map
# rather than mirrored: a block or a career below it has no anchor rating there
# either, and two copies of the number would let this measurement describe a
# population the mapping never fitted.
from pipeline.map_rating_to_rank import MIN_ATTEMPTS  # noqa: E402
from utils.constants import RETIRED_IDS_PATH, not_ignored_sql, read_retired_ids  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.fit_rank_curves import CALIBRATION_P, fit_user_rating  # noqa: E402
from utils.rating.ratings_io import (  # noqa: E402
    interp_extrap_vec, load_M, load_lapse, read_problem_params,
)

TIER = "about"

# The rank estimator's window length, mirrored from `WIN_CAP` in
# pwa/shared/rank-window.js (the one module the client and the Worker share for
# it). The span figures describe the reach of exactly that many attempts, so a
# drift between the two would silently re-define the measurement; `_check_win_cap`
# reads the JS and refuses to run on a disagreement.
WIN_CAP = 250
RANK_WINDOW_JS = os.path.join(PROJECT_ROOT, "pwa", "shared", "rank-window.js")
_WIN_CAP_RE = re.compile(r"export\s+const\s+WIN_CAP\s*=\s*(\d+)\s*;")

# Position bands within a user's own first-time history, 1-based and inclusive;
# `None` is the open end. The last band is what "past their first few hundred
# problems" means.
POSITION_BINS = ((1, 25), (26, 100), (101, 300), (301, None))

# The share-above threshold the span sentence quotes, in days.
SPAN_LONG_DAYS = 90.0

# The SHIPPED artifacts the ratings-joined half reads. Nothing here reads a
# backend fit: a figure about what a player met has to come off what was served.
TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "pwa", "tsumegos")
PROBLEM_PARAMS_BIN = os.path.join(TSUMEGOS_DIR, "problem_params.bin")
SHIPPED_CONSTANTS = os.path.join(TSUMEGOS_DIR, "population_constants.json")
# The backend's own copy of the lapse, read only to check the shipped one against.
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
RATING_CONSTANTS = os.path.join(RATINGS_DIR, "rating_constants.json")

# Gap-bin edges in RATING units, ascending; bins are half-open [lo, hi), with an
# open bin below the first edge and above the last. The two extremes and the
# central bin are the ones quoted, the rest exist so "rising monotonically" is a
# statement about a ladder rather than about three points.
GAP_ZERO_HALF = 0.1     # the central bin is |gap| within this of zero
GAP_TAIL = 2.0          # the open bins start this far out
GAP_BIN_EDGES = (-GAP_TAIL, -1.0, -0.5, -GAP_ZERO_HALF,
                 GAP_ZERO_HALF, 0.5, 1.0, GAP_TAIL)
GAP_BIN_BELOW = 0                        # gap < -GAP_TAIL: the player is far below
GAP_BIN_ZERO = GAP_BIN_EDGES.index(GAP_ZERO_HALF)
GAP_BIN_ABOVE = len(GAP_BIN_EDGES)       # gap >= GAP_TAIL: the player is far above

# Rank distances the recorded-history spread is quoted at.
RANK_GAP_NEAR, RANK_GAP_FAR = 2.0, 3.0

# How much of the log the retired-id list is allowed to leave unexplained. The
# join loss is retirement by construction; anything past this is the shipped
# parameter table and the attempt log describing different corpora.
UNEXPLAINED_JOIN_SHARE = 0.001

_NEEDED_COLUMNS = {"id", "user_id", "problem", "success", "created_at", "is_repeat"}
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _check_win_cap():
    require(os.path.exists(RANK_WINDOW_JS), f"no {RANK_WINDOW_JS}")
    with open(RANK_WINDOW_JS, encoding="utf-8") as f:
        m = _WIN_CAP_RE.search(f.read())
    require(m is not None, f"{RANK_WINDOW_JS}: no `export const WIN_CAP = <n>;` to mirror")
    require(int(m.group(1)) == WIN_CAP,
            f"WIN_CAP is {m.group(1)} in rank-window.js but {WIN_CAP} here — "
            "the span figures describe a window of the client's length")


def _epoch(ts):
    require(isinstance(ts, str) and _TS_RE.match(ts) is not None,
            f"created_at {ts!r} is not the log's ISO-with-milliseconds-and-Z format")
    try:
        dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError as exc:                      # unreachable past the regex
        raise AblationError(f"created_at {ts!r}: {exc}") from exc
    return dt.timestamp()


def _load_histories():
    """{user_id: [(created_at, problem, success), ...]} — each user's first-time
    attempts in event order, one row per (user, problem)."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(tsumego_reports)")}
        require(columns, "no tsumego_reports table in the local snapshot")
        missing = sorted(_NEEDED_COLUMNS - columns)
        require(not missing, f"tsumego_reports is missing column(s) {missing}")
        rows = con.execute(
            "SELECT created_at, id, user_id, problem, success FROM tsumego_reports "
            f"WHERE is_repeat = 0 AND {not_ignored_sql()}").fetchall()
    finally:
        con.close()
    require(rows, "tsumego_reports holds no first-time attempts (is_repeat = 0)")

    histories, seen = {}, set()
    # created_at is a fixed-width ISO instant, so the string sort IS the time sort;
    # the regex in `_epoch` is what keeps that true.
    for created_at, row_id, user_id, problem, success in sorted(rows):
        require(_TS_RE.match(created_at or "") is not None,
                f"created_at {created_at!r} (row {row_id}) is not the log's format")
        require(success in (0, 1), f"success {success!r} (row {row_id}) is not 0 or 1")
        pair = (user_id, problem)
        if pair in seen:
            continue                        # a later row for a problem this user has met
        seen.add(pair)
        histories.setdefault(user_id, []).append((created_at, problem, success))
    return histories


# --- the shipped side of the join -------------------------------------------------

def _problem_params():
    """The shipped per-problem triple keyed by id, plus each problem's rank through
    the shipped rating→rank curve and the curve's own knots.

    Everything is read from `pwa/tsumegos/`: these figures describe the problems a
    player was actually served, and the backend CSVs can already be a refit ahead of
    what shipped."""
    require(os.path.exists(PROBLEM_PARAMS_BIN),
            f"no {PROBLEM_PARAMS_BIN} — run pipeline.package_for_pwa")
    p = read_problem_params(PROBLEM_PARAMS_BIN)
    ids = p["id"]
    require(len(set(ids)) == len(ids), f"{PROBLEM_PARAMS_BIN}: duplicate problem ids")
    for name in ("rating", "slope", "floor"):
        require(bool(np.all(np.isfinite(p[name]))),
                f"{PROBLEM_PARAMS_BIN}: non-finite {name}")
    m_ratings, m_internal = load_M(TSUMEGOS_DIR)
    require(m_ratings.size >= 2 and bool(np.all(np.diff(m_ratings) > 0)),
            "the shipped rating→rank curve is not strictly increasing in rating")
    ranks = interp_extrap_vec(p["rating"], m_ratings, m_internal)
    return {
        "rating": dict(zip(ids, p["rating"].tolist())),
        "slope": dict(zip(ids, p["slope"].tolist())),
        "floor": dict(zip(ids, p["floor"].tolist())),
        "rank": dict(zip(ids, ranks.tolist())),
        "m_ratings": m_ratings,
        "m_internal": m_internal,
    }


def _shipped_lapse():
    """The global response ceiling λ, from the SHIPPED constants.

    Checked against the fit's own copy: the anchor rating is only consistent with
    the ratings it is fitted against while the two agree, and a packaging step that
    has not run since the last fit is exactly the state where they do not."""
    require(os.path.exists(SHIPPED_CONSTANTS),
            f"no {SHIPPED_CONSTANTS} — run pipeline.package_for_pwa")
    require(os.path.exists(RATING_CONSTANTS),
            f"no {RATING_CONSTANTS} — run pipeline.train_mirt")
    shipped, fitted = load_lapse(TSUMEGOS_DIR), load_lapse(RATINGS_DIR)
    require(0.0 <= shipped < 1.0, f"shipped lapse {shipped!r} is not a probability")
    require(shipped == fitted,
            f"the shipped lapse {shipped} is not the fitted lapse {fitted} — "
            "package_for_pwa has not run since the last fit")
    return shipped


def _anchor_rating(attempts, params, lapse):
    """The player's calibration-point rating over `attempts` [(problem id, success)],
    or None where the response pattern does not identify one (see the module
    docstring's account of the two dropped patterns)."""
    joined = [(t, s) for t, s in attempts if t in params["rating"]]
    if len(joined) < MIN_ATTEMPTS:
        return None
    clears = sum(s for _t, s in joined)
    if clears == 0 or clears == len(joined):
        return None
    R, n = fit_user_rating(attempts, params["rating"], min_n=MIN_ATTEMPTS,
                           slopes=params["slope"], floors=params["floor"], lapse=lapse)
    return None if (R is None or n < MIN_ATTEMPTS) else float(R)


def _to_rank(rating, params):
    return float(interp_extrap_vec(np.array([rating], dtype=float),
                                   params["m_ratings"], params["m_internal"])[0])


def _join_coverage(histories, params):
    """How much of the log joins the shipped parameter table, and whether retirement
    accounts for the rest."""
    require(os.path.exists(RETIRED_IDS_PATH),
            f"no {RETIRED_IDS_PATH} — the join loss cannot be accounted for without it")
    retired = read_retired_ids()
    require(retired, f"{RETIRED_IDS_PATH} holds no ids")
    ratings = params["rating"]
    total = unjoined = unexplained = 0
    for attempts in histories.values():
        for _created_at, problem, _success in attempts:
            total += 1
            if problem in ratings:
                continue
            unjoined += 1
            if problem not in retired:
                unexplained += 1
    require(unexplained <= UNEXPLAINED_JOIN_SHARE * total,
            f"{unexplained} of {total} first-time attempts are on problems that are "
            f"neither published nor retired — the shipped parameter table and the "
            f"attempt log describe different corpora")
    return {"total": total, "unjoined": unjoined, "unexplained": unexplained}


def measure():
    _check_win_cap()
    histories = _load_histories()

    # Clear rate by position in the user's own first-time history.
    bands = {b: [0, 0] for b in POSITION_BINS}      # (low, high) -> [attempts, clears]
    for attempts in histories.values():
        for position, (_created_at, _problem, success) in enumerate(attempts, start=1):
            for low, high in POSITION_BINS:
                if low <= position and (high is None or position <= high):
                    band = bands[(low, high)]
                    band[0] += 1
                    band[1] += success
                    break
    for (low, high), (n, _clears) in bands.items():
        require(n > 0, f"position band {low}..{high or '∞'} holds no attempts")

    # Span of a FULL window: the time from a user's WIN_CAP-th most recent
    # first-time attempt to their most recent one.
    spans = []
    for attempts in histories.values():
        if len(attempts) < WIN_CAP:
            continue
        window = attempts[-WIN_CAP:]
        spans.append((_epoch(window[-1][0]) - _epoch(window[0][0])) / 86400.0)
    require(spans, f"no user has {WIN_CAP} first-time attempts — no full window to measure")
    spans = np.array(sorted(spans), dtype=float)
    require(bool((spans >= 0).all()), "a window span came out negative")

    # --- the ratings-joined half --------------------------------------------------
    params = _problem_params()
    lapse = _shipped_lapse()
    coverage = _join_coverage(histories, params)
    users = sorted(histories)

    # One anchor rating per player over their WHOLE first-time history. This is the
    # in-sample rating the opening trend and the gap bins are measured against.
    career = {}
    for uid in users:
        R = _anchor_rating([(problem, success) for _t, problem, success in histories[uid]],
                           params, lapse)
        if R is not None:
            career[uid] = R
    require(career, "no player has a fittable anchor rating — nothing to join")

    # Gap by position band, and clear rate by gap bin, over the same rows.
    gap_bands = {b: [0, 0.0] for b in POSITION_BINS}
    n_bins = len(GAP_BIN_EDGES) + 1
    bin_n, bin_clears = [0] * n_bins, [0] * n_bins
    for uid in users:
        R = career.get(uid)
        if R is None:
            continue
        for position, (_t, problem, success) in enumerate(histories[uid], start=1):
            rating = params["rating"].get(problem)
            if rating is None:
                continue
            gap = R - rating
            for low, high in POSITION_BINS:
                if low <= position and (high is None or position <= high):
                    band = gap_bands[(low, high)]
                    band[0] += 1
                    band[1] += gap
                    break
            j = bisect.bisect_right(GAP_BIN_EDGES, gap)
            bin_n[j] += 1
            bin_clears[j] += success
    for (low, high), (n, _total) in gap_bands.items():
        require(n > 0, f"position band {low}..{high or '∞'} joins no rated attempt")
    for j in (GAP_BIN_BELOW, GAP_BIN_ZERO, GAP_BIN_ABOVE):
        require(bin_n[j] > 0, f"gap bin {j} holds no attempt")
    rates = [c / n if n else float("nan") for n, c in zip(bin_n, bin_clears)]
    ses = [float(np.sqrt(r * (1.0 - r) / n)) if n else float("nan")
           for r, n in zip(rates, bin_n)]
    # "Rising monotonically" as a count: adjacent populated bins whose rate falls.
    populated = [j for j in range(n_bins) if bin_n[j] > 0]
    inversions = sum(1 for a, b in zip(populated, populated[1:]) if rates[b] < rates[a])

    # The recorded-history spread, in ranks, against a per-block rating and against
    # one career rating — the same attempts both ways, so the pair is paired.
    block_gaps, career_gaps = [], []
    n_blocks = n_multi_block_users = 0
    for uid in users:
        seq = histories[uid]
        career_rank = None if uid not in career else _to_rank(career[uid], params)
        fitted_blocks = 0
        for start in range(0, len(seq), WIN_CAP):
            chunk = seq[start:start + WIN_CAP]
            R_block = _anchor_rating([(problem, success) for _t, problem, success in chunk],
                                     params, lapse)
            if R_block is None:
                continue
            require(career_rank is not None,
                    f"a {WIN_CAP}-attempt block of {uid[:8]} has an anchor rating but "
                    "the whole career it sits inside does not")
            n_blocks += 1
            fitted_blocks += 1
            block_rank = _to_rank(R_block, params)
            for _t, problem, _success in chunk:
                problem_rank = params["rank"].get(problem)
                if problem_rank is None:
                    continue
                block_gaps.append(abs(block_rank - problem_rank))
                career_gaps.append(abs(career_rank - problem_rank))
        if fitted_blocks > 1:
            n_multi_block_users += 1
    require(block_gaps, f"no attempt sits inside a fittable {WIN_CAP}-attempt block")
    block_gaps = np.asarray(block_gaps, dtype=float)
    career_gaps = np.asarray(career_gaps, dtype=float)

    def spread(gaps):
        return {
            "over_near": float((gaps > RANK_GAP_NEAR).mean()),
            "over_far": float((gaps > RANK_GAP_FAR).mean()),
            "median": float(np.median(gaps)),
            "mean": float(gaps.mean()),
        }

    return {
        "bands": bands,
        "attempts": sum(n for n, _ in bands.values()),
        "users": len(histories),
        "spans": spans,
        "lapse": lapse,
        "coverage": coverage,
        "rated_users": len(career),
        "gap_bands": gap_bands,
        "bin_n": bin_n,
        "bin_rates": rates,
        "bin_ses": ses,
        "bin_inversions": inversions,
        "blocks": n_blocks,
        "multi_block_users": n_multi_block_users,
        "scored_attempts": int(block_gaps.size),
        "block": spread(block_gaps),
        "career": spread(career_gaps),
    }


def figures():
    m = measure()
    (n1, c1), (n2, c2), (n3, c3), (n4, c4) = (m["bands"][b] for b in POSITION_BINS)
    (gn1, g1), (gn2, g2), (gn3, g3), (gn4, g4) = (m["gap_bands"][b] for b in POSITION_BINS)
    spans = m["spans"]
    position_note = ("first-time attempts, one row per (user, problem), by 1-based "
                     "position in the user's own first-time history")
    window_note = (f"span in days of a FULL rank-estimator window — the {WIN_CAP} most "
                   "recent first-time attempts — over the users who have that many")
    gap_note = ("mean of (the player's own anchor rating − the problem's shipped "
                f"rating) in rating units, positive when the problem sits BELOW the "
                f"player; the anchor is their {CALIBRATION_P:.0%}-clear rating fitted "
                f"over their WHOLE first-time history, so a player needs at least "
                f"{MIN_ATTEMPTS} rated attempts and a mixed record to enter; "
                + position_note)
    bin_note = ("first-time clear rate in one bin of (player anchor rating − problem "
                "rating), bins half-open [lo, hi) in rating units; ± is the binomial "
                "standard error of that bin")
    spread_note = (f"over first-time attempts whose problem still ships and whose "
                   f"{WIN_CAP}-attempt block has an anchor rating; the gap in ranks is "
                   f"the player's rank minus the problem's, each mapped through the "
                   f"SHIPPED rating→rank curve and then differenced")
    block_note = (f"the player's rating is fitted over their own consecutive "
                  f"{WIN_CAP}-attempt block, so within-career drift is out; " + spread_note)
    career_note = ("the player's rating is one anchor over their whole first-time "
                   "history, scored on exactly the rows the block arm scores so the two "
                   "are paired; " + spread_note)
    return [
        Figure("serve.clear_first25", c1 / n1, "%.1f",
               f"clear rate over positions 1-25; {position_note}"),
        Figure("serve.clear_25_100", c2 / n2, "%.1f",
               f"clear rate over positions 26-100; {position_note}"),
        Figure("serve.clear_100_300", c3 / n3, "%.1f",
               f"clear rate over positions 101-300; {position_note}"),
        # ONE key behind two sentences: "past their first few hundred problems" and
        # the last position band are the same rows, so they cannot disagree.
        Figure("serve.clear_after300", c4 / n4, "%.1f",
               f"clear rate past position 300; {position_note}. Also the rate quoted "
               "for users past their first few hundred problems — the same population"),
        Figure("serve.gap_first25", g1 / gn1, "+.2f",
               f"mean rating gap over positions 1-25; {gap_note}"),
        Figure("serve.gap_25_100", g2 / gn2, "+.2f",
               f"mean rating gap over positions 26-100; {gap_note}"),
        Figure("serve.gap_100_300", g3 / gn3, "+.2f",
               f"mean rating gap over positions 101-300; {gap_note}"),
        Figure("serve.gap_after300", g4 / gn4, "+.2f",
               f"mean rating gap past position 300; {gap_note}"),
        Figure("serve.gapbin_clear_below2", m["bin_rates"][GAP_BIN_BELOW], "%.1f",
               f"clear rate where the player sits more than {GAP_TAIL:.0f} rating units "
               f"BELOW the problem (gap < -{GAP_TAIL:.0f}); " + bin_note),
        Figure("serve.gapbin_se_below2", m["bin_ses"][GAP_BIN_BELOW], "%.1f",
               "± on serve.gapbin_clear_below2; " + bin_note),
        Figure("serve.gapbin_clear_zero", m["bin_rates"][GAP_BIN_ZERO], "%.1f",
               f"clear rate within {GAP_ZERO_HALF:.1f} of a rating unit of zero gap; "
               f"in-sample, the player's anchor being fitted on these same attempts; "
               + bin_note),
        Figure("serve.gapbin_se_zero", m["bin_ses"][GAP_BIN_ZERO], "%.1f",
               "± on serve.gapbin_clear_zero; " + bin_note),
        Figure("serve.gapbin_clear_above2", m["bin_rates"][GAP_BIN_ABOVE], "%.1f",
               f"clear rate where the player sits more than {GAP_TAIL:.0f} rating units "
               f"ABOVE the problem (gap >= {GAP_TAIL:.0f}); " + bin_note),
        Figure("serve.gapbin_se_above2", m["bin_ses"][GAP_BIN_ABOVE], "%.1f",
               "± on serve.gapbin_clear_above2; " + bin_note),
        Figure("serve.gapbin_inversions", m["bin_inversions"], "d",
               "adjacent gap bins across the whole ladder whose clear rate FALLS as the "
               "gap rises; 0 is the monotone rise the response model implies"),
        Figure("serve.lapse_bound", 1.0 - m["bin_rates"][GAP_BIN_ABOVE], ".3f",
               "largest global lapse the log admits: no attempt can clear more often "
               "than 1 − lapse, so the top gap bin's clear rate bounds it from above "
               "at 1 − that rate"),
        Figure("serve.fitted_ceiling", 1.0 - m["lapse"], ".3f",
               "1 − the SHIPPED global lapse: the upper asymptote of the fitted "
               "response curve, which no bin of finite gap reaches"),
        Figure("serve.lapse_headroom", (1.0 - m["bin_rates"][GAP_BIN_ABOVE]) / m["lapse"],
               ".1f",
               "the log's lapse bound ÷ the shipped lapse — how far inside the bound "
               "the fitted value sits"),
        Figure("history.rank_gap_over2_pct", m["block"]["over_near"], "%.0f",
               f"share of attempts more than {RANK_GAP_NEAR:.0f} ranks from the player's "
               f"rank; " + block_note),
        Figure("history.rank_gap_over3_pct", m["block"]["over_far"], "%.0f",
               f"share of attempts more than {RANK_GAP_FAR:.0f} ranks from the player's "
               f"rank; " + block_note),
        Figure("history.rank_gap_median", m["block"]["median"], ".1f",
               "median gap in ranks; " + block_note),
        Figure("history.rank_gap_over2_career_pct", m["career"]["over_near"], "%.0f",
               f"share of attempts more than {RANK_GAP_NEAR:.0f} ranks from the player's "
               f"rank; " + career_note),
        Figure("history.rank_gap_over3_career_pct", m["career"]["over_far"], "%.0f",
               f"share of attempts more than {RANK_GAP_FAR:.0f} ranks from the player's "
               f"rank; " + career_note),
        Figure("history.rank_gap_median_career", m["career"]["median"], ".1f",
               "median gap in ranks; " + career_note),
        Figure("history.rank_gap_career_excess",
               m["career"]["over_near"] - m["block"]["over_near"], "%.1f",
               f"how much WIDER the career arm's spread reads than the per-block arm's, "
               f"in percentage points of the more-than-{RANK_GAP_NEAR:.0f}-ranks share — "
               f"what within-career drift adds once it is no longer taken out. Positive "
               f"means the career arm is wider"),
        Figure("window.span_median_days", float(np.percentile(spans, 50)), ".0f",
               f"median {window_note}"),
        Figure("window.span_p75_days", float(np.percentile(spans, 75)), ".0f",
               f"upper-quartile {window_note}"),
        Figure("window.span_over_90d", float((spans > SPAN_LONG_DAYS).mean()), "%.0f",
               f"share of those users whose window spans more than {SPAN_LONG_DAYS:.0f} days; "
               + window_note),
        Figure("window.span_max_days", float(spans.max()), ".0f",
               "longest " + window_note + "; bounded from above by the length of the log"),
    ]


def _bin_label(j):
    if j == 0:
        return f"< {GAP_BIN_EDGES[0]:+.1f}"
    if j == len(GAP_BIN_EDGES):
        return f">= {GAP_BIN_EDGES[-1]:+.1f}"
    return f"[{GAP_BIN_EDGES[j - 1]:+.1f}, {GAP_BIN_EDGES[j]:+.1f})"


def main():
    m = measure()
    spans, bands = m["spans"], m["bands"]
    clears = sum(c for _n, c in bands.values())

    print("First-time attempts (is_repeat = 0, earliest per user+problem), by position")
    print("in the user's OWN first-time history. 'clear' = the attempt succeeded.")
    print("'gap' = the player's own anchor rating minus the problem's shipped rating,")
    print("in RATING units, positive when the problem sits below the player.\n")
    print(f"{'positions':>12}{'attempts':>12}{'clears':>12}{'clear rate':>12}"
          f"{'rated':>12}{'mean gap':>11}")
    for low, high in POSITION_BINS:
        n, c = bands[(low, high)]
        gn, g = m["gap_bands"][(low, high)]
        label = f"{low}-{high}" if high is not None else f"{low}+"
        print(f"{label:>12}{n:>12,d}{c:>12,d}{c / n:>11.1%}{gn:>12,d}{g / gn:>+11.3f}")
    print(f"{'all':>12}{m['attempts']:>12,d}{clears:>12,d}"
          f"{clears / m['attempts']:>11.1%}   over {m['users']:,d} users")
    cov = m["coverage"]
    print(f"\n  {m['rated_users']:,d} of {m['users']:,d} players have an anchor rating "
          f"(at least {MIN_ATTEMPTS} rated first-time attempts, not all correct and not "
          f"all wrong).")
    print(f"  {cov['unjoined']:,d} of {cov['total']:,d} first-time attempts are on "
          f"problems that no longer ship and carry no gap; {cov['unexplained']:,d} of "
          f"those are not on the retired list.")

    print(f"\nFirst-time clear rate by RATING gap (player anchor − problem, half-open "
          f"bins).\n'se' is the binomial standard error. The model implies this rises "
          f"monotonically.\n")
    print(f"{'gap bin':>16}{'attempts':>12}{'clears':>12}{'clear rate':>12}{'se':>9}")
    for j, (n, r, se) in enumerate(zip(m["bin_n"], m["bin_rates"], m["bin_ses"])):
        if not n:
            continue
        print(f"{_bin_label(j):>16}{n:>12,d}{int(round(r * n)):>12,d}"
              f"{r:>11.1%}{se:>9.1%}")
    print(f"{'inversions':>16}{m['bin_inversions']:>12,d}   adjacent bins where the rate "
          f"falls instead of rising")
    top = m["bin_rates"][GAP_BIN_ABOVE]
    print(f"{'lapse bound':>16}{1.0 - top:>12.4f}   1 − the top bin's rate; the shipped "
          f"lapse is {m['lapse']:.4f} ({(1.0 - top) / m['lapse']:.1f}x inside it)")

    print(f"\nRecorded-history spread: how far a first-time attempt sits from the "
          f"player's rank,\nboth sides mapped through the shipped rating→rank curve. "
          f"{m['scored_attempts']:,d} attempts in\n{m['blocks']:,d} blocks of up to "
          f"{WIN_CAP} ({m['multi_block_users']:,d} players carry more than one block, so "
          f"the two arms\ncan differ at all). Both arms score the SAME rows.\n")
    print(f"{'player rating':>18}{'>%.0f ranks' % RANK_GAP_NEAR:>12}"
          f"{'>%.0f ranks' % RANK_GAP_FAR:>12}{'median':>10}{'mean':>10}")
    for label, key in ((f"own {WIN_CAP}-block", "block"), ("whole career", "career")):
        s = m[key]
        print(f"{label:>18}{s['over_near']:>11.1%}{s['over_far']:>11.1%}"
              f"{s['median']:>10.2f}{s['mean']:>10.2f}")
    print(f"{'career − block':>18}"
          f"{m['career']['over_near'] - m['block']['over_near']:>+11.2%}"
          f"{m['career']['over_far'] - m['block']['over_far']:>+11.2%}"
          f"{m['career']['median'] - m['block']['median']:>+10.2f}"
          f"{m['career']['mean'] - m['block']['mean']:>+10.2f}")

    print(f"\nSpan in DAYS of a full rank-estimator window: the time from a user's "
          f"{WIN_CAP}th-most-recent\nfirst-time attempt to their most recent one, over "
          f"the {len(spans):,d} users who have {WIN_CAP} of them.\n")
    print(f"{'quantile':>12}{'days':>12}")
    for q in (0, 10, 25, 50, 75, 90, 100):
        print(f"{('min' if q == 0 else 'max' if q == 100 else 'p%d' % q):>12}"
              f"{np.percentile(spans, q):>12.1f}")
    print(f"{'>%d days' % SPAN_LONG_DAYS:>12}{(spans > SPAN_LONG_DAYS).mean():>11.1%}"
          f"   of those users")


if __name__ == "__main__":
    main()
