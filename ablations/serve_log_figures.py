#!/usr/bin/env python3
"""Two plain rates read straight off the attempt log: what a first-time attempt
clears at, and how far back the rank estimator's window reaches in time.

Both are raw means over the local snapshot of the production attempt table. No
ratings, no fitted model, no corpus and no GPU — which is the point of quoting
them: the clear rate is the one check on the serve target that survives whatever
the rating fit is doing, and the window span is a property of how people play
rather than of anything the system computed.

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

    ./.venv/bin/python3 -m ablations.serve_log_figures
"""

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
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "about"

# The rank estimator's window length, mirrored from `WIN_CAP` in
# pwa/shared/rank-window.js (the one module the client and the Worker share for
# it). The span figures describe the reach of exactly that many attempts, so a
# drift between the two would silently re-define the measurement; `_check_win_cap`
# reads the JS and refuses to run on a disagreement.
WIN_CAP = 500
RANK_WINDOW_JS = os.path.join(PROJECT_ROOT, "pwa", "shared", "rank-window.js")
_WIN_CAP_RE = re.compile(r"export\s+const\s+WIN_CAP\s*=\s*(\d+)\s*;")

# Position bands within a user's own first-time history, 1-based and inclusive;
# `None` is the open end. The last band is what "past their first few hundred
# problems" means.
POSITION_BINS = ((1, 25), (26, 100), (101, 300), (301, None))

# The share-above threshold the span sentence quotes, in days.
SPAN_LONG_DAYS = 90.0

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
    """{user_id: [(created_at, success), ...]} — each user's first-time attempts in
    event order, one row per (user, problem)."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(tsumego_reports)")}
        require(columns, "no tsumego_reports table in the local snapshot")
        missing = sorted(_NEEDED_COLUMNS - columns)
        require(not missing, f"tsumego_reports is missing column(s) {missing}")
        rows = con.execute(
            "SELECT created_at, id, user_id, problem, success FROM tsumego_reports "
            "WHERE is_repeat = 0").fetchall()
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
        histories.setdefault(user_id, []).append((created_at, success))
    return histories


def measure():
    _check_win_cap()
    histories = _load_histories()

    # Clear rate by position in the user's own first-time history.
    bands = {b: [0, 0] for b in POSITION_BINS}      # (low, high) -> [attempts, clears]
    for attempts in histories.values():
        for position, (_created_at, success) in enumerate(attempts, start=1):
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

    return {
        "bands": bands,
        "attempts": sum(n for n, _ in bands.values()),
        "users": len(histories),
        "spans": spans,
    }


def figures():
    m = measure()
    (n1, c1), (n2, c2), (n3, c3), (n4, c4) = (m["bands"][b] for b in POSITION_BINS)
    spans = m["spans"]
    position_note = ("first-time attempts, one row per (user, problem), by 1-based "
                     "position in the user's own first-time history")
    window_note = (f"span in days of a FULL rank-estimator window — the {WIN_CAP} most "
                   "recent first-time attempts — over the users who have that many")
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


def main():
    m = measure()
    spans, bands = m["spans"], m["bands"]
    clears = sum(c for _n, c in bands.values())

    print("First-time attempts (is_repeat = 0, earliest per user+problem), by position")
    print("in the user's OWN first-time history. 'clear' = the attempt succeeded.\n")
    print(f"{'positions':>12}{'attempts':>12}{'clears':>12}{'clear rate':>12}")
    for low, high in POSITION_BINS:
        n, c = bands[(low, high)]
        label = f"{low}-{high}" if high is not None else f"{low}+"
        print(f"{label:>12}{n:>12,d}{c:>12,d}{c / n:>11.1%}")
    print(f"{'all':>12}{m['attempts']:>12,d}{clears:>12,d}"
          f"{clears / m['attempts']:>11.1%}   over {m['users']:,d} users")

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
