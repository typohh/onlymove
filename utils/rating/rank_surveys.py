"""Shared rank-survey + attempts plumbing for the rank-calibration stage.

The rank→EGF calibration (pipeline.map_rating_to_rank), the population-constant fit
(pipeline.fit_population_constants), the joint rank-curve helpers (utils.rating.fit_rank_curves), and
the auxiliary survey diagnostics all read the same things: the `rank_surveys` declarations and
the per-user `tsumego_reports` attempts out of the local D1 dump, plus the rank-label↔scalar
convention they share. Those loaders + conversions live here (a dependency-free leaf — no
auxiliary/ or pipeline/ imports) so production and research both import them from one place.
"""

import datetime
import os
import re
import sqlite3
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from utils.paths import DB_SQL, PROJECT_ROOT

SQL_PATH_DEFAULT = DB_SQL

# The servers a user can declare a rank on. Under the consensus mapping
# (map_rating_to_rank) EGF is NOT special — it's a free coupling like every
# other server, not the identity/calibration hub it was in the retired model.
SERVERS = ["OGS", "KGS", "Fox", "Tygem", "WBaduk", "IGS", "GoQuest", "EGF", "AGA"]

# INSERT INTO "rank_surveys" VALUES(123,'<user-id>','OGS','5k','2025-09-12T…');
# country, rank_label and last_game_at are optional trailing columns (quoted, or
# NULL, or absent on an older dump); anything after them (future adds) is ignored.
RANK_SURVEY_RE = re.compile(
    r"INSERT INTO [\"`]?rank_surveys[\"`]?\s*(?:\([^)]*\))?\s*VALUES\("
    r"\d+,'([^']+)','([^']+)','([^']+)','([^']+)'"
    r"(?:,(?:NULL|'([^']*)'))?"      # country
    r"(?:,(?:NULL|'([^']*)'))?"      # rank_label
    r"(?:,(?:NULL|'([^']*)'))?"      # last_game_at
    r"(?:,.*)?\);"
)

# The MACHINE heartbeat rows that share this table with the human declarations
# (written weekly — see pwa/src/leaderboard.js), mapped to the ranking system each
# one speaks. They are not declarations: nobody self-reports them, and `rank` holds
# a raw external RATING rather than one of our bands, so `rank_value` stays None for
# them (rank_to_value parses kyu/dan labels only) and they never enter the ordinal
# rank-map fit. Each rating is the same ladder we already couple to, just scaled:
# 'EGD' is the EGF system at 100 GoR per rank, 'AGAR' the AGA rating scale.
VERIFIED_SERVERS = {"EGD": "EGF", "AGAR": "AGA"}

# Captures: user_id, problem, success, size (optional), is_repeat (optional).
# The trailing columns (size, move, time_used, review_time, is_repeat, ...) are
# only present on a full-schema dump; keep them optional so a narrower/legacy
# dump still parses — is_repeat then reads as None and is treated as first-time.
FULL_ATTEMPT_RE = re.compile(
    r"INSERT INTO [\"`]?tsumego_reports[\"`]?\s*(?:\([^)]*\))?\s*VALUES\("
    r"\d+,'([^']+)','([^']+)',([01]),'[^']+'"
    r"(?:,(\d+)"                                                    # size
    r"(?:,(?:NULL|'[^']*'),(?:NULL|-?\d+),(?:NULL|-?\d+),([01]))?"  # move,time,review,is_repeat
    r")?"
)


def rank_to_value(s):
    """Kyu/dan label -> the CENTRE of that band on the project rank scale (0 is the
    1k/1d boundary, every rank one unit wide, integers are band EDGES; see
    pwa/shared/rank-math.js, which this mirrors exactly). "1k" -> -0.5, "5k" -> -4.5,
    "1d" -> 0.5, "9d" -> 8.5. Code wanting the band's LOWER edge subtracts 0.5."""
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    m = re.fullmatch(r"(\d{1,2})([kd])", s)
    if not m:
        return None
    n = int(m.group(1))
    return (0.5 - n) if m.group(2) == "k" else (n - 0.5)


def rank_band(v):
    """Which whole rank a value falls in — FLOOR, since integers are band edges.
    Mirrors rankBand in pwa/shared/rank-math.js. Band b spans [b, b+1), so its
    LOWER EDGE is exactly b and its CENTRE is b + 0.5."""
    return int(np.floor(v))


def value_to_rank(v):
    """Project rank -> its kyu/dan label. FLOOR, because integers are band edges."""
    if v is None or not np.isfinite(v):
        return "?"
    b = int(np.floor(v))
    return f"{-b}k" if b < 0 else f"{b + 1}d"


def gor_to_value(gor):
    """EGD GoR -> a CONTINUOUS rank on the EGF scale. The standard EGF grade
    linearization — 100 GoR per rank, 1k = 2000 / 1d = 2100 — with the trailing
    -0.5 landing GoR 2000 on the 1k band CENTRE, where a human declaring "1k" also
    lands (rank_to_value), so a GoR and a self-report are directly comparable. The
    only difference is resolution: the GoR resolves within the band instead of
    collapsing to its centre. Mirrors gorToRank in pwa/src/leaderboard.js. None on a
    non-positive/garbage GoR."""
    try:
        g = float(gor)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(g) or g <= 0:
        return None
    return (g - 2000.0) / 100.0 - 0.5


def aga_rating_to_value(rating):
    """AGA rating -> a CONTINUOUS rank on our scale, LABEL-PRESERVING: AGA's
    convention is dan = floor(rating) / kyu = floor(|rating|) (see pwa/src/aga.js),
    so AGA's "5d" is rating [5, 6) while ours is [4, 5) — shift the dan half down by
    1 and the kyu half up by 1 and every AGA band lands exactly on ours. The jump at
    0 is AGA's own (-1, 1) gap, which holds no real rating; None there (and on a
    legacy heartbeat row that stored a label rather than a rating). The shift is
    exact on band interiors; AGA closes its kyu bands at the STRONG end where we
    close ours at the weak end, so a rating landing on an exact integer tick reads
    one band strong.

    Mirrors agaRatingToRank in pwa/src/leaderboard.js exactly — the /leaderboard page
    puts an AGA rating, an AGA declaration and the AGA-coupled estimate in one column,
    so all three have to reach this scale by the same shift."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(r):
        return None
    if r >= 1:
        return r - 1.0
    if r <= -1:
        return r + 1.0
    return None


def verified_rank_value(server, rank):
    """The continuous rank a machine heartbeat row carries, on the scale of the
    system it belongs to (VERIFIED_SERVERS[server]) — the verified counterpart of
    rank_to_value. None for a human survey row or an unparseable rating."""
    if server == "EGD":
        return gor_to_value(rank)
    if server == "AGAR":
        return aga_rating_to_value(rank)
    return None


def verified_tier_rank(server, rank, rank_label):
    """The coarse rank the activity window is chosen from: the official label when it
    parses (a pro grade counts as dan), else the rating itself. Mirrors the tier
    resolution in readLeaderboardRows. Note EGD's Grade is the player's CLUB grade,
    not a GoR readout, so the two can disagree — for a window this coarse it doesn't
    matter which wins."""
    if isinstance(rank_label, str):
        s = rank_label.strip().lower()
        if re.fullmatch(r"\d{1,2}\s*p", s):
            return 6.5                      # pro -> dan tier (7d), as the Worker does
        v = rank_to_value(s.replace(" ", ""))
        if v is not None:
            return v
    return verified_rank_value(server, rank)


def is_verified_active(tier_rank, last_game_at, today=None):
    """EGF activity criterion, mirroring isVerifiedActive in pwa/src/leaderboard.js:
    a verified rating counts while the last tournament is within 2 years for dan (and
    pro), 1 year for single-digit kyu, 6 months for double-digit kyu. Tier boundaries
    are the integers, since those are band EDGES. FAILS OPEN on a missing or
    unparseable date — a register-seeded heartbeat carries none until its first
    weekly refresh."""
    if not last_game_at:
        return True
    try:
        played = datetime.date.fromisoformat(str(last_game_at)[:10])
    except ValueError:
        return True
    days = ((today or datetime.date.today()) - played).days
    if tier_rank is None or not np.isfinite(tier_rank):
        window = 730
    elif tier_rank >= 0:
        window = 730
    elif tier_rank >= -9:
        window = 365
    else:
        window = 183
    return days <= window


def load_rank_surveys(sql_path):
    """Return DataFrame [user_id, server, rank, rank_value, created_at, country,
    rank_label, last_game_at]. country is Cloudflare's edge geo-IP recorded
    server-side (None on pre-country rows/dumps); rank_label + last_game_at are the
    machine heartbeats' official grade and last-tournament date (None on human survey
    rows). Survey duplicates per (user, server) are kept; downstream code picks the
    most recent."""
    if not os.path.exists(sql_path):
        raise FileNotFoundError(
            f"{sql_path} not found. Run `./.venv/bin/python3 "
            f"-m pipeline.fetch_database` first.")
    rows = []
    with open(sql_path, "r") as f:
        for line in f:
            m = RANK_SURVEY_RE.match(line)
            if not m:
                continue
            (user_id, server, rank, created_at,
             country, rank_label, last_game_at) = m.groups()
            rows.append({
                "user_id": user_id,
                "server": server,
                "rank": rank,
                "rank_value": rank_to_value(rank),
                "created_at": created_at,
                "country": country,
                "rank_label": rank_label,
                "last_game_at": last_game_at,
            })
    return pd.DataFrame(rows)


def load_attempts_for_users(sql_path, user_ids, first_time_only=False):
    """Parse all tsumego_reports rows belonging to a set of full user_ids.
    Returns dict (user_prefix8, size) -> list of (tsumego_id, success).

    first_time_only=True drops repeat attempts (is_repeat=1), leaving the
    first-time-only view the client's rank estimator uses — so an anchor built
    from these mirrors the client instead of being inflated by the higher
    success rate on re-solved (spaced-repetition) problems."""
    targets = set(user_ids)
    out = defaultdict(list)
    with open(sql_path, "r") as f:
        for line in f:
            m = FULL_ATTEMPT_RE.match(line)
            if not m:
                continue
            user_id, tid, success_str, size_str, is_repeat_str = m.groups()
            if user_id not in targets:
                continue
            if first_time_only and is_repeat_str == "1":
                continue
            size = int(size_str) if size_str else 9
            out[(user_id[:8], size)].append((tid, int(success_str)))
    return out


def latest_per_user_server(surveys):
    """Pick the latest declaration per (user, server) so multi-month
    re-surveys collapse to the user's current self-assessment."""
    if surveys.empty:
        return surveys
    surveys = surveys.sort_values("created_at")
    return (surveys.groupby(["user_id", "server"], as_index=False)
            .tail(1).reset_index(drop=True))


# --- Client-mirror solve-time window ---------------------------------------
# The time dock's solve-time feature, defined exactly as the CLIENT defines it:
# the median non-repeat, timed, AFK-capped solve minutes over the user's WORKING
# SET — the unified most-recent ANSWER_HISTORY_WORKING_SET attempts across both
# sizes (getMergedAnswerHistory), repeats included in the window — not whole
# history. `pipeline.map_rating_to_rank` deliberately keeps its OWN copy of these
# numbers: the mapping's server-side feature is no longer a strict mirror (see the
# note beside its AFK_CAP_MS), so the two must be free to diverge.
TIME_ADJUST_AFK_CAP_MS = 180000
TIME_ADJUST_MIN_TIMED = 20
WORKING_SET = 3000  # ANSWER_HISTORY_WORKING_SET in pwa/js/constants.js


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
