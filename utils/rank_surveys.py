"""Shared rank-survey + attempts plumbing for the rank-calibration stage.

The rank→EGF calibration (pipeline.map_rating_to_rank), the population-constant fit
(pipeline.fit_population_constants), the joint rank-curve helpers (utils.fit_rank_curves), and
the auxiliary survey diagnostics all read the same things: the `rank_surveys` declarations and
the per-user `tsumego_reports` attempts out of the local D1 dump, plus the rank-label↔scalar
convention they share. Those loaders + conversions live here (a dependency-free leaf — no
auxiliary/ or pipeline/ imports) so production and research both import them from one place.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SQL_PATH_DEFAULT = os.path.join(PROJECT_ROOT, "attempts", "database.sql")

# The servers a user can declare a rank on. Under the consensus mapping
# (map_rating_to_rank) EGF is NOT special — it's a free coupling like every
# other server, not the identity/calibration hub it was in the retired model.
SERVERS = ["OGS", "KGS", "Fox", "Tygem", "WBaduk", "IGS", "GoQuest", "EGF", "AGA"]

# INSERT INTO "rank_surveys" ("id","user_id","server","rank","created_at")
#   VALUES(123,'<redacted-user-id>…','OGS','5k','2025-09-12T…');
RANK_SURVEY_RE = re.compile(
    r"INSERT INTO [\"`]?rank_surveys[\"`]?\s*(?:\([^)]*\))?\s*VALUES\("
    r"\d+,'([^']+)','([^']+)','([^']+)','([^']+)'\);"
)

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


def rank_to_scalar(s):
    """1k=0, 2k=-1, ..., 25k=-24; 1d=1, 2d=2, ..., 9d=9. No zero gap."""
    if not isinstance(s, str):
        return None
    s = s.strip().lower()
    m = re.fullmatch(r"(\d{1,2})([kd])", s)
    if not m:
        return None
    n = int(m.group(1))
    return (1 - n) if m.group(2) == "k" else n


def scalar_to_rank(v):
    if v is None or not np.isfinite(v):
        return "?"
    # Nearest-integer kyu/dan label.
    k = int(round(v))
    if k <= 0:
        return f"{1 - k}k"
    return f"{k}d"


def load_rank_surveys(sql_path):
    """Return DataFrame [user_id, server, rank, rank_scalar, created_at].
    Survey duplicates per (user, server) are kept; downstream code picks
    the most recent."""
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
            user_id, server, rank, created_at = m.groups()
            rows.append({
                "user_id": user_id,
                "server": server,
                "rank": rank,
                "rank_scalar": rank_to_scalar(rank),
                "created_at": created_at,
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
