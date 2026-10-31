"""The handful of values both halves of utils/ need.

This module imports from neither `utils.katago` nor `utils.rating`: both sides
read it, so an import here would load one side into every process of the other.
"""
import functools
import os
import re
from datetime import datetime

from utils.paths import PROJECT_ROOT, TSUMEGOS_DIR

DEBUG = False
ALL_BOARD_SIZES = (9, 13)
KOMI = 6.5
RULES = "japanese"

# Ids of problems removed from the corpus that still carry attempt rows in the
# DB. Lives here because both sides of the id invariant need it and they import
# nothing from each other: the WRITER (pipeline.generate_tsumegos, which must
# never reissue one) and the READERS (utils.rating.features, which must never
# find one back in the corpus).
RETIRED_IDS_PATH = os.path.join(TSUMEGOS_DIR, "retired_ids.txt")


@functools.lru_cache(maxsize=1)
def read_retired_ids():
    """The retired id set, read once per process.

    A missing file yields an empty set: this backs a guard against reuse, and a
    generator run should not be blocked because the list has not been produced
    yet."""
    retired = set()
    try:
        with open(RETIRED_IDS_PATH) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    retired.add(line)
    except FileNotFoundError:
        pass
    return retired


# Accounts whose attempts are in the DB but must not shape any fit: the LLM
# players (auxiliary/llm_player/), which play through /report like a person so
# the Worker can rank them on the leaderboard. Their rows stay in D1 and in the
# local mirror — the leaderboard refresh rebuilds their rank from them — and
# every pipeline reader of the attempt log drops them instead.
IGNORED_USERS_PATH = os.path.join(PROJECT_ROOT, "db", "ignored_users.txt")
_USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8}$")   # pwa/src/guards.js USER_ID_RE


@functools.lru_cache(maxsize=1)
def read_ignored_users():
    """The ignored user ids, read once per process: one per line, `#` starts a
    comment. A malformed id raises rather than being skipped, since a typo here
    would silently let that account back into the fit."""
    ignored = set()
    try:
        with open(IGNORED_USERS_PATH) as fh:
            for n, line in enumerate(fh, 1):
                uid = line.split("#", 1)[0].strip()
                if not uid:
                    continue
                if not _USER_ID_RE.match(uid):
                    raise ValueError(f"{IGNORED_USERS_PATH}:{n}: {uid!r} is not a user id")
                ignored.add(uid)
    except FileNotFoundError:
        pass
    return frozenset(ignored)


def not_ignored_sql(column="user_id"):
    """A SQL condition keeping only rows whose `column` is not an ignored user.
    The ids are inlined: read_ignored_users has already held each to the
    user-id alphabet, which has no quote in it."""
    ids = sorted(read_ignored_users())
    if not ids:
        return "1"
    return f"{column} NOT IN ({', '.join(repr(i) for i in ids)})"


def debug(message: str):
    if DEBUG:
        print(datetime.now().strftime("%H:%M:%S"), message)
