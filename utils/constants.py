"""The handful of values both halves of utils/ need.

This module imports from neither `utils.katago` nor `utils.rating`: both sides
read it, so an import here would load one side into every process of the other.
"""
import functools
import os
from datetime import datetime

from utils.paths import DB_SQLITE, TSUMEGOS_DIR

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
# every pipeline reader of the attempt log drops them instead. The list is the
# `ignored_users` table (pwa/schema.sql), mirrored by pipeline.fetch_database
# with the rest of the database: an account code is the account's credential,
# so it is never committed.
def not_ignored_sql(column="user_id"):
    """A SQL condition keeping only rows whose `column` is not an ignored user.
    A subquery against the database the query reads, so a snapshot other than
    db/ filters by its own list, and a mirror without the table fails the query
    rather than letting those accounts back into the fit."""
    return f"{column} NOT IN (SELECT user_id FROM ignored_users)"


@functools.lru_cache(maxsize=None)
def read_ignored_users(db_path=DB_SQLITE):
    """The ignored user ids in the sqlite mirror at `db_path`, read once per
    process. Raises when the mirror has no such table — one fetched before it
    existed — rather than returning an empty set."""
    import sqlite3
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"{db_path} not found — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT user_id FROM ignored_users").fetchall()
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"{db_path} has no ignored_users table — "
                           f"re-run pipeline.fetch_database") from e
    finally:
        con.close()
    return frozenset(r[0] for r in rows)


def debug(message: str):
    if DEBUG:
        print(datetime.now().strftime("%H:%M:%S"), message)
