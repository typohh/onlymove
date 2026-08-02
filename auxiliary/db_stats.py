"""Print summary statistics from the local D1 dump at db/d1.sql.

Loads the wrangler-exported SQL into an in-memory SQLite, then renders:
  - Totals.
  - Per-week active-user and attempt counts for the most recent 12 weeks.
  - A 2D cohort grid bucketing users by tenure (weeks between first and
    last attempt) and average attempts per active week.

Run after pipeline.fetch_database refreshes the dump:
    ./.venv/bin/python3 auxiliary/db_stats.py
"""

import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.paths import DB_SQL
SQL_PATH = DB_SQL

# Right-exclusive ranges: low <= value < high.
TENURE_BUCKETS = [
    (0, 2,   "0-1w"),
    (2, 5,   "2-4w"),
    (5, 13,  "5-12w"),
    (13, 27, "13-26w"),
    (27, float("inf"), "27+w"),
]
RATE_BUCKETS = [
    (0, 11,   "1-10"),
    (11, 51,  "11-50"),
    (51, 201, "51-200"),
    (201, float("inf"), "201+"),
]


def load_db():
    if not os.path.exists(SQL_PATH):
        sys.exit(f"Missing {SQL_PATH}. Run `python -m pipeline.fetch_database` first.")
    con = sqlite3.connect(":memory:")
    with open(SQL_PATH, "r") as f:
        con.executescript(f.read())
    return con


def bucket(value, buckets):
    for low, high, label in buckets:
        if low <= value < high:
            return label
    return buckets[-1][2]


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", ""))


def main():
    con = load_db()
    cur = con.cursor()

    total_attempts, n_users, n_problems = cur.execute(
        "SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT problem) FROM tsumego_reports"
    ).fetchone()
    min_date, max_date = cur.execute(
        "SELECT MIN(created_at), MAX(created_at) FROM tsumego_reports"
    ).fetchone()

    print("=== Totals ===")
    print(f"Attempts:        {total_attempts:>10,}")
    print(f"Unique users:    {n_users:>10,}")
    print(f"Unique problems: {n_problems:>10,}")
    print(f"Date range:      {min_date[:10]} → {max_date[:10]}")
    print()

    weekly = cur.execute(
        """
        SELECT strftime('%Y-W%W', created_at) AS week,
               COUNT(DISTINCT user_id),
               COUNT(*)
        FROM tsumego_reports
        GROUP BY week
        ORDER BY week DESC
        LIMIT 12
        """
    ).fetchall()

    print("=== Weekly activity (most recent 12 weeks) ===")
    print(f"{'Week':<10} {'Active users':>14} {'Attempts':>10}")
    for week, users, attempts in reversed(weekly):
        print(f"{week:<10} {users:>14,} {attempts:>10,}")
    print()

    rows = cur.execute(
        """
        SELECT user_id,
               COUNT(*) AS total_attempts,
               COUNT(DISTINCT strftime('%Y-%W', created_at)) AS active_weeks,
               MIN(created_at) AS first_dt,
               MAX(created_at) AS last_dt
        FROM tsumego_reports
        GROUP BY user_id
        """
    ).fetchall()

    grid_lifetime = defaultdict(int)
    grid_today = defaultdict(int)
    # Anchor "today" to the latest timestamp in the dump rather than
    # datetime.now() — keeps results stable if you re-run on an old dump.
    anchor = parse_dt(max_date)
    for _user_id, total, active_weeks, first_dt, last_dt in rows:
        first = parse_dt(first_dt)
        last = parse_dt(last_dt)
        tenure_lifetime = (last - first).days // 7
        tenure_today = (anchor - first).days // 7
        rate = total / max(1, active_weeks)
        rb = bucket(rate, RATE_BUCKETS)
        grid_lifetime[(bucket(tenure_lifetime, TENURE_BUCKETS), rb)] += 1
        grid_today[(bucket(tenure_today, TENURE_BUCKETS), rb)] += 1

    tenure_labels = [l for _, _, l in TENURE_BUCKETS]
    rate_labels = [l for _, _, l in RATE_BUCKETS]
    col_w = max(8, max(len(l) for l in rate_labels + ["Total"]) + 2)

    def render(title, grid):
        print(title)
        print(" " * 12 + "".join(f"{l:>{col_w}}" for l in rate_labels) + f"{'Total':>{col_w}}")
        col_totals = defaultdict(int)
        for tl in tenure_labels:
            row_total = 0
            line = f"{tl:<12}"
            for rl in rate_labels:
                n = grid[(tl, rl)]
                line += f"{n:>{col_w},}"
                row_total += n
                col_totals[rl] += n
            line += f"{row_total:>{col_w},}"
            print(line)
        line = f"{'Total':<12}"
        grand_total = 0
        for rl in rate_labels:
            line += f"{col_totals[rl]:>{col_w},}"
            grand_total += col_totals[rl]
        line += f"{grand_total:>{col_w},}"
        print(line)

    render(
        "=== Cohort grid: lifetime tenure (last − first attempt) × rate ===",
        grid_lifetime,
    )
    print()
    render(
        f"=== Cohort grid: today-anchored tenure ({anchor.date()} − first attempt) × rate ===",
        grid_today,
    )


if __name__ == "__main__":
    main()
