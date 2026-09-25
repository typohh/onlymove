"""Per-week activity summary for a single player.

For a given user_id, prints one row per ISO week that has activity, with:
  - problem-rank mean and stddev (rank / stone units; 0 = the 1k/1d boundary)
  - median solve time (seconds)
  - number of attempts

Problem rank = raw problem rating mapped through the shipped rating->rank curve
(rating_to_rank.json), the same headline (untimed) convention the PWA uses for
the onlymove rank. Rows whose problem is not in the rated corpus are skipped for
the rank stats but still counted for time / attempt totals.

Usage:
    ./.venv/bin/python3 -m auxiliary.weekly_player_stats <user_id>
"""

import math
import csv
import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.paths import DB_SQLITE
from utils.rating.ratings_io import interp_extrap
DB_PATH = DB_SQLITE
RATING_TO_RANK = os.path.join(PROJECT_ROOT, "ratings", "rating_to_rank.json")
RATINGS_CSV = os.path.join(PROJECT_ROOT, "ratings", "{size}", "tsumego_ratings.csv")


def rank_to_label(v):
    b = math.floor(v)
    return f"{-b}k" if b < 0 else f"{b + 1}d"


def load_rank_curves():
    d = json.load(open(RATING_TO_RANK))
    curves = {}
    for size in ("9", "13"):
        curves[int(size)] = (d[size]["ratings"], d[size]["ranks"])
    return curves


def load_ratings():
    ratings = {}
    for size in (9, 13):
        path = RATINGS_CSV.format(size=size)
        with open(path) as f:
            for row in csv.DictReader(f):
                ratings[(size, row["tsumego_id"])] = float(row["rating"])
    return ratings


def iso_week(created_at):
    # created_at like '2026-07-03T15:21:39.631Z'
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -m auxiliary.weekly_player_stats <user_id>")
    user_id = sys.argv[1]

    curves = load_rank_curves()
    ratings = load_ratings()

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT created_at, size, problem, time_used FROM tsumego_reports WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No attempts found for user_id {user_id!r}.")
        return

    weeks = defaultdict(lambda: {"ranks": [], "times": [], "n": 0})
    for created_at, size, problem, time_used in rows:
        wk = weeks[iso_week(created_at)]
        wk["n"] += 1
        if time_used is not None:
            wk["times"].append(time_used / 1000.0)  # ms -> s
        rating = ratings.get((size, problem))
        if rating is not None:
            xs, ys = curves[size]
            wk["ranks"].append(interp_extrap(rating, xs, ys))

    header = f"{'week':<9} {'n':>4} {'rank_mean':>9} {'(label)':>8} {'rank_sd':>7} {'med_time_s':>10}"
    print(f"user_id: {user_id}   total attempts: {len(rows)}")
    print(header)
    print("-" * len(header))
    for wk_name in sorted(weeks):
        wk = weeks[wk_name]
        ranks, times = wk["ranks"], wk["times"]
        if ranks:
            mean = statistics.fmean(ranks)
            sd = statistics.pstdev(ranks) if len(ranks) > 1 else 0.0
            mean_s, label, sd_s = f"{mean:9.2f}", rank_to_label(mean), f"{sd:7.2f}"
        else:
            mean_s, label, sd_s = f"{'--':>9}", f"{'--':>8}", f"{'--':>7}"
        med_time = f"{statistics.median(times):10.1f}" if times else f"{'--':>10}"
        print(f"{wk_name:<9} {wk['n']:>4} {mean_s} {label:>8} {sd_s} {med_time}")


if __name__ == "__main__":
    main()
