"""Export real per-user attempt histories + per-problem rating/slope/floor for the
offline "displayed change" simulation (runner.mjs).

The Node harness replays each user's attempts through the REAL client recall model
(pwa/js/model.js) and rank estimator (pwa/js/rank-decomposition.js) to measure how
often the post-answer overlay shows a change whose SIGN contradicts the answer
(correct attempt → rank/recall-delay moves the "wrong" way, or vice-versa).

Output (default: auxiliary/displayed_change_sim/sim_input.json — gitignored):
  {
    "problems": { tid: [rating, slope, floor] },   # one entry per problem touched
    "users":    { userId: [ [tid, success01, size, time_used_ms|null,
                             review_time_ms|null, is_repeat01, created_at_ms], ... ] }
  }
Attempts ascending by created_at. Attempts to problems with no current rating are
dropped (the client likewise can't rank/score an unrated problem) — count reported.

Usage:
  ./.venv/bin/python3 -m auxiliary.displayed_change_sim.export_sim_data \
      --top 40 --min-total 300 --min-repeats 100
  ./.venv/bin/python3 -m auxiliary.displayed_change_sim.export_sim_data \
      --users <id>,<id>,...
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.ratings_io import load_size_value_maps  # noqa: E402

SIZES = [9, 13]
DB_DEFAULT = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_input.json")


def parse_ms(ts):
    """ISO-8601 (trailing Z) → epoch milliseconds (int)."""
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


def select_users(con, args):
    if args.users:
        prefixes = [u.strip() for u in args.users.split(",") if u.strip()]
        ids = []
        for p in prefixes:
            row = con.execute(
                "SELECT user_id, COUNT(*) c FROM tsumego_reports WHERE user_id LIKE ? "
                "GROUP BY user_id ORDER BY c DESC LIMIT 1", (p + "%",)).fetchone()
            if row:
                ids.append(row[0])
        return ids
    rows = con.execute(
        "SELECT user_id, COUNT(*) total, SUM(is_repeat) repeats "
        "FROM tsumego_reports GROUP BY user_id "
        "HAVING total >= ? AND repeats >= ? ORDER BY total DESC LIMIT ?",
        (args.min_total, args.min_repeats, args.top)).fetchall()
    return [r[0] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--users", default=None,
                    help="comma-separated user-id prefixes (overrides auto-select)")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-total", type=int, default=300)
    ap.add_argument("--min-repeats", type=int, default=0)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found — run pipeline.fetch_database first.")

    print("loading ratings/slope/floor maps ...", file=sys.stderr)
    ratings, slopes, floors = {}, {}, {}
    for size in SIZES:
        r, s, f = load_size_value_maps(RATINGS_DIR, size, missing_ok=True)
        ratings.update(r); slopes.update(s); floors.update(f)
    if not ratings:
        sys.exit(f"No ratings under {RATINGS_DIR} — run train_rating.")
    print(f"  {len(ratings)} rated problems", file=sys.stderr)

    con = sqlite3.connect(args.db)
    user_ids = select_users(con, args)
    print(f"selected {len(user_ids)} users", file=sys.stderr)

    users, problems = {}, {}
    dropped_unrated = total_attempts = 0
    for uid in user_ids:
        rows = con.execute(
            "SELECT problem, success, size, time_used, review_time, is_repeat, created_at "
            "FROM tsumego_reports WHERE user_id = ? ORDER BY created_at, id", (uid,)
        ).fetchall()
        out_rows = []
        for problem, success, size, time_used, review_time, is_repeat, created_at in rows:
            if problem not in ratings:
                dropped_unrated += 1
                continue
            if problem not in problems:
                problems[problem] = [ratings[problem],
                                     slopes.get(problem, 1.6),    # A_REF fallback
                                     floors.get(problem, 0.0)]
            out_rows.append([
                problem, 1 if success else 0, int(size),
                int(time_used) if time_used is not None else None,
                int(review_time) if review_time is not None else None,
                1 if is_repeat else 0, parse_ms(created_at),
            ])
        if out_rows:
            users[uid] = out_rows
            total_attempts += len(out_rows)
    con.close()

    with open(args.out, "w") as fh:
        json.dump({"problems": problems, "users": users}, fh)

    print(f"wrote {args.out}", file=sys.stderr)
    print(f"  users={len(users)}  attempts={total_attempts}  "
          f"problems={len(problems)}  dropped_unrated={dropped_unrated}", file=sys.stderr)
    for uid, rws in sorted(users.items(), key=lambda kv: -len(kv[1])):
        reps = sum(r[5] for r in rws)
        print(f"    {uid[:12]:<14} total={len(rws):>6}  repeats={reps:>5}  "
              f"firsttime={len(rws) - reps:>6}", file=sys.stderr)


if __name__ == "__main__":
    main()
