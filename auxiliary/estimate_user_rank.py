"""Estimate a single user's MLE rating from their attempts, then map it to a rank.

This is a diagnostic for "what rank does the current pipeline assign user X?".
It reproduces the SAME two steps the production client does on the front page:

  1. MLE rating  — the user's 80%-clear v2-rating (`fit_user_rating` @
     CALIBRATION_P, the in-loop estimator shared by the client
     `rank-decomposition.js` and the EGF anchor `map_rating_to_egf`). It is a
     2PL MLE over the user's (problem-rating, success) attempts, using each
     problem's own discrimination a_p as the success-curve slope. Optionally
     restricted to recent attempts (`--recent-days` / `--last-n`) since the live
     front-page rank is a recency-weighted estimate.
  2. Rank        — that rating mapped through the shipped rating→rank curve
     (`ratings_v2/rating_to_rank.json`, the M the client interpolates), reported
     as a rank index, a kyu/dan scalar (1k=0, 1d=1) and a label.

It reads the LOCAL DB copy (`attempts/database.sqlite` by default — the binary is
far faster than re-parsing the .sql dump; see CLAUDE.md) and the backend
`ratings_v2/` artifacts. No network, no GPU.

Usage:
  ./.venv/bin/python3 -m auxiliary.estimate_user_rank <redacted-user-id>
  ./.venv/bin/python3 -m auxiliary.estimate_user_rank <redacted-user-id> --recent-days 90
  ./.venv/bin/python3 -m auxiliary.estimate_user_rank <redacted-user-id> --last-n 500
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.fit_rank_curves import (  # noqa: E402
    CALIBRATION_P, fit_user_rating, interp_extrap_vec,
)

SIZES = [9, 13]
DB_DEFAULT = os.path.join(PROJECT_ROOT, "attempts", "database.sqlite")
V2_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "ratings_v2")


def _parse_ts(ts):
    """ISO-8601 (with trailing Z) → aware datetime."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_problem_maps(v2_dir):
    """Merge per-size {tsumego_id: rating} and {tsumego_id: slope a_p} across
    boards — v2 b_p is one shared cross-board scale, so a problem's id is unique
    and the merge is lossless."""
    ratings, slopes = {}, {}
    for size in SIZES:
        rpath = os.path.join(v2_dir, str(size), "tsumego_ratings_final.csv")
        qpath = os.path.join(v2_dir, str(size), "tsumego_quality_final.csv")
        if os.path.exists(rpath):
            df = pd.read_csv(rpath)
            ratings.update(zip(df["tsumego_id"], df["rating"].astype(float)))
        if os.path.exists(qpath):
            df = pd.read_csv(qpath)
            slopes.update(zip(df["tsumego_id"], df["quality"].astype(float)))
    return ratings, slopes


def load_curve(v2_dir):
    """The shared rating→rank curve (both board keys identical). Returns
    (ratings_knots, rank_index_knots)."""
    path = os.path.join(v2_dir, "rating_to_rank.json")
    with open(path) as f:
        m = json.load(f)
    c = m["9"]
    return np.asarray(c["ratings"], float), np.asarray(c["rank_indices"], float)


def rating_to_rank(R, xs, ys):
    """Map a rating through M. rank_index = scalar + 19 (1k → scalar 0 →
    rank_index 19). Returns (rank_index, scalar, label)."""
    rank_index = float(interp_extrap_vec(np.array([R], float), xs, ys)[0])
    scalar = rank_index - 19.0
    k = int(round(scalar))
    label = f"{1 - k}k" if k <= 0 else f"{k}d"
    return rank_index, scalar, label


def load_user_attempts(db_path, user_id, include_repeats):
    """All matching attempts as (tid, success, size, is_repeat, ts), ascending.
    `user_id` matches as a prefix (the DB stores full ids; the front-page /
    survey code keys on the first 8 chars)."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT problem, success, size, is_repeat, created_at "
            "FROM tsumego_reports WHERE user_id LIKE ? ORDER BY created_at",
            (user_id + "%",),
        ).fetchall()
        surveys = con.execute(
            "SELECT server, rank, created_at FROM rank_surveys "
            "WHERE user_id LIKE ? ORDER BY created_at",
            (user_id + "%",),
        ).fetchall()
    finally:
        con.close()
    if not include_repeats:
        rows = [r for r in rows if not r[3]]
    return rows, surveys


def select_recent(rows, recent_days, last_n):
    """Apply the recency window (most recent attempts only)."""
    if recent_days is not None and rows:
        cutoff = _parse_ts(rows[-1][4]) - timedelta(days=recent_days)
        rows = [r for r in rows if _parse_ts(r[4]) >= cutoff]
    if last_n is not None:
        rows = rows[-last_n:]
    return rows


def estimate(rows, ratings, slopes, xs, ys):
    """fit_user_rating over (tid, success) → rating → rank. Returns a dict."""
    att = [(tid, succ) for tid, succ, *_ in rows]
    matched = sum(1 for tid, _ in att if tid in ratings)
    R, n = fit_user_rating(att, ratings, target=CALIBRATION_P, slopes=slopes,
                           min_n=1)
    if R is None:
        return None
    rank_index, scalar, label = rating_to_rank(R, xs, ys)
    # How many attempts land in the curve's extrapolation region (their problem
    # rating exceeds the top knot) — surfaces the scale mismatch that over-rates
    # strong users.
    above = sum(1 for tid, _ in att
                if tid in ratings and ratings[tid] > xs[-1])
    return {"n_fit": n, "n_matched": matched, "R": R,
            "rank_index": rank_index, "scalar": scalar, "label": label,
            "n_above_top_knot": above}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("user_id", help="user id (prefix-matched against the DB)")
    ap.add_argument("--db", default=DB_DEFAULT, help="local sqlite DB copy")
    ap.add_argument("--v2-dir", default=V2_DIR_DEFAULT)
    ap.add_argument("--recent-days", type=int, default=None,
                    help="only attempts within this many days of the user's "
                         "latest attempt")
    ap.add_argument("--last-n", type=int, default=None,
                    help="only the N most recent attempts")
    ap.add_argument("--include-repeats", action="store_true",
                    help="include is_repeat attempts (default: first-time only, "
                         "matching the client's rank decomposition)")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found — run pipeline.fetch_database first.")

    ratings, slopes = load_problem_maps(args.v2_dir)
    if not ratings:
        sys.exit(f"No v2 ratings under {args.v2_dir} — run train_rating.")
    xs, ys = load_curve(args.v2_dir)

    rows, surveys = load_user_attempts(args.db, args.user_id,
                                       args.include_repeats)
    if not rows:
        sys.exit(f"No attempts for user prefix '{args.user_id}'.")

    repeat_note = "incl. repeats" if args.include_repeats else "first-time only"
    print(f"User '{args.user_id}'  ({repeat_note})")
    by_size = {s: sum(1 for r in rows if r[2] == s) for s in SIZES}
    print(f"  attempts: {len(rows)} total  "
          + "  ".join(f"{s}x{s}={by_size.get(s, 0)}" for s in SIZES))
    print(f"  span: {rows[0][4]}  →  {rows[-1][4]}")
    if surveys:
        decl = "  ".join(f"{srv}:{rk}" for srv, rk, _ in surveys)
        print(f"  declared survey rank(s): {decl}")
    print(f"  curve knot span: rating {xs[0]:.3f} ({ys[0]-19:+.1f}) .. "
          f"{xs[-1]:.3f} ({ys[-1]-19:+.1f}) [scalar]")
    print()

    # Headline estimate (respecting --recent-days / --last-n), then a small
    # window sweep for context on recency sensitivity.
    windows = [("selected", args.recent_days, args.last_n)]
    if args.recent_days is None and args.last_n is None:
        windows = [("all", None, None), ("last 365d", 365, None),
                   ("last 90d", 90, None), ("last 500", None, 500)]

    print(f"{'window':<12}{'n_fit':>7}{'rating R':>11}"
          f"{'rank_idx':>10}{'scalar':>8}{'label':>7}{'>topknot':>10}")
    print("-" * 65)
    for name, rd, ln in windows:
        sub = select_recent(rows, rd, ln)
        res = estimate(sub, ratings, slopes, xs, ys)
        if res is None:
            print(f"{name:<12}{'(no matched attempts)':>30}")
            continue
        print(f"{name:<12}{res['n_fit']:>7}{res['R']:>11.3f}"
              f"{res['rank_index']:>10.2f}{res['scalar']:>8.2f}"
              f"{res['label']:>7}{res['n_above_top_knot']:>10}")


if __name__ == "__main__":
    main()
