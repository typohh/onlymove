"""Bucket recently-active users by estimated rank and tsumegos-per-day.

Reads the local D1 dump at attempts/database.sql (from pipeline.fetch_database)
and, for the last 24h and last 7d, reports who is playing and how much, split by
rank. The DB stores no rank, so we proxy each user's rank by the median internal
rank of the problems they were served (the PWA serves problems near the user's
own rank). Problem rating -> internal rank uses pwa/tsumegos/rating_to_rank.json,
the same M curve and floor()-to-kyu/dan convention the client uses.

Three views per window:
  - by rank band:     active users, attempts, median tsumegos/user/day, success
  - by activity band: users, median rank, success  (tsumegos/day = attempts/window)
  - rank x activity cross-tab of user counts

Per-problem ratings are read from the packaged shard zips in pwa/tsumegos/<size>/;
attempts on problems no longer shipped fall into the "unknown" rank bucket but
still count toward volume.

Run after pipeline.fetch_database refreshes the dump:
  ./.venv/bin/python3 -m auxiliary.rank_activity_buckets
  ./.venv/bin/python3 -m auxiliary.rank_activity_buckets --size 9
  ./.venv/bin/python3 -m auxiliary.rank_activity_buckets --now 2026-05-22T00:00:00Z --no-plot
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from statistics import median

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.fit_rank_curves import interp_extrap  # noqa: E402

SQL_PATH = os.path.join(PROJECT_ROOT, "attempts", "database.sql")
TSUMEGOS_DIR = os.path.join(PROJECT_ROOT, "pwa", "tsumegos")

# Internal rank index ri = floor(applyM(rating)); getDescriptiveRank maps
# ri<20 -> (20-ri)k, ri>=20 -> (ri-19)d.  Bands keyed by ri (low <= ri < high).
RANK_BANDS = [
    (float("-inf"), 5,            "16k & weaker"),
    (5,             10,           "15k-11k"),
    (10,            15,           "10k-6k"),
    (15,            20,           "5k-1k"),
    (20,            float("inf"), "1d+"),
]
# tsumegos/day bands (low <= rate < high).
ACTIVITY_BANDS = [
    (0,   5,            "<5"),
    (5,   15,           "5-15"),
    (15,  40,           "15-40"),
    (40,  100,          "40-100"),
    (100, float("inf"), "100+"),
]
WINDOWS = [("last 24h", timedelta(hours=24), 1.0),
           ("last 7d", timedelta(days=7), 7.0)]


# ---------- loading ----------

def load_db():
    if not os.path.exists(SQL_PATH):
        sys.exit(f"Missing {SQL_PATH}. Run `python -m pipeline.fetch_database` first.")
    con = sqlite3.connect(":memory:")
    with open(SQL_PATH) as f:
        con.executescript(f.read())
    return con


def load_curves():
    path = os.path.join(TSUMEGOS_DIR, "rating_to_rank.json")
    if not os.path.exists(path):
        sys.exit(f"Missing {path}. Run `python -m pipeline.map_rating_to_rank` "
                 f"then `python -m pipeline.package_for_pwa` first.")
    return json.load(open(path))


def build_rating_map(sizes):
    """{(size, tsumego_id): rating} from every packaged shard zip's metadata."""
    rating_map = {}
    for size in sizes:
        for zp in glob.glob(os.path.join(TSUMEGOS_DIR, str(size), "*.zip")):
            try:
                with zipfile.ZipFile(zp) as z:
                    meta = json.loads(z.read("metadata.json"))
            except (KeyError, zipfile.BadZipFile):
                continue
            for rec in meta:
                tid = rec.get("filename", "").removesuffix(".sgf")
                if tid and "rating" in rec:
                    rating_map[(size, tid)] = float(rec["rating"])
    return rating_map


# ---------- rank conversion (mirrors pwa/js/rank-decomposition.js) ----------

def rank_index(rating, size, curves):
    c = curves.get(str(size))
    if not c:
        return None
    return interp_extrap(rating, c["ratings"], c["rank_indices"])


def rank_label(ri):
    """Kyu/dan label for an internal rank index, matching getDescriptiveRank."""
    i = int(ri // 1)  # floor, including negatives
    return f"{20 - i}k" if i < 20 else f"{i - 19}d"


def band_label(value, bands):
    for lo, hi, lab in bands:
        if lo <= value < hi:
            return lab
    return bands[-1][2]


# ---------- per-window aggregation ----------

def aggregate(rows, cut, ref, window_days, rating_map, curves):
    """rows: (user_id, problem, success, dt, size). Returns per-user dict."""
    users = defaultdict(lambda: {"attempts": 0, "succ": 0, "repeats": 0, "ris": []})
    for uid, prob, succ, dt, size, isrep in rows:
        if not (cut < dt <= ref):
            continue
        u = users[uid]
        u["attempts"] += 1
        u["succ"] += succ
        u["repeats"] += isrep
        rating = rating_map.get((size, prob))
        if rating is not None:
            ri = rank_index(rating, size, curves)
            if ri is not None:
                u["ris"].append(ri)
    out = {}
    for uid, u in users.items():
        ri = median(u["ris"]) if u["ris"] else None
        out[uid] = {
            "attempts": u["attempts"],
            "succ": u["succ"],
            "repeats": u["repeats"],
            "rate": u["attempts"] / window_days,
            "rank_index": ri,
            "rank_band": band_label(ri, RANK_BANDS) if ri is not None else "unknown",
        }
    return out


# ---------- printing ----------

def _pct(num, den):
    return f"{100 * num / den:5.1f}%" if den else "    - "


def print_window(label, cut, ref, window_days, users, min_attempts):
    users = {u: d for u, d in users.items() if d["attempts"] >= min_attempts}
    print(f"\n{'='*78}\n{label.upper()}  (since {cut.isoformat()})\n{'='*78}")
    if not users:
        print("  no activity in window.")
        return
    n = len(users)
    tot = sum(d["attempts"] for d in users.values())
    succ = sum(d["succ"] for d in users.values())
    reps = sum(d["repeats"] for d in users.values())
    known = sum(1 for d in users.values() if d["rank_index"] is not None)
    print(f"  active users: {n}    attempts: {tot}    "
          f"success: {_pct(succ, tot).strip()}    repeats: {_pct(reps, tot).strip()}")
    print(f"  users with an estimable rank: {known}/{n} "
          f"(rest attempted only unshipped problems -> 'unknown')")

    # --- by rank band ---
    band_order = [b[2] for b in RANK_BANDS] + ["unknown"]
    by_band = defaultdict(list)
    for d in users.values():
        by_band[d["rank_band"]].append(d)
    print(f"\n  By rank band:")
    print(f"    {'band':<14}{'users':>7}{'attempts':>10}"
          f"{'med t/day':>11}{'success':>9}")
    for band in band_order:
        ds = by_band.get(band)
        if not ds:
            continue
        a = sum(d["attempts"] for d in ds)
        s = sum(d["succ"] for d in ds)
        med_rate = median([d["rate"] for d in ds])
        print(f"    {band:<14}{len(ds):>7}{a:>10}{med_rate:>11.1f}"
              f"{_pct(s, a):>9}")

    # --- by activity band ---
    act_order = [b[2] for b in ACTIVITY_BANDS]
    by_act = defaultdict(list)
    for d in users.values():
        by_act[band_label(d["rate"], ACTIVITY_BANDS)].append(d)
    print(f"\n  By activity (tsumegos/day = attempts / {window_days:g} day"
          f"{'s' if window_days != 1 else ''}):")
    print(f"    {'band':<8}{'users':>7}{'med rank':>10}{'attempts':>10}{'success':>9}")
    for band in act_order:
        ds = by_act.get(band)
        if not ds:
            continue
        a = sum(d["attempts"] for d in ds)
        s = sum(d["succ"] for d in ds)
        ris = [d["rank_index"] for d in ds if d["rank_index"] is not None]
        rk = rank_label(median(ris)) if ris else "-"
        print(f"    {band:<8}{len(ds):>7}{rk:>10}{a:>10}{_pct(s, a):>9}")

    # --- cross-tab (user counts) ---
    print(f"\n  Rank x activity (user counts):")
    header = f"    {'band':<14}" + "".join(f"{b:>9}" for b in act_order) + f"{'all':>7}"
    print(header)
    for band in band_order:
        ds = by_band.get(band)
        if not ds:
            continue
        cells = []
        for ab in act_order:
            c = sum(1 for d in ds if band_label(d["rate"], ACTIVITY_BANDS) == ab)
            cells.append(c)
        row = f"    {band:<14}" + "".join(f"{c:>9}" for c in cells) + f"{len(ds):>7}"
        print(row)


def make_plot(per_window, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n  (plot skipped: {e})")
        return
    band_order = [b[2] for b in RANK_BANDS] + ["unknown"]
    fig, axes = plt.subplots(1, len(per_window), figsize=(7 * len(per_window), 5),
                             squeeze=False)
    for ax, (label, users) in zip(axes[0], per_window):
        by_band = defaultdict(list)
        for d in users.values():
            by_band[d["rank_band"]].append(d)
        counts = [len(by_band.get(b, [])) for b in band_order]
        med_rate = [median([d["rate"] for d in by_band[b]]) if by_band.get(b) else 0
                    for b in band_order]
        x = range(len(band_order))
        bars = ax.bar(x, counts, color="C0", alpha=0.8)
        for rect, mr, c in zip(bars, med_rate, counts):
            if c:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                        f"{mr:.0f}/d", ha="center", va="bottom", fontsize=8,
                        color="C3")
        ax.set_xticks(list(x))
        ax.set_xticklabels(band_order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("active users")
        ax.set_title(f"{label}: users by rank band\n(red = median tsumegos/day)")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\n  plot written to {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, choices=(9, 13),
                    help="restrict to one board size (default: both)")
    ap.add_argument("--now", help="reference 'now' ISO timestamp "
                    "(default: latest attempt in the dump)")
    ap.add_argument("--min-attempts", type=int, default=1,
                    help="drop users below this many attempts in the window")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        PROJECT_ROOT, "auxiliary", "rank_activity_buckets.png"))
    args = ap.parse_args()

    sizes = [args.size] if args.size else [9, 13]
    curves = load_curves()
    print("building problem -> rating map from shard zips ...")
    rating_map = build_rating_map(sizes)
    print(f"  {len(rating_map)} problems with ratings across sizes {sizes}")

    con = load_db()
    q = "SELECT user_id, problem, success, created_at, size, is_repeat FROM tsumego_reports"
    rows = []
    for uid, prob, succ, ts, size, isrep in con.execute(q):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        sz = int(size) if size is not None else 9
        if args.size and sz != args.size:
            continue
        rows.append((uid, prob, int(succ), dt, sz, int(isrep) if isrep is not None else 0))
    if not rows:
        sys.exit("no parseable attempt rows found.")

    ref = (datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now
           else max(r[3] for r in rows))
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    print(f"  reference 'now': {ref.isoformat()}    "
          f"({'override' if args.now else 'latest attempt in dump'})")
    if args.size:
        print(f"  board size filter: {args.size}")

    per_window = []
    for label, delta, window_days in WINDOWS:
        users = aggregate(rows, ref - delta, ref, window_days, rating_map, curves)
        print_window(label, ref - delta, ref, window_days, users, args.min_attempts)
        per_window.append((label, {u: d for u, d in users.items()
                                   if d["attempts"] >= args.min_attempts}))

    if not args.no_plot:
        make_plot(per_window, args.out)


if __name__ == "__main__":
    main()
