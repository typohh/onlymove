#!/usr/bin/env python3
"""How much players improve while using the app — the table pwa/about.html shows,
measured two independent ways and split into double-digit kyu, single-digit kyu
and dan.

  Rating  each player's rank from the SHIPPED window estimator on their first
          WINDOW first-time attempts against their latest WINDOW, for every player
          with at least 2×WINDOW (so the two windows never overlap). The estimator
          runs unmodified under node (`ablations/improvement.mjs`) on the published
          problem parameters, so both windows are read on today's scale. It is the
          posterior MEAN, not the shown rank: the display offset tracks how loose the
          posterior is, which differs between a first and a latest window for reasons
          that are not strength.
  Survey  the rank a player declared on a server, first declaration against last,
          for every (player, server) with declarations on two or more UTC days.
          The declarations the rank mapping screened out as implausible
          (`ratings/declaration_exclusions.csv`) void their whole (player, server)
          pair. A player on several servers counts once, averaged over them.

A player's band is the midpoint of their two readings, never the first alone: banding
on the first reading puts the players it under-read in the weakest band, and they
then "improve" by regression to the mean alone.

The change is normalised to 30 days as the band's mean change over its median days
between readings (for the rating, between the two windows' middle attempts); its
spread is the across-player SD scaled by the same factor. "Improved" is a gain of at
least IMPROVED_RATING ranks on the continuous rating, and any gain on the survey,
whose ranks are whole bands — so both count roughly one band's step.

    improvement.window_problems                the WINDOW itself
    improvement.{rating,survey}_{band}_n       players in the band
    improvement.{rating,survey}_{band}_per30   mean change per 30 days, ranks
    improvement.{rating,survey}_{band}_per30_sd  its across-player spread
    improvement.{rating,survey}_{band}_improved  share of players who improved
    improvement.{rating,survey}_days           median days between readings, all bands

    ./.venv/bin/python3 -m ablations.improvement_figures
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import AblationError, Figure, require  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import SERVERS, rank_band, rank_to_value  # noqa: E402
from utils.rating.ratings_io import load_declaration_exclusions  # noqa: E402

TIER = "about"

PWA_ROOT = os.path.join(PROJECT_ROOT, "pwa")
PARAMS_BIN = os.path.join(PWA_ROOT, "tsumegos", "problem_params.bin")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
HELPER = os.path.join(PROJECT_ROOT, "ablations", "improvement.mjs")

WINDOW = 150
IMPROVED_RATING = 0.5
PER_DAYS = 30
# Band edges on the internal rank scale (0 = the 1k/1d boundary, integers are band
# EDGES): double-digit kyu is 10k and weaker, dan is 1d and stronger.
DDK_BELOW = -9.0
DAN_FROM = 0.0
BANDS = ("ddk", "sdk", "dan")
BAND_LABEL = {"ddk": "DDK", "sdk": "SDK", "dan": "Dan"}


def band_of(v):
    return "ddk" if v < DDK_BELOW else ("dan" if v >= DAN_FROM else "sdk")


def _days(a, b):
    return (np.datetime64(b[:23]) - np.datetime64(a[:23])) / np.timedelta64(1, "D")


# --- rating ------------------------------------------------------------------------

def load_params():
    require(os.path.exists(PARAMS_BIN), f"no {PARAMS_BIN} — run pipeline.package_for_pwa")
    raw = np.fromfile(PARAMS_BIN, dtype=np.uint8)
    require(raw.size % 16 == 0, "problem_params.bin is not a multiple of 16 bytes")
    raw = raw.reshape(-1, 16)
    vals = raw[:, 4:].copy().view("<f4").reshape(-1, 3)
    return {row.tobytes().hex(): tuple(float(x) for x in v)
            for row, v in zip(raw[:, :4], vals)}


def rating_windows(params):
    """Per player with at least 2×WINDOW usable first-time attempts, the first and
    latest WINDOW of them. Deduped exactly as the leaderboard's buffer rebuild is
    (GROUP BY problem, created_at keeps the ORIGINAL row), in insert order, and
    restricted to problems still published — the estimator has nothing to read an
    attempt on a retired problem with."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT user_id, problem, success, created_at FROM ("
            "  SELECT MIN(id) AS id, user_id, problem, success, created_at"
            "    FROM tsumego_reports WHERE is_repeat = 0"
            "   GROUP BY user_id, problem, created_at)"
            " ORDER BY user_id, id").fetchall()
    finally:
        con.close()
    require(rows, "tsumego_reports holds no first-time attempts")
    by_user = {}
    for user, problem, success, at in rows:
        p = params.get(problem)
        if p is not None:
            by_user.setdefault(user, []).append((*p, 1 if success else 0, at))
    players = []
    for user, att in by_user.items():
        if len(att) < 2 * WINDOW:
            continue
        first, last = att[:WINDOW], att[-WINDOW:]
        players.append({
            "user": user,
            "first": [a[:4] for a in first], "last": [a[:4] for a in last],
            "days": _days(first[WINDOW // 2][4], last[WINDOW // 2][4]),
        })
    require(players, f"no player has {2 * WINDOW} usable first-time attempts")
    return players


def run_helper(players):
    node = shutil.which("node")
    require(node is not None, "no `node` on PATH — the client estimator runs under node")
    require(os.path.exists(HELPER), f"no {HELPER}")
    windows = [w for p in players for w in (p["first"], p["last"])]
    with tempfile.TemporaryDirectory(prefix="improvement-") as tmp:
        path = os.path.join(tmp, "payload.json")
        with open(path, "w") as f:
            json.dump({"pwa_root": PWA_ROOT + os.sep, "windows": windows}, f)
        proc = subprocess.run([node, HELPER, path], cwd=PROJECT_ROOT,
                              capture_output=True, text=True)
    require(proc.returncode == 0,
            f"improvement.mjs failed ({proc.returncode}):\n{proc.stderr.strip()}")
    try:
        rank = json.loads(proc.stdout)["rank"]
    except (json.JSONDecodeError, KeyError) as e:
        raise AblationError(f"improvement.mjs wrote no rank list: {e}") from None
    require(len(rank) == len(windows),
            f"improvement.mjs returned {len(rank)} ranks for {len(windows)} windows")
    require(all(np.isfinite(r) for r in rank), "the estimator returned a non-finite rank")
    return [{"first": rank[2 * i], "last": rank[2 * i + 1], "days": p["days"]}
            for i, p in enumerate(players)]


def measure_rating():
    readings = run_helper(rating_windows(load_params()))
    return [{"band": band_of((r["first"] + r["last"]) / 2), "change": r["last"] - r["first"],
             "days": r["days"], "improved": r["last"] - r["first"] >= IMPROVED_RATING}
            for r in readings]


# --- survey ------------------------------------------------------------------------

def measure_survey():
    """One entry per player who re-declared a server rank on a later day. The
    machine heartbeat rows (EGD, AGAR) share the table but are not declarations,
    and are left out by keeping SERVERS only."""
    excluded = load_declaration_exclusions(RATINGS_DIR)
    require(os.path.exists(os.path.join(RATINGS_DIR, "declaration_exclusions.csv")),
            "no ratings/declaration_exclusions.csv — run pipeline.map_rating_to_rank")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT user_id, server, rank, created_at FROM rank_surveys "
            f"WHERE server IN ({','.join('?' * len(SERVERS))}) ORDER BY id",
            SERVERS).fetchall()
    finally:
        con.close()
    require(rows, "rank_surveys holds no declarations")
    pairs, voided = {}, set()
    for user, server, rank, at in rows:
        v = rank_to_value(rank)
        if v is None:                         # a professional rank has no band here
            continue
        if (user[:8], server, rank_band(v)) in excluded:
            voided.add((user, server))
        pairs.setdefault((user, server), []).append((v, at))
    per_user = {}
    for key, decl in pairs.items():
        if key in voided or len({at[:10] for _, at in decl}) < 2:
            continue
        (v0, a0), (v1, a1) = decl[0], decl[-1]
        per_user.setdefault(key[0], []).append((v0, v1, _days(a0, a1)))
    require(per_user, "no player re-declared a server rank on a later day")
    out = []
    for decl in per_user.values():
        change = float(np.mean([v1 - v0 for v0, v1, _ in decl]))
        mid = float(np.mean([(v0 + v1) / 2 for v0, v1, _ in decl]))
        out.append({"band": band_of(mid), "change": change,
                    "days": float(np.median([d for _, _, d in decl])),
                    "improved": change > 0})
    return out, len(voided)


# --- the table ---------------------------------------------------------------------

def summarize(players, source):
    rows = {}
    for band in BANDS:
        g = [p for p in players if p["band"] == band]
        require(len(g) >= 2, f"{source}: {len(g)} players in {BAND_LABEL[band]}, "
                             "too few for a spread")
        change = np.array([p["change"] for p in g])
        days = float(np.median([p["days"] for p in g]))
        require(days > 0, f"{source}: {BAND_LABEL[band]}'s median days between readings is {days}")
        scale = PER_DAYS / days
        rows[band] = {
            "n": len(g), "days": days, "mean": float(change.mean()),
            "per30": float(change.mean()) * scale,
            "per30_sd": float(change.std(ddof=1)) * scale,
            "improved": float(np.mean([p["improved"] for p in g])),
        }
    rows["days"] = float(np.median([p["days"] for p in players]))
    return rows


def measure():
    survey, n_voided = measure_survey()
    return {"rating": summarize(measure_rating(), "rating"),
            "survey": summarize(survey, "survey"), "voided": n_voided}


def figures():
    m = measure()
    out = [Figure("improvement.window_problems", WINDOW, "d",
                  "first-time attempts in each of the two windows the rating change "
                  "compares (a player's first and latest)")]
    what = {
        "rating": "the shipped window estimator's rank, first against latest "
                  f"{WINDOW} first-time attempts",
        "survey": "declared server rank, first against last declaration, screened "
                  "declarations excluded",
    }
    for source in ("rating", "survey"):
        s = m[source]
        fmt = "+.1f" if source == "rating" else "+.2f"
        for band in BANDS:
            r = s[band]
            b = BAND_LABEL[band]
            out += [
                Figure(f"improvement.{source}_{band}_n", r["n"], "d",
                       f"{b} players measured by {what[source]}"),
                Figure(f"improvement.{source}_{band}_per30", r["per30"], fmt,
                       f"{b}: mean change per {PER_DAYS} days in ranks, {what[source]}"),
                Figure(f"improvement.{source}_{band}_per30_sd", r["per30_sd"], fmt[1:],
                       f"{b}: across-player SD of that change, same scaling"),
                Figure(f"improvement.{source}_{band}_improved", r["improved"], "%.0f",
                       f"{b}: share of players who improved, {what[source]}"),
            ]
        out.append(Figure(f"improvement.{source}_days", s["days"], ".0f",
                          f"median days between the two readings, {what[source]}"))
    return out


def main():
    m = measure()
    print("\n" + "=" * 78)
    print(f"Rating: shipped window-estimator rank, first vs latest {WINDOW} first-time "
          f"attempts (players with {2 * WINDOW}+).")
    print(f"Survey: declared server rank, first vs last declaration on a later day; "
          f"{m['voided']} (player, server) pairs voided by the mapping's screen.")
    print(f"Band = midpoint of the two readings. Per {PER_DAYS} days = mean change / "
          f"median days × {PER_DAYS}; SD scaled the same.")
    print(f"Improved = rating gain >= {IMPROVED_RATING} rank, survey gain > 0.\n")
    print(f"  {'':<8}{'band':<6}{'players':>8}{'days':>7}{'mean':>8}"
          f"{'per 30d':>10}{'sd':>8}{'improved':>10}")
    for source in ("rating", "survey"):
        for band in BANDS:
            r = m[source][band]
            print(f"  {source:<8}{BAND_LABEL[band]:<6}{r['n']:>8}{r['days']:>7.0f}"
                  f"{r['mean']:>+8.2f}{r['per30']:>+10.2f}{r['per30_sd']:>8.2f}"
                  f"{100 * r['improved']:>9.0f}%")
        print(f"  {source:<8}median days between readings, all bands: "
              f"{m[source]['days']:.0f}")


if __name__ == "__main__":
    main()
