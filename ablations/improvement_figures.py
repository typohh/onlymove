#!/usr/bin/env python3
"""How much players improve while using the app — the table pwa/about.html shows,
measured two independent ways and split into double-digit kyu, single-digit kyu
and dan, every change stated per 100 first-time problems solved.

  Rating  each player's rank from the SHIPPED window estimator on three windows of
          WINDOW first-time attempts — their first, the one centred on the middle of
          their history, and their latest — for every player with at least
          3×WINDOW (so the windows never overlap). EARLY is first → middle, LATE is
          middle → latest, and the problems a step spans are those between its two
          windows' centres. The estimator runs unmodified under node
          (`ablations/improvement.mjs`) on the published problem parameters, so every
          window is read on today's scale. It is the posterior MEAN, not the shown
          rank: the display offset tracks how loose the posterior is, which differs
          between windows for reasons that are not strength.
  Survey  the rank a player declared on a server, first declaration against last,
          for every (player, server) with declarations on two or more UTC days, over
          the first-time problems solved between the two. The declarations the rank
          mapping screened out as implausible (`ratings/declaration_exclusions.csv`)
          void their whole (player, server) pair. A player on several servers counts
          once, averaged over them.

A player's band is the midpoint of their first and last readings, never the first
alone: banding on the first reading puts the players it under-read in the weakest
band, and they then "improve" by regression to the mean alone.

The rate is the band's TOTAL change over its TOTAL problems (mean change / mean
problems × 100). Per player it would be dominated by whoever solved almost nothing
between two readings, and over the median it credits them with problems they did not
solve. Its spread is the across-player SD of the change scaled by the same factor.
"Improved" is a gain of at least IMPROVED_RATING ranks on the continuous rating, and
any gain on the survey, whose ranks are whole bands — so both count roughly one
band's step.

    improvement.window_problems                  the WINDOW itself
    improvement.{rating,survey}_{band}_n         players in the band
    improvement.rating_{band}_{early,late}_per100[_sd], _{early,late}_improved
    improvement.survey_{band}_per100[_sd], survey_{band}_improved
    improvement.rating_{early,late}_problems     median problems a step spans
    improvement.survey_problems                  median problems between declarations

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
from utils.constants import not_ignored_sql  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402
from utils.rating.rank_surveys import SERVERS, rank_band, rank_to_value  # noqa: E402
from utils.rating.ratings_io import load_declaration_exclusions  # noqa: E402

TIER = "about"

PWA_ROOT = os.path.join(PROJECT_ROOT, "pwa")
PARAMS_BIN = os.path.join(PWA_ROOT, "tsumegos", "problem_params.bin")
RATINGS_DIR = os.path.join(PROJECT_ROOT, "ratings")
HELPER = os.path.join(PROJECT_ROOT, "ablations", "improvement.mjs")

WINDOW = 100
IMPROVED_RATING = 0.5
PER_PROBLEMS = 100
# Band edges on the internal rank scale (0 = the 1k/1d boundary, integers are band
# EDGES): double-digit kyu is 10k and weaker, dan is 1d and stronger.
DDK_BELOW = -9.0
DAN_FROM = 0.0
BANDS = ("ddk", "sdk", "dan")
BAND_LABEL = {"ddk": "DDK", "sdk": "SDK", "dan": "Dan"}


def band_of(v):
    return "ddk" if v < DDK_BELOW else ("dan" if v >= DAN_FROM else "sdk")


# --- rating ------------------------------------------------------------------------

def load_params():
    require(os.path.exists(PARAMS_BIN), f"no {PARAMS_BIN} — run pipeline.package_for_pwa")
    raw = np.fromfile(PARAMS_BIN, dtype=np.uint8)
    require(raw.size % 16 == 0, "problem_params.bin is not a multiple of 16 bytes")
    raw = raw.reshape(-1, 16)
    vals = raw[:, 4:].copy().view("<f4").reshape(-1, 3)
    return {row.tobytes().hex(): tuple(float(x) for x in v)
            for row, v in zip(raw[:, :4], vals)}


def usable_attempts(params, users=None):
    """{user: [(rating, slope, floor, success, created_at), ...]} of first-time
    attempts, deduped exactly as the leaderboard's buffer rebuild is (GROUP BY
    problem, created_at keeps the ORIGINAL row), in insert order, and restricted to
    problems still published — the estimator has nothing to read an attempt on a
    retired problem with, and the survey counts problems on the same footing."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT user_id, problem, success, created_at FROM ("
            "  SELECT MIN(id) AS id, user_id, problem, success, created_at"
            f"    FROM tsumego_reports WHERE is_repeat = 0 AND {not_ignored_sql()}"
            "   GROUP BY user_id, problem, created_at)"
            " ORDER BY user_id, id").fetchall()
    finally:
        con.close()
    require(rows, "tsumego_reports holds no first-time attempts")
    by_user = {}
    for user, problem, success, at in rows:
        if users is not None and user not in users:
            continue
        p = params.get(problem)
        if p is not None:
            by_user.setdefault(user, []).append((*p, 1 if success else 0, at))
    return by_user


def rating_windows(params):
    """Per player with at least 3×WINDOW usable first-time attempts, the first,
    middle and latest WINDOW of them, and the problems between consecutive windows'
    centres."""
    players = []
    for user, att in usable_attempts(params).items():
        n = len(att)
        if n < 3 * WINDOW:
            continue
        m0 = n // 2 - WINDOW // 2
        players.append({
            "windows": [[a[:4] for a in w] for w in
                        (att[:WINDOW], att[m0:m0 + WINDOW], att[-WINDOW:])],
            "early_problems": m0, "late_problems": n - WINDOW - m0,
        })
    require(players, f"no player has {3 * WINDOW} usable first-time attempts")
    return players


def run_helper(windows):
    """The shipped estimator's posterior-mean rank for each window."""
    node = shutil.which("node")
    require(node is not None, "no `node` on PATH — the client estimator runs under node")
    require(os.path.exists(HELPER), f"no {HELPER}")
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
    return rank


def measure_rating(params):
    players = rating_windows(params)
    rank = run_helper([w for p in players for w in p["windows"]])
    out = []
    for i, p in enumerate(players):
        first, middle, last = rank[3 * i:3 * i + 3]
        out.append({
            "band": band_of((first + last) / 2),
            "early": {"change": middle - first, "problems": p["early_problems"],
                      "improved": middle - first >= IMPROVED_RATING},
            "late": {"change": last - middle, "problems": p["late_problems"],
                     "improved": last - middle >= IMPROVED_RATING},
        })
    return out


# --- survey ------------------------------------------------------------------------

def measure_survey(params):
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
            f"WHERE server IN ({','.join('?' * len(SERVERS))}) AND {not_ignored_sql()} ORDER BY id",
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
        per_user.setdefault(key[0], []).append((v0, v1, a0, a1))
    require(per_user, "no player re-declared a server rank on a later day")
    attempts = usable_attempts(params, set(per_user))
    out = []
    for user, decl in per_user.items():
        att = attempts.get(user, [])
        change = float(np.mean([v1 - v0 for v0, v1, _, _ in decl]))
        mid = float(np.mean([(v0 + v1) / 2 for v0, v1, _, _ in decl]))
        solved = float(np.mean([sum(1 for a in att if a0 < a[4] <= a1)
                                for _, _, a0, a1 in decl]))
        out.append({"band": band_of(mid), "all": {
            "change": change, "problems": solved, "improved": change > 0}})
    return out, len(voided)


# --- the table ---------------------------------------------------------------------

def rate(steps, source):
    """Total change over total problems, per PER_PROBLEMS; see the module note."""
    change = np.array([s["change"] for s in steps])
    problems = np.array([s["problems"] for s in steps], dtype=float)
    require(problems.mean() > 0, f"{source}: its players solved no problems between readings")
    scale = PER_PROBLEMS / problems.mean()
    return {"per100": float(change.mean()) * scale,
            "per100_sd": float(change.std(ddof=1)) * scale,
            "improved": float(np.mean([s["improved"] for s in steps])),
            "problems": float(np.median(problems))}


def summarize(players, source, phases):
    rows = {}
    for band in BANDS:
        g = [p for p in players if p["band"] == band]
        require(len(g) >= 2, f"{source}: {len(g)} players in {BAND_LABEL[band]}, "
                             "too few for a spread")
        rows[band] = {"n": len(g)}
        for ph in phases:
            rows[band][ph] = rate([p[ph] for p in g], f"{source} {BAND_LABEL[band]} {ph}")
    for ph in phases:
        rows[f"{ph}_problems"] = float(np.median([p[ph]["problems"] for p in players]))
    return rows


def measure():
    params = load_params()
    survey, n_voided = measure_survey(params)
    return {"rating": summarize(measure_rating(params), "rating", ("early", "late")),
            "survey": summarize(survey, "survey", ("all",)), "voided": n_voided}


def figures():
    m = measure()
    rat, sur = m["rating"], m["survey"]
    rating_what = (f"the shipped window estimator's rank over {WINDOW}-attempt windows, "
                   "per 100 first-time problems")
    survey_what = ("declared server rank, first against last declaration, per 100 "
                   "first-time problems solved between them, screened declarations excluded")
    out = [Figure("improvement.window_problems", WINDOW, "d",
                  "first-time attempts in each of the three windows the rating change "
                  "compares (a player's first, middle and latest)")]
    for band in BANDS:
        b = BAND_LABEL[band]
        out.append(Figure(f"improvement.rating_{band}_n", rat[band]["n"], "d",
                          f"{b} players with {3 * WINDOW}+ first-time attempts"))
        for ph, span in (("early", "first to middle window"),
                         ("late", "middle to latest window")):
            r = rat[band][ph]
            out += [
                Figure(f"improvement.rating_{band}_{ph}_per100", r["per100"], "+.2f",
                       f"{b}, {span}: band total change over total problems, {rating_what}"),
                Figure(f"improvement.rating_{band}_{ph}_per100_sd", r["per100_sd"], ".2f",
                       f"{b}, {span}: across-player SD of the change, same scaling"),
                Figure(f"improvement.rating_{band}_{ph}_improved", r["improved"], "%.0f",
                       f"{b}, {span}: share of players gaining {IMPROVED_RATING}+ rank"),
            ]
        r = sur[band]["all"]
        out += [
            Figure(f"improvement.survey_{band}_n", sur[band]["n"], "d",
                   f"{b} players who re-declared a server rank on a later day"),
            Figure(f"improvement.survey_{band}_per100", r["per100"], "+.2f",
                   f"{b}: band total change over total problems, {survey_what}"),
            Figure(f"improvement.survey_{band}_per100_sd", r["per100_sd"], ".2f",
                   f"{b}: across-player SD of the declared change, same scaling"),
            Figure(f"improvement.survey_{band}_improved", r["improved"], "%.0f",
                   f"{b}: share of players whose declared rank rose"),
        ]
    out += [
        Figure("improvement.rating_early_problems", rat["early_problems"], ".0f",
               "median first-time problems between the first and middle windows' centres"),
        Figure("improvement.rating_late_problems", rat["late_problems"], ".0f",
               "median first-time problems between the middle and latest windows' centres"),
        Figure("improvement.survey_problems", sur["all_problems"], ".0f",
               "median first-time problems solved between a player's first and last "
               "declaration"),
    ]
    return out


def main():
    m = measure()
    rat, sur = m["rating"], m["survey"]
    print("\n" + "=" * 78)
    print(f"Rating: shipped window-estimator rank on a player's first, middle and latest "
          f"{WINDOW} first-time attempts (players with {3 * WINDOW}+); early = first → "
          f"middle, late = middle → latest.")
    print(f"Survey: declared server rank, first vs last declaration on a later day; "
          f"{m['voided']} (player, server) pairs voided by the mapping's screen.")
    print(f"Band = midpoint of first and last readings. Per {PER_PROBLEMS} problems = "
          f"band total change / total problems; SD scaled the same.")
    print(f"Improved = rating gain >= {IMPROVED_RATING} rank, survey gain > 0.\n")
    print(f"  {'band':<6}{'n':>5}{'early/100':>11}{'sd':>7}{'up':>6}"
          f"{'late/100':>11}{'sd':>7}{'up':>6} |{'n':>5}{'survey/100':>12}{'sd':>7}{'up':>6}")
    for band in BANDS:
        e, l, s = rat[band]["early"], rat[band]["late"], sur[band]["all"]
        print(f"  {BAND_LABEL[band]:<6}{rat[band]['n']:>5}"
              f"{e['per100']:>+11.2f}{e['per100_sd']:>7.2f}{100 * e['improved']:>5.0f}%"
              f"{l['per100']:>+11.2f}{l['per100_sd']:>7.2f}{100 * l['improved']:>5.0f}% |"
              f"{sur[band]['n']:>5}{s['per100']:>+12.2f}{s['per100_sd']:>7.2f}"
              f"{100 * s['improved']:>5.0f}%")
    print(f"  median problems spanned: early {rat['early_problems']:.0f}, late "
          f"{rat['late_problems']:.0f}, between declarations {sur['all_problems']:.0f}")


if __name__ == "__main__":
    main()
