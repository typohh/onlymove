#!/usr/bin/env python3
"""Fit the Rank tab's skill smoother constants on the attempt log.

The career skill track (`pwa/js/skill-smoother.js`) models skill as a random walk
in calendar time with a per-session form offset. Its two constants —
SKILL_DRIFT_VAR_PER_DAY (how fast skill wanders) and FORM_SD (how far one sitting
sits from it) — are population maximum-likelihood estimates: the summed log
probability of every first-time attempt's outcome given the same player's attempts
before it, over every player in the local DB.

The objective is computed by `ablations/skill_drift_fit.mjs`, which runs the
shipped smoother and the shipped flattenAttempt unmodified under node; this module
joins the attempt log to the shipped per-problem triple (`problem_params.bin`) and
prints the fit. Two arms are fitted — the walk alone, and the walk with the form
offset — so the table shows what the form term is worth and how far the drift
moves without it.

    ./.venv/bin/python3 -m ablations.skill_drift_fit
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import require  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

PWA_ROOT = os.path.join(PROJECT_ROOT, "pwa")
TSUMEGOS = os.path.join(PWA_ROOT, "tsumegos")
HELPER = os.path.join(PROJECT_ROOT, "ablations", "skill_drift_fit.mjs")


def load_params():
    """problem id -> (rating, slope, floor) from the published problem_params.bin
    (16 bytes per problem: a 4-byte id, then three little-endian float32)."""
    path = os.path.join(TSUMEGOS, "problem_params.bin")
    require(os.path.exists(path), f"no {path} — run pipeline.package_for_pwa")
    raw = np.fromfile(path, dtype=np.uint8)
    require(raw.size % 16 == 0, "problem_params.bin is not a multiple of 16 bytes")
    raw = raw.reshape(-1, 16)
    vals = raw[:, 4:].copy().view("<f4").reshape(-1, 3)
    return {row[:4].tobytes().hex(): tuple(float(v) for v in vals[i])
            for i, row in enumerate(raw)}


def parse_ms(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def load_users(params):
    """Every user's first-time attempts as [t_ms, rating, slope, floor, correct, size].
    Attempts on a problem the published bank no longer carries are dropped, as the
    client drops them (a retired problem's attempts fall out of every estimate)."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(DB_SQLITE)
    try:
        rows = con.execute(
            "SELECT user_id, problem, size, success, created_at FROM tsumego_reports "
            "WHERE is_repeat = 0 ORDER BY user_id, id").fetchall()
    finally:
        con.close()
    users, cur, cur_id, dropped = [], None, None, 0
    for user_id, problem, size, success, created_at in rows:
        if user_id != cur_id:
            cur, cur_id = [], user_id
            users.append(cur)
        p = params.get(problem)
        if p is None:
            dropped += 1
            continue
        cur.append([parse_ms(created_at), p[0], p[1], p[2], 1 if success else 0, int(size)])
    return [u for u in users if u], len(rows), dropped


def main():
    require(shutil.which("node"), "node is not on PATH")
    params = load_params()
    users, n_rows, dropped = load_users(params)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "payload.json")
        with open(path, "w") as f:
            json.dump({"pwa_root": PWA_ROOT + "/", "users": users}, f)
        out = subprocess.run(["node", HELPER, path], capture_output=True, text=True,
                             cwd=os.path.dirname(HELPER))
    require(out.returncode == 0, f"skill_drift_fit.mjs failed:\n{out.stderr}")
    r = json.loads(out.stdout)

    n = r["nAttempts"]
    print(f"first-time attempts: {n_rows} in the log, {dropped} on problems no longer "
          f"published, {n} scored over {r['nUsers']} players")
    print("\nmean log loss is per attempt, in nats (lower is better)\n")
    print(f"{'arm':<22}{'drift var/day':>15}{'drift SD/30d':>14}{'form SD':>10}{'mean log loss':>16}")
    for name, a in (("walk only", r["noForm"]), ("walk + session form", r["form"])):
        sd30 = (a["driftVarPerDay"] * 30) ** 0.5
        form = f"{a['formSd']:.3f}" if name != "walk only" else "—"
        print(f"{name:<22}{a['driftVarPerDay']:>15.5f}{sd30:>14.3f}{form:>10}{-a['logLik'] / n:>16.5f}")
    print(f"\ndrift profile at form SD {r['form']['formSd']:.3f}:")
    print(f"{'drift var/day':>15}{'drift SD/30d':>14}{'mean log loss':>16}")
    for p in r["profile"]:
        print(f"{p['driftVarPerDay']:>15.5f}{(p['driftVarPerDay'] * 30) ** 0.5:>14.3f}"
              f"{-p['logLik'] / n:>16.5f}")


if __name__ == "__main__":
    main()
