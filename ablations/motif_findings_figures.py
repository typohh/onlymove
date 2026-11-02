#!/usr/bin/env python3
"""How many findings the motif pane produces, by history length — the figures
docs/methodology.md §2.9 quotes about what a user is told at one and at two
standard errors as a function of how much record they have.

Every count comes from RUNNING the shipped readings on recorded histories:
`ablations/motif_findings.mjs` imports `pwa/js/motif-readings.js` (the two
per-axis readings), `pwa/shared/rank-window.js` (the rank the readings are
conditioned on) and `pwa/js/math-utils.js` (the stored-vector quantization)
unmodified under node. This module assembles what the client would hold for
each attempt — the published per-problem triple, the correct-move vector, the
played move's per-cell vector and the availability-null masses, from the
published sidecars exactly as `/refit` serves them — runs the helper, and
turns its output into ledger figures.

Buckets. Below the shipped window, "a user with N attempts" is every account's
FIRST N attempts: one window per account, since that is the record such a user
has. At the shipped window and beyond, every position from N onward is a
window — the latest N attempts at that point — averaged within the account and
then across accounts with one weight each, so every window an account ever
showed counts and no account counts more than once. Windows wider than the
shipped one are hypothetical: the readings run with the window widened and
everything else as shipped.

  findings.readings_per_user   readings a user can carry at most (two per axis)
  findings.null_1sd_pct        share of readings the gate passes by chance alone
  findings.null_2sd_pct        the same at twice the gate
  and per bucket:
  findings.<bucket>.users      accounts in the bucket
  findings.<bucket>.readings   readings that exist at all, per user
  findings.<bucket>.mean_1sd   findings per user clearing the shared gate
                               (|z| > MOTIF_Z), both readings, all axes
  findings.<bucket>.mean_2sd   the same at twice the gate — the pane's full shade
  findings.<bucket>.null_1sd   findings a user with nothing behind any reading
  findings.<bucket>.null_2sd     would collect: the readings times the pass share
  findings.<bucket>.surplus_1sd  the count over that — what the pane is reporting
  findings.<bucket>.surplus_2sd

    ./.venv/bin/python3 -m ablations.motif_findings_figures
"""

import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import AblationError, Figure, require  # noqa: E402
from pipeline import package_for_pwa as pkg  # noqa: E402
from utils.katago.coords import gtp_to_xy  # noqa: E402
from utils.constants import not_ignored_sql  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "about"

PWA_ROOT = os.path.join(PROJECT_ROOT, "pwa")
TSUMEGOS = os.path.join(PWA_ROOT, "tsumegos")
HELPER = os.path.join(PROJECT_ROOT, "ablations", "motif_findings.mjs")

# (name, history length, mode). `first` reads one window per account, its first
# N attempts; `last` reads every window of the latest N from position N onward.
BUCKETS = (
    ("first50", 50, "first"),
    ("first100", 100, "first"),
    ("first200", 200, "first"),
    ("first300", 300, "first"),
    ("last500", 500, "last"),
    ("last1000", 1000, "last"),
    ("last2000", 2000, "last"),
)
SHIPPED_BUCKET = "last500"
MIN_HISTORY = min(n for _, n, _ in BUCKETS)

# The cell map's reserved pass code; no board coordinate can produce it.
PASS_POINT_CODE = 255


# --- the published corpus, addressed the way /refit addresses it -------------------

def _read_json(path):
    require(os.path.exists(path), f"no {path} — run pipeline.package_for_pwa")
    with open(path) as f:
        return json.load(f)


def load_published():
    """Everything a re-derive reads per problem, keyed by id: the lookup row
    (the address into every sidecar), the shard, the board size, the published
    triple, the correct-move vector and the availability-null masses; plus the
    per-cell block addresses for the played-move vectors."""
    meta = _read_json(os.path.join(TSUMEGOS, "embedding_meta.json"))
    K = meta.get("K")
    require(isinstance(K, int) and K > 0, "embedding_meta.json has no usable K")
    require(meta.get("dtype") == "float16",
            f"published vectors are {meta.get('dtype')!r}, not float16")
    bytes_per_cell = int(meta["bytes_per_cell"])
    require(bytes_per_cell == 2 * K, f"bytes_per_cell {bytes_per_cell} is not 2 x K")
    offsets_row = int(meta["offsets_bytes_per_row"])
    require(offsets_row == 5, f"offsets row is {offsets_row} bytes, expected 5")
    altstats_row = int(meta["altstats_bytes_per_row"])
    require(meta.get("altstats_dtype") == "uint32" and altstats_row == 8 * K,
            "altstats layout is not 2K little-endian uint32 masses")

    rows = pkg._lookup_rows(PWA_ROOT)
    require(rows, f"{PWA_ROOT}/lookup.txt is empty — run pipeline.package_for_pwa")
    n = len(rows)
    pos = {tid: i for i, (tid, _, _) in enumerate(rows)}
    require(len(pos) == n, "lookup.txt holds a duplicate id")

    def sidecar(name, dtype, per_row):
        path = os.path.join(TSUMEGOS, name)
        require(os.path.exists(path), f"no {path} — run pipeline.package_for_pwa")
        raw = np.fromfile(path, dtype=dtype)
        require(raw.size == n * per_row,
                f"{name} holds {raw.size / per_row:g} rows against lookup.txt's {n}")
        return raw.reshape(n, per_row)

    emb = sidecar("lookup.embeddings.bin", "<f2", K).astype(np.float32)
    alt = sidecar("lookup.altstats.bin", "<u4", 2 * K)
    off = sidecar("lookup.offsets.bin", np.uint8, offsets_row)
    offsets = off[:, :4].copy().view("<u4").reshape(n).astype(np.int64)
    n_cells = off[:, 4].astype(np.int64)
    require(int((n_cells == 0).sum()) == 0,
            "lookup.offsets.bin carries the absent sentinel (n_cells 0) for some problem")

    params_path = os.path.join(TSUMEGOS, "problem_params.bin")
    require(os.path.exists(params_path), f"no {params_path} — run pipeline.package_for_pwa")
    raw = np.fromfile(params_path, dtype=np.uint8)
    require(raw.size % 16 == 0, "problem_params.bin is not a multiple of 16 bytes")
    raw = raw.reshape(-1, 16)
    ids = [row.tobytes().hex() for row in raw[:, :4]]
    vals = raw[:, 4:].copy().view("<f4").reshape(-1, 3)
    require(len(ids) == n, f"problem_params.bin holds {len(ids)} problems against {n} lookup rows")
    params = {}
    for tid, (r, s, f) in zip(ids, vals):
        require(tid in pos, f"problem_params id {tid} is absent from lookup.txt")
        params[tid] = (float(r), float(s), float(f))

    return {
        "K": K, "bytes_per_cell": bytes_per_cell, "rows": rows, "pos": pos,
        "emb": emb, "alt": alt, "offsets": offsets, "n_cells": n_cells, "params": params,
    }


def move_point_code(move, size):
    """The published cell map's code for a played move, `y * size + x` in the
    board frame the SGFs are written in (y = 0 at the top). Mirrors
    `movePointCode` in pwa/src/handlers/refit.js, through the one coordinate
    helper (`gtp_to_xy` counts rows from the bottom). -1 when the move is not a
    point on this board."""
    if not isinstance(move, str) or not move or move.lower() == "pass":
        return -1
    try:
        x, y_from_bottom = gtp_to_xy(move.upper())
    except (ValueError, IndexError):
        return -1
    if not (0 <= x < size and 0 <= y_from_bottom < size):
        return -1
    return (size - 1 - y_from_bottom) * size + x


# --- the attempt log, as each client would hold it ---------------------------------

def load_histories(pub):
    """{user_id: [attempt, ...]} in event order — every first-time attempt on a
    published problem, for every account with at least MIN_HISTORY of them.
    Each attempt is the client's row: [size, correct, rating, slope, floor,
    correct-move vector, played-move vector | None, above masses | None, below
    masses | None]; the last three are filled for wrong attempts below."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        # Event order, not insert order: a backfill inserts historical event
        # times at the id frontier, and the client's record is in event order.
        cur = con.execute(
            "SELECT user_id, problem, size, success, move FROM tsumego_reports "
            f" WHERE is_repeat = 0 AND {not_ignored_sql()} ORDER BY user_id, created_at, id")
        by_user = {}
        for uid, tid, size, ok, move in cur:
            tid = str(tid)
            if tid not in pub["pos"]:
                continue        # retired: the client strips its fields and skips it
            by_user.setdefault(uid, []).append((tid, int(size), 1 if ok else 0, move))
    finally:
        con.close()
    require(by_user, "tsumego_reports holds no first-time attempts on published problems")

    K = pub["K"]
    histories, wanted = {}, {}
    for uid in sorted(by_user):
        recs = by_user[uid]
        if len(recs) < MIN_HISTORY:
            continue
        rows = []
        for tid, size, ok, move in recs:
            i = pub["pos"][tid]
            _, zip_hash, lookup_size = pub["rows"][i]
            require(lookup_size == size,
                    f"{tid}: recorded on a {size}x{size} board, published as {lookup_size}")
            vec = pub["emb"][i]
            # An all-zero row is packaging's placeholder for an id with no
            # vector — absent, not a vector of zeros (refit.js).
            emb = None if not np.any(vec) else pkg._js_quantize(vec).tolist()
            above, below = None, None
            if not ok:
                a = pub["alt"][i]
                if int(a.sum()) > 0:
                    above = a[:K].astype(int).tolist()
                    below = a[K:].astype(int).tolist()
            r, s, f = pub["params"][tid]
            row = [size, ok, r, s, f, emb, None, above, below]
            rows.append(row)
            if not ok:
                code = move_point_code(move, size)
                if code >= 0:
                    wanted.setdefault((size, zip_hash), []).append((i, code, row))
        histories[uid] = rows
    require(histories, f"no account has {MIN_HISTORY} first-time attempts on published problems")
    fill_picked(pub, wanted)
    return histories


def fill_picked(pub, wanted):
    """The played move's per-cell vector for every wrong attempt whose move
    resolves to a root variation: the cell map turns the coordinate into the
    variation index, the block address plus that index into the vector — the
    same walk /refit makes, one shard read per shard."""
    K = pub["K"]
    bpc = pub["bytes_per_cell"]
    for (size, zip_hash), items in sorted(wanted.items()):
        cells_path = os.path.join(TSUMEGOS, str(size), f"{zip_hash}.cells.bin")
        bin_path = os.path.join(TSUMEGOS, str(size), f"{zip_hash}.embeddings.bin")
        require(os.path.exists(cells_path) and os.path.exists(bin_path),
                f"shard {size}/{zip_hash} lacks its cells or embeddings sidecar")
        codes = np.fromfile(cells_path, dtype=np.uint8)
        blob = np.fromfile(bin_path, dtype="<f2")
        for i, code, row in items:
            start = pub["offsets"][i] // bpc
            n = pub["n_cells"][i]
            require(start + n <= codes.size,
                    f"{pub['rows'][i][0]}: its block runs past {size}/{zip_hash}.cells.bin")
            block = codes[start:start + n]
            hit = np.flatnonzero(block == code)
            if hit.size == 0:
                continue        # the move matches no variation: no vector, as on the client
            idx = int(hit[0])
            vec = blob[(start + idx) * K:(start + idx + 1) * K]
            require(vec.size == K,
                    f"{pub['rows'][i][0]}: cell {idx} runs past {size}/{zip_hash}.embeddings.bin")
            # An all-zero cell is the pass placeholder, which no click selects.
            if np.any(vec):
                row[6] = pkg._js_quantize(vec.astype(np.float32)).tolist()


# --- the helper --------------------------------------------------------------------

def run_helper(pub, histories):
    node = shutil.which("node")
    require(node is not None, "no `node` on PATH — the client modules run under node")
    require(os.path.exists(HELPER), f"no {HELPER}")
    with tempfile.TemporaryDirectory(prefix="motif-findings-") as tmp:
        payload_path = os.path.join(tmp, "payload.json")
        with open(payload_path, "w") as f:
            json.dump({
                "pwa_root": PWA_ROOT + os.sep,
                "K": pub["K"],
                "buckets": [{"name": nm, "n": n, "mode": mode} for nm, n, mode in BUCKETS],
                "users": [{"id": uid, "rows": rows} for uid, rows in histories.items()],
            }, f)
        proc = subprocess.run([node, HELPER, payload_path], cwd=PROJECT_ROOT,
                              capture_output=True, text=True)
        require(proc.returncode == 0,
                f"motif_findings.mjs failed ({proc.returncode}):\n{proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise AblationError(f"motif_findings.mjs wrote no JSON: {e}") from None


# --- the measurement ---------------------------------------------------------------

FIELDS = ("avail", "one", "two", "t_avail", "t_one", "t_two", "s_avail", "s_one", "s_two")


def two_sided_tail(z):
    """P(|Z| > z) for a standard normal — what a reading with no effect behind
    it passes the gate with."""
    return math.erfc(z / math.sqrt(2.0))


def measure():
    pub = load_published()
    histories = load_histories(pub)
    t0 = time.time()
    out = run_helper(pub, histories)
    elapsed = time.time() - t0

    require(out["K"] == pub["K"], "the helper and the producer disagree about K")
    motif_z, strong_z = float(out["motif_z"]), float(out["strong_z"])
    require(motif_z > 0 and strong_z > motif_z, "the gate levels are not ordered")
    p1, p2 = two_sided_tail(motif_z), two_sided_tail(strong_z)

    buckets = {}
    for name, n, mode in BUCKETS:
        users = out["buckets"][name]["users"]
        expected = sum(1 for rows in histories.values() if len(rows) >= n)
        require(len(users) == expected,
                f"{name}: the helper measured {len(users)} accounts, {expected} qualify")
        require(users, f"{name}: no account has {n} attempts")
        b = {"n": n, "mode": mode, "users": len(users),
             "windows": int(sum(u["windows"] for u in users))}
        for f in FIELDS:
            b[f] = float(np.mean([u[f] for u in users]))
        b["null_one"] = b["avail"] * p1
        b["null_two"] = b["avail"] * p2
        b["surplus_one"] = b["one"] - b["null_one"]
        b["surplus_two"] = b["two"] - b["null_two"]
        buckets[name] = b
    return {
        "K": pub["K"], "motif_z": motif_z, "strong_z": strong_z,
        "shipped_window": out["shipped_window"], "rank_window": out["rank_window"],
        "p1": p1, "p2": p2, "accounts": len(histories), "elapsed": elapsed,
        "buckets": buckets,
    }


def figures():
    m = measure()
    K = m["K"]
    figs = [
        Figure("findings.readings_per_user", 2 * K, "d",
               f"readings the motif pane can show one user at most: two per axis, {K} axes"),
        Figure("findings.null_1sd_pct", m["p1"], "%.0f",
               f"share of readings with nothing behind them that the gate passes (|z| > {m['motif_z']:g})"),
        Figure("findings.null_2sd_pct", m["p2"], "%.0f",
               f"the same at twice the gate (|z| > {m['strong_z']:g}), the pane's full shade"),
    ]
    for name, n, mode in BUCKETS:
        b = m["buckets"][name]
        if mode == "first":
            how = f"each account's first {n} attempts, one window per account"
        else:
            how = (f"every window of the latest {n} attempts from position {n} onward, "
                   f"averaged within the account, then across accounts"
                   + ("" if name == SHIPPED_BUCKET else " — a hypothetical width"))
        figs += [
            Figure(f"findings.{name}.users", b["users"], "d",
                   f"accounts with at least {n} first-time attempts on published problems"),
            Figure(f"findings.{name}.readings", b["avail"], ".1f",
                   f"readings that exist at all per user over {how} (a tendency reading "
                   f"needs enough directional errors, an accuracy reading enough attempts)"),
            Figure(f"findings.{name}.mean_1sd", b["one"], ".1f",
                   f"findings per user clearing the gate (|z| > {m['motif_z']:g}), both "
                   f"readings over all axes, on the shipped readings run over {how}"),
            Figure(f"findings.{name}.mean_2sd", b["two"], ".1f",
                   f"the same at twice the gate (|z| > {m['strong_z']:g})"),
            Figure(f"findings.{name}.null_1sd", b["null_one"], ".1f",
                   "findings a user with nothing behind any reading would collect there: "
                   "the readings that exist times the gate's chance pass share"),
            Figure(f"findings.{name}.null_2sd", b["null_two"], ".1f",
                   "the same at twice the gate"),
            Figure(f"findings.{name}.surplus_1sd", b["surplus_one"], ".1f",
                   "findings at the gate beyond that chance count — what the pane is "
                   "actually reporting"),
            Figure(f"findings.{name}.surplus_2sd", b["surplus_two"], ".1f",
                   "the same at twice the gate"),
        ]
    return figs


def main():
    m = measure()
    print("\n" + "=" * 96)
    print(f"motif findings by history length — {m['accounts']} accounts with "
          f">= {MIN_HISTORY} first-time attempts on published problems; K={m['K']} axes, "
          f"gate |z| > {m['motif_z']:g}, full shade |z| > {m['strong_z']:g}; shipped "
          f"window {m['shipped_window']}, rank window {m['rank_window']}  "
          f"[helper {m['elapsed']:.0f}s]")
    print("null pass share per reading: "
          f"{100 * m['p1']:.1f}% at the gate, {100 * m['p2']:.1f}% at twice it")
    print()
    print(f"  {'bucket':<10}{'accounts':>9}{'windows':>9}{'avail':>7} | "
          f"{'1sd lean':>9}{'1sd acc':>8}{'1sd all':>8}{'null':>6}{'surplus':>8} | "
          f"{'2sd lean':>9}{'2sd acc':>8}{'2sd all':>8}{'null':>6}{'surplus':>8}")
    for name, n, mode in BUCKETS:
        b = m["buckets"][name]
        print(f"  {name:<10}{b['users']:>9d}{b['windows']:>9d}{b['avail']:>7.1f} | "
              f"{b['t_one']:>9.2f}{b['s_one']:>8.2f}{b['one']:>8.2f}{b['null_one']:>6.2f}"
              f"{b['surplus_one']:>8.2f} | "
              f"{b['t_two']:>9.2f}{b['s_two']:>8.2f}{b['two']:>8.2f}{b['null_two']:>6.2f}"
              f"{b['surplus_two']:>8.2f}")
    print()
    print("  first<N>: each account's first N attempts, one window per account.")
    print("  last<N>:  every window of the latest N from position N onward, averaged")
    print("            within the account, then across accounts (one weight each);")
    print(f"            {SHIPPED_BUCKET} is what ships, the wider ones are hypothetical.")
    print("  avail:    readings per user that exist at all (2 per axis at most);")
    print("            lean = the tendency reading, acc = the accuracy (strength) reading.")
    print("  null:     findings a user with nothing behind any reading would collect,")
    print("            avail x the gate's two-sided normal tail; surplus = all - null.")


if __name__ == "__main__":
    main()
