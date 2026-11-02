#!/usr/bin/env python3
"""Four corpus-level claims, measured against what ships.

None of these is a property of a fit's quality; each is a property of the SET —
which problems anyone has played, where an axis's extremes sit relative to that,
how the published variation lists are ordered, and how often the pass is the
position's own best refutation. They share a producer because they share their
inputs: the published index, the local attempt log, and the two sidecars beside
them.

**Never attempted.** A problem in the published index with no first-time human
attempt in the attempt log. Repeats are excluded on the reading that a repeat is
a second look at a problem already counted, so the question "has anyone met this
position" is answered by the first-time rows alone; the two populations differ
by a handful of problems either way. This is the base rate the per-problem
calibration argument and the axis-tail argument both stand against.

**Axis-tail enrichment.** Motif axes are taught one-sided: a bare rule name in
the taught specification puts the rule at the axis's POSITIVE tail and leaves the
negative tail as untaught residual (an `A/B` specification teaches both tails, so
such an axis has no untaught end and is excluded from the comparison and counted
separately). Per axis this takes the thousand most extreme problems at each tail
and reports the never-attempted share there, which is what says whether a name
checked on the taught tail is checked against problems people have played.

**Pass position.** Every problem's published per-cell block is in move-tree order
with an all-zero placeholder row for the pass variation, so that block row *i* is
variation index *i*. The placeholder only matters where the pass is not the LAST
variation, which is what this counts, per board size. It is read from the
published sidecars alone — `lookup.offsets.bin` for each problem's block address
and `<size>/<hash>.cells.bin` for the same cells' board points, one uint8 each
with 255 reserved for the pass — so no SGF is parsed and the figure describes the
shipped bytes rather than the corpus they were built from.

**Ownership-swing pass baseline.** The swing signal prices a position by
differencing the ownership map after the correct move against the map after the
strongest RESOLVED refutation, and a resolved pass is eligible to be that
baseline. Two counts say what that eligibility buys: the problems where the pass
is the ONLY resolved refutation (excluding it there leaves no measurement at all)
and the share where the pass outranks at least one resolved stone move and wins.

Everything the reads assume is checked on the way through, so a run is also a
check on the shipped artifacts: the index rows are well formed and unique, the
per-problem axis table covers exactly the published set with one column per
taught axis, the offsets table is row-aligned to the index with no absent rows,
every problem's cell run holds exactly one pass code, each cell sidecar is
exactly as long as the blocks addressed into it, and the swing cache covers every
published problem.

    ./.venv/bin/python3 -m ablations.corpus_claim_figures
"""

import csv
import json
import os
import sqlite3
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.constants import ALL_BOARD_SIZES, not_ignored_sql  # noqa: E402
from utils.paths import DB_SQLITE  # noqa: E402

TIER = "about"

PWA_DIR = os.path.join(PROJECT_ROOT, "pwa")
TSUMEGOS_DIR = os.path.join(PWA_DIR, "tsumegos")
LOOKUP_PATH = os.path.join(PWA_DIR, "lookup.txt")
OFFSETS_PATH = os.path.join(TSUMEGOS_DIR, "lookup.offsets.bin")
META_PATH = os.path.join(TSUMEGOS_DIR, "embedding_meta.json")
LABELS_PATH = os.path.join(PROJECT_ROOT, "axes", "labels.json")
PER_PROBLEM_PATH = os.path.join(PROJECT_ROOT, "axes", "per_problem.csv")
SWING_PATH = os.path.join(PROJECT_ROOT, "activations", "_signal_cache",
                          "ownership_swing.csv")

# Reserved cell code for the pass variation in the published cell map: a board
# point is y*size + x, which tops out well below this on both shipped boards.
PASS_POINT_CODE = 255

# Problems taken from each end of an axis. The claim this backs is about a
# thousand, and the number is stated here so the figure and the sentence quoting
# it cannot drift apart.
TAIL_N = 1000


# --- the published index ----------------------------------------------------

def _lookup_rows():
    """[(id, zipHash, size)] in published order — the client's own row order, and
    the row order the offsets sidecar is aligned to."""
    require(os.path.exists(LOOKUP_PATH), f"no {LOOKUP_PATH} — run pipeline.package_for_pwa")
    rows = []
    with open(LOOKUP_PATH) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.split()
            require(len(parts) == 3,
                    f"lookup.txt:{lineno}: expected 'id zipHash size', got {line!r}")
            tid, zip_hash, size = parts
            require(len(tid) == 8 and all(c in "0123456789abcdef" for c in tid),
                    f"lookup.txt:{lineno}: id {tid!r} is not 8 hex")
            require(size.isdigit(), f"lookup.txt:{lineno}: size {size!r}")
            rows.append((tid, zip_hash, int(size)))
    require(rows, "lookup.txt is empty")
    ids = [r[0] for r in rows]
    require(len(set(ids)) == len(ids), f"lookup.txt: {len(ids) - len(set(ids))} duplicate ids")
    require({r[2] for r in rows} == set(ALL_BOARD_SIZES),
            f"lookup.txt sizes {sorted({r[2] for r in rows})} != {sorted(ALL_BOARD_SIZES)}")
    return rows


def _never_attempted(ids):
    """Boolean array over `ids`: True where no first-time attempt exists."""
    require(os.path.exists(DB_SQLITE), f"no {DB_SQLITE} — run pipeline.fetch_database")
    con = sqlite3.connect(f"file:{DB_SQLITE}?mode=ro", uri=True)
    try:
        attempted = {r[0] for r in con.execute(
            "SELECT DISTINCT problem FROM tsumego_reports "
            f"WHERE is_repeat = 0 AND {not_ignored_sql()}")}
    finally:
        con.close()
    require(attempted, "tsumego_reports holds no first-time attempts")
    return np.array([tid not in attempted for tid in ids], dtype=bool)


# --- the taught axis specification ------------------------------------------

def _taught_axes():
    """[(axis index, taught tail sign)] for the axes with an UNTAUGHT tail, plus
    the count of two-sided axes that therefore have none.

    A bare rule name teaches the positive tail and leaves the negative untaught;
    an `A/B` specification teaches both, so the never-attempted comparison this
    module makes has no untaught side on such an axis and it is excluded rather
    than assigned an arbitrary orientation."""
    require(os.path.exists(LABELS_PATH), f"no {LABELS_PATH}")
    with open(LABELS_PATH, encoding="utf-8") as f:
        components = json.load(f).get("components")
    require(isinstance(components, list) and components,
            f"{LABELS_PATH}: 'components' is not a non-empty list")
    require(all(isinstance(c, str) and c.strip() for c in components),
            f"{LABELS_PATH}: every component must be a rule name or 'A/B'")

    # The published labels are the same specification after name resolution, so
    # the two must agree on which axes are two-sided. They are written by
    # different steps; a disagreement means the shipped shards and the fit's own
    # taught spec describe different axes.
    require(os.path.exists(META_PATH), f"no {META_PATH} — run pipeline.package_for_pwa")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    published = meta.get("labels", {}).get("components")
    require(isinstance(published, list) and len(published) == len(components),
            f"{META_PATH}: {len(published) if isinstance(published, list) else '?'} published "
            f"components against {len(components)} taught")
    require(int(meta.get("K", -1)) == len(components),
            f"{META_PATH}: K={meta.get('K')} against {len(components)} taught components")
    for k, (spec, pub) in enumerate(zip(components, published)):
        require((pub.get("negative") is not None) == ("/" in spec),
                f"axis {k}: taught spec {spec!r} and published labels disagree on "
                f"whether the negative tail is taught")

    one_sided = [(k, +1) for k, spec in enumerate(components) if "/" not in spec]
    require(one_sided, "every axis teaches both tails — no untaught end to measure")
    return one_sided, len(components) - len(one_sided), len(components)


def _axis_table(ids):
    """The per-problem axis loadings, row-aligned to the published index."""
    require(os.path.exists(PER_PROBLEM_PATH), f"no {PER_PROBLEM_PATH} — run pipeline.train_mirt")
    order = {tid: i for i, tid in enumerate(ids)}
    values = None
    seen = 0
    with open(PER_PROBLEM_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        require(header[:2] == ["tsumego_id", "size"],
                f"{PER_PROBLEM_PATH}: unexpected leading columns {header[:2]}")
        k = len(header) - 2
        require(k > 0, f"{PER_PROBLEM_PATH}: no embedding columns")
        values = np.full((len(ids), k), np.nan, dtype=np.float64)
        for row in reader:
            i = order.get(row[0])
            if i is None:
                continue
            values[i] = [float(x) for x in row[2:]]
            seen += 1
    require(seen == len(ids),
            f"{PER_PROBLEM_PATH} covers {seen} of the {len(ids)} published problems")
    require(np.all(np.isfinite(values)), f"{PER_PROBLEM_PATH}: non-finite loadings")
    return values


# --- the published cell map -------------------------------------------------

def _pass_not_last(rows):
    """{size: (problems, problems whose pass code is not the last cell)}.

    Addressed the way any published-asset reader addresses it: the offsets
    sidecar gives a problem's block offset and cell count, and the cell map runs
    in lockstep at one byte per cell, so the codes start at
    `offset // bytes_per_cell` and run `n_cells` bytes."""
    require(os.path.exists(META_PATH), f"no {META_PATH} — run pipeline.package_for_pwa")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    bytes_per_cell = int(meta["bytes_per_cell"])
    bytes_per_row = int(meta["offsets_bytes_per_row"])
    require(bytes_per_cell > 0 and bytes_per_row == 5,
            f"{META_PATH}: unexpected sidecar layout "
            f"({bytes_per_cell} B/cell, {bytes_per_row} B/offset row)")

    require(os.path.exists(OFFSETS_PATH), f"no {OFFSETS_PATH} — run pipeline.package_for_pwa")
    raw = np.fromfile(OFFSETS_PATH, dtype=np.uint8)
    require(raw.size == len(rows) * bytes_per_row,
            f"lookup.offsets.bin holds {raw.size / bytes_per_row:g} rows against "
            f"lookup.txt's {len(rows)}")
    raw = raw.reshape(len(rows), bytes_per_row)
    offsets = raw[:, :4].copy().view(np.uint32).reshape(len(rows)).astype(np.int64)
    n_cells = raw[:, 4].astype(np.int64)
    absent = int((n_cells == 0).sum())
    require(absent == 0,
            f"lookup.offsets.bin: {absent} problems carry the absent sentinel (n_cells 0)")

    by_shard = {}
    for i, (_, zip_hash, size) in enumerate(rows):
        by_shard.setdefault((size, zip_hash), []).append(i)

    totals = {size: 0 for size in ALL_BOARD_SIZES}
    not_last = {size: 0 for size in ALL_BOARD_SIZES}
    for (size, zip_hash), members in sorted(by_shard.items()):
        path = os.path.join(TSUMEGOS_DIR, str(size), f"{zip_hash}.cells.bin")
        require(os.path.exists(path), f"no {path} — named by lookup.txt")
        codes = np.fromfile(path, dtype=np.uint8)
        span = max(offsets[i] // bytes_per_cell + n_cells[i] for i in members)
        require(codes.size == span,
                f"{size}/{zip_hash}.cells.bin is {codes.size} B against the "
                f"{span} B its blocks address")
        for i in members:
            start = offsets[i] // bytes_per_cell
            block = codes[start:start + n_cells[i]]
            where = np.flatnonzero(block == PASS_POINT_CODE)
            require(where.size == 1,
                    f"{rows[i][0]}: {where.size} pass codes in its {n_cells[i]}-cell run, "
                    f"expected exactly 1")
            totals[size] += 1
            if where[0] != n_cells[i] - 1:
                not_last[size] += 1
    return totals, not_last


# --- the ownership-swing baseline -------------------------------------------

def _swing_pass_baseline(ids):
    """(problems where the pass is the only resolved refutation, problems where a
    resolved pass outranks every resolved stone move).

    The cache records the chosen baseline (`alt_move`, the literal `pass` when it
    is the pass) beside the number of resolved refutations it was chosen from
    (`n_te_alts`, which counts the pass like any other), so both counts read
    straight off it. A problem with no resolved refutation at all carries no
    baseline and belongs to neither count."""
    require(os.path.exists(SWING_PATH),
            f"no {SWING_PATH} — run pipeline.extract_ownership_swing")
    wanted = set(ids)
    only, outranks, no_baseline, seen = 0, 0, 0, set()
    with open(SWING_PATH) as f:
        reader = csv.DictReader(f)
        for column in ("tid", "n_te_alts", "alt_move"):
            require(column in (reader.fieldnames or []),
                    f"{SWING_PATH}: no '{column}' column")
        for row in reader:
            tid = row["tid"]
            if tid not in wanted:
                continue
            seen.add(tid)
            n_alts = int(row["n_te_alts"])
            if n_alts == 0:
                no_baseline += 1
                continue
            if row["alt_move"] != "pass":
                continue
            if n_alts == 1:
                only += 1
            else:
                outranks += 1
    missing = len(wanted) - len(seen)
    require(missing == 0,
            f"{SWING_PATH} is missing {missing} of the {len(wanted)} published problems")
    return only, outranks, no_baseline


# --- the measurement --------------------------------------------------------

def measure():
    rows = _lookup_rows()
    ids = [r[0] for r in rows]
    never = _never_attempted(ids)

    one_sided, two_sided, n_axes = _taught_axes()
    values = _axis_table(ids)
    require(values.shape[1] == n_axes,
            f"{PER_PROBLEM_PATH} has {values.shape[1]} axis columns against "
            f"{n_axes} taught components")
    require(len(ids) >= 2 * TAIL_N,
            f"only {len(ids)} published problems — the {TAIL_N}-problem tails overlap")

    per_axis = []
    for k, taught_sign in one_sided:
        order = np.argsort(values[:, k], kind="stable")
        low, high = order[:TAIL_N], order[-TAIL_N:]
        taught, untaught = (high, low) if taught_sign > 0 else (low, high)
        per_axis.append((k, float(never[taught].mean()), float(never[untaught].mean())))

    totals, not_last = _pass_not_last(rows)
    only, outranks, no_baseline = _swing_pass_baseline(ids)

    return {
        "n": len(ids),
        "never_attempted": int(never.sum()),
        "never_share": float(never.mean()),
        "axes": per_axis,
        "axes_two_sided": two_sided,
        "pass_totals": totals,
        "pass_not_last": not_last,
        "swing_pass_only": only,
        "swing_pass_outranks": outranks,
        "swing_no_baseline": no_baseline,
    }


def figures():
    m = measure()
    taught = [t for _, t, _ in m["axes"]]
    untaught = [u for _, _, u in m["axes"]]
    figs = [
        Figure("corpus.never_attempted_pct", m["never_share"], "%.1f",
               "share of the published problems with no first-time human attempt "
               "in the attempt log"),
        Figure("axistail.axes_measured", len(m["axes"]), "d",
               f"axes with an untaught tail, i.e. taught by a single rule; the "
               f"{m['axes_two_sided']} two-sided axes have no untaught end and are "
               f"excluded"),
        Figure("axistail.untaught_exceeds_taught_axes",
               sum(1 for _, t, u in m["axes"] if u > t), "d",
               f"axes whose UNTAUGHT tail holds the larger never-attempted share of "
               f"the two, over each tail's {TAIL_N} most extreme problems"),
        Figure("axistail.untaught_never_min", min(untaught), "%.0f",
               f"smallest never-attempted share among the axes' untaught tails "
               f"({TAIL_N} most extreme problems each)"),
        Figure("axistail.untaught_never_max", max(untaught), "%.0f",
               f"largest never-attempted share among the axes' untaught tails "
               f"({TAIL_N} most extreme problems each)"),
        Figure("axistail.taught_never_min", min(taught), "%.0f",
               f"smallest never-attempted share among the axes' taught, named tails "
               f"({TAIL_N} most extreme problems each)"),
        Figure("axistail.taught_never_max", max(taught), "%.0f",
               f"largest never-attempted share among the axes' taught, named tails "
               f"({TAIL_N} most extreme problems each)"),
        Figure("swing.pass_only_baseline", m["swing_pass_only"], "d",
               "published problems whose only resolved refutation is the pass, so "
               "excluding it would leave no ownership-swing measurement at all"),
        Figure("swing.pass_outranks_stones_pct",
               m["swing_pass_outranks"] / m["n"], "%.1f",
               "share of the published problems where a resolved pass is the swing "
               "baseline over at least one resolved stone move"),
    ]
    for size in sorted(m["pass_totals"]):
        figs.append(Figure(
            f"corpus.pass_not_last_{size}_pct",
            m["pass_not_last"][size] / m["pass_totals"][size], "%.1f",
            f"share of published {size}×{size} problems whose pass variation is not "
            f"the last one, so the placeholder row it gets in the per-cell block is "
            f"what keeps every later row on its own variation"))
    return figs


def main():
    m = measure()
    print(f"{m['n']:,d} published problems; "
          f"{m['never_attempted']:,d} ({m['never_share'] * 100:.1f}%) never attempted "
          f"(no first-time row in the attempt log)\n")

    print(f"never-attempted share in each axis tail's {TAIL_N} most extreme problems")
    print(f"{'axis':>6}{'taught tail':>14}{'untaught tail':>16}{'':>4}")
    for k, taught, untaught in m["axes"]:
        flag = "" if untaught > taught else "  <- untaught tail is the more played"
        print(f"{k:>6}{taught * 100:>13.1f}%{untaught * 100:>15.1f}%{flag}")
    if m["axes_two_sided"]:
        print(f"{m['axes_two_sided']} two-sided axes excluded (both tails taught)")

    print(f"\npass variation not last in the published cell map "
          f"(the placeholder row's whole purpose)")
    for size in sorted(m["pass_totals"]):
        total, n = m["pass_totals"][size], m["pass_not_last"][size]
        print(f"{str(size) + '×' + str(size):>6}{n:>10,d} of {total:>8,d}"
              f"{n / total * 100:>8.2f}%")

    print(f"\nownership-swing baseline")
    print(f"{'pass is the only resolved refutation':>44}{m['swing_pass_only']:>10,d}")
    print(f"{'pass outranks every resolved stone move':>44}{m['swing_pass_outranks']:>10,d}"
          f"   {m['swing_pass_outranks'] / m['n'] * 100:.2f}% of the published set, "
          f"one in {m['n'] / max(m['swing_pass_outranks'], 1):.0f}")
    print(f"{'no resolved refutation at all':>44}{m['swing_no_baseline']:>10,d}"
          f"   (no swing measurement)")


if __name__ == "__main__":
    main()
