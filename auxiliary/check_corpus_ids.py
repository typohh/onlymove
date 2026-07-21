"""Corpus-wide tsumego id integrity scan.

Recomputes every SGF's orientation-invariant canonical hash (the same
`get_orientation_invariant_hash` the generator names files with, taken at the
problem position = after the single-child setup chain) and reports:

  1. MISNAMED files — filename id != canonical hash truncated to 8 hex. These
     break the generator's filename-existence dedup (`tsumego_id_taken`) for
     their position, which is how the same position can later be re-generated
     under its correct id and ship as a duplicate.
  2. DUPLICATE POSITIONS — groups of files (across both sizes) whose FULL
     canonical hashes are identical: the same position, up to D4 symmetry,
     served under more than one id. Violates the symmetry-aware single-copy
     invariant.
  3. TRUNCATION COLLISIONS — different positions sharing the same 8-hex id
     across sizes (the known grandfathered kind; listed for completeness).

Usage:
    ./.venv/bin/python3 -m auxiliary.check_corpus_ids [--sizes 9 13] [--workers N]
    ./.venv/bin/python3 -m auxiliary.check_corpus_ids --emit-map surgery/id_map.json

Read-only; scans tsumegos/<size>/*.sgf. With --emit-map it additionally writes
the corpus-surgery id map (docs/db_surgery_and_legacy_removal_plan.md section 3)
consumed by auxiliary.corpus_surgery and the DB surgery script: one entry per
file whose id changes or that is deleted, `{size, old_id, new_id, action}` with
action `rename` (misnamed file, or an elected duplicate-group survivor not yet
at the canonical id), `merge` (duplicate-group loser — its file/artifacts are
deleted and DB rows re-point at new_id), or `quarantine` (a C0 truncation
collision: the position's canonical 8-hex id is already owned by a DIFFERENT
position, so the file moves to tsumegos/broken/ instead; new_id null, DB rows
keep the retired id — the removed-problem precedent). Map emission HARD-STOPS
on any parse error or a C0 collision it cannot auto-resolve by quarantine
(one involving a duplicate group).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sgfmill import sgf, boards  # noqa: E402

from pipeline.generate_tsumegos import get_orientation_invariant_hash  # noqa: E402
from utils.configurations import ALL_BOARD_SIZES  # noqa: E402


def problem_hash(path):
    """(tsumego_id, full canonical hash) for one SGF, or (id, None) on parse
    failure. The problem position is after the single-child setup chain (the
    variation root), matching where the generator hashed it."""
    tid = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        board = boards.Board(game.get_size())
        root = game.get_root()
        for color, points in (("b", root.get_setup_stones()[0]),
                              ("w", root.get_setup_stones()[1])):
            for (r, c) in points:
                board.play(r, c, color)
        node = root
        while len(node) == 1:
            node = node[0]
            color, move = node.get_move()
            if move is not None:
                board.play(move[0], move[1], color)
        return tid, get_orientation_invariant_hash(board)
    except Exception as e:  # noqa: BLE001 — report, don't abort the sweep
        return tid, f"PARSE_ERROR:{e}"


def build_id_map(ok, errors):
    """The surgery id map from scan results (docs plan section 3).

    ok: [(size, tid, full_hash)] for every parseable SGF. Returns the map dict.
    Raises SystemExit on parse errors or C0 violations — a map must never be
    emitted from a scan that couldn't establish every file's canonical id.
    """
    if errors:
        raise SystemExit(
            f"--emit-map refused: {len(errors)} SGFs failed to parse; a map "
            "needs every file's canonical hash. Fix or quarantine them first.")

    by_full = defaultdict(list)
    for s, t, h in ok:
        by_full[h].append((s, t))

    entries = []
    for h, members in sorted(by_full.items()):
        canon = h[:8]
        sizes = {s for s, _ in members}
        assert len(sizes) == 1, (
            f"duplicate group spans board sizes (impossible: the hash covers "
            f"the board): {h[:16]} -> {sorted(members)}")
        if len(members) == 1:
            s, t = members[0]
            if t != canon:
                entries.append({"size": s, "old_id": t, "new_id": canon,
                                "action": "rename", "full_hash": h})
            continue
        # Duplicate group: survivor = the member already at the canonical id,
        # else the lexicographically-first old id (deterministic election).
        named = [(s, t) for s, t in members if t == canon]
        assert len(named) <= 1  # same dir ⇒ one file per name
        survivor = named[0] if named else min(members, key=lambda st: st[1])
        if survivor[1] != canon:
            entries.append({"size": survivor[0], "old_id": survivor[1],
                            "new_id": canon, "action": "rename", "full_hash": h})
        for s, t in members:
            if (s, t) == survivor:
                continue
            entries.append({"size": s, "old_id": t, "new_id": canon,
                            "action": "merge", "full_hash": h})

    # C0 — final-state uniqueness: after the whole map is applied, every 8-hex
    # id must belong to exactly one position (full hash) across both sizes.
    # A collision here is the 32-bit birthday effect: two DIFFERENT positions
    # whose full canonical hashes share the first 8 hex chars (they don't
    # collide as filenames today only because one member is misnamed).
    # Resolution (docs plan section 3): QUARANTINE the contender(s) —
    # the member already sitting at the canonical id keeps it; a misnamed
    # single-member group that would rename onto an occupied id is moved to
    # tsumegos/broken/ instead (action `quarantine`, no new id; its DB rows
    # keep the retired id and go rating-less, the removed-problem precedent).
    # If NO member is already at the id, the lexicographically-first claimant
    # keeps its rename and the rest quarantine. Duplicate GROUPS in a
    # collision (survivor + losers) are not auto-resolvable — hard-stop.
    changed = {(e["size"], e["old_id"]): e for e in entries}
    final_claims = defaultdict(list)  # final_id -> [(full_hash, size, old_id, entry|None)]
    for s, t, h in ok:
        e = changed.get((s, t))
        if e and e["action"] == "merge":
            continue  # loser: deleted, occupies no final id
        final_claims[e["new_id"] if e else t].append((h, s, t, e))
    quarantined = []
    for fid, claims in sorted(final_claims.items()):
        if len({h for h, _, _, _ in claims}) <= 1:
            continue
        incumbents = [c for c in claims if c[3] is None]  # already named fid
        assert len(incumbents) <= 1
        contenders = sorted((c for c in claims if c[3] is not None),
                            key=lambda c: c[2])
        losers = contenders if incumbents else contenders[1:]
        for h, s, t, e in losers:
            if any(x["action"] == "merge" and x["full_hash"] == h for x in entries):
                raise SystemExit(
                    f"C0: colliding id {fid} involves a duplicate group "
                    f"({s}/{t}) — not auto-resolvable, decide manually.")
            e["action"] = "quarantine"
            e["new_id"] = None
            quarantined.append((s, t, fid))
            print(f"C0 collision on {fid}: quarantining {s}/{t} "
                  f"(position {h[:16]}...)")

    n_ren = sum(1 for e in entries if e["action"] == "rename")
    n_mrg = sum(1 for e in entries if e["action"] == "merge")
    n_qrt = sum(1 for e in entries if e["action"] == "quarantine")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scanned_files": len(ok),
        "renames": n_ren,
        "merges": n_mrg,
        "quarantines": n_qrt,
        "entries": entries,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--sizes", type=int, nargs="*", default=list(ALL_BOARD_SIZES))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--emit-map", metavar="PATH", default=None,
                    help="write the corpus-surgery id map JSON here (requires a "
                         "clean full-corpus scan: both sizes, zero parse errors)")
    args = ap.parse_args()
    if args.emit_map and sorted(args.sizes) != sorted(ALL_BOARD_SIZES):
        raise SystemExit("--emit-map requires a full scan (all sizes): C0 is "
                         "a cross-size check.")

    paths = []
    for size in args.sizes:
        d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
        with os.scandir(d) as it:
            paths.extend((size, e.path) for e in it
                         if e.name.endswith(".sgf") and e.is_file())
    print(f"Scanning {len(paths)} SGFs with {args.workers} workers ...")

    results = []  # (size, tid, full_hash)
    with Pool(processes=args.workers) as pool:
        for k, ((size, path), (tid, h)) in enumerate(zip(
                paths, pool.imap(problem_hash, (p for _, p in paths),
                                 chunksize=256))):
            results.append((size, tid, h))
            if (k + 1) % 20000 == 0:
                print(f"  {k + 1}/{len(paths)}")

    errors = [(s, t, h) for s, t, h in results if h is None or h.startswith("PARSE_ERROR")]
    ok = [(s, t, h) for s, t, h in results if not (h is None or h.startswith("PARSE_ERROR"))]

    print()
    print(f"parsed {len(ok)} / {len(results)}  (errors: {len(errors)})")
    for s, t, h in errors[:20]:
        print(f"  ERROR {s}/{t}: {h}")

    misnamed = [(s, t, h) for s, t, h in ok if h[:8] != t]
    print()
    print(f"1. MISNAMED (id != canonical hash prefix): {len(misnamed)}")
    for s, t, h in sorted(misnamed):
        print(f"  {s:>2}/{t}.sgf  position hashes to {h[:8]}  (full {h[:16]}...)")

    by_full = defaultdict(list)
    for s, t, h in ok:
        by_full[h].append((s, t))
    dups = {h: v for h, v in by_full.items() if len(v) > 1}
    print()
    print(f"2. DUPLICATE POSITIONS (same full canonical hash, >1 file): {len(dups)}")
    for h, v in sorted(dups.items()):
        print(f"  {h[:16]}...: " + ", ".join(f"{s}/{t}" for s, t in sorted(v)))

    by_tid = defaultdict(list)
    for s, t, h in ok:
        by_tid[t].append((s, h))
    coll = {t: v for t, v in by_tid.items()
            if len(v) > 1 and len({h for _, h in v}) > 1}
    print()
    print(f"3. TRUNCATION COLLISIONS (same id, different positions): {len(coll)}")
    for t, v in sorted(coll.items()):
        print(f"  {t}: " + ", ".join(f"size {s} -> {h[:12]}" for s, h in sorted(v)))

    if args.emit_map:
        id_map = build_id_map(ok, errors)
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_map)), exist_ok=True)
        with open(args.emit_map, "w") as f:
            json.dump(id_map, f, indent=1)
        print()
        print(f"id map -> {args.emit_map}: {id_map['renames']} renames, "
              f"{id_map['merges']} merges (of {id_map['scanned_files']} files)")


if __name__ == "__main__":
    main()
