#!/usr/bin/env python3
"""Recover the attempt-id CUTOFF a recorded holdout hash was drawn from.

Every held-out set in `auxiliary/rating_dev/holdout_set.py` is a seeded draw over
a universe read out of `db/d1.sqlite`, and the short hash printed beside it is a
content hash of the draw. So the hash identifies the DB the run assumed, and a
`pipeline.fetch_database` moves it: the universe grows, the draw moves, and a
dump trained last week can no longer be scored against its own holdout. Keeping a
pinned copy of the DB (`db/d1.baseline-<hash>.sqlite`) is one answer and it does
not scale — one file per recorded result.

The other answer is this script, and it rests on one property: **fetch_database
appends rows strictly above the local MAX(id) and never rewrites or deletes
below it**, so an id PREFIX of `tsumego_reports` is not an arbitrary subset but
an exact earlier state of the table. Recovering an earlier snapshot is therefore
recovering a single integer, and `--max-attempt-id` on `holdout_set`,
`eval_user_holdout` and `train_joint_mirt` replays it.

WHAT THE TWO HASHES COVER. Both universes are read through
`utils.rating.features.load_first_time_attempts` (`is_repeat = 0`, deduplicated
to the earliest attempt per `(user_id, problem)`) and then restricted to
`holdout_set.SIZES`:

  problems  sorted set of (size, tid) with >=1 first-time attempt; the hash is
            over the DRAWN subset, `f"{size}:{tid}\\n"` per key, sha256[:16].
            Printed by `holdout_set --kind problem` and by the trainer's
            `--holdout-problem-frac`.
  users     sorted set of user_id over the same rows; hash `f"{user}\\n"` per
            key. Printed by `eval_user_holdout` as `seed S: hash H` and by the
            trainer as `HELD-OUT USERS: hash H`.
  rows      sorted set of (user_id, problem) — the sample holdout.

THE DEDUP COMMUTES WITH THE CUTOFF, for these hashes. The dedup chooses WHICH of
a pair's rows survives; it never changes whether the pair, its user or its
problem is present at all. Every key above is a set membership, so it is
invariant to that choice and each universe is monotone in the cutoff — which is
what makes the descending walk below well posed. The choice is NOT irrelevant to
what the surviving row SAYS: backfill inserts historical event times at the id
frontier, so a row above the cutoff can carry an earlier created_at than one
below it, and a cutoff applied AFTER the dedup would drop such a pair entirely
instead of falling back to the row the snapshot really held. That is why
`load_first_time_attempts` caps inside the query. `--verify` re-derives the
universe at the found cutoff through the real loader, dedup included, and
compares — so the commutation is measured on this DB rather than assumed.

THE WALK. Candidate cutoffs descend from MAX(id). Per-key attempt counts are
carried incrementally, one decrement per row, so a step costs O(keys the row
touched); the universe changes only when a count reaches zero, and only there is
a hash recomputed. Between two such change points every cutoff yields the same
universe and therefore the same hash, so a match is reported as the PLATEAU
[lo, hi] of cutoffs that reproduce it — a coarser answer than the true MAX(id),
and the honest one: nothing in the hash distinguishes them. Pick `hi` unless a
second recorded quantity (a row count, say) narrows it.

  ./.venv/bin/python3 -m auxiliary.rating_dev.reconstruct_snapshot \\
      --target-hash 6619fdc487345fc0 --hash-kind problems:0:0.15 --verify
  ./.venv/bin/python3 -m auxiliary.rating_dev.reconstruct_snapshot \\
      --target-hash 27ada6e897734b01 --hash-kind users:0:0.15 --verify
"""

import argparse
import bisect
import os
import sqlite3
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from auxiliary.rating_dev.holdout_set import (  # noqa: E402
    SIZES, all_rows, all_users, attempted_keys, held_out_keys, held_out_rows,
    held_out_users)
from utils.rating.features import SQLITE_PATH  # noqa: E402

# One entry per hash the codebase prints. `key` builds the universe member from a
# raw row, `draw` is holdout_set's OWN generator (passed an explicit universe, so
# no second copy of the draw or the hash lives here), `universe` is its DB reader
# for --verify, and `label` names what a member is.
KINDS = {
    "problems": {
        "cols": "size, problem",
        "key": lambda r: (int(r[0]), str(r[1])),
        "draw": held_out_keys,
        "universe": attempted_keys,
        "label": "attempted problems (size, tid)",
    },
    "users": {
        "cols": "user_id",
        "key": lambda r: str(r[0]),
        "draw": held_out_users,
        "universe": all_users,
        "label": "user ids",
    },
    "rows": {
        "cols": "user_id, problem",
        "key": lambda r: (str(r[0]), str(r[1])),
        "draw": held_out_rows,
        "universe": all_rows,
        "label": "first-time (user, problem) pairs",
    },
}


def parse_kind(spec):
    """'users:0:0.15' -> ('users', 0, 0.15). Seed and frac default to the values
    every recorded hash in docs/mirt_benchmark.md was drawn at."""
    parts = spec.split(":")
    name = parts[0]
    if name not in KINDS:
        raise SystemExit(f"--hash-kind must start with one of {sorted(KINDS)}")
    seed = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    frac = float(parts[2]) if len(parts) > 2 and parts[2] else 0.15
    return name, seed, frac


def load_rows(db, kind):
    """(id, key) for every first-time attempt in SIZES, ascending by id.

    Read RAW — no dedup — because every universe here is a set of keys and the
    dedup only decides which of a pair's rows survives, never which keys exist.
    `--verify` checks that against the real loader rather than trusting it."""
    spec = KINDS[kind]
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.execute(
            f"SELECT id, {spec['cols']} FROM tsumego_reports WHERE is_repeat = 0 "
            f"AND size IN ({','.join(str(s) for s in SIZES)}) ORDER BY id")
        out = [(int(r[0]), spec["key"](r[1:])) for r in cur]
    finally:
        con.close()
    return out


def walk(rows, kind, seed, frac, min_id=None, progress_every=2000):
    """Descend the cutoffs, yielding (cutoff, n_universe, hash) once per DISTINCT
    universe, largest cutoff first.

    The universe is a Python list kept sorted by construction (bisect + pop), so
    holdout_set's draw receives it in the order it expects and the incremental
    maintenance costs a memmove rather than a re-sort. Counts are decremented one
    row at a time; a universe (and a hash) is only produced when a count reaches
    zero, which is the only way the key set can move."""
    draw = KINDS[kind]["draw"]
    counts = {}
    for _id, k in rows:
        counts[k] = counts.get(k, 0) + 1
    universe = sorted(counts)
    n_changes, t0 = 0, time.time()
    yield rows[-1][0], len(universe), draw(seed, frac, universe)[1]
    for i in range(len(rows) - 1, -1, -1):
        rid, k = rows[i]
        if min_id is not None and rid - 1 < min_id:
            return
        c = counts[k] - 1
        counts[k] = c
        if c:
            continue                      # the key survives: same universe, same hash
        del counts[k]
        universe.pop(bisect.bisect_left(universe, k))
        n_changes += 1
        h = draw(seed, frac, universe)[1]
        if progress_every and n_changes % progress_every == 0:
            print(f"    ... {n_changes} universes tried, at cutoff {rid - 1} "
                  f"({len(universe)} keys, {time.time() - t0:.0f}s)", flush=True)
        yield rid - 1, len(universe), h


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-hash", required=True,
                    help="the 16-hex hash a run printed")
    ap.add_argument("--hash-kind", default="users:0:0.15",
                    help="problems|users|rows, optionally :SEED:FRAC "
                         "(default seed 0, frac 0.15)")
    ap.add_argument("--db", default=SQLITE_PATH,
                    help="attempts DB to search (default: the canonical one)")
    ap.add_argument("--min-id", type=int, default=None,
                    help="stop the descent here instead of walking to the "
                         "bottom of the table")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive the universe at the found cutoff through "
                         "load_first_time_attempts itself — dedup included — and "
                         "confirm the hash. This is the dedup-commutation check.")
    ap.add_argument("--nearest", type=int, default=5,
                    help="how many candidate cutoffs to print at each end when "
                         "nothing matches")
    args = ap.parse_args()

    kind, seed, frac = parse_kind(args.hash_kind)
    target = args.target_hash.strip().lower()
    print(f"target {target} | kind {kind} (seed {seed}, frac {frac:g}) | "
          f"universe = {KINDS[kind]['label']} | db {args.db}")

    t0 = time.time()
    rows = load_rows(args.db, kind)
    if not rows:
        raise SystemExit("no first-time attempts in that DB")
    print(f"  {len(rows)} first-time attempt rows in sizes {SIZES}, "
          f"id {rows[0][0]}..{rows[-1][0]} ({time.time() - t0:.0f}s)", flush=True)

    gen = walk(rows, kind, seed, frac, args.min_id)
    seen, hit = [], None
    for cutoff, n_uni, h in gen:
        seen.append((cutoff, n_uni, h))
        if h == target:
            hit = (cutoff, n_uni, h)
            break

    if hit is None:
        print(f"\nNOT REACHABLE: {len(seen)} distinct universes tried down to "
              f"cutoff {seen[-1][0]}, none hashing to {target}.")
        print("  cutoff        keys  hash")
        head = seen if len(seen) <= 2 * args.nearest else seen[:args.nearest]
        for c, n, h in head:
            print(f"  {c:>9}  {n:>9}  {h}")
        if len(seen) > 2 * args.nearest:
            print(f"  {'...':>9}  {'...':>9}  ({len(seen) - 2 * args.nearest} more)")
            for c, n, h in seen[-args.nearest:]:
                print(f"  {c:>9}  {n:>9}  {h}")
        print("\n  A miss means the recorded hash was not drawn from an id "
              "prefix of THIS table — a different seed or fraction, a different "
              "kind, or a DB that is not this one's ancestor.")
        sys.exit(1)

    hi, n_uni, _ = hit
    # The plateau's lower end is the NEXT change point, one more pull on the same
    # generator: every cutoff between the two yields this universe, so the hash
    # cannot tell them apart and reporting only `hi` would overstate the answer.
    lo = None
    for cutoff, _n, _h in gen:
        lo = cutoff + 1
        break
    n_rows = bisect.bisect_right([r[0] for r in rows], hi)
    print(f"\nMATCH at cutoff id {hi}")
    print(f"  universe            {n_uni} {KINDS[kind]['label']}")
    print(f"  first-time rows     {n_rows} at id <= {hi} "
          f"(of {len(rows)} in the full table)")
    print(f"  universes tried     {len(seen)} (descending from {seen[0][0]})")
    if lo is not None:
        print(f"  plateau             every cutoff in [{lo}, {hi}] gives this "
              f"same universe; {hi} is the largest")
    print(f"\n  replay:  --max-attempt-id {hi}")

    if args.verify:
        t1 = time.time()
        uni = KINDS[kind]["universe"](hi)
        h2 = KINDS[kind]["draw"](seed, frac, uni)[1]
        raw = {k for _i, k in rows if _i <= hi}
        ok = (h2 == target)
        print(f"\n  VERIFY via load_first_time_attempts(max_id={hi}) "
              f"({time.time() - t1:.0f}s)")
        print(f"    deduped universe  {len(uni)} keys, hash {h2} "
              f"{'MATCH' if ok else 'MISMATCH'}")
        print(f"    raw-row universe  {len(raw)} keys — dedup commutes with the "
              f"cutoff: {'yes' if set(uni) == raw else 'NO'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
