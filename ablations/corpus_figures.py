#!/usr/bin/env python3
"""The published corpus, counted from what ships.

Reads `pwa/lookup.txt` — the row-per-problem index `package_for_pwa` writes, which
is what the client resolves a problem through — and counts it whole and per
board size. Reading the published index rather than `tsumegos/` means the figure
is the set a player can actually be served, not the set on disk, which also holds
`tsumegos/new/` and problems retired since the last packaging.

Everything about the index that the client relies on is checked on the way
through, so a run is also a check on the packaging: every row is `id zipHash size`
with an 8-hex id, ids are unique, the sizes are exactly the shipped board sizes,
every shard the index points at exists, and `problem_params.bin` — the second
index of the same set — holds exactly the same ids.

    ./.venv/bin/python3 -m ablations.corpus_figures
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
from utils.constants import ALL_BOARD_SIZES  # noqa: E402
from utils.rating.ratings_io import read_problem_params  # noqa: E402

TIER = "about"

PWA_DIR = os.path.join(PROJECT_ROOT, "pwa")
LOOKUP_PATH = os.path.join(PWA_DIR, "lookup.txt")
PARAMS_PATH = os.path.join(PWA_DIR, "tsumegos", "problem_params.bin")


def measure():
    require(os.path.exists(LOOKUP_PATH), f"no {LOOKUP_PATH} — run pipeline.package_for_pwa")
    ids, by_size, shards = [], {}, set()
    with open(LOOKUP_PATH) as f:
        for lineno, line in enumerate(f, 1):
            parts = line.split()
            require(len(parts) == 3, f"lookup.txt:{lineno}: expected 'id zipHash size', got {line!r}")
            tid, zip_hash, size = parts
            require(len(tid) == 8 and all(c in "0123456789abcdef" for c in tid),
                    f"lookup.txt:{lineno}: id {tid!r} is not 8 hex")
            require(size.isdigit(), f"lookup.txt:{lineno}: size {size!r}")
            ids.append(tid)
            by_size[int(size)] = by_size.get(int(size), 0) + 1
            shards.add((int(size), zip_hash))
    require(ids, "lookup.txt is empty")
    require(len(set(ids)) == len(ids), f"lookup.txt: {len(ids) - len(set(ids))} duplicate ids")
    require(set(by_size) == set(ALL_BOARD_SIZES),
            f"lookup.txt sizes {sorted(by_size)} != shipped sizes {sorted(ALL_BOARD_SIZES)}")
    missing = [f"{s}/{h}.zip" for s, h in sorted(shards)
               if not os.path.exists(os.path.join(PWA_DIR, "tsumegos", str(s), f"{h}.zip"))]
    require(not missing, f"{len(missing)} shard(s) named in lookup.txt do not exist: {missing[:5]}")

    params = read_problem_params(PARAMS_PATH)
    require(set(params["id"]) == set(ids),
            f"problem_params.bin ids ({len(params['id'])}) != lookup.txt ids ({len(ids)})")
    return {"total": len(ids), "by_size": by_size, "shards": len(shards)}


def figures():
    m = measure()
    figs = [Figure("corpus.problems_total", m["total"], "~1000",
                   "problems in the published index, to the nearest thousand")]
    for size in sorted(m["by_size"]):
        figs.append(Figure(f"corpus.problems_{size}", m["by_size"][size], "~1000",
                           f"published {size}×{size} problems, to the nearest thousand"))
    # The truncated-id arithmetic at the corpus's current size: ids are the leading
    # 32 bits of the canonical hash, so distinct positions collide at the birthday
    # rate. Pure functions of the count, emitted so the sentences quoting them
    # move with it.
    n = m["total"]
    figs.append(Figure("corpus.expected_id_collisions", n * (n - 1) / 2**33, ".0f",
                       "expected accidental id collisions among the published problems, "
                       "n(n-1)/2^33 — each one a silently discarded candidate, "
                       "never a corrupted record"))
    figs.append(Figure("corpus.id_rejection_odds", 2**32 / n, "~1000",
                       "one-in-N odds that a fresh candidate's id is already taken, 2^32/n"))
    return figs


def main():
    m = measure()
    print(f"{'board':>8}{'problems':>12}")
    for size in sorted(m["by_size"]):
        print(f"{size:>5}×{size:<2}{m['by_size'][size]:>12,d}")
    print(f"{'total':>8}{m['total']:>12,d}   in {m['shards']} shards; "
          f"problem_params.bin holds the same ids")


if __name__ == "__main__":
    main()
