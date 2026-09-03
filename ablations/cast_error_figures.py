#!/usr/bin/env python3
"""What the half-precision cast costs the published per-cell motif vectors, and
how that compares against the finest distinction anything reads off them.

Everything the client reads a picked move's motif vector from is
`pwa/tsumegos/<size>/<hash>.embeddings.bin`, half-precision, K floats per cell.
The store those rows are copied from (`axes/per_cell/`) is ALREADY half
precision — `write_mirt_per_cell` casts on the GPU at write time — so packaging
performs no arithmetic and the cast is a property of the projection step, not of
publishing. That means the error cannot be measured by differencing two files:
the full-precision value is not on disk anywhere.

What IS exactly computable is the bound. A half-precision value is the
nearest representable number to whatever the projection produced, so the error is
at most half an interval between neighbours — 2^(E-26) for a value whose stored
exponent field is E, exactly, with no sampling and no re-projection. The two
figures below are that bound over the WHOLE published set, read off the exponent
bits directly:

  cast.worst_sd     the largest half-interval over every non-zero published
                    per-cell value.
  cast.p999_sd      the same at the 99.9th percentile of those values, which is
                    what says the bulk of the set sits an order of magnitude
                    finer than the worst case.

Both are in the axis's own unit, its corpus root-mean-square, which costs no
conversion: the axis gauge divides every loading by that RMS, so a loading is
already in those units. The unit is NOT a standard deviation and must not be
called one: nothing in the objective pins an axis's mean over problems, and the
axis of a one-sided rule holds most of the corpus below zero with its
rule-positive end above, so its spread about its mean can sit well under one
while its RMS is exactly one. The RMS is what is checked against
`axes/per_problem.csv` rather than assumed; the mean is reported beside it.

The comparison that makes the bound mean something is the within-problem one,
because both client readings compare two cells of the SAME problem and a
component constant across a problem's cells cancels in each. The sign test's live
comparison is the cell a player picked against the answer's cell, so the proxy
here is the answer's row against each OTHER variation row of the same problem,
per axis, over a seeded problem sample:

  cast.contrast_median_sd       median |answer - alternative| over those pairs.
  cast.contrast_finer_one_in    one in how many of them is finer than
                                cast.worst_sd, i.e. how often the cast could
                                decide the sign. A flip there is a coin toss
                                either way rather than a bias.

The answer is row 0 by construction — generation pins the verified solution as
the first root variation and packaging preserves variation order — which is the
same reference `emit_alt_stats` uses and asserts.

The read doubles as a check on the published ordering contract. For every
sampled problem the block is matched back to the store BY BOARD COORDINATE
through the shard's `.cells.bin` codes (never by position: the store is in
board-scan order and the block is in move-tree order), and the two must agree
bit for bit, with exactly one all-zero placeholder row for the pass and the
answer not on it.

    ./.venv/bin/python3 -m ablations.cast_error_figures
"""

import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from ablations._figures import Figure, require  # noqa: E402
# The per-cell store lives at axes/per_cell/ or at the sibling a refit renamed
# the previous one to; resolved through the one helper that knows both.
from ablations.percell_variance import resolve_store  # noqa: E402
from utils.rating.model_constants import K_AXES, SIZES  # noqa: E402

TIER = "methodology"

PWA_DIR = os.path.join(PROJECT_ROOT, "pwa")
TSUMEGOS_DIR = os.path.join(PWA_DIR, "tsumegos")
LOOKUP_PATH = os.path.join(PWA_DIR, "lookup.txt")
META_PATH = os.path.join(TSUMEGOS_DIR, "embedding_meta.json")
OFFSETS_PATH = os.path.join(TSUMEGOS_DIR, "lookup.offsets.bin")
PER_PROBLEM = os.path.join(PROJECT_ROOT, "axes", "per_problem.csv")

OFFSETS_BYTES_PER_ROW = 5      # <u4 offset + u1 n_cells, package_for_pwa's layout
PASS_POINT_CODE = 255          # the reserved code no board point can carry

# Problems sampled per board size for the within-problem contrast. Sized for
# stability rather than for coverage: at this count the median moves by ~0.003
# and the one-in-N by under a percent across seeds, which is well inside the
# quoted digits. The cast bound itself is not sampled at all.
N_PER_SIZE = 10000
SEED = 0


def load_meta():
    require(os.path.exists(META_PATH), f"no {META_PATH} — run pipeline.package_for_pwa")
    with open(META_PATH) as f:
        meta = json.load(f)
    require(int(meta["K"]) == K_AXES,
            f"embedding_meta.json ships K={meta['K']}, the fit ships K_AXES={K_AXES}")
    require(meta["dtype"] == "float16",
            f"the per-cell binaries are {meta['dtype']}, not float16 — this "
            f"measurement is about the half-precision cast")
    require(int(meta["bytes_per_cell"]) == 2 * K_AXES,
            f"bytes_per_cell {meta['bytes_per_cell']} is not 2 x K_AXES")
    return meta


def load_index():
    """lookup.txt joined to lookup.offsets.bin: per published problem, its shard
    and the address of its per-cell block inside that shard's binary."""
    require(os.path.exists(LOOKUP_PATH), f"no {LOOKUP_PATH}")
    rows = []
    with open(LOOKUP_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tid, zip_hash, size = line.split()
                rows.append((tid, zip_hash, int(size)))
    require(rows, "lookup.txt is empty")
    bad = sorted({s for _, _, s in rows} - set(SIZES))
    require(not bad, f"lookup.txt carries board sizes {bad} the fit does not ship")

    require(os.path.exists(OFFSETS_PATH), f"no {OFFSETS_PATH}")
    raw = np.fromfile(OFFSETS_PATH, dtype=np.uint8)
    require(raw.size == len(rows) * OFFSETS_BYTES_PER_ROW,
            f"lookup.offsets.bin holds {raw.size} bytes against {len(rows)} "
            f"lookup rows x {OFFSETS_BYTES_PER_ROW} — the sidecar is not aligned "
            f"to this lookup.txt")
    recs = raw.reshape(-1, OFFSETS_BYTES_PER_ROW)
    offsets = recs[:, :4].copy().view("<u4").ravel().astype(np.int64)
    n_cells = recs[:, 4].astype(np.int64)
    require(np.all(n_cells > 0),
            f"{int((n_cells == 0).sum())} published problems have no per-cell "
            f"block — the offsets sidecar was written without the shard bins")
    return rows, offsets, n_cells


def cast_bound(rows, offsets, n_cells, bytes_per_cell):
    """The exact half-precision rounding bound over every published per-cell
    value, by exponent field.

    A half-precision number with stored exponent field E >= 1 sits in the binade
    [2^(E-15), 2^(E-14)), whose neighbours are 2^(E-25) apart, so the rounding
    that produced it moved the value by at most 2^(E-26). E == 0 is the subnormal
    binade, spaced 2^-24, hence 2^-25. Reading the field rather than the value
    keeps this exact arithmetic on integers.

    Only the shards `lookup.txt` names are read: a directory can hold the bins of
    shards a later run replaced, and those bytes ship to nobody.

    Exact zeros are excluded and counted. Every problem carries one all-zero
    placeholder row for the pass, which is structurally unreachable and never
    read; zero is also exactly representable, so it carries no cast error at all
    and would only deflate the percentile."""
    shards = sorted({(size, zip_hash) for _, zip_hash, size in rows})
    counts = np.zeros(32, np.int64)
    n_zero = n_total = n_bytes = 0
    for size, zip_hash in shards:
        path = os.path.join(TSUMEGOS_DIR, str(size), f"{zip_hash}.embeddings.bin")
        require(os.path.exists(path), f"lookup.txt names shard {zip_hash} but {path} is missing")
        bits = np.fromfile(path, dtype="<u2")
        n_bytes += bits.size * 2
        nz = (bits & 0x7FFF) != 0
        n_zero += int((~nz).sum())
        n_total += bits.size
        counts += np.bincount((bits[nz] >> 10) & 0x1F, minlength=32)
    expected = int((n_cells * bytes_per_cell).sum())
    require(n_bytes == expected,
            f"the shard binaries hold {n_bytes} bytes against {expected} the "
            f"offsets sidecar accounts for — the two disagree about the "
            f"published set")
    # One all-zero row per problem for the pass; anything beyond that is a real
    # cell whose value happened to land on zero, which is rare and reported.
    require(n_zero >= K_AXES * len(rows),
            f"{n_zero} zero values against {K_AXES * len(rows)} pass placeholders "
            f"— a problem is missing its placeholder row")
    nz_total = counts.sum()
    require(nz_total > 0, "every published per-cell value is zero")
    present = np.flatnonzero(counts)
    half_ulp = np.where(np.arange(32) == 0, 2.0 ** -25, 2.0 ** (np.arange(32) - 26.0))
    cum = np.cumsum(counts)
    p999 = int(np.searchsorted(cum, 0.999 * nz_total))
    return {
        "worst": float(half_ulp[present.max()]),
        "p999": float(half_ulp[p999]),
        "median": float(half_ulp[int(np.searchsorted(cum, 0.5 * nz_total))]),
        "n_values": int(n_total),
        "n_nonzero": int(nz_total),
        "n_zero": int(n_zero),
        "n_zero_excess": int(n_zero - K_AXES * len(rows)),
        "n_shards": len(shards),
        "n_bytes": n_bytes,
    }


def check_axis_units():
    """The published values are quoted in axis units, and that is a claim about
    the gauge rather than a rescaling: `problem_params` divides every loading by
    its own axis's corpus RMS, so an axis's root-mean-square over the corpus is
    one by construction. Verified here so the unit cannot silently become
    something else. The mean is reported, not asserted: a one-sided rule's axis
    is legitimately off-centre, and the gauge never claimed otherwise."""
    import pandas as pd
    require(os.path.exists(PER_PROBLEM), f"no {PER_PROBLEM} — run pipeline.train_mirt")
    per = pd.read_csv(PER_PROBLEM)
    cols = [f"embed_{k}" for k in range(K_AXES)]
    missing = [c for c in cols if c not in per.columns]
    require(not missing, f"{PER_PROBLEM} lacks {missing}")
    G = per[cols].to_numpy(np.float64)
    rms = np.sqrt((G ** 2).mean(axis=0))
    require(np.all(np.abs(rms - 1.0) < 0.01),
            f"the loadings are not gauged: per-axis RMS {rms.round(4)} is not one")
    return {"rms": rms, "mean": G.mean(axis=0), "sd": G.std(axis=0), "n": len(G)}


def sample_rows(rows, n_per_size, seed):
    """The seeded problem draw: one generator advanced across sizes in SIZES
    order, so the sample is a function of (seed, n_per_size, published set)."""
    by_size = {}
    for i, (_, _, size) in enumerate(rows):
        by_size.setdefault(size, []).append(i)
    rng = np.random.default_rng(seed)
    out = []
    for size in SIZES:
        idx = np.asarray(by_size.get(size, []), dtype=np.int64)
        require(len(idx) > 0, f"no published problems at size {size}")
        n = min(n_per_size, len(idx))
        out.extend(int(i) for i in np.sort(rng.choice(idx, size=n, replace=False)))
    return out


def contrast(rows, offsets, n_cells, sample, bytes_per_cell, store):
    """|answer - alternative| per axis over each sampled problem's own cells,
    plus the published-vs-store agreement the read is checked against.

    The store is matched by BOARD COORDINATE through the shard's cells sidecar,
    which is the whole point of that sidecar: the store is in board-scan order
    and the published block is in move-tree order, so a positional match would
    line up two different cells and still look well-formed."""
    diffs = []
    n_pairs = n_checked = 0
    by_shard = {}
    for i in sample:
        _, zip_hash, size = rows[i]
        by_shard.setdefault((size, zip_hash), []).append(i)
    for (size, zip_hash), items in sorted(by_shard.items()):
        emb_path = os.path.join(TSUMEGOS_DIR, str(size), f"{zip_hash}.embeddings.bin")
        cells_path = os.path.join(TSUMEGOS_DIR, str(size), f"{zip_hash}.cells.bin")
        require(os.path.exists(cells_path), f"no {cells_path} beside its embeddings bin")
        with open(emb_path, "rb") as fe, open(cells_path, "rb") as fc:
            for i in items:
                tid = rows[i][0]
                off, n = int(offsets[i]), int(n_cells[i])
                fe.seek(off)
                raw = fe.read(n * bytes_per_cell)
                require(len(raw) == n * bytes_per_cell,
                        f"{tid}: block runs past the end of {emb_path}")
                block = np.frombuffer(raw, dtype="<f2").reshape(n, K_AXES)
                # The codes need no addressing of their own: one byte per cell
                # against bytes_per_cell of vector, so the block's byte offset
                # divides straight through.
                require(off % bytes_per_cell == 0,
                        f"{tid}: block offset {off} is not a whole number of cells")
                fc.seek(off // bytes_per_cell)
                codes = np.frombuffer(fc.read(n), dtype=np.uint8)
                require(len(codes) == n, f"{tid}: cell map runs past the end of {cells_path}")

                is_pass = codes == PASS_POINT_CODE
                require(int(is_pass.sum()) == 1,
                        f"{tid}: {int(is_pass.sum())} pass rows, expected exactly 1")
                zero_rows = (block == 0).all(axis=1)
                require(bool(zero_rows[is_pass][0]) and int(zero_rows.sum()) == 1,
                        f"{tid}: the all-zero placeholder is not the pass row")
                require(not bool(is_pass[0]),
                        f"{tid}: row 0 is the pass placeholder, so the answer is "
                        f"not first among the root variations")

                real = block[~is_pass].astype(np.float64)
                answer = real[0]
                if len(real) > 1:
                    diffs.append(np.abs(real[1:] - answer))
                    n_pairs += len(real) - 1

                npz = os.path.join(store, str(size), tid + ".npz")
                require(os.path.exists(npz), f"no per-cell store entry {npz}")
                with np.load(npz, allow_pickle=False) as z:
                    emb, xys = z["embeddings"], z["xys"]
                xy_to_row = {(int(x), int(y)): r for r, (x, y) in enumerate(xys)}
                want = [xy_to_row.get((int(c) % size, int(c) // size))
                        for c in codes[~is_pass]]
                require(all(r is not None for r in want),
                        f"{tid}: the published block names a cell the store does not hold")
                require(np.array_equal(emb[np.asarray(want, dtype=np.intp)], block[~is_pass]),
                        f"{tid}: the published block and the per-cell store "
                        f"disagree once matched by board coordinate")
                n_checked += 1

    require(diffs, "no sampled problem carried an alternative to compare")
    return np.concatenate(diffs, axis=0), n_pairs, n_checked


def measure():
    meta = load_meta()
    bytes_per_cell = int(meta["bytes_per_cell"])
    units = check_axis_units()
    rows, offsets, n_cells = load_index()
    bound = cast_bound(rows, offsets, n_cells, bytes_per_cell)

    store = resolve_store()
    require(store is not None,
            "no per-cell store under axes/per_cell* — the published blocks "
            "cannot be checked against what was projected")
    sample = sample_rows(rows, N_PER_SIZE, SEED)
    D, n_pairs, n_checked = contrast(rows, offsets, n_cells, sample,
                                     bytes_per_cell, store)
    flat = D.ravel()
    finer = float((flat < bound["worst"]).mean())
    require(finer > 0.0, "no within-problem comparison is finer than the cast "
                         "bound, so the reciprocal below is undefined")
    return {
        "meta": meta, "units": units, "bound": bound,
        "store": os.path.relpath(store, PROJECT_ROOT),
        "n_problems": len(rows), "n_sampled": len(sample), "n_checked": n_checked,
        "n_pairs": n_pairs, "n_comparisons": int(flat.size),
        "median": float(np.median(flat)),
        "per_axis_median": np.median(D, axis=0),
        "per_axis_finer": (D < bound["worst"]).mean(axis=0),
        "finer_share": finer,
        "finer_one_in": 1.0 / finer,
        "exact_ties": float((flat == 0.0).mean()),
    }


def figures():
    m = measure()
    return [
        Figure("cast.worst_sd", m["bound"]["worst"], ".4f",
               "largest half-precision rounding error the published per-cell "
               "motif vectors can carry, in axis units (the loadings are gauged "
               "to unit corpus RMS, so a raw unit is one)"),
        Figure("cast.p999_sd", m["bound"]["p999"], ".4f",
               "the same bound at the 99.9th percentile of published per-cell "
               "values"),
        Figure("cast.contrast_median_sd", m["median"], ".2f",
               "median |answer cell - alternative cell| per axis, over the "
               "variation cells of a seeded sample of published problems — the "
               "within-problem distinction the tendency sign test draws"),
        Figure("cast.contrast_finer_one_in", m["finer_one_in"], "~10",
               "one in how many of those within-problem comparisons is finer "
               "than the worst-case cast error"),
    ]


def main():
    m = measure()
    b, u = m["bound"], m["units"]
    print(f"\nhalf-precision cast error of the published per-cell motif vectors, "
          f"against the\nwithin-problem distinctions read off them. "
          f"{m['n_problems']:,d} published problems in "
          f"{b['n_shards']} shards,\n{b['n_bytes'] / 1e6:.0f} MB of "
          f"{m['meta']['dtype']} at {m['meta']['bytes_per_cell']} bytes per cell, "
          f"K={m['meta']['K']}.")
    print(f"\nunits: a loading is divided by its own axis's corpus RMS, so one "
          f"raw unit IS one axis RMS — per-axis RMS "
          f"{u['rms'].min():.4f}..{u['rms'].max():.4f} over {u['n']:,d} "
          f"problems; the axes are not centred (mean "
          f"{u['mean'].min():+.3f}..{u['mean'].max():+.3f}, sd "
          f"{u['sd'].min():.3f}..{u['sd'].max():.3f}), which the unit does "
          f"not require.")

    print("\n=== the cast bound, exact over every published value ===")
    print("half the spacing between neighbouring half-precision numbers at each "
          "value's own\nmagnitude, read off the exponent field. Not a sample and "
          "not a re-projection:\nthe full-precision value is on no disk, and this "
          "bounds the error without it.")
    print(f"{'values published':>28}{b['n_values']:>14,d}")
    print(f"{'exact zeros (pass rows)':>28}{b['n_zero']:>14,d}"
          f"   {b['n_zero_excess']:+,d} against one placeholder row per problem")
    print(f"{'worst case':>28}{b['worst']:>14.6f}   axis sd")
    print(f"{'99.9th percentile':>28}{b['p999']:>14.6f}")
    print(f"{'median':>28}{b['median']:>14.6f}")

    print(f"\n=== the within-problem comparison, {m['n_sampled']:,d} problems "
          f"sampled (seed {SEED}) ===")
    print("|answer cell - alternative cell| per axis, the answer being row 0 of "
          "the problem's\nown block. Every sampled block is checked against the "
          "per-cell store\n"
          f"({m['store']}) by BOARD COORDINATE through the shard's cell map.")
    print(f"{'blocks checked':>28}{m['n_checked']:>14,d}")
    print(f"{'cell pairs':>28}{m['n_pairs']:>14,d}")
    print(f"{'comparisons (pairs x axes)':>28}{m['n_comparisons']:>14,d}")
    print(f"{'median |difference|':>28}{m['median']:>14.4f}   axis sd")
    print(f"{'finer than the worst case':>28}{m['finer_share']:>13.3%}   "
          f"one in {m['finer_one_in']:.0f}")
    print(f"{'exactly equal':>28}{m['exact_ties']:>13.3%}")
    print(f"\n  per axis:")
    print(f"  {'axis':<6}{'median |diff|':>15}{'finer than worst':>19}")
    for k in range(K_AXES):
        print(f"  {k:<6}{m['per_axis_median'][k]:>15.4f}"
              f"{m['per_axis_finer'][k]:>18.2%}")


if __name__ == "__main__":
    main()
