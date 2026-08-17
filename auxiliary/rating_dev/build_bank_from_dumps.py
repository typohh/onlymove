"""Assemble an axis bank out of axes that a previous fit already converged on.

The rule-seeded banks (auxiliary.labeler.export_bank --from-rules) exist to get a
motif STARTED: a binary column that says "these are ko problems" and nothing else.
Once a fit has run, the axis it settled on is a strictly better description of the
same motif — it is continuous, it covers the corpus at the fit's own confidence,
and it is expressed in the space the trunk actually learned. So a second-generation
bank is built from dumped loadings, not from rules again.

A column is z-scored before it is combined or written. `load_bank` z-scores too,
so for a one-sided column this changes nothing; it matters for the BIPOLAR form
`A/B`, where two motifs are put on one axis as `z(A) - z(B)` and the two poles have
to arrive on a common scale or the wider one simply owns the axis.

A BIPOLAR column subtracts two TAIL-WEIGHTED poles: `f(z(A)) - f(z(B))` with `f`
from utils.rating.tail_agreement — the identity above `TAIL_HINGE`, decaying
exponentially to zero below it, the two branches meeting in value and slope.

Weighting rather than subtracting raw z-scores is the load-bearing part. Each
source axis is itself two-sided, and its negative end is a THIRD region rather
than the absence of anything — measured, the negative end of the life-and-death
axis is substantively "defend territory", agreeing with that motif's own tail at
+0.44. Raw subtraction would place a strongly-not-A problem at the B pole for no
reason but its lacking A, importing a motif nobody asked for.

The DECAY, rather than a hard floor `max(z, at)`, is what makes a problem near
neither pole contribute nothing instead of cancelling two equal plateaus, and
keeps a problem just short of A's tail leaning slightly toward A rather than not
at all. A plain exp() would reach the same floor and is rejected for the reason
it is rejected for rule columns: an exponentially heavy POSITIVE tail lets one
confident problem outweigh the rest of the column in the teaching MSE. This form
is linear up there and exponential only below.

Pairing does NOT require anti-correlated motifs. Measured pairs sit near r = -0.06
and every pole survives; what a pair needs is that both motifs are individually
reproducible, since retention tracks each motif's own strength rather than which
end of the column it was assigned to.

Orientation is written, never inferred. Every column is named by its positive tail,
so to flip an axis, write the poles the other way round (`B/A`) or prefix a
one-sided source with `-`. Nothing here guesses which end a motif belongs on.

  ./.venv/bin/python3 -m auxiliary.rating_dev.build_bank_from_dumps \\
      --dir /tmp/arms \\
      --columns 'control_s0#0, drop_counting_s0#1, drop_lad_s0#0' \\
      --labels  'escape atari, life and death, ko' \\
      --out /tmp/banks/baseline.npz
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.tail_agreement import TAIL_HINGE, tail_weight  # noqa: E402

ANCHORS = os.path.join(PROJECT_ROOT, "axes", "label_anchors.npz")


def _parse_source(tok):
    """'arm#axis', '-arm#axis', or a bare store id 'ax_xxxxxxxx' / '-ax_...'.

    A store id addresses a direction the motif store holds, which is the only
    place legacy directions exist — they predate every dump. The leading '-'
    takes the column's other pole, which is how one end of a stored BIPOLAR entry
    is used on its own."""
    tok = tok.strip()
    sign = 1.0
    if tok.startswith("-"):
        sign, tok = -1.0, tok[1:].strip()
    if tok.startswith("ax_") and "#" not in tok:
        return ("__store__", tok, sign)
    if "#" not in tok:
        raise SystemExit(f"source {tok!r} is not 'arm#axis' or a store id 'ax_...'")
    arm, _, axis = tok.rpartition("#")
    try:
        axis = int(axis)
    except ValueError:
        raise SystemExit(f"source {tok!r}: axis index is not an integer")
    return arm.strip(), axis, sign


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/tmp/arms",
                    help="directory holding the dumped .npz fits")
    ap.add_argument("--columns", required=True,
                    help="comma-separated column specs IN BANK ORDER (the first "
                         "K_AXES are what a fit ships). Each is 'arm#axis', "
                         "optionally '-arm#axis' to flip it, or 'A#i/B#j' for a "
                         "bipolar column z(A) - z(B).")
    ap.add_argument("--labels", required=True,
                    help="comma-separated names in the same order, 'A/B' for a "
                         "bipolar column. Names the POSITIVE tail first.")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--combine", choices=("weighted", "hinge", "linear"),
                    default="weighted",
                    help="how a bipolar column joins its poles. 'weighted' "
                         "(default) is f(z(A)) - f(z(B)) with the shared tail "
                         "weight. 'hinge' is the hard floor max(z,at) it "
                         "replaces, which flattens every sub-threshold problem to "
                         "one value. 'linear' is z(A) - z(B), which places a "
                         "strongly-not-A problem at the B pole; both are kept only "
                         "to reproduce older banks.")
    ap.add_argument("--hinge-at", type=float, default=TAIL_HINGE,
                    help="z threshold where the tail begins")
    args = ap.parse_args()

    specs = [s for s in args.columns.split(",") if s.strip()]
    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(specs) != len(labels):
        raise SystemExit(f"{len(specs)} columns but {len(labels)} labels")

    cache, ref = {}, None

    def load(arm):
        if arm not in cache:
            path = os.path.join(args.dir, arm if arm.endswith(".npz") else arm + ".npz")
            if not os.path.exists(path):
                raise SystemExit(f"no such dump: {path}")
            cache[arm] = np.load(path, allow_pickle=True)
        return cache[arm]

    store = {"z": None}

    def column(tok):
        nonlocal ref
        arm, axis, sign = _parse_source(tok)
        if arm == "__store__":
            if store["z"] is None:
                if not os.path.exists(ANCHORS):
                    raise SystemExit(f"no motif store at {ANCHORS}")
                store["z"] = np.load(ANCHORS, allow_pickle=False)
            z = store["z"]
            sids = [str(x) for x in z["ids"]]
            if axis not in sids:
                raise SystemExit(f"{axis} is not in the motif store")
            c = np.asarray(z["M"][sids.index(axis)], np.float64) * sign
            key = np.stack([z["sizes"].astype(str), z["tids"].astype(str)], axis=1)
            if ref is None:
                ref = (key, z["sizes"], z["tids"])
            elif not np.array_equal(key, ref[0]):
                raise SystemExit(
                    f"{axis}: the store's problem set differs from the first "
                    f"source. Mixing store and dump sources needs them aligned; "
                    f"regenerate the store against this corpus first.")
            fin = np.isfinite(c)
            if fin.sum() < 2:
                raise SystemExit(f"{axis}: no finite values")
            out = np.full(len(c), np.nan)
            out[fin] = (c[fin] - c[fin].mean()) / max(c[fin].std(), 1e-9)
            return out
        d = load(arm)
        load_arr = d["loading"]
        if not 0 <= axis < load_arr.shape[1]:
            raise SystemExit(f"{arm}: axis {axis} out of range (K={load_arr.shape[1]})")
        key = np.stack([d["size"].astype(str), d["tid"]], axis=1)
        # Every dump comes from the same feature assembly, so a disagreement here
        # is a stale dump rather than a reorder to be fixed up — refuse it.
        if ref is None:
            ref = (key, d["size"], d["tid"])
        elif not np.array_equal(key, ref[0]):
            raise SystemExit(f"{arm}: problem set differs from the first dump — "
                             f"these fits are not comparable row-for-row")
        c = load_arr[:, axis].astype(np.float64) * sign
        return (c - c.mean()) / max(c.std(), 1e-9)

    cols = []
    for spec in specs:
        poles = [p for p in spec.split("/") if p.strip()]
        if len(poles) == 1:
            cols.append(column(poles[0]))
        elif len(poles) == 2:
            a, b = column(poles[0]), column(poles[1])
            if args.combine == "weighted":
                a = tail_weight(a, args.hinge_at)
                b = tail_weight(b, args.hinge_at)
            elif args.combine == "hinge":
                a = np.maximum(a, args.hinge_at) - args.hinge_at
                b = np.maximum(b, args.hinge_at) - args.hinge_at
            cols.append(a - b)
        else:
            raise SystemExit(f"column {spec!r}: at most two poles")

    loading = np.stack(cols, axis=1)
    size, tid = ref[1], ref[2]

    # A bank row must be CARRIED, not merely present. Legacy directions are NaN
    # outside their own corpus generation, and a NaN reaching the trainer poisons
    # the per-column z-scoring for every problem at once. Emitting only complete
    # rows says the same thing the loader's mask does — absent is not a value —
    # and says it once, here, where the incompleteness is known.
    full = np.isfinite(loading).all(axis=1)
    if not full.any():
        raise SystemExit("no problem is covered by every column")
    if not full.all():
        print(f"  {int((~full).sum())} of {len(full)} problems dropped — not "
              f"covered by every column (legacy directions predate the corpus)")
    loading, size, tid = loading[full], size[full], tid[full]

    np.savez_compressed(args.out, loading=loading.astype(np.float32),
                        size=size, tid=tid, labels=np.array(labels),
                        provenance=np.array(args.columns.strip()))

    # Over the WHOLE corpus, unlike axis_correlations, which scores identity over
    # attempted problems only. The bank is taught on every row it covers, so the
    # corpus figure is the one that describes this file; the two do not match and
    # are not meant to.
    print(f"{'column':<34} {'source':<34} {'sd':>7} {'|r| corpus':>14}")
    R = np.corrcoef(loading.T) if loading.shape[1] > 1 else np.ones((1, 1))
    for j, (label, spec) in enumerate(zip(labels, specs)):
        off = np.abs(np.delete(R[j], j)) if loading.shape[1] > 1 else np.array([0.0])
        print(f"{label:<34} {spec.strip():<34} {loading[:, j].std():>7.3f} "
              f"{off.max():>14.3f}")
    print(f"\n-> {args.out}  loading {loading.shape}, {len(tid)} problems")


if __name__ == "__main__":
    main()
