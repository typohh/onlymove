"""Persist a fit's axes into the motif store — `axes/label_anchors.npz` + `axes/labels.json`.

THE STORE IS THE ONLY THING THAT OUTLIVES A REFIT. A fit's axis space is defined
only up to rotation, so a motif's name survives a refit exactly as long as the
direction it was named for is kept somewhere and taught back. A fit's teaching
target is a SELECTION from this store rather than a file of its own; the store is
the accumulated set every future selection is drawn FROM, so a motif discovered
once need never be rediscovered.

Two files, one entry per motif, joined by an `ax_<8hex>` id:
  label_anchors.npz  ids, tids, sizes, M (n_axes, n_problems)  — the DIRECTIONS
  labels.json        bank[id] = {positive, negative, i18n, comment} — the NAMES
                     components = ordered ids — WHICH motifs this fit ships

`components` is the axis bank: an ordered list of store ids, so the shipping
decision is readable text bound BY NAME. A separate bank file would carry the
same directions in an implicit column order that `labels.json` is indexed
against with nothing enforcing agreement — reorder it and every motif silently
renames onto a direction it was not written for.

M is float16. The teaching target is z-scored before use, so only the direction
survives and float16 resolves it far past what a rescaled MSE can see; the store
is tracked and rewritten whole on every addition, so halving it is halving what
each addition costs in git history.

The id is a CONTENT HASH of the direction, so re-adding the same column is
idempotent and two different directions can never collide onto one name.

TIDS ARE THE CURRENT CORPUS, and rows that predate it are NaN-filled outside
their own coverage rather than dropped. A motif from an older corpus generation
still names a real direction over the problems it saw; discarding it because the
corpus moved would throw away exactly the history the store exists to keep. The
bank export is what has to make a dense matrix, and it intersects there.

  ./.venv/bin/python3 -m auxiliary.rating_dev.update_motif_store \\
      --add '/tmp/arms/ortho_s1.npz#2=ko' \\
      --comment 'ko=joint MIRT ortho-b seed 1; taught from ...' \\
      --dry-run
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.paths import tsumego_dir  # noqa: E402

ANCHORS = os.path.join(PROJECT_ROOT, "axes", "label_anchors.npz")
LABELS = os.path.join(PROJECT_ROOT, "axes", "labels.json")


def axis_id(col):
    """Content hash of the ORIENTED direction, scale-free.

    z-scored first, so a rescale is the same motif. Deliberately NOT
    sign-canonicalised: the id has to describe the column AS STORED, because the
    labels name its two tails by sign. Pinning the sign by skew and then storing
    the column unflipped gives the same id to a column and its mirror image, and
    whichever is added second keeps the first's labels while carrying opposite
    values — every name silently attached to the wrong tail, which is the failure
    the store exists to prevent. Skew is also a poor pivot for a bipolar axis,
    where it sits near zero and a small data change flips it.

    A mirror image therefore gets its OWN id, and `mirror_of` reports it at add
    time so a human decides whether it is a duplicate."""
    c = np.asarray(col, np.float64)
    c = c[np.isfinite(c)]
    c = (c - c.mean()) / max(c.std(), 1e-12)
    return "ax_" + hashlib.sha256(np.round(c, 4).tobytes()).hexdigest()[:8]


def mirror_of(col, ids, M, bank):
    """Existing motifs this column is strongly ANTI-correlated with — the same
    direction found the other way round, whose two tail names would need swapping
    rather than repeating."""
    c = np.asarray(col, np.float64)
    out = []
    for a, other in zip(ids, M):
        m = np.isfinite(c) & np.isfinite(other)
        if m.sum() < 1000:
            continue
        x, y = c[m], other[m]
        if x.std() <= 0 or y.std() <= 0:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r <= -0.7:
            e = bank.get(a, {})
            out.append((r, a, e.get("positive"), e.get("negative")))
    return sorted(out)


def corpus_tids():
    out = {}
    for size in (9, 13):
        d = tsumego_dir(size)
        for f in os.listdir(d):
            if f.endswith(".sgf"):
                out[f[:-4]] = size
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--add", action="append", default=[], metavar="DUMP#AXIS=NAME",
                    help="a column to store, repeatable. NAME may be 'A/B' for a "
                         "bipolar motif, naming the positive tail first.")
    ap.add_argument("--comment", action="append", default=[], metavar="NAME=TEXT",
                    help="provenance for a motif, repeatable. Say which fit it "
                         "came from and anything known to be wrong with it — the "
                         "store outlives the run logs, so an undocumented column "
                         "is a direction nobody can judge later.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change and write nothing")
    args = ap.parse_args()

    have = corpus_tids()
    tids = np.array(sorted(have))
    row = {t: i for i, t in enumerate(tids)}

    # ── existing store, remapped onto the current corpus tid axis ────────────
    ids, M = [], []
    if os.path.exists(ANCHORS):
        z = np.load(ANCHORS, allow_pickle=True)
        old_t = z["tids"].astype(str)
        idx = np.array([row.get(t, -1) for t in old_t])
        keep = idx >= 0
        for j, a in enumerate(z["ids"].astype(str)):
            col = np.full(len(tids), np.nan)
            col[idx[keep]] = z["M"][j][keep]
            ids.append(a)
            M.append(col)
        print(f"existing store: {len(ids)} motifs, {keep.sum()}/{len(old_t)} of "
              f"their problems still in the corpus")

    store = {"version": 4, "space": "mixed", "bank": {}, "components": []}
    if os.path.exists(LABELS):
        store = json.load(open(LABELS))
    store.setdefault("bank", {})

    comments = {}
    for c in args.comment:
        k, _, v = c.partition("=")
        comments[k.strip()] = v.strip()

    added, updated = [], []
    for spec in args.add:
        left, _, name = spec.partition("=")
        dump, _, k = left.rpartition("#")
        k = int(k)
        d = np.load(dump, allow_pickle=True)
        keys = [f"{s}:{t}" for s, t in zip(d["size"].astype(str), d["tid"].astype(str))]
        col = np.full(len(tids), np.nan)
        n_hit = 0
        for key, v in zip(keys, d["loading"][:, k].astype(np.float64)):
            t = key.split(":", 1)[1]
            i = row.get(t)
            if i is not None:
                col[i] = v
                n_hit += 1
        aid = axis_id(col)
        for r, other, op, on in mirror_of(col, ids, M, store["bank"]):
            print(f"    NOTE: r={r:+.2f} against {other} "
                  f"(+{op!r} / -{on!r}) — this looks like the SAME direction "
                  f"found mirrored. Its tails are this column's the other way "
                  f"round; store it only if you mean to keep both orientations.")
        pos, _, neg = name.partition("/")
        entry = store["bank"].get(aid, {})
        entry.update({
            "positive": pos.strip() or None,
            "negative": neg.strip() or None,
            "comment": comments.get(name.strip(),
                                    entry.get("comment", f"from {os.path.basename(dump)} axis {k}")),
        })
        entry.setdefault("positive_i18n", None)
        entry.setdefault("negative_i18n", None)
        store["bank"][aid] = entry
        if aid in ids:
            M[ids.index(aid)] = col
            updated.append((aid, name))
        else:
            ids.append(aid)
            M.append(col)
            added.append((aid, name))
        print(f"  {name:<30} {aid}  {n_hit}/{len(tids)} problems  "
              f"<- {os.path.basename(dump)}#{k}")

    Mn = np.stack(M, axis=0).astype(np.float16)
    sizes = np.array([have[t] for t in tids], np.int16)
    print(f"\nstore -> {len(ids)} motifs x {len(tids)} problems "
          f"({len(added)} new, {len(updated)} refreshed)")
    if args.dry_run:
        print("dry run — nothing written")
        return
    np.savez_compressed(ANCHORS, ids=np.array(ids), tids=tids, sizes=sizes, M=Mn)
    json.dump(store, open(LABELS, "w"), indent=2, ensure_ascii=False)
    print(f"wrote {ANCHORS}\nwrote {LABELS}")


if __name__ == "__main__":
    main()
