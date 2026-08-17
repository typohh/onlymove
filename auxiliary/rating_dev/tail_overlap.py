"""Overlap between the TAILS of stored motifs — which is not what correlation measures.

An axis is a trunk output, not a linear map on nameable inputs, so two motifs can
be uncorrelated over the bulk of the corpus and still select overlapping extremes.
Correlation answers "do these two order the corpus alike"; a NAME is a claim about
one end only, so what decides whether two names describe the same problems is how
much their tails share. The two can disagree in both directions: a pair at r=0.03
can share a third of a tail, and a pair at r=0.30 can share almost none if the
correlation lives in the middle.

Each motif contributes TWO tails, and they are separate objects: the axes are
bipolar, so a negative loading is a real direction rather than absence, and
`ko+` overlapping `counting-` is a finding about ko and about counting's opposite
pole, not about counting.

Scored on ATTEMPTED problems only, the standing convention for anything read off
the tails: a third of the corpus is never attempted, its loadings there are
unconstrained extrapolation, and it is ENRICHED in the extremes rather than merely
present — so including it would report the model's guesswork as motif structure.

Chance overlap is n_tail/n_problems (~1% at the default), so the number to read is
the LIFT over that, printed alongside.

  ./.venv/bin/python3 -m auxiliary.rating_dev.tail_overlap            # shipped axes
  ./.venv/bin/python3 -m auxiliary.rating_dev.tail_overlap --all-named
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import load_first_time_attempts  # noqa: E402

ANCHORS = os.path.join(PROJECT_ROOT, "axes", "label_anchors.npz")
LABELS = os.path.join(PROJECT_ROOT, "axes", "labels.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="tail size per end")
    ap.add_argument("--all-named", action="store_true",
                    help="every motif in the store with at least one named tail, "
                         "not just the shipped `components`")
    ap.add_argument("--ids", default=None,
                    help="comma-separated motif ids, overriding the two above")
    ap.add_argument("--all-problems", action="store_true",
                    help="include never-attempted problems (see the docstring)")
    ap.add_argument("--min-lift", type=float, default=3.0,
                    help="only list pairs at or above this lift over chance")
    ap.add_argument("--top", type=int, default=10,
                    help="how many of the strongest pairs to list (0 = all "
                         "above --min-lift)")
    ap.add_argument("--include-unnamed", action="store_true",
                    help="also score tails with no name. Off by default: an "
                         "unnamed pole makes no claim, so it has nothing to "
                         "collide with — it is where everything not at the named "
                         "end accumulates, and it overlaps other such poles at "
                         "rates that describe the residual rather than a motif.")
    ap.add_argument("--matrix", action="store_true",
                    help="print the full pairwise matrix (suppressed by default "
                         "beyond 14 tails, where it stops being readable)")
    args = ap.parse_args()

    meta = json.load(open(LABELS))
    bank = meta["bank"]
    z = np.load(ANCHORS, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]

    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()]
    elif args.all_named:
        want = [a for a in ids
                if bank.get(a, {}).get("positive") or bank.get(a, {}).get("negative")]
    else:
        want = [str(x) for x in meta.get("components", [])]
    missing = [a for a in want if a not in ids]
    if missing:
        raise SystemExit(f"not in the store: {missing}")

    M = np.asarray(z["M"], np.float64)[[ids.index(a) for a in want]]
    tids = z["tids"].astype(str)
    if not args.all_problems:
        att = set(load_first_time_attempts()["problem"].astype(str))
        keep = np.array([t in att for t in tids])
        M, tids = M[:, keep], tids[keep]

    # A motif is comparable only where it is defined; legacy directions are NaN
    # outside their own corpus generation. Intersecting keeps every pair scored
    # on one population, so the numbers in a row are comparable with each other.
    fin = np.isfinite(M).all(axis=0)
    M, tids = M[:, fin], tids[fin]
    n_prob = M.shape[1]
    if n_prob < 4 * args.n:
        raise SystemExit(f"only {n_prob} problems shared by all {len(want)} "
                         f"motifs — tails of {args.n} would not be tails")

    # UNNAMED TAILS ARE DROPPED by default. The question this answers is whether
    # two NAMES describe the same problems, and an unnamed pole makes no claim to
    # collide with — it is just where everything not at the named end accumulates,
    # which is why those poles overlap each other at rates that mean nothing.
    tails, names, owner = [], [], []
    for j, a in enumerate(want):
        e = bank.get(a, {})
        axis = e.get("positive") or e.get("negative") or a
        order = np.argsort(M[j])
        for sign, idx, lab in ((+1, order[-args.n:], e.get("positive")),
                               (-1, order[:args.n], e.get("negative"))):
            if lab is None and not args.include_unnamed:
                continue
            tails.append(set(idx.tolist()))
            names.append(f"{lab or (axis + '?')}{'+' if sign > 0 else '-'}")
            owner.append(a)
    # The store holds several fits' takes on one motif, so a bare name is not
    # unique — tag the repeats with their id or the table cannot be read.
    dup = {n for n in names if names.count(n) > 1}
    names = [f"{n} <{o[3:7]}>" if n in dup else n for n, o in zip(names, owner)]
    if len(tails) < 2:
        raise SystemExit("fewer than two named tails selected — nothing to compare")

    chance = args.n / n_prob
    print(f"tail overlap over {n_prob} {'' if args.all_problems else 'attempted '}"
          f"problems shared by all {len(want)} motifs, tails of {args.n} "
          f"(chance {chance:.1%})\n")
    # Columns are NUMBERED, not headed by truncated names: 'escape atari+' and
    # 'escape atari?-' both clip to 'escape a', so a truncated header silently
    # labels two different tails identically.
    if args.matrix or len(names) <= 14:
        w = max(len(x) for x in names) + 6
        print(" " * w + "".join(f"{j:>7}" for j in range(len(names))))
        for i, ni in enumerate(names):
            cells = []
            for j in range(len(names)):
                if i == j:
                    cells.append(f"{'—':>7}")
                else:
                    share = len(tails[i] & tails[j]) / args.n
                    cells.append(f"{share:>6.1%} " if share else f"{'·':>7}")
            print(f"{('[' + str(i) + '] ' + ni):<{w}}" + "".join(cells))
        print()

    hits = []
    for i, j in itertools.combinations(range(len(names)), 2):
        share = len(tails[i] & tails[j]) / args.n
        if share >= args.min_lift * chance:
            hits.append((share, names[i], names[j]))
    hits.sort(reverse=True)
    shown = hits[:args.top] if args.top else hits
    print(f"strongest {len(shown)} of {len(hits)} pairs at lift >= "
          f"{args.min_lift:g}x chance, over {len(names)} named tails:")
    wn = max((len(a) for _, a, _ in shown), default=10)
    for share, a, b in shown:
        print(f"  {a:<{wn}}  {b:<{wn}}  {share:>7.1%}  ({share / chance:>5.1f}x)")
    if not hits:
        print("  none — every named tail selects a distinct set of problems")


if __name__ == "__main__":
    main()
