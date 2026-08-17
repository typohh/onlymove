"""How do the axes of several fits relate — the same motif across runs, and
different motifs against each other?

Two questions, one matrix, and they are asked of the same numbers on purpose:

  WITHIN a motif, across runs: how reproducible is a taught direction? The
  round-1 design fits each rule motif in three separate arms, so every motif has
  three independent realizations and three pairwise agreements. That spread is
  the YARDSTICK any schedule change has to be judged against — a decay length is
  "sufficient" when it retains a direction about as well as two independently
  taught arms already agree, and no length can do better than that.

  BETWEEN motifs: which pair is most negatively correlated? That is the pairing a
  BIPOLAR bank column wants, since it asserts one motif at each end of an axis.
  Read the number, not just its rank: near-orthogonal motifs produce a "most
  negative" pair at r of a few hundredths, which is noise rather than opposition.
  Forcing those into one column tests whether the fit accepts an INVENTED
  opposition, which is a fair experiment but a different one, and a null result
  then means "these are not opposites" rather than "bipolar banks fail".

Correlations are over ATTEMPTED problems by default. A third of the corpus is
never attempted and its loadings are unconstrained extrapolation — enriched in
the tails rather than merely present — so including it measures the feature net's
guesswork alongside the thing being asked about.

Tail overlap is reported beside correlation for the within-motif table because a
NAME is a claim about the extremes, and correlation describes the bulk. A binary
bank teaches membership without ordering, so agreement at small N is expected to
be lower than the correlation suggests; that is the seed's design, not drift.

  ./.venv/bin/python3 -m auxiliary.rating_dev.axis_correlations \\
      --dir /tmp/arms --banks /tmp/banks \\
      --names control="escape atari,defend territory,difficulty"
"""

import argparse
import itertools
import os
import re
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import load_first_time_attempts  # noqa: E402


def load_axes(dump, bank=None, names=None):
    """[(motif, {tid: value})] for one fit. Names come from an explicit list, else
    from the bank the run was taught with — a dump stores no names of its own."""
    z = np.load(dump, allow_pickle=False)
    tids = [str(t) for t in z["tid"]]
    L = np.asarray(z["loading"], np.float64)
    if names is None:
        if bank is None:
            raise SystemExit(
                f"{os.path.basename(dump)}: no bank and no --names. A free fit's "
                f"axes have no names anywhere, so they must be given by hand; a "
                f"taught fit takes them from its bank.")
        b = np.load(bank, allow_pickle=False)
        if "labels" not in b.files:
            raise SystemExit(f"{bank} carries no 'labels' array")
        names = [str(x) for x in b["labels"]]
    if len(names) < L.shape[1]:
        raise SystemExit(f"{os.path.basename(dump)}: {L.shape[1]} axes but "
                         f"{len(names)} names")
    return [(names[k], dict(zip(tids, L[:, k]))) for k in range(L.shape[1])]


def tail_overlap(a, b, n):
    """Shared membership of the top n, sign-aligned so an anti-correlated pair is
    compared end-for-end rather than scored as total disagreement."""
    s = 1.0 if np.corrcoef(a, b)[0, 1] >= 0 else -1.0
    return len(set(np.argsort(-a)[:n]) & set(np.argsort(-s * b)[:n])) / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="/tmp/arms", help="directory of --dump-params npz")
    ap.add_argument("--banks", default="/tmp/banks",
                    help="directory of banks; <arm>.npz supplies <arm>'s axis names")
    ap.add_argument("--names", action="append", default=[], metavar="ARM=a,b,c",
                    help="axis names for an arm with no bank (repeatable)")
    ap.add_argument("--all-problems", action="store_true",
                    help="include never-attempted problems (see the docstring)")
    ap.add_argument("--tail", type=int, default=2000, help="tail size for overlap")
    ap.add_argument("--csv", default=None, help="also write the matrix here")
    ap.add_argument("--pick", type=int, default=0, metavar="K",
                    help="choose the K mutually LEAST correlated axes out of every "
                         "converged instance, and print them as a bank recipe. The "
                         "choice is over INSTANCES, not motifs: each rule motif is "
                         "fitted in several arms and those realizations differ, so "
                         "which arm a column is taken from is part of the decision. "
                         "Two instances of one motif are never chosen together — "
                         "they correlate far too highly to survive the objective.")
    ap.add_argument("--pairs", action="store_true",
                    help="enumerate every way of splitting the motifs into "
                         "disjoint BIPOLAR pairs and rank by total correlation. "
                         "The pairing is a joint choice, not three separate ones: "
                         "greedily taking the most negative pair can strand the "
                         "rest, so all perfect matchings are scored.")
    ap.add_argument("--exclude", default="", metavar="a,b",
                    help="motifs to leave out of --pairs (e.g. difficulty, which "
                         "is not a motif and will not be shipped)")
    args = ap.parse_args()

    override = {}
    for spec in args.names:
        arm, _, lst = spec.partition("=")
        override[arm] = [s.strip() for s in lst.split(",") if s.strip()]

    cols, labels = [], []
    for fn in sorted(os.listdir(args.dir)):
        if not fn.endswith(".npz"):
            continue
        arm = re.sub(r"_s\d+\.npz$", "", fn)
        bank = os.path.join(args.banks, f"{arm}.npz")
        axes = load_axes(os.path.join(args.dir, fn),
                         bank if os.path.exists(bank) else None,
                         override.get(arm))
        for k, (motif, series) in enumerate(axes):
            labels.append((arm, motif, k))
            cols.append(series)
    if len(cols) < 2:
        raise SystemExit(f"found {len(cols)} axes in {args.dir}; need at least 2")

    shared = set.intersection(*[set(c) for c in cols])
    if not args.all_problems:
        shared &= set(load_first_time_attempts()["problem"].astype(str))
    shared = sorted(shared)
    X = np.array([[c[t] for c in cols] for t in shared])
    C = np.corrcoef(X, rowvar=False)
    print(f"{len(cols)} axes from {len({a for a, _, _ in labels})} fits, "
          f"{len(shared)} shared "
          f"{'problems' if args.all_problems else 'attempted problems'}\n")

    w = max(len(f"{a}:{m}") for a, m, _ in labels) + 2
    print(" " * w + "".join(f"{i:>7}" for i in range(len(labels))))
    for i, (a, m, _) in enumerate(labels):
        print(f"{a}:{m}".ljust(w) + "".join(f"{C[i, j]:>+7.2f}" for j in range(len(labels)))
              + f"   [{i}]")

    # --- the same motif, fitted more than once
    by_motif = {}
    for i, (a, m, _) in enumerate(labels):
        by_motif.setdefault(m, []).append((a, i))
    print(f"\nWITHIN MOTIF — reproducibility across fits (overlap at top {args.tail})")
    for m, entries in sorted(by_motif.items()):
        if len(entries) < 2:
            continue
        rs = []
        for (a1, i), (a2, j) in itertools.combinations(entries, 2):
            ov = tail_overlap(X[:, i], X[:, j], args.tail)
            rs.append(C[i, j])
            print(f"  {m:<18} {a1:>14} x {a2:<14} r={C[i, j]:+.3f}  top{args.tail} {ov:.1%}")
        print(f"  {m:<18} {'':>14}   {'mean':<14} r={np.mean(rs):+.3f}  "
              f"min r={min(rs):+.3f}  over {len(rs)} pair(s)\n")

    # --- different motifs: the bipolar pairing question
    print("MOST NEGATIVE cross-motif pairs (candidate bipolar poles), averaged "
          "over fit pairs:")
    agg = {}
    for i, (a1, m1, _) in enumerate(labels):
        for j, (a2, m2, _) in enumerate(labels):
            if j <= i or m1 == m2:
                continue
            agg.setdefault(tuple(sorted((m1, m2))), []).append(C[i, j])
    for (m1, m2), rs in sorted(agg.items(), key=lambda kv: np.mean(kv[1]))[:10]:
        print(f"  {m1:<18} x {m2:<18} mean r={np.mean(rs):+.3f}  "
              f"min {min(rs):+.3f}  max {max(rs):+.3f}  ({len(rs)} pairs)")

    if args.pick:
        drop = {s.strip() for s in args.exclude.split(",") if s.strip()}
        cand = [i for i, (_, m, _) in enumerate(labels) if m not in drop]
        best = []
        for combo in itertools.combinations(cand, args.pick):
            if len({labels[i][1] for i in combo}) < args.pick:
                continue                      # never two takes of one motif
            worst = max(abs(C[i, j]) for i, j in itertools.combinations(combo, 2))
            tot = sum(abs(C[i, j]) for i, j in itertools.combinations(combo, 2))
            best.append((worst, tot, combo))
        best.sort()
        print(f"\nLEAST-CORRELATED {args.pick} of {len(cand)} converged instances"
              f"{' — excluding ' + ', '.join(sorted(drop)) if drop else ''}"
              f"\n  ranked by the WORST pairwise |r| (that is what ORTHO_WEIGHT "
              f"acts on; the sum is shown as a tie-break)")
        for worst, tot, combo in best[:6]:
            print(f"  max|r| {worst:.3f}  sum {tot:.3f}   " +
                  "   ".join(f"{labels[i][0]}:{labels[i][1]}" for i in combo))
        print("\n  bank recipe (arm, axis index within that dump, motif):")
        for i in best[0][2]:
            arm, motif, k = labels[i]
            print(f"    {arm}_s0.npz  axis {k}  -> {motif!r}")

    if args.pairs:
        drop = {s.strip() for s in args.exclude.split(",") if s.strip()}
        motifs = sorted({m for _, m, _ in labels} - drop)
        if len(motifs) % 2:
            raise SystemExit(f"{len(motifs)} motifs after exclusions — pairing "
                             f"needs an even count: {motifs}")
        R = {}
        for m1, m2 in itertools.combinations(motifs, 2):
            R[(m1, m2)] = float(np.mean(agg[tuple(sorted((m1, m2)))]))

        def matchings(rest):
            if not rest:
                yield []
                return
            a, others = rest[0], rest[1:]
            for i, b in enumerate(others):
                for tail in matchings(others[:i] + others[i + 1:]):
                    yield [(a, b)] + tail

        def r(m1, m2):
            return 1.0 if m1 == m2 else R[tuple(sorted((m1, m2)))]

        def between(p, q):
            """corr of the columns (a1-b1) and (a2-b2), in closed form. The
            motifs are unit-variance (the bank z-scores every column), so
            var(a-b) = 2 - 2r(a,b) and the cross term expands directly."""
            (a1, b1), (a2, b2) = p, q
            cov = r(a1, a2) - r(a1, b2) - r(b1, a2) + r(b1, b2)
            return cov / np.sqrt((2 - 2 * r(a1, b1)) * (2 - 2 * r(a2, b2)))

        scored = []
        for mm in matchings(motifs):
            tot = sum(R[tuple(sorted(p))] for p in mm)
            # What ORTHO_WEIGHT will see. A pairing that is excellent within
            # pairs and correlated between them is a bank the objective fights
            # from the first epoch, so both numbers have to be read together.
            worst = max(abs(between(p, q)) for p, q in itertools.combinations(mm, 2))
            scored.append((tot, worst, mm))
        scored.sort(key=lambda s: (s[0], s[1]))
        print(f"\nBIPOLAR PAIRINGS over {len(motifs)} motifs "
              f"({len(scored)} perfect matchings){' — excluding ' + ', '.join(sorted(drop)) if drop else ''}"
              f"\n  ranked by TOTAL correlation (most negative first):")
        print(f"  {'total':>8} {'max|between|':>13}   pairs")
        for tot, worst, mm in scored[:6]:
            parts = "   ".join(f"{a}/{b} ({R[tuple(sorted((a, b)))]:+.3f})" for a, b in mm)
            print(f"  {tot:>+8.3f} {worst:>13.3f}   {parts}")
        best_ortho = min(scored, key=lambda s: s[1])
        print(f"\n  most ORTHOGONAL matching (what the objective will accept):")
        print(f"  {best_ortho[0]:>+8.3f} {best_ortho[1]:>13.3f}   " +
              "   ".join(f"{a}/{b} ({R[tuple(sorted((a, b)))]:+.3f})"
                         for a, b in best_ortho[2]))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("," + ",".join(f"{a}:{m}" for a, m, _ in labels) + "\n")
            for i, (a, m, _) in enumerate(labels):
                f.write(f"{a}:{m}," + ",".join(f"{v:.6f}" for v in C[i]) + "\n")
        print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()
