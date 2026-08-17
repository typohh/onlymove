"""Choose which stored motifs to pair on one axis, and in which pairing.

Gluing two motifs onto one axis as `c = f(z(A)) - f(z(B))` can fail in two ways,
and they are not the same failure. Both are scored here on the GLUED column, not
on the raw directions — the reshape is what decides both, so a criterion computed
before `f` (bulk correlation, say) cannot see either.

CANCELLATION. Where both tails are active the difference subtracts, and those
problems leave the column. `Var(f_A - f_B) = Var(f_A) + Var(f_B) - 2 Cov`, so the
mean squared difference between the two reshaped poles IS the surviving signal and
maximising it is exactly minimising cancellation. Reported as `cancel`, the share
of the poles' summed variance that the subtraction destroys — `2 Cov / (V_A +
V_B)`, zero for independent poles, 1 for identical ones, negative for poles that
reinforce.

DOMINANCE. A column can have ample variance and still carry only one motif, if
one pole's reshaped spread swamps the other's; the weak pole then has no tail to
be named by. This is the failure that actually shows up — measured, a first
pairing returned poles at 0.17 and 0.26 agreement while their partners sat at 0.86
and 0.97. Reported as `balance`, `min(V) / max(V)` over the two reshaped poles, 1
being an even split.

Neither is a correlation between the motifs and the two do not imply each other: a
pair can cancel almost nothing and still be wholly dominated by one side, which is
why a single number cannot rank pairings. `score` combines them as
`(1 - cancel) * balance` — the surviving signal, discounted by how unevenly it is
shared — but read the columns, since which term binds tells you what to change.

  ./.venv/bin/python3 -m auxiliary.rating_dev.pair_motifs --all-named
  ./.venv/bin/python3 -m auxiliary.rating_dev.pair_motifs --ids ax_1,ax_2,... --top 8
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
from utils.rating.tail_agreement import TAIL_HINGE, tail_weight  # noqa: E402

ANCHORS = os.path.join(PROJECT_ROOT, "axes", "label_anchors.npz")
LABELS = os.path.join(PROJECT_ROOT, "axes", "labels.json")


def pair_stats(fa, fb):
    """cancel, balance and both-tail share for one glued column."""
    m = np.isfinite(fa) & np.isfinite(fb)
    if m.sum() < 1000:
        return None
    a, b = fa[m], fb[m]
    va, vb = a.var(), b.var()
    if va <= 0 or vb <= 0:
        return None
    cov = float(np.cov(a, b)[0, 1])
    cancel = 2 * cov / (va + vb)
    balance = min(va, vb) / max(va, vb)
    # Problems ABOVE the threshold on both poles: the ones the subtraction sends
    # to neutral. Distinct from `cancel`, which is a variance share — this says
    # how many named problems are actually lost.
    both = float(((a > TAIL_HINGE) & (b > TAIL_HINGE)).mean())
    return dict(cancel=float(cancel), balance=float(balance), both=both,
                score=float((1 - cancel) * balance))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default=None, help="comma-separated motif ids")
    ap.add_argument("--all-named", action="store_true",
                    help="every store motif with a named positive tail")
    ap.add_argument("--hinge", type=float, default=TAIL_HINGE)
    ap.add_argument("--top", type=int, default=8, help="pairings to list")
    ap.add_argument("--all-problems", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(LABELS))
    bank = meta["bank"]
    z = np.load(ANCHORS, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]

    if args.ids:
        want = [x.strip() for x in args.ids.split(",") if x.strip()]
    elif args.all_named:
        want = [a for a in ids if bank.get(a, {}).get("positive")]
    else:
        want = [str(x) for x in meta.get("components", [])]
    missing = [a for a in want if a not in ids]
    if missing:
        raise SystemExit(f"not in the store: {missing}")
    if len(want) % 2:
        raise SystemExit(f"{len(want)} motifs — pairing needs an even number")

    M = np.asarray(z["M"], np.float64)[[ids.index(a) for a in want]]
    tids = z["tids"].astype(str)
    if not args.all_problems:
        att = set(load_first_time_attempts()["problem"].astype(str))
        keep = np.array([t in att for t in tids])
        M = M[:, keep]
    names = [bank[a].get("positive") or a for a in want]
    F = [tail_weight(M[j], args.hinge) for j in range(len(want))]

    print(f"pairing diagnostics on the GLUED column, {len(want)} motifs, "
          f"tail at z >= {args.hinge}\n")
    stats = {}
    print(f"  {'pair':<46}{'cancel':>9}{'balance':>9}{'both%':>8}{'score':>8}")
    for i, j in itertools.combinations(range(len(want)), 2):
        st = pair_stats(F[i], F[j])
        if st is None:
            continue
        stats[(i, j)] = st
        print(f"  {names[i] + ' / ' + names[j]:<46}{st['cancel']:>+9.3f}"
              f"{st['balance']:>9.3f}{st['both'] * 100:>8.2f}{st['score']:>8.3f}")

    n = len(want)
    best = []
    idx = list(range(n))

    def matchings(rest):
        if not rest:
            yield []
            return
        a = rest[0]
        for k in range(1, len(rest)):
            b = rest[k]
            for tail in matchings([x for x in rest[1:] if x != b]):
                yield [(a, b)] + tail

    for mm in matchings(idx):
        sts = [stats.get(tuple(sorted(p))) for p in mm]
        if any(s is None for s in sts):
            continue
        best.append((min(s["score"] for s in sts),
                     float(np.mean([s["score"] for s in sts])), mm, sts))
    best.sort(reverse=True)
    print(f"\nBEST PAIRINGS, ranked by the WORST pair's score — an axis is only as "
          f"good as its weakest\n")
    for worst, mean, mm, sts in best[:args.top]:
        parts = " | ".join(
            f"{names[i]}/{names[j]} ({s['score']:.2f})"
            for (i, j), s in zip(mm, sts))
        print(f"  worst {worst:.3f}  mean {mean:.3f}   {parts}")


if __name__ == "__main__":
    main()
