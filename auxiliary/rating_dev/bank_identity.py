"""How far a dev fit's axes moved off the bank they were taught from.

This is `pipeline.train_mirt`'s BANK IDENTITY report applied to a dumped dev fit,
so an experimental arm can be judged on the thing a bank exists for — whether the
axis a name is attached to is still the axis that came out — rather than only on
logloss. An arm that wins the walk-forward and loses the motif has not won.

Scored on ATTEMPTED problems only, and both numbers are printed, for the reasons
in `report_bank_identity`: a third of the corpus is never attempted and its
extremes churn freely, and a name is a claim about the EXTREMES, which reproduce
worse than the correlation implies.

  ./.venv/bin/python3 -m auxiliary.rating_dev.bank_identity \\
      --dump /tmp/arms/baseline_s1.npz --bank /tmp/banks/baseline.npz
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import load_first_time_attempts  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="dumped fit (.npz with loading)")
    ap.add_argument("--bank", required=True, help="the bank it was taught from")
    ap.add_argument("--n-tail", type=int, default=500,
                    help="tail size per end (production uses 500)")
    ap.add_argument("--all-problems", action="store_true",
                    help="include never-attempted problems — their loadings are "
                         "unconstrained extrapolation and are ENRICHED in the "
                         "tails, so this understates what the fit holds")
    ap.add_argument("--cross", action="store_true",
                    help="drop the axis-k-vs-column-k pairing and print the full "
                         "axes x columns matrix. This is what a BIPOLAR arm needs: "
                         "a column taught as z(A) - z(B) is one target but two "
                         "claims, and the paired number cannot say whether the fit "
                         "kept both poles or collapsed onto whichever one carried "
                         "more signal. Point it at a bank of the individual poles.")
    args = ap.parse_args()

    d = np.load(args.dump, allow_pickle=True)
    b = np.load(args.bank, allow_pickle=True)
    g = d["loading"].astype(np.float64)
    load = b["loading"].astype(np.float64)
    labels = [str(x) for x in b["labels"]] if "labels" in b.files else \
        [f"axis {k}" for k in range(load.shape[1])]

    K = load.shape[1] if args.cross else min(g.shape[1], load.shape[1])
    row = {f"{s}:{t}": i for i, (s, t) in
           enumerate(zip(b["size"].astype(str), b["tid"].astype(str)))}
    keys = [f"{s}:{t}" for s, t in
            zip(d["size"].astype(str), d["tid"].astype(str))]

    idx = np.array([row.get(k, -1) for k in keys])
    ok = idx >= 0
    tgt = np.full((len(keys), K), np.nan)
    tgt[ok] = load[idx[ok], :K]
    tgt[ok] = (tgt[ok] - tgt[ok].mean(0)) / (tgt[ok].std(0) + 1e-12)

    if args.all_problems:
        sel = np.where(ok)[0]
        pop = "banked problems"
    else:
        att = set(load_first_time_attempts()["problem"].astype(str))
        is_att = np.array([t in att for t in d["tid"].astype(str)])
        sel = np.where(ok & is_att)[0]
        pop = "attempted+banked problems"

    print(f"{os.path.basename(args.dump)} vs {os.path.basename(args.bank)}")
    n = args.n_tail

    if args.cross:
        # TAIL OVERLAP, not correlation. This mode exists to catch a bipolar
        # column keeping one pole and dropping the other, and correlation cannot
        # see that: an axis whose extremes are a motif's extremes can still have
        # middling bulk correlation with it, because the axis is a trunk output
        # rather than a linear map and the two need not agree away from the ends.
        # Measured on the bipolar arm, correlation reported a collapse that the
        # tails show did not happen — every pole was retained at 10x chance or
        # better. Both of a fitted axis's ends are shown, since which end a motif
        # landed on is the question.
        chance = n / len(sel)
        print(f"  tail overlap over {len(sel)} {pop}, tails of {n} "
              f"(chance {chance:.1%})")
        print(f"  rows = a fitted axis's own tails, cols = each bank column's "
              f"POSITIVE tail\n")
        print("        " + "".join(f"{lab[:11]:>13}" for lab in labels))
        col_tail = [set(np.argsort(tgt[sel, j])[-n:].tolist()) for j in range(K)]
        for k in range(g.shape[1]):
            order = np.argsort(g[sel, k])
            for sign, t in (("+", set(order[-n:].tolist())),
                            ("-", set(order[:n].tolist()))):
                print(f"  ax{k} {sign}  " + "".join(
                    f"{len(t & col_tail[j]) / n:>12.1%} " for j in range(K)))
        return

    print(f"  BANK IDENTITY over {len(sel)} {pop} (tails of {2 * args.n_tail}):")
    for k in range(K):
        x, y = g[sel, k], tgt[sel, k]
        r = float(np.corrcoef(x, y)[0, 1])
        ox, oy = np.argsort(x), np.argsort(y)
        lo = len(set(ox[:n]) & set(oy[:n]))
        hi = len(set(ox[-n:]) & set(oy[-n:]))
        print(f"    axis {k} {labels[k]:<22} corr {r:+.3f} | "
              f"tails {lo + hi}/{2 * n}  (- {lo}, + {hi})")


if __name__ == "__main__":
    main()
