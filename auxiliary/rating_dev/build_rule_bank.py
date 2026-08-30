#!/usr/bin/env python3
"""Build a teach bank for train_joint_mirt directly from the motif proxies.

Production's `pipeline.train_mirt.load_bank` reads `axes/labels.json`'s
`components` — the SHIPPING decision, K_AXES entries. A dev arm asking "which
rules deserve a slot" needs the same construction over an arbitrary rule set,
which labels.json cannot express without editing a tracked file to say
something it does not mean. This emits the identical per-problem columns —
`tail_weight(A)` or `tail_weight(A) - tail_weight(B)`, z-scored — as an npz in
the `--teach-directions` format ({size, tid, loading}), one column per
requested rule in the order given.

    ./.venv/bin/python3 -m auxiliary.rating_dev.build_rule_bank \
        --out /tmp/rule_bank_all9.npz            # every rule in LABEL_SET
    ./.venv/bin/python3 -m auxiliary.rating_dev.build_rule_bank \
        --rules "ko,yose,opening/territory" --out /tmp/bank.npz

The trainer re-standardizes each column over the problems it matches, so the
z-scoring here (over every proxy row) is a convenience, not the final scale.
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.motif_rules import LABEL_SET  # noqa: E402
from utils.rating.tail_agreement import tail_weight as tail_weight_np  # noqa: E402

MOTIF_PROXIES = os.path.join(PROJECT_ROOT, "axes", "motif_proxies.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=None,
                    help="comma-separated axis specs, each a rule name or "
                         "'A/B' for a bipolar column (default: every rule in "
                         "LABEL_SET, one unipolar column each)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    spec = ([s.strip() for s in args.rules.split(",") if s.strip()]
            if args.rules else list(LABEL_SET))

    z = np.load(MOTIF_PROXIES, allow_pickle=False)
    rules = [str(x) for x in z["rules"]]
    lab = np.asarray(z["labels"], bool)
    tid = z["tids"] if "tids" in z.files else z["tid"]
    size = z["size"]

    wanted = {r for col in spec for r in col.split("/")}
    missing = sorted(wanted - set(rules))
    if missing:
        raise SystemExit(f"proxies do not carry: {missing}. Available: {rules}")

    loading = np.zeros((len(tid), len(spec)), np.float64)
    for k, col in enumerate(spec):
        poles = col.split("/")
        w = tail_weight_np(lab[:, rules.index(poles[0])].astype(np.float64))
        if len(poles) == 2:
            w = w - tail_weight_np(lab[:, rules.index(poles[1])].astype(np.float64))
        loading[:, k] = (w - w.mean()) / (w.std() + 1e-12)
        n_pos = int(lab[:, rules.index(poles[0])].sum())
        extra = (f", {int(lab[:, rules.index(poles[1])].sum())} at the negative pole"
                 if len(poles) == 2 else "")
        print(f"  axis {k}: {col}  ({n_pos} positive-labelled{extra})")

    np.savez_compressed(args.out, size=size, tid=tid, loading=loading,
                        spec=np.array(spec))
    print(f"wrote {args.out}: {loading.shape[0]} problems x K={loading.shape[1]}")


if __name__ == "__main__":
    main()
