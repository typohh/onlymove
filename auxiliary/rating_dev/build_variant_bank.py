#!/usr/bin/env python3
"""Build a teach bank with ONE axis column swapped for a rule VARIANT — the
same rule at a different bar, or its mirror — computed from the raw quantities
dump rather than the baked proxies.

    ./.venv/bin/python3 -m auxiliary.rating_dev.build_variant_bank \
        --replace "sente=gote@4" --out /tmp/bank_gote4.npz
    ./.venv/bin/python3 -m auxiliary.rating_dev.build_variant_bank \
        --replace "sente=sente@6" --out /tmp/bank_sente6.npz

`--replace COL=RULE@BAR` keeps every other LABEL_SET column exactly as
`build_rule_bank` would emit it from axes/motif_proxies.npz, and rebuilds the
named column from axes/rule_quantities.csv with the requested bar, so an arm
differs from the baseline in that one direction and nothing else. RULE is
sente or gote. The spec array records the variant name, so a dump's provenance
says what was taught.

The baseline bank teaches LABEL_SET minus gote (the original 9), so variant
arms are column-for-column comparable with the all-9 baseline fits.
"""

import argparse
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.motif_rules import rule_gote, rule_sacrifice, rule_sente  # noqa: E402
from utils.rating.tail_agreement import tail_weight as tail_weight_np  # noqa: E402

MOTIF_PROXIES = os.path.join(PROJECT_ROOT, "axes", "motif_proxies.npz")
QUANTITIES = os.path.join(PROJECT_ROOT, "axes", "rule_quantities.csv")
OWNERSHIP = os.path.join(PROJECT_ROOT, "activations", "_signal_cache",
                         "ownership_swing.csv")
BASE_SPEC = ["life and death", "yose", "ko", "opening", "territory",
             "shape", "tenuki", "sente", "semeai"]


def variant_verdicts(rule, bar):
    """{(size, tid): bool} for the variant rule at the given bar."""
    out = {}
    if rule in ("sacrifice", "netsac"):
        with open(OWNERSHIP, newline="") as fh:
            for rec in csv.DictReader(fh):
                if not rec.get("alt_move"):
                    continue
                out[(int(rec["size"]), rec["tid"])] = bool(rule_sacrifice(
                    rec, min_stones=bar, net=(rule == "netsac")))
        return out
    with open(QUANTITIES, newline="") as fh:
        for rec in csv.DictReader(fh):
            ar = int(rec["answer_reply"]) if rec["answer_reply"] else None
            rr = int(rec["refuted_reply"]) if rec["refuted_reply"] else None
            if rule == "sente":
                v = rule_sente(ar, rr, min_dist=bar)
            elif rule == "gote":
                v = rule_gote(rec["answer_kind"], ar, rr, min_dist=bar)
            else:  # fast@K: at most K orthogonally-adjacent stones
                v = rec["adj_stones"] != "" and int(rec["adj_stones"]) <= bar
            out[(int(rec["size"]), rec["tid"])] = bool(v)
    return out


VARIANT_RULES = ("sente", "gote", "fast", "sacrifice", "netsac")


def column(token, rules, lab, tid, size):
    """One bool column for a spec token: a plain LABEL_SET rule name, or
    RULE@BAR resolved through variant_verdicts."""
    if "@" not in token:
        return lab[:, rules.index(token)]
    rule, bar = token.split("@")
    if rule not in VARIANT_RULES:
        raise SystemExit(f"variant rule must be one of {VARIANT_RULES}, "
                         f"got {rule}")
    bar = float(bar) if rule in ("sacrifice", "netsac") else int(bar)
    verd = variant_verdicts(rule, bar)
    vcol = np.zeros(len(tid), bool)
    missing = 0
    for i, (s, t) in enumerate(zip(size, tid)):
        try:
            vcol[i] = verd[(int(s), str(t))]
        except KeyError:
            missing += 1
    if missing:
        raise SystemExit(f"{missing} proxy rows have no source row for "
                         f"{token} — re-run dump_rule_quantities / "
                         f"extract_ownership_swing against this corpus")
    return vcol


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replace", metavar="COL=RULE@BAR",
                   help="BASE_SPEC with one column swapped for a variant")
    g.add_argument("--spec", metavar="TOK,TOK,...",
                   help="an arbitrary axis list: each token a LABEL_SET rule "
                        "name or RULE@BAR variant, one axis per token — the "
                        "composite-set mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.spec:
        spec_in = [t.strip() for t in args.spec.split(",") if t.strip()]
    else:
        col_name, spec_v = args.replace.split("=")
        if col_name not in BASE_SPEC:
            raise SystemExit(f"{col_name} not in {BASE_SPEC}")
        spec_in = list(BASE_SPEC)
        spec_in[BASE_SPEC.index(col_name)] = spec_v

    z = np.load(MOTIF_PROXIES, allow_pickle=False)
    rules = [str(x) for x in z["rules"]]
    lab = np.asarray(z["labels"], bool)
    tid = z["tids"] if "tids" in z.files else z["tid"]
    size = z["size"]

    loading = np.zeros((len(tid), len(spec_in)), np.float64)
    for k, token in enumerate(spec_in):
        raw = column(token, rules, lab, tid, size)
        w = tail_weight_np(raw.astype(np.float64))
        loading[:, k] = (w - w.mean()) / (w.std() + 1e-12)
        print(f"  axis {k}: {token}  ({int(raw.sum())} positive)")

    np.savez_compressed(args.out, size=size, tid=tid, loading=loading,
                        spec=np.array(spec_in))
    print(f"wrote {args.out}: {loading.shape[0]} problems x K={loading.shape[1]}")


if __name__ == "__main__":
    main()
