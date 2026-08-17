"""Turn labelled motifs into an axis bank the MIRT can be taught onto.

Each label becomes one column of per-problem loadings. The label itself only
covers the pool (a few thousand problems); the BANK has to cover the corpus, and
the thing that generalises from one to the other is the probe — which is why the
probe is fitted in the MIRT's own feature space to begin with. So a column is the
probe's decision value over all 160k problems, not the rule's verdict over 5k.

THE COLUMN IS HINGED AT ZERO, and that is a claim about what a seeded motif is.
A label like "ko" names ONE tail: its positives are a real, describable set,
while its negatives are just everything else and carry no shared property. Left
as a raw decision value the column asserts an ordering down there too — this
problem is more not-ko than that one — and the teaching MSE would spend capacity
reproducing it. A hinge collapses that region to a single value, so the target
says "these are alike" where nothing distinguishes them, and the fit is free to
put whatever it finds at that end once the teaching weight decays.

The hinge does NOT make the column end at zero, and it is worth being clear that
it cannot: `load_bank` z-scores every column over the problems it covers, so a
floor at zero comes out wherever the mean puts it. What the hinge buys is the
absence of invented structure below the threshold, not the location of the floor.

exp() was the alternative and is worse on the same reasoning: it keeps a little
ordering in the flat region, and it makes the positive tail exponentially heavy,
so a single confident problem can outweigh the rest of the column in an MSE.

Scoring the corpus REUSES THE POOL'S STANDARDIZATION. The probe was fitted after
`(F - mean)/std` over the pool rows; re-standardizing over the corpus would move
every column under the fitted coefficients and evaluate the probe in a space it
was never fitted in. Same failure as the stale-pool one the server guards against
— it still produces numbers, and nothing downstream can tell.

  ./.venv/bin/python3 -m auxiliary.labeler.export_bank \\
      --labels "ko,life and death,counting" --out axes/axis_bank.npz
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.rating.features import load_joint_mirt_features  # noqa: E402
from utils.rating.model_constants import PCA_DIM, SEED, SIZES  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
LABELS_PATH = os.path.join(HERE, "labels.json")


def fit_probe(X, y, c=0.3, l1_ratio=0.5, max_iter=3000):
    """The server's probe, at its fixed cold C. Deliberately NOT the CV variant:
    a bank column is a direction, and re-selecting C per label would make the
    columns differ in how hard they were shrunk for reasons unrelated to the
    motif."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty="elasticnet", solver="saga",
                             l1_ratio=l1_ratio, C=c, max_iter=max_iter, tol=1e-3)
    clf.fit(X, y)
    return clf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True,
                    help="comma-separated label names, IN COLUMN ORDER — the "
                         "first K_AXES are what a fit ships, so this is the "
                         "shipping decision. A column may be BIPOLAR, written "
                         "'A/B': A becomes the positive tail and B the negative "
                         "one, so two motifs share an axis. That is the only way "
                         "to seed more motifs than there are axes, and unlike a "
                         "one-sided column it makes a real claim — that a user "
                         "strong at A is weak at B — which is exactly what the "
                         "fit is being asked to confirm or reject.")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--hinge", type=float, default=0.0,
                    help="decision-value floor; values below become the floor "
                         "(default 0.0 = the probe's own boundary)")
    ap.add_argument("--no-hinge", action="store_true",
                    help="write the raw decision value (both tails asserted)")
    ap.add_argument("--from-rules", default=None, metavar="RULES.json",
                    help="build the columns from a CORPUS-wide rule labelling "
                         "(auxiliary.labeler.rules --corpus) instead of from the "
                         "probe. THIS IS THE DEFAULT CHOICE FOR A BANK. The rule "
                         "answers for every problem, while the probe is fitted on "
                         "a pool capped at 1d and would supply the ~30%% of the "
                         "corpus harder than that by extrapolation — teaching the "
                         "trunk a fabricated verdict exactly where the axes are "
                         "least constrained. A binary column is also the shape a "
                         "one-sided seed wants: positives at one value, the rest "
                         "at a single constant, no ordering invented anywhere.")
    args = ap.parse_args()

    # "A/B" is one bipolar column; "A" is one one-sided column.
    specs = [[p.strip() for p in s.split("/")]
             for s in args.labels.split(",") if s.strip()]
    labels = ["/".join(sp) for sp in specs]
    store = json.load(open(args.from_rules or LABELS_PATH))
    ann = store["annotations"]
    missing = [l for sp in specs for l in sp if l not in store["label_set"]]
    if missing:
        raise SystemExit(f"not in {args.from_rules or LABELS_PATH}: {missing}")
    if any(len(sp) > 2 for sp in specs):
        raise SystemExit("a column has at most two poles ('A' or 'A/B')")

    if args.from_rules:
        # Straight from ground truth: no probe, no feature matrix, no
        # standardization to keep in step. `load_bank` z-scores the column, so
        # the two levels need no scale of their own — only their separation
        # means anything.
        keys = sorted(ann)
        cols = []
        for sp in specs:
            pos = np.array([1.0 if sp[0] in ann[t]["labels"] else 0.0 for t in keys])
            if len(sp) == 2:
                neg = np.array([1.0 if sp[1] in ann[t]["labels"] else 0.0
                                for t in keys])
                # A problem carrying BOTH poles is dropped to the neutral level
                # rather than summed: it is evidence for each end and so evidence
                # for neither ordering, and letting it cancel to 0 says that,
                # while +1-1 = 0 arrived at by addition would hide it.
                pos = np.where(pos * neg > 0, 0.0, pos - neg)
            cols.append(pos)
        load = np.stack(cols, axis=1)
        sizes, tids = [], []
        for size in (9, 13):
            d = os.path.join(PROJECT_ROOT, "tsumegos", str(size))
            have = {f[:-4] for f in os.listdir(d) if f.endswith(".sgf")}
            for t in keys:
                if t in have:
                    sizes.append(size)
                    tids.append(t)
        if len(tids) != len(keys):
            raise SystemExit(f"{len(keys) - len(tids)} labelled ids are not in the "
                             f"corpus — the rule file is stale against tsumegos/")
        order = {t: i for i, t in enumerate(keys)}
        load = load[[order[t] for t in tids]]
        np.savez_compressed(args.out, loading=load.astype(np.float32),
                            size=np.array(sizes), tid=np.array(tids),
                            labels=np.array(labels))
        print(f"{'column':<26} {'+ tail':>8} {'- tail':>8} {'neutral':>9}")
        for j, label in enumerate(labels):
            c = load[:, j]
            print(f"{label:<26} {int((c > 0).sum()):>8} {int((c < 0).sum()):>8} "
                  f"{int((c == 0).sum()):>9}")
        print(f"\ncolumn order (first K_AXES ship): {labels}")
        print(f"-> {args.out}  loading {load.shape}, {len(tids)} problems")
        return

    pool = json.load(open(os.path.join(CACHE_DIR, "pool.json")))
    Fp = np.load(os.path.join(CACHE_DIR, "features.npy")).astype(np.float64)
    mu, sd = Fp.mean(0), np.maximum(Fp.std(0), 1e-6)   # THE pool's transform
    Fp = (Fp - mu) / sd
    row = {t: i for i, t in enumerate(pool["tids"])}

    print(f"corpus features (sizes {list(SIZES)}, pca_dim {PCA_DIM}, seed {SEED})...",
          flush=True)
    feat, keys, _, _ = load_joint_mirt_features(SIZES, PCA_DIM, SEED)
    if feat.shape[1] != Fp.shape[1]:
        raise SystemExit(
            f"POOL AND CORPUS DISAGREE ON WIDTH: pool {Fp.shape[1]}, corpus "
            f"{feat.shape[1]}. The pool cache was built in a different feature "
            f"space — rebuild it with auxiliary.labeler.precompute_features.")
    Fc = (feat.astype(np.float64) - mu) / sd          # pool transform, corpus rows
    print(f"  {Fc.shape[0]} problems x {Fc.shape[1]} features")

    cols, report = [], []
    for label in labels:
        idx, y = [], []
        for t, rec in ann.items():
            i = row.get(t)
            if i is None or label not in rec["known"]:
                continue
            idx.append(i)
            y.append(1 if label in rec["labels"] else 0)
        y = np.array(y)
        if y.sum() < 5 or y.sum() == len(y):
            raise SystemExit(f"{label!r}: {int(y.sum())} positives of {len(y)} "
                             f"— not enough to fit a probe")
        clf = fit_probe(Fp[idx], y)
        d = clf.decision_function(Fc)
        if not args.no_hinge:
            d = np.maximum(d, args.hinge)
        cols.append(d)
        report.append((label, int(y.sum()), len(y), float((d > args.hinge).mean())))

    load = np.stack(cols, axis=1)
    sizes = np.array([k[0] for k in keys])
    tids = np.array([k[1] for k in keys])
    np.savez_compressed(args.out, loading=load.astype(np.float32),
                        size=sizes, tid=tids,
                        labels=np.array(labels))
    print(f"\n{'label':<18} {'trained on':>14} {'above hinge':>13}")
    for label, pos, n, frac in report:
        print(f"{label:<18} {pos:>6}/{n:<7} {frac:>12.1%}")
    print(f"\ncolumn order (first K_AXES ship): {labels}")
    print(f"-> {args.out}  loading {load.shape}, {len(tids)} problems")


if __name__ == "__main__":
    main()
