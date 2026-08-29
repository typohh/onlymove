"""Run train_joint_mirt with the corpus frame restricted to problems that
have an activation file.

The encoder-migration parity gate pairs two trainer runs (raw-store features
vs decoded-store features) on identical data. Problems added to the corpus
after the last extraction have no activation rows in either store — and the
raw arm can never gain them — so the paired frame is corpus ∩ activation
store. They also have no attempts yet, so the restriction drops no training
signal; without it the feature assembly fail-louds on the first new problem,
by design.

Every trainer flag passes through unchanged:
  ./.venv/bin/python3 -m auxiliary.run_mirt_on_activation_frame --k 3 ...
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import utils.rating.features as features

_orig_corpus_tids_by_size = features.corpus_tids_by_size


def _restricted(sizes):
    tids = _orig_corpus_tids_by_size(sizes)
    out = {}
    for s in sizes:
        act_dir = os.path.join(PROJECT_ROOT, "activations", str(s))
        have = {f[:-4] for f in os.listdir(act_dir) if f.endswith(".npz")}
        out[s] = [t for t in tids[s] if t in have]
        dropped = len(tids[s]) - len(out[s])
        if dropped:
            print(f"  frame: size {s} restricted to {len(out[s])} problems "
                  f"with activations ({dropped} newer corpus problems dropped)")
    return out


features.corpus_tids_by_size = _restricted

from auxiliary.rating_dev import train_joint_mirt  # noqa: E402

if __name__ == "__main__":
    train_joint_mirt.main()
