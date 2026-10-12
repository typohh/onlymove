"""data_version twins: the stamp packaging writes and the build's recomputation.

  compute_data_version (pipeline/package_for_pwa.py)
  computeDataVersion   (pwa/scripts/data-check.mjs)

The build refuses a tree whose stamp is not the hash of the files beside it, so
the two must agree on which files are folded, in what order, and what stands in
for a missing one.
"""

import os

from pipeline.package_for_pwa import DATA_VERSION_EPOCH, compute_data_version
from utils.constants import ALL_BOARD_SIZES

DATA_CHECK_JS = "pwa/scripts/data-check.mjs"

FILES = (
    [os.path.join("tsumegos", str(size), "metadata.json") for size in ALL_BOARD_SIZES]
    + [os.path.join("tsumegos", name) for name in (
        "problem_params.bin", "lookup.embeddings.bin", "lookup.offsets.bin",
        "lookup.altstats.bin", "se_floor.json")]
)


def _tree(root, present):
    for i, rel in enumerate(FILES):
        if rel not in present:
            continue
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(bytes([i, 255, 0, 10]) * (i + 3) + rel.encode())
    return root


def test_board_sizes_agree(js):
    assert js.run([(DATA_CHECK_JS, "ALL_BOARD_SIZES", None)]) == [list(ALL_BOARD_SIZES)]


def test_epoch_agrees(js):
    assert js.run([(DATA_CHECK_JS, "DATA_VERSION_EPOCH", None)]) == [DATA_VERSION_EPOCH.decode()]


def test_data_version_agrees(js, tmp_path):
    # The full tree, then each file missing in turn, then nothing at all.
    cases = [set(FILES)] + [set(FILES) - {rel} for rel in FILES] + [set()]
    roots = []
    for i, present in enumerate(cases):
        roots.append(_tree(str(tmp_path / f"case{i}"), present))
        os.makedirs(roots[-1], exist_ok=True)
    got = js.run([(DATA_CHECK_JS, "computeDataVersion", [r]) for r in roots])
    want = [compute_data_version(r) for r in roots]
    assert got == want
    assert len(set(want)) == len(want)
