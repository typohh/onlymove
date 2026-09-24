"""The availability null's two producers must agree EXACTLY.

  _js_quantize + alt_stat_masses  (pipeline/package_for_pwa.py) — packaging,
                                  written to lookup.altstats.bin
  computeAltStats                 (pwa/js/picked-embedding.js) — the live click path

Both take the problem's per-cell block (float16 on disk, row 0 the answer) and
one integer visit weight per row, and both masses are integer sums, so the
comparison is equality, not a tolerance. The inputs are float16 values decoded
to float64 — what both producers actually see — and deliberately include the
cases where a rounding rule decides a count: values landing on an exact .5
after scaling, rows that tie the answer on some axes, the all-zeros pass row,
and rows small enough to quantize to it.
"""

import random

import numpy as np

from pipeline.package_for_pwa import _js_quantize, alt_stat_masses

PICKED = "pwa/js/picked-embedding.js"
K = 7


def _block(rng, n_cells):
    """One problem's per-cell block as the client decodes it: float16 → float64."""
    b = rng.normal(0.0, 0.4, size=(n_cells, K))
    # Exact float16 values whose ×100 lands on .5: the JS round-half-up tie.
    ties = np.array([0.125, -0.125, 0.375, -0.375, 0.625, 0.875, 1.125])
    mask = rng.random(b.shape) < 0.15
    b[mask] = rng.choice(ties, size=int(mask.sum()))
    # Rows copying the answer on some axes, so the strict comparisons tie.
    for r in range(1, n_cells):
        same = rng.random(K) < 0.2
        b[r, same] = b[0, same]
    # The pass placeholder, and a row small enough to quantize onto it.
    if n_cells > 2:
        b[rng.integers(1, n_cells)] = 0.0
    if n_cells > 3:
        b[rng.integers(1, n_cells)] = rng.uniform(-0.004, 0.004, size=K)
    return b.astype(np.float16).astype(np.float64)


def _cases():
    rng = np.random.default_rng(5)
    prng = random.Random(5)
    cases = []
    for _ in range(300):
        n = int(rng.integers(2, 41))
        block = _block(rng, n)
        visits = [prng.choice([0, 1, 30, 30, 100, 1000, 12345]) for _ in range(n)]
        cases.append((block, visits))
    return cases


def test_masses_agree_exactly(js):
    cases = _cases()
    got = js.run([(PICKED, "computeAltStats", (b, b[0], w)) for b, w in cases])
    for (block, visits), j in zip(cases, got):
        above, below = alt_stat_masses(_js_quantize(block), visits)
        assert j is not None
        assert j["altAbove"] == above.tolist()
        assert j["altBelow"] == below.tolist()


def test_refuses_without_matching_weights(js):
    """No weights, or weights for different variations than the rows, is a
    refusal (null) — never a count-shaped answer in the wrong units."""
    block = _cases()[0][0]
    rows = block.tolist()
    n = len(rows)
    got = js.run([
        (PICKED, "computeAltStats", (rows, rows[0])),
        (PICKED, "computeAltStats", (rows, rows[0], [30] * (n + 1))),
        (PICKED, "computeAltStats", (rows, rows[0], [30.5] * n)),
    ])
    assert got == [None, None, None]
