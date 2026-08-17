"""Hinged tail agreement — the one measure of whether two directions name the same thing.

A NAME is a claim about one END of an axis, so what decides whether two names
describe the same problems is agreement in the tail, not over the corpus. Three
instruments were tried and only the third survives:

BULK CORRELATION is wrong because an axis is a trunk output, not a linear map on
nameable inputs, so its extremes and its ordering of the middle are free to
diverge. Measured, it reversed which of two arms held identity better, and it
reported a bipolar column collapsing onto one pole when the tails show every pole
retained.

TOP-N OVERLAP is wrong in three ways at once: N is arbitrary, the cutoff is hard
so problem N and N+1 are total agreement and total disagreement, and membership
discards magnitude so a problem 5 sd into the tail is the same evidence as one
0.4 sd in. It manufactures matches out of problems that merely landed inside the
boundary — measured, an axis whose only real match is ko also showed two motifs
at "4%" that vanish entirely under a weighted measure.

HINGED CORRELATION is what this module implements: z-score each direction, floor
it at `TAIL_HINGE` (`max(z, at)`), correlate over the problems both cover. It has no N,
weights linearly by depth into the tail, and is EXACTLY ZERO under independence —
hinging two independent variables leaves them independent — so there is no chance
baseline to quote and the number reads directly.

It also works where top-N cannot: against a BINARY bank column, whose ~100k tied
zeros make `argsort` pick an arbitrary tail. Hinging a binary column leaves the
positives at one value and everything else at another, so the correlation
measures whether the fitted tail's weight concentrates on the rule's positives —
a real question, and the reason rule-seeded arms can be scored at all.
"""

import numpy as np

# THE TAIL THRESHOLD, in z units, shared by every tool so their numbers are
# comparable. 1.0 keeps ~8% of the corpus — a real tail, and still thousands of
# problems rather than an arbitrary few hundred.
#
# It is NOT a free parameter to leave at the mean. Hinging at 0 keeps the whole
# upper half and reproduces bulk correlation almost exactly (measured 0.867 bulk
# against 0.879 hinged), so it inherits the defect the hinge exists to remove.
# The choice is load-bearing: measured across arms, agreement on two motifs
# CROSSES OVER between 0.5 and 1.0 — one arm leads at the mean and the other in
# the tail — and it is the tail that a name is a claim about. Above 1.0 the
# ranking is stable through 2.0, so the value sits inside a plateau rather than
# on a knife edge.
TAIL_HINGE = 1.0


def hinge_z(v, at=TAIL_HINGE):
    """z-score then floor at `at`. NaN in, NaN out — legacy directions do not
    cover the whole corpus and must not be imputed to the mean, which would read
    as "average on this motif" for a problem the motif never saw."""
    v = np.asarray(v, dtype=np.float64)
    m = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if m.sum() < 2:
        return out
    s = v[m].std()
    if s <= 0:
        return out
    out[m] = np.maximum((v[m] - v[m].mean()) / s, at)
    return out


def hinged_r(a, b, at=TAIL_HINGE, min_n=100, pre_hinged=False):
    """Agreement between two tails, over the problems both cover.

    Returns NaN when they share too little, or when either hinged side is
    constant — a direction whose tail is empty on this population makes no claim
    and must not be scored 0, which would read as "measured, independent".
    """
    a = np.asarray(a, np.float64) if pre_hinged else hinge_z(a, at)
    b = np.asarray(b, np.float64) if pre_hinged else hinge_z(b, at)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float("nan")
    x, y = a[m], b[m]
    if x.std() <= 0 or y.std() <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
