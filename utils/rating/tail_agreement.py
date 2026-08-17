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

TAIL-WEIGHTED CORRELATION is what this module implements: z-score each direction,
weight it linearly in the tail and exponentially decayed below (see `tail_weight`),
correlate over the problems both cover. It has no N,
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

# WHERE THE TAIL STARTS, in z units, shared by every tool so their numbers are
# comparable. 1.0 puts ~8% of the corpus in the linear region.
#
# Under the SMOOTH weight this is deliberately not a decisive knob, and that is
# the reason for the smooth weight. With a hard floor the threshold decided the
# answer: two arms swapped places between 0.5 and 1.0, so any conclusion came
# with an argument about where to cut. Sweeping the smooth form from 1.0 to 3.0
# moves the numbers by ~0.01 and never changes an ordering, because the weight
# below the threshold decays rather than flattening, so nothing is discarded at
# the cut. Do not tune it to move a result.
TAIL_HINGE = 1.0


def tail_weight(v, at=TAIL_HINGE):
    """z-score, then LINEAR in the tail and exponentially decayed below it:

        f(z) = z                        z >= at
        f(z) = at * exp((z - at) / at)  z <  at

    The two pieces meet in VALUE and in SLOPE at `at` (both equal `at` and 1), so
    the weight is C1 — there is no kink for a squared error to chase and no
    threshold effect at the join.

    This is what a tail weight has to do, and a hard floor `max(z, at)` gets two
    of the three right. Above the threshold both are the identity, which is what
    keeps ONE confident problem from dominating a mean squared error — the reason
    a plain exp() is unusable here. Below it, the floor collapses everything to a
    single plateau, so a problem just short of the tail and one far from it are
    declared identical; the exponential keeps their ORDER while decaying it toward
    zero, which is the honest statement — barely-out is not the same as far-out,
    and neither is evidence for the opposite pole.

    Decaying toward ZERO rather than to a floor also makes the bipolar difference
    f(A) - f(B) mean what it should: a problem in neither tail contributes nothing
    instead of cancelling two equal plateaus, and one just outside A's tail leans
    slightly toward A rather than not at all.

    NaN in, NaN out — legacy directions do not cover the whole corpus and must not
    be imputed, which would read as "average on this motif" for a problem the
    motif never saw.
    """
    if at <= 0:
        raise ValueError("tail threshold must be positive; the exponential branch "
                         "is scaled by it")
    v = np.asarray(v, dtype=np.float64)
    m = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if m.sum() < 2:
        return out
    s = v[m].std()
    if s <= 0:
        return out
    z = (v[m] - v[m].mean()) / s
    out[m] = np.where(z >= at, z, at * np.exp((z - at) / at))
    return out


# The former name; the hard floor it described is retired.
hinge_z = tail_weight


def hinged_r(a, b, at=TAIL_HINGE, min_n=100, pre_hinged=False):
    """Agreement between two tails, over the problems both cover.

    Returns NaN when they share too little, or when either hinged side is
    constant — a direction whose tail is empty on this population makes no claim
    and must not be scored 0, which would read as "measured, independent".
    """
    a = np.asarray(a, np.float64) if pre_hinged else tail_weight(a, at)
    b = np.asarray(b, np.float64) if pre_hinged else tail_weight(b, at)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return float("nan")
    x, y = a[m], b[m]
    if x.std() <= 0 or y.std() <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
