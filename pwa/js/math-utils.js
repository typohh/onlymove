// js/math-utils.js
//
// Tiny shared numeric helpers used across the client models
// (model.js, weakness.js, rank-decomposition.js, game-logic.js,
// stats-compute.js, the stats chart modules, bivariate-normal.js), kept in one
// place so the definitions can't drift.

// Numerically stable logistic. Avoids Math.exp overflow on large-magnitude z
// (the naive 1/(1+exp(-z)) saturates fine in JS but this form is exact on both
// tails).
export function sigmoid(z) {
    if (z >= 0) {
        return 1 / (1 + Math.exp(-z));
    }
    const e = Math.exp(z);
    return e / (1 + e);
}

// Standard normal CDF via math.js's erf (the `math` global from lib/math.min.js,
// loaded before the module bundle, so it's defined by the time these run).
export function normCdf(z) {
    return 0.5 * (1 + math.erf(z / Math.SQRT2));
}

// Round an embedding's floats to 2 decimals before persisting it on an
// answerHistory entry — the localStorage footprint of K floats × many entries
// matters and 2dp is ample for the weakness model. Used everywhere an embedding
// (correct-move or picked) is stored.
export function quantizeEmbedding(arr) {
    return arr.map(v => Math.round(v * 100) / 100);
}

// The (untruncated) Gaussian kernel weight exp(-0.5·x²), shared by the stats
// smoothers (stats-compute.js, rank-decomposition.js). rank-decomposition keeps
// its own named TRUNCATED variant (exact 0 past KERNEL_TRUNC_SIGMA) built on
// this one.
export function gaussianWeight(x) {
    return Math.exp(-0.5 * x * x);
}

// First index in ascending `arr` whose value is >= `target` (binary search).
// `valueOf` extracts the ordering key — identity for plain number arrays;
// rank-decomposition passes `a => a.t` for its {t, …} attempt rows.
export function lowerBound(arr, target, valueOf = (x) => x) {
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (valueOf(arr[m]) < target) lo = m + 1; else hi = m; }
    return lo;
}

// Plain linear-interpolation quantile of an ASCENDING-sorted numeric array
// (unweighted counterpart of weightedQuantile below).
export function quantileSorted(sorted, p) {
    if (sorted.length === 1) return sorted[0];
    const idx = p * (sorted.length - 1);
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

// Kernel-weighted empirical quantile over a window. `pairs` is ascending by
// value with `{ v, w }` entries; `total` is Σw. Walk the cumulative weight to
// p·total and linearly interpolate between the two bracketing samples, so the
// edge moves smoothly as the window slides instead of stepping from one sample
// value to the next. Used by the stats charts' kernel smoothers (the centre
// line at p=0.5 and the spread band at 0.25/0.75).
export function weightedQuantile(pairs, total, p) {
    const target = p * total;
    let cum = 0;
    for (let i = 0; i < pairs.length; i++) {
        const next = cum + pairs[i].w;
        if (next >= target) {
            if (i === 0) return pairs[0].v;
            return pairs[i - 1].v + (pairs[i].v - pairs[i - 1].v) * ((target - cum) / pairs[i].w);
        }
        cum = next;
    }
    return pairs[pairs.length - 1].v;
}
