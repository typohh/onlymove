// js/math-utils.js
//
// Tiny shared numeric helpers used across the client models. These were
// previously copy-pasted into model.js, weakness.js, rank-decomposition.js,
// game-logic.js, stats-ui.js and bivariate-normal.js; consolidated here so the
// definitions can't drift.

// Numerically stable logistic. Avoids Math.exp overflow on large-magnitude z
// (the naive 1/(1+exp(-z)) saturates fine in JS but this form is exact on both
// tails and is what the weakness fitter used).
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
