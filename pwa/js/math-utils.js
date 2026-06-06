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
