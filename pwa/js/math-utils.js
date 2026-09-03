// js/math-utils.js
//
// Tiny shared numeric helpers used across the client models
// (recall-model.js, weakness.js, rank-decomposition.js, game-logic.js,
// stats-compute.js, the stats chart modules), kept in one
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

// The error function: W. J. Cody's rational Chebyshev approximations (the
// algorithm math.js ships), in three pieces — a direct series for small |x|, and
// erfc for the middle and the tail, each split into exp(-ysq²)·exp(-del) with
// ysq truncated to sixteenths so the exponent is evaluated exactly.
const ERF_THRESH = 0.46875;
const ERF_SQRPI = 0.5641895835477563;
const ERF_P = [
    [3.1611237438705655, 113.86415415105016, 377.485237685302, 3209.3775891384694, 0.18577770618460315],
    [0.5641884969886701, 8.883149794388377, 66.11919063714163, 298.6351381974001, 881.952221241769,
     1712.0476126340707, 2051.0783778260716, 1230.3393547979972, 2.1531153547440383e-8],
    [0.30532663496123236, 0.36034489994980445, 0.12578172611122926, 0.016083785148742275,
     0.0006587491615298378, 0.016315387137302097],
];
const ERF_Q = [
    [23.601290952344122, 244.02463793444417, 1282.6165260773723, 2844.236833439171],
    [15.744926110709835, 117.6939508913125, 537.1811018620099, 1621.3895745666903, 3290.7992357334597,
     4362.619090143247, 3439.3676741437216, 1230.3393548037495],
    [2.568520192289822, 1.8729528499234604, 0.5279051029514285, 0.06051834131244132, 0.0023352049762686918],
];

function erf1(y) {
    const ysq = y * y;
    let xnum = ERF_P[0][4] * ysq, xden = ysq;
    for (let i = 0; i < 3; i++) {
        xnum = (xnum + ERF_P[0][i]) * ysq;
        xden = (xden + ERF_Q[0][i]) * ysq;
    }
    return y * (xnum + ERF_P[0][3]) / (xden + ERF_Q[0][3]);
}

function erfc2(y) {
    let xnum = ERF_P[1][8] * y, xden = y;
    for (let i = 0; i < 7; i++) {
        xnum = (xnum + ERF_P[1][i]) * y;
        xden = (xden + ERF_Q[1][i]) * y;
    }
    const result = (xnum + ERF_P[1][7]) / (xden + ERF_Q[1][7]);
    const ysq = Math.trunc(16 * y) / 16;
    const del = (y - ysq) * (y + ysq);
    return Math.exp(-ysq * ysq) * Math.exp(-del) * result;
}

function erfc3(y) {
    let ysq = 1 / (y * y);
    let xnum = ERF_P[2][5] * ysq, xden = ysq;
    for (let i = 0; i < 4; i++) {
        xnum = (xnum + ERF_P[2][i]) * ysq;
        xden = (xden + ERF_Q[2][i]) * ysq;
    }
    let result = ysq * (xnum + ERF_P[2][4]) / (xden + ERF_Q[2][4]);
    result = (ERF_SQRPI - result) / y;
    ysq = Math.trunc(16 * y) / 16;
    const del = (y - ysq) * (y + ysq);
    return Math.exp(-ysq * ysq) * Math.exp(-del) * result;
}

export function erf(x) {
    const y = Math.abs(x);
    if (y >= 2 ** 53) return Math.sign(x);
    if (y <= ERF_THRESH) return Math.sign(x) * erf1(y);
    if (y <= 4) return Math.sign(x) * (1 - erfc2(y));
    return Math.sign(x) * (1 - erfc3(y));
}

// Standard normal CDF.
export function normCdf(z) {
    return 0.5 * (1 + erf(z / Math.SQRT2));
}

// Inverse of a small dense square matrix (an array of rows) by Gauss-Jordan
// elimination with partial pivoting. Returns a new matrix; throws on a singular
// one, which callers treat as the fit having failed.
export function invertMatrix(a) {
    const n = a.length;
    const m = a.map((row, i) => {
        if (row.length !== n) throw new Error('invertMatrix: matrix is not square');
        const out = row.slice();
        for (let j = 0; j < n; j++) out.push(i === j ? 1 : 0);
        return out;
    });
    for (let c = 0; c < n; c++) {
        let pivot = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[pivot][c])) pivot = r;
        if (m[pivot][c] === 0) throw new Error('invertMatrix: matrix is singular');
        if (pivot !== c) [m[c], m[pivot]] = [m[pivot], m[c]];
        const inv = 1 / m[c][c];
        for (let j = 0; j < 2 * n; j++) m[c][j] *= inv;
        for (let r = 0; r < n; r++) {
            if (r === c) continue;
            const f = m[r][c];
            if (f === 0) continue;
            for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j];
        }
    }
    return m.map(row => row.slice(n));
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
