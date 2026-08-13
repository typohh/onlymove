// js/optimizer.js
//
// Shared damped-Newton step. The recall model (recall-model.js) and the weakness
// model (weakness.js) both fit a small penalized likelihood — a handful of
// parameters over a bounded history window — so they step with curvature rather
// than with a first-order rule. Each fitter still owns its loss, gradient and
// curvature; only the linear solve, the backtracking accept/reject and the
// convergence measure live here.
//
// Curvature is not a speed optimization here, it is what makes the fit land on
// the quantity each model is READ for. Both objectives are anisotropic in the
// direction that matters: the weakness scorer reads the correct-picked contrast,
// which the correct/picked collinearity makes the flattest direction of the
// likelihood, and the recall model's difficulty multiplier is prior-dominated.
// A diagonally-preconditioned first-order method converges slowest along exactly
// those directions, and a stopping rule watching the loss is least sensitive to
// them — the loss is flat there, which is the whole point. A Newton step is
// scaled by the inverse curvature, so it is equally exact in the flat and the
// steep directions and the fit is a deterministic function of the history rather
// than of where an iteration budget ran out.

// Armijo sufficient-decrease constant, and the number of halvings before a step
// is abandoned. Both are the textbook defaults; the fitters here are penalized
// and well-conditioned, so a full step is accepted essentially always and the
// backtrack is a guard, not a working part of the algorithm.
const ARMIJO_C = 1e-4;
const MAX_BACKTRACKS = 30;

// Solve A·x = b by Gauss-Jordan elimination with partial pivoting. A is a small
// dense array-of-rows (order = the block size, never more than a few dozen), so
// the cubic cost is irrelevant next to the per-row passes the caller makes.
// Returns null on a singular pivot rather than propagating Infinity/NaN into the
// parameters — the caller treats that as "leave the model alone".
export function solveLinear(A, b) {
    const n = b.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
        let piv = c;
        for (let r = c + 1; r < n; r++) {
            if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
        }
        if (!(Math.abs(M[piv][c]) > 0)) return null;
        const tmp = M[c]; M[c] = M[piv]; M[piv] = tmp;
        for (let r = 0; r < n; r++) {
            if (r === c) continue;
            const f = M[r][c] / M[c][c];
            for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
        }
    }
    const x = new Array(n);
    for (let i = 0; i < n; i++) x[i] = M[i][n] / M[i][i];
    return x.every(Number.isFinite) ? x : null;
}

// One damped Newton step over `indices` — a BLOCK of params, so a model whose
// objective separates (recall: the correctness head owns the logit params, the
// retention head owns the base-stability one) can step each block against the
// loss that block actually owns.
//
// `grad` and `hess` are the block's gradient and curvature at the current
// params, in whatever scaling the caller uses for `lossFn` — raw sums or
// per-attempt means, but the same one for all three. `lossFn()` is evaluated at
// the current (mutated in place) params. On return the params hold the accepted
// point, or the starting point when no step was accepted.
//
// Returns { decrement, accepted }: `decrement` is the Newton decrement gᵀH⁻¹g at
// the STARTING point, i.e. twice the loss reduction a quadratic model predicts
// the step will make, and is the convergence measure the caller should watch —
// it is a curvature-weighted gradient norm, so unlike a loss delta it does not
// go small merely because the objective is flat. `accepted` is false when the
// solve was singular or every backtrack failed.
export function newtonBlockStep(params, indices, grad, hess, lossFn) {
    const step = solveLinear(hess, grad);
    if (step === null) return { decrement: 0, accepted: false };

    let decrement = 0;
    for (let j = 0; j < indices.length; j++) decrement += grad[j] * step[j];
    // A non-positive decrement means the curvature was not positive definite in
    // this direction; the step is not guaranteed to descend, so don't take it.
    if (!(decrement > 0)) return { decrement: 0, accepted: false };

    const loss0 = lossFn();
    const start = indices.map(j => params[j]);
    let t = 1;
    for (let tries = 0; tries < MAX_BACKTRACKS; tries++) {
        for (let j = 0; j < indices.length; j++) params[indices[j]] = start[j] - t * step[j];
        const loss = lossFn();
        if (Number.isFinite(loss) && loss <= loss0 - ARMIJO_C * t * decrement) {
            return { decrement, accepted: true };
        }
        t /= 2;
    }
    for (let j = 0; j < indices.length; j++) params[indices[j]] = start[j];
    return { decrement, accepted: false };
}
