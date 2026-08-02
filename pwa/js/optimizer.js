// js/optimizer.js
//
// Shared Adam parameter update. The recall model (recall-model.js) and the weakness
// model (weakness.js) each ran their own hand-rolled, identical Adam inner
// loop; this extracts the bias-corrected update step so the two fitters share
// one optimizer. Each fitter still owns its own loss/gradient computation and
// its own regularization — only the per-parameter moment update lives here.

export const ADAM_DEFAULTS = { beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };

// In-place Adam step over every parameter. `state` is { m: [], v: [], t }; the
// caller increments state.t once per epoch BEFORE calling this (so bias
// correction uses the current step count). `grad` must already be the final,
// fully-accumulated gradient (mean-reduced, regularization folded in) — this
// function only applies the moment update.
export function adamUpdate(params, grad, state, learningRate, opts = ADAM_DEFAULTS) {
    const { beta1, beta2, epsilon } = opts;
    const bc1 = 1 - Math.pow(beta1, state.t);
    const bc2 = 1 - Math.pow(beta2, state.t);
    for (let j = 0; j < params.length; j++) {
        state.m[j] = beta1 * state.m[j] + (1 - beta1) * grad[j];
        state.v[j] = beta2 * state.v[j] + (1 - beta2) * grad[j] * grad[j];
        const mHat = state.m[j] / bc1;
        const vHat = state.v[j] / bc2;
        params[j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
    }
}
