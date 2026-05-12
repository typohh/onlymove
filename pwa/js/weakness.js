// js/weakness.js
//
// Per-component weakness/strength tracking and scoring.
//
// Trains a logistic regression on the user's answer history for a given size:
//   P(correct) = sigmoid(intercept + gamma * rating + sum_k beta_k * embedding_k)
//
// beta_k > 0 means "user does relatively better on high-component-k problems"
// (a strength). beta_k < 0 means a weakness in that direction. The recommender
// scores a candidate by sum_k (-beta_k) * embedding_k(candidate) — bigger =
// more aligned with the user's weaknesses, more aligned away from strengths.
//
// The model is ridge-regularized: beta is shrunk toward 0, so before there's
// enough signal it produces near-uniform scores. Cold start is handled by
// MIN_TRAINING_ATTEMPTS — below that, no model trains and the scorer returns 0.

import * as Sentry from "@sentry/browser";
import {
    state,
    getAnswerHistoryForSize, getWeaknessParamsForSize,
    setWeaknessParamsForSize, setWeaknessAdamForSize, setWeaknessSeForSize,
} from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';

// Recommender temperature for weight = quality * exp(T * weaknessScore).
// Higher T = more aggressive mining of weak directions. Owned here because
// any reporting/analytics that wants to mirror the recommender's behavior
// (expected logit penalty etc.) needs the same constant.
export const WEAKNESS_TEMPERATURE = 0.5;

const MIN_TRAINING_ATTEMPTS = 20;
// Prior precision on each per-PC beta_k. Calibrated against the production
// attempts DB by held-out CV (auxiliary/calibrate_priors.py): the basin of
// equivalent NLL spans 100-300; 200 is the point estimate. Translates to
// beta_k ~ N(0, 1/200), i.e. prior sigma ~0.07.
const RIDGE_LAMBDA = 200.0;
const MAX_EPOCHS = 200;
const MAX_TRAINING_TIME_MS = 1000;
const LEARNING_RATE = 0.05;
const LOSS_CONVERGENCE = 1e-6;

const trainingInProgress = new Set();

function sigmoid(z) {
    if (z >= 0) {
        return 1 / (1 + Math.exp(-z));
    }
    const e = Math.exp(z);
    return e / (1 + e);
}

// Train for the currently-active size (called after each attempt).
export async function trainWeakness() {
    return trainWeaknessForSize(state.boardSize);
}

// Train for an arbitrary size (called from backfill so non-active sizes
// also get a calibrated model on app load).
export async function trainWeaknessForSize(size) {
    if (trainingInProgress.has(size)) return;
    trainingInProgress.add(size);
    try {
        const allHistory = getAnswerHistoryForSize(size);
        const history = allHistory
            .filter(a => !a.isRepeat &&
                         a.embedding && Array.isArray(a.embedding) &&
                         typeof a.problemRating === 'number' &&
                         typeof a.correct === 'boolean')
            .slice(-MODEL_INPUT_HISTORY_LENGTH);
        if (history.length < MIN_TRAINING_ATTEMPTS) return;

        const K = history[0].embedding.length;
        if (history.some(a => a.embedding.length !== K)) {
            console.warn(`weakness[${size}]: inconsistent embedding lengths; skipping`);
            return;
        }

        const numParams = 2 + K;
        const existingParams = getWeaknessParamsForSize(size);
        let params = (Array.isArray(existingParams) && existingParams.length === numParams)
            ? [...existingParams]
            : new Array(numParams).fill(0);
        let adam = { m: new Array(numParams).fill(0), v: new Array(numParams).fill(0), t: 0 };

        const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        const N = history.length;
        const startTime = Date.now();
        let prevLoss = Infinity;

        for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            adam.t++;
            const grad = new Array(numParams).fill(0);
            let loss = 0;

            for (let i = 0; i < N; i++) {
                const a = history[i];
                let z = params[0] + params[1] * a.problemRating;
                for (let k = 0; k < K; k++) z += params[2 + k] * a.embedding[k];

                const p = sigmoid(z);
                const clipped = Math.max(1e-15, Math.min(1 - 1e-15, p));
                const y = a.correct ? 1 : 0;
                loss += -y * Math.log(clipped) - (1 - y) * Math.log(1 - clipped);
                const err = p - y;
                grad[0] += err;
                grad[1] += err * a.problemRating;
                for (let k = 0; k < K; k++) grad[2 + k] += err * a.embedding[k];
            }

            let l2 = 0;
            for (let k = 0; k < K; k++) l2 += params[2 + k] * params[2 + k];
            // Mean-normalized NLL + L2 also divided by N. Equivalent to the
            // standard MAP objective (sum_NLL + (RIDGE_LAMBDA/2) * ||beta||^2)
            // up to a global 1/N factor, so the prior precision on each beta_k
            // is RIDGE_LAMBDA regardless of history size — matching what
            // calibrate_priors.py optimized. Without the /N on the L2 term,
            // effective prior precision scales with N and over-shrinks.
            loss = loss / N + (RIDGE_LAMBDA / (2 * N)) * l2;

            for (let j = 0; j < numParams; j++) {
                grad[j] /= N;
                if (j >= 2) grad[j] += (RIDGE_LAMBDA / N) * params[j];
                adam.m[j] = beta1 * adam.m[j] + (1 - beta1) * grad[j];
                adam.v[j] = beta2 * adam.v[j] + (1 - beta2) * grad[j] * grad[j];
                const mHat = adam.m[j] / (1 - Math.pow(beta1, adam.t));
                const vHat = adam.v[j] / (1 - Math.pow(beta2, adam.t));
                params[j] -= LEARNING_RATE * mHat / (Math.sqrt(vHat) + eps);
            }

            if (Math.abs(prevLoss - loss) < LOSS_CONVERGENCE) break;
            if (Date.now() - startTime > MAX_TRAINING_TIME_MS) break;
            prevLoss = loss;
        }

        if (!params.every(p => Number.isFinite(p))) {
            console.warn(`weakness[${size}]: non-finite params; resetting`);
            setWeaknessParamsForSize(size, null);
            setWeaknessAdamForSize(size, null);
            setWeaknessSeForSize(size, null);
            return;
        }
        setWeaknessParamsForSize(size, params);
        setWeaknessAdamForSize(size, adam);

        // Laplace-approx posterior std per coefficient. Hessian at the MAP is
        // H = X' diag(p(1-p)) X + diag(lam_vec); posterior cov ~= H^-1.
        // Consumed by model.js to display P(weakness) = Phi(-beta_k / se_k).
        try {
            const X = history.map(a => {
                const row = new Array(numParams);
                row[0] = 1;
                row[1] = a.problemRating;
                for (let k = 0; k < K; k++) row[2 + k] = a.embedding[k];
                return row;
            });
            const pHat = history.map(a => {
                let z = params[0] + params[1] * a.problemRating;
                for (let k = 0; k < K; k++) z += params[2 + k] * a.embedding[k];
                return sigmoid(z);
            });
            const lamVec = new Array(numParams).fill(1e-6);  // tiny guard on intercept/rating
            for (let k = 0; k < K; k++) lamVec[2 + k] = RIDGE_LAMBDA;
            const W = math.diag(pHat.map(p => p * (1 - p)));
            const Xm = math.matrix(X);
            const fisher = math.multiply(math.transpose(Xm), W, Xm);
            const H = math.add(fisher, math.diag(lamVec));
            const cov = math.inv(H);
            const diag = math.diag(cov).toArray();
            const se = diag.map(v => Math.sqrt(Math.max(v, 1e-12)));
            if (se.every(s => Number.isFinite(s))) {
                setWeaknessSeForSize(size, se);
            } else {
                setWeaknessSeForSize(size, null);
            }
        } catch (err) {
            console.warn(`weakness[${size}]: SE computation failed`, err);
            setWeaknessSeForSize(size, null);
        }
    } catch (err) {
        console.error(`weakness[${size}] training failed`, err);
        try { Sentry.captureException(err); } catch (_) {}
    } finally {
        trainingInProgress.delete(size);
    }
}

// Returns alignment with user's weakness directions for a candidate problem.
// Positive = mine for this; negative = user is strong here. 0 if no model.
export function weaknessScore(problem) {
    const params = state.weaknessParams;  // active size
    if (!params) return 0;
    const K = params.length - 2;
    if (!problem.embedding || problem.embedding.length !== K) return 0;
    let score = 0;
    for (let k = 0; k < K; k++) score += -params[2 + k] * problem.embedding[k];
    return score;
}

// Expected logit penalty per recommender pick, under the same weighting the
// recommender uses to select problems. Concretely:
//
//   weight_i ∝ quality_i · exp(WEAKNESS_TEMPERATURE · weaknessScore_i)
//             = quality_i · exp(-WEAKNESS_TEMPERATURE · (β · embedding_i))
//   E[β · embedding] = Σ weight_i · (β · embedding_i) / Σ weight_i
//
// Because exp(-T · dot) up-weights problems with negative dot (= weakness-
// aligned), the expectation is ≤ 0 whenever the model has any signal. The
// returned `expectedLogit` is meant to be presented as a *drag* on success
// rate — "the recommender systematically exposes you to problems this much
// harder than rating alone would predict."
//
// Returns null when there's no trained model or no candidate pool to
// evaluate (which can happen on a fresh session before zips load).
export function computeExpectedRecommenderPenalty() {
    const params = state.weaknessParams;
    if (!params || params.length < 3) return null;
    const K = params.length - 2;
    const betas = params.slice(2);

    // Prefer the in-memory candidate pool — that's what the recommender
    // actually picks from. On initial loads this is empty (loadTsumego
    // short-circuits to a cached problem without fetching a zip), so fall
    // back to the user's embedded answerHistory entries: same rating range,
    // same shape of distribution.
    let candidates = (state.allProblems || []).filter(
        p => Array.isArray(p.embedding) && p.embedding.length === K
    );
    if (candidates.length < 5) {
        const fallback = (state.answerHistory || [])
            .filter(a => !a.isRepeat &&
                         Array.isArray(a.embedding) && a.embedding.length === K)
            .slice(-MODEL_INPUT_HISTORY_LENGTH);
        candidates = fallback.map(a => ({
            embedding: a.embedding,
            quality: typeof a.problemQuality === "number" ? a.problemQuality : 1.0,
        }));
    }
    if (candidates.length < 5) return null;

    let logZ = -Infinity;
    const dots = new Array(candidates.length);
    const logWeights = new Array(candidates.length);
    let unifSum = 0;
    for (let i = 0; i < candidates.length; i++) {
        const e = candidates[i].embedding;
        let s = 0;
        for (let k = 0; k < K; k++) s += betas[k] * e[k];
        dots[i] = s;
        unifSum += s;
        const q = Math.max(1e-9, candidates[i].quality || 1e-9);
        logWeights[i] = Math.log(q) + (-WEAKNESS_TEMPERATURE * s);
        if (logWeights[i] > logZ) logZ = logWeights[i];
    }
    let zSum = 0, dotSum = 0, dotSqSum = 0;
    for (let i = 0; i < candidates.length; i++) {
        const w = Math.exp(logWeights[i] - logZ);
        zSum += w;
        dotSum += w * dots[i];
        dotSqSum += w * dots[i] * dots[i];
    }
    if (!(zSum > 0)) return null;
    const meanRec = dotSum / zSum;
    const meanUnif = unifSum / candidates.length;
    // Drag = recommender-weighted mean minus uniform-weighted mean. Always
    // <= 0 for any monotone exp(-T·X) weight, since X and exp(-T·X) are
    // negatively correlated. This isolates the recommender's effect from
    // the candidate pool's natural skew.
    const drag = meanRec - meanUnif;
    const variance = Math.max(0, dotSqSum / zSum - meanRec * meanRec);
    return {
        n: candidates.length,
        expectedLogit: drag,
        stdLogit: Math.sqrt(variance),
    };
}

