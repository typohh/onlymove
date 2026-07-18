// js/weakness.js
//
// Augmented per-component weakness/strength tracking and scoring.
//
// Trains a logistic regression on the user's answer history for a given size:
//   P(correct) = sigmoid(α + γ·rating
//                          + Σ_k β_correct[k] · correct_emb[k]
//                          + Σ_k β_picked[k]  · picked_emb[k])
//
// β_correct[k] captures the effect of the *problem's* loading on axis k
// (treating the correct move's embedding as a property of the position).
// β_picked[k] captures the effect of the user's *picked* move along the
// same axis — independent of whether the picked move equals the correct
// move. For correct attempts, picked_emb = correct_emb by definition, so
// those rows inform β_correct + β_picked jointly. Wrong attempts with a
// picked-move embedding from the sidecar (or backfilled later) let the
// model disambiguate the two.
//
// Recommendation scoring of a candidate problem uses the CONTRAST
// β_correct − β_picked applied to the candidate's correct-move embedding
// (see weaknessScore for the full rationale). That contrast is the identified
// over/under weakness direction — the same thing the Weaknesses pane's level-2
// sign test estimates — so a correct-move-only score becomes direction-aware:
// it up-weights motifs the user underplays / is bad at and down-weights motifs
// the user overplays, with no thresholds.
//
// Model ridge-regularized: β shrunk toward 0, so before there's enough
// signal the scorer returns 0 (cold-start, ≤ MIN_TRAINING_ATTEMPTS).

import * as Sentry from "@sentry/browser";
import { state, getMergedAnswerHistory, clearWeaknessModel } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { sigmoid } from './math-utils.js';
import { adamUpdate } from './optimizer.js';
// Baked into the bundle at build time; population-constants.js is the validated
// loader for the rest of this blob. We read only the weakness ridge prior here.
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

// Recommender temperature for weight = quality * exp(T * weaknessScore).
// Higher T = more aggressive mining of weak directions. Owned here because
// any reporting/analytics that wants to mirror the recommender's behavior
// (expected logit penalty etc.) needs the same constant.
export const WEAKNESS_TEMPERATURE = 0.5;

const MIN_TRAINING_ATTEMPTS = 20;
// Prior precision on each per-PC beta_k (beta_k ~ N(0, 1/lambda_k)):
// weakness_axis_lambda — a per-axis vector fit by empirical Bayes in
// pipeline.fit_population_constants (the between-user variance tau_k^2 of each
// axis's weakness coefficient, shipped as lambda_k = 2/tau_k^2 and applied to
// BOTH beta_correct[k] and beta_picked[k], so the prior on the identified sum
// beta_correct + beta_picked is exactly N(0, tau_k^2); axes users genuinely
// differ along get a loose prior, no-signal axes stay pinned). It's inlined
// from population_constants.json at build time (main.js aborts boot if that's
// missing) and is MANDATORY — the pipeline ships this vector or fails, so there
// is no scalar fallback. If it's ever absent or its length doesn't match the
// embedding K here (only a transient cache skew), we skip training rather than
// fabricate a prior.
const AXIS_LAMBDA = POPULATION_CONSTANTS.weakness_axis_lambda;
const MAX_EPOCHS = 200;
const MAX_TRAINING_TIME_MS = 1000;
const LEARNING_RATE = 0.05;
const LOSS_CONVERGENCE = 1e-6;

let trainingInProgress = false;

// Train the single shared weakness model on the union of both sizes' answer
// histories. Every per-problem feature it reads (problemRank, the joint-ICA
// embedding axes) is board-independent, so pooling is on one common scale.
// Called after each attempt and once after backfill.
export async function trainWeakness() {
    if (trainingInProgress) return;
    trainingInProgress = true;
    try {
        const allHistory = getMergedAnswerHistory();
        const history = allHistory
            .filter(a => !a.isRepeat &&
                         a.embedding && Array.isArray(a.embedding) &&
                         typeof a.problemRank === 'number' &&
                         typeof a.correct === 'boolean')
            .slice(-MODEL_INPUT_HISTORY_LENGTH);
        if (history.length < MIN_TRAINING_ATTEMPTS) return;

        const K = history[0].embedding.length;
        if (history.some(a => a.embedding.length !== K)) {
            console.warn(`weakness: inconsistent embedding lengths; skipping`);
            return;
        }

        // Per-coefficient prior precision from the mandatory empirical-Bayes
        // per-axis vector; lambda_k applies to both the beta_correct[k] and
        // beta_picked[k] slots of axis k. It's absent/mismatched only on a
        // transient cache skew (stale K) — skip training until it matches rather
        // than fall back to a fabricated prior (there is no scalar fallback).
        const axisLambda = (Array.isArray(AXIS_LAMBDA) && AXIS_LAMBDA.length === K &&
            AXIS_LAMBDA.every(v => Number.isFinite(v) && v > 0)) ? AXIS_LAMBDA : null;
        if (!axisLambda) {
            console.warn(`weakness: no valid per-axis prior for K=${K}; skipping training`);
            return;
        }
        const lambdaForAxis = (k) => axisLambda[k];

        // Augmented features: [1, rating, correct_emb_0..K-1, picked_emb_0..K-1].
        // For correct attempts picked = correct (definition); for wrong attempts
        // we use the sidecar-fetched pickedEmbedding when available, falling
        // back to correct_emb if not. The fallback rows only inform the SUM
        // (β_correct + β_picked) — the model gracefully degrades to a
        // diff-form mean estimate when no picked data exists.
        const numParams = 2 + 2 * K;
        const existingParams = state.weaknessParams;
        let params = (Array.isArray(existingParams) && existingParams.length === numParams)
            ? [...existingParams]
            : new Array(numParams).fill(0);
        let adam = { m: new Array(numParams).fill(0), v: new Array(numParams).fill(0), t: 0 };

        // Precompute the picked vector per row to avoid re-deriving in the
        // hot training loop.
        const pickedFor = history.map(a => {
            if (Array.isArray(a.pickedEmbedding) && a.pickedEmbedding.length === K) {
                return a.pickedEmbedding;
            }
            return a.embedding;  // fallback: row only informs β_correct + β_picked
        });

        const N = history.length;
        const startTime = Date.now();
        let prevLoss = Infinity;

        for (let epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            adam.t++;
            const grad = new Array(numParams).fill(0);
            let loss = 0;

            for (let i = 0; i < N; i++) {
                const a = history[i];
                const corr = a.embedding;
                const pick = pickedFor[i];
                let z = params[0] + params[1] * a.problemRank;
                for (let k = 0; k < K; k++) z += params[2 + k] * corr[k];
                for (let k = 0; k < K; k++) z += params[2 + K + k] * pick[k];

                const p = sigmoid(z);
                const clipped = Math.max(1e-15, Math.min(1 - 1e-15, p));
                const y = a.correct ? 1 : 0;
                loss += -y * Math.log(clipped) - (1 - y) * Math.log(1 - clipped);
                const err = p - y;
                grad[0] += err;
                grad[1] += err * a.problemRank;
                for (let k = 0; k < K; k++) grad[2 + k] += err * corr[k];
                for (let k = 0; k < K; k++) grad[2 + K + k] += err * pick[k];
            }

            let l2 = 0;
            // Ridge over all β coefficients (correct and picked alike), not
            // the intercept or rating. lambdaForAxis(k) hits both of axis k's
            // coefficient slots — the per-axis empirical-Bayes prior when
            // shipped, the scalar CV lambda otherwise.
            for (let k = 0; k < K; k++) {
                const lam = lambdaForAxis(k);
                l2 += lam * (params[2 + k] * params[2 + k] +
                             params[2 + K + k] * params[2 + K + k]);
            }
            loss = loss / N + l2 / (2 * N);

            // Finalize the gradient (mean-reduce + ridge on the β block, j ≥ 2),
            // then apply the shared Adam moment update.
            for (let j = 0; j < numParams; j++) grad[j] /= N;
            for (let k = 0; k < K; k++) {
                const lam = lambdaForAxis(k);
                grad[2 + k] += (lam / N) * params[2 + k];
                grad[2 + K + k] += (lam / N) * params[2 + K + k];
            }
            adamUpdate(params, grad, adam, LEARNING_RATE);

            if (Math.abs(prevLoss - loss) < LOSS_CONVERGENCE) break;
            if (Date.now() - startTime > MAX_TRAINING_TIME_MS) break;
            prevLoss = loss;
        }

        if (!params.every(p => Number.isFinite(p))) {
            console.warn(`weakness: non-finite params; resetting`);
            clearWeaknessModel();
            return;
        }
        state.weaknessParams = params;
        // (adam is NOT persisted: every retrain starts fresh moments at line
        // ~110, so storing it was a dead localStorage write per attempt. The
        // clear/wipe paths still remove legacy weaknessAdam blobs.)

        // Laplace-approx posterior cov. H = X' diag(p(1-p)) X + diag(lam_vec);
        // posterior cov ≈ H^-1. We pull out the per-axis 2x2 block on
        // (β_correct[k], β_picked[k]) so the stats panel can compute the
        // posterior of the identified SUM β_correct + β_picked per axis
        // (var_c + var_p + 2·cov) — the Weaknesses pane's level-1 detection
        // (stats-weaknesses.js). Storing the full (2 + 2K)x(2 + 2K) matrix
        // would blow localStorage at K=64; the diagonal (weaknessSe) +
        // per-axis off-diagonals (weaknessAxisCov) is sufficient for that.
        try {
            const X = history.map((a, i) => {
                const row = new Array(numParams);
                row[0] = 1;
                row[1] = a.problemRank;
                const corr = a.embedding;
                const pick = pickedFor[i];
                for (let k = 0; k < K; k++) row[2 + k] = corr[k];
                for (let k = 0; k < K; k++) row[2 + K + k] = pick[k];
                return row;
            });
            const pHat = history.map((a, i) => {
                const corr = a.embedding;
                const pick = pickedFor[i];
                let z = params[0] + params[1] * a.problemRank;
                for (let k = 0; k < K; k++) z += params[2 + k] * corr[k];
                for (let k = 0; k < K; k++) z += params[2 + K + k] * pick[k];
                return sigmoid(z);
            });
            const lamVec = new Array(numParams).fill(1e-6);  // tiny guard on intercept/rating
            for (let k = 0; k < K; k++) {
                lamVec[2 + k] = lambdaForAxis(k);
                lamVec[2 + K + k] = lambdaForAxis(k);
            }
            const W = math.diag(pHat.map(p => p * (1 - p)));
            const Xm = math.matrix(X);
            const fisher = math.multiply(math.transpose(Xm), W, Xm);
            const H = math.add(fisher, math.diag(lamVec));
            const cov = math.inv(H);
            const diag = math.diag(cov).toArray();
            const se = diag.map(v => Math.sqrt(Math.max(v, 1e-12)));
            if (se.every(s => Number.isFinite(s))) {
                state.weaknessSe = se;
            } else {
                state.weaknessSe = null;
            }

            // Off-diagonal cov[β_c[k], β_p[k]] per axis. cov is a math.js
            // Matrix; .get([i, j]) is the cheap path. Stored as flat length-K
            // array — only the cross terms inside each axis k.
            const covArr = cov.toArray();
            const axisCov = new Array(K);
            for (let k = 0; k < K; k++) {
                const v = covArr[2 + k][2 + K + k];
                axisCov[k] = Number.isFinite(v) ? v : 0;
            }
            state.weaknessAxisCov = axisCov;
        } catch (err) {
            console.warn(`weakness: SE/cov computation failed`, err);
            state.weaknessSe = null;
            state.weaknessAxisCov = null;
        }
    } catch (err) {
        console.error(`weakness training failed`, err);
        try { Sentry.captureException(err); } catch (_) {}
    } finally {
        trainingInProgress = false;
    }
}

// Returns alignment with user's weakness directions for a candidate problem.
// Positive = mine for this; negative = user is strong here. 0 if no model.
//
// Scores the CONTRAST β_correct − β_picked against the candidate's correct-move
// embedding. β_correct is the position-difficulty coefficient (effect of the
// correct move's motif) and β_picked is the move-SELECTION coefficient (effect
// of the move the user actually picked). Their contrast is the model-identified
// over/under direction of the weakness — the continuous analogue of the
// Weaknesses pane's level-2 picked−correct sign test — and it sets the tilt on
// a candidate whose correct move loads on motif k:
//   - underplay / bad-at a motif → β_correct < 0, β_picked ≥ 0 → contrast < 0
//     → up-weight (drill finding / handling the motif);
//   - overplay a motif → β_correct > 0, β_picked < 0 → contrast > 0
//     → down-weight (stop rewarding the instinct).
//
// Equivalently this treats the "generic wrong move" as −correct_emb (to first
// order the user, when wrong, plays away from the correct move along the weak
// axis), which is candidate-varying and so — unlike a fixed generic move —
// actually informs selection. β_picked must enter with THIS sign: the old sum
// β_correct + β_picked applied β_picked to the correct move directly, a category
// error that inverts the overplay case (it up-weights the very motif-k-correct
// problems an overplayer should see LESS of). The per-axis empirical-Bayes prior
// shrinks β_picked toward 0 with no picking signal, so the contrast degrades
// gracefully to β_correct alone when there's no direction to read.
// Whether the shared weakness model has trained parameters at all — the
// two-regime selector falls back to the information (slope²) regime when this
// is false, rather than sampling uniformly over flat zero scores.
export function weaknessModelReady() {
    return Array.isArray(state.weaknessParams) && state.weaknessParams.length >= 4;
}

export function weaknessScore(problem) {
    const params = state.weaknessParams;  // shared across sizes
    if (!params || params.length < 4) return 0;
    const K = (params.length - 2) / 2;
    if (!Number.isInteger(K) || K <= 0) return 0;
    if (!problem.embedding || problem.embedding.length !== K) return 0;
    let score = 0;
    for (let k = 0; k < K; k++) {
        // β_correct[k] − β_picked[k]: the identified over/under weakness
        // direction (see the note above on why β_picked enters with a MINUS,
        // not the inverted + of the old sum).
        const betaEff = params[2 + k] - params[2 + K + k];
        score += -betaEff * problem.embedding[k];
    }
    return score;
}

