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
import { newtonBlockStep } from './optimizer.js';
// Baked into the bundle at build time; population-constants.js is the validated
// loader for the rest of this blob. We read only the weakness ridge prior here.
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

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
// Population mean of the CONTRAST β_correct − β_picked, per axis, from the same
// empirical-Bayes pass as AXIS_LAMBDA (pipeline.fit_population_constants). Scoring
// subtracts it; see weaknessScore for why the contrast — unlike the sum — is centred
// on a shipped mean rather than on zero. Validated at boot by
// population-constants.js, which is what lets weaknessScore index it unguarded.
const AXIS_CONTRAST_MEAN = POPULATION_CONSTANTS.weakness_contrast_mean;
// Population centre of the identified SUM (beta_correct[k] + beta_picked[k]).
// The prior is centred HERE rather than at zero: an axis may legitimately carry
// a population offset — people differ in how sharply a problem's discrimination
// affects them, around a mean that need not be zero — and pinning the prior at
// zero would both inflate tau^2 by mu^2 (a looser prior than the data asks for)
// and make the level-one test in stats-weaknesses.js measure every user against
// a point the typical user does not occupy, turning the population's own centre
// into a per-user finding. Mandatory and length-checked like the precision
// vector; a missing or stale entry skips training rather than substituting zero,
// because zero is exactly the wrong answer when the true centre is not zero.
export const AXIS_SUM_MEAN = POPULATION_CONSTANTS.weakness_axis_mean;
// Newton iteration budget. The fit converges quadratically from a cold start —
// measured at 4 iterations to a numerically exact optimum across N = 50..500 —
// so the cap is a runaway guard, not a budget the fit is expected to spend, and
// there is deliberately no wall-clock cap beside it: a step costs two passes
// over the window plus a solve of order (2K+2), which is milliseconds.
const MAX_NEWTON_ITERS = 50;
// Convergence on the Newton decrement (twice the loss reduction the quadratic
// model predicts, in nats over the whole window). This is what replaces a
// loss-delta threshold: the decrement is curvature-weighted, so it does not go
// small in the flat contrast direction the scorer reads while the gradient there
// is still large. Quadratic convergence makes the tolerance nearly free.
const NEWTON_TOL = 1e-8;
// Numerical guard on the two unpenalized diagonal entries (intercept and rank).
// Same value the posterior-covariance step has always used, now shared with the
// step that produces the parameters it describes: a short all-correct history is
// separable, and without it the curvature for the intercept vanishes as the fit
// saturates and the solve has no pivot.
const RIDGE_GUARD = 1e-6;

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
        // Half the sum's centre per slot: the penalty below pulls each of the two
        // slots toward it, so their SUM is pulled toward the population centre
        // while their CONTRAST is still pulled toward zero — which is what the
        // separately shipped contrast centre handles.
        const axisSumMean = (Array.isArray(AXIS_SUM_MEAN) &&
            AXIS_SUM_MEAN.length === K &&
            AXIS_SUM_MEAN.every(Number.isFinite)) ? AXIS_SUM_MEAN : null;
        if (!axisSumMean) {
            console.warn(`weakness: no valid per-axis sum centre for K=${K}; skipping training`);
            return;
        }
        const halfMean = (k) => axisSumMean[k] / 2;

        // Augmented features: [1, rating, correct_emb_0..K-1, picked_emb_0..K-1].
        // For correct attempts picked = correct (definition); for wrong attempts
        // we use the sidecar-fetched pickedEmbedding when available, falling
        // back to correct_emb if not. The fallback rows only inform the SUM
        // (β_correct + β_picked) — the model gracefully degrades to a
        // diff-form mean estimate when no picked data exists.
        const numParams = 2 + 2 * K;
        const N = history.length;

        // Precompute the picked vector per row to avoid re-deriving it in the
        // per-row passes below.
        const pickedFor = history.map(a => {
            if (Array.isArray(a.pickedEmbedding) && a.pickedEmbedding.length === K) {
                return a.pickedEmbedding;
            }
            return a.embedding;  // fallback: row only informs β_correct + β_picked
        });

        // Design matrix, built once. The gradient, the curvature and the final
        // posterior covariance all read the same rows, so the covariance
        // describes exactly the fit that produced the parameters.
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
        const yObs = history.map(a => (a.correct ? 1 : 0));

        // Ridge over all β coefficients (correct and picked alike): lambdaForAxis(k)
        // hits both of axis k's coefficient slots, pulling each toward half the
        // sum's population centre, so their SUM is pulled toward that centre while
        // their CONTRAST is still pulled toward zero. The intercept and rank slots
        // carry only the numerical guard.
        const lamVec = new Array(numParams).fill(RIDGE_GUARD);
        const priorMean = new Array(numParams).fill(0);
        for (let k = 0; k < K; k++) {
            lamVec[2 + k] = lamVec[2 + K + k] = lambdaForAxis(k);
            priorMean[2 + k] = priorMean[2 + K + k] = halfMean(k);
        }

        // Everything below is in RAW SUMS over the window (not per-attempt means),
        // the scaling the posterior covariance is defined in — H is then the
        // penalized observed information directly, with no rescaling between the
        // step and the covariance read off it.
        const probs = (p) => {
            const out = new Array(N);
            for (let i = 0; i < N; i++) {
                const row = X[i];
                let z = 0;
                for (let j = 0; j < numParams; j++) z += p[j] * row[j];
                out[i] = sigmoid(z);
            }
            return out;
        };
        const lossAt = (p) => {
            let loss = 0;
            const pr = probs(p);
            for (let i = 0; i < N; i++) {
                const clipped = Math.max(1e-15, Math.min(1 - 1e-15, pr[i]));
                loss += -yObs[i] * Math.log(clipped) - (1 - yObs[i]) * Math.log(1 - clipped);
            }
            for (let j = 0; j < numParams; j++) {
                const d = p[j] - priorMean[j];
                loss += 0.5 * lamVec[j] * d * d;
            }
            return loss;
        };
        const gradAt = (p) => {
            const grad = new Array(numParams).fill(0);
            const pr = probs(p);
            for (let i = 0; i < N; i++) {
                const err = pr[i] - yObs[i];
                const row = X[i];
                for (let j = 0; j < numParams; j++) grad[j] += err * row[j];
            }
            for (let j = 0; j < numParams; j++) grad[j] += lamVec[j] * (p[j] - priorMean[j]);
            return grad;
        };
        // H = X' diag(p(1-p)) X + diag(lam_vec) — the exact Hessian of the
        // penalized objective (logistic regression's observed and expected
        // information coincide), so it is both the Newton curvature and, at the
        // optimum, the inverse Laplace posterior covariance.
        const hessAt = (p) => {
            const H = Array.from({ length: numParams }, () => new Array(numParams).fill(0));
            const pr = probs(p);
            for (let i = 0; i < N; i++) {
                const w = pr[i] * (1 - pr[i]);
                const row = X[i];
                for (let a = 0; a < numParams; a++) {
                    const wa = w * row[a];
                    if (wa === 0) continue;
                    for (let b = a; b < numParams; b++) H[a][b] += wa * row[b];
                }
            }
            for (let a = 0; a < numParams; a++) {
                H[a][a] += lamVec[a];
                for (let b = a + 1; b < numParams; b++) H[b][a] = H[a][b];
            }
            return H;
        };

        // Newton from zero rather than from the stored parameters: the fit
        // converges in a handful of steps either way, and starting cold makes it
        // a pure function of the history — the same window gives the same model
        // on every device, with no path dependence on how the previous fit ended.
        const allIndices = Array.from({ length: numParams }, (_, j) => j);
        const params = new Array(numParams).fill(0);
        let steps = 0;
        let stopReason = 'maxIters';
        for (let iter = 0; iter < MAX_NEWTON_ITERS; iter++) {
            const { decrement, accepted } =
                newtonBlockStep(params, allIndices, gradAt(params), hessAt(params), () => lossAt(params));
            if (!accepted) { stopReason = 'stalled'; break; }
            steps++;
            if (decrement / 2 < NEWTON_TOL) { stopReason = 'converged'; break; }
        }
        if (steps === 0) {
            // Singular curvature or a step that could not descend — the window
            // says nothing usable. Leave whatever model is already stored alone
            // rather than overwriting it with an unfitted one.
            console.warn(`weakness: no usable Newton step; keeping the existing model`);
            return;
        }
        if (stopReason !== 'converged') {
            console.warn(`weakness: fit ${stopReason} after ${steps} Newton steps`);
        }

        if (!params.every(p => Number.isFinite(p))) {
            console.warn(`weakness: non-finite params; resetting`);
            clearWeaknessModel();
            return;
        }
        state.weaknessParams = params;

        // Laplace-approx posterior cov = H^-1 at the optimum — which is where the
        // fit now actually is, so the SE describes the mode it is quoted for. We
        // pull out the per-axis 2x2 block on (β_correct[k], β_picked[k]) so the
        // stats panel can compute the posterior of the identified SUM
        // β_correct + β_picked per axis (var_c + var_p + 2·cov) — the Weaknesses
        // pane's level-1 detection (stats-weaknesses.js). Storing the full
        // (2 + 2K)x(2 + 2K) matrix would blow localStorage at K=64; the diagonal
        // (weaknessSe) + per-axis off-diagonals (weaknessAxisCov) is enough.
        try {
            const cov = math.inv(math.matrix(hessAt(params)));
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
//
// The contrast is CENTRED on a shipped population mean before it is scored, and that
// is what makes it personal. A wrong pick differs from the answer by a systematic
// corpus offset — a description of what wrong moves look like, not of who played
// them — and the likelihood routes that offset into β_picked, so every user's raw
// contrast carries the same component. Scoring it uncentred ranks candidates by a
// fixed corpus preference and serves everyone the same corner of the pool in the
// shape of personalization; subtracting the mean leaves the part that actually
// differs between users. Note this is the opposite treatment from the SUM, which is
// centred on zero and whose population displacement HALTS the offline fit: a shared
// sum coefficient is residual difficulty belonging upstream in the axis fit, so
// centring it would hide a defect, while a shared contrast is measured geometry and
// centring it removes a known nuisance.

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
    // Also what makes AXIS_CONTRAST_MEAN[k] safe to index below: it is length K for the
    // axis set package_for_pwa shipped (check_axis_prior refuses otherwise), and this
    // line establishes that the trained model's K is that same shipped K.
    if (!problem.embedding || problem.embedding.length !== K) return 0;
    let score = 0;
    for (let k = 0; k < K; k++) {
        // (β_correct[k] − β_picked[k]) − population mean: the identified over/under
        // weakness direction (see the note above on why β_picked enters with a MINUS,
        // not the inverted + of the old sum), with the offset every user shares
        // removed so what's left is this user's own deviation.
        const betaEff = params[2 + k] - params[2 + K + k] - AXIS_CONTRAST_MEAN[k];
        score += -betaEff * problem.embedding[k];
    }
    return score;
}

