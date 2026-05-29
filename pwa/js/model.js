// js/model.js

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, setModelParams, setAdamState, getMergedAnswerHistory } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, RECALL_CANDIDATE_HISTORY_LENGTH } from './constants.js';

// ------------------------------------------------------------------------------------------------
// --- Moved from problem-loader.js ---------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

export function findFirst(sortedArr, target) {
    let low = 0;
    let high = sortedArr.length - 1;
    let result = -1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        if (sortedArr[mid] >= target) {
            if (sortedArr[mid] === target) result = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return result;
}

export function findLast(sortedArr, target) {
    let low = 0;
    let high = sortedArr.length - 1;
    let result = -1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        if (sortedArr[mid] <= target) {
            if (sortedArr[mid] === target) result = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return result;
}

export function calculatePercentile(sortedArr, target) {
    if (sortedArr.length === 0) return 0;
    const firstIndex = findFirst(sortedArr, target);
    if (firstIndex === -1) return 0;
    const lastIndex = findLast(sortedArr, target);
    const below = firstIndex;
    const equal = lastIndex - firstIndex + 1;
    return (below + 0.5 * equal) / sortedArr.length;
}

// --- Retention/stability model ----------------------------------------------
// Recall is an FSRS-style retention curve anchored at R(0)=1, NOT a logit on
// population correctness (the old additive `base + βR·log(...)` form let a
// problem's difficulty offset the curve downward so it could be "due" the
// instant it was solved). For a REPEAT encounter the shared linear predictor is
// the LOG of memory stability S (in days):
//
//   lnS = INTERCEPT + TIME_SOLVING·timePct + REVIEW_TIME·reviewPct
//         + PROBLEM_DIFFICULTY·problemRank + SIZE_IS_13·size13
//         + SUCCESS_SINCE_FAILURE·r + LN_C
//   R(t) = exp(-t_days / exp(lnS))
//
// so R(0)=1 always (no instant-due) and every per-problem signal scales the
// time-constant multiplicatively instead of offsetting the curve — difficulty's
// only action is on how S compounds, exactly the FSRS picture. Each consecutive
// success multiplies S by e^{SUCCESS_SINCE_FAILURE} (geometric growth), and a
// lapse resets r→0 (post-lapse reset).
//
// The SAME features (with LN_C and SUCCESS_SINCE_FAILURE zeroed) feed the
// NOVEL-encounter cold-solve logit P = σ(z). That head is what trains
// PROBLEM_DIFFICULTY / TIME_SOLVING / SIZE_IS_13 and backs the stats-panel
// readouts; REVIEW_TIME and SUCCESS_SINCE_FAILURE are identified purely from
// repeat rows (both are 0 on novel rows). There is no IS_NOVEL indicator: the
// novel/repeat regime is carried by which head a row enters, and an indicator
// that is 1 on every novel row would just be collinear with the intercept.
const NEG_LN_080 = -Math.log(0.80); // 0.22314 — t/S at the 0.80 window-open
const NEG_LN_060 = -Math.log(0.60); // 0.51083 — t/S at the 0.60 window-close
// Stability floor/ceiling (days). The floor keeps a just-failed problem from
// coming due sooner than ~18h (NEG_LN_080·MIN ≈ 18h window-open, matching the
// old min-hours clamp); the ceiling guards against runaway intervals/overflow.
const MIN_STABILITY_DAYS = (18 / 24) / NEG_LN_080;
const MAX_STABILITY_DAYS = 365;

// Shared linear-predictor feature vector. `isRepeat` toggles the retention-only
// features (LN_C and SUCCESS_SINCE_FAILURE carry signal only on repeats); for
// prediction `timePct` defaults to the neutral 0.5.
function buildFeatures({ isRepeat, timePct = 0.5, reviewPct = 0, problemRank = 0, size, r = 0 }) {
    const f = Array(Object.keys(MODEL_INDICES).length).fill(0);
    f[MODEL_INDICES.INTERCEPT] = 1;
    f[MODEL_INDICES.TIME_SOLVING] = timePct;
    f[MODEL_INDICES.REVIEW_TIME] = reviewPct;
    f[MODEL_INDICES.PROBLEM_DIFFICULTY] = problemRank ?? 0;
    // 9x9 is the baseline; SIZE_IS_13 carries the 13x13 offset. Merged-history
    // entries carry `size`; active-play entries don't, so assume the active size.
    f[MODEL_INDICES.SIZE_IS_13] = (size ?? state.boardSize) === 13 ? 1 : 0;
    if (isRepeat) {
        f[MODEL_INDICES.SUCCESS_SINCE_FAILURE] = r;
        f[MODEL_INDICES.LN_C] = 1;
    }
    return f;
}

function dot(features, params) {
    let s = 0;
    for (let j = 0; j < features.length; j++) s += features[j] * params[j];
    return s;
}

// Clamped stability (days) from a repeat-row linear predictor (lnS = eta).
function stabilityDaysFromEta(eta) {
    const S = Math.exp(eta);
    if (!isFinite(S)) return MAX_STABILITY_DAYS;
    return Math.min(MAX_STABILITY_DAYS, Math.max(MIN_STABILITY_DAYS, S));
}

// Window-open delay (hours) for a just-failed (r=0, no prior review) repeat at a
// representative difficulty — the stats panel's "a missed problem comes back in
// ~X h" readout. Returns null when params aren't ready.
export function freshFailureReviewHours(params, { problemRank = 0, size } = {}) {
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return null;
    const eta = dot(buildFeatures({ isRepeat: true, reviewPct: 0, problemRank, size, r: 0 }), params);
    return NEG_LN_080 * stabilityDaysFromEta(eta) * 24;
}

// Builds the training feature vector for history row i (board-independent).
function featuresForTrainingRow(td, i) {
    const p = td.history[i];
    return buildFeatures({
        isRepeat: !!p.isRepeat,
        timePct: td.percentileTime[i],
        reviewPct: td.percentileReview[i],
        problemRank: p.problemRank,
        size: p.size,
        r: p.successSinceFailure || 0,
    });
}

// Per-row predicted probability + dLoss/dEta under the dual-head likelihood:
// novel rows are a logistic on the cold-solve logit; repeat rows are the
// retention curve R = exp(-t/S), S = exp(eta).
function predictAndGrad(eta, { isRepeat, elapsedDays, y }) {
    if (!isRepeat) {
        const p = 1 / (1 + Math.exp(-eta));
        const clipped = Math.max(1e-15, Math.min(1 - 1e-15, p));
        const loss = -y * Math.log(clipped) - (1 - y) * Math.log(1 - clipped);
        return { p, loss, dEta: p - y, fisherW: p * (1 - p) };
    }
    const S = Math.exp(eta);
    const u = elapsedDays / S;              // ≥ 0; u=0 ⇒ R=1 (just solved)
    const R = Math.exp(-u);
    const clipped = Math.max(1e-15, Math.min(1 - 1e-15, R));
    const loss = -y * Math.log(clipped) - (1 - y) * Math.log(1 - clipped);
    // dR/dEta = R·u, so dLoss/dEta = (R-y)·u/(1-R); Fisher = (R·u)²/(R(1-R)).
    const oneMinusR = Math.max(1 - R, 1e-9);
    return { p: R, loss, dEta: (R - y) * u / oneMinusR, fisherW: R * u * u / oneMinusR };
}


export function getRecallCandidates(options = {}) {
    const {
        lowerProbabilityBound = 0.60,
        upperProbabilityBound = 0.80,
        requireLastAnswerIncorrect = false,
        // Default to the active size's history (play path: a recall pick can
        // only be re-shown in its own size). The stats panel passes the merged
        // history so its counts span both sizes.
        history = state.answerHistory,
    } = options;

    const now = Date.now();
    // User-controlled pause: while active, no problem can be picked as a
    // recall (and URL-loaded recalls are routed as plain linked solves).
    if (state.recallPausedUntil > now) return [];
    const recallCandidates = [];

    const numParams = Object.keys(MODEL_INDICES).length;
    if (!state.modelParams || state.modelParams.length !== numParams) {
        return []; // Model not ready (or a stale model awaiting retrain)
    }
    const params = state.modelParams;

    const recentHistory = history.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);

    const allReviewTimes = recentHistory
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0);
    allReviewTimes.sort((a, b) => a - b);

    const lastAnswers = {};
    for (const answer of recentHistory) {
        lastAnswers[answer.problemId] = answer;
    }

    const DAY_MS = 24 * 3600 * 1000;
    for (const problemId in lastAnswers) {
        const lastAnswer = lastAnswers[problemId];

        if (lastAnswer.ignored) continue; // Exclude ignored problems
        if (state.unavailableProblemIds.has(problemId)) continue; // Exclude problems that failed to load this session
        if (!lastAnswer.date) continue;

        let condition = lastAnswer.problemRank < state.userRank && lastAnswer.successSinceFailure !== undefined;
        if (requireLastAnswerIncorrect) {
            condition = condition && !lastAnswer.correct;
        }
        if (!condition) continue;

        const reviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
        const reviewPct = calculatePercentile(allReviewTimes, reviewTime);
        const eta = dot(buildFeatures({
            isRepeat: true,
            reviewPct,
            problemRank: lastAnswer.problemRank,
            size: lastAnswer.size,
            r: lastAnswer.successSinceFailure || 0,
        }), params);
        const S = stabilityDaysFromEta(eta);

        const tDays = (now - lastAnswer.date) / DAY_MS;
        if (tDays < 0) continue;
        const probability = Math.exp(-tDays / S); // retention R(t); R(0)=1

        if (probability >= lowerProbabilityBound && probability <= upperProbabilityBound) {
            recallCandidates.push({
                problemId,
                recallRate: probability,
            });
        }
    }

    return recallCandidates;
}

// Predicts the unix-ms timestamps at which each candidate problem will cross
// R=0.80 (recall window opens) and R=0.60 (recall window closes), inverting the
// retention curve R(t)=exp(-t/S) analytically: t = -ln(R)·S. Because R(0)=1,
// both crossings are strictly in the future of the last attempt. Mirrors the
// eligibility filters of getRecallCandidates. Reads modelParams/answerHistory/
// userRank directly so callers can request a size other than the active one.
//
// Output rows are bounded:
//   - exit_at must be in the future (we never ping about a window already past)
//   - enter_at must be within a 30-day horizon (avoids accumulating stale
//     schedule rows for problems that won't be due for months; the schedule
//     will get re-uploaded after subsequent attempts anyway).
export function buildRecallSchedule(size) {
    // The recall model and rank are shared (single keys); only the answer
    // history is per-size — a recall reminder must point at a problem in `size`.
    const readSharedKey = (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        try { return JSON.parse(raw); } catch { return null; }
    };
    const readPerSizeKey = (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        try {
            const parsed = JSON.parse(raw);
            return parsed?.[String(size)] ?? null;
        } catch {
            return null;
        }
    };

    const params = readSharedKey('modelParams');
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return [];
    const answerHistory = readPerSizeKey('answerHistory');
    if (!Array.isArray(answerHistory) || answerHistory.length === 0) return [];
    const userRank = readSharedKey('userRank');
    if (typeof userRank !== 'number') return [];

    const recentHistory = answerHistory.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);
    const allReviewTimes = recentHistory
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0);
    allReviewTimes.sort((a, b) => a - b);

    const lastAnswers = {};
    for (const answer of recentHistory) lastAnswers[answer.problemId] = answer;

    const now = Date.now();
    const DAY_MS = 24 * 3600 * 1000;
    const HORIZON_MS = 30 * DAY_MS;
    const result = [];

    for (const problemId in lastAnswers) {
        const lastAnswer = lastAnswers[problemId];
        if (lastAnswer.ignored) continue;
        if (lastAnswer.problemRank === undefined || lastAnswer.problemRank >= userRank) continue;
        if (lastAnswer.successSinceFailure === undefined) continue;
        if (!lastAnswer.date) continue;

        const reviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
        const reviewPct = calculatePercentile(allReviewTimes, reviewTime);

        // size is the explicit per-size arg here, not the active board.
        const eta = dot(buildFeatures({
            isRepeat: true,
            reviewPct,
            problemRank: lastAnswer.problemRank,
            size,
            r: lastAnswer.successSinceFailure || 0,
        }), params);
        const S = stabilityDaysFromEta(eta);
        if (!isFinite(S)) continue;

        const tEnterDays = NEG_LN_080 * S; // R=0.80
        const tExitDays = NEG_LN_060 * S;  // R=0.60

        const enterAt = lastAnswer.date + tEnterDays * DAY_MS;
        const exitAt = lastAnswer.date + tExitDays * DAY_MS;

        if (exitAt <= now) continue;
        if (enterAt > now + HORIZON_MS) continue;

        result.push({
            problemId,
            enterAt: Math.round(enterAt),
            exitAt: Math.round(exitAt),
        });
    }
    return result;
}

// ------------------------------------------------------------------------------------------------
// --- Moved from user-stats.js -------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

export const MODEL_INDICES = {
    INTERCEPT: 0,
    TIME_SOLVING: 1,
    REVIEW_TIME: 2,
    PROBLEM_DIFFICULTY: 3,
    // 9x9 is the baseline (feature 0); this carries the 13x13 offset at equal
    // internal-rank difficulty. Its coefficient → ranks is the stats panel's
    // "stronger/weaker on 13x13" readout.
    SIZE_IS_13: 4,
    // Retention-only (zero on novel rows). Each consecutive success adds this to
    // lnS, i.e. multiplies memory stability by e^{coeff} (FSRS geometric SInc).
    SUCCESS_SINCE_FAILURE: 5,
    // Retention-only intercept: the log days-scale that converts the shared
    // linear predictor into stability S = exp(lnS). Only the repeat head sees it.
    LN_C: 6,
};

const L2_REGULARIZATION_TARGETS = {
    [MODEL_INDICES.TIME_SOLVING]: -0.1,
    [MODEL_INDICES.REVIEW_TIME]: -0.1,
    [MODEL_INDICES.PROBLEM_DIFFICULTY]: -10,
    // Shrink toward 0 = "no skill difference between sizes" until data says so.
    [MODEL_INDICES.SIZE_IS_13]: 0,
    // Prior: each consecutive success ~doubles stability (ln 2 ≈ 0.69).
    [MODEL_INDICES.SUCCESS_SINCE_FAILURE]: Math.log(2),
    // Prior: base stability ~e^{-1} ≈ 0.37 day before the other terms move it.
    [MODEL_INDICES.LN_C]: -1,
};

// Population prior on logit-per-rank: a 1-rank skill gap shifts the log-odds
// of a correct answer by roughly this much. Treated as a Gaussian prior whose
// posterior is updated by the user's own fitted PROBLEM_DIFFICULTY slope.
const LOGIT_PER_RANK_PRIOR = 0.28;
const LOGIT_PER_RANK_PRIOR_SIGMA = 0.1;

// Precision-weighted blend of the prior and the user's own fitted slope from
// `state.statsResults`. Falls back to the prior when no usable model fit is
// available. Floored to keep downstream divisions well-conditioned.
export function getLogitPerRank(statsResults) {
    const priorPrecision = 1 / (LOGIT_PER_RANK_PRIOR_SIGMA * LOGIT_PER_RANK_PRIOR_SIGMA);
    const pd = statsResults && statsResults.results
        ? statsResults.results.find(r => r.varIndex === MODEL_INDICES.PROBLEM_DIFFICULTY)
        : null;
    if (!pd || !isFinite(pd.coeff) || !isFinite(pd.se) || pd.se <= 0) {
        return LOGIT_PER_RANK_PRIOR;
    }
    const userSlope = Math.abs(pd.coeff);
    const userPrecision = 1 / (pd.se * pd.se);
    const blended = (priorPrecision * LOGIT_PER_RANK_PRIOR + userPrecision * userSlope)
        / (priorPrecision + userPrecision);
    return Math.max(0.05, blended);
}

let isCalculatingStats = false;

export async function calculateAndStoreStatsResults() {
    if (isCalculatingStats) return;
    isCalculatingStats = true;

    try {
        // Shared recall model: trained on the union of both sizes' histories.
        // Every feature it reads (problemRank, time/review percentiles, the
        // recall-formula term) is board-independent, so pooling is on one scale.
        const validHistory = getMergedAnswerHistory().filter(p => p.problemRank !== undefined && p.timeTaken !== undefined && p.date !== undefined);
        if (validHistory.length < 10) {
            setStatsResults(null);
            return;
        }
        const history = validHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);

        const trainingData = prepareTrainingData(history, validHistory);

        const numParams = Object.keys(MODEL_INDICES).length;
        let params = (state.modelParams && state.modelParams.length === numParams) ? [...state.modelParams] : Array(numParams).fill(0);

        let adamState = (state.adamState && state.adamState.m.length === numParams)
            ? JSON.parse(JSON.stringify(state.adamState))
            : { m: Array(numParams).fill(0), v: Array(numParams).fill(0), t: 0 };

        const { trainedParams, finalAdamState } = await trainUserModel(trainingData, { params, adamState });

        // Sanity check for non-numeric values
        const areParamsValid = trainedParams.every(p => isFinite(p));

        if (areParamsValid) {
            const finalResults = calculateResultsAndStdErrors(trainedParams, trainingData);
            setModelParams(trainedParams);
            setAdamState(finalAdamState);
            setStatsResults(finalResults);
        } else {
            console.error("Model training resulted in non-numeric weights. Wiping model state.");
            Sentry.captureMessage("Model training resulted in non-numeric weights.");
            setModelParams(null);
            setAdamState(null);
            setStatsResults(null);
        }

    } catch (error) {
        console.error('Unified model training failed:', error);
        Sentry.captureException(error);
    } finally {
        isCalculatingStats = false;
    }
}

export function prepareTrainingData(history, validHistory) {
    const n = history.length;
    const p_low = 0.45, p_high = 0.55;

    const timeTakenValues = history.map(p => p.timeTaken);
    const sortedTime = [...timeTakenValues].sort((a, b) => a - b);
    const time_per_percentile_min = (sortedTime[Math.ceil(p_high * (n - 1))] - sortedTime[Math.floor(p_low * (n - 1))]) / (60000 * (p_high - p_low));
    
    const percentileTime = timeTakenValues.map(t => calculatePercentile(sortedTime, t));

    const previousReviewTimes = history.map(p => {
        if (p.isRepeat) {
            return p.accumulatedReview || 0;
        }
        return 0;
    });
    const reviewTimes = previousReviewTimes.filter(t => t > 0);
    const sortedReview = [...reviewTimes].sort((a, b) => a - b);
    let time_per_percentile_min_r = 0;
    if (reviewTimes.length > 0) {
        const idx_low_r = Math.floor(p_low * (sortedReview.length - 1));
        const idx_high_r = Math.ceil(p_high * (sortedReview.length - 1));
        time_per_percentile_min_r = (sortedReview[idx_high_r] - sortedReview[idx_low_r]) / (60000 * (p_high - p_low));
    }
    
    const percentileReview = previousReviewTimes.map(t => calculatePercentile(sortedReview, t));

    // Elapsed days since each repeat row's previous attempt of the same problem
    // (the retention head's t). Zero for novel rows (they use the logistic head).
    const DAY = 1000 * 60 * 60 * 24;
    const elapsedDays = history.map(p => {
        if (!p.isRepeat) return 0;
        const prev = validHistory.filter(a => a.problemId === p.problemId && a.date < p.date).pop();
        if (!prev) return 0;
        return Math.max(0, (p.date - prev.date) / DAY);
    });
    const isRepeatArr = history.map(p => !!p.isRepeat);

    return {
        y: history.map(p => p.correct ? 1 : 0),
        weights: Array.from({ length: n }, (_, i) => (i + 1) / n),
        percentileTime,
        percentileReview,
        elapsedDays,
        isRepeatArr,
        time_per_percentile_min,
        time_per_percentile_min_r,
        history,
        validHistory,
        n
    };
}

async function trainUserModel(trainingData, modelState) {
    const { y, weights, elapsedDays, isRepeatArr, n } = trainingData;
    let { params, adamState } = modelState;

    const learningRate = 0.02, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    const maxEpochs = 10000, maxTrainingTimeMs = 2000, lossConvergenceThreshold = 1e-6;
    const lambda = 10; // L2 regularization parameter

    let prevLoss = Infinity;
    const startTime = Date.now();

    // Features are independent of params, so build the design matrix once.
    const numFeatures = Object.keys(MODEL_INDICES).length;
    const X = Array.from({ length: n }, (_, i) => featuresForTrainingRow(trainingData, i));

    for (let epoch = 0; epoch < maxEpochs; epoch++) {
        adamState.t++;

        const gradient = Array(params.length).fill(0);
        let currentLoss = 0;

        for (let i = 0; i < n; i++) {
            const eta = dot(X[i], params);
            const { loss, dEta } = predictAndGrad(eta, { isRepeat: isRepeatArr[i], elapsedDays: elapsedDays[i], y: y[i] });
            currentLoss += weights[i] * loss;
            const error = weights[i] * dEta;
            for (let j = 0; j < numFeatures; j++) gradient[j] += error * X[i][j];
        }

        let l2_penalty = 0;
        for (let j = 0; j < params.length; j++) {
            if (j === MODEL_INDICES.INTERCEPT) continue;
            const target = L2_REGULARIZATION_TARGETS[j] || 0;
            l2_penalty += Math.pow(params[j] - target, 2);
        }
        currentLoss += (lambda / 2) * l2_penalty;
        currentLoss /= n;

        for (let j = 0; j < params.length; j++) {
            gradient[j] /= n;
            if (j !== MODEL_INDICES.INTERCEPT) {
                const target = L2_REGULARIZATION_TARGETS[j] || 0;
                gradient[j] += (lambda / n) * (params[j] - target);
            }
            adamState.m[j] = beta1 * adamState.m[j] + (1 - beta1) * gradient[j];
            adamState.v[j] = beta2 * adamState.v[j] + (1 - beta2) * Math.pow(gradient[j], 2);
            const m_hat = adamState.m[j] / (1 - Math.pow(beta1, adamState.t));
            const v_hat = adamState.v[j] / (1 - Math.pow(beta2, adamState.t));
            params[j] -= learningRate * m_hat / (Math.sqrt(v_hat) + epsilon);
        }

        if (Math.abs(prevLoss - currentLoss) < lossConvergenceThreshold || Date.now() - startTime > maxTrainingTimeMs) break;
        prevLoss = currentLoss;
    }
    return { trainedParams: params, finalAdamState: adamState };
}

export function calculateResultsAndStdErrors(finalParams, trainingData) {
    const { elapsedDays, isRepeatArr, weights, y, n, time_per_percentile_min, time_per_percentile_min_r } = trainingData;

    const X = Array.from({ length: n }, (_, i) => featuresForTrainingRow(trainingData, i));

    const X_matrix = math.matrix(X);
    // Per-row Fisher weight under the dual-head likelihood: p(1-p) for logistic
    // (novel) rows, R·u²/(1-R) for retention (repeat) rows.
    const fisherWeights = X.map((xi, i) => {
        const eta = dot(xi, finalParams);
        const { fisherW } = predictAndGrad(eta, { isRepeat: isRepeatArr[i], elapsedDays: elapsedDays[i], y: y[i] });
        return weights[i] * fisherW;
    });
    const W = math.diag(fisherWeights);
    const Xt = math.transpose(X_matrix);
    const fisherInfo = math.add(math.multiply(Xt, W, X_matrix), math.diag(Array(X[0].length).fill(1e-6)));
    const varCovarMatrix = math.inv(fisherInfo);
    const stdErrors = math.map(math.diag(varCovarMatrix), math.sqrt).toArray();

    const results = [];
    for (let i = 1; i < Object.keys(MODEL_INDICES).length; i++) {
        let coeff = finalParams[i], se = stdErrors[i], scaleFactor = 1;
        if (i === MODEL_INDICES.TIME_SOLVING && time_per_percentile_min > 0) scaleFactor = 1 / time_per_percentile_min;
        else if (i === MODEL_INDICES.REVIEW_TIME && time_per_percentile_min_r > 0) scaleFactor = 1 / time_per_percentile_min_r;
        else if (i === MODEL_INDICES.PROBLEM_DIFFICULTY) scaleFactor = 1 / 29;
        results.push({ coeff: coeff * scaleFactor, se: se * Math.abs(scaleFactor), varIndex: i, scaleFactor });
    }
    return { results, varCovarMatrix };
}


