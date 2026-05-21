// js/model.js

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, setModelParams, setAdamState } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, RECALL_CANDIDATE_HISTORY_LENGTH, TOTAL_RANKS } from './constants.js';
import { getTerm } from './i18n.js';
import { quadrantProbs } from './bivariate-normal.js';
import { computeRankDecomposition, evaluateAt } from './rank-decomposition.js';

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

function getFeaturesForPrediction(lastAnswer, { allReviewTimes, now }) {
    const features = Array(Object.keys(MODEL_INDICES).length).fill(0);
    features[MODEL_INDICES.INTERCEPT] = 1;
    features[MODEL_INDICES.TIME_SOLVING] = 0.5; // Neutral value for prediction

    const candidateReviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
    const reviewTimeFeature = calculatePercentile(allReviewTimes, candidateReviewTime);
    features[MODEL_INDICES.REVIEW_TIME] = reviewTimeFeature;

    features[MODEL_INDICES.PROBLEM_DIFFICULTY] = lastAnswer.problemRank;
    features[MODEL_INDICES.IS_NOVEL] = 0; // This is a recall problem

    const t = (now - lastAnswer.date) / (1000 * 60 * 60 * 24);
    const r = lastAnswer.successSinceFailure !== undefined ? lastAnswer.successSinceFailure : 0;
    const interactionTerm = Math.log(1 + t * Math.pow(0.5, r));
    features[MODEL_INDICES.RECALL_FORMULA] = interactionTerm;
    
    return features;
}

function getFeaturesForTraining(p, i, { percentileTime, percentileReview, validHistory }) {
    const features = Array(Object.keys(MODEL_INDICES).length).fill(0);
    features[MODEL_INDICES.INTERCEPT] = 1;
    features[MODEL_INDICES.TIME_SOLVING] = percentileTime[i];
    features[MODEL_INDICES.REVIEW_TIME] = percentileReview[i];
    features[MODEL_INDICES.PROBLEM_DIFFICULTY] = p.problemRank;

    if (!p.isRepeat) {
        features[MODEL_INDICES.IS_NOVEL] = 1;
    } else {
        const prevAnswer = validHistory.filter(ans => ans.problemId === p.problemId && ans.date < p.date).pop();
        const t = (p.date - (prevAnswer ? prevAnswer.date : p.date)) / (1000 * 60 * 60 * 24);
        const r = p.successSinceFailure !== undefined ? p.successSinceFailure : 0;
        const interactionTerm = Math.log( 1 + t * Math.pow(0.5,r));
        features[MODEL_INDICES.RECALL_FORMULA] = interactionTerm;
    }
    return features;
}


export function getRecallCandidates(options = {}) {
    const {
        lowerProbabilityBound = 0.60,
        upperProbabilityBound = 0.80,
        requireLastAnswerIncorrect = false
    } = options;

    const now = Date.now();
    const recallCandidates = [];

    if (!state.modelParams || state.modelParams.length === 0) {
        return []; // Model not ready
    }
    const finalParams = state.modelParams;

    const clippedParams = [...finalParams];
    const N = clippedParams[MODEL_INDICES.IS_NOVEL];
    const R = clippedParams[MODEL_INDICES.RECALL_FORMULA];

    const min_hours = 18;
    const max_hours = 168; // 7 days
    const min_ratio = Math.log(1 + min_hours / 24);
    const max_ratio = Math.log(1 + max_hours / 24);
    if (Math.abs(R) < 1e-9) return [];
    const current_ratio = N / R;

    if (current_ratio < min_ratio) {
        clippedParams[MODEL_INDICES.RECALL_FORMULA] = N / min_ratio;
    } else if (current_ratio > max_ratio) {
        clippedParams[MODEL_INDICES.RECALL_FORMULA] = N / max_ratio;
    }

    const recentHistory = state.answerHistory.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);

    const allReviewTimes = recentHistory
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0);
    allReviewTimes.sort((a, b) => a - b);

    const lastAnswers = {};
    for (const answer of recentHistory) {
        lastAnswers[answer.problemId] = answer;
    }

    for (const problemId in lastAnswers) {
        const lastAnswer = lastAnswers[problemId];

        if (lastAnswer.ignored) continue; // Exclude ignored problems
        if (state.unavailableProblemIds.has(problemId)) continue; // Exclude problems that failed to load this session

        let condition = lastAnswer.problemRank < state.userRank && lastAnswer.successSinceFailure !== undefined;
        if (requireLastAnswerIncorrect) {
            condition = condition && !lastAnswer.correct;
        }

        if (condition) {
            const features = getFeaturesForPrediction(lastAnswer, { allReviewTimes, now });
            const logit = features.reduce((acc, val, j) => acc + val * clippedParams[j], 0);
            const probability = 1 / (1 + Math.exp(-logit));

            if (probability >= lowerProbabilityBound && probability <= upperProbabilityBound) {
                recallCandidates.push({
                    problemId,
                    recallRate: probability,
                });
            }
        }
    }

    return recallCandidates;
}

// Inverts the logistic recall model to predict the unix-ms timestamps at
// which each candidate problem will cross P=0.80 (recall window opens) and
// P=0.60 (recall window closes). Mirrors the eligibility filters of
// getRecallCandidates and applies the same min/max-hour clipping to the
// recall coefficient. Reads modelParams/answerHistory/userRank directly so
// callers can request a size other than the currently active one.
//
// Output rows are bounded:
//   - exit_at must be in the future (we never ping about a window already past)
//   - enter_at must be within a 30-day horizon (avoids accumulating stale
//     schedule rows for problems that won't be due for months; the schedule
//     will get re-uploaded after subsequent attempts anyway).
export function buildRecallSchedule(size) {
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

    const params = readPerSizeKey('modelParams');
    if (!Array.isArray(params) || params.length === 0) return [];
    const answerHistory = readPerSizeKey('answerHistory');
    if (!Array.isArray(answerHistory) || answerHistory.length === 0) return [];
    const userRank = readPerSizeKey('userRank');
    if (typeof userRank !== 'number') return [];

    const clippedParams = [...params];
    const N = clippedParams[MODEL_INDICES.IS_NOVEL];
    const R = clippedParams[MODEL_INDICES.RECALL_FORMULA];
    const min_hours = 18;
    const max_hours = 168;
    const min_ratio = Math.log(1 + min_hours / 24);
    const max_ratio = Math.log(1 + max_hours / 24);
    if (Math.abs(R) < 1e-9) return [];
    const current_ratio = N / R;
    if (current_ratio < min_ratio) {
        clippedParams[MODEL_INDICES.RECALL_FORMULA] = N / min_ratio;
    } else if (current_ratio > max_ratio) {
        clippedParams[MODEL_INDICES.RECALL_FORMULA] = N / max_ratio;
    }

    const betaRecall = clippedParams[MODEL_INDICES.RECALL_FORMULA];
    if (!isFinite(betaRecall) || betaRecall >= 0) return [];

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
        const reviewPercentile = calculatePercentile(allReviewTimes, reviewTime);

        const base = clippedParams[MODEL_INDICES.INTERCEPT]
                   + clippedParams[MODEL_INDICES.TIME_SOLVING] * 0.5
                   + clippedParams[MODEL_INDICES.REVIEW_TIME] * reviewPercentile
                   + clippedParams[MODEL_INDICES.PROBLEM_DIFFICULTY] * lastAnswer.problemRank;

        const r = lastAnswer.successSinceFailure || 0;
        const factor = Math.pow(2, r);

        const daysFor = (p) => {
            const logitTarget = Math.log(p / (1 - p));
            const recallFeature = (logitTarget - base) / betaRecall;
            return (Math.exp(recallFeature) - 1) * factor;
        };

        const tEnterDays = daysFor(0.80);
        const tExitDays = daysFor(0.60);
        if (!isFinite(tEnterDays) || !isFinite(tExitDays)) continue;
        if (tExitDays <= tEnterDays) continue;

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
    IS_NOVEL: 4,
    RECALL_FORMULA: 5,
};

const L2_REGULARIZATION_TARGETS = {
    [MODEL_INDICES.TIME_SOLVING]: -0.1,
    [MODEL_INDICES.REVIEW_TIME]: -0.1,
    [MODEL_INDICES.PROBLEM_DIFFICULTY]: -10,
    [MODEL_INDICES.IS_NOVEL]: -1,
    [MODEL_INDICES.RECALL_FORMULA]: -1,
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
        const validHistory = state.answerHistory.filter(p => p.problemRank !== undefined && p.timeTaken !== undefined && p.date !== undefined);
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

function prepareTrainingData(history, validHistory) {
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

    return {
        y: history.map(p => p.correct ? 1 : 0),
        weights: Array.from({ length: n }, (_, i) => (i + 1) / n),
        percentileTime,
        percentileReview,
        time_per_percentile_min,
        time_per_percentile_min_r,
        history,
        validHistory,
        n
    };
}

async function trainUserModel(trainingData, modelState) {
    const { y, weights, percentileTime, percentileReview, history, validHistory, n } = trainingData;
    let { params, adamState } = modelState;

    const learningRate = 0.02, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    const maxEpochs = 10000, maxTrainingTimeMs = 2000, lossConvergenceThreshold = 1e-6;
    const lambda = 10; // L2 regularization parameter

    let prevLoss = Infinity;
    const startTime = Date.now();

    for (let epoch = 0; epoch < maxEpochs; epoch++) {
        adamState.t++;

        const X = history.map((p, i) => getFeaturesForTraining(p, i, { percentileTime, percentileReview, validHistory }));

        const gradient = Array(params.length).fill(0);
        let currentLoss = 0;

        for (let i = 0; i < n; i++) {
            const logit = X[i].reduce((acc, val, j) => acc + val * params[j], 0);
            const predicted = 1 / (1 + Math.exp(-logit));
            const clipped = Math.max(1e-15, Math.min(1 - 1e-15, predicted));
            currentLoss += weights[i] * (-y[i] * Math.log(clipped) - (1 - y[i]) * Math.log(1 - clipped));
            const error = weights[i] * (predicted - y[i]);
            for (let j = 0; j < Object.keys(MODEL_INDICES).length; j++) gradient[j] += error * X[i][j];
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

function calculateResultsAndStdErrors(finalParams, trainingData) {
    const { percentileTime, percentileReview, history, validHistory, n, time_per_percentile_min, time_per_percentile_min_r } = trainingData;

    const X = history.map((p, i) => getFeaturesForTraining(p, i, { percentileTime, percentileReview, validHistory }));

    const X_matrix = math.matrix(X);
    const p_hat = X.map(xi => 1 / (1 + Math.exp(-xi.reduce((acc, val, j) => acc + val * finalParams[j], 0))));
    const W = math.diag(p_hat.map((pi, i) => trainingData.weights[i] * pi * (1 - pi)));
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


export async function showDetailedStats() {
    if (state.answerHistory.length < 10) {
        alert(getTerm('statsNotEnough').replace('{count}', state.answerHistory.length));
        return;
    }

    if (!state.modelParams) {
        alert(getTerm('statsFailed'));
        return;
    }

    if (!state.statsResults) {
        const validHistory = state.answerHistory.filter(p => p.problemRank !== undefined && p.timeTaken !== undefined && p.date !== undefined);
        const history = validHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
        const trainingData = prepareTrainingData(history, validHistory);
        const finalResults = calculateResultsAndStdErrors(state.modelParams, trainingData);
        setStatsResults(finalResults);
    }

    if (!state.statsResults) {
        alert(getTerm('statsFailed'));
        return;
    }

    try {
        const { results, varCovarMatrix } = state.statsResults;
        const finalParams = state.modelParams;
        const logitPerRank = getLogitPerRank(state.statsResults);
        let resultText = '';

        // Block 1: Observations
        for (const result of results) {
            const { coeff, se, varIndex } = result;
            let termKey;
            if (varIndex === MODEL_INDICES.TIME_SOLVING) termKey = 'statsSolvingTimeNatural';
            else if (varIndex === MODEL_INDICES.REVIEW_TIME) termKey = 'statsReviewTimeNatural';

            if (termKey) {
                const z = coeff / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = cdf * 100;
                resultText += getTerm(termKey).replace('{percent_change}', conf.toFixed(0)) + '\n';
            }
        }

        // Separator
        resultText += '\n';

        // Block 2: Summary Text
        if (varCovarMatrix) {
            const N_idx = MODEL_INDICES.IS_NOVEL;
            const R_idx = MODEL_INDICES.RECALL_FORMULA;
            const N = finalParams[N_idx];
            const R = finalParams[R_idx];

            if (Math.abs(R) > 1e-9) {
                const t_eq_days = Math.max(0, Math.exp(N / R) - 1);
                const allRepeats = getRecallCandidates({
                    lowerProbabilityBound: 0.6,
                    upperProbabilityBound: 1.0,
                    requireLastAnswerIncorrect: false
                }).length;
                const firstTimeFailures = getRecallCandidates({
                    lowerProbabilityBound: 0.6,
                    upperProbabilityBound: 1.0,
                    requireLastAnswerIncorrect: true
                }).length;
                const firsttimeFailureTime = t_eq_days * 24;

                resultText += getTerm('statsFailures').replace('{count_recent}', firstTimeFailures).replace('{hours}', Math.round(firsttimeFailureTime)) + '\n';
                resultText += getTerm('statsOldFailures').replace('{count_other}', Math.max(0, allRepeats - firstTimeFailures)) + '\n';
            }
        }

        const novelProblemCount = new Set(state.answerHistory.filter(p => !p.isRepeat).map(p => p.problemId)).size;
        resultText += getTerm('statsNovelProblems').replace('{count}', novelProblemCount) + '\n';

        const slippedProblemCount = getRecallCandidates({
            lowerProbabilityBound: 0,
            upperProbabilityBound: 0.6,
            requireLastAnswerIncorrect: false
        }).length;
        const ignoredProblemCount = state.answerHistory.filter(p => p.ignored).length;
        resultText += getTerm('statsIgnoredProblems').replace('{count_ignored}', ignoredProblemCount).replace('{count_slipped}', slippedProblemCount) + '\n';

        // Difficulty sensitivity as a natural-language percent deviation:
        // |β_difficulty / logitPerRank| − 1 is the fractional gap between the
        // user's success-vs-rank slope and the prior-blended baseline. Lives
        // here (not in the model block) because it's a personal stat about
        // the user, not a model-output effect.
        const pdResult = results.find(r => r.varIndex === MODEL_INDICES.PROBLEM_DIFFICULTY);
        if (pdResult) {
            const ratio = Math.abs(pdResult.coeff) / logitPerRank;
            const pct = (ratio - 1) * 100;
            const direction = pct >= 0 ? 'more' : 'less';
            resultText += `You are ${Math.abs(pct).toFixed(0)}% ${direction} sensitive to problem difficulty than typical.\n`;
        }

        // Two-component rank decomposition (R = skill S + form M). Gives a
        // long-term-trend number and a session-to-session swing number that
        // can't be read off the single-line smoother in the chart. See
        // rank-decomposition.js for the model and the backfitting approach.
        const decomp = computeRankDecomposition(state.answerHistory);
        if (decomp) {
            const { summary } = decomp;
            if (summary.currentSkill !== null) {
                const skillLabel = rankLabelFromIndex(Math.round(summary.currentSkill));
                const seStr = Number.isFinite(summary.currentSkillSE)
                    ? ` (±${summary.currentSkillSE.toFixed(1)} ranks)`
                    : '';
                resultText += `Long-term skill estimate: ${skillLabel}${seStr}.\n`;
            }
            if (summary.longTermDrift !== null && summary.longTermSpanMs > 30 * 24 * 3600 * 1000) {
                const months = summary.longTermSpanMs / (30 * 24 * 3600 * 1000);
                const sign = summary.longTermDrift >= 0 ? '+' : '−';
                resultText += `Skill has drifted ${sign}${Math.abs(summary.longTermDrift).toFixed(1)} ranks over the past ${months.toFixed(0)} months.\n`;
            }
            if (typeof summary.sigmaM === 'number') {
                resultText += summary.sigmaM >= 0.1
                    ? `Session-to-session performance varies by ±${summary.sigmaM.toFixed(2)} ranks.\n`
                    : `Session-to-session performance: no significant swing detected (under ±${summary.sigmaMUpper.toFixed(2)} ranks).\n`;
            }
        }

        // Augmented-model posterior on (β_correct[k], β_picked[k]) per axis,
        // turned into sign-quadrant probabilities via bivariate-normal.js and
        // rendered as a DOM block (instead of inlined into resultText) so each
        // issue can carry an interactive `examples` toggle to past mistakes.
        const issuesElement = buildLikelyIssuesElement();
        const chartElement = buildRatingHistoryChart();
        const extras = document.createElement('div');
        if (issuesElement) extras.appendChild(issuesElement);
        if (chartElement) extras.appendChild(chartElement);
        showSelectableTextModal(resultText, extras);

    } catch (error) {
        console.error('Failed to generate stats text:', error);
        Sentry.captureException(error);
        alert(getTerm('statsFailed'));
    }
}

// Builds the "Likely issues" DOM block consumed by showDetailedStats. Each
// surfaced (axis k, quadrant) issue is one row with a toggle that lazily
// fetches matching past mistakes from answerHistory. Quadrants mirror the
// augmented model's sign analysis on (β_correct[k], β_picked[k]):
//   tooMuch    β_c > 0, β_p < 0  → user mis-applies motif k
//   notEnough  β_c < 0, β_p > 0  → user under-applies motif k
//   badAt      β_c < 0, β_p < 0  → user struggles with motif k overall
// `offDiagonal: true` quadrants need cov(β_c, β_p) ≥ 0 — anti-correlated
// posteriors (typical when most rows have picked = correct fallback) can land
// high mass in BOTH off-diagonal quadrants simultaneously without the data
// actually distinguishing over- vs under-application. badAt skips the gate
// because its both-negative sign-pattern is identifiable from β_c + β_p.
function buildLikelyIssuesElement() {
    const PROB_THRESHOLD = 0.50;
    const weaknessParams = state.weaknessParams;
    const weaknessSe = state.weaknessSe;
    const weaknessAxisCov = state.weaknessAxisCov;
    const labelMap = state.embeddingLabels || {};
    const labelFor = (k) => {
        const entry = labelMap[String(k)];
        return (entry && entry.positive) || `style #${k}`;
    };
    const QUADRANTS = [
        { key: 'tooMuch',   offDiagonal: true,  phrase: (l) => `You ${l} when you shouldn't.` },
        { key: 'notEnough', offDiagonal: true,  phrase: (l) => `You don't ${l} often enough.` },
        { key: 'badAt',     offDiagonal: false, phrase: (l) => `You inconsistently ${l}.` },
    ];

    const root = document.createElement('div');
    root.className = 'stats-issues';

    const heading = document.createElement('div');
    heading.className = 'stats-issues-heading';
    heading.textContent = 'Likely issues';
    root.appendChild(heading);

    const haveModel = (
        Array.isArray(weaknessParams) && weaknessParams.length >= 4 &&
        Array.isArray(weaknessSe) && weaknessSe.length === weaknessParams.length &&
        Array.isArray(weaknessAxisCov)
    );
    if (!haveModel) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = '(not enough data yet)';
        root.appendChild(empty);
        return root;
    }

    const K = (weaknessParams.length - 2) / 2;
    const issues = [];
    for (let k = 0; k < K; k++) {
        const mu_c = weaknessParams[2 + k];
        const mu_p = weaknessParams[2 + K + k];
        const var_c = weaknessSe[2 + k] * weaknessSe[2 + k];
        const var_p = weaknessSe[2 + K + k] * weaknessSe[2 + K + k];
        const cov = weaknessAxisCov[k] || 0;
        if (![mu_c, mu_p, var_c, var_p, cov].every(Number.isFinite)) continue;
        if (!(var_c > 0) || !(var_p > 0)) continue;
        const probs = quadrantProbs(mu_c, mu_p, var_c, var_p, cov);
        for (const q of QUADRANTS) {
            if (q.offDiagonal && cov < 0) continue;
            const p = probs[q.key];
            if (p >= PROB_THRESHOLD) {
                issues.push({ k, quadrant: q.key, p, sentence: q.phrase(labelFor(k)) });
            }
        }
    }
    issues.sort((a, b) => b.p - a.p);

    if (issues.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = `(none above ${(PROB_THRESHOLD * 100).toFixed(0)}% confidence yet)`;
        root.appendChild(empty);
        return root;
    }

    // Problems currently in the user's recall window — withhold their links so
    // a glance at the examples panel can't be used to ace an imminent recall.
    const recallSet = new Set(getRecallCandidates().map(c => c.problemId));

    for (const issue of issues) {
        root.appendChild(buildIssueRow(issue, recallSet));
    }
    return root;
}

// Renders one (axis k, quadrant) row: sentence + probability + an "examples"
// toggle that expands a list of links to past attempts matching the issue.
function buildIssueRow(issue, recallSet) {
    const row = document.createElement('div');
    row.className = 'stats-issue';

    const head = document.createElement('div');
    head.className = 'stats-issue-head';

    const prob = document.createElement('span');
    prob.className = 'stats-issue-prob';
    prob.textContent = `${(issue.p * 100).toFixed(0)}%`;

    const text = document.createElement('span');
    text.className = 'stats-issue-text';
    text.textContent = issue.sentence;

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'stats-issue-examples-toggle';
    toggle.textContent = 'examples';

    head.append(prob, text, toggle);
    row.appendChild(head);

    const list = document.createElement('div');
    list.className = 'stats-issue-examples-list';
    list.hidden = true;
    row.appendChild(list);

    let populated = false;
    toggle.addEventListener('click', () => {
        if (!populated) {
            populateExamplesList(list, issue, recallSet);
            populated = true;
        }
        list.hidden = !list.hidden;
        toggle.textContent = list.hidden ? 'examples' : 'hide';
    });
    return row;
}

function populateExamplesList(list, issue, recallSet) {
    // Threshold per-axis evidence strength to match the issue's posterior
    // confidence — see findExamplesForIssue for the rationale.
    const threshold = probit(issue.p);
    const examples = findExamplesForIssue(state.answerHistory, issue.k, issue.quadrant, recallSet, threshold);
    if (examples.length === 0) {
        const none = document.createElement('div');
        none.className = 'stats-issue-examples-empty';
        none.textContent = '(no past examples available)';
        list.appendChild(none);
        return;
    }
    const MAX_EXAMPLES = 10;
    const shown = examples.slice(0, MAX_EXAMPLES);
    for (const ex of shown) {
        const link = document.createElement('a');
        link.className = 'stats-issue-example-link';
        link.href = `?id=${encodeURIComponent(ex.problemId)}&o=0&close_on_next=1`;
        link.target = '_blank';
        link.rel = 'noopener';
        // Primary-click goes through window.open() so the spawned tab is
        // script-opened — window.close() in the spawned window is only
        // permitted when the opener was a script. Modifier-click and
        // middle-click fall through to the native target=_blank behavior.
        link.addEventListener('click', (e) => {
            if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
            e.preventDefault();
            window.open(link.href, '_blank', 'noopener');
        });
        link.textContent = `${ex.problemId} · ${formatSignalStrength(issue.quadrant, ex.ec, ex.ep)}`;
        list.appendChild(link);
    }
    if (examples.length > shown.length) {
        const more = document.createElement('div');
        more.className = 'stats-issue-examples-empty';
        more.textContent = `(+${examples.length - shown.length} more)`;
        list.appendChild(more);
    }
}

// Compact per-quadrant signal-strength readout. Embeddings are unit-variance
// z-scores along each axis, so values are reported in σ — sign included
// because each quadrant's evidence shape involves a specific signed pattern.
function formatSignalStrength(quadrant, ec, ep) {
    const fmt = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}σ`;
    // List the more-emphasized axis first per quadrant:
    //   tooMuch: motif in pick (positive ep) is the foreground signal
    //   notEnough: motif in correct (positive ec) is the foreground signal
    //   badAt: both are foreground; surface ec first to match the labelling
    if (quadrant === 'tooMuch') return `picked ${fmt(ep)} · correct ${fmt(ec)}`;
    if (quadrant === 'notEnough') return `correct ${fmt(ec)} · picked ${fmt(ep)}`;
    return `correct ${fmt(ec)} · picked ${fmt(ep)}`;
}

// Picks wrong attempts whose embedding signature aligns with the (k, quadrant)
// issue. Each quadrant requires BOTH axes to land on the side the quadrant's
// joint sign demands — a one-sided gate would include attempts that only
// support one axis, which is misleading for the both-negative `badAt` case
// (and for the off-diagonals it lets same-sign-ordered attempts slip through).
//
// The minimum |ec|, |ep| per attempt scales with the issue's posterior
// probability via `threshold = probit(p)` — so a 90%-confident issue needs
// ~1.28σ-strong evidence per axis, while a 50%-confident issue collapses to
// the pure sign constraint. Without this scaling, a 51%-borderline issue and
// a 95%-certain one would surface the same attempts, even though the latter
// should be backed by stronger per-attempt evidence.
//
// After the gate, per-attempt evidence strength is the sum of the two relevant
// signed σ contributions — monotone in either axis so a 5σ-on-one + threshold-
// on-the-other attempt ranks above one that just clears the threshold on both.
// (The earlier `min` tied any two attempts at the weaker axis regardless of
// the other one, which made the ordering look arbitrary on the panel.)
// Embeddings are unit-variance per component (skew-ICA standardized), so
// z-scale thresholds and sums are directly interpretable as σ above/below
// baseline.
function findExamplesForIssue(answerHistory, k, quadrant, excludedIds, threshold = 0) {
    const t = Number.isFinite(threshold) ? Math.max(0, threshold) : 0;
    const byId = new Map();
    for (const a of answerHistory) {
        if (!a || a.correct) continue;
        if (a.isRepeat) continue;  // mirror weakness.js training filter — repeats are excluded there
        if (a.ignored) continue;
        if (!a.problemId) continue;
        if (excludedIds && excludedIds.has(a.problemId)) continue;
        if (!Array.isArray(a.embedding) || a.embedding.length <= k) continue;
        if (!Array.isArray(a.pickedEmbedding) || a.pickedEmbedding.length <= k) continue;
        const ec = a.embedding[k];
        const ep = a.pickedEmbedding[k];
        if (!Number.isFinite(ec) || !Number.isFinite(ep)) continue;
        let score;
        if (quadrant === 'tooMuch') {
            if (!(ep > t) || !(-ec > t)) continue;
            score = ep + (-ec);
        } else if (quadrant === 'notEnough') {
            if (!(ec > t) || !(-ep > t)) continue;
            score = ec + (-ep);
        } else {  // badAt
            if (!(ec > t) || !(ep > t)) continue;
            score = ec + ep;
        }
        const prev = byId.get(a.problemId);
        if (!prev || score > prev.score) {
            byId.set(a.problemId, { problemId: a.problemId, date: a.date, score, ec, ep });
        }
    }
    return Array.from(byId.values()).sort((a, b) => b.score - a.score);
}

// Inverse standard-normal CDF via bisection over `math.erf`. ~50 iterations
// converges to ≪ 1e-9; cheap once per issue at panel-open time. Returns 0 for
// p ≤ 0.5 since callers only ever pass posterior probabilities in [0.5, 1).
function probit(p) {
    if (!(p > 0.5)) return 0;
    if (p >= 1) return Infinity;
    let lo = 0, hi = 8;
    for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        const cdf = 0.5 * (1 + math.erf(mid / Math.SQRT2));
        if (cdf < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}

function showSelectableTextModal(text, extraElement = null) {
    const backdrop = document.createElement('div');
    backdrop.className = 'stats-modal-backdrop';

    const card = document.createElement('div');
    card.className = 'stats-modal-card';

    const pre = document.createElement('pre');
    pre.className = 'stats-modal-pre';
    pre.textContent = text;

    const closeBtn = document.createElement('button');
    closeBtn.className = 'stats-modal-close';
    closeBtn.textContent = 'Close';

    const close = () => {
        document.removeEventListener('keydown', onKey);
        backdrop.remove();
    };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    closeBtn.onclick = close;
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.addEventListener('keydown', onKey);

    const body = document.createElement('div');
    body.className = 'stats-modal-body';
    body.append(pre);
    if (extraElement) body.append(extraElement);

    card.append(body, closeBtn);
    backdrop.appendChild(card);
    document.body.appendChild(backdrop);
}

function rankLabelFromIndex(rankIndex) {
    return rankIndex < 20 ? `${20 - rankIndex}k` : `${rankIndex - 19}d`;
}

function buildRatingHistoryChart() {
    const entries = state.answerHistory
        .filter(a => typeof a.problemRank === 'number' && typeof a.date === 'number'
                     && !a.isRepeat && !a.linked)
        .map(a => ({ date: a.date, rankValue: a.problemRank * TOTAL_RANKS, correct: !!a.correct }));
    if (entries.length < 2) return null;

    // Two-kernel rank decomposition: see rank-decomposition.js. We evaluate the
    // skill smoother Ŝ(t), its measurement SE seS(t), and the time-varying
    // performance swing σ_M(t) continuously on a regular t-grid (the kernels are
    // exact functions of t — no bucketing, no interpolation between sparse
    // attempt timestamps), so line and bands shape correctly across gaps.
    const t0 = entries[0].date;
    const t1 = entries[entries.length - 1].date;
    const tSpan = Math.max(1, t1 - t0);
    const decomp = computeRankDecomposition(state.answerHistory);
    const GRID_POINTS = 300;
    const Z = 1.96;
    // Half-widths in rank space: inner band = measurement uncertainty (how well
    // we know your skill); outer band = a prediction interval (where a given
    // session's performance lands), measurement and swing added in quadrature.
    const innerHalf = e => Z * e.seS;
    const outerHalf = e => Z * Math.hypot(e.seS, e.sigmaM);
    let gridEvals = [];
    if (decomp) {
        for (let i = 0; i < GRID_POINTS; i++) {
            const t = Math.round(t0 + tSpan * (i / (GRID_POINTS - 1)));
            const e = evaluateAt(decomp, t);
            if (e) gridEvals.push({ t, S: e.S, seS: e.seS, sigmaM: e.sigmaM });
        }
    }

    // Drop the lowest-rated 5% (capped at 50) when picking the y-range from
    // the scatter, so a handful of very-easy outliers don't squash the visible
    // part of the chart. Skill band always pushes outward.
    const sortedRanks = entries.map(e => e.rankValue).slice().sort((a, b) => a - b);
    const dropCount = Math.min(50, Math.floor(sortedRanks.length * 0.20));
    const trimmedMin = sortedRanks[dropCount];
    const maxRank = sortedRanks[sortedRanks.length - 1];
    let bandMin = Infinity, bandMax = -Infinity;
    for (const e of gridEvals) {
        const h = outerHalf(e);
        bandMin = Math.min(bandMin, e.S - h);
        bandMax = Math.max(bandMax, e.S + h);
    }
    if (!Number.isFinite(bandMin)) { bandMin = trimmedMin; bandMax = maxRank; }
    let yHigh = Math.ceil(Math.max(maxRank, bandMax));
    let yLow = Math.floor(Math.min(trimmedMin, bandMin));
    if (Number.isInteger(Math.min(trimmedMin, bandMin))) yLow -= 1;
    if (yHigh <= yLow) yHigh = yLow + 1;

    const container = document.createElement('div');
    container.className = 'stats-chart';

    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    title.textContent = `Skill since ${new Date(t0).toISOString().slice(0, 10)}`;
    container.append(title);

    const plotRow = document.createElement('div');
    plotRow.className = 'stats-chart-plot-row';

    // Grid line + label at every integer rank in the trimmed range. Labels
    // that would visually overlap their neighbor get hidden after layout
    // (see ResizeObserver below).
    const gridRanks = [];
    for (let r = yLow; r <= yHigh; r++) gridRanks.push(r);
    const labelRanks = gridRanks;

    // Y-axis labels live in HTML so they inherit the surrounding pre font
    // (13px monospace) exactly. The hidden spacer sizes the column to the
    // widest tick label.
    const yaxis = document.createElement('div');
    yaxis.className = 'stats-chart-yaxis';
    const spacer = document.createElement('span');
    spacer.className = 'y-spacer';
    spacer.textContent = '20k';
    yaxis.append(spacer);
    const topPct = v => (1 - (v - yLow) / (yHigh - yLow)) * 100;
    for (const r of labelRanks) {
        const tick = document.createElement('div');
        tick.className = 'y-tick';
        tick.style.top = `${topPct(r)}%`;
        tick.textContent = rankLabelFromIndex(r);
        yaxis.append(tick);
    }

    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-chart-svg-wrap';

    // SVG holds only the grid + dots. 1:1 aspect via square viewBox.
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const W = 400, H = 400;
    const xOf = t => ((t - t0) / tSpan) * W;
    const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;

    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');

    const gridStroke = 'rgba(128,128,128,0.35)';
    for (const r of gridRanks) {
        const y = yOf(r);
        const line = document.createElementNS(SVG_NS, 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('x2', W);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', gridStroke);
        line.setAttribute('stroke-width', '0.5');
        line.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(line);
    }

    // Two nested bands, drawn before scatter and line so raw data stays on top.
    // Outer (lighter) = prediction interval: skill ± where a session's form
    // actually lands. Inner (darker) = measurement uncertainty on skill. The
    // gap between them is the genuine session-to-session swing σ_M(t).
    const SKILL_FILL_OUTER = 'rgba(59,130,246,0.10)';
    const SKILL_FILL_INNER = 'rgba(59,130,246,0.22)';
    const SKILL_STROKE = '#2563eb';

    const bandPath = (halfOf) => {
        let up = '', dn = '';
        for (let i = 0; i < gridEvals.length; i++) {
            const e = gridEvals[i];
            const h = halfOf(e);
            const x = xOf(e.t).toFixed(1);
            const yU = yOf(e.S + h).toFixed(1);
            const yL = yOf(e.S - h).toFixed(1);
            up += (i === 0 ? 'M' : 'L') + `${x} ${yU} `;
            dn = `L${x} ${yL} ` + dn;
        }
        return up + dn + 'Z';
    };

    if (gridEvals.length >= 2) {
        for (const [halfOf, fill] of [[outerHalf, SKILL_FILL_OUTER], [innerHalf, SKILL_FILL_INNER]]) {
            const band = document.createElementNS(SVG_NS, 'path');
            band.setAttribute('d', bandPath(halfOf));
            band.setAttribute('fill', fill);
            band.setAttribute('stroke', 'none');
            svg.appendChild(band);
        }
    }

    // Per-attempt scatter: green=correct, red=wrong. On top of band so raw
    // data stays readable; under the skill line so the line remains visible.
    for (const e of entries) {
        const dot = document.createElementNS(SVG_NS, 'circle');
        dot.setAttribute('cx', xOf(e.date).toFixed(1));
        dot.setAttribute('cy', yOf(e.rankValue).toFixed(1));
        dot.setAttribute('r', '1.6');
        dot.setAttribute('fill', e.correct ? 'rgba(80,180,80,0.7)' : 'rgba(220,80,80,0.7)');
        svg.appendChild(dot);
    }

    if (gridEvals.length >= 2) {
        const skillPts = gridEvals
            .map(e => `${xOf(e.t).toFixed(1)},${yOf(e.S).toFixed(1)}`)
            .join(' ');
        const skillLine = document.createElementNS(SVG_NS, 'polyline');
        skillLine.setAttribute('points', skillPts);
        skillLine.setAttribute('fill', 'none');
        skillLine.setAttribute('stroke', SKILL_STROKE);
        skillLine.setAttribute('stroke-width', '1.5');
        skillLine.setAttribute('stroke-linejoin', 'round');
        skillLine.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(skillLine);
    }

    svgWrap.append(svg);

    const xaxis = document.createElement('div');
    xaxis.className = 'stats-chart-xaxis';
    xaxis.textContent = new Date(t1).toISOString().slice(0, 10);
    svgWrap.append(xaxis);

    plotRow.append(yaxis, svgWrap);
    container.append(plotRow);

    // After the chart hits the DOM and gets a real height, greedy-hide any
    // y-axis labels that would physically overlap a previously-shown one.
    // Re-runs on resize so density adapts to viewport.
    const updateLabelVisibility = () => {
        const h = svg.getBoundingClientRect().height;
        if (h === 0) return;
        const tickEls = [...yaxis.querySelectorAll('.y-tick')]
            .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
        const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
        let lastShownPx = -Infinity;
        for (const el of tickEls) {
            const topPx = (parseFloat(el.style.top) / 100) * h;
            if (topPx - lastShownPx >= minSepPx) {
                el.style.visibility = '';
                lastShownPx = topPx;
            } else {
                el.style.visibility = 'hidden';
            }
        }
    };
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(updateLabelVisibility).observe(svg);
    } else {
        queueMicrotask(updateLabelVisibility);
    }

    return container;
}
