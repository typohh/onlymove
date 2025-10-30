// js/model.js

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, setModelParams, setAdamState } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, REVIEW_CANDIDATE_HISTORY_LENGTH } from './constants.js';
import { getTerm } from './i18n.js';

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

export function getReviewCandidates(options = {}) {
    const {
        lowerProbabilityBound = 0.60,
        upperProbabilityBound = 0.80,
        requireLastAnswerIncorrect = false
    } = options;

    const now = Date.now();
    const reviewCandidates = [];

    if (!state.modelParams || state.modelParams.length === 0) {
        return []; // Model not ready
    }
    const finalParams = state.modelParams;

    const recentHistory = state.answerHistory.slice(-REVIEW_CANDIDATE_HISTORY_LENGTH);

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

        let condition = lastAnswer.problemRating < state.userRank;
        if (requireLastAnswerIncorrect) {
            condition = condition && !lastAnswer.correct;
        } else {
            condition = condition && lastAnswer.successSinceFailure !== undefined;
        }

        if (condition) {
            const features = Array(finalParams.length).fill(0);

            features[MODEL_INDICES.INTERCEPT] = 1;
            features[MODEL_INDICES.TIME_SOLVING] = 0.5; // Neutral value for prediction

            const candidateReviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
            const reviewTimeFeature = calculatePercentile(allReviewTimes, candidateReviewTime);
            features[MODEL_INDICES.REVIEW_TIME] = reviewTimeFeature;

            features[MODEL_INDICES.PROBLEM_DIFFICULTY] = lastAnswer.problemRating;
            features[MODEL_INDICES.PROBLEM_QUALITY] = 0;
            features[MODEL_INDICES.DATE] = 1.0; // For a new attempt, it's the latest
            features[MODEL_INDICES.IS_NOVEL] = 0; // This is a review problem

            const t = (now - lastAnswer.date) / (1000 * 60 * 60 * 24);
            const r = lastAnswer.successSinceFailure !== undefined ? lastAnswer.successSinceFailure : 0;
            const interactionTerm = Math.log(1 + t * Math.pow(0.5, r));
            features[MODEL_INDICES.RECALL_FORMULA] = interactionTerm;

            const logit = features.reduce((acc, val, j) => acc + val * finalParams[j], 0);
            const probability = 1 / (1 + Math.exp(-logit));

            if (probability >= lowerProbabilityBound && probability <= upperProbabilityBound) {
                reviewCandidates.push({
                    problemId,
                    recallRate: probability,
                    rating: lastAnswer.problemRating
                });
            }
        }
    }

    return reviewCandidates;
}

// ------------------------------------------------------------------------------------------------
// --- Moved from user-stats.js -------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

export const MODEL_INDICES = {
    INTERCEPT: 0,
    TIME_SOLVING: 1,
    REVIEW_TIME: 2,
    PROBLEM_DIFFICULTY: 3,
    PROBLEM_QUALITY: 4,
    DATE: 5,
    IS_NOVEL: 6,
    RECALL_FORMULA: 7,
};

export function probToRank(start, end) {
    const epsilon = 1e-9;
    start = Math.max(epsilon, Math.min(1 - epsilon, start));
    end = Math.max(epsilon, Math.min(1 - epsilon, end));
    const logOddsStart = Math.log(start / (1 - start));
    const logOddsEnd = Math.log(end / (1 - end));
    return (1 / 0.28) * (logOddsEnd - logOddsStart);
}

export function logitsToRank(startLogit, endLogit) {
    return (1 / 0.28) * (endLogit - startLogit);
}

let isCalculatingStats = false;

export async function calculateAndStoreStatsResults() {
    if (isCalculatingStats) return;
    isCalculatingStats = true;

    try {
        const validHistory = state.answerHistory.filter(p => p.problemRating !== undefined && p.timeTaken !== undefined && p.date !== undefined);
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
    const lambda = 1; // L2 regularization parameter

    let prevLoss = Infinity;
    const startTime = Date.now();

    for (let epoch = 0; epoch < maxEpochs; epoch++) {
        adamState.t++;

        const X = history.map((p, i) => {
            const features = Array(Object.keys(MODEL_INDICES).length).fill(0);
            features[MODEL_INDICES.INTERCEPT] = 1;
            features[MODEL_INDICES.TIME_SOLVING] = percentileTime[i];
            features[MODEL_INDICES.REVIEW_TIME] = percentileReview[i];
            features[MODEL_INDICES.PROBLEM_DIFFICULTY] = p.problemRating;
            features[MODEL_INDICES.PROBLEM_QUALITY] = 0;
         features[MODEL_INDICES.DATE] = i / n;
            features[MODEL_INDICES.DATE] = 0;
            if (!p.isRepeat) {
                features[MODEL_INDICES.IS_NOVEL] = 1;
            } else {
                const prevAnswer = validHistory.filter(ans => ans.problemId === p.problemId && ans.date < p.date).pop();
                const t = (p.date - (prevAnswer ? prevAnswer.date : p.date)) / (1000 * 60 * 60 * 24);
                const r = p.successSinceFailure !== undefined ? p.successSinceFailure : 0;

                const interactionTerm = Math.log( 1 + t * Math.pow(0.5,r))
                
                features[MODEL_INDICES.RECALL_FORMULA] = interactionTerm;
            }
            return features;
        });

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
            let target = 0;
            if (j === MODEL_INDICES.RECALL_FORMULA) {
                target = -1;
            } else if (j === MODEL_INDICES.PROBLEM_DIFFICULTY) {
                target = -10;
            } else if (j === MODEL_INDICES.IS_NOVEL) {
                target = -1;
            }
            l2_penalty += Math.pow(params[j] - target, 2);
        }
        currentLoss += (lambda / 2) * l2_penalty;
        currentLoss /= n;

        for (let j = 0; j < params.length; j++) {
            gradient[j] /= n;
            if (j !== MODEL_INDICES.INTERCEPT) {
                let target = 0;
                if (j === MODEL_INDICES.RECALL_FORMULA) {
                    target = -1;
                } else if (j === MODEL_INDICES.PROBLEM_DIFFICULTY) {
                    target = -10;
                } else if (j === MODEL_INDICES.IS_NOVEL) {
                    target = -1;
                }
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

    const X = history.map((p, i) => {
        const features = Array(Object.keys(MODEL_INDICES).length).fill(0);
        features[MODEL_INDICES.INTERCEPT] = 1;
        features[MODEL_INDICES.TIME_SOLVING] = percentileTime[i];
        features[MODEL_INDICES.REVIEW_TIME] = percentileReview[i];
        features[MODEL_INDICES.PROBLEM_DIFFICULTY] = p.problemRating;
        features[MODEL_INDICES.PROBLEM_QUALITY] = p.problemQuality || 0.5;
        features[MODEL_INDICES.DATE] = i / n;
        if (p.problemType === 'novel') {
            features[MODEL_INDICES.IS_NOVEL] = 1;
        } else {
            const prevAnswer = validHistory.filter(ans => ans.problemId === p.problemId && ans.date < p.date).pop();
            const t = (p.date - (prevAnswer ? prevAnswer.date : p.date)) / (1000 * 60 * 60 * 24);
            const r = p.successSinceFailure !== undefined ? p.successSinceFailure : 0;
            
            const interactionTerm = Math.log( 1 + t * Math.pow(0.5,r))
            
            features[MODEL_INDICES.RECALL_FORMULA] = interactionTerm;
        }
        return features;
    });

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
        else if (i === MODEL_INDICES.DATE) scaleFactor = n;
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
        const validHistory = state.answerHistory.filter(p => p.problemRating !== undefined && p.timeTaken !== undefined && p.date !== undefined);
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
        let resultText = '';

        // Block 1: Observations
        for (const result of results) {
            const { coeff, se, varIndex } = result;
            let termKey;
            if (varIndex === MODEL_INDICES.TIME_SOLVING) termKey = 'statsSolvingTimeNatural';
            else if (varIndex === MODEL_INDICES.REVIEW_TIME) termKey = 'statsReviewTimeNatural';

            if (termKey) {
                const z = Math.abs(coeff) / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = (1 - cdf) * 100;
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
                const t_eq_days = Math.exp(N / R) - 1;
                const firstTimeFailures = getReviewCandidates({
                    lowerProbabilityBound: 0.6,
                    upperProbabilityBound: 1.0,
                    requireLastAnswerIncorrect: true
                }).length;
                const firsttimeFailureTime = t_eq_days * 24;
                const allRepeats = getReviewCandidates({
                    lowerProbabilityBound: 0.6,
                    upperProbabilityBound: 1.0,
                    requireLastAnswerIncorrect: false
                }).length;

                resultText += getTerm('statsFailures').replace('{count_recent}', firstTimeFailures).replace('{hours}', Math.round(firsttimeFailureTime)) + '\n';
                resultText += getTerm('statsOldFailures').replace('{count_other}', allRepeats - firstTimeFailures) + '\n';
            }
        }

        const novelProblemCount = state.answerHistory.filter(p => !p.isRepeat).length;
        resultText += getTerm('statsNovelProblems').replace('{count}', novelProblemCount) + '\n';

        const slippedProblemCount = getReviewCandidates({
            lowerProbabilityBound: 0,
            upperProbabilityBound: 0.6,
            requireLastAnswerIncorrect: false
        }).length;
        const ignoredProblemCount = state.answerHistory.filter(p => p.ignored).length;
        resultText += getTerm('statsIgnoredProblems').replace('{count_ignored}', ignoredProblemCount).replace('{count_slipped}', slippedProblemCount) + '\n';

        // Separator
        resultText += '\n';

        // Block 3: OLS Model Coefficients
        resultText += getTerm('statsModelCoefficients');
        resultText += `\nINTERCEPT ${finalParams[MODEL_INDICES.INTERCEPT].toFixed(2)}`;
        for (const result of results) {
            const { coeff, se, varIndex } = result;
            const varName = Object.keys(MODEL_INDICES).find(key => MODEL_INDICES[key] === varIndex);
            if (varIndex !== MODEL_INDICES.PROBLEM_QUALITY && varIndex !== MODEL_INDICES.DATE) {
                resultText += `\n${varName}: ${coeff.toFixed(2)} ± ${se.toFixed(2)}`;
            }
        }

        alert(resultText);

    } catch (error) {
        console.error('Failed to generate stats text:', error);
        Sentry.captureException(error);
        alert(getTerm('statsFailed'));
    }
}
