// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateAnswerHistory, addToAccumulatedSolvingTime, setStatsResults, setModelParams, setAdamState } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS } from './constants.js';
import { getTerm } from './i18n.js';
import { getReviewCandidates } from './problem-loader.js';

// Indices for the features in our statistical model.
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

function probToRank(start, end) {
    const epsilon = 1e-9;
    start = Math.max(epsilon, Math.min(1 - epsilon, start));
    end = Math.max(epsilon, Math.min(1 - epsilon, end));
    const logOddsStart = Math.log(start / (1 - start));
    const logOddsEnd = Math.log(end / (1 - end));
    return (1 / 0.28) * (logOddsEnd - logOddsStart);
}

function logitsToRank(startLogit, endLogit) {
    return (1 / 0.28) * (endLogit - startLogit);
}

export async function sendReport(success) {
    try {
        const problemFilename = state.problemHistory[0].filename;
        const problemId = problemFilename.replace(/\.[^/.]+$/, "");
        await fetch('/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: state.userId, problem: problemId, success: success })
        });
    } catch (error) {
        console.error('Failed to send report:', error);
    }
}

const getRankInfo = (rating) => {
    const rankValue = rating * TOTAL_RANKS;
    const rankIndex = Math.floor(rankValue);
    const progress = (rankValue - rankIndex) * 100;
    return { rankIndex, progress };
}

export function updateUserStats(isCorrect) {
    const history = state.problemHistory;
    const currentProblem = history[0];

    if (!currentProblem || currentProblem.linked) {
        if (currentProblem.linked) updateProblemHistory(history);
        return {};
    }

    const currentSessionTime = state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0;
    addToAccumulatedSolvingTime(currentSessionTime);
    const timeTaken = state.accumulatedSolvingTime;

    const problemId = currentProblem.filename.replace('.sgf', '');
    let successSinceFailure;
    if (currentProblem.isRepeat) {
        const previousAnswers = state.answerHistory.filter(ans => ans.problemId === problemId);
        const lastAnswer = previousAnswers.pop();
        if (!isCorrect) {
            successSinceFailure = 0;
        } else {
            successSinceFailure = (lastAnswer && lastAnswer.successSinceFailure !== undefined) ? lastAnswer.successSinceFailure + 1 : 1;
        }
    }

    const answer = {
        problemId,
        problemRating: currentProblem.rating,
        problemQuality: currentProblem.quality || 0,
        timeTaken,
        reviewTime: 0,
        correct: isCorrect,
        date: Date.now(),
        successSinceFailure,
        problemType: !currentProblem.isRepeat ? 'novel' : 'repeat',
    };

    updateAnswerHistory([...state.answerHistory, answer]);
    state.currentSolvingStartTime = null;

    const oldRank = state.userRank;
    let rank = oldRank;
    let momentum = state.momentum;
    const scalingFactor = (currentProblem.quality || 0.5) * (currentProblem.isRepeat ? 1 / 5 : 1);

    if (isCorrect) {
        const maxIncrease = (1.0 - rank) / 9;
        let intendedRatingIncrease = scalingFactor * momentum / 600;
        let intendedMomentumIncrease = scalingFactor * 0.1;
        if (intendedRatingIncrease > maxIncrease) {
            intendedMomentumIncrease *= maxIncrease / intendedRatingIncrease;
            intendedRatingIncrease = maxIncrease;
        }
        if (!currentProblem.isRepeat) {
            momentum *= 1 + intendedMomentumIncrease;
        }
        rank += intendedRatingIncrease;
    } else {
        momentum -= scalingFactor * (momentum - (momentum + 1) / 3);
        rank = Math.max(0, rank - 4 * scalingFactor * (momentum / 600));
    }

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemHistory(history);
    updateAllUI();

    const oldRankInfo = getRankInfo(oldRank);
    const newRankInfo = getRankInfo(state.userRank);
    let rankChangeDirection = 'same';
    if (newRankInfo.rankIndex > oldRankInfo.rankIndex) rankChangeDirection = 'up';
    else if (newRankInfo.rankIndex < oldRankInfo.rankIndex) rankChangeDirection = 'down';

    setTimeout(calculateAndStoreStatsResults, 0);

    return { rankChange: state.userRank - oldRank, rankChangeDirection, isCorrect };
}


let isCalculatingStats = false;

async function calculateAndStoreStatsResults() {
    if (isCalculatingStats) return;
    isCalculatingStats = true;

    try {
        const validHistory = state.answerHistory.filter(p => p.problemRating !== undefined && p.timeTaken !== undefined && p.date !== undefined);
        if (validHistory.length < 10) {
            setStatsResults(null);
            return;
        }
        const history = validHistory.slice(-500);

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
    const percentileTime = timeTakenValues.map(t => {
        const below = sortedTime.filter(s => s < t).length;
        const equal = sortedTime.filter(s => s === t).length;
        return n > 0 ? (below + 0.5 * equal) / n : 0;
    });

    const previousReviewTimes = history.map(p => {
        if (p.problemType === 'repeat') {
            const prevs = validHistory.filter(ans => ans.problemId === p.problemId && ans.date < p.date);
            return prevs.reduce((sum, ans) => sum + (ans.reviewTime || 0), 0);
        }
        return 0;
    });
    const reviewTimes = previousReviewTimes.filter(t => t > 0);
    let time_per_percentile_min_r = 0;
    if (reviewTimes.length > 0) {
        const sortedReview = [...reviewTimes].sort((a, b) => a - b);
        const idx_low_r = Math.floor(p_low * (sortedReview.length - 1));
        const idx_high_r = Math.ceil(p_high * (sortedReview.length - 1));
        time_per_percentile_min_r = (sortedReview[idx_high_r] - sortedReview[idx_low_r]) / (60000 * (p_high - p_low));
    }
    const percentileReview = previousReviewTimes.map(t => {
        if (t === 0 || reviewTimes.length === 0) return 0;
        const below = reviewTimes.filter(s => s < t).length;
        const equal = reviewTimes.filter(s => s === t).length;
        return (below + 0.5 * equal) / reviewTimes.length;
    });

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
    const lambda = 0.001; // L2 regularization parameter

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
            features[MODEL_INDICES.PROBLEM_QUALITY] = p.problemQuality || 0.5;
            features[MODEL_INDICES.DATE] = i / n;
            if (p.problemType === 'novel') {
                features[MODEL_INDICES.IS_NOVEL] = 1;
            } else {
                const prevAnswer = validHistory.filter(ans => ans.problemId === p.problemId && ans.date < p.date).pop();
                const t = p.date - (prevAnswer ? prevAnswer.date : p.date);
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
            const t = p.date - (prevAnswer ? prevAnswer.date : p.date);
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

    if (isCalculatingStats) {
        Sentry.captureMessage("showDetailedStats called while isCalculatingStats is true, showing previous results");
    } else {
        await calculateAndStoreStatsResults();
    }

    if (state.statsResults === null) {
        alert(getTerm('statsFailed'));
        return;
    }

    try {
        const { results } = state.statsResults;
        const analysisProblems = Math.min(state.answerHistory.length, 500);
        const observations = [];
        
        // Get the raw model parameters from state
        const finalParams = state.modelParams || Array(Object.keys(MODEL_INDICES).length).fill(0);

        // Calculate base probability for a typical non-novel problem
        const base_logit = finalParams[MODEL_INDICES.INTERCEPT] +
            finalParams[MODEL_INDICES.TIME_SOLVING] * 0.5 +
            finalParams[MODEL_INDICES.PROBLEM_DIFFICULTY] * state.userRank +
            finalParams[MODEL_INDICES.PROBLEM_QUALITY] * 0.5 +
            finalParams[MODEL_INDICES.DATE] * 0.5;
        const base_prob = 1 / (1 + Math.exp(-base_logit));

        if( true ) {
            // Console log all logit weights and standard deviations
            console.log("=== Logit Weights and Standard Deviations ===");

            // Calculate standard errors using the results (reverse the scaling)
            const stdErrors = Array(Object.keys(MODEL_INDICES).length).fill(0);
            results.forEach(result => {
                const scaleFactor = result.scaleFactor || 1;
                stdErrors[result.varIndex] = result.se / Math.abs(scaleFactor);
            });

            // Log each parameter with its name
            Object.entries(MODEL_INDICES).forEach(([name, index]) => {
                const mean = finalParams[index];
                const sd = stdErrors[index];
                console.log(`${index}_${name} : ${mean.toFixed(4)}±${sd.toFixed(4)}`);
            });
            console.log("==============================================");
        }

        for (const result of results) {
            const { coeff, se, varIndex } = result;
            if (varIndex === MODEL_INDICES.IS_NOVEL || 
                varIndex === MODEL_INDICES.RECALL_FORMULA) continue;

            let termKey;
            if (varIndex === MODEL_INDICES.TIME_SOLVING) termKey = 'statsSolvingTime';
            else if (varIndex === MODEL_INDICES.REVIEW_TIME) termKey = 'statsReviewTime';
            else if (varIndex === MODEL_INDICES.PROBLEM_QUALITY) termKey = 'statsProblemQuality';

            if (termKey) {
                const z = Math.abs(coeff) / se;
                const conf = (2 * (0.5 * (1 + math.erf(z / Math.sqrt(2)))) - 1) * 100;

                const rank_change = logitsToRank(base_logit, base_logit + coeff);
                const rank_stddev = Math.abs(se / 0.28);
                const formattedRankChange = rank_change > 0 ? `+${rank_change.toFixed(1)}` : rank_change.toFixed(1);
                observations.push({
                    message: getTerm(termKey).replace('{rank_change}', formattedRankChange).replace('{stddev}', rank_stddev.toFixed(1)),
                    conf
                });
            }
        }
        observations.sort((a, b) => b.conf - a.conf);

        const allLines = observations.map(o => o.message);

        // Refactored calculation for the forgetting curve
        const { varCovarMatrix } = state.statsResults;
        if (varCovarMatrix) {
            const N_idx = MODEL_INDICES.IS_NOVEL;
            const R_idx = MODEL_INDICES.RECALL_FORMULA;

            const N = finalParams[N_idx];
            const R = finalParams[R_idx];

            if (Math.abs(R) > 1e-9) {
                const t_eq_ms = Math.exp(N / R) - 1;

                const k = (24 * 3600 * 1000) / 0.28; // 24 hours in milliseconds
                const rank_change_per_day = k * R * Math.exp(-N / R);

                const var_N = varCovarMatrix.get([N_idx, N_idx]);
                const var_R = varCovarMatrix.get([R_idx, R_idx]);
                const cov_NR = varCovarMatrix.get([N_idx, R_idx]);

                const exp_term = Math.exp(-N / R);
                const d_dN = -k * exp_term;
                const d_dR = k * exp_term * (1 + N / R);

                const var_f = d_dN**2 * var_N + d_dR**2 * var_R + 2 * d_dN * d_dR * cov_NR;
                const rank_stddev_per_day = Math.sqrt(var_f);

                let time_point_text = "";
                if (t_eq_ms > 0) {
                    const hours = t_eq_ms / (3600 * 1000);
                    const days = Math.floor(hours / 24);
                    const remaining_hours = hours % 24;

                    if (days > 0) {
                        time_point_text = `${days}d ${remaining_hours.toFixed(1)}h`;
                    } else {
                        time_point_text = `${hours.toFixed(1)}h`;
                    }
                }

                const formattedRankChange = rank_change_per_day > 0 ? `+${rank_change_per_day.toFixed(2)}` : rank_change_per_day.toFixed(2);
                const outputStdDev = isFinite(rank_stddev_per_day) ? rank_stddev_per_day.toFixed(2) : "???";
                allLines.push(getTerm('statsForgettingCurveWithStdDev').replace('{rank_change}', formattedRankChange).replace('{stddev}', outputStdDev));

                const reviewCandidatesForCount = getReviewCandidates({
                    lowerProbabilityBound: 0,
                    upperProbabilityBound: Infinity,
                    requireLastAnswerIncorrect: true
                });
                const x = reviewCandidatesForCount.length;
                const y = t_eq_ms / (3600 * 1000);

                if (x > 0 && y > 0) {
                    allLines.push(getTerm('statsFailuresRepeat').replace('{count}', x).replace('{hours}', Math.round(y)));
                }
            }
        }

        let resultText;
        const totalProblems = state.answerHistory.length;
        if (allLines.length === 0) {
            // This case should be practically unreachable now that forgetting curve text is unconditioned
            resultText = getTerm('statsNoTrends').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
        } else {
            const intro = getTerm('statsIntroNew').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro + "- " + allLines.join("\n- ");
        }
        alert(resultText);
    } catch (error) {
        console.error('Failed to generate stats text:', error);
        Sentry.captureException(error);
        alert('Could not display statistical analysis.');
    }
}