// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateAnswerHistory, addToAccumulatedSolvingTime, setStatsResults, setUserK } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS, OLS_INDICES } from './constants.js';
import { getTerm } from './i18n.js';
import { calculateRecallRate } from './problem-loader.js';

// Function to send report to Cloudflare Worker
export async function sendReport(success) {
    try {
        const problemFilename = state.problemHistory[0].filename;
        const problemId = problemFilename.replace(/\.[^/.]+$/, "");
        
        const reportData = {
            userId: state.userId,
            problem: problemId,
            success: success
        };
        
        await fetch('/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(reportData)
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

    if (!currentProblem) return {};

    if (currentProblem.linked) {
        updateProblemHistory(history);
        return {};
    }

    const currentSessionTime = state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0;
    addToAccumulatedSolvingTime(currentSessionTime);
    const timeTaken = state.accumulatedSolvingTime;

    let successSinceFailure;
    const problemId = currentProblem.filename.replace('.sgf', '');
    const previousAnswers = state.answerHistory.filter(ans => ans.problemId === problemId);
    const lastAnswer = previousAnswers.length > 0 ? previousAnswers[previousAnswers.length - 1] : null;

    if (!isCorrect) {
        successSinceFailure = 0;
    } else { // isCorrect
        if (lastAnswer && lastAnswer.successSinceFailure !== undefined) {
            successSinceFailure = lastAnswer.successSinceFailure + 1;
        } else if (previousAnswers.some(a => a.correct === false)) {
            // First success after a failure
            successSinceFailure = 1;
        }
        // If it was never failed, successSinceFailure remains undefined.
    }

    const answer = {
        problemId: currentProblem.filename.replace('.sgf', ''),
        problemRating: currentProblem.rating,
        problemQuality: currentProblem.quality || 0,
        timeTaken,
        reviewTime: 0,
        correct: isCorrect,
        date: Date.now(),
        successSinceFailure,
        problemType: !currentProblem.isRepeat ? 'novel' : 'repeat',
    };

    const answerHistory = [...state.answerHistory, answer];
    updateAnswerHistory(answerHistory);

    // Reset solving timer after recording
    state.currentSolvingStartTime = null;

    const oldRank = state.userRank;
    const oldRankInfo = getRankInfo(oldRank);
    let rank = oldRank;
    let momentum = state.momentum;

    if (isCorrect ) {
        const maxIncrease = (1.0 - rank) / 9;
        let intendedIncrease = momentum / 600;
        if(intendedIncrease > maxIncrease) {
            momentum *= maxIncrease / intendedIncrease;
        } else {
            momentum *= 1.1;
        }
        intendedIncrease *= (currentProblem.quality || 0.5);
        rank += Math.min(intendedIncrease, maxIncrease);
    } else if (!isCorrect) { // Incorrect
        momentum = (momentum + 1) / 3;
        let rankChange = 4 * (momentum / 600);
        rankChange *= (currentProblem.quality || 0.5);
        rank = Math.max(0, rank - rankChange);
    }

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemHistory(history);
    updateAllUI();

    const newRank = state.userRank;
    const newRankInfo = getRankInfo(newRank);

    const rankChange = newRank - oldRank;

    let rankChangeDirection = 'same';
    if (newRankInfo.rankIndex > oldRankInfo.rankIndex) {
        rankChangeDirection = 'up';
    } else if (newRankInfo.rankIndex < oldRankInfo.rankIndex) {
        rankChangeDirection = 'down';
    }

    // Trigger background calculation of detailed stats and K estimation
    setTimeout(() => {
        calculateAndStoreStatsResults();
        estimateUserK();
    }, 0);

    return {
        rankChange,
        rankChangeDirection,
        isCorrect
    };
}

function estimateUserK() {
    const B = 1.3;

    // Prior data points to anchor the estimation
    const priors = [
        { t: 82800000, r: 0, correct: true }, // 23 hours
        { t: 82800000, r: 0, correct: true }, // 23 hours
        { t: 82800000, r: 0, correct: true }, // 23 hours
        { t: 90000000, r: 0, correct: false }, // 25 hours
    ];

    const history = [];
    const answersByProblem = {};
    const answerHistory = state.answerHistory.slice(-500);
    for (const answer of answerHistory) {
        if (!answersByProblem[answer.problemId]) {
            answersByProblem[answer.problemId] = [];
        }
        answersByProblem[answer.problemId].push(answer);
    }

    for (const problemId in answersByProblem) {
        const problemAnswers = answersByProblem[problemId];
        for (let i = 1; i < problemAnswers.length; i++) {
            const prevAnswer = problemAnswers[i - 1];
            const currentAnswer = problemAnswers[i];

            if (prevAnswer.successSinceFailure !== undefined) {
                const t = currentAnswer.date - prevAnswer.date;
                const r = prevAnswer.successSinceFailure;
                if (t > 0) {
                    history.push({ t, r, correct: currentAnswer.correct });
                }
            }
        }
    }

    if (history.length === 0) {
        return; // Not enough data to estimate K
    }

    const fullHistory = [...priors, ...history];

    const logLikelihoodDerivative = (k) => {
        let derivative = 0;
        for (const item of fullHistory) {
            const { t, r, correct } = item;
            const R = calculateRecallRate(t, r, k);
            
            if (R === 1) { // Avoid division by zero if R is 1
                if (!correct) derivative -= 1e9; // Large penalty
                continue;
            }
            if (R === 0) { // Avoid division by zero if R is 0
                if (correct) derivative += 1e9; // Large penalty
                continue;
            }

            const c = Math.pow(2.5, r);
            const dR_dk = (B * t * Math.pow(1 + t / (c * k), -B - 1)) / (c * k * k);

            if (correct) {
                derivative += dR_dk / R;
            } else {
                derivative -= dR_dk / (1 - R);
            }
        }
        return derivative;
    };

    let k_min = 1e7;
    let k_max = 1e11;
    let d_min = logLikelihoodDerivative(k_min);
    let d_max = logLikelihoodDerivative(k_max);

    if (d_min < 0) {
        setUserK(k_min);
        return;
    }
    if (d_max > 0) {
        setUserK(k_max);
        return;
    }

    for (let i = 0; i < 10; i++) { // 10 iterations for high precision
        const k_mid = (k_min + k_max) / 2;
        const d_mid = logLikelihoodDerivative(k_mid);
        if (d_mid > 0) {
            k_min = k_mid;
        } else {
            k_max = k_mid;
        }
    }

    const estimatedK = (k_min + k_max) / 2;
    setUserK(estimatedK);
}

let isCalculatingStats = false;

async function calculateAndStoreStatsResults() {
    if (isCalculatingStats) return;
    isCalculatingStats = true;
    try {
        const validHistory = state.answerHistory.filter(p =>
            p.problemRating !== undefined &&
            p.problemQuality !== undefined &&
            p.timeTaken !== undefined &&
            p.reviewTime !== undefined &&
            p.date !== undefined
        );

        if (validHistory.length < 5) {
            setStatsResults(null);
            return;
        }
        const history = validHistory.slice(-500);

        // Time taken calculations (no changes)
        const timeTakenValues = history.map(p => p.timeTaken);
        const sortedTime = timeTakenValues.slice().sort((a, b) => a - b);
        const p_low = 0.45;
        const p_high = 0.55;
        const idx_low = Math.floor(p_low * (sortedTime.length - 1));
        const idx_high = Math.floor(p_high * (sortedTime.length - 1));
        const time_low = sortedTime[idx_low];
        const time_high = sortedTime[idx_high];
        const delta_time_ms = time_high - time_low;
        const time_per_percentile_min = delta_time_ms / (60000 * 0.1);
        const percentileTime = timeTakenValues.map(t => {
            const below = sortedTime.filter(s => s < t).length;
            const equal = sortedTime.filter(s => s === t).length;
            return (below + 0.5 * equal) / sortedTime.length;
        });

        // Merged Review Time calculations
        const previousReviewTimes = history.map((p, i) => {
            if (p.problemType === 'repeat') {
                const prevs = history.slice(0, i).filter(ans => ans.problemId === p.problemId);
                return prevs.reduce((sum, ans) => sum + ans.reviewTime, 0);
            }
            return 0;
        });

        const reviewTimes = history.map((p, i) => p.problemType === 'repeat' ? previousReviewTimes[i] : 0).filter(t => t > 0);
        let time_per_percentile_min_r = 0;
        if (reviewTimes.length > 0) {
            const sortedReview = reviewTimes.slice().sort((a, b) => a - b);
            const p_low_r = 0.45;
            const p_high_r = 0.55;
            const idx_low_r = Math.floor(p_low_r * (sortedReview.length - 1));
            const idx_high_r = Math.floor(p_high_r * (sortedReview.length - 1));
            const time_low_r = sortedReview[idx_low_r];
            const time_high_r = sortedReview[idx_high_r];
            const delta_time_ms_r = time_high_r - time_low_r;
            time_per_percentile_min_r = delta_time_ms_r / (60000 * 0.1);
        }

        const percentileReview = previousReviewTimes.map((t, i) => {
            if (t === 0 || history[i].problemType !== 'repeat' || reviewTimes.length === 0) return 0;
            const below = reviewTimes.filter(s => s < t).length;
            const equal = reviewTimes.filter(s => s === t).length;
            return (below + 0.5 * equal) / reviewTimes.length;
        });

        const y = history.map(p => p.correct ? 1 : 0);
        const X_original = history.map((p, i) => [
            1,
            percentileTime[i],
            percentileReview[i],
            p.problemRating,
            p.problemQuality,
            i,
            p.problemType === 'repeat' ? 1 : 0,
        ]);

        const stdDevs = [];
        for (let j = 1; j < X_original[0].length; j++) {
            const col = X_original.map(row => row[j]);
            stdDevs.push(math.std(col));
        }

        const includedIndices = [0];
        for (let j = 1; j < stdDevs.length + 1; j++) {
            if (stdDevs[j - 1] > 0) includedIndices.push(j);
        }

        const X = X_original.map(row => includedIndices.map(idx => row[idx]));
        const n = X.length;
        const k = X[0].length;

        const Xt = math.transpose(X);
        const XtX = math.multiply(Xt, X);
        const XtX_inv = math.inv(XtX);
        const Xty = math.multiply(Xt, y);
        const beta = math.multiply(XtX_inv, Xty);

        const y_hat = math.multiply(X, beta);
        const residuals = math.subtract(y, y_hat);
        const ssr = math.multiply(math.transpose(residuals), residuals);
        const s2 = ssr / (n - k);
        const varCovarMatrix = math.multiply(s2, XtX_inv);
        const stdErrors = math.map(math.diag(varCovarMatrix), math.sqrt);

        const diff_idx_in_X = includedIndices.indexOf(OLS_INDICES.PROBLEM_DIFFICULTY + 1);
        let beta_diff = 0;
        if (diff_idx_in_X > -1) {
            beta_diff = beta[diff_idx_in_X];
        }

        const results = [];
        for (let i = 1; i < includedIndices.length; i++) {
            const varIndex = includedIndices[i] - 1;
            let coeff = beta[i];
            let se = stdErrors[i];

            let scaleFactor = 1;
            if (varIndex === OLS_INDICES.TIME_SOLVING && time_per_percentile_min > 0) scaleFactor = 1 / time_per_percentile_min;
            else if (varIndex === OLS_INDICES.REVIEW_TIME && time_per_percentile_min_r > 0) scaleFactor = 1 / time_per_percentile_min_r;
            else if (varIndex === OLS_INDICES.PROBLEM_DIFFICULTY) scaleFactor = 1 / 29;
            else if (varIndex === OLS_INDICES.PROBLEM_QUALITY) scaleFactor = -1;
            else if (varIndex === OLS_INDICES.DATE) scaleFactor = n;

            coeff *= scaleFactor;
            se *= Math.abs(scaleFactor);
            
            results.push({ coeff, se, varIndex });
        }
        setStatsResults({ results, beta_diff });

    } catch (error) {
        console.error('OLS Regression failed during background calculation:', error);
        Sentry.captureException(error);
        setStatsResults(null); // Set empty results on failure
    } finally {
        isCalculatingStats = false;
    }
}

// Function to show detailed stats when user clicks on rank
export async function showDetailedStats() {
    if (isCalculatingStats) {
        Sentry.captureMessage("showDetailedStats called while isCalculatingStats is true");
        return;
    }

    if (state.statsResults === null && state.answerHistory.length >= 5) {
        await calculateAndStoreStatsResults();
    }

    if (state.answerHistory.length < 5) {
        alert(getTerm('statsNotEnough').replace('{count}', state.answerHistory.length));
        return;
    }

    if (state.statsResults === null) {
        alert(getTerm('statsFailed'));
        return;
    }

    try {
        const { results, beta_diff } = state.statsResults;
        const analysisProblems = Math.min(state.answerHistory.length, 500);

        const rank_conversion_factor = 1 / 0.28; // User-defined constant
        const observations = [];

        for (const result of results) {
            const { coeff, se, varIndex } = result;
            if (se > 0) {
                const z = Math.abs(coeff) / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = (2 * cdf - 1) * 100;

                if (conf >= 0 && !([OLS_INDICES.TIME_SOLVING, OLS_INDICES.REVIEW_TIME].includes(varIndex) && coeff <= 0)) {
                    const rank_change = coeff * rank_conversion_factor;

                    if (Math.abs(rank_change) < 0.1) {
                        continue;
                    }

                    let termKey;
                    let rank_change_val = rank_change;

                    if (varIndex === OLS_INDICES.TIME_SOLVING) termKey = 'statsSolvingTime';
                    else if (varIndex === OLS_INDICES.REVIEW_TIME) termKey = 'statsReviewTime';
                    else if (varIndex === OLS_INDICES.PROBLEM_QUALITY) {
                        termKey = rank_change > 0 ? 'statsQualityAnalytical' : 'statsQualityIntuitive';
                        rank_change_val = Math.abs(rank_change);
                    } else if (varIndex === OLS_INDICES.DATE) {
                        termKey = rank_change > 0 ? 'statsProgress' : 'statsRegression';
                        rank_change_val = Math.abs(rank_change);
                    }

                    if (termKey) {
                        let message = getTerm(termKey)
                            .replace('{rank_change}', rank_change_val.toFixed(1))
                            .replace('{conf}', conf.toFixed(0));
                        
                        if (varIndex === OLS_INDICES.DATE) {
                            message = message.replace('{analysis_count}', analysisProblems);
                        }
                        
                        observations.push({ message, conf });
                    }
                }
            }
        }

        observations.sort((a, b) => b.conf - a.conf);

        const totalProblems = state.answerHistory.length;
        let resultText = "";

        const validObservations = observations.filter(o => o.message);
        if (validObservations.length > 0) {
            const intro = getTerm('statsIntroNew').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro + "- " + validObservations.map(o => o.message).join("\n- ");
        } else {
            const intro = getTerm('statsNoTrends').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro;
        }

        if (state.userK) {
            // t = k * (R^(-1/B) - 1) where R=0.8, B=1.3
            const t = 0.2966 * state.userK;
            const hours = Math.round(t / (1000 * 3600));
            const forgettingLine = getTerm('statsForgettingCurve').replace('{hours}', hours);
            resultText += '\n- ' + forgettingLine;
        }

        alert(resultText);
    } catch (error) {
        console.error('Failed to generate stats text:', error);
        alert('Could not display statistical analysis.');
    }
}