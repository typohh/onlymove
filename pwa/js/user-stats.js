// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateTsumegoStats, updateAnswerHistory, addToAccumulatedSolvingTime } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS, PROBLEM_STATUS } from './constants.js';
import { getTerm } from './i18n.js';

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

    const oldStatus = currentProblem.status;

    if (isCorrect) {
        if (oldStatus === PROBLEM_STATUS.FAILURE) {
            currentProblem.status = PROBLEM_STATUS.IN_BETWEEN;
        } else if (oldStatus === PROBLEM_STATUS.IN_BETWEEN) {
            currentProblem.status = PROBLEM_STATUS.SUCCESS;
        }
    } else {
        currentProblem.status = PROBLEM_STATUS.FAILURE;
    }

    if (currentProblem.linked) {
        updateProblemHistory(history);
        return {};
    }

    const currentSessionTime = state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0;
    addToAccumulatedSolvingTime(currentSessionTime);
    const timeTaken = state.accumulatedSolvingTime;
    const answer = {
        problemId: currentProblem.filename.replace('.sgf', ''),
        problemRating: currentProblem.rating,
        problemQuality: currentProblem.quality || 0,
        timeTaken,
        reviewTime: 0,
        correct: isCorrect,
        date: Date.now(),
        problemType: !currentProblem.isRepeat
            ? 'novel'
            : oldStatus === PROBLEM_STATUS.FAILURE
                ? 'short-term-repeat'
                : oldStatus === PROBLEM_STATUS.IN_BETWEEN
                    ? 'long-term-repeat'
                    : 'repeat',
    };

    const answerHistory = [...state.answerHistory, answer];
    updateAnswerHistory(answerHistory);

    // Reset solving timer after recording
    state.currentSolvingStartTime = null;

    const oldRank = state.userRank;
    const oldRankInfo = getRankInfo(oldRank);
    let rank = oldRank;
    let momentum = state.momentum;
    const stats = { ...state.tsumegoStats };

    if (oldStatus === PROBLEM_STATUS.FAILURE) {
        stats.totalOnFail++;
        if (!isCorrect) {
            stats.failOnFail++;
        }
    } else if (oldStatus === PROBLEM_STATUS.IN_BETWEEN) {
        stats.totalOnInBetween++;
        if (!isCorrect) {
            stats.failOnInBetween++;
        }
    } else { // success or new
        stats.totalOnSuccess++;
        if (!isCorrect) {
            stats.failOnSuccess++;
        }
    }

    if (isCorrect ) {
        const maxIncrease = (1.0 - rank) / 9;
        let intendedIncrease = momentum / 600;
        if( oldStatus !== PROBLEM_STATUS.FAILURE && oldStatus !== PROBLEM_STATUS.IN_BETWEEN ) {
            if(intendedIncrease > maxIncrease) {
                momentum *= maxIncrease / intendedIncrease;
            } else {
                momentum *= 1.1;
            }
        } else {
            intendedIncrease *= 0.1
        }
        rank += Math.min(intendedIncrease, maxIncrease);
    } else if (!isCorrect) { // Incorrect
        momentum = (momentum + 1) / 3;
        rank = Math.max(0, rank - 4 * (momentum / 600));
    }

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemHistory(history);
    updateTsumegoStats(stats);
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

    return {
        rankChange,
        rankChangeDirection,
        isCorrect
    };
}

// Function to show detailed stats when user clicks on rank
export function showDetailedStats() {
    // Remove invalid answerHistory entries that lack required fields
    const originalLength = state.answerHistory.length;
    const validHistory = state.answerHistory.filter(p =>
        p.problemRating !== undefined &&
        p.problemQuality !== undefined &&
        p.timeTaken !== undefined &&
        p.reviewTime !== undefined &&
        p.date !== undefined
    );
    const removedCount = originalLength - validHistory.length;
    if (removedCount > 0) {
        console.warn(`Removed ${removedCount} invalid answerHistory entries out of ${originalLength}`);
        updateAnswerHistory(validHistory);
    }

    const history = validHistory.slice(-500);

    if (history.length < 5) { // Need enough data for regression
        alert(getTerm('statsNotEnough').replace('{count}', history.length));
        return;
    }

    try {
        // Transform timeTaken to percentiles
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

        // Calculate previous review time for repeats
        const previousReviewTimes = history.map((p, i) => {
            if (p.problemType === 'short-term-repeat' || p.problemType === 'long-term-repeat') {
                // Find the last previous answer for same problemId with reviewTime > 0
                const prev = history.slice(0, i).reverse().find(ans => ans.problemId === p.problemId && ans.reviewTime > 0);
                return prev ? prev.reviewTime : 0;
            }
            return 0;
        });

        // Transform review times to percentiles for short-term and long-term separately
        const shortTermReviewTimes = history.map((p, i) => p.problemType === 'short-term-repeat' ? previousReviewTimes[i] : 0).filter(t => t > 0);
        const longTermReviewTimes = history.map((p, i) => p.problemType === 'long-term-repeat' ? previousReviewTimes[i] : 0).filter(t => t > 0);
        let percentileReviewShort = previousReviewTimes.map(t => 0);
        let percentileReviewLong = previousReviewTimes.map(t => 0);
        let time_per_percentile_min_r_short = 0;
        let time_per_percentile_min_r_long = 0;
        if (shortTermReviewTimes.length > 0) {
            const sortedReviewShort = shortTermReviewTimes.slice().sort((a, b) => a - b);
            const p_low_r = 0.45;
            const p_high_r = 0.55;
            const idx_low_r = Math.floor(p_low_r * (sortedReviewShort.length - 1));
            const idx_high_r = Math.floor(p_high_r * (sortedReviewShort.length - 1));
            const time_low_r = sortedReviewShort[idx_low_r];
            const time_high_r = sortedReviewShort[idx_high_r];
            const delta_time_ms_r = time_high_r - time_low_r;
            time_per_percentile_min_r_short = delta_time_ms_r / (60000 * 0.1);
            percentileReviewShort = previousReviewTimes.map((t, i) => {
                if (t === 0 || history[i].problemType !== 'short-term-repeat') return 0;
                const below = sortedReviewShort.filter(s => s < t).length;
                const equal = sortedReviewShort.filter(s => s === t).length;
                return (below + 0.5 * equal) / sortedReviewShort.length;
            });
        }
        if (longTermReviewTimes.length > 0) {
            const sortedReviewLong = longTermReviewTimes.slice().sort((a, b) => a - b);
            const p_low_r = 0.45;
            const p_high_r = 0.55;
            const idx_low_r = Math.floor(p_low_r * (sortedReviewLong.length - 1));
            const idx_high_r = Math.floor(p_high_r * (sortedReviewLong.length - 1));
            const time_low_r = sortedReviewLong[idx_low_r];
            const time_high_r = sortedReviewLong[idx_high_r];
            const delta_time_ms_r = time_high_r - time_low_r;
            time_per_percentile_min_r_long = delta_time_ms_r / (60000 * 0.1);
            percentileReviewLong = previousReviewTimes.map((t, i) => {
                if (t === 0 || history[i].problemType !== 'long-term-repeat') return 0;
                const below = sortedReviewLong.filter(s => s < t).length;
                const equal = sortedReviewLong.filter(s => s === t).length;
                return (below + 0.5 * equal) / sortedReviewLong.length;
            });
        }

        const y = history.map(p => p.correct ? 1 : 0);
        const X_original = history.map((p, i) => [
            1, // Intercept
            percentileTime[i], // Time taken percentile
            percentileReviewShort[i], // Short-term review time percentile
            percentileReviewLong[i], // Long-term review time percentile
            p.problemRating,
            p.problemQuality,
            i, // History index
            p.problemType === 'short-term-repeat' ? 1 : 0,
            p.problemType === 'long-term-repeat' ? 1 : 0
        ]);

        // Check stddev for each variable (skip intercept)
        const stdDevs = [];
        for (let j = 1; j < X_original[0].length; j++) {
            const col = X_original.map(row => row[j]);
            const stdDev = math.std(col);
            stdDevs.push(stdDev);
        }

        // Include intercept always, exclude variables with 0 stddev
        const includedIndices = [0]; // intercept
        for (let j = 1; j < stdDevs.length + 1; j++) {
            if (stdDevs[j - 1] > 0) {
                includedIndices.push(j);
            }
        }

        // Create X with only included columns
        const X = X_original.map(row => includedIndices.map(idx => row[idx]));
        const n = X.length;
        const k = X[0].length;

        const Xt = math.transpose(X);
        const XtX = math.multiply(Xt, X);
        const XtX_inv = math.inv(XtX);
        const Xty = math.multiply(Xt, y);
        const beta = math.multiply(XtX_inv, Xty);

        // Calculate Standard Errors
        const y_hat = math.multiply(X, beta);
        const residuals = math.subtract(y, y_hat);
        const ssr = math.multiply(math.transpose(residuals), residuals);
        const s2 = ssr / (n - k);
        const varCovarMatrix = math.multiply(s2, XtX_inv);
        const stdErrors = math.map(math.diag(varCovarMatrix), math.sqrt);

        const variableNames = [getTerm('timeSpent'), getTerm('reviewTimeShortTerm'), getTerm('reviewTimeLongTerm'), getTerm('problemDifficulty'), getTerm('problemQuality'), getTerm('date'), getTerm('shortTermRepeat'), getTerm('longTermRepeat')];
        const finalValues = new Array(variableNames.length).fill('-');
        const finalValues65 = new Array(variableNames.length).fill('-');
        const coeffs = new Array(variableNames.length).fill(0);

        for (let i = 1; i < includedIndices.length; i++) { // Skip intercept
            const varIndex = includedIndices[i] - 1;
            let coeff = beta[i];
            let se = stdErrors[i];

            // Scale coefficient and SE
            let scaleFactor = 1;
            if (varIndex === 0 && time_per_percentile_min > 0) { // timeSpent
                scaleFactor = 1 / time_per_percentile_min;
            } else if (varIndex === 1 && time_per_percentile_min_r_short > 0) { // reviewTimeShortTerm
                scaleFactor = 1 / time_per_percentile_min_r_short;
            } else if (varIndex === 2 && time_per_percentile_min_r_long > 0) { // reviewTimeLongTerm
                scaleFactor = 1 / time_per_percentile_min_r_long;
            } else if (varIndex === 3) { // problemDifficulty
                scaleFactor = 1 / 29;
            } else if (varIndex === 4) { // problemQuality
                scaleFactor = -1;
            } else if (varIndex === 5) { // date
                scaleFactor = 100; // Scale to per 100 problems
            }

            coeff *= scaleFactor;
            coeffs[varIndex] = coeff;
            se *= Math.abs(scaleFactor); // Use absolute for SE

            // Calculate 85% CI
            const lowerBound = coeff - 1.44 * se;
            const upperBound = coeff + 1.44 * se;

            let finalCoeff = 0;
            if (lowerBound > 0) {
                finalCoeff = lowerBound; // Entirely positive CI
            } else if (upperBound < 0) {
                finalCoeff = upperBound; // Entirely negative CI
            }

            finalCoeff *= 100; // Convert to %
            finalValues[varIndex] = finalCoeff.toFixed(1);

            // Calculate 65% CI
            const lowerBound65 = coeff - 0.93 * se;
            const upperBound65 = coeff + 0.93 * se;

            let finalCoeff65 = 0;
            if (lowerBound65 > 0) {
                finalCoeff65 = lowerBound65; // Entirely positive CI
            } else if (upperBound65 < 0) {
                finalCoeff65 = upperBound65; // Entirely negative CI
            }

            finalCoeff65 *= 100; // Convert to %
            finalValues65[varIndex] = finalCoeff65.toFixed(1);
        }

        const finalNumericValues = finalValues.map(v => v === '-' ? 0 : parseFloat(v));
        const finalNumericValues65 = finalValues65.map(v => v === '-' ? 0 : parseFloat(v));

        const observations = [];
        const handledIndices = [0, 1, 2, 4, 5];
        for (let idx of handledIndices) {
            const coeff = coeffs[idx];
            const se = stdErrors[includedIndices.indexOf(idx + 1)]; // since includedIndices has 1-based
            if (se > 0) {
                const z = Math.abs(coeff) / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = (2 * cdf - 1) * 100;
                if (conf >= 50) {
                    observations.push({mean: (coeff * 100).toFixed(1), conf: conf.toFixed(0), varIndex: idx});
                }
            }
        }
        observations.sort((a, b) => b.conf - a.conf);

        for (let obs of observations) {
            let termKey;
            let meanVal = obs.mean;
            let meanNum = parseFloat(obs.mean);
            if (obs.varIndex === 0) {
                termKey = 'statsSolvingTime';
            } else if (obs.varIndex === 1) {
                termKey = 'statsReviewTimeShort';
            } else if (obs.varIndex === 2) {
                termKey = 'statsReviewTimeLong';
            } else if (obs.varIndex === 4) {
                termKey = meanNum > 0 ? 'statsQualityAnalytical' : 'statsQualityIntuitive';
                meanVal = Math.abs(meanNum).toFixed(1);
            } else if (obs.varIndex === 5) {
                termKey = meanNum > 0 ? 'statsProgress' : 'statsRegression';
                meanVal = Math.abs(meanNum).toFixed(1);
            }
            obs.message = getTerm(termKey).replace('{mean}', meanVal).replace('{conf}', obs.conf);
        }

        const totalProblems = originalLength;
        const analysisProblems = history.length;
        let resultText = "";

        if (observations.length > 0) {
            const intro = getTerm('statsIntroNew').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro + "- " + observations.map(o => o.message).join("\n- ");
        } else {
            const intro = getTerm('statsNoTrends').replace('{total}', totalProblems).replace('{analysis}', analysisProblems);
            resultText = intro;
        }

        alert(resultText);
    } catch (error) {
        console.error('OLS Regression failed:', error);
        alert('Could not perform statistical analysis.');
    }
}