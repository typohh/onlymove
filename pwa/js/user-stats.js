// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateTsumegoStats, updateAnswerHistory, addToAccumulatedSolvingTime } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS } from './constants.js';
import { getTerm } from './utils.js';

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

function getRankInfo(rating) {
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
        if (oldStatus === 'failure') {
            currentProblem.status = 'in-between';
        } else if (oldStatus === 'in-between') {
            currentProblem.status = 'success';
        }
    } else {
        currentProblem.status = 'failure';
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
            : oldStatus === 'failure'
                ? 'short-term-repeat'
                : oldStatus === 'in-between'
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

    if (oldStatus === 'failure') {
        stats.totalOnFail++;
        if (!isCorrect) {
            stats.failOnFail++;
        }
    } else if (oldStatus === 'in-between') {
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
        if( oldStatus !== 'failure' && oldStatus !== 'in-between' ) {
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

        const Xt = math.transpose(X);
        const XtX = math.multiply(Xt, X);
        const XtX_inv = math.inv(XtX);
        const Xty = math.multiply(Xt, y);
        const beta = math.multiply(XtX_inv, Xty);

        const variableNames = [getTerm('timeSpent'), getTerm('reviewTimeShortTerm'), getTerm('reviewTimeLongTerm'), getTerm('problemDifficulty'), getTerm('problemQuality'), getTerm('date'), getTerm('shortTermRepeat'), getTerm('longTermRepeat')];
        const fullBeta = new Array(variableNames.length).fill('-');
        for (let i = 1; i < includedIndices.length; i++) { // Skip intercept
            const varIndex = includedIndices[i] - 1;
            let coeff = beta[i];
            if (varIndex === 0 && time_per_percentile_min > 0) { // timeSpent
                coeff /= time_per_percentile_min;
            }
            if (varIndex === 1 && time_per_percentile_min_r_short > 0) { // reviewTimeShortTerm
                coeff /= time_per_percentile_min_r_short;
            }
            if (varIndex === 2 && time_per_percentile_min_r_long > 0) { // reviewTimeLongTerm
                coeff /= time_per_percentile_min_r_long;
            }
            if (varIndex === 3) { // problemDifficulty
                coeff /= 29;
            }
            if (varIndex === 4) { // problemQuality
                coeff = -coeff;
            }
            coeff *= 100;  // Convert to %
            fullBeta[varIndex] = coeff.toFixed(1);
        }

        let resultText = getTerm('statsIntro').replace('{count}', history.length) + '\n';
        fullBeta.forEach((b, i) => {
            if (i === 6 || i === 7) return; // Skip shortTermRepeat and longTermRepeat
            let formattedB = b === '-' ? '-' : (parseFloat(b) > 0 ? '+' + b : b);
            let unit = '';
            if (i === 0) unit = getTerm('statsPerMinuteSolving');
            else if (i === 1) unit = getTerm('statsPerMinuteReviewingShortTerm');
            else if (i === 2) unit = getTerm('statsPerMinuteReviewingLongTerm');
            else if (i === 3) unit = getTerm('statsPerProblemRank');
            else if (i === 4) unit = getTerm('statsIfIntuitiveProblem');
            else if (i === 5) unit = getTerm('statsPerProblemSolved');
            else unit = getTerm('statsDefault');
            resultText += `${formattedB}${unit}
`;
        });

        

        alert(resultText);
    } catch (error) {
        console.error('OLS Regression failed:', error);
        alert('Could not perform statistical analysis.');
    }
}
