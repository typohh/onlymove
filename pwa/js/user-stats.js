// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateTsumegoStats, updateAnswerHistory } from './state.js';
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

    const timeTaken = Date.now() - state.problemStartTime;
    const answer = {
        problemId: currentProblem.filename.replace('.sgf', ''),
        problemRating: currentProblem.rating,
        problemQuality: currentProblem.quality || 0,
        timeTaken,
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
        p.date !== undefined
    );
    const removedCount = originalLength - validHistory.length;
    if (removedCount > 0) {
        console.warn(`Removed ${removedCount} invalid answerHistory entries out of ${originalLength}`);
        updateAnswerHistory(validHistory);
    }

    const history = validHistory.slice(-500);

    console.log('Time values for stats calculation (in ms):', history.map(p => p.timeTaken));

    if (history.length < 5) { // Need enough data for regression
        alert(getTerm('statsNotEnough').replace('{count}', history.length));
        return;
    }

    // Debug: Check for invalid values
    for (const p of history) {
        if (typeof p.timeTaken !== 'number' || isNaN(p.timeTaken)) console.log('Invalid timeTaken:', p.timeTaken, p);
        if (typeof p.problemRating !== 'number' || isNaN(p.problemRating)) console.log('Invalid problemRating:', p.problemRating, p);
        if (typeof p.problemQuality !== 'number' || isNaN(p.problemQuality)) console.log('Invalid problemQuality:', p.problemQuality, p);
        if (typeof p.date !== 'number' || isNaN(p.date)) console.log('Invalid date:', p.date, p);
    }

    try {
        // Cap timeTaken at 2 std deviations
        const timeTakenValues = history.map(p => p.timeTaken);
        const timeMean = math.mean(timeTakenValues);
        const timeStdDev = math.std(timeTakenValues);
        const timeCap = timeMean + 2 * timeStdDev;
        const cappedTimeTaken = timeTakenValues.map(t => Math.min(t, timeCap));

        // Normalize date to be 0 for the earliest entry
        const dates = history.map(p => p.date);
        const minDate = Math.min(...dates);

        const y = history.map(p => p.correct ? 1 : 0);
        const X_original = history.map((p, i) => [
            1, // Intercept
            cappedTimeTaken[i] / 1000, // Time in seconds
            p.problemRating,
            p.problemQuality,
            p.date - minDate,
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

        const variableNames = [getTerm('timeSpent'), getTerm('problemDifficulty'), getTerm('problemQuality'), getTerm('date'), getTerm('shortTermRepeat'), getTerm('longTermRepeat')];
        const fullBeta = new Array(variableNames.length).fill('-');
        for (let i = 1; i < includedIndices.length; i++) { // Skip intercept
            const varIndex = includedIndices[i] - 1;
            fullBeta[varIndex] = beta[i].toFixed(6);
        }

        let resultText = getTerm('statsIntro').replace('{count}', history.length) + '\n';
        fullBeta.forEach((b, i) => {
            let formattedB = b === '-' ? '-' : (parseFloat(b) > 0 ? '+' + b : b);
            if (formattedB !== '-' && Math.abs(parseFloat(b)) > 5) {
                formattedB = '-';
            }
            resultText += `${variableNames[i]}: ${formattedB}\n`;
        });

        resultText += '\n' + getTerm('statsExplanation');

        alert(resultText);
    } catch (error) {
        console.error('OLS Regression failed:', error);
        alert('Could not perform statistical analysis.');
    }
}
