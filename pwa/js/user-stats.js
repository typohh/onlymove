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
    const history = state.answerHistory.slice(-500);

    if (history.length < 10) { // Need enough data for regression
        alert(getTerm('noHistory'));
        return;
    }

    try {
        const y = history.map(p => p.correct ? 1 : 0);
        const X_original = history.map(p => [
            1, // Intercept
            p.timeTaken / 1000, // Time in seconds
            p.problemRating,
            p.problemQuality,
            p.date,
            p.problemType === 'novel' ? 1 : 0,
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
        // If long-term repeat has 0 stddev, exclude short-term repeat too
        const includedIndices = [0]; // intercept
        for (let j = 1; j < stdDevs.length + 1; j++) {
            if (stdDevs[j - 1] > 0 && !(j === 6 && stdDevs[6] === 0)) {
                includedIndices.push(j);
            }
        }

        // Create X with only included columns
        const X = X_original.map(row => includedIndices.map(idx => row[idx]));

        // Add noise to prevent perfect collinearity (only for included variables with stddev > 0)
        for (let j = 1; j < X[0].length; j++) { // Start from 1 to skip intercept
            const col = X.map(row => row[j]);
            const stdDev = math.std(col);
            const varIndex = includedIndices[j];
            let effectiveStdDev = stdDev;
            if (varIndex >= 5 && varIndex <= 7) { // For category variables (novel, short-term, long-term), assume stddev >= 1
                effectiveStdDev = Math.max(stdDev, 1);
            }

            for (let i = 0; i < X.length; i++) {
                const noise = effectiveStdDev * 0.0001 * (Math.random() * 2 - 1); // Noise between -0.01% and +0.01% of stdDev
                X[i][j] += noise;
            }
        }

        const Xt = math.transpose(X);
        const XtX = math.multiply(Xt, X);
        const XtX_inv = math.inv(XtX);
        const Xty = math.multiply(Xt, y);
        const beta = math.multiply(XtX_inv, Xty);

        const variableNames = ['Time Spent', 'Problem Difficulty', 'Problem Quality', 'Date', 'Novel', 'Short-term Repeat', 'Long-term Repeat'];
        const fullBeta = new Array(variableNames.length).fill('-');
        for (let i = 1; i < includedIndices.length; i++) { // Skip intercept
            const varIndex = includedIndices[i] - 1;
            fullBeta[varIndex] = beta[i].toFixed(3);
        }

        // If long-term repeat has stddev 0, set short-term to negative of novel
        if (stdDevs[6] === 0) {
            if (fullBeta[4] !== '-') {
                fullBeta[5] = (-parseFloat(fullBeta[4])).toFixed(3);
            } else {
                fullBeta[5] = '-';
            }
        }

        let resultText = `Factors affecting your result from past ${history.length} problems:\n`;
        fullBeta.forEach((b, i) => {
            let formattedB = b === '-' ? '-' : (parseFloat(b) > 0 ? '+' + b : b);
            if (formattedB !== '-' && Math.abs(parseFloat(b)) > 5) {
                formattedB = '-';
            }
            resultText += `${variableNames[i]}: ${formattedB}\n`;
        });

        resultText += '\nPositive beta coefficients suggest a higher chance of a correct answer, negative beta coefficients suggest a lower chance.';

        alert(resultText);
    } catch (error) {
        console.error('OLS Regression failed:', error);
        alert('Could not perform statistical analysis.');
    }
}
