// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateTsumegoStats } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS } from './constants.js';

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

    if (isCorrect && oldStatus !== 'failure' && oldStatus !== 'in-between') {
        const maxIncrease = (1.0 - rank) / 9;
        const intendedIncrease = momentum / 600;
        if(intendedIncrease > maxIncrease) {
            momentum *= maxIncrease / intendedIncrease;
        } else {
            momentum *= 1.1;
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
