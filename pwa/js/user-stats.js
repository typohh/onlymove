// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateTsumegoStats } from './state.js';
import { updateAllUI } from './ui.js';

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

export function updateUserStats(isCorrect) {
    if (state.openedFromLink) return;

    let rank = state.userRank;
    let momentum = state.momentum;
    const history = state.problemHistory;
    const currentProblem = history[0];

    if (!currentProblem) return;

    const oldStatus = currentProblem.status;
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

    if (isCorrect) {
        if (oldStatus === 'failure') {
            currentProblem.status = 'in-between';
        } else if (oldStatus === 'in-between') {
            currentProblem.status = 'success';
        }

        if (oldStatus === 'success') {
            const maxIncrease = (1.0 - rank) / 2;
            const intendedIncrease = momentum / 600;
            if(intendedIncrease > maxIncrease) {
                momentum = Math.max( 0.5 , momentum * 0.8 );
            } else {
                momentum *= 1.1;
            }
            rank += Math.min(intendedIncrease, maxIncrease);
        }
    } else { // Incorrect
        momentum = (momentum + 1) / 3;
        rank = Math.max(0, rank - 4 * (momentum / 600));
        currentProblem.status = 'failure';
    }

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemHistory(history);
    updateTsumegoStats(stats);
    updateAllUI();
}
