// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemFrequencies } from './state.js';
import { updateAllUI } from './ui.js';

// Function to send report to Cloudflare Worker
export async function sendReport(problemFilename, success) {
    try {
        if (state.problemFrequencies.hasOwnProperty(problemFilename)) {
            return;
        }
        
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
    let frequencies = { ...state.problemFrequencies };

    if (isCorrect) {
        const currentFreq = frequencies[state.currentProblemFilename] || 1;
        frequencies[state.currentProblemFilename] = currentFreq / 5;
        const maxIncrease = (1.0 - rank) / 2;
        const intendedIncrease = momentum / 600;
        if(intendedIncrease > maxIncrease) {
            momentum = Math.max( 0.5 , momentum * 0.8 );
        } else {
            momentum *= 1.1;
        }
        rank += Math.min(intendedIncrease, maxIncrease);
    } else {
        momentum = (momentum + 1) / 3;
        frequencies[state.currentProblemFilename] = 25;
        rank = Math.max(0, rank - 4 * (momentum / 600));
    }

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemFrequencies(frequencies);
    updateAllUI();
}
