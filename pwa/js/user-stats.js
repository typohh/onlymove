// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateAnswerHistory, addToAccumulatedSolvingTime } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS } from './constants.js';
import { calculateAndStoreStatsResults } from './model.js';

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

    const prevs = state.answerHistory.filter(ans => ans.problemId === problemId);
    const accumulatedReview = prevs.reduce((sum, ans) => sum + (ans.reviewTime || 0), 0);

    const answer = {
        problemId,
        problemRating: currentProblem.rating,
        problemQuality: currentProblem.quality || 0,
        timeTaken,
        reviewTime: 0,
        accumulatedReview,
        correct: isCorrect,
        date: Date.now(),
        successSinceFailure,
        isRepeat: currentProblem.isRepeat || false,
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