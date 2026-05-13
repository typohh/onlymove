// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateAnswerHistory, addToAccumulatedSolvingTime, setCurrentSolvingStartTime } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS } from './constants.js';
import { calculateAndStoreStatsResults } from './model.js';
import { trainWeakness } from './weakness.js';
import { maybeUploadSchedule } from './notifications.js';

export async function sendReport(success) {
    try {
        const problemFilename = state.problemHistory[0].filename;
        const problemId = problemFilename.replace(/\.[^/.]+$/, "");
        await fetch('/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: state.userId, problem: problemId, success: success, size: state.boardSize })
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

// Match (by problemId + creation date) the just-recorded answer entry and
// fold a picked-move embedding into it after the network fetch resolves.
// Idempotent — no-op if the entry already has one. Skipped if the entry has
// been pruned away by the time the fetch lands (the user may have moved on
// far enough that the trainer no longer cares).
function patchPickedEmbedding(problemId, attemptDate, embedding) {
    if (!Array.isArray(embedding) || embedding.length === 0) return;
    const history = state.answerHistory;
    for (let i = history.length - 1; i >= 0; i--) {
        const a = history[i];
        if (a.problemId === problemId && a.date === attemptDate) {
            if (a.pickedEmbedding) return;
            a.pickedEmbedding = embedding.map(v => Math.round(v * 100) / 100);
            updateAnswerHistory(history);
            return;
        }
    }
}

export function updateUserStats(isCorrect, {
    pickedCellIndex = null,
    pickedEmbedding = null,
    pickedEmbeddingPromise = null,
} = {}) {
    const history = state.problemHistory;
    const currentProblem = history[0];

    if (!currentProblem || currentProblem.linked) {
        if (currentProblem && currentProblem.linked) updateProblemHistory(history);
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
    } else if (!isCorrect) {
        successSinceFailure = 0;
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
    if (Array.isArray(currentProblem.embedding)) {
        // Quantize to 2 decimal places to keep localStorage footprint small.
        // PCA scores are unit-variance per component; 2 decimals = ±0.005
        // resolution, which is well under the noise floor for the ridge-
        // regularized weakness regression.
        answer.embedding = currentProblem.embedding.map(v => Math.round(v * 100) / 100);
    }
    if (typeof pickedCellIndex === 'number' && pickedCellIndex >= 0) {
        // Record the SGF variation index the user picked so embedding-backfill
        // can look up the cell embedding later, even if the live fetch fails
        // or hasn't returned by the time the answer is logged.
        answer.pickedCellIndex = pickedCellIndex;
    }
    if (Array.isArray(pickedEmbedding)) {
        answer.pickedEmbedding = pickedEmbedding.map(v => Math.round(v * 100) / 100);
    }

    updateAnswerHistory([...state.answerHistory, answer]);

    // Defer-patch from the in-flight fetch. Never blocks the caller — if the
    // fetch fails or the entry gets evicted later, the next app-load backfill
    // will pick this up using `pickedCellIndex`.
    if (pickedEmbeddingPromise && !answer.pickedEmbedding) {
        pickedEmbeddingPromise.then(emb => patchPickedEmbedding(problemId, answer.date, emb))
            .catch(_err => { /* swallowed; backfill handles retry */ });
    }
    setCurrentSolvingStartTime(null);

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

    setTimeout(calculateAndStoreStatsResults, 1000); // a bit crude, but animations should be completed in 1 second.
    setTimeout(trainWeakness, 1100);  // run shortly after the recall model so they don't compete for the main thread
    // After the recall model retrains, push the updated schedule so the
    // notification cron has fresh enter/exit timestamps. No-op for users
    // who haven't enabled notifications.
    setTimeout(maybeUploadSchedule, 1200);

    return { rankChange: state.userRank - oldRank, rankChangeDirection, isCorrect };
}