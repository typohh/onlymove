// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateProblemHistory, updateAnswerHistory, addToAccumulatedSolvingTime, setCurrentSolvingStartTime, setReloadPending, clampMomentum } from './state.js';
import { updateAllUI } from './ui.js';
import { TOTAL_RANKS } from './constants.js';
import { calculateAndStoreStatsResults, MODEL_INDICES } from './model.js';
import { trainWeakness } from './weakness.js';

export async function sendReport({ success, move = null, timeUsed = null, reviewTime = null, isRepeat = false } = {}) {
    try {
        const problemFilename = state.problemHistory[0].filename;
        const problemId = problemFilename.replace(/\.[^/.]+$/, "");
        const res = await fetch('/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: state.userId,
                problem: problemId,
                success,
                size: state.boardSize,
                move,
                timeUsed,
                reviewTime,
                isRepeat,
            })
        });
        const serverVersion = res.headers.get('X-App-Version');
        if (serverVersion && serverVersion !== APP_VERSION) {
            setReloadPending(true);
        }
    } catch (error) {
        console.error('Failed to send report:', error);
    }
}

// Fire-and-forget share-link analytics. `type` is 'create' (user generated a
// link) or 'open' (arrived via a shared link). For opens we pass
// document.referrer as `referrer` — the server records that plus the edge
// geo-IP country. Never blocks the UI; failures are swallowed.
export async function recordShareEvent({ type, problemId = null, referrer = null } = {}) {
    try {
        await fetch('/share-event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type,
                userId: state.userId,
                problemId,
                size: state.boardSize,
                referrer,
            }),
        });
    } catch (error) {
        console.error('Failed to record share event:', error);
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

export async function updateUserStats(isCorrect, {
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
        // Raw black-box rating (rating-space skill decomposition reads this);
        // problemRank is its normalized 0..1 rank under M (recall + weakness
        // models read this). Both captured at ingest from the served problem.
        problemRating: currentProblem.rating,
        problemRank: currentProblem.rank,
        problemQuality: currentProblem.quality || 0,
        // Raw 2PL discrimination a_p of the problem; the skill decomposition uses
        // it as the per-attempt success-curve slope (see flattenAttempts).
        problemSlope: currentProblem.slope,
        // Per-problem 3PL guessing floor c_p (bot-derived); the skill decomposition
        // uses it as the success-curve lower asymptote so a far-above-rank clear is
        // correctly down-weighted (see flattenAttempts / poolLogistic).
        problemFloor: currentProblem.floor,
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
    // `currentProblem.quality` is now the RAW 2PL Fisher-information selection
    // weight a_p² (median ~6, up to ~20), NOT the normalized [0,1] quality this
    // momentum loop was designed for. Feeding it raw makes scalingFactor ≫ 1,
    // which turns the failure recurrence momentum ← momentum − sf·(2·momentum−1)/3
    // into an unstable, sign-flipping oscillator (stable only for sf < 3): momentum
    // diverges and goes negative, after which a win lowers the rank and a loss
    // raises it (the wild 8d↔50k swings). Bound it back to [0,1] so the loop is the
    // designed contraction. (NB: a_p² saturates the clamp for almost all problems,
    // so this no longer modulates the step much — a proper [0,1] confidence mapping
    // of a_p could restore that, but stability is the priority here.)
    const qualityConfidence = Math.min(1, Math.max(0, currentProblem.quality || 0.5));
    const scalingFactor = qualityConfidence * (currentProblem.isRepeat ? 1 / 5 : 1);

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

    // Guardrail: keep momentum positive and bounded every step so a future
    // mis-scaled input can never let the recurrence run away again.
    momentum = clampMomentum(momentum);

    updateUserRank(rank);
    updateMomentum(momentum);
    updateProblemHistory(history);
    updateAllUI();

    const oldRankInfo = getRankInfo(oldRank);
    const newRankInfo = getRankInfo(state.userRank);
    let rankChangeDirection = 'same';
    if (newRankInfo.rankIndex > oldRankInfo.rankIndex) rankChangeDirection = 'up';
    else if (newRankInfo.rankIndex < oldRankInfo.rankIndex) rankChangeDirection = 'down';

    // Time-efficiency delta — the time analogue of rankChange. The recall
    // model's cold-solve logit already carries a per-user TIME_SOLVING
    // coefficient with difficulty/review/size partialled out; that coefficient
    // IS the model's estimate of "does this user convert time into correct
    // answers". So we read it, fold in this attempt by retraining, and read it
    // again — the shift is the efficiency gain/loss for the attempt. The
    // retrain runs INLINE here (it used to be a deferred setTimeout) so the
    // before/after bracket exactly this attempt and the value is ready for the
    // result overlay. Raw param (not the SE-scaled statsResults figure) so the
    // unit is stable across retrains — the per-minute rescale would otherwise
    // leak into the delta.
    const M = MODEL_INDICES;
    const numParams = Object.keys(M).length;
    const modelReady = (p) => Array.isArray(p) && p.length === numParams;
    const effBefore = modelReady(state.modelParams) ? state.modelParams[M.TIME_SOLVING] : null;
    await calculateAndStoreStatsResults();
    const effAfter = modelReady(state.modelParams) ? state.modelParams[M.TIME_SOLVING] : null;
    const timeEfficiencyChange = (effBefore !== null && effAfter !== null) ? (effAfter - effBefore) : 0;

    setTimeout(trainWeakness, 100);  // weakness model trains off the merged history; deferred so it never blocks the overlay
    // NB: the recall schedule is deliberately NOT re-uploaded per attempt — an
    // active user is suppressed by the cron anyway, so per-attempt replace-all
    // writes were pure D1 churn. It's uploaded on load, on a ~30-min heartbeat,
    // and when the tab is hidden (see main.js / notifications.uploadSchedule).

    return { rankChange: state.userRank - oldRank, rankChangeDirection, isCorrect, timeEfficiencyChange };
}