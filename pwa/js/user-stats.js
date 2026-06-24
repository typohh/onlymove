// js/user-stats.js

import { state, updateUserRank, updateMomentum, updateAnswerHistory, addToAccumulatedSolvingTime, setCurrentSolvingStartTime, setReloadPending } from './state.js';
import { updateAllUI } from './ui.js';
import { tsumegoIdOf } from './sgf.js';
import { TOTAL_RANKS } from './constants.js';
import { calculateAndStoreStatsResults } from './model.js';
import { trainWeakness } from './weakness.js';
import { recordReportAck, checkSyncSignal } from './account-sync.js';
// The momentum-rank update itself (step sizes, headroom cap, loss penalty,
// quality modulation, momentum guards) lives in momentum-rank.js — one pure
// step function shared with the account-sync anchored replay.
import { momentumRankStep } from './momentum-rank.js';

export async function sendReport({ success, move = null, timeUsed = null, reviewTime = null, isRepeat = false } = {}) {
    // Linked-mode problems are rank-frozen URL-shares: updateUserStats already
    // early-returns on currentProblem.linked so they never enter local history,
    // and they must not be logged to D1 either (no rank-frozen share attempt
    // should count toward the corpus or anyone's stats). Gate on the SAME flag so
    // "report to D1" and "enter local history" stay in lockstep — one source of
    // truth for "don't record this attempt".
    if (state.currentProblem?.linked) return;
    try {
        const problemId = tsumegoIdOf(state.currentProblem.filename);
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
        // The Worker echoes its server-stamped created_at; stamping it on the
        // matching answerHistory entry lets account-sync pulls recognize this
        // device's own rows. prevCreatedAt (the user's latest report before
        // this one) is the cross-device signal: unrecognized → another device
        // wrote it → account-sync pulls. Older Workers return plain text —
        // tolerate that.
        if (res.ok) {
            const body = await res.json().catch(() => null);
            if (body && typeof body.createdAt === 'string') {
                recordReportAck(problemId, body.createdAt);
            }
            if (body) checkSyncSignal(body.prevCreatedAt);
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

// Run heavy, non-urgent work (model retraining) when the main thread is idle so
// it never competes with the result overlay's animation or the next-problem
// load. Falls back to a short timer where requestIdleCallback is unavailable
// (e.g. older Safari). The `timeout` bounds the wait so it still runs if the
// page never reports an idle period.
function scheduleIdle(fn, timeout = 2000) {
    if (typeof requestIdleCallback === 'function') {
        requestIdleCallback(() => fn(), { timeout });
    } else {
        setTimeout(fn, 100);
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
    const currentProblem = state.currentProblem;

    // Linked-mode (shared-link) attempts are never recorded — not to D1 (see
    // sendReport) and not to local history here. Keep both guards on the same
    // flag so "don't record this attempt" has one source of truth.
    if (!currentProblem || currentProblem.linked) {
        return {};
    }

    const currentSessionTime = state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0;
    addToAccumulatedSolvingTime(currentSessionTime);
    const timeTaken = state.accumulatedSolvingTime;

    const problemId = tsumegoIdOf(currentProblem.filename);
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
    const { rank, momentum } = momentumRankStep(oldRank, state.momentum, {
        correct: isCorrect,
        isRepeat: currentProblem.isRepeat || false,
        quality: currentProblem.quality,
    });

    updateUserRank(rank);
    updateMomentum(momentum);
    // The just-answered problem is moved into the previous slot, and the
    // challenge reselected if it was one, by completeCurrentProblem() in the
    // caller (game-logic) — recording the attempt and rotating the slots are
    // kept separate. No `answered` flag is needed anymore: a reselected
    // challenge is never the just-answered problem.
    updateAllUI();

    const oldRankInfo = getRankInfo(oldRank);
    const newRankInfo = getRankInfo(state.userRank);
    let rankChangeDirection = 'same';
    if (newRankInfo.rankIndex > oldRankInfo.rankIndex) rankChangeDirection = 'up';
    else if (newRankInfo.rankIndex < oldRankInfo.rankIndex) rankChangeDirection = 'down';

    // Recall + weakness models retrain off the merged history. Their training
    // loops block the main thread (~up to 2s worst case), so we schedule them on
    // browser idle time: requestIdleCallback fires only once the thread is free,
    // so the work lands DURING the result overlay without janking its fade-in or
    // the next-problem load (incorrect-path overlays sit idle while shown;
    // correct-path runs it once the load settles). The `timeout` guarantees it
    // still runs on a page that never goes idle. Nothing needs the fresh model
    // before the overlay — the incorrect-path overlay message only needs stable
    // coefficients, not this attempt folded in.
    scheduleIdle(() => { calculateAndStoreStatsResults(); });
    scheduleIdle(trainWeakness);
    // NB: the recall schedule is deliberately NOT re-uploaded per attempt — an
    // active user is suppressed by the cron anyway, so per-attempt replace-all
    // writes were pure D1 churn. It's uploaded on load, on a ~30-min heartbeat,
    // and when the tab is hidden (see main.js / notifications.uploadSchedule).

    return { rankChange: state.userRank - oldRank, rankChangeDirection, isCorrect };
}