// js/user-stats.js

import { state, updateUserRank, updateServeRank, updateAnswerHistory, getMergedAnswerHistory, addToAccumulatedSolvingTime, setCurrentSolvingStartTime, setReloadPending } from './state.js';
import { updateAllUI } from './ui.js';
import { tsumegoIdOf } from './sgf.js';
import { quantizeEmbedding } from './math-utils.js';
import { TOTAL_RANKS, RECALL_CANDIDATE_HISTORY_LENGTH } from './constants.js';
import { calculateAndStoreStatsResults, recallImpactWithRefine, MIN_RECALL_INTERVAL_MS } from './model.js';
import { trainWeakness } from './weakness.js';
import { recordReportAck, checkSyncSignal } from './account-sync.js';
// EGD-leaderboard GoR refresh piggybacks on /report: the response advertises stale
// PINs, we fetch their GoR from EGD (the Worker can't) and send it on the next report.
import { takePendingGorUpdates, requestGorRefresh } from './gor-refresh.js';
// The live rank is a linear-information window 4PL MLE over the recent first-time
// history; the estimator lives in rank-decomposition.js alongside the per-attempt
// slope/floor conversion it shares with the skill chart.
import { estimateWindowRank } from './rank-decomposition.js';
import { postJson } from './net.js';

// --- Cross-tab duplicate guard --------------------------------------------
// Two tabs of the same origin boot the same standing challenge, and each
// records the solve as a NOVEL attempt. The tab lock (tab-lock.js) prevents
// concurrent tabs where Web Locks exist; this is the record-time fallback.
// At answer time, re-read the PERSISTED working set — another tab's write is
// visible there but not in this tab's stale in-memory copy — and classify a
// supposedly-novel attempt with the standard novelty window (the same
// RECALL_CANDIDATE_HISTORY_LENGTH check that made the served problem novel to
// begin with):
//   - prior attempt within the minimum recall interval (MIN_RECALL_INTERVAL_MS,
//     the recall model's ~18h stability floor): a same-session duplicate →
//     'drop' (recorded nowhere, reported nowhere);
//   - older prior attempt (a long-stale tab): a legitimate repeat →
//     'repeat', so it never counts as a second first-time attempt.
// Classified once per answer and memoized on the problem object so sendReport
// and updateUserStats agree (updateUserStats appends to the persisted history,
// so a second unmemoized read would see our own row).
function classifyCrossTabAttempt(currentProblem) {
    if (currentProblem.crossTabClass !== undefined) return currentProblem.crossTabClass;
    let cls = 'record';
    if (!currentProblem.isRepeat && !currentProblem.linked) {
        try {
            const parsed = JSON.parse(localStorage.getItem('answerHistory') || 'null');
            if (Array.isArray(parsed)) {
                const problemId = tsumegoIdOf(currentProblem.filename);
                const recent = parsed.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);
                let last = null;
                for (let i = recent.length - 1; i >= 0; i--) {
                    if (recent[i] && recent[i].problemId === problemId) { last = recent[i]; break; }
                }
                if (last) {
                    cls = (Date.now() - (last.date || 0) < MIN_RECALL_INTERVAL_MS) ? 'drop' : 'repeat';
                }
            }
        } catch { /* unreadable persisted history — record as-is */ }
    }
    currentProblem.crossTabClass = cls;
    return cls;
}

export async function sendReport({ success, move = null, timeUsed = null, reviewTime = null, isRepeat = false } = {}) {
    // Linked-mode problems are rank-frozen URL-shares: updateUserStats already
    // early-returns on currentProblem.linked so they never enter local history,
    // and they must not be logged to D1 either (no rank-frozen share attempt
    // should count toward the corpus or anyone's stats). Gate on the SAME flag so
    // "report to D1" and "enter local history" stay in lockstep — one source of
    // truth for "don't record this attempt".
    if (state.currentProblem?.linked) return;
    // Cross-tab duplicate: 'drop' reports nowhere (the other tab's report
    // already logged this problem minutes ago); 'repeat' reports honestly.
    const crossTab = classifyCrossTabAttempt(state.currentProblem);
    if (crossTab === 'drop') return;
    if (crossTab === 'repeat') isRepeat = true;
    try {
        const problemId = tsumegoIdOf(state.currentProblem.filename);
        // Attach any GoR values fetched from EGD after a previous report advertised
        // stale PINs (leaderboard refresh; the Worker can't fetch EGD itself).
        const gorUpdates = takePendingGorUpdates();
        const res = await postJson('/report', {
            userId: state.userId,
            problem: problemId,
            success,
            size: state.boardSize,
            move,
            timeUsed,
            reviewTime,
            isRepeat,
            ...(gorUpdates ? { gorUpdates } : {}),
        });
        const serverVersion = res.headers.get('X-App-Version');
        if (serverVersion && serverVersion !== APP_VERSION) {
            setReloadPending(true);
        }
        // The Worker echoes its server-stamped created_at + row id; stamping
        // them on the matching answerHistory entry lets account-sync pulls
        // recognize this device's own rows (serverCreatedAt = dedup key,
        // serverId = id-cursor/digest key). userSeq is the per-profile attempt
        // count: if it jumps past what this device expects, another device
        // wrote → pull (the seq sync signal).
        if (res.ok) {
            const body = await res.json().catch(() => null);
            if (body && typeof body.createdAt === 'string') {
                recordReportAck(problemId, body.createdAt, body.serverId);
            }
            if (body) checkSyncSignal(body.userSeq);
            // Server advertised registered PINs whose GoR is stale this week — fetch
            // them from EGD now; they ride the next report back (see gor-refresh.js).
            if (body && Array.isArray(body.refreshPins)) requestGorRefresh(body.refreshPins);
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
        await postJson('/share-event', {
            type,
            userId: state.userId,
            problemId,
            size: state.boardSize,
            referrer,
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
            a.pickedEmbedding = quantizeEmbedding(embedding);
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

    // Cross-tab duplicate (see classifyCrossTabAttempt): 'drop' records
    // nothing — another tab recorded this problem within the minimum recall
    // interval; 'repeat' records honestly as a repeat attempt.
    const crossTab = classifyCrossTabAttempt(currentProblem);
    if (crossTab === 'drop') {
        setCurrentSolvingStartTime(null);
        return {};
    }
    const isRepeatAttempt = currentProblem.isRepeat || crossTab === 'repeat';

    const currentSessionTime = state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0;
    addToAccumulatedSolvingTime(currentSessionTime);
    const timeTaken = state.accumulatedSolvingTime;

    const problemId = tsumegoIdOf(currentProblem.filename);
    let successSinceFailure;
    // The prior attempt on this problem — kept for the recall-impact overlay below
    // (its date gives the elapsed gap, its streak the r going into this attempt).
    let recallLastAnswer = null;
    if (isRepeatAttempt) {
        const previousAnswers = state.answerHistory.filter(ans => ans.problemId === problemId);
        recallLastAnswer = previousAnswers.length ? previousAnswers[previousAnswers.length - 1] : null;
        if (!isCorrect) {
            successSinceFailure = 0;
        } else {
            successSinceFailure = (recallLastAnswer && recallLastAnswer.successSinceFailure !== undefined) ? recallLastAnswer.successSinceFailure + 1 : 1;
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
        isRepeat: isRepeatAttempt,
    };
    if (Array.isArray(currentProblem.embedding)) {
        // Quantize to 2 decimal places to keep localStorage footprint small.
        // PCA scores are unit-variance per component; 2 decimals = ±0.005
        // resolution, which is well under the noise floor for the ridge-
        // regularized weakness regression.
        answer.embedding = quantizeEmbedding(currentProblem.embedding);
    }
    if (typeof pickedCellIndex === 'number' && pickedCellIndex >= 0) {
        // Record the SGF variation index the user picked so backfill.js
        // can look up the cell embedding later, even if the live fetch fails
        // or hasn't returned by the time the answer is logged.
        answer.pickedCellIndex = pickedCellIndex;
    }
    if (Array.isArray(pickedEmbedding)) {
        answer.pickedEmbedding = quantizeEmbedding(pickedEmbedding);
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
    // Re-fit the linear-information window MLE over the merged first-time history
    // (the current attempt was appended above). oldRank·TOTAL_RANKS warm-starts the
    // fit AND is the R_before for the overlay delta below: because the sync path
    // updates userRank SILENTLY when foreign attempts arrive (see account-sync's
    // retrainAfterMerge), oldRank already reflects them, so the reported rank
    // change is the effect of THIS tsumego alone — never the synced batch.
    const est = estimateWindowRank(getMergedAnswerHistory(), oldRank * TOTAL_RANKS);
    if (est) {
        updateUserRank(est.rankIndex / TOTAL_RANKS);
        updateServeRank(est.serveRankIndex / TOTAL_RANKS);
    }
    // The just-answered problem is moved into the previous slot, and the
    // challenge reselected if it was one, by completeCurrentProblem() in the
    // caller (game-logic) — recording the attempt and rotating the slots are
    // kept separate.
    updateAllUI();

    const oldRankInfo = getRankInfo(oldRank);
    const newRankInfo = getRankInfo(state.userRank);
    let rankChangeDirection = 'same';
    if (newRankInfo.rankIndex > oldRankInfo.rankIndex) rankChangeDirection = 'up';
    else if (newRankInfo.rankIndex < oldRankInfo.rankIndex) rankChangeDirection = 'down';

    // For a recall (repeat) attempt the rank barely moves (rank is fit on
    // first-time attempts only). Instead the overlay shows how recording this
    // answer ACTUALLY changed the recall model: we run the real model retrain now
    // — the same fit the idle path runs — and measure the time until first repeat
    // of a generic problem before vs after teaching it this attempt. No formula,
    // no shortcut, no cap: the delta is the genuine before/after the model
    // produces (a correct recall generally pushes it later, a miss earlier, but
    // since it's the real retrain over a finite window the magnitude — and rarely
    // the sign — is whatever the model actually learns). Worth the synchronous
    // retrain block (≤~2s, usually far less warm-started) because the overlay
    // cannot report the true value without first training on the answer.
    let recallImpact = null;
    if (isRepeatAttempt) {
        // The genuine before/after the recall model produces from teaching it this
        // answer (generic average-difficulty problem), THEN — if the displayed delay
        // moved the wrong way for this answer — a few extra Adam steps one at a time
        // until it agrees with the answer's sign (recallImpactWithRefine, shared with
        // the offline contradiction sim so both run the identical path).
        const { beforeHours, afterHours } = await recallImpactWithRefine(isCorrect);
        if (beforeHours !== null && afterHours !== null) {
            recallImpact = { beforeHours, afterHours };
        }
    } else {
        // First-time attempt: nothing to display. Keep the retrain off the
        // critical path — deferred to browser idle (requestIdleCallback fires once
        // the thread is free, with a timeout fallback) so it never janks the
        // overlay fade-in or the next-problem load.
        scheduleIdle(() => { calculateAndStoreStatsResults(); });
    }
    scheduleIdle(trainWeakness);
    // NB: the recall schedule is deliberately NOT re-uploaded per attempt — an
    // active user is suppressed by the cron anyway, so per-attempt replace-all
    // writes would be pure D1 churn. It's uploaded on load, on a ~30-min heartbeat,
    // and when the tab is hidden (see main.js / notifications.uploadSchedule).

    return { rankChange: state.userRank - oldRank, rankChangeDirection, isCorrect, recallImpact };
}