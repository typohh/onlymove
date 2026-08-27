// js/attempt-recorder.js
//
// What happens when an attempt is finished: update the local history and rank
// (updateUserStats), then report it to the Worker (sendReport, plus the share-event
// sibling). Named `user-stats.js` until 2026-08-03, which read as a member of the
// stats-* family — the detailed-stats modal — when it is the write path for an
// answer, not a display of one.

import { state, updateAnswerHistory, getMergedAnswerHistory, addToAccumulatedSolvingTime, setCurrentSolvingStartTime, setReloadPending } from './state.js';
import { updateAllUI } from './ui.js';
import { tsumegoIdOf } from './sgf.js';
import { quantizeEmbedding } from './math-utils.js';
import { motifDataIsCurrent, storedVersionMatches, RATINGS_VERSION } from './data-version.js';
import { RECALL_CANDIDATE_HISTORY_LENGTH, RATINGS_VERSION_KEY, RANK_CURVE_VERSION_KEY } from './constants.js';
import { rankBand } from '../shared/rank-math.js';
import { calculateAndStoreStatsResults, recallImpact, MIN_RECALL_INTERVAL_MS } from './recall-model.js';
import { syncRecallNotification } from './notifications.js';
import { recordReportAck, rebaseUnackedEntryDate, checkSyncSignal } from './account-sync.js';
// EGD-leaderboard GoR refresh piggybacks on /report: the response advertises stale
// PINs, we fetch their GoR from EGD (the Worker can't) and send it on the next report.
import { takePendingGorUpdates, requestGorRefresh } from './gor-refresh.js';
// The live rank is a linear-information window 4PL MLE over the recent first-time
// history; the estimator lives in rank-decomposition.js alongside the per-attempt
// slope/floor conversion it shares with the skill chart.
import { reanchorRank, ratingRankCurveVersion } from './rank-decomposition.js';
import { postJson } from './worker-post.js';

// --- Cross-tab duplicate guard --------------------------------------------
// Two tabs of the same origin boot the same standing challenge, and each
// records the solve as a NOVEL attempt. This record-time guard is the only
// defense — concurrent tabs are allowed.
// At answer time, re-read the PERSISTED working set — another tab's write is
// visible there but not in this tab's stale in-memory copy — and classify a
// supposedly-novel attempt with the standard novelty window (the same
// RECALL_CANDIDATE_HISTORY_LENGTH check that made the served problem novel to
// begin with):
//   - prior attempt within the minimum recall interval (MIN_RECALL_INTERVAL_MS,
//     the recall model's ~6h stability floor): a same-session duplicate →
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

export async function sendReport({ success, move = null, timeUsed = null, reviewTime = null, isRepeat = false, at = null } = {}) {
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
        // The attempt's event time — minted in the click handler and shared
        // verbatim with the answerHistory entry's `date` (updateUserStats), so
        // the inserted row's created_at equals the local entry EXACTLY. The
        // server inserts it verbatim or refuses with a bound error (below);
        // it never rewrites it.
        const sentAt = Number.isFinite(at) ? at : Date.now();
        const gorUpdates = takePendingGorUpdates();
        const payload = {
            userId: state.userId,
            problem: problemId,
            success,
            size: state.boardSize,
            move,
            timeUsed,
            reviewTime,
            isRepeat,
            createdAt: new Date(sentAt).toISOString(),
            // GoR values fetched from EGD after a previous report advertised
            // stale PINs (leaderboard refresh; the Worker can't fetch EGD itself).
            ...(gorUpdates ? { gorUpdates } : {}),
        };
        let res = await postJson('/report', payload);
        const serverVersion = res.headers.get('X-App-Version');
        if (serverVersion && serverVersion !== APP_VERSION) {
            setReloadPending(true);
        }
        // Timestamp refused (this device's clock is out of bounds): self-correct
        // ONCE to the server's clock and resend. The entry's date is rebased AND
        // PERSISTED before the resend (rebaseUnackedEntryDate) — from the moment
        // the server could hold the corrected row, the local entry already
        // carries the identical date, so an abrupt kill anywhere leaves
        // entry === row and the next pull's exact claim heals it. If the rebase
        // can't find the entry or the resend fails, the entry stays un-acked and
        // the outbox's backfill route (with its own per-row correction) picks it
        // up later — a permanent refusal never sits retrying in this hot path.
        let ackAt = sentAt;
        if (!res.ok && res.status === 400) {
            const errBody = await res.json().catch(() => null);
            if (errBody && (errBody.error === 'future_timestamp' || errBody.error === 'stale_timestamp') &&
                typeof errBody.serverNow === 'string') {
                const correctedAt = Date.parse(errBody.serverNow);
                if (Number.isFinite(correctedAt) &&
                    await rebaseUnackedEntryDate(problemId, sentAt, correctedAt)) {
                    ackAt = correctedAt;
                    res = await postJson('/report', { ...payload, createdAt: errBody.serverNow });
                }
            }
        }
        // On ack, stamp the row id on the matching answerHistory entry (serverId
        // is identity, and its presence = acked; there is no createdAt echo — the
        // accepted value is the one this client sent). userSeq is the per-profile
        // attempt count: if it jumps past what this device expects, another
        // device wrote → pull (the seq sync signal).
        if (res.ok) {
            const body = await res.json().catch(() => null);
            if (body) {
                recordReportAck(problemId, ackAt, body.serverId);
                checkSyncSignal(body.userSeq, body.digest);
            }
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

// Does this answer earn a "rank up/down to X" line? The shown rank before this
// attempt against the shown rank after it, in bands. Bands come from the shared
// rankBand, so a line fires exactly when the badge's label changes
// (getDescriptiveRank uses the same band), and only when the crossing AGREES with the
// answer — "rank down" under a correct answer reads as a bug.
//
// Re-crossing an edge back and forth, and being told each time, is CORRECT: the shown
// rank really is moving across the boundary and the badge really is changing with it.
// What this must not do is report a crossing the answer did not cause, which is why
// both sides come from the same estimator over the same corpus floor — for a user with
// a full window a single answer moves the shown rank by well under a tenth of a rank,
// so a whole-band jump means an input changed, not the player.
function rankBandCrossing(oldRank, isCorrect) {
    const oldBand = rankBand(oldRank);
    const newBand = rankBand(state.displayRank);
    if (newBand === oldBand) return 'same';
    const crossing = newBand > oldBand ? 'up' : 'down';
    return crossing === (isCorrect ? 'up' : 'down') ? crossing : 'same';
}

// Match (by problemId + creation date) the just-recorded answer entry and
// fold the wrong-attempt embedding data into it after the network fetch
// resolves: the picked-move vector and the availability-null moments
// (computeAltStats), both from the same per-cell block. Idempotent — each
// field is written only if absent. Skipped if the entry has been pruned away
// by the time the fetch lands (the user may have moved on far enough that
// the trainer no longer cares).
function patchWrongAttemptStats(problemId, attemptDate, stats) {
    if (!stats) return;
    const { picked, alt } = stats;
    const history = state.answerHistory;
    for (let i = history.length - 1; i >= 0; i--) {
        const a = history[i];
        if (a.problemId === problemId && a.date === attemptDate) {
            // Re-asked here rather than trusted from the call site: this lands
            // after an await, and the wipe that advances the device's fit could
            // have run in between.
            if (!motifDataIsCurrent()) return;
            let touched = false;
            if (!a.pickedEmbedding && Array.isArray(picked) && picked.length > 0) {
                a.pickedEmbedding = quantizeEmbedding(picked);
                touched = true;
            }
            if (a.altMean === undefined && alt) {
                a.altMean = alt.altMean;
                a.altSd = alt.altSd;
                a.altAbove = alt.altAbove;
                a.altBelow = alt.altBelow;
                touched = true;
            }
            if (touched) updateAnswerHistory(history);
            return;
        }
    }
}

export async function updateUserStats(isCorrect, {
    pickedCellIndex = null,
    correctCellIndex = null,
    pickedEmbedding = null,
    wrongStatsPromise = null,
    // The event time minted in the click handler — the SAME ms sendReport put
    // in the report's createdAt, so entry.date === the server row exactly.
    at = null,
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

    // Every field below that this build DERIVED from shipped data is written only
    // when the generation it came from is the device's — the one rule, applied to
    // each generation separately (storedVersionMatches). An instance a release
    // behind is still serving problems out of its own shards and curve: what it
    // would write is the previous fit's, and it cannot tell, but it can see its
    // version and the device's disagree. Writing nothing is what the repopulate
    // exists to complete, from set-wide data, at current values.
    const ratingsCurrent =
        storedVersionMatches(RATINGS_VERSION_KEY, RATINGS_VERSION) &&
        storedVersionMatches(RANK_CURVE_VERSION_KEY, ratingRankCurveVersion(state.boardSize));
    const answer = {
        problemId,
        timeTaken,
        reviewTime: 0,
        accumulatedReview,
        correct: isCorrect,
        date: Number.isFinite(at) ? at : Date.now(),
        successSinceFailure,
        isRepeat: isRepeatAttempt,
    };
    if (ratingsCurrent) {
        // Raw black-box rating (the rating-space skill decomposition reads this);
        // problemRank is its normalized rank under M (recall + the motif readings
        // read this). Both captured at ingest from the served problem.
        answer.problemRating = currentProblem.rating;
        answer.problemRank = currentProblem.rank;
        answer.problemQuality = currentProblem.quality || 0;
        // Raw 2PL discrimination a_p; the skill decomposition uses it as the
        // per-attempt success-curve slope (see flattenAttempts).
        answer.problemSlope = currentProblem.slope;
        // Per-problem 3PL guessing floor c_p; the skill decomposition uses it as
        // the success-curve lower asymptote so a far-above-rank clear is correctly
        // down-weighted (see flattenAttempts / poolLogistic).
        answer.problemFloor = currentProblem.floor;
    }
    // Vectors are recorded only when THIS BUILD's motif data is the device's
    // current fit. An instance that is a release behind is still serving problems
    // out of its own shards, and every vector it can produce belongs to the
    // previous fit — it cannot know it is behind, but it can see that its version
    // and the device's disagree, which is exactly this test. Recording no vector
    // costs nothing: the row keeps `pickedCellIndex` below, and the fill pass
    // gives it the current vector from the shipped sidecar.
    const fitIsCurrent = motifDataIsCurrent();
    if (fitIsCurrent && Array.isArray(currentProblem.embedding)) {
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
    if (typeof correctCellIndex === 'number' && correctCellIndex >= 0) {
        // The answer node's variation index, same index space and same
        // fit-independence as pickedCellIndex: a refill of the availability
        // null needs to exclude the correct move from the alternatives, and
        // this pins WHICH row that was without re-deriving it from the SGF.
        answer.correctCellIndex = correctCellIndex;
    }
    // The picked vector comes from the same shard the problem did, so it is
    // current exactly when the correct-move vector is — and the two must come from
    // ONE fit in any case, the tendency reading being a comparison between them
    // inside a single row.
    if (fitIsCurrent && Array.isArray(pickedEmbedding)) {
        answer.pickedEmbedding = quantizeEmbedding(pickedEmbedding);
    }

    updateAnswerHistory([...state.answerHistory, answer]);

    // Defer-patch from the in-flight fetch. Never blocks the caller — if the
    // fetch fails or the entry gets evicted later, the next app-load backfill
    // will pick this up using `pickedCellIndex`/`correctCellIndex`.
    if (fitIsCurrent && wrongStatsPromise) {
        wrongStatsPromise.then(stats => patchWrongAttemptStats(problemId, answer.date, stats))
            .catch(_err => { /* swallowed; backfill handles retry */ });
    }
    setCurrentSolvingStartTime(null);

    const oldRank = state.displayRank;
    // Re-fit the window posterior over the merged first-time history (the current
    // attempt was appended above). oldRank is the R_before for the overlay delta
    // below, and it is the DISPLAYED rank because that is the number whose change
    // the overlay reports. Because the sync path updates the ranks SILENTLY when
    // foreign attempts arrive (see account-sync's retrainAfterMerge), oldRank
    // already reflects them, so the reported change is the effect of THIS tsumego
    // alone — never the synced batch.
    reanchorRank(getMergedAnswerHistory());
    // The just-answered problem is moved into the previous slot, and the
    // challenge reselected if it was one, by completeCurrentProblem() in the
    // caller (game-logic) — recording the attempt and rotating the slots are
    // kept separate.
    updateAllUI();
    // Count a lingering recall reminder down by whatever this attempt cleared —
    // and take it down once nothing is waiting. Runs after the history append
    // above, so the just-answered problem has already left the due set.
    syncRecallNotification();

    const rankChangeDirection = rankBandCrossing(oldRank, isCorrect);

    // For a recall (repeat) attempt the rank barely moves (rank is fit on
    // first-time attempts only). Instead the overlay shows how recording this
    // answer ACTUALLY changed the recall model: the real model retrain — the same
    // fit the idle path runs — measures the time until first repeat of a generic
    // problem before vs after teaching it this attempt. No formula, no shortcut,
    // no cap: the delta is the genuine before/after the model produces (a correct
    // recall generally pushes it later, a miss earlier, but since it's the real
    // retrain over a finite window the magnitude — and rarely the sign — is
    // whatever the model actually learns). The retrain is deferred to idle so the
    // result overlay's fade starts immediately, and the overlay reserves the
    // impact line and fills it in when this promise resolves
    // (createOverlayContent).
    let recallImpactPromise = null;
    if (isRepeatAttempt) {
        // The genuine before/after the recall model produces from teaching it this
        // answer (generic average-difficulty problem).
        recallImpactPromise = new Promise(resolve => {
            scheduleIdle(() => {
                try {
                    const { beforeHours, afterHours } = recallImpact();
                    resolve((beforeHours !== null && afterHours !== null)
                        ? { beforeHours, afterHours }
                        : null);
                } catch (error) {
                    console.error('recall-impact retrain failed:', error);
                    resolve(null);
                }
            });
        });
    } else {
        // First-time attempt: nothing to display. Keep the retrain off the
        // critical path — deferred to browser idle (requestIdleCallback fires once
        // the thread is free, with a timeout fallback) so it never janks the
        // overlay fade-in or the next-problem load.
        scheduleIdle(() => { calculateAndStoreStatsResults(); });
    }
    // NB: the recall schedule is deliberately NOT re-uploaded per attempt — an
    // active user is suppressed by the cron anyway, so per-attempt replace-all
    // writes would be pure D1 churn. It's uploaded on load, on a ~30-min heartbeat,
    // and when the tab is hidden (see main.js / notifications.uploadSchedule).

    return { rankChange: state.displayRank - oldRank, rankChangeDirection, isCorrect, recallImpactPromise };
}