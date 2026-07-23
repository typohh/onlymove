// js/game-logic.js

import * as Sentry from "@sentry/browser";

import { state, setAcceptClicks, setAppMode, setReviewStartTime } from './state.js';
import { getCurrentNode, getLegalMoveNode, playMoveByIndex, playNextMove } from './board.js';
import { fadeOverlay } from './overlays.js';
import { playStoneSound } from './audio.js';
import { isCorrectMove, tsumegoIdOf, toBoardNotation, toCanonicalXy } from './sgf.js';
import { APP_MODES, MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { MODEL_INDICES, benefitConfidenceForVar } from './model.js';
import { getTerm } from './i18n.js';
import { sendReport, updateUserStats } from './user-stats.js';
import { completeCurrentProblem, advanceToNextProblem } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';
import { fetchPickedEmbedding } from './picked-embedding.js';
import { clearShownNotifications } from './notifications.js';

const median = (values) => {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// --- "Would more time help THIS user?" gate (rushing / insufficient-review overlays) ---
// The stats-model solving/review-time coefficient (logit per minute) is biased
// NEGATIVE by residual-difficulty confounding: a problem harder-than-its-rating
// for this user takes longer AND fails more, so the within-user time effect soaks
// up difficulty the model didn't capture (it controls only for MEASURED rank).
// Quantified offline in auxiliary/analyze_time_benefit.py: the population-mean
// solving-time coefficient is ~ -1.25 logit/min (the bias); review time shows no
// such bias (~0). We recenter by that bias, then require the corrected estimate to
// be positive with BENEFIT_CONFIDENCE certainty. That second step also handles
// cold start for free: a user with little data has a wide SE, so the recentered
// estimate sits near 50% and never clears the bar — no nagging until there's real
// evidence this user benefits from slowing down. Captures ~11% of users at 0.80,
// ~18% at 0.65 (solving time, client units). Raise the bar to nag fewer users.
const BENEFIT_CONFIDENCE = 0.80;      // P(true benefit > 0) required to show the nudge
// benefitConfidenceForVar (recenters the per-user time/review coeff by the
// population confound) and setTimeBias now live in model.js, alongside the L2
// prior the same coefficient is shrunk toward — they must stay in lockstep.


// `at` is the attempt's event time (ms), minted ONCE per answer in the click
// handler and shared verbatim between the report payload (createdAt) and the
// answerHistory entry (date, via updateUserStats) — the client owns the event
// time and the server inserts it as created_at, so row and entry match EXACTLY
// (the basis of account-sync's exact own-row matching; no fuzzy window).
const buildReportPayload = (success, x, y, at) => {
    const currentProblem = state.currentProblem;
    const timeUsed = state.accumulatedSolvingTime + (state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0);
    const problemId = tsumegoIdOf(currentProblem?.filename) ?? null;
    const prevs = problemId ? state.answerHistory.filter(a => a.problemId === problemId) : [];
    const reviewTime = prevs.reduce((sum, ans) => sum + (ans.reviewTime || 0), 0);
    // Report the move in the CANONICAL orientation-0 frame. The board (and the
    // in-memory kifu) live in the SERVED frame — for repeats a random D4
    // reorientation, which no other device can map back — so the click is
    // inverse-transformed through the problem's orientation before encoding.
    // This makes the reported coordinate match the shard SGF for every
    // attempt, letting cross-device sync derive the picked cell for repeats
    // too (rows from before CANONICAL_MOVE_ERA_MS are served-frame; see
    // account-sync.mergeRowsForSize for the legacy gate).
    const canonical = toCanonicalXy(x, y, currentProblem?.orientation ?? 0, state.boardSize);
    return {
        success,
        move: toBoardNotation(canonical.x, canonical.y, state.boardSize),
        timeUsed,
        reviewTime,
        isRepeat: !!currentProblem?.isRepeat,
        at,
    };
};

const handleCorrectAnswer = async (x, y) => {
    setAppMode(APP_MODES.SOLVING);
    const at = Date.now();
    sendReport(buildReportPayload(true, x, y, at));
    const stats = await updateUserStats(true, { at });

    // Move the solved problem into the previous slot and — if it was the standing
    // challenge — start reselecting one NOW, but do NOT await it: the reselection
    // can need a shard download, which now runs under the white overlay instead
    // of delaying its start. The synchronous prefix of completeCurrentProblem
    // still runs before the fade (previous slot set, resume slot cleared — a
    // reload mid-transition can't re-serve the answered problem), and once the
    // reselection lands, challengeProblem is persisted so a reopen can never
    // re-serve the just-answered problem (the bug the old `answered` flag tried
    // and failed to patch); a kill before it lands is covered by
    // challengeAlreadyAnswered at the next selection.
    const completed = completeCurrentProblem();
    completed.catch(() => {}); // surfaced via the overlay callback's catch below

    // The overlay holds until the reselection AND the next-problem load finish
    // (never lifting onto an empty board). advanceToNextProblem must stay inside
    // the callback — it repaints the board, which has to happen under cover.
    fadeOverlay(1000, () => {
        return completed
            .then(() => advanceToNextProblem())
            .catch(error => console.error("Error loading next tsumego:", error));
    }, stats);
}

async function _handleIncorrectMoveSequence(moveIndex, x, y) {
    Sentry.addBreadcrumb({
        category: 'flow',
        message: '_handleIncorrectMoveSequence started',
        level: 'info',
        data: { moveIndex }
    });
    const delay = (ms) => new Promise(res => setTimeout(res, ms));

    // Fire the picked-move embedding fetch in the background. We never block
    // on it — the answer is recorded immediately with just the cell index,
    // and updateUserStats attaches a defer-patch handler. Embedding-backfill
    // also re-fetches missing pickedEmbeddings on the next app load, so a
    // failed/slow live fetch is recoverable.
    const currentProblem = state.currentProblem;
    const tsumegoId = tsumegoIdOf(currentProblem?.filename);
    const zipFilename = currentProblem?.zipFilename;
    let pickedEmbeddingPromise = null;
    if (tsumegoId && zipFilename && state.appMode === APP_MODES.SOLVING) {
        pickedEmbeddingPromise = fetchPickedEmbedding({
            tsumegoId,
            cellIndex: moveIndex,
            zipFilename,
            boardSize: state.boardSize,
        });
    }

    playMoveByIndex(moveIndex);
    playStoneSound();
    await delay(1000);

    playNextMove();
    playStoneSound();

    const at = Date.now();
    sendReport(buildReportPayload(false, x, y, at));
    const stats = await updateUserStats(false, {
        pickedCellIndex: moveIndex,
        pickedEmbeddingPromise,
        at,
    });

    // Calculate overlay duration and message based on time spent
    const timeSpent = state.accumulatedSolvingTime + (state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0);
    let duration = 2000; // 2s default
    let message = '';
    let buttonText = null;
    const statsResults = state.statsResults ? state.statsResults.results : null;
    const statsAreAvailable = statsResults && statsResults.length > 0;

    if (state.currentProblem?.isRepeat) {
        let reviewConf = 0;
        let medianReviewTime = 0;
        let previousReviewTime = 0;
        if (statsAreAvailable) {
            const recentHistory = state.answerHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
            const reviewTimeValues = recentHistory.map(p => p.reviewTime).filter(t => t !== undefined && t > 0);
            medianReviewTime = median(reviewTimeValues);

            // Get accumulated review time for this problem
            const currentProblem = state.currentProblem;
            const problemId = tsumegoIdOf(currentProblem.filename);
            const previousAnswers = state.answerHistory.filter(ans => ans.problemId === problemId);
            if (previousAnswers.length > 0) {
                const lastAnswer = previousAnswers[previousAnswers.length - 1];
                previousReviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
            }

            const reviewStat = statsResults.find(r => r.varIndex === MODEL_INDICES.REVIEW_TIME);
            reviewConf = benefitConfidenceForVar(reviewStat, MODEL_INDICES.REVIEW_TIME);
        }

        if (reviewConf >= BENEFIT_CONFIDENCE && previousReviewTime < medianReviewTime) {
            duration = 6000;
            const x = Math.round(previousReviewTime / 1000);
            message = getTerm('insufficientReviewMessage', { seconds: x, confidence: (reviewConf * 100).toFixed(0) });
            buttonText = getTerm('iWontRushIt');
        }
    } else { // Careless mistake for new problems
        let timeConf = 0;
        let medianSolvingTime = 0;
        if (statsAreAvailable) {
            const recentHistory = state.answerHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
            const timeTakenValues = recentHistory.map(p => p.timeTaken).filter(t => t !== undefined);
            medianSolvingTime = median(timeTakenValues);
            const timeStat = statsResults.find(r => r.varIndex === MODEL_INDICES.TIME_SOLVING);
            timeConf = benefitConfidenceForVar(timeStat, MODEL_INDICES.TIME_SOLVING);
        }

        if (timeConf >= BENEFIT_CONFIDENCE && state.accumulatedSolvingTime < medianSolvingTime) {
            duration = 6000; // This duration will be ignored as buttonText is set
            const x = Math.round(timeSpent / 1000);
            message = getTerm('rushingMessage', { seconds: x, confidence: (timeConf * 100).toFixed(0) });
            buttonText = getTerm('iWontRushIt');
        }
    }

    // Show the red result overlay and, under its cover, transition into review:
    // completeCurrentProblem moves the failed problem into the previous slot,
    // clears currentProblem (so review-exit advances to a fresh selection and a
    // mid-review reload can't resume the already-answered problem) and
    // reselects the challenge — the next-problem PREFETCH — then
    // enterAnalysisMode paints the review board. The overlay holds until that
    // work is done, so it never lifts onto a stale board; the NEXT problem is
    // deliberately NOT loaded here — redisplayCurrentProblem selects it when
    // the user leaves review (behind a grey loading fade if selection is slow).
    await fadeOverlay(duration, async () => {
        await completeCurrentProblem();
        await enterAnalysisMode();
    }, stats, message, { background: 'rgba(139, 0, 0, 0.9)', text: 'white' }, { buttonText });

    // The review clock starts when the user can actually SEE the review board:
    // enterAnalysisMode stamped reviewStartTime at the under-cover paint, which
    // would bill the overlay hold (indefinite on the button-gated rushing
    // nudge) as review time.
    setReviewStartTime(Date.now());
}

// Returns whether the click played a move — the typed-coordinate path
// (board.playTypedMove) uses this for its invalid-input feedback.
export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return false;

    const moveNode = getLegalMoveNode(x, y);
    if (!moveNode) {
        return false;
    }

    setAcceptClicks(false);

    // The user is actively solving — dismiss any lingering recall reminder in
    // the notification tray. Fire-and-forget; doesn't gate the move.
    clearShownNotifications();

    const children = getCurrentNode().children;
    const moveIndex = children.indexOf(moveNode);
    if (isCorrectMove(moveNode, children)) {
        playMoveByIndex(moveIndex);
        playStoneSound();
        handleCorrectAnswer(x, y);
    } else {
        _handleIncorrectMoveSequence(moveIndex, x, y);
    }
    return true;
}