// js/game-logic.js

import * as Sentry from "@sentry/browser";

import { state, setAcceptClicks, setAppMode } from './state.js';
import { getCurrentNode, getLegalMoveNode, playMoveByIndex, playNextMove } from './board.js';
import { displayOverlay, fadeOverlay } from './overlays.js';
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


const buildReportPayload = (success, x, y) => {
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
    };
};

const handleCorrectAnswer = async (x, y) => {
    setAppMode(APP_MODES.SOLVING);
    sendReport(buildReportPayload(true, x, y));
    const stats = await updateUserStats(true);

    // Move the solved problem into the previous slot and — if it was the standing
    // challenge — reselect one NOW. challengeProblem is persisted, so once it's
    // replaced a reopen can never re-serve the just-answered problem (the bug the
    // old `answered` flag tried and failed to patch).
    await completeCurrentProblem();

    // Return the load promise so fadeOverlay holds the white screen up until
    // the next problem is actually loaded (never lifting onto an empty board).
    fadeOverlay(1000, () => {
        return advanceToNextProblem().catch(error => console.error("Error loading next tsumego:", error));
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

    sendReport(buildReportPayload(false, x, y));
    const stats = await updateUserStats(false, {
        pickedCellIndex: moveIndex,
        pickedEmbeddingPromise,
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

    await new Promise((resolve) => displayOverlay(message, duration, resolve, undefined, stats, buttonText));

    // Move the failed problem into the previous slot (reselecting the challenge
    // if it was one), then load the NEXT problem underneath as currentProblem.
    await completeCurrentProblem();
    await advanceToNextProblem();
    // advanceToNextProblem re-enabled clicks while appMode is still SOLVING; keep
    // them disabled across the SOLVING→ANALYZING flip so a tap landing in that
    // window can't be mis-routed as a solving-mode action. enterAnalysisMode
    // re-enables once it has switched to ANALYZING and paints the previous problem.
    setAcceptClicks(false);
    await enterAnalysisMode();
}

export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return;

    const moveNode = getLegalMoveNode(x, y);
    if (!moveNode) {
        return;
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
}