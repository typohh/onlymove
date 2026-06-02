// js/game-logic.js

import { state, setAcceptClicks, setAppMode } from './state.js';
import { getCurrentNode, getLegalMoveNode, playMoveByIndex, playNextMove } from './board.js';
import { displayOverlay, fadeOverlay } from './overlays.js';
import { playStoneSound } from './audio.js';
import { isCorrectMove } from './sgf.js';
import { APP_MODES, MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { MODEL_INDICES } from './model.js';
import { getTerm } from './i18n.js';
import { sendReport, updateUserStats } from './user-stats.js';
import { loadTsumego } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';
import { fetchPickedEmbedding } from './picked-embedding.js';
import { clearShownNotifications } from './notifications.js';

const median = (values) => {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// Standard Go column letters skip 'I' (so A..H, J..T). Row number is 1-based
// from the bottom; WGo's y=0 is at the top, so row = boardSize - y.
const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';
const toBoardNotation = (x, y, size) => {
    if (x < 0 || y < 0 || x >= size || y >= size || x >= COL_LETTERS.length) return null;
    return COL_LETTERS[x] + (size - y);
};

const buildReportPayload = (success, x, y) => {
    const currentProblem = state.problemHistory[0];
    const timeUsed = state.accumulatedSolvingTime + (state.currentSolvingStartTime ? Date.now() - state.currentSolvingStartTime : 0);
    const problemId = currentProblem?.filename?.replace(/\.sgf$/, '') ?? null;
    const prevs = problemId ? state.answerHistory.filter(a => a.problemId === problemId) : [];
    const reviewTime = prevs.reduce((sum, ans) => sum + (ans.reviewTime || 0), 0);
    return {
        success,
        move: toBoardNotation(x, y, state.boardSize),
        timeUsed,
        reviewTime,
        isRepeat: !!currentProblem?.isRepeat,
    };
};

const handleCorrectAnswer = async (x, y) => {
    setAppMode(APP_MODES.SOLVING);
    sendReport(buildReportPayload(true, x, y));
    const stats = await updateUserStats(true);

    // Return the load promise so fadeOverlay holds the white screen up until
    // the next problem is actually loaded (never lifting onto an empty board).
    fadeOverlay(1000, () => {
        return loadTsumego(true).catch(error => console.error("Error loading next tsumego:", error));
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
    const currentProblem = state.problemHistory[0];
    const tsumegoId = currentProblem?.filename?.replace(/\.sgf$/, '');
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

    if (state.problemHistory[0]?.isRepeat) {
        let reviewTimeIsBeneficial = false;
        let medianReviewTime = 0;
        let previousReviewTime = 0;
        let reviewStat;
        if (statsAreAvailable) {
            const recentHistory = state.answerHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
            const reviewTimeValues = recentHistory.map(p => p.reviewTime).filter(t => t !== undefined && t > 0);
            medianReviewTime = median(reviewTimeValues);

            // Get accumulated review time for this problem
            const currentProblem = state.problemHistory[0];
            const problemId = currentProblem.filename.replace('.sgf', '');
            const previousAnswers = state.answerHistory.filter(ans => ans.problemId === problemId);
            if (previousAnswers.length > 0) {
                const lastAnswer = previousAnswers[previousAnswers.length - 1];
                previousReviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
            }

            const targetVarIndex = MODEL_INDICES.REVIEW_TIME;
            reviewStat = statsResults.find(r => r.varIndex === targetVarIndex);
            if (reviewStat && reviewStat.coeff > 0) {
                reviewTimeIsBeneficial = true;
            }
        }

        if (reviewTimeIsBeneficial && previousReviewTime < medianReviewTime) {
            duration = 6000;
            if (reviewStat) {
                const { coeff, se } = reviewStat;
                const z = Math.abs(coeff) / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = cdf * 100;
                const x = Math.round(previousReviewTime / 1000);
                message = getTerm('insufficientReviewMessage').replace('{seconds}', x).replace('{confidence}', conf.toFixed(0));
                buttonText = getTerm('iWontRushIt');
            }
        }
    } else { // Careless mistake for new problems
        let timeIsBeneficial = false;
        let medianSolvingTime = 0;
        if (statsAreAvailable) {
            const recentHistory = state.answerHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
            const timeTakenValues = recentHistory.map(p => p.timeTaken).filter(t => t !== undefined);
            medianSolvingTime = median(timeTakenValues);
            const timeStat = statsResults.find(r => r.varIndex === MODEL_INDICES.TIME_SOLVING);
            timeIsBeneficial = timeStat ? timeStat.coeff > 0 : false;
        }

        if (timeIsBeneficial && state.accumulatedSolvingTime < medianSolvingTime) {
            duration = 6000; // This duration will be ignored as buttonText is set
            const timeStat = statsResults.find(r => r.varIndex === MODEL_INDICES.TIME_SOLVING);
            const { coeff, se } = timeStat;
            const z = Math.abs(coeff) / se;
            const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
            const conf = cdf * 100;
            const x = Math.round(timeSpent / 1000);
            message = getTerm('rushingMessage').replace('{seconds}', x).replace('{confidence}', conf.toFixed(0));
            buttonText = getTerm('iWontRushIt');
        }
    }

    await new Promise((resolve) => displayOverlay(message, duration, resolve, undefined, stats, buttonText));

    await loadTsumego(true);
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