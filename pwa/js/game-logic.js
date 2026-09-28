// js/game-logic.js

import { addBreadcrumb } from "@sentry/browser";

import { state, setAcceptClicks, setAppMode, setReviewStartTime } from './state.js';
import { getCurrentNode, getLegalMoveNode, playMoveByIndex, playNextMove } from './board.js';
import { fadeOverlay } from './overlays.js';
import { playStoneSound } from './audio.js';
import { isCorrectMove, tsumegoIdOf, toBoardNotation, toCanonicalXy } from './sgf.js';
import { APP_MODES } from './constants.js';
import { sendReport, updateUserStats } from './attempt-recorder.js';
import { completeCurrentProblem, advanceToNextProblem } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';
import { fetchWrongAttemptStats } from './picked-embedding.js';

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
    // isCorrect is set here rather than trusted from updateUserStats: a linked
    // or cross-tab-dropped attempt returns {} (nothing recorded), and the
    // overlay's "Correct!" line must still show for it.
    const stats = { ...(await updateUserStats(true, { at })), isCorrect: true };

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

async function _handleIncorrectMoveSequence(moveIndex, x, y, variationNodes) {
    addBreadcrumb({
        category: 'flow',
        message: '_handleIncorrectMoveSequence started',
        level: 'info',
        data: { moveIndex }
    });
    const delay = (ms) => new Promise(res => setTimeout(res, ms));

    // Fire the wrong-attempt embedding fetch in the background — one Range
    // request for the problem's whole per-cell block, from which both the
    // picked-move vector and the availability-null masses come. We never
    // block on it — the answer is recorded immediately with just the cell
    // indices, and updateUserStats attaches a defer-patch handler.
    // Embedding-backfill also re-fetches missing fields on the next app load,
    // so a failed/slow live fetch is recoverable.
    //
    // `variationNodes` is the decision node's children as the click saw them,
    // handed down rather than re-read so nothing depends on where the board has
    // advanced to by now. It is the only reason this path may COMPUTE the null
    // instead of reading the published one: the visit weight of each alternative
    // is the third field of its own SGF comment, which is on the board here and
    // nowhere else in the client.
    const currentProblem = state.currentProblem;
    const tsumegoId = tsumegoIdOf(currentProblem?.filename);
    const zipFilename = currentProblem?.zipFilename;
    // The answer is variation 0 — the corpus contract isCorrectMove reads, and
    // the same row the published per-cell block puts the null's reference at.
    const correctIndex = 0;
    let wrongStatsPromise = null;
    if (tsumegoId && zipFilename && state.appMode === APP_MODES.SOLVING) {
        wrongStatsPromise = fetchWrongAttemptStats({
            tsumegoId,
            pickedIndex: moveIndex,
            correctIndex,
            variationNodes,
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
        wrongStatsPromise,
        at,
    });

    // Show the red result overlay and, under its cover, transition into review:
    // completeCurrentProblem moves the failed problem into the previous slot,
    // clears currentProblem (so review-exit advances to a fresh selection and a
    // mid-review reload can't resume the already-answered problem) and
    // reselects the challenge — the next-problem PREFETCH — then
    // enterAnalysisMode paints the review board. The overlay holds until that
    // work is done, so it never lifts onto a stale board; the NEXT problem is
    // deliberately NOT loaded here — redisplayCurrentProblem selects it when
    // the user leaves review (behind a grey loading fade if selection is slow).
    await fadeOverlay(2000, async () => {
        await completeCurrentProblem();
        await enterAnalysisMode();
    }, stats, null, { background: 'rgba(139, 0, 0, 0.9)', text: 'white' });

    // The review clock starts when the user can actually SEE the review board:
    // enterAnalysisMode stamped reviewStartTime at the under-cover paint, which
    // would bill the overlay hold as review time.
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

    const children = getCurrentNode().children;
    const moveIndex = children.indexOf(moveNode);
    if (isCorrectMove(moveNode, children)) {
        playMoveByIndex(moveIndex);
        playStoneSound();
        handleCorrectAnswer(x, y);
    } else {
        _handleIncorrectMoveSequence(moveIndex, x, y, children);
    }
    return true;
}