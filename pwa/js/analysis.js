// js/analysis.js

import * as Sentry from "@sentry/browser";

import { state, setAppMode, setAcceptClicks, setInitialAnalysisNode, setGhostMove, setReviewStartTime, setAnalysisHintMove, setCandidateMoves } from './state.js';
import { fetchAnalysis } from './analysis-api.js';
import { loadSgf, playMoveOnBranch, undoMove, getCurrentNode, redraw } from './board.js';
import { updateAnalysisUI, updateAllUI } from './ui.js';
import { playStoneSound } from './audio.js';
import { getTerm } from './i18n.js';
import { getScoreFromComment, getWinrateFromComment, findBestMoveFromChildren, formatEvalComment, setEvalComment } from './sgf.js';
import { APP_MODES, ANALYSIS_STATUS } from './constants.js';
import { populateMotifBubble } from './motif-bubble.js';
import { onEnterReview as onboardingEnterReview } from './onboarding.js';


/**
 * Whether `node` is a branch the pipeline played out to a DOUBLE PASS.
 *
 * Follows children[0]: the pipeline appends children in its candidate order and
 * always recurses into the first, so the first child IS the continuation it
 * followed. Measured on rebuilt 13x13 files this holds for 120/120 answers and
 * 120/120 best refutations, and accepts 0 of 10,645 other root variations.
 *
 * Two cheaper-looking tests are both wrong, and were measured to be. "A pass
 * appears somewhere in the subtree" is the deduction behind two corpus purges.
 * "A pass-PAIR appears somewhere" is not much better: the pipeline's shallow
 * sweep writes a move and its best response, so a pass answered by a pass
 * manufactures the shape with nothing played out — 32,445 extra children. And
 * following the highest-visit child drifts off the line entirely, because the
 * engine ranks its candidates by LCB rather than raw visits.
 *
 * This says the line was PLAYED OUT, not that it resolves the way the problem
 * claims — 7.5% of principal lines end with the solver losing. Whether a
 * resolved line is a win is read from the evaluation, not from here.
 */
function subtreeReachesConclusion(node) {
    if (!node) return false;
    const isPass = (n) => !!(n.move && n.move.pass);
    let n = node;
    let prevWasPass = isPass(node);
    // The pipeline caps a variation at MAX_VARIATION_DEPTH = 800 plies.
    for (let step = 0; step < 802; step++) {
        const kids = (n.children || []).filter(c => c.move);
        if (kids.length === 0) return false;
        n = kids[0];
        const cur = isPass(n);
        if (cur && prevWasPass) return true;
        prevWasPass = cur;
    }
    return false;
}

/**
 * Initializes and enters the analysis mode.
 */
export function enterAnalysisMode() {
    Sentry.addBreadcrumb({
        category: 'flow',
        message: 'enterAnalysisMode started',
        level: 'info'
    });
    return new Promise((resolve) => {
        setGhostMove(null); // Clear any ghost stone from previous modes
        setAnalysisHintMove(null); // Clear stale hint from a prior analysis session
        setCandidateMoves([]);

        if (!state.previousProblem || !state.previousProblem.sgf) {
            setAppMode(APP_MODES.SOLVING);
            // Nothing to review — stay clickable in solving mode so a button1
            // lock taken by the caller (see main.js) is always released.
            setAcceptClicks(true);
            updateAllUI();
            resolve();
            return;
        }

        setAppMode(APP_MODES.ANALYZING);
        setAcceptClicks(true);
        setReviewStartTime(Date.now());
        loadSgf(state.previousProblem.sgf);

        // Defer the rest of the initialization to the next event loop tick.
        // This is a workaround for a subtle timing issue in the wgo.js library where
        // the kifuReader object is not fully initialized immediately after creation.
        // This was causing issues where the initial analysis would fail because the
        // children of the current node were not yet available.
        setTimeout(() => {
            setInitialAnalysisNode(getCurrentNode());
            updateAllUI();
            // Surface the correct move's motifs for the problem under review
            // (the previous problem, same entry whose SGF was loaded above).
            populateMotifBubble(state.previousProblem);
            // First-run onboarding: offer a review-context bubble (explore-any-
            // move, problem rank, share). No-op for existing users.
            onboardingEnterReview();
            requestAndDisplayAnalysis();
            resolve();
        }, 0);
    });
}

/**
 * Handles clicks on the board during analysis mode.
 */
// Returns whether the click played a move — the typed-coordinate path
// (board.playTypedMove) uses this for its invalid-input feedback.
export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return false;

    if (!playMoveOnBranch(x, y)) return false;
    playStoneSound();
    updateAllUI();
    requestAndDisplayAnalysis();
    return true;
}

/**
 * Handles the "Undo" button click.
 */
export function undoLastMove() {
    if (!state.acceptClicks) return;

    if (getCurrentNode() === state.initialAnalysisNode) {
        return; // Don't undo past the initial position
    }

    if (undoMove()) {
        updateAllUI(); // Update the whole UI, including button states
        requestAndDisplayAnalysis();
    }
}

/**
 * Converts the current board state to an SGF string for the server.
 */
const getCurrentSgf = () => {
    let sgf = "(;FF[4]GM[1]CA[UTF-8]SZ[" + state.board.size + "]";

    if (state.kifuReader.kifu.info && state.kifuReader.kifu.info.RU) {
        sgf += "RU[" + state.kifuReader.kifu.info.RU + "]";
    } else {
        sgf += "RU[Japanese]";
    }

    if (state.kifuReader.kifu.info && state.kifuReader.kifu.info.KM) {
        sgf += "KM[" + state.kifuReader.kifu.info.KM + "]";
    } else {
        sgf += "KM[0.5]";
    }

    const moves = [];
    let currentNode = getCurrentNode();

    while (currentNode && currentNode.move) {
        moves.unshift(currentNode.move);
        currentNode = currentNode.parent;
    }

    for (const move of moves) {
        if (move.pass) {
            sgf += ";" + (move.c === WGo.B ? "B" : "W") + "[tt]";
        } else {
            const x = String.fromCharCode(97 + move.x);
            const y = String.fromCharCode(97 + move.y);
            sgf += ";" + (move.c === WGo.B ? "B" : "W") + "[" + x + y + "]";
        }
    }

    sgf += ")";
    return sgf;
}

/**
 * Fetches analysis from the server and updates the UI without blocking.
 */
async function requestAndDisplayAnalysis() {
    setAnalysisHintMove(null);
    setCandidateMoves([]);
    redraw();

    try {
        const currentNode = getCurrentNode();
        const children = currentNode.children;

        // Collect every engine-evaluated child. SGF-baked children carry a "X%/Y" comment
        // from the pipeline; API-added nodes (this session only) carry the same comment
        // shape plus _origin='api'. Anything without a %/ comment is a user-explored
        // branch and isn't a candidate.
        //
        // Display rule:
        //   - Black to play: show only the best move (the move the user is being asked
        //     to find).
        //   - White to play: waterfall —
        //       1. SGF moves whose line was played out to a conclusion → among those, show the move
        //          with the highest winrate and the move with the highest score, both
        //          from white's perspective (i.e. lowest stored values, since
        //          bots/analysis.cfg reports from black). One or two ghost stones.
        //       2. Otherwise, any SGF move at all → show only the best SGF move.
        //       3. Otherwise (only API additions exist) → show the best API move.
        //     Pipeline-baked variations terminate in a DOUBLE pass once solved, and
        //     subtreeReachesConclusion follows the first-child chain to check for one,
        //     so a branch it rejects is a dead-end evaluation that wasn't pursued to a
        //     conclusion.
        // The API-cache check still uses *all* %/ children so we don't re-query the
        // API for a position we've already evaluated this session.
        const allEvaluatedCandidates = [];
        const resolvedChildren = [];  // branches the pipeline played out to a double pass
        let bestChild = null;        // first %/ child of any origin (best overall)
        let bestSgfChild = null;     // first %/ child that isn't an API addition
        for (const child of children) {
            if (!child.move || !child.comment || !child.comment.match(/%\//)) continue;
            allEvaluatedCandidates.push({ x: child.move.x, y: child.move.y });
            const isApi = child._origin === 'api';
            if (!isApi && subtreeReachesConclusion(child)) {
                resolvedChildren.push(child);
            }
            if (!bestChild) bestChild = child;
            if (!bestSgfChild && !isApi) bestSgfChild = child;
        }

        // Prefer the branches that were played out to a conclusion wherever the
        // SGF has them. Selection below is unchanged — only the pool narrows, so
        // a node with two resolved white moves still draws two ghosts.
        const displayPool = resolvedChildren.length > 0 ? resolvedChildren : null;

        if (allEvaluatedCandidates.length > 0) {
            const isBlackToPlay = state.kifuReader.game.turn === WGo.B;

            let displayChild;
            let candidatesToShow;
            if (isBlackToPlay) {
                // Show the SAME move the solve path accepts as correct
                // (findBestMoveFromChildren — winrate-best, score tie-break), NOT
                // children[0]. A minority of problems have the generator's
                // first-listed variation be a losing move (e.g. 53dee5db:
                // children[0]=lg 60.6%/-0.5 loses, real answer ji 78.5%/+2.3), and
                // trusting document order marked that losing move as the solution —
                // disagreeing with what isCorrectMove would accept.
                //
                // The pool narrows to the resolved branches when there are any,
                // but the RULE stays findBestMoveFromChildren so that display
                // and isCorrectMove cannot diverge. Correctness itself is never
                // read off resolution: how far a line was searched must not
                // redefine what counts as the right answer.
                displayChild = findBestMoveFromChildren(displayPool || children)
                    || findBestMoveFromChildren(children) || bestChild;
                candidatesToShow = [{ x: displayChild.move.x, y: displayChild.move.y }];
            } else if (displayPool) {
                // Comments are reported with reportAnalysisWinratesAs = BLACK, so
                // white's best = lowest winrate / lowest score among the candidates.
                let bestWinrateChild = null;
                let bestWinrate = Infinity;
                let bestScoreChild = null;
                let bestScore = Infinity;
                for (const child of (displayPool || sgfChildrenLeadingToEnd)) {
                    const wr = getWinrateFromComment(child.comment);
                    if (wr !== null && wr < bestWinrate) {
                        bestWinrate = wr;
                        bestWinrateChild = child;
                    }
                    const sc = getScoreFromComment(child.comment);
                    if (sc !== null && sc < bestScore) {
                        bestScore = sc;
                        bestScoreChild = child;
                    }
                }
                const picks = [];
                const seen = new Set();
                for (const child of [bestWinrateChild, bestScoreChild]) {
                    if (!child) continue;
                    const key = `${child.move.x},${child.move.y}`;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    picks.push(child);
                }
                if (picks.length === 0) {
                    // Fall back to the SGF order when neither metric was
                    // parseable, from the same pool we just scanned.
                    picks.push(displayPool[0]);
                }
                displayChild = picks[0];
                candidatesToShow = picks.map(c => ({ x: c.move.x, y: c.move.y }));
            } else if (bestSgfChild) {
                displayChild = bestSgfChild;
                candidatesToShow = [{ x: bestSgfChild.move.x, y: bestSgfChild.move.y }];
            } else {
                // Only API children exist — show the API best.
                displayChild = bestChild;
                candidatesToShow = [{ x: bestChild.move.x, y: bestChild.move.y }];
            }

            const displayScore = getScoreFromComment(displayChild.comment);
            const scoreText = displayScore !== null ? `${displayScore.toFixed(1)}` : 'N/A';
            setAnalysisHintMove(displayChild.move);
            setCandidateMoves(candidatesToShow);
            updateAnalysisUI(scoreText, displayChild.move);
            redraw();
            return;
        }

        updateAnalysisUI(getTerm('analyzing'), null);

        const sgf = getCurrentSgf();
        let { bestMove: bestMoveSgf, winrate, score, visits } = await fetchAnalysis(sgf);

        if (score === ANALYSIS_STATUS.RETRY) {
            const currentSgfCheck = getCurrentSgf();
            if (currentSgfCheck === sgf) {
                ({ bestMove: bestMoveSgf, winrate, score, visits } = await fetchAnalysis(sgf));
            }
        }

        if (state.appMode === APP_MODES.ANALYZING && bestMoveSgf) {
            const currentSgfCheck = getCurrentSgf();
            if (currentSgfCheck === sgf) {
                // The server can answer a literal "pass" — there is no board
                // coordinate to decode (charCodeAt on "pass" fabricates (15, 0):
                // a phantom node whose hint marker renders nowhere). Show the
                // score with no on-board hint instead.
                if (bestMoveSgf === 'pass' || bestMoveSgf.length < 2) {
                    setAnalysisHintMove(null);
                    setCandidateMoves([]);
                    updateAnalysisUI(score, null);
                    redraw();
                    return;
                }
                const bestMove = {
                    x: bestMoveSgf.charCodeAt(0) - 97,
                    y: bestMoveSgf.charCodeAt(1) - 97,
                    c: state.kifuReader.game.turn
                };
                // The early-return above guarantees no existing child has a %/ comment, so any
                // matching existing child is a user-exploration node — safe to attach API data.
                const existingChild = children.find(child => child.move && child.move.x === bestMove.x && child.move.y === bestMove.y);
                // One visit is the raw policy — an opinion with no reading behind
                // it, which is what the blinking-red marker warns about. More
                // than one means the server actually searched, so the number has
                // been earned and the warning would be misleading. _origin stays
                // 'api' either way: the move-selection waterfall above keys on it
                // to tell server answers from the SGF's own.
                const searched = typeof visits === 'number' && visits > 1;
                // Record the server's OWN numbers in the corpus comment format,
                // so an API node reads back through the same parsers as an
                // SGF-baked one — including the visit count, which is what tells
                // a policy opinion from a reading. Both engines report from
                // BLACK (analysis_server/analysis.cfg, bots/analysis.cfg), so
                // there is no frame to convert. A response missing any of the
                // three carries no comment rather than an invented one: the
                // cache check above then re-queries this position later, and the
                // score shown below comes from the response itself either way.
                const evaluated = Number.isFinite(winrate) && Number.isFinite(score)
                    && Number.isFinite(visits);
                if (existingChild) {
                    if (evaluated) setEvalComment(existingChild, winrate, score, visits);
                    existingChild._origin = 'api';
                    existingChild._searched = searched;
                    const index = children.indexOf(existingChild);
                    if (index > 0) {
                        children.splice(index, 1);
                        children.unshift(existingChild);
                    }
                } else {
                    const newNode = new WGo.KNode(evaluated
                        ? { move: bestMove, comment: formatEvalComment(winrate, score, visits) }
                        : { move: bestMove });
                    newNode._origin = 'api';
                    newNode._searched = searched;
                    currentNode.children.unshift(newNode);
                    newNode.parent = currentNode;
                }
                // Path 2 only fires when the cache check found no SGF or API
                // children at all, so this position falls into white-case-3 of
                // the waterfall ("only API available — show best API"). Black's
                // rule is the same: show the single best move. Either way we
                // render the API best as the on-board hint.
                setAnalysisHintMove(bestMove);
                setCandidateMoves([{ x: bestMove.x, y: bestMove.y }]);
                updateAnalysisUI(score, bestMove);
                redraw();
            }
        } else if (state.appMode !== APP_MODES.ANALYZING) {
            updateAnalysisUI(score, null);
        } else if (getCurrentSgf() === sgf) {
            // Still analyzing this position but the server gave no move back:
            // fetchAnalysis swallows network/server failures into bestMove:null
            // (only 499 maps to RETRY), so without this branch the pane stays
            // on "Analyzing…" forever.
            updateAnalysisUI(getTerm('error'), null);
        }

    } catch (error) {
        console.error("Analysis request failed:", error);
        if (state.appMode === APP_MODES.ANALYZING) {
            updateAnalysisUI(getTerm('error'), null);
        }
    }
}