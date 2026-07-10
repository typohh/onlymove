// js/analysis.js

import * as Sentry from "@sentry/browser";

import { state, setAppMode, setAcceptClicks, setInitialAnalysisNode, setGhostMove, setReviewStartTime, setAnalysisHintMove, setCandidateMoves } from './state.js';
import { fetchAnalysis } from './api.js';
import { loadSgf, playMoveOnBranch, undoMove, getCurrentNode, redraw } from './board.js';
import { updateAnalysisUI, updateAllUI } from './ui.js';
import { playStoneSound } from './audio.js';
import { getTerm } from './i18n.js';
import { getScoreFromComment, getWinrateFromComment, findBestMoveFromChildren } from './sgf.js';
import { APP_MODES, ANALYSIS_STATUS } from './constants.js';
import { populateMotifBubble } from './motif-bubble.js';


// Iterative DFS — true if the subtree rooted at `node` contains a pass move.
// Pipeline-baked variations terminate with a pass once the position is fully
// resolved; subtrees with no pass anywhere are dead-end engine evaluations
// that weren't pursued to a conclusion.
function subtreeReachesPass(node) {
    if (!node) return false;
    const stack = [node];
    while (stack.length > 0) {
        const n = stack.pop();
        if (n.move && n.move.pass) return true;
        if (n.children && n.children.length > 0) {
            for (const c of n.children) stack.push(c);
        }
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
            requestAndDisplayAnalysis();
            resolve();
        }, 0);
    });
}

/**
 * Handles clicks on the board during analysis mode.
 */
export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return;

    if (playMoveOnBranch(x, y)) {
        playStoneSound();
        updateAllUI();
        requestAndDisplayAnalysis();
    }
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
        //       1. SGF moves whose subtree reaches a pass → among those, show the move
        //          with the highest winrate and the move with the highest score, both
        //          from white's perspective (i.e. lowest stored values, since
        //          bots/analysis.cfg reports from black). One or two ghost stones.
        //       2. Otherwise, any SGF move at all → show only the best SGF move.
        //       3. Otherwise (only API additions exist) → show the best API move.
        //     Pipeline-baked variations terminate with a pass once solved, so a subtree
        //     with no pass anywhere is a dead-end engine evaluation that wasn't pursued
        //     to a conclusion.
        // The API-cache check still uses *all* %/ children so we don't re-query the
        // API for a position we've already evaluated this session.
        const allEvaluatedCandidates = [];
        const sgfChildrenLeadingToEnd = [];
        let bestChild = null;        // first %/ child of any origin (best overall)
        let bestSgfChild = null;     // first %/ child that isn't an API addition
        for (const child of children) {
            if (!child.move || !child.comment || !child.comment.match(/%\//)) continue;
            allEvaluatedCandidates.push({ x: child.move.x, y: child.move.y });
            const isApi = child._origin === 'api';
            if (!isApi && subtreeReachesPass(child)) {
                sgfChildrenLeadingToEnd.push(child);
            }
            if (!bestChild) bestChild = child;
            if (!bestSgfChild && !isApi) bestSgfChild = child;
        }

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
                displayChild = findBestMoveFromChildren(children) || bestChild;
                candidatesToShow = [{ x: displayChild.move.x, y: displayChild.move.y }];
            } else if (sgfChildrenLeadingToEnd.length > 0) {
                // Comments are reported with reportAnalysisWinratesAs = BLACK, so
                // white's best = lowest winrate / lowest score among the candidates.
                let bestWinrateChild = null;
                let bestWinrate = Infinity;
                let bestScoreChild = null;
                let bestScore = Infinity;
                for (const child of sgfChildrenLeadingToEnd) {
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
                    // Fall back to the SGF order when neither metric was parseable.
                    picks.push(sgfChildrenLeadingToEnd[0]);
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
        let { bestMove: bestMoveSgf, score } = await fetchAnalysis(sgf);

        if (score === ANALYSIS_STATUS.RETRY) {
            const currentSgfCheck = getCurrentSgf();
            if (currentSgfCheck === sgf) {
                ({ bestMove: bestMoveSgf, score } = await fetchAnalysis(sgf));
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
                if (existingChild) {
                    existingChild.comment = `100%/${score}`;
                    existingChild._origin = 'api';
                    const index = children.indexOf(existingChild);
                    if (index > 0) {
                        children.splice(index, 1);
                        children.unshift(existingChild);
                    }
                } else {
                    const newNode = new WGo.KNode({
                        move: bestMove,
                        comment: `100%/${score}`
                    });
                    newNode._origin = 'api';
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