// js/analysis.js

import { state, setAppMode, setAcceptClicks, setInitialAnalysisNode, setGhostMove, setReviewStartTime, setAnalysisHintMove, setCandidateMoves } from './state.js';
import { fetchAnalysis } from './api.js';
import { loadSgf, playMoveOnBranch, undoMove, getCurrentNode, redraw } from './board.js';
import { updateAnalysisUI, updateAllUI } from './ui.js';
import { playStoneSound } from './audio.js';
import { getTerm } from './i18n.js';
import { getScoreFromComment } from './sgf.js';
import { APP_MODES, ANALYSIS_STATUS } from './constants.js';


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

        if (!state.problemHistory || state.problemHistory.length < 2 || !state.problemHistory[1].sgf ) {
            setAppMode(APP_MODES.SOLVING);
            updateAllUI();
            resolve();
            return;
        }

        setAppMode(APP_MODES.ANALYZING);
        setAcceptClicks(true);
        setReviewStartTime(Date.now());
        loadSgf(state.problemHistory[1].sgf);

        // Defer the rest of the initialization to the next event loop tick.
        // This is a workaround for a subtle timing issue in the wgo.js library where
        // the kifuReader object is not fully initialized immediately after creation.
        // This was causing issues where the initial analysis would fail because the
        // children of the current node were not yet available.
        setTimeout(() => {
            setInitialAnalysisNode(getCurrentNode());
            updateAllUI();
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

        // Collect every engine-evaluated child as a candidate. SGF-baked children carry a
        // "X%/Y" comment from the pipeline; API-added nodes (this session only) carry the
        // same comment shape plus _origin='api'. Anything without a %/ comment is a
        // user-explored branch and isn't a candidate.
        const sgfCandidates = [];
        let bestSgfChild = null;
        for (const child of children) {
            if (!child.move || !child.comment || !child.comment.match(/%\//)) continue;
            sgfCandidates.push({ x: child.move.x, y: child.move.y });
            if (!bestSgfChild) bestSgfChild = child;
        }

        // If the SGF already evaluated this position, never call the API — baked analysis is
        // higher quality. Show all baked candidates and use the first as the score-display hint.
        if (sgfCandidates.length > 0) {
            const bestMove = bestSgfChild.move;
            const bestScore = getScoreFromComment(bestSgfChild.comment);
            const scoreText = bestScore !== null ? `${bestScore.toFixed(1)}` : 'N/A';
            setAnalysisHintMove(bestMove);
            setCandidateMoves(sgfCandidates);
            updateAnalysisUI(scoreText, bestMove);
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
                setAnalysisHintMove(bestMove);
                setCandidateMoves([{ x: bestMove.x, y: bestMove.y }]);
                updateAnalysisUI(score, bestMove);
                redraw();
            }
        } else if (state.appMode !== APP_MODES.ANALYZING) {
            updateAnalysisUI(score, null);
        }

    } catch (error) {
        console.error("Analysis request failed:", error);
        if (state.appMode === APP_MODES.ANALYZING) {
            updateAnalysisUI(getTerm('error'), null);
        }
    }
}