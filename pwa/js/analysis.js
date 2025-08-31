// js/analysis.js

import { state, setAppMode, setAcceptClicks } from './state.js';
import { fetchAnalysis } from './api.js';
import { loadSgf, playMoveOnBranch, undoMove, getCurrentNode, redraw } from './board.js';
import { updateAnalysisUI, updateAllUI } from './ui.js';
import { playStoneSound } from './audio.js';
import { getTerm, updateUrlWithTsumegoId } from './utils.js';
import { findBestMoveFromChildren, getScoreFromComment } from './sgf.js';

let analysisHintMove = null;
let initialAnalysisNode = null;

/**
 * Initializes and enters the analysis mode.
 */
export function enterAnalysisMode() {
    const sgfToAnalyze = state.previousSGF;
    const tsumegoFilename = state.previousProblemFilename;

    if (!sgfToAnalyze) {
        console.error("No SGF found to enter analysis mode.");
        setAppMode('solving');
        updateAllUI();
        return;
    }

    setAppMode('analyzing');
    updateAllUI();
    setAcceptClicks(true);

    loadSgf(sgfToAnalyze);
    initialAnalysisNode = getCurrentNode();

    if (tsumegoFilename) {
        const tsumegoId = tsumegoFilename.replace('.sgf', '');
        updateUrlWithTsumegoId(tsumegoId);
    }

    requestAndDisplayAnalysis();
}

/**
 * Handles clicks on the board during analysis mode.
 */
export function handleBoardClick(x, y) {
    if (!state.acceptClicks) return;

    if (playMoveOnBranch(x, y)) {
        playStoneSound();
        requestAndDisplayAnalysis();
    }
}

/**
 * Handles the "Undo" button click.
 */
export function undoLastMove() {
    if (!state.acceptClicks) return;

    if (getCurrentNode() === initialAnalysisNode) {
        return; // Don't undo past the initial position
    }

    if (undoMove()) {
        requestAndDisplayAnalysis();
    }
}

/**
 * Converts the current board state to an SGF string for the server.
 */
function getCurrentSgf() {
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
            sgf += ";" + (move.c === WGo.B ? "B" : "W") + "[]";
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
    updateAnalysisUI(getTerm('analyzing'), null);
    analysisHintMove = null;
    redraw();

    try {
        const children = getCurrentNode().children;
        const bestMoveNode = findBestMoveFromChildren(children);

        if (bestMoveNode) {
            const bestMove = bestMoveNode.move;
            const bestScore = getScoreFromComment(bestMoveNode.comment);
            const scoreText = bestScore !== null ? `${bestScore.toFixed(1)}` : 'N/A';
            analysisHintMove = bestMove;
            updateAnalysisUI(scoreText, bestMove);
            redraw(analysisHintMove);
            return;
        }

        const sgf = getCurrentSgf();
        const { bestMove: bestMoveSgf, score } = await fetchAnalysis(sgf);

        if (state.appMode === 'analyzing' && bestMoveSgf) {
            const currentSgf = getCurrentSgf();
            if (currentSgf === sgf) {
                const bestMove = {
                    x: bestMoveSgf.charCodeAt(0) - 97,
                    y: bestMoveSgf.charCodeAt(1) - 97,
                };
                analysisHintMove = bestMove;
                updateAnalysisUI(score, bestMove);
                redraw(analysisHintMove);
            }
        } else if (state.appMode !== 'analyzing') {
            updateAnalysisUI(score, null);
        }

    } catch (error) {
        console.error("Analysis request failed:", error);
        if (state.appMode === 'analyzing') {
            updateAnalysisUI(getTerm('error'), null);
        }
    }
}