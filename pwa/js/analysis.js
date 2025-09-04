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
    if (!state.problemHistory || state.problemHistory.length < 2) {
        console.error("Not enough history to enter analysis mode.");
        setAppMode('solving');
        updateAllUI();
        return;
    }

    const problemToAnalyze = state.problemHistory[1];
    const sgfToAnalyze = problemToAnalyze.sgf;
    const tsumegoFilename = problemToAnalyze.filename;

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
    // Set initialAnalysisNode after a small delay to ensure the SGF is fully loaded
    setTimeout(() => {
        initialAnalysisNode = getCurrentNode();
        requestAndDisplayAnalysis();
    }, 0);
    
    if (tsumegoFilename) {
        const tsumegoId = tsumegoFilename.replace('.sgf', '');
        updateUrlWithTsumegoId(tsumegoId);
    }
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
    analysisHintMove = null;
    redraw();

    try {
        const currentNode = getCurrentNode();
        const children = currentNode.children;
        const bestMoveNode = findBestMoveFromChildren(children);

        // Special case: if we're at the initial position and it has no score,
        // but its children do, use the best child's score
        // Check if we're at the initial position by checking if it's the same node object
        // or if it has the same properties (as a fallback)
        const isInitialPosition = currentNode === initialAnalysisNode || 
            (!currentNode.parent && currentNode.comment && currentNode.comment.includes('Q'));
            
        if (isInitialPosition && bestMoveNode) {
            const bestMove = bestMoveNode.move;
            const bestScore = getScoreFromComment(bestMoveNode.comment);
            
            // Use score if available, otherwise 'N/A'
            let scoreText = 'N/A';
            if (bestScore !== null) {
                scoreText = `${bestScore.toFixed(1)}`;
            }
            
            analysisHintMove = bestMove;
            updateAnalysisUI(scoreText, bestMove);
            redraw(analysisHintMove);
            return;
        }

        // For all other cases, show analyzing state first
        updateAnalysisUI(getTerm('analyzing'), null);

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
        let { bestMove: bestMoveSgf, score } = await fetchAnalysis(sgf);

        if (score === 'Retry') {
            const currentSgfCheck = getCurrentSgf();
            if (currentSgfCheck === sgf) {
                ({ bestMove: bestMoveSgf, score } = await fetchAnalysis(sgf));
            }
        }

        if (state.appMode === 'analyzing' && bestMoveSgf) {
            const currentSgfCheck = getCurrentSgf();
            if (currentSgfCheck === sgf) {
                const bestMove = {
                    x: bestMoveSgf.charCodeAt(0) - 97,
                    y: bestMoveSgf.charCodeAt(1) - 97,
                    c: state.kifuReader.game.turn
                };
                // Check if the move already exists in children
                const existingChild = children.find(child => child.move && child.move.x === bestMove.x && child.move.y === bestMove.y);
                if (existingChild) {
                    // Set comment if not set
                    if (!existingChild.comment) {
                        existingChild.comment = `100%/${score}`;
                    }
                    // Move to first if not already
                    const index = children.indexOf(existingChild);
                    if (index > 0) {
                        children.splice(index, 1);
                        children.unshift(existingChild);
                    }
                } else {
                    // Create new node
                    const newNode = new WGo.KNode({
                        move: bestMove,
                        comment: `100%/${score}`
                    });
                    // Prepend
                    currentNode.children.unshift(newNode);
                    newNode.parent = currentNode;
                }
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