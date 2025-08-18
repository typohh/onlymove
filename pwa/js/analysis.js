// js/analysis.js

import { state } from './state.js';
import { fetchAnalysis } from './api.js';
import { updateBoard, initializeBoard, initializeKifuFromSGF } from './board.js';
import { updateAnalysisUI, toggleAnalysisModeUI } from './ui.js';
import { playStoneSound } from './audio.js';

let analysisHintMove = null;

/**
 * Initializes and enters the analysis mode.
 */
export function enterAnalysisMode() {
    const sgfToAnalyze = state.previousSGF || state.currentSGF;

    if (!sgfToAnalyze) {
        console.error("No SGF found to enter analysis mode.");
        toggleAnalysisModeUI(false); 
        return;
    }

    state.appMode = 'analyzing';
    state.acceptClicks = false; // Will be enabled after the first analysis.

    // Use the same initialization as tsumego mode to ensure we start in the same position
    state.kifuReader = initializeKifuFromSGF(sgfToAnalyze);
    
    // Initialize board with the correct size
    initializeBoard(state.kifuReader.kifu.size);
    
    updateBoard();

    // Fetch analysis for the starting position. Clicks are enabled after it's done.
    requestAndDisplayAnalysis().then(() => {
        state.acceptClicks = true;
    });
}

/**
 * Handles clicks on the board during analysis mode.
 */
export function handleBoardClick(x, y) {
    if (!state.acceptClicks || !state.kifuReader) return;

    const game = state.kifuReader.game;
    const position = game.getPosition();
    
    // Check if there's already a stone at this position
    if (position.get(x, y) !== 0) return;

    // In analysis mode, we allow any legal move, so we create a new node and navigate to it
    playStoneSound();
    // Append the new move to the kifu tree
    const newNode = new WGo.KNode({
        move: { x: x, y: y, c: game.turn }
    });
    state.kifuReader.node.appendChild(newNode);
    state.kifuReader.next(state.kifuReader.node.children.length - 1);
    
    // Update the board immediately to show the user's stone
    updateBoard();
    
    // Request analysis in the background without blocking the UI
    requestAndDisplayAnalysis();
}

/**
 * Handles the "Undo" button click.
 */
export function undoLastMove() {
    if (!state.acceptClicks || !state.kifuReader || !state.kifuReader.node.parent) return;
    
    state.kifuReader.previous();
    updateBoard();
    
    // Request analysis for the new position in the background
    requestAndDisplayAnalysis();
}

/**
 * Converts the current board state to an SGF string for the server.
 * This generates a clean SGF with only the main line of moves.
 */
function getCurrentSgf() {
    // Create a new, clean SGF with required properties
    let sgf = "(;FF[4]GM[1]CA[UTF-8]SZ[" + state.kifuReader.kifu.size + "]";
    
    // Add rules and komi if available
    if (state.kifuReader.kifu.info && state.kifuReader.kifu.info.RU) {
        sgf += "RU[" + state.kifuReader.kifu.info.RU + "]";
    } else {
        sgf += "RU[Japanese]"; // Default rules
    }
    
    if (state.kifuReader.kifu.info && state.kifuReader.kifu.info.KM) {
        sgf += "KM[" + state.kifuReader.kifu.info.KM + "]";
    } else {
        sgf += "KM[0.5]"; // Default komi for tsumego
    }
    
    // Collect moves by traversing backwards from current node to root
    const moves = [];
    let currentNode = state.kifuReader.node;
    
    // Traverse backwards to collect moves
    while (currentNode && currentNode.move) {
        moves.unshift(currentNode.move); // Add to beginning of array
        currentNode = currentNode.parent;
    }
    
    // Add moves to SGF in correct order
    for (const move of moves) {
        // Handle pass moves
        if (move.pass) {
            sgf += ";" + (move.c === WGo.B ? "B" : "W") + "[]";
        } else {
            // Convert coordinates to SGF format (aa-ss)
            const x = String.fromCharCode(97 + move.x);
            const y = String.fromCharCode(97 + move.y);
            
            // Add move to SGF
            sgf += ";" + (move.c === WGo.B ? "B" : "W") + "[" + x + y + "]";
        }
    }
    
    sgf += ")";
    return sgf;
}

/**
 * Fetches analysis from the server and updates the UI without blocking.
 * This is now a self-contained async function.
 */
async function requestAndDisplayAnalysis() {
    const sgf = getCurrentSgf();
    updateAnalysisUI('Analyzing...', null); // Show loading state, clear old hint
    analysisHintMove = null;
    updateBoard(); // Redraw to remove the old hint marker immediately

    try {
        const { bestMove, score } = await fetchAnalysis(sgf);
        
        // Check if we're still in analysis mode and the board state hasn't changed
        if (state.appMode === 'analyzing' && bestMove) {
            // Verify the SGF hasn't changed since we sent the request
            const currentSgf = getCurrentSgf();
            if (currentSgf === sgf) {
                // If the server call was successful and the board state is still relevant, update the UI with new data.
                analysisHintMove = bestMove;
                updateAnalysisUI(score, bestMove);
                updateBoard(analysisHintMove); // Redraw board with the new hint
            }
        } else if (state.appMode !== 'analyzing') {
            // If we've exited analysis mode, just update the score without the hint
            updateAnalysisUI(score, null);
        }
    } catch (error) {
        console.error("Analysis request failed:", error);
        // Only update UI if we're still in analysis mode
        if (state.appMode === 'analyzing') {
            updateAnalysisUI('Error', null); // Show error state in the UI
        }
    }
}
