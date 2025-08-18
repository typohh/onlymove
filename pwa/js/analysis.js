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

    state.analysisKifuReader = initializeKifuFromSGF(sgfToAnalyze);
    
    initializeBoard(state.analysisKifuReader.kifu.size);
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
    if (!state.acceptClicks || !state.analysisKifuReader) return;

    const game = state.analysisKifuReader.game;
    
    // Check if the move is valid
    if (!game.isValid(x, y)) return;

    // Play the move on the game state
    const result = game.play(x, y, game.turn);
    
    // If the move was successful, update the board
    if (result !== 1 && result !== 2 && result !== 3 && result !== 4) {
        playStoneSound();
        // Update the board to show the new stone
        updateBoard();
        
        // Request analysis in the background without blocking the UI
        requestAndDisplayAnalysis();
    }
}

/**
 * Handles the "Undo" button click.
 */
export function undoLastMove() {
    if (!state.acceptClicks || !state.analysisKifuReader) return;
    
    const game = state.analysisKifuReader.game;
    
    // Check if we can undo (if there are moves to undo)
    if (game.stack.length > 1) {
        game.popPosition();
        updateBoard();
        
        // Request analysis for the new position in the background
        requestAndDisplayAnalysis();
    }
}

/**
 * Converts the current board state to an SGF string for the server.
 */
function getCurrentSgf() {
    return state.analysisKifuReader.kifu.toSgf();
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
        
        // If the server call was successful, update the UI with new data.
        analysisHintMove = bestMove;
        updateAnalysisUI(score, bestMove);
        updateBoard(analysisHintMove); // Redraw board with the new hint
    } catch (error) {
        console.error("Analysis request failed:", error);
        updateAnalysisUI('Error', null); // Show error state in the UI
    }
}
