// js/ui.js

import { state } from './state.js';
import { getTerm, getDescriptiveRank } from './utils.js';
import { playRandomFailSound, playSuccessSound } from './audio.js';

const boardContainer = document.getElementById('board-container');

export function updateAllUI() {
    // Update app title
    document.getElementById('app-title').textContent = getTerm('appTitle');
    
    // Update UI based on current app mode
    if (state.appMode === 'solving') {
        updateSolvingModeUI();
    } else if (state.appMode === 'analyzing') {
        updateAnalysisModeUI();
    }
}

function updateSolvingModeUI() {
    // Update Rank Display in the text area
    if (state.openedFromLink) {
        // When opened from a link, show the tsumego's rank (and no stddev)
        // Find the current problem in the loaded problems
        const currentProblem = state.allProblems.find(p => p.filename === state.currentProblemFilename);
        if (currentProblem && currentProblem.rank !== undefined) {
            const rank = getDescriptiveRank(currentProblem.rank);
            document.getElementById('text1').textContent = `${rank}`;
        } else {
            // Fallback if rank not found
            document.getElementById('text1').textContent = '...';
        }
    } else {
        // Normal display with rank information
        const rank = getDescriptiveRank(state.userRank);
        const stddev = state.momentum / 4;
        document.getElementById('text1').textContent = `${rank} (±${stddev.toFixed(1)})`;
    }

    // Update "Review" link (button1)
    const prevTsumegoLink = document.getElementById('button1');
    prevTsumegoLink.textContent = getTerm('review');
    
    // Determine which SGF to use for review
    // Always use previous SGF for review, regardless of showCorrectMove state
    const sgfToReview = state.previousSGF;
    const problemFilenameToReview = state.previousProblemFilename;
    
    // Enable review button if we have a previous problem to review
    if (state.previousSGF && state.previousProblemFilename) {
        prevTsumegoLink.classList.remove('disabled');
    } else {
        prevTsumegoLink.classList.add('disabled');
    }
    
    // Update "About" link (button2)
    document.getElementById('button2').textContent = getTerm('about');
    
    // Update "Share" link (button3)
    document.getElementById('button3').textContent = getTerm('share');
}

function updateAnalysisModeUI() {
    // Update text area with analysis information
    document.getElementById('text1').textContent = getTerm('analyzing');

    // Update "Exit" link (button1)
    const exitLink = document.getElementById('button1');
    exitLink.textContent = getTerm('exit');
    exitLink.classList.remove('disabled'); // Always enabled in analysis mode
    
    // Update "Undo" link (button2)
    document.getElementById('button2').textContent = getTerm('undo');
    
    // Update "Share" link (button3)
    document.getElementById('button3').textContent = getTerm('share');
}

export function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)') {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: color, display: 'flex', justifyContent: 'center',
        alignItems: 'center', zIndex: '1000', opacity: '0'
    });

    // If a message is passed, this is a simple error/info overlay.
    if (message) {
        const messageElement = document.createElement('div');
        Object.assign(messageElement.style, {
            color: 'white', fontWeight: 'bold', textAlign: 'center',
            fontSize: `${boardContainer.offsetWidth / 25}px`, padding: '20px'
        });
        messageElement.textContent = message;
        overlay.appendChild(messageElement);
        document.body.appendChild(overlay);

        // Fast fade in/out for messages
        overlay.style.transition = 'opacity 0.3s ease-in-out';
        requestAnimationFrame(() => {
            overlay.style.opacity = '0.9';
        });

        setTimeout(() => {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.remove();
                if (callback) callback();
            }, 300);
        }, duration);

    } else {
        // If no message, this is the "failed problem" overlay. Use original logic.
        if (state.currentPointLoss) {
            const pointLossElement = document.createElement('div');
            Object.assign(pointLossElement.style, {
                color: 'white', fontWeight: 'bold', textAlign: 'center',
                fontSize: `${boardContainer.offsetWidth / 25}px`
            });
            pointLossElement.textContent = state.currentPointLoss;
            overlay.appendChild(pointLossElement);
        }
        document.body.appendChild(overlay);

        setTimeout(() => {
            overlay.style.transition = 'opacity 4s linear';
            requestAnimationFrame(() => {
                overlay.style.opacity = '0.9';
                playRandomFailSound();
            });
        }, 1000);

        setTimeout(() => {
            overlay.remove();
            state.currentPointLoss = null;
            if (callback) callback();
        }, 5000);
    }
}

export function fadeOverlay(duration, callback) {
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'white', zIndex: '1000', opacity: '0'
    });
    document.body.appendChild(overlay);

    // Fade in (0.5s)
    requestAnimationFrame(() => {
        overlay.style.transition = 'opacity 0.5s ease-in-out';
        requestAnimationFrame(() => {
            overlay.style.opacity = '1';
        });
        playSuccessSound();
    });

    // Halfway through, swap content
    setTimeout(() => {
        if (callback) callback();
    }, 500);

    // At the end, fade out (0.5s) and remove
    setTimeout(() => {
        overlay.style.transition = 'opacity 0.5s ease-in-out';
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 500);
    }, 1000);
}


export function updateBoardSize() {
    // Board sizing is now handled by CSS using vi units
    // We still need to update the WGo board dimensions
    if (!state.board) return;

    // Get the computed size from CSS
    const boardElement = document.getElementById('board-container');
    const computedStyle = window.getComputedStyle(boardElement);
    const boardSize = parseInt(computedStyle.width);

    // Update WGo board dimensions
    state.board.setWidth(boardSize);
    state.board.setHeight(boardSize);
}

// No need for toggleAnalysisModeUI anymore since we're using a single div
// and updating button functionality based on the current mode

export function updateAnalysisUI(score, bestMove) {
    // In analysis mode, display the score in the text1 element
    if (state.appMode === 'analyzing') {
        // Format score to 1 decimal place if it's a number
        const formattedScore = typeof score === 'number' ? score.toFixed(1) : score;
        const scoreText = getTerm('score');
        document.getElementById('text1').textContent = `${scoreText}: ${formattedScore}`;
    }
    // Future: could also use bestMove to highlight a button or provide a hint
}
