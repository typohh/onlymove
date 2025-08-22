// js/ui.js

import { state } from './state.js';
import { getTerm, getDescriptiveRank } from './utils.js';
import { playRandomFailSound, playSuccessSound } from './audio.js';

const boardContainer = document.getElementById('board-container');

export function updateAllUI() {
    // Update app title
    document.getElementById('app-title').textContent = getTerm('appTitle');
    
    // Update Rank Display
    const rank = getDescriptiveRank(state.userRank);
    const stddev = state.momentum / 4;
    document.getElementById('rank-display').textContent = `${rank} (±${stddev.toFixed(1)})`;

    // Update "Review" link
    const prevTsumegoLink = document.getElementById('prev-tsumego');
    prevTsumegoLink.textContent = getTerm('review');
    
    // Determine which SGF to use for review
    // If we're showing the correct move (hint mode after failing), use current SGF
    // Otherwise, use previous SGF
    const sgfToReview = state.showCorrectMove ? state.currentSGF : state.previousSGF;
    const problemFilenameToReview = state.showCorrectMove ? state.currentProblemFilename : state.previousProblemFilename;
    
    if (sgfToReview && problemFilenameToReview) {
        prevTsumegoLink.classList.remove('disabled');
    } else {
        prevTsumegoLink.classList.add('disabled');
    }
    
    // Update "About" link
    document.getElementById('komi').textContent = getTerm('about');
    
    // Update analysis UI elements
    document.getElementById('analysis-exit').textContent = getTerm('exit');
    document.getElementById('analysis-undo').textContent = getTerm('undo');
    
    // Update initial analysis score text
    updateAnalysisUI('N/A', null);
}

export function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)') {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: color, display: 'flex', justifyContent: 'center',
        alignItems: 'center', zIndex: '1000', opacity: '0'
    });

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
    if (!state.board) return;

    const infoContainer = document.getElementById('info-container');
    const analysisContainer = document.getElementById('analysis-container');
    const h = window.innerHeight;
    const controlSize = 1 / 13; // For both info and analysis containers

    const availableHeight = h / (1 + controlSize);
    const boardSize = Math.min(window.innerWidth, availableHeight);

    boardContainer.style.width = boardContainer.style.height = boardSize + 'px';
    state.board.setWidth(boardSize);
    state.board.setHeight(boardSize);

    const controlHeight = (boardSize * controlSize);
    const controlHeightPx = controlHeight + 'px';
    const fontSize = boardSize / 20 + 'px'; // Changed from 25 to 20

    // Calculate proportional padding
    // Original design had 16px padding with 12px overlap, net 4px visibility
    // We maintain the same proportional relationship
    const overlapRatio = 12 / 16; // 0.75
    const paddingRatio = 16 / 12; // 1.333...

    // Style info container
    infoContainer.style.width = boardSize + 'px';
    infoContainer.style.height = controlHeightPx;
    infoContainer.style.marginTop = '-' + controlHeightPx; // Dynamic overlap
    infoContainer.style.paddingTop = (controlHeight * paddingRatio) + 'px'; // Proportional padding
    document.getElementById('rank-display').style.fontSize = fontSize;
    document.getElementById('komi').style.fontSize = fontSize;
    document.getElementById('prev-tsumego').style.fontSize = fontSize;

    // Style analysis container
    analysisContainer.style.width = boardSize + 'px';
    analysisContainer.style.height = controlHeightPx;
    analysisContainer.style.marginTop = '-' + controlHeightPx; // Dynamic overlap
    analysisContainer.style.paddingTop = (controlHeight * paddingRatio) + 'px'; // Proportional padding
    analysisContainer.querySelectorAll('button, div').forEach(el => {
        el.style.fontSize = fontSize;
    });
}

export function toggleAnalysisModeUI(show) {
    document.getElementById('info-container').style.display = show ? 'none' : 'flex';
    document.getElementById('analysis-container').style.display = show ? 'flex' : 'none';
}

export function updateAnalysisUI(score, bestMove) {
    const scoreEl = document.getElementById('analysis-score');
    // Format score to 1 decimal place if it's a number
    const formattedScore = typeof score === 'number' ? score.toFixed(1) : score;
    const scoreText = getTerm('score');
    scoreEl.textContent = `${scoreText}: ${formattedScore}`;
    // Future: could also use bestMove to highlight a button or provide a hint
}
