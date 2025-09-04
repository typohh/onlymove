// js/ui.js

import { state } from './state.js';
import { getTerm, getDescriptiveRank } from './utils.js';

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
    // Update Title Display
    document.getElementById('title').textContent = getTerm('tsumegoMode');
    document.getElementById('title-h').textContent = getTerm('tsumegoMode');
    
    // Update Rank Display in the text area
    if (state.openedFromLink) {
        // When opened from a link, show the tsumego's rank (and no stddev)
        // Find the current problem in the loaded problems
        const currentProblem = state.allProblems.find(p => p.filename === state.problemHistory[0].filename);
        if (currentProblem && currentProblem.rank !== undefined) {
            const rank = getDescriptiveRank(currentProblem.rank);
            document.getElementById('text1').textContent = `${rank}`;
            document.getElementById('text1-h').textContent = `${rank}`;
        } else {
            // Fallback if rank not found
            document.getElementById('text1').textContent = '...';
            document.getElementById('text1-h').textContent = '...';
        }
    } else {
        // Normal display with rank information
        const rank = getDescriptiveRank(state.userRank);
        const stddev = state.momentum / 4;
        document.getElementById('text1').textContent = `${rank} (±${stddev.toFixed(1)})`;
        document.getElementById('text1-h').textContent = `${rank} (±${stddev.toFixed(1)})`;
        
        // Set styles for click (event added in main.js)
        const text1Elements = [document.getElementById('text1'), document.getElementById('text1-h')];
        text1Elements.forEach(element => {
            element.style.cursor = 'pointer';
            element.title = 'Click for detailed stats';
            element.style.userSelect = 'none';
            element.style.webkitUserSelect = 'none';
            element.style.MozUserSelect = 'none';
        });
    }

    // Update "Review" link (button1)
    const prevTsumegoLink = document.getElementById('button1');
    prevTsumegoLink.textContent = getTerm('review');
    
    // Enable review button if there is a problem to review in the history
    if (state.problemHistory.length > 1 && state.problemHistory[1].sgf) {
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
    // Update Title Display
    document.getElementById('title').textContent = getTerm('reviewMode');
    document.getElementById('title-h').textContent = getTerm('reviewMode');
    
    // Update text area with analysis information
    document.getElementById('text1').textContent = getTerm('analyzing');
    document.getElementById('text1-h').textContent = getTerm('analyzing');

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

    // Check if this is a failed problem overlay (has currentPointLoss)
    const isFailedOverlay = state.currentPointLoss !== null;

    if (isFailedOverlay) {
        // Failed problem overlay
        const container = document.createElement('div');
        Object.assign(container.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
        });

        if (state.currentPointLoss) {
            const pointLossElement = document.createElement('div');
            Object.assign(pointLossElement.style, {
                color: 'white', fontWeight: 'bold', textAlign: 'center',
                fontSize: `${boardContainer.offsetWidth / 25}px`,
                marginBottom: message ? '20px' : '0'
            });
            pointLossElement.textContent = state.currentPointLoss;
            container.appendChild(pointLossElement);
        }

        if (message) {
            const messageElement = document.createElement('div');
            Object.assign(messageElement.style, {
                color: 'white', fontWeight: 'bold', textAlign: 'center',
                fontSize: `${boardContainer.offsetWidth / 30}px`,
                maxWidth: '80%'
            });
            messageElement.textContent = message;
            container.appendChild(messageElement);
        }

        overlay.appendChild(container);
        document.body.appendChild(overlay);

        // Fade in over duration, then remove
        setTimeout(() => {
            overlay.style.transition = `opacity ${duration}ms ease-out`;
            overlay.style.opacity = '0.9';
        }, 0);

        setTimeout(() => {
            overlay.remove();
            state.currentPointLoss = null;
            if (callback) callback();
        }, duration);
    } else {
        // Simple message overlay
        if (message) {
            const messageElement = document.createElement('div');
            Object.assign(messageElement.style, {
                color: 'white', fontWeight: 'bold', textAlign: 'center',
                fontSize: `${boardContainer.offsetWidth / 25}px`, padding: '20px'
            });
            messageElement.textContent = message;
            overlay.appendChild(messageElement);
        }
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


// Function to show detailed stats when user clicks on rank
export function showDetailedStats() {
    // Calculate progress to next rank
    const rating = state.userRank;
    const totalRanks = 29;
    const rankValue = rating * totalRanks;
    const currentRankIndex = Math.floor(rankValue);
    const progress = (rankValue - currentRankIndex) * 100;
    
    const stats = state.tsumegoStats;
    const totalTsumegos = stats.totalOnFail + stats.totalOnInBetween + stats.totalOnSuccess;
    const shortTermMemory = stats.totalOnFail > 0 ? Math.round(100 - stats.failOnFail / stats.totalOnFail * 100) : 100;
    const longTermMemory = stats.totalOnInBetween > 0 ? Math.round(100 - stats.failOnInBetween / stats.totalOnInBetween * 100) : 100;

    // Display the information
    const detailText = `On the way to next rank: ${progress.toFixed(0)}%
Total Tsumegos: ${totalTsumegos}
Short Term Memory: ${shortTermMemory}%
Long Term Memory: ${longTermMemory}%`;
    
    // Show in an alert for now (can be improved with a better UI later)
    alert(detailText);
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
        document.getElementById('text1-h').textContent = `${scoreText}: ${formattedScore}`;
    }
    // Future: could also use bestMove to highlight a button or provide a hint
}
