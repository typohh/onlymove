// js/ui.js

import { state } from './state.js';
import { getTerm, getDescriptiveRank } from './utils.js';
import { TOTAL_RANKS } from './constants.js';

const boardContainer = document.getElementById('board-container');

export function updateAllUI() {
    // Update UI based on current app mode
    if (state.appMode === 'solving' || state.appMode === 'linked') {
        updateSolvingModeUI();
    } else if (state.appMode === 'analyzing') {
        updateAnalysisModeUI();
    }
}

function updateSolvingModeUI() {
    // Update Title Display
    if (state.appMode === 'linked') {
        document.getElementById('title').textContent = getTerm('viewMode');
        document.getElementById('title-h').textContent = getTerm('viewMode');
    } else {
        document.getElementById('title').textContent = getTerm('tsumegoMode');
        document.getElementById('title-h').textContent = getTerm('tsumegoMode');
    }
    
    // Update Rank Display in the text area
    if (state.appMode === 'linked') {
        // When opened from a link, show the tsumego's rank from the history object.
        const currentProblem = state.problemHistory[0];
        if (currentProblem && currentProblem.rating !== undefined) {
            const rank = getDescriptiveRank(currentProblem.rating);
            document.getElementById('text1').textContent = `${rank}`;
            document.getElementById('text1-h').textContent = `${rank}`;
        } else {
            // Fallback if rank not found
            document.getElementById('text1').textContent = '...';
            document.getElementById('text1-h').textContent = '...';
        }
    } else {
        // Normal display with tsumego rank information
        const currentProblem = state.problemHistory[0];
        if (currentProblem && currentProblem.rating !== undefined) {
            const rank = getDescriptiveRank(currentProblem.rating);
            document.getElementById('text1').textContent = `${rank}`;
            document.getElementById('text1-h').textContent = `${rank}`;
        } else {
            // Fallback if rank not found
            document.getElementById('text1').textContent = '...';
            document.getElementById('text1-h').textContent = '...';
        }

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

function createOverlayContent(stats, message, textColor) {
    const container = document.createElement('div');
    Object.assign(container.style, {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: textColor,
        fontWeight: 'bold',
        textAlign: 'center',
    });

    if (stats && stats.rankChange) {
        const rankChange = stats.rankChange;
        if (rankChange !== 0) {
            const rankChangeInUnits = rankChange * TOTAL_RANKS;
            const rankChangeElement = document.createElement('div');
            Object.assign(rankChangeElement.style, {
                fontSize: `${boardContainer.offsetWidth / 20}px`,
                marginBottom: '20px',
            });
            const sign = rankChangeInUnits > 0 ? '+' : '';
            const formattedChange = `${sign}${rankChangeInUnits.toFixed(3)}`;
            rankChangeElement.textContent = getTerm('rankChangeFormat').replace('{change}', formattedChange);
            container.appendChild(rankChangeElement);
        }

        if (stats.rankChangeDirection === 'up' || stats.rankChangeDirection === 'down') {
            const rankChangeMessageElement = document.createElement('div');
            Object.assign(rankChangeMessageElement.style, {
                fontSize: `${boardContainer.offsetWidth / 25}px`,
                marginBottom: '20px',
            });
            const newRank = getDescriptiveRank(state.userRank);
            const term = stats.rankChangeDirection === 'up' ? 'rankUpTo' : 'rankDownTo';
            rankChangeMessageElement.textContent = getTerm(term).replace('{rank}', newRank);
            container.appendChild(rankChangeMessageElement);
        }
    }

    if (message) {
        const messageElement = document.createElement('div');
        Object.assign(messageElement.style, {
            fontSize: `${boardContainer.offsetWidth / 30}px`,
            maxWidth: '80%',
        });
        messageElement.textContent = message;
        container.appendChild(messageElement);
    }

    return container;
}

export function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)', stats = null) {
    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: color, display: 'flex', justifyContent: 'center',
        alignItems: 'center', zIndex: '1000', opacity: '0'
    });

    if (stats) { // This is the new "failed problem" overlay
        const content = createOverlayContent(stats, message, 'white');
        overlay.appendChild(content);
        document.body.appendChild(overlay);

        // Fade in over duration, then remove
        setTimeout(() => {
            overlay.style.transition = `opacity ${duration}ms ease-out`;
            overlay.style.opacity = '0.9';
        }, 0);

        setTimeout(() => {
            overlay.remove();
            if (callback) callback();
        }, duration);

    } else { // This is the simple message overlay
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

export function fadeOverlay(duration, callback, stats = null) {
    const overlay = document.createElement('div');
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'white', zIndex: '1000', opacity: '0',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    });

    if (stats) {
        const content = createOverlayContent(stats, '', 'black');
        overlay.appendChild(content);
    }

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
    const rankValue = rating * TOTAL_RANKS;
    const currentRankIndex = Math.floor(rankValue);
    const progress = (rankValue - currentRankIndex) * 100;
    
    const stddev = state.momentum / 4;
    const stats = state.tsumegoStats;
    const totalTsumegos = stats.totalOnFail + stats.totalOnInBetween + stats.totalOnSuccess;
    const shortTermRecollectionValue = stats.totalOnFail > 0 ? Math.round(100 - stats.failOnFail / stats.totalOnFail * 100) : null;
    const longTermRecollectionValue = stats.totalOnInBetween > 0 ? Math.round(100 - stats.failOnInBetween / stats.totalOnInBetween * 100) : null;

    const shortTermRecollectionText = shortTermRecollectionValue !== null ? `${shortTermRecollectionValue}%` : '-';
    const longTermRecollectionText = longTermRecollectionValue !== null ? `${longTermRecollectionValue}%` : '-';

    // Display the information
    const userRank = getDescriptiveRank(state.userRank);
    const detailText = `${getTerm('userRank')}: ${userRank}
` +
                       `${getTerm('onTheWayToNextRank')}: ${Math.floor(progress)}%
` +
                       `${getTerm('standardDeviation')}: ${stddev.toFixed(1)} ranks
` +
                       `${getTerm('totalTsumegos')}: ${totalTsumegos}
` +
                       `${getTerm('shortTermRecollection')}: ${shortTermRecollectionText}
` +
                       `${getTerm('longTermRecollection')}: ${longTermRecollectionText}`;
    
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