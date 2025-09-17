// js/ui.js

import { state, setDeferredInstallPrompt } from './state.js';
import { getTerm, getDescriptiveRank } from './utils.js';
import { TOTAL_RANKS } from './constants.js';
import { getCurrentNode } from './board.js';

export function updateCapturesUI() {
    if (state.kifuReader) {
        const captures = state.kifuReader.game.position.capCount;
        document.getElementById('black-captures').textContent = captures.white;
        document.getElementById('white-captures').textContent = captures.black;
        document.getElementById('black-captures-h').textContent = captures.white;
        document.getElementById('white-captures-h').textContent = captures.black;
    }
}

const boardContainer = document.getElementById('board-container');

function showInstallPrompt() {
    const deferredPrompt = state.deferredInstallPrompt;
    if (!deferredPrompt) return;

    const ratio = window.innerWidth / window.innerHeight;
    let displayMethod;

    if (ratio < 0.65) {
        displayMethod = 'notification';
    } else if (ratio > 1.3) {
        displayMethod = 'below-buttons';
    } else {
        displayMethod = 'popup';
    }

    const promptElement = document.createElement('div');
    promptElement.id = 'install-prompt';
    promptElement.innerHTML = `
        <button class="close-btn">×</button>
        <div class="install-message">${getTerm('installPrompt')}</div>
        <button class="install-btn">${getTerm('install')}</button>
    `;

    const installBtn = promptElement.querySelector('.install-btn');
    const closeBtn = promptElement.querySelector('.close-btn');

    if (displayMethod === 'notification') {
        promptElement.classList.add('install-notification');
        document.body.appendChild(promptElement);
        setTimeout(() => promptElement.style.top = '0', 100);
    } else if (displayMethod === 'below-buttons') {
        promptElement.classList.add('install-below');
        document.getElementById('info-container').appendChild(promptElement);
    } else { // popup
        promptElement.classList.add('install-popup');
        document.body.appendChild(promptElement);
    }

    promptElement.querySelector('.install-btn').addEventListener('click', async () => {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        console.log(`User response to install prompt: ${outcome}`);
        setDeferredInstallPrompt(null);
        promptElement.remove();
    });

    promptElement.querySelector('.close-btn').addEventListener('click', () => {
        setDeferredInstallPrompt(null);
        promptElement.remove();
    });
}

function checkAndShowInstallPrompt() {
    const historySize = state.problemHistory.length;
    const isSolvingMode = state.appMode === 'solving';
    const isPromptVisible = !!document.getElementById('install-prompt');

    if (state.deferredInstallPrompt && !isPromptVisible && isSolvingMode && [2, 19, 190].includes(historySize)) {
        showInstallPrompt();
    }
}

export function updateAllUI() {
    checkAndShowInstallPrompt();
    updateCapturesUI();
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
    
    const text1Elements = [document.getElementById('text1'), document.getElementById('text1-h')];

    // Update Rank Display in the text area
    if (state.appMode === 'linked') {
        // In view mode, don't show any rank
        text1Elements.forEach(element => element.textContent = '');
        text1Elements.forEach(element => {
            element.classList.remove('progress-bar-wrapper');
            element.style.removeProperty('--progress');
        });
    } else {
        // Normal display with user rank information
        const rank = getDescriptiveRank(state.userRank);
        text1Elements.forEach(element => element.textContent = `${rank}`);

        // Add progress bar
        const rating = state.userRank;
        const rankValue = rating * TOTAL_RANKS;
        const currentRankIndex = Math.floor(rankValue);
        const progress = (rankValue - currentRankIndex) * 100;

        text1Elements.forEach(element => {
            element.classList.add('progress-bar-wrapper');
            element.style.setProperty('--progress', `${progress}%`);
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
    document.getElementById('button3').textContent = getTerm('stats');
}

function updateAnalysisModeUI() {
    // Update Title Display
    document.getElementById('title').textContent = getTerm('reviewMode');
    document.getElementById('title-h').textContent = getTerm('reviewMode');
    
    // Update text area with analysis information

    const text1Elements = [document.getElementById('text1'), document.getElementById('text1-h')];
    if (state.kifuReader && state.kifuReader.node === state.initialAnalysisNode) {
        // At starting position, show rank
        const problem = state.problemHistory[1];
        if (problem && problem.rating !== undefined) {
            const rank = getDescriptiveRank(problem.rating);
            text1Elements.forEach(element => element.textContent = rank);
        } else {
            text1Elements.forEach(element => element.textContent = '');
        }
    } else {
        // For other positions, show analyzing initially
        text1Elements.forEach(element => element.textContent = getTerm('analyzing'));
    }
    text1Elements.forEach(element => {
        element.classList.remove('progress-bar-wrapper');
        element.style.removeProperty('--progress');
    });

    // Update "Exit" link (button1)
    const exitLink = document.getElementById('button1');
    exitLink.textContent = getTerm('exit');
    exitLink.classList.remove('disabled'); // Always enabled in analysis mode
    
    // Update "Undo" link (button2)
    const undoLink = document.getElementById('button2');
    undoLink.textContent = getTerm('undo');
    if (state.kifuReader && state.kifuReader.node.parent && state.kifuReader.node !== state.initialAnalysisNode) {
        undoLink.classList.remove('disabled');
    } else {
        undoLink.classList.add('disabled');
    }
    
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
    // In analysis mode, display rank at starting position, score otherwise
    if (state.appMode === 'analyzing') {
        if (state.kifuReader && state.kifuReader.node === state.initialAnalysisNode) {
            // At starting position, show rank
            const problem = state.problemHistory[1];
            if (problem && problem.rating !== undefined) {
                const rank = getDescriptiveRank(problem.rating);
                document.getElementById('text1').textContent = rank;
                document.getElementById('text1-h').textContent = rank;
            }
        } else {
            // For other positions, show score
            const formattedScore = typeof score === 'number' ? score.toFixed(1) : score;
            const scoreText = getTerm('score');
            const currentNode = getCurrentNode();
            let isUnreliable = false;
            if (bestMove) {
                const child = currentNode.children.find(c => c.move && c.move.x === bestMove.x && c.move.y === bestMove.y);
                if (child && child.comment && child.comment.includes('100%/')) {
                    isUnreliable = true;
                }
            }
            const displayText = `${scoreText}: ${formattedScore}`;
            if (isUnreliable) {
                const warningMsg = getTerm('unreliableWarning');
                const html = `<span class=\"unreliable\" onclick=\"alert('${warningMsg}')\" style=\"user-select: none; -webkit-user-select: none;\">${displayText}</span>`;
                document.getElementById('text1').innerHTML = html;
                document.getElementById('text1-h').innerHTML = html;
            } else {
                document.getElementById('text1').textContent = displayText;
                document.getElementById('text1-h').textContent = displayText;
            }
        }
        // Prevent text selection on Android to treat as button
        const text1 = document.getElementById('text1');
        const text1h = document.getElementById('text1-h');
        text1.style.userSelect = 'none';
        text1.style.webkitUserSelect = 'none';
        text1h.style.userSelect = 'none';
        text1h.style.webkitUserSelect = 'none';
    }
    // Future: could also use bestMove to highlight a button or provide a hint
}