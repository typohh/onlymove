// js/ui.js

import { state, setDeferredInstallPrompt } from './state.js';
import { getTerm, getDescriptiveRank } from './i18n.js';
import { TOTAL_RANKS, APP_MODES } from './constants.js';
import { getCurrentNode } from './board.js';
import { getReviewCandidates } from './model.js';

const formatScore = (score) => score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1);

export function updateCapturesUI() {
    if (state.kifuReader) {
        const captures = state.kifuReader.game.position.capCount;
        const komi = 6.5;
        const blackCaptures = captures.black;
        const whiteCaptures = captures.white;
        const score = komi + whiteCaptures - blackCaptures;
        const formattedScore = formatScore(score);
        document.getElementById('capture-score').textContent = formattedScore;
        document.getElementById('capture-score-h').textContent = formattedScore;
    }
}

const boardContainer = document.getElementById('board-container');

const showInstallPrompt = () => {
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
        <button class="btn install-btn">${getTerm('install')}</button>
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

const checkAndShowInstallPrompt = () => {
    const historySize = state.answerHistory.length;
    const isSolvingMode = state.appMode === APP_MODES.SOLVING;
    const isPromptVisible = !!document.getElementById('install-prompt');

    if (state.deferredInstallPrompt && !isPromptVisible && isSolvingMode && [2, 19, 190].includes(historySize)) {
        showInstallPrompt();
    }
}

export function updateAllUI() {
    checkAndShowInstallPrompt();
    updateCapturesUI();
    // Update UI based on current app mode
    if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
        updateSolvingModeUI();
    } else if (state.appMode === APP_MODES.ANALYZING) {
        updateAnalysisModeUI();
    }
}

const updateSolvingModeUI = () => {
    if (state.appMode === APP_MODES.LINKED) {
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
    const reviewCandidates = getReviewCandidates();
    const prevTsumegoLink = document.getElementById('button1');
    if (reviewCandidates.length > 0) {
        prevTsumegoLink.textContent = `${getTerm('review')} (${reviewCandidates.length})`;
    } else {
        prevTsumegoLink.textContent = getTerm('review');
    }
    
    // Enable review button if there is a problem to review in the history
    if (state.problemHistory.length > 1 && state.problemHistory[1].sgf) {
        prevTsumegoLink.classList.remove('disabled');
    } else {
        prevTsumegoLink.classList.add('disabled');
    }
    
    // Update "About" link (button2)
    const aboutLink = document.getElementById('button2');
    aboutLink.textContent = getTerm('about');
    aboutLink.classList.remove('disabled'); // Ensure enabled in solving mode
    
    // Update "Share" link (button3)
    document.getElementById('button3').textContent = getTerm('stats');

    // Hide button4 in solving mode
    document.getElementById('button4').style.display = 'none';
}

const updateAnalysisModeUI = () => {
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

    // Show and configure button4 ('Ignore')
    const ignoreButton = document.getElementById('button4');
    const problem = state.problemHistory[1];

    if (problem && problem.linked) {
        ignoreButton.style.display = 'none';
    } else if (problem) {
        ignoreButton.style.display = 'inline-block';
        ignoreButton.textContent = getTerm('ignore');

        const lastAnswer = state.answerHistory[state.answerHistory.length - 1];
        const wasFailure = lastAnswer && lastAnswer.problemId === problem.filename.replace('.sgf', '') && !lastAnswer.correct;
        const isRepeat = problem.isRepeat;

        if (isRepeat || wasFailure) {
            ignoreButton.classList.remove('disabled');
        } else {
            ignoreButton.classList.add('disabled');
        }
    } else {
        ignoreButton.style.display = 'none';
    }
}

import { displayOverlay, fadeOverlay } from './overlays.js';




// Function to show capture details when user clicks on capture score
export function showCaptureDetails() {
    if (!state.kifuReader) return;
    const captures = state.kifuReader.game.position.capCount;
    const komi = 6.5;
    const blackCaptures = captures.black;
    const whiteCaptures = captures.white;
    const score = komi + whiteCaptures - blackCaptures;

    const detailText = `${getTerm('komi')}: ${komi}
` +
                       `${getTerm('blackCaptures')}: ${blackCaptures}
` +
                       `${getTerm('whiteCaptures')}: ${whiteCaptures}
` +
                       `${getTerm('whiteAdjustment')}: ${formatScore(score)}`;

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
    if (state.appMode === APP_MODES.ANALYZING) {
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
            let displayText;
            if (formattedScore === getTerm('analyzing')) {
                displayText = formattedScore;
            } else {
                displayText = `${scoreText}: ${formattedScore}`;
            }
            if (isUnreliable) {
                const warningMsg = getTerm('unreliableWarning');
                const html = `<span class="unreliable" onclick="alert('${warningMsg}')" role="button" tabindex="0">${displayText}</span>`;
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