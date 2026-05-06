// js/ui.js

import { state, setDeferredInstallPrompt } from './state.js';
import { getTerm, getDescriptiveRank } from './i18n.js';

/**
 * Returns true if the app is already running as an installed PWA.
 */
const isInstalled = () =>
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;
import { TOTAL_RANKS, APP_MODES } from './constants.js';
import { getCurrentNode } from './board.js';

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

/**
 * Shows an iOS-style modal with step-by-step instructions to add to Home Screen.
 */
const showIosInstallModal = () => {
    const overlay = document.createElement('div');
    overlay.id = 'ios-install-overlay';

    const modal = document.createElement('div');
    modal.id = 'ios-install-modal';

    const title = document.createElement('div');
    title.id = 'ios-install-title';
    title.textContent = getTerm('installIosTitle');

    const instructions = document.createElement('div');
    instructions.id = 'ios-install-instructions';
    // Replace literal \n with line breaks
    getTerm('installIosInstructions').split('\n').forEach((line, i, arr) => {
        instructions.appendChild(document.createTextNode(line));
        if (i < arr.length - 1) instructions.appendChild(document.createElement('br'));
    });

    const closeBtn = document.createElement('button');
    closeBtn.id = 'ios-install-close';
    closeBtn.className = 'btn';
    closeBtn.textContent = getTerm('installIosClose');
    closeBtn.addEventListener('click', () => overlay.remove());

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.remove();
    });

    modal.appendChild(title);
    modal.appendChild(instructions);
    modal.appendChild(closeBtn);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
};

/**
 * Handles a click on the Install button.
 * - If a deferred install prompt exists (Chrome/Android), triggers it.
 * - Otherwise shows an iOS instructions modal.
 */
export const handleInstallClick = async () => {
    if (isInstalled()) return;

    if (state.deferredInstallPrompt) {
        const prompt = state.deferredInstallPrompt;
        prompt.prompt();
        const { outcome } = await prompt.userChoice;
        console.log(`User response to install prompt: ${outcome}`);
        setDeferredInstallPrompt(null);
        // Refresh button state
        updateInstallButton();
    } else {
        showIosInstallModal();
    }
};

/**
 * Updates the install button (button4) text and disabled state in solving/linked mode.
 */
const updateInstallButton = () => {
    const btn = document.getElementById('button4');
    btn.style.display = 'inline-block';
    if (isInstalled()) {
        btn.textContent = getTerm('installed');
        btn.classList.add('disabled');
    } else {
        btn.textContent = getTerm('install');
        btn.classList.remove('disabled');
    }
};

export function updateAllUI() {
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
        const currentProblem = state.problemHistory[0];
        const titleKey = (currentProblem && currentProblem.isRepeat) ? 'recallMode' : 'tsumegoMode';
        document.getElementById('title').textContent = getTerm(titleKey);
        document.getElementById('title-h').textContent = getTerm(titleKey);
    }

    const text1Elements = [document.getElementById('text1'), document.getElementById('text1-h')];

    // Update Rank Display in the text area
    if (state.appMode === APP_MODES.LINKED) {
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

    // Update button1 (Review/Exit)
    const prevTsumegoLink = document.getElementById('button1');
    if (state.appMode === APP_MODES.LINKED) {
        // In view mode, button1 exits back to challenge mode
        prevTsumegoLink.textContent = getTerm('exit');
        prevTsumegoLink.classList.remove('disabled');
    } else {
        // In solving mode, button1 is review
        prevTsumegoLink.textContent = getTerm('review');

        // Enable review button if there is a problem to review in the history
        if (state.problemHistory.length > 1 && state.problemHistory[1].sgf) {
            prevTsumegoLink.classList.remove('disabled');
        } else {
            prevTsumegoLink.classList.add('disabled');
        }
    }

    // Update "About" link (button2)
    const aboutLink = document.getElementById('button2');
    aboutLink.textContent = getTerm('about');
    aboutLink.classList.remove('disabled'); // Ensure enabled in solving mode

    // Update "Share" link (button3)
    document.getElementById('button3').textContent = getTerm('stats');

    // Show button4 as Install button
    updateInstallButton();
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
                if (child && child._origin === 'api') {
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
                for (const id of ['text1', 'text1-h']) {
                    const el = document.getElementById(id);
                    el.textContent = displayText;
                    const span = document.createElement('span');
                    span.className = 'unreliable';
                    span.setAttribute('role', 'button');
                    span.setAttribute('tabindex', '0');
                    span.textContent = displayText;
                    span.addEventListener('click', () => alert(warningMsg));
                    el.textContent = '';
                    el.appendChild(span);
                }
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