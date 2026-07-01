// js/ui.js

import { state, setDeferredInstallPrompt, SUPPORTED_BOARD_SIZES } from './state.js';
import { tsumegoIdOf } from './sgf.js';
import {
    supportsNotifications,
    notificationPermission,
    getCurrentSubscription,
    enableNotifications,
} from './notifications.js';
import { getTerm, getDescriptiveRank } from './i18n.js';

/**
 * Returns true if the app is already running as an installed PWA.
 */
const isInstalled = () =>
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;
import { TOTAL_RANKS, APP_MODES, HISTORY_BACKFILL_DONE_KEY } from './constants.js';
import { getCurrentNode } from './board.js';
import { syncMotifBubble } from './motif-bubble.js';

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
    showModal({
        build: (modal, close) => {
            modal.appendChild(modalTitle(getTerm('installIosTitle')));

            const instructions = document.createElement('div');
            instructions.id = 'ios-install-instructions';
            // Replace literal \n with line breaks
            getTerm('installIosInstructions').split('\n').forEach((line, i, arr) => {
                instructions.appendChild(document.createTextNode(line));
                if (i < arr.length - 1) instructions.appendChild(document.createElement('br'));
            });
            modal.appendChild(instructions);

            // Keeps the dedicated #ios-install-close styling rather than .btn-inline.
            const closeBtn = document.createElement('button');
            closeBtn.id = 'ios-install-close';
            closeBtn.className = 'btn';
            closeBtn.textContent = getTerm('installIosClose');
            closeBtn.addEventListener('click', close);
            modal.appendChild(closeBtn);
        },
    });
};

// State-machine cache for button4. refreshNotifState() refills it async;
// updateInstallButton paints from this snapshot synchronously.
let notifState = { installed: false, permission: 'default', subscribed: false };

export async function refreshNotifState() {
    notifState.installed = isInstalled();
    notifState.permission = notificationPermission();
    notifState.subscribed = supportsNotifications() ? !!(await getCurrentSubscription()) : false;
}

/**
 * Single click handler for button4 in solving/linked mode. Advances through
 * Install → Notify → Reminders, using cached notifState to decide which
 * action to take. After each step, refreshes the cache and repaints.
 */
export const handleInstallClick = async () => {
    if (!notifState.installed) {
        if (state.deferredInstallPrompt) {
            const prompt = state.deferredInstallPrompt;
            prompt.prompt();
            const { outcome } = await prompt.userChoice;
            console.log(`User response to install prompt: ${outcome}`);
            setDeferredInstallPrompt(null);
        } else {
            // No native prompt (Safari/iOS, or Chrome before the event fires).
            showIosInstallModal();
        }
        await refreshNotifState();
        updateInstallButton();
        return;
    }

    // Installed. The next gesture either subscribes to notifications or
    // (when already subscribed / blocked) is a no-op — the button is
    // visually disabled in those terminal states.
    if (notifState.permission === 'denied') return;
    if (notifState.permission === 'granted' && notifState.subscribed) return;

    try {
        await enableNotifications();
    } catch (err) {
        console.error('enableNotifications failed:', err);
    }
    await refreshNotifState();
    updateInstallButton();
};

/**
 * Paints button4 from the cached notifState. Single source of truth for the
 * label and disabled state across Install / Notify / Reminders / Blocked.
 */
const updateInstallButton = () => {
    const btn = document.getElementById('button4');
    btn.style.display = 'inline-block';
    btn.classList.remove('disabled');

    if (!notifState.installed) {
        btn.textContent = getTerm('install');
        return;
    }
    if (notifState.permission === 'denied') {
        btn.textContent = getTerm('remindersBlocked');
        btn.classList.add('disabled');
        return;
    }
    if (notifState.permission === 'granted' && notifState.subscribed) {
        btn.textContent = getTerm('reminders');
        btn.classList.add('disabled');
        return;
    }
    btn.textContent = getTerm('notify');
};

export function updateAllUI() {
    updateCapturesUI();
    // Update UI based on current app mode
    if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
        updateSolvingModeUI();
    } else if (state.appMode === APP_MODES.ANALYZING) {
        updateAnalysisModeUI();
    }
    // Visibility derives from current state (mode + whether we're at the
    // initial analysis node), so refresh it on every UI update.
    syncMotifBubble();
}

const updateSolvingModeUI = () => {
    if (state.appMode === APP_MODES.LINKED) {
        document.getElementById('title').textContent = getTerm('viewMode');
        document.getElementById('title-h').textContent = getTerm('viewMode');
    } else {
        const currentProblem = state.currentProblem;
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
            element.style.cursor = '';
            element.removeAttribute('title');
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
            // Tappable: opens the server rank-translation modal (main.js).
            element.style.cursor = 'pointer';
            element.title = getTerm('rankTranslationTitle');
        });


    }

    // Update button1 (Review/Exit)
    const prevTsumegoLink = document.getElementById('button1');
    prevTsumegoLink.style.display = '';
    if (state.appMode === APP_MODES.LINKED) {
        // In view mode, button1 exits back to challenge mode
        prevTsumegoLink.textContent = getTerm('exit');
        prevTsumegoLink.classList.remove('disabled');
    } else {
        // In solving mode, button1 is review
        prevTsumegoLink.textContent = getTerm('review');

        // Enable review button if there is a previous problem to review and we
        // are not in the middle of a post-solve load.
        if (state.previousProblem && state.previousProblem.sgf && state.acceptClicks) {
            prevTsumegoLink.classList.remove('disabled');
        } else {
            prevTsumegoLink.classList.add('disabled');
        }
    }

    // About (button2) and Stats (button3) live in the hamburger menu now;
    // their DOM nodes stay so the menu items can dispatch clicks through them.
    document.getElementById('button2').style.display = 'none';
    document.getElementById('button3').style.display = 'none';

    // Show button4 as Install button
    updateInstallButton();

    updateMenuItems();
}

const updateAnalysisModeUI = () => {
    // Update Title Display
    document.getElementById('title').textContent = getTerm('reviewMode');
    document.getElementById('title-h').textContent = getTerm('reviewMode');

    // Update text area with analysis information

    const text1Elements = [document.getElementById('text1'), document.getElementById('text1-h')];
    if (state.kifuReader && state.kifuReader.node === state.initialAnalysisNode) {
        // At starting position, show rank
        const problem = state.previousProblem;
        if (problem && problem.rank !== undefined) {
            const rank = getDescriptiveRank(problem.rank);
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
    exitLink.style.display = '';
    exitLink.textContent = getTerm('exit');
    exitLink.classList.remove('disabled'); // Always enabled in analysis mode

    // Update "Undo" link (button2). Strip the about-mode href/target so this
    // click is handled in JS only and doesn't navigate away to about.html.
    const undoLink = document.getElementById('button2');
    undoLink.style.display = '';
    undoLink.textContent = getTerm('undo');
    undoLink.setAttribute('href', '#');
    undoLink.removeAttribute('target');
    undoLink.removeAttribute('rel');
    if (state.kifuReader && state.kifuReader.node.parent && state.kifuReader.node !== state.initialAnalysisNode) {
        undoLink.classList.remove('disabled');
    } else {
        undoLink.classList.add('disabled');
    }

    // Share (button3) and Ignore (button4) live in the hamburger menu now;
    // their DOM nodes stay so the menu items can dispatch clicks through them.
    document.getElementById('button3').style.display = 'none';
    document.getElementById('button4').style.display = 'none';

    updateMenuItems();
}

const updateMenuItems = () => {
    const sizeItem = document.getElementById('menu-board-size');
    const pauseItem = document.getElementById('menu-pause-recalls');
    const aboutItem = document.getElementById('menu-about');
    const statsItem = document.getElementById('menu-stats');
    const shareItem = document.getElementById('menu-share');
    const ignoreItem = document.getElementById('menu-ignore');
    const reviewGroup = document.getElementById('menu-group-review');

    // Category headers — static labels, set here so they follow the active language.
    document.getElementById('menu-cat-practice').textContent = getTerm('menuCatPractice');
    document.getElementById('menu-cat-review').textContent = getTerm('menuCatReview');
    document.getElementById('menu-cat-more').textContent = getTerm('menuCatMore');

    // Practice group — always available.
    const otherSize = SUPPORTED_BOARD_SIZES.find((s) => s !== state.boardSize) ?? state.boardSize;
    sizeItem.textContent = getTerm('switchTo', { size: `${otherSize}x${otherSize}` });
    // Pause/Resume recalls label flips on the live timestamp — an expired pause
    // auto-flips back to "Pause recalls".
    const recallsArePaused = state.recallPausedUntil > Date.now();
    pauseItem.textContent = getTerm(recallsArePaused ? 'resumeRecalls' : 'pauseRecalls');
    // Sound toggle label names the action, not the current state.
    document.getElementById('menu-audio').textContent =
        getTerm(state.audioEnabled ? 'disableSound' : 'enableSound');

    // More group — Stats and About are always available, in every mode.
    statsItem.textContent = getTerm('stats');
    statsItem.classList.remove('disabled');
    // Account appears only once the one-off history backfill has confirmed
    // every local attempt is on the server (fresh devices pass immediately) —
    // until then, logging this account in elsewhere would rebuild from an
    // incomplete server log and silently lose the unreported history.
    const accountItem = document.getElementById('menu-account');
    accountItem.style.display = localStorage.getItem(HISTORY_BACKFILL_DONE_KEY) ? '' : 'none';
    accountItem.textContent = getTerm('menuAccount');
    document.getElementById('menu-leaderboard').textContent = getTerm('leaderboard');
    aboutItem.textContent = getTerm('about');
    aboutItem.classList.remove('disabled');
    document.getElementById('menu-discord').textContent = getTerm('discord');

    // Review group — Share/Ignore act on the problem under review, so the whole
    // category only appears in analysis mode.
    if (state.appMode === APP_MODES.ANALYZING) {
        reviewGroup.style.display = '';

        shareItem.textContent = getTerm('share');
        shareItem.classList.remove('disabled');

        // Ignore: enabled only when there's a current problem and it's a
        // retry/recall (matches the analysis-mode ignore-button rule).
        ignoreItem.textContent = getTerm('ignore');
        const problem = state.previousProblem;
        if (problem) {
            const lastAnswer = state.answerHistory[state.answerHistory.length - 1];
            const wasFailure = lastAnswer && lastAnswer.problemId === tsumegoIdOf(problem.filename) && !lastAnswer.correct;
            const isRepeat = problem.isRepeat;
            if (!problem.linked && (isRepeat || wasFailure)) {
                ignoreItem.classList.remove('disabled');
            } else {
                ignoreItem.classList.add('disabled');
            }
        } else {
            ignoreItem.classList.add('disabled');
        }
    } else {
        reviewGroup.style.display = 'none';
    }
};

import { displayOverlay, fadeOverlay, showModal, modalTitle } from './overlays.js';




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
            const problem = state.previousProblem;
            if (problem && problem.rank !== undefined) {
                const rank = getDescriptiveRank(problem.rank);
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
