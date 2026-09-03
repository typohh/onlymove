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
import { rankProgress } from '../shared/rank-math.js';

/**
 * Returns true if the app is already running as an installed PWA.
 */
const isInstalled = () =>
    window.matchMedia('(display-mode: standalone)').matches ||
    window.navigator.standalone === true;

/**
 * Returns true on iOS/iPadOS. Only iOS lacks a native install prompt AND uses
 * the Safari Share → Add to Home Screen flow; every other prompt-less browser
 * (Vanadium/GrapheneOS, Firefox, other Chromium forks that don't fire
 * beforeinstallprompt) installs via its own menu, so those must NOT be shown the
 * Safari instructions. iPadOS 13+ reports a Mac UA, so also check touch points.
 */
const isIOS = () => {
    const ua = navigator.userAgent || '';
    return /iPad|iPhone|iPod/.test(ua) ||
        (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
};

/**
 * Returns true on Android. Only there is the Play Store TWA (net.onlymove.twa)
 * a usable install target, so the store button in the manual-install modal is
 * gated on this.
 */
const isAndroid = () => /Android/.test(navigator.userAgent || '');

// The Play Store listing for the bubblewrapped TWA (net.onlymove.twa) — the
// same web app, wrapped. Offered as the reliable install path on Android when
// no native beforeinstallprompt fired.
const PLAY_STORE_URL = 'https://play.google.com/store/apps/details?id=net.onlymove.twa';

/**
 * True when the Play Store TWA is already installed on the device while the user
 * browses in a normal Chrome tab (declared via related_applications in
 * manifest.json). Chromium-Android only; resolves false — never throws —
 * everywhere else, so installed-detection degrades to display-mode alone.
 */
const hasInstalledStoreApp = async () => {
    try {
        if (!navigator.getInstalledRelatedApps) return false;
        const apps = await navigator.getInstalledRelatedApps();
        return apps.some((a) => a.platform === 'play' && a.id === 'net.onlymove.twa');
    } catch {
        return false;
    }
};
import { APP_MODES, HISTORY_BACKFILL_DONE_KEY } from './constants.js';
import { getCurrentNode } from './board.js';
import { syncMotifBubble } from './motif-bubble.js';
import { waitingRecallCount } from './recall-model.js';
import { onReviewScoreShown as onboardingReviewScoreShown, onScoreUnreliable as onboardingScoreUnreliable, onRecallBadgeShown as onboardingRecallBadgeShown } from './onboarding.js';

const formatScore = (score) => score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1);

const KOMI = 6.5;
// White's capture-adjusted score for the current position: komi plus the
// capture differential. Shared by the header readout and the details alert.
function computeCaptureScore() {
    const captures = state.kifuReader.game.position.capCount;
    return {
        komi: KOMI,
        blackCaptures: captures.black,
        whiteCaptures: captures.white,
        score: KOMI + captures.white - captures.black,
    };
}

export function updateCapturesUI() {
    if (state.kifuReader) {
        const { score } = computeCaptureScore();
        const formattedScore = formatScore(score);
        document.getElementById('capture-score').textContent = formattedScore;
        document.getElementById('capture-score-h').textContent = formattedScore;
    }
}

const boardContainer = document.getElementById('board-container');

/**
 * Shows a modal with step-by-step instructions to install the app manually,
 * used when no native install prompt is available. The instructions are
 * platform-specific: iOS gets the Safari Share flow; every other prompt-less
 * browser (Vanadium/GrapheneOS, Firefox, …) gets the generic browser-menu flow.
 */
const showManualInstallModal = () => {
    const instructionsKey = isIOS() ? 'installIosInstructions' : 'installManualInstructions';
    showModal({
        build: (modal, close) => {
            modal.appendChild(modalTitle(getTerm('installIosTitle')));

            // On Android the Play Store TWA is the reliable install path when no
            // native prompt fired (Firefox, Samsung Internet, Vanadium, …). Offer
            // it as the primary action; the browser-menu instructions stay below
            // as a fallback for Google-Play-less devices (e.g. GrapheneOS).
            if (isAndroid()) {
                const storeBtn = document.createElement('a');
                storeBtn.id = 'play-store-install';
                storeBtn.className = 'btn';
                storeBtn.href = PLAY_STORE_URL;
                storeBtn.target = '_blank';
                storeBtn.rel = 'noopener';
                storeBtn.textContent = getTerm('installPlayStore');
                storeBtn.addEventListener('click', close);
                modal.appendChild(storeBtn);
            }

            const instructions = document.createElement('div');
            instructions.id = 'ios-install-instructions';
            // Replace literal \n with line breaks
            getTerm(instructionsKey).split('\n').forEach((line, i, arr) => {
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

/**
 * Fallback instructions shown only when the Play TWA is installed but the
 * in-app permission bridge (below) did not launch — i.e. the installed shell
 * predates the bridge activity (versionCode < 3), so the intent:// navigation
 * resolved to nothing. Background: on Android 13+ the app's notification
 * permission is OFF by default, and the delegation flow that should raise the
 * native grant prompt when the site subscribes is broken
 * (android-browser-helper #563): subscribe() succeeds, but the app stays
 * blocked, no reminder can display, and Chrome's next push-subscription
 * revalidation deletes the subscription (the "Notify forgets itself" loop).
 * Walking the user to the system toggle before that sweep legitimizes the
 * fresh subscription. Worded as a check so it stays correct for users already
 * granted. Once the Play rollout of the bridge saturates, this modal (and its
 * i18n strings) can be dropped entirely — the shell itself deep-links the
 * settings screen in the cases that need it.
 */
const showTwaNotificationSettingsModal = () => {
    showModal({
        build: (modal, close) => {
            modal.appendChild(modalTitle(getTerm('twaNotifSettingsTitle')));

            const instructions = document.createElement('div');
            instructions.id = 'twa-notif-instructions';
            getTerm('twaNotifSettingsInstructions').split('\n').forEach((line, i, arr) => {
                instructions.appendChild(document.createTextNode(line));
                if (i < arr.length - 1) instructions.appendChild(document.createElement('br'));
            });
            modal.appendChild(instructions);

            const closeBtn = document.createElement('button');
            closeBtn.className = 'btn';
            closeBtn.textContent = getTerm('installIosClose');
            closeBtn.addEventListener('click', close);
            modal.appendChild(closeBtn);
        },
    });
};

// The in-app notification permission bridge in the TWA shell (versionCode 3+):
// a transparent activity that raises the native POST_NOTIFICATIONS prompt —
// the one Chrome's broken delegation never shows — or deep-links the app's
// notification settings screen when the prompt would be silently auto-denied.
// The explicit package= pins resolution to our shell, so on any other device
// state the navigation is a no-op (the page never unloads either way).
const TWA_BRIDGE_INTENT =
    'intent://notifications#Intent;scheme=onlymove-bridge;package=net.onlymove.twa;end';

/**
 * Fires the permission bridge and reports whether it actually launched.
 * There is no return channel from the shell, but launching is observable:
 * the bridge activity (or its permission dialog / settings screen) covers the
 * TWA, so the page goes hidden. No hidden transition within the timeout means
 * the intent resolved to nothing — an installed shell that predates the
 * bridge — and the caller should fall back to the manual instructions modal.
 * Never rejects.
 */
const launchTwaNotificationBridge = () => new Promise((resolve) => {
    let timer = null;
    const settle = (launched) => {
        document.removeEventListener('visibilitychange', onVisibility);
        clearTimeout(timer);
        resolve(launched);
    };
    const onVisibility = () => {
        if (document.visibilityState === 'hidden') settle(true);
    };
    document.addEventListener('visibilitychange', onVisibility);
    timer = setTimeout(() => settle(false), 3000);
    try {
        window.location.href = TWA_BRIDGE_INTENT;
    } catch {
        settle(false);
    }
});

// State-machine cache for button4. refreshNotifState() refills it async;
// updateInstallButton paints from this snapshot synchronously.
let notifState = { installed: false, permission: 'default', subscribed: false, storeApp: false };

export async function refreshNotifState() {
    // display-mode covers running inside the TWA / installed PWA; the related-app
    // probe additionally treats "TWA installed, browsing in Chrome" as installed
    // so we don't nag those users (and don't invite a duplicate WebAPK install).
    // storeApp is kept separately: with the Play TWA installed, Chrome delegates
    // this origin's notifications to the app, which gates the Android-settings
    // reminder shown after opt-in (see showTwaNotificationSettingsModal).
    notifState.storeApp = await hasInstalledStoreApp();
    notifState.installed = isInstalled() || notifState.storeApp;
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
            // No native prompt: iOS Safari, prompt-less browsers (Vanadium/
            // GrapheneOS, Firefox), or Chrome before the event fires. The modal
            // picks iOS vs. generic browser-menu instructions by platform.
            showManualInstallModal();
        }
        await refreshNotifState();
        updateInstallButton();
        return;
    }

    // Installed. The next gesture either subscribes to notifications or
    // (when already subscribed / blocked) is a no-op — the button is
    // visually disabled in those terminal states.
    if (notifState.permission === 'denied') return;
    if (notifState.subscribed) return;

    let result = null;
    try {
        result = await enableNotifications();
    } catch (err) {
        console.error('enableNotifications failed:', err);
    }
    await refreshNotifState();
    // Inside the TWA, Notification.permission is delegated to the Android app
    // and the renderer's cached value can still read 'default' right after the
    // user accepts the native dialog — so trust the enable flow's own outcome
    // (it holds the live subscription) over the re-read.
    if (result?.ok) {
        notifState.permission = 'granted';
        notifState.subscribed = true;
    }
    updateInstallButton();
    // With the Play TWA installed, the subscription alone isn't enough: the
    // Android app's own notification permission must be on, and the broken
    // delegation prompt means it usually isn't. Fire the shell's in-app
    // permission bridge, which raises the native grant dialog (or the settings
    // screen when the dialog would silently auto-deny). Only when the bridge
    // doesn't launch — a pre-bridge shell version — fall back to the manual
    // walk-to-settings instructions.
    if (result?.ok && notifState.storeApp) {
        const bridged = await launchTwaNotificationBridge();
        if (!bridged) showTwaNotificationSettingsModal();
    }
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
    // A live push subscription implies granted permission, even when the
    // permission property disagrees: in the TWA the delegated
    // Notification.permission can keep reporting 'default' after the Android
    // grant, so requiring 'granted' here would pin the button on "Notify"
    // forever despite working reminders.
    if (notifState.subscribed) {
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
        setInfoLabel(null);
        text1Elements.forEach(element => {
            element.classList.remove('progress-bar-wrapper');
            element.style.removeProperty('--progress');
            element.style.cursor = '';
            element.removeAttribute('title');
        });
    } else {
        // Normal display with user rank information. displayRank, NOT userRank:
        // the badge is the display layer, and it is deliberately conservative while
        // the posterior is wide (see state.js's three-rank note).
        const rank = getDescriptiveRank(state.displayRank);
        text1Elements.forEach(element => element.textContent = `${rank}`);
        setInfoLabel('labelYou');

        // Add progress bar. Same band + progress as getDescriptiveRank above and the
        // modal/Rank tab, via the shared rankProgress — deriving it inline here would
        // desync the bar from the label.
        const progress = rankProgress(state.displayRank).percent;

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

    // About (button2) lives in the hamburger menu now.
    document.getElementById('button2').style.display = 'none';

    // Show button4 as Install button
    updateInstallButton();

    // Last button slot: Recall while solving, Share (disabled — it acts on the
    // problem under review) in view mode. Exactly one is shown, so the other
    // buttons keep their positions across mode switches.
    const shareButton = document.getElementById('button-share');
    const recallButton = document.getElementById('button-recall');
    if (state.appMode === APP_MODES.LINKED) {
        recallButton.style.display = 'none';
        shareButton.style.display = '';
        shareButton.textContent = getTerm('share');
        shareButton.classList.add('disabled');
    } else {
        shareButton.style.display = 'none';
        recallButton.style.display = '';
        updateRecallButton();
    }

    updateMenuItems();
}

/**
 * Paints the Recall button: label, the waiting-recall badge, and its enabled
 * state. It brings a due recall forward, so it is disabled once one is already
 * on screen — as it is with nothing waiting, and during the post-answer
 * transition (acceptClicks, same guard as button1).
 */
const updateRecallButton = () => {
    const btn = document.getElementById('button-recall');
    document.getElementById('button-recall-label').textContent = getTerm('recall');

    const waiting = waitingRecallCount();
    const badge = document.getElementById('button-recall-count');
    badge.textContent = String(waiting);
    badge.hidden = waiting === 0;
    // First-run onboarding: teach the button the first time the badge shows.
    if (waiting > 0) onboardingRecallBadgeShown();

    const onRecall = !!state.currentProblem?.isRepeat;
    if (waiting > 0 && !onRecall && state.acceptClicks) btn.classList.remove('disabled');
    else btn.classList.add('disabled');
};

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
            setInfoLabel('labelProblem');
        } else {
            text1Elements.forEach(element => element.textContent = '');
            setInfoLabel(null);
        }
    } else {
        // For other positions, show analyzing initially
        text1Elements.forEach(element => element.textContent = getTerm('analyzing'));
        setInfoLabel('score');
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

    // Ignore lives in the hamburger menu; button4 has no analyzing-mode role.
    document.getElementById('button4').style.display = 'none';

    // Share acts on the problem under review, so this is where it enables —
    // taking the last slot back from Recall (see updateSolvingModeUI).
    document.getElementById('button-recall').style.display = 'none';
    const shareButton = document.getElementById('button-share');
    shareButton.style.display = '';
    shareButton.textContent = getTerm('share');
    shareButton.classList.remove('disabled');

    updateMenuItems();
}

// The corner mark's privacy label, localized ONCE at boot. It is static chrome
// on a language that cannot change while the app runs (getLang reads the
// browser's setting), so it does not belong in updateAllUI's per-repaint work.
// index.html ships the English text so a fetch that never runs this bundle —
// Google's consent-screen check on the home page, a crawler — still reads a
// label rather than an empty element; this replaces it for everyone else.
export function initCornerMark() {
    const link = document.getElementById('corner-privacy');
    if (link) link.textContent = getTerm('privacyPolicy');
}

const updateMenuItems = () => {
    const sizeItem = document.getElementById('menu-board-size');
    const pauseItem = document.getElementById('menu-pause-recalls');
    const aboutItem = document.getElementById('menu-about');
    const statsItem = document.getElementById('menu-stats');
    const ignoreItem = document.getElementById('menu-ignore');
    const reviewGroup = document.getElementById('menu-group-review');

    // Category headers — static labels, set here so they follow the active language.
    document.getElementById('menu-cat-practice').textContent = getTerm('menuCatPractice');
    document.getElementById('menu-cat-review').textContent = getTerm('menuCatReview');
    document.getElementById('menu-cat-community').textContent = getTerm('menuCatCommunity');
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

    // More group — My stats and About are always available, in every mode.
    statsItem.textContent = getTerm('stats');
    statsItem.classList.remove('disabled');
    // Account appears only once the one-off history backfill has confirmed
    // every local attempt is on the server (fresh devices pass immediately) —
    // until then, logging this account in elsewhere would rebuild from an
    // incomplete server log and silently lose the unreported history.
    const accountItem = document.getElementById('menu-account');
    accountItem.style.display = localStorage.getItem(HISTORY_BACKFILL_DONE_KEY) ? '' : 'none';
    accountItem.textContent = getTerm('menuAccount');
    // Community group — the two standalone Worker pages and the Discord invite.
    document.getElementById('menu-leaderboard').textContent = getTerm('leaderboard');
    document.getElementById('menu-statistics').textContent = getTerm('statistics');
    aboutItem.textContent = getTerm('about');
    aboutItem.classList.remove('disabled');
    document.getElementById('menu-discord').textContent = getTerm('discord');

    // Review group — Ignore acts on the problem under review, so the whole
    // category only appears in analysis mode. (Share lives in the main
    // button row, not the menu.)
    if (state.appMode === APP_MODES.ANALYZING) {
        reviewGroup.style.display = '';

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
    const { komi, blackCaptures, whiteCaptures, score } = computeCaptureScore();

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

// The small label above the header slot, naming what the slot shows: the
// user's rank, the problem's rank or the review score. null clears it.
function setInfoLabel(termKey) {
    const text = termKey ? getTerm(termKey) : '';
    for (const id of ['text1-label', 'text1-h-label']) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }
}

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
                setInfoLabel('labelProblem');
            }
        } else {
            // For other positions, show the score. The slot's label names it, so
            // the value stands alone; status strings stand alone too.
            setInfoLabel('score');
            const formattedScore = typeof score === 'number' ? score.toFixed(1) : score;
            const currentNode = getCurrentNode();
            let isUnreliable = false;
            if (bestMove) {
                const child = currentNode.children.find(c => c.move && c.move.x === bestMove.x && c.move.y === bestMove.y);
                if (child && child._origin === 'api' && !child._searched) {
                    isUnreliable = true;
                }
            }
            const displayText = formattedScore;
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
            // First-run onboarding: explain the review score readout the first
            // time a real one appears (the blinking-red "unreliable" case gets
            // its own, higher-priority bubble). No-op for existing users.
            const isRealScore = displayText !== getTerm('analyzing') && displayText !== getTerm('error');
            if (isRealScore) {
                if (isUnreliable) onboardingScoreUnreliable();
                onboardingReviewScoreShown();
            }
        }
    }
    // Future: could also use bestMove to highlight a button or provide a hint
}
