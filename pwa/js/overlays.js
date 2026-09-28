// js/overlays.js

import { state, setAcceptClicks } from './state.js';
import { getTerm, getDescriptiveRank } from './i18n.js';
// Deliberate ESM cycle (ui.js imports this module): updateAllUI is only ever
// called at runtime, never at module-eval time — same rule as board.js.
import { updateAllUI } from './ui.js';

// Sizes below are CSS token expressions, not measurements. The result overlay's
// type is a fraction of --board-side (the board's edge length, which style.css
// already derives per layout) rather than a read of #board-container.offsetWidth
// — measuring made this file a scaling system of its own, on a basis nothing
// else shared, and it re-measured on every overlay build.

// Display resolution of each of the two numbers below, in that number's own unit:
// what one step of the last digit the format prints is worth. Both double as the
// smallest movement the overlay will claim (signedAtLeast).
const RANK_CHANGE_DECIMALS = 3;
const RANK_CHANGE_STEP = 10 ** -RANK_CHANGE_DECIMALS;
const RECALL_DELTA_STEP_HOURS = 1 / 3600;   // one second, the finest unit printed below

// Signed time change for the recall-impact overlay, in the largest unit that keeps
// the number readable: seconds under a minute, minutes under an hour, else hours.
function formatRecallDelta(deltaHours) {
    const sign = deltaHours < 0 ? '−' : '+';
    const secs = Math.abs(deltaHours) * 3600;
    if (secs < 60) return `${sign}${Math.round(secs)} s`;
    if (secs < 3600) return `${sign}${Math.round(secs / 60)} min`;
    return `${sign}${(secs / 3600).toFixed(1)} h`;
}

// The overlay only ever reports an adjustment that AGREES with the answer, and it
// never reports nothing: a correct answer shows a gain of at least one step of the
// last digit shown, a miss a loss of at least that much. Both numbers it shows are
// honest outputs that can nonetheless land on the wrong side of zero — the rank is a
// window posterior whose display offset moves with the standard error, and the recall
// delay is a real before/after retrain whose two heads can disagree on a window
// (recallImpact) — and a "+" after a miss, or a flat zero after a correct
// answer, each read as a bug to the user. The floor is at the display resolution
// rather than at zero for the same reason: "+0.000" claims the answer bought nothing,
// which is never what the model means. DISPLAY-ONLY: the stored rank and the recall
// model keep whatever they actually computed.
function signedAtLeast(value, isCorrect, step) {
    if (!Number.isFinite(value)) return value;
    return isCorrect ? Math.max(value, step) : Math.min(value, -step);
}


function createOverlayContent(stats, message, textColor) {
    const container = document.createElement('div');
    container.className = 'result-content';
    container.style.color = textColor;

    // The "Correct!" line stands above whatever the answer changed. It keys off
    // the verdict alone, so a linked or dropped attempt — which records nothing
    // and so has no rank or recall line — still gets it.
    if (stats && stats.isCorrect) {
        const banner = document.createElement('div');
        banner.className = 'result-banner';
        banner.textContent = getTerm('correctBanner');
        container.appendChild(banner);
    }

    if (stats && stats.recallImpactPromise) {
        // Recall (repeat) attempt: the rank barely moves, so show how this answer
        // shifted the time until first repeat of a generic problem instead (one
        // line, e.g. "+7 min repeat delay"). The numbers come from the retrain
        // updateUserStats defers off the overlay's critical path, so the line is
        // reserved blank up front (no layout shift under any message below it)
        // and filled when the promise resolves — usually well within the
        // overlay's hold. A null impact (model not ready) removes the blank line.
        const el = document.createElement('div');
        el.className = 'result-line';
        el.textContent = ' '; // U+00A0 — a plain space collapses to zero height
        container.appendChild(el);
        const fill = ({ beforeHours, afterHours }) => {
            const change = signedAtLeast(afterHours - beforeHours, stats.isCorrect, RECALL_DELTA_STEP_HOURS);
            el.textContent = getTerm('recallDelayFormat', { change: formatRecallDelta(change) });
        };
        stats.recallImpactPromise
            .then(impact => { if (impact) fill(impact); else el.remove(); })
            .catch(() => el.remove());
    } else if (stats && Number.isFinite(stats.rankChange)) {
        // Floored to one step of the last digit shown, in the answer's own direction:
        // a wrong-signed move is reported as the smallest movement rather than
        // suppressed, so the line is always there and always says the answer counted.
        const rankChange = signedAtLeast(stats.rankChange, stats.isCorrect, RANK_CHANGE_STEP);
        const rankChangeElement = document.createElement('div');
        rankChangeElement.className = 'result-line';
        const sign = rankChange < 0 ? '−' : '+';
        const formattedChange = `${sign}${Math.abs(rankChange).toFixed(RANK_CHANGE_DECIMALS)}`;
        rankChangeElement.textContent = getTerm('rankChangeFormat', { change: formattedChange });
        container.appendChild(rankChangeElement);

        // The band-crossing lines: OnlyMove's, then every server whose label in
        // the badge bubble changed. attempt-recorder has already decided each
        // (rankBandCrossing / serverBandCrossings) and only reports a crossing
        // that agrees with the answer — so no line can contradict the number
        // above. Every line names its system, OnlyMove's included.
        const crossings = [];
        if (stats.rankChangeDirection !== 'same') {
            crossings.push({ system: 'OnlyMove', direction: stats.rankChangeDirection,
                rank: getDescriptiveRank(state.displayRank) });
        }
        for (const c of stats.serverRankCrossings || []) {
            crossings.push({ system: c.server, direction: c.direction, rank: getDescriptiveRank(c.band) });
        }
        if (crossings.length > 0) {
            const block = document.createElement('div');
            block.className = 'result-line-sm';
            for (const c of crossings) {
                const line = document.createElement('div');
                line.textContent = getTerm(c.direction === 'up' ? 'rankUpTo' : 'rankDownTo',
                    { rank: c.rank, system: c.system });
                block.appendChild(line);
            }
            container.appendChild(block);
        }
    }

    if (message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'result-message';
        messageElement.textContent = message;
        container.appendChild(messageElement);
    }

    return container;
}

// Empty board average color: the spatial mean of the kaya wood texture (wood_1024.jpg).
const EMPTY_BOARD_AVG_COLOR = '#e4b25c';
const INK_TEXT_COLOR = '#211f1c';

function getOverlayParent() {
    return document.getElementById('board-container') || document.body;
}

function getBoardWrapper(parent) {
    return document.getElementById('board-wrapper') || parent.querySelector('#board-wrapper') || parent.firstElementChild;
}

// Simple message overlay (errors, notices). The answer-result overlay with the
// rank-change stats block is fadeOverlay below (which holds through the swap);
// this one just shows a message and auto-dismisses (or waits for the button).
// Scoped to the board (#board-container).
export function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)', buttonText = null) {
    const parent = getOverlayParent();
    const boardWrapper = getBoardWrapper(parent);
    const existing = parent.querySelectorAll('.result-overlay');
    existing.forEach(el => el.remove());

    const overlay = document.createElement('div');
    // `app-blocking-overlay` lets onboarding.js hold a bubble back until this
    // transition lifts (so a callout never floats over the fade screen).
    overlay.className = 'overlay app-blocking-overlay result-overlay';
    overlay.setAttribute('data-nosnippet', '');
    overlay.style.backgroundColor = color;

    const content = document.createElement('div');
    content.className = 'result-content';
    content.style.color = 'white';
    if (message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'result-message result-message-lg';
        messageElement.textContent = message;
        content.appendChild(messageElement);
    }
    overlay.appendChild(content);
    parent.appendChild(overlay);

    if (boardWrapper) {
        boardWrapper.classList.add('board-blurred');
    }

    // Fast fade in/out for messages
    void overlay.offsetWidth;
    overlay.style.transition = 'opacity 0.3s ease-in-out';
    overlay.style.opacity = '0.9';

    const dismiss = () => {
        if (boardWrapper) {
            boardWrapper.classList.remove('board-blurred');
        }
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.remove();
            if (callback) callback();
        }, 300);
    };

    if (buttonText) {
        // A confirm button holds the message up until the user acknowledges
        // it, rather than auto-dismissing after `duration`.
        const button = document.createElement('button');
        button.textContent = buttonText;
        button.className = 'result-btn';
        button.onclick = dismiss;
        content.appendChild(button);
    } else {
        setTimeout(dismiss, duration);
    }
}

function captureBoardSnapshot(boardWrapper) {
    if (!boardWrapper) return null;
    const canvases = Array.from(boardWrapper.querySelectorAll('canvas'))
        .sort((a, b) => (parseInt(a.style.zIndex, 10) || 0) - (parseInt(b.style.zIndex, 10) || 0));
    if (!canvases.length) return null;

    const first = canvases[0];
    const snapshot = document.createElement('canvas');
    snapshot.width = first.width;
    snapshot.height = first.height;
    snapshot.className = 'board-snapshot';

    // Copy matching wood background from .wgo-board so the snapshot is 100% opaque
    const wgoBoard = boardWrapper.querySelector('.wgo-board');
    if (wgoBoard && wgoBoard.style.backgroundImage) {
        snapshot.style.backgroundImage = wgoBoard.style.backgroundImage;
    }
    if (wgoBoard && wgoBoard.style.backgroundColor) {
        snapshot.style.backgroundColor = wgoBoard.style.backgroundColor;
    }

    const ctx = snapshot.getContext('2d');
    if (!ctx) return null;

    for (const c of canvases) {
        ctx.drawImage(c, 0, 0);
    }
    return snapshot;
}

// The default board fade is the established "correct answer / next problem"
// cue (red = wrong answer) — callers signalling a NEUTRAL event (e.g. the
// post-sync problem swap) pass their own colors so the fade isn't misread as
// an answer result.
// Overlays are scoped to the board (#board-container). For transparent transitions
// (e.g. correct answers), a snapshot of the current board is captured and blurred,
// holding the old position while the new problem loads underneath, then dissolving
// into the new board as it unblurs — giving a seamless cross-fade with no color cover.
// Returns a promise that resolves when the fade-out starts, i.e. when the
// board underneath becomes visible to the user.
let fadeGeneration = 0;
export function fadeOverlay(duration, callback, stats = null, message = null, colors = { background: 'transparent', text: 'white' }) {
    const parent = getOverlayParent();
    const boardWrapper = getBoardWrapper(parent);
    const existing = parent.querySelectorAll('.result-overlay');
    existing.forEach(el => el.remove());
    const existingSnapshots = parent.querySelectorAll('.board-snapshot');
    existingSnapshots.forEach(el => el.remove());

    const snapshot = captureBoardSnapshot(boardWrapper);
    if (snapshot) {
        parent.appendChild(snapshot);
    }

    const overlay = document.createElement('div');
    // See displayOverlay: marks this as a fullscreen transition so onboarding.js
    // defers any bubble until it lifts.
    overlay.className = 'app-blocking-overlay result-overlay';
    overlay.style.backgroundColor = colors.background;

    if (stats || message) {
        overlay.appendChild(createOverlayContent(stats, message || '', colors.text));
    }

    parent.appendChild(overlay);

    // Commit initial state (opacity: 0 on overlay, clean snapshot) to layout
    // before triggering the transition, so newly inserted DOM elements transition
    // smoothly rather than snapping instantly to their target values.
    if (snapshot) void snapshot.offsetWidth;
    void overlay.offsetWidth;

    if (boardWrapper) boardWrapper.classList.add('board-blurred');
    if (snapshot) snapshot.classList.add('board-blurred');
    overlay.style.opacity = '1';

    // Halfway through the transition (at 500ms), under full blur cover,
    // run the callback (which loads the next problem). If it returns a promise,
    // hold the overlay until that work finishes — the cover must never lift onto
    // an unready board.
    const swapped = new Promise(resolve => setTimeout(resolve, 500))
        .then(() => (callback ? callback() : undefined));
    // ...but stay up at least `duration` so the transition never flickers when
    // the load is instant (cached problem).
    const minDisplay = new Promise(resolve => setTimeout(resolve, duration));

    // Clicks stay off until the cover LIFTS, not until the swap underneath
    // has painted: the swap re-enables them (displayProblem, enterAnalysisMode),
    // and the overlay covers only the board, so for the rest of the hold the
    // action buttons were live over a board the user could not see — a
    // Previous click there entered review under the correct-answer fade. The
    // callback's own re-enable is undone here once it settles, and the lift
    // restores it; a newer fade started meanwhile owns the lock instead.
    const generation = ++fadeGeneration;
    const relocked = swapped.catch(() => {}).then(() => {
        if (generation !== fadeGeneration) return;
        setAcceptClicks(false);
        updateAllUI();
    });

    return Promise.all([relocked, minDisplay]).catch(() => {}).then(() => {
        if (generation === fadeGeneration) {
            setAcceptClicks(true);
            updateAllUI();
        }
        if (boardWrapper) {
            boardWrapper.classList.remove('board-blurred');
        }
        if (snapshot) {
            snapshot.style.opacity = '0';
        }
        overlay.style.pointerEvents = 'none';
        overlay.style.transition = 'opacity 0.5s ease-in-out';
        overlay.style.opacity = '0';
        setTimeout(() => {
            if (snapshot) snapshot.remove();
            overlay.remove();
        }, 500);
    });
}

// Generic loading spinner shown while the app waits on async work (startup file
// loads, the stats compute, etc.). Reference-counted so overlapping callers
// don't tear down each other's spinner — the overlay lifts only when the last
// showLoading() has been matched by a hideLoading(). Scoped to #board-container.
let _loadingCount = 0;
let _loadingEl = null;

export function showLoading(message = null) {
    _loadingCount++;
    if (_loadingEl) return;
    const parent = getOverlayParent();
    const boardWrapper = getBoardWrapper(parent);
    if (boardWrapper) {
        boardWrapper.classList.add('board-blurred');
    }

    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    overlay.appendChild(spinner);
    // Optional static text — the heavy synchronous builds (e.g. the stats
    // charts) freeze the spinner's animation, so a plain explanatory line is
    // the reliable "still working" signal. It paints with the overlay and
    // stays put through the block.
    if (message) {
        const label = document.createElement('div');
        label.className = 'loading-message';
        label.textContent = message;
        overlay.appendChild(label);
    }
    parent.appendChild(overlay);
    _loadingEl = overlay;
    void overlay.offsetWidth;
    overlay.classList.add('visible');
}

export function hideLoading() {
    _loadingCount = Math.max(0, _loadingCount - 1);
    if (_loadingCount > 0 || !_loadingEl) return;
    const parent = getOverlayParent();
    const boardWrapper = getBoardWrapper(parent);
    if (boardWrapper) {
        boardWrapper.classList.remove('board-blurred');
    }
    const overlay = _loadingEl;
    _loadingEl = null;
    overlay.classList.remove('visible');
    setTimeout(() => overlay.remove(), 200);
}

// --- Shared modal-dialog scaffold ------------------------------------------
// The app's dialogs (board-size picker, confirm, share chooser, iOS install,
// rank survey, rank translation) all share the #ios-install-overlay /
// #ios-install-modal shell, optional backdrop-click dismissal, and the same
// content-sized button styling. These helpers build that shell once so callers
// only supply the body.

/**
 * Builds the overlay + modal shell, optionally wires backdrop-click dismissal,
 * and runs `build(modal, close)` so the caller fills the (empty) modal and wires
 * its own buttons. Returns `{ overlay, modal, close }`. `close()` is idempotent.
 */
export function showModal({ build, dismissable = true, onDismiss } = {}) {
    const overlay = document.createElement('div');
    overlay.id = 'ios-install-overlay';
    overlay.setAttribute('data-nosnippet', '');
    const modal = document.createElement('div');
    modal.id = 'ios-install-modal';
    overlay.appendChild(modal);

    let closed = false;
    const close = () => {
        if (closed) return;
        closed = true;
        overlay.remove();
    };

    if (dismissable) {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                close();
                if (onDismiss) onDismiss();
            }
        });
    }

    if (build) build(modal, close);
    document.body.appendChild(overlay);
    return { overlay, modal, close };
}

/**
 * Custom Yes/Cancel confirmation dialog. Replaces the native blocking
 * `confirm()` (which is janky in an installed PWA and inconsistent with the
 * app's other custom overlays). Resolves to true on confirm, false on
 * cancel/backdrop.
 */
export function showConfirm(message) {
    return new Promise((resolve) => {
        let settled = false;
        const settle = (result, close) => {
            if (settled) return;
            settled = true;
            close();
            resolve(result);
        };
        showModal({
            onDismiss: () => settle(false, () => {}),
            build: (modal, close) => {
                modal.appendChild(modalTitle(message));
                modal.appendChild(modalButtonRow([
                    modalButton(getTerm('confirmNo'), () => settle(false, close)),
                    modalButton(getTerm('confirmYes'), () => settle(true, close)),
                ]));
            },
        });
    });
}

/** A modal title row (#ios-install-title). */
export function modalTitle(text) {
    const el = document.createElement('div');
    el.id = 'ios-install-title';
    el.textContent = text;
    return el;
}

/**
 * A content-sized modal button. Plain `.btn` stretches full-width in the
 * landscape side-panel media query; `.btn-inline` resets it to size-to-content.
 */
export function modalButton(label, onClick) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-inline';
    btn.textContent = label;
    if (onClick) btn.addEventListener('click', onClick);
    return btn;
}

/** A centered flex row of modal buttons. */
export function modalButtonRow(buttons) {
    const row = document.createElement('div');
    row.className = 'modal-button-row';
    for (const b of buttons) row.appendChild(b);
    return row;
}

/**
 * A centered two-column (label, value) grid, scroll-capped at 60vh — shared by
 * the rank-survey and rank-translation dialogs. Callers append their own
 * label/value (or label/select) pairs.
 */
export function modalRankGrid() {
    const grid = document.createElement('div');
    grid.className = 'modal-rank-grid';
    return grid;
}

// Shared centered container that STACKS toasts vertically: one flex column
// centers the group and lays them out, so any two alive at once (e.g. the
// "preparing…" toast plus a result, or rapid retries) don't render on the exact
// same spot and overlap into garbled text.
let toastContainer = null;
function getToastContainer() {
    if (toastContainer && toastContainer.isConnected) return toastContainer;
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-stack';
    document.body.appendChild(toastContainer);
    return toastContainer;
}

/**
 * A small centered toast. Auto-dismisses after 2s unless `autoDismiss` is
 * false (the caller then removes the returned element itself, e.g. to keep a
 * "preparing…" message up across an async op). Returns the toast element.
 * Multiple live toasts stack vertically rather than overlapping.
 */
export function showToast(text, { isError = false, autoDismiss = true } = {}) {
    const message = document.createElement('div');
    message.textContent = text;
    message.className = isError ? 'toast toast-error' : 'toast';
    getToastContainer().appendChild(message);
    if (autoDismiss) {
        setTimeout(() => {
            if (message.parentNode) message.parentNode.removeChild(message);
        }, 2000);
    }
    return message;
}
