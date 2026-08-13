// js/overlays.js

import { state } from './state.js';
import { getTerm, getDescriptiveRank } from './i18n.js';

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
    Object.assign(container.style, {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color: textColor,
        fontWeight: 'bold',
        textAlign: 'center',
    });

    if (stats && stats.recallImpactPromise) {
        // Recall (repeat) attempt: the rank barely moves, so show how this answer
        // shifted the time until first repeat of a generic problem instead (one
        // line, e.g. "+7 min repeat delay"). The numbers come from the retrain
        // updateUserStats defers off the overlay's critical path, so the line is
        // reserved blank up front (no layout shift under any message below it)
        // and filled when the promise resolves — usually well within the
        // overlay's hold. A null impact (model not ready) removes the blank line.
        const el = document.createElement('div');
        Object.assign(el.style, {
            fontSize: `calc(var(--board-side) / 20)`,
            marginBottom: '20px',
        });
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
        Object.assign(rankChangeElement.style, {
            fontSize: `calc(var(--board-side) / 20)`,
            marginBottom: '20px',
        });
        const sign = rankChange < 0 ? '−' : '+';
        const formattedChange = `${sign}${Math.abs(rankChange).toFixed(RANK_CHANGE_DECIMALS)}`;
        rankChangeElement.textContent = getTerm('rankChangeFormat', { change: formattedChange });
        container.appendChild(rankChangeElement);

        // The band-crossing line. attempt-recorder's rankBandCrossing has already
        // decided it, and only reports a crossing that agrees with the answer — so
        // this line can never contradict the number above it.
        if (stats.rankChangeDirection !== 'same') {
            const rankChangeMessageElement = document.createElement('div');
            Object.assign(rankChangeMessageElement.style, {
                fontSize: `calc(var(--board-side) / 25)`,
                marginBottom: '20px',
            });
            const newRank = getDescriptiveRank(state.displayRank);
            const term = stats.rankChangeDirection === 'up' ? 'rankUpTo' : 'rankDownTo';
            rankChangeMessageElement.textContent = getTerm(term, { rank: newRank });
            container.appendChild(rankChangeMessageElement);
        }
    }

    if (message) {
        const messageElement = document.createElement('div');
        Object.assign(messageElement.style, {
            fontSize: `calc(var(--board-side) / 30)`,
            maxWidth: '80%',
        });
        messageElement.textContent = message;
        container.appendChild(messageElement);
    }

    return container;
}

// Simple message overlay (errors, notices). The answer-result overlay with the
// rank-change stats block is fadeOverlay below (which holds through the swap);
// this one just shows a message and auto-dismisses (or waits for the button).
export function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)', buttonText = null) {
    const overlay = document.createElement('div');
    // `app-blocking-overlay` lets onboarding.js hold a bubble back until this
    // transition lifts (so a callout never floats over the fade screen).
    overlay.className = 'overlay app-blocking-overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: color, display: 'flex', justifyContent: 'center',
        alignItems: 'center', zIndex: '1000', opacity: '0'
    });

    const content = document.createElement('div');
    Object.assign(content.style, {
        display: 'flex', flexDirection: 'column', alignItems: 'center',
    });
    if (message) {
        const messageElement = document.createElement('div');
        Object.assign(messageElement.style, {
            color: 'white', fontWeight: 'bold', textAlign: 'center',
            fontSize: `calc(var(--board-side) / 25)`, padding: '20px'
        });
        messageElement.textContent = message;
        content.appendChild(messageElement);
    }
    overlay.appendChild(content);
    document.body.appendChild(overlay);

    // Fast fade in/out for messages
    overlay.style.transition = 'opacity 0.3s ease-in-out';
    requestAnimationFrame(() => {
        overlay.style.opacity = '0.9';
    });

    const dismiss = () => {
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
        Object.assign(button.style, {
            marginTop: '20px', padding: '10px 20px',
            fontSize: `calc(var(--board-side) / 30)`, cursor: 'pointer',
            border: '1px solid white', borderRadius: '5px',
            backgroundColor: 'transparent', color: 'white', fontWeight: 'bold',
        });
        button.onclick = dismiss;
        content.appendChild(button);
    } else {
        setTimeout(dismiss, duration);
    }
}

// The default white fade is the established "correct answer / next problem"
// cue (red = wrong answer) — callers signalling a NEUTRAL event (e.g. the
// post-sync problem swap) pass their own colors so the fade isn't misread as
// an answer result.
// With `buttonText` the overlay is additionally held until the user taps the
// button (the min-duration gate is skipped — the button IS the dismissal);
// the swap gate still applies, so it can never lift onto an unready board.
// Returns a promise that resolves when the fade-out starts, i.e. when the
// board underneath becomes visible to the user.
export function fadeOverlay(duration, callback, stats = null, message = null, colors = { background: 'white', text: 'black' }, { buttonText = null } = {}) {
    const overlay = document.createElement('div');
    // See displayOverlay: marks this as a fullscreen transition so onboarding.js
    // defers any bubble until it lifts.
    overlay.className = 'app-blocking-overlay';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: colors.background, zIndex: '1000', opacity: '0',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    });

    let buttonClicked = Promise.resolve();
    if (stats || message || buttonText) {
        const content = createOverlayContent(stats, message || '', colors.text);
        if (buttonText) {
            buttonClicked = new Promise((resolve) => {
                const button = document.createElement('button');
                button.textContent = buttonText;
                Object.assign(button.style, {
                    marginTop: '20px',
                    padding: '10px 20px',
                    fontSize: `calc(var(--board-side) / 30)`,
                    cursor: 'pointer',
                    border: `1px solid ${colors.text}`,
                    borderRadius: '5px',
                    backgroundColor: 'transparent',
                    color: colors.text,
                    fontWeight: 'bold',
                });
                button.onclick = () => {
                    button.disabled = true;
                    resolve();
                };
                content.appendChild(button);
            });
        }
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

    // Halfway through the fade-in, run the callback (which loads the next
    // problem). If it returns a promise, hold the overlay until that work
    // finishes — the white screen must never lift onto a board that isn't ready.
    const swapped = new Promise(resolve => setTimeout(resolve, 500))
        .then(() => (callback ? callback() : undefined));
    // ...but stay up at least `duration` so the transition never flickers when
    // the load is instant (cached problem).
    const minDisplay = buttonText
        ? Promise.resolve()
        : new Promise(resolve => setTimeout(resolve, duration));

    return Promise.all([swapped, minDisplay, buttonClicked]).catch(() => {}).then(() => {
        overlay.style.transition = 'opacity 0.5s ease-in-out';
        overlay.style.opacity = '0';
        setTimeout(() => overlay.remove(), 500);
    });
}

// Generic loading spinner shown while the app waits on async work (startup file
// loads, the stats compute, etc.). Reference-counted so overlapping callers
// don't tear down each other's spinner — the overlay lifts only when the last
// showLoading() has been matched by a hideLoading().
let _loadingCount = 0;
let _loadingEl = null;

export function showLoading(message = null) {
    _loadingCount++;
    if (_loadingEl) return;
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
    document.body.appendChild(overlay);
    _loadingEl = overlay;
    requestAnimationFrame(() => overlay.classList.add('visible'));
}

export function hideLoading() {
    _loadingCount = Math.max(0, _loadingCount - 1);
    if (_loadingCount > 0 || !_loadingEl) return;
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
    Object.assign(toastContainer.style, {
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 'calc(1 * var(--u))',
        zIndex: '10000',
        pointerEvents: 'none',
    });
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
    Object.assign(message.style, {
        backgroundColor: isError ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '0.3333em 0.6667em',
        borderRadius: '0.1667em',
        fontWeight: 'bold',
        fontSize: 'calc(3 * var(--u))',
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
        textAlign: 'center',
    });
    getToastContainer().appendChild(message);
    if (autoDismiss) {
        setTimeout(() => {
            if (message.parentNode) message.parentNode.removeChild(message);
        }, 2000);
    }
    return message;
}
