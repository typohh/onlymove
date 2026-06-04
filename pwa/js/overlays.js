// js/overlays.js

import { state } from './state.js';
import { getTerm, getDescriptiveRank } from './i18n.js';
import { TOTAL_RANKS } from './constants.js';

const boardContainer = document.getElementById('board-container');

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
            rankChangeElement.textContent = getTerm('rankChangeFormat', { change: formattedChange });
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
            rankChangeMessageElement.textContent = getTerm(term, { rank: newRank });
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

export function displayOverlay(message, duration, callback, color = 'rgba(139, 0, 0, 1)', stats = null, buttonText = null) {
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

        requestAnimationFrame(() => {
            overlay.style.transition = `opacity 0.5s ease-out`; // Faster fade-in
            requestAnimationFrame(() => {
                overlay.style.opacity = '0.9';
            });
        });

        const dismiss = () => {
            overlay.remove();
            if (callback) callback();
        };

        if (buttonText) {
            const button = document.createElement('button');
            button.textContent = buttonText;
            Object.assign(button.style, {
                marginTop: '20px',
                padding: '10px 20px',
                fontSize: `${boardContainer.offsetWidth / 30}px`,
                cursor: 'pointer',
                border: '1px solid white',
                borderRadius: '5px',
                backgroundColor: 'transparent',
                color: 'white',
                fontWeight: 'bold',
            });
            button.onclick = dismiss;
            content.appendChild(button);
        } else {
            setTimeout(dismiss, duration);
        }

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

    // Halfway through the fade-in, run the callback (which loads the next
    // problem). If it returns a promise, hold the overlay until that work
    // finishes — the white screen must never lift onto a board that isn't ready.
    const swapped = new Promise(resolve => setTimeout(resolve, 500))
        .then(() => (callback ? callback() : undefined));
    // ...but stay up at least `duration` so the transition never flickers when
    // the load is instant (cached problem).
    const minDisplay = new Promise(resolve => setTimeout(resolve, duration));

    Promise.all([swapped, minDisplay]).catch(() => {}).then(() => {
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
 * A small centered toast. Auto-dismisses after 2s unless `autoDismiss` is
 * false (the caller then removes the returned element itself, e.g. to keep a
 * "preparing…" message up across an async op). Returns the toast element.
 */
export function showToast(text, { isError = false, autoDismiss = true } = {}) {
    const message = document.createElement('div');
    message.textContent = text;
    Object.assign(message.style, {
        position: 'fixed',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        backgroundColor: isError ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '1vmin 2vmin',
        borderRadius: '0.5vmin',
        fontWeight: 'bold',
        fontSize: '3vmin',
        zIndex: isError ? '10000' : '9999',
        pointerEvents: 'none',
        whiteSpace: 'nowrap',
        textAlign: 'center',
    });
    document.body.appendChild(message);
    if (autoDismiss) {
        setTimeout(() => {
            if (message.parentNode) message.parentNode.removeChild(message);
        }, 2000);
    }
    return message;
}
