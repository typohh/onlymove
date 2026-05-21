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

export function showLoading() {
    _loadingCount++;
    if (_loadingEl) return;
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    overlay.appendChild(spinner);
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
