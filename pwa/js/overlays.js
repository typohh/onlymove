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
