// js/rank-translation.js
//
// Tap your rank → see the equivalent rank on each surveyed go server.
//
// The mapping (pwa/tsumegos/rank_map.json) is produced by
// pipeline.fit_rank_curves. Our singular rank scale is the hub: it is coupled
// to each server by one board-independent fit, and to the per-board ratings by
// the cubic. So the translation is keyed only on the rank label — the
// per-board difference already lives in the user's displayed rank itself.
// Keyed {ourRankLabel: {server: serverRank}}.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import { TOTAL_RANKS, APP_MODES } from './constants.js';

let _mapPromise = null;

function loadRankMap() {
    if (_mapPromise === null) {
        _mapPromise = fetch(`tsumegos/rank_map.json?v=${APP_VERSION}`)
            .then(r => (r.ok ? r.json() : null))
            .catch(() => null);
    }
    return _mapPromise;
}

// The same integer index getDescriptiveRank uses (floor(rating · 29)), in the
// rank_map's key format (kyu/dan with a k/d suffix, not localized).
function rankIndexToKey(rankIndex) {
    return rankIndex < 20 ? `${20 - rankIndex}k` : `${rankIndex - 19}d`;
}

export async function showRankTranslation() {
    // Only meaningful for the user's own rank in solving mode (analysis/linked
    // modes show a problem's rank or nothing).
    if (state.appMode !== APP_MODES.SOLVING) return;
    const rating = state.userRank;
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return;

    const map = await loadRankMap();
    const rankValue = rating * TOTAL_RANKS;
    const rankIndex = Math.floor(rankValue);
    const key = rankIndexToKey(rankIndex);
    const servers = (map && map[key]) || null;

    const overlay = document.createElement('div');
    overlay.id = 'ios-install-overlay';
    const modal = document.createElement('div');
    modal.id = 'ios-install-modal';

    // Our own rank as "<rank>, <progress>% towards <next rank>", shown above the
    // other-server translation. Progress is the fractional position within the
    // current rank band; the next rank is one step stronger (higher index).
    const progress = document.createElement('div');
    progress.style.textAlign = 'center';
    progress.style.marginBottom = '1vmin';
    progress.style.fontWeight = 'bold';
    progress.textContent = getTerm('rankProgressFormat')
        .replace('{rank}', key)
        .replace('{percent}', Math.round((rankValue - rankIndex) * 100))
        .replace('{nextRank}', rankIndexToKey(rankIndex + 1));
    modal.appendChild(progress);

    const title = document.createElement('div');
    title.id = 'ios-install-title';
    title.textContent = getTerm('rankTranslationTitle');
    modal.appendChild(title);

    if (servers && Object.keys(servers).length > 0) {
        const grid = document.createElement('div');
        grid.style.display = 'grid';
        grid.style.gridTemplateColumns = 'max-content max-content';
        grid.style.justifyContent = 'center';
        grid.style.gap = '0.8vmin 1.5vmin';
        grid.style.alignItems = 'center';
        grid.style.maxHeight = '60vh';
        grid.style.overflowY = 'auto';
        for (const [server, rank] of Object.entries(servers)) {
            const label = document.createElement('div');
            label.textContent = server;
            label.style.fontWeight = 'bold';
            label.style.textAlign = 'right';
            const val = document.createElement('div');
            val.textContent = rank;
            grid.appendChild(label);
            grid.appendChild(val);
        }
        modal.appendChild(grid);
    } else {
        const empty = document.createElement('div');
        empty.style.textAlign = 'center';
        empty.textContent = getTerm('rankTranslationEmpty');
        modal.appendChild(empty);
    }

    const buttonRow = document.createElement('div');
    buttonRow.style.display = 'flex';
    buttonRow.style.justifyContent = 'center';
    buttonRow.style.marginTop = '1vmin';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'btn';
    closeBtn.textContent = getTerm('rankTranslationClose');
    // .btn stretches to full width in the landscape side-panel media query;
    // reset so the dialog button sizes to content (same as rank-survey.js).
    closeBtn.style.flex = '0 0 auto';
    closeBtn.style.width = 'auto';
    closeBtn.style.padding = '0.6vmin 2vmin';
    closeBtn.style.borderRadius = 'var(--radius)';
    closeBtn.style.minWidth = '6em';
    closeBtn.addEventListener('click', () => overlay.remove());
    buttonRow.appendChild(closeBtn);
    modal.appendChild(buttonRow);

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) overlay.remove();
    });
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}
