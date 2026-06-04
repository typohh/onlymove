// js/rank-translation.js
//
// Tap your rank → see the equivalent rank on each surveyed go server.
//
// The mapping (pwa/tsumegos/rank_map.json) is produced by
// pipeline.fit_rank_curves as per-server interpolation knots
// {server: {our, server}} in rank-scalar units (1k=0, 1d=1; no zero gap). Our
// singular rank scale is the hub: it is coupled to each server by one
// board-independent fit, and to the per-board ratings by the cubic. So the
// translation depends only on our rank — the per-board difference already lives
// in the user's displayed rank itself. We interpolate the user's CONTINUOUS
// internal rank through each server's knots and round only for display, so a
// weak 3d and a strong 3d can translate to different server ranks. EGF is the
// identity hub.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import { TOTAL_RANKS, APP_MODES } from './constants.js';
import { interpExtrap, rankScalarToLabel } from '../shared/rank-math.js';
import { showModal, modalTitle, modalButton, modalButtonRow } from './overlays.js';

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
// kyu/dan label format (k/d suffix, not localized).
function rankIndexToKey(rankIndex) {
    return rankIndex < 20 ? `${20 - rankIndex}k` : `${rankIndex - 19}d`;
}

// internal-rank index → rank scalar (1k=0 convention): rank_index = scalar + 19.
const RANK_INDEX_OFFSET = 19;
// Clamp translated server ranks to the real rank range (30k..9d) so extreme
// extrapolation can't print impossible labels like "15 dan" (matches the
// our_lo/our_hi clamp the fit used to apply server-side).
const RANK_SCALAR_LO = -29, RANK_SCALAR_HI = 9;

// interpExtrap + rankScalarToLabel now live in shared/rank-math.js (one copy
// shared with the Worker).

// {server: label} by interpolating the CONTINUOUS internal-rank scalar
// `ourScalar` through each server's knots, then clamping + rounding — so a weak
// vs strong same-band player can land on different server ranks. EGF is the
// identity hub, so it shows the user's own displayed rank (`egfLabel`) verbatim,
// keeping the table consistent with the header rather than re-rounding identity.
function serverRanksFor(map, ourScalar, egfLabel) {
    if (!map) return null;
    const out = {};
    for (const [server, c] of Object.entries(map)) {
        if (server === 'EGF') { out.EGF = egfLabel; continue; }
        if (!c || !Array.isArray(c.our) || !Array.isArray(c.server)
            || c.our.length < 2) continue;
        let s = interpExtrap(ourScalar, c.our, c.server);
        s = Math.max(RANK_SCALAR_LO, Math.min(RANK_SCALAR_HI, s));
        out[server] = rankScalarToLabel(s);
    }
    return out;
}

export async function showRankTranslation() {
    // Only meaningful for the user's own rank in solving mode (analysis/linked
    // modes show a problem's rank or nothing).
    if (state.appMode !== APP_MODES.SOLVING) return;
    const rating = state.userRank;
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return;

    const map = await loadRankMap();
    const rankValue = rating * TOTAL_RANKS;       // continuous internal-rank index
    const rankIndex = Math.floor(rankValue);      // own-rank band (for the header)
    const key = rankIndexToKey(rankIndex);
    // Translate the CONTINUOUS internal rank (not the floored band) through each
    // server's knots, so a weak vs strong same-rank player can map differently.
    const servers = serverRanksFor(map, rankValue - RANK_INDEX_OFFSET, key);

    showModal({
        build: (modal, close) => {
            // Our own rank as "<rank>, <progress>% towards <next rank>", shown
            // above the other-server translation. Progress is the fractional
            // position within the current rank band; the next rank is one step
            // stronger (higher index).
            const progress = document.createElement('div');
            progress.style.textAlign = 'center';
            progress.style.marginBottom = '1vmin';
            progress.style.fontWeight = 'bold';
            progress.textContent = getTerm('rankProgressFormat', {
                rank: key,
                percent: Math.round((rankValue - rankIndex) * 100),
                nextRank: rankIndexToKey(rankIndex + 1),
            });
            modal.appendChild(progress);

            modal.appendChild(modalTitle(getTerm('rankTranslationTitle')));

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

            modal.appendChild(modalButtonRow([
                modalButton(getTerm('rankTranslationClose'), close),
            ]));
        },
    });
}
