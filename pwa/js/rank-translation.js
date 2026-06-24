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

import { state, getMergedAnswerHistory } from './state.js';
import { getTerm } from './i18n.js';
import { TOTAL_RANKS, APP_MODES } from './constants.js';
import { interpExtrap, rankScalarToLabel, rankIndexToLabel, RANK_INDEX_OFFSET } from '../shared/rank-math.js';
import { rankIndexAfterRatingDelta } from './rank-decomposition.js';
import { timeAdjustDeltaRating } from './population-constants.js';
import { showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid } from './overlays.js';
// rank_map.json is inlined at build time (like rating_to_rank.json in
// rank-decomposition.js / src/index.js), so it's never fetched at runtime —
// build.mjs drops the served copy from dist. A curve re-fit
// (map_rating_to_egf) is published by rebuilding + deploying, same as the
// rating→rank curve it ships alongside.
import RANK_MAP from '../tsumegos/rank_map.json';

// rankIndexToLabel (internal index → kyu/dan label) and RANK_INDEX_OFFSET (the
// index↔scalar offset) are imported from shared/rank-math.js — one kyu/dan
// ladder, shared with the Worker and the stats UI.

// Clamp translated server ranks to the real rank range (30k..9d) so extreme
// extrapolation can't print impossible labels like "15 dan" (matches the
// our_lo/our_hi clamp the fit used to apply server-side).
const RANK_SCALAR_LO = -29, RANK_SCALAR_HI = 9;

// interpExtrap + rankScalarToLabel now live in shared/rank-math.js (one copy
// shared with the Worker).

// The surveyed servers, in rank_map.json order. EGF is the identity hub but is
// still a key here (serverRanksFor shortcuts it). Consumed by the stats Rank
// tab's server dropdown.
export const RANK_TRANSLATION_SERVERS = Object.keys(RANK_MAP || {});

// internal-rank INDEX (~0..29, floor(rating·29) scale) → that server's CONTINUOUS
// rank scalar (1k=0, 1d=1), clamped to the real rank range. EGF is identity.
// Returns null if the server has no usable curve. This is the scalar form of
// serverRanksFor (which returns rounded labels) — the Rank chart needs the
// continuous value to place the skill line, not a label.
export function internalIndexToServerScalar(rankIndex, server) {
    if (!Number.isFinite(rankIndex)) return null;
    const ourScalar = rankIndex - RANK_INDEX_OFFSET;
    if (server === 'EGF') {
        return Math.max(RANK_SCALAR_LO, Math.min(RANK_SCALAR_HI, ourScalar));
    }
    const c = RANK_MAP && RANK_MAP[server];
    if (!c || !Array.isArray(c.our) || !Array.isArray(c.server) || c.our.length < 2) return null;
    const s = interpExtrap(ourScalar, c.our, c.server);
    return Math.max(RANK_SCALAR_LO, Math.min(RANK_SCALAR_HI, s));
}

// {server: label} translating the (already time-adjusted) CONTINUOUS internal
// rank INDEX through each server's curve via internalIndexToServerScalar — so a
// weak vs strong same-band player can land on different server ranks. EGF is the
// identity hub; its label is passed in (`egfLabel`), the caller deciding whether
// it shows the time-adjusted identity or the raw header rank.
function serverRanksFor(adjRankIndex, egfLabel) {
    const out = {};
    for (const server of RANK_TRANSLATION_SERVERS) {
        if (server === 'EGF') { out.EGF = egfLabel; continue; }
        const s = internalIndexToServerScalar(adjRankIndex, server);
        if (Number.isFinite(s)) out[server] = rankScalarToLabel(s);
    }
    return out;
}

export async function showRankTranslation() {
    // Only meaningful for the user's own rank in solving mode (analysis/linked
    // modes show a problem's rank or nothing).
    if (state.appMode !== APP_MODES.SOLVING) return;
    const rating = state.userRank;
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return;

    const rankValue = rating * TOTAL_RANKS;       // continuous internal-rank index
    const rankIndex = Math.floor(rankValue);      // own-rank band (for the header)
    const key = rankIndexToLabel(rankIndex);

    // The header keeps the raw, untimed onlymove rank; the server translations
    // are TIME-ADJUSTED — practice here is untimed while server ranks reflect
    // timed play, so a slow solver's untimed rank overstates their timed
    // strength. Same single-point correction the Rank-chart server views use
    // (whole-history median solve time → rating delta → mapped through M).
    const deltaRating = timeAdjustDeltaRating(getMergedAnswerHistory(), rankValue);
    const adjusted = deltaRating !== null;
    const adjValue = adjusted ? rankIndexAfterRatingDelta(rankValue, deltaRating) : rankValue;
    const adjIndex = Number.isFinite(adjValue) ? adjValue : rankValue;

    // EGF (identity hub): the time-adjusted identity when adjusted (it then
    // differs from the header, by design); otherwise mirror the header rank so
    // the table stays consistent with the progress line (avoids a floor-vs-round
    // mismatch on the unadjusted identity).
    let egfLabel = key;
    if (adjusted) {
        const egfScalar = internalIndexToServerScalar(adjIndex, 'EGF');
        if (Number.isFinite(egfScalar)) egfLabel = rankScalarToLabel(egfScalar);
    }
    const servers = serverRanksFor(adjIndex, egfLabel);

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
                nextRank: rankIndexToLabel(rankIndex + 1),
            });
            modal.appendChild(progress);

            modal.appendChild(modalTitle(getTerm('rankTranslationTitle')));

            if (servers && Object.keys(servers).length > 0) {
                const grid = modalRankGrid();
                for (const [server, rank] of Object.entries(servers)) {
                    const label = document.createElement('div');
                    label.textContent = server;
                    label.className = 'modal-rank-label';
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
