// js/rank-translation.js
//
// Tap your rank → see the equivalent rank on each surveyed go server.
//
// The mapping (pwa/tsumegos/rank_map.json) is produced by
// pipeline.map_rating_to_rank as per-server interpolation knots
// {server: {our, server}} in rank-scalar units (1k=0, 1d=1; no zero gap). Our
// singular rank scale is the hub: it is coupled to each server by one
// board-independent fit, and to the per-board ratings by the cubic. So the
// translation depends only on our rank — the per-board difference already lives
// in the user's displayed rank itself. We interpolate the user's CONTINUOUS
// internal rank through each server's knots and round only for display, so a
// weak 3d and a strong 3d can translate to different server ranks. Our rank is a
// CONSENSUS; EGF is one server coupling among many (no identity special-case).

import { state, getMergedAnswerHistory } from './state.js';
import { getTerm } from './i18n.js';
import { TOTAL_RANKS, APP_MODES } from './constants.js';
import { interpExtrap, rankScalarToLabel, rankIndexToLabel, RANK_INDEX_OFFSET } from '../shared/rank-math.js';
import { rankIndexAfterRatingDelta, estimateWindowRank } from './rank-decomposition.js';
import { timeAdjustDeltaRating } from './population-constants.js';
import { showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid } from './overlays.js';
// rank_map.json is inlined at build time (like rating_to_rank.json in
// rank-decomposition.js / src/index.js), so it's never fetched at runtime —
// build.mjs drops the served copy from dist. A curve re-fit
// (map_rating_to_rank) is published by rebuilding + deploying, same as the
// rating→rank curve it ships alongside.
import RANK_MAP from '../tsumegos/rank_map.json';

// rankIndexToLabel (internal index → kyu/dan label) and RANK_INDEX_OFFSET (the
// index↔scalar offset) are imported from shared/rank-math.js — one kyu/dan
// ladder, shared with the Worker and the stats UI.

// Real rank-ladder bounds (30k..9d). Used ONLY to clamp a translated server rank
// for a displayed LABEL (boundServerScalar) so extreme extrapolation can't print
// an impossible label like "15 dan" on a server whose ladder tops out at 9d. The
// translation itself (internalIndexToServerScalar) is never clamped — see its
// note; the Rank chart needs the honest continuous value.
const RANK_SCALAR_LO = -29, RANK_SCALAR_HI = 9;

// Clamp a translated server scalar to the real rank ladder, for a displayed LABEL.
function boundServerScalar(s) {
    return Math.max(RANK_SCALAR_LO, Math.min(RANK_SCALAR_HI, s));
}

// interpExtrap + rankScalarToLabel now live in shared/rank-math.js (one copy
// shared with the Worker).

// The surveyed servers, in rank_map.json order (EGF is one of them — a normal
// coupling). Consumed by the stats Rank tab's server dropdown.
export const RANK_TRANSLATION_SERVERS = Object.keys(RANK_MAP || {});

// internal-rank INDEX (~0..29, floor(rating·29) scale) → that server's CONTINUOUS
// rank scalar (1k=0, 1d=1) via its rank_map coupling. Returns null if the server has
// no usable curve. This is the scalar form of serverRanksFor (which returns rounded
// labels) — the Rank chart needs the continuous value to place the skill line.
//
// Always the HONEST translated value — there is no rank-ladder clamp here. A
// hidden cap once lived in this function and silently pinned every above-9d value
// to the 8d-9d border: harmless for a rank LABEL but wrong for the Rank chart,
// which plots problem difficulties that extrapolate far past 9d and relies on its
// own band-focused y-window to drop the out-of-range ones. The label-ceiling cap
// now lives at the one place that builds labels (serverRanksFor → boundServerScalar).
export function internalIndexToServerScalar(rankIndex, server) {
    if (!Number.isFinite(rankIndex)) return null;
    const ourScalar = rankIndex - RANK_INDEX_OFFSET;
    const c = RANK_MAP && RANK_MAP[server];
    if (!c || !Array.isArray(c.our) || !Array.isArray(c.server) || c.our.length < 2) return null;
    return interpExtrap(ourScalar, c.our, c.server);
}

// {server: label} translating the (already time-adjusted) CONTINUOUS internal
// rank INDEX through each server's curve via internalIndexToServerScalar — so a
// weak vs strong same-band player can land on different server ranks. EGF is one
// of these couplings now, timed like the rest (no identity special-case).
function serverRanksFor(adjRankIndex) {
    const out = {};
    for (const server of RANK_TRANSLATION_SERVERS) {
        const s = internalIndexToServerScalar(adjRankIndex, server);
        if (Number.isFinite(s)) out[server] = rankScalarToLabel(boundServerScalar(s));
    }
    return out;
}

export async function showRankTranslation() {
    // Only meaningful for the user's own rank in solving mode (analysis/linked
    // modes show a problem's rank or nothing).
    if (state.appMode !== APP_MODES.SOLVING) return;
    const rating = state.userRank;
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return;

    const history = getMergedAnswerHistory();
    const rankValue = rating * TOTAL_RANKS;       // continuous internal-rank index
    const rankIndex = Math.floor(rankValue);      // own-rank band (for the header)
    const key = rankIndexToLabel(rankIndex);

    // 95% CI half-width on the displayed rank, from the window-MLE posterior
    // standard error. `se` is already in internal-rank-index units (= stones =
    // ranks; see shared/rank-window.js), so 1.96·se is directly "in ranks" — no
    // /TOTAL_RANKS or logit conversion. Warm-start the refit at the current rank
    // so it reproduces the front-page estimate. null until there's usable history
    // (the CI line is then omitted).
    const rankEst = estimateWindowRank(history, rankValue);
    const ciHalfRanks = rankEst && Number.isFinite(rankEst.se) ? 1.96 * rankEst.se : null;

    // The header keeps the raw, untimed onlymove (consensus) rank; the server
    // translations — EGF among them — are TIME-ADJUSTED: practice here is untimed
    // while server ranks reflect timed play, so a slow solver's untimed rank
    // overstates their timed strength. Same single-point correction the Rank-chart
    // server views use (whole-history median solve time → rating delta → mapped via M).
    const deltaRating = timeAdjustDeltaRating(history);
    const adjusted = deltaRating !== null;
    const adjValue = adjusted ? rankIndexAfterRatingDelta(rankValue, deltaRating) : rankValue;
    const adjIndex = Number.isFinite(adjValue) ? adjValue : rankValue;

    const servers = serverRanksFor(adjIndex);

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

            // Second line: the 95% CI on the rank, phrased as a continuation of the
            // progress line above ("…, with a 95% confidence interval of ± x.xx
            // ranks"). Omitted when the posterior SE isn't available yet.
            if (ciHalfRanks !== null) {
                const ci = document.createElement('div');
                ci.style.textAlign = 'center';
                ci.style.marginBottom = '1vmin';
                ci.style.fontSize = '0.85em';
                ci.style.opacity = '0.75';
                ci.textContent = getTerm('rankConfidenceFormat', {
                    ci: ciHalfRanks.toFixed(2),
                });
                modal.appendChild(ci);
            }

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
