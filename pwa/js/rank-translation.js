// js/rank-translation.js
//
// Tap your rank → see the equivalent rank on each surveyed go server.
//
// The mapping (pwa/tsumegos/rank_map.json) is produced by
// pipeline.map_rating_to_rank as per-server interpolation knots
// {server: {our, server}} in the shared rank units (shared/rank-math.js). Our
// singular rank scale is the hub: it is coupled to each server by one
// board-independent fit, and to the per-board ratings by the cubic. So the
// translation depends only on our rank — the per-board difference already lives
// in the user's displayed rank itself. We interpolate the user's CONTINUOUS
// internal rank through each server's knots and round only for display, so a
// weak 3d and a strong 3d can translate to different server ranks. Our rank is a
// CONSENSUS; EGF is one server coupling among many (no identity special-case).

import { state, getMergedAnswerHistory } from './state.js';
import { getTerm } from './i18n.js';
import { APP_MODES } from './constants.js';
import { interpExtrap, rankToLabel, clampRankBand, rankProgress } from '../shared/rank-math.js';
import { rankAfterRatingDelta } from './rank-decomposition.js';
import { timeAdjustDeltaRating } from './population-constants.js';
import { showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid } from './overlays.js';
// rank_map.json is inlined at build time (like rating_to_rank.json in
// rank-decomposition.js / src/index.js), so it's never fetched at runtime —
// build.mjs drops the served copy from dist. A curve re-fit
// (map_rating_to_rank) is published by rebuilding + deploying, same as the
// rating→rank curve it ships alongside.
import RANK_MAP from '../tsumegos/rank_map.json';

// rankToLabel (rank → kyu/dan label) and clampRankBand (the real 30k..9d ladder
// bounds, for LABELS only) come from shared/rank-math.js — one kyu/dan ladder,
// shared with the Worker and the stats UI. The translation itself
// (internalRankToServerRank) is never clamped — see its note; the Rank chart needs
// the honest continuous value.

// The surveyed servers, in rank_map.json order (EGF is one of them — a normal
// coupling). Consumed by the stats Rank tab's server dropdown.
export const RANK_TRANSLATION_SERVERS = Object.keys(RANK_MAP || {});

// internal rank → that server's CONTINUOUS rank via its rank_map coupling. Returns
// null if the server has no usable curve. This is the CONTINUOUS translated value —
// the Rank chart needs it to place the skill line, and the translation modal floors
// it to a band + progress.
//
// Always the HONEST translated value — there is no rank-ladder clamp here. A
// hidden cap once lived in this function and silently pinned every above-9d value
// to the 8d-9d border: harmless for a rank LABEL but wrong for the Rank chart,
// which plots problem difficulties that extrapolate far past 9d and relies on its
// own band-focused y-window to drop the out-of-range ones. The label-ceiling cap
// now lives at the one place that builds labels (the modal row, via boundServerScalar).
export function internalRankToServerRank(rank, server) {
    if (!Number.isFinite(rank)) return null;
    const c = RANK_MAP && RANK_MAP[server];
    if (!c || !Array.isArray(c.our) || !Array.isArray(c.server) || c.our.length < 2) return null;
    return interpExtrap(rank, c.our, c.server);
}

export async function showRankTranslation() {
    // Only meaningful for the user's own rank in solving mode (analysis/linked
    // modes show a problem's rank or nothing).
    if (state.appMode !== APP_MODES.SOLVING) return;
    // displayRank, NOT userRank: this modal answers "what is the rank on the
    // badge, on each server", so it must translate the number the badge shows or
    // the two disagree by the whole conservatism offset. Translating the
    // conservative value is self-consistent — the couplings are fitted on the
    // backend from surveys, never from anything the client displays, so no
    // calibration loop closes here.
    const rating = state.displayRank;
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return;

    const history = getMergedAnswerHistory();
    const rankValue = rating;                     // continuous internal rank

    // The 95% CI on the rank now lives in the stats Rank tab (buildRankSummaryText
    // in stats-rank-pane.js), not here.

    // OnlyMove keeps the raw, untimed consensus rank; the server translations —
    // EGF among them — are TIME-ADJUSTED: practice here is untimed while server
    // ranks reflect timed play, so a slow solver's untimed rank overstates their
    // timed strength. Same single-point correction the Rank-chart server views use
    // (whole-history median solve time → rating delta → mapped via M).
    const deltaRating = timeAdjustDeltaRating(history);
    const adjusted = deltaRating !== null;
    const adjValue = adjusted ? rankAfterRatingDelta(rankValue, deltaRating) : rankValue;
    const adjRank = Number.isFinite(adjValue) ? adjValue : rankValue;

    showModal({
        build: (modal, close) => {
            modal.appendChild(modalTitle(getTerm('rankTranslationTitle')));

            // Caption under the header explaining the parenthetical percentage on
            // each row; small + muted so it reads as a note, not a heading.
            const caption = document.createElement('div');
            caption.textContent = getTerm('rankProgressCaption');
            caption.className = 'modal-note';
            modal.appendChild(caption);

            // One system per row: bold name on the left, "<rank> (<progress>%)" on
            // the right. rankProgress (shared/rank-math.js) is THE decimal-rank →
            // {label, percent} decomposition — round band, progress from the band's
            // lower edge, GoR/EGF-consistent. `labelFor` selects the label space
            // (index for OnlyMove, ladder-clamped server-scalar for the servers).
            // The percentage is a smaller, muted span so the rank stays the focus.
            const grid = modalRankGrid();
            const addRow = (name, value, labelFor) => {
                const { label: rankLabel, percent } = rankProgress(value, labelFor);
                const label = document.createElement('div');
                label.textContent = name;
                label.className = 'modal-rank-label';
                const val = document.createElement('div');
                // Left-set (.modal-rank-value): the grid sits inside a
                // text-align:center modal, so without it narrower rows (a
                // 1-digit percent) drift right and open an uneven gap.
                val.className = 'modal-rank-value';
                val.textContent = rankLabel + ' ';
                const pct = document.createElement('span');
                pct.textContent = `(${percent}%)`;
                pct.className = 'modal-rank-pct';
                val.appendChild(pct);
                grid.appendChild(label);
                grid.appendChild(val);
            };

            // OnlyMove is one ranking system among the servers, listed first. Its
            // rank is the raw, untimed value, labelled straight off the shared ladder
            // (no clamp — see internalRankToServerRank).
            addRow('OnlyMove', rankValue, rankToLabel);

            // Then each server: the time-adjusted rank translated through its
            // coupling to a continuous server scalar, labels ladder-clamped.
            const serverLabel = band => rankToLabel(clampRankBand(band));
            for (const server of RANK_TRANSLATION_SERVERS) {
                const s = internalRankToServerRank(adjRank, server);
                if (Number.isFinite(s)) addRow(server, s, serverLabel);
            }

            modal.appendChild(grid);

            modal.appendChild(modalButtonRow([
                modalButton(getTerm('rankTranslationClose'), close),
            ]));
        },
    });
}
