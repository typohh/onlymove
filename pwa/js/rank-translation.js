// js/rank-translation.js
//
// Tap your rank → see the equivalent rank on each surveyed go server, in a hint
// bubble beside the badge (js/hint-bubble.js). A bubble rather than a dialog
// because this is a READING of the thing the user pointed at, not a task: it
// wants no decision, no dismissal and nothing behind it hidden.
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
import { modalRankGrid } from './overlays.js';
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

/**
 * A shown rank as it reads on every server: `rank` time-adjusted for the solving
 * speed in `history`, then translated through each coupling. The same per-server
 * value the badge bubble labels, so the result overlay's server crossings
 * (attempt-recorder) name exactly the rows whose label in that bubble changed.
 * Servers with no usable curve are absent.
 */
export function serverRanksFor(rank, history) {
    const out = {};
    if (!Number.isFinite(rank)) return out;
    const deltaRating = timeAdjustDeltaRating(history);
    const adjValue = deltaRating !== null ? rankAfterRatingDelta(rank, deltaRating) : rank;
    const adjRank = Number.isFinite(adjValue) ? adjValue : rank;
    for (const server of RANK_TRANSLATION_SERVERS) {
        const v = internalRankToServerRank(adjRank, server);
        if (Number.isFinite(v)) out[server] = v;
    }
    return out;
}

/**
 * The bubble body for the rank badge: one row per ranking system, OnlyMove
 * first. Returns null wherever the badge is not the user's own rank — analysis
 * and view mode show a problem's rank or nothing — which the hint anchor reads
 * as "nothing to say here", so the badge is simply inert in those modes.
 */
export function rankTranslationContent() {
    if (state.appMode !== APP_MODES.SOLVING) return null;
    // displayRank, NOT userRank: this answers "what is the rank on the badge, on
    // each server", so it must translate the number the badge shows or the two
    // disagree by the whole conservatism offset. Translating the conservative
    // value is self-consistent — the couplings are fitted on the backend from
    // surveys, never from anything the client displays, so no calibration loop
    // closes here.
    const rating = state.displayRank;
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return null;

    const history = getMergedAnswerHistory();
    const rankValue = rating;                     // continuous internal rank

    // The 95% CI on the rank lives in the stats Rank tab (buildRankSummaryText
    // in stats-rank-pane.js), not here.

    // OnlyMove keeps the raw, untimed consensus rank; the server translations —
    // EGF among them — are TIME-ADJUSTED: practice here is untimed while server
    // ranks reflect timed play, so a slow solver's untimed rank overstates their
    // timed strength. Same single-point correction the Rank-chart server views use
    // (whole-history median solve time → rating delta → mapped via M).
    const serverRanks = serverRanksFor(rankValue, history);

    const frag = document.createDocumentFragment();

    const title = document.createElement('div');
    title.className = 'ui-bubble-title';
    title.textContent = getTerm('rankTranslationTitle');
    frag.appendChild(title);

    // Explains the parenthetical percentage on each row; small + muted so it
    // reads as a note, not a heading.
    const caption = document.createElement('div');
    caption.className = 'modal-note';
    caption.textContent = getTerm('rankProgressCaption');
    frag.appendChild(caption);

    // One system per row: bold name on the left, "<rank> (<progress>%)" on the
    // right. rankProgress (shared/rank-math.js) is THE decimal-rank →
    // {label, percent} decomposition — round band, progress from the band's
    // lower edge, GoR/EGF-consistent. `labelFor` selects the label space (index
    // for OnlyMove, ladder-clamped server-scalar for the servers). The
    // percentage is a smaller, muted span so the rank stays the focus.
    const grid = modalRankGrid();
    const addRow = (name, value, labelFor) => {
        const { label: rankLabel, percent } = rankProgress(value, labelFor);
        const label = document.createElement('div');
        label.textContent = name;
        label.className = 'modal-rank-label';
        const val = document.createElement('div');
        val.className = 'modal-rank-value';
        val.textContent = rankLabel + ' ';
        const pct = document.createElement('span');
        pct.textContent = `(${percent}%)`;
        pct.className = 'modal-rank-pct';
        val.appendChild(pct);
        grid.appendChild(label);
        grid.appendChild(val);
    };

    // OnlyMove is one ranking system among the servers, listed first. Its rank is
    // the raw, untimed value, labelled straight off the shared ladder (no clamp —
    // see internalRankToServerRank).
    addRow('OnlyMove', rankValue, rankToLabel);

    // Then each server: the time-adjusted rank translated through its coupling to
    // a continuous server scalar, labels ladder-clamped.
    const serverLabel = band => rankToLabel(clampRankBand(band));
    for (const [server, v] of Object.entries(serverRanks)) addRow(server, v, serverLabel);
    frag.appendChild(grid);
    return frag;
}
