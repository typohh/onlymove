// js/stats-rank-pane.js
//
// The Rank pane of the detailed-stats modal (stats-ui.js is the orchestrator):
// the rating-history chart with its server-translation dropdown and reactive
// current-rank readout, plus the rank-decomposition summary text. Both take the
// rank decomposition precomputed ONCE by showDetailedStats (no fallback here).

import { state } from './state.js';
import { getTerm } from './i18n.js';
import { evaluateOnGrid, rankAfterRatingDelta, windowRankSeries, estimateWindowRank, windowMedianSolveMinutesSeries } from './rank-decomposition.js';
import { getPopulationConstants, TIME_ADJUST_MIN_TIMED } from './population-constants.js';
import { RANK_TRANSLATION_SERVERS, internalRankToServerRank } from './rank-translation.js';
import { rankToLabel, rankProgress } from '../shared/rank-math.js';
import {
    svgEl, makeChartSvg, hGridLine, buildLegend,
    hideOverlappingYTicks, observeResize,
    makeChartFrame, buildYAxis, buildPlotRow, chartViewBoxHeight,
} from './chart-svg.js';
import { getLogitPerRank, MODEL_INDICES } from './recall-model.js';

// medianNovelSolveMinutes + the TIME_ADJUST_* feature constants + the scalar
// timeAdjustDeltaRating live in population-constants.js (one definition, shared with
// the rank-translation modal). The Rank-tab chart's per-time median uses
// windowMedianSolveMinutesSeries (rank-decomposition.js) — the exact time counterpart
// of windowRankSeries, over the same positional window — so the dock matches the rank.

// The rank decomposition text (long-term skill + drift + session swing) and
// the relative 13x13 readout, built as part of the Rank tab pane
// (computeRankDecomposition is heavy, but the whole
// modal is constructed up front under the "constructing charts" overlay). Reads
// state.statsResults (already computed by the time this runs).
export function buildRankSummaryText(history, decomp) {
    let rankText = '';

    // 95% CI on the front-page rank, from the window-MLE posterior standard error.
    // `se` is already in internal-rank-index units (= stones = ranks; see
    // shared/rank-window.js), so 1.96·se is directly "in ranks". Warm-start the
    // refit at the current rank so it reproduces the front-page estimate. Sits
    // first, right under the reactive current-rank readout (this line used to live
    // in the front-page rank-translation modal). Omitted until there's usable
    // history / a valid current rank.
    const rankValue = (typeof state.userRank === 'number' && Number.isFinite(state.userRank))
        ? state.userRank : null;
    const rankEst = rankValue !== null ? estimateWindowRank(history, rankValue) : null;
    if (rankEst && Number.isFinite(rankEst.se)) {
        rankText += getTerm('rankConfidenceFormat', { ci: (1.96 * rankEst.se).toFixed(2) }) + '\n';
    }

    // Two-component rank decomposition (R = skill S + form M). Gives a
    // long-term-trend number and a session-to-session swing number that can't
    // be read off the single-line smoother in the chart. See
    // rank-decomposition.js for the model and the backfitting approach. `decomp`
    // is computed once for the whole Rank pane and passed in.
    if (decomp) {
        const { summary } = decomp;
        // The current rank line (raw onlymove rank or the selected server's
        // time-adjusted rank) lives below the CHART, where it reacts to the
        // server dropdown — see the readout in buildRatingHistoryChart. Here we
        // keep only the variance/size readouts the chart can't show.
        if (typeof summary.sigmaM === 'number') {
            rankText += summary.sigmaM >= 0.1
                ? getTerm('statsSessionSwing', {
                    up: summary.sigmaUp.toFixed(2),
                    down: summary.sigmaDown.toFixed(2),
                }) + '\n'
                : getTerm('statsSessionNoSwing', {
                    upper: summary.sigmaMUpper.toFixed(2),
                }) + '\n';
        }
    }

    // Relative 13x13 skill. The recall model's SIZE_IS_13 coefficient is the
    // logit offset for 13x13 at equal internal-rank difficulty (9x9 is the
    // baseline), so coeff / logit-per-rank is the skill gap in ranks. Only
    // shown once both sizes have enough non-repeat data to identify it.
    const results = state.statsResults?.results || [];
    const MIN_SIZE_COMPARISON = 20;
    const count9 = history.filter(p => !p.isRepeat && p.size === 9 && p.problemRank !== undefined).length;
    const count13 = history.filter(p => !p.isRepeat && p.size === 13 && p.problemRank !== undefined).length;
    const sizeResult = results.find(r => r.varIndex === MODEL_INDICES.SIZE_IS_13);
    if (sizeResult && count9 >= MIN_SIZE_COMPARISON && count13 >= MIN_SIZE_COMPARISON
        && Number.isFinite(sizeResult.coeff) && Number.isFinite(sizeResult.se) && sizeResult.se > 0) {
        const logitPerRank = getLogitPerRank(state.statsResults);
        const ranksDiff = sizeResult.coeff / logitPerRank;   // + = stronger on 13x13
        const seRanks = sizeResult.se / logitPerRank;
        rankText += getTerm(ranksDiff >= 0 ? 'statsSizeStronger' : 'statsSizeWeaker')
            .replace('{amount}', Math.abs(ranksDiff).toFixed(1))
            .replace('{se}', seRanks.toFixed(1)) + '\n';
    }

    return rankText;
}

export function buildRatingHistoryChart(history = state.answerHistory, decomp) {
    const entries = history
        .filter(a => typeof a.problemRank === 'number' && typeof a.date === 'number'
                     && !a.isRepeat && !a.linked)
        .map(a => ({ date: a.date, rankValue: a.problemRank, correct: !!a.correct }));
    if (entries.length < 2) return null;

    // Two-kernel rank decomposition: see rank-decomposition.js. We evaluate the
    // skill smoother Ŝ(t), its measurement SE seS(t), and the time-varying
    // performance swing σ_M(t) on a regular t-grid, so line and bands shape
    // correctly across gaps. The skill line + inner band are read via
    // evaluateOnGrid, which interpolates S/seS from the decomp's precomputed
    // wideGrid (the 30-day kernel is locally linear over the grid spacing) rather
    // than re-running the wide MLE at each of the 300 points; σ_M(t) is still an
    // exact per-point estimate.
    const t0 = entries[0].date;
    const t1 = entries[entries.length - 1].date;
    const tSpan = Math.max(1, t1 - t0);
    // `decomp` is passed in by the Rank pane (computed once for both chart +
    // summary).
    const GRID_POINTS = 300;
    const Z = 1.96;
    // Half-widths in rank space: inner band = measurement uncertainty (how well
    // we know your skill), symmetric. Outer band = a prediction interval (where a
    // given session's performance lands); the form swing is directional, so the
    // outer band is ASYMMETRIC — good-day reach above, bad-day dip below — each
    // added to the measurement SE in quadrature.
    const innerHalf = e => Z * e.seS;
    const outerUp = e => Z * Math.hypot(e.seS, e.sigmaUp);
    const outerDown = e => Z * Math.hypot(e.seS, e.sigmaDown);
    let gridEvals = [];
    if (decomp) {
        for (let i = 0; i < GRID_POINTS; i++) {
            const t = Math.round(t0 + tSpan * (i / (GRID_POINTS - 1)));
            const e = evaluateOnGrid(decomp, t);
            if (e) gridEvals.push({ t, S: e.S, seS: e.seS, sigmaUp: e.sigmaUp, sigmaDown: e.sigmaDown });
        }
    }

    // Per-grid time-adjustment shift (RATING units), used only for the timed
    // (non-onlymove) server views: the rank line is corrected for the user's typical
    // solving time AT THAT POINT IN TIME (medMin varies over history, so the
    // correction is temporal) — practice here is untimed while server ranks reflect
    // timed play. The median at each grid time is windowMedianSolveMinutesSeries: the
    // SAME trailing WIN_CAP window and (M − age)/M positional weights as
    // windowRankSeries, over the SAME decomp.attempts — so the dock and the rank it
    // docks share one weighted sample (no temporal-kernel-vs-count-window mismatch;
    // this replaced the old makeMedianSolveMinutesAt Gaussian-in-time smoother). alpha
    // + center are population constants (map_rating_to_rank via fit_population_constants).
    // delta = 0 (constants or timed data unavailable) ⇒ the timed view is pure rank
    // translation with no time correction.
    const ta = getPopulationConstants()?.time_adjust;
    const haveTimeAdjust = !!(ta && Number.isFinite(ta.alpha_per_log10_min)
        && Number.isFinite(ta.center_log10_min));
    const medSeries = haveTimeAdjust
        ? windowMedianSolveMinutesSeries(decomp, gridEvals.map(e => e.t), TIME_ADJUST_MIN_TIMED)
        : null;
    for (let i = 0; i < gridEvals.length; i++) {
        const e = gridEvals[i];
        let d = 0;
        if (medSeries) {
            const m = medSeries[i];
            if (Number.isFinite(m) && m > 0) {
                d = ta.alpha_per_log10_min * (Math.log10(m) - ta.center_log10_min);
            }
        }
        e.delta = d;
    }

    // Front-page rank over time: the live headline's window MLE reconstructed
    // across history (windowRankSeries), evaluated at the SAME grid times as the
    // skill line so each point reuses that time's display transform + time-adjust
    // delta. In internal-rank index space (onlymove); its right endpoint equals
    // the current front-page rank. This is a DIFFERENT, more reactive estimator
    // than the slow skill line S(t) — plotted so the user can see the actual
    // displayed rank's trajectory, not just the smoothed skill trend.
    const winSeries = windowRankSeries(decomp, gridEvals.map(e => e.t));
    for (let i = 0; i < gridEvals.length; i++) gridEvals[i].win = winSeries[i];

    // Server selector — sits INSIDE the chart title ("[server ▾] rank since
    // <date>"). "OnlyMove" shows the raw, untimed internal rank (no time
    // adjustment). Every other option shows the rank as it
    // would read ON THAT SERVER: the internal skill TIME-ADJUSTED for the user's
    // solving speed, then translated through that server's rank curve. Only
    // servers with a usable translation curve are offered. EGF is the identity
    // hub, so "EGF" vs "OnlyMove" differ only by the time adjustment.
    const ONLYMOVE = 'OnlyMove';
    const serverOptions = [ONLYMOVE, ...RANK_TRANSLATION_SERVERS.filter(
        s => Number.isFinite(internalRankToServerRank(-0.5, s)))];   // probe at 1k
    let selectedServer = ONLYMOVE;
    const select = document.createElement('select');
    select.className = 'stats-rank-server-select';
    for (const s of serverOptions) {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;   // option value IS the display text ("OnlyMove", "OGS", …)
        select.appendChild(opt);
    }
    select.value = selectedServer;

    const container = makeChartFrame([select, document.createTextNode(
        ' ' + getTerm('statsChartRankSince', { date: new Date(t0).toISOString().slice(0, 10) }))]);

    // Display transforms for the selected server. The skill line + its band edges
    // are SKILL values, so they take the (temporal) time adjustment then translate;
    // the scatter dots are PROBLEM difficulties (independent of how fast the user
    // solves), so they translate only. onlymove = identity (the raw internal rank).
    // internalRankToServerRank returns the HONEST (unclamped) server rank:
    // the dots are problem difficulties that extrapolate past 9d, and the
    // band-focused y-window below drops the out-of-range ones. (A rank-ladder cap
    // would instead pile every above-9d dot onto the 8d-9d border — only the
    // translation-table LABELS clamp; see rank-translation.js.)
    const skillToDisplay = (internalRank, delta) => {
        if (selectedServer === ONLYMOVE) return internalRank;
        const adj = (Number.isFinite(delta) && delta !== 0)
            ? rankAfterRatingDelta(internalRank, delta) : internalRank;
        return internalRankToServerRank(Number.isFinite(adj) ? adj : internalRank, selectedServer);
    };
    const rankToDisplay = (internalRank) =>
        selectedServer === ONLYMOVE ? internalRank
            : internalRankToServerRank(internalRank, selectedServer);
    // Band → label. One scale now, so onlymove and the servers share one labeller.
    const displayLabel = rankToLabel;

    // The plot (y-axis + svg + x-date) is rebuilt into this holder on every
    // server change; the title, readout and legend persist around it.
    const plotHolder = document.createElement('div');

    // Current-rank readout below the graph: the selected server's current rank
    // (raw for onlymove, time-adjusted + translated otherwise). Reacts to the
    // dropdown. Same element/class as the decomposition text (stats-modal-pre)
    // so it reads
    // as just another line of that text, not a distinct callout.
    const readout = document.createElement('pre');
    readout.className = 'stats-modal-pre stats-rank-readout';

    // Legend (server-independent text); skill curve + bands listed only when the
    // decomposition produced them.
    const SKILL_FILL_OUTER = 'rgba(59,130,246,0.10)';
    const SKILL_FILL_INNER = 'rgba(59,130,246,0.22)';
    const SKILL_STROKE = '#2563eb';
    // Front-page (window) rank line: amber, distinct from the blue skill line.
    const FRONTPAGE_STROKE = '#f59e0b';
    const DOT_CORRECT = '#50b450', DOT_WRONG = '#dc5050';
    const legendItems = [
        { kind: 'dot', color: DOT_CORRECT, text: getTerm('statsLegendCorrect') },
        { kind: 'dot', color: DOT_WRONG, text: getTerm('statsLegendWrong') },
    ];
    if (gridEvals.length >= 2) {
        legendItems.push(
            { kind: 'line', color: SKILL_STROKE, text: getTerm('statsLegendRank') },
            { kind: 'line', color: FRONTPAGE_STROKE, text: getTerm('statsLegendFrontPageRank') },
            { kind: 'band', color: SKILL_FILL_INNER, text: getTerm('statsLegend95Confidence') },
            { kind: 'band', color: SKILL_FILL_OUTER, text: getTerm('statsLegendDailyVariance') },
        );
    }
    const legend = buildLegend(legendItems);

    // Rebuild the plot for the current selectedServer. Everything below is in
    // DISPLAY space (the internal rank for onlymove, that server's own rank
    // otherwise); the band edges are transformed individually so the monotone
    // adjust+translate keeps them ordered across server-curve knots.
    let labelObserver = null;
    const renderPlot = () => {
        const gd = [];
        for (const e of gridEvals) {
            const S = skillToDisplay(e.S, e.delta);
            const up = skillToDisplay(e.S + outerUp(e), e.delta);
            const dn = skillToDisplay(e.S - outerDown(e), e.delta);
            const iu = skillToDisplay(e.S + innerHalf(e), e.delta);
            const id = skillToDisplay(e.S - innerHalf(e), e.delta);
            if ([S, up, dn, iu, id].every(Number.isFinite)) gd.push({ t: e.t, S, up, dn, iu, id });
        }

        // Front-page rank line in display space (same transform + per-time delta
        // as the skill line, so it reacts to the server dropdown identically).
        const winPts = [];
        for (const e of gridEvals) {
            if (typeof e.win !== 'number' || !Number.isFinite(e.win)) continue;
            const y = skillToDisplay(e.win, e.delta);
            if (Number.isFinite(y)) winPts.push({ t: e.t, y });
        }

        // Readout = the FRONT-PAGE rank (the amber line's right endpoint = the
        // window MLE the home page shows) in the selected server's scale — so the
        // text matches the front page for OnlyMove (time-adjusted + translated for
        // other servers). Hidden when there's too little data for the line.
        if (winPts.length >= 1) {
            const cur = winPts[winPts.length - 1].y;
            // Shared decimal-rank → {label, percent, nextLabel} (band floor,
            // progress from the band's lower edge; GoR/EGF-consistent).
            const { label, percent, nextLabel } = rankProgress(cur, displayLabel);
            readout.textContent = getTerm('statsSelectedServerRank', {
                server: selectedServer,
                cur: label,
                pct: percent,
                next: nextLabel,
            });
            readout.hidden = false;
        } else {
            readout.hidden = true;
        }

        const dots = [];
        for (const e of entries) {
            const y = rankToDisplay(e.rankValue);
            if (Number.isFinite(y)) dots.push({ t: e.date, y, correct: e.correct });
        }

        // Focus the y-range on the skill band — not the (far wider) spread of
        // problem difficulties. Dots outside this window are dropped below.
        let bandMin = Infinity, bandMax = -Infinity;
        for (const g of gd) { bandMin = Math.min(bandMin, g.dn); bandMax = Math.max(bandMax, g.up); }
        // The front-page line deliberately does NOT widen the y-range — the window
        // focuses on the skill band, and the front-page line clips at the chart
        // edge where it wanders outside it.
        if (!Number.isFinite(bandMin)) {
            // No decomposition — fall back to the scatter range.
            const ys = dots.map(d => d.y).slice().sort((a, b) => a - b);
            bandMin = ys.length ? ys[0] : 0;
            bandMax = ys.length ? ys[ys.length - 1] : 1;
        }
        const Y_PAD = 1; // exactly one rank beyond the outer CI band
        let yHigh = bandMax + Y_PAD;
        let yLow = bandMin - Y_PAD;
        if (yHigh <= yLow) yHigh = yLow + 1;

        // Grid lines mark the rank CUTOFFS (round-band boundaries, at the
        // half-integers); each label sits AT the integer band center between two
        // cutoffs, so it reads as "this band is rank X". Round bands (rank n spans
        // [n-0.5, n+0.5)) match the GoR/EGF convention and getDescriptiveRank.
        // Overlapping labels get hidden after layout (see below).
        const gridLines = [];
        for (let n = Math.floor(yLow); n <= Math.ceil(yHigh); n++) {
            const cut = n + 0.5;
            if (cut >= yLow && cut <= yHigh) gridLines.push(cut);
        }
        const labelTicks = [];
        for (let n = Math.ceil(yLow); n <= Math.floor(yHigh); n++) {
            labelTicks.push({ pos: n, text: displayLabel(n) });
        }

        const topPct = v => (1 - (v - yLow) / (yHigh - yLow)) * 100;
        const { yaxis } = buildYAxis('20k',
            labelTicks.map(t => ({ top: topPct(t.pos), text: t.text })));

        const W = 400, H = chartViewBoxHeight(W);
        const xOf = t => ((t - t0) / tSpan) * W;
        const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;
        const svg = makeChartSvg(W, H);

        for (const r of gridLines) svg.appendChild(hGridLine(yOf(r), W));

        // Per-attempt scatter: green=correct, red=wrong, drawn UNDER the bands so
        // the curve always reads on top. Cap the rendered count (stride) and fade
        // opacity by the count actually INSIDE the window (most dots clip out, as
        // problem ranks scatter far wider than the band).
        const visibleDots = dots.filter(d => d.y >= yLow && d.y <= yHigh);
        const SCATTER_CAP = 1500;
        const stride = Math.max(1, Math.ceil(visibleDots.length / SCATTER_CAP));
        const shownDots = stride > 1 ? visibleDots.filter((_, i) => i % stride === 0) : visibleDots;
        const dotR = shownDots.length > 600 ? '1.4' : '1.6';
        const dotAlpha = Math.max(0.15, Math.min(0.6, 50 / Math.max(1, shownDots.length))).toFixed(3);
        for (const d of shownDots) {
            svg.appendChild(svgEl('circle', {
                cx: xOf(d.t).toFixed(1), cy: yOf(d.y).toFixed(1), r: dotR,
                fill: d.correct ? DOT_CORRECT : DOT_WRONG, 'fill-opacity': dotAlpha,
            }));
        }

        // Two nested bands: outer (lighter) = prediction interval (where a
        // session's form lands), inner (darker) = measurement uncertainty. Each
        // edge is already in display space; top runs left→right, bottom closes it
        // right→left.
        const bandPath = (top, bot) => {
            let d = '';
            for (let i = 0; i < gd.length; i++) d += (i === 0 ? 'M' : 'L') + `${xOf(gd[i].t).toFixed(1)} ${yOf(gd[i][top]).toFixed(1)} `;
            for (let i = gd.length - 1; i >= 0; i--) d += `L${xOf(gd[i].t).toFixed(1)} ${yOf(gd[i][bot]).toFixed(1)} `;
            return d + 'Z';
        };
        if (gd.length >= 2) {
            svg.appendChild(svgEl('path', { d: bandPath('up', 'dn'), fill: SKILL_FILL_OUTER, stroke: 'none' }));
            svg.appendChild(svgEl('path', { d: bandPath('iu', 'id'), fill: SKILL_FILL_INNER, stroke: 'none' }));
            const skillPts = gd.map(g => `${xOf(g.t).toFixed(1)},${yOf(g.S).toFixed(1)}`).join(' ');
            svg.appendChild(svgEl('polyline', {
                points: skillPts, fill: 'none', stroke: SKILL_STROKE,
                'stroke-width': '1.5', 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke',
            }));
        }

        // Front-page rank line drawn last (on top), solid. No band — it's the
        // point estimate the headline shows.
        if (winPts.length >= 2) {
            const winLine = winPts.map(w => `${xOf(w.t).toFixed(1)},${yOf(w.y).toFixed(1)}`).join(' ');
            svg.appendChild(svgEl('polyline', {
                points: winLine, fill: 'none', stroke: FRONTPAGE_STROKE,
                'stroke-width': '1.5', 'stroke-linejoin': 'round',
                'vector-effect': 'non-scaling-stroke',
            }));
        }

        const plotRow = buildPlotRow(yaxis, svg);   // no x-axis end-date label

        // Greedy-hide y-axis labels that would physically overlap; re-runs on
        // resize. Re-created per render, so disconnect the prior observer first.
        if (labelObserver) { labelObserver.disconnect(); labelObserver = null; }
        labelObserver = observeResize(svg, () => hideOverlappingYTicks(svg, yaxis));

        plotHolder.replaceChildren(plotRow);
    };

    select.addEventListener('change', () => {
        selectedServer = select.value;
        renderPlot();
    });
    renderPlot();

    container.append(plotHolder, legend);
    // The readout is NOT appended here — it's returned so the Rank pane can place
    // it in the text section (below the chart). renderPlot still drives it via the
    // shared reference, so it stays reactive to the dropdown wherever it's placed.
    return { chart: container, readout };
}
