// js/stats-time-chart.js
//
// The Time pane's chart in the detailed-stats modal (stats-ui.js is the
// orchestrator; it pairs this chart with the timing-probability text). The
// kernel-quantile smoother itself lives in stats-compute.js
// (kernelQuantileGrid); this module is rendering-only.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import { quantileSorted } from './math-utils.js';
import {
    svgEl, makeChartSvg, gridLine, buildLegend,
    hideOverlappingYTicks, observeResize,
    makeChartFrame, buildYAxis, buildPlotRow, chartViewBoxHeight, niceStep,
} from './chart-svg.js';
import { kernelQuantileGrid } from './stats-compute.js';

// Daily time spent solving and reviewing, smoothed. A Gaussian-kernel smoother
// over the ACTUAL attempt timestamps gives the trend line: a day with no
// attempts contributes nothing and is carried by its neighbours rather than
// pulled to zero, and stretches with no attempts within the kernel's reach
// leave a GAP in the line (we never invent a zero where there is no data). The
// band is the SPREAD of individual attempts, not a confidence interval for the
// line: at each date it's the kernel-weighted 25%/75% quantiles of the
// nearby attempts' times — i.e. "the middle 50% of attempts land in here" (the
// interquartile range, read off the empirical
// skewed distribution, not a normal fit). This
// is a direct weighted empirical quantile (sort the window, walk the cumulative
// kernel weight to the target), so no bootstrap/resampling is involved; the
// heavy idle-time right tail just lands in the upper quantile naturally. The
// y-window clips to a robust quantile of the upper edge so one idle spell can't
// flatten the rest.
export function buildTimeUsageChart(history = state.answerHistory) {
    const DAY_MS = 24 * 3600 * 1000;

    const seriesAttempts = (valueOf) => {
        const out = [];
        for (const a of history) {
            if (typeof a.date !== 'number') continue;
            const v = valueOf(a);
            if (v === null) continue;
            out.push({ t: a.date, v });
        }
        out.sort((p, q) => p.t - q.t);
        return out;
    };
    // Solving time is logged on every attempt. Review time accrues only when the
    // user actually reviews, so the review series keeps attempts with positive
    // reviewTime — its distribution is "how long a review takes", not diluted to
    // near-zero by the many attempts that were never reviewed. Values in seconds.
    const solving = seriesAttempts(a => (typeof a.timeTaken === 'number' && a.timeTaken >= 0) ? a.timeTaken / 1000 : null);
    const review = seriesAttempts(a => (typeof a.reviewTime === 'number' && a.reviewTime > 0) ? a.reviewTime / 1000 : null);

    const all = [...solving, ...review];
    if (all.length < 4) return null;
    const t0 = Math.min(...all.map(p => p.t));
    const t1 = Math.max(...all.map(p => p.t));
    const tSpan = t1 - t0;
    if (tSpan < 0.75 * DAY_MS) return null; // too short a span to read as a daily trend

    // Kernel-weighted median line + a weighted-quantile spread band on a regular
    // t-grid (kernelQuantileGrid in stats-compute.js): an array of
    // {t, mid, lo, hi}, null at grid points with too little nearby data (those
    // break the line).
    //
    // Bandwidth is a fixed FRACTION of the visible span, so the smoothing scale
    // is relative to the GRAPH, not to an absolute number of days. The x-axis
    // always spans t0..t1 across the same pixel width, so a kernel of bw=tSpan/N
    // covers a constant slice of the width at any span — the curve reads with the
    // same visual smoothness whether it's a month or three years of history. The
    // 3-day floor keeps a short history from over-smoothing to a flat line; there
    // is deliberately no upper cap — an absolute-days cap would shrink to a sliver
    // of a long-time user's multi-year axis and leave the line jagged.
    const G = 140, ESS_MIN = 1.5;
    const bw = Math.max(3 * DAY_MS, tSpan / 36);
    const smooth = (pts) => kernelQuantileGrid(pts, { t0, tSpan, bw, gridPoints: G, essMin: ESS_MIN });

    // Break a smoothed grid into runs of consecutive present points so the line
    // and band stop across no-data gaps instead of bridging them.
    const segments = (grid) => {
        const runs = [];
        let run = [];
        for (const e of grid) {
            if (e) run.push(e);
            else { if (run.length) runs.push(run); run = []; }
        }
        if (run.length) runs.push(run);
        return runs;
    };

    const sGrid = smooth(solving);
    const rGrid = smooth(review);
    if (!sGrid && !rGrid) return null;

    // Robust ceiling from the upper envelope so most of the band shows; the most
    // extreme spell clips at the top edge (overflow:hidden). Use max(mean, hi)
    // per grid point, NOT hi alone: with a heavy right tail the weighted mean can
    // sit ABOVE the 82.5% band edge, so a hi-only ceiling would clip the mean
    // line out of the window.
    const hiVals = [];
    for (const grid of [sGrid, rGrid]) if (grid) for (const e of grid) if (e) hiVals.push(Math.max(e.hi, e.mid));
    hiVals.sort((a, b) => a - b);
    const rawHigh = Math.max(quantileSorted(hiVals, 0.90) * 1.05, 1);
    const yLow = 0;
    // Two-tier y ticks. The CI band can reach several minutes while both median
    // lines sit under a minute, so a single uniform step leaves the medians
    // stranded between gridlines. Instead: a COARSE step (~4 ticks) spans the
    // whole CI range, and a FINER step (~4 ticks) adds resolution across the
    // median band [0, max median]. Both steps come off a seconds→minutes ladder
    // (niceStep in chart-svg.js), so every tick is a whole minute or a round
    // number of seconds, and the two tiers share ticks where they coincide
    // (e.g. 0, 1m). The render-time overlap pass below then drops whatever
    // doesn't fit — coarse first — so we show as many as there's room for and
    // never overlap.
    // value → tier (0 = coarse/preferred, 1 = fine). A Map dedupes shared ticks.
    let medMax = 0;
    for (const grid of [sGrid, rGrid]) if (grid) for (const e of grid) if (e) medMax = Math.max(medMax, e.mid);
    const hiStep = niceStep(rawHigh, 4);
    const yHigh = Math.ceil(rawHigh / hiStep) * hiStep;
    // Median band hugs the actual medians (small headroom so a fine tick lands
    // just above the highest one), capped at the ceiling. niceStep self-scales to
    // ~4 ticks, so a sub-minute band yields sub-minute ticks — DON'T inflate it to
    // a full coarse step: that would spread the dense ticks more than twice past
    // the medians (e.g. 30s/1m/90s when every median is under a minute).
    medMax = medMax > 0 ? Math.min(medMax * 1.1, yHigh) : 0;
    const lowStep = niceStep(medMax, 4);
    const fmt = (s) => s === 0 ? '0' : (s % 60 === 0 ? `${s / 60}m` : `${Math.round(s)}s`);
    const tierByVal = new Map();
    for (let v = 0; v <= yHigh + 1e-6; v += hiStep) tierByVal.set(Math.round(v), 0);
    for (let v = lowStep; v <= medMax + 1e-6; v += lowStep)
        if (!tierByVal.has(Math.round(v))) tierByVal.set(Math.round(v), 1);
    const yTicks = [...tierByVal.entries()].map(([v, pri]) => ({ v, pri })).sort((a, b) => a.v - b.v);

    const container = makeChartFrame(
        getTerm('statsChartTimePerProblemSince', { date: new Date(t0).toISOString().slice(0, 10) }));

    const leftLabels = yTicks.map(t => fmt(t.v));
    const widest = arr => arr.reduce((a, b) => (b.length > a.length ? b : a), '');
    const topPct = v => (1 - (v - yLow) / (yHigh - yLow)) * 100;
    // Size the y-axis column to the WIDEST label, not just the top one — an
    // intermediate tick ("45s") can be wider than the ceiling ("3m") and
    // would otherwise overflow the column and clip on the left.
    const { yaxis, tickEls } = buildYAxis(widest(leftLabels),
        yTicks.map(t => ({ top: topPct(t.v), text: fmt(t.v) })));
    // tickRows pairs each label with its grid line (created below) and tier, so
    // the overlap pass can hide both together and keep coarse ticks first.
    const tickRows = yTicks.map((t, i) => ({ v: t.v, pri: t.pri, labelEl: tickEls[i], lineEl: null }));

    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - t0) / tSpan) * W;
    const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;

    const svg = makeChartSvg(W, H);
    svg.style.overflow = 'hidden'; // clip clipped-day spikes at the top edge

    // Grid lines here can be vertical (x-ticks) too, so gridLine (chart-svg.js)
    // takes explicit endpoints rather than using hGridLine.
    // Horizontal grid at each y-tick, kept in sync with its label's visibility.
    for (const row of tickRows) {
        row.lineEl = svg.appendChild(gridLine(0, yOf(row.v), W, yOf(row.v)));
    }

    // Evenly-spaced x-axis date ticks, each with a matching vertical grid line.
    const N_XTICKS = 3;
    const xTicks = [];
    for (let i = 0; i < N_XTICKS; i++) xTicks.push(t0 + tSpan * (i / (N_XTICKS - 1)));
    for (const t of xTicks) {
        const x = xOf(t);
        svg.appendChild(gridLine(x, 0, x, H));
    }

    const SERIES = [
        { grid: sGrid, line: '#2e9e5b', band: 'rgba(46,158,91,0.20)' },
        { grid: rGrid, line: '#d98a2b', band: 'rgba(217,138,43,0.20)' },
    ];

    // Bands first, then median lines on top.
    for (const { grid, band } of SERIES) {
        if (!grid) continue;
        for (const run of segments(grid)) {
            if (run.length < 2) continue;
            let d = '';
            for (let i = 0; i < run.length; i++) {
                d += (i === 0 ? 'M' : 'L') + `${xOf(run[i].t).toFixed(1)} ${yOf(run[i].hi).toFixed(1)} `;
            }
            for (let i = run.length - 1; i >= 0; i--) {
                d += `L${xOf(run[i].t).toFixed(1)} ${yOf(run[i].lo).toFixed(1)} `;
            }
            svg.appendChild(svgEl('path', { d: d + 'Z', fill: band, stroke: 'none' }));
        }
    }

    for (const { grid, line } of SERIES) {
        if (!grid) continue;
        for (const run of segments(grid)) {
            if (run.length === 1) {
                svg.appendChild(svgEl('circle', {
                    cx: xOf(run[0].t).toFixed(1), cy: yOf(run[0].mid).toFixed(1),
                    r: '2', fill: line,
                }));
                continue;
            }
            const pts = run.map(e => `${xOf(e.t).toFixed(1)},${yOf(e.mid).toFixed(1)}`).join(' ');
            svg.appendChild(svgEl('polyline', {
                points: pts, fill: 'none', stroke: line,
                'stroke-width': '1.5', 'stroke-linejoin': 'round',
                'vector-effect': 'non-scaling-stroke',
            }));
        }
    }

    // Date labels under the plot, aligned to the vertical grid lines above. A
    // hidden spacer matching the widest y-label (plus the plot-row gap) offsets
    // the track so it sits flush under the SVG, then each label is positioned by
    // its tick's x-fraction (ends anchored inward so they don't overflow).
    const fmtDate = (t) => new Date(t).toISOString().slice(5, 10); // MM-DD
    const xaxis = document.createElement('div');
    xaxis.className = 'stats-chart-xaxis stats-chart-xaxis--ticks';
    const xSpacer = document.createElement('span');
    xSpacer.className = 'y-spacer';
    xSpacer.textContent = widest(leftLabels);
    const xTrack = document.createElement('div');
    xTrack.className = 'stats-chart-xaxis-track';
    for (let i = 0; i < xTicks.length; i++) {
        const lbl = document.createElement('span');
        lbl.className = 'x-tick';
        lbl.style.left = `${(xOf(xTicks[i]) / W) * 100}%`;
        lbl.style.transform = i === 0 ? 'translateX(0)'
            : i === xTicks.length - 1 ? 'translateX(-100%)'
            : 'translateX(-50%)';
        lbl.textContent = fmtDate(xTicks[i]);
        xTrack.append(lbl);
    }
    xaxis.append(xSpacer, xTrack);

    container.append(buildPlotRow(yaxis, svg, xaxis));

    const legend = buildLegend([
        { kind: 'line', color: '#2e9e5b', text: getTerm('statsLegendSolving') },
        { kind: 'line', color: '#d98a2b', text: getTerm('statsLegendReviewing') },
        { kind: 'band', color: 'rgba(128,128,128,0.30)', text: getTerm('statsLegend50Spread') },
    ]);
    container.append(legend);

    // Keep as many ticks as physically fit, coarse tier first (the tiered
    // `rows` mode of hideOverlappingYTicks): coarse ticks are placed before
    // fine ones, a tick is kept only if it clears minSep from every already-
    // kept tick — so the CI-spanning coarse ticks always win and the finer
    // median-band ticks fill the gaps where there's room, never overlapping.
    // Each label and its grid line toggle as one.
    observeResize(svg, () => hideOverlappingYTicks(svg, yaxis, tickRows));

    return container;
}
