// js/stats-activity-chart.js
//
// The Activity pane's chart in the detailed-stats modal (stats-ui.js is the
// orchestrator; it pairs this chart with the review-queue summary text).

import { state } from './state.js';
import { getTerm } from './i18n.js';
import {
    svgEl, makeChartSvg, hGridLine,
    hideOverlappingYTicks, observeResize,
    makeChartFrame, buildYAxis, buildPlotRow, chartViewBoxHeight,
} from './chart-svg.js';

// Number of problems attempted per calendar day, as a bar chart. Each attempt
// is binned into its local-midnight day; the bars span first→last active day,
// so idle days simply show no bar. Lives on the Activity tab. Returns null when
// there's too little data to read as a daily trend.
export function buildProblemsPerDayChart(history = state.answerHistory) {
    const DAY_MS = 24 * 3600 * 1000;
    const dayMid = (t) => { const d = new Date(t); d.setHours(0, 0, 0, 0); return d.getTime(); };

    const counts = new Map(); // day-midnight ms → attempt count
    for (const a of history) {
        if (typeof a.date !== 'number') continue;
        const k = dayMid(a.date);
        counts.set(k, (counts.get(k) || 0) + 1);
    }
    if (counts.size < 2) return null;

    const days = [...counts.keys()].sort((p, q) => p - q);
    const d0 = days[0];
    const d1 = days[days.length - 1];
    const xStart = d0, xEnd = d1 + DAY_MS;
    const total = xEnd - xStart;
    const maxCount = Math.max(...counts.values());

    // Integer y-ticks: 5 evenly-spaced labels at 0..(step·4).
    const step = Math.max(1, Math.ceil(maxCount / 4));
    const yHigh = step * 4;

    const container = makeChartFrame(
        getTerm('statsChartProblemsPerDaySince', { date: new Date(d0).toISOString().slice(0, 10) }));

    const ticks = [];
    for (let i = 0; i <= 4; i++) {
        const v = step * i;
        ticks.push({ top: (1 - v / yHigh) * 100, text: String(v) });
    }
    const { yaxis } = buildYAxis(String(yHigh), ticks);

    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - xStart) / total) * W;
    const yOf = v => (1 - v / yHigh) * H;

    const svg = makeChartSvg(W, H);

    // Horizontal grid at each y-tick.
    for (let i = 0; i <= 4; i++) {
        svg.appendChild(hGridLine(yOf(step * i), W));
    }

    // One bar per active day. Leave a thin gap between adjacent days unless the
    // span is so long that bars are already sub-pixel.
    const barW = (DAY_MS / total) * W;
    const gap = barW > 2 ? Math.min(0.6, barW * 0.15) : 0;
    for (const [k, c] of counts) {
        const x = xOf(k) + gap / 2;
        const w = Math.max(0.4, barW - gap);
        const y = yOf(c);
        svg.appendChild(svgEl('rect', {
            x: x.toFixed(2), y: y.toFixed(2),
            width: w.toFixed(2), height: (H - y).toFixed(2),
            fill: '#6f8bbf',
        }));
    }

    const xaxis = document.createElement('div');
    xaxis.className = 'stats-chart-xaxis';
    xaxis.textContent = new Date(d1).toISOString().slice(0, 10);

    container.append(buildPlotRow(yaxis, svg, xaxis));

    // Collapse overlapping y-tick labels, mirroring the Time chart.
    observeResize(svg, () => hideOverlappingYTicks(svg, yaxis));

    return container;
}
