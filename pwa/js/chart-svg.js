// js/chart-svg.js
//
// Shared SVG/DOM chart primitives for the detailed-stats charts
// (stats-rank-pane.js, stats-activity-chart.js, stats-time-chart.js). The three
// chart builders all create raw SVG nodes, the same chart frame (container +
// title + plot row + y-axis), a legend strip, and a y-tick overlap-hiding
// observer; these helpers provide the shared building blocks.
import { appMetrics, resolveLength } from './viewport.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element, applying `attrs` via setAttribute. */
export function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
}

/** An <svg> chart canvas with a `0 0 W H` viewBox. */
export function makeChartSvg(W, H) {
    return svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
}

/** A faint gridline from (x1,y1) to (x2,y2) — horizontal or vertical. */
export function gridLine(x1, y1, x2, y2) {
    return svgEl('line', {
        x1, x2, y1, y2,
        stroke: 'rgba(128,128,128,0.35)', 'stroke-width': '0.5',
        'vector-effect': 'non-scaling-stroke',
    });
}

/** A faint horizontal gridline at `y` spanning the full chart width. */
export function hGridLine(y, W) {
    return gridLine(0, y, W, y);
}

// Pick a viewBox height (for a fixed width W) so the chart's aspect ratio
// approximates the modal's visible area. The SVG renders at width:100% /
// height:auto with uniform scaling, so a W×H viewBox paints at
// renderedW × renderedW·(H/W); matching H/W to (visible height)/(width) makes
// the whole chart block (title + plot + legend) fit inside the scrollable
// modal body without being taller than the viewport — visible whole on scroll.
//
// The modal's box is READ from the --stats-* tokens style.css declares, not
// restated as fractions here: this used to carry its own copies of 70vh / 90vw
// / 3vmin, which a change on the CSS side would silently invalidate with no
// error — the charts would just size themselves for a modal that no longer
// exists. `chrome` and `yAxisCol` stay literals because they estimate rendered
// content (title, legend, axis labels), not a stylesheet constant.
export function chartViewBoxHeight(W) {
    const vp = appMetrics();
    const bodyVisibleH = resolveLength('var(--stats-body-max-h)', 0.7 * vp.h);
    const cardPadX = resolveLength('var(--stats-card-pad-x)', 3 * vp.u);
    const cardMaxW = resolveLength('var(--stats-card-max-w)', 0.9 * vp.w);
    const chrome = 120;                        // title + legend + x-axis + margins (px)
    const plotH = Math.max(150, bodyVisibleH - chrome);
    const yAxisCol = 42;                       // y-tick label column
    const plotW = Math.max(220, cardMaxW - 2 * cardPadX - yAxisCol);
    // Clamp to a sane band: never taller than square, never an extreme letterbox.
    const ratio = Math.max(0.42, Math.min(plotH / plotW, 1.0));
    return Math.round(W * ratio);
}

/**
 * The common chart scaffold: a `.stats-chart` container opening with a
 * `.stats-chart-title`. `titleContent` is a string, or an array of nodes for a
 * composite title (the Rank chart embeds its server <select> in the title).
 */
export function makeChartFrame(titleContent) {
    const container = document.createElement('div');
    container.className = 'stats-chart';
    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    if (Array.isArray(titleContent)) title.append(...titleContent);
    else title.textContent = titleContent;
    container.append(title);
    return container;
}

/**
 * A `.stats-chart-yaxis` column: a `.y-spacer` sizing element (give it the
 * widest expected label) followed by one absolutely-positioned `.y-tick` per
 * `{ top, text }` entry (`top` in percent of the plot height). Returns the
 * axis element plus the tick elements in input order, so a caller can pair
 * them with grid lines (see hideOverlappingYTicks' `rows` mode).
 */
export function buildYAxis(spacerText, ticks) {
    const yaxis = document.createElement('div');
    yaxis.className = 'stats-chart-yaxis';
    const spacer = document.createElement('span');
    spacer.className = 'y-spacer';
    spacer.textContent = spacerText;
    yaxis.append(spacer);
    const tickEls = ticks.map(t => {
        const tick = document.createElement('div');
        tick.className = 'y-tick';
        tick.style.top = `${t.top}%`;
        tick.textContent = t.text;
        yaxis.append(tick);
        return tick;
    });
    return { yaxis, tickEls };
}

/**
 * A `.stats-chart-plot-row`: the y-axis and the svg (wrapped in a
 * `.stats-chart-svg-wrap`) side by side in a `.stats-chart-plot-main`, with an
 * optional x-axis element below.
 */
export function buildPlotRow(yaxis, svg, xaxis = null) {
    const plotRow = document.createElement('div');
    plotRow.className = 'stats-chart-plot-row';
    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-chart-svg-wrap';
    svgWrap.append(svg);
    const plotMain = document.createElement('div');
    plotMain.className = 'stats-chart-plot-main';
    plotMain.append(yaxis, svgWrap);
    plotRow.append(plotMain);
    if (xaxis) plotRow.append(xaxis);
    return plotRow;
}

/**
 * A `.stats-chart-legend` strip built from `[{ kind, color, text }]`. `kind`
 * picks the swatch shape via the `legend-<kind>` class (dot/line/band).
 */
export function buildLegend(items) {
    const legend = document.createElement('div');
    legend.className = 'stats-chart-legend';
    for (const item of items) {
        const el = document.createElement('span');
        el.className = `legend-item legend-${item.kind}`;
        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = item.color;
        el.append(swatch, document.createTextNode(item.text));
        legend.append(el);
    }
    return legend;
}

// The y-tick "nice step" ladder for time-valued (seconds) axes: every step is
// a whole minute or a round number of seconds. `niceStep(span, target)` picks
// the smallest step that fits ~`target` ticks across `span`.
export const NICE_TIME_STEPS = [10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 7200];
export function niceStep(span, target, steps = NICE_TIME_STEPS) {
    return steps.find(s => s >= span / target) ?? steps[steps.length - 1];
}

/**
 * Greedily hide y-axis tick labels that would physically overlap. No-op until
 * the SVG has a measured height.
 *
 * Default mode (no `rows`): the `.y-tick` labels in `yaxis`, top to bottom — a
 * tick is shown only if it clears one line-height from every shown one.
 *
 * `rows` mode ([{ labelEl, lineEl, pri }]): ticks are walked by tier priority
 * first (pri 0 = coarse/preferred, then finer tiers), then by position, and a
 * kept label toggles its paired grid line with it — so the coarse ticks always
 * win and the finer ones fill the gaps where there's room, never overlapping.
 */
export function hideOverlappingYTicks(svg, yaxis, rows = null) {
    const h = svg.getBoundingClientRect().height;
    if (h === 0) return;
    const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
    const entries = (rows
        ? rows.map(r => ({ r, px: (parseFloat(r.labelEl.style.top) / 100) * h }))
        : [...yaxis.querySelectorAll('.y-tick')]
            .map(el => ({ r: { labelEl: el, lineEl: null, pri: 0 }, px: (parseFloat(el.style.top) / 100) * h }))
    ).sort((a, b) => ((a.r.pri || 0) - (b.r.pri || 0)) || (a.px - b.px));
    const kept = [];
    for (const { r, px } of entries) {
        const show = kept.every(p => Math.abs(p - px) >= minSepPx);
        r.labelEl.style.visibility = show ? '' : 'hidden';
        if (r.lineEl) r.lineEl.style.display = show ? '' : 'none';
        if (show) kept.push(px);
    }
}

/**
 * Run `fn` whenever `svg` resizes (via ResizeObserver), with a one-shot
 * microtask fallback when ResizeObserver is unavailable. Returns the observer
 * (or null) so a caller that rebuilds can disconnect the prior one.
 */
export function observeResize(svg, fn) {
    if (typeof ResizeObserver !== 'undefined') {
        const obs = new ResizeObserver(fn);
        obs.observe(svg);
        return obs;
    }
    queueMicrotask(fn);
    return null;
}
