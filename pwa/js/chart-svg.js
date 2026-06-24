// js/chart-svg.js
//
// Shared SVG/DOM chart primitives for the detailed-stats charts (stats-ui.js).
// The three chart builders all create raw SVG nodes, a legend strip, and a
// y-tick overlap-hiding observer; these helpers replace the repeated
// createElementNS boilerplate and the byte-identical legend / tick-visibility
// blocks that used to live inline in each builder.
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

/** A faint horizontal gridline at `y` spanning the full chart width. */
export function hGridLine(y, W) {
    return svgEl('line', {
        x1: 0, x2: W, y1: y, y2: y,
        stroke: 'rgba(128,128,128,0.35)', 'stroke-width': '0.5',
        'vector-effect': 'non-scaling-stroke',
    });
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

/**
 * Greedily hide `.y-tick` labels in `yaxis` that would physically overlap, top
 * to bottom: a tick is shown only if it clears one line-height from the last
 * shown one. No-op until the SVG has a measured height.
 */
export function hideOverlappingYTicks(svg, yaxis) {
    const h = svg.getBoundingClientRect().height;
    if (h === 0) return;
    const tickEls = [...yaxis.querySelectorAll('.y-tick')]
        .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
    const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
    let lastShownPx = -Infinity;
    for (const el of tickEls) {
        const topPx = (parseFloat(el.style.top) / 100) * h;
        if (topPx - lastShownPx >= minSepPx) { el.style.visibility = ''; lastShownPx = topPx; }
        else el.style.visibility = 'hidden';
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
