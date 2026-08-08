// js/viewport.js
//
// The RESOLVED pixel values of style.css's viewport-basis tokens (--app-h,
// --app-w, --u, --safe-t/--safe-l), for the two places that have to do layout
// arithmetic in JS: the motif bubble's portrait anchoring and the stats charts'
// aspect ratio.
//
// Reading the tokens back rather than re-deriving from window.innerWidth /
// innerHeight is the whole point. innerHeight is the LARGE viewport — the
// height with a mobile browser's toolbar retracted — and knows nothing about
// the safe-area insets, so a JS-computed offset and a CSS-computed one would
// disagree by exactly the toolbar plus the notch, in the direction that pushes
// content off an overflow:hidden screen. See the basis comment in style.css.
//
// The values are read off a hidden probe element rather than
// getComputedStyle(root).getPropertyValue('--app-h'): an unregistered custom
// property computes to its own token stream ("calc(100dvh - 0px - 0px)"), not
// to pixels, so the property lookup would return a string no arithmetic can
// use. Assigning the tokens to real layout properties makes the browser resolve
// them, and one probe carries all four at once so a call costs a single style
// flush.

let probe = null;

function ensureProbe() {
    if (probe && probe.isConnected) return probe;
    if (!document.body) return null;
    probe = document.createElement('div');
    probe.setAttribute('aria-hidden', 'true');
    // content-box so the padding (carrying the safe insets) can't eat the
    // height/width we want to read back.
    probe.style.cssText = [
        'position:fixed', 'top:0', 'left:0',
        'visibility:hidden', 'pointer-events:none', 'box-sizing:content-box',
        'height:var(--app-h)', 'width:var(--app-w)',
        'margin-top:var(--u)',
        'padding-top:var(--safe-t)', 'padding-left:var(--safe-l)',
    ].join(';');
    document.body.appendChild(probe);
    return probe;
}

const px = (value, fallback) => {
    const n = parseFloat(value);
    return Number.isFinite(n) ? n : fallback;
};

/**
 * `{ h, w, min, u, safeTop, safeLeft }` in CSS pixels — the usable viewport
 * after the toolbar and the safe-area insets, matching the CSS tokens exactly.
 * Falls back to the raw window box before <body> exists, which is the
 * pre-safe-area behaviour and never worse than not answering.
 */
export function appMetrics() {
    const el = ensureProbe();
    const rawH = window.innerHeight || 768;
    const rawW = window.innerWidth || 1024;
    if (!el) {
        return {
            h: rawH, w: rawW, min: Math.min(rawH, rawW),
            u: Math.min(rawH, rawW) / 100, safeTop: 0, safeLeft: 0,
        };
    }
    const cs = getComputedStyle(el);
    const h = px(cs.height, rawH);
    const w = px(cs.width, rawW);
    const min = Math.min(h, w);
    return {
        h, w, min,
        u: px(cs.marginTop, min / 100),
        safeTop: px(cs.paddingTop, 0),
        safeLeft: px(cs.paddingLeft, 0),
    };
}

/**
 * Resolve any CSS length expression — typically `var(--some-token)` — to CSS
 * pixels, so a number a stylesheet already states never has to be restated as
 * a JS literal. Returns `fallback` if the value doesn't resolve (unknown token,
 * no document yet).
 */
export function resolveLength(value, fallback = 0) {
    const el = ensureProbe();
    if (!el) return fallback;
    const prev = el.style.height;
    el.style.height = value;
    const out = px(getComputedStyle(el).height, NaN);
    el.style.height = prev;
    return Number.isFinite(out) ? out : fallback;
}

/**
 * Which layout style.css's media queries selected: 'portrait' (portrait and
 * squarish) or 'landscape'. Read from the --layout-mode token rather than a
 * second matchMedia() copy of the aspect ratio, so the breakpoint lives in one
 * file and JS agrees with the cascade even at the exact cutoff, where both
 * aspect-ratio queries match and order decides.
 */
export function layoutMode() {
    const mode = getComputedStyle(document.documentElement)
        .getPropertyValue('--layout-mode').trim();
    return mode === 'portrait' ? 'portrait' : 'landscape';
}
