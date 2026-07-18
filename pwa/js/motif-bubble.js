// js/motif-bubble.js
//
// Review-mode "why this move" bubble. Reads the reviewed problem's correct-move
// embedding (the K-dim, skew-ICA-oriented vector shipped per problem in
// metadata.json) and surfaces its strongest motifs as human-readable reasons.
//
// Under skew-ICA the positive tail is "motif present" and scores are
// unit-variance, so a component score is effectively a z-score: we include a
// motif only above MIN_Z and show at most MAX_MOTIFS, highest first. Labels are
// localized via the shared i18n resolver, falling back to English.
//
// Layout (see style.css): in landscape the bubble is a plain in-flow block in
// the controls column, below the buttons, always fully visible. In portrait it
// is a position:fixed overlay (out of the row flow, so it can never resize or
// shift the board) parked below the controls; its tab lifts it over the board.
// The "why this move" title lives in the tab bar itself, not in the content.

import { state } from './state.js';
import { getTerm, localizedMotif } from './i18n.js';
import { APP_MODES } from './constants.js';

const MIN_Z = 3.0;        // include a motif only if its component score >= 3 std
const MAX_MOTIFS = 3;

// Recomputed per problem in populateMotifBubble(); gates visibility in
// syncMotifBubble() so we never show an empty bubble.
let _hasMotifs = false;

const el = (id) => document.getElementById(id);

const isPortrait = () => window.matchMedia('(max-aspect-ratio: 125/100)').matches;

/**
 * Portrait parking is board/buttons-relative but the buttons' bottom edge isn't
 * CSS-knowable (vertically-centred column, font-dependent heights), so we
 * measure it here and feed the CSS via custom properties:
 *   --mb-park-top : viewport-y just below the buttons (parked top edge)
 *   --mb-lift     : translateY that lands the bubble's centre at screen centre
 * Landscape ignores both (the bubble is in-flow there). No-op unless the
 * bubble is currently visible in portrait.
 */
export function refreshMotifBubbleMetrics() {
    const bubble = el('motif-bubble');
    if (!bubble || bubble.classList.contains('motif-bubble-hidden')) return;
    if (!isPortrait()) return;
    const controls = el('controls-wrapper');
    if (!controls) return;
    const gap = Math.min(window.innerWidth, window.innerHeight) * 0.01;  // ~1vmin
    const parkTop = controls.getBoundingClientRect().bottom + gap;
    bubble.style.setProperty('--mb-park-top', `${parkTop}px`);
    // Height is transform-invariant, so measuring while parked or lifted is fine.
    const bubbleHeight = bubble.getBoundingClientRect().height;
    const lift = (window.innerHeight / 2) - (parkTop + bubbleHeight / 2);
    bubble.style.setProperty('--mb-lift', `${lift}px`);
}

/**
 * Top motifs of a correct-move embedding: labeled components whose |score|
 * clears MIN_Z, strongest first, capped at MAX_MOTIFS. The axes are BIPOLAR
 * (residual skill axes, 2026-07-18): a large NEGATIVE loading is the
 * negative-tail motif, not "no motif" — each side uses its own label, and a
 * side left unnamed in labels.json is skipped. Returns localized strings.
 */
function topMotifs(embedding) {
    const labels = state.embeddingLabels;
    if (!Array.isArray(embedding) || !labels) return [];
    const scored = [];
    for (let k = 0; k < embedding.length; k++) {
        const v = embedding[k];
        if (!Number.isFinite(v) || !(Math.abs(v) >= MIN_Z)) continue;
        const text = localizedMotif(labels[String(k)],
                                    v >= 0 ? 'positive' : 'negative');
        if (!text) continue;
        scored.push({ v: Math.abs(v), text });
    }
    scored.sort((a, b) => b.v - a.v);
    return scored.slice(0, MAX_MOTIFS).map((m) => m.text);
}

/**
 * Fill the bubble from a problem (a history entry carrying `.embedding`) and
 * reset it to the parked position. Records whether there's anything to show;
 * actual visibility is decided by syncMotifBubble().
 */
export function populateMotifBubble(problem) {
    const bubble = el('motif-bubble');
    const content = el('motif-bubble-content');
    if (!bubble || !content) return;

    const motifs = topMotifs(problem && problem.embedding);
    _hasMotifs = motifs.length > 0;

    // The title lives in the tab bar; set it here so it follows the language.
    const tabLabel = el('motif-bubble-tab-label');
    if (tabLabel) tabLabel.textContent = getTerm('motifTitle');

    content.textContent = '';
    if (_hasMotifs) {
        const ul = document.createElement('ul');
        ul.className = 'motif-bubble-list';
        for (const m of motifs) {
            const li = document.createElement('li');
            li.textContent = m;
            ul.appendChild(li);
        }
        content.appendChild(ul);
    }

    bubble.classList.remove('over-board');   // always (re)appear parked
    syncMotifBubble();
}

/**
 * Show the bubble only while reviewing the problem's first move (the initial
 * analysis node); hide it everywhere else — in solving/linked play, or once a
 * variation is entered, since we have no motifs for deeper positions. Safe to
 * call on every UI refresh; visibility is derived purely from current state.
 */
export function syncMotifBubble() {
    const bubble = el('motif-bubble');
    if (!bubble) return;
    const atRoot = !!state.kifuReader && state.kifuReader.node === state.initialAnalysisNode;
    const show = state.appMode === APP_MODES.ANALYZING && _hasMotifs && atRoot;
    bubble.classList.toggle('motif-bubble-hidden', !show);
    bubble.setAttribute('aria-hidden', show ? 'false' : 'true');
    if (!show) {
        bubble.classList.remove('over-board');
    } else {
        refreshMotifBubbleMetrics();   // (re)anchor below the buttons in portrait
    }
}

/** Wire the edge tab: toggles the bubble over the board and back. */
export function initMotifBubble() {
    const tab = el('motif-bubble-tab');
    const bubble = el('motif-bubble');
    if (!tab || !bubble) return;
    tab.addEventListener('click', (e) => {
        e.preventDefault();
        bubble.classList.toggle('over-board');
    });

    // Keep the portrait anchor in sync with viewport / orientation changes.
    let raf = 0;
    window.addEventListener('resize', () => {
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; refreshMotifBubbleMetrics(); });
    });
}
