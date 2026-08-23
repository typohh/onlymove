// js/motif-bubble.js
//
// The motif bubble, with one content per mode. In review mode it is the "why
// this move" bubble: it reads the reviewed problem's correct-move embedding
// (the K-dim vector shipped per problem in metadata.json) and surfaces its
// strongest motifs as human-readable reasons. In solving mode it is the
// TRAINING FOCUS list: the user's flagged weaknesses — the same serve
// categories selection draws from (weaknessServeCategories) — so it announces
// what the served problems are being aimed at. Deliberately the STATIC flagged
// list, never which category the current problem was drawn from: the list is
// near-constant across a session and so leaks no per-problem hint, while the
// drawn category would be one.
//
// The axes are z-scored per axis over the corpus, so a component score is a
// z-score: a problem motif is included only above the shared MOTIF_Z gate
// (constants.js — the same constant gating serving and the stats pane), at
// most MAX_MOTIFS, highest first. Labels are localized via the shared i18n
// resolver, falling back to English.
//
// Layout (see style.css): in landscape the bubble is a plain in-flow block in
// the controls column, below the buttons, always fully visible. In portrait it
// is a position:fixed overlay (out of the row flow, so it can never resize or
// shift the board) parked below the controls; its tab lifts it over the board.
// The "why this move" title lives in the tab bar itself, not in the content.

import { state } from './state.js';
import { getTerm, localizedMotif } from './i18n.js';
import { APP_MODES, MOTIF_Z } from './constants.js';
import { appMetrics, layoutMode } from './viewport.js';
import { flaggedMotifCategories } from './weakness.js';
import { onTrainingFocusShown } from './onboarding.js';

const MIN_Z = MOTIF_Z;    // include a problem motif only past the shared gate
const MAX_MOTIFS = 3;

// Review-mode content, recomputed per problem in populateMotifBubble(). The
// solving-mode training-focus content is derived fresh in syncMotifBubble()
// (it depends on the weakness posterior, not on the shown problem). A cache
// key of the last-rendered content keeps the per-repaint rebuild cheap.
let _problemMotifs = [];
let _renderedKey = null;

const el = (id) => document.getElementById(id);

// Asks style.css which layout it selected instead of re-testing the aspect
// ratio here — see layoutMode().
const isPortrait = () => layoutMode() === 'portrait';

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
    const vp = appMetrics();
    // getBoundingClientRect and `top` on a position:fixed element share an
    // origin, so the park offset needs no inset correction of its own.
    const parkTop = controls.getBoundingClientRect().bottom + vp.u;  // the 1·--u park gap
    bubble.style.setProperty('--mb-park-top', `${parkTop}px`);
    // Height is transform-invariant, so measuring while parked or lifted is fine.
    const bubbleHeight = bubble.getBoundingClientRect().height;
    // Centre of the USABLE band, not of the screen: under a notch or a browser
    // toolbar those differ, and the board this lifts over is centred in the
    // usable band too.
    const usableCentre = vp.safeTop + vp.h / 2;
    const lift = usableCentre - (parkTop + bubbleHeight / 2);
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
 * The user's training focus: localized labels of the flagged serve categories
 * (flaggedMotifCategories — named tails whose own-attempt reading clears the
 * shared confidence gate, selection-excluded motifs included), minus the
 * `strong` direction, strongest first, same MAX_MOTIFS cap as the
 * problem list. Empty whenever nothing is flagged or the model is absent,
 * which is what keeps the bubble silent rather than showing a "none found".
 */
function trainingFocusMotifs() {
    const labels = state.embeddingLabels;
    if (!labels) return [];
    const scored = [];
    for (const { k, tail, kind, z } of flaggedMotifCategories()) {
        // `strong` is a serve direction, not a training direction: it tells
        // selection to spend LESS time on a motif the user already handles,
        // which does not belong under a heading about what to work on.
        if (kind === 'strong') continue;
        // `tail` is the NAMED tail describing the category, which for an
        // overplay is the opposite of its serve direction — an overplay of a
        // one-sided motif serves anti-motif problems but is described through
        // the motif's own name, with a "too much" qualifier.
        const name = localizedMotif(labels[String(k)], tail);
        if (!name) continue;
        // Each kind needs its own wording. A bare name reads as "weak at this",
        // so a tendency finding must not use one — an underplay shown bare beside
        // a POSITIVE strength reading in the stats table looks like a
        // contradiction rather than two different statements about the motif.
        const text = kind === 'overplay' ? getTerm('trainingFocusTooMuch', { motif: name })
                   : kind === 'underplay' ? getTerm('trainingFocusTooLittle', { motif: name })
                   : name;
        scored.push({ z, text });
    }
    scored.sort((a, b) => b.z - a.z);
    return scored.slice(0, MAX_MOTIFS).map((m) => m.text);
}

/**
 * Fill the review-mode content from a problem (a history entry carrying
 * `.embedding`) and reset the bubble to the parked position. Actual visibility
 * and rendering are decided by syncMotifBubble().
 */
export function populateMotifBubble(problem) {
    const bubble = el('motif-bubble');
    if (!bubble) return;
    _problemMotifs = topMotifs(problem && problem.embedding);
    bubble.classList.remove('over-board');   // always (re)appear parked
    syncMotifBubble();
}

/**
 * Derive content and visibility from current state; safe to call on every UI
 * refresh. Review mode shows the reviewed problem's motifs, and only at the
 * root position (no motifs exist for deeper nodes). Solving mode shows the
 * training-focus list. Linked play and empty lists show nothing — there is
 * deliberately no empty-state text.
 */
export function syncMotifBubble() {
    const bubble = el('motif-bubble');
    const content = el('motif-bubble-content');
    if (!bubble || !content) return;
    const atRoot = !!state.kifuReader && state.kifuReader.node === state.initialAnalysisNode;

    let motifs = [];
    let titleKey = null;
    if (state.appMode === APP_MODES.ANALYZING && atRoot) {
        motifs = _problemMotifs;
        titleKey = 'motifTitle';
    } else if (state.appMode === APP_MODES.SOLVING) {
        motifs = trainingFocusMotifs();
        titleKey = 'trainingFocusTitle';
    }
    const show = motifs.length > 0;

    // Rebuild the DOM only when the content actually changed — this runs on
    // every repaint.
    const key = show ? `${titleKey}|${motifs.join('|')}` : null;
    if (show && key !== _renderedKey) {
        const tabLabel = el('motif-bubble-tab-label');
        if (tabLabel) tabLabel.textContent = getTerm(titleKey);
        content.textContent = '';
        const ul = document.createElement('ul');
        ul.className = 'motif-bubble-list';
        for (const m of motifs) {
            const li = document.createElement('li');
            li.textContent = m;
            ul.appendChild(li);
        }
        content.appendChild(ul);
        _renderedKey = key;
    }

    bubble.classList.toggle('motif-bubble-hidden', !show);
    bubble.setAttribute('aria-hidden', show ? 'false' : 'true');
    if (!show) {
        bubble.classList.remove('over-board');
    } else {
        refreshMotifBubbleMetrics();   // (re)anchor below the buttons in portrait
        // First-appearance announcement for the training-focus list; the hook
        // is once-ever and self-gating, so calling it per repaint is fine.
        if (titleKey === 'trainingFocusTitle') onTrainingFocusShown();
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
