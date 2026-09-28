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
// shift the board) parked below the controls; its tab lifts it over the board,
// but only while the parked bubble actually runs off the bottom of the usable
// band — a list short enough to be entirely visible where it parks gets the
// landscape treatment instead (inert tab, no arrows), so the strip below the
// buttons carries no target for an accidental tap that would do nothing.
// The "why this move" title lives in the tab bar itself, not in the content.

import { state } from './state.js';
import { getTerm, localizedMotif } from './i18n.js';
import { APP_MODES, MOTIF_Z } from './constants.js';
import { appMetrics, layoutMode } from './viewport.js';
import { flaggedMotifCategories } from './weakness.js';
import { onTrainingFocusShown } from './onboarding.js';
import { hintAnchor } from './hint-bubble.js';
import { explainMotif, explainPace } from './motif-explain.js';
import { computePaceReadings, PACE_KINDS } from './pace-readings.js';
import { computeMotifStats } from './motif-stats.js';
import { orientReadings, isFlagged } from './motif-readings.js';

const MIN_Z = MOTIF_Z;    // include a problem motif only past the shared gate
const MAX_MOTIFS = 3;
// Two standard errors: above it the weakness sentence is stated flatly, between
// the admission gate and it the tentative wording is used instead. NOT written
// as a multiple of MOTIF_Z — this is the two-sigma point, not "twice the bar",
// and it must stay there if the shared gate is ever retuned.
const CONFIDENT_Z = 2;
// Sub-pixel slack on the fits-on-screen comparison: both sides are fractional
// CSS pixels, and a bubble whose last text row ends a rounding error past the
// band is visible.
const FIT_SLACK = 0.5;

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
 * It also decides `.motif-bubble-fits` (see below). Landscape ignores all of
 * it (the bubble is in-flow there). No-op unless the bubble is currently
 * visible in portrait.
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
    // A short list can end above the bottom of the usable band, in which case
    // the lift reveals nothing and the tab is just a target for accidental
    // taps in the strip below the buttons. `.motif-bubble-fits` then makes it
    // inert and drops the arrows, which is the state landscape is in by
    // construction. The test cannot flap: the arrows are the tallest thing in
    // the tab, so dropping them can only SHRINK the bubble, and one that fits
    // with them still fits without.
    const fits = parkTop + bubbleHeight <= vp.safeTop + vp.h + FIT_SLACK;
    bubble.classList.toggle('motif-bubble-fits', fits);
    if (fits) bubble.classList.remove('over-board');
}

/**
 * Top motifs of a correct-move embedding: labeled components whose |score|
 * clears MIN_Z, strongest first, capped at MAX_MOTIFS. The axes are BIPOLAR
 * (residual skill axes, 2026-07-18): a large NEGATIVE loading is the
 * negative-tail motif, not "no motif" — each side uses its own label, and a
 * side left unnamed in labels.json is skipped. Returns localized strings.
 */
function topMotifs(problem) {
    const labels = state.embeddingLabels;
    const embedding = problem && problem.embedding;
    if (!Array.isArray(embedding) || !labels) return [];
    const scored = [];
    for (let k = 0; k < embedding.length; k++) {
        const v = embedding[k];
        if (!Number.isFinite(v) || !(Math.abs(v) >= MIN_Z)) continue;
        const side = v >= 0 ? 'positive' : 'negative';
        const component = labels[String(k)];
        const text = localizedMotif(component, side);
        if (!text) continue;
        scored.push({ v: Math.abs(v), text,
                      explain: () => explainMotif({ component, side, name: text, kind: 'problem' }) });
    }
    scored.sort((a, b) => b.v - a.v);
    return scored.slice(0, MAX_MOTIFS);
}

/**
 * The user's training focus: localized labels of the flagged serve categories
 * (flaggedMotifCategories — named tails whose own-attempt reading clears the
 * shared confidence gate), minus the
 * `strong` direction, strongest first, same MAX_MOTIFS cap as the
 * problem list. Empty whenever nothing is flagged or the model is absent,
 * which is what keeps the bubble silent rather than showing a "none found".
 */
function trainingFocusMotifs() {
    const labels = state.embeddingLabels;
    if (!labels) return [];
    // ONE LINE PER MOTIF, the most confident finding on it. An axis can flag on
    // both readings at once — weak at the motif AND over-reaching for it — and
    // they are two genuine findings, which is why selection keeps them as two
    // separate serve slots. As a READING LIST they are one motif twice, and the
    // second line spends a slot of a short list restating the first's subject
    // while the user is trying to see what to work on.
    //
    // Deliberately NOT done in flaggedMotifCategories(): weaknessServeCategories
    // reads the same list, and collapsing it there would silently drop a serve
    // criterion. This is a presentation rule and it belongs here.
    //
    // `z` arrives as a positive magnitude for every kind (weakness.js negates
    // the ones whose raw z is negative), so the larger one is the more confident
    // finding whichever readings are being compared.
    const best = new Map();
    for (const cat of flaggedMotifCategories()) {
        // `strong` is a serve direction, not a training direction: it tells
        // selection to spend LESS time on a motif the user already handles,
        // which does not belong under a heading about what to work on. Dropped
        // before the comparison, so it cannot win a slot and hide a real one.
        if (cat.kind === 'strong') continue;
        const prev = best.get(cat.k);
        if (!prev || cat.z > prev.z) best.set(cat.k, cat);
    }

    const scored = [];
    for (const { k, tail, kind, z } of best.values()) {
        // `tail` is the NAMED tail describing the category, which for an
        // overplay is the opposite of its serve direction — an overplay of a
        // one-sided motif serves anti-motif problems but is described through
        // the motif's own name, as something the user overvalues.
        const name = localizedMotif(labels[String(k)], tail);
        if (!name) continue;
        // Each kind is a different statement and gets its own SENTENCE. A bare
        // name is not one of them: it reads as "weak at this" by implication
        // only, which leaves the one finding that IS about skill looking like a
        // label somebody forgot to finish, and makes a tendency line shown bare
        // beside a positive strength reading in the stats table look like a
        // contradiction rather than two statements about one motif. The
        // sentence is localized WHOLE rather than assembled from a name and a
        // qualifier — the motif sits in a different place in each language, and
        // Korean needs a phrasing that avoids an object particle the motif
        // names do not all take the same form of.
        //
        // Every sentence also carries the CONFIDENCE of the reading behind it:
        // stated flatly past CONFIDENT_Z, hedged between the admission gate and
        // it. Six sentences rather than three plus a qualifier, for the reason
        // above — the hedge sits in a different place in each language, and in
        // Japanese and Korean it rewrites the verb ending.
        const confident = z > CONFIDENT_Z;
        const text = kind === 'overplay'
                   ? getTerm(confident ? 'trainingFocusOvervalued'
                                       : 'trainingFocusOvervaluedTentative', { motif: name })
                   : kind === 'underplay'
                   ? getTerm(confident ? 'trainingFocusUndervalued'
                                       : 'trainingFocusUndervaluedTentative', { motif: name })
                   : getTerm(confident ? 'trainingFocusWeak'
                                       : 'trainingFocusWeakTentative', { motif: name });
        // The explanation quotes the reading behind the sentence — the rank
        // effect, or the share of errors — which the front page never shows.
        // Computed when the bubble opens, not here: this list is rebuilt on
        // repaint, and the readings walk the whole window.
        const component = labels[String(k)];
        const readingKind = kind === 'weak' ? 'strength' : 'tendency';
        const explain = () => {
            const stats = computeMotifStats();
            const st = stats && stats.find((s) => s.k === k);
            const oriented = st ? orientReadings(st, tail) : null;
            return explainMotif({ component, side: tail, name, kind: readingKind,
                                  reading: oriented ? oriented[readingKind] : null });
        };
        scored.push({ z, text, explain });
    }
    // The pace readings compete for the same slots on the same z. Only the
    // weak side is a training direction — "more time helps you less than most"
    // is nothing to work on, the counterpart of `strong` above.
    const pace = computePaceReadings();
    for (const kind of PACE_KINDS) {
        const r = pace[kind];
        if (!isFlagged(r) || r.z > 0) continue;
        const key = kind === 'review' ? 'trainingFocusShortReview' : 'trainingFocusRushing';
        const z = -r.z;
        scored.push({ z, text: getTerm(z > CONFIDENT_Z ? key : `${key}Tentative`),
                      explain: () => explainPace({ pace: kind, reading: computePaceReadings()[kind] }) });
    }
    scored.sort((a, b) => b.z - a.z);
    return scored.slice(0, MAX_MOTIFS);
}

/**
 * Fill the review-mode content from a problem (a history entry carrying
 * `.embedding`) and reset the bubble to the parked position. Actual visibility
 * and rendering are decided by syncMotifBubble().
 */
export function populateMotifBubble(problem) {
    const bubble = el('motif-bubble');
    if (!bubble) return;
    _problemMotifs = topMotifs(problem);
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
    const key = show ? `${titleKey}|${motifs.map((m) => m.text).join('|')}` : null;
    if (show && key !== _renderedKey) {
        const tabLabel = el('motif-bubble-tab-label');
        if (tabLabel) tabLabel.textContent = getTerm(titleKey);
        content.textContent = '';
        const ul = document.createElement('ul');
        ul.className = 'motif-bubble-list';
        for (const m of motifs) {
            const li = document.createElement('li');
            // The whole line is the tap target: a short list with no room for
            // a separate "?" beside each entry.
            const span = document.createElement('span');
            span.className = 'motif-hint';
            span.textContent = m.text;
            hintAnchor(span, m.explain);
            li.appendChild(span);
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
        // Guarded in JS as well as by the stylesheet's pointer-events, because
        // the tab is a <button>: it stays keyboard-focusable when
        // pointer-events are off, and Enter would still fire this click.
        if (bubble.classList.contains('motif-bubble-fits')) return;
        bubble.classList.toggle('over-board');
    });

    // Keep the portrait anchor in sync with viewport / orientation changes.
    let raf = 0;
    window.addEventListener('resize', () => {
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; refreshMotifBubbleMetrics(); });
    });
}
