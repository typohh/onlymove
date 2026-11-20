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
// Layout (see style.css): the tab is a button in both layouts — it carries the
// title and a count badge, and tapping it shows or hides the list under it. In
// landscape the bubble is an in-flow block in the controls column, below the
// buttons. In portrait it is a position:fixed overlay (out of the row flow, so
// it can never resize or shift the board) parked below the controls. A list
// that ends on screen where it parks simply opens there; one that would run
// off the bottom of the usable band opens AND lifts over the board, and
// closing it parks it again.
//
// That is the WEAKNESSES list. The review list ("what this tests") is about
// the problem on the board and is always open: its tab is a plain title, with
// no badge, and is live only in portrait when the list runs off the screen,
// where it lifts the bubble over the board and back.
//
// The badge counts every finding the list has, not only the MAX_MOTIFS it
// prints, so a closed bubble still says how much is behind it.

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
// Whether the training-focus list shows TENTATIVE findings — those between the
// admission gate and CONFIDENT_Z, worded "you might…". Off, the list carries
// only confident findings; serving and the stats panes are unaffected either
// way, since they read the flagged categories directly.
const SHOW_POSSIBLE_WEAKNESSES = false;
// Sub-pixel slack on the fits-on-screen comparison: both sides are fractional
// CSS pixels, and a bubble whose last text row ends a rounding error past the
// band is visible.
const FIT_SLACK = 0.5;
// The user's open/closed choice, kept per device. It governs wherever the open
// list fits where it sits — landscape always, portrait when the list ends on
// screen. A portrait list that does NOT fit is never opened by this: open there
// means covering the board, which is a per-look decision (`_lifted`), not a
// standing one.
const OPEN_KEY = 'motifBubbleOpen';

// Review-mode content, recomputed per problem in populateMotifBubble(). The
// solving-mode training-focus content is derived fresh in syncMotifBubble()
// (it depends on the weakness posterior, not on the shown problem). A cache
// key of the last-rendered content keeps the per-repaint rebuild cheap.
let _problemMotifs = { items: [], total: 0 };
let _renderedKey = null;
let _open = readOpen();
let _lifted = false;   // portrait, list does not fit: opened over the board
let _fits = true;      // whether the OPEN bubble ends on screen where it sits
let _static = false;   // the review list: always open, never collapsible

function readOpen() {
    try { return localStorage.getItem(OPEN_KEY) !== '0'; } catch { return true; }
}
function writeOpen(open) {
    try { localStorage.setItem(OPEN_KEY, open ? '1' : '0'); } catch { /* choice lasts the session */ }
}

const el = (id) => document.getElementById(id);

// Asks style.css which layout it selected instead of re-testing the aspect
// ratio here — see layoutMode().
const isPortrait = () => layoutMode() === 'portrait';

/**
 * Decide the bubble's state — closed, open in place, or open over the board —
 * and apply it. Safe to call on every repaint and resize; no-op while hidden.
 *
 * Portrait parking is board/buttons-relative but the buttons' bottom edge isn't
 * CSS-knowable (vertically-centred column, font-dependent heights), so it is
 * measured here and fed to the CSS via custom properties:
 *   --mb-park-top : viewport-y just below the buttons (parked top edge)
 *   --mb-lift     : translateY that lands the OPEN bubble's centre at screen centre
 * Landscape needs neither (the bubble is in-flow there) and always fits.
 */
export function refreshMotifBubbleMetrics() {
    const bubble = el('motif-bubble');
    if (!bubble || bubble.classList.contains('motif-bubble-hidden')) return;
    let fits = true;
    const controls = el('controls-wrapper');
    if (isPortrait() && controls) {
        const vp = appMetrics();
        // getBoundingClientRect and `top` on a position:fixed element share an
        // origin, so the park offset needs no inset correction of its own.
        const parkTop = controls.getBoundingClientRect().bottom + vp.u;  // the 1·--u park gap
        bubble.style.setProperty('--mb-park-top', `${parkTop}px`);
        // The fit question is about the OPEN bubble, so measure it open; the
        // class is put back below before anything paints. Height is
        // transform-invariant, so measuring while parked or lifted is fine.
        bubble.classList.remove('motif-bubble-collapsed');
        const bubbleHeight = bubble.getBoundingClientRect().height;
        // Centre of the USABLE band, not of the screen: under a notch or a
        // browser toolbar those differ, and the board this lifts over is
        // centred in the usable band too.
        const usableCentre = vp.safeTop + vp.h / 2;
        const lift = usableCentre - (parkTop + bubbleHeight / 2);
        bubble.style.setProperty('--mb-lift', `${lift}px`);
        fits = parkTop + bubbleHeight <= vp.safeTop + vp.h + FIT_SLACK;
    }
    _fits = fits;
    if (fits) _lifted = false;
    const open = _static || (fits ? _open : _lifted);
    bubble.classList.toggle('motif-bubble-static', _static);
    bubble.classList.toggle('motif-bubble-fits', fits);
    bubble.classList.toggle('motif-bubble-collapsed', !open);
    bubble.classList.toggle('over-board', !fits && _lifted);
    const tab = el('motif-bubble-tab');
    if (tab) tab.setAttribute('aria-expanded', open ? 'true' : 'false');
}

/**
 * Top motifs of a correct-move embedding: labeled components whose |score|
 * clears MIN_Z, strongest first — `items` capped at MAX_MOTIFS, `total` the
 * count before the cap. The axes are BIPOLAR
 * (residual skill axes, 2026-07-18): a large NEGATIVE loading is the
 * negative-tail motif, not "no motif" — each side uses its own label, and a
 * side left unnamed in labels.json is skipped. Returns localized strings.
 */
function topMotifs(problem) {
    const labels = state.embeddingLabels;
    const embedding = problem && problem.embedding;
    if (!Array.isArray(embedding) || !labels) return { items: [], total: 0 };
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
    return { items: scored.slice(0, MAX_MOTIFS), total: scored.length };
}

/**
 * The user's training focus: localized labels of the flagged serve categories
 * (flaggedMotifCategories — named tails whose own-attempt reading clears the
 * shared confidence gate), minus the
 * `strong` direction, strongest first, same MAX_MOTIFS cap as the
 * problem list (`total` counts the findings before that cap). Empty whenever
 * nothing is flagged or the model is absent,
 * which is what keeps the bubble silent rather than showing a "none found".
 */
function trainingFocusMotifs() {
    const labels = state.embeddingLabels;
    if (!labels) return { items: [], total: 0 };
    // One line per motif. Only a `weak` category reaches the list (weakness.js
    // makes no category of the error lean), so a motif appears at most once;
    // the map keeps that true if a second listable kind is ever added.
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
    for (const { k, tail, z } of best.values()) {
        // `tail` is the NAMED tail the category is described through.
        const name = localizedMotif(labels[String(k)], tail);
        if (!name) continue;
        // The finding is a whole SENTENCE, never a bare motif name, and it is
        // localized whole rather than assembled from a name and a qualifier —
        // the motif sits in a different place in each language. It carries the
        // CONFIDENCE of the reading behind it: stated flatly past CONFIDENT_Z,
        // hedged between the admission gate and it, as two sentences because
        // in Japanese and Korean the hedge rewrites the verb ending.
        const confident = z > CONFIDENT_Z;
        const text = getTerm(confident ? 'trainingFocusWeak'
                                       : 'trainingFocusWeakTentative', { motif: name });
        // The explanation quotes the reading behind the sentence — the rank
        // effect, or the share of errors — which the front page never shows.
        // Computed when the bubble opens, not here: this list is rebuilt on
        // repaint, and the readings walk the whole window.
        const component = labels[String(k)];
        const explain = () => {
            const stats = computeMotifStats();
            const st = stats && stats.find((s) => s.k === k);
            const oriented = st ? orientReadings(st, tail) : null;
            return explainMotif({ component, side: tail, name, kind: 'strength',
                                  reading: oriented ? oriented.strength : null });
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
    const shown = SHOW_POSSIBLE_WEAKNESSES ? scored : scored.filter((m) => m.z > CONFIDENT_Z);
    return { items: shown.slice(0, MAX_MOTIFS), total: shown.length };
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
    _lifted = false;   // always (re)appear parked
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

    let found = { items: [], total: 0 };
    let titleKey = null;
    if (state.appMode === APP_MODES.ANALYZING && atRoot) {
        found = _problemMotifs;
        titleKey = 'motifTitle';
    } else if (state.appMode === APP_MODES.SOLVING) {
        found = trainingFocusMotifs();
        titleKey = 'trainingFocusTitle';
    }
    const motifs = found.items;
    const show = motifs.length > 0;
    _static = titleKey === 'motifTitle';

    // Rebuild the DOM only when the content actually changed — this runs on
    // every repaint.
    const key = show ? `${titleKey}|${found.total}|${motifs.map((m) => m.text).join('|')}` : null;
    if (show && key !== _renderedKey) {
        const tabLabel = el('motif-bubble-tab-label');
        if (tabLabel) tabLabel.textContent = getTerm(titleKey);
        const count = el('motif-bubble-count');
        if (count) count.textContent = String(found.total);
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
        _lifted = false;
        bubble.classList.remove('over-board');
    } else {
        refreshMotifBubbleMetrics();   // open/closed state + the portrait anchor
        // First-appearance announcement for the training-focus list; the hook
        // is once-ever and self-gating, so calling it per repaint is fine.
        if (titleKey === 'trainingFocusTitle') onTrainingFocusShown();
    }
}

/** Wire the tab: shows and hides the list, lifting it over the board in
 *  portrait when it would not fit where it parks. */
export function initMotifBubble() {
    const tab = el('motif-bubble-tab');
    const bubble = el('motif-bubble');
    if (!tab || !bubble) return;
    tab.addEventListener('click', (e) => {
        e.preventDefault();
        // `_fits` is from the last layout pass, which every repaint and resize
        // runs, so it describes the bubble being tapped.
        // The review list has no closed state: its tab only lifts a list that
        // does not fit. Guarded here as well as by the stylesheet's
        // pointer-events, because a <button> stays keyboard-focusable.
        if (_static && _fits) return;
        if (_fits) { _open = !_open; writeOpen(_open); }
        else _lifted = !_lifted;
        refreshMotifBubbleMetrics();
    });

    // Keep the portrait anchor in sync with viewport / orientation changes.
    let raf = 0;
    window.addEventListener('resize', () => {
        if (raf) return;
        raf = requestAnimationFrame(() => { raf = 0; refreshMotifBubbleMetrics(); });
    });
}
