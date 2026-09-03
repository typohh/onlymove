// js/stats-weaknesses.js
//
// The Weaknesses pane of the detailed-stats modal (stats-ui.js is the
// orchestrator): a row for EVERY shipped motif, always, carrying the two
// readings motif-stats.js computes and their intervals, plus — behind an
// unlabelled click on the row itself — links to past mistakes on the motifs
// where a deficit is credible. The intervals come from
// motif-stats.js at the MOTIF_Z gate's own width, NOT at a conventional 95%:
// locked that way, "band clear of the neutral line" and "flagged" are the same
// statement, so a flagged row can never show a band straddling its neutral point.
//
// Both numbers come from the user's OWN attempts and nothing else — see
// motif-stats.js for why no population centre appears anywhere in this path.
//
//   weak <-> strong   the 3PL residual slope on the motif loading, in ranks.
//                     0 = the motif costs you nothing beyond the problem's
//                     difficulty.
//   undervalued <-> overvalued
//                     the share of your wrong answers that erred TOWARD the
//                     motif rather than away from it. The neutral point is the
//                     AVAILABILITY NULL — the share a direction-blind error on
//                     the same problems would have had (motif-stats.js) — and
//                     the track is drawn as the deviation from it, so the
//                     menu's lean is subtracted before anything is called a
//                     lean of the player's.
//
// Why all motifs rather than only the credible ones: a row carrying a visible
// interval makes no claim the interval does not support, so an unresolved axis
// renders as a dot at the neutral point inside a band spanning most of the
// scale, which reads as "we don't know" without a sentence having to say so.
// The claim-shaped affordances still gate on confidence: colour appears only past
// MOTIF_Z (in two steps, see toneWeight), and the mistake-example links appear
// only where the strength reading is credibly negative.
//
// An axis with no named tail is skipped. A placeholder like "style #2" is not a
// finding a user can act on, and the axis keeps steering selection either way.

import { state, setAcceptClicks } from './state.js';
import { MOTIF_Z } from './constants.js';
import { getTerm, localizedMotif } from './i18n.js';
import { normCdf } from './math-utils.js';
import { findExamplesForIssue, probit } from './stats-compute.js';
import { getRecallCandidates } from './recall-model.js';
import { computeMotifStats, isFlagged, usableMotifAttempts } from './motif-stats.js';
import { MIN_ATTEMPTS } from './motif-readings.js';
import { beginExampleReview } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';
import { hideStatsModal, restoreStatsModal, discardStatsModal } from './stats-ui.js';

// Guards the example-link click handler against a double-tap racing the async
// shard load — the second activation would nest a begin inside the first.
let exampleOpenInFlight = false;

// Full-scale of the ranks track each way — the bar's clamp, not a statistic.
// The numeric value is always printed beside the bar, so a clamp costs the bar's
// precision and nothing else.
const RANK_FULL_SCALE = 2.0;
// Same for the tendency track, in lean units (share − null share). The track is
// drawn in DEVIATION space so the availability null is always the visual
// centre: the null share varies per axis and per user, and a neutral mark
// wandering along the track makes the row unreadable — the reader has to
// relocate "balanced" on every row before the dot means anything.
const TENDENCY_FULL_SCALE = 0.25;
// Colour comes in TWO STEPS rather than a continuous ramp: nothing below the
// MOTIF_Z gate, a muted shade from there to twice it, the full shade beyond. The
// steps are readable as a legend — a row's shade names which side of 2 SE it
// falls on — where a smooth ramp only ever said "more confident than that one".
const TONE_STRONG_Z = 2 * MOTIF_Z;

// Builds the motif table consumed by showDetailedStats.
// One pane per READING, not one pane carrying both. `kind` is 'strength'
// (weak <-> strong: how the user SCORES on motif-heavy problems) or 'tendency'
// (undervalued <-> overvalued: which way their mistakes LEAN, against the
// availability null — a wrong pick that loads HIGHER on the motif than the
// answer did is the user valuing it above what the board offered). They answer different questions and are read one at a
// time, so a reader comparing rows within a pane is comparing like with like
// instead of scanning across two axes at once.
export function buildMotifPane(kind, history = state.answerHistory) {
    const labelMap = state.embeddingLabels || {};
    const labelFor = (k, side) => localizedMotif(labelMap[String(k)], side);

    const root = document.createElement('div');
    root.className = 'stats-issues';

    const stats = computeMotifStats(history);
    if (!stats || stats.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNotEnough', { min: MIN_ATTEMPTS, count: usableMotifAttempts(history) });
        root.appendChild(empty);
        return root;
    }

    const rows = [];
    for (const st of stats) {
        // Prefer the positive tail's name; an axis named only on its negative
        // tail is described from that end, with both readings mirrored so
        // "weak" and "overplay" keep meaning the named motif throughout.
        const side = labelFor(st.k, 'positive') ? 'positive'
                   : (labelFor(st.k, 'negative') ? 'negative' : null);
        if (!side) continue;
        const flip = side === 'negative';
        const t = st.tendency, g = st.strength;
        rows.push({
            k: st.k,
            side,
            name: labelFor(st.k, side),
            // A standard error is unsigned, so mirroring leaves it alone.
            strength: g && {
                ranks: flip ? -g.ranks : g.ranks,
                lo: flip ? -g.hi : g.lo,
                hi: flip ? -g.lo : g.hi,
                z: flip ? -g.z : g.z,
                se: g.se,
            },
            // Mirroring a share reflects it — null share included — since
            // "toward the named motif" flips with the tail that names it.
            tendency: t && {
                share: flip ? 1 - t.share : t.share,
                nullShare: flip ? 1 - t.nullShare : t.nullShare,
                lo: flip ? 1 - t.hi : t.lo,
                hi: flip ? 1 - t.lo : t.hi,
                z: flip ? -t.z : t.z,
                n: t.n,
                se: t.se,
            },
        });
    }
    if (rows.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNoAxes');
        root.appendChild(empty);
        return root;
    }

    // Strongest evidence first, on the reading this pane shows.
    rows.sort((a, b) => Math.abs(b[kind]?.z ?? 0) - Math.abs(a[kind]?.z ?? 0));

    const table = document.createElement('div');
    table.className = 'wk-table';
    table.appendChild(buildHeaderRow(kind));

    // Problems currently in the user's recall window — withhold their links so a
    // glance at the examples panel can't be used to ace an imminent recall.
    const recallSet = new Set(getRecallCandidates({ history }).map(c => c.problemId));
    for (const row of rows) table.appendChild(buildMotifRow(kind, row, recallSet, history));
    root.appendChild(table);
    root.appendChild(buildIntervalNote());

    return root;
}

// Names the band's width in the pane's smallest type. Worth one line because
// the width is NOT the conventional one: an unlabelled band reads as ~95%. The
// figures are DERIVED from MOTIF_Z rather than written into the sentence, so
// retuning the shared gate retunes the note with it.
function buildIntervalNote() {
    const note = document.createElement('div');
    note.className = 'wk-note';
    note.textContent = getTerm('statsMotifCiNote',
        { z: MOTIF_Z, pct: Math.round((2 * normCdf(MOTIF_Z) - 1) * 100) })
        + ' ' + getTerm('statsMotifSdNote', { z1: MOTIF_Z, z2: TONE_STRONG_Z });
    return note;
}

// The two poles of the reading this pane shows, each at its own end of the track
// below with the arrow centred, so the header reads as the axis the bar is drawn
// on.
const POLES = {
    strength: ['statsMotifWeak', 'statsMotifStrong'],
    tendency: ['statsMotifUnder', 'statsMotifOver'],
};

function buildHeaderRow(kind) {
    const head = document.createElement('div');
    head.className = 'wk-row wk-head';
    const cell = document.createElement('div');
    cell.className = 'wk-col-head';
    const [lo, hi] = POLES[kind];
    for (const term of [getTerm(lo), '↔', getTerm(hi)]) {
        const span = document.createElement('span');
        span.textContent = term;
        cell.appendChild(span);
    }
    head.appendChild(cell);
    return head;
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));
// Unclamped first: the SE ticks below have to know when a position falls OFF
// the track, which a clamped fraction can no longer say.
const rawRankFrac = (r) => 0.5 + r / (2 * RANK_FULL_SCALE);
const rankFrac = (r) => clamp01(rawRankFrac(r));

// The posterior's own scale, drawn on the track: thin ticks either side of the
// neutral mark at the gate's width and at twice it — the SAME two thresholds
// toneWeight steps at, taken from those constants rather than restated, so a
// reader can see WHY a row is uncoloured, muted or full instead of holding the
// legend in the note. That also gives the track a unit: without them the dot's
// distance from centre is only readable against the printed value, and how far
// out it is IN STANDARD ERRORS — the thing the row is actually a claim about —
// is not on the picture at all. A tick landing outside the track is DROPPED
// rather than clamped: parked on the end it would claim a threshold at a place
// the scale does not reach.
function sdTicks(rawFrac, neutral, se) {
    if (!(se > 0)) return [];
    const ticks = [];
    for (const level of [MOTIF_Z, TONE_STRONG_Z]) {
        for (const dir of [-1, 1]) {
            const frac = rawFrac(neutral + dir * level * se);
            if (frac > 0 && frac < 1) ticks.push({ frac, strong: level === TONE_STRONG_Z });
        }
    }
    return ticks;
}

// Colour weight for a statistic sitting z posterior SEs from its neutral point:
// none below the MOTIF_Z gate, half-strength between the gate and TONE_STRONG_Z,
// full beyond it.
function toneWeight(z) {
    if (!Number.isFinite(z)) return 0;
    const a = Math.abs(z);
    if (a <= MOTIF_Z) return 0;
    return a <= TONE_STRONG_Z ? 0.5 : 1;
}

// One metric cell: a track with the neutral point marked, the credible interval
// as a band, and the point estimate as a dot. `tone` names the direction for CSS;
// `weight` is how strongly to paint it.
//
// The track carries the interval and NO printed range. The band already shows
// how wide the posterior is, in the one place it can be compared against the
// neutral mark at a glance, so the numbers restated it in a form that has to be
// read twice — and the row is a claim about a direction, not a measurement
// anybody acts on to two decimals. The point estimate stays, on the label's own
// line, because that is what names the size of the effect.
function buildMetricCell({ frac, loFrac, hiFrac, ticks = [] }) {
    const cell = document.createElement('div');
    cell.className = 'wk-metric';

    const track = document.createElement('div');
    track.className = 'wk-track';

    const band = document.createElement('span');
    band.className = 'wk-ci';
    const lo = Math.min(loFrac, hiFrac);
    const hi = Math.max(loFrac, hiFrac);
    band.style.left = `${(lo * 100).toFixed(2)}%`;
    band.style.width = `${Math.max(0.5, (hi - lo) * 100).toFixed(2)}%`;
    track.appendChild(band);

    // Over the band, not under it: the ticks and the band overlap on most rows
    // — the band is the gate's width around the estimate and the ticks are the
    // same width around the neutral point — and a tick hidden exactly where the
    // reading sits is hidden where it is being read.
    for (const t of ticks) {
        const tick = document.createElement('span');
        tick.className = 'wk-sd';
        if (t.strong) tick.dataset.strong = '1';
        tick.style.left = `${(t.frac * 100).toFixed(2)}%`;
        track.appendChild(tick);
    }

    // The neutral mark sits at the track's centre on every pane: each reading
    // is drawn as its deviation from its own neutral point (the tendency
    // pane's availability null included — see TENDENCY_FULL_SCALE).
    const mid = document.createElement('span');
    mid.className = 'wk-mid';
    track.appendChild(mid);

    const dot = document.createElement('span');
    dot.className = 'wk-dot';
    dot.style.left = `${(frac * 100).toFixed(2)}%`;
    track.appendChild(dot);

    cell.appendChild(track);
    return cell;
}

const fmtRank = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
// The tendency reading is stored as the SHARE of errors that lean toward the
// motif — but "42%" gives a reader no way to see where balance sits. Shown as
// a signed deviation from the NULL SHARE instead (the share a direction-blind
// error on the same problems would have had), read as the share of the user's
// errors that would have to go the other way to match a typical error's lean:
// `share - nullShare` exactly, undoubled — doubling would state the swing
// between the two sides rather than the count of choices to change, which is
// not what the number is read as. The bar is drawn in the same deviation
// space (cellFor), so the dot's distance from the centre mark IS this number.
const fmtLean = (share, nullShare) => {
    const v = Math.round((share - nullShare) * 100);
    return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)}%`;
};

// A reading with no data renders as a neutral dot spanning the whole track —
// visibly "we don't know" rather than a confident-looking zero.
const EMPTY_CELL = { frac: 0.5, loFrac: 0, hiFrac: 1, tone: 'none', weight: 0 };

// The reading this pane draws, as the cell geometry plus the text that names it.
// Reading-specific in three ways that do not generalise: the scale the value sits
// on, how it is formatted, and whether the two ends are VALENCED. Strength is —
// worse on motif-heavy problems is bad and better is good — so it takes the
// red/green pair. Tendency is not: both ends are the same finding, a lean the
// player has and could train away, so both take ONE colour and the direction is
// carried by the dot's side of the centre mark alone. A two-hue pair there read
// as a valence the reading does not have.
function cellFor(kind, row) {
    if (kind === 'strength') {
        const g = row.strength;
        if (!g) return null;
        return {
            frac: rankFrac(g.ranks),
            loFrac: rankFrac(g.lo),
            hiFrac: rankFrac(g.hi),
            ticks: sdTicks(rawRankFrac, 0, g.se),
            valueText: getTerm('statsMotifRanks', { value: fmtRank(g.ranks) }),
            tone: g.z < 0 ? 'weak' : 'strong',
            weight: toneWeight(g.z),
        };
    }
    const t = row.tendency;
    if (!t) return null;
    // Deviation space: the dot's distance from the centre IS the printed lean,
    // and the centre IS the availability null, on every row.
    const rawLeanFrac = (v) => 0.5 + (v - t.nullShare) / (2 * TENDENCY_FULL_SCALE);
    const leanFrac = (v) => clamp01(rawLeanFrac(v));
    return {
        frac: leanFrac(t.share),
        loFrac: leanFrac(t.lo),
        hiFrac: leanFrac(t.hi),
        ticks: sdTicks(rawLeanFrac, t.nullShare, t.se),
        valueText: fmtLean(t.share, t.nullShare),
        tone: t.z > 0 ? 'over' : 'under',
        weight: toneWeight(t.z),
    };
}

function buildMotifRow(kind, row, recallSet, history) {
    const el = document.createElement('div');
    el.className = 'wk-row';

    const spec = cellFor(kind, row);
    // Tone and weight go on the ROW, not on the track: the value line and the
    // track are siblings and both read them, so the number cannot end up
    // disagreeing with the bar underneath it.
    el.dataset.tone = spec ? spec.tone : 'none';
    el.style.setProperty('--wk-weight', (spec ? spec.weight : 0).toFixed(3));

    // Name and value share one centred line above the track they describe, so a
    // row is read left-to-right as one statement rather than as a label in one
    // column and a number in another.
    const lead = document.createElement('div');
    lead.className = 'wk-lead';
    const name = document.createElement('span');
    name.className = 'wk-name';
    name.textContent = row.name;
    lead.appendChild(name);
    const value = document.createElement('span');
    value.className = 'wk-value';
    value.textContent = spec ? spec.valueText : getTerm('statsMotifNoData');
    lead.appendChild(value);
    el.appendChild(lead);

    el.appendChild(buildMetricCell(spec || EMPTY_CELL));

    // Mistake examples are offered only where the strength reading is credibly
    // negative — they illustrate a deficit, and there is nothing for them to
    // show on an axis the data has not resolved or has resolved as a strength.
    // So they belong to the STRENGTH pane and appear nowhere else, even though
    // WHICH mistakes is still read from the tendency reading, which is what keeps
    // the examples and the row above them telling one story.
    if (kind !== 'strength') return el;
    if (!row.strength || !isFlagged(row.strength) || row.strength.z > 0) return el;
    const t2 = row.tendency;
    const quadrant = (t2 && isFlagged(t2))
        ? (t2.z > 0 ? 'tooMuch' : 'notEnough')
        : 'badAt';
    const issue = {
        k: row.k,
        quadrant,
        side: row.side,
        // Per-attempt evidence is thresholded to match the row's own confidence
        // (see findExamplesForIssue). Capped at the 99% probit: the figure
        // saturates on a well-resolved axis, and probit(1) = Infinity would
        // demand impossible evidence and blank the list for exactly the rows the
        // user most wants examples from.
        p: Math.min(normCdf(Math.abs(row.strength.z)), 0.99),
    };

    const list = document.createElement('div');
    list.className = 'stats-issue-examples-list wk-examples-list';
    list.hidden = true;
    el.appendChild(list);

    // The ROW ITSELF opens the list, and nothing says so: no label, no cursor
    // change, no marker distinguishing a row that has examples from one that
    // does not. The pane is a set of readings to be read, and a visible control
    // on every credible row turned it into a menu of things to click; unlabelled,
    // the examples stay there for a reader who tries the row and cost the layout
    // nothing for everyone else.
    //
    // Clicks inside the list belong to the example links — letting them through
    // would fold the list shut under the pointer as one is opened.
    let populated = false;
    el.addEventListener('click', (e) => {
        if (list.contains(e.target)) return;
        if (!populated) {
            populateExamplesList(list, issue, recallSet, history);
            populated = true;
        }
        list.hidden = !list.hidden;
    });
    return el;
}

function populateExamplesList(list, issue, recallSet, history = state.answerHistory) {
    const threshold = probit(issue.p);
    const examples = findExamplesForIssue(history, issue.k, issue.quadrant, recallSet,
                                          threshold, issue.side === 'negative' ? -1 : 1);
    if (examples.length === 0) {
        const none = document.createElement('div');
        none.className = 'stats-issue-examples-empty';
        none.textContent = getTerm('statsNoExamples');
        list.appendChild(none);
        return;
    }
    const MAX_EXAMPLES = 10;
    const shown = examples.slice(0, MAX_EXAMPLES);
    for (const ex of shown) {
        const link = document.createElement('a');
        link.className = 'stats-issue-example-link';
        // In-instance review: the example opens over this instance in analysis
        // mode rather than spawning a second app instance. The stats modal hides
        // (DOM intact) and reappears when the review exits via the analysis Exit
        // button (main.js → endExampleReview). No href — every activation routes
        // through this handler.
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            if (exampleOpenInFlight) return;
            // A post-answer transition owns the board (acceptClicks false) —
            // entering a review over it would race the in-flight load.
            if (!state.acceptClicks) return;
            exampleOpenInFlight = true;
            // Lock the board for the async shard load: once the modal hides, the
            // problem underneath would otherwise take stray clicks until
            // enterAnalysisMode (which re-enables clicks) takes over.
            setAcceptClicks(false);
            try {
                hideStatsModal();
                const staged = await beginExampleReview(ex.problemId, {
                    onExit: restoreStatsModal,
                    onAbandon: discardStatsModal,
                });
                if (!staged) {
                    restoreStatsModal();
                    setAcceptClicks(true);
                    return;
                }
                await enterAnalysisMode();
            } finally {
                exampleOpenInFlight = false;
            }
        });
        const sizeLabel = (ex.size === 9 || ex.size === 13) ? `${ex.size}x${ex.size} · ` : '';
        link.textContent = `${ex.problemId} · ${sizeLabel}${formatSignalStrength(issue.quadrant, ex.ec, ex.ep)}`;
        list.appendChild(link);
    }
    if (examples.length > shown.length) {
        const more = document.createElement('div');
        more.className = 'stats-issue-examples-empty';
        more.textContent = getTerm('statsMoreExamples', { count: examples.length - shown.length });
        list.appendChild(more);
    }
}

// Compact per-quadrant signal-strength readout. Embeddings are unit-variance
// z-scores along each axis, so values are reported in σ — sign included because
// each quadrant's evidence shape involves a specific signed pattern.
function formatSignalStrength(quadrant, ec, ep) {
    const fmt = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}σ`;
    // List the more-emphasized axis first per quadrant:
    //   tooMuch: motif in pick (positive ep) is the foreground signal
    //   notEnough/badAt: motif in correct (positive ec) is the foreground signal
    const vs = getTerm('statsSignalVs');
    if (quadrant === 'tooMuch') return `${fmt(ep)} ${vs} ${fmt(ec)}`;
    return `${fmt(ec)} ${vs} ${fmt(ep)}`;
}
