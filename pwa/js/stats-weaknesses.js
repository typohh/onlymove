// js/stats-weaknesses.js
//
// The Weaknesses pane of the detailed-stats modal (stats-ui.js is the
// orchestrator): a row for EVERY shipped motif, always, carrying the two
// readings motif-stats.js computes and their 95% intervals, plus links to past
// mistakes on the motifs where a deficit is credible.
//
// Both numbers come from the user's OWN attempts and nothing else — see
// motif-stats.js for why no population centre appears anywhere in this path.
//
//   weak <-> strong   the 3PL residual slope on the motif loading, in ranks.
//                     0 = the motif costs you nothing beyond the problem's
//                     difficulty.
//   underplay <-> overplay
//                     the share of your wrong answers that erred TOWARD the
//                     motif rather than away from it. 50% = your mistakes lean
//                     neither way, which is balance, not an average of other
//                     players.
//
// Why all motifs rather than only the credible ones: a row carrying a visible
// interval makes no claim the interval does not support, so an unresolved axis
// renders as a dot at the neutral point inside a band spanning most of the
// scale, which reads as "we don't know" without a sentence having to say so.
// The claim-shaped affordances still gate on confidence: colour ramps from
// neutral only past MOTIF_Z, and the mistake-example links appear only where the
// strength reading is credibly negative.
//
// An axis with no named tail is skipped. A placeholder like "style #2" is not a
// finding a user can act on, and the axis keeps steering selection either way.

import { state, setAcceptClicks } from './state.js';
import { MOTIF_Z } from './constants.js';
import { getTerm, localizedMotif } from './i18n.js';
import { normCdf } from './math-utils.js';
import { findExamplesForIssue, probit } from './stats-compute.js';
import { getRecallCandidates } from './recall-model.js';
import { computeMotifStats, isFlagged } from './motif-stats.js';
import { beginExampleReview } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';
import { hideStatsModal, restoreStatsModal, discardStatsModal } from './stats-ui.js';

// Guards the example-link click handler against a double-tap racing the async
// shard load — the second activation would nest a begin inside the first.
let exampleOpenInFlight = false;

// Credible-interval width. 95% is what "CI" means unqualified, and it is wide
// on purpose: the contrast is the direction the picked = correct collinearity
// starves, so for most users that column's band SHOULD span most of the scale.
// Full-scale of the ranks track each way. The numeric value is always printed
// beside the bar, so a clamp costs the bar's precision and nothing else.
const RANK_FULL_SCALE = 2.0;
// Colour ramps from fully neutral at the MOTIF_Z gate to fully saturated one
// further SE out, so a row below the gate carries no colour at all — the pane
// makes no claim there, and a tinted row reads as one.
const TONE_FULL_Z = MOTIF_Z + 1;

// Builds the motif table consumed by showDetailedStats.
export function buildLikelyIssuesElement(history = state.answerHistory) {
    const labelMap = state.embeddingLabels || {};
    const labelFor = (k, side) => localizedMotif(labelMap[String(k)], side);

    const root = document.createElement('div');
    root.className = 'stats-issues';

    const stats = computeMotifStats(history);
    if (!stats || stats.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNotEnough');
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
            strength: g && {
                ranks: flip ? -g.ranks : g.ranks,
                lo: flip ? -g.hi : g.lo,
                hi: flip ? -g.lo : g.hi,
                z: flip ? -g.z : g.z,
            },
            // Mirroring a share reflects it about the 50% balance point.
            tendency: t && {
                share: flip ? 1 - t.share : t.share,
                lo: flip ? 1 - t.hi : t.lo,
                hi: flip ? 1 - t.lo : t.hi,
                z: flip ? -t.z : t.z,
                n: t.n,
            },
        });
    }
    if (rows.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNotEnough');
        root.appendChild(empty);
        return root;
    }

    // Strongest evidence first, on the strength column — the one the headers
    // lead with and the one the example links gate on.
    rows.sort((a, b) => Math.abs(b.strength?.z ?? 0) - Math.abs(a.strength?.z ?? 0));

    const table = document.createElement('div');
    table.className = 'wk-table';
    table.appendChild(buildHeaderRow());

    // Problems currently in the user's recall window — withhold their links so a
    // glance at the examples panel can't be used to ace an imminent recall.
    const recallSet = new Set(getRecallCandidates({ history }).map(c => c.problemId));
    for (const row of rows) table.appendChild(buildMotifRow(row, recallSet, history));
    root.appendChild(table);

    return root;
}

function buildHeaderRow() {
    const head = document.createElement('div');
    head.className = 'wk-row wk-head';
    head.appendChild(document.createElement('div')).className = 'wk-name';
    const metrics = document.createElement('div');
    metrics.className = 'wk-metrics';
    // Each pole sits at its own end of the track it labels, arrow centred, so the
    // header reads as the axis the bar below it is drawn on.
    for (const [lo, hi] of [['statsMotifWeak', 'statsMotifStrong'],
                            ['statsMotifUnder', 'statsMotifOver']]) {
        const cell = document.createElement('div');
        cell.className = 'wk-col-head';
        for (const t of [getTerm(lo), '↔', getTerm(hi)]) {
            const span = document.createElement('span');
            span.textContent = t;
            cell.appendChild(span);
        }
        metrics.appendChild(cell);
    }
    head.appendChild(metrics);
    return head;
}

const clamp01 = (v) => Math.max(0, Math.min(1, v));
const rankFrac = (r) => clamp01(0.5 + r / (2 * RANK_FULL_SCALE));

// Colour weight for a statistic sitting z posterior SEs from the population
// centre: nothing at all below the MOTIF_Z gate, ramping to full one SE past it.
function toneWeight(z) {
    if (!Number.isFinite(z)) return 0;
    return clamp01((Math.abs(z) - MOTIF_Z) / (TONE_FULL_Z - MOTIF_Z));
}

// One metric cell: a track with the population centre marked, the credible
// interval as a band, the point estimate as a dot, and the numbers beside it.
// `tone` names the direction for CSS; `weight` is how strongly to paint it.
function buildMetricCell({ frac, loFrac, hiFrac, valueText, ciText, tone, weight }) {
    const cell = document.createElement('div');
    cell.className = 'wk-metric';
    cell.dataset.tone = tone;
    cell.style.setProperty('--wk-weight', weight.toFixed(3));

    const track = document.createElement('div');
    track.className = 'wk-track';

    const band = document.createElement('span');
    band.className = 'wk-ci';
    const lo = Math.min(loFrac, hiFrac);
    const hi = Math.max(loFrac, hiFrac);
    band.style.left = `${(lo * 100).toFixed(2)}%`;
    band.style.width = `${Math.max(0.5, (hi - lo) * 100).toFixed(2)}%`;
    track.appendChild(band);

    const mid = document.createElement('span');
    mid.className = 'wk-mid';
    track.appendChild(mid);

    const dot = document.createElement('span');
    dot.className = 'wk-dot';
    dot.style.left = `${(frac * 100).toFixed(2)}%`;
    track.appendChild(dot);

    cell.appendChild(track);

    const value = document.createElement('div');
    value.className = 'wk-value';
    value.textContent = valueText;
    cell.appendChild(value);

    const ci = document.createElement('div');
    ci.className = 'wk-ci-text';
    ci.textContent = ciText;
    cell.appendChild(ci);
    return cell;
}

const fmtRank = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`;
// The tendency reading is stored as the SHARE of errors that lean toward the
// motif, whose neutral point is 0.5 — but "42%" gives a reader no way to see
// that 50 is the balance and 42 sits below it. Shown as a signed deviation
// instead, and specifically as the SHARE OF THE USER'S ERRORS THAT WOULD HAVE TO
// GO THE OTHER WAY to reach balance: 6 errors leaning toward the motif against 4
// leaning away is one error out of ten, i.e. +10%. That is `share - 0.5`
// exactly, undoubled — doubling would state the swing between the two sides
// rather than the count of choices to change, which is not what the number is
// read as. The bar geometry is unchanged (already centred on 0.5); only the
// printed value moves.
const fmtLean = (share) => {
    const v = Math.round((share - 0.5) * 100);
    return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v)}%`;
};

// A reading with no data renders as a neutral dot spanning the whole track —
// visibly "we don't know" rather than a confident-looking zero.
const EMPTY_CELL = { frac: 0.5, loFrac: 0, hiFrac: 1, tone: 'none', weight: 0 };

function buildMotifRow(row, recallSet, history) {
    const el = document.createElement('div');
    el.className = 'wk-row';

    const name = document.createElement('div');
    name.className = 'wk-name';
    name.textContent = row.name;
    el.appendChild(name);

    const metrics = document.createElement('div');
    metrics.className = 'wk-metrics';
    const g = row.strength;
    metrics.appendChild(buildMetricCell(g ? {
        frac: rankFrac(g.ranks),
        loFrac: rankFrac(g.lo),
        hiFrac: rankFrac(g.hi),
        valueText: getTerm('statsMotifRanks', { value: fmtRank(g.ranks) }),
        ciText: `${fmtRank(g.lo)} … ${fmtRank(g.hi)}`,
        // Valenced: worse on motif-heavy problems is bad, better is good.
        tone: g.z < 0 ? 'weak' : 'strong',
        weight: toneWeight(g.z),
    } : { ...EMPTY_CELL, valueText: getTerm('statsMotifNoData'), ciText: '' }));
    const t = row.tendency;
    metrics.appendChild(buildMetricCell(t ? {
        frac: t.share,
        loFrac: t.lo,
        hiFrac: t.hi,
        valueText: fmtLean(t.share),
        ciText: `${fmtLean(t.lo)} … ${fmtLean(t.hi)}`,
        // NOT valenced: both ends are training directions, neither is a virtue,
        // so this column must not reuse the weak/strong red-green pair.
        tone: t.z > 0 ? 'over' : 'under',
        weight: toneWeight(t.z),
    } : { ...EMPTY_CELL, valueText: getTerm('statsMotifNoData'), ciText: '' }));
    el.appendChild(metrics);

    // Mistake examples are offered only where the strength reading is credibly
    // negative — they illustrate a deficit, and there is nothing for them to
    // show on an axis the data has not resolved or has resolved as a strength.
    // WHICH mistakes is read from the tendency column, so the examples and the
    // numbers above them tell one story.
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

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'stats-issue-examples-toggle wk-examples-toggle';
    toggle.textContent = getTerm('statsExamples');
    el.appendChild(toggle);

    const list = document.createElement('div');
    list.className = 'stats-issue-examples-list wk-examples-list';
    list.hidden = true;
    el.appendChild(list);

    let populated = false;
    toggle.addEventListener('click', () => {
        if (!populated) {
            populateExamplesList(list, issue, recallSet, history);
            populated = true;
        }
        list.hidden = !list.hidden;
        toggle.textContent = list.hidden ? getTerm('statsExamples') : getTerm('statsHide');
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
