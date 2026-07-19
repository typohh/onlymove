// js/stats-weaknesses.js
//
// The Weaknesses pane of the detailed-stats modal (stats-ui.js is the
// orchestrator): the "Likely issues" list derived from the augmented weakness
// model's posterior, each issue expandable into links to past mistakes.
// Rendering-only — the numeric example selection (findExamplesForIssue) and the
// probit threshold live in stats-compute.js.

import { state, setAcceptClicks } from './state.js';
import { getTerm, localizedMotif } from './i18n.js';
import { normCdf } from './math-utils.js';
import { findExamplesForIssue, probit, computeAxisDirections } from './stats-compute.js';
import { getRecallCandidates } from './model.js';
import { beginExampleReview } from './problem-loader.js';
import { enterAnalysisMode } from './analysis.js';
import { hideStatsModal, restoreStatsModal, discardStatsModal } from './stats-ui.js';

// Guards the example-link click handler against a double-tap racing the async
// shard load — the second activation would nest a begin inside the first.
let exampleOpenInFlight = false;

// Builds the "Likely issues" DOM block consumed by showDetailedStats. Each
// surfaced (axis k, category) issue is one row with a toggle that lazily
// fetches matching past mistakes from answerHistory.
//
// Two-level detection:
//   Level 1 — IS axis k a weakness? — from the augmented model's posterior on
//   the SUM β_correct[k] + β_picked[k] (mean from weaknessParams, variance
//   var_c + var_p + 2·cov from weaknessSe/weaknessAxisCov). The sum is the
//   direction the training data identifies: ~80% of rows have picked = correct
//   (the fallback + every correct attempt), so the correct/picked SPLIT is
//   prior-dominated and must not drive detection. The old sign-quadrant
//   scheme did exactly that, and its cov<0 gate (structural under the
//   collinearity) left `badAt` as the only category that could ever render.
//   Level 2 — WHICH WAY does the user err? — a paired sign test on the
//   (picked − correct) axis loading over the user's own wrong attempts
//   (computeAxisDirections in stats-compute.js): overshoot → tooMuch,
//   undershoot → notEnough, inconclusive → badAt ("inconsistently") as the
//   honest residual. Needs no model and no assumption about which moves are
//   reasonable alternatives.
export function buildLikelyIssuesElement(history = state.answerHistory) {
    // P(β_correct + β_picked < 0) needed to flag an axis as a weakness. This is
    // a one-sided posterior probability on a single identified quantity (a
    // 50% bar would flag half the axes under a flat posterior), unlike the old
    // 0.50 bar on four-way quadrant mass.
    const WEAK_PROB_THRESHOLD = 0.80;
    // Sign-test posterior needed to call a direction, and the minimum number
    // of attributed errors before a direction can be called at all.
    const DIRECTION_PROB_THRESHOLD = 0.85;
    const MIN_DIRECTION_N = 3;
    const weaknessParams = state.weaknessParams;
    const weaknessSe = state.weaknessSe;
    const weaknessAxisCov = state.weaknessAxisCov;
    const labelMap = state.embeddingLabels || {};
    // side: which TAIL of the bipolar axis the weakness sits on. A negative-
    // side weakness is only surfaced when that tail is actually named
    // (labelFor returns null then and the axis is skipped) — some axes only
    // carry a motif on one side (labels.json v4).
    const labelFor = (k, side = 'positive') => {
        const entry = labelMap[String(k)];
        const text = localizedMotif(entry, side);
        if (text) return text;
        return side === 'positive' ? getTerm('statsStyleFallback', { k }) : null;
    };
    // Skill-noun templates (labels are nouns like "reading"/"shape", not move
    // recipes): the level-2 direction picks one of three blunt sentences about
    // the flagged motif — rely-too-much / don't-trust-enough / weak-at.
    const PHRASES = {
        tooMuch:   (l) => getTerm('statsIssueTooMuch', { motif: l }),
        notEnough: (l) => getTerm('statsIssueNotEnough', { motif: l }),
        badAt:     (l) => getTerm('statsIssueBadAt', { motif: l }),
    };

    const root = document.createElement('div');
    root.className = 'stats-issues';

    const heading = document.createElement('div');
    heading.className = 'stats-issues-heading';
    heading.textContent = getTerm('statsIssuesHeading');
    root.appendChild(heading);

    const haveModel = (
        Array.isArray(weaknessParams) && weaknessParams.length >= 4 &&
        Array.isArray(weaknessSe) && weaknessSe.length === weaknessParams.length &&
        Array.isArray(weaknessAxisCov)
    );
    if (!haveModel) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNotEnough');
        root.appendChild(empty);
        return root;
    }

    const K = (weaknessParams.length - 2) / 2;

    // Level 1: flag axes whose identified sum coefficient is credibly non-zero
    // in EITHER direction — the axes are BIPOLAR (residual skill axes,
    // 2026-07-18): a credibly NEGATIVE sum = weak at the positive-tail motif,
    // a credibly POSITIVE sum = weak at the NEGATIVE-tail motif. The negative
    // side only surfaces when that tail is named in labels.json; an unnamed
    // tail has nothing actionable to say and is skipped.
    const flagged = [];
    for (let k = 0; k < K; k++) {
        const mu_c = weaknessParams[2 + k];
        const mu_p = weaknessParams[2 + K + k];
        const var_c = weaknessSe[2 + k] * weaknessSe[2 + k];
        const var_p = weaknessSe[2 + K + k] * weaknessSe[2 + K + k];
        const cov = weaknessAxisCov[k] || 0;
        if (![mu_c, mu_p, var_c, var_p, cov].every(Number.isFinite)) continue;
        const varSum = var_c + var_p + 2 * cov;
        if (!(varSum > 0)) continue;
        const pWeakPos = normCdf(-(mu_c + mu_p) / Math.sqrt(varSum));
        if (pWeakPos >= WEAK_PROB_THRESHOLD) {
            flagged.push({ k, p: pWeakPos, side: 'positive' });
        } else if (1 - pWeakPos >= WEAK_PROB_THRESHOLD
                   && labelFor(k, 'negative')) {
            flagged.push({ k, p: 1 - pWeakPos, side: 'negative' });
        }
    }

    // Level 2: qualify each flagged axis's direction from the paired
    // picked-vs-correct contrast; inconclusive stays "inconsistently".
    // pOver is P(picked loads HIGHER than correct on axis k) — overshoot
    // toward the POSITIVE tail. For a negative-side weakness the named motif
    // is the negative pole, so over/under invert.
    const directions = computeAxisDirections(history, flagged.map(f => f.k));
    const issues = flagged.map(({ k, p, side }) => {
        let quadrant = 'badAt';
        const dir = directions.get(k);
        if (dir && dir.nPos + dir.nNeg >= MIN_DIRECTION_N) {
            const pOver = side === 'negative' ? 1 - dir.pOver : dir.pOver;
            if (pOver >= DIRECTION_PROB_THRESHOLD) quadrant = 'tooMuch';
            else if (1 - pOver >= DIRECTION_PROB_THRESHOLD) quadrant = 'notEnough';
        }
        return { k, side, quadrant, p,
                 sentence: PHRASES[quadrant](labelFor(k, side)) };
    });
    issues.sort((a, b) => b.p - a.p);

    if (issues.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNoneAbove', { percent: (WEAK_PROB_THRESHOLD * 100).toFixed(0) });
        root.appendChild(empty);
        return root;
    }

    // Problems currently in the user's recall window — withhold their links so
    // a glance at the examples panel can't be used to ace an imminent recall.
    const recallSet = new Set(getRecallCandidates({ history }).map(c => c.problemId));

    for (const issue of issues) {
        root.appendChild(buildIssueRow(issue, recallSet, history));
    }
    return root;
}

// Renders one (axis k, quadrant) row: sentence + probability + an "examples"
// toggle that expands a list of links to past attempts matching the issue.
function buildIssueRow(issue, recallSet, history = state.answerHistory) {
    const row = document.createElement('div');
    row.className = 'stats-issue';

    const head = document.createElement('div');
    head.className = 'stats-issue-head';

    const prob = document.createElement('span');
    prob.className = 'stats-issue-prob';
    prob.textContent = `${(issue.p * 100).toFixed(0)}%`;

    const text = document.createElement('span');
    text.className = 'stats-issue-text';
    text.textContent = issue.sentence;

    const toggle = document.createElement('button');
    toggle.type = 'button';
    toggle.className = 'stats-issue-examples-toggle';
    toggle.textContent = getTerm('statsExamples');

    head.append(prob, text, toggle);
    row.appendChild(head);

    const list = document.createElement('div');
    list.className = 'stats-issue-examples-list';
    list.hidden = true;
    row.appendChild(list);

    let populated = false;
    toggle.addEventListener('click', () => {
        if (!populated) {
            populateExamplesList(list, issue, recallSet, history);
            populated = true;
        }
        list.hidden = !list.hidden;
        toggle.textContent = list.hidden ? getTerm('statsExamples') : getTerm('statsHide');
    });
    return row;
}

function populateExamplesList(list, issue, recallSet, history = state.answerHistory) {
    // Threshold per-attempt evidence strength to match the issue's posterior
    // confidence — see findExamplesForIssue for the rationale. Capped at the
    // 99% probit (~2.33σ): the level-1 posterior saturates to 1.0 on strong
    // axes, and probit(1) = Infinity would demand impossible evidence and
    // blank the examples list for exactly the most confident issues.
    const threshold = probit(Math.min(issue.p, 0.99));
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
        // In-instance review: a spawned tab would land on the tab-lock overlay,
        // so the example opens over this instance in analysis mode. The stats
        // modal hides (DOM intact) and reappears when the review exits via the
        // analysis Exit button (main.js → endExampleReview). No href — every
        // activation routes through this handler.
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            if (exampleOpenInFlight) return;
            // A post-answer transition owns the board (acceptClicks false) —
            // entering a review over it would race the in-flight load.
            if (!state.acceptClicks) return;
            exampleOpenInFlight = true;
            // Lock the board for the async shard load: once the modal hides,
            // the problem underneath would otherwise take stray clicks until
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
// z-scores along each axis, so values are reported in σ — sign included
// because each quadrant's evidence shape involves a specific signed pattern.
function formatSignalStrength(quadrant, ec, ep) {
    const fmt = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}σ`;
    // List the more-emphasized axis first per quadrant:
    //   tooMuch: motif in pick (positive ep) is the foreground signal
    //   notEnough/badAt: motif in correct (positive ec) is the foreground signal
    if (quadrant === 'tooMuch') return `${fmt(ep)} vs ${fmt(ec)}`;
    return `${fmt(ec)} vs ${fmt(ep)}`;
}
