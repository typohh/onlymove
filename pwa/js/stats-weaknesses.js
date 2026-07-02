// js/stats-weaknesses.js
//
// The Weaknesses pane of the detailed-stats modal (stats-ui.js is the
// orchestrator): the "Likely issues" list derived from the augmented weakness
// model's posterior, each issue expandable into links to past mistakes.
// Rendering-only — the numeric example selection (findExamplesForIssue) and the
// probit threshold live in stats-compute.js.

import { state } from './state.js';
import { getTerm, localizedMotif } from './i18n.js';
import { normCdf } from './math-utils.js';
import { findExamplesForIssue, probit, computeAxisDirections } from './stats-compute.js';
import { getRecallCandidates } from './model.js';

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
    const labelFor = (k) => {
        const entry = labelMap[String(k)];
        return localizedMotif(entry) || getTerm('statsStyleFallback', { k });
    };
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

    // Level 1: flag axes whose identified sum coefficient is credibly negative.
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
        const pWeak = normCdf(-(mu_c + mu_p) / Math.sqrt(varSum));
        if (pWeak >= WEAK_PROB_THRESHOLD) flagged.push({ k, p: pWeak });
    }

    // Level 2: qualify each flagged axis's direction from the paired
    // picked-vs-correct contrast; inconclusive stays "inconsistently".
    const directions = computeAxisDirections(history, flagged.map(f => f.k));
    const issues = flagged.map(({ k, p }) => {
        let quadrant = 'badAt';
        const dir = directions.get(k);
        if (dir && dir.nPos + dir.nNeg >= MIN_DIRECTION_N) {
            if (dir.pOver >= DIRECTION_PROB_THRESHOLD) quadrant = 'tooMuch';
            else if (1 - dir.pOver >= DIRECTION_PROB_THRESHOLD) quadrant = 'notEnough';
        }
        return { k, quadrant, p, sentence: PHRASES[quadrant](labelFor(k)) };
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
    const examples = findExamplesForIssue(history, issue.k, issue.quadrant, recallSet, threshold);
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
        link.href = `?id=${encodeURIComponent(ex.problemId)}&o=0&close_on_next=1`;
        link.target = '_blank';
        link.rel = 'noopener';
        // Primary-click goes through window.open() so the spawned tab is
        // script-opened — window.close() in the spawned window is only
        // permitted when the opener was a script. Modifier-click and
        // middle-click fall through to the native target=_blank behavior.
        link.addEventListener('click', (e) => {
            if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
            e.preventDefault();
            window.open(link.href, '_blank', 'noopener');
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
