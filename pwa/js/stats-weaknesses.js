// js/stats-weaknesses.js
//
// The Weaknesses pane of the detailed-stats modal (stats-ui.js is the
// orchestrator): the "Likely issues" list derived from the augmented weakness
// model's posterior, each issue expandable into links to past mistakes.
// Rendering-only — the numeric example selection (findExamplesForIssue) and the
// probit threshold live in stats-compute.js.

import { state } from './state.js';
import { getTerm, localizedMotif } from './i18n.js';
import { quadrantProbs } from './bivariate-normal.js';
import { findExamplesForIssue, probit } from './stats-compute.js';
import { getRecallCandidates } from './model.js';

// Builds the "Likely issues" DOM block consumed by showDetailedStats. Each
// surfaced (axis k, quadrant) issue is one row with a toggle that lazily
// fetches matching past mistakes from answerHistory. Quadrants mirror the
// augmented model's sign analysis on (β_correct[k], β_picked[k]):
//   tooMuch    β_c > 0, β_p < 0  → user mis-applies motif k
//   notEnough  β_c < 0, β_p > 0  → user under-applies motif k
//   badAt      β_c < 0, β_p < 0  → user struggles with motif k overall
// `offDiagonal: true` quadrants need cov(β_c, β_p) ≥ 0 — anti-correlated
// posteriors (typical when most rows have picked = correct fallback) can land
// high mass in BOTH off-diagonal quadrants simultaneously without the data
// actually distinguishing over- vs under-application. badAt skips the gate
// because its both-negative sign-pattern is identifiable from β_c + β_p.
export function buildLikelyIssuesElement(history = state.answerHistory) {
    const PROB_THRESHOLD = 0.50;
    const weaknessParams = state.weaknessParams;
    const weaknessSe = state.weaknessSe;
    const weaknessAxisCov = state.weaknessAxisCov;
    const labelMap = state.embeddingLabels || {};
    const labelFor = (k) => {
        const entry = labelMap[String(k)];
        return localizedMotif(entry) || getTerm('statsStyleFallback', { k });
    };
    const QUADRANTS = [
        { key: 'tooMuch',   offDiagonal: true,  phrase: (l) => getTerm('statsIssueTooMuch', { motif: l }) },
        { key: 'notEnough', offDiagonal: true,  phrase: (l) => getTerm('statsIssueNotEnough', { motif: l }) },
        { key: 'badAt',     offDiagonal: false, phrase: (l) => getTerm('statsIssueBadAt', { motif: l }) },
    ];

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
    const issues = [];
    for (let k = 0; k < K; k++) {
        const mu_c = weaknessParams[2 + k];
        const mu_p = weaknessParams[2 + K + k];
        const var_c = weaknessSe[2 + k] * weaknessSe[2 + k];
        const var_p = weaknessSe[2 + K + k] * weaknessSe[2 + K + k];
        const cov = weaknessAxisCov[k] || 0;
        if (![mu_c, mu_p, var_c, var_p, cov].every(Number.isFinite)) continue;
        if (!(var_c > 0) || !(var_p > 0)) continue;
        const probs = quadrantProbs(mu_c, mu_p, var_c, var_p, cov);
        for (const q of QUADRANTS) {
            if (q.offDiagonal && cov < 0) continue;
            const p = probs[q.key];
            if (p >= PROB_THRESHOLD) {
                issues.push({ k, quadrant: q.key, p, sentence: q.phrase(labelFor(k)) });
            }
        }
    }
    issues.sort((a, b) => b.p - a.p);

    if (issues.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'stats-issues-empty';
        empty.textContent = getTerm('statsIssuesNoneAbove', { percent: (PROB_THRESHOLD * 100).toFixed(0) });
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
    // Threshold per-axis evidence strength to match the issue's posterior
    // confidence — see findExamplesForIssue for the rationale.
    const threshold = probit(issue.p);
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
