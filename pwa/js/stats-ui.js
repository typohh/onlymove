// js/stats-ui.js
//
// The detailed-stats modal: all DOM/SVG rendering for the tabbed stats panel
// (Summary / Weaknesses / Rank / Time / Community) and its charts. Split out of
// model.js, which now holds only the numeric recall model + stats computation.
// The dependency is one-directional: this module consumes the model's exported
// helpers; model.js never reaches back into the UI.

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, getMergedAnswerHistory } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, TOTAL_RANKS } from './constants.js';
import { getTerm, localizedMotif } from './i18n.js';
import { quadrantProbs } from './bivariate-normal.js';
import { computeRankDecomposition, evaluateAt } from './rank-decomposition.js';
import { rankScalarToLabel } from '../shared/rank-math.js';
import { showLoading, hideLoading } from './overlays.js';
import {
    getRecallCandidates,
    getLogitPerRank,
    prepareTrainingData,
    calculateResultsAndStdErrors,
    freshFailureReviewHours,
    MODEL_INDICES,
} from './model.js';

export async function showDetailedStats() {
    // The stats panel merges both board sizes: one rank, one recall model, one
    // weakness model. Build the union once and thread it through every block.
    const merged = getMergedAnswerHistory();
    if (merged.length < 10) {
        alert(getTerm('statsNotEnough').replace('{count}', merged.length));
        return;
    }

    if (!state.modelParams) {
        alert(getTerm('statsFailed'));
        return;
    }

    // Static explanatory text (not just a spinner): the synchronous chart build
    // below freezes any animation, so a plain "constructing…" line is the
    // reliable signal that the click registered and work is underway.
    showLoading(getTerm('statsConstructing'));
    // Yield a paint so the message is actually visible before the synchronous
    // decomposition + chart build (computeRankDecomposition, buildRatingHistory-
    // Chart, …) take over the main thread.
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    try {
        if (!state.statsResults) {
            const validHistory = merged.filter(p => p.problemRank !== undefined && p.timeTaken !== undefined && p.date !== undefined);
            const history = validHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
            const trainingData = prepareTrainingData(history, validHistory);
            const finalResults = calculateResultsAndStdErrors(state.modelParams, trainingData);
            setStatsResults(finalResults);
        }

        if (!state.statsResults) {
            alert(getTerm('statsFailed'));
            return;
        }

        const { results, varCovarMatrix } = state.statsResults;
        const finalParams = state.modelParams;
        // The old single "overview" pane is now split across tabs: timing
        // probabilities ride with the Time chart, the rank decomposition rides
        // with the Rank chart, and only the review-queue counts stay on the
        // (now last) summary pane.
        let timingText = '';
        let resultText = '';

        // Timing observations → Time tab.
        for (const result of results) {
            const { coeff, se, varIndex } = result;
            let termKey;
            if (varIndex === MODEL_INDICES.TIME_SOLVING) termKey = 'statsSolvingTimeNatural';
            else if (varIndex === MODEL_INDICES.REVIEW_TIME) termKey = 'statsReviewTimeNatural';

            if (termKey) {
                const z = coeff / se;
                const cdf = 0.5 * (1 + math.erf(z / Math.sqrt(2)));
                const conf = cdf * 100;
                timingText += getTerm(termKey).replace('{percent_change}', conf.toFixed(0)) + '\n';
            }
        }

        // Block 2: Summary Text. The "freshly-failed problem comes back in ~X h"
        // figure is now the retention window-open delay (NEG_LN_080·S·24) for a
        // just-failed (r=0) repeat at the user's typical repeat difficulty.
        if (varCovarMatrix) {
            const repeatRanks = merged.filter(p => p.isRepeat && typeof p.problemRank === 'number').map(p => p.problemRank);
            const repRank = repeatRanks.length ? repeatRanks.reduce((a, b) => a + b, 0) / repeatRanks.length : 0;
            const firsttimeFailureTime = freshFailureReviewHours(finalParams, { problemRank: repRank, size: state.boardSize, userRank: state.userRank });

            if (firsttimeFailureTime !== null) {
                const allRepeats = getRecallCandidates({
                    lowerProbabilityBound: 0.6,
                    upperProbabilityBound: 1.0,
                    requireLastAnswerIncorrect: false,
                    history: merged,
                }).length;
                const firstTimeFailures = getRecallCandidates({
                    lowerProbabilityBound: 0.6,
                    upperProbabilityBound: 1.0,
                    requireLastAnswerIncorrect: true,
                    history: merged,
                }).length;

                resultText += getTerm('statsFailures').replace('{count_recent}', firstTimeFailures).replace('{hours}', Math.round(firsttimeFailureTime)) + '\n';
                resultText += getTerm('statsOldFailures').replace('{count_other}', Math.max(0, allRepeats - firstTimeFailures)) + '\n';
            }
        }

        const novelProblemCount = new Set(merged.filter(p => !p.isRepeat).map(p => p.problemId)).size;
        resultText += getTerm('statsNovelProblems').replace('{count}', novelProblemCount) + '\n';

        const slippedProblemCount = getRecallCandidates({
            lowerProbabilityBound: 0,
            upperProbabilityBound: 0.6,
            requireLastAnswerIncorrect: false,
            history: merged,
        }).length;
        const ignoredProblemCount = merged.filter(p => p.ignored).length;
        resultText += getTerm('statsIgnoredProblems').replace('{count_ignored}', ignoredProblemCount).replace('{count_slipped}', slippedProblemCount) + '\n';

        // Augmented-model posterior on (β_correct[k], β_picked[k]) per axis,
        // turned into sign-quadrant probabilities via bivariate-normal.js and
        // rendered as a DOM block (instead of inlined into resultText) so each
        // issue can carry an interactive `examples` toggle to past mistakes.
        const issuesElement = buildLikelyIssuesElement(merged);

        // Activity pane: the problems-per-day chart (moved here off the Time
        // chart's secondary axis) above the review-queue summary text.
        const activityPane = combineChartWithText(buildProblemsPerDayChart(merged), resultText);

        // Community/server-wide stats (null if the endpoint isn't ready or the
        // fetch fails — the tab is then dropped).
        const serverPane = await buildServerStatsPane(merged);

        // The Rank and Time charts are the heavy part (rank decomposition +
        // kernel smoothing + bootstrap CI), but they're built eagerly here —
        // under the "constructing charts" overlay shown above — so the modal
        // opens fully-ready and every tab switch is instant. The overlay
        // explains the build delay, which is why no in-modal lazy-load / spinner
        // / pre-warm is needed anymore. The timing probabilities and rank
        // decomposition text ride with their charts.
        const rankPane = combineChartWithText(buildRatingHistoryChart(merged), buildRankSummaryText(merged));
        const timePane = combineChartWithText(buildTimeUsageChart(merged), timingText);

        // Weaknesses is the default (first) page; Community is last. Empty panes
        // (no weaknesses, too little data for a chart, server stats unavailable)
        // are dropped by showStatsModal.
        showStatsModal([
            { label: getTerm('statsTabWeaknesses'), content: issuesElement },
            { label: getTerm('statsTabRank'), content: rankPane },
            { label: getTerm('statsTabTime'), content: timePane },
            { label: getTerm('statsTabActivity'), content: activityPane },
            { label: getTerm('statsTabCommunity'), content: serverPane },
        ]);

    } catch (error) {
        console.error('Failed to generate stats text:', error);
        Sentry.captureException(error);
        alert(getTerm('statsFailed'));
    } finally {
        hideLoading();
    }
}

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
function buildLikelyIssuesElement(history = state.answerHistory) {
    const PROB_THRESHOLD = 0.50;
    const weaknessParams = state.weaknessParams;
    const weaknessSe = state.weaknessSe;
    const weaknessAxisCov = state.weaknessAxisCov;
    const labelMap = state.embeddingLabels || {};
    const labelFor = (k) => {
        const entry = labelMap[String(k)];
        return localizedMotif(entry) || getTerm('statsStyleFallback').replace('{k}', k);
    };
    const QUADRANTS = [
        { key: 'tooMuch',   offDiagonal: true,  phrase: (l) => getTerm('statsIssueTooMuch').replace('{motif}', l) },
        { key: 'notEnough', offDiagonal: true,  phrase: (l) => getTerm('statsIssueNotEnough').replace('{motif}', l) },
        { key: 'badAt',     offDiagonal: false, phrase: (l) => getTerm('statsIssueBadAt').replace('{motif}', l) },
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
        empty.textContent = getTerm('statsIssuesNoneAbove').replace('{percent}', (PROB_THRESHOLD * 100).toFixed(0));
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
        more.textContent = getTerm('statsMoreExamples').replace('{count}', examples.length - shown.length);
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

// Picks wrong attempts whose embedding signature aligns with the (k, quadrant)
// issue. Each quadrant requires BOTH axes to land on the side the quadrant's
// joint sign demands — a one-sided gate would include attempts that only
// support one axis, which is misleading for the both-negative `badAt` case
// (and for the off-diagonals it lets same-sign-ordered attempts slip through).
//
// The minimum |ec|, |ep| per attempt scales with the issue's posterior
// probability via `threshold = probit(p)` — so a 90%-confident issue needs
// ~1.28σ-strong evidence per axis, while a 50%-confident issue collapses to
// the pure sign constraint. Without this scaling, a 51%-borderline issue and
// a 95%-certain one would surface the same attempts, even though the latter
// should be backed by stronger per-attempt evidence.
//
// After the gate, per-attempt evidence strength is the sum of the two relevant
// signed σ contributions — monotone in either axis so a 5σ-on-one + threshold-
// on-the-other attempt ranks above one that just clears the threshold on both.
// (The earlier `min` tied any two attempts at the weaker axis regardless of
// the other one, which made the ordering look arbitrary on the panel.)
// Embeddings are unit-variance per component (skew-ICA standardized), so
// z-scale thresholds and sums are directly interpretable as σ above/below
// baseline.
function findExamplesForIssue(answerHistory, k, quadrant, excludedIds, threshold = 0) {
    const t = Number.isFinite(threshold) ? Math.max(0, threshold) : 0;
    const byId = new Map();
    for (const a of answerHistory) {
        if (!a || a.correct) continue;
        if (a.isRepeat) continue;  // mirror weakness.js training filter — repeats are excluded there
        if (a.ignored) continue;
        if (!a.problemId) continue;
        if (excludedIds && excludedIds.has(a.problemId)) continue;
        if (!Array.isArray(a.embedding) || a.embedding.length <= k) continue;
        if (!Array.isArray(a.pickedEmbedding) || a.pickedEmbedding.length <= k) continue;
        const ec = a.embedding[k];
        const ep = a.pickedEmbedding[k];
        if (!Number.isFinite(ec) || !Number.isFinite(ep)) continue;
        let score;
        if (quadrant === 'tooMuch') {
            if (!(ep > t) || !(-ec > t)) continue;
            score = ep + (-ec);
        } else if (quadrant === 'notEnough') {
            if (!(ec > t) || !(-ep > t)) continue;
            score = ec + (-ep);
        } else {  // badAt
            if (!(ec > t) || !(ep > t)) continue;
            score = ec + ep;
        }
        const prev = byId.get(a.problemId);
        if (!prev || score > prev.score) {
            byId.set(a.problemId, { problemId: a.problemId, date: a.date, score, ec, ep, size: a.size });
        }
    }
    return Array.from(byId.values()).sort((a, b) => b.score - a.score);
}

// Inverse standard-normal CDF via bisection over `math.erf`. ~50 iterations
// converges to ≪ 1e-9; cheap once per issue at panel-open time. Returns 0 for
// p ≤ 0.5 since callers only ever pass posterior probabilities in [0.5, 1).
function probit(p) {
    if (!(p > 0.5)) return 0;
    if (p >= 1) return Infinity;
    let lo = 0, hi = 8;
    for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        const cdf = 0.5 * (1 + math.erf(mid / Math.SQRT2));
        if (cdf < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}

// Community pane: server-wide stats over the preceding 7 days (computed by the
// Worker cron, served from /server-stats). Returns null on any failure or when
// there's no activity, so showStatsModal drops the tab. Labels are localized
// via getTerm (stats* keys in i18n.js).
// Locate the current user within each Community distribution, reproducing the
// Worker's per-user bucketing (computeServerStats in src/index.js) on the
// client's own last-7-day merged history. Returns a bin index per bar-list
// category (-1 when not determinable) plus an answer-derived rank label.
// `timeBins` is the coarse 10s-bucket → display-bin table from the caller.
function computeUserCommunityBuckets(history, timeBins) {
    const out = { attemptsPerDay: -1, solve: -1, review: -1, recall: -1, rankLabel: null };
    if (!Array.isArray(history) || !history.length) return out;

    const cutoff = Date.now() - 7 * 24 * 3600e3;
    const recent = history.filter(a => typeof a.date === 'number' && a.date >= cutoff);
    if (!recent.length) return out;

    const median = (arr) => {
        if (!arr.length) return null;
        const s = arr.slice().sort((a, b) => a - b);
        const m = s.length >> 1;
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };
    const timeBucket = (ms) => Math.min(Math.floor(ms / 10000), 60);   // matches server
    const coarseTimeBin = (bucket) => timeBins.findIndex(b => bucket >= b.lo && bucket < b.hi);

    // Attempts per active (UTC) day — every attempt counts, repeats included.
    const PER_USER_BINS = [[1, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, Infinity]];
    const days = new Set(recent.map(a => new Date(a.date).toISOString().slice(0, 10)));
    if (days.size) {
        const rate = recent.length / days.size;
        out.attemptsPerDay = PER_USER_BINS.findIndex(([lo, hi]) => rate >= lo && rate < hi);
    }

    // Solving time: median of per-attempt solve times (ms), then coarse-binned.
    const solveMed = median(recent.map(a => a.timeTaken).filter(t => typeof t === 'number' && t > 0));
    if (solveMed != null) out.solve = coarseTimeBin(timeBucket(solveMed));

    // Review time: per distinct problem the cumulative review of its LAST
    // attempt (the running total, as the server keeps), median across problems.
    const lastByProblem = new Map();
    for (const a of recent) {
        if (!a.problemId) continue;
        const prev = lastByProblem.get(a.problemId);
        if (!prev || a.date > prev.date) lastByProblem.set(a.problemId, a);
    }
    const reviewMed = median([...lastByProblem.values()]
        .map(a => (a.accumulatedReview || 0) + (a.reviewTime || 0))
        .filter(t => t > 0));
    if (reviewMed != null) out.review = coarseTimeBin(timeBucket(reviewMed));

    // Recall time: shortest failure→next-attempt gap that is ≥12h, per problem.
    const RECALL_MIN_MS = 12 * 3600e3;
    const RECALL_HI = [24, 48, 72, 120, Infinity].map(h => h * 3600e3);
    const seqByProblem = new Map();
    for (const a of recent) {
        if (!a.problemId) continue;
        let seq = seqByProblem.get(a.problemId);
        if (!seq) { seq = []; seqByProblem.set(a.problemId, seq); }
        seq.push({ t: a.date, success: !!a.correct });
    }
    let recallMin = Infinity;
    for (const seq of seqByProblem.values()) {
        if (seq.length < 2) continue;
        seq.sort((x, y) => x.t - y.t);
        for (let i = 1; i < seq.length; i++) {
            if (seq[i - 1].success) continue;            // recall after a success — skip
            const gap = seq[i].t - seq[i - 1].t;
            if (gap >= RECALL_MIN_MS && gap < recallMin) recallMin = gap;
        }
    }
    if (recallMin < Infinity) out.recall = RECALL_HI.findIndex(hi => recallMin < hi);

    // Answer-derived rank: mean internal-rank index of distinct first-time
    // problems → EGF scalar → label (the server's recipe, minus shard rounding;
    // problemRank is already the per-problem M index normalized to 0..1).
    const MIN_RANK_PROBLEMS = 10;
    const RANK_INDEX_OFFSET = 19;
    const firstByProblem = new Map();
    for (const a of recent) {
        if (a.isRepeat || typeof a.problemRank !== 'number') continue;
        if (!firstByProblem.has(a.problemId)) firstByProblem.set(a.problemId, a.problemRank);
    }
    if (firstByProblem.size >= MIN_RANK_PROBLEMS) {
        let sum = 0;
        for (const r of firstByProblem.values()) sum += r * TOTAL_RANKS;
        const scalar = Math.max(-29, Math.min(9, Math.round(sum / firstByProblem.size) - RANK_INDEX_OFFSET));
        out.rankLabel = rankScalarToLabel(scalar);
    }

    return out;
}

async function buildServerStatsPane(history = []) {
    let data;
    try {
        // Version the URL so a deploy bypasses any stale edge-cached response.
        const res = await fetch(`/server-stats?v=${APP_VERSION}`);
        if (!res.ok) return null;
        data = await res.json();
    } catch {
        return null;
    }
    if (!data || !data.totals || !(data.totals.attempts > 0)) return null;

    // Coarse display bins over the server's 10s-bucket histogram (bucket b spans
    // [b·10, (b+1)·10)s; bucket 60 is the ≥600s cap). Returns [{label, value}].
    const TIME_BINS = [
        { lo: 0, hi: 1, label: '0–10s' },
        { lo: 1, hi: 2, label: '10–20s' },
        { lo: 2, hi: 3, label: '20–30s' },
        { lo: 3, hi: 6, label: '30–60s' },
        { lo: 6, hi: 12, label: '1–2m' },
        { lo: 12, hi: 30, label: '2–5m' },
        { lo: 30, hi: 60, label: '5–10m' },
        { lo: 60, hi: Infinity, label: '10m+' },
    ];
    const binHist = (hist) => {
        const rows = TIME_BINS.map(b => ({ label: b.label, value: 0 }));
        for (const r of (hist || [])) {
            const i = TIME_BINS.findIndex(b => r.bucket >= b.lo && r.bucket < b.hi);
            if (i >= 0) rows[i].value += r.n;
        }
        return rows;
    };
    const ago = (iso) => {
        const ms = Date.now() - Date.parse(iso);
        if (!Number.isFinite(ms) || ms < 0) return getTerm('statsAgoJustNow');
        const h = ms / 3600000;
        if (h < 1) return getTerm('statsAgoMinutes').replace('{n}', Math.round(ms / 60000));
        if (h < 24) return getTerm('statsAgoHours').replace('{n}', Math.round(h));
        return getTerm('statsAgoDays').replace('{n}', Math.round(h / 24));
    };
    // `highlightIndex` (optional) marks the bucket the current user falls into
    // by recolouring that row. The legend explaining the colour is shown once
    // for the whole pane (see below), not per chart.
    const makeBarList = (rows, highlightIndex = -1) => {
        const max = Math.max(1, ...rows.map(r => r.value));
        const wrap = document.createElement('div');
        wrap.className = 'server-bars';
        rows.forEach((r, i) => {
            const row = document.createElement('div');
            row.className = 'server-bar-row';
            if (i === highlightIndex) row.classList.add('server-bar-row--you');
            const label = document.createElement('span');
            label.className = 'server-bar-label';
            label.textContent = r.label;
            const track = document.createElement('span');
            track.className = 'server-bar-track';
            const fill = document.createElement('span');
            fill.className = 'server-bar-fill';
            fill.style.width = `${(r.value / max * 100).toFixed(1)}%`;
            track.append(fill);
            const val = document.createElement('span');
            val.className = 'server-bar-value';
            val.textContent = String(r.value);
            row.append(label, track, val);
            wrap.append(row);
        });
        return wrap;
    };

    // Where the current user lands in each distribution, computed from their own
    // last-7-day history with the SAME bucket boundaries the Worker uses
    // server-side (see computeServerStats in src/index.js), so the highlighted
    // bar is consistent with the community bars around it.
    const you = computeUserCommunityBuckets(history, TIME_BINS);

    const pane = document.createElement('div');
    pane.className = 'server-stats';

    const head = document.createElement('div');
    head.className = 'stats-chart-title';
    head.textContent = getTerm('statsCommunityHeader')
        .replace('{days}', data.windowDays)
        .replace('{ago}', ago(data.computedAt));
    pane.append(head);

    const attempts = data.totals.attempts || 0;
    const wins = data.totals.wins || 0;
    const rate = attempts ? Math.round(wins / attempts * 100) : 0;
    const dayCount = (data.daily && data.daily.length) ? data.daily.length : 0;
    const attemptsPerDay = dayCount ? Math.round(attempts / dayCount) : 0;
    let txt = getTerm('statsActiveUsers').replace('{count}', data.totals.users) + '\n';
    txt += getTerm('statsAttemptsPerDay').replace('{count}', attemptsPerDay) + '\n';
    txt += getTerm('statsAttemptsSolved').replace('{count}', attempts).replace('{rate}', rate) + '\n';
    for (const s of (data.bySize || [])) {
        txt += getTerm('statsBySizeLine')
            .replace(/\{size\}/g, s.size)
            .replace('{attempts}', s.attempts)
            .replace('{users}', s.users) + '\n';
    }
    const summary = document.createElement('pre');
    summary.className = 'stats-modal-pre';
    summary.textContent = txt.trimEnd();
    pane.append(summary);

    // Distribution of users by their attempts-per-active-day (binned server-side):
    // how many users average ~5/day, ~10/day, etc. x = attempts/day bin, y = users.
    if (data.attemptsPerUserDist && data.attemptsPerUserDist.length) {
        const h = document.createElement('div');
        h.className = 'server-stats-heading';
        h.textContent = getTerm('statsUsersByAttemptsPerDay');
        pane.append(h, makeBarList(data.attemptsPerUserDist.map(b => ({ label: b.label, value: b.value })), you.attemptsPerDay));
    }

    if (data.solveHist && data.solveHist.length) {
        const h = document.createElement('div');
        h.className = 'server-stats-heading';
        h.textContent = getTerm('statsSolvingTimeMedian');
        pane.append(h, makeBarList(binHist(data.solveHist), you.solve));
    }

    if (data.reviewHist && data.reviewHist.length) {
        const h = document.createElement('div');
        h.className = 'server-stats-heading';
        h.textContent = getTerm('statsReviewTimeMedian');
        pane.append(h, makeBarList(binHist(data.reviewHist), you.review));
    }

    // Recall time: per user, the shortest failure→next-attempt gap (pre-binned
    // server-side). x = recall-time bin, y = users.
    if (data.recallDist && data.recallDist.some(b => b.value)) {
        const h = document.createElement('div');
        h.className = 'server-stats-heading';
        h.textContent = getTerm('statsRecallTime');
        pane.append(h, makeBarList(data.recallDist.map(b => ({ label: b.label, value: b.value })), you.recall));
    }

    // Ranks already arrive ordered weakest→strongest on the EGF scale, one entry
    // per active user (answer-derived from the last 7 days), so render as-is.
    if (data.ranks && data.ranks.length) {
        const h = document.createElement('div');
        h.className = 'server-stats-heading';
        h.textContent = getTerm('statsRanks');
        const rankIdx = you.rankLabel != null
            ? data.ranks.findIndex(r => r.rank === you.rankLabel)
            : -1;
        pane.append(h, makeBarList(data.ranks.map(r => ({ label: r.rank, value: r.n })), rankIdx));
    }

    // One legend for the whole pane explaining the highlighted bar, at the very
    // bottom, shown only when at least one category has a "you" bucket.
    const hasHighlight = you.attemptsPerDay >= 0 || you.solve >= 0
        || you.review >= 0 || you.recall >= 0 || you.rankLabel != null;
    if (hasHighlight) {
        const legend = document.createElement('div');
        legend.className = 'server-bar-legend';
        const swatch = document.createElement('span');
        swatch.className = 'server-bar-legend-swatch';
        const text = document.createElement('span');
        text.textContent = getTerm('statsCommunityYouLegend');
        legend.append(swatch, text);
        pane.append(legend);
    }

    return pane;
}

// The rank decomposition text (long-term skill + drift + session swing) and
// the relative 13x13 readout that used to sit on the summary pane. Built as
// part of the Rank tab pane (computeRankDecomposition is heavy, but the whole
// modal is constructed up front under the "constructing charts" overlay). Reads
// state.statsResults (already computed by the time this runs).
function buildRankSummaryText(history) {
    let rankText = '';

    // Two-component rank decomposition (R = skill S + form M). Gives a
    // long-term-trend number and a session-to-session swing number that can't
    // be read off the single-line smoother in the chart. See
    // rank-decomposition.js for the model and the backfitting approach.
    const decomp = computeRankDecomposition(history);
    if (decomp) {
        const { summary } = decomp;
        if (summary.currentSkill !== null) {
            const lower = Math.floor(summary.currentSkill);
            const pct = Math.round((summary.currentSkill - lower) * 100);
            const curLabel = rankLabelFromIndex(lower);
            const nextLabel = rankLabelFromIndex(lower + 1);
            const seStr = Number.isFinite(summary.currentSkillSE)
                ? getTerm('statsRankSESuffix').replace('{se}', summary.currentSkillSE.toFixed(1))
                : '';
            rankText += getTerm('statsLatestLongTermRank')
                .replace('{cur}', curLabel)
                .replace('{pct}', pct)
                .replace('{next}', nextLabel)
                .replace('{se}', seStr) + '\n';
        }
        if (summary.longTermDrift !== null && summary.longTermSpanMs > 30 * 24 * 3600 * 1000) {
            const months = summary.longTermSpanMs / (30 * 24 * 3600 * 1000);
            const sign = summary.longTermDrift >= 0 ? '+' : '−';
            rankText += getTerm('statsSkillDrift')
                .replace('{sign}', sign)
                .replace('{amount}', Math.abs(summary.longTermDrift).toFixed(1))
                .replace('{months}', months.toFixed(0)) + '\n';
        }
        if (typeof summary.sigmaM === 'number') {
            rankText += summary.sigmaM >= 0.1
                ? getTerm('statsSessionSwing')
                    .replace('{up}', summary.sigmaUp.toFixed(2))
                    .replace('{down}', summary.sigmaDown.toFixed(2)) + '\n'
                : getTerm('statsSessionNoSwing')
                    .replace('{upper}', summary.sigmaMUpper.toFixed(2)) + '\n';
        }
    }

    // Relative 13x13 skill. The recall model's SIZE_IS_13 coefficient is the
    // logit offset for 13x13 at equal internal-rank difficulty (9x9 is the
    // baseline), so coeff / logit-per-rank is the skill gap in ranks. Only
    // shown once both sizes have enough non-repeat data to identify it.
    const results = state.statsResults?.results || [];
    const MIN_SIZE_COMPARISON = 20;
    const count9 = history.filter(p => !p.isRepeat && p.size === 9 && p.problemRank !== undefined).length;
    const count13 = history.filter(p => !p.isRepeat && p.size === 13 && p.problemRank !== undefined).length;
    const sizeResult = results.find(r => r.varIndex === MODEL_INDICES.SIZE_IS_13);
    if (sizeResult && count9 >= MIN_SIZE_COMPARISON && count13 >= MIN_SIZE_COMPARISON
        && Number.isFinite(sizeResult.coeff) && Number.isFinite(sizeResult.se) && sizeResult.se > 0) {
        const logitPerRank = getLogitPerRank(state.statsResults);
        const ranksDiff = sizeResult.coeff / logitPerRank;   // + = stronger on 13x13
        const seRanks = sizeResult.se / logitPerRank;
        rankText += getTerm(ranksDiff >= 0 ? 'statsSizeStronger' : 'statsSizeWeaker')
            .replace('{amount}', Math.abs(ranksDiff).toFixed(1))
            .replace('{se}', seRanks.toFixed(1)) + '\n';
    }

    return rankText;
}

// A tab pane that pairs a chart with the explanatory text that used to live on
// the summary page (timing probabilities for Time, the rank decomposition for
// Rank). Either piece may be missing: returns null only when both are, so an
// empty tab is dropped by showStatsModal.
function combineChartWithText(chartElement, text) {
    const hasText = typeof text === 'string' && text.trim().length > 0;
    if (!chartElement && !hasText) return null;

    const wrap = document.createElement('div');
    wrap.className = 'stats-chart-pane';
    if (chartElement) wrap.append(chartElement);
    if (hasText) {
        const pre = document.createElement('pre');
        pre.className = 'stats-modal-pre';
        pre.textContent = text;
        wrap.append(pre);
    }
    return wrap;
}

// Tabbed stats modal. `tabs` is [{ label, content: node }]; every pane is built
// before this is called (under the "constructing charts" overlay), so tab
// switching is pure show/hide and instant. Tabs with null content are dropped,
// and the tab bar is omitted when only one pane remains.
function showStatsModal(tabs) {
    const entries = tabs
        .filter(t => t.content)
        .map(t => ({ label: t.label, node: t.content }));
    if (entries.length === 0) return;

    const backdrop = document.createElement('div');
    backdrop.className = 'stats-modal-backdrop';

    const card = document.createElement('div');
    card.className = 'stats-modal-card';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'stats-modal-close';
    closeBtn.textContent = getTerm('statsClose');

    const close = () => {
        document.removeEventListener('keydown', onKey);
        backdrop.remove();
    };
    const onKey = (e) => { if (e.key === 'Escape') close(); };
    closeBtn.onclick = close;
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.addEventListener('keydown', onKey);

    const tabbar = document.createElement('div');
    tabbar.className = 'stats-tabbar';
    const body = document.createElement('div');
    body.className = 'stats-modal-body';

    let current = entries[0];

    const renderBody = () => {
        body.replaceChildren(current.node);
        body.scrollTop = 0;
    };
    const renderTabbar = () => {
        // The bar is omitted when only one pane remains.
        tabbar.style.display = entries.length > 1 ? '' : 'none';
        tabbar.replaceChildren();
        for (const e of entries) {
            const b = document.createElement('button');
            b.className = 'stats-tab';
            b.classList.toggle('active', e === current);
            b.textContent = e.label;
            b.onclick = () => { current = e; renderTabbar(); renderBody(); };
            tabbar.append(b);
        }
    };

    card.append(tabbar, body, closeBtn);
    backdrop.appendChild(card);
    document.body.appendChild(backdrop);

    renderTabbar();
    renderBody();
}

function rankLabelFromIndex(rankIndex) {
    return rankIndex < 20 ? `${20 - rankIndex}k` : `${rankIndex - 19}d`;
}

// Pick a viewBox height (for a fixed width W) so the chart's aspect ratio
// approximates the modal's visible area. The SVG renders at width:100% /
// height:auto with uniform scaling, so a W×H viewBox paints at
// renderedW × renderedW·(H/W); matching H/W to (visible height)/(width) makes
// the whole chart block (title + plot + legend) fit inside the scrollable
// 70vh body without being taller than the viewport — visible whole on scroll.
function chartViewBoxHeight(W) {
    const vw = window.innerWidth || 1024;
    const vh = window.innerHeight || 768;
    const bodyVisibleH = 0.70 * vh;            // .stats-modal-body max-height: 70vh
    const chrome = 120;                        // title + legend + x-axis + margins (px)
    const plotH = Math.max(150, bodyVisibleH - chrome);
    const cardPad = 0.06 * Math.min(vw, vh);   // .stats-modal-card padding: 3vmin each side
    const yAxisCol = 42;                       // y-tick label column
    const plotW = Math.max(220, 0.9 * vw - cardPad - yAxisCol);
    // Clamp to a sane band: never taller than square, never an extreme letterbox.
    const ratio = Math.max(0.42, Math.min(plotH / plotW, 1.0));
    return Math.round(W * ratio);
}

function buildRatingHistoryChart(history = state.answerHistory) {
    const entries = history
        .filter(a => typeof a.problemRank === 'number' && typeof a.date === 'number'
                     && !a.isRepeat && !a.linked)
        .map(a => ({ date: a.date, rankValue: a.problemRank * TOTAL_RANKS, correct: !!a.correct }));
    if (entries.length < 2) return null;

    // Two-kernel rank decomposition: see rank-decomposition.js. We evaluate the
    // skill smoother Ŝ(t), its measurement SE seS(t), and the time-varying
    // performance swing σ_M(t) continuously on a regular t-grid (the kernels are
    // exact functions of t — no bucketing, no interpolation between sparse
    // attempt timestamps), so line and bands shape correctly across gaps.
    const t0 = entries[0].date;
    const t1 = entries[entries.length - 1].date;
    const tSpan = Math.max(1, t1 - t0);
    const decomp = computeRankDecomposition(history);
    const GRID_POINTS = 300;
    const Z = 1.96;
    // Half-widths in rank space: inner band = measurement uncertainty (how well
    // we know your skill), symmetric. Outer band = a prediction interval (where a
    // given session's performance lands); the form swing is directional, so the
    // outer band is ASYMMETRIC — good-day reach above, bad-day dip below — each
    // added to the measurement SE in quadrature.
    const innerHalf = e => Z * e.seS;
    const outerUp = e => Z * Math.hypot(e.seS, e.sigmaUp);
    const outerDown = e => Z * Math.hypot(e.seS, e.sigmaDown);
    let gridEvals = [];
    if (decomp) {
        for (let i = 0; i < GRID_POINTS; i++) {
            const t = Math.round(t0 + tSpan * (i / (GRID_POINTS - 1)));
            const e = evaluateAt(decomp, t);
            if (e) gridEvals.push({ t, S: e.S, seS: e.seS, sigmaUp: e.sigmaUp, sigmaDown: e.sigmaDown });
        }
    }

    // Focus the y-range on the user's rank — the skill curve and its prediction
    // band — not on the spread of problem difficulties. Problem ranks scatter
    // far wider than the user's skill, so anchoring on them squashes the part of
    // the chart that matters. Dots outside this window are dropped (see scatter).
    let bandMin = Infinity, bandMax = -Infinity;
    for (const e of gridEvals) {
        bandMin = Math.min(bandMin, e.S - outerDown(e));
        bandMax = Math.max(bandMax, e.S + outerUp(e));
    }
    if (!Number.isFinite(bandMin)) {
        // No decomposition (too little data) — fall back to the scatter range.
        const sortedRanks = entries.map(e => e.rankValue).slice().sort((a, b) => a - b);
        bandMin = sortedRanks[0];
        bandMax = sortedRanks[sortedRanks.length - 1];
    }
    // Window exactly one rank beyond the outer CI band — no rounding out to
    // integers (that slop nearly doubled the padding). Grid lines below still
    // land on the integer ranks that fall inside this range.
    const Y_PAD = 1; // exactly one rank beyond the outer CI band
    let yHigh = bandMax + Y_PAD;
    let yLow = bandMin - Y_PAD;
    if (yHigh <= yLow) yHigh = yLow + 1;

    const container = document.createElement('div');
    container.className = 'stats-chart';

    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    title.textContent = getTerm('statsChartRankSince').replace('{date}', new Date(t0).toISOString().slice(0, 10));
    container.append(title);

    const plotRow = document.createElement('div');
    plotRow.className = 'stats-chart-plot-row';

    // Grid lines mark the rank CUTOFFS (integer rank boundaries); the label for
    // a rank sits BETWEEN its two bounding lines, so it reads as "this band is
    // rank X" — a range — rather than a value pinned to one line. Labels that
    // would visually overlap their neighbor get hidden after layout (see
    // ResizeObserver below).
    const gridRanks = [];
    for (let r = Math.ceil(yLow); r <= Math.floor(yHigh); r++) gridRanks.push(r);
    // One label per rank band, centered at r+0.5 (between cutoff lines r and
    // r+1). Values in [r, r+1) carry rank rankLabelFromIndex(r). Bands clipped
    // at a chart edge keep the same midpoint and just get clipped off.
    const labelTicks = [];
    for (let r = Math.floor(yLow); r <= Math.ceil(yHigh); r++) {
        const mid = r + 0.5;
        if (mid < yLow || mid > yHigh) continue;
        labelTicks.push({ pos: mid, text: rankLabelFromIndex(r) });
    }

    // Y-axis labels live in HTML so they inherit the surrounding pre font
    // (13px monospace) exactly. The hidden spacer sizes the column to the
    // widest tick label.
    const yaxis = document.createElement('div');
    yaxis.className = 'stats-chart-yaxis';
    const spacer = document.createElement('span');
    spacer.className = 'y-spacer';
    spacer.textContent = '20k';
    yaxis.append(spacer);
    const topPct = v => (1 - (v - yLow) / (yHigh - yLow)) * 100;
    for (const t of labelTicks) {
        const tick = document.createElement('div');
        tick.className = 'y-tick';
        tick.style.top = `${topPct(t.pos)}%`;
        tick.textContent = t.text;
        yaxis.append(tick);
    }

    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-chart-svg-wrap';

    // SVG holds only the grid + dots. Aspect ratio tracks the visible modal
    // area (see chartViewBoxHeight) so the chart fits whole when scrolled to.
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - t0) / tSpan) * W;
    const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;

    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');

    const gridStroke = 'rgba(128,128,128,0.35)';
    for (const r of gridRanks) {
        const y = yOf(r);
        const line = document.createElementNS(SVG_NS, 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('x2', W);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', gridStroke);
        line.setAttribute('stroke-width', '0.5');
        line.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(line);
    }

    // Per-attempt scatter: green=correct, red=wrong. Drawn UNDER the bands and
    // skill line so the curve and its CIs always read on top — at high volume
    // (count is typically ~200, up to ~10k) the dots stay faint density texture
    // instead of an opaque slab. The narrow y-window clips most dots — problem
    // ranks scatter far wider than the band — so key the overplotting guards to
    // the count of dots that actually fall INSIDE the window, not the raw total.
    // (Keying off the total drove opacity to the floor while only a handful of
    // dots were visible, so they vanished.) Two guards: cap the rendered dots
    // (uniform stride), and scale per-dot opacity down as the visible count
    // grows. The 0.15 floor matters: dots sit beneath the translucent bands, so
    // anything fainter is swallowed by the band tint near the curve.
    const visibleDots = entries.filter(e => e.rankValue >= yLow && e.rankValue <= yHigh);
    const SCATTER_CAP = 1500;
    const stride = Math.max(1, Math.ceil(visibleDots.length / SCATTER_CAP));
    const shownDots = stride > 1 ? visibleDots.filter((_, i) => i % stride === 0) : visibleDots;
    const dotR = shownDots.length > 600 ? '1.4' : '1.6';
    const dotAlpha = Math.max(0.15, Math.min(0.6, 50 / Math.max(1, shownDots.length))).toFixed(3);
    const DOT_CORRECT = '#50b450', DOT_WRONG = '#dc5050';
    for (const e of shownDots) {
        const dot = document.createElementNS(SVG_NS, 'circle');
        dot.setAttribute('cx', xOf(e.date).toFixed(1));
        dot.setAttribute('cy', yOf(e.rankValue).toFixed(1));
        dot.setAttribute('r', dotR);
        dot.setAttribute('fill', e.correct ? DOT_CORRECT : DOT_WRONG);
        dot.setAttribute('fill-opacity', dotAlpha);
        svg.appendChild(dot);
    }

    // Two nested bands, drawn after the scatter so the curve + CIs sit on top.
    // Outer (lighter) = prediction interval: skill ± where a session's form
    // actually lands. Inner (darker) = measurement uncertainty on skill. The
    // gap between them is the genuine session-to-session swing σ_M(t).
    const SKILL_FILL_OUTER = 'rgba(59,130,246,0.10)';
    const SKILL_FILL_INNER = 'rgba(59,130,246,0.22)';
    const SKILL_STROKE = '#2563eb';

    // Separate upper/lower half-widths so the outer band can be asymmetric. The
    // top edge runs left→right at S + upHalf, the bottom edge runs right→left at
    // S − downHalf, closing the ring. The inner band passes innerHalf for both.
    const bandPath = (upHalf, downHalf) => {
        let d = '';
        for (let i = 0; i < gridEvals.length; i++) {
            const e = gridEvals[i];
            d += (i === 0 ? 'M' : 'L') + `${xOf(e.t).toFixed(1)} ${yOf(e.S + upHalf(e)).toFixed(1)} `;
        }
        for (let i = gridEvals.length - 1; i >= 0; i--) {
            const e = gridEvals[i];
            d += `L${xOf(e.t).toFixed(1)} ${yOf(e.S - downHalf(e)).toFixed(1)} `;
        }
        return d + 'Z';
    };

    if (gridEvals.length >= 2) {
        for (const [uh, dh, fill] of [[outerUp, outerDown, SKILL_FILL_OUTER], [innerHalf, innerHalf, SKILL_FILL_INNER]]) {
            const band = document.createElementNS(SVG_NS, 'path');
            band.setAttribute('d', bandPath(uh, dh));
            band.setAttribute('fill', fill);
            band.setAttribute('stroke', 'none');
            svg.appendChild(band);
        }
    }

    if (gridEvals.length >= 2) {
        const skillPts = gridEvals
            .map(e => `${xOf(e.t).toFixed(1)},${yOf(e.S).toFixed(1)}`)
            .join(' ');
        const skillLine = document.createElementNS(SVG_NS, 'polyline');
        skillLine.setAttribute('points', skillPts);
        skillLine.setAttribute('fill', 'none');
        skillLine.setAttribute('stroke', SKILL_STROKE);
        skillLine.setAttribute('stroke-width', '1.5');
        skillLine.setAttribute('stroke-linejoin', 'round');
        skillLine.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(skillLine);
    }

    svgWrap.append(svg);

    // y-axis + SVG share one row whose height is exactly the SVG height, so the
    // y-tick percentages (positioned against the y-axis column) line up with the
    // grid lines. The x-axis date sits BELOW this row, not inside the SVG column
    // — folding it in there stretched the y-axis column past the SVG and pushed
    // every label down.
    const plotMain = document.createElement('div');
    plotMain.className = 'stats-chart-plot-main';
    plotMain.append(yaxis, svgWrap);

    const xaxis = document.createElement('div');
    xaxis.className = 'stats-chart-xaxis';
    xaxis.textContent = new Date(t1).toISOString().slice(0, 10);

    plotRow.append(plotMain, xaxis);
    container.append(plotRow);

    // Legend below the plot. One entry per visual element; swatch shapes match
    // how each is drawn (dot / line / filled band). Only list the skill curve
    // and its bands when the decomposition actually produced them.
    const legend = document.createElement('div');
    legend.className = 'stats-chart-legend';
    const legendItems = [
        { kind: 'dot', color: DOT_CORRECT, text: getTerm('statsLegendCorrect') },
        { kind: 'dot', color: DOT_WRONG, text: getTerm('statsLegendWrong') },
    ];
    if (gridEvals.length >= 2) {
        legendItems.push(
            { kind: 'line', color: SKILL_STROKE, text: getTerm('statsLegendRank') },
            { kind: 'band', color: SKILL_FILL_INNER, text: getTerm('statsLegend95Confidence') },
            { kind: 'band', color: SKILL_FILL_OUTER, text: getTerm('statsLegendDailyVariance') },
        );
    }
    for (const item of legendItems) {
        const el = document.createElement('span');
        el.className = `legend-item legend-${item.kind}`;
        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = item.color;
        el.append(swatch, document.createTextNode(item.text));
        legend.append(el);
    }
    container.append(legend);

    // After the chart hits the DOM and gets a real height, greedy-hide any
    // y-axis labels that would physically overlap a previously-shown one.
    // Re-runs on resize so density adapts to viewport.
    const updateLabelVisibility = () => {
        const h = svg.getBoundingClientRect().height;
        if (h === 0) return;
        const tickEls = [...yaxis.querySelectorAll('.y-tick')]
            .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
        const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
        let lastShownPx = -Infinity;
        for (const el of tickEls) {
            const topPx = (parseFloat(el.style.top) / 100) * h;
            if (topPx - lastShownPx >= minSepPx) {
                el.style.visibility = '';
                lastShownPx = topPx;
            } else {
                el.style.visibility = 'hidden';
            }
        }
    };
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(updateLabelVisibility).observe(svg);
    } else {
        queueMicrotask(updateLabelVisibility);
    }

    return container;
}

// Number of problems attempted per calendar day, as a bar chart. Each attempt
// is binned into its local-midnight day; the bars span first→last active day,
// so idle days simply show no bar. Lives on the Activity tab (it used to ride
// the Time chart's secondary right axis as a smoothed rate). Returns null when
// there's too little data to read as a daily trend.
function buildProblemsPerDayChart(history = state.answerHistory) {
    const DAY_MS = 24 * 3600 * 1000;
    const dayMid = (t) => { const d = new Date(t); d.setHours(0, 0, 0, 0); return d.getTime(); };

    const counts = new Map(); // day-midnight ms → attempt count
    for (const a of history) {
        if (typeof a.date !== 'number') continue;
        const k = dayMid(a.date);
        counts.set(k, (counts.get(k) || 0) + 1);
    }
    if (counts.size < 2) return null;

    const days = [...counts.keys()].sort((p, q) => p - q);
    const d0 = days[0];
    const d1 = days[days.length - 1];
    const xStart = d0, xEnd = d1 + DAY_MS;
    const total = xEnd - xStart;
    const maxCount = Math.max(...counts.values());

    // Integer y-ticks: 5 evenly-spaced labels at 0..(step·4).
    const step = Math.max(1, Math.ceil(maxCount / 4));
    const yHigh = step * 4;

    const container = document.createElement('div');
    container.className = 'stats-chart';

    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    title.textContent = getTerm('statsChartProblemsPerDaySince').replace('{date}', new Date(d0).toISOString().slice(0, 10));
    container.append(title);

    const plotRow = document.createElement('div');
    plotRow.className = 'stats-chart-plot-row';

    const yaxis = document.createElement('div');
    yaxis.className = 'stats-chart-yaxis';
    const spacer = document.createElement('span');
    spacer.className = 'y-spacer';
    spacer.textContent = String(yHigh);
    yaxis.append(spacer);
    for (let i = 0; i <= 4; i++) {
        const v = step * i;
        const tick = document.createElement('div');
        tick.className = 'y-tick';
        tick.style.top = `${(1 - v / yHigh) * 100}%`;
        tick.textContent = String(v);
        yaxis.append(tick);
    }

    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-chart-svg-wrap';
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - xStart) / total) * W;
    const yOf = v => (1 - v / yHigh) * H;

    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');

    // Horizontal grid at each y-tick.
    for (let i = 0; i <= 4; i++) {
        const y = yOf(step * i);
        const line = document.createElementNS(SVG_NS, 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('x2', W);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(128,128,128,0.35)');
        line.setAttribute('stroke-width', '0.5');
        line.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(line);
    }

    // One bar per active day. Leave a thin gap between adjacent days unless the
    // span is so long that bars are already sub-pixel.
    const barW = (DAY_MS / total) * W;
    const gap = barW > 2 ? Math.min(0.6, barW * 0.15) : 0;
    for (const [k, c] of counts) {
        const x = xOf(k) + gap / 2;
        const w = Math.max(0.4, barW - gap);
        const y = yOf(c);
        const rect = document.createElementNS(SVG_NS, 'rect');
        rect.setAttribute('x', x.toFixed(2));
        rect.setAttribute('y', y.toFixed(2));
        rect.setAttribute('width', w.toFixed(2));
        rect.setAttribute('height', (H - y).toFixed(2));
        rect.setAttribute('fill', '#6f8bbf');
        svg.appendChild(rect);
    }

    svgWrap.append(svg);

    const plotMain = document.createElement('div');
    plotMain.className = 'stats-chart-plot-main';
    plotMain.append(yaxis, svgWrap);

    const xaxis = document.createElement('div');
    xaxis.className = 'stats-chart-xaxis';
    xaxis.textContent = new Date(d1).toISOString().slice(0, 10);

    plotRow.append(plotMain, xaxis);
    container.append(plotRow);

    // Collapse overlapping y-tick labels, mirroring the Time chart.
    const updateLabelVisibility = () => {
        const h = svg.getBoundingClientRect().height;
        if (h === 0) return;
        const tickEls = [...yaxis.querySelectorAll('.y-tick')]
            .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
        const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
        let lastShownPx = -Infinity;
        for (const el of tickEls) {
            const topPx = (parseFloat(el.style.top) / 100) * h;
            if (topPx - lastShownPx >= minSepPx) {
                el.style.visibility = '';
                lastShownPx = topPx;
            } else {
                el.style.visibility = 'hidden';
            }
        }
    };
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(updateLabelVisibility).observe(svg);
    } else {
        queueMicrotask(updateLabelVisibility);
    }

    return container;
}

// Daily time spent solving and reviewing, smoothed. A Gaussian-kernel smoother
// over the ACTUAL attempt timestamps gives the trend line: a day with no
// attempts contributes nothing and is carried by its neighbours rather than
// pulled to zero, and stretches with no attempts within the kernel's reach
// leave a GAP in the line (we never invent a zero where there is no data). The
// band is the SPREAD of individual attempts, not a confidence interval for the
// line: at each date it's the kernel-weighted 17.5%/82.5% quantiles of the
// nearby attempts' times — i.e. "the middle 65% of attempts land in here" (the
// central range ~1σ covers for symmetric data, but read off the empirical
// skewed distribution, not a normal fit). This
// is a direct weighted empirical quantile (sort the window, walk the cumulative
// kernel weight to the target), so no bootstrap/resampling is involved; the
// heavy idle-time right tail just lands in the upper quantile naturally. The
// y-window clips to a robust quantile of the upper edge so one idle spell can't
// flatten the rest.
function buildTimeUsageChart(history = state.answerHistory) {
    const DAY_MS = 24 * 3600 * 1000;

    const seriesAttempts = (valueOf) => {
        const out = [];
        for (const a of history) {
            if (typeof a.date !== 'number') continue;
            const v = valueOf(a);
            if (v === null) continue;
            out.push({ t: a.date, v });
        }
        out.sort((p, q) => p.t - q.t);
        return out;
    };
    // Solving time is logged on every attempt. Review time accrues only when the
    // user actually reviews, so the review series keeps attempts with positive
    // reviewTime — its distribution is "how long a review takes", not diluted to
    // near-zero by the many attempts that were never reviewed. Values in seconds.
    const solving = seriesAttempts(a => (typeof a.timeTaken === 'number' && a.timeTaken >= 0) ? a.timeTaken / 1000 : null);
    const review = seriesAttempts(a => (typeof a.reviewTime === 'number' && a.reviewTime > 0) ? a.reviewTime / 1000 : null);

    const all = [...solving, ...review];
    if (all.length < 4) return null;
    const t0 = Math.min(...all.map(p => p.t));
    const t1 = Math.max(...all.map(p => p.t));
    const tSpan = t1 - t0;
    if (tSpan < 0.75 * DAY_MS) return null; // too short a span to read as a daily trend

    const quantile = (sorted, p) => {
        if (sorted.length === 1) return sorted[0];
        const idx = p * (sorted.length - 1);
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    };
    const lowerBound = (arr, target) => {
        let lo = 0, hi = arr.length;
        while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < target) lo = m + 1; else hi = m; }
        return lo;
    };
    // Kernel-WEIGHTED empirical quantile over a window. `pairs` is ascending by
    // value; `total` is Σw. Walk the cumulative weight to p·total and linearly
    // interpolate between the two bracketing samples, so the edge moves smoothly
    // as the window slides instead of stepping from one sample value to the next.
    const wQuantile = (pairs, total, p) => {
        const target = p * total;
        let cum = 0;
        for (let i = 0; i < pairs.length; i++) {
            const next = cum + pairs[i].w;
            if (next >= target) {
                if (i === 0) return pairs[0].v;
                return pairs[i - 1].v + (pairs[i].v - pairs[i - 1].v) * ((target - cum) / pairs[i].w);
            }
            cum = next;
        }
        return pairs[pairs.length - 1].v;
    };

    // Kernel-weighted median line + a weighted-quantile spread band on a regular
    // t-grid. Returns an array of {t, mean, lo, hi}, or null at grid points with
    // too little nearby data (those break the line).
    //
    // Bandwidth is a fixed FRACTION of the visible span, so the smoothing scale
    // is relative to the GRAPH, not to an absolute number of days. The x-axis
    // always spans t0..t1 across the same pixel width, so a kernel of bw=tSpan/N
    // covers a constant slice of the width at any span — the curve reads with the
    // same visual smoothness whether it's a month or three years of history. The
    // old code capped bw at 14 days, which for a long-time user was a sliver of a
    // multi-year axis and left the line jagged (the cap, not the data, was the
    // problem). The 3-day floor keeps a short history from over-smoothing to a
    // flat line; there is deliberately no upper cap.
    const G = 140, ESS_MIN = 1.5;
    const bw = Math.max(3 * DAY_MS, tSpan / 36);
    const smooth = (pts) => {
        if (pts.length < 4) return null;
        const ts = pts.map(p => p.t);
        const grid = new Array(G);
        for (let g = 0; g < G; g++) {
            const t = t0 + tSpan * (g / (G - 1));
            const i0 = lowerBound(ts, t - 3.5 * bw);
            const i1 = lowerBound(ts, t + 3.5 * bw);
            // Sum over EVERY sample in the window (no thinning): both the centre
            // line and the band are kernel-weighted quantiles of the same window.
            // Using the full window keeps them smooth functions of t — strided
            // subsampling here was the earlier sawtooth bug.
            let W = 0, sumW2 = 0;
            const pairs = [];
            for (let i = i0; i < i1; i++) {
                const d = (t - ts[i]) / bw;
                const w = Math.exp(-0.5 * d * d);
                W += w; sumW2 += w * w;
                pairs.push({ v: pts[i].v, w });
            }
            const ess = W > 0 ? (W * W) / sumW2 : 0; // effective sample size near t
            if (ess < ESS_MIN) { grid[g] = null; continue; }
            pairs.sort((a, b) => a.v - b.v);
            // Centre line = kernel-weighted MEDIAN (not the mean): time usage is
            // heavily right-skewed (idle-tab outliers), so the mean is dragged up
            // by the tail and isn't a typical day. The median is, and it always
            // sits inside the band. Band = 17.5%/82.5% edges = the middle 65% of
            // attempts (the central range ~1σ covers for symmetric data, read off
            // the EMPIRICAL skewed distribution — no normal assumption).
            grid[g] = {
                t,
                mid: wQuantile(pairs, W, 0.5),
                lo: wQuantile(pairs, W, 0.175),
                hi: wQuantile(pairs, W, 0.825),
            };
        }
        return grid;
    };

    // Break a smoothed grid into runs of consecutive present points so the line
    // and band stop across no-data gaps instead of bridging them.
    const segments = (grid) => {
        const runs = [];
        let run = [];
        for (const e of grid) {
            if (e) run.push(e);
            else { if (run.length) runs.push(run); run = []; }
        }
        if (run.length) runs.push(run);
        return runs;
    };

    const sGrid = smooth(solving);
    const rGrid = smooth(review);
    if (!sGrid && !rGrid) return null;

    // Robust ceiling from the upper envelope so most of the band shows; the most
    // extreme spell clips at the top edge (overflow:hidden). Use max(mean, hi)
    // per grid point, NOT hi alone: with a heavy right tail the weighted mean can
    // sit ABOVE the 82.5% band edge, so a hi-only ceiling would clip the mean
    // line out of the window (the bug the 65% band exposed).
    const hiVals = [];
    for (const grid of [sGrid, rGrid]) if (grid) for (const e of grid) if (e) hiVals.push(Math.max(e.hi, e.mid));
    hiVals.sort((a, b) => a - b);
    const rawHigh = Math.max(quantile(hiVals, 0.90) * 1.05, 1);
    const yLow = 0;
    // Nice y ticks: every tick lands on a whole number of seconds OR a whole
    // number of minutes — never a fractional minute (the old `2.5m` bug). Pick
    // the smallest "nice" step ≥ rawHigh/4 off a seconds→minutes ladder, then
    // snap the ceiling up to a multiple of it so 0, step, 2·step, … are round.
    // Minute steps are multiples of 60, so when we switch to minute labels every
    // tick divides evenly; sub-minute steps stay in seconds.
    const STEP_LADDER = [5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 7200];
    const step = STEP_LADDER.find(s => s >= rawHigh / 4) ?? STEP_LADDER[STEP_LADDER.length - 1];
    const yHigh = Math.ceil(rawHigh / step) * step;
    const useMin = step % 60 === 0; // minute steps ⇒ whole-minute ticks
    const fmt = (s) => useMin ? `${Math.round(s / 60)}m` : `${Math.round(s)}s`;
    const yTickVals = [];
    for (let v = 0; v <= yHigh + 1e-6; v += step) yTickVals.push(v);

    const container = document.createElement('div');
    container.className = 'stats-chart';

    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    title.textContent = getTerm('statsChartTimePerProblemSince').replace('{date}', new Date(t0).toISOString().slice(0, 10));
    container.append(title);

    const plotRow = document.createElement('div');
    plotRow.className = 'stats-chart-plot-row';

    const yaxis = document.createElement('div');
    yaxis.className = 'stats-chart-yaxis';
    const leftLabels = yTickVals.map(fmt);
    const widest = arr => arr.reduce((a, b) => (b.length > a.length ? b : a), '');
    const spacer = document.createElement('span');
    spacer.className = 'y-spacer';
    // Size the column to the WIDEST label, not just the top one — an
    // intermediate tick ("2.5m") can be wider than the ceiling ("10m") and
    // would otherwise overflow the column and clip on the left.
    spacer.textContent = widest(leftLabels);
    yaxis.append(spacer);
    const topPct = v => (1 - (v - yLow) / (yHigh - yLow)) * 100;
    for (let i = 0; i < yTickVals.length; i++) {
        const tick = document.createElement('div');
        tick.className = 'y-tick';
        tick.style.top = `${topPct(yTickVals[i])}%`;
        tick.textContent = leftLabels[i];
        yaxis.append(tick);
    }

    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-chart-svg-wrap';
    const SVG_NS = 'http://www.w3.org/2000/svg';
    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - t0) / tSpan) * W;
    const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;

    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('role', 'img');
    svg.style.overflow = 'hidden'; // clip clipped-day spikes at the top edge

    const gridLine = (x1, y1, x2, y2) => {
        const line = document.createElementNS(SVG_NS, 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('x2', x2);
        line.setAttribute('y1', y1);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', 'rgba(128,128,128,0.35)');
        line.setAttribute('stroke-width', '0.5');
        line.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(line);
    };

    // Horizontal grid at each y-tick.
    for (const v of yTickVals) {
        const y = yOf(v);
        gridLine(0, y, W, y);
    }

    // Evenly-spaced x-axis date ticks, each with a matching vertical grid line.
    const N_XTICKS = 3;
    const xTicks = [];
    for (let i = 0; i < N_XTICKS; i++) xTicks.push(t0 + tSpan * (i / (N_XTICKS - 1)));
    for (const t of xTicks) {
        const x = xOf(t);
        gridLine(x, 0, x, H);
    }

    const SERIES = [
        { grid: sGrid, line: '#2e9e5b', band: 'rgba(46,158,91,0.20)' },
        { grid: rGrid, line: '#d98a2b', band: 'rgba(217,138,43,0.20)' },
    ];

    // Bands first, then median lines on top.
    for (const { grid, band } of SERIES) {
        if (!grid) continue;
        for (const run of segments(grid)) {
            if (run.length < 2) continue;
            let d = '';
            for (let i = 0; i < run.length; i++) {
                d += (i === 0 ? 'M' : 'L') + `${xOf(run[i].t).toFixed(1)} ${yOf(run[i].hi).toFixed(1)} `;
            }
            for (let i = run.length - 1; i >= 0; i--) {
                d += `L${xOf(run[i].t).toFixed(1)} ${yOf(run[i].lo).toFixed(1)} `;
            }
            const path = document.createElementNS(SVG_NS, 'path');
            path.setAttribute('d', d + 'Z');
            path.setAttribute('fill', band);
            path.setAttribute('stroke', 'none');
            svg.appendChild(path);
        }
    }

    for (const { grid, line } of SERIES) {
        if (!grid) continue;
        for (const run of segments(grid)) {
            if (run.length === 1) {
                const dot = document.createElementNS(SVG_NS, 'circle');
                dot.setAttribute('cx', xOf(run[0].t).toFixed(1));
                dot.setAttribute('cy', yOf(run[0].mid).toFixed(1));
                dot.setAttribute('r', '2');
                dot.setAttribute('fill', line);
                svg.appendChild(dot);
                continue;
            }
            const pts = run.map(e => `${xOf(e.t).toFixed(1)},${yOf(e.mid).toFixed(1)}`).join(' ');
            const poly = document.createElementNS(SVG_NS, 'polyline');
            poly.setAttribute('points', pts);
            poly.setAttribute('fill', 'none');
            poly.setAttribute('stroke', line);
            poly.setAttribute('stroke-width', '1.5');
            poly.setAttribute('stroke-linejoin', 'round');
            poly.setAttribute('vector-effect', 'non-scaling-stroke');
            svg.appendChild(poly);
        }
    }

    svgWrap.append(svg);

    const plotMain = document.createElement('div');
    plotMain.className = 'stats-chart-plot-main';
    plotMain.append(yaxis, svgWrap);

    // Date labels under the plot, aligned to the vertical grid lines above. A
    // hidden spacer matching the widest y-label (plus the plot-row gap) offsets
    // the track so it sits flush under the SVG, then each label is positioned by
    // its tick's x-fraction (ends anchored inward so they don't overflow).
    const fmtDate = (t) => new Date(t).toISOString().slice(5, 10); // MM-DD
    const xaxis = document.createElement('div');
    xaxis.className = 'stats-chart-xaxis stats-chart-xaxis--ticks';
    const xSpacer = document.createElement('span');
    xSpacer.className = 'y-spacer';
    xSpacer.textContent = widest(leftLabels);
    const xTrack = document.createElement('div');
    xTrack.className = 'stats-chart-xaxis-track';
    for (let i = 0; i < xTicks.length; i++) {
        const lbl = document.createElement('span');
        lbl.className = 'x-tick';
        lbl.style.left = `${(xOf(xTicks[i]) / W) * 100}%`;
        lbl.style.transform = i === 0 ? 'translateX(0)'
            : i === xTicks.length - 1 ? 'translateX(-100%)'
            : 'translateX(-50%)';
        lbl.textContent = fmtDate(xTicks[i]);
        xTrack.append(lbl);
    }
    xaxis.append(xSpacer, xTrack);

    plotRow.append(plotMain, xaxis);
    container.append(plotRow);

    const legend = document.createElement('div');
    legend.className = 'stats-chart-legend';
    const legendItems = [
        { kind: 'line', color: '#2e9e5b', text: getTerm('statsLegendSolving') },
        { kind: 'line', color: '#d98a2b', text: getTerm('statsLegendReviewing') },
        { kind: 'band', color: 'rgba(128,128,128,0.30)', text: getTerm('statsLegend65Spread') },
    ];
    for (const item of legendItems) {
        const el = document.createElement('span');
        el.className = `legend-item legend-${item.kind}`;
        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = item.color;
        el.append(swatch, document.createTextNode(item.text));
        legend.append(el);
    }
    container.append(legend);

    const updateLabelVisibility = () => {
        const h = svg.getBoundingClientRect().height;
        if (h === 0) return;
        const tickEls = [...yaxis.querySelectorAll('.y-tick')]
            .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
        const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
        let lastShownPx = -Infinity;
        for (const el of tickEls) {
            const topPx = (parseFloat(el.style.top) / 100) * h;
            if (topPx - lastShownPx >= minSepPx) {
                el.style.visibility = '';
                lastShownPx = topPx;
            } else {
                el.style.visibility = 'hidden';
            }
        }
    };
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(updateLabelVisibility).observe(svg);
    } else {
        queueMicrotask(updateLabelVisibility);
    }

    return container;
}
