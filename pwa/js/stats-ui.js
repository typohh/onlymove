// js/stats-ui.js
//
// The detailed-stats modal: all DOM/SVG rendering for the tabbed stats panel
// (Summary / Weaknesses / Rank / Time / Community) and its charts. Split out of
// model.js, which now holds only the numeric recall model + stats computation.
// The dependency is one-directional: this module consumes the model's exported
// helpers; model.js never reaches back into the UI.

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, getMergedFullAnswerHistory, getTailEvictionStatus } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, TOTAL_RANKS } from './constants.js';
import { getTerm, localizedMotif } from './i18n.js';
import { quadrantProbs } from './bivariate-normal.js';
import { computeRankDecomposition, evaluateAt, rankIndexAfterRatingDelta } from './rank-decomposition.js';
import {
    getPopulationConstants,
    TIME_ADJUST_AFK_CAP_MS,
    TIME_ADJUST_MIN_TIMED,
} from './population-constants.js';
import { RANK_TRANSLATION_SERVERS, internalIndexToServerScalar } from './rank-translation.js';
import { rankScalarToLabel } from '../shared/rank-math.js';
import { showLoading, hideLoading, showConfirm } from './overlays.js';
import {
    refreshHistoryFromServer,
    historyRefreshesLeftToday,
    HISTORY_REFRESH_DAILY_CAP,
} from './account-sync.js';
import { normCdf } from './math-utils.js';
import {
    getRecallCandidates,
    getLogitPerRank,
    prepareTrainingData,
    calculateResultsAndStdErrors,
    freshFailureReviewHours,
    recentFailureWindowHours,
    benefitConfidenceForVar,
    MODEL_INDICES,
} from './model.js';

// --- SVG chart primitives --------------------------------------------------
// The three charts below all build raw SVG nodes; these helpers replace the
// repeated createElementNS + per-attribute setAttribute boilerplate.
const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element, applying `attrs` via setAttribute. */
function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
}

/** An <svg> chart canvas with a `0 0 W H` viewBox. */
function makeChartSvg(W, H) {
    return svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
}

/** A faint horizontal gridline at `y` spanning the full chart width. */
function hGridLine(y, W) {
    return svgEl('line', {
        x1: 0, x2: W, y1: y, y2: y,
        stroke: 'rgba(128,128,128,0.35)', 'stroke-width': '0.5',
        'vector-effect': 'non-scaling-stroke',
    });
}

// If the browser silently evicted history-tail rows, tell the user how many are
// missing and offer a server refresh (subject to the daily cap). No-op when the
// tail is intact, IDB is unavailable, or the user declines.
async function maybeOfferHistoryRefresh() {
    let ev;
    try { ev = await getTailEvictionStatus(); } catch { return; }
    if (!ev || ev.missing <= 0) return;

    const left = historyRefreshesLeftToday();
    if (left <= 0) {
        alert(getTerm('historyEvictedNoRefreshLeft', { count: ev.missing, max: HISTORY_REFRESH_DAILY_CAP }));
        return;
    }
    if (!await showConfirm(getTerm('historyEvictedConfirm', { count: ev.missing, left, max: HISTORY_REFRESH_DAILY_CAP }))) {
        return;
    }
    showLoading(getTerm('historyRefreshing'));
    try {
        const r = await refreshHistoryFromServer();
        if (r.ok) {
            alert(getTerm('historyRefreshDone', { count: r.restored, left: r.left }));
        } else if (r.reason === 'cap') {
            alert(getTerm('historyRefreshCapReached', { max: HISTORY_REFRESH_DAILY_CAP }));
        } else {
            alert(getTerm('historyRefreshFailed'));
        }
    } catch {
        alert(getTerm('historyRefreshFailed'));
    } finally {
        hideLoading();
    }
}

export async function showDetailedStats() {
    // Silent-eviction recovery: if the browser dropped tail rows since our last
    // write, offer to restore them from the server (capped per day — each is a
    // full per-user D1 scan). Done before building so the charts use the
    // recovered history.
    await maybeOfferHistoryRefresh();

    // The stats panel merges both board sizes: one rank, one recall model, one
    // weakness model. Build the union once and thread it through every block.
    // Full career here (working set + IndexedDB cold tail) so the charts and
    // lifetime totals span the whole history, not just the synchronous working
    // set; degrades to the working set if IDB is unavailable.
    const merged = await getMergedFullAnswerHistory();

    // Static explanatory text (not just a spinner): the synchronous chart build
    // below freezes any animation, so a plain "constructing…" line is the
    // reliable signal that the click registered and work is underway.
    showLoading(getTerm('statsConstructing'));
    // Yield a paint so the message is actually visible before the synchronous
    // decomposition + chart build (computeRankDecomposition, buildRatingHistory-
    // Chart, …) take over the main thread.
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    try {
        // The recall model (modelParams) and its derived statsResults drive only
        // the timing-probability text and the freshly-failed window summary; they
        // need a trained model + attempts. When they're missing (a brand-new
        // device with few/no attempts) we skip those blocks and still render
        // whatever panes have data: Community is server-wide and always
        // available, and every chart builder returns null below its own data
        // threshold, so showStatsModal just drops the now-empty tabs. No gate —
        // opening stats with very few or zero attempts is a no-op-safe path.
        if (state.modelParams && !state.statsResults) {
            const validHistory = merged.filter(p => p.problemRank !== undefined && p.timeTaken !== undefined && p.date !== undefined);
            if (validHistory.length) {
                const history = validHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
                const trainingData = prepareTrainingData(history, validHistory);
                const finalResults = calculateResultsAndStdErrors(state.modelParams, trainingData);
                setStatsResults(finalResults);
            }
        }

        const { results = [], varCovarMatrix } = state.statsResults || {};
        const finalParams = state.modelParams;
        // The old single "overview" pane is now split across tabs: timing
        // probabilities ride with the Time chart, the rank decomposition rides
        // with the Rank chart, and only the review-queue counts stay on the
        // (now last) summary pane.
        let timingText = '';
        let resultText = '';

        // Timing observations → Time tab. Recenter by the population confounding
        // bias (benefitConfidenceForVar) so the displayed "% chance more time helps"
        // is the SAME corrected figure the rushing/insufficient-review nudge gate
        // tests against — not the raw, difficulty-confounded coefficient.
        for (const result of results) {
            const { varIndex } = result;
            let termKey;
            if (varIndex === MODEL_INDICES.TIME_SOLVING) termKey = 'statsSolvingTimeNatural';
            else if (varIndex === MODEL_INDICES.REVIEW_TIME) termKey = 'statsReviewTimeNatural';

            if (termKey) {
                const conf = benefitConfidenceForVar(result, varIndex) * 100;
                timingText += getTerm(termKey, { percent_change: conf.toFixed(0) }) + '\n';
            }
        }

        // Block 2: Summary Text. The "freshly-failed problem comes back in ~X h"
        // figure is now the retention window-open delay (NEG_LN_080·S·24) for a
        // just-failed (r=0) repeat at the user's typical repeat difficulty.
        if (varCovarMatrix) {
            const repeatRanks = merged.filter(p => p.isRepeat && typeof p.problemRank === 'number').map(p => p.problemRank);
            const repRank = repeatRanks.length ? repeatRanks.reduce((a, b) => a + b, 0) / repeatRanks.length : 0;
            // Prefer scoring the user's actual recent failures through the model
            // as-if-just-failed-now (real per-problem difficulty/slope/floor +
            // each failure's review percentile, mirroring getRecallCandidates);
            // fall back to the idealized average-difficulty figure when recent
            // failures are too sparse to be representative.
            const firsttimeFailureTime =
                recentFailureWindowHours(finalParams, merged, { userRank: state.userRank })
                ?? freshFailureReviewHours(finalParams, { problemRank: repRank, size: state.boardSize, userRank: state.userRank });

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

                resultText += getTerm('statsFailures', { count_recent: firstTimeFailures, hours: Math.round(firsttimeFailureTime) }) + '\n';
                resultText += getTerm('statsOldFailures', { count_other: Math.max(0, allRepeats - firstTimeFailures) }) + '\n';
            }
        }

        const novelProblemCount = new Set(merged.filter(p => !p.isRepeat).map(p => p.problemId)).size;
        resultText += getTerm('statsNovelProblems', { count: novelProblemCount }) + '\n';

        const slippedProblemCount = getRecallCandidates({
            lowerProbabilityBound: 0,
            upperProbabilityBound: 0.6,
            requireLastAnswerIncorrect: false,
            history: merged,
        }).length;
        const ignoredProblemCount = merged.filter(p => p.ignored).length;
        resultText += getTerm('statsIgnoredProblems', { count_ignored: ignoredProblemCount, count_slipped: slippedProblemCount }) + '\n';

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
        // computeRankDecomposition is the heavy part of this pane; the chart and
        // the summary text both need it, so compute it ONCE here and pass it to
        // both (null = too little data; both handle it).
        const rankDecomp = computeRankDecomposition(merged);
        const rankPane = buildRankPane(
            buildRatingHistoryChart(merged, rankDecomp),
            buildRankSummaryText(merged, rankDecomp));
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

// Inverse standard-normal CDF via bisection over `normCdf`. ~50 iterations
// converges to ≪ 1e-9; cheap once per issue at panel-open time. Returns 0 for
// p ≤ 0.5 since callers only ever pass posterior probabilities in [0.5, 1).
function probit(p) {
    if (!(p > 0.5)) return 0;
    if (p >= 1) return Infinity;
    let lo = 0, hi = 8;
    for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        const cdf = normCdf(mid);
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
// `timeBins` is the seconds-based display-bin table from the caller (TIME_BINS).
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
    // timeBins are in SECONDS now (see TIME_BINS in buildServerStatsPane); the
    // edges align to the server's bucket width, so binning the median directly
    // lands in the same bin the server would have bucketed it into.
    const binSeconds = (ms) => timeBins.findIndex(b => (ms / 1000) >= b.lo && (ms / 1000) < b.hi);

    // Attempts per active (UTC) day — every attempt counts, repeats included.
    const PER_USER_BINS = [[1, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, Infinity]];
    const days = new Set(recent.map(a => new Date(a.date).toISOString().slice(0, 10)));
    if (days.size) {
        const rate = recent.length / days.size;
        out.attemptsPerDay = PER_USER_BINS.findIndex(([lo, hi]) => rate >= lo && rate < hi);
    }

    // Solving time: median of per-attempt solve times (ms), then coarse-binned.
    const solveMed = median(recent.map(a => a.timeTaken).filter(t => typeof t === 'number' && t > 0));
    if (solveMed != null) out.solve = binSeconds(solveMed);

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
    if (reviewMed != null) out.review = binSeconds(reviewMed);

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

    // Display bins in SECONDS over the server's fine-bucket histogram. Solving
    // time is log-normal (the hump sits at ~5–20s), so the low end is split
    // finely and the long tail coarsely — this reveals the bell instead of
    // collapsing it into one tall "0–10s" bar. Every edge is a multiple of the
    // server's bucket width, so a bucket maps cleanly to exactly one display bin.
    const bucketSec = data.bucketSec || 10;
    const TIME_BINS = [
        { lo: 0, hi: 5, label: '0–5s' },
        { lo: 5, hi: 10, label: '5–10s' },
        { lo: 10, hi: 15, label: '10–15s' },
        { lo: 15, hi: 20, label: '15–20s' },
        { lo: 20, hi: 30, label: '20–30s' },
        { lo: 30, hi: 45, label: '30–45s' },
        { lo: 45, hi: 60, label: '45–60s' },
        { lo: 60, hi: 90, label: '1–1.5m' },
        { lo: 90, hi: 120, label: '1.5–2m' },
        { lo: 120, hi: 180, label: '2–3m' },
        { lo: 180, hi: 300, label: '3–5m' },
        { lo: 300, hi: Infinity, label: '5m+' },
    ];
    const binHist = (hist) => {
        const rows = TIME_BINS.map(b => ({ label: b.label, value: 0 }));
        for (const r of (hist || [])) {
            const sec = r.bucket * bucketSec;   // lower edge of the server bucket
            const i = TIME_BINS.findIndex(b => sec >= b.lo && sec < b.hi);
            if (i >= 0) rows[i].value += r.n;
        }
        return rows;
    };
    const ago = (iso) => {
        const ms = Date.now() - Date.parse(iso);
        if (!Number.isFinite(ms) || ms < 0) return getTerm('statsAgoJustNow');
        const h = ms / 3600000;
        if (h < 1) return getTerm('statsAgoMinutes', { n: Math.round(ms / 60000) });
        if (h < 24) return getTerm('statsAgoHours', { n: Math.round(h) });
        return getTerm('statsAgoDays', { n: Math.round(h / 24) });
    };
    // `highlightIndex` (optional) marks the bucket the current user falls into
    // by recolouring that row. The legend explaining the colour is shown once
    // for the whole pane (see below), not per chart.
    // `dropEmpty` (default on) hides zero-count bins so a fixed bin table never
    // renders dead bars — except the user's own bin, which is kept even when no
    // one else lands there so the "you" marker doesn't vanish. Ranks opt out
    // (their empty rungs are a meaningful gap in the ladder, kept on purpose).
    const makeBarList = (rows, highlightIndex = -1, dropEmpty = true) => {
        let display = rows.map((r, i) => ({ ...r, _orig: i }));
        if (dropEmpty) display = display.filter(r => r.value > 0 || r._orig === highlightIndex);
        const hi = display.findIndex(r => r._orig === highlightIndex);
        const max = Math.max(1, ...display.map(r => r.value));
        const wrap = document.createElement('div');
        wrap.className = 'server-bars';
        display.forEach((r, i) => {
            const row = document.createElement('div');
            row.className = 'server-bar-row';
            if (i === hi) row.classList.add('server-bar-row--you');
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
    head.textContent = getTerm('statsCommunityHeader', {
        days: data.windowDays,
        ago: ago(data.computedAt),
    });
    pane.append(head);

    const attempts = data.totals.attempts || 0;
    const wins = data.totals.wins || 0;
    const rate = attempts ? Math.round(wins / attempts * 100) : 0;
    const dayCount = (data.daily && data.daily.length) ? data.daily.length : 0;
    const attemptsPerDay = dayCount ? Math.round(attempts / dayCount) : 0;
    let txt = getTerm('statsActiveUsers', { count: data.totals.users }) + '\n';
    txt += getTerm('statsAttemptsPerDay', { count: attemptsPerDay }) + '\n';
    txt += getTerm('statsAttemptsSolved', { count: attempts, rate }) + '\n';
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
        pane.append(h, makeBarList(data.ranks.map(r => ({ label: r.rank, value: r.n })), rankIdx, false));
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

// medianNovelSolveMinutes + the TIME_ADJUST_* feature constants + the scalar
// timeAdjustDeltaRating now live in population-constants.js (one definition,
// shared with the rank-translation modal). They are imported above.

// A TIME-VARYING version of medianNovelSolveMinutes: the kernel-weighted median
// (in minutes) of the SAME capped-novel timed series the population time-adjust
// alpha was fit against, evaluated at any t. Smoothed at the skill kernel's
// ~30-day scale so the correction tracks the slow skill line, not session noise.
// Where a window has thin timed support the estimate is shrunk toward the
// whole-history median (empirical Bayes in log-minute space) — so the adjusted
// line stays CONTINUOUS, degrading smoothly to today's single-scalar correction
// rather than breaking. NB the inclusion rule (non-repeat, >0, AFK-capped) must
// match user_median_log10_minutes in fit_population_constants — the Time tab's
// solving series is broader (repeats, no AFK cap), so we reuse the SMOOTHER idea
// here, not that series. Returns null (no time adjustment possible) when the
// whole history has fewer than TIME_ADJUST_MIN_TIMED timed attempts.
const TIME_ADJUST_TAU_MS = 30 * 24 * 3600 * 1000;   // matches TAU_S_MS (skill kernel)
const TIME_ADJUST_SHRINK_ESS = 8;                   // log-space shrink prior, in attempts
function makeMedianSolveMinutesAt(history) {
    const pts = history
        .filter(p => !p.isRepeat && Number.isFinite(p.timeTaken)
            && p.timeTaken > 0 && p.timeTaken <= TIME_ADJUST_AFK_CAP_MS
            && typeof p.date === 'number')
        .map(p => ({ t: p.date, v: p.timeTaken }))
        .sort((a, b) => a.t - b.t);
    if (pts.length < TIME_ADJUST_MIN_TIMED) return null;
    const ts = pts.map(p => p.t);
    const sortedAll = pts.map(p => p.v).slice().sort((a, b) => a - b);
    const gm = sortedAll.length % 2 ? sortedAll[sortedAll.length >> 1]
        : (sortedAll[(sortedAll.length >> 1) - 1] + sortedAll[sortedAll.length >> 1]) / 2;
    const globalLog = Math.log10(gm / 60000);

    const wMedian = (pairs, total) => {
        const target = 0.5 * total;
        let cum = 0;
        for (let i = 0; i < pairs.length; i++) {
            const next = cum + pairs[i].w;
            if (next >= target) {
                return i === 0 ? pairs[0].v
                    : pairs[i - 1].v + (pairs[i].v - pairs[i - 1].v) * ((target - cum) / pairs[i].w);
            }
            cum = next;
        }
        return pairs[pairs.length - 1].v;
    };

    return (t) => {
        let W = 0, sumW2 = 0;
        const pairs = [];
        for (let i = 0; i < pts.length; i++) {
            const d = (t - ts[i]) / TIME_ADJUST_TAU_MS;
            if (Math.abs(d) > 4) continue;
            const w = Math.exp(-0.5 * d * d);
            W += w; sumW2 += w * w;
            pairs.push({ v: pts[i].v, w });
        }
        if (W <= 0 || pairs.length < 2) return Math.pow(10, globalLog);
        const ess = (W * W) / sumW2;
        pairs.sort((a, b) => a.v - b.v);
        const localLog = Math.log10(wMedian(pairs, W) / 60000);
        // Shrink the local median toward the global one by effective sample size.
        const wl = ess / (ess + TIME_ADJUST_SHRINK_ESS);
        return Math.pow(10, wl * localLog + (1 - wl) * globalLog);
    };
}

// The rank decomposition text (long-term skill + drift + session swing) and
// the relative 13x13 readout that used to sit on the summary pane. Built as
// part of the Rank tab pane (computeRankDecomposition is heavy, but the whole
// modal is constructed up front under the "constructing charts" overlay). Reads
// state.statsResults (already computed by the time this runs).
function buildRankSummaryText(history, decomp) {
    let rankText = '';

    // Two-component rank decomposition (R = skill S + form M). Gives a
    // long-term-trend number and a session-to-session swing number that can't
    // be read off the single-line smoother in the chart. See
    // rank-decomposition.js for the model and the backfitting approach. `decomp`
    // is normally passed in (computed once for the whole Rank pane); fall back to
    // computing it for any standalone caller.
    if (decomp === undefined) decomp = computeRankDecomposition(history);
    if (decomp) {
        const { summary } = decomp;
        // The current rank line (raw onlymove rank or the selected server's
        // time-adjusted rank) now lives below the CHART, where it reacts to the
        // server dropdown — see the readout in buildRatingHistoryChart. Here we
        // keep only the variance/size readouts the chart can't show.
        if (typeof summary.sigmaM === 'number') {
            rankText += summary.sigmaM >= 0.1
                ? getTerm('statsSessionSwing', {
                    up: summary.sigmaUp.toFixed(2),
                    down: summary.sigmaDown.toFixed(2),
                }) + '\n'
                : getTerm('statsSessionNoSwing', {
                    upper: summary.sigmaMUpper.toFixed(2),
                }) + '\n';
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

// The Rank pane: the chart (which returns { chart, readout }) plus a text
// section. The reactive current-rank readout (driven by the chart's server
// dropdown) leads the text section, above the static decomposition text — so the
// "Latest <server> rank is …" sentence reads as part of the text, not the graph.
function buildRankPane(build, text) {
    const chart = build && build.chart;
    const readout = build && build.readout;
    const hasText = typeof text === 'string' && text.trim().length > 0;
    if (!chart && !readout && !hasText) return null;

    const wrap = document.createElement('div');
    wrap.className = 'stats-chart-pane';
    if (chart) wrap.append(chart);
    if (readout || hasText) {
        const textPart = document.createElement('div');
        textPart.className = 'stats-rank-textpart';
        if (readout) textPart.append(readout);
        if (hasText) {
            const pre = document.createElement('pre');
            pre.className = 'stats-modal-pre';
            pre.textContent = text;
            textPart.append(pre);
        }
        wrap.append(textPart);
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

function buildRatingHistoryChart(history = state.answerHistory, decomp) {
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
    // Passed in by the Rank pane (computed once for both chart + summary); fall
    // back to computing for any standalone caller.
    if (decomp === undefined) decomp = computeRankDecomposition(history);
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

    // Per-grid time-adjustment shift (RATING units), used only for the timed
    // (non-onlymove) server views: the skill line is corrected for the user's
    // typical solving time AT THAT POINT IN TIME — medMin varies over history, so
    // the correction is temporal — since practice here is untimed while server
    // ranks reflect timed play. alpha and the time-vs-rank line (g0/g1) are
    // population constants (fit_population_constants); the per-point center is the
    // time EXPECTED for the skill at that point (e.S, a rank index), so only time
    // unexplained by rank shifts the line. Only the median is temporal. Where time
    // data is thin the shift degrades smoothly to the whole-history correction (see
    // makeMedianSolveMinutesAt). delta = 0 (constants or timed data unavailable) ⇒
    // the timed view becomes pure rank translation with no time correction.
    const ta = getPopulationConstants()?.time_adjust;
    const medAt = makeMedianSolveMinutesAt(history);
    const haveTimeAdjust = !!(ta && Number.isFinite(ta.alpha_per_log10_min)
        && Number.isFinite(ta.time_vs_rank_intercept)
        && Number.isFinite(ta.time_vs_rank_slope) && medAt);
    for (const e of gridEvals) {
        let d = 0;
        if (haveTimeAdjust) {
            const m = medAt(e.t);
            if (Number.isFinite(m) && m > 0) {
                const expectedLog10 = ta.time_vs_rank_intercept + ta.time_vs_rank_slope * e.S;
                d = ta.alpha_per_log10_min * (Math.log10(m) - expectedLog10);
            }
        }
        e.delta = d;
    }

    const container = document.createElement('div');
    container.className = 'stats-chart';

    // Server selector — sits INSIDE the chart title ("[server ▾] rank since
    // <date>"). "OnlyMove" shows the raw, untimed internal rank (the historical
    // behaviour — no time adjustment). Every other option shows the rank as it
    // would read ON THAT SERVER: the internal skill TIME-ADJUSTED for the user's
    // solving speed, then translated through that server's rank curve. Only
    // servers with a usable translation curve are offered. EGF is the identity
    // hub, so "EGF" vs "OnlyMove" differ only by the time adjustment.
    const ONLYMOVE = 'OnlyMove';
    const serverOptions = [ONLYMOVE, ...RANK_TRANSLATION_SERVERS.filter(
        s => Number.isFinite(internalIndexToServerScalar(19, s)))];
    let selectedServer = ONLYMOVE;
    const select = document.createElement('select');
    select.className = 'stats-rank-server-select';
    for (const s of serverOptions) {
        const opt = document.createElement('option');
        opt.value = s;
        opt.textContent = s;   // option value IS the display text ("OnlyMove", "OGS", …)
        select.appendChild(opt);
    }
    select.value = selectedServer;

    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    title.append(select, document.createTextNode(
        ' ' + getTerm('statsChartRankSince', { date: new Date(t0).toISOString().slice(0, 10) })));
    container.append(title);

    // Display transforms for the selected server. The skill line + its band edges
    // are SKILL values, so they take the (temporal) time adjustment then translate;
    // the scatter dots are PROBLEM difficulties (independent of how fast the user
    // solves), so they translate only. onlymove = identity (raw internal index).
    const skillToDisplay = (internalIdx, delta) => {
        if (selectedServer === ONLYMOVE) return internalIdx;
        const adj = (Number.isFinite(delta) && delta !== 0)
            ? rankIndexAfterRatingDelta(internalIdx, delta) : internalIdx;
        return internalIndexToServerScalar(Number.isFinite(adj) ? adj : internalIdx, selectedServer);
    };
    const rankToDisplay = (internalIdx) =>
        selectedServer === ONLYMOVE ? internalIdx
            : internalIndexToServerScalar(internalIdx, selectedServer);
    // Integer display value (rank-band floor) → label: an internal index uses the
    // kyu/dan index labeller; a server scalar uses the shared scalar labeller.
    const displayLabel = (intVal) =>
        selectedServer === ONLYMOVE ? rankLabelFromIndex(intVal) : rankScalarToLabel(intVal);

    // The plot (y-axis + svg + x-date) is rebuilt into this holder on every
    // server change; the title, readout and legend persist around it.
    const plotHolder = document.createElement('div');

    // Current-rank readout below the graph: the selected server's current rank
    // (raw for onlymove, time-adjusted + translated otherwise). Reacts to the
    // dropdown — this replaced the old "stable / time adjusted rank" summary text.
    // Same element/class as the decomposition text (stats-modal-pre) so it reads
    // as just another line of that text, not a distinct callout.
    const readout = document.createElement('pre');
    readout.className = 'stats-modal-pre stats-rank-readout';

    // Legend (server-independent text); skill curve + bands listed only when the
    // decomposition produced them.
    const SKILL_FILL_OUTER = 'rgba(59,130,246,0.10)';
    const SKILL_FILL_INNER = 'rgba(59,130,246,0.22)';
    const SKILL_STROKE = '#2563eb';
    const DOT_CORRECT = '#50b450', DOT_WRONG = '#dc5050';
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

    // Rebuild the plot for the current selectedServer. Everything below is in
    // DISPLAY space (internal index for onlymove, that server's rank scalar
    // otherwise); the band edges are transformed individually so the monotone
    // adjust+translate keeps them ordered across server-curve knots.
    let labelObserver = null;
    const renderPlot = () => {
        const gd = [];
        for (const e of gridEvals) {
            const S = skillToDisplay(e.S, e.delta);
            const up = skillToDisplay(e.S + outerUp(e), e.delta);
            const dn = skillToDisplay(e.S - outerDown(e), e.delta);
            const iu = skillToDisplay(e.S + innerHalf(e), e.delta);
            const id = skillToDisplay(e.S - innerHalf(e), e.delta);
            if ([S, up, dn, iu, id].every(Number.isFinite)) gd.push({ t: e.t, S, up, dn, iu, id });
        }

        // Readout = the current rank (the line's right endpoint) in the selected
        // server's scale. Hidden when there's no skill curve (too little data).
        if (gd.length >= 1) {
            const cur = gd[gd.length - 1].S;
            const lower = Math.floor(cur);
            readout.textContent = getTerm('statsSelectedServerRank', {
                server: selectedServer,
                cur: displayLabel(lower),
                pct: Math.round((cur - lower) * 100),
                next: displayLabel(lower + 1),
            });
            readout.hidden = false;
        } else {
            readout.hidden = true;
        }

        const dots = [];
        for (const e of entries) {
            const y = rankToDisplay(e.rankValue);
            if (Number.isFinite(y)) dots.push({ t: e.date, y, correct: e.correct });
        }

        // Focus the y-range on the skill band — not the (far wider) spread of
        // problem difficulties. Dots outside this window are dropped below.
        let bandMin = Infinity, bandMax = -Infinity;
        for (const g of gd) { bandMin = Math.min(bandMin, g.dn); bandMax = Math.max(bandMax, g.up); }
        if (!Number.isFinite(bandMin)) {
            // No decomposition — fall back to the scatter range.
            const ys = dots.map(d => d.y).slice().sort((a, b) => a - b);
            bandMin = ys.length ? ys[0] : 0;
            bandMax = ys.length ? ys[ys.length - 1] : 1;
        }
        const Y_PAD = 1; // exactly one rank beyond the outer CI band
        let yHigh = bandMax + Y_PAD;
        let yLow = bandMin - Y_PAD;
        if (yHigh <= yLow) yHigh = yLow + 1;

        const plotRow = document.createElement('div');
        plotRow.className = 'stats-chart-plot-row';

        // Grid lines mark the rank CUTOFFS (integer boundaries); each label sits
        // BETWEEN its two bounding lines (at r+0.5) so it reads as "this band is
        // rank X". Overlapping labels get hidden after layout (see below).
        const gridLines = [];
        for (let r = Math.ceil(yLow); r <= Math.floor(yHigh); r++) gridLines.push(r);
        const labelTicks = [];
        for (let r = Math.floor(yLow); r <= Math.ceil(yHigh); r++) {
            const mid = r + 0.5;
            if (mid < yLow || mid > yHigh) continue;
            labelTicks.push({ pos: mid, text: displayLabel(r) });
        }

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
        const W = 400, H = chartViewBoxHeight(W);
        const xOf = t => ((t - t0) / tSpan) * W;
        const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;
        const svg = makeChartSvg(W, H);

        for (const r of gridLines) svg.appendChild(hGridLine(yOf(r), W));

        // Per-attempt scatter: green=correct, red=wrong, drawn UNDER the bands so
        // the curve always reads on top. Cap the rendered count (stride) and fade
        // opacity by the count actually INSIDE the window (most dots clip out, as
        // problem ranks scatter far wider than the band).
        const visibleDots = dots.filter(d => d.y >= yLow && d.y <= yHigh);
        const SCATTER_CAP = 1500;
        const stride = Math.max(1, Math.ceil(visibleDots.length / SCATTER_CAP));
        const shownDots = stride > 1 ? visibleDots.filter((_, i) => i % stride === 0) : visibleDots;
        const dotR = shownDots.length > 600 ? '1.4' : '1.6';
        const dotAlpha = Math.max(0.15, Math.min(0.6, 50 / Math.max(1, shownDots.length))).toFixed(3);
        for (const d of shownDots) {
            svg.appendChild(svgEl('circle', {
                cx: xOf(d.t).toFixed(1), cy: yOf(d.y).toFixed(1), r: dotR,
                fill: d.correct ? DOT_CORRECT : DOT_WRONG, 'fill-opacity': dotAlpha,
            }));
        }

        // Two nested bands: outer (lighter) = prediction interval (where a
        // session's form lands), inner (darker) = measurement uncertainty. Each
        // edge is already in display space; top runs left→right, bottom closes it
        // right→left.
        const bandPath = (top, bot) => {
            let d = '';
            for (let i = 0; i < gd.length; i++) d += (i === 0 ? 'M' : 'L') + `${xOf(gd[i].t).toFixed(1)} ${yOf(gd[i][top]).toFixed(1)} `;
            for (let i = gd.length - 1; i >= 0; i--) d += `L${xOf(gd[i].t).toFixed(1)} ${yOf(gd[i][bot]).toFixed(1)} `;
            return d + 'Z';
        };
        if (gd.length >= 2) {
            svg.appendChild(svgEl('path', { d: bandPath('up', 'dn'), fill: SKILL_FILL_OUTER, stroke: 'none' }));
            svg.appendChild(svgEl('path', { d: bandPath('iu', 'id'), fill: SKILL_FILL_INNER, stroke: 'none' }));
            const skillPts = gd.map(g => `${xOf(g.t).toFixed(1)},${yOf(g.S).toFixed(1)}`).join(' ');
            svg.appendChild(svgEl('polyline', {
                points: skillPts, fill: 'none', stroke: SKILL_STROKE,
                'stroke-width': '1.5', 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke',
            }));
        }

        svgWrap.append(svg);
        const plotMain = document.createElement('div');
        plotMain.className = 'stats-chart-plot-main';
        plotMain.append(yaxis, svgWrap);
        plotRow.append(plotMain);   // no x-axis end-date label

        // Greedy-hide y-axis labels that would physically overlap; re-runs on
        // resize. Re-created per render, so disconnect the prior observer first.
        const updateLabelVisibility = () => {
            const h = svg.getBoundingClientRect().height;
            if (h === 0) return;
            const tickEls = [...yaxis.querySelectorAll('.y-tick')]
                .sort((a, b) => parseFloat(a.style.top) - parseFloat(b.style.top));
            const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
            let lastShownPx = -Infinity;
            for (const el of tickEls) {
                const topPx = (parseFloat(el.style.top) / 100) * h;
                if (topPx - lastShownPx >= minSepPx) { el.style.visibility = ''; lastShownPx = topPx; }
                else el.style.visibility = 'hidden';
            }
        };
        if (labelObserver) { labelObserver.disconnect(); labelObserver = null; }
        if (typeof ResizeObserver !== 'undefined') {
            labelObserver = new ResizeObserver(updateLabelVisibility);
            labelObserver.observe(svg);
        } else {
            queueMicrotask(updateLabelVisibility);
        }

        plotHolder.replaceChildren(plotRow);
    };

    select.addEventListener('change', () => {
        selectedServer = select.value;
        renderPlot();
    });
    renderPlot();

    container.append(plotHolder, legend);
    // The readout is NOT appended here — it's returned so the Rank pane can place
    // it in the text section (below the chart). renderPlot still drives it via the
    // shared reference, so it stays reactive to the dropdown wherever it's placed.
    return { chart: container, readout };
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
    title.textContent = getTerm('statsChartProblemsPerDaySince', { date: new Date(d0).toISOString().slice(0, 10) });
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
    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - xStart) / total) * W;
    const yOf = v => (1 - v / yHigh) * H;

    const svg = makeChartSvg(W, H);

    // Horizontal grid at each y-tick.
    for (let i = 0; i <= 4; i++) {
        svg.appendChild(hGridLine(yOf(step * i), W));
    }

    // One bar per active day. Leave a thin gap between adjacent days unless the
    // span is so long that bars are already sub-pixel.
    const barW = (DAY_MS / total) * W;
    const gap = barW > 2 ? Math.min(0.6, barW * 0.15) : 0;
    for (const [k, c] of counts) {
        const x = xOf(k) + gap / 2;
        const w = Math.max(0.4, barW - gap);
        const y = yOf(c);
        svg.appendChild(svgEl('rect', {
            x: x.toFixed(2), y: y.toFixed(2),
            width: w.toFixed(2), height: (H - y).toFixed(2),
            fill: '#6f8bbf',
        }));
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
// line: at each date it's the kernel-weighted 25%/75% quantiles of the
// nearby attempts' times — i.e. "the middle 50% of attempts land in here" (the
// interquartile range, read off the empirical
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
            // sits inside the band. Band = 25%/75% edges = the middle 50% of
            // attempts (the interquartile range, read off the EMPIRICAL skewed
            // distribution — no normal assumption).
            grid[g] = {
                t,
                mid: wQuantile(pairs, W, 0.5),
                lo: wQuantile(pairs, W, 0.25),
                hi: wQuantile(pairs, W, 0.75),
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
    // line out of the window (the bug the spread band exposed).
    const hiVals = [];
    for (const grid of [sGrid, rGrid]) if (grid) for (const e of grid) if (e) hiVals.push(Math.max(e.hi, e.mid));
    hiVals.sort((a, b) => a - b);
    const rawHigh = Math.max(quantile(hiVals, 0.90) * 1.05, 1);
    const yLow = 0;
    // Two-tier y ticks. The CI band can reach several minutes while both median
    // lines sit under a minute, so a single uniform step leaves the medians
    // stranded between gridlines. Instead: a COARSE step (~4 ticks) spans the
    // whole CI range, and a FINER step (~4 ticks) adds resolution across the
    // median band [0, max median]. Both steps come off a seconds→minutes ladder,
    // so every tick is a whole minute or a round number of seconds, and the two
    // tiers share ticks where they coincide (e.g. 0, 1m). The render-time
    // overlap pass below then drops whatever doesn't fit — coarse first — so we
    // show as many as there's room for and never overlap.
    const NICE = [10, 15, 20, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 3600, 7200];
    const niceStep = (span, target) => NICE.find(s => s >= span / target) ?? NICE[NICE.length - 1];
    // value → tier (0 = coarse/preferred, 1 = fine). A Map dedupes shared ticks.
    let medMax = 0;
    for (const grid of [sGrid, rGrid]) if (grid) for (const e of grid) if (e) medMax = Math.max(medMax, e.mid);
    const hiStep = niceStep(rawHigh, 4);
    const yHigh = Math.ceil(rawHigh / hiStep) * hiStep;
    // Median band hugs the actual medians (small headroom so a fine tick lands
    // just above the highest one), capped at the ceiling. niceStep self-scales to
    // ~4 ticks, so a sub-minute band yields sub-minute ticks — DON'T inflate it to
    // a full coarse step: that spread the dense ticks more than twice past the
    // medians (e.g. 30s/1m/90s when every median was under a minute).
    medMax = medMax > 0 ? Math.min(medMax * 1.1, yHigh) : 0;
    const lowStep = niceStep(medMax, 4);
    const fmt = (s) => s === 0 ? '0' : (s % 60 === 0 ? `${s / 60}m` : `${Math.round(s)}s`);
    const tierByVal = new Map();
    for (let v = 0; v <= yHigh + 1e-6; v += hiStep) tierByVal.set(Math.round(v), 0);
    for (let v = lowStep; v <= medMax + 1e-6; v += lowStep)
        if (!tierByVal.has(Math.round(v))) tierByVal.set(Math.round(v), 1);
    const yTicks = [...tierByVal.entries()].map(([v, pri]) => ({ v, pri })).sort((a, b) => a.v - b.v);

    const container = document.createElement('div');
    container.className = 'stats-chart';

    const title = document.createElement('div');
    title.className = 'stats-chart-title';
    title.textContent = getTerm('statsChartTimePerProblemSince', { date: new Date(t0).toISOString().slice(0, 10) });
    container.append(title);

    const plotRow = document.createElement('div');
    plotRow.className = 'stats-chart-plot-row';

    const yaxis = document.createElement('div');
    yaxis.className = 'stats-chart-yaxis';
    const leftLabels = yTicks.map(t => fmt(t.v));
    const widest = arr => arr.reduce((a, b) => (b.length > a.length ? b : a), '');
    const spacer = document.createElement('span');
    spacer.className = 'y-spacer';
    // Size the column to the WIDEST label, not just the top one — an
    // intermediate tick ("45s") can be wider than the ceiling ("3m") and
    // would otherwise overflow the column and clip on the left.
    spacer.textContent = widest(leftLabels);
    yaxis.append(spacer);
    const topPct = v => (1 - (v - yLow) / (yHigh - yLow)) * 100;
    // tickRows pairs each label with its grid line (created below) and tier, so
    // the overlap pass can hide both together and keep coarse ticks first.
    const tickRows = yTicks.map(t => {
        const labelEl = document.createElement('div');
        labelEl.className = 'y-tick';
        labelEl.style.top = `${topPct(t.v)}%`;
        labelEl.textContent = fmt(t.v);
        yaxis.append(labelEl);
        return { v: t.v, pri: t.pri, labelEl, lineEl: null };
    });

    const svgWrap = document.createElement('div');
    svgWrap.className = 'stats-chart-svg-wrap';
    const W = 400, H = chartViewBoxHeight(W);
    const xOf = t => ((t - t0) / tSpan) * W;
    const yOf = v => (1 - (v - yLow) / (yHigh - yLow)) * H;

    const svg = makeChartSvg(W, H);
    svg.style.overflow = 'hidden'; // clip clipped-day spikes at the top edge

    // Grid lines here can be vertical (x-ticks) too, so this takes explicit
    // endpoints rather than using hGridLine.
    const gridLine = (x1, y1, x2, y2) => {
        const line = svgEl('line', {
            x1, x2, y1, y2,
            stroke: 'rgba(128,128,128,0.35)', 'stroke-width': '0.5',
            'vector-effect': 'non-scaling-stroke',
        });
        svg.appendChild(line);
        return line;
    };

    // Horizontal grid at each y-tick, kept in sync with its label's visibility.
    for (const row of tickRows) {
        row.lineEl = gridLine(0, yOf(row.v), W, yOf(row.v));
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
            svg.appendChild(svgEl('path', { d: d + 'Z', fill: band, stroke: 'none' }));
        }
    }

    for (const { grid, line } of SERIES) {
        if (!grid) continue;
        for (const run of segments(grid)) {
            if (run.length === 1) {
                svg.appendChild(svgEl('circle', {
                    cx: xOf(run[0].t).toFixed(1), cy: yOf(run[0].mid).toFixed(1),
                    r: '2', fill: line,
                }));
                continue;
            }
            const pts = run.map(e => `${xOf(e.t).toFixed(1)},${yOf(e.mid).toFixed(1)}`).join(' ');
            svg.appendChild(svgEl('polyline', {
                points: pts, fill: 'none', stroke: line,
                'stroke-width': '1.5', 'stroke-linejoin': 'round',
                'vector-effect': 'non-scaling-stroke',
            }));
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
        { kind: 'band', color: 'rgba(128,128,128,0.30)', text: getTerm('statsLegend50Spread') },
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

    // Keep as many ticks as physically fit, coarse tier first. Walk ticks by
    // priority (coarse before fine), then by position; keep one only if it
    // clears minSep from every already-kept tick — so the CI-spanning coarse
    // ticks always win and the finer median-band ticks fill the gaps where
    // there's room, never overlapping. Each label and its grid line toggle as one.
    const updateLabelVisibility = () => {
        const h = svg.getBoundingClientRect().height;
        if (h === 0) return;
        const minSepPx = parseFloat(getComputedStyle(yaxis).lineHeight) || 18;
        const rows = tickRows
            .map(r => ({ r, px: (parseFloat(r.labelEl.style.top) / 100) * h }))
            .sort((a, b) => (a.r.pri - b.r.pri) || (a.px - b.px));
        const kept = [];
        for (const { r, px } of rows) {
            const show = kept.every(p => Math.abs(p - px) >= minSepPx);
            r.labelEl.style.visibility = show ? '' : 'hidden';
            if (r.lineEl) r.lineEl.style.display = show ? '' : 'none';
            if (show) kept.push(px);
        }
    };
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(updateLabelVisibility).observe(svg);
    } else {
        queueMicrotask(updateLabelVisibility);
    }

    return container;
}
