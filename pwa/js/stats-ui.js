// js/stats-ui.js
//
// The detailed-stats modal ORCHESTRATOR: gathers the merged full history,
// computes the shared numeric inputs once (the recall model's statsResults,
// the rank decomposition), builds every tab pane, and shows the tabbed modal
// (Weaknesses / Rank / Time / Activity / Community). The panes live in sibling
// modules — stats-weaknesses.js, stats-rank-pane.js, stats-time-chart.js,
// stats-activity-chart.js, stats-community.js — built on the shared SVG/chart
// primitives in chart-svg.js and the pure-numeric helpers in stats-compute.js.
// model.js holds the numeric recall model + stats computation; the dependency
// is one-directional: this module consumes the model's exported helpers;
// model.js never reaches back into the UI.

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, getMergedFullAnswerHistory, getTailEvictionStatus } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { getTerm } from './i18n.js';
import { computeRankDecomposition } from './rank-decomposition.js';
import { showLoading, hideLoading, showConfirm } from './overlays.js';
import {
    refreshHistoryFromServer,
    historyRefreshesLeftToday,
    HISTORY_REFRESH_DAILY_CAP,
} from './account-sync.js';
import { buildLikelyIssuesElement } from './stats-weaknesses.js';
import { buildRankSummaryText, buildRatingHistoryChart } from './stats-rank-pane.js';
import { buildTimeUsageChart } from './stats-time-chart.js';
import { buildProblemsPerDayChart } from './stats-activity-chart.js';
import { buildServerStatsPane } from './stats-community.js';
import {
    getRecallCandidates,
    prepareTrainingData,
    calculateResultsAndStdErrors,
    freshFailureReviewHours,
    recentFailureWindowHours,
    averageRepeatRank,
    benefitConfidenceForVar,
    MODEL_INDICES,
} from './model.js';

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
        // Timing probabilities ride with the Time chart, the rank decomposition
        // rides with the Rank chart, and only the review-queue counts stay on
        // the Activity pane.
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

        // Block 2: Activity text. The "freshly-failed problem comes back in ~X h"
        // figure is the retention window-open delay (NEG_LN_080·S·24) for a
        // just-failed (r=0) repeat at the user's typical repeat difficulty.
        if (varCovarMatrix) {
            const repRank = averageRepeatRank(merged, 0);
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

        // Two-level weakness detection (stats-weaknesses.js): the augmented
        // model's posterior on the identified sum β_correct[k] + β_picked[k]
        // flags weak axes, and a paired picked-vs-correct sign test qualifies
        // the direction. Rendered as a DOM block (instead of inlined into
        // resultText) so each issue can carry an interactive `examples` toggle
        // to past mistakes.
        const issuesElement = buildLikelyIssuesElement(merged);

        // Activity pane: the problems-per-day chart above the review-queue
        // summary text.
        const activityPane = combineChartWithText(buildProblemsPerDayChart(merged), resultText);

        // Community/server-wide stats (null if the endpoint isn't ready or the
        // fetch fails — the tab is then dropped).
        const serverPane = await buildServerStatsPane(merged);

        // The Rank and Time charts are the heavy part (rank decomposition +
        // kernel smoothing + bootstrap CI), but they're built eagerly here —
        // under the "constructing charts" overlay shown above — so the modal
        // opens fully-ready and every tab switch is instant. The overlay
        // explains the build delay, which is why no in-modal lazy-load / spinner
        // / pre-warm is needed. The timing probabilities and rank
        // decomposition text ride with their charts.
        // computeRankDecomposition is the heavy part of this pane; the chart and
        // the summary text both need it, so compute it ONCE here and pass it to
        // both (null = too little data; both handle it).
        const rankDecomp = computeRankDecomposition(merged);
        const rankBuild = buildRatingHistoryChart(merged, rankDecomp);
        const rankPane = combineChartWithText(
            rankBuild && rankBuild.chart,
            buildRankSummaryText(merged, rankDecomp),
            rankBuild && rankBuild.readout);
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

// A tab pane that pairs a chart with its explanatory text (timing
// probabilities for Time, the rank decomposition for Rank). Either piece may
// be missing: returns null only when nothing is there, so an
// empty tab is dropped by showStatsModal. `lead` is an optional reactive
// element that heads the text section — the Rank pane's current-rank readout
// (driven by the chart's server dropdown): readout + text are grouped in a
// `.stats-rank-textpart` so the "Latest <server> rank is …" sentence reads as
// part of the text, not the graph.
function combineChartWithText(chartElement, text, lead = null) {
    const hasText = typeof text === 'string' && text.trim().length > 0;
    if (!chartElement && !lead && !hasText) return null;

    const wrap = document.createElement('div');
    wrap.className = 'stats-chart-pane';
    if (chartElement) wrap.append(chartElement);
    const makePre = () => {
        const pre = document.createElement('pre');
        pre.className = 'stats-modal-pre';
        pre.textContent = text;
        return pre;
    };
    if (lead) {
        const textPart = document.createElement('div');
        textPart.className = 'stats-rank-textpart';
        textPart.append(lead);
        if (hasText) textPart.append(makePre());
        wrap.append(textPart);
    } else if (hasText) {
        wrap.append(makePre());
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
