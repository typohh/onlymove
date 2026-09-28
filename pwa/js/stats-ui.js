// js/stats-ui.js
//
// The detailed-stats modal ORCHESTRATOR: gathers the merged full history,
// computes the shared numeric inputs once (the recall model's statsResults,
// the rank decomposition), builds every tab pane, and shows the tabbed modal
// (Weaknesses / Tendencies / Rank / Time / Activity). The panes live in sibling
// modules — stats-weaknesses.js, stats-rank-pane.js, stats-time-chart.js,
// stats-activity-chart.js — built on the shared SVG/chart
// primitives in chart-svg.js and the pure-numeric helpers in stats-compute.js.
// recall-model.js holds the numeric recall model + stats computation; the dependency
// is one-directional: this module consumes the model's exported helpers;
// recall-model.js never reaches back into the UI.

import { captureException } from "@sentry/browser";
import { state, setStatsResults, getMergedAnswerHistory, getMergedFullAnswerHistory, getTailEvictionStatus } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { getTerm } from './i18n.js';
import { computeRankDecomposition } from './rank-decomposition.js';
import { showLoading, hideLoading, showConfirm } from './overlays.js';
import {
    refreshHistoryFromServer,
    historyRefreshesLeftToday,
    HISTORY_REFRESH_DAILY_CAP,
} from './account-sync.js';
import { buildMotifPane } from './stats-weaknesses.js';
import { buildRankSummaryText, buildRatingHistoryChart } from './stats-rank-pane.js';
import { buildTimeUsageChart } from './stats-time-chart.js';
import { buildProblemsPerDayChart } from './stats-activity-chart.js';
import {
    getRecallCandidates,
    prepareTrainingData,
    calculateResultsAndStdErrors,
    freshFailureReviewHours,
    recentFailureWindowHours,
    averageRepeatRank,
    benefitConfidenceForVar,
    MODEL_INDICES,
} from './recall-model.js';

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
        // whatever panes have data: every chart builder returns null below its
        // own data threshold, so showStatsModal just drops the now-empty tabs. No gate —
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
        // is the SAME corrected figure the pace readings (pace-readings.js) are built
        // from — not the raw, difficulty-confounded coefficient.
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

        // Distinct novel problems attempted, split per board size on one line —
        // 9×9 and 13×13 are disjoint corpora, so the pooled count hid how much
        // of each the user has worked through.
        const novelCountForSize = (size) => new Set(
            merged.filter(p => !p.isRepeat && p.size === size).map(p => p.problemId)
        ).size;
        resultText += getTerm('statsNovelProblemsBySize', {
            count9: novelCountForSize(9),
            count13: novelCountForSize(13),
        }) + '\n';

        const slippedProblemCount = getRecallCandidates({
            lowerProbabilityBound: 0,
            upperProbabilityBound: 0.6,
            requireLastAnswerIncorrect: false,
            history: merged,
        }).length;
        const ignoredProblemCount = merged.filter(p => p.ignored).length;
        resultText += getTerm('statsIgnoredProblems', { count_ignored: ignoredProblemCount, count_slipped: slippedProblemCount }) + '\n';

        // Per-motif readings (stats-weaknesses.js): a 3PL-residual slope on each
        // motif loading and a sign test on the user's own wrong picks, both
        // over their own attempts. Rendered as a DOM block (instead of inlined
        // into resultText) so a row can carry an interactive `examples` toggle
        // to past mistakes.
        //
        // The WORKING SET, not `merged`: this is the one block that reads
        // embedding vectors, and a vector is only trustworthy where the
        // embedding_version gate can reach it. That gate (backfill's refit
        // wipe) rewrites the working set alone, so a working-set vector is
        // always from the shipped fit, while the cold tail is outside it — and
        // a superseded fit's vector is not detectably wrong, since a refit can
        // keep K and move every axis under it. Nothing is lost: the readings
        // slice to MODEL_INPUT_HISTORY_LENGTH, six times inside this window.
        // Two panes over ONE computation of the readings — the split is in the
        // rendering, so the strength pane and the tendency pane can never
        // disagree about a motif.
        const motifHistory = getMergedAnswerHistory();
        const weaknessPane = buildMotifPane('strength', motifHistory);
        const tendencyPane = buildMotifPane('tendency', motifHistory);

        // Activity pane: the problems-per-day chart above the review-queue
        // summary text.
        const activityPane = combineChartWithText(buildProblemsPerDayChart(merged), resultText);

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
        // The three chart panes are also handed a `make` so a resize can
        // re-derive their aspect ratio (see showStatsModal). Each closes over
        // the data already computed above — rankDecomp especially, which is the
        // expensive part of the Rank pane — so a rebuild redraws the chart
        // without redoing the analysis.
        const makeRankPane = () => {
            const build = buildRatingHistoryChart(merged, rankDecomp);
            return combineChartWithText(
                build && build.chart,
                buildRankSummaryText(merged, rankDecomp),
                build && build.readout);
        };
        const makeTimePane = () => combineChartWithText(buildTimeUsageChart(merged), timingText);
        const makeActivityPane = () => combineChartWithText(buildProblemsPerDayChart(merged), resultText);

        // Weaknesses is the default (first) page. Empty panes (no weaknesses, too
        // little data for a chart) are dropped by showStatsModal.
        showStatsModal([
            { label: getTerm('statsTabWeaknesses'), content: weaknessPane },
            { label: getTerm('statsTabTendencies'), content: tendencyPane },
            { label: getTerm('statsTabRank'), content: makeRankPane(), make: makeRankPane },
            { label: getTerm('statsTabTime'), content: makeTimePane(), make: makeTimePane },
            { label: getTerm('statsTabActivity'), content: activityPane, make: makeActivityPane },
        ]);

    } catch (error) {
        console.error('Failed to generate stats text:', error);
        captureException(error);
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
// The open (possibly hidden) stats modal — hidden while an in-instance
// weakness-example review holds the board (stats-weaknesses.js), restored
// with its full DOM state (active tab, scroll, expanded lists) on exit.
let activeStatsModal = null;

// Hide/restore around an example review; discard when the review is abandoned
// through a side door (problem-loader.js's abandonExampleReviewIfActive) so a
// stale modal can't linger hidden in the DOM.
export function hideStatsModal() {
    if (activeStatsModal) activeStatsModal.backdrop.style.display = 'none';
}
export function restoreStatsModal() {
    if (activeStatsModal) activeStatsModal.backdrop.style.display = '';
}
export function discardStatsModal() {
    if (activeStatsModal) activeStatsModal.close();
}

// A tab is `{ label, content }` for a static pane, or `{ label, content, make }`
// where `make()` rebuilds the pane from scratch. `make` marks a pane whose
// layout is viewport-dependent (the three chart panes): its chart bakes an
// aspect ratio chosen from the modal's visible box at build time
// (chartViewBoxHeight), so a rotate or a window resize would otherwise leave
// the chart at the previous viewport's proportions — the exact thing that
// sizing is for. Panes are rebuilt lazily: a resize invalidates every
// viewport-dependent pane, the visible one is rebuilt at once and the rest when
// their tab is next selected, so a rotate costs one chart rebuild rather than
// re-running every pane's data pipeline.
function showStatsModal(tabs) {
    const entries = tabs
        .filter(t => t.content)
        .map(t => ({ label: t.label, node: t.content, make: t.make || null }));
    if (entries.length === 0) return;

    // A modal still hidden behind an example review is superseded by a freshly
    // opened one (menu → Stats is reachable from analysis mode).
    if (activeStatsModal) activeStatsModal.close();

    const backdrop = document.createElement('div');
    backdrop.className = 'stats-modal-backdrop';

    const card = document.createElement('div');
    card.className = 'stats-modal-card';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'stats-modal-close';
    closeBtn.textContent = getTerm('statsClose');

    const close = () => {
        document.removeEventListener('keydown', onKey);
        window.removeEventListener('resize', onResize);
        clearTimeout(resizeTimer);
        backdrop.remove();
        if (activeStatsModal && activeStatsModal.backdrop === backdrop) activeStatsModal = null;
    };
    // Escape must not close the modal while it's hidden behind an example
    // review — the review's exit would then have nothing to restore.
    const onKey = (e) => { if (e.key === 'Escape' && backdrop.style.display !== 'none') close(); };
    closeBtn.onclick = close;
    backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
    document.addEventListener('keydown', onKey);

    const tabbar = document.createElement('div');
    tabbar.className = 'stats-tabbar';
    const body = document.createElement('div');
    body.className = 'stats-modal-body';

    let current = entries[0];

    // Rebuild a pane invalidated by a resize. A make() that returns nothing
    // leaves the existing node in place rather than blanking the tab.
    const nodeFor = (entry) => {
        if (!entry.node && entry.make) entry.node = entry.make() || entry.node;
        return entry.node;
    };

    const renderBody = ({ keepScroll = false } = {}) => {
        const scroll = body.scrollTop;
        body.replaceChildren(nodeFor(current));
        body.scrollTop = keepScroll ? scroll : 0;
    };

    // Debounced: a drag-resize fires continuously, and a rebuild redraws a
    // chart. Only the visible pane is rebuilt now; the others are dropped and
    // rebuilt when selected.
    let resizeTimer = 0;
    const onResize = () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            let changed = false;
            for (const e of entries) {
                if (e.make) { e.node = null; changed = true; }
            }
            if (changed && current.make) renderBody({ keepScroll: true });
        }, 200);
    };
    window.addEventListener('resize', onResize);
    const renderTabbar = () => {
        // The bar is omitted when only one pane remains.
        tabbar.style.display = entries.length > 1 ? '' : 'none';
        tabbar.replaceChildren();
        for (const e of entries) {
            const b = document.createElement('button');
            b.className = 'stats-tab';
            b.classList.toggle('active', e === current);
            b.textContent = e.label;
            b.onclick = () => {
                current = e; renderTabbar(); renderBody();
                // The bar scrolls rather than wraps; bring the chosen tab fully in.
                tabbar.querySelector('.stats-tab.active')?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
            };
            tabbar.append(b);
        }
    };

    card.append(tabbar, body, closeBtn);
    backdrop.appendChild(card);
    document.body.appendChild(backdrop);
    activeStatsModal = { backdrop, close };

    renderTabbar();
    renderBody();
}
