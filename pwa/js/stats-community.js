// js/stats-community.js
//
// The Community pane of the detailed-stats modal (stats-ui.js is the
// orchestrator). Rendering-only: the display-bin table (TIME_BINS), the
// server-histogram folding (binHist), and the user's own bucket placement
// (computeUserCommunityBuckets) live in stats-compute.js.

import { getTerm } from './i18n.js';
import { computeUserCommunityBuckets, TIME_BINS, binHist } from './stats-compute.js';

// Community pane: server-wide stats over the preceding 7 days (computed by the
// Worker cron, served from /server-stats). Returns null on any failure or when
// there's no activity, so showStatsModal drops the tab. Labels are localized
// via getTerm (stats* keys in i18n.js).
export async function buildServerStatsPane(history = []) {
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

    // Server fine-bucket width (seconds); TIME_BINS' edges are all multiples of
    // it, so binHist folds each server bucket into exactly one display bin.
    const bucketSec = data.bucketSec || 10;
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
    // server-side (see computeServerStats in src/server-stats.js), so the highlighted
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

    // Distribution of users by their attempts-per-day (binned server-side): the
    // week's attempts averaged over 7 days, so how many users average ~2/day,
    // ~5/day, etc. x = attempts/day bin, y = users.
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
        pane.append(h, makeBarList(binHist(data.solveHist, bucketSec), you.solve));
    }

    if (data.reviewHist && data.reviewHist.length) {
        const h = document.createElement('div');
        h.className = 'server-stats-heading';
        h.textContent = getTerm('statsReviewTimeMedian');
        pane.append(h, makeBarList(binHist(data.reviewHist, bucketSec), you.review));
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
