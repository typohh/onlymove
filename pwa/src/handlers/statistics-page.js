// pwa/src/handlers/statistics-page.js
//
// GET /statistics — the community's week as a standalone, directly-linkable,
// server-rendered page (loads without booting the SPA, like /leaderboard). It
// renders the cached server_stats blob (src/server-stats.js: the totals, the
// per-user distributions, the rank histogram and the country list over the
// preceding 7 days), folded here into the display bins the page draws and
// inlined as JSON for the page's own script. The JSON form stays at
// /server-stats.
//
// Per-request identity enters in exactly two ways. The requester's edge geo-IP
// country is stamped into the JSON for the country chart's own bar, which is why
// the response is no-store: it is one cached-row read, so an edge cache would
// save nothing and would hand one viewer's country to the next. The reader's own
// bin in every OTHER distribution arrives in the URL FRAGMENT, written by the
// app's Statistics menu link from its local history at click time
// (communityYouFragment, js/stats-compute.js) — the page holds no history of its
// own. A fragment never reaches the server, and a link shared without one shows
// the plain community data.

import { loadServerStats } from '../server-stats.js';
import { normalizeCountry } from '../leaderboard.js';
import { binHist } from '../../shared/community-bins.js';
import { baseCss, flag } from './page-shell.js';

// Localization for this server-rendered page: resolved on the CLIENT from
// navigator.language like the leaderboard page (the same getLang() prefix match
// and {param} interpolation as the SPA's i18n.js), so one response serves every
// viewer. "OnlyMove" is a brand name and stays untranslated.
const STR = {
    en: {
        title: 'Statistics',
        window: '(last {days} days, updated {ago})',
        agoJustNow: 'just now',
        agoMinutes: '{n}m ago',
        agoHours: '{n}h ago',
        agoDays: '{n}d ago',
        activeUsers: 'Active users: {count}',
        attemptsPerDay: 'Attempts per day: {count}',
        attemptsSolved: 'Attempts: {count} ({rate}% solved)',
        bySizeLine: '{size}×{size}: {attempts} attempts, {users} users',
        usersByAttemptsPerDay: 'Attempts per day (average)',
        solvingTimeMedian: 'Solving time (median)',
        reviewTimeMedian: 'Review time (median)',
        recallTime: 'Recall time',
        ranks: 'Ranks (OnlyMove)',
        countries: 'Countries',
        youLegend: 'You are here',
        empty: 'No activity in the past {days} days.',
    },
    zh: {
        title: '统计',
        window: '（最近 {days} 天，更新于 {ago}）',
        agoJustNow: '刚刚',
        agoMinutes: '{n} 分钟前',
        agoHours: '{n} 小时前',
        agoDays: '{n} 天前',
        activeUsers: '活跃用户：{count}',
        attemptsPerDay: '每日尝试次数：{count}',
        attemptsSolved: '尝试次数：{count}（{rate}% 已解决）',
        bySizeLine: '{size}×{size}：{attempts} 次尝试，{users} 名用户',
        usersByAttemptsPerDay: '每日尝试次数（平均）',
        solvingTimeMedian: '解题时间（中位数）',
        reviewTimeMedian: '复习时间（中位数）',
        recallTime: '回忆时间',
        ranks: '段位分布（OnlyMove）',
        countries: '国家/地区',
        youLegend: '你在这里',
        empty: '最近 {days} 天没有活动。',
    },
    ja: {
        title: '統計',
        window: '（直近 {days} 日間、更新 {ago}）',
        agoJustNow: 'たった今',
        agoMinutes: '{n} 分前',
        agoHours: '{n} 時間前',
        agoDays: '{n} 日前',
        activeUsers: 'アクティブユーザー：{count}',
        attemptsPerDay: '1日あたりの挑戦数：{count}',
        attemptsSolved: '挑戦数：{count}（{rate}% 正解）',
        bySizeLine: '{size}×{size}：挑戦 {attempts} 回、{users} 人',
        usersByAttemptsPerDay: '1日あたりの挑戦数（平均）',
        solvingTimeMedian: '解答時間（中央値）',
        reviewTimeMedian: '復習時間（中央値）',
        recallTime: '想起までの時間',
        ranks: 'ランク分布（OnlyMove）',
        countries: '国・地域',
        youLegend: 'あなたの位置',
        empty: '直近 {days} 日間の活動はありません。',
    },
    ko: {
        title: '통계',
        window: '(최근 {days}일, 업데이트 {ago})',
        agoJustNow: '방금 전',
        agoMinutes: '{n}분 전',
        agoHours: '{n}시간 전',
        agoDays: '{n}일 전',
        activeUsers: '활성 사용자: {count}',
        attemptsPerDay: '하루 시도 횟수: {count}',
        attemptsSolved: '시도: {count} ({rate}% 해결)',
        bySizeLine: '{size}×{size}: 시도 {attempts}회, 사용자 {users}명',
        usersByAttemptsPerDay: '하루 시도 횟수 (평균)',
        solvingTimeMedian: '풀이 시간 (중앙값)',
        reviewTimeMedian: '복습 시간 (중앙값)',
        recallTime: '회상 시간',
        ranks: '랭크 분포 (OnlyMove)',
        countries: '국가·지역',
        youLegend: '내 위치',
        empty: '최근 {days}일간 활동이 없습니다.',
    },
};

export async function handleStatistics(env, request) {
    let data = null;
    try {
        data = await loadServerStats(env);
    } catch (e) {
        console.error('statistics read failed:', e);
    }
    const youCountry = normalizeCountry(request?.cf?.country);
    return new Response(renderPage(buildView(data, youCountry)), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'no-store',
        },
    });
}

// The blob, reduced to what the page draws: the headline figures and one
// {label, value} row list per chart. `key` is the fragment parameter that names
// the reader's own row in that chart (an index; for `rank` the row's label); a
// chart that resolves its own row server-side carries `you` instead. Rows with
// a zero count are dropped at draw time except in the rank ladder, where an
// empty rung is a meaningful gap and is kept.
function buildView(data, youCountry) {
    if (!data || !data.totals || !(data.totals.attempts > 0)) return null;
    const bucketSec = data.bucketSec || 10;
    const attempts = data.totals.attempts || 0;
    const wins = data.totals.wins || 0;
    const dayCount = Array.isArray(data.daily) ? data.daily.length : 0;
    const charts = [];
    if (Array.isArray(data.attemptsPerUserDist) && data.attemptsPerUserDist.length) {
        charts.push({ key: 'apd', title: 'usersByAttemptsPerDay',
                      rows: data.attemptsPerUserDist.map(b => ({ label: b.label, value: b.value })) });
    }
    if (Array.isArray(data.solveHist) && data.solveHist.length) {
        charts.push({ key: 'solve', title: 'solvingTimeMedian', rows: binHist(data.solveHist, bucketSec) });
    }
    if (Array.isArray(data.reviewHist) && data.reviewHist.length) {
        charts.push({ key: 'review', title: 'reviewTimeMedian', rows: binHist(data.reviewHist, bucketSec) });
    }
    if (Array.isArray(data.recallDist) && data.recallDist.some(b => b.value)) {
        charts.push({ key: 'recall', title: 'recallTime',
                      rows: data.recallDist.map(b => ({ label: b.label, value: b.value })) });
    }
    // Ranks arrive ordered weakest→strongest, one entry per active player with a
    // stored rank (computeLeaderboardStats); drawn as-is.
    if (Array.isArray(data.ranks) && data.ranks.length) {
        charts.push({ key: 'rank', title: 'ranks', keepEmpty: true,
                      rows: data.ranks.map(r => ({ label: r.rank, value: r.n })) });
    }
    // Every country, commonest first — folding the tail into one "Other" row
    // would hide exactly the small communities the list is interesting for.
    if (Array.isArray(data.countries) && data.countries.length) {
        const sorted = data.countries.slice().sort((a, b) => b.n - a.n);
        charts.push({
            key: 'country', title: 'countries',
            rows: sorted.map(c => { const f = flag(c.country); return { label: (f ? f + ' ' : '') + c.country, value: c.n }; }),
            you: youCountry ? sorted.findIndex(c => c.country === youCountry) : -1,
        });
    }
    return {
        computedAt: data.computedAt,
        windowDays: data.windowDays || 7,
        users: data.totals.users,
        attempts,
        rate: attempts ? Math.round(wins / attempts * 100) : 0,
        attemptsPerDay: dayCount ? Math.round(attempts / dayCount) : 0,
        bySize: (data.bySize || []).map(s => ({ size: s.size, attempts: s.attempts, users: s.users })),
        charts,
    };
}

function renderPage(view) {
    const json = JSON.stringify(view).replace(/</g, '\\u003c');
    const strJson = JSON.stringify(STR).replace(/</g, '\\u003c');
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Statistics</title>
<meta name="description" content="Community statistics for Only Move: attempts, solvers, rank distribution and served difficulty over the past week.">
<link rel="canonical" href="https://onlymove.net/statistics">
<style>
  ${baseCss(640)}
  ul.totals { list-style: none; padding: 0; margin: 0.75rem 0 .5rem; font-variant-numeric: tabular-nums; }
  ul.totals li { margin-bottom: 0.15rem; }
  ul.totals li.size { padding-left: 1rem; color: var(--ink-muted); }
  .chart h2 { font-size: 1rem; margin: 2rem 0 .6rem; padding-top: 1rem; }
  .bars { display: flex; flex-direction: column; gap: 2px; }
  .bar { display: flex; align-items: center; gap: .5rem; font-variant-numeric: tabular-nums; }
  .bar .label { flex: 0 0 auto; min-width: 4.5rem; }
  .bar .track { flex: 1; height: .8rem; border-radius: 3px; background: var(--surface); }
  .bar .fill { display: block; height: 100%; border-radius: 3px; background: #2e9e5b; }
  .bar .value { flex: 0 0 auto; min-width: 2.5rem; text-align: right; }
  /* The reader's own bin: warm highlight on the fill and a bold label, so "where
     am I" reads at a glance against the green community bars; the legend at the
     bottom names the colour. */
  .bar.you .label, .bar.you .value { font-weight: 600; color: var(--accent); }
  .bar.you .fill { background: var(--accent); }
  .bar.you .track { box-shadow: 0 0 0 1px var(--accent); }
  .legend { display: flex; align-items: center; gap: .4rem; margin-top: 1.3rem; color: var(--ink-muted); font-size: .9rem; }
  .swatch { display: inline-block; width: .9rem; height: .9rem; border-radius: 3px; background: var(--accent); }
  @media (max-width: 420px) {
    body { padding-left: .6rem; padding-right: .6rem; }
  }
</style>
</head>
<body>
<h1><span id="stH1">Statistics</span> <span class="hnote" id="stWindow"></span></h1>
<ul class="totals" id="totals"></ul>
<div id="charts"></div>
<p class="legend" id="legend" hidden><span class="swatch"></span><span id="legendText"></span></p>
<script>
(function () {
  var STR = ${strJson};
  var VIEW = ${json};
  function getLang() {
    var l = navigator.language || 'en';
    if (l.indexOf('zh') === 0) return 'zh';
    if (l.indexOf('ja') === 0) return 'ja';
    if (l.indexOf('ko') === 0) return 'ko';
    return 'en';
  }
  var LANG = getLang();
  function t(key, params) {
    var db = STR[LANG] || STR.en;
    var s = (key in db) ? db[key] : STR.en[key];
    if (params && typeof s === 'string') {
      s = s.replace(/\\{(\\w+)\\}/g, function (m, n) { return (n in params) ? String(params[n]) : m; });
    }
    return s;
  }
  document.documentElement.lang = LANG;
  document.title = 'onlymove — ' + t('title');
  document.getElementById('stH1').textContent = t('title');

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  var charts = document.getElementById('charts');
  if (!VIEW) {
    charts.appendChild(el('p', 'empty', t('empty', { days: 7 })));
    return;
  }

  function ago(iso) {
    var ms = Date.now() - Date.parse(iso);
    if (!isFinite(ms) || ms < 0) return t('agoJustNow');
    var h = ms / 3600000;
    if (h < 1) return t('agoMinutes', { n: Math.round(ms / 60000) });
    if (h < 24) return t('agoHours', { n: Math.round(h) });
    return t('agoDays', { n: Math.round(h / 24) });
  }
  document.getElementById('stWindow').textContent = t('window', { days: VIEW.windowDays, ago: ago(VIEW.computedAt) });

  var totals = document.getElementById('totals');
  totals.appendChild(el('li', null, t('activeUsers', { count: VIEW.users })));
  totals.appendChild(el('li', null, t('attemptsPerDay', { count: VIEW.attemptsPerDay })));
  totals.appendChild(el('li', null, t('attemptsSolved', { count: VIEW.attempts, rate: VIEW.rate })));
  VIEW.bySize.forEach(function (s) {
    totals.appendChild(el('li', 'size', t('bySizeLine', { size: s.size, attempts: s.attempts, users: s.users })));
  });

  // The reader's own rows, from the fragment the app's menu link wrote (an
  // index per chart key; the rank ladder is keyed by the row's label). A chart
  // that resolved its row server-side (the country) carries it in VIEW.
  var frag = new URLSearchParams((location.hash || '').replace(/^#/, ''));
  function youIndex(chart) {
    if (chart.you != null) return chart.you;
    var v = frag.get(chart.key);
    if (v == null) return -1;
    if (chart.key === 'rank') return chart.rows.findIndex(function (r) { return r.label === v; });
    var i = parseInt(v, 10);
    return (i >= 0 && i < chart.rows.length) ? i : -1;
  }

  var anyYou = false;
  VIEW.charts.forEach(function (chart) {
    var hi = youIndex(chart);
    var rows = chart.rows.map(function (r, i) { return { label: r.label, value: r.value, orig: i }; });
    if (!chart.keepEmpty) rows = rows.filter(function (r) { return r.value > 0 || r.orig === hi; });
    if (!rows.length) return;
    var max = Math.max.apply(null, [1].concat(rows.map(function (r) { return r.value; })));
    var section = el('section', 'chart');
    section.appendChild(el('h2', null, t(chart.title)));
    var bars = el('div', 'bars');
    rows.forEach(function (r) {
      var row = el('div', 'bar' + (r.orig === hi ? ' you' : ''));
      if (r.orig === hi) anyYou = true;
      row.appendChild(el('span', 'label', r.label));
      var track = el('span', 'track');
      var fill = el('span', 'fill');
      fill.style.width = (r.value / max * 100).toFixed(1) + '%';
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('span', 'value', String(r.value)));
      bars.appendChild(row);
    });
    section.appendChild(bars);
    charts.appendChild(section);
  });
  if (anyYou) {
    document.getElementById('legendText').textContent = t('youLegend');
    document.getElementById('legend').hidden = false;
  }
})();
</script>
</body>
</html>`;
}
