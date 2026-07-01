// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Each row shows current / this-week / last-week
// and a top-level toggle switches all three between absolute (estimated EGF rank) and
// relative (estimated − GoR). Both modes are inlined so the toggle is instant.
// See docs/leaderboard_plan.md.

import { readLeaderboardRows } from '../leaderboard.js';
import { rankScalarToLabel } from '../../shared/rank-math.js';

export async function handleLeaderboard(env) {
    let rows = [];
    try {
        rows = await readLeaderboardRows(env);
    } catch (e) {
        console.error('leaderboard read failed:', e);
    }
    return new Response(renderPage(rows), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'public, max-age=60',
        },
    });
}

const fmtAbs = (v) => (Number.isFinite(v) ? rankScalarToLabel(v) : '—');
const fmtRel = (v) => (Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) : '—');
const flag = (cc) => (typeof cc === 'string' && cc.length === 2)
    ? cc.toUpperCase().replace(/./g, (c) => String.fromCodePoint(0x1F1E6 + c.charCodeAt(0) - 65)) : '';

function renderPage(rows) {
    // Precompute display strings server-side (via the shared label fn); the client
    // script just sorts by the numeric key and paints the strings for the active mode.
    const data = rows.map((r) => ({
        name: r.name || '—',
        flag: flag(r.country),
        gor: Number.isFinite(r.gor) ? r.gor : '—',
        // absolute (higher = stronger) and relative (higher = more over-performance)
        sortAbs: Number.isFinite(r.currentAbs) ? r.currentAbs : -1e9,
        sortRel: Number.isFinite(r.currentRel) ? r.currentRel : -1e9,
        abs: [fmtAbs(r.currentAbs), fmtAbs(r.thisWeekAbs), fmtAbs(r.prevWeekAbs)],
        rel: [fmtRel(r.currentRel), fmtRel(r.thisWeekRel), fmtRel(r.prevWeekRel)],
    }));
    const json = JSON.stringify(data).replace(/</g, '\\u003c');

    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — EGD leaderboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 1.5rem 1rem 3rem;
         max-width: 760px; margin-inline: auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
  p.sub { margin: 0 0 1.25rem; opacity: .6; font-size: .85rem; }
  .toggle { display: inline-flex; border: 1px solid currentColor; border-radius: 999px; overflow: hidden;
            margin-bottom: 1rem; opacity: .9; }
  .toggle button { font: inherit; padding: .35rem 1rem; border: 0; background: transparent; color: inherit;
                   cursor: pointer; }
  .toggle button[aria-pressed="true"] { background: currentColor; }
  .toggle button[aria-pressed="true"] span { color: Canvas; mix-blend-mode: difference; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: .5rem .4rem; text-align: right; white-space: nowrap; }
  th:nth-child(2), td:nth-child(2) { text-align: left; width: 99%; }
  th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; font-weight: 600;
       border-bottom: 1px solid; }
  tbody tr:nth-child(even) { background: color-mix(in srgb, currentColor 5%, transparent); }
  td.rank { opacity: .5; font-variant-numeric: tabular-nums; }
  td.val { font-variant-numeric: tabular-nums; font-weight: 600; }
  td.gor { opacity: .7; font-variant-numeric: tabular-nums; }
  .name { overflow: hidden; text-overflow: ellipsis; max-width: 16rem; display: inline-block; vertical-align: bottom; }
  .empty { opacity: .5; padding: 2rem 0; text-align: center; }
</style>
</head>
<body>
<h1>EGD leaderboard</h1>
<p class="sub">Estimated onlymove rank vs European Go Database GoR. Rank starts after 100 solved problems.</p>
<div class="toggle" role="group" aria-label="mode">
  <button id="mAbs" aria-pressed="true"><span>Absolute</span></button>
  <button id="mRel" aria-pressed="false"><span>Relative</span></button>
</div>
<table>
  <thead><tr><th>#</th><th>Player</th><th>GoR</th><th>Current</th><th>This week</th><th>Last week</th></tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
(function () {
  var DATA = ${json};
  var mode = 'abs';
  var tbody = document.getElementById('rows');
  function render() {
    var key = mode === 'abs' ? 'sortAbs' : 'sortRel';
    var sorted = DATA.slice().sort(function (a, b) { return b[key] - a[key]; });
    if (!sorted.length) { tbody.innerHTML = '<tr><td colspan="6" class="empty">No ranked players yet.</td></tr>'; return; }
    tbody.innerHTML = sorted.map(function (r, i) {
      var cells = r[mode];
      return '<tr><td class="rank">' + (i + 1) + '</td>' +
        '<td>' + (r.flag ? r.flag + ' ' : '') + '<span class="name">' + esc(r.name) + '</span></td>' +
        '<td class="gor">' + r.gor + '</td>' +
        '<td class="val">' + cells[0] + '</td>' +
        '<td class="val">' + cells[1] + '</td>' +
        '<td class="val">' + cells[2] + '</td></tr>';
    }).join('');
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function setMode(m) {
    mode = m;
    document.getElementById('mAbs').setAttribute('aria-pressed', m === 'abs');
    document.getElementById('mRel').setAttribute('aria-pressed', m === 'rel');
    render();
  }
  document.getElementById('mAbs').onclick = function () { setMode('abs'); };
  document.getElementById('mRel').onclick = function () { setMode('rel'); };
  render();
})();
</script>
</body>
</html>`;
}
