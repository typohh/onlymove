// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Two toggles drive one estimate column
// (mobile-friendly): a timeframe toggle (current / this-week peak / last-week peak)
// and a mode toggle (absolute = estimated GoR / relative = estimated − GoR). The GoR
// column follows the timeframe — current GoR for current/this-week, the frozen
// last-week GoR for last-week peak. All values are inlined so switching is instant,
// and any header sorts. See docs/leaderboard_plan.md.

import { readLeaderboardRows } from '../leaderboard.js';

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

// The estimate is an EGF rank scalar; display it as GoR — the inverse of the
// Worker's gorToRankScalar (1k=2000, 1d=2100, ~100 GoR/stone). Relative mode
// shows the GoR gap (estimated GoR − actual GoR) = exactly 100× the scalar diff.
const scalarToGor = (v) => Math.round(2000 + 100 * v);
// Numeric values the client both sorts by AND formats. Absolute = estimated GoR;
// relative = the GoR gap (estimated − actual GoR) = 100× the scalar diff. null
// where a timeframe has no data (an idle week) → the client renders "—" and sorts
// it last regardless of direction.
const numAbs = (v) => (Number.isFinite(v) ? scalarToGor(v) : null);
const numRel = (v) => (Number.isFinite(v) ? Math.round(100 * v) : null);
const flag = (cc) => (typeof cc === 'string' && cc.length === 2)
    ? cc.toUpperCase().replace(/./g, (c) => String.fromCodePoint(0x1F1E6 + c.charCodeAt(0) - 65)) : '';

function renderPage(rows) {
    // Per timeframe [current, thisWeekPeak, lastWeekPeak]: the GoR to show, the
    // absolute estimate (GoR) and the relative estimate (GoR gap). The client picks one
    // timeframe + one mode and paints a single estimate column — narrow enough for
    // mobile, and any header can sort. The GoR follows the timeframe: current GoR for
    // current/this-week, the frozen last-week GoR for last-week peak.
    const data = rows.map((r) => {
        const curGor = Number.isFinite(r.gor) ? r.gor : null;
        const pwGor = Number.isFinite(r.prevWeekGor) ? r.prevWeekGor : null;
        return {
            name: r.name || '—',
            flag: flag(r.country),
            gors: [curGor, curGor, pwGor],
            abs: [numAbs(r.currentAbs), numAbs(r.thisWeekAbs), numAbs(r.prevWeekAbs)],
            rel: [numRel(r.currentRel), numRel(r.thisWeekRel), numRel(r.prevWeekRel)],
        };
    });
    const json = JSON.stringify(data).replace(/</g, '\\u003c');

    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Leaderboard</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 1.5rem 1rem 3rem;
         max-width: 760px; margin-inline: auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  p.sub { margin: 0 0 1rem; opacity: .6; font-size: .85rem; }
  .toggles { display: flex; flex-wrap: wrap; gap: .5rem .75rem; margin-bottom: 1rem; }
  .toggle { display: inline-flex; border: 1px solid currentColor; border-radius: 999px; overflow: hidden; }
  .toggle button { font: inherit; padding: .35rem .8rem; border: 0; background: transparent; color: currentColor;
                   cursor: pointer; opacity: .6; white-space: nowrap; }
  .toggle button[aria-pressed="true"] { background: currentColor; opacity: 1; }
  /* Selected label = the page background color (Canvas) so it always contrasts the
     currentColor fill, in dark AND light mode. No mix-blend-mode: difference — it
     resolved white-on-white on the white fill in dark mode. */
  .toggle button[aria-pressed="true"] span { color: Canvas; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: .5rem .5rem; text-align: right; white-space: nowrap; }
  td:nth-child(2) { text-align: left; width: 99%; }
  th.player { text-align: left; }
  thead th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; font-weight: 600;
             border-bottom: 1px solid; }
  /* vertical divider separating the EGD GoR from the onlymove estimate */
  th.om-start, td.om-start { border-left: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  thead th[data-sort] { cursor: pointer; user-select: none; -webkit-user-select: none; }
  thead th[data-sort]:hover { opacity: .85; }
  thead th.sorted { opacity: .95; }
  tbody tr:nth-child(even) { background: color-mix(in srgb, currentColor 5%, transparent); }
  td.rank { opacity: .5; font-variant-numeric: tabular-nums; }
  td.val { font-variant-numeric: tabular-nums; font-weight: 600; }
  td.gor { opacity: .7; font-variant-numeric: tabular-nums; }
  .name { overflow: hidden; text-overflow: ellipsis; max-width: 16rem; display: inline-block; vertical-align: bottom; }
  .empty { opacity: .5; padding: 2rem 0; text-align: center; }
  footer { margin-top: 1.5rem; opacity: .55; font-size: .82rem; }
  footer p { margin: .35rem 0; }
  footer a { color: inherit; }
  @media (max-width: 420px) {
    body { padding-left: .6rem; padding-right: .6rem; }
    .toggle button { padding: .35rem .6rem; font-size: .92rem; }
  }
</style>
</head>
<body>
<h1>Leaderboard</h1>
<p class="sub">onlymove-estimated GoR vs European Go Database GoR. A player needs 100 attempts to be included.</p>
<div class="toggles">
  <div class="toggle" role="group" aria-label="timeframe">
    <button id="tf0" aria-pressed="true"><span>Current</span></button>
    <button id="tf1" aria-pressed="false"><span>This week peak</span></button>
    <button id="tf2" aria-pressed="false"><span>Last week peak</span></button>
  </div>
  <div class="toggle" role="group" aria-label="mode">
    <button id="mAbs" aria-pressed="true"><span>Absolute</span></button>
    <button id="mRel" aria-pressed="false"><span>Relative</span></button>
  </div>
</div>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th class="player" data-sort="name">Player</th>
      <th data-sort="gor">GoR</th>
      <th class="om-start" data-sort="est">onlymove</th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
<footer>
  <p>Invite only — ask in <a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a> if interested.</p>
</footer>
<script>
(function () {
  var DATA = ${json};
  var mode = 'abs';    // 'abs' | 'rel'
  var tf = 0;          // timeframe: 0 current, 1 this-week peak, 2 last-week peak
  var sortKey = 'est'; // 'name' | 'gor' | 'est'
  var sortDir = -1;    // 1 ascending, -1 descending
  var tbody = document.getElementById('rows');

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }

  // Numeric -> display string. Absolute = plain GoR; relative = signed GoR gap.
  function fmtCell(v) {
    if (v === null || v === undefined) return '—';
    return mode === 'rel' ? (v >= 0 ? '+' : '−') + Math.abs(v) : String(v);
  }
  function fmtGor(v) { return (v === null || v === undefined) ? '—' : String(v); }

  function keyOf(th) { return th.getAttribute('data-sort'); }   // 'name' | 'gor' | 'est'
  function sortValue(r) {
    if (sortKey === 'name') return r.name || '';
    if (sortKey === 'gor') return r.gors[tf];
    return r[mode][tf];                       // 'est' — active mode's value for the timeframe
  }
  function cmp(a, b) {
    var va = sortValue(a), vb = sortValue(b);
    var na = (va === null || va === undefined), nb = (vb === null || vb === undefined);
    if (na && nb) return 0;
    if (na) return 1;                         // missing values always sort to the bottom
    if (nb) return -1;
    if (sortKey === 'name') return sortDir * String(va).localeCompare(String(vb));
    return sortDir * (va - vb);
  }

  // The 'onlymove' header reflects the mode: an estimated GoR, or the gap vs GoR.
  function headLabel(th) {
    if (keyOf(th) === 'est') return mode === 'abs' ? 'onlymove' : 'vs GoR';
    return th.getAttribute('data-label');
  }
  var headers = Array.prototype.slice.call(document.querySelectorAll('th[data-sort]'));
  headers.forEach(function (th) {
    th.setAttribute('data-label', th.textContent);
    th.addEventListener('click', function () {
      var key = keyOf(th);
      if (key === sortKey) { sortDir = -sortDir; }               // same column -> flip
      else { sortKey = key; sortDir = (key === 'name') ? 1 : -1; }  // names A→Z, numbers high→low
      render();
    });
  });
  function updateHeaders() {
    headers.forEach(function (th) {
      var active = keyOf(th) === sortKey;
      th.classList.toggle('sorted', active);
      th.textContent = headLabel(th) + (active ? (sortDir === 1 ? ' ▲' : ' ▼') : '');
    });
  }

  function render() {
    updateHeaders();
    if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">No ranked players yet.</td></tr>'; return; }
    var sorted = DATA.slice().sort(cmp);
    tbody.innerHTML = sorted.map(function (r, i) {
      return '<tr><td class="rank">' + (i + 1) + '</td>' +
        '<td>' + (r.flag ? r.flag + ' ' : '') + '<span class="name">' + esc(r.name) + '</span></td>' +
        '<td class="gor">' + fmtGor(r.gors[tf]) + '</td>' +
        '<td class="val om-start">' + fmtCell(r[mode][tf]) + '</td></tr>';
    }).join('');
  }

  function setTf(i) {
    tf = i;
    for (var j = 0; j < 3; j++) document.getElementById('tf' + j).setAttribute('aria-pressed', j === i);
    render();
  }
  function setMode(m) {
    mode = m;
    document.getElementById('mAbs').setAttribute('aria-pressed', m === 'abs');
    document.getElementById('mRel').setAttribute('aria-pressed', m === 'rel');
    render();
  }
  for (var j = 0; j < 3; j++) (function (k) {
    document.getElementById('tf' + k).onclick = function () { setTf(k); };
  })(j);
  document.getElementById('mAbs').onclick = function () { setMode('abs'); };
  document.getElementById('mRel').onclick = function () { setMode('rel'); };
  render();
})();
</script>
</body>
</html>`;
}
