// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Lists EVERY player active in the past week:
// invite-registered players with their EGD name/GoR, everyone else as "Anonymous"
// (rank estimate only, no GoR, so the relative mode shows "—" for them). Two
// toggles drive one estimate column (mobile-friendly): a timeframe toggle
// (current / this-week peak / last-week peak) and a mode toggle (absolute =
// estimated GoR / relative = estimated − GoR). The GoR column follows the
// timeframe — current GoR for current/this-week, the frozen last-week GoR for
// last-week peak. All values are inlined so switching is instant, and any header
// sorts.

import { readLeaderboardRows, RANK_START_MIN } from '../leaderboard.js';
import { baseCss, ESC_SNIPPET, numAbs, numRel, flag } from './page-shell.js';

const ACTIVE_DAYS = 7;   // the visible board = players active in the past week

export async function handleLeaderboard(env) {
    let rows = [];
    try {
        rows = await readLeaderboardRows(env, { activeDays: ACTIVE_DAYS });
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

// numAbs/numRel/flag come from page-shell.js (shared with the /tournament page's
// embedded standings). Absolute = estimated GoR; relative = the GoR gap
// (estimated − actual GoR) = 100× the scalar diff.

function renderPage(rows) {
    // Per timeframe [current, thisWeekPeak, lastWeekPeak]: the GoR to show, the
    // absolute estimate (GoR) and the relative estimate (GoR gap). The client picks one
    // timeframe + one mode and paints a single estimate column — narrow enough for
    // mobile, and any header can sort. The GoR follows the timeframe: current GoR for
    // current/this-week, the frozen last-week GoR for last-week peak.
    const data = rows.map((r) => {
        const curGor = Number.isFinite(r.gor) ? r.gor : null;
        const pwGor = Number.isFinite(r.prevWeekGor) ? r.prevWeekGor : null;
        const row = {
            // Anonymous (unregistered) rows carry anon: true and no name — the client
            // renders a muted "Anonymous" label for them.
            name: r.registered ? (r.name || '—') : null,
            anon: !r.registered,
            flag: flag(r.country),
            gors: [curGor, curGor, pwGor],
            abs: [numAbs(r.currentAbs), numAbs(r.thisWeekAbs), numAbs(r.prevWeekAbs)],
            rel: [numRel(r.currentRel), numRel(r.thisWeekRel), numRel(r.prevWeekRel)],
        };
        // Not yet ranked: the onlymove column shows valid-attempts/threshold progress
        // instead of an estimate, so it's obvious they haven't qualified. abs/rel are
        // all null → this row sorts last in the estimate sort (missing values do).
        // usableCount is the VALID (still-in-corpus) attempt count — the exact set the
        // rank fit needs — so it reaches RANK_START_MIN only when a rank is ready (never
        // a "60/60" without one). The Math.min is a defensive guard against ">60/60".
        if (!r.ranked) row.progress = Math.min(r.usableCount || 0, RANK_START_MIN) + '/' + RANK_START_MIN;
        return row;
    });
    const json = JSON.stringify(data).replace(/</g, '\\u003c');

    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Leaderboard</title>
<style>
  ${baseCss(760)}
  .toggles { display: flex; flex-wrap: wrap; gap: .5rem .75rem; margin-bottom: 1rem; }
  .toggle { display: inline-flex; border: 1px solid currentColor; border-radius: 999px; overflow: hidden; }
  .toggle button { font: inherit; padding: .35rem .8rem; border: 0; background: transparent; color: currentColor;
                   cursor: pointer; opacity: .6; white-space: nowrap; }
  .toggle button[aria-pressed="true"] { background: currentColor; opacity: 1; }
  /* Selected label = the page background color (Canvas) so it always contrasts the
     currentColor fill, in dark AND light mode. No mix-blend-mode: difference — it
     resolved white-on-white on the white fill in dark mode. */
  .toggle button[aria-pressed="true"] span { color: Canvas; }
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
  td.rank { opacity: .5; font-variant-numeric: tabular-nums; }
  td.val { font-variant-numeric: tabular-nums; font-weight: 600; }
  /* Not-yet-ranked progress (attempts/threshold) — muted so it reads as "pending". */
  td.val.progress { font-weight: 400; opacity: .5; }
  td.gor { opacity: .7; font-variant-numeric: tabular-nums; }
  .name { overflow: hidden; text-overflow: ellipsis; max-width: 16rem; display: inline-block; vertical-align: bottom; }
  /* Unregistered players — muted so named (registered) rows stand out. */
  .name.anon { opacity: .5; font-style: italic; }
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
<p class="sub">Every player active in the past week, ranked by onlymove-estimated GoR. Registered players show their name and European Go Database GoR; everyone else is anonymous. Newly registered players show their attempts toward a rank (${RANK_START_MIN} needed) until they qualify.</p>
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
  <p>Want your name and GoR shown instead of "Anonymous"? Registration is by invite — ask in <a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a>.</p>
</footer>
<script>
(function () {
  var DATA = ${json};
  var mode = 'abs';    // 'abs' | 'rel'
  var tf = 0;          // timeframe: 0 current, 1 this-week peak, 2 last-week peak
  var sortKey = 'est'; // 'name' | 'gor' | 'est'
  var sortDir = -1;    // 1 ascending, -1 descending
  var tbody = document.getElementById('rows');

  ${ESC_SNIPPET}

  // Numeric -> display string. Absolute = plain GoR; relative = signed GoR gap.
  function fmtCell(v) {
    if (v === null || v === undefined) return '—';
    return mode === 'rel' ? (v >= 0 ? '+' : '−') + Math.abs(v) : String(v);
  }
  function fmtGor(v) { return (v === null || v === undefined) ? '—' : String(v); }

  function keyOf(th) { return th.getAttribute('data-sort'); }   // 'name' | 'gor' | 'est'
  function nameOf(r) { return r.anon ? 'Anonymous' : (r.name || ''); }
  function sortValue(r) {
    if (sortKey === 'name') return nameOf(r);
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
    if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">No players yet.</td></tr>'; return; }
    var sorted = DATA.slice().sort(cmp);
    tbody.innerHTML = sorted.map(function (r, i) {
      // Not-yet-ranked rows show attempts/threshold progress (mode/timeframe-independent)
      // in the estimate column instead of a rank; muted via .progress.
      var estCell = r.progress
        ? '<td class="val om-start progress">' + esc(r.progress) + '</td>'
        : '<td class="val om-start">' + fmtCell(r[mode][tf]) + '</td>';
      return '<tr><td class="rank">' + (i + 1) + '</td>' +
        '<td>' + (r.flag ? r.flag + ' ' : '') +
        '<span class="name' + (r.anon ? ' anon' : '') + '">' + esc(nameOf(r)) + '</span></td>' +
        '<td class="gor">' + fmtGor(r.gors[tf]) + '</td>' +
        estCell + '</tr>';
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
