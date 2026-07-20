// pwa/src/handlers/leaderboard-page.js
//
// GET /leaderboard — a standalone, directly-linkable, server-rendered page (loads
// without booting the tsumego SPA). Computes on read (no cron blob); a short edge
// cache bounds D1 reads under traffic. Lists every player active in the past week
// who has at least RANK_START_MIN solves here (so their onlymove rank is known):
// invite-registered players with their EGD name, everyone else as "Anonymous".
// Players still below the gate are only counted, in a footer tally.
// Two value columns, both in ONLYMOVE RANK labels ("3k", "1d"):
//   * Declared — the range where the player's external evidence (EGD GoR +
//     self-declared server ranks from rank_surveys) lands when back-mapped
//     through the inverse consensus couplings (rangeMin..rangeMax from
//     readLeaderboardRows; single label when the sources agree). Shown for any
//     row with a mappable source, anonymous included.
//   * onlymove — the current rank estimated from their solving here.
// Any header sorts (labels sort by their underlying continuous scalar).

import { readLeaderboardRows, RANK_START_MIN } from '../leaderboard.js';
import { rankScalarToLabel } from '../../shared/rank-math.js';
import { baseCss, ESC_SNIPPET, flag } from './page-shell.js';

const ACTIVE_DAYS = 7;   // the visible board = players active in the past week

// Shown when we can't resolve a country for a row (no GoR/invitation/geo-IP, or a
// hidden profile). The UN flag reads as "nation unspecified" and renders as a real
// regional-indicator glyph like every other flag; swap to '🏴‍☠️' etc. here to taste.
const UNKNOWN_FLAG = '🇺🇳';

export async function handleLeaderboard(env) {
    let rows = [], belowMin = 0;
    try {
        ({ rows, belowMin } = await readLeaderboardRows(env, { activeDays: ACTIVE_DAYS }));
    } catch (e) {
        console.error('leaderboard read failed:', e);
    }
    return new Response(renderPage(rows, belowMin), {
        status: 200,
        headers: {
            'Content-Type': 'text/html; charset=utf-8',
            'Cache-Control': 'public, max-age=60',
        },
    });
}

function renderPage(rows, belowMin) {
    // Per row: `decl` = the external-evidence range (v: midpoint scalar for
    // sorting, t: "3k–1d" label), `est` = the current onlymove estimate (v:
    // internal scalar, t: label). BOTH are UNTIMED onlymove-frame internal scalars
    // (readLeaderboardRows already inverted the EGF/server couplings AND removed the
    // user's time dock), so the two columns speak the same language the client's
    // front-page rank does — no EGF, no time adjustment.
    const toEst = (s) => (Number.isFinite(s) ? { v: s, t: rankScalarToLabel(s) } : null);
    const data = rows.map((r) => {
        let decl = null;
        if (Number.isFinite(r.rangeMin) && Number.isFinite(r.rangeMax)) {
            const lo = rankScalarToLabel(r.rangeMin);
            const hi = rankScalarToLabel(r.rangeMax);
            decl = { v: (r.rangeMin + r.rangeMax) / 2, t: lo === hi ? lo : lo + '–' + hi };
        }
        const row = {
            // Anonymous (unregistered) rows carry anon: true and no name — the client
            // renders a muted "Anonymous" label for them.
            name: r.registered ? (r.name || '—') : null,
            anon: !r.registered,
            flag: flag(r.country) || UNKNOWN_FLAG,
            decl,
            est: toEst(r.currentInternal),
        };
        return row;
    });
    const json = JSON.stringify(data).replace(/</g, '\\u003c');
    // Footer tally of active players still below the list-entry gate (no per-user
    // sample count is shown; this aggregate is the only nod to them).
    const belowNote = belowMin
        ? `<p>${belowMin} more active player${belowMin === 1 ? '' : 's'} ${belowMin === 1 ? 'has' : 'have'} fewer than ${RANK_START_MIN} solves and ${belowMin === 1 ? "isn't" : "aren't"} shown yet.</p>`
        : '';

    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>onlymove — Leaderboard</title>
<style>
  ${baseCss(760)}
  th, td { padding: .5rem .5rem; text-align: right; white-space: nowrap; }
  td:nth-child(2) { text-align: left; width: 99%; }
  th.player { text-align: left; }
  thead th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; font-weight: 600;
             border-bottom: 1px solid; }
  /* vertical divider separating the declared range from the onlymove estimate */
  th.om-start, td.om-start { border-left: 1px solid color-mix(in srgb, currentColor 15%, transparent); }
  thead th[data-sort] { cursor: pointer; user-select: none; -webkit-user-select: none; }
  thead th[data-sort]:hover { opacity: .85; }
  thead th.sorted { opacity: .95; }
  td.rank { opacity: .5; font-variant-numeric: tabular-nums; }
  td.val { font-variant-numeric: tabular-nums; font-weight: 600; }
  td.decl { opacity: .7; font-variant-numeric: tabular-nums; }
  .name { overflow: hidden; text-overflow: ellipsis; max-width: 16rem; display: inline-block; vertical-align: bottom; }
  /* Unregistered players — muted so named (registered) rows stand out. */
  .name.anon { opacity: .5; font-style: italic; }
  .empty { opacity: .5; padding: 2rem 0; text-align: center; }
  footer { margin-top: 1.5rem; opacity: .55; font-size: .82rem; }
  footer p { margin: .35rem 0; }
  footer a { color: inherit; }
  /* "Hide my profile" control — shown only to visitors with a userId. */
  .hide-profile { margin-top: 1.1rem; padding-top: 1rem; border-top: 1px solid color-mix(in srgb, currentColor 12%, transparent); }
  .hide-profile label { display: inline-flex; align-items: flex-start; gap: .5rem; cursor: pointer; }
  .hide-profile input { margin: .15rem 0 0; flex: none; }
  .hide-profile input:disabled + span { opacity: .5; }
  .hide-status { display: block; margin-top: .4rem; min-height: 1.2em; opacity: .85; }
  @media (max-width: 420px) {
    body { padding-left: .6rem; padding-right: .6rem; }
  }
</style>
</head>
<body>
<h1>Leaderboard</h1>
<p class="sub">Every player active in the past week with at least ${RANK_START_MIN} solves, ordered by their onlymove rank estimate. "Declared" is where the player's external ranks land on the onlymove scale.</p>
<table>
  <thead>
    <tr>
      <th>#</th>
      <th class="player" data-sort="name">Player</th>
      <th data-sort="decl">Declared</th>
      <th class="om-start" data-sort="est">onlymove</th>
    </tr>
  </thead>
  <tbody id="rows"></tbody>
</table>
<footer>
  ${belowNote}
  <p>Want your name shown instead of "Anonymous"? Registration is by invite — ask in <a href="https://discord.gg/ZU7jJNnkbB" target="_blank" rel="noopener">Discord</a>.</p>
  <div id="hideProfile" class="hide-profile" hidden>
    <label>
      <input type="checkbox" id="hideChk" disabled>
      <span>Hide my profile — withhold my name, country, and declared ranks from this board. My onlymove rank still shows, but anonymously.</span>
    </label>
    <span id="hideStatus" class="hide-status" role="status" aria-live="polite"></span>
  </div>
</footer>
<script>
(function () {
  var DATA = ${json};
  var sortKey = 'est'; // 'name' | 'decl' | 'est'
  var sortDir = -1;    // 1 ascending, -1 descending
  var tbody = document.getElementById('rows');

  ${ESC_SNIPPET}

  function fmt(cell) { return cell ? esc(cell.t) : '—'; }

  function keyOf(th) { return th.getAttribute('data-sort'); }   // 'name' | 'decl' | 'est'
  function nameOf(r) { return r.anon ? 'Anonymous' : (r.name || ''); }
  function sortValue(r) {
    if (sortKey === 'name') return nameOf(r);
    if (sortKey === 'decl') return r.decl ? r.decl.v : null;
    return r.est ? r.est.v : null;              // 'est' — the current estimate
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

  var headers = Array.prototype.slice.call(document.querySelectorAll('th[data-sort]'));
  headers.forEach(function (th) {
    th.setAttribute('data-label', th.textContent);
    th.addEventListener('click', function () {
      var key = keyOf(th);
      if (key === sortKey) { sortDir = -sortDir; }               // same column -> flip
      else { sortKey = key; sortDir = (key === 'name') ? 1 : -1; }  // names A→Z, ranks high→low
      render();
    });
  });
  function updateHeaders() {
    headers.forEach(function (th) {
      var active = keyOf(th) === sortKey;
      th.classList.toggle('sorted', active);
      th.textContent = th.getAttribute('data-label') + (active ? (sortDir === 1 ? ' ▲' : ' ▼') : '');
    });
  }

  function render() {
    updateHeaders();
    if (!DATA.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty">No players yet.</td></tr>'; return; }
    var sorted = DATA.slice().sort(cmp);
    tbody.innerHTML = sorted.map(function (r, i) {
      var estCell = '<td class="val om-start">' + fmt(r.est) + '</td>';
      return '<tr><td class="rank">' + (i + 1) + '</td>' +
        '<td>' + (r.flag ? r.flag + ' ' : '') +
        '<span class="name' + (r.anon ? ' anon' : '') + '">' + esc(nameOf(r)) + '</span></td>' +
        '<td class="decl">' + fmt(r.decl) + '</td>' +
        estCell + '</tr>';
    }).join('');
  }

  render();
})();
</script>
<script>
// "Hide my profile" control. This page doesn't boot the SPA, but it's same-origin
// with it, so the account's userId is readable from localStorage. Shown only when
// a userId exists (a visitor who never used the app has no profile to hide).
(function () {
  var uid = null;
  try { uid = localStorage.getItem('userId'); } catch (e) {}
  if (!uid || !/^[A-Za-z0-9_-]{1,40}$/.test(uid)) return;

  var box = document.getElementById('hideProfile');
  var chk = document.getElementById('hideChk');
  var status = document.getElementById('hideStatus');
  box.hidden = false;

  // Reflect the current server state before enabling the checkbox.
  fetch('/profile/hide?userId=' + encodeURIComponent(uid))
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (j) { if (j) chk.checked = !!j.hidden; })
    .catch(function () {})
    .then(function () { chk.disabled = false; });

  chk.addEventListener('change', function () {
    var want = chk.checked;
    chk.disabled = true;
    status.textContent = 'Saving…';
    fetch('/profile/hide', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userId: uid, hidden: want }),
    })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(); })
      .then(function (j) {
        chk.checked = !!j.hidden;
        status.textContent = (j.hidden
          ? 'Your profile is hidden.'
          : 'Your profile is visible.') + ' The board updates within a minute.';
      })
      .catch(function () {
        chk.checked = !want;   // revert on failure
        status.textContent = 'Could not save — please try again.';
      })
      .then(function () { chk.disabled = false; });
  });
})();
</script>
</body>
</html>`;
}
