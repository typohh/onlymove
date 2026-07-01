// pwa/src/handlers/admin-invite.js
//
// Admin invite generator — the web equivalent of generate_invite.py. A standalone
// page at GET /admin/invite lets an admin search EGD (by name or PIN) and mint an
// invitation link (an `invitation` row + the /register?t= URL).
//
//   GET  /admin/invite                 -> the admin page
//   POST /admin/check {code}           -> is this account code an admin? (page gate)
//   POST /admin/create-invite {code, pin, name, gor, country} -> insert invitation
//
// Auth is the admin's own account code (user_id), read automatically from the
// browser's localStorage (same key the PWA uses) and verified against
// env.ADMIN_USER_IDS (a wrangler secret, comma-separated — see wrangler.toml). It
// FAILS CLOSED when the secret is unset. The POST endpoints are also per-IP
// rate-limited (RL_WRITE) in the dispatcher.
//
// EGD is fetched CLIENT-SIDE (the browser can — EGD sends CORS `*`), because the
// deployed Worker cannot: EGD sits behind Cloudflare and a Worker→Cloudflare-zone
// fetch is bot-blocked. So the page does the search itself and hands the chosen
// player's fields to /admin/create-invite; the Worker only writes the row.

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';

// The account code is the admin key. Compare against the configured admin id(s);
// fail closed if the secret is unset so the page can't be an open invite faucet.
function isAdmin(code, env) {
    const cfg = (env.ADMIN_USER_IDS || '').trim();
    if (!cfg) return false;
    if (typeof code !== 'string' || !isValidUserId(code)) return false;
    return cfg.split(',').map((s) => s.trim()).filter(Boolean).includes(code);
}

const TOKEN_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
function generateToken() {
    const bytes = crypto.getRandomValues(new Uint8Array(8));
    let s = '';
    for (let i = 0; i < 8; i++) s += TOKEN_ALPHABET[bytes[i] % TOKEN_ALPHABET.length];
    return s;
}

// Insert an invitation with a fresh token; retry on the (astronomically unlikely)
// primary-key collision. Mirrors generate_invite.py's row shape.
async function insertInvitation(env, player) {
    const nowIso = new Date().toISOString();
    for (let attempt = 0; ; attempt++) {
        const token = generateToken();
        try {
            await env.DB.prepare(
                `INSERT INTO invitation
                    (invite_token, egd_pin, egd_name, country_code, gor, created_at,
                     claimed_by_user_id, claimed_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL)`
            ).bind(token, player.pin, player.name, player.country, player.gor, nowIso).run();
            return token;
        } catch (e) {
            if (attempt >= 4) throw e;
        }
    }
}

export async function handleAdminPage(env) {
    return new Response(renderAdminPage(), {
        status: 200,
        headers: { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' },
    });
}

// Page gate: does this browser's account code have admin rights?
export async function handleAdminCheck(request, env) {
    try {
        const { code } = await request.json();
        if (!isAdmin(code, env)) return jsonResponse({ ok: false, error: 'unauthorized' }, 403);
        return jsonResponse({ ok: true });
    } catch (e) {
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}

export async function handleAdminCreate(request, env) {
    try {
        const { code, pin, name, gor, country } = await request.json();
        if (!isAdmin(code, env)) return jsonResponse({ ok: false, error: 'unauthorized' }, 403);
        if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);

        // The player fields come from the admin-authed client's own EGD fetch (the
        // Worker can't reach EGD). Validate shape; trust the values.
        const pinStr = String(pin || '');
        if (!/^\d+$/.test(pinStr)) return jsonResponse({ ok: false, error: 'bad_pin' }, 400);
        const egdName = typeof name === 'string' ? name.trim().slice(0, 100) : '';
        if (!egdName) return jsonResponse({ ok: false, error: 'bad_name' }, 400);
        const gorInt = Number.isFinite(Number(gor)) ? Math.round(Number(gor)) : null;
        if (gorInt === null) return jsonResponse({ ok: false, error: 'bad_gor' }, 400);
        const cc = (typeof country === 'string' ? country : '').slice(0, 2).toUpperCase();

        // 1:1 binding: no point inviting a PIN that already has a registered account
        // (the claim would be rejected anyway). Surface it as a clean error.
        const existing = await env.DB.prepare(
            `SELECT user_id FROM leaderboard WHERE egd_pin = ?1`
        ).bind(pinStr).first();
        if (existing) return jsonResponse({ ok: false, error: 'already_registered' }, 409);

        const token = await insertInvitation(env, { pin: pinStr, name: egdName, country: cc, gor: gorInt });
        return jsonResponse({
            ok: true,
            inviteToken: token,
            link: `https://onlymove.net/register?t=${token}`,
            egdName, gor: gorInt, country: cc,
        });
    } catch (e) {
        console.error('admin create failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}

function renderAdminPage() {
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>onlymove — admin tools</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 1.5rem 1rem 3rem;
         max-width: 720px; margin-inline: auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  p.sub { margin: 0 0 1.25rem; opacity: .6; font-size: .85rem; }
  #gate { margin: 2rem 0; font-size: .95rem; }
  #gate.err { color: #c0392b; }
  #tools { display: none; }
  .searchrow { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .5rem; }
  #q { flex: 1 1 12rem; font: inherit; padding: .45rem .6rem; border: 1px solid; border-radius: 8px;
       background: transparent; color: inherit; box-sizing: border-box; }
  button { font: inherit; padding: .45rem .9rem; border: 1px solid currentColor; border-radius: 8px;
           background: transparent; color: inherit; cursor: pointer; white-space: nowrap; }
  button:disabled { opacity: .5; cursor: default; }
  #msg { min-height: 1.4em; margin: .5rem 0 1rem; font-size: .9rem; opacity: .8; }
  #msg.err { color: #c0392b; opacity: 1; }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: .45rem .4rem; text-align: right; white-space: nowrap; }
  th.l, td.l { text-align: left; width: 99%; overflow: hidden; text-overflow: ellipsis; max-width: 12rem; }
  th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; border-bottom: 1px solid; }
  tbody tr:nth-child(even) { background: color-mix(in srgb, currentColor 5%, transparent); }
  td button { padding: .3rem .7rem; font-size: .85rem; }
  #out { display: none; margin-top: 1.5rem; padding: 1rem; border: 1px solid;
         border-radius: 10px; background: color-mix(in srgb, currentColor 5%, transparent); }
  #out .linkrow { display: flex; gap: .5rem; margin-top: .5rem; }
  #out input { flex: 1; font: inherit; padding: .45rem .6rem; border: 1px solid; border-radius: 8px;
               background: transparent; color: inherit; }
</style>
</head>
<body>
<h1>Generate invitation</h1>
<p class="sub">Admin tools — the web equivalent of generate_invite.py. Search EGD by name or PIN and mint an invite link.</p>
<div id="gate">Checking access…</div>
<div id="tools">
  <div class="searchrow">
    <input type="text" id="q" placeholder="EGD name or PIN — e.g. Hemming, Henry Hemming, or 16025130" autocomplete="off">
    <button id="searchBtn">Search</button>
  </div>
  <div id="msg"></div>
  <div id="results"></div>
  <div id="out"></div>
</div>
<script>
(function () {
  var EGD = 'https://www.europeangodatabase.eu/EGD/';
  var gate = document.getElementById('gate');
  var tools = document.getElementById('tools');
  var qEl, msg, results, out;
  var currentPlayers = [];

  // The admin key is THIS browser's account code (the PWA's user_id) — read it
  // automatically, no prompt. Same origin => same localStorage as the app.
  var code = '';
  try { code = localStorage.getItem('userId') || ''; } catch (e) {}

  var errMap = {
    unauthorized: 'This account is not authorized for admin tools.',
    already_registered: 'That EGD player is already registered.',
    bad_pin: 'Invalid PIN.', bad_name: 'Missing name.', bad_gor: 'Missing GoR.'
  };
  function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function setMsg(t, err) { msg.textContent = t || ''; msg.className = err ? 'err' : ''; }
  function showGate(t) { gate.textContent = t; gate.className = 'err'; tools.style.display = 'none'; }

  async function post(url, body) {
    var res;
    try { res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); }
    catch (e) { return { status: 0, data: null }; }
    var data = null; try { data = await res.json(); } catch (e) {}
    return { status: res.status, data: data };
  }

  function mapPlayer(p, pinFallback) {
    var gor = parseInt(p.Gor, 10);
    return {
      pin: p.Pin_Player || pinFallback || '',
      name: ((p.Name || '') + ' ' + (p.Last_Name || '')).trim(),
      country: p.Country_Code || '', rank: p.Grade || '',
      gor: isNaN(gor) ? null : gor
    };
  }

  // EGD fetched directly from the browser (CORS *). One word -> lastname + name
  // (name-only is a no-op, EGD needs a lastname); two words -> both orders; deduped.
  async function egdByName(q) {
    var tokens = q.split(/\\s+/);
    var sets = tokens.length <= 1
      ? [{ lastname: q }, { name: q }]
      : [{ name: tokens[0], lastname: tokens.slice(1).join(' ') },
         { name: tokens.slice(1).join(' '), lastname: tokens[0] }];
    var seen = {}, order = [];
    for (var i = 0; i < sets.length; i++) {
      var usp = new URLSearchParams(sets[i]);
      try {
        var res = await fetch(EGD + 'GetPlayerDataByData.php?' + usp.toString());
        if (!res.ok) continue;
        var d = await res.json();
        if (!d || d.retcode !== 'Ok' || !Array.isArray(d.players)) continue;
        for (var j = 0; j < d.players.length; j++) {
          var p = d.players[j];
          if (p.Pin_Player && !seen[p.Pin_Player]) { seen[p.Pin_Player] = 1; order.push(mapPlayer(p)); }
        }
      } catch (e) { /* skip this query */ }
    }
    return order;
  }
  async function egdByPin(pin) {
    try {
      var res = await fetch(EGD + 'GetPlayerDataByPIN.php?pin=' + encodeURIComponent(pin));
      if (!res.ok) return [];
      var d = await res.json();
      if (!d || d.retcode !== 'Ok') return [];
      return [mapPlayer(d, pin)];
    } catch (e) { return []; }
  }

  async function search() {
    var q = qEl.value.trim();
    if (!q) { setMsg('Enter a name or PIN to search.', true); return; }
    setMsg('Searching EGD…'); results.innerHTML = ''; out.style.display = 'none';
    var players = /^\\d+$/.test(q) ? await egdByPin(q) : await egdByName(q);
    currentPlayers = players;
    if (!players.length) { setMsg('No EGD players found.'); return; }
    setMsg(players.length + ' result(s):');
    results.innerHTML = '<table><thead><tr><th>PIN</th><th class="l">Name</th><th>CC</th><th>Rank</th><th>GoR</th><th></th></tr></thead><tbody>' +
      players.map(function (p, i) {
        return '<tr><td>' + esc(p.pin) + '</td><td class="l">' + esc(p.name) + '</td><td>' + esc(p.country) +
          '</td><td>' + esc(p.rank) + '</td><td>' + (p.gor == null ? '—' : esc(String(p.gor))) +
          '</td><td><button class="gen" data-idx="' + i + '">Generate</button></td></tr>';
      }).join('') + '</tbody></table>';
    Array.prototype.forEach.call(results.querySelectorAll('button.gen'), function (b) {
      b.addEventListener('click', function () { generate(currentPlayers[+b.getAttribute('data-idx')], b); });
    });
  }

  async function generate(p, btn) {
    if (!p) return;
    btn.disabled = true; btn.textContent = '…';
    var r = await post('/admin/create-invite',
      { code: code, pin: p.pin, name: p.name, gor: p.gor, country: p.country });
    if (!r.data || !r.data.ok) {
      var err = (r.data && r.data.error) || 'error';
      setMsg(errMap[err] || ('Could not create invitation (' + err + ').'), true);
      btn.disabled = false; btn.textContent = 'Generate'; return;
    }
    btn.textContent = 'Done';
    showLink(r.data);
  }

  function showLink(d) {
    setMsg('');
    out.style.display = 'block';
    out.innerHTML = '<p>Invitation for <strong>' + esc(d.egdName) + '</strong> (GoR ' + esc(String(d.gor)) + '):</p>' +
      '<div class="linkrow"><input id="lnk" readonly value="' + esc(d.link) + '"><button id="copy">Copy</button></div>';
    var lnk = document.getElementById('lnk');
    document.getElementById('copy').addEventListener('click', function () {
      lnk.focus(); lnk.select();
      var done = function () { var c = document.getElementById('copy'); c.textContent = 'Copied'; setTimeout(function(){ c.textContent = 'Copy'; }, 1500); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(d.link).then(done, function () { try { document.execCommand('copy'); done(); } catch (e) {} });
      } else { try { document.execCommand('copy'); done(); } catch (e) {} }
    });
  }

  function wireTools() {
    qEl = document.getElementById('q');
    msg = document.getElementById('msg');
    results = document.getElementById('results');
    out = document.getElementById('out');
    document.getElementById('searchBtn').addEventListener('click', search);
    qEl.addEventListener('keydown', function (e) { if (e.key === 'Enter') search(); });
    tools.style.display = 'block';
    gate.style.display = 'none';
    qEl.focus();
  }

  // Auto-verify admin from the stored account code, then reveal the tools.
  (async function init() {
    if (!code) { showGate('No onlymove account found in this browser. Open this page where you use the app.'); return; }
    var r = await post('/admin/check', { code: code });
    if (r.status === 403) { showGate(errMap.unauthorized); return; }
    if (!r.data || !r.data.ok) { showGate('Could not verify admin access — reload to retry.'); return; }
    wireTools();
  })();
})();
</script>
</body>
</html>`;
}
