// pwa/src/handlers/admin-invite.js
//
// Admin invite generator — the web equivalent of generate_invite.py. A standalone
// page at GET /admin/invite lets an admin search EGD (by name or PIN) and mint an
// invitation link (an `invitation` row + the /register?t= URL).
//
//   GET  /admin/invite               -> the admin page (inert without a valid code)
//   POST /admin/egd-search {code, q} -> EGD search (name or PIN), admin-gated
//   POST /admin/create-invite {code, pin} -> insert invitation, return the link
//
// Auth is the admin's own account code (user_id) checked against env.ADMIN_USER_IDS
// (a wrangler secret, comma-separated for multiple admins — see wrangler.toml). It
// FAILS CLOSED when unset, so a missing secret means nobody can generate invites.
// The write endpoints are also per-IP rate-limited (RL_WRITE) in the dispatcher,
// which bounds any code-guessing on top of the 62^8 token space.

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import { fetchEgdPlayer, searchEgdPlayers } from '../leaderboard.js';

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

export async function handleAdminSearch(request, env) {
    try {
        const { code, query } = await request.json();
        if (!isAdmin(code, env)) return jsonResponse({ ok: false, error: 'unauthorized' }, 403);
        if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);
        const q = String(query || '').trim();
        if (!q) return jsonResponse({ ok: true, players: [] });
        // A numeric query is a PIN (single lookup); otherwise a name search.
        let players;
        if (/^\d+$/.test(q)) {
            const p = await fetchEgdPlayer(q);
            players = p ? [{ pin: q, name: p.egd_name || '', country: p.country_code || '', rank: '', gor: p.gor }] : [];
        } else {
            players = await searchEgdPlayers(q);
        }
        return jsonResponse({ ok: true, players });
    } catch (e) {
        console.error('admin search failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}

export async function handleAdminCreate(request, env) {
    try {
        const { code, pin } = await request.json();
        if (!isAdmin(code, env)) return jsonResponse({ ok: false, error: 'unauthorized' }, 403);
        if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);
        const pinStr = String(pin || '');
        if (!/^\d+$/.test(pinStr)) return jsonResponse({ ok: false, error: 'bad_pin' }, 400);

        // Validate against EGD (rejects a bad PIN) and capture the current name/GoR/country.
        const p = await fetchEgdPlayer(pinStr);
        if (!p || !p.egd_name || p.gor === null) return jsonResponse({ ok: false, error: 'egd_not_found' }, 404);

        // 1:1 binding: no point inviting a PIN that already has a registered account
        // (the claim would be rejected anyway). Surface it as a clean error.
        const existing = await env.DB.prepare(
            `SELECT user_id FROM leaderboard WHERE egd_pin = ?1`
        ).bind(pinStr).first();
        if (existing) return jsonResponse({ ok: false, error: 'already_registered' }, 409);

        const token = await insertInvitation(env, {
            pin: pinStr, name: p.egd_name, country: p.country_code || '', gor: p.gor,
        });
        return jsonResponse({
            ok: true,
            inviteToken: token,
            link: `https://onlymove.net/register?t=${token}`,
            egdName: p.egd_name, gor: p.gor, country: p.country_code || '',
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
<title>onlymove — invite admin</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0; padding: 1.5rem 1rem 3rem;
         max-width: 720px; margin-inline: auto; }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  p.sub { margin: 0 0 1.25rem; opacity: .6; font-size: .85rem; }
  label { display: block; font-size: .8rem; opacity: .7; margin: 0 0 .2rem; }
  input[type=text], input[type=password] { font: inherit; padding: .45rem .6rem; border: 1px solid;
         border-radius: 8px; background: transparent; color: inherit; box-sizing: border-box; }
  .field { margin-bottom: .9rem; }
  #code { width: 16rem; max-width: 100%; }
  .searchrow { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1rem; }
  #q { flex: 1 1 12rem; }
  button { font: inherit; padding: .45rem .9rem; border: 1px solid currentColor; border-radius: 8px;
           background: transparent; color: inherit; cursor: pointer; white-space: nowrap; }
  button:disabled { opacity: .5; cursor: default; }
  #msg { min-height: 1.4em; margin: .25rem 0 1rem; font-size: .9rem; opacity: .8; }
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
<p class="sub">Admin only — the web equivalent of generate_invite.py. Enter your account code, search EGD by name or PIN, and mint an invite link.</p>
<div class="field">
  <label for="code">Admin account code</label>
  <input type="password" id="code" autocomplete="off" spellcheck="false">
</div>
<div class="field">
  <label for="q">EGD name or PIN</label>
  <div class="searchrow">
    <input type="text" id="q" placeholder="e.g. Hemming, or Henry Hemming, or 16025130" autocomplete="off">
    <button id="searchBtn">Search</button>
  </div>
</div>
<div id="msg"></div>
<div id="results"></div>
<div id="out"></div>
<script>
(function () {
  var codeEl = document.getElementById('code');
  var qEl = document.getElementById('q');
  var msg = document.getElementById('msg');
  var results = document.getElementById('results');
  var out = document.getElementById('out');
  var SAVED = 'onlymove_admin_code';
  try { codeEl.value = localStorage.getItem(SAVED) || ''; } catch (e) {}

  var errMap = {
    unauthorized: 'Not authorized — check your account code.',
    already_registered: 'That EGD player is already registered.',
    egd_not_found: 'EGD PIN not found.',
    bad_pin: 'Invalid PIN.'
  };
  function setMsg(t, err) { msg.textContent = t || ''; msg.className = err ? 'err' : ''; }
  function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function saveCode(){ try { localStorage.setItem(SAVED, codeEl.value.trim()); } catch (e) {} }

  async function post(url, body) {
    var res;
    try { res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); }
    catch (e) { return { status: 0, data: null }; }
    var data = null; try { data = await res.json(); } catch (e) {}
    return { status: res.status, data: data };
  }

  async function search() {
    var code = codeEl.value.trim(), q = qEl.value.trim();
    if (!code) { setMsg('Enter your admin account code.', true); return; }
    if (!q) { setMsg('Enter a name or PIN to search.', true); return; }
    saveCode();
    setMsg('Searching…'); results.innerHTML = ''; out.style.display = 'none';
    var r = await post('/admin/egd-search', { code: code, query: q });
    if (r.status === 403) { setMsg(errMap.unauthorized, true); return; }
    if (!r.data || !r.data.ok) { setMsg('Search failed.', true); return; }
    var players = r.data.players || [];
    if (!players.length) { setMsg('No EGD players found.'); return; }
    setMsg(players.length + ' result(s):');
    results.innerHTML = '<table><thead><tr><th>PIN</th><th class="l">Name</th><th>CC</th><th>Rank</th><th>GoR</th><th></th></tr></thead><tbody>' +
      players.map(function (p) {
        return '<tr><td>' + esc(p.pin) + '</td><td class="l">' + esc(p.name) + '</td><td>' + esc(p.country||'') +
          '</td><td>' + esc(p.rank||'') + '</td><td>' + (p.gor==null?'—':esc(String(p.gor))) +
          '</td><td><button class="gen" data-pin="' + esc(p.pin) + '">Generate</button></td></tr>';
      }).join('') + '</tbody></table>';
    Array.prototype.forEach.call(results.querySelectorAll('button.gen'), function (b) {
      b.addEventListener('click', function () { generate(b.getAttribute('data-pin'), b); });
    });
  }

  async function generate(pin, btn) {
    var code = codeEl.value.trim();
    if (!code) { setMsg('Enter your admin account code.', true); return; }
    btn.disabled = true; btn.textContent = '…';
    var r = await post('/admin/create-invite', { code: code, pin: pin });
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

  document.getElementById('searchBtn').addEventListener('click', search);
  qEl.addEventListener('keydown', function (e) { if (e.key === 'Enter') search(); });
  codeEl.addEventListener('change', saveCode);
})();
</script>
</body>
</html>`;
}
