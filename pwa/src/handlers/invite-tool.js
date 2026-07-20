// pwa/src/handlers/invite-tool.js
//
// Invite generator — the web equivalent of generate_invite.py. A standalone page
// at GET /invite lets an eligible user search BOTH rating databases at once and mint
// an invitation link (an `invitation` row + the /register?t= URL, plus a QR code).
//
//   GET  /invite                        -> the invite page (no ?t=; ?t= is the
//                                          preview JSON in handlers/register.js)
//   GET  /admin/invite                  -> same page (legacy alias)
//   POST /invite/eligible {userId}      -> may this account create invites? (page gate)
//   POST /invite/create {userId, service, ...} -> insert invitation + QR
//
// Eligibility is NO LONGER admin-only: an account may invite if it is an admin
// (env.ADMIN_USER_IDS) OR it is a registered member (has an `external_ids`
// binding — the trust boundary, since registration itself is invite-only).
// The account code (user_id) is read automatically from the browser's localStorage
// (same key the PWA uses) and is recorded on the invitation as created_by_user_id
// — both the abuse-attribution key and the "who invited this person" record. The
// POST endpoints are also per-IP rate-limited (RL_WRITE) in the dispatcher.
//
// ONE unified search across both databases, CLIENT-SIDE (both send CORS `*`):
//   * EGD — europeangodatabase.eu (GetPlayerDataByData.php / GetPlayerDataByPIN.php).
//   * AGA — the AGA Ratings Explorer JSON api (aga-ratings-explorer.azurewebsites.net,
//     the current source embedded on usgo.org/ratings), which returns each player's
//     agaid + name + rating + OFFICIAL rank_label. The browser must search EGD (the
//     Worker can't reach it — CF orange-to-orange) so, for one search box, it searches
//     AGA client-side too and MERGES the results into one list labelled per source.
// The chosen row's fields are handed to /invite/create; the Worker only writes the row
// (it trusts the client's fetched values, as the EGD flow always has — no manual entry).
// The QR is rendered server-side from the minted link and returned in the response.

import qrcode from 'qrcode-generator';
import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import { isValidAgaId } from '../aga.js';
import { baseCss, ESC_SNIPPET } from './page-shell.js';

// Admins (configured account codes) can always invite. Fails closed if the secret
// is unset — an admin code is never accidentally granted.
function isAdmin(code, env) {
    const cfg = (env.ADMIN_USER_IDS || '').trim();
    if (!cfg) return false;
    if (typeof code !== 'string' || !isValidUserId(code)) return false;
    return cfg.split(',').map((s) => s.trim()).filter(Boolean).includes(code);
}

// A REGISTERED member (has an external_ids binding) may invite. Registration is
// itself invite-only, so this is the trust boundary — no need to require a
// computed rank (pred_egf_rank), which would only delay a freshly-registered
// player. (Deliberately NOT "has a leaderboard row": since the all-users
// expansion every reporting user has one of those lazily.)
async function isRegisteredMember(code, env) {
    if (!env.DB || typeof code !== 'string' || !isValidUserId(code)) return false;
    const row = await env.DB.prepare(
        `SELECT 1 AS ok FROM external_ids WHERE user_id = ?1`
    ).bind(code).first();
    return !!row;
}

// Combined gate used by both the page reveal and the create endpoint.
async function inviteEligibility(code, env) {
    const admin = isAdmin(code, env);
    const member = admin ? false : await isRegisteredMember(code, env);
    return { eligible: admin || member, admin, member };
}

const TOKEN_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
function generateToken() {
    const bytes = crypto.getRandomValues(new Uint8Array(8));
    let s = '';
    for (let i = 0; i < 8; i++) s += TOKEN_ALPHABET[bytes[i] % TOKEN_ALPHABET.length];
    return s;
}

// Render the invitation link as a scannable QR (SVG string). The lib draws a white
// background rect + black modules, so it scans on either light or dark chrome.
function makeQrSvg(link) {
    try {
        const qr = qrcode(0, 'M'); // typeNumber 0 = auto-size, EC level M
        qr.addData(link);
        qr.make();
        return qr.createSvgTag({ cellSize: 8, margin: 32, scalable: true });
    } catch (e) {
        console.error('qr generation failed:', e);
        return null; // page falls back to the copyable link alone
    }
}

// Insert an invitation with a fresh token; retry on the (astronomically unlikely)
// primary-key collision. Handles both services: `service` = 'EGD' (pin + gor) or
// 'AGA' (agaId + agaRating); egd_name/country are the generic display fields.
async function insertInvitation(env, inv, createdBy) {
    const nowIso = new Date().toISOString();
    for (let attempt = 0; ; attempt++) {
        const token = generateToken();
        try {
            await env.DB.prepare(
                `INSERT INTO invitation
                    (invite_token, egd_pin, egd_name, country_code, gor, created_at,
                     created_by_user_id, claimed_by_user_id, claimed_at,
                     service, aga_id, aga_rating)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, NULL, ?8, ?9, ?10)`
            ).bind(token, inv.pin ?? null, inv.name, inv.country, inv.gor ?? null, nowIso,
                   createdBy, inv.service, inv.agaId ?? null, inv.agaRating ?? null).run();
            return token;
        } catch (e) {
            if (attempt >= 4) throw e;
        }
    }
}

export async function handleInvitePage(env) {
    return new Response(renderInvitePage(), {
        status: 200,
        headers: { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' },
    });
}

// Page gate: may this browser's account code create invites?
export async function handleInviteEligible(request, env) {
    try {
        const { userId } = await request.json();
        const elig = await inviteEligibility(userId, env);
        return jsonResponse({ ok: true, ...elig });
    } catch (e) {
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}

export async function handleInviteCreate(request, env) {
    try {
        const body = await request.json();
        const { userId } = body;
        const elig = await inviteEligibility(userId, env);
        if (!elig.eligible) return jsonResponse({ ok: false, error: 'unauthorized' }, 403);
        if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);

        // Generic display fields, shared by both services.
        const name = typeof body.name === 'string' ? body.name.trim().slice(0, 100) : '';
        if (!name) return jsonResponse({ ok: false, error: 'bad_name' }, 400);
        const cc = (typeof body.country === 'string' ? body.country : '').slice(0, 2).toUpperCase();

        if (body.service === 'AGA') return createAgaInvite(env, { userId, name, cc, body });
        return createEgdInvite(env, { userId, name, cc, body });
    } catch (e) {
        console.error('invite create failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}

// EGD: the player fields come from the eligible client's own EGD fetch (the Worker
// can't reach EGD). Validate shape; trust the values.
async function createEgdInvite(env, { userId, name, cc, body }) {
    const pinStr = String(body.pin || '');
    if (!/^\d+$/.test(pinStr)) return jsonResponse({ ok: false, error: 'bad_pin' }, 400);
    const gorInt = Number.isFinite(Number(body.gor)) ? Math.round(Number(body.gor)) : null;
    if (gorInt === null) return jsonResponse({ ok: false, error: 'bad_gor' }, 400);

    // 1:1 binding: no point inviting a PIN that already has a registered account
    // (the claim would be rejected anyway). Surface it as a clean error.
    const existing = await env.DB.prepare(
        `SELECT user_id FROM external_ids WHERE egd_pin = ?1`
    ).bind(pinStr).first();
    if (existing) return jsonResponse({ ok: false, error: 'already_registered' }, 409);

    // Idempotent: if this PIN already has an UNCLAIMED invitation, reuse it (refresh
    // its stored EGD fields) instead of piling up duplicate links for one person.
    // created_by is backfilled only if it was empty (e.g. an admin-CLI invite).
    const openInvite = await env.DB.prepare(
        `SELECT invite_token FROM invitation
          WHERE egd_pin = ?1 AND claimed_by_user_id IS NULL
          ORDER BY created_at DESC LIMIT 1`
    ).bind(pinStr).first();
    if (openInvite) {
        await env.DB.prepare(
            `UPDATE invitation
                SET egd_name = ?2, country_code = ?3, gor = ?4, service = 'EGD',
                    created_by_user_id = COALESCE(created_by_user_id, ?5)
              WHERE invite_token = ?1`
        ).bind(openInvite.invite_token, name, cc, gorInt, userId).run();
        const link = `https://onlymove.net/register?t=${openInvite.invite_token}`;
        return jsonResponse({
            ok: true, inviteToken: openInvite.invite_token, link, qrSvg: makeQrSvg(link),
            egdName: name, service: 'EGD', gor: gorInt, country: cc, reused: true,
        });
    }

    const token = await insertInvitation(env,
        { service: 'EGD', pin: pinStr, name, country: cc, gor: gorInt }, userId);
    const link = `https://onlymove.net/register?t=${token}`;
    return jsonResponse({
        ok: true, inviteToken: token, link, qrSvg: makeQrSvg(link),
        egdName: name, service: 'EGD', gor: gorInt, country: cc,
    });
}

// AGA: the fields come from the client's own Ratings Explorer search (the rating is
// snapshotted for display; the authoritative rank_label is (re)fetched Worker-side at
// claim time + weekly by the cron — see register.js / refreshAgaRatings). Rating is
// never inviter-entered: a non-numeric rating is simply dropped to null.
async function createAgaInvite(env, { userId, name, cc, body }) {
    const agaId = String(body.agaId || body.pin || '');
    if (!isValidAgaId(agaId)) return jsonResponse({ ok: false, error: 'bad_id' }, 400);
    const agaRating = Number.isFinite(Number(body.rating)) ? Number(body.rating) : null;

    const existing = await env.DB.prepare(
        `SELECT user_id FROM external_ids WHERE aga_id = ?1`
    ).bind(agaId).first();
    if (existing) return jsonResponse({ ok: false, error: 'already_registered' }, 409);

    const openInvite = await env.DB.prepare(
        `SELECT invite_token FROM invitation
          WHERE service = 'AGA' AND aga_id = ?1 AND claimed_by_user_id IS NULL
          ORDER BY created_at DESC LIMIT 1`
    ).bind(agaId).first();
    if (openInvite) {
        await env.DB.prepare(
            `UPDATE invitation
                SET egd_name = ?2, country_code = ?3, aga_rating = ?4,
                    created_by_user_id = COALESCE(created_by_user_id, ?5)
              WHERE invite_token = ?1`
        ).bind(openInvite.invite_token, name, cc, agaRating, userId).run();
        const link = `https://onlymove.net/register?t=${openInvite.invite_token}`;
        return jsonResponse({
            ok: true, inviteToken: openInvite.invite_token, link, qrSvg: makeQrSvg(link),
            egdName: name, service: 'AGA', agaId, rating: agaRating, country: cc, reused: true,
        });
    }

    const token = await insertInvitation(env,
        { service: 'AGA', name, country: cc, agaId, agaRating }, userId);
    const link = `https://onlymove.net/register?t=${token}`;
    return jsonResponse({
        ok: true, inviteToken: token, link, qrSvg: makeQrSvg(link),
        egdName: name, service: 'AGA', agaId, rating: agaRating, country: cc,
    });
}

function renderInvitePage() {
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<title>onlymove — invite a player</title>
<style>
  ${baseCss(760)}
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
  th, td { padding: .45rem .4rem; text-align: right; white-space: nowrap; }
  th.l, td.l { text-align: left; width: 99%; overflow: hidden; text-overflow: ellipsis; max-width: 12rem; }
  th { font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; opacity: .55; border-bottom: 1px solid; }
  td button { padding: .3rem .7rem; font-size: .85rem; }
  .src { font-size: .68rem; font-weight: 700; letter-spacing: .04em; padding: .1rem .4rem; border-radius: 6px;
         border: 1px solid currentColor; }
  .src.egd { color: #2e78c0; } .src.aga { color: #c0392b; }
  #out { display: none; margin-top: 1.5rem; padding: 1rem; border: 1px solid;
         border-radius: 10px; background: color-mix(in srgb, currentColor 5%, transparent); }
  #out .qr { margin: .75rem 0; text-align: center; }
  #out .qr svg { width: 200px; height: 200px; max-width: 100%; background: #fff; border-radius: 8px; }
  #out .linkrow { display: flex; gap: .5rem; margin-top: .5rem; }
  #out input { flex: 1; font: inherit; padding: .45rem .6rem; border: 1px solid; border-radius: 8px;
               background: transparent; color: inherit; }
</style>
</head>
<body>
<h1>Invite a player</h1>
<p class="sub">As you are already on the leaderboard, you can invite others to it. Search by name or id — results from the European (EGD) and American (AGA) rating databases are listed together, each labelled by source. Pick a player and share the link or QR code below; when they open it in their onlymove app it binds their account to that identity and adds them to the leaderboard. (An account links to one database — EGD or AGA — not both.)</p>
<div id="gate">Checking access…</div>
<div id="tools">
  <div class="searchrow">
    <input type="text" id="q" placeholder="Name, EGD PIN, or AGA id — e.g. Henry Hemming, 16025130, or 20414" autocomplete="off">
    <button id="searchBtn">Search</button>
  </div>
  <div id="msg"></div>
  <div id="results"></div>
  <div id="out"></div>
</div>
<script>
(function () {
  var EGD = 'https://www.europeangodatabase.eu/EGD/';
  var AGA = 'https://aga-ratings-explorer.azurewebsites.net/api/ratings-explorer/';
  var gate = document.getElementById('gate');
  var tools = document.getElementById('tools');
  var qEl, msg, results, out;
  var currentPlayers = [];

  // The invite key is THIS browser's account code (the PWA's user_id) — read it
  // automatically, no prompt. Same origin => same localStorage as the app. It is
  // also stored on the invitation as who invited the recipient.
  var userId = '';
  try { userId = localStorage.getItem('userId') || ''; } catch (e) {}

  var errMap = {
    unauthorized: 'This account cannot create invitations.',
    already_registered: 'That player is already registered.',
    bad_pin: 'Invalid PIN.', bad_name: 'Missing name.', bad_gor: 'Missing GoR.',
    bad_id: 'Invalid AGA member id.'
  };
  ${ESC_SNIPPET}
  function setMsg(t, err) { msg.textContent = t || ''; msg.className = err ? 'err' : ''; }
  function showGate(t) { gate.textContent = t; gate.className = 'err'; tools.style.display = 'none'; }

  async function post(url, body) {
    var res;
    try { res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) }); }
    catch (e) { return { status: 0, data: null }; }
    var data = null; try { data = await res.json(); } catch (e) {}
    return { status: res.status, data: data };
  }

  // --- EGD (europeangodatabase.eu, CORS *) --------------------------------------
  function mapEgd(p, pinFallback) {
    var gor = parseInt(p.Gor, 10);
    return {
      service: 'EGD', id: p.Pin_Player || pinFallback || '',
      name: ((p.Name || '') + ' ' + (p.Last_Name || '')).trim(),
      loc: p.Country_Code || '', rank: p.Grade || '',
      rating: isNaN(gor) ? null : gor,     // GoR
      pin: p.Pin_Player || pinFallback || '', gor: isNaN(gor) ? null : gor, country: p.Country_Code || ''
    };
  }
  // One word -> lastname + name; two words -> both orders; deduped by PIN.
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
          if (p.Pin_Player && !seen[p.Pin_Player]) { seen[p.Pin_Player] = 1; order.push(mapEgd(p)); }
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
      return [mapEgd(d, pin)];
    } catch (e) { return []; }
  }

  // --- AGA (Ratings Explorer, CORS *) -------------------------------------------
  function mapAga(p) {
    var r = (typeof p.rating === 'number') ? p.rating : parseFloat(p.rating);
    return {
      service: 'AGA', id: String(p.agaid || ''),
      name: p.display_name || ((p.first_name || '') + ' ' + (p.last_name || '')).trim(),
      loc: p.state || '', rank: p.rank_label || '',
      rating: isFinite(r) ? r : null,      // AGA rating (rank comes from rank_label)
      agaid: String(p.agaid || '')
    };
  }
  var AGA_FILTERS = { status: 'all', recent_activity_years: 'none', limit: '25', page: '0' };
  async function agaByName(q) {
    var tokens = q.split(/\\s+/);
    var sets = tokens.length <= 1
      ? [{ last_name: q }, { first_name: q }]
      : [{ first_name: tokens[0], last_name: tokens.slice(1).join(' ') },
         { first_name: tokens.slice(1).join(' '), last_name: tokens[0] }];
    var seen = {}, order = [];
    for (var i = 0; i < sets.length; i++) {
      var params = {}; for (var k in AGA_FILTERS) params[k] = AGA_FILTERS[k];
      for (var k2 in sets[i]) params[k2] = sets[i][k2];
      try {
        var res = await fetch(AGA + 'players?' + new URLSearchParams(params).toString());
        if (!res.ok) continue;
        var d = await res.json();
        var arr = (d && d.results) || [];
        for (var j = 0; j < arr.length; j++) {
          var p = arr[j];
          if (p.agaid && !seen[p.agaid]) { seen[p.agaid] = 1; order.push(mapAga(p)); }
        }
      } catch (e) { /* skip */ }
    }
    return order;
  }
  async function agaById(id) {
    try {
      var res = await fetch(AGA + 'player?agaid=' + encodeURIComponent(id));
      if (!res.ok) return [];
      var d = await res.json();
      if (!d || !d.player) return [];
      return [mapAga(d.player)];
    } catch (e) { return []; }
  }

  // --- unified search (both databases in parallel, merged, labelled) ------------
  async function search() {
    var q = qEl.value.trim();
    if (!q) { setMsg('Enter a name or id to search.', true); return; }
    setMsg('Searching EGD + AGA…'); results.innerHTML = ''; out.style.display = 'none';
    var isNum = /^\\d+$/.test(q);
    // An all-digits query could be either an EGD PIN or an AGA id — try both.
    var settled = await Promise.all(
      isNum ? [egdByPin(q), agaById(q)] : [egdByName(q), agaByName(q)]
    );
    var players = settled[0].concat(settled[1]);
    currentPlayers = players;
    if (!players.length) { setMsg('No players found in EGD or AGA.'); return; }
    setMsg(players.length + ' result(s):');
    results.innerHTML = '<table><thead><tr><th class="l">Source</th><th>ID</th><th class="l">Name</th><th>Loc</th><th>Rank</th><th>Rating</th><th></th></tr></thead><tbody>' +
      players.map(function (p, i) {
        var badge = '<span class="src ' + (p.service === 'AGA' ? 'aga' : 'egd') + '">' + p.service + '</span>';
        return '<tr><td class="l">' + badge + '</td><td>' + esc(p.id) + '</td><td class="l">' + esc(p.name) +
          '</td><td>' + esc(p.loc) + '</td><td>' + esc(p.rank) + '</td><td>' +
          (p.rating == null ? '—' : esc(String(p.rating))) +
          '</td><td><button class="gen" data-idx="' + i + '">Generate</button></td></tr>';
      }).join('') + '</tbody></table>';
    Array.prototype.forEach.call(results.querySelectorAll('button.gen'), function (b) {
      b.addEventListener('click', function () { generate(currentPlayers[+b.getAttribute('data-idx')], b); });
    });
  }

  async function generate(p, btn) {
    if (!p) return;
    btn.disabled = true; btn.textContent = '…';
    // AGA sends no country (the Explorer gives a US state, not a country code) — the
    // leaderboard's geo-IP fallback supplies the flag. EGD sends its country code.
    var body = p.service === 'AGA'
      ? { userId: userId, service: 'AGA', agaId: p.agaid, name: p.name, rating: p.rating }
      : { userId: userId, service: 'EGD', pin: p.pin, name: p.name, gor: p.gor, country: p.country };
    var r = await post('/invite/create', body);
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
    var reused = d.reused ? ' <em>(existing unclaimed link — reused)</em>' : '';
    var ratingTxt = d.service === 'AGA'
      ? 'AGA rating ' + (d.rating == null ? '—' : esc(String(d.rating)))
      : 'GoR ' + esc(String(d.gor));
    // d.qrSvg is Worker-generated SVG built from the minted link (no user content) — safe to inline.
    out.innerHTML = '<p>Invitation for <strong>' + esc(d.egdName) + '</strong> (' + esc(d.service) + ' · ' + ratingTxt + ')' + reused + ':</p>' +
      (d.qrSvg ? '<div class="qr">' + d.qrSvg + '</div>' : '') +
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

  // Auto-verify eligibility from the stored account code, then reveal the tools.
  (async function init() {
    if (!userId) { showGate('No onlymove account found in this browser. Open this page where you use the app.'); return; }
    var r = await post('/invite/eligible', { userId: userId });
    if (!r.data || !r.data.ok) { showGate('Could not verify access — reload to retry.'); return; }
    if (!r.data.eligible) { showGate('Only registered leaderboard players can create invitations. Once you are on the leaderboard, this page unlocks.'); return; }
    wireTools();
  })();
})();
</script>
</body>
</html>`;
}
