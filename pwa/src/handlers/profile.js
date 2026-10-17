// pwa/src/handlers/profile.js
//
// Per-user "hide my profile" preference for the public /leaderboard page.
// Presence of a hidden_profiles row = hidden: the user's display name, country,
// and declared / verified ranks are withheld from readLeaderboardRows (see
// src/leaderboard.js). A hidden player still appears on /leaderboard carrying
// only their predicted rank, under their self-chosen alias if they have one and
// as Anonymous otherwise.
//
// Two entry points, one flag: the checkbox in the app's Account dialog
// (js/account-ui.js), which is also where the leaderboard identity is linked, and
// the one at the foot of the in-app rank survey (js/rank-survey-prompt.js), which
// is where a declaration is made in the first place. Both read the flag before
// offering it, so each writes only on an actual change — unchecked-by-default plus
// an unconditional write would silently un-hide whoever opted out in the other
// place.
//
// Keyed by user_id and applying to ANY account — a player who never linked a
// federation identity can still have declared ranks (rank_surveys) that this
// hides — so the flag lives in its own table, not on external_ids.
//
//   POST /profile/hide  { userId, hidden: bool }  -> { ok: true, hidden }
//   POST /profile/alias { userId, adjective, color, animal } | { userId, clear: true }
//   GET  /profile?userId=<id>
//       -> { hidden: bool, linked: {...}|null, google: bool,
//            alias: {adjective, color, animal}|null,
//            declared: { <server>: <rank> } } (no-store)
//
// The GET answers the Account dialog's whole panel — the flag, the current
// federation binding, whether a Google account can sign in to this code, AND the
// account's own leaderboard alias — in one request, so opening it costs one
// round trip rather than four. `google` is a bare boolean because a bare boolean
// is all that is stored: the binding keeps Google's opaque subject id and no
// address (see handlers/google.js). `alias` is the raw ENGLISH triplet, not a
// display string: the dialog localizes it with the same shared/alias-words.js
// the picker offers, so what the user edits is exactly what is stored.
// `declared` is the account's latest survey answer per server, which the rank
// survey (the other reader of this GET) prefills its pickers with.

import { jsonResponse, CORS_HEADERS } from '../http.js';
import { isValidUserId } from '../guards.js';
import { readLinkedIdentity } from './link.js';
import { hasGoogleBinding } from './google.js';
import { readAlias } from '../aliases.js';
import { readDeclaredRanks } from './rank-survey.js';

// POST: set/clear the hidden flag. INSERT OR IGNORE to hide, DELETE to unhide —
// idempotent either way. Format-validated (isValidUserId + boolean) before any
// D1 write, like the other write handlers.
export async function handleProfileHide(request, env) {
  let data;
  try { data = await request.json(); } catch { return jsonResponse({ error: 'Invalid body' }, 400); }
  if (!isValidUserId(data?.userId) || typeof data.hidden !== 'boolean') {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }
  if (!env.DB) return jsonResponse({ error: 'Database not configured' }, 500);
  try {
    if (data.hidden) {
      await env.DB.prepare(
        `INSERT OR IGNORE INTO hidden_profiles (user_id, created_at) VALUES (?1, ?2)`
      ).bind(data.userId, new Date().toISOString()).run();
    } else {
      await env.DB.prepare(
        `DELETE FROM hidden_profiles WHERE user_id = ?1`
      ).bind(data.userId).run();
    }
    return jsonResponse({ ok: true, hidden: data.hidden });
  } catch (e) {
    console.error('handleProfileHide failed:', e);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}

// GET: the account's leaderboard profile — the hidden flag, the linked
// federation identity, the chosen alias and the latest survey declarations, so a
// dialog renders every control in its true state without guessing. no-store so
// the per-user answer is never edge-cached against another user's request (the
// service worker is told to skip this path too — see sw.js).
export async function handleProfile(request, env) {
  const userId = new URL(request.url).searchParams.get('userId');
  if (!isValidUserId(userId)) return jsonResponse({ error: 'Invalid userId' }, 400);
  if (!env.DB) return jsonResponse({ error: 'Database not configured' }, 500);
  try {
    const [row, linked, google, alias, declared] = await Promise.all([
      env.DB.prepare(`SELECT 1 AS x FROM hidden_profiles WHERE user_id = ?1`).bind(userId).first(),
      readLinkedIdentity(env, userId),
      hasGoogleBinding(env, userId),
      readAlias(env, userId),
      readDeclaredRanks(env, userId),
    ]);
    return new Response(JSON.stringify({ hidden: !!row, linked: linked || null, google, alias, declared }), {
      status: 200,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch (e) {
    console.error('handleProfile failed:', e);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}
