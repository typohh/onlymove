// pwa/src/handlers/profile.js
//
// Per-user "hide my profile" preference for the public /leaderboard +
// /tournament pages. Presence of a hidden_profiles row = hidden: the user's
// display name, country, and self-declared / GoR survey ranks are withheld from
// readLeaderboardRows (see src/leaderboard.js). A hidden player still appears on
// /leaderboard as an anonymous row carrying only their onlymove estimate, and is
// dropped from the registered-centric /tournament listing.
//
// Keyed by user_id and applying to ANY account — a never-registered player can
// still have declared ranks (rank_surveys) that this hides — so the flag lives
// in its own table, not on external_ids.
//
//   POST /profile/hide  { userId, hidden: bool }  -> { ok: true, hidden }
//   GET  /profile/hide?userId=<id>                -> { hidden: bool } (no-store)

import { jsonResponse, CORS_HEADERS } from '../http.js';
import { isValidUserId } from '../guards.js';

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

// GET: the current hidden state, so the leaderboard page renders its checkbox
// correctly. A point lookup on the PK. no-store so the per-user answer is never
// edge-cached against another user's request.
export async function handleProfileHidden(request, env) {
  const userId = new URL(request.url).searchParams.get('userId');
  if (!isValidUserId(userId)) return jsonResponse({ error: 'Invalid userId' }, 400);
  if (!env.DB) return jsonResponse({ error: 'Database not configured' }, 500);
  try {
    const row = await env.DB.prepare(
      `SELECT 1 AS x FROM hidden_profiles WHERE user_id = ?1`
    ).bind(userId).first();
    return new Response(JSON.stringify({ hidden: !!row }), {
      status: 200,
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch (e) {
    console.error('handleProfileHidden failed:', e);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}
