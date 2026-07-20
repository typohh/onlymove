import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';

// Format-only validation (same policy as the other write endpoints): the
// endpoint must be a real https URL — the recall cron fetch()es it on every due
// tick, so a garbage/attacker-chosen value would otherwise be dialed forever —
// and the fields must fit sane length caps so a spoofed body can't stuff the
// row. Real push endpoints run ~100-400 chars; the keys are fixed-size base64url
// (p256dh ~88, auth ~24).
function isValidEndpoint(e) {
  if (typeof e !== 'string' || e.length === 0 || e.length > 1024) return false;
  try { return new URL(e).protocol === 'https:'; } catch { return false; }
}
function isValidKey(k, max) {
  return typeof k === 'string' && k.length > 0 && k.length <= max && /^[A-Za-z0-9_=-]+$/.test(k);
}

// Fold the client's navigator.language into one of the notification-string
// languages. Mirrors getLang() in js/i18n.js — keep the prefix set in sync with
// the string table in recall-notifications.js. Unknown/absent -> 'en'.
function resolveLang(l) {
  if (typeof l !== 'string') return 'en';
  if (l.startsWith('zh')) return 'zh';
  if (l.startsWith('ja')) return 'ja';
  if (l.startsWith('ko')) return 'ko';
  return 'en';
}

export async function handleSubscribe(request, env) {
  const data = await request.json();
  const sub = data && data.subscription;
  if (!isValidUserId(data?.userId) || !sub || !isValidEndpoint(sub.endpoint)
      || !isValidKey(sub.keys?.p256dh, 256) || !isValidKey(sub.keys?.auth, 256)) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }
  const now = Date.now();
  const lang = resolveLang(data.lang);
  // Keyed by endpoint (one row per device). On conflict the same physical push
  // channel is re-registering — re-point user_id (it may have switched accounts)
  // and refresh keys, but keep last_notified_at so a spurious re-subscribe can't
  // reset the anti-spam floor and re-nag. last_seen_at is set on insert and left
  // alone on conflict (the schedule upload that follows enableNotifications
  // bumps it immediately). lang is refreshed on conflict so a device that
  // changed its UI language re-points the reminders too.
  await env.DB.prepare(
    `INSERT INTO push_subscriptions (endpoint, user_id, p256dh, auth, last_notified_at, created_at, updated_at, last_seen_at, lang)
     VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?5, ?5, ?6)
     ON CONFLICT(endpoint) DO UPDATE SET
       user_id = excluded.user_id,
       p256dh = excluded.p256dh,
       auth = excluded.auth,
       updated_at = excluded.updated_at,
       lang = excluded.lang`
  ).bind(sub.endpoint, data.userId, sub.keys.p256dh, sub.keys.auth, now, lang).run();
  return jsonResponse({ ok: true });
}

export async function handleUnsubscribe(request, env) {
  const data = await request.json();
  if (!isValidUserId(data?.userId)) return jsonResponse({ error: 'Invalid body' }, 400);
  // Multi-device: remove only THIS device's subscription (scoped to the account
  // so an endpoint can't be deleted out from under another user). A legacy
  // client that doesn't send its endpoint falls back to clearing the whole
  // account.
  if (typeof data.endpoint === 'string' && data.endpoint) {
    await env.DB.prepare('DELETE FROM push_subscriptions WHERE endpoint = ? AND user_id = ?')
      .bind(data.endpoint, data.userId).run();
  } else {
    await env.DB.prepare('DELETE FROM push_subscriptions WHERE user_id = ?').bind(data.userId).run();
  }
  return jsonResponse({ ok: true });
}
