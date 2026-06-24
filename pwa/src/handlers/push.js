import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';

export async function handleSubscribe(request, env) {
  const data = await request.json();
  const sub = data && data.subscription;
  if (!isValidUserId(data?.userId) || !sub?.endpoint || !sub.keys?.p256dh || !sub.keys?.auth) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }
  const now = Date.now();
  // Keyed by endpoint (one row per device). On conflict the same physical push
  // channel is re-registering — re-point user_id (it may have switched accounts)
  // and refresh keys, but keep last_notified_at so a spurious re-subscribe can't
  // reset the anti-spam floor and re-nag. last_seen_at is set on insert and left
  // alone on conflict (the schedule upload that follows enableNotifications
  // bumps it immediately).
  await env.DB.prepare(
    `INSERT INTO push_subscriptions (endpoint, user_id, p256dh, auth, last_notified_at, created_at, updated_at, last_seen_at)
     VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?5, ?5)
     ON CONFLICT(endpoint) DO UPDATE SET
       user_id = excluded.user_id,
       p256dh = excluded.p256dh,
       auth = excluded.auth,
       updated_at = excluded.updated_at`
  ).bind(sub.endpoint, data.userId, sub.keys.p256dh, sub.keys.auth, now).run();
  return jsonResponse({ ok: true });
}

export async function handleUnsubscribe(request, env) {
  const data = await request.json();
  if (!isValidUserId(data?.userId)) return jsonResponse({ error: 'Invalid body' }, 400);
  // Multi-device: remove only THIS device's subscription (scoped to the account
  // so an endpoint can't be deleted out from under another user). A legacy
  // client that doesn't send its endpoint falls back to the old clear-the-whole-
  // account behavior.
  if (typeof data.endpoint === 'string' && data.endpoint) {
    await env.DB.prepare('DELETE FROM push_subscriptions WHERE endpoint = ? AND user_id = ?')
      .bind(data.endpoint, data.userId).run();
  } else {
    await env.DB.prepare('DELETE FROM push_subscriptions WHERE user_id = ?').bind(data.userId).run();
  }
  return jsonResponse({ ok: true });
}
