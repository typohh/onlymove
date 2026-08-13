import { buildPushPayload } from '@block65/webcrypto-web-push';

const TWO_HOURS_MS = 2 * 3600 * 1000;
const SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000;
// Anti-spam floor between consecutive pushes to the same user. With an hourly
// cron, multiple problems may become due back-to-back; this caps each user at
// roughly four pushes per day in the worst case.
const MIN_NOTIFY_INTERVAL_MS = 6 * 3600 * 1000;
// Suppress recall reminders for users who were active since the previous cron
// tick. Every /schedule call (load / tab-hidden / heartbeat, even an empty
// diff) bumps push_subscriptions.last_seen_at, so a recent watermark means
// they're already practicing — a "ready to review" push would be noise.
// (recall_schedule.updated_at doubles as the watermark for legacy replace-all
// clients.) We don't mark anything notified, so the reminder simply defers to
// the next tick once they've gone idle. Matches the hourly cron in wrangler.toml.
const ACTIVE_SUPPRESS_MS = 1 * 3600 * 1000;

export async function runRecallNotifications(env) {
  if (!env.DB || !env.VAPID_PRIVATE_KEY || !env.VAPID_PUBLIC_KEY) {
    console.error('Recall cron: missing DB or VAPID configuration; skipping.');
    return;
  }

  const now = Date.now();

  // Due problems for users that have at least one push subscription. No join to
  // push_subscriptions here: a user can have several device rows, which would
  // multiply each problem row — we read the problems once (per user) and fan
  // out to that user's devices below. The IN-subquery scopes the
  // recall_schedule read to subscribed users.
  const due = await env.DB.prepare(
    // exit_at <= ?1 + ?2 (not exit_at - ?2 <= ?1): the arithmetic must stay off
    // the column so the partial index idx_recall_due_exit(exit_at) stays usable
    // — the column-side form is a full scan of every un-notified row.
    `SELECT user_id, problem_id, enter_at, exit_at,
            enter_notified_at, exit_notified_at, updated_at
     FROM recall_schedule
     WHERE ((enter_notified_at IS NULL AND enter_at <= ?1)
         OR (exit_notified_at  IS NULL AND exit_at <= ?1 + ?2 AND exit_at > ?1))
       AND user_id IN (SELECT user_id FROM push_subscriptions)`
  ).bind(now, TWO_HOURS_MS).all();

  // Full per-user backlog of problems currently inside the review window,
  // regardless of whether each individual row has already been notified.
  // Used to render "N problems ready to review" rather than the
  // delta-since-last-cron-tick — the user wants to see their total pending
  // load, not just what newly entered in the last hour.
  const backlog = await env.DB.prepare(
    // Scoped to subscribed users like the due query above — the counts are only
    // ever read for users being notified, and without the scope this scans every
    // user's schedule rows hourly.
    `SELECT user_id, COUNT(*) AS n
     FROM recall_schedule
     WHERE enter_at <= ?1 AND exit_at > ?1
       AND user_id IN (SELECT user_id FROM push_subscriptions)
     GROUP BY user_id`
  ).bind(now).all();
  const backlogByUser = new Map();
  for (const row of (backlog.results || [])) backlogByUser.set(row.user_id, row.n);

  const byUser = new Map();
  for (const row of (due.results || [])) {
    let u = byUser.get(row.user_id);
    if (!u) {
      u = { lastActiveAt: 0, enteredIds: [], exitingIds: [], subs: [] };
      byUser.set(row.user_id, u);
    }
    // Activity watermark from the schedule side: recall_schedule.updated_at
    // covers legacy replace-all clients. push_subscriptions.last_seen_at
    // (bumped on every /schedule call) is folded in per-device below.
    u.lastActiveAt = Math.max(u.lastActiveAt, row.updated_at || 0);
    if (row.enter_notified_at === null && row.enter_at <= now) {
      u.enteredIds.push(row.problem_id);
    }
    if (row.exit_notified_at === null && row.exit_at - TWO_HOURS_MS <= now && row.exit_at > now) {
      u.exitingIds.push(row.problem_id);
    }
  }

  if (byUser.size > 0) {
    // Attach every device of each due user. push_subscriptions is small (one row
    // per opted-in device), so read it whole and bucket in JS rather than
    // building an IN-list over the variable due-user set.
    const subsAll = await env.DB.prepare(
      `SELECT endpoint, user_id, p256dh, auth, last_notified_at, last_seen_at, lang
       FROM push_subscriptions`
    ).all();
    for (const s of (subsAll.results || [])) {
      const u = byUser.get(s.user_id);
      if (!u) continue;
      u.subs.push(s);
      u.lastActiveAt = Math.max(u.lastActiveAt, s.last_seen_at || 0);
    }

    const vapid = {
      subject: env.VAPID_SUBJECT,
      publicKey: env.VAPID_PUBLIC_KEY,
      privateKey: env.VAPID_PRIVATE_KEY,
    };

    const updates = [];
    for (const [userId, u] of byUser) {
      if (u.subs.length === 0) continue;
      if (u.enteredIds.length === 0 && u.exitingIds.length === 0) continue;
      // Active on SOME device since the previous tick — they're already
      // practicing, so defer the reminder for the whole account (suppression is
      // per-user, not per-device) rather than nagging mid-session.
      if (u.lastActiveAt && (now - u.lastActiveAt) < ACTIVE_SUPPRESS_MS) continue;

      const backlog = backlogByUser.get(userId) || 0;

      // Fan out to every device. Throttle is per-device (last_notified_at lives
      // on the row); dead endpoints are pruned individually. The message is
      // rendered per device in that device's stored language (sub.lang), so an
      // account's phone and laptop can read the reminder in different languages.
      let deliveredToAny = false;
      for (const sub of u.subs) {
        if (sub.last_notified_at && (now - sub.last_notified_at) < MIN_NOTIFY_INTERVAL_MS) continue;

        const message = buildNotificationMessage(u.enteredIds, u.exitingIds, backlog, sub.lang);
        const payload = { data: JSON.stringify(message), options: { ttl: 6 * 3600 } };
        const subscription = {
          endpoint: sub.endpoint,
          expirationTime: null,
          keys: { p256dh: sub.p256dh, auth: sub.auth },
        };
        let res;
        try {
          const init = await buildPushPayload(payload, subscription, vapid);
          res = await fetch(sub.endpoint, init);
        } catch (err) {
          console.error(`Push send threw for ${userId}:`, err);
          continue;
        }
        if (res.status === 404 || res.status === 410) {
          updates.push(env.DB.prepare(
            'DELETE FROM push_subscriptions WHERE endpoint = ?'
          ).bind(sub.endpoint));
          continue;
        }
        if (!res.ok) {
          console.error(`Push failed for ${userId}: HTTP ${res.status}`);
          continue;
        }
        deliveredToAny = true;
        updates.push(env.DB.prepare(
          'UPDATE push_subscriptions SET last_notified_at = ? WHERE endpoint = ?'
        ).bind(now, sub.endpoint));
      }

      // The notified flags are per (user, problem), not per device. Mark them
      // only if at least one device got the push, so a user whose every device
      // was throttled/failed simply retries next tick.
      if (deliveredToAny) {
        for (const id of u.enteredIds) {
          updates.push(env.DB.prepare(
            'UPDATE recall_schedule SET enter_notified_at = ? WHERE user_id = ? AND problem_id = ?'
          ).bind(now, userId, id));
        }
        for (const id of u.exitingIds) {
          updates.push(env.DB.prepare(
            'UPDATE recall_schedule SET exit_notified_at = ? WHERE user_id = ? AND problem_id = ?'
          ).bind(now, userId, id));
        }
      }
    }

    if (updates.length) await env.DB.batch(updates);
  }

  // Purge rows whose window expired long enough ago that no useful
  // notification could still be sent — keeps the table from accumulating
  // dead state for users the client never reached again.
  await env.DB.prepare('DELETE FROM recall_schedule WHERE exit_at < ?')
    .bind(now - SEVEN_DAYS_MS).run();
}

// Notification string table. Server-side counterpart of js/i18n.js (the client
// dictionary can't be imported into the Worker). Keep the language set in sync
// with resolveLang() in handlers/push.js. `{n}` / `{m}` are the entered /
// expiring counts. Singular vs plural is handled by separate *One / *Many keys
// (CJK has no plural inflection, so both forms read the same there).
const NOTIF_STR = {
  en: {
    titleOne: '1 problem ready to review',
    titleMany: '{n} problems ready to review',
    bodyBoth: '{n} new, {m} expiring soon',
    bodyExpiringOne: '1 expiring soon',
    bodyExpiringMany: '{m} expiring soon',
    bodyTap: 'Tap to open',
  },
  zh: {
    titleOne: '有 1 道题可以复习了',
    titleMany: '有 {n} 道题可以复习了',
    bodyBoth: '{n} 道新题，{m} 道即将到期',
    bodyExpiringOne: '1 道即将到期',
    bodyExpiringMany: '{m} 道即将到期',
    bodyTap: '点击打开',
  },
  ja: {
    titleOne: '復習できる問題が1問あります',
    titleMany: '復習できる問題が{n}問あります',
    bodyBoth: '新規{n}問、まもなく期限切れ{m}問',
    bodyExpiringOne: 'まもなく期限切れ1問',
    bodyExpiringMany: 'まもなく期限切れ{m}問',
    bodyTap: 'タップして開く',
  },
  ko: {
    titleOne: '복습할 문제가 1개 있습니다',
    titleMany: '복습할 문제가 {n}개 있습니다',
    bodyBoth: '새 문제 {n}개, 곧 만료 {m}개',
    bodyExpiringOne: '곧 만료되는 문제 1개',
    bodyExpiringMany: '곧 만료되는 문제 {m}개',
    bodyTap: '탭하여 열기',
  },
};

function notifTerm(lang, key, params) {
  const db = NOTIF_STR[lang] || NOTIF_STR.en;
  let s = key in db ? db[key] : NOTIF_STR.en[key];
  if (params) s = s.replace(/\{(\w+)\}/g, (mm, name) => (name in params ? String(params[name]) : mm));
  return s;
}

function buildNotificationMessage(enteredIds, exitingIds, backlog, lang) {
  const n = enteredIds.length;
  const m = exitingIds.length;
  // Backlog should already include both newly-entered and not-yet-exited
  // problems, but max() guards against a backlog read that raced with the
  // delta read (e.g. a problem entered between the two queries).
  const total = Math.max(backlog || 0, n + m);

  const title = total === 1
    ? notifTerm(lang, 'titleOne')
    : notifTerm(lang, 'titleMany', { n: total });
  let body;
  if (n > 0 && m > 0) {
    body = notifTerm(lang, 'bodyBoth', { n, m });
  } else if (m > 0) {
    body = m === 1 ? notifTerm(lang, 'bodyExpiringOne') : notifTerm(lang, 'bodyExpiringMany', { m });
  } else {
    body = notifTerm(lang, 'bodyTap');
  }

  // Never deep-link a specific problem: this message is a snapshot, and by
  // the time the user taps it the problem may already have been reviewed (or
  // left its window). `?recall=1` tells the client to pick a due recall
  // problem itself from its own, fresher scheduler state — and to fall back
  // to normal selection if nothing is due anymore.
  const allIds = [...enteredIds, ...exitingIds];
  const url = '/?recall=1';

  // `count` + `t` drive the client-side countdown: as the user works through
  // the backlog the service worker re-renders the title at the new count and
  // replaces the tray entry in place. The templates travel with the push (with
  // `{n}` left unsubstituted) because only the server holds this string table,
  // and the reminder must count down in the language it was sent in.
  const t = {
    one: notifTerm(lang, 'titleOne'),
    many: notifTerm(lang, 'titleMany'),
    tap: notifTerm(lang, 'bodyTap'),
  };

  return { title, body, url, ids: allIds, count: total, t };
}
