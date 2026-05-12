import { buildPushPayload } from '@block65/webcrypto-web-push';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

const TWO_HOURS_MS = 2 * 3600 * 1000;
const SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000;
// Anti-spam floor between consecutive pushes to the same user. With an hourly
// cron, multiple problems may become due back-to-back; this caps each user at
// roughly four pushes per day in the worst case.
const MIN_NOTIFY_INTERVAL_MS = 6 * 3600 * 1000;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    if (url.pathname === '/report' && request.method === 'POST') {
      return handleReport(request, env);
    }
    if (url.pathname === '/push/vapid-public-key' && request.method === 'GET') {
      return new Response(env.VAPID_PUBLIC_KEY || '', {
        headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' },
      });
    }
    if (url.pathname === '/push/subscribe' && request.method === 'POST') {
      return handleSubscribe(request, env);
    }
    if (url.pathname === '/push/unsubscribe' && request.method === 'POST') {
      return handleUnsubscribe(request, env);
    }
    if (url.pathname === '/schedule' && request.method === 'POST') {
      return handleSchedule(request, env);
    }

    // SPA fallback for navigations only. Static assets are served before
    // the worker; if no asset matches and the request looks like a page
    // navigation (e.g. /9, /13, or `/`), we serve index.html so the URL
    // stays intact for the SPA to read.
    //
    // Restricting to navigations matters: after a rebuild the bundle hash
    // changes, and a stale service-worker cache may request the *old*
    // hashed asset. Falling back to index.html for that request returns
    // HTML where JS is expected and trips strict MIME checking. For asset
    // misses we want a real 404 so the SW invalidates and re-fetches the
    // current HTML.
    if (env.ASSETS && request.method === 'GET') {
      const isNavigation = request.mode === 'navigate' ||
                           request.headers.get('sec-fetch-dest') === 'document' ||
                           (request.headers.get('accept') || '').includes('text/html');
      if (isNavigation) {
        const indexUrl = new URL('/index.html', url);
        return env.ASSETS.fetch(new Request(indexUrl, request));
      }
    }

    return new Response('Not found', { status: 404 });
  },

  async scheduled(event, env, ctx) {
    ctx.waitUntil(runRecallNotifications(env));
  },
};

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
  });
}

async function handleReport(request, env) {
  try {
    const data = await request.json();

    if (!data.userId || !data.problem || typeof data.success !== 'boolean') {
      return new Response('Invalid request body', {
        status: 400,
        headers: { 'Access-Control-Allow-Origin': '*' },
      });
    }

    // Older clients didn't send size; default to 9 to match the historical corpus.
    const size = (data.size === 9 || data.size === 13) ? data.size : 9;

    if (!env.DB) {
      return new Response('Database not configured', {
        status: 500,
        headers: { 'Access-Control-Allow-Origin': '*' },
      });
    }

    try {
      await env.DB.prepare(
        'INSERT INTO tsumego_reports (user_id, problem, success, created_at, size) VALUES (?, ?, ?, ?, ?)'
      )
        .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString(), size)
        .run();

      return new Response('Report saved', {
        status: 201,
        headers: { 'Access-Control-Allow-Origin': '*' },
      });
    } catch (insertError) {
      if (insertError.message && insertError.message.includes('UNIQUE constraint failed')) {
        return new Response('Report already exists', {
          status: 409,
          headers: { 'Access-Control-Allow-Origin': '*' },
        });
      }
      throw insertError;
    }
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', {
      status: 500,
      headers: { 'Access-Control-Allow-Origin': '*' },
    });
  }
}

async function handleSubscribe(request, env) {
  const data = await request.json();
  const sub = data && data.subscription;
  if (!data?.userId || !sub?.endpoint || !sub.keys?.p256dh || !sub.keys?.auth) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }
  const now = Date.now();
  await env.DB.prepare(
    `INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, last_notified_at, created_at, updated_at)
     VALUES (?1, ?2, ?3, ?4, NULL, ?5, ?5)
     ON CONFLICT(user_id) DO UPDATE SET
       endpoint = excluded.endpoint,
       p256dh = excluded.p256dh,
       auth = excluded.auth,
       last_notified_at = NULL,
       updated_at = excluded.updated_at`
  ).bind(data.userId, sub.endpoint, sub.keys.p256dh, sub.keys.auth, now).run();
  return jsonResponse({ ok: true });
}

async function handleUnsubscribe(request, env) {
  const data = await request.json();
  if (!data?.userId) return jsonResponse({ error: 'Invalid body' }, 400);
  await env.DB.prepare('DELETE FROM push_subscriptions WHERE user_id = ?').bind(data.userId).run();
  return jsonResponse({ ok: true });
}

async function handleSchedule(request, env) {
  const data = await request.json();
  if (!data?.userId || !Array.isArray(data.schedule)) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }

  const now = Date.now();
  const statements = [];
  const keptIds = [];

  for (const row of data.schedule) {
    if (!row?.problemId || typeof row.enterAt !== 'number' || typeof row.exitAt !== 'number') continue;
    keptIds.push(row.problemId);
    statements.push(env.DB.prepare(
      `INSERT INTO recall_schedule (user_id, problem_id, enter_at, exit_at, updated_at)
       VALUES (?1, ?2, ?3, ?4, ?5)
       ON CONFLICT(user_id, problem_id) DO UPDATE SET
         enter_at = excluded.enter_at,
         exit_at = excluded.exit_at,
         enter_notified_at = CASE WHEN excluded.enter_at > recall_schedule.enter_at
                                  THEN NULL ELSE recall_schedule.enter_notified_at END,
         exit_notified_at = CASE WHEN excluded.exit_at > recall_schedule.exit_at
                                 THEN NULL ELSE recall_schedule.exit_notified_at END,
         updated_at = excluded.updated_at`
    ).bind(data.userId, row.problemId, row.enterAt, row.exitAt, now));
  }

  // Replace-all semantics: anything previously scheduled for this user but
  // absent from the new upload gets deleted. ON CONFLICT above preserves the
  // notified flags for surviving rows.
  if (keptIds.length === 0) {
    statements.push(env.DB.prepare(
      'DELETE FROM recall_schedule WHERE user_id = ?'
    ).bind(data.userId));
  } else {
    const placeholders = keptIds.map(() => '?').join(',');
    statements.push(env.DB.prepare(
      `DELETE FROM recall_schedule WHERE user_id = ? AND problem_id NOT IN (${placeholders})`
    ).bind(data.userId, ...keptIds));
  }

  if (statements.length > 0) await env.DB.batch(statements);
  return jsonResponse({ ok: true });
}

async function runRecallNotifications(env) {
  if (!env.DB || !env.VAPID_PRIVATE_KEY || !env.VAPID_PUBLIC_KEY) {
    console.error('Recall cron: missing DB or VAPID configuration; skipping.');
    return;
  }

  const now = Date.now();

  const due = await env.DB.prepare(
    `SELECT s.user_id, s.problem_id, s.enter_at, s.exit_at,
            s.enter_notified_at, s.exit_notified_at,
            p.endpoint, p.p256dh, p.auth, p.last_notified_at
     FROM recall_schedule s
     JOIN push_subscriptions p ON p.user_id = s.user_id
     WHERE (s.enter_notified_at IS NULL AND s.enter_at <= ?1)
        OR (s.exit_notified_at  IS NULL AND s.exit_at - ?2 <= ?1 AND s.exit_at > ?1)`
  ).bind(now, TWO_HOURS_MS).all();

  const byUser = new Map();
  for (const row of (due.results || [])) {
    let u = byUser.get(row.user_id);
    if (!u) {
      u = {
        endpoint: row.endpoint,
        p256dh: row.p256dh,
        auth: row.auth,
        lastNotifiedAt: row.last_notified_at,
        enteredIds: [],
        exitingIds: [],
      };
      byUser.set(row.user_id, u);
    }
    if (row.enter_notified_at === null && row.enter_at <= now) {
      u.enteredIds.push(row.problem_id);
    }
    if (row.exit_notified_at === null && row.exit_at - TWO_HOURS_MS <= now && row.exit_at > now) {
      u.exitingIds.push(row.problem_id);
    }
  }

  const vapid = {
    subject: env.VAPID_SUBJECT,
    publicKey: env.VAPID_PUBLIC_KEY,
    privateKey: env.VAPID_PRIVATE_KEY,
  };

  for (const [userId, u] of byUser) {
    if (u.lastNotifiedAt && (now - u.lastNotifiedAt) < MIN_NOTIFY_INTERVAL_MS) continue;
    if (u.enteredIds.length === 0 && u.exitingIds.length === 0) continue;

    const message = buildNotificationMessage(u.enteredIds, u.exitingIds);
    const subscription = {
      endpoint: u.endpoint,
      expirationTime: null,
      keys: { p256dh: u.p256dh, auth: u.auth },
    };

    let res;
    try {
      const init = await buildPushPayload(
        { data: JSON.stringify(message), options: { ttl: 6 * 3600 } },
        subscription,
        vapid,
      );
      res = await fetch(subscription.endpoint, init);
    } catch (err) {
      console.error(`Push send threw for ${userId}:`, err);
      continue;
    }

    if (res.status === 404 || res.status === 410) {
      await env.DB.prepare('DELETE FROM push_subscriptions WHERE user_id = ?').bind(userId).run();
      continue;
    }
    if (!res.ok) {
      console.error(`Push failed for ${userId}: HTTP ${res.status}`);
      continue;
    }

    const updates = [];
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
    updates.push(env.DB.prepare(
      'UPDATE push_subscriptions SET last_notified_at = ? WHERE user_id = ?'
    ).bind(now, userId));
    await env.DB.batch(updates);
  }

  // Purge rows whose window expired long enough ago that no useful
  // notification could still be sent — keeps the table from accumulating
  // dead state for users the client never reached again.
  await env.DB.prepare('DELETE FROM recall_schedule WHERE exit_at < ?')
    .bind(now - SEVEN_DAYS_MS).run();
}

function buildNotificationMessage(enteredIds, exitingIds) {
  const n = enteredIds.length;
  const m = exitingIds.length;

  let title, body;
  if (n > 0 && m > 0) {
    title = 'Time to review';
    body = `${n} new, ${m} expiring soon`;
  } else if (n > 0) {
    title = n === 1 ? '1 problem ready to review' : `${n} problems ready to review`;
    body = 'Tap to open';
  } else {
    title = m === 1 ? '1 problem expiring' : `${m} problems expiring`;
    body = 'Review before you forget';
  }

  // Deep-link to a single problem when there's only one; otherwise drop the
  // user at the app root. The PWA's link handler routes in-window problems
  // through recall mode rather than linked/view mode.
  const allIds = [...enteredIds, ...exitingIds];
  const url = allIds.length === 1 ? `/?id=${allIds[0]}` : '/';

  return { title, body, url, ids: allIds };
}
