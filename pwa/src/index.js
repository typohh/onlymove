import { buildPushPayload } from '@block65/webcrypto-web-push';
import { APP_VERSION } from './version.js';

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Expose-Headers': 'X-App-Version',
};

const TWO_HOURS_MS = 2 * 3600 * 1000;
const SEVEN_DAYS_MS = 7 * 24 * 3600 * 1000;
// Anti-spam floor between consecutive pushes to the same user. With an hourly
// cron, multiple problems may become due back-to-back; this caps each user at
// roughly four pushes per day in the worst case.
const MIN_NOTIFY_INTERVAL_MS = 6 * 3600 * 1000;
// Suppress recall reminders for users who were active since the previous cron
// tick. The client re-uploads the schedule on app load and after every solved
// attempt (stamping recall_schedule.updated_at = now), so a recent updated_at
// means they're already practicing — a "ready to review" push would be noise.
// We don't mark anything notified, so the reminder simply defers to the next
// tick once they've gone idle. Matches the hourly cron in wrangler.toml.
const ACTIVE_SUPPRESS_MS = 1 * 3600 * 1000;

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
    if (url.pathname === '/rank-survey' && request.method === 'POST') {
      return handleRankSurvey(request, env);
    }
    if (url.pathname === '/share-event' && request.method === 'POST') {
      return handleShareEvent(request, env);
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
  // Every response carries the deployed Worker version so the client can
  // detect a stale bundle and trigger a quiescent reload (see
  // js/state.js maybeReload).
  const reportHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Expose-Headers': 'X-App-Version',
    'X-App-Version': APP_VERSION,
  };
  try {
    const data = await request.json();

    if (!data.userId || !data.problem || typeof data.success !== 'boolean') {
      return new Response('Invalid request body', { status: 400, headers: reportHeaders });
    }

    // Older clients didn't send size; default to 9 to match the historical corpus.
    const size = (data.size === 9 || data.size === 13) ? data.size : 9;

    // New per-attempt fields; older clients (pre-rollout) won't send them, so
    // every column is nullable except is_repeat which defaults to 0.
    const move = typeof data.move === 'string' ? data.move : null;
    const timeUsed = Number.isFinite(data.timeUsed) ? Math.round(data.timeUsed) : null;
    const reviewTime = Number.isFinite(data.reviewTime) ? Math.round(data.reviewTime) : null;
    const isRepeat = data.isRepeat ? 1 : 0;

    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers: reportHeaders });
    }

    await env.DB.prepare(
      'INSERT INTO tsumego_reports (user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
    )
      .bind(data.userId, data.problem, data.success ? 1 : 0, new Date().toISOString(), size, move, timeUsed, reviewTime, isRepeat)
      .run();

    return new Response('Report saved', { status: 201, headers: reportHeaders });
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { status: 500, headers: reportHeaders });
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

  for (const row of data.schedule) {
    if (!row?.problemId || typeof row.enterAt !== 'number' || typeof row.exitAt !== 'number') continue;
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
  // notified flags for surviving rows. We identify "untouched" rows by their
  // updated_at watermark — every row in this upload was just stamped with
  // `now`, so anything still carrying an earlier timestamp wasn't in the
  // payload. A NOT IN (?,?,…) version would hit D1's 100-bound-params cap
  // per statement once a user's schedule exceeds ~100 problems.
  statements.push(env.DB.prepare(
    'DELETE FROM recall_schedule WHERE user_id = ? AND updated_at < ?'
  ).bind(data.userId, now));

  if (statements.length > 0) await env.DB.batch(statements);
  return jsonResponse({ ok: true });
}

const SURVEY_SERVERS = ['OGS', 'KGS', 'Fox', 'Tygem', 'WBaduk', 'IGS', 'GoQuest', 'EGF', 'AGA'];
const SURVEY_RANK_RE = /^(?:[1-9]|1\d|2[0-5])k$|^[1-9]d$/;

async function handleRankSurvey(request, env) {
  try {
    const data = await request.json();
    if (!data?.userId || !data.ranks || typeof data.ranks !== 'object') {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    if (!env.DB) {
      return jsonResponse({ error: 'Database not configured' }, 500);
    }

    const createdAt = new Date().toISOString();
    const statements = [];
    for (const server of SURVEY_SERVERS) {
      const rank = data.ranks[server];
      if (typeof rank !== 'string') continue;
      if (!SURVEY_RANK_RE.test(rank)) continue;
      statements.push(env.DB.prepare(
        'INSERT INTO rank_surveys (user_id, server, rank, created_at) VALUES (?, ?, ?, ?)'
      ).bind(data.userId, server, rank, createdAt));
    }

    if (statements.length > 0) await env.DB.batch(statements);
    return jsonResponse({ ok: true, recorded: statements.length });
  } catch (err) {
    console.error('Error handling rank survey:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}

async function handleShareEvent(request, env) {
  try {
    const data = await request.json();
    const type = (data?.type === 'create' || data?.type === 'open') ? data.type : null;
    if (!type) {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    if (!env.DB) {
      return jsonResponse({ error: 'Database not configured' }, 500);
    }

    const size = (data.size === 9 || data.size === 13) ? data.size : null;
    const problemId = typeof data.problemId === 'string' ? data.problemId : null;
    const userId = typeof data.userId === 'string' ? data.userId : null;
    // "Where the link came from": the client sends document.referrer (the Referer
    // of the navigation that opened the shared link) for opens; creates send
    // none. We don't fall back to this POST's own Referer — that's just the app
    // origin and would be noise. Capped defensively.
    const referrer = (typeof data.referrer === 'string' && data.referrer)
      ? data.referrer.slice(0, 500) : null;
    // Country is Cloudflare's edge geo-IP, not something the client can report.
    const country = (request.cf && request.cf.country) ? request.cf.country : null;

    await env.DB.prepare(
      'INSERT INTO share_events (event_type, user_id, problem_id, size, country, referrer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
    )
      .bind(type, userId, problemId, size, country, referrer, new Date().toISOString())
      .run();

    return jsonResponse({ ok: true });
  } catch (err) {
    console.error('Error handling share event:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}

async function runRecallNotifications(env) {
  if (!env.DB || !env.VAPID_PRIVATE_KEY || !env.VAPID_PUBLIC_KEY) {
    console.error('Recall cron: missing DB or VAPID configuration; skipping.');
    return;
  }

  const now = Date.now();

  const due = await env.DB.prepare(
    `SELECT s.user_id, s.problem_id, s.enter_at, s.exit_at,
            s.enter_notified_at, s.exit_notified_at, s.updated_at,
            p.endpoint, p.p256dh, p.auth, p.last_notified_at
     FROM recall_schedule s
     JOIN push_subscriptions p ON p.user_id = s.user_id
     WHERE (s.enter_notified_at IS NULL AND s.enter_at <= ?1)
        OR (s.exit_notified_at  IS NULL AND s.exit_at - ?2 <= ?1 AND s.exit_at > ?1)`
  ).bind(now, TWO_HOURS_MS).all();

  // Full per-user backlog of problems currently inside the review window,
  // regardless of whether each individual row has already been notified.
  // Used to render "N problems ready to review" rather than the
  // delta-since-last-cron-tick — the user wants to see their total pending
  // load, not just what newly entered in the last hour.
  const backlog = await env.DB.prepare(
    `SELECT user_id, COUNT(*) AS n
     FROM recall_schedule
     WHERE enter_at <= ?1 AND exit_at > ?1
     GROUP BY user_id`
  ).bind(now).all();
  const backlogByUser = new Map();
  for (const row of (backlog.results || [])) backlogByUser.set(row.user_id, row.n);

  const byUser = new Map();
  for (const row of (due.results || [])) {
    let u = byUser.get(row.user_id);
    if (!u) {
      u = {
        endpoint: row.endpoint,
        p256dh: row.p256dh,
        auth: row.auth,
        lastNotifiedAt: row.last_notified_at,
        lastActiveAt: 0,
        enteredIds: [],
        exitingIds: [],
      };
      byUser.set(row.user_id, u);
    }
    // After any /schedule upload all of a user's surviving rows share the same
    // updated_at, but take the max defensively in case a read races an upload.
    u.lastActiveAt = Math.max(u.lastActiveAt, row.updated_at || 0);
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
    // Active since the previous tick — they're already in the app, so defer the
    // reminder to a later tick rather than nagging mid-session.
    if (u.lastActiveAt && (now - u.lastActiveAt) < ACTIVE_SUPPRESS_MS) continue;
    if (u.enteredIds.length === 0 && u.exitingIds.length === 0) continue;

    const message = buildNotificationMessage(u.enteredIds, u.exitingIds, backlogByUser.get(userId) || 0);
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

function buildNotificationMessage(enteredIds, exitingIds, backlog) {
  const n = enteredIds.length;
  const m = exitingIds.length;
  // Backlog should already include both newly-entered and not-yet-exited
  // problems, but max() guards against a backlog read that raced with the
  // delta read (e.g. a problem entered between the two queries).
  const total = Math.max(backlog || 0, n + m);

  const title = total === 1 ? '1 problem ready to review' : `${total} problems ready to review`;
  let body;
  if (n > 0 && m > 0) {
    body = `${n} new, ${m} expiring soon`;
  } else if (m > 0) {
    body = m === 1 ? '1 expiring soon' : `${m} expiring soon`;
  } else {
    body = 'Tap to open';
  }

  // Deep-link to a single problem when there's only one; otherwise drop the
  // user at the app root. The PWA's link handler routes in-window problems
  // through recall mode rather than linked/view mode.
  const allIds = [...enteredIds, ...exitingIds];
  const url = allIds.length === 1 ? `/?id=${allIds[0]}` : '/';

  return { title, body, url, ids: allIds };
}
