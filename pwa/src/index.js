import { buildPushPayload } from '@block65/webcrypto-web-push';
import { APP_VERSION } from './version.js';
import { interpExtrap, rankScalarToLabel } from '../shared/rank-math.js';
// M (rating→rank curve) is inlined at build time — same source the client bundles
// (pwa/tsumegos/rating_to_rank.json) — so the Community rank histogram needs no
// asset fetch for it. Other tsumego assets (lookup.txt, metadata.json) stay served.
import RATING_RANK_MAP from '../tsumegos/rating_to_rank.json';

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
// tick. Every /schedule call (load / tab-hidden / heartbeat, even an empty
// diff) bumps push_subscriptions.last_seen_at, so a recent watermark means
// they're already practicing — a "ready to review" push would be noise.
// (recall_schedule.updated_at doubles as the watermark for legacy replace-all
// clients.) We don't mark anything notified, so the reminder simply defers to
// the next tick once they've gone idle. Matches the hourly cron in wrangler.toml.
const ACTIVE_SUPPRESS_MS = 1 * 3600 * 1000;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // Per-IP rate limiting on the write endpoints, before any handler touches
    // D1. /report is high-volume (one per solved move) so it gets a generous
    // dedicated limiter; the rest are low-frequency and share a tighter one.
    if (url.pathname === '/report' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_REPORT')) return tooManyRequests();
      return handleReport(request, env);
    }
    if (url.pathname === '/sync' && request.method === 'POST') {
      // A read, not a write — but it's per-user-keyed and quota-relevant
      // (rows read), so it shares the low-frequency limiter with the writes.
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleSync(request, env);
    }
    if (url.pathname === '/report-backfill' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleReportBackfill(request, env);
    }
    if (url.pathname === '/report-enrich' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleReportEnrich(request, env);
    }
    if (url.pathname === '/push/vapid-public-key' && request.method === 'GET') {
      return new Response(env.VAPID_PUBLIC_KEY || '', {
        headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' },
      });
    }
    if (url.pathname === '/push/subscribe' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleSubscribe(request, env);
    }
    if (url.pathname === '/push/unsubscribe' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleUnsubscribe(request, env);
    }
    if (url.pathname === '/schedule' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleSchedule(request, env);
    }
    if (url.pathname === '/rank-survey' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleRankSurvey(request, env);
    }
    if (url.pathname === '/share-event' && request.method === 'POST') {
      if (await rateLimited(request, env, 'RL_WRITE')) return tooManyRequests();
      return handleShareEvent(request, env);
    }
    if (url.pathname === '/server-stats' && request.method === 'GET') {
      return handleServerStats(env);
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
    ctx.waitUntil(maybeComputeServerStats(env));
    ctx.waitUntil(pruneHistoryRefreshLog(env));
  },
};

// Drop history_refresh_log rows older than a small window — only the current
// UTC day's counts matter for the cap, so anything older is dead weight.
async function pruneHistoryRefreshLog(env) {
  if (!env.DB) return;
  try {
    const cutoff = new Date(Date.now() - 2 * 24 * 3600 * 1000).toISOString().slice(0, 10);
    await env.DB.prepare(`DELETE FROM history_refresh_log WHERE day < ?`).bind(cutoff).run();
  } catch (e) {
    console.error('pruneHistoryRefreshLog failed:', e);
  }
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
  });
}

// --- Abuse / D1-quota defenses ----------------------------------------------
// Two cheap, in-Worker layers guard every D1 write (see also the outer
// Cloudflare WAF rate-limit rule documented in wrangler.toml):
//   1. A per-IP rate limit, checked BEFORE any D1 work, so a flood is dropped by
//      a cheap Worker invocation instead of spending D1 write quota (and a
//      spoofed row would also keep inflating the server-stats window scan for
//      7 days — read quota — until purged).
//   2. In-memory format validation (no D1 reads) so malformed/garbage writes
//      never reach an INSERT. Deliberately format-only: we don't verify the
//      problem id against the real corpus (that needs a bundled set); the rate
//      limiter bounds the volume of valid-looking spoofs.

// Tsumego ids are the symmetry-canonical hash truncated to 8 hex chars
// (see get_orientation_invariant_hash). user_id is a short client-generated
// token (observed 8 chars, a few legacy ids up to 23); bound it generously.
const PROBLEM_ID_RE = /^[0-9a-f]{8}$/;
const USER_ID_RE = /^[A-Za-z0-9_-]{1,40}$/;
const isValidProblemId = (v) => typeof v === 'string' && PROBLEM_ID_RE.test(v);
const isValidUserId = (v) => typeof v === 'string' && USER_ID_RE.test(v);

// Per-IP rate limit via a Workers ratelimit binding. Returns true when the
// caller should be rejected. Fails OPEN on a missing binding (e.g. some local
// dev) or a limiter error, so a misconfiguration never drops legitimate writes.
async function rateLimited(request, env, binding) {
  const limiter = env[binding];
  if (!limiter) return false;
  try {
    const key = request.headers.get('CF-Connecting-IP') || 'unknown';
    const { success } = await limiter.limit({ key });
    return !success;
  } catch {
    return false;
  }
}

function tooManyRequests() {
  return new Response('Too many requests', {
    status: 429,
    headers: { ...CORS_HEADERS, 'Retry-After': '60' },
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

    // Identity fields must be well-formed: an 8-hex tsumego id and a short
    // user token. Rejecting here (no D1 read) keeps garbage/spoofed ids from
    // ever reaching an INSERT. (A deploy smoke test must therefore post a
    // real 8-hex problem id, not a free-form marker.)
    if (!isValidUserId(data.userId) || !isValidProblemId(data.problem) ||
        typeof data.success !== 'boolean') {
      return new Response('Invalid request body', { status: 400, headers: reportHeaders });
    }

    // Older clients didn't send size; default to 9 to match the historical corpus.
    const size = (data.size === 9 || data.size === 13) ? data.size : 9;

    // New per-attempt fields; older clients (pre-rollout) won't send them, so
    // every column is nullable except is_repeat which defaults to 0. move is a
    // GTP coordinate (≤3 chars) or "pass"; bound its length so a malformed
    // payload can't store an unbounded blob. Times must be non-negative;
    // idle-tab outliers (hours) are left intact — the stats layer caps them.
    const move = (typeof data.move === 'string' && data.move.length <= 4) ? data.move : null;
    const timeUsed = (Number.isFinite(data.timeUsed) && data.timeUsed >= 0) ? Math.round(data.timeUsed) : null;
    const reviewTime = (Number.isFinite(data.reviewTime) && data.reviewTime >= 0) ? Math.round(data.reviewTime) : null;
    const isRepeat = data.isRepeat ? 1 : 0;

    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers: reportHeaders });
    }

    // The server-stamped timestamp is echoed back so the client can store it
    // on the matching answerHistory entry (serverCreatedAt) — that's how a
    // /sync pull recognizes the device's own rows instead of re-merging them.
    //
    // prevCreatedAt — the user's latest report BEFORE this one — is the
    // account-sync piggyback signal: a device that doesn't recognize the
    // timestamp knows another device wrote it and pulls, so active sessions
    // need no sync polling at all. The SELECT rides the same batch round trip
    // and reads ~1 row via idx_tsumego_reports_user_created.
    const createdAt = new Date().toISOString();
    const [prevRes] = await env.DB.batch([
      env.DB.prepare(
        'SELECT max(created_at) AS last FROM tsumego_reports WHERE user_id = ?'
      ).bind(data.userId),
      env.DB.prepare(
        'INSERT INTO tsumego_reports (user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'
      ).bind(data.userId, data.problem, data.success ? 1 : 0, createdAt, size, move, timeUsed, reviewTime, isRepeat),
    ]);
    const prevCreatedAt = (prevRes && prevRes.results && prevRes.results[0])
      ? (prevRes.results[0].last ?? null)
      : null;

    return new Response(JSON.stringify({ createdAt, prevCreatedAt }), {
      status: 201,
      headers: { ...reportHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { status: 500, headers: reportHeaders });
  }
}

// --- Account sync ------------------------------------------------------------
// Returns this user's reports newer than the client's cursor, ordered by
// created_at, so a device can rebuild/merge its local history (multi-device
// accounts: the account credential IS the userId). Reports are append-only, so
// a created_at cursor fully captures "what changed since last sync" — no
// content hash needed. The query REQUIRES idx_tsumego_reports_user_created:
// with it an up-to-date device's poll reads ~0 rows; without it every poll is
// a scan (the read-quota failure mode the server-stats index fixed).
const SYNC_PAGE_LIMIT = 1000;
// Server-enforced cap on full-history pulls (POST /sync with since='' — the
// from-the-beginning scan) per user per UTC day: the stats eviction-recovery
// refresh, account adoption, and the backfill/enrich reconciles all count.
// Each is a full per-user table scan; client-side counting is bypassable by
// clearing localStorage, so the cap lives here. Incremental syncs (real cursor)
// and existence probes (limit:1) are not full pulls and don't count.
const HISTORY_REFRESH_DAILY_CAP = 3;

async function handleSync(request, env) {
  const syncHeaders = {
    ...CORS_HEADERS,
    'Content-Type': 'application/json',
    'X-App-Version': APP_VERSION,
  };
  try {
    const data = await request.json();
    if (!isValidUserId(data.userId)) {
      return new Response('Invalid request body', { status: 400, headers: syncHeaders });
    }
    // Cursor is an ISO created_at string ('' = from the beginning). Bound its
    // length; anything malformed simply matches nothing or everything — the
    // merge on the client is idempotent either way.
    const since = (typeof data.since === 'string' && data.since.length <= 40) ? data.since : '';
    // Optional smaller page, used as a cheap existence probe (limit: 1 reads
    // ~1 row instead of a full page). NB: paging onward from a small-limit
    // response can skip timestamp ties past the cut — fine for probes, which
    // discard the cursor; real pulls use the default.
    const limit = (Number.isInteger(data.limit) && data.limit > 0)
      ? Math.min(data.limit, SYNC_PAGE_LIMIT)
      : SYNC_PAGE_LIMIT;

    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers: syncHeaders });
    }

    // Full-history pull (since the beginning): the expensive full per-user scan.
    // ALL of them are capped per UTC day server-side — the stats eviction-recovery
    // refresh, account adoption (switchAccount resets the cursor to ''), and the
    // one-off backfill/enrich reconciles all land here. Capping the query itself
    // (since === '') rather than a client flag means it can't be bypassed: a
    // capped adoption that leaves the cursor at '' can't then sneak a free full
    // scan through the next "normal" pull. Probes (limit:1) and incremental
    // syncs (since = a real cursor) are not full pulls and skip this. Page 2+ of
    // a multi-page full pull carries a real cursor, so one pull counts once.
    const isFullPull = since === '' && limit > 1;
    let refreshesLeft;
    if (isFullPull) {
      const day = new Date().toISOString().slice(0, 10);
      const row = await env.DB.prepare(
        `SELECT count FROM history_refresh_log WHERE user_id = ? AND day = ?`
      ).bind(data.userId, day).first();
      const used = row ? row.count : 0;
      if (used >= HISTORY_REFRESH_DAILY_CAP) {
        return new Response(
          JSON.stringify({ reports: [], cursor: since, hasMore: false, capExceeded: true, refreshesLeft: 0 }),
          { status: 200, headers: syncHeaders }
        );
      }
      await env.DB.prepare(
        `INSERT INTO history_refresh_log (user_id, day, count) VALUES (?, ?, 1)
         ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1`
      ).bind(data.userId, day).run();
      refreshesLeft = HISTORY_REFRESH_DAILY_CAP - (used + 1);
    }

    // Fetch one row past the page limit to detect truncation.
    const { results } = await env.DB.prepare(
      `SELECT problem, success, size, move, time_used, review_time, is_repeat, created_at
       FROM tsumego_reports WHERE user_id = ? AND created_at > ?
       ORDER BY created_at LIMIT ?`
    ).bind(data.userId, since, limit + 1).all();

    let rows = results || [];
    let hasMore = false;
    if (rows.length > limit) {
      hasMore = true;
      rows = rows.slice(0, limit);
      // The cursor is exclusive (`> created_at`), so a page must not end in
      // the middle of a run of equal timestamps — the next page would skip the
      // run's remainder. Cut back to the last timestamp change; if the whole
      // page somehow shares one timestamp (pathological), keep it as-is.
      const boundary = rows[rows.length - 1].created_at;
      let cut = rows.length;
      while (cut > 0 && rows[cut - 1].created_at === boundary) cut--;
      if (cut > 0) rows = rows.slice(0, cut);
    }

    const reports = rows.map((r) => ({
      problem: r.problem,
      success: !!r.success,
      size: r.size,
      move: r.move,
      timeUsed: r.time_used,
      reviewTime: r.review_time,
      isRepeat: !!r.is_repeat,
      createdAt: r.created_at,
    }));
    const cursor = rows.length > 0 ? rows[rows.length - 1].created_at : since;
    const payload = { reports, cursor, hasMore };
    if (refreshesLeft !== undefined) payload.refreshesLeft = refreshesLeft;
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: syncHeaders,
    });
  } catch (error) {
    console.error('Error handling sync:', error);
    return new Response('Internal server error', { status: 500, headers: syncHeaders });
  }
}

// --- One-off historical report backfill (TEMPORARY — remove after rollout) ---
// Before 2026-05-15 clients reported only one row per problem (no repeats, no
// move/time fields), so the server log is missing ~36% of real attempts that
// still live in clients' localStorage. account-sync.maybeBackfillLocalHistory
// uploads those once per device: it first pulls the full server history and
// fuzzy-claims matches, then posts the provably-absent entries here WITH THEIR
// CLIENT TIMESTAMPS (created_at far in the past, so the rows never enter the
// server-stats 7-day window). The client staggers itself by hash(userId) over
// 14 days; fleet-wide volume ≈ 49k rows total vs the ~3.2k/day live write
// rate (measured June 2026). Once actives have run it (~2 stagger cycles,
// say August 2026), delete this handler, the route, and the client one-off.
const BACKFILL_MAX_BATCH = 500;
// Backfilled timestamps must be historical: after the site could plausibly
// have data, and old enough not to collide with live reporting.
const BACKFILL_MIN_CREATED = '2024-01-01T00:00:00.000Z';

async function handleReportBackfill(request, env) {
  const headers = {
    ...CORS_HEADERS,
    'Content-Type': 'application/json',
    'X-App-Version': APP_VERSION,
  };
  try {
    const data = await request.json();
    if (!isValidUserId(data.userId) || !Array.isArray(data.reports) ||
        data.reports.length === 0 || data.reports.length > BACKFILL_MAX_BATCH) {
      return new Response('Invalid request body', { status: 400, headers });
    }
    const maxCreated = new Date(Date.now() - 3600 * 1000).toISOString();
    const stmts = [];
    for (const r of data.reports) {
      const createdAt = (typeof r.createdAt === 'string' &&
          /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(r.createdAt) &&
          r.createdAt > BACKFILL_MIN_CREATED && r.createdAt < maxCreated)
        ? r.createdAt : null;
      if (!isValidProblemId(r.problem) || typeof r.success !== 'boolean' || !createdAt) {
        return new Response('Invalid request body', { status: 400, headers });
      }
      const size = (r.size === 9 || r.size === 13) ? r.size : 9;
      const timeUsed = (Number.isFinite(r.timeUsed) && r.timeUsed >= 0) ? Math.round(r.timeUsed) : null;
      const reviewTime = (Number.isFinite(r.reviewTime) && r.reviewTime >= 0) ? Math.round(r.reviewTime) : null;
      stmts.push(env.DB.prepare(
        'INSERT INTO tsumego_reports (user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)'
      ).bind(data.userId, r.problem, r.success ? 1 : 0, createdAt, size, timeUsed, reviewTime, r.isRepeat ? 1 : 0));
    }
    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers });
    }
    await env.DB.batch(stmts);
    return new Response(JSON.stringify({ inserted: stmts.length }), { status: 201, headers });
  } catch (error) {
    console.error('Error handling report backfill:', error);
    return new Response('Internal server error', { status: 500, headers });
  }
}

// --- One-off historical detail enrichment (TEMPORARY — remove after rollout) --
// /report-backfill (round 1) was INSERT-only: it added the repeat attempts the
// server was missing but left the pre-era skeleton rows (one-per-problem, from
// the legacy one-row-per-problem era) without time_used/review_time, because the
// reconcile CLAIMED those rows (fuzzy-matched, not re-uploaded) and there was no
// UPDATE path. This fills them: the client soft-matches its detail-bearing local
// entries to the server rows on a fresh full pull (same proven matcher as round
// 1) and posts (problem, created_at, time, review) keyed on the AUTHORITATIVE
// pulled created_at — never a stored local field. COALESCE + the NULL guard make
// every write fill-only (it can't overwrite a real value) and idempotent; the
// created_at is months old, so the row's server-stats window membership and its
// /sync cursor position are both unchanged. Same staggered ramp + RL_WRITE limit
// as round 1. Once actives have run it, delete this handler, the route, and the
// client one-off (account-sync.maybeEnrichServerHistory).
const ENRICH_MAX_BATCH = 500;

async function handleReportEnrich(request, env) {
  const headers = {
    ...CORS_HEADERS,
    'Content-Type': 'application/json',
    'X-App-Version': APP_VERSION,
  };
  try {
    const data = await request.json();
    if (!isValidUserId(data.userId) || !Array.isArray(data.updates) ||
        data.updates.length === 0 || data.updates.length > ENRICH_MAX_BATCH) {
      return new Response('Invalid request body', { status: 400, headers });
    }
    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers });
    }
    // Only ever touch historical rows (old enough not to be in flight), so a
    // malformed/spoofed payload can't disturb live reporting.
    const maxCreated = new Date(Date.now() - 3600 * 1000).toISOString();
    const stmts = [];
    for (const u of data.updates) {
      const createdAt = (typeof u.createdAt === 'string' &&
          /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(u.createdAt) &&
          u.createdAt > BACKFILL_MIN_CREATED && u.createdAt < maxCreated)
        ? u.createdAt : null;
      const timeUsed = (Number.isFinite(u.timeUsed) && u.timeUsed >= 0) ? Math.round(u.timeUsed) : null;
      const reviewTime = (Number.isFinite(u.reviewTime) && u.reviewTime >= 0) ? Math.round(u.reviewTime) : null;
      // Nothing to fill ⇒ reject the whole batch (the client never sends such a
      // row; a payload that does is malformed).
      if (!isValidProblemId(u.problem) || !createdAt || (timeUsed === null && reviewTime === null)) {
        return new Response('Invalid request body', { status: 400, headers });
      }
      stmts.push(env.DB.prepare(
        `UPDATE tsumego_reports
            SET time_used = COALESCE(time_used, ?), review_time = COALESCE(review_time, ?)
          WHERE user_id = ? AND problem = ? AND created_at = ?
            AND (time_used IS NULL OR review_time IS NULL)`
      ).bind(timeUsed, reviewTime, data.userId, u.problem, createdAt));
    }
    const results = await env.DB.batch(stmts);
    let updated = 0;
    for (const r of results) updated += (r && r.meta && Number.isFinite(r.meta.changes)) ? r.meta.changes : 0;
    return new Response(JSON.stringify({ updated }), { status: 200, headers });
  } catch (error) {
    console.error('Error handling report enrich:', error);
    return new Response('Internal server error', { status: 500, headers });
  }
}

async function handleSubscribe(request, env) {
  const data = await request.json();
  const sub = data && data.subscription;
  if (!isValidUserId(data?.userId) || !sub?.endpoint || !sub.keys?.p256dh || !sub.keys?.auth) {
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
  if (!isValidUserId(data?.userId)) return jsonResponse({ error: 'Invalid body' }, 400);
  await env.DB.prepare('DELETE FROM push_subscriptions WHERE user_id = ?').bind(data.userId).run();
  return jsonResponse({ ok: true });
}

// /schedule accepts two payload shapes:
//   { userId, schedule: [...] }            — LEGACY replace-all (older clients;
//     every row upserted, then untouched rows swept by updated_at watermark).
//   { userId, upserts: [...], removals: [...] } — diff mode: the client keeps a
//     per-row snapshot of what it last uploaded and sends only rows whose
//     hour-rounded window moved, plus explicit removals. Each upsert bills the
//     row + its two partial indexes (~3 write units), so the diff cuts the
//     dominant D1 write consumer to the handful of rows an attempt actually
//     changes. An EMPTY diff is still a valid call — it lands only the
//     activity-watermark bump below.
// Either way the call bumps push_subscriptions.last_seen_at (one row, no-op
// for users without a subscription): the cron's mid-session suppression reads
// it, since with diff uploads recall_schedule.updated_at no longer refreshes
// across the board on every upload.
const SCHEDULE_MAX_ROWS = 1000;

async function handleSchedule(request, env) {
  const data = await request.json();
  if (!isValidUserId(data?.userId)) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }
  const legacy = Array.isArray(data.schedule);
  const diff = Array.isArray(data.upserts) && Array.isArray(data.removals);
  if (!legacy && !diff) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }

  const now = Date.now();
  const statements = [];
  const pushUpsert = (row) => {
    if (!isValidProblemId(row?.problemId) || typeof row.enterAt !== 'number' || typeof row.exitAt !== 'number') return;
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
  };

  if (legacy) {
    if (data.schedule.length > SCHEDULE_MAX_ROWS) {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    for (const row of data.schedule) pushUpsert(row);

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
  } else {
    if (data.upserts.length > SCHEDULE_MAX_ROWS || data.removals.length > SCHEDULE_MAX_ROWS) {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    for (const row of data.upserts) pushUpsert(row);
    // Explicit removals, chunked under D1's ~100-bound-params-per-statement cap.
    const ids = data.removals.filter(isValidProblemId);
    for (let i = 0; i < ids.length; i += 50) {
      const chunk = ids.slice(i, i + 50);
      statements.push(env.DB.prepare(
        `DELETE FROM recall_schedule WHERE user_id = ? AND problem_id IN (${chunk.map(() => '?').join(',')})`
      ).bind(data.userId, ...chunk));
    }
  }

  statements.push(env.DB.prepare(
    'UPDATE push_subscriptions SET last_seen_at = ? WHERE user_id = ?'
  ).bind(now, data.userId));

  await env.DB.batch(statements);
  return jsonResponse({ ok: true });
}

const SURVEY_SERVERS = ['OGS', 'KGS', 'Fox', 'Tygem', 'WBaduk', 'IGS', 'GoQuest', 'EGF', 'AGA'];
const SURVEY_RANK_RE = /^(?:[1-9]|1\d|2[0-5])k$|^[1-9]d$/;

async function handleRankSurvey(request, env) {
  try {
    const data = await request.json();
    if (!isValidUserId(data?.userId) || !data.ranks || typeof data.ranks !== 'object') {
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
    const problemId = isValidProblemId(data.problemId) ? data.problemId : null;
    const userId = isValidUserId(data.userId) ? data.userId : null;
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

// Serve the cached server-wide stats. Lazily (re)computes when nothing is
// cached yet OR the cache was produced by an older code version (so a deploy
// that changes the computation refreshes on the next read, not 24h later); the
// cron keeps it fresh by age thereafter.
async function handleServerStats(env) {
  try {
    if (!env.DB) return jsonResponse({ error: 'Database not configured' }, 500);
    const row = await env.DB.prepare(`SELECT data FROM server_stats WHERE key = 'latest'`).first();
    let stale = true;
    if (row && row.data) {
      try { stale = JSON.parse(row.data).version !== APP_VERSION; } catch { stale = true; }
    }
    const body = stale ? JSON.stringify(await computeServerStats(env)) : row.data;
    return new Response(body, {
      status: 200,
      // The worker already serves from a D1 cache and manages freshness (daily
      // cron + version invalidation); an edge/browser HTTP cache on top would
      // serve stale data and bypass that logic, so opt out of caching here.
      headers: { ...CORS_HEADERS, 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch (err) {
    console.error('Error serving server stats:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}

// Recompute when the cache is older than ~23h (the cron is hourly, so this fires
// once a day with slack for tick jitter) OR when it was produced by an older
// code version (so a logic change is picked up on the next tick).
async function maybeComputeServerStats(env) {
  if (!env.DB) return;
  try {
    const row = await env.DB.prepare(`SELECT data, computed_at FROM server_stats WHERE key = 'latest'`).first();
    let staleVersion = true;
    if (row && row.data) {
      try { staleVersion = JSON.parse(row.data).version !== APP_VERSION; } catch { staleVersion = true; }
    }
    const last = row?.computed_at ? Date.parse(row.computed_at) : 0;
    if (staleVersion || Date.now() - last >= 23 * 3600 * 1000) {
      await computeServerStats(env);
    }
  } catch (err) {
    console.error('Error computing server stats:', err);
  }
}


// Aggregate the preceding 7 days into the cached JSON blob the PWA renders.
// Everything is a cheap GROUP BY; time histograms cap at 600s (5s buckets) so
// idle-tab outliers don't blow out the distribution.
async function computeServerStats(env) {
  const since = new Date(Date.now() - SEVEN_DAYS_MS).toISOString();

  // ONE pass over the last 7 days, aggregated in JS. idx_tsumego_reports_created
  // turns `created_at >= ?` into a range scan (reads only the window's rows, not
  // the whole table), and sharing a single SELECT across every metric keeps D1
  // rows-read to roughly one window's worth per recompute.
  const rows = (await env.DB.prepare(
    `SELECT user_id, problem, success, size, time_used, review_time, is_repeat, created_at, substr(created_at, 1, 10) AS day
     FROM tsumego_reports WHERE created_at >= ?`
  ).bind(since).all()).results || [];

  const usersAll = new Set();
  let attemptsAll = 0, winsAll = 0;
  const byDay = new Map();     // day → { attempts, users:Set }
  const bySizeMap = new Map(); // size → { attempts, wins, users:Set }
  // Solving/review time is summarised PER USER (one sample per user) rather than
  // per attempt, so heavy users don't dominate the distribution. Solving time is
  // per-attempt, so we keep every sample. Review time is only meaningful on
  // REPEAT (review) attempts, so we gate on is_repeat. It is also CUMULATIVE
  // across a problem's repeats (each repeat re-reports the running total), so per
  // (user, problem) we keep only the last instance — the final cumulative value —
  // and earlier repeats are dropped to avoid massive double counting.
  const userSolve = new Map();   // user → number[] of time_used (ms)
  const userReview = new Map();  // user → Map<problem, { t, at }> last cumulative review_time
  const perUserActivity = new Map(); // user → { attempts, days:Set }
  // Recall time: how soon a user comes back to a problem they FAILED. Collect
  // each (user, problem)'s attempt timeline; below we measure the gap from a
  // failure to the very next attempt on that problem (the "first recall after a
  // failure"). Recalls following a SUCCESS are excluded — their interval is set
  // by the spaced-repetition schedule (it lengthens with each success), so it
  // isn't comparable across users; restricting to post-failure recalls
  // standardizes the measure.
  const userProblemSeq = new Map(); // user → Map<problem, [{ t, success }]>
  // Answer-derived rank: per user, the set of DISTINCT first-time problems they
  // attempted (with board size). Problems are served difficulty-matched (~80%
  // clear) near the user's rank, so the mean internal rank of these problems
  // approximates the user's own rank — coarse but enough for the histogram.
  // Repeats (recall) are excluded: they're re-shown by the spaced-repetition
  // schedule, not freshly difficulty-matched. neededProblems is the union we
  // must resolve to a rating below.
  const userFirstProblems = new Map(); // user → Map<problem, size>
  const neededProblems = new Set();    // every distinct first-time problem id

  for (const r of rows) {
    attemptsAll++;
    if (r.success) winsAll++;
    usersAll.add(r.user_id);

    let d = byDay.get(r.day);
    if (!d) { d = { attempts: 0, users: new Set() }; byDay.set(r.day, d); }
    d.attempts++; d.users.add(r.user_id);

    let s = bySizeMap.get(r.size);
    if (!s) { s = { attempts: 0, wins: 0, users: new Set() }; bySizeMap.set(r.size, s); }
    s.attempts++; if (r.success) s.wins++; s.users.add(r.user_id);

    if (r.time_used != null && r.time_used > 0) {
      let arr = userSolve.get(r.user_id);
      if (!arr) { arr = []; userSolve.set(r.user_id, arr); }
      arr.push(r.time_used);
    }
    if (r.is_repeat && r.review_time != null && r.review_time > 0) {
      let m = userReview.get(r.user_id);
      if (!m) { m = new Map(); userReview.set(r.user_id, m); }
      const prev = m.get(r.problem);
      if (!prev || r.created_at > prev.at) m.set(r.problem, { t: r.review_time, at: r.created_at });
    }

    let u = perUserActivity.get(r.user_id);
    if (!u) { u = { attempts: 0, days: new Set() }; perUserActivity.set(r.user_id, u); }
    u.attempts++; u.days.add(r.day);

    const ts = Date.parse(r.created_at);
    if (Number.isFinite(ts)) {
      let pm = userProblemSeq.get(r.user_id);
      if (!pm) { pm = new Map(); userProblemSeq.set(r.user_id, pm); }
      let seq = pm.get(r.problem);
      if (!seq) { seq = []; pm.set(r.problem, seq); }
      seq.push({ t: ts, success: !!r.success });
    }

    if (!r.is_repeat) {
      let pm = userFirstProblems.get(r.user_id);
      if (!pm) { pm = new Map(); userFirstProblems.set(r.user_id, pm); }
      if (!pm.has(r.problem)) pm.set(r.problem, r.size);
      neededProblems.add(r.problem);
    }
  }

  // Per-user median time → fine bucket; the histogram then counts USERS per
  // bucket, so binHist (client) reads it with "users" as the y-axis. Solving
  // time is log-normal and most of the mass sits under ~30s, so a coarse 10s
  // bucket lumped the whole hump into one bar; SOLVE_BUCKET_SEC=5 keeps the
  // low-end detail (bucketSec ships in the blob so the client can decode it,
  // and any future width change stays self-describing). Capped at 600s.
  const SOLVE_BUCKET_SEC = 5;
  const SOLVE_BUCKET_CAP = Math.round(600 / SOLVE_BUCKET_SEC);
  const median = (arr) => {
    if (!arr.length) return null;
    const s = arr.slice().sort((a, b) => a - b);
    const mid = s.length >> 1;
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  };
  const timeBucket = (ms) => Math.min(Math.floor(ms / (SOLVE_BUCKET_SEC * 1000)), SOLVE_BUCKET_CAP);
  const solveB = new Map();    // 10s bucket → user count
  const reviewB = new Map();
  for (const arr of userSolve.values()) {
    const med = median(arr);
    if (med == null) continue;
    const b = timeBucket(med);
    solveB.set(b, (solveB.get(b) || 0) + 1);
  }
  for (const perProblem of userReview.values()) {
    const med = median([...perProblem.values()].map(v => v.t));
    if (med == null) continue;
    const b = timeBucket(med);
    reviewB.set(b, (reviewB.get(b) || 0) + 1);
  }

  // Recall time, one sample per user: the SHORTEST failure→next-attempt gap
  // across all their problems. Pre-binned server-side (the spans run minutes to
  // days, so the fine solving/review buckets don't apply) and rendered with
  // makeBarList like attemptsPerUserDist. Only gaps where the earlier attempt
  // FAILED count (see userProblemSeq note); the 7-day window naturally bounds
  // how long a gap we can observe, which is fine since we take the minimum.
  // Gaps under 12h are ignored entirely (too short to be a meaningful recall
  // interval), so the per-user value is the shortest failure→next-attempt gap
  // that is at least 12h.
  const RECALL_MIN_MS = 12 * 3600e3;
  const RECALL_BINS = [
    { hi: 24 * 3600e3, label: '<1d' },
    { hi: 48 * 3600e3, label: '1–2d' },
    { hi: 72 * 3600e3, label: '2–3d' },
    { hi: 120 * 3600e3, label: '3–5d' },
    { hi: Infinity, label: '5d+' },
  ];
  const recallDist = RECALL_BINS.map(b => ({ label: b.label, value: 0 }));
  for (const pm of userProblemSeq.values()) {
    let userMin = Infinity;
    for (const seq of pm.values()) {
      if (seq.length < 2) continue;
      seq.sort((a, b) => a.t - b.t);
      for (let i = 1; i < seq.length; i++) {
        if (seq[i - 1].success) continue;           // recall after a success — skip
        const gap = seq[i].t - seq[i - 1].t;
        if (gap >= RECALL_MIN_MS && gap < userMin) userMin = gap;
      }
    }
    if (userMin < Infinity) {
      const bi = RECALL_BINS.findIndex(b => userMin < b.hi);
      if (bi >= 0) recallDist[bi].value += 1;
    }
  }

  const totals = { attempts: attemptsAll, wins: winsAll, users: usersAll.size };
  const daily = [...byDay.entries()]
    .map(([day, v]) => ({ day, attempts: v.attempts, users: v.users.size }))
    .sort((a, b) => (a.day < b.day ? -1 : 1));
  const bySize = [...bySizeMap.entries()]
    .map(([size, v]) => ({ size, attempts: v.attempts, wins: v.wins, users: v.users.size }));
  const solveHist = [...solveB.entries()].map(([bucket, n]) => ({ bucket, n })).sort((a, b) => a.bucket - b.bucket);
  const reviewHist = [...reviewB.entries()].map(([bucket, n]) => ({ bucket, n })).sort((a, b) => a.bucket - b.bucket);

  // Per-user attempts-per-active-day, binned: "how many users average ~5/day,
  // ~10/day, …". One sample per user (attempts ÷ distinct active days), so it
  // counts users, not user-days.
  const PER_USER_BINS = [[1, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, Infinity]];
  const PER_USER_LABELS = ['1–5', '5–10', '10–20', '20–40', '40–80', '80+'];
  const attemptsPerUserDist = PER_USER_LABELS.map(label => ({ label, value: 0 }));
  for (const u of perUserActivity.values()) {
    const rate = u.attempts / u.days.size;
    const i = PER_USER_BINS.findIndex(([lo, hi]) => rate >= lo && rate < hi);
    if (i >= 0) attemptsPerUserDist[i].value += 1;
  }

  // Answer-derived rank histogram (EGF scale, weakest→strongest). Each active
  // user's rank is estimated from the last 7 days of first-time attempts — the
  // mean internal rank of the difficulty-matched problems they were served —
  // not from self-reported server ranks. Users with too little data are dropped.
  const ranks = await computeAnswerRankHist(env, userFirstProblems, neededProblems);

  const data = {
    // Stamp the code version so a deploy that changes this computation
    // invalidates the cache on the next read/cron, even within the 24h window.
    version: APP_VERSION,
    computedAt: new Date().toISOString(),
    windowDays: 7,
    totals: totals || { attempts: 0, wins: 0, users: 0 },
    bucketSec: SOLVE_BUCKET_SEC,
    daily, attemptsPerUserDist, bySize, solveHist, reviewHist, recallDist, ranks,
  };

  await env.DB.prepare(
    `INSERT INTO server_stats (key, data, computed_at) VALUES ('latest', ?, ?)
     ON CONFLICT(key) DO UPDATE SET data = excluded.data, computed_at = excluded.computed_at`
  ).bind(JSON.stringify(data), data.computedAt).run();

  return data;
}

// Minimum distinct first-time problems a user needs in the window before we
// trust their answer-derived rank. Below this we leave the user out entirely.
const MIN_RANK_PROBLEMS = 10;

function fetchTsumegoAsset(env, path) {
  return env.ASSETS
    .fetch(new Request('https://onlymove.net' + path))
    .then(res => (res.ok ? res : null))
    .catch(e => { console.error('asset fetch failed:', path, e); return null; });
}

// Build the Community "Ranks" histogram from answers. Each problem's coarse
// rating is the midpoint of the shard it lives in (lookup.txt resolves a
// problem id → shard hash; the per-size metadata.json gives that shard's
// min/max rating); that rating maps to an internal rank index via M
// (rating_to_rank.json). A user's rank is the mean index over their first-time
// problems — sound because selection serves problems near the user's rank.
// Returns [{ rank: label, n }] ordered weakest→strongest; [] if assets missing.
async function computeAnswerRankHist(env, userFirstProblems, neededProblems) {
  if (!neededProblems.size) return [];

  // 1. problem id → shard hash, parsed straight off lookup.txt ("id hash size"
  //    per line). Keep only the ids we need to avoid a 143k-entry map.
  const lookupRes = await fetchTsumegoAsset(env, '/lookup.txt');
  if (!lookupRes) return [];
  const text = await lookupRes.text();
  const idToShard = new Map();
  for (let start = 0; start < text.length;) {
    let nl = text.indexOf('\n', start);
    if (nl < 0) nl = text.length;
    const sp = text.indexOf(' ', start);
    if (sp > start && sp < nl) {
      const id = text.slice(start, sp);
      if (neededProblems.has(id)) {
        const rest = text.slice(sp + 1, nl).trim();
        idToShard.set(id, rest.split(/\s+/)[0]);
      }
    }
    start = nl + 1;
  }

  // 2. shard hash → midpoint raw rating (board-independent scale, so a single
  //    map across both sizes is fine).
  const shardRating = new Map();
  for (const size of [9, 13]) {
    const res = await fetchTsumegoAsset(env, `/tsumegos/${size}/metadata.json`);
    if (!res) continue;
    const shards = await res.json();
    for (const s of shards) {
      const hash = String(s.filename).split('.')[0];
      shardRating.set(hash, (s.min_rating + s.max_rating) / 2);
    }
  }

  // 3. M curve (raw rating → internal rank index), per board size — inlined at
  //    build time (see the import at the top), so no asset fetch is needed.
  const curves = RATING_RANK_MAP;
  const rankIndexOf = (rating, size) => {
    const c = curves[String(size)] || curves['9'];
    if (!c || !Array.isArray(c.ratings) || !Array.isArray(c.rank_indices)) return null;
    return interpExtrap(rating, c.ratings, c.rank_indices);
  };

  // 4. Per user: mean internal-rank index → EGF scalar (index 19 == 1k == 0).
  const RANK_INDEX_OFFSET = 19;
  const rankHist = new Map();
  for (const pm of userFirstProblems.values()) {
    let sum = 0, n = 0;
    for (const [pid, size] of pm) {
      const hash = idToShard.get(pid);
      if (!hash) continue;
      const rating = shardRating.get(hash);
      if (rating == null) continue;
      const idx = rankIndexOf(rating, size);
      if (idx == null || !Number.isFinite(idx)) continue;
      sum += idx; n++;
    }
    if (n < MIN_RANK_PROBLEMS) continue;
    const scalar = Math.max(-29, Math.min(9, Math.round(sum / n) - RANK_INDEX_OFFSET));
    rankHist.set(scalar, (rankHist.get(scalar) || 0) + 1);
  }
  return [...rankHist.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([v, n]) => ({ rank: rankScalarToLabel(v), n }));
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
            p.endpoint, p.p256dh, p.auth, p.last_notified_at, p.last_seen_at
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
    // Activity watermark: push_subscriptions.last_seen_at is bumped on every
    // /schedule call (including empty diffs); per-row updated_at still covers
    // legacy replace-all clients, so take the max of both.
    u.lastActiveAt = Math.max(u.lastActiveAt, row.updated_at || 0, row.last_seen_at || 0);
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

  // Never deep-link a specific problem: this message is a snapshot, and by
  // the time the user taps it the problem may already have been reviewed (or
  // left its window). `?recall=1` tells the client to pick a due recall
  // problem itself from its own, fresher scheduler state — and to fall back
  // to normal selection if nothing is due anymore.
  const allIds = [...enteredIds, ...exitingIds];
  const url = '/?recall=1';

  return { title, body, url, ids: allIds };
}
