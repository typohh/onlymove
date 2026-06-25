import { CORS_HEADERS, tooManyRequests } from './http.js';
import { rateLimited } from './guards.js';
import { handleReport } from './handlers/report.js';
import { handleSync } from './handlers/sync.js';
import { handleReportBackfill } from './handlers/backfill.js';
import { handleSubscribe, handleUnsubscribe } from './handlers/push.js';
import { handleSchedule } from './handlers/schedule.js';
import { handleRankSurvey } from './handlers/rank-survey.js';
import { handleShareEvent } from './handlers/share-event.js';
import { handleServerStats, maybeComputeServerStats } from './server-stats.js';
import { runRecallNotifications } from './recall-notifications.js';

// Per-IP rate-limited endpoints, checked BEFORE any handler touches D1 so a
// flood is dropped by a cheap Worker invocation instead of spending D1 quota.
// /report is high-volume (one per solved move) so it gets a generous dedicated
// limiter; the rest are low-frequency and share a tighter one. /sync is a read,
// not a write, but it's per-user-keyed and quota-relevant (rows read), so it
// shares the low-frequency limiter.
const WRITE_ROUTES = [
  { path: '/report', method: 'POST', limiter: 'RL_REPORT', handler: handleReport },
  { path: '/sync', method: 'POST', limiter: 'RL_WRITE', handler: handleSync },
  { path: '/report-backfill', method: 'POST', limiter: 'RL_WRITE', handler: handleReportBackfill },
  { path: '/push/subscribe', method: 'POST', limiter: 'RL_WRITE', handler: handleSubscribe },
  { path: '/push/unsubscribe', method: 'POST', limiter: 'RL_WRITE', handler: handleUnsubscribe },
  { path: '/schedule', method: 'POST', limiter: 'RL_WRITE', handler: handleSchedule },
  { path: '/rank-survey', method: 'POST', limiter: 'RL_WRITE', handler: handleRankSurvey },
  { path: '/share-event', method: 'POST', limiter: 'RL_WRITE', handler: handleShareEvent },
];

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    for (const route of WRITE_ROUTES) {
      if (url.pathname === route.path && request.method === route.method) {
        if (await rateLimited(request, env, route.limiter)) return tooManyRequests();
        return route.handler(request, env);
      }
    }

    if (url.pathname === '/push/vapid-public-key' && request.method === 'GET') {
      return new Response(env.VAPID_PUBLIC_KEY || '', {
        headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' },
      });
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
