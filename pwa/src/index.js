import { CORS_HEADERS, jsonResponse, tooManyRequests } from './http.js';
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
import { handleRegister, handleInvitePreview } from './handlers/register.js';
import { handleLeaderboard } from './handlers/leaderboard-page.js';
import { handleInvitePage, handleInviteEligible, handleInviteCreate } from './handlers/invite-tool.js';
import { reconcileBuffers } from './leaderboard.js';

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
  { path: '/register', method: 'POST', limiter: 'RL_WRITE', handler: handleRegister },
  // Invite tool (account-code gated in the handler — admins + registered leaderboard
  // members; EGD search is client-side). Rate-limited too, which bounds any
  // code-guessing on top of the account-code check.
  { path: '/invite/eligible', method: 'POST', limiter: 'RL_WRITE', handler: handleInviteEligible },
  { path: '/invite/create', method: 'POST', limiter: 'RL_WRITE', handler: handleInviteCreate },
];

// GET routes, dispatched like WRITE_ROUTES (reads, so no limiter). Every entry
// must sit here rather than in the SPA fallback below — the fallback swallows
// any unmatched navigation into index.html.
const GET_ROUTES = [
  { path: '/push/vapid-public-key', handler: (request, env) => new Response(env.VAPID_PUBLIC_KEY || '', {
      headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' },
    }) },
  { path: '/server-stats', handler: (request, env) => handleServerStats(env) },
  // Standalone EGD leaderboard page (HTML).
  { path: '/leaderboard', handler: (request, env) => handleLeaderboard(env) },
  // /invite is dual-purpose on the same path: with ?t=<token> it's the invitation
  // preview (JSON) for the /register confirmation dialog; without a token it's the
  // invite generator page (HTML). (GET /register itself is a navigation and falls
  // through to the SPA, which reads ?t= and POSTs.)
  { path: '/invite', handler: (request, env) =>
      new URL(request.url).searchParams.has('t')
        ? handleInvitePreview(request, env)
        : handleInvitePage(env) },
  // Legacy alias for the invite generator page. Inert without an eligible account
  // code, which the POST endpoints verify.
  { path: '/admin/invite', handler: (request, env) => handleInvitePage(env) },
];

// Last-resort error boundary so a handler that throws (malformed JSON body, a
// D1 hiccup, a missing binding) always yields a well-formed CORS JSON 500
// instead of workerd's raw error page. Handlers still return their own
// specific 4xx/5xx responses; this only catches what they didn't.
async function dispatch(handler, request, env, ctx) {
  try {
    return await handler(request, env, ctx);
  } catch (e) {
    console.error(`${new URL(request.url).pathname} handler failed:`, e);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    for (const route of WRITE_ROUTES) {
      if (url.pathname === route.path && request.method === route.method) {
        if (await rateLimited(request, env, route.limiter)) return tooManyRequests();
        return dispatch(route.handler, request, env, ctx);
      }
    }

    if (request.method === 'GET') {
      for (const route of GET_ROUTES) {
        if (url.pathname === route.path) {
          return dispatch(route.handler, request, env, ctx);
        }
      }
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
    // EGD leaderboard: reconcile buffers from tsumego_reports (heals drift + freezes
    // last week's GoR at the week boundary). Weekly (Monday 00:00 UTC). GoR itself is
    // refreshed by active clients piggybacking on /report (the Worker can't reach EGD).
    const now = new Date();
    if (now.getUTCDay() === 1 && now.getUTCHours() === 0) {
      ctx.waitUntil(reconcileBuffers(env));
    }
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
