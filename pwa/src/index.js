import { CORS_HEADERS, jsonResponse, tooManyRequests } from './http.js';
import { maintenanceFrozen, rateLimited, serviceUnavailable } from './guards.js';
import { handleReport } from './handlers/report.js';
import { handleSync } from './handlers/sync.js';
import { handleReportBackfill } from './handlers/report-backfill.js';
import { handleSubscribe, handleUnsubscribe } from './handlers/push.js';
import { handleSchedule } from './handlers/schedule.js';
import { handleRankSurvey } from './handlers/rank-survey.js';
import { handleShareEvent } from './handlers/share-event.js';
import { handleServerStats, maybeComputeServerStats } from './server-stats.js';
import { runRecallNotifications } from './recall-notifications.js';
import { handleLink, handleUnlink } from './handlers/link.js';
import { handleLeaderboard } from './handlers/leaderboard-page.js';
import { handleTournament } from './handlers/tournament-page.js';
import { handleProfileHide, handleProfile } from './handlers/profile.js';
import { handleGoogleSignIn, handleGoogleLink, handleGoogleUnlink } from './handlers/google.js';
import { handleRefit } from './handlers/refit.js';
import { reconcileBuffers, pruneIdleLeaderboardRows, refreshAgaRatings } from './leaderboard.js';

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
  // Leaderboard identity: the user binds their account to an EGD/AGA player they
  // picked out of the Account dialog's search, or drops the binding again. The
  // search itself is client-side (the Worker cannot reach EGD at all), so these
  // only write. GET counterpart (read current binding + hide flag) is /profile.
  { path: '/link', method: 'POST', limiter: 'RL_WRITE', handler: handleLink },
  { path: '/unlink', method: 'POST', limiter: 'RL_WRITE', handler: handleUnlink },
  // "Hide my profile" toggle for the leaderboard (Account dialog, and the foot of
  // the rank survey). GET counterpart (read current state) is /profile.
  { path: '/profile/hide', method: 'POST', limiter: 'RL_WRITE', handler: handleProfileHide },
  // Google sign-in: a way back into an account code on a device that does not
  // have it. /google/signin only RESOLVES (which code, if any, this Google
  // account is bound to) and the client decides between logging in and linking;
  // the other two write. POST throughout — an ID token must never ride in a URL,
  // where it would land in logs and in the Referer header. Whether a Google
  // account is linked is read back through /profile with the rest of the panel.
  { path: '/google/signin', method: 'POST', limiter: 'RL_WRITE', handler: handleGoogleSignIn },
  { path: '/google/link', method: 'POST', limiter: 'RL_WRITE', handler: handleGoogleLink },
  { path: '/google/unlink', method: 'POST', limiter: 'RL_WRITE', handler: handleGoogleUnlink },
  // History re-derive: one batch of problems in, every derived field the walk
  // used to open shard zips for out. A READ of published assets — it touches no
  // D1 and records nothing — but it sits here rather than in GET_ROUTES for two
  // reasons: a body of a few hundred ids does not belong in a URL, and the
  // maintenance freeze SHOULD cover it, since a snapshot-relative window is
  // exactly when the sidecars it reads may be half-replaced. Clients already
  // treat a 503 as being offline and retry, which is the right behaviour here.
  { path: '/refit', method: 'POST', limiter: 'RL_WRITE', handler: handleRefit },
];

// GET routes, dispatched like WRITE_ROUTES (reads, so no limiter). Every entry
// must sit here rather than in the SPA fallback below — the fallback swallows
// any unmatched navigation into index.html.
const GET_ROUTES = [
  { path: '/push/vapid-public-key', handler: (request, env) => new Response(env.VAPID_PUBLIC_KEY || '', {
      headers: { ...CORS_HEADERS, 'Content-Type': 'text/plain' },
    }) },
  { path: '/server-stats', handler: (request, env) => handleServerStats(env, request) },
  // Standalone EGD leaderboard page (HTML).
  { path: '/leaderboard', handler: (request, env) => handleLeaderboard(env) },
  // Static FINAL-RESULTS page for the concluded 2026 tournament (baked-in
  // standings, no D1 read).
  { path: '/tournament', handler: () => handleTournament() },
  // The account's leaderboard profile: the hide flag + the linked EGD/AGA identity,
  // which is everything the Account dialog paints. Read; no limiter. no-store, set
  // in the handler.
  { path: '/profile', handler: (request, env) => handleProfile(request, env) },
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
        // Maintenance freeze, checked before the limiter and before any handler
        // touches D1 — see maintenanceFrozen(). Covers every POST route,
        // including /sync, so no client observes a half-applied surgery.
        if (maintenanceFrozen(env)) return serviceUnavailable();
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
    // The cron writes recall_schedule notification stamps and the leaderboard
    // buffers, both of which DB surgery edits — so the freeze covers it too.
    // Skipping a tick is harmless: reminders are re-evaluated next hour and the
    // weekly jobs are idempotent.
    if (maintenanceFrozen(env)) return;
    ctx.waitUntil(runRecallNotifications(env));
    ctx.waitUntil(maybeComputeServerStats(env));
    ctx.waitUntil(pruneHistoryRefreshLog(env));
    // EGD leaderboard, weekly (Monday 00:00 UTC): first delete idle rows (lazily
    // re-created on their user's next novel /report; registration lives in
    // external_ids so nothing durable is lost), then reconcile the remaining
    // buffers from tsumego_reports (heals drift). GoR itself is refreshed by
    // active clients piggybacking on /report (the Worker can't reach EGD).
    // AGA ratings, in contrast, are refreshed HERE (the Worker can reach AGAGD
    // server-side); refreshAgaRatings is best-effort and no-ops when up to date.
    const now = new Date();
    if (now.getUTCDay() === 1 && now.getUTCHours() === 0) {
      ctx.waitUntil(pruneIdleLeaderboardRows(env).then(() => reconcileBuffers(env)));
      ctx.waitUntil(refreshAgaRatings(env));
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
