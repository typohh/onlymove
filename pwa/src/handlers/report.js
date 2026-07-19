import { versionedJsonHeaders } from '../http.js';
import { isValidUserId, isValidProblemId } from '../guards.js';
import { rowHash } from '../../shared/sync-digest.js';
import { coerceAttemptFields, prepareAttemptInsert } from '../attempt-insert.js';
import { updateLeaderboardOnReport, applyGorUpdates, getStaleGorPins } from '../leaderboard.js';

// The client OWNS the attempt's event time (docs/client_owned_event_time_plan.md):
// createdAt is REQUIRED in the payload and inserted VERBATIM, so the stored row
// equals the client's local entry exactly and sync matching is exact (no fuzzy
// window). The server only sanity-bounds it — and REFUSES, never rewrites: a
// silently substituted stamp would create a row the client's exact matcher can't
// recognize as its own, which is precisely the duplicate-minting precondition.
const REPORT_FUTURE_SLACK_MS = 2 * 60 * 1000;      // clock skew allowance
const REPORT_PAST_WINDOW_MS = 7 * 24 * 3600 * 1000; // live reports are sent at attempt time
const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/;

function timestampRejection(createdAt, nowMs) {
  if (typeof createdAt !== 'string' || !ISO_RE.test(createdAt)) return 'missing_timestamp';
  const ts = Date.parse(createdAt);
  if (!Number.isFinite(ts)) return 'missing_timestamp';
  if (ts > nowMs + REPORT_FUTURE_SLACK_MS) return 'future_timestamp';
  if (ts < nowMs - REPORT_PAST_WINDOW_MS) return 'stale_timestamp';
  return null;
}

export async function handleReport(request, env, ctx) {
  // Every response carries the deployed Worker version so the client can
  // detect a stale bundle and trigger a quiescent reload (see
  // js/state.js maybeReload).
  const reportHeaders = versionedJsonHeaders();
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

    // Timestamp bounds: refused loudly with serverNow so the client can
    // self-correct its entry and resend (js/user-stats.js sendReport). A
    // stamp-less payload can only be a not-yet-reloaded old bundle's first
    // report(s) around the deploy — its attempt stays un-acked and reaches the
    // server via the outbox (/report-backfill) after the forced reload.
    const nowMs = Date.now();
    const rejection = timestampRejection(data.createdAt, nowMs);
    if (rejection) {
      return new Response(
        JSON.stringify({ error: rejection, serverNow: new Date(nowMs).toISOString() }),
        { status: 400, headers: reportHeaders });
    }
    const createdAt = data.createdAt;

    const { size, move, timeUsed, reviewTime } = coerceAttemptFields(data);
    const isRepeat = data.isRepeat ? 1 : 0;

    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers: reportHeaders });
    }

    // The INSERT (attempt-insert.js) also lazy-computes the per-profile sync
    // digest in the same statement; the response `digest` is the INCLUSIVE
    // value running_sum + rowHash(id), which the client compares against its
    // local Σ rowHash(serverId). serverId/userSeq/digest are the
    // change-detection signal (a userSeq past what the device expects = another
    // device wrote → pull). No createdAt echo: the client sent the value, and
    // the server either inserted it verbatim or refused above.
    const insRes = await prepareAttemptInsert(env, {
      userId: data.userId, problem: data.problem, success: data.success,
      createdAt, size, move, timeUsed, reviewTime, isRepeat,
    }).all();

    const ins = (insRes && insRes.results && insRes.results[0]) ? insRes.results[0] : null;
    const serverId = ins ? ins.id : null;
    const userSeq = ins ? ins.user_seq : null;
    // Inclusive digest = exclusive prefix on the new row + the new row's own
    // hash (id known only post-insert), matching the client's full Σ.
    const digest = ins ? (ins.running_sum + rowHash(ins.id)) : null;

    // Leaderboard: fan a NOVEL attempt into the user's rank buffer AFTER the
    // response (ctx.waitUntil) so the high-volume /report path isn't slowed. Every
    // user has a row (created lazily + history-seeded on their first novel report).
    // Best-effort; the buffer is a derived cache (the cron reconciles).
    if (isRepeat === 0 && ctx && typeof ctx.waitUntil === 'function') {
      ctx.waitUntil(
        updateLeaderboardOnReport(env, data.userId, data.problem, data.success, timeUsed)
          .catch((e) => console.error('leaderboard update failed:', e))
      );
    }

    // EGD-leaderboard GoR piggyback (see leaderboard.js): the Worker can't reach EGD,
    // so active clients fetch registered players' GoR and post it here as `gorUpdates`;
    // the response advertises which PINs are still stale this week (`refreshPins`) for
    // a client to fetch next. Best-effort and fully isolated — a failure here must
    // never affect the attempt report. getStaleGorPins is in-memory-cached (cheap).
    const respBody = { serverId, userSeq, digest };
    try {
      if (Array.isArray(data.gorUpdates) && data.gorUpdates.length) {
        await applyGorUpdates(env, data.gorUpdates);
      }
      const refreshPins = await getStaleGorPins(env);
      if (refreshPins.length) respBody.refreshPins = refreshPins;
    } catch (e) {
      console.error('gor piggyback failed:', e);
    }

    return new Response(JSON.stringify(respBody), { status: 201, headers: reportHeaders });
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { status: 500, headers: reportHeaders });
  }
}
