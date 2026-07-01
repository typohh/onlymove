import { versionedJsonHeaders } from '../http.js';
import { isValidUserId, isValidProblemId } from '../guards.js';
import { coerceAttemptFields, prepareAttemptInsert } from '../attempt-insert.js';

// --- Historical report upload (the durable outbox's upload path) -------------
// PERMANENT endpoint for the client's offline outbox (account-sync.flushOutbox):
// any attempt recorded locally but never acked — an offline / failed /report,
// or a pre-2026-05-15 one-row-per-problem gap — is uploaded here WITH ITS
// RECORDED EVENT TIMESTAMP (a client-supplied historical created_at, distinct
// from /report which stamps `now`). The client first reconciles a full pull and
// fuzzy-claims rows the server already has, so only provably-absent entries land
// here. Rows lazy-seed user_seq/running_sum exactly like /report (shared
// ID_HASH_SQL) and RETURN their ids so the client can stamp serverId; an old
// created_at keeps the row out of the server-stats 7-day window, and the id
// cursor still propagates it to other devices.
const BACKFILL_MAX_BATCH = 500;
// Backfilled timestamps must be historical: after the site could plausibly
// have data, and old enough not to collide with live reporting.
const BACKFILL_MIN_CREATED = '2024-01-01T00:00:00.000Z';

export async function handleReportBackfill(request, env) {
  const headers = versionedJsonHeaders();
  try {
    const data = await request.json();
    if (!isValidUserId(data.userId) || !Array.isArray(data.reports) ||
        data.reports.length === 0 || data.reports.length > BACKFILL_MAX_BATCH) {
      return new Response('Invalid request body', { status: 400, headers });
    }
    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers });
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
      // Same lazy-seeded user_seq/running_sum INSERT as /report (one source of
      // truth: attempt-insert.js), so a backfilled/outbox-flushed row keeps the
      // per-profile digest consistent — without it the latest row would be
      // unseeded and the client's sync check would stall. Batches run
      // sequentially in one transaction, so each statement's subquery sees the
      // prior insert and the seq advances across the batch. RETURNING id lets
      // the client stamp serverId on the flushed entry. move is always NULL
      // here: the outbox doesn't round-trip picked moves.
      const { size, timeUsed, reviewTime } = coerceAttemptFields(r);
      stmts.push(prepareAttemptInsert(env, {
        userId: data.userId, problem: r.problem, success: r.success,
        createdAt, size, move: null, timeUsed, reviewTime, isRepeat: r.isRepeat,
      }));
    }
    const results = await env.DB.batch(stmts);
    // Inserted ids in request order, so the client can stamp serverId on each
    // flushed entry (the digest key).
    const ids = results.map((res) => (res && res.results && res.results[0]) ? res.results[0].id : null);
    return new Response(JSON.stringify({ inserted: stmts.length, ids }), { status: 201, headers });
  } catch (error) {
    console.error('Error handling report backfill:', error);
    return new Response('Internal server error', { status: 500, headers });
  }
}
