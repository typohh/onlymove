import { versionedJsonHeaders } from '../http.js';
import { isValidUserId, isValidProblemId } from '../guards.js';
import { ID_HASH_SQL } from '../../shared/sync-digest.js';

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
//
// This whole module is the temporary surface: both handlers, the route entries
// in src/index.js, and the client one-offs go together when the rollout is done.
const BACKFILL_MAX_BATCH = 500;
// Backfilled timestamps must be historical: after the site could plausibly
// have data, and old enough not to collide with live reporting.
const BACKFILL_MIN_CREATED = '2024-01-01T00:00:00.000Z';
const ENRICH_MAX_BATCH = 500;

export async function handleReportBackfill(request, env) {
  const headers = versionedJsonHeaders();
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
      // Same lazy-seeded user_seq/running_sum INSERT as /report (one source of
      // truth: shared ID_HASH_SQL), so a backfilled/outbox-flushed row keeps the
      // per-profile digest consistent — without it the latest row would be
      // unseeded and the client's sync check would stall. Batches run
      // sequentially in one transaction, so each statement's subquery sees the
      // prior insert and the seq advances across the batch. RETURNING id lets
      // the client stamp serverId on the flushed entry.
      stmts.push(env.DB.prepare(
        `INSERT INTO tsumego_reports
           (user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat, user_seq, running_sum)
         SELECT ?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7, ?8,
           COALESCE((SELECT user_seq FROM tsumego_reports WHERE user_id = ?1 ORDER BY id DESC LIMIT 1),
                    (SELECT COUNT(*) FROM tsumego_reports WHERE user_id = ?1)) + 1,
           COALESCE((SELECT running_sum + ${ID_HASH_SQL} FROM tsumego_reports WHERE user_id = ?1 ORDER BY id DESC LIMIT 1),
                    (SELECT COALESCE(SUM(${ID_HASH_SQL}), 0) FROM tsumego_reports WHERE user_id = ?1))
         RETURNING id`
      ).bind(data.userId, r.problem, r.success ? 1 : 0, createdAt, size, timeUsed, reviewTime, r.isRepeat ? 1 : 0));
    }
    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers });
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
export async function handleReportEnrich(request, env) {
  const headers = versionedJsonHeaders();
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
