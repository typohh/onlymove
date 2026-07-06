import { versionedJsonHeaders } from '../http.js';
import { isValidUserId } from '../guards.js';
import { rowHash } from '../../shared/sync-digest.js';

// --- Account sync ------------------------------------------------------------
// Returns this user's reports newer than the client's id cursor so a device can
// rebuild/merge its local history (multi-device accounts: the account
// credential IS the userId). Reports are append-only, so the cursor fully
// captures "what changed since last sync" — no content hash needed.
//
// Pages by AUTOINCREMENT id (`id > sinceId ORDER BY id`): id is arrival order,
// so a row inserted with a historical created_at (an offline flush, a backfill)
// still gets a fresh high id and is pulled. Requires idx_tsumego_reports_user_id.
// Every response also carries a per-profile digest summary
// (serverSeq/serverDigest/serverMaxId) for the client's cheap in-sync check.
// See docs/account_sync_redesign.md.
const SYNC_PAGE_LIMIT = 1000;
// Server-enforced cap on full-history pulls (sinceId=0) per user per UTC day:
// the stats eviction-recovery refresh, account adoption, and the outbox
// reconcile all count. Each is a full per-user table scan; client-side counting
// is bypassable by clearing localStorage, so the cap lives here. Incremental
// syncs (real cursor), existence/frontier probes (limit:1), and full pulls
// that return NO rows (fresh empty accounts) don't count.
const HISTORY_REFRESH_DAILY_CAP = 3;

export async function handleSync(request, env) {
  const syncHeaders = versionedJsonHeaders();
  try {
    const data = await request.json();
    if (!isValidUserId(data.userId)) {
      return new Response('Invalid request body', { status: 400, headers: syncHeaders });
    }
    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers: syncHeaders });
    }

    // id cursor only. A request without a valid sinceId gets an empty no-op so it
    // doesn't trigger a full-table scan, rather than syncing. No data is lost
    // (its /report writes are unaffected); it resyncs in full on reload.
    if (!Number.isInteger(data.sinceId) || data.sinceId < 0) {
      return new Response(
        JSON.stringify({ reports: [], cursorId: 0, hasMore: false, serverSeq: null, serverDigest: 0, serverMaxId: 0 }),
        { status: 200, headers: syncHeaders }
      );
    }
    const sinceId = data.sinceId;

    // Optional smaller page, used as a cheap existence/frontier probe (limit: 1
    // reads ~1 row instead of a full page).
    const limit = (Number.isInteger(data.limit) && data.limit > 0)
      ? Math.min(data.limit, SYNC_PAGE_LIMIT)
      : SYNC_PAGE_LIMIT;

    // Full-history pull (from the beginning, sinceId 0): the expensive full
    // per-user scan. Capping the query itself (sinceId === 0) rather than a
    // client flag means it can't be bypassed. Probes (limit:1) and incremental
    // syncs skip it; page 2+ of a full pull carries a real cursor, so one pull
    // counts once. Billing happens after the query, and only when rows were
    // actually served: an empty result (a fresh account with no reports yet —
    // a cursor-less device retries from 0 on every load/visibility trigger)
    // reads ~0 rows and must not eat the day's slots.
    const isFullPull = sinceId === 0 && limit > 1;
    const day = new Date().toISOString().slice(0, 10);
    let fullPullUsed = 0;
    if (isFullPull) {
      const row = await env.DB.prepare(
        `SELECT count FROM history_refresh_log WHERE user_id = ? AND day = ?`
      ).bind(data.userId, day).first();
      fullPullUsed = row ? row.count : 0;
      if (fullPullUsed >= HISTORY_REFRESH_DAILY_CAP) {
        return new Response(
          JSON.stringify({ reports: [], cursorId: sinceId, hasMore: false, capExceeded: true, refreshesLeft: 0 }),
          { status: 200, headers: syncHeaders }
        );
      }
    }

    // Fetch one row past the page limit to detect truncation. id is unique, so
    // no page-boundary tie handling is needed.
    const { results } = await env.DB.prepare(
      `SELECT id, problem, success, size, move, time_used, review_time, is_repeat, created_at
       FROM tsumego_reports WHERE user_id = ? AND id > ?
       ORDER BY id LIMIT ?`
    ).bind(data.userId, sinceId, limit + 1).all();

    let rows = results || [];
    let hasMore = false;
    if (rows.length > limit) {
      hasMore = true;
      rows = rows.slice(0, limit);
    }

    // Bill the full pull only now that we know it served something.
    let refreshesLeft;
    if (isFullPull) {
      if (rows.length > 0) {
        await env.DB.prepare(
          `INSERT INTO history_refresh_log (user_id, day, count) VALUES (?, ?, 1)
           ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1`
        ).bind(data.userId, day).run();
        fullPullUsed += 1;
      }
      refreshesLeft = HISTORY_REFRESH_DAILY_CAP - fullPullUsed;
    }

    const reports = rows.map((r) => ({
      id: r.id,
      problem: r.problem,
      success: !!r.success,
      size: r.size,
      move: r.move,
      timeUsed: r.time_used,
      reviewTime: r.review_time,
      isRepeat: !!r.is_repeat,
      createdAt: r.created_at,
    }));
    const cursorId = rows.length > 0 ? rows[rows.length - 1].id : sinceId;

    // Per-profile digest summary (the user's latest row), so an up-to-date
    // device can confirm it is in sync without making a /report: compare
    // serverSeq to its local row count and serverDigest to its local
    // Σ rowHash(serverId). O(1) via idx_tsumego_reports_user_id. serverMaxId is
    // the current id frontier — the one-time migrateIdCursor() probe adopts it
    // as cursorId for a device coming off the old created_at cursor.
    //   - no rows at all      → serverSeq 0,    serverDigest 0,   serverMaxId 0
    //   - rows exist, unseeded (latest row predates the digest columns and the
    //     user hasn't reported since) → serverSeq null (client skips the check
    //     until the user's first post-migration /report seeds it); serverMaxId
    //     is still the real frontier so the cursor can migrate.
    //   - seeded              → the real values
    let serverSeq = 0, serverDigest = 0, serverMaxId = 0;
    const latest = await env.DB.prepare(
      `SELECT id, user_seq, running_sum FROM tsumego_reports WHERE user_id = ? ORDER BY id DESC LIMIT 1`
    ).bind(data.userId).first();
    if (latest) {
      serverMaxId = latest.id;
      if (latest.user_seq != null && latest.running_sum != null) {
        serverSeq = latest.user_seq;
        serverDigest = latest.running_sum + rowHash(latest.id);
      } else {
        serverSeq = null; serverDigest = null;
      }
    }

    const payload = { reports, cursorId, hasMore, serverSeq, serverDigest, serverMaxId };
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
