import { versionedJsonHeaders } from '../http.js';
import { isValidUserId } from '../guards.js';
import { rowHash } from '../../shared/sync-digest.js';

// --- Account sync ------------------------------------------------------------
// Returns this user's reports newer than the client's cursor so a device can
// rebuild/merge its local history (multi-device accounts: the account
// credential IS the userId). Reports are append-only, so a cursor fully
// captures "what changed since last sync" — no content hash needed.
//
// TWO cursor modes (docs/account_sync_redesign.md):
//   - id mode (NEW): page by AUTOINCREMENT id (`id > sinceId ORDER BY id`).
//     id is arrival order, so a row inserted with a historical created_at (an
//     offline flush, a backfill) still gets a fresh high id and is pulled —
//     the gap the old created_at cursor left. Requires idx_tsumego_reports_user_id.
//   - created_at mode (LEGACY): `created_at > since ORDER BY created_at`, kept
//     for pre-redesign clients until they all send sinceId. Uses
//     idx_tsumego_reports_user_created.
// Either way an up-to-date device's poll reads ~0 rows.
const SYNC_PAGE_LIMIT = 1000;
// Server-enforced cap on full-history pulls (from the beginning — since='' or
// sinceId=0) per user per UTC day: the stats eviction-recovery refresh, account
// adoption, and the backfill/enrich reconciles all count. Each is a full
// per-user table scan; client-side counting is bypassable by clearing
// localStorage, so the cap lives here. Incremental syncs (real cursor) and
// existence probes (limit:1) are not full pulls and don't count.
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

    // Pick cursor mode. A client that sends a (non-negative integer) sinceId is
    // on the id cursor; everyone else stays on the legacy created_at string.
    const idMode = Number.isInteger(data.sinceId) && data.sinceId >= 0;
    const since = (typeof data.since === 'string' && data.since.length <= 40) ? data.since : '';
    const sinceId = idMode ? data.sinceId : 0;

    // Optional smaller page, used as a cheap existence probe (limit: 1 reads
    // ~1 row instead of a full page). NB: paging onward from a small-limit
    // response can skip ties past the cut in created_at mode — fine for probes,
    // which discard the cursor; real pulls use the default.
    const limit = (Number.isInteger(data.limit) && data.limit > 0)
      ? Math.min(data.limit, SYNC_PAGE_LIMIT)
      : SYNC_PAGE_LIMIT;

    // Full-history pull (from the beginning): the expensive full per-user scan.
    // Capping the query itself (cursor at its start sentinel) rather than a
    // client flag means it can't be bypassed. Probes (limit:1) and incremental
    // syncs skip it; page 2+ of a full pull carries a real cursor, so one pull
    // counts once.
    const atBeginning = idMode ? (sinceId === 0) : (since === '');
    const isFullPull = atBeginning && limit > 1;
    let refreshesLeft;
    if (isFullPull) {
      const day = new Date().toISOString().slice(0, 10);
      const row = await env.DB.prepare(
        `SELECT count FROM history_refresh_log WHERE user_id = ? AND day = ?`
      ).bind(data.userId, day).first();
      const used = row ? row.count : 0;
      if (used >= HISTORY_REFRESH_DAILY_CAP) {
        return new Response(
          JSON.stringify({ reports: [], cursor: since, cursorId: sinceId, hasMore: false, capExceeded: true, refreshesLeft: 0 }),
          { status: 200, headers: syncHeaders }
        );
      }
      await env.DB.prepare(
        `INSERT INTO history_refresh_log (user_id, day, count) VALUES (?, ?, 1)
         ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1`
      ).bind(data.userId, day).run();
      refreshesLeft = HISTORY_REFRESH_DAILY_CAP - (used + 1);
    }

    // Fetch one row past the page limit to detect truncation. Both modes now
    // return `id` per row so a transitional client can adopt the id cursor.
    const cols = `id, problem, success, size, move, time_used, review_time, is_repeat, created_at`;
    const query = idMode
      ? env.DB.prepare(
          `SELECT ${cols} FROM tsumego_reports WHERE user_id = ? AND id > ?
           ORDER BY id LIMIT ?`
        ).bind(data.userId, sinceId, limit + 1)
      : env.DB.prepare(
          `SELECT ${cols} FROM tsumego_reports WHERE user_id = ? AND created_at > ?
           ORDER BY created_at LIMIT ?`
        ).bind(data.userId, since, limit + 1);
    const { results } = await query.all();

    let rows = results || [];
    let hasMore = false;
    if (rows.length > limit) {
      hasMore = true;
      rows = rows.slice(0, limit);
      if (!idMode) {
        // created_at is not unique, so a page must not end in the middle of a
        // run of equal timestamps — the exclusive cursor would skip the run's
        // remainder. Cut back to the last timestamp change. id is unique, so
        // the id path needs no such handling.
        const boundary = rows[rows.length - 1].created_at;
        let cut = rows.length;
        while (cut > 0 && rows[cut - 1].created_at === boundary) cut--;
        if (cut > 0) rows = rows.slice(0, cut);
      }
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
    const cursor = rows.length > 0 ? rows[rows.length - 1].created_at : since;
    const cursorId = rows.length > 0 ? rows[rows.length - 1].id : sinceId;

    // Per-profile digest summary (the user's latest row), so an up-to-date
    // device can confirm it is in sync without making a /report: compare
    // serverSeq to its local row count and serverDigest to its local
    // Σ rowHash(serverId). O(1) via idx_tsumego_reports_user_id. serverMaxId is
    // the current id frontier — an existing device adopting the id cursor for
    // the first time inits cursorId to it ("cursor = now" in id space; the
    // digest check backstops any rows that skips).
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

    const payload = { reports, cursor, cursorId, hasMore, serverSeq, serverDigest, serverMaxId };
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
