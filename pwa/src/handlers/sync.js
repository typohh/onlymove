import { versionedJsonHeaders } from '../http.js';
import { isValidUserId } from '../guards.js';

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

export async function handleSync(request, env) {
  const syncHeaders = versionedJsonHeaders();
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
