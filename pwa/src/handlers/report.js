import { APP_VERSION } from '../version.js';
import { isValidUserId, isValidProblemId } from '../guards.js';

export async function handleReport(request, env) {
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
