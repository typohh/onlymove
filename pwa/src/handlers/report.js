import { APP_VERSION } from '../version.js';
import { isValidUserId, isValidProblemId } from '../guards.js';
import { rowHash, ID_HASH_SQL } from '../../shared/sync-digest.js';

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
    // The INSERT also lazy-computes the per-profile sync digest in the same
    // statement (docs/account_sync_redesign.md): user_seq = the user's running
    // attempt count, running_sum = the EXCLUSIVE prefix of rowHash(id) over the
    // user's earlier rows. Both fall back to a one-time COUNT/SUM the first time
    // a user reports after the column was added (latest row NULL), then ride the
    // O(1) latest-row lookup via idx_tsumego_reports_user_id. RETURNING hands
    // back the new row's id + the two digest fields; the response `digest` is
    // the INCLUSIVE value running_sum + rowHash(id), which the client compares
    // against its local Σ rowHash(serverId). serverId/userSeq/digest are the
    // change-detection signal (a userSeq past what the device expects = another
    // device wrote → pull).
    const createdAt = new Date().toISOString();
    const insRes = await env.DB.prepare(
      `INSERT INTO tsumego_reports
         (user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat, user_seq, running_sum)
       SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
         COALESCE((SELECT user_seq FROM tsumego_reports WHERE user_id = ?1 ORDER BY id DESC LIMIT 1),
                  (SELECT COUNT(*) FROM tsumego_reports WHERE user_id = ?1)) + 1,
         COALESCE((SELECT running_sum + ${ID_HASH_SQL} FROM tsumego_reports WHERE user_id = ?1 ORDER BY id DESC LIMIT 1),
                  (SELECT COALESCE(SUM(${ID_HASH_SQL}), 0) FROM tsumego_reports WHERE user_id = ?1))
       RETURNING id, user_seq, running_sum`
    ).bind(data.userId, data.problem, data.success ? 1 : 0, createdAt, size, move, timeUsed, reviewTime, isRepeat).all();

    const ins = (insRes && insRes.results && insRes.results[0]) ? insRes.results[0] : null;
    const serverId = ins ? ins.id : null;
    const userSeq = ins ? ins.user_seq : null;
    // Inclusive digest = exclusive prefix on the new row + the new row's own
    // hash (id known only post-insert), matching the client's full Σ.
    const digest = ins ? (ins.running_sum + rowHash(ins.id)) : null;

    return new Response(JSON.stringify({ createdAt, serverId, userSeq, digest }), {
      status: 201,
      headers: { ...reportHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Error handling report:', error);
    return new Response('Internal server error', { status: 500, headers: reportHeaders });
  }
}
