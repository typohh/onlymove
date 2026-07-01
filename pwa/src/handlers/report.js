import { versionedJsonHeaders } from '../http.js';
import { isValidUserId, isValidProblemId } from '../guards.js';
import { rowHash } from '../../shared/sync-digest.js';
import { coerceAttemptFields, prepareAttemptInsert } from '../attempt-insert.js';
import { updateLeaderboardOnReport, applyGorUpdates, getStaleGorPins } from '../leaderboard.js';

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

    const { size, move, timeUsed, reviewTime } = coerceAttemptFields(data);
    const isRepeat = data.isRepeat ? 1 : 0;

    if (!env.DB) {
      return new Response('Database not configured', { status: 500, headers: reportHeaders });
    }

    // The server-stamped timestamp is echoed back so the client can store it
    // on the matching answerHistory entry (serverCreatedAt) — that's how a
    // /sync pull recognizes the device's own rows instead of re-merging them.
    //
    // The INSERT (attempt-insert.js) also lazy-computes the per-profile sync
    // digest in the same statement; the response `digest` is the INCLUSIVE
    // value running_sum + rowHash(id), which the client compares against its
    // local Σ rowHash(serverId). serverId/userSeq/digest are the
    // change-detection signal (a userSeq past what the device expects = another
    // device wrote → pull).
    const createdAt = new Date().toISOString();
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

    // EGD leaderboard: fan a NOVEL attempt into the registered user's rank buffer
    // AFTER the response (ctx.waitUntil) so the high-volume /report path isn't slowed.
    // The point read inside is the registration check — unregistered users are a
    // cheap no-op. Best-effort; the buffer is a derived cache (the cron reconciles).
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
    const respBody = { createdAt, serverId, userSeq, digest };
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
