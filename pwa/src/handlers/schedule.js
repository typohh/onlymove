import { jsonResponse } from '../http.js';
import { isValidUserId, isValidProblemId } from '../guards.js';

// /schedule accepts two payload shapes:
//   { userId, schedule: [...] }            — LEGACY replace-all (older clients;
//     every row upserted, then untouched rows swept by updated_at watermark).
//   { userId, upserts: [...], removals: [...] } — diff mode: the client keeps a
//     per-row snapshot of what it last uploaded and sends only rows whose
//     hour-rounded window moved, plus explicit removals. Each upsert bills the
//     row + its two partial indexes (~3 write units), so the diff cuts the
//     dominant D1 write consumer to the handful of rows an attempt actually
//     changes. An EMPTY diff is still a valid call — it lands only the
//     activity-watermark bump below.
// Either way the call bumps push_subscriptions.last_seen_at for the account's
// device row(s) (no-op for users without a subscription): the cron's
// mid-session suppression is per-user (active on any device defers the whole
// account), so bumping all of them by user_id is correct, and it reads this
// watermark because a diff upload doesn't refresh recall_schedule.updated_at
// across the board.
const SCHEDULE_MAX_ROWS = 1000;

export async function handleSchedule(request, env) {
  const data = await request.json();
  if (!isValidUserId(data?.userId)) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }
  const legacy = Array.isArray(data.schedule);
  const diff = Array.isArray(data.upserts) && Array.isArray(data.removals);
  if (!legacy && !diff) {
    return jsonResponse({ error: 'Invalid body' }, 400);
  }

  const now = Date.now();
  const statements = [];
  const pushUpsert = (row) => {
    if (!isValidProblemId(row?.problemId) || typeof row.enterAt !== 'number' || typeof row.exitAt !== 'number') return;
    statements.push(env.DB.prepare(
      `INSERT INTO recall_schedule (user_id, problem_id, enter_at, exit_at, updated_at)
       VALUES (?1, ?2, ?3, ?4, ?5)
       ON CONFLICT(user_id, problem_id) DO UPDATE SET
         enter_at = excluded.enter_at,
         exit_at = excluded.exit_at,
         enter_notified_at = CASE WHEN excluded.enter_at > recall_schedule.enter_at
                                  THEN NULL ELSE recall_schedule.enter_notified_at END,
         exit_notified_at = CASE WHEN excluded.exit_at > recall_schedule.exit_at
                                 THEN NULL ELSE recall_schedule.exit_notified_at END,
         updated_at = excluded.updated_at`
    ).bind(data.userId, row.problemId, row.enterAt, row.exitAt, now));
  };

  if (legacy) {
    if (data.schedule.length > SCHEDULE_MAX_ROWS) {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    for (const row of data.schedule) pushUpsert(row);

    // Replace-all semantics: anything previously scheduled for this user but
    // absent from the new upload gets deleted. ON CONFLICT above preserves the
    // notified flags for surviving rows. We identify "untouched" rows by their
    // updated_at watermark — every row in this upload was just stamped with
    // `now`, so anything still carrying an earlier timestamp wasn't in the
    // payload. A NOT IN (?,?,…) version would hit D1's 100-bound-params cap
    // per statement once a user's schedule exceeds ~100 problems.
    statements.push(env.DB.prepare(
      'DELETE FROM recall_schedule WHERE user_id = ? AND updated_at < ?'
    ).bind(data.userId, now));
  } else {
    if (data.upserts.length > SCHEDULE_MAX_ROWS || data.removals.length > SCHEDULE_MAX_ROWS) {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    for (const row of data.upserts) pushUpsert(row);
    // Explicit removals, chunked under D1's ~100-bound-params-per-statement cap.
    const ids = data.removals.filter(isValidProblemId);
    for (let i = 0; i < ids.length; i += 50) {
      const chunk = ids.slice(i, i + 50);
      statements.push(env.DB.prepare(
        `DELETE FROM recall_schedule WHERE user_id = ? AND problem_id IN (${chunk.map(() => '?').join(',')})`
      ).bind(data.userId, ...chunk));
    }
  }

  statements.push(env.DB.prepare(
    'UPDATE push_subscriptions SET last_seen_at = ? WHERE user_id = ?'
  ).bind(now, data.userId));

  await env.DB.batch(statements);
  return jsonResponse({ ok: true });
}
