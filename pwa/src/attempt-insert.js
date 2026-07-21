// pwa/src/attempt-insert.js
//
// The one INSERT into tsumego_reports, shared by /report (live attempts,
// server-stamped created_at) and /report-backfill (outbox flush, historical
// client-supplied created_at). The lazy-seeded digest arithmetic below is
// correctness-critical for account sync, so the two paths must stay
// byte-identical — hence one module.
//
// The INSERT maintains the per-profile sync digest in the same statement
// (docs/account_sync_redesign.md): user_seq = the user's running attempt
// count, running_sum = the EXCLUSIVE prefix of Σ rowHash(id) over the user's
// earlier rows. The chain is a TABLE INVARIANT — every existing row carries
// its values (the 2026-07 DB surgery materialized all pre-column rows), so
// the only fallback case is a brand-new user with no rows at all (the
// constants 0). Rides the O(1) latest-row lookup via
// idx_tsumego_reports_user_id. RETURNING hands back the new row's id + the
// two digest fields (/report echoes all three; the backfill only reads id).
import { ID_HASH_SQL } from '../shared/sync-digest.js';

const ATTEMPT_INSERT_SQL =
  `INSERT INTO tsumego_reports
     (user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat, user_seq, running_sum)
   SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
     COALESCE((SELECT user_seq FROM tsumego_reports WHERE user_id = ?1 ORDER BY id DESC LIMIT 1), 0) + 1,
     COALESCE((SELECT running_sum + ${ID_HASH_SQL} FROM tsumego_reports WHERE user_id = ?1 ORDER BY id DESC LIMIT 1), 0)
   RETURNING id, user_seq, running_sum`;

// Optional per-attempt fields; older clients don't send them, so every column
// is nullable. move is a GTP coordinate or "pass" (≤4 chars); bound its length
// so a malformed payload can't store an unbounded blob. Times must be
// non-negative; idle-tab outliers (hours) are left intact — the stats layer
// caps them. size defaults to 9 to match the historical corpus.
export function coerceAttemptFields(r) {
  return {
    size: (r.size === 9 || r.size === 13) ? r.size : 9,
    move: (typeof r.move === 'string' && r.move.length <= 4) ? r.move : null,
    timeUsed: (Number.isFinite(r.timeUsed) && r.timeUsed >= 0) ? Math.round(r.timeUsed) : null,
    reviewTime: (Number.isFinite(r.reviewTime) && r.reviewTime >= 0) ? Math.round(r.reviewTime) : null,
  };
}

export function prepareAttemptInsert(env, { userId, problem, success, createdAt, size, move, timeUsed, reviewTime, isRepeat }) {
  return env.DB.prepare(ATTEMPT_INSERT_SQL)
    .bind(userId, problem, success ? 1 : 0, createdAt, size, move, timeUsed, reviewTime, isRepeat ? 1 : 0);
}
