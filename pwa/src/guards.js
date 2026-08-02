// --- Abuse / D1-quota defenses ----------------------------------------------
// Two cheap, in-Worker layers guard every D1 write (see also the outer
// Cloudflare WAF rate-limit rule documented in wrangler.toml):
//   1. A per-IP rate limit, checked BEFORE any D1 work, so a flood is dropped by
//      a cheap Worker invocation instead of spending D1 write quota (and a
//      spoofed row would also keep inflating the server-stats window scan for
//      7 days — read quota — until purged).
//   2. In-memory format validation (no D1 reads) so malformed/garbage writes
//      never reach an INSERT. Deliberately format-only: we don't verify the
//      problem id against the real corpus (that needs a bundled set); the rate
//      limiter bounds the volume of valid-looking spoofs.

// Tsumego ids are the symmetry-canonical hash truncated to 8 hex chars
// (see get_orientation_invariant_hash). user_id is exactly 8 chars — the
// 2026-07 DB surgery normalized the few legacy long ids (S7), so the
// grandfathered 1-40 bound is retired.
const PROBLEM_ID_RE = /^[0-9a-f]{8}$/;
const USER_ID_RE = /^[A-Za-z0-9_-]{8}$/;
export const isValidProblemId = (v) => typeof v === 'string' && PROBLEM_ID_RE.test(v);
export const isValidUserId = (v) => typeof v === 'string' && USER_ID_RE.test(v);

// --- Maintenance freeze ------------------------------------------------------
// DB surgery (auxiliary/purge_tsumegos.mjs) computes its repair against a
// SNAPSHOT of tsumego_reports: the digest fix-ups are stated relative to the
// values that were stored when the snapshot was taken. A row inserted between
// the snapshot and the apply would be shifted against a chain it never joined,
// so the table has to stand still for the duration.
//
// Freezing is safe because the client already treats a failed report as an
// offline one: sendReport only acks on res.ok, so a refused attempt stays
// un-acked in answerHistory and the durable outbox re-uploads it through
// /report-backfill afterwards WITH ITS ORIGINAL event time. Nothing is lost and
// nothing needs to be told about the freeze — this reuses the offline path
// rather than adding a maintenance mode to the client.
//
// Reads are deliberately left alone, so the app keeps serving problems while
// the surgery runs. The one writer this does not cover is the lazy
// server_stats recompute behind GET /server-stats, which only rewrites its own
// cache row (the cron reconciles it) and never touches the tables under surgery.
export function maintenanceFrozen(env) {
  return String(env && env.MAINTENANCE_MODE || '') === '1';
}

export function serviceUnavailable() {
  return new Response(
    JSON.stringify({ error: 'maintenance', retryAfterSeconds: 900 }),
    {
      status: 503,
      headers: {
        'Content-Type': 'application/json',
        'Retry-After': '900',
        'Cache-Control': 'no-store',
        'Access-Control-Allow-Origin': '*',
      },
    });
}

// Per-IP rate limit via a Workers ratelimit binding. Returns true when the
// caller should be rejected. Fails OPEN on a missing binding (e.g. some local
// dev) or a limiter error, so a misconfiguration never drops legitimate writes.
export async function rateLimited(request, env, binding) {
  const limiter = env[binding];
  if (!limiter) return false;
  try {
    const key = request.headers.get('CF-Connecting-IP') || 'unknown';
    const { success } = await limiter.limit({ key });
    return !success;
  } catch {
    return false;
  }
}
