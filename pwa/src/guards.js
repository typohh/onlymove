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
// (see get_orientation_invariant_hash). user_id is a short client-generated
// token (observed 8 chars, a few legacy ids up to 23); bound it generously.
const PROBLEM_ID_RE = /^[0-9a-f]{8}$/;
const USER_ID_RE = /^[A-Za-z0-9_-]{1,40}$/;
export const isValidProblemId = (v) => typeof v === 'string' && PROBLEM_ID_RE.test(v);
export const isValidUserId = (v) => typeof v === 'string' && USER_ID_RE.test(v);

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
