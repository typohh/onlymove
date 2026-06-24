import { jsonResponse } from '../http.js';
import { isValidProblemId, isValidUserId } from '../guards.js';

export async function handleShareEvent(request, env) {
  try {
    const data = await request.json();
    const type = (data?.type === 'create' || data?.type === 'open') ? data.type : null;
    if (!type) {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    if (!env.DB) {
      return jsonResponse({ error: 'Database not configured' }, 500);
    }

    const size = (data.size === 9 || data.size === 13) ? data.size : null;
    const problemId = isValidProblemId(data.problemId) ? data.problemId : null;
    const userId = isValidUserId(data.userId) ? data.userId : null;
    // "Where the link came from": the client sends document.referrer (the Referer
    // of the navigation that opened the shared link) for opens; creates send
    // none. We don't fall back to this POST's own Referer — that's just the app
    // origin and would be noise. Capped defensively.
    const referrer = (typeof data.referrer === 'string' && data.referrer)
      ? data.referrer.slice(0, 500) : null;
    // Country is Cloudflare's edge geo-IP, not something the client can report.
    const country = (request.cf && request.cf.country) ? request.cf.country : null;

    await env.DB.prepare(
      'INSERT INTO share_events (event_type, user_id, problem_id, size, country, referrer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)'
    )
      .bind(type, userId, problemId, size, country, referrer, new Date().toISOString())
      .run();

    return jsonResponse({ ok: true });
  } catch (err) {
    console.error('Error handling share event:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}
