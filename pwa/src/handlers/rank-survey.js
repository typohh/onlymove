import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';

const SURVEY_SERVERS = ['OGS', 'KGS', 'Fox', 'Tygem', 'WBaduk', 'IGS', 'GoQuest', 'EGF', 'AGA'];
const SURVEY_RANK_RE = /^(?:[1-9]|1\d|2[0-5])k$|^[1-9]d$/;

export async function handleRankSurvey(request, env) {
  try {
    const data = await request.json();
    if (!isValidUserId(data?.userId) || !data.ranks || typeof data.ranks !== 'object') {
      return jsonResponse({ error: 'Invalid body' }, 400);
    }
    if (!env.DB) {
      return jsonResponse({ error: 'Database not configured' }, 500);
    }

    const createdAt = new Date().toISOString();
    const statements = [];
    for (const server of SURVEY_SERVERS) {
      const rank = data.ranks[server];
      if (typeof rank !== 'string') continue;
      if (!SURVEY_RANK_RE.test(rank)) continue;
      statements.push(env.DB.prepare(
        'INSERT INTO rank_surveys (user_id, server, rank, created_at) VALUES (?, ?, ?, ?)'
      ).bind(data.userId, server, rank, createdAt));
    }

    if (statements.length > 0) await env.DB.batch(statements);
    return jsonResponse({ ok: true, recorded: statements.length });
  } catch (err) {
    console.error('Error handling rank survey:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}
