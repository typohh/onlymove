import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';

const SURVEY_SERVERS = ['OGS', 'KGS', 'Fox', 'Tygem', 'WBaduk', 'IGS', 'GoQuest', 'EGF', 'AGA'];
const SURVEY_RANK_RE = /^(?:[1-9]|1\d|2[0-5])k$|^[1-9]d$/;

// The account's latest human declaration per survey server, { server: rank }, so
// the survey opens prefilled with what the user last said. Machine heartbeat rows
// ('EGD'/'AGAR') are not survey answers and are excluded by the server list. The
// newest row wins by id; a resubmitted unchanged answer adds a row the pipeline
// collapses the same way (latest per user and server).
export async function readDeclaredRanks(env, userId) {
  if (!env.DB || !isValidUserId(userId)) return {};
  const placeholders = SURVEY_SERVERS.map((_, i) => `?${i + 2}`).join(', ');
  const { results } = await env.DB.prepare(
    `SELECT server, rank FROM rank_surveys
      WHERE user_id = ?1 AND server IN (${placeholders})
      ORDER BY id DESC`
  ).bind(userId, ...SURVEY_SERVERS).all();
  const ranks = {};
  for (const r of results || []) {
    if (!(r.server in ranks) && SURVEY_RANK_RE.test(r.rank)) ranks[r.server] = r.rank;
  }
  return ranks;
}

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
    // Country is Cloudflare's edge geo-IP, not something the client can report.
    const country = (request.cf && request.cf.country) ? request.cf.country : null;
    const statements = [];
    for (const server of SURVEY_SERVERS) {
      const rank = data.ranks[server];
      if (typeof rank !== 'string') continue;
      if (!SURVEY_RANK_RE.test(rank)) continue;
      statements.push(env.DB.prepare(
        'INSERT INTO rank_surveys (user_id, server, rank, created_at, country) VALUES (?, ?, ?, ?, ?)'
      ).bind(data.userId, server, rank, createdAt, country));
    }

    if (statements.length > 0) await env.DB.batch(statements);
    return jsonResponse({ ok: true, recorded: statements.length });
  } catch (err) {
    console.error('Error handling rank survey:', err);
    return jsonResponse({ error: 'Internal server error' }, 500);
  }
}
