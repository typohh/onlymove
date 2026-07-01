// pwa/src/handlers/register.js
//
// EGD-leaderboard registration. Two endpoints:
//   GET  /invite?t=<token>  — preview an invitation's EGD name for the confirmation
//                             dialog (no side effect).
//   POST /register {token, userId} — claim an invitation: bind the browser's user_id
//                             to the PIN, seed the buffer + rank from the user's
//                             existing history, mark the invitation claimed (one-time).
// The token is an opaque 8-char DB key (no crypto). See docs/leaderboard_plan.md.

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import {
    getProblemParams, buildRowStateFromReports, gorToRankScalar, serializeBuffer, weekIdAt,
} from '../leaderboard.js';

const TOKEN_RE = /^[A-Za-z0-9]{8}$/;

export async function handleInvitePreview(request, env) {
    const token = new URL(request.url).searchParams.get('t') || '';
    if (!TOKEN_RE.test(token) || !env.DB) return jsonResponse({ ok: false, error: 'bad_request' }, 400);
    const inv = await env.DB.prepare(
        `SELECT egd_name, claimed_by_user_id FROM invitation WHERE invite_token = ?1`
    ).bind(token).first();
    if (!inv) return jsonResponse({ ok: false, error: 'invalid' }, 404);
    if (inv.claimed_by_user_id) return jsonResponse({ ok: false, error: 'claimed' }, 409);
    return jsonResponse({ ok: true, egdName: inv.egd_name });
}

export async function handleRegister(request, env) {
    try {
        const data = await request.json();
        const token = data.token;
        const userId = data.userId;
        if (!TOKEN_RE.test(token || '') || !isValidUserId(userId) || !env.DB) {
            return jsonResponse({ ok: false, error: 'bad_request' }, 400);
        }
        const inv = await env.DB.prepare(
            `SELECT egd_pin, egd_name, country_code, gor, claimed_by_user_id
               FROM invitation WHERE invite_token = ?1`
        ).bind(token).first();
        if (!inv) return jsonResponse({ ok: false, error: 'invalid' }, 404);
        if (inv.claimed_by_user_id) return jsonResponse({ ok: false, error: 'claimed' }, 409);

        // 1:1 binding: reject if the PIN is already registered to a DIFFERENT account
        // (UNIQUE(egd_pin) is the hard backstop; this gives a clean error message).
        const existing = await env.DB.prepare(
            `SELECT user_id FROM leaderboard WHERE egd_pin = ?1`
        ).bind(inv.egd_pin).first();
        if (existing && existing.user_id !== userId) {
            return jsonResponse({ ok: false, error: 'pin_taken' }, 409);
        }

        // Seed buffer + rank from the user's existing tsumego_reports so an
        // established player is immediately rankable (else pred_egf_rank stays NULL
        // until they accumulate RANK_START_MIN novel attempts).
        const params = await getProblemParams(env);
        const st = await buildRowStateFromReports(env, userId, params);
        const nowIso = new Date().toISOString();
        const ranked = st.predEgfRank !== null;
        const weekId = ranked ? weekIdAt(Date.now()) : null;
        const thisMax = ranked ? st.predEgfRank : null;

        await env.DB.batch([
            env.DB.prepare(
                `INSERT INTO leaderboard
                    (user_id, egd_pin, egd_name, country_code, gor, gor_updated_at, buffer,
                     pred_egf_rank, pred_egf_se, gor_rank, prev_week_gor_rank, week_id,
                     this_week_max, prev_week_max, last_active)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,NULL,?11,?12,NULL,?6)
                 ON CONFLICT(user_id) DO UPDATE SET
                    egd_pin=excluded.egd_pin, egd_name=excluded.egd_name,
                    country_code=excluded.country_code, gor=excluded.gor,
                    gor_updated_at=excluded.gor_updated_at, buffer=excluded.buffer,
                    pred_egf_rank=excluded.pred_egf_rank, pred_egf_se=excluded.pred_egf_se,
                    gor_rank=excluded.gor_rank, week_id=excluded.week_id,
                    this_week_max=excluded.this_week_max, last_active=excluded.last_active`
            ).bind(userId, inv.egd_pin, inv.egd_name, inv.country_code, inv.gor, nowIso,
                   serializeBuffer(st.buffer), st.predEgfRank, st.se, gorToRankScalar(inv.gor),
                   weekId, thisMax),
            env.DB.prepare(
                `UPDATE invitation SET claimed_by_user_id = ?2, claimed_at = ?3 WHERE invite_token = ?1`
            ).bind(token, userId, nowIso),
        ]);
        return jsonResponse({ ok: true, egdName: inv.egd_name });
    } catch (e) {
        console.error('register failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
