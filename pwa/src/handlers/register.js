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
    getProblemParams, buildRowStateFromReports, serializeBuffer, weekStartIso,
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

        // 1:1 binding, enforced both directions against external_ids:
        //  - PIN side: reject if this invite's PIN is already registered to a DIFFERENT
        //    account (UNIQUE(egd_pin) is the hard backstop; this gives a clean message).
        const existing = await env.DB.prepare(
            `SELECT user_id FROM external_ids WHERE egd_pin = ?1`
        ).bind(inv.egd_pin).first();
        if (existing && existing.user_id !== userId) {
            return jsonResponse({ ok: false, error: 'pin_taken' }, 409);
        }
        //  - Account side: an account binds to exactly ONE EGD id, for life. If it
        //    already has a binding under a DIFFERENT pin, reject — rebinding would
        //    orphan the old pin and let one account hop EGD identities. Re-claiming
        //    its OWN pin is a harmless refresh: the INSERT OR IGNORE below no-ops.
        const selfRow = await env.DB.prepare(
            `SELECT egd_pin FROM external_ids WHERE user_id = ?1`
        ).bind(userId).first();
        if (selfRow && selfRow.egd_pin !== inv.egd_pin) {
            return jsonResponse({ ok: false, error: 'account_bound' }, 409);
        }

        // Seed buffer + rank from the user's existing tsumego_reports so an
        // established player is immediately rankable (else pred_egf_rank stays NULL
        // until they accumulate RANK_START_MIN novel attempts).
        const params = await getProblemParams(env);
        const st = await buildRowStateFromReports(env, userId, params);
        const nowIso = new Date().toISOString();

        const stmts = [
            // The binding — identity only (external_ids). IGNORE covers the
            // re-claim-own-pin refresh; the guards above rejected every conflicting
            // combination.
            env.DB.prepare(
                `INSERT OR IGNORE INTO external_ids (user_id, egd_pin) VALUES (?1, ?2)`
            ).bind(userId, inv.egd_pin),
            env.DB.prepare(
                `INSERT INTO leaderboard
                    (user_id, buffer, pred_egf_rank, pred_egf_se, last_active)
                 VALUES (?1,?2,?3,?4,?5)
                 ON CONFLICT(user_id) DO UPDATE SET
                    buffer=excluded.buffer,
                    pred_egf_rank=excluded.pred_egf_rank, pred_egf_se=excluded.pred_egf_se,
                    last_active=excluded.last_active`
            ).bind(userId, serializeBuffer(st.buffer), st.predEgfRank, st.se, nowIso),
            env.DB.prepare(
                `UPDATE invitation SET claimed_by_user_id = ?2, claimed_at = ?3 WHERE invite_token = ?1`
            ).bind(token, userId, nowIso),
        ];
        // Seed the first GoR heartbeat from the invitation's EGD snapshot so the
        // player shows a GoR immediately — same weekly-guarded shape as
        // applyGorUpdates (a re-claim in the same week no-ops).
        if (Number.isFinite(inv.gor)) {
            stmts.push(env.DB.prepare(
                `INSERT INTO rank_surveys (user_id, server, rank, created_at, country)
                 SELECT ?1, 'EGD', ?2, ?3, ?4
                  WHERE NOT EXISTS (SELECT 1 FROM rank_surveys
                                     WHERE user_id = ?1 AND server = 'EGD'
                                       AND created_at >= ?5)`
            ).bind(userId, String(inv.gor), nowIso, inv.country_code || null,
                   weekStartIso(Date.now())));
        }
        await env.DB.batch(stmts);
        return jsonResponse({ ok: true, egdName: inv.egd_name });
    } catch (e) {
        console.error('register failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
