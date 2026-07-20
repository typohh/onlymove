// pwa/src/handlers/register.js
//
// Leaderboard registration (EGD or AGA, chosen by the invitation's `service`). Two
// endpoints:
//   GET  /invite?t=<token>  — preview an invitation's player name for the confirmation
//                             dialog (no side effect).
//   POST /register {token, userId} — claim an invitation: bind the browser's user_id
//                             to the external id (egd_pin or aga_id), seed the buffer
//                             + rank from the user's existing history, mark the
//                             invitation claimed (one-time). The account binds to a
//                             single external identity — one-per-account is a POLICY
//                             enforced here, not a DB constraint.
// The token is an opaque 8-char DB key (no crypto). See docs/leaderboard_plan.md.

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import {
    getProblemParams, buildRowStateFromReports, serializeBuffer, weekStartIso,
} from '../leaderboard.js';
import { fetchAgaPlayer } from '../aga.js';

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
            `SELECT egd_pin, egd_name, country_code, gor, service, aga_id, aga_rating,
                    claimed_by_user_id
               FROM invitation WHERE invite_token = ?1`
        ).bind(token).first();
        if (!inv) return jsonResponse({ ok: false, error: 'invalid' }, 404);
        if (inv.claimed_by_user_id) return jsonResponse({ ok: false, error: 'claimed' }, 409);

        // Which external service this invite targets. 'AGA' uses aga_id; anything
        // else (incl. NULL, the legacy default) is EGD using egd_pin.
        const isAga = inv.service === 'AGA';
        const idCol = isAga ? 'aga_id' : 'egd_pin';
        const idVal = isAga ? (inv.aga_id != null ? String(inv.aga_id) : '')
                            : (inv.egd_pin != null ? String(inv.egd_pin) : '');
        if (!idVal) return jsonResponse({ ok: false, error: 'invalid' }, 404);

        // 1:1 binding, enforced both directions against external_ids:
        //  - id side: reject if this invite's external id is already registered to a
        //    DIFFERENT account (the per-column UNIQUE index is the hard backstop; this
        //    gives a clean message).
        const existing = await env.DB.prepare(
            `SELECT user_id FROM external_ids WHERE ${idCol} = ?1`
        ).bind(idVal).first();
        if (existing && existing.user_id !== userId) {
            return jsonResponse({ ok: false, error: isAga ? 'id_taken' : 'pin_taken' }, 409);
        }
        //  - Account side: an account binds to exactly ONE external identity, for life
        //    (POLICY — the schema itself allows both an egd_pin and an aga_id; the
        //    one-per-account rule is enforced HERE). Reject if the account already
        //    holds the OTHER service, or a DIFFERENT id of this service. Re-claiming
        //    its OWN id is a harmless refresh: the INSERT OR IGNORE below no-ops.
        const selfRow = await env.DB.prepare(
            `SELECT egd_pin, aga_id FROM external_ids WHERE user_id = ?1`
        ).bind(userId).first();
        if (selfRow) {
            const already = isAga ? selfRow.aga_id : selfRow.egd_pin;
            const other = isAga ? selfRow.egd_pin : selfRow.aga_id;
            if (other != null || (already != null && String(already) !== idVal)) {
                return jsonResponse({ ok: false, error: 'account_bound' }, 409);
            }
        }

        // For AGA, fetch the current official rank_label from the Ratings Explorer
        // (server-side; Azure is reachable) so the player shows a rank immediately.
        // Best-effort: if it can't be fetched, no heartbeat is seeded and the weekly
        // cron fills it in later.
        let agaRankLabel = null;
        if (isAga) {
            let player = null;
            try { player = await fetchAgaPlayer(idVal); } catch (e) { player = null; }
            if (player && player.rankLabel) agaRankLabel = player.rankLabel;
        }

        // Seed buffer + rank from the user's existing tsumego_reports so an
        // established player is immediately rankable (else pred_egf_rank stays NULL
        // until they accumulate RANK_START_MIN novel attempts).
        const params = await getProblemParams(env);
        const st = await buildRowStateFromReports(env, userId, params);
        const nowIso = new Date().toISOString();
        const weekStart = weekStartIso(Date.now());

        const stmts = [
            // The binding — identity only (external_ids). IGNORE covers the
            // re-claim-own-id refresh; the guards above rejected every conflicting
            // combination.
            env.DB.prepare(
                `INSERT OR IGNORE INTO external_ids (user_id, ${idCol}) VALUES (?1, ?2)`
            ).bind(userId, idVal),
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
        // Seed the first rating heartbeat from the snapshot so the player shows a rank
        // immediately — same weekly-guarded shape as applyGorUpdates/refreshAgaRatings
        // (a re-claim in the same week no-ops). 'EGD' (GoR) or 'AGAR' (AGA rating).
        if (!isAga && Number.isFinite(inv.gor)) {
            stmts.push(env.DB.prepare(
                `INSERT INTO rank_surveys (user_id, server, rank, created_at, country)
                 SELECT ?1, 'EGD', ?2, ?3, ?4
                  WHERE NOT EXISTS (SELECT 1 FROM rank_surveys
                                     WHERE user_id = ?1 AND server = 'EGD'
                                       AND created_at >= ?5)`
            ).bind(userId, String(inv.gor), nowIso, inv.country_code || null, weekStart));
        } else if (isAga && agaRankLabel) {
            stmts.push(env.DB.prepare(
                `INSERT INTO rank_surveys (user_id, server, rank, created_at, country)
                 SELECT ?1, 'AGAR', ?2, ?3, NULL
                  WHERE NOT EXISTS (SELECT 1 FROM rank_surveys
                                     WHERE user_id = ?1 AND server = 'AGAR'
                                       AND created_at >= ?4)`
            ).bind(userId, agaRankLabel, nowIso, weekStart));
        }
        await env.DB.batch(stmts);
        return jsonResponse({ ok: true, egdName: inv.egd_name });
    } catch (e) {
        console.error('register failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
