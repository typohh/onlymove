// pwa/src/handlers/link.js
//
// Federation identity linking, self-service. A user finds themselves in the EGD
// or AGA database from the app's Account dialog and binds their account to that
// player; the /leaderboard page then shows their name instead of "Anonymous" and
// their verified rating in the Declared column.
//
//   POST /link   {userId, service, id, name, country} -> {ok:true, linked}
//   POST /unlink {userId}                             -> {ok:true}
//
// (The current binding is read through GET /profile — handlers/profile.js — which
// answers the Account dialog's whole panel in one request.)
//
// NOTHING HERE VERIFIES THAT THE USER IS THE PLAYER, and no such check exists to
// add: neither database authenticates anyone, so a linked name is a claim, not a
// credential. What is enforced is that the claim is EXCLUSIVE — a player id bound
// to one account cannot be bound to another — and that it is REVERSIBLE, since a
// self-service pick is a self-service mistake. Unlink frees the id for whoever
// asks next.
//
// The two services differ in what the Worker can check for itself:
//   AGA — the Ratings Explorer is on Azure, so the Worker fetches the member
//         directly. The id must resolve or the link is REFUSED, and the stored
//         name + the seeded rating heartbeat both come from that fetch, never
//         from the request body.
//   EGD — behind Cloudflare, so a Worker->EGD subrequest is blocked (403, "error
//         code: 1106", the orange-to-orange restriction). The name and country
//         are therefore taken from the linking client's own EGD fetch. No rating
//         is accepted here at all: the GoR arrives through the existing weekly
//         client piggyback (gor-refresh.js -> applyGorUpdates), which already
//         bounds and week-guards what it stores, so linking opens no new path
//         for a made-up rating.

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import {
    getProblemParams, buildRowStateFromReports, serializeBuffer,
    invalidateStaleGor, shortRankLabel, gorToRank, agaRatingToRank,
} from '../leaderboard.js';
import { rankToLabel, rankBand, clampRankBand } from '../../shared/rank-math.js';
import { fetchAgaPlayer, isValidAgaId } from '../aga.js';

const NAME_MAX = 100;

// The account's current binding, or null. Shared with GET /profile so the Account
// dialog reads its whole panel in one request.
//
// `rank` is read LIVE off the account's latest machine heartbeat rather than being
// snapshotted onto the binding: the heartbeat is already the one place that rating
// is kept, it is refreshed weekly, and it is what the leaderboard's Declared column
// reads — so the dialog and the board cannot disagree, and a player who changes
// grade does not have to be relinked to say so. Prefer the federation's OWN label
// ('4d', the same precedence readLeaderboardRows uses for its activity tier) and
// derive one from the raw rating only for a legacy heartbeat written before that
// column existed. The scan is a scan of ~700 rows, on a route that fires when
// somebody opens a dialog.
export async function readLinkedIdentity(env, userId) {
    if (!env.DB || !isValidUserId(userId)) return null;
    const [row, beat] = await Promise.all([
        env.DB.prepare(
            `SELECT egd_pin, aga_id, display_name, country_code
               FROM external_ids WHERE user_id = ?1`
        ).bind(userId).first(),
        env.DB.prepare(
            `SELECT server, rank, rank_label FROM rank_surveys
              WHERE user_id = ?1 AND server IN ('EGD', 'AGAR')
              ORDER BY id DESC LIMIT 1`
        ).bind(userId).first(),
    ]);
    if (!row) return null;
    const isAga = row.aga_id != null;
    const id = isAga ? String(row.aga_id) : (row.egd_pin != null ? String(row.egd_pin) : '');
    if (!id) return null;
    return {
        service: isAga ? 'AGA' : 'EGD',
        id,
        name: row.display_name || '',
        country: row.country_code || '',
        rank: heartbeatLabel(beat),
    };
}

// A heartbeat row -> the grade to show, or ''. The stored label wins; failing that
// the raw value is mapped through that service's coupling and labelled on our own
// ladder, which is a band-edge approximation of the federation's grade rather than
// the grade itself — acceptable only because the rows without a label are legacy
// ones that age out on their next weekly refresh.
function heartbeatLabel(beat) {
    if (!beat) return '';
    const official = shortRankLabel(beat.rank_label);
    if (official) return official;
    const value = Number(beat.rank);
    if (!Number.isFinite(value)) return '';
    const rank = beat.server === 'AGAR' ? agaRatingToRank(value) : gorToRank(value);
    if (!Number.isFinite(rank)) return '';
    return rankToLabel(clampRankBand(rankBand(rank)));
}

// Every statement that removes a binding removes the machine rating heartbeats
// with it. They are keyed by user_id, not by player id, so one left behind keeps
// reporting the PREVIOUS player's verified rating as this account's — on the
// leaderboard's Declared column, under whatever name is linked next. The human
// survey declarations ('EGF'/'AGA'/... rows) are the user's own and stay.
function clearHeartbeats(env, userId) {
    return env.DB.prepare(
        `DELETE FROM rank_surveys WHERE user_id = ?1 AND server IN ('EGD', 'AGAR')`
    ).bind(userId);
}

export async function handleLink(request, env) {
    let data;
    try { data = await request.json(); } catch { return jsonResponse({ ok: false, error: 'bad_request' }, 400); }
    if (!isValidUserId(data?.userId) || !env.DB) {
        return jsonResponse({ ok: false, error: 'bad_request' }, 400);
    }
    const userId = data.userId;
    const isAga = data.service === 'AGA';
    const idVal = String(data.id || '');
    if (isAga ? !isValidAgaId(idVal) : !/^\d{1,12}$/.test(idVal)) {
        return jsonResponse({ ok: false, error: 'bad_id' }, 400);
    }

    // AGA is fetched Worker-side, so an id that doesn't resolve is refused rather
    // than stored on trust — and the name comes back with it.
    let agaPlayer = null;
    if (isAga) {
        try { agaPlayer = await fetchAgaPlayer(idVal); } catch (e) { agaPlayer = null; }
        if (!agaPlayer) return jsonResponse({ ok: false, error: 'unreachable' }, 502);
    }
    const name = isAga
        ? (agaPlayer.displayName || '')
        : (typeof data.name === 'string' ? data.name.trim().slice(0, NAME_MAX) : '');
    if (!name) return jsonResponse({ ok: false, error: 'bad_name' }, 400);
    // AGA's Explorer reports a US STATE, not a country, so an AGA link carries no
    // country and the leaderboard's edge geo-IP fallback supplies the flag.
    const country = isAga ? null
        : ((typeof data.country === 'string' ? data.country : '').slice(0, 2).toUpperCase() || null);

    try {
        // Exclusivity, the one thing linking does enforce. The per-column UNIQUE
        // index is the hard backstop; this is what turns it into a clean message.
        const idCol = isAga ? 'aga_id' : 'egd_pin';
        const holder = await env.DB.prepare(
            `SELECT user_id FROM external_ids WHERE ${idCol} = ?1`
        ).bind(idVal).first();
        if (holder && holder.user_id !== userId) {
            return jsonResponse({ ok: false, error: 'id_taken' }, 409);
        }

        // Seed buffer + rank from the account's existing tsumego_reports so an
        // established player is rankable immediately, rather than waiting to
        // re-accumulate RANK_START_MIN novel attempts.
        const params = await getProblemParams(env);
        const st = await buildRowStateFromReports(env, userId, params);
        const nowIso = new Date().toISOString();

        // One row per account: relinking REPLACES, so the other service's column is
        // cleared in the same write and the previous identity's heartbeats go with
        // it (relinking to a different player of the same service would otherwise
        // inherit their verified rating).
        const stmts = [
            clearHeartbeats(env, userId),
            env.DB.prepare(
                `INSERT INTO external_ids (user_id, egd_pin, aga_id, display_name, country_code)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(user_id) DO UPDATE SET
                    egd_pin = excluded.egd_pin, aga_id = excluded.aga_id,
                    display_name = excluded.display_name, country_code = excluded.country_code`
            ).bind(userId, isAga ? null : idVal, isAga ? idVal : null, name, country),
            env.DB.prepare(
                `INSERT INTO leaderboard
                    (user_id, buffer, pred_rank, pred_rank_se, time_delta_rating, last_active)
                 VALUES (?1,?2,?3,?4,?5,?6)
                 ON CONFLICT(user_id) DO UPDATE SET
                    buffer=excluded.buffer,
                    pred_rank=excluded.pred_rank, pred_rank_se=excluded.pred_rank_se,
                    time_delta_rating=excluded.time_delta_rating,
                    last_active=excluded.last_active`
            ).bind(userId, serializeBuffer(st.buffer), st.rank, st.se, st.deltaRating, nowIso),
        ];
        // AGA: the fetch above is authoritative, so the first heartbeat is written
        // now and the player shows a verified rank straight away. EGD gets none —
        // its GoR comes from the client piggyback, which the cache invalidation
        // below puts this pin in line for on the very next /report.
        if (isAga && Number.isFinite(agaPlayer.rating)) {
            stmts.push(env.DB.prepare(
                `INSERT INTO rank_surveys (user_id, server, rank, created_at, country, rank_label, last_game_at)
                 VALUES (?1, 'AGAR', ?2, ?3, NULL, ?4, ?5)`
            ).bind(userId, String(agaPlayer.rating), nowIso,
                agaPlayer.rankLabel, agaPlayer.latestEventDate || null));
        }
        await env.DB.batch(stmts);
        if (!isAga) invalidateStaleGor();

        // The AGA fetch above wrote a heartbeat, so its grade is already known here.
        // An EGD link has none until the GoR piggyback lands on the next /report;
        // the client fills that one render from the search result it just picked.
        return jsonResponse({
            ok: true,
            linked: {
                service: isAga ? 'AGA' : 'EGD', id: idVal, name,
                country: country || '', rank: isAga ? agaPlayer.rankLabel : '',
            },
        });
    } catch (e) {
        console.error('handleLink failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}

export async function handleUnlink(request, env) {
    let data;
    try { data = await request.json(); } catch { return jsonResponse({ ok: false, error: 'bad_request' }, 400); }
    if (!isValidUserId(data?.userId) || !env.DB) {
        return jsonResponse({ ok: false, error: 'bad_request' }, 400);
    }
    try {
        // Identity + heartbeats go together (see clearHeartbeats). The leaderboard
        // row is deliberately LEFT: it is the account's own rank state, rebuilt
        // from tsumego_reports and shown for every active player, linked or not —
        // unlinking makes the row anonymous, it does not remove the player.
        await env.DB.batch([
            clearHeartbeats(env, data.userId),
            env.DB.prepare(`DELETE FROM external_ids WHERE user_id = ?1`).bind(data.userId),
        ]);
        invalidateStaleGor();
        return jsonResponse({ ok: true });
    } catch (e) {
        console.error('handleUnlink failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
