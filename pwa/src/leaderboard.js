// pwa/src/leaderboard.js
//
// EGD leaderboard core (Cloudflare Worker side). Reuses the SHARED window estimator
// so a registered user's leaderboard rank is computed by the SAME code as the
// front-page rank, from the user's materialized 500-attempt buffer. Used by the
// /report hot path, /register (claim seeding), and the cron. See
// docs/leaderboard_plan.md.

import { interpExtrap, RANK_INDEX_OFFSET } from '../shared/rank-math.js';
import {
    estimateWindowRankFromFlat, weightedMedianOverWindow, flattenAttempt, WIN_CAP,
} from '../shared/rank-window.js';
// Inlined at build time (wrangler bundles JSON imports, like src/server-stats.js).
import RATING_RANK_MAP from '../tsumegos/rating_to_rank.json';
import RANK_MAP from '../tsumegos/rank_map.json';
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

// Rank is emitted only once the buffer holds this many usable novel attempts (the
// first 100 are INCLUDED in the fit; the gate only delays emitting a rank). A user
// with pred_egf_rank NULL is absent from the board.
export const RANK_START_MIN = 100;
const TIME_MIN_TIMED = 20;              // >= this many valid solve times for a time dock
export const BUFFER_CAP = WIN_CAP;      // 500 — the buffer + the rank window share it

// One shared cross-board curve (map_rating_to_rank writes both size keys identical).
const CURVE = RATING_RANK_MAP['9'] || Object.values(RATING_RANK_MAP)[0] || null;
const EGF = (RANK_MAP && RANK_MAP.EGF) || null;
const TIME_ADJUST = POPULATION_CONSTANTS.time_adjust || null;

// --- in-memory problem_params map: the WHOLE asset, loaded once per isolate -------
// Immutable for a deploy's lifetime (every deploy resets isolates -> auto-fresh);
// a cold isolate reloads it from the ASSETS file, never serving stale data. Purely
// a faster lookup — no partial/LRU cache, no per-problem fetch.
let _params = null;
let _paramsPromise = null;

async function loadProblemParams(env) {
    const res = await env.ASSETS.fetch(new Request('https://onlymove.net/tsumegos/problem_params.bin'));
    if (!res || !res.ok) throw new Error('problem_params.bin unavailable');
    const bytes = new Uint8Array(await res.arrayBuffer());
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const REC = 16;   // 4-byte id + 3 little-endian float32 (rating, slope, floor)
    const map = new Map();
    for (let off = 0; off + REC <= bytes.length; off += REC) {
        let id = '';
        for (let k = 0; k < 4; k++) id += bytes[off + k].toString(16).padStart(2, '0');
        map.set(id, {
            rating: dv.getFloat32(off + 4, true),
            slope: dv.getFloat32(off + 8, true),
            floor: dv.getFloat32(off + 12, true),
        });
    }
    return map;
}

export async function getProblemParams(env) {
    if (_params) return _params;
    if (!_paramsPromise) {
        _paramsPromise = loadProblemParams(env)
            .then((m) => { _params = m; return m; })
            .catch((e) => { _paramsPromise = null; throw e; });
    }
    return _paramsPromise;
}

// internal rank INDEX -> EGF rank scalar (mirrors rank-translation.js
// internalIndexToServerScalar for the EGF coupling; same shared interpExtrap + data).
function internalIndexToEgf(rankIndex) {
    if (!EGF || !Array.isArray(EGF.our) || !Array.isArray(EGF.server) || EGF.our.length < 2) return null;
    if (!Number.isFinite(rankIndex)) return null;
    return interpExtrap(rankIndex - RANK_INDEX_OFFSET, EGF.our, EGF.server);
}

// Apply a rating-space delta to a rank index (mirrors rank-decomposition.js
// rankIndexAfterRatingDelta; invert M, shift in rating units, re-apply M).
function rankIndexAfterRatingDelta(rankIndex, deltaRating) {
    if (!CURVE || !Number.isFinite(rankIndex) || !Number.isFinite(deltaRating)) return rankIndex;
    const rating = interpExtrap(rankIndex, CURVE.rank_indices, CURVE.ratings);
    return interpExtrap(rating + deltaRating, CURVE.ratings, CURVE.rank_indices);
}

// GoR (integer EGD rating) -> onlymove rank scalar (1k = 0, 1d = 1). Standard EGF
// ~100-GoR/stone linearization anchored at 1k = 2000, 1d = 2100 (kept simple; the
// full nonlinear EGF relation could replace it later). Returns null on a bad GoR.
export function gorToRankScalar(gor) {
    if (typeof gor !== 'number' || !Number.isFinite(gor) || gor <= 0) return null;
    return (gor - 2000) / 100;
}

// The user's leaderboard rank from a buffer of { p, s, t } (newest FIRST). Resolves
// rating/slope/floor from `params` (dropping problems no longer in the corpus),
// flattens through the SHARED transform, runs the SHARED window estimator, then
// time-adjusts (weighted-median solve time over the same window) and maps to EGF.
// Returns { predEgfRank, se } — predEgfRank NULL when fewer than RANK_START_MIN
// usable attempts (no rank yet).
export function computeRank(buffer, params) {
    const flat = [];
    const times = [];
    for (let i = buffer.length - 1; i >= 0; i--) {   // oldest -> newest
        const e = buffer[i];
        const pp = e && typeof e.p === 'string' ? params.get(e.p) : null;
        if (!pp) continue;                            // left the corpus / malformed
        const f = flattenAttempt(pp.rating, pp.slope, pp.floor, e.s, CURVE);
        if (!f) continue;
        flat.push(f);
        times.push(Number.isFinite(e.t) ? e.t : NaN);
    }
    if (flat.length < RANK_START_MIN) return { predEgfRank: null, se: null };

    // No server-side warm-start: the window MLE is a pure function of the recent
    // history (the prior is weak, WIN_PRIOR_SD = 4, vs hundreds of attempts), so the
    // recent-mean fallback converges to the same R the client's prev-rank seed does.
    const est = estimateWindowRankFromFlat(flat, NaN);
    if (!est) return { predEgfRank: null, se: null };

    let rankIndex = est.rankIndex;
    const medMs = weightedMedianOverWindow(times, TIME_MIN_TIMED);
    if (medMs !== null && TIME_ADJUST
        && Number.isFinite(TIME_ADJUST.alpha_per_log10_min)
        && Number.isFinite(TIME_ADJUST.center_log10_min)) {
        const deltaRating = TIME_ADJUST.alpha_per_log10_min
            * (Math.log10(medMs / 60000) - TIME_ADJUST.center_log10_min);
        rankIndex = rankIndexAfterRatingDelta(rankIndex, deltaRating);
    }
    return { predEgfRank: internalIndexToEgf(rankIndex), se: est.se };
}

// --- week bucketing (Monday-00:00-UTC week) -------------------------------------
// weekId increments by 1 each Monday 00:00 UTC. 1970-01-05 was the first Monday.
export function weekIdAt(ms) {
    return Math.floor((Math.floor(ms / 86400000) - 4) / 7);
}

// Roll this-week / previous-week maxes for a fresh pred_egf_rank. `row` carries the
// stored week_id / this_week_max / prev_week_max / prev_week_gor_rank / gor_rank.
// Previous-week's GoR is FROZEN at rollover so a mid-week GoR change can't rewrite
// last week's relative delta. Returns the four week fields to persist.
export function rollWeek(row, nowWeek, predEgfRank) {
    const wid = row.week_id;
    if (wid === nowWeek) {
        const prev = Number.isFinite(row.this_week_max) ? row.this_week_max : -Infinity;
        return {
            week_id: nowWeek,
            this_week_max: Math.max(prev, predEgfRank),
            prev_week_max: row.prev_week_max ?? null,
            prev_week_gor_rank: row.prev_week_gor_rank ?? null,
        };
    }
    if (wid === nowWeek - 1) {
        return {
            week_id: nowWeek,
            this_week_max: predEgfRank,
            prev_week_max: Number.isFinite(row.this_week_max) ? row.this_week_max : null,
            prev_week_gor_rank: row.gor_rank ?? null,   // freeze the ending week's GoR
        };
    }
    // First report (week_id null), or an idle gap of > 1 week: no previous week.
    return {
        week_id: nowWeek,
        this_week_max: predEgfRank,
        prev_week_max: null,
        prev_week_gor_rank: null,
    };
}

// --- buffer (de)serialization ---------------------------------------------------
// Stored as a JSON array of { p, s, t }, newest first, capped at BUFFER_CAP.
export function parseBuffer(text) {
    if (!text) return [];
    try {
        const arr = JSON.parse(text);
        return Array.isArray(arr) ? arr : [];
    } catch {
        return [];
    }
}

export function serializeBuffer(buffer) {
    return JSON.stringify(buffer.length > BUFFER_CAP ? buffer.slice(0, BUFFER_CAP) : buffer);
}

// Prepend a new novel attempt (newest first), capping at BUFFER_CAP.
export function pushBuffer(buffer, problemId, success, timeUsed) {
    const next = [{ p: problemId, s: success ? 1 : 0, t: Number.isFinite(timeUsed) ? timeUsed : null }];
    for (let i = 0; i < buffer.length && next.length < BUFFER_CAP; i++) next.push(buffer[i]);
    return next;
}

// --- EGD lookup -----------------------------------------------------------------
// Public JSON, no key. Returns { egd_name, gor, country_code } or null on a network
// error / non-Ok retcode (callers keep the last known value on a transient miss).
export async function fetchEgdPlayer(pin) {
    try {
        const res = await fetch(
            `https://www.europeangodatabase.eu/EGD/GetPlayerDataByPIN.php?pin=${encodeURIComponent(pin)}`);
        if (!res || !res.ok) return null;
        const d = await res.json();
        if (!d || d.retcode !== 'Ok') return null;
        const name = `${d.Name || ''} ${d.Last_Name || ''}`.trim();
        const gor = Number(d.Gor);
        return {
            egd_name: name || null,
            gor: Number.isFinite(gor) ? Math.round(gor) : null,
            country_code: d.Country_Code || null,
        };
    } catch {
        return null;
    }
}

// --- seed a row's buffer + rank from tsumego_reports (the source of truth) -------
// The user's last <= BUFFER_CAP novel attempts, newest first, + the derived rank.
// Used by /register (seed an established player) and the cron reconcile.
export async function buildRowStateFromReports(env, userId, params) {
    const res = await env.DB.prepare(
        `SELECT problem AS p, success AS s, time_used AS t
           FROM tsumego_reports
          WHERE user_id = ?1 AND is_repeat = 0
          ORDER BY id DESC LIMIT ?2`
    ).bind(userId, BUFFER_CAP).all();
    const rows = (res && res.results) || [];
    const buffer = rows.map((r) => ({ p: r.p, s: r.s ? 1 : 0, t: Number.isFinite(r.t) ? r.t : null }));
    const { predEgfRank, se } = computeRank(buffer, params);
    return { buffer, predEgfRank, se };
}

// --- /report hot path -----------------------------------------------------------
// Best-effort: if `userId` is registered, append this NOVEL attempt to their buffer,
// recompute the rank (only once the buffer holds >= RANK_START_MIN), roll the week
// maxes, and persist. The point read IS the registration check. Never throws — the
// caller's tsumego_reports INSERT already succeeded and the cron reconcile heals any
// drift. Callers pass only novel (is_repeat = 0) attempts.
export async function updateLeaderboardOnReport(env, userId, problemId, success, timeUsed) {
    if (!env.DB) return;
    const row = await env.DB.prepare(
        `SELECT buffer, week_id, this_week_max, prev_week_max, prev_week_gor_rank, gor_rank
           FROM leaderboard WHERE user_id = ?1`
    ).bind(userId).first();
    if (!row) return;   // unregistered

    const params = await getProblemParams(env);
    const buffer = pushBuffer(parseBuffer(row.buffer), problemId, success, timeUsed);
    const { predEgfRank, se } = computeRank(buffer, params);
    const nowIso = new Date().toISOString();

    if (predEgfRank === null) {
        await env.DB.prepare(
            `UPDATE leaderboard SET buffer = ?2, last_active = ?3 WHERE user_id = ?1`
        ).bind(userId, serializeBuffer(buffer), nowIso).run();
        return;
    }
    const wk = rollWeek(row, weekIdAt(Date.now()), predEgfRank);
    await env.DB.prepare(
        `UPDATE leaderboard SET buffer = ?2, pred_egf_rank = ?3, pred_egf_se = ?4,
            week_id = ?5, this_week_max = ?6, prev_week_max = ?7, prev_week_gor_rank = ?8,
            last_active = ?9
          WHERE user_id = ?1`
    ).bind(userId, serializeBuffer(buffer), predEgfRank, se,
           wk.week_id, wk.this_week_max, wk.prev_week_max, wk.prev_week_gor_rank, nowIso).run();
}

// --- cron: weekly GoR refresh (leaves prev_week_gor_rank frozen) -----------------
export async function refreshGorForAll(env) {
    if (!env.DB) return;
    const res = await env.DB.prepare(`SELECT user_id, egd_pin FROM leaderboard`).all();
    for (const row of ((res && res.results) || [])) {
        const p = await fetchEgdPlayer(row.egd_pin);
        if (!p || p.gor === null) continue;   // keep last known on a transient miss
        await env.DB.prepare(
            `UPDATE leaderboard SET egd_name = ?2, country_code = ?3, gor = ?4,
                gor_rank = ?5, gor_updated_at = ?6 WHERE user_id = ?1`
        ).bind(row.user_id, p.egd_name, p.country_code, p.gor,
               gorToRankScalar(p.gor), new Date().toISOString()).run();
    }
}

// --- cron: rebuild buffers from tsumego_reports (heal drift + re-rate refresh) ---
// Preserves week history (week_id / prev_week_max / prev_week_gor_rank); only the
// buffer + current pred_egf_rank + this_week_max are refreshed.
export async function reconcileBuffers(env) {
    if (!env.DB) return;
    const params = await getProblemParams(env);
    const res = await env.DB.prepare(`SELECT user_id, this_week_max FROM leaderboard`).all();
    for (const row of ((res && res.results) || [])) {
        const st = await buildRowStateFromReports(env, row.user_id, params);
        if (st.predEgfRank === null) {
            await env.DB.prepare(
                `UPDATE leaderboard SET buffer = ?2, pred_egf_rank = NULL WHERE user_id = ?1`
            ).bind(row.user_id, serializeBuffer(st.buffer)).run();
            continue;
        }
        const twm = Math.max(Number.isFinite(row.this_week_max) ? row.this_week_max : -Infinity, st.predEgfRank);
        await env.DB.prepare(
            `UPDATE leaderboard SET buffer = ?2, pred_egf_rank = ?3, pred_egf_se = ?4,
                this_week_max = ?5 WHERE user_id = ?1`
        ).bind(row.user_id, serializeBuffer(st.buffer), st.predEgfRank, st.se, twm).run();
    }
}

// --- compute-on-read: ranked rows for the /leaderboard page ---------------------
// Applies the week rollover read-only (so an idle-this-week user displays correctly)
// and derives absolute + relative for current / this-week / last-week. Relative =
// rank − gor_rank; current + this-week use the live gor_rank, previous-week uses its
// frozen prev_week_gor_rank.
export async function readLeaderboardRows(env) {
    if (!env.DB) return [];
    const res = await env.DB.prepare(
        `SELECT user_id, egd_name, country_code, gor, gor_rank, prev_week_gor_rank,
                pred_egf_rank, week_id, this_week_max, prev_week_max
           FROM leaderboard WHERE pred_egf_rank IS NOT NULL`
    ).all();
    const nowWeek = weekIdAt(Date.now());
    return ((res && res.results) || []).map((r) => {
        let thisMax = r.this_week_max, prevMax = r.prev_week_max, prevGor = r.prev_week_gor_rank;
        if (r.week_id === nowWeek - 1) {
            prevMax = r.this_week_max; prevGor = r.gor_rank; thisMax = r.pred_egf_rank;
        } else if (r.week_id !== nowWeek) {
            prevMax = null; prevGor = null; thisMax = r.pred_egf_rank;
        }
        const gr = r.gor_rank;
        const rel = (v, g) => (Number.isFinite(v) && Number.isFinite(g)) ? v - g : null;
        return {
            name: r.egd_name, country: r.country_code, gor: r.gor, gorRank: gr,
            currentAbs: r.pred_egf_rank, thisWeekAbs: thisMax, prevWeekAbs: prevMax,
            currentRel: rel(r.pred_egf_rank, gr), thisWeekRel: rel(thisMax, gr),
            prevWeekRel: rel(prevMax, prevGor),
        };
    });
}
