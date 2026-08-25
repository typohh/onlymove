// pwa/src/leaderboard.js
//
// Leaderboard core (Cloudflare Worker side). ONE ROW PER USER, created lazily on
// the first novel /report (seeded from tsumego_reports) — every row is
// rebuildable, so the weekly cron prunes idle ones. The federation identity lives
// OUTSIDE this table: the EGD/AGA binding plus its display name and country in
// `external_ids` (written by handlers/link.js when the user links themselves), and
// the GoR history as weekly machine 'EGD' heartbeat rows in `rank_surveys` (rank =
// raw GoR digits; the latest row is also the "checked this week" marker). Reuses the
// SHARED window estimator so a user's leaderboard rank is computed by the SAME code
// as the front-page rank, from the user's materialized 500-attempt buffer. Used by
// the /report hot path, /link (seeding a freshly linked player), and the cron.

import { interpExtrap } from '../shared/rank-math.js';
import { rankLabelToRank } from '../shared/rank-math.js';
import {
    estimateWindowRankFromFlat, weightedMedianOverWindow, flattenAttempt, WIN_CAP,
    applyM, slopeInRankSpace, seFloorForMeanInfo, RANK_START_MIN,
} from '../shared/rank-window.js';
import { anchorInfo } from '../js/irt.js';
import { fetchAgaPlayer } from './aga.js';
// Inlined at build time (wrangler bundles JSON imports, like src/server-stats.js).
import RATING_RANK_MAP from '../tsumegos/rating_to_rank.json';
import RANK_MAP from '../tsumegos/rank_map.json';
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

// Rank is emitted — and the player LISTED — only once the buffer holds RANK_START_MIN
// VALID (still-in-corpus) novel attempts, the exact set computeRank fits. This is the
// uniform list-entry gate: a user below it (pred_rank NULL) is NOT shown regardless of
// registration (no EGD-pin bypass), and the page shows no sample count — only a footer
// tally of how many active players are still below it. The count is the valid one, NOT
// the raw buffer length: problems that leave the corpus drop out, so a player can fall
// back below the gate until they solve more still-in-corpus problems. It lives in
// shared/rank-window.js because the client applies the same gate to its Community
// "you" marker; re-exported here, where every Worker-side consumer already reads it.
export { RANK_START_MIN };
const TIME_MIN_TIMED = 20;              // >= this many valid solve times for a time dock
export const BUFFER_CAP = WIN_CAP;      // 500 — the buffer + the rank window share it

// One shared cross-board curve (map_rating_to_rank writes both size keys identical).
const CURVE = RATING_RANK_MAP['9'] || Object.values(RATING_RANK_MAP)[0] || null;
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

// The rank-dependent SE floor is no longer measured here (nor on the client): it is
// shipped as a corpus curve in se_floor.json and read through rank-window's seFloorAt,
// which estimateWindowRankFromFlat applies internally. That removal is the point —
// pred_rank is the SHOWN rank, so the Worker and the badge must price the floor from
// the same numbers, and a per-caller scan could not guarantee it (the client saw only
// the shards it had loaded; the offline rebuilds, which load problem_params.bin
// themselves and never call getProblemParams, saw a different index again).

export async function getProblemParams(env) {
    if (_params) return _params;
    if (!_paramsPromise) {
        _paramsPromise = loadProblemParams(env)
            .then((m) => { _params = m; return m; })
            .catch((e) => { _paramsPromise = null; throw e; });
    }
    return _paramsPromise;
}

// Every ranking system we hold a coupling for, in rank_map.json order — the
// /leaderboard page's units selector offers all of them (plus onlymove itself),
// and they are exactly the systems a user can declare a rank on in the survey
// (utils/rating/rank_surveys.py SERVERS). EGF is one of them, not a special case.
export const RANK_SYSTEMS = Object.keys(RANK_MAP || {});

// Forward coupling: an internal (time-adjusted-frame) rank -> a server's rank. The
// /leaderboard page's units selector shows the onlymove estimate in every server's
// units via this — EGF is NOT special here, it is one coupling among the others
// (see the consensus-mapping note in CLAUDE.md). Feed it the TIME-ADJUSTED rank:
// the couplings were fitted against time-adjusted ratings, so passing the untimed
// rank silently shifts every estimate by the user's dock.
export function internalToServerRank(server, internalRank) {
    const m = RANK_MAP && RANK_MAP[server];
    if (!m || !Array.isArray(m.our) || !Array.isArray(m.server) || m.our.length < 2) return null;
    if (internalRank === null || !Number.isFinite(internalRank)) return null;
    return interpExtrap(internalRank, m.our, m.server);
}

// Apply a rating-space delta to a rank (mirrors rank-decomposition.js
// rankAfterRatingDelta; invert M, shift in rating units, re-apply M).
function rankAfterRatingDelta(rank, deltaRating) {
    if (!CURVE || !Number.isFinite(rank) || !Number.isFinite(deltaRating)) return rank;
    const rating = interpExtrap(rank, CURVE.ranks, CURVE.ratings);
    return interpExtrap(rating + deltaRating, CURVE.ratings, CURVE.ranks);
}

// Add a personal time dock to a stored UNTIMED internal rank, returning the
// TIME-ADJUSTED rank the server couplings (and a real GoR / AGA rating) natively
// speak. The stored rank is untimed — the frame the client's front-page headline
// and the Community histogram use — so this is applied only where a server-frame
// number is being built. Monotone, so applying it to a range's min/max keeps them
// the min/max.
export function addTimeDockToInternalRank(internalRank, deltaRating) {
    if (!Number.isFinite(internalRank)) return null;
    if (!deltaRating) return internalRank;
    const adj = rankAfterRatingDelta(internalRank, deltaRating);
    return Number.isFinite(adj) ? adj : internalRank;
}

// The personal time dock in RATING units from a set of valid solve times (ms,
// NaN where untimed): alpha·(log10(weighted-median minutes) − center). 0 when
// there's no time signal or no TIME_ADJUST constants. This is the SAME dock
// computeRank applies to move the untimed onlymove rank into the server frame,
// factored out so readLeaderboardRows can recover and subtract it back off.
function timeDockFromTimes(times) {
    if (!TIME_ADJUST || !Number.isFinite(TIME_ADJUST.alpha_per_log10_min)
        || !Number.isFinite(TIME_ADJUST.center_log10_min)) return 0;
    const medMs = weightedMedianOverWindow(times, TIME_MIN_TIMED);
    if (medMs === null) return 0;
    return TIME_ADJUST.alpha_per_log10_min * (Math.log10(medMs / 60000) - TIME_ADJUST.center_log10_min);
}

// The personal dock straight from a buffer of { p, s, t } (newest first): the same
// valid-in-corpus solve times computeRank's median uses. The Worker's read paths get
// the dock from the stored time_delta_rating column instead; this stays exported for
// auxiliary/leaderboard_player_stats.mjs, which recomputes every row from raw reports
// and so has no stored value to read.
export function timeDeltaRatingFromBuffer(buffer, params) {
    const times = [];
    for (let i = buffer.length - 1; i >= 0; i--) {
        const e = buffer[i];
        const pp = e && typeof e.p === 'string' ? params.get(e.p) : null;
        if (!pp) continue;                            // left the corpus / malformed
        times.push(Number.isFinite(e.t) ? e.t : NaN);
    }
    return timeDockFromTimes(times);
}

// A Cloudflare edge geo-IP country (request.cf.country) is an ISO 3166-1 alpha-2
// code, or one of the non-geographic sentinels 'XX' (unknown) / 'T1' (Tor). Keep
// only a real 2-letter code (uppercased); the sentinels render as a broken flag
// glyph, and this column is the UNIVERSAL flag fallback (hits most anonymous rows),
// so garbage here is far more visible than in the old rare survey fallback. Returns
// null otherwise, so a COALESCE write never clobbers a known-good country with an
// unknown one.
export function normalizeCountry(cc) {
    if (typeof cc !== 'string') return null;
    const u = cc.trim().toUpperCase();
    if (!/^[A-Z]{2}$/.test(u) || u === 'XX' || u === 'T1') return null;
    return u;
}

// GoR (integer EGD rating) -> our rank. Standard EGF ~100-GoR/stone linearization
// anchored at 1k = 2000, 1d = 2100 (kept simple; the full nonlinear EGF relation
// could replace it later); the trailing -0.5 lands on the band CENTRE, since GoR
// 2000 is a mid-1k rather than the 1k/1d boundary our 0 marks. Null on a bad GoR.
export function gorToRank(gor) {
    if (typeof gor !== 'number' || !Number.isFinite(gor) || gor <= 0) return null;
    return (gor - 2000) / 100 - 0.5;
}

// --- external-rating helpers (the /leaderboard "Declared" column) -----------------
// The column shows the player's rank on the SELECTED ranking system: their own
// survey declaration, overridden by the machine heartbeat (EGD GoR / AGA rating)
// on EGF and AGA when they have a federation account linked.

// AGA rating -> a continuous rank on OUR scale, LABEL-PRESERVING: AGA's convention
// is dan = floor(rating) / kyu = floor(|rating|) (see src/aga.js), so AGA's "5d" is
// rating [5, 6) while ours is [4, 5) — shift the dan half down by 1 and the kyu half
// up by 1 and every AGA band lands exactly on ours. The jump at 0 is AGA's own
// (-1, 1) gap, which holds no real rating; null there (and on a legacy heartbeat row
// that stored a label rather than a rating). Mirrors aga_rating_to_value in
// utils/rating/rank_surveys.py, so a heartbeat reads the same band here as it does
// in the calibration pipeline — which is what puts an AGA rating and an AGA
// declaration on ONE scale in this column, and the estimate's AGA coupling with them.
export function agaRatingToRank(rating) {
    if (typeof rating !== 'number' || !Number.isFinite(rating)) return null;
    if (rating >= 1) return rating - 1;
    if (rating <= -1) return rating + 1;
    return null;
}

// Official short rank label ("4d" / "17k" / "3p") normalized, or null. Accepts the
// EGD Grade and AGA short-label forms; 'p' (pro) is kept verbatim.
export function shortRankLabel(label) {
    if (typeof label !== 'string') return null;
    const m = label.trim().toLowerCase().match(/^(\d{1,2})\s*([dkp])$/);
    return m ? m[1] + m[2] : null;
}

// Tier rank from an official label; pro grades count as dan tier. Null when the
// label is absent/unparseable (callers fall back to the rating-derived tier).
function tierRankFromLabel(label) {
    if (typeof label !== 'string') return null;
    if (/^\d{1,2}p$/.test(label)) return 6.5;    // pro -> dan tier (7d)
    return rankLabelToRank(label);
}

// EGD Last_Appearance tournament code ("T220122H") -> ISO date, or null. Two-digit
// year pivots at 70 (EGD data starts in 1996).
export function egdLastAppearanceToIso(code) {
    if (typeof code !== 'string') return null;
    const m = code.trim().toUpperCase().match(/^T(\d{2})(\d{2})(\d{2})[A-Z]?$/);
    if (!m) return null;
    const yy = parseInt(m[1], 10), mo = parseInt(m[2], 10), dd = parseInt(m[3], 10);
    if (mo < 1 || mo > 12 || dd < 1 || dd > 31) return null;
    const year = yy >= 70 ? 1900 + yy : 2000 + yy;
    return `${year}-${m[2]}-${m[3]}`;
}

// Bare "YYYY-MM-DD" or null (defensive re-validation when reading stored rows).
function isoDateOrNull(s) {
    if (typeof s !== 'string') return null;
    const m = s.trim().match(/^(\d{4}-\d{2}-\d{2})/);
    return m ? m[1] : null;
}

// EGF activity criterion: a rating is "active" while the player's last tournament
// is within 2 years for dan (and pro), 1 year for single-digit kyu, 6 months for
// double-digit kyu. Tier boundaries are band EDGES, which under this convention
// are the integers: dan >= 0 (the 1k/1d boundary), double-digit kyu < -9 (10k and
// weaker). FAIL-OPEN when the date is unknown (a register-seeded heartbeat has
// none until the first weekly refresh).
export function isVerifiedActive(tierRank, lastGameAt, nowMs = Date.now()) {
    if (!lastGameAt) return true;
    const t = Date.parse(lastGameAt);
    if (!Number.isFinite(t)) return true;
    const days = (nowMs - t) / 86400000;
    const win = !Number.isFinite(tierRank) ? 730
        : tierRank >= 0 ? 730
            : tierRank >= -9 ? 365
                : 183;
    return days <= win;
}

// The user's leaderboard rank from a buffer of { p, s, t } (newest FIRST). Resolves
// rating/slope/floor from `params` (dropping problems no longer in the corpus),
// flattens through the SHARED transform and runs the SHARED window estimator.
// Returns { rank, se, usable, deltaRating } — the estimator's OWN output frame,
// stored as-is: `rank` is the UNTIMED internal rank and `se` its
// standard error in the same units (stones), both straight from the window MLE with no
// coupling applied. `deltaRating` is the personal time dock in RATING units, computed
// here (the weighted-median solve time over the same window) and stored alongside, so a
// reader building a server-frame number applies it forward via addTimeDockToInternalRank
// instead of re-deriving it from the buffer on every read. `usable` is the count of
// buffered attempts still in the corpus (what the fit uses AND what the progress readout
// shows); rank is NULL when usable < RANK_START_MIN.
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
    if (flat.length < RANK_START_MIN) return { rank: null, se: null, usable: flat.length, deltaRating: 0 };

    // No seed of any kind: the estimator is a posterior over the population rank
    // prior, so it is a pure function of the recent history and reproduces the
    // client's number exactly.
    //
    // We store est.displayRank — the SHOWN rank, not the posterior mean. The
    // leaderboard has to agree with the badge the player is looking at, and with the
    // Community histogram built from the same column. Storing the mean here would put
    // a player's own front page and their leaderboard row several ranks apart while
    // their estimate is still wide. This does not reopen the calibration loop the
    // display offset is otherwise kept out of: the rank-map couplings are fitted on
    // the backend from rank_surveys + fit_user_rating, and read nothing from this
    // table. The offset scales with the se above a RANK-DEPENDENT floor, which comes
    // off the shipped se_floor.json curve inside the estimator — one set of numbers
    // for the Worker, the offline rebuilds and the client's own badge.
    const est = estimateWindowRankFromFlat(flat);
    if (!est) return { rank: null, se: null, usable: flat.length, deltaRating: 0 };

    // The dock is DERIVED here but NOT applied: the rank is stored untimed (the frame
    // both the front page and the Community histogram want), and the two consumers that
    // need a server frame add it back forward.
    return {
        rank: est.displayRank,
        se: est.se,
        usable: flat.length,
        deltaRating: timeDockFromTimes(times),
    };
}

// --- week bucketing (Monday-00:00-UTC week) -------------------------------------
// weekId increments by 1 each Monday 00:00 UTC. 1970-01-05 was the first Monday.
export function weekIdAt(ms) {
    return Math.floor((Math.floor(ms / 86400000) - 4) / 7);
}
// ISO timestamp of the current week's Monday 00:00 UTC (the "GoR is fresh this week"
// cutoff). weekId*7+4 is that Monday's day index since the epoch.
export function weekStartIso(ms) {
    return new Date((weekIdAt(ms) * 7 + 4) * 86400000).toISOString();
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
// --- GoR refresh via client piggyback on /report --------------------------------
// The Worker CANNOT fetch EGD (it's behind Cloudflare; a Worker→CF-zone subrequest
// is blocked, HTTP 403 "error code: 1106" — the orange-to-orange restriction). GoR
// is public (keyed on the EGD PIN, no user secret), so instead any ACTIVE client
// fetches registered players' current GoR from EGD (CORS *) and posts it back on its
// next /report. The report response advertises which PINs are still stale this week;
// updates are accepted once per week per PIN (the week's 'EGD' heartbeat row in
// rank_surveys IS the flag).

const GOR_STALE_TTL_MS = 5 * 60 * 1000;   // in-memory cache so /report doesn't scan D1 each call
const GOR_MIN = 100, GOR_MAX = 3000;      // plausibility bound on a submitted GoR

// Per-isolate cache of the PINs whose GoR hasn't been refreshed this week.
let _staleGor = { pins: [], at: 0, weekStart: '' };
// Drop the cache so the next /report re-scans. Called after a GoR write (the pin
// is fresh now) and after a link/unlink (the set of pins itself changed — a
// newly linked pin would otherwise wait out the TTL before any client is asked
// to fetch its GoR).
export function invalidateStaleGor() { _staleGor.at = 0; }

// PINs still needing a GoR refresh this week, cached in memory (keeps the high-volume
// /report path off D1 — the scan reads only external_ids + one pass of rank_surveys,
// both small). Stale = no 'EGD' heartbeat row this week (the latest heartbeat's
// created_at IS the freshness marker; see the rank_surveys schema note).
export async function getStaleGorPins(env) {
    if (!env.DB) return [];
    const now = Date.now();
    const weekStart = weekStartIso(now);
    if (_staleGor.weekStart === weekStart && (now - _staleGor.at) < GOR_STALE_TTL_MS) return _staleGor.pins;
    try {
        const res = await env.DB.prepare(
            `SELECT x.egd_pin FROM external_ids x
              LEFT JOIN (SELECT user_id, MAX(created_at) AS checked_at
                           FROM rank_surveys WHERE server = 'EGD' GROUP BY user_id) s
                     ON s.user_id = x.user_id
              WHERE s.checked_at IS NULL OR s.checked_at < ?1`
        ).bind(weekStart).all();
        const pins = ((res && res.results) || []).map((r) => r.egd_pin).filter(Boolean);
        _staleGor = { pins, at: now, weekStart };
        return pins;
    } catch (e) {
        console.error('getStaleGorPins failed:', e);
        return [];
    }
}

// Apply client-submitted GoR updates: one weekly heartbeat row per registered
// player into rank_surveys (server 'EGD', rank = raw GoR digits, country = EGD
// federation code, rank_label = the EGD Grade, last_game_at = the parsed
// Last_Appearance tournament date) — inserted even when the GoR is unchanged,
// since the row is also the "checked this week" marker. Once-per-week +
// plausibility + the pin-must-be-registered check are all enforced inside the
// INSERT ... SELECT, so a stale/duplicate/spoofed re-submit after the first is
// a silent no-op. A spoof is self-correcting: next week an honest client writes
// the true value.
export async function applyGorUpdates(env, updates) {
    if (!env.DB || !Array.isArray(updates) || !updates.length) return;
    const weekStart = weekStartIso(Date.now());
    const nowIso = new Date().toISOString();
    const stmts = [];
    for (const u of updates.slice(0, 50)) {
        const pin = String((u && u.pin) || '');
        if (!/^\d+$/.test(pin)) continue;
        const gor = Math.round(Number(u && u.gor));
        if (!Number.isFinite(gor) || gor < GOR_MIN || gor > GOR_MAX) continue;
        const cc = (typeof u.country === 'string' ? u.country : '').slice(0, 2).toUpperCase();
        const grade = shortRankLabel(u && u.grade);
        const lastGameAt = egdLastAppearanceToIso(u && u.lastAppearance);
        stmts.push(env.DB.prepare(
            `INSERT INTO rank_surveys (user_id, server, rank, created_at, country, rank_label, last_game_at)
             SELECT x.user_id, 'EGD', ?2, ?3, ?4, ?6, ?7
               FROM external_ids x
              WHERE x.egd_pin = ?1
                AND NOT EXISTS (SELECT 1 FROM rank_surveys s
                                 WHERE s.user_id = x.user_id AND s.server = 'EGD'
                                   AND s.created_at >= ?5)`
        ).bind(pin, String(gor), nowIso, cc || null, weekStart, grade, lastGameAt));
    }
    if (!stmts.length) return;
    try {
        await env.DB.batch(stmts);
        invalidateStaleGor();
    } catch (e) {
        console.error('applyGorUpdates failed:', e);
    }
}

// --- AGA rating refresh (Worker-side; weekly via cron) --------------------------
// Unlike EGD, the Worker CAN fetch the AGA Ratings Explorer (Azure, not Cloudflare),
// so AGA ranks are refreshed by the Worker itself on the weekly cron — no client
// piggyback. If it's ever unreachable, no heartbeat is written and the last one
// persists: AGA linking still works, just without that week's refresh. Mirrors the
// EGD get-stale / write-heartbeat shape, but self-fetches instead of accepting
// client-submitted values. Stores rank = the raw AGA rating (e.g. '5.12'/'-3.2' —
// what the "Declared" column shows), rank_label = the AGA's own official short
// label ("5d"), last_game_at = latest_event_date. (Pre-2026-07-21 heartbeat rows
// stored the label IN `rank`; readers treat those as rating-less and they age out
// with the next weekly heartbeat.)

const AGA_REFRESH_MAX = 200;   // cap Explorer fetches per cron run (best-effort)
const AGA_RATING_MIN = -40, AGA_RATING_MAX = 15;   // plausibility bound

// AGA member ids whose rating hasn't been refreshed this week (no 'AGAR' heartbeat
// since the week start). Small scan (external_ids + one grouped pass of
// rank_surveys), the AGA analog of getStaleGorPins. Not cached: called only from the
// weekly cron, not the hot /report path.
export async function getStaleAgaIds(env) {
    if (!env.DB) return [];
    const weekStart = weekStartIso(Date.now());
    try {
        const res = await env.DB.prepare(
            `SELECT x.aga_id FROM external_ids x
              LEFT JOIN (SELECT user_id, MAX(created_at) AS checked_at
                           FROM rank_surveys WHERE server = 'AGAR' GROUP BY user_id) s
                     ON s.user_id = x.user_id
              WHERE x.aga_id IS NOT NULL
                AND (s.checked_at IS NULL OR s.checked_at < ?1)`
        ).bind(weekStart).all();
        return ((res && res.results) || []).map((r) => r.aga_id).filter(Boolean);
    } catch (e) {
        console.error('getStaleAgaIds failed:', e);
        return [];
    }
}

// Fetch each stale AGA id's current rating from AGAGD (server-side) and write one
// weekly 'AGAR' heartbeat row per registered player. Best-effort: an unreachable
// AGAGD / bad fetch leaves that id stale for next week (the snapshot persists).
// Weekly-guarded + registration-checked inside the INSERT ... WHERE NOT EXISTS, so
// a re-run in the same week is a no-op — same shape as applyGorUpdates.
export async function refreshAgaRatings(env) {
    if (!env.DB) return;
    const ids = (await getStaleAgaIds(env)).slice(0, AGA_REFRESH_MAX);
    if (!ids.length) return;
    const weekStart = weekStartIso(Date.now());
    for (const agaId of ids) {
        let player = null;
        try { player = await fetchAgaPlayer(agaId); } catch (e) { player = null; }
        // Unreachable / bad / no numeric rating -> keep the last heartbeat.
        if (!player || !player.rankLabel || !Number.isFinite(player.rating)
            || player.rating < AGA_RATING_MIN || player.rating > AGA_RATING_MAX) continue;
        try {
            await env.DB.prepare(
                `INSERT INTO rank_surveys (user_id, server, rank, created_at, country, rank_label, last_game_at)
                 SELECT x.user_id, 'AGAR', ?2, ?3, NULL, ?4, ?5
                   FROM external_ids x
                  WHERE x.aga_id = ?1
                    AND NOT EXISTS (SELECT 1 FROM rank_surveys s
                                     WHERE s.user_id = x.user_id AND s.server = 'AGAR'
                                       AND s.created_at >= ?6)`
            ).bind(agaId, String(player.rating), new Date().toISOString(),
                player.rankLabel, player.latestEventDate || null, weekStart).run();
        } catch (e) {
            console.error('refreshAgaRatings insert failed:', e);
        }
    }
}

// --- seed a row's buffer + rank from tsumego_reports (the source of truth) -------
// The user's last <= BUFFER_CAP novel attempts, newest first, + the derived rank.
// Used by /link (seed a freshly linked established player) and the cron reconcile.
export async function buildRowStateFromReports(env, userId, params) {
    const res = await env.DB.prepare(
        `SELECT problem AS p, success AS s, time_used AS t
           FROM tsumego_reports
          WHERE user_id = ?1 AND is_repeat = 0
          ORDER BY id DESC LIMIT ?2`
    ).bind(userId, BUFFER_CAP).all();
    const rows = (res && res.results) || [];
    const buffer = rows.map((r) => ({ p: r.p, s: r.s ? 1 : 0, t: Number.isFinite(r.t) ? r.t : null }));
    const { rank, se, deltaRating } = computeRank(buffer, params);
    return { buffer, rank, se, deltaRating };
}

// --- /report hot path -----------------------------------------------------------
// Best-effort: append this NOVEL attempt to the user's buffer, recompute the rank
// (only once the buffer holds >= RANK_START_MIN), and persist. A user without a
// row yet gets one created here, seeded from their full tsumego_reports history
// (a one-time <= 500-row read per user; the just-inserted report is included
// since the caller's INSERT already succeeded). Never throws — the cron
// reconcile heals any drift. Callers pass only novel (is_repeat = 0) attempts.
export async function updateLeaderboardOnReport(env, userId, problemId, success, timeUsed, country) {
    if (!env.DB) return;
    const row = await env.DB.prepare(
        `SELECT buffer FROM leaderboard WHERE user_id = ?1`
    ).bind(userId).first();
    if (!row) { await createLeaderboardRow(env, userId, country); return; }

    const params = await getProblemParams(env);
    const buffer = pushBuffer(parseBuffer(row.buffer), problemId, success, timeUsed);
    const { rank, se, deltaRating } = computeRank(buffer, params);
    const nowIso = new Date().toISOString();
    // Stamp the request's edge geo-IP country into the SAME row write (no extra
    // write) — this column is the universal flag fallback for anonymous rows.
    // COALESCE keeps the last KNOWN country when this report's geo is unknown/Tor
    // (normalizeCountry -> null), so a sentinel never clobbers a good code; a real
    // code wins (last-writer-wins among valid codes, so the flag follows travel).
    const cc = normalizeCountry(country);

    if (rank === null) {
        await env.DB.prepare(
            `UPDATE leaderboard SET buffer = ?2, last_active = ?3,
                country = COALESCE(?4, country)
              WHERE user_id = ?1`
        ).bind(userId, serializeBuffer(buffer), nowIso, cc).run();
        return;
    }
    await env.DB.prepare(
        `UPDATE leaderboard SET buffer = ?2, pred_rank = ?3, pred_rank_se = ?4,
            time_delta_rating = ?5, last_active = ?6, country = COALESCE(?7, country)
          WHERE user_id = ?1`
    ).bind(userId, serializeBuffer(buffer), rank, se, deltaRating, nowIso, cc).run();
}

// Same, for a BATCH of novel attempts landing at once — the outbox flush behind
// /report-backfill. One read + one write regardless of batch size, instead of a
// per-row round trip.
//
// The buffer is ordered by INSERT ORDER (id), which is the one ordering every
// writer and every rebuild agrees on (buildRowStateFromReports, the cron
// reconcile, auxiliary/update_leaderboard_ranks.mjs). A backfilled row carries a
// historical created_at but a fresh id, so under that convention it IS the
// newest and prepending is correct — `attempts` must therefore arrive in
// ascending id order, i.e. the order the batch INSERT ran. Ordering the buffer
// by event time instead would need a (user_id, created_at) index that was
// deliberately dropped, and would make the heaviest user's rebuild read their
// whole history rather than the last BUFFER_CAP rows.
//
// Skipping the buffer entirely (the behaviour before this) left flushed attempts
// invisible to the leaderboard until the weekly cron reconcile.
export async function updateLeaderboardOnBackfill(env, userId, attempts, country) {
    if (!env.DB || !attempts.length) return;
    const row = await env.DB.prepare(
        `SELECT buffer FROM leaderboard WHERE user_id = ?1`
    ).bind(userId).first();
    // No row yet: seed from the full history, which already contains the rows
    // this batch just inserted — so the seed subsumes the push.
    if (!row) { await createLeaderboardRow(env, userId, country); return; }

    const params = await getProblemParams(env);
    let buffer = parseBuffer(row.buffer);
    for (const a of attempts) buffer = pushBuffer(buffer, a.problemId, a.success, a.timeUsed);
    const { rank, se, deltaRating } = computeRank(buffer, params);
    // last_active tracks PRESENCE, not the attempts' event time: the user is
    // online right now to be flushing, so an old created_at must not leave them
    // looking idle to pruneIdleLeaderboardRows.
    const nowIso = new Date().toISOString();
    const cc = normalizeCountry(country);
    if (rank === null) {
        await env.DB.prepare(
            `UPDATE leaderboard SET buffer = ?2, last_active = ?3,
                country = COALESCE(?4, country)
              WHERE user_id = ?1`
        ).bind(userId, serializeBuffer(buffer), nowIso, cc).run();
        return;
    }
    await env.DB.prepare(
        `UPDATE leaderboard SET buffer = ?2, pred_rank = ?3, pred_rank_se = ?4,
            time_delta_rating = ?5, last_active = ?6, country = COALESCE(?7, country)
          WHERE user_id = ?1`
    ).bind(userId, serializeBuffer(buffer), rank, se, deltaRating, nowIso, cc).run();
}

// First novel report from a row-less user (brand new, or pruned while idle): seed
// the row from their full report history (buildRowStateFromReports, which already
// includes the attempt that triggered this call). INSERT OR IGNORE makes the
// concurrent-first-reports race harmless — both seeds compute from the same
// reports, the loser is a no-op. Registration is not this table's concern
// (external_ids), so a returning registered player gets the identical seed.
async function createLeaderboardRow(env, userId, country) {
    const params = await getProblemParams(env);
    const st = await buildRowStateFromReports(env, userId, params);
    const nowIso = new Date().toISOString();
    await env.DB.prepare(
        `INSERT OR IGNORE INTO leaderboard
            (user_id, buffer, pred_rank, pred_rank_se, time_delta_rating, last_active, country)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)`
    ).bind(userId, serializeBuffer(st.buffer), st.rank, st.se, st.deltaRating, nowIso,
        normalizeCountry(country)).run();
}

// (The old cron-side refreshGorForAll was removed: the Worker can't reach EGD —
// see the GoR-piggyback note above — so GoR is refreshed by active clients now.)

// --- cron: prune idle rows (weekly, Monday 00:00 UTC) ----------------------------
// Every leaderboard row is fully rebuildable from tsumego_reports (registration
// lives in external_ids; createLeaderboardRow re-seeds a returning user on their
// next novel report — a <= BUFFER_CAP-row read + one insert), so idle rows are
// deleted rather than kept forever: the table stays at roughly weekly-active
// size, keeping every full-scan read (/leaderboard page, the reconcile below)
// cheap without a last_active index — which would cost an index write per novel
// report (see schema.sql). The >= 7-day cutoff matches the /leaderboard page's
// active-past-week filter, so a pruned row is one the page had already stopped
// listing — deletion loses nothing the system would ever serve.
// A racing /report is harmless: its UPDATE against the deleted row is a no-op and
// the attempt lives in tsumego_reports, so the next report re-seeds it in full.
export async function pruneIdleLeaderboardRows(env) {
    if (!env.DB) return;
    try {
        const cutoff = new Date(Date.now() - 7 * 86400000).toISOString();
        await env.DB.prepare(
            `DELETE FROM leaderboard
              WHERE last_active IS NULL OR last_active < ?1`
        ).bind(cutoff).run();
    } catch (e) {
        console.error('pruneIdleLeaderboardRows failed:', e);
    }
}

// --- cron: rebuild buffers from tsumego_reports (heal drift + re-rate refresh) ---
// Refreshes the buffer + current pred_rank/se/dock. Runs right after the prune,
// so the whole (small) table is the active set: each rebuild is a <= 500-row
// tsumego_reports read. A ratings re-ship refreshes every row via the offline
// auxiliary/update_leaderboard_ranks.mjs instead.
export async function reconcileBuffers(env) {
    if (!env.DB) return;
    const params = await getProblemParams(env);
    const res = await env.DB.prepare(
        `SELECT user_id FROM leaderboard`
    ).all();
    for (const row of ((res && res.results) || [])) {
        const st = await buildRowStateFromReports(env, row.user_id, params);
        if (st.rank === null) {
            await env.DB.prepare(
                `UPDATE leaderboard SET buffer = ?2, pred_rank = NULL WHERE user_id = ?1`
            ).bind(row.user_id, serializeBuffer(st.buffer)).run();
            continue;
        }
        await env.DB.prepare(
            `UPDATE leaderboard SET buffer = ?2, pred_rank = ?3, pred_rank_se = ?4,
                time_delta_rating = ?5
              WHERE user_id = ?1`
        ).bind(row.user_id, serializeBuffer(st.buffer), st.rank, st.se, st.deltaRating).run();
    }
}

// Resolve one leaderboard row's rank: the stored pred_rank (UNTIMED internal rank,
// 0 = the 1k/1d boundary) if present, else a recompute from the row's buffer (the stored value can lag
// a threshold change or corpus churn; the recompute surfaces a rank the instant the
// valid count reaches the gate, without waiting for the reconcile cron — and it is the
// path a row takes for one cycle after the pred_egf_rank -> pred_rank column change,
// which is why no backfill script is needed). `deltaRating` is the user's personal time
// dock in rating units, stored alongside the rank so the server-frame columns can add it
// forward without a second pass over the buffer. Returns rank = null for a row
// below RANK_START_MIN valid in-corpus attempts. Shared by the /leaderboard page and the
// Community rank histogram so the two can never disagree about a player's rank.
export function resolveRowRank(row, params) {
    let rank = Number.isFinite(row.pred_rank) ? row.pred_rank : null;
    let deltaRating = Number.isFinite(row.time_delta_rating) ? row.time_delta_rating : 0;
    if (rank === null) {
        const buf = row.buffer ? parseBuffer(row.buffer) : [];
        if (params && buf.length) {
            const c = computeRank(buf, params);
            deltaRating = c.deltaRating || 0;
            if (Number.isFinite(c.rank)) rank = c.rank;
        }
    }
    return { rank, deltaRating };
}

// Every player active in the window, with their UNTIMED onlymove rank scalar —
// the same quantity the front page displays and the /leaderboard page's onlymove
// units column shows. ONE scan of the (weekly-pruned) leaderboard table plus the
// small hidden_profiles read; the Community aggregation builds both its rank
// histogram and its country distribution from this, so it costs one scan, not two.
// `internal` is null below the RANK_START_MIN gate (tallied in `belowGate`), and
// `hidden` marks a user who opted out of the board — their rank still counts (the
// onlymove rank is never hidden), their country does not.
export async function readActivePlayers(env, { activeDays = 7 } = {}) {
    if (!env.DB) return { players: [], belowGate: 0 };
    const since = new Date(Date.now() - activeDays * 86400000).toISOString();
    const [res, hres] = await Promise.all([
        env.DB.prepare(
            `SELECT user_id, pred_rank, time_delta_rating, buffer, country FROM leaderboard
              WHERE last_active >= ?1`
        ).bind(since).all(),
        env.DB.prepare(`SELECT user_id FROM hidden_profiles`).all(),
    ]);
    const rows = (res && res.results) || [];
    const hidden = new Set(((hres && hres.results) || []).map((h) => h.user_id));
    let params = null;
    if (rows.some((r) => r.buffer)) {
        try { params = await getProblemParams(env); }
        catch (e) { console.error('active players params load failed:', e); }
    }
    const players = [];
    let belowGate = 0;
    for (const r of rows) {
        // The stored rank IS the untimed onlymove scalar this consumer wants — no
        // coupling and no dock to undo.
        const { rank } = resolveRowRank(r, params);
        if (rank === null) belowGate++;
        players.push({
            userId: r.user_id,
            country: r.country || null,
            hidden: hidden.has(r.user_id),
            internal: Number.isFinite(rank) ? rank : null,
        });
    }
    return { players, belowGate };
}

// --- compute-on-read: rows for the /leaderboard page -----------------------------
// Walks the (pruned, small) leaderboard table limited to rows active in the
// trailing `activeDays` window and returns { rows, belowMin }: `rows` are the
// players AT OR ABOVE the uniform list-entry gate (>= RANK_START_MIN valid
// in-corpus novel attempts — a real onlymove rank), `belowMin` is the count of
// active players still under it. The gate is uniform: a linked identity grants NO
// bypass, so a sub-gate linked player is counted in `belowMin`, not shown,
// and the page displays no sample count — only the aggregate tally. A hidden row
// is filtered below. The identity's display fields live OFF the leaderboard row:
// name/country on the `external_ids` binding (frozen at link time), and per-user
// external ranks from ONE latest-per-(user, server) pass over rank_surveys,
// stitched in JS.
//
// Each shown row carries two per-system maps keyed by RANK_SYSTEMS, so the page's
// units selector can switch frames with no further server round trip:
//   `est`  — the onlymove estimate coupled into each system's frame, TIME-ADJUSTED
//            (the frame those systems' own ratings speak, so the two columns are
//            comparable). `currentInternal` is the UNTIMED onlymove scalar the
//            onlymove-units column shows, matching the client's front-page headline.
//   `decl` — that system's rank for this player as { v, exact }: `v` a continuous
//            rank on the shared scale, `exact` true when it came from a MACHINE
//            heartbeat (sub-band precision — the page shows a progress percent) and
//            false for a human survey declaration (a band label, so `v` is only its
//            centre). The heartbeats WIN on their own system — the 'EGD' GoR on EGF
//            and the 'AGAR' rating on AGA — for a player with that federation
//            account linked, so a verified rank always displaces the self-report.
//            A heartbeat that is inactive per the EGF activity criterion
//            (isVerifiedActive; fail-open on a missing date) is dropped and the
//            player's own declaration stands in its place if they made one.
// `registered` (has an external_ids row; unlinked rows have name/country null)
// completes the row.
export async function readLeaderboardRows(env, { activeDays = 7 } = {}) {
    if (!env.DB) return { rows: [], belowMin: 0 };
    // buffer is shipped for EVERY row because a rank-less row (re)computes its rank from
    // it; a RANKED row no longer needs it, since the rank is stored in the frame this
    // page's default column shows and the dock rides along in its own column. Pulling it
    // regardless costs nothing extra (D1 bills rows, not the blob's bytes) and keeps the
    // one-cycle post-migration recompute path fed. ALL active rows are pulled — no
    // registration/rank filter — so the sub-gate rows can be tallied for the footer;
    // they're dropped from the visible list in JS below.
    // Name + country come off the BINDING (external_ids.display_name/country_code),
    // snapshotted from EGD/AGA when the user linked. One join, and unlinking takes
    // the name with it — which the claimed-invitation row this replaced could not do,
    // since a claim is a historical fact rather than a current state.
    const stmt = env.DB.prepare(
        `SELECT l.user_id, x.display_name, x.country_code, l.pred_rank, l.time_delta_rating,
                l.buffer, l.country,
                (x.user_id IS NOT NULL) AS registered
           FROM (SELECT *, user_id AS u FROM leaderboard
                  WHERE last_active >= ?1) l
           LEFT JOIN external_ids x ON x.user_id = u`
    ).bind(new Date(Date.now() - activeDays * 86400000).toISOString());
    // Latest declaration per (user, server) — one pass over the (small)
    // rank_surveys table; stitched onto rows in JS below. `country`/`rank_label`/
    // `last_game_at` are bare columns alongside MAX(id), so SQLite returns them
    // from the latest row (documented SQLite behavior for a bare column beside an
    // aggregate): the 'EGD' heartbeat carries the EGD federation code, other
    // servers the geo-IP country.
    const [res, declRes] = await Promise.all([
        stmt.all(),
        env.DB.prepare(
            `SELECT user_id, server, rank, country, rank_label, last_game_at, MAX(id)
               FROM rank_surveys
              GROUP BY user_id, server`
        ).all(),
    ]);
    const declByUser = new Map();
    for (const d of ((declRes && declRes.results) || [])) {
        if (!declByUser.has(d.user_id)) declByUser.set(d.user_id, []);
        declByUser.get(d.user_id).push(d);
    }
    const rows = (res && res.results) || [];
    // A row with no stored rank is (re)computed on read: (a) so a player who has just
    // crossed the gate is shown a rank immediately, without waiting for the reconcile
    // cron, and (b) so sub-gate players can be tallied for the footer. params is the
    // per-isolate cached asset; load it if any row has a buffer (needed to recompute a
    // row's rank AND to derive every row's time dock).
    let params = null;
    if (rows.some((r) => r.buffer)) {
        try { params = await getProblemParams(env); }
        catch (e) { console.error('leaderboard params load failed:', e); }
    }
    // Users who opted to hide their profile (bottom of the /leaderboard page, or
    // the checkbox in the rank survey — one flag, one table): withhold name,
    // country, and every declared / verified rank. Small opt-in table, read whole.
    // A hidden user still gets their onlymove estimate (they render as an
    // anonymous row carrying only that estimate).
    let hidden = new Set();
    try {
        const hres = await env.DB.prepare(`SELECT user_id FROM hidden_profiles`).all();
        for (const h of ((hres && hres.results) || [])) hidden.add(h.user_id);
    } catch (e) { console.error('hidden_profiles read failed:', e); }
    // Below-gate rows (< RANK_START_MIN valid in-corpus attempts) are COUNTED for the
    // footer tally and dropped; at-or-above-gate rows are stitched into display
    // identities. The gate is uniform — a linked identity gives no bypass — so an
    // unlinked ranked player and a linked one are shown alike, and a linked
    // sub-gate player is not shown at all.
    let belowMin = 0;
    const visible = [];
    for (const r of rows) {
        const isHidden = hidden.has(r.user_id);
        // Rank + personal time dock via the shared resolver (see resolveRowRank): the
        // rank arrives untimed, and the dock is added forward for the server frames.
        const { rank, deltaRating } = resolveRowRank(r, params);
        // Below the list-entry gate: not shown, only tallied. (A hidden sub-gate row is
        // counted too — the tally is a raw activity aggregate, not identifying.)
        if (rank === null) { belowMin++; continue; }
        // External ranks: the machine heartbeats ('EGD' GoR, 'AGAR' AGA rating) and
        // the player's own survey declarations, which the Declared column shows for
        // every system they have no linked account on. Country is harvested in the
        // same pass — the 'EGD' heartbeat's country is the EGD federation code (the
        // most reliable, since it comes from EGD), a human survey's is Cloudflare
        // geo-IP. All skipped entirely for a hidden user — their ranks and country
        // are exactly what they've opted to hide.
        let egd = null;             // { gor, label, lastGameAt } — verified EGD data
        let aga = null;             // { rating, label, lastGameAt } — verified AGA data
        let egdCountry = null;      // 'EGD' heartbeat — EGD federation code (reliable)
        let surveyCountry = null;   // any human survey row — Cloudflare geo-IP
        const declared = new Map(); // system -> the player's own latest declared label
        if (!isHidden) for (const d of (declByUser.get(r.user_id) || [])) {
            if (d.server === 'EGD') {
                const gor = parseInt(d.rank, 10);
                if (Number.isFinite(gor) && gor > 0) {
                    egd = {
                        gor,
                        label: shortRankLabel(d.rank_label),
                        lastGameAt: isoDateOrNull(d.last_game_at),
                    };
                }
                if (d.country) egdCountry = d.country;
            } else if (d.server === 'AGAR') {
                // New-format heartbeats store the raw AGA rating in `rank`; legacy
                // rows stored the label ("5d" -> NaN here) and read as rating-less.
                const rating = Number(d.rank);
                if (Number.isFinite(rating) && Math.abs(rating) >= 1) {
                    aga = {
                        rating,
                        label: shortRankLabel(d.rank_label),
                        lastGameAt: isoDateOrNull(d.last_game_at),
                    };
                }
            } else {
                // A human declaration. Unknown/retired system names are ignored —
                // `decl` is keyed by the systems we hold a coupling for.
                if (RANK_MAP && RANK_MAP[d.server]) declared.set(d.server, d.rank);
                if (d.country && !surveyCountry) surveyCountry = d.country;
            }
        }
        // EGF activity criterion: an inactive rating (last tournament too long ago for
        // its tier) is dropped, which under the precedence below hands that system's
        // cell back to the player's own declaration if they made one. Tier comes from
        // the official label when present (pro = dan tier), else from the rating
        // itself; a missing date fails open (register-seeded heartbeats have none for
        // <= a week).
        if (egd) {
            const tier = tierRankFromLabel(egd.label) ?? gorToRank(egd.gor);
            if (!isVerifiedActive(tier, egd.lastGameAt)) egd = null;
        }
        if (aga) {
            const tier = tierRankFromLabel(aga.label) ?? agaRatingToRank(aga.rating);
            if (!isVerifiedActive(tier, aga.lastGameAt)) aga = null;
        }
        // The onlymove estimate in every frame the units selector displays: the stored
        // untimed internal scalar (onlymove mode + that mode's sort key), and one
        // TIME-ADJUSTED coupling per ranking system — the frame those systems' own
        // ratings speak, so the dock MUST be added before any coupling is applied.
        const currentInternal = rank;
        const internalTimed = addTimeDockToInternalRank(rank, deltaRating);
        const est = {};
        for (const server of RANK_SYSTEMS) {
            const v = internalToServerRank(server, internalTimed);
            if (Number.isFinite(v)) est[server] = v;
        }
        // The Declared column's value per system, on the shared rank scale. A
        // declaration is a band label, so its scalar is that band's CENTRE and
        // carries no within-band progress (exact: false); a heartbeat resolves
        // within the band, and OVERRIDES the declaration on its own system.
        const decl = {};
        for (const server of RANK_SYSTEMS) {
            const v = rankLabelToRank(declared.get(server));
            if (Number.isFinite(v)) decl[server] = { v, exact: false };
        }
        if (egd) {
            const v = gorToRank(egd.gor);
            if (Number.isFinite(v)) decl.EGF = { v, exact: true };
        }
        if (aga) {
            const v = agaRatingToRank(aga.rating);
            if (Number.isFinite(v)) decl.AGA = { v, exact: true };
        }
        // A hidden user is stripped of name/country and reported as unlinked, so the
        // page renders them as an anonymous ranked row. Country falls back GoR (EGD, most
        // reliable) → the binding (EGD, frozen at link time) → leaderboard row (edge geo-IP,
        // stamped on every /report — the universal fallback that gives ANY active player a
        // flag) → survey (older geo-IP), so the flag shows for anyone with a country from
        // any source, not just linked rows.
        const name = isHidden ? null : r.display_name;
        const country = isHidden ? null
            : (egdCountry || r.country_code || r.country || surveyCountry || null);
        const registered = isHidden ? false : !!r.registered;
        visible.push({ name, country, registered, currentInternal, est, decl });
    }
    return { rows: visible, belowMin };
}
