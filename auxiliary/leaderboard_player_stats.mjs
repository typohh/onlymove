#!/usr/bin/env node
// auxiliary/leaderboard_player_stats.mjs
//
// Read-only report: for every leaderboard player with a LINKED external account, print
//   - novel (is_repeat = 0) 9x9 attempt count
//   - novel 13x13 attempt count
//   - median solve time over their novel attempts
//   - attempts per local day of the ongoing week
//   - their onlymove rank    (the untimed front-page window MLE, decimal)
//   - their est. EGF rank    (the time-adjusted EGF-coupled rank, decimal)
//   - their real EGF rank    (from the verified 'EGD' GoR heartbeat)
//   - the same pair in AGA units, when any listed player has a verified AGA rating
// Then a summary CORRELATING onlymove's predicted strength against the player's real
// verified rating — Pearson + Spearman (+ MAE / bias) over three cohorts: everyone,
// the production >= RANK_START_MIN gate, and that gate plus production's
// verified-activity criterion (a player's 20-year-old federation rating is stale
// ground truth, not a model error). Pearson r is affine-invariant, so these
// scalar-space figures equal the predicted-GoR vs actual-GoR correlations.
//
// The ranks are computed BIT-IDENTICALLY to production: this script esbuild-bundles
// pwa/src/leaderboard.js itself (so the curve/coupling/time-adjust data and every
// helper — RANK_START_MIN, the EGF/AGA couplings, the personal time dock, the
// verified-activity gate — are the shipped ones) and runs the shared window
// estimator over each row's STORED buffer (the last <=500 novel attempts, the exact
// input the Worker's computeRank uses). The onlymove rank is the untimed internal
// rank scalar (est.rankIndex - RANK_INDEX_OFFSET) — what the client's front page and
// the /leaderboard page's onlymove units show; est. EGF/AGA re-apply the
// median-solve-time dock and map through that server's coupling, the frame a real
// GoR / AGA rating natively speaks. All in the 1k=0 / 1d=1 scalar convention.
//
// Display identity comes from the same three tables the /leaderboard page stitches:
// the rank buffer off `leaderboard`, the name off the claimed `invitation` row, and
// the verified ratings off the latest-per-(user, server) machine heartbeats in
// `rank_surveys` ('EGD' rank = raw GoR digits, 'AGAR' rank = raw AGA rating). Human
// survey declarations are NOT read — they are calibration input, not ground truth
// about a player's federation rating.
//
// Data source: attempts/database.sqlite (the local D1 mirror) + the current local
// pwa/tsumegos/{problem_params.bin, rating_to_rank.json, rank_map.json,
// population_constants.json}. Run `./.venv/bin/python3 -m pipeline.fetch_database`
// first so the leaderboard rows + tsumego_reports are fresh.
//
// Usage:
//   node auxiliary/leaderboard_player_stats.mjs            # linked players, strongest first
//   node auxiliary/leaderboard_player_stats.mjs --all      # every leaderboard row (all users)
//   node auxiliary/leaderboard_player_stats.mjs --json     # machine-readable JSON ({ correlations, players })
//   node auxiliary/leaderboard_player_stats.mjs --user <redacted-user-id>   # limit (repeatable)

import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PWA_DIR = path.join(REPO, 'pwa');
const DB_PATH = path.join(REPO, 'attempts', 'database.sqlite');
const PARAMS_BIN = path.join(PWA_DIR, 'tsumegos', 'problem_params.bin');
const STALE_WARN_MS = 6 * 3600 * 1000;

// --- CLI ------------------------------------------------------------------------
const args = process.argv.slice(2);
let asJson = false;
let showAll = false;
const onlyUsers = new Set();
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--json') asJson = true;
    else if (a === '--all') showAll = true;
    else if (a === '--user') onlyUsers.add(args[++i]);
    else { console.error(`unknown arg: ${a}`); process.exit(1); }
}

// --- bundle the production rank code ---------------------------------------------
// Imports pwa/src/leaderboard.js directly, so the couplings, the time dock and the
// activity gate can't drift from what the Worker ships. computeRankBoth mirrors its
// computeRank but ALSO returns the untimed onlymove rank and the AGA-frame estimate,
// and deliberately drops the RANK_START_MIN gate (see below).
async function loadRankModule() {
    const require = createRequire(path.join(PWA_DIR, 'package.json'));
    const esbuild = require('esbuild');
    const res = await esbuild.build({
        stdin: {
            contents: `
                import { interpExtrap, RANK_INDEX_OFFSET, rankScalarToLabel, rankLabelToScalar } from './shared/rank-math.js';
                import { estimateWindowRankFromFlat, flattenAttempt } from './shared/rank-window.js';
                import {
                    RANK_START_MIN, BUFFER_CAP, gorToRankScalar, agaRatingToTierScalar, shortRankLabel,
                    isVerifiedActive, internalToServerScalar, timeDeltaRatingFromBuffer, parseBuffer,
                } from './src/leaderboard.js';
                import RATING_RANK_MAP from './tsumegos/rating_to_rank.json';

                // One shared cross-board curve, as in leaderboard.js.
                const CURVE = RATING_RANK_MAP['9'] || Object.values(RATING_RANK_MAP)[0] || null;

                export {
                    RANK_START_MIN, BUFFER_CAP, gorToRankScalar, agaRatingToTierScalar,
                    isVerifiedActive, rankScalarToLabel, parseBuffer,
                };

                // Mirrors leaderboard.js's (private) rankIndexAfterRatingDelta.
                function rankIndexAfterRatingDelta(rankIndex, deltaRating) {
                    if (!CURVE || !Number.isFinite(rankIndex) || !Number.isFinite(deltaRating)) return rankIndex;
                    const rating = interpExtrap(rankIndex, CURVE.rank_indices, CURVE.ratings);
                    return interpExtrap(rating + deltaRating, CURVE.ratings, CURVE.rank_indices);
                }

                // Mirrors leaderboard.js's (private) tierScalarFromLabel, with the
                // caller's rating-derived tier as the fallback — the activity window
                // the /leaderboard page gates a verified rating on.
                export function verifiedTierScalar(label, ratingTier) {
                    const short = shortRankLabel(label);
                    if (short) {
                        if (/^\\d{1,2}p$/.test(short)) return 7;         // pro -> dan tier
                        const v = rankLabelToScalar(short);
                        if (Number.isFinite(v)) return v;
                    }
                    return ratingTier;
                }

                // buffer: [{p,s,t}] newest first (the stored leaderboard buffer).
                export function computeRankBoth(buffer, params) {
                    const flat = [];
                    for (let i = buffer.length - 1; i >= 0; i--) {   // oldest -> newest
                        const e = buffer[i];
                        const pp = e && typeof e.p === 'string' ? params.get(e.p) : null;
                        if (!pp) continue;                            // left the corpus
                        const f = flattenAttempt(pp.rating, pp.slope, pp.floor, e.s, CURVE);
                        if (!f) continue;
                        flat.push(f);
                    }
                    // NOTE: production gates a rank on flat.length >= RANK_START_MIN.
                    // This report deliberately DROPS that gate — the window MLE runs on
                    // whatever novel attempts exist (a weak prior, WIN_PRIOR_SD=4, keeps it
                    // well-posed), so even not-yet-listed players get a (low-confidence) rank.
                    const est = estimateWindowRankFromFlat(flat, NaN);
                    if (!est) {
                        return { onlymoveRank: null, egfRank: null, agaRank: null, se: null, usable: flat.length };
                    }
                    const onlymoveRank = est.rankIndex - RANK_INDEX_OFFSET;   // untimed headline
                    // The personal time dock moves the untimed rank into the server frame
                    // the couplings (and a real GoR / AGA rating) speak.
                    const deltaRating = timeDeltaRatingFromBuffer(buffer, params);
                    const timedIndex = deltaRating
                        ? rankIndexAfterRatingDelta(est.rankIndex, deltaRating) : est.rankIndex;
                    const timedScalar = timedIndex - RANK_INDEX_OFFSET;
                    return {
                        onlymoveRank,
                        egfRank: internalToServerScalar('EGF', timedScalar),
                        agaRank: internalToServerScalar('AGA', timedScalar),
                        se: est.se, usable: flat.length,
                    };
                }
            `,
            resolveDir: PWA_DIR,
            sourcefile: 'aux-player-stats-entry.mjs',
        },
        bundle: true,
        format: 'esm',
        platform: 'neutral',
        write: false,
        logLevel: 'silent',
    });
    const code = res.outputFiles[0].text;
    return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}

// Same 16-byte record as leaderboard.js loadProblemParams: 4-byte id + 3 LE float32.
function loadProblemParamsLocal() {
    const bytes = fs.readFileSync(PARAMS_BIN);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const REC = 16;
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

function sqliteJson(sql) {
    const out = execFileSync('sqlite3', ['-json', DB_PATH, sql],
        { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 });
    const text = out.trim();
    return text ? JSON.parse(text) : [];
}

// The mirror can predate a migration (pwa/migrations/*.sql lands on the live D1 first),
// so every column added after this script was written is probed before it's selected.
function tableColumns(table) {
    return new Set(sqliteJson(`PRAGMA table_info(${table})`).map((r) => r.name));
}

const rankStr = (mod, v) => (Number.isFinite(v) ? `${v >= 0 ? ' ' : ''}${v.toFixed(2)} (${mod.rankScalarToLabel(v)})` : '—');
function medianStr(ms) {
    if (!Number.isFinite(ms)) return '—';
    const s = ms / 1000;
    if (s < 120) return `${s.toFixed(1)}s`;
    return `${(s / 60).toFixed(1)}m`;
}
function pad(s, n) { s = String(s); return s.length >= n ? s : s + ' '.repeat(n - s.length); }
function padl(s, n) { s = String(s); return s.length >= n ? s : ' '.repeat(n - s.length) + s; }

// --- current-week day columns ---------------------------------------------------
// Days of the ONGOING week (Monday .. today) in LOCAL time, so each column matches
// "today" for whoever runs it (the machine here is EEST = the Finnish tournament TZ).
// created_at is stored as ISO-8601 UTC, so the per-day SQL groups on
// date(created_at,'localtime') to bucket by the same local calendar day.
const DAYW = 7;
const WEEKDAY = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
function localDateStr(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;   // 'YYYY-MM-DD', lexically comparable + matches SQLite date()
}
function currentWeekDays() {
    const now = new Date();
    const isoDow = (now.getDay() + 6) % 7;   // 0=Mon .. 6=Sun
    const days = [];
    for (let i = 0; i <= isoDow; i++) {
        const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - isoDow + i);
        days.push({ date: localDateStr(d), label: `${WEEKDAY[i]} ${d.getDate()}` });
    }
    return days;
}

// --- correlation helpers (Pearson + tie-aware Spearman) -------------------------
function pearson(xs, ys) {
    const n = Math.min(xs.length, ys.length);
    if (n < 2) return NaN;
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
        const dx = xs[i] - mx, dy = ys[i] - my;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    const d = Math.sqrt(sxx * syy);
    return d > 0 ? sxy / d : NaN;
}
// Fractional ranks with ties averaged (so Spearman handles equal values correctly).
function rankdata(a) {
    const order = a.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0]);
    const ranks = new Array(a.length);
    for (let i = 0; i < order.length;) {
        let j = i;
        while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++;
        const avg = (i + j) / 2 + 1;   // 1-based average rank over the tie block
        for (let k = i; k <= j; k++) ranks[order[k][1]] = avg;
        i = j + 1;
    }
    return ranks;
}
function spearman(xs, ys) { return pearson(rankdata(xs), rankdata(ys)); }

// Correlation + error summary over [predicted, actual] pairs.
function corrStats(pairs) {
    const xs = pairs.map((p) => p[0]);
    const ys = pairs.map((p) => p[1]);
    const errs = pairs.map((p) => p[0] - p[1]);
    const n = pairs.length;
    const mean = (arr) => arr.reduce((s, v) => s + v, 0) / arr.length;
    return {
        n,
        pearson: pearson(xs, ys),
        spearman: spearman(xs, ys),
        mae: n ? mean(errs.map(Math.abs)) : NaN,
        bias: n ? mean(errs) : NaN,
        rmse: n ? Math.sqrt(mean(errs.map((e) => e * e))) : NaN,
    };
}

async function main() {
    if (!fs.existsSync(DB_PATH)) {
        console.error(`missing ${DB_PATH} — run: ./.venv/bin/python3 -m pipeline.fetch_database`);
        process.exit(1);
    }
    if (!fs.existsSync(PARAMS_BIN)) {
        console.error(`missing ${PARAMS_BIN} — run: ./.venv/bin/python3 -m pipeline.package_for_pwa`);
        process.exit(1);
    }

    const [{ m: latest }] = sqliteJson('SELECT MAX(created_at) AS m FROM tsumego_reports');
    const ageMs = Date.now() - Date.parse(latest || 0);
    if (!Number.isFinite(ageMs) || ageMs > STALE_WARN_MS) {
        console.warn(`WARNING: local mirror's newest report is ${latest || 'unknown'} `
            + `(${Number.isFinite(ageMs) ? Math.round(ageMs / 3600000) : '?'}h old). `
            + 'Run pipeline.fetch_database first for fresh numbers.\n');
    }

    const mod = await loadRankModule();
    const params = loadProblemParamsLocal();

    const lbRows = sqliteJson('SELECT user_id, buffer FROM leaderboard');

    // Display name off the claimed invitation row (frozen at invite time) — the same
    // source readLeaderboardRows stitches. MAX(claimed_at) with bare columns picks the
    // newest claim (documented SQLite behavior).
    const invRows = sqliteJson(
        `SELECT claimed_by_user_id AS user_id, egd_name, MAX(claimed_at)
           FROM invitation WHERE claimed_by_user_id IS NOT NULL
          GROUP BY claimed_by_user_id`
    );
    const invBy = new Map(invRows.map((r) => [r.user_id, r]));
    const linkedIds = new Set(sqliteJson('SELECT user_id FROM external_ids').map((r) => r.user_id));

    // Verified external ratings: the latest MACHINE heartbeat per (user, server) —
    // 'EGD' rank = raw GoR digits, 'AGAR' rank = raw AGA rating. Human survey rows
    // (OGS/KGS/Fox/…) are deliberately not read. rank_label/last_game_at arrived with
    // pwa/migrations/2026-07-21_rank_surveys_verified.sql, so they're probed first.
    const surveyCols = tableColumns('rank_surveys');
    const hasVerifiedCols = surveyCols.has('rank_label') && surveyCols.has('last_game_at');
    const verifiedSelect = hasVerifiedCols ? 'rank_label, last_game_at,' : '';
    const hbRows = sqliteJson(
        `SELECT user_id, server, rank, ${verifiedSelect} MAX(id)
           FROM rank_surveys WHERE server IN ('EGD', 'AGAR')
          GROUP BY user_id, server`
    );
    const egdBy = new Map(), agaBy = new Map();
    for (const h of hbRows) (h.server === 'EGD' ? egdBy : agaBy).set(h.user_id, h);

    // Default scope: players with a linked external account (the successor to the old
    // "everyone on the EGD leaderboard" — the table now holds one row per ACTIVE USER,
    // anonymous rows included). --all lists every leaderboard row too.
    const isLinked = (uid) => linkedIds.has(uid) || invBy.has(uid) || egdBy.has(uid) || agaBy.has(uid);
    // The weekly cron PRUNES idle leaderboard rows, so a linked player who stopped
    // playing has no row (and no stored buffer) — they'd silently vanish from the
    // ground-truth set. Include every linked user regardless, rebuilding the missing
    // buffer from tsumego_reports below exactly as buildRowStateFromReports does.
    const byId = new Map(lbRows.map((r) => [r.user_id, { ...r, rebuilt: false }]));
    for (const uid of new Set([...linkedIds, ...invBy.keys(), ...egdBy.keys(), ...agaBy.keys()])) {
        if (!byId.has(uid)) byId.set(uid, { user_id: uid, buffer: null, rebuilt: true });
    }
    let rows = [...byId.values()].filter((r) => showAll || isLinked(r.user_id));
    if (onlyUsers.size) rows = rows.filter((r) => onlyUsers.has(r.user_id));
    if (!rows.length) { console.log('no matching players'); return; }

    const idList = rows.map((r) => `'${r.user_id.replace(/'/g, "''")}'`).join(',');

    // Rebuild the pruned rows' buffers: the last BUFFER_CAP novel attempts, newest
    // first — the same set (and order) buildRowStateFromReports seeds a row from.
    const rebuildIds = rows.filter((r) => r.rebuilt).map((r) => `'${r.user_id.replace(/'/g, "''")}'`);
    if (rebuildIds.length) {
        const seed = sqliteJson(
            `SELECT user_id, p, s, t FROM (
                SELECT user_id, problem AS p, success AS s, time_used AS t,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id DESC) AS rn
                  FROM tsumego_reports
                 WHERE is_repeat = 0 AND user_id IN (${rebuildIds.join(',')})
             ) WHERE rn <= ${mod.BUFFER_CAP} ORDER BY user_id, rn`
        );
        const seedBy = new Map();
        for (const s of seed) {
            if (!seedBy.has(s.user_id)) seedBy.set(s.user_id, []);
            seedBy.get(s.user_id).push({ p: s.p, s: s.s ? 1 : 0, t: Number.isFinite(s.t) ? s.t : null });
        }
        for (const r of rows) if (r.rebuilt) r.buffer = JSON.stringify(seedBy.get(r.user_id) || []);
    }

    // Novel attempt counts by size (whole history, is_repeat = 0).
    const counts = sqliteJson(
        `SELECT user_id, size, COUNT(*) AS n
           FROM tsumego_reports
          WHERE is_repeat = 0 AND user_id IN (${idList})
          GROUP BY user_id, size`
    );
    const countBy = new Map();   // user_id -> {9, 13}
    for (const c of counts) {
        if (!countBy.has(c.user_id)) countBy.set(c.user_id, { 9: 0, 13: 0 });
        countBy.get(c.user_id)[c.size] = c.n;
    }

    // Median solve time over novel attempts (time_used > 0 ms; <=0 = untimed, dropped
    // — same convention the rank's weightedMedianOverWindow uses).
    const meds = sqliteJson(
        `WITH novel AS (
            SELECT user_id, time_used
              FROM tsumego_reports
             WHERE is_repeat = 0 AND time_used > 0
               AND user_id IN (${idList})
         ), ranked AS (
            SELECT user_id, time_used,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY time_used) AS rn,
                   COUNT(*)     OVER (PARTITION BY user_id)                     AS cnt
              FROM novel
         )
         SELECT user_id, AVG(time_used) AS median_ms, MAX(cnt) AS n_timed
           FROM ranked
          WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
          GROUP BY user_id`
    );
    const medBy = new Map(meds.map((m) => [m.user_id, m]));

    // Attempts (all reports, incl. repeats) per LOCAL calendar day of the ongoing week.
    const weekDays = currentWeekDays();
    const weekStart = weekDays[0].date;
    const perDayRows = sqliteJson(
        `SELECT user_id, date(created_at, 'localtime') AS d, COUNT(*) AS n
           FROM tsumego_reports
          WHERE date(created_at, 'localtime') >= '${weekStart}'
            AND user_id IN (${idList})
          GROUP BY user_id, d`
    );
    const perDayBy = new Map();   // user_id -> { 'YYYY-MM-DD': n }
    for (const dr of perDayRows) {
        if (!perDayBy.has(dr.user_id)) perDayBy.set(dr.user_id, {});
        perDayBy.get(dr.user_id)[dr.d] = dr.n;
    }

    const out = rows.map((r) => {
        const { onlymoveRank, egfRank, agaRank, se, usable } =
            mod.computeRankBoth(mod.parseBuffer(r.buffer), params);
        const c = countBy.get(r.user_id) || { 9: 0, 13: 0 };
        const md = medBy.get(r.user_id) || {};
        const perDayObj = perDayBy.get(r.user_id) || {};
        const perDay = {};
        let weekTotal = 0;
        for (const d of weekDays) { const n = perDayObj[d.date] || 0; perDay[d.date] = n; weekTotal += n; }

        // Real (verified) strength from the machine heartbeats, in the same rank-scalar
        // frame as the estimates. `stale` = production would WITHHOLD this rating as out
        // of date (EGF activity criterion); shown with a * here, not dropped.
        const egd = egdBy.get(r.user_id) || null;
        const aga = agaBy.get(r.user_id) || null;
        const gor = egd ? parseInt(egd.rank, 10) : NaN;
        const agaRating = aga ? Number(aga.rank) : NaN;   // legacy label rows -> NaN
        const realEgfRank = Number.isFinite(gor) && gor > 0 ? mod.gorToRankScalar(gor) : null;
        const realAgaRank = Number.isFinite(agaRating) && Math.abs(agaRating) >= 1
            ? mod.agaRatingToTierScalar(agaRating) : null;
        const staleOf = (rec, tierFallback) => (!rec || !hasVerifiedCols ? false
            : !mod.isVerifiedActive(mod.verifiedTierScalar(rec.rank_label, tierFallback), rec.last_game_at));

        return {
            userId: r.user_id, name: (invBy.get(r.user_id) || {}).egd_name || r.user_id,
            registered: linkedIds.has(r.user_id), rebuilt: !!r.rebuilt,
            gor: Number.isFinite(gor) ? gor : null,
            agaRating: Number.isFinite(agaRating) ? agaRating : null,
            novel9: c[9] || 0, novel13: c[13] || 0,
            medianMs: Number.isFinite(md.median_ms) ? md.median_ms : null,
            perDay, weekTotal,
            onlymoveRank, estEgfRank: egfRank, realEgfRank,
            estAgaRank: agaRank, realAgaRank,
            egfStale: realEgfRank !== null && staleOf(egd, realEgfRank),
            agaStale: realAgaRank !== null && staleOf(aga, realAgaRank),
            se, usableNovel: usable,
        };
    });
    // Strongest first by onlymove rank; unranked (null) sink to the bottom.
    out.sort((a, b) => (b.onlymoveRank ?? -Infinity) - (a.onlymoveRank ?? -Infinity));

    // --- correlations: onlymove's predicted strength vs the player's real rating ----
    // Pearson r is invariant under the affine GoR<->scalar map (gor = 2000 + 100·scalar),
    // so these scalar-space numbers ARE the predicted-GoR vs actual-GoR correlations.
    // Three cohorts: everyone with a rating, production's rank gate, and — since a
    // long-idle federation rating is stale ground truth, not a model error — that gate
    // plus the verified-activity criterion production itself displays on.
    const RANKED_MIN = mod.RANK_START_MIN;   // production's list-entry / rank gate
    const fin = (v) => Number.isFinite(v);
    const pairsOf = (key, actualKey, keep) => out
        .filter((p) => fin(p[key]) && fin(p[actualKey]) && keep(p))
        .map((p) => [p[key], p[actualKey]]);
    const cohorts = (key, actualKey, staleKey) => ({
        all: corrStats(pairsOf(key, actualKey, () => true)),
        ranked: corrStats(pairsOf(key, actualKey, (p) => p.usableNovel >= RANKED_MIN)),
        rankedActive: corrStats(pairsOf(key, actualKey,
            (p) => p.usableNovel >= RANKED_MIN && !p[staleKey])),
    });
    const correlations = {
        estVsRealEgf: cohorts('estEgfRank', 'realEgfRank', 'egfStale'),
        onlymoveVsRealEgf: cohorts('onlymoveRank', 'realEgfRank', 'egfStale'),
        estVsRealAga: cohorts('estAgaRank', 'realAgaRank', 'agaStale'),
        onlymoveVsRealAga: cohorts('onlymoveRank', 'realAgaRank', 'agaStale'),
    };

    if (asJson) { console.log(JSON.stringify({ correlations, players: out }, null, 2)); return; }

    // AGA columns only earn their width once someone has a verified AGA rating.
    const anyAga = out.some((p) => Number.isFinite(p.realAgaRank));
    const W = { name: 20, s9: 8, s13: 8, med: 9, n: 6, wk: 7, om: 15, est: 15, real: 16 };
    const dayHeader = weekDays.map((d) => padl(d.label, DAYW)).join('');
    const dayCells = (p) => weekDays.map((d) => padl((p.perDay[d.date] || 0).toLocaleString('en-US'), DAYW)).join('');
    const realStr = (v, stale) => (Number.isFinite(v) ? rankStr(mod, v) + (stale ? ' *' : '') : '—');
    console.log(
        pad('Player', W.name) + padl('9×9', W.s9) + padl('13×13', W.s13)
        + padl('median', W.med) + padl('buf', W.n) + '  ' + dayHeader + padl('wk', W.wk) + '  '
        + pad('onlymove', W.om) + pad('est. EGF', W.est) + pad('real EGF', W.real)
        + (anyAga ? pad('est. AGA', W.est) + pad('real AGA', W.real) : ''));
    console.log('─'.repeat(W.name + W.s9 + W.s13 + W.med + W.n + 2 + weekDays.length * DAYW + W.wk + 2
        + W.om + W.est + W.real + (anyAga ? W.est + W.real : 0)));
    for (const p of out) {
        console.log(
            pad(p.name + (p.rebuilt ? ' †' : ''), W.name)
            + padl(p.novel9.toLocaleString('en-US'), W.s9)
            + padl(p.novel13.toLocaleString('en-US'), W.s13)
            + padl(medianStr(p.medianMs), W.med)
            + padl(p.usableNovel, W.n) + '  '
            + dayCells(p) + padl(p.weekTotal.toLocaleString('en-US'), W.wk) + '  '
            + pad(rankStr(mod, p.onlymoveRank), W.om)
            + pad(rankStr(mod, p.estEgfRank), W.est)
            + pad(realStr(p.realEgfRank, p.egfStale), W.real)
            + (anyAga ? pad(rankStr(mod, p.estAgaRank), W.est)
                + pad(realStr(p.realAgaRank, p.agaStale), W.real) : ''));
    }
    console.log(
        `\n${out.length} ${showAll ? 'leaderboard' : 'externally-linked'} player(s). `
        + `"novel" = distinct-attempt (is_repeat=0) count over all history; `
        + `median = median solve time over novel timed attempts; buf = still-in-corpus attempts in the rank buffer.\n`
        + `${weekDays.map((d) => d.label).join(' / ')} = attempts (all reports, incl. repeats) that LOCAL day (${weekStart}→today); wk = week total so far.\n`
        + `onlymove  = untimed front-page window rank.\n`
        + `est. EGF/AGA = the time-adjusted estimate mapped through that server's coupling (the frame a real rating speaks).\n`
        + `real EGF/AGA = the verified machine heartbeat — the EGD GoR ((GoR−2000)/100) / the AGA rating.\n`
        + `Model ranks are shown for EVERYONE (the production buf ≥ ${RANKED_MIN} gate is dropped here); `
        + `players below that threshold are low-confidence — see se in --json.`
        + (out.some((p) => p.rebuilt) ? `\n† = no leaderboard row (pruned as idle); buffer rebuilt from tsumego_reports, as the Worker would on their next report.` : '')
        + (hasVerifiedCols ? `\n* = production would withhold this verified rating as out of date (EGF activity criterion); kept here.` : '')
        + (showAll ? '' : `\nRun with --all to list every leaderboard row (one per active user, anonymous included).`));

    // --- correlation summary --------------------------------------------------------
    const fc = (v) => (Number.isFinite(v) ? v.toFixed(3) : '  —  ');
    const fe = (v) => (Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}` : '—');
    const corrLine = (label, c, withErr) => {
        let s = `    ${pad(label, 12)} n=${padl(c.n, 4)}   Pearson r=${fc(c.pearson)}   Spearman ρ=${fc(c.spearman)}`;
        if (withErr) s += `   MAE=${fe(c.mae)}   bias=${fe(c.bias)} stones`;
        return s;
    };
    const block = (title, corr, withErr) => {
        if (!corr.all.n) return;
        console.log(title);
        console.log(corrLine('all', corr.all, withErr));
        console.log(corrLine(`buf ≥${RANKED_MIN}`, corr.ranked, withErr));
        // The cohort production actually displays; identical without the migration's
        // last_game_at (nothing is knowable as stale), so it's only worth a line when
        // it drops someone.
        if (corr.rankedActive.n && corr.rankedActive.n < corr.ranked.n) {
            console.log(corrLine('+ active', corr.rankedActive, withErr));
        }
    };
    console.log('\nCorrelations — onlymove prediction vs the verified external rating');
    console.log('(Pearson r is affine-invariant: scalar-space r == predicted-GoR vs actual-GoR r.)\n');
    block('  est. EGF (time-adjusted GoR prediction) vs real EGF:', correlations.estVsRealEgf, true);
    block('  onlymove headline (untimed, not EGF-mapped) vs real EGF — association only:',
        correlations.onlymoveVsRealEgf, false);
    block('  est. AGA (time-adjusted AGA prediction) vs real AGA:', correlations.estVsRealAga, true);
    block('  onlymove headline (untimed, not AGA-mapped) vs real AGA — association only:',
        correlations.onlymoveVsRealAga, false);
}

await main();
