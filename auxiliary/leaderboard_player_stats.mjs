#!/usr/bin/env node
// auxiliary/leaderboard_player_stats.mjs
//
// Read-only report: for every player on the EGD leaderboard, print
//   - novel (is_repeat = 0) 9x9 attempt count
//   - novel 13x13 attempt count
//   - median solve time over their novel attempts
//   - their onlymove rank   (the untimed front-page window MLE, decimal)
//   - their EGF rank         (the time-adjusted leaderboard rank, decimal)
// Then a summary CORRELATING onlymove's predicted EGF strength against each player's
// real EGD GoR — Pearson + Spearman (+ MAE / bias), over all players and the
// production ≥60-novel cohort. Pearson r is affine-invariant, so these scalar-space
// figures equal the predicted-GoR vs actual-GoR correlations.
//
// The two ranks are computed BIT-IDENTICALLY to production: this script
// esbuild-bundles the shared window estimator (pwa/shared/rank-window.js) +
// the rating->rank / EGF-coupling / time-adjust data that leaderboard.js inlines,
// and runs it over each row's STORED buffer (the last <=500 novel attempts, the
// exact input the Worker's computeRank uses). The onlymove rank is the untimed
// internal-rank index (est.rankIndex - RANK_INDEX_OFFSET); the EGF rank re-applies
// the median-solve-time dock and maps through the EGF server coupling -- i.e. it
// is leaderboard.js's predEgfRank. Both are in the 1k=0 / 1d=1 scalar convention.
//
// Data source: attempts/database.sqlite (the local D1 mirror) + the current local
// pwa/tsumegos/{problem_params.bin, rating_to_rank.json, rank_map.json,
// population_constants.json}. Run `./.venv/bin/python3 -m pipeline.fetch_database`
// first so the leaderboard rows + tsumego_reports are fresh.
//
// Usage:
//   node auxiliary/leaderboard_player_stats.mjs            # table, strongest first
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
const onlyUsers = new Set();
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--json') asJson = true;
    else if (a === '--user') onlyUsers.add(args[++i]);
    else { console.error(`unknown arg: ${a}`); process.exit(1); }
}

// --- bundle the production rank code (inlines the local curve/coupling JSONs) ----
// Mirrors leaderboard.js's computeRank, but ALSO returns the untimed onlymove rank
// (est.rankIndex - RANK_INDEX_OFFSET) alongside the time-adjusted EGF rank.
async function loadRankModule() {
    const require = createRequire(path.join(PWA_DIR, 'package.json'));
    const esbuild = require('esbuild');
    const res = await esbuild.build({
        stdin: {
            contents: `
                import { interpExtrap, RANK_INDEX_OFFSET, rankScalarToLabel } from './shared/rank-math.js';
                import { estimateWindowRankFromFlat, weightedMedianOverWindow, flattenAttempt } from './shared/rank-window.js';
                export { gorToRankScalar } from './src/leaderboard.js';   // GoR -> real-EGF scalar (same scale as predEgfRank)
                import RATING_RANK_MAP from './tsumegos/rating_to_rank.json';
                import RANK_MAP from './tsumegos/rank_map.json';
                import POPULATION_CONSTANTS from './tsumegos/population_constants.json';

                const CURVE = RATING_RANK_MAP['9'] || Object.values(RATING_RANK_MAP)[0] || null;
                const EGF = (RANK_MAP && RANK_MAP.EGF) || null;
                const TIME_ADJUST = POPULATION_CONSTANTS.time_adjust || null;
                export const RANK_START_MIN = 60;
                const TIME_MIN_TIMED = 20;

                function internalIndexToEgf(rankIndex) {
                    if (!EGF || !Array.isArray(EGF.our) || !Array.isArray(EGF.server) || EGF.our.length < 2) return null;
                    if (!Number.isFinite(rankIndex)) return null;
                    return interpExtrap(rankIndex - RANK_INDEX_OFFSET, EGF.our, EGF.server);
                }
                function rankIndexAfterRatingDelta(rankIndex, deltaRating) {
                    if (!CURVE || !Number.isFinite(rankIndex) || !Number.isFinite(deltaRating)) return rankIndex;
                    const rating = interpExtrap(rankIndex, CURVE.rank_indices, CURVE.ratings);
                    return interpExtrap(rating + deltaRating, CURVE.ratings, CURVE.rank_indices);
                }

                // buffer: [{p,s,t}] newest first (the stored leaderboard buffer).
                export function computeRankBoth(buffer, params) {
                    const flat = [], times = [];
                    for (let i = buffer.length - 1; i >= 0; i--) {   // oldest -> newest
                        const e = buffer[i];
                        const pp = e && typeof e.p === 'string' ? params.get(e.p) : null;
                        if (!pp) continue;                            // left the corpus
                        const f = flattenAttempt(pp.rating, pp.slope, pp.floor, e.s, CURVE);
                        if (!f) continue;
                        flat.push(f);
                        times.push(Number.isFinite(e.t) ? e.t : NaN);
                    }
                    // NOTE: production gates a rank on flat.length >= RANK_START_MIN (60).
                    // This report deliberately DROPS that gate — the window MLE runs on
                    // whatever novel attempts exist (a weak prior, WIN_PRIOR_SD=4, keeps it
                    // well-posed), so even not-yet-ranked players get a (low-confidence) rank.
                    const est = estimateWindowRankFromFlat(flat, NaN);
                    if (!est) return { onlymoveRank: null, egfRank: null, se: null, usable: flat.length };
                    const onlymoveRank = est.rankIndex - RANK_INDEX_OFFSET;   // untimed headline
                    let rankIndex = est.rankIndex;
                    const medMs = weightedMedianOverWindow(times, TIME_MIN_TIMED);
                    if (medMs !== null && TIME_ADJUST
                        && Number.isFinite(TIME_ADJUST.alpha_per_log10_min)
                        && Number.isFinite(TIME_ADJUST.center_log10_min)) {
                        const deltaRating = TIME_ADJUST.alpha_per_log10_min
                            * (Math.log10(medMs / 60000) - TIME_ADJUST.center_log10_min);
                        rankIndex = rankIndexAfterRatingDelta(rankIndex, deltaRating);
                    }
                    return { onlymoveRank, egfRank: internalIndexToEgf(rankIndex), se: est.se, usable: flat.length };
                }
                export { rankScalarToLabel };
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

function parseBuffer(text) {
    if (!text) return [];
    try { const a = JSON.parse(text); return Array.isArray(a) ? a : []; } catch { return []; }
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

    let rows = sqliteJson('SELECT user_id, egd_name, gor, buffer FROM leaderboard');
    if (onlyUsers.size) rows = rows.filter((r) => onlyUsers.has(r.user_id));
    if (!rows.length) { console.log('no leaderboard rows'); return; }

    // Novel attempt counts by size (whole history, is_repeat = 0).
    const counts = sqliteJson(
        `SELECT user_id, size, COUNT(*) AS n
           FROM tsumego_reports
          WHERE is_repeat = 0 AND user_id IN (SELECT user_id FROM leaderboard)
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
               AND user_id IN (SELECT user_id FROM leaderboard)
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
            AND user_id IN (SELECT user_id FROM leaderboard)
          GROUP BY user_id, d`
    );
    const perDayBy = new Map();   // user_id -> { 'YYYY-MM-DD': n }
    for (const dr of perDayRows) {
        if (!perDayBy.has(dr.user_id)) perDayBy.set(dr.user_id, {});
        perDayBy.get(dr.user_id)[dr.d] = dr.n;
    }

    const out = rows.map((r) => {
        const { onlymoveRank, egfRank, se, usable } = mod.computeRankBoth(parseBuffer(r.buffer), params);
        const c = countBy.get(r.user_id) || { 9: 0, 13: 0 };
        const md = medBy.get(r.user_id) || {};
        const perDayObj = perDayBy.get(r.user_id) || {};
        const perDay = {};
        let weekTotal = 0;
        for (const d of weekDays) { const n = perDayObj[d.date] || 0; perDay[d.date] = n; weekTotal += n; }
        const realEgfRank = mod.gorToRankScalar(r.gor);   // from the player's live EGD GoR
        return {
            userId: r.user_id, name: r.egd_name || r.user_id, gor: r.gor,
            novel9: c[9] || 0, novel13: c[13] || 0,
            medianMs: Number.isFinite(md.median_ms) ? md.median_ms : null,
            perDay, weekTotal,
            onlymoveRank, estEgfRank: egfRank, realEgfRank, se, usableNovel: usable,
        };
    });
    // Strongest first by onlymove rank; unranked (null) sink to the bottom.
    out.sort((a, b) => (b.onlymoveRank ?? -Infinity) - (a.onlymoveRank ?? -Infinity));

    // --- correlations: onlymove's predicted strength vs the player's real EGD GoR ---
    // Pearson r is invariant under the affine GoR<->scalar map (gor = 2000 + 100·scalar),
    // so these scalar-space numbers ARE the predicted-GoR vs actual-GoR correlations.
    const RANKED_MIN = mod.RANK_START_MIN;   // production's ≥60-novel-attempt rank gate
    const fin = (v) => Number.isFinite(v);
    const pairsOf = (key, gate) => out
        .filter((p) => fin(p[key]) && fin(p.realEgfRank) && (!gate || p.usableNovel >= RANKED_MIN))
        .map((p) => [p[key], p.realEgfRank]);
    const correlations = {
        estVsRealEgf: { all: corrStats(pairsOf('estEgfRank', false)), ranked: corrStats(pairsOf('estEgfRank', true)) },
        onlymoveVsRealEgf: { all: corrStats(pairsOf('onlymoveRank', false)), ranked: corrStats(pairsOf('onlymoveRank', true)) },
    };

    if (asJson) { console.log(JSON.stringify({ correlations, players: out }, null, 2)); return; }

    const W = { name: 20, s9: 8, s13: 8, med: 9, wk: 7, om: 15, est: 15, real: 15 };
    const dayHeader = weekDays.map((d) => padl(d.label, DAYW)).join('');
    const dayCells = (p) => weekDays.map((d) => padl((p.perDay[d.date] || 0).toLocaleString('en-US'), DAYW)).join('');
    console.log(
        pad('Player', W.name) + padl('9×9', W.s9) + padl('13×13', W.s13)
        + padl('median', W.med) + '  ' + dayHeader + padl('wk', W.wk) + '  '
        + pad('onlymove', W.om) + pad('est. EGF', W.est) + pad('real EGF', W.real));
    console.log('─'.repeat(W.name + W.s9 + W.s13 + W.med + 2 + weekDays.length * DAYW + W.wk + 2 + W.om + W.est + W.real));
    for (const p of out) {
        console.log(
            pad(p.name, W.name)
            + padl(p.novel9.toLocaleString('en-US'), W.s9)
            + padl(p.novel13.toLocaleString('en-US'), W.s13)
            + padl(medianStr(p.medianMs), W.med) + '  '
            + dayCells(p) + padl(p.weekTotal.toLocaleString('en-US'), W.wk) + '  '
            + pad(rankStr(mod, p.onlymoveRank), W.om)
            + pad(rankStr(mod, p.estEgfRank), W.est)
            + pad(rankStr(mod, p.realEgfRank), W.real));
    }
    console.log(
        `\n${out.length} leaderboard player(s). "novel" = distinct-attempt (is_repeat=0) count over all history; `
        + `median = median solve time over novel timed attempts.\n`
        + `${weekDays.map((d) => d.label).join(' / ')} = attempts (all reports, incl. repeats) that LOCAL day (${weekStart}→today); wk = week total so far.\n`
        + `onlymove  = untimed front-page window rank.\n`
        + `est. EGF  = time-adjusted leaderboard rank (the model's estimate of EGF strength).\n`
        + `real EGF  = the player's actual EGD strength derived from their live GoR ((GoR−2000)/100).\n`
        + `Model ranks are shown for EVERYONE (the production ≥60-novel-attempt gate is dropped here); `
        + `players below that threshold are low-confidence — see se in --json.`);

    // --- correlation summary --------------------------------------------------------
    const fc = (v) => (Number.isFinite(v) ? v.toFixed(3) : '  —  ');
    const fe = (v) => (Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${v.toFixed(2)}` : '—');
    const corrLine = (label, c, withErr) => {
        let s = `    ${pad(label, 10)} n=${padl(c.n, 4)}   Pearson r=${fc(c.pearson)}   Spearman ρ=${fc(c.spearman)}`;
        if (withErr) s += `   MAE=${fe(c.mae)}   bias=${fe(c.bias)} stones`;
        return s;
    };
    console.log('\nCorrelations — onlymove prediction vs actual EGD strength (real GoR)');
    console.log('(Pearson r is affine-invariant: scalar-space r == predicted-GoR vs actual-GoR r.)\n');
    console.log('  est. EGF (time-adjusted GoR prediction) vs real EGF:');
    console.log(corrLine('all', correlations.estVsRealEgf.all, true));
    console.log(corrLine(`≥${RANKED_MIN} novel`, correlations.estVsRealEgf.ranked, true));
    console.log('  onlymove headline (untimed, not EGF-mapped) vs real EGF — association only:');
    console.log(corrLine('all', correlations.onlymoveVsRealEgf.all, false));
    console.log(corrLine(`≥${RANKED_MIN} novel`, correlations.onlymoveVsRealEgf.ranked, false));
}

await main();
