#!/usr/bin/env node
// auxiliary/update_leaderboard_ranks.mjs
//
// Retroactively recompute every leaderboard row's rank fields — buffer,
// pred_egf_rank, pred_egf_se, week_id, this_week_max, prev_week_max — and apply
// them to the leaderboard D1 (--local = miniflare, --remote = production,
// neither = dry run). Purpose: after the ratings / rating→rank curve /
// population constants change (a re-rate + `map_rating_to_rank` +
// `package_for_pwa`), the stored current rank AND the frozen week peaks are on
// the OLD scale; the weekly cron reconcile only refreshes the current rank and
// deliberately preserves the peaks, so a scale change needs this script.
//
// The math is BIT-IDENTICAL to production: the script esbuild-bundles
// pwa/src/leaderboard.js (which inlines the CURRENT local rating_to_rank.json /
// rank_map.json / population_constants.json, exactly like the Worker build) and
// reads the CURRENT local pwa/tsumegos/problem_params.bin. So it recomputes with
// whatever curve/params sit in pwa/tsumegos/ right now.
//
// Data flow (read-cost discipline — D1 bills rows read):
//   reads  <- attempts/database.sqlite  (the local mirror; run
//             `./.venv/bin/python3 -m pipeline.fetch_database` FIRST so the
//             leaderboard rows + tsumego_reports are fresh)
//   writes -> a generated SQL file, applied to the LOCAL miniflare D1
//             (--local) or to production (--remote); neither = dry run
//
// Per user (novel attempts only, EVENT order — created_at with id tiebreak):
//   buffer          = last BUFFER_CAP novel attempts, newest first. NOTE this
//                     deliberately differs from buildRowStateFromReports'
//                     ORDER BY id: /report-backfill inserts historical rows at
//                     the id frontier (and the 2026-07-07 outbox incident minted
//                     exact-duplicate rows), so id order can put months-old
//                     attempts "on top" — event order matches what the live
//                     /report-maintained buffer actually holds. Exact duplicate
//                     rows (same problem + created_at) are collapsed to one.
//   current rank/se = computeRank over that buffer (window MLE + time adjust)
//   this_week_max   = max computeRank over the trailing window ending at each
//                     novel attempt inside the row's week (stored week_id, or the
//                     week of the last attempt when week_id is NULL) — the
//                     retroactive equivalent of the per-report rollWeek max
//   prev_week_max   = same over week_id - 1 (NULL if no rankable attempt then)
// Peaks REPLACE the stored values (no max() with the old-scale number — that is
// the point). prev_week_gor_rank (frozen GoR) is rating-independent and is left
// untouched, except a row whose recomputed rank is NULL gets all week fields
// nulled (matches rollWeek's fresh start, so a stale old-scale peak can never be
// promoted later). last_active is untouched.
//
// Order of operations for a re-rate: package_for_pwa -> `npm run deploy` (so the
// Worker computes on the new scale too) -> fetch_database -> this script. If you
// run it before deploying, the very next /report recomputes the user's current
// rank on the OLD scale and overwrites it (peaks survive: rollWeek only maxes).
//
// Usage (dry run by default; pick the apply target explicitly):
//   node auxiliary/update_leaderboard_ranks.mjs                   # dry run: print + write SQL file
//   node auxiliary/update_leaderboard_ranks.mjs --local           # ... and execute against the local miniflare D1
//                                                                 #     (stop `npm run dev` first)
//   node auxiliary/update_leaderboard_ranks.mjs --remote          # ... against the REMOTE production D1
//   node auxiliary/update_leaderboard_ranks.mjs --populate        # ALSO seed anonymous rows for users active in the
//                                                                 #     past week with no leaderboard row (production
//                                                                 #     only adds them lazily on their next /report)
//   node auxiliary/update_leaderboard_ranks.mjs --user <redacted-user-id>   # limit to one user (repeatable; also scopes populate)

import { execFileSync, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PWA_DIR = path.join(REPO, 'pwa');
const DB_PATH = path.join(REPO, 'attempts', 'database.sqlite');
const PARAMS_BIN = path.join(PWA_DIR, 'tsumegos', 'problem_params.bin');
const USER_ID_RE = /^[A-Za-z0-9_-]{1,40}$/;
const DAY_MS = 86400000;
const STALE_WARN_MS = 6 * 3600 * 1000;
// --populate seeds active-but-rowless users; the window matches the /leaderboard
// page's ACTIVE_DAYS (src/handlers/leaderboard-page.js), so a seeded row's
// last_active clears that page's active-window filter.
const ACTIVE_WINDOW_DAYS = 7;

// --- CLI ------------------------------------------------------------------------
const args = process.argv.slice(2);
let applyLocal = false, applyRemote = false, populate = false;
const onlyUsers = new Set();
let sqlOut = path.join(os.tmpdir(), 'leaderboard_rank_update.sql');
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--local') applyLocal = true;
    else if (a === '--remote') applyRemote = true;
    else if (a === '--populate') populate = true;
    else if (a === '--user') onlyUsers.add(args[++i]);
    else if (a === '--sql-out') sqlOut = path.resolve(args[++i]);
    else { console.error(`unknown arg: ${a}`); process.exit(1); }
}
if (applyLocal && applyRemote) { console.error('--local and --remote are mutually exclusive'); process.exit(1); }

// --- bundle the production rank code (leaderboard.js inlines the local JSONs) ----
async function loadLeaderboardModule() {
    const require = createRequire(path.join(PWA_DIR, 'package.json'));
    const esbuild = require('esbuild');
    const res = await esbuild.build({
        stdin: {
            contents: `
                export { computeRank, weekIdAt, serializeBuffer, RANK_START_MIN, BUFFER_CAP } from './src/leaderboard.js';
                export { rankScalarToLabel } from './shared/rank-math.js';
            `,
            resolveDir: PWA_DIR,
            sourcefile: 'aux-leaderboard-entry.mjs',
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

// Same 16-byte record format as leaderboard.js loadProblemParams (which is tied to
// env.ASSETS, hence re-parsed here): 4-byte id + 3 little-endian float32.
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

// --- local sqlite reads -----------------------------------------------------------
function sqliteJson(sql) {
    const out = execFileSync('sqlite3', ['-json', DB_PATH, sql],
        { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 });
    const text = out.trim();
    return text ? JSON.parse(text) : [];
}

// --- SQL literal helpers -----------------------------------------------------------
const sqlNum = (v) => (Number.isFinite(v) ? String(v) : 'NULL');
const sqlStr = (v) => `'${String(v).replace(/'/g, "''")}'`;

// --- week helpers ------------------------------------------------------------------
// Monday-00:00-UTC week window for a weekIdAt() id (weekId*7+4 = that Monday's
// epoch-day index, same arithmetic as leaderboard.js weekStartIso).
const weekStartMs = (wid) => (wid * 7 + 4) * DAY_MS;

// Windowed, deduped, EVENT-ordered novel attempts for one user (the buffer feed —
// see the header note on event order vs id order). `uid` is the sql-quoted user id;
// `wid` bounds the peak scan back to the start of week wid-1, plus the trailing
// `cap` window feeding the oldest in-window attempt. Returns ascending [{p,s,t,ms}].
function fetchWindowedAttempts(uid, wid, cap) {
    const sinceIso = new Date(weekStartMs(wid - 1)).toISOString();
    // Deduped counts/rows: GROUP BY (problem, created_at) collapses the minted
    // exact-duplicate rows; SQLite's bare-column rule with the single MIN(id)
    // aggregate keeps the ORIGINAL row's values. Ordered by created_at (event
    // time), NOT id — see the header note.
    const [{ n: inWindow }] = sqliteJson(
        `SELECT COUNT(*) AS n FROM (
            SELECT 1 FROM tsumego_reports
             WHERE user_id = ${uid} AND is_repeat = 0 AND created_at >= '${sinceIso}'
             GROUP BY problem, created_at)`
    );
    return sqliteJson(
        `SELECT p, s, t, c FROM (
            SELECT MIN(id) AS id, problem AS p, success AS s, time_used AS t, created_at AS c
              FROM tsumego_reports
             WHERE user_id = ${uid} AND is_repeat = 0
             GROUP BY problem, created_at
             ORDER BY c DESC, id DESC LIMIT ${inWindow + cap})
          ORDER BY c ASC, id ASC`
    ).map((r) => ({
        p: r.p, s: r.s ? 1 : 0, t: Number.isFinite(r.t) ? r.t : null, ms: Date.parse(r.c),
    }));
}

// --- per-user recompute --------------------------------------------------------------
// attempts: ascending EVENT order (created_at, id tiebreak), exact dups collapsed,
// [{ p, s, t, ms }]. Buffer at index i = the trailing <= BUFFER_CAP attempts ending
// at i, newest first (the live-buffer order at that moment in event time).
function bufferEndingAt(attempts, i, cap) {
    const lo = Math.max(0, i - cap + 1);
    const buf = [];
    for (let j = i; j >= lo; j--) buf.push({ p: attempts[j].p, s: attempts[j].s, t: attempts[j].t });
    return buf;
}

// Max rank over the attempts whose created_at falls in [msLo, msHi) — the
// retroactive equivalent of updateLeaderboardOnReport's per-report rollWeek max.
function peakOverWindow(lb, attempts, params, msLo, msHi) {
    let peak = null;
    for (let i = 0; i < attempts.length; i++) {
        const ms = attempts[i].ms;
        if (!Number.isFinite(ms) || ms < msLo || ms >= msHi) continue;
        if (i + 1 < lb.RANK_START_MIN) continue;   // cheap skip; computeRank would return null
        const { predEgfRank } = lb.computeRank(bufferEndingAt(attempts, i, lb.BUFFER_CAP), params);
        if (predEgfRank !== null && (peak === null || predEgfRank > peak)) peak = predEgfRank;
    }
    return peak;
}

function recomputeRow(lb, row, attempts, params) {
    const buffer = bufferEndingAt(attempts, attempts.length - 1, lb.BUFFER_CAP);
    const { predEgfRank, se } = lb.computeRank(buffer, params);

    if (predEgfRank === null) {
        return { buffer, predEgfRank: null, se: null, weekId: null, thisWeekMax: null, prevWeekMax: null };
    }
    // The row's display week: the stored week_id (what readLeaderboardRows pairs the
    // peaks + frozen GoR against), falling back to the last attempt's week for a row
    // that was never ranked before.
    const lastMs = attempts.length ? attempts[attempts.length - 1].ms : NaN;
    const weekId = Number.isFinite(row.week_id) ? row.week_id
        : (Number.isFinite(lastMs) ? lb.weekIdAt(lastMs) : null);
    if (weekId === null) {
        return { buffer, predEgfRank, se, weekId: null, thisWeekMax: null, prevWeekMax: null };
    }
    const thisPeak = peakOverWindow(lb, attempts, params, weekStartMs(weekId), weekStartMs(weekId + 1));
    const prevPeak = peakOverWindow(lb, attempts, params, weekStartMs(weekId - 1), weekStartMs(weekId));
    return {
        buffer, predEgfRank, se, weekId,
        // No rankable attempt inside the week (arrival-vs-event-time skew) -> fall
        // back to the current rank, the value rollWeek would have seeded.
        thisWeekMax: thisPeak !== null ? thisPeak : predEgfRank,
        prevWeekMax: prevPeak,
    };
}

// --- output ---------------------------------------------------------------------------
function fmtRank(lb, v) {
    return Number.isFinite(v) ? `${v.toFixed(2)} (${lb.rankScalarToLabel(v)})` : '—';
}
function fmtDelta(lb, oldV, newV) {
    return `${fmtRank(lb, oldV)} -> ${fmtRank(lb, newV)}`;
}

function findMiniflareD1() {
    const dir = path.join(PWA_DIR, '.wrangler', 'state', 'v3', 'd1', 'miniflare-D1DatabaseObject');
    const files = fs.existsSync(dir)
        ? fs.readdirSync(dir).filter((f) => f.endsWith('.sqlite') && !f.startsWith('metadata'))
        : [];
    if (files.length !== 1) {
        console.error(`expected exactly one miniflare D1 file in ${dir}, found ${files.length}`);
        process.exit(1);
    }
    return path.join(dir, files[0]);
}

// --- main -------------------------------------------------------------------------------
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
            + 'Run pipeline.fetch_database first for fresh ranks.');
    }

    const lb = await loadLeaderboardModule();
    const params = loadProblemParamsLocal();
    console.log(`problem_params: ${params.size} problems\n`);

    let rows = sqliteJson(
        'SELECT user_id, week_id, pred_egf_rank, pred_egf_se, this_week_max, prev_week_max FROM leaderboard'
    );
    if (onlyUsers.size) rows = rows.filter((r) => onlyUsers.has(r.user_id));
    if (!rows.length && !populate) { console.log('no leaderboard rows to update'); return; }

    const statements = [];
    for (const row of rows) {
        if (!USER_ID_RE.test(row.user_id)) {
            console.warn(`skipping malformed user_id: ${JSON.stringify(row.user_id)}`);
            continue;
        }
        // How far back the peaks can reach: everything since the start of week_id - 1,
        // plus the trailing BUFFER_CAP window feeding the oldest in-window attempt.
        const wid = Number.isFinite(row.week_id) ? row.week_id : lb.weekIdAt(Date.now());
        const uid = sqlStr(row.user_id);
        const attempts = fetchWindowedAttempts(uid, wid, lb.BUFFER_CAP);

        const out = recomputeRow(lb, row, attempts, params);
        console.log(`${row.user_id}  buffer ${attempts.length >= lb.BUFFER_CAP ? lb.BUFFER_CAP : attempts.length} attempts`);
        console.log(`  current   ${fmtDelta(lb, row.pred_egf_rank, out.predEgfRank)}   se ${Number.isFinite(row.pred_egf_se) ? row.pred_egf_se.toFixed(3) : '—'} -> ${Number.isFinite(out.se) ? out.se.toFixed(3) : '—'}`);
        console.log(`  this-week ${fmtDelta(lb, row.this_week_max, out.thisWeekMax)}`);
        console.log(`  prev-week ${fmtDelta(lb, row.prev_week_max, out.prevWeekMax)}   week_id ${row.week_id ?? '—'} -> ${out.weekId ?? '—'}`);

        // A NULL recomputed rank also nulls the week fields (incl. the frozen GoR),
        // matching rollWeek's fresh start — see the header note.
        const weekSql = out.predEgfRank === null
            ? 'week_id = NULL, this_week_max = NULL, prev_week_max = NULL, prev_week_gor_rank = NULL'
            : `week_id = ${sqlNum(out.weekId)}, this_week_max = ${sqlNum(out.thisWeekMax)}, prev_week_max = ${sqlNum(out.prevWeekMax)}`;
        statements.push(
            `UPDATE leaderboard SET buffer = ${sqlStr(lb.serializeBuffer(out.buffer))}, `
            + `pred_egf_rank = ${sqlNum(out.predEgfRank)}, pred_egf_se = ${sqlNum(out.se)}, ${weekSql} `
            + `WHERE user_id = ${uid};`
        );
    }

    // --- populate: seed anonymous rows for active-but-rowless users -----------------
    // Production adds users LAZILY (createLeaderboardRow on their first /report after the
    // all-users rollout), so a freshly-migrated board omits everyone who hasn't reported
    // since. This backfills them: any user with an attempt in the past ACTIVE_WINDOW_DAYS
    // and no leaderboard row gets an anonymous row seeded with the SAME retroactive
    // recompute the update pass uses.
    if (populate) {
        const cutoffIso = new Date(Date.now() - ACTIVE_WINDOW_DAYS * DAY_MS).toISOString();
        // Local-mirror read (not billed); tag each candidate with its most-recent
        // activity so last_active clears the /leaderboard active-window filter.
        let active = sqliteJson(
            `SELECT user_id, MAX(created_at) AS last_created
               FROM tsumego_reports
              WHERE created_at >= '${cutoffIso}'
                AND user_id NOT IN (SELECT user_id FROM leaderboard)
              GROUP BY user_id`
        );
        if (onlyUsers.size) active = active.filter((r) => onlyUsers.has(r.user_id));
        console.log(`\npopulate: ${active.length} user(s) active in the past ${ACTIVE_WINDOW_DAYS}d without a leaderboard row`);

        let ranked = 0, unranked = 0;
        for (const a of active) {
            if (!USER_ID_RE.test(a.user_id)) {
                console.warn(`skipping malformed user_id: ${JSON.stringify(a.user_id)}`);
                continue;
            }
            const uid = sqlStr(a.user_id);
            // No stored week for a new row: recomputeRow derives the week from the
            // user's last novel attempt and scans this/prev-week peaks.
            const attempts = fetchWindowedAttempts(uid, lb.weekIdAt(Date.now()), lb.BUFFER_CAP);
            const out = recomputeRow(lb, { user_id: a.user_id, week_id: null }, attempts, params);
            if (out.predEgfRank === null) unranked++; else ranked++;
            console.log(`  + ${a.user_id}  ${attempts.length} novel attempts  ${out.predEgfRank === null ? 'unranked' : fmtRank(lb, out.predEgfRank)}`);
            // Same seed shape as createLeaderboardRow; INSERT OR IGNORE is a no-op if
            // the row was created concurrently (a live report during a --remote
            // apply). A NULL rank leaves the week fields NULL — matches the unranked seed.
            statements.push(
                `INSERT OR IGNORE INTO leaderboard `
                + `(user_id, buffer, pred_egf_rank, pred_egf_se, week_id, this_week_max, prev_week_max, last_active) `
                + `VALUES (${uid}, ${sqlStr(lb.serializeBuffer(out.buffer))}, `
                + `${sqlNum(out.predEgfRank)}, ${sqlNum(out.se)}, ${sqlNum(out.weekId)}, `
                + `${sqlNum(out.thisWeekMax)}, ${sqlNum(out.prevWeekMax)}, ${sqlStr(a.last_created)});`
            );
        }
        console.log(`populate: ${ranked} ranked + ${unranked} unranked seeded (unranked rows stay hidden on the board until ${lb.RANK_START_MIN} novel attempts)`);
    }

    if (!statements.length) { console.log('\nnothing to write'); return; }
    fs.writeFileSync(sqlOut, statements.join('\n') + '\n');
    console.log(`\n${statements.length} statement(s) written to ${sqlOut}`);

    if (applyRemote) {
        console.log('applying to REMOTE production D1...');
        const res = spawnSync('npx', ['wrangler', 'd1', 'execute', 'DB', '--remote', '--yes', `--file=${sqlOut}`],
            { cwd: PWA_DIR, stdio: 'inherit' });
        process.exit(res.status ?? 1);
    } else if (applyLocal) {
        const dbPath = findMiniflareD1();
        console.log(`applying to local miniflare D1: ${dbPath} (stop npm run dev first)`);
        execFileSync('sqlite3', [dbPath, `.read ${sqlOut}`], { stdio: 'inherit' });
    } else {
        console.log('dry run — re-run with --local (miniflare D1) or --remote (production D1) to write.');
    }
}

await main();
