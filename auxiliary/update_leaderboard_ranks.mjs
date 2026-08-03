#!/usr/bin/env node
// auxiliary/update_leaderboard_ranks.mjs
//
// Retroactively recompute every leaderboard row's rank fields — buffer, pred_rank,
// pred_rank_se, time_delta_rating — and apply them to the leaderboard D1 (--local =
// miniflare, --remote = production, neither = dry run). Purpose: after the
// ratings / rating→rank curve / population constants change (a re-rate +
// `map_rating_to_rank` + `package_for_pwa`), the stored rank is on the OLD
// scale; this script refreshes every row at once instead of waiting for each
// user's next /report (or the weekly cron reconcile) to do it one by one.
//
// The math is BIT-IDENTICAL to production: the script esbuild-bundles
// pwa/src/leaderboard.js (which inlines the CURRENT local rating_to_rank.json /
// rank_map.json / population_constants.json, exactly like the Worker build) and
// reads the CURRENT local pwa/tsumegos/problem_params.bin. So it recomputes with
// whatever curve/params sit in pwa/tsumegos/ right now.
//
// Data flow (read-cost discipline — D1 bills rows read):
//   reads  <- db/d1.sqlite  (the local mirror; run
//             `./.venv/bin/python3 -m pipeline.fetch_database` FIRST so the
//             leaderboard rows + tsumego_reports are fresh)
//   writes -> a generated SQL file, applied to the LOCAL miniflare D1
//             (--local) or to production (--remote); neither = dry run
//
// Per user (novel attempts only, INSERT order — the AUTOINCREMENT id):
//   buffer          = last BUFFER_CAP novel attempts, newest first, MATCHING
//                     buildRowStateFromReports' ORDER BY id. Id order is the
//                     single convention every writer and rebuild shares: the
//                     live /report push, /report-backfill's batched push
//                     (updateLeaderboardOnBackfill), createLeaderboardRow and
//                     the weekly cron reconcile. This script used to order by
//                     created_at instead, on the grounds that /report-backfill
//                     inserts historical rows at the id frontier so id order
//                     could seat a months-old attempt "on top" — true, but the
//                     fix was to make backfill maintain the buffer like every
//                     other writer, not to give one rebuilder its own ordering.
//                     Event order would also need a (user_id, created_at) index
//                     that was deliberately dropped, turning a bounded
//                     BUFFER_CAP read into a full per-user history scan.
//                     Exact duplicate rows (same problem + created_at) are still
//                     collapsed to one (the 2026-07-07 outbox incident).
//   current rank/se = computeRank over that buffer — the UNTIMED internal rank
//                     (0 = the 1k/1d boundary) + its stones SE, plus the personal
//                     time dock in rating units, each stored in its own column.
// last_active is untouched.
//
// (Before 2026-07-27 the columns held an EGF-frame rank, so this script's dry-run
// output labelled a GoR scalar with the internal-rank labeller and printed nonsense
// ranks. The stored value is now genuinely an internal scalar, so fmtRank is right.)
//
// Order of operations for a re-rate: package_for_pwa -> `npm run deploy` (so the
// Worker computes on the new scale too) -> fetch_database -> this script. If you
// run it before deploying, the very next /report recomputes the user's current
// rank on the OLD scale and overwrites it.
//
// RUN IT UNDER THE MAINTENANCE FREEZE. The `buffer` column is maintained
// INCREMENTALLY in production: updateLeaderboardOnReport reads the stored
// buffer, pushes the one new attempt and writes it back, and NOTHING rebuilds
// it from tsumego_reports afterwards. So every attempt reported between this
// script's snapshot and its write is silently truncated out of that user's
// buffer when the UPDATE lands — permanently. Later reports append on top of
// the shortened list rather than restoring it, and the only rebuild path is
// createLeaderboardRow, which runs only for a user who has NO row at all. There
// is no error and no recovery; the damage is a slightly wrong rank for anyone
// who happened to be playing.
//
// With MAINTENANCE_MODE = "1" every POST 503s, so nothing can land in that gap
// and the race is not merely unlikely but IMPOSSIBLE. That is why the refresh
// belongs inside the surgery window (docs/db_surgery_runbook.md step 8), and it
// applies to a standalone post-re-rate refresh too:
//
//   1. set MAINTENANCE_MODE = "1" in pwa/wrangler.toml, `cd pwa && npm run deploy`
//   2. ./.venv/bin/python3 -m pipeline.fetch_database
//   3. node auxiliary/update_leaderboard_ranks.mjs --remote
//   4. set MAINTENANCE_MODE = "0", `npm run deploy`
//
// --remote ENFORCES this rather than trusting the operator (see preflightRemote):
// it refuses unless production is actually frozen AND the local mirror's newest
// tsumego_reports id equals production's, then re-checks both after the apply so
// a freeze that lapsed mid-run is reported instead of silently corrupting rows.
// `--allow-unfrozen` overrides deliberately; chaining the fetch into the apply
// only NARROWS the window and is a fallback, never the guarantee.
//
// Usage (dry run by default; pick the apply target explicitly):
//   node auxiliary/update_leaderboard_ranks.mjs                   # dry run: print + write SQL file
//   node auxiliary/update_leaderboard_ranks.mjs --local           # ... and execute against the local miniflare D1
//                                                                 #     (stop `npm run dev` first)
//   node auxiliary/update_leaderboard_ranks.mjs --remote          # ... against the REMOTE production D1
//   node auxiliary/update_leaderboard_ranks.mjs --populate        # ALSO seed anonymous rows for users active in the
//                                                                 #     past week with no leaderboard row (production
//                                                                 #     only adds them lazily on their next /report)
//   node auxiliary/update_leaderboard_ranks.mjs --user <id>       # limit to one user (repeatable; also scopes populate)
//   node auxiliary/update_leaderboard_ranks.mjs --remote --allow-unfrozen
//                                                                 #     skip the freeze/snapshot preflight (see above)
//   node auxiliary/update_leaderboard_ranks.mjs --remote --origin https://…
//                                                                 #     probe a different origin (default: the custom
//                                                                 #     domain in pwa/wrangler.toml)

import { execFileSync, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PWA_DIR = path.join(REPO, 'pwa');
const DB_PATH = path.join(REPO, 'db', 'd1.sqlite');
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
let applyLocal = false, applyRemote = false, populate = false, allowUnfrozen = false;
let originOverride = null;
const onlyUsers = new Set();
let sqlOut = path.join(os.tmpdir(), 'leaderboard_rank_update.sql');
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--local') applyLocal = true;
    else if (a === '--remote') applyRemote = true;
    else if (a === '--populate') populate = true;
    else if (a === '--allow-unfrozen') allowUnfrozen = true;
    else if (a === '--origin') originOverride = args[++i];
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
                export { computeRank, serializeBuffer, RANK_START_MIN, BUFFER_CAP } from './src/leaderboard.js';
                export { rankToLabel } from './shared/rank-math.js';
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

// --- production write-freeze preflight (--remote only) ------------------------------
// The buffer rebuild is a read-modify-write against a snapshot, so it is only safe
// while production cannot write. Both checks below are cheap: the freeze probe costs
// one Worker invocation and zero D1, and MAX(id) is served by the AUTOINCREMENT PK.

// wrangler normally keeps its banners on stderr, but defensively locate where the
// --json array starts (a line that is exactly "[") rather than trusting stdout to be
// pure JSON — and by LINE, not by the first "[" character, since a "▲ [WARNING]"
// banner contains one. A clean stdout just matches at line 0.
const stripAnsi = (s) => s.replace(/\u001B\[[0-9;]*m/g, '');

function remoteQueryRows(sql) {
    const res = spawnSync('npx',
        ['wrangler', 'd1', 'execute', 'DB', '--remote', '--json', '--command', sql],
        { cwd: PWA_DIR, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
    if (res.status !== 0) {
        throw new Error(`wrangler d1 execute failed (${res.status}): ${stripAnsi(res.stderr || res.stdout || '')}`);
    }
    const lines = stripAnsi(res.stdout || '').split('\n');
    const start = lines.findIndex((l) => l.trim() === '[');
    const text = start >= 0 ? lines.slice(start).join('\n') : (res.stdout || '');
    return JSON.parse(text)[0]?.results ?? [];
}

// Default to the custom domain declared in wrangler.toml rather than hardcoding it.
function productionOrigin() {
    if (originOverride) return originOverride.replace(/\/+$/, '');
    const toml = fs.readFileSync(path.join(PWA_DIR, 'wrangler.toml'), 'utf8');
    const m = toml.match(/pattern\s*=\s*"([^"]+)"\s*,\s*custom_domain\s*=\s*true/);
    if (!m) throw new Error('no custom_domain route found in pwa/wrangler.toml — pass --origin');
    return `https://${m[1]}`;
}

// POST /sync is the probe: it is in WRITE_ROUTES (so the freeze covers it) but is a
// READ, and an empty body fails isValidUserId before any D1 access — so an unfrozen
// probe costs nothing and writes nothing. Frozen returns 503 {error:'maintenance'},
// which distinguishes the freeze from a generic edge 503.
async function probeProductionFrozen(origin) {
    let res;
    try {
        res = await fetch(`${origin}/sync`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}',
        });
    } catch (err) {
        return { frozen: false, detail: `request failed: ${err.message}` };
    }
    if (res.status !== 503) return { frozen: false, detail: `POST /sync -> ${res.status} (expected 503)` };
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON 503 = not our freeze */ }
    if (body?.error !== 'maintenance') {
        return { frozen: false, detail: 'got a 503 that is not the maintenance freeze (edge error?)' };
    }
    return { frozen: true, detail: 'POST /sync -> 503 {error:"maintenance"}' };
}

const localMaxReportId = () => sqliteJson('SELECT MAX(id) AS m FROM tsumego_reports')[0]?.m ?? null;
const remoteMaxReportId = () => remoteQueryRows('SELECT MAX(id) AS m FROM tsumego_reports')[0]?.m ?? null;

// Fails CLOSED: any error reaching production is a refusal, because "we could not
// confirm the freeze" and "the freeze is on" must not be the same outcome here.
async function preflightRemote() {
    const origin = productionOrigin();
    console.log(`preflight: ${origin}`);

    const { frozen, detail } = await probeProductionFrozen(origin);
    console.log(`  freeze    ${frozen ? 'ON' : 'OFF'} — ${detail}`);
    if (!frozen) {
        console.error(
            '\nREFUSING: production is not frozen.\n'
            + '  This script rebuilds each row\'s buffer from a snapshot and writes it back;\n'
            + '  an attempt landing in between is truncated out of that buffer permanently.\n'
            + '  Freeze first: set MAINTENANCE_MODE = "1" in pwa/wrangler.toml, then\n'
            + '  `cd pwa && npm run deploy`. Unfreeze the same way when done.\n'
            + '  (--allow-unfrozen overrides this deliberately.)');
        process.exit(1);
    }

    const local = localMaxReportId();
    const remote = remoteMaxReportId();
    console.log(`  snapshot  local MAX(id) ${local} vs remote ${remote}`);
    if (local !== remote) {
        console.error(
            `\nREFUSING: the local mirror is ${remote > local ? `${remote - local} row(s) behind` : 'ahead of'} production.\n`
            + '  The buffers would be rebuilt from a stale snapshot. Re-fetch INSIDE the\n'
            + '  freeze: ./.venv/bin/python3 -m pipeline.fetch_database\n'
            + '  (A purge changes rows without changing MAX(id) — after surgery, verify with\n'
            + '  the runbook\'s COUNT/SUM check rather than relying on this one.)');
        process.exit(1);
    }
    return { origin, maxId: remote };
}

// Turns "the window was closed before we started" into "it stayed closed throughout".
async function postflightRemote(before) {
    const { frozen, detail } = await probeProductionFrozen(before.origin);
    const remote = remoteMaxReportId();
    const ok = frozen && remote === before.maxId;
    console.log(`\npostflight: freeze ${frozen ? 'ON' : 'OFF'} (${detail}); MAX(id) ${before.maxId} -> ${remote}`);
    if (ok) { console.log('window held — no attempt landed during the apply.'); return true; }
    console.error(
        '\nWARNING: the write freeze did not hold for the whole run.\n'
        + `  ${frozen ? '' : 'Production accepted writes again before the apply finished.\n  '}`
        + `${remote === before.maxId ? '' : `${remote - before.maxId} attempt(s) landed, so those users' buffers may have been truncated.\n  `}`
        + '  Re-freeze, re-run pipeline.fetch_database, and re-run this script to rebuild\n'
        + '  every buffer from a clean snapshot — the rebuild is idempotent and repairs it.');
    return false;
}

// --- SQL literal helpers -----------------------------------------------------------
const sqlNum = (v) => (Number.isFinite(v) ? String(v) : 'NULL');
const sqlStr = (v) => `'${String(v).replace(/'/g, "''")}'`;

// Deduped, ID-ordered last <= cap novel attempts for one user (the buffer feed).
// `uid` is the sql-quoted user id. Returns ascending [{p,s,t}].
function fetchBufferAttempts(uid, cap) {
    // Deduped rows: GROUP BY (problem, created_at) collapses the minted
    // exact-duplicate rows; SQLite's bare-column rule with the single MIN(id)
    // aggregate keeps the ORIGINAL row's values.
    return sqliteJson(
        `SELECT p, s, t FROM (
            SELECT MIN(id) AS id, problem AS p, success AS s, time_used AS t
              FROM tsumego_reports
             WHERE user_id = ${uid} AND is_repeat = 0
             GROUP BY problem, created_at
             ORDER BY id DESC LIMIT ${cap})
          ORDER BY id ASC`
    ).map((r) => ({
        p: r.p, s: r.s ? 1 : 0, t: Number.isFinite(r.t) ? r.t : null,
    }));
}

// --- per-user recompute --------------------------------------------------------------
// attempts: ascending id order, exact dups collapsed, [{ p, s, t }]. The buffer is
// those attempts newest first (the live-buffer order).
function recomputeRow(lb, attempts, params) {
    const buffer = attempts.slice().reverse().map(({ p, s, t }) => ({ p, s, t }));
    const { rank, se, deltaRating } = lb.computeRank(buffer, params);
    return {
        buffer,
        rank,
        se: rank === null ? null : se,
        deltaRating: rank === null ? null : deltaRating,
    };
}

// --- output ---------------------------------------------------------------------------
function fmtRank(lb, v) {
    return Number.isFinite(v) ? `${v.toFixed(2)} (${lb.rankToLabel(v)})` : '—';
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

    // Preflight BEFORE the per-user recompute, so a missing freeze costs seconds
    // rather than minutes of work thrown away.
    let frozenWindow = null;
    if (applyRemote && !allowUnfrozen) {
        frozenWindow = await preflightRemote();
    } else if (applyRemote) {
        console.warn('WARNING: --allow-unfrozen — skipping the freeze/snapshot preflight.\n'
            + '  Any attempt reported during this run is truncated out of that user\'s buffer.');
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
        'SELECT user_id, pred_rank, pred_rank_se FROM leaderboard'
    );
    if (onlyUsers.size) rows = rows.filter((r) => onlyUsers.has(r.user_id));
    if (!rows.length && !populate) { console.log('no leaderboard rows to update'); return; }

    const statements = [];
    for (const row of rows) {
        if (!USER_ID_RE.test(row.user_id)) {
            console.warn(`skipping malformed user_id: ${JSON.stringify(row.user_id)}`);
            continue;
        }
        const uid = sqlStr(row.user_id);
        const attempts = fetchBufferAttempts(uid, lb.BUFFER_CAP);

        const out = recomputeRow(lb, attempts, params);
        console.log(`${row.user_id}  buffer ${attempts.length} attempts`);
        console.log(`  current   ${fmtDelta(lb, row.pred_rank, out.rank)}   se ${Number.isFinite(row.pred_rank_se) ? row.pred_rank_se.toFixed(3) : '—'} -> ${Number.isFinite(out.se) ? out.se.toFixed(3) : '—'}`);

        statements.push(
            `UPDATE leaderboard SET buffer = ${sqlStr(lb.serializeBuffer(out.buffer))}, `
            + `pred_rank = ${sqlNum(out.rank)}, pred_rank_se = ${sqlNum(out.se)}, `
            + `time_delta_rating = ${sqlNum(out.deltaRating)} `
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
            const attempts = fetchBufferAttempts(uid, lb.BUFFER_CAP);
            const out = recomputeRow(lb, attempts, params);
            if (out.rank === null) unranked++; else ranked++;
            console.log(`  + ${a.user_id}  ${attempts.length} novel attempts  ${out.rank === null ? 'unranked' : fmtRank(lb, out.rank)}`);
            // Same seed shape as createLeaderboardRow; INSERT OR IGNORE is a no-op if
            // the row was created concurrently (a live report during a --remote apply).
            statements.push(
                `INSERT OR IGNORE INTO leaderboard `
                + `(user_id, buffer, pred_rank, pred_rank_se, time_delta_rating, last_active) `
                + `VALUES (${uid}, ${sqlStr(lb.serializeBuffer(out.buffer))}, `
                + `${sqlNum(out.rank)}, ${sqlNum(out.se)}, ${sqlNum(out.deltaRating)}, `
                + `${sqlStr(a.last_created)});`
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
        if (res.status !== 0) process.exit(res.status ?? 1);
        // Re-check the window only if we established one; --allow-unfrozen opted out.
        if (frozenWindow && !await postflightRemote(frozenWindow)) process.exit(1);
        process.exit(0);
    } else if (applyLocal) {
        const dbPath = findMiniflareD1();
        console.log(`applying to local miniflare D1: ${dbPath} (stop npm run dev first)`);
        execFileSync('sqlite3', [dbPath, `.read ${sqlOut}`], { stdio: 'inherit' });
    } else {
        console.log('dry run — re-run with --local (miniflare D1) or --remote (production D1) to write.');
    }
}

await main();
