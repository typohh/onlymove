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
//   reads  <- db/d1.sqlite  for --remote / dry runs (the local mirror; run
//             `./.venv/bin/python3 -m pipeline.fetch_database` FIRST so the
//             leaderboard rows + tsumego_reports are fresh)
//          <- the LOCAL miniflare D1 itself for --local, so the update is computed
//             from the same data it writes and a dev session's newer rows are kept
//   writes -> a generated SQL file, applied to the LOCAL miniflare D1
//             (--local) or to production (--remote); neither = dry run
//
// IT NEVER WRITES db/d1.sqlite. The --remote path needs a mirror fetched INSIDE the
// freeze (below), but it takes that into a PRIVATE COPY under $TMPDIR and deletes it
// on the way out, so the canonical db/d1.{sqlite,sql} and the wrangler dev D1 are
// exactly as they were — a re-rate leaves other work reading the snapshot it was
// reading, and `npm run dev` may stay up. `--keep-refresh` opts into the in-place
// refresh instead, which is then a plain `pipeline.fetch_database` with all of its
// usual effects (and so requires `npm run dev` stopped).
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
// The monthly PEAK tracker (best_rank / best_rank_timed / best_rank_month) is
// CLEARED for every row, not carried over: this script exists because the rank
// scale changed, and a peak recorded on the old scale is not comparable with one
// recorded on the new — keeping it would mix two scales in the same month's
// snapshot, silently. Clearing can only OMIT a user from the current month's
// board until their next novel /report seeds a fresh peak, whereas carrying a
// stale-scale peak would publish a wrong number. The finished months already
// frozen in monthly_leaderboards are untouched (they are history).
//
// (Before 2026-07-27 the columns held an EGF-frame rank, so this script's dry-run
// output labelled a GoR scalar with the internal-rank labeller and printed nonsense
// ranks. The stored value is now genuinely an internal scalar, so fmtRank is right.)
//
// Order of operations for a re-rate: package_for_pwa -> `npm run deploy` (so the
// Worker computes on the new scale too) -> this script. If you run it before
// deploying, the very next /report recomputes the user's current rank on the OLD
// scale and overwrites it.
//
// IT RUNS UNDER THE MAINTENANCE FREEZE, AND --remote OPENS AND CLOSES THAT WINDOW
// ITSELF. The `buffer` column is maintained INCREMENTALLY in production:
// updateLeaderboardOnReport reads the stored buffer, pushes the one new attempt
// and writes it back, and NOTHING rebuilds it from tsumego_reports afterwards. So
// every attempt reported between this script's snapshot and its write is silently
// truncated out of that user's buffer when the UPDATE lands — permanently. Later
// reports append on top of the shortened list rather than restoring it, and the
// only rebuild path is createLeaderboardRow, which runs only for a user who has NO
// row at all. There is no error and no recovery; the damage is a slightly wrong
// rank for anyone who happened to be playing.
//
// With MAINTENANCE_MODE = "1" every POST 503s, so nothing can land in that gap and
// the race is not merely unlikely but IMPOSSIBLE. So `--remote` is one call:
//
//   freeze (wrangler secret put MAINTENANCE_MODE=1)  ->  wait for a probe to
//   confirm it  ->  fetch a private mirror  ->  snapshot check  ->  recompute
//   ->  apply  ->  postflight  ->  thaw (MAINTENANCE_MODE=0), verified.
//
// MAINTENANCE_MODE is a wrangler SECRET rather than a [vars] entry precisely so
// this script can flip it: `secret put` pushes a new Worker version reusing the
// existing code bundle, so there is no build, no commit and no clean-tree gate,
// while propagation stays a real version rollout rather than an edge write. That
// rollout takes seconds and lands UNEVENLY, so awaitStableFreezeState requires the
// same answer on several consecutive probes before believing it — a single probe
// during propagation can be answered by a colo on either version.
//
// THE THAW RUNS ON EVERY EXIT PATH — a throw, a refusal, SIGINT — because the
// failure mode of this script owning the freeze is leaving production frozen. If
// the thaw itself cannot be confirmed the script says so loudly and exits non-zero;
// lift it by hand with `printf 0 | npx wrangler secret put MAINTENANCE_MODE`.
//
// IF PRODUCTION IS ALREADY FROZEN when it starts, it does NOT take ownership: it
// leaves the freeze ON at the end and does NOT re-fetch the local mirror. That is
// the db_surgery_runbook step-8 case — an outer window whose step 9 does the thaw,
// and whose local mirror is the intended POST-surgery state that an incremental
// fetch cannot reproduce (fetch_database pulls `WHERE id > MAX(local id)` and never
// re-reads the rows a purge rewrote).
//
// Usage (dry run by default; pick the apply target explicitly):
//   node auxiliary/update_leaderboard_ranks.mjs                   # dry run: print + write SQL file
//   node auxiliary/update_leaderboard_ranks.mjs --local           # ... and execute against the local miniflare D1
//                                                                 #     (stop `npm run dev` first; the monthly columns
//                                                                 #     and table are added automatically if missing)
//   node auxiliary/update_leaderboard_ranks.mjs --remote          # ... against the REMOTE production D1
//   node auxiliary/update_leaderboard_ranks.mjs --populate        # ALSO seed anonymous rows for users active in the
//                                                                 #     past week with no leaderboard row (production
//                                                                 #     only adds them lazily on their next /report)
//   node auxiliary/update_leaderboard_ranks.mjs --user <id>       # limit to one user (repeatable; also scopes populate)
//                                                                 #     and SKIPS the history rebuild (full-table only)
//   node auxiliary/update_leaderboard_ranks.mjs --no-history      # skip the monthly-history rebuild (see below) and
//                                                                 #     just CLEAR the peak columns, the pre-history
//                                                                 #     behaviour for a re-rate
//   node auxiliary/update_leaderboard_ranks.mjs --remote --keep-refresh
//                                                                 #     take the in-window fetch into db/d1.sqlite
//                                                                 #     itself (overwrites it + the dev D1; stop
//                                                                 #     `npm run dev` first) instead of a private copy
//   node auxiliary/update_leaderboard_ranks.mjs --remote --allow-unfrozen
//                                                                 #     manage the freeze yourself: no freeze, no fetch,
//                                                                 #     no snapshot check, no thaw (see above)
//   node auxiliary/update_leaderboard_ranks.mjs --remote --origin https://…
//                                                                 #     probe a different origin (default: the custom
//                                                                 #     domain in pwa/wrangler.toml)
//
// By DEFAULT the run also REBUILDS THE MONTHLY HISTORY. A re-rate changes the
// rank scale, and monthly_leaderboards stores ranks, so every frozen month is
// stale until the same reconstruction runs: for each user it walks their whole
// NOVEL attempt history (created_at order) once, tracks the peak rank per UTC
// month, and rewrites `monthly_leaderboards` for every FINISHED month plus the
// live row's `best_*` for the current one. This is also what gives a fresh
// deployment full history instead of only the months since the tracker shipped.
// It is a full-table operation, so it is skipped (with a warning) when --user
// scopes the run; `--no-history` opts out and merely clears the peak columns.
// The walk is O(n) over the attempt log — see windowRankWalk.

import { execFileSync, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PWA_DIR = path.join(REPO, 'pwa');
// The canonical local mirror. Read-only here: with --keep-refresh the in-window
// fetch updates it in place, otherwise `mirrorDb` is repointed at a private copy.
const DB_PATH = path.join(REPO, 'db', 'd1.sqlite');
let mirrorDb = DB_PATH;
const PARAMS_BIN = path.join(PWA_DIR, 'tsumegos', 'problem_params.bin');
const VENV_PY = path.join(REPO, '.venv', 'bin', 'python3');
// The freeze flag is a wrangler secret (see pwa/wrangler.toml) so it can be flipped
// without a build or a deploy. "1" = frozen; anything else, including unset, = open.
const FREEZE_SECRET = 'MAINTENANCE_MODE';
// A secret push is a version rollout, so it propagates over seconds like a deploy —
// and unevenly, so a state counts as read only once it repeats (awaitStableFreezeState).
const FREEZE_PROBE_RETRIES = 15;
const FREEZE_PROBE_DELAY_MS = 2000;
const FREEZE_STABLE_PROBES = 3;
// `wrangler dev`'s port. An in-place fetch overwrites the miniflare D1 file, and
// doing that under a live workerd corrupts its view — so --keep-refresh refuses
// while it is up. The private refresh writes neither file and does not care.
const DEV_PORT = 8787;
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
let keepRefresh = false;
// The monthly-history rebuild (see `--no-history`): ON by default, because a
// re-rate invalidates the frozen monthly boards and the same walk that refreshes
// the current ranks also reconstructs them.
let history = true;
let originOverride = null;
const onlyUsers = new Set();
let sqlOut = path.join(os.tmpdir(), 'leaderboard_rank_update.sql');
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--local') applyLocal = true;
    else if (a === '--remote') applyRemote = true;
    else if (a === '--populate') populate = true;
    else if (a === '--allow-unfrozen') allowUnfrozen = true;
    else if (a === '--keep-refresh') keepRefresh = true;
    else if (a === '--no-history') history = false;
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
                export { computeRank, serializeBuffer, RANK_START_MIN, BUFFER_CAP, monthlyPeak,
                         monthlyPeaksFromAttempts, monthKeyAt, declaredJsonSql,
                         resolveRowCountry } from './src/leaderboard.js';
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
    const out = execFileSync('sqlite3', ['-json', mirrorDb, sql],
        { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 });
    const text = out.trim();
    return text ? JSON.parse(text) : [];
}

// Same, against an explicit DB file (the miniflare target, which is not `mirrorDb`).
function sqliteJsonOn(dbPath, sql) {
    const out = execFileSync('sqlite3', ['-json', dbPath, sql], { encoding: 'utf8' });
    const text = out.trim();
    return text ? JSON.parse(text) : [];
}

// The schema the generated SQL names, added to a --local miniflare D1 when missing.
// The guard, not the migration, is what makes this idempotent: the migration's
// ALTERs would fail on a re-run, so it is only applied when a piece is absent.
// (Remote is checked, not migrated — see remoteHasLeaderboardSchema.)
const MONTHLY_MIGRATION = path.join(PWA_DIR, 'migrations', '2026-09-21_monthly_leaderboard.sql');
const DECLARED_MIGRATIONS = ['2026-09-23_leaderboard_declared.sql', '2026-09-23_leaderboard_declared_backfill.sql']
    .map((f) => path.join(PWA_DIR, 'migrations', f));

function ensureLocalSchema(dbPath) {
    const cols = new Set(sqliteJsonOn(dbPath, "PRAGMA table_info(leaderboard)").map((c) => c.name));
    const has = (c) => cols.has(c);
    const hasTable = sqliteJsonOn(dbPath,
        "SELECT name FROM sqlite_master WHERE type='table' AND name='monthly_leaderboards'").length > 0;
    const apply = (file) => {
        console.log(`applying ${path.relative(REPO, file)} to the local miniflare D1...`);
        execFileSync('sqlite3', [dbPath], { input: fs.readFileSync(file, 'utf8'), encoding: 'utf8' });
    };
    if (!(has('best_rank') && has('best_rank_timed') && has('best_rank_month') && hasTable)) {
        apply(MONTHLY_MIGRATION);
    }
    if (!has('declared')) for (const file of DECLARED_MIGRATIONS) apply(file);
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

// The generated SQL names monthly_leaderboards and the leaderboard best_* columns,
// so the remote target must already carry them. Refuse BEFORE the freeze/fetch
// when it does not, rather than letting half the file apply and fail. Applying the
// migration is a deliberate, separate step (see the migration header).
function remoteHasLeaderboardSchema() {
    const cols = new Set(remoteQueryRows("SELECT name FROM pragma_table_info('leaderboard')").map((r) => r.name));
    const hasTable = remoteQueryRows(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='monthly_leaderboards'").length > 0;
    return cols.has('best_rank') && cols.has('best_rank_timed') && cols.has('best_rank_month') && hasTable
        && cols.has('declared');
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
//
// TRI-STATE on purpose. A network error is 'unknown', NOT 'open': this is polled to
// confirm a thaw, and collapsing "we could not reach production" into "writes are
// open" would report a freeze lifted that is still on.
async function probeProductionFreeze(origin) {
    let res;
    try {
        res = await fetch(`${origin}/sync`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: '{}',
        });
    } catch (err) {
        return { state: 'unknown', detail: `request failed: ${err.message}` };
    }
    if (res.status !== 503) {
        return res.status < 500
            ? { state: 'open', detail: `POST /sync -> ${res.status}` }
            : { state: 'unknown', detail: `POST /sync -> ${res.status} (edge error?)` };
    }
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON 503 = not our freeze */ }
    if (body?.error !== 'maintenance') {
        return { state: 'unknown', detail: 'got a 503 that is not the maintenance freeze (edge error?)' };
    }
    return { state: 'frozen', detail: 'POST /sync -> 503 {error:"maintenance"}' };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Flip the freeze. `secret put` reuses the deployed code bundle, so this is a version
// rollout with no build and no commit — see the header and pwa/wrangler.toml.
function setFreezeSecret(on) {
    const res = spawnSync('npx', ['wrangler', 'secret', 'put', FREEZE_SECRET],
        { cwd: PWA_DIR, encoding: 'utf8', input: on ? '1' : '0' });
    if (res.status !== 0) {
        throw new Error(`wrangler secret put ${FREEZE_SECRET} failed (${res.status}): `
            + stripAnsi(res.stderr || res.stdout || '').trim());
    }
}

// Poll until production reports a state STABLY — the same answer on
// FREEZE_STABLE_PROBES consecutive probes. One probe is not enough and that is
// measured, not defensive: a version rollout does not flip every colo at the same
// instant, and inside that propagation window consecutive probes disagree. A run
// that read a single 503 from a straggler concluded it was inside an outer surgery
// window; probes seconds later were unanimously OPEN. A single reading therefore
// says what one edge answered, not what the fleet is doing.
//
// This narrows the window rather than abolishing it — the authoritative check that
// nothing slipped through stays the MAX(id) comparison in postflightRemote.
//
// Fails CLOSED in both directions: 'unknown' never satisfies a `want`, and with
// want=null (reading the state we START in) an unstable fleet throws instead of
// picking one of the answers.
async function awaitStableFreezeState(origin, want, label) {
    let run = 0, prev = null, last = '';
    for (let attempt = 1; attempt <= FREEZE_PROBE_RETRIES; attempt++) {
        const { state, detail } = await probeProductionFreeze(origin);
        last = detail;
        run = state === prev ? run + 1 : 1;
        prev = state;
        const satisfied = want === null ? state !== 'unknown' : state === want;
        if (run >= FREEZE_STABLE_PROBES && satisfied) return { state, detail };
        if (attempt < FREEZE_PROBE_RETRIES) await sleep(FREEZE_PROBE_DELAY_MS);
    }
    const waited = Math.round(FREEZE_PROBE_RETRIES * FREEZE_PROBE_DELAY_MS / 1000);
    throw new Error(`${label}: production never held ${want ? `writes ${want.toUpperCase()}` : 'one state'} `
        + `across ${FREEZE_STABLE_PROBES} consecutive probes in ${waited}s — last: ${last}`);
}

// --- the write freeze this run owns -------------------------------------------------
// Set the moment we ASK for the freeze (not once it is confirmed), so an interrupt
// during the confirmation poll still thaws. Null whenever we do not own one: a dry
// run, --allow-unfrozen, or an outer window that was already frozen when we started.
let heldFreeze = null;

async function releaseFreeze() {
    if (!heldFreeze) return true;
    const { origin } = heldFreeze;
    heldFreeze = null;
    console.log(`\nthawing writes (${FREEZE_SECRET}=0)...`);
    try {
        setFreezeSecret(false);
        console.log(`  freeze    OFF — ${(await awaitStableFreezeState(origin, 'open', 'thaw')).detail}`);
        return true;
    } catch (err) {
        console.error(`\n*** PRODUCTION MAY STILL BE FROZEN — EVERY WRITE IS BEING REFUSED ***\n`
            + `  ${err.message}\n`
            + '  Lift it by hand, now:\n'
            + `    cd pwa && printf 0 | npx wrangler secret put ${FREEZE_SECRET}\n`
            + '    node auxiliary/check_live_status.mjs --expect-unfrozen');
        return false;
    }
}

// --- the in-window mirror refresh -----------------------------------------------------
function devServerRunning() {
    return new Promise((resolve) => {
        const sock = net.connect({ host: '127.0.0.1', port: DEV_PORT });
        const done = (v) => { sock.destroy(); resolve(v); };
        sock.setTimeout(500);
        sock.once('connect', () => done(true));
        sock.once('timeout', () => done(false));
        sock.once('error', () => done(false));
    });
}

// The private mirror this run owns, or null. Removed on EVERY exit path (the
// process 'exit' hook below), so a crash inside the freeze leaves no orphan copy
// of the whole database behind.
let privateMirrorDir = null;

function discardPrivateMirror() {
    if (!privateMirrorDir) return;
    const dir = privateMirrorDir;
    privateMirrorDir = null;
    try {
        fs.rmSync(dir, { recursive: true, force: true });
    } catch (err) {
        console.error(`could not remove the private mirror at ${dir}: ${err.message}`);
    }
}

// Pull the rows that landed between the last ordinary fetch and the freeze.
//
// By DEFAULT this goes into a private copy: db/d1.sqlite is copied into a temp dir,
// `--db-dir` points the incremental fetch at that copy (which is also its id
// baseline, so the pull stays the usual `WHERE id > MAX(local id)` handful of rows),
// and everything below reads it via `mirrorDb`. The canonical mirror is a shared
// input — other analysis is mid-flight against it and a re-rate is reproducible only
// against a fixed snapshot — so a leaderboard refresh has no business advancing it.
// The private dir also skips the dev-D1 sync, so `npm run dev` may stay up.
//
// --keep-refresh fetches in place and keeps the result, for when you WANT the
// canonical mirror advanced to the freeze boundary anyway. It then carries every
// effect of an ordinary fetch, the miniflare overwrite included.
async function refreshLocalMirror() {
    if (!fs.existsSync(VENV_PY)) {
        throw new Error(`missing ${VENV_PY} — cannot refresh the local mirror inside the freeze`);
    }
    const fetchArgs = ['-u', '-m', 'pipeline.fetch_database'];
    let privateDb = null;
    if (keepRefresh) {
        if (await devServerRunning()) {
            throw new Error(`something is listening on 127.0.0.1:${DEV_PORT} (wrangler dev?).\n`
                + '  --keep-refresh overwrites the miniflare D1 file, and doing that under a live\n'
                + '  workerd corrupts its view. Stop `npm run dev`, or drop --keep-refresh so the\n'
                + '  fetch goes to a private copy and touches neither file.');
        }
        console.log(`\nrefreshing ${DB_PATH} IN PLACE inside the freeze (--keep-refresh)...`);
    } else {
        privateMirrorDir = fs.mkdtempSync(path.join(os.tmpdir(), 'leaderboard-mirror-'));
        privateDb = path.join(privateMirrorDir, path.basename(DB_PATH));
        console.log(`\nrefreshing a PRIVATE copy of the mirror inside the freeze:\n  ${privateDb}`
            + `\n  (${DB_PATH} and the dev D1 are left untouched — pass --keep-refresh to advance them)`);
        fs.copyFileSync(DB_PATH, privateDb);
        fetchArgs.push('--db-dir', privateMirrorDir);
    }
    const res = spawnSync(VENV_PY, fetchArgs, { cwd: REPO, stdio: 'inherit' });
    if (res.status !== 0) throw new Error(`pipeline.fetch_database failed (${res.status})`);
    if (privateDb) mirrorDb = privateDb;
}

const localMaxReportId = () => sqliteJson('SELECT MAX(id) AS m FROM tsumego_reports')[0]?.m ?? null;
const remoteMaxReportId = () => remoteQueryRows('SELECT MAX(id) AS m FROM tsumego_reports')[0]?.m ?? null;

// Open the window: freeze unless someone else already has, refresh the mirror inside
// it, and refuse unless the snapshot matches production exactly. Throws rather than
// exits so the caller's finally still thaws whatever we turned on.
async function openWindow() {
    const origin = productionOrigin();
    console.log(`preflight: ${origin}`);

    const before = await awaitStableFreezeState(origin, null, 'initial freeze read');
    console.log(`  freeze    ${before.state.toUpperCase()} — ${before.detail}`);

    const wasAlreadyFrozen = before.state === 'frozen';
    if (wasAlreadyFrozen) {
        console.log('  production was ALREADY frozen — this is an outer surgery window, so\n'
            + '  leaving it frozen on exit and trusting db/d1.sqlite as it stands (no fetch:\n'
            + '  an incremental pull cannot see rows a purge rewrote).');
    } else {
        console.log(`  freezing writes (${FREEZE_SECRET}=1)...`);
        heldFreeze = { origin };
        setFreezeSecret(true);
        console.log(`  freeze    ON — ${(await awaitStableFreezeState(origin, 'frozen', 'freeze')).detail}`);
        await refreshLocalMirror();
    }

    const local = localMaxReportId();
    const remote = remoteMaxReportId();
    console.log(`\n  snapshot  local MAX(id) ${local} vs remote ${remote}`);
    if (local !== remote) {
        throw new Error(
            `the local mirror is ${remote > local ? `${remote - local} row(s) behind` : 'ahead of'} production.\n`
            + '  The buffers would be rebuilt from a stale snapshot.\n'
            + (wasAlreadyFrozen
                ? '  Inside an outer window this script does not re-fetch — reconcile the mirror\n'
                  + '  yourself (see docs/db_surgery_runbook.md) and re-run.\n'
                : '  The in-window pipeline.fetch_database should have made these equal.\n')
            + '  (A purge changes rows without changing MAX(id) — after surgery, verify with\n'
            + '  the runbook\'s COUNT/SUM check rather than relying on this one.)');
    }
    return { origin, maxId: remote };
}

// Turns "the window was closed before we started" into "it stayed closed throughout".
async function postflightRemote(before) {
    const { state, detail } = await probeProductionFreeze(before.origin);
    const remote = remoteMaxReportId();
    const ok = state === 'frozen' && remote === before.maxId;
    console.log(`\npostflight: freeze ${state.toUpperCase()} (${detail}); MAX(id) ${before.maxId} -> ${remote}`);
    if (ok) { console.log('window held — no attempt landed during the apply.'); return true; }
    console.error(
        '\nWARNING: the write freeze did not hold for the whole run.\n'
        + `  ${state === 'frozen' ? '' : 'Production was not confirmed frozen at the end of the apply.\n  '}`
        + `${remote === before.maxId ? '' : `${remote - before.maxId} attempt(s) landed, so those users' buffers may have been truncated.\n  `}`
        + '  Just re-run this script: it re-freezes, re-fetches and rebuilds every buffer\n'
        + '  from a clean snapshot — the rebuild is idempotent and repairs it.');
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

// The user's ENTIRE novel history for the monthly reconstruction, ascending id
// order and deduped exactly like fetchBufferAttempts, plus each attempt's
// `created_at` as `at` (ms). The month is taken from `at` — the month the attempt
// HAPPENED, which is what a historical reconstruction wants; the live tracker
// uses the report timestamp instead. NO REPEATS: the rank window is novel-only.
function fetchHistoryAttempts(uid) {
    return sqliteJson(
        `SELECT p, s, t, at FROM (
            SELECT MIN(id) AS id, problem AS p, success AS s, time_used AS t, created_at AS at
              FROM tsumego_reports
             WHERE user_id = ${uid} AND is_repeat = 0
             GROUP BY problem, created_at)
          ORDER BY id ASC`
    ).map((r) => ({
        p: r.p, s: r.s ? 1 : 0,
        t: Number.isFinite(r.t) ? r.t : null,
        at: Date.parse(r.at),
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
    if (!fs.existsSync(PARAMS_BIN)) {
        throw new Error(`missing ${PARAMS_BIN} — run: ./.venv/bin/python3 -m pipeline.package_for_pwa`);
    }
    // --local reads the LOCAL D1 ITSELF, not the canonical mirror: the local
    // database is the source of truth for its own data, so a dev session's writes
    // since the last fetch_database are not rebuilt away from a stale snapshot.
    // --remote and dry runs read the mirror (the freeze snapshot).
    if (applyLocal) {
        mirrorDb = findMiniflareD1();
        console.log(`reading local D1: ${mirrorDb}`);
    } else if (!fs.existsSync(DB_PATH)) {
        throw new Error(`missing ${DB_PATH} — run: ./.venv/bin/python3 -m pipeline.fetch_database --full`);
    }

    // The generated SQL names the monthly schema, and the remote migration is a
    // deliberate separate step, so refuse before freezing/fetching if it is absent.
    if (applyRemote && !remoteHasLeaderboardSchema()) {
        throw new Error('remote D1 is missing the monthly-leaderboard or declared-cache schema.\n'
            + '  Apply migrations/2026-09-21_monthly_leaderboard.sql and/or\n'
            + '  migrations/2026-09-23_leaderboard_declared.sql FIRST (their headers have the\n'
            + '  commands and the order), then deploy, then re-run this script.');
    }

    // Open the window BEFORE the per-user recompute, so a refusal costs seconds
    // rather than minutes of work thrown away. It also refreshes the local mirror,
    // so everything below reads the in-window snapshot.
    let frozenWindow = null;
    if (applyRemote && !allowUnfrozen) {
        frozenWindow = await openWindow();
    } else if (applyRemote) {
        console.warn('WARNING: --allow-unfrozen — not freezing, not fetching, not checking the\n'
            + '  snapshot, not thawing. Any attempt reported during this run is truncated out\n'
            + '  of that user\'s buffer.');
    }
    if (keepRefresh && !frozenWindow) {
        console.warn('NOTE: --keep-refresh does nothing here — only the freeze window fetches, and\n'
            + '  this run does not open one. Run pipeline.fetch_database yourself.');
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
    if (!rows.length && !populate && !history) { console.log('no leaderboard rows to update'); return; }

    // Users the populate pass adds to the live table; the history pass must be able
    // to write their current-month peak onto the row it creates.
    const populatedUsers = new Set();

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
            + `time_delta_rating = ${sqlNum(out.deltaRating)}, `
            + `best_rank = NULL, best_rank_timed = NULL, best_rank_month = NULL `
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
            populatedUsers.add(a.user_id);
            const attempts = fetchBufferAttempts(uid, lb.BUFFER_CAP);
            const out = recomputeRow(lb, attempts, params);
            if (out.rank === null) unranked++; else ranked++;
            console.log(`  + ${a.user_id}  ${attempts.length} novel attempts  ${out.rank === null ? 'unranked' : fmtRank(lb, out.rank)}`);
            // A seeded row's recomputed rank is its first PEAK for the current month
            // (the row did not exist, so there is no earlier observation to keep).
            const peak = out.rank === null ? null : lb.monthlyPeak(null, out.rank, out.deltaRating, Date.now());
            // Same seed shape as createLeaderboardRow, the declared cache included (it
            // is computed by the TARGET database from its own rank_surveys, not from the
            // mirror); INSERT OR IGNORE is a no-op if the row was created concurrently
            // (a live report during a --remote apply).
            statements.push(
                `INSERT OR IGNORE INTO leaderboard `
                + `(user_id, buffer, pred_rank, pred_rank_se, time_delta_rating, last_active, `
                + `best_rank, best_rank_timed, best_rank_month, declared) `
                + `VALUES (${uid}, ${sqlStr(lb.serializeBuffer(out.buffer))}, `
                + `${sqlNum(out.rank)}, ${sqlNum(out.se)}, ${sqlNum(out.deltaRating)}, `
                + `${sqlStr(a.last_created)}, `
                + `${peak ? sqlNum(peak.best_rank) : 'NULL'}, `
                + `${peak ? sqlNum(peak.best_rank_timed) : 'NULL'}, `
                + `${peak ? sqlStr(peak.best_rank_month) : 'NULL'}, `
                + `${lb.declaredJsonSql(uid)});`
            );
        }
        console.log(`populate: ${ranked} ranked + ${unranked} unranked seeded (unranked rows stay hidden on the board until ${lb.RANK_START_MIN} novel attempts)`);
    }

    // --- monthly history: rebuild every finished month + the live peak --------------
    // A re-rate changes the rank scale and monthly_leaderboards stores ranks, so the
    // frozen months are stale until rebuilt; the same walk gives a fresh deploy full
    // history. Per user we replay the WHOLE novel history once (created_at order) and
    // take the peak rank in each UTC month. Finished months go to
    // monthly_leaderboards; the CURRENT, still-open month goes to the live row's
    // best_* (the cron freezes it when it ends). Full-table only: --user scopes the
    // run and would rewrite a month from one user, so it opts out.
    if (history && !onlyUsers.size) {
        const curMonth = lb.monthKeyAt(Date.now());
        // Only users who can possibly carry a rank (>= RANK_START_MIN novel attempts);
        // the walk itself still applies the gate per month.
        const histUsers = sqliteJson(
            `SELECT user_id FROM tsumego_reports WHERE is_repeat = 0
              GROUP BY user_id HAVING COUNT(*) >= ${lb.RANK_START_MIN}`
        ).map((r) => r.user_id).filter((u) => USER_ID_RE.test(u));
        console.log(`\nhistory: replaying ${histUsers.length} user(s) for the monthly boards...`);
        const t0 = Date.now();
        const months = new Map();       // month -> Map<userId, { rank, timed }>
        const currentPeak = new Map();  // userId -> { rank, timed } for the open month
        let replayed = 0;
        for (const u of histUsers) {
            const attempts = fetchHistoryAttempts(sqlStr(u));
            if (attempts.length < lb.RANK_START_MIN) continue;
            const peaks = lb.monthlyPeaksFromAttempts(attempts, params);
            for (const [month, peak] of peaks) {
                if (month === curMonth) { currentPeak.set(u, peak); continue; }
                if (month > curMonth) continue;
                let m = months.get(month);
                if (!m) months.set(month, m = new Map());
                m.set(u, peak);
            }
            if (++replayed % 250 === 0) console.log(`  ${replayed}/${histUsers.length}...`);
        }
        // Each entry's baked country (see recordMonthlyLeaderboards), in the read's
        // own precedence: a user whose live row survives gets its current country;
        // otherwise the one already in the month's snapshot, resolved at the freeze
        // while that row still held the geo-IP; otherwise what survives of the rest —
        // the binding and the declared cache recomputed from rank_surveys.
        const bakedCountry = new Map();  // month -> Map<userId, country>
        const hasMonthly = sqliteJson(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='monthly_leaderboards'").length > 0;
        if (hasMonthly) for (const { month, data } of sqliteJson('SELECT month, data FROM monthly_leaderboards')) {
            let arr = [];
            try { arr = JSON.parse(data) || []; } catch { arr = []; }
            const m = new Map();
            for (const e of arr) if (Array.isArray(e) && typeof e[3] === 'string' && e[3]) m.set(e[0], e[3]);
            bakedCountry.set(month, m);
        }
        const freshCountry = new Map();  // userId -> { country, live }
        for (const r of sqliteJson(
            `SELECT u.user_id, l.country, x.country_code, ${lb.declaredJsonSql('u.user_id')} AS declared,
                    (l.user_id IS NOT NULL) AS live
               FROM (SELECT DISTINCT user_id FROM tsumego_reports WHERE is_repeat = 0) u
               LEFT JOIN leaderboard l ON l.user_id = u.user_id
               LEFT JOIN external_ids x ON x.user_id = u.user_id`
        )) freshCountry.set(r.user_id, { country: lb.resolveRowCountry(r), live: !!r.live });
        const nowIso = new Date().toISOString();
        const orderedMonths = [...months.keys()].sort();
        let entryCount = 0, withCountry = 0;
        for (const month of orderedMonths) {
            const baked = bakedCountry.get(month);
            const entries = [...months.get(month).entries()]
                .map(([u, p]) => {
                    const f = freshCountry.get(u) || { country: null, live: false };
                    const country = (f.live ? f.country : (baked && baked.get(u)) || f.country) || null;
                    if (country) withCountry++;
                    return [u, p.rank, Number.isFinite(p.timed) ? p.timed : null, country];
                });
            entryCount += entries.length;
            statements.push(
                `INSERT INTO monthly_leaderboards (month, data, created_at) `
                + `VALUES (${sqlStr(month)}, ${sqlStr(JSON.stringify(entries))}, ${sqlStr(nowIso)}) `
                + `ON CONFLICT(month) DO UPDATE SET data = excluded.data, created_at = excluded.created_at;`
            );
        }
        const liveUsers = new Set(rows.map((r) => r.user_id));
        for (const u of populatedUsers) liveUsers.add(u);
        let currentRows = 0;
        for (const [u, p] of currentPeak) {
            if (!liveUsers.has(u)) continue;   // no live row to hold it (UPDATE would no-op anyway)
            currentRows++;
            statements.push(
                `UPDATE leaderboard SET best_rank = ${sqlNum(p.rank)}, `
                + `best_rank_timed = ${sqlNum(p.timed)}, best_rank_month = ${sqlStr(curMonth)} `
                + `WHERE user_id = ${sqlStr(u)};`
            );
        }
        console.log(`history: ${orderedMonths.length} finished month(s), ${entryCount} entries `
            + `(${withCountry} with a country); `
            + `${currentRows} live row(s) get the ${curMonth} peak (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
        if (!orderedMonths.length) {
            console.warn('history: no finished months reconstructed — nothing to write to monthly_leaderboards');
        }
    } else if (history && onlyUsers.size) {
        console.warn('NOTE: --user scopes the run, so the monthly-history rebuild is SKIPPED (it is a\n'
            + '  full-table operation and would rewrite a month from one user). The peak columns are\n'
            + '  CLEARED instead; re-run without --user to rebuild history.');
    }

    if (!statements.length) { console.log('\nnothing to write'); return 0; }
    fs.writeFileSync(sqlOut, statements.join('\n') + '\n');
    console.log(`\n${statements.length} statement(s) written to ${sqlOut}`);

    if (applyRemote) {
        console.log('applying to REMOTE production D1...');
        const res = spawnSync('npx', ['wrangler', 'd1', 'execute', 'DB', '--remote', '--yes', `--file=${sqlOut}`],
            { cwd: PWA_DIR, stdio: 'inherit' });
        if (res.status !== 0) throw new Error(`wrangler d1 execute failed (${res.status})`);
        // Re-check the window only if we established one; --allow-unfrozen opted out.
        if (frozenWindow && !await postflightRemote(frozenWindow)) return 1;
        return 0;
    } else if (applyLocal) {
        const dbPath = mirrorDb;
        console.log(`applying to local miniflare D1: ${dbPath} (stop npm run dev first)`);
        ensureLocalSchema(dbPath);
        execFileSync('sqlite3', [dbPath, `.read ${sqlOut}`], { stdio: 'inherit' });
    } else {
        console.log('dry run — re-run with --local (miniflare D1) or --remote (production D1) to write.');
    }
    return 0;
}

// --- exit paths ---------------------------------------------------------------------
// The thaw has to survive EVERY exit path, or a crash mid-run leaves production
// refusing writes indefinitely. THREE of them do not reach the try/finally below,
// and the first is not hypothetical — it is how this was found:
//
//   - A CLOSED STDOUT. Piping into `head` makes the next console.log raise EPIPE,
//     which unwinds past main() without settling it, so the finally never runs and
//     the freeze stays on. Swallowed here instead: a run that owns a production
//     freeze must not die because its READER went away. It finishes the work and
//     thaws with its output discarded.
//   - An uncaught exception or rejection, for the same unwinding reason.
//   - A signal, which delivers no exception at all.
//
// process.exit() skips finally blocks, so nothing below may call it before
// releaseFreeze() has run.
for (const stream of [process.stdout, process.stderr]) {
    stream.on('error', (err) => { if (err.code !== 'EPIPE') throw err; });
}

// The private mirror is discarded from 'exit' rather than from the finally below, for
// the same reason: it has to survive the paths that never reach it. 'exit' runs
// synchronously on process.exit() too, so the signal/crash handler covers it as well.
process.on('exit', discardPrivateMirror);

// SYNCHRONOUS, and that is the whole design of it. An async handler yields at its
// first await, which RESUMES THE SUSPENDED MAIN FLOW — measured: a SIGINT during the
// freeze-confirmation poll thawed production while the run carried on and applied its
// snapshot-relative writes with writes OPEN, the exact race the freeze exists to
// prevent (the postflight caught it, which is why the postflight exists). Pushing the
// secret is already a blocking spawnSync, so the handler can complete the thaw and
// exit before yielding at all, and main never gets another turn.
//
// What that costs is the CONFIRMATION poll — it needs an await, so this path pushes
// the thaw and tells the operator to verify rather than verifying itself. A push that
// wrangler reports as failed still throws here, so the loud case is still covered;
// only the propagation check is deferred. The ordinary exit paths below keep the
// confirmed thaw.
let exiting = false;
function bailOut(label, code) {
    if (exiting) process.exit(code);   // a second signal gives up and leaves it frozen
    exiting = true;
    console.error(`\n${label} — releasing the write freeze before exiting...`);
    if (!heldFreeze) process.exit(code);
    heldFreeze = null;
    try {
        setFreezeSecret(false);
        console.error(`  thaw pushed (${FREEZE_SECRET}=0) but NOT confirmed — verify:\n`
            + '    node auxiliary/check_live_status.mjs --expect-unfrozen');
        process.exit(code);
    } catch (err) {
        console.error('\n*** PRODUCTION MAY STILL BE FROZEN — EVERY WRITE IS BEING REFUSED ***\n'
            + `  ${err.message}\n`
            + '  Lift it by hand, now:\n'
            + `    cd pwa && printf 0 | npx wrangler secret put ${FREEZE_SECRET}\n`
            + '    node auxiliary/check_live_status.mjs --expect-unfrozen');
        process.exit(1);
    }
}
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => bailOut(sig, 130));
process.on('uncaughtException', (err) => bailOut(`uncaught exception: ${err?.stack || err}`, 1));
process.on('unhandledRejection', (err) => bailOut(`unhandled rejection: ${err?.stack || err}`, 1));


let exitCode = 0;
try {
    exitCode = await main() ?? 0;
} catch (err) {
    console.error(`\nREFUSING: ${err.message}`);
    exitCode = 1;
} finally {
    if (!await releaseFreeze()) exitCode = 1;
}
process.exit(exitCode);
