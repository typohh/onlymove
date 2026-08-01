// One-off DB surgery.
//
// Runs against a LOCAL SQLite copy of the D1 export (never remote D1 —
// perform locally, verify, upload the result as a staging table). Imports the
// Worker's exact rowHash from pwa/shared/sync-digest.js so the rewritten
// digest chains are byte-identical to what the live INSERT computes.
//
//   node auxiliary/db_surgery.mjs --db surgery/database.surgery.sqlite \
//        --map surgery/id_map.json [--apply] [--emit-upload surgery/staging_upload.sql]
//
// Steps, in order (each gated on the previous; dry-run prints counts only):
//   pre  — rowHash JS<->SQL parity + verify stored chains of seeded users
//   S7a  — delete the 3 smoke-test rows
//   S7b  — truncate legacy long user ids to 8 chars (2 merge onto existing
//          accounts; affected (user, problem) pairs feed S5)
//   S1   — delete identical-content duplicate rows (keep MIN(id))
//   S3   — NULL unrecoverable pre-2026-07-07 failed-repeat moves
//   S4   — rewrite problem ids through the corpus id map
//   S5   — repeat-flag repair on merge-affected (user, problem) pairs only
//   S6   — recall_schedule problem_id rewrite (merge collision: latest wins)
//   S8   — orphaned recall_schedule / push_subscriptions rows (no attempts)
//   S9   — leaderboard full reset (rows lazily rebuild from repaired reports)
//   S2   — materialize user_seq / running_sum for EVERY user (full chains)
//   post — re-run the section-2 measurement queries; verify every chain
//
// The script is idempotent: re-running on an already-repaired DB finds zero
// rows to touch and the verification still passes.
//
// GOTCHA: when re-copying a fresh DB over a previous rehearsal target, delete
// the -wal/-shm sidecars first (`rm -f target.sqlite*` before `cp`) — a stale
// WAL from the prior run attaches to the same-named main file and makes the
// fresh copy read as already-repaired (verified during rehearsal).

import { DatabaseSync } from 'node:sqlite';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const { rowHash, ID_HASH_SQL } = await import(
  join(HERE, '..', 'pwa', 'shared', 'sync-digest.js'));

// ------------------------------------------------------------------ args
const args = process.argv.slice(2);
function argOf(name, dflt = null) {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
}
const DB_PATH = argOf('--db');
const MAP_PATH = argOf('--map');
const APPLY = args.includes('--apply');
const EMIT_UPLOAD = argOf('--emit-upload');
if (!DB_PATH || !MAP_PATH) {
  console.error('usage: node auxiliary/db_surgery.mjs --db <sqlite> --map <id_map.json> [--apply] [--emit-upload <path>]');
  process.exit(2);
}

const SMOKE_USERS = ['smoketest', 'zzsmoke09', 'deployv1'];
const MOVE_NULL_CUTOFF = '2026-07-07T00:00:00Z'; // CANONICAL_MOVE_ERA_MS

const db = new DatabaseSync(DB_PATH);
db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;');
const idMap = JSON.parse(readFileSync(MAP_PATH, 'utf8'));
console.log(`db: ${DB_PATH}\nmap: ${idMap.renames} renames + ${idMap.merges} merges\nmode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);

const one = (sql, ...p) => db.prepare(sql).get(...p);
const all = (sql, ...p) => db.prepare(sql).all(...p);
const run = (sql, ...p) => db.prepare(sql).run(...p);
const count = (sql, ...p) => Number(Object.values(one(sql, ...p))[0]);

function fail(msg) {
  console.error(`\nHARD STOP: ${msg}`);
  process.exit(1);
}

// ------------------------------------------------- pre: rowHash parity
{
  const probe = all(`SELECT id, ${ID_HASH_SQL} AS h FROM tsumego_reports ORDER BY id DESC LIMIT 5`);
  for (const r of probe) {
    if (rowHash(Number(r.id)) !== Number(r.h)) {
      fail(`rowHash parity: JS ${rowHash(Number(r.id))} != SQL ${r.h} for id ${r.id}`);
    }
  }
  console.log(`pre: rowHash JS<->SQL parity OK (${probe.length} probes)`);

  // Verify stored chains of up to 25 users whose rows are fully seeded — the
  // "recompute an unaffected user's chain against stored values" gate.
  const seeded = all(
    `SELECT user_id FROM tsumego_reports GROUP BY user_id
      HAVING COUNT(*) = COUNT(user_seq) AND COUNT(*) >= 3 LIMIT 25`);
  let checked = 0;
  for (const { user_id } of seeded) {
    const rows = all(
      `SELECT id, user_seq, running_sum FROM tsumego_reports WHERE user_id = ? ORDER BY id`, user_id);
    let seq = 0, sum = 0;
    for (const r of rows) {
      seq += 1;
      if (Number(r.user_seq) !== seq || Number(r.running_sum) !== sum) {
        fail(`pre: stored chain mismatch for ${user_id} at id ${r.id}: stored (${r.user_seq}, ${r.running_sum}) recomputed (${seq}, ${sum})`);
      }
      sum += rowHash(Number(r.id));
    }
    checked++;
  }
  console.log(`pre: ${checked} fully-seeded users' stored chains verified against JS recompute`);
}

if (APPLY) db.exec('BEGIN');

// ------------------------------------------------- S7a: smoke rows
{
  const n = count(
    `SELECT COUNT(*) FROM tsumego_reports WHERE user_id IN ('${SMOKE_USERS.join("','")}')`);
  if (n !== 3 && n !== 0) console.warn(`S7a: expected 3 smoke rows, found ${n} (proceeding)`);
  if (APPLY) run(`DELETE FROM tsumego_reports WHERE user_id IN ('${SMOKE_USERS.join("','")}')`);
  console.log(`S7a: smoke-test rows deleted: ${n}`);
}

// ------------------------------------------------- S7b: long user ids
let s7MergePairs = [];
{
  // Exclude the smoke users: S7a deletes them, but in a dry run they are
  // still present and two are 9 chars long.
  const longs = all(
    `SELECT DISTINCT user_id FROM tsumego_reports
      WHERE length(user_id) != 8 AND user_id NOT IN ('${SMOKE_USERS.join("','")}')`);
  const merges = [];
  for (const { user_id } of longs) {
    if (user_id.length < 8) fail(`S7b: user id shorter than 8: ${user_id}`);
    const trunc = user_id.slice(0, 8);
    const exists = count(
      `SELECT COUNT(*) FROM tsumego_reports WHERE user_id = ?`, trunc) > 0;
    if (exists) merges.push([user_id, trunc]);
  }
  console.log(`S7b: ${longs.length} long user ids (expected 6); ${merges.length} merge onto existing accounts (expected 2)`);
  if (longs.length > 0 && longs.length !== 6) console.warn('S7b: count drifted from the 2026-07-20 measurement');
  // Capture (user, problem) pairs that the account merge will unify — S5 input.
  for (const [longId, trunc] of merges) {
    const shared = all(
      `SELECT DISTINCT problem FROM tsumego_reports
        WHERE user_id = ? AND problem IN
          (SELECT problem FROM tsumego_reports WHERE user_id = ?)`,
      longId, trunc);
    s7MergePairs.push(...shared.map(r => [trunc, r.problem]));
  }
  if (APPLY && longs.length) {
    run(`UPDATE tsumego_reports SET user_id = substr(user_id, 1, 8)
          WHERE length(user_id) != 8 AND user_id NOT IN ('${SMOKE_USERS.join("','")}')`);
  }
  console.log(`S7b: ${s7MergePairs.length} (user, problem) pairs unified by the account merge -> S5`);
}

// ------------------------------------------------- S1: duplicate rows
{
  const n = count(
    `SELECT COUNT(*) FROM tsumego_reports WHERE id NOT IN (
       SELECT MIN(id) FROM tsumego_reports
       GROUP BY user_id, problem, created_at, success, move, time_used, review_time, is_repeat, size)`);
  console.log(`S1: identical-content duplicate rows: ${n} (2026-07-20 measurement: 7154)`);
  if (APPLY && n) {
    run(`DELETE FROM tsumego_reports WHERE id NOT IN (
           SELECT MIN(id) FROM tsumego_reports
           GROUP BY user_id, problem, created_at, success, move, time_used, review_time, is_repeat, size)`);
  }
}

// ------------------------------------------------- S3: unrecoverable moves
{
  const n = count(
    `SELECT COUNT(*) FROM tsumego_reports
      WHERE created_at < ? AND is_repeat = 1 AND success = 0 AND move IS NOT NULL`,
    MOVE_NULL_CUTOFF);
  console.log(`S3: pre-era failed-repeat moves to NULL: ${n} (measurement: 4086)`);
  if (APPLY && n) {
    run(`UPDATE tsumego_reports SET move = NULL
          WHERE created_at < ? AND is_repeat = 1 AND success = 0 AND move IS NOT NULL`,
      MOVE_NULL_CUTOFF);
  }
}

// ------------------------------------------------- S4: problem id rewrite
let s4MergePairs = [];
{
  db.exec('DROP TABLE IF EXISTS temp.id_map');
  db.exec('CREATE TEMP TABLE id_map (old TEXT PRIMARY KEY, new TEXT NOT NULL)');
  const ins = db.prepare('INSERT INTO temp.id_map (old, new) VALUES (?, ?)');
  // Quarantined ids (C0 collisions) have NO final id: their rows keep the
  // retired id (rating-less, the removed-problem precedent) — excluded from
  // the rewrite map.
  for (const e of idMap.entries) {
    if (e.action !== 'quarantine') ins.run(e.old_id, e.new_id);
  }

  // Merge-affected (user, final problem) pairs — captured BEFORE the rewrite:
  // a user with rows on >1 distinct ORIGINAL ids that land on one final id.
  s4MergePairs = all(
    `WITH mapped AS (
       SELECT r.user_id, COALESCE(m.new, r.problem) AS fp, r.problem AS orig
         FROM tsumego_reports r LEFT JOIN temp.id_map m ON r.problem = m.old)
     SELECT user_id, fp FROM mapped GROUP BY user_id, fp
     HAVING COUNT(DISTINCT orig) > 1`).map(r => [r.user_id, r.fp]);

  const n = count(
    `SELECT COUNT(*) FROM tsumego_reports WHERE problem IN (SELECT old FROM temp.id_map)`);
  console.log(`S4: rows on retired ids to rewrite: ${n} (measurement: 31850 misnamed + 10808 dup-member, overlapping)`);
  console.log(`S4: ${s4MergePairs.length} merge-affected (user, problem) pairs -> S5`);
  if (APPLY && n) {
    run(`UPDATE tsumego_reports
            SET problem = (SELECT new FROM temp.id_map WHERE old = problem)
          WHERE problem IN (SELECT old FROM temp.id_map)`);
  }
}

// ------------------------------------------------- S5: repeat-flag repair
{
  // s7MergePairs were captured pre-S4 with ORIGINAL problem ids; map them
  // to their final ids so the row queries below (post-rewrite) match.
  const finalId = new Map(idMap.entries
    .filter((e) => e.action !== 'quarantine')
    .map((e) => [e.old_id, e.new_id]));
  const pairs = new Map();
  for (const [u, p0] of [...s4MergePairs, ...s7MergePairs.map(([u, p]) => [u, finalId.get(p) ?? p])]) {
    pairs.set(`${u}\x00${p0}`, [u, p0]);
  }
  // Rows flipped here were RECORDED as non-repeats (served at identity
  // orientation → canonical move frame), so their moves stay valid; the S3
  // post-check excludes them by id. (Dry-run note: without S4's rewrite the
  // pair query below misses loser rows still under old ids, so the dry-run
  // flip count undercounts — the apply count is the real one.)
  db.exec('DROP TABLE IF EXISTS temp.s5_flipped');
  db.exec('CREATE TEMP TABLE s5_flipped (id INTEGER PRIMARY KEY)');
  const noteFlip = db.prepare('INSERT OR IGNORE INTO temp.s5_flipped (id) VALUES (?)');
  let flipped = 0;
  const sel = db.prepare(
    `SELECT id, is_repeat FROM tsumego_reports
      WHERE user_id = ? AND problem = ? ORDER BY created_at, id`);
  const flip = db.prepare(`UPDATE tsumego_reports SET is_repeat = 1 WHERE id = ?`);
  for (const [u, p] of pairs.values()) {
    const rows = sel.all(u, p);
    for (let i = 1; i < rows.length; i++) {
      if (Number(rows[i].is_repeat) === 0) {
        if (APPLY) flip.run(rows[i].id);
        noteFlip.run(rows[i].id);
        flipped++;
      }
    }
  }
  console.log(`S5: later is_repeat=0 rows flipped to 1 across ${pairs.size} pairs: ${flipped}`);
}

// ------------------------------------------------- S6: recall_schedule
{
  // Quarantined ids first: the problem left the corpus entirely, so a
  // schedule row for it would make the cron push reminders for a problem
  // that 404s — delete outright (zero rows today, cheap).
  const qIds = idMap.entries.filter((e) => e.action === 'quarantine').map((e) => e.old_id);
  if (qIds.length) {
    const ph = qIds.map(() => '?').join(',');
    const nq = count(`SELECT COUNT(*) FROM recall_schedule WHERE problem_id IN (${ph})`, ...qIds);
    console.log(`S6: recall_schedule rows on quarantined ids: ${nq}`);
    if (APPLY && nq) run(`DELETE FROM recall_schedule WHERE problem_id IN (${ph})`, ...qIds);
  }
  const affected = all(
    `SELECT rs.user_id, rs.problem_id, m.new, rs.updated_at
       FROM recall_schedule rs JOIN temp.id_map m ON rs.problem_id = m.old`);
  console.log(`S6: recall_schedule rows on retired ids: ${affected.length} (measurement: 15)`);
  if (APPLY) {
    for (const r of affected) {
      const clash = one(
        `SELECT updated_at FROM recall_schedule WHERE user_id = ? AND problem_id = ?`,
        r.user_id, r.new);
      if (clash) {
        // (user, new) already scheduled: keep the later-updated row.
        if (Number(clash.updated_at) >= Number(r.updated_at)) {
          run(`DELETE FROM recall_schedule WHERE user_id = ? AND problem_id = ?`,
            r.user_id, r.problem_id);
        } else {
          run(`DELETE FROM recall_schedule WHERE user_id = ? AND problem_id = ?`,
            r.user_id, r.new);
          run(`UPDATE recall_schedule SET problem_id = ? WHERE user_id = ? AND problem_id = ?`,
            r.new, r.user_id, r.problem_id);
        }
      } else {
        run(`UPDATE recall_schedule SET problem_id = ? WHERE user_id = ? AND problem_id = ?`,
          r.new, r.user_id, r.problem_id);
      }
    }
  }
}

// ------------------------------------------------- S8: orphaned aux rows
{
  const nRs = count(
    `SELECT COUNT(*) FROM recall_schedule WHERE user_id NOT IN (SELECT DISTINCT user_id FROM tsumego_reports)`);
  const nPs = count(
    `SELECT COUNT(*) FROM push_subscriptions WHERE user_id NOT IN (SELECT DISTINCT user_id FROM tsumego_reports)`);
  console.log(`S8: orphaned recall_schedule rows: ${nRs}; orphaned push_subscriptions: ${nPs}`);
  if (APPLY) {
    run(`DELETE FROM recall_schedule WHERE user_id NOT IN (SELECT DISTINCT user_id FROM tsumego_reports)`);
    run(`DELETE FROM push_subscriptions WHERE user_id NOT IN (SELECT DISTINCT user_id FROM tsumego_reports)`);
  }
}

// ------------------------------------------------- S9: leaderboard reset
{
  const n = count(`SELECT COUNT(*) FROM leaderboard`);
  console.log(`S9: leaderboard rows to reset (buffers hold pre-rewrite problem ids): ${n}`);
  if (APPLY) run(`DELETE FROM leaderboard`);
}

// ------------------------------------------------- S2: full materialization
{
  const users = all(`SELECT DISTINCT user_id FROM tsumego_reports`);
  console.log(`S2: materializing digest chains for ${users.length} users ...`);
  if (APPLY) {
    const sel = db.prepare(
      `SELECT id, user_seq, running_sum FROM tsumego_reports WHERE user_id = ? ORDER BY id`);
    const upd = db.prepare(
      `UPDATE tsumego_reports SET user_seq = ?, running_sum = ? WHERE id = ?`);
    let written = 0;
    for (const { user_id } of users) {
      let seq = 0, sum = 0;
      for (const r of sel.all(user_id)) {
        seq += 1;
        if (Number(r.user_seq) !== seq || Number(r.running_sum) !== sum) {
          upd.run(seq, sum, r.id);
          written++;
        }
        sum += rowHash(Number(r.id));
      }
    }
    console.log(`S2: chain values written: ${written}`);
  }
}

if (APPLY) db.exec('COMMIT');

// ------------------------------------------------- post-verification
if (APPLY) {
  console.log('\npost-verification:');
  const checks = [
    ['rows on retired problem ids', `SELECT COUNT(*) FROM tsumego_reports WHERE problem IN (SELECT old FROM temp.id_map)`, 0],
    ['identical-content dup rows', `SELECT COUNT(*) FROM tsumego_reports WHERE id NOT IN (SELECT MIN(id) FROM tsumego_reports GROUP BY user_id, problem, created_at, success, move, time_used, review_time, is_repeat, size)`, 0],
    ['pre-era failed-repeat moves (excl. S5 flips — recorded as non-repeats, so their moves are canonical)', `SELECT COUNT(*) FROM tsumego_reports WHERE created_at < '${MOVE_NULL_CUTOFF}' AND is_repeat = 1 AND success = 0 AND move IS NOT NULL AND id NOT IN (SELECT id FROM temp.s5_flipped)`, 0],
    ['non-8-char user ids', `SELECT COUNT(*) FROM tsumego_reports WHERE length(user_id) != 8`, 0],
    ['non-8-hex problem ids', `SELECT COUNT(*) FROM tsumego_reports WHERE problem NOT GLOB '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'`, 0],
    ['NULL digest cells', `SELECT COUNT(*) FROM tsumego_reports WHERE user_seq IS NULL OR running_sum IS NULL`, 0],
  ];
  let bad = 0;
  for (const [label, sql, expect] of checks) {
    const n = count(sql);
    const ok = n === expect;
    if (!ok) bad++;
    console.log(`  ${ok ? 'OK  ' : 'FAIL'} ${label}: ${n}`);
  }
  // Full chain verification, every user.
  {
    const rows = all(`SELECT user_id, id, user_seq, running_sum FROM tsumego_reports ORDER BY user_id, id`);
    let seq = 0, sum = 0, prevUser = null, chainBad = 0;
    for (const r of rows) {
      if (r.user_id !== prevUser) { prevUser = r.user_id; seq = 0; sum = 0; }
      seq += 1;
      if (Number(r.user_seq) !== seq || Number(r.running_sum) !== sum) chainBad++;
      sum += rowHash(Number(r.id));
    }
    if (chainBad) bad++;
    console.log(`  ${chainBad === 0 ? 'OK  ' : 'FAIL'} digest chains (all ${rows.length} rows): ${chainBad} mismatches`);
  }
  console.log(bad === 0 ? '\npost-verification PASSED' : `\npost-verification: ${bad} FAILURES`);
  if (bad) process.exit(1);
}

// ------------------------------------------------- staging upload emission
if (EMIT_UPLOAD && APPLY) {
  const cols = ['id', 'user_id', 'problem', 'success', 'created_at', 'size',
    'move', 'time_used', 'review_time', 'is_repeat', 'user_seq', 'running_sum'];
  const q = (v) => v === null ? 'NULL'
    : typeof v === 'number' || typeof v === 'bigint' ? String(v)
      : `'${String(v).replace(/'/g, "''")}'`;
  const rows = all(`SELECT ${cols.join(',')} FROM tsumego_reports ORDER BY id`);
  const parts = [];
  parts.push(`-- Staging upload generated by auxiliary/db_surgery.mjs.
-- Apply with: npx wrangler d1 execute tsumego-reports --remote --file=<this file>
-- Then run the catch-up + swap (see docs plan section 4 / the cutover helper).
DROP TABLE IF EXISTS tsumego_reports_new;
CREATE TABLE tsumego_reports_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    problem TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    created_at TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 9,
    move TEXT,
    time_used INTEGER,
    review_time INTEGER,
    is_repeat INTEGER NOT NULL DEFAULT 0,
    user_seq INTEGER,
    running_sum INTEGER
);`);
  for (let i = 0; i < rows.length; i += 200) {
    const chunk = rows.slice(i, i + 200)
      .map(r => `(${cols.map(c => q(r[c])).join(',')})`).join(',\n');
    parts.push(`INSERT INTO tsumego_reports_new (${cols.join(',')}) VALUES\n${chunk};`);
  }
  writeFileSync(EMIT_UPLOAD, parts.join('\n'));
  console.log(`\nstaging upload SQL -> ${EMIT_UPLOAD} (${rows.length} rows)`);

  // Companion statements for the small tables, applied to remote D1 AFTER the
  // staging swap (S8's orphan checks read the swapped-in tsumego_reports).
  // recall_schedule rewrites are the locally-computed post-surgery rows —
  // residual drift during the surgery window (an old bundle re-upserting a
  // retired-id row) is accepted: the epoch rollout makes every device's first
  // schedule upload a replace-all, which rebuilds its rows wholesale, and a
  // stale reminder's ?recall link just falls back to the standing challenge.
  const smallPath = EMIT_UPLOAD.replace(/\.sql$/, '') + '.small_tables.sql';
  const sp = ['-- Small-table statements generated by auxiliary/db_surgery.mjs.',
    '-- Apply AFTER the staging-table swap:',
    '--   npx wrangler d1 execute tsumego-reports --remote --file=<this file>'];
  const qIds2 = idMap.entries.filter((e) => e.action === 'quarantine').map((e) => e.old_id);
  if (qIds2.length) {
    sp.push(`DELETE FROM recall_schedule WHERE problem_id IN (${qIds2.map((x) => `'${x}'`).join(',')});`);
  }
  // Snapshot of the repaired recall_schedule rows for every problem_id the map
  // touches: delete the affected (user, problem) rows remotely, re-insert the
  // locally-repaired versions (collision rule already applied here).
  const mappedOld = idMap.entries.filter((e) => e.action !== 'quarantine').map((e) => e.old_id);
  const mappedNew = [...new Set(idMap.entries.filter((e) => e.action !== 'quarantine').map((e) => e.new_id))];
  const chunk = (arr, n) => arr.length ? Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, (i + 1) * n)) : [];
  for (const c of chunk(mappedOld, 500)) {
    sp.push(`DELETE FROM recall_schedule WHERE problem_id IN (${c.map((x) => `'${x}'`).join(',')});`);
  }
  const repairedRows = all(`SELECT user_id, problem_id, enter_at, exit_at, enter_notified_at, exit_notified_at, updated_at FROM recall_schedule WHERE problem_id IN (${mappedNew.map(() => '?').join(',')})`, ...mappedNew);
  for (const r of repairedRows) {
    sp.push('INSERT OR REPLACE INTO recall_schedule (user_id, problem_id, enter_at, exit_at, enter_notified_at, exit_notified_at, updated_at) VALUES ('
      + [r.user_id, r.problem_id].map((x) => `'${String(x).replace(/'/g, "''")}'`).join(',') + ','
      + [r.enter_at, r.exit_at, r.enter_notified_at, r.exit_notified_at, r.updated_at].map((x) => (x === null ? 'NULL' : String(x))).join(',') + ');');
  }
  // S8 orphans (generic, evaluated against the swapped-in table) + S9 reset.
  sp.push('DELETE FROM recall_schedule WHERE user_id NOT IN (SELECT DISTINCT user_id FROM tsumego_reports);');
  sp.push('DELETE FROM push_subscriptions WHERE user_id NOT IN (SELECT DISTINCT user_id FROM tsumego_reports);');
  sp.push('DELETE FROM leaderboard;');
  writeFileSync(smallPath, sp.join('\n') + '\n');
  console.log(`small-table SQL -> ${smallPath} (${repairedRows.length} repaired recall rows)`);
}

db.close();
