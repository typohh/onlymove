// Move attempt rows from one account to another: sessions someone else played
// on a borrowed account code become that player's own account.
//
// Runs against a LOCAL SQLite copy of the D1 export (never remote D1 — perform
// locally, verify, then apply the emitted SQL remotely), and imports the
// Worker's exact rowHash from pwa/shared/sync-digest.js so the rebuilt digest
// chains are byte-identical to what the live INSERT computes.
//
//   node auxiliary/move_attempts.mjs --db db/d1.sqlite --from <userId> \
//        [--to <userId>] --window <isoStart>/<isoEnd> [--window …] \
//        --expect <n> [--apply] [--emit-sql surgery/move.sql]
//
// Rows are selected by (source user, created_at inside a window, inclusive),
// not by a row-id list, so the selection is re-derived from whatever snapshot
// the run is given — a late outbox flush carrying a historical created_at into
// a window is picked up rather than left behind. --expect is the guard that
// the snapshot holds exactly the rows the windows were drawn around. --to
// defaults to a freshly generated code (same alphabet and length as the
// client's generateUserId) that must not appear in any table.
//
// What it touches:
//   tsumego_reports — UPDATE user_id on the selected rows; ids are kept, so
//                     nothing else keyed by row id moves. Both accounts' digest
//                     chains (user_seq = 1-based count, running_sum = Σ rowHash
//                     over STRICTLY EARLIER rows, both in id order) are rewritten
//                     as ABSOLUTE values for every row whose stored pair
//                     changes, so the emitted file is safe to apply twice.
// It leaves the leaderboard alone: run auxiliary/update_leaderboard_ranks.mjs
// inside the same freeze, which rebuilds the source's buffer and monthly peaks
// and seeds the destination's row. rank_surveys, external_ids and
// recall_schedule stay with the source account (the recall schedule is
// re-uploaded by the source's device once it has reconciled).
//
// CLIENTS SELF-HEAL. The source's devices see a digest disagreement on their
// next /report or /sync and run the full mirror reconcile, whose deletion pass
// drops the moved rows; a device that enters the destination code adopts it
// through the ordinary account switch and pulls them.

import { DatabaseSync } from 'node:sqlite';
import { writeFileSync, mkdirSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const { rowHash } = await import(join(ROOT, 'pwa', 'shared', 'sync-digest.js'));

const args = process.argv.slice(2);
const argOf = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const argsOf = (name) => args.flatMap((a, i) => (a === name ? [args[i + 1]] : []));
const DB_PATH = argOf('--db');
const FROM = argOf('--from');
let TO = argOf('--to');
const WINDOWS = argsOf('--window').map((w) => w.split('/'));
const EXPECT = argOf('--expect');
const APPLY = args.includes('--apply');
const EMIT_SQL = argOf('--emit-sql');
const fail = (msg) => { console.error(`\nHARD STOP: ${msg}`); process.exit(1); };
if (!DB_PATH || !FROM || !WINDOWS.length || EXPECT === null) {
  console.error('usage: node auxiliary/move_attempts.mjs --db <sqlite> --from <userId> [--to <userId>] '
    + '--window <isoStart>/<isoEnd> [...] --expect <n> [--apply] [--emit-sql <path>]');
  process.exit(2);
}
const USER_ID_RE = /^[A-Za-z0-9_-]{8}$/;
const ISO_RE = /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/;
for (const w of WINDOWS) {
  if (w.length !== 2 || !ISO_RE.test(w[0]) || !ISO_RE.test(w[1]) || w[0] > w[1]) fail(`bad window ${w.join('/')}`);
}

const db = new DatabaseSync(DB_PATH);
const all = (sql, ...p) => db.prepare(sql).all(...p);
const one = (sql, ...p) => db.prepare(sql).get(...p);
const sq = (v) => `'${String(v).replace(/'/g, "''")}'`;

// Every table carrying a user_id column: the destination must be unused in all.
const userTables = all(`SELECT name FROM sqlite_master WHERE type = 'table'`)
  .map((r) => r.name)
  .filter((t) => all(`PRAGMA table_info(${sq(t)})`).some((c) => c.name === 'user_id'));

function generateUserId() {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let id = '';
  while (id.length < 8) {
    for (const v of webcrypto.getRandomValues(new Uint8Array(16))) {
      if (id.length < 8 && v < 252) id += alphabet[v % 36];
    }
  }
  return id;
}
const usedAnywhere = (uid) => userTables.some((t) => one(`SELECT 1 AS x FROM ${t} WHERE user_id = ? LIMIT 1`, uid));
if (!TO) { do { TO = generateUserId(); } while (usedAnywhere(TO)); }
if (!USER_ID_RE.test(FROM) || !USER_ID_RE.test(TO) || FROM === TO) fail('bad --from/--to');

// Chain audit for one user: rows whose stored pair disagrees with a recompute.
function chain(uid) {
  let seq = 0, sum = 0;
  return all(`SELECT id, user_seq, running_sum FROM tsumego_reports WHERE user_id = ? ORDER BY id`, uid)
    .map((r) => {
      seq += 1;
      const out = { id: Number(r.id), seq, sum, storedSeq: r.user_seq, storedSum: r.running_sum };
      sum += rowHash(Number(r.id));
      return out;
    });
}
const broken = (uid) => chain(uid).filter((r) => r.storedSeq === null || Number(r.storedSeq) !== r.seq
  || Number(r.storedSum) !== r.sum);

const windowSql = WINDOWS.map(() => '(created_at >= ? AND created_at <= ?)').join(' OR ');
const windowArgs = WINDOWS.flat();
const selected = all(`SELECT id, created_at FROM tsumego_reports WHERE user_id = ? AND (${windowSql}) ORDER BY id`,
  FROM, ...windowArgs).map((r) => Number(r.id));

console.log(`db:     ${DB_PATH}`);
console.log(`from:   ${FROM}   to: ${TO}${usedAnywhere(TO) ? '  (EXISTING account)' : '  (new)'}`);
console.log(`mode:   ${APPLY ? 'APPLY' : 'DRY RUN'}`);
for (const [a, b] of WINDOWS) {
  const n = one(`SELECT COUNT(*) AS n FROM tsumego_reports WHERE user_id = ? AND created_at >= ? AND created_at <= ?`, FROM, a, b).n;
  console.log(`window  ${a} .. ${b}  ${n} rows`);
}
console.log(`selected ${selected.length} rows (expected ${EXPECT})`);

if (selected.length === 0 && Number(one(`SELECT COUNT(*) AS n FROM tsumego_reports WHERE user_id = ? AND (${windowSql})`,
  TO, ...windowArgs).n) === Number(EXPECT)) {
  console.log('already moved: the destination holds the expected rows in these windows');
}
else if (selected.length !== Number(EXPECT)) fail(`selected ${selected.length}, expected ${EXPECT}`);

const preBroken = [FROM, TO].flatMap(broken);
if (preBroken.length) fail(`${preBroken.length} row(s) already break a digest chain before the move`);

db.exec('BEGIN');
const upd = db.prepare(`UPDATE tsumego_reports SET user_id = ? WHERE id = ? AND user_id = ?`);
for (const id of selected) upd.run(TO, id, FROM);
const fixes = [FROM, TO].flatMap((uid) => broken(uid));
const fix = db.prepare(`UPDATE tsumego_reports SET user_seq = ?, running_sum = ? WHERE id = ?`);
for (const r of fixes) fix.run(r.seq, r.sum, r.id);

const post = [FROM, TO].flatMap(broken);
if (post.length) { db.exec('ROLLBACK'); fail(`post-verification: ${post.length} broken row(s)`); }
const digest = (uid) => {
  const r = one(`SELECT COUNT(*) AS n, MAX(user_seq) AS seq, MAX(id) AS maxId FROM tsumego_reports WHERE user_id = ?`, uid);
  const c = chain(uid);
  const last = c[c.length - 1];
  return { rows: r.n, userSeq: r.seq, maxId: r.maxId, digest: last ? last.sum + rowHash(last.id) : 0 };
};
console.log(`chain rows rewritten: ${fixes.length}`);
console.log(`post ${FROM}:`, digest(FROM));
console.log(`post ${TO}:`, digest(TO));
const tot = one(`SELECT COUNT(*) AS rows, CAST(SUM(user_seq) AS TEXT) AS seq_sum, CAST(SUM(running_sum) AS TEXT) AS sum_sum FROM tsumego_reports`);
console.log('post table:', tot);
console.log('post-verification PASSED');

if (EMIT_SQL) {
  const lines = [];
  if (selected.length) {
    lines.push(`UPDATE tsumego_reports SET user_id = ${sq(TO)} WHERE user_id = ${sq(FROM)} AND id IN (${selected.join(',')});`);
  }
  for (const r of fixes) lines.push(`UPDATE tsumego_reports SET user_seq = ${r.seq}, running_sum = ${r.sum} WHERE id = ${r.id};`);
  mkdirSync(dirname(EMIT_SQL), { recursive: true });
  writeFileSync(EMIT_SQL, lines.join('\n') + '\n');
  console.log(`emitted ${lines.length} statement(s) -> ${EMIT_SQL}`);
}
if (APPLY) { db.exec('COMMIT'); console.log('APPLIED to local db'); }
else { db.exec('ROLLBACK'); console.log('dry run: rolled back'); }
