// Rebuild every broken per-profile digest chain in place, deleting nothing.
//
// Runs against a LOCAL SQLite copy of the D1 export, then the emitted SQL is
// applied remotely (docs/db_surgery_runbook.md: freeze, snapshot, run, apply,
// verify, thaw). The chain is user_seq = 1-based count and running_sum = Σ
// rowHash over STRICTLY EARLIER rows, both in id order, with rowHash imported
// from pwa/shared/sync-digest.js so the values match the live INSERT exactly.
// Every row whose stored pair disagrees — a NULL left by a writer that never
// maintained the chain, or a value shifted against the wrong base — is written
// its ABSOLUTE value, so the emitted file is safe to apply twice.
//
// This is the non-destructive alternative to purge_tsumegos.mjs
// --purge-broken-digest, for rows that are real attempts and only lack their
// digest. Clients self-heal: a repaired account's devices see the server digest
// change on their next /report or /sync and run the full mirror reconcile.
//
//   node auxiliary/repair_digest_chains.mjs --db db/d1.sqlite [--apply] [--emit-sql surgery/repair.sql]

import { DatabaseSync } from 'node:sqlite';
import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const { rowHash } = await import(join(ROOT, 'pwa', 'shared', 'sync-digest.js'));

const args = process.argv.slice(2);
const argOf = (name) => { const i = args.indexOf(name); return i >= 0 ? args[i + 1] : null; };
const DB_PATH = argOf('--db');
const EMIT_SQL = argOf('--emit-sql');
const APPLY = args.includes('--apply');
if (!DB_PATH) {
  console.error('usage: node auxiliary/repair_digest_chains.mjs --db <sqlite> [--apply] [--emit-sql <path>]');
  process.exit(2);
}

const db = new DatabaseSync(DB_PATH);
const audit = () => {
  const fixes = [];
  const users = new Map();
  let prev = null, seq = 0, sum = 0;
  for (const r of db.prepare(`SELECT user_id, id, user_seq, running_sum FROM tsumego_reports ORDER BY user_id, id`).iterate()) {
    if (r.user_id !== prev) { prev = r.user_id; seq = 0; sum = 0; }
    seq += 1;
    if (r.user_seq === null || r.running_sum === null || Number(r.user_seq) !== seq || Number(r.running_sum) !== sum) {
      fixes.push({ id: Number(r.id), seq, sum, nulled: r.user_seq === null });
      users.set(r.user_id, (users.get(r.user_id) || 0) + 1);
    }
    sum += rowHash(Number(r.id));
  }
  return { fixes, users };
};

console.log(`db:   ${DB_PATH}`);
console.log(`mode: ${APPLY ? 'APPLY' : 'DRY RUN'}`);
const { fixes, users } = audit();
console.log(`broken: ${fixes.length} row(s) across ${users.size} account(s), `
  + `${fixes.filter((f) => f.nulled).length} of them NULL`);
for (const [u, n] of users) console.log(`  ${u}  ${n}`);

db.exec('BEGIN');
const fix = db.prepare(`UPDATE tsumego_reports SET user_seq = ?, running_sum = ? WHERE id = ?`);
for (const f of fixes) fix.run(f.seq, f.sum, f.id);
const after = audit();
if (after.fixes.length) {
  db.exec('ROLLBACK');
  console.error(`\nHARD STOP: ${after.fixes.length} row(s) still broken after the repair`);
  process.exit(1);
}
const tot = db.prepare(`SELECT COUNT(*) AS rows, CAST(SUM(user_seq) AS TEXT) AS seq_sum, `
  + `CAST(SUM(running_sum) AS TEXT) AS sum_sum FROM tsumego_reports`).get();
console.log('post table:', { ...tot });
console.log('post-verification PASSED');

if (EMIT_SQL) {
  const lines = fixes.map((f) => `UPDATE tsumego_reports SET user_seq = ${f.seq}, running_sum = ${f.sum} WHERE id = ${f.id};`);
  mkdirSync(dirname(EMIT_SQL), { recursive: true });
  writeFileSync(EMIT_SQL, lines.join('\n') + '\n');
  console.log(`emitted ${lines.length} statement(s) -> ${EMIT_SQL}`);
}
if (APPLY) { db.exec('COMMIT'); console.log('APPLIED to local db'); }
else { db.exec('ROLLBACK'); console.log('dry run: rolled back'); }
