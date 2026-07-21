// Remote-D1 cutover driver for the DB surgery (docs plan section 4).
//
// Applies the locally-verified surgery to the live database via a staging
// table, phase by phase — run each phase explicitly, in order, during a
// low-traffic window (no write freeze needed; the outbox tolerates a failed
// report during the swap instant):
//
//   node auxiliary/db_surgery_cutover.mjs --phase upload    # create + fill tsumego_reports_new
//   node auxiliary/db_surgery_cutover.mjs --phase catchup   # copy live rows > staged max (repeat until "quiescent")
//   node auxiliary/db_surgery_cutover.mjs --phase swap      # drop old, rename staging, recreate indexes
//   node auxiliary/db_surgery_cutover.mjs --phase small     # small-table statements (S6/S8/S9)
//   node auxiliary/db_surgery_cutover.mjs --phase verify    # row count + digest spot-checks
//
// Inputs: surgery/staging_upload.sql + surgery/staging_upload.small_tables.sql
// (from db_surgery.mjs --apply --emit-upload). Catch-up recomputes each copied
// row's digest chain AGAINST THE STAGING TABLE (the live Worker seeded it
// against the un-repaired table, so window-era rows of surgery-affected users
// would otherwise carry inconsistent chains — plan section 4). Copied rows
// keep their live ids; the chain-continuation subqueries run inside one D1
// batch in id order, so each row sees its predecessors.
//
// All wrangler calls go to api.cloudflare.com (not the WAF-guarded zone).

import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PWA_DIR = join(HERE, '..', 'pwa');
const UPLOAD_SQL = join(HERE, '..', 'surgery', 'staging_upload.sql');
const SMALL_SQL = join(HERE, '..', 'surgery', 'staging_upload.small_tables.sql');
// Keep in lockstep with shared/sync-digest.js ID_HASH_SQL, expressed over the
// live table's column for the catch-up INSERT...SELECT.
const ID_HASH_SQL_NEW = '((id*2654435761)%2147483647)';

const phase = (() => {
  const i = process.argv.indexOf('--phase');
  return i >= 0 ? process.argv[i + 1] : null;
})();

function wrangler(args) {
  return execFileSync('npx', ['wrangler', 'd1', 'execute', 'tsumego-reports', '--remote', ...args],
    { cwd: PWA_DIR, encoding: 'utf8', maxBuffer: 1024 * 1024 * 256 });
}

function queryJson(command) {
  const out = wrangler(['--json', '--command', command]);
  const parsed = JSON.parse(out);
  return parsed[0].results;
}

async function main() {
  if (phase === 'upload') {
    if (!existsSync(UPLOAD_SQL)) throw new Error(`missing ${UPLOAD_SQL} — run db_surgery.mjs --apply --emit-upload first`);
    console.log('uploading staging table (this can take a while) ...');
    wrangler(['--file', UPLOAD_SQL]);
    const [{ n }] = queryJson('SELECT COUNT(*) AS n FROM tsumego_reports_new');
    console.log(`staging table filled: ${n} rows`);
    return;
  }

  if (phase === 'catchup') {
    const [{ m: stagedMax }] = queryJson('SELECT COALESCE(MAX(id),0) AS m FROM tsumego_reports_new');
    const rows = queryJson(
      `SELECT id, user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat
         FROM tsumego_reports WHERE id > ${stagedMax} ORDER BY id`);
    if (rows.length === 0) {
      console.log(`quiescent: staging is caught up to live (max id ${stagedMax}). Proceed to --phase swap.`);
      return;
    }
    console.log(`catch-up: copying ${rows.length} live rows > ${stagedMax} with chains recomputed against staging ...`);
    const q = (v) => v === null || v === undefined ? 'NULL'
      : typeof v === 'number' ? String(v) : `'${String(v).replace(/'/g, "''")}'`;
    const stmts = rows.map((r) => {
      const uid = q(r.user_id);
      return `INSERT INTO tsumego_reports_new (id, user_id, problem, success, created_at, size, move, time_used, review_time, is_repeat, user_seq, running_sum)
        SELECT ${q(r.id)}, ${uid}, ${q(r.problem)}, ${q(r.success)}, ${q(r.created_at)}, ${q(r.size)}, ${q(r.move)}, ${q(r.time_used)}, ${q(r.review_time)}, ${q(r.is_repeat)},
          COALESCE((SELECT user_seq FROM tsumego_reports_new WHERE user_id = ${uid} ORDER BY id DESC LIMIT 1), 0) + 1,
          COALESCE((SELECT running_sum + ${ID_HASH_SQL_NEW} FROM tsumego_reports_new WHERE user_id = ${uid} ORDER BY id DESC LIMIT 1), 0);`;
    });
    wrangler(['--command', stmts.join('\n')]);
    console.log(`copied ${rows.length} rows. Re-run --phase catchup until quiescent.`);
    return;
  }

  if (phase === 'swap') {
    const [{ a }] = queryJson('SELECT COALESCE(MAX(id),0) AS a FROM tsumego_reports');
    const [{ b }] = queryJson('SELECT COALESCE(MAX(id),0) AS b FROM tsumego_reports_new');
    if (a !== b) throw new Error(`NOT quiescent (live max ${a} vs staged ${b}) — run --phase catchup first`);
    console.log('swapping ...');
    wrangler(['--command', [
      'DROP TABLE tsumego_reports;',
      'ALTER TABLE tsumego_reports_new RENAME TO tsumego_reports;',
      'CREATE INDEX IF NOT EXISTS idx_tsumego_reports_created ON tsumego_reports(created_at);',
      'CREATE INDEX IF NOT EXISTS idx_tsumego_reports_user_id ON tsumego_reports(user_id, id);',
    ].join('\n')]);
    console.log('swap complete.');
    return;
  }

  if (phase === 'small') {
    if (!existsSync(SMALL_SQL)) throw new Error(`missing ${SMALL_SQL}`);
    wrangler(['--file', SMALL_SQL]);
    console.log('small-table statements applied.');
    return;
  }

  if (phase === 'verify') {
    const [{ n }] = queryJson('SELECT COUNT(*) AS n FROM tsumego_reports');
    const [{ d }] = queryJson('SELECT COUNT(*) AS d FROM tsumego_reports WHERE user_seq IS NULL OR running_sum IS NULL');
    const [{ l }] = queryJson("SELECT COUNT(*) AS l FROM tsumego_reports WHERE length(user_id) != 8");
    const seq = queryJson("SELECT COUNT(*) AS c FROM sqlite_master WHERE type='table' AND name='tsumego_reports_new'");
    console.log(`rows: ${n}; NULL digest cells: ${d}; non-8-char user ids: ${l}; staging table still present: ${seq[0].c}`);
    console.log('(full post-conditions: pipeline.fetch_database --full, then re-run the section-2 queries locally)');
    return;
  }

  throw new Error('usage: --phase upload|catchup|swap|small|verify');
}

main().catch((e) => { console.error(e.message || e); process.exit(1); });
