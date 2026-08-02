// Purge tsumegos by id: remove every trace of them from user history.
//
// Runs against a LOCAL SQLite copy of the D1 export (never remote D1 — perform
// locally, verify, then apply the emitted SQL remotely), and imports the
// Worker's exact rowHash from pwa/shared/sync-digest.js so the repaired digest
// chains are byte-identical to what the live INSERT computes.
//
//   node auxiliary/purge_tsumegos.mjs --db attempts/database.sqlite \
//        --ids data/tsumego_audit/bad_terminal_results.txt \
//        [--apply] [--emit-sql surgery/purge.sql] [--emit-corpus-map surgery/purge_map.json]
//
// What it touches, and why:
//   tsumego_reports  — DELETE the attempts. This is the load-bearing part: the
//                      table carries a per-profile digest chain (user_seq =
//                      1-based attempt count, running_sum = Σ rowHash(id) over
//                      STRICTLY EARLIER rows), and deleting a row invalidates
//                      every LATER row of that user. Both are repaired here.
//   recall_schedule  — DELETE the problem's pending reminders (PK user+problem).
//   share_events     — DELETE its share/open analytics rows.
//   leaderboard      — strip the purged ids out of each affected user's
//                      materialized buffer and NULL the cached rank columns.
//                      The buffer is maintained INCREMENTALLY and is never
//                      rebuilt from tsumego_reports, so it has to be edited in
//                      place; NULLing pred_rank is safe because resolveRowRank
//                      recomputes from the buffer whenever the stored rank is
//                      NULL. country/last_active are deliberately preserved
//                      (deleting the row would lose them).
//
// The digest repair is expressed as RANGE updates, not per-row ones: deleting a
// user's rows d1<d2<...<dk shifts every surviving row in (dj, dj+1) by exactly
// (-j, -Σ rowHash(d1..dj)), so one UPDATE per gap replaces one UPDATE per row.
// D1 bills rows WRITTEN either way, so this does not change the write cost — it
// collapses the emitted statement count from ~one-per-surviving-row to
// ~one-per-deleted-row, which is what makes the remote file tractable.
//
// CLIENTS SELF-HEAL — no client-side migration is needed. Deleting server rows
// makes each device's stored digest disagree with the server's on the next
// /report or /sync, which escalates to the full mirror reconcile; its deletion
// pass (mirrorDeletePass) drops local acked rows the server no longer has, from
// both the working set and the IndexedDB tail. That is the designed convergence
// path for exactly this situation; do not add a bespoke one.
//
// This script does NOT touch the SGF corpus or the id-keyed pipeline artifacts.
// Use --emit-corpus-map to write an id map of `quarantine` entries and feed it
// to the existing corpus surgery, which already moves the SGF to
// tsumegos/broken/<size>/ and drops the id from activations, per-cell
// embeddings, and every table-shaped artifact:
//   ./.venv/bin/python3 -m auxiliary.corpus_surgery --map surgery/purge_map.json --apply
//
// Idempotent: a re-run finds zero rows to delete and the verification passes.

import { DatabaseSync } from 'node:sqlite';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const { rowHash, ID_HASH_SQL } = await import(
  join(ROOT, 'pwa', 'shared', 'sync-digest.js'));

// ------------------------------------------------------------------ args
const args = process.argv.slice(2);
const argOf = (name, dflt = null) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : dflt;
};
const DB_PATH = argOf('--db');
const IDS_PATH = argOf('--ids');
const IDS_INLINE = argOf('--id');
const APPLY = args.includes('--apply');
const PURGE_BROKEN = args.includes('--purge-broken-digest');
const EMIT_SQL = argOf('--emit-sql');
const EMIT_MAP = argOf('--emit-corpus-map');
if (!DB_PATH || (!IDS_PATH && !IDS_INLINE)) {
  console.error('usage: node auxiliary/purge_tsumegos.mjs --db <sqlite> (--ids <file> | --id a,b,c) '
    + '[--apply] [--emit-sql <path>] [--emit-corpus-map <path>]');
  process.exit(2);
}

const rawIds = IDS_PATH
  ? readFileSync(IDS_PATH, 'utf8').split('\n')
  : IDS_INLINE.split(',');
const ids = [...new Set(rawIds.map((s) => s.trim().toLowerCase()).filter(Boolean))];
const badFormat = ids.filter((t) => !/^[0-9a-f]{8}$/.test(t));
if (badFormat.length) {
  console.error(`HARD STOP: ${badFormat.length} id(s) are not 8-hex, e.g. ${badFormat.slice(0, 3).join(', ')}`);
  process.exit(1);
}

const db = new DatabaseSync(DB_PATH);
db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;');
const one = (sql, ...p) => db.prepare(sql).get(...p);
const all = (sql, ...p) => db.prepare(sql).all(...p);
const run = (sql, ...p) => db.prepare(sql).run(...p);
const count = (sql, ...p) => Number(Object.values(one(sql, ...p))[0]);
const fail = (msg) => { console.error(`\nHARD STOP: ${msg}`); process.exit(1); };
const sq = (v) => `'${String(v).replace(/'/g, "''")}'`;

console.log(`db:   ${DB_PATH}`);
console.log(`ids:  ${ids.length} tsumego(s) to purge`);
console.log(`mode: ${APPLY ? 'APPLY' : 'DRY RUN'}\n`);

db.exec('CREATE TEMP TABLE purge_id(problem TEXT PRIMARY KEY)');
{
  const ins = db.prepare('INSERT OR IGNORE INTO purge_id VALUES (?)');
  for (const t of ids) ins.run(t);
}

// Every row whose stored (user_seq, running_sum) disagrees with an independent
// recompute. Returned as a Set of row ids so the post-check can prove the purge
// added nothing, rather than demanding a table that is already clean: the live
// table currently carries rows with NULL digest cells that no code path in
// pwa/src explains, and inheriting someone else's breakage must not block a
// purge — nor be silently absorbed into it.
function chainAudit() {
  const bad = new Set();
  const byUser = new Map();
  const rows = all(`SELECT user_id, id, user_seq, running_sum FROM tsumego_reports ORDER BY user_id, id`);
  let seq = 0, sum = 0, prev = null;
  for (const r of rows) {
    if (r.user_id !== prev) { prev = r.user_id; seq = 0; sum = 0; }
    seq += 1;
    if (r.user_seq === null || r.running_sum === null
        || Number(r.user_seq) !== seq || Number(r.running_sum) !== sum) {
      bad.add(Number(r.id));
      byUser.set(r.user_id, (byUser.get(r.user_id) || 0) + 1);
    }
    sum += rowHash(Number(r.id));
  }
  return { bad, byUser, total: rows.length };
}

// ------------------------------------------------- pre: rowHash + chain sanity
const baseline = (() => {
  const probe = all(`SELECT id, ${ID_HASH_SQL} AS h FROM tsumego_reports ORDER BY id DESC LIMIT 5`);
  for (const r of probe) {
    if (rowHash(Number(r.id)) !== Number(r.h)) {
      fail(`rowHash parity: JS ${rowHash(Number(r.id))} != SQL ${r.h} for id ${r.id}`);
    }
  }
  console.log(`pre: rowHash JS<->SQL parity OK (${probe.length} probes)`);

  const audit = chainAudit();
  if (audit.bad.size) {
    console.warn(`pre: WARNING — ${audit.bad.size} row(s) across ${audit.byUser.size} user(s) `
      + `ALREADY have a broken digest chain, before this purge:`);
    for (const [u, n] of [...audit.byUser].slice(0, 10)) console.warn(`       ${u}: ${n} row(s)`);
    console.warn(`       Pre-existing and out of scope here; the purge must not add to it.`);
  } else {
    console.log(`pre: digest chains clean across all ${audit.total.toLocaleString()} rows`);
  }

  // A user we are about to REWRITE must start from a correct chain, or the
  // relative shift would bake the corruption in at new values. That is a hard
  // stop — unless --purge-broken-digest is deleting the offending rows, in which
  // case that user is rebuilt from absolute values instead of shifted.
  if (!PURGE_BROKEN) {
    const affectedBroken = all(
      `SELECT DISTINCT user_id FROM tsumego_reports WHERE problem IN (SELECT problem FROM purge_id)`)
      .map((r) => r.user_id).filter((u) => audit.byUser.has(u));
    if (affectedBroken.length) {
      fail(`pre: ${affectedBroken.length} user(s) whose history this purge rewrites already have a `
        + `broken digest chain (${affectedBroken.slice(0, 5).join(', ')}). Repair those chains `
        + `first, or pass --purge-broken-digest to delete the offending rows.`);
    }
  }
  console.log(`pre: every user this purge touches starts from a verified-correct chain`);
  return audit;
})();

// Rows deleted because their stored digest is absent or wrong. Opt-in: these are
// real attempts, and the two digest columns are recomputable from (user_id, id),
// so deleting is a DATA LOSS choice — the rows also carry NULL move/time_used/
// review_time, which nothing can restore, and the call was made to drop them
// rather than keep a degraded record. Whatever writes them is a separate,
// still-open bug; this only clears what is already in the table.
const brokenIds = PURGE_BROKEN ? [...baseline.bad] : [];
if (PURGE_BROKEN) {
  console.log(`\n--purge-broken-digest: ${brokenIds.length} row(s) across `
    + `${baseline.byUser.size} user(s) will ALSO be deleted`);
  const gone = all(
    `SELECT user_id, COUNT(*) AS n FROM tsumego_reports WHERE user_id IN (${
      [...baseline.byUser.keys()].map(sq).join(',')}) GROUP BY user_id`)
    .filter((r) => Number(r.n) === baseline.byUser.get(r.user_id));
  if (gone.length) {
    console.log(`  of those, ${gone.length} user(s) lose their ENTIRE history: `
      + gone.map((r) => `${r.user_id} (${r.n})`).join(', '));
  }
}

// ------------------------------------------------- rows to delete
// One temp set drives the local apply, the digest planning and the emitted SQL,
// so the three can never disagree about what is going away.
db.exec('CREATE TEMP TABLE purge_row(id INTEGER PRIMARY KEY)');
db.exec(`INSERT OR IGNORE INTO purge_row
           SELECT id FROM tsumego_reports WHERE problem IN (SELECT problem FROM purge_id)`);
if (brokenIds.length) {
  const ins = db.prepare('INSERT OR IGNORE INTO purge_row VALUES (?)');
  for (const id of brokenIds) ins.run(id);
}

// ------------------------------------------------- scope report
const affected = all(
  `SELECT DISTINCT user_id FROM tsumego_reports WHERE id IN (SELECT id FROM purge_row)`
).map((r) => r.user_id);

{
  const rep = one(`SELECT COUNT(*) AS rows, SUM(is_repeat=0) AS novel, SUM(is_repeat=1) AS repeats,
                          COUNT(DISTINCT problem) AS problems
                     FROM tsumego_reports WHERE id IN (SELECT id FROM purge_row)`);
  const rs = count(`SELECT COUNT(*) FROM recall_schedule WHERE problem_id IN (SELECT problem FROM purge_id)`);
  const se = count(`SELECT COUNT(*) FROM share_events WHERE problem_id IN (SELECT problem FROM purge_id)`);
  const total = count(`SELECT COUNT(*) FROM tsumego_reports`);
  console.log(`\nscope:`);
  console.log(`  tsumego_reports rows to delete: ${Number(rep.rows).toLocaleString()} `
    + `(${Number(rep.novel || 0).toLocaleString()} novel, ${Number(rep.repeats || 0).toLocaleString()} repeats)`
    + ` = ${(100 * Number(rep.rows) / total).toFixed(2)}% of the table`);
  console.log(`  purged ids that were ever attempted: ${rep.problems} of ${ids.length}`);
  console.log(`  users whose history changes:         ${affected.length}`);
  console.log(`  recall_schedule rows to delete:      ${rs}`);
  console.log(`  share_events rows to delete:         ${se}`);
}

// ------------------------------------------------- plan the digest repair
// Per user: the ids being deleted, and the resulting (seq, sum) shift applied to
// each surviving range between them. Emitted as range updates; see the header.
const repairPlan = [];   // { user_id, ranges: [{ afterId, beforeId|null, dSeq, dSum }] }
const absUpdates = [];   // { id, seq, sum } — full rebuild for broken-chain users
let rowsRewritten = 0;
{
  const selDel = db.prepare(
    `SELECT id FROM tsumego_reports WHERE user_id = ? AND id IN (SELECT id FROM purge_row) ORDER BY id`);
  const selAfter = db.prepare(
    `SELECT COUNT(*) AS n FROM tsumego_reports WHERE user_id = ? AND id > ?
       AND id NOT IN (SELECT id FROM purge_row)`);
  for (const user_id of affected) {
    // A user whose stored chain is wrong cannot be SHIFTED (the shift is
    // relative to values that are themselves wrong) — rebuild from absolutes.
    if (baseline.byUser.has(user_id)) continue;
    const deleted = selDel.all(user_id).map((r) => Number(r.id));
    const ranges = [];
    let dSeq = 0, dSum = 0;
    for (let i = 0; i < deleted.length; i++) {
      dSeq += 1;
      dSum += rowHash(deleted[i]);
      ranges.push({
        afterId: deleted[i],
        beforeId: i + 1 < deleted.length ? deleted[i + 1] : null,
        dSeq, dSum,
      });
    }
    repairPlan.push({ user_id, ranges });
    rowsRewritten += Number(selAfter.get(user_id, deleted[0]).n);
  }
  // Absolute rematerialization for the broken-chain users: recompute the whole
  // surviving chain from (user_id, id) — the same derivation the INSERT does —
  // and keep only the rows whose stored value actually differs.
  const selKeep = db.prepare(
    `SELECT id, user_seq, running_sum FROM tsumego_reports
      WHERE user_id = ? AND id NOT IN (SELECT id FROM purge_row) ORDER BY id`);
  for (const user_id of baseline.byUser.keys()) {
    let seq = 0, sum = 0;
    for (const r of selKeep.all(user_id)) {
      seq += 1;
      if (r.user_seq === null || r.running_sum === null
          || Number(r.user_seq) !== seq || Number(r.running_sum) !== sum) {
        absUpdates.push({ id: Number(r.id), seq, sum });
      }
      sum += rowHash(Number(r.id));
    }
  }
  if (absUpdates.length) {
    console.log(`  chain rebuild: ${absUpdates.length} surviving row(s) of `
      + `${baseline.byUser.size} previously-broken user(s) get absolute values`);
  }
  console.log(`  digest repair: ${rowsRewritten.toLocaleString()} surviving rows shift, `
    + `expressed as ${repairPlan.reduce((a, p) => a + p.ranges.length, 0).toLocaleString()} range updates`);
}

// ------------------------------------------------- leaderboard buffers
const lbEdits = [];   // { user_id, buffer }
{
  const purged = new Set(ids);
  const rows = all(`SELECT user_id, buffer FROM leaderboard WHERE buffer IS NOT NULL`);
  let scanned = 0, dropped = 0;
  for (const r of rows) {
    let buf;
    try { buf = JSON.parse(r.buffer); } catch { continue; }
    if (!Array.isArray(buf)) continue;
    scanned++;
    const kept = buf.filter((e) => !purged.has(e && e.p));
    if (kept.length !== buf.length) {
      dropped += buf.length - kept.length;
      lbEdits.push({ user_id: r.user_id, buffer: JSON.stringify(kept) });
    }
  }
  console.log(`  leaderboard: ${lbEdits.length} of ${scanned} buffers lose `
    + `${dropped.toLocaleString()} entries (cached rank columns NULLed so they recompute)`);
}

if (!APPLY) {
  console.log('\nDRY RUN — nothing written. Re-run with --apply.');
} else {
  db.exec('BEGIN');
  const inPurge = `(SELECT problem FROM purge_id)`;
  const nRep = run(`DELETE FROM tsumego_reports WHERE id IN (SELECT id FROM purge_row)`).changes;
  const nRs = run(`DELETE FROM recall_schedule WHERE problem_id IN ${inPurge}`).changes;
  const nSe = run(`DELETE FROM share_events WHERE problem_id IN ${inPurge}`).changes;

  const updRange = db.prepare(
    `UPDATE tsumego_reports SET user_seq = user_seq - ?, running_sum = running_sum - ?
      WHERE user_id = ? AND id > ?` + ' AND (? IS NULL OR id < ?)');
  let ranges = 0;
  for (const { user_id, ranges: rs } of repairPlan) {
    for (const g of rs) {
      updRange.run(g.dSeq, g.dSum, user_id, g.afterId, g.beforeId, g.beforeId);
      ranges++;
    }
  }
  const updAbs = db.prepare(
    `UPDATE tsumego_reports SET user_seq = ?, running_sum = ? WHERE id = ?`);
  for (const u of absUpdates) updAbs.run(u.seq, u.sum, u.id);
  const updLb = db.prepare(
    `UPDATE leaderboard SET buffer = ?, pred_rank = NULL, pred_rank_se = NULL,
            time_delta_rating = NULL WHERE user_id = ?`);
  for (const e of lbEdits) updLb.run(e.buffer, e.user_id);
  db.exec('COMMIT');
  console.log(`\napplied: ${nRep} report rows, ${nRs} recall rows, ${nSe} share rows deleted; `
    + `${ranges} digest range updates, ${absUpdates.length} absolute chain rebuilds; `
    + `${lbEdits.length} leaderboard buffers rewritten`);
}

// ------------------------------------------------- post-verification
if (APPLY) {
  console.log('\npost-verification:');
  let bad = 0;
  const checks = [
    ['reports on purged ids', `SELECT COUNT(*) FROM tsumego_reports WHERE problem IN (SELECT problem FROM purge_id)`, 0],
    ['recall rows on purged ids', `SELECT COUNT(*) FROM recall_schedule WHERE problem_id IN (SELECT problem FROM purge_id)`, 0],
    ['share rows on purged ids', `SELECT COUNT(*) FROM share_events WHERE problem_id IN (SELECT problem FROM purge_id)`, 0],
  ];
  for (const [label, sql, expect] of checks) {
    const n = count(sql);
    if (n !== expect) bad++;
    console.log(`  ${n === expect ? 'OK  ' : 'FAIL'} ${label}: ${n}`);
  }
  // Full independent recompute of every chain — not just the touched users, so a
  // range update that overshot into a neighbour is caught. Judged against the
  // pre-existing breakage: the purge must introduce no NEW broken row (and the
  // rows it deleted may legitimately leave the bad set).
  {
    const after = chainAudit();
    // With --purge-broken-digest the offending rows are gone and their users were
    // rebuilt, so the bar is ZERO broken rows. Otherwise the bar is "no NEW ones".
    const added = [...after.bad].filter((id) => !baseline.bad.has(id));
    const failed = PURGE_BROKEN ? after.bad.size > 0 : added.length > 0;
    if (failed) bad++;
    console.log(`  ${failed ? 'FAIL' : 'OK  '} digest chains (all `
      + `${after.total.toLocaleString()} rows): ${after.bad.size} broken`
      + (PURGE_BROKEN ? ' (must be 0)' : `, ${added.length} of them NEW (must be 0)`));
    if (failed) console.log(`       first broken row ids: ${[...after.bad].slice(0, 10).join(', ')}`);
  }
  // No leaderboard buffer may still name a purged id.
  {
    const purged = new Set(ids);
    let stale = 0;
    for (const r of all(`SELECT buffer FROM leaderboard WHERE buffer IS NOT NULL`)) {
      let buf; try { buf = JSON.parse(r.buffer); } catch { continue; }
      if (Array.isArray(buf) && buf.some((e) => purged.has(e && e.p))) stale++;
    }
    if (stale) bad++;
    console.log(`  ${stale === 0 ? 'OK  ' : 'FAIL'} leaderboard buffers naming a purged id: ${stale}`);
  }
  console.log(bad === 0 ? '\npost-verification PASSED' : `\npost-verification: ${bad} FAILURES`);
  if (bad) process.exit(1);
}

// ------------------------------------------------- remote SQL emission
if (EMIT_SQL) {
  const chunk = (arr, n) => Array.from({ length: Math.ceil(arr.length / n) }, (_, i) => arr.slice(i * n, (i + 1) * n));
  const parts = [
    `-- Purge of ${ids.length} tsumego id(s), generated by auxiliary/purge_tsumegos.mjs.`,
    `-- Apply with: npx wrangler d1 execute tsumego-reports --remote --file=<this file>`,
    `--`,
    `-- ORDER MATTERS: the digest shifts below are relative to the CURRENT stored`,
    `-- values, so they must run exactly once, after the DELETEs, against the same`,
    `-- snapshot this file was generated from. Re-run pipeline.fetch_database and`,
    `-- regenerate if new attempts have landed on a purged id in the meantime.`,
  ];
  for (const c of chunk(ids, 400)) {
    const list = c.map(sq).join(',');
    parts.push(`DELETE FROM tsumego_reports WHERE problem IN (${list});`);
    parts.push(`DELETE FROM recall_schedule WHERE problem_id IN (${list});`);
    parts.push(`DELETE FROM share_events WHERE problem_id IN (${list});`);
  }
  for (const c of chunk(brokenIds, 400)) {
    parts.push(`DELETE FROM tsumego_reports WHERE id IN (${c.join(',')});`);
  }
  for (const { user_id, ranges } of repairPlan) {
    for (const g of ranges) {
      const bound = g.beforeId === null ? '' : ` AND id < ${g.beforeId}`;
      parts.push(`UPDATE tsumego_reports SET user_seq = user_seq - ${g.dSeq}, `
        + `running_sum = running_sum - ${g.dSum} `
        + `WHERE user_id = ${sq(user_id)} AND id > ${g.afterId}${bound};`);
    }
  }
  for (const u of absUpdates) {
    parts.push(`UPDATE tsumego_reports SET user_seq = ${u.seq}, running_sum = ${u.sum} `
      + `WHERE id = ${u.id};`);
  }
  for (const e of lbEdits) {
    parts.push(`UPDATE leaderboard SET buffer = ${sq(e.buffer)}, pred_rank = NULL, `
      + `pred_rank_se = NULL, time_delta_rating = NULL WHERE user_id = ${sq(e.user_id)};`);
  }
  mkdirSync(dirname(EMIT_SQL), { recursive: true });
  writeFileSync(EMIT_SQL, parts.join('\n') + '\n');
  console.log(`\nremote SQL -> ${EMIT_SQL} (${parts.length - 7} statements)`);
}

// ------------------------------------------------- corpus id map emission
if (EMIT_MAP) {
  const entries = [];
  let missing = 0;
  for (const t of ids) {
    let size = null;
    for (const s of [9, 13]) {
      if (existsSync(join(ROOT, 'tsumegos', String(s), `${t}.sgf`))) { size = s; break; }
    }
    if (size === null) { missing++; continue; }
    entries.push({ size, old_id: t, new_id: null, action: 'quarantine' });
  }
  const map = { renames: 0, merges: 0, quarantines: entries.length, entries };
  mkdirSync(dirname(EMIT_MAP), { recursive: true });
  writeFileSync(EMIT_MAP, JSON.stringify(map, null, 2));
  console.log(`corpus id map -> ${EMIT_MAP} (${entries.length} quarantine entries`
    + (missing ? `, ${missing} id(s) had no SGF and were skipped` : '') + ')');
  console.log(`  apply with: ./.venv/bin/python3 -m auxiliary.corpus_surgery --map ${EMIT_MAP} --apply`);
}

db.close();
