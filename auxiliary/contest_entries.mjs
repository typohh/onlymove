#!/usr/bin/env node
// auxiliary/contest_entries.mjs
//
// The organiser's tool for the /contest page's entry lists (contest_entries in
// pwa/schema.sql). The page itself only reads; every entry is added and edited
// here, through `wrangler d1 execute`. Each write also drops the page's cached
// blob (server_stats key 'contest'), so the next view of /contest recomputes
// instead of waiting out the 12-hour refresh.
//
// An entry names an ACCOUNT, never a user id typed by hand: a strong player is
// found through the EGD/AGA record they linked in the app (by PIN, AGA id or the
// linked display name), a video entrant through the leaderboard alias they picked
// on the account they played. The user id is the account credential, so this
// script prints none.
//
// Usage (from anywhere; one of --remote / --local is required):
//   node auxiliary/contest_entries.mjs list                                   --remote
//   node auxiliary/contest_entries.mjs add-strength --egd 12345678            --remote
//   node auxiliary/contest_entries.mjs add-strength --aga 12345               --remote
//   node auxiliary/contest_entries.mjs add-strength --name "Henry Hemming" --reference --remote
//   node auxiliary/contest_entries.mjs add-video --alias "Brave Crimson Fox" \
//        --url https://youtu.be/... --problems 60 [--views 120] [--quality 7.5] --remote
//   node auxiliary/contest_entries.mjs set --id 3 [--views N] [--quality Q] [--problems N] [--url U] [--date YYYY-MM-DD] --remote
//   node auxiliary/contest_entries.mjs remove --id 3                          --remote
//   node auxiliary/contest_entries.mjs refresh                                --remote
//
// `--date YYYY-MM-DD` on an add sets the entry date (default: now, UTC). The entry
// date is the day the entry appears on the public list — the date the rules
// favour — so set it only to correct a mistake. `--quality` is on a 0–10 scale.

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PWA_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'pwa');
const DB_NAME = 'tsumego-reports';

const argv = process.argv.slice(2);
const cmd = argv[0];
const opts = {};
for (let i = 1; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) die(`unexpected argument: ${a}`);
    const key = a.slice(2);
    if (['remote', 'local', 'reference'].includes(key)) opts[key] = true;
    else {
        if (i + 1 >= argv.length) die(`${a} needs a value`);
        opts[key] = argv[++i];
    }
}

function die(msg) {
    console.error(`contest_entries: ${msg}`);
    process.exit(1);
}

if (!!opts.remote === !!opts.local) die('pass exactly one of --remote / --local');
const target = opts.remote ? '--remote' : '--local';

// SQL literal helpers. wrangler's --command takes no bind parameters, so every
// value is validated here and strings are quoted.
const sq = (s) => `'${String(s).replace(/'/g, "''")}'`;
function intArg(name, { min = 0 } = {}) {
    const v = opts[name];
    if (v === undefined) return undefined;
    if (!/^\d+$/.test(v) || Number(v) < min) die(`--${name} must be an integer >= ${min}`);
    return Number(v);
}
function qualityArg() {
    const v = opts.quality;
    if (v === undefined) return undefined;
    const q = Number(v);
    if (!Number.isFinite(q) || q < 0 || q > 10) die('--quality must be a number from 0 to 10');
    return q;
}
function urlArg() {
    const v = opts.url;
    if (v === undefined) return undefined;
    if (!/^https:\/\/\S+$/.test(v)) die('--url must be an https:// link');
    return v;
}
function dateArg() {
    const v = opts.date;
    if (v === undefined) return undefined;
    if (!/^\d{4}-\d{2}-\d{2}$/.test(v) || !Number.isFinite(Date.parse(v))) die('--date must be YYYY-MM-DD');
    return `${v}T12:00:00.000Z`;
}

function d1(sql) {
    const r = spawnSync('npx', ['wrangler', 'd1', 'execute', DB_NAME, target, '--json', '--command', sql], {
        cwd: PWA_DIR, encoding: 'utf8', maxBuffer: 64 * 1024 * 1024,
    });
    if (r.status !== 0) die(`wrangler failed:\n${r.stderr || r.stdout}`);
    let out;
    try { out = JSON.parse(r.stdout); } catch { die(`unreadable wrangler output:\n${r.stdout}`); }
    return out.flatMap((s) => s.results || []);
}

const DROP_CACHE = `DELETE FROM server_stats WHERE key = 'contest'`;

function resolveStrengthAccount() {
    let where;
    if (opts.egd !== undefined) {
        if (!/^\d+$/.test(opts.egd)) die('--egd must be a numeric PIN');
        where = `egd_pin = ${sq(opts.egd)}`;
    } else if (opts.aga !== undefined) {
        if (!/^\d+$/.test(opts.aga)) die('--aga must be a numeric id');
        where = `aga_id = ${sq(opts.aga)}`;
    } else if (opts.name !== undefined) {
        where = `display_name = ${sq(opts.name)} COLLATE NOCASE`;
    } else die('add-strength needs --egd, --aga or --name');
    const rows = d1(`SELECT user_id, display_name, egd_pin, aga_id FROM external_ids WHERE ${where}`);
    if (!rows.length) die('no account has linked that EGD/AGA record — the player links it in the app first (menu → Account)');
    if (rows.length > 1) die(`${rows.length} accounts match; use --egd or --aga`);
    const r = rows[0];
    console.log(`account: ${r.display_name} (${r.egd_pin ? `EGD ${r.egd_pin}` : `AGA ${r.aga_id}`})`);
    return r.user_id;
}

function resolveVideoAccount() {
    const words = String(opts.alias || '').trim().split(/\s+/);
    if (words.length !== 3 || !words.every((w) => /^[A-Za-z]+$/.test(w))) {
        die('--alias must be the three English words of the alias, e.g. "Brave Crimson Fox"');
    }
    const [adj, color, animal] = words;
    const rows = d1(`SELECT user_id FROM user_aliases
                      WHERE adjective = ${sq(adj)} COLLATE NOCASE AND color = ${sq(color)} COLLATE NOCASE
                        AND animal = ${sq(animal)} COLLATE NOCASE`);
    if (!rows.length) die(`no account has the alias "${words.join(' ')}"`);
    console.log(`account: alias ${words.join(' ')}`);
    return rows[0].user_id;
}

function insert(category, userId, fields) {
    const cols = ['category', 'user_id', 'submitted_at', 'reference'];
    const vals = [sq(category), sq(userId), sq(dateArg() || new Date().toISOString()), opts.reference ? 1 : 0];
    for (const [k, v] of Object.entries(fields)) {
        if (v === undefined) continue;
        cols.push(k);
        vals.push(typeof v === 'string' ? sq(v) : String(v));
    }
    const dup = d1(`SELECT id FROM contest_entries WHERE category = ${sq(category)} AND user_id = ${sq(userId)}`);
    if (dup.length) die(`that account already has a ${category} entry (id ${dup[0].id}); edit it with set --id ${dup[0].id}`);
    d1(`INSERT INTO contest_entries (${cols.join(', ')}) VALUES (${vals.join(', ')}); ${DROP_CACHE}`);
    console.log(`added ${category} entry`);
}

function list() {
    const rows = d1(`SELECT e.id, e.category, e.submitted_at, e.reference, e.video_url, e.video_problems,
                            e.views, e.quality, x.display_name, x.egd_pin, x.aga_id,
                            a.adjective || ' ' || a.color || ' ' || a.animal AS alias,
                            (h.user_id IS NOT NULL) AS hidden
                       FROM contest_entries e
                       LEFT JOIN external_ids x ON x.user_id = e.user_id
                       LEFT JOIN user_aliases a ON a.user_id = e.user_id
                       LEFT JOIN hidden_profiles h ON h.user_id = e.user_id
                      ORDER BY e.category, e.submitted_at, e.id`);
    if (!rows.length) { console.log('no entries'); return; }
    for (const r of rows) {
        const who = r.display_name
            ? `${r.display_name} (${r.egd_pin ? `EGD ${r.egd_pin}` : `AGA ${r.aga_id}`})`
            : (r.alias || '(no name, no alias)');
        const bits = [`#${r.id}`, r.category.padEnd(8), r.submitted_at.slice(0, 10), who];
        if (r.hidden) bits.push('[hidden]');
        if (r.reference) bits.push('[reference]');
        if (r.category === 'video') {
            bits.push(r.video_url || '(no url)', `problems=${r.video_problems ?? '—'}`,
                      `views=${r.views ?? '—'}`, `quality=${r.quality ?? '—'}`);
        }
        console.log(bits.join('  '));
    }
}

switch (cmd) {
    case 'list':
        list();
        break;
    case 'add-strength':
        insert('strength', resolveStrengthAccount(), {});
        break;
    case 'add-video': {
        const url = urlArg();
        const problems = intArg('problems', { min: 1 });
        if (!url) die('add-video needs --url');
        if (problems === undefined) die('add-video needs --problems (how many problems were solved in the video)');
        insert('video', resolveVideoAccount(), {
            video_url: url, video_problems: problems, views: intArg('views'), quality: qualityArg(),
        });
        break;
    }
    case 'set': {
        const id = intArg('id', { min: 1 });
        if (id === undefined) die('set needs --id');
        const sets = [];
        const views = intArg('views'), quality = qualityArg(), problems = intArg('problems', { min: 1 });
        const url = urlArg(), date = dateArg();
        if (views !== undefined) sets.push(`views = ${views}`);
        if (quality !== undefined) sets.push(`quality = ${quality}`);
        if (problems !== undefined) sets.push(`video_problems = ${problems}`);
        if (url !== undefined) sets.push(`video_url = ${sq(url)}`);
        if (date !== undefined) sets.push(`submitted_at = ${sq(date)}`);
        if (!sets.length) die('set needs at least one of --views --quality --problems --url --date');
        if (!d1(`SELECT id FROM contest_entries WHERE id = ${id}`).length) die(`no entry #${id}`);
        d1(`UPDATE contest_entries SET ${sets.join(', ')} WHERE id = ${id}; ${DROP_CACHE}`);
        console.log(`updated entry #${id}`);
        break;
    }
    case 'remove': {
        const id = intArg('id', { min: 1 });
        if (id === undefined) die('remove needs --id');
        if (!d1(`SELECT id FROM contest_entries WHERE id = ${id}`).length) die(`no entry #${id}`);
        d1(`DELETE FROM contest_entries WHERE id = ${id}; ${DROP_CACHE}`);
        console.log(`removed entry #${id}`);
        break;
    }
    case 'refresh':
        d1(DROP_CACHE);
        console.log('cache dropped; the next view of /contest recomputes');
        break;
    default:
        die('commands: list, add-strength, add-video, set, remove, refresh (see the header of this file)');
}
