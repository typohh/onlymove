#!/usr/bin/env node
// auxiliary/top_users_report.mjs
//
// Build a self-contained HTML report of the strongest users of ALL TIME by their
// LATEST estimated rank, reading the full local mirror (db/d1.sqlite), and open it in
// Chrome.
//
// ALL TIME, NOT THE LEADERBOARD TABLE. `leaderboard` holds one row per currently-active
// user and the weekly cron PRUNES idle ones, so ranking it answers "who is strong this
// week". This walks tsumego_reports instead — every user who ever reported — and
// rebuilds each one's rank buffer exactly as production would for a user it has no row
// for: buildRowStateFromReports' own query, the last BUFFER_CAP novel (is_repeat = 0)
// attempts, newest first by id. Their rank is therefore what the app would show them on
// their next novel report, and a player who stopped a year ago keeps the rank their
// final window supports.
//
// The rank is PRODUCTION'S, not a reimplementation: computeRank from the esbuild bundle
// in ./leaderboard_rank_lib.mjs — the shipped M curve, the rank-dependent SE floor, the
// problem params and RANK_START_MIN. It is the untimed internal SHOWN rank (0 = the
// 1k/1d boundary), the number the front-page badge and the stored pred_rank carry; the
// `±` is that estimate's standard error. Production gates a rank on RANK_START_MIN
// usable in-corpus attempts and this report keeps the gate (--min-usable lowers it,
// which is why the `±` column matters).
//
// "LATEST" IS THE LAST WINDOW, NOT A PEAK — a user's best-ever 500 attempts are not
// consulted. (Ranking peak form instead is a different, unbuilt report.)
//
// Behavioural columns come from tsumego_reports; name and country come from the same
// three tables /leaderboard stitches (EGD heartbeat country -> the external_ids binding
// -> the row's edge geo-IP -> survey geo-IP). Following
// auxiliary/leaderboard_player_stats.mjs, an operator report does NOT apply the
// hidden_profiles opt-out; hidden users are marked instead.
//
// Sanity check (printed, and footnoted in the HTML): every `leaderboard` row stores the
// pred_rank it was last updated with, so recomputing from its STORED buffer pins the
// bundle + params, and recomputing from the buffer RECONSTRUCTED here pins the
// reconstruction. A mismatch on either is reported rather than hidden.
//
// Usage:
//   node auxiliary/top_users_report.mjs                       # top 30 -> tmp/top_users_report.html
//   node auxiliary/top_users_report.mjs --top 50 --open
//   node auxiliary/top_users_report.mjs --min-usable 0 --top 100
//   node auxiliary/top_users_report.mjs --no-user-ids --out tmp/top_users_no_ids.html
//
// Reads db/d1.sqlite and pwa/tsumegos/; run `./.venv/bin/python3 -m pipeline.fetch_database`
// (and package_for_pwa, for problem_params.bin) first for fresh numbers.

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

import {
    REPO, DB_PATH, PARAMS_BIN, STALE_WARN_MS,
    loadRankModule, loadProblemParamsLocal, sqliteJson, tableColumns, durationStr,
} from './leaderboard_rank_lib.mjs';

// --- CLI ------------------------------------------------------------------------
const USAGE = `Top users of all time by their latest estimated rank, as an HTML report.

Usage: node auxiliary/top_users_report.mjs [options]

  --top N            how many users to report (default 30)
  --min-usable N     minimum usable in-corpus novel attempts for a rank
                     (default production's RANK_START_MIN); 0 disables the gate
  --db PATH          local sqlite mirror (default db/d1.sqlite)
  --out PATH         HTML output (default tmp/top_users_report.html)
  --json             also write the rows as JSON beside the HTML
  --no-user-ids      omit the user_id column (name, country and everything else stay),
                     for a copy that can leave this machine; pairs with --out
  --open             open the HTML with the google-chrome CLI when written
  --help, -h         show this message`;

const argv = process.argv.slice(2);
const opts = { top: 30, minUsable: null, db: DB_PATH, out: null, json: false, open: false, noUserIds: false };
for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') { console.log(USAGE); process.exit(0); }
    else if (a === '--top') opts.top = Number(argv[++i]);
    else if (a === '--min-usable') opts.minUsable = Number(argv[++i]);
    else if (a === '--db') opts.db = argv[++i];
    else if (a === '--out') opts.out = argv[++i];
    else if (a === '--json') opts.json = true;
    else if (a === '--no-user-ids') opts.noUserIds = true;
    else if (a === '--open') opts.open = true;
    else { console.error(`unknown arg: ${a}\n\n${USAGE}`); process.exit(1); }
}
if (!Number.isInteger(opts.top) || opts.top < 1) { console.error('--top needs a positive integer'); process.exit(1); }
if (opts.minUsable !== null && !(opts.minUsable >= 0)) { console.error('--min-usable needs a number >= 0'); process.exit(1); }
if (opts.out === undefined || opts.out === null) opts.out = path.join(REPO, 'tmp', 'top_users_report.html');

// --- formatting -----------------------------------------------------------------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const int = (n) => Number(n).toLocaleString('en-US');
const dateOnly = (ts) => (typeof ts === 'string' && ts.length >= 10 ? ts.slice(0, 10) : '—');
// One cell's worth of rank evidence. `declared` is the user's own server-rank claim
// (latest row per server, newest first); `verified` is the machine-fetched EGD GoR /
// AGA rating. They are separate systems and are never merged or compared as though one
// converted into the other.
//
// For a VERIFIED row the rank shown is OUR scale: the raw rating put through the same
// conversion the rest of the pipeline uses — gorToRank for EGD (100 GoR per band, with
// 1k anchored at 2000 so the band CENTRES sit on the round hundreds, which puts the
// 3d/4d border at 2350), agaRatingToRank for AGA. The federation's OWN label in
// `rank_label` is a DIFFERENT quantity and is shown only when it disagrees — for EGD it
// is the player's club grade, which is not a GoR readout at all (in this corpus GoR 2225
// is graded "3d" while 2226-2246 are "2d"), so printing it as the GoR's rank invents a
// conversion that does not exist.
const declStr = (r) => (r.declared.length
    ? r.declared.map((d) => `${d.server} ${d.rank}`).join(' · ') : '');
function verPart(v, mod) {
    const svc = v.server === 'AGAR' ? 'AGA' : v.server;
    const rating = Number(v.rank);
    const value = v.server === 'AGAR' ? mod.agaRatingToRank(rating) : mod.gorToRank(rating);
    const band = Number.isFinite(value) ? mod.rankToLabel(value) : null;
    const official = v.label || null;
    return {
        svc, value, band, official,
        text: `${svc} ${v.rank}${band ? ` → ${band}` : ''}`
            + (official && official !== band ? ` (${svc} says ${official})` : ''),
    };
}
const verStr = (r, mod) => (r.verified.length
    ? r.verified.map((v) => verPart(v, mod).text).join(' · ') : '');
const declTitle = (r) => (r.declared.length
    ? r.declared.map((d) => `${d.server} ${d.rank} — declared ${dateOnly(d.at)}`).join('\n')
    : 'no declaration');
const verTitle = (r, mod) => (r.verified.length
    ? r.verified.map((v) => {
        const p = verPart(v, mod);
        return `${p.svc} ${v.rank}` + (Number.isFinite(p.value) ? ` = ${p.value.toFixed(2)} on our rank scale (${p.band})` : '')
            + (p.official && p.official !== p.band ? `; ${p.svc}'s own label is ${p.official}` : '')
            + ` — heartbeat ${dateOnly(v.at)}`;
    }).join('\n')
    : 'no verified rating');
function median(xs) {
    if (!xs.length) return null;
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function flagEmoji(cc) {
    if (!/^[A-Z]{2}$/.test(cc || '')) return '';
    return String.fromCodePoint(...[...cc].map((c) => 0x1f1e6 + c.charCodeAt(0) - 65));
}

async function main() {
    if (!fs.existsSync(opts.db)) {
        console.error(`missing ${opts.db} — run: ./.venv/bin/python3 -m pipeline.fetch_database`);
        process.exit(1);
    }
    if (!fs.existsSync(PARAMS_BIN)) {
        console.error(`missing ${PARAMS_BIN} — run: ./.venv/bin/python3 -m pipeline.package_for_pwa`);
        process.exit(1);
    }

    const [{ m: latest }] = sqliteJson('SELECT MAX(created_at) AS m FROM tsumego_reports', opts.db);
    const ageMs = Date.now() - Date.parse(latest || 0);
    const stale = !Number.isFinite(ageMs) || ageMs > STALE_WARN_MS;
    if (stale) {
        console.warn(`WARNING: local mirror's newest report is ${latest || 'unknown'} `
            + `(${Number.isFinite(ageMs) ? Math.round(ageMs / 3600000) : '?'}h old). `
            + 'Run pipeline.fetch_database first for fresh numbers.\n');
    }

    const mod = await loadRankModule();
    const params = loadProblemParamsLocal();
    const MIN_USABLE = opts.minUsable === null ? mod.RANK_START_MIN : opts.minUsable;

    // --- identity: the same three tables /leaderboard stitches ----------------------
    const idBy = new Map(sqliteJson(
        'SELECT user_id, display_name, country_code FROM external_ids', opts.db
    ).map((r) => [r.user_id, r]));
    const hidden = new Set(sqliteJson('SELECT user_id FROM hidden_profiles', opts.db).map((r) => r.user_id));
    const lbCountry = new Map(sqliteJson(
        'SELECT user_id, country FROM leaderboard', opts.db
    ).map((r) => [r.user_id, r.country]).filter(([, c]) => c));

    // Latest row per (user, server) — the bare columns come from the row holding MAX(id),
    // which lets one pass serve three purposes: the 'EGD' heartbeat's country is the EGD
    // federation code (most reliable) and every other server's is Cloudflare geo-IP; the
    // HUMAN declarations (everything that is not a weekly machine heartbeat) are the
    // server ranks the user told us; and 'EGD'/'AGAR' carry the VERIFIED rating the
    // system fetched on their behalf. Declared and verified stay apart on purpose — a
    // declaration is calibration input, not ground truth (docs/methodology.md §1.12).
    const egdCountry = new Map(), surveyCountry = new Map();
    const declared = new Map(), verified = new Map();
    const push = (m, k, v) => { if (!m.has(k)) m.set(k, []); m.get(k).push(v); };
    for (const h of sqliteJson(
        `SELECT user_id, server, rank, rank_label, created_at, country, MAX(id)
           FROM rank_surveys GROUP BY user_id, server`, opts.db)) {
        if (h.server === 'EGD') {
            if (h.country) egdCountry.set(h.user_id, h.country);
            push(verified, h.user_id, h);
        } else if (h.server === 'AGAR') {
            push(verified, h.user_id, h);
        } else {
            if (h.country && !surveyCountry.has(h.user_id)) surveyCountry.set(h.user_id, h.country);
            push(declared, h.user_id, h);
        }
    }
    // Report-only last resort, filled once the reported ids are known: the edge geo-IP
    // Cloudflare stamped on this user's own share-link events. It is the SAME source as
    // leaderboard.country, recorded in a second table that readLeaderboardRows never
    // reads — which matters here precisely because the weekly cron prunes idle
    // leaderboard rows, so for a stopped player this is the only geo-IP left. Both
    // event types are the user's own geo: an 'open' row's user_id is the opener
    // (state.userId in recordShareEvent), and its country is that same request's
    // cf.country. It sits LAST so it can only fill a blank, never override production.
    const shareCountry = new Map();
    const countryOf = (uid) => {
        const chain = [
            ['EGD', egdCountry.get(uid)],
            ['binding', (idBy.get(uid) || {}).country_code],
            ['geo-IP', lbCountry.get(uid)],
            ['survey', surveyCountry.get(uid)],
            ['share geo-IP', shareCountry.get(uid)],
        ];
        for (const [source, raw] of chain) {
            const cc = mod.normalizeCountry(raw);
            if (cc) return { cc, source };
        }
        return { cc: null, source: null };
    };

    // --- rank every user who ever reported -----------------------------------------
    // Production's buffer, for every user rather than the pruned leaderboard rows:
    // last <= BUFFER_CAP novel attempts, newest first by id (buildRowStateFromReports).
    // The HAVING prefilters to users who could possibly clear the gate; it cannot drop
    // anyone computeRank would rank (it counts novel rows, the gate counts usable ones).
    console.log(`ranking every user with >= ${MIN_USABLE} novel attempts (all time) ...`);
    const bufs = new Map();
    for (const r of sqliteJson(
        `SELECT user_id, p, s, t FROM (
            SELECT user_id, problem AS p, success AS s, time_used AS t,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id DESC) AS rn
              FROM tsumego_reports
             WHERE is_repeat = 0
               AND user_id IN (SELECT user_id FROM tsumego_reports WHERE is_repeat = 0
                                GROUP BY user_id HAVING COUNT(*) >= ${MIN_USABLE})
         ) WHERE rn <= ${mod.BUFFER_CAP} ORDER BY user_id, rn`, opts.db)) {
        if (!bufs.has(r.user_id)) bufs.set(r.user_id, []);
        bufs.get(r.user_id).push({ p: r.p, s: r.s ? 1 : 0, t: Number.isFinite(r.t) ? r.t : null });
    }

    const ranked = [];
    for (const [uid, buf] of bufs) {
        const { onlymoveRank, se, usable } = mod.computeRankBoth(buf, params);
        if (!Number.isFinite(onlymoveRank) || usable < MIN_USABLE) continue;
        ranked.push({ userId: uid, rank: onlymoveRank, se, usable });
    }
    if (!ranked.length) { console.log(`no user clears ${MIN_USABLE} usable attempts — nothing to report`); return; }
    ranked.sort((a, b) => (b.rank - a.rank) || (b.usable - a.usable));
    const top = ranked.slice(0, opts.top);

    // --- sanity check: the stored pred_rank is the ground truth we can check against --
    // Recomputed from the STORED buffer it must reproduce (pins bundle + params); from
    // the buffer RECONSTRUCTED here it must reproduce (pins the reconstruction).
    const deltas = { stored: [], rebuilt: [] };
    for (const r of sqliteJson(
        `SELECT user_id, buffer, pred_rank FROM leaderboard
          WHERE pred_rank IS NOT NULL AND buffer IS NOT NULL`, opts.db)) {
        const pred = Number(r.pred_rank);
        if (!Number.isFinite(pred)) continue;
        const fromStored = mod.computeRankBoth(mod.parseBuffer(r.buffer), params).onlymoveRank;
        if (Number.isFinite(fromStored)) deltas.stored.push(Math.abs(fromStored - pred));
        const buf = bufs.get(r.user_id);
        if (buf) {
            const fromRebuilt = mod.computeRankBoth(buf, params).onlymoveRank;
            if (Number.isFinite(fromRebuilt)) deltas.rebuilt.push(Math.abs(fromRebuilt - pred));
        }
    }
    const agree = (xs) => (xs.length
        ? `${xs.length} rows, median |Δ| ${median(xs).toFixed(3)}, ${(100 * xs.filter((d) => d < 0.05).length / xs.length).toFixed(0)}% within 0.05`
        : 'no rows to check');

    // --- behaviour + identity for the reported users ---------------------------------
    const idList = top.map((t) => `'${t.userId.replace(/'/g, "''")}'`).join(',');
    if (tableColumns('share_events', opts.db).has('country')) {
        // Most frequent wins (a single travelling session should not outvote a home
        // country); a tie falls to whichever SQLite returns first for that count.
        for (const s of sqliteJson(
            `SELECT user_id, country, COUNT(*) AS n FROM share_events
              WHERE country IS NOT NULL AND user_id IN (${idList})
              GROUP BY user_id, country ORDER BY user_id, n DESC`, opts.db)) {
            if (!shareCountry.has(s.user_id)) shareCountry.set(s.user_id, s.country);
        }
    }
    const byUser = new Map();
    for (const r of sqliteJson(
        `SELECT user_id, problem, success, size, created_at, time_used, is_repeat
           FROM tsumego_reports WHERE user_id IN (${idList})
          ORDER BY user_id, created_at, id`, opts.db)) {
        if (!byUser.has(r.user_id)) byUser.set(r.user_id, []);
        byUser.get(r.user_id).push(r);
    }

    const rows = top.map((t, i) => {
        const all = byUser.get(t.userId) || [];
        const novel = all.filter((r) => !r.is_repeat);
        const times = novel.map((r) => Number(r.time_used)).filter((v) => Number.isFinite(v) && v > 0);
        const per = { 9: { n: 0, probs: new Set() }, 13: { n: 0, probs: new Set() } };
        for (const r of novel) {
            const b = per[r.size];
            if (!b) continue;
            b.n++; b.probs.add(r.problem);
        }
        // "% answered novel problems correctly, excluding the first 30": drop the
        // earliest 30 first-time attempts (warm-up), then the success rate of the rest.
        // The rows arrive ordered by created_at, id — event order, matching "first".
        const after = novel.slice(30);
        const ident = idBy.get(t.userId) || {};
        const { cc, source } = countryOf(t.userId);
        return {
            place: i + 1,
            userId: t.userId,
            name: ident.display_name || null,
            hidden: hidden.has(t.userId),
            country: cc, countrySource: source,
            rank: t.rank, se: t.se, usable: t.usable,
            attempts: all.length, novelAttempts: novel.length,
            novel9: per[9].n, novel13: per[13].n,
            distinct9: per[9].probs.size, distinct13: per[13].probs.size,
            medianMs: median(times), nTimed: times.length,
            first: all.length ? all[0].created_at : null,
            last: all.length ? all[all.length - 1].created_at : null,
            accN: after.length,
            acc: after.length ? after.reduce((s, r) => s + (r.success ? 1 : 0), 0) / after.length : null,
            // Newest declaration first; a re-declaration on the same server replaces the
            // older one in the cell but stays in the title.
            declared: (declared.get(t.userId) || []).slice()
                .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
                .map((d) => ({ server: d.server, rank: d.rank, at: d.created_at })),
            verified: (verified.get(t.userId) || []).slice()
                .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
                .map((v) => ({ server: v.server, rank: v.rank, label: v.rank_label, at: v.created_at })),
        };
    });

    // --- render ---------------------------------------------------------------------
    const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
    const pad = (s, n) => String(s).padEnd(n);
    const padl = (s, n) => String(s).padStart(n);
    const uid = (s) => (opts.noUserIds ? '' : s);
    console.log(`\n${pad('#', 3)}${pad('rank', 14)}${uid(pad('user', 10))}${pad('name', 20)}`
        + `${pad('ctry', 6)}${pad('declared (latest/server)', 30)}${pad('verified (our scale)', 30)}`
        + `${padl('attempts', 10)}${padl('9×9', 7)}${padl('13×13', 7)}`
        + `${padl('median', 9)}${pad('  first', 14)}${pad('  latest', 14)}${padl('novel acc', 14)}`);
    for (const r of rows) {
        const rank = `${r.rank.toFixed(2)} ${mod.rankToLabel(r.rank)}`;
        const namec = (r.name || '—') + (r.hidden ? ' (anon)' : '');
        console.log(`${pad(r.place, 3)}${pad(rank, 14)}${uid(pad(r.userId, 10))}${pad(namec.slice(0, 19), 20)}`
            + `${pad(r.country || '—', 6)}${pad(declStr(r).slice(0, 29) || '—', 30)}${pad(verStr(r, mod).slice(0, 33) || '—', 34)}`
            + `${padl(int(r.attempts), 10)}${padl(int(r.novel9), 7)}${padl(int(r.novel13), 7)}`
            + `${padl(durationStr(r.medianMs), 9)}${pad('  ' + dateOnly(r.first), 14)}${pad('  ' + dateOnly(r.last), 14)}`
            + `${padl(r.acc === null ? '—' : `${(100 * r.acc).toFixed(1)}% (n=${int(r.accN)})`, 14)}`);
    }
    console.log(`\n${ranked.length} user(s) clear the ${MIN_USABLE}-usable gate over all time; showing the top ${top.length}.`);
    console.log(`Sanity check vs stored pred_rank — from the STORED buffer: ${agree(deltas.stored)}`
        + `\n                              from the REBUILT buffer: ${agree(deltas.rebuilt)}`);

    const html = renderHtml({ mod, rows, stamp, latest, stale, minUsable: MIN_USABLE, total: ranked.length,
        noUserIds: opts.noUserIds,
        deltas, agree, db: opts.db, paramsMtime: fs.statSync(PARAMS_BIN).mtime.toISOString(),
        bufferCap: mod.BUFFER_CAP, rankStartMin: mod.RANK_START_MIN });
    fs.mkdirSync(path.dirname(path.resolve(opts.out)), { recursive: true });
    fs.writeFileSync(opts.out, html);
    console.log(`\nHTML: ${opts.out}`);
    if (opts.json) {
        // --no-user-ids has to hold for the JSON too, or the copy that was meant to leave
        // the machine carries the id in a second file beside it.
        const outRows = opts.noUserIds ? rows.map(({ userId, ...rest }) => rest) : rows;
        const jsonPath = opts.out.replace(/\.html?$/i, '') + '.json';
        fs.writeFileSync(jsonPath, JSON.stringify({ generated: stamp, dataAsOf: latest, rows: outRows }, null, 2));
        console.log(`JSON: ${jsonPath}`);
    }
    if (opts.open) openInChrome(opts.out);
}

// Chrome needs an absolute file:// URL; a relative one is parsed against the CWD and
// silently opens the wrong thing.
function openInChrome(outPath) {
    const url = `file://${path.resolve(outPath)}`;
    const bin = process.env.CHROME_PATH || 'google-chrome';
    try {
        const child = spawn(bin, ['--new-window', url], { detached: true, stdio: 'ignore' });
        child.on('error', (e) => console.error(`could not launch ${bin} (${e.message}); open ${url} yourself`));
        child.unref();
        console.log(`opened in ${bin}: ${url}`);
    } catch (e) {
        console.error(`could not launch ${bin} (${e.message}); open ${url} yourself`);
    }
}

function renderHtml({ mod, rows, stamp, latest, stale, minUsable, total, noUserIds, deltas, agree, db, paramsMtime, bufferCap, rankStartMin }) {
    const accTint = (a) => {
        if (a === null) return 'transparent';
        const t = Math.max(0, Math.min(1, (a - 0.4) / 0.5));   // 40% -> 90%
        return `hsl(${(0.08 + 0.25 * t) * 360} 70% ${93 - 8 * t}%)`;
    };
    const body = rows.map((r) => `
      <tr>
        <td class="n">${r.place}</td>
        <td class="rank"><span class="rv">${r.rank.toFixed(2)}</span> <span class="band">${esc(mod.rankToLabel(r.rank))}</span><span class="se">±${r.se.toFixed(2)}</span></td>
        ${noUserIds ? '' : `<td class="uid">${esc(r.userId)}</td>`}
        <td class="name">${r.name ? esc(r.name) : '<span class="dim">—</span>'}${r.hidden ? '<span class="chip" title="hidden_profiles opt-out set: the public /leaderboard renders this account as Anonymous, withholding its name, country and declared ranks. This operator report is not that page — it keeps the name on record and marks the opt-out here.">anonymized</span>' : ''}</td>
        <td class="cc">${r.country ? `${flagEmoji(r.country)} <span title="source: ${esc(r.countrySource)}">${esc(r.country)}</span>` : '<span class="dim">—</span>'}</td>
        <td class="decl" title="${esc(declTitle(r))}">${r.declared.length ? esc(declStr(r)) : '<span class="dim">—</span>'}</td>
        <td class="ver" title="${esc(verTitle(r, mod))}">${r.verified.length ? esc(verStr(r, mod)) : '<span class="dim">—</span>'}</td>
        <td class="n" title="all reports, repeats included">${int(r.attempts)}</td>
        <td class="n" title="${int(r.novel9)} novel attempts over ${int(r.distinct9)} distinct problems">${int(r.novel9)}</td>
        <td class="n" title="${int(r.novel13)} novel attempts over ${int(r.distinct13)} distinct problems">${int(r.novel13)}</td>
        <td class="n" title="median over ${int(r.nTimed)} novel attempts that recorded a time">${durationStr(r.medianMs)}</td>
        <td class="d">${dateOnly(r.first)}</td>
        <td class="d">${dateOnly(r.last)}</td>
        <td class="n acc" style="background:${accTint(r.acc)}" title="novel success rate after dropping the first 30 first-time attempts">${r.acc === null ? '—' : `<b>${(100 * r.acc).toFixed(1)}%</b> <span class="dim">n=${int(r.accN)}</span>`}</td>
      </tr>`).join('');
    return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Strongest users of all time — onlymove</title>
<style>
  :root { --ink:#1b1d21; --dim:#8a9099; --line:#e3e6ea; --head:#f6f7f9; }
  * { box-sizing: border-box; }
  body { margin:0; padding:28px 32px 56px; background:#fff; color:var(--ink);
         font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
  h1 { font-size:21px; margin:0 0 4px; letter-spacing:-.01em; }
  .sub { color:var(--dim); margin:0 0 18px; }
  .sub code { color:#4a5058; background:var(--head); padding:1px 4px; border-radius:3px; }
  .warn { background:#fff8e6; border:1px solid #f2d98a; border-radius:6px; padding:8px 12px; margin:0 0 16px; }
  table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
  th, td { padding:6px 9px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
  th { position:sticky; top:0; background:var(--head); font-size:12px; text-transform:uppercase;
       letter-spacing:.04em; color:#5b626b; border-bottom:1px solid #d6dae0; }
  tbody tr:hover { background:#fbfcfd; }
  td.n { text-align:right; }
  td.rank { font-weight:600; }
  .rv { font-size:15px; }
  .band { color:#3d7a3d; font-weight:500; }
  .se { color:var(--dim); font-weight:400; font-size:12px; margin-left:5px; }
  td.uid { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; color:#4a5058; }
  .dim { color:var(--dim); }
  .chip { margin-left:6px; font-size:10.5px; text-transform:uppercase; letter-spacing:.04em;
          color:#8a6d1f; background:#fdf3d6; border:1px solid #f0e0ab; border-radius:3px; padding:1px 4px; }
  td.cc { font-size:15px; }
  td.decl, td.ver { white-space:normal; max-width:240px; font-size:13px; }
  td.ver { color:#3d5a7a; }
  td.acc { border-left:1px solid var(--line); }
  footer { margin-top:26px; max-width:1100px; color:#4a5058; font-size:12.5px; }
  footer h2 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:#5b626b; margin:18px 0 6px; }
  footer li { margin:2px 0; }
  footer code { background:var(--head); padding:1px 4px; border-radius:3px; }
</style></head><body>
<h1>Strongest users of all time — latest estimated rank</h1>
<p class="sub">Ranks computed by the shipped estimator (<code>computeRank</code>) over each user's last ≤${bufferCap} novel attempts, rebuilt from <code>${esc(db)}</code> for <b>every user who ever reported</b> — not the <code>leaderboard</code> table, which the cron prunes to the currently active. Generated ${esc(stamp)} UTC; mirror's newest report ${esc(latest || 'unknown')}.${noUserIds ? ' <b>User ids omitted.</b>' : ''}</p>
${stale ? '<div class="warn"><b>Stale mirror.</b> The local DB copy is more than 6h behind; run <code>./.venv/bin/python3 -m pipeline.fetch_database</code> for fresh numbers.</div>' : ''}
<table>
  <thead><tr>
    <th>#</th><th>Rank ±se</th>${noUserIds ? '' : '<th>user</th>'}<th>Name</th><th>Ctry</th><th>Declared ranks</th><th>Verified (our scale)</th>
    <th style="text-align:right">Attempts</th><th style="text-align:right">9×9</th><th style="text-align:right">13×13</th>
    <th style="text-align:right">Median</th><th>First</th><th>Latest</th><th style="text-align:right">Novel acc</th>
  </tr></thead>
  <tbody>${body}
  </tbody>
</table>
<footer>
  <h2>How to read this</h2>
  <ul>
    <li><b>Rank</b> is the untimed internal onlymove SHOWN rank (0 = the 1k/1d boundary, one unit per band) — the same number the front-page badge and the stored <code>pred_rank</code> carry, computed from the user's <b>latest</b> window, not their peak. <b>±</b> is the estimate's standard error.</li>
    <li><b>Attempts</b> counts every report (repeats included). <b>9×9 / 13×13</b> count novel (<code>is_repeat = 0</code>) attempts; hover for the distinct-problem count. <b>Median</b> is over novel attempts that recorded a time.</li>
    <li><b>Novel acc</b> is the success rate over first-time attempts <b>after dropping the earliest 30</b> (the warm-up); <code>n</code> is how many remain.</li>
    <li><b>Declared ranks</b> and <b>Verified</b> are two SEPARATE systems, not two views of one quantity. A declaration is the server rank the user themself reported (<code>rank_surveys</code>, latest row per server; hover for every declaration and its date) — calibration input, not ground truth. Verified is the rating the system fetched on their behalf (<code>EGD</code> GoR / <code>AGAR</code> AGA rating). Nothing here derives one from the other, and a declaration is not evidence about a heartbeat or vice versa.</li>
    <li><b>Verified (our scale)</b> puts the raw rating through <b>our</b> conversion — <code>gorToRank</code> for EGD, <code>agaRatingToRank</code> for AGA, the functions <code>utils/rating/rank_surveys.py</code> mirrors — because that is the frame the estimate beside it lives in. For EGD that is 100 GoR per band with the band <i>centres</i> on the round hundreds, so the 3d/4d border sits at <b>2350</b> and GoR 2347 reads 3d. EGD's own <code>rank_label</code> is a <i>different</i> quantity — the player's club grade, not a GoR readout — so it is shown next to the raw GoR only when it disagrees with this band.</li>
    <li><b>Ctry</b> uses production's /leaderboard chain first — EGD federation code → the linked identity's country (<code>external_ids</code>) → the leaderboard row's edge geo-IP → a survey's geo-IP — and only then, for a user that chain leaves blank, the edge geo-IP on their own share-link events (<code>share_events</code>; report-only, production never reads it, which matters because the cron prunes idle leaderboard rows). Hover shows which source won.</li>
    <li><b>Gate:</b> ${total} user(s) clear ${minUsable} usable in-corpus novel attempts${minUsable === rankStartMin ? ` (production's RANK_START_MIN = ${rankStartMin})` : ` (overridden; production's gate is ${rankStartMin})`}; below it production itself lists no rank.</li>
    <li><b>anonymized</b> marks a <code>hidden_profiles</code> opt-out — the account the public /leaderboard renders as "Anonymous", with its name, country and declared ranks withheld. This report is not that page: names are shown for hidden users too, with the opt-out marked instead of applied (the same choice <code>auxiliary/leaderboard_player_stats.mjs</code> makes).</li>
  </ul>
  <h2>Provenance &amp; check</h2>
  <ul>
    <li>Rank params: <code>pwa/tsumegos/problem_params.bin</code> (mtime ${esc(paramsMtime)}), bundled with <code>pwa/src/leaderboard.js</code> at run time, so the curve/floor/params cannot drift from the Worker's.</li>
    <li>The buffer is production's own: <code>ORDER BY id DESC LIMIT ${bufferCap}</code> over novel attempts — the query in <code>buildRowStateFromReports</code>.</li>
    <li><b>Sanity check</b> against the ${deltas.stored.length} stored <code>pred_rank</code> row(s) that also hold a buffer — from the STORED buffer: ${esc(agree(deltas.stored))}; from the buffer REBUILT here: ${esc(agree(deltas.rebuilt))}. A median |Δ| near 0 pins both the bundle/params and the reconstruction.</li>
  </ul>
</footer>
</body></html>
`;
}

await main();
