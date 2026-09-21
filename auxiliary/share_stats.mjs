#!/usr/bin/env node
// auxiliary/share_stats.mjs
//
// Read-only operator report on the share-link analytics in `share_events`:
//
//   1. shares CREATED per country   — who generated share links, by edge geo-IP
//   2. shares OPENED per country    — who arrived via a shared link, by edge geo-IP
//   3. created / opened per month   — the two series' progress over time
//   4. top sharers                  — name, rank, country (+ how many links/problems)
//
// WHAT A ROW MEANS. `POST /share-event` (pwa/src/handlers/share-event.js) writes one
// row per event, and `country` is always Cloudflare's edge geo-IP for THAT request:
//
//   event_type = 'create'  the SHARER generated a share link. user_id = the sharer.
//   event_type = 'open'    someone ARRIVED via a shared link (main.js records it when
//                          the URL carries ?id). user_id = the OPENER/VIEWER.
//
// So the created-per-country and opened-per-country tables describe two DIFFERENT
// populations (creators vs viewers), and an open cannot be attributed to a particular
// create: the row keeps no share/link token, only the problem id. The two are reported
// side by side and never as a conversion funnel — an opens/creates ratio is views per
// link generated, nothing more.
//
// "TOP SHARERS" therefore means top by CREATE count only. An open row's user_id is the
// viewer, so counting a user's open rows would rank them by how much they READ other
// people's links, not by how much they shared.
//
// RANK is production's own shown rank, not a reimplementation: the esbuild bundle from
// ./leaderboard_rank_lib.mjs (the shipped M curve, SE floor, time dock, couplings and
// problem params). It is computed from the user's stored `leaderboard` buffer when they
// have one and REBUILT from `tsumego_reports` (buildRowStateFromReports' own query) when
// the weekly cron has pruned their row — so a sharer who has stopped playing still gets
// the rank their last window supports. Production's RANK_START_MIN gate is DROPPED (as
// in auxiliary/leaderboard_player_stats.mjs) and a below-gate rank is marked `*`.
//
// COUNTRY is the one identity field with several independent sources, and they can
// disagree because they are separate Cloudflare edge captures taken on different
// requests:
//
//   the `leaderboard` row's geo-IP  — stamped on every /report, last non-empty wins,
//                                     so it follows travel (the /leaderboard page's
//                                     universal flag fallback)
//   a `rank_surveys` row's geo-IP   — frozen at the moment the survey was declared
//   `external_ids.country_code`     — snapshotted from EGD/AGA at link time
//   the 'EGD' heartbeat's country   — the EGD federation code, not geo-IP
//   the user's own share_events     — this table's edge geo-IP (operator-only fallback;
//                                     production never reads share_events for a flag)
//
// The top-sharers table prints the resolved code with the winning source beside it, and
// the run reports where the two person-level captures — the leaderboard row and the
// latest survey row — disagree, rather than hiding the conflict behind one value. The
// per-country tables need none of this: their country IS the event's own edge geo-IP.
//
// Data source: db/d1.sqlite (the local D1 mirror). Run
// `./.venv/bin/python3 -m pipeline.fetch_database` first for fresh numbers. The rank
// columns additionally need pwa/tsumegos/problem_params.bin (`package_for_pwa`).
//
// Usage:
//   node auxiliary/share_stats.mjs
//   node auxiliary/share_stats.mjs --top 30 --months 6 --countries 10
//   node auxiliary/share_stats.mjs --json
//
// Run with --help for every flag.

import fs from 'node:fs';

import {
    DB_PATH, PARAMS_BIN, STALE_WARN_MS,
    loadRankModule, loadProblemParamsLocal, sqliteJson, tableColumns, rankStr,
} from './leaderboard_rank_lib.mjs';

// --- CLI ------------------------------------------------------------------------
const USAGE = `Share-link analytics from share_events: created/opened per country, the two
series by month, and the top sharers with name, production rank and country.

Usage: node auxiliary/share_stats.mjs [options]

  --db PATH         local sqlite mirror (default db/d1.sqlite)
  --top N           how many top sharers to list (default 20; 0 = all)
  --months N        how many recent months in the progress table (default 12; 0 = all)
  --countries N     how many rows per per-country table (default 15; 0 = all)
  --json            machine-readable JSON instead of the text report
  --no-user-ids     omit the user_id column (for a copy that can leave this machine)
  --help, -h        show this message

Reads db/d1.sqlite and pwa/tsumegos/problem_params.bin. Run
\`./.venv/bin/python3 -m pipeline.fetch_database\` first for fresh numbers.`;

const argv = process.argv.slice(2);
const opts = { db: DB_PATH, top: 20, months: 12, countries: 15, json: false, noUserIds: false };
for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--help' || a === '-h') { console.log(USAGE); process.exit(0); }
    else if (a === '--db') opts.db = argv[++i];
    else if (a === '--top') opts.top = Number(argv[++i]);
    else if (a === '--months') opts.months = Number(argv[++i]);
    else if (a === '--countries') opts.countries = Number(argv[++i]);
    else if (a === '--json') opts.json = true;
    else if (a === '--no-user-ids') opts.noUserIds = true;
    else { console.error(`unknown arg: ${a}\n\n${USAGE}`); process.exit(1); }
}
for (const [name, v] of [['--top', opts.top], ['--months', opts.months], ['--countries', opts.countries]]) {
    if (!Number.isInteger(v) || v < 0) { console.error(`${name} needs an integer >= 0`); process.exit(1); }
}
if (opts.db === undefined) { console.error('--db needs a path'); process.exit(1); }

// --- formatting -----------------------------------------------------------------
const pad = (s, n) => { s = String(s); return s.length >= n ? s : s + ' '.repeat(n - s.length); };
const padl = (s, n) => { s = String(s); return s.length >= n ? s : ' '.repeat(n - s.length) + s; };
const int = (n) => Number(n).toLocaleString('en-US');
const pct = (n, total) => (total > 0 ? `${((100 * n) / total).toFixed(1)}%` : '—');
const day = (ts) => (typeof ts === 'string' && ts.length >= 10 ? ts.slice(0, 10) : '—');
// ASCII bar on a SHARED scale, so a creates bar and an opens bar of the same length
// mean the same number of events (opens outnumber creates several-fold here, and
// per-series scaling would hide exactly that).
function bar(v, max, width) {
    if (!v) return '';
    return '█'.repeat(Math.max(1, Math.round((v / Math.max(1, max)) * width)));
}
const UNKNOWN = '(unknown)';

// Month helpers: created_at is ISO-8601 UTC, and SQLite's date functions read it as
// UTC, so months bucket on the UTC calendar. Fill the gaps between the first and last
// month so "progress" reads as a continuous series.
function monthRange(first, last) {
    const out = [];
    let [y, m] = first.split('-').map(Number);
    const [ly, lm] = last.split('-').map(Number);
    while (y < ly || (y === ly && m <= lm)) {
        out.push(`${y}-${String(m).padStart(2, '0')}`);
        m += 1;
        if (m > 12) { m = 1; y += 1; }
    }
    return out;
}

// The /leaderboard page's chain, then the operator-only share_events fallback:
// EGD federation code -> the linked identity's country -> the leaderboard row's edge
// geo-IP -> a survey's edge geo-IP -> the user's own share events' geo-IP.
// `normalizeCountry` is the SHIPPED filter (real ISO-alpha-2 only, sentinels -> null),
// so a placeholder from one source cannot clobber a real code from another.
function resolveCountry(mod, src) {
    const chain = [
        ['EGD', src.egd],
        ['linked', src.linked],
        ['leaderboard', src.leaderboard],
        ['survey', src.survey],
        ['share_events', src.shareEvents],
    ];
    for (const [source, raw] of chain) {
        const country = mod.normalizeCountry(raw);
        if (country) return { country, source };
    }
    return { country: null, source: null };
}

async function main() {
    if (!fs.existsSync(opts.db)) {
        console.error(`missing ${opts.db} — run: ./.venv/bin/python3 -m pipeline.fetch_database`);
        process.exit(1);
    }

    const q = (sql) => sqliteJson(sql, opts.db);
    if (!tableColumns('share_events', opts.db).has('event_type')) {
        console.error(`${opts.db} has no share_events table — fetch a mirror that carries it.`);
        process.exit(1);
    }
    // country arrived with the table, but probe it so an older snapshot degrades to a
    // single (unknown) bucket instead of failing the query.
    const hasCountry = tableColumns('share_events', opts.db).has('country');
    const ccExpr = hasCountry ? "COALESCE(NULLIF(country, ''), NULL)" : 'NULL';

    const totals = q(
        `SELECT COUNT(*) AS events,
                SUM(event_type = 'create') AS creates,
                SUM(event_type = 'open')   AS opens,
                COUNT(DISTINCT CASE WHEN event_type = 'create' THEN user_id END) AS sharers,
                COUNT(DISTINCT CASE WHEN event_type = 'open'   THEN user_id END) AS viewers,
                COUNT(DISTINCT CASE WHEN event_type = 'create' THEN problem_id END) AS sharedProblems,
                MIN(created_at) AS firstAt,
                MAX(created_at) AS lastAt
           FROM share_events`)[0];
    if (!totals.events) { console.log('share_events is empty.'); return; }
    const ageMs = Date.now() - Date.parse(totals.lastAt || 0);
    if (!Number.isFinite(ageMs) || ageMs > STALE_WARN_MS) {
        console.warn(`WARNING: local mirror's newest share event is ${totals.lastAt || 'unknown'} `
            + `(${Number.isFinite(ageMs) ? Math.round(ageMs / 3600000) : '?'}h old). `
            + 'Run pipeline.fetch_database first for fresh numbers.\n');
    }

    // --- per-country tables ------------------------------------------------------
    // Distinct users = distinct SHARERS on creates, distinct VIEWERS on opens.
    const byCountry = (type) => q(
        `SELECT ${ccExpr} AS cc, COUNT(*) AS n,
                COUNT(DISTINCT user_id) AS users,
                COUNT(DISTINCT problem_id) AS problems
           FROM share_events
          WHERE event_type = '${type}'
          GROUP BY cc
          ORDER BY n DESC, cc`).map((r) => ({ ...r, cc: r.cc || null }));
    const createdByCountry = byCountry('create');
    const openedByCountry = byCountry('open');
    const createdTotal = createdByCountry.reduce((s, r) => s + r.n, 0) || 1;
    const openedTotal = openedByCountry.reduce((s, r) => s + r.n, 0) || 1;
    const nCountries = new Set([...createdByCountry, ...openedByCountry]
        .map((r) => r.cc).filter(Boolean)).size;
    const unknown = {
        creates: (createdByCountry.find((r) => !r.cc) || {}).n || 0,
        opens: (openedByCountry.find((r) => !r.cc) || {}).n || 0,
    };

    // --- monthly progress --------------------------------------------------------
    const monthlyRows = q(
        `SELECT strftime('%Y-%m', created_at) AS month,
                SUM(event_type = 'create') AS creates,
                SUM(event_type = 'open')   AS opens,
                COUNT(DISTINCT CASE WHEN event_type = 'create' THEN user_id END) AS sharers,
                COUNT(DISTINCT CASE WHEN event_type = 'open'   THEN user_id END) AS viewers
           FROM share_events
          GROUP BY month
          ORDER BY month`);
    const byMonth = new Map(monthlyRows.map((r) => [r.month, r]));
    let months = [];
    if (monthlyRows.length) {
        for (const m of monthRange(monthlyRows[0].month, monthlyRows[monthlyRows.length - 1].month)) {
            const r = byMonth.get(m);
            months.push({
                month: m,
                creates: r ? r.creates : 0,
                opens: r ? r.opens : 0,
                sharers: r ? r.sharers : 0,
                viewers: r ? r.viewers : 0,
            });
        }
    }
    if (opts.months > 0) months = months.slice(-opts.months);
    let cumCreates = 0, cumOpens = 0;
    for (const m of months) {
        cumCreates += m.creates; cumOpens += m.opens;
        m.cumCreates = cumCreates; m.cumOpens = cumOpens;
    }
    const monthMax = Math.max(1, ...months.flatMap((m) => [m.creates, m.opens]));

    // --- top sharers -------------------------------------------------------------
    const sharerRows = q(
        `SELECT user_id,
                COUNT(*) AS creates,
                COUNT(DISTINCT problem_id) AS problems,
                MIN(created_at) AS firstAt,
                MAX(created_at) AS lastAt
           FROM share_events
          WHERE event_type = 'create'
          GROUP BY user_id
          ORDER BY creates DESC, lastAt DESC`);
    const sharers = opts.top > 0 ? sharerRows.slice(0, opts.top) : sharerRows;
    const nSharers = sharerRows.length;

    // Identity, country and rank for the reported sharers only.
    const ids = sharers.map((s) => `'${s.user_id.replace(/'/g, "''")}'`).join(',') || "''";
    const idBy = new Map(q(
        `SELECT user_id, display_name, country_code FROM external_ids
          WHERE user_id IN (${ids})`).map((r) => [r.user_id, r]));
    // `user_aliases` is a later migration than some mirrors, so probe before selecting.
    const aliasBy = new Map(tableColumns('user_aliases', opts.db).has('adjective')
        ? q(`SELECT user_id, adjective, color, animal FROM user_aliases
              WHERE user_id IN (${ids})`).map((r) => [r.user_id, r])
        : []);
    const hidden = new Set(tableColumns('hidden_profiles', opts.db).size
        ? q(`SELECT user_id FROM hidden_profiles WHERE user_id IN (${ids})`).map((r) => r.user_id)
        : []);

    const lbBy = new Map(q(
        `SELECT user_id, buffer, pred_rank, pred_rank_se, country FROM leaderboard
          WHERE user_id IN (${ids})`).map((r) => [r.user_id, r]));

    // The LATEST row of each kind per user — bare columns beside MAX(id) take the value
    // from that max-id row in SQLite, so this is the most recent capture, not an
    // arbitrary server. The two person-level sources are kept SEPARATE: a survey's
    // geo-IP is frozen at declaration time, while the leaderboard row's is overwritten
    // by every /report and so follows travel. rank_label/last_game_at are later
    // migrations and this report shows neither, so they are not assumed.
    const surveyBy = new Map();   // user_id -> country of the latest non-heartbeat survey
    const egdBy = new Map();      // user_id -> country of the latest EGD heartbeat
    for (const r of q(
        `SELECT user_id, country, MAX(id) AS id FROM rank_surveys
          WHERE server NOT IN ('EGD', 'AGAR') AND country IS NOT NULL AND user_id IN (${ids})
          GROUP BY user_id`)) surveyBy.set(r.user_id, r.country);
    for (const r of q(
        `SELECT user_id, country, MAX(id) AS id FROM rank_surveys
          WHERE server = 'EGD' AND country IS NOT NULL AND user_id IN (${ids})
          GROUP BY user_id`)) egdBy.set(r.user_id, r.country);

    // The user's own share events' most frequent country — the operator-only fallback
    // (production never reads share_events for a flag). Most frequent beats newest: a
    // single travelling session should not outvote a home country.
    const shareCountry = new Map();
    for (const r of q(
        `SELECT user_id, country, COUNT(*) AS n FROM share_events
          WHERE ${hasCountry ? 'country IS NOT NULL AND' : ''} user_id IN (${ids})
          GROUP BY user_id, country ORDER BY user_id, n DESC`)) {
        if (!shareCountry.has(r.user_id)) shareCountry.set(r.user_id, r.country);
    }

    // normalizeCountry / RANK_START_MIN / BUFFER_CAP / parseBuffer come from the
    // production bundle, so the country filter and the rank gate cannot drift. The
    // problem params are only needed to actually compute a rank.
    const mod = await loadRankModule();
    const haveParams = fs.existsSync(PARAMS_BIN);
    if (!haveParams) {
        console.warn(`WARNING: missing ${PARAMS_BIN} — no rank column. `
            + 'Run `./.venv/bin/python3 -m pipeline.package_for_pwa` for ranks.\n');
    }
    const params = haveParams ? loadProblemParamsLocal() : null;

    // The two person-level country captures, cross-checked over the WHOLE database
    // (not just the listed sharers): a user who holds both a leaderboard row and at
    // least one older survey, with real ISO codes on both sides, and they differ. The
    // two are separate edge captures — the leaderboard's follows travel, the survey's
    // is frozen at declaration — so a disagreement is a fact about the data, not a bug
    // to resolve silently.
    const lbCountryAll = new Map(q(
        `SELECT user_id, country FROM leaderboard WHERE country IS NOT NULL`)
        .map((r) => [r.user_id, r.country]));
    const svCountryAll = new Map(q(
        `SELECT user_id, country, MAX(id) AS id FROM rank_surveys
          WHERE server NOT IN ('EGD', 'AGAR') AND country IS NOT NULL GROUP BY user_id`)
        .map((r) => [r.user_id, r.country]));
    const ccBoth = [], ccDisagree = [];
    for (const [uid, lbCc0] of lbCountryAll) {
        const lbCc = mod.normalizeCountry(lbCc0);
        const svCc = mod.normalizeCountry(svCountryAll.get(uid));
        if (!lbCc || !svCc) continue;
        ccBoth.push(uid);
        if (lbCc !== svCc) ccDisagree.push({ userId: uid, leaderboard: lbCc, survey: svCc });
    }
    const countryConsistency = { both: ccBoth.length, disagree: ccDisagree.length, users: ccDisagree };

    // Rank from the stored buffer; else REBUILD it the way buildRowStateFromReports
    // seeds a pruned row (last BUFFER_CAP novel attempts, newest first by id); else
    // fall back to the stored pred_rank if the row has one.
    const needRebuild = sharers
        .filter((s) => !(lbBy.get(s.user_id) || {}).buffer)
        .map((s) => `'${s.user_id.replace(/'/g, "''")}'`);
    const rebuiltBy = new Map();
    if (haveParams && needRebuild.length) {
        const seed = q(
            `SELECT user_id, p, s, t FROM (
                SELECT user_id, problem AS p, success AS s, time_used AS t,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id DESC) AS rn
                  FROM tsumego_reports
                 WHERE is_repeat = 0 AND user_id IN (${needRebuild.join(',')})
             ) WHERE rn <= ${mod.BUFFER_CAP} ORDER BY user_id, rn`);
        for (const r of seed) {
            if (!rebuiltBy.has(r.user_id)) rebuiltBy.set(r.user_id, []);
            rebuiltBy.get(r.user_id).push({ p: r.p, s: r.s ? 1 : 0, t: Number.isFinite(r.t) ? r.t : null });
        }
    }
    const rankOf = (uid) => {
        const stored = lbBy.get(uid) || {};
        if (!haveParams) return { rank: null, se: null, usable: null, source: null };
        const attempts = stored.buffer ? mod.parseBuffer(stored.buffer) : (rebuiltBy.get(uid) || []);
        const est = mod.computeRankBoth(attempts, params);
        if (est && Number.isFinite(est.onlymoveRank) && est.usable > 0) {
            return { rank: est.onlymoveRank, se: est.se, usable: est.usable,
                source: stored.buffer ? 'stored buffer' : 'rebuilt' };
        }
        if (Number.isFinite(stored.pred_rank)) {
            return { rank: stored.pred_rank, se: stored.pred_rank_se,
                usable: est ? est.usable : null, source: 'stored pred_rank' };
        }
        return { rank: null, se: null, usable: est ? est.usable : 0, source: null };
    };

    const top = sharers.map((s) => {
        const ident = idBy.get(s.user_id) || {};
        const aliasRow = aliasBy.get(s.user_id);
        const alias = aliasRow ? `${aliasRow.adjective} ${aliasRow.color} ${aliasRow.animal}` : null;
        const countrySources = {
            egd: egdBy.get(s.user_id) || null,
            linked: ident.country_code || null,
            leaderboard: (lbBy.get(s.user_id) || {}).country || null,
            survey: surveyBy.get(s.user_id) || null,
            shareEvents: shareCountry.get(s.user_id) || null,
        };
        const { country, source: countrySource } = resolveCountry(mod, countrySources);
        // The two person-level edge-IP captures, compared with the shipped filter so a
        // sentinel on one side is not read as a conflict.
        const lbCc = mod.normalizeCountry(countrySources.leaderboard);
        const svCc = mod.normalizeCountry(countrySources.survey);
        const rk = rankOf(s.user_id);
        return {
            userId: s.user_id,
            name: ident.display_name || alias || null,
            nameSource: ident.display_name ? 'linked' : (alias ? 'alias' : null),
            alias,
            linked: !!ident.display_name,
            hidden: hidden.has(s.user_id),
            creates: s.creates,
            problems: s.problems,
            firstAt: s.firstAt,
            lastAt: s.lastAt,
            country,
            countrySource,
            countrySources,
            countryConflict: !!(lbCc && svCc && lbCc !== svCc),
            rank: rk.rank,
            se: rk.se,
            usable: rk.usable,
            rankSource: rk.source,
            belowGate: rk.rank !== null && Number.isFinite(rk.usable)
                && rk.usable < mod.RANK_START_MIN,
        };
    });

    const result = {
        generatedAt: new Date().toISOString(),
        db: opts.db,
        totals: { ...totals, countries: nCountries, unknownCountry: unknown },
        createdByCountry,
        openedByCountry,
        months,
        countrySourceConsistency: countryConsistency,
        topSharers: top,
    };
    if (opts.json) { console.log(JSON.stringify(result, null, 2)); return; }

    // --- render ------------------------------------------------------------------
    console.log('=== Share-link activity (share_events) ===');
    console.log(`Events:            ${int(totals.events)}  `
        + `(${int(totals.creates)} created, ${int(totals.opens)} opened)`);
    console.log(`Range:             ${day(totals.firstAt)} → ${day(totals.lastAt)}`);
    console.log(`Sharers / viewers: ${int(totals.sharers)} / ${int(totals.viewers)}`);
    console.log(`Countries seen:    ${int(nCountries)}`
        + (unknown.creates || unknown.opens
            ? `   (no geo-IP: ${int(unknown.creates)} creates, ${int(unknown.opens)} opens)` : ''));
    console.log(`Problems shared:   ${int(totals.sharedProblems)}`);
    console.log(`Country sources:   ${int(countryConsistency.both)} user(s) hold BOTH a leaderboard`
        + ` and a survey geo-IP; ${int(countryConsistency.disagree)} disagree`);

    function countryTable(title, rows, total, unit) {
        console.log(`\n=== ${title} ===`);
        const shown = opts.countries > 0 ? rows.slice(0, opts.countries) : rows;
        const nameW = Math.max(7, ...shown.map((r) => (r.cc ? r.cc.length + 3 : UNKNOWN.length)));
        const maxN = Math.max(1, ...shown.map((r) => r.n));
        console.log(pad('Country', nameW + 2) + padl('Count', 8) + padl('%', 7)
            + '  ' + pad(unit, 10) + padl('Problems', 10) + '  bar');
        for (const r of shown) {
            const label = r.cc ? `${r.cc} ${mod.flag(r.cc)}` : UNKNOWN;
            console.log(pad(label, nameW + 2)
                + padl(int(r.n), 8) + padl(pct(r.n, total), 7) + '  '
                + pad(int(r.users), 10) + padl(int(r.problems || 0), 10) + '  '
                + bar(r.n, maxN, 24));
        }
        const rest = rows.slice(shown.length);
        if (rest.length) {
            console.log(`(${rest.length} more countr${rest.length === 1 ? 'y' : 'ies'} below the cutoff, `
                + `${int(rest.reduce((s, r) => s + r.n, 0))} more ${unit.toLowerCase()})`);
        }
    }
    countryTable('Shares created, per country (sharer location)',
        createdByCountry, createdTotal, 'Sharers');
    countryTable('Shares opened, per country (viewer location)',
        openedByCountry, openedTotal, 'Viewers');

    console.log('\n=== Monthly progress (UTC months; both bars share one scale; cum. = within window) ===');
    console.log(pad('Month', 10) + padl('Created', 9) + '  ' + pad('', 22)
        + padl('Opened', 8) + '  ' + pad('', 22)
        + padl('Sharers', 9) + padl('Viewers', 9) + padl('cum. cr', 9) + padl('cum. op', 9));
    for (const m of months) {
        console.log(pad(m.month, 10) + padl(int(m.creates), 9) + '  ' + pad(bar(m.creates, monthMax, 22), 22)
            + padl(int(m.opens), 8) + '  ' + pad(bar(m.opens, monthMax, 22), 22)
            + padl(int(m.sharers), 9) + padl(int(m.viewers), 9)
            + padl(int(m.cumCreates), 9) + padl(int(m.cumOpens), 9));
    }

    console.log('\n=== Top sharers (by links created) ===');
    if (!top.length) {
        console.log('(no create events)');
        return;
    }
    const cell = (p) => (p.name || 'Anonymous') + (p.hidden ? ' H' : '');
    const nameW = Math.max(12, ...top.map((p) => cell(p).length));
    const srcW = Math.max(11, ...top.map((p) => (p.countrySource || '—').length));
    const rankW = haveParams ? 15 : 0;
    console.log(pad('#', 4) + pad('Name', nameW + 2) + (opts.noUserIds ? '' : pad('user_id', 10))
        + (haveParams ? pad('Rank', rankW) : '') + pad('Ctry', 8) + pad('country src', srcW + 2)
        + padl('Created', 9) + padl('Probs', 7) + '  ' + pad('First', 12) + pad('Last', 12));
    top.forEach((p, i) => {
        // `!` marks a sharer whose two independent person-level captures disagree —
        // the leaderboard row's geo-IP and the latest survey's (see countrySources in
        // --json); the resolved country above follows the chain, it does not average.
        const cc = p.country ? `${p.country} ${mod.flag(p.country)}${p.countryConflict ? ' !' : ''}` : '—';
        console.log(pad(i + 1, 4) + pad(cell(p), nameW + 2)
            + (opts.noUserIds ? '' : pad(p.userId, 10))
            + (haveParams ? pad(rankStr(mod, p.rank) + (p.belowGate ? '*' : ''), rankW) : '')
            + pad(cc, 8) + pad(p.countrySource || '—', srcW + 2)
            + padl(int(p.creates), 9) + padl(int(p.problems), 7) + '  '
            + pad(day(p.firstAt), 12) + pad(day(p.lastAt), 12));
    });

    const conflicted = top.filter((p) => p.countryConflict);
    console.log('\nNotes:');
    console.log('  create = a user generated a share link (user_id = the SHARER, country = the');
    console.log('  sharer\'s edge geo-IP). open = someone arrived via a shared link (user_id =');
    console.log('  the VIEWER, country = the viewer\'s edge geo-IP). The two populations differ,');
    console.log('  and no share token is stored, so an open cannot be attributed to a create —');
    console.log('  the counts above are not a conversion funnel.');
    console.log('  Sharer country = first non-empty of: EGD federation code → linked identity');
    console.log('  (external_ids) → the leaderboard row\'s geo-IP → a survey\'s geo-IP → the');
    console.log('  user\'s own share events. "country src" names the winning source; the');
    console.log('  per-country tables instead use each event\'s own edge geo-IP.');
    if (conflicted.length) {
        console.log(`  ! marks the ${conflicted.length} listed sharer(s) whose leaderboard geo-IP and latest survey`);
        console.log('  geo-IP disagree — see --json for the pair; the country shown follows the chain.');
    }
    if (countryConsistency.disagree) {
        const examples = countryConsistency.users.slice(0, 5)
            .map((u) => `${u.userId} ${u.leaderboard}/${u.survey}`).join(', ');
        console.log(`  Database-wide, ${countryConsistency.disagree} of ${countryConsistency.both} users who hold both captures`);
        console.log(`  disagree (${examples}${countryConsistency.users.length > 5 ? ', …' : ''}); they are independent edge`);
        console.log('  captures, not two views of one stored value.');
    } else {
        console.log('  Database-wide, every user with both a leaderboard and a survey geo-IP agrees.');
    }
    if (haveParams) {
        console.log(`  Rank = production's untimed shown rank, from the stored leaderboard buffer or`);
        console.log(`  rebuilt from tsumego_reports when the row was pruned. * = below production's`);
        console.log(`  RANK_START_MIN (${mod.RANK_START_MIN}) usable attempts, whose gate is dropped here.`);
        console.log('  H = hidden_profiles opt-out (this operator report marks it, does not apply it).');
    } else {
        console.log('  No rank column: pwa/tsumegos/problem_params.bin is missing (run package_for_pwa).');
    }
    console.log('  Country is Cloudflare\'s edge geo-IP — where the request came from, not a');
    console.log('  user-declared nationality.');
}

await main();
