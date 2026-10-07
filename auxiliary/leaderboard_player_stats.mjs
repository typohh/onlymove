#!/usr/bin/env node
// auxiliary/leaderboard_player_stats.mjs
//
// Read-only report: for every player the /leaderboard page LISTS — active in the trailing
// week and at or above the RANK_START_MIN list-entry gate, linked external account or
// not — in the page's default order (strongest first by the untimed onlymove rank), print
//   - their country      (the /leaderboard page's flag, on every row incl. anonymous)
//   - novel (is_repeat = 0) 9x9 attempt count
//   - novel 13x13 attempt count
//   - median solve time over their novel attempts, and its standard deviation (raw:
//     solve time is right-skewed and a left-open tab lands in the tail, so the sd is
//     an outlier-sensitive spread, unlike the median beside it)
//   - attempts per local day over the past 7 days
//   - their onlymove rank    (the untimed SHOWN rank, decimal)
//   - their est. EGF rank    (the time-adjusted EGF-coupled rank, decimal)
//   - their real EGF rank    (from the verified 'EGD' GoR heartbeat)
//   - with --aga, the same pair in AGA units
//   - how many rank SURVEYS they have declared — human self-reports only, since the
//     weekly machine heartbeats are written for them and would else swamp the count
//
// This is a report on what the SYSTEM SHOWS each player, not a calibration check:
// every rank here is production's SHOWN rank (estimateWindowRankFromFlat's
// displayRank — the posterior mean less K_DISPLAY excess-se, over the rank-dependent
// SE floor), the same quantity the front-page badge, the stored pred_rank and the
// /leaderboard page's onlymove column all carry. It therefore inherits that value's
// deliberate conservatism while a posterior is wide, which is a display policy rather
// than a model error — so do NOT read the est-vs-real columns here as the mapping's
// bias. ablations/check_mapped_vs_declared_rank.py is the script for that: it scores
// the calibrated estimate against every declared and verified rating.
//
// The ranks are computed BIT-IDENTICALLY to production: this script esbuild-bundles
// pwa/src/leaderboard.js itself (so the curve/coupling/time-adjust data and every
// helper — RANK_START_MIN, the EGF/AGA couplings, the personal time dock, the
// rank-dependent SE floor, the verified-activity gate — are the shipped ones) and
// runs the shared window estimator over each row's STORED buffer (the last <=250
// novel attempts, the exact input the Worker's computeRank uses). est. EGF/AGA
// re-apply the median-solve-time dock to that same shown rank and map through the
// server's coupling — the frame a real GoR / AGA rating natively speaks — in the
// same order readLeaderboardRows does. All on the one internal rank scale, where 0
// is the 1k/1d boundary and every band is one unit wide (integers are band EDGES).
//
// Display identity comes from the same three tables the /leaderboard page stitches:
// the rank buffer + edge geo-IP country off `leaderboard`, the name + country off
// the `external_ids` binding, and the verified ratings off the
// latest-per-(user, server) machine heartbeats in `rank_surveys` ('EGD' rank = raw
// GoR digits, 'AGAR' rank = raw AGA rating). Human survey declarations are NOT read
// as ratings — they are calibration input, not ground truth about a player's
// federation rating — but their country IS read, as the page's last-resort flag
// fallback. Unlike the page, an operator report does not apply the hidden_profiles
// opt-out: a hidden player still shows their name and country here.
//
// Data source: db/d1.sqlite (the local D1 mirror) + the current local
// pwa/tsumegos/{problem_params.bin, rating_to_rank.json, rank_map.json,
// population_constants.json}. Run `./.venv/bin/python3 -m pipeline.fetch_database`
// first so the leaderboard rows + tsumego_reports are fresh.
//
// Run with --help for the flags. The activity window is measured from NOW, as the
// Worker does, so a stale mirror under-lists — the staleness warning below says when.

import fs from 'node:fs';
import {
    DB_PATH, PARAMS_BIN, STALE_WARN_MS,
    loadRankModule, loadProblemParamsLocal, sqliteJson, tableColumns,
    rankStr, durationStr,
} from './leaderboard_rank_lib.mjs';

// --- CLI ------------------------------------------------------------------------
const USAGE = `Per-player leaderboard report: country, attempt counts, solve time, the last
7 days' activity, and the rank onlymove SHOWS each player beside their verified EGF/AGA
rating. Not a calibration check — see ablations/check_mapped_vs_declared_rank.py.

Usage: node auxiliary/leaderboard_player_stats.mjs [options]

  --inactive        also list the leaderboard table's rows idle for over a week
                    (kept by its longer retention, but not shown on the page),
                    merged into the same order and marked ‡
  --aga             add the est. AGA and real AGA columns (off by default)
  --user <id>       limit to this user id; repeatable. A named user is shown even
                    when inactive or below the list-entry gate
  --json            machine-readable JSON: { players }
  --help, -h        show this message

Reads db/d1.sqlite and the local pwa/tsumegos/ data files; run
\`./.venv/bin/python3 -m pipeline.fetch_database\` first for fresh numbers.`;

const args = process.argv.slice(2);
let asJson = false;
let withAga = false;
let withInactive = false;
const onlyUsers = new Set();
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--help' || a === '-h') { console.log(USAGE); process.exit(0); }
    else if (a === '--json') asJson = true;
    else if (a === '--aga') withAga = true;
    else if (a === '--inactive') withInactive = true;
    else if (a === '--user') {
        const uid = args[++i];
        if (uid === undefined) { console.error('--user needs a user id\n\n' + USAGE); process.exit(1); }
        onlyUsers.add(uid);
    } else { console.error(`unknown arg: ${a}\n\n${USAGE}`); process.exit(1); }
}

// The production rank bundle, the params reader and the sqlite/format helpers moved to
// ./leaderboard_rank_lib.mjs, shared with auxiliary/top_users_report.mjs.


function pad(s, n) { s = String(s); return s.length >= n ? s : s + ' '.repeat(n - s.length); }
function padl(s, n) { s = String(s); return s.length >= n ? s : ' '.repeat(n - s.length) + s; }

// --- rolling 7-day columns -------------------------------------------------------
// The 7 days ending TODAY inclusive, in LOCAL time, so each column matches "today"
// for whoever runs it (the machine here is EEST = the Finnish tournament TZ) and the
// window is the same width every day of the week. created_at is stored as ISO-8601
// UTC, so the per-day SQL groups on date(created_at,'localtime') to bucket by the
// same local calendar day.
const DAYW = 7;
const WINDOW_DAYS = 7;
const WEEKDAY = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
function localDateStr(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${m}-${day}`;   // 'YYYY-MM-DD', lexically comparable + matches SQLite date()
}
function pastSevenDays() {
    const now = new Date();
    const days = [];
    for (let i = WINDOW_DAYS - 1; i >= 0; i--) {
        // Day-of-month arithmetic on the local calendar day; the Date constructor
        // normalizes across month/year boundaries.
        const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - i);
        days.push({ date: localDateStr(d), label: `${WEEKDAY[(d.getDay() + 6) % 7]} ${d.getDate()}` });
    }
    return days;
}

async function main() {
    if (!fs.existsSync(DB_PATH)) {
        console.error(`missing ${DB_PATH} — run: ./.venv/bin/python3 -m pipeline.fetch_database`);
        process.exit(1);
    }
    if (!fs.existsSync(PARAMS_BIN)) {
        console.error(`missing ${PARAMS_BIN} — run: ./.venv/bin/python3 -m pipeline.package_for_pwa`);
        process.exit(1);
    }

    const [{ m: latest }] = sqliteJson('SELECT MAX(created_at) AS m FROM tsumego_reports');
    const ageMs = Date.now() - Date.parse(latest || 0);
    if (!Number.isFinite(ageMs) || ageMs > STALE_WARN_MS) {
        console.warn(`WARNING: local mirror's newest report is ${latest || 'unknown'} `
            + `(${Number.isFinite(ageMs) ? Math.round(ageMs / 3600000) : '?'}h old). `
            + 'Run pipeline.fetch_database first for fresh numbers.\n');
    }

    const mod = await loadRankModule();
    const params = loadProblemParamsLocal();

    // `country` here is the edge geo-IP stamped on every /report — the page's universal
    // flag fallback, and the only country an anonymous row has.
    // The page's activity window: readLeaderboardRows' `last_active >= now - 7 days`.
    // The table's retention is longer than that, so the filter is not implied by the
    // cron's pruning.
    const ACTIVE_DAYS = 7;
    const activeSince = new Date(Date.now() - ACTIVE_DAYS * 86400000).toISOString();
    const lbRows = sqliteJson('SELECT user_id, buffer, country, last_active FROM leaderboard');

    // Display name + country off the `external_ids` binding, snapshotted from
    // EGD/AGA when the user linked — the same source readLeaderboardRows stitches.
    const idRows = sqliteJson('SELECT user_id, display_name, country_code FROM external_ids');
    const idBy = new Map(idRows.map((r) => [r.user_id, r]));
    const linkedIds = new Set(idRows.map((r) => r.user_id));

    // ONE latest-per-(user, server) pass over rank_surveys, exactly as
    // readLeaderboardRows does. Only the MACHINE heartbeats are read as RATINGS —
    // 'EGD' rank = raw GoR digits, 'AGAR' rank = raw AGA rating; a human survey row
    // (OGS/KGS/Fox/…) contributes nothing but its geo-IP country, the page's
    // last-resort flag fallback. rank_label/last_game_at arrived with
    // pwa/migrations/2026-07-21_rank_surveys_verified.sql, so they're probed first.
    const surveyCols = tableColumns('rank_surveys');
    const hasVerifiedCols = surveyCols.has('rank_label') && surveyCols.has('last_game_at');
    const verifiedSelect = hasVerifiedCols ? 'rank_label, last_game_at,' : '';
    const hbRows = sqliteJson(
        `SELECT user_id, server, rank, country, ${verifiedSelect} MAX(id)
           FROM rank_surveys GROUP BY user_id, server`
    );
    const egdBy = new Map(), agaBy = new Map();
    // The 'EGD' heartbeat's country is the EGD federation code (most reliable, it comes
    // from EGD); every other server's is Cloudflare geo-IP.
    const egdCountryBy = new Map(), surveyCountryBy = new Map();
    for (const h of hbRows) {
        if (h.server === 'EGD') {
            egdBy.set(h.user_id, h);
            if (h.country) egdCountryBy.set(h.user_id, h.country);
        } else if (h.server === 'AGAR') {
            agaBy.set(h.user_id, h);
        } else if (h.country && !surveyCountryBy.has(h.user_id)) {
            surveyCountryBy.set(h.user_id, h.country);
        }
    }

    // Default scope: the rows the page lists — active in the window, linked or not. The
    // list-entry gate needs the rank, so it is applied after the per-row computation.
    const isLinked = (uid) => linkedIds.has(uid) || egdBy.has(uid) || agaBy.has(uid);
    // Naming a user with --user IS the request to see them, so it bypasses both the
    // activity window and the gate. A named user whose row the cron has pruned gets its
    // buffer rebuilt from tsumego_reports below, exactly as buildRowStateFromReports does.
    let rows;
    if (onlyUsers.size) {
        const byId = new Map(lbRows.map((r) => [r.user_id, { ...r, rebuilt: false }]));
        rows = [...onlyUsers].map((uid) => byId.get(uid)
            || { user_id: uid, buffer: null, country: null, last_active: null, rebuilt: true });
    } else {
        rows = lbRows.filter((r) => withInactive || (r.last_active && r.last_active >= activeSince))
            .map((r) => ({ ...r, rebuilt: false }));
    }
    if (!rows.length) { console.log('no matching players'); return; }

    const idList = rows.map((r) => `'${r.user_id.replace(/'/g, "''")}'`).join(',');

    // Rebuild the pruned rows' buffers: the last BUFFER_CAP novel attempts, newest
    // first — the same set (and order) buildRowStateFromReports seeds a row from.
    const rebuildIds = rows.filter((r) => r.rebuilt).map((r) => `'${r.user_id.replace(/'/g, "''")}'`);
    if (rebuildIds.length) {
        const seed = sqliteJson(
            `SELECT user_id, p, s, t FROM (
                SELECT user_id, problem AS p, success AS s, time_used AS t,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY id DESC) AS rn
                  FROM tsumego_reports
                 WHERE is_repeat = 0 AND user_id IN (${rebuildIds.join(',')})
             ) WHERE rn <= ${mod.BUFFER_CAP} ORDER BY user_id, rn`
        );
        const seedBy = new Map();
        for (const s of seed) {
            if (!seedBy.has(s.user_id)) seedBy.set(s.user_id, []);
            seedBy.get(s.user_id).push({ p: s.p, s: s.s ? 1 : 0, t: Number.isFinite(s.t) ? s.t : null });
        }
        for (const r of rows) if (r.rebuilt) r.buffer = JSON.stringify(seedBy.get(r.user_id) || []);
    }

    // Novel attempt counts by size (whole history, is_repeat = 0).
    const counts = sqliteJson(
        `SELECT user_id, size, COUNT(*) AS n
           FROM tsumego_reports
          WHERE is_repeat = 0 AND user_id IN (${idList})
          GROUP BY user_id, size`
    );
    const countBy = new Map();   // user_id -> {9, 13}
    for (const c of counts) {
        if (!countBy.has(c.user_id)) countBy.set(c.user_id, { 9: 0, 13: 0 });
        countBy.get(c.user_id)[c.size] = c.n;
    }

    // Median solve time over novel attempts (time_used > 0 ms; <=0 = untimed, dropped
    // — same convention the rank's weightedMedianOverWindow uses), plus the standard
    // deviation over that SAME population. SQLite ships no stddev(), so the spread CTE
    // returns the first two moments and JS finishes the sample sd; the seconds unit
    // keeps the sum of squares far from double's exact-integer range, and solve time is
    // spread widely enough that the E[t²]−E[t]² form loses no significant digits.
    const meds = sqliteJson(
        `WITH novel AS (
            SELECT user_id, time_used / 1000.0 AS t
              FROM tsumego_reports
             WHERE is_repeat = 0 AND time_used > 0
               AND user_id IN (${idList})
         ), ranked AS (
            SELECT user_id, t,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY t) AS rn,
                   COUNT(*)     OVER (PARTITION BY user_id)            AS cnt
              FROM novel
         ), med AS (
            SELECT user_id, AVG(t) AS median_s, MAX(cnt) AS n_timed
              FROM ranked
             WHERE rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
             GROUP BY user_id
         ), spread AS (
            SELECT user_id, AVG(t) AS mean_s, AVG(t * t) AS mean_sq_s
              FROM novel
             GROUP BY user_id
         )
         SELECT med.user_id, med.median_s, med.n_timed, spread.mean_s, spread.mean_sq_s
           FROM med JOIN spread USING (user_id)`
    );
    const medBy = new Map(meds.map((m) => {
        const n = m.n_timed;
        const varS = n > 1
            ? Math.max(0, m.mean_sq_s - m.mean_s * m.mean_s) * (n / (n - 1))   // sample sd
            : NaN;
        return [m.user_id, {
            medianMs: m.median_s * 1000,
            sdMs: Number.isFinite(varS) ? Math.sqrt(varS) * 1000 : NaN,
            nTimed: n,
        }];
    }));

    // How many rank surveys the player has DECLARED: every rank_surveys row that isn't a
    // machine heartbeat. Rows, not distinct servers — a re-declaration on the same server
    // is another value reported, and re-declaring is common. The heartbeats ('EGD',
    // 'AGAR') are excluded because the system writes them weekly on the player's behalf,
    // so counting them would measure how long they have been linked, not what they told us.
    const surveyCounts = sqliteJson(
        `SELECT user_id, COUNT(*) AS n
           FROM rank_surveys
          WHERE server NOT IN ('EGD', 'AGAR') AND user_id IN (${idList})
          GROUP BY user_id`
    );
    const surveyCountBy = new Map(surveyCounts.map((s) => [s.user_id, s.n]));

    // Attempts (all reports, incl. repeats) per LOCAL calendar day, last 7 days.
    const weekDays = pastSevenDays();
    const weekStart = weekDays[0].date;
    const perDayRows = sqliteJson(
        `SELECT user_id, date(created_at, 'localtime') AS d, COUNT(*) AS n
           FROM tsumego_reports
          WHERE date(created_at, 'localtime') >= '${weekStart}'
            AND user_id IN (${idList})
          GROUP BY user_id, d`
    );
    const perDayBy = new Map();   // user_id -> { 'YYYY-MM-DD': n }
    for (const dr of perDayRows) {
        if (!perDayBy.has(dr.user_id)) perDayBy.set(dr.user_id, {});
        perDayBy.get(dr.user_id)[dr.d] = dr.n;
    }

    const RANKED_MIN = mod.RANK_START_MIN;   // production's list-entry / rank gate
    let out = rows.map((r) => {
        const { onlymoveRank, egfRank, agaRank, se, usable } =
            mod.computeRankBoth(mod.parseBuffer(r.buffer), params);
        const c = countBy.get(r.user_id) || { 9: 0, 13: 0 };
        const md = medBy.get(r.user_id) || {};
        const perDayObj = perDayBy.get(r.user_id) || {};
        const perDay = {};
        let weekTotal = 0;
        for (const d of weekDays) { const n = perDayObj[d.date] || 0; perDay[d.date] = n; weekTotal += n; }

        // Real (verified) strength from the machine heartbeats, in the same rank-scalar
        // frame as the estimates. `stale` = production would WITHHOLD this rating as out
        // of date (EGF activity criterion); shown with a * here, not dropped.
        const egd = egdBy.get(r.user_id) || null;
        const aga = agaBy.get(r.user_id) || null;
        const gor = egd ? parseInt(egd.rank, 10) : NaN;
        const agaRating = aga ? Number(aga.rank) : NaN;   // legacy label rows -> NaN
        const realEgfRank = Number.isFinite(gor) && gor > 0 ? mod.gorToRank(gor) : null;
        const realAgaRank = Number.isFinite(agaRating) && Math.abs(agaRating) >= 1
            ? mod.agaRatingToRank(agaRating) : null;
        const staleOf = (rec, tierFallback) => (!rec || !hasVerifiedCols ? false
            : !mod.isVerifiedActive(mod.verifiedTierRank(rec.rank_label, tierFallback), rec.last_game_at));

        // Country, on the page's own fallback chain: EGD federation code (from the 'EGD'
        // heartbeat, the most reliable) -> the binding (EGD, frozen at link time) ->
        // the leaderboard row's edge geo-IP (stamped on every /report — the universal
        // fallback that gives ANY active player, anonymous included, a flag) -> an older
        // survey's geo-IP. A rebuilt (pruned) row has no leaderboard geo-IP to offer.
        const ident = idBy.get(r.user_id) || {};
        const country = egdCountryBy.get(r.user_id) || ident.country_code || r.country
            || surveyCountryBy.get(r.user_id) || null;

        return {
            userId: r.user_id, name: ident.display_name || r.user_id,
            country,
            linked: isLinked(r.user_id),
            registered: linkedIds.has(r.user_id), rebuilt: !!r.rebuilt,
            lastActive: r.last_active || null,
            inactive: !r.last_active || r.last_active < activeSince,
            gor: Number.isFinite(gor) ? gor : null,
            agaRating: Number.isFinite(agaRating) ? agaRating : null,
            novel9: c[9] || 0, novel13: c[13] || 0,
            medianMs: Number.isFinite(md.medianMs) ? md.medianMs : null,
            sdMs: Number.isFinite(md.sdMs) ? md.sdMs : null,
            nTimed: md.nTimed || 0,
            perDay, weekTotal,
            onlymoveRank, estEgfRank: egfRank, realEgfRank,
            estAgaRank: agaRank, realAgaRank,
            surveys: surveyCountBy.get(r.user_id) || 0,
            egfStale: realEgfRank !== null && staleOf(egd, realEgfRank),
            agaStale: realAgaRank !== null && staleOf(aga, realAgaRank),
            se, usableNovel: usable,
        };
    });
    // The page's list-entry gate (resolveRowRank: no rank below RANK_START_MIN usable
    // attempts). Sub-gate active players are only tallied, as the page's footer does.
    let belowGate = 0;
    if (!onlyUsers.size) {
        const listed = out.filter((p) => p.usableNovel >= RANKED_MIN && p.onlymoveRank !== null);
        belowGate = out.length - listed.length;
        out = listed;
        if (!out.length) { console.log(`no listed players (${belowGate} active below the gate)`); return; }
    }
    // The page's default order: strongest first by the untimed onlymove rank, one list
    // regardless of link status; an unranked (null) --user row sinks to the bottom.
    out.sort((a, b) => (b.onlymoveRank ?? -Infinity) - (a.onlymoveRank ?? -Infinity));

    if (asJson) { console.log(JSON.stringify({ players: out, belowGate }, null, 2)); return; }

    const nameCell = (p) => p.name + (p.rebuilt ? ' †' : p.inactive ? ' ‡' : '');
    // Widen to the longest name rather than truncating: the country column sits right
    // after it, so a fixed width would run the two together on a long name.
    const W = {
        name: Math.max(20, ...out.map((p) => nameCell(p).length + 1)),
        cc: 5, s9: 8, s13: 8, med: 9, sd: 9, n: 6, wk: 7, om: 15, est: 15, real: 16, srv: 8,
    };
    const dayHeader = weekDays.map((d) => padl(d.label, DAYW)).join('');
    const dayCells = (p) => weekDays.map((d) => padl((p.perDay[d.date] || 0).toLocaleString('en-US'), DAYW)).join('');
    const realStr = (v, stale) => (Number.isFinite(v) ? rankStr(mod, v) + (stale ? ' *' : '') : '—');
    const ruleW = W.name + W.cc + W.s9 + W.s13 + W.med + W.sd + W.n + 2 + weekDays.length * DAYW + W.wk + 2
        + W.om + W.est + W.real + (withAga ? W.est + W.real : 0) + W.srv;
    console.log(
        pad('Player', W.name) + pad('ctry', W.cc) + padl('9×9', W.s9) + padl('13×13', W.s13)
        + padl('median', W.med) + padl('sd', W.sd) + padl('buf', W.n) + '  ' + dayHeader + padl('wk', W.wk) + '  '
        + pad('onlymove', W.om) + pad('est. EGF', W.est) + pad('real EGF', W.real)
        + (withAga ? pad('est. AGA', W.est) + pad('real AGA', W.real) : '')
        + padl('surveys', W.srv));
    console.log('─'.repeat(ruleW));
    for (const p of out) {
        console.log(
            pad(nameCell(p), W.name)
            + pad(p.country || '—', W.cc)
            + padl(p.novel9.toLocaleString('en-US'), W.s9)
            + padl(p.novel13.toLocaleString('en-US'), W.s13)
            + padl(durationStr(p.medianMs), W.med)
            + padl(durationStr(p.sdMs), W.sd)
            + padl(p.usableNovel, W.n) + '  '
            + dayCells(p) + padl(p.weekTotal.toLocaleString('en-US'), W.wk) + '  '
            + pad(rankStr(mod, p.onlymoveRank), W.om)
            + pad(rankStr(mod, p.estEgfRank), W.est)
            + pad(realStr(p.realEgfRank, p.egfStale), W.real)
            + (withAga ? pad(rankStr(mod, p.estAgaRank), W.est)
                + pad(realStr(p.realAgaRank, p.agaStale), W.real) : '')
            + padl(p.surveys || '—', W.srv));
    }
    const nLinked = out.filter((p) => p.linked).length;
    console.log(
        (onlyUsers.size
            ? `\n${out.length} named player(s), shown regardless of activity and the list-entry gate. `
            : `\n${out.length} listed player(s) (${nLinked} with a linked external account`
                + (withInactive ? `, ${out.filter((p) => p.inactive).length} idle ‡` : '')
                + `), ${withInactive ? 'on the leaderboard table' : `active since ${activeSince.slice(0, 10)}`}`
                + ` with ≥ ${RANKED_MIN} usable attempts; `
                + `${belowGate} more below that gate, not listed. `)
        + `"novel" = distinct-attempt (is_repeat=0) count over all history; `
        + `median / sd = median and standard deviation of solve time over novel timed attempts `
        + `(the sd is raw, so a left-open tab inflates it — solve time is right-skewed); `
        + `buf = still-in-corpus attempts in the rank buffer.\n`
        + `ctry = the flag the /leaderboard page would show: EGD federation code → the binding → the row's edge geo-IP → survey geo-IP.\n`
        + `${weekDays.map((d) => d.label).join(' / ')} = attempts (all reports, incl. repeats) that LOCAL day (the 7 days ${weekStart}→today); wk = that window's total.\n`
        + `onlymove  = the untimed rank the player is SHOWN (the front-page badge and the stored pred_rank).\n`
        + `est. EGF${withAga ? '/AGA' : ''} = that same shown rank, time-adjusted and mapped through the server's coupling (the frame a real rating speaks).\n`
        + `real EGF${withAga ? '/AGA' : ''} = the verified machine heartbeat — the EGD GoR ((GoR−2000)/100)${withAga ? ' / the AGA rating' : ''}.\n`
        + `surveys = rank declarations the player has SELF-REPORTED (rows, so a re-declaration counts again); the weekly 'EGD'/'AGAR' machine heartbeats are excluded.\n`
        + `The shown rank is deliberately conservative while a posterior is wide, so est-vs-real here is NOT the mapping's bias — `
        + `ablations/check_mapped_vs_declared_rank.py measures that.\n`
        + `Order = the /leaderboard page's default (onlymove units), linked or not; a player with no linked account has no real EGF${withAga ? '/AGA' : ''}.`
        + (out.some((p) => p.inactive && !p.rebuilt) ? `\n‡ = idle since before ${activeSince.slice(0, 10)}: still in the leaderboard table, but not shown on the page.` : '')
        + (out.some((p) => p.rebuilt) ? `\n† = no leaderboard row (pruned as idle); buffer rebuilt from tsumego_reports, as the Worker would on their next report.` : '')
        + (hasVerifiedCols ? `\n* = production would withhold this verified rating as out of date (EGF activity criterion); kept here.` : ''));
}

await main();
