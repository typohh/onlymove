// pwa/src/contest.js
//
// The /contest page's data: every contest_entries row resolved into what the page
// shows, cached as one JSON blob in server_stats under key 'contest' (the same
// single-row cache the Community stats use under 'latest'). Recomputed ON REQUEST
// when the blob is older than CONTEST_MAX_AGE_MS or was produced by an older code
// version; the organiser's script (auxiliary/contest_entries.mjs) deletes the row
// after an edit so the next view recomputes.
//
// Nothing identifying the account leaves this module: the blob carries no user id
// (the id is the account credential), only the display fields below.
//
// Per entry, resolved off the ACCOUNT rather than stored on the entry:
//   name     — linked EGD/AGA display_name, else the leaderboard alias, else null
//              (Anonymous). A hidden profile (hidden_profiles) withholds both —
//              which is how an entrant stays anonymous; there is no contest flag.
//   irl      — strength only: the discrete federation grade off the latest 'EGD' /
//              'AGAR' heartbeat, shown for hidden profiles too (the grade is the
//              entry, the name is not).
//   ranks    — a replay of the account's novel attempts through rankWalkFromAttempts,
//              the shown (untimed) OnlyMove rank after each one. Strength: the mean
//              over the WHOLE history past the rank-start gate, plus the count of
//              in-corpus novel attempts. Video: the peak over the first
//              `video_problems` novel attempts — the ones solved on camera.
//
// The organiser is not an entry. The minimum criterion is "stronger than the
// organiser", so the blob carries the organiser's CURRENT figures for the page to
// print beside it: the EGF rank off the latest 'EGD' GoR heartbeat of the account
// linked to ORGANISER_EGD_PIN, and that account's current shown OnlyMove rank.

import { APP_VERSION } from './version.js';
import {
    getProblemParams, rankWalkFromAttempts, resolveRowCountry, declaredJsonSql,
    shortRankLabel, gorToRank, agaRatingToRank, buildRowStateFromReports,
} from './leaderboard.js';
import { rankToLabel } from '../shared/rank-math.js';

const CONTEST_KEY = 'contest';
export const CONTEST_MAX_AGE_MS = 12 * 3600 * 1000;
const ORGANISER_EGD_PIN = '16025130';

let computeInFlight = null;
function computeContestOnce(env) {
    if (!computeInFlight) {
        computeInFlight = computeContest(env).finally(() => { computeInFlight = null; });
    }
    return computeInFlight;
}

export async function loadContest(env) {
    const row = await env.DB.prepare(
        `SELECT data, computed_at FROM server_stats WHERE key = ?1`
    ).bind(CONTEST_KEY).first();
    let cached = null;
    if (row && row.data) {
        try { cached = JSON.parse(row.data); } catch { cached = null; }
    }
    const age = row?.computed_at ? Date.now() - Date.parse(row.computed_at) : Infinity;
    if (cached && cached.version === APP_VERSION && age < CONTEST_MAX_AGE_MS) return cached;
    return computeContestOnce(env);
}

// The federation grade from the account's declared cache: the heartbeat's own
// official label, else the band its rating falls in.
function federationGrade(declaredJson) {
    let arr;
    try { arr = JSON.parse(declaredJson || '[]'); } catch { arr = []; }
    for (const [server, fed, toRank] of [['EGD', 'EGF', (v) => gorToRank(parseInt(v, 10))],
                                         ['AGAR', 'AGA', (v) => agaRatingToRank(Number(v))]]) {
        const d = Array.isArray(arr) ? arr.find((x) => x && x.server === server) : null;
        if (!d) continue;
        const label = shortRankLabel(d.rank_label);
        if (label) return { fed, label };
        const v = toRank(d.rank);
        if (Number.isFinite(v)) return { fed, label: rankToLabel(v) };
    }
    return null;
}

// The organiser's current EGF rank (continuous, from the GoR) and current shown
// OnlyMove rank — the same computation as the organiser's leaderboard row, run
// here because an idle row is pruned. Null fields when the account or its
// heartbeat is missing; the page then prints the criterion without them.
async function organiserFigures(env, params) {
    const x = await env.DB.prepare(
        `SELECT o.user_id, ${declaredJsonSql('o.user_id')} AS declared
           FROM external_ids o WHERE o.egd_pin = ?1`
    ).bind(ORGANISER_EGD_PIN).first();
    if (!x) return { gor: null, egfRank: null, onlymoveRank: null };
    let gor = null;
    try {
        const d = (JSON.parse(x.declared || '[]') || []).find((e) => e && e.server === 'EGD');
        const v = d ? parseInt(d.rank, 10) : NaN;
        if (Number.isFinite(v) && v > 0) gor = v;
    } catch { gor = null; }
    const { rank } = await buildRowStateFromReports(env, x.user_id, params);
    return {
        gor,
        egfRank: gor === null ? null : gorToRank(gor),
        onlymoveRank: Number.isFinite(rank) ? rank : null,
    };
}

async function computeContest(env) {
    const res = await env.DB.prepare(
        `SELECT e.id, e.category, e.user_id, e.submitted_at, e.video_url,
                e.video_problems, e.views, e.quality,
                x.display_name, x.country_code, a.adjective, a.color, a.animal,
                (h.user_id IS NOT NULL) AS hidden,
                ${declaredJsonSql('e.user_id')} AS declared
           FROM contest_entries e
           LEFT JOIN external_ids x ON x.user_id = e.user_id
           LEFT JOIN user_aliases a ON a.user_id = e.user_id
           LEFT JOIN hidden_profiles h ON h.user_id = e.user_id
          ORDER BY e.submitted_at, e.id`
    ).all();
    const rows = (res && res.results) || [];
    const params = await getProblemParams(env);

    const strength = [], video = [];
    for (const r of rows) {
        const att = await env.DB.prepare(
            `SELECT problem AS p, success AS s FROM tsumego_reports
              WHERE user_id = ?1 AND is_repeat = 0 ORDER BY id`
        ).bind(r.user_id).all();
        const attempts = ((att && att.results) || []).map((a) => ({ p: a.p, s: a.s ? 1 : 0 }));
        const hidden = !!r.hidden;
        const alias = (r.adjective && r.color && r.animal)
            ? { adjective: r.adjective, color: r.color, animal: r.animal } : null;
        const base = {
            submittedAt: r.submitted_at,
            name: hidden ? null : (r.display_name || null),
            alias: hidden ? null : alias,
            country: hidden ? null
                : resolveRowCountry({ declared: r.declared, country_code: r.country_code, country: null }),
        };
        if (r.category === 'strength') {
            const ranks = rankWalkFromAttempts(attempts, params);
            let sum = 0, n = 0;
            for (const v of ranks) if (Number.isFinite(v)) { sum += v; n++; }
            const problems = attempts.filter((a) => params.has(a.p)).length;
            strength.push({ ...base, irl: federationGrade(r.declared), avgRank: n ? sum / n : null, problems });
        } else {
            const count = Number.isFinite(r.video_problems) && r.video_problems > 0
                ? Math.min(r.video_problems, attempts.length) : attempts.length;
            const ranks = rankWalkFromAttempts(attempts.slice(0, count), params);
            let peak = null;
            for (const v of ranks) if (Number.isFinite(v) && (peak === null || v > peak)) peak = v;
            video.push({
                ...base,
                url: r.video_url || null,
                problems: Number.isFinite(r.video_problems) ? r.video_problems : null,
                views: Number.isFinite(r.views) ? r.views : null,
                quality: Number.isFinite(r.quality) ? r.quality : null,
                peakRank: peak,
            });
        }
    }

    const data = {
        version: APP_VERSION, computedAt: new Date().toISOString(),
        organiser: await organiserFigures(env, params), strength, video,
    };
    await env.DB.prepare(
        `INSERT INTO server_stats (key, data, computed_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET data = excluded.data, computed_at = excluded.computed_at`
    ).bind(CONTEST_KEY, JSON.stringify(data), data.computedAt).run();
    return data;
}
