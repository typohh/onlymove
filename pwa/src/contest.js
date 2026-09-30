// pwa/src/contest.js
//
// The /contest page's data: every LISTED video contest_entries row (pending ones,
// sent from the page's form and not yet verified, stay out) resolved into what the
// page shows, cached as one JSON blob in server_stats under key 'contest' (the same
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
//   peakRank — a replay of the account's novel attempts through rankWalkFromAttempts,
//              the shown (untimed) OnlyMove rank after each one, peaked over the
//              first `video_problems` of them — the ones solved on camera.

import { APP_VERSION } from './version.js';
import { getProblemParams, rankWalkFromAttempts, resolveRowCountry, declaredJsonSql } from './leaderboard.js';

const CONTEST_KEY = 'contest';
export const CONTEST_MAX_AGE_MS = 12 * 3600 * 1000;
// Entries close here: the page counts down to it and POST /contest/submit refuses after it.
export const CONTEST_DEADLINE_ISO = '2026-12-31T23:59:59Z';

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

async function computeContest(env) {
    const res = await env.DB.prepare(
        `SELECT e.id, e.user_id, e.listed_at, e.video_url, e.video_problems, e.views, e.quality,
                x.display_name, x.country_code, a.adjective, a.color, a.animal,
                (h.user_id IS NOT NULL) AS hidden,
                ${declaredJsonSql('e.user_id')} AS declared
           FROM contest_entries e
           LEFT JOIN external_ids x ON x.user_id = e.user_id
           LEFT JOIN user_aliases a ON a.user_id = e.user_id
           LEFT JOIN hidden_profiles h ON h.user_id = e.user_id
          WHERE e.category = 'video' AND e.listed_at IS NOT NULL
          ORDER BY e.listed_at, e.id`
    ).all();
    const rows = (res && res.results) || [];
    const params = await getProblemParams(env);

    const video = [];
    for (const r of rows) {
        const att = await env.DB.prepare(
            `SELECT problem AS p, success AS s FROM tsumego_reports
              WHERE user_id = ?1 AND is_repeat = 0 ORDER BY id`
        ).bind(r.user_id).all();
        const attempts = ((att && att.results) || []).map((a) => ({ p: a.p, s: a.s ? 1 : 0 }));
        const hidden = !!r.hidden;
        const alias = (r.adjective && r.color && r.animal)
            ? { adjective: r.adjective, color: r.color, animal: r.animal } : null;
        const count = Number.isFinite(r.video_problems) && r.video_problems > 0
            ? Math.min(r.video_problems, attempts.length) : attempts.length;
        const ranks = rankWalkFromAttempts(attempts.slice(0, count), params);
        let peak = null;
        for (const v of ranks) if (Number.isFinite(v) && (peak === null || v > peak)) peak = v;
        video.push({
            listedAt: r.listed_at,
            name: hidden ? null : (r.display_name || null),
            alias: hidden ? null : alias,
            country: hidden ? null
                : resolveRowCountry({ declared: r.declared, country_code: r.country_code, country: null }),
            url: r.video_url || null,
            problems: Number.isFinite(r.video_problems) ? r.video_problems : null,
            views: Number.isFinite(r.views) ? r.views : null,
            quality: Number.isFinite(r.quality) ? r.quality : null,
            peakRank: peak,
        });
    }

    const data = { version: APP_VERSION, computedAt: new Date().toISOString(), video };
    await env.DB.prepare(
        `INSERT INTO server_stats (key, data, computed_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET data = excluded.data, computed_at = excluded.computed_at`
    ).bind(CONTEST_KEY, JSON.stringify(data), data.computedAt).run();
    return data;
}
