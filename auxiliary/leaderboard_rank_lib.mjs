// auxiliary/leaderboard_rank_lib.mjs
//
// Shared plumbing for the operator reports that must read a rank the way PRODUCTION
// computes it, rather than reimplementing it:
//
//   * loadRankModule() esbuild-bundles pwa/src/leaderboard.js (plus the shipped
//     rating_to_rank / rank_map / population_constants JSON and se_floor), so the M
//     curve, the rank-dependent SE floor, the time dock, the couplings and
//     RANK_START_MIN are the Worker's own. Its bundle entry exports computeRankBoth —
//     computeRank WITHOUT production's RANK_START_MIN gate, so a caller that wants the
//     gate applies it to the returned `usable`. One entry point on purpose: a gated and
//     an ungated variant of the same rank in two files is how a report drifts from what
//     the app shows.
//   * loadProblemParamsLocal() reads the same 16-byte records the Worker reads.
//   * sqliteJson()/tableColumns() read the local mirror read-only.
//   * rankStr()/durationStr() format the two quantities every such report prints.
//
// Extracted from auxiliary/leaderboard_player_stats.mjs, which held the only copy;
// auxiliary/top_users_report.mjs is the second user. Nothing here writes.

import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const PWA_DIR = path.join(REPO, 'pwa');
export const DB_PATH = path.join(REPO, 'db', 'd1.sqlite');
export const PARAMS_BIN = path.join(PWA_DIR, 'tsumegos', 'problem_params.bin');
export const STALE_WARN_MS = 6 * 3600 * 1000;

// --- bundle the production rank code ---------------------------------------------
// Imports pwa/src/leaderboard.js directly, so the couplings, the time dock, the
// SE floor and the activity gate can't drift from what the Worker ships.
// computeRankBoth mirrors its computeRank — same flatten, same shown rank, same
// dock-then-couple order — but ALSO returns the AGA-frame estimate, and deliberately
// drops the RANK_START_MIN gate (see below).
export async function loadRankModule() {
    const require = createRequire(path.join(PWA_DIR, 'package.json'));
    const esbuild = require('esbuild');
    const res = await esbuild.build({
        stdin: {
            contents: `
                import { interpExtrap, rankToLabel, rankLabelToRank } from './shared/rank-math.js';
                import { estimateWindowRankFromFlat, flattenAttempt } from './shared/rank-window.js';
                import {
                    RANK_START_MIN, BUFFER_CAP, gorToRank, agaRatingToRank, shortRankLabel,
                    isVerifiedActive, internalToServerRank, timeDeltaRatingFromBuffer, parseBuffer,
                    normalizeCountry,
                } from './src/leaderboard.js';
                import RATING_RANK_MAP from './tsumegos/rating_to_rank.json';

                // One shared cross-board curve, as in leaderboard.js.
                const CURVE = RATING_RANK_MAP['9'] || Object.values(RATING_RANK_MAP)[0] || null;

                export {
                    RANK_START_MIN, BUFFER_CAP, gorToRank, agaRatingToRank,
                    isVerifiedActive, rankToLabel, parseBuffer, normalizeCountry,
                };

                // Mirrors leaderboard.js's (private) rankAfterRatingDelta.
                function rankAfterRatingDelta(rank, deltaRating) {
                    if (!CURVE || !Number.isFinite(rank) || !Number.isFinite(deltaRating)) return rank;
                    const rating = interpExtrap(rank, CURVE.ranks, CURVE.ratings);
                    return interpExtrap(rating + deltaRating, CURVE.ratings, CURVE.ranks);
                }

                // Mirrors leaderboard.js's (private) tierRankFromLabel, with the
                // caller's rating-derived tier as the fallback — the activity window
                // the /leaderboard page gates a verified rating on.
                export function verifiedTierRank(label, ratingTier) {
                    const short = shortRankLabel(label);
                    if (short) {
                        if (/^\\d{1,2}p$/.test(short)) return 6.5;       // pro -> dan tier (7d)
                        const v = rankLabelToRank(short);
                        if (Number.isFinite(v)) return v;
                    }
                    return ratingTier;
                }

                // buffer: [{p,s,t}] newest first (the stored leaderboard buffer).
                export function computeRankBoth(buffer, params) {
                    const flat = [];
                    for (let i = buffer.length - 1; i >= 0; i--) {   // oldest -> newest
                        const e = buffer[i];
                        const pp = e && typeof e.p === 'string' ? params.get(e.p) : null;
                        if (!pp) continue;                            // left the corpus
                        const f = flattenAttempt(pp.rating, pp.slope, pp.floor, e.s, CURVE);
                        if (!f) continue;
                        flat.push(f);
                    }
                    // NOTE: production gates a rank on flat.length >= RANK_START_MIN. This
                    // report deliberately DROPS that gate — the estimator is a posterior over
                    // the measured population rank prior, so it is well-posed on any number of
                    // attempts and even not-yet-listed players get a (low-confidence) rank.
                    // Below the gate the posterior is still near the prior and the shown rank
                    // sits well under the mean, which is what production shows a new player.
                    const est = estimateWindowRankFromFlat(flat);
                    if (!est) {
                        return { onlymoveRank: null, egfRank: null, agaRank: null, se: null, usable: flat.length };
                    }
                    // The SHOWN rank, exactly as computeRank stores it into pred_rank: the
                    // posterior mean less K_DISPLAY excess-se over the rank-dependent floor.
                    const onlymoveRank = est.displayRank;
                    // The personal time dock moves that untimed rank into the server frame
                    // the couplings (and a real GoR / AGA rating) speak — dock BEFORE
                    // coupling, the order readLeaderboardRows uses.
                    const deltaRating = timeDeltaRatingFromBuffer(buffer, params);
                    const timedRank = deltaRating
                        ? rankAfterRatingDelta(onlymoveRank, deltaRating) : onlymoveRank;
                    return {
                        onlymoveRank,
                        egfRank: internalToServerRank('EGF', timedRank),
                        agaRank: internalToServerRank('AGA', timedRank),
                        se: est.se, usable: flat.length,
                    };
                }
            `,
            resolveDir: PWA_DIR,
            sourcefile: 'aux-player-stats-entry.mjs',
        },
        bundle: true,
        format: 'esm',
        platform: 'neutral',
        write: false,
        logLevel: 'silent',
    });
    const code = res.outputFiles[0].text;
    return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}

// Same 16-byte record as leaderboard.js loadProblemParams: 4-byte id + 3 LE float32.
export function loadProblemParamsLocal() {
    const bytes = fs.readFileSync(PARAMS_BIN);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const REC = 16;
    const map = new Map();
    for (let off = 0; off + REC <= bytes.length; off += REC) {
        let id = '';
        for (let k = 0; k < 4; k++) id += bytes[off + k].toString(16).padStart(2, '0');
        map.set(id, {
            rating: dv.getFloat32(off + 4, true),
            slope: dv.getFloat32(off + 8, true),
            floor: dv.getFloat32(off + 12, true),
        });
    }
    return map;
}

// `dbPath` defaults to the canonical local mirror; a caller checking a snapshot
// (e.g. auxiliary/top_users_report.mjs --db) passes its own.
export function sqliteJson(sql, dbPath = DB_PATH) {
    const out = execFileSync('sqlite3', ['-json', dbPath, sql],
        { encoding: 'utf8', maxBuffer: 512 * 1024 * 1024 });
    const text = out.trim();
    return text ? JSON.parse(text) : [];
}

// The mirror can predate a migration (pwa/migrations/*.sql lands on the live D1 first),
// so every column added after this script was written is probed before it's selected.
export function tableColumns(table, dbPath = DB_PATH) {
    return new Set(sqliteJson(`PRAGMA table_info(${table})`, dbPath).map((r) => r.name));
}

export const rankStr = (mod, v) => (Number.isFinite(v) ? `${v >= 0 ? ' ' : ''}${v.toFixed(2)} (${mod.rankToLabel(v)})` : '—');
export function durationStr(ms) {
    if (!Number.isFinite(ms)) return '—';
    const s = ms / 1000;
    if (s < 120) return `${s.toFixed(1)}s`;
    return `${(s / 60).toFixed(1)}m`;
}
