#!/usr/bin/env node
/**
 * How many attempts does a user need before AXIS-DRIVEN SELECTION turns on?
 *
 * The client only weights problem selection by weakness once BOTH gates open
 * (problem-loader.js): `weaknessModelReady()`, which needs 20 attempts carrying
 * embeddings, and `rankConverged()`, which is `se < 2 * seFloorAt(rank)`. The
 * second is far slower, so it is what decides when the motif axes begin to
 * affect what anyone is served — and therefore which users an evaluation of the
 * axes should be measuring at all.
 *
 * WHY A SIMULATOR RATHER THAN A REPLAY OF REAL HISTORIES. Ratings are revised
 * retroactively on every refit, so a replay would feed today's ratings into
 * serve decisions taken under older ones. The information each attempt carried
 * at the time is not recoverable from the stored history, and the convergence a
 * replay reported would be one that never happened. In a simulation the ratings
 * the server picks from and the ratings the estimator scores against are the
 * same numbers by construction.
 *
 * WHAT IS IMPORTED VERSUS REIMPLEMENTED. The estimator, the SE floor, the
 * display/serve offsets and the streak cap are imported from the SHIPPED
 * modules (`pwa/shared/rank-window.js`, `pwa/js/irt.js`) and run unmodified, so
 * there is no port to drift out of step. Reimplemented here, because they live
 * inside app code that cannot be loaded headless: the candidate pool (the
 * nearest POOL_SIZE problems by rank to the serve target, standing in for the
 * client's shard load) and the info-weighted draw within it. The pre-convergence
 * regime is the one being measured and it selects on information alone, so the
 * weakness path is deliberately not modelled.
 *
 * THE RESPONSE MODEL IS THE CLIENT'S OWN 4PL, which makes this an OPTIMISTIC
 * bound: a simulated user is exactly as predictable as the estimator assumes.
 * Real users deviate from their own curve, so real convergence is slower than
 * anything reported here. Read the output as "no sooner than".
 *
 * Usage:
 *   node /tmp/sim.mjs [--reps 40] [--max 400] [--ranks -25,-20,...]
 * built with:
 *   pwa/node_modules/.bin/esbuild auxiliary/simulate_rank_convergence.mjs \
 *     --bundle --platform=node --format=esm --outfile=/tmp/sim.mjs
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
    applyM, slopeInRankSpace, flattenAttempt, estimateWindowRankFromFlat,
    coldStartEstimate, seFloorAt, PRIOR_MEAN, PRIOR_SD,
} from '../pwa/shared/rank-window.js';
import { clampFloor, clearKappa, clearSpan, anchorInfo } from '../pwa/js/irt.js';
import RATING_TO_RANK from '../pwa/tsumegos/rating_to_rank.json';

// The bundle runs from wherever esbuild wrote it, so import.meta.url points at
// the artifact rather than the repo. Take the root from the environment (or the
// working directory) instead.
const ROOT = process.env.ONLYMOVE_ROOT || process.cwd();
const POOL_SIZE = 2000;          // ~ one shard's worth of problems near the target

function arg(name, dflt) {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

// ---------------------------------------------------------------- corpus
// problem_params.bin: 16 bytes per problem — 4-byte id then three little-endian
// float32 (rating, slope, floor). The same file the client and Worker read.
function loadProblems(curve) {
    const buf = readFileSync(join(ROOT, 'pwa/tsumegos/problem_params.bin'));
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const out = [];
    for (let o = 0; o + 16 <= buf.byteLength; o += 16) {
        const rating = dv.getFloat32(o + 4, true);
        const slope = dv.getFloat32(o + 8, true);
        const floorRaw = dv.getFloat32(o + 12, true);
        if (!Number.isFinite(rating) || !Number.isFinite(slope)) continue;
        const floor = clampFloor(floorRaw);
        const s = slopeInRankSpace(rating, slope, curve);
        if (!(s > 0)) continue;
        out.push({
            rating, aP: slope, floorRaw,
            x: applyM(rating, curve), slope: s, floor,
            kappa: clearKappa(floor), span: clearSpan(floor),
            info: s * s * anchorInfo(floor),
        });
    }
    out.sort((a, b) => a.x - b.x);
    return out;
}

// Nearest POOL_SIZE problems by rank to `target`, with a cumulative info weight
// table for the draw. Precomputed per pool start so the inner loop is a binary
// search plus one uniform draw.
function poolAt(problems, target) {
    let lo = 0, hi = problems.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (problems[mid].x < target) lo = mid + 1; else hi = mid;
    }
    let a = Math.max(0, lo - (POOL_SIZE >> 1));
    let b = Math.min(problems.length, a + POOL_SIZE);
    a = Math.max(0, b - POOL_SIZE);
    return [a, b];
}

function drawProblem(problems, a, b, rnd) {
    let total = 0;
    for (let i = a; i < b; i++) total += problems[i].info;
    let r = rnd() * total;
    for (let i = a; i < b; i++) {
        r -= problems[i].info;
        if (r <= 0) return problems[i];
    }
    return problems[b - 1];
}

// deterministic RNG so a rerun reproduces
function mulberry32(seed) {
    let t = seed >>> 0;
    return function () {
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
}

// -------------------------------------------------------------- simulation
function runOne(problems, curve, theta, rnd, maxAttempts) {
    const history = [];
    for (let t = 1; t <= maxAttempts; t++) {
        const est = history.length ? estimateWindowRankFromFlat(history)
                                   : coldStartEstimate();
        const [a, b] = poolAt(problems, est.serveRank);
        const p = drawProblem(problems, a, b, rnd);
        const z = p.slope * (theta - p.x) + p.kappa;
        const prob = p.floor + p.span / (1 + Math.exp(-z));
        const correct = rnd() < prob;
        history.push(flattenAttempt(p.rating, p.aP, p.floorRaw, correct, curve));
        const after = estimateWindowRankFromFlat(history);
        if (after.se < 2 * seFloorAt(after.rank)) return { t, est: after.rank };
    }
    return { t: null, est: null };
}

function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    const i = (sorted.length - 1) * q;
    const lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function main() {
    const reps = parseInt(arg('reps', '40'), 10);
    const maxAttempts = parseInt(arg('max', '400'), 10);
    const ranks = arg('ranks', '-25,-20,-15,-10,-6,-3,-1,1,3,5')
        .split(',').map(Number);

    const curve = RATING_TO_RANK['9'] || Object.values(RATING_TO_RANK)[0];
    const problems = loadProblems(curve);
    const cold = coldStartEstimate();
    console.log(`corpus ${problems.length} problems, rank span `
        + `${problems[0].x.toFixed(1)} .. ${problems[problems.length - 1].x.toFixed(1)}`);
    console.log(`cold start: rank ${cold.rank.toFixed(2)} se ${cold.se.toFixed(2)} `
        + `serve ${cold.serveRank.toFixed(2)} (prior mean ${PRIOR_MEAN.toFixed(2)}, `
        + `sd ${PRIOR_SD.toFixed(2)})`);
    console.log(`gate: se < 2 * seFloorAt(rank); ${reps} reps/rank, cap ${maxAttempts}\n`);
    // mean and sd are the headline: they are what a threshold would be set from.
    // Reported beside the median because the distribution is right-skewed — a
    // user whose true rank sits far from the population prior spends attempts
    // travelling before the posterior tightens, so the tail is long and the mean
    // sits above the median rather than on it.
    console.log(' true rank   label     mean    sd   median  p25   p75   never   floor');
    for (const theta of ranks) {
        const ts = [];
        let never = 0;
        for (let r = 0; r < reps; r++) {
            const { t } = runOne(problems, curve, theta,
                                 mulberry32(1000 * (theta + 40) + r),
                                 maxAttempts);
            if (t === null) never++; else ts.push(t);
        }
        ts.sort((a, b) => a - b);
        const lab = theta < 0 ? `${Math.ceil(-theta)}k` : `${Math.floor(theta) + 1}d`;
        const mean = ts.length ? ts.reduce((s, v) => s + v, 0) / ts.length : NaN;
        const sd = ts.length > 1
            ? Math.sqrt(ts.reduce((s, v) => s + (v - mean) ** 2, 0) / (ts.length - 1))
            : NaN;
        console.log(
            `${String(theta).padStart(9)}   ${lab.padEnd(7)} `
            + `${mean.toFixed(1).padStart(6)} `
            + `${sd.toFixed(1).padStart(5)} `
            + `${String(quantile(ts, 0.5)).padStart(6)} `
            + `${String(quantile(ts, 0.25)).padStart(5)} `
            + `${String(quantile(ts, 0.75)).padStart(5)} `
            + `${String(never).padStart(5)}  `
            + `${seFloorAt(theta).toFixed(3).padStart(6)}`);
    }
}

main();
