// ablations/serve_loop.mjs
//
// The measurement half of `ablations/serve_loop_figures.py`: how much the serve
// loop — a flagged weakness pulling its own motif's problems into the draw —
// tightens the weakness reading on a SIMULATED player whose true per-motif
// effect is planted, computed by RUNNING the shipped readings, the shipped rank
// estimator and the shipped serve draw rather than restating them.
//
// Imported unmodified from the shipping tree:
//   pwa/js/motif-readings.js   motifReadings (the accuracy reading), isFlagged
//   pwa/shared/rank-window.js  windowMLE, seFloorAt, displayRankFor, serveGapCap,
//                              flattenAttempt, applyM, SERVE_UCB_K, WIN_CAP
//   pwa/js/irt.js              anchorInfo, clearSpan (the 4PL response curve)
//   pwa/js/math-utils.js       quantizeEmbedding (the stored-vector resolution)
//   pwa/js/constants.js        MOTIF_Z, MODEL_INPUT_HISTORY_LENGTH
//
// MIRRORED, minimally, because their own modules reach app state — each named
// at its use site:
//   estimateFromWindow                (pwa/shared/rank-window.js, private)
//   rankConverged                     (pwa/js/rank-decomposition.js)
//   problemInfoWeight                 (pwa/js/rank-decomposition.js)
//   flaggedMotifCategories, strength half (pwa/js/weakness.js)
//   selectChallengeCandidate's pool and slot draw (pwa/js/problem-loader.js)
//   curveForSize                      (pwa/js/rank-decomposition.js)
//
// The player: a fixed true rank plus a planted per-axis effect, so that on a
// problem with correct-move loading e the player answers as though their rank
// were  trueRank + Σ_k delta_k · e_k.  That is exactly the quantity the shipped
// accuracy reading estimates (ranks of residual effect per unit loading), so
// the reading recovering delta is the check that the loop moves precision and
// not the estimate. Only the ACCURACY reading is exercised: no move is picked
// on a wrong attempt, so the tendency reading has no rows and stays null, and
// the loop's tendency-driven slots are therefore never in the draw.
//
// Usage:  node ablations/serve_loop.mjs <payload.json>
// Every random draw comes from a seeded PRNG, so two runs on one payload agree
// bit for bit. The slot draw is rolled from the same PRNG rather than the
// client's date hash — the hash exists for resume determinism and is uniform.

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./json_module_loader.mjs', import.meta.url);

const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const PWA = payload.pwa_root;
const imp = (rel) => import(pathToFileURL(PWA + rel).href);

const { motifReadings, isFlagged } = await imp('js/motif-readings.js');
const {
    windowMLE, seFloorAt, displayRankFor, serveGapCap, flattenAttempt, applyM,
    SERVE_UCB_K, WIN_CAP,
} = await imp('shared/rank-window.js');
const { anchorInfo, clearSpan } = await imp('js/irt.js');
const { quantizeEmbedding } = await imp('js/math-utils.js');
const { MOTIF_Z, MODEL_INPUT_HISTORY_LENGTH } = await imp('js/constants.js');

const K = payload.K;
if (!Number.isInteger(K) || K <= 0) throw new Error('payload carries no axis count');

const CURVES = JSON.parse(readFileSync(PWA + 'tsumegos/rating_to_rank.json', 'utf8'));
// curveForSize (rank-decomposition.js), mirrored: its module imports state.js.
function curveFor(size) {
    const m = CURVES[String(size)];
    return (m && Array.isArray(m.ratings) && Array.isArray(m.ranks) && m.ratings.length >= 2)
        ? m : null;
}

// --- the shipped corpus ---------------------------------------------------------
// One record per problem: <f4 rating <f4 slope <f4 floor u1 size, then K <f4
// correct-move loadings — the published vector, not a tail mask, because the
// serve criterion argmaxes the loading itself.
const REC = 13 + 4 * K;
const raw = readFileSync(payload.problems_path);
const N = raw.length / REC;
if (!Number.isInteger(N)) throw new Error(`problems file is not a multiple of ${REC} bytes`);
const rating = new Float64Array(N), slope = new Float64Array(N), floor = new Float64Array(N);
const rank = new Float64Array(N), quality = new Float64Array(N);
const bsize = new Uint8Array(N);
const emb = [];
for (let i = 0; i < N; i++) {
    const o = i * REC;
    rating[i] = raw.readFloatLE(o);
    slope[i] = raw.readFloatLE(o + 4);
    floor[i] = raw.readFloatLE(o + 8);
    bsize[i] = raw[o + 12];
    const v = new Array(K);
    for (let k = 0; k < K; k++) v[k] = raw.readFloatLE(o + 13 + 4 * k);
    // Stored at the resolution every stored vector gets (attempt-recorder.js).
    emb.push(quantizeEmbedding(v));
    rank[i] = applyM(rating[i], curveFor(bsize[i]));
    // problemInfoWeight (rank-decomposition.js), mirrored: slope²·anchorInfo(floor).
    quality[i] = slope[i] * slope[i] * anchorInfo(floor[i]);
}

const BY_SIZE = new Map();
for (const s of payload.board_sizes) {
    const ix = [];
    for (let i = 0; i < N; i++) if (bsize[i] === s) ix.push(i);
    ix.sort((a, b) => rank[a] - rank[b]);
    BY_SIZE.set(s, { ix, rk: Float64Array.from(ix, (i) => rank[i]) });
}

function lowerBound(rk, x) {
    let lo = 0, hi = rk.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (rk[m] < x) lo = m + 1; else hi = m; }
    return lo;
}

// selectChallengeCandidate's pool (problem-loader.js): the nearest `k` problems
// to the serve rank, nearest first, minus the already answered (the client
// excludes the last RECALL_CANDIDATE_HISTORY_LENGTH answers, which exceeds any
// horizon simulated here). The whole corpus stands in for the loaded shard run.
function poolAt(size, x, excluded, k) {
    const { ix, rk } = BY_SIZE.get(size);
    let a = lowerBound(rk, x), b = a;
    const out = [];
    while (out.length < k && (a > 0 || b < rk.length)) {
        let t;
        if (a === 0) t = b++;
        else if (b === rk.length) t = --a;
        else t = (x - rk[a - 1] <= rk[b] - x) ? --a : b++;
        const gi = ix[t];
        if (!excluded.has(gi)) out.push(gi);
    }
    return out;
}

function rng32(seed) {
    let a = seed | 0;
    return function () {
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// estimateFromWindow (rank-window.js), mirrored whole.
function estimate(flats) {
    const { R, se } = windowMLE(flats);
    const excess = Math.max(0, se - seFloorAt(R));
    const displayRank = displayRankFor(R, se);
    return {
        rank: R, se, displayRank,
        serveRank: Math.min(R + SERVE_UCB_K * excess, displayRank + serveGapCap(flats)),
    };
}
// rankConverged (rank-decomposition.js), mirrored: se below twice the local floor.
function converged(est) {
    const f = seFloorAt(est.rank);
    return Number.isFinite(est.se) && f > 0 && est.se < 2 * f;
}

// flaggedMotifCategories (weakness.js), the strength half, mirrored. Every
// shipped axis is named on its positive tail, which the producer asserts, so
// the tail flip is the identity here: weak -> serve high loadings, strong ->
// serve low ones.
function categories(readings) {
    const out = [];
    if (!readings) return out;
    for (const { k, strength } of readings) {
        if (strength && isFlagged(strength)) {
            out.push(strength.z < 0 ? { k, sign: 1, kind: 'weak' }
                                    : { k, sign: -1, kind: 'strong' });
        }
    }
    return out;
}

// One serve decision (problem-loader.js), mirrored: a uniform slot draw over
// the flagged categories, the information slot while the rank is unconverged,
// and the neutral head. `arm` decides which motif slots are in the draw:
//   shipped   the flagged categories, as the client;
//   neutral   none — the loop opened, exposure at the pool's own rate;
//   oracle    the planted axis's weak slot always present, flagged or not —
//             the loop with zero flag latency, an upper bound on what it can do.
function serve(pool, cats, withInfo, rnd) {
    const slots = cats.length + (withInfo ? 1 : 0) + 1;
    const pick = Math.min(Math.floor(rnd() * slots), slots - 1);
    if (pick < cats.length) {
        const { k, sign } = cats[pick];
        let best = pool[0], bestScore = -Infinity;
        for (const g of pool) {
            const s = Math.max(sign * emb[g][k], 0);
            if (s > bestScore) { bestScore = s; best = g; }
        }
        return { g: best, slot: 'motif' };
    }
    if (withInfo && pick === cats.length) {
        // selectFromTopFraction: a uniform draw from the top INFO_TOP_FRACTION by
        // quality, the fraction carried in the payload from the producer's read
        // of the constant.
        const scored = pool.map((g) => [quality[g], g]).sort((a, b) => b[0] - a[0]);
        const n = Math.max(1, Math.ceil(scored.length * payload.info_top_fraction));
        return { g: scored[Math.min(Math.floor(rnd() * n), n - 1)][1], slot: 'info' };
    }
    return { g: pool[0], slot: 'neutral' };
}

// One player: true rank, the planted per-axis effect vector, one arm. Reads the
// planted axis's reading at every checkpoint and every other axis's flag state.
function simulate({ trueRank, delta, axis, arm, size, horizon, seed, checkpoints }) {
    const rnd = rng32(seed);
    const history = [], flats = [], excluded = new Set();
    const cp = new Set(checkpoints);
    const rows = [];
    let servedTail = 0, servedMotif = 0, servedGap = 0, effectSum = 0;
    let firstFlag = null, everFlagged = false;
    for (let t = 0; t < horizon; t++) {
        // The rank the readings and the serve are conditioned on: the live
        // estimator over the last WIN_CAP first-time attempts.
        const est = estimate(flats.slice(-WIN_CAP));
        const theta = est.rank;
        const readings = motifReadings(history, theta, K, curveFor, MODEL_INPUT_HISTORY_LENGTH);
        let cats;
        if (arm === 'neutral') cats = [];
        else if (arm === 'oracle') cats = [{ k: axis, sign: 1, kind: 'weak' }];
        else cats = categories(readings);
        const planted = readings && readings[axis] && readings[axis].strength;
        const flaggedWeak = !!(planted && isFlagged(planted) && planted.z < 0);
        if (flaggedWeak && firstFlag === null) firstFlag = t;
        if (flaggedWeak) everFlagged = true;
        if (cp.has(t)) {
            let falseWeak = 0, falseAny = 0, testable = 0;
            if (readings) for (const r of readings) {
                if (r.k === axis || !r.strength) continue;
                testable++;
                if (isFlagged(r.strength)) { falseAny++; if (r.strength.z < 0) falseWeak++; }
            }
            rows.push({
                t, rank_err: theta - trueRank, rank_se: est.se,
                converged: converged(est) ? 1 : 0,
                est: planted ? planted.ranks : null,
                se: planted ? planted.se : null,
                z: planted ? planted.z : null,
                flagged_weak: flaggedWeak ? 1 : 0,
                first_flag: firstFlag,
                other_testable: testable, other_flagged: falseAny, other_weak: falseWeak,
                served_tail: servedTail / t, served_motif: servedMotif / t,
                served_gap: servedGap / t, served_effect: effectSum / t,
            });
        }
        const pool = poolAt(size, est.serveRank, excluded, payload.pool_size);
        const { g, slot } = serve(pool, cats, !converged(est), rnd);
        excluded.add(g);
        const e = emb[g];
        if (e[axis] > MOTIF_Z) servedTail++;
        if (slot === 'motif') servedMotif++;
        servedGap += rank[g] - trueRank;
        let eff = trueRank;
        for (let k = 0; k < K; k++) eff += delta[k] * e[k];
        effectSum += eff - trueRank;
        const f = flattenAttempt(rating[g], slope[g], floor[g], 1, curveFor(size));
        if (!f) throw new Error('flattenAttempt refused a shipped problem');
        const praw = 1 / (1 + Math.exp(-(f.slope * (eff - f.x) + f.kappa)));
        const p = f.floor + clearSpan(f.floor) * praw;
        const correct = rnd() < p;
        history.push({
            size, correct, isRepeat: false, linked: false,
            problemRating: rating[g], problemSlope: slope[g], problemFloor: floor[g],
            embedding: e,
        });
        flats.push(flattenAttempt(rating[g], slope[g], floor[g], correct ? 1 : 0, curveFor(size)));
    }
    return { rows, ever_flagged: everFlagged ? 1 : 0, first_flag: firstFlag };
}

const cfg = payload.sim;
const out = { motif_z: MOTIF_Z, window: MODEL_INPUT_HISTORY_LENGTH, rank_window: WIN_CAP,
              K, cells: [] };
let cell = 0;
for (const trueRank of cfg.true_ranks) {
    for (const axis of cfg.axes) {
        for (const d of cfg.deltas) {
            const delta = new Array(K).fill(0);
            delta[axis] = d;
            for (const arm of cfg.arms) {
                const seeds = [];
                for (let s = 0; s < cfg.seeds; s++) {
                    seeds.push(simulate({
                        trueRank, delta, axis, arm, size: cfg.size, horizon: cfg.horizon,
                        // One seed stream per (rank, axis, delta, seed): the arms of a
                        // cell share it, so their contrast is paired on the player's
                        // own dice where the served problems coincide.
                        seed: payload.seed + 7919 * s + 104729 * cell,
                        checkpoints: cfg.checkpoints,
                    }));
                }
                out.cells.push({ true_rank: trueRank, axis, delta: d, arm, seeds });
            }
            cell++;
        }
    }
}
process.stdout.write(JSON.stringify(out) + '\n');
