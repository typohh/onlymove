// ablations/client_estimators.mjs
//
// The measurement half of `ablations/client_estimator_figures.py`: everything in
// docs/methodology.md §2.3, §2.4 and §2.11 that is a property of the CLIENT's own
// estimators is computed here, by importing the shipped modules and calling them,
// so a formula change in pwa/ moves the figure instead of silently invalidating it.
//
// Imported unmodified from the shipping tree:
//   pwa/shared/rank-window.js  windowMLE (the grid posterior), seFloorAt,
//                              displayRankFor, serveGapCap, SERVE_UCB_K,
//                              FRESH_RECORD_BONUS, flattenAttempt, applyM
//   pwa/js/irt.js              anchorInfo, clearSpan (the 4PL response curve)
//   pwa/js/constants.js        MOTIF_Z
// Both JSON-importing modules load through json_module_loader.mjs.
//
// MIRRORED, minimally, because their own modules reach DOM/app state and cannot be
// imported here — each is named at its use site:
//   problemInfoWeight            (pwa/js/rank-decomposition.js)
//   estimateFromWindow           (pwa/shared/rank-window.js, private)
//   selectChallengeCandidate's pool + neutral slot   (pwa/js/problem-loader.js)
//
// Usage:  node ablations/client_estimators.mjs <payload.json>
// Reads the payload written by the producer, writes one JSON object to stdout.
// Every random draw comes from a seeded PRNG, so two runs on one payload agree bit
// for bit.

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./json_module_loader.mjs', import.meta.url);

const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const PWA = payload.pwa_root;
const imp = (rel) => import(pathToFileURL(PWA + rel).href);

const {
    windowMLE, seFloorAt, displayRankFor, serveGapCap, flattenAttempt, applyM,
    SERVE_UCB_K, FRESH_RECORD_BONUS, RANK_START_MIN, WIN_CAP,
} = await imp('shared/rank-window.js');
const { anchorInfo, clearSpan } = await imp('js/irt.js');
const { MOTIF_Z } = await imp('js/constants.js');

const CURVE = JSON.parse(readFileSync(PWA + 'tsumegos/rating_to_rank.json', 'utf8'))['9'];
if (!CURVE || !Array.isArray(CURVE.ratings)) throw new Error('rating_to_rank.json has no usable curve');

// --- the shipped corpus ----------------------------------------------------------
// One 14-byte record per problem, written by the producer from problem_params.bin
// joined to lookup.txt and lookup.embeddings.bin:
//   <f4 rating  <f4 slope  <f4 floor  u1 board size  u1 motif-tail bitmask
// The bitmask's bit k is set when the problem's correct-move loading on axis k
// clears MOTIF_Z on the axis's NAMED (positive) tail — the client's own
// problem-motif gate (motif-bubble.js).
const REC = 14;
const raw = readFileSync(payload.problems_path);
const N = raw.length / REC;
if (!Number.isInteger(N)) throw new Error(`problems file is not a multiple of ${REC} bytes`);
const rating = new Float64Array(N), slope = new Float64Array(N), floor = new Float64Array(N);
const rank = new Float64Array(N), quality = new Float64Array(N);
const bsize = new Uint8Array(N), motif = new Uint8Array(N);
for (let i = 0; i < N; i++) {
    const o = i * REC;
    rating[i] = raw.readFloatLE(o);
    slope[i] = raw.readFloatLE(o + 4);
    floor[i] = raw.readFloatLE(o + 8);
    bsize[i] = raw[o + 12];
    motif[i] = raw[o + 13];
    rank[i] = applyM(rating[i], CURVE);
    // problemInfoWeight (rank-decomposition.js): the 4PL Fisher information one
    // at-rank attempt carries, slope²·anchorInfo(floor). Mirrored because its own
    // module imports state.js; anchorInfo itself is the shipped one.
    quality[i] = slope[i] * slope[i] * anchorInfo(floor[i]);
}

// Problems of one board size, ascending by rank — the client's pool is per size
// (state.allProblems holds the active size alone).
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

// selectChallengeCandidate's pool (problem-loader.js): the `k` problems nearest the
// serve rank, minus the recently answered. The whole corpus stands in for the loaded
// shard run, which is the run's own limit case. Ties go to the weaker problem, a
// deterministic stand-in for the client's stable sort over load order.
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

// Deterministic PRNG (mulberry32). Never Math.random: the figures have to reproduce.
function rng32(seed) {
    let a = seed | 0;
    return function () {
        a = (a + 0x6D2B79F5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// --- §2.3: the three rank slots, with the serve-gap cap as the arm ----------------
// estimateFromWindow (rank-window.js, not exported) with `serveGapCap` lifted into
// an argument. Everything else — the grid posterior, the SE floor, the display
// offset and the information-seeking term — is the shipped code.
function estimate(attempts, capFn) {
    const { R, se } = windowMLE(attempts);
    const excess = Math.max(0, se - seFloorAt(R));
    const displayRank = displayRankFor(R, se);
    return {
        R, se, displayRank,
        serveRank: Math.min(R + SERVE_UCB_K * excess, displayRank + capFn(attempts)),
    };
}

const ARMS = {
    // What ships: one rank per consecutive clear, plus one, plus the unblemished bonus.
    shipped: serveGapCap,
    // The standing alternative the method chapter argues against.
    constant: () => payload.serve_margin.constant_margin,
    // The alternative that also delivers the speed: the same expression with the
    // trailing RUN replaced by the attempt COUNT, so the margin opens on a fixed
    // schedule rather than an earned one. Identical to `shipped` on a flawless
    // record, which is why it matches its onboarding speed.
    attempt_count: (a) => {
        let unblemished = true;
        for (const x of a) if (!x.correct) unblemished = false;
        return 1 + a.length + (unblemished ? FRESH_RECORD_BONUS : 0);
    },
};

// One simulated player of fixed true rank, answering through the shipped 4PL.
// Returns the served problems' rank gaps above the player and, if it happened, the
// first-time attempt count at which the SHOWN rank first reached `danAt`.
function simulate(trueRank, capFn, size, horizon, seed, danAt) {
    const rnd = rng32(seed);
    const attempts = [], excluded = new Set(), gaps = [];
    let reached = null;
    for (let t = 0; t < horizon; t++) {
        const est = estimate(attempts, capFn);
        if (danAt !== null && reached === null && est.displayRank >= danAt) { reached = t; break; }
        const pool = poolAt(size, est.serveRank, excluded, payload.pool_size);
        // The NEUTRAL slot's criterion: the pool's nearest-to-rank head. That is the
        // serve rule itself; the draw's other slots score motif loading and Fisher
        // information, which are not what the margin decides.
        const g = pool[0];
        excluded.add(g);
        const f = flattenAttempt(rating[g], slope[g], floor[g], 1, CURVE);
        if (!f) throw new Error('flattenAttempt refused a shipped problem');
        // The shipped response curve at this player's rank: the identical expression
        // attemptLogLik evaluates on the posterior grid.
        const praw = 1 / (1 + Math.exp(-(f.slope * (trueRank - f.x) + f.kappa)));
        const p = trueRank === null ? 1 : f.floor + clearSpan(f.floor) * praw;
        gaps.push(rank[g] - trueRank);
        attempts.push({ x: f.x, slope: f.slope, floor: f.floor, kappa: f.kappa,
                        correct: rnd() < p ? 1 : 0 });
    }
    return { gaps, reached };
}

function serveMargin() {
    const cfg = payload.serve_margin;
    const out = { abilities: [], badge: {} };
    for (const trueRank of cfg.true_ranks) {
        const row = { true_rank: trueRank, arms: {} };
        for (const [name, capFn] of Object.entries(ARMS)) {
            let over = cfg.gap_thresholds.map(() => 0), n = 0;
            for (let s = 0; s < cfg.seeds; s++) {
                const { gaps } = simulate(trueRank, capFn, cfg.size, cfg.horizon,
                                          payload.seed + 7919 * s, null);
                for (const d of gaps) {
                    n++;
                    cfg.gap_thresholds.forEach((th, j) => { if (d > th) over[j]++; });
                }
            }
            row.arms[name] = { over_share: over.map((c) => c / n), attempts: n };
        }
        out.abilities.push(row);
    }
    // The unlimited-strength solver: always correct, so the whole trajectory is
    // deterministic (the neutral slot's pick is a function of the serve rank alone).
    // The seed is threaded anyway so the run stays reproducible if that changes.
    for (const [name, capFn] of Object.entries(ARMS)) {
        const { reached } = simulate(null, capFn, cfg.size, cfg.badge_max_attempts,
                                     payload.seed, cfg.badge_rank);
        out.badge[name] = reached;
    }
    return out;
}

// --- §2.4: what argmaxing information picks out of a difficulty-matched pool ------
// For each (board size, serve rank) pool: whether the top-quality candidate sits in
// each named motif tail, the pool's own rate in that tail, and the quality of the
// best and fifth-best candidate against the pool mean.
function infoSlot() {
    const cfg = payload.info_slot;
    const K = cfg.axes;
    const acc = {};
    for (const size of payload.board_sizes) {
        const argmaxTail = new Float64Array(K), poolTail = new Float64Array(K);
        let best = 0, fifthAdv = 0, sliceOverBest = 0, bestAdv = 0, wsum = 0;
        cfg.grid.forEach((x, gi) => {
            const w = cfg.weights[gi];
            const pool = poolAt(size, x, new Set(), payload.pool_size);
            const q = pool.map((gp) => quality[gp]);
            let am = 0;
            for (let j = 1; j < q.length; j++) if (q[j] > q[am]) am = j;
            const mean = q.reduce((a, b) => a + b, 0) / q.length;
            const sorted = [...q].sort((a, b) => b - a);
            for (let k = 0; k < K; k++) {
                if (motif[pool[am]] & (1 << k)) argmaxTail[k] += w;
                let c = 0;
                for (const gp of pool) if (motif[gp] & (1 << k)) c++;
                poolTail[k] += w * (c / pool.length);
            }
            best += w * (sorted[0] / mean);
            bestAdv += w * (sorted[0] - mean);
            fifthAdv += w * (sorted[cfg.advantage_index] - mean);
            // What the slot actually draws from: the highest-scoring INFO_TOP_FRACTION
            // of the pool (problem-loader.js's selectFromTopFraction, mirrored — its
            // own module imports state.js). Scored as the information a uniform draw
            // from that slice carries against the argmax's own, which is what the
            // slot gives up by not taking the maximum.
            const slice = Math.max(1, Math.ceil(sorted.length * cfg.top_fraction));
            let s = 0;
            for (let j = 0; j < slice; j++) s += sorted[j];
            sliceOverBest += w * (s / slice / sorted[0]);
            wsum += w;
        });
        acc[size] = {
            argmax_tail: Array.from(argmaxTail, (v) => v / wsum),
            pool_tail: Array.from(poolTail, (v) => v / wsum),
            best_over_pool: best / wsum,
            fifth_advantage: fifthAdv / bestAdv,
            slice_over_best: sliceOverBest / wsum,
        };
    }
    // Equal weight per board size: a player sees one size, and neither is the
    // canonical one.
    const sizes = payload.board_sizes;
    const avg = (f) => sizes.reduce((a, s) => a + f(acc[s]), 0) / sizes.length;
    const enrich = [];
    for (let k = 0; k < K; k++) {
        const a = avg((r) => r.argmax_tail[k]), p = avg((r) => r.pool_tail[k]);
        enrich.push({ axis: k, argmax: a, pool: p, ratio: p > 0 ? a / p : null });
    }
    return {
        motif_z: MOTIF_Z,
        per_size: acc,
        tails: enrich,
        best_over_pool: avg((r) => r.best_over_pool),
        fifth_advantage: avg((r) => r.fifth_advantage),
        slice_over_best: avg((r) => r.slice_over_best),
    };
}

// --- §2.11: the histogram's own estimator, per player -----------------------------
// computeRank (pwa/src/leaderboard.js) with the params lookup already done by the
// producer: flatten each buffered novel attempt through the shared transform, run the
// shared window estimator, report the SHOWN rank — the column the leaderboard stores
// and the histogram bins.
function community() {
    const users = payload.community.users.map((u) => {
        const flat = [];
        for (const [r, s, f, ok] of u.attempts) {
            const a = flattenAttempt(r, s, f, ok, CURVE);
            if (a) flat.push(a);
        }
        if (!flat.length) return { id: u.id, usable: 0, rank: null };
        const est = estimate(flat, serveGapCap);
        return { id: u.id, usable: flat.length, rank: est.displayRank, se: est.se };
    });
    return { rank_start_min: RANK_START_MIN, buffer_cap: WIN_CAP, users };
}

process.stdout.write(JSON.stringify({
    serve_margin: serveMargin(),
    info_slot: infoSlot(),
    community: community(),
}) + '\n');
