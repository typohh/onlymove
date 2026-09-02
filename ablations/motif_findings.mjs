// ablations/motif_findings.mjs
//
// The measurement half of `ablations/motif_findings_figures.py`: how many
// findings the motif pane shows a user, by history length, computed by RUNNING
// the shipped readings on recorded histories rather than restating them.
//
// Imported unmodified from the shipping tree:
//   pwa/js/motif-readings.js   motifReadings (both per-axis readings), isFlagged
//   pwa/shared/rank-window.js  estimateWindowRankFromFlat (the rank the readings
//                              are conditioned on), flattenAttempt, WIN_CAP
//   pwa/js/math-utils.js       quantizeEmbedding (the stored-vector resolution)
//   pwa/js/constants.js        MOTIF_Z, MODEL_INPUT_HISTORY_LENGTH
// The JSON-importing modules load through json_module_loader.mjs.
//
// MIRRORED, minimally, because their own modules reach app state — each named
// at its use site:
//   curveForSize        (pwa/js/rank-decomposition.js)
//   flattenAttempts'    main path (pwa/js/rank-decomposition.js)
//   TONE_STRONG_Z       (pwa/js/stats-weaknesses.js)
//
// Usage:  node ablations/motif_findings.mjs <payload.json>
// Reads the payload written by the producer, writes one JSON object to stdout.
// Nothing here draws a random number: the output is a function of the payload.

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./json_module_loader.mjs', import.meta.url);

const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const PWA = payload.pwa_root;
const imp = (rel) => import(pathToFileURL(PWA + rel).href);

const { motifReadings, isFlagged } = await imp('js/motif-readings.js');
const { quantizeEmbedding } = await imp('js/math-utils.js');
const { estimateWindowRankFromFlat, flattenAttempt, WIN_CAP } = await imp('shared/rank-window.js');
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
// TONE_STRONG_Z (stats-weaknesses.js), mirrored: the pane's full shade begins at
// twice the gate, so "a finding at two standard errors" is |z| past this.
const STRONG_Z = 2 * MOTIF_Z;

// A recorded attempt as the client would hold it: the published triple, the
// correct-move vector, and — on a wrong attempt — the played move's per-cell
// vector and the availability-null masses. Vectors pass through the same
// quantization every stored vector gets (attempt-recorder.js, refit-endpoint.js).
function toAttempt(r) {
    const [size, correct, rating, slope, floor, emb, picked, above, below] = r;
    const a = {
        size, correct: correct === 1, isRepeat: false, linked: false,
        problemRating: rating, problemSlope: slope, problemFloor: floor,
    };
    if (emb) a.embedding = quantizeEmbedding(emb);
    if (picked) a.pickedEmbedding = quantizeEmbedding(picked);
    if (above && below) { a.altAbove = above; a.altBelow = below; }
    return a;
}

// flattenAttempts (rank-decomposition.js), main path, mirrored: each attempt
// through its own size's curve via the shared transform.
const flat = (a) => flattenAttempt(a.problemRating, a.problemSlope, a.problemFloor,
                                   a.correct, curveFor(a.size));

const FIELDS = ['avail', 'one', 'two', 't_avail', 't_one', 't_two', 's_avail', 's_one', 's_two'];

function tally(readings) {
    const t = Object.fromEntries(FIELDS.map((f) => [f, 0]));
    if (!readings) return t;
    for (const r of readings) {
        for (const [kind, x] of [['t', r.tendency], ['s', r.strength]]) {
            if (!x || !Number.isFinite(x.z)) continue;
            t.avail++; t[`${kind}_avail`]++;
            if (isFlagged(x)) { t.one++; t[`${kind}_one`]++; }
            if (Math.abs(x.z) > STRONG_Z) { t.two++; t[`${kind}_two`]++; }
        }
    }
    return t;
}

// The user's rank with `end` attempts on record — the live estimator over that
// record (it reads its own last WIN_CAP). One per position, whatever widths
// are then read at it: the rank does not depend on the reading's window.
function rankAt(flats, end) {
    const est = estimateWindowRankFromFlat(
        flats.slice(Math.max(0, end - WIN_CAP), end).filter(Boolean));
    return est ? est.rank : NaN;
}

// The pane as the user would have seen it with `end` attempts on record: the
// readings over the last `windowLength` usable attempts of that record.
function windowTally(rows, theta, end, windowLength) {
    return tally(motifReadings(rows.slice(0, end), theta, K, curveFor, windowLength));
}

const FIRST = payload.buckets.filter((b) => b.mode === 'first');
const LAST = payload.buckets.filter((b) => b.mode === 'last');
if (FIRST.length + LAST.length !== payload.buckets.length) throw new Error('unknown bucket mode');

const out = {
    motif_z: MOTIF_Z, strong_z: STRONG_Z, K,
    shipped_window: MODEL_INPUT_HISTORY_LENGTH, rank_window: WIN_CAP,
    buckets: {},
};
for (const b of payload.buckets) out.buckets[b.name] = { users: [] };

function accumulator() {
    return { sum: Object.fromEntries(FIELDS.map((f) => [f, 0])), windows: 0 };
}
function add(acc, t) {
    for (const f of FIELDS) acc.sum[f] += t[f];
    acc.windows++;
}

for (const u of payload.users) {
    const rows = u.rows.map(toAttempt);
    const flats = rows.map(flat);
    const acc = {};
    // One window per account: its first n attempts, read at the shipped width.
    for (const b of FIRST) {
        if (rows.length < b.n) continue;
        acc[b.name] = accumulator();
        add(acc[b.name], windowTally(rows, rankAt(flats, b.n), b.n, MODEL_INPUT_HISTORY_LENGTH));
    }
    // Every position from n onward, the latest n attempts at each, at a width
    // of n — the shipped width for the shipped bucket, wider for the
    // hypothetical ones. The rank at a position is shared across the widths.
    const eligible = LAST.filter((b) => rows.length >= b.n);
    if (eligible.length) {
        for (const b of eligible) acc[b.name] = accumulator();
        const from = Math.min(...eligible.map((b) => b.n));
        for (let end = from; end <= rows.length; end++) {
            const theta = rankAt(flats, end);
            for (const b of eligible) {
                if (end >= b.n) add(acc[b.name], windowTally(rows, theta, end, b.n));
            }
        }
    }
    for (const [name, a] of Object.entries(acc)) {
        const mean = Object.fromEntries(FIELDS.map((f) => [f, a.sum[f] / a.windows]));
        out.buckets[name].users.push({ id: u.id, history: rows.length, windows: a.windows, ...mean });
    }
}

process.stdout.write(JSON.stringify(out) + '\n');
