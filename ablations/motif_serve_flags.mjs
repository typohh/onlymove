// ablations/motif_serve_flags.mjs
//
// The node half of `ablations.motif_serve_figures`: for every user and every
// position t in their history, the serve directions the client's readings would
// have supported BEFORE attempt t — the shipped readings over rows[0, t) at the
// shipped window, conditioned on the live rank estimator over the same record.
// ALL FOUR kinds are emitted. The client makes categories of `weak` and `strong`
// only (`flaggedMotifCategories`, pwa/js/weakness.js); `overplay` and
// `underplay` are the two it declines, emitted here because this measurement is
// the reason it declines them.
//
// Imported unmodified: motifReadings/isFlagged, estimateWindowRankFromFlat,
// flattenAttempt, quantizeEmbedding, MODEL_INPUT_HISTORY_LENGTH. Mirrored (their
// modules reach app state): curveForSize, flattenAttempts' main path, and the
// strength half of flaggedMotifCategories.
//
// Usage:  node ablations/motif_serve_flags.mjs <payload.json> <out.json>

import { register } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./json_module_loader.mjs', import.meta.url);

const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const PWA = payload.pwa_root;
const imp = (rel) => import(pathToFileURL(PWA + rel).href);

const { motifReadings, isFlagged } = await imp('js/motif-readings.js');
const { quantizeEmbedding } = await imp('js/math-utils.js');
const { estimateWindowRankFromFlat, flattenAttempt, WIN_CAP } = await imp('shared/rank-window.js');
const { MODEL_INPUT_HISTORY_LENGTH } = await imp('js/constants.js');

const K = payload.K;
const LABELS = payload.labels;   // embedding_meta.json labels.components
const CURVES = JSON.parse(readFileSync(PWA + 'tsumegos/rating_to_rank.json', 'utf8'));
function curveFor(size) {
    const m = CURVES[String(size)];
    return (m && Array.isArray(m.ratings) && m.ratings.length >= 2) ? m : null;
}

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
const flat = (a) => flattenAttempt(a.problemRating, a.problemSlope, a.problemFloor,
                                   a.correct, curveFor(a.size));

// The strength half is flaggedMotifCategories (weakness.js) mirrored over a
// readings array; the tendency half is the two directions the client declines.
function categories(readings) {
    const out = [];
    if (!readings) return out;
    for (const { k, tendency, strength } of readings) {
        const comp = LABELS[k];
        if (!comp) continue;
        const tail = comp.positive ? 'positive' : (comp.negative ? 'negative' : null);
        if (!tail) continue;
        const dir = tail === 'negative' ? -1 : 1;
        if (strength && isFlagged(strength)) {
            const z = dir * strength.z;
            out.push(z < 0 ? [k, dir, 'weak'] : [k, -dir, 'strong']);
        }
        if (tendency && isFlagged(tendency)) {
            const z = dir * tendency.z;
            out.push(z > 0 ? [k, -dir, 'overplay'] : [k, dir, 'underplay']);
        }
    }
    return out;
}

const result = {};
let done = 0;
for (const u of payload.users) {
    const rows = u.rows.map(toAttempt);
    const flats = rows.map(flat);
    const perPos = [];
    for (let t = 0; t < rows.length; t++) {
        if (t < 20) { perPos.push([]); continue; }
        const est = estimateWindowRankFromFlat(
            flats.slice(Math.max(0, t - WIN_CAP), t).filter(Boolean));
        const theta = est ? est.rank : NaN;
        perPos.push(categories(motifReadings(rows.slice(0, t), theta, K, curveFor,
                                             MODEL_INPUT_HISTORY_LENGTH)));
    }
    result[u.id] = perPos;
    if (++done % 100 === 0) process.stderr.write(`  motif flags: ${done}/${payload.users.length} users\n`);
}
writeFileSync(process.argv[3], JSON.stringify(result));
