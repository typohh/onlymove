// ablations/skill_drift_fit.mjs
//
// The measurement half of `ablations/skill_drift_fit.py`: what each term of the
// Rank tab's skill smoother (pwa/js/skill-smoother.js) is worth, by population
// maximum likelihood — the summed log probability of every first-time attempt's
// outcome given that player's attempts before it — for the walk alone, the walk
// with the per-session form offset, and both with the cap on a step's variance.
// Runs the SHIPPED smoother and flattenAttempt unmodified (JSON imports through
// json_module_loader.mjs), so the objective is exactly what the client computes;
// the shipped constants themselves come from pipeline.fit_population_constants.
//
// Usage:  node ablations/skill_drift_fit.mjs <payload.json>
// The payload holds { pwa_root, users: [[t_ms, rating, slope, floor, correct, size], …] }
// with each user's rows in id order. Writes one JSON object to stdout.

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./json_module_loader.mjs', import.meta.url);

const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const PWA = payload.pwa_root;
const imp = (rel) => import(pathToFileURL(PWA + rel).href);
const { flattenAttempt } = await imp('shared/rank-window.js');
const { skillLogLik, DEFAULT_PARAMS } = await imp('js/skill-smoother.js');

const CURVES = JSON.parse(readFileSync(PWA + 'tsumegos/rating_to_rank.json', 'utf8'));

// Flatten exactly as flattenAttempts (rank-decomposition.js) does: each attempt
// through its own size's curve, malformed rows dropped, time-sorted.
const users = [];
let nAttempts = 0;
for (const rows of payload.users) {
    const flat = [];
    for (const [t, rating, slope, floor, correct, size] of rows) {
        const f = flattenAttempt(rating, slope, floor, correct, CURVES[String(size)]);
        if (f) flat.push({ ...f, t });
    }
    flat.sort((a, b) => a.t - b.t);
    if (flat.length) { users.push(flat); nAttempts += flat.length; }
}

const total = (params) => {
    let s = 0;
    for (const u of users) s += skillLogLik(u, params);
    return s;
};

// Golden-section search on one log-parameter, the others held.
function golden(f, lo, hi, tol = 0.02) {
    const g = (Math.sqrt(5) - 1) / 2;
    let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a);
    let fc = f(c), fd = f(d);
    while (b - a > tol) {
        if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
        else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
    }
    return (a + b) / 2;
}

// Three nested arms, each fitted by coordinate ascent over the logs of its free
// parameters. An arm without a term pins it where it has no effect: a negligible
// form SD, a cap far beyond any step's variance.
const NO_FORM = Math.log(1e-3), NO_CAP = Math.log(1e9);
const RANGES = { q: [Math.log(1e-5), Math.log(1)], c: [Math.log(0.05), Math.log(50)], f: [Math.log(0.02), Math.log(3)] };
function fit(free) {
    const v = { q: Math.log(0.04), c: free.includes('c') ? Math.log(1) : NO_CAP, f: free.includes('f') ? Math.log(1) : NO_FORM };
    const P = () => ({ ...DEFAULT_PARAMS, driftVarPerDay: Math.exp(v.q), gapVarCap: Math.exp(v.c), formSd: Math.exp(v.f) });
    for (let round = 0; round < 4; round++) {
        for (const k of free) {
            v[k] = golden(x => { const o = v[k]; v[k] = x; const r = total(P()); v[k] = o; return r; }, ...RANGES[k]);
        }
    }
    return { driftVarPerDay: Math.exp(v.q), gapVarCap: free.includes('c') ? Math.exp(v.c) : null,
             formSd: free.includes('f') ? Math.exp(v.f) : null, logLik: total(P()) };
}

const arms = { walk: fit(['q']), walkForm: fit(['q', 'f']), walkFormCap: fit(['q', 'f', 'c']) };
const shipped = total(DEFAULT_PARAMS);
process.stdout.write(JSON.stringify({ nUsers: users.length, nAttempts, arms, shipped, shippedParams: DEFAULT_PARAMS }));
