// ablations/skill_drift_fit.mjs
//
// The measurement half of `ablations/skill_drift_fit.py`: fits the Rank tab's
// skill smoother constants (SKILL_DRIFT_VAR_PER_DAY, FORM_SD in
// pwa/js/skill-smoother.js) by population maximum likelihood — the summed log
// probability of every first-time attempt's outcome given that player's attempts
// before it. Runs the SHIPPED smoother and flattenAttempt unmodified (JSON imports
// through json_module_loader.mjs), so the objective is exactly what the client
// computes.
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
const withParams = (logQ, logF) => ({ ...DEFAULT_PARAMS, driftVarPerDay: Math.exp(logQ), formSd: Math.exp(logF) });

// Golden-section search on one log-parameter, the other held.
function golden(f, lo, hi, tol = 0.01) {
    const g = (Math.sqrt(5) - 1) / 2;
    let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a);
    let fc = f(c), fd = f(d);
    while (b - a > tol) {
        if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
        else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
    }
    return (a + b) / 2;
}

// Coordinate ascent over (log drift variance, log form SD). The no-form arm pins
// the form SD at a negligible value, so the two arms differ in that term alone.
const NO_FORM = Math.log(1e-3);
function fit(withForm) {
    let lq = Math.log(0.01), lf = withForm ? Math.log(0.5) : NO_FORM;
    for (let round = 0; round < 4; round++) {
        lq = golden(x => total(withParams(x, lf)), Math.log(1e-5), Math.log(1));
        if (withForm) lf = golden(x => total(withParams(lq, x)), Math.log(0.02), Math.log(3));
    }
    return { driftVarPerDay: Math.exp(lq), formSd: Math.exp(lf), logLik: total(withParams(lq, lf)) };
}

const noForm = fit(false);
const form = fit(true);
// A profile over the drift at the fitted form SD, for the printed table.
const profile = [];
for (const q of [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]) {
    profile.push({ driftVarPerDay: q, logLik: total(withParams(Math.log(q), Math.log(form.formSd))) });
}
process.stdout.write(JSON.stringify({ nUsers: users.length, nAttempts, noForm, form, profile }));
