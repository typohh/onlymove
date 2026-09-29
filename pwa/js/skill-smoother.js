// js/skill-smoother.js
//
// State-space skill smoother for the Rank tab's career track. Skill is a hidden
// level that wanders as a random walk in calendar time; every first-time attempt
// is one noisy 4PL reading of it, taken through a per-SESSION form offset that is
// drawn fresh at each session start and held for the session:
//
//   S(t') = S(t) + N(0, C·(1 − exp(−q·days(t' − t) / C)))   q = drift, C = gap cap
//   F     ~ N(0, formSd²), redrawn when a gap of SESSION_GAP_MS or more opens a session
//   P(correct_j) = c_j + (1 − c_j − LAPSE)·σ(slope_j·(S + F − x_j) + κ_j)
//
// The step variance is q·days for a short step and saturates at C for a long one:
// skill changes through play, while an absence adds a bounded one-off uncertainty
// rather than one that grows with its length. The form offset is what keeps the
// drift honest: without it a good or a bad sitting can only be explained as skill
// moving, so the fitted drift absorbs the session-to-session swing and the skill
// line follows every session.
//
// The answer is the posterior of S at any time t given the WHOLE history (a
// Laplace filter forward, Rauch–Tung–Striebel backward). What that buys over a
// fixed-width kernel is that the amount of history behind an estimate follows the
// data: dense play leans on recent attempts, sparse play on a longer stretch, and
// across a gap the uncertainty grows by what the gap allows while both sides of it
// still pin the level. A short return session after a long absence is read
// against the level before it, never fitted on its own.
//
// The start of a career is the shipped population rank prior (PRIOR_MEAN /
// PRIOR_SD, a Gaussian summary of it), so the first attempts move the estimate
// away from where new players land rather than from nowhere.
//
// q, C and formSd are fitted by pipeline.fit_population_constants (population
// maximum likelihood of every attempt's outcome given the same player's earlier
// ones, through the Python mirror of this filter in utils/rating/skill_smoother.py)
// and inlined from population_constants.json. Mandatory, no fallback.
//
// Pure: imports no app state, so it runs under node unmodified (the twin test and
// ablations/skill_drift_fit.mjs).

import { clearSpan } from './irt.js';
import { sigmoid } from './math-utils.js';
import { PRIOR_MEAN, PRIOR_SD } from '../shared/rank-window.js';
// Baked at build time from the pipeline's output (pwa/tsumegos/population_constants.json).
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

const DAY_MS = 24 * 3600 * 1000;

// A gap at least this long ends a session. Near the within-session correlation
// length of the form kernel, rounded to an even divisor of the day.
export const SESSION_GAP_MS = 6 * 3600 * 1000;

const FITTED = (() => {
    const c = POPULATION_CONSTANTS.skill_smoother;
    const v = c && {
        driftVarPerDay: c.drift_var_per_day,
        gapVarCap: c.gap_var_cap,
        formSd: c.form_sd,
    };
    if (!v || !Object.values(v).every(x => Number.isFinite(x) && x > 0)) {
        throw new Error('population_constants.json has no usable skill_smoother block '
            + '(run pipeline.fit_population_constants, then package_for_pwa)');
    }
    return v;
})();

export const DEFAULT_PARAMS = Object.freeze({
    ...FITTED,
    priorMean: PRIOR_MEAN,
    priorSd: PRIOR_SD,
});

// Skill variance a step of `days` adds: q·days, saturating at the cap.
function stepVar(days, params) {
    return -Math.expm1(-params.driftVarPerDay * days / params.gapVarCap) * params.gapVarCap;
}

// One attempt's Laplace update of the combined reading u = S + F, prior N(mu, Pu).
// Returns the posterior mode and variance (curvature by the 4PL Fisher information,
// the same information every other estimator here uses) and the predictive
// probability of the observed outcome under the prior (the probit approximation of
// the logistic-normal integral), which is what the fit scores.
function laplaceUpdate(mu, Pu, a) {
    const span = clearSpan(a.floor);
    const zPred = (a.slope * (mu - a.x) + a.kappa) / Math.sqrt(1 + Math.PI * a.slope * a.slope * Pu / 8);
    const pCorrect = a.floor + span * sigmoid(zPred);
    const pObs = a.correct ? pCorrect : 1 - pCorrect;

    let u = mu, h = 0;
    for (let it = 0; it < 30; it++) {
        const praw = sigmoid(a.slope * (u - a.x) + a.kappa);
        let p = a.floor + span * praw;
        if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
        const g = span * a.slope * praw * (1 - praw);   // dP/du
        h = g * g / (p * (1 - p));
        const score = (a.correct - p) * g / (p * (1 - p)) - (u - mu) / Pu;
        let step = score / (1 / Pu + h);
        if (step > 2) step = 2; else if (step < -2) step = -2;
        u += step;
        if (Math.abs(step) < 1e-7) break;
    }
    return { u, Pu2: 1 / (1 / Pu + h), pObs };
}

// Forward Laplace filter over time-sorted flattened attempts ({t, x, slope, floor,
// kappa, correct}). The state is [S, F] with a symmetric covariance [pSS, pSF, pFF].
// Returns the per-attempt predicted and filtered moments, the session-start flags
// and the summed log predictive probability.
function filter(attempts, params) {
    const n = attempts.length;
    const f2 = params.formSd * params.formSd;
    const mp = new Float64Array(2 * n), Pp = new Float64Array(3 * n);
    const mf = new Float64Array(2 * n), Pf = new Float64Array(3 * n);
    const reset = new Uint8Array(n);
    let mS = params.priorMean, mF = 0;
    let pSS = params.priorSd * params.priorSd, pSF = 0, pFF = f2;
    let logLik = 0;
    for (let j = 0; j < n; j++) {
        if (j > 0) {
            const dt = attempts[j].t - attempts[j - 1].t;
            pSS += stepVar(dt / DAY_MS, params);
            if (dt >= SESSION_GAP_MS) { reset[j] = 1; mF = 0; pSF = 0; pFF = f2; }
        } else {
            reset[j] = 1;
        }
        mp[2 * j] = mS; mp[2 * j + 1] = mF;
        Pp[3 * j] = pSS; Pp[3 * j + 1] = pSF; Pp[3 * j + 2] = pFF;

        // Observation of u = S + F: Laplace in u, then the Gaussian pseudo-observation
        // (z, R) that reproduces that posterior conditions the full state.
        const mu = mS + mF, Pu = pSS + 2 * pSF + pFF;
        const { u, Pu2, pObs } = laplaceUpdate(mu, Pu, attempts[j]);
        logLik += Math.log(Math.max(pObs, 1e-300));
        const hEff = 1 / Pu2 - 1 / Pu;
        if (hEff > 1e-12) {
            const R = 1 / hEff;
            const z = R * (u / Pu2 - mu / Pu);
            const Sv = Pu + R;
            const kS = (pSS + pSF) / Sv, kF = (pSF + pFF) / Sv;
            const innov = z - mu;
            mS += kS * innov; mF += kF * innov;
            pSS -= kS * kS * Sv; pSF -= kS * kF * Sv; pFF -= kF * kF * Sv;
        }
        mf[2 * j] = mS; mf[2 * j + 1] = mF;
        Pf[3 * j] = pSS; Pf[3 * j + 1] = pSF; Pf[3 * j + 2] = pFF;
    }
    return { mp, Pp, mf, Pf, reset, logLik };
}

// Summed log predictive probability of every outcome given the ones before it —
// the population fit's objective. Forward pass only.
export function skillLogLik(attempts, params = DEFAULT_PARAMS) {
    return attempts.length ? filter(attempts, params).logLik : 0;
}

// One RTS step: the smoothed [S, F] at a point with filtered-or-propagated moments
// (m, P), given the smoothed and predicted moments of the NEXT attempt and whether
// that attempt opens a session (the form component does not carry across).
// Returns [mS, mF, pSS, pSF, pFF].
function rtsStep(m0, m1, a, b, d, nextReset, msN, PsN, mpN, PpN) {
    // Predicted covariance at the next attempt, and its inverse.
    const [pa, pb, pd] = PpN;
    const det = pa * pd - pb * pb;
    const ia = pd / det, ib = -pb / det, id = pa / det;
    // Cross-covariance P·Aᵀ, A = I within a session, diag(1, 0) across a reset.
    const c00 = a, c01 = nextReset ? 0 : b, c10 = b, c11 = nextReset ? 0 : d;
    // Gain C = P·Aᵀ·Pp⁻¹.
    const g00 = c00 * ia + c01 * ib, g01 = c00 * ib + c01 * id;
    const g10 = c10 * ia + c11 * ib, g11 = c10 * ib + c11 * id;
    const dm0 = msN[0] - mpN[0], dm1 = msN[1] - mpN[1];
    const e00 = PsN[0] - pa, e01 = PsN[1] - pb, e11 = PsN[2] - pd;
    // C·E·Cᵀ.
    const t00 = g00 * e00 + g01 * e01, t01 = g00 * e01 + g01 * e11;
    const t10 = g10 * e00 + g11 * e01, t11 = g10 * e01 + g11 * e11;
    return [
        m0 + g00 * dm0 + g01 * dm1,
        m1 + g10 * dm0 + g11 * dm1,
        a + t00 * g00 + t01 * g01,
        b + t00 * g10 + t01 * g11,
        d + t10 * g10 + t11 * g11,
    ];
}

// Fit the whole history once. Returns an object whose at(t) gives the posterior
// mean and standard deviation of SKILL (the form offset excluded) at any time t,
// exactly: between two attempts it is the random-walk bridge between their
// smoothed states — the step's variance spread evenly over its duration, so the
// bridge reproduces the step the filter took — and after the last it is the last
// state plus the step variance since.
// Returns null for an empty history.
export function smoothSkill(attempts, params = DEFAULT_PARAMS) {
    const n = attempts.length;
    if (!n) return null;
    const { mp, Pp, mf, Pf, reset, logLik } = filter(attempts, params);
    const ms = new Float64Array(2 * n), Ps = new Float64Array(3 * n);
    ms[2 * (n - 1)] = mf[2 * (n - 1)]; ms[2 * (n - 1) + 1] = mf[2 * (n - 1) + 1];
    for (let k = 0; k < 3; k++) Ps[3 * (n - 1) + k] = Pf[3 * (n - 1) + k];
    const at3 = (arr, j) => [arr[3 * j], arr[3 * j + 1], arr[3 * j + 2]];
    const at2 = (arr, j) => [arr[2 * j], arr[2 * j + 1]];
    for (let j = n - 2; j >= 0; j--) {
        const r = rtsStep(mf[2 * j], mf[2 * j + 1], Pf[3 * j], Pf[3 * j + 1], Pf[3 * j + 2],
            reset[j + 1], at2(ms, j + 1), at3(Ps, j + 1), at2(mp, j + 1), at3(Pp, j + 1));
        ms[2 * j] = r[0]; ms[2 * j + 1] = r[1];
        Ps[3 * j] = r[2]; Ps[3 * j + 1] = r[3]; Ps[3 * j + 2] = r[4];
    }

    const times = attempts.map(a => a.t);
    const at = (t) => {
        // Last attempt at or before t.
        let lo = 0, hi = n;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (times[mid] <= t) lo = mid + 1; else hi = mid; }
        const j = lo - 1;
        if (j < 0) return { S: ms[0], se: Math.sqrt(Ps[0]) };   // before the first attempt
        if (j === n - 1) {
            const days = (t - times[j]) / DAY_MS;
            return { S: ms[2 * j], se: Math.sqrt(Ps[3 * j] + stepVar(days, params)) };
        }
        if (t === times[j]) return { S: ms[2 * j], se: Math.sqrt(Ps[3 * j]) };
        // Filtered state at j propagated to t (skill drifts, form held), then one
        // RTS step against the next attempt. The step's variance is split in
        // proportion to time, so the next attempt's predicted moments are the same
        // whether the walk is split at t or not.
        const frac = (t - times[j]) / (times[j + 1] - times[j]);
        const vPart = frac * stepVar((times[j + 1] - times[j]) / DAY_MS, params);
        const r = rtsStep(mf[2 * j], mf[2 * j + 1],
            Pf[3 * j] + vPart, Pf[3 * j + 1], Pf[3 * j + 2],
            reset[j + 1], at2(ms, j + 1), at3(Ps, j + 1), at2(mp, j + 1), at3(Pp, j + 1));
        return { S: r[0], se: Math.sqrt(Math.max(r[2], 0)) };
    };
    return { at, logLik };
}
