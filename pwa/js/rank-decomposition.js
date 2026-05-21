// js/rank-decomposition.js
//
// Two-component rank decomposition:  R(t) = S(t) + M(t)
//   S(t) — slow "skill",  M(t) — fast "form".
//
// We estimate skill in RATING space and convert to rank with M only at the end.
// Rationale (the rank loop): the survey calibrates M (rating→rank) so that a
// user's 80%-clear *rating* maps to their reported rank, and the same M labels
// the problems. So the displayed skill must also be computed in rating space —
// "the rating at which you clear CALIBRATION_P" — and then mapped with M. A
// fixed-slope logistic in *rank* space is biased (the slope isn't constant per
// rank once M is nonlinear), which is what made a 4d read as 6d. So:
//
//   1. Per attempt, work in rating space: x = its raw rating (problemRawRating;
//      legacy entries fall back to M⁻¹(problemRating·29)).
//   2. S(t) = kernel-weighted pooled logistic over x at the 80% target — your
//      rating over time. Form M(t) = per-session deviation from S, shrunk to 0.
//   3. Map S, M and their SEs to rank with M (delta-method for the SEs) so the
//      summary and chart stay in rank units.
//
// M (rating_to_rank.json) is required: package_for_pwa ships only raw ratings,
// so the client can't rank anything without it. main.js awaits loadRatingRankMap()
// at init and aborts if it fails; computeRankDecomposition returns null if the
// curve for the active size is missing.

import { state } from './state.js';
import { TOTAL_RANKS } from './constants.js';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

const BUCKET_MS = 6 * 60 * 60 * 1000;       // 6h bucket = the S/M time grid
const TAU_S_MS = 30 * DAY_MS;               // skill: ~1-month scale
const TAU_M_MS = 60 * 60 * 1000;            // form: 60 min
const KERNEL_TRUNC_SIGMA = 4;

// rank = the level you clear at this rate (matches model.js / fit_rank_curves).
const CALIBRATION_P = 0.8;
const LOGIT_CALIBRATION = Math.log(CALIBRATION_P / (1 - CALIBRATION_P));

// Success-curve slope, in rating space (the only space now — M is required).
// Must match pipeline/fit_rank_curves.LOGIT_PER_RATING.
const LOGIT_PER_RATING = 2.5;

// Form (session swing) prior σ_M, in rating units (~1 rank near the user).
const SIGMA_M_PRIOR_RATING = 0.4;

const X_CLAMP_LO = -40;
const X_CLAMP_HI = 60;

// ---------------------------------------------------------------------------
// M: rating → rank map (pwa/tsumegos/rating_to_rank.json), per board size.
// ---------------------------------------------------------------------------
let _ratingRankMap = null;
let _loadPromise = null;

export function loadRatingRankMap() {
    if (_loadPromise) return _loadPromise;
    _loadPromise = fetch(`tsumegos/rating_to_rank.json?v=${APP_VERSION}`)
        .then(r => (r.ok ? r.json() : null))
        .then(m => { _ratingRankMap = m; return m; })
        .catch(() => { _ratingRankMap = null; return null; });
    return _loadPromise;
}

function curveForSize(size) {
    const m = _ratingRankMap && _ratingRankMap[String(size)];
    return (m && Array.isArray(m.ratings) && Array.isArray(m.rank_indices)
            && m.ratings.length >= 2) ? m : null;
}

// Linear interp with end-slope extrapolation; xs strictly ascending.
function interpExtrap(x, xs, ys) {
    const n = xs.length;
    if (x <= xs[0]) return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0]);
    if (x >= xs[n - 1]) return ys[n - 1] + (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]) * (x - xs[n - 1]);
    let a = 0, b = n - 1;
    while (b - a > 1) { const md = (a + b) >> 1; if (xs[md] <= x) a = md; else b = md; }
    return ys[a] + (ys[b] - ys[a]) * (x - xs[a]) / (xs[b] - xs[a]);
}

function applyM(rating, c) { return interpExtrap(rating, c.ratings, c.rank_indices); }
function Mslope(rating, c) {
    const h = 1e-3;
    return (applyM(rating + h, c) - applyM(rating - h, c)) / (2 * h);
}
function Minv(rankIndex, c) { return interpExtrap(rankIndex, c.rank_indices, c.ratings); }

// Raw rating → normalized 0..1 rank for `size` (rank_index / TOTAL_RANKS, the
// scale getDescriptiveRank and the recall/weakness models use). Returns null
// when M isn't loaded; the whole client converts ratings to ranks through this,
// so package_for_pwa never has to.
export function normalizedRankForSize(rawRating, size) {
    const c = curveForSize(size);
    if (!c || typeof rawRating !== 'number') return null;
    return applyM(rawRating, c) / TOTAL_RANKS;
}

export function isRatingRankMapLoaded() {
    return _ratingRankMap != null;
}

// Stable per-size signature of the loaded curve. Stored ranks are invalidated
// when this changes (a fit_rank_curves re-fit that didn't also change ratings).
export function ratingRankCurveVersion(size) {
    const c = curveForSize(size);
    return c ? JSON.stringify([c.ratings, c.rank_indices]) : null;
}

// ---------------------------------------------------------------------------

function gaussianWeight(dtOverTau) {
    if (Math.abs(dtOverTau) > KERNEL_TRUNC_SIGMA) return 0;
    return Math.exp(-0.5 * dtOverTau * dtOverTau);
}

function sigmoid(x) {
    if (x > 40) return 1;
    if (x < -40) return 0;
    return 1 / (1 + Math.exp(-x));
}

function lowerBound(attempts, lo) {
    let a = 0, b = attempts.length;
    while (a < b) { const m = (a + b) >> 1; if (attempts[m].t < lo) a = m + 1; else b = m; }
    return a;
}

// Kernel-weighted pooled logistic MLE for the working coordinate (rating, or
// rank in fallback) at time t0: P(correct)=σ(slope·(R−x)+logit(0.8)).
// Returns { R, se, effN }; attempts MUST be sorted by t.
function poolLogistic(attempts, t0, tau, slope) {
    const win = KERNEL_TRUNC_SIGMA * tau;
    const lo = lowerBound(attempts, t0 - win);
    const hi = lowerBound(attempts, t0 + win + 1);
    if (hi <= lo) return { R: NaN, se: Infinity, effN: 0 };

    let sumW = 0, sumW2 = 0, sumWx = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        sumW += w; sumW2 += w * w; sumWx += w * attempts[j].x;
    }
    if (sumW <= 0) return { R: NaN, se: Infinity, effN: 0 };
    const effN = (sumW * sumW) / sumW2;
    let R = sumWx / sumW;

    for (let it = 0; it < 15; it++) {
        let score = 0, info = 0;
        for (let j = lo; j < hi; j++) {
            const w = gaussianWeight((attempts[j].t - t0) / tau);
            if (w <= 0) continue;
            const p = sigmoid(slope * (R - attempts[j].x) + LOGIT_CALIBRATION);
            score += w * slope * (attempts[j].correct - p);
            info += w * slope * slope * p * (1 - p);
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 2) step = 2; else if (step < -2) step = -2;
        R += step;
        if (R < X_CLAMP_LO) R = X_CLAMP_LO; else if (R > X_CLAMP_HI) R = X_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }

    // Sandwich SE: Var ≈ Σ w²v / (Σ wv)², v = slope²·p(1−p).
    let A = 0, B = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        const p = sigmoid(slope * (R - attempts[j].x) + LOGIT_CALIBRATION);
        const v = slope * slope * p * (1 - p);
        A += w * v; B += w * w * v;
    }
    const se = A > 0 ? Math.sqrt(B) / A : Infinity;
    return { R, se, effN };
}

// One session's own logistic (no prior), seeded at the skill estimate.
function fitSession(att, skillInit, slope) {
    let R = skillInit;
    for (let it = 0; it < 20; it++) {
        let score = 0, info = 0;
        for (const a of att) {
            const p = sigmoid(slope * (R - a.x) + LOGIT_CALIBRATION);
            score += slope * (a.correct - p);
            info += slope * slope * p * (1 - p);
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 3) step = 3; else if (step < -3) step = -3;
        R += step;
        if (R < X_CLAMP_LO) R = X_CLAMP_LO; else if (R > X_CLAMP_HI) R = X_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }
    let info = 0;
    for (const a of att) {
        const p = sigmoid(slope * (R - a.x) + LOGIT_CALIBRATION);
        info += slope * slope * p * (1 - p);
    }
    return { R, sigma2: 1 / Math.max(info, 1e-9) };
}

// Flatten answerHistory to sorted [{t, x, correct}] in rating space (x = raw
// rating). `problemRating` is the raw rating; pre-migration entries that still
// carry a separate `problemRawRating` (raw) prefer it.
function flattenAttempts(answerHistory) {
    return answerHistory
        .filter(a => typeof a.problemRating === 'number'
                     && typeof a.date === 'number'
                     && !a.isRepeat
                     && !a.linked)
        .map(a => {
            const x = (typeof a.problemRawRating === 'number')
                ? a.problemRawRating
                : a.problemRating;
            return { t: a.date, x, correct: a.correct ? 1 : 0 };
        })
        .sort((a, b) => a.t - b.t);
}

function bucketize(attempts) {
    const map = new Map();
    for (const a of attempts) {
        const key = Math.floor(a.t / BUCKET_MS);
        let b = map.get(key);
        if (!b) { b = { tSum: 0, att: [] }; map.set(key, b); }
        b.tSum += a.t;
        b.att.push({ x: a.x, correct: a.correct });
    }
    const arr = [...map.values()].map(b => ({ t: b.tSum / b.att.length, n: b.att.length, att: b.att }));
    arr.sort((a, b) => a.t - b.t);
    return arr;
}

export function computeRankDecomposition(answerHistory, _options = {}) {
    const c = curveForSize(state.boardSize);
    if (!c) return null;   // no M ⇒ no rating→rank; the app shouldn't reach here.
    const slope = LOGIT_PER_RATING;
    const smPrior = SIGMA_M_PRIOR_RATING;
    // working coordinate (raw rating) → rank, plus the local derivative for the
    // delta-method SEs.
    const toRank = x => applyM(x, c);
    const dRank = x => Math.abs(Mslope(x, c));

    const attempts = flattenAttempts(answerHistory);
    if (attempts.length < 20) return null;
    const buckets = bucketize(attempts);
    const N = buckets.length;
    if (N < 10) return null;

    const sm2 = smPrior * smPrior;
    const S = new Array(N), seS = new Array(N), effNS = new Array(N);
    const M = new Array(N), seM = new Array(N), effNM = new Array(N);
    const devX = new Array(N);   // per-session form deviation, working units
    const sig2 = new Array(N);

    for (let i = 0; i < N; i++) {
        const sk = poolLogistic(attempts, buckets[i].t, TAU_S_MS, slope);   // skill in working units
        effNS[i] = sk.effN;
        const f = fitSession(buckets[i].att, Number.isFinite(sk.R) ? sk.R : buckets[i].att[0].x, slope);
        const skX = Number.isFinite(sk.R) ? sk.R : f.R;
        const dev = f.R - skX;
        const g = dRank(skX);                          // d(rank)/d(working) at the skill
        devX[i] = dev; sig2[i] = f.sigma2;
        const shrink = sm2 / (sm2 + f.sigma2);
        // Map skill + form to rank (delta-method on the SEs).
        S[i] = toRank(skX);
        seS[i] = sk.se * g;
        M[i] = dev * shrink * g;
        seM[i] = Math.sqrt(sm2 * f.sigma2 / (sm2 + f.sigma2)) * g;
        effNM[i] = buckets[i].n;
    }

    // σ_M (session swing) in working units, then mapped to rank at current skill.
    let sigmaM2 = 0;
    for (let iter = 0; iter < 30; iter++) {
        let score = 0, info = 0;
        for (let i = 0; i < N; i++) {
            const v = sigmaM2 + sig2[i];
            const r2 = devX[i] * devX[i];
            score += (r2 - v) / (2 * v * v);
            info += 1 / (2 * v * v);
        }
        if (sigmaM2 === 0 && score <= 0) break;
        const delta = score / Math.max(info, 1e-12);
        const next = Math.max(0, sigmaM2 + delta);
        if (Math.abs(next - sigmaM2) < 1e-7) { sigmaM2 = next; break; }
        sigmaM2 = next;
    }
    let info0 = 0;
    for (const sv of sig2) info0 += 1 / (2 * sv * sv);
    const seSigmaM2_null = info0 > 0 ? 1 / Math.sqrt(info0) : Infinity;

    let firstIdx = -1, lastIdx = -1;
    for (let i = 0; i < N; i++) {
        if (effNS[i] >= 10 && Number.isFinite(S[i])) {
            if (firstIdx < 0) firstIdx = i;
            lastIdx = i;
        }
    }
    if (lastIdx < 0) lastIdx = N - 1;
    const gLast = (lastIdx >= 0 && Number.isFinite(S[lastIdx]))
        ? dRank(Minv(S[lastIdx], c)) : 1;
    const sigmaM = Math.sqrt(sigmaM2) * gLast;
    const sigmaMDetectionFloor = Math.sqrt(Math.max(0, 1.645 * seSigmaM2_null)) * gLast;

    const haveSpan = firstIdx >= 0 && lastIdx > firstIdx;
    const longTermDrift = haveSpan ? S[lastIdx] - S[firstIdx] : null;
    const longTermSpanMs = haveSpan ? buckets[lastIdx].t - buckets[firstIdx].t : null;

    return {
        buckets, attempts,
        S, M, effNS, effNM, seS, seM,
        summary: {
            sigmaM,
            sigmaMDetectionFloor,
            longTermDrift,
            longTermSpanMs,
            currentSkill: Number.isFinite(S[lastIdx]) ? S[lastIdx] : null,
            currentSkillSE: seS[lastIdx],
            currentForm: M[lastIdx],
            currentFormSE: seM[lastIdx],
            bucketCount: N,
        },
    };
}

// Evaluate Ŝ(t), M̂(t) and SEs at arbitrary t by linear interpolation of the
// per-bucket (already rank-space) arrays. Returns null if t has no support.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { buckets, S, M, seS, seM, effNS, effNM } = decomp;
    const N = buckets.length;
    if (N === 0) return null;
    if (t <= buckets[0].t) {
        if (!Number.isFinite(S[0])) return null;
        return { S: S[0], M: M[0], seS: seS[0], seM: seM[0], effNS: effNS[0], effNM: effNM[0] };
    }
    if (t >= buckets[N - 1].t) {
        const i = N - 1;
        if (!Number.isFinite(S[i])) return null;
        return { S: S[i], M: M[i], seS: seS[i], seM: seM[i], effNS: effNS[i], effNM: effNM[i] };
    }
    let a = 0, b = N - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (buckets[m].t <= t) a = m; else b = m; }
    if (!Number.isFinite(S[a]) || !Number.isFinite(S[b])) return null;
    const span = buckets[b].t - buckets[a].t;
    const u = span > 0 ? (t - buckets[a].t) / span : 0;
    const mix = (arr) => arr[a] + (arr[b] - arr[a]) * u;
    return { S: mix(S), M: mix(M), seS: mix(seS), seM: mix(seM), effNS: mix(effNS), effNM: mix(effNM) };
}
