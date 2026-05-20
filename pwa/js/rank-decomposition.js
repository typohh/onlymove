// js/rank-decomposition.js
//
// Two-component rank decomposition:  R(t) = S(t) + M(t)
//   S(t) — slow "skill"   GP, ℓ_S ~ months,   σ_S ~ 0.7 ranks
//   M(t) — fast "form"    GP, ℓ_M ~ 1 hour,   σ_M ~ 1 rank
//   y_i | t_i, r_i  ~  Bernoulli( σ( κ · (R(t_i) − r_i) + logit(0.8) ) )
//
// What the *correct* fit would be
// -------------------------------
// Joint state-space filter on the stacked Markov state (S_t, M_t) with a
// Bernoulli observation and EM for the hyperparameters. We approximate it.
//
// What we actually do (this file)
// -------------------------------
// SKILL S(t): a **kernel-weighted pooled logistic** over the raw attempts.
//   To get S at time t we fit one R by MLE over every attempt, weighting
//   attempt i by a Gaussian time-kernel exp(−½((t_i−t)/τ_S)²):
//       P(correct_i) = σ( κ·(R − r_i) + logit(0.8) )
//   This estimates logit(mean p), NOT mean(logit p). That distinction matters:
//   the older method fit a point rank R_b per 6h bucket and then *smoothed the
//   R_b's*. Because logit is convex for p>0.5 (where most users sit), per-
//   session noise rectifies through that convex link into a systematic UPWARD
//   bias — averaging point-MLEs of a convex transform overshoots (Jensen). The
//   smaller the bucket, the worse it got. Pooling the likelihood first removes
//   the bias by construction, and recency falls out of the kernel.
//
// FORM M(t): the session-to-session deviation from skill. Per 6h bucket we fit
//   the session's own logistic (init at S, no anchor-to-served prior) and take
//   M = (session R − S), shrunk toward 0 with a σ_M ≈ 1 rank prior. The earlier
//   code anchored each bucket's prior to that bucket's *mean served difficulty*
//   (rBar) — circular, since serving is driven by the displayed rank. The prior
//   now lives only in the form's shrink-to-skill, never in the skill itself.
//
// σ_M (swing) is the precision-weighted MLE of the form spread over sessions,
// with a detection floor. Skill, form, and SEs are evaluated at the 6h bucket
// centroids; evaluateAt() interpolates them for a fine chart grid.

import { TOTAL_RANKS } from './constants.js';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

// 6h bucket = the time grid S/M are reported on (≈ one session for most users).
const BUCKET_MS = 6 * 60 * 60 * 1000;

// Kernel timescales. Gaussian, both directions.
const TAU_S_MS = 30 * DAY_MS;              // skill: ~1-month scale
const TAU_M_MS = 60 * 60 * 1000;           // form: 60 min

// Truncate the Gaussian at 4σ (e^(−8) ≈ 3e-4 past it). Keeps pooling bounded
// and lets us window the (time-sorted) attempts by binary search.
const KERNEL_TRUNC_SIGMA = 4;

// Problem-rank calibration target (matches model.js): rank = the problem rank
// you'd clear at this rate.
const CALIBRATION_P = 0.8;
const LOGIT_CALIBRATION = Math.log(CALIBRATION_P / (1 - CALIBRATION_P));

// Logit-per-rank — keep in sync with model.js LOGIT_PER_RANK_PRIOR.
const LOGIT_PER_RANK = 0.28;

// Physical prior on the marginal stdev of form M(t), in ranks (~±3 ranks at
// 3σ for severe swings ⇒ σ_M ≈ 1). Used only to shrink the form toward 0 when a
// session's data is thin; skill gets no such prior.
const SIGMA_M_PRIOR = 1.0;

// Sane clamp on a fitted rank_index (covers ~40k..~30d) so a degenerate
// all-correct / all-wrong fit can't run to ±Infinity.
const RANK_CLAMP_LO = -40;
const RANK_CLAMP_HI = 60;

function gaussianWeight(dtOverTau) {
    if (Math.abs(dtOverTau) > KERNEL_TRUNC_SIGMA) return 0;
    return Math.exp(-0.5 * dtOverTau * dtOverTau);
}

function sigmoid(x) {
    if (x > 40) return 1;
    if (x < -40) return 0;
    return 1 / (1 + Math.exp(-x));
}

// First index with attempts[i].t >= lo (attempts sorted ascending by t).
function lowerBound(attempts, lo) {
    let a = 0, b = attempts.length;
    while (a < b) { const m = (a + b) >> 1; if (attempts[m].t < lo) a = m + 1; else b = m; }
    return a;
}

// Kernel-weighted pooled logistic MLE for the rank at time t0.
//   P(correct_i) = σ(κ(R − r_i) + logit(0.8)),  weight w_i = gaussianWeight(Δt/τ)
// Returns { R, se, effN }. se is the kernel-weighted (sandwich) SE; effN is the
// Kish effective attempt count in the kernel. attempts MUST be sorted by t.
function poolLogistic(attempts, t0, tau) {
    const win = KERNEL_TRUNC_SIGMA * tau;
    const lo = lowerBound(attempts, t0 - win);
    const hi = lowerBound(attempts, t0 + win + 1);
    if (hi <= lo) return { R: NaN, se: Infinity, effN: 0 };

    // Precompute weights + Kish effN, and seed R at the weighted-mean rank.
    let sumW = 0, sumW2 = 0, sumWr = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        sumW += w; sumW2 += w * w; sumWr += w * attempts[j].rank;
    }
    if (sumW <= 0) return { R: NaN, se: Infinity, effN: 0 };
    const effN = (sumW * sumW) / sumW2;
    let R = sumWr / sumW;

    for (let it = 0; it < 15; it++) {
        let score = 0, info = 0;
        for (let j = lo; j < hi; j++) {
            const w = gaussianWeight((attempts[j].t - t0) / tau);
            if (w <= 0) continue;
            const p = sigmoid(LOGIT_PER_RANK * (R - attempts[j].rank) + LOGIT_CALIBRATION);
            score += w * LOGIT_PER_RANK * (attempts[j].correct - p);
            info += w * LOGIT_PER_RANK * LOGIT_PER_RANK * p * (1 - p);
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 2) step = 2; else if (step < -2) step = -2;
        R += step;
        if (R < RANK_CLAMP_LO) R = RANK_CLAMP_LO;
        else if (R > RANK_CLAMP_HI) R = RANK_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }

    // Sandwich SE for a weighted M-estimator: Var ≈ Σ w²v / (Σ wv)², v=κ²p(1−p).
    let A = 0, B = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        const p = sigmoid(LOGIT_PER_RANK * (R - attempts[j].rank) + LOGIT_CALIBRATION);
        const v = LOGIT_PER_RANK * LOGIT_PER_RANK * p * (1 - p);
        A += w * v; B += w * w * v;
    }
    const se = A > 0 ? Math.sqrt(B) / A : Infinity;
    return { R, se, effN };
}

// One session's own logistic (no served-difficulty prior), seeded at the skill
// estimate. Returns { R, sigma2 } where sigma2 = 1/Fisher-info at R. Degenerate
// (all-correct / all-wrong) sessions clamp R and get sigma2 → large, so the
// precision-weighted form/σ_M downstream ignore them.
function fitSession(att, skillInit) {
    let R = skillInit;
    for (let it = 0; it < 20; it++) {
        let score = 0, info = 0;
        for (const a of att) {
            const p = sigmoid(LOGIT_PER_RANK * (R - a.rank) + LOGIT_CALIBRATION);
            score += LOGIT_PER_RANK * (a.correct - p);
            info += LOGIT_PER_RANK * LOGIT_PER_RANK * p * (1 - p);
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 3) step = 3; else if (step < -3) step = -3;
        R += step;
        if (R < RANK_CLAMP_LO) R = RANK_CLAMP_LO;
        else if (R > RANK_CLAMP_HI) R = RANK_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }
    let info = 0;
    for (const a of att) {
        const p = sigmoid(LOGIT_PER_RANK * (R - a.rank) + LOGIT_CALIBRATION);
        info += LOGIT_PER_RANK * LOGIT_PER_RANK * p * (1 - p);
    }
    return { R, sigma2: 1 / Math.max(info, 1e-9) };
}

// Flatten answerHistory to sorted [{t, rank, correct}] (skip repeats / linked).
function flattenAttempts(answerHistory) {
    return answerHistory
        .filter(a => typeof a.problemRating === 'number'
                     && typeof a.date === 'number'
                     && !a.isRepeat
                     && !a.linked)
        .map(a => ({ t: a.date, rank: a.problemRating * TOTAL_RANKS, correct: a.correct ? 1 : 0 }))
        .sort((a, b) => a.t - b.t);
}

// Group sorted attempts into 6h buckets → [{t, n, att:[{rank,correct}]}].
function bucketize(attempts) {
    const map = new Map();
    for (const a of attempts) {
        const key = Math.floor(a.t / BUCKET_MS);
        let b = map.get(key);
        if (!b) { b = { tSum: 0, att: [] }; map.set(key, b); }
        b.tSum += a.t;
        b.att.push({ rank: a.rank, correct: a.correct });
    }
    const arr = [...map.values()].map(b => ({ t: b.tSum / b.att.length, n: b.att.length, att: b.att }));
    arr.sort((a, b) => a.t - b.t);
    return arr;
}

export function computeRankDecomposition(answerHistory, _options = {}) {
    const attempts = flattenAttempts(answerHistory);
    if (attempts.length < 20) return null;
    const buckets = bucketize(attempts);
    const N = buckets.length;
    if (N < 10) return null;

    const S = new Array(N), seS = new Array(N), effNS = new Array(N);
    const M = new Array(N), seM = new Array(N), effNM = new Array(N);
    const sm2 = SIGMA_M_PRIOR * SIGMA_M_PRIOR;

    for (let i = 0; i < N; i++) {
        const sk = poolLogistic(attempts, buckets[i].t, TAU_S_MS);
        S[i] = sk.R; seS[i] = sk.se; effNS[i] = sk.effN;

        // Form: this session's deviation from skill, shrunk toward 0.
        const f = fitSession(buckets[i].att, Number.isFinite(sk.R) ? sk.R : buckets[i].att[0].rank);
        buckets[i].R = f.R;
        buckets[i].sigma2 = f.sigma2;
        const dev = f.R - (Number.isFinite(sk.R) ? sk.R : f.R);
        const shrink = sm2 / (sm2 + f.sigma2);
        M[i] = dev * shrink;
        seM[i] = Math.sqrt(sm2 * f.sigma2 / (sm2 + f.sigma2));
        effNM[i] = buckets[i].n;
    }

    // σ_M (session-to-session swing): MLE of the form spread from the per-session
    // deviations dev_i ~ N(0, σ_M² + σ_ε,i²). Precision-weighted, so noisy
    // (single-attempt / saturated) sessions contribute ~nothing.
    const residuals = buckets.map((b, i) => (Number.isFinite(S[i]) ? b.R - S[i] : 0));
    const noiseVars = buckets.map(b => b.sigma2);
    let sigmaM2 = 0;
    for (let iter = 0; iter < 30; iter++) {
        let score = 0, info = 0;
        for (let i = 0; i < residuals.length; i++) {
            const v = sigmaM2 + noiseVars[i];
            const r2 = residuals[i] * residuals[i];
            score += (r2 - v) / (2 * v * v);
            info += 1 / (2 * v * v);
        }
        if (sigmaM2 === 0 && score <= 0) break;
        const delta = score / Math.max(info, 1e-12);
        const next = Math.max(0, sigmaM2 + delta);
        if (Math.abs(next - sigmaM2) < 1e-7) { sigmaM2 = next; break; }
        sigmaM2 = next;
    }
    const sigmaM = Math.sqrt(sigmaM2);
    let info0 = 0;
    for (const sv of noiseVars) info0 += 1 / (2 * sv * sv);
    const seSigmaM2_null = info0 > 0 ? 1 / Math.sqrt(info0) : Infinity;
    const sigmaMDetectionFloor = Math.sqrt(Math.max(0, 1.645 * seSigmaM2_null));

    // Long-term trend over well-supported buckets (effN_S ≥ 10 attempts).
    let firstIdx = -1, lastIdx = -1;
    for (let i = 0; i < N; i++) {
        if (effNS[i] >= 10 && Number.isFinite(S[i])) {
            if (firstIdx < 0) firstIdx = i;
            lastIdx = i;
        }
    }
    if (lastIdx < 0) lastIdx = N - 1;
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

// Evaluate Ŝ(t), M̂(t) and their SEs at an arbitrary time t by linear
// interpolation of the per-bucket arrays (bucket centroids are 6h apart, so a
// straight line between them is plenty for a chart). Returns null if t has no
// supporting bucket within the skill kernel.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { buckets, S, M, seS, seM, effNS, effNM } = decomp;
    const N = buckets.length;
    if (N === 0) return null;

    // Clamp to endpoints outside the bucket span.
    if (t <= buckets[0].t) {
        if (!Number.isFinite(S[0])) return null;
        return { S: S[0], M: M[0], seS: seS[0], seM: seM[0], effNS: effNS[0], effNM: effNM[0] };
    }
    if (t >= buckets[N - 1].t) {
        const i = N - 1;
        if (!Number.isFinite(S[i])) return null;
        return { S: S[i], M: M[i], seS: seS[i], seM: seM[i], effNS: effNS[i], effNM: effNM[i] };
    }
    // Find the bracketing buckets and lerp.
    let a = 0, b = N - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (buckets[m].t <= t) a = m; else b = m; }
    if (!Number.isFinite(S[a]) || !Number.isFinite(S[b])) return null;
    const span = buckets[b].t - buckets[a].t;
    const u = span > 0 ? (t - buckets[a].t) / span : 0;
    const mix = (arr) => arr[a] + (arr[b] - arr[a]) * u;
    return {
        S: mix(S), M: mix(M),
        seS: mix(seS), seM: mix(seM),
        effNS: mix(effNS), effNM: mix(effNM),
    };
}
