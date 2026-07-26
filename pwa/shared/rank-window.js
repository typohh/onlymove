// pwa/shared/rank-window.js
//
// The positional-linear WINDOW rank estimator — the live front-page rank — plus
// the matching positional-weighted solve-time median, factored out of
// rank-decomposition.js so the Cloudflare Worker (EGD leaderboard) and the browser
// client run BIT-IDENTICAL rank code. See docs/leaderboard_plan.md.
//
// Both consumers feed already-"flattened" attempts in INTERNAL-RANK space, sorted
// oldest -> newest: { x, slope, correct(0/1), floor, kappa }. The client flattens
// via rank-decomposition.js `flattenAttempts` (state/curve-aware); the Worker
// flattens from its buffer + problem_params + the inlined M curve. Keeping the fit
// itself here guarantees the two produce identical numbers.
//
// Imported helpers are pure + Worker-safe: sigmoid / weightedQuantile (math-utils.js
// touches the `math` global only inside normCdf, never called here) and clearSpan
// (irt.js, reads only the inlined population_constants.json).

import { sigmoid, weightedQuantile } from '../js/math-utils.js';
import { interpExtrap } from './rank-math.js';
import { clearSpan, clampFloor, clearKappa } from '../js/irt.js';

// --- Per-attempt transform to internal-rank space -------------------------------
// Shared so the client's flattenAttempts (rank-decomposition.js) and the Worker's
// buffer flatten produce IDENTICAL inputs. The problem's raw rating maps to the
// internal-rank index x = applyM(rating); the success-curve slope is the problem's
// own 2PL discrimination a_p converted to logit-per-internal-rank via M's local
// steepness. LOGIT_PER_RATING is only the fallback a_p for legacy entries without
// a stored slope; LOGIT_PER_RANK_FALLBACK is the slope when M' is unusable.
export const LOGIT_PER_RATING = 1.6;
export const LOGIT_PER_RANK_FALLBACK = 0.28;
const SLOPE_LO = 0.05, SLOPE_HI = 3.0;
const MSLOPE_FLOOR = 1e-3;

// Raw rating -> internal-rank index through a size's M curve {ratings, rank_indices}.
export function applyM(rating, curve) {
    return interpExtrap(rating, curve.ratings, curve.rank_indices);
}

function mSlope(rating, curve) {
    const h = 1e-3;
    return (applyM(rating + h, curve) - applyM(rating - h, curve)) / (2 * h);
}

// Clamp a converted per-attempt slope; degenerate/absent slope -> the rank fallback.
export function clampSlope(s) {
    if (!Number.isFinite(s) || s <= 0) return LOGIT_PER_RANK_FALLBACK;
    return Math.min(SLOPE_HI, Math.max(SLOPE_LO, s));
}

// a_p (raw 2PL discrimination) -> logit-per-internal-rank at `rating`, guarded
// against pathological tiny |M'|. Falls back to LOGIT_PER_RATING when a_p is absent.
export function slopeInRankSpace(rating, aP, curve) {
    const a = (typeof aP === 'number' && aP > 0) ? aP : LOGIT_PER_RATING;
    return clampSlope(a / Math.max(Math.abs(mSlope(rating, curve)), MSLOPE_FLOOR));
}

// One flattened attempt in internal-rank space: { x, slope, correct, floor, kappa }.
// `curve` is the attempt's size curve; `rating` its raw rating; `aP` its problemSlope
// (fallback applied if absent); `floorRaw` its problemFloor (3PL guessing floor).
// Returns null when the curve or rating is unusable — the caller may then apply its
// own legacy fallback (e.g. a stored normalized problemRank).
export function flattenAttempt(rating, aP, floorRaw, correct, curve) {
    if (!curve || typeof rating !== 'number') return null;
    const floor = clampFloor(floorRaw);
    return {
        x: applyM(rating, curve),
        slope: slopeInRankSpace(rating, aP, curve),
        correct: correct ? 1 : 0,
        floor,
        kappa: clearKappa(floor),
    };
}

// Working coordinate is the internal-rank index; clamp for very weak/strong users.
const X_CLAMP_LO = -15;
const X_CLAMP_HI = 45;

export const WIN_CAP = 500;                   // hard sample-count window + the ramp's denominator
const WIN_ITARGET = 16;                       // design steady-state info — the CONSTANT SE-floor fallback
const WIN_PRIOR_SD = 4;                       // weak prior on R, in stones
const WIN_PRIOR_INFO = 1 / (WIN_PRIOR_SD * WIN_PRIOR_SD);
const WIN_UCB_K = 1;                          // serve at estimate + k·(se − se_floor)
const WIN_SE_FLOOR = 1 / Math.sqrt(WIN_PRIOR_INFO + WIN_ITARGET / 2);

// --- Rank-dependent SE floor for the UCB serve bonus ----------------------------
// The serve bonus is k·(se − se_floor): serve above the estimate while the posterior
// is loose, collapse to the user's OWN rank as it tightens. But "tight" is not a
// global constant. The floor is the irreducible SE the window MLE would report with a
// FULL window of problems AT the user's rank, each cleared at the 80% calibration
// point — i.e. the SE at which serving MORE at-rank problems stops helping. That
// summed Fisher information is Σw · <I>: Σw the positional-weight mass of a full
// window, and <I> the mean per-attempt information of problems near the user's rank.
// Flat kyu curves (small slope) genuinely can't pin a rank as tightly as sharp dan
// ones, so a constant floor mis-scales by ~4× across the ladder and leaves flat-region
// users perpetually served above their rank, never converging. The caller measures <I>
// from the loaded corpus pool near the estimate and passes it via seFloorFn.
//
// <I> is the FULL 4PL information anchorInfo(floor)·slope² (irt.js), matching the
// g²/(p(1−p)) windowMLE accumulates below. It used to be the 2PL constant
// 0.8·0.2·<slope²>, which ignored the guessing floor and the lapse and so claimed
// STRICTLY more information than a real at-rank window can deliver — putting the
// floor below the se the estimator actually converges to. Two consequences, both
// structural rather than a matter of degree: the bonus never reached zero, so every
// converged user carried a permanent over-serve; and rankConverged (se < 2×floor)
// fired later than its "within 2× of the best achievable" intent. Keeping the two
// definitions in one formula is what makes the floor REACHABLE at all.
// The mean is UNWEIGHTED on purpose: it describes the post-convergence steady state,
// where selection is weakness-driven rather than information-driven.
export const WIN_STEADY_WEIGHT = (WIN_CAP + 1) / 2;   // Σ of the (M−age)/M ramp over a full window
export function seFloorForMeanInfo(meanInfo) {
    const info = WIN_PRIOR_INFO + WIN_STEADY_WEIGHT * meanInfo;
    return info > 0 ? 1 / Math.sqrt(info) : Infinity;
}

// Weighted 4PL Newton for the internal rank R over the positionally-weighted window
// (walking back from the newest: weight 1 at the newest, ~1/M at the oldest in-window,
// M = min(#attempts, WIN_CAP)). `attempts` sorted oldest -> newest.
export function windowMLE(attempts, refR) {
    const M = Math.min(attempts.length, WIN_CAP);
    const win = [];
    for (let j = attempts.length - 1; j >= 0 && win.length < WIN_CAP; j--) {
        const w = (M - win.length) / M;
        win.push({ a: attempts[j], w });
    }
    let R = refR, info = WIN_PRIOR_INFO;
    for (let it = 0; it < 15; it++) {
        let score = WIN_PRIOR_INFO * (refR - R);
        info = WIN_PRIOR_INFO;
        for (const { a, w } of win) {
            const span = clearSpan(a.floor);
            const praw = sigmoid(a.slope * (R - a.x) + a.kappa);
            let p = a.floor + span * praw;
            if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
            const g = span * a.slope * praw * (1 - praw);
            score += w * (a.correct - p) * g / (p * (1 - p));
            info += w * g * g / (p * (1 - p));
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 2) step = 2; else if (step < -2) step = -2;
        R += step;
        if (R < X_CLAMP_LO) R = X_CLAMP_LO; else if (R > X_CLAMP_HI) R = X_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }
    return { R, se: info > 0 ? 1 / Math.sqrt(info) : Infinity };
}

// Public window rank estimate from PRE-FLATTENED attempts (oldest -> newest).
// prevRankIndex warm-starts the fit; else the recent-mean fallback. Returns
//   { rankIndex, se, serveRankIndex }   or null when there are no attempts.
export function estimateWindowRankFromFlat(attempts, prevRankIndex, seFloorFn = null) {
    if (!attempts.length) return null;
    let ref = prevRankIndex;
    if (!Number.isFinite(ref)) {
        const tail = attempts.slice(-Math.min(attempts.length, 20));
        ref = tail.reduce((s, a) => s + a.x, 0) / tail.length;
    }
    const { R, se } = windowMLE(attempts, ref);
    // Rank-dependent floor from the caller (measured from the corpus pool near R);
    // fall back to the constant only when it can't be measured yet (nothing loaded
    // near the rank — e.g. the Worker, or a client before any shard has loaded).
    let seFloor = WIN_SE_FLOOR;
    if (seFloorFn) {
        const f = seFloorFn(R);
        if (Number.isFinite(f) && f > 0) seFloor = f;
    }
    const bonus = WIN_UCB_K * Math.max(0, se - seFloor);
    return { rankIndex: R, se, serveRankIndex: R + bonus };
}

// Positional-weighted median of a per-attempt series over the SAME last-WIN_CAP
// window and the SAME (M − age)/M weights as the rank fit. `values` is per-attempt
// (oldest -> newest, novel-only, aligned 1:1 with the rank window); non-finite or
// <= 0 entries are treated as missing (zero weight, and don't count toward the
// minimum — there is deliberately NO upper cap, since a median is robust to a few
// idle-tab outliers). Returns the weighted median, or null when fewer than
// `minCount` valid values. Client + Worker share it so the time-adjustment matches.
export function weightedMedianOverWindow(values, minCount) {
    const M = Math.min(values.length, WIN_CAP);
    const pairs = [];
    let total = 0;
    for (let j = values.length - 1, age = 0; j >= 0 && age < WIN_CAP; j--, age++) {
        const v = values[j];
        if (Number.isFinite(v) && v > 0) {
            const w = (M - age) / M;
            pairs.push({ v, w });
            total += w;
        }
    }
    if (pairs.length < minCount || total <= 0) return null;
    pairs.sort((a, b) => a.v - b.v);
    return weightedQuantile(pairs, total, 0.5);
}
