// js/rank-decomposition.js
//
// Two-kernel rank decomposition, fully continuous (no time bucketing):
//   S(t)    — slow "skill": a wide-kernel pooled logistic (τ_S ≈ 1 month).
//   σ_M(t)  — fast "form" swing: the genuine session-to-session performance
//             variance, separated from measurement noise.
//
// We work directly in INTERNAL-RANK space, pooling across board sizes. The
// internal rank (EGF-anchored, board-independent) is the one common skill axis
// shared by 9x9 and 13x13, so a single merged skill curve is well-defined only
// there. Each attempt is mapped through ITS OWN size's M before pooling:
//   - x = applyM(rawRating, sizeCurve)            (internal rank index)
//   - slope = LOGIT_PER_RATING / |M'(rawRating)|  (logit per internal rank)
// The per-attempt slope preserves calibration: the success curve has constant
// slope in rating space, so in rank space the slope must vary with the local
// steepness of M. With it, a
// 4d still reads as a 4d. Then:
//
//   1. Per attempt: x (internal rank) and its own slope, from its size's curve.
//   2. S(t) = wide-kernel (τ_S) pooled logistic over x at the 80% target — your
//      internal rank over time, evaluated exactly at any t. seS(t) is its
//      measurement SE (how precisely we know skill — shrinks with data).
//   3. A narrow kernel (τ_M, ~session scale) tracks form+skill; the residual
//      r(t) = S_narrow(t) − S_wide(t) ~ N(0, σ_M² + v) carries the session swing
//      PLUS the narrow estimator's own noise v. The time-varying performance
//      variance σ_M²(t) is a variance-components ML over a τ_V window (NOT
//      method-of-moments r²−v: v is wildly heteroscedastic — all-correct/all-
//      wrong sessions have v→∞ — and ML's 1/(σ_M²+v)² weight downweights those,
//      where r²−v would be poisoned by them). The swing is fit SEPARATELY by
//      direction: σ_up² from residuals above long-term skill (good days), σ_down²
//      from those below (bad days), so form can be lopsided. The split is clean
//      because the half of N(0,σ²+v) on one side of 0 carries the SAME
//      σ²-dependent log-likelihood as the full normal (the truncation mass is
//      exactly ½, independent of σ²), so the same ML estimator runs per side.
//      Each side's local σ²(t) is shrunk toward its own global ML by the
//      effective number of independent sessions in the window (Σ k/m, empirical
//      Bayes): sparse windows fall back to the whole-history swing so one
//      isolated session can't spike the band. The summary reports global σ_up /
//      σ_down, or — when the pooled swing is ~0 — a 95% profile-likelihood upper
//      bound ("under ±X, not detected").
//   4. S, seS, σ_up, σ_down are already in rank units (the working coordinate IS
//      internal rank), so the chart's inner band (S ± z·seS, measurement) and
//      outer band (up: S + z·√(seS²+σ_up²), down: S − z·√(seS²+σ_down²), where a
//      session lands) need no remap.
//
// M (rating_to_rank.json) is required: package_for_pwa ships only raw ratings,
// so the client can't rank anything without it. It is inlined into the bundle at
// build time (the import below), so it is as available as the code itself — no
// runtime fetch, no service-worker precache, hence no Cache-eviction/offline path
// that can brick boot. Attempts whose size curve is missing are dropped;
// computeRankDecomposition returns null if too few survive.

import { state } from './state.js';
import { TOTAL_RANKS } from './constants.js';
import { interpExtrap } from '../shared/rank-math.js';
import { estimateWindowRankFromFlat, windowMLE, WIN_CAP, applyM, flattenAttempt, slopeInRankSpace, seFloorForMeanSqSlope, LOGIT_PER_RANK_FALLBACK, weightedMedianOverWindow } from '../shared/rank-window.js';
import { sigmoid, gaussianWeight as gaussianWeightUntruncated, lowerBound } from './math-utils.js';
import { clearKappa, clampFloor, clearSpan } from './irt.js';
// Baked at build time from the pipeline's output (pwa/tsumegos/rating_to_rank.json).
import RATING_RANK_MAP from '../tsumegos/rating_to_rank.json';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

const TAU_S_MS = 30 * DAY_MS;               // skill (slow): ~1-month kernel
const TAU_M_MS = 4 * HOUR_MS;               // form (fast): ~session kernel
const TAU_V_MS = 10 * DAY_MS;               // σ_M(t) variance-smoothing window
const KERNEL_TRUNC_SIGMA = 4;

// rank = the level you clear at this rate (CALIBRATION_P; matches model.js /
// fit_rank_curves). The per-problem 3PL guessing floor c (problemFloor, shipped
// by train_rating from the bot ladder) gives P(clear) = c + (1−c)·σ(slope·
// (R−x)+κ(c)); κ(c) keeps P(clear)=80% at R==x for any c. Far below a problem
// (R≪x) P→c, so a success there is weak evidence of skill — which correctly
// discounts a strong user's lucky above-rank clears. The 3PL constants + κ/floor
// helpers live in irt.js.

// Per attempt the success-curve slope is the problem's OWN 2PL discrimination
// a_p (stored as `problemSlope`, shipped raw by package_for_pwa), converted to a
// logit-per-internal-rank slope by dividing by the local steepness of M. The
// EGF anchor (map_rating_to_rank → fit_user_rating) uses the same per-problem
// a_p, so the rank loop stays closed. LOGIT_PER_RATING is only the FALLBACK slope
// for legacy entries that predate `problemSlope`; set near the corpus-mean a_p
// (~1.6) so an un-backfilled attempt isn't mis-scaled against the re-anchored M.
// The per-attempt slope constants (LOGIT_PER_RATING, LOGIT_PER_RANK_FALLBACK,
// SLOPE_LO/HI, MSLOPE_FLOOR) + the applyM / slope-in-rank-space transform now live
// in pwa/shared/rank-window.js so the Worker's buffer flatten matches this one;
// applyM, flattenAttempt and LOGIT_PER_RANK_FALLBACK are imported above.

// Working coordinate is the internal rank index (~0..29, with extrapolation
// past either end for very weak/strong users).
const X_CLAMP_LO = -15;
const X_CLAMP_HI = 45;

const MIN_ATTEMPTS = 20;        // below this, no decomposition
const WIDE_GRID_POINTS = 120;   // coarse skill grid for residual interpolation
const SUPPORT_EFFN = 10;        // effective-N floor for "supported" skill range

// The wide (30-day) skill grid is a kernel-weighted MLE; a career with tens of
// attempts a day is enormously oversampled for a month-scale kernel, so the fit is
// run on frequency-weighted BINS instead of raw attempts. The binning is 2-D — a
// TIME slice (WIDE_TIME_BINS equal-count contiguous ranges — dense stretches get
// finer slices, all far shorter than the 30-day kernel) crossed with an x (rank)
// bucket of width WIDE_X_BIN. Binning by x too is load-bearing: attempts scatter
// ±2–3 ranks even between neighbours (problems are served across a band), so a
// pure time-bin's mean-x would lose real curvature and BIAS the logistic (Jensen);
// a narrow x-bucket keeps each cell near-constant in x, so its (mean x, success
// rate, count) is an unbiased binomial sufficient statistic. The count rides as
// poolLogistic's FREQUENCY weight `f` (not a magnitude weight), preserving BOTH the
// point estimate and its measurement SE. Validated: skill-line deviation ≤ ~0.05
// rank vs the unthinned fit on real dense histories. The narrow/session fits and
// the exact evaluateAt run on the FULL attempts (f = 1) — only the wide grid bins.
const WIDE_TIME_BINS = 200;    // equal-count time slices fed to the thinned wide fit
const WIDE_X_BIN = 0.33;       // internal-rank width of each x-bucket within a slice

// σ_M is session-to-session variance: it needs several INDEPENDENT sessions in
// the τ_V window to estimate reliably. With few, a local window's estimate is
// noise (or one isolated session blows the band into a spike). So the local
// estimate is shrunk toward the global one by effective session count (empirical
// Bayes): sparse windows fall back to the whole-history swing, dense windows keep
// their local value. SESSION_CORR_MS de-duplicates within-session near-copies.
const SESSION_CORR_MS = TAU_M_MS * Math.SQRT2;  // within-session correlation length
const SIGMA_PRIOR_SESSIONS = 8;  // shrinkage prior strength, in effective sessions
// A gap larger than this ends a session (starts a new one). Set near the within-
// session correlation length (SESSION_CORR_MS ≈ 5.66h) so the hard clustering
// matches the smooth Gaussian down-weighting the old per-attempt multiplicity
// applied — attempts closer than this were counted as ~the same sitting, farther
// as independent sessions — but rounded to 6h, an even divisor of the day (≤4
// sittings/day) rather than the √2 coincidence. Validated a wash vs 5.66h on real
// heavy histories (session counts and σ_M band within noise of each other).
const SESSION_GAP_MS = 6 * HOUR_MS;

// ---------------------------------------------------------------------------
// M: rating → rank map (pwa/tsumegos/rating_to_rank.json), per board size.
// ---------------------------------------------------------------------------
// M ships inside the bundle (see the import above), so there is nothing to fetch
// and nothing to fail: it is present the instant the module evaluates.
// loadRatingRankMap serves the call site in main.js, which keeps a null-guard
// against a malformed/empty packaged file (a build-time error, not a runtime one).
let _ratingRankMap = RATING_RANK_MAP;

export function loadRatingRankMap() {
    return _ratingRankMap;
}

// Raw 2PL slope a_p → problem-selection weight ∝ the Fisher information the
// problem provides about a user's ability: I(θ) = a_p²·P(1−P). Selection draws
// from a difficulty-matched pool (the 50 problems nearest the user's rank, all
// ≈80%-clear), so P(1−P) is ~constant across candidates and information ∝ a_p².
// Returned RAW (not on any [0,1] range): weightedRandomSelect normalizes, so only
// the relative a_p² matters — a sharp problem is quadratically preferred and a
// flat one strongly down-weighted, by genuine information rather than a min-max
// remap.
export function slopeToQuality(slope) {
    if (typeof slope !== 'number' || !Number.isFinite(slope) || slope <= 0) return 0;
    return slope * slope;
}

function curveForSize(size) {
    const m = _ratingRankMap && _ratingRankMap[String(size)];
    return (m && Array.isArray(m.ratings) && Array.isArray(m.rank_indices)
            && m.ratings.length >= 2) ? m : null;
}

// interpExtrap lives in shared/rank-math.js; applyM + the per-attempt flatten
// transform live in shared/rank-window.js (one copy shared with the Worker).
// applyM is imported above.

// Raw rating → normalized 0..1 rank for `size` (rank_index / TOTAL_RANKS, the
// scale getDescriptiveRank and the recall/weakness models use). Returns null
// when M isn't loaded; the whole client converts ratings to ranks through this,
// so package_for_pwa never has to.
export function normalizedRankForSize(rawRating, size) {
    const c = curveForSize(size);
    if (!c || typeof rawRating !== 'number') return null;
    return applyM(rawRating, c) / TOTAL_RANKS;
}

// Apply a RATING-space delta to a rank index: invert M at the rank index, shift
// by deltaRating (raw rating units), re-apply M. Exact rather than linearized
// via M', so it stays correct across knot boundaries. The curve is shared
// across both board sizes (map_rating_to_rank writes one curve to both keys),
// so size 9's key stands in for either. Used by the stats Rank tab to turn the
// population time-adjustment (alpha · Δlog10-minutes, fit in rating units by
// pipeline.fit_population_constants) into a rank-space shift.
export function rankIndexAfterRatingDelta(rankIndex, deltaRating) {
    const c = curveForSize(9);
    if (!c || !Number.isFinite(rankIndex) || !Number.isFinite(deltaRating)) return null;
    const rating = interpExtrap(rankIndex, c.rank_indices, c.ratings);
    return applyM(rating + deltaRating, c);
}

// Rating units spanned by ONE normalized-rank unit (the full 0..1 = 29-index
// scale), from the shipped curve's endpoints (~5.4). This is the unit constant
// between a rank gap in normalized 0..1 units and the same gap in raw-rating
// units; the recall model needs it to center its PROBLEM_DIFFICULTY prior (its
// gap feature is normalized-rank while a_p is per RATING unit). An endpoint
// mean — local knot slopes vary ~2-11 ranks/rating — the model's fitted
// multiplier absorbs the local mismatch.
export function ratingSpanPerNormalizedRank() {
    const c = curveForSize(9);   // one shared curve, written to both size keys
    if (!c) return null;
    const dRating = c.ratings[c.ratings.length - 1] - c.ratings[0];
    const dRankNorm = (c.rank_indices[c.rank_indices.length - 1] - c.rank_indices[0]) / TOTAL_RANKS;
    return dRankNorm > 0 ? dRating / dRankNorm : null;
}

// Stable per-size signature of the loaded curve. Stored ranks are invalidated
// when this changes (a fit_rank_curves re-fit that didn't also change ratings).
export function ratingRankCurveVersion(size) {
    const c = curveForSize(size);
    return c ? JSON.stringify([c.ratings, c.rank_indices]) : null;
}

// ---------------------------------------------------------------------------

// Truncated variant of math-utils' gaussianWeight: exactly 0 past
// KERNEL_TRUNC_SIGMA so the kernel windows stay finite.
function gaussianWeight(dtOverTau) {
    if (Math.abs(dtOverTau) > KERNEL_TRUNC_SIGMA) return 0;
    return gaussianWeightUntruncated(dtOverTau);
}

// Weak prior on the local time-slope used by the LOCAL-LINEAR wide kernel (in
// ability units per kernel width τ). It ridge-shrinks the fitted drift toward 0
// so a thin or one-sided (boundary) window degrades gracefully back to the
// local-constant estimate instead of extrapolating a noisy slope — the standard
// penalized-local-linear boundary fix. As SLOPE_PRIOR_SD → 0 the linear fit
// collapses to local-constant (point AND sandwich SE); larger = more boundary
// de-biasing at the cost of more boundary variance.
//
// Validated on real histories (auxiliary/eval_rank_endpoint_bias.py — interior
// held-out lag test, 10 users): today's local-constant endpoint LAGS the lag-free
// two-sided reference by ~1.2 stones on TRENDING stretches; this is a genuine
// bias/variance dial (no sharp knee). SD=2.0 removes ~40% of that lag (−1.16 →
// −0.67) for a ~20% band widening and keeps boundary over-extrapolation on thin
// histories in check; 1.5 is more conservative, 2.5 more aggressive. (RMS-vs-
// reference is flat across SD because the reference is itself noise-dominated, so
// it cancels the bias signal — the signed trending lag is the metric that selects.)
//
// IMPORTANT — this is a SMOOTHING BANDWIDTH, not a drift prior. The measured
// population SD of real 30-day skill drift is only ~0.6 stones/30d (robust ~0.23);
// 2.0 is deliberately ~3× looser because the ridge fights the boundary slope's
// MEASUREMENT noise, not real trend rates. Do NOT "derive it from the population"
// and set it to ~0.6 — that over-shrinks the slope and silently RE-INTRODUCES the
// lag (≈10% removed vs ≈40% here). If anything, ~0.6 is a FLOOR (never regularize
// tighter than people actually trend), never the value. See the design note in
// docs / the project memory before changing this.
const SLOPE_PRIOR_SD = 2.0;                  // ranks per τ_S (bandwidth, not a drift prior)
const SLOPE_PRIOR_INFO = 1 / (SLOPE_PRIOR_SD * SLOPE_PRIOR_SD);

// Kernel-weighted pooled 3PL MLE for the internal-rank skill R at time t0:
// P(correct)=c_j+(1−c_j)·σ(slope_j·(R−x_j)+κ_j), where each attempt j carries its own
// slope (logit per internal rank), guessing floor c_j (problemFloor) and κ_j=κ(c_j)
// (so P=80% at R==x_j) — all set in flattenAttempts. Returns { R, se, effN, drift };
// attempts MUST be sorted by t.
//
// `linear` selects the time model:
//   false (default) — LOCAL-CONSTANT in time: ability is one scalar R over the
//     whole kernel window. Correct for the narrow form kernel (a session can't
//     support a trend), but at the RIGHT edge (now) a symmetric kernel sees only
//     past data, so the estimate is a backward-weighted average that trails a
//     trending user by ~0.8·τ — the boundary lag.
//   true — LOCAL-LINEAR in time: fit R(t)=a+b·u, u=(t−t0)/τ, and report a (the
//     level AT t0); the drift b is ridge-shrunk toward 0 (SLOPE_PRIOR_INFO). The
//     intercept of a local-linear fit has O(h²) boundary bias (vs O(h) for
//     local-constant), so the lag cancels. `drift` (b) is returned for free.
function poolLogistic(attempts, t0, tau, linear = false) {
    const win = KERNEL_TRUNC_SIGMA * tau;
    const lo = lowerBound(attempts, t0 - win, a => a.t);
    const hi = lowerBound(attempts, t0 + win + 1, a => a.t);
    if (hi <= lo) return { R: NaN, se: Infinity, effN: 0, drift: 0 };

    // `f` is an optional per-attempt FREQUENCY weight (this row stands for f real
    // attempts — used by the thinned wide grid). It enters linearly everywhere
    // (Σf·w, Σf·w², score, Fisher info, sandwich meat Σf·w²·v), so a thinned+
    // reweighted fit reproduces the unthinned point estimate AND its SE. Absent ⇒ 1.
    let sumW = 0, sumW2 = 0, sumWx = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        const fw = (attempts[j].f || 1) * w;
        sumW += fw; sumW2 += fw * w; sumWx += fw * attempts[j].x;
    }
    if (sumW <= 0) return { R: NaN, se: Infinity, effN: 0, drift: 0 };
    const effN = (sumW * sumW) / sumW2;

    if (linear) return poolLinear(attempts, t0, tau, lo, hi, sumWx / sumW, effN);

    let R = sumWx / sumW;

    // 4PL MLE for R: P = c + (1−c−LAPSE)·σ(slope·(R−x)+κ). dP/dR = span·slope·praw·(1−praw),
    // span = clearSpan(c) = 1−c−LAPSE. Fisher info per attempt v = (dP/dR)²/(P(1−P));
    // Newton step = Σ w·(y−P)·(dP/dR)/(P(1−P)) / Σ w·v. The LAPSE cap is what bounds a
    // misclick on a far-below-rank problem: there praw→1 so g→0 while P(1−P)→LAPSE·(1−LAPSE)
    // stays > 0, so the slip's score → 0 instead of yanking R down.
    for (let it = 0; it < 15; it++) {
        let score = 0, info = 0;
        for (let j = lo; j < hi; j++) {
            const w = gaussianWeight((attempts[j].t - t0) / tau);
            if (w <= 0) continue;
            const fw = (attempts[j].f || 1) * w;
            const slope = attempts[j].slope, c = attempts[j].floor, span = clearSpan(c);
            const praw = sigmoid(slope * (R - attempts[j].x) + attempts[j].kappa);
            let p = c + span * praw;
            if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
            const g = span * slope * praw * (1 - praw);   // dP/dR
            score += fw * (attempts[j].correct - p) * g / (p * (1 - p));
            info += fw * g * g / (p * (1 - p));
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 2) step = 2; else if (step < -2) step = -2;
        R += step;
        if (R < X_CLAMP_LO) R = X_CLAMP_LO; else if (R > X_CLAMP_HI) R = X_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }

    // Sandwich SE: Var ≈ Σ f·w²·v / (Σ f·w·v)², v = (dP/dR)²/(P(1−P)) (4PL Fisher info).
    let A = 0, B = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        const fw = (attempts[j].f || 1) * w;
        const slope = attempts[j].slope, c = attempts[j].floor, span = clearSpan(c);
        const praw = sigmoid(slope * (R - attempts[j].x) + attempts[j].kappa);
        let p = c + span * praw;
        if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
        const g = span * slope * praw * (1 - praw);
        const v = g * g / (p * (1 - p));
        A += fw * v; B += fw * w * v;
    }
    const se = A > 0 ? Math.sqrt(B) / A : Infinity;
    return { R, se, effN, drift: 0 };
}

// Local-LINEAR-in-time wide-kernel fit (see poolLogistic's `linear` doc). Fits
// R(t)=a+b·u over the same kernel window, u=(t−t0)/τ, reports the intercept a and
// drift b. The drift carries a weak ridge (SLOPE_PRIOR_INFO) so a thin/one-sided
// window can't extrapolate noise; as that prior tightens the whole thing collapses
// to the local-constant estimate (point AND SE), so SLOPE_PRIOR_SD is a clean
// "how much boundary de-biasing" knob. SE(a) is the 2×2 robust (sandwich) SE
// Cov=B⁻¹·M·B⁻¹, the exact generalization of the scalar sandwich the local-
// constant path uses — kernel weights aren't true probabilities, so the naive
// model-info SE would mis-scale; in the tight-prior limit this reproduces √B/A.
function poolLinear(attempts, t0, tau, lo, hi, x0, effN) {
    let a = x0, b = 0;
    for (let it = 0; it < 25; it++) {
        let Sa = 0, Sb = -SLOPE_PRIOR_INFO * b;
        let Iaa = 0, Iab = 0, Ibb = SLOPE_PRIOR_INFO;
        for (let j = lo; j < hi; j++) {
            const u = (attempts[j].t - t0) / tau;
            const w = gaussianWeight(u);
            if (w <= 0) continue;
            const fw = (attempts[j].f || 1) * w;
            const slope = attempts[j].slope, c = attempts[j].floor, span = clearSpan(c);
            const praw = sigmoid(slope * (a + b * u - attempts[j].x) + attempts[j].kappa);
            let p = c + span * praw;
            if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
            const g = span * slope * praw * (1 - praw);     // dP/dR
            const wv = fw * g * g / (p * (1 - p));          // weighted Fisher info
            const sc = fw * (attempts[j].correct - p) * g / (p * (1 - p));
            Sa += sc;  Sb += sc * u;
            Iaa += wv; Iab += wv * u; Ibb += wv * u * u;
        }
        const det = Iaa * Ibb - Iab * Iab;
        if (Math.abs(det) < 1e-12) break;
        let da = (Ibb * Sa - Iab * Sb) / det;
        let db = (Iaa * Sb - Iab * Sa) / det;
        if (da > 2) da = 2; else if (da < -2) da = -2;
        if (db > 2) db = 2; else if (db < -2) db = -2;
        a += da; b += db;
        if (a < X_CLAMP_LO) a = X_CLAMP_LO; else if (a > X_CLAMP_HI) a = X_CLAMP_HI;
        if (Math.abs(da) < 1e-6 && Math.abs(db) < 1e-6) break;
    }
    // 2×2 sandwich SE of the intercept: bread B = Σ f·w·v·JJᵀ + ridge, meat
    // M = Σ f·w²·v·JJᵀ (J=[1,u]); Cov₀₀ = (B_bb²·M_aa − 2 B_ab B_bb M_ab + B_ab² M_bb)/detB².
    let Baa = 0, Bab = 0, Bbb = SLOPE_PRIOR_INFO, Maa = 0, Mab = 0, Mbb = 0;
    for (let j = lo; j < hi; j++) {
        const u = (attempts[j].t - t0) / tau;
        const w = gaussianWeight(u);
        if (w <= 0) continue;
        const fw = (attempts[j].f || 1) * w;
        const slope = attempts[j].slope, c = attempts[j].floor, span = clearSpan(c);
        const praw = sigmoid(slope * (a + b * u - attempts[j].x) + attempts[j].kappa);
        let p = c + span * praw;
        if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
        const g = span * slope * praw * (1 - praw);
        const v = g * g / (p * (1 - p));
        const wv = fw * v, w2v = fw * w * v;
        Baa += wv;  Bab += wv * u;  Bbb += wv * u * u;
        Maa += w2v; Mab += w2v * u; Mbb += w2v * u * u;
    }
    const detB = Baa * Bbb - Bab * Bab;
    let se = Infinity;
    if (detB > 0) {
        const cov00 = (Bbb * Bbb * Maa - 2 * Bab * Bbb * Mab + Bab * Bab * Mbb) / (detB * detB);
        se = cov00 > 0 ? Math.sqrt(cov00) : Infinity;
    }
    return { R: a, se, effN, drift: b };
}

// Flatten answerHistory to sorted [{t, x, slope, correct}] in INTERNAL-RANK
// space. Each attempt is mapped through ITS OWN size's curve (merged entries
// carry `size`; otherwise the active size is assumed): x is the internal rank
// index applyM(rawRating), slope is the logit-per-internal-rank at that rating
// = a_p / |M'(rating)|, where a_p is the problem's own 2PL discrimination
// (`problemSlope`). Legacy entries without `problemSlope` fall back to the
// LOGIT_PER_RATING constant. `problemRating` holds the raw rating (legacy entries
// may carry a separate `problemRawRating`). Attempts whose size curve is missing
// fall back to the stored problemRank with a representative slope, or are dropped.
export function flattenAttempts(answerHistory) {
    return answerHistory
        .filter(a => typeof a.date === 'number' && !a.isRepeat && !a.linked)
        .map(a => {
            const size = a.size ?? state.boardSize;
            const c = curveForSize(size);
            const raw = (typeof a.problemRawRating === 'number')
                ? a.problemRawRating
                : a.problemRating;
            // Main path: raw rating mapped through the size curve via the SHARED
            // transform (identical to the Worker's buffer flatten).
            const f = flattenAttempt(raw, a.problemSlope, a.problemFloor, a.correct, c);
            if (f) return { ...f, t: a.date, src: a };
            // Legacy fallback: no usable curve, but a stored normalized problemRank.
            if (typeof a.problemRank === 'number') {
                const flr = clampFloor(a.problemFloor);
                return { t: a.date, x: a.problemRank * TOTAL_RANKS,
                         slope: LOGIT_PER_RANK_FALLBACK, correct: a.correct ? 1 : 0,
                         floor: flr, kappa: clearKappa(flr), src: a };
            }
            return null;
        })
        .filter(Boolean)
        .sort((a, b) => a.t - b.t);
}

// ---------------------------------------------------------------------------
// Positional-linear window estimator — the live front-page rank.
// ---------------------------------------------------------------------------
// The displayed/served internal rank is the positionally-weighted 4PL MLE over the
// most recent FIRST-TIME attempts. Walking back from the newest, each attempt's
// weight ramps linearly 1 → 1/M with its POSITION across the AVAILABLE window (newest
// = 1, the oldest in-window ≈ 0), where M = min(#attempts, WIN_CAP) — a plain
// (M − age)/M ramp that always spans whatever recent history exists. The point is the
// edge: the attempt aging out at the cap leaves at ~0 weight BY CONSTRUCTION, so a
// per-attempt rank delta is never contaminated by an old problem dropping off.
//
// The positional weight ignores how informative each attempt is, so the effective
// sample size is ~3·M/4 (Σw = (M+1)/2, Σw² ≈ M/3). That gives up nothing real: served
// at the user's rank the per-attempt info is so low the window always bounds on
// WIN_CAP, not on accumulated information. WIN_CAP bounds the walk by SAMPLE COUNT
// only (no time cutoff: a long-idle return still uses its real recent history). The
// lapse term in the 4PL (clearSpan/clearKappa) already discounts below-level
// misclicks, so early random slips fade on their own.
// The window constants, the weighted 4PL Newton (windowMLE) and the positional
// weighted-median now live in pwa/shared/rank-window.js so the Cloudflare Worker
// (EGD leaderboard) runs BIT-IDENTICAL rank code. estimateWindowRank here just
// flattens the client's answerHistory (size/curve-aware, first-time only, pooled
// across sizes through each attempt's own M — pass getMergedAnswerHistory()) and
// delegates. prevRankIndex warm-starts the fit (the live displayed rank · TOTAL_RANKS);
// on a cold device the shared helper falls back to the recent attempts' mean. Returns:
//   rankIndex      — the estimate to DISPLAY / score (internal-rank index)
//   se             — its posterior standard error (rank units)
//   serveRankIndex — rankIndex + the uncertainty-scaled UCB serve bonus (large while
//                    the posterior is loose, vanishing as it tightens); problem
//                    SELECTION uses this, display uses rankIndex. The bonus subtracts a
//                    RANK-DEPENDENT floor (localSeFloor below): the irreducible SE a
//                    full at-rank window would leave, set by the local corpus slope, so
//                    serving collapses to the user's OWN rank at convergence on every
//                    part of the ladder — not just where a constant floor happened to fit.
// null when no usable (first-time, curve-mapped) attempt exists.
// Rank-dependent UCB serve-bonus floor (see seFloorForMeanSqSlope in rank-window.js):
// the irreducible SE the window MLE converges to once the user has a full window of
// AT-their-rank problems. We estimate the local mean squared success-curve slope (rank
// space) from the ≤50 corpus problems nearest the estimate that are ALREADY LOADED on
// the active size — no fetch on the hot path; the selector's normal shard loading keeps
// this pool warm and it self-refines as the user's rank drifts into new bands. slope²
// is evaluated only on the 50 nearest (the expensive per-problem transform), not the
// whole pool. Returns null when nothing is loaded yet (or the curve is missing), so the
// estimator keeps the constant WIN_SE_FLOOR — a harmless fallback since se dominates the
// bonus that early anyway.
function localSeFloor(rankIndex) {
    const pool = state.allProblems;
    if (!pool || pool.length === 0) return null;
    const curve = curveForSize(state.boardSize);
    if (!curve) return null;
    const target = rankIndex / TOTAL_RANKS;               // pool ranks are normalized 0..1
    const nearest = pool
        .slice()
        .sort((a, b) => Math.abs(a.rank - target) - Math.abs(b.rank - target))
        .slice(0, 50);
    let sumSq = 0, n = 0;
    for (const p of nearest) {
        if (typeof p.rating !== 'number') continue;
        const s = slopeInRankSpace(p.rating, p.slope, curve);
        sumSq += s * s;
        n++;
    }
    return n > 0 ? seFloorForMeanSqSlope(sumSq / n) : null;
}

export function estimateWindowRank(answerHistory, prevRankIndex) {
    return estimateWindowRankFromFlat(flattenAttempts(answerHistory), prevRankIndex, localSeFloor);
}

// The front-page (window) rank as it evolved across history: the SAME positional
// window 4PL MLE the live headline uses (windowMLE over the trailing WIN_CAP
// first-time attempts), re-evaluated at each of `times`. A pure function of
// decomp.attempts — already flattened to internal-rank space and time-sorted by
// flattenAttempts — so its RIGHT endpoint reproduces the live front-page rank
// (state.userRank·TOTAL_RANKS) exactly, and the Rank chart can draw the headline's
// own trajectory alongside the slow skill line. `times` MUST be ascending; the
// return is aligned 1:1 with `times`, each element the internal-rank index or null
// (no first-time attempt exists yet at that time). Only the trailing WIN_CAP
// attempts affect windowMLE, so we pass just that suffix — bit-identical to the
// whole prefix (both have M = min(prefixLen, WIN_CAP)) but O(WIN_CAP) per point.
// Each point warm-starts from the previous (the fit converges regardless; this
// only saves Newton steps).
export function windowRankSeries(decomp, times) {
    const out = new Array(times.length).fill(null);
    const attempts = decomp && decomp.attempts;
    if (!attempts || !attempts.length) return out;
    let idx = 0;            // count of attempts with t <= the current grid time
    let ref = NaN;          // warm-start from the previous evaluated point
    for (let i = 0; i < times.length; i++) {
        while (idx < attempts.length && attempts[idx].t <= times[i]) idx++;
        if (idx === 0) continue;   // no first-time attempt yet
        const prefix = attempts.slice(Math.max(0, idx - WIN_CAP), idx);
        let r = ref;
        if (!Number.isFinite(r)) {
            const tail = prefix.slice(-Math.min(prefix.length, 20));
            r = tail.reduce((s, a) => s + a.x, 0) / tail.length;
        }
        const { R } = windowMLE(prefix, r);
        if (Number.isFinite(R)) { out[i] = R; ref = R; }
    }
    return out;
}

// The median solve time (MINUTES) as it evolved across history — the exact time
// counterpart of windowRankSeries. At each of `times` it takes the SAME trailing
// WIN_CAP window of decomp.attempts and runs weightedMedianOverWindow with the SAME
// (M − age)/M positional weights the rank fit uses, so the Rank-tab time-dock is over
// the identical weighted sample as the rank line it docks — never a mismatch. Solve
// times come from each flattened attempt's `.src` (the original answer). `times` MUST
// be ascending; the return aligns 1:1 with `times` (null where < minTimed valid times
// in the window). `minTimed` is passed in (population-constants' TIME_ADJUST_MIN_TIMED)
// to keep this module free of a population-constants import.
export function windowMedianSolveMinutesSeries(decomp, times, minTimed) {
    const out = new Array(times.length).fill(null);
    const attempts = decomp && decomp.attempts;
    if (!attempts || !attempts.length) return out;
    let idx = 0;
    for (let i = 0; i < times.length; i++) {
        while (idx < attempts.length && attempts[idx].t <= times[i]) idx++;
        if (idx === 0) continue;
        const prefix = attempts.slice(Math.max(0, idx - WIN_CAP), idx);
        const medMs = weightedMedianOverWindow(prefix.map(a => a.src?.timeTaken), minTimed);
        if (medMs !== null) out[i] = medMs / 60000;
    }
    return out;
}

// Coarse, regularly-spaced evaluation of the wide (skill) kernel. Used only to
// interpolate the wide track when forming form residuals; the rendered skill
// line evaluates the kernel exactly (see evaluateAt). The 30-day kernel is
// locally linear over the grid spacing, so interpolation here is exact to well
// under a rank.
function buildWideGrid(attempts, points) {
    const t0 = attempts[0].t;
    const t1 = attempts[attempts.length - 1].t;
    const span = Math.max(1, t1 - t0);
    const grid = [];
    for (let i = 0; i < points; i++) {
        const t = t0 + span * (i / (points - 1));
        const w = poolLogistic(attempts, t, TAU_S_MS, true);   // local-linear wide skill
        grid.push({ t, R: w.R, se: w.se, effN: w.effN });
    }
    return grid;
}

// Thin a time-sorted attempt array to frequency-weighted 2-D (time × x) BINS for
// the wide-grid fit (see WIDE_TIME_BINS / WIDE_X_BIN). Each cell aggregates its
// members to their MEAN (t, x, slope, floor) with a FRACTIONAL success rate
// `correct` and `f` = the cell count — the exact binomial sufficient statistic for
// a logistic bin, so poolLogistic's f·(correct − p) = successes − f·p and
// f·g²/(P(1−P)) reproduce the cell's full score and Fisher information. The x-bin
// keeps each cell near-constant in x (no Jensen bias from averaging scattered x);
// the ×f frequency weight keeps the SE faithful. Returns the array unchanged
// (f defaults to 1) when there aren't enough attempts for binning to help.
function thinForWide(attempts) {
    const N = attempts.length;
    if (N <= WIDE_TIME_BINS * 4) return attempts;
    const out = [];
    for (let k = 0; k < WIDE_TIME_BINS; k++) {
        const lo = Math.floor(k * N / WIDE_TIME_BINS);
        const hi = Math.floor((k + 1) * N / WIDE_TIME_BINS);
        if (hi <= lo) continue;
        const cells = new Map();   // x-bucket index → running sums
        for (let j = lo; j < hi; j++) {
            const a = attempts[j];
            const xb = Math.round(a.x / WIDE_X_BIN);
            let c = cells.get(xb);
            if (!c) { c = { st: 0, sx: 0, ss: 0, sfl: 0, sc: 0, n: 0 }; cells.set(xb, c); }
            c.st += a.t; c.sx += a.x; c.ss += a.slope; c.sfl += a.floor; c.sc += a.correct; c.n++;
        }
        for (const c of cells.values()) {
            const floor = c.sfl / c.n;
            out.push({
                t: c.st / c.n, x: c.sx / c.n, slope: c.ss / c.n, floor,
                correct: c.sc / c.n,          // fractional success rate over the cell
                kappa: clearKappa(floor),     // recomputed from the mean floor
                f: c.n,
            });
        }
    }
    out.sort((p, q) => p.t - q.t);   // cells from adjacent slices interleave slightly
    return out;
}

// Linear interpolation of a precomputed ascending grid on field `key`.
function interpGrid(grid, t, key) {
    const n = grid.length;
    if (t <= grid[0].t) return grid[0][key];
    if (t >= grid[n - 1].t) return grid[n - 1][key];
    let a = 0, b = n - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (grid[m].t <= t) a = m; else b = m; }
    const span = grid[b].t - grid[a].t;
    const u = span > 0 ? (t - grid[a].t) / span : 0;
    return grid[a][key] + (grid[b][key] - grid[a][key]) * u;
}

// Form residuals — ONE per SESSION (a contiguous run of attempts with < SESSION_
// GAP_MS between neighbours), not one per attempt. r = narrow-kernel skill −
// wide-kernel skill (internal-rank space) evaluated at the session's centre; v is
// the narrow estimator's measurement variance there. The sign of r is kept (`up`)
// so the swing can be fit separately above (r ≥ 0, good days) and below (r < 0,
// bad days) long-term skill.
//
// Why per session: σ_M is a session-TO-session variance, so a session is one
// draw. The old code evaluated the narrow kernel at every attempt (~O(N) fits,
// each near-identical to its session-mates) and then divided each residual by an
// O(N²)-computed within-session multiplicity m to recover one INDEPENDENT session
// — two halves of a single "collapse a sitting to one observation" mechanism.
// Evaluating once per session does that directly: m ≡ 1, no multiplicity scan, a
// handful-hundred fits instead of tens of thousands. A large session still counts
// more PRECISELY (its narrow fit pools more attempts ⇒ smaller v ⇒ larger ML
// weight 1/(σ_M²+v)²), which is the right way for size to matter — not as extra
// session mass. Attempts arrive time-sorted, so sessions are a single linear pass.
function buildResidualTrack(attempts, wideGrid) {
    const resid = [];
    let s = 0;
    while (s < attempts.length) {
        let e = s + 1;
        while (e < attempts.length && attempts[e].t - attempts[e - 1].t < SESSION_GAP_MS) e++;
        // Centre = mean attempt time; the 4h narrow kernel centred there pools the
        // whole sitting (evaluated over the FULL attempts, so the fit is exact).
        let sumT = 0;
        for (let j = s; j < e; j++) sumT += attempts[j].t;
        const tc = sumT / (e - s);
        s = e;
        const nj = poolLogistic(attempts, tc, TAU_M_MS);
        if (!Number.isFinite(nj.R)) continue;
        const Rw = interpGrid(wideGrid, tc, 'R');
        if (!Number.isFinite(Rw)) continue;
        const r = nj.R - Rw;
        resid.push({ t: tc, r2: r * r, up: r >= 0, v: nj.se * nj.se, w: nj.effN, m: 1 });
    }
    return resid;
}

// Variance-components ML for the performance variance σ_M² (rating²): the form
// residuals are modeled r ~ N(0, σ_M² + v), and we maximize the (weighted)
// likelihood over σ_M² ≥ 0 by Newton. This is the right estimator because v is
// wildly heteroscedastic — an all-correct/all-wrong session can't pin your level
// (v → huge), and the ML weight 1/(σ_M²+v)² downweights exactly those, instead
// of the method-of-moments r²−v which they poison. `rows` carries {r2, v, wgt}.
// Reused unchanged for the one-sided (good-day / bad-day) fits: the half of
// N(0,σ²+v) on one side of 0 has the identical σ²-dependent log-likelihood (its
// truncation mass is exactly ½, independent of σ²), so feeding it only the rows
// of one sign yields that side's ML σ².
function mlSigma2(rows) {
    let s2 = 0;
    for (let it = 0; it < 60; it++) {
        let score = 0, info = 0;
        for (const d of rows) {
            const vv = s2 + d.v;
            score += d.wgt * (-0.5 / vv + 0.5 * d.r2 / (vv * vv));
            info  += d.wgt * (0.5 / (vv * vv));
        }
        const step = score / Math.max(info, 1e-15);
        const next = Math.max(0, s2 + step);
        if (Math.abs(next - s2) < 1e-12) { s2 = next; break; }
        s2 = next;
    }
    return s2;
}

// Time-varying performance variance at t (rating²), split by direction:
// kernel-localized variance-components ML per side, each shrunk toward its own
// global σ². Each residual is weighted by k (τ_V time kernel) ÷ m (de-duplicate
// within-session near-copies) so the window counts INDEPENDENT sessions, and v
// downweights uninformative ones. Empirical-Bayes shrinkage by effective
// sessions (nSess = Σ k/m) pulls sparse windows toward the side's global σ² — so
// a lone isolated session can't spike the band, and the band never claims more
// local swing than the data supports. Splitting halves the sessions per side, so
// the shrinkage matters more here (good) — sparse one-sided windows lean on the
// whole-history good-/bad-day swing rather than on a couple of residuals.
function localPerformanceVar(resid, t, g2Up, g2Down) {
    const rowsUp = [], rowsDown = [];
    let nUp = 0, nDown = 0;
    for (const d of resid) {
        if (!Number.isFinite(d.v)) continue;     // uninformative session: no signal
        const k = gaussianWeight((d.t - t) / TAU_V_MS);
        if (k <= 0) continue;
        const wgt = k / d.m;
        if (d.up) { rowsUp.push({ r2: d.r2, v: d.v, wgt }); nUp += wgt; }
        else      { rowsDown.push({ r2: d.r2, v: d.v, wgt }); nDown += wgt; }
    }
    const fit = (rows, n, glob) => {
        if (!rows.length) return glob;
        const w = n / (n + SIGMA_PRIOR_SESSIONS);
        return w * mlSigma2(rows) + (1 - w) * glob;
    };
    return {
        varUp: fit(rowsUp, nUp, g2Up),
        varDown: fit(rowsDown, nDown, g2Down),
        nSess: nUp + nDown,
    };
}

// Global (whole-history) σ_M² point estimate + its 95% profile-likelihood upper
// bound (rating²), plus the per-direction globals σ_up² / σ_down² (shrinkage
// targets for the local band, fit with the same half-normal ML — see mlSigma2).
// Weighted by 1/m so each session counts once. The pooled (both-sign) sigma2 +
// upper2 drive the summary's detection test: when the pooled point estimate is
// ~0 we report the upper bound ("swing is under ±X, not detected") instead of a
// misleading ±0.00.
function globalPerformanceSigma(resid) {
    const rows = [], up = [], down = [];
    for (const d of resid) {
        if (!Number.isFinite(d.v)) continue;
        const row = { r2: d.r2, v: d.v, wgt: 1 / d.m };
        rows.push(row);
        (d.up ? up : down).push(row);
    }
    if (rows.length < 2) return { sigma2: 0, upper2: 0, sigma2Up: 0, sigma2Down: 0 };
    const s2 = mlSigma2(rows);
    const LL = x => rows.reduce((a, d) => {
        const vv = x + d.v;
        return a + d.wgt * (-0.5 * Math.log(vv) - 0.5 * d.r2 / vv);
    }, 0);
    // 95% drop for a parameter tested at its boundary (σ² ≥ 0): the LR statistic
    // is a ½χ²₀ + ½χ²₁ mixture there, so the one-sided 95% point is
    // χ²₁(0.90)/2 = 1.353 — the two-sided χ²₁(0.95)/2 = 1.92 over-widened the bound.
    const target = LL(s2) - 1.353;
    let lo = s2, hi = s2 + 1, guard = 0;
    while (LL(hi) > target && guard++ < 80) hi *= 2;
    for (let it = 0; it < 80; it++) {
        const mid = 0.5 * (lo + hi);
        if (LL(mid) > target) lo = mid; else hi = mid;
    }
    return {
        sigma2: s2,
        upper2: 0.5 * (lo + hi),
        // A side with no residuals falls back to the pooled estimate.
        sigma2Up: up.length ? mlSigma2(up) : s2,
        sigma2Down: down.length ? mlSigma2(down) : s2,
    };
}

// Core decomposition from an already-flattened, time-sorted attempt array — the
// heavy lifting (wide skill grid, form residuals, global swing, summary).
// Returns null below MIN_ATTEMPTS.
function decompose(attempts) {
    if (attempts.length < MIN_ATTEMPTS) return null;

    // Coarse wide track (for residual interpolation AND the chart's dense skill
    // line) + per-attempt form residuals. The decomp carries enough to evaluate
    // S, seS and σ_M(t) continuously at any t: the wide kernel exactly via
    // evaluateAt, or interpolated from wideGrid via evaluateOnGrid (the dense
    // render path — the 30-day kernel is locally linear over the grid spacing, so
    // interpolating S/seS matches the exact fit to well under a rank while sparing
    // the per-point re-fit). wideGrid is kept on the decomp for that path.
    const wideGrid = buildWideGrid(thinForWide(attempts), WIDE_GRID_POINTS);
    const resid = buildResidualTrack(attempts, wideGrid);

    // Global σ_M (point + 95% upper bound) — also the shrinkage target the
    // time-varying band pulls toward, so it must be on the decomp before any
    // evaluateAt call. Already in internal-rank units (the working coordinate).
    const gp = globalPerformanceSigma(resid);
    const decomp = { attempts, resid, wideGrid, sigma2Up: gp.sigma2Up, sigma2Down: gp.sigma2Down };

    // Current skill reads the last "supported" grid point — the latest with
    // ≥ SUPPORT_EFFN effective pooled attempts (and a finite estimate).
    let lastT = null;
    for (const g of wideGrid) {
        if (g.effN >= SUPPORT_EFFN && Number.isFinite(g.R)) lastT = g.t;
    }
    if (lastT === null) lastT = wideGrid[wideGrid.length - 1].t;

    const cur = evaluateAt(decomp, lastT);

    decomp.summary = {
        currentSkill: cur && Number.isFinite(cur.S) ? cur.S : null,
        currentSkillSE: cur ? cur.seS : Infinity,
        sigmaM: Math.sqrt(gp.sigma2),         // pooled swing — drives detection test
        sigmaMUpper: Math.sqrt(gp.upper2),
        sigmaUp: Math.sqrt(gp.sigma2Up),      // good-day overperformance (ranks)
        sigmaDown: Math.sqrt(gp.sigma2Down),  // bad-day underperformance (ranks)
    };
    return decomp;
}

// The official skill decomposition — skill line, 95% measurement CI and the
// daily-variance band — over the user's FULL first-time history.
//
// Far-from-rank attempts carry systematic bias (a strong user's lucky above-rank
// clears over-credit skill; the guessing floor inflates far-below clears), but the
// 4PL MLE in poolLogistic absorbs that WITHIN the estimate, so no pre-filter is
// needed: the per-problem guessing floor c discounts far-below clears (P→c there,
// so a clear is weak evidence of skill) and the LAPSE upper asymptote caps the
// score of a far-above slip (P(1−P)→LAPSE·(1−LAPSE) while the gradient →0).
export function computeRankDecomposition(answerHistory) {
    // flattenAttempts maps each attempt through its own size's M, so a merged
    // 9x9+13x13 history pools on the internal-rank axis. Attempts whose size curve
    // is missing are dropped there.
    const all = flattenAttempts(answerHistory);
    if (all.length < MIN_ATTEMPTS) return null;
    return decompose(all);
}

// Continuous evaluation at arbitrary t (no buckets). The wide kernel is
// evaluated exactly here, so the rendered skill line and inner band are exact;
// σ_M(t) comes from the local performance-variance estimate. The working
// coordinate IS the internal rank, so S/seS/σ_M need no remap. Returns null if
// t has no support.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { attempts, resid } = decomp;
    const sk = poolLogistic(attempts, t, TAU_S_MS, true);    // exact local-linear wide skill
    if (!Number.isFinite(sk.R)) return null;
    const pv = localPerformanceVar(resid, t, decomp.sigma2Up || 0, decomp.sigma2Down || 0);
    return {
        S: sk.R,                      // skill, internal-rank index
        seS: sk.se,                   // measurement SE, rank units
        sigmaUp: Math.sqrt(pv.varUp),     // good-day swing above S, rank units
        sigmaDown: Math.sqrt(pv.varDown), // bad-day swing below S, rank units
        effNS: sk.effN,
    };
}

// Dense-render evaluation: same result shape as evaluateAt, but S and its
// measurement SE seS are INTERPOLATED from the precomputed wideGrid instead of
// re-running the wide-kernel local-linear MLE at t. The wide (30-day) kernel is
// locally linear over the grid spacing — the exact property buildResidualTrack
// already relies on — so the interpolated skill line and inner band match the
// exact fit to well under a rank, while the per-point wide MLE (by far the
// costliest part of the chart on dense histories) is skipped entirely. Only the
// directional performance swing (localPerformanceVar) is still computed per point.
// Falls back to the exact evaluateAt when the grid is absent (older decomp shape).
// Used by the Rank chart's 300-point grid; the exact evaluateAt stays for the
// single "current skill" readout, where precision matters and cost is trivial.
export function evaluateOnGrid(decomp, t) {
    if (!decomp) return null;
    const { wideGrid, resid } = decomp;
    if (!wideGrid || wideGrid.length < 2) return evaluateAt(decomp, t);
    const S = interpGrid(wideGrid, t, 'R');
    if (!Number.isFinite(S)) return null;
    const seS = interpGrid(wideGrid, t, 'se');
    const pv = localPerformanceVar(resid, t, decomp.sigma2Up || 0, decomp.sigma2Down || 0);
    return {
        S,                                // skill, internal-rank index (interpolated)
        seS,                              // measurement SE, rank units (interpolated)
        sigmaUp: Math.sqrt(pv.varUp),     // good-day swing above S, rank units
        sigmaDown: Math.sqrt(pv.varDown), // bad-day swing below S, rank units
        effNS: NaN,                       // not used by the chart render path
    };
}
