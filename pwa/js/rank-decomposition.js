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
// The per-attempt slope preserves the calibration that estimating in raw-rating
// space used to give: the success curve has constant slope in rating space, so
// in rank space the slope must vary with the local steepness of M. With it, a
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
import { sigmoid } from './math-utils.js';
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
// by train_rating_v2 from the bot ladder) gives P(clear) = c + (1−c)·σ(slope·
// (R−x)+κ(c)); κ(c) keeps P(clear)=80% at R==x for any c. Far below a problem
// (R≪x) P→c, so a success there is weak evidence of skill — which correctly
// discounts a strong user's lucky above-rank clears (the early over-credit
// behind the Rank-tab dip). The 3PL constants + κ/floor helpers live in irt.js.

// Per attempt the success-curve slope is the problem's OWN 2PL discrimination
// a_p (stored as `problemSlope`, shipped raw by package_for_pwa), converted to a
// logit-per-internal-rank slope by dividing by the local steepness of M. The
// EGF anchor (map_rating_v2_to_egf → fit_user_rating) uses the same per-problem
// a_p, so the rank loop stays closed. LOGIT_PER_RATING is only the FALLBACK slope
// for legacy entries that predate `problemSlope`; set near the corpus-mean a_p
// (~1.6) so an un-backfilled attempt isn't mis-scaled against the re-anchored M.
const LOGIT_PER_RATING = 1.6;

// Representative logit-per-internal-rank used when M's local slope is unusable
// (degenerate curve segment, or a legacy entry with only problemRank stored).
const LOGIT_PER_RANK_FALLBACK = 0.28;
// Guard the converted per-attempt slope against pathological M' values. SLOPE_HI
// is set above the realized a_p/|M'| range (~2.1) so genuinely sharp problems
// aren't clipped; it mainly catches the tiny-|M'| (MSLOPE_FLOOR) pathology.
const SLOPE_LO = 0.05, SLOPE_HI = 3.0;
const MSLOPE_FLOOR = 1e-3;

// Working coordinate is the internal rank index (~0..29, with extrapolation
// past either end for very weak/strong users).
const X_CLAMP_LO = -15;
const X_CLAMP_HI = 45;

const MIN_ATTEMPTS = 20;        // below this, no decomposition
const WIDE_GRID_POINTS = 120;   // coarse skill grid for residual interpolation
const SUPPORT_EFFN = 10;        // effective-N floor for "supported" skill range

// σ_M is session-to-session variance: it needs several INDEPENDENT sessions in
// the τ_V window to estimate reliably. With few, a local window's estimate is
// noise (or one isolated session blows the band into a spike). So the local
// estimate is shrunk toward the global one by effective session count (empirical
// Bayes): sparse windows fall back to the whole-history swing, dense windows keep
// their local value. SESSION_CORR_MS de-duplicates within-session near-copies.
const SESSION_CORR_MS = TAU_M_MS * Math.SQRT2;  // within-session correlation length
const SIGMA_PRIOR_SESSIONS = 8;  // shrinkage prior strength, in effective sessions

// ---------------------------------------------------------------------------
// M: rating → rank map (pwa/tsumegos/rating_to_rank.json), per board size.
// ---------------------------------------------------------------------------
// M ships inside the bundle now (see the import above), so there is nothing to
// fetch and nothing to fail: it is present the instant the module evaluates.
// loadRatingRankMap is retained for the existing call site in main.js, which keeps
// a null-guard against a malformed/empty packaged file (a build-time error, not a
// runtime one).
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
// remap. (Replaces the old arbitrary [0,1] quality.)
export function slopeToQuality(slope) {
    if (typeof slope !== 'number' || !Number.isFinite(slope) || slope <= 0) return 0;
    return slope * slope;
}

function curveForSize(size) {
    const m = _ratingRankMap && _ratingRankMap[String(size)];
    return (m && Array.isArray(m.ratings) && Array.isArray(m.rank_indices)
            && m.ratings.length >= 2) ? m : null;
}

// interpExtrap lives in shared/rank-math.js (one copy shared with the Worker).

function applyM(rating, c) { return interpExtrap(rating, c.ratings, c.rank_indices); }
function Mslope(rating, c) {
    const h = 1e-3;
    return (applyM(rating + h, c) - applyM(rating - h, c)) / (2 * h);
}

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


function lowerBound(attempts, lo) {
    let a = 0, b = attempts.length;
    while (a < b) { const m = (a + b) >> 1; if (attempts[m].t < lo) a = m + 1; else b = m; }
    return a;
}

// Kernel-weighted pooled 3PL MLE for the internal-rank skill R at time t0:
// P(correct)=c_j+(1−c_j)·σ(slope_j·(R−x_j)+κ_j), where each attempt j carries its own
// slope (logit per internal rank), guessing floor c_j (problemFloor) and κ_j=κ(c_j)
// (so P=80% at R==x_j) — all set in flattenAttempts. Returns { R, se, effN };
// attempts MUST be sorted by t.
function poolLogistic(attempts, t0, tau) {
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
            const slope = attempts[j].slope, c = attempts[j].floor, span = clearSpan(c);
            const praw = sigmoid(slope * (R - attempts[j].x) + attempts[j].kappa);
            let p = c + span * praw;
            if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
            const g = span * slope * praw * (1 - praw);   // dP/dR
            score += w * (attempts[j].correct - p) * g / (p * (1 - p));
            info += w * g * g / (p * (1 - p));
        }
        if (info < 1e-9) break;
        let step = score / info;
        if (step > 2) step = 2; else if (step < -2) step = -2;
        R += step;
        if (R < X_CLAMP_LO) R = X_CLAMP_LO; else if (R > X_CLAMP_HI) R = X_CLAMP_HI;
        if (Math.abs(step) < 1e-6) break;
    }

    // Sandwich SE: Var ≈ Σ w²v / (Σ wv)², v = (dP/dR)²/(P(1−P)) (4PL Fisher info).
    let A = 0, B = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        if (w <= 0) continue;
        const slope = attempts[j].slope, c = attempts[j].floor, span = clearSpan(c);
        const praw = sigmoid(slope * (R - attempts[j].x) + attempts[j].kappa);
        let p = c + span * praw;
        if (p < 1e-9) p = 1e-9; else if (p > 1 - 1e-9) p = 1 - 1e-9;
        const g = span * slope * praw * (1 - praw);
        const v = g * g / (p * (1 - p));
        A += w * v; B += w * w * v;
    }
    const se = A > 0 ? Math.sqrt(B) / A : Infinity;
    return { R, se, effN };
}

function clampSlope(s) {
    if (!Number.isFinite(s) || s <= 0) return LOGIT_PER_RANK_FALLBACK;
    return Math.min(SLOPE_HI, Math.max(SLOPE_LO, s));
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
function flattenAttempts(answerHistory) {
    return answerHistory
        .filter(a => typeof a.date === 'number' && !a.isRepeat && !a.linked)
        .map(a => {
            const size = a.size ?? state.boardSize;
            const c = curveForSize(size);
            const raw = (typeof a.problemRawRating === 'number')
                ? a.problemRawRating
                : a.problemRating;
            const aP = (typeof a.problemSlope === 'number' && a.problemSlope > 0)
                ? a.problemSlope : LOGIT_PER_RATING;
            let x, slope;
            if (c && typeof raw === 'number') {
                x = applyM(raw, c);
                slope = clampSlope(aP / Math.max(Math.abs(Mslope(raw, c)), MSLOPE_FLOOR));
            } else if (typeof a.problemRank === 'number') {
                x = a.problemRank * TOTAL_RANKS;
                slope = LOGIT_PER_RANK_FALLBACK;
            } else {
                return null;
            }
            // 3PL guessing floor (problemFloor, shipped per problem). Legacy entries
            // without it fall back to 0 → the 2PL curve, unchanged behaviour.
            const flr = clampFloor(a.problemFloor);
            return { t: a.date, x, slope, correct: a.correct ? 1 : 0,
                     floor: flr, kappa: clearKappa(flr), src: a };
        })
        .filter(Boolean)
        .sort((a, b) => a.t - b.t);
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
        const w = poolLogistic(attempts, t, TAU_S_MS);
        grid.push({ t, R: w.R, se: w.se, effN: w.effN });
    }
    return grid;
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

// Form residual at each attempt time: r = narrow-kernel skill − wide-kernel
// skill (internal-rank space), with v = the narrow estimator's measurement
// variance and w = its effective sample size (reliability weight for the σ_M²
// fit). The sign of r is kept (`up`) so the swing can be fit separately above
// (r ≥ 0, good days) and below (r < 0, bad days) long-term skill. Each residual
// also gets `m`, its within-session multiplicity (≈ how many residuals belong to
// the same sitting): residuals from one session are near-duplicates, so dividing
// by m lets us count INDEPENDENT sessions downstream.
function buildResidualTrack(attempts, wideGrid) {
    const resid = [];
    for (let j = 0; j < attempts.length; j++) {
        const tj = attempts[j].t;
        const nj = poolLogistic(attempts, tj, TAU_M_MS);
        if (!Number.isFinite(nj.R)) continue;
        const Rw = interpGrid(wideGrid, tj, 'R');
        if (!Number.isFinite(Rw)) continue;
        const r = nj.R - Rw;
        resid.push({ t: tj, r2: r * r, up: r >= 0, v: nj.se * nj.se, w: nj.effN });
    }
    for (let i = 0; i < resid.length; i++) {
        let m = 0;
        for (let j = 0; j < resid.length; j++) {
            const dd = (resid[i].t - resid[j].t) / SESSION_CORR_MS;
            if (Math.abs(dd) > KERNEL_TRUNC_SIGMA) continue;
            m += Math.exp(-0.5 * dd * dd);
        }
        resid[i].m = m;
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
    const target = LL(s2) - 1.92;   // χ²₁ 95% one-parameter drop
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
// heavy lifting (wide skill grid, form residuals, global swing, summary). Both
// the near-rank filter's seed pass and the final official pass route through
// here. Returns null below MIN_ATTEMPTS.
function decompose(attempts) {
    if (attempts.length < MIN_ATTEMPTS) return null;

    // Coarse wide track (for residual interpolation only) + per-attempt form
    // residuals. The decomp carries enough to evaluate S, seS and σ_M(t)
    // continuously at any t via evaluateAt — no bucket grid.
    const wideGrid = buildWideGrid(attempts, WIDE_GRID_POINTS);
    const resid = buildResidualTrack(attempts, wideGrid);

    // Global σ_M (point + 95% upper bound) — also the shrinkage target the
    // time-varying band pulls toward, so it must be on the decomp before any
    // evaluateAt call. Already in internal-rank units (the working coordinate).
    const gp = globalPerformanceSigma(resid);
    const decomp = { attempts, resid, sigma2Up: gp.sigma2Up, sigma2Down: gp.sigma2Down };

    // "Supported" range: grid points whose skill estimate pools ≥ SUPPORT_EFFN
    // effective attempts. Current skill and long-term drift read its endpoints.
    let firstT = null, lastT = null;
    for (const g of wideGrid) {
        if (g.effN >= SUPPORT_EFFN && Number.isFinite(g.R)) {
            if (firstT === null) firstT = g.t;
            lastT = g.t;
        }
    }
    if (lastT === null) lastT = wideGrid[wideGrid.length - 1].t;

    const cur = evaluateAt(decomp, lastT);
    const haveSpan = firstT !== null && lastT > firstT;
    const sFirst = haveSpan ? evaluateAt(decomp, firstT) : null;

    decomp.summary = {
        currentSkill: cur && Number.isFinite(cur.S) ? cur.S : null,
        currentSkillSE: cur ? cur.seS : Infinity,
        sigmaM: Math.sqrt(gp.sigma2),         // pooled swing — drives detection test
        sigmaMUpper: Math.sqrt(gp.upper2),
        sigmaUp: Math.sqrt(gp.sigma2Up),      // good-day overperformance (ranks)
        sigmaDown: Math.sqrt(gp.sigma2Down),  // bad-day underperformance (ranks)
        longTermDrift: (sFirst && cur) ? cur.S - sFirst.S : null,
        longTermSpanMs: haveSpan ? lastT - firstT : null,
    };
    return decomp;
}

// The official skill decomposition — skill line, 95% measurement CI and the
// daily-variance band — computed from NEAR-RANK attempts only. Attempts taken
// far from the user's rank carry systematic bias: a strong user's lucky
// above-rank clears over-credit skill, and the guessing floor inflates far-below
// clears, both of which distort the level AND fatten the residual that feeds the
// daily-variance band. So we drop them before estimating. This is a seeded
// one-pass fixed point, NOT a circular filter:
//   1. Seed S0(t): the wide-kernel skill over the FULL history, from the win/loss
//      sequence alone (independent of any distance filter). Only S0 is needed to
//      measure distance, so the seed skips the resid/σ machinery.
//   2. Each attempt is scored against the CONTEMPORANEOUS seed rank S0(t_attempt)
//      — not the user's current rank — so "what counts as far" tracks the rank as
//      it moved.
//   3. The decomposition is re-run on the survivors → the official curve + bands.
// One pass suffices: the filter drops exactly the low-Fisher-information attempts
// the estimator already down-weights, so the result moves from the full-history
// estimate only where far-from-rank attempts were genuinely biasing it.
//
// "Far" is measured in the SUCCESS-CURVE'S OWN LATENT z = slope·(S0 − x), NOT in
// probability. z = 0 ⟺ problem exactly at rank ⟺ P = 0.8 (κ is defined to make
// that so), so a band symmetric in z is symmetric around the 80% serving target.
// Probability is the wrong space for symmetry: it's bounded at 1 (0.8 sits near
// the ceiling, so equal ΔP means unequal rank gaps up vs down) and the floor c
// squashes the low end (P→c, not 0) per-problem. z is floor-independent (c only
// rescales P, it doesn't move z=0) and slope-aware (a sharp problem decides at a
// smaller rank gap than a flat one — genuinely "farther" at equal Δrank). We take
// the easy-side edge as a single probability knob pHi and MIRROR it in z: keep
//   |slope·(S0 − x)| ≤ L,   L = logit(pHi) − logit(0.8)
// pHi = 0.95 ⇒ L ≈ 1.56.
//
// If too few attempts survive the filter (a short or very erratic history), fall
// back to the unfiltered decomposition rather than blanking the chart.
const NEAR_RANK_P_HI = 0.95;
const LOGIT_80 = Math.log(0.8 / 0.2);
export function computeRankDecomposition(answerHistory) {
    // flattenAttempts maps each attempt through its own size's M, so a merged
    // 9x9+13x13 history pools on the internal-rank axis. Attempts whose size curve
    // is missing are dropped there.
    const all = flattenAttempts(answerHistory);
    if (all.length < MIN_ATTEMPTS) return null;

    // Seed: only the wide-kernel skill S0(t) is needed to measure each attempt's
    // contemporaneous distance from rank, so the seed carries no resid track
    // (evaluateAt's variance terms fall back to 0 on an empty resid).
    const seed = { attempts: all, resid: [] };
    const L = Math.log(NEAR_RANK_P_HI / (1 - NEAR_RANK_P_HI)) - LOGIT_80;
    const near = all.filter(a => {
        const e = evaluateAt(seed, a.t);
        return e && Number.isFinite(e.S) && Math.abs(a.slope * (e.S - a.x)) <= L;
    });

    return decompose(near.length >= MIN_ATTEMPTS ? near : all);
}

// Continuous evaluation at arbitrary t (no buckets). The wide kernel is
// evaluated exactly here, so the rendered skill line and inner band are exact;
// σ_M(t) comes from the local performance-variance estimate. The working
// coordinate IS the internal rank, so S/seS/σ_M need no remap. Returns null if
// t has no support.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { attempts, resid } = decomp;
    const sk = poolLogistic(attempts, t, TAU_S_MS);          // exact wide skill
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
