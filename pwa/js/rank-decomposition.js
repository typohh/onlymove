// js/rank-decomposition.js
//
// Two-kernel rank decomposition, fully continuous (no time bucketing):
//   S(t)    — slow "skill": a wide-kernel pooled logistic (τ_S ≈ 1 month).
//   σ_M(t)  — fast "form" swing: the genuine session-to-session performance
//             variance, separated from measurement noise.
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
//   2. S(t) = wide-kernel (τ_S) pooled logistic over x at the 80% target — your
//      rating over time, evaluated exactly at any t. seS(t) is its measurement
//      SE (how precisely we know skill — shrinks with data).
//   3. A narrow kernel (τ_M, ~session scale) tracks form+skill; the residual
//      r(t) = S_narrow(t) − S_wide(t) ~ N(0, σ_M² + v) carries the session swing
//      PLUS the narrow estimator's own noise v. The time-varying performance
//      variance σ_M²(t) is a variance-components ML over a τ_V window (NOT
//      method-of-moments r²−v: v is wildly heteroscedastic — all-correct/all-
//      wrong sessions have v→∞ — and ML's 1/(σ_M²+v)² weight downweights those,
//      where r²−v would be poisoned by them). The local σ_M²(t) is then shrunk
//      toward the global ML σ_M² by the effective number of independent sessions
//      in the window (Σ k/m, empirical Bayes): sparse windows fall back to the
//      whole-history swing so one isolated session can't spike the band. The
//      summary reports the global ML σ_M, or — when it's ~0 — a 95% profile-
//      likelihood upper bound ("under ±X, not detected").
//   4. Map S, seS and σ_M to rank with M (delta-method) so the chart's inner
//      band (S ± z·seS, measurement) and outer band (S ± z·√(seS²+σ_M²),
//      where a session lands) stay in rank units.
//
// M (rating_to_rank.json) is required: package_for_pwa ships only raw ratings,
// so the client can't rank anything without it. main.js awaits loadRatingRankMap()
// at init and aborts if it fails; computeRankDecomposition returns null if the
// curve for the active size is missing.

import { state } from './state.js';
import { TOTAL_RANKS } from './constants.js';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

const TAU_S_MS = 30 * DAY_MS;               // skill (slow): ~1-month kernel
const TAU_M_MS = 4 * HOUR_MS;               // form (fast): ~session kernel
const TAU_V_MS = 10 * DAY_MS;               // σ_M(t) variance-smoothing window
const KERNEL_TRUNC_SIGMA = 4;

// rank = the level you clear at this rate (matches model.js / fit_rank_curves).
const CALIBRATION_P = 0.8;
const LOGIT_CALIBRATION = Math.log(CALIBRATION_P / (1 - CALIBRATION_P));

// Success-curve slope, in rating space (the only space now — M is required).
// Must match pipeline/fit_rank_curves.LOGIT_PER_RATING.
const LOGIT_PER_RATING = 2.5;

const X_CLAMP_LO = -40;
const X_CLAMP_HI = 60;

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

// Coarse, regularly-spaced evaluation of the wide (skill) kernel. Used only to
// interpolate the wide track when forming form residuals; the rendered skill
// line evaluates the kernel exactly (see evaluateAt). The 30-day kernel is
// locally linear over the grid spacing, so interpolation here is exact to well
// under a rank.
function buildWideGrid(attempts, slope, points) {
    const t0 = attempts[0].t;
    const t1 = attempts[attempts.length - 1].t;
    const span = Math.max(1, t1 - t0);
    const grid = [];
    for (let i = 0; i < points; i++) {
        const t = t0 + span * (i / (points - 1));
        const w = poolLogistic(attempts, t, TAU_S_MS, slope);
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
// skill (rating space), with v = the narrow estimator's measurement variance
// and w = its effective sample size (reliability weight for the σ_M² fit). Each
// residual also gets `m`, its within-session multiplicity (≈ how many residuals
// belong to the same sitting): residuals from one session are near-duplicates,
// so dividing by m lets us count INDEPENDENT sessions, not attempts, downstream.
function buildResidualTrack(attempts, wideGrid, slope) {
    const resid = [];
    for (let j = 0; j < attempts.length; j++) {
        const tj = attempts[j].t;
        const nj = poolLogistic(attempts, tj, TAU_M_MS, slope);
        if (!Number.isFinite(nj.R)) continue;
        const Rw = interpGrid(wideGrid, tj, 'R');
        if (!Number.isFinite(Rw)) continue;
        const r = nj.R - Rw;
        resid.push({ t: tj, r2: r * r, v: nj.se * nj.se, w: nj.effN });
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

// Time-varying performance variance at t (rating²): kernel-localized
// variance-components ML, shrunk toward the global σ². Each residual is weighted
// by k (τ_V time kernel) ÷ m (de-duplicate within-session near-copies) so the
// window counts INDEPENDENT sessions, and v downweights uninformative ones.
// Empirical-Bayes shrinkage by effective sessions (nSess = Σ k/m) pulls sparse
// windows toward sigma2Global — so a lone isolated session can't spike the band,
// and the band never claims more local swing than the data supports.
function localPerformanceVar(resid, t, sigma2Global) {
    const rows = [];
    let nSess = 0;
    for (const d of resid) {
        if (!Number.isFinite(d.v)) continue;     // uninformative session: no signal
        const k = gaussianWeight((d.t - t) / TAU_V_MS);
        if (k <= 0) continue;
        const wgt = k / d.m;
        rows.push({ r2: d.r2, v: d.v, wgt });
        nSess += wgt;
    }
    if (!rows.length) return { var: sigma2Global, nSess: 0 };
    const local = mlSigma2(rows);
    const w = nSess / (nSess + SIGMA_PRIOR_SESSIONS);
    return { var: w * local + (1 - w) * sigma2Global, nSess };
}

// Global (whole-history) σ_M² point estimate + its 95% profile-likelihood upper
// bound (rating²). Weighted by 1/m so each session counts once. Used for the
// summary line: when the point estimate is ~0 we report the upper bound ("swing
// is under ±X, not detected") instead of a misleading ±0.00.
function globalPerformanceSigma(resid) {
    const rows = [];
    for (const d of resid) {
        if (!Number.isFinite(d.v)) continue;
        rows.push({ r2: d.r2, v: d.v, wgt: 1 / d.m });
    }
    if (rows.length < 2) return { sigma2: 0, upper2: 0 };
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
    return { sigma2: s2, upper2: 0.5 * (lo + hi) };
}

export function computeRankDecomposition(answerHistory) {
    const c = curveForSize(state.boardSize);
    if (!c) return null;   // no M ⇒ no rating→rank; the app shouldn't reach here.
    const slope = LOGIT_PER_RATING;

    const attempts = flattenAttempts(answerHistory);
    if (attempts.length < MIN_ATTEMPTS) return null;

    // Coarse wide track (for residual interpolation only) + per-attempt form
    // residuals. The decomp carries enough to evaluate S, seS and σ_M(t)
    // continuously at any t via evaluateAt — no bucket grid.
    const wideGrid = buildWideGrid(attempts, slope, WIDE_GRID_POINTS);
    const resid = buildResidualTrack(attempts, wideGrid, slope);

    // Global σ_M (point + 95% upper bound) — also the shrinkage target the
    // time-varying band pulls toward, so it must be on the decomp before any
    // evaluateAt call.
    const gp = globalPerformanceSigma(resid);
    const decomp = { attempts, resid, c, slope, sigma2Global: gp.sigma2 };

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

    // Map the global σ_M to rank at the current skill for the summary line.
    const g = cur ? Math.abs(Mslope(cur.ratingR, c)) : 1;

    decomp.summary = {
        currentSkill: cur && Number.isFinite(cur.S) ? cur.S : null,
        currentSkillSE: cur ? cur.seS : Infinity,
        sigmaM: Math.sqrt(gp.sigma2) * g,
        sigmaMUpper: Math.sqrt(gp.upper2) * g,
        longTermDrift: (sFirst && cur) ? cur.S - sFirst.S : null,
        longTermSpanMs: haveSpan ? lastT - firstT : null,
    };
    return decomp;
}

// Continuous evaluation at arbitrary t (no buckets). The wide kernel is
// evaluated exactly here, so the rendered skill line and inner band are exact;
// σ_M(t) comes from the local performance-variance estimate. All mapped to
// rank space with M (delta-method via g). Returns null if t has no support.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { attempts, resid, c, slope } = decomp;
    const sk = poolLogistic(attempts, t, TAU_S_MS, slope);   // exact wide skill
    if (!Number.isFinite(sk.R)) return null;
    const g = Math.abs(Mslope(sk.R, c));                     // d(rank)/d(rating)
    const pv = localPerformanceVar(resid, t, decomp.sigma2Global || 0);
    return {
        S: applyM(sk.R, c),                  // skill, rank space
        seS: sk.se * g,                      // measurement SE, rank space
        sigmaM: Math.sqrt(pv.var) * g,       // performance swing, rank space
        effNS: sk.effN,
        ratingR: sk.R,
    };
}
