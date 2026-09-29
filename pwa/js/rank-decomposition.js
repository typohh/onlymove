// js/rank-decomposition.js
//
// Career rank decomposition, fully continuous (no time bucketing):
//   S(t)    — slow "skill": the state-space smoother of skill-smoother.js (a
//             random walk read through per-session form, smoothed over the whole
//             history).
//   σ_M(t)  — fast "form" swing: the genuine session-to-session performance
//             variance, separated from measurement noise.
//
// We work directly in INTERNAL-RANK space, pooling across board sizes. The
// internal rank (EGF-anchored, board-independent) is the one common skill axis
// shared by 9x9 and 13x13, so a single merged skill curve is well-defined only
// there. Each attempt is mapped through ITS OWN size's M before pooling:
//   - x = applyM(rawRating, sizeCurve)            (internal rank)
//   - slope = a_p / |M'(rawRating)|               (logit per internal rank)
// The per-attempt slope preserves calibration: the success curve has constant
// slope in rating space, so in rank space the slope must vary with the local
// steepness of M. With it, a
// 4d still reads as a 4d. Then:
//
//   1. Per attempt: x (internal rank) and its own slope, from its size's curve.
//   2. S(t) = the posterior mean of skill at t given the whole history, at the
//      80% target — your internal rank over time, evaluated exactly at any t.
//      seS(t) is its posterior SD: it shrinks with data and grows across gaps by
//      the drift the gap allows.
//   3. A narrow kernel (τ_M, ~session scale) tracks form+skill; the residual
//      r(t) = S_narrow(t) − S(t) ~ N(0, σ_M² + v) carries the session swing
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

import { state, updateUserRank, updateServeRank, updateDisplayRank, updateRankSe } from './state.js';
import { interpExtrap } from '../shared/rank-math.js';
import { estimateWindowRankFromFlat, windowMLE, displayRankFor, WIN_CAP, applyM, flattenAttempt, seFloorAt, weightedMedianOverWindow } from '../shared/rank-window.js';
import { sigmoid, gaussianWeight as gaussianWeightUntruncated, lowerBound } from './math-utils.js';
import { clearSpan, anchorInfo } from './irt.js';
import { smoothSkill, SESSION_GAP_MS } from './skill-smoother.js';
// Baked at build time from the pipeline's output (pwa/tsumegos/rating_to_rank.json).
import RATING_RANK_MAP from '../tsumegos/rating_to_rank.json';

const HOUR_MS = 3600 * 1000;
const DAY_MS = 24 * HOUR_MS;

const TAU_M_MS = 4 * HOUR_MS;               // form (fast): ~session kernel
const TAU_V_MS = 10 * DAY_MS;               // σ_M(t) variance-smoothing window
const KERNEL_TRUNC_SIGMA = 4;

// rank = the level you clear at this rate (CALIBRATION_P; matches recall-model.js /
// fit_rank_curves). The per-problem 3PL guessing floor c (problemFloor, shipped
// by train_mirt from the bot ladder) gives P(clear) = c + (1−c)·σ(slope·
// (R−x)+κ(c)); κ(c) keeps P(clear)=80% at R==x for any c. Far below a problem
// (R≪x) P→c, so a success there is weak evidence of skill — which correctly
// discounts a strong user's lucky above-rank clears. The 3PL constants + κ/floor
// helpers live in irt.js.

// Per attempt the success-curve slope is the problem's OWN 2PL discrimination
// a_p (stored as `problemSlope`, shipped raw by package_for_pwa), converted to a
// logit-per-internal-rank slope by dividing by the local steepness of M. The
// EGF anchor (map_rating_to_rank → fit_user_rating) uses the same per-problem
// a_p, so the rank loop stays closed. The per-attempt slope constants
// (SLOPE_LO/HI, MSLOPE_FLOOR) + the applyM / slope-in-rank-space transform live
// in pwa/shared/rank-window.js so the Worker's buffer flatten matches this one;
// applyM and flattenAttempt are imported above.

// Working coordinate is the internal rank (see shared/rank-math.js for the
// convention), clamped past either end for very weak/strong users.
const X_CLAMP_LO = -34.5;
const X_CLAMP_HI = 25.5;

const MIN_ATTEMPTS = 20;        // below this, no decomposition

// σ_M is session-to-session variance: it needs several INDEPENDENT sessions in
// the τ_V window to estimate reliably. With few, a local window's estimate is
// noise (or one isolated session blows the band into a spike). So the local
// estimate is shrunk toward the global one by effective session count (empirical
// Bayes): sparse windows fall back to the whole-history swing, dense windows keep
// their local value. Sessions are split at SESSION_GAP_MS (skill-smoother.js), the
// same boundary at which the skill smoother redraws its form offset.
const SIGMA_PRIOR_SESSIONS = 8;  // shrinkage prior strength, in effective sessions

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

// Per-problem (raw 2PL slope a_p, guessing floor c) → problem-selection weight
// ∝ the Fisher information the problem provides about a user's ability at the
// 80%-clear anchor: I = a_p²·anchorInfo(c). Selection draws from a difficulty-
// matched pool (the 50 problems nearest the user's rank, all ≈80%-clear), so
// evaluating every candidate at its own anchor is the right common reference.
// The floor factor is NOT constant across that pool — it is the 4PL correction
// the plain a_p² weight used to omit (down-weighting problems a user can guess
// into, which tell you less about their rank for the same 80% clear rate). Expect
// its effect HERE to be second-order: a difficulty-matched pool holds problems of
// similar floor, so this factor varies far less across candidates than a_p² does.
// It is the SE floor (seFloorAt, off the shipped corpus curve) where the same 2PL
// constant was load-bearing; this site is the consistency half of that fix — both
// read the same anchorInfo. Returned RAW (not
// on any [0,1] range):
// selection only ranks candidates, so only the ORDER matters — a sharp problem is
// quadratically preferred and a flat or guessable one down-weighted, by genuine
// information rather than a min-max remap. A problem missing its slope or floor
// is malformed and weighs nothing; there is no default for either.
export function problemInfoWeight(slope, floor) {
    if (!Number.isFinite(slope) || slope <= 0 || !Number.isFinite(floor)) return 0;
    return slope * slope * anchorInfo(floor);
}

export function curveForSize(size) {
    const m = _ratingRankMap && _ratingRankMap[String(size)];
    return (m && Array.isArray(m.ratings) && Array.isArray(m.ranks)
            && m.ratings.length >= 2) ? m : null;
}

// interpExtrap lives in shared/rank-math.js; applyM + the per-attempt flatten
// transform live in shared/rank-window.js (one copy shared with the Worker).
// applyM is imported above.

// Raw rating → internal RANK for `size` (shared/rank-math.js's one convention).
// Returns null when M isn't loaded; the whole client converts ratings to ranks
// through this, so package_for_pwa never has to.
//
// Deliberately NOT clamped to the kyu/dan ladder. The ladder is what we LABEL, not
// a bound on the value: M's own knots don't span it, and applyM extrapolates
// linearly past both end knots, so a problem easier than the weakest knot goes
// below and one harder than the strongest goes above. Both are normal — the corpus
// hard tail is unattempted KataGo-hard material that extrapolates well past the
// human knots. Consumers must treat this as an unbounded continuous coordinate.
export function rankForSize(rawRating, size) {
    const c = curveForSize(size);
    if (!c || typeof rawRating !== 'number') return null;
    return applyM(rawRating, c);
}

// Apply a RATING-space delta to a rank: invert M at the rank, shift by deltaRating
// (raw rating units), re-apply M. Exact rather than linearized via M', so it stays
// correct across knot boundaries. The curve is shared across both board sizes
// (map_rating_to_rank writes one curve to both keys), so size 9's key stands in for
// either. Used by the stats Rank tab to turn the population time-adjustment
// (alpha · Δlog10-minutes, fit in rating units by pipeline.fit_population_constants)
// into a rank-space shift.
export function rankAfterRatingDelta(rank, deltaRating) {
    const c = curveForSize(9);
    if (!c || !Number.isFinite(rank) || !Number.isFinite(deltaRating)) return null;
    const rating = interpExtrap(rank, c.ranks, c.ratings);
    return applyM(rating + deltaRating, c);
}

// Rating units spanned by ONE rank, from the shipped curve's endpoints. This is the
// unit constant between a gap in rank units and the same gap in raw-rating units;
// the recall model needs it to center its PROBLEM_DIFFICULTY prior (its gap feature
// is in rank units while a_p is per RATING unit). An endpoint mean — local knot
// slopes vary a few fold — the model's fitted multiplier absorbs the local mismatch.
export function ratingPerRank() {
    const c = curveForSize(9);   // one shared curve, written to both size keys
    if (!c) return null;
    const dRating = c.ratings[c.ratings.length - 1] - c.ratings[0];
    const dRank = c.ranks[c.ranks.length - 1] - c.ranks[0];
    return dRank > 0 ? dRating / dRank : null;
}

// Stable per-size signature of the loaded curve. Stored ranks are invalidated
// when this changes (a fit_rank_curves re-fit that didn't also change ratings).
export function ratingRankCurveVersion(size) {
    const c = curveForSize(size);
    return c ? JSON.stringify([c.ratings, c.ranks]) : null;
}

// ---------------------------------------------------------------------------

// Truncated variant of math-utils' gaussianWeight: exactly 0 past
// KERNEL_TRUNC_SIGMA so the kernel windows stay finite.
function gaussianWeight(dtOverTau) {
    if (Math.abs(dtOverTau) > KERNEL_TRUNC_SIGMA) return 0;
    return gaussianWeightUntruncated(dtOverTau);
}

// Kernel-weighted pooled 4PL MLE for the internal-rank skill R at time t0 — the
// narrow (session-scale) fit behind the form residuals:
// P(correct)=c_j+(1−c_j−LAPSE)·σ(slope_j·(R−x_j)+κ_j), where each attempt j carries
// its own slope (logit per internal rank), guessing floor c_j (problemFloor) and
// κ_j=κ(c_j) (so P=80% at R==x_j) — all set in flattenAttempts. Local-constant in
// time: a session cannot support a trend. Returns { R, se }; attempts MUST be
// sorted by t.
function poolLogistic(attempts, t0, tau) {
    const win = KERNEL_TRUNC_SIGMA * tau;
    const lo = lowerBound(attempts, t0 - win, a => a.t);
    const hi = lowerBound(attempts, t0 + win + 1, a => a.t);
    if (hi <= lo) return { R: NaN, se: Infinity };

    let sumW = 0, sumWx = 0;
    for (let j = lo; j < hi; j++) {
        const w = gaussianWeight((attempts[j].t - t0) / tau);
        sumW += w; sumWx += w * attempts[j].x;
    }
    if (sumW <= 0) return { R: NaN, se: Infinity };

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

    // Sandwich SE: Var ≈ Σ w²·v / (Σ w·v)², v = (dP/dR)²/(P(1−P)) (4PL Fisher info).
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
    return { R, se };
}

// Flatten answerHistory to sorted [{t, x, slope, correct}] in INTERNAL-RANK
// space. Each attempt is mapped through ITS OWN size's curve (merged entries
// carry `size`; otherwise the active size is assumed): x is the internal rank
// index applyM(rawRating), slope is the logit-per-internal-rank at that rating
// = a_p / |M'(rating)|, where a_p is the problem's own 2PL discrimination
// (`problemSlope`). `problemRating` holds the raw rating (legacy entries may
// carry a separate `problemRawRating`). An entry missing its size curve, rating,
// slope or floor is dropped (flattenAttempt returns null for it).
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
            return f ? { ...f, t: a.date, src: a } : null;
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
// delegates. There is no seed: the fit starts from the population prior every time.
// Returns:
//   rank        — the posterior MEAN, the honest estimate everything CALIBRATED
//                 scores against (server translations, recall, weakness, pred_rank)
//   se          — the posterior standard deviation (rank units)
//   displayRank — what the front page shows: conservative while se is wide
//   serveRank   — what problem SELECTION targets; ramps from displayRank onto rank
// Both offsets scale with the EXCESS se over a RANK-DEPENDENT floor (seFloorAt in
// rank-window.js, off the shipped corpus curve): the irreducible SE a full at-rank
// window would leave, set by the local corpus slope, so both collapse onto the user's
// OWN rank at convergence on every part of the ladder — not just where a constant
// floor happened to fit.
// null when no usable (first-time, curve-mapped) attempt exists.
export function estimateWindowRank(answerHistory) {
    return estimateWindowRankFromFlat(flattenAttempts(answerHistory));
}

// Recompute the three rank slots from the merged history and persist them. The ONE
// place that writes them, so every writer prices the SE floor the same way. A pure
// function of the history — and, since the floor comes off a shipped curve rather
// than the loaded shards, of nothing else — so it is idempotent and needs no
// ordering against problem loading.
export function reanchorRank(answerHistory) {
    const est = estimateWindowRank(answerHistory);
    if (!est) return null;
    updateUserRank(est.rank);
    updateServeRank(est.serveRank);
    updateDisplayRank(est.displayRank);
    updateRankSe(est.se);
    return est;
}

// Two-regime selection switch (problem-loader.js): the rank estimate counts as
// CONVERGED once its posterior SE is within 2x the local irreducible floor —
// "known to ~a stone", good enough to serve against. Information scales as
// 1/se², so this fires at ~1/4 of terminal information: deliberately early,
// because past this point extra rank information is worth less than weakness
// personalization (and a user parked in a band depletes its high-slope tail
// anyway, so slope-priority stops changing the served set). Fails CLOSED
// (not converged → information regime) when no se is stored yet.
export function rankConverged() {
    const se = state.rankSe;
    if (typeof se !== 'number' || !Number.isFinite(se)) return false;
    const floor = seFloorAt(state.userRank);
    if (!(floor > 0) || !Number.isFinite(floor)) return false;
    return se < 2 * floor;
}

// The front-page (window) rank as it evolved across history: the SAME positional
// window 4PL MLE the live headline uses (windowMLE over the trailing WIN_CAP
// first-time attempts), re-evaluated at each of `times`. A pure function of
// decomp.attempts — already flattened to internal-rank space and time-sorted by
// flattenAttempts — and it returns the SHOWN rank (displayRankFor), not the raw
// posterior mean, so its right endpoint reproduces the front-page badge
// (state.displayRank) exactly and the chart never contradicts the home screen. The
// slow skill line S(t) beside it is deliberately NOT display-adjusted: that one is
// the stable skill estimate, and the conservatism offset belongs only to the number
// we present as "your rank". `times` MUST be ascending; the
// return is aligned 1:1 with `times`, each element the internal rank or null
// (no first-time attempt exists yet at that time). Only the trailing WIN_CAP
// attempts affect windowMLE, so we pass just that suffix — bit-identical to the
// whole prefix (both have M = min(prefixLen, WIN_CAP)) but O(WIN_CAP) per point.
// There is no warm start: the posterior is built from the population prior every
// time, which is what makes each point independent of the ones before it.
export function windowRankSeries(decomp, times) {
    const out = new Array(times.length).fill(null);
    const attempts = decomp && decomp.attempts;
    if (!attempts || !attempts.length) return out;
    let idx = 0;            // count of attempts with t <= the current grid time
    for (let i = 0; i < times.length; i++) {
        while (idx < attempts.length && attempts[idx].t <= times[i]) idx++;
        if (idx === 0) continue;   // no first-time attempt yet
        const prefix = attempts.slice(Math.max(0, idx - WIN_CAP), idx);
        const { R, se } = windowMLE(prefix);
        // Same rank-dependent floor the live badge uses, taken at this point's own
        // estimate — so the line is the badge's own trajectory, not an approximation
        // of it.
        if (Number.isFinite(R)) out[i] = displayRankFor(R, se);
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

// Form residuals — ONE per SESSION (a contiguous run of attempts with < SESSION_
// GAP_MS between neighbours), not one per attempt. r = narrow-kernel skill −
// smoothed skill (internal-rank space) evaluated at the session's centre; v is
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
function buildResidualTrack(attempts, skill) {
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
        const r = nj.R - skill.at(tc).S;
        resid.push({ t: tc, r2: r * r, up: r >= 0, v: nj.se * nj.se, m: 1 });
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
// heavy lifting (skill smoother, form residuals, global swing, summary).
// Returns null below MIN_ATTEMPTS.
function decompose(attempts) {
    if (attempts.length < MIN_ATTEMPTS) return null;

    // The skill smoother is fitted once over the whole history and evaluated
    // exactly at any t (skill-smoother.js); the form residuals are read against it.
    const skill = smoothSkill(attempts);
    const resid = buildResidualTrack(attempts, skill);

    // Global σ_M (point + 95% upper bound) — also the shrinkage target the
    // time-varying band pulls toward, so it must be on the decomp before any
    // evaluateAt call. Already in internal-rank units (the working coordinate).
    const gp = globalPerformanceSigma(resid);
    const decomp = { attempts, skill, resid, sigma2Up: gp.sigma2Up, sigma2Down: gp.sigma2Down };
    decomp.summary = {
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
// 4PL likelihood absorbs that WITHIN the estimate, so no pre-filter is needed: the
// per-problem guessing floor c discounts far-below clears (P→c there, so a clear is
// weak evidence of skill) and the LAPSE upper asymptote caps the score of a
// far-above slip (P(1−P)→LAPSE·(1−LAPSE) while the gradient →0).
export function computeRankDecomposition(answerHistory) {
    // flattenAttempts maps each attempt through its own size's M, so a merged
    // 9x9+13x13 history pools on the internal-rank axis. Attempts whose size curve
    // is missing are dropped there.
    const all = flattenAttempts(answerHistory);
    if (all.length < MIN_ATTEMPTS) return null;
    return decompose(all);
}

// Continuous evaluation at arbitrary t (no buckets): skill and its posterior SD
// from the smoother, σ_M(t) from the local performance-variance estimate. The
// working coordinate IS the internal rank, so S/seS/σ_M need no remap.
export function evaluateAt(decomp, t) {
    if (!decomp) return null;
    const { S, se } = decomp.skill.at(t);
    const pv = localPerformanceVar(decomp.resid, t, decomp.sigma2Up || 0, decomp.sigma2Down || 0);
    return {
        S,                                // skill, internal rank
        seS: se,                          // posterior SD of skill, rank units
        sigmaUp: Math.sqrt(pv.varUp),     // good-day swing above S, rank units
        sigmaDown: Math.sqrt(pv.varDown), // bad-day swing below S, rank units
    };
}
