// js/model.js

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, setModelParams, setAdamState, getMergedAnswerHistory } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, RECALL_CANDIDATE_HISTORY_LENGTH } from './constants.js';
import { sigmoid } from './math-utils.js';
import { clearKappa, clampFloor } from './irt.js';
import { adamUpdate } from './optimizer.js';

// ------------------------------------------------------------------------------------------------
// --- Moved from problem-loader.js ---------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

export function findFirst(sortedArr, target) {
    let low = 0;
    let high = sortedArr.length - 1;
    let result = -1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        if (sortedArr[mid] >= target) {
            if (sortedArr[mid] === target) result = mid;
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return result;
}

export function findLast(sortedArr, target) {
    let low = 0;
    let high = sortedArr.length - 1;
    let result = -1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        if (sortedArr[mid] <= target) {
            if (sortedArr[mid] === target) result = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return result;
}

export function calculatePercentile(sortedArr, target) {
    if (sortedArr.length === 0) return 0;
    const firstIndex = findFirst(sortedArr, target);
    if (firstIndex === -1) return 0;
    const lastIndex = findLast(sortedArr, target);
    const below = firstIndex;
    const equal = lastIndex - firstIndex + 1;
    return (below + 0.5 * equal) / sortedArr.length;
}

// --- Retention/stability model ----------------------------------------------
// Recall is an FSRS-style retention curve anchored at R(0)=1. A REPEAT
// encounter's memory stability S (days) reads the base model's cold-solve logit
// z = logit(p) — which already integrates difficulty, review-time and
// solving-time — and maps it to stability:
//
//   z    = INTERCEPT + TIME_SOLVING·timePct + REVIEW_TIME·reviewPct
//          + PROBLEM_DIFFICULTY·problemRank + SIZE_IS_13·size13      (= logit p)
//   lnS  = LN_C + KAPPA·(z − Z0) + GROWTH·r
//   R(t) = exp(-t_days / exp(lnS))
//
// i.e. S = exp(LN_C) · odds(p/0.7)^KAPPA · GROWTH_FACTOR^r. R(0)=1 (no instant-
// due), and z enters LINEARLY (NOT through ln σ(z), whose `≈z` hard-side tail
// collapsed hard problems to the floor). KAPPA, GROWTH and Z0 are POPULATION
// constants fit across 88 users on recall-era, post-first-observed-failure
// events (the only data where the elapsed gap is a true retention interval and
// the success-count r is established): difficulty/skill drives stability with
// slope KAPPA (LR≈148, decisive), each success multiplies S by GROWTH_FACTOR,
// and Z0=logit(0.7) is the reference difficulty so LN_C is the base stability at
// p=0.7. Only LN_C — the per-user base durability — is fit; it is L2-shrunk to
// the population prior so it stays sane with little data and adapts with more.
//
// The same logit z is the NOVEL cold-solve head P=σ(z); that head is trained on
// correctness over ALL rows (so REVIEW_TIME, which is 0 on novel rows, is still
// identified from repeats) and backs the stats readouts. The retention head
// reads z but does NOT backprop into the logit params (z is owned by the
// correctness head); it only fits LN_C. No IS_NOVEL indicator.
const NEG_LN_080 = -Math.log(0.80); // 0.22314 — t/S at the 0.80 window-open
const NEG_LN_060 = -Math.log(0.60); // 0.51083 — t/S at the 0.60 window-close
// Stability floor/ceiling (days). Floor keeps a just-failed problem from coming
// due sooner than ~18h; ceiling guards against runaway intervals/overflow.
const MIN_STABILITY_DAYS = (18 / 24) / NEG_LN_080;
const MAX_STABILITY_DAYS = 365;
// Retention constants — loaded at boot from the fitted population_constants.json (a
// MANDATORY asset; main.js aborts if it can't load) via applyRetentionConstants().
// NO baked fallback: the pipeline always ships a valid file (fit_population_constants
// writes population-prior defaults when data is sparse), so the client trusts it.
// `let`, uninitialized until applyRetentionConstants() runs — before any train/schedule.
let KAPPA;        // stability's sensitivity to the cold-solve logit
let GROWTH;       // log stability growth per consecutive success
let Z0;           // reference difficulty: LN_C is the base S at p=0.7 (= logit(0.7))
let LN_C_PRIOR;   // per-user base log-stability prior; LN_C is L2-shrunk to this
// Training-time lnS guard (wider than the scheduling clamp) — prevents exp()
// overflow without flattening the gradient where the model still needs to learn.
const LN_S_TRAIN_MIN = -5;
const LN_S_TRAIN_MAX = 8;

// --- Per-problem 3PL ---------------------------------------------------------
// Cold-solve probability is P = floor + (1−floor)·σ(inner + κ(floor)), the same
// 3PL the rank estimator uses (rank-decomposition.js), anchored so P = 80% when
// the user's rank equals the problem's (CALIBRATION_P). Two per-problem inputs,
// both shipped and stored on each answer-history row: the guessing FLOOR c_p
// (problemFloor) and the discrimination a_p (problemSlope). a_p is used RAW as
// the weight on the rank gap — no rank-unit conversion (M is near-linear by its
// uniform-spacing prior), so the fitted PROBLEM_DIFFICULTY multiplier absorbs
// the rank↔rating unit constant. Legacy rows without a_p/c_p fall back to
// a_p = A_REF and c_p = 0, i.e. the previous 2PL behaviour.
const A_REF = 1.6;   // representative a_p (corpus mean); legacy-row default slope

// Inner logit (linear in params): excludes the per-problem κ(floor) offset and
// the retention-only SUCCESS_SINCE_FAILURE / LN_C. Reads MODEL_INDICES at call
// time (defined below).
function logitZ(features, params) {
    const M = MODEL_INDICES;
    return params[M.INTERCEPT] * features[M.INTERCEPT]
         + params[M.TIME_SOLVING] * features[M.TIME_SOLVING]
         + params[M.REVIEW_TIME] * features[M.REVIEW_TIME]
         + params[M.PROBLEM_DIFFICULTY] * features[M.PROBLEM_DIFFICULTY]
         + params[M.SIZE_IS_13] * features[M.SIZE_IS_13];
}

// Cold-solve probability under the per-problem 3PL.
function solveProb(features, params, floor) {
    const c = clampFloor(floor);
    return c + (1 - c) * sigmoid(logitZ(features, params) + clearKappa(c));
}

// Cold-solve LOG-ODDS — what the retention head consumes as z. Still logit(P),
// now computed through the 3PL, so the KAPPA·(z−Z0) stability calibration keeps
// its meaning (z = log-odds of a cold solve).
function retentionLogit(features, params, floor) {
    const P = Math.max(1e-9, Math.min(1 - 1e-9, solveProb(features, params, floor)));
    return Math.log(P / (1 - P));
}

// Shared feature vector. `isRepeat` toggles the retention-only features
// (SUCCESS_SINCE_FAILURE carries r; LN_C the per-user base). For prediction
// `timePct` defaults to the neutral 0.5.
//
// PROBLEM_DIFFICULTY is the RELATIVE difficulty (userRank − problemRank, both
// normalized 0..1) WEIGHTED by the problem's own 3PL discrimination a_p, so a
// sharper problem separates abilities more steeply. Feeding the user-vs-problem
// delta (not absolute rank) makes "how hard is this *for this user*" explicit
// and robust at cold start instead of leaning on the INTERCEPT to absorb
// userRank. The fitted coefficient is now a dimensionless ~1 multiplier on a_p
// (it also soaks up the rank↔rating unit constant). `userRank` defaults to the
// active shared rank; training rows approximate it with the current rank. When
// userRank is unknown (very early), falls back to absolute. `slope` defaults to
// A_REF for legacy rows that predate per-problem a_p.
function buildFeatures({ isRepeat, timePct = 0.5, reviewPct = 0, problemRank = 0, size, r = 0, userRank, slope }) {
    const f = Array(Object.keys(MODEL_INDICES).length).fill(0);
    const uRank = (typeof userRank === 'number') ? userRank : (typeof state.userRank === 'number' ? state.userRank : 0);
    const aP = (typeof slope === 'number' && slope > 0) ? slope : A_REF;
    f[MODEL_INDICES.INTERCEPT] = 1;
    f[MODEL_INDICES.TIME_SOLVING] = timePct;
    f[MODEL_INDICES.REVIEW_TIME] = reviewPct;
    f[MODEL_INDICES.PROBLEM_DIFFICULTY] = aP * (uRank - (problemRank ?? 0));
    // 9x9 is the baseline; SIZE_IS_13 carries the 13x13 offset. Merged-history
    // entries carry `size`; active-play entries don't, so assume the active size.
    f[MODEL_INDICES.SIZE_IS_13] = (size ?? state.boardSize) === 13 ? 1 : 0;
    if (isRepeat) {
        f[MODEL_INDICES.SUCCESS_SINCE_FAILURE] = r;
        f[MODEL_INDICES.LN_C] = 1;
    }
    return f;
}

// Unclamped log-stability for a repeat feature row: lnS = LN_C + KAPPA·(z−Z0) + GROWTH·r,
// z = the 3PL cold-solve log-odds (carries the problem's floor).
function lnStabilityForRepeat(features, params, floor) {
    const M = MODEL_INDICES;
    return params[M.LN_C]
         + KAPPA * (retentionLogit(features, params, floor) - Z0)
         + GROWTH * features[M.SUCCESS_SINCE_FAILURE];
}

// Clamped stability (days) for scheduling/prediction.
function stabilityDaysForRepeat(features, params, floor) {
    const S = Math.exp(lnStabilityForRepeat(features, params, floor));
    if (!isFinite(S)) return MAX_STABILITY_DAYS;
    return Math.min(MAX_STABILITY_DAYS, Math.max(MIN_STABILITY_DAYS, S));
}

// Window-open delay (hours) for a just-failed (r=0, no prior review) repeat at a
// representative difficulty — the stats panel's "a missed problem comes back in
// ~X h" readout. Returns null when params aren't ready. Uses representative
// per-problem 3PL inputs (corpus-mean a_p, no floor).
export function freshFailureReviewHours(params, { problemRank = 0, size, userRank, slope = A_REF, floor = 0 } = {}) {
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return null;
    const S = stabilityDaysForRepeat(buildFeatures({ isRepeat: true, reviewPct: 0, problemRank, size, r: 0, userRank, slope }), params, floor);
    return NEG_LN_080 * S * 24;
}

// Builds the training feature vector for history row i (board-independent).
function featuresForTrainingRow(td, i) {
    const p = td.history[i];
    return buildFeatures({
        isRepeat: !!p.isRepeat,
        timePct: td.percentileTime[i],
        reviewPct: td.percentileReview[i],
        problemRank: p.problemRank,
        size: p.size,
        r: p.successSinceFailure || 0,
        userRank: state.userRank,   // current rank approximates the rank at attempt time
        slope: p.problemSlope,      // per-problem a_p (3PL discrimination); A_REF if absent
    });
}

// Per-row loss + gradient under the two-head model. EVERY row contributes a
// logistic (cold-solve correctness) term that trains the logit params; REPEAT
// rows additionally contribute a retention term that trains LN_C only (z is read
// but treated as constant — the correctness head owns it). `pLogistic` is
// returned for the Fisher/SE computation. Gradient is the full param vector.
function predictAndGrad(features, params, { isRepeat, elapsedDays, y, floor }) {
    const M = MODEL_INDICES;
    const np = params.length;
    const grad = new Array(np).fill(0);

    // 3PL correctness head (all rows): P = c + (1−c)·σ(inner + κ(c)).
    const c = clampFloor(floor);
    const praw = sigmoid(logitZ(features, params) + clearKappa(c));
    const P = c + (1 - c) * praw;
    const pc = Math.max(1e-15, Math.min(1 - 1e-15, P));
    let loss = -y * Math.log(pc) - (1 - y) * Math.log(1 - pc);
    // dLoss/dinner = (P−y)/(P(1−P)) · dP/dinner, dP/dinner = (1−c)·praw·(1−praw).
    // Reduces to (P−y) when c=0 (the old logistic head).
    const eL = (pc - y) / (pc * (1 - pc)) * (1 - c) * praw * (1 - praw);
    grad[M.INTERCEPT] += eL * features[M.INTERCEPT];
    grad[M.TIME_SOLVING] += eL * features[M.TIME_SOLVING];
    grad[M.REVIEW_TIME] += eL * features[M.REVIEW_TIME];
    grad[M.PROBLEM_DIFFICULTY] += eL * features[M.PROBLEM_DIFFICULTY];
    grad[M.SIZE_IS_13] += eL * features[M.SIZE_IS_13];

    // Retention head (repeat rows): fits LN_C only; z is stop-gradient.
    if (isRepeat) {
        const z = retentionLogit(features, params, c);
        let lnS = params[M.LN_C] + KAPPA * (z - Z0) + GROWTH * features[M.SUCCESS_SINCE_FAILURE];
        lnS = Math.max(LN_S_TRAIN_MIN, Math.min(LN_S_TRAIN_MAX, lnS));
        const S = Math.exp(lnS);
        const u = elapsedDays / S;          // ≥ 0; u=0 ⇒ R=1 (just solved)
        const R = Math.exp(-u);
        const Rc = Math.max(1e-15, Math.min(1 - 1e-15, R));
        loss += -y * Math.log(Rc) - (1 - y) * Math.log(1 - Rc);
        const oneMinusR = Math.max(1 - R, 1e-9);
        grad[M.LN_C] += (R - y) * u / oneMinusR;   // dlnS/dLN_C = 1
    }
    return { loss, grad, pLogistic: P };
}


export function getRecallCandidates(options = {}) {
    const {
        lowerProbabilityBound = 0.60,
        upperProbabilityBound = 0.80,
        requireLastAnswerIncorrect = false,
        // Default to the active size's history (play path: a recall pick can
        // only be re-shown in its own size). The stats panel passes the merged
        // history so its counts span both sizes.
        history = state.answerHistory,
    } = options;

    const now = Date.now();
    // User-controlled pause: while active, no problem can be picked as a
    // recall (and URL-loaded recalls are routed as plain linked solves).
    if (state.recallPausedUntil > now) return [];
    const recallCandidates = [];

    const numParams = Object.keys(MODEL_INDICES).length;
    if (!state.modelParams || state.modelParams.length !== numParams) {
        return []; // Model not ready (or a stale model awaiting retrain)
    }
    const params = state.modelParams;

    const recentHistory = history.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);

    const allReviewTimes = recentHistory
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0);
    allReviewTimes.sort((a, b) => a - b);

    const lastAnswers = {};
    for (const answer of recentHistory) {
        lastAnswers[answer.problemId] = answer;
    }

    const DAY_MS = 24 * 3600 * 1000;
    for (const problemId in lastAnswers) {
        const lastAnswer = lastAnswers[problemId];

        if (lastAnswer.ignored) continue; // Exclude ignored problems
        if (state.unavailableProblemIds.has(problemId)) continue; // Exclude problems that failed to load this session
        if (!lastAnswer.date) continue;

        let condition = lastAnswer.problemRank < state.userRank && lastAnswer.successSinceFailure !== undefined;
        if (requireLastAnswerIncorrect) {
            condition = condition && !lastAnswer.correct;
        }
        if (!condition) continue;

        const reviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
        const reviewPct = calculatePercentile(allReviewTimes, reviewTime);
        const S = stabilityDaysForRepeat(buildFeatures({
            isRepeat: true,
            reviewPct,
            problemRank: lastAnswer.problemRank,
            size: lastAnswer.size,
            r: lastAnswer.successSinceFailure || 0,
            userRank: state.userRank,
            slope: lastAnswer.problemSlope,
        }), params, lastAnswer.problemFloor);

        const tDays = (now - lastAnswer.date) / DAY_MS;
        if (tDays < 0) continue;
        const probability = Math.exp(-tDays / S); // retention R(t); R(0)=1

        if (probability >= lowerProbabilityBound && probability <= upperProbabilityBound) {
            recallCandidates.push({
                problemId,
                recallRate: probability,
            });
        }
    }

    return recallCandidates;
}

// Predicts the unix-ms timestamps at which each candidate problem will cross
// R=0.80 (recall window opens) and R=0.60 (recall window closes), inverting the
// retention curve R(t)=exp(-t/S) analytically: t = -ln(R)·S. Because R(0)=1,
// both crossings are strictly in the future of the last attempt. Mirrors the
// eligibility filters of getRecallCandidates. Reads modelParams/answerHistory/
// userRank directly so callers can request a size other than the active one.
//
// Output rows are bounded:
//   - exit_at must be in the future (we never ping about a window already past)
//   - enter_at must be within a 30-day horizon (avoids accumulating stale
//     schedule rows for problems that won't be due for months; the schedule
//     will get re-uploaded after subsequent attempts anyway).
export function buildRecallSchedule(size) {
    // The recall model and rank are shared (single keys); only the answer
    // history is per-size — a recall reminder must point at a problem in `size`.
    const readSharedKey = (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        try { return JSON.parse(raw); } catch { return null; }
    };
    const readPerSizeKey = (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        try {
            const parsed = JSON.parse(raw);
            return parsed?.[String(size)] ?? null;
        } catch {
            return null;
        }
    };

    const params = readSharedKey('modelParams');
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return [];
    const answerHistory = readPerSizeKey('answerHistory');
    if (!Array.isArray(answerHistory) || answerHistory.length === 0) return [];
    const userRank = readSharedKey('userRank');
    if (typeof userRank !== 'number') return [];

    const recentHistory = answerHistory.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);
    const allReviewTimes = recentHistory
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0);
    allReviewTimes.sort((a, b) => a - b);

    const lastAnswers = {};
    for (const answer of recentHistory) lastAnswers[answer.problemId] = answer;

    const now = Date.now();
    const DAY_MS = 24 * 3600 * 1000;
    const HORIZON_MS = 30 * DAY_MS;
    const result = [];

    for (const problemId in lastAnswers) {
        const lastAnswer = lastAnswers[problemId];
        if (lastAnswer.ignored) continue;
        if (lastAnswer.problemRank === undefined || lastAnswer.problemRank >= userRank) continue;
        if (lastAnswer.successSinceFailure === undefined) continue;
        if (!lastAnswer.date) continue;

        const reviewTime = (lastAnswer.accumulatedReview || 0) + (lastAnswer.reviewTime || 0);
        const reviewPct = calculatePercentile(allReviewTimes, reviewTime);

        // size is the explicit per-size arg here, not the active board.
        const S = stabilityDaysForRepeat(buildFeatures({
            isRepeat: true,
            reviewPct,
            problemRank: lastAnswer.problemRank,
            size,
            r: lastAnswer.successSinceFailure || 0,
            userRank,
            slope: lastAnswer.problemSlope,
        }), params, lastAnswer.problemFloor);
        if (!isFinite(S)) continue;

        const tEnterDays = NEG_LN_080 * S; // R=0.80
        const tExitDays = NEG_LN_060 * S;  // R=0.60

        const enterAt = lastAnswer.date + tEnterDays * DAY_MS;
        const exitAt = lastAnswer.date + tExitDays * DAY_MS;

        if (exitAt <= now) continue;
        if (enterAt > now + HORIZON_MS) continue;

        result.push({
            problemId,
            enterAt: Math.round(enterAt),
            exitAt: Math.round(exitAt),
        });
    }
    return result;
}

// ------------------------------------------------------------------------------------------------
// --- Moved from user-stats.js -------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------

export const MODEL_INDICES = {
    INTERCEPT: 0,
    TIME_SOLVING: 1,
    REVIEW_TIME: 2,
    PROBLEM_DIFFICULTY: 3,
    // 9x9 is the baseline (feature 0); this carries the 13x13 offset at equal
    // internal-rank difficulty. Its coefficient → ranks is the stats panel's
    // "stronger/weaker on 13x13" readout.
    SIZE_IS_13: 4,
    // Feature slot only: holds the success-since-failure count r, consumed by the
    // retention head as the fixed population growth GROWTH·r. It has no fitted
    // coefficient (growth is a population constant, not per-user).
    SUCCESS_SINCE_FAILURE: 5,
    // Per-user base log-stability (S at p=0.7). The only retention parameter that
    // is fit; L2-shrunk to LN_C_PRIOR (see trainUserModel).
    LN_C: 6,
};

const L2_REGULARIZATION_TARGETS = {
    [MODEL_INDICES.TIME_SOLVING]: -0.1,
    [MODEL_INDICES.REVIEW_TIME]: -0.1,
    // Now a dimensionless multiplier on the per-problem a_p·(rank gap): 1.0 = the
    // shipped 3PL discrimination is right as-is for this user. Shrunk to 1 (was a
    // raw in-band slope of -1.5 in the pre-3PL parameterization, before a_p and
    // the κ(floor) anchor carried the difficulty shape).
    [MODEL_INDICES.PROBLEM_DIFFICULTY]: 1.0,
    // Shrink toward 0 = "no skill difference between sizes" until data says so.
    [MODEL_INDICES.SIZE_IS_13]: 0,
    // SUCCESS_SINCE_FAILURE carries no fitted coefficient (growth is constant) —
    // shrink its unused slot to 0.
    [MODEL_INDICES.SUCCESS_SINCE_FAILURE]: 0,
    // Per-user base log-stability, shrunk to the population prior. Set for real by
    // applyRetentionConstants() at boot (LN_C_PRIOR is undefined until then); no
    // training reads this before the mandatory population-constants load completes.
    [MODEL_INDICES.LN_C]: LN_C_PRIOR,
};

// Apply the retention constants from the (MANDATORY) population_constants.json;
// population-constants.js calls this at boot before any training/scheduling, and also
// re-points the LN_C L2 target at the prior. Throws on a missing/invalid field so a
// corrupt blob is fatal (the caller aborts boot) rather than silently leaving the
// constants unset — there is no baked default to fall back to.
export function applyRetentionConstants({ kappa, z0, growth, ln_c_prior } = {}) {
    if (!(Number.isFinite(kappa) && kappa > 0 && kappa < 20)) throw new Error(`population constants: bad kappa ${kappa}`);
    if (!Number.isFinite(z0)) throw new Error(`population constants: bad z0 ${z0}`);
    if (!(Number.isFinite(growth) && growth > -2 && growth < 4)) throw new Error(`population constants: bad growth ${growth}`);
    if (!Number.isFinite(ln_c_prior)) throw new Error(`population constants: bad ln_c_prior ${ln_c_prior}`);
    KAPPA = kappa;
    Z0 = z0;
    GROWTH = growth;
    LN_C_PRIOR = ln_c_prior;
    L2_REGULARIZATION_TARGETS[MODEL_INDICES.LN_C] = ln_c_prior;
}

// Population prior on logit-per-rank: a 1-rank skill gap shifts the log-odds
// of a correct answer by roughly this much. Treated as a Gaussian prior whose
// posterior is updated by the user's own fitted PROBLEM_DIFFICULTY slope.
const LOGIT_PER_RANK_PRIOR = 0.28;
const LOGIT_PER_RANK_PRIOR_SIGMA = 0.1;

// Precision-weighted blend of the prior and the user's own fitted slope from
// `state.statsResults`. Falls back to the prior when no usable model fit is
// available. Floored to keep downstream divisions well-conditioned.
export function getLogitPerRank(statsResults) {
    const priorPrecision = 1 / (LOGIT_PER_RANK_PRIOR_SIGMA * LOGIT_PER_RANK_PRIOR_SIGMA);
    const pd = statsResults && statsResults.results
        ? statsResults.results.find(r => r.varIndex === MODEL_INDICES.PROBLEM_DIFFICULTY)
        : null;
    if (!pd || !isFinite(pd.coeff) || !isFinite(pd.se) || pd.se <= 0) {
        return LOGIT_PER_RANK_PRIOR;
    }
    const userSlope = Math.abs(pd.coeff);
    const userPrecision = 1 / (pd.se * pd.se);
    const blended = (priorPrecision * LOGIT_PER_RANK_PRIOR + userPrecision * userSlope)
        / (priorPrecision + userPrecision);
    return Math.max(0.05, blended);
}

let isCalculatingStats = false;

// NOTE: the one-shot wipe of pre-3PL persisted params lives in migrations.js
// (migrateModelArch3pl), which runMigrations() runs at boot before any training.

export async function calculateAndStoreStatsResults() {
    if (isCalculatingStats) return;
    isCalculatingStats = true;

    try {
        // Shared recall model: trained on the union of both sizes' histories.
        // Every feature it reads (problemRank, time/review percentiles, the
        // recall-formula term) is board-independent, so pooling is on one scale.
        const validHistory = getMergedAnswerHistory().filter(p => p.problemRank !== undefined && p.timeTaken !== undefined && p.date !== undefined);
        if (validHistory.length < 10) {
            setStatsResults(null);
            return;
        }
        const history = validHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);

        const trainingData = prepareTrainingData(history, validHistory);

        const numParams = Object.keys(MODEL_INDICES).length;
        let params;
        if (state.modelParams && state.modelParams.length === numParams) {
            params = [...state.modelParams];
        } else {
            // Fresh start: seed the base-stability level at the population prior
            // so first repeats schedule sanely (~25h) while LN_C is untrained.
            params = Array(numParams).fill(0);
            params[MODEL_INDICES.LN_C] = LN_C_PRIOR;
        }

        let adamState = (state.adamState && state.adamState.m.length === numParams)
            ? JSON.parse(JSON.stringify(state.adamState))
            : { m: Array(numParams).fill(0), v: Array(numParams).fill(0), t: 0 };

        const { trainedParams, finalAdamState } = await trainUserModel(trainingData, { params, adamState });

        // Sanity check for non-numeric values
        const areParamsValid = trainedParams.every(p => isFinite(p));

        if (areParamsValid) {
            const finalResults = calculateResultsAndStdErrors(trainedParams, trainingData);
            setModelParams(trainedParams);
            setAdamState(finalAdamState);
            setStatsResults(finalResults);
        } else {
            console.error("Model training resulted in non-numeric weights. Wiping model state.");
            Sentry.captureMessage("Model training resulted in non-numeric weights.");
            setModelParams(null);
            setAdamState(null);
            setStatsResults(null);
        }

    } catch (error) {
        console.error('Unified model training failed:', error);
        Sentry.captureException(error);
    } finally {
        isCalculatingStats = false;
    }
}

export function prepareTrainingData(history, validHistory) {
    const n = history.length;
    const p_low = 0.45, p_high = 0.55;

    const timeTakenValues = history.map(p => p.timeTaken);
    const sortedTime = [...timeTakenValues].sort((a, b) => a - b);
    const time_per_percentile_min = (sortedTime[Math.ceil(p_high * (n - 1))] - sortedTime[Math.floor(p_low * (n - 1))]) / (60000 * (p_high - p_low));
    
    const percentileTime = timeTakenValues.map(t => calculatePercentile(sortedTime, t));

    const previousReviewTimes = history.map(p => {
        if (p.isRepeat) {
            return p.accumulatedReview || 0;
        }
        return 0;
    });
    const reviewTimes = previousReviewTimes.filter(t => t > 0);
    const sortedReview = [...reviewTimes].sort((a, b) => a - b);
    let time_per_percentile_min_r = 0;
    if (reviewTimes.length > 0) {
        const idx_low_r = Math.floor(p_low * (sortedReview.length - 1));
        const idx_high_r = Math.ceil(p_high * (sortedReview.length - 1));
        time_per_percentile_min_r = (sortedReview[idx_high_r] - sortedReview[idx_low_r]) / (60000 * (p_high - p_low));
    }
    
    const percentileReview = previousReviewTimes.map(t => calculatePercentile(sortedReview, t));

    // Elapsed days since each repeat row's previous attempt of the same problem
    // (the retention head's t). Zero for novel rows (they use the logistic head).
    const DAY = 1000 * 60 * 60 * 24;
    const elapsedDays = history.map(p => {
        if (!p.isRepeat) return 0;
        const prev = validHistory.filter(a => a.problemId === p.problemId && a.date < p.date).pop();
        if (!prev) return 0;
        return Math.max(0, (p.date - prev.date) / DAY);
    });
    const isRepeatArr = history.map(p => !!p.isRepeat);

    return {
        y: history.map(p => p.correct ? 1 : 0),
        weights: Array.from({ length: n }, (_, i) => (i + 1) / n),
        percentileTime,
        percentileReview,
        // Per-row 3PL guessing floor (problemFloor); 0 for legacy rows → 2PL.
        floors: history.map(p => clampFloor(p.problemFloor)),
        elapsedDays,
        isRepeatArr,
        time_per_percentile_min,
        time_per_percentile_min_r,
        history,
        validHistory,
        n
    };
}

async function trainUserModel(trainingData, modelState) {
    const { y, weights, elapsedDays, isRepeatArr, n, floors } = trainingData;
    let { params, adamState } = modelState;

    const learningRate = 0.02;
    const maxEpochs = 10000, maxTrainingTimeMs = 2000, lossConvergenceThreshold = 1e-6;
    const lambda = 10; // L2 regularization parameter

    let prevLoss = Infinity;
    const startTime = Date.now();

    // Features are independent of params, so build the design matrix once.
    const numFeatures = Object.keys(MODEL_INDICES).length;
    const X = Array.from({ length: n }, (_, i) => featuresForTrainingRow(trainingData, i));

    for (let epoch = 0; epoch < maxEpochs; epoch++) {
        adamState.t++;

        const gradient = Array(params.length).fill(0);
        let currentLoss = 0;

        for (let i = 0; i < n; i++) {
            const { loss, grad } = predictAndGrad(X[i], params, { isRepeat: isRepeatArr[i], elapsedDays: elapsedDays[i], y: y[i], floor: floors[i] });
            currentLoss += weights[i] * loss;
            for (let j = 0; j < numFeatures; j++) gradient[j] += weights[i] * grad[j];
        }

        // INTERCEPT (logit level) is the only unregularized parameter; LN_C is
        // shrunk to its population prior (LN_C_PRIOR) via L2 like the others.
        const isUnregularized = (j) => j === MODEL_INDICES.INTERCEPT;

        let l2_penalty = 0;
        for (let j = 0; j < params.length; j++) {
            if (isUnregularized(j)) continue;
            const target = L2_REGULARIZATION_TARGETS[j] || 0;
            l2_penalty += Math.pow(params[j] - target, 2);
        }
        currentLoss += (lambda / 2) * l2_penalty;
        currentLoss /= n;

        // Finalize the gradient (mean-reduce + L2 toward each param's target),
        // then apply the shared Adam moment update.
        for (let j = 0; j < params.length; j++) {
            gradient[j] /= n;
            if (!isUnregularized(j)) {
                const target = L2_REGULARIZATION_TARGETS[j] || 0;
                gradient[j] += (lambda / n) * (params[j] - target);
            }
        }
        adamUpdate(params, gradient, adamState, learningRate);

        if (Math.abs(prevLoss - currentLoss) < lossConvergenceThreshold || Date.now() - startTime > maxTrainingTimeMs) break;
        prevLoss = currentLoss;
    }
    return { trainedParams: params, finalAdamState: adamState };
}

export function calculateResultsAndStdErrors(finalParams, trainingData) {
    const { elapsedDays, isRepeatArr, weights, y, n, floors, time_per_percentile_min, time_per_percentile_min_r } = trainingData;

    const M = MODEL_INDICES;
    const np = Object.keys(M).length;
    const X = Array.from({ length: n }, (_, i) => featuresForTrainingRow(trainingData, i));

    // SE for the cold-solve coefficients (the only results the stats panel reads)
    // come from the 3PL Fisher info: I = Σ wᵢ·vᵢ·xᵢ·xᵢᵀ over the LOGIT features
    // (which are linear in the inner logit), with vᵢ = (dP/dinner)²/(P(1−P)) the
    // per-row 3PL information — at floor c=0 this reduces to p(1−p). The retention
    // features (SUCCESS_SINCE_FAILURE / LN_C) don't enter, so their columns are
    // zeroed; the +1e-6 ridge keeps the matrix invertible (large unused SE there).
    const logitIdx = [M.INTERCEPT, M.TIME_SOLVING, M.REVIEW_TIME, M.PROBLEM_DIFFICULTY, M.SIZE_IS_13];
    const D = X.map(xi => {
        const row = new Array(np).fill(0);
        for (const j of logitIdx) row[j] = xi[j];
        return row;
    });
    const W = math.diag(X.map((xi, i) => {
        const c = clampFloor(floors[i]);
        const praw = sigmoid(logitZ(xi, finalParams) + clearKappa(c));
        const P = Math.max(1e-9, Math.min(1 - 1e-9, c + (1 - c) * praw));
        const dP = (1 - c) * praw * (1 - praw);   // dP/dinner
        return weights[i] * dP * dP / (P * (1 - P));
    }));
    const Dm = math.matrix(D);
    const fisherInfo = math.add(math.multiply(math.transpose(Dm), W, Dm), math.diag(Array(np).fill(1e-6)));
    const varCovarMatrix = math.inv(fisherInfo);
    const stdErrors = math.map(math.diag(varCovarMatrix), math.sqrt).toArray();

    const results = [];
    for (let i = 1; i < Object.keys(MODEL_INDICES).length; i++) {
        let coeff = finalParams[i], se = stdErrors[i], scaleFactor = 1;
        if (i === MODEL_INDICES.TIME_SOLVING && time_per_percentile_min > 0) scaleFactor = 1 / time_per_percentile_min;
        else if (i === MODEL_INDICES.REVIEW_TIME && time_per_percentile_min_r > 0) scaleFactor = 1 / time_per_percentile_min_r;
        // PROBLEM_DIFFICULTY is now a multiplier on a_p·gap, so the effective
        // logit-per-rank is coeff·a_p. Fold a representative a_p (A_REF) into the
        // reported coeff so getLogitPerRank / the size-comparison readout keep
        // the same "logit per rank index" scale as the pre-3PL model (was 1/29).
        else if (i === MODEL_INDICES.PROBLEM_DIFFICULTY) scaleFactor = A_REF / 29;
        results.push({ coeff: coeff * scaleFactor, se: se * Math.abs(scaleFactor), varIndex: i, scaleFactor });
    }
    return { results, varCovarMatrix };
}


