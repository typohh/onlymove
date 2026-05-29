// js/model.js

import * as Sentry from "@sentry/browser";
import { state, setStatsResults, setModelParams, setAdamState, getMergedAnswerHistory } from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH, RECALL_CANDIDATE_HISTORY_LENGTH } from './constants.js';

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
// Recall is an FSRS-style retention curve anchored at R(0)=1, NOT a logit on
// population correctness (the old additive `base + βR·log(...)` form let a
// problem's difficulty offset the curve downward so it could be "due" the
// instant it was solved). A REPEAT encounter's memory stability S (days) is:
//
//   z   = INTERCEPT + TIME_SOLVING·timePct + REVIEW_TIME·reviewPct
//         + PROBLEM_DIFFICULTY·problemRank + SIZE_IS_13·size13   (the cold-solve logit)
//   S   = c · σ(z) · exp(SUCCESS_SINCE_FAILURE·r),   c = exp(LN_C)
//   R(t)= exp(-t_days / S)
//
// so R(0)=1 always (no instant-due) and difficulty scales the time-constant
// multiplicatively, exactly the FSRS picture. Crucially S uses σ(z) (BOUNDED in
// (0,1)), NOT raw e^z: cold-solve difficulty must only mildly move stability
// (σ spans ~[0.6,0.95] over eligible sub-rank problems, ≈1.5×), because a
// problem you find a little harder to *solve cold* is not forgotten 10-20×
// faster. Using e^z over-disperses S so the lower-z problems collapse to the
// floor and cycle every ~18h while you still recall them perfectly — the bug
// this form fixes. Geometric review growth comes from r (S ×e^{β_r} per
// consecutive success, unbounded), a lapse resets r→0 (post-lapse reset), and
// `c` (LN_C) is the free, data-driven base-stability level — it is NOT L2-
// regularized (see trainUserModel) and is seeded to a sane base for cold start,
// so it can't be pinned down to the floor by a prior.
//
// The SAME logit z feeds the NOVEL-encounter cold-solve head P = σ(z); that head
// trains PROBLEM_DIFFICULTY / TIME_SOLVING / SIZE_IS_13 and backs the stats
// readouts. REVIEW_TIME / SUCCESS_SINCE_FAILURE / LN_C are identified purely
// from repeat rows. There is no IS_NOVEL indicator: the regime is carried by
// which head a row enters (a 1-on-every-novel-row indicator is collinear with
// the intercept).
const NEG_LN_080 = -Math.log(0.80); // 0.22314 — t/S at the 0.80 window-open
const NEG_LN_060 = -Math.log(0.60); // 0.51083 — t/S at the 0.60 window-close
// Stability floor/ceiling (days). The floor keeps a just-failed problem from
// coming due sooner than ~18h (NEG_LN_080·MIN ≈ 18h window-open); the ceiling
// guards against runaway intervals/overflow.
const MIN_STABILITY_DAYS = (18 / 24) / NEG_LN_080;
const MAX_STABILITY_DAYS = 365;
// Cold-start base stability level: c = e^{LN_C}. Seeded so a fresh (untrained)
// model schedules first repeats at a sane ~2-3 days rather than the floor; the
// retention data then moves it per-user (LN_C is unregularized).
const LN_C_INIT = Math.log(10);
// Training-time lnS guard (wider than the scheduling clamp) — prevents exp()
// overflow without flattening the gradient where the model still needs to learn.
const LN_S_TRAIN_MIN = -5;
const LN_S_TRAIN_MAX = 8;

function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

// Shared cold-solve logit z over the logit features only (excludes the
// retention-only LN_C / SUCCESS_SINCE_FAILURE). Reads MODEL_INDICES at call time
// (it is defined later in the module).
function logitZ(features, params) {
    const M = MODEL_INDICES;
    return params[M.INTERCEPT] * features[M.INTERCEPT]
         + params[M.TIME_SOLVING] * features[M.TIME_SOLVING]
         + params[M.REVIEW_TIME] * features[M.REVIEW_TIME]
         + params[M.PROBLEM_DIFFICULTY] * features[M.PROBLEM_DIFFICULTY]
         + params[M.SIZE_IS_13] * features[M.SIZE_IS_13];
}

// Shared feature vector. `isRepeat` toggles the retention-only features
// (LN_C, SUCCESS_SINCE_FAILURE carry signal only on repeats); for prediction
// `timePct` defaults to the neutral 0.5.
function buildFeatures({ isRepeat, timePct = 0.5, reviewPct = 0, problemRank = 0, size, r = 0 }) {
    const f = Array(Object.keys(MODEL_INDICES).length).fill(0);
    f[MODEL_INDICES.INTERCEPT] = 1;
    f[MODEL_INDICES.TIME_SOLVING] = timePct;
    f[MODEL_INDICES.REVIEW_TIME] = reviewPct;
    f[MODEL_INDICES.PROBLEM_DIFFICULTY] = problemRank ?? 0;
    // 9x9 is the baseline; SIZE_IS_13 carries the 13x13 offset. Merged-history
    // entries carry `size`; active-play entries don't, so assume the active size.
    f[MODEL_INDICES.SIZE_IS_13] = (size ?? state.boardSize) === 13 ? 1 : 0;
    if (isRepeat) {
        f[MODEL_INDICES.SUCCESS_SINCE_FAILURE] = r;
        f[MODEL_INDICES.LN_C] = 1;
    }
    return f;
}

// Unclamped log-stability for a repeat feature row: lnS = LN_C + ln σ(z) + β_r·r.
function lnStabilityForRepeat(features, params) {
    const M = MODEL_INDICES;
    const p = Math.max(1e-6, Math.min(1 - 1e-6, sigmoid(logitZ(features, params))));
    return params[M.LN_C] * features[M.LN_C]
         + Math.log(p)
         + params[M.SUCCESS_SINCE_FAILURE] * features[M.SUCCESS_SINCE_FAILURE];
}

// Clamped stability (days) for scheduling/prediction.
function stabilityDaysForRepeat(features, params) {
    const S = Math.exp(lnStabilityForRepeat(features, params));
    if (!isFinite(S)) return MAX_STABILITY_DAYS;
    return Math.min(MAX_STABILITY_DAYS, Math.max(MIN_STABILITY_DAYS, S));
}

// Window-open delay (hours) for a just-failed (r=0, no prior review) repeat at a
// representative difficulty — the stats panel's "a missed problem comes back in
// ~X h" readout. Returns null when params aren't ready.
export function freshFailureReviewHours(params, { problemRank = 0, size } = {}) {
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return null;
    const S = stabilityDaysForRepeat(buildFeatures({ isRepeat: true, reviewPct: 0, problemRank, size, r: 0 }), params);
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
    });
}

// Per-row likelihood pieces under the dual-head model. Returns the predicted
// probability, the loss, the scalar dLoss/dη (η = z for novel rows, lnS for
// repeat rows), the score-direction vector v = dη/dparams, and the Bernoulli
// Fisher weight. The training gradient for a row is dScalar·v and its Fisher
// contribution is fisherW·v·vᵀ — one shared v for both.
function predictAndGrad(features, params, { isRepeat, elapsedDays, y }) {
    const M = MODEL_INDICES;
    const n = features.length;
    if (!isRepeat) {
        const p = sigmoid(logitZ(features, params));
        const clipped = Math.max(1e-15, Math.min(1 - 1e-15, p));
        const loss = -y * Math.log(clipped) - (1 - y) * Math.log(1 - clipped);
        // η = z, dη/dparams = features (retention-only entries are 0 here).
        return { p, loss, dScalar: p - y, v: features, fisherW: p * (1 - p) };
    }
    const p = Math.max(1e-6, Math.min(1 - 1e-6, sigmoid(logitZ(features, params))));
    let lnS = params[M.LN_C] * features[M.LN_C]
            + Math.log(p)
            + params[M.SUCCESS_SINCE_FAILURE] * features[M.SUCCESS_SINCE_FAILURE];
    lnS = Math.max(LN_S_TRAIN_MIN, Math.min(LN_S_TRAIN_MAX, lnS));
    const S = Math.exp(lnS);
    const u = elapsedDays / S;              // ≥ 0; u=0 ⇒ R=1 (just solved)
    const R = Math.exp(-u);
    const clipped = Math.max(1e-15, Math.min(1 - 1e-15, R));
    const loss = -y * Math.log(clipped) - (1 - y) * Math.log(1 - clipped);
    // dR/dlnS = R·u ⇒ dLoss/dlnS = (R-y)·u/(1-R); Fisher = (R·u)²/(R(1-R)).
    const oneMinusR = Math.max(1 - R, 1e-9);
    // v = dlnS/dparams: logit features chain through d(ln σ(z))/dz = (1-p);
    // LN_C and SUCCESS_SINCE_FAILURE enter lnS linearly via their features.
    const oneMinusP = 1 - p;
    const v = new Array(n).fill(0);
    v[M.INTERCEPT] = oneMinusP * features[M.INTERCEPT];
    v[M.TIME_SOLVING] = oneMinusP * features[M.TIME_SOLVING];
    v[M.REVIEW_TIME] = oneMinusP * features[M.REVIEW_TIME];
    v[M.PROBLEM_DIFFICULTY] = oneMinusP * features[M.PROBLEM_DIFFICULTY];
    v[M.SIZE_IS_13] = oneMinusP * features[M.SIZE_IS_13];
    v[M.SUCCESS_SINCE_FAILURE] = features[M.SUCCESS_SINCE_FAILURE];
    v[M.LN_C] = features[M.LN_C];
    return { p: R, loss, dScalar: (R - y) * u / oneMinusR, v, fisherW: R * u * u / oneMinusR };
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
        }), params);

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
        }), params);
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
    // Retention-only (zero on novel rows). Each consecutive success adds this to
    // lnS, i.e. multiplies memory stability by e^{coeff} (FSRS geometric SInc).
    SUCCESS_SINCE_FAILURE: 5,
    // Retention-only, unregularized: ln of the base-stability scale c, so
    // S = c·σ(z)·e^{SUCCESS_SINCE_FAILURE·r}. Only the repeat head sees it.
    LN_C: 6,
};

const L2_REGULARIZATION_TARGETS = {
    [MODEL_INDICES.TIME_SOLVING]: -0.1,
    [MODEL_INDICES.REVIEW_TIME]: -0.1,
    [MODEL_INDICES.PROBLEM_DIFFICULTY]: -10,
    // Shrink toward 0 = "no skill difference between sizes" until data says so.
    [MODEL_INDICES.SIZE_IS_13]: 0,
    // Prior: each consecutive success ~doubles stability (ln 2 ≈ 0.69).
    [MODEL_INDICES.SUCCESS_SINCE_FAILURE]: Math.log(2),
    // LN_C is intentionally absent — it is unregularized (see trainUserModel) so
    // the base-stability level tracks the user's true retention.
};

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
            // Fresh start: seed the base-stability level so first repeats schedule
            // at a sane interval instead of the floor while LN_C is still untrained.
            params = Array(numParams).fill(0);
            params[MODEL_INDICES.LN_C] = LN_C_INIT;
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
    const { y, weights, elapsedDays, isRepeatArr, n } = trainingData;
    let { params, adamState } = modelState;

    const learningRate = 0.02, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
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
            const { loss, dScalar, v } = predictAndGrad(X[i], params, { isRepeat: isRepeatArr[i], elapsedDays: elapsedDays[i], y: y[i] });
            currentLoss += weights[i] * loss;
            const error = weights[i] * dScalar;
            for (let j = 0; j < numFeatures; j++) gradient[j] += error * v[j];
        }

        // INTERCEPT (logit level) and LN_C (free base-stability level) are not
        // regularized — LN_C must be free to track the user's true retention,
        // not pinned to a prior that would collapse S to the floor.
        const isUnregularized = (j) => j === MODEL_INDICES.INTERCEPT || j === MODEL_INDICES.LN_C;

        let l2_penalty = 0;
        for (let j = 0; j < params.length; j++) {
            if (isUnregularized(j)) continue;
            const target = L2_REGULARIZATION_TARGETS[j] || 0;
            l2_penalty += Math.pow(params[j] - target, 2);
        }
        currentLoss += (lambda / 2) * l2_penalty;
        currentLoss /= n;

        for (let j = 0; j < params.length; j++) {
            gradient[j] /= n;
            if (!isUnregularized(j)) {
                const target = L2_REGULARIZATION_TARGETS[j] || 0;
                gradient[j] += (lambda / n) * (params[j] - target);
            }
            adamState.m[j] = beta1 * adamState.m[j] + (1 - beta1) * gradient[j];
            adamState.v[j] = beta2 * adamState.v[j] + (1 - beta2) * Math.pow(gradient[j], 2);
            const m_hat = adamState.m[j] / (1 - Math.pow(beta1, adamState.t));
            const v_hat = adamState.v[j] / (1 - Math.pow(beta2, adamState.t));
            params[j] -= learningRate * m_hat / (Math.sqrt(v_hat) + epsilon);
        }

        if (Math.abs(prevLoss - currentLoss) < lossConvergenceThreshold || Date.now() - startTime > maxTrainingTimeMs) break;
        prevLoss = currentLoss;
    }
    return { trainedParams: params, finalAdamState: adamState };
}

export function calculateResultsAndStdErrors(finalParams, trainingData) {
    const { elapsedDays, isRepeatArr, weights, y, n, time_per_percentile_min, time_per_percentile_min_r } = trainingData;

    const X = Array.from({ length: n }, (_, i) => featuresForTrainingRow(trainingData, i));

    // Fisher info I = Σ wᵢ·fisherWᵢ·vᵢ·vᵢᵀ, where vᵢ is the per-row score
    // direction dη/dparams (= features for logistic novel rows, but chained
    // through (1-p) on the logit features for retention repeat rows) and
    // fisherW is the Bernoulli weight (p(1-p) novel, R·u²/(1-R) repeat).
    const V = [], fisherWeights = [];
    for (let i = 0; i < n; i++) {
        const { v, fisherW } = predictAndGrad(X[i], finalParams, { isRepeat: isRepeatArr[i], elapsedDays: elapsedDays[i], y: y[i] });
        V.push(Array.from(v));
        fisherWeights.push(weights[i] * fisherW);
    }
    const V_matrix = math.matrix(V);
    const W = math.diag(fisherWeights);
    const Vt = math.transpose(V_matrix);
    const fisherInfo = math.add(math.multiply(Vt, W, V_matrix), math.diag(Array(V[0].length).fill(1e-6)));
    const varCovarMatrix = math.inv(fisherInfo);
    const stdErrors = math.map(math.diag(varCovarMatrix), math.sqrt).toArray();

    const results = [];
    for (let i = 1; i < Object.keys(MODEL_INDICES).length; i++) {
        let coeff = finalParams[i], se = stdErrors[i], scaleFactor = 1;
        if (i === MODEL_INDICES.TIME_SOLVING && time_per_percentile_min > 0) scaleFactor = 1 / time_per_percentile_min;
        else if (i === MODEL_INDICES.REVIEW_TIME && time_per_percentile_min_r > 0) scaleFactor = 1 / time_per_percentile_min_r;
        else if (i === MODEL_INDICES.PROBLEM_DIFFICULTY) scaleFactor = 1 / 29;
        results.push({ coeff: coeff * scaleFactor, se: se * Math.abs(scaleFactor), varIndex: i, scaleFactor });
    }
    return { results, varCovarMatrix };
}


