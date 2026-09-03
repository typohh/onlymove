// js/recall-model.js
//
// The spaced-repetition (recall) model: the retention formula, its per-user fit,
// the due-candidate search and the serve-share gate. Named `model.js` until
// 2026-08-03, which said nothing — every sibling model here is named for what it
// models (weakness.js, rank-decomposition.js).
//
// calculateAndStoreStatsResults is this module's TRAINER, not a stats helper: it
// drives prepareTrainingData / MODEL_INDICES / fitUserModel's two Newton blocks
// and stores the fit. It reads "stats" because the output lands in
// state.statsResults.

import { captureException, captureMessage } from "@sentry/browser";
import { state, setStatsResults, setModelParams, getMergedAnswerHistory, getAnswerHistoryForSize } from './state.js';
// The population's central rank — the stand-in for a GENERIC problem's difficulty when
// a caller has none to give. A different job from the state slots' cold start (which is
// about an unknown USER), so it reads the prior's mean directly rather than sharing it.
import { PRIOR_MEAN } from '../shared/rank-window.js';
import { MODEL_INPUT_HISTORY_LENGTH, RECALL_CANDIDATE_HISTORY_LENGTH } from './constants.js';
import { tsumegoIdOf } from './sgf.js';
import { sigmoid, normCdf, invertMatrix } from './math-utils.js';
import { clearKappa, clampFloor, clearSpan } from './irt.js';
import { ratingPerRank } from './rank-decomposition.js';
import { newtonBlockStep } from './optimizer.js';

function findFirst(sortedArr, target) {
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

function findLast(sortedArr, target) {
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
//   lnS  = LN_C + KAPPA·(z − Z0) + GROWTH·r + FIRST_SUCCESS_STEP·[r ≥ 1]
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
// FIRST_SUCCESS_STEP is a POST-LAPSE term, not another rung of the ladder. r is
// 0 exactly when the previous attempt FAILED (that is how a problem joins the
// pool) and 1 when it was the first success after a failure, so r=0 and r≥1 are
// the post-lapse and post-success states. Established schedulers give a lapse
// its own formula; this one folds it into LN_C, and without the step every rung
// above 0 is served before the forgetting the window waits for has happened —
// measured on the attempt log, they clear AT or ABOVE the 0.80 the window opens
// at, while rung 0 lands correctly near the window's middle. The carrier is
// BINARY (did the last attempt succeed), not the count: the second success adds
// nothing distinguishable from zero, so this is one indicator and not a second
// growth rate. It is a POPULATION constant — the within-user contrast is close
// to identical across users (see pipeline.fit_population_constants), and 42% of
// users have no r≥1 rows at all inside this model's training window, so a
// per-user version would be its own prior for most of them.
//
// The same logit z is the NOVEL cold-solve head P=σ(z); that head is trained on
// correctness over ALL rows (so REVIEW_TIME, which is 0 on novel rows, is still
// identified from repeats) and backs the stats readouts. The retention head
// reads z but does NOT backprop into the logit params (z is owned by the
// correctness head); it only fits LN_C. No IS_NOVEL indicator.
const NEG_LN_080 = -Math.log(0.80); // 0.22314 — t/S at the 0.80 window-open
const NEG_LN_060 = -Math.log(0.60); // 0.51083 — t/S at the 0.60 window-close
export const MIN_RECALL_HOURS = 6;
// Stability floor (days). Floor keeps a just-failed problem from coming due
// sooner than the minimum recall hours. There is deliberately NO ceiling: the
// streak's exponential stability growth is what graduates a learned problem —
// its due date eventually falls beyond the RECALL_CANDIDATE_HISTORY_LENGTH
// working window and it leaves the recall pool. (A 365-day ceiling here used to
// turn that graduation into a perpetual ~81-day recycle, so the pool could only
// grow; overflow needs no ceiling — lnS is a small linear form, and the
// !isFinite fallback below covers degenerate params.)
const MIN_STABILITY_DAYS = (MIN_RECALL_HOURS / 24) / NEG_LN_080;
// The floor expressed as the earliest a repeat can come due (ms) — the minimum
// recall interval. Exported for the cross-tab duplicate guard (attempt-recorder.js):
// a second attempt closer than this can never be a legitimate recall.
export const MIN_RECALL_INTERVAL_MS = MIN_RECALL_HOURS * 3600 * 1000;
// Retention constants — loaded at boot from the fitted population_constants.json (a
// MANDATORY asset; main.js aborts if it can't load) via applyRetentionConstants().
// NO baked fallback: the pipeline always ships a valid file (fit_population_constants
// writes population-prior defaults when data is sparse), so the client trusts it.
// `let`, uninitialized until applyRetentionConstants() runs — before any train/schedule.
let KAPPA;        // stability's sensitivity to the cold-solve logit
let GROWTH;       // log stability growth per consecutive success
let FIRST_SUCCESS_STEP;  // extra log stability once the previous attempt was a success
let Z0;           // reference difficulty: LN_C is the base S at p=0.7 (= logit(0.7))
let LN_C_PRIOR;   // per-user base log-stability prior; LN_C is L2-shrunk to this
// Population-mean time/review benefit confound (logit per MINUTE), from
// population_constants.json via setTimeBias(). Two roles, both load-bearing and
// they MUST agree: (1) it is the prior the per-user TIME_SOLVING/REVIEW_TIME
// coefficients are L2-shrunk toward (converted to per-percentile per training
// run, see fitUserModel), and (2) the displayed/gated "% chance more time
// helps" recenters the fitted coefficient by it (benefitConfidence). If the
// shrinkage prior and the recentering offset disagree, an uninformative user
// can never read 50% — anchoring both here keeps them consistent.
let TIME_BIAS_PER_MIN;
let REVIEW_BIAS_PER_MIN;
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
// a_p = A_REF and c_p = 0, i.e. plain 2PL behaviour.
const A_REF = 1.6;   // representative a_p (corpus mean); legacy-row default slope

// Neutral value of the PROBLEM_DIFFICULTY multiplier — the coefficient at which
// coeff·a_p·(rank gap) equals the 2PL's a_p·(gap in raw-rating units): the rating
// units spanned by ONE rank, read off the shipped curve. The L2 prior shrinks
// toward THIS, not 1.0: a_p is per rating unit while the gap is in rank units, so
// a target of 1.0 is unit-inconsistent by exactly that factor — and since the
// prior dominates the under-a-nat of Fisher information even a full 500-attempt
// history carries on this coefficient, an inconsistent target would pin the fitted
// difficulty slope far off the logit-per-rank everything else is calibrated to (and
// a retention z far from the rating-space z KAPPA was fit against in
// pipeline.fit_population_constants).
const DIFFICULTY_NEUTRAL = ratingPerRank();

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

// Cold-solve probability under the per-problem 4PL (upper asymptote 1 − LAPSE).
function solveProb(features, params, floor) {
    const c = clampFloor(floor);
    return c + clearSpan(c) * sigmoid(logitZ(features, params) + clearKappa(c));
}

// Cold-solve LOG-ODDS — what the retention head consumes as z. logit(P)
// computed through the 3PL, so the KAPPA·(z−Z0) stability calibration keeps
// its meaning (z = log-odds of a cold solve).
function retentionLogit(features, params, floor) {
    const P = Math.max(1e-9, Math.min(1 - 1e-9, solveProb(features, params, floor)));
    return Math.log(P / (1 - P));
}

// Shared feature vector. `isRepeat` toggles the retention-only features
// (SUCCESS_SINCE_FAILURE carries r; LN_C the per-user base). For prediction
// `timePct` defaults to the neutral 0.5.
//
// PROBLEM_DIFFICULTY is the RELATIVE difficulty (userRank − problemRank, both in
// rank units) WEIGHTED by the problem's own 3PL discrimination a_p, so a
// sharper problem separates abilities more steeply. Feeding the user-vs-problem
// delta (not absolute rank) makes "how hard is this *for this user*" explicit
// and robust at cold start instead of leaning on the INTERCEPT to absorb
// userRank. The fitted coefficient is a multiplier on a_p whose neutral value
// is DIFFICULTY_NEUTRAL ≈ 5.4 — the rank↔rating unit constant, since the gap is
// in rank units while a_p is per rating unit. `userRank` defaults to the
// active shared rank; training rows approximate it with the current rank. When
// userRank is unknown (very early), falls back to absolute. `slope` defaults to
// A_REF for legacy rows that predate per-problem a_p.
function buildFeatures({ isRepeat, timePct = 0.5, reviewPct = 0, problemRank = PRIOR_MEAN, size, r = 0, userRank, slope }) {
    const f = Array(Object.keys(MODEL_INDICES).length).fill(0);
    const uRank = (typeof userRank === 'number') ? userRank : (typeof state.userRank === 'number' ? state.userRank : PRIOR_MEAN);
    const aP = (typeof slope === 'number' && slope > 0) ? slope : A_REF;
    f[MODEL_INDICES.INTERCEPT] = 1;
    f[MODEL_INDICES.TIME_SOLVING] = timePct;
    f[MODEL_INDICES.REVIEW_TIME] = reviewPct;
    f[MODEL_INDICES.PROBLEM_DIFFICULTY] = aP * (uRank - (problemRank ?? PRIOR_MEAN));
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
         + GROWTH * features[M.SUCCESS_SINCE_FAILURE]
         + (features[M.SUCCESS_SINCE_FAILURE] >= 1 ? FIRST_SUCCESS_STEP : 0);
}

// Clamped stability (days) for scheduling/prediction.
function stabilityDaysForRepeat(features, params, floor) {
    const S = Math.exp(lnStabilityForRepeat(features, params, floor));
    // Degenerate params (NaN) → never-due: R(t) = exp(-t/∞) = 1, so the
    // problem is simply not a candidate (buildRecallSchedule skips it too).
    if (!isFinite(S)) return Infinity;
    return Math.max(MIN_STABILITY_DAYS, S);
}

// Window-open delay (hours) for a just-failed (r=0, no prior review) repeat at a
// representative difficulty — the stats panel's "a missed problem comes back in
// ~X h" readout. Returns null when params aren't ready. Uses representative
// per-problem 3PL inputs (corpus-mean a_p, no floor).
export function freshFailureReviewHours(params, { problemRank = PRIOR_MEAN, size, userRank, slope = A_REF, floor = 0 } = {}) {
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return null;
    const S = stabilityDaysForRepeat(buildFeatures({ isRepeat: true, reviewPct: 0, problemRank, size, r: 0, userRank, slope }), params, floor);
    return NEG_LN_080 * S * 24;
}

// Median window-open delay (hours) over the user's recent failures, each scored
// AS IF it just failed now — using each problem's real difficulty/slope/floor and
// that failure's own review percentile, exactly as getRecallCandidates schedules
// them (buildFeatures + stabilityDaysForRepeat). Reflects when the recommender would actually re-serve
// the user's recent misses, unlike freshFailureReviewHours which evaluates a
// single idealized average-difficulty, no-review, reference-slope, zero-floor
// failure (the minimal-stability corner, so a systematic underestimate). Returns
// null when params aren't ready or too few recent failures exist; callers fall
// back to freshFailureReviewHours then.
export function recentFailureWindowHours(params, history, { userRank, days = 7, minFailures = 5 } = {}) {
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return null;
    if (!Array.isArray(history)) return null;
    const cutoff = Date.now() - days * 24 * 3600e3;
    const recent = history.filter(a => typeof a.date === 'number' && a.date >= cutoff);
    // Review-percentile basis: the same window the scheduler ranks against
    // (getRecallCandidates: the last RECALL_CANDIDATE_HISTORY_LENGTH entries),
    // not just the 7-day failure window being scored.
    const allReviewTimes = history.slice(-RECALL_CANDIDATE_HISTORY_LENGTH)
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0)
        .sort((a, b) => a - b);
    const hrs = [];
    for (const a of recent) {
        if (a.correct || typeof a.problemRank !== 'number') continue;   // failures only
        const reviewPct = calculatePercentile(allReviewTimes,
            (a.accumulatedReview || 0) + (a.reviewTime || 0));
        const S = stabilityDaysForRepeat(buildFeatures({
            isRepeat: true, reviewPct, problemRank: a.problemRank,
            size: a.size, r: 0, userRank, slope: a.problemSlope,
        }), params, a.problemFloor);
        hrs.push(NEG_LN_080 * S * 24);
    }
    if (hrs.length < minFailures) return null;          // too sparse → caller falls back
    hrs.sort((x, y) => x - y);
    const m = hrs.length >> 1;
    return hrs.length % 2 ? hrs[m] : (hrs[m - 1] + hrs[m]) / 2;
}

// --- Recall serving share (constant-volume equilibrium) -----------------------
// Target: keep recalls and novels in balance at the volume a LIGHT day delivers
// — SHARE_VOLUME_FRACTION of the user's own trailing-week daily average — rather
// than at the average itself, so a user practising below their usual pace still
// gets through every due recall; on an average day the pool is then over-served
// and drains. At that target volume A the working-window guillotine
// (RECALL_CANDIDATE_HISTORY_LENGTH attempts) falls at ladder rung
//   k = ln(WINDOW / (A · τ0_days)) / GROWTH
// where τ0 is the user's own base window-open delay (their fitted stability at
// r=0, so between-user memory variance is priced in). The window is measured in
// ATTEMPTS, so a lower target volume spans more days, entries must climb further
// before graduating out of it, and the share rises — the fraction below 1 buys
// recall headroom, it does not cut recalls. A problem entering the ladder is
// expected to consume
//   E = (x^-k − 1) / (1 − x),   x = 0.80 (retention at window open = per-serve
//                                success probability, resets included)
// recall serves before it graduates past the window, and with e = ladder
// entries per novel serve (recent failed-below-rank fraction) the balanced
// recall:novel ratio is ρ = e·E. Serving a due recall with probability
// ρ/(1+ρ) — else the standing challenge — holds the mix at that equilibrium in
// expectation: a pool above it produces more dues than the share serves and the
// excess ages out unserved; a pool below it lets novels refill it. Returns the
// share in [0,1], or null when the inputs aren't estimable yet (no model, no
// volume, too few recent novels, no rank) — callers then always serve due
// recalls, the pre-gate behavior.
const X_SERVE = 0.8;                    // retention at the R=0.80 window open
const SHARE_WEEK_MS = 7 * 24 * 3600e3;  // volume-target window
const SHARE_VOLUME_FRACTION = 0.5;      // solve the equilibrium at this share of the week's daily average
const SHARE_NOVEL_SAMPLE = 200;         // entry-rate sample (recent novels)
const SHARE_MIN_NOVELS = 20;            // below this the rate is noise → null
export function recallServeShare() {
    const params = state.modelParams;
    if (typeof state.userRank !== 'number') return null;
    const history = getMergedAnswerHistory();
    if (!Array.isArray(history) || history.length === 0) return null;

    const cutoff = Date.now() - SHARE_WEEK_MS;
    let weekCount = 0;
    for (let i = history.length - 1; i >= 0; i--) {
        const d = history[i].date;
        if (typeof d !== 'number' || d < cutoff) break;
        weekCount++;
    }
    if (weekCount === 0) return null;
    const targetVolumePerDay = (weekCount / 7) * SHARE_VOLUME_FRACTION;

    let tau0Hours = recentFailureWindowHours(params, history, { userRank: state.userRank });
    if (tau0Hours == null) tau0Hours = freshFailureReviewHours(params, { userRank: state.userRank });
    if (tau0Hours == null || !(tau0Hours > 0)) return null;

    // Ladder entries per novel serve: only a FAILED novel below the user's rank
    // gets successSinceFailure defined + passes the candidate rank filter.
    let novels = 0, entries = 0;
    for (let i = history.length - 1; i >= 0 && novels < SHARE_NOVEL_SAMPLE; i--) {
        const a = history[i];
        if (a.isRepeat) continue;
        novels++;
        if (!a.correct && typeof a.problemRank === 'number' && a.problemRank < state.userRank) entries++;
    }
    if (novels < SHARE_MIN_NOVELS) return null;
    const entryRate = entries / novels;

    const k = Math.log(RECALL_CANDIDATE_HISTORY_LENGTH / (targetVolumePerDay * (tau0Hours / 24))) / GROWTH;
    if (!isFinite(k)) return null;
    // k ≤ 0: the FIRST due-interval already outlives the working window at the
    // target volume — a served recall could never survive to a second serve, so
    // the equilibrium share is genuinely 0.
    if (k <= 0) return 0;
    const expectedServes = (Math.pow(X_SERVE, -k) - 1) / (1 - X_SERVE);
    const rho = entryRate * expectedServes;
    return rho / (1 + rho);
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

    // 4PL correctness head (all rows): P = c + span·σ(inner + κ(c)), span = 1−c−LAPSE.
    const c = clampFloor(floor), span = clearSpan(c);
    const praw = sigmoid(logitZ(features, params) + clearKappa(c));
    const P = c + span * praw;
    const pc = Math.max(1e-15, Math.min(1 - 1e-15, P));
    let loss = -y * Math.log(pc) - (1 - y) * Math.log(1 - pc);
    // dLoss/dinner = (P−y)/(P(1−P)) · dP/dinner, dP/dinner = span·praw·(1−praw).
    // Reduces to (P−y) when c=0 and LAPSE=0 (the plain logistic limit).
    const eL = (pc - y) / (pc * (1 - pc)) * span * praw * (1 - praw);
    grad[M.INTERCEPT] += eL * features[M.INTERCEPT];
    grad[M.TIME_SOLVING] += eL * features[M.TIME_SOLVING];
    grad[M.REVIEW_TIME] += eL * features[M.REVIEW_TIME];
    grad[M.PROBLEM_DIFFICULTY] += eL * features[M.PROBLEM_DIFFICULTY];
    grad[M.SIZE_IS_13] += eL * features[M.SIZE_IS_13];

    // Retention head (repeat rows): fits LN_C only; z is stop-gradient. Skipped
    // when the elapsed gap is unknown (elapsedDays === null: the previous attempt
    // fell outside the history window) — without a true t the retention
    // likelihood is meaningless (an elapsedDays=0 fallback would inject a
    // constant ~34-nat loss on failed rows while contributing zero gradient).
    if (isRepeat && elapsedDays !== null) {
        const z = retentionLogit(features, params, c);
        const lnSRaw = params[M.LN_C] + KAPPA * (z - Z0) + GROWTH * features[M.SUCCESS_SINCE_FAILURE]
            + (features[M.SUCCESS_SINCE_FAILURE] >= 1 ? FIRST_SUCCESS_STEP : 0);
        const lnS = Math.max(LN_S_TRAIN_MIN, Math.min(LN_S_TRAIN_MAX, lnSRaw));
        const S = Math.exp(lnS);
        const u = elapsedDays / S;          // ≥ 0; u=0 ⇒ R=1 (just solved)
        const R = Math.exp(-u);
        const Rc = Math.max(1e-15, Math.min(1 - 1e-15, R));
        loss += -y * Math.log(Rc) - (1 - y) * Math.log(1 - Rc);
        const oneMinusR = Math.max(1 - R, 1e-9);
        // dlnS/dLN_C = 1 inside the clamp, 0 at the rails — a clamped row
        // contributes loss but no LN_C gradient (the clamp owns lnS there).
        if (lnSRaw === lnS) grad[M.LN_C] += (R - y) * u / oneMinusR;
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

// Whether the recall model can answer "what is due" at all. getRecallCandidates
// returns an empty list both when nothing is due and when the model isn't
// loaded yet, which are the same answer for a badge (show nothing) but opposite
// answers for the notification countdown (leave it alone vs. take it down).
export function recallModelReady() {
    return !!state.modelParams && state.modelParams.length === Object.keys(MODEL_INDICES).length;
}

// How many due recalls are WAITING — the Recall button's badge. The one on
// screen is being served, not waiting, so it is excluded; answering it drops it
// out of its window anyway, so the badge falls by one either way and never
// counts the same problem twice. Never throws: the badge must not be able to
// break a repaint (0 = no badge, the safe reading).
//
// Defaults to the ACTIVE size, because the button can only serve a recall on
// the board in front of the user. Callers counting a cross-size backlog — the
// notification countdown, whose schedule is uploaded for both sizes — pass the
// merged history explicitly.
export function waitingRecallCount(history = undefined) {
    try {
        const shown = state.currentProblem;
        const shownId = shown && shown.isRepeat ? tsumegoIdOf(shown.filename) : null;
        const candidates = getRecallCandidates(history ? { history } : {});
        return shownId ? candidates.filter((c) => c.problemId !== shownId).length : candidates.length;
    } catch (error) {
        console.error('waiting recall count failed:', error);
        return 0;
    }
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
    // answerHistory is ONE unified cross-size array (working-set split), so
    // slice it per-size via getAnswerHistoryForSize.
    const readSharedKey = (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        try { return JSON.parse(raw); } catch { return null; }
    };

    const params = readSharedKey('modelParams');
    if (!Array.isArray(params) || params.length !== Object.keys(MODEL_INDICES).length) return [];
    const answerHistory = getAnswerHistoryForSize(size);
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
    // is fit; L2-shrunk to LN_C_PRIOR (see fitUserModel).
    LN_C: 6,
};

const L2_REGULARIZATION_TARGETS = {
    // Placeholders only — fitUserModel OVERRIDES these per training run with the
    // population time/review confound (TIME_BIAS_PER_MIN/REVIEW_BIAS_PER_MIN)
    // converted from per-minute to this feature's per-percentile scale via that
    // run's time_per_percentile_min. The prior the coefficient shrinks toward MUST
    // equal the bias the confidence readout recenters by, else an uninformative
    // user can't read 50% (saturating to ~100% instead).
    [MODEL_INDICES.TIME_SOLVING]: -0.1,
    [MODEL_INDICES.REVIEW_TIME]: -0.1,
    // Multiplier on the per-problem a_p·(rank gap). Neutral ("the shipped
    // discrimination is right as-is for this user") is DIFFICULTY_NEUTRAL, NOT 1.0:
    // the gap feature is in rank units while a_p is per raw-rating unit (see
    // DIFFICULTY_NEUTRAL above).
    [MODEL_INDICES.PROBLEM_DIFFICULTY]: DIFFICULTY_NEUTRAL,
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

// Per-parameter multiplier on the shared L2 lambda (1 = the shared strength).
//
// PROBLEM_DIFFICULTY is the one exception. Lambda was tuned when the rank gap was
// measured in units of the WHOLE kyu/dan ladder rather than single ranks; the gap
// is now in single ranks, so the coefficient is smaller by the ladder span and an
// unscaled penalty would be weaker by its SQUARE. Restating the SAME prior on the
// per-rank coefficient is that square. This is load-bearing, not cosmetic: the data
// carries under a nat about this coefficient even over a full window, so the prior
// IS the estimate — at the shared lambda it would float free.
const LADDER_SPAN_AT_LAMBDA_TUNING = 29;
const LAMBDA_SCALE_BY_PARAM = {
    [MODEL_INDICES.PROBLEM_DIFFICULTY]: LADDER_SPAN_AT_LAMBDA_TUNING ** 2,
};
const LAMBDA_SCALE = (j) => LAMBDA_SCALE_BY_PARAM[j] ?? 1;

// Apply the retention constants from the (MANDATORY) population_constants.json;
// population-constants.js calls this at boot before any training/scheduling, and also
// re-points the LN_C L2 target at the prior. Throws on a missing/invalid field so a
// corrupt blob is fatal (the caller aborts boot) rather than silently leaving the
// constants unset — there is no baked default to fall back to.
export function applyRetentionConstants({ kappa, z0, growth, ln_c_prior, first_success_step } = {}) {
    if (!(Number.isFinite(kappa) && kappa > 0 && kappa < 20)) throw new Error(`population constants: bad kappa ${kappa}`);
    if (!Number.isFinite(z0)) throw new Error(`population constants: bad z0 ${z0}`);
    if (!(Number.isFinite(growth) && growth > -2 && growth < 4)) throw new Error(`population constants: bad growth ${growth}`);
    if (!Number.isFinite(ln_c_prior)) throw new Error(`population constants: bad ln_c_prior ${ln_c_prior}`);
    if (!(Number.isFinite(first_success_step) && first_success_step >= 0 && first_success_step < 2)) throw new Error(`population constants: bad first_success_step ${first_success_step}`);
    KAPPA = kappa;
    Z0 = z0;
    GROWTH = growth;
    FIRST_SUCCESS_STEP = first_success_step;
    LN_C_PRIOR = ln_c_prior;
    L2_REGULARIZATION_TARGETS[MODEL_INDICES.LN_C] = ln_c_prior;
}

// Apply the time-benefit confound biases from the (MANDATORY) population
// constants. Throws on a non-finite field so a corrupt blob is fatal (the caller
// aborts boot) rather than silently leaving the biases unset. The value is the
// backend's fitted bias verbatim (the fitter owns its sign/magnitude; the
// frontend does not second-guess it). population-constants.js calls this at boot
// before any training; the per-user time coefficients are L2-shrunk toward it
// (see fitUserModel) and the confidence readout recenters by it.
export function setTimeBias({ timePerMin, reviewPerMin } = {}) {
    if (!Number.isFinite(timePerMin)) throw new Error(`population constants: bad time_bias_per_min ${timePerMin}`);
    if (!Number.isFinite(reviewPerMin)) throw new Error(`population constants: bad review_bias_per_min ${reviewPerMin}`);
    TIME_BIAS_PER_MIN = timePerMin;
    REVIEW_BIAS_PER_MIN = reviewPerMin;
}

// One-sided confidence that the bias-corrected coefficient is positive (in
// [0,1]), or 0 when the stat is missing/degenerate. SIGNED test (not |coeff|/se):
// after recentering by the population confound the sign carries the meaning. The
// coeff/se passed in are already per-MINUTE (calculateResultsAndStdErrors scales
// them by 1/time_per_percentile_min), matching the per-minute bias.
function benefitConfidence(stat, biasPerMin) {
    if (!stat || !(stat.se > 0)) return 0;
    const z = (stat.coeff - biasPerMin) / stat.se;   // bias is negative ⇒ shifts the estimate up
    return normCdf(z);
}

// Var-keyed wrapper: picks the population bias for this model term and returns the
// recentered one-sided benefit confidence. Exported so the stats Time tab displays
// the SAME corrected "% chance more time helps" the nudge gate tests against —
// otherwise the tab would show the raw, confound-biased coeff.
export function benefitConfidenceForVar(stat, varIndex) {
    const biasPerMin = varIndex === MODEL_INDICES.REVIEW_TIME ? REVIEW_BIAS_PER_MIN : TIME_BIAS_PER_MIN;
    return benefitConfidence(stat, biasPerMin);
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
    // Signed: a negative fitted slope is weak/contrary evidence and should pull
    // the blend DOWN toward the floor, not flip into positive evidence (a
    // Math.abs would do exactly that). The 0.05 floor below keeps the result usable.
    const userSlope = pd.coeff;
    const userPrecision = 1 / (pd.se * pd.se);
    const blended = (priorPrecision * LOGIT_PER_RANK_PRIOR + userPrecision * userSlope)
        / (priorPrecision + userPrecision);
    return Math.max(0.05, blended);
}

// NOTE: the one-shot wipe of pre-3PL persisted params lives in migrations.js
// (migrateModelArch3pl), which runMigrations() runs at boot before any training.
//
// No in-flight guard and no coalescing queue: the fit is synchronous end to end
// (a few Newton steps over the window, no yielding), so two callers cannot
// interleave and there is nothing to drop or to re-run.
export function calculateAndStoreStatsResults() {
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

        // Every fit starts from the same point rather than from the stored
        // parameters: both blocks converge in a handful of Newton steps either
        // way, and starting cold makes the model a pure function of the history —
        // the same window gives the same fit on every device, with no path
        // dependence on where a previous run's iterations happened to end. LN_C
        // starts at the population prior so the first steps are well scaled (it
        // is also the base stability a user with no repeats keeps, since the
        // penalty is then the whole of its objective).
        const numParams = Object.keys(MODEL_INDICES).length;
        const params = Array(numParams).fill(0);
        params[MODEL_INDICES.LN_C] = LN_C_PRIOR;

        const { trainedParams } = fitUserModel(trainingData, params);

        // Sanity check for non-numeric values
        const areParamsValid = trainedParams.every(p => isFinite(p));

        if (areParamsValid) {
            const finalResults = calculateResultsAndStdErrors(trainedParams, trainingData);
            setModelParams(trainedParams);
            setStatsResults(finalResults);
        } else {
            console.error("Model training resulted in non-numeric weights. Wiping model state.");
            captureMessage("Model training resulted in non-numeric weights.");
            setModelParams(null);
            setStatsResults(null);
        }

    } catch (error) {
        console.error('Unified model training failed:', error);
        captureException(error);
    }
}

// Mean internal rank over the user's repeats — the "typical repeat difficulty"
// a hypothetical fresh failure is scored at. Shared by the recall-impact
// refinement below and the stats Activity text; falls back to `fallback` when
// no ranked repeats exist yet.
export function averageRepeatRank(history, fallback) {
    const ranks = history.filter(p => p.isRepeat && typeof p.problemRank === 'number').map(p => p.problemRank);
    return ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : fallback;
}

// Per-attempt recall impact for the result overlay: the time until first repeat
// of a GENERIC problem, before and after teaching the model this attempt.
// Returns { beforeHours, afterHours } (nulls if the model isn't ready).
//
// There is no refine loop. Extra optimizer steps used to be taken, one at a
// time, whenever the displayed delay moved against the answer's sign — which
// only ever worked because the fit was stopping short of its optimum, so more
// steps still had somewhere to go. From a converged fit they have nowhere: the
// remaining disagreement is the two heads genuinely disagreeing on this window,
// and the overlay shows the delta the model actually produces.
export function recallImpact() {
    const genericProblemRank = averageRepeatRank(getMergedAnswerHistory(), state.userRank);
    const opts = { problemRank: genericProblemRank, size: state.boardSize, userRank: state.userRank };

    const beforeHours = freshFailureReviewHours(state.modelParams, opts);
    calculateAndStoreStatsResults();
    const afterHours = freshFailureReviewHours(state.modelParams, opts);
    return { beforeHours, afterHours };
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
    // Percentile basis: the SAME distribution the scheduler ranks against
    // (getRecallCandidates / buildRecallSchedule) — review totals
    // (accumulatedReview + reviewTime, positive only) over the last
    // RECALL_CANDIDATE_HISTORY_LENGTH entries — so the trained coefficient is
    // applied at scheduling on the percentile scale it was fit on.
    const sortedReview = validHistory.slice(-RECALL_CANDIDATE_HISTORY_LENGTH)
        .map(p => (p.accumulatedReview || 0) + (p.reviewTime || 0))
        .filter(t => t > 0)
        .sort((a, b) => a - b);
    let time_per_percentile_min_r = 0;
    if (sortedReview.length > 0) {
        const idx_low_r = Math.floor(p_low * (sortedReview.length - 1));
        const idx_high_r = Math.ceil(p_high * (sortedReview.length - 1));
        time_per_percentile_min_r = (sortedReview[idx_high_r] - sortedReview[idx_low_r]) / (60000 * (p_high - p_low));
    }
    
    const percentileReview = previousReviewTimes.map(t => calculatePercentile(sortedReview, t));

    // Elapsed days since each repeat row's previous attempt of the same problem
    // (the retention head's t). Zero for novel rows (they use the logistic head);
    // null for a repeat whose previous attempt fell outside validHistory — there
    // is no usable gap, so predictAndGrad skips its retention term (the row
    // still trains the correctness head).
    const DAY = 1000 * 60 * 60 * 24;
    const elapsedDays = history.map(p => {
        if (!p.isRepeat) return 0;
        const prev = validHistory.filter(a => a.problemId === p.problemId && a.date < p.date).pop();
        if (!prev) return null;
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

// Newton iteration cap per block, and the convergence tolerance on the Newton
// decrement (twice the loss reduction the quadratic model predicts, in nats over
// the whole window). Both blocks converge in a handful of steps, so the cap is a
// runaway guard rather than a budget the fit is expected to spend — which is why
// there is no wall-clock cap beside it and no cooperative yielding: a fit is now
// tens of passes over the window instead of thousands, well inside one frame.
//
// The decrement is what replaces a loss-delta stop. It is curvature-weighted, so
// it does not go small merely because the objective is flat in a direction — and
// the difficulty multiplier, which the whole recall schedule is scaled by, sits
// in exactly such a direction (its prior carries it, the window holding well
// under a nat about it).
const MAX_NEWTON_ITERS = 50;
const NEWTON_TOL = 1e-8;
// Numerical guard on the unpenalized diagonal (the intercept). Same value the
// posterior-covariance step uses, now shared with the step that produces the
// parameters it describes.
const RIDGE_GUARD = 1e-6;

// Fit the two heads by damped Newton, each against the loss it owns.
//
// The objective SEPARATES, and the fit follows that structure rather than
// working around it: the correctness head owns the five logit parameters, and
// the retention head owns LN_C alone while reading the cold-solve logit z as a
// constant (predictAndGrad's stop-gradient — the correctness head owns z). So
// the correctness block's objective does not involve LN_C at all and is fit
// first, to convergence; LN_C is then fit against the z that block settled on.
// No alternation is needed for that to be exact.
//
// Curvature per block: the correctness head steps on the EXPECTED information
// Σ wᵢ·vᵢ·xᵢxᵢᵀ with vᵢ = (dP/dinner)²/(P(1−P)) — the same Fisher matrix the SE
// step computes — because a 4PL log-likelihood with a non-zero guessing floor is
// not concave everywhere, and the expected information is positive semi-definite
// where the observed Hessian need not be. The retention head uses its exact
// second derivative, which is non-negative for both outcomes (for a success it
// is u; for a failure u·R·[u − (1−R)]/(1−R)², and u ≥ 1−e^(−u) for all u ≥ 0).
function fitUserModel(trainingData, params) {
    const { y, weights, elapsedDays, isRepeatArr, n, floors,
            time_per_percentile_min, time_per_percentile_min_r } = trainingData;
    const M = MODEL_INDICES;
    const lambda = 10; // L2 regularization parameter

    // Per-run L2 targets. The TIME_SOLVING/REVIEW_TIME coefficients live in
    // per-percentile units (the feature is a 0..1 percentile), but the population
    // confound is shipped per-MINUTE; convert it to this run's per-percentile
    // scale (×time_per_percentile_min) so the coefficient is shrunk toward exactly
    // the offset the confidence readout later recenters by — an uninformative user
    // then reads ~50%, not the ~100% a fixed -0.1 target would produce.
    const targets = { ...L2_REGULARIZATION_TARGETS };
    targets[M.TIME_SOLVING] = (TIME_BIAS_PER_MIN ?? 0) * (time_per_percentile_min || 0);
    targets[M.REVIEW_TIME] = (REVIEW_BIAS_PER_MIN ?? 0) * (time_per_percentile_min_r || 0);

    // INTERCEPT (logit level) is the only unregularized parameter; LN_C is
    // shrunk to its population prior (LN_C_PRIOR) via L2 like the others.
    const targetFor = (j) => targets[j] || 0;
    const precisionFor = (j) => (j === M.INTERCEPT ? 0 : LAMBDA_SCALE(j) * lambda);

    // Features are independent of params, so build the design matrix once.
    const X = Array.from({ length: n }, (_, i) => featuresForTrainingRow(trainingData, i));

    // Everything below is in RAW SUMS over the window (not per-attempt means):
    // the loss, gradient and curvature share one scaling, which is all a Newton
    // step needs, and it is the scaling the Fisher/SE step already works in.

    // --- Block A: the correctness head over the logit parameters ---------------
    const LOGIT_IDX = [M.INTERCEPT, M.TIME_SOLVING, M.REVIEW_TIME, M.PROBLEM_DIFFICULTY, M.SIZE_IS_13];

    // Per-row 4PL quantities: the clipped probability and dP/dinner.
    const solveRow = (p, i) => {
        const c = clampFloor(floors[i]), span = clearSpan(c);
        const praw = sigmoid(logitZ(X[i], p) + clearKappa(c));
        const P = Math.max(1e-15, Math.min(1 - 1e-15, c + span * praw));
        return { P, dP: span * praw * (1 - praw) };
    };
    const solveLoss = (p) => {
        let loss = 0;
        for (let i = 0; i < n; i++) {
            const { P } = solveRow(p, i);
            loss += weights[i] * (-y[i] * Math.log(P) - (1 - y[i]) * Math.log(1 - P));
        }
        for (const j of LOGIT_IDX) {
            const d = p[j] - targetFor(j);
            loss += 0.5 * precisionFor(j) * d * d;
        }
        return loss;
    };
    const solveGrad = (p) => {
        const g = new Array(LOGIT_IDX.length).fill(0);
        for (let i = 0; i < n; i++) {
            const { P, dP } = solveRow(p, i);
            const e = weights[i] * (P - y[i]) / (P * (1 - P)) * dP;
            for (let b = 0; b < LOGIT_IDX.length; b++) g[b] += e * X[i][LOGIT_IDX[b]];
        }
        for (let b = 0; b < LOGIT_IDX.length; b++) {
            const j = LOGIT_IDX[b];
            g[b] += precisionFor(j) * (p[j] - targetFor(j));
        }
        return g;
    };
    const solveHess = (p) => {
        const nb = LOGIT_IDX.length;
        const H = Array.from({ length: nb }, () => new Array(nb).fill(0));
        for (let i = 0; i < n; i++) {
            const { P, dP } = solveRow(p, i);
            const v = weights[i] * dP * dP / (P * (1 - P));
            for (let a = 0; a < nb; a++) {
                const va = v * X[i][LOGIT_IDX[a]];
                if (va === 0) continue;
                for (let b = a; b < nb; b++) H[a][b] += va * X[i][LOGIT_IDX[b]];
            }
        }
        for (let a = 0; a < nb; a++) {
            H[a][a] += precisionFor(LOGIT_IDX[a]) + RIDGE_GUARD;
            for (let b = a + 1; b < nb; b++) H[b][a] = H[a][b];
        }
        return H;
    };

    // --- Block B: the retention head over LN_C ---------------------------------
    // Only repeat rows with a known elapsed gap carry retention information;
    // without a true interval the retention likelihood is meaningless (see
    // predictAndGrad). A row whose stability lands on a training rail contributes
    // its loss but no derivative — the clamp owns lnS there.
    const retentionRows = [];
    for (let i = 0; i < n; i++) {
        if (isRepeatArr[i] && elapsedDays[i] !== null) retentionRows.push(i);
    }
    const retentionRow = (p, i) => {
        const c = clampFloor(floors[i]);
        const z = retentionLogit(X[i], p, c);
        const lnSRaw = p[M.LN_C] + KAPPA * (z - Z0) + GROWTH * X[i][M.SUCCESS_SINCE_FAILURE]
            + (X[i][M.SUCCESS_SINCE_FAILURE] >= 1 ? FIRST_SUCCESS_STEP : 0);
        const lnS = Math.max(LN_S_TRAIN_MIN, Math.min(LN_S_TRAIN_MAX, lnSRaw));
        const u = elapsedDays[i] / Math.exp(lnS);
        const R = Math.exp(-u);
        return { clamped: lnSRaw !== lnS, u, R };
    };
    const retentionLoss = (p) => {
        let loss = 0;
        for (const i of retentionRows) {
            const { R } = retentionRow(p, i);
            const Rc = Math.max(1e-15, Math.min(1 - 1e-15, R));
            loss += weights[i] * (-y[i] * Math.log(Rc) - (1 - y[i]) * Math.log(1 - Rc));
        }
        const d = p[M.LN_C] - targetFor(M.LN_C);
        loss += 0.5 * precisionFor(M.LN_C) * d * d;
        return loss;
    };
    const retentionGrad = (p) => {
        let g = 0;
        for (const i of retentionRows) {
            const { clamped, u, R } = retentionRow(p, i);
            if (clamped) continue;
            g += weights[i] * (R - y[i]) * u / Math.max(1 - R, 1e-9);
        }
        return [g + precisionFor(M.LN_C) * (p[M.LN_C] - targetFor(M.LN_C))];
    };
    const retentionHess = (p) => {
        let h = 0;
        for (const i of retentionRows) {
            const { clamped, u, R } = retentionRow(p, i);
            if (clamped) continue;
            const om = Math.max(1 - R, 1e-9);
            h += weights[i] * (-u * (R - y[i]) / om + u * u * R * (1 - y[i]) / (om * om));
        }
        return [[h + precisionFor(M.LN_C) + RIDGE_GUARD]];
    };

    const runBlock = (indices, gradFn, hessFn, lossFn) => {
        let steps = 0, stopReason = 'maxIters';
        for (let iter = 0; iter < MAX_NEWTON_ITERS; iter++) {
            const { decrement, accepted } =
                newtonBlockStep(params, indices, gradFn(params), hessFn(params), () => lossFn(params));
            if (!accepted) { stopReason = 'stalled'; break; }
            steps++;
            if (decrement / 2 < NEWTON_TOL) { stopReason = 'converged'; break; }
        }
        return { steps, stopReason };
    };

    const solveFit = runBlock(LOGIT_IDX, solveGrad, solveHess, solveLoss);
    const retentionFit = runBlock([M.LN_C], retentionGrad, retentionHess, retentionLoss);

    // SUCCESS_SINCE_FAILURE is a feature slot with no fitted coefficient (growth
    // is a population constant), so the penalty is the whole of its objective and
    // its minimizer is the target exactly.
    params[M.SUCCESS_SINCE_FAILURE] = targetFor(M.SUCCESS_SINCE_FAILURE);

    return { trainedParams: params, solveFit, retentionFit };
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
    // features (SUCCESS_SINCE_FAILURE / LN_C) don't enter, so their rows and
    // columns stay zero; the ridge guard keeps the matrix invertible (large unused
    // SE there). This is the same information the correctness block steps on.
    //
    // Accumulated straight into the (np × np) array rather than as Dᵀ·W·D: W is a
    // DIAGONAL of per-row weights, and materializing it as a dense n × n matrix
    // made this step — not the fit — the dominant cost of a retrain.
    const logitIdx = [M.INTERCEPT, M.TIME_SOLVING, M.REVIEW_TIME, M.PROBLEM_DIFFICULTY, M.SIZE_IS_13];
    const info = Array.from({ length: np }, () => new Array(np).fill(0));
    for (let i = 0; i < n; i++) {
        const xi = X[i];
        const c = clampFloor(floors[i]), span = clearSpan(c);
        const praw = sigmoid(logitZ(xi, finalParams) + clearKappa(c));
        const P = Math.max(1e-9, Math.min(1 - 1e-9, c + span * praw));
        const dP = span * praw * (1 - praw);   // dP/dinner
        const v = weights[i] * dP * dP / (P * (1 - P));
        for (const a of logitIdx) {
            const va = v * xi[a];
            if (va === 0) continue;
            for (const b of logitIdx) info[a][b] += va * xi[b];
        }
    }
    for (let j = 0; j < np; j++) info[j][j] += RIDGE_GUARD;
    const varCovarMatrix = invertMatrix(info);
    const stdErrors = varCovarMatrix.map((row, j) => Math.sqrt(row[j]));

    const results = [];
    for (let i = 1; i < Object.keys(MODEL_INDICES).length; i++) {
        let coeff = finalParams[i], se = stdErrors[i], scaleFactor = 1;
        if (i === MODEL_INDICES.TIME_SOLVING && time_per_percentile_min > 0) scaleFactor = 1 / time_per_percentile_min;
        else if (i === MODEL_INDICES.REVIEW_TIME && time_per_percentile_min_r > 0) scaleFactor = 1 / time_per_percentile_min_r;
        // PROBLEM_DIFFICULTY is a multiplier on a_p·gap, so the effective
        // logit-per-rank is coeff·a_p. Fold a representative a_p (A_REF) into the
        // reported coeff so getLogitPerRank / the size-comparison readout keep a
        // consistent "logit per rank" scale.
        else if (i === MODEL_INDICES.PROBLEM_DIFFICULTY) scaleFactor = A_REF;
        results.push({ coeff: coeff * scaleFactor, se: se * Math.abs(scaleFactor), varIndex: i, scaleFactor });
    }
    return { results, varCovarMatrix };
}


