// js/stats-compute.js
//
// Pure-numeric helpers consumed by the detailed-stats UI (stats-ui.js) — no DOM,
// no rendering. Split out of stats-ui so the UI module is rendering-only and the
// numeric work (community-bucket mirroring, the time-adjust smoother, the
// weakness-example selector) sits with the rest of the model math.

import { TOTAL_RANKS } from './constants.js';
import { rankScalarToLabel, RANK_INDEX_OFFSET } from '../shared/rank-math.js';
import { TIME_ADJUST_AFK_CAP_MS, TIME_ADJUST_MIN_TIMED } from './population-constants.js';
import { normCdf, weightedQuantile } from './math-utils.js';

// Locate the current user within each Community distribution, reproducing the
// Worker's per-user bucketing (computeServerStats in src/server-stats.js) on the
// client's own last-7-day merged history. Returns a bin index per bar-list
// category (-1 when not determinable) plus an answer-derived rank label.
// `timeBins` is the seconds-based display-bin table from the caller (TIME_BINS).
export function computeUserCommunityBuckets(history, timeBins) {
    const out = { attemptsPerDay: -1, solve: -1, review: -1, recall: -1, rankLabel: null };
    if (!Array.isArray(history) || !history.length) return out;

    const cutoff = Date.now() - 7 * 24 * 3600e3;
    const recent = history.filter(a => typeof a.date === 'number' && a.date >= cutoff);
    if (!recent.length) return out;

    const median = (arr) => {
        if (!arr.length) return null;
        const s = arr.slice().sort((a, b) => a - b);
        const m = s.length >> 1;
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };
    // timeBins are in SECONDS now (see TIME_BINS in buildServerStatsPane); the
    // edges align to the server's bucket width, so binning the median directly
    // lands in the same bin the server would have bucketed it into.
    const binSeconds = (ms) => timeBins.findIndex(b => (ms / 1000) >= b.lo && (ms / 1000) < b.hi);

    // Attempts per active (UTC) day — every attempt counts, repeats included.
    const PER_USER_BINS = [[1, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, 160], [160, Infinity]];
    const days = new Set(recent.map(a => new Date(a.date).toISOString().slice(0, 10)));
    if (days.size) {
        const rate = recent.length / days.size;
        out.attemptsPerDay = PER_USER_BINS.findIndex(([lo, hi]) => rate >= lo && rate < hi);
    }

    // Solving time: median of per-attempt solve times (ms), then coarse-binned.
    const solveMed = median(recent.map(a => a.timeTaken).filter(t => typeof t === 'number' && t > 0));
    if (solveMed != null) out.solve = binSeconds(solveMed);

    // Review time: per distinct problem the cumulative review of its LAST
    // attempt (the running total, as the server keeps), median across problems.
    const lastByProblem = new Map();
    for (const a of recent) {
        if (!a.problemId) continue;
        const prev = lastByProblem.get(a.problemId);
        if (!prev || a.date > prev.date) lastByProblem.set(a.problemId, a);
    }
    const reviewMed = median([...lastByProblem.values()]
        .map(a => (a.accumulatedReview || 0) + (a.reviewTime || 0))
        .filter(t => t > 0));
    if (reviewMed != null) out.review = binSeconds(reviewMed);

    // Recall time: shortest failure→next-attempt gap that is ≥12h, per problem.
    const RECALL_MIN_MS = 12 * 3600e3;
    const RECALL_HI = [24, 48, 72, 96, 120, Infinity].map(h => h * 3600e3);
    const seqByProblem = new Map();
    for (const a of recent) {
        if (!a.problemId) continue;
        let seq = seqByProblem.get(a.problemId);
        if (!seq) { seq = []; seqByProblem.set(a.problemId, seq); }
        seq.push({ t: a.date, success: !!a.correct });
    }
    let recallMin = Infinity;
    for (const seq of seqByProblem.values()) {
        if (seq.length < 2) continue;
        seq.sort((x, y) => x.t - y.t);
        for (let i = 1; i < seq.length; i++) {
            if (seq[i - 1].success) continue;            // recall after a success — skip
            const gap = seq[i].t - seq[i - 1].t;
            if (gap >= RECALL_MIN_MS && gap < recallMin) recallMin = gap;
        }
    }
    if (recallMin < Infinity) out.recall = RECALL_HI.findIndex(hi => recallMin < hi);

    // Answer-derived rank: mean internal-rank index of distinct first-time
    // problems → EGF scalar → label (the server's recipe, minus shard rounding;
    // problemRank is already the per-problem M index normalized to 0..1).
    const MIN_RANK_PROBLEMS = 10;
    const firstByProblem = new Map();
    for (const a of recent) {
        if (a.isRepeat || typeof a.problemRank !== 'number') continue;
        if (!firstByProblem.has(a.problemId)) firstByProblem.set(a.problemId, a.problemRank);
    }
    if (firstByProblem.size >= MIN_RANK_PROBLEMS) {
        let sum = 0;
        for (const r of firstByProblem.values()) sum += r * TOTAL_RANKS;
        const scalar = Math.max(-29, Math.min(9, Math.round(sum / firstByProblem.size) - RANK_INDEX_OFFSET));
        out.rankLabel = rankScalarToLabel(scalar);
    }

    return out;
}

// A TIME-VARYING version of medianNovelSolveMinutes: the kernel-weighted median
// (in minutes) of the SAME capped-novel timed series the population time-adjust
// alpha was fit against, evaluated at any t. Smoothed at the skill kernel's
// ~30-day scale so the correction tracks the slow skill line, not session noise.
// Where a window has thin timed support the estimate is shrunk toward the
// whole-history median (empirical Bayes in log-minute space) — so the adjusted
// line stays CONTINUOUS, degrading smoothly to today's single-scalar correction
// rather than breaking. NB the inclusion rule (non-repeat, >0, AFK-capped) must
// match user_median_log10_minutes in fit_population_constants — the Time tab's
// solving series is broader (repeats, no AFK cap), so we reuse the SMOOTHER idea
// here, not that series. Returns null (no time adjustment possible) when the
// whole history has fewer than TIME_ADJUST_MIN_TIMED timed attempts.
const TIME_ADJUST_TAU_MS = 30 * 24 * 3600 * 1000;   // matches TAU_S_MS (skill kernel)
const TIME_ADJUST_SHRINK_ESS = 8;                   // log-space shrink prior, in attempts
export function makeMedianSolveMinutesAt(history) {
    const pts = history
        .filter(p => !p.isRepeat && Number.isFinite(p.timeTaken)
            && p.timeTaken > 0 && p.timeTaken <= TIME_ADJUST_AFK_CAP_MS
            && typeof p.date === 'number')
        .map(p => ({ t: p.date, v: p.timeTaken }))
        .sort((a, b) => a.t - b.t);
    if (pts.length < TIME_ADJUST_MIN_TIMED) return null;
    const ts = pts.map(p => p.t);
    const sortedAll = pts.map(p => p.v).slice().sort((a, b) => a - b);
    const gm = sortedAll.length % 2 ? sortedAll[sortedAll.length >> 1]
        : (sortedAll[(sortedAll.length >> 1) - 1] + sortedAll[sortedAll.length >> 1]) / 2;
    const globalLog = Math.log10(gm / 60000);

    return (t) => {
        let W = 0, sumW2 = 0;
        const pairs = [];
        for (let i = 0; i < pts.length; i++) {
            const d = (t - ts[i]) / TIME_ADJUST_TAU_MS;
            if (Math.abs(d) > 4) continue;
            const w = Math.exp(-0.5 * d * d);
            W += w; sumW2 += w * w;
            pairs.push({ v: pts[i].v, w });
        }
        if (W <= 0 || pairs.length < 2) return Math.pow(10, globalLog);
        const ess = (W * W) / sumW2;
        pairs.sort((a, b) => a.v - b.v);
        const localLog = Math.log10(weightedQuantile(pairs, W, 0.5) / 60000);
        // Shrink the local median toward the global one by effective sample size.
        const wl = ess / (ess + TIME_ADJUST_SHRINK_ESS);
        return Math.pow(10, wl * localLog + (1 - wl) * globalLog);
    };
}

// Picks wrong attempts whose embedding signature aligns with the (k, quadrant)
// issue. Each quadrant requires BOTH axes to land on the side the quadrant's
// joint sign demands — a one-sided gate would include attempts that only
// support one axis, which is misleading for the both-negative `badAt` case
// (and for the off-diagonals it lets same-sign-ordered attempts slip through).
//
// The minimum |ec|, |ep| per attempt scales with the issue's posterior
// probability via `threshold = probit(p)` — so a 90%-confident issue needs
// ~1.28σ-strong evidence per axis, while a 50%-confident issue collapses to
// the pure sign constraint. Without this scaling, a 51%-borderline issue and
// a 95%-certain one would surface the same attempts, even though the latter
// should be backed by stronger per-attempt evidence.
//
// After the gate, per-attempt evidence strength is the sum of the two relevant
// signed σ contributions — monotone in either axis so a 5σ-on-one + threshold-
// on-the-other attempt ranks above one that just clears the threshold on both.
// (The earlier `min` tied any two attempts at the weaker axis regardless of
// the other one, which made the ordering look arbitrary on the panel.)
// Embeddings are unit-variance per component (skew-ICA standardized), so
// z-scale thresholds and sums are directly interpretable as σ above/below
// baseline.
export function findExamplesForIssue(answerHistory, k, quadrant, excludedIds, threshold = 0) {
    const t = Number.isFinite(threshold) ? Math.max(0, threshold) : 0;
    const byId = new Map();
    for (const a of answerHistory) {
        if (!a || a.correct) continue;
        if (a.isRepeat) continue;  // mirror weakness.js training filter — repeats are excluded there
        if (a.ignored) continue;
        if (!a.problemId) continue;
        if (excludedIds && excludedIds.has(a.problemId)) continue;
        if (!Array.isArray(a.embedding) || a.embedding.length <= k) continue;
        if (!Array.isArray(a.pickedEmbedding) || a.pickedEmbedding.length <= k) continue;
        const ec = a.embedding[k];
        const ep = a.pickedEmbedding[k];
        if (!Number.isFinite(ec) || !Number.isFinite(ep)) continue;
        let score;
        if (quadrant === 'tooMuch') {
            if (!(ep > t) || !(-ec > t)) continue;
            score = ep + (-ec);
        } else if (quadrant === 'notEnough') {
            if (!(ec > t) || !(-ep > t)) continue;
            score = ec + (-ep);
        } else {  // badAt
            if (!(ec > t) || !(ep > t)) continue;
            score = ec + ep;
        }
        const prev = byId.get(a.problemId);
        if (!prev || score > prev.score) {
            byId.set(a.problemId, { problemId: a.problemId, date: a.date, score, ec, ep, size: a.size });
        }
    }
    return Array.from(byId.values()).sort((a, b) => b.score - a.score);
}

// Inverse standard-normal CDF via bisection over `normCdf`. ~50 iterations
// converges to ≪ 1e-9; cheap once per issue at panel-open time. Returns 0 for
// p ≤ 0.5 since callers only ever pass posterior probabilities in [0.5, 1).
export function probit(p) {
    if (!(p > 0.5)) return 0;
    if (p >= 1) return Infinity;
    let lo = 0, hi = 8;
    for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        const cdf = normCdf(mid);
        if (cdf < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}
