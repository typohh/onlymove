// js/stats-compute.js
//
// Pure-numeric helpers consumed by the detailed-stats pane modules
// (stats-community.js, stats-rank-pane.js, stats-time-chart.js,
// stats-weaknesses.js) — no DOM, no rendering. Split out of the stats UI so
// those modules are rendering-only and the numeric work (community-bucket
// mirroring, the kernel smoothers, the weakness-example selector) sits with
// the rest of the model math.

import { TOTAL_RANKS, MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { rankScalarToLabel, RANK_INDEX_OFFSET } from '../shared/rank-math.js';
import { normCdf, weightedQuantile, gaussianWeight, lowerBound } from './math-utils.js';

// Display bins in SECONDS over the server's fine-bucket histogram (the
// Community pane's solving/review-time bar lists). Solving time is log-normal
// (the hump sits at ~5–20s), so the low end is split finely and the long tail
// coarsely — this reveals the bell instead of collapsing it into one tall
// "0–10s" bar. Every edge is a multiple of the server's bucket width, so a
// bucket maps cleanly to exactly one display bin.
export const TIME_BINS = [
    { lo: 0, hi: 5, label: '0–5s' },
    { lo: 5, hi: 10, label: '5–10s' },
    { lo: 10, hi: 15, label: '10–15s' },
    { lo: 15, hi: 20, label: '15–20s' },
    { lo: 20, hi: 30, label: '20–30s' },
    { lo: 30, hi: 45, label: '30–45s' },
    { lo: 45, hi: 60, label: '45–60s' },
    { lo: 60, hi: 90, label: '1–1.5m' },
    { lo: 90, hi: 120, label: '1.5–2m' },
    { lo: 120, hi: 180, label: '2–3m' },
    { lo: 180, hi: 300, label: '3–5m' },
    { lo: 300, hi: Infinity, label: '5m+' },
];

// Fold a server fine-bucket histogram ([{bucket, n}] rows, bucket width
// `bucketSec` seconds) into the TIME_BINS display bins.
export function binHist(hist, bucketSec) {
    const rows = TIME_BINS.map(b => ({ label: b.label, value: 0 }));
    for (const r of (hist || [])) {
        const sec = r.bucket * bucketSec;   // lower edge of the server bucket
        const i = TIME_BINS.findIndex(b => sec >= b.lo && sec < b.hi);
        if (i >= 0) rows[i].value += r.n;
    }
    return rows;
}

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
    // timeBins are in SECONDS now (see TIME_BINS above); the
    // edges align to the server's bucket width, so binning the median directly
    // lands in the same bin the server would have bucketed it into.
    const binSeconds = (ms) => timeBins.findIndex(b => (ms / 1000) >= b.lo && (ms / 1000) < b.hi);

    // Attempts per day — the week's attempts averaged over the full 7-day window
    // (÷ 7, NOT ÷ distinct active days); every attempt counts, repeats included.
    // Bins must mirror PER_USER_BINS in computeServerStats (src/server-stats.js).
    const PER_USER_BINS = [[0, 1], [1, 2], [2, 5], [5, 10], [10, 20], [20, 50], [50, Infinity]];
    const rate = recent.length / 7;
    out.attemptsPerDay = PER_USER_BINS.findIndex(([lo, hi]) => rate >= lo && rate < hi);

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

    // Recall time: median novel→first-repeat gap ≥6h across problems — per
    // problem the gap from the novel (first non-repeat) attempt, only when it
    // FAILED, to the very next attempt (mirrors the Worker's recallDist).
    const RECALL_MIN_MS = 6 * 3600e3;
    const RECALL_HI = [24, 48, 72, 96, 120, Infinity].map(h => h * 3600e3);
    const seqByProblem = new Map();
    for (const a of recent) {
        if (!a.problemId) continue;
        let seq = seqByProblem.get(a.problemId);
        if (!seq) { seq = []; seqByProblem.set(a.problemId, seq); }
        seq.push({ t: a.date, success: !!a.correct, repeat: !!a.isRepeat });
    }
    const recallGaps = [];
    for (const seq of seqByProblem.values()) {
        if (seq.length < 2) continue;
        seq.sort((x, y) => x.t - y.t);
        // A leading repeat means the novel attempt predates the 7-day window —
        // that problem contributes nothing (same rule as the Worker).
        const ni = seq.findIndex(e => !e.repeat);
        if (ni < 0 || ni + 1 >= seq.length) continue;
        if (seq[ni].success) continue;               // recall after a success — skip
        const gap = seq[ni + 1].t - seq[ni].t;
        if (gap >= RECALL_MIN_MS) recallGaps.push(gap);
    }
    const recallMed = median(recallGaps);
    if (recallMed != null) out.recall = RECALL_HI.findIndex(hi => recallMed < hi);

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

// Kernel-weighted median line + a weighted-quantile spread band on a regular
// t-grid (the Time chart's solving/review smoother, buildTimeUsageChart).
// `pts` is [{t, v}] ascending by t; the grid spans [t0, t0 + tSpan] at
// `gridPoints` points with kernel bandwidth `bw`. Returns an array of
// {t, mid, lo, hi}, with null at grid points with too little nearby data
// (those break the line), or null outright when the series is too short.
//
// The chart's series is broad (repeats included, no AFK cap); the window is
// ±3.5·bw via binary search on a caller-chosen (span-relative) bandwidth; thin
// data yields a GAP (null grid point) via the ESS gate; the output is three raw
// quantiles (25/50/75%) in value space. (The rank-tab time-adjust median is a
// different, positional-window computation — windowMedianSolveMinutesSeries in
// rank-decomposition.js — matched to the rank fit, not this chart smoother.)
export function kernelQuantileGrid(pts, { t0, tSpan, bw, gridPoints, essMin }) {
    if (pts.length < 4) return null;
    const ts = pts.map(p => p.t);
    const grid = new Array(gridPoints);
    for (let g = 0; g < gridPoints; g++) {
        const t = t0 + tSpan * (g / (gridPoints - 1));
        const i0 = lowerBound(ts, t - 3.5 * bw);
        const i1 = lowerBound(ts, t + 3.5 * bw);
        // Sum over EVERY sample in the window (no thinning): both the centre
        // line and the band are kernel-weighted quantiles of the same window.
        // Using the full window keeps them smooth functions of t — strided
        // subsampling here would reintroduce a sawtooth.
        let W = 0, sumW2 = 0;
        const pairs = [];
        for (let i = i0; i < i1; i++) {
            const d = (t - ts[i]) / bw;
            const w = gaussianWeight(d);
            W += w; sumW2 += w * w;
            pairs.push({ v: pts[i].v, w });
        }
        const ess = W > 0 ? (W * W) / sumW2 : 0; // effective sample size near t
        if (ess < essMin) { grid[g] = null; continue; }
        pairs.sort((a, b) => a.v - b.v);
        // Centre line = kernel-weighted MEDIAN (not the mean): time usage is
        // heavily right-skewed (idle-tab outliers), so the mean is dragged up
        // by the tail and isn't a typical day. The median is, and it always
        // sits inside the band. Band = 25%/75% edges = the middle 50% of
        // attempts (the interquartile range, read off the EMPIRICAL skewed
        // distribution — no normal assumption).
        grid[g] = {
            t,
            mid: weightedQuantile(pairs, W, 0.5),
            lo: weightedQuantile(pairs, W, 0.25),
            hi: weightedQuantile(pairs, W, 0.75),
        };
    }
    return grid;
}

// P(θ > 1/2) for θ ~ Beta(1 + pos, 1 + neg) — the exact posterior direction
// probability of a sign test under a uniform prior on the direction rate.
// Via the identity I_x(a, b) = P(Bin(a+b−1, x) ≥ a) for integer a, b, this is
// the binomial lower tail P(Bin(n+1, 1/2) ≤ pos) with n = pos + neg, computed
// by the stable term recurrence term_j = term_{j−1}·(N−j+1)/j starting from
// term_0 = 2^{−N}. 2^{−N} stays a normal double for N ≲ 1000, far above the
// ≤ MODEL_INPUT_HISTORY_LENGTH wrong attempts this ever sees.
export function betaProbAboveHalf(pos, neg) {
    const N = pos + neg + 1;
    let term = Math.pow(0.5, N);
    let sum = term;
    for (let j = 1; j <= pos; j++) {
        term *= (N - j + 1) / j;
        sum += term;
    }
    return Math.min(1, Math.max(0, sum));
}

// Level-2 direction statistic for the Weaknesses pane: for each level-1-flagged
// axis, a paired sign test on the (picked − correct) axis loading over the
// user's own wrong attempts. This needs no model and no assumption about which
// moves are reasonable alternatives — every problem has exactly one canonical
// answer (the generator's single-solution invariant), so the contrast is
// defined for every wrong attempt whose picked-move embedding was fetched.
// A predominantly POSITIVE contrast means the user's errors overshoot the
// motif (they reach for it when it isn't called for); predominantly NEGATIVE
// means they miss it (the correct move loads on it, their pick doesn't).
//
// The flagged axes co-load on the same problems, so each wrong attempt is
// attributed to the single flagged axis where its |contrast| is largest —
// without attribution one error would count toward every correlated axis at
// once and the per-axis counts would not be independent evidence.
//
// The attempt window mirrors the weakness.js training filter + slice so level
// 2 reads the same era of attempts level 1 was trained on.
//
// Returns Map k → {nPos, nNeg, pOver} with pOver = betaProbAboveHalf(nPos, nNeg).
export function computeAxisDirections(history, flaggedKs) {
    const out = new Map();
    if (!Array.isArray(flaggedKs) || flaggedKs.length === 0) return out;
    for (const k of flaggedKs) out.set(k, { nPos: 0, nNeg: 0, pOver: 0.5 });
    const rows = history
        .filter(a => a && !a.isRepeat &&
                     a.embedding && Array.isArray(a.embedding) &&
                     typeof a.problemRank === 'number' &&
                     typeof a.correct === 'boolean')
        .slice(-MODEL_INPUT_HISTORY_LENGTH);
    for (const a of rows) {
        if (a.correct) continue;
        if (!Array.isArray(a.pickedEmbedding)) continue;
        let bestK = -1, bestAbs = 0, bestD = 0;
        for (const k of flaggedKs) {
            if (a.embedding.length <= k || a.pickedEmbedding.length <= k) continue;
            const ec = a.embedding[k];
            const ep = a.pickedEmbedding[k];
            if (!Number.isFinite(ec) || !Number.isFinite(ep)) continue;
            const d = ep - ec;
            if (Math.abs(d) > bestAbs) { bestAbs = Math.abs(d); bestK = k; bestD = d; }
        }
        if (bestK < 0) continue;
        const s = out.get(bestK);
        if (bestD > 0) s.nPos++; else s.nNeg++;
    }
    for (const s of out.values()) s.pOver = betaProbAboveHalf(s.nPos, s.nNeg);
    return out;
}

// Picks wrong attempts that illustrate a surfaced (axis k, category) issue,
// mirroring the two-level detection in stats-weaknesses.js:
//   badAt (weak at the motif, direction inconclusive): the correct move loads
//     on the axis (ec > t) and the user failed — the level-1 evidence shape.
//   tooMuch (overplays): the picked move loads HIGHER than the correct one
//     (d = ep − ec > t) — the user reached for the motif when it wasn't
//     called for.
//   notEnough (misses): the correct move loads higher than the pick
//     (−d > t) — the motif was called for and the user played away from it.
//
// The minimum evidence per attempt scales with the issue's level-1 posterior
// probability via `threshold = probit(p)` — so a 90%-confident issue needs
// ~1.28σ-strong evidence per attempt, while a threshold-confidence issue
// collapses toward the pure sign constraint. Without this scaling, a
// borderline issue and a 95%-certain one would surface the same attempts,
// even though the latter should be backed by stronger per-attempt evidence.
// Embeddings are unit-variance per component (skew-ICA standardized), so the
// gate and the score are directly interpretable on the σ scale.
// `sign` (+1/−1) selects which TAIL of the bipolar axis the issue names:
// −1 mirrors the axis so a negative-tail issue ("weak at the negative-pole
// motif") reads its evidence with ec/d inverted — the same three quadrant
// shapes then apply verbatim on the mirrored axis.
export function findExamplesForIssue(answerHistory, k, quadrant, excludedIds, threshold = 0, sign = 1) {
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
        const ec = sign * a.embedding[k];
        const ep = sign * a.pickedEmbedding[k];
        if (!Number.isFinite(ec) || !Number.isFinite(ep)) continue;
        const d = ep - ec;
        let score;
        if (quadrant === 'tooMuch') {
            if (!(d > t)) continue;
            score = d;
        } else if (quadrant === 'notEnough') {
            if (!(-d > t)) continue;
            score = -d;
        } else {  // badAt
            if (!(ec > t)) continue;
            score = ec;
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
