// js/stats-compute.js
//
// Pure-numeric helpers consumed by the detailed-stats pane modules
// (stats-rank-pane.js, stats-time-chart.js, stats-weaknesses.js) and by the
// Statistics menu link — no DOM, no rendering. Split out of the stats UI so
// those modules are rendering-only and the numeric work (community-bucket
// mirroring, the kernel smoothers, the weakness-example selector) sits with
// the rest of the model math.

import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { rankToLabel, rankBand, clampRankBand } from '../shared/rank-math.js';
import { RANK_START_MIN } from '../shared/rank-window.js';
import { TIME_BINS } from '../shared/community-bins.js';
import { normCdf, weightedQuantile, gaussianWeight, lowerBound } from './math-utils.js';
import { SHIPPED_K } from './data-version.js';

// The URL fragment the app's Statistics menu link carries so the standalone
// /statistics page (src/handlers/statistics-page.js) can mark the reader's own
// bar in each community distribution: the bin indices computeUserCommunityBuckets
// finds — only those it could determine — and the rank LABEL, keyed the way the
// page reads them. '' when nothing is determinable. A fragment never reaches
// the server, and a link shared without one shows the plain community data.
export function communityYouFragment(history, rankContext) {
    const you = computeUserCommunityBuckets(history, TIME_BINS, rankContext);
    const p = new URLSearchParams();
    for (const [key, field] of [['apd', 'attemptsPerDay'], ['total', 'totalAttempts'], ['solve', 'solve'], ['review', 'review'], ['recall', 'recall']]) {
        if (you[field] >= 0) p.set(key, String(you[field]));
    }
    if (you.rankLabel != null) p.set('rank', you.rankLabel);
    return p.toString();
}

// Locate the current user within each community distribution, reproducing the
// Worker's per-user bucketing (computeServerStats in src/server-stats.js) on the
// client's own last-7-day merged history. Returns a bin index per bar-list
// category (-1 when not determinable) plus the rank label to mark.
// `timeBins` is the seconds-based display-bin table (TIME_BINS in
// shared/community-bins.js — the SAME table the Worker folds the page's bars
// into, so a median lands in the bin its bar is drawn in).
// `rankContext` carries the rank half, which is NOT derived here — see the rank
// block at the bottom — and `totalAttempts`, the account's lifetime attempt
// count as the server last echoed it (the sync state's lastSeq), which the
// local history cannot supply once its cold tail is evicted. On the rank half,
// `shownRank` is state.displayRank (the badge's own number) and
// `usableAttempts` is the flattened first-time count that gates it.
export function computeUserCommunityBuckets(history, timeBins, rankContext = {}) {
    const out = { attemptsPerDay: -1, totalAttempts: -1, solve: -1, review: -1, recall: -1, rankLabel: null };
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

    // Lifetime attempts, from the server's own count. Bins mirror TOTAL_BINS in
    // computeServerStats.
    const TOTAL_BINS = [[1, 100], [100, 300], [300, 1000], [1000, 3000], [3000, 10000], [10000, Infinity]];
    const { totalAttempts = null } = rankContext;
    if (Number.isFinite(totalAttempts)) {
        out.totalAttempts = TOTAL_BINS.findIndex(([lo, hi]) => totalAttempts >= lo && totalAttempts < hi);
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

    // Rank: the SHOWN rank (state.displayRank) banded exactly as the Worker bands
    // the bars — the histogram is one bin per active player's stored pred_rank,
    // which IS this same displayRank off the same window posterior
    // (computeLeaderboardStats, src/server-stats.js), so the marker and the bar it
    // sits in are one number, and one number with the front-page badge too.
    //
    // Do NOT go back to averaging the difficulty of the problems the user was
    // SERVED in the window. That proxy was retired on the Worker side and the same
    // reasons apply here: selection centres on serveRank (the estimate plus a
    // non-negative UCB bonus), not on the estimate, so an unconverged user reads
    // too strong against bars drawn in the display frame — which is the mean LESS
    // the display offset, i.e. the gap is the two offsets added, not cancelled;
    // it never sees whether the user actually solved anything; and weakness-regime
    // selection weights by weaknessScore rather than by proximity to the rank, so
    // the served mean is not an estimator of anything.
    //
    // Gated on the leaderboard's own RANK_START_MIN so the marker appears only for
    // a user who would BE one of the bars — below it the Worker stores no rank and
    // the player is absent from the histogram, while the client's displayRank getter
    // still answers (the cold-start prior), which would plant a "you" bar for someone
    // the distribution excludes. The 7-day early return above is the other half of
    // that population match: no attempt in the window, no leaderboard activity, no
    // marker.
    const { shownRank = null, usableAttempts = 0 } = rankContext;
    if (Number.isFinite(shownRank) && usableAttempts >= RANK_START_MIN) {
        out.rankLabel = rankToLabel(clampRankBand(rankBand(shownRank)));
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
        // Shape guard for the indexing below (provenance is settled by the wipe
        // — see motif-stats.js). Against the SHIPPED axis count rather than `k`,
        // so a vector of another length cannot be indexed at all.
        if (!Array.isArray(a.embedding) || a.embedding.length !== SHIPPED_K) continue;
        if (!Array.isArray(a.pickedEmbedding) || a.pickedEmbedding.length !== SHIPPED_K) continue;
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
