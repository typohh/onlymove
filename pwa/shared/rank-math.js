// shared/rank-math.js
//
// Rank-mapping primitives shared by BOTH the Cloudflare Worker (src/index.js,
// bundled by wrangler) and the client bundle (js/*, bundled by esbuild). One
// source of truth, so the rank math can't drift between server-computed community
// stats and the client's rank display.
//
// Rank-scalar convention (matches rank_map.json / rank-translation.js): 1k = 0,
// 1d = 1, with no zero gap. "5k" → -4, "3d" → 3.

/**
 * Linear interpolation with end-slope extrapolation. `xs` must be strictly
 * ascending with length ≥ 2.
 */
export function interpExtrap(x, xs, ys) {
    const n = xs.length;
    if (x <= xs[0]) return ys[0] + (ys[1] - ys[0]) / (xs[1] - xs[0]) * (x - xs[0]);
    if (x >= xs[n - 1]) return ys[n - 1] + (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]) * (x - xs[n - 1]);
    let a = 0, b = n - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (xs[m] <= x) a = m; else b = m; }
    return ys[a] + (ys[b] - ys[a]) * (x - xs[a]) / (xs[b] - xs[a]);
}

/**
 * Rank scalar → nearest kyu/dan label. Rounds to the nearest band first, so a
 * continuous scalar maps to the closest whole rank.
 */
export function rankScalarToLabel(v) {
    const k = Math.round(v);
    return k <= 0 ? `${1 - k}k` : `${k}d`;
}

// Internal rank INDEX → rank scalar. The index is the rating·TOTAL_RANKS scale
// where 1k = 19 (what getDescriptiveRank, the Rank chart, and the server histogram
// use); the scalar is the 1k = 0 convention rankScalarToLabel speaks, so
// index = scalar + RANK_INDEX_OFFSET.
export const RANK_INDEX_OFFSET = 19;

/**
 * Internal rank INDEX → kyu/dan label. Delegates to rankScalarToLabel so the
 * kyu/dan ladder lives in exactly one place. Callers pass an integer index.
 */
export function rankIndexToLabel(rankIndex) {
    return rankScalarToLabel(rankIndex - RANK_INDEX_OFFSET);
}

/**
 * Which whole rank a continuous rank falls in. ROUND, because the scalar/index is
 * the CENTER of each band (1 dan = scalar 1 = index 20), so rank n spans
 * [n-0.5, n+0.5) — matching the GoR/EGF convention (3d/4d boundary at GoR 2350).
 * This is the ONE definition of "which rank is this"; every label + progress site
 * (client and Worker) goes through it, so floor/round can't drift between surfaces.
 */
export function rankBand(value) {
    return Math.round(value);
}

/**
 * THE decimal-rank → { text rank, fraction } decomposition. Given a continuous
 * rank, returns its whole-rank label plus the percent of the way to the next rank,
 * measured from the band's LOWER edge (band − 0.5) so it reads 0% at the boundary
 * just crossed and 50% at the exact center — the same fraction the GoR carries
 * (EGF scalar 3.89 = GoR 2389 = "4d 39%").
 *
 * `value` may be a rank INDEX or a rank SCALAR; pass the matching `labelFn`
 * (rankIndexToLabel for index space, rankScalarToLabel for scalar space, or a
 * ladder-clamped server labeller). Both spaces share this band math because they
 * differ only by the constant RANK_INDEX_OFFSET and every band has width 1.
 *
 * Returns { band, label, percent, nextLabel }. Use this everywhere a rank is shown
 * with its progress — do NOT re-derive round/−0.5/percent inline (that divergence
 * was the floor-vs-round half-rank bug).
 */
export function rankProgress(value, labelFn = rankScalarToLabel) {
    const band = rankBand(value);
    return {
        band,
        label: labelFn(band),
        percent: Math.round((value - (band - 0.5)) * 100),
        nextLabel: labelFn(band + 1),
    };
}
