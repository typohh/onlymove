// shared/rank-math.js
//
// Rank-mapping primitives shared by BOTH the Cloudflare Worker (src/index.js,
// bundled by wrangler) and the client bundle (js/*, bundled by esbuild). One
// source of truth, so the rank math can't drift between server-computed community
// stats and the client's rank display.
//
// THE rank convention, AGA-style, used by every internal rank in the project
// (client, Worker and pipeline alike): 0 is the 1k/1d boundary, every rank is one
// unit wide, and integers are band EDGES. So 1d = [0, 1), 2d = [1, 2), 1k =
// [-1, 0), 2k = [-2, -1) — "2 dan" averages 1.5. There is no zero gap and there is
// no second scale: the normalized (rating·29) and rank-index (1k = 19) coordinates
// that used to coexist with this one were collapsed into it.
//
// AGA-STYLE, not AGA. Real AGA ratings carry a (-1, 1) gap and put 2d at [2, 3);
// converting to that frame is a real conversion and lives in the AGA display path
// (agaScalarToRating / agaPctRank in src/handlers/leaderboard-page.js).

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
 * Rank → its kyu/dan label. The band is the FLOOR (integers are band edges), so
 * 0 → "1d", -0.001 → "1k", 2.5 → "3d", -5.5 → "6k".
 */
export function rankToLabel(v) {
    const b = rankBand(v);
    return b < 0 ? `${-b}k` : `${b + 1}d`;
}

/**
 * Kyu/dan label → the CENTRE of that band ("5k" → -4.5, "3d" → 2.5). Case-
 * insensitive, tolerates surrounding whitespace; returns null for anything
 * unparseable.
 *
 * The centre, not an edge, because callers use this to place a DECLARED label
 * somewhere on the scale and the centre is its best single representative. Code
 * that wants the band's lower edge subtracts 0.5 from this (the idiom the server
 * couplings in map_rating_to_rank use).
 */
export function rankLabelToRank(label) {
    if (typeof label !== 'string') return null;
    const m = /^\s*(\d{1,2})\s*([kd])\s*$/i.exec(label);
    if (!m) return null;
    const n = parseInt(m[1], 10);
    if (!Number.isFinite(n) || n < 1) return null;
    return (m[2].toLowerCase() === 'k' ? -n : n - 1) + 0.5;
}

/**
 * Which whole rank a continuous rank falls in. FLOOR, because integers are band
 * EDGES: rank band n spans [n, n+1), so 1 dan is band 0 and 1 kyu is band -1.
 * This is the ONE definition of "which rank is this"; every label + progress site
 * (client and Worker) goes through it, so the band math can't drift between
 * surfaces.
 */
export function rankBand(value) {
    return Math.floor(value);
}

// The real kyu/dan ladder as BANDS (rankBand values): 30 kyu .. 9 dan. Used ONLY
// to clamp a value before it is LABELLED, so an extreme extrapolation can't print
// an impossible "15 dan". Never clamp a rank you are going to COMPUTE with — the
// scale is deliberately unbounded past both ends (see rankForSize); a hidden clamp
// in the translation path once pinned every above-9d value to the 8d/9d border.
export const RANK_BAND_LO = -30;   // 30 kyu
export const RANK_BAND_HI = 8;     // 9 dan

export function clampRankBand(band) {
    return Math.max(RANK_BAND_LO, Math.min(RANK_BAND_HI, band));
}

/**
 * THE decimal-rank → { text rank, fraction } decomposition. Given a continuous
 * rank, returns its whole-rank label plus the percent of the way to the next rank,
 * measured from the band's LOWER edge — which under this convention IS the band
 * integer, so it reads 0% at the boundary just crossed. This is the same fraction
 * the GoR carries (EGF 2389 = "4d 39%").
 *
 * `labelFn` exists only so a caller can CLAMP the ladder before labelling (the
 * rank-translation modal does, so an extreme extrapolation can't print "15 dan" on
 * a server whose ladder tops out at 9d). It is not a scale selector — there is one
 * scale.
 *
 * The percent is FLOORED, not rounded: a value just under the next band would
 * otherwise display "100%" (of the current rank, at ≥ .995 progress), which reads
 * as a contradiction next to the current rank label. Floor caps the display at 99%.
 *
 * Returns { band, label, percent, nextLabel }. Use this everywhere a rank is shown
 * with its progress — do NOT re-derive the band/percent inline (that divergence was
 * the floor-vs-round half-rank bug).
 */
export function rankProgress(value, labelFn = rankToLabel) {
    const band = rankBand(value);
    return {
        band,
        label: labelFn(band),
        percent: Math.floor((value - band) * 100),
        nextLabel: labelFn(band + 1),
    };
}
