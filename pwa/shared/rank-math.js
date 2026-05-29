// shared/rank-math.js
//
// Rank-mapping primitives shared by BOTH the Cloudflare Worker (src/index.js,
// bundled by wrangler) and the client bundle (js/*, bundled by esbuild). These
// used to be copy-pasted into each side — the Worker's copy literally carried a
// "Mirrors interpExtrap in js/rank-translation.js" comment — which meant the
// rank math could silently drift between server-computed community stats and
// the client's rank display. One source of truth now.
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
 * Parse a kyu/dan label ("5k", "3d") to a rank scalar. Returns null on a
 * malformed label.
 */
export function rankLabelToScalar(label) {
    const m = /^(\d+)([kd])$/.exec(String(label || ''));
    if (!m) return null;
    const n = parseInt(m[1], 10);
    return m[2] === 'k' ? 1 - n : n;
}

/**
 * Rank scalar → nearest kyu/dan label. Rounds to the nearest band first, so a
 * continuous scalar maps to the closest whole rank.
 */
export function rankScalarToLabel(v) {
    const k = Math.round(v);
    return k <= 0 ? `${1 - k}k` : `${k}d`;
}
