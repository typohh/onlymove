// js/picked-embedding.js
//
// Fetch a tsumego's per-cell K-dim embeddings from the per-shard
// .embeddings.bin sidecar. Used on wrong attempts so the answer history
// records what the user *should* have played (the correct-move embedding in
// metadata.json), what they actually picked, and the availability null — the
// motif profile a typical error on THIS problem would carry, computed from
// the alternatives the board actually offered (computeAltStats below).
//
// Wire protocol:
//   - .embeddings.bin holds raw fp16 little-endian floats, K per cell, with
//     cells concatenated in SGF variation document order across all problems
//     in the shard. The pass variation (never embedded upstream) occupies an
//     all-zeros placeholder row, so bin row i == SGF child index i exactly —
//     including for children after a non-last pass (bin format rev 2; rev 1
//     skipped the pass and was off by one there). The pass row itself is
//     never fetched: a board click can't select the pass node.
//     embeddings_index.json (inside the zip) gives the per-tsumego byte
//     offset + n_cells.
//   - Layout parameters (K, dtype, bytes_per_cell) live in the joint
//     pwa/tsumegos/embedding_meta.json, mirrored into state.embeddingMeta
//     by backfill.js.
//
// The cell-index input is the **SGF root-variation index** as observed by
// game-logic.handleBoardClick (children.indexOf(moveNode)). Packaging emits
// bin rows in the same SGF document order, so no recomputation of legality
// or coordinate flips is required client-side.

import { state, embeddingsIndexCache } from './state.js';
import { DATA_VERSION, EMBEDDING_META_INLINE } from './data-version.js';
import { quantizeEmbedding } from './math-utils.js';

// Convert one fp16 (uint16 representation) to a JS number, per IEEE 754 half-
// precision. JS doesn't ship a native Float16Array yet (Stage 3 proposal), and
// we only ever decode ~K=64 values per call, so a scalar helper is fine.
function fp16ToFloat32(u16) {
    const sign = (u16 & 0x8000) >> 15;
    const exp  = (u16 & 0x7C00) >> 10;
    const frac =  u16 & 0x03FF;
    if (exp === 0) {
        // subnormal: (-1)^s * 2^-14 * frac/1024
        const v = Math.pow(2, -14) * (frac / 1024);
        return sign ? -v : v;
    }
    if (exp === 0x1F) {
        if (frac === 0) return sign ? -Infinity : Infinity;
        return NaN;
    }
    // normal: (-1)^s * 2^(exp-15) * (1 + frac/1024)
    const v = Math.pow(2, exp - 15) * (1 + frac / 1024);
    return sign ? -v : v;
}

function decodeFp16Buffer(arrayBuffer, K) {
    const u16 = new Uint16Array(arrayBuffer);
    if (u16.length < K) return null;  // short read: bail
    const out = new Array(K);
    for (let i = 0; i < K; i++) out[i] = fp16ToFloat32(u16[i]);
    return out;
}

// Decode a whole problem's bin block into one K-vector per cell (SGF variation
// order, pass placeholder rows included). Null on a short read.
function decodeFp16Rows(arrayBuffer, K, nRows) {
    const u16 = new Uint16Array(arrayBuffer);
    if (u16.length < K * nRows) return null;
    const rows = new Array(nRows);
    for (let r = 0; r < nRows; r++) {
        const row = new Array(K);
        for (let i = 0; i < K; i++) row[i] = fp16ToFloat32(u16[r * K + i]);
        rows[r] = row;
    }
    return rows;
}

// Parse a Content-Range header ("bytes <start>-<end>/<total>"). Returns
// { start, end, total } with total null for the unknown form ("*"), or null
// when the header is absent/malformed.
function parseContentRange(header) {
    const m = /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(header || '');
    if (!m) return null;
    return {
        start: parseInt(m[1]),
        end: parseInt(m[2]),
        total: m[3] === '*' ? null : parseInt(m[3]),
    };
}

// Resolve a Range response to the requested slice, defensively:
//   - 206: the Content-Range start must match what we asked for — a
//     misbehaving cache/proxy answering with a DIFFERENT range would otherwise
//     silently decode some other cell's floats. Reject on mismatch or a
//     missing/malformed header.
//   - 200: the host ignored Range (wrangler dev's asset server; Cloudflare
//     assets on a cache miss) and served the whole file — slice our bytes out.
//   - Short reads fall through to decodeFp16Buffer's length check via null.
async function sliceRangeResponse(resp, byteStart, byteLength) {
    if (resp.status === 206) {
        const cr = parseContentRange(resp.headers.get('Content-Range'));
        if (!cr || cr.start !== byteStart) {
            console.warn(`picked-embedding: 206 range mismatch (asked ${byteStart}, got ${cr ? cr.start : 'unparseable'})`);
            return null;
        }
        return resp.arrayBuffer();
    }
    if (!resp.ok) return null;
    const buf = await resp.arrayBuffer();
    if (buf.byteLength > byteLength) {
        if (byteStart + byteLength > buf.byteLength) return null;
        return buf.slice(byteStart, byteStart + byteLength);
    }
    return buf;
}

// Internal: Range-fetch + decode a problem's WHOLE bin block (all variation
// cells) given an already-resolved index entry. One request per problem —
// n_cells x bytes_per_cell is under a kilobyte at K=5 — shared by the live
// (game-logic) and backfill paths. Returns Array<K-vector> in SGF variation
// order, or null.
async function fetchProblemCellsFromIndexEntry({
    boardSize, zipFilename, indexEntry, meta,
}) {
    const nCells = indexEntry.n_cells;
    if (!Number.isInteger(nCells) || nCells <= 0) return null;
    const byteLength = nCells * meta.bytes_per_cell;
    const byteStart = indexEntry.offset;
    const byteEnd = byteStart + byteLength - 1;
    const hash = zipFilename.replace(/\.zip$/, '');
    const url = `tsumegos/${boardSize}/${hash}.embeddings.bin`;
    try {
        const resp = await fetch(url, {
            headers: { Range: `bytes=${byteStart}-${byteEnd}` },
        });
        const buf = await sliceRangeResponse(resp, byteStart, byteLength);
        return buf ? decodeFp16Rows(buf, meta.K, nCells) : null;
    } catch (_err) {
        return null;
    }
}

// The uniform-over-available error null for one problem, from its bin block.
//
// The support is exactly the moves a click can pick: the problem's SGF
// variations minus the pass placeholder (all-zeros row — a click cannot select
// the pass) and minus the correct move itself. Per axis it returns, over those
// alternatives:
//
//   altMean/altSd   mean and (population) standard deviation of the loading —
//                   the null moments of a typical error's motif profile, which
//                   the judgment reading standardises each pick against.
//   altAbove/altBelow  how many alternatives load strictly above / below the
//                   CORRECT move's loading. Their ratio is the null probability
//                   that an error leans TOWARD the motif, which is what the
//                   tendency test compares the observed direction against —
//                   ties carry no direction and are outside both counts,
//                   mirroring the observed-side tie drop.
//
// Everything is computed in quantizeEmbedding's own resolution so a stored
// pick compared against a stored correct vector resolves ties exactly as the
// counts here did. The correct row is excluded by index when the caller knows
// it (live path) and by quantized-vector equality otherwise (backfill, where
// only the stored correct-move embedding is in hand); the equality rule also
// runs under a known index, which additionally drops any alternative that is
// indistinguishable from the answer at this resolution — a row that could
// carry no signal either way. Null when fewer than two alternatives remain.
export function computeAltStats(rows, { correctIndex = null, correctVec = null } = {}) {
    if (!Array.isArray(rows) || rows.length === 0) return null;
    const K = rows[0].length;
    const source = Array.isArray(correctVec) && correctVec.length === K
        ? correctVec
        : (Number.isInteger(correctIndex) && Array.isArray(rows[correctIndex])
            ? rows[correctIndex] : null);
    if (!source) return null;
    const correctQ = quantizeEmbedding(source);

    const alts = [];
    for (let i = 0; i < rows.length; i++) {
        if (i === correctIndex) continue;
        const q = quantizeEmbedding(rows[i]);
        if (q.length !== K) continue;
        if (q.every((v) => v === 0)) continue;                    // pass placeholder
        if (q.every((v, k) => v === correctQ[k])) continue;       // the correct cell
        alts.push(q);
    }
    if (alts.length < 2) return null;

    const altMean = new Array(K).fill(0);
    const altSd = new Array(K).fill(0);
    const altAbove = new Array(K).fill(0);
    const altBelow = new Array(K).fill(0);
    for (let k = 0; k < K; k++) {
        let s = 0;
        for (const a of alts) s += a[k];
        const m = s / alts.length;
        let ss = 0;
        for (const a of alts) {
            const d = a[k] - m;
            ss += d * d;
            if (a[k] > correctQ[k]) altAbove[k]++;
            else if (a[k] < correctQ[k]) altBelow[k]++;
        }
        altMean[k] = m;
        altSd[k] = Math.sqrt(ss / alts.length);
    }
    return {
        altMean: quantizeEmbedding(altMean),
        altSd: quantizeEmbedding(altSd),
        altAbove,
        altBelow,
    };
}

function getMeta() {
    let meta = state.embeddingMeta;
    if (!meta || typeof meta.K !== 'number' || typeof meta.bytes_per_cell !== 'number') {
        // state.embeddingMeta is populated by backfillMissingEmbeddings, which
        // runs AFTER the account-adoption synced fill — fall back to the
        // build-time inlined meta (data-version.js) so fills that run earlier
        // in boot aren't silently skipped. Same K/dtype/bytes_per_cell values.
        const m = EMBEDDING_META_INLINE;
        meta = (m && typeof m.K === 'number') ? {
            K: m.K,
            dtype: typeof m.dtype === 'string' ? m.dtype : 'float16',
            bytes_per_cell: typeof m.bytes_per_cell === 'number' ? m.bytes_per_cell : m.K * 2,
        } : null;
    }
    if (!meta) return null;
    if (meta.dtype && meta.dtype !== 'float16') {
        console.warn(`picked-embedding: unsupported dtype ${meta.dtype}; skipping`);
        return null;
    }
    return meta;
}

// Live path: called from game-logic on a wrong attempt. Looks up the shard's
// in-memory embeddings_index (loaded alongside metadata.json by problem-loader),
// Range-fetches the problem's whole bin block ONCE, and returns both what the
// motif statistics need from it: the picked cell's vector and the
// availability-null moments (computeAltStats). Returns
// Promise<{ picked, alt } | null>; either half may be null; never throws.
export async function fetchWrongAttemptStats({
    tsumegoId,
    pickedIndex,       // SGF variation index from handleBoardClick (children.indexOf)
    correctIndex,      // same index space, the answer node — excluded from the null
    zipFilename,       // e.g. "00be9212.zip"; sidecar at "00be9212.embeddings.bin"
    boardSize,         // for URL composition: tsumegos/<size>/<zipFilename>
}) {
    const meta = getMeta();
    if (!meta) return null;
    const indexPayload = embeddingsIndexCache.get(zipFilename);
    if (!indexPayload || !indexPayload.problems) return null;
    const indexEntry = indexPayload.problems[tsumegoId];
    if (!indexEntry) return null;
    const rows = await fetchProblemCellsFromIndexEntry({
        boardSize, zipFilename, indexEntry, meta,
    });
    if (!rows) return null;
    const picked = (Number.isInteger(pickedIndex) && pickedIndex >= 0 && pickedIndex < rows.length)
        ? rows[pickedIndex] : null;
    const alt = computeAltStats(rows, { correctIndex });
    return { picked, alt };
}

// Correct-move sidecar: tsumegos/lookup.embeddings.bin — row i is the K-dim
// fp16 correct-cell embedding of lookup.txt line i (both sizes; emitted
// row-aligned by package_for_pwa.emit_correct_embeddings). Range-fetches one
// row so backfills never need the shard zip for a correct-move embedding.
//
// `expectedRows` (the caller's lookup.txt line count) guards row alignment:
// the 206's Content-Range total must equal expectedRows * bytes_per_cell.
// data_version cache-busting already pins lookup.txt and the sidecar to the
// same packaging run, so a mismatch means a half-updated server mid-session —
// every row address would be garbage, so the sidecar is poisoned for the rest
// of the session rather than retried. Returns null for a missing/absent row
// (an all-zeros row is the packaging placeholder for an id with no embedding).
let correctSidecarBroken = false;
let correctSidecarFull = null;   // whole-file ArrayBuffer when the host ignores Range
let correctSidecarProbe = null;  // first request in flight; later calls wait on it
export async function fetchCorrectEmbedding(rowIndex, expectedRows) {
    if (correctSidecarBroken) return null;
    const meta = getMeta();
    if (!meta) return null;
    if (!Number.isInteger(rowIndex) || rowIndex < 0) return null;
    const bpc = meta.bytes_per_cell;
    const byteStart = rowIndex * bpc;

    const rowFromWhole = () => {
        if (!correctSidecarFull || byteStart + bpc > correctSidecarFull.byteLength) return null;
        return correctSidecarFull.slice(byteStart, byteStart + bpc);
    };

    const fetchRow = async () => {
        const resp = await fetch(`tsumegos/lookup.embeddings.bin?v=${DATA_VERSION}`, {
            headers: { Range: `bytes=${byteStart}-${byteStart + bpc - 1}` },
        });
        if (!resp.ok && resp.status !== 206) return null;
        const cr = resp.status === 206 ? parseContentRange(resp.headers.get('Content-Range')) : null;
        if (resp.status === 206 && (!cr || cr.start !== byteStart)) {
            // A 206 for some other range (misbehaving cache/proxy) would make
            // every decoded row garbage — reject rather than mis-slice.
            console.warn(`picked-embedding: sidecar 206 range mismatch (asked ${byteStart}, got ${cr ? cr.start : 'unparseable'})`);
            return null;
        }
        const body = await resp.arrayBuffer();
        // Row-alignment guard: the sidecar's byte length must equal lookup.txt's
        // line count x row size, else the two files are from different packaging
        // runs and every row address is wrong — poison it for the session.
        const total = resp.status === 206 ? (cr ? cr.total : null) : body.byteLength;
        if (Number.isFinite(expectedRows) && Number.isFinite(total) && total !== expectedRows * bpc) {
            console.warn('picked-embedding: lookup.embeddings.bin size mismatches lookup.txt — skipping correct-move fills this session');
            correctSidecarBroken = true;
            return null;
        }
        if (resp.status === 200 && body.byteLength > bpc) {
            // Host ignored Range (wrangler dev's asset server does; Cloudflare
            // assets on a cache miss) — keep the whole file and serve every
            // later row from memory.
            correctSidecarFull = body;
            return rowFromWhole();
        }
        return body;
    };

    try {
        let buf;
        if (correctSidecarFull) {
            buf = rowFromWhole();
        } else if (correctSidecarProbe) {
            // First request probes Range support; waiting on it means a
            // Range-less host costs ONE full download, not one per row.
            await correctSidecarProbe;
            buf = correctSidecarFull ? rowFromWhole()
                : (correctSidecarBroken ? null : await fetchRow());
        } else {
            const p = fetchRow();
            correctSidecarProbe = p.then(() => {}, () => {});
            buf = await p;
        }
        if (!buf) return null;
        const vec = decodeFp16Buffer(buf, meta.K);
        if (!vec || vec.every((v) => v === 0)) return null;
        return vec;
    } catch (_err) {
        return null;
    }
}

// Backfill path: called from backfill.js when iterating shards. The
// caller has already parsed the shard's embeddings_index.json and passes the
// entry directly — no cache dependency. Returns the problem's whole bin block
// (Array<K-vector> in SGF variation order) so one fetch serves both the
// picked-cell refill and the availability-null moments.
export async function fetchProblemCellsWithIndexEntry({
    zipFilename,
    boardSize,
    indexEntry,
}) {
    const meta = getMeta();
    if (!meta || !indexEntry) return null;
    return fetchProblemCellsFromIndexEntry({
        boardSize, zipFilename, indexEntry, meta,
    });
}
