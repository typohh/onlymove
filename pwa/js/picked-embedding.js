// js/picked-embedding.js
//
// Fetch a tsumego's per-cell K-dim embeddings from the per-shard
// .embeddings.bin sidecar. Used on wrong attempts so the answer history
// records what the user *should* have played (the correct-move embedding in
// metadata.json), what they actually picked, and the availability null — the
// direction a typical error on THIS problem would lean, weighed over the
// alternatives the board actually offered (computeAltStats below).
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
import { variationVisitWeights } from './sgf.js';

// One availability-null quantity is a little-endian uint32 (ALTSTATS_DTYPE in
// pipeline/package_for_pwa.py). A mass is (visits per variation) x (variations
// on the board), so it is not bounded by the cell count the way a count was.
const ALTSTATS_BYTES_PER_QUANTITY = 4;

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

// The availability null for one problem, from its bin block: per axis, the
// summed VISITS of the moves a click could pick that load strictly above /
// below the CORRECT move's loading (`correctVec`). Their ratio is the null
// probability that an error leans TOWARD the motif, which is what the tendency
// test compares the observed direction against.
//
// The quantities are MASSES, not counts. An alternative is weighed by the
// visits its own evaluation rests on (`weights[i]`, parallel to `rows`) rather
// than counted, because the moves a click could pick are not equally pickable
// and the search spent its depth where the position invited it.
//
// EXACT-AGREEMENT INVARIANT: for one problem this must return what
// `alt_stat_masses` (pipeline/package_for_pwa.py) writes into
// lookup.altstats.bin for it — not nearly, exactly. Both sides sum the same
// integer visit counts over the same quantized comparisons, so no rounding
// stands between them and the two producers can only agree or differ. That is
// what lets a live attempt's own quartet sit in the same window as one filled
// from the published sidecar. Which means the weights are not optional: a
// caller with no weights gets null rather than a count-shaped quartet, since a
// count and a mass are the same shape and mean different things.
//
// Ties carry no direction and are outside both masses, mirroring the
// observed-side tie drop — a tie drops WITH its weight rather than being split
// between the sides — which is also why the correct row needs no special
// exclusion: it ties with itself on every axis and falls out of the strict
// comparisons. The pass placeholder (all-zeros row) is skipped — a click cannot
// select the pass. Everything is compared in quantizeEmbedding's own
// resolution, so a stored pick compared against a stored correct vector
// resolves ties exactly as the masses here did.
export function computeAltStats(rows, correctVec, weights) {
    if (!Array.isArray(rows) || rows.length === 0) return null;
    if (!Array.isArray(correctVec) || correctVec.length === 0) return null;
    // A length disagreement means the weights belong to different variations
    // than the block rows — every weight would then be on the wrong cell while
    // both sides stayed individually well-formed, which is the one failure the
    // packaging producer also refuses rather than ships.
    if (!Array.isArray(weights) || weights.length !== rows.length) return null;
    if (!weights.every((w) => Number.isInteger(w) && w >= 0)) return null;
    const K = correctVec.length;
    const correctQ = quantizeEmbedding(correctVec);
    const altAbove = new Array(K).fill(0);
    const altBelow = new Array(K).fill(0);
    for (let r = 0; r < rows.length; r++) {
        const row = rows[r];
        if (!Array.isArray(row) || row.length !== K) continue;
        const q = quantizeEmbedding(row);
        if (q.every((v) => v === 0)) continue;   // pass placeholder
        const w = weights[r];
        for (let k = 0; k < K; k++) {
            if (q[k] > correctQ[k]) altAbove[k] += w;
            else if (q[k] < correctQ[k]) altBelow[k] += w;
        }
    }
    return { altAbove, altBelow };
}

function getMeta() {
    // state.embeddingMeta is populated by backfillMissingEmbeddings, which runs
    // AFTER the account-adoption synced fill — fall back to the build-time
    // inlined meta (data-version.js) so fills that run earlier in boot aren't
    // silently skipped. Same values either way; normalized here so every reader
    // below gets the same shape whichever source answered.
    const m = (state.embeddingMeta && typeof state.embeddingMeta.K === 'number')
        ? state.embeddingMeta : EMBEDDING_META_INLINE;
    if (!m || typeof m.K !== 'number' || m.K <= 0) return null;
    const dtype = typeof m.dtype === 'string' ? m.dtype : 'float16';
    if (dtype !== 'float16') {
        console.warn(`picked-embedding: unsupported dtype ${dtype}; skipping`);
        return null;
    }
    return {
        K: m.K,
        dtype,
        bytes_per_cell: typeof m.bytes_per_cell === 'number' ? m.bytes_per_cell : m.K * 2,
        // lookup.altstats.bin's row layout: K above-masses then K below-masses.
        // Read from the manifest rather than hardcoded so a width change is a
        // packaging fact and not a second constant to keep in step.
        altstats_bytes_per_row: typeof m.altstats_bytes_per_row === 'number'
            ? m.altstats_bytes_per_row
            : 2 * m.K * ALTSTATS_BYTES_PER_QUANTITY,
        altstats_dtype: typeof m.altstats_dtype === 'string' ? m.altstats_dtype : 'uint32',
    };
}

// Live path: called from game-logic on a wrong attempt. Looks up the shard's
// in-memory embeddings_index (loaded alongside metadata.json by problem-loader),
// Range-fetches the problem's whole bin block ONCE, and returns both what the
// motif statistics need from it: the picked cell's vector and the
// availability-null masses (computeAltStats, referenced to the correct cell's
// own row). Returns Promise<{ picked, alt } | null>; either half may be null;
// never throws.
//
// This is the one path that COMPUTES the null rather than reading the published
// one, and it may because the problem is on the board: `variationNodes` is the
// decision node's children, so the visit weights come off the same SGF comments
// packaging weighed, and the reference is the block's own row 0 exactly as
// packaging's is. Without them the null is not computed at all — a caller that
// cannot name the variations gets `alt: null` and the published masses fill the
// row later.
export async function fetchWrongAttemptStats({
    tsumegoId,
    pickedIndex,       // SGF variation index from handleBoardClick (children.indexOf)
    correctIndex,      // same index space, the answer node — the null's reference
    variationNodes,    // the decision node's children, SGF order — the weights' source
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
    const correctVec = (Number.isInteger(correctIndex) && correctIndex >= 0 && correctIndex < rows.length)
        ? rows[correctIndex] : null;
    const weights = variationVisitWeights(variationNodes);
    const alt = (correctVec && weights) ? computeAltStats(rows, correctVec, weights) : null;
    return { picked, alt };
}

// One ROW out of a lookup-aligned sidecar. Both files served this way —
// lookup.embeddings.bin (the correct-move vectors) and lookup.altstats.bin (the
// availability null) — are row i == lookup.txt line i, so a row is a fixed-width
// slice at rowIndex * bytesPerRow and a backfill never opens a shard zip for
// either. Their failure modes are identical too, which is why one reader serves
// both:
//
//   - a 206 for some OTHER range (a misbehaving cache/proxy) would make every
//     decoded row garbage, so it is rejected rather than mis-sliced;
//   - a host that ignores Range (wrangler dev's asset server; Cloudflare assets
//     on a cache miss) serves the whole file, which is then kept and every later
//     row read from memory — and the first request is held as a PROBE so such a
//     host costs ONE download rather than one per row;
//   - a byte length that is not `expectedRows * bytesPerRow` means the file and
//     lookup.txt came from different packaging runs, so every row address is
//     wrong. data_version cache-busting already pins them to the same run, so
//     this is a half-updated server mid-session: the file is poisoned for the
//     rest of it rather than retried.
//
// State (poison flag, whole-file copy, in-flight probe) is per file.
function makeSidecarRowReader(filename) {
    let broken = false;
    let whole = null;
    let probe = null;

    const rowFromWhole = (byteStart, bytesPerRow) => {
        if (!whole || byteStart + bytesPerRow > whole.byteLength) return null;
        return whole.slice(byteStart, byteStart + bytesPerRow);
    };

    const fetchRow = async (byteStart, bytesPerRow, expectedRows) => {
        const resp = await fetch(`tsumegos/${filename}?v=${DATA_VERSION}`, {
            headers: { Range: `bytes=${byteStart}-${byteStart + bytesPerRow - 1}` },
        });
        if (!resp.ok && resp.status !== 206) return null;
        const cr = resp.status === 206 ? parseContentRange(resp.headers.get('Content-Range')) : null;
        if (resp.status === 206 && (!cr || cr.start !== byteStart)) {
            console.warn(`picked-embedding: ${filename} 206 range mismatch (asked ${byteStart}, got ${cr ? cr.start : 'unparseable'})`);
            return null;
        }
        const body = await resp.arrayBuffer();
        const total = resp.status === 206 ? (cr ? cr.total : null) : body.byteLength;
        if (Number.isFinite(expectedRows) && Number.isFinite(total)
                && total !== expectedRows * bytesPerRow) {
            console.warn(`picked-embedding: ${filename} size mismatches lookup.txt — skipping its fills this session`);
            broken = true;
            return null;
        }
        if (resp.status === 200 && body.byteLength > bytesPerRow) {
            whole = body;
            return rowFromWhole(byteStart, bytesPerRow);
        }
        return body;
    };

    return async function readRow(rowIndex, bytesPerRow, expectedRows) {
        if (broken) return null;
        if (!Number.isInteger(rowIndex) || rowIndex < 0) return null;
        if (!Number.isInteger(bytesPerRow) || bytesPerRow <= 0) return null;
        const byteStart = rowIndex * bytesPerRow;
        try {
            if (whole) return rowFromWhole(byteStart, bytesPerRow);
            if (probe) {
                await probe;
                if (whole) return rowFromWhole(byteStart, bytesPerRow);
                return broken ? null : await fetchRow(byteStart, bytesPerRow, expectedRows);
            }
            const p = fetchRow(byteStart, bytesPerRow, expectedRows);
            probe = p.then(() => {}, () => {});
            return await p;
        } catch (_err) {
            return null;
        }
    };
}

const readCorrectRow = makeSidecarRowReader('lookup.embeddings.bin');
const readAltStatsRow = makeSidecarRowReader('lookup.altstats.bin');

// Correct-move sidecar: row i is the K-dim fp16 correct-cell embedding of
// lookup.txt line i (both sizes; emitted row-aligned by
// package_for_pwa.emit_correct_embeddings). `expectedRows` is the caller's
// lookup.txt line count. Returns null for a missing/absent row (an all-zeros row
// is the packaging placeholder for an id with no embedding).
export async function fetchCorrectEmbedding(rowIndex, expectedRows) {
    const meta = getMeta();
    if (!meta) return null;
    const buf = await readCorrectRow(rowIndex, meta.bytes_per_cell, expectedRows);
    if (!buf) return null;
    const vec = decodeFp16Buffer(buf, meta.K);
    if (!vec || vec.every((v) => v === 0)) return null;
    return vec;
}

// Availability-null sidecar: row i is lookup.txt line i's PUBLISHED masses — K
// little-endian uint32 above-masses then K below-masses, written by
// package_for_pwa.emit_alt_stats.
//
// This is what every path that does NOT have the problem's SGF in hand reads.
// The masses are integer sums of the variations' visit counts, referenced to row
// 0 of the problem's own per-cell block, so a row filled from here and a row the
// live path computed for itself are the same quartet rather than two nearly-equal
// ones — and rediscovering them per device would mean reading the whole per-cell
// block, hence the shard, for a fact that depends on nothing about the player.
//
// Returns null for the absent sentinel (an all-zero row, which packaging writes
// for an id whose shard carries no block and which already fails motif-stats'
// nAvail guard).
export async function fetchAltStatMasses(rowIndex, expectedRows) {
    const meta = getMeta();
    if (!meta) return null;
    if (meta.altstats_dtype !== 'uint32') {
        console.warn(`picked-embedding: unsupported altstats dtype ${meta.altstats_dtype}; skipping`);
        return null;
    }
    const bytesPerRow = meta.altstats_bytes_per_row;
    if (bytesPerRow !== 2 * meta.K * ALTSTATS_BYTES_PER_QUANTITY) {
        console.warn(`picked-embedding: altstats row width ${bytesPerRow} does not match K=${meta.K}; skipping`);
        return null;
    }
    const buf = await readAltStatsRow(rowIndex, bytesPerRow, expectedRows);
    if (!buf || buf.byteLength < bytesPerRow) return null;
    const dv = new DataView(buf);
    const altAbove = new Array(meta.K);
    const altBelow = new Array(meta.K);
    for (let k = 0; k < meta.K; k++) {
        altAbove[k] = dv.getUint32(k * ALTSTATS_BYTES_PER_QUANTITY, true);
        altBelow[k] = dv.getUint32((meta.K + k) * ALTSTATS_BYTES_PER_QUANTITY, true);
    }
    if (!altAbove.some((v, k) => v + altBelow[k] > 0)) return null;
    return { altAbove, altBelow };
}

// Backfill path: called from backfill.js when iterating shards. The
// caller has already parsed the shard's embeddings_index.json and passes the
// entry directly — no cache dependency. Returns the problem's whole bin block
// (Array<K-vector> in SGF variation order) so one fetch serves both the
// picked-cell refill and the availability-null counts.
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
