// js/picked-embedding.js
//
// Fetch the K-dim embedding for a specific (tsumego, picked-move) pair from
// the per-shard .embeddings.bin sidecar. Used on wrong attempts so the answer
// history records both what the user *should* have played (the correct-move
// embedding in metadata.json) and what they actually picked.
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

// Internal: Range-fetch + decode given an already-resolved index entry. Used
// by both the live (game-logic) and backfill paths so they share one network
// + decode codepath.
async function fetchCellEmbeddingFromIndexEntry({
    boardSize, zipFilename, indexEntry, cellIndex, meta,
}) {
    if (typeof cellIndex !== 'number' || cellIndex < 0) return null;
    if (cellIndex >= indexEntry.n_cells) {
        console.warn(`picked-embedding: cellIndex ${cellIndex} >= n_cells ${indexEntry.n_cells}`);
        return null;
    }
    const byteStart = indexEntry.offset + cellIndex * meta.bytes_per_cell;
    const byteEnd = byteStart + meta.bytes_per_cell - 1;
    const hash = zipFilename.replace(/\.zip$/, '');
    const url = `tsumegos/${boardSize}/${hash}.embeddings.bin`;
    try {
        const resp = await fetch(url, {
            headers: { Range: `bytes=${byteStart}-${byteEnd}` },
        });
        // 206 Partial Content is the success status for Range. A host that
        // ignores Range (wrangler dev's asset server does) serves the whole
        // file with 200 — slice our cell out of it rather than decoding the
        // first cell of the bin.
        if (!resp.ok && resp.status !== 206) {
            return null;
        }
        let buf = await resp.arrayBuffer();
        if (resp.status === 200 && buf.byteLength > meta.bytes_per_cell) {
            if (byteStart + meta.bytes_per_cell > buf.byteLength) return null;
            buf = buf.slice(byteStart, byteStart + meta.bytes_per_cell);
        }
        return decodeFp16Buffer(buf, meta.K);
    } catch (_err) {
        return null;
    }
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
// in-memory embeddings_index (loaded alongside metadata.json by problem-loader)
// and Range-fetches the picked cell. Returns Promise<Array<number> | null>;
// never throws.
export async function fetchPickedEmbedding({
    tsumegoId,
    cellIndex,         // SGF variation index from handleBoardClick (children.indexOf)
    zipFilename,       // e.g. "00be9212.zip"; sidecar at "00be9212.embeddings.bin"
    boardSize,         // for URL composition: tsumegos/<size>/<zipFilename>
}) {
    const meta = getMeta();
    if (!meta) return null;
    const indexPayload = embeddingsIndexCache.get(zipFilename);
    if (!indexPayload || !indexPayload.problems) return null;
    const indexEntry = indexPayload.problems[tsumegoId];
    if (!indexEntry) return null;
    return fetchCellEmbeddingFromIndexEntry({
        boardSize, zipFilename, indexEntry, cellIndex, meta,
    });
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
        const body = await resp.arrayBuffer();
        const m = /\/(\d+)\s*$/.exec(resp.headers.get('Content-Range') || '');
        const total = resp.status === 206 ? (m ? parseInt(m[1]) : null) : body.byteLength;
        if (Number.isFinite(expectedRows) && Number.isFinite(total) && total !== expectedRows * bpc) {
            console.warn('picked-embedding: lookup.embeddings.bin size mismatches lookup.txt — skipping correct-move fills this session');
            correctSidecarBroken = true;
            return null;
        }
        if (resp.status === 200 && body.byteLength > bpc) {
            // Host ignored Range (wrangler dev's asset server does) — keep the
            // whole file and serve every later row from memory.
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
// entry directly — no cache dependency.
export async function fetchPickedEmbeddingWithIndexEntry({
    tsumegoId,
    cellIndex,
    zipFilename,
    boardSize,
    indexEntry,
}) {
    const meta = getMeta();
    if (!meta || !indexEntry) return null;
    return fetchCellEmbeddingFromIndexEntry({
        boardSize, zipFilename, indexEntry, cellIndex, meta,
    });
}
