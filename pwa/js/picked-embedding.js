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
//     in the shard. embeddings_index.json (inside the zip) gives the per-
//     tsumego byte offset + n_cells.
//   - Layout parameters (K, dtype, bytes_per_cell) live in the joint
//     pwa/tsumegos/embedding_meta.json, mirrored into state.embeddingMeta
//     by backfill.js.
//
// The cell-index input is the **SGF root-variation index** as observed by
// game-logic.handleBoardClick (children.indexOf(moveNode)). Extraction +
// packaging emit cells in the same SGF document order, so no recomputation
// of legality or coordinate flips is required client-side.

import { state, embeddingsIndexCache } from './state.js';

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
        // 206 Partial Content is the success status for Range. Some hosts may
        // serve 200 with the full body if Range is ignored — both arrayBuffer
        // decodes to the same K floats off the front; we accept either.
        if (!resp.ok && resp.status !== 206) {
            return null;
        }
        const buf = await resp.arrayBuffer();
        return decodeFp16Buffer(buf, meta.K);
    } catch (_err) {
        return null;
    }
}

function getMeta() {
    const meta = state.embeddingMeta;
    if (!meta || typeof meta.K !== 'number' || typeof meta.bytes_per_cell !== 'number') return null;
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
