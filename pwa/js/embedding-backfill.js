// js/embedding-backfill.js
//
// Retroactively fills the `embedding` field on legacy answerHistory entries
// (recorded before the per-component recommender shipped, or before the most
// recent PCA refit). Lets long-time users get the weakness model trained
// immediately instead of waiting 20+ new attempts.
//
// Runs for every supported size (not just the active one) so a size switch
// doesn't require an extra reload to re-trigger backfill. Each size has its
// own flag in localStorage to avoid redundant work.
//
// Invalidation is automatic: the server publishes an `embedding_version`
// content-hash in tsumegos/<size>/embedding_meta.json (computed in
// pipeline/build_embedding.py from the model state). The PWA compares it to
// the version stored in localStorage; on mismatch, old embeddings are wiped
// and refilled. No manual version bump needed when re-fitting the PCA.

import * as Sentry from "@sentry/browser";
import {
    state,
    getAnswerHistoryForSize, setAnswerHistoryForSize,
    setWeaknessParamsForSize, setWeaknessAdamForSize,
    SUPPORTED_BOARD_SIZES,
} from './state.js';
import { trainWeaknessForSize } from './weakness.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { fetchPickedEmbeddingWithIndexEntry } from './picked-embedding.js';

// Only the last N answerHistory entries need embeddings — that's all the
// trainer ever uses. Anything older eats localStorage for no benefit.
const KEEP_EMBEDDING_LAST = MODEL_INPUT_HISTORY_LENGTH;

const VERSION_KEY = (size) => `weaknessEmbeddingVersion_${size}`;
// Old per-version flag keys; cleared on first run after the upgrade so they
// don't linger in users' localStorage. The new scheme stores a content-hash
// version string under VERSION_KEY instead.
const STALE_FLAG_KEYS = (size) => [
    `weaknessEmbeddingBackfill_v1_${size}`,
    `weaknessEmbeddingBackfill_v2_${size}`,
    `weaknessEmbeddingBackfill_v3_${size}`,
];

export async function backfillMissingEmbeddings() {
    // Joint metadata (version + labels + sidecar layout) is shared across
    // sizes because the PCA + ICA fit is joint. Fetch once, then iterate
    // sizes for the per-size wipe/refill logic. The same fetch populates
    // state.embeddingMeta with the K/dtype/bytes_per_cell tuple used by
    // picked-embedding.js to decode per-cell .embeddings.bin sidecars.
    let serverVersion = 'unknown';
    try {
        const r = await fetch(`tsumegos/embedding_meta.json?v=${APP_VERSION}`);
        if (r.ok) {
            const meta = await r.json();
            if (meta && typeof meta.embedding_version === 'string') {
                serverVersion = meta.embedding_version;
            }
            if (meta && meta.labels && meta.labels.components) {
                state.embeddingLabels = meta.labels.components;
            }
            if (meta && typeof meta.K === 'number') {
                state.embeddingMeta = {
                    K: meta.K,
                    dtype: typeof meta.dtype === 'string' ? meta.dtype : 'float16',
                    bytes_per_cell: typeof meta.bytes_per_cell === 'number'
                        ? meta.bytes_per_cell
                        : meta.K * 2,  // sensible fp16 default
                    embedding_version: serverVersion,
                };
            }
        }
    } catch (_) { /* fall through to 'unknown' */ }

    const sizesFilled = [];
    for (const size of SUPPORTED_BOARD_SIZES) {
        const filled = await backfillForSize(size, serverVersion);
        if (filled > 0) sizesFilled.push(size);
    }
    // Train each size that got freshly populated, so the recommender is
    // calibrated for whichever size the user lands on.
    for (const size of sizesFilled) {
        setTimeout(() => trainWeaknessForSize(size), 0);
    }
}

async function backfillForSize(size, serverVersion) {
    const storedVersion = localStorage.getItem(VERSION_KEY(size));
    if (storedVersion === serverVersion) return 0;

    try {
        const history = getAnswerHistoryForSize(size);

        // Wipe any stale embeddings + the model trained on them. Idempotent
        // when the user is fresh.
        let cleared = 0;
        for (const a of history) {
            if (a.embedding) {
                delete a.embedding;
                cleared++;
            }
        }
        if (cleared > 0) {
            setWeaknessParamsForSize(size, null);
            setWeaknessAdamForSize(size, null);
            setAnswerHistoryForSize(size, history);
            console.log(`embedding-backfill[${size}]: cleared ${cleared} stale embeddings (${storedVersion || 'first run'} -> ${serverVersion})`);
        }
        for (const k of STALE_FLAG_KEYS(size)) localStorage.removeItem(k);

        if (!history || history.length === 0) {
            localStorage.setItem(VERSION_KEY(size), serverVersion);
            return 0;
        }

        // Restrict backfill to the last N entries; older history doesn't
        // need embeddings for training and storing them blows localStorage.
        const startIdx = Math.max(0, history.length - KEEP_EMBEDDING_LAST);
        const needIds = new Set();
        for (let i = startIdx; i < history.length; i++) {
            const a = history[i];
            // Pull a shard if either embedding (correct or picked) is missing —
            // we visit each shard once and fill whatever we can.
            const needsCorrect = !a.embedding;
            const needsPicked = (
                a.correct === false &&
                typeof a.pickedCellIndex === 'number' &&
                !a.pickedEmbedding
            );
            if (a.problemId && (needsCorrect || needsPicked)) needIds.add(a.problemId);
        }
        // Also strip stray embeddings from entries older than the window —
        // they may have been written by an earlier version that backfilled
        // everything. Includes pickedEmbedding + pickedCellIndex so the trim
        // is symmetric with the correct-move backfill.
        let strippedOld = 0;
        for (let i = 0; i < startIdx; i++) {
            const a = history[i];
            if (!a) continue;
            let touched = false;
            if (a.embedding !== undefined) { delete a.embedding; touched = true; }
            if (a.pickedEmbedding !== undefined) { delete a.pickedEmbedding; touched = true; }
            if (a.pickedCellIndex !== undefined) { delete a.pickedCellIndex; touched = true; }
            if (touched) strippedOld++;
        }
        if (strippedOld > 0) {
            setAnswerHistoryForSize(size, history);
            console.log(`embedding-backfill[${size}]: stripped embedding fields from ${strippedOld} old entries (beyond keep-last ${KEEP_EMBEDDING_LAST})`);
        }
        if (needIds.size === 0) {
            localStorage.setItem(VERSION_KEY(size), serverVersion);
            return 0;
        }

        // Read lookup.txt and pick rows matching this size.
        const lookupResp = await fetch(`lookup.txt?v=${APP_VERSION}`);
        if (!lookupResp.ok) return 0;
        const text = await lookupResp.text();
        const lookupMap = new Map();
        for (const line of text.split('\n')) {
            const parts = line.trim().split(/\s+/);
            if (parts.length >= 3 && parseInt(parts[2]) === size) {
                lookupMap.set(parts[0], parts[1]);
            }
        }

        // Map zipHash -> zipFilename via per-size metadata.json.
        const metaResp = await fetch(`tsumegos/${size}/metadata.json?v=${APP_VERSION}`);
        if (!metaResp.ok) return 0;
        const zipMeta = await metaResp.json();
        const zipFilenameByHash = new Map();
        for (const z of zipMeta) {
            zipFilenameByHash.set(z.filename.split('.')[0], z.filename);
        }

        const zipToIds = new Map();
        for (const id of needIds) {
            const zipHash = lookupMap.get(id);
            if (!zipHash) continue;
            if (!zipToIds.has(zipHash)) zipToIds.set(zipHash, []);
            zipToIds.get(zipHash).push(id);
        }

        let totalFilled = 0;
        let totalFilledPicked = 0;
        for (const [zipHash, _ids] of zipToIds) {
            const zipFilename = zipFilenameByHash.get(zipHash);
            if (!zipFilename) continue;
            try {
                const resp = await fetch(`tsumegos/${size}/${zipFilename}`);
                if (!resp.ok) continue;
                const buf = await resp.arrayBuffer();
                const zip = await new JSZip().loadAsync(buf);
                const metaFile = zip.file('metadata.json');
                if (!metaFile) continue;
                const items = JSON.parse(await metaFile.async('text'));
                const embeddingByProblem = new Map();
                for (const item of items) {
                    if (Array.isArray(item.embedding)) {
                        // Quantize to 2 decimals to keep localStorage small.
                        embeddingByProblem.set(
                            item.filename.replace(/\.sgf$/, ''),
                            item.embedding.map(v => Math.round(v * 100) / 100),
                        );
                    }
                }
                for (let i = startIdx; i < history.length; i++) {
                    const a = history[i];
                    if (!a.embedding && embeddingByProblem.has(a.problemId)) {
                        a.embedding = embeddingByProblem.get(a.problemId);
                        totalFilled++;
                    }
                }

                // Picked-move backfill: same shard, Range-fetched per-cell
                // sidecar. Skipped silently for shards that don't ship one
                // (older builds) or when there are no eligible attempts.
                const idxFile = zip.file('embeddings_index.json');
                if (idxFile) {
                    let indexPayload = null;
                    try { indexPayload = JSON.parse(await idxFile.async('text')); }
                    catch (e) {
                        console.warn(`embedding-backfill[${size}]: parse embeddings_index.json failed for ${zipFilename}:`, e);
                    }
                    if (indexPayload && indexPayload.problems) {
                        // Sequential per-entry on purpose: Range requests are
                        // cheap (~256B), and a sequential loop within a shard
                        // keeps the simultaneous-request count predictable —
                        // Cloudflare's free-plan rate limits are forgiving but
                        // not unlimited.
                        for (let i = startIdx; i < history.length; i++) {
                            const a = history[i];
                            if (a.correct !== false) continue;
                            if (typeof a.pickedCellIndex !== 'number') continue;
                            if (a.pickedEmbedding) continue;
                            const indexEntry = indexPayload.problems[a.problemId];
                            if (!indexEntry) continue;
                            const emb = await fetchPickedEmbeddingWithIndexEntry({
                                tsumegoId: a.problemId,
                                cellIndex: a.pickedCellIndex,
                                zipFilename,
                                boardSize: size,
                                indexEntry,
                            });
                            if (Array.isArray(emb)) {
                                a.pickedEmbedding = emb.map(v => Math.round(v * 100) / 100);
                                totalFilledPicked++;
                            }
                        }
                    }
                }
            } catch (err) {
                console.warn(`embedding-backfill[${size}]: failed for ${zipFilename}:`, err);
                continue;
            }
        }

        if (totalFilled > 0 || totalFilledPicked > 0) {
            setAnswerHistoryForSize(size, history);
            console.log(`embedding-backfill[${size}]: filled correct=${totalFilled} picked=${totalFilledPicked}`);
        }
        localStorage.setItem(VERSION_KEY(size), serverVersion);
        return totalFilled + totalFilledPicked;
    } catch (err) {
        console.error(`embedding-backfill[${size}] failed`, err);
        try { Sentry.captureException(err); } catch (_) {}
        return 0;
    }
}
