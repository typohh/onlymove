// js/embedding-backfill.js
//
// Retroactively patches per-shard fields on answerHistory entries. Two
// independent jobs share one shard walk:
//
//   1. Embedding fill (last MODEL_INPUT_HISTORY_LENGTH entries only)
//      Triggered by `embedding_version` change in embedding_meta.json.
//      Wipes any stale embeddings, then fills `embedding` + optional
//      `pickedEmbedding` for entries inside the keep window. Older
//      entries get their embedding fields stripped to keep localStorage
//      bounded.
//
//   2. Rating refresh (all history entries, no keep-window cap)
//      Triggered by per-size `ratings_version` change. Walks every shard
//      that contains a referenced problem and rewrites
//      `answerHistory[*].problemRating` to the current shard value. The
//      chart, weakness training, and recall logic all read this field,
//      so a stale rating shows up everywhere.
//
// Both versions are tracked separately in localStorage (one key each per
// size) so a rating-only refresh doesn't wipe the embedding model and
// vice-versa. The walk runs for every supported size on every app load;
// each size short-circuits when both versions match.

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
import { normalizedRankForSize, ratingRankCurveVersion } from './rank-decomposition.js';

// Only the last N answerHistory entries need embeddings — that's all the
// trainer ever uses. Anything older eats localStorage for no benefit.
const KEEP_EMBEDDING_LAST = MODEL_INPUT_HISTORY_LENGTH;

const VERSION_KEY = (size) => `weaknessEmbeddingVersion_${size}`;
const RATINGS_VERSION_KEY = (size) => `ratingsVersion_${size}`;
const RANK_CURVE_VERSION_KEY = (size) => `rankCurveVersion_${size}`;
// Old per-version flag keys; cleared on first run after the upgrade so they
// don't linger in users' localStorage. The new scheme stores a content-hash
// version string under VERSION_KEY instead.
const STALE_FLAG_KEYS = (size) => [
    `weaknessEmbeddingBackfill_v1_${size}`,
    `weaknessEmbeddingBackfill_v2_${size}`,
    `weaknessEmbeddingBackfill_v3_${size}`,
];

export async function backfillMissingEmbeddings() {
    // Joint metadata (embedding version + labels + sidecar layout + per-size
    // ratings_version) is shared across sizes. Fetch once, then iterate sizes
    // for per-size wipe/refill. Also populates state.embeddingMeta with the
    // K/dtype/bytes_per_cell tuple used by picked-embedding.js to decode
    // per-cell .embeddings.bin sidecars.
    let serverVersion = 'unknown';
    let ratingsVersionsBySize = {};
    try {
        const r = await fetch(`tsumegos/embedding_meta.json?v=${APP_VERSION}`);
        if (r.ok) {
            const meta = await r.json();
            if (meta && typeof meta.embedding_version === 'string') {
                serverVersion = meta.embedding_version;
            }
            if (meta && meta.ratings_version && typeof meta.ratings_version === 'object') {
                ratingsVersionsBySize = meta.ratings_version;
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
        // ratings_version is keyed by string in the JSON; tolerate either.
        const ratingsVersion = ratingsVersionsBySize[size]
            ?? ratingsVersionsBySize[String(size)]
            ?? null;
        const filled = await backfillForSize(size, serverVersion, ratingsVersion);
        if (filled > 0) sizesFilled.push(size);
    }
    // Train each size that got freshly populated, so the recommender is
    // calibrated for whichever size the user lands on.
    for (const size of sizesFilled) {
        setTimeout(() => trainWeaknessForSize(size), 0);
    }
}

async function backfillForSize(size, serverEmbeddingVersion, serverRatingsVersion) {
    const storedEmbeddingVersion = localStorage.getItem(VERSION_KEY(size));
    const storedRatingsVersion = localStorage.getItem(RATINGS_VERSION_KEY(size));
    const curveVersion = ratingRankCurveVersion(size);
    const storedCurveVersion = localStorage.getItem(RANK_CURVE_VERSION_KEY(size));
    const embeddingsChanged = storedEmbeddingVersion !== serverEmbeddingVersion;
    const ratingsChanged = serverRatingsVersion !== null
        && storedRatingsVersion !== serverRatingsVersion;
    // problemRank is derived from the raw rating through M, so a curve re-fit
    // that left ratings untouched still shifts every stored rank.
    const curveChanged = curveVersion !== null && storedCurveVersion !== curveVersion;
    if (!embeddingsChanged && !ratingsChanged && !curveChanged) return 0;

    const persistVersions = () => {
        if (embeddingsChanged) localStorage.setItem(VERSION_KEY(size), serverEmbeddingVersion);
        if (ratingsChanged) localStorage.setItem(RATINGS_VERSION_KEY(size), serverRatingsVersion);
        if (curveChanged) localStorage.setItem(RANK_CURVE_VERSION_KEY(size), curveVersion);
    };

    try {
        const history = getAnswerHistoryForSize(size);

        // Embedding wipe — only when embedding_version changed. Rating-only
        // refreshes keep the trained model intact.
        if (embeddingsChanged) {
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
                console.log(`embedding-backfill[${size}]: cleared ${cleared} stale embeddings (${storedEmbeddingVersion || 'first run'} -> ${serverEmbeddingVersion})`);
            }
            for (const k of STALE_FLAG_KEYS(size)) localStorage.removeItem(k);
        }

        if (!history || history.length === 0) {
            persistVersions();
            return 0;
        }

        // Curve-only refit: ranks shift but raw ratings don't, so no shard
        // fetch is needed — recompute problemRank straight from the stored raw
        // problemRating. When ratings also changed, the shard walk below
        // refreshes both, so skip this to avoid a double pass.
        let curveRankUpdated = 0;
        if (curveChanged && !ratingsChanged) {
            for (const a of history) {
                if (typeof a.problemRating !== 'number') continue;
                const newRank = normalizedRankForSize(a.problemRating, size);
                if (newRank !== null && a.problemRank !== newRank) {
                    a.problemRank = newRank;
                    curveRankUpdated++;
                }
            }
            if (curveRankUpdated > 0) {
                setAnswerHistoryForSize(size, history);
                console.log(`embedding-backfill[${size}]: recomputed problemRank for ${curveRankUpdated} entries (curve change)`);
            }
        }

        // Two needs sets feed one shard walk:
        //   - Embedding needs: last KEEP_EMBEDDING_LAST entries with missing fields
        //   - Rating needs:    every entry (no cap; the chart shows full history)
        const startIdx = Math.max(0, history.length - KEEP_EMBEDDING_LAST);
        const needIdsForEmbed = new Set();
        const needIdsForRatings = new Set();
        if (embeddingsChanged) {
            for (let i = startIdx; i < history.length; i++) {
                const a = history[i];
                const needsCorrect = !a.embedding;
                const needsPicked = (
                    a.correct === false &&
                    typeof a.pickedCellIndex === 'number' &&
                    !a.pickedEmbedding
                );
                if (a.problemId && (needsCorrect || needsPicked)) needIdsForEmbed.add(a.problemId);
            }
        }
        if (ratingsChanged) {
            for (let i = 0; i < history.length; i++) {
                if (history[i].problemId) needIdsForRatings.add(history[i].problemId);
            }
        }

        // Strip stray embeddings from entries older than the keep window —
        // only on embedding refits (this is a wipe, not a refill).
        if (embeddingsChanged) {
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
        }

        const allNeedIds = new Set([...needIdsForEmbed, ...needIdsForRatings]);
        if (allNeedIds.size === 0) {
            persistVersions();
            return curveRankUpdated;
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
        for (const id of allNeedIds) {
            const zipHash = lookupMap.get(id);
            if (!zipHash) continue;
            if (!zipToIds.has(zipHash)) zipToIds.set(zipHash, []);
            zipToIds.get(zipHash).push(id);
        }

        let totalFilled = 0;
        let totalFilledPicked = 0;
        let totalRatingsUpdated = 0;
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
                const ratingByProblem = new Map();
                for (const item of items) {
                    const tid = item.filename.replace(/\.sgf$/, '');
                    if (Array.isArray(item.embedding)) {
                        // Quantize to 2 decimals to keep localStorage small.
                        embeddingByProblem.set(tid, item.embedding.map(v => Math.round(v * 100) / 100));
                    }
                    if (typeof item.rating === 'number') {
                        ratingByProblem.set(tid, item.rating);
                    }
                }

                // Ratings: rewrite across the full history. Cheap (just a map
                // lookup per entry) and the chart needs old entries updated.
                if (ratingsChanged) {
                    for (let i = 0; i < history.length; i++) {
                        const a = history[i];
                        if (!a.problemId) continue;
                        if (!ratingByProblem.has(a.problemId)) continue;
                        const newRating = ratingByProblem.get(a.problemId);
                        if (a.problemRating !== newRating) {
                            a.problemRating = newRating;
                            totalRatingsUpdated++;
                        }
                        // Refresh the derived rank, and migrate legacy entries:
                        // problemRating now holds the raw value, so the separate
                        // problemRawRating field is redundant.
                        const newRank = normalizedRankForSize(newRating, size);
                        if (newRank !== null && a.problemRank !== newRank) a.problemRank = newRank;
                        if (a.problemRawRating !== undefined) delete a.problemRawRating;
                    }
                }

                if (embeddingsChanged) {
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
                }
            } catch (err) {
                console.warn(`embedding-backfill[${size}]: failed for ${zipFilename}:`, err);
                continue;
            }
        }

        if (totalFilled > 0 || totalFilledPicked > 0 || totalRatingsUpdated > 0) {
            setAnswerHistoryForSize(size, history);
            console.log(`embedding-backfill[${size}]: filled correct=${totalFilled} picked=${totalFilledPicked} ratings=${totalRatingsUpdated}`);
        }
        persistVersions();
        return totalFilled + totalFilledPicked + totalRatingsUpdated + curveRankUpdated;
    } catch (err) {
        console.error(`embedding-backfill[${size}] failed`, err);
        try { Sentry.captureException(err); } catch (_) {}
        return 0;
    }
}
