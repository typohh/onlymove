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
    SUPPORTED_BOARD_SIZES,
} from './state.js';
import { trainWeakness } from './weakness.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { fetchPickedEmbeddingWithIndexEntry } from './picked-embedding.js';
import { fetchLookupLines } from './problem-loader.js';
import { normalizedRankForSize, ratingRankCurveVersion, slopeToQuality } from './rank-decomposition.js';
import { boardNotationToXy } from './sgf.js';

// Only the last N answerHistory entries need embeddings — that's all the
// trainer ever uses. Anything older eats localStorage for no benefit.
const KEEP_EMBEDDING_LAST = MODEL_INPUT_HISTORY_LENGTH;

const VERSION_KEY = (size) => `weaknessEmbeddingVersion_${size}`;
const RATINGS_VERSION_KEY = (size) => `ratingsVersion_${size}`;
const RANK_CURVE_VERSION_KEY = (size) => `rankCurveVersion_${size}`;

// Removed-problem prune. Problems get dropped from the corpus on regeneration
// (e.g. a third of the Sep–Oct 2025 9x9 set is gone). Their answerHistory
// entries keep the rating/rank stored at attempt time — the shard no longer
// exists, so the rating-refresh walk above silently skips them — and that stale
// early-era rating drags the rank chart down (the spurious early "dip"). lookup.txt
// is the authoritative current-corpus manifest, so an id absent from it is gone
// for good; we strip its model fields (the entry stays for activity/time/count).
const PRUNE_SIGNATURE_KEY = 'prunedCorpusSignature';
// Fields stripped from a removed-problem entry: everything the rank chart,
// weakness model, and recall trainer key on. date/timeTaken/size/problemId/
// correct stay, so the attempt still counts toward activity/time/solved stats.
const PRUNE_STRIP_FIELDS = [
    'problemRating', 'problemRawRating', 'problemRank', 'problemSlope',
    'problemQuality', 'problemFloor', 'embedding', 'pickedEmbedding', 'pickedCellIndex',
];
// Old per-version flag keys; cleared on first run after the upgrade so they
// don't linger in users' localStorage. The new scheme stores a content-hash
// version string under VERSION_KEY instead.
const STALE_FLAG_KEYS = (size) => [
    `weaknessEmbeddingBackfill_v1_${size}`,
    `weaknessEmbeddingBackfill_v2_${size}`,
    `weaknessEmbeddingBackfill_v3_${size}`,
];

// Persist answerHistory edits WITHOUT clobbering attempts logged while the
// backfill awaited its (potentially minutes-long) shard + sidecar walk.
//
// THE BUG THIS FIXES: backfillForSize / pruneRemovedProblems capture the live
// answerHistory array up front (`const history = getAnswerHistoryForSize(size)`),
// then await a long network walk. Every solve in that window runs updateUserStats
// → updateAnswerHistory([...state.answerHistory, answer]), which REPLACES
// perSize.answerHistory[size] with a fresh array. Writing the captured snapshot
// back via setAnswerHistoryForSize(size, history) silently dropped all those
// attempts, reverting the history to its session-start state — which made
// getRecallCandidates re-serve already-solved recalls (the sub-18h repeat bug)
// and the novel filter re-serve novel problems.
//
// The backfill edits entries IN PLACE and updateUserStats shallow-spreads, so the
// live array shares our edited entry objects AND additionally holds the new
// attempts. Persisting the live array therefore keeps BOTH the backfilled fields
// and the session's attempts. The length guard falls back to the snapshot only in
// the degenerate case where the bucket somehow shrank out from under us (never on
// the append-only updateUserStats path).
function persistAnswerHistory(size, snapshot) {
    const live = getAnswerHistoryForSize(size);
    setAnswerHistoryForSize(size, live.length >= snapshot.length ? live : snapshot);
}

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

    // Strip stale fields off entries whose problem left the corpus, before the
    // per-size rating walk (which can't refresh them — the shard is gone).
    const pruned = await pruneRemovedProblems(serverVersion, ratingsVersionsBySize);

    const sizesFilled = [];
    for (const size of SUPPORTED_BOARD_SIZES) {
        // ratings_version is keyed by string in the JSON; tolerate either.
        const ratingsVersion = ratingsVersionsBySize[size]
            ?? ratingsVersionsBySize[String(size)]
            ?? null;
        const filled = await backfillForSize(size, serverVersion, ratingsVersion);
        if (filled > 0) sizesFilled.push(size);
    }
    // One shared weakness model trained on the union of both histories — retrain
    // once if any size's embeddings/ratings were refreshed OR a prune changed
    // what the model sees.
    if (sizesFilled.length > 0 || pruned) {
        setTimeout(() => trainWeakness(), 0);
    }
}

// Strip model fields from answerHistory entries whose problem is no longer in
// the corpus. Gated by the corpus signature so it scans only when the packaged
// set actually changed. The one guard that matters: never prune against a
// failed/empty lookup fetch (that would strip everything) — a real false
// positive is harmless anyway, since the rating-refresh walk refills the fields
// next time that problem's shard is touched. Returns true if it stripped fields.
async function pruneRemovedProblems(embeddingVersion, ratingsVersionsBySize) {
    const signature = JSON.stringify({ e: embeddingVersion, r: ratingsVersionsBySize });
    // Corpus unchanged since the last prune — nothing new to remove.
    if (localStorage.getItem(PRUNE_SIGNATURE_KEY) === signature) return false;

    // Authoritative current-corpus membership. Bail on any fetch failure or an
    // empty manifest so a transient blip can't strip a legitimate history.
    const lines = await fetchLookupLines();
    if (!lines) return false;
    const currentIds = new Set();
    for (const line of lines) {
        const id = line.split(/\s+/)[0];
        if (id) currentIds.add(id);
    }
    if (currentIds.size === 0) return false;

    let prunedAny = false;
    for (const size of SUPPORTED_BOARD_SIZES) {
        const history = getAnswerHistoryForSize(size);
        if (!history || history.length === 0) continue;

        // Entries whose problem the corpus no longer has and that still carry a
        // strippable field. No problemId (linked/analysis rows) ⇒ never pruned.
        const removed = history.filter(a =>
            a && typeof a.problemId === 'string'
            && !currentIds.has(a.problemId)
            && PRUNE_STRIP_FIELDS.some(k => a[k] !== undefined));
        if (removed.length === 0) continue;

        for (const a of removed) {
            for (const k of PRUNE_STRIP_FIELDS) delete a[k];
        }
        persistAnswerHistory(size, history);
        prunedAny = true;
        console.log(`prune[${size}]: stripped removed-problem fields from ${removed.length} entries`);
    }

    localStorage.setItem(PRUNE_SIGNATURE_KEY, signature);
    return prunedAny;
}

// problemId → shard resolution for one size: lookup.txt (id → zip hash) plus
// the per-size metadata.json (zip hash → zip filename). Shared by the
// version-driven walk below and the synced-entry fill. Null on any fetch
// failure so callers can bail without partial work.
async function buildShardLocator(size) {
    const lookupLines = await fetchLookupLines();
    if (!lookupLines) return null;
    const lookupMap = new Map();
    for (const line of lookupLines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 3 && parseInt(parts[2]) === size) {
            lookupMap.set(parts[0], parts[1]);
        }
    }

    const metaResp = await fetch(`tsumegos/${size}/metadata.json?v=${APP_VERSION}`);
    if (!metaResp.ok) return null;
    const zipMeta = await metaResp.json();
    const zipFilenameByHash = new Map();
    for (const z of zipMeta) {
        zipFilenameByHash.set(z.filename.split('.')[0], z.filename);
    }
    return { lookupMap, zipFilenameByHash };
}

// Shard-walk concurrency. Both walks below (the version-driven rating/
// embedding refresh and the account-sync synced-entry fill) process one shard
// fully independently of the others — each zip's writes touch only the
// history entries whose problems live in that zip — so shards can be fetched
// and processed in parallel. The bound keeps the simultaneous-request count
// predictable (per-cell sidecar Range fetches stay sequential WITHIN a shard,
// so total in-flight requests ≤ this), which is what the old fully-sequential
// loop was protecting: Cloudflare's free-plan limits are forgiving, not
// unlimited. 4 shards in flight cuts the account-restore walk to roughly a
// quarter while staying far from any limit.
const SHARD_WALK_CONCURRENCY = 4;

async function mapWithConcurrency(items, limit, fn) {
    const queue = [...items];
    const workers = Array.from({ length: Math.min(limit, queue.length) }, async () => {
        while (queue.length > 0) {
            await fn(queue.shift());
        }
    });
    await Promise.all(workers);
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
                // Wipe the single shared weakness model — it was trained on this
                // size's now-stale embeddings (jointly with the other size's).
                state.weaknessParams = null;
                state.weaknessAdam = null;
                state.weaknessSe = null;
                state.weaknessAxisCov = null;
                persistAnswerHistory(size, history);
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
                persistAnswerHistory(size, history);
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
                persistAnswerHistory(size, history);
                console.log(`embedding-backfill[${size}]: stripped embedding fields from ${strippedOld} old entries (beyond keep-last ${KEEP_EMBEDDING_LAST})`);
            }
        }

        const allNeedIds = new Set([...needIdsForEmbed, ...needIdsForRatings]);
        if (allNeedIds.size === 0) {
            persistVersions();
            return curveRankUpdated;
        }

        const locator = await buildShardLocator(size);
        if (!locator) return 0;
        const { lookupMap, zipFilenameByHash } = locator;

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
        await mapWithConcurrency([...zipToIds.keys()], SHARD_WALK_CONCURRENCY, async (zipHash) => {
            const zipFilename = zipFilenameByHash.get(zipHash);
            if (!zipFilename) return;
            try {
                const resp = await fetch(`tsumegos/${size}/${zipFilename}`);
                if (!resp.ok) return;
                const buf = await resp.arrayBuffer();
                const zip = await new JSZip().loadAsync(buf);
                const metaFile = zip.file('metadata.json');
                if (!metaFile) return;
                const items = JSON.parse(await metaFile.async('text'));
                const embeddingByProblem = new Map();
                const ratingByProblem = new Map();
                const slopeByProblem = new Map();
                const floorByProblem = new Map();
                for (const item of items) {
                    const tid = item.filename.replace(/\.sgf$/, '');
                    if (Array.isArray(item.embedding)) {
                        // Quantize to 2 decimals to keep localStorage small.
                        embeddingByProblem.set(tid, item.embedding.map(v => Math.round(v * 100) / 100));
                    }
                    if (typeof item.rating === 'number') {
                        ratingByProblem.set(tid, item.rating);
                    }
                    if (typeof item.slope === 'number') {
                        slopeByProblem.set(tid, item.slope);
                    }
                    if (typeof item.floor === 'number') {
                        floorByProblem.set(tid, item.floor);
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
                        // Raw 2PL slope a_p (the rank estimator's per-attempt
                        // success-curve slope) + its derived [0,1] quality. Fills
                        // pre-existing entries the first time `slope` ships.
                        if (slopeByProblem.has(a.problemId)) {
                            const newSlope = slopeByProblem.get(a.problemId);
                            if (a.problemSlope !== newSlope) a.problemSlope = newSlope;
                            a.problemQuality = slopeToQuality(newSlope);
                        }
                        // Per-problem 3PL guessing floor c_p (rank estimator's
                        // success-curve lower asymptote). Fills pre-existing entries
                        // the first time `floor` ships.
                        if (floorByProblem.has(a.problemId)) {
                            const newFloor = floorByProblem.get(a.problemId);
                            if (a.problemFloor !== newFloor) a.problemFloor = newFloor;
                        }
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
                            // Sequential per-entry WITHIN the shard: Range
                            // requests are cheap (~256B), and with shards
                            // running SHARD_WALK_CONCURRENCY-wide this bounds
                            // the total in-flight request count.
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
            }
        });

        if (totalFilled > 0 || totalFilledPicked > 0 || totalRatingsUpdated > 0) {
            persistAnswerHistory(size, history);
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

// --- Synced-entry field fill (account-sync) ----------------------------------
// account-sync merges /sync rows as skeleton entries: problemId, correct, date,
// timing — none of the model fields. This walk fills them from the shards,
// independent of the version bookkeeping above (the versions haven't changed;
// the ENTRIES are new):
//   - rating/rank/slope/quality/floor for every synced entry missing them
//   - correct-move `embedding` inside the keep-last window
//   - `pickedCellIndex` derived from the reported move (`syncedMove`, present
//     only on wrong NON-repeat attempts — new problems are served at
//     orientation 0 = identity, so the reported coordinate matches the shard
//     SGF; repeats are served randomly reoriented and unrecoverable). The SGF
//     root variations are the same order extraction/packaging use, so the
//     index addresses the per-cell sidecar directly.
//   - `pickedEmbedding` via the sidecar, same as the version-driven walk.
export async function fillSyncedHistoryFields(sizes) {
    let total = 0;
    for (const size of sizes) {
        if (!SUPPORTED_BOARD_SIZES.includes(size)) continue;
        try {
            total += await fillSyncedForSize(size);
        } catch (err) {
            console.error(`synced-fill[${size}] failed`, err);
            try { Sentry.captureException(err); } catch (_) {}
        }
    }
    return total;
}

async function fillSyncedForSize(size) {
    const history = getAnswerHistoryForSize(size);
    if (!history || history.length === 0) return 0;
    const startIdx = Math.max(0, history.length - KEEP_EMBEDDING_LAST);

    const needIds = new Set();
    for (let i = 0; i < history.length; i++) {
        const a = history[i];
        if (!a || !a.synced || !a.problemId) continue;
        const needsRating = typeof a.problemRating !== 'number';
        const inWindow = i >= startIdx;
        const needsEmbed = inWindow && !a.embedding;
        const needsPickedIndex = inWindow && a.correct === false &&
            typeof a.syncedMove === 'string' && typeof a.pickedCellIndex !== 'number';
        const needsPicked = inWindow && a.correct === false &&
            !a.pickedEmbedding &&
            (typeof a.pickedCellIndex === 'number' || needsPickedIndex);
        if (needsRating || needsEmbed || needsPickedIndex || needsPicked) {
            needIds.add(a.problemId);
        }
    }
    if (needIds.size === 0) return 0;

    const locator = await buildShardLocator(size);
    if (!locator) return 0;
    const { lookupMap, zipFilenameByHash } = locator;

    const zipToIds = new Map();
    for (const id of needIds) {
        const zipHash = lookupMap.get(id);
        if (!zipHash) continue;  // problem left the corpus — entry stays a skeleton
        if (!zipToIds.has(zipHash)) zipToIds.set(zipHash, []);
        zipToIds.get(zipHash).push(id);
    }

    let filled = 0;
    await mapWithConcurrency([...zipToIds.entries()], SHARD_WALK_CONCURRENCY, async ([zipHash, ids]) => {
        const zipFilename = zipFilenameByHash.get(zipHash);
        if (!zipFilename) return;
        try {
            const resp = await fetch(`tsumegos/${size}/${zipFilename}`);
            if (!resp.ok) return;
            const buf = await resp.arrayBuffer();
            const zip = await new JSZip().loadAsync(buf);
            const metaFile = zip.file('metadata.json');
            if (!metaFile) return;
            const items = JSON.parse(await metaFile.async('text'));
            const itemByProblem = new Map();
            for (const item of items) {
                itemByProblem.set(item.filename.replace(/\.sgf$/, ''), item);
            }
            let indexPayload = null;
            const idxFile = zip.file('embeddings_index.json');
            if (idxFile) {
                try { indexPayload = JSON.parse(await idxFile.async('text')); }
                catch (_) { /* shard without a usable sidecar index */ }
            }
            const idsHere = new Set(ids);

            for (let i = 0; i < history.length; i++) {
                const a = history[i];
                if (!a || !a.synced || !idsHere.has(a.problemId)) continue;
                const item = itemByProblem.get(a.problemId);
                if (!item) continue;
                const inWindow = i >= startIdx;

                if (typeof a.problemRating !== 'number' && typeof item.rating === 'number') {
                    a.problemRating = item.rating;
                    const rank = normalizedRankForSize(item.rating, size);
                    if (rank !== null) a.problemRank = rank;
                    if (typeof item.slope === 'number') {
                        a.problemSlope = item.slope;
                        a.problemQuality = slopeToQuality(item.slope);
                    }
                    if (typeof item.floor === 'number') a.problemFloor = item.floor;
                    filled++;
                }

                if (inWindow && !a.embedding && Array.isArray(item.embedding)) {
                    a.embedding = item.embedding.map(v => Math.round(v * 100) / 100);
                    filled++;
                }

                if (inWindow && a.correct === false && typeof a.syncedMove === 'string') {
                    const idx = await cellIndexForMove(zip, a.problemId, a.syncedMove, size);
                    if (typeof idx === 'number' && idx >= 0) a.pickedCellIndex = idx;
                    // One-shot either way: an unresolvable move (illegal after a
                    // corpus change, parse failure) would otherwise re-walk this
                    // shard on every sync.
                    delete a.syncedMove;
                }

                if (inWindow && a.correct === false && !a.pickedEmbedding &&
                    typeof a.pickedCellIndex === 'number' &&
                    indexPayload && indexPayload.problems) {
                    const indexEntry = indexPayload.problems[a.problemId];
                    if (indexEntry) {
                        const emb = await fetchPickedEmbeddingWithIndexEntry({
                            tsumegoId: a.problemId,
                            cellIndex: a.pickedCellIndex,
                            zipFilename,
                            boardSize: size,
                            indexEntry,
                        });
                        if (Array.isArray(emb)) {
                            a.pickedEmbedding = emb.map(v => Math.round(v * 100) / 100);
                            filled++;
                        }
                    }
                }
            }
        } catch (err) {
            console.warn(`synced-fill[${size}]: failed for ${zipFilename}:`, err);
        }
    });

    if (filled > 0) {
        persistAnswerHistory(size, history);
        console.log(`synced-fill[${size}]: filled ${filled} fields on synced entries`);
    }
    return filled;
}

// Map a reported board-notation move (orientation-0 frame) to the problem's
// SGF root-variation index — the same `children.indexOf(moveNode)` the live
// click path records, and the index the per-cell sidecar is keyed by.
async function cellIndexForMove(zip, problemId, move, size) {
    const xy = boardNotationToXy(move, size);
    if (!xy) return null;
    const sgfFile = zip.file(`${problemId}.sgf`);
    if (!sgfFile || typeof WGo === 'undefined') return null;
    try {
        const kifu = WGo.Kifu.fromSgf(await sgfFile.async('text'));
        const children = (kifu && kifu.root && kifu.root.children) || [];
        return children.findIndex(c => c.move && c.move.x === xy.x && c.move.y === xy.y);
    } catch (_) {
        return null;
    }
}
