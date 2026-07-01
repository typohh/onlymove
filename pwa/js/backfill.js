// js/backfill.js
//
// Retroactively patches per-problem fields on answerHistory entries. Two
// independent jobs, on separate data sources:
//
//   1. Embedding fill (last MODEL_INPUT_HISTORY_LENGTH entries only)
//      Triggered by `embedding_version` change in embedding_meta.json.
//      Walks the shard zips (embeddings live in metadata.json + per-cell
//      sidecars): wipes any stale embeddings, then fills `embedding` +
//      optional `pickedEmbedding` for entries inside the keep window.
//      Older entries get their embedding fields stripped to keep
//      localStorage bounded.
//
//   2. Rating refresh (all history entries, no keep-window cap)
//      Triggered by `ratings_version` change. Reads the single
//      `problem_params.bin` asset (rating/slope/floor by problem id, both
//      boards) and rewrites `answerHistory[*].problemRating` (+ derived
//      rank, slope/quality, floor) to the current value — no shard walk.
//      The chart, weakness training, and recall logic all read these
//      fields, so a stale rating shows up everywhere.
//
// Both versions are tracked separately in localStorage so a rating-only
// refresh doesn't wipe the embedding model and vice-versa. The work runs
// for every supported size on every app load; each size short-circuits
// when both versions match.

import * as Sentry from "@sentry/browser";
import {
    state,
    getAnswerHistoryForSize, setAnswerHistoryForSize,
    clearWeaknessModel,
    SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE,
} from './state.js';
import { trainWeakness } from './weakness.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { fetchPickedEmbeddingWithIndexEntry } from './picked-embedding.js';
import { fetchLookupLines } from './problem-loader.js';
import { normalizedRankForSize, ratingRankCurveVersion, slopeToQuality } from './rank-decomposition.js';
import { DATA_VERSION, EMBEDDING_META_INLINE, RATINGS_VERSION } from './data-version.js';
import { quantizeEmbedding } from './math-utils.js';
import { getTail, updateTail } from './history-store.js';
import { boardNotationToXy, tsumegoIdOf } from './sgf.js';

// Only the last N answerHistory entries need embeddings — that's all the
// trainer ever uses. Anything older eats localStorage for no benefit.
const KEEP_EMBEDDING_LAST = MODEL_INPUT_HISTORY_LENGTH;

// One stamp per axis (NOT per-size). The embedding fit, the rating fit, and the rank
// curve are each a single shared/joint artifact — a re-ship of any re-derives BOTH
// boards, so there is no per-board version. The gate + stamp live in
// backfillMissingEmbeddings (once, around the per-size work loop).
const EMBEDDING_VERSION_KEY = 'weaknessEmbeddingVersion';
const RATINGS_VERSION_KEY = 'ratingsVersion';
const RANK_CURVE_VERSION_KEY = 'rankCurveVersion';

// Removed-problem prune. Problems get dropped from the corpus on regeneration.
// Their answerHistory entries keep the rating/rank stored at attempt time — the
// shard no longer exists, so the rating-refresh walk above silently skips them —
// and that stale early-era rating drags the rank chart down (the spurious early
// "dip"). lookup.txt is the authoritative current-corpus manifest, so an id absent
// from it is gone for good; we strip its model fields (the entry stays for
// activity/time/count).
const PRUNE_SIGNATURE_KEY = 'prunedCorpusSignature';
// Fields stripped from a removed-problem entry: everything the rank chart,
// weakness model, and recall trainer key on. date/timeTaken/size/problemId/
// correct stay, so the attempt still counts toward activity/time/solved stats.
const PRUNE_STRIP_FIELDS = [
    'problemRating', 'problemRawRating', 'problemRank', 'problemSlope',
    'problemQuality', 'problemFloor', 'embedding', 'pickedEmbedding', 'pickedCellIndex',
];

// The rating fields the cold-tail re-derive writes (present problems) or strips
// (problems that left the corpus — never leave a stale rating on a removed problem;
// flattenAttempts then drops the rating-less attempt from the chart). Subset of
// PRUNE_STRIP_FIELDS without the embedding fields, which the tail never carries.
const RATING_FIELDS = [
    'problemRating', 'problemRawRating', 'problemRank',
    'problemSlope', 'problemQuality', 'problemFloor',
];
// Legacy per-version flag keys, cleared on each embedding refit so they don't
// linger in users' localStorage.
const STALE_FLAG_KEYS = (size) => [
    `weaknessEmbeddingBackfill_v1_${size}`,
    `weaknessEmbeddingBackfill_v2_${size}`,
    `weaknessEmbeddingBackfill_v3_${size}`,
];

// Persist answerHistory edits WITHOUT clobbering attempts logged while the
// backfill awaited its (potentially minutes-long) shard + sidecar walk.
//
// backfillForSize / pruneRemovedProblems capture the live answerHistory array up
// front (`const history = getAnswerHistoryForSize(size)`), then await a long
// network walk. Every solve in that window runs updateUserStats →
// updateAnswerHistory([...state.answerHistory, answer]), which REPLACES
// perSize.answerHistory[size] with a fresh array. Writing the captured snapshot
// back via setAnswerHistoryForSize(size, history) would silently drop all those
// attempts, reverting the history to its session-start state — which would make
// getRecallCandidates re-serve already-solved recalls and the novel filter
// re-serve novel problems.
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
    // Joint metadata (embedding version + labels + sidecar layout + ratings_version)
    // is shared across sizes and INLINED into the bundle (data-version.js), so it can
    // never be served stale by a cache — the read is synchronous and always matches the
    // deployed code. Populates state.embeddingMeta with the K/dtype/bytes_per_cell tuple
    // used by picked-embedding.js to decode per-cell .embeddings.bin sidecars.
    const meta = EMBEDDING_META_INLINE;
    const serverEmbeddingVersion = (typeof meta.embedding_version === 'string')
        ? meta.embedding_version : 'unknown';
    if (meta.labels && meta.labels.components) {
        state.embeddingLabels = meta.labels.components;
    }
    if (typeof meta.K === 'number') {
        state.embeddingMeta = {
            K: meta.K,
            dtype: typeof meta.dtype === 'string' ? meta.dtype : 'float16',
            bytes_per_cell: typeof meta.bytes_per_cell === 'number'
                ? meta.bytes_per_cell
                : meta.K * 2,  // sensible fp16 default
            embedding_version: serverEmbeddingVersion,
        };
    }

    // SINGLE version gates. Each axis is one shared/joint artifact (the embedding fit,
    // RATINGS_VERSION = the joint cross-board rating fit, and the shared rank curve), so
    // a change re-derives BOTH boards — no per-board version. Decide once here; do the
    // per-size WORK in backfillForSize (problems live in per-size shards); then stamp
    // the single keys ONCE, after every size's walk has persisted. The stamp is the only
    // "done" marker, so a shutdown mid-loop leaves nothing stamped and the whole backfill
    // re-runs idempotently. (problemRank is derived from the raw rating through M, so a
    // curve re-fit with untouched ratings still shifts every stored rank — hence its own
    // gate.) The curve is the same for both sizes; query it with the default board.
    const curveVersion = ratingRankCurveVersion(DEFAULT_BOARD_SIZE);
    const embeddingsChanged = localStorage.getItem(EMBEDDING_VERSION_KEY) !== serverEmbeddingVersion;
    const ratingsChanged = RATINGS_VERSION !== null
        && localStorage.getItem(RATINGS_VERSION_KEY) !== RATINGS_VERSION;
    const curveChanged = curveVersion !== null
        && localStorage.getItem(RANK_CURVE_VERSION_KEY) !== curveVersion;

    // Strip stale fields off entries whose problem left the corpus, before the rating
    // walk (which can't refresh them — the shard is gone).
    const pruned = await pruneRemovedProblems(serverEmbeddingVersion, RATINGS_VERSION);

    const sizesFilled = [];
    if (embeddingsChanged || ratingsChanged || curveChanged) {
        // Bracket the whole run: a "triggered" line (which axes moved + the new
        // ratings version) and a "complete" line (elapsed + which sizes were rewritten).
        // These straddle the async shard walk, so the console shows when a re-derive
        // started and that it finished — the front-page rank can shift between them.
        const changed = [
            embeddingsChanged && 'embeddings',
            ratingsChanged && 'ratings',
            curveChanged && 'curve',
        ].filter(Boolean).join('+');
        const t0 = performance.now();
        console.log(`backfill: triggered (${changed} changed; ratings_version=${RATINGS_VERSION}) — re-deriving stored history`);
        for (const size of SUPPORTED_BOARD_SIZES) {
            const filled = await backfillForSize(size, {
                embeddingsChanged, ratingsChanged, curveChanged, serverEmbeddingVersion,
            });
            if (filled > 0) sizesFilled.push(size);
        }
        // Stamp the single versions ONCE, only now that every size has persisted.
        if (embeddingsChanged) localStorage.setItem(EMBEDDING_VERSION_KEY, serverEmbeddingVersion);
        if (ratingsChanged) localStorage.setItem(RATINGS_VERSION_KEY, RATINGS_VERSION);
        if (curveChanged) localStorage.setItem(RANK_CURVE_VERSION_KEY, curveVersion);
        console.log(`backfill: complete in ${Math.round(performance.now() - t0)}ms (sizes rewritten: ${sizesFilled.join(',') || 'none'})`);
    }

    // Retrain the single shared weakness model if any size refreshed, a prune changed
    // its training set, OR the model is missing — the last heals a device whose model
    // was cleared by an interrupted embedding refit and then re-loaded after the stamp
    // (so no refit re-fires). trainWeakness no-ops below its min training size, so a
    // cold-start user just gets a cheap early return.
    const weaknessUntrained = !state.weaknessParams || state.weaknessParams.length < 4;
    if (sizesFilled.length > 0 || pruned || weaknessUntrained) {
        setTimeout(() => trainWeakness(), 0);
    }
}

// Strip model fields from answerHistory entries whose problem is no longer in
// the corpus. Gated by the corpus signature so it scans only when the packaged
// set actually changed. The one guard that matters: never prune against a
// failed/empty lookup fetch (that would strip everything) — a real false
// positive is harmless anyway, since the rating-refresh walk refills the fields
// next time that problem's shard is touched. Returns true if it stripped fields.
async function pruneRemovedProblems(embeddingVersion, ratingsVersion) {
    const signature = JSON.stringify({ e: embeddingVersion, r: ratingsVersion });
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

    const metaResp = await fetch(`tsumegos/${size}/metadata.json?v=${DATA_VERSION}`);
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
// so total in-flight requests ≤ this): Cloudflare's free-plan limits are
// forgiving, not unlimited. 4 shards in flight keeps the account-restore walk
// fast while staying far from any limit.
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

// Fetch + open one shard zip and parse its metadata.json. Returns { zip, items }
// or null on any fetch/parse miss, so a caller can skip the shard. Shared by all
// three shard walks below (version-driven refresh, synced-entry fill, cold-tail
// fill) — the per-shard processing differs, but the fetch+open+parse prologue is
// identical.
async function loadShard(size, zipFilename) {
    const resp = await fetch(`tsumegos/${size}/${zipFilename}`);
    if (!resp.ok) return null;
    const zip = await new JSZip().loadAsync(await resp.arrayBuffer());
    const metaFile = zip.file('metadata.json');
    if (!metaFile) return null;
    const items = JSON.parse(await metaFile.async('text'));
    return { zip, items };
}

// Group problem ids by the shard zip that holds them, via the locator's
// id → zip-hash map. Ids absent from the corpus are dropped (the problem left
// the set — the entry stays a skeleton). Returns Map(zipHash → id[]).
function groupIdsByZip(lookupMap, ids) {
    const zipToIds = new Map();
    for (const id of ids) {
        const zipHash = lookupMap.get(id);
        if (!zipHash) continue;
        if (!zipToIds.has(zipHash)) zipToIds.set(zipHash, []);
        zipToIds.get(zipHash).push(id);
    }
    return zipToIds;
}

// Load + decode the single problem_params.bin asset into
// Map<hexId, {rating, slope, floor}>. This is the rating refresh's source (rating/
// slope/floor for EVERY problem, both boards, no embeddings), replacing the shard
// walk — cache-busted by DATA_VERSION like metadata.json / lookup.txt. Layout: a
// flat sequence of 16-byte records, each a 4-byte big-endian problem id (byte k ->
// two hex chars) followed by 3 little-endian float32 rating/slope/floor. Matches the
// Worker decoder in src/leaderboard.js loadProblemParams. Null on any fetch failure
// so a caller bails (leaving the version un-stamped to retry) rather than doing
// partial work.
async function loadProblemParams() {
    try {
        const resp = await fetch(`tsumegos/problem_params.bin?v=${DATA_VERSION}`);
        if (!resp.ok) return null;
        const bytes = new Uint8Array(await resp.arrayBuffer());
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const REC = 16;   // 4-byte id + 3 little-endian float32 (rating, slope, floor)
        const map = new Map();
        for (let off = 0; off + REC <= bytes.length; off += REC) {
            let id = '';
            for (let k = 0; k < 4; k++) id += bytes[off + k].toString(16).padStart(2, '0');
            map.set(id, {
                rating: dv.getFloat32(off + 4, true),
                slope: dv.getFloat32(off + 8, true),
                floor: dv.getFloat32(off + 12, true),
            });
        }
        return map;
    } catch (err) {
        console.warn('backfill: problem_params.bin load failed:', err);
        return null;
    }
}

// Per-size WORK for one board, driven by the GLOBAL change flags the caller already
// decided (the versions are single/shared — see backfillMissingEmbeddings). Problems
// live in per-size shards so the walk is necessarily per-size, but this function NEVER
// touches the version stamps — the caller stamps the single versions once, after every
// size has returned and persisted, so a shutdown mid-loop leaves nothing stamped and
// the whole backfill re-runs idempotently.
async function backfillForSize(size, { embeddingsChanged, ratingsChanged, curveChanged, serverEmbeddingVersion }) {
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
                clearWeaknessModel();
                persistAnswerHistory(size, history);
                console.log(`backfill[${size}]: cleared ${cleared} stale embeddings (-> ${serverEmbeddingVersion})`);
            }
            for (const k of STALE_FLAG_KEYS(size)) localStorage.removeItem(k);
        }

        if (!history || history.length === 0) {
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
                console.log(`backfill[${size}]: recomputed problemRank for ${curveRankUpdated} entries (curve change)`);
            }
        }

        // Curve-only tail remap: when ONLY the curve changed (ratings didn't),
        // recompute the cold tail's problemRank straight from its stored raw rating —
        // no shard fetch needed. When ratings ALSO changed, the shard-backed tail
        // re-derive below rewrites rank from the fresh rating, so skip this.
        if (curveChanged && !ratingsChanged) {
            let tailUpdated = 0;
            await updateTail((entry) => {
                if (!entry || entry.size !== size) return null;
                if (typeof entry.problemRating !== 'number') return null;
                const newRank = normalizedRankForSize(entry.problemRating, size);
                if (newRank !== null && entry.problemRank !== newRank) {
                    entry.problemRank = newRank;
                    tailUpdated++;
                    return entry;
                }
                return null;
            });
            if (tailUpdated > 0) {
                console.log(`backfill[${size}]: recomputed problemRank for ${tailUpdated} cold-tail entries (curve change)`);
            }
        }

        // Ratings refresh reads the single problem_params.bin asset (rating/slope/
        // floor by id, both boards) — built ONCE per size, replacing the shard walk
        // for the rating fields. Embeddings still walk the shards below. Null on a
        // transient fetch failure -> bail so the version stays un-stamped and retries
        // on the next load.
        let problemParams = null;
        if (ratingsChanged) {
            problemParams = await loadProblemParams();
            if (!problemParams) return curveRankUpdated;
        }

        // Working-set ratings rewrite: one pass over the FULL history (no keep-window
        // cap — the chart needs old entries updated), reading the current value from
        // problem_params. Present problems get the CURRENT rating (+ derived rank,
        // slope/quality, floor via the shared applyRatingFields); a problem that left
        // the corpus keeps its stale value for pruneRemovedProblems to strip. Edited
        // in place; persisted below alongside any embedding fills.
        let totalRatingsUpdated = 0;
        if (ratingsChanged) {
            for (let i = 0; i < history.length; i++) {
                const a = history[i];
                if (!a.problemId) continue;
                const item = problemParams.get(a.problemId);
                if (!item || typeof item.rating !== 'number') continue;
                if (a.problemRating !== item.rating) totalRatingsUpdated++;
                applyRatingFields(a, item, size);
                // problemRating holds the raw value, so the separate
                // problemRawRating field is redundant.
                if (a.problemRawRating !== undefined) delete a.problemRawRating;
            }
        }

        // Does the cold tail carry this size's rows to re-derive? (ratings change
        // only). Its ratings are frozen at eviction time, so after a re-rate the old
        // attempts carry stale (and device-dependent — each device evicted at a
        // different rating era) ratings, and the Rank-tab chart plots them. The tail
        // rows are rewritten from problem_params after the embedding walk below.
        let tailNeedsRederive = false;
        if (ratingsChanged) {
            const tail = await getTail();
            for (const e of tail) {
                if (e && e.size === size && typeof e.problemId === 'string') {
                    tailNeedsRederive = true;
                    break;
                }
            }
        }

        // Embedding needs: last KEEP_EMBEDDING_LAST entries with missing fields. These
        // drive the shard walk (embeddings live in metadata.json + the per-cell
        // sidecars; problem_params carries none).
        const startIdx = Math.max(0, history.length - KEEP_EMBEDDING_LAST);
        const needIdsForEmbed = new Set();
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
                console.log(`backfill[${size}]: stripped embedding fields from ${strippedOld} old entries (beyond keep-last ${KEEP_EMBEDDING_LAST})`);
            }
        }

        // Embedding shard walk (problem_params has no embeddings, so this stays on the
        // zips). Skipped entirely when nothing in the keep window needs a fill — so a
        // ratings-only refresh never fetches a shard.
        let totalFilled = 0;
        let totalFilledPicked = 0;
        if (needIdsForEmbed.size > 0) {
            const locator = await buildShardLocator(size);
            if (locator) {
                const { lookupMap, zipFilenameByHash } = locator;
                const zipToIds = groupIdsByZip(lookupMap, needIdsForEmbed);
                await mapWithConcurrency([...zipToIds.keys()], SHARD_WALK_CONCURRENCY, async (zipHash) => {
                    const zipFilename = zipFilenameByHash.get(zipHash);
                    if (!zipFilename) return;
                    try {
                        const shard = await loadShard(size, zipFilename);
                        if (!shard) return;
                        const { zip, items } = shard;
                        const embeddingByProblem = new Map();
                        for (const item of items) {
                            if (Array.isArray(item.embedding)) {
                                // Quantize to 2 decimals to keep localStorage small.
                                embeddingByProblem.set(tsumegoIdOf(item.filename), quantizeEmbedding(item.embedding));
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
                                console.warn(`backfill[${size}]: parse embeddings_index.json failed for ${zipFilename}:`, e);
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
                                        a.pickedEmbedding = quantizeEmbedding(emb);
                                        totalFilledPicked++;
                                    }
                                }
                            }
                        }
                    } catch (err) {
                        console.warn(`backfill[${size}]: failed for ${zipFilename}:`, err);
                    }
                });
            }
        }

        if (totalFilled > 0 || totalFilledPicked > 0 || totalRatingsUpdated > 0) {
            persistAnswerHistory(size, history);
            console.log(`backfill[${size}]: filled correct=${totalFilled} picked=${totalFilledPicked} ratings=${totalRatingsUpdated}`);
        }

        // Re-derive the cold tail from problem_params (ratings change only). Present
        // problems get the CURRENT rating; a problem that left the corpus has its rating
        // fields STRIPPED — never leave a stale rating on a removed problem — so
        // flattenAttempts drops the now-rating-less attempt from the chart. One IDB
        // cursor pass. (The working-set copy wins in getMergedFullAnswerHistory, so an
        // attempt present in both is unaffected here; this pass handles the tail-ONLY
        // attempts.)
        if (tailNeedsRederive && problemParams) {
            let tailRederived = 0, tailStripped = 0;
            await updateTail((entry) => {
                if (!entry || entry.size !== size || typeof entry.problemId !== 'string') return null;
                const item = problemParams.get(entry.problemId);
                if (item) {
                    if (entry.problemRating === item.rating) return null;  // already current
                    applyRatingFields(entry, item, size);
                    if (entry.problemRawRating !== undefined) delete entry.problemRawRating;
                    tailRederived++;
                    return entry;
                }
                // Absent from problem_params ⇒ the problem left the corpus: strip its
                // rating fields (flattenAttempts then drops the attempt from the chart).
                if (!problemParams.has(entry.problemId)) {
                    let touched = false;
                    for (const k of RATING_FIELDS) {
                        if (entry[k] !== undefined) { delete entry[k]; touched = true; }
                    }
                    if (touched) { tailStripped++; return entry; }
                }
                return null;
            });
            if (tailRederived || tailStripped) {
                console.log(`backfill[${size}]: cold-tail re-derived ${tailRederived}, stripped ${tailStripped} removed-problem ratings`);
            }
        }
        return totalFilled + totalFilledPicked + totalRatingsUpdated + curveRankUpdated;
    } catch (err) {
        console.error(`backfill[${size}] failed`, err);
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
//
// `includeTail` (set on a full pull) also fills synced skeletons that landed in
// the IndexedDB cold tail — a full pull evicts its old rows there mid-pull, out
// of reach of the working-set walk. Incremental pulls only merge recent rows
// (working set), so they leave it false and skip the tail scan.
export async function fillSyncedHistoryFields(sizes, { includeTail = false } = {}) {
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
    if (includeTail) total += (await fillSyncedTailFields(sizes)).filled;
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

    const zipToIds = groupIdsByZip(lookupMap, needIds);

    let filled = 0;
    await mapWithConcurrency([...zipToIds.entries()], SHARD_WALK_CONCURRENCY, async ([zipHash, ids]) => {
        const zipFilename = zipFilenameByHash.get(zipHash);
        if (!zipFilename) return;
        try {
            const shard = await loadShard(size, zipFilename);
            if (!shard) return;
            const { zip, items } = shard;
            const itemByProblem = new Map();
            for (const item of items) {
                itemByProblem.set(tsumegoIdOf(item.filename), item);
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
                    applyRatingFields(a, item, size);
                    filled++;
                }

                if (inWindow && !a.embedding && Array.isArray(item.embedding)) {
                    a.embedding = quantizeEmbedding(item.embedding);
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
                            a.pickedEmbedding = quantizeEmbedding(emb);
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

// --- Cold-tail synced fill (steady state) ---
// History spans the localStorage working set AND the IndexedDB cold tail. A
// from-the-beginning pull (adoption / full refresh) evicts its old rows to the
// tail mid-pull, so the working-set fill (fillSyncedForSize) can't reach them.
// This completes the sync there: synced skeletons in the tail get their
// problem-level rating fields filled from the shards (one fetch per problem
// fills every attempt of it). Incremental pulls never reach the tail, so the
// caller opts in (includeTail) only on a full pull.

function entrySize(e) {
    return SUPPORTED_BOARD_SIZES.includes(e.size) ? e.size : DEFAULT_BOARD_SIZE;
}

// Apply a shard metadata.json item's problem-level fields to a history entry.
function applyRatingFields(e, item, size) {
    e.problemRating = item.rating;
    const rank = normalizedRankForSize(item.rating, size);
    if (rank !== null && rank !== undefined) e.problemRank = rank;
    if (typeof item.slope === 'number') {
        e.problemSlope = item.slope;
        e.problemQuality = slopeToQuality(item.slope);
    }
    if (typeof item.floor === 'number') e.problemFloor = item.floor;
}

// Walk `size`'s shards for the given problem ids → Map(id → metadata item). Ids
// absent from the corpus are omitted. `ok` is false only when the shard locator
// couldn't be built (offline), so a caller can retry rather than treat the fill
// as complete.
async function fetchShardItems(size, ids) {
    const out = new Map();
    if (!ids || ids.size === 0) return { items: out, ok: true };
    const locator = await buildShardLocator(size);
    if (!locator) return { items: out, ok: false };
    const { lookupMap, zipFilenameByHash } = locator;
    const zipToIds = groupIdsByZip(lookupMap, ids);
    await mapWithConcurrency([...zipToIds.entries()], SHARD_WALK_CONCURRENCY, async ([zipHash, idList]) => {
        const zipFilename = zipFilenameByHash.get(zipHash);
        if (!zipFilename) return;
        try {
            const shard = await loadShard(size, zipFilename);
            if (!shard) return;
            const want = new Set(idList);
            for (const item of shard.items) {
                const id = tsumegoIdOf(item.filename);
                if (want.has(id) && typeof item.rating === 'number') out.set(id, item);
            }
        } catch (err) {
            console.warn(`synced-tail-fill[${size}]: shard ${zipFilename} failed:`, err);
        }
    });
    return { items: out, ok: true };
}

// Fill one size's cold-tail synced skeletons. Returns { filled, ok }.
async function fillSyncedTailForSize(size) {
    const tail = await getTail();
    if (!tail || tail.length === 0) return { filled: 0, ok: true };
    const need = new Set();
    for (const e of tail) {
        if (!e || !e.synced || !e.problemId) continue;
        if (entrySize(e) !== size) continue;
        if (typeof e.problemRating !== 'number') need.add(e.problemId);
    }
    if (need.size === 0) return { filled: 0, ok: true };

    const { items, ok } = await fetchShardItems(size, need);
    if (items.size === 0) return { filled: 0, ok };

    let filled = 0;
    await updateTail((entry) => {
        if (!entry || typeof entry.problemRating === 'number') return null;
        if (entrySize(entry) !== size) return null;
        const item = items.get(entry.problemId);
        if (!item) return null;
        applyRatingFields(entry, item, size);
        filled++;
        return entry;
    });
    if (filled > 0) console.log(`synced-tail-fill[${size}]: filled ${filled} cold-tail entries`);
    return { filled, ok };
}

// Fill the cold tail's synced skeletons across sizes. Returns { filled, ok }
// (ok=false if any needed size's shards couldn't be fetched). Exported so the
// one-time tail migration can reuse it on already-damaged tails.
export async function fillSyncedTailFields(sizes) {
    let filled = 0;
    let ok = true;
    for (const size of sizes) {
        if (!SUPPORTED_BOARD_SIZES.includes(size)) continue;
        try {
            const r = await fillSyncedTailForSize(size);
            filled += r.filled;
            ok = ok && r.ok;
        } catch (err) {
            console.warn(`synced-tail-fill[${size}] failed:`, err);
            try { Sentry.captureException(err); } catch (_) {}
            ok = false;
        }
    }
    return { filled, ok };
}
