// js/backfill.js
//
// Retroactively patches per-problem fields on answerHistory entries. Two
// independent jobs, on separate data sources:
//
//   1. Embedding fill (last MODEL_INPUT_HISTORY_LENGTH entries only)
//      Triggered by `embedding_version` change in embedding_meta.json.
//      Wipes any stale embeddings, then fills `embedding` from the
//      lookup-aligned correct-move sidecar (tsumegos/lookup.embeddings.bin,
//      one ~130-byte Range per problem — no shard zips) and optional
//      `pickedEmbedding` via each shard's per-cell bin (byte offsets live in
//      the zip's embeddings_index.json, so only shards carrying a picked
//      need are downloaded). Older entries get their embedding fields
//      stripped to keep localStorage bounded.
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
    getMergedAnswerHistory,
    SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE,
} from './state.js';
import { MODEL_INPUT_HISTORY_LENGTH } from './constants.js';
import { fetchPickedEmbeddingWithIndexEntry, fetchCorrectEmbedding } from './picked-embedding.js';
import { fetchLookupLines } from './problem-loader.js';
import { rankForSize, ratingRankCurveVersion, problemInfoWeight } from './rank-decomposition.js';
import { DATA_VERSION, EMBEDDING_META_INLINE, RATINGS_VERSION } from './data-version.js';
import { detectRatingsCorrection } from './corrections-notice.js';
import { quantizeEmbedding } from './math-utils.js';
import { getTail, updateTail } from './history-store.js';
import { boardNotationToXy } from './sgf.js';

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

// The per-attempt fields that are DERIVED from a pipeline artifact, declared once
// here and cleared only through these lists. Three passes below remove them for
// three different reasons (a refit invalidates them, they age out of the keep
// window, or their problem left the corpus), and each used to spell its own list
// out by hand — which is how the refit wipe came to clear `embedding` and leave
// `pickedEmbedding` behind. A survivor is unreachable after that: every fill is
// fill-if-MISSING, so a stale vector is never refetched, and it is a well-formed
// K-array that nothing downstream can tell from a current one. Naming the fields
// in one place is what stops the next derived field being wired into the fills
// and missed by one of the clears.

// The rating fields the cold-tail re-derive writes (present problems) or strips
// (problems that left the corpus — never leave a stale rating on a removed problem;
// flattenAttempts then drops the rating-less attempt from the chart). The tail
// never carries embeddings, which is why this list stands alone.
const RATING_FIELDS = [
    'problemRating', 'problemRawRating', 'problemRank',
    'problemSlope', 'problemQuality', 'problemFloor',
];

// The FIT-DERIVED vectors: an embedding refit invalidates exactly these, so this
// is the list the refit wipe clears.
const EMBEDDING_VECTOR_FIELDS = ['embedding', 'pickedEmbedding'];

// The vectors plus `pickedCellIndex`. The index is NOT fit-derived — it addresses
// the SGF's root variations, so a refit cannot move it, and it is the key the
// picked refill is looked up by. So the refit wipe keeps it (dropping it would
// make the wiped vector unrecoverable) while the two passes that are NOT going to
// refill — the keep-window strip and the removed-problem prune — drop it too.
const EMBEDDING_ALL_FIELDS = [...EMBEDDING_VECTOR_FIELDS, 'pickedCellIndex'];

// Fields stripped from a removed-problem entry: everything the rank chart,
// weakness model, and recall trainer key on. date/timeTaken/size/problemId/
// correct stay, so the attempt still counts toward activity/time/solved stats.
const PRUNE_STRIP_FIELDS = [...RATING_FIELDS, ...EMBEDDING_ALL_FIELDS];

// Delete `fields` from one entry; true iff any of them were present. The single
// removal primitive the three passes share, so "which fields" is always a list
// argument and never something a caller spells out.
function deleteFields(a, fields) {
    let touched = false;
    for (const f of fields) {
        if (a[f] !== undefined) {
            delete a[f];
            touched = true;
        }
    }
    return touched;
}
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
// and the session's attempts. The live array is ALSO authoritative for
// membership: the working set legitimately SHRINKS during our awaits — the
// eviction moves rows to the IndexedDB tail mid-pull, and the mirror deletion
// pass removes rows the server no longer has. (An earlier length guard here
// restored the pre-await snapshot on any shrink, which re-added just-evicted
// rows to the working set while their copies sat in the tail — the next evict
// then re-appended them as tail duplicates, a permanent digest surplus. Found
// by the sum-first digest during the epoch rehearsal.)
function persistAnswerHistory(size, _snapshot) {
    setAnswerHistoryForSize(size, getAnswerHistoryForSize(size));
}

// Stamp the three data-version gates at their CURRENT values when the device
// holds no history yet. An empty history is trivially consistent with the
// current corpus, and whatever gets written next (the account adoption's
// synced fill) is derived from current-version assets. Without this an adopted
// device boots stamp-less: backfillMissingEmbeddings right after adoption
// treats the freshly filled history as stale, wipes its embeddings, and
// re-walks the same shards a second time — doubling a walk that is already the
// heaviest thing a new device does. No-op (returns false) if any history
// exists; then the normal version gate decides.
export function stampVersionsForFreshHistory() {
    if (getMergedAnswerHistory().length > 0) return false;
    const meta = EMBEDDING_META_INLINE;
    const embeddingVersion = (typeof meta.embedding_version === 'string')
        ? meta.embedding_version : 'unknown';
    localStorage.setItem(EMBEDDING_VERSION_KEY, embeddingVersion);
    if (RATINGS_VERSION !== null) localStorage.setItem(RATINGS_VERSION_KEY, RATINGS_VERSION);
    const curveVersion = ratingRankCurveVersion(DEFAULT_BOARD_SIZE);
    if (curveVersion !== null) localStorage.setItem(RANK_CURVE_VERSION_KEY, curveVersion);
    return true;
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

    // Arm the one-time "ratings/ranks were corrected" notice. The version-CHANGED
    // flags are the whole signal — deliberately NOT also gated on a non-null prior
    // stamp. A freshly-adopted device is already excluded because
    // maybeFinishAccountSwitch stamps the CURRENT versions before pulling history
    // (account-sync.js), so its ratings/curve are unchanged (flags false); a
    // brand-new device is excluded by the < MIN_ATTEMPTS gate inside detect (and
    // gets stamped on this same first load). That leaves null-stamp-with-history =
    // a returning user who predates the stamp — who SHOULD be told their ranks
    // moved, and a non-null guard would wrongly silence. Runs BEFORE the stamps
    // are advanced below; the overlay fires from main.js at the end of this chain.
    detectRatingsCorrection({
        ratingsCorrected: ratingsChanged,
        curveCorrected: curveChanged,
    });

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
        let allOk = true;
        for (const size of SUPPORTED_BOARD_SIZES) {
            const { count, ok } = await backfillForSize(size, {
                embeddingsChanged, ratingsChanged, curveChanged, serverEmbeddingVersion,
            });
            if (count > 0) sizesFilled.push(size);
            if (!ok) allOk = false;
        }
        // Stamp the single versions ONCE, only now that every size has persisted —
        // and only when every size actually completed. A failed size (offline
        // problem_params fetch, shard-walk exception) leaves the stamps untouched so
        // the whole backfill re-runs idempotently on the next load; stamping over a
        // failure would freeze stale per-attempt rating/slope/floor under a matching
        // version (and, on an embedding refit, lock in wiped-but-never-refilled
        // embeddings, since the wipe runs before the fills can fail).
        if (allOk) {
            if (embeddingsChanged) localStorage.setItem(EMBEDDING_VERSION_KEY, serverEmbeddingVersion);
            if (ratingsChanged) localStorage.setItem(RATINGS_VERSION_KEY, RATINGS_VERSION);
            if (curveChanged) localStorage.setItem(RANK_CURVE_VERSION_KEY, curveVersion);
        }
        console.log(`backfill: complete in ${Math.round(performance.now() - t0)}ms (sizes rewritten: ${sizesFilled.join(',') || 'none'}${allOk ? '' : '; INCOMPLETE — versions left un-stamped, retrying next load'})`);
    }

    // No model to retrain: the motif readings (motif-stats.js) are a pure
    // function of the answer history, recomputed by their readers, so a refreshed
    // or pruned history is picked up on the next read with nothing to keep in
    // sync and nothing to heal after an interrupted refit.
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

        for (const a of removed) deleteFields(a, PRUNE_STRIP_FIELDS);
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

// Shard-walk concurrency. All the walks below (the version-driven rating/
// embedding refresh and the account-sync synced-entry fills) process one shard
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
// or null on any fetch/parse miss, so the walk can skip the shard. Every shard
// walk (version-driven refresh, synced-entry fill, cold-tail fill) goes through
// walkShards below, which owns this fetch+open+parse prologue.
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

// Shared shard-walk prologue for the three walks in this file (version-driven
// embedding refresh, synced-entry fill, cold-tail fill): build the locator,
// group the wanted problem ids by shard zip, then fetch + open each shard
// (SHARD_WALK_CONCURRENCY-wide) and hand `{ zip, items, zipFilename, shardIds }`
// to `perShard`. A per-shard failure goes to `onShardError` (each walk logs
// under its own tag) and never aborts the walk. Returns false ONLY when the
// shard locator couldn't be built (offline / fetch failure) — no shard was
// touched — so a caller can distinguish "walked" from "retry later".
async function walkShards(size, ids, perShard, onShardError) {
    const locator = await buildShardLocator(size);
    if (!locator) return false;
    const { lookupMap, zipFilenameByHash } = locator;
    const zipToIds = groupIdsByZip(lookupMap, ids);
    await mapWithConcurrency([...zipToIds.entries()], SHARD_WALK_CONCURRENCY, async ([zipHash, shardIds]) => {
        const zipFilename = zipFilenameByHash.get(zipHash);
        if (!zipFilename) return;
        try {
            const shard = await loadShard(size, zipFilename);
            if (!shard) return;
            await perShard({ zip: shard.zip, items: shard.items, zipFilename, shardIds });
        } catch (err) {
            onShardError(zipFilename, err);
        }
    });
    return true;
}

// Parse a shard's embeddings_index.json (the per-cell sidecar index), if the
// shard ships one — older builds don't. Returns the payload or null (missing
// file / parse failure); `onParseError` lets each walk decide whether a parse
// failure warrants a console line.
async function parseEmbeddingsIndex(zip, onParseError) {
    const idxFile = zip.file('embeddings_index.json');
    if (!idxFile) return null;
    try {
        return JSON.parse(await idxFile.async('text'));
    } catch (e) {
        if (onParseError) onParseError(e);
        return null;
    }
}

// Fill one entry's `pickedEmbedding` from the shard's per-cell sidecar, when
// it's an eligible wrong attempt (known pickedCellIndex, not yet filled, and
// the shard's index knows the problem). Range-fetches one cell. Returns true
// iff it filled — shared by the version-driven walk and the synced-entry fill,
// which count fills differently but fill identically.
async function fillPickedEmbeddingFromSidecar(a, { indexPayload, zipFilename, size }) {
    if (a.correct !== false) return false;
    if (typeof a.pickedCellIndex !== 'number') return false;
    if (a.pickedEmbedding) return false;
    if (!indexPayload || !indexPayload.problems) return false;
    const indexEntry = indexPayload.problems[a.problemId];
    if (!indexEntry) return false;
    const emb = await fetchPickedEmbeddingWithIndexEntry({
        tsumegoId: a.problemId,
        cellIndex: a.pickedCellIndex,
        zipFilename,
        boardSize: size,
        indexEntry,
    });
    if (Array.isArray(emb)) {
        a.pickedEmbedding = quantizeEmbedding(emb);
        return true;
    }
    return false;
}

// Binary-search the id-sorted lookup.txt lines ("id zipHash size") for a
// problem id; returns its LINE INDEX — which is also the problem's row address
// in the lookup-aligned correct-move sidecar — or -1 when unknown.
function lookupLineIndex(lines, tsumegoId) {
    let lo = 0;
    let hi = lines.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const id = lines[mid].slice(0, 8);
        if (id === tsumegoId) return mid;
        if (id < tsumegoId) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

// Correct-move `embedding` fills via the lookup-aligned sidecar
// (tsumegos/lookup.embeddings.bin): one ~130-byte Range fetch per distinct
// problem — repeats share a memoized fetch — and NO shard zips (the same
// vector used to be read from each shard's metadata.json, which meant
// downloading the whole ~2MB zip per touched shard). Entries are mutated in
// place; returns the fill count. Fills that can't resolve (lookup/sidecar
// unavailable, id gone from the corpus) are skipped; a later fill run retries.
const CORRECT_FILL_CONCURRENCY = 8;
async function fillCorrectEmbeddingsViaSidecar(entries) {
    const needy = entries.filter((a) => a && a.problemId && !a.embedding);
    if (needy.length === 0) return 0;
    const lines = await fetchLookupLines();
    if (!lines) return 0;
    const vecByProblem = new Map();
    const vecFor = (pid) => {
        if (!vecByProblem.has(pid)) {
            vecByProblem.set(pid, (async () => {
                const row = lookupLineIndex(lines, pid);
                if (row < 0) return null;
                const vec = await fetchCorrectEmbedding(row, lines.length);
                return Array.isArray(vec) ? quantizeEmbedding(vec) : null;
            })());
        }
        return vecByProblem.get(pid);
    };
    let filled = 0;
    await mapWithConcurrency(needy, CORRECT_FILL_CONCURRENCY, async (a) => {
        const vec = await vecFor(a.problemId);
        if (vec && !a.embedding) {
            a.embedding = vec.slice();
            filled++;
        }
    });
    return filled;
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
//
// Returns { count, ok }: `count` is the number of entries rewritten (drives the
// weakness retrain); `ok: false` means this size's re-derive did NOT complete
// (params fetch failed, or an exception mid-walk) — the caller must then leave
// the version stamps untouched so the run retries on the next load.
//
// Orchestrates the per-size steps in a fixed order; each step lives in its own
// helper directly below, in execution order.
async function backfillForSize(size, { embeddingsChanged, ratingsChanged, curveChanged, serverEmbeddingVersion }) {
    try {
        const history = getAnswerHistoryForSize(size);
        // The working set can hold NONE of this size's rows (a lapsed size whose
        // whole career sits in the IndexedDB cold tail — the unified working set
        // keeps only the most-recent attempts across BOTH sizes), but the
        // full-career chart still reads the tail, so the tail remap/re-derive
        // below run even when this size's working-set view is empty. Only the
        // working-set passes and the embedding fills (model inputs are
        // working-set-only by design) are gated on it.
        const hasWorkingSet = !!(history && history.length > 0);

        if (embeddingsChanged) {
            wipeStaleEmbeddings(size, history, serverEmbeddingVersion);
        }

        // Curve-only refit: ranks shift but raw ratings don't, so no shard
        // fetch is needed. When ratings also changed, the ratings rewrite below
        // refreshes both, so skip these to avoid a double pass.
        let curveRankUpdated = 0;
        if (curveChanged && !ratingsChanged) {
            if (hasWorkingSet) curveRankUpdated = remapCurveOnlyWorkingSet(size, history);
            await remapCurveOnlyTail(size);
        }

        // Does the cold tail carry this size's rows to re-derive? (ratings change
        // only). Decided BEFORE the long embedding walk; the tail rows themselves
        // are rewritten from problem_params after it.
        const tailNeedsRederive = ratingsChanged && await tailHasRowsForSize(size);

        // Ratings refresh reads the single problem_params.bin asset (rating/slope/
        // floor by id, both boards) — built ONCE per size, replacing the shard walk
        // for the rating fields. Embeddings still walk the shards below. Null on a
        // transient fetch failure -> bail so the version stays un-stamped and retries
        // on the next load. Skipped when this size has no rows anywhere (neither
        // working set nor tail).
        let problemParams = null;
        if (ratingsChanged && (hasWorkingSet || tailNeedsRederive)) {
            problemParams = await loadProblemParams();
            if (!problemParams) return { count: curveRankUpdated, ok: false };
        }

        const totalRatingsUpdated = (ratingsChanged && hasWorkingSet)
            ? rewriteWorkingSetRatings(size, history, problemParams)
            : 0;

        let totalFilled = 0;
        let totalFilledPicked = 0;
        if (embeddingsChanged && hasWorkingSet) {
            const startIdx = Math.max(0, history.length - KEEP_EMBEDDING_LAST);
            stripOldEmbeddingFields(size, history, startIdx);
            // Correct-move embeddings straight from the lookup-aligned sidecar —
            // zero shard zips. Only pickedEmbedding still lives in the per-shard
            // bins, addressed by the STORED pickedCellIndex, which the wipe
            // keeps: it indexes the SGF's root variations, so a REFIT (new
            // axes, new embedding_version) cannot move it. It is NOT immune to
            // the SGF itself changing — reordering a problem's root variations
            // silently re-points every stored index for it, and the refetch
            // below would then return a different cell's vector. Only
            // fillSyncedHistoryFields re-derives the index (from the reported
            // move, on adopt/sync); a locally-recorded entry keeps whatever it
            // was given. So a corpus surgery that reorders variations either
            // rides a state-epoch bump or knowingly leaves those entries stale.
            // The zip walk shrinks to shards carrying such a need.
            totalFilled = await fillCorrectEmbeddingsViaSidecar(history.slice(startIdx));
            const pickedNeedIds = collectPickedNeedIds(history, startIdx);
            if (pickedNeedIds.size > 0) {
                totalFilledPicked = await walkPickedShards(size, history, startIdx, pickedNeedIds);
            }
        }

        if (totalFilled > 0 || totalFilledPicked > 0 || totalRatingsUpdated > 0) {
            persistAnswerHistory(size, history);
            console.log(`backfill[${size}]: filled correct=${totalFilled} picked=${totalFilledPicked} ratings=${totalRatingsUpdated}`);
        }

        if (tailNeedsRederive && problemParams) {
            await rederiveTailRatings(size, problemParams);
        }
        return { count: totalFilled + totalFilledPicked + totalRatingsUpdated + curveRankUpdated, ok: true };
    } catch (err) {
        console.error(`backfill[${size}] failed`, err);
        try { Sentry.captureException(err); } catch (_) {}
        return { count: 0, ok: false };
    }
}

// Embedding wipe — only when embedding_version changed. Rating-only
// refreshes keep the trained model intact.
//
// Clears EMBEDDING_VECTOR_FIELDS, so `pickedCellIndex` survives to address the
// refill — see the field lists at the top for why that one is not fit-derived.
// What makes a survivor here worse than a missed refill: the fills are
// fill-if-MISSING, so a vector this pass leaves behind is never refetched, and it
// reaches the weakness fit through beta_picked — which the pane reads as part of
// the SUM and selection reads as the CONTRAST, so one stale field moves both the
// reported weakness and the problems served for it.
function wipeStaleEmbeddings(size, history, serverEmbeddingVersion) {
    let cleared = 0;
    for (const a of history) {
        if (deleteFields(a, EMBEDDING_VECTOR_FIELDS)) cleared++;
    }
    if (cleared > 0) {
        // Nothing to wipe beside the embeddings: the motif readings are derived
        // on demand from whatever entries still carry one.
        persistAnswerHistory(size, history);
        console.log(`backfill[${size}]: cleared ${cleared} stale embeddings (-> ${serverEmbeddingVersion})`);
    }
    for (const k of STALE_FLAG_KEYS(size)) localStorage.removeItem(k);
}

// Curve-only working-set remap: recompute problemRank straight from the stored
// raw problemRating. Returns the number of entries whose rank moved.
function remapCurveOnlyWorkingSet(size, history) {
    let curveRankUpdated = 0;
    for (const a of history) {
        if (typeof a.problemRating !== 'number') continue;
        const newRank = rankForSize(a.problemRating, size);
        if (newRank !== null && a.problemRank !== newRank) {
            a.problemRank = newRank;
            curveRankUpdated++;
        }
    }
    if (curveRankUpdated > 0) {
        persistAnswerHistory(size, history);
        console.log(`backfill[${size}]: recomputed problemRank for ${curveRankUpdated} entries (curve change)`);
    }
    return curveRankUpdated;
}

// Curve-only tail remap: recompute the cold tail's problemRank straight from
// its stored raw rating — no shard fetch needed. (When ratings ALSO changed,
// the shard-backed tail re-derive rewrites rank from the fresh rating, so the
// caller skips this.)
async function remapCurveOnlyTail(size) {
    let tailUpdated = 0;
    await updateTail((entry) => {
        if (!entry || entry.size !== size) return null;
        if (typeof entry.problemRating !== 'number') return null;
        const newRank = rankForSize(entry.problemRating, size);
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

// Working-set ratings rewrite: one pass over the FULL history (no keep-window
// cap — the chart needs old entries updated), reading the current value from
// problem_params. Present problems get the CURRENT rating (+ derived rank,
// slope/quality, floor via the shared applyRatingFields); a problem that left
// the corpus keeps its stale value for pruneRemovedProblems to strip. Edited
// in place; the caller persists alongside any embedding fills. Returns the
// number of entries whose rating actually changed.
function rewriteWorkingSetRatings(size, history, problemParams) {
    let totalRatingsUpdated = 0;
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
    return totalRatingsUpdated;
}

// Does the cold tail carry this size's rows? The tail's ratings are frozen at
// eviction time, so after a re-rate the old attempts carry stale (and
// device-dependent — each device evicted at a different rating era) ratings,
// and the Rank-tab chart plots them.
async function tailHasRowsForSize(size) {
    const tail = await getTail();
    for (const e of tail) {
        if (e && e.size === size && typeof e.problemId === 'string') {
            return true;
        }
    }
    return false;
}

// Picked-embedding needs: in-window wrong attempts with a stored
// pickedCellIndex but no pickedEmbedding. These are the only embedding fills
// that still drive a shard-zip walk — the per-problem byte offsets live in the
// zip's embeddings_index.json. (Correct-move embeddings come from the
// lookup-aligned sidecar; ratings from problem_params.)
function collectPickedNeedIds(history, startIdx) {
    const ids = new Set();
    for (let i = startIdx; i < history.length; i++) {
        const a = history[i];
        if (a && a.problemId && a.correct === false &&
            typeof a.pickedCellIndex === 'number' && !a.pickedEmbedding) {
            ids.add(a.problemId);
        }
    }
    return ids;
}

// Strip stray embeddings from entries older than the keep window —
// only on embedding refits (this is a wipe, not a refill).
function stripOldEmbeddingFields(size, history, startIdx) {
    let strippedOld = 0;
    for (let i = 0; i < startIdx; i++) {
        const a = history[i];
        if (!a) continue;
        if (deleteFields(a, EMBEDDING_ALL_FIELDS)) strippedOld++;
    }
    if (strippedOld > 0) {
        persistAnswerHistory(size, history);
        console.log(`backfill[${size}]: stripped embedding fields from ${strippedOld} old entries (beyond keep-last ${KEEP_EMBEDDING_LAST})`);
    }
}

// Picked-embedding shard walk: the per-problem byte offsets for the per-cell
// bins live in each zip's embeddings_index.json, so this is the one embedding
// fill still on the zips. The caller skips it entirely when nothing in the
// keep window needs a fill.
async function walkPickedShards(size, history, startIdx, pickedNeedIds) {
    let filledPicked = 0;
    await walkShards(size, pickedNeedIds, async ({ zip, zipFilename }) => {
        const indexPayload = await parseEmbeddingsIndex(zip, (e) => {
            console.warn(`backfill[${size}]: parse embeddings_index.json failed for ${zipFilename}:`, e);
        });
        if (!indexPayload || !indexPayload.problems) return;
        // Sequential per-entry WITHIN the shard: Range requests are cheap
        // (~256B), and with shards running SHARD_WALK_CONCURRENCY-wide this
        // bounds the total in-flight request count.
        for (let i = startIdx; i < history.length; i++) {
            if (await fillPickedEmbeddingFromSidecar(history[i], { indexPayload, zipFilename, size })) {
                filledPicked++;
            }
        }
    }, (zipFilename, err) => {
        console.warn(`backfill[${size}]: failed for ${zipFilename}:`, err);
    });
    return filledPicked;
}

// Re-derive the cold tail from problem_params (ratings change only). Present
// problems get the CURRENT rating; a problem that left the corpus has its rating
// fields STRIPPED — never leave a stale rating on a removed problem — so
// flattenAttempts drops the now-rating-less attempt from the chart. One IDB
// cursor pass. (The working-set copy wins in getMergedFullAnswerHistory, so an
// attempt present in both is unaffected here; this pass handles the tail-ONLY
// attempts.)
async function rederiveTailRatings(size, problemParams) {
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

// --- Synced-entry field fill (account-sync) ----------------------------------
// account-sync merges /sync rows as skeleton entries: problemId, correct, date,
// timing — none of the model fields. This fills them, independent of the
// version bookkeeping above (the versions haven't changed; the ENTRIES are
// new):
//   - rating/rank/slope/quality/floor for every synced entry missing them,
//     from problem_params.bin (no shard fetch)
//   - correct-move `embedding` inside the keep-last window, from the
//     lookup-aligned sidecar (~130-byte Range per problem, no shard fetch)
//   - `pickedCellIndex` derived from the reported move (`syncedMove` — set on
//     wrong attempts whose reported coordinate is in the canonical
//     orientation-0 frame: all rows since CANONICAL_MOVE_ERA_MS, plus older
//     non-repeats, which were always served at identity; see
//     account-sync.mergeRowsForSize). This is the one fill that still needs
//     the shard ZIP (the SGF). The SGF root variations are the same order
//     extraction/packaging use, so the index addresses the per-cell bin
//     directly.
//   - `pickedEmbedding` via the per-cell bin, addressed by the zip's
//     embeddings_index.json — fetched while that zip is open anyway.
//
// `includeTail` (set on a full pull) also fills synced skeletons that landed in
// the IndexedDB cold tail — a full pull evicts its old rows there mid-pull, out
// of reach of the working-set walk. Incremental pulls only merge recent rows
// (working set), so they leave it false and skip the tail scan.
// An includeTail fill that didn't complete (params fetch failed mid-adoption,
// tab killed). Incremental pulls never pass includeTail, so without this marker
// the tail's synced skeletons would stay unfilled until the next full pull —
// the marker makes every subsequent fill resume the tail until it succeeds.
const SYNC_TAIL_FILL_PENDING_KEY = 'syncTailFillPending';

export async function fillSyncedHistoryFields(sizes, { includeTail = false } = {}) {
    if (localStorage.getItem(SYNC_TAIL_FILL_PENDING_KEY)) includeTail = true;
    if (includeTail) localStorage.setItem(SYNC_TAIL_FILL_PENDING_KEY, '1');

    // Rating fields come from the single problem_params.bin asset — the same
    // source the version-driven rating refresh reads — loaded ONCE and shared
    // by the working-set and tail fills. Routing them through the shard walk
    // instead (as this used to) meant an adopted account's WHOLE career drove
    // the walk: real 2k-attempt accounts span 100-374 shard zips (~2MB each),
    // and downloading + unzipping hundreds of MB crashed the tab on low-memory
    // devices. A null map (offline / fetch failure) just leaves skeletons for
    // a later pull to fill.
    let problemParams = null;
    const needsParams = includeTail || sizes.some((size) =>
        SUPPORTED_BOARD_SIZES.includes(size) &&
        getAnswerHistoryForSize(size).some((a) =>
            a && a.synced && a.problemId && typeof a.problemRating !== 'number'));
    if (needsParams) problemParams = await loadProblemParams();

    let total = 0;
    for (const size of sizes) {
        if (!SUPPORTED_BOARD_SIZES.includes(size)) continue;
        try {
            total += await fillSyncedForSize(size, problemParams);
        } catch (err) {
            console.error(`synced-fill[${size}] failed`, err);
            try { Sentry.captureException(err); } catch (_) {}
        }
    }
    if (includeTail) {
        // A resumed tail fill may involve sizes the triggering pull didn't
        // touch, so always scan every size (each size's scan is cheap and
        // no-ops when nothing is missing).
        const r = await fillSyncedTailFields(SUPPORTED_BOARD_SIZES, problemParams);
        total += r.filled;
        if (r.ok) localStorage.removeItem(SYNC_TAIL_FILL_PENDING_KEY);
    }
    return total;
}

async function fillSyncedForSize(size, problemParams = null) {
    const history = getAnswerHistoryForSize(size);
    if (!history || history.length === 0) return 0;
    const startIdx = Math.max(0, history.length - KEEP_EMBEDDING_LAST);

    let filled = 0;
    // Rating fields for EVERY synced entry straight from problem_params — no
    // shard fetch. Ids absent from the corpus stay skeletons (the problem left
    // the set), matching what the shard walk's lookup drop used to do.
    if (problemParams) {
        for (const a of history) {
            if (!a || !a.synced || !a.problemId) continue;
            if (typeof a.problemRating === 'number') continue;
            const item = problemParams.get(a.problemId);
            if (!item || typeof item.rating !== 'number') continue;
            applyRatingFields(a, item, size);
            filled++;
        }
    }

    // Correct-move embeddings for in-window synced entries via the
    // lookup-aligned sidecar — a ~130-byte Range per distinct problem, no zips.
    const embedNeedy = [];
    for (let i = startIdx; i < history.length; i++) {
        const a = history[i];
        if (a && a.synced && a.problemId && !a.embedding) embedNeedy.push(a);
    }
    filled += await fillCorrectEmbeddingsViaSidecar(embedNeedy);

    // The zip walk remains ONLY where the zip's contents are genuinely needed:
    // deriving pickedCellIndex from a synced move coordinate (needs the SGF)
    // and the picked-embedding fill (byte offsets live in the zip's
    // embeddings_index.json). Wrong attempts are a minority and skew
    // rank-local, so this walk touches a fraction of the shards the correct
    // fills used to.
    const needIds = new Set();
    for (let i = startIdx; i < history.length; i++) {
        const a = history[i];
        if (!a || !a.synced || !a.problemId || a.correct !== false) continue;
        const needsPickedIndex =
            typeof a.syncedMove === 'string' && typeof a.pickedCellIndex !== 'number';
        const needsPicked = !a.pickedEmbedding &&
            (typeof a.pickedCellIndex === 'number' || needsPickedIndex);
        if (needsPickedIndex || needsPicked) {
            needIds.add(a.problemId);
        }
    }
    if (needIds.size === 0) {
        if (filled > 0) {
            persistAnswerHistory(size, history);
            console.log(`synced-fill[${size}]: filled ${filled} fields on synced entries`);
        }
        return filled;
    }
    await walkShards(size, needIds, async ({ zip, zipFilename, shardIds }) => {
        // A parse failure = shard without a usable sidecar index; the
        // picked-embedding fill below just skips (no console line).
        const indexPayload = await parseEmbeddingsIndex(zip);
        const idsHere = new Set(shardIds);

        for (let i = startIdx; i < history.length; i++) {
            const a = history[i];
            if (!a || !a.synced || !idsHere.has(a.problemId)) continue;

            if (a.correct === false && typeof a.syncedMove === 'string' &&
                typeof a.pickedCellIndex !== 'number') {
                const idx = await cellIndexForMove(zip, a.problemId, a.syncedMove, size);
                if (typeof idx === 'number' && idx >= 0) a.pickedCellIndex = idx;
            }
            if (a.correct === false && typeof a.syncedMove === 'string') {
                // One-shot either way: an unresolvable move (illegal after a
                // corpus change, parse failure) would otherwise re-walk this
                // shard on every sync.
                delete a.syncedMove;
            }

            if (await fillPickedEmbeddingFromSidecar(a, { indexPayload, zipFilename, size })) {
                filled++;
            }
        }
    }, (zipFilename, err) => {
        console.warn(`synced-fill[${size}]: failed for ${zipFilename}:`, err);
    });

    if (filled > 0) {
        persistAnswerHistory(size, history);
        console.log(`synced-fill[${size}]: filled ${filled} fields on synced entries`);
    }
    return filled;
}

// Map a reported board-notation move (canonical orientation-0 frame) to the
// problem's variation index — the same `children.indexOf(moveNode)` the live
// click path records, and the index the per-cell sidecar is keyed by.
//
// The tsumego SGFs are full game records: the main line builds the position
// and the candidate first moves hang off the LAST main-line node, NOT the
// root. Walk the single-child chain to that branch point exactly like the
// live path (board.js loadSgf: `while (node.children.length === 1) next()`)
// and the pipeline's parse_tsumego_sgf, so the derived index matches the
// per-cell bin. (Indexing root.children instead — as this originally did —
// found only the first main-line move, so every derivation returned -1.)
async function cellIndexForMove(zip, problemId, move, size) {
    const xy = boardNotationToXy(move, size);
    if (!xy) return null;
    const sgfFile = zip.file(`${problemId}.sgf`);
    if (!sgfFile || typeof WGo === 'undefined') return null;
    try {
        const kifu = WGo.Kifu.fromSgf(await sgfFile.async('text'));
        let node = kifu && kifu.root;
        while (node && node.children && node.children.length === 1) node = node.children[0];
        const children = (node && node.children) || [];
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
    const rank = rankForSize(item.rating, size);
    if (rank !== null && rank !== undefined) e.problemRank = rank;
    if (typeof item.slope === 'number') {
        e.problemSlope = item.slope;
        e.problemQuality = problemInfoWeight(item.slope, item.floor);
    }
    if (typeof item.floor === 'number') e.problemFloor = item.floor;
}

// Fill one size's cold-tail synced skeletons from problem_params (only the
// rating fields — the tail never carries embeddings). Returns { filled, ok };
// ok=false when the params map is unavailable (offline), so a caller can retry
// rather than treat the fill as complete.
async function fillSyncedTailForSize(size, problemParams) {
    const tail = await getTail();
    if (!tail || tail.length === 0) return { filled: 0, ok: true };
    let needed = false;
    for (const e of tail) {
        if (!e || !e.synced || !e.problemId) continue;
        if (entrySize(e) !== size) continue;
        if (typeof e.problemRating !== 'number') { needed = true; break; }
    }
    if (!needed) return { filled: 0, ok: true };
    if (!problemParams) return { filled: 0, ok: false };

    let filled = 0;
    await updateTail((entry) => {
        if (!entry || !entry.synced || typeof entry.problemRating === 'number') return null;
        if (entrySize(entry) !== size) return null;
        const item = problemParams.get(entry.problemId);
        if (!item || typeof item.rating !== 'number') return null;
        applyRatingFields(entry, item, size);
        filled++;
        return entry;
    });
    if (filled > 0) console.log(`synced-tail-fill[${size}]: filled ${filled} cold-tail entries`);
    return { filled, ok: true };
}

// Fill the cold tail's synced skeletons across sizes. Returns { filled, ok }
// (ok=false if the params asset couldn't be fetched while fills were needed).
// Exported so the one-time tail migration can reuse it on already-damaged
// tails; when the caller has no params map at hand, it is loaded here.
export async function fillSyncedTailFields(sizes, problemParams = null) {
    let filled = 0;
    let ok = true;
    if (!problemParams) problemParams = await loadProblemParams();
    for (const size of sizes) {
        if (!SUPPORTED_BOARD_SIZES.includes(size)) continue;
        try {
            const r = await fillSyncedTailForSize(size, problemParams);
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
