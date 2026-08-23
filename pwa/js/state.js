// js/state.js
import { APP_MODES, ANSWER_HISTORY_WORKING_SET, MODEL_INPUT_HISTORY_LENGTH, EMBEDDING_VECTOR_FIELDS } from './constants.js';
import { appendTail, getTail, countTail, requestPersist, clearTail } from './history-store.js';
import { isQuotaError, readJSON, readJSONObject, writeJSON } from './persist.js';
import { runStateEpochGate } from './state-epoch.js';
import { coldStartEstimate } from '../shared/rank-window.js';

const SUPPORTED_BOARD_SIZES = [9, 13];
const DEFAULT_BOARD_SIZE = 9;

// --- Import-time state-epoch gate (js/state-epoch.js) ---
// The permanent replacement for the migration ladders: a stale/absent epoch
// wipes the account-derived localStorage and routes init through the standard
// account-adopt restore. MUST stay above the state reads — a pre-epoch shape
// would crash the normal read path below. Dependency-light (imports nothing).
runStateEpochGate();

// --- Board size + per-size backing buckets ---

// Number of most-recent entries that should keep their `embedding` field.
// Older entries get their embedding stripped on quota pressure (the entry
// itself stays — preserving stats like novel-problem count and the recall
// model's history). MODEL_INPUT_HISTORY_LENGTH is what the trainers actually
// use, so the two can't drift.
const ANSWER_HISTORY_EMBEDDING_KEEP_LAST = MODEL_INPUT_HISTORY_LENGTH;

const perSize = {
    // The always-assigned challenge problem (full object incl. sgf), persisted
    // so a reopen restores the user's standing challenge. currentProblem is NOT
    // persisted — it's recomputed on load (reopen shows a due recall or this
    // challenge; see problem-loader.advanceToNextProblem).
    challengeProblem: readJSONObject('challengeProblem'),
    // The last-completed problem (full object incl. sgf), persisted so a reopen
    // keeps the Review button live — review/share/ignore act on it, and it is
    // never re-served as a problem to solve, so persisting it can't re-serve an
    // already-answered problem.
    previousProblem: readJSONObject('previousProblem'),
};

// --- Unified answer history (the synchronous working set) ---
// One date-ordered array across both board sizes, each entry tagged with
// `size`. This is the WORKING SET — the most recent ANSWER_HISTORY_WORKING_SET
// attempts, kept in localStorage so the hot path never awaits. Older attempts
// are evicted to IndexedDB (history-store.js) and merged back only for the
// full-history charts via getMergedFullAnswerHistory(). The active-size view
// that most consumers read (state.answerHistory, getAnswerHistoryForSize) is
// derived by filtering on `size`.

function loadUnifiedHistory() {
    const raw = localStorage.getItem('answerHistory');
    if (raw === null) return [];
    let parsed;
    try { parsed = JSON.parse(raw); } catch { return []; }
    if (Array.isArray(parsed)) return parsed;
    // Legacy {9:[...], 13:[...]} bucket shape (left by the board-size migrations
    // above): flatten, tag each entry with its bucket's size, sort by date.
    // initHistoryStorage() rewrites localStorage as the unified array + evicts
    // the tail on this same boot.
    if (parsed && typeof parsed === 'object') {
        const merged = [];
        for (const sz of Object.keys(parsed)) {
            const arr = parsed[sz];
            if (!Array.isArray(arr)) continue;
            const size = Number(sz);
            for (const a of arr) merged.push(a && a.size != null ? a : { ...a, size });
        }
        merged.sort((x, y) => (x.date || 0) - (y.date || 0));
        return merged;
    }
    return [];
}

let unifiedHistory = loadUnifiedHistory();
let activeView = null;        // cached state.answerHistory (filtered by active size)
let activeViewSize = null;

function invalidateActiveView() { activeView = null; }

// Persist the working set. On quota pressure, strip the heavy embedding fields
// from entries beyond the keep-last window and retry — a belt-and-suspenders
// that effectively never fires now that the working set is capped.
function persistUnifiedHistory() {
    try {
        localStorage.setItem('answerHistory', JSON.stringify(unifiedHistory));
        return;
    } catch (e) {
        if (!isQuotaError(e)) { console.error('answerHistory write failed', e); return; }
    }
    let stripped = 0;
    const cutoff = Math.max(0, unifiedHistory.length - ANSWER_HISTORY_EMBEDDING_KEEP_LAST);
    for (let i = 0; i < cutoff; i++) {
        const a = unifiedHistory[i];
        if (!a) continue;
        // Vectors only: `pickedCellIndex` is what a later refill is looked up
        // by, and a working-set row is still in refill range.
        for (const f of EMBEDDING_VECTOR_FIELDS) {
            if (a[f] !== undefined) { delete a[f]; stripped++; }
        }
    }
    try {
        localStorage.setItem('answerHistory', JSON.stringify(unifiedHistory));
        if (stripped > 0) console.warn(`answerHistory quota relieved by stripping ${stripped} old embeddings`);
    } catch {
        console.error('answerHistory still over quota after stripping; write skipped');
    }
}

// Single reconciliation point: replace every entry of `size` in the unified
// array with `history` (tagged `size`), keep date order, persist, schedule
// eviction. Entries already carrying the right `size` keep their object
// identity, so in-place mutations elsewhere (e.g. the refit) survive.
function reconcileSize(size, history) {
    const next = [];
    for (const a of unifiedHistory) if (a.size !== size) next.push(a);
    for (const a of history) next.push(a && a.size === size ? a : { ...a, size });
    next.sort((x, y) => (x.date || 0) - (y.date || 0));
    unifiedHistory = next;
    invalidateActiveView();
    persistUnifiedHistory();
    scheduleEvict();
}

let evictTimer = null;
let evicting = false;
function scheduleEvict() {
    if (evictTimer) return;
    evictTimer = setTimeout(() => { evictTimer = null; evictColdTail(); }, 2000);
}

// High-water mark of how many rows we've written to the IndexedDB tail (since
// the last clear). The actual tail count dropping below this means the browser
// silently evicted history — surfaced to the user on the stats panel. Updated
// only on append (monotonic) and reset by a clear or a server refresh.
const TAIL_EXPECTED_KEY = 'historyTailExpected';
// Set once we've asked for durable storage (so we ask at most once, and only
// when there is actually a tail worth protecting).
const PERSIST_REQUESTED_KEY = 'historyPersistRequested';

// A row the offline outbox still needs to upload: reportable (real problem id)
// but never acked by the server. account-sync's outbox scan and claim only see
// the working set, so such a row must never be evicted to the cold tail — once
// there it would be permanently un-uploadable. Acked = carries a serverId (the
// row id; mirror-by-id is the only representation).
export function isUnackedReportable(a) {
    return !!a && !Number.isFinite(a.serverId)
        && typeof a.problemId === 'string' && /^[0-9a-f]{8}$/.test(a.problemId);
}

// --- Ignored-problems list (durable across the epoch wipe / re-adoption) ---
// The runtime representation is the `ignored` flag on a problem's LAST
// answerHistory entry (recall-model.js reads lastAnswer.ignored to exclude recalls).
// That flag round-trips nowhere server-side, so the curated list is ALSO kept
// as a standalone id set: performIgnore adds to it, the state-epoch gate
// carries it through the wipe, and applyStoredIgnoredFlags() re-marks the
// restored history after an adopt pull.
const IGNORED_PROBLEMS_KEY = 'ignoredProblems_v1';

export function addIgnoredProblem(problemId) {
    if (typeof problemId !== 'string') return;
    const set = new Set(readJSON(IGNORED_PROBLEMS_KEY) || []);
    if (set.has(problemId)) return;
    set.add(problemId);
    writeJSON(IGNORED_PROBLEMS_KEY, [...set]);
}

export function applyStoredIgnoredFlags() {
    const ids = readJSON(IGNORED_PROBLEMS_KEY);
    if (!Array.isArray(ids) || ids.length === 0) return;
    const ignored = new Set(ids);
    const lastByProblem = new Map();
    for (const a of unifiedHistory) {
        if (a && ignored.has(a.problemId)) lastByProblem.set(a.problemId, a);
    }
    let changed = false;
    for (const a of lastByProblem.values()) {
        if (!a.ignored) { a.ignored = true; changed = true; }
    }
    if (changed) {
        invalidateActiveView();
        persistUnifiedHistory();
    }
}

// Move attempts beyond the cap into the IndexedDB tail, OLDEST first.
// Crash-safe: append to IDB and confirm BEFORE trimming the in-memory /
// localStorage working set, so a kill mid-evict leaves a (de-duped) copy in
// both rather than a gap. Removes exactly the appended rows by reference, so a
// concurrent append during the await is preserved. Un-acked reportable rows are
// held back (the working set stays slightly over the cap until they're acked
// and a later evict moves them) — a consequence is that the tail's internal
// order is only approximately date-ordered; every tail consumer either sorts
// (getMergedFullAnswerHistory) or is order-independent.
export async function evictColdTail() {
    if (evicting) return;
    if (unifiedHistory.length <= ANSWER_HISTORY_WORKING_SET) return;
    evicting = true;
    try {
        const overflow = unifiedHistory
            .slice(0, unifiedHistory.length - ANSWER_HISTORY_WORKING_SET)
            .filter(a => !isUnackedReportable(a));
        if (overflow.length === 0) return;
        const ok = await appendTail(overflow);
        if (!ok) return; // IDB unavailable — keep the rows in the working set
        const overflowSet = new Set(overflow);
        unifiedHistory = unifiedHistory.filter(a => !overflowSet.has(a));
        invalidateActiveView();
        persistUnifiedHistory();
        // Bump the eviction high-water mark and, the first time a tail exists,
        // request durable storage (so the browser won't silently evict the
        // expensive-to-re-pull deep history out from under us).
        const expected = (parseInt(localStorage.getItem(TAIL_EXPECTED_KEY), 10) || 0) + overflow.length;
        localStorage.setItem(TAIL_EXPECTED_KEY, String(expected));
        if (!localStorage.getItem(PERSIST_REQUESTED_KEY)) {
            localStorage.setItem(PERSIST_REQUESTED_KEY, '1');
            requestPersist().catch(() => {});
        }
    } finally {
        evicting = false;
    }
}

// Full career = IDB tail (older) ⧺ working set (newer), de-duped by
// (problemId, date) with the working set winning, in date order. Returns
// shallow copies (read-only). Used only by the full-history charts.
export async function getMergedFullAnswerHistory() {
    const tail = await getTail();
    const seen = new Set(unifiedHistory.map(a => `${a.problemId}|${a.date}`));
    const merged = [];
    for (const a of tail) {
        const k = `${a.problemId}|${a.date}`;
        if (seen.has(k)) continue;
        seen.add(k);   // also de-dup WITHIN the tail — it can hold exact (problemId,date)
                       // dup rows that would otherwise double-plot
        merged.push(a.size != null ? { ...a } : { ...a, size: 9 });
    }
    for (const a of unifiedHistory) merged.push({ ...a });
    merged.sort((x, y) => (x.date || 0) - (y.date || 0));
    return merged;
}

// One-time at boot (after runMigrations): normalize localStorage to the unified
// array and push anything beyond the cap into the tail, freeing localStorage.
// Idempotent — safe to call every boot.
export async function initHistoryStorage() {
    await evictColdTail();
    persistUnifiedHistory();
    // Baseline the eviction high-water mark when none has been recorded yet (no
    // prior eviction, or a tail that needed no eviction this boot): assume the
    // tail is intact now and start watching from here.
    if (localStorage.getItem(TAIL_EXPECTED_KEY) === null) {
        try { localStorage.setItem(TAIL_EXPECTED_KEY, String(await countTail())); } catch {}
    }
}

// Silent-eviction check for the stats panel: `missing` is how many tail rows the
// browser dropped since our last write (expected high-water mark − actual count).
export async function getTailEvictionStatus() {
    const expected = parseInt(localStorage.getItem(TAIL_EXPECTED_KEY), 10) || 0;
    const actual = await countTail();
    return { expected, actual, missing: Math.max(0, expected - actual) };
}

// Clear the IndexedDB tail AND its tracking (account switch). The new account
// re-establishes its own high-water mark + persist request on first eviction.
export async function clearHistoryTail() {
    await clearTail();
    localStorage.removeItem(TAIL_EXPECTED_KEY);
    localStorage.removeItem(PERSIST_REQUESTED_KEY);
}

// After a server refresh has rebuilt the tail, reset the high-water mark to the
// restored actual count so the recovered rows aren't double-counted and the
// missing-row signal clears.
export async function syncTailExpectedToActual() {
    try { localStorage.setItem(TAIL_EXPECTED_KEY, String(await countTail())); } catch {}
}

// --- Shared-across-sizes state (single values, not bucketed). ---
// The single shared rank is the window-MLE estimate (rank-decomposition
// .estimateWindowRank); the recall (modelParams) and weakness (weaknessParams/
// Se/AxisCov) models train on the union of both sizes' histories. The adamState
// and weaknessAdam slots beside them are legacy blobs kept only so the wipe
// lists still clear them — neither fitter carries optimizer state any more. See
// mergeSharedAcrossSizes_v2.
const shared = {
    userRank: readJSON('userRank', null),
    // Three ranks, all in the same internal units, all from one posterior
    // (shared/rank-window.js estimateWindowRankFromFlat):
    //   userRank    the posterior MEAN. The honest, calibrated number — server
    //               translations, the recall and weakness models and the stats panes
    //               all read this, and it is what the Worker stores as pred_rank.
    //   displayRank what the user is TOLD: conservative while the posterior is
    //               wide, converging onto userRank as it tightens. Everything the
    //               user reads as "my rank" derives from this — the badge, the
    //               rank-change overlay, and the server translations, which have to
    //               translate the shown number or they contradict the badge. It must
    //               NOT flow back into anything fitted (the rank-map couplings, the
    //               leaderboard's pred_rank), or the calibration loop closes on a
    //               deliberately shifted number.
    //   serveRank   what problem SELECTION targets: information-seeking (above
    //               userRank) but capped so it never runs more than serveGapCap()
    //               above displayRank — one rank per consecutive clear, reset by any
    //               miss. The cap is what stops a beginner being handed dan problems,
    //               and it binds for most of onboarding, so it is the cap rather than
    //               the UCB term that sets how fast a strong user is found.
    serveRank: readJSON('serveRank', null),
    displayRank: readJSON('displayRank', null),
    // Posterior SE of the window rank estimate (rank-index units), stored with
    // each re-fit. Selection reads it to decide ONE thing: whether the INFORMATION
    // slot (slope prioritization) is in the serve draw, which it is while
    // se >= 2x the local floor. The motif and neutral slots don't consult it.
    rankSe: readJSON('rankSe', null),
    modelParams: readJSON('modelParams', null),
    // All five are LEGACY blobs; nothing writes them. The recall fitter no longer
    // carries optimizer state across retrains (it converges from a cold start),
    // and the weakness model they belonged to is gone entirely — the motif
    // readings are now a pure function of the answer history, derived on demand
    // by motif-stats.js with no trained parameters to persist. The slots stay
    // only so the clear/wipe paths keep removing what older versions stored.
    adamState: readJSON('adamState', null),
    weaknessParams: readJSON('weaknessParams', null),
    weaknessAdam: readJSON('weaknessAdam', null),
    weaknessSe: readJSON('weaknessSe', null),
    weaknessAxisCov: readJSON('weaknessAxisCov', null),
};

// The 6 model/weakness shared values below share an identical accessor pair —
// `get` returns the stored value or null, `set` assigns and persists. They're
// defined on `state` via a loop after the object literal
// to avoid six copies of the same template. userRank and serveRank differ (they
// have non-null fallbacks), so they keep explicit accessors in the literal.
const NULLABLE_SHARED_KEYS = [
    'modelParams', 'adamState',
    'weaknessParams', 'weaknessAdam', 'weaknessSe', 'weaknessAxisCov',
];

// Per-size session state — not persisted, but bucketed so it doesn't bleed
// across sizes if the user switches mid-session.
const sessionPerSize = {
    unavailableProblemIds: {},  // size -> Set
    accumulatedSolvingTime: {}, // size -> number
    currentSolvingStartTime: {},// size -> number|null
};

function getOrCreateSessionSet(key) {
    const size = state.boardSize;
    if (!sessionPerSize[key][size]) sessionPerSize[key][size] = new Set();
    return sessionPerSize[key][size];
}

let _boardSize = (() => {
    const stored = parseInt(localStorage.getItem('boardSize'));
    return SUPPORTED_BOARD_SIZES.includes(stored) ? stored : DEFAULT_BOARD_SIZE;
})();

// The userId doubles as the account credential: entering it on another device
// logs that device into the same server-side history (see account-sync.js).
// That makes it a read capability, not just a write identity, so new ids must
// be crypto-strength — Math.random() is predictable. Existing ids are
// grandfathered as-is. 8 chars of base36 ≈ 41 bits; enumeration is bounded by
// the Worker's per-IP rate limits.
export const generateUserId = () => {
    const alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789';
    if (globalThis.crypto && typeof crypto.getRandomValues === 'function') {
        let id = '';
        while (id.length < 8) {
            const bytes = crypto.getRandomValues(new Uint8Array(16));
            for (const v of bytes) {
                // Rejection-sample: 252 = 7·36, so v % 36 is unbiased below it.
                if (id.length < 8 && v < 252) id += alphabet[v % 36];
            }
        }
        return id;
    }
    return Math.random().toString(36).substring(2, 10);
};

const getOrCreateUserId = () => {
    let userId = localStorage.getItem('userId');
    if (!userId) {
        userId = generateUserId();
        localStorage.setItem('userId', userId);
    } else if (userId.length > 8) {
        userId = userId.substring(0, 8);
        localStorage.setItem('userId', userId);
    }
    return userId;
};

// What a device shows and serves before it has answered anything: the ordinary rank
// rule evaluated on an empty window, which is the population prior exactly. Computed
// once at import — rank-window pulls in only leaf modules (math-utils, irt, rank-math)
// and never state.js, so no cycle. No shard is loaded this early, so the SE floor is
// unmeasurable and the estimate concedes the full posterior width; the boot re-anchor
// (rank-decomposition.reanchorRank, run once the first shard is in) replaces it with
// the measured-floor value before anything the user sees depends on it.
const COLD_START = coldStartEstimate();

export const state = {
    appMode: APP_MODES.SOLVING,

    // --- The three problem slots (see migrateToChallengeModel_v1). ---
    // currentProblem: what's displayed/interacted with now (challenge, recall,
    //   or a linked/shared problem). In-memory only — recomputed on load.
    // previousProblem: the last completed problem; review/share/ignore act on it.
    //   Per-size persisted (getter/setter further down) so the Review button
    //   survives a reopen. A reopen still lands on the challenge/recall, not on
    //   review — but the previous slot stays reviewable.
    // challengeProblem is also a per-size persisted getter/setter further down.
    currentProblem: null,

    // Board and Game state
    board: null,
    kifuReader: null,

    // Tsumego state — per-size in-memory pool. Cleared on size switch.
    zipMetadata: [],
    loadedZips: {},
    allProblems: [],

    userId: getOrCreateUserId(),

    acceptClicks: false,
    deferredInstallPrompt: null,
    userHasInteracted: false,

    // Set when /report responds with an X-App-Version that differs from the
    // build-time APP_VERSION. The next time the app is quiescent (not in
    // analysis/review, no animation in flight) maybeReload() picks this up
    // and refreshes the page to pull the new bundle. See setAcceptClicks.
    reloadPending: false,

    initialAnalysisNode: null,
    ghostMove: null,
    analysisHintMove: null,
    candidateMoves: [],
    reviewStartTime: null,

    statsResults: null,

    // Unix-ms timestamp until which recalls are paused (no recall candidates
    // are returned). 0 / NaN / past timestamps all mean "not paused". Shared
    // across board sizes — the recall model itself is shared, so the pause is
    // too. Read directly from localStorage on each access so an expired pause
    // re-enables recalls without a reload.
    get recallPausedUntil() {
        const raw = parseInt(localStorage.getItem('recallPausedUntil'), 10);
        return Number.isFinite(raw) ? raw : 0;
    },
    set recallPausedUntil(ts) {
        if (ts && ts > Date.now()) {
            localStorage.setItem('recallPausedUntil', String(ts));
        } else {
            localStorage.removeItem('recallPausedUntil');
        }
    },

    // Stone-placement sound toggle (hamburger menu). Shared across board
    // sizes. Stored as a "disabled" flag so the missing-key default is ON.
    get audioEnabled() {
        return localStorage.getItem('audioDisabled') === null;
    },
    set audioEnabled(on) {
        if (on) {
            localStorage.removeItem('audioDisabled');
        } else {
            localStorage.setItem('audioDisabled', '1');
        }
    },

    // In-memory cache of labels.components (joint across board sizes —
    // the PCA + ICA fit is joint, so labels are shared). Populated by
    // backfill.js from the inlined embedding_meta.json (data-version.js).
    // Used by the stats panel to name top weakness PCs. Not persisted.
    embeddingLabels: null,

    // Joint embedding meta consumed by the per-cell sidecar reader (K,
    // bytes_per_cell, dtype). Populated by backfill.js from the
    // same fetch that gives us embeddingLabels — same file, joint meta.
    // Null until that fetch has run; readers must short-circuit on null.
    embeddingMeta: null,

    // --- Per-size state (read/write proxies onto the active size's bucket). ---
    get boardSize() { return _boardSize; },
    set boardSize(size) {
        if (!SUPPORTED_BOARD_SIZES.includes(size)) return;
        if (_boardSize === size) return;
        _boardSize = size;
        localStorage.setItem('boardSize', String(size));
    },

    get userRank() {
        return typeof shared.userRank === 'number' ? shared.userRank : COLD_START.rank;
    },
    set userRank(v) {
        shared.userRank = v;
        writeJSON('userRank', v);
    },

    // Before any estimate has run all three slots come from the same empty-window
    // posterior (COLD_START above). They differ on purpose — serving sits above the
    // mean to keep learning, the display below it to avoid a rank that walks back.
    get serveRank() {
        return typeof shared.serveRank === 'number' ? shared.serveRank : COLD_START.serveRank;
    },
    set serveRank(v) {
        shared.serveRank = v;
        writeJSON('serveRank', v);
    },

    // Falls back to the STORED honest rank, never to the userRank getter: that getter's
    // own fallback is the posterior MEAN, and showing the mean is the one thing the
    // display offset exists to prevent while the posterior is prior-wide.
    get displayRank() {
        if (typeof shared.displayRank === 'number') return shared.displayRank;
        if (typeof shared.userRank === 'number') return shared.userRank;
        return COLD_START.displayRank;
    },
    set displayRank(v) {
        shared.displayRank = v;
        writeJSON('displayRank', v);
    },

    get rankSe() {
        return typeof shared.rankSe === 'number' ? shared.rankSe : null;
    },
    set rankSe(v) {
        shared.rankSe = v;
        writeJSON('rankSe', v);
    },

    get challengeProblem() {
        return perSize.challengeProblem[_boardSize] ?? null;
    },
    set challengeProblem(v) {
        perSize.challengeProblem[_boardSize] = v;
        writeJSON('challengeProblem', perSize.challengeProblem, { rethrowNonQuota: true });
    },

    get previousProblem() {
        return perSize.previousProblem[_boardSize] ?? null;
    },
    set previousProblem(v) {
        perSize.previousProblem[_boardSize] = v;
        writeJSON('previousProblem', perSize.previousProblem, { rethrowNonQuota: true });
    },

    // Active-size view, derived from the unified working set and cached until a
    // write or a board-size switch (the size check below picks up switches with
    // no explicit invalidation). Read a few times per attempt, never in a tight
    // loop, so the filter cost is negligible.
    get answerHistory() {
        if (activeView === null || activeViewSize !== _boardSize) {
            activeView = unifiedHistory.filter(a => a.size === _boardSize);
            activeViewSize = _boardSize;
        }
        return activeView;
    },
    set answerHistory(v) {
        reconcileSize(_boardSize, v);
    },

    // modelParams / adamState / weakness{Params,Adam,Se,AxisCov} accessors are
    // generated from NULLABLE_SHARED_KEYS just below the literal (identical
    // get-or-null + set-and-persist template).

    get unavailableProblemIds() { return getOrCreateSessionSet('unavailableProblemIds'); },

    get accumulatedSolvingTime() {
        return sessionPerSize.accumulatedSolvingTime[_boardSize] ?? 0;
    },
    set accumulatedSolvingTime(v) {
        sessionPerSize.accumulatedSolvingTime[_boardSize] = v;
    },

    get currentSolvingStartTime() {
        return sessionPerSize.currentSolvingStartTime[_boardSize] ?? null;
    },
    set currentSolvingStartTime(v) {
        sessionPerSize.currentSolvingStartTime[_boardSize] = v;
    },
};

// Generate the identical get-or-null + set-and-persist accessors for the
// nullable shared values (the recall + weakness model state), so the template
// lives once rather than as six copies in the object literal above.
for (const key of NULLABLE_SHARED_KEYS) {
    Object.defineProperty(state, key, {
        get() { return shared[key] ?? null; },
        set(v) { shared[key] = v; writeJSON(key, v); },
        enumerable: true,
        configurable: true,
    });
}

// Cache for loaded zip files. Cleared on size switch.
export const zipCache = new Map();

// Per-shard embeddings_index payloads (the JSON file shipped inside each
// zip alongside metadata.json). Keyed by the same zipFilename as zipCache.
// Shape: { problems: { tsumego_id: { offset, n_cells } } }. Read by the
// per-cell sidecar fetcher in picked-embedding.js. Cleared on size switch.
export const embeddingsIndexCache = new Map();

// --- Mutators ---

export function setBoard(newBoard) {
    state.board = newBoard;
}

export function setAppMode(mode) {
    state.appMode = mode;
}

export function setKifuReader(kifuReader) {
    state.kifuReader = kifuReader;
}

// Hooks fired on setAcceptClicks(true) — registered from main.js so state.js
// doesn't import UI modules directly.
const acceptClicksHooks = [];
export function registerAcceptClicksHook(fn) {
    acceptClicksHooks.push(fn);
}

export function setAcceptClicks(canAccept) {
    state.acceptClicks = canAccept;
    // After loadTsumego in the wrong-answer path, setAcceptClicks(true) fires
    // before the awaited enterAnalysisMode() flips appMode to ANALYZING. A
    // synchronous reload here would briefly see "SOLVING + acceptClicks" and
    // refresh right when the user is about to enter review. Deferring to a
    // macrotask lets that continuation run first, so the recheck inside
    // maybeReload sees ANALYZING and skips.
    if (canAccept && state.reloadPending) {
        setTimeout(maybeReload, 0);
    }
    if (canAccept) {
        for (const fn of acceptClicksHooks) setTimeout(fn, 0);
    }
}

export function setReloadPending(pending) {
    state.reloadPending = pending;
}

let reloadInFlight = false;

// Resolves once the registration's incoming worker (installing/waiting)
// reaches 'activated' — i.e. the NEW worker now controls what a reload will
// fetch — or after timeoutMs as a fallback. 'redundant' (install failed, or
// superseded by an even newer worker) also resolves rather than hanging. No
// incoming worker resolves immediately: the update check found nothing new
// (or it already activated), so there is nothing to wait for.
function waitForSwActivation(reg, timeoutMs) {
    return new Promise((resolve) => {
        const incoming = reg.installing || reg.waiting;
        if (!incoming) return resolve();
        const timer = setTimeout(finish, timeoutMs);
        function finish() {
            clearTimeout(timer);
            incoming.removeEventListener('statechange', check);
            resolve();
        }
        function check() {
            if (incoming.state === 'activated' || incoming.state === 'redundant') finish();
        }
        incoming.addEventListener('statechange', check);
        check();
    });
}

async function maybeReload() {
    if (reloadInFlight) return;
    if (!state.reloadPending) return;
    if (state.appMode === APP_MODES.ANALYZING) return;
    if (!state.acceptClicks) return;
    reloadInFlight = true;
    try {
        const reg = await navigator.serviceWorker?.getRegistration?.();
        if (reg) {
            await reg.update?.();
            // Wait for the updated worker to take over BEFORE reloading, so a
            // single reload picks up the new hashed bundles. Without the wait
            // the reload is served cache-first by the OLD worker (still the
            // old bundle), and the new worker's later takeover fires
            // index.html's controllerchange listener — a second visible
            // reload, typically on the next tab resume. The takeover itself
            // (clients.claim → controllerchange) may start the reload during
            // this wait; racing our own location.reload() below is harmless —
            // pending navigations coalesce into one. On timeout (slow install)
            // we reload anyway, degrading to the old two-step behavior instead
            // of hanging between problems.
            await waitForSwActivation(reg, 10000);
        }
    } catch (_) {}
    location.reload();
}

export function setDeferredInstallPrompt(prompt) {
    state.deferredInstallPrompt = prompt;
}

export function setUserHasInteracted(interacted) {
    state.userHasInteracted = interacted;
}

export function setInitialAnalysisNode(node) {
    state.initialAnalysisNode = node;
}

export function setGhostMove(move) {
    state.ghostMove = move;
}

export function setAnalysisHintMove(move) {
    state.analysisHintMove = move;
}

export function setCandidateMoves(moves) {
    state.candidateMoves = moves || [];
}

export function updateUserRank(rank) {
    state.userRank = rank;
}

export function updateServeRank(rank) {
    state.serveRank = rank;
}

export function updateDisplayRank(rank) {
    state.displayRank = rank;
}

export function updateRankSe(se) {
    state.rankSe = (typeof se === 'number' && Number.isFinite(se)) ? se : null;
}

// The persisted problem objects for `size`: the standing challenge and the
// last-completed problem, either possibly absent. Returned LIVE — the same
// objects state hands to the board — so a caller refreshes them by mutating in
// place and calling persistProblemObjects(). That the on-screen problem is the
// SAME reference is the point: a refresh landing mid-session reaches a
// challenge already painted, not just the copy in localStorage.
export function getPersistedProblemsForSize(size) {
    return [perSize.challengeProblem[size], perSize.previousProblem[size]]
        .filter(p => p && typeof p === 'object');
}

// Write both per-size buckets back after such an in-place refresh.
export function persistProblemObjects() {
    writeJSON('challengeProblem', perSize.challengeProblem);
    writeJSON('previousProblem', perSize.previousProblem);
}

export function setChallengeProblem(problem) {
    state.challengeProblem = problem;
}

// sessionStorage slot holding the on-screen problem (with its exact orientation)
// so a same-session reload can restore it verbatim. sessionStorage is the right
// scope: it survives location.reload() (the version-driven graceful reload, a
// missing-zip reload, a manual F5) but is cleared on a genuine reopen, so a fresh
// session still runs advanceToNextProblem and can surface a newly-due recall.
const RESUME_PROBLEM_KEY = 'resumeProblem';

export function setCurrentProblem(problem) {
    state.currentProblem = problem;
    // Persist the on-screen problem so a same-session reload restores the SAME
    // problem at the SAME orientation instead of re-picking/re-orienting it.
    // Recall problems get a random orientation (the pattern can't be recognized),
    // so without this a reload would re-randomize and visibly rotate the board.
    // Linked/shared detours (and a cleared slot) are intentionally not resumable
    // — clear the slot so a reload falls back to the normal flow.
    // Best-effort: sessionStorage can throw in private mode.
    try {
        if (problem && !problem.linked) {
            sessionStorage.setItem(RESUME_PROBLEM_KEY, JSON.stringify({ size: _boardSize, problem }));
        } else {
            sessionStorage.removeItem(RESUME_PROBLEM_KEY);
        }
    } catch (_) { /* sessionStorage unavailable — resume simply won't fire */ }
}

// Reads the resume slot written by setCurrentProblem. Returns the stored problem
// (with its exact orientation and already-transformed SGF) when it's resumable on
// THIS load — same board size, well-formed, non-linked — else null. The boot path
// uses it to restore the on-screen problem after a same-session reload without
// re-picking/re-orienting it; see problem-loader.resumeCurrentProblem.
export function getResumeProblem() {
    try {
        const raw = sessionStorage.getItem(RESUME_PROBLEM_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        const problem = parsed && parsed.problem;
        if (!problem || typeof problem.sgf !== 'string' || !problem.filename) return null;
        if (problem.linked) return null;
        if (parsed.size !== _boardSize) return null;
        return problem;
    } catch (_) {
        return null;
    }
}

export function setPreviousProblem(problem) {
    state.previousProblem = problem;
}

export function updateAnswerHistory(history) {
    state.answerHistory = history;
}

// Cross-size accessors for migrations / backfill that need to operate on a
// non-active size's data without mutating state.boardSize (which would clear
// the in-memory zip pool). Use sparingly.
export function getAnswerHistoryForSize(size) {
    // Entry references (filter preserves identity), so in-place mutation by the
    // refit lands in the unified array; persist via setAnswerHistoryForSize.
    return unifiedHistory.filter(a => a.size === size);
}

export function setAnswerHistoryForSize(size, history) {
    reconcileSize(size, history);
}

// LIVE references to the whole unified working set (both sizes, date order).
// For the sync layer only: the mirror migration needs to match every local
// entry (not a per-size filtered view) against the server's rows, and the
// id-first merge builds its dedup index from it. Mutations to the returned
// entry objects land in the real array; persist via the setters.
export function getWorkingSetEntries() {
    return [...unifiedHistory];
}

// Replace the ENTIRE unified working set in one step (mirror migration: the
// rebuilt post-split working set). Persists, but deliberately does NOT
// schedule an eviction — the caller has already split working set vs tail and
// written the tail itself.
export function replaceUnifiedHistory(entries) {
    unifiedHistory = [...entries].sort((x, y) => (x.date || 0) - (y.date || 0));
    invalidateActiveView();
    persistUnifiedHistory();
}

// Read-only WORKING-SET union (both sizes), each entry tagged with its `size`
// and sorted by date. The shared recall + weakness models train on this (they
// slice the last ≤MODEL_INPUT_HISTORY_LENGTH, so the working set suffices). The
// stats panel uses getMergedFullAnswerHistory() instead for whole-career charts.
// Entries are shallow copies, so callers must not expect mutations to persist —
// use the per-size setters for that.
export function getMergedAnswerHistory() {
    // The unified array IS the merged view (already tagged with size and date-
    // ordered); this is the WORKING SET only. Charts that need the full career
    // use getMergedFullAnswerHistory(). Shallow copies preserve the existing
    // no-mutation contract.
    return unifiedHistory.map(a => (a.size != null ? { ...a } : { ...a, size: 9 }));
}

export function setReviewStartTime(time) {
    state.reviewStartTime = time;
}

export function setCurrentSolvingStartTime(time) {
    state.currentSolvingStartTime = time;
}

export function addToAccumulatedSolvingTime(time) {
    state.accumulatedSolvingTime = state.accumulatedSolvingTime + time;
}

export function resetAccumulatedSolvingTime() {
    state.accumulatedSolvingTime = 0;
}

export function setStatsResults(results) {
    state.statsResults = results;
}

export function setModelParams(params) {
    state.modelParams = params;
}

// Switch to another supported size. Resets the in-memory problem pool so the
// new size's zips load fresh. Persisted per-size state is preserved.
export function setBoardSize(size) {
    if (!SUPPORTED_BOARD_SIZES.includes(size)) return;
    if (state.boardSize === size) return;
    state.boardSize = size;

    // Drop in-memory problem pool — it belonged to the previous size.
    state.zipMetadata = [];
    state.loadedZips = {};
    state.allProblems = [];
    zipCache.clear();
    embeddingsIndexCache.clear();

    // currentProblem is an in-memory display slot for the old size; the caller
    // reloads (advanceToNextProblem) and repopulates it. challengeProblem and
    // previousProblem are per-size and switch automatically.
    state.currentProblem = null;
}

export { SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE };
