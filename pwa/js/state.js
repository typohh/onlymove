// js/state.js
import { APP_MODES } from './constants.js';

const SUPPORTED_BOARD_SIZES = [9, 13];
const DEFAULT_BOARD_SIZE = 9;

// --- localStorage migration ---
// Old layout stored a single value per key (e.g. userRank: "0.5",
// problemHistory: "[...]") because the app only supported 9x9.
// New layout stores an object keyed by board size: {"9": <oldValue>}.
// This must run before any state property below is read.
(function migrateBoardSizeBuckets() {
    if (localStorage.getItem('boardSizeMigrated_v1')) return;

    const perSizeKeys = [
        'userRank',
        'problemHistory',
        'answerHistory',
        'nextTsumego',
        'momentum',
        'modelParams',
        'adamState',
        'weaknessParams',
        'weaknessAdam',
        'weaknessSe',
        'weaknessAxisCov',
    ];

    for (const key of perSizeKeys) {
        const raw = localStorage.getItem(key);
        if (raw === null) continue;

        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch {
            const num = parseFloat(raw);
            parsed = Number.isNaN(num) ? raw : num;
        }

        // Already in the new shape — leave alone.
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed) &&
            ('9' in parsed || '13' in parsed)) {
            continue;
        }

        localStorage.setItem(key, JSON.stringify({ [DEFAULT_BOARD_SIZE]: parsed }));
    }

    // Existing users (those who had any state at all) keep playing 9x9.
    // Truly fresh users have no boardSize and will see the cold-start dialog.
    const hadAnyState = perSizeKeys.some(k => localStorage.getItem(k) !== null);
    if (hadAnyState && !localStorage.getItem('boardSize')) {
        localStorage.setItem('boardSize', String(DEFAULT_BOARD_SIZE));
    }

    localStorage.setItem('boardSizeMigrated_v1', 'true');
})();

// --- Cross-size merge: rank, momentum, and the learned models are now shared ---
// The two board sizes share one rank (displayed everywhere, used to pick
// problems) and one each of the recall + weakness models (trained on the union
// of both histories — every per-problem feature they read is board-independent).
// This migration collapses the per-size buckets v1 created back into single
// values: rank = max of the two (a strong 9x9 player should immediately see
// hard 13x13 problems), momentum follows the higher-ranked size, and the model
// buckets are dropped so the shared models retrain from scratch on next load.
// answerHistory / problemHistory / nextTsumego stay per-size.
(function mergeSharedAcrossSizes_v2() {
    if (localStorage.getItem('sharedAcrossSizesMigrated_v2')) return;

    const readBucket = (key) => {
        const raw = localStorage.getItem(key);
        if (raw === null) return {};
        try {
            const parsed = JSON.parse(raw);
            return (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) ? parsed : {};
        } catch { return {}; }
    };

    const rankBucket = readBucket('userRank');
    let maxRank = 0, maxRankSize = null;
    for (const [sz, v] of Object.entries(rankBucket)) {
        if (typeof v === 'number' && v > maxRank) { maxRank = v; maxRankSize = sz; }
    }
    localStorage.setItem('userRank', JSON.stringify(maxRank));

    const momentumBucket = readBucket('momentum');
    const momentum = (maxRankSize !== null && typeof momentumBucket[maxRankSize] === 'number')
        ? momentumBucket[maxRankSize]
        : 30;
    localStorage.setItem('momentum', JSON.stringify(momentum));

    // Models retrain from scratch — drop the per-size buckets entirely.
    for (const key of ['modelParams', 'adamState', 'weaknessParams',
                       'weaknessAdam', 'weaknessSe', 'weaknessAxisCov']) {
        localStorage.removeItem(key);
    }

    localStorage.setItem('sharedAcrossSizesMigrated_v2', 'true');
})();

// --- Board size + per-size backing buckets ---

function readPerSize(key, fallback) {
    const raw = localStorage.getItem(key);
    if (raw === null) return {};
    try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return parsed;
        }
    } catch {}
    return {};
}

// Number of most-recent entries that should keep their `embedding` field.
// Older entries get their embedding stripped on quota pressure (the entry
// itself stays — preserving stats like novel-problem count and the recall
// model's history). Aligned with MODEL_INPUT_HISTORY_LENGTH which is what
// the trainers actually use.
const ANSWER_HISTORY_EMBEDDING_KEEP_LAST = 500;

function writePerSize(key, obj) {
    try {
        localStorage.setItem(key, JSON.stringify(obj));
        return;
    } catch (e) {
        const isQuota = e && (
            e.name === 'QuotaExceededError' ||
            e.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
            e.code === 22 || e.code === 1014
        );
        if (!isQuota) throw e;
    }
    if (key !== 'answerHistory' || !obj || typeof obj !== 'object') {
        console.error(`localStorage quota exceeded for ${key}; write skipped`);
        return;
    }
    // Strip the heavy `embedding` field from entries beyond the keep-last
    // window. Preserve the entry shape otherwise so stats and the recall
    // model still see the full history.
    let stripped = 0;
    for (const sz of Object.keys(obj)) {
        const arr = obj[sz];
        if (!Array.isArray(arr)) continue;
        const cutoff = Math.max(0, arr.length - ANSWER_HISTORY_EMBEDDING_KEEP_LAST);
        for (let i = 0; i < cutoff; i++) {
            const entry = arr[i];
            if (entry && entry.embedding !== undefined) {
                delete entry.embedding;
                stripped++;
            }
        }
    }
    try {
        localStorage.setItem(key, JSON.stringify(obj));
        if (stripped > 0) {
            console.warn(`localStorage quota relieved by stripping embedding from ${stripped} old answerHistory entries`);
        }
    } catch (e2) {
        console.error(`localStorage still over quota after stripping ${stripped} embeddings; write skipped to keep play flow alive`);
    }
}

const perSize = {
    problemHistory: readPerSize('problemHistory'),
    answerHistory: readPerSize('answerHistory'),
    nextTsumego: readPerSize('nextTsumego'),
};

// --- Shared-across-sizes state (single values, not bucketed). ---
// rank + momentum are one control loop over a single shared rank; the recall
// (modelParams/adamState) and weakness (weaknessParams/Adam/Se/AxisCov) models
// train on the union of both sizes' histories. See mergeSharedAcrossSizes_v2.
function readShared(key, fallback) {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    try {
        const parsed = JSON.parse(raw);
        return parsed === null ? fallback : parsed;
    } catch { return fallback; }
}

function writeShared(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        console.error(`localStorage write failed for shared key ${key}`, e);
    }
}

const shared = {
    userRank: readShared('userRank', null),
    momentum: readShared('momentum', null),
    modelParams: readShared('modelParams', null),
    adamState: readShared('adamState', null),
    weaknessParams: readShared('weaknessParams', null),
    weaknessAdam: readShared('weaknessAdam', null),
    weaknessSe: readShared('weaknessSe', null),
    // Per-axis cov(β_correct[k], β_picked[k]) for the augmented model.
    // Flat array of K floats. Combined with weaknessSe (for the diagonals)
    // and weaknessParams, this is the minimum state needed to compute
    // sign-quadrant probabilities per axis without retraining.
    weaknessAxisCov: readShared('weaknessAxisCov', null),
};

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

const getOrCreateUserId = () => {
    let userId = localStorage.getItem('userId');
    if (!userId) {
        userId = Math.random().toString(36).substring(2, 10);
        localStorage.setItem('userId', userId);
    } else if (userId.length > 8) {
        userId = userId.substring(0, 8);
        localStorage.setItem('userId', userId);
    }
    return userId;
};

// Defaults applied when a size's bucket is empty.
const DEFAULT_USER_RANK = 0;
const DEFAULT_MOMENTUM = 30;

export const state = {
    appMode: APP_MODES.SOLVING,

    // Board and Game state
    board: null,
    game: null,
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

    linkedId: null,

    statsResults: null,
    learningRate: parseFloat(localStorage.getItem('learningRate')) || 5,

    // In-memory cache of labels.components (joint across board sizes —
    // the PCA + ICA fit is joint, so labels are shared). Populated by
    // embedding-backfill.js after fetching any size's embedding_meta.json.
    // Used by the stats panel to name top weakness PCs. Not persisted.
    embeddingLabels: null,

    // Joint embedding meta consumed by the per-cell sidecar reader (K,
    // bytes_per_cell, dtype). Populated by embedding-backfill.js from the
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
        return typeof shared.userRank === 'number' ? shared.userRank : DEFAULT_USER_RANK;
    },
    set userRank(v) {
        shared.userRank = v;
        writeShared('userRank', v);
    },

    get problemHistory() {
        if (!Array.isArray(perSize.problemHistory[_boardSize])) {
            perSize.problemHistory[_boardSize] = [];
        }
        return perSize.problemHistory[_boardSize];
    },
    set problemHistory(v) {
        perSize.problemHistory[_boardSize] = v;
        writePerSize('problemHistory', perSize.problemHistory);
    },

    get answerHistory() {
        if (!Array.isArray(perSize.answerHistory[_boardSize])) {
            perSize.answerHistory[_boardSize] = [];
        }
        return perSize.answerHistory[_boardSize];
    },
    set answerHistory(v) {
        perSize.answerHistory[_boardSize] = v;
        writePerSize('answerHistory', perSize.answerHistory);
    },

    get nextTsumego() {
        return perSize.nextTsumego[_boardSize] ?? null;
    },
    set nextTsumego(v) {
        perSize.nextTsumego[_boardSize] = v;
        writePerSize('nextTsumego', perSize.nextTsumego);
    },

    get momentum() {
        return typeof shared.momentum === 'number' ? shared.momentum : DEFAULT_MOMENTUM;
    },
    set momentum(v) {
        shared.momentum = v;
        writeShared('momentum', v);
    },

    get modelParams() {
        return shared.modelParams ?? null;
    },
    set modelParams(v) {
        shared.modelParams = v;
        writeShared('modelParams', v);
    },

    get adamState() {
        return shared.adamState ?? null;
    },
    set adamState(v) {
        shared.adamState = v;
        writeShared('adamState', v);
    },

    get weaknessParams() {
        return shared.weaknessParams ?? null;
    },
    set weaknessParams(v) {
        shared.weaknessParams = v;
        writeShared('weaknessParams', v);
    },

    get weaknessAdam() {
        return shared.weaknessAdam ?? null;
    },
    set weaknessAdam(v) {
        shared.weaknessAdam = v;
        writeShared('weaknessAdam', v);
    },

    get weaknessSe() {
        return shared.weaknessSe ?? null;
    },
    set weaknessSe(v) {
        shared.weaknessSe = v;
        writeShared('weaknessSe', v);
    },

    get weaknessAxisCov() {
        return shared.weaknessAxisCov ?? null;
    },
    set weaknessAxisCov(v) {
        shared.weaknessAxisCov = v;
        writeShared('weaknessAxisCov', v);
    },

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
    state.game = kifuReader ? kifuReader.game : null;
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
async function maybeReload() {
    if (reloadInFlight) return;
    if (!state.reloadPending) return;
    if (state.appMode === APP_MODES.ANALYZING) return;
    if (!state.acceptClicks) return;
    reloadInFlight = true;
    try {
        const reg = await navigator.serviceWorker?.getRegistration?.();
        await reg?.update?.();
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

export function updateMomentum(momentum) {
    state.momentum = momentum;
}

export function updateProblemHistory(history) {
    state.problemHistory = history;
}

export function updateNextTsumego(nextTsumego) {
    state.nextTsumego = nextTsumego;
}

export function updateAnswerHistory(history) {
    state.answerHistory = history;
}

// Cross-size accessors for migrations / backfill that need to operate on a
// non-active size's data without mutating state.boardSize (which would clear
// the in-memory zip pool). Use sparingly.
export function getAnswerHistoryForSize(size) {
    return Array.isArray(perSize.answerHistory[size]) ? perSize.answerHistory[size] : [];
}

export function setAnswerHistoryForSize(size, history) {
    perSize.answerHistory[size] = history;
    writePerSize('answerHistory', perSize.answerHistory);
}

// Read-only union of every size's answerHistory, each entry tagged with its
// `size` and sorted by date. The shared recall + weakness models train on this,
// and the stats panel renders from it. Entries are shallow copies, so callers
// must not expect mutations to persist — use the per-size setters for that.
export function getMergedAnswerHistory() {
    const merged = [];
    for (const size of SUPPORTED_BOARD_SIZES) {
        const arr = perSize.answerHistory[size];
        if (!Array.isArray(arr)) continue;
        for (const a of arr) merged.push({ ...a, size });
    }
    merged.sort((x, y) => (x.date || 0) - (y.date || 0));
    return merged;
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

export function setAdamState(adamState) {
    state.adamState = adamState;
}

export function setLearningRate(rate) {
    state.learningRate = rate;
    localStorage.setItem('learningRate', rate.toString());
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
}

export { SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE };
