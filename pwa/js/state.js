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

function writePerSize(key, obj) {
    localStorage.setItem(key, JSON.stringify(obj));
}

const perSize = {
    userRank: readPerSize('userRank'),
    problemHistory: readPerSize('problemHistory'),
    answerHistory: readPerSize('answerHistory'),
    nextTsumego: readPerSize('nextTsumego'),
    momentum: readPerSize('momentum'),
    modelParams: readPerSize('modelParams'),
    adamState: readPerSize('adamState'),
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

    initialAnalysisNode: null,
    ghostMove: null,
    analysisHintMove: null,
    candidateMoves: [],
    reviewStartTime: null,

    linkedId: null,

    statsResults: null,
    learningRate: parseFloat(localStorage.getItem('learningRate')) || 5,

    // --- Per-size state (read/write proxies onto the active size's bucket). ---
    get boardSize() { return _boardSize; },
    set boardSize(size) {
        if (!SUPPORTED_BOARD_SIZES.includes(size)) return;
        if (_boardSize === size) return;
        _boardSize = size;
        localStorage.setItem('boardSize', String(size));
    },

    get userRank() {
        const v = perSize.userRank[_boardSize];
        return typeof v === 'number' ? v : DEFAULT_USER_RANK;
    },
    set userRank(v) {
        perSize.userRank[_boardSize] = v;
        writePerSize('userRank', perSize.userRank);
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
        const v = perSize.momentum[_boardSize];
        return typeof v === 'number' ? v : DEFAULT_MOMENTUM;
    },
    set momentum(v) {
        perSize.momentum[_boardSize] = v;
        writePerSize('momentum', perSize.momentum);
    },

    get modelParams() {
        return perSize.modelParams[_boardSize] ?? null;
    },
    set modelParams(v) {
        perSize.modelParams[_boardSize] = v;
        writePerSize('modelParams', perSize.modelParams);
    },

    get adamState() {
        return perSize.adamState[_boardSize] ?? null;
    },
    set adamState(v) {
        perSize.adamState[_boardSize] = v;
        writePerSize('adamState', perSize.adamState);
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

export function setAcceptClicks(canAccept) {
    state.acceptClicks = canAccept;
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
}

export { SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE };
