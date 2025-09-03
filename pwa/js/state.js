// js/state.js

// Generate or retrieve a persistent user ID
function getOrCreateUserId() {
    let userId = localStorage.getItem('userId');
    if (!userId) {
        // Generate a random user ID (8 characters)
        userId = Math.random().toString(36).substring(2, 10);
        localStorage.setItem('userId', userId);
    } else if (userId.length > 8) {
        // Truncate existing user ID to 8 characters
        userId = userId.substring(0, 8);
        localStorage.setItem('userId', userId);
    }
    return userId;
}

export const state = {
    // App mode: 'solving' or 'analyzing'
    appMode: 'solving',

    // Board and Game state
    board: null,
    game: null,
    kifuReader: null,

    // Tsumego state
    zipMetadata: [],
    loadedZips: {},
    allProblems: [],
    
    // Link state
    openedFromLink: false,
    
    // User identification
    userId: getOrCreateUserId(),
    
    // UI state
    acceptClicks: false,
    currentPointLoss: null,
    
    // Analysis mode state
    lastMoveInAnalysis: null, // Track last move in analysis mode

    // User stats
    userRank: parseFloat(localStorage.getItem('userRank')) || 0,
    problemHistory: (JSON.parse(localStorage.getItem('problemHistory')) || []).map(item => ({ ...item, linked: item.linked || false })),
    nextTsumego: JSON.parse(localStorage.getItem('nextTsumego')) || null,
    momentum: parseFloat(localStorage.getItem('momentum')) || 30,

    // More stats
    tsumegoStats: JSON.parse(localStorage.getItem('tsumegoStats')) || {
        failOnFail: 0,
        totalOnFail: 0,
        failOnInBetween: 0,
        totalOnInBetween: 0,
        failOnSuccess: 0,
        totalOnSuccess: 0,
    },
};

// MODIFICATION: Cache for loaded zip files
export const zipCache = new Map();

// Functions to modify state
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



export function updateUserRank(rank) {
    state.userRank = rank;
    localStorage.setItem('userRank', rank.toString());
}

export function updateMomentum(momentum) {
    state.momentum = momentum;
    localStorage.setItem('momentum', momentum.toString());
}

export function updateTsumegoStats(stats) {
    state.tsumegoStats = stats;
    localStorage.setItem('tsumegoStats', JSON.stringify(stats));
}

export function updateProblemHistory(history) {
    state.problemHistory = history;
    localStorage.setItem('problemHistory', JSON.stringify(history));
}

export function updateNextTsumego(nextTsumego) {
    state.nextTsumego = nextTsumego;
    localStorage.setItem('nextTsumego', JSON.stringify(nextTsumego));
}

export function setOpenedFromLink(value) {
    state.openedFromLink = value;
}

export function setCurrentPointLoss(value) {
    state.currentPointLoss = value;
}