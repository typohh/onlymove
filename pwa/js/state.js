// js/state.js
import { APP_MODES } from './constants.js';

// Generate or retrieve a persistent user ID
const getOrCreateUserId = () => {
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
    // App mode: 'solving', 'analyzing', or 'linked'
    appMode: APP_MODES.SOLVING,

    // Board and Game state
    board: null,
    game: null,
    kifuReader: null,

    // Tsumego state
    zipMetadata: [],
    loadedZips: {},
    allProblems: [],
    
    // User identification
    userId: getOrCreateUserId(),
    
    // UI state
    acceptClicks: false,
    deferredInstallPrompt: null,
    userHasInteracted: false,
    

    currentSolvingStartTime: null,
    accumulatedSolvingTime: 0,
    
    // Analysis mode state
    lastMoveInAnalysis: null, // Track last move in analysis mode
    initialAnalysisNode: null,
    ghostMove: null, // Pending move for double-click confirmation
    reviewStartTime: null,

    // User stats
    userRank: parseFloat(localStorage.getItem('userRank')) || 0,
    problemHistory: (JSON.parse(localStorage.getItem('problemHistory')) || []).map(item => ({ ...item, linked: item.linked || false, rating: item.rating || 0 })),
    nextTsumego: JSON.parse(localStorage.getItem('nextTsumego')) || null,
    momentum: parseFloat(localStorage.getItem('momentum')) || 30,

    // Linked problem ID
    linkedId: null,

    // More stats
    tsumegoStats: JSON.parse(localStorage.getItem('tsumegoStats')) || {
        failOnFail: 0,
        totalOnFail: 0,
        failOnInBetween: 0,
        totalOnInBetween: 0,
        failOnSuccess: 0,
        totalOnSuccess: 0,
    },

    answerHistory: JSON.parse(localStorage.getItem('answerHistory')) || [],
    statsResults: null,
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

export function updateAnswerHistory(history) {
    state.answerHistory = history;
    localStorage.setItem('answerHistory', JSON.stringify(history));
}





export function setReviewStartTime(time) {
    state.reviewStartTime = time;
}

export function setCurrentSolvingStartTime(time) {
    state.currentSolvingStartTime = time;
}

export function addToAccumulatedSolvingTime(time) {
    state.accumulatedSolvingTime += time;
}

export function resetAccumulatedSolvingTime() {
    state.accumulatedSolvingTime = 0;
}

export function setStatsResults(results) {
    state.statsResults = results;
}
