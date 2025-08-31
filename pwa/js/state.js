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
    // analysisKifuReader removed - using single kifuReader for both modes

    // Tsumego state
    zipMetadata: [],
    loadedZips: {},
    allProblems: [],
    currentProblemFilename: null,
    previousProblemFilename: localStorage.getItem('previousTsumego'),
    currentSGF: null,
    previousSGF: localStorage.getItem('previousTsumegoContent'),
    
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
    problemFrequencies: JSON.parse(localStorage.getItem('problemFrequencies')) || {},
    momentum: parseFloat(localStorage.getItem('momentum')) || 30,
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
