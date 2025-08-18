// js/state.js

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
    
    // UI state
    showCorrectMove: false,
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
