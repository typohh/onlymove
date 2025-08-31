// js/tsumego.js

import { state, zipCache } from './state.js';
import { updateAllUI, displayOverlay, fadeOverlay } from './ui.js';
import { initializeBoard, updateBoard, initializeKifuFromSGF } from './board.js';
import { playStoneSound } from './audio.js';
import { getTerm, updateUrlWithTsumegoId, getWinrateFromComment, getScoreFromComment } from './utils.js';
import { enterAnalysisMode } from './analysis.js';

// --- Tsumego Loading and Selection ---

function selectProblem() {
    if (state.allProblems.length === 0) return null;

    let availableProblems = state.allProblems.filter(p => p.filename !== state.currentProblemFilename && p.filename !== state.previousProblemFilename);
    if (availableProblems.length === 0) availableProblems = state.allProblems;

    availableProblems.sort((a, b) => Math.abs(a.rank - state.userRank) - Math.abs(b.rank - state.userRank));
    const candidateProblems = availableProblems.slice(0, 500);

    const weightedProblems = candidateProblems.map(problem => ({ problem, weight: state.problemFrequencies[problem.filename] || 1 }));
    const totalWeight = weightedProblems.reduce((sum, p) => sum + p.weight, 0);

    if (totalWeight <= 0) return candidateProblems[Math.floor(Math.random() * candidateProblems.length)];

    let random = Math.random() * totalWeight;
    for (const { problem, weight } of weightedProblems) {
        random -= weight;
        if (random <= 0) return problem;
    }
    return candidateProblems[Math.floor(Math.random() * candidateProblems.length)];
}


// --- User Stats ---

// Function to send report to Cloudflare Worker
async function sendReport(problemFilename, success) {
    try {
        // Check if this problem has already been attempted (exists in problemFrequencies)
        if (state.problemFrequencies.hasOwnProperty(problemFilename)) {
            return; // Already attempted, don't send report
        }
        
        // Extract problem ID (filename without extension)
        const problemId = problemFilename.replace(/\.[^/.]+$/, "");
        
        const reportData = {
            userId: state.userId,
            problem: problemId,
            success: success
        };
        
        // Use relative path to work with both local and production environments
        const response = await fetch('/report', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(reportData)
        });
        
        // We don't need to do anything special on success/failure since we're using
        // problemFrequencies as our tracking mechanism
    } catch (error) {
        console.error('Failed to send report:', error);
    }
}

function updateUserStats(isCorrect) {
    // If the tsumego was opened from a link, don't update stats
    if (state.openedFromLink) return;
    
    if (isCorrect) {
        const currentFreq = state.problemFrequencies[state.currentProblemFilename] || 1;
        state.problemFrequencies[state.currentProblemFilename] = currentFreq / 5;
        const maxIncrease = (1.0 - state.userRank) / 2;
        const intendedIncrease = state.momentum / 600;
        if(intendedIncrease > maxIncrease) {
            state.momentum = Math.max( 0.5 , state.momentum * 0.8 );
        } else {
            state.momentum *= 1.1;
        }
        state.userRank += Math.min(intendedIncrease, maxIncrease);
    } else {
        state.momentum = (state.momentum + 1) / 3;
        state.problemFrequencies[state.currentProblemFilename] = 25;
        state.userRank = Math.max(0, state.userRank - 4 * (state.momentum / 600));
    }

    localStorage.setItem('userRank', state.userRank.toString());
    localStorage.setItem('problemFrequencies', JSON.stringify(state.problemFrequencies));
    localStorage.setItem('momentum', state.momentum.toString());
    updateAllUI();
}

// --- Board Click and Answer Handling ---

function handleCorrectAnswer() {
    // Send success report
    sendReport(state.currentProblemFilename, true);
    updateUserStats(true);

    fadeOverlay(1000, () => {
        loadTsumego(true).catch(error => console.error("Error loading next tsumego:", error));
    });
}

function _calculatePointLossText(playedNode, currentKifuNode) {
    const children = currentKifuNode.children;
    if (!children || children.length === 0) return '';

    const hasWinrateComments = children.some(child => child.comment && getWinrateFromComment(child.comment) !== null);

    let correctNode;
    if (hasWinrateComments) {
        let bestWinrate = -1;
        for (const child of children) {
            if (!child.move) continue;
            const winrate = getWinrateFromComment(child.comment);
            if (winrate !== null && winrate > bestWinrate) {
                bestWinrate = winrate;
                correctNode = child;
            }
        }
    } else {
        correctNode = children.find(child => child.comment?.includes('Correct'));
    }

    if (!correctNode) return '';

    const correctScore = getScoreFromComment(correctNode.comment);
    const playedScore = getScoreFromComment(playedNode.comment);

    if (correctScore !== null && playedScore !== null) {
        const pointLoss = Math.abs(correctScore - playedScore);
        return `-${pointLoss.toFixed(1)} ${getTerm('points')}`;
    }

    // Fallback for old format "Score: ..."
    const correctScoreMatch = correctNode.comment?.match(/Score:\s*([+-]?\d+\.?\d*)/);
    const playedScoreMatch = playedNode.comment?.match(/Score:\s*([+-]?\d+\.?\d*)/);

    if (correctScoreMatch && playedScoreMatch) {
        const pointLoss = Math.abs(parseFloat(correctScoreMatch[1]) - parseFloat(playedScoreMatch[1]));
        return `-${pointLoss.toFixed(1)} ${getTerm('points')}`;
    }

    return '';
}


async function _handleIncorrectMoveSequence(moveIndex) {
    const delay = (ms) => new Promise(res => setTimeout(res, ms));

    // User move
    state.kifuReader.next(moveIndex);
    playStoneSound();
    updateBoard();
    await delay(1000);

    // Opponent response
    state.kifuReader.next();
    playStoneSound();
    updateBoard();

    // Send failure report
    sendReport(state.currentProblemFilename, false);
    updateUserStats(false);
    
    await new Promise((resolve) => displayOverlay('', 5000, resolve));

    // Load the next problem and then enter analysis mode for the failed one.
    await loadTsumego(true);
    enterAnalysisMode();
}


export async function handleBoardClick(x, y) {
    if (!state.acceptClicks || !state.kifuReader) return;

    const { node } = state.kifuReader;
    const children = node.children;

    if (!children || children.length === 0) {
        return;
    }

    // Find the move played by the user
    const moveNode = children.find(child => child.move && child.move.x === x && child.move.y === y);
    if (!moveNode) {
        return;
    }

    state.acceptClicks = false;
    const moveIndex = children.indexOf(moveNode);

    // Determine correctness
    const hasWinrateComments = children.some(child => child.comment && getWinrateFromComment(child.comment) !== null);
    let isCorrect = false;

    if (hasWinrateComments) {
        let bestWinrate = -1;
        let correctMove = null;
        for (const child of children) {
            if (!child.move) continue;
            const winrate = getWinrateFromComment(child.comment);
            if (winrate !== null && winrate > bestWinrate) {
                bestWinrate = winrate;
                correctMove = child.move;
            }
        }
        isCorrect = correctMove && correctMove.x === x && correctMove.y === y;
    } else {
        // Fallback to old logic
        isCorrect = moveNode.comment?.includes('Correct') || false;
    }

    if (isCorrect) {
        state.kifuReader.next(moveIndex);
        updateBoard();
        playStoneSound();
        handleCorrectAnswer();
    } else {
        state.currentPointLoss = _calculatePointLossText(moveNode, node);
        await _handleIncorrectMoveSequence(moveIndex);
    }
}


// --- Caching and Data Fetching ---

async function fetchSgfFromZip(filename, zipFilename) {
    if (!state.loadedZips[zipFilename]) await loadZipProblems(zipFilename);
    const zip = zipCache.get(zipFilename);
    if (!zip) throw new Error(`Zip file ${zipFilename} not found in cache.`);
    const file = zip.file(filename);
    if (!file) throw new Error(`File ${filename} not found in ${zipFilename}`);
    return await file.async('text');
}

export async function loadZipMetadata() {
    if (state.zipMetadata.length === 0) {
        const response = await fetch('metadata.json');
        state.zipMetadata = await response.json();
    }
}

export async function loadZipProblems(zipFilename) {
    if (state.loadedZips[zipFilename]) return;

    const cache = await caches.open('tsumego-cache-v1');
    const cacheKey = `tsumegos/${zipFilename}`;
    let zipData;
    const cachedResponse = await cache.match(cacheKey);

    if (cachedResponse) {
        zipData = await cachedResponse.arrayBuffer();
    } else {
        const zipResponse = await fetch(cacheKey);
        zipData = await zipResponse.arrayBuffer();
        await cache.put(cacheKey, new Response(zipData));
    }

    const zip = await new JSZip().loadAsync(zipData);
    const metadataFile = zip.file('metadata.json');
    if (!metadataFile) throw new Error(`metadata.json not found in ${zipFilename}`);

    const metadataContent = await metadataFile.async('text');
    const problems = JSON.parse(metadataContent).map(p => ({ ...p, zipFilename }));

    state.loadedZips[zipFilename] = problems;
    state.allProblems.push(...problems);
    zipCache.set(zipFilename, zip);
}

function findClosestZip() {
    return state.zipMetadata.reduce((closest, zip) => {
        const diff = Math.min(Math.abs(state.userRank - zip.min_rank), Math.abs(state.userRank - zip.max_rank));
        return diff < closest.minDiff ? { zip, minDiff: diff } : closest;
    }, { zip: null, minDiff: Infinity }).zip;
}

async function backgroundLoadCloserZip() {
    const closestZip = findClosestZip();
    if (closestZip && !state.loadedZips[closestZip.filename]) {
        try {
            await loadZipProblems(closestZip.filename);
        } catch (error) {
            console.error(`Failed to background load ${closestZip.filename}:`, error);
        }
    }
}

// --- Main Tsumego Loading Function ---

export async function loadTsumego(update) {
    try {
        state.acceptClicks = false;
        await loadZipMetadata();
        
        const savedFile = localStorage.getItem('currentTsumego');
        const savedContent = localStorage.getItem('currentTsumegoContent');
        let selectedProblem = null;

        if (savedFile && savedContent && !update) {
            state.currentProblemFilename = savedFile;
            state.currentSGF = savedContent;
            // Ensure previous values are set from localStorage
            state.previousProblemFilename = localStorage.getItem('previousTsumego');
            state.previousSGF = localStorage.getItem('previousTsumegoContent');
        } else {
            selectedProblem = selectProblem();
            if (!selectedProblem) {
                const closestZip = findClosestZip();
                if (closestZip) {
                    await loadZipProblems(closestZip.filename);
                    selectedProblem = selectProblem();
                    if (!selectedProblem) throw new Error("Failed to select a problem after loading zip.");
                } else {
                    throw new Error("No zip files found in metadata.");
                }
            }

            state.previousSGF = state.currentSGF;
            state.previousProblemFilename = state.currentProblemFilename;
            state.currentProblemFilename = selectedProblem.filename;
            state.currentSGF = await fetchSgfFromZip(state.currentProblemFilename, selectedProblem.zipFilename);

            localStorage.setItem('currentTsumego', state.currentProblemFilename);
            localStorage.setItem('currentTsumegoContent', state.currentSGF);
            if (state.previousProblemFilename) {
                localStorage.setItem('previousTsumego', state.previousProblemFilename);
                localStorage.setItem('previousTsumegoContent', state.previousSGF);
            }
            state.openedFromLink = false;
        }

        state.kifuReader = initializeKifuFromSGF(state.currentSGF);
        state.game = state.kifuReader.game;
        initializeBoard(state.kifuReader.kifu.size);
        updateBoard();
        updateAllUI();

        // Update URL to match current tsumego
        const tsumegoId = state.currentProblemFilename.replace('.sgf', '');
        updateUrlWithTsumegoId(tsumegoId);

        state.acceptClicks = true;
        backgroundLoadCloserZip();

    } catch (error) {
        console.error(`Error loading tsumego:`, error);
        displayOverlay('Error loading problem.', 3000);
    }
}


// Load a specific tsumego by ID
export async function loadTsumegoById(tsumegoId) {
    try {
        // Find the zip ID for this tsumego
        const zipId = await findZipIdForTsumegoId(tsumegoId);
        if (!zipId) {
            console.error(`Tsumego ID ${tsumegoId} not found in lookup`);
            return false;
        }

        // Load metadata to find which zip file contains this tsumego
        await loadZipMetadata();

        // Find the zip file that contains this tsumego
        let zipFilename = null;
        for (const zip of state.zipMetadata) {
            if (zip.filename.startsWith(zipId)) {
                zipFilename = zip.filename;
                break;
            }
        }

        if (!zipFilename) {
            console.error(`Zip file for tsumego ID ${tsumegoId} not found`);
            return false;
        }

        // Load the zip file
        await loadZipProblems(zipFilename);

        // Load the SGF content
        const sgfContent = await fetchSgfFromZip(`${tsumegoId}.sgf`, zipFilename);

        // Check if this is the same as the current tsumego
        const isSameAsCurrent = state.currentProblemFilename === `${tsumegoId}.sgf`;
        
        // Set the state directly
        state.previousSGF = state.currentSGF;
        state.previousProblemFilename = state.currentProblemFilename;
        state.currentProblemFilename = `${tsumegoId}.sgf`;
        state.currentSGF = sgfContent;

        // Set openedFromLink flag only if it's not the same as current
        if (!isSameAsCurrent) {
            state.openedFromLink = true;
        }

        // Save to localStorage
        localStorage.setItem('currentTsumego', state.currentProblemFilename);
        localStorage.setItem('currentTsumegoContent', state.currentSGF);
        if (state.previousProblemFilename) {
            localStorage.setItem('previousTsumego', state.previousProblemFilename);
            localStorage.setItem('previousTsumegoContent', state.previousSGF);
        }

        // Update URL to match current tsumego
        updateUrlWithTsumegoId(tsumegoId);

        return true;
    } catch (error) {
        console.error(`Error loading tsumego by ID ${tsumegoId}:`, error);
        return false;
    }
}

// Binary search lookup.txt for a tsumego ID
export async function findZipIdForTsumegoId(tsumegoId) {
    const response = await fetch('lookup.txt');
    const text = await response.text();
    const lines = text.split('\n').filter(line => line.trim() !== '');

    let low = 0;
    let high = lines.length - 1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const [id, zipId] = lines[mid].split(' ');

        if (id === tsumegoId) {
            return zipId;
        } else if (id < tsumegoId) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return null; // Not found
}