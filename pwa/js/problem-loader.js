// js/problem-loader.js

import { state, zipCache, setAcceptClicks, setOpenedFromLink, updateProblemHistory } from './state.js';
import { loadSgf } from './board.js';
import { updateAllUI, displayOverlay } from './ui.js';
import { updateUrlWithTsumegoId } from './utils.js';

function selectProblem() {
    if (state.allProblems.length === 0) return null;

    // Check for re-selection from history
    for (let i = 0; i < state.problemHistory.length; i++) {
        const historyItem = state.problemHistory[i];
        let probability = 0;

        if (historyItem.status === 'failure') {
            if (i >= 20) {
                probability = Math.min(1, (i - 19) / 30);
            }
        } else if (historyItem.status === 'in-between') {
            if (i >= 150) {
                probability = Math.min(1, (i - 149) / 50);
            }
        }

        if (Math.random() < probability) {
            return historyItem;
        }
    }

    // Default selection logic
    const historyFilenames = new Set(state.problemHistory.map(p => p.filename));
    let availableProblems = state.allProblems.filter(p => !historyFilenames.has(p.filename));

    if (availableProblems.length === 0) {
        // If all problems are in history, just use all problems
        availableProblems = state.allProblems;
    }
    
    if (availableProblems.length === 0) availableProblems = state.allProblems;

    availableProblems.sort((a, b) => Math.abs(a.rank - state.userRank) - Math.abs(b.rank - state.userRank));
    const candidateProblems = availableProblems.slice(0, 500);

    const problem = candidateProblems[Math.floor(Math.random() * candidateProblems.length)];
    return { ...problem, status: 'success' };
}

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
        const response = await fetch(`metadata.json?v=${APP_VERSION}`);
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

export async function loadTsumego(isUpdate) {
    try {
        setAcceptClicks(false);

        if (!isUpdate && state.problemHistory.length > 0) {
            // On initial load, just load the last problem from history
            loadSgf(state.problemHistory[0].sgf);
            updateUrlWithTsumegoId(state.problemHistory[0].filename.replace('.sgf', ''));
            setAcceptClicks(true);
            return;
        }

        await loadZipMetadata();

        let selectedProblem = selectProblem();

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

        const sgf = selectedProblem.sgf || await fetchSgfFromZip(selectedProblem.filename, selectedProblem.zipFilename);

        const history = state.problemHistory;
        
        const existingIndex = history.findIndex(p => p.filename === selectedProblem.filename);
        if (existingIndex > -1) {
            history.splice(existingIndex, 1);
        }

        history.unshift({
            filename: selectedProblem.filename,
            sgf: sgf,
            status: selectedProblem.status
        });

        if (history.length > 200) {
            history.length = 200;
        }

        if (history.length > 2 && history[2].status === 'success') {
            history[2].sgf = null;
        }
        
        updateProblemHistory(history);
        setOpenedFromLink(false);
        loadSgf(sgf);
        updateUrlWithTsumegoId(selectedProblem.filename.replace('.sgf', ''));
        setAcceptClicks(true);
        backgroundLoadCloserZip();
        updateAllUI();

    } catch (error) {
        console.error(`Error loading tsumego:`, error);
        displayOverlay('Error loading problem.', 3000);
    }
}

export async function loadTsumegoById(tsumegoId) {
    try {
        if (state.problemHistory.length > 0 && state.problemHistory[0].filename === `${tsumegoId}.sgf`) {
            return false;
        }

        const zipId = await findZipIdForTsumegoId(tsumegoId);
        if (!zipId) {
            const message = `No tsumego with id ${tsumegoId}`;
            console.error(message);
            displayOverlay(message, 3000);
            return false;
        }

        await loadZipMetadata();

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

        await loadZipProblems(zipFilename);

        const sgfContent = await fetchSgfFromZip(`${tsumegoId}.sgf`, zipFilename);

        const history = state.problemHistory;
        history.unshift({
            filename: `${tsumegoId}.sgf`,
            sgf: sgfContent,
            status: 'success'
        });
        if (history.length > 200) {
            history.length = 200;
        }

        if (history.length > 2 && history[2].status === 'success') {
            history[2].sgf = null;
        }

        updateProblemHistory(history);
        
        setOpenedFromLink(true);

        loadSgf(sgfContent);
        updateUrlWithTsumegoId(tsumegoId);
        setAcceptClicks(true);
        updateAllUI();

        return true;
    } catch (error) {
        console.error(`Error loading tsumego by ID ${tsumegoId}:`, error);
        return false;
    }
}

export async function findZipIdForTsumegoId(tsumegoId) {
    const response = await fetch(`lookup.txt?v=${APP_VERSION}`);
    const text = await response.text();
    const lines = text.split('\n').filter(line => line.trim() !== '');

    let low = 0;
    let high = lines.length - 1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const [id, zipId] = lines[mid].trim().split(/\s+/);

        if (id === tsumegoId) {
            return zipId;
        } else if (id < tsumegoId) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return null;
}
