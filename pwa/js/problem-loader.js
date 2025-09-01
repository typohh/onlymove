// js/problem-loader.js

import { state, zipCache, setAcceptClicks, setCurrentProblem, setPreviousProblem, setOpenedFromLink } from './state.js';
import { loadSgf } from './board.js';
import { updateAllUI, displayOverlay } from './ui.js';
import { updateUrlWithTsumegoId } from './utils.js';

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

export async function loadTsumego(update) {
    try {
        setAcceptClicks(false);
        await loadZipMetadata();
        
        const savedFile = localStorage.getItem('currentTsumego');
        const savedContent = localStorage.getItem('currentTsumegoContent');

        if (savedFile && savedContent && !update) {
            setCurrentProblem(savedFile, savedContent);
            const prevFile = localStorage.getItem('previousTsumego');
            const prevSgf = localStorage.getItem('previousTsumegoContent');
            setPreviousProblem(prevFile, prevSgf);
        } else {
            const selectedProblem = selectProblem();
            if (!selectedProblem) {
                const closestZip = findClosestZip();
                if (closestZip) {
                    await loadZipProblems(closestZip.filename);
                    const newProblem = selectProblem();
                    if (!newProblem) throw new Error("Failed to select a problem after loading zip.");
                    setPreviousProblem(state.currentProblemFilename, state.currentSGF);
                    const sgf = await fetchSgfFromZip(newProblem.filename, newProblem.zipFilename);
                    setCurrentProblem(newProblem.filename, sgf);
                } else {
                    throw new Error("No zip files found in metadata.");
                }
            } else {
                setPreviousProblem(state.currentProblemFilename, state.currentSGF);
                const sgf = await fetchSgfFromZip(selectedProblem.filename, selectedProblem.zipFilename);
                setCurrentProblem(selectedProblem.filename, sgf);
            }
            setOpenedFromLink(false);
        }

        loadSgf(state.currentSGF);

        const tsumegoId = state.currentProblemFilename.replace('.sgf', '');
        updateUrlWithTsumegoId(tsumegoId);

        setAcceptClicks(true);
        backgroundLoadCloserZip();

    } catch (error) {
        console.error(`Error loading tsumego:`, error);
        displayOverlay('Error loading problem.', 3000);
    }
}

export async function loadTsumegoById(tsumegoId) {
    try {
        // On reload, the state is not fully initialized. We need to load the current problem from localStorage first.
        if (!state.currentSGF) {
            const savedFile = localStorage.getItem('currentTsumego');
            const savedContent = localStorage.getItem('currentTsumegoContent');
            if (savedFile && savedContent) {
                setCurrentProblem(savedFile, savedContent);
            }
        }

        // If the requested tsumego is already the current one, do nothing.
        if (state.currentProblemFilename === `${tsumegoId}.sgf`) {
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

        const isSameAsCurrent = state.currentProblemFilename === `${tsumegoId}.sgf`;
        
        setPreviousProblem(state.currentProblemFilename, state.currentSGF);
        setCurrentProblem(`${tsumegoId}.sgf`, sgfContent);

        if (!isSameAsCurrent) {
            setOpenedFromLink(true);
        }

        loadSgf(sgfContent);
        updateUrlWithTsumegoId(tsumegoId);
        setAcceptClicks(true);

        return true;
    } catch (error) {
        console.error(`Error loading tsumego by ID ${tsumegoId}:`, error);
        return false;
    }
}

export async function findZipIdForTsumegoId(tsumegoId) {
    const response = await fetch('lookup.txt');
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
