// js/problem-loader.js

import { state, zipCache, embeddingsIndexCache, setAcceptClicks, updateProblemHistory, updateNextTsumego, setAppMode, setCurrentSolvingStartTime, resetAccumulatedSolvingTime, setBoardSize } from './state.js';
import { loadSgf } from './board.js';
import { updateAllUI } from './ui.js';
import { displayOverlay } from './overlays.js';
import { reorientSgf } from './sgf.js';
import { MODEL_INPUT_HISTORY_LENGTH, APP_MODES } from './constants.js';
import { getRecallCandidates } from './model.js';
import { weaknessScore, WEAKNESS_TEMPERATURE } from './weakness.js';

async function simpleHash(str) {
    const encoder = new TextEncoder();
    const data = encoder.encode(str);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

function weightedRandomSelect(items, weightFn) {
    const totalWeight = items.reduce((sum, item) => sum + Math.max(0, weightFn(item)), 0);
    if (totalWeight === 0) return items[0]; // fallback to first if no weights
    let random = Math.random() * totalWeight;
    for (const item of items) {
        random -= Math.max(0, weightFn(item));
        if (random <= 0) return item;
    }
    return items[0]; // fallback
}

async function selectProblem() {
    if (state.allProblems.length === 0) return null;

    // Check for nextTsumego first
    if (state.nextTsumego) {
        const problem = state.nextTsumego;
        updateNextTsumego(null);
        return problem;
    }

    const recallCandidates = getRecallCandidates();

    if (recallCandidates.length > 0) {
        recallCandidates.sort((a, b) => a.recallRate - b.recallRate);
        const chosenProblemInfo = recallCandidates[0];
        let problem = state.allProblems.find(p => p.filename === `${chosenProblemInfo.problemId}.sgf`);

        if (problem) {
            return { ...problem, isRepeat: true };
        }

        // If problem is not in memory, load its zip file
        console.warn(`Problem ${chosenProblemInfo.problemId} not found in allProblems. Attempting to load it.`);

        const lookupResult = await findZipIdForTsumegoId(chosenProblemInfo.problemId);
        // Recall problems must belong to the active size (answerHistory is per-size).
        // Skip cross-size matches defensively rather than load the wrong pool.
        if (lookupResult && (lookupResult.size === null || lookupResult.size === state.boardSize)) {
            await loadZipMetadata(); // Ensure metadata is loaded
            const zipInfo = state.zipMetadata.find(zip => zip.filename.startsWith(lookupResult.zipId));
            if (zipInfo) {
                await loadZipProblems(zipInfo.filename);
                // Try to find the problem again
                problem = state.allProblems.find(p => p.filename === `${chosenProblemInfo.problemId}.sgf`);
                if (problem) {
                    return { ...problem, isRepeat: true };
                }
            }
        }
        
        // If we still can't find it, log an error and mark it unavailable for this session.
        // Next session we'll try again — files may come back, or the user may update the app.
        console.error(`Failed to load recall problem ${chosenProblemInfo.problemId}.`);
        state.unavailableProblemIds.add(chosenProblemInfo.problemId);
        return await selectProblem();
    }

    // Default selection logic (fallback)
    const recentAnswers = state.answerHistory.slice(-MODEL_INPUT_HISTORY_LENGTH);
    const historyFilenames = new Set(recentAnswers.map(a => `${a.problemId}.sgf`));
    let availableProblems = state.allProblems.filter(p => !historyFilenames.has(p.filename));

    if (availableProblems.length === 0) {
        return null;
    }

    availableProblems.sort((a, b) => Math.abs(a.rating - state.userRank) - Math.abs(b.rating - state.userRank));
    const candidateProblems = availableProblems.slice(0, 50);

    // Check if we need to start background loading
    if (candidateProblems.length < 50) {
        backgroundLoadCloserZip();
    } else {
        const closestZip = findClosestUnloadedZip();
        if (closestZip) {
            const userRank = state.userRank;
            const candidateDiff = Math.abs(candidateProblems[49].rating - userRank);
            const zipDiff = Math.min(Math.abs(userRank - closestZip.min_rating), Math.abs(userRank - closestZip.max_rating));
            if (zipDiff < candidateDiff) {
                backgroundLoadCloserZip();
            }
        }
    }

    // Weight by quality, modulated by alignment with the user's weakness vector.
    // weaknessScore returns 0 until the per-component model has trained, so
    // the cold-start path here matches the prior pure-quality selection.
    const problem = weightedRandomSelect(candidateProblems, p => {
        const q = Math.max(0, p.quality || 0);
        const w = weaknessScore(p);
        return q * Math.exp(WEAKNESS_TEMPERATURE * w);
    });
    return { ...problem, isRepeat: false };
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
        const response = await fetch(`tsumegos/${state.boardSize}/metadata.json?v=${APP_VERSION}`);
        state.zipMetadata = await response.json();
        const metadataStr = JSON.stringify(state.zipMetadata);
        console.log('Metadata hash:', await simpleHash(metadataStr));
    }
}

export async function loadZipProblems(zipFilename) {
    if (state.loadedZips[zipFilename]) return;

    const cacheKey = `tsumegos/${state.boardSize}/${zipFilename}`;
        const zipResponse = await fetch(cacheKey);

    const contentType = zipResponse.headers.get('content-type');
    if (!zipResponse.ok || (contentType && !contentType.includes('application/zip') && !contentType.includes('application/octet-stream'))) {
        console.error(`Invalid response for ${zipFilename}: status ${zipResponse.status}, type ${contentType}`);
        // Check if likely missing file: 404 or wrong content type (indirect 404 detection)
        const isLikelyMissing = zipResponse.status === 404 || (contentType && !contentType.includes('application/zip') && !contentType.includes('application/octet-stream'));
        if (isLikelyMissing) {
            const lastReloadTime = localStorage.getItem('lastReloadTime');
            const now = Date.now();
            const reloadThreshold = 30000; // 30 seconds
            const hasRecentReload = lastReloadTime && (now - parseInt(lastReloadTime)) < reloadThreshold;
            if (state.userHasInteracted && !hasRecentReload) {
                Sentry.captureMessage('Reloading due to missing zip file', {
                    level: 'warning',
                    extra: { zipFilename }
                });
                console.log('Detected missing zip file, likely due to new version. Reloading page.');
                localStorage.setItem('lastReloadTime', now.toString());
                location.reload();
                return; // Prevent further execution
            }
        }
        throw new Error(`Invalid response for ${zipFilename}: status ${zipResponse.status}, type ${contentType}`);
    }

    const zipData = await zipResponse.arrayBuffer();

    let zip;
    try {
        zip = await new JSZip().loadAsync(zipData);
    } catch (error) {
        if (error.message && error.message.includes("Can't find end of central directory")) {
            console.log(`Zip file error: filename=${zipFilename}, filesize=${zipData.byteLength}`);
        }
        throw error;
    }
    const metadataFile = zip.file('metadata.json');
    if (!metadataFile) throw new Error(`metadata.json not found in ${zipFilename}`);

    const metadataContent = await metadataFile.async('text');
    const problems = JSON.parse(metadataContent).map(p => ({ ...p, zipFilename }));

    state.loadedZips[zipFilename] = problems;
    state.allProblems.push(...problems);
    zipCache.set(zipFilename, zip);

    // Pull the per-cell embedding offset table out of the zip. Older zips
    // (pre-sidecar) won't have it — that's fine; picked-embedding lookups
    // will see no entry and return null, which is what we want.
    const indexFile = zip.file('embeddings_index.json');
    if (indexFile) {
        try {
            const payload = JSON.parse(await indexFile.async('text'));
            if (payload && payload.problems) {
                embeddingsIndexCache.set(zipFilename, payload);
            }
        } catch (e) {
            console.warn(`embeddings_index.json parse failed for ${zipFilename}:`, e);
        }
    }
}

function findClosestUnloadedZip() {
    const unloadedZips = state.zipMetadata.filter(zip => !state.loadedZips[zip.filename]);
    return unloadedZips.reduce((closest, zip) => {
        const diff = Math.min(Math.abs(state.userRank - zip.min_rating), Math.abs(state.userRank - zip.max_rating));
        return diff < closest.minDiff ? { zip, minDiff: diff } : closest;
    }, { zip: null, minDiff: Infinity }).zip;
}

async function backgroundLoadCloserZip() {
    const closestZip = findClosestUnloadedZip();
    if (closestZip) {
        try {
            await loadZipProblems(closestZip.filename);
        } catch (error) {
            console.error(`Failed to background load ${closestZip.filename}:`, error);
        }
    }
}

function loadNextZip() {
    const nextZip = findClosestUnloadedZip();
    if (nextZip) {
        loadZipProblems(nextZip.filename);
    }
}

// One pass through select → fetch SGF → install. Throws on any failure so the
// outer retry loop can mark the offending candidate unavailable and try again.
async function loadOneTsumego() {
    let selectedProblem = null;
    let rawSgf = null;
    let reorientedSgf = null;
    let chosenOrientation = null;
    try {
        await loadZipMetadata();

        selectedProblem = await selectProblem();
        if (!selectedProblem) {
            const closestZip = findClosestUnloadedZip();
            if (!closestZip) throw new Error("No zip files found in metadata.");
            await loadZipProblems(closestZip.filename);
            selectedProblem = await selectProblem();
            if (!selectedProblem) throw new Error("Failed to select a problem after loading zip.");
        }

        let sgfContent = selectedProblem.sgf || await fetchSgfFromZip(selectedProblem.filename, selectedProblem.zipFilename);
        if (!sgfContent || !sgfContent.trim()) {
            throw new Error('Invalid or empty SGF');
        }
        // Stash for the catch-block diagnostics — when the kifu fails to play
        // back, the actual SGF (with reorientation) is the only useful clue.
        rawSgf = sgfContent;

        const isRepeat = selectedProblem.isRepeat;
        const orientationParam = isRepeat ? null : 0;
        const reoriented = reorientSgf(sgfContent, orientationParam);
        const sgf = reoriented.sgf;
        const orientation = reoriented.orientation;
        reorientedSgf = sgf;
        chosenOrientation = orientation;

        const history = state.problemHistory;
        const filename = selectedProblem.filename;
        const indicesToRemove = [];
        for (let i = 0; i < history.length; i++) {
            if (history[i].filename === filename) indicesToRemove.push(i);
        }
        for (let i = indicesToRemove.length - 1; i >= 0; i--) {
            history.splice(indicesToRemove[i], 1);
        }

        history.unshift({
            filename: selectedProblem.filename,
            sgf: sgf,
            linked: false,
            rating: selectedProblem.rating,
            quality: selectedProblem.quality || 0,
            embedding: selectedProblem.embedding,
            isRepeat: selectedProblem.isRepeat || false,
            orientation: orientation,
            // Retained so picked-embedding can locate the sidecar .embeddings.bin.
            zipFilename: selectedProblem.zipFilename
        });

        if (history.length > 2) history.length = 2;
        updateProblemHistory(history);

        try {
            loadSgf(sgf);
        } catch (error) {
            throw new Error('Invalid SGF: ' + error.message);
        }

        setCurrentSolvingStartTime(Date.now());
        resetAccumulatedSolvingTime();
        updateAllUI();
    } catch (error) {
        // Tag the offending candidate so the retry loop can blacklist it,
        // and attach the SGF (raw + reoriented) so the debug log shows what
        // we actually fed into the kifu reader when it choked.
        if (selectedProblem && selectedProblem.filename) {
            error.failedProblemId = selectedProblem.filename.replace(/\.sgf$/, '');
        }
        if (rawSgf !== null) error.rawSgf = rawSgf;
        if (reorientedSgf !== null) error.reorientedSgf = reorientedSgf;
        if (chosenOrientation !== null) error.chosenOrientation = chosenOrientation;
        throw error;
    }
}

const MAX_LOAD_ATTEMPTS = 5;
const RELOAD_COOLDOWN_MS = 30000;

export async function loadTsumego(isUpdate) {
    setAcceptClicks(false);

    if (!isUpdate && state.problemHistory.length > 0) {
        // On initial load, just load the last problem from history.
        try {
            loadSgf(state.problemHistory[0].sgf);
            setCurrentSolvingStartTime(Date.now());
            resetAccumulatedSolvingTime();
            setAcceptClicks(true);
            return;
        } catch (error) {
            console.error('Error loading cached tsumego, falling through to new selection:', error);
            try { Sentry.captureException(error); } catch (_) {}
            // Fall through to a fresh selection.
        }
    }

    let lastError = null;
    for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt++) {
        try {
            await loadOneTsumego();
            setAcceptClicks(true);
            // loadOneTsumego already called updateAllUI(), but acceptClicks was
            // still false then — repaint so button1's disabled state matches.
            updateAllUI();
            return;
        } catch (error) {
            lastError = error;
            console.error(`Tsumego load attempt ${attempt + 1}/${MAX_LOAD_ATTEMPTS} failed:`, error);
            if (error.rawSgf) {
                console.error(`  raw SGF: ${error.rawSgf}`);
            }
            if (error.reorientedSgf) {
                console.error(`  reoriented SGF (orientation=${error.chosenOrientation}): ${error.reorientedSgf}`);
            }
            try {
                Sentry.captureException(error, {
                    extra: {
                        attempt: attempt + 1,
                        failedProblemId: error.failedProblemId,
                        boardSize: state.boardSize,
                        rawSgf: error.rawSgf,
                        reorientedSgf: error.reorientedSgf,
                        chosenOrientation: error.chosenOrientation,
                    },
                });
            } catch (_) {}
            // Blacklist the failed candidate for this session so selectProblem
            // picks a different one on the next iteration.
            if (error.failedProblemId) {
                state.unavailableProblemIds.add(error.failedProblemId);
            }
        }
    }

    // All retries exhausted. Restore clicks so the user is never frozen, then
    // try one cooled-down reload — fresh metadata may unblock us. If we already
    // reloaded recently, give up gracefully with a visible error.
    setAcceptClicks(true);
    console.error(`Tsumego load failed after ${MAX_LOAD_ATTEMPTS} attempts; last error:`, lastError);
    const lastReload = parseInt(localStorage.getItem('lastReloadTime') || '0', 10);
    const recentlyReloaded = Date.now() - lastReload < RELOAD_COOLDOWN_MS;
    if (state.userHasInteracted && !recentlyReloaded) {
        try {
            Sentry.captureMessage('Reloading after repeated tsumego load failures', { level: 'warning' });
        } catch (_) {}
        localStorage.setItem('lastReloadTime', String(Date.now()));
        displayOverlay('Error loading problem. Reloading…', 1500);
        setTimeout(() => location.reload(), 1500);
    } else {
        displayOverlay('Error loading problem. Try refreshing the page.', 5000);
    }
}

export async function loadTsumegoById(tsumegoId, orientation = null, { asRecall = false } = {}) {
    try {
        if (state.problemHistory.length > 0 && state.problemHistory[0].filename === `${tsumegoId}.sgf` &&
           (orientation === null || state.problemHistory[0].orientation === orientation)) {
            return false;
        }

        const lookupResult = await findZipIdForTsumegoId(tsumegoId);
        if (!lookupResult) {
            const message = `No tsumego with id ${tsumegoId}`;
            console.error(message);
            displayOverlay(message, 3000);
            return false;
        }

        const { zipId, size } = lookupResult;

        // The lookup column is authoritative — switch sizes if the linked
        // problem belongs to a different board than the active one.
        if (size && size !== state.boardSize) {
            setBoardSize(size);
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

        const problem = state.allProblems.find(p => p.filename === `${tsumegoId}.sgf`);
        const problemRank = problem ? problem.rating : 0;
        const problemQuality = problem ? problem.quality : 0;
        const problemEmbedding = problem ? problem.embedding : undefined;

        let sgfContent = await fetchSgfFromZip(`${tsumegoId}.sgf`, zipFilename);
        
        const reoriented = reorientSgf(sgfContent, orientation);
        sgfContent = reoriented.sgf;
        const actualOrientation = reoriented.orientation;

        const history = state.problemHistory;

        const newProblem = {
            filename: `${tsumegoId}.sgf`,
            sgf: sgfContent,
            linked: !asRecall,
            rating: problemRank,
            quality: problemQuality,
            embedding: problemEmbedding,
            isRepeat: asRecall,
            orientation: actualOrientation,
            // Retained so picked-embedding can locate the sidecar .embeddings.bin.
            zipFilename: zipFilename
        };

        // If current problem is linked, replace it instead of shifting
        if (history.length > 0 && history[0].linked) {
            history[0] = newProblem;
        } else {
            // Move current to nextTsumego if not already set
            if (!state.nextTsumego && history.length > 0) {
                updateNextTsumego({ ...history[0], linked: false });
            }
            history.unshift(newProblem);
        }
        if (history.length > 2) {
            history.length = 2;
        }

        updateProblemHistory(history);

        setAppMode(asRecall ? APP_MODES.SOLVING : APP_MODES.LINKED);

        loadSgf(sgfContent);
        
        setAcceptClicks(true);
        setCurrentSolvingStartTime(Date.now());
        resetAccumulatedSolvingTime();
        updateAllUI();

        return true;
    } catch (error) {
        console.error(`Error loading tsumego by ID ${tsumegoId}:`, error);
        return false;
    }
}

// Resolves a tsumego id to its zip hash and board size by binary-searching
// the global lookup.txt. Each row is "id zipHash size". Returns null when
// the id is unknown, otherwise { zipId, size }.
export async function findZipIdForTsumegoId(tsumegoId) {
    const response = await fetch(`lookup.txt?v=${APP_VERSION}`);
    const text = await response.text();
    const lines = text.split('\n').filter(line => line.trim() !== '');

    let low = 0;
    let high = lines.length - 1;

    while (low <= high) {
        const mid = Math.floor((low + high) / 2);
        const parts = lines[mid].trim().split(/\s+/);
        const id = parts[0];

        if (id === tsumegoId) {
            const zipId = parts[1];
            const size = parts.length >= 3 ? parseInt(parts[2]) : null;
            return { zipId, size };
        } else if (id < tsumegoId) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return null;
}