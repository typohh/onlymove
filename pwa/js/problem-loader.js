// js/problem-loader.js

import { state, zipCache, setAcceptClicks, updateProblemHistory, updateNextTsumego, setAppMode, setCurrentSolvingStartTime, resetAccumulatedSolvingTime } from './state.js';
import { loadSgf } from './board.js';
import { updateAllUI } from './ui.js';
import { displayOverlay } from './overlays.js';
import { reorientSgf } from './sgf.js';
import { MODEL_INDICES } from './user-stats.js';

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



export function getReviewCandidates() {
    const now = Date.now();
    const reviewCandidates = [];

    if (!state.modelParams || state.modelParams.length === 0) {
        return []; // Model not ready
    }
    const finalParams = state.modelParams;

    const recentHistory = state.answerHistory.slice(-500);

    const lastAnswers = {};
    for (const answer of recentHistory) {
        lastAnswers[answer.problemId] = answer;
    }

    for (const problemId in lastAnswers) {
        const lastAnswer = lastAnswers[problemId];

        // Conditions: failed at least once, and problem rating is lower than user rank.
        if (lastAnswer.successSinceFailure !== undefined && lastAnswer.problemRating < state.userRank) {
            
            const features = Array(finalParams.length).fill(0);

            features[MODEL_INDICES.INTERCEPT] = 1;
            features[MODEL_INDICES.TIME_SOLVING] = 0.5; // Neutral value for prediction

            // Approximate review time percentile
            const reviewTime = state.answerHistory
                .filter(ans => ans.problemId === problemId && ans.date < now)
                .reduce((sum, ans) => sum + (ans.reviewTime || 0), 0);
            features[MODEL_INDICES.REVIEW_TIME] = reviewTime > 0 ? 0.5 : 0;

            features[MODEL_INDICES.PROBLEM_DIFFICULTY] = lastAnswer.problemRating;
            features[MODEL_INDICES.PROBLEM_QUALITY] = lastAnswer.problemQuality || 0.5;
            features[MODEL_INDICES.DATE] = 1.0; // For a new attempt, it's the latest
            features[MODEL_INDICES.IS_NOVEL] = 0; // This is a review problem

            const t = now - lastAnswer.date;
            const r = lastAnswer.successSinceFailure;
            const interactionTerm = Math.log(1 + t * Math.pow(0.5, r));
            features[MODEL_INDICES.RECALL_FORMULA] = interactionTerm;

            const logit = features.reduce((acc, val, j) => acc + val * finalParams[j], 0);
            const probability = 1 / (1 + Math.exp(-logit));

            if (probability >= 0.60 && probability <= 0.80) {
                reviewCandidates.push({ 
                    problemId,
                    recallRate: probability,
                    rating: lastAnswer.problemRating 
                });
            }
        }
    }

    return reviewCandidates;
}

async function selectProblem() {
    if (state.allProblems.length === 0) return null;

    // Check for nextTsumego first
    if (state.nextTsumego) {
        const problem = state.nextTsumego;
        updateNextTsumego(null);
        return problem;
    }

    const reviewCandidates = getReviewCandidates();

    if (reviewCandidates.length > 0) {
        reviewCandidates.sort((a, b) => a.recallRate - b.recallRate);
        const chosenProblemInfo = reviewCandidates[0];
        let problem = state.allProblems.find(p => p.filename === `${chosenProblemInfo.problemId}.sgf`);

        if (problem) {
            return { ...problem, isRepeat: true };
        }

        // If problem is not in memory, load its zip file
        console.warn(`Problem ${chosenProblemInfo.problemId} not found in allProblems. Attempting to load it.`);

        const zipId = await findZipIdForTsumegoId(chosenProblemInfo.problemId);
        if (zipId) {
            await loadZipMetadata(); // Ensure metadata is loaded
            const zipInfo = state.zipMetadata.find(zip => zip.filename.startsWith(zipId));
            if (zipInfo) {
                await loadZipProblems(zipInfo.filename);
                // Try to find the problem again
                problem = state.allProblems.find(p => p.filename === `${chosenProblemInfo.problemId}.sgf`);
                if (problem) {
                    return { ...problem, isRepeat: true };
                }
            }
        }
        
        // If we still can't find it, log an error and fall back.
        console.error(`Failed to load review problem ${chosenProblemInfo.problemId}.`);
    }

    // Default selection logic (fallback)
    const recentAnswers = state.answerHistory.slice(-500);
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

    // Select a problem using weighted random sampling based on quality
    const problem = weightedRandomSelect(candidateProblems, p => p.quality);
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
        const response = await fetch(`metadata.json?v=${APP_VERSION}`);
        state.zipMetadata = await response.json();
        const metadataStr = JSON.stringify(state.zipMetadata);
        console.log('Metadata hash:', await simpleHash(metadataStr));
    }
}

export async function loadZipProblems(zipFilename) {
    if (state.loadedZips[zipFilename]) return;

    const cacheKey = `tsumegos/${zipFilename}`;
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

export async function loadTsumego(isUpdate) {
    try {
        
        setAcceptClicks(false);

        if (!isUpdate && state.problemHistory.length > 0) {
            // On initial load, just load the last problem from history
            loadSgf(state.problemHistory[0].sgf);
            setCurrentSolvingStartTime(Date.now());
            resetAccumulatedSolvingTime();
            setAcceptClicks(true);
            return;
        }

        await loadZipMetadata();

        let selectedProblem = await selectProblem(); // Await the result
        if (selectedProblem) {
        }

        if (!selectedProblem) {
            const closestZip = findClosestUnloadedZip();
            if (closestZip) {
                await loadZipProblems(closestZip.filename);
                selectedProblem = await selectProblem(); // Await the result
                if (!selectedProblem) throw new Error("Failed to select a problem after loading zip.");
            } else {
                throw new Error("No zip files found in metadata.");
            }
        }

        

        let sgf = selectedProblem.sgf || await fetchSgfFromZip(selectedProblem.filename, selectedProblem.zipFilename);
        if (!sgf || !sgf.trim()) {
            throw new Error('Invalid or empty SGF');
        }
        sgf = reorientSgf(sgf);

        const history = state.problemHistory;
        
        // Remove all entries with the same filename
        const filename = selectedProblem.filename;
        const indicesToRemove = [];
        for (let i = 0; i < history.length; i++) {
            if (history[i].filename === filename) {
                indicesToRemove.push(i);
            }
        }
        // Remove in reverse order to maintain indices
        for (let i = indicesToRemove.length - 1; i >= 0; i--) {
            history.splice(indicesToRemove[i], 1);
        }

        history.unshift({
            filename: selectedProblem.filename,
            sgf: sgf,
            linked: false,
            rating: selectedProblem.rating,
            quality: selectedProblem.quality || 0,
            isRepeat: selectedProblem.isRepeat || false
        });

        if (history.length > 2) {
            history.length = 2;
        }
        
        updateProblemHistory(history);

        try {
            loadSgf(sgf);
        } catch (error) {
            throw new Error('Invalid SGF: ' + error.message);
        }
        
        setAcceptClicks(true);
        setCurrentSolvingStartTime(Date.now());
        resetAccumulatedSolvingTime();
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

        const problem = state.allProblems.find(p => p.filename === `${tsumegoId}.sgf`);
        const problemRank = problem ? problem.rating : 0;
        const problemQuality = problem ? problem.quality : 0;

        let sgfContent = await fetchSgfFromZip(`${tsumegoId}.sgf`, zipFilename);
        sgfContent = reorientSgf(sgfContent);

        const history = state.problemHistory;

        const newProblem = {
            filename: `${tsumegoId}.sgf`,
            sgf: sgfContent,
            linked: true,
            rating: problemRank,
            quality: problemQuality,
            isRepeat: false
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

        setAppMode('linked');

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