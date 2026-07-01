// js/problem-loader.js

import * as Sentry from "@sentry/browser";

import { state, zipCache, embeddingsIndexCache, setAcceptClicks, setChallengeProblem, setCurrentProblem, setPreviousProblem, setAppMode, setCurrentSolvingStartTime, resetAccumulatedSolvingTime, setBoardSize, getResumeProblem } from './state.js';
import { loadSgf } from './board.js';
import { updateAllUI } from './ui.js';
import { displayOverlay } from './overlays.js';
import { reorientSgf, tsumegoIdOf } from './sgf.js';
import { RECALL_CANDIDATE_HISTORY_LENGTH, APP_MODES } from './constants.js';
import { getRecallCandidates } from './model.js';
import { weaknessScore, WEAKNESS_TEMPERATURE } from './weakness.js';
import { normalizedRankForSize, slopeToQuality } from './rank-decomposition.js';
import { DATA_VERSION } from './data-version.js';

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

// Picks the most-due recall candidate (a raw pool problem, no SGF yet), or null
// when none is due / available. Tries the in-memory pool, then loads the
// candidate's zip; gives up on a candidate by marking it unavailable and
// recursing. Recall problems must belong to the active size (answerHistory is
// per-size) — cross-size matches are skipped defensively.
async function selectRecallCandidate() {
    const recallCandidates = getRecallCandidates();
    if (recallCandidates.length === 0) return null;

    recallCandidates.sort((a, b) => a.recallRate - b.recallRate);
    const chosen = recallCandidates[0];

    let problem = state.allProblems.find(p => p.filename === `${chosen.problemId}.sgf`);
    if (problem) return problem;

    console.warn(`Recall problem ${chosen.problemId} not in pool. Loading its zip.`);
    const lookupResult = await findZipIdForTsumegoId(chosen.problemId);
    if (lookupResult && (lookupResult.size === null || lookupResult.size === state.boardSize)) {
        await loadZipMetadata();
        const zipInfo = state.zipMetadata.find(zip => zip.filename.startsWith(lookupResult.zipId));
        if (zipInfo) {
            await loadZipProblems(zipInfo.filename);
            problem = state.allProblems.find(p => p.filename === `${chosen.problemId}.sgf`);
            if (problem) return problem;
        }
    }

    console.error(`Failed to load recall problem ${chosen.problemId}.`);
    state.unavailableProblemIds.add(chosen.problemId);
    return await selectRecallCandidate();
}

// Picks a fresh (novel) challenge candidate near the user's rank — a raw pool
// problem, no SGF yet — or null when none is available (caller loads a closer
// zip and retries). Excludes recently-answered problems and any blacklisted by
// the session's load-failure retry loop.
function selectChallengeCandidate() {
    const recentAnswers = state.answerHistory.slice(-RECALL_CANDIDATE_HISTORY_LENGTH);
    const excluded = new Set(recentAnswers.map(a => `${a.problemId}.sgf`));
    const available = state.allProblems.filter(p =>
        !excluded.has(p.filename) &&
        !state.unavailableProblemIds.has(tsumegoIdOf(p.filename)));

    if (available.length === 0) return null;

    // Serve near the SERVE rank (estimate + the uncertainty-scaled UCB bonus), not
    // the displayed rank: while the rank estimate is still loose this pushes
    // problems slightly harder (faster, more informative onboarding), collapsing to
    // at-level once the posterior tightens. Display/scoring still read userRank.
    const serveRank = state.serveRank;
    available.sort((a, b) => Math.abs(a.rank - serveRank) - Math.abs(b.rank - serveRank));
    const candidateProblems = available.slice(0, 50);

    // Kick off background loading of a closer zip if our candidate pool is thin
    // or a not-yet-loaded shard sits nearer the serve rank.
    if (candidateProblems.length < 50) {
        backgroundLoadCloserZip();
    } else {
        const closestZip = findClosestUnloadedZip();
        if (closestZip) {
            const candidateDiff = Math.abs(candidateProblems[49].rank - serveRank);
            const zipDiff = Math.min(Math.abs(serveRank - closestZip.minRank), Math.abs(serveRank - closestZip.maxRank));
            if (zipDiff < candidateDiff) backgroundLoadCloserZip();
        }
    }

    // Weight by quality, modulated by alignment with the user's weakness vector.
    // weaknessScore returns 0 until the per-component model has trained, so the
    // cold-start path here matches the prior pure-quality selection.
    return weightedRandomSelect(candidateProblems, p => {
        const q = Math.max(0, p.quality || 0);
        const w = weaknessScore(p);
        return q * Math.exp(WEAKNESS_TEMPERATURE * w);
    });
}

// Turns a raw pool candidate into a ready-to-display problem object: fetches the
// SGF (from the cached zip) and reorients it. Novel/challenge problems are served
// at identity orientation 0 (the share-link contract / picked-embedding indexing
// assume it); recalls are randomly reoriented so the visual pattern can't be
// recognized. Throws on a missing/invalid SGF so callers can blacklist + retry.
async function buildProblem(raw, { isRepeat, linked, orientation }) {
    const sgfContent = raw.sgf || await fetchSgfFromZip(raw.filename, raw.zipFilename);
    if (!sgfContent || !sgfContent.trim()) throw new Error('Invalid or empty SGF');
    // Deep links may pass an explicit orientation (the share-link ?o contract,
    // where null = randomize); otherwise novel problems are served at identity
    // and repeats randomized.
    const reoriented = reorientSgf(sgfContent, orientation !== undefined ? orientation : (isRepeat ? null : 0));
    return {
        filename: raw.filename,
        sgf: reoriented.sgf,
        linked,
        // `rating` is the raw black-box rating; `rank` is its normalized 0..1
        // rank under the current curve M (derived at ingest).
        rating: raw.rating,
        rank: raw.rank,
        quality: raw.quality || 0,
        slope: raw.slope,
        embedding: raw.embedding,
        isRepeat,
        orientation: reoriented.orientation,
        // Retained so picked-embedding can locate the sidecar .embeddings.bin.
        zipFilename: raw.zipFilename,
    };
}

async function fetchSgfFromZip(filename, zipFilename) {
    if (!state.loadedZips[zipFilename]) await loadZipProblems(zipFilename);
    const zip = zipCache.get(zipFilename);
    if (!zip) throw new Error(`Zip file ${zipFilename} not found in cache.`);
    const file = zip.file(filename);
    if (!file) throw new Error(`File ${filename} not found in ${zipFilename}`);
    return await file.async('text');
}

async function loadZipMetadata() {
    if (state.zipMetadata.length === 0) {
        const response = await fetch(`tsumegos/${state.boardSize}/metadata.json?v=${DATA_VERSION}`);
        const zips = await response.json();
        // Shard min/max_rating are raw ratings; derive their rank bounds with M
        // so selection (which works in rank space against state.userRank) can use
        // them. M is monotone, so min/max map to min/max.
        for (const z of zips) {
            z.minRank = normalizedRankForSize(z.min_rating, state.boardSize);
            z.maxRank = normalizedRankForSize(z.max_rating, state.boardSize);
        }
        state.zipMetadata = zips;
    }
}

async function loadZipProblems(zipFilename) {
    if (state.loadedZips[zipFilename]) return;

    const cacheKey = `tsumegos/${state.boardSize}/${zipFilename}`;
    let zipResponse;
    try {
        zipResponse = await fetch(cacheKey);
    } catch (fetchError) {
        // A network-level *reject* lands here — distinct from the resolved-but-404
        // case handled below. On mobile Safari a dropped/backgrounded connection
        // surfaces as a bare `TypeError: Load failed`, and navigating away mid-fetch
        // throws an AbortError. Re-throw a clearly labelled, retryable error so the
        // outer retry/reload loop handles it and Sentry groups it as a zip-fetch
        // failure rather than an unattributed "Load failed".
        const networkError = new Error(`Failed to fetch zip ${zipFilename}: ${fetchError && fetchError.message}`);
        networkError.networkError = true;
        networkError.cause = fetchError;
        // Aborts are almost always benign (tab close / navigation) — flag so the
        // retry loop can skip the Sentry report.
        if (fetchError && fetchError.name === 'AbortError') networkError.aborted = true;
        throw networkError;
    }

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
    // `p.rating` is the raw black-box rating shipped by package_for_pwa; derive
    // the normalized 0..1 rank here with M (the client owns rating→rank). `p.slope`
    // is the raw 2PL discrimination a_p; keep it raw (the rank estimator uses it)
    // and derive the selection weight (∝ Fisher information a_p²) from it here.
    const problems = JSON.parse(metadataContent).map(p => ({
        ...p, zipFilename,
        rank: normalizedRankForSize(p.rating, state.boardSize),
        quality: slopeToQuality(p.slope),
    }));

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
    const serveRank = state.serveRank;   // load shards near where we SERVE, not display
    const unloadedZips = state.zipMetadata.filter(zip => !state.loadedZips[zip.filename]);
    return unloadedZips.reduce((closest, zip) => {
        const diff = Math.min(Math.abs(serveRank - zip.minRank), Math.abs(serveRank - zip.maxRank));
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

const MAX_LOAD_ATTEMPTS = 5;
const RELOAD_COOLDOWN_MS = 30000;

// Strips the share/deep-link params (`?id`/`?o`/`?recall`) and the entry-only
// `/9`|`/13` size-hint path, collapsing the address bar to `/`. We deliberately
// do NOT clean while a linked (shared) problem is on screen — its `?id`/`?o`
// stay in the URL so that ANY reload (a service-worker update takeover, a manual
// refresh, an in-app-browser reload) re-enters the SAME shared problem at boot
// instead of silently dropping the user onto the standing challenge. The URL is
// cleaned the moment they move on to a non-linked problem — every such transition
// repaints through displayProblem below. No-ops once the URL is already clean.
function cleanUrl() {
    const url = new URL(window.location);
    url.searchParams.delete('id');
    url.searchParams.delete('o');
    url.searchParams.delete('recall');
    let newPath = url.pathname.replace(/index\.html$/, '');
    if (/^\/(9|13)\/?$/.test(newPath)) newPath = '/';
    const next = newPath + url.search + url.hash;
    const current = window.location.pathname + window.location.search + window.location.hash;
    if (next !== current) window.history.replaceState({}, '', next);
}

// Installs a ready (SGF-carrying) problem as the on-screen problem: sets it as
// currentProblem, paints the board, resets the solve timer, re-enables clicks.
// Throws if the SGF won't load so the caller can blacklist + reselect.
function displayProblem(problem, appMode) {
    setCurrentProblem(problem);
    setAppMode(appMode);
    // Keep a shared problem's deep-link in the address bar so a reload re-enters
    // it; strip it the instant we paint any non-linked problem (see cleanUrl).
    if (appMode !== APP_MODES.LINKED) cleanUrl();
    try {
        loadSgf(problem.sgf);
    } catch (error) {
        const e = new Error('Invalid SGF: ' + error.message);
        e.failedProblemId = tsumegoIdOf(problem.filename);
        throw e;
    }
    setCurrentSolvingStartTime(Date.now());
    resetAccumulatedSolvingTime();
    setAcceptClicks(true);
    updateAllUI();
}

function reportLoadError(error, attempt) {
    console.error(`Tsumego load attempt ${attempt + 1}/${MAX_LOAD_ATTEMPTS} failed:`, error);
    // An AbortError from navigating away mid-fetch is benign noise — skip Sentry.
    if (!error.aborted) {
        try {
            Sentry.captureException(error, {
                extra: { attempt: attempt + 1, failedProblemId: error.failedProblemId, boardSize: state.boardSize },
            });
        } catch (_) {}
    }
    if (error.failedProblemId) state.unavailableProblemIds.add(error.failedProblemId);
}

// All loads failed. Restore clicks so the user is never frozen, then try one
// cooled-down reload (fresh metadata may unblock us); otherwise show an error.
function handleLoadFailure(lastError) {
    setAcceptClicks(true);
    console.error(`Tsumego load failed after ${MAX_LOAD_ATTEMPTS} attempts; last error:`, lastError);
    const lastReload = parseInt(localStorage.getItem('lastReloadTime') || '0', 10);
    const recentlyReloaded = Date.now() - lastReload < RELOAD_COOLDOWN_MS;
    if (state.userHasInteracted && !recentlyReloaded) {
        try { Sentry.captureMessage('Reloading after repeated tsumego load failures', { level: 'warning' }); } catch (_) {}
        localStorage.setItem('lastReloadTime', String(Date.now()));
        displayOverlay('Error loading problem. Reloading…', 1500);
        setTimeout(() => location.reload(), 1500);
    } else {
        displayOverlay('Error loading problem. Try refreshing the page.', 5000);
    }
}

// Selects + builds a fresh challenge and stores it in state.challengeProblem.
// Does NOT display it. This is the single point where the standing challenge is
// (re)assigned: on first load, and whenever the challenge itself is attempted
// (completeCurrentProblem). Retries past bad SGFs, blacklisting each.
export async function assignNewChallenge() {
    await loadZipMetadata();
    let lastError = null;
    for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt++) {
        let raw = selectChallengeCandidate();
        if (!raw) {
            const closestZip = findClosestUnloadedZip();
            if (!closestZip) { lastError = new Error('No zip files found in metadata.'); break; }
            await loadZipProblems(closestZip.filename);
            raw = selectChallengeCandidate();
            if (!raw) { lastError = new Error('Failed to select a challenge after loading zip.'); break; }
        }
        try {
            const problem = await buildProblem(raw, { isRepeat: false, linked: false });
            setChallengeProblem(problem);
            return problem;
        } catch (error) {
            lastError = error;
            error.failedProblemId = tsumegoIdOf(raw.filename);
            reportLoadError(error, attempt);
        }
    }
    console.error('assignNewChallenge exhausted retries:', lastError);
    return state.challengeProblem; // possibly null — advanceToNextProblem handles it
}

// True if the stored challenge has already been answered (it sits in the recent
// answer window). Guards the tiny window where the app is killed after an attempt
// is recorded but before completeCurrentProblem reassigns the challenge: on the
// next load we must NOT re-serve that already-solved problem as a novel challenge.
function challengeAlreadyAnswered() {
    const c = state.challengeProblem;
    if (!c) return false;
    const id = tsumegoIdOf(c.filename);
    return state.answerHistory.slice(-RECALL_CANDIDATE_HISTORY_LENGTH).some(a => a.problemId === id);
}

// Builds the most-due recall as a ready problem, or null when none is due or it
// can't be loaded (caller falls back to the challenge).
async function getDueRecallProblem() {
    let raw;
    try {
        raw = await selectRecallCandidate();
    } catch (error) {
        console.error('recall selection failed:', error);
        return null;
    }
    if (!raw) return null;
    try {
        return await buildProblem(raw, { isRepeat: true, linked: false });
    } catch (error) {
        console.error('recall build failed; falling back to challenge:', error);
        state.unavailableProblemIds.add(tsumegoIdOf(raw.filename));
        return null;
    }
}

// Shows the next problem the user should work on: a due recall if one exists,
// otherwise the standing challenge (assigned first if absent). The single entry
// point for "I need a problem on screen" — used on load, after a solve, on
// notification-with-no-recall, etc. Recalls never disturb challengeProblem.
export async function advanceToNextProblem() {
    setAcceptClicks(false);
    try {
        await loadZipMetadata();
    } catch (_) { /* fall through; empty-challenge path handles total failure */ }

    // A challenge is ALWAYS assigned — even when a recall is what we show, the
    // challenge stays off-screen, ready for when recalls are exhausted. (This
    // also reselects a stale challenge left by a kill mid-completion; see
    // challengeAlreadyAnswered.)
    if (!state.challengeProblem || challengeAlreadyAnswered()) await assignNewChallenge();

    const recall = await getDueRecallProblem();
    if (recall) {
        try { displayProblem(recall, APP_MODES.SOLVING); return; }
        catch (error) { reportLoadError(error, 0); /* fall back to challenge */ }
    }

    let lastError = null;
    for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt++) {
        const challenge = state.challengeProblem;
        if (!challenge) { lastError = lastError || new Error('No challenge available.'); break; }
        try { displayProblem(challenge, APP_MODES.SOLVING); return; }
        catch (error) {
            lastError = error;
            reportLoadError(error, attempt);
            // The stored challenge's SGF is bad — drop it and reselect a fresh one.
            setChallengeProblem(null);
            await assignNewChallenge();
        }
    }

    handleLoadFailure(lastError);
}

// Re-shows the current problem (it's already loaded underneath) — used when
// leaving review (which painted the previous problem). The mode follows the
// current problem: a linked/shared problem returns to LINKED view, otherwise
// SOLVING. Falls forward to a fresh advance if there's nothing to redisplay.
export async function redisplayCurrentProblem() {
    if (!state.currentProblem) return advanceToNextProblem();
    const mode = state.currentProblem.linked ? APP_MODES.LINKED : APP_MODES.SOLVING;
    try { displayProblem(state.currentProblem, mode); }
    catch (error) { reportLoadError(error, 0); return advanceToNextProblem(); }
}

// On a same-session reload (notably the version-driven graceful reload, but also
// a missing-zip reload or a manual refresh), restore the exact problem that was on
// screen — at its stored orientation — instead of re-running the pick/orient flow.
// Recalls are served at a random orientation; re-running advanceToNextProblem would
// re-randomize it and visibly rotate the board, which confused users. Returns true
// if it restored a problem; false (no resume slot — i.e. a fresh session, or a bad
// SGF) so the caller falls through to advanceToNextProblem.
export async function resumeCurrentProblem() {
    const resumed = getResumeProblem();
    if (!resumed) return false;
    try {
        // Make the resumed problem fully first-class: load its shard so the
        // picked-move embedding sidecar is available if the user gets it wrong
        // (matches what the normal load path guarantees). Best-effort — the
        // problem still displays from its stored SGF if the zip can't load.
        await loadZipMetadata();
        if (resumed.zipFilename) await loadZipProblems(resumed.zipFilename);
    } catch (error) {
        reportLoadError(error, 0);
    }
    try {
        displayProblem(resumed, APP_MODES.SOLVING);
        return true;
    } catch (error) {
        reportLoadError(error, 0);
        return false;
    }
}

// Exits a linked/shared view back to the user's standing challenge.
export async function returnToChallenge() {
    if (!state.challengeProblem || challengeAlreadyAnswered()) await assignNewChallenge();
    if (!state.challengeProblem) return advanceToNextProblem();
    try { displayProblem(state.challengeProblem, APP_MODES.SOLVING); }
    catch (error) { reportLoadError(error, 0); return advanceToNextProblem(); }
}

// Called once the current problem's attempt has been recorded. Moves it into the
// previous slot (for review/share/ignore) and, if it was the standing challenge,
// assigns a fresh one immediately — so the just-answered challenge is replaced
// in persisted state before any reopen can re-serve it. Recalls and linked
// problems leave the challenge untouched.
export async function completeCurrentProblem() {
    const done = state.currentProblem;
    setPreviousProblem(done);
    if (done && !done.isRepeat && !done.linked) {
        await assignNewChallenge();
    }
}

export async function loadTsumegoById(tsumegoId, orientation = null, { asRecall = false } = {}) {
    try {
        const showing = state.currentProblem;
        if (showing && showing.filename === `${tsumegoId}.sgf` &&
           (orientation === null || showing.orientation === orientation)) {
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

        // A deep-linked share is a `linked` problem (view-only, never recorded);
        // a notification recall is a normal recall. Either way this becomes the
        // on-screen currentProblem WITHOUT touching the standing challengeProblem
        // (which stays assigned and persisted, shown when the user exits this
        // detour — see returnToChallenge / advanceToNextProblem). The raw pool
        // problem may be missing (defensive) — fall back to a bare descriptor so
        // the SGF still loads, just without rating/embedding fields.
        const problem = state.allProblems.find(p => p.filename === `${tsumegoId}.sgf`)
            || { filename: `${tsumegoId}.sgf`, zipFilename, rating: 0, rank: 0, quality: 0 };
        const newProblem = await buildProblem(problem, {
            isRepeat: asRecall, linked: !asRecall, orientation,
        });

        displayProblem(newProblem, asRecall ? APP_MODES.SOLVING : APP_MODES.LINKED);

        return true;
    } catch (error) {
        console.error(`Error loading tsumego by ID ${tsumegoId}:`, error);
        return false;
    }
}

// Fetches the global lookup.txt corpus manifest (one "id zipHash size" row
// per problem, sorted by id) and returns its non-empty lines, or null on any
// fetch failure so callers can bail rather than act on a partial manifest.
// The single fetch/split for every lookup.txt consumer (the resolver below
// and backfill.js) — keep it that way so the parses can't drift.
export async function fetchLookupLines() {
    try {
        const response = await fetch(`lookup.txt?v=${DATA_VERSION}`);
        if (!response.ok) return null;
        const text = await response.text();
        return text.split('\n').filter(line => line.trim() !== '');
    } catch (_) {
        return null;
    }
}

// Resolves a tsumego id to its zip hash and board size by binary-searching
// the global lookup.txt. Each row is "id zipHash size". Returns null when
// the id is unknown (or the manifest can't be fetched), otherwise
// { zipId, size }.
async function findZipIdForTsumegoId(tsumegoId) {
    const lines = await fetchLookupLines();
    if (!lines) return null;

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