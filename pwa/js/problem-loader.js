// js/problem-loader.js

import * as Sentry from "@sentry/browser";

import { state, zipCache, embeddingsIndexCache, setAcceptClicks, setChallengeProblem, setCurrentProblem, setPreviousProblem, setAppMode, setCurrentSolvingStartTime, resetAccumulatedSolvingTime, setBoardSize, getResumeProblem, getMergedAnswerHistory } from './state.js';
import { loadSgf } from './board.js';
import { updateAllUI } from './ui.js';
import { displayOverlay, fadeOverlay, showToast } from './overlays.js';
import { getTerm } from './i18n.js';
import { reorientSgf, tsumegoIdOf } from './sgf.js';
import { RECALL_CANDIDATE_HISTORY_LENGTH, APP_MODES } from './constants.js';
import { getRecallCandidates, recallServeShare } from './recall-model.js';
import { weaknessScore, weaknessModelReady } from './weakness.js';
import { rankForSize, problemInfoWeight, rankConverged } from './rank-decomposition.js';
import { DATA_VERSION } from './data-version.js';
import { onSolvingProblemShown as onboardingSolvingProblemShown } from './onboarding.js';

// The best candidate by `scoreFn`, ties going to the earliest — which, because the
// pool arrives sorted by distance from the serve rank, means the closest-to-rank of
// equally good problems. Both selection regimes take the MAXIMUM rather than sampling
// in proportion to it: a proportional draw over 50 difficulty-matched candidates
// barely departs from uniform (the weights span a narrow range, so the effective pool
// stays near its full size), which spends the whole candidate set to move the served
// problem almost not at all. Variety comes from the pool turning over instead — every
// answered problem is excluded for the next RECALL_CANDIDATE_HISTORY_LENGTH attempts,
// so the maximum is a different problem each time.
function selectBest(items, scoreFn) {
    let best = items[0];
    let bestScore = -Infinity;
    for (const item of items) {
        const s = scoreFn(item);
        if (Number.isFinite(s) && s > bestScore) {
            bestScore = s;
            best = item;
        }
    }
    return best;
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

    // TWO REGIMES, each taking the BEST candidate in its own criterion. While the
    // rank estimate is still converging, serve for INFORMATION: quality = the 4PL
    // Fisher information at the 80%-clear anchor, slope²·anchorInfo(floor) — the pool
    // is difficulty-matched so the anchor is the right common reference, but the floor
    // factor varies across it. Once converged (rankConverged: se < 2× the local
    // floor), serve for PERSONALIZATION: the centred weakness contrast alone, in its
    // own units. The regimes partition where the physics does — slope-priority only
    // changes the served set while the local pool is FRESH (a moving user), because a
    // user parked in a band consumes its high-slope tail and converges to the band
    // average regardless; and that fresh phase is exactly when se is still high.
    // Boundary flapping between consecutive picks is harmless, the two criteria being
    // applied to the same pool. Falls back to the information regime while the
    // weakness model is untrained (flat zero scores would otherwise pick whichever
    // candidate happened to sort first).
    if (rankConverged() && weaknessModelReady()) {
        return selectBest(candidateProblems, weaknessScore);
    }
    return selectBest(candidateProblems, p => Math.max(0, p.quality || 0));
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
        // `rating` is the raw black-box rating; `rank` is its normalized rank
        // under the current curve M (derived at ingest; unbounded, not a 0..1
        // fraction — see rankForSize).
        rating: raw.rating,
        rank: raw.rank,
        quality: raw.quality || 0,
        slope: raw.slope,
        // Per-problem 3PL guessing floor — user-stats records it as problemFloor
        // (the rank estimator's Δ-coupled floor discount reads it; omitting it
        // made clampFloor treat every live attempt as floor 0).
        floor: raw.floor,
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
            z.minRank = rankForSize(z.min_rating, state.boardSize);
            z.maxRank = rankForSize(z.max_rating, state.boardSize);
        }
        state.zipMetadata = zips;
    }
}

// In-flight de-dup: loadedZips[zipFilename] is only set AFTER the fetch+parse
// awaits, so two concurrent loads of the same shard (fetchSgfFromZip racing a
// backgroundLoadCloserZip) would both pass the loaded check and push the
// shard's problems into the pool twice — double selection weight, double
// memory — for the rest of the session.
const zipLoadsInFlight = new Map();
function loadZipProblems(zipFilename) {
    if (state.loadedZips[zipFilename]) return Promise.resolve();
    if (zipLoadsInFlight.has(zipFilename)) return zipLoadsInFlight.get(zipFilename);
    const p = doLoadZipProblems(zipFilename)
        .finally(() => zipLoadsInFlight.delete(zipFilename));
    zipLoadsInFlight.set(zipFilename, p);
    return p;
}

async function doLoadZipProblems(zipFilename) {
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
    // the normalized rank here with M (the client owns rating→rank). `p.slope`
    // is the raw 2PL discrimination a_p; keep it raw (the rank estimator uses it)
    // and derive the selection weight (∝ the 4PL Fisher information a_p²·anchorInfo(floor))
    // from it and the shipped guessing floor here.
    const problems = JSON.parse(metadataContent).map(p => ({
        ...p, zipFilename,
        rank: rankForSize(p.rating, state.boardSize),
        quality: problemInfoWeight(p.slope, p.floor),
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

// Unloaded shards sorted nearest-first by rank distance to the serve rank —
// the order both the normal closest-zip load and the exhausted-pool expansion
// walk. (Shards are loaded near where we SERVE, not where we display.)
function unloadedZipsByDistance() {
    const serveRank = state.serveRank;
    const dist = zip => Math.min(Math.abs(serveRank - zip.minRank), Math.abs(serveRank - zip.maxRank));
    return state.zipMetadata
        .filter(zip => !state.loadedZips[zip.filename])
        .sort((a, b) => dist(a) - dist(b));
}

function findClosestUnloadedZip() {
    return unloadedZipsByDistance()[0] || null;
}

// --- Off-rank serving mode (offline fallback) --------------------------------
// Entered when the exhausted-pool expansion could only produce a candidate
// AFTER skipping a nearer shard that failed to load — we are serving problems
// farther from the user's rank than the corpus allows because the nearer
// shards are unreachable (offline / network failure). Exited when the nearest
// unloaded shard loads again (the expansion's first pick, a background
// closer-zip load, or the connectivity-restored retry below). Only the
// TRANSITIONS show a toast; repeated serves inside the mode stay silent so
// the user isn't nagged before every problem.
let offRankServing = false;

function serveModeToast(termKey) {
    try {
        const el = showToast(getTerm(termKey), { autoDismiss: false });
        setTimeout(() => el.remove(), 5000);
    } catch (_) { /* purely informational */ }
}

function enterOffRankServing() {
    if (offRankServing) return;
    offRankServing = true;
    serveModeToast('offRankServingStart');
}

function exitOffRankServing() {
    if (!offRankServing) return;
    offRankServing = false;
    serveModeToast('offRankServingEnd');
}

// Connectivity back: retry the nearest shard right away so the mode exits
// (and the pool re-centers on the rank) without waiting for the next serve
// decision to notice.
window.addEventListener('online', () => {
    if (offRankServing) backgroundLoadCloserZip();
});

// True when the shard zip already sits in the service-worker runtime cache
// (sw.js caches shard zips by their plain URL, cache-first) — i.e. loadable
// without network. Any failure (Cache API unavailable) reads as "not
// cached"; the caller then treats the load as a network attempt. The HTTP
// browser cache can additionally satisfy an "uncached" fetch offline — the
// probe is an optimization, never a correctness gate.
async function isZipCached(zipFilename) {
    try {
        const url = new URL(`tsumegos/${state.boardSize}/${zipFilename}`, window.location.href);
        return !!(await caches.match(url.href));
    } catch (_) {
        return false;
    }
}

// Bounds for the exhausted-pool expansion: cached shards are cheap to try,
// but genuine network fetches can hang on a flaky connection, and every
// loaded shard costs parse time + memory — so both are capped per selection
// call. Progress persists across calls (loadedZips accumulates), so a later
// serve resumes where this one stopped.
const EXPAND_NETWORK_ATTEMPTS = 3;
const EXPAND_ZIP_LOADS = 8;

// Exhausted-pool fallback: the loaded pool has no serveable problem left, so
// walk the unloaded shards nearest-first until one yields a candidate —
// service-worker-cached shards work offline, so probing the cache lets an
// offline user keep playing from whatever is on the device instead of dead-
// ending into the stale re-serve. Returns null when nothing loadable yields
// a candidate (the caller then falls through to the stale re-serve terminal
// path). Also drives the off-rank mode transitions: a candidate that needed
// skipping a FAILED nearer shard enters it; a candidate found with no
// failures means the nearest shards are reachable again — exit.
async function expandPoolForCandidate() {
    let loads = 0;
    let networkAttempts = 0;
    let failures = 0;
    for (const zip of unloadedZipsByDistance()) {
        if (loads >= EXPAND_ZIP_LOADS) break;
        if (!(await isZipCached(zip.filename))) {
            if (networkAttempts >= EXPAND_NETWORK_ATTEMPTS) continue;
            networkAttempts++;
        }
        try {
            await loadZipProblems(zip.filename);
        } catch (error) {
            console.error(`expandPoolForCandidate: zip load failed for ${zip.filename}:`, error);
            failures++;
            continue;
        }
        loads++;
        // A loaded shard can still yield nothing (all its problems recently
        // answered) — that's legitimate exhaustion, not a failure; keep
        // walking outward.
        const raw = selectChallengeCandidate();
        if (raw) {
            if (failures > 0) enterOffRankServing();
            else exitOffRankServing();
            return raw;
        }
    }
    return null;
}

async function backgroundLoadCloserZip() {
    const closestZip = findClosestUnloadedZip();
    if (closestZip) {
        try {
            await loadZipProblems(closestZip.filename);
            // The nearest shard is reachable again → the pool re-centers on
            // the user's rank.
            exitOffRankServing();
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
    abandonExampleReviewIfActive();
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
    // First-run onboarding: a bubble may be offered on the freshly-painted
    // solving board (intro, recall, rank, komi, install, notify). No-op for
    // linked/review paints and for anyone but a genuinely new user.
    if (appMode === APP_MODES.SOLVING) onboardingSolvingProblemShown(problem);
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
    // The selection below belongs to THIS size: a size switch during the awaits
    // must not store an old-size problem into the new size's challenge bucket
    // (it would later be reported with the wrong size and a mis-framed
    // canonical move).
    const sizeAtStart = state.boardSize;
    // A failed metadata fetch (offline) flows into the normal failure path —
    // the caller's challenge-less handling ends in handleLoadFailure, which
    // restores clicks — instead of escaping as an unhandled rejection that
    // leaves the board frozen (acceptClicks false).
    try {
        await loadZipMetadata();
    } catch (error) {
        console.error('assignNewChallenge: metadata load failed:', error);
        return state.challengeProblem;
    }
    let lastError = null;
    for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt++) {
        let raw = selectChallengeCandidate();
        if (!raw) {
            // Loaded pool exhausted — expand outward through unloaded shards
            // (cached ones work offline; see expandPoolForCandidate). Null
            // means nothing loadable yielded a candidate: fall through to the
            // stale-challenge terminal path, same as the pre-expansion
            // zip-load-failure return.
            raw = await expandPoolForCandidate();
            if (!raw) { lastError = new Error('No candidate from any loadable zip.'); break; }
        }
        try {
            const problem = await buildProblem(raw, { isRepeat: false, linked: false });
            if (state.boardSize !== sizeAtStart) return state.challengeProblem;
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

// Deterministic uniform [0,1) from a string: FNV-1a 32-bit with a murmur3
// finalizer mix (the raw FNV low bits are too regular for a threshold compare).
function hashToUnit(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    h ^= h >>> 16;
    h = Math.imul(h, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h, 0xc2b2ae35);
    h ^= h >>> 16;
    return (h >>> 0) / 0x100000000;
}

// Stochastic recall gate: with probability recallServeShare() a serve decision
// admits a due recall; otherwise the novel challenge is served even though a
// recall is due (the skipped recall stays in its window and can win a later
// roll, or ages out — that pruning is the equilibrium's decay channel). The
// share is the constant-volume balance point for a FRACTION of the user's own
// trailing-week daily volume — the headroom that lets a below-average day still
// clear every due recall — and for their memory (see recall-model.js
// recallServeShare); while it is not yet
// estimable (null), due recalls always win — the pre-gate behavior. Rolled
// once per serve decision, BEFORE the recall is built (a gated-out recall
// costs no shard fetch); P(recall | one is due) = share either way since the
// roll is independent of dueness. Notification clicks (?recall=1, main.js)
// deliberately bypass this — the user explicitly came for a recall.
//
// The roll is DETERMINISTIC — hashed from (most-due candidate id, last
// recorded attempt date), not Math.random(). refreshShownProblemAfterSync
// re-runs this gate on EVERY tab-visible resume, so an independent roll there
// would compound: n task-switches over an idle novel give the due recall
// 1−(1−share)^n chances to swap onto the board instead of share. Seeded, a
// resume re-derives the SAME outcome as the serve decision it re-runs (same
// candidate, same history ⇒ same seed). Every new attempt — local or a synced
// foreign one — appends a fresh client-minted date, re-seeding the roll, so
// genuine serve decisions remain independent Bernoulli(share) draws and the
// equilibrium math is untouched. (The share itself drifts with wall-clock
// trailing-week volume, so a long idle can still flip the comparison — rare,
// and the next attempt re-seeds anyway.)
function recallGateAllows() {
    // Fail open (serve the recall): selectNextProblem's never-throws contract
    // must not hinge on the share estimator, and the pre-gate behavior is the
    // safe default.
    try {
        const share = recallServeShare();
        if (share == null) return true;
        // Same cheap sync selection as selectRecallCandidate — no shard fetch.
        const candidates = getRecallCandidates();
        if (candidates.length === 0) return true; // nothing due — outcome moot
        candidates.sort((a, b) => a.recallRate - b.recallRate);
        const merged = getMergedAnswerHistory();
        const lastDate = (merged.length && merged[merged.length - 1].date) || 0;
        return hashToUnit(`${candidates[0].problemId}|${lastDate}`) < share;
    } catch (error) {
        console.error('recall gate failed; serving the recall:', error);
        return true;
    }
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

// Selects (but does not display) the next problem the user should work on: a
// due recall if one exists, otherwise the standing challenge (assigned first if
// absent). Never throws; returns null when nothing could be selected — the
// display phase then runs the failure path. Recalls never disturb
// challengeProblem.
async function selectNextProblem() {
    try {
        await loadZipMetadata();
    } catch (_) { /* fall through; empty-challenge path handles total failure */ }

    // A challenge is ALWAYS assigned — even when a recall is what we show, the
    // challenge stays off-screen, ready for when recalls are exhausted. (This
    // also reselects a stale challenge left by a kill mid-completion; see
    // challengeAlreadyAnswered.)
    if (!state.challengeProblem || challengeAlreadyAnswered()) await assignNewChallenge();

    const recall = recallGateAllows() ? await getDueRecallProblem() : null;
    return recall || state.challengeProblem;
}

// A freshly painted challenge that is still inside the recent-answer window
// means challenge reselection failed — selectChallengeCandidate never picks
// from that window, so the only way an already-answered problem lands here is
// assignNewChallenge's failure returns keeping the stale slot (offline / a
// shard fetch failure once the loaded pool is exhausted). The re-serve is
// deliberate (the board stays usable, and due recalls from cached shards can
// still interleave), but silently repeating a just-solved problem reads as a
// bug — explain it. Button-held so it can't be missed under the answer-result
// fade it appears over; grey per the neutral-overlay convention (white =
// correct, red = wrong).
function warnIfServedAnsweredChallenge() {
    if (!challengeAlreadyAnswered()) return;
    displayOverlay(getTerm('outOfProblemsMessage'), 0, null,
        'rgb(128, 128, 128)', getTerm('outOfProblemsButton'));
}

// Paints a selected problem, falling forward past failures: a bad recall falls
// back to the challenge; a bad challenge is blacklisted and a fresh one
// reselected (up to MAX_LOAD_ATTEMPTS).
async function displayNextProblem(next) {
    if (next && next.isRepeat) {
        try { displayProblem(next, APP_MODES.SOLVING); return; }
        catch (error) { reportLoadError(error, 0); /* fall back to challenge */ }
    }

    let lastError = null;
    for (let attempt = 0; attempt < MAX_LOAD_ATTEMPTS; attempt++) {
        const challenge = state.challengeProblem;
        if (!challenge) { lastError = lastError || new Error('No challenge available.'); break; }
        try { displayProblem(challenge, APP_MODES.SOLVING); warnIfServedAnsweredChallenge(); return; }
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

// Shows the next problem the user should work on: a due recall if one exists,
// otherwise the standing challenge. The single entry point for "I need a
// problem on screen" — used on load, after a solve, on
// notification-with-no-recall, etc.
export async function advanceToNextProblem() {
    setAcceptClicks(false);
    await displayNextProblem(await selectNextProblem());
}

// Review-exit advance: after a wrong answer the next problem is deliberately
// NOT loaded before review (review starts under the red overlay with nothing
// underneath — see completeCurrentProblem/_handleIncorrectMoveSequence), so
// selection happens only now, when the user leaves review. The common case —
// prefetched challenge, in-pool recall — settles within the grace window and
// paints directly, matching the old instant redisplay. Only when selection is
// still in flight after the grace period (e.g. a due recall's shard has to be
// downloaded) does a grey "loading next problem" fade cover the wait, held up
// until the swap has painted so it never lifts onto the stale review board.
const SELECT_GRACE_MS = 150;
async function paintWhenSelected(selection, paint) {
    let settled = false;
    const tracked = selection.finally(() => { settled = true; });
    await Promise.race([tracked, new Promise(resolve => setTimeout(resolve, SELECT_GRACE_MS))]);
    if (settled) {
        await paint(await tracked);
        return;
    }
    await new Promise((resolve) => {
        fadeOverlay(500, async () => {
            try { await paint(await tracked); }
            finally { resolve(); }
        }, null, getTerm('loadingNextProblem'), { background: 'rgb(128, 128, 128)', text: 'white' });
    });
}

async function advanceAfterReview() {
    setAcceptClicks(false);
    await paintWhenSelected(selectNextProblem(), displayNextProblem);
}

// The Recall button: the user explicitly asked for a due recall, so this
// bypasses the stochastic serve gate exactly as a notification click does
// (recallGateAllows shapes what we serve UNPROMPTED). Otherwise it is an
// ordinary recall serve — same most-due pick, same random reorientation — and
// the standing challenge is left untouched, so it comes back as soon as this
// recall is answered.
export async function showDueRecall() {
    setAcceptClicks(false);
    await paintWhenSelected(getDueRecallProblem(), async (recall) => {
        // The candidate left its window (or its shard wouldn't load) between
        // the button's paint and this click: leave the board alone and let the
        // repaint settle the button's own state.
        if (!recall) {
            setAcceptClicks(true);
            updateAllUI();
            return;
        }
        await displayNextProblem(recall);
    });
}

// After a session resume (tab-visible pull) or a sync that merged foreign
// attempts, re-run the normal "what should be on screen" presentation: a due
// recall may have appeared — from the merged history, or from time alone
// while the tab was hidden — or the on-screen problem may have been solved on
// another device. The standing challengeProblem is kept by default — a sync
// can never let the user dodge a challenge that was selected for them; recalls
// merely delay it, and it stays the next challenge they face. The single
// exception: the user completed that exact problem on another client, so it
// now sits inside the recent-repeat window (challengeAlreadyAnswered) and is
// no longer legal to serve as a novel challenge — only then is it discarded
// and a fresh one selected. Only touches the board when a problem is sitting
// idle in SOLVING mode — never mid-answer sequence, in review/analysis, or
// over a linked (shared) problem — and an actual swap goes through the
// standard white fadeOverlay (with a short notice) so the board never changes
// abruptly under the user.
let refreshInFlight = false;
export async function refreshShownProblemAfterSync() {
    // Overlapping triggers (a merge-driven refresh racing the tab-visible one)
    // must not double-swap: the loser would rebuild the same recall at a fresh
    // random orientation and visibly rotate the board.
    if (refreshInFlight) return;
    refreshInFlight = true;
    try {
        const shown = state.currentProblem;
        if (!shown || shown.linked) return;
        if (state.appMode !== APP_MODES.SOLVING || !state.acceptClicks) return;

        if (!state.challengeProblem || challengeAlreadyAnswered()) await assignNewChallenge();

        // The gate applies only to INTRODUCING a recall over an idle novel — a
        // recall already on screen keeps its ungated re-selection (most-due may
        // have shifted, or it fell out of dueness → challenge), so a resume
        // roll can never visibly swap a served recall back out for a novel.
        // The roll itself is deterministic (seeded on candidate + last attempt,
        // see recallGateAllows), so a no-merge resume re-derives the serve
        // decision's own outcome — repeated task switching can't re-roll a
        // gated-out recall onto the board.
        const recall = (shown.isRepeat || recallGateAllows()) ? await getDueRecallProblem() : null;
        const next = recall || state.challengeProblem;
        if (!next) return;

        // Re-check after the awaits: a click that landed mid-selection started
        // the answer sequence, which owns the board now.
        if (state.currentProblem !== shown || state.appMode !== APP_MODES.SOLVING || !state.acceptClicks) return;

        // Selection landed on what's already on screen — leave it be (a repaint
        // would reset the solve timer and re-randomize a recall's orientation).
        if (next.filename === shown.filename) return;

        // Commit: block clicks for the overlay's pre-swap window, then swap
        // under a fade + notice (displayProblem re-enables clicks). GREY, not
        // the answer-result colors: white = correct, red = wrong, and this is
        // neither. On a display failure fall forward like every other
        // displayProblem caller: displayProblem mutates currentProblem before
        // the SGF parse can throw, so bailing would leave the board showing one
        // problem while state points at another — advanceToNextProblem retries
        // past the blacklisted problem and repaints something coherent.
        setAcceptClicks(false);
        await new Promise((resolve) => {
            fadeOverlay(1500, async () => {
                try { displayProblem(next, APP_MODES.SOLVING); }
                catch (error) { reportLoadError(error, 0); await advanceToNextProblem(); }
                finally { resolve(); }
            }, null, getTerm('problemUpdated'), { background: 'rgb(128, 128, 128)', text: 'white' });
        });
    } finally {
        refreshInFlight = false;
    }
}

// Re-shows the current problem (it's already loaded underneath) — used when
// leaving review (which painted the previous problem). The mode follows the
// current problem: a linked/shared problem returns to LINKED view, otherwise
// SOLVING. A null currentProblem is the normal post-answer review exit (the
// answered problem was cleared by completeCurrentProblem and no next problem
// was loaded underneath) — advance to a freshly selected problem, behind a
// grey loading fade if selection is slow.
export async function redisplayCurrentProblem() {
    if (!state.currentProblem) return advanceAfterReview();
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
    try { displayProblem(state.challengeProblem, APP_MODES.SOLVING); warnIfServedAnsweredChallenge(); }
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
    // The attempt is recorded — there is no problem to "return to" anymore.
    // Clear the on-screen slot (and with it the sessionStorage resume slot)
    // BEFORE the challenge reselection awaits: a reload landing anywhere in
    // the post-answer transition must fall through to a fresh advance at boot,
    // never resume — i.e. re-serve — the already-answered problem.
    setCurrentProblem(null);
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

// --- In-instance example review (stats Weaknesses pane) ---------------------
//
// Weakness-example links review a past problem inside the running instance —
// a spawned tab would run a second full app instance against the same
// localStorage. The example is staged as a TEMPORARY previousProblem so every analysis-mode
// consumer (rank label, motif bubble, share, ignore gating) coherently acts on
// it, and the swap is undone on exit so "Review" after closing stats still
// reviews the true previous problem. The context is in-memory only: an example
// review never survives a reload (the stats modal it returns to is gone too).
let exampleReview = null;

export function isExampleReviewActive() { return exampleReview !== null; }

// Stages `tsumegoId` for review: loads its shard (switching board size if the
// example lives on the other board — the on-screen problem is saved for the
// way back), swaps it into previousProblem, and remembers how to restore
// everything. Returns true when staged (the caller then enters analysis mode);
// false leaves all state as it was. `onExit` re-shows the hidden stats modal
// after a normal exit; `onAbandon` discards it after a side-door exit (see
// abandonExampleReviewIfActive).
export async function beginExampleReview(tsumegoId, { onExit, onAbandon } = {}) {
    // A second example opened from a re-opened stats modal (menu → Stats is
    // reachable from analysis mode) chains: unwind the prior swap, but keep
    // restoring to the board size / on-screen problem from BEFORE the first
    // example. The prior modal was already superseded by the one this click
    // came from, so its callbacks are dropped without firing.
    const prior = exampleReview;
    exampleReview = null;
    if (prior && state.boardSize === prior.exampleSize) {
        setPreviousProblem(prior.previousProblem);
    }
    const saved = prior
        ? { boardSize: prior.boardSize, currentProblem: prior.currentProblem, onExit, onAbandon }
        : { boardSize: state.boardSize, currentProblem: state.currentProblem, onExit, onAbandon };
    try {
        const lookupResult = await findZipIdForTsumegoId(tsumegoId);
        if (!lookupResult) throw new Error(`No tsumego with id ${tsumegoId}`);
        const { zipId, size } = lookupResult;
        if (size && size !== state.boardSize) setBoardSize(size);
        await loadZipMetadata();
        const zipFilename = state.zipMetadata.find(z => z.filename.startsWith(zipId))?.filename;
        if (!zipFilename) throw new Error(`Zip file for tsumego ID ${tsumegoId} not found`);
        await loadZipProblems(zipFilename);
        const raw = state.allProblems.find(p => p.filename === `${tsumegoId}.sgf`)
            || { filename: `${tsumegoId}.sgf`, zipFilename, rating: 0, rank: 0, quality: 0 };
        // linked: view-only — nothing about this detour is ever recorded.
        const problem = await buildProblem(raw, { isRepeat: false, linked: true, orientation: 0 });
        // previousProblem is per-size, so capture the bucket being clobbered
        // only now, after any size switch above.
        saved.previousProblem = state.previousProblem;
        saved.exampleSize = state.boardSize;
        setPreviousProblem(problem);
        exampleReview = saved;
        return true;
    } catch (error) {
        reportLoadError(error, 0);
        // Undo a size switch that happened before the failure.
        if (state.boardSize !== saved.boardSize) {
            setBoardSize(saved.boardSize);
            setCurrentProblem(saved.currentProblem);
        }
        return false;
    }
}

// Exits an example review: restores the swapped previousProblem (and board
// size / on-screen problem after a cross-size example), repaints what was on
// screen before the detour, then hands control back to the stats panel via
// onExit. Called from main.js's button1 analysis-exit branch.
export async function endExampleReview() {
    const saved = exampleReview;
    // Cleared before the repaint: displayProblem below must not read this as
    // a side-door exit (abandonExampleReviewIfActive).
    exampleReview = null;
    if (!saved) return;
    setPreviousProblem(saved.previousProblem);
    if (state.boardSize !== saved.boardSize) {
        setBoardSize(saved.boardSize);
        setCurrentProblem(saved.currentProblem);
    }
    await redisplayCurrentProblem();
    if (saved.onExit) saved.onExit();
}

// Any problem painted while an example review is staged means the user left
// through a side door (e.g. the always-available board-size menu item, which
// advances to a fresh problem) — drop the stale context so a later analysis
// exit can't restore a bygone stats panel, and let the owner discard its
// hidden modal. The previousProblem swap is unwound only when we're still on
// the example's board (the setter writes the active size's bucket); a
// side-door that already switched boards leaves the example as that size's
// reviewable previous — harmless, and replaced on the next completed problem.
function abandonExampleReviewIfActive() {
    const saved = exampleReview;
    if (!saved) return;
    exampleReview = null;
    if (state.boardSize === saved.exampleSize) {
        setPreviousProblem(saved.previousProblem);
    }
    if (saved.onAbandon) saved.onAbandon();
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