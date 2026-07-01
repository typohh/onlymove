// js/main.js
import * as Sentry from "@sentry/browser";
import { initDebugLog } from './debug-log.js';

// Capture console messages early so the hidden viewer (5 taps on the title)
// has the full session log, including any errors during init.
initDebugLog();

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, setUserHasInteracted, updateAnswerHistory, setReviewStartTime, setBoardSize, SUPPORTED_BOARD_SIZES, registerAcceptClicksHook, getAnswerHistoryForSize, initHistoryStorage } from './state.js';
import { runMigrations, runTailMigration, dedupTailExact, maybeInitWindowRankOnce } from './migrations.js';
import { DATA_VERSION, RATINGS_VERSION } from './data-version.js';
import { maybeShowRankSurvey } from './rank-survey.js';
import { showRankTranslation } from './rank-translation.js';
registerAcceptClicksHook(maybeShowRankSurvey);

try {
    // Initialize Sentry
    Sentry.init({
        dsn: 'https://50b747ee522c1bc41c8243563ce7adc1@o4510091105665024.ingest.de.sentry.io/4510091110187088',
        // Tag every event with the build version (the same git-hash string the
        // Worker stamps on X-App-Version), so errors bucket by deploy. APP_VERSION
        // is substituted at build time by esbuild's `define`.
        release: APP_VERSION,
        // Drop third-party browser-extension noise we don't control and can't
        // fix — e.g. the Zalo extension injects a script that throws
        // "zaloJSV2 is not defined".
        ignoreErrors: [/zaloJSV2/],
    });
    window.Sentry = Sentry; // Make Sentry globally available
    Sentry.setUser({ id: state.userId });
} catch (error) {
    console.error("Sentry initialization failed:", error);
}

import { updateAllUI, updateBoardSize, showCaptureDetails, handleInstallClick, refreshNotifState } from './ui.js';
import { getRecallCandidates } from './model.js';
import { showDetailedStats } from './stats-ui.js';
import { maybeUploadSchedule } from './notifications.js';
import { recordShareEvent } from './user-stats.js';
import { redraw, initializeBoard } from './board.js';
import { tsumegoIdOf } from './sgf.js';
import { getTerm } from './i18n.js';

/**
 * Reads `/9` or `/13` from the pathname. Returns null if the path is anything
 * else. The caller decides whether to apply it as a size hint.
 */
function getBoardSizeFromPath() {
    const match = window.location.pathname.match(/^\/(9|13)\/?$/);
    if (!match) return null;
    const size = parseInt(match[1]);
    return SUPPORTED_BOARD_SIZES.includes(size) ? size : null;
}

/**
 * Cold-start picker shown to brand-new users (no `boardSize` in localStorage
 * and no URL hint). Returns a Promise that resolves to the chosen size.
 */
function showBoardSizePicker({ dismissable = false } = {}) {
    return new Promise((resolve) => {
        showModal({
            dismissable,
            onDismiss: () => resolve(null),
            build: (modal, close) => {
                modal.style.width = 'max-content';
                modal.appendChild(modalTitle(getTerm('boardSizePickerTitle')));
                modal.appendChild(modalButtonRow(
                    SUPPORTED_BOARD_SIZES.map((size) =>
                        modalButton(`${size}x${size}`, () => { close(); resolve(size); }))
                ));
            },
        });
    });
}

import { advanceToNextProblem, redisplayCurrentProblem, returnToChallenge, loadTsumegoById, resumeCurrentProblem } from './problem-loader.js';
import { backfillMissingEmbeddings } from './backfill.js';
import { loadRatingRankMap } from './rank-decomposition.js';
import { loadPopulationConstants } from './population-constants.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { initMotifBubble } from './motif-bubble.js';
import { showLoading, hideLoading, showModal, modalTitle, modalButton, modalButtonRow, showConfirm } from './overlays.js';
import { hasPendingAccountSwitch, maybeFinishAccountSwitch, startBackgroundSync } from './account-sync.js';
import { showAccountModal } from './account-ui.js';
import { maybeHandleRegistration } from './registration.js';
import { performShare } from './share.js';
import { APP_MODES } from './constants.js';

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    setDeferredInstallPrompt(e);
});

document.addEventListener('click', () => setUserHasInteracted(true));
document.addEventListener('touchstart', () => setUserHasInteracted(true));

/**
 * Drop the pure-CSS boot splash (index.html) now that the app's own JS is in
 * control. Called at the very top of the init handler: every subsequent screen
 * the user can see — the size picker, the showLoading() spinner over the first
 * zip load, or the loaded board — is rendered by JS within this same handler,
 * so the splash hands off without ever exposing the static placeholder layout.
 */
function hideBootSplash() {
    const el = document.getElementById('boot-splash');
    if (!el) return;
    el.classList.add('boot-hidden');
    setTimeout(() => el.remove(), 300);
}

// --- Init phases ------------------------------------------------------------
// The DOMContentLoaded handler at the bottom is the orchestrator; each step is
// a named function here so the boot sequence reads as a list rather than one
// ~500-line closure. handleSizeToggle / performIgnore / recoverBoardIfNeeded
// are module-level (they only touch imported state + helpers).

// Startup version log — three global identities, handy for confirming two devices
// are on the same build + same shipped data (the cross-device-divergence debug):
//   - the app:    the running main file's bundle hash (which deploy)
//   - the data:   data_version, the content hash that cache-busts the served data
//   - the rating: RATINGS_VERSION — ONE value. The rating model is a single joint
//     cross-board fit, so a re-rate changes both boards together; there is no
//     independent per-board version (data-version.js collapses the manifest's per-size
//     sub-hashes to this single string).
function logStartupVersions() {
    const scriptElement = document.currentScript || document.querySelector('script[src*="main"]');
    if (scriptElement) {
        const hash = scriptElement.src.split('.').slice(-2, -1)[0]; // main.[hash].js
        console.log(`Main file hash: ${hash}`);
    }
    console.log('Data version:', DATA_VERSION);
    console.log('Ratings version:', RATINGS_VERSION ?? 'unknown');
}

// Switch board size (the menu names the target, e.g. "Switch to 13x13", so it
// passes it explicitly; a no-arg call pops the picker). Linked mode exits — the
// linked challenge belongs to the previous size's problem space.
async function handleSizeToggle(next) {
    if (next === undefined) {
        next = await showBoardSizePicker({ dismissable: true });
    }
    if (next === null || next === state.boardSize) return;
    setBoardSize(next);
    setAppMode(APP_MODES.SOLVING);
    showLoading();
    try {
        await advanceToNextProblem();
    } catch (err) {
        console.error('Failed to load tsumego after size switch:', err);
    }
    updateAllUI();
    hideLoading();
}

// Ignore the just-reviewed problem (reachable from the analysis-mode hamburger
// menu): mark the matching answerHistory entry, then exit analysis and reveal
// the current problem (loaded underneath).
async function performIgnore() {
    if (!await showConfirm(getTerm('ignoreConfirm'))) return;
    const problem = state.previousProblem;
    if (!problem) return;

    const problemId = tsumegoIdOf(problem.filename);
    const answerHistory = [...state.answerHistory];
    const lastAnswer = answerHistory[answerHistory.length - 1];

    if (lastAnswer && lastAnswer.problemId === problemId) {
        lastAnswer.ignored = true;
        updateAnswerHistory(answerHistory);
    } else {
        Sentry.captureMessage("Ignore button: last answer does not match problem.", {
            extra: {
                lastAnswer,
                problemId
            }
        });
    }

    redisplayCurrentProblem();
    updateAllUI();
}

// Hamburger menu: open on either (portrait/landscape) trigger, close on
// backdrop click. Menu items close it before dispatching.
function wireMenu() {
    const menuOverlay = document.getElementById('menu-overlay');
    const openMenu = (e) => {
        if (e) e.preventDefault();
        // Re-paint before showing so labels reflect live state (board size +
        // pause label flip when the recall pause expires).
        updateAllUI();
        menuOverlay.classList.remove('menu-overlay-hidden');
    };
    const closeMenu = () => menuOverlay.classList.add('menu-overlay-hidden');
    document.getElementById('menu-toggle').addEventListener('click', openMenu);
    document.getElementById('menu-toggle-h').addEventListener('click', openMenu);
    menuOverlay.addEventListener('click', (e) => {
        if (e.target === menuOverlay) closeMenu();
    });

    document.getElementById('menu-board-size').addEventListener('click', (e) => {
        e.preventDefault();
        closeMenu();
        // Switch straight to the other size (the only alternative) instead of
        // popping a picker — the menu item already names the destination.
        const otherSize = SUPPORTED_BOARD_SIZES.find((s) => s !== state.boardSize);
        handleSizeToggle(otherSize);
    });
    document.getElementById('menu-pause-recalls').addEventListener('click', (e) => {
        e.preventDefault();
        const RECALL_PAUSE_MS = 2 * 60 * 60 * 1000;  // 2 hours
        if (state.recallPausedUntil > Date.now()) {
            state.recallPausedUntil = 0;  // resume immediately
        } else {
            state.recallPausedUntil = Date.now() + RECALL_PAUSE_MS;
        }
        closeMenu();
        updateAllUI();
    });
    document.getElementById('menu-audio').addEventListener('click', (e) => {
        e.preventDefault();
        state.audioEnabled = !state.audioEnabled;
        closeMenu();
        updateAllUI();
    });
    // menu-leaderboard is a real anchor to the standalone /leaderboard page
    // (target=_blank); let the browser navigate.
    document.getElementById('menu-leaderboard').addEventListener('click', () => {
        closeMenu();
    });
    // menu-about is a real anchor (target=_blank); let the browser navigate.
    document.getElementById('menu-about').addEventListener('click', () => {
        closeMenu();
    });
    document.getElementById('menu-stats').addEventListener('click', (e) => {
        e.preventDefault();
        closeMenu();
        showDetailedStats();
    });
    document.getElementById('menu-account').addEventListener('click', (e) => {
        e.preventDefault();
        closeMenu();
        showAccountModal();
    });
    document.getElementById('menu-share').addEventListener('click', (e) => {
        e.preventDefault();
        closeMenu();
        performShare();
    });
    document.getElementById('menu-ignore').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.currentTarget.classList.contains('disabled')) return;
        closeMenu();
        performIgnore();
    });
}

// Tapping the rank display shows how it translates to ranks on other servers
// (showRankTranslation no-ops outside solving mode). The elements are
// persistent, so this listener is registered once rather than on every repaint.
function wireRankDisplayTap() {
    ['text1', 'text1-h'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', () => showRankTranslation());
    });
}

// The four action buttons; behavior keys off state.appMode.
function wireActionButtons() {
    // Handle button1 click - behavior depends on current mode
    document.getElementById('button1').addEventListener('click', (e) => {
        e.preventDefault();
        // Stats-panel example links carry close_on_next=1 (see model.js's
        // populateExamplesList). Tapping button1 in that spawned window
        // closes it instead of continuing the session, so paging through
        // examples on an installed PWA doesn't accumulate Android instances.
        const closeOnNext = new URLSearchParams(window.location.search).get('close_on_next') === '1';
        if (closeOnNext) {
            window.close();
            return;
        }
        if (state.appMode === APP_MODES.LINKED) {
            // In view mode, button1 exits the shared problem back to the user's
            // standing challenge (which was never disturbed by the detour).
            returnToChallenge();
            updateAllUI();
        } else if (state.appMode === APP_MODES.SOLVING) {
            // acceptClicks is false during the post-solve transition (fadeOverlay
            // → loadTsumego). Without this guard, a fast Previous-click races
            // the in-flight history.unshift + loadSgf and lands the user in
            // ANALYZING mode on top of the next problem.
            if (!state.acceptClicks) return;
            // Lock immediately so a rapid second tap can't re-enter before the
            // transition this click starts has settled. enterAnalysisMode()
            // re-enables clicks once the review view (or its solving fallback)
            // is ready; the disabled/no-op path unlocks here.
            setAcceptClicks(false);
            if (!e.target.classList.contains('disabled')) {
                enterAnalysisMode();
            } else {
                setAcceptClicks(true);
            }
        } else if (state.appMode === APP_MODES.ANALYZING) {
            if (!state.acceptClicks) return;
            // Lock against a fast double-tap re-entering before loadTsumego
            // re-enables clicks once the next problem is actually ready.
            setAcceptClicks(false);
            if (state.reviewStartTime !== null) {
                const reviewTime = Date.now() - state.reviewStartTime;
                const answerHistory = [...state.answerHistory];
                const lastAnswer = answerHistory[answerHistory.length - 1];
                if (lastAnswer) {
                    lastAnswer.reviewTime += reviewTime;
                    updateAnswerHistory(answerHistory);
                }
                setReviewStartTime(null);
            }
            // Reveal the current problem (loaded underneath while reviewing the
            // previous one); redisplay sets the mode from currentProblem.linked.
            redisplayCurrentProblem();
            updateAllUI();
        }
    });

    // Handle button2 click - behavior depends on current mode.
    // In solving mode the anchor's href="about.html" target="_blank" handles
    // navigation natively (see ui.js); we deliberately don't preventDefault
    // because installed PWAs block window.open popups but honor native
    // target=_blank links. Only intercept in analysis mode for undo.
    document.getElementById('button2').addEventListener('click', (e) => {
        if (state.appMode === APP_MODES.ANALYZING) {
            e.preventDefault();
            undoLastMove();
        }
    });

    // Handle capture score click
    document.getElementById('capture-score').addEventListener('click', (e) => {
        e.preventDefault();
        showCaptureDetails();
    });
    document.getElementById('capture-score-h').addEventListener('click', (e) => {
        e.preventDefault();
        showCaptureDetails();
    });

    // Handle button4 click - Install (solving/linked). Ignore lives in the
    // hamburger menu; button4 has no analyzing-mode role.
    document.getElementById('button4').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.target.classList.contains('disabled')) return;
        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
            handleInstallClick();
        }
    });
}

// M (rating→rank) and the population constants are inlined into the bundle at
// build time and REQUIRED: package_for_pwa ships only raw ratings, so without M
// the client can't rank a single problem (selection, UI, recall), and the
// population constants back the recall scheduler and the rushing/review nudges
// with no baked fallback. Being inlined they can't fail to load at runtime —
// the guard only trips on a malformed/empty package (a build-time error), so
// bail loudly.
function loadRequiredAssetsOrThrow() {
    const ratingRankMap = loadRatingRankMap();
    if (!ratingRankMap) {
        const msg = 'rating_to_rank.json (rating→rank map M) missing/empty in bundle; cannot continue.';
        try { Sentry.captureMessage(msg, { level: 'fatal' }); } catch (_) {}
        throw new Error(msg);
    }
    const populationConstants = loadPopulationConstants();
    if (!populationConstants) {
        const msg = 'population_constants.json (model constants) missing/empty in bundle; cannot continue.';
        try { Sentry.captureMessage(msg, { level: 'fatal' }); } catch (_) {}
        throw new Error(msg);
    }
}

// A just-completed account switch (switchAccount wiped local state and reloaded)
// pulls the account's full server history NOW, before anything derives state
// from local history: the pulled-and-re-anchored rank drives the first problem
// selection, and the backfill below sees the rebuilt entries. Needs M (loaded
// above) for the rank refill. No-op when there's no pending switch.
async function maybeFinishPendingAccountSwitch() {
    if (!hasPendingAccountSwitch()) return;
    // The full-history pull + shard field fill + retrain can take on the order
    // of a minute; say so instead of showing a bare spinner.
    showLoading(getTerm('accountRestoring'));
    let switchResult = null;
    try {
        switchResult = await maybeFinishAccountSwitch();
    } catch (e) {
        console.error('account switch pull failed:', e);
    } finally {
        hideLoading();
    }
    // Adoption counts against the daily full-history cap; if it was exceeded the
    // device adopted the account but pulled no history (it fills via incremental
    // sync once the cap resets). Tell the user rather than show a silently empty
    // profile.
    if (switchResult && switchResult.capExceeded) {
        alert(getTerm('accountHistoryCapped'));
    }
}

// Parse the URL's intent: a shared problem id, the recall-notification flag, and
// an optional explicit orientation (0..7). A bare id defaults orientation to 0.
function parseUrlIntent() {
    const urlParams = new URLSearchParams(window.location.search);
    const tsumegoId = urlParams.get('id');
    // Recall-notification clicks land on /?recall=1 — an intent, not a problem
    // id. The client picks which recall problem to serve from its own scheduler
    // state, which is fresher than the Worker's snapshot at push time.
    const wantsRecall = urlParams.get('recall') === '1';
    const orientationParam = urlParams.get('o');
    let orientation = null;
    if (orientationParam !== null) {
        const parsed = parseInt(orientationParam);
        if (!isNaN(parsed) && parsed >= 0 && parsed <= 7) {
            orientation = parsed;
        }
    } else if (tsumegoId) {
        // If id is present but o is missing, default to 0 (user request)
        orientation = 0;
    }
    return { urlParams, tsumegoId, wantsRecall, orientation, orientationParam };
}

// Resolve board size before any tsumego load. Order of precedence:
//   1. ?id=X — let loadTsumegoById set the size from the lookup column.
//   2. /9 or /13 path hint — set immediately (also persisted: a user who
//      deep-linked /13 expects the next plain visit to land on 13).
//   3. Persisted boardSize — already in state.
//   4. Cold start — show the size picker and persist the choice.
async function resolveInitialBoardSize(tsumegoId) {
    const pathSize = getBoardSizeFromPath();
    if (pathSize !== null && !tsumegoId) {
        setBoardSize(pathSize);
        // Force-persist because setBoardSize is a no-op when the value matches.
        localStorage.setItem('boardSize', String(pathSize));
    }

    if (!tsumegoId && pathSize === null && !localStorage.getItem('boardSize')) {
        // Render an empty board behind the picker so the dialog overlays a
        // real-looking board instead of empty void.
        initializeBoard(state.boardSize);
        const chosen = await showBoardSizePicker();
        setBoardSize(chosen);
        // Same reason as above — force-persist when chosen equals the default.
        localStorage.setItem('boardSize', String(chosen));
    }
}

// Load the first problem under the initial spinner: a shared link (?id), a
// recall-notification intent (?recall=1), or the standing challenge / due
// recall on a plain open. hideLoading() is left to the caller (after first
// paint). The URL is cleaned by displayProblem (only for non-linked problems, so
// a shared problem's deep-link survives a reload) — see cleanUrl in problem-loader.
async function loadInitialProblem({ urlParams, tsumegoId, wantsRecall, orientation, orientationParam }) {
    // Spinner over the initial problem load (fetching/parsing the zip archive)
    // so cold loads don't sit on a blank board. Lifted after the first paint.
    showLoading();
    try {
        if (tsumegoId && tsumegoId.length > 0) {
            // Arrived via a shared link. If this problem is in the user's current
            // recall window, route it as a recall solve (randomize orientation so
            // the pattern can't be recognized); otherwise it's a view-only linked
            // problem. Either way the standing challenge is untouched and resumes
            // when the user exits (returnToChallenge). An explicit ?o=N still wins
            // (the share-link contract). Record the open with referrer for
            // analytics — but skip internal stats-panel example links
            // (close_on_next=1) so they don't inflate the share 'open' count.
            if (urlParams.get('close_on_next') !== '1') {
                recordShareEvent({ type: 'open', problemId: tsumegoId, referrer: document.referrer || null });
            }
            const explicitOrientation = orientationParam !== null;
            const isRecall = getRecallCandidates().some(c => c.problemId === tsumegoId);
            const effectiveOrientation = (isRecall && !explicitOrientation) ? null : orientation;
            const loaded = await loadTsumegoById(tsumegoId, effectiveOrientation, { asRecall: isRecall });
            if (!loaded) await advanceToNextProblem();
        } else if (wantsRecall) {
            // Notification click: pick the due recall ourselves and show it via
            // loadTsumegoById's asRecall path (the standing challenge stays
            // assigned and returns once recalls are exhausted). Prefer the active
            // board; recalls can only be re-shown on their own board, so if the
            // active size has none due, take one from the other size (the uploaded
            // schedule spans both) and let loadTsumegoById switch boards via the
            // lookup column. If nothing is due anywhere (already reviewed since the
            // push, or recall paused), fall back to the normal challenge/recall flow.
            let candidates = getRecallCandidates();
            if (candidates.length === 0) {
                for (const size of SUPPORTED_BOARD_SIZES) {
                    if (size === state.boardSize) continue;
                    candidates = getRecallCandidates({ history: getAnswerHistoryForSize(size) });
                    if (candidates.length > 0) break;
                }
            }
            let loaded = false;
            if (candidates.length > 0) {
                candidates.sort((a, b) => a.recallRate - b.recallRate);
                loaded = await loadTsumegoById(candidates[0].problemId, null, { asRecall: true });
            }
            if (!loaded) await advanceToNextProblem();
        } else {
            // Plain open (no URL id, no notification). On a same-session reload
            // (most importantly the version-driven graceful reload) restore the
            // exact problem that was on screen at its stored orientation, so a
            // recall doesn't visibly rotate. On a genuine fresh session there's no
            // resume slot and we fall through to the normal due-recall / standing-
            // challenge flow.
            if (!await resumeCurrentProblem()) await advanceToNextProblem();
        }
    } catch (error) {
        console.error("Failed to load initial tsumego:", error);
    }
}

// Background services started after the first paint: notif-state repaint
// (button4), schedule upload, cross-device sync pull, and the schedule
// heartbeat that keeps the cron's active-user suppression watermark fresh.
function startBackgroundServices() {
    // Initialize button4 state from the SW subscription status (async — first
    // paint uses the default cache which renders Install; this repaints once we
    // know the real (installed, permission, subscribed) tuple).
    refreshNotifState().then(updateAllUI);
    // Push the current recall schedule to the Worker so the notification cron
    // has up-to-date enter/exit timestamps. No-op for users who haven't opted in.
    maybeUploadSchedule();

    // Account sync: pull attempts made on other devices under this userId — on
    // load (now) and when the tab becomes visible. During active play the
    // per-report piggyback signal (checkSyncSignal) covers it with no polling.
    startBackgroundSync();

    // Heartbeat: keep the server's schedule — and its updated_at watermark,
    // which the cron uses to suppress reminders for active users — fresh during
    // a long session without re-uploading on every attempt. uploadSchedule()
    // self-throttles and skips when the (hourly-resolution) schedule is
    // unchanged, so this writes at most ~twice an hour and only when something
    // actually moved. Gated to a visible tab: a backgrounded tab shouldn't keep
    // marking the user "active" (let suppression lapse so they can be reminded).
    setInterval(() => {
        if (document.visibilityState === 'visible') maybeUploadSchedule();
    }, 30 * 60 * 1000);
}

/**
 * Check if the board container has valid canvases and reinitialize if needed.
 * Addresses Android PWA canvas loss after task switching. Returns true if the
 * board was recovered, false if a reload is needed.
 */
function recoverBoardIfNeeded() {
    // Force re-initialization to handle a broken/white canvas context: ignore
    // whether the canvas exists, because it might be technically present but
    // visually broken.
    if (state.kifuReader) {
        // Fully reinitialize the board instead of just redrawing.
        try {
            const size = state.kifuReader.game.position.size;
            initializeBoard(size);
            redraw();
            return true;
        } catch (e) {
            Sentry.captureMessage('Board recovery failed: ' + e.message, 'error');
            return false;
        }
    }
    // If no kifuReader, we can't really recover the board content.
    return false;
}

// Recovery from a killed session or bfcache: re-init the board on bfcache
// restore / tab re-show / manual resume-overlay click; reload if that fails.
function wireLifecycleRecovery() {
    const resumeOverlay = document.getElementById('resume-overlay');

    window.addEventListener('pageshow', (event) => {
        if (event.persisted) {
            // From bfcache, attempt recovery
            if (!recoverBoardIfNeeded()) {
                location.reload();
            }
        }
    });

    window.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            resumeOverlay.className = 'resume-overlay';
            // The user is leaving — persist the freshest schedule now (keepalive
            // lets it land through teardown). Skips if unchanged since the last
            // upload; this is the main replacement for the per-attempt upload.
            maybeUploadSchedule();
        } else if (document.visibilityState === 'visible') {
            resumeOverlay.className = 'resume-overlay-hidden';

            requestAnimationFrame(() => {
                if (!recoverBoardIfNeeded()) {
                    Sentry.captureMessage('Canvas recovery failed on visibilitychange, reloading...', 'error');
                    location.reload();
                }
            });
        }
    });

    // Resume overlay click triggers full board recreation for manual recovery —
    // same path as the pageshow/visibilitychange handlers.
    resumeOverlay.addEventListener('click', () => {
        resumeOverlay.className = 'resume-overlay-hidden';
        if (!recoverBoardIfNeeded()) location.reload();
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    hideBootSplash();
    logStartupVersions();

    updateBoardSize();
    // Using CSS vi units for responsive sizing, but still need to update the WGo
    // board on resize.
    window.addEventListener('resize', () => updateBoardSize());

    // Static UI wiring (persistent elements, registered once).
    wireMenu();
    initMotifBubble();
    wireRankDisplayTap();
    wireActionButtons();

    // Legacy answerHistory-shape migrations (js/migrations.js). The board-size
    // bucket migrations already ran at state.js import time, so these operate
    // through state.answerHistory (the active size's view).
    runMigrations();

    // Normalize answer-history storage to the unified working set + IndexedDB
    // tail (history-store.js). Runs AFTER runMigrations so the legacy {9,13}
    // buckets are migrated in full first, then everything beyond the cap is
    // evicted to IDB — freeing localStorage before the embedding backfill below
    // writes through the (now bounded) working set. Awaited so a quota-stressed
    // device is relieved before any further write; cheap on already-split state.
    await initHistoryStorage();

    loadRequiredAssetsOrThrow();

    await maybeFinishPendingAccountSwitch();

    // Fire-and-forget backfill: fills `embedding` and refreshes problemRating
    // (raw) + problemRank on legacy answerHistory entries from already-deployed
    // zips and the now-cached curve, so returning users don't need to solve 20+
    // new problems before the weakness model has data. The one-time cold-tail
    // migration (de-dup + reconstruct + eviction-baseline fix) settles the tail
    // FIRST so the backfill's curve-recompute doesn't race its writes. Both are
    // background — the first problem load doesn't wait.
    runTailMigration()
        .catch(() => {})
        .then(() => dedupTailExact())
        .catch(() => {})
        .then(() => backfillMissingEmbeddings());

    maybeInitWindowRankOnce();

    // Resolve board size, then load the first problem (shared link / recall
    // notification / standing challenge).
    const intent = parseUrlIntent();
    await resolveInitialBoardSize(intent.tsumegoId);
    await loadInitialProblem(intent);

    document.getElementById('title').textContent = getTerm('tsumegoMode');
    updateAllUI();
    hideLoading();

    startBackgroundServices();
    wireLifecycleRecovery();

    // EGD-leaderboard registration: if the PWA booted on an invite link
    // (/register?t=<token>), preview the invitation and offer to bind this
    // account. Fire-and-forget after boot so it overlays a painted board;
    // gated on ?t= so normal opens are entirely unaffected.
    if (new URLSearchParams(window.location.search).has('t')) {
        maybeHandleRegistration();
    }
});
