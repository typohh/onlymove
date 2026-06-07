// js/main.js
import * as Sentry from "@sentry/browser";
import { initDebugLog } from './debug-log.js';

// Capture console messages early so the hidden viewer (5 taps on the title)
// has the full session log, including any errors during init.
initDebugLog();

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, updateProblemHistory, setUserHasInteracted, updateAnswerHistory, setReviewStartTime, updateNextTsumego, setBoardSize, SUPPORTED_BOARD_SIZES, registerAcceptClicksHook, updateUserRank, getMergedAnswerHistory } from './state.js';
import { runMigrations } from './migrations.js';
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
import { loadSgf, redraw, initializeBoard } from './board.js';
import { getTerm } from './i18n.js';

import { initAudio } from './audio.js';

/**
 * Cleans the URL by removing the 'id' parameter, the size-hint path
 * (`/9` or `/13`), and 'index.html'. The path always collapses to '/'
 * so the deep-link doesn't stick in the address bar.
 */
function cleanUrl() {
    const url = new URL(window.location);
    url.searchParams.delete('id');
    let newPath = url.pathname.replace(/index\.html$/, '');
    if (/^\/(9|13)\/?$/.test(newPath)) {
        newPath = '/';
    }
    window.history.replaceState({}, '', newPath + url.search + url.hash);
}

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

/**
 * Custom Yes/Cancel confirmation dialog. Replaces the native blocking
 * `confirm()` (which is janky in an installed PWA and inconsistent with the
 * app's other custom overlays). Resolves to true on confirm, false on
 * cancel/backdrop. Reuses the shared #ios-install-overlay/#ios-install-modal
 * styling.
 */
function showConfirm(message) {
    return new Promise((resolve) => {
        let settled = false;
        const settle = (result, close) => {
            if (settled) return;
            settled = true;
            close();
            resolve(result);
        };
        showModal({
            onDismiss: () => settle(false, () => {}),
            build: (modal, close) => {
                modal.appendChild(modalTitle(message));
                modal.appendChild(modalButtonRow([
                    modalButton(getTerm('confirmNo'), () => settle(false, close)),
                    modalButton(getTerm('confirmYes'), () => settle(true, close)),
                ]));
            },
        });
    });
}
import { loadTsumego, loadTsumegoById, findZipIdForTsumegoId, loadZipMetadata, loadZipProblems } from './problem-loader.js';
import { backfillMissingEmbeddings } from './embedding-backfill.js';
import { loadRatingRankMap, computeRankDecomposition } from './rank-decomposition.js';
import { loadPopulationConstants } from './population-constants.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { initMotifBubble } from './motif-bubble.js';
import { showLoading, hideLoading, showModal, modalTitle, modalButton, modalButtonRow } from './overlays.js';
import { performShare } from './share.js';
import { APP_MODES, TOTAL_RANKS } from './constants.js';

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

document.addEventListener('DOMContentLoaded', async () => {
    hideBootSplash();

    // Log the hash of the main file
    const scriptElement = document.currentScript || document.querySelector('script[src*="main"]');
    if (scriptElement) {
        const scriptSrc = scriptElement.src;
        const hash = scriptSrc.split('.').slice(-2, -1)[0]; // Extract hash from main.[hash].js
        console.log(`Main file hash: ${hash}`);
    }

    updateBoardSize();
    // Using CSS vi units for responsive sizing, but still need to update WGo board on resize
    window.addEventListener('resize', () => {
        updateBoardSize();
    });

    const handleSizeToggle = async (next) => {
        // With only two supported sizes the menu item names the target size
        // ("Switch to 13x13"), so the menu path passes it explicitly and we
        // switch directly. Other callers (none currently) can omit it to get
        // the size picker.
        if (next === undefined) {
            next = await showBoardSizePicker({ dismissable: true });
        }
        if (next === null || next === state.boardSize) return;
        setBoardSize(next);
        // Linked mode exits on size switch: the linked challenge belongs to
        // the previous size's problem space.
        setAppMode(APP_MODES.SOLVING);
        showLoading();
        try {
            await loadTsumego(false);
        } catch (err) {
            console.error('Failed to load tsumego after size switch:', err);
        }
        if (state.problemHistory.length > 0 && state.problemHistory[0].linked) {
            setAppMode(APP_MODES.LINKED);
        }
        updateAllUI();
        hideLoading();
    };

    // Hamburger menu: open on click of either (portrait/landscape) trigger,
    // close on backdrop click. Menu items themselves close it before dispatching.
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
    // menu-about is a real anchor (target=_blank); let the browser navigate.
    document.getElementById('menu-about').addEventListener('click', () => {
        closeMenu();
    });
    document.getElementById('menu-stats').addEventListener('click', (e) => {
        e.preventDefault();
        closeMenu();
        showDetailedStats();
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

    // Review-mode motif bubble: wire its edge tab (slide over board / back).
    initMotifBubble();

    // Tapping the rank display shows how it translates to ranks on other
    // servers (showRankTranslation no-ops outside solving mode). The elements
    // are persistent, so this listener is registered once here rather than on
    // every updateUI repaint.
    ['text1', 'text1-h'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', () => showRankTranslation());
    });



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
            // In view mode, button1 exits back to the challenge tsumego
            if (state.nextTsumego) {
                const history = state.problemHistory;
                // Remove the linked problem from history — the challenge problem
                // is already at history[1] (it was pushed there when the linked
                // problem was loaded), so after the shift it becomes history[0].
                if (history.length > 0 && history[0].linked) {
                    history.shift();
                }
                updateProblemHistory(history);
                updateNextTsumego(null);
                setAppMode(APP_MODES.SOLVING);
                loadTsumego(false);
                updateAllUI();
            }
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
            if (state.nextTsumego) {
                setAppMode(APP_MODES.LINKED);
            } else {
                setAppMode(APP_MODES.SOLVING);
            }
            loadTsumego(false);
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
    // hamburger menu now; button4 has no analyzing-mode role.
    document.getElementById('button4').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.target.classList.contains('disabled')) return;
        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
            handleInstallClick();
        }
    });

    // Ignore is reachable from the analysis-mode hamburger menu.
    const performIgnore = async () => {
        if (!await showConfirm(getTerm('ignoreConfirm'))) return;
        const problem = state.problemHistory[1];
        if (!problem) return;

        const problemId = problem.filename.replace('.sgf', '');
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

        // Exit analysis mode and load next problem
        if (state.nextTsumego) {
            setAppMode(APP_MODES.LINKED);
        } else {
            setAppMode(APP_MODES.SOLVING);
        }
        loadTsumego(false);
        updateAllUI();
    };

    // Legacy answerHistory-shape migrations (see js/migrations.js). The
    // board-size bucket migrations already ran at state.js import time, so
    // these operate through state.answerHistory (the active size's view).
    runMigrations();

    // M (rating→rank) is REQUIRED: package_for_pwa ships only raw ratings, so
    // the client cannot rank a single problem (selection, UI, recall) without it.
    // It is now inlined into the bundle at build time, so it cannot fail to load
    // at runtime; the guard only trips on a malformed/empty package (a build-time
    // error). Bail loudly if so.
    const ratingRankMap = loadRatingRankMap();
    if (!ratingRankMap) {
        const msg = 'rating_to_rank.json (rating→rank map M) missing/empty in bundle; cannot continue.';
        try { Sentry.captureMessage(msg, { level: 'fatal' }); } catch (_) {}
        throw new Error(msg);
    }

    // Population constants (time/review bias + retention) are REQUIRED — they back
    // the recall scheduler and the rushing/review nudges, and the modules carry no
    // baked fallback. Also inlined at build time (same as M); apply before the
    // model trains or schedules. Bail loudly if the package is malformed/empty.
    const populationConstants = loadPopulationConstants();
    if (!populationConstants) {
        const msg = 'population_constants.json (model constants) missing/empty in bundle; cannot continue.';
        try { Sentry.captureMessage(msg, { level: 'fatal' }); } catch (_) {}
        throw new Error(msg);
    }

    // Fire-and-forget backfill: fills `embedding` and refreshes problemRating
    // (raw) + problemRank on legacy answerHistory entries from already-deployed
    // zips and the now-cached curve, so returning users don't need to solve 20+
    // new problems before the weakness model has data. Curve must be cached
    // first (above) so the rank refresh can run.
    backfillMissingEmbeddings();

    // One-time rank re-anchor for users whose momentum was corrupted by the a_p²
    // scalingFactor bug (see repairCorruptedMomentum_v3 in state.js). The inverted
    // rank update may have driven their front-page rank to an extreme, so re-anchor
    // it to the principled skill-decomposition estimate — the same quantity the
    // Rank tab shows — now that M and the merged history are available. Healthy
    // users never set the flag and are untouched; users with too little history
    // (decomposition returns null) keep their current rank and re-converge on play.
    if (localStorage.getItem('rankNeedsReanchor_v3')) {
        try {
            const decomp = computeRankDecomposition(getMergedAnswerHistory());
            const skill = decomp && decomp.summary ? decomp.summary.currentSkill : null;
            if (typeof skill === 'number' && Number.isFinite(skill)) {
                updateUserRank(Math.min(1, Math.max(0, skill / TOTAL_RANKS)));
            }
        } catch (e) {
            console.warn('rank re-anchor after momentum repair failed:', e);
        }
        localStorage.removeItem('rankNeedsReanchor_v3');
    }

    // Check for URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const tsumegoId = urlParams.get('id');
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

    // Resolve board size before any tsumego load. Order of precedence:
    //   1. ?id=X — let loadTsumegoById set the size from the lookup column.
    //   2. /9 or /13 path hint — set immediately, then strip the path.
    //   3. Persisted boardSize — already in state.
    //   4. Cold start — show the size picker and persist the choice.
    const pathSize = getBoardSizeFromPath();
    if (pathSize !== null && !tsumegoId) {
        setBoardSize(pathSize);
        // The URL path is also a persistent choice (a user who deep-linked
        // /13 expects the next plain visit to land on 13). Force-persist
        // because setBoardSize is a no-op when the value already matches.
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

    // Spinner over the initial problem load (fetching/parsing the zip archive)
    // so cold loads don't sit on a blank board. Lifted after the first paint.
    showLoading();
    try {
        if (tsumegoId && tsumegoId.length > 0) {
            const history = state.problemHistory;

            // Case 1: Review mode
            if (history.length > 1 && history[1].filename.startsWith(tsumegoId)) {
                setAppMode(APP_MODES.ANALYZING);
                enterAnalysisMode();
            }
            // Case 2: Refresh of existing problem (solving or linked)
            else if (history.length > 0 && history[0].filename.startsWith(tsumegoId)) {
                if (history[0].linked) {
                    setAppMode(APP_MODES.LINKED);
                } else {
                    setAppMode(APP_MODES.SOLVING);
                }
                await loadTsumego(false); // In both refresh cases, we just load what's at history[0]
            }
            // Case 3: Default - new linked problem. If this problem is in
            // the user's current recall window, route it as a recall solve
            // instead — randomize orientation so the user can't just
            // recognize the visual pattern. An explicit ?o=N URL parameter
            // still wins (the share-link contract).
            else {
                // The user arrived via a shared link — record the open with the
                // referrer (where the link was clicked from) for analytics.
                // Skip stats-panel example links (close_on_next=1): those are
                // internal, same-origin navigations and would inflate the
                // share_events 'open' count indistinguishably from real inbound
                // shared-link traffic.
                if (urlParams.get('close_on_next') !== '1') {
                    recordShareEvent({ type: 'open', problemId: tsumegoId, referrer: document.referrer || null });
                }
                const explicitOrientation = orientationParam !== null;
                const isRecall = getRecallCandidates().some(c => c.problemId === tsumegoId);
                const effectiveOrientation = (isRecall && !explicitOrientation) ? null : orientation;
                await loadTsumegoById(tsumegoId, effectiveOrientation, { asRecall: isRecall });
            }
        } else {
            // No ID in URL
            await loadTsumego(false);
            if (state.problemHistory.length > 0 && state.problemHistory[0].linked) {
                setAppMode(APP_MODES.LINKED);
            }
        }

    } catch (error) {
        console.error("Failed to load initial tsumego:", error);
    }

    cleanUrl();

    document.getElementById('title').textContent = getTerm('tsumegoMode');
    updateAllUI();
    hideLoading();

    // Initialize button4 state from the SW subscription status (async — first
    // paint above uses the default cache which renders Install; this repaints
    // once we know the real (installed, permission, subscribed) tuple).
    refreshNotifState().then(updateAllUI);
    // Push the current recall schedule to the Worker so the notification
    // cron has up-to-date enter/exit timestamps. No-op for users who haven't
    // opted in.
    maybeUploadSchedule();

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



    const resumeOverlay = document.getElementById('resume-overlay');
    const boardContainer = document.getElementById('board-container');

    /**
     * Check if board container has valid canvases and reinitialize if needed.
     * This addresses Android PWA canvas loss after task switching.
     * Returns true if board was recovered, false if reload is needed.
     */
    const recoverBoardIfNeeded = () => {
        // Solution #6: Force re-initialization to handle broken/white canvas context
        // We ignore whether canvas exists or not, because it might be technically present but visually broken
        
        if (state.kifuReader) {
            // Solution #2: Fully reinitialize the board instead of just redrawing
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
        
        // If no kifuReader, we can't really recover the board content
        return false;
    };

    // Recovery from killed session or bfcache
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

    // Solution #4: Resume overlay click triggers full board recreation for manual
    // recovery — same path as the pageshow/visibilitychange handlers.
    resumeOverlay.addEventListener('click', () => {
        resumeOverlay.className = 'resume-overlay-hidden';
        if (!recoverBoardIfNeeded()) location.reload();
    });

});
