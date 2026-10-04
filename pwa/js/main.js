// js/main.js
import { captureMessage, init, setUser } from "@sentry/browser";
import { initDebugLog } from './debug-log.js';

// Capture console messages early so the hidden viewer (5 taps on the title)
// has the full session log, including any errors during init.
initDebugLog();

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, setUserHasInteracted, updateAnswerHistory, setReviewStartTime, setBoardSize, SUPPORTED_BOARD_SIZES, registerAcceptClicksHook, getAnswerHistoryForSize, getMergedAnswerHistory, initHistoryStorage, addIgnoredProblem } from './state.js';
import { DATA_VERSION, RATINGS_VERSION } from './data-version.js';
import { maybeShowRankSurvey } from './rank-survey-prompt.js';
import { rankTranslationContent } from './rank-translation.js';
registerAcceptClicksHook(maybeShowRankSurvey);

try {
    // Initialize Sentry
    init({
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
    // The console global is built from the named imports on purpose: a
    // namespace import assigned here keeps every integration in the bundle.
    window.Sentry = { captureMessage, setUser };
    setUser({ id: state.userId });
} catch (error) {
    console.error("Sentry initialization failed:", error);
}

import { updateAllUI, updateBoardSize, captureDetailsText, handleInstallClick, refreshNotifState, initCornerMark } from './ui.js';
import { getRecallCandidates, calculateAndStoreStatsResults } from './recall-model.js';
import { showDetailedStats } from './stats-ui.js';
import { maybeUploadSchedule, syncRecallNotification } from './notifications.js';
import { recordShareEvent } from './attempt-recorder.js';
import { redraw, initializeBoard } from './board.js';
import { tsumegoIdOf } from './sgf.js';
import { getTerm } from './i18n.js';
import { hintAnchor } from './hint-bubble.js';
import { initOnboarding, onResume as onboardingResume, onPreviousClicked as onboardingPreviousClicked } from './onboarding.js';

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
 * Render a picker pitch string into `el`. The strings carry two marks:
 * `**text**` for the subject of each paragraph (games, tsumego, OnlyMove) and
 * `*text*` for the contrasting "but". Built as text
 * nodes and spans, never innerHTML, so a translation cannot inject markup.
 */
function appendPitchText(el, text) {
    for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/)) {
        if (!part) continue;
        if (part.startsWith('**')) {
            const b = document.createElement('strong');
            b.textContent = part.slice(2, -2);
            el.appendChild(b);
        } else if (part.startsWith('*')) {
            const em = document.createElement('em');
            em.textContent = part.slice(1, -1);
            el.appendChild(em);
        } else {
            el.appendChild(document.createTextNode(part));
        }
    }
}

/**
 * Cold-start picker shown to brand-new users (no `boardSize` in localStorage
 * and no URL hint). Returns a Promise that resolves to the chosen size.
 */
function showBoardSizePicker({ dismissable = false, showIntro = false } = {}) {
    return new Promise((resolve) => {
        showModal({
            dismissable,
            onDismiss: () => resolve(null),
            build: (modal, close) => {
                modal.style.width = 'max-content';
                // The picker renders as stacked panels rather than the shared
                // single-panel modal, so the dialog's own chrome is dropped and
                // each panel supplies its surface (`.board-size-picker`, style.css).
                modal.classList.add('board-size-picker');

                // Top panel, first-time-only: who we are and why whole-board
                // positions at the right difficulty beat games or a tsumego
                // collection. A returning user switching sizes already knows
                // the site. Our answer leads in its own tinted block; games and
                // tsumego follow as the muted paragraphs it answers.
                if (showIntro) {
                    const pitch = document.createElement('div');
                    pitch.className = 'picker-pitch';
                    const brand = document.createElement('div');
                    brand.className = 'picker-brand';
                    brand.textContent = getTerm('boardSizePickerBrand');
                    pitch.appendChild(brand);
                    for (const key of ['boardSizePickerOurs', 'boardSizePickerGames', 'boardSizePickerTsumego']) {
                        const body = document.createElement('div');
                        body.className = key === 'boardSizePickerOurs' ? 'modal-body picker-ours' : 'modal-body';
                        appendPitchText(body, getTerm(key));
                        pitch.appendChild(body);
                    }
                    modal.appendChild(pitch);
                }

                // Bottom panel: the question and its two answers.
                const question = document.createElement('div');
                question.className = 'picker-question';
                question.appendChild(modalTitle(getTerm('boardSizePickerTitle')));
                question.appendChild(modalButtonRow(
                    SUPPORTED_BOARD_SIZES.map((size) =>
                        modalButton(`${size}x${size}`, () => { close(); resolve(size); }))
                ));
                modal.appendChild(question);
            },
        });
    });
}

import { advanceToNextProblem, redisplayCurrentProblem, returnToChallenge, loadTsumegoById, resumeCurrentProblem, isExampleReviewActive, endExampleReview, showDueRecall } from './problem-loader.js';
import { backfillMissingEmbeddings, stampVersionsForFreshHistory } from './backfill.js';
import { maybeShowRatingsCorrectionNotice } from './corrections-notice.js';
import { loadRatingRankMap, reanchorRank, flattenAttempts } from './rank-decomposition.js';
import { communityYouFragment } from './stats-compute.js';
import { loadPopulationConstants } from './population-constants.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { initMotifBubble } from './motif-bubble.js';
import { initCoordInput } from './coord-input.js';
import { showLoading, hideLoading, showModal, modalTitle, modalButton, modalButtonRow, showConfirm } from './overlays.js';
import { hasPendingAccountSwitch, pendingAccountSwitchKind, maybeFinishAccountSwitch, consumeAccountSwitchFailure, startBackgroundSync, syncedAttemptTotal } from './account-sync.js';
import { showAccountModal } from './account-ui.js';
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
 * control and the initial board UI has rendered. Guarded so multiple calls
 * (e.g. cold-start size picker handoff vs standard problem load) are safe.
 */
// The bundle has run: disarm index.html's bundle-failure fallback, which would
// otherwise drop the splash at a fixed 15s over a boot that is merely slow.
// From here the splash is lifted by hideBootSplash() alone.
clearTimeout(window.__bootSplashTimer);

function hideBootSplash() {
    const el = document.getElementById('boot-splash');
    if (!el || el.classList.contains('boot-hidden')) return;
    el.classList.add('boot-hidden');
    setTimeout(() => el.remove(), 400);
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
// menu): mark the matching answerHistory entry, then exit analysis — back to
// the untouched problem for a voluntary review, or on to a freshly selected
// one after a wrong answer (see redisplayCurrentProblem).
async function performIgnore() {
    if (!await showConfirm(getTerm('ignoreConfirm'))) return;
    const problem = state.previousProblem;
    if (!problem) return;

    const problemId = tsumegoIdOf(problem.filename);
    const answerHistory = [...state.answerHistory];
    const lastAnswer = answerHistory[answerHistory.length - 1];

    if (lastAnswer && lastAnswer.problemId === problemId) {
        lastAnswer.ignored = true;
        // Also record it in the durable id set — the entry flag round-trips
        // nowhere, so this is what survives an epoch wipe / re-adoption
        // (state.applyStoredIgnoredFlags re-marks the restored history).
        addIgnoredProblem(problemId);
        updateAnswerHistory(answerHistory);
    } else {
        captureMessage("Ignore button: last answer does not match problem.", {
            extra: {
                lastAnswer,
                problemId
            }
        });
    }

    redisplayCurrentProblem();
    updateAllUI();
}

// The role="button" elements that aren't native <button>s don't activate from
// the keyboard on their own: the capture-score <div>s respond to no key at
// all, and <a role="button">s respond to Enter but not Space. Delegate the
// missing keys to a click so every role button is keyboard-operable.
function wireRoleButtonKeys() {
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const el = e.target instanceof Element ? e.target.closest('[role="button"]') : null;
        if (!el || el.tagName === 'BUTTON') return;
        // Enter on an <a href> already clicks natively — don't double-fire.
        if (e.key === 'Enter' && el.tagName === 'A') return;
        e.preventDefault();
        el.click();
    });
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
    const menuToggle = document.getElementById('menu-toggle');
    const menuToggleH = document.getElementById('menu-toggle-h');
    menuToggle.addEventListener('click', openMenu);
    menuToggleH.addEventListener('click', openMenu);
    // Localize the two static menu-toggle labels (the vertical toggle is the ☰
    // glyph with an aria-label; the horizontal one shows the word). index.html
    // ships English defaults; everything else the app sets goes through getTerm.
    menuToggle.setAttribute('aria-label', getTerm('menu'));
    menuToggleH.textContent = getTerm('menu');
    menuOverlay.addEventListener('click', (e) => {
        if (e.target === menuOverlay) closeMenu();
    });
    // Keyboard equivalent of the backdrop click.
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !menuOverlay.classList.contains('menu-overlay-hidden')) closeMenu();
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
    // menu-statistics is a real anchor to the standalone /statistics page
    // (target=_blank). Its fragment carries the reader's own bins for that page's
    // "you are here" markers, written here from the local working-set history at
    // click time — the browser reads the href after the listeners run, and a
    // fragment never reaches the server. The rank marker is placed from the rank
    // the user is actually SHOWN and the count of first-time attempts backing it,
    // the same two quantities the Worker gates and stores per player, so the
    // marker lands in the frame the bars are drawn in (computeUserCommunityBuckets).
    document.getElementById('menu-statistics').addEventListener('click', (e) => {
        const merged = getMergedAnswerHistory();
        const frag = communityYouFragment(merged, {
            shownRank: state.displayRank,
            usableAttempts: flattenAttempts(merged).length,
            totalAttempts: syncedAttemptTotal(),
        });
        e.currentTarget.href = '/statistics' + (frag ? '#' + frag : '');
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
    document.getElementById('menu-ignore').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.currentTarget.classList.contains('disabled')) return;
        closeMenu();
        performIgnore();
    });
}

// The rank badge and the capture score are READINGS of themselves: pointing at
// either says how the number shown is arrived at — the rank on each surveyed
// server, the komi and captures behind the score. Both are hint anchors
// (js/hint-bubble.js), so they hover open on a mouse, tap open on a touchscreen,
// and dismiss themselves. Each resolver declines outside the mode its reading
// belongs to, which leaves the control inert there; the elements are persistent,
// so this runs once rather than on every repaint.
function wireHeaderReadingHints() {
    for (const id of ['text1', 'text1-h']) {
        const el = document.getElementById(id);
        if (el) hintAnchor(el, rankTranslationContent);
    }
    for (const id of ['capture-score', 'capture-score-h']) {
        const el = document.getElementById(id);
        if (el) hintAnchor(el, captureDetailsText);
    }
}

// Nothing in the app is dragged, so no drag may START here and no drop may
// LAND here. The stylesheet's user-drag rule already stops anchors and images;
// this also stops a dragged text selection, which that rule does not cover.
// The reason is the freeze a stray drag causes: once a native drag session
// starts, Blink discards every mouse event until the compositor ends it, and
// on Wayland Chrome a drag begun from a press that has already released never
// ends — the window stops responding entirely. A file dropped on the window
// would otherwise NAVIGATE the installed app to that file.
function wireNoDrag() {
    document.addEventListener('dragstart', (e) => e.preventDefault());
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// The four action buttons; behavior keys off state.appMode.
function wireActionButtons() {
    // Handle button1 click - behavior depends on current mode
    document.getElementById('button1').addEventListener('click', (e) => {
        e.preventDefault();
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
                // First-run onboarding: the user used Previous themselves, so
                // the bubble teaching it is retired unshown.
                onboardingPreviousClicked();
                enterAnalysisMode();
            } else {
                setAcceptClicks(true);
            }
        } else if (state.appMode === APP_MODES.ANALYZING) {
            if (!state.acceptClicks) return;
            // Lock against a fast double-tap re-entering before loadTsumego
            // re-enables clicks once the next problem is actually ready.
            setAcceptClicks(false);
            // A stats-panel example review is unrecorded (the example's attempt
            // is an old history entry, not the last answer, so the review-time
            // attribution below would credit the wrong problem): drop the
            // timer, restore the pre-detour state, and re-show the hidden
            // stats modal over the untouched problem.
            if (isExampleReviewActive()) {
                setReviewStartTime(null);
                endExampleReview();
                return;
            }
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
            // Leave review: a voluntary review (entered from solving) still has
            // the untouched problem loaded underneath and redisplays it; after a
            // wrong answer there is none (currentProblem was cleared when the
            // attempt completed), so redisplay advances to a freshly selected
            // problem instead. The mode follows currentProblem.linked.
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

    // Handle button4 click - Install (solving/linked). Ignore lives in the
    // hamburger menu; button4 has no analyzing-mode role.
    document.getElementById('button4').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.target.classList.contains('disabled')) return;
        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
            handleInstallClick();
        }
    });

    // Share button — same action as the hamburger menu's Share item; shown in
    // analysis and view mode (Recall takes the slot while solving).
    document.getElementById('button-share').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.target.classList.contains('disabled')) return;
        performShare();
    });

    // Recall button — brings the most-due recall onto the board. currentTarget,
    // not target: the click can land on the label or the badge span.
    document.getElementById('button-recall').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.currentTarget.classList.contains('disabled')) return;
        // The button is repainted on updateAllUI, so a mid-answer-sequence
        // click can reach here with the board already locked — swapping the
        // problem out from under it would strand the attempt.
        if (!state.acceptClicks) return;
        showDueRecall();
    });
}

// M (rating→rank) and the population constants are inlined into the bundle at
// build time and REQUIRED: package_for_pwa ships only raw ratings, so without M
// the client can't rank a single problem (selection, UI, recall), and the
// population constants back the recall scheduler and the pace readings
// with no baked fallback. Being inlined they can't fail to load at runtime —
// the guard only trips on a malformed/empty package (a build-time error), so
// bail loudly.
function loadRequiredAssetsOrThrow() {
    const ratingRankMap = loadRatingRankMap();
    if (!ratingRankMap) {
        const msg = 'rating_to_rank.json (rating→rank map M) missing/empty in bundle; cannot continue.';
        try { captureMessage(msg, { level: 'fatal' }); } catch (_) {}
        throw new Error(msg);
    }
    const populationConstants = loadPopulationConstants();
    if (!populationConstants) {
        const msg = 'population_constants.json (model constants) missing/empty in bundle; cannot continue.';
        try { captureMessage(msg, { level: 'fatal' }); } catch (_) {}
        throw new Error(msg);
    }
}

// A just-completed account switch (switchAccount wiped local state and reloaded)
// pulls the account's full server history NOW, before anything derives state
// from local history: the pulled-and-re-anchored rank drives the first problem
// selection, and the backfill below sees the rebuilt entries. Needs M (loaded
// above) for the rank refill. No-op when there's no pending switch.
// A FAILED adoption (daily cap / network) never half-adopts: account-sync falls
// back to a fresh account id and reloads, leaving a notice we alert here on the
// post-fallback boot.
async function maybeFinishPendingAccountSwitch() {
    const failReason = consumeAccountSwitchFailure();
    if (failReason) {
        alert(getTerm(failReason === 'cap' ? 'accountAdoptFailedCap' : 'accountAdoptFailedNetwork'));
    }
    if (!hasPendingAccountSwitch()) return;
    // The full-history pull + shard field fill + retrain can take on the order
    // of a minute; say so instead of showing a bare spinner. Only a code-entry
    // adoption is a login — the epoch gate re-adopts this device's own account,
    // where "logging in" would describe an account change that isn't happening.
    const isLogin = pendingAccountSwitchKind() === 'switch';
    showLoading(getTerm(isLogin ? 'accountRestoring' : 'historyRestoring'));
    try {
        await maybeFinishAccountSwitch();
    } catch (e) {
        console.error('account switch pull failed:', e);
    } finally {
        hideLoading();
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
    return { tsumegoId, wantsRecall, orientation, orientationParam };
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
        hideBootSplash();
        const chosen = await showBoardSizePicker({ showIntro: true });
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
async function loadInitialProblem({ tsumegoId, wantsRecall, orientation, orientationParam }) {
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
            // analytics.
            recordShareEvent({ type: 'open', problemId: tsumegoId, referrer: document.referrer || null });
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
    // Bring a reminder still sitting in the tray up to date with the backlog as
    // it stands now — it was rendered when the push was sent, and another device
    // may have worked through some of it since.
    syncRecallNotification();

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
            captureMessage('Board recovery failed: ' + e.message, 'error');
            return false;
        }
    }
    // If no kifuReader, we can't really recover the board content.
    return false;
}

// Recovery from a killed session or bfcache: re-init the board on bfcache
// restore, on tab re-show, and on a resume that delivers no visible event at
// all — the Android task-switch case this handler exists for.
//
// THE NO-EVENT RESUME IS DETECTED BY A FRAME, AND NOTHING IS PUT ON SCREEN ON
// THE WAY OUT. rAF callbacks do not run for a page that is not being painted,
// so one requested while hidden fires exactly when the app is composited again
// — no dependence on the visibility IPC arriving, and none on when Android
// decided to pause the activity. The rejected alternative is arming a
// full-screen "Click to continue" overlay while hidden, immediately or behind a
// delay: either way it sits in the committed layer tree for the whole
// background period, so the resume is free to paint it, while removing it again
// is JS racing the first composited frame of a thawing renderer. A frozen
// process loses that race by construction — the arm timer it could not run in
// the background runs at thaw instead — which is why the delay did not help and
// the overlay met the user on nearly every task switch. The frame probe repairs
// the board directly, so there is nothing to tap and nothing to race.
function wireLifecycleRecovery() {
    // Rebuild the board once the pending frame has PAINTED. A rAF callback runs
    // before its own frame's paint, so recovering inside the first one holds
    // back the very frame the resume is waiting on — a full initializeBoard()
    // (fresh canvases, wood background, redraw) on a still-thawing renderer.
    // The inner rAF runs after that frame has been painted; the cost is one
    // frame of the possibly-broken board before it is replaced.
    //
    // One rebuild per background round, whichever signal gets there first: an
    // ordinary resume delivers both the visibility event and the frame, and the
    // second rebuild would only redo what the first just did. The guard sits on
    // the REBUILD rather than on entry, so a caller whose frames never arrive
    // cannot claim the round and leave the board unrepaired — both paths queue
    // against the same frames, so whichever one a frame reaches first does the
    // work and the other finds it done.
    let recoveredThisResume = false;
    let sawVisibleEvent = false;
    const recoverOnce = (source) => {
        requestAnimationFrame(() => requestAnimationFrame(() => {
            if (recoveredThisResume) return;
            recoveredThisResume = true;
            if (!recoverBoardIfNeeded()) {
                captureMessage(`Canvas recovery failed on ${source}, reloading...`, 'error');
                location.reload();
            }
        }));
    };

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
            recoveredThisResume = false;
            sawVisibleEvent = false;
            // Frames stop for a hidden page, so this callback IS the resume —
            // the one signal that survives a resume carrying no visibilitychange.
            requestAnimationFrame(() => {
                // Logged for the on-device debug viewer: a line here is the
                // record that this resume was carried by a frame, with the
                // visibility event either late or absent.
                if (!sawVisibleEvent) console.log('Resume: frame ahead of visibilitychange');
                recoverOnce(sawVisibleEvent ? 'resume frame' : 'frame without a visible event');
            });
            // The user is leaving — persist the freshest schedule now (keepalive
            // lets it land through teardown). Skips if unchanged since the last
            // upload; this is the main replacement for the per-attempt upload.
            maybeUploadSchedule();
        } else if (document.visibilityState === 'visible') {
            sawVisibleEvent = true;
            recoverOnce('visibilitychange');
            // First-run onboarding: re-evaluate ambient bubbles on resume (a new
            // session may have crossed the install/notify thresholds while
            // hidden). No-op for existing users.
            onboardingResume();
        }
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    logStartupVersions();

    updateBoardSize();
    // The WGo canvas takes its pixel size from #board-container, so it follows
    // the CONTAINER, not the window. The container is sized by viewport tokens
    // that include env(safe-area-inset-*), and an inset change fires no window
    // 'resize' — a window listener leaves the canvas at whatever size the
    // container had at init until something (a rotation) does fire one.
    new ResizeObserver(() => updateBoardSize())
        .observe(document.getElementById('board-container'));

    // Static UI wiring (persistent elements, registered once).
    initCornerMark();
    wireMenu();
    wireRoleButtonKeys();
    initMotifBubble();
    initCoordInput();
    wireHeaderReadingHints();
    wireActionButtons();
    wireNoDrag();

    // Normalize answer-history storage to the unified working set + IndexedDB
    // tail (history-store.js): everything beyond the cap is evicted to IDB —
    // freeing localStorage before the embedding backfill below writes through
    // the (now bounded) working set. Awaited so a quota-stressed device is
    // relieved before any further write; cheap on already-split state. (The
    // state-epoch gate at state.js import time replaced the migration
    // ladders; the working set is always the current unified-array shape.)
    await initHistoryStorage();

    loadRequiredAssetsOrThrow();

    await maybeFinishPendingAccountSwitch();

    // Re-derive the three rank slots from local history under the CURRENTLY SHIPPED
    // SE-floor curve. A stored rank was written against whatever curve was shipped
    // when it was last computed, so a re-ship moves it — and it is read before it is
    // written this boot (the first shard is picked from serveRank, the badge painted
    // from displayRank), which is why this runs here rather than after the load.
    // Idempotent and history-pure, so it costs one window fit and needs no version
    // stamp or migration to decide whether it applies.
    reanchorRank(getMergedAnswerHistory());

    // The recall model's readouts (statsResults) are not persisted, and the
    // front-page pace readings read them before any answer this session would
    // retrain. One fit from local history, synchronous and history-pure.
    calculateAndStoreStatsResults();

    // Fire-and-forget backfill: fills `embedding` and refreshes problemRating
    // (raw) + problemRank on stale answerHistory entries from already-deployed
    // zips and the now-cached curve, so returning users don't need to solve 20+
    // new problems before the weakness model has data. Background — the first
    // problem load doesn't wait.
    // Seed a genuinely-fresh device (empty history) to the CURRENT data
    // versions BEFORE the version gate reads them. This is what guarantees a
    // brand-new user is never shown the "ratings corrected" notice: with
    // current stamps its ratingsChanged/curveChanged are false, so the notice
    // is excluded structurally — not left to the attempt-count gate, which a
    // new user can cross (play 2, reload) before the backfill stamps them.
    // No-op once any history exists, so an established/returning user's real
    // stamps still drive the notice. (The account-switch flow already calls
    // this for adopted devices; this covers the normal first boot.)
    Promise.resolve()
        .then(() => stampVersionsForFreshHistory())
        .then(() => backfillMissingEmbeddings())
        // After the re-derive that actually shifted stored ranks, surface the
        // one-time "ratings corrected" notice (armed inside the backfill) over
        // the already-corrected board. Guarded to established users only.
        .then(() => maybeShowRatingsCorrectionNotice())
        .catch(() => {});

    // Resolve board size, then load the first problem (shared link / recall
    // notification / standing challenge).
    const intent = parseUrlIntent();
    await resolveInitialBoardSize(intent.tsumegoId);
    try {
        await loadInitialProblem(intent);
    } finally {
        document.getElementById('title').textContent = getTerm('tsumegoMode');
        updateAllUI();
        hideLoading();
        hideBootSplash();
    }

    // First-run onboarding bubbles. Runs after the first problem is painted and
    // history is loaded, so it can decide new-user vs existing-user and evaluate
    // whatever is already on screen. Self-disables for existing users.
    initOnboarding();

    startBackgroundServices();
    wireLifecycleRecovery();
});
