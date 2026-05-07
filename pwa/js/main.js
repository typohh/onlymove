// js/main.js
import * as Sentry from "@sentry/browser";

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, updateProblemHistory, setUserHasInteracted, updateAnswerHistory, setReviewStartTime, updateNextTsumego, setBoardSize, SUPPORTED_BOARD_SIZES } from './state.js';

try {
    // Initialize Sentry
    Sentry.init({ dsn: 'https://50b747ee522c1bc41c8243563ce7adc1@o4510091105665024.ingest.de.sentry.io/4510091110187088' });
    window.Sentry = Sentry; // Make Sentry globally available
    Sentry.setUser({ id: state.userId });
} catch (error) {
    console.error("Sentry initialization failed:", error);
}

import { updateAllUI, updateBoardSize, showCaptureDetails, handleInstallClick } from './ui.js';
import { showDetailedStats } from './model.js';
import { loadSgf, redraw, initializeBoard, getBoardImageBlob } from './board.js';
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
        const overlay = document.createElement('div');
        overlay.id = 'ios-install-overlay';

        const modal = document.createElement('div');
        modal.id = 'ios-install-modal';
        modal.style.width = 'max-content';

        const title = document.createElement('div');
        title.id = 'ios-install-title';
        title.textContent = getTerm('boardSizePickerTitle');

        const buttonRow = document.createElement('div');
        buttonRow.style.display = 'flex';
        buttonRow.style.gap = '1.5vmin';
        buttonRow.style.justifyContent = 'center';

        for (const size of SUPPORTED_BOARD_SIZES) {
            const btn = document.createElement('button');
            btn.className = 'btn';
            btn.textContent = `${size}x${size}`;
            btn.style.flex = '0 0 auto';
            btn.style.width = 'auto';
            btn.style.minWidth = '6em';
            btn.style.borderRadius = 'var(--radius)';
            btn.addEventListener('click', () => {
                overlay.remove();
                resolve(size);
            });
            buttonRow.appendChild(btn);
        }

        if (dismissable) {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    overlay.remove();
                    resolve(null);
                }
            });
        }

        modal.appendChild(title);
        modal.appendChild(buttonRow);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
    });
}
import { loadTsumego, loadTsumegoById, findZipIdForTsumegoId, loadZipMetadata, loadZipProblems } from './problem-loader.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { APP_MODES } from './constants.js';

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    setDeferredInstallPrompt(e);
});

document.addEventListener('click', () => setUserHasInteracted(true));
document.addEventListener('touchstart', () => setUserHasInteracted(true));





document.addEventListener('DOMContentLoaded', async () => {
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

    const refreshSizeToggleLabel = () => {
        const label = `${state.boardSize}x${state.boardSize}`;
        for (const id of ['board-size-toggle', 'board-size-toggle-h']) {
            const el = document.getElementById(id);
            if (el) el.textContent = label;
        }
    };

    const handleSizeToggle = async (e) => {
        e.preventDefault();
        const next = await showBoardSizePicker({ dismissable: true });
        if (next === null || next === state.boardSize) return;
        setBoardSize(next);
        // Linked mode exits on size switch: the linked challenge belongs to
        // the previous size's problem space.
        setAppMode(APP_MODES.SOLVING);
        try {
            await loadTsumego(false);
        } catch (err) {
            console.error('Failed to load tsumego after size switch:', err);
        }
        if (state.problemHistory.length > 0 && state.problemHistory[0].linked) {
            setAppMode(APP_MODES.LINKED);
        }
        refreshSizeToggleLabel();
        updateAllUI();
    };
    document.getElementById('board-size-toggle').addEventListener('click', handleSizeToggle);
    document.getElementById('board-size-toggle-h').addEventListener('click', handleSizeToggle);



    // Handle button1 click - behavior depends on current mode
    document.getElementById('button1').addEventListener('click', (e) => {
        e.preventDefault();
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
            if (!e.target.classList.contains('disabled')) {
                enterAnalysisMode();
            }
        } else if (state.appMode === APP_MODES.ANALYZING) {
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

    // Handle button2 click - behavior depends on current mode
    document.getElementById('button2').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === APP_MODES.SOLVING) {
            // In solving mode, button2 is "About"
            window.open('about.html', '_blank');
        } else if (state.appMode === APP_MODES.ANALYZING) {
            // In analysis mode, button2 is "Undo"
            undoLastMove();
        }
    });

    // Handle button3 click - share functionality
    let isSharing = false;
    document.getElementById('button3').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
            showDetailedStats();
            return;
        }
        // analyzing
        if (isSharing) return;
        if (!state.problemHistory[1]) {
            console.error('No current problem to share');
            return;
        }
        isSharing = true;
        const releaseGuard = () => { isSharing = false; };

        const tsumegoId = state.problemHistory[1].filename.replace('.sgf', '');
        const orientation = state.problemHistory[1].orientation;
        let shareUrl = `${window.location.origin}${window.location.pathname}?id=${tsumegoId}`;
        if (typeof orientation === 'number' && orientation !== 0) {
            shareUrl += `&o=${orientation}`;
        }

        const showToast = (text, { isError = false, autoDismiss = true } = {}) => {
            const message = document.createElement('div');
            message.textContent = text;
            Object.assign(message.style, {
                position: 'fixed',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                backgroundColor: isError ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 0, 0, 0.8)',
                color: 'white',
                padding: '1vmin 2vmin',
                borderRadius: '0.5vmin',
                fontWeight: 'bold',
                fontSize: '3vmin',
                zIndex: isError ? '10000' : '9999',
                pointerEvents: 'none',
                whiteSpace: 'nowrap',
                textAlign: 'center'
            });
            document.body.appendChild(message);
            if (autoDismiss) {
                setTimeout(() => {
                    if (message.parentNode) message.parentNode.removeChild(message);
                }, 2000);
            }
            return message;
        };

        const copyToClipboard = () => {
            return navigator.clipboard.writeText(shareUrl).then(() => {
                showToast(getTerm('linkCopied'));
            }).catch(err => {
                console.error('Failed to copy URL: ', err);
                showToast(getTerm('error'), { isError: true });
            });
        };

        const shareLink = () => {
            if (navigator.share && navigator.canShare && navigator.canShare({ url: shareUrl })) {
                return navigator.share({ url: shareUrl }).catch(err => {
                    if (err.name === 'AbortError') return;
                    console.error('Share link failed: ', err);
                    showToast(getTerm('shareFailed'), { isError: true });
                    return copyToClipboard();
                });
            }
            return copyToClipboard();
        };

        // Generate the board image with a visible "preparing…" toast and a
        // hard timeout so canvas/render hangs surface as an error rather than
        // a silent forever-spinner.
        const generateImageFile = async () => {
            const toast = showToast(getTerm('preparingImage'), { autoDismiss: false });
            const cleanup = () => { if (toast.parentNode) toast.parentNode.removeChild(toast); };
            try {
                const blob = await Promise.race([
                    getBoardImageBlob(),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('image generation timed out')), 10000))
                ]);
                cleanup();
                if (!blob) throw new Error('No blob returned from getBoardImageBlob');
                return new File([blob], 'tsumego.png', { type: 'image/png' });
            } catch (err) {
                cleanup();
                console.error('Failed to capture image: ', err);
                showToast(getTerm('imageCaptureFailed'), { isError: true });
                throw err;
            }
        };

        const shareImage = async () => {
            let file;
            try {
                file = await generateImageFile();
            } catch {
                return shareLink();
            }
            const data = { files: [file], title: getTerm('tsumegoMode') };
            if (navigator.share && navigator.canShare && navigator.canShare(data)) {
                return navigator.share(data).catch(err => {
                    if (err.name === 'AbortError') return;
                    console.error('Share image failed: ', err);
                    showToast(getTerm('shareFailed'), { isError: true });
                    return shareLink();
                });
            }
            return shareLink();
        };

        const shareBoth = async () => {
            let file;
            try {
                file = await generateImageFile();
            } catch {
                return shareLink();
            }
            const data = { files: [file], url: shareUrl, title: getTerm('tsumegoMode') };
            if (navigator.share && navigator.canShare && navigator.canShare(data)) {
                return navigator.share(data).catch(err => {
                    if (err.name === 'AbortError') return;
                    console.error('Share both failed: ', err);
                    showToast(getTerm('shareFailed'), { isError: true });
                    return copyToClipboard();
                });
            }
            return shareLink();
        };

        const showShareChooser = () => {
            const overlay = document.createElement('div');
            overlay.id = 'ios-install-overlay';

            const modal = document.createElement('div');
            modal.id = 'ios-install-modal';

            const bothBtn = document.createElement('button');
            bothBtn.className = 'btn';
            bothBtn.textContent = getTerm('shareBoth');
            bothBtn.addEventListener('click', () => { overlay.remove(); shareBoth().finally(releaseGuard); });

            const linkBtn = document.createElement('button');
            linkBtn.className = 'btn';
            linkBtn.textContent = getTerm('shareLink');
            linkBtn.addEventListener('click', () => { overlay.remove(); shareLink().finally(releaseGuard); });

            const imageBtn = document.createElement('button');
            imageBtn.className = 'btn';
            imageBtn.textContent = getTerm('shareImage');
            imageBtn.addEventListener('click', () => { overlay.remove(); shareImage().finally(releaseGuard); });

            overlay.addEventListener('click', (ev) => {
                if (ev.target === overlay) {
                    overlay.remove();
                    releaseGuard();
                }
            });

            modal.appendChild(bothBtn);
            modal.appendChild(linkBtn);
            modal.appendChild(imageBtn);
            overlay.appendChild(modal);
            document.body.appendChild(overlay);
        };

        // Probe whether the browser can share files at all, without actually generating one.
        let filesSupported = false;
        try {
            filesSupported = !!(navigator.share && navigator.canShare &&
                navigator.canShare({ files: [new File([], 'p.png', { type: 'image/png' })] }));
        } catch {
            filesSupported = false;
        }

        if (filesSupported) {
            showShareChooser();
        } else {
            shareLink().finally(releaseGuard);
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

    // Handle button4 click - Install (solving/linked), Ignore (analyzing)
    document.getElementById('button4').addEventListener('click', (e) => {
        e.preventDefault();
        if (e.target.classList.contains('disabled')) return;

        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
            handleInstallClick();
        } else if (state.appMode === APP_MODES.ANALYZING) {
            if (confirm(getTerm('ignoreConfirm'))) {
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
            }
        }
    });

    // The board-size migration in state.js has already wrapped any pre-existing
    // answerHistory under the 9 bucket. These legacy migrations therefore read
    // and write via state.answerHistory (the active size's view) rather than
    // touching the raw localStorage value directly.
    const migrateAccumulatedReview = () => {
        if (localStorage.getItem('answerHistoryMigrated_v4_accumulatedReview')) {
            return;
        }
        console.log('Running answer history migration for accumulatedReview...');

        const answerHistory = state.answerHistory;
        const answersByProblem = {};

        for (const answer of answerHistory) {
            if (!answersByProblem[answer.problemId]) {
                answersByProblem[answer.problemId] = [];
            }
            answersByProblem[answer.problemId].push(answer);
        }

        for (const problemId in answersByProblem) {
            answersByProblem[problemId].sort((a, b) => a.date - b.date);
        }

        const newHistory = [];
        for (const answer of answerHistory) {
            const problemAnswers = answersByProblem[answer.problemId];
            const prevAnswers = problemAnswers.filter(a => a.date < answer.date);
            const accumulatedReview = prevAnswers.reduce((sum, a) => sum + (a.reviewTime || 0), 0);

            const newAnswer = { ...answer };
            if (newAnswer.accumulatedReview === undefined) {
                newAnswer.accumulatedReview = accumulatedReview;
            }
            newHistory.push(newAnswer);
        }

        updateAnswerHistory(newHistory);
        localStorage.setItem('answerHistoryMigrated_v4_accumulatedReview', 'true');
        console.log('Answer history migration for accumulatedReview complete.');
    };

    migrateAccumulatedReview();

    const migrateFailedNovelSuccessSinceFailure = () => {
        if (localStorage.getItem('answerHistoryMigrated_v5_failedNovelSsf')) {
            return;
        }
        console.log('Running answer history migration for failed novel successSinceFailure...');

        const answerHistory = [...state.answerHistory];
        let changed = 0;
        for (const answer of answerHistory) {
            if (!answer.correct && answer.successSinceFailure === undefined) {
                answer.successSinceFailure = 0;
                changed++;
            }
        }

        if (changed > 0) {
            updateAnswerHistory(answerHistory);
            console.log(`Patched ${changed} answers with missing successSinceFailure.`);
        }
        localStorage.setItem('answerHistoryMigrated_v5_failedNovelSsf', 'true');
        console.log('Answer history migration for failed novel successSinceFailure complete.');
    };

    migrateFailedNovelSuccessSinceFailure();

    const migrateProblemTypeToIsRepeat = () => {
        if (localStorage.getItem('answerHistoryMigrated_v3_isRepeat')) {
            return;
        }
        console.log('Running answer history migration for isRepeat...');

        const answerHistory = state.answerHistory;
        const newHistory = answerHistory.map(answer => {
            if (answer.problemType !== undefined) {
                answer.isRepeat = answer.problemType === 'repeat';
                delete answer.problemType;
            }
            return answer;
        });

        updateAnswerHistory(newHistory);
        localStorage.setItem('answerHistoryMigrated_v3_isRepeat', 'true');
        console.log('Answer history migration for isRepeat complete.');
    };

    migrateProblemTypeToIsRepeat();

    const migrateClearIgnoredFlag = () => {
        if (localStorage.getItem('answerHistoryMigrated_v6_clearIgnored')) {
            return;
        }
        console.log('Running answer history migration to clear confounded ignored flags...');

        const answerHistory = [...state.answerHistory];
        let cleared = 0;
        for (const answer of answerHistory) {
            if (answer.ignored !== undefined) {
                delete answer.ignored;
                cleared++;
            }
        }

        if (cleared > 0) {
            updateAnswerHistory(answerHistory);
            console.log(`Cleared ignored flag from ${cleared} answers.`);
        }
        localStorage.setItem('answerHistoryMigrated_v6_clearIgnored', 'true');
        console.log('Answer history migration for clearing ignored flags complete.');
    };

    migrateClearIgnoredFlag();

    const migrateAnswerHistory = () => {
        if (localStorage.getItem('answerHistoryMigrated_v2')) {
            return;
        }

        console.log('Running answer history migration for successSinceFailure...');

        const answerHistory = state.answerHistory;
        const answersByProblem = {};

        for (const answer of answerHistory) {
            if (!answersByProblem[answer.problemId]) {
                answersByProblem[answer.problemId] = [];
            }
            answersByProblem[answer.problemId].push(answer);
        }

        for (const problemId in answersByProblem) {
            const problemAnswers = answersByProblem[problemId];
            let successCounter = 0;
            let hasFailed = false;

            for (const answer of problemAnswers) {
                if (!answer.correct) {
                    hasFailed = true;
                    successCounter = 0;
                    answer.successSinceFailure = 0;
                } else if (hasFailed) {
                    successCounter++;
                    answer.successSinceFailure = successCounter;
                }
            }
        }

        updateAnswerHistory(answerHistory);
        localStorage.setItem('answerHistoryMigrated_v2', 'true');
        console.log('Answer history migration complete.');
    };

    migrateAnswerHistory();

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
        // Render an empty board (and refresh the size pill) behind the picker
        // so the dialog overlays a real-looking board instead of empty void.
        initializeBoard(state.boardSize);
        refreshSizeToggleLabel();
        const chosen = await showBoardSizePicker();
        setBoardSize(chosen);
        // Same reason as above — force-persist when chosen equals the default.
        localStorage.setItem('boardSize', String(chosen));
    }

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
            // Case 3: Default - new linked problem
            else {
                await loadTsumegoById(tsumegoId, orientation);
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
    refreshSizeToggleLabel();
    updateAllUI();



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

    // Solution #4: Resume overlay click triggers full board recreation for manual recovery
    resumeOverlay.addEventListener('click', () => {
        resumeOverlay.className = 'resume-overlay-hidden';
        
        // Force full board recreation when user clicks to resume
        if (state.kifuReader) {
            try {
                const size = state.kifuReader.game.position.size;
                initializeBoard(size);
                redraw();
            } catch (e) {
                Sentry.captureMessage('Manual board recovery failed: ' + e.message, 'error');
                location.reload();
            }
        }
    });

});
