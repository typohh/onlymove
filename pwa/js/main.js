// js/main.js
import * as Sentry from "@sentry/browser";

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, updateProblemHistory, setUserHasInteracted, updateAnswerHistory, setReviewStartTime } from './state.js';

try {
    // Initialize Sentry
    Sentry.init({ dsn: 'https://50b747ee522c1bc41c8243563ce7adc1@o4510091105665024.ingest.de.sentry.io/4510091110187088' });
    window.Sentry = Sentry; // Make Sentry globally available
    Sentry.setUser({ id: state.userId });
} catch (error) {
    console.error("Sentry initialization failed:", error);
}

import { updateAllUI, updateBoardSize, showCaptureDetails } from './ui.js';
import { showDetailedStats } from './user-stats.js';
import { loadSgf, redraw } from './board.js';
import { getTerm } from './i18n.js';

import { initAudio } from './audio.js';

/**
 * Cleans the URL by removing the 'id' parameter and 'index.html'.
 */
function cleanUrl() {
    const url = new URL(window.location);
    url.searchParams.delete('id');
    const newPath = url.pathname.replace(/index\.html$/, '');
    window.history.replaceState({}, '', newPath + url.search + url.hash);
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
        const installPrompt = document.getElementById('install-prompt');
        if (installPrompt) {
            installPrompt.remove();
        }
    });

    

    // Handle button1 click - behavior depends on current mode
    document.getElementById('button1').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
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
            loadTsumego(false); // In both refresh cases, we just load what's at history[0]
            updateAllUI(); // Update UI to show solving mode
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
    document.getElementById('button3').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === APP_MODES.SOLVING || state.appMode === APP_MODES.LINKED) {
            showDetailedStats();
        } else { // 'analyzing'
            if (!state.problemHistory[1]) {
                console.error('No current problem to share');
                return;
            }
            const tsumegoId = state.problemHistory[1].filename.replace('.sgf', '');
            const shareUrl = `${window.location.origin}${window.location.pathname}?id=${tsumegoId}`;
            if (navigator.share) {
                navigator.share({
                    url: shareUrl
                }).then(() => {
                    // No visible message on success
                }).catch(err => {
                    if (err.name !== 'AbortError') {
                        console.error('Share failed: ', err);
                    }
                });
            } else {
                // Fallback to copy to clipboard
                navigator.clipboard.writeText(shareUrl).then(() => {
                    // Show a temporary confirmation message
                    const message = document.createElement('div');
                    message.textContent = getTerm('linkCopied');
                    Object.assign(message.style, {
                        position: 'fixed',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        color: 'white',
                        padding: '1vmin 2vmin',
                        borderRadius: '0.5vmin',
                        fontWeight: 'bold',
                        fontSize: '3vmin',
                        zIndex: '9999',
                        pointerEvents: 'none',
                        whiteSpace: 'nowrap',
                        textAlign: 'center'
                    });

                    document.body.appendChild(message);

                    // Remove after 2 seconds
                    setTimeout(() => {
                        if (message.parentNode) {
                            message.parentNode.removeChild(message);
                        }
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy URL: ', err);
                    // Show error message
                    const message = document.createElement('div');
                    message.textContent = getTerm('error');
                    Object.assign(message.style, {
                        position: 'fixed',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        backgroundColor: 'rgba(255, 0, 0, 0.8)',
                        color: 'white',
                        padding: '1vmin 2vmin',
                        borderRadius: '0.5vmin',
                        fontWeight: 'bold',
                        fontSize: '3vmin',
                        zIndex: '10000',
                        pointerEvents: 'none',
                        whiteSpace: 'nowrap',
                        textAlign: 'center'
                    });

                    document.body.appendChild(message);

                    // Remove after 2 seconds
                    setTimeout(() => {
                        if (message.parentNode) {
                            message.parentNode.removeChild(message);
                        }
                    }, 2000);
                });
            }
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
                await loadTsumegoById(tsumegoId);
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



    const resumeOverlay = document.getElementById('resume-overlay');

    // Recovery from killed session
    window.addEventListener('pageshow', (event) => {
        if (event.persisted) {
            // From bfcache, check if state is intact
            if (!state.board || !state.game) {
                location.reload();
            }
        }
    });

    window.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            resumeOverlay.className = 'resume-overlay';
        } else if (document.visibilityState === 'visible') {
            resumeOverlay.className = 'resume-overlay-hidden';
            if (!state.board || !state.game) {
                Sentry.captureMessage('Reloading due to missing board or game state on visibilitychange', 'error');
                location.reload();
            } else {
                setTimeout(() => redraw(), 0);
            }
        }
    });

    resumeOverlay.addEventListener('click', () => {
        resumeOverlay.className = 'resume-overlay-hidden';
    });

});
