// js/main.js

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, updateProblemHistory, setUserHasInteracted, setProblemStartTime } from './state.js';
import { updateAllUI, updateBoardSize, showCaptureDetails } from './ui.js';
import { showDetailedStats } from './user-stats.js';
import { loadSgf } from './board.js';

import { initAudio } from './audio.js';
import { loadTsumego, loadTsumegoById, findZipIdForTsumegoId, loadZipMetadata, loadZipProblems } from './problem-loader.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { getTerm, cleanUrl } from './utils.js';

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    setDeferredInstallPrompt(e);
});

document.addEventListener('click', () => setUserHasInteracted(true));
document.addEventListener('touchstart', () => setUserHasInteracted(true));

async function fixupZeroRatedProblems() {
    const history = state.problemHistory;
    const problemsToFix = history.filter(p => p.rating === 0 && p.status !== 'correct');

    if (problemsToFix.length === 0) {
        return;
    }

    console.log(`Found ${problemsToFix.length} problems with zero rating to fix.`);

    // Run in the background
    setTimeout(async () => {
        let updated = false;
        await loadZipMetadata();

        for (const problem of problemsToFix) {
            try {
                const tsumegoId = problem.filename.replace('.sgf', '');
                const zipId = await findZipIdForTsumegoId(tsumegoId);

                if (!zipId) {
                    console.warn(`Could not find zipId for ${tsumegoId}`);
                    continue;
                }

                let zipFilename = null;
                for (const zip of state.zipMetadata) {
                    if (zip.filename.startsWith(zipId)) {
                        zipFilename = zip.filename;
                        break;
                    }
                }

                if (!zipFilename) {
                    console.warn(`Could not find zip filename for zipId ${zipId}`);
                    continue;
                }

                // Ensure zip is loaded
                if (!state.loadedZips[zipFilename]) {
                    await loadZipProblems(zipFilename);
                }

                const problemInfo = state.allProblems.find(p => p.filename === problem.filename);

                if (problemInfo && problemInfo.rating) {
                    problem.rating = problemInfo.rating;
                    updated = true;
                }
            } catch (error) {
                console.error(`Failed to fix rating for ${problem.filename}:`, error);
            }
        }

        if (updated) {
            updateProblemHistory(history);
            console.log("Finished fixing zero-rated problems in history.");
        }
    }, 0);
}



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
        if (state.appMode === 'solving' || state.appMode === 'linked') {
            // In solving/linked mode, button1 is "Review"
            if (!e.target.classList.contains('disabled')) {
                setAppMode('analyzing');
                enterAnalysisMode();
                updateAllUI(); // Update UI to show analysis mode
            }
        } else if (state.appMode === 'analyzing') {
            if (state.nextTsumego) {
                setAppMode('linked');
            } else {
                setAppMode('solving');
            }
            loadTsumego(false); // In both refresh cases, we just load what's at history[0]
            updateAllUI(); // Update UI to show solving mode
        }
    });

    // Handle button2 click - behavior depends on current mode
    document.getElementById('button2').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === 'solving') {
            // In solving mode, button2 is "About"
            window.open('about.html', '_blank');
        } else if (state.appMode === 'analyzing') {
            // In analysis mode, button2 is "Undo"
            if (state.kifuReader && state.kifuReader.node.parent) {
                undoLastMove();
            }
        }
    });

    // Handle button3 click - share functionality
    document.getElementById('button3').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === 'solving' || state.appMode === 'linked') {
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

    // Check for URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const tsumegoId = urlParams.get('id');

    try {
        if (tsumegoId && tsumegoId.length > 0) {
            const history = state.problemHistory;

            // Case 1: Review mode
            if (history.length > 1 && history[1].filename.startsWith(tsumegoId)) {
                setAppMode('analyzing');
                enterAnalysisMode();
            } 
            // Case 2: Refresh of existing problem (solving or linked)
            else if (history.length > 0 && history[0].filename.startsWith(tsumegoId)) {
                if (history[0].linked) {
                    setAppMode('linked');
                } else {
                    setAppMode('solving');
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
                setAppMode('linked');
            }
        }

    } catch (error) {
        console.error("Failed to load initial tsumego:", error);
    }

    cleanUrl();

    document.getElementById('title').textContent = getTerm('tsumegoMode');
    updateAllUI();

    fixupZeroRatedProblems();

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
        if (document.visibilityState === 'visible') {
            if (!state.board || !state.game) {
                location.reload();
            }
        }
    });
});
