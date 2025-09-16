// js/main.js

import { state, setAppMode, setAcceptClicks, setDeferredInstallPrompt, updateProblemHistory } from './state.js';
import { updateAllUI, updateBoardSize, showDetailedStats } from './ui.js';
import { loadSgf } from './board.js';
import { initAudio } from './audio.js';
import { loadTsumego, loadTsumegoById, findZipIdForTsumegoId, loadZipMetadata, loadZipProblems } from './problem-loader.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { getTerm, updateUrlWithTsumegoId } from './utils.js';

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    setDeferredInstallPrompt(e);
});

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
    window.addEventListener('resize', updateBoardSize);

    

    // Handle button1 click - behavior depends on current mode
    document.getElementById('button1').addEventListener('click', (e) => {
        e.preventDefault();
        if (state.appMode === 'solving') {
            // In solving mode, button1 is "Review"
            if (!e.target.classList.contains('disabled')) {
                setAppMode('analyzing');
                enterAnalysisMode();
                updateAllUI(); // Update UI to show analysis mode
            }
        } else if (state.appMode === 'analyzing') {
            // In analysis mode, button1 is "Exit"
            setAppMode('solving');
            // Update URL to match current tsumego
            if (state.currentProblemFilename) {
                const tsumegoId = state.currentProblemFilename.replace('.sgf', '');
                updateUrlWithTsumegoId(tsumegoId);
            }
            loadTsumego(false);
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
            // Copy current URL to clipboard
            navigator.clipboard.writeText(window.location.href).then(() => {
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
                if (state.nextTsumego) {
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
        }

    } catch (error) {
        console.error("Failed to load initial tsumego:", error);
    }

    document.getElementById('app-title').textContent = getTerm('appTitle');
    updateAllUI();

    fixupZeroRatedProblems();
});