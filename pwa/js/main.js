// js/main.js

import { state, setAppMode, setAcceptClicks } from './state.js';
import { updateAllUI, updateBoardSize, showDetailedStats } from './ui.js';
import { loadSgf } from './board.js';
import { initAudio } from './audio.js';
import { loadTsumego, loadTsumegoById } from './problem-loader.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { getTerm, updateUrlWithTsumegoId } from './utils.js';

// --- INITIALIZATION ---

let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    if (document.readyState === 'complete') {
        showInstallPrompt();
    } else {
        document.addEventListener('DOMContentLoaded', showInstallPrompt);
    }
});

window.addEventListener('resize', () => {
    const prompt = document.getElementById('install-prompt');
    if (prompt) {
        prompt.remove();
        if (deferredPrompt) {
            showInstallPrompt();
        }
    }
});

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

    // Add click event to rank display for detailed stats
    const text1Elements = [document.getElementById('text1'), document.getElementById('text1-h')];
    text1Elements.forEach(element => {
        if (element) {
            element.style.cursor = 'pointer';
            element.title = 'Click for detailed stats';
            element.style.userSelect = 'none';
            element.style.webkitUserSelect = 'none';
            element.style.MozUserSelect = 'none';
            element.addEventListener('click', (event) => {
                event.preventDefault();
                showDetailedStats();
            });
        }
    });

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

});

function showInstallPrompt() {
    if (!deferredPrompt) return;

    const ratio = window.innerWidth / window.innerHeight;
    let displayMethod;

    if (ratio < 0.65) {
        displayMethod = 'notification';
    } else if (ratio > 1.3) {
        displayMethod = 'below-buttons';
    } else {
        displayMethod = 'popup';
    }

    const promptElement = document.createElement('div');
    promptElement.id = 'install-prompt';
    promptElement.innerHTML = `
        <button class="close-btn">×</button>
        <div class="install-message">Install this app for a better experience!</div>
        <button class="install-btn">Install</button>
    `;

    const installBtn = promptElement.querySelector('.install-btn');
    const closeBtn = promptElement.querySelector('.close-btn');

    if (displayMethod === 'notification') {
        Object.assign(promptElement.style, {
            position: 'fixed',
            top: '-100px',
            left: '0',
            width: '100%',
            backgroundColor: '#333',
            color: 'white',
            padding: '10px',
            textAlign: 'center',
            zIndex: '10000',
            transition: 'top 0.5s ease-in-out',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '10px'
        });
        closeBtn.style.position = 'absolute';
        closeBtn.style.top = '5px';
        closeBtn.style.right = '10px';
        document.body.appendChild(promptElement);
        setTimeout(() => promptElement.style.top = '0', 100);
    } else if (displayMethod === 'below-buttons') {
        Object.assign(promptElement.style, {
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '10px',
            marginTop: '1vmin',
            marginLeft: '4vmin',
            marginRight: '1vmin',
            padding: '1.5vmin 2.5vmin',
            border: '1px solid #ccc',
            borderRadius: '5px',
            position: 'relative',
            width: 'calc(100% - 5vmin)'
        });
        closeBtn.style.position = 'absolute';
        closeBtn.style.top = '1vmin';
        closeBtn.style.right = '1vmin';
        closeBtn.style.verticalAlign = 'top';
        // Apply dynamic font size to all text elements
        const dynamicFontSize = 'calc(0.035 * min(0.84 * 100vh, 100vw))';
        promptElement.style.fontSize = dynamicFontSize;
        promptElement.querySelector('.install-message').style.fontSize = dynamicFontSize;
        const installBtn = promptElement.querySelector('.install-btn');
        installBtn.style.fontSize = dynamicFontSize;
        installBtn.style.borderRadius = 'var(--radius)';
        installBtn.style.padding = '0.5vmin 2vmin';
        closeBtn.style.fontSize = dynamicFontSize;
        document.getElementById('info-container').appendChild(promptElement);
    } else { // popup
        Object.assign(promptElement.style, {
            position: 'fixed',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            backgroundColor: '#333',
            color: 'white',
            padding: '20px',
            borderRadius: '10px',
            textAlign: 'center',
            zIndex: '10000',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '10px',
            minWidth: '200px'
        });
        closeBtn.style.position = 'absolute';
        closeBtn.style.top = '5px';
        closeBtn.style.right = '10px';
        document.body.appendChild(promptElement);
    }

    promptElement.querySelector('.install-btn').addEventListener('click', async () => {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        console.log(`User response to install prompt: ${outcome}`);
        deferredPrompt = null;
        promptElement.remove();
    });

    promptElement.querySelector('.close-btn').addEventListener('click', () => {
        deferredPrompt = null;
        promptElement.remove();
    });
}
