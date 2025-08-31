// js/main.js

import { state } from './state.js';
import { updateAllUI, updateBoardSize } from './ui.js';
import { initAudio } from './audio.js';
import { loadTsumego, loadTsumegoById } from './problem-loader.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { getTerm, updateUrlWithTsumegoId } from './utils.js';

// --- NEW AUDIO UNLOCK LOGIC ---
let isAudioInitialized = false;
function initializeAudioOnInteraction() {
    if (isAudioInitialized) return;
    try {
        initAudio();
        isAudioInitialized = true;
    } catch (e) {
        console.error("Audio initialization failed on interaction:", e);
    }
    // Remove the event listener after it has run once
    document.body.removeEventListener('click', initializeAudioOnInteraction);
    document.body.removeEventListener('touchend', initializeAudioOnInteraction);
}

// --- INITIALIZATION ---

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
                state.appMode = 'analyzing';
                enterAnalysisMode();
                updateAllUI(); // Update UI to show analysis mode
            }
        } else if (state.appMode === 'analyzing') {
            // In analysis mode, button1 is "Exit"
            state.appMode = 'solving';
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

    document.body.addEventListener('click', initializeAudioOnInteraction);
    document.body.addEventListener('touchend', initializeAudioOnInteraction);

    // Check for URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const tsumegoId = urlParams.get('id');
    
    try {
        let loadedById = false;
        if (tsumegoId) {
            loadedById = await loadTsumegoById(tsumegoId);
        }
        if (!loadedById) {
            await loadTsumego(false);
        }
        // Update UI after tsumego is loaded
        updateAllUI();
    } catch (error) {
        console.error("Failed to load initial tsumego:", error);
        // Still update UI even if loading fails
        updateAllUI();
    }
});
