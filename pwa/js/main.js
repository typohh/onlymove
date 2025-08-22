// js/main.js

import { state } from './state.js';
import { updateAllUI, updateBoardSize } from './ui.js';
import { initAudio } from './audio.js';
import { loadTsumego, loadTsumegoById } from './tsumego.js';
import { enterAnalysisMode, undoLastMove } from './analysis.js';
import { updateUrlWithTsumegoId } from './utils.js';

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

    updateAllUI();
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

    document.body.addEventListener('click', initializeAudioOnInteraction);
    document.body.addEventListener('touchend', initializeAudioOnInteraction);

    // Check for URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const tsumegoId = urlParams.get('id');
    
    try {
        if (tsumegoId) {
            const success = await loadTsumegoById(tsumegoId);
        }
        await loadTsumego(false);
    } catch (error) {
        console.error("Failed to load initial tsumego:", error);
    }
});
