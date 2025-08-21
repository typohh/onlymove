// js/main.js

import { state } from './state.js';
import { updateAllUI, updateBoardSize, toggleAnalysisModeUI, showRulesInfo } from './ui.js';
import { initAudio } from './audio.js';
import { loadTsumego, loadTsumegoById, handleBoardClick } from './tsumego.js';
import { enterAnalysisMode } from './analysis.js';

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
    window.addEventListener('resize', updateBoardSize);
    window.addEventListener('orientationchange', updateBoardSize);

    document.getElementById('prev-tsumego').addEventListener('click', (e) => {
        e.preventDefault();
        if (!e.target.classList.contains('disabled')) {
            toggleAnalysisModeUI(true);
            enterAnalysisMode();
        }
    });

    document.getElementById('komi').addEventListener('click', (e) => {
        e.preventDefault();
        showRulesInfo();
    });

    document.getElementById('analysis-exit').addEventListener('click', () => {
        state.appMode = 'solving';
        toggleAnalysisModeUI(false);
        loadTsumego(false);
    });

    document.getElementById('analysis-undo').addEventListener('click', () => {
        if (state.appMode === 'analyzing' && state.kifuReader && state.kifuReader.node.parent) {
            import('./analysis.js').then(module => {
                module.undoLastMove();
            });
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
