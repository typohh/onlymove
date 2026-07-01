// js/audio.js
import { state } from './state.js';

let stoneSound = null;
let isAudioInitialized = false;

const _playSound = (audioObject) => {
    if (audioObject) {
        try {
            audioObject.currentTime = 0;
            audioObject.play().catch(e => console.log('Audio play failed:', e));
        } catch (e) {
            console.log('Audio play failed:', e);
        }
    }
}

function initAudio() {
    try {
        stoneSound = new Audio('sound/stone.wav');
        stoneSound.preload = 'auto';
        isAudioInitialized = true;
    } catch (e) {
        console.log('Audio initialization failed:', e);
    }
}

export function playStoneSound() {
    if (!state.audioEnabled) return;
    // If audio hasn't been initialized yet, initialize it now
    if (!isAudioInitialized) {
        initAudio();
    }
    _playSound(stoneSound);
}
