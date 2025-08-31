// js/audio.js

let stoneSound = null;
let failSound1 = null;
let failSound2 = null;
let successSound = null;

function _playSound(audioObject) {
    if (audioObject) {
        try {
            audioObject.currentTime = 0;
            audioObject.play().catch(e => console.log('Audio play failed:', e));
        } catch (e) {
            console.log('Audio play failed:', e);
        }
    }
}

export function initAudio() {
    try {
        stoneSound = new Audio('sound/stone.wav');
        stoneSound.preLoad = 'auto';
//        failSound1 = new Audio('sound/fail1.wav');
//        failSound2 = new Audio('sound/fail2.wav');
//        successSound = new Audio('sound/success.wav');
//        [stoneSound, failSound1, failSound2, successSound].forEach(s => s.preload = 'auto');
    } catch (e) {
        console.log('Audio initialization failed:', e);
    }
}

export function playStoneSound() {
    _playSound(stoneSound);
}

export function playRandomFailSound() {
//    if (!failSound1 || !failSound2) return;
//    const randomSound = Math.random() < 0.5 ? failSound1 : failSound2;
//    _playSound(randomSound);
}

export function playSuccessSound() {
    // _playSound(successSound);
}
