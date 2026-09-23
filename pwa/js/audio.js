// js/audio.js
//
// The stone click plays through Web Audio, not an <audio> element. A media
// element makes the OS treat the page as a media player: Android grants it
// audio focus (pausing Spotify and the like) and iOS switches the page's
// audio session to playback (interrupting other apps). A Web Audio buffer
// source requests no audio focus on Android, and the 'ambient' audio session
// on iOS mixes with other apps' audio and respects the silent switch.
import { state } from './state.js';

const STONE_SOUND_URL = '/sound/stone.wav';
// An idle running context keeps the audio output open; suspend it after this
// long without a sound. Must exceed the delay before the opponent's reply to a
// wrong move (game-logic.js): that play is not a user gesture, and iOS may
// refuse to resume a suspended context outside one.
const IDLE_SUSPEND_MS = 1500;

let audioContext = null;
let stoneBuffer = null;
let stoneBufferPromise = null;
let idleTimer = null;

function getAudioContext() {
    if (audioContext) return audioContext;
    try {
        if (navigator.audioSession) navigator.audioSession.type = 'ambient';
    } catch (e) {
        console.log('Audio session setup failed:', e);
    }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audioContext = new Ctx();
    return audioContext;
}

function loadStoneBuffer(ctx) {
    if (!stoneBufferPromise) {
        stoneBufferPromise = fetch(STONE_SOUND_URL)
            .then(r => {
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return r.arrayBuffer();
            })
            // Callback form: older Safari's decodeAudioData returns no promise.
            .then(data => new Promise((resolve, reject) => ctx.decodeAudioData(data, resolve, reject)))
            .then(buffer => { stoneBuffer = buffer; return buffer; })
            .catch(e => {
                stoneBufferPromise = null; // let a later play retry
                throw e;
            });
    }
    return stoneBufferPromise;
}

function scheduleIdleSuspend(ctx) {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
        if (ctx.state === 'running') ctx.suspend().catch(() => {});
    }, IDLE_SUSPEND_MS);
}

function playBuffer(ctx, buffer) {
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();
    scheduleIdleSuspend(ctx);
}

export function playStoneSound() {
    if (!state.audioEnabled) return;
    try {
        const ctx = getAudioContext();
        if (!ctx) return;
        clearTimeout(idleTimer); // don't let a pending suspend race this play
        // resume() must be called from the user gesture that triggered the
        // move, so it goes before any await.
        const resumed = ctx.state === 'running' ? Promise.resolve() : ctx.resume();
        if (stoneBuffer) {
            resumed.then(() => playBuffer(ctx, stoneBuffer))
                .catch(e => console.log('Audio play failed:', e));
        } else {
            Promise.all([resumed, loadStoneBuffer(ctx)])
                .then(([, buffer]) => playBuffer(ctx, buffer))
                .catch(e => console.log('Audio play failed:', e));
        }
    } catch (e) {
        console.log('Audio play failed:', e);
    }
}
