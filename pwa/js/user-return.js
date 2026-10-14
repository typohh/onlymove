// js/user-return.js
//
// ONE "the user is back" signal for everything that refreshes on resume.
//
// visibilitychange alone misses the desktop return: a tab left in front while
// the user walks away, or one sitting visible on a second monitor while they
// work in another window, is never hidden, so nothing fires when they come
// back. The second signal is input after an idle gap — the first pointer, key,
// wheel or window-focus event after IDLE_RETURN_MS without any counts as a
// return. Window focus is in the set so an alt-tab back is read the moment it
// happens rather than at the next mouse move; a short alt-tab stays below the
// gap and fires nothing.
//
// The gap trades a cheap false positive (an up-to-date pull reads ~0–1 D1 rows,
// and the shown-problem refresh no-ops unless the selection actually moved)
// against a stale board. Below a few minutes it fires on a user who is merely
// thinking over a hard problem without touching the mouse.
//
// Board recovery is NOT a return callback: it repairs a renderer Android
// killed in the background, which cannot happen to a page that never hid, so
// it stays on visibilitychange in main.js.

const IDLE_RETURN_MS = 5 * 60 * 1000;
const ACTIVITY_EVENTS = ['pointermove', 'pointerdown', 'keydown', 'wheel', 'focus'];

const callbacks = [];
let lastActivity = Date.now();
let wired = false;

function fireReturn(source) {
    for (const cb of callbacks) {
        try { cb(source); } catch (e) { console.error('onUserReturn callback failed', e); }
    }
}

function wire() {
    if (wired) return;
    wired = true;
    window.addEventListener('visibilitychange', () => {
        // A hidden period is its own return signal; restarting the clock on
        // both edges keeps the first input after it from firing a second one.
        lastActivity = Date.now();
        if (document.visibilityState === 'visible') fireReturn('visible');
    });
    const onActivity = () => {
        const now = Date.now();
        const idle = now - lastActivity;
        lastActivity = now;
        if (idle >= IDLE_RETURN_MS && document.visibilityState === 'visible') fireReturn('idle');
    };
    // Capture + passive: sees every event before any handler can stop it, and
    // never delays scrolling. The handler is a timestamp compare, so
    // pointermove's rate costs nothing.
    for (const type of ACTIVITY_EVENTS) {
        window.addEventListener(type, onActivity, { capture: true, passive: true });
    }
}

// Register a callback for every return — a tab re-shown, or input after an
// idle gap while it stayed visible. The callback receives 'visible' or 'idle'.
export function onUserReturn(cb) {
    callbacks.push(cb);
    wire();
}
