// js/tab-lock.js
//
// Single-active-tab guard. The whole client is architected single-writer
// (state.js holds the one mutable object; localStorage writes are whole-array
// last-writer-wins), so two live tabs corrupt each other: both serve the same
// standing challenge and record it twice, and the second tab's history write
// clobbers the first's. Instead of merging (only answerHistory is mergeable —
// the optimizer states and rank aren't), only one tab runs the app at a time.
//
// Mechanism: a Web Lock held for the tab's lifetime (auto-released on
// close/crash — no stale-flag problem). A tab that can't acquire it shows a
// blocking overlay with a "Use here" takeover button: it signals the holder
// over a BroadcastChannel, the holder marks itself as yielding (per-tab
// sessionStorage, so it survives the reload) and reloads, and the requester's
// queued lock request is granted, upon which it reloads to boot as the active
// tab. The yield mark is what makes the handoff deterministic: a Web Lock
// cannot survive the requester's navigation (it is auto-released when the
// requester's document unloads, before its fresh boot can re-request), so the
// ex-holder's earlier-started reload would otherwise land in that window,
// re-acquire the lock, and re-block the requester. Its fresh boot instead sees
// the mark and goes straight to the blocked overlay without competing ("Use
// here" still recovers it if the requester died mid-takeover). URL intents
// (?id / ?recall) survive the takeover reload.
//
// Browsers without navigator.locks skip the guard entirely (today's behavior;
// the record-time duplicate guard in user-stats.js is the fallback there).

import { getTerm } from './i18n.js';

const LOCK_NAME = 'onlymove-active-tab';
const CHANNEL_NAME = 'onlymove-tab-lock';
// Set (with a timestamp) just before the holder's takeover reload; per-tab.
const YIELD_KEY = 'tabLockYield';
// A mark older than this is a leftover from an interrupted reload, not an
// active takeover — ignore it and compete normally.
const YIELD_FRESH_MS = 10000;

// Returns true when this tab may run the app. Returns false after painting the
// blocked overlay — the caller must halt init.
export async function initTabLock() {
    if (!(navigator.locks && typeof navigator.locks.request === 'function')) return true;

    // This tab yielded to a takeover moments ago: land on the blocked overlay
    // without competing for the lock (see the header — the requester's reload
    // leaves the lock momentarily free, and our earlier-started boot would win
    // that race and re-block it).
    try {
        const ts = parseInt(sessionStorage.getItem(YIELD_KEY), 10);
        if (!Number.isNaN(ts)) sessionStorage.removeItem(YIELD_KEY);
        if (Number.isFinite(ts) && Date.now() - ts < YIELD_FRESH_MS) {
            showBlockedOverlay();
            return false;
        }
    } catch { /* sessionStorage unavailable → compete normally */ }

    let acquired;
    try {
        acquired = await new Promise((resolve) => {
            navigator.locks.request(LOCK_NAME, { ifAvailable: true }, (lock) => {
                if (!lock) {
                    resolve(false);
                    return;
                }
                resolve(true);
                // Hold the lock until the tab dies (or reloads on takeover).
                return new Promise(() => {});
            }).catch(() => resolve(true)); // lock API failure → fail open
        });
    } catch {
        return true;
    }

    if (acquired) {
        // Active tab: yield when a blocked tab requests takeover. The reload
        // releases the lock; this tab's fresh boot then lands on the blocked
        // overlay (the requester holds the lock by then).
        try {
            const channel = new BroadcastChannel(CHANNEL_NAME);
            channel.onmessage = (ev) => {
                if (ev && ev.data === 'takeover') {
                    try { sessionStorage.setItem(YIELD_KEY, String(Date.now())); } catch {}
                    location.reload();
                }
            };
        } catch { /* no BroadcastChannel → no takeover; blocked tabs stay blocked */ }
        return true;
    }

    showBlockedOverlay();
    return false;
}

function showBlockedOverlay() {
    const overlay = document.createElement('div');
    overlay.id = 'tab-lock-overlay';
    Object.assign(overlay.style, {
        position: 'fixed', inset: '0', zIndex: '99999',
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.55)',
    });
    const card = document.createElement('div');
    Object.assign(card.style, {
        maxWidth: '80vw', padding: '6vmin', borderRadius: '2vmin',
        backgroundColor: '#fff', color: '#222', textAlign: 'center',
        font: 'inherit', fontSize: 'max(16px, 2.2vmin)', lineHeight: '1.5',
    });
    const message = document.createElement('div');
    message.textContent = getTerm('tabAlreadyOpen');
    const button = document.createElement('button');
    button.textContent = getTerm('tabUseHere');
    Object.assign(button.style, {
        marginTop: '4vmin', padding: '2vmin 5vmin', fontSize: 'inherit',
        borderRadius: '1.5vmin', border: '1px solid #888',
        backgroundColor: '#f2f2f2', color: '#222', cursor: 'pointer',
    });
    button.addEventListener('click', () => {
        button.disabled = true;
        button.textContent = getTerm('tabTakingOver');
        try {
            new BroadcastChannel(CHANNEL_NAME).postMessage('takeover');
        } catch { /* holder unreachable — we still queue for the lock below */ }
        // Queue for the lock; granted once the holder's reload releases it.
        // Reload to boot cleanly as the active tab off the freshest storage.
        // Hold the lock across our reload (auto-released when this document
        // unloads) — releasing it on callback return would leave it free while
        // we reload, letting the former holder's fresh boot re-acquire it first
        // and land US back on the blocked overlay.
        navigator.locks.request(LOCK_NAME, () => {
            location.reload();
            return new Promise(() => {});
        }).catch(() => {});
    });
    card.appendChild(message);
    card.appendChild(button);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
}
