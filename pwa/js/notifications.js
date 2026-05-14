import { state } from './state.js';
import { buildRecallSchedule } from './model.js';

const SUPPORTED_SIZES = [9, 13];

let cachedVapidPublicKey = null;

export function supportsNotifications() {
    return typeof Notification !== 'undefined' &&
           typeof navigator !== 'undefined' &&
           'serviceWorker' in navigator &&
           'PushManager' in window;
}

export function notificationPermission() {
    return supportsNotifications() ? Notification.permission : 'denied';
}

export async function getCurrentSubscription() {
    if (!supportsNotifications()) return null;
    try {
        const reg = await navigator.serviceWorker.ready;
        return await reg.pushManager.getSubscription();
    } catch {
        return null;
    }
}

async function fetchVapidPublicKey() {
    if (cachedVapidPublicKey) return cachedVapidPublicKey;
    const res = await fetch('/push/vapid-public-key');
    if (!res.ok) throw new Error(`vapid-public-key HTTP ${res.status}`);
    const key = (await res.text()).trim();
    if (!key) throw new Error('VAPID public key not configured on server');
    cachedVapidPublicKey = key;
    return key;
}

function base64UrlToUint8Array(b64u) {
    const padding = '='.repeat((4 - b64u.length % 4) % 4);
    const b64 = (b64u + padding).replace(/-/g, '+').replace(/_/g, '/');
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

function uint8ArrayToBase64Url(buf) {
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function subscriptionToJson(sub) {
    return {
        endpoint: sub.endpoint,
        keys: {
            p256dh: uint8ArrayToBase64Url(sub.getKey('p256dh')),
            auth: uint8ArrayToBase64Url(sub.getKey('auth')),
        },
    };
}

// Requests permission (if not yet granted), creates a PushSubscription, and
// uploads it to the Worker. Must be called from a user gesture.
export async function enableNotifications() {
    if (!supportsNotifications()) throw new Error('Notifications not supported');

    let perm = Notification.permission;
    if (perm === 'default') {
        perm = await Notification.requestPermission();
    }
    if (perm !== 'granted') return { ok: false, permission: perm };

    const reg = await navigator.serviceWorker.ready;
    let sub = await reg.pushManager.getSubscription();
    if (!sub) {
        const vapidKey = await fetchVapidPublicKey();
        sub = await reg.pushManager.subscribe({
            userVisibleOnly: true,
            applicationServerKey: base64UrlToUint8Array(vapidKey),
        });
    }

    await fetch('/push/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: state.userId, subscription: subscriptionToJson(sub) }),
    });

    // Push the current recall schedule right away so the cron has something
    // to fire on without waiting for the next attempt.
    uploadSchedule();

    return { ok: true, permission: 'granted' };
}

export async function disableNotifications() {
    const sub = await getCurrentSubscription();
    if (sub) {
        try { await sub.unsubscribe(); } catch {}
    }
    try {
        await fetch('/push/unsubscribe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: state.userId }),
        });
    } catch {}
}

let pendingUpload = false;

// Computes the recall schedule across both board sizes and POSTs it to the
// Worker. Idempotent — replace-all semantics on the server side, so calling
// this twice in quick succession is fine. Fails silently because losing one
// upload just delays the next notification by one tick.
//
// Skips the POST when the schedule is empty: the worker treats an empty
// array as "delete every row for this user", so a transient empty result
// (e.g. recent slice happens to contain only high-streak successes) would
// wipe a previously-good schedule and silently disable reminders.
export async function uploadSchedule() {
    if (pendingUpload) return;
    pendingUpload = true;
    try {
        const schedule = [];
        for (const size of SUPPORTED_SIZES) {
            for (const row of buildRecallSchedule(size)) {
                schedule.push(row);
            }
        }
        if (schedule.length === 0) return;
        await fetch('/schedule', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: state.userId, schedule }),
        });
    } catch (err) {
        console.warn('Schedule upload failed:', err);
    } finally {
        pendingUpload = false;
    }
}

// Uploads the schedule only if the user already has notifications enabled.
// Avoids pinging the Worker for users who never opted in.
export async function maybeUploadSchedule() {
    if (notificationPermission() !== 'granted') return;
    const sub = await getCurrentSubscription();
    if (!sub) return;
    return uploadSchedule();
}
