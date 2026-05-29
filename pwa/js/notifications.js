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

// Dismisses any recall reminder the service worker is still displaying. Called
// when the user attempts to solve a problem: by then the reminder has done its
// job (it brought them back, or they're here anyway), so leaving it stacked in
// the tray is just noise. Fire-and-forget and silent — a failed clear is
// cosmetic. Closes every notification owned by this registration; they all
// share the 'recall-batch' tag today, but closing all is future-proof and
// avoids a needless tag filter.
export async function clearShownNotifications() {
    if (!supportsNotifications()) return;
    try {
        const reg = await navigator.serviceWorker.ready;
        const notes = await reg.getNotifications();
        for (const note of notes) note.close();
    } catch {}
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
    // to fire on without waiting for the next attempt. Force past the
    // unchanged/throttle guards — the user just opted in, so persist now.
    uploadSchedule({ force: true });

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
// Session-scoped (in-memory, reset on reload) dedup state. Each upload does a
// full replace-all on the server (N row-writes), so we avoid re-POSTing when
// the schedule hasn't materially moved. On a fresh load the fingerprint is
// null, so the first call always uploads — re-establishing the server's rows
// and stamping a fresh updated_at ("active now") for that session.
let lastUploadFingerprint = null;
let lastUploadAt = 0;
// Unchanged uploads are gated to at most once per this interval. Changed
// schedules always upload. The gap lets the heartbeat (see main.js) refresh
// the server's updated_at watermark on long idle-but-open sessions without
// spamming writes; kept under the heartbeat interval so a 30-min heartbeat
// always clears it.
const UNCHANGED_UPLOAD_THROTTLE_MS = 20 * 60 * 1000;

// Canonical, hourly-resolution fingerprint of a schedule. The recall cron runs
// hourly, so sub-hour drift in enter/exit (from per-attempt model retrains) is
// invisible to it — rounding to the hour lets us treat those as "unchanged"
// and skip the write.
function fingerprintSchedule(schedule) {
    const HOUR_MS = 3600 * 1000;
    return schedule
        .map(r => `${r.problemId}:${Math.round(r.enterAt / HOUR_MS)}:${Math.round(r.exitAt / HOUR_MS)}`)
        .sort()
        .join('|');
}

// Computes the recall schedule across both board sizes and POSTs it to the
// Worker. Idempotent — replace-all semantics on the server side, so calling
// this twice in quick succession is fine. Fails silently because losing one
// upload just delays the next notification by one tick.
//
// Skips the POST when the schedule is empty: the worker treats an empty
// array as "delete every row for this user", so a transient empty result
// (e.g. recent slice happens to contain only high-streak successes) would
// wipe a previously-good schedule and silently disable reminders.
//
// Also skips when the schedule is unchanged (hourly fingerprint) and a recent
// upload already covers it — unless `force` is set or it's been long enough to
// warrant refreshing updated_at. This is what keeps active-session write volume
// down: an active user is suppressed by the cron anyway, so the schedule only
// needs to reach the server fresh once they go idle (see the hidden-tab and
// heartbeat triggers in main.js).
export async function uploadSchedule({ force = false } = {}) {
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

        const fingerprint = fingerprintSchedule(schedule);
        const changed = fingerprint !== lastUploadFingerprint;
        const sinceLast = Date.now() - lastUploadAt;
        if (!force && !changed && sinceLast < UNCHANGED_UPLOAD_THROTTLE_MS) return;

        await fetch('/schedule', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId: state.userId, schedule }),
            // Let the request outlive a page being hidden/torn down so the
            // visibilitychange-time upload in main.js actually lands.
            keepalive: true,
        });
        lastUploadFingerprint = fingerprint;
        lastUploadAt = Date.now();
    } catch (err) {
        console.warn('Schedule upload failed:', err);
    } finally {
        pendingUpload = false;
    }
}

// Uploads the schedule only if the user already has notifications enabled.
// Avoids pinging the Worker for users who never opted in.
export async function maybeUploadSchedule(opts) {
    if (notificationPermission() !== 'granted') return;
    const sub = await getCurrentSubscription();
    if (!sub) return;
    return uploadSchedule(opts);
}
