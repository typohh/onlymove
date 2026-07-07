import { state } from './state.js';
import { buildRecallSchedule } from './model.js';
import { postJson } from './net.js';

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

    await postJson('/push/subscribe', { userId: state.userId, subscription: subscriptionToJson(sub) });

    // Push the current recall schedule right away so the cron has something
    // to fire on without waiting for the next attempt. Force past the
    // unchanged/throttle guards — the user just opted in, so persist now.
    uploadSchedule({ force: true });

    return { ok: true, permission: 'granted' };
}

export async function disableNotifications() {
    const sub = await getCurrentSubscription();
    // Capture the endpoint before unsubscribe so the server removes only THIS
    // device's row (subscriptions are per-device), not the whole account.
    const endpoint = sub?.endpoint;
    if (sub) {
        try { await sub.unsubscribe(); } catch {}
    }
    try {
        // No endpoint = nothing identifiable to remove for THIS device. POSTing
        // without one would hit the Worker's legacy account-wide fallback and
        // delete EVERY device's subscription for the account; a stale server row
        // for this device is instead cleaned by the cron's dead-endpoint
        // handling (the push service 404/410s it).
        if (endpoint) await postJson('/push/unsubscribe', { userId: state.userId, endpoint });
    } catch {}
}

let pendingUpload = false;
let lastUploadAt = 0;
// Empty-diff uploads (pure activity-watermark refreshes) are gated to at most
// once per this interval. Real changes always upload. The gap lets the
// heartbeat (see main.js) keep the server's last_seen_at watermark fresh on
// long idle-but-open sessions without spamming; kept under the heartbeat
// interval so a 30-min heartbeat always clears it.
const UNCHANGED_UPLOAD_THROTTLE_MS = 20 * 60 * 1000;
const HOUR_MS = 3600 * 1000;

// Per-row snapshot of what the server holds for this account, persisted so it
// survives reloads. Rows are hour-rounded — the recall cron runs hourly, so
// sub-hour drift from per-attempt model retrains is invisible to it. This is
// what turns the upload into a DIFF: each upsert bills the row plus its two
// partial indexes on D1, so a replace-all — which rewrites mostly-unchanged
// rows — is the heaviest write path. The snapshot is only advanced after the
// server confirms the POST, so a dropped request merely re-sends the same
// idempotent upserts next time.
const SCHEDULE_SNAPSHOT_KEY = 'recallScheduleSnapshot';

function readScheduleSnapshot() {
    try {
        const s = JSON.parse(localStorage.getItem(SCHEDULE_SNAPSHOT_KEY));
        if (s && s.userId === state.userId && s.rows && typeof s.rows === 'object') return s.rows;
    } catch { /* fall through */ }
    return null;
}

function writeScheduleSnapshot(rows) {
    try {
        localStorage.setItem(SCHEDULE_SNAPSHOT_KEY, JSON.stringify({ userId: state.userId, rows }));
    } catch { /* snapshot is an optimization; losing it only costs re-upserts */ }
}

// Computes the recall schedule across both board sizes and POSTs the CHANGES
// to the Worker. Fails silently because losing one upload just delays the
// next notification by one tick.
//
// The schedule is a pure projection of the account's synced answer history, so
// the latest upload from any device is authoritative — there's no per-device row
// ownership. Init awaits the full-history pull (initHistoryStorage +
// maybeFinishAccountSwitch) before the first upload, so by the time we run the
// history is complete and an empty recompute genuinely means "caught up." We
// therefore ALWAYS upload, empty or not, and let latest win: a diff removes our
// now-stale rows; a snapshot-less replace-all clears all and re-seeds. Do NOT add
// an empty-skip: it defeats the empty-diff watermark heartbeat for caught-up
// users, is asymmetric with the removal path, and masks bugs by making "empty
// because nothing's due" indistinguishable from "empty because the inputs broke".
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

        const current = {};
        for (const r of schedule) {
            current[r.problemId] = {
                e: Math.round(r.enterAt / HOUR_MS),
                x: Math.round(r.exitAt / HOUR_MS),
            };
        }

        const prev = readScheduleSnapshot();
        const upserts = [];
        const removals = [];
        for (const [pid, v] of Object.entries(current)) {
            const p = prev && prev[pid];
            if (!p || p.e !== v.e || p.x !== v.x) {
                upserts.push({ problemId: pid, enterAt: v.e * HOUR_MS, exitAt: v.x * HOUR_MS });
            }
        }
        if (prev) {
            for (const pid of Object.keys(prev)) {
                if (!current[pid]) removals.push(pid);
            }
        }

        const changed = upserts.length > 0 || removals.length > 0;
        const sinceLast = Date.now() - lastUploadAt;
        if (!force && !changed && sinceLast < UNCHANGED_UPLOAD_THROTTLE_MS) return;

        // No snapshot yet (first upload for this account on this device):
        // send the legacy replace-all shape once, so any rows left behind by
        // older clients get swept; diff thereafter.
        const payload = prev
            ? { userId: state.userId, upserts, removals }
            : { userId: state.userId, schedule: upserts };
        // keepalive lets the request outlive a page being hidden/torn down so
        // the visibilitychange-time upload in main.js actually lands.
        const res = await postJson('/schedule', payload, { keepalive: true });
        if (res && res.ok) {
            writeScheduleSnapshot(current);
            lastUploadAt = Date.now();
        }
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
