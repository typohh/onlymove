import { state } from './state.js';
import { buildRecallSchedule, waitingRecallCount } from './recall-model.js';
import { postJson } from './worker-post.js';

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

// Tells the service worker how many problems are still waiting to be reviewed,
// so a reminder sitting in the tray counts down as the user works through the
// backlog and is taken down when nothing is left. Called after an attempt is
// recorded and once at boot (another device may have cleared the backlog).
//
// Only the count crosses the boundary: the service worker owns the tray and
// knows whether a reminder is outstanding at all, which this side cannot see —
// getNotifications() is empty here for a notification shown by a since-dead
// browser process, which is every reminder that arrived while the app was
// closed. Fire-and-forget and silent; a lost message just leaves the tray
// showing a number that is one attempt stale.
export async function syncRecallNotification() {
    if (!supportsNotifications()) return;
    try {
        const count = waitingRecallCount();
        const reg = await navigator.serviceWorker.ready;
        const worker = navigator.serviceWorker.controller || reg.active;
        if (worker) worker.postMessage({ type: 'recall-count', count });
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

// iOS/iPadOS Safari is the ONE platform where pushManager.subscribe() does not
// surface the notification-permission prompt itself — it requires permission to
// already be 'granted' (iPadOS 13+ reports a Mac UA, so also probe touch points).
function isIOS() {
    const ua = navigator.userAgent || '';
    return /iPad|iPhone|iPod/.test(ua) ||
        (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

// Requests permission (only where subscribe() won't ask itself), creates a
// PushSubscription, and uploads it to the Worker. Must be called from a user
// gesture.
export async function enableNotifications() {
    if (!supportsNotifications()) throw new Error('Notifications not supported');

    const reg = await navigator.serviceWorker.ready;
    let sub = await reg.pushManager.getSubscription();
    if (!sub) {
        // On iOS, subscribe() below won't prompt — it needs permission granted
        // first — so request it explicitly here. Everywhere else (desktop,
        // Android Chrome, AND the TWA) subscribe({userVisibleOnly:true}) requests
        // permission on its own, so pre-calling requestPermission() is redundant
        // — and in the TWA actively harmful: the delegated Notification.permission
        // reads stale 'default' after the Android grant, so subscribe() re-asks
        // and a SECOND permission dialog pops. Gate the pre-request on iOS to
        // leave exactly one prompt on every platform. (Same delegated-permission
        // lag noted in maybeUploadSchedule / updateInstallButton.)
        if (isIOS() && Notification.permission === 'default') {
            const perm = await Notification.requestPermission();
            if (perm !== 'granted') return { ok: false, permission: perm };
        }

        const vapidKey = await fetchVapidPublicKey();
        try {
            sub = await reg.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: base64UrlToUint8Array(vapidKey),
            });
        } catch {
            // User dismissed/denied the prompt (or push is unavailable). No
            // subscription = not opted in; report the live permission so the
            // caller paints Blocked vs. Notify correctly.
            return { ok: false, permission: Notification.permission };
        }
    }

    // Send the device UI language so the recall cron renders each reminder in
    // the language the user actually reads; the server folds it to a supported
    // code (resolveLang in handlers/push.js) and stores it per subscription.
    await postJson('/push/subscribe', {
        userId: state.userId,
        subscription: subscriptionToJson(sub),
        lang: navigator.language || 'en',
    });

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
// Avoids pinging the Worker for users who never opted in. Opted-in is judged
// by the push subscription existing, NOT by Notification.permission — in the
// TWA the delegated permission property can keep reading 'default' after the
// Android grant, and gating on it silently stopped all schedule uploads (and
// the last_seen_at mid-session-suppression watermark) for TWA sessions.
export async function maybeUploadSchedule(opts) {
    const sub = await getCurrentSubscription();
    if (!sub) return;
    return uploadSchedule(opts);
}
