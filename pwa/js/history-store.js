// js/history-store.js
//
// IndexedDB backing for the answer-history COLD TAIL — the attempts older than
// the unified working set that state.js keeps in localStorage. The working set
// (most recent ANSWER_HISTORY_WORKING_SET attempts, see constants.js) stays in
// synchronous localStorage so the hot path never awaits; everything older lives
// here and is read back only when the full-history charts, a model refit, or an
// account switch needs it.
//
// Design notes:
//   - Single object store with OUT-OF-LINE autoincrement keys, so stored
//     records are the answer-history entries verbatim (each carries its `size`)
//     — no synthetic key field leaks back into the history array.
//   - Insertion order == eviction order == date order (state.evictColdTail
//     always appends the OLDEST overflow first), so getTail() returns the tail
//     oldest→newest with no index needed.
//   - Every method is defensive: if IndexedDB is missing (private mode, old
//     browser) or a transaction errors, reads resolve to [] and writes resolve
//     to false. A false from appendTail tells the caller NOT to trim the
//     working set, so an unavailable tail degrades to "charts show the working
//     set only" rather than losing attempts.

import { rowHash } from '../shared/sync-digest.js';

const DB_NAME = 'onlymove';
const DB_VERSION = 1;
const STORE = 'answerHistoryTail';

let dbPromise = null;

function idbAvailable() {
    try {
        return typeof indexedDB !== 'undefined' && indexedDB !== null;
    } catch {
        return false;
    }
}

// Lazily open (and create) the DB. Resolves to the IDBDatabase, or null if IDB
// is unavailable / the open failed. Durable-storage is NOT requested here — see
// requestPersist(), which state.evictColdTail calls only once there is actually
// a tail worth protecting (avoids a permission prompt for users with no tail).
function openDB() {
    if (!idbAvailable()) return Promise.resolve(null);
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve) => {
        let req;
        try {
            req = indexedDB.open(DB_NAME, DB_VERSION);
        } catch {
            resolve(null);
            return;
        }
        req.onupgradeneeded = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE)) {
                db.createObjectStore(STORE, { autoIncrement: true });
            }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
        req.onblocked = () => resolve(null);
    });
    return dbPromise;
}

// Append rows to the tail (oldest first). Resolves true once durably written
// (or trivially for an empty batch), false if IDB is unavailable or the write
// failed — the caller must keep those rows in the working set on a false.
export async function appendTail(rows) {
    if (!rows || rows.length === 0) return true;
    const db = await openDB();
    if (!db) return false;
    return new Promise((resolve) => {
        let tx;
        try {
            tx = db.transaction(STORE, 'readwrite');
        } catch {
            resolve(false);
            return;
        }
        const store = tx.objectStore(STORE);
        for (const r of rows) {
            try { store.add(r); } catch {}
        }
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
        tx.onabort = () => resolve(false);
    });
}

// Number of rows currently in the tail (cheap — does not load the rows).
// Used to detect silent browser eviction against the expected high-water mark.
// 0 if unavailable/empty.
export async function countTail() {
    const db = await openDB();
    if (!db) return 0;
    return new Promise((resolve) => {
        let req;
        try {
            req = db.transaction(STORE, 'readonly').objectStore(STORE).count();
        } catch {
            resolve(0);
            return;
        }
        req.onsuccess = () => resolve(req.result || 0);
        req.onerror = () => resolve(0);
    });
}

// Request durable (non-evictable) storage. Best-effort: resolves to the
// browser's decision (true/false) or false if unsupported. Called once, gated
// on the tail being non-empty, so users with no cold tail are never prompted.
export async function requestPersist() {
    try {
        if (navigator.storage?.persist) return await navigator.storage.persist();
    } catch {}
    return false;
}

// All tail rows, oldest→newest. [] if unavailable/empty.
export async function getTail() {
    const db = await openDB();
    if (!db) return [];
    return new Promise((resolve) => {
        let req;
        try {
            req = db.transaction(STORE, 'readonly').objectStore(STORE).getAll();
        } catch {
            resolve([]);
            return;
        }
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
    });
}

// One cold-tail scan for the sync digest: how many tail rows are acked
// (serverCreatedAt present), the Σ rowHash(serverId) over those that carry a
// serverId, and whether every acked row has one (allHaveId=false disables the
// content check, leaving the count check). Defensive: empty info on any error.
export async function getTailDigestInfo() {
    const db = await openDB();
    const info = { ackedCount: 0, idSum: 0, allHaveId: true };
    if (!db) return info;
    return new Promise((resolve) => {
        let req;
        try {
            req = db.transaction(STORE, 'readonly').objectStore(STORE).openCursor();
        } catch {
            resolve(info);
            return;
        }
        req.onsuccess = () => {
            const cursor = req.result;
            if (!cursor) { resolve(info); return; }
            const v = cursor.value;
            if (v && v.serverCreatedAt) {
                info.ackedCount++;
                if (Number.isFinite(v.serverId)) info.idSum += rowHash(v.serverId);
                else info.allHaveId = false;
            }
            cursor.continue();
        };
        req.onerror = () => resolve(info);
    });
}

// One cursor pass building everything the sync merge needs to know about the
// tail: the acked row ids (id-first dedup), the serverCreatedAt stamp set
// (legacy timestamp dedup), and per-stamp identity fields (the id-first merge's
// identical-content check, which decides whether a same-timestamp server row is
// a redundant duplicate copy to absorb or a genuine same-millisecond collision
// to insert). Stamps can collide (that's the point), so `stamps` maps a stamp
// to the LIST of rows carrying it. Empty index if IDB is unavailable — the
// merge then dedups against the working set only, same degradation as before.
export async function getTailSyncIndex() {
    const db = await openDB();
    const index = { ids: new Set(), stampSet: new Set(), stamps: new Map() };
    if (!db) return index;
    return new Promise((resolve) => {
        let req;
        try {
            req = db.transaction(STORE, 'readonly').objectStore(STORE).openCursor();
        } catch {
            resolve(index);
            return;
        }
        req.onsuccess = () => {
            const cursor = req.result;
            if (!cursor) { resolve(index); return; }
            const v = cursor.value;
            if (v && v.serverCreatedAt) {
                index.stampSet.add(v.serverCreatedAt);
                if (Number.isFinite(v.serverId)) index.ids.add(v.serverId);
                let list = index.stamps.get(v.serverCreatedAt);
                if (!list) { list = []; index.stamps.set(v.serverCreatedAt, list); }
                list.push({
                    problemId: v.problemId,
                    correct: !!v.correct,
                    isRepeat: !!v.isRepeat,
                    size: v.size,
                });
            }
            cursor.continue();
        };
        req.onerror = () => resolve(index);
    });
}

// Cursor walk for a model refit. `transform(entry)` may mutate and return the
// entry to persist the change, or return null/undefined to leave it untouched.
// Resolves true when the walk completes, false on unavailability/error.
export async function updateTail(transform) {
    const db = await openDB();
    if (!db) return false;
    return new Promise((resolve) => {
        let tx, req;
        try {
            tx = db.transaction(STORE, 'readwrite');
            req = tx.objectStore(STORE).openCursor();
        } catch {
            resolve(false);
            return;
        }
        req.onsuccess = () => {
            const cursor = req.result;
            if (!cursor) return; // walk done — tx.oncomplete resolves
            let updated;
            try { updated = transform(cursor.value); } catch { updated = null; }
            if (updated !== undefined && updated !== null) {
                try { cursor.update(updated); } catch {}
            }
            cursor.continue();
        };
        req.onerror = () => resolve(false);
        tx.oncomplete = () => resolve(true);
        tx.onabort = () => resolve(false);
    });
}

// Drop the whole tail (account switch). Resolves true on success.
export async function clearTail() {
    const db = await openDB();
    if (!db) return false;
    return new Promise((resolve) => {
        let tx;
        try {
            tx = db.transaction(STORE, 'readwrite');
            tx.objectStore(STORE).clear();
        } catch {
            resolve(false);
            return;
        }
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
        tx.onabort = () => resolve(false);
    });
}

// Replace the ENTIRE tail with `rows` (oldest→newest) in ONE transaction, so a
// crash can't leave it empty between a clear and a refill. Used by the one-time
// tail repair (de-dup + rating reconstruction). Resolves true on success, false
// on unavailability/error (caller should not treat the repair as done).
export async function replaceTail(rows) {
    const db = await openDB();
    if (!db) return false;
    return new Promise((resolve) => {
        let tx;
        try {
            tx = db.transaction(STORE, 'readwrite');
            const store = tx.objectStore(STORE);
            store.clear();
            for (const r of (rows || [])) {
                try { store.add(r); } catch {}
            }
        } catch {
            resolve(false);
            return;
        }
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
        tx.onabort = () => resolve(false);
    });
}
