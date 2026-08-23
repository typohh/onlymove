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
//     — no synthetic key field leaks back into the history array. The one
//     exception is the embedding block, dropped on write (see stripEmbeddings).
//   - THE TAIL HOLDS NO EMBEDDING VECTORS, enforced here at the write boundary
//     rather than assumed by the callers. A tail row is by definition older
//     than the working set, hence far outside MODEL_INPUT_HISTORY_LENGTH, so
//     no model or fill will ever read or refill its vectors — they are dead
//     weight in IndexedDB, and worse, they are frozen at the embedding_version
//     of the day they were evicted, while the only pass that clears stale
//     vectors (backfill's refit wipe) can reach the working set alone. Left in,
//     they outlive every refit and reappear as wrong-K rows in any consumer
//     that reads the full career.
//   - Insertion order == eviction order == date order (state.evictColdTail
//     always appends the OLDEST overflow first), so getTail() returns the tail
//     oldest→newest with no index needed.
//   - Every method is defensive: if IndexedDB is missing (private mode, old
//     browser) or a transaction errors, reads resolve to [] and writes resolve
//     to false. A false from appendTail tells the caller NOT to trim the
//     working set, so an unavailable tail degrades to "charts show the working
//     set only" rather than losing attempts.

import { rowHash } from '../shared/sync-digest.js';
import { EMBEDDING_ALL_FIELDS } from './constants.js';

const DB_NAME = 'onlymove';
const DB_VERSION = 1;
const STORE = 'answerHistoryTail';

let dbPromise = null;

// A copy of `row` without the embedding block, or `row` itself when it carries
// none (the steady state — nothing to copy). Never mutates the caller's object:
// appendTail's contract is that a FAILED write leaves the working set intact,
// and evictColdTail keeps those rows on a false.
function stripEmbeddings(row) {
    if (!row || !EMBEDDING_ALL_FIELDS.some((f) => row[f] !== undefined)) return row;
    const out = { ...row };
    for (const f of EMBEDDING_ALL_FIELDS) delete out[f];
    return out;
}

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
    invalidateTailDigest();
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
            try { store.add(stripEmbeddings(r)); } catch {}
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

// Acked = carries a serverId (the row id /report or /sync stamped; identity
// and digest key). Mirror-by-id is the only representation — every acked
// entry has an id.
function isAcked(v) {
    return !!v && Number.isFinite(v.serverId);
}

// Cached tail digest: the tail changes only through this module's writers
// (appendTail / updateTail / pruneTail / clearTail / replaceTail), each of
// which invalidates the cache — so the per-report digest check reads it
// without an IDB cursor walk.
let tailDigestCache = null;
function invalidateTailDigest() { tailDigestCache = null; }

// One cold-tail scan for the sync digest: how many tail rows are acked and the
// Σ rowHash(serverId) over them. Defensive: empty info on any error (an
// unavailable tail contributes nothing, same degradation as every reader).
export async function getTailDigestInfo() {
    if (tailDigestCache) return tailDigestCache;
    const db = await openDB();
    const info = { ackedCount: 0, idSum: 0 };
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
            if (!cursor) { tailDigestCache = info; resolve(info); return; }
            const v = cursor.value;
            if (isAcked(v)) {
                info.ackedCount++;
                info.idSum += rowHash(v.serverId);
            }
            cursor.continue();
        };
        req.onerror = () => resolve(info);
    });
}

// The acked row ids in the tail — the sync merge's id-first dedup. Empty set
// if IDB is unavailable (the merge then dedups against the working set only).
export async function getTailSyncIndex() {
    const db = await openDB();
    const index = { ids: new Set() };
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
            if (isAcked(v)) index.ids.add(v.serverId);
            cursor.continue();
        };
        req.onerror = () => resolve(index);
    });
}

// Delete every tail row for which `predicate(row)` is true, in one
// transaction. The mirror-complete reconcile's deletion pass uses this to drop
// acked rows the server no longer has, and to collapse evict-seam duplicates
// (a row present in both the working set and the tail after a crash
// mid-evict). Resolves the number of rows deleted, or false on
// unavailability/error.
export async function pruneTail(predicate) {
    const db = await openDB();
    if (!db) return false;
    invalidateTailDigest();
    return new Promise((resolve) => {
        let tx, req, deleted = 0;
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
            let hit = false;
            try { hit = !!predicate(cursor.value); } catch { hit = false; }
            if (hit) {
                try { cursor.delete(); deleted++; } catch {}
            }
            cursor.continue();
        };
        req.onerror = () => resolve(false);
        tx.oncomplete = () => resolve(deleted);
        tx.onabort = () => resolve(false);
    });
}

// Cursor walk for a model refit. `transform(entry)` may mutate and return the
// entry to persist the change, or return null/undefined to leave it untouched.
// Resolves true when the walk completes, false on unavailability/error.
export async function updateTail(transform) {
    invalidateTailDigest();
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

// Clear the embedding block from rows already stored. Two callers, both in
// backfill.js: the refit sweep (every embedding_version change, so the
// no-vectors invariant has a REPAIR and not only the assertion above) and the
// one-time legacy repair for tails written before that assertion existed.
// Resolves { cleared, ok }; ok=false (IDB unavailable, walk failed) means the
// caller must not record a repair as done.
export async function dropStoredEmbeddings() {
    let cleared = 0;
    const ok = await updateTail((entry) => {
        const stripped = stripEmbeddings(entry);
        if (stripped === entry) return null;   // carried none — leave untouched
        cleared++;
        return stripped;
    });
    return { cleared, ok: !!ok };
}

// Drop the whole tail (account switch). Resolves true on success.
export async function clearTail() {
    invalidateTailDigest();
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
    invalidateTailDigest();
    const db = await openDB();
    if (!db) return false;
    return new Promise((resolve) => {
        let tx;
        try {
            tx = db.transaction(STORE, 'readwrite');
            const store = tx.objectStore(STORE);
            store.clear();
            for (const r of (rows || [])) {
                try { store.add(stripEmbeddings(r)); } catch {}
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
