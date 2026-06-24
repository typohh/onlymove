// js/persist.js
//
// Tiny localStorage JSON helpers, shared so the parse + quota handling can't
// drift across state.js's call sites (the per-size buckets, the shared-value
// store, and the boot migrations all read/write JSON the same way).

export function isQuotaError(e) {
    return e && (
        e.name === 'QuotaExceededError' ||
        e.name === 'NS_ERROR_DOM_QUOTA_REACHED' ||
        e.code === 22 || e.code === 1014
    );
}

// Parse a JSON value, returning `fallback` when the key is absent, holds JSON
// null, or fails to parse.
export function readJSON(key, fallback = null) {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    try {
        const parsed = JSON.parse(raw);
        return parsed === null ? fallback : parsed;
    } catch { return fallback; }
}

// Parse a JSON *object* (the per-size / bucket shape), returning {} for an
// absent key, a parse failure, or a non-plain-object value (array or scalar).
export function readJSONObject(key) {
    const raw = localStorage.getItem(key);
    if (raw === null) return {};
    try {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    } catch {}
    return {};
}

// Write a JSON value. A quota error is logged and skipped (the write is lost,
// not fatal). By default any other error is also logged+swallowed; pass
// `rethrowNonQuota` to re-raise non-quota errors — the per-size buckets do, so a
// genuine serialization bug surfaces instead of being silently dropped.
export function writeJSON(key, value, { rethrowNonQuota = false } = {}) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        if (rethrowNonQuota && !isQuotaError(e)) throw e;
        console.error(`localStorage write failed for ${key}`, e);
    }
}
