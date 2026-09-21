// pwa/src/handlers/alias.js
//
// POST /profile/alias — the account's SELF-CHOSEN leaderboard alias, set,
// replaced or cleared. One route, like /profile/hide: a `clear: true` body
// removes the row, a triplet body writes one.
//
//   POST /profile/alias  { userId, adjective, color, animal }
//       -> { ok: true, alias: { adjective, color, animal } }
//   POST /profile/alias  { userId, clear: true }
//       -> { ok: true, alias: null }
//   Rejections: 400 bad_request | 400 invalid_alias (a word is not in its list)
//               409 alias_taken (another account holds this triplet)
//
// The words are matched EXACTLY against the English names in the three D1 word
// tables (src/aliases.js). Translations never travel over the wire in either
// direction: the client offers its own localized labels, but what it submits —
// and what comes back — is the canonical English triplet, because that is what
// the UNIQUE index is on and what the leaderboard re-translates per viewer.
//
// Nothing here knows whether the account has linked an EGD/AGA identity: an
// alias is allowed either way and simply is not shown while a linked name is.
// Keeping the choice is the point — unlinking restores it.
//
// Format-validated before any D1 write, like every other write handler, and
// rate-limited by RL_WRITE at the route (src/index.js).

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import { canonicalAlias, setAlias, clearAlias } from '../aliases.js';

// Mirrors the longest word in shared/alias-words.js with headroom; anything
// longer cannot be a legal word, so it is rejected without touching D1.
const MAX_WORD_LEN = 24;

export async function handleAlias(request, env) {
    let data;
    try { data = await request.json(); } catch { return jsonResponse({ ok: false, error: 'bad_request' }, 400); }
    if (!isValidUserId(data?.userId)) return jsonResponse({ ok: false, error: 'bad_request' }, 400);
    if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);

    if (data.clear === true) {
        try {
            await clearAlias(env, data.userId);
            return jsonResponse({ ok: true, alias: null });
        } catch (e) {
            console.error('handleAlias clear failed:', e);
            return jsonResponse({ ok: false, error: 'server' }, 500);
        }
    }

    const words = [data.adjective, data.color, data.animal];
    if (!words.every((w) => typeof w === 'string' && w.length > 0 && w.length <= MAX_WORD_LEN)) {
        return jsonResponse({ ok: false, error: 'invalid_alias' }, 400);
    }

    try {
        // Legality first (one round trip), then the write; the UNIQUE index is
        // what settles a same-triplet race between two accounts.
        const canonical = await canonicalAlias(env, data.adjective, data.color, data.animal);
        if (!canonical) return jsonResponse({ ok: false, error: 'invalid_alias' }, 400);
        const result = await setAlias(env, data.userId, canonical);
        if (result === 'taken') return jsonResponse({ ok: false, error: 'alias_taken' }, 409);
        if (result !== 'ok') return jsonResponse({ ok: false, error: 'server' }, 500);
        return jsonResponse({ ok: true, alias: canonical });
    } catch (e) {
        console.error('handleAlias failed:', e);
        return jsonResponse({ ok: false, error: 'server' }, 500);
    }
}
