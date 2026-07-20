// pwa/src/aga.js
//
// Server-side AGA (American Go Association) rating fetch — the AGA counterpart to
// the EGD GoR piggyback, but WORKER-SIDE. The AGAGD exposes a public JSON rating
// history at
//   GET https://agagd.usgo.org/api/ratings/<player_id>/<time_period>/
// returning an array of { elab_date, rating, sigma } (time_period in years, use 1).
// The most recent entry is the player's current AGA rating.
//
// Why Worker-side (unlike EGD's client piggyback): the AGAGD api sends no CORS
// headers, so a browser cross-origin fetch is blocked — but a Worker->origin fetch
// has no CORS constraint. The only risk is a Cloudflare orange-to-orange block (the
// exact reason the Worker can't reach EGD); it is unverifiable until deploy, so
// every caller treats a null return as "keep the last known snapshot" — AGA linking
// still works from the invite-time snapshot even if this never succeeds.
//
// Best-effort throughout: any network error / bad payload / implausible value
// returns null, never throws.

const AGA_RATINGS_URL = 'https://agagd.usgo.org/api/ratings/'; // + '<id>/1/'
const FETCH_TIMEOUT_MS = 5000;
// Plausibility bound on an AGA rating (kyu negative, dan positive, ~1/stone):
// roughly 50k .. 9d with headroom. Rejects garbage / a changed payload shape.
const AGA_RATING_MIN = -60, AGA_RATING_MAX = 12;

// True for a well-formed AGA member id (positive integer, as the AGAGD path
// segment). Kept lax on length — member ids are small but not fixed-width.
export function isValidAgaId(id) {
    return typeof id === 'string' ? /^\d{1,12}$/.test(id) : (Number.isInteger(id) && id > 0 && id < 1e12);
}

// Fetch a member's CURRENT AGA rating (the latest elab_date), or null on any
// failure (unreachable Worker->AGAGD, bad JSON, empty history, implausible value).
export async function fetchAgaRating(agaId) {
    const id = String(agaId || '');
    if (!isValidAgaId(id)) return null;
    let res;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
        res = await fetch(`${AGA_RATINGS_URL}${encodeURIComponent(id)}/1/`, { signal: ctrl.signal });
    } catch (e) {
        return null;                      // network error / abort / CF block
    } finally {
        clearTimeout(timer);
    }
    if (!res || !res.ok) return null;
    let data;
    try { data = await res.json(); } catch (e) { return null; }
    // The api returns an array of { elab_date, rating, sigma }; tolerate a single
    // object too. Pick the entry with the newest elab_date.
    const rows = Array.isArray(data) ? data : (data && typeof data === 'object' ? [data] : []);
    let best = null;
    for (const r of rows) {
        if (!r || r.rating === null || r.rating === undefined) continue;
        const rating = Number(r.rating);
        if (!Number.isFinite(rating)) continue;
        const when = String(r.elab_date || '');
        if (!best || when > best.when) best = { rating, when };
    }
    if (!best) return null;
    if (best.rating < AGA_RATING_MIN || best.rating > AGA_RATING_MAX) return null;
    return best.rating;
}
