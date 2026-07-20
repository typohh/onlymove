// pwa/src/aga.js
//
// AGA (American Go Association) rating lookup, server-side. Source is the AGA
// Ratings Explorer (the current official ratings app embedded on usgo.org/ratings):
//   https://aga-ratings-explorer.azurewebsites.net/api/ratings-explorer
// It is a JSON API on Azure (NOT Cloudflare), so a Worker->origin fetch reaches it
// (no orange-to-orange block like EGD), and it sends Access-Control-Allow-Origin: *
// so the browser can hit it too (the invite tool searches it client-side).
//
// Endpoints used:
//   GET /players?last_name=&first_name=&agaid=&status=all&recent_activity_years=none&limit=25&page=0
//       -> { results: [ { agaid, display_name, rating, sigma, rank_label, state, ... } ] }
//   GET /player?agaid=<id>  -> { player: { agaid, rating, rank_label, ... } }
//
// The `rank_label` ("10 dan", "3 kyu") is the AGA's OWN official rank — dan = floor(rating),
// kyu = floor(|rating|), with a gap around 0 that no linear rating->rank offset can model.
// So placement uses the label (via the AGA coupling), never a computed offset. This module
// returns the SHORT label ("10d"/"3k") that pwa/shared/rank-math.js rankLabelToScalar parses.
//
// Best-effort: any network error / bad payload returns null, never throws.

const AGA_API = 'https://aga-ratings-explorer.azurewebsites.net/api/ratings-explorer';
const FETCH_TIMEOUT_MS = 6000;

// True for a well-formed AGA id (positive integer, the AGAID path/query value).
export function isValidAgaId(id) {
    return typeof id === 'string' ? /^\d{1,12}$/.test(id) : (Number.isInteger(id) && id > 0 && id < 1e12);
}

// AGA rank_label ("10 dan" / "3 kyu") -> the short "10d" / "3k" form rankLabelToScalar
// parses. Returns null if it isn't a recognizable dan/kyu label.
export function agaShortRankLabel(rankLabel) {
    if (typeof rankLabel !== 'string') return null;
    const m = rankLabel.trim().toLowerCase().match(/^(\d{1,2})\s*(dan|kyu|d|k)\b/);
    if (!m) return null;
    return m[1] + (m[2][0] === 'd' ? 'd' : 'k');
}

async function fetchJson(url) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
        const res = await fetch(url, { signal: ctrl.signal });
        if (!res || !res.ok) return null;
        return await res.json();
    } catch (e) {
        return null;                      // network error / abort / bad JSON
    } finally {
        clearTimeout(timer);
    }
}

// Fetch a member's current AGA rating + official rank by id. Returns
// { rating, rankLabel } (rankLabel = the SHORT "10d"/"3k" form) or null on any failure.
export async function fetchAgaPlayer(agaId) {
    const id = String(agaId || '');
    if (!isValidAgaId(id)) return null;
    const data = await fetchJson(`${AGA_API}/player?agaid=${encodeURIComponent(id)}`);
    const p = data && data.player ? data.player : null;
    if (!p) return null;
    const rating = Number(p.rating);
    const rankLabel = agaShortRankLabel(p.rank_label);
    if (!rankLabel) return null;          // no usable rank -> treat as absent
    return { rating: Number.isFinite(rating) ? rating : null, rankLabel };
}
