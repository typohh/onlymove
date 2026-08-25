// js/identity-search.js
//
// One search box over BOTH rating databases, run from the browser.
//
// Client-side is not a preference: the Worker CANNOT reach EGD at all (it sits
// behind Cloudflare, and a Worker->CF-zone subrequest is blocked with HTTP 403
// "error code: 1106", the orange-to-orange restriction), while the page can — EGD
// sends `Access-Control-Allow-Origin: *`. The AGA Ratings Explorer sends it too,
// so searching AGA here as well is what makes ONE box possible; the Worker
// re-fetches the chosen AGA player itself at link time, where it is authoritative.
//
// A result is whatever the two APIs agree to call a player, normalized to:
//   { service:'EGD'|'AGA', id, name, loc, rank, rating, country, egd? }
// `loc` is a country code (EGD) or a US state (AGA), `rank` the database's own
// official grade, `rating` its GoR / AGA rating — all display only. `egd` carries
// the raw fields the GoR piggyback wants (see gor-refresh.js), present on EGD rows.
//
// Best-effort throughout: a database that errors or times out contributes nothing
// and the other's results still show. A caller gets an array, never a rejection.

const EGD = 'https://www.europeangodatabase.eu/EGD/';
const AGA = 'https://aga-ratings-explorer.azurewebsites.net/api/ratings-explorer/';
const TIMEOUT_MS = 8000;
// A `name` with no `lastname` is REFUSED ("you must specify the country or at
// least 3 characters of the last name"), so a given-name-only query needs a
// surname that matches everybody. `lastname` is a substring LIKE and honours SQL
// wildcards, and `_` survives the three-character check that `%` does not, so
// three of them read as "any surname of at least three letters". EGD cannot find
// a shorter surname by ANY query — the same check blocks searching for one
// directly — so this is its own floor, not one imposed here. Should the wildcard
// ever stop passing through, the set contributes nothing and the surname set
// still answers.
const ANY_SURNAME = '___';
// The Explorer pages its results; `status: all` + `recent_activity_years: none`
// keep retired/inactive members findable, since a player linking their account is
// looking for themselves, not for a currently-active opponent.
const AGA_FILTERS = { status: 'all', recent_activity_years: 'none', limit: '25', page: '0' };

async function fetchJson(url) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
        const res = await fetch(url, { signal: ctrl.signal });
        return res.ok ? await res.json() : null;
    } catch (_) {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

// One EGD player record (from either endpoint) -> a result row. `pinFallback`
// covers GetPlayerDataByPIN.php, whose payload is the player itself and omits
// Pin_Player on some records.
function mapEgd(p, pinFallback) {
    const gor = parseInt(p.Gor, 10);
    const pin = p.Pin_Player || pinFallback || '';
    return {
        service: 'EGD',
        id: pin,
        name: `${p.Name || ''} ${p.Last_Name || ''}`.trim(),
        loc: p.Country_Code || '',
        rank: p.Grade || '',
        rating: Number.isFinite(gor) ? gor : null,
        country: p.Country_Code || '',
        // Raw fields for the GoR piggyback: linking already fetched this player,
        // so the account's first weekly heartbeat needs no second EGD round trip.
        egd: {
            pin,
            gor: Number.isFinite(gor) ? gor : null,
            name: `${p.Name || ''} ${p.Last_Name || ''}`.trim(),
            country: p.Country_Code || '',
            grade: p.Grade || '',
            lastAppearance: p.Last_Appearance || '',
        },
    };
}

// EGD splits a name into `name` + `lastname` and matches each as a substring, so
// a query is tried with EACH TOKEN IN TURN AS THE SURNAME: nobody typing their
// own name knows which half the database considers the surname, and a middle name
// belongs to neither field — glued onto either one it matches nothing. Pairing is
// cyclic, so each set still carries a given name to narrow on and the request
// count stays one per token. Deduped by PIN, first hit wins.
async function egdByName(q) {
    const tokens = q.split(/\s+/);
    const sets = tokens.length <= 1
        ? [{ lastname: q }, { name: q, lastname: ANY_SURNAME }]
        : tokens.map((t, i) => ({ name: tokens[(i + 1) % tokens.length], lastname: t }));
    const seen = new Set();
    const out = [];
    for (const set of sets) {
        const d = await fetchJson(EGD + 'GetPlayerDataByData.php?' + new URLSearchParams(set));
        if (!d || d.retcode !== 'Ok' || !Array.isArray(d.players)) continue;
        for (const p of d.players) {
            if (p.Pin_Player && !seen.has(p.Pin_Player)) {
                seen.add(p.Pin_Player);
                out.push(mapEgd(p));
            }
        }
    }
    return out;
}

async function egdByPin(pin) {
    const d = await fetchJson(EGD + 'GetPlayerDataByPIN.php?pin=' + encodeURIComponent(pin));
    return (d && d.retcode === 'Ok') ? [mapEgd(d, pin)] : [];
}

function mapAga(p) {
    const rating = Number(p.rating);
    return {
        service: 'AGA',
        id: String(p.agaid || ''),
        name: p.display_name || `${p.first_name || ''} ${p.last_name || ''}`.trim(),
        loc: p.state || '',                 // a US state, not a country code
        rank: p.rank_label || '',
        rating: Number.isFinite(rating) ? rating : null,
        country: '',                        // AGA has none; the geo-IP flag stands in
    };
}

// Same each-token-as-the-surname treatment as EGD, over the Explorer's first/last
// split. A lone token needs no wildcard here: unlike EGD, the Explorer accepts a
// `first_name` on its own.
async function agaByName(q) {
    const tokens = q.split(/\s+/);
    const sets = tokens.length <= 1
        ? [{ last_name: q }, { first_name: q }]
        : tokens.map((t, i) => ({ first_name: tokens[(i + 1) % tokens.length], last_name: t }));
    const seen = new Set();
    const out = [];
    for (const set of sets) {
        const d = await fetchJson(AGA + 'players?' + new URLSearchParams({ ...AGA_FILTERS, ...set }));
        for (const p of ((d && d.results) || [])) {
            if (p.agaid && !seen.has(p.agaid)) {
                seen.add(p.agaid);
                out.push(mapAga(p));
            }
        }
    }
    return out;
}

async function agaById(id) {
    const d = await fetchJson(AGA + 'player?agaid=' + encodeURIComponent(id));
    return (d && d.player) ? [mapAga(d.player)] : [];
}

/**
 * Search both databases for `query` and return the merged, source-labelled
 * results (EGD first, then AGA). An all-digits query is ambiguous — it could be
 * an EGD PIN or an AGA member id — so it is looked up as both. Never rejects;
 * a database that fails simply contributes nothing.
 */
export async function searchIdentities(query) {
    const q = (query || '').trim();
    if (!q) return [];
    const numeric = /^\d+$/.test(q);
    const [egd, aga] = await Promise.all(
        numeric ? [egdByPin(q), agaById(q)] : [egdByName(q), agaByName(q)]
    );
    return egd.concat(aga);
}
