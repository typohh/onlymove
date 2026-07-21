// js/gor-refresh.js
//
// Client-side GoR refresh for the EGD leaderboard. The Worker cannot fetch EGD (it
// sits behind Cloudflare; a Worker→CF-zone subrequest is blocked with HTTP 403
// "error code: 1106" — the orange-to-orange restriction). GoR is public (keyed on
// the EGD PIN, no user secret), so any active client fetches registered players'
// current GoR from EGD directly (CORS *) and piggybacks the results back on its next
// /report. The /report response advertises which PINs are still stale this week
// (`refreshPins`); the Worker accepts each PIN's update once per week.
//
// Best-effort throughout: a failed EGD fetch just leaves that PIN stale for the next
// client to retry, so nothing is ever permanently lost.

const EGD_PIN_URL = 'https://www.europeangodatabase.eu/EGD/GetPlayerDataByPIN.php?pin=';
const MAX_PER_RESPONSE = 20;   // cap EGD fetches kicked off per report response

let pending = [];              // fetched {pin, gor, name, country, grade, lastAppearance} awaiting the next report
const inFlight = new Set();    // PINs currently being fetched (dedupe within a session)

// Take + clear the buffered updates to attach to a /report request body. Returns
// null when there's nothing to send (so the report body stays unchanged).
export function takePendingGorUpdates() {
    if (!pending.length) return null;
    const out = pending;
    pending = [];
    return out;
}

// Given the server's `refreshPins`, fetch each stale PIN's current GoR from EGD and
// buffer it for the next report. Non-blocking; skips PINs already in flight or
// already buffered, and caps how many it starts per response.
export function requestGorRefresh(pins) {
    if (!Array.isArray(pins) || !pins.length) return;
    const toFetch = pins
        .map(String)
        .filter((k) => /^\d+$/.test(k) && !inFlight.has(k) && !pending.some((u) => String(u.pin) === k))
        .slice(0, MAX_PER_RESPONSE);
    for (const pin of toFetch) {
        inFlight.add(pin);
        fetchOne(pin).finally(() => inFlight.delete(pin));
    }
}

async function fetchOne(pin) {
    try {
        const res = await fetch(EGD_PIN_URL + encodeURIComponent(pin));
        if (!res.ok) return;
        const d = await res.json();
        if (!d || d.retcode !== 'Ok') return;
        const gor = parseInt(d.Gor, 10);
        if (!Number.isFinite(gor)) return;
        const name = `${d.Name || ''} ${d.Last_Name || ''}`.trim();
        // Grade is EGD's own official rank ("4d"/"7k"/"3p"); Last_Appearance encodes
        // the last tournament date ("T220122H" -> 2022-01-22). Both are passed raw —
        // the Worker validates/parses (applyGorUpdates) before storing.
        pending.push({
            pin, gor, name, country: d.Country_Code || '',
            grade: d.Grade || '', lastAppearance: d.Last_Appearance || '',
        });
    } catch (e) {
        /* leave this PIN stale; the next client re-advertises + retries */
    }
}
