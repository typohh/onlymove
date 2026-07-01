// js/net.js
//
// The one JSON-POST helper for the Worker's endpoints. Every write endpoint
// takes a JSON body; keeping the fetch boilerplate here stops the
// header/stringify pattern from drifting across call sites. Returns the raw
// Response (callers differ on how they read it) and throws on network failure
// exactly like fetch; extra fetch options (e.g. keepalive) ride via `init`.
export function postJson(url, body, init = {}) {
    return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        ...init,
    });
}
