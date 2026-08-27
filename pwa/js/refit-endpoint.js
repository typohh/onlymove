// js/refit-endpoint.js
//
// Batch fill of a history re-derive's per-attempt fields through POST /refit.
//
// What it replaces: the same fields used to come from a per-shard walk, and the
// per-cell byte offsets lived only inside each 2MB shard zip, so the walk paid a
// shard download per problem with almost nothing to amortise it over — a median
// 500-attempt window's wrong attempts spread across ~20 distinct shards. The
// Worker holds the lookup-aligned corpus sidecars resident and answers a whole
// batch from them, touching a shard bin only for `pickedEmbedding`.
//
// This is an ACCELERATOR, never a dependency. Every call is best-effort: on any
// failure (offline, 503 under the maintenance freeze, a data_version skew) it
// fills nothing and returns, and the existing sidecar + shard passes run behind
// it exactly as before and fill whatever is still missing. That is also why it
// runs FIRST — the passes behind it are already gated on the fields being
// absent, so anything filled here simply shrinks their work.

import { DATA_VERSION } from './data-version.js';
import { fetchLookupLines } from './problem-loader.js';

// Matches MAX_BATCH in src/handlers/refit.js. Batching is what makes the pass
// interruption-safe: the caller checkpoints per batch, so a kill costs one
// batch rather than the whole walk.
const BATCH = 200;

// Binary-search the id-sorted lookup lines ("id zipHash size") for a problem,
// returning the LINE INDEX — which is the problem's row address in every
// lookup-aligned sidecar, and the address /refit resolves against. Kept here
// rather than imported so this module stands alone if backfill's copy moves.
function lookupRow(lines, tsumegoId) {
    let lo = 0;
    let hi = lines.length - 1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const id = lines[mid].slice(0, 8);
        if (id === tsumegoId) return mid;
        if (id < tsumegoId) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

// Can this entry name the move that was played? Either it holds the variation
// index (recorded locally) or the board coordinate a sync delivered — the
// coordinate being what the record keeps, since reordering a problem's root
// variations re-points every index while leaving the coordinate correct.
function playedMove(a) {
    if (typeof a.pickedCellIndex === 'number' && a.pickedCellIndex >= 0) {
        return { picked: a.pickedCellIndex };
    }
    if (typeof a.syncedMove === 'string') return { move: a.syncedMove };
    return null;
}

// Does this entry still want anything /refit can supply?
function wants(a) {
    if (!a || typeof a.problemId !== 'string') return false;
    if (!a.embedding && !a.noEmb) return true;
    if (a.correct !== false) return false;
    const played = playedMove(a);
    if (!played) return false;
    // A synced entry with no index yet wants the resolution itself, which is
    // the whole reason the endpoint accepts a coordinate.
    if (played.move) return true;
    if (!a.pickedEmbedding) return true;
    // The availability null is referenced to the correct move, so it is only
    // requestable once the correct-move vector is in hand (or coming in the
    // same batch, which it is — both are filled from one response).
    if (a.altAbove === undefined) return true;
    return false;
}

// Apply one result slot. Only ever FILLS — never overwrites a value the device
// already holds, so a partially-filled entry converges and a re-run is a no-op.
function apply(a, r) {
    let touched = false;
    if (r.embedding && !a.embedding) { a.embedding = r.embedding; touched = true; }
    if (Number.isInteger(r.pickedCellIndex) && typeof a.pickedCellIndex !== 'number') {
        a.pickedCellIndex = r.pickedCellIndex;
        // The coordinate has done its job. Dropping it matches the shard walk's
        // one-shot rule and is what keeps a resolved entry from re-selecting its
        // shard; an entry the endpoint could NOT resolve keeps it, so the walk
        // still gets its turn.
        delete a.syncedMove;
        touched = true;
    }
    if (r.pickedEmbedding && !a.pickedEmbedding && a.correct === false) {
        a.pickedEmbedding = r.pickedEmbedding;
        touched = true;
    }
    if (r.altAbove && r.altBelow && a.altAbove === undefined && a.correct === false) {
        a.altAbove = r.altAbove;
        a.altBelow = r.altBelow;
        touched = true;
    }
    return touched;
}

// Fill what /refit can, for entries[startIdx..] of one size's history.
// `checkpoint` is invoked after each applied batch. Returns the number of
// entries touched; 0 (and no throw) whenever the endpoint is unusable.
export async function fillViaRefitEndpoint(history, startIdx, checkpoint = null) {
    const needy = [];
    for (let i = startIdx; i < history.length; i++) {
        if (wants(history[i])) needy.push(history[i]);
    }
    if (needy.length === 0) return 0;

    const lines = await fetchLookupLines();
    if (!lines) return 0;

    // Resolve every row up front so a problem that left the corpus costs no
    // request slot at all.
    const items = [];
    const owners = [];
    for (const a of needy) {
        const row = lookupRow(lines, a.problemId);
        if (row < 0) continue;
        const parts = lines[row].split(' ');
        const item = { id: a.problemId, row };
        const played = a.correct === false ? playedMove(a) : null;
        if (played) {
            if (played.picked !== undefined) item.picked = played.picked;
            else item.move = played.move;
            item.zip = parts[1];
            item.size = parseInt(parts[2], 10);
        }
        items.push(item);
        owners.push(a);
    }
    if (items.length === 0) return 0;

    let touched = 0;
    for (let start = 0; start < items.length; start += BATCH) {
        const slice = items.slice(start, start + BATCH);
        let payload;
        try {
            const resp = await fetch('/refit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataVersion: DATA_VERSION, problems: slice }),
            });
            // 409 is a data_version skew: this device's lookup.txt is from a
            // different packaging run, so every row address it just sent was
            // against the wrong table. Stop — the shard passes behind us read
            // their offsets from the shard itself and are unaffected.
            if (resp.status === 409) {
                console.warn('refit-endpoint: data_version skew, falling back to the shard walk');
                return touched;
            }
            if (!resp.ok) return touched;
            payload = await resp.json();
        } catch (_err) {
            // Offline, or the freeze returning 503 — indistinguishable and
            // handled the same way: leave it to the passes behind us.
            return touched;
        }
        const results = payload && Array.isArray(payload.results) ? payload.results : null;
        if (!results || results.length !== slice.length) return touched;

        let touchedHere = 0;
        for (let j = 0; j < results.length; j++) {
            if (results[j] && apply(owners[start + j], results[j])) touchedHere++;
        }
        touched += touchedHere;
        if (touchedHere > 0 && checkpoint) checkpoint();
    }
    return touched;
}
