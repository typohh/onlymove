// src/handlers/refit.js
//
// POST /refit — everything a history re-derive needs for a batch of problems,
// in one request.
//
// Why this exists: the per-cell byte offsets used to live only in
// embeddings_index.json INSIDE each 2MB shard zip, so filling `pickedEmbedding`
// and the availability null meant downloading a shard per problem. There is
// almost no locality to amortise that over — a median 500-attempt window's wrong
// attempts spread across ~20 distinct shards, ~1.15 problems per shard — so the
// walk cost tens of megabytes to deliver a few kilobytes of vectors.
//
// Three of the four fields are now lookup-aligned corpus sidecars this Worker
// holds resident (~8.8MB across the correct-move vectors, the availability-null
// counts, the block offsets and problem_params). Only `pickedEmbedding` is
// user-dependent, so only it touches a shard bin — and only for wrong attempts.
//
// The client sends the lookup ROW it already resolved (it binary-searches
// lookup.txt in every flow that reaches here), which is what keeps this Worker
// off a 215k-entry id map. `dataVersion` is the guard that makes that safe: the
// row index is an address into THIS packaging run's sidecars, so a client whose
// lookup.txt came from a different run must be refused rather than served
// another problem's vectors. Everything here is a read of published assets —
// no D1, and nothing is recorded.

import { CORS_HEADERS, jsonResponse } from '../http.js';
import EMBEDDING_META from '../../tsumegos/embedding_meta.json';
import { getProblemParams } from '../leaderboard.js';

const K = EMBEDDING_META.K || 0;
const BYTES_PER_CELL = EMBEDDING_META.bytes_per_cell || K * 2;
const OFFSETS_ROW = EMBEDDING_META.offsets_bytes_per_row || 5;
const ALTSTATS_ROW = EMBEDDING_META.altstats_bytes_per_row || 2 * K;
const DATA_VERSION = EMBEDDING_META.data_version || null;

// A batch is capped so one request stays a bounded unit of work: the client
// pages and checkpoints per batch anyway (a re-derive is a network walk that
// gets killed routinely), and the cap bounds how many distinct shard bins one
// invocation can be made to pull.
const MAX_BATCH = 200;

// --- resident lookup-aligned sidecars ---------------------------------------
// Loaded once per isolate, like problem_params: they are immutable for a
// deploy's lifetime, and every deploy resets isolates, so a cold isolate
// reloads rather than ever serving a previous run's rows.
const SIDECARS = {
    embeddings: 'lookup.embeddings.bin',
    altstats: 'lookup.altstats.bin',
    offsets: 'lookup.offsets.bin',
};
const _resident = {};
const _residentPromises = {};

async function loadSidecar(env, key) {
    const res = await env.ASSETS.fetch(
        new Request(`https://onlymove.net/tsumegos/${SIDECARS[key]}`));
    if (!res || !res.ok) throw new Error(`${SIDECARS[key]} unavailable`);
    return new Uint8Array(await res.arrayBuffer());
}

function sidecar(env, key) {
    if (_resident[key]) return Promise.resolve(_resident[key]);
    if (!_residentPromises[key]) {
        _residentPromises[key] = loadSidecar(env, key)
            .then((b) => { _resident[key] = b; return b; })
            .catch((e) => { _residentPromises[key] = null; throw e; });
    }
    return _residentPromises[key];
}

// --- picked cells: cache the CELLS, never the shard bins --------------------
// The full per-cell space is ~156MB against a 128MB isolate, so shard bins
// cannot be held. But the space this endpoint is ever asked about is not the
// full one: only cells a player actually picked reach a refit, and across the
// whole attempt log that is ~39k distinct (problem, cell) pairs — a mean of
// 1.15 picked cells per failed problem. Ten bytes each, so the working set is
// well under a megabyte of vectors.
//
// So a shard bin is read, the cells this batch wants are lifted out of it, and
// the bin is DROPPED. What persists is the cells. A bin LRU would be the wrong
// structure at these ratios: a 200-problem batch spans ~173 distinct shards, so
// bins almost never repeat and holding 241KB to serve 10 bytes buys nothing.
//
// Self-warming rather than precomputed, deliberately: a table built from a DB
// snapshot at packaging time would couple packaging to the attempt log AND go
// stale for every move picked after it, while this converges on the same set
// and stays current. Cold isolates simply re-read a few shards.
const CELL_CACHE_MAX = 120000;   // ~3x the observed corpus-wide picked set
const _cellCache = new Map();    // `${size}/${zip}/${byteOffset}` -> vector

function cacheCell(key, vec) {
    _cellCache.set(key, vec);
    // Insertion-ordered eviction: at 3x headroom this effectively never fires,
    // and it exists to bound a pathological caller rather than to manage a hot
    // set that does not exist.
    while (_cellCache.size > CELL_CACHE_MAX) {
        _cellCache.delete(_cellCache.keys().next().value);
    }
}

async function shardBin(env, size, zipHash) {
    const res = await env.ASSETS.fetch(
        new Request(`https://onlymove.net/tsumegos/${size}/${zipHash}.embeddings.bin`));
    if (!res || !res.ok) return null;
    return new Uint8Array(await res.arrayBuffer());
}

// --- decoding ---------------------------------------------------------------
// IEEE-754 half precision. Mirrors fp16ToFloat32 in js/picked-embedding.js;
// the two must agree because the client compares vectors this Worker produced
// against vectors it decoded itself.
function fp16ToFloat32(u16) {
    const sign = (u16 & 0x8000) >> 15;
    const exp = (u16 & 0x7C00) >> 10;
    const frac = u16 & 0x03FF;
    if (exp === 0) {
        const v = Math.pow(2, -14) * (frac / 1024);
        return sign ? -v : v;
    }
    if (exp === 0x1F) return frac === 0 ? (sign ? -Infinity : Infinity) : NaN;
    const v = Math.pow(2, exp - 15) * (1 + frac / 1024);
    return sign ? -v : v;
}

// quantizeEmbedding (js/math-utils.js) — the resolution every stored vector is
// held at, applied here so the client stores exactly what it would have
// computed and tie-resolution downstream is unchanged.
function decodeVec(bytes, byteOffset) {
    if (byteOffset < 0 || byteOffset + BYTES_PER_CELL > bytes.length) return null;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Array(K);
    for (let k = 0; k < K; k++) {
        out[k] = Math.round(fp16ToFloat32(dv.getUint16(byteOffset + k * 2, true)) * 100) / 100;
    }
    return out;
}

export async function handleRefit(request, env) {
    if (!K) return jsonResponse({ error: 'no embedding metadata deployed' }, 503);

    let body;
    try {
        body = await request.json();
    } catch {
        return jsonResponse({ error: 'malformed JSON body' }, 400);
    }

    // Row indices are addresses into THIS run's sidecars. A client carrying a
    // different packaging run's lookup.txt would silently read a stranger's
    // vectors, so the mismatch is refused and named — the client re-pulls
    // lookup.txt and retries rather than storing garbage.
    if (DATA_VERSION && body.dataVersion && body.dataVersion !== DATA_VERSION) {
        return jsonResponse({ error: 'data_version mismatch', dataVersion: DATA_VERSION }, 409);
    }

    const items = Array.isArray(body.problems) ? body.problems : null;
    if (!items) return jsonResponse({ error: 'problems must be an array' }, 400);
    if (items.length > MAX_BATCH) {
        return jsonResponse({ error: `batch exceeds ${MAX_BATCH}`, maxBatch: MAX_BATCH }, 400);
    }

    let embeddings; let altstats; let offsets; let params;
    try {
        [embeddings, altstats, offsets, params] = await Promise.all([
            sidecar(env, 'embeddings'), sidecar(env, 'altstats'),
            sidecar(env, 'offsets'), getProblemParams(env),
        ]);
    } catch (e) {
        console.error('/refit sidecar load failed:', e);
        return jsonResponse({ error: 'corpus sidecars unavailable' }, 503);
    }

    const nRows = Math.floor(embeddings.length / BYTES_PER_CELL);
    const offsetsView = new DataView(offsets.buffer, offsets.byteOffset, offsets.byteLength);

    // Results are ALIGNED TO THE REQUEST ARRAY, not keyed by problem id: two
    // attempts on one problem can have picked different cells, so an id-keyed
    // map would collide on exactly the field that differs between them. A slot
    // the Worker could not resolve is null.
    const byShard = new Map();
    const results = new Array(items.length).fill(null);

    for (let i = 0; i < items.length; i++) {
        const it = items[i];
        if (!it || typeof it.id !== 'string' || !/^[0-9a-f]{8}$/.test(it.id)) continue;
        const row = it.row;
        if (!Number.isInteger(row) || row < 0 || row >= nRows) continue;

        const out = {};
        const emb = decodeVec(embeddings, row * BYTES_PER_CELL);
        // An all-zero row is packaging's placeholder for an id with no
        // embedding — absent, not a vector of zeros.
        if (emb && !emb.every((v) => v === 0)) out.embedding = emb;

        const aOff = row * ALTSTATS_ROW;
        if (aOff + ALTSTATS_ROW <= altstats.length) {
            const above = []; const below = [];
            for (let k = 0; k < K; k++) {
                above.push(altstats[aOff + k]);
                below.push(altstats[aOff + K + k]);
            }
            // altAbove + altBelow == 0 across every axis is the absent sentinel
            // (it already fails the client's nAvail guard) — omit rather than
            // ship counts that say nothing.
            if (above.some((v, k) => v + below[k] > 0)) {
                out.altAbove = above;
                out.altBelow = below;
            }
        }

        const p = params.get(it.id);
        if (p) { out.rating = p.rating; out.slope = p.slope; out.floor = p.floor; }

        results[i] = out;

        // The one user-dependent field, and the only one that touches a shard.
        if (Number.isInteger(it.picked) && it.picked >= 0
            && typeof it.zip === 'string' && /^[0-9a-f]{8}$/.test(it.zip)
            && (it.size === 9 || it.size === 13)) {
            const blockOffset = offsetsView.getUint32(row * OFFSETS_ROW, true);
            const nCells = offsetsView.getUint8(row * OFFSETS_ROW + 4);
            if (nCells > 0 && it.picked < nCells) {
                const key = `${it.size}/${it.zip}`;
                if (!byShard.has(key)) byShard.set(key, []);
                byShard.get(key).push({
                    slot: i,
                    byteOffset: blockOffset + it.picked * BYTES_PER_CELL,
                });
            }
        }
    }

    for (const [key, wanted] of byShard) {
        // Anything already cached needs no shard read at all; only a genuinely
        // new pick does, which is why the shard fetch is inside the miss check.
        const misses = [];
        for (const w of wanted) {
            const hit = _cellCache.get(`${key}/${w.byteOffset}`);
            if (hit) results[w.slot].pickedEmbedding = hit;
            else misses.push(w);
        }
        if (misses.length === 0) continue;
        const [size, zipHash] = key.split('/');
        const bin = await shardBin(env, size, zipHash);
        if (!bin) continue;
        for (const w of misses) {
            const vec = decodeVec(bin, w.byteOffset);
            // An all-zero cell is the pass placeholder, which no click can
            // select — don't cache it and don't answer with it.
            if (vec && !vec.every((v) => v === 0)) {
                results[w.slot].pickedEmbedding = vec;
                cacheCell(`${key}/${w.byteOffset}`, vec);
            }
        }
        // `bin` goes out of scope here — the cells are what we keep.
    }

    return new Response(JSON.stringify({
        K,
        dataVersion: DATA_VERSION,
        embeddingVersion: EMBEDDING_META.embedding_version || null,
        ratingsVersion: EMBEDDING_META.ratings_version || null,
        results,
    }), {
        headers: {
            ...CORS_HEADERS,
            'Content-Type': 'application/json',
            // Per-user request shape, but the payload is pure published corpus
            // data — safe to hold briefly, never revalidated against D1.
            'Cache-Control': 'private, max-age=300',
        },
    });
}
