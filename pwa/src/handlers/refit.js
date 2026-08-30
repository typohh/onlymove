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
// holds resident (the correct-move vectors, the availability null, the block
// offsets, plus problem_params). Only `pickedEmbedding` is user-dependent, so
// only it touches a shard bin — and only for wrong attempts.
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
// The availability null ships K above-masses then K below-masses as
// little-endian uint32 — MASSES, not counts: packaging weighs each alternative
// by the visits its evaluation rests on (a comment with no third field is a
// 30-visit value by convention), and a tie drops with its weight rather than
// being split. They are integer sums of integer visit counts, so this Worker and
// the client's own live-path computation agree exactly. The reader-side
// semantic is unchanged: p = above / (above + below).
const ALTSTATS_BYTES = 4;
const ALTSTATS_ROW = EMBEDDING_META.altstats_bytes_per_row || 2 * K * ALTSTATS_BYTES;
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

async function shardAsset(env, size, zipHash, suffix) {
    const res = await env.ASSETS.fetch(
        new Request(`https://onlymove.net/tsumegos/${size}/${zipHash}.${suffix}`));
    if (!res || !res.ok) return null;
    return new Uint8Array(await res.arrayBuffer());
}

// --- resolving a reported move to a variation index ------------------------
// An attempt that arrived from another device carries the played move as a
// board coordinate, not as an index — the coordinate is what the record keeps,
// because reordering a problem's root variations re-points every index while
// leaving the coordinate correct. Turning it back into an index needs the
// problem's variations in order, which is why the cell map is published: one
// byte per cell, `y * size + x`, running in lockstep with the embedding block
// so a problem's codes start at offset / bytes_per_cell and run n_cells bytes.
// Without it this resolution needs the SGF, hence the whole shard archive.
const PASS_POINT_CODE = 255;
const COL_LETTERS = 'ABCDEFGHJKLMNOPQRST';   // no I, as in the client's sgf.js

// Board notation -> point code. Mirrors boardNotationToXy in pwa/js/sgf.js and
// must keep mirroring it: the coordinate is minted there, in the canonical
// orientation-0 frame the published SGFs are written in.
function movePointCode(move, size) {
    if (typeof move !== 'string') return -1;
    const m = /^([A-HJ-T])(\d{1,2})$/.exec(move.toUpperCase());
    if (!m) return -1;
    const x = COL_LETTERS.indexOf(m[1]);
    const y = size - parseInt(m[2], 10);
    if (x < 0 || x >= size || y < 0 || y >= size) return -1;
    return y * size + x;
}

// Resolved indices are cached for the same reason picked cells are: the set of
// (problem, played move) pairs a re-derive asks about is small and repeats
// across users, so the map is read once and the answers persist.
const _indexCache = new Map();   // `${size}/${zip}/${row}/${code}` -> index

function lookupIndexInMap(map, size, zipHash, cellStart, nCells, code) {
    const key = `${size}/${zipHash}/${cellStart}/${code}`;
    const hit = _indexCache.get(key);
    if (hit !== undefined) return hit;
    // A map that failed to load is a transient miss, not an answer — returning
    // -1 is right for this request but caching it would poison the shard for
    // the isolate's life.
    if (!map || cellStart + nCells > map.length) return -1;
    let found = -1;
    for (let i = 0; i < nCells; i++) {
        if (map[cellStart + i] === code) { found = i; break; }
    }
    // A pass can never answer this: it carries no board coordinate, so no code
    // derived from one can equal the reserved value.
    if (code === PASS_POINT_CODE) found = -1;
    _indexCache.set(key, found);
    while (_indexCache.size > CELL_CACHE_MAX) {
        _indexCache.delete(_indexCache.keys().next().value);
    }
    return found;
}

// Whether any of a shard's wanted moves still needs the map — a batch whose
// coordinates are all cached, or all already indices, must not fetch it.
function needsMap(wanted, size, zipHash) {
    return wanted.some((w) => w.picked < 0 && _indexCache.get(
        `${size}/${zipHash}/${w.blockOffset / BYTES_PER_CELL}/${w.code}`) === undefined);
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
    const altstatsView = new DataView(
        altstats.buffer, altstats.byteOffset, altstats.byteLength);

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
                above.push(altstatsView.getUint32(aOff + k * ALTSTATS_BYTES, true));
                below.push(altstatsView.getUint32(
                    aOff + (K + k) * ALTSTATS_BYTES, true));
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

        // The one user-dependent part, and the only one that reads a shard. The
        // caller supplies the played move as whichever it holds: an index for a
        // locally recorded attempt, a board coordinate for one that arrived
        // from another device. Only the coordinate needs resolving, and the
        // resolved index is always echoed back so the caller can store it.
        const hasIndex = Number.isInteger(it.picked) && it.picked >= 0;
        const hasMove = typeof it.move === 'string';
        if ((hasIndex || hasMove)
            && typeof it.zip === 'string' && /^[0-9a-f]{8}$/.test(it.zip)
            && (it.size === 9 || it.size === 13)) {
            const blockOffset = offsetsView.getUint32(row * OFFSETS_ROW, true);
            const nCells = offsetsView.getUint8(row * OFFSETS_ROW + 4);
            const code = hasIndex ? -1 : movePointCode(it.move, it.size);
            if (nCells > 0 && (hasIndex ? it.picked < nCells : code >= 0)) {
                const key = `${it.size}/${it.zip}`;
                if (!byShard.has(key)) byShard.set(key, []);
                byShard.get(key).push({
                    slot: i,
                    picked: hasIndex ? it.picked : -1,
                    code,
                    blockOffset,
                    nCells,
                });
            }
        }
    }

    // Shards are read CONCURRENTLY. Each one costs two asset reads at tens of
    // milliseconds apiece in production (a local disk read is sub-millisecond,
    // which hides this completely), and a batch spans ~170 distinct shards
    // because picked cells have almost no shard locality — so a sequential walk
    // is ~20s of pure round-trip latency for work that is entirely independent
    // per shard. The cap keeps the in-flight count well inside a Worker's
    // simultaneous-connection budget.
    const SHARD_CONCURRENCY = 10;
    const shardKeys = [...byShard.keys()];
    let nextShard = 0;
    const runShard = async (key) => {
        const [size, zipHash] = key.split('/');
        const wanted = byShard.get(key);
        // One map read per shard, not per move: several coordinates in one shard
        // share it, and a shard with nothing left to resolve reads none.
        const map = needsMap(wanted, size, zipHash)
            ? await shardAsset(env, size, zipHash, 'cells.bin')
            : null;
        const misses = [];
        for (const w of wanted) {
            let picked = w.picked;
            if (picked < 0) {
                picked = lookupIndexInMap(
                    map, size, zipHash, w.blockOffset / BYTES_PER_CELL,
                    w.nCells, w.code);
                if (picked < 0) continue;          // move matches no variation
                results[w.slot].pickedCellIndex = picked;
            }
            // Anything already cached needs no shard read at all; only a
            // genuinely new pick does, which is why the fetch is behind this.
            const byteOffset = w.blockOffset + picked * BYTES_PER_CELL;
            const hit = _cellCache.get(`${key}/${byteOffset}`);
            if (hit) results[w.slot].pickedEmbedding = hit;
            else misses.push({ slot: w.slot, byteOffset });
        }
        if (misses.length === 0) return;
        const bin = await shardAsset(env, size, zipHash, 'embeddings.bin');
        if (!bin) return;
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
    };
    await Promise.all(
        Array.from({ length: Math.min(SHARD_CONCURRENCY, shardKeys.length) }, async () => {
            for (let i = nextShard++; i < shardKeys.length; i = nextShard++) {
                await runShard(shardKeys[i]);
            }
        }),
    );

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
