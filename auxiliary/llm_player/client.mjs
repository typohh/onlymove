// auxiliary/llm_player/client.mjs
//
// The shipped PWA client, loaded under node, plus the few things the browser
// gives it for free: a localStorage, the vendored WGo.js and JSZip, and fetch
// against the served data. Shared by describe_problem.mjs (what a player is
// shown) and drive.mjs (playing an account from the command line).
//
// Everything rank- and serve-shaped is CALLED from the shipping tree, never
// restated: state.js holds the history, rank-decomposition.js re-fits the rank
// (reanchorRank, rankConverged, rankForSize, problemInfoWeight), weakness.js
// names the flagged serve categories, sgf.js converts coordinates. What is
// mirrored is only what sits in modules that need a DOM at load
// (problem-loader.js, ui.js) — each mirror names its source at its use site.

import { register, createRequire } from 'node:module';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PWA = path.join(ROOT, 'pwa');

// The client imports its deployment constants as plain JSON modules, which
// esbuild resolves at bundle time and bare Node refuses.
register(pathToFileURL(path.join(ROOT, 'ablations', 'json_module_loader.mjs')).href);

// state.js reads and persists through localStorage. An in-memory one makes every
// process a FRESH device: whatever the account knows comes from the history the
// driver hands it, exactly as a new device adopting the account would.
const store = new Map();
globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: (k) => { store.delete(k); },
    clear: () => { store.clear(); },
    key: (i) => [...store.keys()][i] ?? null,
    get length() { return store.size; },
};

const imp = (rel) => import(pathToFileURL(path.join(PWA, rel)).href);

export const stateMod = await imp('js/state.js');
export const rankMod = await imp('js/rank-decomposition.js');
export const weaknessMod = await imp('js/weakness.js');
export const sgfMod = await imp('js/sgf.js');
export const { quantizeEmbedding } = await imp('js/math-utils.js');
export const { rankToLabel, interpExtrap } = await imp('shared/rank-math.js');
export const popMod = await imp('js/population-constants.js');
popMod.loadPopulationConstants();   // main.js does this at boot
const RANK_MAP = (await imp('tsumegos/rank_map.json')).default;

// rank-translation.js serverRanksFor (its module needs the DOM at load): the shown
// rank, time-adjusted for the history's solving speed, read through each
// server's coupling — the values the rank badge's bubble labels.
export function serverRanksFor(rank, history) {
    const out = {};
    if (!Number.isFinite(rank)) return out;
    const deltaRating = popMod.timeAdjustDeltaRating(history);
    const adjValue = deltaRating !== null ? rankMod.rankAfterRatingDelta(rank, deltaRating) : rank;
    const adjRank = Number.isFinite(adjValue) ? adjValue : rank;
    for (const [server, c] of Object.entries(RANK_MAP || {})) {
        if (!c || !Array.isArray(c.our) || !Array.isArray(c.server) || c.our.length < 2) continue;
        const v = interpExtrap(adjRank, c.our, c.server);
        if (Number.isFinite(v)) out[server] = v;
    }
    return out;
}
export const { DATA_VERSION, EMBEDDING_META_INLINE, SHIPPED_K } = await imp('js/data-version.js');
export const { APP_VERSION } = await imp('src/version.js');

// The vendored WGo.js is written for a browser and touches the DOM while it
// loads; an inert, infinitely deep stub satisfies every such touch (the same
// harness as tests/js_bridge.mjs). Its Kifu/KifuReader/Game are pure.
function loadWGo() {
    const stub = () => new Proxy(function () {}, {
        get: (t, k) => (k === Symbol.toPrimitive ? () => '' : k === 'length' ? 0 : stub()),
        apply: () => stub(),
        construct: () => stub(),
        set: () => true,
    });
    const ctx = {
        console,
        navigator: { userAgent: '' },
        document: new Proxy({}, {
            get: (t, k) => (k === 'getElementsByTagName' ? () => [{ src: 'lib/wgo/wgo.min.js' }] : stub()),
        }),
    };
    ctx.window = new Proxy(ctx, { get: (t, k) => (k in t ? t[k] : stub()) });
    vm.createContext(ctx);
    for (const f of ['wgo.min.js', 'wgo.player.min.js']) {
        vm.runInContext(readFileSync(path.join(PWA, 'lib', 'wgo', f), 'utf8'), ctx);
    }
    return ctx.WGo;
}
export const WGo = loadWGo();
const JSZip = createRequire(import.meta.url)(path.join(PWA, 'lib', 'jszip.min.js'));

// --- served data ----------------------------------------------------------------
// Everything a problem needs is read from the SERVED tree — over HTTP from a
// running Worker, or from a local pwa/ directory for offline description — so
// what the driver plays is what that server's players are served. Files are
// cached on disk under the data version they were fetched at, so a re-ship
// re-downloads instead of mixing two packaging runs.
export class DataSource {
    constructor(base, cacheDir) {
        this.remote = /^https?:\/\//.test(base);
        this.base = this.remote ? base.replace(/\/+$/, '') : path.resolve(base);
        this.cacheDir = this.remote
            ? path.join(cacheDir, new URL(this.base).host.replace(/[^A-Za-z0-9.-]/g, '_'), DATA_VERSION)
            : null;
        this.zips = new Map();
    }

    // `busted`: the client fetches this name with ?v=DATA_VERSION (a fixed name
    // whose content changes with a re-ship); content-hashed names go bare.
    async bytes(rel, { busted = false } = {}) {
        if (!this.remote) return readFileSync(path.join(this.base, rel));
        const cached = path.join(this.cacheDir, rel);
        if (existsSync(cached)) return readFileSync(cached);
        const url = `${this.base}/${rel}${busted ? `?v=${DATA_VERSION}` : ''}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`GET ${url} → ${resp.status}`);
        const buf = Buffer.from(await resp.arrayBuffer());
        mkdirSync(path.dirname(cached), { recursive: true });
        writeFileSync(cached, buf);
        return buf;
    }

    async json(rel, opts) { return JSON.parse((await this.bytes(rel, opts)).toString('utf8')); }

    async zip(size, zipFilename) {
        const key = `${size}/${zipFilename}`;
        if (!this.zips.has(key)) {
            this.zips.set(key, this.bytes(`tsumegos/${size}/${zipFilename}`)
                .then((buf) => new JSZip().loadAsync(buf)));
        }
        return this.zips.get(key);
    }

    // lookup.txt: "id zipHash size" per line; the row index is what the
    // lookup-aligned sidecars (lookup.altstats.bin) are addressed by.
    async lookup() {
        if (!this._lookup) {
            const lines = (await this.bytes('lookup.txt', { busted: true })).toString('utf8')
                .split('\n').filter((l) => l.trim() !== '');
            const byId = new Map();
            lines.forEach((line, row) => {
                const [id, zipHash, size] = line.trim().split(/\s+/);
                byId.set(id, { row, zipFilename: `${zipHash}.zip`, size: Number(size) });
            });
            this._lookup = { byId, rows: lines.length };
        }
        return this._lookup;
    }

    async sgf(size, zipFilename, id) {
        const zip = await this.zip(size, zipFilename);
        const f = zip.file(`${id}.sgf`);
        if (!f) throw new Error(`${id}.sgf not in ${zipFilename}`);
        return f.async('text');
    }
}

// --- the position a player is shown ---------------------------------------------
// board.js loadSgf: the reader walks forward while the node has a single child,
// so the shown position is the first branching node — the decision node, whose
// children are every point a click can play (getLegalMoveNode). The moves that
// led there are NOT part of what a player sees — only the stones, the last-move
// marker and the capture counts — so nothing of the sequence is returned.
export function readPosition(sgfText) {
    const kifu = WGo.Kifu.fromSgf(sgfText);
    const reader = new WGo.KifuReader(kifu);
    while (reader.node.children.length === 1) reader.next();
    const game = reader.game;
    const position = game.position;
    const size = position.size;
    const grid = [];
    for (let y = 0; y < size; y++) {
        const row = [];
        for (let x = 0; x < size; x++) row.push(position.get(x, y));
        grid.push(row);
    }
    const last = reader.node.move && !reader.node.move.pass ? reader.node.move : null;
    // board.js markKoPoints: the last move's orthogonal neighbours that are
    // illegal to play because of ko (WGo's play(..., true) code 4) carry a marker.
    const koPoints = [];
    if (last) {
        for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            if (game.play(last.x + dx, last.y + dy, game.turn, true) === 4) {
                koPoints.push({ x: last.x + dx, y: last.y + dy });
            }
        }
    }
    const children = reader.node.children;
    const candidates = [];
    children.forEach((c, index) => {
        if (c.move && !c.move.pass && Number.isInteger(c.move.x)) {
            candidates.push({ index, x: c.move.x, y: c.move.y });
        }
    });
    // Why a point cannot be played, in WGo's own terms (Game.play's dry-run
    // codes) — null when it can. A click there does nothing on the board.
    const ILLEGAL = { 1: 'off the board', 2: 'occupied', 3: 'suicide', 4: 'ko' };
    const illegalReason = (x, y) => {
        const code = game.play(x, y, game.turn, true);
        return code === false ? null : (ILLEGAL[code] || 'illegal');
    };
    return {
        illegalReason,
        size,
        toPlay: game.turn,                  // WGo.B = 1, WGo.W = -1
        grid,
        last: last ? { x: last.x, y: last.y, c: last.c } : null,
        koPoints,
        captures: { black: position.capCount.black, white: position.capCount.white },
        candidates,
        variationNodes: children,
    };
}

// ui.js computeCaptureScore: the header's one number, komi plus White's captures
// minus Black's (a fixed KOMI there, not the SGF's KM).
const UI_KOMI = 6.5;

const colorName = (c) => (c === WGo.B ? 'Black' : 'White');

export function describePosition(pos) {
    const { size } = pos;
    const cols = [];
    for (let x = 0; x < size; x++) cols.push(sgfMod.toBoardNotation(x, 0, size)[0]);
    const pad = String(size).length;
    const header = `${' '.repeat(pad + 1)} ${cols.join(' ')}`;
    const lines = [header];
    for (let y = 0; y < size; y++) {
        const label = String(size - y).padStart(pad);
        const cells = pos.grid[y].map((v) => (v === WGo.B ? 'X' : v === WGo.W ? 'O' : '.'));
        lines.push(`${label}  ${cells.join(' ')}  ${label}`);
    }
    lines.push(header);
    const at = (p) => sgfMod.toBoardNotation(p.x, p.y, size);
    const adjustment = UI_KOMI + pos.captures.white - pos.captures.black;
    const fmt = (s) => (s > 0 ? `+${s.toFixed(1)}` : s.toFixed(1));
    const out = [
        `${size}x${size} board, Japanese rules. ${colorName(pos.toPlay)} to play.`,
        `Exactly one move wins the game for ${colorName(pos.toPlay)}; find it.`,
        '',
        `X = Black, O = White, . = empty. Columns ${cols[0]}-${cols[size - 1]} (no I), row ${size} at the top.`,
        '',
        ...lines,
        '',
    ];
    if (pos.last) out.push(`Last move: ${colorName(pos.last.c)} ${at(pos.last)}.`);
    if (pos.koPoints.length) {
        out.push(`Ko: ${colorName(pos.toPlay)} may not play ${pos.koPoints.map(at).join(', ')} this turn (immediate ko recapture).`);
    }
    out.push(`Komi ${UI_KOMI}; Black has captured ${pos.captures.black}, White has captured ${pos.captures.white}. ` +
             `Effective komi for White: ${fmt(adjustment)}.`);
    return out.join('\n');
}

// --- problem objects --------------------------------------------------------------
// problem-loader.js doLoadZipProblems: a shard's metadata.json entries, with the
// rank and the selection weight derived on the client.
export async function shardProblems(src, size, zipFilename) {
    const zip = await src.zip(size, zipFilename);
    const meta = JSON.parse(await zip.file('metadata.json').async('text'));
    return meta.map((p) => ({
        ...p, zipFilename, size,
        rank: rankMod.rankForSize(p.rating, size),
        quality: rankMod.problemInfoWeight(p.slope, p.floor),
    }));
}

export async function embeddingsIndex(src, size, zipFilename) {
    const zip = await src.zip(size, zipFilename);
    const f = zip.file('embeddings_index.json');
    return f ? JSON.parse(await f.async('text')).problems : null;
}

// picked-embedding.js decodeFp16Rows (private there).
function fp16(u16) {
    const sign = (u16 & 0x8000) >> 15;
    const exp = (u16 & 0x7C00) >> 10;
    const frac = u16 & 0x03FF;
    let v;
    if (exp === 0) v = Math.pow(2, -14) * (frac / 1024);
    else if (exp === 0x1F) v = frac === 0 ? Infinity : NaN;
    else v = Math.pow(2, exp - 15) * (1 + frac / 1024);
    return sign ? -v : v;
}

// The problem's per-cell block (one K-vector per root variation, SGF order), out
// of the shard's embeddings sidecar at the offset its in-zip index records.
export async function problemCells(src, size, zipFilename, id) {
    const meta = EMBEDDING_META_INLINE;
    const index = await embeddingsIndex(src, size, zipFilename);
    const entry = index && index[id];
    if (!entry) return null;
    const bin = await src.bytes(`tsumegos/${size}/${zipFilename.replace(/\.zip$/, '')}.embeddings.bin`);
    const rows = [];
    for (let r = 0; r < entry.n_cells; r++) {
        const row = new Array(meta.K);
        for (let k = 0; k < meta.K; k++) row[k] = fp16(bin.readUInt16LE(entry.offset + (r * meta.K + k) * 2));
        rows.push(row);
    }
    return rows;
}

// picked-embedding.js fetchAltStatMasses: the PUBLISHED availability null, K
// little-endian uint32 above-masses then K below-masses at the lookup row.
export async function altStatMasses(src, id) {
    const meta = EMBEDDING_META_INLINE;
    const { byId, rows } = await src.lookup();
    const hit = byId.get(id);
    if (!hit) return null;
    const bin = await src.bytes('tsumegos/lookup.altstats.bin', { busted: true });
    const width = meta.altstats_bytes_per_row;
    if (bin.length !== rows * width) throw new Error('lookup.altstats.bin does not match lookup.txt');
    const o = hit.row * width;
    const altAbove = [], altBelow = [];
    for (let k = 0; k < meta.K; k++) {
        altAbove.push(bin.readUInt32LE(o + 4 * k));
        altBelow.push(bin.readUInt32LE(o + 4 * (meta.K + k)));
    }
    return altAbove.some((v, k) => v + altBelow[k] > 0) ? { altAbove, altBelow } : null;
}
