// tests/js_bridge.mjs
//
// The JS half of the twin tests: calls the SHIPPED client modules, unmodified,
// on inputs the Python side built, so a test can hold a Python function and its
// JS mirror to the same answer.
//
// Usage:  node tests/js_bridge.mjs < calls.json
// Reads [{module, fn, args}], writes [{ok: value} | {error: message}] in order.
// `module` is a repo-relative path to an ES module ("pwa/js/sgf.js"), whose
// export `fn` is called with `args` — or, when `args` is absent, read as a
// value (a constant). The module "bridge" names the helpers below instead,
// which exist where a call needs an object the client builds itself (a WGo SGF
// tree, a node to write a comment onto).
//
// Numbers the JSON grammar cannot carry (NaN, ±Infinity, -0) cross in both
// directions as {"__num__": "NaN" | "Infinity" | "-Infinity" | "-0"}.

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// The client modules import their deployment constants as plain JSON modules,
// which esbuild resolves at bundle time and bare Node refuses.
register(pathToFileURL(path.join(ROOT, 'ablations', 'json_module_loader.mjs')).href);

// state.js reads localStorage at module load; an in-memory one is all it needs.
const store = new Map();
globalThis.localStorage = {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: (k) => { store.delete(k); },
    clear: () => { store.clear(); },
    key: (i) => [...store.keys()][i] ?? null,
    get length() { return store.size; },
};

// The client parses SGF with the vendored WGo.js, which is written for a
// browser and touches the DOM while it loads. An inert, infinitely deep stub
// satisfies every such touch; the SGF parser itself uses none of it.
let wgo = null;
function WGo() {
    if (wgo) return wgo;
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
        vm.runInContext(readFileSync(path.join(ROOT, 'pwa', 'lib', 'wgo', f), 'utf8'), ctx);
    }
    wgo = ctx.WGo;
    return wgo;
}

// The node reached from the root by child indices, as board.js's loadSgf sees it.
function nodeAt(sgf, childPath) {
    let n = WGo().Kifu.fromSgf(sgf).root;
    for (const i of childPath) n = n.children[i];
    return n;
}

const sgfModule = () => import(pathToFileURL(path.join(ROOT, 'pwa', 'js', 'sgf.js')).href);

const helpers = {
    // subtreeReachesConclusion on the node at `childPath` of a WGo-parsed tree.
    async conclusionAt(sgf, childPath) {
        const { subtreeReachesConclusion } = await sgfModule();
        return subtreeReachesConclusion(nodeAt(sgf, childPath));
    },
    // Apply setEvalComment writes in order to one node starting from `initial`
    // (null = no comment); returns each write's result and the final comment.
    async evalWrites(initial, writes) {
        const { setEvalComment } = await sgfModule();
        const node = initial === null ? {} : { comment: initial };
        const wrote = writes.map(([w, s, v]) => setEvalComment(node, w, s, v));
        return { wrote, comment: node.comment ?? null };
    },
    // variationVisitWeights over nodes carrying these comments (null = none).
    async visitWeights(comments) {
        const { variationVisitWeights } = await sgfModule();
        return variationVisitWeights(comments.map((c) => (c === null ? {} : { comment: c })));
    },
};

const decode = (v) => {
    if (Array.isArray(v)) return v.map(decode);
    if (v && typeof v === 'object') {
        if (Object.keys(v).length === 1 && '__num__' in v) {
            return { NaN: NaN, Infinity: Infinity, '-Infinity': -Infinity, '-0': -0 }[v.__num__];
        }
        return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, decode(x)]));
    }
    return v;
};

const encode = (v) => {
    if (typeof v === 'number') {
        if (Number.isNaN(v)) return { __num__: 'NaN' };
        if (v === Infinity) return { __num__: 'Infinity' };
        if (v === -Infinity) return { __num__: '-Infinity' };
        if (Object.is(v, -0)) return { __num__: '-0' };
        return v;
    }
    if (ArrayBuffer.isView(v)) return Array.from(v, encode);
    if (Array.isArray(v)) return v.map(encode);
    if (v && typeof v === 'object') {
        return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, encode(x)]));
    }
    return v === undefined ? null : v;
};

const calls = decode(JSON.parse(readFileSync(0, 'utf8')));
const out = [];
for (const { module, fn, args } of calls) {
    try {
        const mod = module === 'bridge'
            ? helpers
            : await import(pathToFileURL(path.join(ROOT, module)).href);
        if (!(fn in mod)) throw new Error(`${module} has no export ${fn}`);
        const value = args === undefined ? mod[fn] : await mod[fn](...args);
        out.push({ ok: encode(value) });
    } catch (e) {
        out.push({ error: String(e && e.stack ? e.stack : e) });
    }
}
process.stdout.write(JSON.stringify(out));
