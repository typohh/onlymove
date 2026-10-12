// Build-time check that the served data is ONE packaging run.
//
// pwa/tsumegos/ is gitignored and travels between machines by file copy, so a
// tree can hold one run's embedding_meta.json beside another run's shards. Every
// file in such a tree is well-formed, and the bundle built from it tells clients
// their cached metadata.json / lookup.txt (keyed by `?v=<data_version>`) are
// current while the shards those name are gone — each cached client then 404s on
// every zip it asks for, and a reload cannot help because the key never moved.
//
// So the stamp is recomputed from the files it claims to describe and the build
// fails on a disagreement. `computeDataVersion` mirrors `compute_data_version`
// (pipeline/package_for_pwa.py) byte for byte — same files, same order, same
// stand-ins for a missing one — and tests/test_js_twins_data_version.py holds the
// two to the same answer. The remedy for a failure is re-running
// `python -m pipeline.package_for_pwa` on this tree (or copying the WHOLE of
// pwa/tsumegos/ plus pwa/lookup.txt from the machine that packaged), never
// editing the stamp.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

// utils/constants.py ALL_BOARD_SIZES.
export const ALL_BOARD_SIZES = [9, 13];
// pipeline/package_for_pwa.py DATA_VERSION_EPOCH.
export const DATA_VERSION_EPOCH = 'epoch-2';

function readOrNull(file) {
    try { return fs.readFileSync(file); } catch { return null; }
}

export function computeDataVersion(outputDir) {
    const h = crypto.createHash('sha256');
    h.update(DATA_VERSION_EPOCH);
    const fold = (file, missing) => h.update(readOrNull(file) ?? missing);
    const t = path.join(outputDir, 'tsumegos');
    for (const size of ALL_BOARD_SIZES) {
        fold(path.join(t, String(size), 'metadata.json'), `missing-${size}`);
    }
    fold(path.join(t, 'problem_params.bin'), 'missing-problem-params');
    fold(path.join(t, 'lookup.embeddings.bin'), 'missing-correct-embeddings');
    for (const name of ['lookup.offsets.bin', 'lookup.altstats.bin']) {
        fold(path.join(t, name), `missing-${name}`);
    }
    fold(path.join(t, 'se_floor.json'), 'missing-se-floor');
    return h.digest('hex').slice(0, 16);
}

// Throws unless `outputDir` holds one packaging run: the stamp matches the files,
// and every shard and lookup row points at a zip that is there. Returns the
// counts for the build log. A tree with no embedding_meta.json has never been
// packaged and is left alone (`npm run dev` on a fresh checkout).
export function checkServedData(outputDir = '.') {
    const t = path.join(outputDir, 'tsumegos');
    const metaBytes = readOrNull(path.join(t, 'embedding_meta.json'));
    if (metaBytes === null) return null;
    const stamped = JSON.parse(metaBytes.toString('utf-8')).data_version;
    const actual = computeDataVersion(outputDir);
    if (stamped !== actual) {
        throw new Error(`served data is not one packaging run: embedding_meta.json says `
            + `data_version ${stamped}, the files beside it hash to ${actual}. `
            + `Re-run pipeline.package_for_pwa here, or copy ALL of pwa/tsumegos/ and `
            + `pwa/lookup.txt from the machine that packaged.`);
    }
    let shards = 0;
    const listed = {};
    for (const size of ALL_BOARD_SIZES) {
        const entries = JSON.parse(fs.readFileSync(path.join(t, String(size), 'metadata.json'), 'utf-8'));
        listed[size] = new Set(entries.map(e => e.filename.replace(/\.zip$/, '')));
        for (const e of entries) {
            if (!fs.existsSync(path.join(t, String(size), e.filename))) {
                throw new Error(`served data is not one packaging run: tsumegos/${size}/metadata.json `
                    + `lists ${e.filename}, which is not on disk.`);
            }
            shards++;
        }
    }
    // lookup.txt is not in the stamp, and it is the other file a client caches
    // under it: every row must name a shard this run's metadata lists.
    let rows = 0;
    for (const line of fs.readFileSync(path.join(outputDir, 'lookup.txt'), 'utf-8').split('\n')) {
        if (!line) continue;
        const [id, shard, size] = line.split(' ');
        if (!listed[size] || !listed[size].has(shard)) {
            throw new Error(`served data is not one packaging run: lookup.txt sends ${id} to `
                + `shard ${shard} (${size}), which this tree's metadata.json does not list.`);
        }
        rows++;
    }
    return { dataVersion: actual, shards, rows };
}
