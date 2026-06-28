// js/data-version.js
//
// The served data manifest, INLINED into the bundle (not fetched). embedding_meta.json
// is the root of the data-version chain: it carries `data_version` (a hash of the
// served metadata.json files) plus the granular ratings/embedding versions and the
// label bank. Because it's part of the JS bundle, it can never be served stale by a
// cache — the exact failure that desynced two devices (a fresh version stamp read off
// one cached file while the per-attempt values came from another, stale-cached file).
//
// DATA_VERSION is the single cache-bust key for the *fetched* data files
// (metadata.json + lookup.txt). It moves ONLY when the shipped data changes, so a
// pure-JS deploy doesn't needlessly re-pull the multi-MB lookup.txt. The shard .zip /
// .embeddings.bin files need no key — their filenames are content hashes. The
// rating→rank curve and population constants are likewise inlined elsewhere, so the
// only query-busted fetches in the app are these two index files.
//
// data_version deliberately does NOT gate the history backfill: a ship that only adds
// new tsumegos bumps data_version (new shards → new metadata.json) but leaves
// ratings_version untouched, so new problems become serveable lazily while stored
// answer-history values are left alone. A ratings change bumps ratings_version and
// triggers the immediate re-derive (see embedding-backfill.js).
//
// Baked at build time from the pipeline's output (pwa/tsumegos/embedding_meta.json);
// build.mjs drops the served copy from dist since nothing fetches it anymore.
import EMBEDDING_META from '../tsumegos/embedding_meta.json';

export const EMBEDDING_META_INLINE = EMBEDDING_META || {};

// Falls back to APP_VERSION (the build define) only if an older packaged manifest
// predates the data_version field — keeps a half-migrated build from breaking.
export const DATA_VERSION =
    (EMBEDDING_META && typeof EMBEDDING_META.data_version === 'string'
        && EMBEDDING_META.data_version.length > 0)
        ? EMBEDDING_META.data_version
        : APP_VERSION;

// ONE ratings version. The rating model is a single joint cross-board fit, so a re-rate
// moves BOTH boards together — there is no independent per-board rating version. The
// manifest's `ratings_version` is keyed by size only because each board hashes a
// different problem set of that one fit; we collapse those sub-hashes (sorted, joined)
// to a single value that the backfill gates on and the startup log prints.
export const RATINGS_VERSION = (() => {
    const rv = EMBEDDING_META && EMBEDDING_META.ratings_version;
    if (!rv || typeof rv !== 'object') return null;
    const parts = Object.keys(rv).sort().map((k) => rv[k]);
    return parts.length ? parts.join('-') : null;
})();
