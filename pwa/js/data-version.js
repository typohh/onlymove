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
// triggers the immediate re-derive (see backfill.js).
//
// Baked at build time from the pipeline's output (pwa/tsumegos/embedding_meta.json);
// build.mjs drops the served copy from dist since nothing fetches it anymore.
import EMBEDDING_META from '../tsumegos/embedding_meta.json';
import { EMBEDDING_VERSION_KEY } from './constants.js';

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

// The motif fit THIS BUILD ships. Compared against the device's stored value
// (EMBEDDING_VERSION_KEY) to decide whether the vectors this build can produce are
// the ones the device's stored history is made of.
export const EMBEDDING_VERSION =
    (EMBEDDING_META && typeof EMBEDDING_META.embedding_version === 'string')
        ? EMBEDDING_META.embedding_version
        : 'unknown';

// Whether a stored-data generation this build ships IS the one the device's
// stored history is made of — the question every writer of a derived field has to
// ask, and the reason no field needs a version stamp of its own.
//
// The app is upgraded between problems, so an instance can be a release behind
// while still running: its shards, its sidecars and its inlined constants are all
// the PREVIOUS generation's, and anything it derives from them belongs to that
// generation, not to the one the stored history is made of. Asked at the moment of
// WRITING rather than at boot, this catches that instance — it has no way to know
// it is behind, but it does know its own version, and the device's stored version
// says what the history holds. A boot-time check cannot: the instance booted when
// the two agreed, and once a newer instance re-stamps the device, no check
// anywhere can fire again, every version by then agreeing.
//
// The mirror of it is the re-derivation: when they disagree the other way (the
// build is AHEAD of the stored history), every stored value of that generation is
// dropped or rewritten and the version recorded, so nothing from the previous one
// outlives the change. That pair — refuse to write when behind, re-derive
// everything when ahead — is what keeps a single device-level version honest with
// no per-row bookkeeping at all.
export function storedVersionMatches(key, version) {
    if (version === null || version === undefined) return true;   // nothing to check against
    try {
        return localStorage.getItem(key) === version;
    } catch (_) {
        return false;   // no storage to vouch for anything
    }
}

// The motif-fit generation: the axis vectors on an attempt and on a problem object.
export function motifDataIsCurrent() {
    return storedVersionMatches(EMBEDDING_VERSION_KEY, EMBEDDING_VERSION);
}

// The shipped axis count. A SHAPE fact, not a provenance test: it says how many
// components a current vector has, so a per-axis loop cannot run off the end of
// one. Which FIT a vector came from is the device-level version above.
export const SHIPPED_K =
    (EMBEDDING_META && Number.isInteger(EMBEDDING_META.K) && EMBEDDING_META.K > 0)
        ? EMBEDDING_META.K
        : null;
