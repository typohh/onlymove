// js/account-sync.js
//
// Multi-device account sync. The account credential IS the userId (an 8-char
// random token — see getOrCreateUserId in state.js): every attempt is already
// reported to D1 under it, so "logging in" on another device means adopting
// the token and rebuilding local history from the server's report log.
//
// Reports are append-only (nothing ever UPDATEs tsumego_reports), so change
// detection needs no content hash — the AUTOINCREMENT id cursor is complete:
// id tracks insertion order, so it stays gap-free even for rows inserted with a
// historical created_at (an offline flush, a backfill). A per-pull digest
// (serverSeq/serverDigest = user_seq + Σ rowHash(id)) rides along as a backstop,
// escalating to a full reconcile if a gap is ever found.
//
// LOCAL REPRESENTATION IS MIRROR-BY-ID (post migrateMirrorById): the acked
// history is a 1:1 mirror of the server's rows keyed by serverId; identity is
// the row id, serverCreatedAt is event-time metadata. Merging therefore needs
// no timestamp dedup, and the digest matches the server BY CONSTRUCTION —
// duplicate created_at rows on the server (accidental double-uploads) are
// held either as entries or as absorbed-id receipts, never silently dropped
// (silently dropping them was the old permanent digest-divergence loop).
// Devices whose migration hasn't completed yet run the legacy timestamp-dedup
// merge below until it does.
//
//   - cursorId only ever advances from /sync PULLS, never from this device's own
//     report acks. Advancing on acks would let a concurrent foreign row with a
//     lower id slip behind the cursor forever.
//   - The device's own rows are recognized on pull by their id (legacy path:
//     exact serverCreatedAt match): /report echoes its server-stamped
//     created_at + row id and sendReport stamps both on the matching
//     answerHistory entry (recordReportAck).
//   - Entries whose ack was lost (offline attempt, response dropped) are
//     claimed fuzzily: same (problemId, correct, isRepeat), no serverCreatedAt
//     yet, nearest local date within ±10 min, greedy 1-1. On the id-first
//     path the claim is tried for EVERY un-held row — never preempted by a
//     timestamp match (that preemption made the outbox re-upload rows whose
//     acked twin's stamp sat in the cold tail: the server-side dup mint).
//
// On an EXISTING device cursorId initializes to the current frontier (the
// migrateIdCursor() probe adopts serverMaxId — "now" in id space), since local
// state already reflects its own history. Only an explicit account switch
// (switchAccount) starts from id 0, pulling the full server history into a wiped
// localStorage.
//
// Merged rows become skeleton entries (problemId/correct/date/timeTaken/
// isRepeat + review accounting); the model fields (rating/rank/slope/floor/
// embeddings, picked-move cell index from the reported move) are filled by
// fillSyncedHistoryFields (backfill.js) from the same sources the
// version-driven backfill uses: problem_params.bin for ratings, the
// lookup-aligned correct-move sidecar for embeddings, and a shard-zip walk
// only for the syncedMove -> pickedCellIndex derivation + picked-embedding
// fill.

import {
    state,
    getAnswerHistoryForSize, setAnswerHistoryForSize,
    getMergedAnswerHistory, updateAnswerHistory, updateUserRank, updateServeRank,
    evictColdTail, clearHistoryTail, syncTailExpectedToActual, isUnackedReportable,
    getWorkingSetEntries, replaceUnifiedHistory,
    SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE, generateUserId,
} from './state.js';
import { TOTAL_RANKS, HISTORY_BACKFILL_DONE_KEY, ANSWER_HISTORY_WORKING_SET } from './constants.js';
import { calculateAndStoreStatsResults } from './model.js';
import { trainWeakness } from './weakness.js';
import { estimateWindowRank } from './rank-decomposition.js';
import { fillSyncedHistoryFields, stampVersionsForFreshHistory } from './backfill.js';
import { refreshShownProblemAfterSync } from './problem-loader.js';
import { getTailSyncIndex, getTailDigestInfo, getTail, replaceTail } from './history-store.js';
import { rowHash } from '../shared/sync-digest.js';
import { readJSON, writeJSON } from './persist.js';
import { postJson } from './net.js';
import { updateAllUI } from './ui.js';

const SYNC_STATE_KEY = 'accountSyncState';
const SWITCH_FLAG_KEY = 'accountSwitchPending';
// First moment the server log carried per-attempt fields (move/time/is_repeat,
// one row per attempt incl. repeats). Rows older than this are the legacy
// one-row-per-problem era — both the claim relaxation and the one-off local
// history backfill key off it.
const PER_ATTEMPT_ERA_MS = Date.parse('2026-05-15T00:00:00Z');
// First moment every client reports the picked move in the CANONICAL
// orientation-0 frame (buildReportPayload inverse-transforms the click through
// the served orientation). Repeat rows from this era on carry a coordinate
// that maps onto the shard SGF, so their picked cell is recoverable
// cross-device; before it, repeats were reported in the SERVED frame (a random
// D4 reorientation) and stay skipped. MUST sit comfortably AFTER the deploy
// that ships canonical reporting — old bundles keep reporting served-frame
// repeat moves until their graceful reload, and a served-frame coordinate
// trusted as canonical can silently resolve to the WRONG variation. If that
// deploy slips past this date, bump the date before deploying (skipping a few
// extra days of repeat moves is harmless; trusting wrong-frame ones is not).
const CANONICAL_MOVE_ERA_MS = Date.parse('2026-07-07T00:00:00Z');
// Ack-less local entries are claimed by a server row when identity fields match
// and the timestamps agree within this window (local Date.now() at attempt vs
// edge time at insert — normally seconds apart; the slack absorbs clock skew).
const FUZZY_WINDOW_MS = 10 * 60 * 1000;
// Pagination safety bound: 100 pages × 1000 rows covers any realistic history.
const MAX_PAGES = 100;

// --- Mirror-by-id representation -------------------------------------------
// Once migrated (migrateMirrorById), the device's acked history is a 1:1
// MIRROR of the server's rows keyed by serverId: identity is the row id, and
// serverCreatedAt is event-time metadata only. Merging then needs no
// timestamp dedup at all — a row is "had" iff its id is held — which makes
// the digest consistent BY CONSTRUCTION (server-side duplicate created_at
// rows, the cause of the perpetual reconcile loop, are simply represented).
// Identical-content duplicate copies are not materialized as visible entries:
// their ids go into the ABSORBED-ID RECEIPT list below, which the digest
// counts alongside real entries. Un-migrated devices keep the legacy
// timestamp-dedup merge until their (cap-aware, resumable) migration runs.
const MIRROR_MIGRATED_KEY = 'mirrorByIdMigrated_v1';
// Receipt ids: server rows represented locally only as "a redundant identical
// copy of an entry we hold" (same user/problem/outcome/created_at — accidental
// double-uploads). Counted by computeLocalDigest and by the merge's dedup, so
// they can neither re-merge as visible duplicates nor leave a digest gap.
// Account-derived → wiped on switchAccount (ACCOUNT_WIPE_KEYS).
const ABSORBED_IDS_KEY = 'syncAbsorbedIds';

function isMirrorMigrated() {
    return !!localStorage.getItem(MIRROR_MIGRATED_KEY);
}

function readAbsorbedIds() {
    const arr = readJSON(ABSORBED_IDS_KEY);
    return Array.isArray(arr) ? arr.filter((v) => Number.isFinite(v)) : [];
}

function persistAbsorbedIds(ids) {
    if (ids.length === 0) return;
    const set = new Set(readAbsorbedIds());
    for (const id of ids) set.add(id);
    writeJSON(ABSORBED_IDS_KEY, [...set]);
}

// Does this server row describe the same attempt as the (acked) local entry?
// The identical-content test behind absorbing a duplicate copy: problem and
// outcome must match; is_repeat too, except for pre-per-attempt-era rows where
// the column didn't exist. Deliberately ignores time_used/review_time — the
// known double-uploads differ there occasionally, and the identity fields are
// what make the copy redundant.
function rowMatchesEntryContent(row, ts, entry) {
    if (entry.problemId !== row.problem) return false;
    if (!!entry.correct !== !!row.success) return false;
    if (ts >= PER_ATTEMPT_ERA_MS && !!entry.isRepeat !== !!row.isRepeat) return false;
    return true;
}

// Build the local skeleton entry for a server row nothing local claimed.
// Shared by the legacy merge, the id-first merge, and the mirror migration;
// fillSyncedHistoryFields later fills the model fields.
function buildSyncedEntry(row, ts, size) {
    const entry = {
        problemId: row.problem,
        correct: !!row.success,
        date: ts,
        size,
        serverCreatedAt: row.createdAt,
        serverId: Number.isFinite(row.id) ? row.id : undefined,
        timeTaken: typeof row.timeUsed === 'number' ? row.timeUsed : 0,
        reviewTime: 0,
        accumulatedReview: typeof row.reviewTime === 'number' ? row.reviewTime : 0,
        isRepeat: !!row.isRepeat,
        synced: true,
    };
    // The row's review_time is the problem's CUMULATIVE review before this
    // attempt; kept so recomputeProblemChains can diff consecutive synced
    // attempts into per-attempt reviewTime.
    if (typeof row.reviewTime === 'number') entry.syncedReviewTotal = row.reviewTime;
    // Since CANONICAL_MOVE_ERA_MS every reported move is in the canonical
    // orientation-0 frame (the client inverse-transforms the click through
    // the served orientation), so the coordinate maps onto the shard SGF
    // for every attempt and the backfill can derive pickedCellIndex from
    // it — repeats included. Older rows are served-frame: non-repeats were
    // always served at orientation 0 (identity — canonical anyway), but
    // repeats were randomly reoriented and unrecoverable; skip those.
    if (!row.success && typeof row.move === 'string' && row.move !== 'pass' &&
        (!row.isRepeat || ts >= CANONICAL_MOVE_ERA_MS)) {
        entry.syncedMove = row.move;
    }
    return entry;
}

// The userId match is the only validity requirement — the state must belong to
// the active account. (An earlier check also demanded the legacy created_at
// `cursor` string, which the id-cursor redesign no longer writes; that rejected
// every post-redesign state, so cursorId/lastSeq were discarded on every read
// and each pull restarted from id 0 — a billed full pull per app-open that
// capped most active accounts daily and killed the seq signal.)
function readSyncState() {
    const parsed = readJSON(SYNC_STATE_KEY);
    if (parsed && parsed.userId === state.userId) {
        return parsed;
    }
    return null;
}

function writeSyncState(sync) {
    writeJSON(SYNC_STATE_KEY, sync);
}

export function isValidAccountCode(code) {
    // 8 chars, matching the Worker's USER_ID_RE alphabet. Longer legacy ids
    // can't round-trip anyway: getOrCreateUserId truncates stored ids to 8.
    return typeof code === 'string' && /^[A-Za-z0-9_-]{8}$/.test(code);
}

// Stamp the server-side created_at + row id onto the answerHistory entry the
// report described, so pulls can skip this device's own rows (serverCreatedAt
// is the dedup key; serverId is the id-cursor / digest key). Called from
// sendReport with the /report response. The entry is created by updateUserStats
// microtasks after sendReport fires, normally long before the network roundtrip
// returns — the one retry covers the pathological ordering.
export function recordReportAck(problemId, createdAt, serverId = null) {
    const attach = () => {
        const history = state.answerHistory;
        for (let i = history.length - 1; i >= 0; i--) {
            const a = history[i];
            if (a.problemId === problemId && !a.serverCreatedAt) {
                a.serverCreatedAt = createdAt;
                if (Number.isFinite(serverId)) a.serverId = serverId;
                updateAnswerHistory(history);
                return true;
            }
        }
        return false;
    };
    if (!attach()) setTimeout(attach, 1500);
}

// Merge one page of /sync rows into the given size's answerHistory.
// Synchronous on purpose: no await between read and write of the history
// bucket, so a concurrent solve can't be clobbered. Returns the inserted
// entry objects (live references — the field fill mutates them in place, and
// the post-merge rank re-fit reads the merged history afterwards).
// `tailKnown` carries the serverCreatedAt values already in the IndexedDB cold
// tail (read once per pull). Without it the dedup would only see the working
// set, so a re-pull (re-adopt / "refresh history") would re-insert every row
// long since evicted to the tail as a fresh duplicate.
function mergeRowsForSize(size, rows, tailKnown = null) {
    const history = getAnswerHistoryForSize(size);
    const known = new Set();
    for (const a of history) {
        if (a.serverCreatedAt) known.add(a.serverCreatedAt);
    }
    if (tailKnown) for (const sc of tailKnown) known.add(sc);

    let claimed = 0;
    const newEntries = [];
    for (const row of rows) {
        if (typeof row.createdAt !== 'string' || typeof row.problem !== 'string') continue;
        if (known.has(row.createdAt)) continue;
        known.add(row.createdAt);
        const ts = Date.parse(row.createdAt);
        if (!Number.isFinite(ts)) continue;

        // Fuzzy claim — greedy nearest-by-date 1-1 against un-acked locals.
        let best = null;
        let bestGap = FUZZY_WINDOW_MS;
        for (const a of history) {
            if (a.serverCreatedAt || a.problemId !== row.problem) continue;
            if (!!a.correct !== !!row.success) continue;
            // Rows from before the per-attempt reporting era never carry
            // is_repeat (the column didn't exist) — don't let the flag block
            // a claim there. Verified on the worst-case real user: matched
            // pairs align to median 1.5s / p99 4.7s.
            if (ts >= PER_ATTEMPT_ERA_MS && !!a.isRepeat !== !!row.isRepeat) continue;
            const gap = Math.abs((a.date || 0) - ts);
            if (gap < bestGap) { best = a; bestGap = gap; }
        }
        if (best) {
            best.serverCreatedAt = row.createdAt;
            if (Number.isFinite(row.id)) best.serverId = row.id;
            claimed++;
            continue;
        }

        newEntries.push(buildSyncedEntry(row, ts, size));
    }

    if (newEntries.length === 0) {
        if (claimed > 0) setAnswerHistoryForSize(size, history);
        return newEntries;
    }

    const merged = history.concat(newEntries).sort((a, b) => (a.date || 0) - (b.date || 0));
    recomputeProblemChains(merged, new Set(newEntries.map((e) => e.problemId)));
    setAnswerHistoryForSize(size, merged);
    return newEntries;
}

// Id-first merge (mirror-by-id devices). Identity is the row id: a row is
// "had" iff its id is in the working set, the tail, or the absorbed receipts —
// no timestamp dedup. Rows not held first try the fuzzy claim against
// UN-ACKED entries (deliberately unconditional: the legacy merge's
// stamp-dedup preempted the claim, which is exactly how the outbox re-uploaded
// rows whose acked twin's stamp sat in the tail). A row that claims nothing
// but is an identical-content copy of an acked entry with the same stamp is a
// redundant duplicate: its id is absorbed as a receipt instead of becoming a
// visible duplicate entry. Everything else is inserted as a skeleton.
// `absorbedSet` is the pull-wide receipt set (mutated here, persisted too).
function mergeRowsForSizeById(size, rows, tailIndex, absorbedSet) {
    const history = getAnswerHistoryForSize(size);
    const wsIds = new Set();
    const wsByStamp = new Map();
    const noteStamp = (a) => {
        if (!a.serverCreatedAt) return;
        let list = wsByStamp.get(a.serverCreatedAt);
        if (!list) { list = []; wsByStamp.set(a.serverCreatedAt, list); }
        list.push(a);
    };
    for (const a of history) {
        if (Number.isFinite(a.serverId)) wsIds.add(a.serverId);
        noteStamp(a);
    }

    let claimed = 0;
    const newlyAbsorbed = [];
    const newEntries = [];
    for (const row of rows) {
        if (typeof row.createdAt !== 'string' || typeof row.problem !== 'string') continue;
        if (Number.isFinite(row.id) &&
            (wsIds.has(row.id) || tailIndex.ids.has(row.id) || absorbedSet.has(row.id))) continue;
        const ts = Date.parse(row.createdAt);
        if (!Number.isFinite(ts)) continue;

        // Fuzzy claim — greedy nearest-by-date 1-1 against un-acked locals.
        let best = null;
        let bestGap = FUZZY_WINDOW_MS;
        for (const a of history) {
            if (a.serverCreatedAt || !rowMatchesEntryContent(row, ts, a)) continue;
            const gap = Math.abs((a.date || 0) - ts);
            if (gap < bestGap) { best = a; bestGap = gap; }
        }
        if (best) {
            best.serverCreatedAt = row.createdAt;
            if (Number.isFinite(row.id)) {
                best.serverId = row.id;
                wsIds.add(row.id);
            }
            noteStamp(best);
            claimed++;
            continue;
        }

        // Redundant identical copy of a row we already hold? Absorb the id as
        // a receipt. A same-stamp entry with DIFFERENT content is a genuine
        // same-millisecond collision — falls through to a real insert. The
        // working-set list is already this size's view; tail twins carry their
        // own size (missing = the pre-size-column default, 9).
        if (Number.isFinite(row.id)) {
            const twins = wsByStamp.get(row.createdAt);
            const tailTwins = tailIndex.stamps.get(row.createdAt);
            const tailSize = (t) => SUPPORTED_BOARD_SIZES.includes(t.size) ? t.size : DEFAULT_BOARD_SIZE;
            const isTwin = (twins && twins.some((a) => rowMatchesEntryContent(row, ts, a))) ||
                (tailTwins && tailTwins.some((t) => tailSize(t) === size && rowMatchesEntryContent(row, ts, t)));
            if (isTwin) {
                newlyAbsorbed.push(row.id);
                absorbedSet.add(row.id);
                continue;
            }
        }

        const entry = buildSyncedEntry(row, ts, size);
        if (Number.isFinite(row.id)) wsIds.add(row.id);
        noteStamp(entry);
        newEntries.push(entry);
    }

    persistAbsorbedIds(newlyAbsorbed);
    if (newEntries.length === 0) {
        if (claimed > 0) setAnswerHistoryForSize(size, history);
        return newEntries;
    }
    const merged = history.concat(newEntries).sort((a, b) => (a.date || 0) - (b.date || 0));
    recomputeProblemChains(merged, new Set(newEntries.map((e) => e.problemId)));
    setAnswerHistoryForSize(size, merged);
    return newEntries;
}

// Rebuild the per-problem derived chains (successSinceFailure, reviewTime,
// accumulatedReview) for problems that just gained entries, replaying
// updateUserStats' incremental rules over the merged chronological order.
function recomputeProblemChains(history, touchedProblems) {
    for (const problemId of touchedProblems) {
        const entries = history.filter((e) => e.problemId === problemId);

        // Per-attempt reviewTime for synced entries: a row's syncedReviewTotal
        // is the cumulative review before that attempt, so the diff between
        // CONSECUTIVE attempts is the earlier one's post-attempt review. A
        // local entry in between owns part of that diff, so it breaks the chain.
        for (let i = 0; i + 1 < entries.length; i++) {
            const cur = entries[i];
            const next = entries[i + 1];
            if (cur.synced && !cur.reviewTime &&
                typeof cur.syncedReviewTotal === 'number' &&
                typeof next.syncedReviewTotal === 'number') {
                cur.reviewTime = Math.max(0, next.syncedReviewTotal - cur.syncedReviewTotal);
            }
        }

        let run = 0;
        let prevSSF;
        for (const e of entries) {
            // The server total is authoritative where known (covers gaps from
            // synced entries whose own reviewTime is unrecoverable).
            if (typeof e.syncedReviewTotal === 'number') run = e.syncedReviewTotal;
            e.accumulatedReview = run;
            run += e.reviewTime || 0;
            // Mirrors updateUserStats exactly, including leaving the field
            // unset on a first-time correct.
            if (e.isRepeat) {
                e.successSinceFailure = e.correct ? ((prevSSF !== undefined) ? prevSSF + 1 : 1) : 0;
            } else if (!e.correct) {
                e.successSinceFailure = 0;
            } else {
                delete e.successSinceFailure;
            }
            prevSSF = e.successSinceFailure;
        }
    }
}

let pullInFlight = false;

// Pull reports newer than the cursor and merge them. Deliberately unthrottled:
// an up-to-date incremental poll reads ~0–1 D1 rows (idx_tsumego_reports_user_id),
// so load / tab-visible / online can each pull freely — pullInFlight is the
// only gate. `mode` controls what the
// merge does to the rank estimate (see retrainAfterMerge): 'feed' (default,
// ongoing sync), 'adopt' (account switch), 'reconcile' (one-off local-history
// backfill — models only, never the rank; the merged rows are
// ancient). `fromStart` ignores the stored cursor and pulls the full history
// (the reconcile needs every row to claim against). Returns
// { inserted, complete } — `complete` is true only if the pull drained the
// server (no page failed mid-way), which the backfill requires before it may
// upload anything.
async function pullReports({ mode = 'feed', fromStart = false, rowSink = null } = {}) {
    if (pullInFlight) return { inserted: 0, complete: false };
    pullInFlight = true;
    try {
        let sync = readSyncState();
        if (!sync) {
            sync = { userId: state.userId };
            writeSyncState(sync);
        }

        // Pure id cursor: page by AUTOINCREMENT id (gap-free for rows inserted
        // with a historical created_at — an offline flush, a backfill). cursorId
        // is established by the one-time migrateIdCursor() probe (existing
        // devices) or advances from the first from-0 pull (fresh accounts /
        // adoption); a fromStart pull always restarts at 0. The new client never
        // sends the legacy created_at cursor.
        let sinceId = fromStart ? 0 : (Number.isInteger(sync.cursorId) ? sync.cursorId : 0);
        // A from-the-beginning pull (adoption / full refresh / a not-yet-migrated
        // device) evicts old rows to the cold tail mid-pull; only then must the
        // synced fill cover the tail.
        const startedFromBeginning = fromStart || !Number.isInteger(sync.cursorId);
        let complete = false;
        let capExceeded = false;
        let refreshesLeft;
        let lastBody = null;
        const insertedEntries = [];
        const touchedSizes = new Set();
        // Dedup against the cold tail too, not just the working set. Snapshot it
        // once up front: rows evicted DURING this pull were merged this pull (so
        // already in the working-set known-set), and /sync pages are disjoint by
        // cursor, so no row re-arrives within one pull. One cursor pass serves
        // both representations: ids + per-stamp content for the id-first merge,
        // the stamp set for the legacy one.
        const mirrored = isMirrorMigrated();
        const absorbedSet = mirrored ? new Set(readAbsorbedIds()) : null;
        let tailIndex = null;
        try { tailIndex = await getTailSyncIndex(); } catch (_) { tailIndex = null; }
        if (!tailIndex) tailIndex = { ids: new Set(), stampSet: new Set(), stamps: new Map() };
        for (let page = 0; page < MAX_PAGES; page++) {
            const res = await postJson('/sync', { userId: sync.userId, sinceId });
            if (!res.ok) break;
            const body = await res.json().catch(() => null);
            if (!body || !Array.isArray(body.reports)) break;
            lastBody = body;
            // The server caps full-from-beginning pulls per day; page 0 of such
            // a pull may come back capExceeded. (Pages 2+ carry a real cursor and
            // are never capped.) refreshesLeft rides those responses.
            if (typeof body.refreshesLeft === 'number') refreshesLeft = body.refreshesLeft;
            if (body.capExceeded) { capExceeded = true; break; }

            const bySize = new Map();
            for (const row of body.reports) {
                const size = SUPPORTED_BOARD_SIZES.includes(row.size) ? row.size : DEFAULT_BOARD_SIZE;
                if (!bySize.has(size)) bySize.set(size, []);
                bySize.get(size).push(row);
            }
            for (const [size, rows] of bySize) {
                // Hand the raw rows to an optional observer BEFORE the merge
                // mutates anything — the outbox's first-run claim soft-matches
                // against them (it needs every server row, incl. the already-
                // claimed pre-era originals the merge skips).
                if (rowSink) rowSink(size, rows);
                const inserted = mirrored
                    ? mergeRowsForSizeById(size, rows, tailIndex, absorbedSet)
                    : mergeRowsForSize(size, rows, tailIndex.stampSet);
                if (inserted.length > 0) {
                    insertedEntries.push(...inserted);
                    touchedSizes.add(size);
                }
            }

            // Advance the id cursor. Never move it backwards: a fromStart pull
            // walks through old ids the normal cursor is already past. Re-read
            // the sync state at each write — checkSyncSignal advances lastSeq
            // concurrently during the awaits above, and writing back the stale
            // object captured at pull start would regress it (self-correcting,
            // but it repeatedly defeats the signal debounce during long pulls).
            if (Number.isInteger(body.cursorId) && body.cursorId > sinceId) {
                sinceId = body.cursorId;
                sync = readSyncState() || sync;
                if (!Number.isInteger(sync.cursorId) || body.cursorId > sync.cursorId) {
                    sync.cursorId = body.cursorId;
                    writeSyncState(sync);
                }
            }
            if (!body.hasMore) { complete = true; break; }
        }

        // Keep lastSeq (the seq-signal high-water) current from the latest
        // summary, so the next /report's userSeq is compared against the right
        // baseline. (Fresh re-read: same stale-object concern as above.)
        sync = readSyncState() || sync;
        if (lastBody && Number.isFinite(lastBody.serverSeq) &&
            (!Number.isFinite(sync.lastSeq) || lastBody.serverSeq > sync.lastSeq)) {
            sync.lastSeq = lastBody.serverSeq;
            writeSyncState(sync);
        }

        // Fill model fields (incl. problemQuality, which the loop feed below
        // reads) before touching the loop state. A full pull also fills the
        // cold tail (its old rows were just evicted there). Runs on EVERY pull,
        // not just when rows merged: a fill that failed mid-way (network drop
        // between the pull and the params/lookup fetches, e.g. during adoption)
        // leaves skeleton entries with no later insert to re-trigger it — the
        // next pull heals them here. Cheap when nothing is missing (the fill
        // scans before it fetches anything).
        let healedFills = 0;
        try {
            healedFills = await fillSyncedHistoryFields(
                insertedEntries.length > 0 ? [...touchedSizes] : SUPPORTED_BOARD_SIZES,
                { includeTail: startedFromBeginning });
        } catch (e) {
            console.warn('account-sync: field fill failed (next pull retries):', e);
        }

        if (insertedEntries.length > 0) {
            console.log(`account-sync: merged ${insertedEntries.length} foreign attempts (sizes: ${[...touchedSizes].join(', ')})`);
            retrainAfterMerge(insertedEntries, mode);
        } else if (healedFills > 0) {
            // A previously-failed fill just completed: entries that were
            // rating-less skeletons (dropped from every estimate) now carry
            // their fields, so the models and rank must recompute.
            console.log(`account-sync: healed ${healedFills} skeleton fields from an earlier failed fill`);
            retrainAfterMerge([], 'feed');
        }

        // Cheap divergence backstop: after an incremental feed pull, confirm the
        // local acked-row count (and content digest) matches the server's. A
        // gap the incremental pull couldn't close (a below-cursor legacy row)
        // escalates to a full reconcile. Deferred a tick so it runs AFTER
        // pullInFlight is released (the reconcile is itself a pullReports).
        // fromStart / reconcile pulls ARE the reconcile, so they never re-check.
        if (!fromStart && mode === 'feed' && lastBody && lastBody.serverSeq != null) {
            const ss = lastBody.serverSeq, sd = lastBody.serverDigest;
            setTimeout(() => { checkInSync(ss, sd).catch(() => {}); }, 0);
        }

        return { inserted: insertedEntries.length, complete, capExceeded, refreshesLeft };
    } catch (e) {
        console.warn('account-sync: pull failed:', e);
        return { inserted: 0, complete: false };
    } finally {
        pullInFlight = false;
    }
}

// Local count of ACKED rows (working set + cold tail) and Σ rowHash(serverId)
// over them. Compared against the server's user_seq / digest. The content sum
// is valid only when every acked row carries a serverId (allHaveId); if any
// acked row lacks one, the check falls back to count-only until a reconcile
// repopulates them.
async function computeLocalDigest() {
    let ackedCount = 0, idSum = 0, allHaveId = true;
    for (const a of getMergedAnswerHistory()) {
        if (!a.serverCreatedAt) continue;
        ackedCount++;
        if (Number.isFinite(a.serverId)) idSum += rowHash(a.serverId);
        else allHaveId = false;
    }
    // Absorbed receipts are server rows we hold as "redundant identical copy
    // of an entry" (see ABSORBED_IDS_KEY) — counted like entries, so the
    // digest matches the server exactly despite the collapse.
    for (const id of readAbsorbedIds()) {
        ackedCount++;
        idSum += rowHash(id);
    }
    try {
        const t = await getTailDigestInfo();
        ackedCount += t.ackedCount;
        idSum += t.idSum;
        if (!t.allHaveId) allHaveId = false;
    } catch (_) { /* tail unavailable → working-set-only digest */ }
    return { ackedCount, idSum, allHaveId };
}

// What (if anything) diverges between the local digest and the server summary:
// 'gap' (fewer acked rows than the server's user_seq), 'content' (same count,
// different Σ rowHash), or null (in sync / not checkable).
function digestMismatch(serverSeq, serverDigest, { ackedCount, idSum, allHaveId }) {
    if (ackedCount < serverSeq) return 'gap';
    if (allHaveId && ackedCount === serverSeq && Number.isFinite(serverDigest) && idSum !== serverDigest) return 'content';
    return null;
}

// A divergence a full reconcile already failed to close (a permanently
// unmergeable server row, e.g. an exact created_at collision the merge can't
// claim). Remembered so the SAME divergence doesn't escalate every pull into a
// daily-capped full pull forever; any change in the server or local digest
// values produces a new signature and re-checks.
const DIGEST_GAP_ACCEPTED_KEY = 'syncDigestGapAccepted';

// Backstop divergence check (after an incremental pull). serverSeq null = the
// server's latest row isn't seeded yet (no post-migration report); skip.
async function checkInSync(serverSeq, serverDigest) {
    if (!Number.isFinite(serverSeq)) return;
    const local = await computeLocalDigest();
    const kind = digestMismatch(serverSeq, serverDigest, local);
    if (!kind) {
        localStorage.removeItem(DIGEST_GAP_ACCEPTED_KEY);
        return;
    }
    const sig = `${serverSeq}|${serverDigest}|${local.ackedCount}|${local.idSum}`;
    if (localStorage.getItem(DIGEST_GAP_ACCEPTED_KEY) === sig) return;
    console.log(`account-sync: digest ${kind} (acked ${local.ackedCount}/Σ ${local.idSum} vs server ${serverSeq}/Σ ${serverDigest}) — full reconcile`);
    const r = await refreshHistoryFromServer();
    // Remember the divergence only when a COMPLETED reconcile failed to close
    // it (a cap-exceeded / failed pull never ran, so it proves nothing).
    if (r && r.ok && digestMismatch(serverSeq, serverDigest, await computeLocalDigest())) {
        console.warn('account-sync: reconcile did not close the digest divergence — accepting it until the digest changes');
        localStorage.setItem(DIGEST_GAP_ACCEPTED_KEY, sig);
    }
}

// Post-merge state update. Models always retrain off the merged history. The rank
// is the linear-information window MLE (rank-decomposition.estimateWindowRank) — a
// PURE FUNCTION of the recent first-time history — so device adoption and an
// incremental foreign sync are the same operation: recompute from scratch (no
// loop state to warm, no anchor to recover), and because every device sees the
// same multiset of recent attempts they converge to the same rank regardless of
// how the attempts were batched. The recompute runs SILENTLY (no overlay), so
// the next live answer's reported rank change is that tsumego's effect alone — a
// freshly-synced foreign batch moves the rank between problems, never inside the
// shown delta (the live loop's R_before is this silently-updated value).
//
//   - adopt (account switch): re-fit over the new account's full merged history,
//     seeded from the data (the prior account's rank is irrelevant).
//   - feed (ongoing merge): re-fit over the merged history, warm-started from the
//     current rank.
//   - 'reconcile' (one-off local-history backfill): models retrain, but the rank
//     is untouched — the merged rows are ancient attempts already in the window or
//     predating it.
function retrainAfterMerge(insertedEntries, mode) {
    try { calculateAndStoreStatsResults(); } catch (e) { console.warn('account-sync: recall retrain failed:', e); }
    try { trainWeakness(); } catch (e) { console.warn('account-sync: weakness retrain failed:', e); }
    try {
        // Both paths re-fit the linear-information window MLE over the merged
        // history. A window estimator is a pure function of the recent first-time
        // history, so an account ADOPTION (full pull) and an incremental FEED of
        // foreign attempts are the same operation — recompute the rank from scratch;
        // there is no loop state to warm and no anchor to recover. This runs SILENTLY
        // (no overlay), so the next live answer's reported delta reflects that
        // tsumego alone and is never muddled by the synced batch. adopt seeds the
        // fit from the data (the prior account's rank is irrelevant); feed warm-
        // starts from the current rank.
        if (mode === 'adopt' || mode === 'feed') {
            const seed = mode === 'feed' ? state.userRank * TOTAL_RANKS : undefined;
            const est = estimateWindowRank(getMergedAnswerHistory(), seed);
            if (est) {
                updateUserRank(est.rankIndex / TOTAL_RANKS);
                updateServeRank(est.serveRankIndex / TOTAL_RANKS);
            }
        }
    } catch (e) {
        console.warn('account-sync: loop state update failed:', e);
    }
    updateAllUI();
    // The merged history may have changed what should be on screen: the shown
    // problem might have been solved on another device, or a recall may now be
    // due. Re-runs the NORMAL presentation selection; the standing challenge
    // is kept by default, and swapped for a fresh one only if the user
    // completed that exact problem on another client (it is no longer legal to
    // serve — see refreshShownProblemAfterSync). Fire-and-forget: no-ops
    // during init (nothing shown yet — 'adopt' runs before the first problem
    // load) or whenever the board isn't idle.
    refreshShownProblemAfterSync().catch((e) => {
        console.warn('account-sync: post-sync problem refresh failed:', e);
    });
}

// Adopt another account on this device: wipe everything derived from the old
// account's history, store the new credential, and reload — init then runs
// maybeFinishAccountSwitch, which pulls the full server history from ''.
// Settings that aren't account-derived (board size, audio, language, corpus
// version stamps) survive the wipe.
// Everything derived from the current account's history — wiped on adoption
// (switchAccount) and again when a failed adoption falls back to a fresh
// account (maybeFinishAccountSwitch drops the partially merged rows).
const ACCOUNT_WIPE_KEYS = [
    'answerHistory', 'challengeProblem', 'previousProblem',
    'userRank', 'serveRank',
    'modelParams', 'adamState',
    'weaknessParams', 'weaknessAdam', 'weaknessSe', 'weaknessAxisCov',
    'recallPausedUntil',
    // The server-side recall schedule snapshot belongs to the old account;
    // the new account's first upload must go replace-all.
    'recallScheduleSnapshot',
    // All bookkeeping about the OLD account's history/tail. (The mirror
    // MIGRATED flag is NOT wiped — it describes this device's data
    // representation, and an empty post-wipe history is trivially a mirror.)
    'syncTailFillPending',
    'syncDigestGapAccepted',
    ABSORBED_IDS_KEY,
    SYNC_STATE_KEY,
];

export function switchAccount(newId) {
    if (!isValidAccountCode(newId) || newId === state.userId) return false;
    for (const k of ACCOUNT_WIPE_KEYS) localStorage.removeItem(k);
    localStorage.setItem('userId', newId);
    writeSyncState({ userId: newId });
    localStorage.setItem(SWITCH_FLAG_KEY, '1');
    location.reload();
    return true;
}

// One /sync frontier probe ({sinceId:0, limit:1} — reads ~1 D1 row, not billed
// against the daily full-pull cap). Returns { serverMaxId, hasReports } — with
// serverMaxId null when the response predates the field — or null when the
// request itself failed (offline, server error, malformed body). Shared by
// accountExists and the id-cursor migration.
async function probeFrontier(userId) {
    try {
        const res = await postJson('/sync', { userId, sinceId: 0, limit: 1 });
        if (!res.ok) return null;
        const body = await res.json().catch(() => null);
        if (!body || !Array.isArray(body.reports)) return null;
        return {
            serverMaxId: Number.isInteger(body.serverMaxId) ? body.serverMaxId : null,
            // serverMaxId > 0 OR any returned row ⇒ the account has reports.
            hasReports: (Number.isInteger(body.serverMaxId) && body.serverMaxId > 0) || body.reports.length > 0,
        };
    } catch {
        return null;
    }
}

// Cheap existence probe before adopting an account code: an account "exists"
// iff it has at least one report (reports are the only server-side state an
// account is rebuilt from, so a code with none is indistinguishable from a
// typo). Returns true/false, or null when the check itself failed.
export async function accountExists(userId) {
    const probe = await probeFrontier(userId);
    return probe === null ? null : probe.hasReports;
}

export function hasPendingAccountSwitch() {
    return !!localStorage.getItem(SWITCH_FLAG_KEY);
}

// A failed adoption leaves a marker for the next boot's user notice ('cap' or
// 'network'); main.js consumes it and alerts.
const SWITCH_FAILED_KEY = 'accountSwitchFailed';

export function consumeAccountSwitchFailure() {
    const reason = localStorage.getItem(SWITCH_FAILED_KEY);
    if (reason) localStorage.removeItem(SWITCH_FAILED_KEY);
    return reason; // 'cap' | 'network' | null
}

// Called once during init, before the first problem load, so a just-switched
// device serves problems matched to the pulled rank instead of rank 0.
// Returns true if a switch was completed (caller may want a spinner around it).
export async function maybeFinishAccountSwitch() {
    if (!localStorage.getItem(SWITCH_FLAG_KEY)) return { switched: false };
    // The IndexedDB cold tail belongs to the adopted-away account (switchAccount
    // wipes localStorage but can't await an async IDB clear before reload), so
    // drop it (and its eviction tracking) before pulling the new account's history.
    await clearHistoryTail();
    // The device is history-less right now, and the pull's synced fill writes
    // fields derived from the current corpus — stamp the data-version gates so
    // the post-init version backfill doesn't mistake the freshly filled history
    // for stale data and wipe + re-walk it.
    stampVersionsForFreshHistory();
    // Adoption is a from-the-beginning full pull, so it counts against the daily
    // cap (capExceeded when spent) and can also fail on the network.
    const result = await pullReports({ mode: 'adopt', fromStart: true });
    if (typeof result.refreshesLeft === 'number') cacheRefreshesLeft(result.refreshesLeft);
    // The adoption is only allowed to complete if the FULL history arrived: a
    // half-adopted device would report under the account while serving from a
    // blank (rank-0) profile. On failure fall back to a FRESH account id — the
    // old account's local state was already wiped by switchAccount, so there is
    // nothing to keep — drop any partially merged rows, leave the notice for
    // the next boot's alert, and reload.
    if (result.capExceeded || !result.complete) {
        for (const k of ACCOUNT_WIPE_KEYS) localStorage.removeItem(k);
        await clearHistoryTail();
        const freshId = generateUserId();
        localStorage.setItem('userId', freshId);
        writeSyncState({ userId: freshId });
        localStorage.setItem(SWITCH_FAILED_KEY, result.capExceeded ? 'cap' : 'network');
        localStorage.removeItem(SWITCH_FLAG_KEY);
        location.reload();
        return { switched: false };
    }
    // pullReports retrains (models + rank re-fit) only when it merged rows. If
    // the completed adopt inserted nothing (rows already local), run the same
    // update explicitly so the device never finishes a switch with models/rank
    // derived from the pre-switch state.
    if (result.inserted === 0) retrainAfterMerge([], 'adopt');
    // The full pull lands the whole career in the working set; split it into the
    // working set + tail now, before the first problem load, freeing localStorage.
    // The synced fill above already filled the tail too (includeTail on a full
    // pull), so the evicted rows carry their ratings.
    await evictColdTail();
    localStorage.removeItem(SWITCH_FLAG_KEY);
    return { switched: true };
}

// --- Daily-capped full-history refresh (cap ENFORCED server-side) ---
// Recovers attempts the browser silently evicted from the IndexedDB tail by
// re-pulling the full server history. Each refresh is a full per-user D1 scan
// (the read-cost the rest of the app fights), so it is hard-capped per UTC day
// by the Worker (client counting is bypassable by clearing localStorage). The
// value below is for DISPLAY only — it mirrors the cap the server enforces.
export const HISTORY_REFRESH_DAILY_CAP = 3;
// Cached "refreshes left" from the server's last response, for the prompt's
// hint only — never the authority. { day: 'YYYY-MM-DD', left }.
const HISTORY_REFRESH_LEFT_KEY = 'historyRefreshLeft';

function cacheRefreshesLeft(left) {
    if (typeof left !== 'number') return;
    const day = new Date().toISOString().slice(0, 10);
    writeJSON(HISTORY_REFRESH_LEFT_KEY, { day, left });
}

// Display hint: the server-reported remaining count if we have one for today,
// else assume the full cap (the server is the real gate either way).
export function historyRefreshesLeftToday() {
    const c = readJSON(HISTORY_REFRESH_LEFT_KEY);
    const today = new Date().toISOString().slice(0, 10);
    if (c && c.day === today && typeof c.left === 'number') return Math.max(0, c.left);
    return HISTORY_REFRESH_DAILY_CAP;
}

// Pull the full server history (fromStart) to restore evicted tail rows, then
// re-split into working set + tail and resync the eviction high-water mark. The
// daily cap is enforced by the Worker on the since='' full pull (shared with
// account adoption) — if exceeded it returns capExceeded and serves no rows.
// `mode: 'reconcile'` retrains the
// models but leaves the rank loop untouched (the restored rows are ancient,
// already reflected in the user's rank). Returns { ok, reason, restored, left }.
export async function refreshHistoryFromServer() {
    const result = await pullReports({ fromStart: true, mode: 'reconcile' });
    if (result.capExceeded) {
        cacheRefreshesLeft(0);
        return { ok: false, reason: 'cap', left: 0 };
    }
    // A network failure (no full pass, nothing served) — the server didn't count
    // it, so don't report it as a spent refresh.
    if (!result.complete && result.inserted === 0 && result.refreshesLeft === undefined) {
        return { ok: false, reason: 'failed', left: historyRefreshesLeftToday() };
    }
    cacheRefreshesLeft(result.refreshesLeft);
    await evictColdTail();
    await syncTailExpectedToActual();
    return { ok: true, restored: result.inserted, left: historyRefreshesLeftToday() };
}

// Seq sync signal (see checkSyncSignal): a /report whose userSeq jumps past the
// device's expected next value means another device wrote → pull now. This is
// what keeps an ACTIVE session in sync without any polling; idle devices are
// covered by the load/visibility pulls below. A debounce absorbs the benign
// in-flight race (the pull merges nothing if it's our own row, and the seq
// baseline self-corrects on the next report).
const SIGNAL_DEBOUNCE_MS = 60 * 1000;
let lastSignalPullAt = 0;

function maybeSignalPull() {
    const now = Date.now();
    if (now - lastSignalPullAt < SIGNAL_DEBOUNCE_MS) return;
    lastSignalPullAt = now;
    pullReports();
}

// Called from sendReport with the /report response's userSeq — the server's
// per-profile attempt count including this report. If it lands past the device's
// expected lastSeq+1, another device wrote in between → pull. lastSeq advances to
// it either way (and is also kept current from /sync's serverSeq). This is the
// in-sync signal that lets an active session need no polling.
export function checkSyncSignal(userSeq) {
    if (!Number.isFinite(userSeq)) return;
    const sync = readSyncState();
    if (!sync) return;
    const expected = Number.isFinite(sync.lastSeq) ? sync.lastSeq + 1 : userSeq;
    const foreign = userSeq > expected;
    sync.lastSeq = userSeq;
    writeSyncState(sync);
    if (foreign) maybeSignalPull();
}

// --- Durable offline outbox --------------------------------------------------
// Any attempt recorded locally but never acked (an offline / failed /report, or
// a pre-PER_ATTEMPT_ERA one-row-per-problem gap) is eventually uploaded so it
// reaches the server and every other device. Permanent (not a one-off): runs on
// load, on tab-visible, and on the `online` event whenever un-acked entries
// exist — a cheap no-op otherwise, self-throttled, bounded by the server daily
// full-pull cap.
//   1. Claim-before-upload — a COMPLETE reconcile pull first, so the fuzzy
//      matcher in mergeRowsForSize claims any row the server ALREADY has (an
//      ack-lost report, or a pre-era one-row-per-problem original) instead of
//      re-uploading it as a duplicate.
//   2. Entries still un-acked after a complete pull are provably absent
//      server-side → uploaded via /report-backfill WITH THEIR RECORDED event
//      timestamps (historical created_at: the id cursor still propagates them to
//      other devices, and old ones stay out of the server-stats 7-day window).
//      The response's row ids are stamped as serverId.
//   3. HISTORY_BACKFILL_DONE_KEY is set whenever the device has no un-acked
//      backlog — the signal (read in ui.js) that the device's history is fully
//      server-side and safe to adopt on another device.
const OUTBOX_DONE_KEY = HISTORY_BACKFILL_DONE_KEY;
const OUTBOX_BATCH = 500;
// Don't run the (full-pull) flush more than ~twice an hour; the server daily
// cap is the hard bound. A no-op (no un-acked entries) is never throttled.
const OUTBOX_MIN_INTERVAL_MS = 30 * 60 * 1000;
let lastOutboxRunAt = 0;

// Local entries provably reportable but never acked: real problem id, a sane
// historical date (≥1h old, so nothing in-flight), and no serverCreatedAt.
function collectUnackedEntries() {
    const out = [];
    const cutoff = Date.now() - 3600 * 1000;
    for (const size of SUPPORTED_BOARD_SIZES) {
        for (const a of getAnswerHistoryForSize(size)) {
            if (!a || a.serverCreatedAt) continue;
            if (typeof a.problemId !== 'string' || !/^[0-9a-f]{8}$/.test(a.problemId)) continue;
            if (typeof a.date !== 'number' || !(a.date < cutoff)) continue;
            out.push({ entry: a, size });
        }
    }
    out.sort((x, y) => x.entry.date - y.entry.date);
    return out;
}

// "No un-acked backlog" is only truthful once the one-time tail rescue below
// has run — before it, un-acked rows stranded in the cold tail are invisible
// to the working-set scan, and the flag would advertise a device as safe to
// adopt while adoption would silently drop that deep history.
function markOutboxDone() {
    if (!localStorage.getItem(OUTBOX_TAIL_RESCUE_KEY)) return;
    localStorage.setItem(OUTBOX_DONE_KEY, '1');
}

export async function flushOutbox({ force = false } = {}) {
    if (collectUnackedEntries().length === 0) {
        // Nothing to flush — the device's history is fully server-side, so it's
        // safe to adopt elsewhere (ui.js reads this to reveal the Account menu).
        markOutboxDone();
        return;
    }
    if (!force && Date.now() - lastOutboxRunAt < OUTBOX_MIN_INTERVAL_MS) return;
    lastOutboxRunAt = Date.now();

    try {
        // Claim everything the server already has. Uploading is only safe
        // against a COMPLETE view of the server's rows.
        const { complete } = await pullReports({ mode: 'reconcile', fromStart: true });
        if (!complete) return;  // incomplete view (network / daily cap) — retry later

        const candidates = collectUnackedEntries();
        if (candidates.length === 0) {
            markOutboxDone();
            return;
        }

        // created_at must be unique per user: the sync dedup set is keyed on
        // the timestamp, so collisions (same-ms entries, or a clash with an
        // existing acked row) get nudged forward 1ms at a time.
        const taken = new Set();
        for (const size of SUPPORTED_BOARD_SIZES) {
            for (const a of getAnswerHistoryForSize(size)) {
                if (a && a.serverCreatedAt) taken.add(a.serverCreatedAt);
            }
        }
        const rows = candidates.map(({ entry, size }) => {
            let ts = entry.date;
            let iso = new Date(ts).toISOString();
            while (taken.has(iso)) iso = new Date(++ts).toISOString();
            taken.add(iso);
            return {
                entry,
                size,
                report: {
                    problem: entry.problemId,
                    success: !!entry.correct,
                    size,
                    createdAt: iso,
                    timeUsed: typeof entry.timeTaken === 'number' ? entry.timeTaken : null,
                    // Cumulative review before the attempt — same semantic the
                    // live report sends.
                    reviewTime: typeof entry.accumulatedReview === 'number' ? entry.accumulatedReview : null,
                    isRepeat: !!entry.isRepeat,
                },
            };
        });

        for (let i = 0; i < rows.length; i += OUTBOX_BATCH) {
            const batch = rows.slice(i, i + OUTBOX_BATCH);
            const res = await postJson('/report-backfill', { userId: state.userId, reports: batch.map(b => b.report) });
            if (!res.ok) {
                console.warn(`account-sync: outbox batch failed (${res.status}); will retry`);
                return;
            }
            const body = await res.json().catch(() => null);
            const ids = (body && Array.isArray(body.ids)) ? body.ids : [];
            // Stamp serverCreatedAt (dedup key) + serverId (digest / id-cursor
            // key) per batch so a later failure never re-uploads what landed.
            const sizesTouched = new Set();
            batch.forEach((b, j) => {
                b.entry.serverCreatedAt = b.report.createdAt;
                if (Number.isFinite(ids[j])) b.entry.serverId = ids[j];
                sizesTouched.add(b.size);
            });
            for (const size of sizesTouched) {
                setAnswerHistoryForSize(size, getAnswerHistoryForSize(size));
            }
        }

        markOutboxDone();
        console.log(`account-sync: outbox flushed ${rows.length} attempts to the server`);
    } catch (e) {
        console.warn('account-sync: outbox flush failed (will retry):', e);
    }
}

// --- One-time outbox tail rescue (remove the flag once all clients have run) --
// Before evictColdTail held back un-acked rows (state.isUnackedReportable), an
// offline attempt or a pre-PER_ATTEMPT_ERA gap row could be evicted to the
// IndexedDB tail — where the outbox (working-set-only by design) could never
// see it again, while OUTBOX_DONE_KEY was still set off the clean working set.
// COPY any such rows back into the working set (append-only on purpose: no
// tail rewrite, so there is no crash/eviction race — the stale un-acked tail
// copy stays behind and is absorbed by the standing (problemId, date) de-dup
// in getMergedFullAnswerHistory / the digest's acked-only counting). The
// standing outbox then claims/uploads them, and the now ack-aware eviction
// re-evicts them once acked. Idempotent + flag-gated.
const OUTBOX_TAIL_RESCUE_KEY = 'outboxTailRescue_v1';

async function rescueUnackedTailRows() {
    if (localStorage.getItem(OUTBOX_TAIL_RESCUE_KEY)) return;
    try {
        const tail = await getTail();
        const unacked = tail.filter((e) => isUnackedReportable(e));
        if (unacked.length === 0) {
            // Nothing stranded (or no tail at all — an unavailable IDB means the
            // tail is unreachable for the outbox too, so there is nothing to do).
            localStorage.setItem(OUTBOX_TAIL_RESCUE_KEY, '1');
            return;
        }
        let moved = 0;
        const bySize = new Map();
        for (const e of unacked) {
            const size = SUPPORTED_BOARD_SIZES.includes(e.size) ? e.size : DEFAULT_BOARD_SIZE;
            if (!bySize.has(size)) bySize.set(size, []);
            bySize.get(size).push(e);
        }
        for (const [size, entries] of bySize) {
            const history = getAnswerHistoryForSize(size);
            const seen = new Set(history.map((a) => `${a.problemId}|${a.date}`));
            const add = entries.filter((e) => !seen.has(`${e.problemId}|${e.date}`));
            if (add.length > 0) {
                setAnswerHistoryForSize(size, history.concat(add));
                moved += add.length;
            }
        }
        localStorage.setItem(OUTBOX_TAIL_RESCUE_KEY, '1');
        if (moved > 0) {
            // There IS a backlog now — the flag may have been lying.
            localStorage.removeItem(OUTBOX_DONE_KEY);
            console.log(`account-sync: rescued ${moved} un-acked attempts from the cold tail into the outbox`);
        }
    } catch (e) {
        console.warn('account-sync: tail rescue failed (retrying next load):', e);
    }
}

// --- One-time id-cursor migration (remove the flag once all clients have run) -
// A device that synced under the old release has a created_at cursor and no
// cursorId, and its old answer-history rows carry no serverId, so it can't
// derive an id cursor from local data. This brings it onto the id cursor WITHOUT
// ever calling the legacy created_at /sync branch — so that branch is needed
// only by not-yet-reloaded old bundles and can be retired on the normal
// client-update schedule (NOT "once every device has transitioned", which is
// unbounded). It probes the current frontier (a limit:1 /sync — not billed
// against the daily full-pull cap) and adopts it as cursorId ("cursor = now" in
// id space); the steady-state digest check then reconciles any gap, including
// the historical-below-cursor gap. Idempotent + flag-gated.
// A fresh/empty account (frontier 0) or a failed probe leaves cursorId unset —
// steady state falls back to a from-0 pull, which is correct and cheap there.
const ID_CURSOR_MIGRATED_KEY = 'idCursorMigrated_v1';

async function migrateIdCursor() {
    if (localStorage.getItem(ID_CURSOR_MIGRATED_KEY)) return;
    const sync = readSyncState();
    if (sync && !Number.isInteger(sync.cursorId)) {
        const probe = await probeFrontier(sync.userId);
        // Probe failed / fieldless response — retry next load (no flag).
        if (probe === null || probe.serverMaxId === null) return;
        if (probe.serverMaxId > 0) {
            sync.cursorId = probe.serverMaxId;
            writeSyncState(sync);
        }
    }
    localStorage.setItem(ID_CURSOR_MIGRATED_KEY, '1');
}

// --- One-time mirror-by-id migration (see the MIRROR_MIGRATED_KEY note) -----
// Brings the device's acked history onto the id-keyed mirror representation.
//
// Cheap path (most devices — young accounts, post-redesign adopters): every
// acked entry already carries its serverId, so the local set IS a mirror —
// set the flag, zero server reads. Any dup-twin rows the device lacks are
// healed later by the standing digest check + an id-first reconcile, which
// absorbs identical copies as receipts rather than visible duplicates.
//
// Full path (legacy histories with id-less acked entries): one raw from-0
// pull (counts against the daily full-pull cap; capExceeded/network failure →
// flag stays unset, retried next boot), then a pure rebuild: match every
// local entry to a server row (by id, then exact stamp + identical content,
// then the fuzzy claim against un-acked locals), absorb identical twin rows
// as receipts, keep unmatched locals as outbox/local-only rows, and DROP
// unmatched copies of attempts the mirror already holds (the rescue-era stale
// duplicates — re-uploading those is exactly how the server dups were minted).
// The rebuild is a pure function of (server rows, local entries), so a crash
// anywhere before the flag write just redoes it.
async function migrateMirrorById() {
    if (isMirrorMigrated()) return;

    const ackedNoId = (a) => a && a.serverCreatedAt && !Number.isFinite(a.serverId);
    let tailProbe;
    try { tailProbe = await getTail(); } catch (_) { tailProbe = []; }
    if (!getWorkingSetEntries().some(ackedNoId) && !tailProbe.some(ackedNoId)) {
        localStorage.setItem(MIRROR_MIGRATED_KEY, '1');
        localStorage.setItem(ID_CURSOR_MIGRATED_KEY, '1');
        console.log('account-sync: mirror-by-id migration — already id-complete');
        return;
    }

    if (pullInFlight) return; // a pull owns the network; retry next boot
    pullInFlight = true;
    let stats = null;
    try {
        // Raw from-0 pull: collect every server row, no merging.
        const rows = [];
        let sinceId = 0;
        let complete = false;
        let lastBody = null;
        for (let page = 0; page < MAX_PAGES; page++) {
            const res = await postJson('/sync', { userId: state.userId, sinceId });
            if (!res.ok) return;
            const body = await res.json().catch(() => null);
            if (!body || !Array.isArray(body.reports)) return;
            lastBody = body;
            if (typeof body.refreshesLeft === 'number') cacheRefreshesLeft(body.refreshesLeft);
            if (body.capExceeded) {
                console.log('account-sync: mirror migration deferred (daily full-pull cap spent) — retrying next boot');
                return;
            }
            rows.push(...body.reports);
            if (Number.isInteger(body.cursorId) && body.cursorId > sinceId) sinceId = body.cursorId;
            if (!body.hasMore) { complete = true; break; }
        }
        if (!complete) return;

        // Snapshot locals AFTER the pull: the matching + rebuild below is one
        // synchronous block, so a solve that lands during the (long) pull is
        // in this snapshot and can't be clobbered by the swap.
        let tail;
        try { tail = await getTail(); } catch (_) { tail = []; }
        const built = buildMirror(rows, getWorkingSetEntries(), tail);

        // Swap: tail first (one transaction), then the working set, then the
        // sync bookkeeping, then the flag — a crash between any two steps
        // leaves the flag unset and the whole migration redoes idempotently.
        // Same split rule as the steady-state eviction (cap + un-acked
        // hold-back). A concurrently-firing eviction timer can at worst
        // re-append a few working-set rows to the tail as exact duplicates —
        // absorbed by the standing (problemId, date) de-dup, never a loss.
        const cut = Math.max(0, built.entries.length - ANSWER_HISTORY_WORKING_SET);
        let wsRows = built.entries;
        if (cut > 0) {
            const tailRows = built.entries.slice(0, cut).filter((a) => !isUnackedReportable(a));
            const ok = await replaceTail(tailRows);
            if (ok) {
                const tailSet = new Set(tailRows);
                wsRows = built.entries.filter((a) => !tailSet.has(a));
            }
            // IDB unavailable: keep everything in the working set, same
            // degradation as the steady-state eviction.
        } else {
            await replaceTail([]);
        }
        replaceUnifiedHistory(wsRows);
        await syncTailExpectedToActual();
        persistAbsorbedIds(built.absorbed);

        const sync = readSyncState() || { userId: state.userId };
        if (!Number.isInteger(sync.cursorId) || sinceId > sync.cursorId) sync.cursorId = sinceId;
        if (lastBody && Number.isFinite(lastBody.serverSeq) &&
            (!Number.isFinite(sync.lastSeq) || lastBody.serverSeq > sync.lastSeq)) {
            sync.lastSeq = lastBody.serverSeq;
        }
        writeSyncState(sync);
        localStorage.setItem(MIRROR_MIGRATED_KEY, '1');
        localStorage.setItem(ID_CURSOR_MIGRATED_KEY, '1'); // subsumed: cursor set from the drained frontier
        stats = { rows: rows.length, ...built.counts };
        console.log(`account-sync: mirror-by-id migration complete — ${stats.rows} rows, matched ${stats.matched}, +${stats.skeletons} new, ${stats.absorbed} absorbed, ${stats.dropped} stale local copies dropped, ${stats.outbox} left for the outbox`);
    } catch (e) {
        console.warn('account-sync: mirror migration failed (retrying next boot):', e);
    } finally {
        pullInFlight = false;
    }
    if (!stats) return;
    // Fill the rebuilt history's model fields (skeletons + tail) and silently
    // recompute the models/rank, exactly like a merge would.
    try {
        await fillSyncedHistoryFields(SUPPORTED_BOARD_SIZES, { includeTail: true });
    } catch (e) {
        console.warn('account-sync: post-migration field fill failed (next pull retries):', e);
    }
    retrainAfterMerge([], 'feed');
}

// The pure rebuild behind migrateMirrorById: match local entries to server
// rows and return the new unified history plus the receipt ids. Matching per
// row, in id order: exact serverId → exact stamp + identical content (pre-id
// acked entries) → fuzzy claim against un-acked locals (lost acks and
// pre-sync-era live history). A row matching nothing is either a redundant
// identical twin of an already-matched entry (→ absorbed receipt) or genuinely
// new to this device (→ synced skeleton). Local entries no row claimed lose
// any stale stamps; copies of attempts the mirror already holds are dropped,
// the rest kept (outbox / local-only).
function buildMirror(rows, ws, tail) {
    const pool = [];
    for (const a of tail) if (a) pool.push(a.size != null ? a : { ...a, size: DEFAULT_BOARD_SIZE });
    for (const a of ws) if (a) pool.push(a.size != null ? a : { ...a, size: DEFAULT_BOARD_SIZE });

    const byServerId = new Map();
    const byStamp = new Map();
    const noteStamp = (a) => {
        let list = byStamp.get(a.serverCreatedAt);
        if (!list) { list = []; byStamp.set(a.serverCreatedAt, list); }
        list.push(a);
    };
    for (const a of pool) {
        if (Number.isFinite(a.serverId)) byServerId.set(a.serverId, a);
        if (a.serverCreatedAt) noteStamp(a);
    }

    const claimed = new Set();
    const mirror = [];
    const absorbed = [];
    const skeletons = [];
    for (const row of rows) {
        if (typeof row.createdAt !== 'string' || typeof row.problem !== 'string') continue;
        const ts = Date.parse(row.createdAt);
        if (!Number.isFinite(ts)) continue;
        const size = SUPPORTED_BOARD_SIZES.includes(row.size) ? row.size : DEFAULT_BOARD_SIZE;

        let e = Number.isFinite(row.id) ? byServerId.get(row.id) : null;
        if (e && !claimed.has(e)) {
            claimed.add(e);
            e.serverCreatedAt = row.createdAt; // authoritative refresh
            mirror.push(e);
            continue;
        }

        const stampCands = byStamp.get(row.createdAt);
        e = stampCands && stampCands.find((a) =>
            !claimed.has(a) && a.size === size && rowMatchesEntryContent(row, ts, a));
        if (e) {
            claimed.add(e);
            if (Number.isFinite(row.id)) e.serverId = row.id;
            mirror.push(e);
            continue;
        }

        let best = null;
        let bestGap = FUZZY_WINDOW_MS;
        for (const a of pool) {
            if (claimed.has(a) || a.serverCreatedAt || a.size !== size) continue;
            if (!rowMatchesEntryContent(row, ts, a)) continue;
            const gap = Math.abs((a.date || 0) - ts);
            if (gap < bestGap) { best = a; bestGap = gap; }
        }
        if (best) {
            claimed.add(best);
            best.serverCreatedAt = row.createdAt;
            if (Number.isFinite(row.id)) best.serverId = row.id;
            noteStamp(best);
            mirror.push(best);
            continue;
        }

        if (Number.isFinite(row.id) && stampCands &&
            stampCands.some((a) => claimed.has(a) && a.size === size && rowMatchesEntryContent(row, ts, a))) {
            absorbed.push(row.id);
            continue;
        }

        const entry = buildSyncedEntry(row, ts, size);
        claimed.add(entry);
        noteStamp(entry);
        mirror.push(entry);
        skeletons.push(entry);
    }

    const mirroredKeys = new Set(mirror.map((a) => `${a.problemId}|${a.date}`));
    const kept = [];
    let dropped = 0;
    for (const a of pool) {
        if (claimed.has(a)) continue;
        if (a.serverCreatedAt !== undefined) delete a.serverCreatedAt;
        if (a.serverId !== undefined) delete a.serverId;
        if (a.problemId && mirroredKeys.has(`${a.problemId}|${a.date}`)) { dropped++; continue; }
        kept.push(a);
    }

    const entries = mirror.concat(kept).sort((x, y) => (x.date || 0) - (y.date || 0));
    recomputeProblemChains(entries, new Set(skeletons.map((s) => s.problemId)));
    return {
        entries,
        absorbed,
        counts: {
            matched: mirror.length - skeletons.length,
            skeletons: skeletons.length,
            absorbed: absorbed.length,
            dropped,
            outbox: kept.filter((a) => isUnackedReportable(a)).length,
        },
    };
}

// Regular syncs: app load, tab becoming visible, and regaining connectivity.
// No interval polling: an active session is kept in sync by the per-report seq
// signal (checkSyncSignal). Each trigger pulls unthrottled — an up-to-date
// incremental poll reads ~0–1 D1 rows, and pullInFlight collapses overlapping
// triggers — so a session resume always merges (and can refresh the shown
// problem) immediately. The outbox rides the same triggers so offline attempts
// upload as soon as connectivity returns; it throttles internally (its flush
// is a full fromStart pull, the expensive path), and a pull's own digest check
// escalates to a reconcile if a gap is found.
export function startBackgroundSync() {
    // Migration order matters: the tail rescue first (so stranded un-acked
    // tail rows participate in the mirror rebuild's matching), then the mirror
    // migration (which subsumes the id-cursor migration when its full path
    // runs), then the normal pull + outbox on the resulting representation.
    migrateIdCursor()
        .then(() => rescueUnackedTailRows())
        .then(() => migrateMirrorById())
        .then(() => pullReports())
        .then(() => flushOutbox())
        .catch(() => {});
    window.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
            // Refresh the shown problem on EVERY resume, merge or not: a recall
            // can enter its window from time alone while the tab was hidden
            // (single-client case). Runs after the pull so a merged foreign
            // batch is reflected too (a merge already fires the refresh itself;
            // the second run no-ops via its in-flight/same-problem guards), and
            // before the outbox flush, whose full-pull path is slow.
            pullReports()
                .then(() => refreshShownProblemAfterSync())
                .then(() => flushOutbox())
                .catch(() => {});
        }
    });
    window.addEventListener('online', () => {
        pullReports().then(() => flushOutbox({ force: true })).catch(() => {});
    });
}
