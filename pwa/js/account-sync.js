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
// historical created_at (an offline flush, a backfill).
//
// LOCAL REPRESENTATION IS MIRROR-BY-ID: the acked history is a 1:1 mirror of
// the server's rows keyed by serverId — identity is the row id, and a row is
// "had" iff its id is held (working set or IndexedDB tail). Merging needs no
// timestamp dedup, so the local digest matches the server's BY CONSTRUCTION.
// A full from-0 reconcile is a TRUE MIRROR sync in both directions: rows the
// device lacks merge in, and local acked rows the server no longer has are
// deleted (the deletion pass — see pullReports). Un-acked entries (the
// outbox) are never touched by either direction.
//
// EVENT TIME IS CLIENT-OWNED: the client
// mints one timestamp per attempt, stores it as the entry's `date`, and sends it
// as /report's createdAt; the server inserts it VERBATIM or refuses (never
// rewrites), so row.created_at === entry.date exactly and own-row matching is
// exact — a lost-ack row is claimed by an EXACT date match, no fuzzy window.
//
// SUM-FIRST DIGEST: the per-profile digest (user_seq count + Σ rowHash(id))
// rides every /sync response AND every /report response. Count and sum are
// co-equal divergence signals — any mismatch (gap / surplus / content)
// escalates to the mirror reconcile, and every divergence direction converges
// to an exact match. The per-report check (checkSyncSignal) makes an active
// session self-verifying with no polling.
//
//   - cursorId only ever advances from /sync PULLS, never from this device's own
//     report acks. Advancing on acks would let a concurrent foreign row with a
//     lower id slip behind the cursor forever.
//   - The device's own rows are recognized on pull by their id: /report returns
//     the row id and sendReport stamps it on the matching answerHistory entry
//     (recordReportAck; serverId presence = acked).
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
    getMergedAnswerHistory, updateAnswerHistory,
    evictColdTail, clearHistoryTail, syncTailExpectedToActual,
    isUnackedReportable, applyStoredIgnoredFlags,
    SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE, generateUserId,
} from './state.js';
import { HISTORY_BACKFILL_DONE_KEY } from './constants.js';
import { calculateAndStoreStatsResults } from './recall-model.js';
import { reanchorRank } from './rank-decomposition.js';
import { fillSyncedHistoryFields, stampVersionsForFreshHistory } from './backfill.js';
import { refreshShownProblemAfterSync } from './problem-loader.js';
import { onUserReturn } from './user-return.js';
import { getTailSyncIndex, getTailDigestInfo, pruneTail } from './history-store.js';
import { rowHash } from '../shared/sync-digest.js';
import { readJSON, writeJSON } from './persist.js';
import { postJson } from './worker-post.js';
import { updateAllUI } from './ui.js';

const SYNC_STATE_KEY = 'accountSyncState';
// Pending-adoption flag: '1' = a user-initiated account switch (falls back to
// a FRESH account if the restore pull fails — the old account was deliberately
// wiped); 'epoch' = the state-epoch gate's re-adoption of the device's OWN
// account (state-epoch.js; on failure the flag stays set and the restore
// retries next boot — the userId is the user's credential and must never be
// silently replaced).
const SWITCH_FLAG_KEY = 'accountSwitchPending';
// Pagination safety bound: 100 pages × 1000 rows covers any realistic history.
const MAX_PAGES = 100;

// Acked = carries the server row id (identity / digest key; its presence =
// acked). Mirror-by-id is the only representation.
function isAckedEntry(a) {
    return Number.isFinite(a.serverId);
}

// Does this server row describe the same attempt as the local entry? The
// identity test behind the lost-ack claim: problem, outcome, and repeat flag
// must match. Deliberately ignores time_used/review_time.
function rowMatchesEntryContent(row, entry) {
    return entry.problemId === row.problem &&
        !!entry.correct === !!row.success &&
        !!entry.isRepeat === !!row.isRepeat;
}

// Build the local skeleton entry for a server row nothing local claimed;
// fillSyncedHistoryFields later fills the model fields.
function buildSyncedEntry(row, ts, size) {
    const entry = {
        problemId: row.problem,
        correct: !!row.success,
        date: ts,
        size,
        serverId: row.id,
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
    // Every stored move is in the canonical orientation-0 frame (the client
    // inverse-transforms the click through the served orientation; the DB
    // surgery NULLed the pre-canonical-era served-frame repeat moves), so the
    // coordinate maps onto the shard SGF and the backfill can derive
    // pickedCellIndex from it — repeats included.
    if (!row.success && typeof row.move === 'string' && row.move !== 'pass') {
        entry.syncedMove = row.move;
    }
    return entry;
}

// The userId match is the only validity requirement — the state must belong to
// the active account.
function readSyncState() {
    const parsed = readJSON(SYNC_STATE_KEY);
    if (parsed && parsed.userId === state.userId) {
        return parsed;
    }
    return null;
}

// The account's lifetime attempt count as the server last reported it, or null
// before any /report or sync has answered for this account.
export function syncedAttemptTotal() {
    const sync = readSyncState();
    return sync && Number.isFinite(sync.lastSeq) ? sync.lastSeq : null;
}

function writeSyncState(sync) {
    writeJSON(SYNC_STATE_KEY, sync);
}

export function isValidAccountCode(code) {
    // 8 chars, matching the Worker's USER_ID_RE alphabet.
    return typeof code === 'string' && /^[A-Za-z0-9_-]{8}$/.test(code);
}

// Stamp the acked row id onto the answerHistory entry the report described, so
// pulls can skip this device's own rows (serverId is the identity / digest key;
// its presence = acked). `atMs` is the event time sendReport sent as createdAt —
// the entry was created with the SAME ms as its date, so the match is exact.
// Called from sendReport with the /report response. The entry is created by
// updateUserStats microtasks after sendReport fires, normally long before the
// network roundtrip returns — the one retry covers the pathological ordering.
export function recordReportAck(problemId, atMs, serverId = null) {
    if (!Number.isFinite(serverId)) return;
    const attach = () => {
        const history = state.answerHistory;
        // Exact event-time match first; fall back to the newest un-acked entry
        // for the same problem so an ack is never dropped.
        let target = null;
        for (let i = history.length - 1; i >= 0; i--) {
            const a = history[i];
            if (a.problemId !== problemId || isAckedEntry(a)) continue;
            if (a.date === atMs) { target = a; break; }
            if (!target) target = a;
        }
        if (!target) return false;
        target.serverId = serverId;
        updateAnswerHistory(history);
        return true;
    };
    if (!attach()) setTimeout(attach, 1500);
}

// Rebase a not-yet-acked entry's event time to the server's clock and PERSIST,
// before sendReport resends a bound-refused report with the corrected
// timestamp. Persist-first is the crash-safety invariant: from the moment the
// server could hold the corrected row, the local entry already carries the
// identical date, so an abrupt kill anywhere (mid-resend, before the ack)
// leaves entry === row and the next pull's exact claim heals it — never a
// duplicate. Same attach-or-retry pattern as recordReportAck (the entry is
// created by updateUserStats after sendReport fires; the refusal roundtrip
// normally means it exists by now). Resolves true when an entry was rebased.
export function rebaseUnackedEntryDate(problemId, fromMs, toMs) {
    const attach = () => {
        const history = state.answerHistory;
        let target = null;
        for (let i = history.length - 1; i >= 0; i--) {
            const a = history[i];
            if (a.problemId !== problemId || isAckedEntry(a)) continue;
            if (a.date === fromMs) { target = a; break; }
            if (!target) target = a;
        }
        if (!target) return false;
        target.date = toMs;
        updateAnswerHistory(history);
        return true;
    };
    return new Promise((resolve) => {
        if (attach()) { resolve(true); return; }
        setTimeout(() => resolve(attach()), 1500);
    });
}

// Merge one page of /sync rows into the given size's answerHistory (id-first).
// Identity is the row id: a row is "had" iff its id is in the working set or
// the tail — no timestamp dedup. Rows not held try the EXACT claim against
// un-acked entries (a lost-ack row equals its recording entry's date to the
// millisecond, client-owned event time); everything else inserts as a
// skeleton. Synchronous on purpose: no await between read and write of the
// history bucket, so a concurrent solve can't be clobbered. Returns the
// inserted entry objects (live references — the field fill mutates them in
// place, and the post-merge rank re-fit reads the merged history afterwards).
function mergeRowsForSize(size, rows, tailIndex) {
    const history = getAnswerHistoryForSize(size);
    const wsIds = new Set();
    for (const a of history) {
        if (Number.isFinite(a.serverId)) wsIds.add(a.serverId);
    }

    let claimed = 0;
    const newEntries = [];
    for (const row of rows) {
        if (typeof row.createdAt !== 'string' || typeof row.problem !== 'string') continue;
        if (!Number.isFinite(row.id)) continue;   // every server row carries its id
        if (wsIds.has(row.id) || tailIndex.ids.has(row.id)) continue;
        const ts = Date.parse(row.createdAt);
        if (!Number.isFinite(ts)) continue;

        // Exact claim: a lost-ack row equals its local entry's date exactly.
        let best = null;
        for (const a of history) {
            if (isAckedEntry(a) || !rowMatchesEntryContent(row, a)) continue;
            if (a.date === ts) { best = a; break; }
        }
        if (best) {
            best.serverId = row.id;
            wsIds.add(row.id);
            claimed++;
            continue;
        }

        const entry = buildSyncedEntry(row, ts, size);
        wsIds.add(row.id);
        newEntries.push(entry);
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

// Mirror-complete deletion pass, run only after a COMPLETE from-0 pull:
// `seenIds` is every row id the server returned, so any local ACKED entry
// whose id is not in it no longer exists server-side — delete it, from the
// working set AND the IndexedDB tail. The same tail walk collapses evict-seam
// duplicates (an id present in both stores after a crash mid-evict; under the
// sum-first digest a persistent seam dup double-counts into a 'surplus' a
// reconcile could otherwise never close). Un-acked entries (the outbox) and
// entries claimed during this pull (their ids are in seenIds) are untouched.
async function mirrorDeletePass(seenIds) {
    let removed = 0;
    const wsIds = new Set();
    for (const size of SUPPORTED_BOARD_SIZES) {
        const history = getAnswerHistoryForSize(size);
        const kept = history.filter((a) => {
            if (!isAckedEntry(a)) return true;
            if (!seenIds.has(a.serverId)) { removed++; return false; }
            wsIds.add(a.serverId);
            return true;
        });
        if (kept.length !== history.length) setAnswerHistoryForSize(size, kept);
    }
    let tailRemoved = 0;
    try {
        // One cursor walk, three delete rules: rows absent from the server,
        // ws∩tail seam duplicates, and tail-INTERNAL duplicates (the same
        // serverId appended twice — e.g. damage minted by the old snapshot-
        // restore bug in backfill.persistAnswerHistory; the first copy wins).
        const seenInTail = new Set();
        const n = await pruneTail((v) => {
            if (!Number.isFinite(v && v.serverId)) return false;
            if (!seenIds.has(v.serverId) || wsIds.has(v.serverId)) return true;
            if (seenInTail.has(v.serverId)) return true;
            seenInTail.add(v.serverId);
            return false;
        });
        if (typeof n === 'number') tailRemoved = n;
    } catch (_) { /* tail unavailable — the working-set pass already ran */ }
    if (tailRemoved > 0) await syncTailExpectedToActual();
    if (removed + tailRemoved > 0) {
        console.log(`account-sync: mirror deletion removed ${removed} working-set + ${tailRemoved} tail rows absent from the server`);
    }
    return removed + tailRemoved;
}

// Pull reports newer than the cursor and merge them. Deliberately unthrottled:
// an up-to-date incremental poll reads ~0–1 D1 rows (idx_tsumego_reports_user_id),
// so load / tab-visible / online can each pull freely — pullInFlight is the
// only gate. `mode` controls what the merge does to the rank estimate (see
// retrainAfterMerge): 'feed' (default, ongoing sync), 'adopt' (account
// switch), 'reconcile' (digest-triggered mirror refresh — models only, never
// the rank). `fromStart` ignores the stored cursor and pulls the full history;
// a COMPLETE from-start pull also runs the mirror deletion pass, making the
// local acked set equal the server's rows exactly. Returns
// { inserted, complete } — `complete` is true only if the pull drained the
// server (no page failed mid-way), which the outbox requires before it may
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
        // with a historical created_at — an offline flush, a backfill). A
        // fresh/empty device just pulls from 0.
        let sinceId = fromStart ? 0 : (Number.isInteger(sync.cursorId) ? sync.cursorId : 0);
        const startedFromBeginning = fromStart || !Number.isInteger(sync.cursorId);
        let complete = false;
        let capExceeded = false;
        let refreshesLeft;
        let lastBody = null;
        const insertedEntries = [];
        const touchedSizes = new Set();
        // Every row id the server returned this pull — the deletion pass input.
        const seenIds = startedFromBeginning ? new Set() : null;
        // Settle any pending debounced eviction BEFORE snapshotting the tail:
        // the tail index is read once up front, so an eviction from a PREVIOUS
        // cycle landing mid-pull would be invisible to the merge — the rows
        // re-insert as working-set copies and the deletion pass then removes
        // the legitimate tail copies as seam dups, churning digest-triggered
        // reconciles against each other (observed on the heaviest account
        // during the epoch rollout). After the flush the snapshot is complete:
        // rows evicted DURING this pull were merged this pull (so already
        // known), and /sync pages are disjoint by cursor, so no row re-arrives
        // within one pull.
        try { await evictColdTail(); } catch (_) { /* tail unavailable — snapshot covers ws only */ }
        let tailIndex = null;
        try { tailIndex = await getTailSyncIndex(); } catch (_) { tailIndex = null; }
        if (!tailIndex) tailIndex = { ids: new Set() };
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
                if (seenIds && Number.isFinite(row.id)) seenIds.add(row.id);
                const size = SUPPORTED_BOARD_SIZES.includes(row.size) ? row.size : DEFAULT_BOARD_SIZE;
                if (!bySize.has(size)) bySize.set(size, []);
                bySize.get(size).push(row);
            }
            for (const [size, rows] of bySize) {
                // Hand the raw rows to an optional observer BEFORE the merge
                // mutates anything — the outbox's claim soft-matches against
                // them.
                if (rowSink) rowSink(size, rows);
                const inserted = mergeRowsForSize(size, rows, tailIndex);
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

        // Mirror-complete deletion: only a COMPLETE from-the-beginning pull saw
        // the server's full row set, so only then is "not seen" proof of
        // absence. A capExceeded / failed / incremental pull keeps the merge
        // additive (still correct on partial data).
        let removed = 0;
        if (startedFromBeginning && complete && seenIds) {
            try { removed = await mirrorDeletePass(seenIds); } catch (e) {
                console.warn('account-sync: mirror deletion pass failed:', e);
            }
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

        if (insertedEntries.length > 0 || removed > 0) {
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
        // local digest (acked-row count AND Σ rowHash) matches the server's.
        // Any mismatch escalates to a full reconcile. Deferred a tick so it
        // runs AFTER pullInFlight is released (the reconcile is itself a
        // pullReports). fromStart / reconcile pulls ARE the reconcile, so they
        // never re-check.
        if (!fromStart && mode === 'feed' && lastBody && Number.isFinite(lastBody.serverSeq)) {
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
// over them. Compared against the server's user_seq / digest. Every acked
// entry carries its serverId (mirror-by-id), so both values are always
// well-defined.
async function computeLocalDigest() {
    let ackedCount = 0, idSum = 0;
    for (const a of getMergedAnswerHistory()) {
        if (!isAckedEntry(a)) continue;
        ackedCount++;
        idSum += rowHash(a.serverId);
    }
    try {
        const t = await getTailDigestInfo();
        ackedCount += t.ackedCount;
        idSum += t.idSum;
    } catch (_) { /* tail unavailable → working-set-only digest */ }
    return { ackedCount, idSum };
}

// What (if anything) diverges between the local digest and the server summary.
// SUM-FIRST: count and sum are co-equal — divergence iff either differs. The
// kinds ('gap' = missing rows, 'surplus' = extra local rows, 'content' = same
// count, different id set) survive only as log labels; every kind triggers
// the same mirror reconcile, and every direction converges to an exact match.
function digestMismatch(serverSeq, serverDigest, { ackedCount, idSum }) {
    if (!Number.isFinite(serverSeq) || !Number.isFinite(serverDigest)) return null;
    if (ackedCount < serverSeq) return 'gap';
    if (ackedCount > serverSeq) return 'surplus';
    if (idSum !== serverDigest) return 'content';
    return null;
}

// Backstop divergence check: compare the local digest to a server summary and
// escalate any mismatch to the full mirror reconcile. With mirror-by-id +
// the deletion pass there is no permanently unmergeable divergence — every
// pulled row matches an id, claims an un-acked entry, or becomes a skeleton,
// and every unseen local acked row is deleted — so no accept-and-remember
// state is needed; a reconcile that somehow fails to converge just logs.
async function checkInSync(serverSeq, serverDigest) {
    const local = await computeLocalDigest();
    const kind = digestMismatch(serverSeq, serverDigest, local);
    if (!kind) return;
    console.log(`account-sync: digest ${kind} (acked ${local.ackedCount}/Σ ${local.idSum} vs server ${serverSeq}/Σ ${serverDigest}) — full reconcile`);
    const r = await refreshHistoryFromServer();
    if (r && r.ok && digestMismatch(serverSeq, serverDigest, await computeLocalDigest())) {
        console.warn('account-sync: reconcile did not close the digest divergence (retries on the next signal)');
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
//   - 'reconcile' (digest-triggered refresh): models retrain, but the rank is
//     untouched — the affected rows are ancient attempts already in the window or
//     predating it.
function retrainAfterMerge(insertedEntries, mode) {
    try { calculateAndStoreStatsResults(); } catch (e) { console.warn('account-sync: recall retrain failed:', e); }
    try {
        // Same call for both modes, because the estimate is a pure function of the
        // merged history: there is no warm start to seed differently and no loop
        // state to carry, so an adoption and an ongoing merge are one operation.
        // 'adopt' runs before the first shard is loaded and that costs it nothing —
        // the SE floor comes from the shipped corpus curve (seFloorAt), not from the
        // problems this device happens to hold — and what it writes selects the
        // first shard. 'reconcile' is excluded: those rows are ancient.
        if (mode === 'adopt' || mode === 'feed') reanchorRank(getMergedAnswerHistory());
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
// maybeFinishAccountSwitch, which pulls the full server history from id 0.
// Settings that aren't account-derived (board size, audio, language, corpus
// version stamps) survive the wipe.
// Everything derived from the current account's history — wiped on adoption
// (switchAccount) and again when a failed adoption falls back to a fresh
// account (maybeFinishAccountSwitch drops the partially merged rows).
const ACCOUNT_WIPE_KEYS = [
    'answerHistory', 'challengeProblem', 'previousProblem',
    'userRank', 'serveRank', 'displayRank', 'rankSe',
    'modelParams', 'adamState',
    'weaknessParams', 'weaknessAdam', 'weaknessSe', 'weaknessAxisCov',
    'recallPausedUntil',
    // The old account's curated ignored-problems list.
    'ignoredProblems_v1',
    // The server-side recall schedule snapshot belongs to the old account;
    // the new account's first upload must go replace-all.
    'recallScheduleSnapshot',
    'syncTailFillPending',
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
// against the daily full-pull cap). Returns { serverMaxId, hasReports } or
// null when the request itself failed (offline, server error, malformed body).
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
// iff it has at least one report, or — only when it has none — a leaderboard
// identity of its own (a chosen alias or a linked EGD/AGA binding), since an
// account can hold an identity before its first attempt. A code with neither is
// indistinguishable from a typo. The identity read is a second request made
// only for the no-reports case. Returns true/false, or null when a check itself
// failed.
export async function accountExists(userId) {
    const probe = await probeFrontier(userId);
    if (probe === null) return null;
    if (probe.hasReports) return true;
    try {
        const res = await fetch(`/profile?userId=${encodeURIComponent(userId)}`);
        if (!res.ok) return null;
        const body = await res.json().catch(() => null);
        if (!body || typeof body !== 'object') return null;
        return !!(body.alias || body.linked);
    } catch {
        return null;
    }
}

export function hasPendingAccountSwitch() {
    return !!localStorage.getItem(SWITCH_FLAG_KEY);
}

// Which flavour of adoption is pending (see SWITCH_FLAG_KEY): 'switch' when the
// user entered another account's code, 'epoch' when the state-epoch gate is
// re-adopting this device's OWN account. Only the first is a login, and the two
// need different wording — an epoch restore that says "logging in" describes an
// account change that is not happening.
export function pendingAccountSwitchKind() {
    const flag = localStorage.getItem(SWITCH_FLAG_KEY);
    if (!flag) return null;
    return flag === 'epoch' ? 'epoch' : 'switch';
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
// Handles BOTH adoption flavors (see SWITCH_FLAG_KEY): a user-initiated
// switch ('1') falls back to a fresh account when the restore pull fails,
// while the state-epoch re-adoption ('epoch') keeps the flag and retries next
// boot — the device's own userId is the user's credential and must survive.
// Returns true if a switch was completed (caller may want a spinner around it).
export async function maybeFinishAccountSwitch() {
    const flag = localStorage.getItem(SWITCH_FLAG_KEY);
    if (!flag) return { switched: false };
    const isEpochAdopt = flag === 'epoch';
    // The IndexedDB cold tail belongs to the pre-adoption state (switchAccount
    // and the epoch gate wipe localStorage but can't await an async IDB clear),
    // so drop it before pulling the account's history.
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
    if (result.capExceeded || !result.complete) {
        if (isEpochAdopt) {
            // The epoch restore failed (offline boot / cap): keep the flag so
            // the next boot retries, and continue on the wiped (empty) local
            // state — play works, reports ack normally, and the retry's from-0
            // pull claims those rows by id.
            console.warn(`account-sync: epoch restore ${result.capExceeded ? 'cap-deferred' : 'failed'} — retrying next boot`);
            return { switched: false };
        }
        // User-initiated switch: the adoption is only allowed to complete if
        // the FULL history arrived — a half-adopted device would report under
        // the account while serving from a blank (rank-0) profile. Fall back
        // to a FRESH account id (the old account's local state was already
        // wiped by switchAccount, so there is nothing to keep), drop any
        // partially merged rows, leave the notice for the next boot's alert,
        // and reload.
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
    // Re-apply the device's preserved ignored-problems list to the restored
    // history (the flags live on answerHistory entries, which the wipe + pull
    // rebuilt as skeletons). Epoch restores carry the pre-wipe list across;
    // a user-initiated switch wiped it (old account's curation).
    applyStoredIgnoredFlags();
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
export const HISTORY_REFRESH_DAILY_CAP = 5;
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

// Pull the full server history (fromStart) to restore evicted tail rows and
// converge any digest divergence (the from-0 pull runs the mirror deletion
// pass), then re-split into working set + tail and resync the eviction
// high-water mark. The daily cap is enforced by the Worker on the sinceId=0
// full pull (shared with account adoption) — if exceeded it returns
// capExceeded and serves no rows. `mode: 'reconcile'` retrains the models but
// leaves the rank loop untouched. Returns { ok, reason, restored, left }.
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

// Per-report digest check (sum-first): when the seq landed exactly where
// expected, the response's digest (the server's post-insert Σ rowHash) still
// verifies CONTENT — an equal-count/different-set divergence is caught on the
// next attempt instead of waiting for the next pull's summary. Deferred a few
// seconds so recordReportAck (and its 1.5s retry) has stamped the new row's id
// before the local digest is computed; recomputed at fire time, and skipped
// when a newer report has landed meanwhile (that report scheduled its own
// check with the newer digest) or a pull owns the state (its own summary
// check covers it).
const REPORT_DIGEST_CHECK_DELAY_MS = 3000;
let pendingReportDigest = null;
let reportDigestTimer = null;

function scheduleReportDigestCheck(seq, digest) {
    pendingReportDigest = { seq, digest };
    if (reportDigestTimer) return;
    reportDigestTimer = setTimeout(() => {
        reportDigestTimer = null;
        const pending = pendingReportDigest;
        pendingReportDigest = null;
        if (!pending || pullInFlight) return;
        const sync = readSyncState();
        if (!sync || sync.lastSeq !== pending.seq) return;
        checkInSync(pending.seq, pending.digest).catch(() => {});
    }, REPORT_DIGEST_CHECK_DELAY_MS);
}

// Called from sendReport with the /report response's userSeq + digest — the
// server's per-profile attempt count and post-insert Σ rowHash including this
// report. A seq past the device's expected lastSeq+1 means another device
// wrote in between → pull. A seq exactly as expected still verifies the sum
// (scheduleReportDigestCheck). lastSeq advances either way (and is also kept
// current from /sync's serverSeq). This is the in-sync signal that lets an
// active session need no polling.
export function checkSyncSignal(userSeq, digest) {
    if (!Number.isFinite(userSeq)) return;
    const sync = readSyncState();
    if (!sync) return;
    const expected = Number.isFinite(sync.lastSeq) ? sync.lastSeq + 1 : userSeq;
    const foreign = userSeq > expected;
    sync.lastSeq = userSeq;
    writeSyncState(sync);
    if (foreign) {
        maybeSignalPull();
        return;
    }
    if (Number.isFinite(digest)) scheduleReportDigestCheck(userSeq, digest);
}

// --- Durable offline outbox --------------------------------------------------
// Any attempt recorded locally but never acked (an offline / failed /report) is
// eventually uploaded so it reaches the server and every other device.
// Permanent (not a one-off): runs on load, on tab-visible, and on the `online`
// event whenever un-acked entries exist — a cheap no-op otherwise,
// self-throttled, bounded by the server daily full-pull cap.
//   1. Claim-before-upload — a COMPLETE reconcile pull first, so the exact
//      claim in mergeRowsForSize claims any row the server ALREADY has (an
//      ack-lost report) instead of re-uploading it as a duplicate.
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
// historical date (≥1h old, so nothing in-flight), and no serverId.
function collectUnackedEntries() {
    const out = [];
    const cutoff = Date.now() - 3600 * 1000;
    for (const size of SUPPORTED_BOARD_SIZES) {
        for (const a of getAnswerHistoryForSize(size)) {
            if (!a || isAckedEntry(a)) continue;
            if (typeof a.problemId !== 'string' || !/^[0-9a-f]{8}$/.test(a.problemId)) continue;
            if (typeof a.date !== 'number' || !(a.date < cutoff)) continue;
            out.push({ entry: a, size });
        }
    }
    out.sort((x, y) => x.entry.date - y.entry.date);
    return out;
}

function markOutboxDone() {
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

        // created_at must be unique per user (the exact claim keys on it), so
        // collisions — same-ms entries, or a clash with an existing acked
        // row — get nudged forward 1ms at a time. The nudge is written back to
        // entry.date too: the uploaded created_at must equal the entry's date
        // EXACTLY (the exact-match invariant).
        const taken = new Set();
        for (const size of SUPPORTED_BOARD_SIZES) {
            for (const a of getAnswerHistoryForSize(size)) {
                if (!a || !isAckedEntry(a)) continue;
                if (Number.isFinite(a.date)) taken.add(new Date(a.date).toISOString());
            }
        }
        const nudgedSizes = new Set();
        const rows = candidates.map(({ entry, size }) => {
            let ts = entry.date;
            let iso = new Date(ts).toISOString();
            while (taken.has(iso)) iso = new Date(++ts).toISOString();
            taken.add(iso);
            if (ts !== entry.date) { entry.date = ts; nudgedSizes.add(size); }
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
        // Persist any nudged dates BEFORE uploading: the uploaded created_at
        // must equal the PERSISTED entry date, or an abrupt kill between the
        // POST and the post-batch persist leaves entry.date one ms off the
        // server row — the exact claim then misses and the next flush would
        // re-upload it as a duplicate.
        for (const size of nudgedSizes) {
            setAnswerHistoryForSize(size, getAnswerHistoryForSize(size));
        }

        for (let i = 0; i < rows.length; i += OUTBOX_BATCH) {
            const batch = rows.slice(i, i + OUTBOX_BATCH);
            const res = await postJson('/report-backfill', { userId: state.userId, reports: batch.map(b => b.report) });
            if (!res.ok) {
                console.warn(`account-sync: outbox batch failed (${res.status}); will retry`);
                return;
            }
            const body = await res.json().catch(() => null);
            const ids = (body && Array.isArray(body.ids)) ? body.ids : [];
            // Stamp serverId (identity / digest key; its presence = acked) per
            // batch so a later failure never re-uploads what landed.
            const sizesTouched = new Set();
            const discard = new Set();
            batch.forEach((b, j) => {
                if (Number.isFinite(ids[j])) b.entry.serverId = ids[j];
                sizesTouched.add(b.size);
            });
            // Per-row rejections: 'future_timestamp' (this device's clock ran
            // ahead when the entry was recorded, or the row is <1h old) →
            // self-correct the entry's date to the server's clock so the next
            // flush retries inside bounds; 'invalid' (malformed junk) →
            // discard the entry outright. A rejected slot carries no id, so
            // the guarded stamp above never ran for it.
            const rejected = (body && Array.isArray(body.rejected)) ? body.rejected : [];
            for (const rej of rejected) {
                const b = batch[rej.i];
                if (!b) continue;
                if (rej.error === 'future_timestamp') {
                    const serverNowMs = Date.parse(rej.serverNow || '');
                    if (Number.isFinite(serverNowMs) && b.entry.date > serverNowMs) {
                        b.entry.date = serverNowMs;
                    }
                    // else: merely <1h old — it ages into the window by itself.
                } else {
                    discard.add(b.entry);
                }
                sizesTouched.add(b.size);
            }
            for (const size of sizesTouched) {
                let history = getAnswerHistoryForSize(size);
                if (discard.size) history = history.filter((a) => !discard.has(a));
                setAnswerHistoryForSize(size, history);
            }
        }

        // A rejected-and-corrected entry is still un-acked (it retries next
        // flush once its corrected date ages past the 1h window), so "done"
        // only when nothing reportable remains.
        if (collectUnackedEntries().length === 0) markOutboxDone();
        console.log(`account-sync: outbox flushed ${rows.length} attempts to the server`);
    } catch (e) {
        console.warn('account-sync: outbox flush failed (will retry):', e);
    }
}

// Regular syncs: app load, the user returning (tab re-shown, or input after an
// idle gap — user-return.js), and regaining connectivity.
// No interval polling: an active session is kept in sync by the per-report
// seq+digest signal (checkSyncSignal). Each trigger pulls unthrottled — an
// up-to-date incremental poll reads ~0–1 D1 rows, and pullInFlight collapses
// overlapping triggers — so a session resume always merges (and can refresh
// the shown problem) immediately. The outbox rides the same triggers so
// offline attempts upload as soon as connectivity returns; it throttles
// internally (its flush is a full fromStart pull, the expensive path), and a
// pull's own digest check escalates to a reconcile if a divergence is found.
export function startBackgroundSync() {
    pullReports()
        .then(() => flushOutbox())
        .catch(() => {});
    onUserReturn(() => {
        // Refresh the shown problem on EVERY return, merge or not: a recall
        // can enter its window from time alone while the user was away
        // (single-client case). Runs after the pull so a merged foreign
        // batch is reflected too (a merge already fires the refresh itself;
        // the second run no-ops via its in-flight/same-problem guards), and
        // before the outbox flush, whose full-pull path is slow.
        pullReports()
            .then(() => refreshShownProblemAfterSync())
            .then(() => flushOutbox())
            .catch(() => {});
    });
    window.addEventListener('online', () => {
        pullReports().then(() => flushOutbox({ force: true })).catch(() => {});
    });
}
