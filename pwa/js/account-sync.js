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
//   - cursorId only ever advances from /sync PULLS, never from this device's own
//     report acks. Advancing on acks would let a concurrent foreign row with a
//     lower id slip behind the cursor forever.
//   - The device's own rows are recognized on pull by exact serverCreatedAt
//     match: /report echoes its server-stamped created_at and sendReport stamps
//     it on the matching answerHistory entry (recordReportAck).
//   - Entries whose ack was lost (offline attempt, response dropped) are
//     claimed fuzzily: same (problemId, correct, isRepeat), no serverCreatedAt
//     yet, nearest local date within ±10 min, greedy 1-1.
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
// fillSyncedHistoryFields (backfill.js) over the same shard walk
// the version-driven backfill uses.

import {
    state,
    getAnswerHistoryForSize, setAnswerHistoryForSize,
    getMergedAnswerHistory, updateAnswerHistory, updateUserRank, updateServeRank,
    evictColdTail, clearHistoryTail, syncTailExpectedToActual,
    SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE, generateUserId,
} from './state.js';
import { TOTAL_RANKS, HISTORY_BACKFILL_DONE_KEY } from './constants.js';
import { calculateAndStoreStatsResults } from './model.js';
import { trainWeakness } from './weakness.js';
import { estimateWindowRank } from './rank-decomposition.js';
import { fillSyncedHistoryFields } from './backfill.js';
import { refreshShownProblemAfterSync } from './problem-loader.js';
import { getTailServerCreatedAtSet, getTailDigestInfo } from './history-store.js';
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
// Ack-less local entries are claimed by a server row when identity fields match
// and the timestamps agree within this window (local Date.now() at attempt vs
// edge time at insert — normally seconds apart; the slack absorbs clock skew).
const FUZZY_WINDOW_MS = 10 * 60 * 1000;
// Pagination safety bound: 100 pages × 1000 rows covers any realistic history.
const MAX_PAGES = 100;

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

        const entry = {
            problemId: row.problem,
            correct: !!row.success,
            date: ts,
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
        // The reported move is in the served board frame. New (non-repeat)
        // problems are always served at orientation 0 — identity — so the
        // coordinate maps straight onto the shard SGF and the backfill can
        // derive pickedCellIndex from it. Repeats are served at a random
        // orientation, unrecoverable from the report; skip those.
        if (!row.success && !row.isRepeat && typeof row.move === 'string' && row.move !== 'pass') {
            entry.syncedMove = row.move;
        }
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
        // cursor, so no row re-arrives within one pull.
        let tailKnown = null;
        try { tailKnown = await getTailServerCreatedAtSet(); } catch (_) { tailKnown = null; }
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
                const inserted = mergeRowsForSize(size, rows, tailKnown);
                if (inserted.length > 0) {
                    insertedEntries.push(...inserted);
                    touchedSizes.add(size);
                }
            }

            // Advance the id cursor. Never move it backwards: a fromStart pull
            // walks through old ids the normal cursor is already past.
            if (Number.isInteger(body.cursorId) && body.cursorId > sinceId) {
                sinceId = body.cursorId;
                if (!Number.isInteger(sync.cursorId) || body.cursorId > sync.cursorId) {
                    sync.cursorId = body.cursorId;
                    writeSyncState(sync);
                }
            }
            if (!body.hasMore) { complete = true; break; }
        }

        // Keep lastSeq (the seq-signal high-water) current from the latest
        // summary, so the next /report's userSeq is compared against the right
        // baseline.
        if (lastBody && Number.isFinite(lastBody.serverSeq) &&
            (!Number.isFinite(sync.lastSeq) || lastBody.serverSeq > sync.lastSeq)) {
            sync.lastSeq = lastBody.serverSeq;
            writeSyncState(sync);
        }

        if (insertedEntries.length > 0) {
            console.log(`account-sync: merged ${insertedEntries.length} foreign attempts (sizes: ${[...touchedSizes].join(', ')})`);
            try {
                // Fill model fields (incl. problemQuality, which the loop feed
                // below reads) before touching the loop state. A full pull also
                // fills the cold tail (its old rows were just evicted there).
                await fillSyncedHistoryFields([...touchedSizes], { includeTail: startedFromBeginning });
            } catch (e) {
                console.warn('account-sync: field fill failed (next app load retries):', e);
            }
            retrainAfterMerge(insertedEntries, mode);
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
    try {
        const t = await getTailDigestInfo();
        ackedCount += t.ackedCount;
        idSum += t.idSum;
        if (!t.allHaveId) allHaveId = false;
    } catch (_) { /* tail unavailable → working-set-only digest */ }
    return { ackedCount, idSum, allHaveId };
}

// Backstop divergence check (after an incremental pull). serverSeq null = the
// server's latest row isn't seeded yet (no post-migration report); skip.
async function checkInSync(serverSeq, serverDigest) {
    if (!Number.isFinite(serverSeq)) return;
    const { ackedCount, idSum, allHaveId } = await computeLocalDigest();
    if (ackedCount < serverSeq) {
        console.log(`account-sync: digest gap (acked ${ackedCount} < server ${serverSeq}) — full reconcile`);
        await refreshHistoryFromServer();
        return;
    }
    if (allHaveId && ackedCount === serverSeq && Number.isFinite(serverDigest) && idSum !== serverDigest) {
        console.log(`account-sync: digest content mismatch (Σ ${idSum} != server ${serverDigest}) — full reconcile`);
        await refreshHistoryFromServer();
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

export async function flushOutbox({ force = false } = {}) {
    if (collectUnackedEntries().length === 0) {
        // Nothing to flush — the device's history is fully server-side, so it's
        // safe to adopt elsewhere (ui.js reads this to reveal the Account menu).
        localStorage.setItem(OUTBOX_DONE_KEY, '1');
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
            localStorage.setItem(OUTBOX_DONE_KEY, '1');
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

        localStorage.setItem(OUTBOX_DONE_KEY, '1');
        console.log(`account-sync: outbox flushed ${rows.length} attempts to the server`);
    } catch (e) {
        console.warn('account-sync: outbox flush failed (will retry):', e);
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
    migrateIdCursor()
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
