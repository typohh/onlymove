// js/account-sync.js
//
// Multi-device account sync. The account credential IS the userId (an 8-char
// random token — see getOrCreateUserId in state.js): every attempt is already
// reported to D1 under it, so "logging in" on another device means adopting
// the token and rebuilding local history from the server's report log.
//
// Reports are append-only (nothing ever UPDATEs tsumego_reports), so change
// detection needs no content hash — a created_at cursor is complete:
//
//   - The cursor only ever advances from /sync PULLS, never from this device's
//     own report acks. Advancing on acks would let a concurrent foreign row
//     with an earlier timestamp slip behind the cursor forever.
//   - The device's own rows are recognized on pull by exact serverCreatedAt
//     match: /report echoes its server-stamped created_at and sendReport stamps
//     it on the matching answerHistory entry (recordReportAck).
//   - Entries whose ack was lost (offline attempt, response dropped) are
//     claimed fuzzily: same (problemId, correct, isRepeat), no serverCreatedAt
//     yet, nearest local date within ±10 min, greedy 1-1.
//
// On an EXISTING device the cursor initializes to "now" — its local state
// already reflects its own history, and pulling the backlog would only churn
// through the legacy fuzzy-dedup minefield. Only an explicit account switch
// (switchAccount) starts from '', pulling the full server history into a
// wiped localStorage.
//
// Merged rows become skeleton entries (problemId/correct/date/timeTaken/
// isRepeat + review accounting); the model fields (rating/rank/slope/floor/
// embeddings, picked-move cell index from the reported move) are filled by
// fillSyncedHistoryFields (embedding-backfill.js) over the same shard walk
// the version-driven backfill uses.

import {
    state,
    getAnswerHistoryForSize, setAnswerHistoryForSize,
    getMergedAnswerHistory, updateAnswerHistory, updateUserRank, updateMomentum,
    SUPPORTED_BOARD_SIZES, DEFAULT_BOARD_SIZE,
} from './state.js';
import { TOTAL_RANKS, HISTORY_BACKFILL_DONE_KEY } from './constants.js';
import { calculateAndStoreStatsResults } from './model.js';
import { trainWeakness } from './weakness.js';
import { computeRankDecomposition, evaluateAt } from './rank-decomposition.js';
import { momentumRankStep, replayMomentumRank, RANK_REPLAY_WINDOW } from './momentum-rank.js';
import { fillSyncedHistoryFields } from './embedding-backfill.js';
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
// Background pulls (load / tab-visible / heartbeat) are throttled; an
// up-to-date poll reads ~0 D1 rows but there's no reason to poll per minute.
const MIN_PULL_INTERVAL_MS = 10 * 60 * 1000;
// Pagination safety bound: 100 pages × 1000 rows covers any realistic history.
const MAX_PAGES = 100;

function readSyncState() {
    try {
        const parsed = JSON.parse(localStorage.getItem(SYNC_STATE_KEY));
        if (parsed && parsed.userId === state.userId && typeof parsed.cursor === 'string') {
            return parsed;
        }
    } catch { /* fall through */ }
    return null;
}

function writeSyncState(sync) {
    localStorage.setItem(SYNC_STATE_KEY, JSON.stringify(sync));
}

export function isValidAccountCode(code) {
    // 8 chars, matching the Worker's USER_ID_RE alphabet. Longer legacy ids
    // can't round-trip anyway: getOrCreateUserId truncates stored ids to 8.
    return typeof code === 'string' && /^[A-Za-z0-9_-]{8}$/.test(code);
}

// Stamp the server-side created_at onto the answerHistory entry the report
// described, so pulls can skip this device's own rows by exact match. Called
// from sendReport with the /report response. The entry is created by
// updateUserStats microtasks after sendReport fires, normally long before the
// network roundtrip returns — the one retry covers the pathological ordering.
export function recordReportAck(problemId, createdAt) {
    const attach = () => {
        const history = state.answerHistory;
        for (let i = history.length - 1; i >= 0; i--) {
            const a = history[i];
            if (a.problemId === problemId && !a.serverCreatedAt) {
                a.serverCreatedAt = createdAt;
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
// the momentum-loop feed reads them afterwards).
function mergeRowsForSize(size, rows) {
    const history = getAnswerHistoryForSize(size);
    const known = new Set();
    for (const a of history) {
        if (a.serverCreatedAt) known.add(a.serverCreatedAt);
    }

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
            claimed++;
            continue;
        }

        const entry = {
            problemId: row.problem,
            correct: !!row.success,
            date: ts,
            serverCreatedAt: row.createdAt,
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
let lastPullAt = 0;

// Pull reports newer than the cursor and merge them. `mode` controls what the
// merge does to the momentum-rank loop state (see retrainAfterMerge): 'feed'
// (default, ongoing sync), 'adopt' (account switch), 'reconcile' (one-off
// local-history backfill — models only, never the loop; the merged rows are
// ancient). `fromStart` ignores the stored cursor and pulls the full history
// (the reconcile needs every row to claim against). Returns
// { inserted, complete } — `complete` is true only if the pull drained the
// server (no page failed mid-way), which the backfill requires before it may
// upload anything.
export async function pullReports({ force = false, mode = 'feed', fromStart = false, rowSink = null } = {}) {
    if (pullInFlight) return { inserted: 0, complete: false };
    if (!force && Date.now() - lastPullAt < MIN_PULL_INTERVAL_MS) return { inserted: 0, complete: false };
    pullInFlight = true;
    try {
        let sync = readSyncState();
        if (!sync) {
            // First sync for this userId on this device. Local state already
            // reflects everything this device did, so start from now; only an
            // account switch rebuilds from ''.
            sync = { userId: state.userId, cursor: new Date().toISOString() };
            writeSyncState(sync);
        }

        let since = fromStart ? '' : sync.cursor;
        let complete = false;
        const insertedEntries = [];
        const touchedSizes = new Set();
        for (let page = 0; page < MAX_PAGES; page++) {
            const res = await fetch('/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: sync.userId, since }),
            });
            if (!res.ok) break;
            const body = await res.json().catch(() => null);
            if (!body || !Array.isArray(body.reports)) break;

            const bySize = new Map();
            for (const row of body.reports) {
                const size = SUPPORTED_BOARD_SIZES.includes(row.size) ? row.size : DEFAULT_BOARD_SIZE;
                if (!bySize.has(size)) bySize.set(size, []);
                bySize.get(size).push(row);
            }
            for (const [size, rows] of bySize) {
                // Hand the raw rows to an optional observer BEFORE the merge
                // mutates anything — round-2 enrichment soft-matches against
                // them (it needs every server row, incl. the already-claimed
                // pre-era originals the merge skips).
                if (rowSink) rowSink(size, rows);
                const inserted = mergeRowsForSize(size, rows);
                if (inserted.length > 0) {
                    insertedEntries.push(...inserted);
                    touchedSizes.add(size);
                }
            }

            if (typeof body.cursor === 'string' && body.cursor) {
                since = body.cursor;
                // Never move the persisted cursor backwards: a fromStart pull
                // walks through old timestamps the normal cursor is already past.
                if (body.cursor > sync.cursor) {
                    sync.cursor = body.cursor;
                    writeSyncState(sync);
                }
            }
            if (!body.hasMore) { complete = true; break; }
        }

        lastPullAt = Date.now();
        if (insertedEntries.length > 0) {
            console.log(`account-sync: merged ${insertedEntries.length} foreign attempts (sizes: ${[...touchedSizes].join(', ')})`);
            try {
                // Fill model fields (incl. problemQuality, which the loop feed
                // below reads) before touching the loop state.
                await fillSyncedHistoryFields([...touchedSizes]);
            } catch (e) {
                console.warn('account-sync: field fill failed (next app load retries):', e);
            }
            retrainAfterMerge(insertedEntries, mode);
        }
        return { inserted: insertedEntries.length, complete };
    } catch (e) {
        console.warn('account-sync: pull failed:', e);
        return { inserted: 0, complete: false };
    } finally {
        pullInFlight = false;
    }
}

// Post-merge state update. Models always retrain off the merged history. The
// momentum-rank loop is handled so that the MLE touches it exactly ONCE per
// device — at adoption — and never again:
//
//   - adopt (account switch; no loop state exists for this account here):
//     hybrid recovery — rank LEVEL from the skill decomposition (an estimator,
//     robust to code eras / injected repairs / log gaps) evaluated at the
//     rank-replay window's start, then the live dynamics replayed on top
//     (momentum warmed long, rank integrated through the short window — see
//     replayMomentumRank for why both halves are load-bearing). If the account
//     is too small for the decomposition, replay everything from scratch —
//     onboarding-equivalent.
//
//   - ongoing merge ('feed'): feed the newly merged foreign attempts THROUGH
//     THE LIVE LOOP from the device's current state, exactly as if they'd been
//     solved here. No re-anchor, no MLE — the account behaves like one
//     continuous momentum loop fed by every device, so momentum stays fully
//     load-bearing for users who alternate devices. Incremental rows are
//     recent, current-era outcomes processed at the device's already-converged
//     momentum, so none of the long-replay pathologies (era-mismatch drift,
//     cold-start transient) apply. Devices agree up to interleaving order
//     (each processes the same multiset of attempts, merely batched
//     differently), which is second-order at converged momentum.
//
//   - 'reconcile' (one-off local-history backfill): models retrain, but the
//     loop is untouched — the merged rows are ancient attempts this device
//     either already lived through or that predate it.
function retrainAfterMerge(insertedEntries, mode) {
    try { calculateAndStoreStatsResults(); } catch (e) { console.warn('account-sync: recall retrain failed:', e); }
    try { trainWeakness(); } catch (e) { console.warn('account-sync: weakness retrain failed:', e); }
    try {
        if (mode === 'adopt') {
            const merged = getMergedAnswerHistory();
            const decomp = computeRankDecomposition(merged);
            let next = null;
            if (decomp) {
                const start = Math.max(0, merged.length - RANK_REPLAY_WINDOW);
                const at = evaluateAt(decomp, merged[start].date || Date.now());
                if (at && Number.isFinite(at.S)) {
                    next = replayMomentumRank(merged, at.S / TOTAL_RANKS);
                }
            } else {
                // Tiny account: no estimator support — replay the whole thing
                // from the cold-start state, like a fresh device onboarding.
                next = replayMomentumRank(merged, 0, {
                    maxLength: merged.length, rankWindow: merged.length,
                });
            }
            if (next && Number.isFinite(next.rank) && Number.isFinite(next.momentum)) {
                updateUserRank(next.rank);
                updateMomentum(next.momentum);
            }
        } else if (mode === 'feed') {
            feedAttemptsThroughLoop(insertedEntries);
        }
    } catch (e) {
        console.warn('account-sync: loop state update failed:', e);
    }
    updateAllUI();
}

// Run merged foreign attempts through the same step the live loop uses, from
// the current device state, in chronological order.
function feedAttemptsThroughLoop(entries) {
    const sorted = [...entries].sort((a, b) => (a.date || 0) - (b.date || 0));
    let rank = state.userRank;
    let momentum = state.momentum;
    for (const a of sorted) {
        ({ rank, momentum } = momentumRankStep(rank, momentum, {
            correct: !!a.correct,
            isRepeat: !!a.isRepeat,
            quality: a.problemQuality,
        }));
    }
    if (Number.isFinite(rank) && Number.isFinite(momentum)) {
        updateUserRank(rank);
        updateMomentum(momentum);
    }
}

// Adopt another account on this device: wipe everything derived from the old
// account's history, store the new credential, and reload — init then runs
// maybeFinishAccountSwitch, which pulls the full server history from ''.
// Settings that aren't account-derived (board size, audio, language, corpus
// version stamps) survive the wipe.
export function switchAccount(newId) {
    if (!isValidAccountCode(newId) || newId === state.userId) return false;
    const wipeKeys = [
        'answerHistory', 'challengeProblem',
        'userRank', 'momentum',
        'modelParams', 'adamState',
        'weaknessParams', 'weaknessAdam', 'weaknessSe', 'weaknessAxisCov',
        'recallPausedUntil',
        // The server-side recall schedule snapshot belongs to the old account;
        // the new account's first upload must go replace-all.
        'recallScheduleSnapshot',
        SYNC_STATE_KEY,
    ];
    for (const k of wipeKeys) localStorage.removeItem(k);
    localStorage.setItem('userId', newId);
    writeSyncState({ userId: newId, cursor: '' });
    localStorage.setItem(SWITCH_FLAG_KEY, '1');
    location.reload();
    return true;
}

// Cheap existence probe before adopting an account code: an account "exists"
// iff it has at least one report (reports are the only server-side state an
// account is rebuilt from, so a code with none is indistinguishable from a
// typo). limit:1 makes the probe read ~1 D1 row. Returns true/false, or null
// when the check itself failed (offline, server error).
export async function accountExists(userId) {
    try {
        const res = await fetch('/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ userId, since: '', limit: 1 }),
        });
        if (!res.ok) return null;
        const body = await res.json().catch(() => null);
        if (!body || !Array.isArray(body.reports)) return null;
        return body.reports.length > 0;
    } catch {
        return null;
    }
}

export function hasPendingAccountSwitch() {
    return !!localStorage.getItem(SWITCH_FLAG_KEY);
}

// Called once during init, before the first problem load, so a just-switched
// device serves problems matched to the pulled rank instead of rank 0.
// Returns true if a switch was completed (caller may want a spinner around it).
export async function maybeFinishAccountSwitch() {
    if (!localStorage.getItem(SWITCH_FLAG_KEY)) return false;
    await pullReports({ force: true, mode: 'adopt' });
    localStorage.removeItem(SWITCH_FLAG_KEY);
    return true;
}

// Piggyback signal: every /report response carries prevCreatedAt — the user's
// latest report BEFORE the one just made. If this device doesn't recognize
// that timestamp (not pulled past it, not one of its own acks), another device
// wrote it → pull now. This is what keeps an ACTIVE session in sync without
// any polling; idle devices are covered by the load/visibility pulls below.
// A debounce absorbs the benign in-flight race (own report's ack not yet
// recorded when the next response arrives — the pull then merges nothing,
// the fuzzy claim adopts the row).
const SIGNAL_DEBOUNCE_MS = 60 * 1000;
let lastSignalPullAt = 0;

export function checkSyncSignal(prevCreatedAt) {
    if (typeof prevCreatedAt !== 'string' || !prevCreatedAt) return;
    const sync = readSyncState();
    // ISO timestamps compare lexicographically = chronologically.
    if (sync && prevCreatedAt <= sync.cursor) return;
    // One of our own acked rows? Scan backwards — the match is almost always
    // among the most recent entries.
    for (const size of SUPPORTED_BOARD_SIZES) {
        const history = getAnswerHistoryForSize(size);
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].serverCreatedAt === prevCreatedAt) return;
        }
    }
    const now = Date.now();
    if (now - lastSignalPullAt < SIGNAL_DEBOUNCE_MS) return;
    lastSignalPullAt = now;
    pullReports({ force: true });
}

// --- One-off historical backfill (TEMPORARY — remove with /report-backfill) --
// Clients hold richer history than the server log: before PER_ATTEMPT_ERA_MS
// the client reported only one row per problem (no repeats, no times), so a
// device switch would lose ~36% of real attempts. This one-off pushes the
// missing attempts up, once per device:
//   1. Full-history reconcile pull ('' cursor): every local entry that DOES
//      have a server row gets claimed (serverCreatedAt stamped) by the merge's
//      fuzzy matcher — verified to align at median 1.5s / p99 4.7s on the
//      worst-case real user.
//   2. Entries still un-acked after a COMPLETE pull are provably absent
//      server-side → uploaded with their client timestamps (historical
//      created_at, so they never enter the server-stats 7-day window).
//   3. serverCreatedAt is stamped per uploaded batch (idempotent under
//      partial failure), the done-flag set only when everything landed.
// Fleet cost is bounded by a per-user ramp: hash(userId) assigns each user a
// slot 0..4, eligible from BACKFILL_RAMP_START + slot days ON and forever
// after (no cycles — once your day has passed you run on the next open).
// Spreads the measured ~49k total rows (834 actives, worst user ~5.7k) to
// ~10k/day during the ramp — far under the 100k/day D1 write quota.
const BACKFILL_DONE_KEY = HISTORY_BACKFILL_DONE_KEY;
const BACKFILL_SPREAD_DAYS = 5;
// Anchor of the ramp. Set to (or just after) the deploy date: if the deploy
// lands later, the slots whose days already passed simply bunch onto the
// first day — still comfortably inside quota.
const BACKFILL_RAMP_START_MS = Date.parse('2026-06-14T00:00:00Z');
const BACKFILL_BATCH = 500;

// Round 2 — historical detail enrichment. Round 1 was insert-only, so the
// pre-era skeleton rows it CLAIMED (the legacy one-row-per-problem originals)
// still lack time_used/review_time on the server. This fills them via
// /report-enrich (UPDATE, fill-only). Its own done-key + ramp so it runs once
// per device independently of round 1; remove with the Worker handler once
// actives have run it. NB: the gap is time/review only — the picked move never
// existed for pre-era attempts (the feature shipped ~the per-attempt era), so
// there is nothing to enrich there. See the round-2 note in src/index.js.
const ENRICH_DONE_KEY = 'historyEnrichDone_v1';
// Anchor of the round-2 ramp — set to (or just after) the deploy date; if the
// deploy slips later, the already-passed slots simply bunch onto day 0 (still
// well inside quota), same as round 1.
const ENRICH_RAMP_START_MS = Date.parse('2026-06-16T00:00:00Z');
const ENRICH_BATCH = 500;

// hash(userId) % BACKFILL_SPREAD_DAYS assigns each user a slot 0..4, eligible
// from rampStart + slot days ON and forever after (no cycles). Shared by both
// one-off rollouts so each spreads its fleet write volume across 5 days.
function rampEligibleToday(rampStartMs) {
    let h = 0;
    for (const ch of state.userId) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    const slot = h % BACKFILL_SPREAD_DAYS;
    return (Date.now() - rampStartMs) / 86400000 >= slot;
}

function backfillEligibleToday() {
    return rampEligibleToday(BACKFILL_RAMP_START_MS);
}

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

export async function maybeBackfillLocalHistory() {
    if (localStorage.getItem(BACKFILL_DONE_KEY)) return;
    if (collectUnackedEntries().length === 0) {
        // Nothing to reconcile (fresh device / already-synced account) —
        // don't burn a full-history pull on it.
        localStorage.setItem(BACKFILL_DONE_KEY, '1');
        return;
    }
    if (!backfillEligibleToday()) return;

    try {
        // Claim everything the server already has. Uploading is only safe
        // against a COMPLETE view of the server's rows.
        const { complete } = await pullReports({ force: true, mode: 'reconcile', fromStart: true });
        if (!complete) return;  // retry on a later eligible day

        const candidates = collectUnackedEntries();
        if (candidates.length === 0) {
            localStorage.setItem(BACKFILL_DONE_KEY, '1');
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

        for (let i = 0; i < rows.length; i += BACKFILL_BATCH) {
            const batch = rows.slice(i, i + BACKFILL_BATCH);
            const res = await fetch('/report-backfill', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: state.userId, reports: batch.map(b => b.report) }),
            });
            if (!res.ok) {
                console.warn(`account-sync: backfill batch failed (${res.status}); will retry on a later eligible day`);
                return;
            }
            // Stamp + persist per batch so a later failure never re-uploads
            // what already landed.
            const sizesTouched = new Set();
            for (const b of batch) {
                b.entry.serverCreatedAt = b.report.createdAt;
                sizesTouched.add(b.size);
            }
            for (const size of sizesTouched) {
                setAnswerHistoryForSize(size, getAnswerHistoryForSize(size));
            }
        }

        localStorage.setItem(BACKFILL_DONE_KEY, '1');
        console.log(`account-sync: backfilled ${rows.length} historical attempts to the server`);
    } catch (e) {
        console.warn('account-sync: history backfill failed (will retry):', e);
    }
}

// --- Round 2: enrich existing server rows with their missing detail ----------
// Round 1 uploaded the attempts the server was MISSING; it left the pre-era
// rows it had to CLAIM (the legacy one-row-per-problem originals) without
// time_used/review_time, because /report-backfill only INSERTs. This fills
// them. It re-runs the SAME soft match as round 1 (problemId + correctness +
// nearest date within ±10 min, is_repeat relaxed pre-era) against a fresh full
// pull, and keys each UPDATE on the matched server row's AUTHORITATIVE pulled
// created_at — never a stored serverCreatedAt. The server side is fill-only
// (COALESCE), so re-runs, races, and over-sends are all harmless.

// Pre-era entries this device solved FIRST-HAND that hold detail the server's
// skeleton rows may lack. Synced entries (pulled from the server on a device
// adoption, flagged by mergeRowsForSize) are excluded: their time/review came
// FROM the server, so re-reporting it is circular, and across a multi-attempt
// problem the fuzzy match could paste one attempt's time onto another row. Only
// the device that actually recorded a solve contributes it — multi-device safe
// (the server-side COALESCE fill-only guard is the second line of defense, and
// this mirrors round 1, which also uploads only un-acked first-hand entries).
function enrichableLocalsForSize(size) {
    const out = [];
    for (const a of getAnswerHistoryForSize(size)) {
        if (!a || a.synced || typeof a.problemId !== 'string') continue;
        if (typeof a.date !== 'number' || a.date >= PER_ATTEMPT_ERA_MS) continue;
        const hasTime = typeof a.timeTaken === 'number' && a.timeTaken > 0;
        const hasReview = typeof a.accumulatedReview === 'number' && a.accumulatedReview > 0;
        if (hasTime || hasReview) out.push(a);
    }
    return out;
}

function hasEnrichableLocalEntries() {
    for (const size of SUPPORTED_BOARD_SIZES) {
        if (enrichableLocalsForSize(size).length > 0) return true;
    }
    return false;
}

// Soft-match this size's detail-bearing pre-era locals to the pulled server
// rows that are MISSING that detail; append {problem, createdAt, timeUsed,
// reviewTime} items keyed on the pulled created_at. Greedy nearest-date 1-1,
// the same matcher as the round-1 claim.
function collectEnrichUpdates(size, rows, out) {
    const locals = enrichableLocalsForSize(size);
    if (locals.length === 0) return;
    const used = new Set();
    for (const row of rows) {
        if (typeof row.createdAt !== 'string' || typeof row.problem !== 'string') continue;
        const rowTs = Date.parse(row.createdAt);
        if (!Number.isFinite(rowTs) || rowTs >= PER_ATTEMPT_ERA_MS) continue;
        const needTime = row.timeUsed === null || row.timeUsed === undefined;
        const needReview = row.reviewTime === null || row.reviewTime === undefined;
        if (!needTime && !needReview) continue;  // row already complete

        let best = null;
        let bestGap = FUZZY_WINDOW_MS;
        for (const a of locals) {
            if (used.has(a) || a.problemId !== row.problem) continue;
            if (!!a.correct !== !!row.success) continue;
            const gap = Math.abs((a.date || 0) - rowTs);
            if (gap < bestGap) { best = a; bestGap = gap; }
        }
        if (!best) continue;

        const timeUsed = (needTime && typeof best.timeTaken === 'number' && best.timeTaken > 0)
            ? Math.round(best.timeTaken) : null;
        const reviewTime = (needReview && typeof best.accumulatedReview === 'number' && best.accumulatedReview > 0)
            ? Math.round(best.accumulatedReview) : null;
        if (timeUsed === null && reviewTime === null) continue;  // match adds nothing the row lacks
        used.add(best);
        out.push({ problem: row.problem, createdAt: row.createdAt, timeUsed, reviewTime });
    }
}

export async function maybeEnrichServerHistory() {
    if (localStorage.getItem(ENRICH_DONE_KEY)) return;
    if (!hasEnrichableLocalEntries()) {
        // No pre-era local detail to contribute (fresh device / post-era-only
        // account) — don't spend a full-history pull on it.
        localStorage.setItem(ENRICH_DONE_KEY, '1');
        return;
    }
    if (!rampEligibleToday(ENRICH_RAMP_START_MS)) return;

    try {
        // Capture every pulled server row (incl. the already-claimed originals
        // the merge skips) so the soft match sees the full set. The pull also
        // re-claims, exactly as round 1 did, so the match is based on fresh
        // server truth rather than any stored field.
        const rowsBySize = new Map();
        const rowSink = (size, rows) => {
            if (!rowsBySize.has(size)) rowsBySize.set(size, []);
            const bucket = rowsBySize.get(size);
            for (const r of rows) bucket.push(r);
        };
        const { complete } = await pullReports({ force: true, mode: 'reconcile', fromStart: true, rowSink });
        if (!complete) return;  // incomplete view — retry on a later eligible day

        const updates = [];
        for (const size of SUPPORTED_BOARD_SIZES) {
            collectEnrichUpdates(size, rowsBySize.get(size) || [], updates);
        }
        if (updates.length === 0) {
            localStorage.setItem(ENRICH_DONE_KEY, '1');
            return;
        }

        for (let i = 0; i < updates.length; i += ENRICH_BATCH) {
            const batch = updates.slice(i, i + ENRICH_BATCH);
            const res = await fetch('/report-enrich', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: state.userId, updates: batch }),
            });
            if (!res.ok) {
                console.warn(`account-sync: enrich batch failed (${res.status}); will retry on a later eligible day`);
                return;
            }
        }

        localStorage.setItem(ENRICH_DONE_KEY, '1');
        console.log(`account-sync: enriched ${updates.length} historical rows (time/review) on the server`);
    } catch (e) {
        console.warn('account-sync: history enrich failed (will retry):', e);
    }
}

// Regular syncs: app load and tab becoming visible — the "user returned to
// this device" signals. No interval polling: while the user actively plays,
// every report response carries the piggyback signal above, which is both
// fresher and free. pullReports throttles internally. The one-off history
// backfill rides the load pull's tail so the two never overlap.
export function startBackgroundSync() {
    pullReports()
        .then(() => maybeBackfillLocalHistory())
        .then(() => maybeEnrichServerHistory())
        .catch(() => {});
    window.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') pullReports();
    });
}
