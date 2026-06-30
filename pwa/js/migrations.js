// js/migrations.js
//
// One-shot answerHistory migrations, run once per browser and guarded by a
// localStorage flag each. These were previously inlined into main.js's
// DOMContentLoaded handler (~130 lines in the boot path); collected here so the
// entry point just calls runMigrations().
//
// NOTE: the import-time localStorage-SHAPE migrations (boardSizeMigrated_v1,
// sharedAcrossSizesMigrated_v2, …) live in boot-migrations.js, not here — they
// must run before state.js reads any property, and this module (via backfill.js)
// is too heavy to import from state.js without a cycle. These are the *later*
// answerHistory-shape migrations, which run after state is live and operate
// through state.answerHistory (the active size's view, already wrapped under
// the size bucket by the boot migrations) and updateAnswerHistory().
//
// NOTE: one more migration lives OUTSIDE both migration files — migrateIdCursor()
// (idCursorMigrated_v1) in account-sync.js. It's deliberately NOT here: it isn't a
// localStorage transform but a one-time sync-CURSOR transition that reads/writes the
// sync state and does a network frontier probe via account-sync's own private
// helpers (readSyncState/writeSyncState/probeFrontier). Pulling it out would mean
// exporting those internals + an account-sync↔migrations import cycle, to relocate
// logic that belongs to the sync subsystem. It runs from startBackgroundSync(),
// before the first pull. So the full migration set = boot-migrations.js (import-time
// shape) + this file (answerHistory/model/tail/window-rank) + that one in
// account-sync.js. Anyone auditing migrations should check all three.

import { state, updateAnswerHistory, updateUserRank, updateServeRank, getMergedAnswerHistory, syncTailExpectedToActual, SUPPORTED_BOARD_SIZES } from './state.js';
import { getTail, replaceTail } from './history-store.js';
import { fillSyncedTailFields } from './backfill.js';
import { estimateWindowRank } from './rank-decomposition.js';
import { TOTAL_RANKS } from './constants.js';

// Backfill accumulatedReview: the total reviewTime spent on a problem across all
// PRIOR attempts (strictly before this entry's date).
function migrateAccumulatedReview() {
    if (localStorage.getItem('answerHistoryMigrated_v4_accumulatedReview')) {
        return;
    }
    console.log('Running answer history migration for accumulatedReview...');

    const answerHistory = state.answerHistory;
    const answersByProblem = {};

    for (const answer of answerHistory) {
        if (!answersByProblem[answer.problemId]) {
            answersByProblem[answer.problemId] = [];
        }
        answersByProblem[answer.problemId].push(answer);
    }

    for (const problemId in answersByProblem) {
        answersByProblem[problemId].sort((a, b) => a.date - b.date);
    }

    const newHistory = [];
    for (const answer of answerHistory) {
        const problemAnswers = answersByProblem[answer.problemId];
        const prevAnswers = problemAnswers.filter(a => a.date < answer.date);
        const accumulatedReview = prevAnswers.reduce((sum, a) => sum + (a.reviewTime || 0), 0);

        const newAnswer = { ...answer };
        if (newAnswer.accumulatedReview === undefined) {
            newAnswer.accumulatedReview = accumulatedReview;
        }
        newHistory.push(newAnswer);
    }

    updateAnswerHistory(newHistory);
    localStorage.setItem('answerHistoryMigrated_v4_accumulatedReview', 'true');
    console.log('Answer history migration for accumulatedReview complete.');
}

// Failed attempts predating the successSinceFailure feature have it undefined;
// a failure resets the counter to 0, so backfill 0.
function migrateFailedNovelSuccessSinceFailure() {
    if (localStorage.getItem('answerHistoryMigrated_v5_failedNovelSsf')) {
        return;
    }
    console.log('Running answer history migration for failed novel successSinceFailure...');

    const answerHistory = [...state.answerHistory];
    let changed = 0;
    for (const answer of answerHistory) {
        if (!answer.correct && answer.successSinceFailure === undefined) {
            answer.successSinceFailure = 0;
            changed++;
        }
    }

    if (changed > 0) {
        updateAnswerHistory(answerHistory);
        console.log(`Patched ${changed} answers with missing successSinceFailure.`);
    }
    localStorage.setItem('answerHistoryMigrated_v5_failedNovelSsf', 'true');
    console.log('Answer history migration for failed novel successSinceFailure complete.');
}

// Replace the old string problemType ('repeat'/'novel') with the boolean
// isRepeat flag.
function migrateProblemTypeToIsRepeat() {
    if (localStorage.getItem('answerHistoryMigrated_v3_isRepeat')) {
        return;
    }
    console.log('Running answer history migration for isRepeat...');

    const answerHistory = state.answerHistory;
    const newHistory = answerHistory.map(answer => {
        if (answer.problemType !== undefined) {
            answer.isRepeat = answer.problemType === 'repeat';
            delete answer.problemType;
        }
        return answer;
    });

    updateAnswerHistory(newHistory);
    localStorage.setItem('answerHistoryMigrated_v3_isRepeat', 'true');
    console.log('Answer history migration for isRepeat complete.');
}

// Earlier builds wrote an `ignored` flag that confounded two distinct cases;
// clear it so the field can be re-derived cleanly.
function migrateClearIgnoredFlag() {
    if (localStorage.getItem('answerHistoryMigrated_v6_clearIgnored')) {
        return;
    }
    console.log('Running answer history migration to clear confounded ignored flags...');

    const answerHistory = [...state.answerHistory];
    let cleared = 0;
    for (const answer of answerHistory) {
        if (answer.ignored !== undefined) {
            delete answer.ignored;
            cleared++;
        }
    }

    if (cleared > 0) {
        updateAnswerHistory(answerHistory);
        console.log(`Cleared ignored flag from ${cleared} answers.`);
    }
    localStorage.setItem('answerHistoryMigrated_v6_clearIgnored', 'true');
    console.log('Answer history migration for clearing ignored flags complete.');
}

// The cold-solve model switched to a per-problem 3PL (model.js): PROBLEM_DIFFICULTY
// went from a raw negative in-band rank slope to a positive multiplier on the
// problem's a_p·(rank gap), and a κ(floor) anchor now carries the difficulty shape.
// A warm start from the old persisted params would be wrong (different units/sign),
// so drop modelParams/adamState once and let the model retrain clean.
function migrateModelArch3pl() {
    if (localStorage.getItem('modelArchMigrated_v1_3plFloor')) {
        return;
    }
    state.modelParams = null;
    state.adamState = null;
    localStorage.setItem('modelArchMigrated_v1_3plFloor', 'true');
    console.log('Model arch migration: dropped pre-3PL model params for clean retrain.');
}

// The backfill's rating re-derive was working-set ONLY until the cold-tail re-derive
// landed: the IndexedDB tail kept each attempt's rating frozen at eviction time, so
// after a re-rate two devices' tails (evicted at different rating eras) diverged and
// the Rank-tab chart disagreed device-to-device. The fix re-derives the tail too, gated
// on the steady-state ratingsChanged check (backfill.js). But devices that already
// stamped the current ratingsVersion via the OLD walk won't re-fire that gate, so clear
// the stamp ONCE: the next backfill then sees ratingsChanged and re-derives everything
// (the working set is a no-op, already current; the tail heals). Reuses the steady-state
// primitive — no special re-derive path in the backfill itself.
function migrateClearRatingsStampForTailRederive() {
    if (localStorage.getItem('tailRederiveMigrated_v1')) return;
    localStorage.removeItem('ratingsVersion');
    localStorage.setItem('tailRederiveMigrated_v1', '1');
    console.log('Migration: cleared ratingsVersion stamp so the next backfill re-derives the cold tail.');
}

// Original successSinceFailure backfill: walk each problem's attempts in date
// order, counting successes since the most recent failure.
function migrateAnswerHistory() {
    if (localStorage.getItem('answerHistoryMigrated_v2')) {
        return;
    }

    console.log('Running answer history migration for successSinceFailure...');

    const answerHistory = state.answerHistory;
    const answersByProblem = {};

    for (const answer of answerHistory) {
        if (!answersByProblem[answer.problemId]) {
            answersByProblem[answer.problemId] = [];
        }
        answersByProblem[answer.problemId].push(answer);
    }

    for (const problemId in answersByProblem) {
        const problemAnswers = answersByProblem[problemId];
        let successCounter = 0;
        let hasFailed = false;

        for (const answer of problemAnswers) {
            if (!answer.correct) {
                hasFailed = true;
                successCounter = 0;
                answer.successSinceFailure = 0;
            } else if (hasFailed) {
                successCounter++;
                answer.successSinceFailure = successCounter;
            }
        }
    }

    updateAnswerHistory(answerHistory);
    localStorage.setItem('answerHistoryMigrated_v2', 'true');
    console.log('Answer history migration complete.');
}

/**
 * Run all answerHistory migrations in their original order. Each self-guards on
 * a localStorage flag, so this is a cheap no-op once a browser has run them.
 * Must be called after state.js is loaded (state.answerHistory is live).
 */
export function runMigrations() {
    migrateAccumulatedReview();
    migrateFailedNovelSuccessSinceFailure();
    migrateProblemTypeToIsRepeat();
    migrateClearIgnoredFlag();
    migrateAnswerHistory();
    migrateModelArch3pl();
    migrateClearRatingsStampForTailRederive();
}

// One-time transition from the momentum loop to the window-MLE rank. Recompute the
// DISPLAYED rank and the SERVE rank from the window estimator over the stored
// history, silently (no overlay), so an existing user switches estimators between
// problems and the first post-deploy answer's reported delta is that tsumego's
// effect alone — not the one-off momentum→window jump. Also seeds serveRank (a new
// field) so the very first served problem already reflects the UCB bonus. A fresh
// device with no usable first-time history is a no-op (serveRank then falls back to
// userRank until the first attempt).
//
// NOT folded into runMigrations(): that runs synchronously before initHistoryStorage,
// whereas this reads the normalized merged history, so main.js calls it separately at
// the matching boot point.
export function maybeInitWindowRankOnce() {
    if (localStorage.getItem('windowRankInit_v1')) return;
    try {
        const est = estimateWindowRank(getMergedAnswerHistory(), state.userRank * TOTAL_RANKS);
        if (est) {
            updateUserRank(est.rankIndex / TOTAL_RANKS);
            updateServeRank(est.serveRankIndex / TOTAL_RANKS);
        }
    } catch (e) {
        console.warn('one-time window-rank init failed:', e);
    }
    localStorage.setItem('windowRankInit_v1', 'true');
}

// De-dup the cold tail by EXACT (problemId,date), keeping the richest copy (so the tail
// stays oldest→newest). Returns rows removed. Exact key ONLY, on purpose: a separate
// attempt can't share a millisecond, so same-timestamp rows are storage dups, whereas
// same-problem rows seconds/minutes apart are legitimate fast repeats that must survive.
async function dedupTailByProblemDate() {
    const tail = await getTail();
    if (!Array.isArray(tail) || tail.length === 0) return 0;
    const richness = (e) =>
        (typeof e.problemRank === 'number' ? 4 : 0) +
        (typeof e.problemRating === 'number' ? 2 : 0) +
        (e.embedding !== undefined ? 1 : 0);
    const best = new Map();
    const order = [];
    for (const e of tail) {
        if (!e) continue;
        const k = `${e.problemId}|${e.date}`;
        if (!best.has(k)) { best.set(k, e); order.push(k); }
        else if (richness(e) > richness(best.get(k))) best.set(k, e);
    }
    const removed = tail.length - order.length;
    if (removed > 0) await replaceTail(order.map((k) => best.get(k)));
    return removed;
}

const TAIL_MIGRATION_KEY = 'tailMigration_v1';

/**
 * ONE-TIME cold-tail cleanup. Async and run separately from the synchronous
 * runMigrations() above (it needs the IndexedDB tail, which only exists after
 * initHistoryStorage). Repairs damage from the pre-tail-aware sync:
 *   - a second full pull used to re-insert already-evicted rows as
 *     (problemId,date) DUPLICATES (the merge deduped only against the working
 *     set), and
 *   - full-pull eviction left synced skeletons in the tail with NO rating, so
 *     they dropped off the full-history Rank chart.
 * Steady state now prevents both — mergeRowsForSize dedups against the tail, and
 * fillSyncedHistoryFields fills the tail on a full pull — so this only fixes
 * EXISTING tails, then never runs again. Migration code, deliberately specific.
 *
 * Must run BEFORE the backfill's curve-recompute touches the tail (main.js
 * sequences them).
 */
export async function runTailMigration() {
    if (localStorage.getItem(TAIL_MIGRATION_KEY)) return;
    // A superseded in-place repair (tailRepaired*) shipped briefly; it removed
    // duplicate rows WITHOUT updating the silent-eviction high-water mark, so any
    // device that ran it has a stale mark even if its tail is now dup-free.
    const ranSupersededRepair = !!(localStorage.getItem('tailRepairedV1') ||
        localStorage.getItem('tailRepairedV2'));
    localStorage.removeItem('tailRepairedV1');
    localStorage.removeItem('tailRepairedV2');

    let reconstructOk = true;
    try {
        const tail = await getTail();
        if (Array.isArray(tail) && tail.length > 0) {
            const dedupedCount = await dedupTailByProblemDate();
            if (dedupedCount > 0) {
                console.log(`tail-migration: removed ${dedupedCount} duplicate rows`);
            }

            // Re-establish the silent-eviction high-water mark when this migration
            // (or the superseded repair) shrank the tail — otherwise the next stats
            // open reads the smaller tail as the BROWSER dropping rows and pops the
            // "restore from server?" prompt. Skipped on untouched tails so a REAL
            // browser eviction is still detected.
            if (dedupedCount > 0 || ranSupersededRepair) {
                try { await syncTailExpectedToActual(); } catch {}
            }

            // Reconstruct ratings for the surviving synced skeletons, reusing the
            // clean steady-state tail fill. ok=false ⇒ a shard fetch failed
            // (offline) ⇒ leave the migration unflagged so it retries next load.
            const r = await fillSyncedTailFields(SUPPORTED_BOARD_SIZES);
            reconstructOk = r.ok;
        }
    } catch (e) {
        console.warn('tail-migration failed (will retry next load):', e);
        return;
    }
    if (reconstructOk) localStorage.setItem(TAIL_MIGRATION_KEY, '1');
}

const TAIL_DEDUP_KEY = 'tailDedupExact_v1';

/**
 * ONE-TIME: remove exact (problemId,date) duplicate rows the cold tail accumulated from
 * the April–May 2026 outbox/sync re-inserts — created AFTER runTailMigration's one-shot
 * dedup, so they slipped past it (~1.6k rows on a heavy device). getMergedFullAnswerHistory
 * now also de-dups the tail at READ time, so the chart no longer double-plots regardless;
 * this is the matching storage cleanup. Runs once, after runTailMigration and before the
 * backfill (main.js sequences them). Left unflagged on a tail-fetch failure so it retries.
 */
export async function dedupTailExact() {
    if (localStorage.getItem(TAIL_DEDUP_KEY)) return;
    try {
        const removed = await dedupTailByProblemDate();
        if (removed > 0) {
            // Shrinking the tail moves the silent-eviction high-water mark, or the next
            // stats open misreads it as the browser dropping rows.
            try { await syncTailExpectedToActual(); } catch {}
            console.log(`tail-dedup: removed ${removed} exact duplicate rows`);
        }
    } catch (e) {
        console.warn('tail-dedup failed (will retry next load):', e);
        return;
    }
    localStorage.setItem(TAIL_DEDUP_KEY, '1');
}
