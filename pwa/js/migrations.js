// js/migrations.js
//
// One-shot answerHistory migrations, each guarded by its own localStorage flag so it
// runs at most once per browser. main.js calls runMigrations() at boot after state is
// live, so these operate through state.answerHistory and updateAnswerHistory().
//
// The full migration set spans three files; an audit should check all three:
//   - boot-migrations.js — import-time localStorage-SHAPE migrations, which must run
//     before state.js reads any property. They stay out of this file because it pulls
//     in backfill.js's graph, which would cycle through state.js.
//   - this file — answerHistory / model / tail / window-rank migrations.
//   - account-sync.js — migrateIdCursor(), a sync-CURSOR transition that uses that
//     module's private sync-state + frontier-probe helpers, so it lives with the sync
//     subsystem and runs from startBackgroundSync() before the first pull.

import { state, updateAnswerHistory, updateUserRank, updateServeRank, setModelParams, setAdamState, getMergedAnswerHistory, syncTailExpectedToActual, SUPPORTED_BOARD_SIZES } from './state.js';
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

// Drop persisted recall-model params once: the cold-solve model is a per-problem 3PL
// whose params are unit/sign-incompatible with any pre-3PL warm start, so the model
// must retrain clean.
function migrateModelArch3pl() {
    if (localStorage.getItem('modelArchMigrated_v1_3plFloor')) {
        return;
    }
    setModelParams(null);
    setAdamState(null);
    localStorage.setItem('modelArchMigrated_v1_3plFloor', 'true');
    console.log('Model arch migration: dropped pre-3PL model params for clean retrain.');
}

// Clear the ratingsVersion stamp once so the next backfill's ratingsChanged gate
// re-derives the cold tail, whose per-attempt ratings are frozen at eviction time and
// otherwise stay stale (diverging device-to-device on the Rank-tab chart). Reuses the
// steady-state re-derive path; no special-case code in the backfill.
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

// One-time: recompute the DISPLAYED and SERVE rank from the window-MLE estimator over the
// stored history, silently (no overlay), so the switch lands between problems and the next
// answer's reported delta reflects that tsumego alone. Seeds serveRank for the first served
// problem's UCB bonus. A device with no usable first-time history is a no-op.
//
// NOT folded into runMigrations(): that runs synchronously before initHistoryStorage,
// whereas this reads the normalized merged history, so main.js calls it separately.
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
 * ONE-TIME cold-tail cleanup, async (needs the IndexedDB tail, which exists only after
 * initHistoryStorage) so it runs separately from the synchronous runMigrations(). On an
 * existing tail it removes (problemId,date) DUPLICATE rows and reconstructs ratings for
 * synced skeletons that have none (which otherwise drop off the full-history Rank chart).
 *
 * Must run BEFORE the backfill's curve-recompute touches the tail (main.js sequences them).
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
 * ONE-TIME: remove exact (problemId,date) duplicate rows from the cold tail (storage
 * cleanup; getMergedFullAnswerHistory also de-dups the tail at READ time). Runs once,
 * after runTailMigration and before the backfill (main.js sequences them). Left unflagged
 * on a tail-fetch failure so it retries.
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
