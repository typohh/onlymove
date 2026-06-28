// js/migrations.js
//
// One-shot answerHistory migrations, run once per browser and guarded by a
// localStorage flag each. These were previously inlined into main.js's
// DOMContentLoaded handler (~130 lines in the boot path); collected here so the
// entry point just calls runMigrations().
//
// NOTE: the board-size bucket migrations (boardSizeMigrated_v1,
// sharedAcrossSizesMigrated_v2) deliberately stay in state.js — they must run
// at import time, before any state property is read. These are the *later*
// answerHistory-shape migrations, which run after state is live and operate
// through state.answerHistory (the active size's view, already wrapped under
// the size bucket by the state.js migrations) and updateAnswerHistory().

import { state, updateAnswerHistory, syncTailExpectedToActual, SUPPORTED_BOARD_SIZES } from './state.js';
import { getTail, replaceTail } from './history-store.js';
import { fillSyncedTailFields } from './backfill.js';

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
            // De-dup by (problemId,date), keeping the richest copy in first
            // position (so the tail stays oldest→newest).
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
            const dedupedCount = tail.length - order.length;
            if (dedupedCount > 0) {
                await replaceTail(order.map((k) => best.get(k)));
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
