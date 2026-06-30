// js/boot-migrations.js
//
// The localStorage-SHAPE migrations that must run at state.js IMPORT time, before it
// reads any persisted property into the live state object. state.js calls
// runBootMigrations() as its first action.
//
// Separate from migrations.js because that module imports state.js (and, transitively,
// backfill.js's graph): importing it from state.js would cycle and force that graph to
// evaluate mid-initialization. This module imports ONLY the leaf persist.js, so it's safe
// to invoke first. migrations.js holds the later, state-live answerHistory migrations.

import { readJSONObject } from './persist.js';

// Frozen migration-local constant, not imported from state.js so this module stays
// dependency-light; these historical migrations don't track the current default.
const DEFAULT_BOARD_SIZE = 9;

// --- localStorage migration ---
// Old layout stored a single value per key (e.g. userRank: "0.5",
// problemHistory: "[...]") because the app only supported 9x9.
// New layout stores an object keyed by board size: {"9": <oldValue>}.
// This must run before any state property below is read.
function migrateBoardSizeBuckets() {
    if (localStorage.getItem('boardSizeMigrated_v1')) return;

    const perSizeKeys = [
        'userRank',
        'problemHistory',
        'answerHistory',
        'nextTsumego',
        'modelParams',
        'adamState',
        'weaknessParams',
        'weaknessAdam',
        'weaknessSe',
        'weaknessAxisCov',
    ];

    for (const key of perSizeKeys) {
        const raw = localStorage.getItem(key);
        if (raw === null) continue;

        let parsed;
        try {
            parsed = JSON.parse(raw);
        } catch {
            const num = parseFloat(raw);
            parsed = Number.isNaN(num) ? raw : num;
        }

        // Already in the new shape — leave alone.
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed) &&
            ('9' in parsed || '13' in parsed)) {
            continue;
        }

        localStorage.setItem(key, JSON.stringify({ [DEFAULT_BOARD_SIZE]: parsed }));
    }

    // Existing users (those who had any state at all) keep playing 9x9.
    // Truly fresh users have no boardSize and will see the cold-start dialog.
    const hadAnyState = perSizeKeys.some(k => localStorage.getItem(k) !== null);
    if (hadAnyState && !localStorage.getItem('boardSize')) {
        localStorage.setItem('boardSize', String(DEFAULT_BOARD_SIZE));
    }

    localStorage.setItem('boardSizeMigrated_v1', 'true');
}

// --- Cross-size merge: rank and the learned models are now shared ---
// The two board sizes share one rank (displayed everywhere, used to pick
// problems) and one each of the recall + weakness models (trained on the union
// of both histories — every per-problem feature they read is board-independent).
// This migration collapses the per-size buckets v1 created back into single
// values: rank = max of the two (a strong 9x9 player should immediately see
// hard 13x13 problems), and the model buckets are dropped so the shared models
// retrain from scratch on next load. answerHistory and challengeProblem stay per-size.
function mergeSharedAcrossSizes_v2() {
    if (localStorage.getItem('sharedAcrossSizesMigrated_v2')) return;

    const rankBucket = readJSONObject('userRank');
    let maxRank = 0;
    for (const v of Object.values(rankBucket)) {
        if (typeof v === 'number' && v > maxRank) maxRank = v;
    }
    localStorage.setItem('userRank', JSON.stringify(maxRank));

    // Models retrain from scratch — drop the per-size buckets entirely.
    for (const key of ['modelParams', 'adamState', 'weaknessParams',
                       'weaknessAdam', 'weaknessSe', 'weaknessAxisCov']) {
        localStorage.removeItem(key);
    }

    localStorage.setItem('sharedAcrossSizesMigrated_v2', 'true');
}

// --- Recall model reset ---
// Drop the persisted recall-model params so the model retrains. The param-vector
// length is unchanged across the reparameterization, so the runtime length-guards
// can't detect the switch — the stored params must be cleared explicitly.
function resetRecallModelForRetention_v5() {
    if (localStorage.getItem('recallRetentionModelMigrated_v5')) return;
    localStorage.removeItem('modelParams');
    localStorage.removeItem('adamState');
    localStorage.setItem('recallRetentionModelMigrated_v5', 'true');
}

// --- Challenge-model migration ---
// Remove the retired `problemHistory` / `nextTsumego` keys (the show-problem state is
// now `challengeProblem` / `currentProblem` / `previousProblem`). challengeProblem is
// reselected fresh on next load, so a user loses at most one in-progress problem and the
// review slot repopulates after their next solve.
function migrateToChallengeModel_v1() {
    if (localStorage.getItem('challengeModelMigrated_v1')) return;
    localStorage.removeItem('problemHistory');
    localStorage.removeItem('nextTsumego');
    localStorage.setItem('challengeModelMigrated_v1', 'true');
}

/**
 * Run the import-time localStorage-shape migrations in order. Each self-guards on
 * a localStorage flag, so this is a cheap no-op once a browser has run them.
 *
 * MUST be called at the top of state.js, before it reads any persisted property —
 * migrateBoardSizeBuckets() establishes the per-size bucket shape the rest of
 * state.js (and mergeSharedAcrossSizes_v2) depend on. Order is load-bearing:
 * buckets first, then the cross-size merge that reads them.
 */
export function runBootMigrations() {
    migrateBoardSizeBuckets();
    mergeSharedAcrossSizes_v2();
    resetRecallModelForRetention_v5();
    migrateToChallengeModel_v1();
}
