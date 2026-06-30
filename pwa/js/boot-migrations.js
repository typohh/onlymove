// js/boot-migrations.js
//
// The localStorage-SHAPE migrations that must run at state.js IMPORT time, before
// state.js reads any persisted property into the live state object. They were
// inline IIFEs at the top of state.js; collected here so the migration code lives
// in one place, while state.js just calls runBootMigrations() at the same point.
//
// Kept SEPARATE from migrations.js on purpose: migrations.js imports state.js (and,
// transitively, backfill.js → the whole heavy graph), so importing it from state.js
// would create a state.js↔migrations.js cycle and force that graph to evaluate
// mid-initialization. This module imports ONLY the leaf persist.js, so it stays safe
// to invoke as the first thing state.js does. The companion runMigrations() in
// migrations.js holds the LATER, state-live answerHistory migrations.

import { readJSONObject } from './persist.js';

// The default board size AT MIGRATION TIME (the app was 9x9-only before buckets
// existed). Frozen here so this module needs nothing from state.js — these are
// historical migrations; their semantics don't track the current default.
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

// --- Recall model → FSRS retention curve reset ---
// The recall model was reparameterized from an additive logistic (features
// INTERCEPT/TIME/REVIEW/DIFFICULTY/IS_NOVEL/RECALL_FORMULA/SIZE_IS_13) into a
// dual-head model with the SAME param-vector length (7) but different feature
// semantics (… SIZE_IS_13/SUCCESS_SINCE_FAILURE/LN_C). Because the length is
// unchanged the runtime length-guards can't detect the switch, so stored params
// must be dropped explicitly and retrained from scratch on next load.
//
// v5: stability is now S = exp(LN_C)·odds(p/0.7)^κ·1.54^r — the cold-solve logit
// z=logit(p) (difficulty+review+time) mapped LINEARLY with a population-fit slope
// κ, growth ×1.54/success, and only the per-user base LN_C fit (shrunk to a
// population prior). Replaces the bounded-σ form whose hard-side ln σ(z)≈z tail
// collapsed hard problems to the 18h floor. Re-wipe so the new params train.
function resetRecallModelForRetention_v5() {
    if (localStorage.getItem('recallRetentionModelMigrated_v5')) return;
    localStorage.removeItem('modelParams');
    localStorage.removeItem('adamState');
    localStorage.setItem('recallRetentionModelMigrated_v5', 'true');
}

// --- Challenge-model migration ---
// The "which problem is shown" state used to be a 2-slot `problemHistory` array
// ([current, previous]) plus a `nextTsumego` stash for an interrupted challenge.
// That conflated three distinct things and let an already-answered problem be
// re-served (via the nextTsumego resume and the reload shortcut). The model is
// now three explicit variables: `challengeProblem` (the always-assigned
// challenge, persisted per-size), `currentProblem` (what's on screen),
// `previousProblem` (last completed). Drop the old keys — challengeProblem is
// reselected fresh on next load, so existing users lose at most one in-progress
// problem; the review slot repopulates after their next solve.
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
