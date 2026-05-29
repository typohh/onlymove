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

import { state, updateAnswerHistory } from './state.js';

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
}
