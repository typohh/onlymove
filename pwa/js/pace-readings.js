// js/pace-readings.js
//
// The two PACE readings, shaped like a motif's strength reading so the
// Weaknesses pane and the front-page list can show them the same way:
//
//   solving — does taking more time before answering help this user more than
//             it helps players generally?
//   review  — does reviewing a failed problem for longer help this user
//             remember it more than it helps players generally?
//
// Both are read off the recall model's time/review coefficients
// (recall-model.js), recentered by the population confound they are shrunk
// toward (benefitForVar), and converted from logit per minute to RANKS per
// minute through the user's logit-per-rank. Oriented like strength: a negative
// reading is a weakness (more time would have helped — the user is rushing),
// so `isFlagged` and the tone rules apply unchanged.

import { state } from './state.js';
import { MOTIF_Z } from './constants.js';
import { MODEL_INDICES, benefitForVar, getLogitPerRank } from './recall-model.js';

// Repeat attempts the model's window needs before the review reading is reported.
// Without repeats the review coefficient sits exactly on its prior, and its
// interval says nothing about the user.
const MIN_REPEATS = 10;

export const PACE_KINDS = ['solving', 'review'];
export const PACE_NAME_KEYS = { solving: 'paceSolvingName', review: 'paceReviewName' };

function readingFor(varIndex, statsResults) {
    const results = statsResults && statsResults.results;
    if (!results) return null;
    const b = benefitForVar(results.find(r => r.varIndex === varIndex), varIndex);
    if (!b) return null;
    const perRank = getLogitPerRank(statsResults);
    const ranks = -b.coeff / perRank;
    const se = b.se / perRank;
    return { ranks, se, z: ranks / se, lo: ranks - MOTIF_Z * se, hi: ranks + MOTIF_Z * se };
}

// { solving, review }, each a reading or null when there is nothing to read.
export function computePaceReadings(statsResults = state.statsResults) {
    const repeats = statsResults ? statsResults.nRepeats ?? 0 : 0;
    return {
        solving: readingFor(MODEL_INDICES.TIME_SOLVING, statsResults),
        review: repeats >= MIN_REPEATS ? readingFor(MODEL_INDICES.REVIEW_TIME, statsResults) : null,
    };
}
