// js/population-constants.js
//
// Loads the fitted population model constants (pwa/tsumegos/population_constants.json,
// produced by pipeline.fit_population_constants and shipped by package_for_pwa) and
// applies them: the retention constants in recall-model.js (KAPPA/Z0/GROWTH/LN_C_PRIOR) and
// the rushing/review time-benefit biases in game-logic.js. Board-independent — one
// blob for both sizes.
//
// MANDATORY, NO fallback: these constants back the recall scheduler and the nudge
// gates, and the modules carry no baked defaults. They are inlined into the bundle
// at build time (the import below) rather than fetched, so they are present the
// instant the app boots — no runtime fetch, no precache, no eviction/offline path
// that could fail. main.js still null-guards against a malformed packaged file.

import { applyRetentionConstants, setTimeBias } from './recall-model.js';
import { weightedMedianOverWindow } from '../shared/rank-window.js';
import { flattenAttempts } from './rank-decomposition.js';
// Baked at build time from the pipeline's output (pwa/tsumegos/population_constants.json).
import POPULATION_CONSTANTS from '../tsumegos/population_constants.json';

let _loaded = null;

export function getPopulationConstants() {
    return _loaded;
}

// Apply the inlined constants (idempotent: memoized after the first call). The
// applyRetentionConstants / setTimeBias side effects must still run, so this is a
// function rather than module-top-level work — main.js calls it during boot.
export function loadPopulationConstants() {
    if (_loaded) return _loaded;
    const c = POPULATION_CONSTANTS;
    applyRetentionConstants(c.retention);   // throws on a bad/missing field
    // The weakness centre selection subtracts before scoring (weakness.js). Asserted
    // here, at boot, because that is the only place it CAN be wrong: package_for_pwa
    // refuses to ship a vector whose length doesn't match the axis count, and
    // weaknessScore's embedding-length guard establishes the trained model's K is that
    // same shipped K, so the scorer indexes it unguarded. What remains is a malformed
    // packaged blob, which is a build fault and should stop the app rather than
    // quietly serve an uncentred contrast — the one failure that looks like working
    // personalization while tilting every user identically.
    if (!Array.isArray(c.weakness_contrast_mean) || c.weakness_contrast_mean.length === 0 ||
        !c.weakness_contrast_mean.every(Number.isFinite)) {
        throw new Error('population constants: weakness_contrast_mean missing or malformed');
    }
    setTimeBias({ timePerMin: c.time_bias_per_min, reviewPerMin: c.review_bias_per_min });
    _loaded = c;
    return c;
}

// --- Time-adjusted rank --------------------------------------------------
// Practice here is untimed, but server ranks reflect timed play, so a slow
// solver's untimed rank overstates their timed strength. `time_adjust` corrects an
// internal RATING by alpha · (log10(median solve minutes) − center), alpha ≤ 0 — a
// single global log-time slope + center. It is PRODUCED by pipeline.map_rating_to_rank
// (fit jointly with the rating→rank curve) and read through into the population
// constants; the per-server couplings absorb the rank-correlated part of solve time on
// their own, so there is no rank-residualization line. Shared
// by the Rank-tab chart/summary (stats-rank-pane.js) and the rank-translation modal
// (rank-translation.js) so the median feature is defined in exactly ONE place.

// The median feature is a POSITIONAL-WEIGHTED median over the last-500 rank window
// (no AFK cap), so the client and the EGD-leaderboard Worker compute an IDENTICAL
// time-adjust from the same window (the Worker only holds a 500 buffer). The window
// and weighting do NOT change what the median estimates — the user's central novel
// solve time — only its variance (fewer effective samples) and recency, so the shipped
// alpha/center (fit against median solve time) still apply as-is; no refit needed.
// Dropping the upper/AFK cap likewise leaves the median value unchanged (a median is
// robust to idle-tab outliers), so there is no AFK cap on ANY rank time-adjust
// (front page, modal, Rank tab, leaderboard) — one weighting everywhere.
export const TIME_ADJUST_MIN_TIMED = 20;

// Median solving time in minutes: a positional-weighted median over the SAME
// flattened sample the rank fit uses — flattenAttempts (first-time only, curve-
// mapped, date-ascending). Extracting the solve times from flattenAttempts' OWN
// output (not an independent novel filter) is load-bearing: flattenAttempts drops
// novel attempts that can't be rating-mapped, so pulling times from `.src` keeps the
// (M − age)/M positional weights aligned 1:1 with windowMLE — the time median and the
// rank it docks are over the exact same weighted sample, never a mismatch. The shared
// helper drops non-positive / non-finite times and returns null below MIN_TIMED valid.
export function medianNovelSolveMinutes(history) {
    const times = flattenAttempts(history).map(f => f.src?.timeTaken);
    const medMs = weightedMedianOverWindow(times, TIME_ADJUST_MIN_TIMED);
    return medMs === null ? null : medMs / 60000;
}

// The whole-history time-adjustment in RATING units:
//   alpha · (log10(medianMin) − center).
// A single global log-time slope around `center` (the population-typical log10 median),
// so a typical-speed solver gets ~0 dock. Returns null when it can't be computed
// (constants missing or too few timed attempts) — callers then show the raw,
// unadjusted rank. Apply the returned delta to an internal rank index via
// rankAfterRatingDelta.
export function timeAdjustDeltaRating(history) {
    const ta = getPopulationConstants()?.time_adjust;
    if (!ta || !Number.isFinite(ta.alpha_per_log10_min)
        || !Number.isFinite(ta.center_log10_min)) {
        return null;
    }
    const medMin = medianNovelSolveMinutes(history);
    if (medMin === null) return null;
    return ta.alpha_per_log10_min * (Math.log10(medMin) - ta.center_log10_min);
}
