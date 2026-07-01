// js/population-constants.js
//
// Loads the fitted population model constants (pwa/tsumegos/population_constants.json,
// produced by pipeline.fit_population_constants and shipped by package_for_pwa) and
// applies them: the retention constants in model.js (KAPPA/Z0/GROWTH/LN_C_PRIOR) and
// the rushing/review time-benefit biases in game-logic.js. Board-independent — one
// blob for both sizes.
//
// MANDATORY, NO fallback: these constants back the recall scheduler and the nudge
// gates, and the modules carry no baked defaults. They are inlined into the bundle
// at build time (the import below) rather than fetched, so they are present the
// instant the app boots — no runtime fetch, no precache, no eviction/offline path
// that could fail. main.js still null-guards against a malformed packaged file.

import { applyRetentionConstants, setTimeBias } from './model.js';
import { weightedMedianOverWindow } from '../shared/rank-window.js';
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
// by the Rank-tab chart/summary (stats-ui.js) and the rank-translation modal
// (rank-translation.js) so the median feature is defined in exactly ONE place.

// The median feature is a POSITIONAL-WEIGHTED median over the last-500 rank window
// (no AFK cap), so the client and the EGD-leaderboard Worker compute an IDENTICAL
// time-adjust from the same window (the Worker only holds a 500 buffer). The window
// and weighting do NOT change what the median estimates — the user's central novel
// solve time — only its variance (fewer effective samples) and recency, so the shipped
// alpha/center (fit against median solve time) still apply as-is; no refit needed.
// Dropping the upper/AFK cap likewise leaves the median value unchanged (a median is
// robust to idle-tab outliers). TIME_ADJUST_AFK_CAP_MS is no longer read here, but
// stats-compute's makeMedianSolveMinutesAt (the Time-chart smoother) still applies
// it, so the two median-solve-time computations deliberately diverge on AFK handling.
export const TIME_ADJUST_AFK_CAP_MS = 180000;
export const TIME_ADJUST_MIN_TIMED = 20;

// Median solving time in minutes: a positional-weighted median over the SAME last-500
// novel window (and the same (M − age)/M weights) as the front-page rank fit — see
// shared/rank-window.js. Filter + sort to match rank-decomposition.js flattenAttempts
// (first-time only, date-ascending) so the weights line up; the shared helper drops
// non-positive / non-finite times (no upper/AFK cap) and returns null below
// TIME_ADJUST_MIN_TIMED valid times.
export function medianNovelSolveMinutes(history) {
    const times = history
        .filter(p => typeof p.date === 'number' && !p.isRepeat && !p.linked)
        .sort((a, b) => a.date - b.date)
        .map(p => p.timeTaken);
    const medMs = weightedMedianOverWindow(times, TIME_ADJUST_MIN_TIMED);
    return medMs === null ? null : medMs / 60000;
}

// The whole-history time-adjustment in RATING units:
//   alpha · (log10(medianMin) − center).
// A single global log-time slope around `center` (the population-typical log10 median),
// so a typical-speed solver gets ~0 dock. Returns null when it can't be computed
// (constants missing or too few timed attempts) — callers then show the raw,
// unadjusted rank. Apply the returned delta to an internal rank index via
// rankIndexAfterRatingDelta.
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
