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
