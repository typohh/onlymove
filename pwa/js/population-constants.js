// js/population-constants.js
//
// Loads the fitted population model constants (pwa/tsumegos/population_constants.json,
// produced by pipeline.fit_population_constants and shipped by package_for_pwa) and
// applies them in place of the hardcoded defaults: the retention constants in
// model.js (KAPPA/Z0/GROWTH/LN_C_PRIOR) and the rushing/review time-benefit biases
// in game-logic.js. Board-independent — one blob for both sizes.
//
// Best-effort: a missing or malformed file leaves the modules on their built-in
// defaults, so the app still works (just on the last manually-set constants).
// main.js awaits this at boot, before the model trains or schedules, so the first
// stats compute already sees the fitted constants.

import { applyRetentionConstants } from './model.js';
import { setTimeBias } from './game-logic.js';

let _loaded = null;

export function getPopulationConstants() {
    return _loaded;
}

export async function loadPopulationConstants() {
    try {
        const r = await fetch(`tsumegos/population_constants.json?v=${APP_VERSION}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const c = await r.json();
        if (c.retention) applyRetentionConstants(c.retention);
        setTimeBias({ timePerMin: c.time_bias_per_min, reviewPerMin: c.review_bias_per_min });
        _loaded = c;
        return c;
    } catch (e) {
        // Non-fatal: keep the hardcoded defaults. Don't block boot on this.
        console.warn('loadPopulationConstants failed; using built-in defaults:', e);
        return null;
    }
}
