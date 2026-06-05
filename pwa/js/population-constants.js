// js/population-constants.js
//
// Loads the fitted population model constants (pwa/tsumegos/population_constants.json,
// produced by pipeline.fit_population_constants and shipped by package_for_pwa) and
// applies them: the retention constants in model.js (KAPPA/Z0/GROWTH/LN_C_PRIOR) and
// the rushing/review time-benefit biases in game-logic.js. Board-independent — one
// blob for both sizes.
//
// MANDATORY asset, NO fallback: these constants back the recall scheduler and the
// nudge gates, and the modules carry no baked defaults. main.js aborts boot if this
// can't load (mirroring the rating→rank map M). The pipeline always ships a valid
// file (the fitter writes population-prior defaults when data is sparse) and sw.js
// precaches it, so a transient network blip — not a real absence — is the only
// realistic failure; retry with backoff before giving up.

import { applyRetentionConstants } from './model.js';
import { setTimeBias } from './game-logic.js';

const LOAD_ATTEMPTS = 4;
let _loaded = null;
let _loadPromise = null;

export function getPopulationConstants() {
    return _loaded;
}

// Memoize the eventual constants, NOT a rejected promise: a failed attempt must not
// poison the cache, so a later caller can re-trigger a load (mirrors loadRatingRankMap).
export function loadPopulationConstants() {
    if (_loadPromise) return _loadPromise;
    _loadPromise = (async () => {
        let lastErr = null;
        for (let attempt = 0; attempt < LOAD_ATTEMPTS; attempt++) {
            try {
                const r = await fetch(`tsumegos/population_constants.json?v=${APP_VERSION}`);
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const c = await r.json();
                applyRetentionConstants(c.retention);   // throws on a bad/missing field
                setTimeBias({ timePerMin: c.time_bias_per_min, reviewPerMin: c.review_bias_per_min });
                _loaded = c;
                return c;
            } catch (e) {
                // Preserve the real cause (network vs HTTP status vs JSON/validation)
                // so a systematic failure is distinguishable from a transient blip.
                lastErr = e;
                console.warn(`loadPopulationConstants attempt ${attempt + 1}/${LOAD_ATTEMPTS} failed:`, e);
                if (attempt < LOAD_ATTEMPTS - 1) {
                    await new Promise(res => setTimeout(res, 300 * 2 ** attempt));
                }
            }
        }
        // Exhausted retries. Drop the memo so a later call can retry, surface the
        // cause to Sentry, and return null for main.js to handle (fatal abort).
        _loadPromise = null;
        _loaded = null;
        try {
            if (typeof Sentry !== 'undefined') {
                Sentry.captureException(lastErr, {
                    extra: { attempts: LOAD_ATTEMPTS, what: 'loadPopulationConstants' },
                });
            }
        } catch (_) {}
        return null;
    })();
    return _loadPromise;
}
