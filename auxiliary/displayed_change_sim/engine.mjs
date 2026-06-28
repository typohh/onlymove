// esbuild entry: pull the REAL client recall model + rank estimator into one
// headless bundle and re-export the functions the runner drives. build.mjs
// aliases ./state.js → state_shim.mjs and @sentry/browser → sentry_stub.mjs, and
// inlines the shipped pwa/tsumegos/*.json (M curve, population constants) exactly
// as the browser bundle does — so this runs the production math, not a re-port.

export { estimateWindowRank, normalizedRankForSize } from '../../pwa/js/rank-decomposition.js';
export { freshFailureReviewHours, calculateAndStoreStatsResults, recallImpactWithRefine, MODEL_INDICES } from '../../pwa/js/model.js';
// loadPopulationConstants() is the real boot path: applies retention constants
// (KAPPA/Z0/GROWTH/LN_C_PRIOR) + the time/review bias into model.js. Idempotent.
export { loadPopulationConstants } from '../../pwa/js/population-constants.js';
// State shim surface the runner reads/writes between attempts.
export { state, getMergedAnswerHistory } from './state_shim.mjs';
