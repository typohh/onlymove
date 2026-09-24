// ablations/improvement.mjs
//
// The measurement half of `ablations/improvement_figures.py`: the rank the SHIPPED
// window estimator gives each window of first-time attempts the producer hands it.
// It imports pwa/shared/rank-window.js unmodified (flattenAttempt, windowMLE), so a
// change to the estimator moves the figure instead of quietly invalidating it. JSON
// modules load through json_module_loader.mjs, as in client_estimators.mjs.
//
// Usage:  node ablations/improvement.mjs <payload.json>
// Payload: { pwa_root, windows: [[[rating, slope, floor, success], ...], ...] },
// each window oldest -> newest. Writes { rank: [posterior mean per window] }.
// Nothing here is random.

import { register } from 'node:module';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

register('./json_module_loader.mjs', import.meta.url);

const payload = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const PWA = payload.pwa_root;
const { windowMLE, flattenAttempt } = await import(pathToFileURL(PWA + 'shared/rank-window.js').href);

// One shared cross-board curve, as pwa/src/leaderboard.js reads it.
const CURVE = JSON.parse(readFileSync(PWA + 'tsumegos/rating_to_rank.json', 'utf8'))['9'];
if (!CURVE || !Array.isArray(CURVE.ratings)) throw new Error('rating_to_rank.json has no usable curve');

const rank = payload.windows.map((w) => {
    const flat = w.map(([r, s, f, ok]) => {
        const a = flattenAttempt(r, s, f, ok, CURVE);
        if (!a) throw new Error('flattenAttempt refused a shipped problem');
        return a;
    });
    // The posterior MEAN, not the shown rank: the display offset depends on the
    // posterior's width, which differs between a player's first and latest windows
    // for reasons that are not a change in strength.
    return windowMLE(flat).R;
});
process.stdout.write(JSON.stringify({ rank }));
