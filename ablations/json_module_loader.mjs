// ablations/json_module_loader.mjs
//
// A Node ESM `load` hook that lets the SHIPPED client modules be imported here
// unmodified. `pwa/shared/rank-window.js` and `pwa/js/irt.js` import their
// deployment constants as plain JSON modules (`import X from '../tsumegos/….json'`),
// which esbuild resolves at bundle time; bare Node instead demands an import
// attribute the source does not carry and refuses the import. Rewriting the JSON
// as a module here is what keeps the measurement running the REAL estimator
// rather than a restatement of it — the alternative would be editing pwa/, which
// would put the ablation in the shipping path.
//
// Registered by client_estimators.mjs via `module.register` before it imports
// anything from pwa/.
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

export async function load(url, context, nextLoad) {
    if (url.startsWith('file:') && url.endsWith('.json')) {
        const source = await readFile(fileURLToPath(url), 'utf8');
        return { format: 'module', shortCircuit: true, source: `export default ${source};` };
    }
    return nextLoad(url, context);
}
