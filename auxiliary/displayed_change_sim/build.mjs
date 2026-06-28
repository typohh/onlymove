// Bundle engine.mjs (which imports the real pwa/js client modules) into a single
// headless ESM file the runner can import under Node. Uses the esbuild installed
// in pwa/node_modules. Aliases the DOM-coupled deps to headless stubs:
//   ./state.js       → state_shim.mjs   (no localStorage / IndexedDB)
//   @sentry/browser  → sentry_stub.mjs  (no transport)
// JSON imports (rating_to_rank.json, population_constants.json) are inlined by
// esbuild's default json loader — the SAME shipped files the browser bundle uses.
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import path from 'path';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '../../');
// Resolve esbuild from the PWA's node_modules.
const require = createRequire(path.join(REPO, 'pwa', 'package.json'));
const esbuild = require('esbuild');

const STATE_SHIM = path.join(HERE, 'state_shim.mjs');
const SENTRY_STUB = path.join(HERE, 'sentry_stub.mjs');

const aliasPlugin = {
  name: 'headless-alias',
  setup(build) {
    // Redirect the real ./state.js imports inside model.js / rank-decomposition.js
    // to the shim. (state_shim.mjs itself ends in _shim.mjs, so it is not matched.)
    build.onResolve({ filter: /(^|\/)state\.js$/ }, () => ({ path: STATE_SHIM }));
    build.onResolve({ filter: /^@sentry\/browser$/ }, () => ({ path: SENTRY_STUB }));
  },
};

await esbuild.build({
  entryPoints: [path.join(HERE, 'engine.mjs')],
  outfile: path.join(HERE, 'engine.bundle.mjs'),
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node18',
  plugins: [aliasPlugin],
  logLevel: 'info',
});
console.log('built engine.bundle.mjs');
