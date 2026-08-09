#!/usr/bin/env node
// auxiliary/check_live_status.mjs
//
// Probe the deployed Worker and report whether it is healthy and whether writes
// are frozen. Run it after a deploy, and on both sides of a maintenance freeze.
//
// It exists because the live system has TWO routes that return 404 on purpose,
// and both read as an outage if you probe them by hand:
//
//   1. NAVIGATION-ONLY SPA FALLBACK. `/`, `/9` and `/13` are not files; the
//      Worker rewrites them to index.html, but ONLY when the request looks like
//      a page navigation (`sec-fetch-dest: document`, or an `Accept` naming
//      text/html) — see the fallback in pwa/src/index.js. An asset-style request
//      must keep getting a real 404, because a stale service worker asking for a
//      bundle hash that no longer exists has to see 404 and re-fetch, not get
//      HTML where JS is expected and trip strict MIME checking. Plain `curl`
//      sends `Accept: */*`, so it gets the 404 and the site looks dead. This
//      script probes each route BOTH ways and asserts both answers, so the
//      distinction is checked rather than remembered.
//
//   2. INLINED DATA FILES. The five entries in build.mjs's INLINED_DATA_FILES
//      (rating_to_rank, rank_map, population_constants, se_floor,
//      embedding_meta) are compiled into the JS bundle and deliberately NOT
//      shipped as assets, so /tsumegos/<those>.json 404. A 200 there is the
//      regression worth catching: it means they are being served under a fixed
//      name, where a client could fetch one without the ?v=DATA_VERSION
//      cache-buster and pin itself to a stale curve.
//
// The freeze probe POSTs `{}` to /sync, which is rejected on its body before any
// D1 access, so it writes nothing. Frozen returns 503 {"error":"maintenance"};
// healthy returns a 4xx from validation. /report is deliberately NOT used — it
// would insert a real attempt row.
//
// A deploy takes a little while to propagate, and during that window the old
// version still answers; --retries re-probes rather than reporting a stale
// answer as a failure. (Observed 2026-08-09: an unfreeze deploy reported success
// while /sync still returned maintenance, which reads as a failed unfreeze.)
//
// Usage:
//   node auxiliary/check_live_status.mjs                      # probe production, report
//   node auxiliary/check_live_status.mjs --expect-frozen       # ... and require writes FROZEN
//   node auxiliary/check_live_status.mjs --expect-unfrozen     # ... and require writes OPEN
//   node auxiliary/check_live_status.mjs --origin http://localhost:8787
//   node auxiliary/check_live_status.mjs --retries 8 --delay 5 # slow propagation
//
// Exits non-zero if any expectation fails, so it can gate a deploy or a runbook step.

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const value = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const ORIGIN = (value('--origin', 'https://onlymove.net')).replace(/\/$/, '');
const RETRIES = Number(value('--retries', '6'));
const DELAY_S = Number(value('--delay', '5'));
const expectFrozen = flag('--expect-frozen');
const expectUnfrozen = flag('--expect-unfrozen');

// A browser navigation. The SPA fallback keys off exactly these.
const NAV_HEADERS = {
  accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
  'sec-fetch-dest': 'document',
  'sec-fetch-mode': 'navigate',
};
// An asset fetch: what a service worker sends for a hashed bundle.
const ASSET_HEADERS = { accept: '*/*' };

// Routes with no file behind them: the Worker rewrites to index.html for a
// navigation, and must 404 for anything else (see note 1 above).
const SPA_ROUTES = ['/', '/9', '/13'];
// Real files. These serve regardless of how they are asked for.
const STATIC_ROUTES = ['/index.html', '/about.html', '/changelog.html', '/privacy.html',
                       '/manifest.json', '/sw.js', '/lookup.txt'];
// Worker-rendered pages (not assets), so they need the navigation headers too.
const PAGE_ROUTES = ['/leaderboard'];
// Packaged data the client Range-fetches at runtime. Must be present.
const SERVED_DATA = ['/tsumegos/lookup.embeddings.bin', '/tsumegos/problem_params.bin'];
// Compiled into the bundle instead of shipped — see note 2 above.
const INLINED_DATA = ['/tsumegos/rating_to_rank.json', '/tsumegos/rank_map.json',
                      '/tsumegos/population_constants.json', '/tsumegos/se_floor.json',
                      '/tsumegos/embedding_meta.json'];

const sleep = (s) => new Promise((r) => setTimeout(r, s * 1000));
const results = [];
let failed = 0;

function record(ok, label, detail) {
  results.push({ ok, label, detail });
  if (!ok) failed++;
}

async function status(path, headers) {
  try {
    const res = await fetch(ORIGIN + path, { headers, redirect: 'manual' });
    return res.status;
  } catch (e) {
    return `ERR ${e.message}`;
  }
}

async function expect(path, headers, want, label) {
  const got = await status(path, headers);
  record(got === want, label, `${path} -> ${got} (want ${want})`);
}

// Frozen / open / unknown, retried: a deploy that has not propagated still
// answers from the previous version.
async function probeFreeze() {
  for (let attempt = 1; attempt <= RETRIES; attempt++) {
    let state = 'unknown', detail = '';
    try {
      const res = await fetch(ORIGIN + '/sync', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: '{}',
      });
      const body = await res.text();
      if (res.status === 503 && body.includes('maintenance')) {
        state = 'frozen';
      } else if (res.status < 500) {
        state = 'open';
      }
      detail = `POST /sync -> ${res.status} ${body.slice(0, 60).trim()}`;
    } catch (e) {
      detail = `POST /sync -> ERR ${e.message}`;
    }
    const settled = (expectFrozen && state === 'frozen') ||
                    (expectUnfrozen && state === 'open') ||
                    (!expectFrozen && !expectUnfrozen && state !== 'unknown');
    if (settled || attempt === RETRIES) return { state, detail, attempts: attempt };
    process.stdout.write(`  writes ${state}; re-probing in ${DELAY_S}s ` +
                         `(${attempt}/${RETRIES - 1})\n`);
    await sleep(DELAY_S);
  }
}

console.log(`probing ${ORIGIN}\n`);

for (const p of SPA_ROUTES) {
  await expect(p, NAV_HEADERS, 200, 'spa route serves a navigation');
  // Deliberate: an asset-style miss must stay a 404 so a stale SW re-fetches.
  await expect(p, ASSET_HEADERS, 404, 'spa route 404s a non-navigation');
}
for (const p of [...STATIC_ROUTES, ...SERVED_DATA]) {
  await expect(p, ASSET_HEADERS, 200, 'static asset served');
}
for (const p of PAGE_ROUTES) {
  await expect(p, NAV_HEADERS, 200, 'worker page served');
}
for (const p of INLINED_DATA) {
  // 200 here = shipped as an asset under a fixed name, bypassing ?v=DATA_VERSION.
  await expect(p, ASSET_HEADERS, 404, 'inlined data file not shipped as an asset');
}

const freeze = await probeFreeze();

const width = Math.max(...results.map((r) => r.label.length));
for (const r of results) {
  console.log(`  ${r.ok ? 'ok  ' : 'FAIL'}  ${r.label.padEnd(width)}  ${r.detail}`);
}
console.log(`\n  writes: ${freeze.state.toUpperCase()}` +
            `${freeze.attempts > 1 ? ` (settled after ${freeze.attempts} probes)` : ''}` +
            `\n          ${freeze.detail}`);

if (expectFrozen && freeze.state !== 'frozen') {
  console.log('\nFAIL: expected writes FROZEN.');
  failed++;
} else if (expectUnfrozen && freeze.state !== 'open') {
  console.log('\nFAIL: expected writes OPEN — if a freeze deploy just went out, ' +
              'it may still be propagating; re-run before treating it as stuck.');
  failed++;
} else if (freeze.state === 'unknown') {
  console.log('\nFAIL: could not determine the write-freeze state.');
  failed++;
}

console.log(failed ? `\n${failed} check(s) FAILED.` : '\nall checks passed.');
process.exit(failed ? 1 : 0);
