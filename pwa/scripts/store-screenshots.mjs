// Play Store listing screenshot generator.
//
// Reproduces the localized store screenshots in android/store-assets/listing/
// against the LOCAL dev server, signed into a real account adopted from the
// local D1 copy — so the rank header, charts and weakness sentences show real
// data. Staged answers (needed to reach review mode) are intercepted before
// POST /report and stripped from the profile afterwards, so neither the local
// D1 nor the account's sync history ever records them.
//
// Prerequisites:
//   1. `python -m pipeline.fetch_database` reasonably fresh (the account's
//      history comes from the local D1 copy).
//   2. `npm run dev` running in pwa/ (http://localhost:8787).
//   3. puppeteer-core (pwa devDependency) + system Chrome at CHROME path.
//
// Usage (from pwa/):
//   node scripts/store-screenshots.mjs seed          # one-time: adopt ACCOUNT into a fresh profile
//   node scripts/store-screenshots.mjs shoot en      # phone set for one language
//   node scripts/store-screenshots.mjs shootpc en    # desktop/PC set for one language
//   node scripts/store-screenshots.mjs all           # seed + shoot + shootpc for all languages
//
// Output files per language (en/ko/ja/zh):
//   phone 1080x1920 (Play "Phone screenshots"):
//     1-solving, 2-review, 3-stats-rank, 4-stats-weaknesses, 5-stats-community, 8-stats-time
//   desktop 1920x1080 (Play "7-inch/10-inch tablet", "Google Play Games on PC", "Chromebook"):
//     6-desktop-solving, 7-desktop-stats, 9-desktop-review, 10-desktop-weaknesses
//
// Related one-off assets (ImageMagick, not part of this script):
//   feature graphic: convert horizontal.png -crop 1080x527+285+276 +repage -resize 1024x500! feature-graphic.png
//   PC logo (600x400): convert ../pwa/icon-512.png -alpha set -fill none -fuzz 2% \
//                      -draw "matte 0,0 floodfill" -draw "matte 511,0 floodfill" \
//                      -draw "matte 0,511 floodfill" -draw "matte 511,511 floodfill" \
//                      -trim +repage -resize 380x380 -background none -gravity center \
//                      -extent 600x400 pc-logo-600x400.png
//
// Gotchas encoded below (learned the hard way):
//   - The shown problem persists across reloads by design, so pausing recalls
//     only takes effect after the persisted recall is consumed or a reload
//     with the pause already set; `seed` + the per-stage reload handle this.
//   - The rank survey and recall serving are snoozed via their localStorage
//     keys before every capture.
//   - Stats tab order is Weaknesses(0) / Rank(1) / Time(2) / Activity(3) /
//     Community(4) — there is no Summary tab.
//   - Board click coordinates are 9x9-specific and viewport-specific.

import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const CHROME = process.env.CHROME_PATH || '/usr/bin/google-chrome';
const BASE = process.env.STORE_SHOTS_BASE || 'http://localhost:8787';
// The account whose history/rank the screenshots show. Pick a data-rich one
// from the local DB; its production data is never modified (see interception).
const ACCOUNT = process.env.STORE_SHOTS_ACCOUNT || '<redacted-user-id>';
const LANGS = ['en', 'ko', 'ja', 'zh'];

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const OUT = path.join(repoRoot, 'android', 'store-assets', 'listing');
// Persistent profile so the adopted account survives across stages/runs.
const PROFILE = path.join(os.tmpdir(), `onlymove-store-shots-${ACCOUNT}`);

const stage = process.argv[2] || 'all';
const langArg = process.argv[3];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    userDataDir: PROFILE,
    args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars'],
});

async function newPage({ width, height, dpr, lang = 'en', scheme = 'light' }) {
    const page = await browser.newPage();
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
    await page.setViewport({ width, height, deviceScaleFactor: dpr });
    await page.setExtraHTTPHeaders({ 'Accept-Language': lang });
    await page.evaluateOnNewDocument((l) => {
        Object.defineProperty(navigator, 'language', { get: () => l });
        Object.defineProperty(navigator, 'languages', { get: () => [l] });
    }, lang);
    return page;
}

// Staged answers must never persist: abort POST /report (local D1 stays
// clean) and strip the un-acked entries from localStorage afterwards (the
// offline outbox would otherwise re-upload them on a later load).
async function interceptReports(page) {
    await page.setRequestInterception(true);
    page.on('request', (req) => {
        if (req.method() === 'POST' && new URL(req.url()).pathname === '/report') req.abort();
        else req.continue();
    });
}

async function stripUnacked(page) {
    await page.evaluate(() => {
        const h = JSON.parse(localStorage.getItem('answerHistory') || '[]');
        localStorage.setItem('answerHistory', JSON.stringify(h.filter(e => e.serverId)));
    });
}

async function snoozePrompts(page) {
    await page.evaluate(() => {
        localStorage.setItem('rankSurveyShownAt_v2', String(Date.now()));
        localStorage.setItem('recallPausedUntil', String(Date.now() + 24 * 3600 * 1000));
    });
}

// Click empty-looking 9x9 points until an answer registers (button1 = Review
// enables once previousProblem exists). Two clicks per point: mobile layouts
// use tap-to-confirm; on desktop the second click is ignored.
async function answerProblem(page, xi, yj) {
    const candidates = [[0, 0], [8, 0], [0, 8], [8, 8], [4, 0], [0, 4], [8, 4], [4, 8], [2, 0], [6, 8]];
    const answered = () => page.evaluate(() => {
        const b = document.getElementById('button1');
        return b && !b.classList.contains('disabled');
    });
    for (const [i, j] of candidates) {
        if (await answered()) break;
        await page.mouse.click(xi(i), yj(j));
        await sleep(500);
        await page.mouse.click(xi(i), yj(j));
        await sleep(3500);
    }
    return answered();
}

const clickTab = (page, idx) => page.evaluate((i) => {
    const tabs = document.querySelectorAll('.stats-tab');
    if (tabs[i]) tabs[i].click();
}, idx);

async function seed() {
    const page = await newPage({ width: 360, height: 640, dpr: 3 });
    await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
    await sleep(3000);
    await page.evaluate((acct) => {
        localStorage.clear();
        localStorage.setItem('userId', acct);
        localStorage.setItem('boardSize', '9');
    }, ACCOUNT);
    await page.reload({ waitUntil: 'networkidle2' });
    let last = -1;
    for (let i = 0; i < 60; i++) {
        await sleep(5000);
        const n = await page.evaluate(() => JSON.parse(localStorage.getItem('answerHistory') || '[]').length);
        if (n !== last) console.log(`seed: t=${(i + 1) * 5}s history=${n}`);
        if (n > 500 && n === last) break;
        last = n;
    }
    await sleep(10000); // let the embedding/ratings backfill settle
    await page.close();
    console.log('seed done');
}

async function shoot(lang) {
    const dir = path.join(OUT, lang);
    fs.mkdirSync(dir, { recursive: true });
    const page = await newPage({ width: 360, height: 640, dpr: 3, lang });
    await interceptReports(page);
    await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
    await snoozePrompts(page);
    await page.reload({ waitUntil: 'networkidle2' });
    await sleep(8000);
    await page.screenshot({ path: `${dir}/1-solving.png` });

    // Stats first, on the clean history (before the staged fail).
    await page.evaluate(() => document.getElementById('menu-toggle').click());
    await sleep(800);
    await page.evaluate(() => document.getElementById('menu-stats').click());
    await sleep(12000); // charts build behind the loading overlay
    await clickTab(page, 1); await sleep(3000);
    await page.screenshot({ path: `${dir}/3-stats-rank.png` });
    await clickTab(page, 0); await sleep(2000);
    await page.screenshot({ path: `${dir}/4-stats-weaknesses.png` });
    await clickTab(page, 2); await sleep(2000);
    await page.screenshot({ path: `${dir}/8-stats-time.png` });
    await clickTab(page, 4); await sleep(5000);
    await page.screenshot({ path: `${dir}/5-stats-community.png` });
    await page.evaluate(() => document.querySelector('.stats-modal-close')?.click());
    await sleep(2000);

    // Portrait 360x640 board geometry (9x9).
    const answered = await answerProblem(page, i => Math.round(29 + 37.9 * i), j => Math.round(159 + 37.6 * j));
    console.log(`${lang}: answered=${answered}`);
    await sleep(3000);
    await page.evaluate(() => document.getElementById('button1').click());
    await sleep(6000); // evals + motif bubble
    await page.screenshot({ path: `${dir}/2-review.png` });

    await stripUnacked(page);
    await page.close();

    const desk = await newPage({ width: 1280, height: 720, dpr: 1.5, lang });
    await desk.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
    await sleep(8000);
    await desk.screenshot({ path: `${dir}/6-desktop-solving.png` });
    await desk.evaluate(() => document.getElementById('menu-toggle-h').click());
    await sleep(800);
    await desk.evaluate(() => document.getElementById('menu-stats').click());
    await sleep(12000);
    await clickTab(desk, 1);
    await sleep(3000);
    await desk.screenshot({ path: `${dir}/7-desktop-stats.png` });
    await desk.close();
    console.log(`${lang}: shoot done`);
}

async function shootpc(lang) {
    const dir = path.join(OUT, lang);
    fs.mkdirSync(dir, { recursive: true });
    const page = await newPage({ width: 1280, height: 720, dpr: 1.5, lang });
    await interceptReports(page);
    await page.goto(BASE, { waitUntil: 'networkidle2', timeout: 60000 });
    await snoozePrompts(page);
    await sleep(8000);

    // Desktop 1280x720 board geometry (9x9).
    const answered = await answerProblem(page, i => Math.round(247 + 75.75 * i), j => Math.round(57 + 75.75 * j));
    console.log(`${lang}: pc answered=${answered}`);
    await sleep(3000);
    await page.evaluate(() => document.getElementById('button1').click());
    await sleep(6000);
    await page.screenshot({ path: `${dir}/9-desktop-review.png` });

    await page.evaluate(() => document.getElementById('button1').click()); // Exit review
    await sleep(2500);
    await page.evaluate(() => document.getElementById('menu-toggle-h').click());
    await sleep(800);
    await page.evaluate(() => document.getElementById('menu-stats').click());
    await sleep(12000);
    await clickTab(page, 0);
    await sleep(2000);
    await page.screenshot({ path: `${dir}/10-desktop-weaknesses.png` });

    await stripUnacked(page);
    await page.close();
    console.log(`${lang}: shootpc done`);
}

if (stage === 'seed') {
    await seed();
} else if (stage === 'shoot') {
    await shoot(langArg || 'en');
} else if (stage === 'shootpc') {
    await shootpc(langArg || 'en');
} else if (stage === 'all') {
    await seed();
    for (const lang of LANGS) await shoot(lang);
    for (const lang of LANGS) await shootpc(lang);
} else {
    console.error(`unknown stage: ${stage} (use seed | shoot <lang> | shootpc <lang> | all)`);
    process.exit(1);
}

await browser.close();
