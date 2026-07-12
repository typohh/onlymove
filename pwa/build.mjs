import esbuild from 'esbuild';
import { execSync, spawn } from 'child_process';
import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';
import crypto from 'node:crypto';

const EMBEDDING_META_PATH = 'tsumegos/embedding_meta.json';

// Data files inlined into the JS/Worker bundles at build time (imported, never
// fetched). They are dead weight in dist (dropped via DIST_DEAD_FILES) AND folded
// into APP_VERSION (computeVersion), so ANY data change moves the reload signal +
// SW precache name — pwa/tsumegos/ is gitignored, so git HEAD alone can't see a
// data-only re-ship.
const INLINED_DATA_FILES = [
    'tsumegos/rating_to_rank.json',
    'tsumegos/rank_map.json',
    'tsumegos/population_constants.json',
    EMBEDDING_META_PATH,
];

const DIST_DIR = 'dist';
const STATIC_ASSETS = [
    'index.html',
    'sound', 'lib', 'tsumegos', 'manifest.json', 'sw.js', 'lookup.txt', 'about.html', 'privacy.html',
    // Digital Asset Links for the Play Store TWA (net.onlymove.twa): must be
    // served at /.well-known/assetlinks.json or the Android app shows the
    // browser URL bar. Carries the upload-key fingerprint + the Play App
    // Signing fingerprint (the cert Play actually signs releases with).
    '.well-known',
    // Workers-assets header rules (must sit at the dist root). The immutable
    // Cache-Control rules on the content-hashed / ?v-busted data files are what
    // let Cloudflare's edge cache store them — and cache HITS are the only path
    // that serves 206 Range responses (the assets layer itself answers Range
    // with 200 + the full body).
    '_headers',
    'corners',
    'apple-touch-icon.png',
    'favicon.ico',
    'icon-192.png',
    'icon-512.png',
    'icon-maskable-192.png',
    'icon-maskable-512.png',
    'horizontal.png',
    'vertical.png',
];

const args = process.argv.slice(2);
const WATCH = args.includes('--watch');
const WITH_WRANGLER = args.includes('--with-wrangler');

// Clear dist contents in place — never rm the directory itself, because
// wrangler's asset binding watches by directory handle and a swapped inode
// would silently 404 every new file until restart.
async function clearDistContents() {
    await fs.mkdir(DIST_DIR, { recursive: true });
    const entries = await fs.readdir(DIST_DIR);
    await Promise.all(entries.map(e => fs.rm(path.join(DIST_DIR, e), { recursive: true, force: true })));
}

// data_version: the content hash package_for_pwa stamped into embedding_meta.json
// (a hash of the served metadata.json files). The client reads the same value from
// the inlined manifest (data-version.js); we stamp it into sw.js so the runtime-cache
// prune keeps DATA_VERSION-busted entries (metadata.json/lookup.txt).
function readDataVersion() {
    try {
        const meta = JSON.parse(fsSync.readFileSync(EMBEDDING_META_PATH, 'utf-8'));
        if (meta && typeof meta.data_version === 'string' && meta.data_version) {
            return meta.data_version;
        }
    } catch { /* no packaged data yet */ }
    return 'nodata';
}

function computeVersion() {
    const gitHash = execSync('git rev-parse HEAD').toString().trim().slice(0, 8);
    // maxBuffer bumped because `package_for_pwa` regenerates thousands of
    // embedding/zip files; default 1 MB buffer overflows on a fresh package.
    const isDirty = execSync('git status --porcelain',
        { maxBuffer: 64 * 1024 * 1024 }).toString().trim().length > 0;
    const base = isDirty ? `${gitHash}-dirty-${Date.now().toString(36)}` : gitHash;
    // pwa/tsumegos/ (the inlined data files) is gitignored, so git HEAD is blind to a
    // data-only re-ship. Fold a hash of every inlined data file in so APP_VERSION (the
    // graceful-reload signal + SW precache name) moves whenever ratings, the curve,
    // labels, or population constants change, even from the same commit.
    // (metadata.json/lookup.txt fetch freshness rides DATA_VERSION separately, so a
    // pure-code deploy still won't re-pull them.)
    let dataTag = '';
    try {
        const h = crypto.createHash('sha256');
        for (const f of INLINED_DATA_FILES) {
            try { h.update(fsSync.readFileSync(f)); } catch { h.update(`missing:${f}`); }
        }
        dataTag = '-d' + h.digest('hex').slice(0, 8);
    } catch { /* ignore */ }
    return base + dataTag;
}

async function writeWorkerVersion(version) {
    await fs.writeFile(
        path.join('src', 'version.js'),
        `export const APP_VERSION = '${version}';\n`,
    );
}

// rating_to_rank.json, rank_map.json and population_constants.json are inlined
// into the JS and Worker bundles at build time (rank-decomposition.js,
// rank-translation.js, population-constants.js, src/index.js), so they're never
// fetched at runtime. The pipeline still writes them to tsumegos/ as the import
// source, but the served copies are dead weight — drop them from dist. Safe
// because every client force-upgrades on a version change, so nothing is still
// fetching the old served path.
// All inlined data files are dead weight in dist (embedding_meta.json joined the set
// once it became an import — data-version.js — so it can never be served stale). The
// build reads them from the SOURCE tree (not dist) for the data tag + sw.js stamp, so
// this dist removal order doesn't matter.
const DIST_DEAD_FILES = INLINED_DATA_FILES;

async function copyStaticAssets() {
    for (const asset of STATIC_ASSETS) {
        const source = path.join(process.cwd(), asset);
        const destination = path.join(DIST_DIR, asset);
        try {
            await fs.cp(source, destination, { recursive: true, force: true });
        } catch (error) {
            console.warn(`Could not copy '${asset}': ${error.message}`);
        }
    }
    for (const dead of DIST_DEAD_FILES) {
        await fs.rm(path.join(DIST_DIR, dead), { force: true });
    }
}

// Reads esbuild's metafile and patches dist/index.html and dist/about.html
// with the current hashed asset paths and version. Called after every esbuild
// build (initial and watch rebuilds) so the served HTML always references
// the freshest hashes. Re-copies the HTML/SW source files before patching so
// the regex always matches against the unpatched source placeholders.
async function patchHtmlAfterBuild(metafile, version) {
    let newScriptPath = '';
    let newCssPath = '';
    let mainJsHash = '';
    for (const outputPath in metafile.outputs) {
        const details = metafile.outputs[outputPath];
        if (details.entryPoint === 'js/main.js') {
            newScriptPath = path.relative(DIST_DIR, outputPath).replace(/\\/g, '/');
            mainJsHash = path.basename(outputPath).split('.')[1];
        }
        if (details.entryPoint === 'style.css') {
            newCssPath = path.relative(DIST_DIR, outputPath).replace(/\\/g, '/');
        }
    }
    if (!newScriptPath || !newCssPath) {
        throw new Error('Could not find hashed script or css path in esbuild metafile.');
    }

    for (const file of ['index.html', 'about.html', 'sw.js']) {
        await fs.cp(path.join(process.cwd(), file), path.join(DIST_DIR, file), { force: true });
    }

    const htmlPath = path.join(DIST_DIR, 'index.html');
    let htmlContent = await fs.readFile(htmlPath, 'utf-8');
    htmlContent = htmlContent.replace(
        /<script[^>]*src\s*=\s*["']js\/main\.js["'][^>]*><\/script>/,
        `<script src="${newScriptPath}" type="module"></script>`,
    );
    htmlContent = htmlContent.replace(
        /<link[^>]*href\s*=\s*["']style\.css["'][^>]*>/,
        `<link rel="stylesheet" href="${newCssPath}">`,
    );
    await fs.writeFile(htmlPath, htmlContent);

    const aboutPath = path.join(DIST_DIR, 'about.html');
    let aboutContent = await fs.readFile(aboutPath, 'utf-8');
    const utcDate = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    aboutContent = aboutContent.replace('<span id="version-date"></span>', `<span id="version-date">${utcDate}</span>`);
    aboutContent = aboutContent.replace('<span id="version-hash"></span>', `<span id="version-hash"> (${mainJsHash})</span>`);
    await fs.writeFile(aboutPath, aboutContent);

    const swPath = path.join(DIST_DIR, 'sw.js');
    let swContent = await fs.readFile(swPath, 'utf-8');
    swContent = swContent.replace(/'initial-version'/, `'${version}'`);
    swContent = swContent.replace(/'initial-data-version'/, `'${readDataVersion()}'`);
    await fs.writeFile(swPath, swContent);

    return { newScriptPath, newCssPath, mainJsHash, utcDate };
}

const esbuildOptions = (version) => ({
    entryPoints: ['js/main.js', 'style.css'],
    bundle: true,
    // Minify the shipped bundle (mobile-first; the client bundle is ~1 MB
    // unminified) but emit external source maps so Sentry stack traces and
    // the in-app debug-log viewer still resolve to readable source. The .map
    // files are copied to dist/assets and served as ordinary static assets.
    minify: true,
    sourcemap: true,
    format: 'esm',
    outdir: `${DIST_DIR}/assets`,
    entryNames: '[name].[hash]',
    assetNames: '[name].[hash]',
    loader: {
        '.png': 'file', '.jpg': 'file', '.jpeg': 'file', '.gif': 'file', '.svg': 'file',
        '.woff': 'file', '.woff2': 'file', '.ttf': 'file', '.eot': 'file',
    },
    metafile: true,
    define: { 'APP_VERSION': `"${version}"` },
    external: ['*.png'],
});

async function oneShotBuild() {
    console.log('🚀 Starting production build...');
    const version = computeVersion();
    console.log(`✅ Version: ${version}`);

    await writeWorkerVersion(version);
    await clearDistContents();
    await copyStaticAssets();
    console.log('✅ Static assets copied.');

    const result = await esbuild.build(esbuildOptions(version));
    console.log('✅ JavaScript and CSS bundled and hashed.');

    const { newScriptPath, newCssPath, utcDate, mainJsHash } = await patchHtmlAfterBuild(result.metafile, version);
    console.log(`✅ index.html → <script src="${newScriptPath}">`);
    console.log(`✅ index.html → <link href="${newCssPath}">`);
    console.log(`✅ about.html updated: ${utcDate} (${mainJsHash})`);
    console.log('✅ Service worker version stamped.');
    console.log('🎉 Build complete!');
}

async function watchBuild() {
    console.log('🚀 Starting watch build...');
    const version = computeVersion();
    await writeWorkerVersion(version);
    await clearDistContents();
    await copyStaticAssets();
    console.log('✅ Initial static copy + version done.');

    // esbuild context with onEnd plugin — every successful rebuild repatches
    // the HTML files so the served bundle stays in sync with the latest hash.
    const ctx = await esbuild.context({
        ...esbuildOptions(version),
        plugins: [{
            name: 'on-rebuild',
            setup(build) {
                build.onEnd(async (result) => {
                    if (result.errors.length > 0) {
                        console.error('❌ Build failed:', result.errors);
                        return;
                    }
                    try {
                        const { newScriptPath, newCssPath } = await patchHtmlAfterBuild(result.metafile, version);
                        console.log(`✅ rebuilt → ${newScriptPath} / ${newCssPath}`);
                    } catch (err) {
                        console.error('❌ Post-build patch failed:', err);
                    }
                });
            },
        }],
    });
    await ctx.watch();
    console.log('👀 Watching js/, style.css, and dependencies...');

    // esbuild only watches the bundle's import graph. index.html and sw.js
    // are static-copied, so watch them separately and trigger a recopy +
    // repatch when they change. Coalesce bursts so a single editor save
    // doesn't fire multiple rebuilds in quick succession.
    let staticRebuildTimer = null;
    const scheduleStaticRebuild = () => {
        if (staticRebuildTimer) clearTimeout(staticRebuildTimer);
        staticRebuildTimer = setTimeout(async () => {
            staticRebuildTimer = null;
            try {
                await copyStaticAssets();
                const result = await ctx.rebuild();
                await patchHtmlAfterBuild(result.metafile, version);
                console.log('✅ static file change → rebuilt');
            } catch (err) {
                console.error('❌ static-driven rebuild failed:', err);
            }
        }, 100);
    };
    for (const file of ['index.html', 'sw.js', 'about.html']) {
        try {
            fsSync.watch(file, () => scheduleStaticRebuild());
        } catch (err) {
            console.warn(`Could not watch '${file}': ${err.message}`);
        }
    }

    if (WITH_WRANGLER) {
        const wrangler = spawn('npx', ['wrangler', 'dev'], { stdio: 'inherit' });
        const shutdown = (code) => {
            try { wrangler.kill(); } catch {}
            ctx.dispose().catch(() => {}).finally(() => process.exit(code ?? 0));
        };
        process.on('SIGINT', () => shutdown(0));
        process.on('SIGTERM', () => shutdown(0));
        wrangler.on('exit', (code) => shutdown(code ?? 0));
    } else {
        console.log('(watch mode active — Ctrl+C to stop)');
    }
}

if (WATCH) {
    await watchBuild();
} else {
    await oneShotBuild();
}
