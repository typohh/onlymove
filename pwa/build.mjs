import esbuild from 'esbuild';
import { execSync } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

const DIST_DIR = 'dist';

async function main() {
    console.log("🚀 Starting production build...");

    const gitHash = execSync('git rev-parse HEAD').toString().trim().slice(0, 8);
    const isDirty = execSync('git status --porcelain').toString().trim().length > 0;
    const version = isDirty ? `${gitHash}-dirty-${Date.now().toString(36)}` : gitHash;
    console.log(`✅ Version: ${version}${isDirty ? ' (dirty tree)' : ''}`);

    await fs.rm(DIST_DIR, { recursive: true, force: true });
    await fs.mkdir(DIST_DIR);

    // tsumegos is now a directory of size-keyed subdirs (e.g. tsumegos/9/,
    // tsumegos/13/), each containing its own zips and metadata.json.
    // fs.cp recursive picks up the whole tree.
    const staticAssets = [
        'index.html',
        'sound', 'lib', 'tsumegos', 'manifest.json', 'sw.js', 'lookup.txt', 'about.html',
        'apple-touch-icon.png',
        'favicon.ico',
        'icon-192.png',
        'icon-512.png',
        'horizontal.png',
        'vertical.png'
    ];
    for (const asset of staticAssets) {
        const source = path.join(process.cwd(), asset);
        const destination = path.join(DIST_DIR, asset);
        try {
            await fs.cp(source, destination, { recursive: true });
        } catch (error) {
            console.warn(`Could not copy '${asset}': ${error.message}`);
        }
    }
    console.log("✅ Static assets copied.");

    const buildResult = await esbuild.build({
        entryPoints: ['js/main.js', 'style.css'],
        bundle: true,
        minify: false,
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
    }).catch(() => process.exit(1));

    console.log("✅ JavaScript and CSS bundled and hashed.");

    const meta = buildResult.metafile.outputs;
    let newScriptPath = '';
    let newCssPath = '';
    let mainJsHash = '';

    for (const outputPath in meta) {
        const outputDetails = meta[outputPath];
        if (outputDetails.entryPoint === 'js/main.js') {
            newScriptPath = path.relative(DIST_DIR, outputPath).replace(/\\/g, '/');
            // Extract hash from the filename (format: main.[hash].js)
            const filename = path.basename(outputPath);
            mainJsHash = filename.split('.')[1];
        }
        if (outputDetails.entryPoint === 'style.css') {
            newCssPath = path.relative(DIST_DIR, outputPath).replace(/\\/g, '/');
        }
    }

    if (!newScriptPath || !newCssPath) {
        console.error("❌ Could not find hashed script or css path in esbuild metafile.");
        process.exit(1);
    }

    const htmlPath = path.join(DIST_DIR, 'index.html');
    let htmlContent = await fs.readFile(htmlPath, 'utf-8');

    // This regex will find the script tag, ignoring extra attributes.
    const scriptRegex = /<script[^>]*src\s*=\s*["']js\/main\.js["'][^>]*><\/script>/;
    htmlContent = htmlContent.replace(scriptRegex, `<script src="${newScriptPath}" type="module"></script>`);
    console.log(`✅ index.html updated with script: <script src="${newScriptPath}"></script>`);

    // This regex will find your link tag, ignoring extra attributes like type="text/css".
    const cssLinkRegex = /<link[^>]*href\s*=\s*["']style\.css["'][^>]*>/;
    htmlContent = htmlContent.replace(cssLinkRegex, `<link rel="stylesheet" href="${newCssPath}">`);
    console.log(`✅ index.html updated with stylesheet: <link rel="stylesheet" href="${newCssPath}">`);

    await fs.writeFile(htmlPath, htmlContent);

    // Update about.html with version information
    const aboutPath = path.join(DIST_DIR, 'about.html');
    let aboutContent = await fs.readFile(aboutPath, 'utf-8');
    
    // Get current UTC date and time
    const now = new Date();
    const utcDate = now.toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
    
    // Update version information
    aboutContent = aboutContent.replace('<span id="version-date"></span>', `<span id="version-date">${utcDate}</span>`);
    aboutContent = aboutContent.replace('<span id="version-hash"></span>', `<span id="version-hash"> (${mainJsHash})</span>`);
    
    await fs.writeFile(aboutPath, aboutContent);
    console.log(`✅ about.html updated with version: ${utcDate} (${mainJsHash})`);

    const swPath = path.join(DIST_DIR, 'sw.js');
    let swContent = await fs.readFile(swPath, 'utf-8');
    swContent = swContent.replace(/'initial-version'/, `'${version}'`);
    await fs.writeFile(swPath, swContent);
    console.log("✅ Service worker updated with new version.");

    console.log("🎉 Build complete!");
}

main();