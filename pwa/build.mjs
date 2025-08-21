import esbuild from 'esbuild';
import { execSync } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

const DIST_DIR = 'dist';

async function main() {
    console.log("🚀 Starting production build...");

    // 1. Get the current git commit hash for versioning
    const gitHash = execSync('git rev-parse --short HEAD').toString().trim();
    console.log(`✅ Version (Git Hash): ${gitHash}`);

    // 2. Clean up the dist directory
    await fs.rm(DIST_DIR, { recursive: true, force: true });
    await fs.mkdir(DIST_DIR);

    // 3. Copy all static assets to dist
    const staticAssets = [
	'index.html', 'style.css', 'sound', 'lib', 'tsumegos', 'metadata.json', 'manifest.json', 'sw.js', 'lookup.txt',
	'apple-touch-icon.png',
	'favicon.ico',
	'icon-192.png',
	'icon-512.png',
	'horizontal.png',
	'vertical.png']
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

    // 4. Build, bundle, minify, and hash the JavaScript
    const buildResult = await esbuild.build({
        entryPoints: ['js/main.js'],
        bundle: true,
        minify: false,  // Disable minification for better debugging
        format: 'esm',
        entryNames: 'assets/[name].[hash]', // This creates the hashed filename
        outdir: `${DIST_DIR}`,
        metafile: true, // Needed to find out the final hashed filename
        define: {
            // Inject the git hash as a global constant named APP_VERSION
            'APP_VERSION': `"${gitHash}"`
        },
    }).catch(() => process.exit(1));

    console.log("✅ JavaScript bundled and minified.");

    // 5. Update index.html to point to the new hashed JS file
    const meta = buildResult.metafile.outputs;
    let newScriptPath = '';

    // Find the output path of our main.js entry point
    for (const key in meta) {
        if (meta[key].entryPoint === 'js/main.js') {
            // Make the path relative to the root for the HTML file
            newScriptPath = key.replace(`${DIST_DIR}/`, '');
            break;
        }
    }

    if (!newScriptPath) {
        console.error("❌ Could not find hashed script path in esbuild metafile.");
        process.exit(1);
    }

    const htmlPath = path.join(DIST_DIR, 'index.html');
    let htmlContent = await fs.readFile(htmlPath, 'utf-8');
    
    // Replace the original script tag with the new one
    htmlContent = htmlContent.replace(
        '<script type="module" src="js/main.js"></script>',
        `<script src="${newScriptPath}" type="module"></script>`
    );
    
    await fs.writeFile(htmlPath, htmlContent);
    console.log(`✅ index.html updated with script: <script src="${newScriptPath}"></script>`);
    
    // 6. Update service worker to exclude index.html from caching
    const swPath = path.join(DIST_DIR, 'sw.js');
    let swContent = await fs.readFile(swPath, 'utf-8');
    
    // Remove index.html from precache list if it's there
    swContent = swContent.replace(/['"]\/index\.html['"],?\s*/g, '');
    
    // Add network-first strategy for index.html
    const networkFirstCode = `
// Network-first strategy for index.html to prevent stale versions
self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .catch(() => caches.match(event.request))
        );
        return;
    }
    
    // For other requests, use the existing cache-first strategy
    // ... existing fetch handler code ...
});`;
    
    // Replace the existing fetch handler with our enhanced version
    const existingFetchHandler = `self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});`;
    
    const newFetchHandler = `// Network-first strategy for navigation requests to prevent stale HTML
self.addEventListener('fetch', event => {
    // For navigation requests (page loads), try network first
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .catch(() => caches.match(event.request))
        );
        return;
    }
    
    // For other requests (assets), use cache-first strategy
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});`;
    
    swContent = swContent.replace(existingFetchHandler, newFetchHandler);
    
    await fs.writeFile(swPath, swContent);
    console.log("✅ Service worker updated with network-first strategy for index.html");
    
    console.log("🎉 Build complete!");
}

main();
