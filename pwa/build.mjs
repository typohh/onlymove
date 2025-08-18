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
	'index.html', 'style.css', 'sound', 'lib', 'tsumegos', 'metadata.json', 'manifest.json', 'sw.js',
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
    console.log("🎉 Build complete!");
}

main();
