// Generates dist/methodology.html from docs/methodology.md at build time, so the
// published page is always the document currently in the repository.
//
// Only the METHOD chapters are published: the editing contract and chapter 3 (the
// mapping onto this repository) are stripped. Every strip rule asserts that it
// matched — a renamed heading fails the build rather than silently publishing the
// section it names — and what survives is checked against the source's own scope
// rule that chapters 1 and 2 name no code and never refer forward to chapter 3, so
// a leak reaches a build error instead of a reader with no repository to check it
// against.

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { marked } from 'marked';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SOURCE = path.resolve(HERE, '..', '..', 'docs', 'methodology.md');
const OUT_NAME = 'methodology.html';

// Heading of a section to drop, from the heading itself to the next heading of any
// level. Matched literally against a whole line.
const DROP_SECTIONS = [
    '## Scope — read this before editing',
];

// Heading from which everything to the end of the file is dropped.
const DROP_FROM = '# Chapter 3 — Project glue';

// Whole lines to drop. These are sentences that read correctly in the repository
// document and point at something the published page does not carry.
const DROP_LINES = [
    'Chapters 1 and 2 describe the method. Chapter 3 maps it onto the actual modules, artifacts and constants.',
];

function fail(message) {
    throw new Error(
        `methodology-to-html: ${message}\n` +
        `  Source: ${SOURCE}\n` +
        `  The published page is generated from that file; fix it there.`,
    );
}

// Fence-aware line classification, so a '#' or a filename inside a display formula
// is never mistaken for a heading or a code reference.
function fenceMask(lines) {
    const inFence = new Array(lines.length).fill(false);
    let open = false;
    for (let i = 0; i < lines.length; i++) {
        if (/^\s*```/.test(lines[i])) {
            inFence[i] = true;
            open = !open;
            continue;
        }
        inFence[i] = open;
    }
    if (open) fail('unterminated code fence');
    return inFence;
}

function findUniqueLine(lines, needle, what) {
    const mask = fenceMask(lines);
    const hits = [];
    for (let i = 0; i < lines.length; i++) {
        if (!mask[i] && lines[i] === needle) hits.push(i);
    }
    if (hits.length === 0) fail(`${what} not found: "${needle}"`);
    if (hits.length > 1) fail(`${what} is ambiguous (${hits.length} matches): "${needle}"`);
    return hits[0];
}

function stripSection(lines, heading) {
    const start = findUniqueLine(lines, heading, 'section heading');
    const mask = fenceMask(lines);
    let end = start + 1;
    while (end < lines.length && !(!mask[end] && /^#{1,6} /.test(lines[end]))) end++;
    return [...lines.slice(0, start), ...lines.slice(end)];
}

function stripFrom(lines, heading) {
    return lines.slice(0, findUniqueLine(lines, heading, 'section heading'));
}

function stripLine(lines, text) {
    const at = findUniqueLine(lines, text, 'line');
    return [...lines.slice(0, at), ...lines.slice(at + 1)];
}

// The scope rules, enforced. Each entry reports every offending line rather than
// only the first, so one build run surfaces the whole cleanup.
const SCOPE_CHECKS = [
    {
        what: 'a forward reference into chapter 3',
        test: /\b(sections?\s+3\.\d|chapter\s+3)\b/i,
    },
    {
        what: 'a file or module name',
        test: /[\w/.-]+\.(js|mjs|py|json|npz|csv|sql|cfg|sgf|txt)\b|\b(pipeline|utils|auxiliary)\.[a-z_]+/,
    },
    {
        what: 'a constant identifier',
        test: /`[A-Z][A-Z0-9]*(_[A-Z0-9]+)+`/,
    },
];

function checkScope(lines) {
    const mask = fenceMask(lines);
    const problems = [];
    for (let i = 0; i < lines.length; i++) {
        if (mask[i]) continue;
        for (const check of SCOPE_CHECKS) {
            const m = lines[i].match(check.test);
            if (m) problems.push(`  line ${i + 1}: ${check.what} — ${m[0]}`);
        }
    }
    if (problems.length) {
        fail(
            'the method chapters must name no code and never refer forward to ' +
            `chapter 3 (see the scope section):\n${problems.join('\n')}`,
        );
    }
}

function slugger() {
    const seen = new Map();
    return (text) => {
        const base = text
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '') || 'section';
        const n = (seen.get(base) ?? 0) + 1;
        seen.set(base, n);
        return n === 1 ? base : `${base}-${n}`;
    };
}

// The method chapters cross-reference each other constantly ("section 1.10", "sections
// 2.8, 2.10"). On a page those are worth following, so turn each into an anchor link.
// A number with no matching heading is left as plain text rather than linked to
// nowhere — the scope check has already ruled out the one case that matters, a
// reference into the chapter that is not published.
function linkSectionReferences(html, toc) {
    const ids = new Map();
    for (const h of toc) {
        const m = h.text.match(/^(\d+\.\d+)\s/);
        if (m) ids.set(m[1], h.id);
    }
    const link = (n) => (ids.has(n) ? `<a href="#${ids.get(n)}">${n}</a>` : n);

    // Display formulas are left alone; every reference lives in prose.
    return html.split(/(<pre>[\s\S]*?<\/pre>)/).map((chunk, i) => {
        if (i % 2 === 1) return chunk;
        let out = chunk.replace(/\b([Ss]ections?\s+)(\d+\.\d+)/g, (_m, lead, n) => lead + link(n));
        // Trailing members of a list — "sections 2.8, 2.10" — hang off the first link.
        for (let pass = 0; pass < 3; pass++) {
            out = out.replace(/(<\/a>)((?:,|\s+and)\s+)(\d+\.\d+)/g, (_m, close, sep, n) => close + sep + link(n));
        }
        return out;
    }).join('');
}

const PAGE_STYLE = `
        /* Palette mirrors the main app (style.css): dark by default with a
           prefers-color-scheme: light override. Kept in sync with about.html /
           changelog.html / privacy.html. */
        :root {
            --bg: #222;
            --text: #ddd;
            --heading: #fff;
            --accent: #d2691e;
            --muted: #888;
            --rule: #444;
            --code-bg: rgba(255, 255, 255, 0.08);
            --code-text: #eee;
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg: #f0f0f0;
                --text: #333;
                --heading: #222;
                --accent: #c25e16;
                --muted: #777;
                --rule: #ccc;
                --code-bg: rgba(0, 0, 0, 0.06);
                --code-text: #222;
            }
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: var(--bg);
        }

        h1 {
            color: var(--heading);
            text-align: center;
            border-bottom: 2px solid var(--accent);
            padding-bottom: 10px;
        }

        h1.chapter {
            text-align: left;
            border-bottom: none;
            margin-top: 2.5em;
            font-size: 1.6em;
        }

        h2 {
            color: var(--heading);
            margin-top: 2em;
            font-size: 1.2em;
            border-bottom: 1px solid var(--rule);
            padding-bottom: 6px;
        }

        strong { color: var(--heading); }

        a {
            color: var(--accent);
            text-decoration: none;
        }

        a:hover { text-decoration: underline; }

        code {
            background-color: var(--code-bg);
            color: var(--code-text);
            padding: 0.1em 0.4em;
            border-radius: 4px;
            font-size: 0.9em;
        }

        /* Display formulas and diagrams: they carry meaningful alignment, so they
           scroll sideways rather than wrap. */
        pre {
            background-color: var(--code-bg);
            color: var(--code-text);
            padding: 12px 14px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.85em;
            line-height: 1.45;
        }

        pre code {
            background: none;
            padding: 0;
            font-size: inherit;
        }

        .table-wrap { overflow-x: auto; }

        table {
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 0.92em;
        }

        th, td {
            border: 1px solid var(--rule);
            padding: 6px 10px;
            text-align: left;
            vertical-align: top;
        }

        th { color: var(--heading); }

        hr {
            border: none;
            border-top: 1px solid var(--rule);
            margin: 2.5em 0;
        }

        blockquote {
            border-left: 3px solid var(--rule);
            margin-left: 0;
            padding-left: 1em;
            color: var(--muted);
        }

        .toc {
            border: 1px solid var(--rule);
            border-radius: 6px;
            padding: 14px 20px;
            margin: 2em 0;
        }

        .toc h2 {
            margin: 0 0 8px;
            border: none;
            padding: 0;
            font-size: 1em;
        }

        .toc ul {
            list-style: none;
            margin: 0;
            padding: 0;
        }

        .toc li { margin: 3px 0; }
        .toc li.section { padding-left: 1.2em; font-size: 0.95em; }
        .toc li.chapter { margin-top: 10px; font-weight: bold; }

        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--rule);
            color: var(--muted);
        }
`;

function renderPage({ title, toc, body }) {
    const tocItems = toc.map(h =>
        `            <li class="${h.depth === 1 ? 'chapter' : 'section'}"><a href="#${h.id}">${h.text}</a></li>`,
    ).join('\n');

    return `<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="dark light">
    <title>Methodology — onlymove.net</title>
    <meta name="description" content="How onlymove.net builds problem ratings, ranks and motifs, and how they are used.">
    <style>${PAGE_STYLE}    </style>
</head>

<body>
    <!-- GENERATED at build time from docs/methodology.md by scripts/methodology-to-html.mjs.
         Do not edit: every build overwrites it. The page is English-only, like the
         changelog. -->
    <h1>${title}</h1>

    <nav class="toc">
        <h2>Contents</h2>
        <ul>
${tocItems}
        </ul>
    </nav>

${body}
    <div class="footer">
        <p><a href="about.html">About onlymove.net</a> &middot; <a href="index.html">Back to the App</a></p>
    </div>

</body>

</html>
`;
}

export async function buildMethodologyPage(distDir) {
    const raw = await fs.readFile(SOURCE, 'utf-8');
    let lines = raw.split('\n');

    for (const heading of DROP_SECTIONS) lines = stripSection(lines, heading);
    lines = stripFrom(lines, DROP_FROM);
    for (const text of DROP_LINES) lines = stripLine(lines, text);

    checkScope(lines);

    // Trim the separator and blank lines the chapter-3 cut leaves dangling.
    while (lines.length && /^(\s*|-{3,})$/.test(lines[lines.length - 1])) lines.pop();

    const html = marked.parse(lines.join('\n'), { gfm: true });

    // Heading ids + table of contents. Post-processing the rendered HTML rather
    // than overriding a renderer keeps this independent of marked's renderer API.
    const slug = slugger();
    const toc = [];
    let title = 'Methodology';
    let seenTitle = false;

    let body = html.replace(/<h([12])>([\s\S]*?)<\/h\1>/g, (_m, depth, inner) => {
        const text = inner.replace(/<[^>]+>/g, '').trim();
        if (!seenTitle) {
            // The document's own title becomes the page title, not a TOC entry.
            seenTitle = true;
            title = text;
            return '';
        }
        const id = slug(text);
        toc.push({ depth: Number(depth), text, id });
        const cls = depth === '1' ? ' class="chapter"' : '';
        return `<h${depth} id="${id}"${cls}>${inner}</h${depth}>`;
    });

    if (!toc.length) fail('no headings survived the strip rules — check them');

    body = linkSectionReferences(body, toc);
    body = body.replace(/<table>[\s\S]*?<\/table>/g, t => `<div class="table-wrap">${t}</div>`);

    const out = path.join(distDir, OUT_NAME);
    await fs.writeFile(out, renderPage({ title, toc, body }));
    return { out, sections: toc.length };
}
