// Build-time check that every figure a document quotes is the one the ledger holds.
//
// A measured number reaches about.html or docs/methodology.md only inside
// `<span data-fig="KEY">text</span>`, and `ablations/figures.json` (written by
// `python -m ablations`) is the record of what KEY was last measured at. This
// compares the two and fails the build on any disagreement, so a figure cannot be
// deployed that no producer backs and a re-measured figure cannot be deployed
// stale. It is a comparison only: the ledger stores the rendered `text` beside the
// raw value precisely so this needs no formatter of its own (the Python side is the
// single formatter — ablations/_figures.py).
//
// The two document paths and the span syntax are stated in ablations/_figures.py
// as well; a new document goes in both places.

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const LEDGER = path.join(ROOT, 'ablations', 'figures.json');
const DOCUMENTS = ['pwa/about.html', 'docs/methodology.md'];
const SPAN_RE = /<span data-fig="([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)">([^<]*)<\/span>/g;

// Claim guards: `<!-- fig-guard: EXPR -->` beside a sentence whose WORDING depends
// on a figure staying in range without quoting it. Same grammar as
// ablations/_figures.py (the authority): `KEY OP NUM` or `NUM OP KEY OP NUM`,
// evaluated against the ledger's raw value. A violated guard means the prose
// contradicts the last measurement, which no span rewriter can fix — fail the
// build so a person rewrites the sentence.
const GUARD_RE = /<!--\s*fig-guard:\s*(.*?)\s*-->/g;
const GUARD_EXPR_RE = new RegExp(
    '^(?:(-?\\d+(?:\\.\\d+)?)\\s*(==|!=|<=|>=|<|>)\\s*)?' +
    '([a-z][a-z0-9_]*(?:\\.[a-z][a-z0-9_]*)+)\\s*' +
    '(?:(==|!=|<=|>=|<|>)\\s*(-?\\d+(?:\\.\\d+)?))?$');
const OPS = {
    '==': (a, b) => a === b, '!=': (a, b) => a !== b,
    '<': (a, b) => a < b, '<=': (a, b) => a <= b,
    '>': (a, b) => a > b, '>=': (a, b) => a >= b,
};

function evalGuard(expr, figures) {
    const m = expr.match(GUARD_EXPR_RE);
    if (!m || (!m[2] && !m[4])) return `unparseable guard "${expr}"`;
    const [, lo, lop, key, rop, hi] = m;
    const entry = figures[key];
    if (!entry) return `guard "${expr}": ${key} has no ledger entry`;
    if (lop && !OPS[lop](Number(lo), entry.value)) return `guard "${expr}" violated: ${key} is ${entry.value}`;
    if (rop && !OPS[rop](entry.value, Number(hi))) return `guard "${expr}" violated: ${key} is ${entry.value}`;
    return null;
}

export async function checkFigures() {
    let ledger;
    try {
        ledger = JSON.parse(await fs.readFile(LEDGER, 'utf-8'));
    } catch (err) {
        throw new Error(`figures-check: cannot read ${LEDGER}: ${err.message}`);
    }
    const figures = ledger?.figures;
    if (!figures || typeof figures !== 'object') {
        throw new Error(`figures-check: ${LEDGER} has no "figures" object`);
    }

    const problems = [];
    let spans = 0;
    let guards = 0;
    for (const rel of DOCUMENTS) {
        const text = await fs.readFile(path.join(ROOT, rel), 'utf-8');
        for (const m of text.matchAll(SPAN_RE)) {
            spans++;
            const [, key, shown] = m;
            const entry = figures[key];
            let problem = null;
            if (!entry) problem = `${rel}: ${key} has no ledger entry`;
            else if (entry.text !== shown) problem = `${rel}: ${key} shows "${shown}", ledger says "${entry.text}"`;
            // One line per (document, key): about.html repeats each key per language block.
            if (problem && !problems.includes(problem)) problems.push(problem);
        }
        for (const m of text.matchAll(GUARD_RE)) {
            guards++;
            const problem = evalGuard(m[1], figures);
            if (problem) problems.push(`${rel}: ${problem} — rewrite the sentence beside it`);
        }
    }
    if (problems.length) {
        throw new Error(
            'figures-check: documents disagree with ablations/figures.json — ' +
            're-run `python -m ablations` (or fix the span/sentence) rather than editing the number:\n' +
            problems.map(p => `  ${p}`).join('\n'),
        );
    }
    return { spans, guards, documents: DOCUMENTS.length };
}
