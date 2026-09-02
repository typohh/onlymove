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
// A document also states the dates its figures were measured between, in two
// `<span data-fig-measured="earliest|latest">` spans. Those are not typed either:
// each is the first or last UTC day over the ledger's per-key `measured.at`
// stamps of every figure THAT DOCUMENT quotes or guards. The same range is
// recomputed here (ablations/_figures.py `measured_range` is the authority) and
// a document that disagrees fails the build — the remedy is
// `python -m ablations --sync`, never editing the date.
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
const MEASURED_SPAN_RE = /<span data-fig-measured="(earliest|latest)">([^<]*)<\/span>/g;
const AT_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/;

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

// The UTC days bounding when the figures `text` depends on were measured. A key
// the ledger lacks is the span/guard checks' finding and is skipped here; a
// ledgered key with no usable stamp is reported, since it cannot be dated.
function measuredRange(rel, text, figures, problems) {
    const keys = new Set();
    for (const m of text.matchAll(SPAN_RE)) keys.add(m[1]);
    for (const m of text.matchAll(GUARD_RE)) {
        const g = m[1].match(GUARD_EXPR_RE);
        if (g) keys.add(g[3]);
    }
    const days = [];
    for (const key of keys) {
        const entry = figures[key];
        if (!entry) continue;
        const at = entry.measured?.at;
        if (typeof at !== 'string' || !AT_RE.test(at)) {
            problems.push(`${rel}: ${key} carries no measured.at stamp in the ledger`);
            continue;
        }
        days.push(at.slice(0, 10));
    }
    if (!days.length) {
        problems.push(`${rel}: depends on no ledgered figure, so it has no measurement dates`);
        return null;
    }
    days.sort();
    return { earliest: days[0], latest: days[days.length - 1] };
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
        const dated = [...text.matchAll(MEASURED_SPAN_RE)];
        if (dated.length) {
            const range = measuredRange(rel, text, figures, problems);
            for (const [, which, shown] of dated) {
                spans++;
                if (range && range[which] !== shown) {
                    problems.push(`${rel}: its figures were measured ${which} on ${range[which]}, the document says "${shown}"`);
                }
            }
        }
    }
    if (problems.length) {
        throw new Error(
            'figures-check: documents disagree with ablations/figures.json — ' +
            're-run `python -m ablations` (or `--sync`, or fix the sentence) rather than editing the number or the date:\n' +
            problems.map(p => `  ${p}`).join('\n'),
        );
    }
    return { spans, guards, documents: DOCUMENTS.length };
}
