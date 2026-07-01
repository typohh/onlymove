// js/debug-log.js
//
// Captures console messages into a ring buffer and exposes a hidden viewer
// reachable by tapping the page title 5 times within 2 seconds. Useful when
// a user hits an intermittent issue on mobile (no devtools) and we need to
// see what happened.

import { getTail } from './history-store.js';

const BUFFER_LIMIT = 300;
const TRIGGER_TAPS = 5;
const TRIGGER_WINDOW_MS = 2000;

const buffer = [];

function fmt(arg) {
    if (arg instanceof Error) return arg.message + '\n' + (arg.stack || '');
    if (typeof arg === 'string') return arg;
    try { return JSON.stringify(arg); } catch { return String(arg); }
}

function record(level, args) {
    const ts = new Date().toISOString().slice(11, 23);
    const line = `[${ts}] [${level}] ` + Array.from(args).map(fmt).join(' ');
    buffer.push(line);
    if (buffer.length > BUFFER_LIMIT) buffer.shift();
}

// Evaluates source code in a way that mirrors the browser console:
// expression-form input prints its value, statement-form input runs without
// trying to return anything. `await` is allowed at the top level by wrapping
// in an async IIFE. Result, if any, is appended to the ring buffer; thrown
// errors land as an `error` line so stacks are visible in the dialog.
async function runEval(src) {
    record('eval', ['> ' + src.replace(/\s+/g, ' ').slice(0, 200)]);
    let result;
    let ran = false;
    try {
        // Expression form first — preserves the return value the way the
        // Chrome console does for one-liners like `state.userRank`.
        const exprFn = new Function('return (async () => (\n' + src + '\n))();');
        result = await exprFn();
        ran = true;
    } catch (e) {
        if (!(e instanceof SyntaxError)) {
            record('error', [e]);
            return;
        }
        // Statement form fallback for inputs that aren't a single expression
        // (declarations, control flow, multi-statement scripts).
        try {
            const stmtFn = new Function('return (async () => {\n' + src + '\n})();');
            result = await stmtFn();
            ran = true;
        } catch (e2) {
            record('error', [e2]);
            return;
        }
    }
    if (ran && result !== undefined) record('result', [result]);
}

function patchConsole() {
    for (const level of ['log', 'info', 'warn', 'error', 'debug']) {
        const orig = console[level];
        console[level] = function (...args) {
            record(level, args);
            return orig.apply(console, args);
        };
    }
    window.addEventListener('error', (e) => {
        record('uncaught', [e.message, e.error && e.error.stack]);
    });
    window.addEventListener('unhandledrejection', (e) => {
        const r = e.reason;
        record('rejection', [r && r.message ? r.message : r, r && r.stack]);
    });
}

// Hand a JSON string to the user as a file. On mobile we prefer the share
// sheet (navigator.share with a File) so it can go straight to another app;
// otherwise fall back to a normal blob download. Restores `btn`'s label after
// success; surfaces failures on the button. Shared by the LS and tail dumps.
async function shareOrDownloadJson(json, fname, btn, label) {
    try {
        try {
            const file = new File([json], fname, { type: 'application/json' });
            if (navigator.canShare && navigator.canShare({ files: [file] })) {
                await navigator.share({ files: [file], title: fname });
                btn.textContent = 'Shared';
                setTimeout(() => { btn.textContent = label; }, 1500);
                return;
            }
        } catch (_) { /* share unavailable/cancelled — fall through to download */ }
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fname;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        btn.textContent = 'Saved';
        setTimeout(() => { btn.textContent = label; }, 1500);
    } catch (e) {
        record('error', [e]);
        btn.textContent = 'Dump failed';
    }
}

function showDialog() {
    const existing = document.getElementById('debug-log-dialog');
    if (existing) { existing.remove(); return; }

    const overlay = document.createElement('div');
    overlay.id = 'debug-log-dialog';
    Object.assign(overlay.style, {
        position: 'fixed', top: '0', left: '0', width: '100%', height: '100%',
        backgroundColor: 'rgba(0,0,0,0.85)', zIndex: '10000',
        display: 'flex', flexDirection: 'column', padding: '20px',
        boxSizing: 'border-box', color: 'white', fontFamily: 'monospace',
    });

    const header = document.createElement('div');
    Object.assign(header.style, {
        display: 'flex', justifyContent: 'space-between',
        alignItems: 'center', marginBottom: '10px', flexShrink: '0',
    });
    const title = document.createElement('div');
    title.textContent = `Debug log (${buffer.length} messages)`;
    title.style.fontWeight = 'bold';
    const buttons = document.createElement('div');
    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copy';
    Object.assign(copyBtn.style, { marginRight: '10px', padding: '6px 12px', cursor: 'pointer' });
    copyBtn.onclick = async () => {
        const text = buffer.join('\n');
        try {
            await navigator.clipboard.writeText(text);
            copyBtn.textContent = 'Copied';
            setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        } catch {
            copyBtn.textContent = 'Copy failed';
        }
    };
    // Dump the entire localStorage as a JSON file (share sheet first, download
    // fallback — see shareOrDownloadJson).
    const dumpBtn = document.createElement('button');
    dumpBtn.textContent = 'Download LS';
    Object.assign(dumpBtn.style, { marginRight: '10px', padding: '6px 12px', cursor: 'pointer' });
    dumpBtn.onclick = () => {
        const dump = {};
        for (let i = 0; i < localStorage.length; i++) {
            const k = localStorage.key(i);
            dump[k] = localStorage.getItem(k);
        }
        const json = JSON.stringify(dump, null, 2);
        const uid = localStorage.getItem('userId') || 'unknown';
        const fname = `onlymove-ls-${uid}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
        shareOrDownloadJson(json, fname, dumpBtn, 'Download LS');
    };

    // Dump the IndexedDB answer-history cold tail (the deep history that
    // localStorage's working set doesn't hold) as a JSON array of entries.
    const tailBtn = document.createElement('button');
    tailBtn.textContent = 'Download Tail';
    Object.assign(tailBtn.style, { marginRight: '10px', padding: '6px 12px', cursor: 'pointer' });
    tailBtn.onclick = async () => {
        const prev = tailBtn.textContent;
        tailBtn.textContent = 'Reading…';
        try {
            const rows = await getTail();  // [] if IndexedDB unavailable/empty
            const json = JSON.stringify(rows);
            const uid = localStorage.getItem('userId') || 'unknown';
            const fname = `onlymove-tail-${uid}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
            await shareOrDownloadJson(json, fname, tailBtn, 'Download Tail');
        } catch (e) {
            record('error', [e]);
            tailBtn.textContent = 'Dump failed';
            setTimeout(() => { tailBtn.textContent = prev; }, 1500);
        }
    };

    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear';
    Object.assign(clearBtn.style, { marginRight: '10px', padding: '6px 12px', cursor: 'pointer' });
    clearBtn.onclick = () => { buffer.length = 0; overlay.remove(); };
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    Object.assign(closeBtn.style, { padding: '6px 12px', cursor: 'pointer' });
    closeBtn.onclick = () => overlay.remove();
    buttons.appendChild(copyBtn);
    buttons.appendChild(dumpBtn);
    buttons.appendChild(tailBtn);
    buttons.appendChild(clearBtn);
    buttons.appendChild(closeBtn);
    header.appendChild(title);
    header.appendChild(buttons);

    const body = document.createElement('pre');
    Object.assign(body.style, {
        flex: '1', overflow: 'auto', whiteSpace: 'pre-wrap',
        wordBreak: 'break-word', backgroundColor: '#111',
        padding: '10px', margin: '0', fontSize: '12px',
        border: '1px solid #444',
    });

    // Repaint the log body and title-count from the ring buffer. Called once
    // on open and again after every eval so newly-recorded lines (including
    // the eval's own input/output/errors) appear immediately.
    const renderBody = () => {
        body.textContent = buffer.length ? buffer.join('\n') : '(no messages yet)';
        title.textContent = `Debug log (${buffer.length} messages)`;
        body.scrollTop = body.scrollHeight;
    };
    renderBody();

    // REPL input row. The textarea is sized for one line by default but
    // grows naturally as the user types multi-line snippets. autocomplete /
    // autocorrect / autocapitalize are all off so phone keyboards don't
    // mangle JS syntax.
    const inputRow = document.createElement('div');
    Object.assign(inputRow.style, {
        display: 'flex', marginTop: '10px', flexShrink: '0', gap: '8px',
    });
    const input = document.createElement('textarea');
    input.rows = 1;
    input.placeholder = '> js expression or statements; Enter runs, Shift+Enter newlines';
    input.setAttribute('autocomplete', 'off');
    input.setAttribute('autocorrect', 'off');
    input.setAttribute('autocapitalize', 'off');
    input.setAttribute('spellcheck', 'false');
    Object.assign(input.style, {
        flex: '1', fontFamily: 'monospace', fontSize: '13px',
        padding: '8px', backgroundColor: '#222', color: 'white',
        border: '1px solid #444', resize: 'vertical', minHeight: '32px',
    });
    const runBtn = document.createElement('button');
    runBtn.textContent = 'Run';
    Object.assign(runBtn.style, {
        padding: '6px 14px', cursor: 'pointer', alignSelf: 'flex-start',
    });
    const doRun = async () => {
        const src = input.value;
        if (!src.trim()) return;
        runBtn.disabled = true;
        try { await runEval(src); } finally { runBtn.disabled = false; }
        renderBody();
        // Keep the source so the user can tweak and re-run without retyping.
        input.focus();
    };
    runBtn.onclick = doRun;
    input.addEventListener('keydown', (e) => {
        // Enter submits; Shift+Enter inserts a newline. Matches the muscle
        // memory of dev REPLs and avoids needing a separate Run tap on mobile.
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            doRun();
        }
    });
    inputRow.appendChild(input);
    inputRow.appendChild(runBtn);

    // Admin tools link (invite generator). Opens in a new tab so the app/debug
    // state isn't lost; the page verifies admin rights from this browser's account
    // code automatically and is inert for non-admins.
    const footer = document.createElement('div');
    Object.assign(footer.style, {
        marginTop: '10px', flexShrink: '0', fontSize: '12px', textAlign: 'right',
    });
    const adminLink = document.createElement('a');
    adminLink.textContent = 'Admin tools →';
    adminLink.href = '/admin/invite';
    adminLink.target = '_blank';
    adminLink.rel = 'noopener';
    Object.assign(adminLink.style, { color: '#8ab4f8', textDecoration: 'underline' });
    footer.appendChild(adminLink);

    overlay.appendChild(header);
    overlay.appendChild(body);
    overlay.appendChild(inputRow);
    overlay.appendChild(footer);
    document.body.appendChild(overlay);
}

function attachTrigger() {
    // Two title elements coexist for vertical / horizontal layouts; only one
    // is visible at a time. Attach to both so the gesture works either way.
    const targets = ['title', 'title-h']
        .map(id => document.getElementById(id))
        .filter(Boolean);
    if (targets.length === 0) return;
    let taps = 0;
    let firstTap = 0;
    const handler = () => {
        const now = Date.now();
        if (now - firstTap > TRIGGER_WINDOW_MS) {
            taps = 0;
            firstTap = now;
        }
        taps++;
        if (taps >= TRIGGER_TAPS) {
            taps = 0;
            showDialog();
        }
    };
    for (const t of targets) {
        t.style.cursor = 'pointer';
        t.addEventListener('click', handler);
    }
}

export function initDebugLog() {
    patchConsole();
    // Expose for manual triggering (e.g. desktop browser console).
    window.__showDebugLog = showDialog;
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attachTrigger);
    } else {
        attachTrigger();
    }
}
