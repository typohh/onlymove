// js/debug-log.js
//
// Captures console messages into a ring buffer and exposes a hidden viewer
// reachable by tapping the page title 5 times within 2 seconds. Useful when
// a user hits an intermittent issue on mobile (no devtools) and we need to
// see what happened.

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
    const clearBtn = document.createElement('button');
    clearBtn.textContent = 'Clear';
    Object.assign(clearBtn.style, { marginRight: '10px', padding: '6px 12px', cursor: 'pointer' });
    clearBtn.onclick = () => { buffer.length = 0; overlay.remove(); };
    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    Object.assign(closeBtn.style, { padding: '6px 12px', cursor: 'pointer' });
    closeBtn.onclick = () => overlay.remove();
    buttons.appendChild(copyBtn);
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
    body.textContent = buffer.length ? buffer.join('\n') : '(no messages yet)';

    overlay.appendChild(header);
    overlay.appendChild(body);
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
