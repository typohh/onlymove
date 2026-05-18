// js/rank-survey.js
//
// Periodic rank-declaration prompt: once a user has a non-trivial answer
// history, ask which server-rated rank they currently hold so we can later
// calibrate the internal rating scale against external scales. Shown at most
// once every two months; dismissing counts as a "show" (the cooldown starts
// the moment the dialog renders, regardless of how the user closes it).

import { state, getAnswerHistoryForSize } from './state.js';
import { getTerm } from './i18n.js';
import { APP_MODES } from './constants.js';

const STORAGE_KEY = 'rankSurveyShownAt';
const COOLDOWN_MS = 60 * 24 * 3600 * 1000; // ~2 months
const MIN_ANSWERS = 100;

const SERVERS = ['OGS', 'KGS', 'Fox', 'Tygem', 'WBaduk', 'IGS', 'GoQuest', 'EGF', 'AGA'];

function buildRankOptions() {
    const ranks = [];
    for (let k = 25; k >= 1; k--) ranks.push(`${k}k`);
    for (let d = 1; d <= 9; d++) ranks.push(`${d}d`);
    return ranks;
}

function totalAnswerCount() {
    return getAnswerHistoryForSize(9).length + getAnswerHistoryForSize(13).length;
}

let dialogOpen = false;

export function maybeShowRankSurvey() {
    if (dialogOpen) return;
    if (state.appMode === APP_MODES.ANALYZING) return;
    const lastShown = parseInt(localStorage.getItem(STORAGE_KEY) || '0', 10);
    if (Number.isFinite(lastShown) && Date.now() - lastShown < COOLDOWN_MS) return;
    if (totalAnswerCount() < MIN_ANSWERS) return;
    showRankSurvey();
}

function showRankSurvey() {
    if (dialogOpen) return;
    dialogOpen = true;
    // Lock in the cooldown the moment the dialog renders, so any kind of
    // dismissal (close, navigate away, refresh) still counts as "shown".
    localStorage.setItem(STORAGE_KEY, String(Date.now()));

    const overlay = document.createElement('div');
    overlay.id = 'ios-install-overlay';

    const modal = document.createElement('div');
    modal.id = 'ios-install-modal';

    const title = document.createElement('div');
    title.id = 'ios-install-title';
    title.textContent = getTerm('rankSurveyTitle');
    modal.appendChild(title);

    const grid = document.createElement('div');
    grid.style.display = 'grid';
    grid.style.gridTemplateColumns = 'max-content max-content';
    grid.style.justifyContent = 'center';
    grid.style.gap = '0.8vmin 1.5vmin';
    grid.style.alignItems = 'center';
    grid.style.maxHeight = '60vh';
    grid.style.overflowY = 'auto';

    const noneLabel = getTerm('rankSurveyNone');
    const rankOptions = buildRankOptions();
    const selects = {};
    for (const server of SERVERS) {
        const label = document.createElement('label');
        label.textContent = server;
        label.style.fontWeight = 'bold';
        label.style.textAlign = 'right';

        const select = document.createElement('select');
        select.style.padding = '0.5vmin 0.8vmin';
        select.style.borderRadius = 'var(--radius)';
        select.style.background = '#444';
        select.style.color = '#fff';
        select.style.border = '0.2vmin solid #555';
        select.style.fontSize = 'inherit';
        select.style.width = '8em';
        select.style.boxSizing = 'border-box';

        const blank = document.createElement('option');
        blank.value = '';
        blank.textContent = noneLabel;
        select.appendChild(blank);
        for (const r of rankOptions) {
            const opt = document.createElement('option');
            opt.value = r;
            opt.textContent = r;
            select.appendChild(opt);
        }

        const id = `rank-survey-${server.toLowerCase()}`;
        select.id = id;
        label.htmlFor = id;

        selects[server] = select;
        grid.appendChild(label);
        grid.appendChild(select);
    }
    modal.appendChild(grid);

    const buttonRow = document.createElement('div');
    buttonRow.style.display = 'flex';
    buttonRow.style.gap = '1.5vmin';
    buttonRow.style.justifyContent = 'center';
    buttonRow.style.marginTop = '0.5vmin';

    const styleBtn = (btn) => {
        // .btn brings flex: 1, vmin padding, and width: 100% in the landscape
        // media query (used by the side panel). Reset all three so these
        // dialog buttons size to content and stay inside the modal.
        btn.style.flex = '0 0 auto';
        btn.style.width = 'auto';
        btn.style.padding = '0.6vmin 2vmin';
        btn.style.borderRadius = 'var(--radius)';
        btn.style.minWidth = '6em';
    };

    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'btn';
    dismissBtn.textContent = getTerm('rankSurveyDismiss');
    styleBtn(dismissBtn);

    const submitBtn = document.createElement('button');
    submitBtn.className = 'btn';
    submitBtn.textContent = getTerm('rankSurveySubmit');
    styleBtn(submitBtn);

    const close = () => {
        if (overlay.parentNode) overlay.parentNode.removeChild(overlay);
        dialogOpen = false;
    };

    dismissBtn.addEventListener('click', close);
    submitBtn.addEventListener('click', () => {
        const ranks = {};
        for (const server of SERVERS) {
            const v = selects[server].value;
            if (v) ranks[server] = v;
        }
        // Fire and forget. The cooldown is already set, so a network failure
        // just means we lose this submission — acceptable at current scale.
        if (Object.keys(ranks).length > 0) {
            fetch('/rank-survey', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userId: state.userId, ranks }),
            }).catch(err => console.warn('Rank survey upload failed:', err));
        }
        close();
    });

    buttonRow.appendChild(dismissBtn);
    buttonRow.appendChild(submitBtn);
    modal.appendChild(buttonRow);

    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}
