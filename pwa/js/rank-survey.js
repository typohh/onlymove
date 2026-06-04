// js/rank-survey.js
//
// Periodic rank-declaration prompt: once a user has a non-trivial answer
// history, ask which server-rated rank they currently hold so we can later
// calibrate the internal rating scale against external scales. Re-prompts
// monthly. The cooldown is committed ONLY when the user actually interacts
// with the dialog (submit or explicit dismiss). If the dialog gets blown
// away by a reload, an SW takeover, or a tab close, the next stable session
// brings it back.

import { state, getAnswerHistoryForSize } from './state.js';
import { getTerm } from './i18n.js';
import { APP_MODES } from './constants.js';
import { showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid } from './overlays.js';

// v2: the v1 key was committed-on-render, so users hit by the version-update
// race got their cooldown burned without ever seeing the dialog. Bumping the
// key gives them a fresh chance.
const STORAGE_KEY = 'rankSurveyShownAt_v2';
const COOLDOWN_MS = 30 * 24 * 3600 * 1000; // ~1 month
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
// Counts setAcceptClicks(true) ticks since module load. The first tick is the
// initial board paint, when a queued maybeReload (X-App-Version mismatch) or
// an SW controllerchange would yank the page out from under any modal. We
// wait for tick ≥ 2 so any /report-driven version check has had a chance to
// land and either trigger the reload or clear.
let tickCount = 0;

export function maybeShowRankSurvey() {
    tickCount++;
    if (dialogOpen) return;
    if (state.appMode === APP_MODES.ANALYZING) return;
    // A pending reload is about to throw the dialog away — and worse, since
    // the cooldown is only set on user interaction, a "rendered but killed"
    // dialog would still come back next session. Save the user the flicker
    // and just skip until the reload settles.
    if (state.reloadPending) return;
    if (tickCount < 2) return;
    const lastShown = parseInt(localStorage.getItem(STORAGE_KEY) || '0', 10);
    if (Number.isFinite(lastShown) && Date.now() - lastShown < COOLDOWN_MS) return;
    if (totalAnswerCount() < MIN_ANSWERS) return;
    showRankSurvey();
}

function showRankSurvey() {
    if (dialogOpen) return;
    dialogOpen = true;
    // NB: the cooldown is committed in the submit/dismiss handlers below, not
    // here. If the page reloads while the dialog is open the user gets to see
    // it again next session — the cooldown should reflect user choice, not
    // render time.

    const noneLabel = getTerm('rankSurveyNone');
    const rankOptions = buildRankOptions();
    const selects = {};

    let modalApi;
    // Commit the cooldown only here, on actual user click. If the user never
    // interacts (page reload, tab close, SW takeover), the dialog comes back
    // next stable session.
    const closeWithCooldown = () => {
        localStorage.setItem(STORAGE_KEY, String(Date.now()));
        modalApi.close();
        dialogOpen = false;
    };

    modalApi = showModal({
        // No backdrop dismissal: the cooldown must reflect an explicit choice.
        dismissable: false,
        build: (modal) => {
            modal.appendChild(modalTitle(getTerm('rankSurveyTitle')));

            const grid = modalRankGrid();

            for (const server of SERVERS) {
                const label = document.createElement('label');
                label.textContent = server;
                label.className = 'modal-rank-label';

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

            const submit = () => {
                const ranks = {};
                for (const server of SERVERS) {
                    const v = selects[server].value;
                    if (v) ranks[server] = v;
                }
                // Fire and forget. Cooldown is committed regardless — a dropped
                // network submission is the user's choice to lose, not a reason
                // to re-prompt every session until it goes through.
                if (Object.keys(ranks).length > 0) {
                    fetch('/rank-survey', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ userId: state.userId, ranks }),
                    }).catch(err => console.warn('Rank survey upload failed:', err));
                }
                closeWithCooldown();
            };

            modal.appendChild(modalButtonRow([
                modalButton(getTerm('rankSurveyDismiss'), closeWithCooldown),
                modalButton(getTerm('rankSurveySubmit'), submit),
            ]));
        },
    });
}
