// js/rank-survey-prompt.js
//
// Periodic rank-declaration prompt: once a user has a non-trivial answer
// history, ask which server-rated rank they currently hold so we can later
// calibrate the internal rating scale against external scales. Re-prompts
// monthly. The cooldown is committed ONLY when the user actually interacts
// with the dialog (submit or explicit dismiss). If the dialog gets blown
// away by a reload, an SW takeover, or a tab close, the next stable session
// brings it back.
//
// A declaration is also SHOWN — it is what the /leaderboard page's Declared
// column reads for every server without a linked federation account — so the
// dialog states what the answers are for and carries the opt-out at the bottom.
// That checkbox is the SAME hidden_profiles flag as the one in the Account dialog
// (GET /profile to read, POST /profile/hide to write), not a second preference: it
// is read back on open so an existing opt-out shows as already checked, and written
// only when the user actually changes it, so submitting a survey can never silently
// un-hide someone who opted out in Account.

import { state, getAnswerHistoryForSize } from './state.js';
import { getTerm } from './i18n.js';
import { APP_MODES } from './constants.js';
import { showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid } from './overlays.js';
import { postJson } from './worker-post.js';

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

const isMobile = window.matchMedia('(pointer: coarse)').matches || window.innerWidth <= 768;

let activePopover = null;
const closeActivePopover = () => {
    if (activePopover) {
        activePopover.remove();
        activePopover = null;
    }
};

const handleDocumentClick = () => {
    closeActivePopover();
};

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
    const selectedRanks = {};
    for (const server of SERVERS) {
        selectedRanks[server] = '';
    }

    // The leaderboard opt-out, reflected from the server on open (see the module
    // note) and written back on close only if the user moved it.
    const hideCheckbox = document.createElement('input');
    hideCheckbox.type = 'checkbox';
    let hiddenAtOpen = false;
    if (state.userId) {
        fetch(`/profile?userId=${encodeURIComponent(state.userId)}`)
            .then(r => (r.ok ? r.json() : null))
            .then(j => {
                if (j && typeof j.hidden === 'boolean') {
                    hiddenAtOpen = j.hidden;
                    hideCheckbox.checked = j.hidden;
                }
            })
            .catch(() => {});
    }

    let modalApi;
    // Commit the cooldown only here, on actual user click. If the user never
    // interacts (page reload, tab close, SW takeover), the dialog comes back
    // next stable session.
    const closeWithCooldown = () => {
        localStorage.setItem(STORAGE_KEY, String(Date.now()));
        // Fire and forget, like the declarations themselves. Applied on BOTH
        // buttons: the visibility choice is independent of whether the user
        // submitted any ranks this time.
        if (state.userId && hideCheckbox.checked !== hiddenAtOpen) {
            postJson('/profile/hide', { userId: state.userId, hidden: hideCheckbox.checked })
                .catch(err => console.warn('Profile visibility update failed:', err));
        }
        document.removeEventListener('click', handleDocumentClick);
        closeActivePopover();
        modalApi.close();
        dialogOpen = false;
    };

    modalApi = showModal({
        // No backdrop dismissal: the cooldown must reflect an explicit choice.
        dismissable: false,
        build: (modal) => {
            modal.appendChild(modalTitle(getTerm('rankSurveyTitle')));

            // What the answers are for, under the header. Small + muted so it
            // reads as a note rather than a second heading (same treatment as
            // the rank-translation modal's caption). Muted by opacity only, so
            // it works on both the dark and light modal backgrounds.
            const intro = document.createElement('div');
            intro.textContent = getTerm('rankSurveyIntro');
            intro.className = 'modal-note';
            modal.appendChild(intro);

            const grid = modalRankGrid();

            // Timeout to prevent the current opening click event from triggering the close handler
            setTimeout(() => {
                document.addEventListener('click', handleDocumentClick);
            }, 0);

            for (const server of SERVERS) {
                const label = document.createElement('label');
                label.textContent = server;
                label.className = 'modal-rank-label';

                if (isMobile) {
                    const select = document.createElement('select');
                    select.className = 'survey-select';

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
                } else {
                    // Desktop Custom Popover Dropdown Button
                    const btn = document.createElement('button');
                    btn.className = 'survey-btn';

                    const textSpan = document.createElement('span');
                    textSpan.textContent = noneLabel;
                    const arrowSpan = document.createElement('span');
                    arrowSpan.textContent = '▼';
                    arrowSpan.className = 'survey-btn-arrow';
                    btn.appendChild(textSpan);
                    btn.appendChild(arrowSpan);

                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();

                        if (activePopover && activePopover.dataset.server === server) {
                            closeActivePopover();
                            return;
                        }
                        closeActivePopover();

                        // Create centered card overlay inside a dark backdrop
                        const backdrop = document.createElement('div');
                        backdrop.className = 'survey-popover-backdrop';
                        backdrop.dataset.server = server;

                        const picker = document.createElement('div');
                        picker.className = 'survey-popover';
                        // The popover hangs off <body>, not the modal, so it
                        // takes the modal's type size explicitly.
                        picker.style.fontSize = window.getComputedStyle(modal).fontSize;

                        const header = document.createElement('div');
                        header.textContent = server;
                        header.className = 'survey-popover-header';
                        picker.appendChild(header);

                        const list = document.createElement('div');
                        list.className = 'survey-popover-list';

                        const addOption = (val, labelText) => {
                            const opt = document.createElement('div');
                            opt.textContent = labelText;
                            opt.className = 'survey-option';
                            if (selectedRanks[server] === val) opt.classList.add('selected');

                            opt.addEventListener('click', (ev) => {
                                ev.stopPropagation();
                                selectedRanks[server] = val;
                                textSpan.textContent = labelText;
                                closeActivePopover();
                            });

                            list.appendChild(opt);
                        };

                        addOption('', noneLabel);
                        for (const r of rankOptions) {
                            addOption(r, r);
                        }

                        picker.appendChild(list);
                        backdrop.appendChild(picker);

                        backdrop.addEventListener('click', (ev) => {
                            if (ev.target === backdrop) {
                                closeActivePopover();
                            }
                        });

                        document.body.appendChild(backdrop);
                        activePopover = backdrop;

                        // Scroll selected rank into view
                        setTimeout(() => {
                            const targetText = selectedRanks[server] || noneLabel;
                            const children = Array.from(list.children);
                            const match = children.find(c => c.textContent === targetText);
                            if (match) {
                                match.scrollIntoView({ block: 'nearest' });
                            }
                        }, 0);
                    });

                    grid.appendChild(label);
                    grid.appendChild(btn);
                }
            }
            modal.appendChild(grid);

            const submit = () => {
                const ranks = {};
                for (const server of SERVERS) {
                    const v = isMobile ? selects[server].value : selectedRanks[server];
                    if (v) ranks[server] = v;
                }
                // Fire and forget. Cooldown is committed regardless — a dropped
                // network submission is the user's choice to lose, not a reason
                // to re-prompt every session until it goes through.
                if (Object.keys(ranks).length > 0) {
                    postJson('/rank-survey', { userId: state.userId, ranks })
                        .catch(err => console.warn('Rank survey upload failed:', err));
                }
                closeWithCooldown();
            };

            const hideRow = document.createElement('label');
            hideRow.style.display = 'flex';
            hideRow.style.alignItems = 'flex-start';
            hideRow.style.gap = '0.3571em';
            hideRow.style.fontSize = '0.8em';
            hideRow.style.opacity = '0.7';
            hideRow.style.textAlign = 'left';
            hideRow.style.maxWidth = '30em';
            hideRow.style.cursor = 'pointer';
            hideCheckbox.style.flex = 'none';
            hideCheckbox.style.marginTop = '0.25em';
            const hideText = document.createElement('span');
            hideText.textContent = getTerm('rankSurveyHide');
            hideRow.appendChild(hideCheckbox);
            hideRow.appendChild(hideText);
            modal.appendChild(hideRow);

            modal.appendChild(modalButtonRow([
                modalButton(getTerm('rankSurveyDismiss'), closeWithCooldown),
                modalButton(getTerm('rankSurveySubmit'), submit),
            ]));
        },
    });
}

