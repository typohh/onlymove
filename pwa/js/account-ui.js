// js/account-ui.js
//
// The Account dialog (hamburger menu → More). Three values, each one a thing the
// user can change by clicking it:
//
//     Code       <user-id>            (?)
//     Profile    Henry Hemming · 4d  (?)
//     Anonymous  No                  (?)
//
// The explanations live in hint bubbles rather than in the layout (see
// js/hint-bubble.js): every one of these needs a sentence, and three sentences
// on the page turn a short list of choices into something nobody reads. The
// values are styled alike because they behave alike — click to change — and the
// account code is not special among them.
//
// Two things this deliberately does NOT have. There is no Copy button: the code
// is selectable text and every platform already has a copy gesture for it. And
// there is no Log in button — a disabled button beside a field the user has not
// touched reads as "something is wrong", so the code field simply asks for
// confirmation when the user finishes editing it. Committing on blur (rather
// than on a button) is what makes "click the value to change it" true of all
// three rows instead of only two.
//
// Nothing here raises a toast. A success needs none — the row the user just
// changed is showing its new value, which is the report. A FAILURE does need
// saying, because a value that quietly snaps back looks like a bug rather than a
// refusal, so those go to a note line inside the dialog, where the user is
// already looking, instead of a banner over the board.
//
// Code is local. Profile (which EGD/AGA player this account is) and Anonymous
// (the leaderboard opt-out) are server state, read together in ONE
// GET /profile and written only on an actual change — the rank survey offers the
// same opt-out, so an unconditional write would let one dialog undo the other.
//
// The identity search runs in the BROWSER (js/identity-search.js): the Worker
// cannot reach EGD at all, so a client-side search is the only kind there is.
// Picking a result POSTs /link, which is where exclusivity is enforced — nothing
// anywhere verifies that the user IS that player, and the dialog does not pretend
// otherwise; what it offers instead is Unlink, so a wrong pick is the user's own
// to undo.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import {
    showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid,
    showConfirm,
} from './overlays.js';
import { isValidAccountCode, switchAccount, accountExists } from './account-sync.js';
import { postJson } from './worker-post.js';
import { searchIdentities } from './identity-search.js';
import { bufferGorUpdate } from './gor-refresh.js';
import { hintButton, closeHint } from './hint-bubble.js';
import { onAccountProfileShown } from './onboarding.js';

// Shown in place of a server value that hasn't arrived, and of one whose read
// failed — the row stays clickable in the second case, which retries.
const PENDING = '…';
const UNKNOWN = '—';

// POST /link rejections that name a cause the user can act on. Anything else —
// 'server', a network failure, an unrecognized body — falls to the generic retry
// line.
function linkErrorTerm(error) {
    switch (error) {
        case 'id_taken':    return 'accountLinkTaken';
        case 'unreachable': return 'accountLinkUnreachable';
        default:            return 'accountLinkFailed';
    }
}

export function showAccountModal() {
    showModal({
        // A hint bubble is a child of <body>, not of the modal, so a click on one
        // reaches the backdrop; without this the dialog would close under the user
        // as they read an explanation.
        onDismiss: closeHint,
        build: (modal, close) => {
            modal.appendChild(modalTitle(getTerm('accountTitle')));

            // The dialog is one of two views: the three-row overview, or the
            // identity search. One thing at a time — the search needs the width
            // and a result list, and interleaving it with the other two rows is
            // what made the previous version feel like a form.
            const host = document.createElement('div');
            modal.appendChild(host);

            // ONE button, always in the same place, meaning "leave this view":
            // Close from the overview, back to the overview from the search. A
            // Cancel beside a Close is two ways out of one dialog, and the reader
            // has to work out which is which.
            const footer = document.createElement('div');
            modal.appendChild(footer);

            new AccountPanel(host, footer, close).showOverview();
        },
    });
}

// One row of the overview: a bold label, a clickable value, and its hint.
// `cellId` is only for the Profile row: js/onboarding.js anchors its
// link-your-record bubble to that cell by id, so don't drop it.
function addRow(grid, labelTerm, hintTerm, valueEl, cellId) {
    const label = document.createElement('div');
    label.className = 'modal-rank-label';
    label.textContent = getTerm(labelTerm);
    grid.appendChild(label);

    const cell = document.createElement('div');
    cell.className = 'account-cell';
    if (cellId) cell.id = cellId;
    cell.appendChild(valueEl);
    cell.appendChild(hintButton(getTerm(hintTerm)));
    grid.appendChild(cell);
}

// A value the user can click to change. `muted` marks a value that isn't set
// yet, so "not set" doesn't read as a name.
// `onActivate` is handed its own element, so a handler that has to swap itself
// out (the code editor) works off the node it was actually attached to rather
// than re-finding it by position in the grid.
function valueEl(text, onActivate, { muted = false } = {}) {
    const el = document.createElement('span');
    el.className = muted ? 'account-value account-value-muted' : 'account-value';
    el.textContent = text;
    el.setAttribute('role', 'button');
    el.setAttribute('tabindex', '0');
    el.addEventListener('click', () => onActivate(el));
    el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onActivate(el); }
    });
    return el;
}

class AccountPanel {
    constructor(host, footer, closeModal) {
        this.host = host;
        this.footer = footer;
        this.closeModal = closeModal;
        this.linked = undefined;   // undefined = not read yet, null = none
        this.hidden = undefined;
        this.loadFailed = false;
        this.notice = '';          // the overview's one error line, '' when quiet
        this.load();
    }

    // Say why something did not happen, on the overview. Stays up until the next
    // action, so a user who looks away and back still finds out.
    fail(term) {
        this.notice = getTerm(term);
        this.showOverview();
    }

    // Profile + Anonymous in one read. Re-runnable: clicking a failed row calls
    // it again rather than making the user reopen the dialog. A retry that fails
    // says so — the boot read can fail quietly into a '—' the user may not have
    // been looking at, but a click they just made must answer them.
    async load({ announceFailure = false } = {}) {
        let profile = null;
        try {
            const res = await fetch(`/profile?userId=${encodeURIComponent(state.userId)}`);
            profile = res.ok ? await res.json() : null;
        } catch (_) {
            profile = null;
        }
        if (!profile) {
            this.loadFailed = true;
            if (announceFailure) this.notice = getTerm('accountLoadFailed');
        } else {
            this.loadFailed = false;
            this.linked = profile.linked || null;
            this.hidden = !!profile.hidden;
        }
        // Only repaint if the overview is still what's on screen — a load that
        // lands while the user is mid-search must not throw their typing away.
        if (this.view === 'overview') {
            this.showOverview();
            // Offer the onboarding bubble here rather than from showOverview:
            // the overview also repaints on an UNLINK, and "link your record"
            // arriving the moment the user chose Anonymous is the one time it
            // must not. This runs on opening the dialog (and on a retry), where
            // the answer is genuinely news.
            onAccountProfileShown(this.linked);
        }
    }

    render(build, exitTerm, onExit) {
        closeHint();
        this.host.textContent = '';
        this.footer.textContent = '';
        build(this.host);
        this.footer.appendChild(modalButtonRow([modalButton(getTerm(exitTerm), onExit)]));
    }

    // --- the three-row overview ---------------------------------------------
    showOverview() {
        this.view = 'overview';
        this.render((host) => {
            const grid = modalRankGrid();
            grid.classList.add('account-grid');
            addRow(grid, 'accountRowCode', 'accountHint', this.codeValue());
            addRow(grid, 'accountProfile', 'accountLinkIntro', this.profileValue(), 'account-profile-cell');
            addRow(grid, 'accountAnonymous', 'accountHideLabel', this.anonymousValue());
            host.appendChild(grid);
            const note = document.createElement('div');
            note.className = 'account-note';
            note.textContent = this.notice;
            host.appendChild(note);
        }, 'accountClose', () => { closeHint(); this.closeModal(); });
    }

    // Code: click swaps the value for an input; leaving the input is what asks.
    codeValue() {
        return valueEl(state.userId, (el) => this.editCode(el));
    }

    editCode(current) {
        this.notice = '';
        const cell = current.parentNode;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'account-value account-input';
        input.maxLength = 8;
        input.value = state.userId;
        input.autocapitalize = 'off';
        input.autocomplete = 'off';
        input.spellcheck = false;
        cell.replaceChild(input, current);
        input.focus();
        input.select();

        let settled = false;
        const revert = () => {
            if (settled) return;
            settled = true;
            if (input.parentNode === cell) cell.replaceChild(this.codeValue(), input);
        };
        // Blur is the commit: Enter blurs, Escape reverts first so the blur it
        // causes finds nothing left to do.
        input.addEventListener('blur', () => {
            const typed = input.value.trim();
            revert();
            this.commitCode(typed);
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
            if (e.key === 'Escape') { e.preventDefault(); input.value = state.userId; input.blur(); }
        });
    }

    async commitCode(typed) {
        if (!typed || typed === state.userId) return;
        if (!isValidAccountCode(typed)) {
            this.fail('accountInvalid');
            return;
        }
        // Only adopt codes that exist server-side (>= 1 report — the only state an
        // account can be rebuilt from), so a typo can't strand the user on an
        // empty account.
        if (await accountExists(typed) !== true) {
            this.fail('accountNotFound');
            return;
        }
        if (!await showConfirm(getTerm('accountSwitchConfirm', { code: typed }))) return;
        closeHint();
        this.closeModal();
        // Wipes local state, stores the new credential, reloads; init then pulls
        // the account's full history from the server.
        switchAccount(typed);
    }

    // Profile: the linked player's own name and official grade, and nothing else
    // — the grade is what tells two same-named players apart, while the member
    // id, having done its job in the search, would only be noise.
    profileValue() {
        if (this.loadFailed) return valueEl(UNKNOWN, () => this.retry(), { muted: true });
        if (this.linked === undefined) return valueEl(PENDING, () => {}, { muted: true });
        if (!this.linked) {
            return valueEl(getTerm('accountNotLinked'), () => this.showSearch(), { muted: true });
        }
        const text = [this.linked.name, this.linked.rank].filter(Boolean).join(' · ');
        return valueEl(text, () => this.showSearch());
    }

    anonymousValue() {
        if (this.loadFailed) return valueEl(UNKNOWN, () => this.retry(), { muted: true });
        if (this.hidden === undefined) return valueEl(PENDING, () => {}, { muted: true });
        return valueEl(getTerm(this.hidden ? 'accountYes' : 'accountNo'), () => this.toggleHidden());
    }

    retry() {
        this.loadFailed = false;
        this.linked = undefined;
        this.hidden = undefined;
        this.showOverview();
        this.load({ announceFailure: true });
    }

    // Optimistic, then reconciled against what the server actually reports: what
    // is shown is always what the server holds.
    async toggleHidden() {
        const want = !this.hidden;
        this.notice = '';
        this.hidden = want;
        this.showOverview();
        try {
            const res = await postJson('/profile/hide', { userId: state.userId, hidden: want });
            const body = await res.json();
            if (!body || body.hidden !== want) throw new Error('rejected');
        } catch (_) {
            this.hidden = !want;
            this.fail('accountHideFailed');
        }
    }

    // --- the identity search -------------------------------------------------
    showSearch() {
        this.view = 'search';
        this.notice = '';
        this.render((host) => {
            const row = document.createElement('div');
            row.className = 'account-searchrow';
            const query = document.createElement('input');
            query.type = 'search';
            query.className = 'account-search';
            query.placeholder = getTerm('accountLinkPlaceholder');
            query.autocapitalize = 'off';
            query.autocomplete = 'off';
            query.spellcheck = false;
            row.appendChild(query);

            const message = document.createElement('div');
            message.className = 'account-note';
            const results = document.createElement('div');
            results.className = 'account-results';

            // The list always ends with Anonymous — the answer for someone who is
            // not in either database, and the way to take a linked name back off
            // the board. It is one of the answers to "who are you", so it belongs
            // among them rather than as a separate destructive-looking button.
            const paint = (players) => {
                results.textContent = '';
                for (const p of players) results.appendChild(this.resultRow(p, message));
                results.appendChild(this.anonymousRow(message));
            };

            let searching = false;
            const run = async () => {
                if (searching) return;
                const q = query.value.trim();
                if (!q) return;
                searching = true;
                searchBtn.classList.add('disabled');
                results.textContent = '';
                message.textContent = getTerm('accountLinkSearching');
                let players = [];
                try { players = await searchIdentities(q); } catch (_) { players = []; }
                searching = false;
                searchBtn.classList.remove('disabled');
                message.textContent = players.length ? '' : getTerm('accountLinkNoResults');
                paint(players);
            };
            const searchBtn = modalButton(getTerm('accountLinkSearch'), run);
            searchBtn.classList.add('account-search-btn');
            row.appendChild(searchBtn);
            query.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); run(); }
            });

            host.appendChild(row);
            host.appendChild(message);
            host.appendChild(results);
            paint([]);
            query.focus();
        }, 'accountCancel', () => this.showOverview());
    }

    // A clickable list line: the label, an optional detail, and what picking it
    // does. No button inside — the line IS the control, which is both a bigger
    // touch target and one fewer thing to explain.
    static row(label, detail, onPick, { muted = false } = {}) {
        const el = document.createElement('div');
        el.className = muted ? 'account-result account-result-muted' : 'account-result';
        el.setAttribute('role', 'button');
        el.setAttribute('tabindex', '0');

        const name = document.createElement('span');
        name.className = 'account-result-name';
        name.textContent = label;
        el.appendChild(name);

        if (detail) {
            const meta = document.createElement('span');
            meta.className = 'account-result-meta';
            meta.textContent = detail;
            el.appendChild(meta);
        }

        const pick = () => {
            if (el.classList.contains('is-busy')) return;
            onPick(el);
        };
        el.addEventListener('click', pick);
        el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
        });
        return el;
    }

    // "None of these" — unlinks if there is something to unlink, and otherwise
    // simply confirms what is already true and returns.
    anonymousRow(message) {
        return AccountPanel.row(getTerm('accountAnonymous'), '', async () => {
            if (!this.linked) { this.showOverview(); return; }
            await this.unlink(message);
        }, { muted: true });
    }

    // One search hit: who they are per their own database, and the pick that
    // claims them. An EGD row also hands its already-fetched GoR to the piggyback
    // buffer, so the account's first verified rating rides the next /report
    // instead of waiting for some other client to be asked to fetch it.
    resultRow(player, message) {
        // The database's own grade and locality — what tells two people of the
        // same name apart, which is this detail's whole job.
        const detail = [player.service, player.rank, player.loc].filter(Boolean).join(' · ');
        return AccountPanel.row(player.name, detail, async (el) => {
            // The same question the Anonymous line asks, for the same reason: this
            // is a claim to be somebody, made from a list where two people can share
            // a name, and it is the one action here that is visible to strangers.
            if (!await showConfirm(getTerm('accountLinkConfirm', { name: player.name, detail }))) return;
            el.classList.add('is-busy');
            message.textContent = '';
            try {
                const res = await postJson('/link', {
                    userId: state.userId,
                    service: player.service,
                    id: player.id,
                    name: player.name,
                    country: player.country,
                });
                const body = await res.json();
                if (!body || !body.ok) {
                    message.textContent = getTerm(linkErrorTerm(body && body.error));
                    el.classList.remove('is-busy');
                    return;
                }
                if (player.egd && Number.isFinite(player.egd.gor)) bufferGorUpdate(player.egd);
                // The server answers with a grade only where it has a heartbeat to
                // read — AGA, which it fetched itself. An EGD link has none until
                // the GoR piggyback lands on the next /report, so this render uses
                // the grade the search just pulled out of EGD: the same fact from
                // the same source, one hop earlier.
                this.linked = { ...body.linked, rank: body.linked.rank || player.rank };
                this.showOverview();
            } catch (_) {
                message.textContent = getTerm('accountLinkFailed');
                el.classList.remove('is-busy');
            }
        });
    }

    async unlink(message) {
        if (!await showConfirm(getTerm('accountUnlinkConfirm', { name: this.linked.name }))) return;
        try {
            const res = await postJson('/unlink', { userId: state.userId });
            const body = await res.json();
            if (!body || !body.ok) throw new Error('rejected');
            this.linked = null;
            this.showOverview();
        } catch (_) {
            message.textContent = getTerm('accountLinkFailed');
        }
    }
}
