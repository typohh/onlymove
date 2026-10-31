// js/account-ui.js
//
// The Account dialog (hamburger menu → More). Five values, each one a thing the
// user can change by clicking it:
//
//     Code       ab12cd34            (?)
//     Google     you@gmail.com       (?)
//     Profile    Henry Hemming · 4d  (?)
//     Alias      Brave Crimson Fox   (?)
//     Anonymous  No                  (?)
//
// Google sits under Code because it is ABOUT the code: linking one is how a
// device that does not have the code gets back to the account without the user
// having had to write it down. It is the only row that can be absent — with no
// client id configured (shared/google-auth.js) the whole feature is off and the
// row is not drawn, rather than drawn dead.
//
// The explanations live in hint bubbles rather than in the layout (see
// js/hint-bubble.js): every one of these needs a sentence, and four sentences
// on the page turn a short list of choices into something nobody reads. The
// values are styled alike because they behave alike — click to change — and the
// account code is special among them in one way only: it is a credential, so it
// is drawn MASKED until the eye beside it is pressed. The dialog is where the
// onboarding tip sends a new user, and a user who follows it while streaming or
// screen-sharing must not have published their account by doing so. The reveal
// lasts for this opening of the dialog and is never stored.
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
// Code is local. Profile (which EGD/AGA player this account is), Alias (the
// self-chosen leaderboard pseudonym) and Anonymous (the leaderboard opt-out) are
// server state, read together in ONE GET /profile and written only on an actual
// change — the rank survey offers the same opt-out, so an unconditional write
// would let one dialog undo the other.
//
// The alias picker offers its words from shared/alias-words.js, the same module
// the Worker renders the leaderboard from, so the choices work offline. What is
// POSTed — and what comes back — is the ENGLISH triplet; the preview and the
// overview row are the only places a translation is applied, and the language
// is resolved locally, so one stored alias reads correctly in all four.
//
// The identity search runs in the BROWSER (js/identity-search.js): the Worker
// cannot reach EGD at all, so a client-side search is the only kind there is.
// Picking a result POSTs /link, which is where exclusivity is enforced — nothing
// anywhere verifies that the user IS that player, and the dialog does not pretend
// otherwise; what it offers instead is Unlink, so a wrong pick is the user's own
// to undo.

import { state } from './state.js';
import { getTerm, getLang } from './i18n.js';
import { ALIAS_ADJECTIVES, ALIAS_COLORS, ALIAS_ANIMALS, composeAlias } from '../shared/alias-words.js';
import {
    showModal, modalTitle, modalButton, modalButtonRow, modalRankGrid,
    showConfirm,
} from './overlays.js';
import { isValidAccountCode, switchAccount, accountExists } from './account-sync.js';
import { postJson } from './worker-post.js';
import { searchIdentities } from './identity-search.js';
import { bufferGorUpdate } from './gor-refresh.js';
import { hintButton, closeHint } from './hint-bubble.js';
import {
    googleSignInAvailable, renderGoogleButton, googleAccountEmail,
    rememberGoogleEmail, readGoogleEmail, forgetGoogleEmail,
} from './google-signin.js';

// Shown in place of a server value that hasn't arrived, and of one whose read
// failed — the row stays clickable in the second case, which retries.
const PENDING = '…';
const UNKNOWN = '—';

// What stands in for the account code while it is hidden: one dot per character
// of a code, so the box does not change width when it is revealed.
const CODE_MASK = '•'.repeat(8);

const EYE_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="2.8"/></svg>';
const EYE_OFF_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z"/><circle cx="12" cy="12" r="2.8"/><path d="M4 4l16 16"/></svg>';

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

// POST /google/* rejections the user can act on. An invalid token is not among
// them: it means the token expired while the dialog sat open, or the client id
// and the Worker disagree — neither is something to explain mid-dialog, and both
// are answered by trying again.
function googleErrorTerm(error) {
    switch (error) {
        case 'google_taken':  return 'accountGoogleTaken';
        case 'account_taken': return 'accountGoogleAccountTaken';
        default:              return 'accountGoogleFailed';
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
// `extra` is a control that belongs to the value rather than to the row (the
// code's reveal toggle), so it sits between the value and the hint.
function addRow(grid, labelTerm, hintTerm, valueEl, extra = null) {
    const label = document.createElement('div');
    label.className = 'modal-rank-label';
    label.textContent = getTerm(labelTerm);
    grid.appendChild(label);

    const cell = document.createElement('div');
    cell.className = 'account-cell';
    cell.appendChild(valueEl);
    if (extra) cell.appendChild(extra);
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
        this.google = undefined;   // undefined = not read yet, bool = binding
        this.alias = undefined;    // undefined = not read yet, null = none
        this.loadFailed = false;
        this.notice = '';          // the overview's one error line, '' when quiet
        this.codeShown = false;    // the code is masked until asked for
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
            this.google = !!profile.google;
            this.alias = profile.alias || null;
            // The server holds the binding; this device holds the address that
            // names it. When the two disagree the server wins and the local name
            // goes — an unlink from another device is exactly that disagreement.
            if (!this.google) forgetGoogleEmail();
        }
        // Only repaint if the overview is still what's on screen — a load that
        // lands while the user is mid-search must not throw their typing away.
        if (this.view === 'overview') {
            this.showOverview();
        }
    }

    // `leadButtons` are laid out BEFORE the exit button, so a view that owns an
    // action (the alias picker's Save/Clear) puts it in the same row as Cancel
    // instead of growing a second button row inside the content.
    render(build, exitTerm, onExit, leadButtons = []) {
        closeHint();
        this.host.textContent = '';
        this.footer.textContent = '';
        build(this.host);
        this.footer.appendChild(modalButtonRow([...leadButtons, modalButton(getTerm(exitTerm), onExit)]));
    }

    // --- the overview --------------------------------------------------------
    showOverview() {
        this.view = 'overview';
        this.render((host) => {
            const grid = modalRankGrid();
            grid.classList.add('account-grid');
            addRow(grid, 'accountRowCode', 'accountHint', this.codeValue(), this.codeToggle());
            if (googleSignInAvailable()) {
                addRow(grid, 'accountRowGoogle', 'accountGoogleHint', this.googleValue());
            }
            addRow(grid, 'accountProfile', 'accountLinkIntro', this.profileValue());
            addRow(grid, 'accountAlias', 'accountAliasHint', this.aliasValue());
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
        return valueEl(this.codeShown ? state.userId : CODE_MASK, (el) => this.editCode(el));
    }

    // The eye beside the code. It repaints the value in place rather than the
    // whole overview, so a notice the user is reading stays where it is. An open
    // editor has already been swapped back by the blur this press caused, which
    // is why the value is found in the cell rather than held.
    codeToggle() {
        const btn = document.createElement('span');
        btn.className = 'account-eye';
        btn.setAttribute('role', 'button');
        btn.setAttribute('tabindex', '0');
        const paint = () => {
            btn.innerHTML = this.codeShown ? EYE_OFF_ICON : EYE_ICON;
            btn.setAttribute('aria-label', getTerm(this.codeShown ? 'accountCodeHide' : 'accountCodeShow'));
            btn.setAttribute('aria-pressed', String(this.codeShown));
        };
        const toggle = () => {
            this.codeShown = !this.codeShown;
            paint();
            const cell = btn.parentNode;
            if (cell) cell.replaceChild(this.codeValue(), cell.firstChild);
        };
        btn.addEventListener('click', toggle);
        btn.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
        });
        paint();
        return btn;
    }

    editCode(current) {
        this.notice = '';
        const cell = current.parentNode;
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'account-value account-input';
        input.maxLength = 8;
        // The editor starts from the current code only when that code is already
        // on screen; while it is masked the field opens empty, and leaving it
        // empty commits nothing.
        input.value = this.codeShown ? state.userId : '';
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
            if (e.key === 'Escape') { e.preventDefault(); input.value = ''; input.blur(); }
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

    // --- Google sign-in ------------------------------------------------------
    // Whether a Google account can reach this code, and the address that does —
    // when THIS device is the one that linked it. A device that arrived by
    // entering the code holds the binding but not the name, and says "Linked"
    // rather than guessing: the server keeps an opaque id and no address, so
    // there is nowhere else for a name to come from.
    googleValue() {
        if (this.loadFailed) return valueEl(UNKNOWN, () => this.retry(), { muted: true });
        if (this.google === undefined) return valueEl(PENDING, () => {}, { muted: true });
        if (!this.google) {
            return valueEl(getTerm('accountNotLinked'), () => this.showGoogleSignIn(), { muted: true });
        }
        const email = readGoogleEmail(state.userId);
        return valueEl(email || getTerm('accountGoogleLinked'), () => this.unlinkGoogle());
    }

    // The sign-in view: Google's own button and nothing else to decide. It makes
    // no claim about whether signing in is going to log the user in or attach the
    // account they pick to this one, because that is not knowable until the token
    // has been resolved against the server.
    showGoogleSignIn() {
        this.view = 'google';
        this.notice = '';
        this.render((host) => {
            const intro = document.createElement('div');
            intro.className = 'account-note';
            intro.textContent = getTerm('accountGoogleIntro');
            host.appendChild(intro);

            const buttonHost = document.createElement('div');
            buttonHost.className = 'account-google-button';
            host.appendChild(buttonHost);

            const message = document.createElement('div');
            message.className = 'account-note';
            message.textContent = PENDING;
            host.appendChild(message);

            renderGoogleButton(buttonHost, (credential) => {
                message.textContent = getTerm('accountGoogleChecking');
                this.resolveGoogle(credential, message);
            }).then(() => { message.textContent = ''; })
              .catch(() => { message.textContent = getTerm('accountGoogleUnavailable'); });
        }, 'accountCancel', () => this.showOverview());
    }

    // One verified token, two possible meanings — log in as the account it is
    // bound to, or bind it to the account in hand (src/handlers/google.js decides
    // which case it is, and writes nothing). Both branches ask first, and both
    // name the CODE in the question: the code is what is actually changing hands,
    // and a returning user whose device just minted a fresh one needs to see that
    // it is the wrong one before they attach their Google account to it.
    async resolveGoogle(credential, message) {
        let body = null;
        try {
            const res = await postJson('/google/signin', { credential, userId: state.userId });
            body = await res.json();
        } catch (_) {
            body = null;
        }
        if (!body || !body.ok) {
            message.textContent = getTerm(googleErrorTerm(body && body.error));
            return;
        }

        if (body.status === 'self') {
            this.google = true;
            this.showOverview();
            return;
        }

        if (body.status === 'login') {
            if (!isValidAccountCode(body.userId)) {
                message.textContent = getTerm('accountGoogleFailed');
                return;
            }
            message.textContent = '';
            if (!await showConfirm(getTerm('accountSwitchConfirm', { code: body.userId }))) return;
            // Stored against the code being ADOPTED, so it survives that account's
            // wipe and names the right account once the reload lands.
            rememberGoogleEmail(body.userId, googleAccountEmail(credential));
            closeHint();
            this.closeModal();
            switchAccount(body.userId);
            return;
        }

        // 'unbound' — the Google account is free, so the offer is to attach it.
        message.textContent = '';
        if (!await showConfirm(getTerm('accountGoogleLinkConfirm', { code: state.userId }))) return;
        message.textContent = getTerm('accountGoogleChecking');
        let linked = null;
        try {
            const res = await postJson('/google/link', { credential, userId: state.userId });
            linked = await res.json();
        } catch (_) {
            linked = null;
        }
        if (!linked || !linked.ok) {
            message.textContent = getTerm(googleErrorTerm(linked && linked.error));
            return;
        }
        rememberGoogleEmail(state.userId, googleAccountEmail(credential));
        this.google = true;
        this.showOverview();
    }

    // Unlinking removes the way back in for anyone who never wrote their code
    // down, so the question NAMES the code: it is what they will need instead,
    // and this dialog is the last place it is guaranteed to be in front of them.
    async unlinkGoogle() {
        this.notice = '';
        if (!await showConfirm(getTerm('accountGoogleUnlinkConfirm', { code: state.userId }))) return;
        try {
            const res = await postJson('/google/unlink', { userId: state.userId });
            const body = await res.json();
            if (!body || !body.ok) throw new Error('rejected');
        } catch (_) {
            this.fail('accountGoogleFailed');
            return;
        }
        forgetGoogleEmail();
        this.google = false;
        this.showOverview();
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

    // Alias: what the public board shows while no federation record is linked or
    // the profile is Anonymous, rendered in the user's own language. Muted when unset, like the other
    // "nothing chosen yet" values — an unset alias is Anonymous, not a name.
    aliasValue() {
        if (this.loadFailed) return valueEl(UNKNOWN, () => this.retry(), { muted: true });
        if (this.alias === undefined) return valueEl(PENDING, () => {}, { muted: true });
        if (!this.alias) return valueEl(getTerm('accountAliasNone'), () => this.showAlias(), { muted: true });
        return valueEl(composeAlias(this.alias.adjective, this.alias.color, this.alias.animal, getLang()),
            () => this.showAlias());
    }

    // The picker: the three word lists as three dropdowns SIDE BY SIDE, so they
    // read left to right as the name itself — there is nothing to label and no
    // separate preview of the result. The stored triplet is always English while
    // the options are the localized words, which is what lets the leaderboard
    // render one alias in four languages.
    showAlias() {
        this.view = 'alias';
        this.notice = '';
        const current = this.alias || {};
        // Opening the picker with nothing stored starts from a RANDOM triplet
        // rather than three first-list words, so the default is a suggestion
        // instead of the same "Brave Crimson Fox" everybody would otherwise be
        // offered. A word the module no longer holds (a retired one on an old
        // alias) also falls back to a random pick from the current list, which
        // keeps every select showing a value it actually offers.
        const pick = (list, chosen) => list.some((w) => w.en === chosen)
            ? chosen
            : list[Math.floor(Math.random() * list.length)].en;
        const picked = {
            adjective: pick(ALIAS_ADJECTIVES, current.adjective),
            color: pick(ALIAS_COLORS, current.color),
            animal: pick(ALIAS_ANIMALS, current.animal),
        };

        const row = document.createElement('div');
        row.className = 'account-alias-row';
        for (const [part, list, labelTerm] of [
            ['adjective', ALIAS_ADJECTIVES, 'accountAliasAdjective'],
            ['color', ALIAS_COLORS, 'accountAliasColor'],
            ['animal', ALIAS_ANIMALS, 'accountAliasAnimal'],
        ]) {
            const sel = document.createElement('select');
            sel.className = 'account-select';
            // No visible label: the part name survives as the accessible name.
            sel.setAttribute('aria-label', getTerm(labelTerm));
            for (const w of list) {
                const opt = document.createElement('option');
                opt.value = w.en;
                opt.textContent = w[getLang()] || w.en;
                sel.appendChild(opt);
            }
            sel.value = picked[part];
            sel.addEventListener('change', () => { picked[part] = sel.value; });
            row.appendChild(sel);
        }

        const message = document.createElement('div');
        message.className = 'account-note';

        const save = modalButton(getTerm('accountAliasSave'), async () => {
            save.classList.add('disabled');
            message.textContent = '';
            let body = null;
            try {
                const res = await postJson('/profile/alias', {
                    userId: state.userId,
                    adjective: picked.adjective,
                    color: picked.color,
                    animal: picked.animal,
                });
                body = await res.json();
            } catch (_) { body = null; }
            save.classList.remove('disabled');
            if (body && body.ok) { this.alias = body.alias; this.showOverview(); return; }
            message.textContent = getTerm(body && body.error === 'alias_taken'
                ? 'accountAliasTaken' : 'accountAliasFailed');
        });

        const buttons = [];
        // Clear exists only when there is something to clear: an absent button
        // states "nothing is set" better than a disabled one.
        if (this.alias) {
            const clear = modalButton(getTerm('accountAliasClear'), async () => {
                clear.classList.add('disabled');
                message.textContent = '';
                let body = null;
                try {
                    const res = await postJson('/profile/alias', { userId: state.userId, clear: true });
                    body = await res.json();
                } catch (_) { body = null; }
                clear.classList.remove('disabled');
                if (body && body.ok) { this.alias = null; this.showOverview(); return; }
                message.textContent = getTerm('accountAliasFailed');
            });
            buttons.push(clear);
        }
        buttons.push(save);

        this.render((host) => {
            host.appendChild(row);
            host.appendChild(message);
        }, 'accountCancel', () => this.showOverview(), buttons);
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
        this.google = undefined;
        this.alias = undefined;
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
