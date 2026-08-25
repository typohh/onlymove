// js/account-ui.js
//
// The Account dialog (hamburger menu → More), in two parts.
//
// ACCOUNT — one editable field holds the account code, which IS the userId, the
// credential every attempt is reported under. Untouched, it shows the current
// account and Copy is active; edited, it becomes a login form and Log in is
// active (the two buttons toggle in opposition). The actual switch (wipe +
// reload + full history pull) lives in account-sync.switchAccount.
//
// LEADERBOARD — the account's public identity on /leaderboard: which EGD or AGA
// player it is linked to, and whether to appear anonymously. Both are server
// state, read in ONE GET /profile when the dialog opens and written back only on
// an actual change (the hide flag is also offered at the foot of the rank survey,
// so writing it unconditionally would let one dialog silently undo the other).
//
// The search runs in the BROWSER (js/identity-search.js): the Worker cannot reach
// EGD at all, so a client-side search is the only kind there is. Picking a row
// POSTs /link, which is where exclusivity is enforced — nothing anywhere verifies
// that the user IS that player, and the dialog does not pretend otherwise; what it
// offers instead is Unlink, so a wrong pick is the user's own to undo.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import {
    showModal, modalTitle, modalButton, modalButtonRow, showConfirm, showToast,
} from './overlays.js';
import { isValidAccountCode, switchAccount, accountExists } from './account-sync.js';
import { postJson } from './worker-post.js';
import { searchIdentities } from './identity-search.js';
import { bufferGorUpdate } from './gor-refresh.js';

// POST /link rejections that name a cause the user can act on. Anything else —
// 'server', a network failure, an unrecognized body — falls to the generic retry
// line, same shape as the account-code errors above it.
function linkErrorTerm(error) {
    switch (error) {
        case 'id_taken':    return 'accountLinkTaken';
        case 'unreachable': return 'accountLinkUnreachable';
        default:            return 'accountLinkFailed';
    }
}

export function showAccountModal() {
    showModal({
        build: (modal, close) => {
            modal.appendChild(modalTitle(getTerm('accountTitle')));

            const input = document.createElement('input');
            input.type = 'text';
            input.maxLength = 8;
            input.value = state.userId;
            input.autocapitalize = 'off';
            input.autocomplete = 'off';
            input.spellcheck = false;
            Object.assign(input.style, {
                display: 'block',
                margin: '0.119em auto',
                padding: '0.119em 0.1905em',
                borderRadius: 'var(--radius)',
                background: '#444',
                color: '#fff',
                border: 'calc(0.2 * var(--u)) solid #555',
                fontFamily: 'monospace',
                fontSize: '1.2em',
                letterSpacing: '0.15em',
                width: '11ch',
                textAlign: 'center',
                boxSizing: 'content-box',
            });
            modal.appendChild(input);

            const hint = document.createElement('div');
            hint.textContent = getTerm('accountHint');
            Object.assign(hint.style, {
                fontSize: '0.8em',
                opacity: '0.8',
                maxWidth: '34ch',
                margin: '0 auto',
            });
            modal.appendChild(hint);

            const copyCode = async () => {
                try {
                    await navigator.clipboard.writeText(state.userId);
                    showToast(getTerm('accountCopied'));
                } catch (_) {
                    // Clipboard API unavailable — the code is right there in
                    // the field for a manual copy.
                }
            };

            let checking = false;
            const login = async () => {
                if (checking) return;
                const newId = input.value.trim();
                if (newId === state.userId) return;
                if (!isValidAccountCode(newId)) {
                    showToast(getTerm('accountInvalid'), { isError: true });
                    return;
                }
                checking = true;
                try {
                    // Only adopt codes that exist server-side (≥1 report —
                    // the only state an account can be rebuilt from), so a
                    // typo can't strand the user on an empty account.
                    if (await accountExists(newId) !== true) {
                        showToast(getTerm('accountNotFound'), { isError: true });
                        return;
                    }
                } finally {
                    checking = false;
                }
                if (!await showConfirm(getTerm('accountSwitchConfirm', { code: newId }))) return;
                close();
                // Wipes local state, stores the new credential, reloads; init
                // then pulls the account's full history from the server.
                switchAccount(newId);
            };

            const copyBtn = modalButton(getTerm('accountCopy'), copyCode);
            const loginBtn = modalButton(getTerm('accountLogin'), login);

            // Copy applies to the account you HAVE, Log in to the one you
            // typed — exactly one is meaningful at a time.
            const refreshButtons = () => {
                const unchanged = input.value.trim() === state.userId;
                copyBtn.classList.toggle('disabled', !unchanged);
                loginBtn.classList.toggle('disabled', unchanged);
            };
            refreshButtons();
            input.addEventListener('input', refreshButtons);
            input.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') login();
            });

            modal.appendChild(modalButtonRow([copyBtn, loginBtn]));
            modal.appendChild(buildLeaderboardSection());
        },
    });
}

// The Leaderboard half: the linked identity (search / link / unlink) and the
// anonymity opt-out. Returns the section element immediately and fills it in when
// GET /profile answers — the account half above must not wait on the network.
function buildLeaderboardSection() {
    const section = document.createElement('div');
    section.className = 'account-section';

    const heading = document.createElement('div');
    heading.className = 'account-section-title';
    heading.textContent = getTerm('accountLeaderboard');
    section.appendChild(heading);

    // Everything below is server state, so nothing is offered until the read
    // lands: a checkbox rendered unchecked before the answer arrives is a wrong
    // answer the user may act on.
    const status = document.createElement('div');
    status.className = 'account-note';
    status.textContent = getTerm('accountLoading');
    section.appendChild(status);

    const identity = document.createElement('div');
    identity.style.display = 'none';
    section.appendChild(identity);

    const hideRow = document.createElement('label');
    hideRow.className = 'account-check';
    hideRow.style.display = 'none';
    const hideBox = document.createElement('input');
    hideBox.type = 'checkbox';
    const hideText = document.createElement('span');
    hideText.textContent = getTerm('accountHideLabel');
    hideRow.appendChild(hideBox);
    hideRow.appendChild(hideText);
    section.appendChild(hideRow);

    // Write only on a real change, and only from the state the server reported —
    // the rank survey offers the same flag, so an unconditional write would
    // un-hide whoever opted out there. On a failed write the box goes back, so
    // what is shown is always what the server holds.
    let hiddenOnServer = false;
    hideBox.addEventListener('change', async () => {
        const want = hideBox.checked;
        if (want === hiddenOnServer) return;
        hideBox.disabled = true;
        try {
            const res = await postJson('/profile/hide', { userId: state.userId, hidden: want });
            const body = await res.json();
            if (!body || body.hidden !== want) throw new Error('rejected');
            hiddenOnServer = want;
        } catch (_) {
            hideBox.checked = hiddenOnServer;
            showToast(getTerm('accountHideFailed'), { isError: true });
        } finally {
            hideBox.disabled = false;
        }
    });

    // --- identity: one of two faces, swapped in place by the link/unlink calls ---
    const renderIdentity = (linked) => {
        identity.textContent = '';
        identity.style.display = '';
        if (linked) renderLinked(identity, linked, renderIdentity);
        else renderSearch(identity, renderIdentity);
    };

    (async () => {
        let profile = null;
        try {
            const res = await fetch(`/profile?userId=${encodeURIComponent(state.userId)}`);
            profile = res.ok ? await res.json() : null;
        } catch (_) {
            profile = null;
        }
        if (!profile) {
            // Offline or a failed read: say so and offer nothing, rather than
            // showing an unchecked box and an empty link state that both read as
            // facts about the account.
            status.textContent = getTerm('accountLoadFailed');
            return;
        }
        status.style.display = 'none';
        hiddenOnServer = !!profile.hidden;
        hideBox.checked = hiddenOnServer;
        hideRow.style.display = '';
        renderIdentity(profile.linked || null);
    })();

    return section;
}

// Linked state: who this account is, and the one action that changes it.
function renderLinked(host, linked, rerender) {
    const line = document.createElement('div');
    line.className = 'account-note';
    line.textContent = getTerm('accountLinkedAs', {
        name: linked.name, service: linked.service, id: linked.id,
    });
    host.appendChild(line);

    const unlinkBtn = modalButton(getTerm('accountUnlink'), async () => {
        if (unlinkBtn.classList.contains('disabled')) return;
        if (!await showConfirm(getTerm('accountUnlinkConfirm', { name: linked.name }))) return;
        unlinkBtn.classList.add('disabled');
        try {
            const res = await postJson('/unlink', { userId: state.userId });
            const body = await res.json();
            if (!body || !body.ok) throw new Error('rejected');
            showToast(getTerm('accountUnlinked'));
            rerender(null);
        } catch (_) {
            unlinkBtn.classList.remove('disabled');
            showToast(getTerm('accountLinkFailed'), { isError: true });
        }
    });
    host.appendChild(modalButtonRow([unlinkBtn]));
}

// Unlinked state: the search box, its results, and the pick that links.
function renderSearch(host, rerender) {
    const intro = document.createElement('div');
    intro.className = 'account-note';
    intro.textContent = getTerm('accountLinkIntro');
    host.appendChild(intro);

    const query = document.createElement('input');
    query.type = 'search';
    query.className = 'account-search';
    query.placeholder = getTerm('accountLinkPlaceholder');
    query.autocapitalize = 'off';
    query.autocomplete = 'off';
    query.spellcheck = false;
    host.appendChild(query);

    const message = document.createElement('div');
    message.className = 'account-note';
    host.appendChild(message);

    const results = document.createElement('div');
    results.className = 'account-results';
    host.appendChild(results);

    let searching = false;
    const runSearch = async () => {
        if (searching) return;
        const q = query.value.trim();
        if (!q) return;
        searching = true;
        searchBtn.classList.add('disabled');
        results.textContent = '';
        message.textContent = getTerm('accountLinkSearching');
        let players = [];
        try {
            players = await searchIdentities(q);
        } catch (_) {
            players = [];
        }
        searching = false;
        searchBtn.classList.remove('disabled');
        message.textContent = players.length ? '' : getTerm('accountLinkNoResults');
        for (const p of players) results.appendChild(resultRow(p, message, rerender));
    };

    const searchBtn = modalButton(getTerm('accountLinkSearch'), runSearch);
    query.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            runSearch();
        }
    });
    host.insertBefore(modalButtonRow([searchBtn]), message);
}

// One search hit: who they are per their own database, and the button that claims
// them. EGD rows also hand their already-fetched GoR to the piggyback buffer, so
// the account's first verified rating rides the next /report instead of waiting
// for some other client to be asked to fetch it.
function resultRow(player, message, rerender) {
    const row = document.createElement('div');
    row.className = 'account-result';

    const label = document.createElement('span');
    label.className = 'account-result-name';
    label.textContent = player.name;
    row.appendChild(label);

    const meta = document.createElement('span');
    meta.className = 'account-result-meta';
    // The database's own grade and locality — what tells two people of the same
    // name apart, which is the whole job of this line.
    meta.textContent = [player.service, player.rank, player.loc].filter(Boolean).join(' · ');
    row.appendChild(meta);

    const pick = modalButton(getTerm('accountLinkThisIsMe'), async () => {
        if (pick.classList.contains('disabled')) return;
        pick.classList.add('disabled');
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
                pick.classList.remove('disabled');
                return;
            }
            if (player.egd && Number.isFinite(player.egd.gor)) bufferGorUpdate(player.egd);
            showToast(getTerm('accountLinked', { name: body.linked.name }));
            rerender(body.linked);
        } catch (_) {
            message.textContent = getTerm('accountLinkFailed');
            pick.classList.remove('disabled');
        }
    });
    row.appendChild(pick);
    return row;
}
