// js/account-ui.js
//
// The Account dialog (hamburger menu → More). One editable field holds the
// account code — which IS the userId, the credential every attempt is
// reported under. Untouched, it shows the current account and Copy is active;
// edited, it becomes a login form and Log in is active (the two buttons
// toggle in opposition). The actual switch (wipe + reload + full history
// pull) lives in account-sync.switchAccount.

import { state } from './state.js';
import { getTerm } from './i18n.js';
import {
    showModal, modalTitle, modalButton, modalButtonRow, showConfirm, showToast,
} from './overlays.js';
import { isValidAccountCode, switchAccount, accountExists } from './account-sync.js';

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
        },
    });
}
