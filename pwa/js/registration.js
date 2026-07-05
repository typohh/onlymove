// js/registration.js
//
// EGD-leaderboard registration confirmation. When the PWA boots on an admin-
// issued invite link (onlymove.net/register?t=<token>), this previews the
// invitation's EGD name and, on confirm, binds THIS browser's account (the
// userId) to that EGD id. The userId travels ONLY in the POST body — it is
// never shown in the UI, which is the whole point of a link-based claim over an
// account-code entry. The account's age + size shown in the prompt are the
// safeguard that replaces the account-code step: a fresh account reads "created
// today, 0 answers" so the user can tell they're on the wrong account before
// confirming. See docs/leaderboard_plan.md "Registration flow".
//
// Endpoint contracts (pwa/src/handlers/register.js):
//   GET  /invite?t=<token> -> {ok:true, egdName} | {ok:false, error:'invalid'|'claimed'|'bad_request'}
//   POST /register {token, userId} -> {ok:true, egdName}
//                                   | {ok:false, error:'pin_taken'|'claimed'|'invalid'|'bad_request'|'server'}

import { state, getMergedFullAnswerHistory } from './state.js';
import { showModal, modalTitle, modalButton, modalButtonRow, showToast } from './overlays.js';
import { postJson } from './net.js';

// Same 8-alphanumeric shape the Worker's TOKEN_RE enforces; a mismatch means the
// URL didn't come from a real invite link, so we don't even hit /invite.
const TOKEN_RE = /^[A-Za-z0-9]{8}$/;

// Map the locked /invite + /register error codes to a user-facing line. Anything
// unrecognized (incl. 'server' / a network failure / no error field) falls to
// the generic retry message.
function mapErrorMessage(error) {
    switch (error) {
        case 'pin_taken':     return 'This EGD id is already registered to another account';
        case 'account_bound': return 'This account is already registered to a different EGD id';
        case 'claimed':       return 'This invite link was already used';
        case 'invalid':       return 'Invalid invite link';
        default:              return 'Registration failed, please try again';
    }
}

// The account's first-attempt date + total attempt count, read from the local
// full-career history (both board sizes, working set + cold IDB tail, de-duped).
// An empty/brand-new account reads "created <today>, 0 answers".
async function accountSummary() {
    let history = [];
    try {
        history = await getMergedFullAnswerHistory();
    } catch (_) {
        history = [];
    }
    const count = history.length;
    // getMergedFullAnswerHistory returns entries in ascending date order, so the
    // first entry is the earliest attempt. Fall back to now for an empty account.
    const firstMs = (count > 0 && history[0] && history[0].date) ? history[0].date : Date.now();
    return { count, dateStr: new Date(firstMs).toLocaleDateString() };
}

function showRegistrationConfirm({ token, egdName, count, dateStr }) {
    showModal({
        build: (modal, close) => {
            modal.appendChild(modalTitle(
                `Register this account (created ${dateStr}, ${count} answers) as ${egdName}?`
            ));

            let submitting = false;
            const confirm = async () => {
                if (submitting) return;
                submitting = true;
                // Keep a "Registering…" toast up across the round-trip (the modal
                // closes immediately so the account code can't linger on screen).
                const pending = showToast('Registering…', { autoDismiss: false });
                close();
                let result = null;
                try {
                    // The userId (account credential) goes ONLY here — never in the UI.
                    const res = await postJson('/register', { token, userId: state.userId });
                    result = await res.json();
                } catch (_) {
                    result = null;
                }
                if (pending.parentNode) pending.parentNode.removeChild(pending);
                if (result && result.ok) {
                    // Registered — drop the token so no later reload (SW takeover,
                    // version reload, refresh) re-runs registration and re-prompts.
                    stripInviteToken();
                    showToast(`Registered as ${result.egdName}`);
                } else if (result) {
                    // A definitive server rejection (claimed / pin_taken / account_bound /
                    // invalid) won't change on retry, so drop the token too. 'claimed'
                    // for a 1:1 invite means it's already done — show it neutrally.
                    stripInviteToken();
                    showToast(mapErrorMessage(result.error), { isError: result.error !== 'claimed' });
                } else {
                    // Network failure only — keep the token in the URL so a reload can
                    // retry the registration rather than silently dropping the invite.
                    showToast(mapErrorMessage(null), { isError: true });
                }
            };

            modal.appendChild(modalButtonRow([
                modalButton('Cancel', close),
                modalButton('Confirm', confirm),
            ]));
        },
    });
}

// Remove the ?t=<token> invite param from the address bar without a navigation,
// preserving the path, any other query params, and the hash. Mirrors cleanUrl()
// in problem-loader.js for the linked-problem params.
function stripInviteToken() {
    const url = new URL(window.location);
    if (!url.searchParams.has('t')) return;
    url.searchParams.delete('t');
    const next = url.pathname + url.search + url.hash;
    window.history.replaceState({}, '', next);
}

/**
 * Boot hook: if the PWA opened on an invite link (?t=<token>), preview the
 * invitation and, on a valid one, show the registration-confirmation modal.
 * A no-op when there's no valid token. Safe to call unconditionally at end of
 * boot — it self-gates on the token.
 */
export async function maybeHandleRegistration() {
    const token = new URLSearchParams(window.location.search).get('t') || '';
    if (!TOKEN_RE.test(token)) return;

    // The token deliberately STAYS in the URL through the async preview + confirm
    // so a reload before the user decides re-shows the prompt: an auto-reload — an
    // SW-takeover `controllerchange` reload (index.html), a version-mismatch
    // maybeReload, a load-failure reload, or a manual refresh — must not silently
    // drop the invitation. It is stripped only once the invite is DEFINITIVELY
    // resolved (registered, or the server reports it already claimed / invalid),
    // after which reloads stop re-prompting. (Stripping it up front instead would
    // lose the registration if a reload fired while the modal was still open.)

    let preview = null;
    try {
        const res = await fetch(`/invite?t=${encodeURIComponent(token)}`);
        preview = await res.json();
    } catch (_) {
        preview = null;
    }
    if (!preview || !preview.ok) {
        // A definitive server response (claimed / invalid / bad_request) is resolved,
        // so strip the token — else a reload keeps re-surfacing the same dead link.
        // A network failure (preview === null) keeps it so a reload retries the
        // preview. 'claimed' means already-registered on a 1:1 invite → show neutral.
        if (preview) stripInviteToken();
        showToast(mapErrorMessage(preview && preview.error),
                  { isError: !(preview && preview.error === 'claimed') });
        return;
    }

    const { count, dateStr } = await accountSummary();
    showRegistrationConfirm({ token, egdName: preview.egdName, count, dateStr });
}
