// js/invite-access.js
//
// Gates the hamburger-menu "Invite" item. An account may create leaderboard
// invitations if it is an admin OR a registered leaderboard member — a fact only
// the server knows (no local signal, and membership can be established on another
// device). So we probe POST /invite/eligible once per session, cache the boolean,
// and reveal the menu item on a positive answer. The standalone /invite page
// re-verifies server-side, so this is display-gating only — never the auth check.

import { state } from './state.js';
import { postJson } from './worker-post.js';

let inviteAccess = false; // may this account create invites?
let confirmed = false;    // got a definitive server answer this session
let inflight = null;      // dedupe concurrent probes

// Read by ui.updateMenuItems to show/hide the Invite item.
export function canCreateInvites() {
    return inviteAccess;
}

// Probe eligibility at most once per session (retries only if the probe never got
// a clean answer — e.g. offline at boot). Calls onChange when access flips on so
// the caller can re-paint the menu. The 99.9% of accounts that aren't members
// cost one cheap point-read; members get the item revealed.
export async function refreshInviteAccess(onChange) {
    if (confirmed || inflight || !state.userId) return inflight;
    inflight = (async () => {
        try {
            const res = await postJson('/invite/eligible', { userId: state.userId });
            const data = await res.json().catch(() => null);
            if (data && data.ok) {
                confirmed = true;
                inviteAccess = !!data.eligible;
                if (inviteAccess && typeof onChange === 'function') onChange();
            }
        } catch (_) {
            // Offline / transient error — leave unconfirmed so a later call retries.
        } finally {
            inflight = null;
        }
    })();
    return inflight;
}
