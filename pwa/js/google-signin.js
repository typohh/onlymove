// js/google-signin.js
//
// The browser half of Google sign-in: load Google Identity Services, render
// Google's own button, and hand the resulting ID token to the caller. Everything
// about what that token MEANS — whether it logs the user in or attaches itself to
// the code this device is already using — is decided by js/account-ui.js against
// the Worker; this file only produces the token.
//
// The library is loaded ON DEMAND, when the Account dialog opens its sign-in
// view, and never at boot. Nothing else in the app depends on it, it is a
// third-party script on a third-party origin, and the app is offline-first — so
// paying for it at startup would slow every launch for a screen almost no launch
// visits, and would put a cross-origin fetch in the path of a cold offline boot.
//
// The button is GOOGLE'S OWN rendered button rather than one of ours, because
// their branding terms require it — and it is what a user is looking for.
//
// The address is decoded here for DISPLAY ONLY, and never sent anywhere: the
// server stores an opaque subject id and nothing else (src/handlers/google.js),
// so this device's own copy is the only place the Account dialog can get a name
// for the linked account. It is kept under the account code it belongs to and
// read back only for a matching code, which is what makes a stale copy from a
// previous account harmless rather than a wrong name on the dialog — the same
// rule the sync state uses.

import { GOOGLE_CLIENT_ID, googleSignInConfigured } from '../shared/google-auth.js';
import { readJSON, writeJSON } from './persist.js';

const GIS_SRC = 'https://accounts.google.com/gsi/client';
const LOCAL_EMAIL_KEY = 'googleAccountEmail';

// Single-flight: the dialog can be opened, closed and reopened, and the script
// must be fetched once.
let gisLoad = null;

export function googleSignInAvailable() {
    return googleSignInConfigured();
}

function loadGis() {
    if (window.google?.accounts?.id) return Promise.resolve();
    if (gisLoad) return gisLoad;
    gisLoad = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = GIS_SRC;
        script.async = true;
        script.defer = true;
        script.onload = () => {
            if (window.google?.accounts?.id) resolve();
            else reject(new Error('gsi loaded without accounts.id'));
        };
        script.onerror = () => reject(new Error('gsi failed to load'));
        document.head.appendChild(script);
    }).catch((e) => {
        // A failed load must not be remembered as a failure forever — the usual
        // cause is being offline, and the user's next attempt may not be.
        gisLoad = null;
        throw e;
    });
    return gisLoad;
}

// Google's button in two skins; the app is dark by default with a light override
// (style.css), so this follows the same media query rather than a stored setting.
function buttonTheme() {
    return window.matchMedia?.('(prefers-color-scheme: light)').matches
        ? 'outline' : 'filled_black';
}

// Render the button into `host`. `onCredential` receives the raw ID token when
// the user completes a sign-in, and is never called if they dismiss the popup —
// Google reports a dismissal to nobody, so the sign-in view simply stays up with
// its Cancel, which is also what the user would reach for.
export async function renderGoogleButton(host, onCredential) {
    await loadGis();
    window.google.accounts.id.initialize({
        client_id: GOOGLE_CLIENT_ID,
        callback: (response) => onCredential(response?.credential || ''),
        // No One Tap and no silent re-selection: the button is the only entry
        // point, so a returning user always gets to choose WHICH Google account
        // they mean. Getting that wrong here would log someone into the wrong
        // account code.
        auto_select: false,
        cancel_on_tap_outside: true,
        ux_mode: 'popup',
    });
    // RECTANGULAR rather than a pill, and that is a dark-mode fix rather than a
    // taste: GIS draws the personalized variant ("Sign in as <name>", once a
    // Google session exists) inside a cross-origin iframe whose document paints
    // an opaque WHITE background of its own. Nothing on this side can reach it —
    // the iframe element is already transparent, `color-scheme: dark` does not
    // apply to a child document that paints, and the frame cannot be hidden
    // because it IS the button.
    //
    // What is left is how much of that white rectangle the button covers, and a
    // rectangle covers it far better than a pill: measured against the live
    // button, a pill leaves roughly 20px of white at each rounded end where the
    // shape curves away from the frame, a rectangle a few px of uniform border.
    // So the white becomes an outline on a button instead of a block behind one.
    window.google.accounts.id.renderButton(host, {
        type: 'standard',
        theme: buttonTheme(),
        size: 'large',
        text: 'signin_with',
        shape: 'rectangular',
        logo_alignment: 'left',
    });
}

// The address inside an ID token, for display. Deliberately unverified — nothing
// is decided on it, so there is nothing for a forged value to buy. Returns '' for
// anything that isn't a JWT carrying one.
export function googleAccountEmail(credential) {
    try {
        const payload = String(credential).split('.')[1];
        if (!payload) return '';
        const padded = payload.replace(/-/g, '+').replace(/_/g, '/');
        const binary = atob(padded + '='.repeat((4 - (padded.length % 4)) % 4));
        const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
        const json = JSON.parse(new TextDecoder().decode(bytes));
        return typeof json.email === 'string' ? json.email : '';
    } catch {
        return '';
    }
}

// Remember which address is linked, against the code it is linked TO. Written on
// a link and on a Google login (where it must survive the account switch's wipe,
// which is why it is keyed rather than listed there).
export function rememberGoogleEmail(userId, email) {
    if (!email) return;
    writeJSON(LOCAL_EMAIL_KEY, { userId, email });
}

export function readGoogleEmail(userId) {
    const stored = readJSON(LOCAL_EMAIL_KEY);
    return (stored && stored.userId === userId && typeof stored.email === 'string')
        ? stored.email : '';
}

export function forgetGoogleEmail() {
    try { localStorage.removeItem(LOCAL_EMAIL_KEY); } catch {}
}
