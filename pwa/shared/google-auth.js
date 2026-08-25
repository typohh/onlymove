// pwa/shared/google-auth.js
//
// The Google OAuth client id, in ONE place. The browser initializes Google
// Identity Services with it and the Worker verifies every ID token's `aud`
// against it, so the two ends cannot drift: a token minted for some other site
// is refused because the audience it names is not this one, and that check is
// only as good as its agreement with what the button asked for.
//
// It is deliberately a source constant rather than a wrangler var or a secret.
// A client id is PUBLIC — it ships inside the page either way, and nothing about
// it is a credential; what it needs to be is the SAME string on both sides, which
// an env var read by only one of them would not guarantee.
//
// EMPTY MEANS THE FEATURE IS OFF, everywhere, with no other switch: the Account
// dialog omits its Google row (js/account-ui.js) and the three /google/* routes
// refuse (src/handlers/google.js). That is what lets the code ship before the
// Google Cloud project exists — turning it on is pasting the id here and
// deploying.
//
// To create one: Google Cloud Console -> APIs & Services -> Credentials -> Create
// credentials -> OAuth client ID -> Web application. Add EVERY origin the button
// is rendered on to "Authorized JavaScript origins" — https://onlymove.net,
// https://www.onlymove.net, and http://localhost:8787 for `npm run dev` — since
// Google refuses to render the button on an origin that is not listed. No
// redirect URI is needed: this is the ID-token flow, which never leaves the page.
// No client SECRET is needed either, and none should be created; verifying an ID
// token needs only Google's public keys.
export const GOOGLE_CLIENT_ID = '';

export function googleSignInConfigured() {
    return typeof GOOGLE_CLIENT_ID === 'string' && GOOGLE_CLIENT_ID.length > 0;
}
