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
// credentials -> OAuth client ID -> Web application. "Authorized JavaScript
// origins" gets https://onlymove.net and https://www.onlymove.net, and NOTHING
// ELSE — Google refuses to render the button on an unlisted origin, so that list
// is the only thing deciding which pages can obtain a token bearing this
// audience, and the audience is all the Worker checks. No redirect URI is needed:
// this is the ID-token flow, which never leaves the page. No client SECRET is
// needed either, and none should be created; verifying an ID token needs only
// Google's public keys.
//
// LOCALHOST BELONGS TO A SEPARATE CLIENT, not to this one. `npm run dev` serves
// on the wrangler DEFAULT port, so authorizing http://localhost:8787 here would
// let any page another project happens to serve there mint a token this Worker
// accepts — the client id is public, being in the shipped bundle, so knowing it
// is no barrier. It is not remotely reachable (a page cannot claim localhost as
// its origin), which makes the developer's own machine the realistic target
// rather than a user's. So: a second "dev" OAuth client carries the localhost
// origin, its id is pasted in here while testing and reverted before committing.
// Committing it by mistake fails LOUDLY rather than quietly — the dev client does
// not authorize onlymove.net, so the button refuses to render in production
// instead of working with a weaker audience.
export const GOOGLE_CLIENT_ID = '214946937470-vc9vmlmdilmcim6b3f55q5v06psidtp0.apps.googleusercontent.com';

export function googleSignInConfigured() {
    return typeof GOOGLE_CLIENT_ID === 'string' && GOOGLE_CLIENT_ID.length > 0;
}
