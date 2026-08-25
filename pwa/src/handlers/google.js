// pwa/src/handlers/google.js
//
// Google sign-in as a SECOND WAY BACK INTO AN ACCOUNT — never a second kind of
// account. The account is still the 8-char code and nothing else; what a link
// adds is a way to look that code up again on a device that does not have it,
// so a user who never wrote the code down is not locked out.
//
//   POST /google/signin {credential, userId} -> {ok:true, status, userId?}
//   POST /google/link   {credential, userId} -> {ok:true}
//   POST /google/unlink {userId}             -> {ok:true}
//
// SIGNIN RESOLVES, IT DOES NOT WRITE, and the split is the point. Presented with
// a verified Google account there are two sensible meanings — "this is me, log me
// in" and "attach this to the account I am already using" — and which one the
// user means is not knowable from the token. So the resolve says which case it is
// and the CLIENT asks (js/account-ui.js), naming the account code in the
// question; an unbound token silently binding itself to whatever code the device
// happens to hold is the trap that avoids, since a returning user's device holds
// a fresh throwaway code at exactly that moment.
//
// `status` is one of:
//   'login'   — the Google account is bound to a DIFFERENT code, returned as
//               `userId`. That is the login: the bearer proved control of the
//               Google account, and the code is what that proves entitlement to.
//   'self'    — bound to the code the caller is already using; nothing to do.
//   'unbound' — no binding anywhere, so linking is available.
//
// The binding is EXCLUSIVE IN BOTH DIRECTIONS — one code per Google account and
// one Google account per code — which is what makes 'login' a single answer
// rather than a list to choose from. Both directions are enforced here and by the
// table's two keys (google_accounts: google_sub PRIMARY KEY, user_id UNIQUE).
//
// WHAT IS STORED IS THE OPAQUE SUBJECT ID AND NOTHING ELSE. The token also
// carries the address and the display name; verifyGoogleIdToken does not return
// them, so there is no email in D1 to leak, to serve, or to have to describe in
// the privacy policy. The cost is that the server cannot say WHICH Google account
// is linked — only that one is — and the Account dialog names it from the token
// it just received, on that device alone.

import { jsonResponse } from '../http.js';
import { isValidUserId } from '../guards.js';
import { googleSignInConfigured } from '../../shared/google-auth.js';
import { verifyGoogleIdToken } from '../google-token.js';

// Is a Google account linked to this code? One point lookup on the UNIQUE index.
// Read by GET /profile so the Account dialog paints its whole panel in one
// request; deliberately a boolean, because a boolean is all we hold.
export async function hasGoogleBinding(env, userId) {
  if (!env.DB || !googleSignInConfigured() || !isValidUserId(userId)) return false;
  const row = await env.DB.prepare(
    `SELECT 1 AS x FROM google_accounts WHERE user_id = ?1`
  ).bind(userId).first();
  return !!row;
}

// Shared preamble: parse, check the feature is switched on at all, and verify the
// token. Returns either { sub, userId } or a Response to send back.
async function readVerifiedRequest(request, env, { needCredential = true } = {}) {
  if (!googleSignInConfigured()) return jsonResponse({ ok: false, error: 'unavailable' }, 404);
  if (!env.DB) return jsonResponse({ ok: false, error: 'server' }, 500);
  let data;
  try { data = await request.json(); } catch { return jsonResponse({ ok: false, error: 'bad_request' }, 400); }
  if (!isValidUserId(data?.userId)) return jsonResponse({ ok: false, error: 'bad_request' }, 400);
  if (!needCredential) return { sub: null, userId: data.userId };
  const sub = await verifyGoogleIdToken(data.credential);
  // One answer for every way a token can fail to be a current, correctly-audienced
  // Google token: the caller can do nothing different with a finer reason, and a
  // finer reason is a probe oracle.
  if (!sub) return jsonResponse({ ok: false, error: 'invalid_token' }, 401);
  return { sub, userId: data.userId };
}

export async function handleGoogleSignIn(request, env) {
  const parsed = await readVerifiedRequest(request, env);
  if (parsed instanceof Response) return parsed;
  const { sub, userId } = parsed;
  try {
    const row = await env.DB.prepare(
      `SELECT user_id FROM google_accounts WHERE google_sub = ?1`
    ).bind(sub).first();
    if (!row) return jsonResponse({ ok: true, status: 'unbound' });
    if (row.user_id === userId) return jsonResponse({ ok: true, status: 'self' });
    return jsonResponse({ ok: true, status: 'login', userId: row.user_id });
  } catch (e) {
    console.error('handleGoogleSignIn failed:', e);
    return jsonResponse({ ok: false, error: 'server' }, 500);
  }
}

export async function handleGoogleLink(request, env) {
  const parsed = await readVerifiedRequest(request, env);
  if (parsed instanceof Response) return parsed;
  const { sub, userId } = parsed;
  try {
    // Both exclusivity directions, checked so each gets its own message. The
    // table's keys are the backstop that makes them true under a race; these two
    // reads are what turn a constraint violation into something the dialog can
    // say. Re-linking the pair that is already bound is a no-op, not an error —
    // it is what a client with a stale view of the row would send.
    const [bySub, byUser] = await Promise.all([
      env.DB.prepare(`SELECT user_id FROM google_accounts WHERE google_sub = ?1`).bind(sub).first(),
      env.DB.prepare(`SELECT google_sub FROM google_accounts WHERE user_id = ?1`).bind(userId).first(),
    ]);
    if (bySub && bySub.user_id !== userId) return jsonResponse({ ok: false, error: 'google_taken' }, 409);
    if (byUser && byUser.google_sub !== sub) return jsonResponse({ ok: false, error: 'account_taken' }, 409);
    if (bySub) return jsonResponse({ ok: true });

    await env.DB.prepare(
      `INSERT INTO google_accounts (google_sub, user_id, created_at) VALUES (?1, ?2, ?3)`
    ).bind(sub, userId, new Date().toISOString()).run();
    return jsonResponse({ ok: true });
  } catch (e) {
    console.error('handleGoogleLink failed:', e);
    return jsonResponse({ ok: false, error: 'server' }, 500);
  }
}

// Unlink takes the account code and no token, exactly like every other write in
// this app: the code IS the credential, so anyone able to send it is already the
// account holder. Requiring a Google token here would instead lock the removal
// behind the very thing a user unlinks BECAUSE they have lost access to it.
export async function handleGoogleUnlink(request, env) {
  const parsed = await readVerifiedRequest(request, env, { needCredential: false });
  if (parsed instanceof Response) return parsed;
  try {
    await env.DB.prepare(`DELETE FROM google_accounts WHERE user_id = ?1`)
      .bind(parsed.userId).run();
    return jsonResponse({ ok: true });
  } catch (e) {
    console.error('handleGoogleUnlink failed:', e);
    return jsonResponse({ ok: false, error: 'server' }, 500);
  }
}
