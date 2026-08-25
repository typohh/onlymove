// pwa/src/google-token.js
//
// Verifies a Google Identity Services ID token (the `credential` string the
// browser's Sign in with Google button hands back) and returns the account's
// OPAQUE SUBJECT ID — nothing else. The subject is what we store; the token also
// carries a name, an email and a picture URL, and this deliberately does not
// return them, so no caller can start keeping them by accident.
//
// The verification is done HERE rather than by calling Google's tokeninfo
// endpoint, which is what Google recommends and is also the cheaper shape: the
// signing keys change rarely, so an isolate that has them verifies with no
// subrequest at all, while tokeninfo would put a round trip to Google in front
// of every sign-in.
//
// A token is accepted only when ALL of these hold, because each one alone is
// forgeable:
//   - the header names RS256, so a token claiming `none` (or a symmetric alg,
//     where the "key" is a public value anyone has) cannot reach the verify;
//   - the signature checks out against the Google key the header's `kid` names;
//   - the issuer is Google;
//   - the audience is OUR client id — without this, an ID token minted for any
//     other site that uses Google sign-in would log its bearer in here;
//   - it has not expired, and was not issued in the future.

import { GOOGLE_CLIENT_ID, googleSignInConfigured } from '../shared/google-auth.js';

const JWKS_URL = 'https://www.googleapis.com/oauth2/v3/certs';
// Google mints tokens with both spellings of the issuer and has done for years;
// both are documented as valid, so both are accepted.
const ISSUERS = new Set(['accounts.google.com', 'https://accounts.google.com']);
// Tolerance on exp/iat for ordinary clock disagreement between Google's clock
// and the Worker's. Small enough that an expired token is not a usable one.
const CLOCK_SKEW_S = 120;
// Bounds on how long the fetched key set is trusted. Google's own Cache-Control
// drives it; the clamp is what stops a missing or absurd header from either
// re-fetching per request or pinning a rotated-out key set for a day.
const JWKS_MIN_TTL_MS = 5 * 60 * 1000;
const JWKS_MAX_TTL_MS = 6 * 60 * 60 * 1000;

// Per-isolate key cache: { keys: Map<kid, CryptoKey>, expiresAt }. An isolate is
// short-lived and there are many, so this is a hit-rate optimization and never a
// correctness assumption — every path can fall back to a fetch.
let jwksCache = null;
let jwksInflight = null;

function b64urlToBytes(part) {
  const padded = part.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(padded + '='.repeat((4 - (padded.length % 4)) % 4));
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

function b64urlToJson(part) {
  return JSON.parse(new TextDecoder().decode(b64urlToBytes(part)));
}

// Fetch + import Google's signing keys. Single-flight, so a burst of sign-ins in
// one isolate costs one fetch rather than one each.
async function fetchJwks() {
  if (jwksInflight) return jwksInflight;
  jwksInflight = (async () => {
    const res = await fetch(JWKS_URL);
    if (!res.ok) throw new Error(`jwks ${res.status}`);
    const body = await res.json();
    if (!body || !Array.isArray(body.keys)) throw new Error('jwks malformed');
    const keys = new Map();
    for (const jwk of body.keys) {
      if (jwk.kty !== 'RSA' || (jwk.alg && jwk.alg !== 'RS256')) continue;
      try {
        keys.set(jwk.kid, await crypto.subtle.importKey(
          'jwk', jwk, { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' }, false, ['verify'],
        ));
      } catch {
        // One unusable key in the set is not a reason to reject the others.
      }
    }
    if (keys.size === 0) throw new Error('jwks empty');
    const maxAge = Number(/max-age=(\d+)/.exec(res.headers.get('cache-control') || '')?.[1]);
    const ttl = Math.min(JWKS_MAX_TTL_MS,
      Math.max(JWKS_MIN_TTL_MS, Number.isFinite(maxAge) ? maxAge * 1000 : JWKS_MIN_TTL_MS));
    jwksCache = { keys, expiresAt: Date.now() + ttl };
    return keys;
  })().finally(() => { jwksInflight = null; });
  return jwksInflight;
}

// The key for one `kid`. A miss re-fetches ONCE even when the cache is still
// inside its TTL: that is exactly what a key rotation looks like from here, and
// treating it as a bad token instead would refuse every sign-in until the TTL
// ran out.
async function keyForKid(kid) {
  if (jwksCache && jwksCache.expiresAt > Date.now()) {
    const hit = jwksCache.keys.get(kid);
    if (hit) return hit;
  }
  const keys = await fetchJwks();
  return keys.get(kid) || null;
}

// Returns the Google subject id (a string), or null when the token is anything
// other than a currently-valid token minted for this site. Never throws: a
// caller cannot act differently on "malformed" than on "forged", so the two
// collapse into one answer.
export async function verifyGoogleIdToken(credential) {
  if (!googleSignInConfigured()) return null;
  if (typeof credential !== 'string' || credential.length === 0) return null;
  const parts = credential.split('.');
  if (parts.length !== 3) return null;

  try {
    const header = b64urlToJson(parts[0]);
    if (header.alg !== 'RS256' || typeof header.kid !== 'string') return null;

    const key = await keyForKid(header.kid);
    if (!key) return null;

    const signed = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
    const ok = await crypto.subtle.verify(
      'RSASSA-PKCS1-v1_5', key, b64urlToBytes(parts[2]), signed);
    if (!ok) return null;

    const claims = b64urlToJson(parts[1]);
    if (!ISSUERS.has(claims.iss)) return null;
    if (claims.aud !== GOOGLE_CLIENT_ID) return null;
    const now = Math.floor(Date.now() / 1000);
    if (!Number.isFinite(claims.exp) || claims.exp + CLOCK_SKEW_S < now) return null;
    if (Number.isFinite(claims.iat) && claims.iat - CLOCK_SKEW_S > now) return null;
    if (typeof claims.sub !== 'string' || !/^\d{1,64}$/.test(claims.sub)) return null;
    return claims.sub;
  } catch (e) {
    console.error('verifyGoogleIdToken failed:', e);
    return null;
  }
}
