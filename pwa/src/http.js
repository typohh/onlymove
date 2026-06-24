import { APP_VERSION } from './version.js';

export const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Expose-Headers': 'X-App-Version',
};

// JSON response headers that also stamp the deployed Worker version, so the
// client can detect a stale bundle and trigger a quiescent reload (see
// js/state.js maybeReload). Returns a fresh object per call.
export function versionedJsonHeaders() {
  return { ...CORS_HEADERS, 'Content-Type': 'application/json', 'X-App-Version': APP_VERSION };
}

export function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS_HEADERS, 'Content-Type': 'application/json' },
  });
}

export function tooManyRequests() {
  return new Response('Too many requests', {
    status: 429,
    headers: { ...CORS_HEADERS, 'Retry-After': '60' },
  });
}
