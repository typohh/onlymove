// These version strings are dynamically replaced by the build script (build.mjs).
// VERSION is the app/build version (moves on any deploy); DATA_VERSION is the data
// content hash (moves only when the shipped tsumego data changes), used to bust
// metadata.json / lookup.txt so a pure-code deploy doesn't re-pull them.
const VERSION = 'initial-version';
const DATA_VERSION = 'initial-data-version';
const PRECACHE_NAME = `only-move-precache-${VERSION}`;
const RUNTIME_CACHE_NAME = 'only-move-runtime';

// Assets that are core to the app and only change when the app version changes.
// These are cached permanently until a new service worker with a new VERSION is installed.
const PRECACHE_URLS = [
    '/lib/wgo/wood_1024.jpg',
    '/lib/wgo/stones/black_128.png',
    '/lib/wgo/stones/white_128.png',
    '/lib/wgo/wgo.min.js',
    '/lib/wgo/wgo.player.min.js',
    '/lib/jszip.min.js',
    '/manifest.json',
    '/sound/stone.wav',
    // Decorative ink-wash corner masks. Fixed filenames (not content-hashed),
    // so precaching them in the version-stamped precache both makes them
    // available offline from first install AND refreshes them on every deploy
    // (the per-VERSION precache is rebuilt + the old one deleted on activate) —
    // avoiding the runtime cache-first staleness that fixed-name assets hit.
    '/corners/corner_tl.png',
    '/corners/corner_tr.png',
    '/corners/corner_bl.png',
    '/corners/corner_br.png',
    // NOTE: rating_to_rank.json (M) and population_constants.json are no longer
    // precached or fetched — they are inlined into the JS bundle at build time
    // (see pwa/js/rank-decomposition.js + population-constants.js), so they ride
    // the hashed bundle and can't fail to load (was Sentry JAVASCRIPT-2V/2W).
];

// Pathname lookup for the list above. The fetch handler routes these to the
// precache BY NAME rather than through caches.match() — see the ordering note
// there — so it needs to recognise one without re-scanning the array.
const PRECACHE_PATHS = new Set(PRECACHE_URLS);

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil((async () => {
        const cache = await caches.open(PRECACHE_NAME);
        console.log('Opened precache and caching static libraries.');
        // Deliberately NOT cache.addAll: that is all-or-nothing, so a single
        // flaky fetch rejects the whole install and the device runs the session
        // with NO precache — every entry then falls through to the fetch
        // handler and is fetched over the network. Per-URL, a failure costs
        // only its own entry, and the handler re-fills it on first use.
        const results = await Promise.allSettled(
            PRECACHE_URLS.map(url => cache.add(url))
        );
        const failed = PRECACHE_URLS.filter((_, i) => results[i].status === 'rejected');
        if (failed.length) console.warn('Precache incomplete, will fill on demand:', failed);
    })());
});

self.addEventListener('activate', event => {
  // Only delete stale precaches. The runtime cache is preserved across
  // versions because its entries are content-addressed (hashed JS/CSS,
  // hashed zip filenames, version-busted query strings on metadata/lookup),
  // so old versions can't poison new ones and re-downloading every release
  // would waste bandwidth on offline-first clients.
  event.waitUntil((async () => {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames
        .filter(name => name.startsWith('only-move-precache-') && name !== PRECACHE_NAME)
        .map(name => {
          console.log('Deleting old precache:', name);
          return caches.delete(name);
        })
    );

    // Prune runtime cache entries whose ?v= matches neither the current build
    // VERSION nor the current DATA_VERSION (lookup.txt / metadata.json are busted by
    // DATA_VERSION now) so stale-version entries don't accumulate one per release.
    const runtimeCache = await caches.open(RUNTIME_CACHE_NAME);
    const requests = await runtimeCache.keys();
    await Promise.all(requests.map(req => {
      const url = new URL(req.url);
      // Cross-origin and dynamic-API entries were cached by older SW versions (the
      // fetch handler no longer intercepts either) — drop the dead weight.
      if (url.origin !== self.location.origin
          || ['/server-stats', '/push/vapid-public-key', '/profile'].includes(url.pathname)) {
        console.log('Pruning no-longer-intercepted cache entry:', req.url);
        return runtimeCache.delete(req);
      }
      // A runtime copy of a PRECACHE asset SHADOWS the precache permanently:
      // caches.match() searches caches in creation order, and activate deletes
      // + recreates the version-stamped precache, which moves it AFTER the
      // runtime cache from the first deploy onward. These URLs carry no ?v=, so
      // the version prune below never reaches them either — a copy that stored
      // badly would be served for the life of the install. The fetch handler no
      // longer creates these; this clears the ones it already did.
      if (PRECACHE_PATHS.has(url.pathname) && !url.search) {
        console.log('Pruning runtime copy shadowing the precache:', req.url);
        return runtimeCache.delete(req);
      }
      const v = url.searchParams.get('v');
      if (v && v !== VERSION && v !== DATA_VERSION) {
        console.log('Pruning stale versioned cache entry:', req.url);
        return runtimeCache.delete(req);
      }
    }));

    await self.clients.claim();
  })());
});

// ---------------------------------------------------------------------------
// Outstanding-reminder state.
//
// getNotifications() cannot answer "is a reminder still in the tray?" on Chrome
// for Android: the displayed set is served out of an in-memory map in the
// browser process that showed the notification, and a recall push is shown
// while the app is closed, so by the time the user opens the app that process
// is gone and the call returns an empty list. We therefore track it ourselves,
// from the events that DO fire — push (shown), notificationclick and
// notificationclose (gone).
//
// Persisted in a Cache rather than a variable because the service worker is
// killed between the push and the user opening the app. Deliberately NOT named
// with the precache prefix, so activate's stale-precache sweep leaves it alone.
const NOTIF_STATE_CACHE = 'only-move-notif-state';
// Synthetic key. Never fetched — the fetch handler only sees network requests,
// and nothing ever requests this path.
const NOTIF_STATE_URL = '/__recall-notification-state';

async function readNotifState() {
    try {
        const cache = await caches.open(NOTIF_STATE_CACHE);
        const res = await cache.match(NOTIF_STATE_URL);
        return res ? await res.json() : null;
    } catch {
        return null;
    }
}

// `null` clears. Failures are swallowed: losing the record costs a stale tray
// entry, which is the behaviour we had before it existed.
async function writeNotifState(next) {
    try {
        const cache = await caches.open(NOTIF_STATE_CACHE);
        if (next === null) await cache.delete(NOTIF_STATE_URL);
        else await cache.put(NOTIF_STATE_URL, new Response(JSON.stringify(next)));
    } catch {}
}

// Re-renders the title at a new count from the templates the push carried, so
// the countdown stays in the language the reminder was sent in. `{n}` is the
// count; a payload without templates (pre-countdown Worker) returns null and
// the caller leaves the text alone.
function renderCountTitle(t, count) {
    if (!t) return null;
    if (count === 1) return typeof t.one === 'string' ? t.one : null;
    return typeof t.many === 'string' ? t.many.replace('{n}', String(count)) : null;
}

// Notification options for a reminder. Split out because the countdown re-shows
// the notification to update it, and every field except title/body must match
// the original — `tag` above all, since that is what makes the platform REPLACE
// the tray entry instead of stacking a second one.
function notifOptions({ body, data, renotify }) {
    return {
        body: body || '',
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        tag: 'recall-batch',          // one stacked notification, not many
        // A review reminder has no deadline attached to it, so it posts without
        // sound or vibration and waits in the tray. `renotify` governs whether a
        // replacement re-surfaces the banner — set on a genuine new push, clear
        // on a countdown update so the number changes in place, unobtrusively.
        // (`silent` must not be set alongside `vibrate`: the pair is a TypeError.)
        silent: true,
        renotify: !!renotify,
        data: data || { url: '/', ids: [] },
    };
}

// Applies the remaining-due count reported by the page. Only ever UPDATES a
// reminder we believe is outstanding — never creates one, since a count arrives
// after every attempt and most of them have no reminder to speak to.
async function applyRecallCount(count) {
    if (!Number.isFinite(count) || count < 0) return;
    const st = await readNotifState();
    if (!st) return;

    if (count > 0) {
        if (count === st.count) return;      // already showing this number
        const title = renderCountTitle(st.t, count);
        if (!title) return;                  // no templates to re-render from
        await self.registration.showNotification(
            title,
            notifOptions({ body: st.t && st.t.tap, data: st.data, renotify: false })
        );
        await writeNotifState({ ...st, title, count });
        return;
    }

    // Nothing left to review — take the reminder down. close() needs a handle
    // from getNotifications(), which is empty for a notification shown by a
    // since-dead browser process (see above). Tag REPLACEMENT has no such
    // problem: it is keyed on origin + tag by the platform, so re-showing
    // adopts the stale tray entry as one this process owns, and that one we can
    // close. Silent and immediately closed, so it costs at most a flicker.
    let shown = await self.registration.getNotifications({ tag: 'recall-batch' });
    if (shown.length === 0) {
        await self.registration.showNotification(
            st.title || renderCountTitle(st.t, 1) || '',
            notifOptions({ body: st.t && st.t.tap, data: st.data, renotify: false })
        );
        shown = await self.registration.getNotifications({ tag: 'recall-batch' });
    }
    for (const note of shown) note.close();
    await writeNotifState(null);
}

// Localized fallback title for a push with no/invalid JSON title. The normal
// path uses the server-rendered payload.title (already localized per the
// device's stored lang); this only covers a malformed payload. Mirrors the
// getLang() prefix match in js/i18n.js.
function fallbackNotifTitle() {
    const l = (self.navigator && self.navigator.language) || 'en';
    if (l.indexOf('zh') === 0) return '该复习了';
    if (l.indexOf('ja') === 0) return '復習の時間です';
    if (l.indexOf('ko') === 0) return '복습할 시간입니다';
    return 'Time to review';
}

self.addEventListener('push', event => {
    let payload = {};
    if (event.data) {
        try {
            payload = event.data.json();
        } catch {
            payload = { title: fallbackNotifTitle(), body: event.data.text() };
        }
    }
    const title = payload.title || fallbackNotifTitle();
    const data = { url: payload.url || '/', ids: payload.ids || [] };
    event.waitUntil((async () => {
        await self.registration.showNotification(
            title,
            notifOptions({ body: payload.body, data, renotify: true })
        );
        // Record it as outstanding so the page's countdown has something to
        // update, and so we never conjure a reminder that was never sent.
        await writeNotifState({
            title,
            count: Number.isFinite(payload.count) ? payload.count : null,
            t: payload.t || null,
            data,
        });
    })());
});

// The page reports how many problems are still waiting after an attempt; the
// service worker owns the tray and decides what that means for it.
self.addEventListener('message', event => {
    const msg = event.data;
    if (!msg || msg.type !== 'recall-count') return;
    event.waitUntil(applyRecallCount(msg.count));
});

// Swiped away. Drop the record so the countdown's dismissal path can't re-post
// a reminder the user has already dealt with.
self.addEventListener('notificationclose', event => {
    event.waitUntil(writeNotifState(null));
});

self.addEventListener('notificationclick', event => {
    event.notification.close();
    const targetUrl = (event.notification.data && event.notification.data.url) || '/';
    event.waitUntil((async () => {
        await writeNotifState(null);

        let allClients = [];
        try {
            allClients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
        } catch {}

        // Try each existing window in turn. Per the WindowClient spec
        // `navigate()` returns the (possibly new) post-navigation client and
        // can invalidate the original ref — using the returned one avoids
        // "focus on stale client" rejections that happen on Android when the
        // PWA is backgrounded. Per-client try/catch so a single flaky client
        // doesn't sink the whole handler before openWindow gets a chance.
        for (const client of allClients) {
            try {
                let target = client;
                if ('navigate' in client) {
                    const navigated = await client.navigate(targetUrl).catch(() => null);
                    if (navigated) target = navigated;
                }
                if ('focus' in target) {
                    await target.focus();
                    return;
                }
            } catch {}
        }

        // No existing window we could focus — open a new one. On installed
        // Android PWAs this respects the manifest scope + launch_handler so
        // it relaunches the standalone window rather than a browser tab.
        if (self.clients.openWindow) {
            try { await self.clients.openWindow(targetUrl); } catch {}
        }
    })());
});

self.addEventListener('fetch', event => {
    const { request } = event;

    // Only handle HTTP/HTTPS requests to avoid issues with chrome-extension or other schemes
    if (!request.url.startsWith('http')) {
        return;
    }

    const requestUrl = new URL(request.url);

    // Never intercept cross-origin requests (e.g. the client-side EGD GoR fetch in
    // gor-refresh.js): they carry no ?v cache-buster, so the cache-first strategy
    // below would freeze third-party data forever (and the activate-time prune only
    // removes ?v-stamped entries, so the frozen copy would survive every deploy).
    // Let the browser handle them normally.
    if (requestUrl.origin !== self.location.origin) {
        return;
    }

    // Same-origin dynamic Worker GETs: the response changes server-side while the
    // URL doesn't (the hourly-recomputed community-stats blob, the account's own
    // leaderboard profile, a rotated VAPID key), so cache-first would freeze the
    // first response for the life of the cache entry (/server-stats?v=APP_VERSION
    // survives the prune for the whole deploy; the other two have no ?v and survive
    // forever). /profile is the sharp one: its URL is the userId, so a cached copy
    // would show the settings the Account dialog just changed as still un-changed,
    // for good. Let the browser hit the network directly — the Worker already sets
    // the appropriate Cache-Control on these.
    if (['/server-stats', '/push/vapid-public-key', '/profile'].includes(requestUrl.pathname)) {
        return;
    }

    // Strategy 1: Network-First for the main HTML page (navigation requests).
    // This ensures you get the latest HTML when online, which in turn requests the
    // correct hashed assets, preventing the 404 race condition.
    if (request.mode === 'navigate') {
        const url = new URL(request.url);
        const strippedUrl = new URL(url.pathname, url.origin);
        event.respondWith(
            fetch(request, { cache: 'no-cache' })
                .then(networkResponse => {
                    const responseToCache = networkResponse.clone();
                    caches.open(RUNTIME_CACHE_NAME).then(cache => {
                        cache.put(strippedUrl.href, responseToCache);
                    }).catch(err => console.error('Cache put failed:', err));
                    return networkResponse;
                })
                .catch(() => {
                    return caches.match(strippedUrl.href);
                })
        );
        return;
    }

    // Strategy 2: Cache-First for all other assets (CSS, JS, images, etc.).
    // Since your assets are hashed by the build script, their URLs are immutable.
    // If we have an asset in the cache, it's guaranteed to be the correct version.
    // Precache assets are fixed-name and unversioned, so they must resolve
    // against their OWN cache, looked up by name. caches.match() searches every
    // cache in creation order and activate recreates the version-stamped
    // precache, putting it after the runtime cache from the first deploy on —
    // so a runtime copy of one of these would outrank the precache forever, and
    // being unversioned it survives the prune too. They are therefore written
    // back to the PRECACHE on a miss (self-repairing an install that only
    // partly succeeded), never to the runtime cache.
    const isPrecacheAsset = PRECACHE_PATHS.has(requestUrl.pathname) && !requestUrl.search;
    const targetCacheName = isPrecacheAsset ? PRECACHE_NAME : RUNTIME_CACHE_NAME;

    event.respondWith((async () => {
        // A Cache Storage read error must degrade to a network fetch, not sink
        // the request: respondWith() rejecting is a hard network error with no
        // browser-level fallback and no retry.
        const cachedResponse = await (isPrecacheAsset
            ? caches.open(PRECACHE_NAME).then(cache => cache.match(request))
            : caches.match(request)
        ).catch(err => {
            console.error('Cache match failed:', err);
            return undefined;
        });

        // If the asset is in the cache, return it immediately.
        if (cachedResponse) {
            return cachedResponse;
        }

        // Otherwise, fetch it from the network, add it to the cache for next time, and return it.
        const networkResponse = await fetch(request);
        // Ensure we have a valid response and it's a GET request before caching
        if (networkResponse && networkResponse.status === 200 && request.method === 'GET') {
            let cacheable = true;
            if (request.url.endsWith('.zip')) {
                const contentType = networkResponse.headers.get('content-type');
                cacheable = !!contentType && (contentType.includes('application/zip') || contentType.includes('application/octet-stream'));
            }
            if (cacheable) {
                const responseToCache = networkResponse.clone();
                caches.open(targetCacheName).then(cache => {
                    cache.put(request, responseToCache);
                }).catch(err => console.error('Cache put failed:', err));
            }
        }
        return networkResponse;
    })());
});
