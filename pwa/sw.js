// This version string is dynamically replaced by the build script (build.mjs)
const VERSION = 'initial-version';
const PRECACHE_NAME = `only-move-precache-${VERSION}`;
const RUNTIME_CACHE_NAME = 'only-move-runtime';

// Assets that are core to the app and only change when the app version changes.
// These are cached permanently until a new service worker with a new VERSION is installed.
const PRECACHE_URLS = [
    '/lib/wgo/wood_1024.jpg',
    '/lib/wgo/stones/black_128.png',
    '/lib/wgo/stones/white_128.png',
    '/lib/wgo/wgo.player.css',
    '/lib/wgo/wgo.min.js',
    '/lib/wgo/wgo.player.min.js',
    '/lib/jszip.min.js',
    '/lib/math.min.js',
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
    // M (rating→rank map) is required at init with no fallback — main.js aborts
    // fatal if it can't load (see Sentry JAVASCRIPT-2V/2W: transient mobile-Safari
    // fetch failures). Precache it so a network blip can't kill the whole app.
    // Version-stamped so ?v=VERSION matches the client's fetch in
    // rank-decomposition.js; the precache is named per-VERSION and deleted on
    // activate, so each deploy fetches a fresh copy — it can't go stale.
    `/tsumegos/rating_to_rank.json?v=${VERSION}`,
    // population_constants.json (fitted time-bias + retention constants) is a
    // MANDATORY asset — the client carries no baked defaults and main.js aborts if it
    // can't load — so it joins the atomic precache like M. Version-stamped, so the
    // per-VERSION precache (deleted + rebuilt on activate) fetches a fresh copy each
    // deploy and it can't go stale. The pipeline always ships a valid file, so addAll
    // won't 404 on a normal deploy.
    `/tsumegos/population_constants.json?v=${VERSION}`,
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
        caches.open(PRECACHE_NAME)
            .then(cache => {
                console.log('Opened precache and caching static libraries.');
                return cache.addAll(PRECACHE_URLS);
            })
    );
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

    // Prune runtime cache entries whose ?v= no longer matches the current
    // VERSION (lookup.txt, metadata.json) so they don't accumulate one
    // entry per release.
    const runtimeCache = await caches.open(RUNTIME_CACHE_NAME);
    const requests = await runtimeCache.keys();
    await Promise.all(requests.map(req => {
      const v = new URL(req.url).searchParams.get('v');
      if (v && v !== VERSION) {
        console.log('Pruning stale versioned cache entry:', req.url);
        return runtimeCache.delete(req);
      }
    }));

    await self.clients.claim();
  })());
});

self.addEventListener('push', event => {
    let payload = {};
    if (event.data) {
        try {
            payload = event.data.json();
        } catch {
            payload = { title: 'Time to review', body: event.data.text() };
        }
    }
    const title = payload.title || 'Time to review';
    const options = {
        body: payload.body || '',
        icon: '/icon-192.png',
        badge: '/icon-192.png',
        tag: 'recall-batch',          // one stacked notification, not many
        renotify: true,
        data: { url: payload.url || '/', ids: payload.ids || [] },
    };
    event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', event => {
    event.notification.close();
    const targetUrl = (event.notification.data && event.notification.data.url) || '/';
    event.waitUntil((async () => {
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
    event.respondWith(
        caches.match(request).then(cachedResponse => {
            // If the asset is in the cache, return it immediately.
            if (cachedResponse) {
                return cachedResponse;
            }
            // Otherwise, fetch it from the network, add it to the cache for next time, and return it.
            return fetch(request).then(networkResponse => {
                // Ensure we have a valid response and it's a GET request before caching
                if (networkResponse && networkResponse.status === 200 && request.method === 'GET') {
                    if (request.url.endsWith('.zip')) {
                        const contentType = networkResponse.headers.get('content-type');
                        if (contentType && (contentType.includes('application/zip') || contentType.includes('application/octet-stream'))) {
                            const responseToCache = networkResponse.clone();
                            caches.open(RUNTIME_CACHE_NAME).then(cache => {
                                cache.put(request, responseToCache);
                            }).catch(err => console.error('Cache put failed:', err));
                        }
                    } else {
                        const responseToCache = networkResponse.clone();
                        caches.open(RUNTIME_CACHE_NAME).then(cache => {
                            cache.put(request, responseToCache);
                        }).catch(err => console.error('Cache put failed:', err));
                    }
                }
                return networkResponse;
            });
        })
    );
});
