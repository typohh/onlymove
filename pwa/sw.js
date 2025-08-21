// This version string is dynamically replaced by the build script (build.mjs)
const VERSION = '1';
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
    '/lib/jszip.min.js'
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
  clients.claim();
  // Clean up old caches that don't match the current PRECACHE_NAME
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName.startsWith('only-move-precache-') && cacheName !== PRECACHE_NAME) {
            console.log('Deleting old precache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

self.addEventListener('fetch', event => {
    const { request } = event;

    // Strategy 1: Network-First for the main HTML page (navigation requests).
    // This ensures you get the latest HTML when online, which in turn requests the
    // correct hashed assets, preventing the 404 race condition.
    if (request.mode === 'navigate') {
        event.respondWith(
            fetch(request)
                .then(networkResponse => {
                    // Good network response? Cache it for offline use and return it.
                    const responseToCache = networkResponse.clone();
                    caches.open(RUNTIME_CACHE_NAME).then(cache => {
                        cache.put(request, responseToCache);
                    });
                    return networkResponse;
                })
                .catch(() => {
                    // Network failed? Serve the cached version (for offline access).
                    return caches.match(request);
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
                // Ensure we have a valid response before caching
                if (networkResponse && networkResponse.status === 200) {
                    const responseToCache = networkResponse.clone();
                    caches.open(RUNTIME_CACHE_NAME).then(cache => {
                        cache.put(request, responseToCache);
                    });
                }
                return networkResponse;
            });
        })
    );
});