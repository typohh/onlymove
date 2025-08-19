const CACHE_NAME = 'yose-tsumego-cache-v3'; // Increment the version to force an update
const urlsToCache = [
    '/style.css',
    '/lib/wgo/wgo.player.css',
    '/lib/wgo/wgo.min.js',
    '/lib/wgo/wgo.player.min.js',
    '/lib/jszip.min.js'
    // We intentionally leave index.html out of this precache list
    // and use a network-first strategy for it instead
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Opened cache and caching static assets');
                return cache.addAll(urlsToCache);
            })
    );
});

self.addEventListener('activate', event => {
  clients.claim();
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});

// Network-first strategy for navigation requests to prevent stale HTML
self.addEventListener('fetch', event => {
    // For navigation requests (page loads), try network first
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .catch(() => caches.match(event.request))
        );
        return;
    }
    
    // For other requests (assets), use cache-first strategy
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});
