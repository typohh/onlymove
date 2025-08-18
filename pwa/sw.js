const CACHE_NAME = 'yose-tsumego-cache-v2'; // Increment the version to force an update
const urlsToCache = [
    '/',
    '/index.html',
    '/style.css',
    '/lib/wgo/wgo.player.css',
    '/lib/wgo/wgo.min.js',
    '/lib/wgo/wgo.player.min.js',
    '/lib/jszip.min.js'
    // We intentionally leave the hashed app.js file out of this precache list
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

self.addEventListener('fetch', event => {
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
