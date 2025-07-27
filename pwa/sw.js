const CACHE_NAME = 'yose-tsumego-cache-v1';
const urlsToCache = [
    '/',
    '/index.html',
    '/appf7ad958e.js',
    '/wgo.js/wgo/wgo.player.css',
    '/wgo.js/wgo/wgo.js',
    '/wgo.js/wgo/sgfparser.js',
    '/wgo.js/wgo/kifu.js'
];

self.addEventListener('install', event => {
  self.skipWaiting();
  event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
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
